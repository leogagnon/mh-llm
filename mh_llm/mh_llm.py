import copy
from functools import partial
import math
import random

import torch
from transformers import AutoTokenizer, PreTrainedTokenizer
from vllm.logprobs import Logprob
import tqdm

from .vllm import LLM, SamplingParams

_ACCEPTANCE_THRESHOLD = 1 - 1e-5


class _dummytqdm:

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    pass

  def update(self, n=1):
    pass

  def set_description(self, desc=None):
    pass


def _copy_sampling_params(
    sampling_params: SamplingParams,
    **kwargs,
) -> SamplingParams:
  """Create a deep copy of the given SamplingParams.

  Args:
      sampling_params (SamplingParams): The sampling parameters to copy.
      **kwargs: keywords to override in the copied SamplingParams.

  Returns:
      SamplingParams: A deep copy of the given sampling parameters.
  """
  new_params = sampling_params.clone()
  for key, value in kwargs.items():
    setattr(new_params, key, value)

  return new_params


class MHLLM:

  def __init__(
      self,
      model: str,
      trust_remote_code: bool = False,
      **vllm_kwargs,
  ):
    self.model_name = model
    self.vllm_kwargs = vllm_kwargs

    # Initialize vLLM
    self.llm = LLM(model, trust_remote_code=trust_remote_code, **vllm_kwargs)
    # Initialize tokenizer
    self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        model,
        trust_remote_code=trust_remote_code,
    )

  def generate(self, *args, **kwargs):
    return self.llm.generate(*args, **kwargs)

  def collective_rpc(self, *args, **kwargs):
    return self.llm.collective_rpc(*args, **kwargs)

  def _extract_logprobs(
      self,
      logprobs: list[dict[int, Logprob]],
  ) -> list[float]:
    """Extract logprob values from vLLM logprobs output.

    Args:
        logprobs (list[dict[int, Logprob]]): The logprobs output from vLLM.

    Returns:
        list[float]: The extracted logprob values.
    """
    return [list(logprob.values())[0].logprob for logprob in logprobs]

  def _generate_intermediate_prompt(
      self,
      prompt_ids: list[int],
      proposed_tokens: list[int],
  ) -> dict:
    """Build a vLLM token-IDs prompt by concatenating prompt and proposed tokens.

    Working in token space avoids re-tokenizing text strings, which would
    cause a double-BOS when the chat template already includes the BOS token.

    Args:
        prompt_ids (list[int]): Pre-tokenized prompt token IDs.
        proposed_tokens (list[int]): Already-generated tokens to prepend as context.

    Returns:
        dict: A vLLM TokensPrompt dict {"prompt_token_ids": [...]}.
    """
    return {"prompt_token_ids": prompt_ids + list(proposed_tokens)}

  def _mh_sample(
      self,
      prompt: str,
      sampling_params: SamplingParams,
      block_size: int = 192,
      max_new_tokens: int = 3_072,
      num_mcmc_steps: int = 10,
  ) -> str:
    """Perform Metropolis-Hastings sampling for a single prompt.

    Args:
        prompt (str): The input prompt to generate text for.
        sampling_params (SamplingParams): The sampling parameters to use for
          generation.
        block_size (int, optional): The size of each generation block.
            Defaults to 192.
        max_new_tokens (int, optional): The maximum number of new tokens to
            generate. Defaults to 3,072.
        num_mcmc_steps (int, optional): The number of MCMC steps to perform.
            Defaults to 10.
    Returns:
        str: The generated text output.
    """
    # Pre-tokenize once with add_special_tokens=False to avoid double-BOS:
    # the chat template already includes the BOS token in the string, so
    # re-tokenizing with the default add_special_tokens=True would prepend
    # an extra one on every vLLM call.
    prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

    # override certain sampling parameter values
    sampling_params = copy.deepcopy(sampling_params)
    sampling_params.n = 1  # always generate 1 sample
    sampling_params.logprobs = 1  # always return 1 logprob

    output_tokens = []
    logprob: list[float] = []
    power_logprob: list[float] = []

    block_steps = (int(math.ceil(max_new_tokens / block_size) + 1))
    for k in range(0, block_steps):
      sampling_params.max_tokens = block_size
      _prompt = self._generate_intermediate_prompt(prompt_ids, output_tokens)
      output = self.llm.generate(
          _prompt,
          sampling_params=sampling_params,
          use_tqdm=False,
      )
      out = output[0].outputs[0]

      output_tokens.extend(out.token_ids)
      logprob.extend(self._extract_logprobs(out.logprobs))
      power_logprob.extend(self._extract_logprobs(out.power_logprobs))

      for _ in range(num_mcmc_steps):
        # Propose new tokens starting from a randomly sampled position
        idx = random.randint(0, len(output_tokens) - 2)
        mcmc_prompt = self._generate_intermediate_prompt(prompt_ids, output_tokens[:idx])
        sampling_params.max_tokens = len(output_tokens) - idx
        mcmc_output = self.llm.generate(
            mcmc_prompt,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        out = mcmc_output[0].outputs[0]

        # get proposed logprobs as a list and as tensor
        proposed_logprobs = self._extract_logprobs(out.logprobs)
        proposed_power_logprobs = self._extract_logprobs(out.power_logprobs)

        # get current logprobs as tensor
        curr_logprobs = logprob[idx:]
        curr_power_logprobs = power_logprob[idx:]

        # Calculate acceptance probabilities
        log_prob_ratio = (sum(proposed_logprobs) - sum(curr_logprobs))
        if sampling_params.alpha == float('inf'):
          log_power_prob_ratio = 0.0  # TODO: check this
        else:
          log_power_prob_ratio = (
              sum(proposed_power_logprobs) -
              sum(curr_power_logprobs)) * sampling_params.alpha

        A = math.exp(log_power_prob_ratio + log_prob_ratio)

        if A >= _ACCEPTANCE_THRESHOLD or random.random() < A:
          # Accept the proposal
          output_tokens = output_tokens[:idx] + out.token_ids
          logprob = logprob[:idx] + proposed_logprobs
          power_logprob = power_logprob[:idx] + proposed_power_logprobs

      if output_tokens[-1] == self.tokenizer.eos_token_id:
        break

    return self.tokenizer.decode(
        output_tokens,
        skip_special_tokens=True,
    )

  def mh_sample(
      self,
      prompts: str | list[str],
      sampling_params: SamplingParams,
      block_size: int = 192,
      max_new_tokens: int = 3_072,
      num_mcmc_steps: int = 10,
      use_tqdm: bool = True,
  ) -> list[str]:
    """Generate text using MH sampling in batch mode.

    Sampled idx are not i.i.d.

    Args:
        prompts (str | list[str]): The input prompts to generate text for.
        sampling_params (SamplingParams): The sampling parameters to use for
          generation.
        block_size (int, optional): The size of each generation block.
            Defaults to 192.
        max_new_tokens (int, optional): The maximum number of new tokens to
            generate. Defaults to 3,072.
        num_mcmc_steps (int, optional): The number of MCMC steps to perform.
            Defaults to 10.
        use_tqdm (bool, optional): Whether to use tqdm for progress tracking.
            Defaults to True.

    Returns:
        str | list[str]: The generated text outputs.
    """
    if isinstance(prompts, str):
      return self._mh_sample(
          prompts,
          sampling_params,
          block_size,
          max_new_tokens,
          num_mcmc_steps,
      )
    # override certain sampling parameter values
    sampling_params = _copy_sampling_params(
        sampling_params,
        n=1,
        logprobs=1,
    )

    # Pre-tokenize all prompts once with add_special_tokens=False
    prompts_ids = [
        self.tokenizer.encode(p, add_special_tokens=False) for p in prompts
    ]

    remaining_prompt_idx = list(range(len(prompts)))

    output_tokens = [[] for _ in prompts]
    logprob: list[list[float]] = [[] for _ in prompts]
    power_logprob: list[list[float]] = [[] for _ in prompts]

    block_steps = (int(math.ceil(max_new_tokens / block_size) + 1))
    _tqdm = tqdm.tqdm if use_tqdm else _dummytqdm
    with _tqdm(
        total=len(remaining_prompt_idx),
        colour='#CBA6F8',
        desc='MH Sampling',
    ) as pbar:
      for k in range(0, block_steps):
        _prompts = [
            self._generate_intermediate_prompt(
                prompts_ids[i],
                output_tokens[i],
            ) for i in remaining_prompt_idx
        ]
        sp_list = [
            _copy_sampling_params(
                sampling_params,
                max_tokens=block_size,
            ) for i in remaining_prompt_idx
        ]

        outputs = self.llm.generate(
            _prompts,
            sampling_params=sp_list,
            use_tqdm=False,
        )
        for batch_idx, i in enumerate(remaining_prompt_idx):
          out = outputs[batch_idx].outputs[0]

          output_tokens[i].extend(out.token_ids)
          logprob[i].extend(self._extract_logprobs(out.logprobs))
          power_logprob[i].extend(self._extract_logprobs(out.power_logprobs))

        # Perform MCMC steps for all remaining prompts
        for _ in range(num_mcmc_steps):
          idx_list = [
              random.randint(
                  0,
                  len(output_tokens[i]) - 1,
              ) if len(output_tokens[i]) > 1 else 0
              for i in remaining_prompt_idx
          ]
          mcmc_prompts = [
              self._generate_intermediate_prompt(
                  prompts_ids[i],
                  output_tokens[i][:idx_list[batch_idx]],
              ) for batch_idx, i in enumerate(remaining_prompt_idx)
          ]
          sp_list = [
              _copy_sampling_params(
                  sampling_params,
                  max_tokens=len(output_tokens[i]) - idx_list[batch_idx],
              ) for batch_idx, i in enumerate(remaining_prompt_idx)
          ]
          mcmc_outputs = self.llm.generate(
              mcmc_prompts,
              sampling_params=sp_list,
              use_tqdm=False,
          )
          outs = [out.outputs[0] for out in mcmc_outputs]

          proposed_logprobs_list = [
              self._extract_logprobs(outs[batch_idx].logprobs)
              for batch_idx in range(len(remaining_prompt_idx))
          ]
          proposed_power_logprobs_list = [
              self._extract_logprobs(outs[batch_idx].power_logprobs)
              for batch_idx in range(len(remaining_prompt_idx))
          ]

          curr_logprobs_list = [
              logprob[i][idx_list[batch_idx]:]
              for batch_idx, i in enumerate(remaining_prompt_idx)
          ]
          curr_power_logprobs_list = [
              power_logprob[i][idx_list[batch_idx]:]
              for batch_idx, i in enumerate(remaining_prompt_idx)
          ]

          # Calculate acceptance probabilities and decide to accept/reject
          log_prob_ratios = [(sum(proposed_logprobs_list[batch_idx]) -
                              sum(curr_logprobs_list[batch_idx]))
                             for batch_idx in range(len(remaining_prompt_idx))]
          if sampling_params.alpha == float('inf'):
            # TODO: check this
            log_power_prob_ratios = [0.0 for _ in remaining_prompt_idx]
          else:
            log_power_prob_ratios = [
                (sum(proposed_power_logprobs_list[batch_idx]) -
                 sum(curr_power_logprobs_list[batch_idx])) *
                sampling_params.alpha
                for batch_idx in range(len(remaining_prompt_idx))
            ]

          for batch_idx, i in enumerate(remaining_prompt_idx):
            try:
              A = math.exp(log_power_prob_ratios[batch_idx] +
                           log_prob_ratios[batch_idx])
            except OverflowError:
              A = 1.0

            if A >= _ACCEPTANCE_THRESHOLD or random.random() < A:
              # Accept the proposal
              output_tokens[i] = (output_tokens[i][:idx_list[batch_idx]] +
                                  outs[batch_idx].token_ids)
              logprob[i] = (logprob[i][:idx_list[batch_idx]] +
                            proposed_logprobs_list[batch_idx])
              power_logprob[i] = (power_logprob[i][:idx_list[batch_idx]] +
                                  proposed_power_logprobs_list[batch_idx])

        # Remove prompts that have generated EOS token
        for i in remaining_prompt_idx:
          if output_tokens[i][-1] == self.tokenizer.eos_token_id:
            remaining_prompt_idx.remove(i)
            pbar.update(1)

        if not remaining_prompt_idx:
          break

      # add remaining results
      for i in remaining_prompt_idx:
        pbar.update(1)

    return [
        self.tokenizer.decode(
            output_tokens[i],
            skip_special_tokens=True,
        ) for i in range(len(prompts))
    ]
