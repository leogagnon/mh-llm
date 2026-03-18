from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import torch
from typing_extensions import TypeAlias

from .outputs import (
    SamplerOutput,
    ModelRunnerOutput,
    EMPTY_MODEL_RUNNER_OUTPUT,
)

import vllm.envs as envs
from vllm.distributed.kv_transfer import has_kv_transfer_group
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
try:
    from vllm.v1.attention.backends.flash_attn import AttentionMetadata
except ImportError:
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata as AttentionMetadata  # vllm >= 0.11.2
# yapf: enable
from vllm.v1.outputs import (
    LogprobsLists,
    LogprobsTensors,
    AsyncModelRunnerOutput,
)
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.utils import is_residual_scattered_for_sp

if TYPE_CHECKING:
  from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)

AttnMetadataDict: TypeAlias = dict[str, AttentionMetadata]
# list when ubatching is enabled
PerLayerAttnMetadata: TypeAlias = Union[
    list[AttnMetadataDict],
    AttnMetadataDict,
]
################

from vllm.v1.worker.gpu_model_runner import (
    GPUModelRunner as BaseGPUModelRunner,
    AsyncGPUModelRunnerOutput,
)
from .sampler import Sampler


class GPUModelRunner(BaseGPUModelRunner):

  def __init__(self, vllm_config, device):
    super().__init__(vllm_config, device)
    self.sampler = Sampler(logprobs_mode=self.model_config.logprobs_mode)

  @torch.inference_mode()
  def execute_model(
      self,
      scheduler_output: "SchedulerOutput",
      intermediate_tensors: Optional[IntermediateTensors] = None,
  ) -> Union[ModelRunnerOutput, AsyncModelRunnerOutput, IntermediateTensors]:
    with record_function_or_nullcontext("Preprocess"):
      with self.synchronize_input_prep():
        # Update persistent batch states.
        self._update_states(scheduler_output)

        if not scheduler_output.total_num_scheduled_tokens:
          if not has_kv_transfer_group():
            # Return empty ModelRunnerOutput if no work to do.
            return EMPTY_MODEL_RUNNER_OUTPUT
          return self.kv_connector_no_forward(
              scheduler_output,
              self.vllm_config,
          )
        if self.cache_config.kv_sharing_fast_prefill:
          assert not self.input_batch.num_prompt_logprobs, (
              "--kv-sharing-fast-prefill produces incorrect "
              "logprobs for prompt tokens, tokens, please disable "
              "it when the requests need prompt logprobs")

        # Prepare the decoder inputs.
        num_reqs = self.input_batch.num_reqs
        req_ids = self.input_batch.req_ids
        tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
        num_scheduled_tokens_np = np.array(tokens, dtype=np.int32)
        max_num_scheduled_tokens = int(num_scheduled_tokens_np.max())

        (logits_indices, spec_decode_metadata, ubatch_slices,
         num_tokens_across_dp) = self._prepare_inputs(
            scheduler_output, num_scheduled_tokens_np, max_num_scheduled_tokens)

        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
        attn_metadata, spec_decode_common_attn_metadata = (
            self._build_attention_metadata(
                total_num_scheduled_tokens=total_num_scheduled_tokens,
                max_num_scheduled_tokens=max_num_scheduled_tokens,
                num_reqs=num_reqs,
                ubatch_slices=ubatch_slices,
                logits_indices=logits_indices,
                use_spec_decode=use_spec_decode,
                scheduled_encoder_inputs=scheduler_output.scheduled_encoder_inputs,
                cascade_attn_prefix_lens=None,
            )
        )

        dp_rank = self.parallel_config.data_parallel_rank
        if ubatch_slices:
          assert num_tokens_across_dp is not None
          num_input_tokens = int(num_tokens_across_dp[dp_rank].item())
          self.pad_out_ubatch_slice(ubatch_slices, num_input_tokens)
        elif num_tokens_across_dp is not None:
          num_input_tokens = int(num_tokens_across_dp[dp_rank].item())
        else:
          num_input_tokens = self._get_num_input_tokens(total_num_scheduled_tokens)

        (input_ids, inputs_embeds, positions, intermediate_tensors,
         model_kwargs, _ec_connector_output) = self._preprocess(
            scheduler_output, num_input_tokens, intermediate_tensors)

      num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
      uniform_decode = (
          max_num_scheduled_tokens == self.uniform_decode_query_len) and (
          num_scheduled_tokens == num_reqs * max_num_scheduled_tokens)
      batch_descriptor = BatchDescriptor(
          num_tokens=num_input_tokens,
          uniform_decode=uniform_decode,
      )
      cudagraph_runtime_mode, batch_descriptor = \
          self.cudagraph_dispatcher.dispatch(batch_descriptor)

    # This is currently to get around the assert in the DPMetadata
    # where it wants `num_tokens_across_dp` to align with `num_tokens`
    if ubatch_slices is not None:
      num_input_tokens = ubatch_slices[0].num_tokens

    # Run the model.
    # Use persistent buffers for CUDA graphs.
    with (set_forward_context(
        attn_metadata,
        self.vllm_config,
        num_tokens=num_input_tokens,
        num_tokens_across_dp=num_tokens_across_dp,
        cudagraph_runtime_mode=cudagraph_runtime_mode,
        batch_descriptor=batch_descriptor,
        ubatch_slices=ubatch_slices,
    ), record_function_or_nullcontext("Forward"),
          self.maybe_get_kv_connector_output(scheduler_output) as
          kv_connector_output):
      model_output = self.model(
          input_ids=input_ids,
          positions=positions,
          intermediate_tensors=intermediate_tensors,
          inputs_embeds=inputs_embeds,
          **model_kwargs,
      )

    with record_function_or_nullcontext("Postprocess"):
      if self.use_aux_hidden_state_outputs:
        # True when EAGLE 3 is used.
        hidden_states, aux_hidden_states = model_output
      else:
        # Common case.
        hidden_states = model_output
        aux_hidden_states = None

      if not self.broadcast_pp_output:
        # Common case.
        if not get_pp_group().is_last_rank:
          # Return the intermediate tensors.
          assert isinstance(hidden_states, IntermediateTensors)
          hidden_states.kv_connector_output = kv_connector_output
          return hidden_states

        if self.is_pooling_model:
          # Return the pooling output.
          output = self._pool(hidden_states, num_scheduled_tokens,
                              num_scheduled_tokens_np)
          output.kv_connector_output = kv_connector_output
          return output

        sample_hidden_states = hidden_states[logits_indices]
        logits = self.model.compute_logits(sample_hidden_states)
      else:
        # Rare case.
        assert not self.is_pooling_model

        if not get_pp_group().is_last_rank:
          all_gather_tensors = {
              "residual":
                  not is_residual_scattered_for_sp(
                      self.vllm_config,
                      num_input_tokens,
                  )
          }
          get_pp_group().send_tensor_dict(
              hidden_states.tensors,
              all_gather_group=get_tp_group(),
              all_gather_tensors=all_gather_tensors,
          )
          logits = None
        else:
          sample_hidden_states = hidden_states[logits_indices]
          logits = self.model.compute_logits(sample_hidden_states)

        model_output_broadcast_data = {}
        if logits is not None:
          model_output_broadcast_data["logits"] = logits.contiguous()

        model_output_broadcast_data = get_pp_group().broadcast_tensor_dict(
            model_output_broadcast_data, src=len(get_pp_group().ranks) - 1)
        assert model_output_broadcast_data is not None
        logits = model_output_broadcast_data["logits"]

      # Structured output bitmasks (grammar_bitmask) moved to sample_tokens()
      # in vllm >= 0.11.2 and are no longer on SchedulerOutput directly.

    with record_function_or_nullcontext("Sample"):
      sampler_output: SamplerOutput = self._sample(logits, spec_decode_metadata)

    def propose_draft_token_ids(sampled_token_ids):
      assert spec_decode_common_attn_metadata is not None
      with record_function_or_nullcontext("Draft"):
        self._draft_token_ids = self.propose_draft_token_ids(
            scheduler_output,
            sampled_token_ids,
            self.input_batch.sampling_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
        )

    use_padded_batch_for_eagle = self.speculative_config and \
        self.speculative_config.use_eagle() and \
        not self.speculative_config.disable_padded_drafter_batch
    effective_drafter_max_model_len = self.max_model_len
    if effective_drafter_max_model_len is None:
      effective_drafter_max_model_len = self.model_config.max_model_len
    if (self.speculative_config and
        self.speculative_config.draft_model_config is not None and
        self.speculative_config.draft_model_config.max_model_len is not None):
      effective_drafter_max_model_len = (
          self.speculative_config.draft_model_config.max_model_len)
    input_fits_in_drafter = spec_decode_common_attn_metadata and (
        spec_decode_common_attn_metadata.seq_lens.max() +
        self.speculative_config.num_speculative_tokens
        <= effective_drafter_max_model_len)
    if use_padded_batch_for_eagle and input_fits_in_drafter:
      # EAGLE speculative decoding can use the GPU sampled tokens
      # as inputs, and does not need to wait for bookkeeping to finish.
      propose_draft_token_ids(sampler_output.sampled_token_ids)

    with record_function_or_nullcontext("Bookkeep"):
      (
          num_nans_in_logits,
          logprobs_lists,
          power_logprobs_lists,
          valid_sampled_token_ids,
          prompt_logprobs_dict,
          req_ids_output_copy,
          req_id_to_index_output_copy,
          invalid_req_indices,
      ) = self._bookkeeping_sync(
          scheduler_output,
          sampler_output,
          logits,
          hidden_states,
          num_scheduled_tokens,
          spec_decode_metadata,
      )

    if (self.speculative_config and not use_padded_batch_for_eagle and
        input_fits_in_drafter):
      # ngram and other speculative decoding methods use the sampled
      # tokens on the CPU, so they are run after bookkeeping.
      propose_draft_token_ids(valid_sampled_token_ids)

    with record_function_or_nullcontext("EPLB"):
      self.eplb_step()

    output = ModelRunnerOutput(
        req_ids=req_ids_output_copy,
        req_id_to_index=req_id_to_index_output_copy,
        sampled_token_ids=valid_sampled_token_ids,
        logprobs=logprobs_lists,
        power_logprobs=power_logprobs_lists,
        prompt_logprobs_dict=prompt_logprobs_dict,
        pooler_output=[],
        kv_connector_output=kv_connector_output,
        num_nans_in_logits=num_nans_in_logits,
    )

    if not self.use_async_scheduling:
      return output

    return AsyncGPUModelRunnerOutput(
        model_runner_output=output,
        sampled_token_ids=sampler_output.sampled_token_ids,
        invalid_req_indices=invalid_req_indices,
        async_output_copy_stream=self.async_output_copy_stream,
    )

  def _bookkeeping_sync(
      self,
      scheduler_output: "SchedulerOutput",
      sampler_output: SamplerOutput,
      logits: Optional[torch.Tensor],
      hidden_states: torch.Tensor,
      num_scheduled_tokens: int,
      spec_decode_metadata=None,  # added in vllm >= 0.11.2
  ) -> tuple[
      dict[str, int],
      Optional[LogprobsLists],
      Optional[LogprobsLists],
      list[list[int]],
      dict[str, Optional[LogprobsTensors]],
      list[str],
      dict[str, int],
      list[int],
  ]:
    num_nans_in_logits = {}
    if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
      num_nans_in_logits = self._get_nans_in_logits(logits)

    discard_sampled_tokens_req_indices = \
        self.discard_request_indices.np[:self.num_discarded_requests]
    for i in discard_sampled_tokens_req_indices:
      gen = self.input_batch.generators.get(int(i))
      if gen is not None:
        gen.set_offset(gen.get_offset() - 4)

    # Copy some objects so they don't get modified after returning.
    # This is important when using async scheduling.
    req_ids_output_copy = self.input_batch.req_ids.copy()
    req_id_to_index_output_copy = \
        self.input_batch.req_id_to_index.copy()

    # NOTE: GPU -> CPU Sync happens here.
    # Move as many CPU operations as possible before this sync point.
    logprobs_tensors = sampler_output.logprobs_tensors
    logprobs_lists = logprobs_tensors.tolists() \
        if logprobs_tensors is not None else None

    # ✨✨✨ This is new ✨✨✨
    # power logprobs list
    power_logprobs_tensors = sampler_output.power_logprobs_tensors
    power_logprobs_lists = power_logprobs_tensors.tolists() \
        if power_logprobs_tensors is not None else None
    # ✨✨✨ end of new code ✨✨✨

    # Compute prompt logprobs if needed.
    prompt_logprobs_dict = self._get_prompt_logprobs_dict(
        hidden_states[:num_scheduled_tokens],
        scheduler_output.num_scheduled_tokens,
    )

    num_sampled_tokens = sampler_output.sampled_token_ids.shape[0]
    sampled_token_ids = sampler_output.sampled_token_ids
    invalid_req_indices = []
    if not self.use_async_scheduling:
      # Get the valid generated tokens.
      max_gen_len = sampled_token_ids.shape[-1]
      if max_gen_len == 1:
        # No spec decode tokens.
        valid_sampled_token_ids = self._to_list(sampled_token_ids)
      else:
        # Includes spec decode tokens.
        valid_sampled_token_ids = self.rejection_sampler.parse_output(
            sampled_token_ids,
            self.input_batch.vocab_size,
        )
      # Mask out the sampled tokens that should not be sampled.
      for i in discard_sampled_tokens_req_indices:
        # vllm >= 0.11.2: _to_list returns list[np.ndarray], not list[list[int]]
        idx = int(i)
        entry = valid_sampled_token_ids[idx]
        if hasattr(entry, 'clear'):
          entry.clear()
        else:
          valid_sampled_token_ids[idx] = entry[:0]
    else:
      valid_sampled_token_ids = []
      invalid_req_indices = discard_sampled_tokens_req_indices.tolist()
      invalid_req_indices_set = set(invalid_req_indices)
      assert sampled_token_ids.shape[-1] == 1

      # Cache the sampled tokens on the GPU and avoid CPU sync.
      # These will be copied into input_ids in the next step
      # when preparing inputs.
      self.input_batch.prev_sampled_token_ids = sampled_token_ids
      self.input_batch.prev_sampled_token_ids_invalid_indices = \
          invalid_req_indices_set
      self.input_batch.prev_req_id_to_index = {
          req_id: i
          for i, req_id in enumerate(self.input_batch.req_ids)
          if i not in invalid_req_indices_set
      }

    # Cache the sampled tokens in the model runner, so that the scheduler
    # doesn't need to send them back.
    # NOTE(woosuk): As an exception, when using PP, the scheduler sends
    # the sampled tokens back, because there's no direct communication
    # between the first-stage worker and the last-stage worker.
    req_ids = self.input_batch.req_ids
    for req_idx in range(num_sampled_tokens):
      if self.use_async_scheduling:
        sampled_ids = [-1] if \
            req_idx not in invalid_req_indices_set else None
      else:
        sampled_ids = valid_sampled_token_ids[req_idx]
      if not sampled_ids:
        continue

      start_idx = self.input_batch.num_tokens_no_spec[req_idx]
      end_idx = start_idx + len(sampled_ids)
      assert end_idx <= self.max_model_len, (
          "Sampled token IDs exceed the max model length. "
          f"Total number of tokens: {end_idx} > max_model_len: "
          f"{self.max_model_len}")

      self.input_batch.token_ids_cpu[
          req_idx,
          start_idx:end_idx,
      ] = sampled_ids
      self.input_batch.is_token_ids[req_idx, start_idx:end_idx] = True
      self.input_batch.num_tokens_no_spec[req_idx] = end_idx
      self.input_batch.num_tokens[req_idx] = end_idx

      req_id = req_ids[req_idx]
      req_state = self.requests[req_id]
      req_state.output_token_ids.extend(sampled_ids)

    return (
        num_nans_in_logits,
        logprobs_lists,
        power_logprobs_lists,
        valid_sampled_token_ids,
        prompt_logprobs_dict,
        req_ids_output_copy,
        req_id_to_index_output_copy,
        invalid_req_indices,
    )
