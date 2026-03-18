import torch

from .outputs import SamplerOutput

from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler as BaseSampler


class Sampler(BaseSampler):

  def compute_logprobs(
      self,
      logits: torch.Tensor,
      dtype=torch.float32,
  ) -> torch.Tensor:
    return logits.log_softmax(dim=-1, dtype=dtype)

  def forward(
      self,
      logits: torch.Tensor,
      sampling_metadata: SamplingMetadata,
  ) -> SamplerOutput:
    # Returns processed logprobs and logprobs calculated before temperature no
    # matter what

    num_logprobs = sampling_metadata.max_num_logprobs

    # Use float32 for the logits.
    logits = logits.to(torch.float32)
    # Apply allowed token ids, bad words, logits processors, and penalties.
    # (vllm >= 0.11.2 consolidated these into apply_logits_processors)
    logits = self.apply_logits_processors(logits, sampling_metadata, False)

    # 💎 MCMC new logic:
    # compute logprobs (don't apply temperature, let users apply temperature)
    if num_logprobs is not None:
      power_logprobs = self.compute_logprobs(logits)

    # Sample the next token.
    sampled, processed_logprobs = self.sample(logits, sampling_metadata)

    # Convert sampled token ids to int64 (long) type to ensure compatibility
    # with subsequent operations that may use these values as indices.
    # This conversion is necessary because FlashInfer sampling operations
    # return int32 (while PyTorch argmax and topk return int64).
    sampled = sampled.long()

    # Gather the logprobs of the topk and sampled token (if requested).
    # Get logprobs and rank tensors (if requested)
    logprobs_tensors = None if num_logprobs is None else self.gather_logprobs(
        processed_logprobs,
        num_logprobs,
        token_ids=sampled,
    )
    power_logprobs_tensors = None if num_logprobs is None else self.gather_logprobs(
        power_logprobs,
        num_logprobs,
        token_ids=sampled,
    )

    # Use int32 to reduce the tensor size.
    sampled = sampled.to(torch.int32)

    # These are GPU tensors.
    sampler_output = SamplerOutput(
        # The sampled tokens are expanded to 2D tensor with shape
        # [num_requests, 1], where each row represents one generated
        # token per request.
        sampled_token_ids=sampled.unsqueeze(-1),
        logprobs_tensors=logprobs_tensors,
        power_logprobs_tensors=power_logprobs_tensors,
    )
    return sampler_output
