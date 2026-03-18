from vllm.sampling_params import RequestOutputKind
from vllm.transformers_utils.detokenizer_utils import AnyTokenizer
from vllm.v1.engine import EngineCoreRequest, FinishReason
from vllm.v1.engine.output_processor import (
    OutputProcessor as BaseOutputProcessor,
    RequestOutputCollector,
    RequestState as BaseRequestState,
)
from vllm.v1.engine.parallel_sampling import ParentRequest

from .logprobs import LogprobsProcessor
from .outputs import CompletionOutput
from .utils import patch as mh_patch


class RequestState(BaseRequestState):

  def _new_completion_output(
      self,
      token_ids: list[int],
      finish_reason: FinishReason | None,
      stop_reason: int | str | None,
  ) -> CompletionOutput:

    assert self.detokenizer is not None
    assert self.logprobs_processor is not None
    finished = finish_reason is not None
    delta = self.output_kind == RequestOutputKind.DELTA

    # Prepare text and token_ids, based on delta mode
    text = self.detokenizer.get_next_output_text(finished, delta)
    if not delta:
      token_ids = self.detokenizer.output_token_ids

    # Prepare logprobs, based on delta mode
    logprobs = self.logprobs_processor.logprobs
    power_logprobs = self.logprobs_processor.power_logprobs
    if delta and logprobs:
      logprobs = logprobs[-len(token_ids):]
      power_logprobs = power_logprobs[-len(token_ids):]

    return CompletionOutput(
        index=self.request_index,
        text=text,
        token_ids=token_ids,
        logprobs=logprobs,
        power_logprobs=power_logprobs,
        cumulative_logprob=self.logprobs_processor.cumulative_logprob,
        finish_reason=str(finish_reason) if finished else None,
        stop_reason=stop_reason if finished else None,
    )

  @classmethod
  def from_new_request(
      cls,
      tokenizer: AnyTokenizer,
      request: EngineCoreRequest,
      prompt: str | None,
      parent_req: ParentRequest | None,
      request_index: int,
      queue: RequestOutputCollector | None,
      log_stats: bool,
      stream_interval: int = 1,  # added in vllm >= 0.11.2
  ) -> "RequestState":
    with mh_patch([{
        'module': 'vllm.v1.engine.output_processor',
        'class': LogprobsProcessor,
    }]):
      return super().from_new_request(
          tokenizer,
          request,
          prompt,
          parent_req,
          request_index,
          queue,
          log_stats,
          stream_interval=stream_interval,
      )


class OutputProcessor(BaseOutputProcessor):

  def add_request(
      self,
      request: EngineCoreRequest,
      prompt: str | None,
      parent_req: ParentRequest | None = None,
      request_index: int = 0,
      queue: RequestOutputCollector | None = None,
  ) -> None:
    with mh_patch([{
        'module': 'vllm.v1.engine.output_processor',
        'class': RequestState,
    }]):
      super().add_request(
          request,
          prompt,
          parent_req,
          request_index,
          queue,
      )
