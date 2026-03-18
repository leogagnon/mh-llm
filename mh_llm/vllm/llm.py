from typing import Any

from vllm.config import (
    CompilationConfig,
    StructuredOutputsConfig,
)
try:
    from vllm.config import ModelDType
except ImportError:
    from vllm.config.model import ModelDType  # vllm >= 0.11.2
try:
    from vllm.config import TokenizerMode
except ImportError:
    from vllm.engine.arg_utils import TokenizerMode  # vllm >= 0.11.2
from vllm.engine.arg_utils import (
    ConvertOption,
    HfOverrides,
    PoolerConfig,
    RunnerOption,
)
# yapf: enable
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.v1.sample.logits_processor import LogitsProcessor

from vllm.entrypoints.llm import LLM as BaseLLM, logger

from .outputs import EngineCoreOutputs
from .logprobs import LogprobsProcessor
from .output_processor import OutputProcessor
from .scheduler import Scheduler
from .worker import Worker

from .utils import patch as mh_patch


class LLM(BaseLLM):

  def __init__(
      self,
      model: str,
      *,
      runner: RunnerOption = "auto",
      convert: ConvertOption = "auto",
      tokenizer: str | None = None,
      tokenizer_mode: TokenizerMode = "auto",
      skip_tokenizer_init: bool = False,
      trust_remote_code: bool = False,
      allowed_local_media_path: str = "",
      allowed_media_domains: list[str] | None = None,
      tensor_parallel_size: int = 1,
      dtype: ModelDType = "auto",
      quantization: QuantizationMethods | None = None,
      revision: str | None = None,
      tokenizer_revision: str | None = None,
      seed: int | None = None,
      gpu_memory_utilization: float = 0.9,
      swap_space: float = 4,
      cpu_offload_gb: float = 0,
      enforce_eager: bool = False,
      disable_custom_all_reduce: bool = False,
      hf_token: bool | str | None = None,
      hf_overrides: HfOverrides | None = None,
      mm_processor_kwargs: dict[str, Any] | None = None,
      pooler_config: PoolerConfig | None = None,
      override_pooler_config: PoolerConfig | None = None,
      structured_outputs_config: dict[str, Any] | StructuredOutputsConfig |
      None = None,
      kv_cache_memory_bytes: int | None = None,
      compilation_config: int | dict[str, Any] | CompilationConfig |
      None = None,
      logits_processors: list[str | type[LogitsProcessor]] | None = None,
      **kwargs: Any,
  ) -> None:
    """LLM constructor."""
    kwargs[
        'worker_cls'] = f'{Worker.__module__}.{Worker.__name__}'  # use our own worker
    logger.info('Using worker_cls: %s', kwargs['worker_cls'])
    kwargs['logprobs_mode'] = 'processed_logprobs'  # force processed logprobs
    kwargs['scheduler_cls'] = Scheduler  # use our own scheduler

    with mh_patch([
        {
            'module': 'vllm.v1.engine.output_processor',
            'class': LogprobsProcessor,
        },
        {
            'module': 'vllm.v1.engine.core_client',
            'class': EngineCoreOutputs,
        },
        {
            'module': 'vllm.v1.engine.llm_engine',
            'class': OutputProcessor,
        },
    ]):
      super().__init__(
          model=model,
          runner=runner,
          convert=convert,
          tokenizer=tokenizer,
          tokenizer_mode=tokenizer_mode,
          skip_tokenizer_init=skip_tokenizer_init,
          trust_remote_code=trust_remote_code,
          allowed_local_media_path=allowed_local_media_path,
          allowed_media_domains=allowed_media_domains,
          tensor_parallel_size=tensor_parallel_size,
          dtype=dtype,
          quantization=quantization,
          revision=revision,
          tokenizer_revision=tokenizer_revision,
          seed=seed,
          gpu_memory_utilization=gpu_memory_utilization,
          swap_space=swap_space,
          cpu_offload_gb=cpu_offload_gb,
          enforce_eager=enforce_eager,
          disable_custom_all_reduce=disable_custom_all_reduce,
          hf_token=hf_token,
          hf_overrides=hf_overrides,
          mm_processor_kwargs=mm_processor_kwargs,
          pooler_config=pooler_config,
          override_pooler_config=override_pooler_config,
          structured_outputs_config=structured_outputs_config,
          kv_cache_memory_bytes=kv_cache_memory_bytes,
          compilation_config=compilation_config,
          logits_processors=logits_processors,
          **kwargs,
      )
