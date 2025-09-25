from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from vllm.utils import FlexibleArgumentParser  # type: ignore
from vllm.engine.arg_utils import EngineArgs as _BaseEngineArgs  # type: ignore
from vllm.engine.arg_utils import AsyncEngineArgs as _BaseAsyncEngineArgs  # type: ignore

@dataclass
class EngineArgs(_BaseEngineArgs):
    """Omni 扩展的 EngineArgs。

    - 新增 `engine_output_type` 字段，用于通过 CLI/代码从 LLM 层设置输出类型，
      在 create_engine_config 阶段写入到 `config.model_config.engine_output_type`。
    """

    engine_output_type: Optional[str] = None

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        # 先添加 vLLM 标准参数
        parser = _BaseEngineArgs.add_cli_args(parser)
        # 再添加 omni 扩展参数
        parser.add_argument(
            "--engine-output-type",
            type=str,
            default=EngineArgs.engine_output_type,
            help=(
                "Declare EngineCoreOutput.output_type (e.g., 'text', 'image', "
                "'text+image', 'latent'). This will be written into "
                "model_config.engine_output_type for schedulers to use."
            ),
        )
        return parser

    def create_engine_config(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        config = super().create_engine_config(*args, **kwargs)
        # 将 CLI/代码传入的输出类型写入 model_config，供调度器读取
        # We simply add the engine_output_type attribute to the config, to avoid the complexity of changing the VllmConfig class
        # and all the files that use it like the vllm.engine.arg_utils.EngineArgs.create_engine_config().
        setattr(config.model_config, "engine_output_type", self.engine_output_type)
        return config


#Also add the engine_output_type to the AsyncEngineArgs
@dataclass
class AsyncEngineArgs(_BaseAsyncEngineArgs):
    """Omni 扩展的 AsyncEngineArgs。

    - 新增 `engine_output_type` 字段，用于通过 CLI/代码从 LLM 层设置输出类型，
      在 create_engine_config 阶段写入到 `config.model_config.engine_output_type`。
    """

    engine_output_type: Optional[str] = None
    
    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser, async_args_only: bool = False) -> FlexibleArgumentParser:
        parser = _BaseAsyncEngineArgs.add_cli_args(parser, async_args_only)
        parser.add_argument("--engine-output-type", type=str, default=AsyncEngineArgs.engine_output_type, help="Declare EngineCoreOutput.output_type (e.g., 'text', 'image', 'text+image', 'latent'). This will be written into model_config.engine_output_type for schedulers to use.")
        return parser
    
    def create_engine_config(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        config = super().create_engine_config(*args, **kwargs)
        # We simply add the engine_output_type attribute to the config, to avoid the complexity of changing the VllmConfig class
        # and all the files that use it like the vllm.engine.arg_utils.AsyncEngineArgs.create_engine_config().
        setattr(config.model_config, "engine_output_type", self.engine_output_type)
        return config