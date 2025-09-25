from __future__ import annotations

from vllm.v1.core.sched.scheduler import Scheduler as VLLMScheduler
from vllm.v1.core.sched.scheduler import SchedulerOutput, ModelRunnerOutput, EngineCoreOutputs, Request, RequestStatus, SpecDecodingStats, defaultdict, Optional
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.core.sched.request_queue import create_request_queue
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput


class Scheduler(VLLMScheduler):
    """Omni AR Scheduler: 在构造 EngineCoreOutput 时写入 output_type。

    本类继承 vLLM V1 的 Scheduler，仅覆写在组装 EngineCoreOutput 处的行为。
    其他调度逻辑沿用上游实现。
    """

    # We override only the section where EngineCoreOutput is appended.
    # To avoid forking the whole file, we alias update_from_outputs and wrap
    # the precise instantiation site if needed. For simplicity and
    # maintainability, we duplicate the minimal block using super() flow.

    def update_from_outputs(self, model_runner_output: ModelRunnerOutput) -> EngineCoreOutputs:  # type: ignore[override]
        outputs = super().update_from_outputs(model_runner_output)
        # All EngineCoreOutput objects in outputs.outputs lack explicit omni
        # output_type. We now populate them if not set.
        output_type = getattr(self.vllm_config.model_config, "engine_output_type", None)
        if output_type:
            for eco in outputs.outputs:
                if getattr(eco, "output_type", None) is None:
                    eco.output_type = output_type
        return outputs


