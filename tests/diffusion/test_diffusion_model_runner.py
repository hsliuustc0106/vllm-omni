# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm_omni.diffusion.worker.diffusion_model_runner as model_runner_module
import vllm_omni.diffusion.worker.diffusion_worker as worker_module
from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.host_weight.ownership import (
    LegacyReason,
    PreparedSessionReady,
    UseLegacy,
    WeightConsumerOwner,
    WeightConsumerOwnerPhase,
)
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch, split_diffusion_output_by_request

pytestmark = [pytest.mark.diffusion]


@contextmanager
def _noop_forward_context(*args, **kwargs):
    del args, kwargs
    yield


class _DummyPipeline:
    supports_request_batch = True

    def __init__(self, output):
        self._output = output
        self.forward_calls = 0
        self.last_req = None

    def forward(self, req):
        self.last_req = req
        self.forward_calls += 1
        return [self._output]


class _SingleRequestBatchPipeline:
    supports_request_batch = False

    def __init__(self):
        self.last_req = None

    def forward(self, req):
        self.last_req = req
        return [DiffusionOutput(output=req.prompts[0])]


class _SingleRequestDiffusionOutputPipeline:
    supports_request_batch = False

    def __init__(self):
        self.last_req = None

    def forward(self, req):
        self.last_req = req
        return DiffusionOutput(output=req.prompts[0])


class _ChunkStepPipeline:
    device = torch.device("cpu")
    supports_step_execution = True

    def __init__(self, outputs):
        self._outputs = outputs
        self.prepare_calls = 0
        self.decode_calls = 0

    def prepare_encode(self, state):
        self.prepare_calls += 1
        state.prompt_embeds = torch.zeros(1, 1, 1)
        state.latents = torch.zeros(1, 1)
        state.timesteps = torch.tensor([1.0, 0.0])
        state.step_index = 0
        state.step_in_chunk = 0
        state.chunk_num_steps = 2
        state.total_chunks = len(self._outputs)
        return state

    def denoise_step(self, input_batch, states):
        del states
        return torch.ones_like(input_batch.latents)

    def step_scheduler(self, state, noise_pred):
        state.latents = noise_pred
        state.step_index += 1
        state.step_in_chunk += 1

    def post_decode(self, state):
        output = self._outputs[self.decode_calls]
        self.decode_calls += 1
        state.chunk_index += 1
        state.step_index = 0
        state.step_in_chunk = 0
        if not state.request_denoise_completed:
            state.latents = torch.zeros(1, 1)
        return output


class _FinalOnlyStepPipeline:
    device = torch.device("cpu")
    supports_step_execution = True

    def prepare_encode(self, state):
        state.prompt_embeds = torch.zeros(1, 1, 1)
        state.latents = torch.zeros(1, 1)
        state.timesteps = torch.tensor([1.0])
        state.step_index = 0
        return state

    def denoise_step(self, input_batch, states):
        del states
        return torch.ones_like(input_batch.latents)

    def step_scheduler(self, state, noise_pred):
        state.latents = noise_pred
        state.step_index += 1

    def post_decode(self, state):
        return DiffusionOutput(output=state.latents.clone())


class _CompileTrackingModel:
    def __init__(self):
        self.compile_calls = []

    def compile(self, *args, **kwargs):
        self.compile_calls.append((args, kwargs))
        return self


def _make_request():
    sampling_params = SimpleNamespace(
        generator=None,
        seed=None,
        generator_device=None,
        num_inference_steps=4,
    )
    return SimpleNamespace(
        request_id="req-test",
        prompt="a prompt",
        sampling_params=sampling_params,
        kv_sender_info=None,
    )


def _make_request_with_params(req_id: str, sampling_params):
    return SimpleNamespace(
        request_id=req_id,
        prompt=f"prompt-{req_id}",
        prompts=[f"prompt-{req_id}"],
        sampling_params=sampling_params,
    )


def _fake_platform_for_peak_memory():
    return SimpleNamespace(
        reset_peak_memory_stats=lambda: None,
        max_memory_reserved=lambda: 0,
        max_memory_allocated=lambda: 0,
    )


def test_request_scoped_cache_dit_lifecycle_is_pipeline_opt_in():
    events = []
    plain_pipeline = object()
    request_scoped_pipeline = SimpleNamespace(
        adopt_cache_dit_backend=lambda backend: events.append(backend),
        is_cache_dit_enabled=lambda: True,
    )
    backend = object()

    assert not model_runner_module.adopt_request_scoped_cache_dit(plain_pipeline, backend)
    assert model_runner_module.adopt_request_scoped_cache_dit(request_scoped_pipeline, backend)
    assert model_runner_module.is_request_scoped_cache_dit_enabled(request_scoped_pipeline)
    assert events == [backend]


def _make_runner(cache_backend, cache_backend_name: str, enable_cache_dit_summary: bool = True):
    runner = object.__new__(DiffusionModelRunner)
    runner.vllm_config = object()
    runner.device = torch.device("cpu")
    runner.pipeline = _DummyPipeline(output=DiffusionOutput(output="ok"))
    runner.cache_backend = cache_backend
    runner.offload_backend = None
    runner.state_cache = {}
    runner.prompt_embed_cache = None
    runner.od_config = SimpleNamespace(
        cache_backend=cache_backend_name,
        enable_cache_dit_summary=enable_cache_dit_summary,
        parallel_config=SimpleNamespace(use_hsdp=False),
        streaming_output=False,
    )
    runner.kv_transfer_manager = SimpleNamespace(
        receive_kv_cache=lambda req, target_device=None: None,
        receive_multi_kv_cache=lambda req, cfg_kv_collect_func=None, target_device=None: None,
        receive_multi_kv_cache_distributed=lambda *a, **k: None,
    )
    runner._kv_prefetch_enabled = False
    return runner


def _make_compile_runner(
    model=None,
    *,
    compile_granularity: str = "regional",
    compile_dynamic: bool = True,
    use_hsdp: bool = False,
):
    runner = object.__new__(DiffusionModelRunner)
    runner.pipeline = SimpleNamespace(transformer=model or SimpleNamespace())
    runner.od_config = SimpleNamespace(
        diffusion_compile_granularity=compile_granularity,
        diffusion_compile_dynamic=compile_dynamic,
        parallel_config=SimpleNamespace(use_hsdp=use_hsdp),
    )
    return runner


@pytest.mark.core_model
@pytest.mark.cpu
def test_update_states_carries_prepared_layout() -> None:
    runner = _make_runner(cache_backend=None, cache_backend_name=None)
    request = _make_request()
    prepared_layout = object()
    request.prepared_layout = prepared_layout
    scheduler_output = SimpleNamespace(
        finished_req_ids=set(),
        scheduled_new_reqs=[SimpleNamespace(request_id=request.request_id, req=request)],
        scheduled_cached_reqs=SimpleNamespace(request_ids=[]),
    )

    states, new_request_ids = DiffusionModelRunner._update_states(runner, scheduler_output)

    assert new_request_ids == [request.request_id]
    assert states[0].prepared_layout is prepared_layout


@pytest.mark.core_model
@pytest.mark.cpu
@pytest.mark.parametrize("use_hsdp", [False, True])
def test_compile_transformer_regionally_compiles_blocks(monkeypatch, use_hsdp):
    runner = _make_compile_runner(use_hsdp=use_hsdp)
    compile_calls = []

    def _regionally_compile(model, *args, **kwargs):
        compile_calls.append((model, args, kwargs))
        return model

    monkeypatch.setattr(model_runner_module, "regionally_compile", _regionally_compile)

    DiffusionModelRunner._compile_transformer(runner, "transformer")

    assert compile_calls == [
        (
            runner.pipeline.transformer,
            (),
            {"dynamic": True},
        )
    ]


@pytest.mark.core_model
@pytest.mark.cpu
def test_compile_transformer_uses_regional_dynamic_false_config(monkeypatch):
    model = _CompileTrackingModel()
    runner = _make_compile_runner(model, compile_dynamic=False)
    compiled_model = object()
    regional_calls = []

    def _regionally_compile(target, **kwargs):
        regional_calls.append((target, kwargs))
        return compiled_model

    monkeypatch.setattr(model_runner_module, "regionally_compile", _regionally_compile)

    DiffusionModelRunner._compile_transformer(runner, "transformer")

    assert regional_calls == [(model, {"dynamic": False})]
    assert model.compile_calls == []
    assert runner.pipeline.transformer is compiled_model


@pytest.mark.core_model
@pytest.mark.cpu
def test_compile_transformer_uses_full_granularity(monkeypatch):
    model = _CompileTrackingModel()
    runner = _make_compile_runner(model, compile_granularity="full", compile_dynamic=False)
    regional_calls = []

    monkeypatch.setattr(
        model_runner_module,
        "regionally_compile",
        lambda *args, **kwargs: regional_calls.append((args, kwargs)),
    )

    DiffusionModelRunner._compile_transformer(runner, "transformer")

    assert model.compile_calls == [((), {"dynamic": False})]
    assert regional_calls == []
    assert runner.pipeline.transformer is model


@pytest.mark.core_model
@pytest.mark.cpu
def test_compile_transformer_falls_back_after_synchronous_setup_failure(monkeypatch, caplog):
    model = _CompileTrackingModel()
    runner = _make_compile_runner(model)

    def _regionally_compile(*args, **kwargs):
        raise RuntimeError("compile setup failed")

    monkeypatch.setattr(model_runner_module, "regionally_compile", _regionally_compile)

    DiffusionModelRunner._compile_transformer(runner, "transformer")

    assert runner.pipeline.transformer is model
    assert "failed before activation" in caplog.text
    assert "lazy compilation errors" in caplog.text


@pytest.mark.core_model
@pytest.mark.cpu
def test_execute_stepwise_streaming_returns_chunks_at_boundaries(monkeypatch):
    """Step streaming returns empty step results until a chunk decode boundary."""
    chunks = [
        DiffusionOutput(output="chunk-0", finished=False, chunk_index=0, total_chunks=2),
        DiffusionOutput(output="chunk-1", finished=True, chunk_index=1, total_chunks=2),
    ]
    runner = _make_runner(cache_backend=None, cache_backend_name=None)
    runner.pipeline = _ChunkStepPipeline(chunks)
    runner.od_config.streaming_output = True
    runner.od_config.step_execution = True
    req = _make_request()
    req.request_id = "req"

    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)
    monkeypatch.setattr(model_runner_module.current_omni_platform, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(model_runner_module.current_omni_platform, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(model_runner_module.current_omni_platform, "max_memory_allocated", lambda: 0)
    scheduler_output = SimpleNamespace(
        finished_req_ids=set(),
        scheduled_new_reqs=[SimpleNamespace(request_id="req", req=req, diffusion_kv_metadata=None)],
        scheduled_cached_reqs=SimpleNamespace(request_ids=[]),
    )

    first = DiffusionModelRunner.execute_stepwise(runner, scheduler_output)
    assert first.get_request_output("req").result is None

    scheduler_output = SimpleNamespace(
        finished_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(request_ids=["req"]),
    )
    second = DiffusionModelRunner.execute_stepwise(runner, scheduler_output)
    assert second.get_request_output("req").result == chunks[0]
    assert second.get_request_output("req").finished is False

    DiffusionModelRunner.execute_stepwise(runner, scheduler_output)
    fourth = DiffusionModelRunner.execute_stepwise(runner, scheduler_output)
    assert fourth.get_request_output("req").result == chunks[1]
    assert fourth.get_request_output("req").finished is True


@pytest.mark.core_model
@pytest.mark.cpu
def test_execute_stepwise_streaming_decodes_final_only_pipeline(monkeypatch):
    """Step streaming still returns a final output when no chunk boundaries exist."""
    runner = _make_runner(cache_backend=None, cache_backend_name=None)
    runner.pipeline = _FinalOnlyStepPipeline()
    runner.od_config.streaming_output = True
    runner.od_config.step_execution = True
    req = _make_request()
    req.request_id = "req"
    req.sampling_params.num_inference_steps = 1

    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)
    monkeypatch.setattr(model_runner_module.current_omni_platform, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(model_runner_module.current_omni_platform, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(model_runner_module.current_omni_platform, "max_memory_allocated", lambda: 0)
    scheduler_output = SimpleNamespace(
        finished_req_ids=set(),
        scheduled_new_reqs=[SimpleNamespace(request_id="req", req=req, diffusion_kv_metadata=None)],
        scheduled_cached_reqs=SimpleNamespace(request_ids=[]),
    )

    output = DiffusionModelRunner.execute_stepwise(runner, scheduler_output).get_request_output("req")

    assert output.finished is True
    assert output.result is not None
    assert torch.equal(output.result.output, torch.ones(1, 1))


@pytest.mark.core_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_execute_model_skips_cache_summary_without_active_cache_backend(monkeypatch):
    """Guard cache diagnostics with runtime backend state to avoid stale-config crashes."""
    runner = _make_runner(cache_backend=None, cache_backend_name="cache_dit")
    req = _make_request()

    cache_summary_calls = []

    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)
    monkeypatch.setattr(model_runner_module.current_omni_platform, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(model_runner_module.current_omni_platform, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(model_runner_module.current_omni_platform, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(
        model_runner_module,
        "cache_summary",
        lambda pipeline, details: cache_summary_calls.append((pipeline, details)),
    )

    output = DiffusionModelRunner.execute_model(runner, req)

    assert output.output == "ok"
    assert cache_summary_calls == []


@pytest.mark.core_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_execute_model_emits_cache_summary_with_active_cache_dit_backend(monkeypatch):
    class _EnabledCacheBackend:
        def __init__(self):
            self.refresh_calls = []

        def is_enabled(self):
            return True

        def refresh(self, pipeline, num_inference_steps, verbose=True):
            self.refresh_calls.append((pipeline, num_inference_steps, verbose))

    cache_backend = _EnabledCacheBackend()
    runner = _make_runner(cache_backend=cache_backend, cache_backend_name="cache_dit")
    req = _make_request()

    cache_summary_calls = []

    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)
    monkeypatch.setattr(model_runner_module.current_omni_platform, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(model_runner_module.current_omni_platform, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(model_runner_module.current_omni_platform, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(
        model_runner_module,
        "cache_summary",
        lambda pipeline, details: cache_summary_calls.append((pipeline, details)),
    )

    output = DiffusionModelRunner.execute_model(runner, req)

    assert output.output == "ok"
    assert cache_summary_calls == [(runner.pipeline, True)]
    assert cache_backend.refresh_calls == [(runner.pipeline, 4, True)]


@pytest.mark.core_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("cache_dit_enabled", [False, True])
def test_execute_model_cache_summary_follows_pipeline_owned_cache_dit_state(monkeypatch, cache_dit_enabled):
    runner = _make_runner(cache_backend=None, cache_backend_name="cache_dit")
    runner.pipeline.adopt_cache_dit_backend = lambda backend: None
    runner.pipeline.is_cache_dit_enabled = lambda: cache_dit_enabled
    req = _make_request()

    cache_summary_calls = []

    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)
    monkeypatch.setattr(model_runner_module.current_omni_platform, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(model_runner_module.current_omni_platform, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(model_runner_module.current_omni_platform, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(
        model_runner_module,
        "cache_summary",
        lambda pipeline, details: cache_summary_calls.append((pipeline, details)),
    )

    output = DiffusionModelRunner.execute_model(runner, req)

    assert output.output == "ok"
    assert cache_summary_calls == ([(runner.pipeline, True)] if cache_dit_enabled else [])


@pytest.mark.core_model
@pytest.mark.cpu
def test_execute_model_passes_single_request_batch_to_non_admission_pipeline(monkeypatch):
    runner = _make_runner(cache_backend=None, cache_backend_name="none")
    runner.pipeline = _SingleRequestBatchPipeline()
    req = _make_request()

    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)

    output = DiffusionModelRunner.execute_model(runner, req)

    assert output.output == "a prompt"
    assert isinstance(runner.pipeline.last_req, DiffusionRequestBatch)
    assert runner.pipeline.last_req.num_reqs == 1


@pytest.mark.core_model
@pytest.mark.cpu
def test_execute_model_accepts_bare_diffusion_output_from_single_request_pipeline(monkeypatch):
    runner = _make_runner(cache_backend=None, cache_backend_name="none")
    runner.pipeline = _SingleRequestDiffusionOutputPipeline()
    req = _make_request()

    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)

    output = DiffusionModelRunner.execute_model(runner, req)

    assert output.output == "a prompt"
    assert isinstance(runner.pipeline.last_req, DiffusionRequestBatch)
    assert runner.pipeline.last_req.num_reqs == 1


@pytest.mark.core_model
@pytest.mark.cpu
def test_profile_run_executes_forward_without_scheduler_kv_validation(monkeypatch):
    runner = _make_runner(cache_backend=None, cache_backend_name="none")
    request = _make_request()
    runner._validate_diffusion_kv_metadata = Mock(side_effect=AssertionError("profile must bypass admission"))
    record_names = []
    original_execute = runner._execute_request_list

    def execute(*args, **kwargs):
        record_names.append(kwargs["record_name"])
        return original_execute(*args, **kwargs)

    runner._execute_request_list = execute
    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)
    reset_peak_memory_stats = Mock()
    monkeypatch.setattr(
        model_runner_module.current_omni_platform,
        "reset_peak_memory_stats",
        reset_peak_memory_stats,
    )
    monkeypatch.setattr(model_runner_module.current_omni_platform, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(model_runner_module.current_omni_platform, "max_memory_allocated", lambda: 0)
    synchronize = Mock()
    monkeypatch.setattr(model_runner_module.current_omni_platform, "synchronize", synchronize)

    runner.profile_run([request])

    assert runner.pipeline.forward_calls == 1
    assert record_names == ["pipeline_memory_profile"]
    runner._validate_diffusion_kv_metadata.assert_not_called()
    reset_peak_memory_stats.assert_not_called()
    synchronize.assert_called_once_with()


@pytest.mark.core_model
@pytest.mark.cpu
def test_profile_run_executes_maximum_step_batch_without_resetting_peak(monkeypatch):
    runner = _make_runner(cache_backend=None, cache_backend_name=None)
    runner.pipeline = _FinalOnlyStepPipeline()
    runner.od_config.step_execution = True
    requests = [_make_request(), _make_request()]
    requests[0].request_id = "profile-0"
    requests[1].request_id = "profile-1"
    for request in requests:
        request.sampling_params.num_inference_steps = 1

    observed_batch_rows = []
    original_denoise_step = runner.pipeline.denoise_step

    def denoise_step(input_batch, states):
        observed_batch_rows.append((len(states), int(input_batch.latents.shape[0])))
        return original_denoise_step(input_batch, states)

    runner.pipeline.denoise_step = denoise_step
    runner._validate_diffusion_kv_metadata = Mock(side_effect=AssertionError("profile must bypass admission"))
    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)
    reset_peak_memory_stats = Mock()
    monkeypatch.setattr(
        model_runner_module.current_omni_platform,
        "reset_peak_memory_stats",
        reset_peak_memory_stats,
    )
    monkeypatch.setattr(model_runner_module.current_omni_platform, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(model_runner_module.current_omni_platform, "max_memory_allocated", lambda: 0)
    synchronize = Mock()
    monkeypatch.setattr(model_runner_module.current_omni_platform, "synchronize", synchronize)

    runner.profile_run(requests)

    assert observed_batch_rows == [(2, 2)]
    assert runner.state_cache == {}
    assert runner.input_batch is None
    runner._validate_diffusion_kv_metadata.assert_not_called()
    reset_peak_memory_stats.assert_not_called()
    synchronize.assert_called_once_with()


class _BatchPipeline:
    """Pipeline returning a configurable list of outputs from forward()."""

    supports_request_batch = True

    def __init__(self, outputs):
        self._outputs = outputs
        self.last_batch = None

    def forward(self, batch):
        self.last_batch = batch
        return list(self._outputs)


class _SingleRequestPipeline:
    def forward(self, batch):
        del batch
        return [DiffusionOutput(output="single")]


class _BatchSingleOutputPipeline:
    supports_request_batch = True

    def forward(self, batch):
        return DiffusionOutput(output=batch.prompts[0])


def _make_batch_runner(pipeline):
    runner = object.__new__(DiffusionModelRunner)
    runner.vllm_config = object()
    runner.device = torch.device("cpu")
    runner.pipeline = pipeline
    runner.cache_backend = None
    runner.offload_backend = None
    runner.od_config = SimpleNamespace(
        cache_backend="none",
        enable_cache_dit_summary=False,
        parallel_config=SimpleNamespace(use_hsdp=False),
    )
    runner.kv_transfer_manager = SimpleNamespace(
        receive_multi_kv_cache_distributed=lambda req, cfg_kv_collect_func=None, target_device=None: None,
    )
    return runner


def _make_scheduler_output(num_reqs: int):
    reqs = [_make_request() for _ in range(num_reqs)]
    for i, req in enumerate(reqs):
        req.request_id = f"req-{i}"
    return SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(request_id=req.request_id, req=req, diffusion_kv_metadata=None) for req in reqs
        ],
    )


@pytest.mark.core_model
@pytest.mark.cpu
def test_execute_model_batch_rejects_output_count_mismatch(monkeypatch):
    """A pipeline returning the wrong number of outputs must fail loudly,
    not silently drop requests or IndexError on the per-request mapping."""
    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)
    monkeypatch.setattr(model_runner_module, "current_omni_platform", _fake_platform_for_peak_memory())
    # forward returns 1 output for 2 scheduled requests
    runner = _make_batch_runner(_BatchPipeline(outputs=[DiffusionOutput(output="only-one")]))
    sched = _make_scheduler_output(num_reqs=2)

    with pytest.raises(RuntimeError, match="returned 1 outputs for 2 requests"):
        DiffusionModelRunner.execute_model_batch(runner, sched, runner.od_config)


@pytest.mark.core_model
@pytest.mark.cpu
def test_execute_model_batch_routes_one_output_per_request(monkeypatch):
    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)
    monkeypatch.setattr(model_runner_module, "current_omni_platform", _fake_platform_for_peak_memory())
    outs = [DiffusionOutput(output="a"), DiffusionOutput(output="b")]
    runner = _make_batch_runner(_BatchPipeline(outputs=outs))
    sched = _make_scheduler_output(num_reqs=2)

    result = DiffusionModelRunner.execute_model_batch(runner, sched, runner.od_config)

    assert len(result.runner_outputs) == 2
    assert result.runner_outputs[0].request_id == "req-0"
    assert result.runner_outputs[0].result.output == "a"
    assert result.runner_outputs[1].request_id == "req-1"
    assert result.runner_outputs[1].result.output == "b"


@pytest.mark.core_model
@pytest.mark.cpu
def test_execute_model_batch_preserves_per_request_sampling_and_seeds_generators(monkeypatch):
    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)
    monkeypatch.setattr(model_runner_module, "current_omni_platform", _fake_platform_for_peak_memory())
    outputs = [DiffusionOutput(output="a"), DiffusionOutput(output="b")]
    pipeline = _BatchPipeline(outputs=outputs)
    runner = _make_batch_runner(pipeline)
    sched = _make_scheduler_output(num_reqs=2)
    sched.scheduled_new_reqs[0].req.sampling_params.seed = 111
    sched.scheduled_new_reqs[1].req.sampling_params.seed = 222

    DiffusionModelRunner.execute_model_batch(runner, sched, runner.od_config)

    assert pipeline.last_batch is not None
    assert [sp.seed for sp in pipeline.last_batch.sampling_params_list] == [111, 222]
    assert [sp.generator.initial_seed() for sp in pipeline.last_batch.sampling_params_list] == [111, 222]
    with pytest.raises(AssertionError, match="multiple requests"):
        _ = pipeline.last_batch.sampling_params


@pytest.mark.core_model
@pytest.mark.cpu
def test_execute_model_batch_rejects_pipeline_without_request_batch_support(monkeypatch):
    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)
    monkeypatch.setattr(model_runner_module, "current_omni_platform", _fake_platform_for_peak_memory())
    runner = _make_batch_runner(_SingleRequestPipeline())
    sched = _make_scheduler_output(num_reqs=2)

    with pytest.raises(RuntimeError, match="does not support request-batch forward"):
        DiffusionModelRunner.execute_model_batch(runner, sched, runner.od_config)


@pytest.mark.core_model
@pytest.mark.cpu
def test_execute_model_batch_rejects_single_diffusion_output(monkeypatch):
    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)
    monkeypatch.setattr(model_runner_module, "current_omni_platform", _fake_platform_for_peak_memory())
    runner = _make_batch_runner(_BatchSingleOutputPipeline())
    sched = _make_scheduler_output(num_reqs=1)

    with pytest.raises(RuntimeError, match="request-batch forward must return list\\[DiffusionOutput\\]"):
        DiffusionModelRunner.execute_model_batch(runner, sched, runner.od_config)


@pytest.mark.core_model
@pytest.mark.cpu
def test_execute_model_batch_uses_runner_output_helper(monkeypatch):
    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)
    monkeypatch.setattr(model_runner_module, "current_omni_platform", _fake_platform_for_peak_memory())
    outs = [DiffusionOutput(output="a"), DiffusionOutput(output="b")]
    runner = _make_batch_runner(_BatchPipeline(outputs=outs))
    sched = _make_scheduler_output(num_reqs=2)
    helper_calls = []
    original_from_outputs = DiffusionModelRunner._runner_output_from_outputs

    def _recording_from_outputs(self, reqs, outputs):
        helper_calls.append(([req.request_id for req in reqs], [output.output for output in outputs]))
        return original_from_outputs(self, reqs, outputs)

    monkeypatch.setattr(DiffusionModelRunner, "_runner_output_from_outputs", _recording_from_outputs)

    result = DiffusionModelRunner.execute_model_batch(runner, sched, runner.od_config)

    assert [runner_output.result.output for runner_output in result.runner_outputs] == ["a", "b"]
    assert helper_calls == [(["req-0", "req-1"], ["a", "b"])]


@pytest.mark.core_model
@pytest.mark.cpu
def test_split_diffusion_output_by_request_slices_single_and_multi_request_outputs():
    reqs = [_make_request(), _make_request()]
    reqs[0].request_id = "req-0"
    reqs[1].request_id = "req-1"
    batch = DiffusionRequestBatch(requests=reqs)
    result = DiffusionOutput(output=["img-0a", "img-0b", "img-1a", "img-1b"], stage_durations={"decode": 1.0})

    outputs = split_diffusion_output_by_request(result, batch, num_outputs_per_prompt=2)

    assert [output.output for output in outputs] == [["img-0a", "img-0b"], ["img-1a", "img-1b"]]
    assert [output.stage_durations for output in outputs] == [{"decode": 1.0}, {"decode": 1.0}]

    single = split_diffusion_output_by_request(
        result, DiffusionRequestBatch(requests=reqs[:1]), num_outputs_per_prompt=2
    )

    assert single[0].output == ["img-0a", "img-0b"]


@pytest.mark.core_model
@pytest.mark.cpu
def test_split_diffusion_output_by_request_slices_tuple_outputs():
    reqs = [_make_request(), _make_request()]
    batch = DiffusionRequestBatch(requests=reqs)
    result = DiffusionOutput(
        output=(
            ["video-0", "video-1"],
            torch.tensor([10, 20]),
        )
    )

    outputs = split_diffusion_output_by_request(result, batch, num_outputs_per_prompt=1)

    assert outputs[0].output[0] == ["video-0"]
    assert torch.equal(outputs[0].output[1], torch.tensor([10]))
    assert outputs[1].output[0] == ["video-1"]
    assert torch.equal(outputs[1].output[1], torch.tensor([20]))


@pytest.mark.core_model
@pytest.mark.cpu
def test_execute_model_runs_forward_after_kv_receive(monkeypatch):
    """execute_model runs the pipeline forward after the KV receive step."""
    runner = _make_runner(cache_backend=None, cache_backend_name=None)
    runner.kv_transfer_manager.receive_multi_kv_cache_distributed = lambda *a, **k: True
    req = _make_request()

    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)
    monkeypatch.setattr(model_runner_module.current_omni_platform, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(model_runner_module.current_omni_platform, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(model_runner_module.current_omni_platform, "max_memory_allocated", lambda: 0)

    output = DiffusionModelRunner.execute_model(runner, req)

    assert output.output == "ok"
    assert runner.pipeline.forward_calls == 1


class _RecordingPreparedWeightSession:
    def __init__(self, pipeline, *, rollback_failures: int = 0):
        self.pipeline = pipeline
        self.rollback_failures = rollback_failures
        self.rollback_calls = 0
        self._runtime = None
        self._artifact = None
        self._binding = None

    def rollback(self):
        self.rollback_calls += 1
        if self.rollback_failures:
            self.rollback_failures -= 1
            raise RuntimeError("prepared rollback still busy")


def _publish_ready_session(
    owner: WeightConsumerOwner,
    prepared: _RecordingPreparedWeightSession,
) -> None:
    runtime = object()
    artifact = object()
    binding = object()
    prepared._runtime = runtime
    prepared._artifact = artifact
    prepared._binding = binding
    cleanup = owner.begin_preparation(runtime)  # type: ignore[arg-type]
    cleanup.adopt_artifact(artifact)  # type: ignore[arg-type]
    cleanup.retain_binding(binding)  # type: ignore[arg-type]
    cleanup.transfer_to_prepared_session(prepared)  # type: ignore[arg-type]
    owner.publish_preparation_result(PreparedSessionReady(prepared))  # type: ignore[arg-type]


class _RecordingWeightConsumer:
    def __init__(
        self,
        *,
        expected_prepared: _RecordingPreparedWeightSession,
        events: list[str],
        disable_failures: int = 0,
    ) -> None:
        self.expected_prepared = expected_prepared
        self.events = events
        self.disable_failures = disable_failures
        self.prepared = None
        self.enable_calls = 0
        self.disable_calls = 0
        self.backend = SimpleNamespace(kind="hwr_backend")

    def adopt_prepared_session(self, prepared) -> None:
        assert prepared is self.expected_prepared
        assert self.prepared is None
        self.prepared = prepared
        self.events.append("session_adopted")

    def enable_transactionally(self) -> None:
        assert self.prepared is self.expected_prepared
        self.enable_calls += 1
        self.events.append("consumer_enabled")

    def disable(self) -> None:
        self.disable_calls += 1
        self.events.append("consumer_disable_attempted")
        if self.disable_failures:
            self.disable_failures -= 1
            raise RuntimeError("consumer cleanup still busy")
        prepared = self.prepared
        if prepared is not None:
            prepared.rollback()
            self.prepared = None
        self.events.append("consumer_disabled")

    def host_weight_diagnostics(self) -> dict[str, object]:
        return {
            "pinned_slot_budget_bytes": 8192,
            "idle_state": {
                "outstanding_units": 0,
                "bindings": 0,
                "events": 0,
            },
        }


class _RecordingWeightConsumerFactory:
    def __init__(
        self,
        *,
        prepared: _RecordingPreparedWeightSession,
        events: list[str],
        consumer_disable_failures: int,
    ) -> None:
        self.prepared = prepared
        self.events = events
        self.consumer_disable_failures = consumer_disable_failures
        self.create_calls = 0
        self.consumer: _RecordingWeightConsumer | None = None

    def create_into(self, *, owner: WeightConsumerOwner, pipeline) -> None:
        self.create_calls += 1
        self.events.append("consumer_created")
        assert owner.prepared_session is self.prepared
        assert pipeline is self.prepared.pipeline
        consumer = _RecordingWeightConsumer(
            expected_prepared=self.prepared,
            events=self.events,
            disable_failures=self.consumer_disable_failures,
        )
        self.consumer = consumer
        owner.publish_consumer(consumer)


class _OwnerPublishingLoader:
    def __init__(
        self,
        *,
        owner: WeightConsumerOwner,
        pipeline,
        prepared: _RecordingPreparedWeightSession,
        events: list[str],
        behavior: str,
        host_weight_plan,
        evidence_error: BaseException | None,
    ) -> None:
        self.owner = owner
        self.pipeline = pipeline
        self.prepared = prepared
        self.events = events
        self.behavior = behavior
        self.host_weight_plan = host_weight_plan
        self.evidence_error = evidence_error
        self.host_weight_runtime_fell_back = behavior == "fallback"
        self._evidence = (
            {
                "outcome": "fallback",
                "events": ["fallback"],
                "code": "optional_artifact_unavailable",
            }
            if self.host_weight_runtime_fell_back
            else {
                "outcome": "ready",
                "events": ["cache_hit", "ready"],
                "artifact_key": "a" * 64,
            }
        )

    def load_model(self, **kwargs):
        del kwargs
        self.events.append("loader_load")
        if self.behavior == "ready":
            _publish_ready_session(self.owner, self.prepared)
            self.events.append("prepared_published")
        elif self.behavior == "fallback":
            self.owner.publish_preparation_result(
                UseLegacy(
                    LegacyReason.OPTIONAL_ARTIFACT_UNAVAILABLE,
                    "single-rank optional miss",
                )
            )
            self.owner.close()
            self.events.append("fallback_owner_closed")
        else:  # pragma: no cover - harness contract
            raise AssertionError(f"unknown loader behavior {self.behavior!r}")
        return self.pipeline

    def take_host_weight_plan(self):
        plan = self.host_weight_plan
        self.host_weight_plan = None
        return plan

    def take_host_weight_runtime_evidence(self):
        if self.evidence_error is not None:
            raise self.evidence_error
        evidence = self._evidence
        self._evidence = None
        return evidence


class _RecordingLegacyBackend:
    def __init__(self) -> None:
        self.enable_calls = 0
        self.disable_calls = 0
        self.pipeline = None

    def enable(self, pipeline) -> None:
        self.enable_calls += 1
        self.pipeline = pipeline

    def disable(self) -> None:
        self.disable_calls += 1
        self.pipeline = None


class _NormalMemoryProfiler:
    consumed_memory = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        return False


def _make_host_weight_load_runner():
    runner = object.__new__(DiffusionModelRunner)
    runner.vllm_config = object()
    runner.device = torch.device("cpu")
    runner.pipeline = None
    runner.cache_backend = None
    runner.offload_backend = None
    runner._weight_consumer_owner = None
    runner.prompt_embed_cache = None
    runner.model_memory_usage = 0
    runner.host_weight_runtime_evidence = {"outcome": "unavailable"}
    runner.host_weight_runtime_fell_back = False
    runner.od_config = SimpleNamespace(
        enable_cpu_offload=True,
        enable_layerwise_offload=False,
        enable_distributed_layerwise_offload=False,
        host_weight_runtime_mode="read_only",
        host_weight_runtime_required=False,
        streaming_output=False,
        step_execution=False,
        model_class_name="TestPipeline",
        enforce_eager=True,
        cache_backend=None,
        cache_config={},
        enable_prompt_embed_cache=False,
        prompt_embed_cache_size=1,
    )
    return runner


def _install_host_weight_load_doubles(
    monkeypatch,
    runner,
    *,
    behavior: str = "ready",
    profiler_cls=_NormalMemoryProfiler,
    prepared_rollback_failures: int = 0,
    consumer_disable_failures: int = 0,
    evidence_error: BaseException | None = None,
):
    events: list[str] = []
    pipeline = SimpleNamespace(transformer=torch.nn.Identity())
    prepared = _RecordingPreparedWeightSession(
        pipeline,
        rollback_failures=prepared_rollback_failures,
    )
    consumer_factory = _RecordingWeightConsumerFactory(
        prepared=prepared,
        events=events,
        consumer_disable_failures=consumer_disable_failures,
    )
    legacy_plan = object() if behavior == "fallback" else None
    legacy_backend = _RecordingLegacyBackend()
    state = SimpleNamespace(
        events=events,
        owner=None,
        owner_seen_by_factory=None,
        loader=None,
        prepared=prepared,
        consumer_factory=consumer_factory,
        pipeline=pipeline,
        legacy_plan=legacy_plan,
        legacy_backend=legacy_backend,
    )

    def construct_owner():
        owner = WeightConsumerOwner()
        state.owner = owner
        events.append("owner_constructed")
        return owner

    def construct_consumer_factory(od_config, device):
        del od_config, device
        assert runner._weight_consumer_owner is state.owner
        state.owner_seen_by_factory = runner._weight_consumer_owner
        events.append("factory_constructed")
        return consumer_factory

    def construct_loader(
        load_config,
        od_config=None,
        *,
        weight_consumer_owner=None,
        weight_consumer_factory=None,
    ):
        del load_config, od_config
        assert runner._weight_consumer_owner is state.owner
        assert weight_consumer_owner is state.owner
        assert weight_consumer_factory is consumer_factory
        events.append("loader_constructed")
        loader = _OwnerPublishingLoader(
            owner=weight_consumer_owner,
            pipeline=pipeline,
            prepared=prepared,
            events=events,
            behavior=behavior,
            host_weight_plan=legacy_plan,
            evidence_error=evidence_error,
        )
        state.loader = loader
        return loader

    def get_legacy_backend(od_config, *, device, host_weight_plan):
        del od_config, device
        events.append("legacy_backend_selected")
        assert behavior == "fallback"
        assert host_weight_plan is legacy_plan
        return legacy_backend

    monkeypatch.setattr(model_runner_module, "LoadConfig", lambda: object())
    monkeypatch.setattr(model_runner_module, "WeightConsumerOwner", construct_owner)
    monkeypatch.setattr(
        model_runner_module,
        "BuiltinOffloadWeightConsumerFactory",
        construct_consumer_factory,
    )
    monkeypatch.setattr(model_runner_module, "DiffusersPipelineLoader", construct_loader)
    monkeypatch.setattr(model_runner_module, "DeviceMemoryProfiler", profiler_cls)
    monkeypatch.setattr(model_runner_module, "get_offload_backend", get_legacy_backend)
    monkeypatch.setattr(model_runner_module, "get_cache_backend", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        model_runner_module,
        "resolve_prompt_embed_cache_config",
        lambda **_kwargs: (False, 1),
    )
    monkeypatch.setattr(
        model_runner_module.current_omni_platform,
        "init_diffusion_model_runner_runtime",
        lambda **_kwargs: None,
    )
    return state


@pytest.mark.core_model
@pytest.mark.cpu
def test_hwr_owner_is_installed_before_factory_and_loader(monkeypatch):
    runner = _make_host_weight_load_runner()
    state = _install_host_weight_load_doubles(monkeypatch, runner)

    DiffusionModelRunner.load_model(runner)

    assert state.events[:3] == [
        "owner_constructed",
        "factory_constructed",
        "loader_constructed",
    ]
    assert state.owner_seen_by_factory is state.owner
    assert state.loader.owner is state.owner
    assert runner._weight_consumer_owner is state.owner
    runner.shutdown()


@pytest.mark.core_model
@pytest.mark.cpu
def test_hwr_ready_consumer_enables_and_exports_diagnostics(monkeypatch):
    runner = _make_host_weight_load_runner()
    state = _install_host_weight_load_doubles(monkeypatch, runner)

    DiffusionModelRunner.load_model(runner)

    owner = state.owner
    consumer = state.consumer_factory.consumer
    assert owner.phase is WeightConsumerOwnerPhase.CONSUMER
    assert state.consumer_factory.create_calls == 1
    assert consumer is not None
    assert consumer.enable_calls == 1
    assert consumer.prepared is state.prepared
    assert runner.pipeline is state.pipeline
    assert runner.offload_backend is consumer.backend
    assert state.legacy_backend.enable_calls == 0
    assert runner.get_host_weight_runtime_evidence() == {
        "outcome": "ready",
        "events": ["cache_hit", "ready"],
        "artifact_key": "a" * 64,
        "pinned_slot_budget_bytes": 8192,
        "idle_state": {
            "outstanding_units": 0,
            "bindings": 0,
            "events": 0,
        },
        "ordinary_loader_fallback": False,
    }

    runner.shutdown()

    assert consumer.disable_calls == 1
    assert state.prepared.rollback_calls == 1
    assert owner.phase is WeightConsumerOwnerPhase.CLOSED
    assert runner._weight_consumer_owner is None
    assert runner.offload_backend is None


@pytest.mark.core_model
@pytest.mark.cpu
def test_optional_hwr_fallback_uses_independent_legacy_backend(monkeypatch):
    runner = _make_host_weight_load_runner()
    state = _install_host_weight_load_doubles(
        monkeypatch,
        runner,
        behavior="fallback",
    )

    DiffusionModelRunner.load_model(runner)

    assert state.owner.phase is WeightConsumerOwnerPhase.CLOSED
    assert runner._weight_consumer_owner is None
    assert state.consumer_factory.create_calls == 0
    assert state.legacy_backend.enable_calls == 1
    assert state.legacy_backend.pipeline is state.pipeline
    assert runner.offload_backend is state.legacy_backend
    assert runner.host_weight_runtime_fell_back is True
    assert runner.get_host_weight_runtime_evidence() == {
        "outcome": "fallback",
        "events": ["fallback"],
        "code": "optional_artifact_unavailable",
        "ordinary_loader_fallback": True,
    }

    runner.shutdown()

    assert state.legacy_backend.disable_calls == 1
    assert runner.offload_backend is None


@pytest.mark.core_model
@pytest.mark.cpu
def test_hwr_ready_owner_retains_failed_preflight_cleanup_for_shutdown_retry(
    monkeypatch,
):
    primary = ValueError("primary evidence extraction failure")
    runner = _make_host_weight_load_runner()
    state = _install_host_weight_load_doubles(
        monkeypatch,
        runner,
        prepared_rollback_failures=1,
        evidence_error=primary,
    )

    with pytest.raises(ValueError, match="primary evidence extraction failure") as exc_info:
        DiffusionModelRunner.load_model(runner)

    assert exc_info.value is primary
    assert any("Closing the HWR owner" in note for note in primary.__notes__)
    assert state.prepared.rollback_calls == 1
    assert runner._weight_consumer_owner is state.owner
    assert state.owner.phase is WeightConsumerOwnerPhase.PREPARATION_RESULT
    assert state.consumer_factory.create_calls == 0

    runner.shutdown()

    assert state.prepared.rollback_calls == 2
    assert state.owner.phase is WeightConsumerOwnerPhase.CLOSED
    assert runner._weight_consumer_owner is None


@pytest.mark.core_model
@pytest.mark.cpu
def test_hwr_active_consumer_cleanup_failure_is_retained_for_shutdown_retry(
    monkeypatch,
):
    primary = RuntimeError("primary cache enable failure")
    runner = _make_host_weight_load_runner()
    runner.od_config.cache_backend = "test"
    state = _install_host_weight_load_doubles(
        monkeypatch,
        runner,
        consumer_disable_failures=1,
    )

    class FailingCacheBackend:
        def enable(self, pipeline):
            assert pipeline is state.pipeline
            raise primary

    monkeypatch.setattr(
        model_runner_module,
        "get_cache_backend",
        lambda *_args, **_kwargs: FailingCacheBackend(),
    )

    with pytest.raises(RuntimeError, match="primary cache enable failure") as exc_info:
        DiffusionModelRunner.load_model(runner)

    assert exc_info.value is primary
    assert any("Closing the HWR owner" in note for note in primary.__notes__)
    consumer = state.consumer_factory.consumer
    assert consumer is not None
    assert consumer.disable_calls == 1
    assert state.prepared.rollback_calls == 0
    assert state.owner.phase is WeightConsumerOwnerPhase.CONSUMER
    assert runner._weight_consumer_owner is state.owner
    assert runner.offload_backend is consumer.backend

    with pytest.raises(RuntimeError, match="still owns an offload lifecycle"):
        DiffusionModelRunner.load_model(runner)
    assert runner._weight_consumer_owner is state.owner
    assert runner.offload_backend is consumer.backend
    assert consumer.disable_calls == 1

    runner.shutdown()

    assert consumer.disable_calls == 2
    assert state.prepared.rollback_calls == 1
    assert state.owner.phase is WeightConsumerOwnerPhase.CLOSED
    assert runner._weight_consumer_owner is None
    assert runner.offload_backend is None


@pytest.mark.core_model
@pytest.mark.cpu
def test_worker_load_shuts_down_runner_after_platform_sync_failure(monkeypatch):
    primary = RuntimeError("primary platform synchronization failure")

    class _Runner:
        def __init__(self):
            self.pipeline = object()
            self.load_calls = 0
            self.shutdown_calls = 0

        def load_model(self, **_kwargs):
            self.load_calls += 1

        def shutdown(self):
            self.shutdown_calls += 1

    runner = _Runner()
    worker = object.__new__(DiffusionWorker)
    worker.model_runner = runner
    worker.vllm_config = object()
    worker.local_rank = 0
    worker.rank = 0
    worker.od_config = SimpleNamespace(
        quantization_config=None,
        force_cutlass_fp8=False,
    )
    monkeypatch.setattr(worker_module, "set_forward_context", _noop_forward_context)
    monkeypatch.setattr(worker_module, "set_current_vllm_config", _noop_forward_context)
    monkeypatch.setattr(
        worker_module.current_omni_platform,
        "synchronize",
        lambda: (_ for _ in ()).throw(primary),
    )

    with pytest.raises(RuntimeError, match="platform synchronization failure") as exc_info:
        DiffusionWorker.load_model(worker)

    assert exc_info.value is primary
    assert runner.load_calls == 1
    assert runner.shutdown_calls == 1


@pytest.mark.core_model
@pytest.mark.cpu
def test_load_model_clears_cache_backend_for_unsupported_pipeline(monkeypatch):
    class _DummyLoader:
        def __init__(
            self,
            load_config,
            od_config=None,
            *,
            weight_consumer_owner=None,
            weight_consumer_factory=None,
        ):
            del load_config, od_config
            assert weight_consumer_owner is None
            assert weight_consumer_factory is None

        def load_model(self, **kwargs):
            del kwargs
            return SimpleNamespace(transformer=torch.nn.Identity())

        def take_host_weight_plan(self):
            return None

        def take_host_weight_runtime_evidence(self):
            return None

        host_weight_runtime_fell_back = False

    class _DummyMemoryProfiler:
        consumed_memory = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return False

    class _DummyCacheBackend:
        def __init__(self):
            self.enabled = False

        def enable(self, pipeline):
            del pipeline
            self.enabled = True

    dummy_cache_backend = _DummyCacheBackend()

    runner = object.__new__(DiffusionModelRunner)
    runner.vllm_config = object()
    runner.device = torch.device("cpu")
    runner.pipeline = None
    runner.cache_backend = None
    runner.offload_backend = None
    runner._weight_consumer_owner = None
    runner.od_config = SimpleNamespace(
        enable_cpu_offload=False,
        enable_layerwise_offload=False,
        enable_distributed_layerwise_offload=False,
        host_weight_runtime_mode="disabled",
        cache_backend="cache_dit",
        cache_config={},
        model_class_name="NextStep11Pipeline",
        enforce_eager=True,
        streaming_output=False,
        step_execution=False,
    )

    monkeypatch.setattr(model_runner_module, "LoadConfig", lambda: object())
    monkeypatch.setattr(model_runner_module, "DiffusersPipelineLoader", _DummyLoader)
    monkeypatch.setattr(model_runner_module, "DeviceMemoryProfiler", _DummyMemoryProfiler)
    monkeypatch.setattr(
        model_runner_module,
        "get_offload_backend",
        lambda od_config, *, device, host_weight_plan: None,
    )
    monkeypatch.setattr(
        model_runner_module.current_omni_platform,
        "init_diffusion_model_runner_runtime",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        model_runner_module, "get_cache_backend", lambda cache_backend, cache_config: dummy_cache_backend
    )

    DiffusionModelRunner.load_model(runner)

    assert runner.cache_backend is None
    assert runner.od_config.cache_backend is None
    assert dummy_cache_backend.enabled is False


@pytest.mark.core_model
@pytest.mark.cpu
def test_set_forward_context_enters_vllm_config_contexts(monkeypatch):
    """Ensure `with set_forward_context(...):` enters vllm's context managers internally and calls desired vllm functions."""
    import vllm.config.vllm as vllm_config_module
    import vllm.ir
    from vllm.config import CompilationConfig, DeviceConfig, VllmConfig

    from vllm_omni.diffusion.forward_context import (
        get_forward_context,
        is_forward_context_available,
        set_forward_context,
    )

    vllm_config = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        compilation_config=CompilationConfig(),
    )
    calls = []

    @contextmanager
    def _set_current_vllm_config(cfg):
        calls.append(("set_current_vllm_config", cfg))
        yield
        calls.append(("set_current_vllm_config_exit", cfg))

    @contextmanager
    def _set_priority(*args, **kwargs):
        del args, kwargs
        calls.append(("ir_op_priority", None))
        yield
        calls.append(("ir_op_priority_exit", None))

    @contextmanager
    def _enable_torch_wrap(flag):
        calls.append(("enable_torch_wrap", flag))
        yield
        calls.append(("enable_torch_wrap_exit", flag))

    monkeypatch.setattr(vllm_config_module, "set_current_vllm_config", _set_current_vllm_config)
    monkeypatch.setattr(vllm_config.kernel_config.ir_op_priority, "set_priority", _set_priority)
    monkeypatch.setattr(vllm.ir, "enable_torch_wrap", _enable_torch_wrap)

    assert not is_forward_context_available()

    with set_forward_context(vllm_config=vllm_config):
        assert is_forward_context_available()
        assert get_forward_context().vllm_config is vllm_config

    assert not is_forward_context_available()
    assert calls == [
        ("set_current_vllm_config", vllm_config),
        ("ir_op_priority", None),
        ("enable_torch_wrap", vllm_config.compilation_config.ir_enable_torch_wrap),
        ("enable_torch_wrap_exit", vllm_config.compilation_config.ir_enable_torch_wrap),
        ("ir_op_priority_exit", None),
        ("set_current_vllm_config_exit", vllm_config),
    ]


@pytest.mark.core_model
@pytest.mark.cpu
def test_vllm_set_forward_context_implementation(monkeypatch):
    """Regression test: ensure that vLLM's set_forward_context implementation has changed."""
    import vllm.forward_context as vllm_forward_context
    import vllm.ir
    from vllm.config import CompilationConfig, DeviceConfig, VllmConfig

    ERROR_MESSAGE = (
        "If this test fails, it likely means that vLLM's set_forward_context (vllm/forward_context.py) implementation has changed. "
        "In this case, we should update our forward_context (vllm_omni/diffusion/forward_context.py) as well. "
        "We should at least confirm that the `try: with (<what's inside?>): yield` part does not miss any information "
        "(typically by calling the same or similar stuff as vLLM). See #3352 for an example. "
        "Then, update this test to reflect the new implementation, and also update test_set_forward_context_enters_vllm_config_contexts."
    )

    vllm_config = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        compilation_config=CompilationConfig(),
    )
    calls = []

    @contextmanager
    def _set_priority():
        calls.append(("ir_op_priority", None))
        yield
        calls.append(("ir_op_priority_exit", None))

    @contextmanager
    def _enable_torch_wrap(flag):
        calls.append(("enable_torch_wrap", flag))
        yield
        calls.append(("enable_torch_wrap_exit", flag))

    def _set_additional_forward_context(**kwargs):
        calls.append(("set_additional_forward_context", tuple(sorted(kwargs.keys()))))
        return {}

    monkeypatch.setattr(vllm_config.kernel_config.ir_op_priority, "set_priority", _set_priority)
    monkeypatch.setattr(vllm.ir, "enable_torch_wrap", _enable_torch_wrap)
    monkeypatch.setattr(
        vllm_forward_context.current_platform,
        "set_additional_forward_context",
        _set_additional_forward_context,
    )

    assert not vllm_forward_context.is_forward_context_available(), ERROR_MESSAGE

    with vllm_forward_context.set_forward_context(None, vllm_config):
        assert vllm_forward_context.is_forward_context_available(), ERROR_MESSAGE
        assert vllm_forward_context.get_forward_context().attn_metadata is None, ERROR_MESSAGE

    assert not vllm_forward_context.is_forward_context_available(), ERROR_MESSAGE
    assert calls == [
        (
            "set_additional_forward_context",
            (
                "attn_metadata",
                "batch_descriptor",
                "cudagraph_runtime_mode",
                "dp_metadata",
                "num_tokens",
                "num_tokens_across_dp",
                "ubatch_slices",
                "vllm_config",
            ),
        ),
    ], ERROR_MESSAGE
