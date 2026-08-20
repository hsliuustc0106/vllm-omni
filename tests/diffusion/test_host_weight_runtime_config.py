# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker


def test_host_weight_runtime_is_disabled_by_default():
    config = OmniDiffusionConfig(model="test")

    assert config.host_weight_runtime_mode == "disabled"
    assert config.host_weight_runtime_root is None
    assert config.host_weight_runtime_required is False
    assert config.host_weight_runtime_wait_timeout_s == 120.0


@pytest.mark.parametrize("mode", ["read_only", "read_write"])
def test_host_weight_runtime_requires_explicit_root(mode: str):
    with pytest.raises(ValueError, match="host_weight_runtime_root is required"):
        OmniDiffusionConfig(model="test", host_weight_runtime_mode=mode)


def test_host_weight_runtime_rejects_unknown_mode(tmp_path):
    with pytest.raises(ValueError, match="host_weight_runtime_mode must be"):
        OmniDiffusionConfig(
            model="test",
            host_weight_runtime_mode="automatic",
            host_weight_runtime_root=str(tmp_path),
        )


@pytest.mark.parametrize(
    "timeout",
    [0.0, -1.0, float("nan"), float("inf"), float("-inf")],
)
def test_host_weight_runtime_rejects_non_positive_or_non_finite_wait_timeout(
    timeout: float,
):
    with pytest.raises(
        ValueError,
        match="host_weight_runtime_wait_timeout_s must be finite and positive",
    ):
        OmniDiffusionConfig(
            model="test",
            host_weight_runtime_wait_timeout_s=timeout,
        )


def test_runner_shutdown_delegates_session_teardown_to_offloader():
    calls: list[str] = []

    class Backend:
        def disable(self) -> None:
            calls.append("offloader")

    runner = DiffusionModelRunner.__new__(DiffusionModelRunner)
    runner.offload_backend = Backend()

    runner.shutdown()

    assert calls == ["offloader"]
    assert runner.offload_backend is None


def test_runner_shutdown_retains_backend_when_terminal_teardown_fails():
    calls: list[str] = []

    class Backend:
        def disable(self) -> None:
            calls.append("offloader")
            raise RuntimeError("drain failed")

    runner = DiffusionModelRunner.__new__(DiffusionModelRunner)
    backend = Backend()
    runner.offload_backend = backend

    with pytest.raises(RuntimeError, match="drain failed"):
        runner.shutdown()

    assert calls == ["offloader"]
    assert runner.offload_backend is backend


def test_worker_shutdown_attempts_runner_after_prefetch_failure(
    monkeypatch,
) -> None:
    calls: list[str] = []

    class PrefetchManager:
        def shutdown_prefetch(self) -> None:
            calls.append("prefetch")
            raise RuntimeError("prefetch failed")

    class Runner:
        kv_transfer_manager = PrefetchManager()

        def shutdown(self) -> None:
            calls.append("runner")

    monkeypatch.setattr(
        "vllm_omni.diffusion.worker.diffusion_worker.destroy_distributed_env",
        lambda: calls.append("distributed"),
    )
    worker = DiffusionWorker.__new__(DiffusionWorker)
    worker.model_runner = Runner()

    with pytest.raises(RuntimeError, match="prefetch failed"):
        worker.shutdown()

    assert calls == ["prefetch", "runner", "distributed"]


def test_worker_shutdown_preserves_first_error_when_aggregation_fails(
    monkeypatch,
) -> None:
    class PrimaryError(RuntimeError):
        def add_note(self, _note: str) -> None:
            raise SystemExit("injected shutdown add_note failure")

    class UnprintableSecondaryError(RuntimeError):
        def __str__(self) -> str:
            raise SystemExit("injected shutdown secondary __str__ failure")

    calls: list[str] = []
    primary = PrimaryError("primary prefetch shutdown failure")

    class PrefetchManager:
        def shutdown_prefetch(self) -> None:
            calls.append("prefetch")
            raise primary

    class Runner:
        kv_transfer_manager = PrefetchManager()

        def shutdown(self) -> None:
            calls.append("runner")
            raise UnprintableSecondaryError("secondary runner shutdown failure")

    monkeypatch.setattr(
        "vllm_omni.diffusion.worker.diffusion_worker.destroy_distributed_env",
        lambda: calls.append("distributed"),
    )
    worker = DiffusionWorker.__new__(DiffusionWorker)
    worker.model_runner = Runner()

    with pytest.raises(RuntimeError, match="primary prefetch shutdown failure") as exc_info:
        worker.shutdown()

    assert exc_info.value is primary
    assert calls == ["prefetch", "runner", "distributed"]
