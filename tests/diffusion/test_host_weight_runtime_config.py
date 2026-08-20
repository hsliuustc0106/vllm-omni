# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.diffusion.data import (
    HostWeightOffloadConfigurationError,
    HostWeightOffloadValidationCode,
    OmniDiffusionConfig,
)
from vllm_omni.diffusion.offloader.base import OffloadConfig, OffloadStrategy
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


@pytest.mark.parametrize(
    "offload_flags",
    [
        {"enable_cpu_offload": True, "enable_layerwise_offload": True},
        {
            "enable_cpu_offload": True,
            "enable_distributed_layerwise_offload": True,
            "dlo_use_allgather": False,
        },
        {
            "enable_layerwise_offload": True,
            "enable_distributed_layerwise_offload": True,
            "dlo_use_allgather": False,
        },
    ],
)
def test_hwr_rejects_conflicting_offload_modes(tmp_path, offload_flags):
    with pytest.raises(HostWeightOffloadConfigurationError) as exc_info:
        OmniDiffusionConfig(
            model="test",
            host_weight_runtime_mode="read_write",
            host_weight_runtime_root=str(tmp_path),
            **offload_flags,
        )

    assert exc_info.value.code is HostWeightOffloadValidationCode.CONFLICTING_OFFLOAD_MODES


def test_hwr_requires_one_offloader(tmp_path):
    with pytest.raises(HostWeightOffloadConfigurationError) as exc_info:
        OmniDiffusionConfig(
            model="test",
            host_weight_runtime_mode="read_write",
            host_weight_runtime_root=str(tmp_path),
        )

    assert exc_info.value.code is HostWeightOffloadValidationCode.HWR_REQUIRES_OFFLOADER


def test_hwr_rejects_dlo_allgather(tmp_path):
    with pytest.raises(HostWeightOffloadConfigurationError) as exc_info:
        OmniDiffusionConfig(
            model="test",
            enable_distributed_layerwise_offload=True,
            dlo_use_allgather=True,
            host_weight_runtime_mode="read_write",
            host_weight_runtime_root=str(tmp_path),
        )

    assert exc_info.value.code is HostWeightOffloadValidationCode.HWR_DLO_ALLGATHER_UNSUPPORTED


def test_required_hwr_cannot_be_disabled():
    with pytest.raises(HostWeightOffloadConfigurationError) as exc_info:
        OmniDiffusionConfig(
            model="test",
            host_weight_runtime_required=True,
        )

    assert exc_info.value.code is HostWeightOffloadValidationCode.HWR_REQUIRED_WHILE_DISABLED


@pytest.mark.parametrize(
    ("offload_flags", "expected_strategy"),
    [
        ({"enable_cpu_offload": True}, OffloadStrategy.MODEL_LEVEL),
        ({"enable_layerwise_offload": True}, OffloadStrategy.LAYER_WISE),
        (
            {
                "enable_distributed_layerwise_offload": True,
                "dlo_use_allgather": False,
            },
            OffloadStrategy.DISTRIBUTED_LAYER_WISE,
        ),
    ],
)
def test_hwr_accepts_exactly_one_supported_offloader(
    tmp_path,
    offload_flags,
    expected_strategy,
):
    config = OmniDiffusionConfig(
        model="test",
        host_weight_runtime_mode="read_write",
        host_weight_runtime_root=str(tmp_path),
        **offload_flags,
    )

    assert OffloadConfig.from_od_config(config).strategy is expected_strategy


def test_disabled_hwr_preserves_legacy_offloader_priority():
    config = OmniDiffusionConfig(
        model="test",
        enable_cpu_offload=True,
        enable_layerwise_offload=True,
        enable_distributed_layerwise_offload=True,
    )

    assert OffloadConfig.from_od_config(config).strategy is OffloadStrategy.DISTRIBUTED_LAYER_WISE


def test_offloader_boundary_revalidates_mutated_hwr_config(tmp_path):
    config = OmniDiffusionConfig(
        model="test",
        enable_cpu_offload=True,
        host_weight_runtime_mode="read_write",
        host_weight_runtime_root=str(tmp_path),
    )
    config.enable_layerwise_offload = True

    with pytest.raises(HostWeightOffloadConfigurationError) as exc_info:
        OffloadConfig.from_od_config(config)

    assert exc_info.value.code is HostWeightOffloadValidationCode.CONFLICTING_OFFLOAD_MODES


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
