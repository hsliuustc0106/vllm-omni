# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for model-level offload through a transactional session."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import vllm_omni.diffusion.offloader.sequential_backend as backend_module
from tests.diffusion.offloader.host_weight_session_fakes import (
    RecordingPreparedSession,
    make_catalog,
    make_unit,
)
from vllm_omni.diffusion.host_weight.session import DetachMode
from vllm_omni.diffusion.host_weight.transfer import TransferPlanKind, UnitKind
from vllm_omni.diffusion.offloader.base import OffloadConfig, OffloadStrategy
from vllm_omni.diffusion.offloader.sequential_backend import (
    HostWeightModelLevelError,
    ModelLevelOffloadBackend,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _InjectedInterruption(BaseException):
    pass


@pytest.fixture(autouse=True)
def _patched_runtime(monkeypatch):
    monkeypatch.setattr(backend_module.current_omni_platform, "synchronize", lambda: None)
    monkeypatch.setattr(backend_module.current_omni_platform, "empty_cache", lambda: None)
    monkeypatch.setattr(backend_module.current_omni_platform, "get_free_memory", lambda: 0)
    monkeypatch.setattr(backend_module.current_omni_platform, "is_xpu", lambda: False)


class _Component(nn.Module):
    def __init__(self, width: int, start: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.arange(start, start + width, dtype=torch.float32),
            requires_grad=False,
        )
        self.register_buffer("scale", torch.tensor([start], dtype=torch.float64))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.weight.sum() + self.scale.float().sum()


class _Pipeline(nn.Module):
    _dit_modules = ["transformer"]
    _encoder_modules = ["text_encoder"]

    def __init__(self) -> None:
        super().__init__()
        self.transformer = _Component(5, 1.0)
        self.text_encoder = nn.Linear(1, 1, bias=False)


class _MultiDiTPipeline(_Pipeline):
    _dit_modules = ["transformer", "transformer_2"]

    def __init__(self) -> None:
        super().__init__()
        self.transformer_2 = _Component(2, 10.0)


def _prepared(pipeline: _Pipeline):
    targets: dict[str, torch.Tensor] = {}
    component = pipeline.transformer
    entries = (
        ("component0.weight", "transformer", component.weight, "parameter"),
        ("component0.scale", "transformer", component.scale, "persistent_buffer"),
    )
    targets.update({tensor_id: tensor for tensor_id, _path, tensor, _kind in entries})
    unit = make_unit("component.transformer", UnitKind.COMPONENT, entries)
    return RecordingPreparedSession(make_catalog((unit,)), targets)


def _config() -> OffloadConfig:
    return OffloadConfig(
        strategy=OffloadStrategy.MODEL_LEVEL,
        pin_cpu_memory=False,
    )


def _assert_no_model_level_session_leaks(
    backend: ModelLevelOffloadBackend,
    pipeline: _Pipeline,
    prepared: RecordingPreparedSession,
) -> None:
    assert not backend.enabled
    assert backend._host_weight_terminal
    assert backend._prepared_weight_session is None
    assert backend._weight_session is None
    assert backend._host_weight_controller is None
    assert backend._offload_modules == []
    assert all(
        getattr(component, "_hook_registry", None) is None
        or component._hook_registry.get_hook("sequential_offload") is None
        for component in (pipeline.transformer, pipeline.text_encoder)
    )
    if prepared.active is not None:
        assert prepared.active.closed
        assert not prepared.active.open_units
        assert not prepared.active.active_bindings


def test_component_session_commits_after_hooks_and_uses_one_host_slot(monkeypatch) -> None:
    pipeline = _Pipeline()
    expected = pipeline.transformer(torch.tensor(0.0))
    prepared = _prepared(pipeline)
    backend = ModelLevelOffloadBackend(
        _config(),
        torch.device("cpu"),
        prepared_weight_session=prepared,
    )

    backend.enable(pipeline)

    assert prepared.commit_count == 1
    assert prepared.rollback_count == 0
    assert prepared.events[0] == "commit"
    assert pipeline.transformer._hook_registry.get_hook("sequential_offload") is not None
    controller = backend._host_weight_controller
    assert controller is not None
    assert controller.capacities == {torch.float32: 5, torch.float64: 1}

    actual = pipeline.transformer(torch.tensor(0.0))
    assert torch.equal(actual, expected)
    assert prepared.events[-3:] == [
        ("copy", "component.transformer"),
        ("unit_close", "component.transformer"),
        ("bind", "component.transformer"),
    ]
    assert prepared.active is not None
    assert len(prepared.active.destination_storage_ids) == 1

    pipeline.text_encoder(torch.ones(1))
    assert ("release", "component.transformer") in prepared.events
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda _tensor: True)
    assert backend.host_weight_diagnostics() == {
        "pinned_slot_budget_bytes": 28,
        "events": 0,
    }

    backend.disable()
    assert backend.host_weight_diagnostics() == {
        "pinned_slot_budget_bytes": 0,
        "events": 0,
    }
    assert prepared.active.closed
    assert prepared.events[-2:] == [
        "suspend",
        ("close", DetachMode.TERMINAL),
    ]
    assert pipeline.transformer._hook_registry.get_hook("sequential_offload") is None


def test_multiple_dit_targets_roll_back_before_hook_install() -> None:
    pipeline = _MultiDiTPipeline()
    prepared = _prepared(pipeline)
    backend = ModelLevelOffloadBackend(
        _config(),
        torch.device("cpu"),
        prepared_weight_session=prepared,
    )

    with pytest.raises(HostWeightModelLevelError, match="exactly one managed DiT target"):
        backend.enable(pipeline)

    assert prepared.commit_count == 0
    assert prepared.rollback_count == 1
    assert getattr(pipeline.transformer, "_hook_registry", None) is None


def test_component_session_rejects_a_different_selected_plan_kind() -> None:
    pipeline = _Pipeline()
    prepared = _prepared(pipeline)
    prepared.capabilities.selected_transfer_plan_kind = TransferPlanKind.BLOCKS_PLUS_RESIDENT
    backend = ModelLevelOffloadBackend(
        _config(),
        torch.device("cpu"),
        prepared_weight_session=prepared,
    )

    with pytest.raises(HostWeightModelLevelError, match="different transfer plan kind"):
        backend.enable(pipeline)

    assert prepared.commit_count == 0
    assert prepared.rollback_count == 1


def test_discovery_failure_rolls_back_prepared_session(monkeypatch) -> None:
    pipeline = _Pipeline()
    prepared = _prepared(pipeline)
    backend = ModelLevelOffloadBackend(
        _config(),
        torch.device("cpu"),
        prepared_weight_session=prepared,
    )
    failure = RuntimeError("injected model-level discovery failure")

    def fail_discovery(_pipeline):
        raise failure

    monkeypatch.setattr(
        backend_module.ModuleDiscovery,
        "discover",
        staticmethod(fail_discovery),
    )

    with pytest.raises(RuntimeError, match="model-level discovery failure") as exc_info:
        backend.enable(pipeline)

    assert exc_info.value is failure
    assert prepared.commit_count == 0
    assert prepared.rollback_count == 1
    _assert_no_model_level_session_leaks(backend, pipeline, prepared)
    backend.disable()
    assert prepared.rollback_count == 1


def test_cleanup_base_exception_is_not_substituted_for_discovery_failure(monkeypatch) -> None:
    class PrimaryError(RuntimeError):
        def add_note(self, _note: str) -> None:
            raise SystemExit("injected model-level add_note failure")

    class CleanupError(RuntimeError):
        def __str__(self) -> str:
            raise SystemExit("injected model-level cleanup __str__ failure")

    pipeline = _Pipeline()
    prepared = _prepared(pipeline)
    backend = ModelLevelOffloadBackend(
        _config(),
        torch.device("cpu"),
        prepared_weight_session=prepared,
    )
    primary = PrimaryError("primary model-level discovery failure")
    cleanup_failure = CleanupError("injected model-level rollback interruption")
    reporting_calls = 0

    def fail_discovery(_pipeline):
        raise primary

    def fail_rollback():
        prepared.rollback_count += 1
        if prepared.rollback_count == 1:
            raise cleanup_failure

    def fail_reporting(*_args, **_kwargs):
        nonlocal reporting_calls
        reporting_calls += 1
        raise SystemExit("injected model-level cleanup logging failure")

    monkeypatch.setattr(
        backend_module.ModuleDiscovery,
        "discover",
        staticmethod(fail_discovery),
    )
    monkeypatch.setattr(prepared, "rollback", fail_rollback)
    monkeypatch.setattr(backend_module.logger, "error", fail_reporting)

    with pytest.raises(RuntimeError, match="primary model-level discovery failure") as exc_info:
        backend.enable(pipeline)

    assert exc_info.value is primary
    assert reporting_calls == 1
    assert prepared.rollback_count == 1
    assert backend._prepared_weight_session is prepared
    backend.disable()
    assert prepared.rollback_count == 2
    _assert_no_model_level_session_leaks(backend, pipeline, prepared)


def test_disable_retries_transient_session_close_failure() -> None:
    pipeline = _Pipeline()
    prepared = _prepared(pipeline)
    backend = ModelLevelOffloadBackend(
        _config(),
        torch.device("cpu"),
        prepared_weight_session=prepared,
    )
    backend.enable(pipeline)
    assert prepared.active is not None
    original_close = prepared.active.close
    failure = RuntimeError("injected model-level session close failure")
    close_calls = 0

    def fail_close_once(mode):
        nonlocal close_calls
        close_calls += 1
        if close_calls == 1:
            raise failure
        original_close(mode)

    prepared.active.close = fail_close_once

    with pytest.raises(RuntimeError, match="session close failure") as exc_info:
        backend.disable()

    assert exc_info.value is failure
    assert backend._weight_session is prepared.active
    assert prepared.active.suspended
    assert not prepared.active.closed

    backend.disable()

    assert close_calls == 2
    _assert_no_model_level_session_leaks(backend, pipeline, prepared)


def test_disable_sync_failure_retains_binding_and_pending_unit_primary(
    monkeypatch,
) -> None:
    pipeline = _Pipeline()
    prepared = _prepared(pipeline)
    backend = ModelLevelOffloadBackend(
        _config(),
        torch.device("cpu"),
        prepared_weight_session=prepared,
    )
    backend.enable(pipeline)
    pipeline.transformer(torch.tensor(0.0))
    controller = backend._host_weight_controller
    assert controller is not None
    active_binding = controller._active_binding
    assert active_binding is not None
    primary = RuntimeError("pending model-level read cleanup failed")

    class _FailOncePendingUnit:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                raise primary

    pending_unit = _FailOncePendingUnit()
    controller._pending_units.append(pending_unit)
    sync_calls = 0

    def fail_once_sync() -> None:
        nonlocal sync_calls
        sync_calls += 1
        if sync_calls == 1:
            raise RuntimeError("model-level teardown sync failed")

    monkeypatch.setattr(backend_module.current_omni_platform, "synchronize", fail_once_sync)

    with pytest.raises(RuntimeError, match="pending model-level read cleanup failed") as exc_info:
        backend.disable()

    assert exc_info.value is primary
    assert any("sync" in note for note in primary.__notes__)
    assert active_binding.release_calls == 0
    assert controller._active_binding is active_binding
    assert backend._host_weight_teardown_phase.value == "active"

    backend.disable()
    assert pending_unit.close_calls == 2
    assert active_binding.release_calls == 1
    _assert_no_model_level_session_leaks(backend, pipeline, prepared)


@pytest.mark.parametrize("phase", ["before_commit", "after_commit"])
def test_base_exception_during_enable_preserves_primary_and_cleans_once(
    monkeypatch,
    phase: str,
) -> None:
    pipeline = _Pipeline()
    prepared = _prepared(pipeline)
    backend = ModelLevelOffloadBackend(
        _config(),
        torch.device("cpu"),
        prepared_weight_session=prepared,
    )
    failure = _InjectedInterruption(f"injected model-level {phase} interruption")

    if phase == "before_commit":

        def interrupt_commit():
            raise failure

        monkeypatch.setattr(prepared, "commit", interrupt_commit)
    else:
        original_attach = backend_module._HostWeightComponentController.attach

        def interrupt_after_attach(controller, session):
            original_attach(controller, session)
            raise failure

        monkeypatch.setattr(
            backend_module._HostWeightComponentController,
            "attach",
            interrupt_after_attach,
        )

    with pytest.raises(_InjectedInterruption) as exc_info:
        backend.enable(pipeline)

    assert exc_info.value is failure
    if phase == "before_commit":
        assert prepared.commit_count == 0
        assert prepared.rollback_count == 1
        assert prepared.active is None
    else:
        assert prepared.commit_count == 1
        assert prepared.rollback_count == 0
        assert prepared.active is not None
        assert prepared.events.count(("close", DetachMode.TERMINAL)) == 1
    _assert_no_model_level_session_leaks(backend, pipeline, prepared)
    backend.disable()
    assert prepared.rollback_count == (phase == "before_commit")
    assert prepared.events.count(("close", DetachMode.TERMINAL)) == (phase == "after_commit")


@pytest.mark.parametrize("store_before_interrupt", [False, True])
def test_commit_publication_interruption_retains_exactly_one_cleanup_owner(
    store_before_interrupt,
) -> None:
    primary = _InjectedInterruption("injected model-level adoption interruption")

    class _InterruptingBackend(ModelLevelOffloadBackend):
        _interrupt_publication = False

        def __setattr__(self, name, value) -> None:
            if name == "_weight_session" and value is not None and self._interrupt_publication:
                object.__setattr__(self, "_interrupt_publication", False)
                if store_before_interrupt:
                    object.__setattr__(self, name, value)
                raise primary
            super().__setattr__(name, value)

    pipeline = _Pipeline()
    prepared = _prepared(pipeline)
    backend = _InterruptingBackend(
        _config(),
        torch.device("cpu"),
        prepared_weight_session=prepared,
    )
    backend._interrupt_publication = True

    with pytest.raises(_InjectedInterruption) as exc_info:
        backend.enable(pipeline)

    assert exc_info.value is primary
    assert prepared.commit_count == 1
    assert prepared.rollback_count == (not store_before_interrupt)
    _assert_no_model_level_session_leaks(backend, pipeline, prepared)


def test_copy_failure_after_commit_tears_down_terminally() -> None:
    pipeline = _Pipeline()
    prepared = _prepared(pipeline)
    prepared.fail_unit_id = "component.transformer"
    backend = ModelLevelOffloadBackend(
        _config(),
        torch.device("cpu"),
        prepared_weight_session=prepared,
    )
    backend.enable(pipeline)

    with pytest.raises(RuntimeError, match="injected session read failure"):
        pipeline.transformer(torch.tensor(0.0))

    assert prepared.commit_count == 1
    assert prepared.rollback_count == 0
    assert not backend.enabled
    assert backend._host_weight_terminal
    assert backend._weight_session is None
    assert backend._host_weight_controller is None
    assert prepared.active is not None
    assert prepared.active.closed
    assert not prepared.active.open_units
    assert not prepared.active.active_bindings
    assert pipeline.transformer._hook_registry.get_hook("sequential_offload") is None


def test_disable_retries_if_suspend_return_is_interrupted_before_phase_update() -> None:
    pipeline = _Pipeline()
    prepared = _prepared(pipeline)
    backend = ModelLevelOffloadBackend(
        _config(),
        torch.device("cpu"),
        prepared_weight_session=prepared,
    )
    backend.enable(pipeline)
    active = prepared.active
    assert active is not None
    original_suspend = active.suspend
    primary = _InjectedInterruption("injected model-level suspend return interruption")

    def suspend_then_interrupt() -> None:
        original_suspend()
        raise primary

    active.suspend = suspend_then_interrupt
    with pytest.raises(_InjectedInterruption) as exc_info:
        backend.disable()

    assert exc_info.value is primary
    assert active.suspended
    assert active.suspend_count == 1
    assert backend._host_weight_teardown_phase.value == "active"

    active.suspend = original_suspend
    backend.disable()

    assert active.suspend_count == 1
    _assert_no_model_level_session_leaks(backend, pipeline, prepared)


def test_hook_primary_survives_fail_closed_cleanup_and_reporting_failure(monkeypatch) -> None:
    pipeline = _Pipeline()
    prepared = _prepared(pipeline)
    backend = ModelLevelOffloadBackend(
        _config(),
        torch.device("cpu"),
        prepared_weight_session=prepared,
    )
    backend.enable(pipeline)
    assert prepared.active is not None
    controller = backend._host_weight_controller
    assert controller is not None

    primary = RuntimeError("primary model-level session read failure")
    cleanup_failure = RuntimeError("injected model-level fail-closed cleanup failure")
    original_open_unit = prepared.active.open_unit
    reporting_calls = 0

    def open_unit_with_failure(request):
        prepared_unit = original_open_unit(request)

        def fail_copy(_destination):
            raise primary

        prepared_unit.copy_into = fail_copy
        return prepared_unit

    def fail_cleanup():
        raise cleanup_failure

    def fail_reporting(*_args, **_kwargs):
        nonlocal reporting_calls
        reporting_calls += 1
        raise SystemExit("injected model-level fail-closed logging failure")

    prepared.active.open_unit = open_unit_with_failure
    controller._fail_closed = fail_cleanup
    monkeypatch.setattr(backend_module.logger, "error", fail_reporting)

    with pytest.raises(RuntimeError, match="primary model-level session read failure") as exc_info:
        pipeline.transformer(torch.tensor(0.0))

    assert exc_info.value is primary
    assert reporting_calls == 1

    backend.disable()

    _assert_no_model_level_session_leaks(backend, pipeline, prepared)
