# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for DLO no-AllGather through a transactional session."""

from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch
from torch import nn

import vllm_omni.diffusion.offloader.distributed_layerwise_backend as backend_module
from tests.diffusion.offloader.host_weight_session_fakes import (
    RecordingPreparedSession,
    make_catalog,
    make_unit,
)
from vllm_omni.diffusion.host_weight.session import DetachMode
from vllm_omni.diffusion.host_weight.transfer import TransferPlanKind, UnitKind
from vllm_omni.diffusion.offloader.base import OffloadConfig, OffloadStrategy
from vllm_omni.diffusion.offloader.distributed_layerwise_backend import (
    DistributedLayerwiseOffloadBackend,
)
from vllm_omni.diffusion.offloader.offload_plan import OffloadPlan

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _InjectedInterruption(BaseException):
    pass


class _DummyStream:
    def wait_stream(self, _stream) -> None:
        return None

    def wait_event(self, _event) -> None:
        return None


class _DummyEvent:
    instances: list[_DummyEvent] = []

    def __init__(self) -> None:
        self.synchronize_count = 0
        self.complete = False
        self.instances.append(self)

    def record(self, _stream) -> None:
        self.complete = False

    def synchronize(self) -> None:
        self.synchronize_count += 1
        self.complete = True

    def query(self) -> bool:
        return self.complete


@contextmanager
def _dummy_stream(_stream):
    yield None


@pytest.fixture(autouse=True)
def _patched_runtime(monkeypatch):
    _DummyEvent.instances.clear()
    synchronization = {"count": 0}
    monkeypatch.setattr(backend_module.current_omni_platform, "Stream", _DummyStream)
    monkeypatch.setattr(backend_module.current_omni_platform, "Event", _DummyEvent)
    monkeypatch.setattr(backend_module.current_omni_platform, "current_stream", _DummyStream)
    monkeypatch.setattr(backend_module.current_omni_platform, "stream", _dummy_stream)
    monkeypatch.setattr(
        backend_module.current_omni_platform,
        "synchronize",
        lambda: synchronization.__setitem__("count", synchronization["count"] + 1),
    )
    monkeypatch.setattr(backend_module.current_omni_platform, "empty_cache", lambda: None)
    return synchronization


class _Block(nn.Module):
    def __init__(self, width: int, start: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.arange(start, start + width, dtype=torch.float32),
            requires_grad=False,
        )
        self.register_buffer("scale", torch.tensor([start], dtype=torch.float64))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.weight.sum() + self.scale.float().sum()


class _Transformer(nn.Module):
    _layerwise_offload_blocks_attrs = ["blocks"]

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block(2, 1.0), _Block(5, 10.0), _Block(3, 20.0)])
        self.bias = nn.Parameter(torch.tensor([0.5]), requires_grad=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            value = block(value)
        return value + self.bias.sum()


class _Pipeline(nn.Module):
    _dit_modules = ["transformer"]

    def __init__(self) -> None:
        super().__init__()
        self.transformer = _Transformer()


class _Auxiliary(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Block(4, 30.0), _Block(2, 40.0)])
        self.bias = nn.Parameter(torch.tensor([1.5]), requires_grad=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            value = layer(value)
        return value + self.bias.sum()


class _PlannedTransformer(_Transformer):
    def __init__(self) -> None:
        super().__init__()
        self.token_refiner = _Auxiliary()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = super().forward(value)
        return self.token_refiner(value)


class _PlannedPipeline(nn.Module):
    _dit_modules = ["transformer"]
    _offload_plan = OffloadPlan(offload_submodules={"token_refiner": "layers"})

    def __init__(self) -> None:
        super().__init__()
        self.transformer = _PlannedTransformer()


def _prepared(pipeline: _Pipeline, *, block_count: int = 3):
    units = []
    targets: dict[str, torch.Tensor] = {}
    for index, block in enumerate(pipeline.transformer.blocks[:block_count]):
        path = f"transformer.blocks.{index}"
        entries = (
            (f"block{index}.weight", path, block.weight, "parameter"),
            (f"block{index}.scale", path, block.scale, "persistent_buffer"),
        )
        targets.update({tensor_id: tensor for tensor_id, _path, tensor, _kind in entries})
        units.append(make_unit(f"block.{index}", UnitKind.BLOCK, entries))
    if block_count == 3:
        resident_entries = (("resident.bias", "transformer", pipeline.transformer.bias, "parameter"),)
        targets["resident.bias"] = pipeline.transformer.bias
        units.append(make_unit("resident.transformer", UnitKind.RESIDENT, resident_entries))
    return RecordingPreparedSession(make_catalog(units), targets)


def _planned_prepared(pipeline: _PlannedPipeline) -> RecordingPreparedSession:
    units = []
    targets: dict[str, torch.Tensor] = {}
    rings = (
        ("main", "transformer.blocks", pipeline.transformer.blocks),
        (
            "refiner",
            "transformer.token_refiner.layers",
            pipeline.transformer.token_refiner.layers,
        ),
    )
    for ring_name, base_path, blocks in rings:
        for index, block in enumerate(blocks):
            path = f"{base_path}.{index}"
            entries = (
                (f"{ring_name}.{index}.weight", path, block.weight, "parameter"),
                (
                    f"{ring_name}.{index}.scale",
                    path,
                    block.scale,
                    "persistent_buffer",
                ),
            )
            targets.update({tensor_id: tensor for tensor_id, _path, tensor, _kind in entries})
            units.append(
                make_unit(
                    f"block.{ring_name}.{index}",
                    UnitKind.BLOCK,
                    entries,
                )
            )
    resident_entries = (
        (
            "resident.transformer_bias",
            "transformer",
            pipeline.transformer.bias,
            "parameter",
        ),
        (
            "resident.refiner_bias",
            "transformer.token_refiner",
            pipeline.transformer.token_refiner.bias,
            "parameter",
        ),
    )
    targets.update({tensor_id: tensor for tensor_id, _path, tensor, _kind in resident_entries})
    units.append(
        make_unit(
            "resident.transformer",
            UnitKind.RESIDENT,
            resident_entries,
        )
    )
    return RecordingPreparedSession(make_catalog(units), targets)


def _config(
    *,
    use_allgather: bool = False,
    dp_size: int = 4,
    resident_layers: int = 0,
) -> OffloadConfig:
    return OffloadConfig(
        strategy=OffloadStrategy.DISTRIBUTED_LAYER_WISE,
        pin_cpu_memory=False,
        dp_size=dp_size,
        dlo_use_allgather=use_allgather,
        dlo_resident_layers=resident_layers,
    )


def _assert_terminal_session_cleanup(
    backend: DistributedLayerwiseOffloadBackend,
    pipeline: _Pipeline,
    prepared: RecordingPreparedSession,
) -> None:
    assert not backend.enabled
    assert backend._host_weight_terminal
    assert backend._weight_session is None
    assert backend._weight_session_handle.session is None
    assert backend._weight_session_blocks == {}
    assert backend._session_slot_bindings == [None, None]
    assert backend._resident_weight_controller is None
    assert backend._stage_resident_weight_controller is None
    assert prepared.active is not None
    assert prepared.active.closed
    assert not prepared.active.open_units
    assert not prepared.active.active_bindings
    assert all(
        getattr(block, "_hook_registry", None) is None
        or block._hook_registry.get_hook("distributed_layerwise_offload") is None
        for block in pipeline.transformer.blocks
    )


def _assert_no_dlo_session_leaks(
    backend: DistributedLayerwiseOffloadBackend,
    pipeline: _Pipeline,
    prepared: RecordingPreparedSession,
) -> None:
    assert not backend.enabled
    assert backend._host_weight_terminal
    assert backend._prepared_weight_session is None
    assert backend._weight_session is None
    assert backend._weight_session_handle.session is None
    assert backend._weight_session_blocks == {}
    assert backend._session_slot_bindings == [None, None]
    assert backend._session_hooked_blocks == []
    assert backend._blocks == []
    assert backend._all_hook_groups == []
    assert backend._resident_blocks == []
    assert backend._resident_weight_controller is None
    assert backend._stage_resident_weight_controller is None
    assert all(
        getattr(block, "_hook_registry", None) is None
        or block._hook_registry.get_hook("distributed_layerwise_offload") is None
        for block in pipeline.transformer.blocks
    )
    if prepared.active is not None:
        assert prepared.active.closed
        assert not prepared.active.open_units
        assert not prepared.active.active_bindings


def test_session_path_is_two_slot_h2d_only_and_terminally_detached(
    monkeypatch,
    _patched_runtime,
) -> None:
    pipeline = _Pipeline()
    expected = pipeline.transformer(torch.tensor(0.0))
    prepared = _prepared(pipeline)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_into_tensor",
        lambda *_args, **_kwargs: pytest.fail("DLO session path entered a collective"),
    )
    backend = DistributedLayerwiseOffloadBackend(
        _config(),
        torch.device("cpu"),
        prepared_weight_session=prepared,
    )

    backend.enable(pipeline)

    assert backend.dp_size == 1
    assert backend.dp_group is None
    assert prepared.commit_count == 1
    hooks = [hook for group in backend._all_hook_groups for hook in group]
    assert hooks
    assert all(not hook.cpu_shards and not hook.cpu_sources for hook in hooks)
    assert len(hooks[0].gpu_buffers) == 2
    assert len(hooks[0].cpu_staging_buffers) == 2
    assert ("bind", "resident.transformer") in prepared.events
    assert backend._session_slot_bindings == [None, None]
    assert prepared.active is not None
    assert {binding.unit.unit_id for binding in prepared.active.active_bindings} == {"resident.transformer"}

    actual = pipeline.transformer(torch.tensor(0.0))
    assert torch.equal(actual, expected)
    block_destinations = {
        destination for destination in prepared.active.destination_storage_ids if len(destination) == 2
    }
    assert len(block_destinations) == 2
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda _tensor: True)
    diagnostics = backend.host_weight_diagnostics()
    assert diagnostics["pinned_slot_budget_bytes"] == 60
    assert diagnostics["events"] > 0
    for event in _DummyEvent.instances:
        event.synchronize()
    assert backend.host_weight_diagnostics()["events"] == 0

    sync_before = _patched_runtime["count"]
    backend.disable()
    assert backend.host_weight_diagnostics() == {
        "pinned_slot_budget_bytes": 0,
        "events": 0,
    }
    assert _patched_runtime["count"] > sync_before
    assert prepared.events[-2:] == [
        "suspend",
        ("close", DetachMode.TERMINAL),
    ]
    assert prepared.active.closed
    assert all(
        block._hook_registry.get_hook("distributed_layerwise_offload") is None for block in pipeline.transformer.blocks
    )


def test_missing_block_unit_rolls_back_before_hook_mutation() -> None:
    pipeline = _Pipeline()
    prepared = _prepared(pipeline, block_count=2)
    backend = DistributedLayerwiseOffloadBackend(
        _config(),
        torch.device("cpu"),
        prepared_weight_session=prepared,
    )

    with pytest.raises(ValueError, match="no execution binding"):
        backend.enable(pipeline)

    assert prepared.commit_count == 0
    assert prepared.rollback_count == 1
    assert all(getattr(block, "_hook_registry", None) is None for block in pipeline.transformer.blocks)


def test_dlo_session_rejects_a_different_selected_plan_kind() -> None:
    pipeline = _Pipeline()
    prepared = _prepared(pipeline)
    prepared.capabilities.selected_transfer_plan_kind = TransferPlanKind.COMPONENT
    backend = DistributedLayerwiseOffloadBackend(
        _config(),
        torch.device("cpu"),
        prepared_weight_session=prepared,
    )

    with pytest.raises(ValueError, match="different transfer plan kind"):
        backend.enable(pipeline)

    assert prepared.commit_count == 0
    assert prepared.rollback_count == 1


def test_discovery_failure_rolls_back_prepared_session(monkeypatch) -> None:
    pipeline = _Pipeline()
    prepared = _prepared(pipeline)
    backend = DistributedLayerwiseOffloadBackend(
        _config(),
        torch.device("cpu"),
        prepared_weight_session=prepared,
    )
    failure = RuntimeError("injected DLO discovery failure")

    def fail_discovery(_pipeline):
        raise failure

    monkeypatch.setattr(
        backend_module.ModuleDiscovery,
        "discover",
        staticmethod(fail_discovery),
    )

    with pytest.raises(RuntimeError, match="DLO discovery failure") as exc_info:
        backend.enable(pipeline)

    assert exc_info.value is failure
    assert prepared.commit_count == 0
    assert prepared.rollback_count == 1
    _assert_no_dlo_session_leaks(backend, pipeline, prepared)
    backend.disable()
    assert prepared.rollback_count == 1


def test_cleanup_base_exception_is_not_substituted_for_discovery_failure(monkeypatch) -> None:
    pipeline = _Pipeline()
    prepared = _prepared(pipeline)
    backend = DistributedLayerwiseOffloadBackend(
        _config(),
        torch.device("cpu"),
        prepared_weight_session=prepared,
    )
    primary = RuntimeError("primary DLO discovery failure")

    def fail_discovery(_pipeline):
        raise primary

    def fail_rollback():
        prepared.rollback_count += 1
        if prepared.rollback_count == 1:
            raise SystemExit("injected DLO rollback interruption")

    monkeypatch.setattr(
        backend_module.ModuleDiscovery,
        "discover",
        staticmethod(fail_discovery),
    )
    monkeypatch.setattr(prepared, "rollback", fail_rollback)

    with pytest.raises(RuntimeError, match="primary DLO discovery failure") as exc_info:
        backend.enable(pipeline)

    assert exc_info.value is primary
    assert any("rollback interruption" in note for note in primary.__notes__)
    assert prepared.rollback_count == 1
    assert backend._prepared_weight_session is prepared
    backend.disable()
    assert prepared.rollback_count == 2
    _assert_no_dlo_session_leaks(backend, pipeline, prepared)


@pytest.mark.parametrize("phase", ["before_commit", "after_commit"])
def test_base_exception_during_enable_preserves_primary_and_cleans_once(
    monkeypatch,
    phase: str,
) -> None:
    pipeline = _Pipeline()
    prepared = _prepared(pipeline)
    backend = DistributedLayerwiseOffloadBackend(
        _config(),
        torch.device("cpu"),
        prepared_weight_session=prepared,
    )
    failure = _InjectedInterruption(f"injected DLO {phase} interruption")

    if phase == "before_commit":

        def interrupt_commit():
            raise failure

        monkeypatch.setattr(prepared, "commit", interrupt_commit)
    else:
        original_load = backend_module._SessionResidentController.load

        def interrupt_after_resident_load(controller, session):
            original_load(controller, session)
            raise failure

        monkeypatch.setattr(
            backend_module._SessionResidentController,
            "load",
            interrupt_after_resident_load,
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
    _assert_no_dlo_session_leaks(backend, pipeline, prepared)
    backend.disable()
    assert prepared.rollback_count == (phase == "before_commit")
    assert prepared.events.count(("close", DetachMode.TERMINAL)) == (phase == "after_commit")


@pytest.mark.parametrize("store_before_interrupt", [False, True])
def test_commit_publication_interruption_retains_exactly_one_cleanup_owner(
    store_before_interrupt,
) -> None:
    primary = _InjectedInterruption("injected DLO adoption interruption")

    class _InterruptingBackend(DistributedLayerwiseOffloadBackend):
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
    _assert_no_dlo_session_leaks(backend, pipeline, prepared)


def test_forward_read_failure_after_commit_tears_down_terminally() -> None:
    pipeline = _Pipeline()
    prepared = _prepared(pipeline)
    backend = DistributedLayerwiseOffloadBackend(
        _config(),
        torch.device("cpu"),
        prepared_weight_session=prepared,
    )
    backend.enable(pipeline)
    assert prepared.active is not None
    prepared.active.fail_unit_id = "block.1"

    with pytest.raises(RuntimeError, match="injected session read failure"):
        pipeline.transformer(torch.tensor(0.0))

    assert not backend.enabled
    assert backend._host_weight_terminal
    assert backend._weight_session is None
    assert backend._weight_session_blocks == {}
    assert backend._session_slot_bindings == [None, None]
    assert prepared.active.closed
    assert not prepared.active.open_units
    assert not prepared.active.active_bindings
    assert all(
        block._hook_registry.get_hook("distributed_layerwise_offload") is None for block in pipeline.transformer.blocks
    )


def test_hook_primary_survives_fail_closed_cleanup_and_reporting_failure(monkeypatch) -> None:
    pipeline = _Pipeline()
    prepared = _prepared(pipeline)
    backend = DistributedLayerwiseOffloadBackend(
        _config(),
        torch.device("cpu"),
        prepared_weight_session=prepared,
    )
    backend.enable(pipeline)
    assert prepared.active is not None

    primary = RuntimeError("primary DLO session read failure")
    cleanup_failure = RuntimeError("injected DLO fail-closed hook removal failure")
    original_open_unit = prepared.active.open_unit
    original_remove = backend_module.remove_distributed_block_hook
    removal_calls = 0
    reporting_calls = 0

    def open_unit_with_failure(request):
        prepared_unit = original_open_unit(request)
        if request.unit_id == "block.1":

            def fail_copy(_destination):
                raise primary

            prepared_unit.copy_into = fail_copy
        return prepared_unit

    def fail_remove_once(block):
        nonlocal removal_calls
        removal_calls += 1
        if removal_calls == 1:
            raise cleanup_failure
        original_remove(block)

    def fail_reporting(*_args, **_kwargs):
        nonlocal reporting_calls
        reporting_calls += 1
        raise SystemExit("injected DLO fail-closed logging failure")

    prepared.active.open_unit = open_unit_with_failure
    monkeypatch.setattr(
        backend_module,
        "remove_distributed_block_hook",
        fail_remove_once,
    )
    monkeypatch.setattr(backend_module.logger, "error", fail_reporting)

    with pytest.raises(RuntimeError, match="primary DLO session read failure") as exc_info:
        pipeline.transformer(torch.tensor(0.0))

    assert exc_info.value is primary
    assert reporting_calls == 1
    assert backend._weight_session_teardown_phase.value == "quiesced"
    assert backend._session_hooked_blocks

    backend.disable()

    _assert_no_dlo_session_leaks(backend, pipeline, prepared)


def test_disable_retries_transient_streamed_binding_release_failure() -> None:
    pipeline = _Pipeline()
    prepared = _prepared(pipeline)
    backend = DistributedLayerwiseOffloadBackend(
        _config(),
        torch.device("cpu"),
        prepared_weight_session=prepared,
    )
    backend.enable(pipeline)
    assert prepared.active is not None
    pipeline.transformer(torch.tensor(0.0))

    state = next(state for state in backend._weight_session_blocks.values() if state.device_binding is not None)
    binding = state.device_binding
    slot = state.slot
    assert binding is not None
    assert slot is not None
    original_release = binding.release
    failure = RuntimeError("injected DLO streamed binding release failure")
    release_calls = 0

    def fail_release_once(target):
        nonlocal release_calls
        release_calls += 1
        if release_calls == 1:
            raise failure
        original_release(target)

    binding.release = fail_release_once

    with pytest.raises(RuntimeError, match="DLO streamed binding release failure") as exc_info:
        backend.disable()

    assert exc_info.value is failure
    assert release_calls == 1
    assert state.device_binding is binding
    assert state.slot == slot
    assert backend._session_slot_bindings[slot] is state
    assert backend._weight_session is prepared.active
    assert backend._weight_session_handle.session is prepared.active
    assert not prepared.active.closed
    assert binding in prepared.active.active_bindings

    backend.disable()

    assert release_calls == 2
    _assert_no_dlo_session_leaks(backend, pipeline, prepared)


def test_disable_retries_transient_session_close_failure() -> None:
    pipeline = _Pipeline()
    prepared = _prepared(pipeline)
    backend = DistributedLayerwiseOffloadBackend(
        _config(),
        torch.device("cpu"),
        prepared_weight_session=prepared,
    )
    backend.enable(pipeline)
    assert prepared.active is not None
    original_close = prepared.active.close
    failure = RuntimeError("injected DLO session close failure")
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
    assert backend._weight_session_handle.session is prepared.active
    assert backend._weight_session_teardown_phase.value == "quiesced"
    assert prepared.active.suspend_count == 1

    backend.disable()

    assert close_calls == 2
    assert prepared.active.suspend_count == 1
    _assert_no_dlo_session_leaks(backend, pipeline, prepared)


def test_disable_sync_failure_retains_all_device_bindings(monkeypatch) -> None:
    pipeline = _Pipeline()
    prepared = _prepared(pipeline)
    backend = DistributedLayerwiseOffloadBackend(
        _config(),
        torch.device("cpu"),
        prepared_weight_session=prepared,
    )
    backend.enable(pipeline)
    pipeline.transformer(torch.tensor(0.0))
    streamed = next(
        state.device_binding for state in backend._weight_session_blocks.values() if state.device_binding is not None
    )
    resident_controller = backend._resident_weight_controller
    assert resident_controller is not None
    resident = next(state.device_binding for state in resident_controller.states if state.device_binding is not None)
    sync_calls = 0

    def fail_once_sync() -> None:
        nonlocal sync_calls
        sync_calls += 1
        if sync_calls == 1:
            raise RuntimeError("DLO teardown sync failed")

    monkeypatch.setattr(backend_module.current_omni_platform, "synchronize", fail_once_sync)

    with pytest.raises(RuntimeError, match="DLO teardown sync failed"):
        backend.disable()

    assert streamed.release_calls == 0
    assert resident.release_calls == 0
    assert backend._weight_session_teardown_phase.value == "active"

    backend.disable()
    assert streamed.release_calls == 1
    assert resident.release_calls == 1
    _assert_no_dlo_session_leaks(backend, pipeline, prepared)


@pytest.mark.parametrize("failure_point", ["hook", "on_demand"])
def test_disable_retries_removal_after_strict_session_suspend(
    monkeypatch,
    failure_point: str,
) -> None:
    pipeline = _Pipeline()
    prepared = _prepared(pipeline)
    backend = DistributedLayerwiseOffloadBackend(
        _config(),
        torch.device("cpu"),
        prepared_weight_session=prepared,
    )
    backend.enable(pipeline)
    assert prepared.active is not None
    live_bindings = tuple(prepared.active.active_bindings)
    assert live_bindings

    failure = RuntimeError(f"injected DLO {failure_point} removal failure")
    handle = None
    if failure_point == "hook":
        original_remove = backend_module.remove_distributed_block_hook
        removal_calls = 0

        def fail_remove_once(block):
            nonlocal removal_calls
            removal_calls += 1
            if removal_calls == 1:
                raise failure
            original_remove(block)

        monkeypatch.setattr(
            backend_module,
            "remove_distributed_block_hook",
            fail_remove_once,
        )
    else:

        class FailOnceHandle:
            def __init__(self) -> None:
                self.remove_calls = 0

            def remove(self) -> None:
                self.remove_calls += 1
                if self.remove_calls == 1:
                    raise failure

        handle = FailOnceHandle()
        backend._on_demand_handles.append(handle)

    with pytest.raises(RuntimeError, match=f"{failure_point} removal failure") as exc_info:
        backend.disable()

    assert exc_info.value is failure
    assert prepared.active.suspended
    assert prepared.active.suspend_count == 1
    assert all(binding.release_calls == 1 for binding in live_bindings)
    assert backend._weight_session_teardown_phase.value == "quiesced"
    if failure_point == "hook":
        assert backend._session_hooked_blocks
    else:
        assert backend._session_hooked_blocks == []
        assert backend._on_demand_handles == [handle]

    backend.disable()

    assert prepared.active.suspend_count == 1
    assert all(binding.release_calls == 1 for binding in live_bindings)
    if handle is not None:
        assert handle.remove_calls == 2
    _assert_no_dlo_session_leaks(backend, pipeline, prepared)


def test_disable_retries_if_suspend_return_is_interrupted_before_phase_update() -> None:
    pipeline = _Pipeline()
    prepared = _prepared(pipeline)
    backend = DistributedLayerwiseOffloadBackend(
        _config(),
        torch.device("cpu"),
        prepared_weight_session=prepared,
    )
    backend.enable(pipeline)
    active = prepared.active
    assert active is not None
    original_suspend = active.suspend
    primary = _InjectedInterruption("injected DLO suspend return interruption")

    def suspend_then_interrupt() -> None:
        original_suspend()
        raise primary

    active.suspend = suspend_then_interrupt
    with pytest.raises(_InjectedInterruption) as exc_info:
        backend.disable()

    assert exc_info.value is primary
    assert active.suspended
    assert active.suspend_count == 1
    assert backend._weight_session_teardown_phase.value == "active"

    active.suspend = original_suspend
    backend.disable()

    assert active.suspend_count == 1
    _assert_no_dlo_session_leaks(backend, pipeline, prepared)


def test_disable_retries_precommit_hook_cleanup_without_active_session(monkeypatch) -> None:
    class PrimaryError(RuntimeError):
        def add_note(self, _note: str) -> None:
            raise SystemExit("injected DLO add_note failure")

    class CleanupError(RuntimeError):
        def __str__(self) -> str:
            raise SystemExit("injected DLO cleanup __str__ failure")

    pipeline = _Pipeline()
    prepared = _prepared(pipeline)
    backend = DistributedLayerwiseOffloadBackend(
        _config(),
        torch.device("cpu"),
        prepared_weight_session=prepared,
    )
    primary = PrimaryError("injected DLO commit failure")
    cleanup_failure = CleanupError("injected DLO precommit hook removal failure")
    original_remove = backend_module.remove_distributed_block_hook
    removal_calls = 0
    reporting_calls = 0

    def fail_commit():
        prepared.commit_count += 1
        raise primary

    def fail_remove_once(block):
        nonlocal removal_calls
        removal_calls += 1
        if removal_calls == 1:
            raise cleanup_failure
        original_remove(block)

    def fail_reporting(*_args, **_kwargs):
        nonlocal reporting_calls
        reporting_calls += 1
        raise SystemExit("injected DLO cleanup logging failure")

    monkeypatch.setattr(prepared, "commit", fail_commit)
    monkeypatch.setattr(
        backend_module,
        "remove_distributed_block_hook",
        fail_remove_once,
    )
    monkeypatch.setattr(backend_module.logger, "error", fail_reporting)

    with pytest.raises(RuntimeError, match="DLO commit failure") as exc_info:
        backend.enable(pipeline)

    assert exc_info.value is primary
    assert prepared.rollback_count == 1
    assert backend._weight_session is None
    assert backend._prepared_weight_session is None
    assert backend._weight_session_teardown_phase.value == "quiesced"
    assert backend._session_hooked_blocks
    assert reporting_calls == 1

    backend.disable()

    assert prepared.rollback_count == 1
    _assert_no_dlo_session_leaks(backend, pipeline, prepared)


def test_stage_resident_read_failure_tears_down_terminally() -> None:
    pipeline = _Pipeline()
    pipeline._offload_plan = OffloadPlan(resident_dit_paths=frozenset({"transformer"}))
    prepared = _prepared(pipeline)
    backend = DistributedLayerwiseOffloadBackend(
        _config(resident_layers=1),
        torch.device("cpu"),
        prepared_weight_session=prepared,
    )
    backend.enable(pipeline)
    assert prepared.active is not None

    failure = RuntimeError("injected stage-resident read failure")
    original_open_unit = prepared.active.open_unit

    def open_unit_with_failure(request):
        prepared_unit = original_open_unit(request)
        if request.unit_id == "block.0":

            def fail_copy(_destination):
                raise failure

            prepared_unit.copy_into = fail_copy
        return prepared_unit

    prepared.active.open_unit = open_unit_with_failure

    with pytest.raises(RuntimeError, match="stage-resident read failure") as exc_info:
        backend.load_resident_layers()

    assert exc_info.value is failure
    _assert_terminal_session_cleanup(backend, pipeline, prepared)


def test_stage_resident_release_failure_tears_down_terminally() -> None:
    pipeline = _Pipeline()
    pipeline._offload_plan = OffloadPlan(resident_dit_paths=frozenset({"transformer"}))
    prepared = _prepared(pipeline)
    backend = DistributedLayerwiseOffloadBackend(
        _config(resident_layers=1),
        torch.device("cpu"),
        prepared_weight_session=prepared,
    )
    backend.enable(pipeline)
    backend.load_resident_layers()

    controller = backend._stage_resident_weight_controller
    assert controller is not None
    binding = controller.states[0].device_binding
    assert binding is not None
    failure = RuntimeError("injected stage-resident release failure")
    original_release = binding.release
    release_calls = 0

    def fail_release_once(target):
        nonlocal release_calls
        release_calls += 1
        if release_calls == 1:
            raise failure
        original_release(target)

    binding.release = fail_release_once

    with pytest.raises(RuntimeError, match="stage-resident release failure") as exc_info:
        backend.offload_resident_layers()

    assert exc_info.value is failure
    assert release_calls == 2
    _assert_terminal_session_cleanup(backend, pipeline, prepared)


def test_session_hard_rejects_allgather_without_taking_ownership() -> None:
    pipeline = _Pipeline()
    prepared = _prepared(pipeline)

    with pytest.raises(ValueError, match="no-AllGather"):
        DistributedLayerwiseOffloadBackend(
            _config(use_allgather=True),
            torch.device("cpu"),
            prepared_weight_session=prepared,
        )

    assert prepared.commit_count == 0
    assert prepared.rollback_count == 0


def test_session_streams_plan_declared_auxiliary_ring(monkeypatch) -> None:
    pipeline = _PlannedPipeline()
    expected = pipeline.transformer(torch.tensor(0.0))
    prepared = _planned_prepared(pipeline)
    monkeypatch.setattr(
        pipeline.transformer.token_refiner,
        "to",
        lambda *_args, **_kwargs: pytest.fail("session-backed auxiliary state must not use Module.to()"),
    )
    backend = DistributedLayerwiseOffloadBackend(
        _config(),
        torch.device("cpu"),
        prepared_weight_session=prepared,
    )

    backend.enable(pipeline)

    assert len(backend._all_hook_groups) == 2
    assert all(hook._uses_weight_session for group in backend._all_hook_groups for hook in group)
    assert {binding.block_path for binding in backend._weight_session_blocks.values()} == {
        "blocks.0",
        "blocks.1",
        "blocks.2",
        "token_refiner.layers.0",
        "token_refiner.layers.1",
    }
    assert torch.equal(pipeline.transformer(torch.tensor(0.0)), expected)

    backend.disable()
    assert prepared.active is not None and prepared.active.closed
