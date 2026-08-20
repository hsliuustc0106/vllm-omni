# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.host_weight import session as session_module
from vllm_omni.diffusion.host_weight.binding import PreparedBindingCommitState
from vllm_omni.diffusion.host_weight.session import (
    BindingStateError,
    DetachMode,
    DeviceBindingLifetime,
    HostCopyMode,
    PreparedWeightAccessSession,
    SessionBusy,
    SessionCapabilities,
    SessionClosed,
    SessionError,
    UnitOutsideSelectedPlan,
    UnitReadRequest,
)
from vllm_omni.diffusion.host_weight.transfer import (
    BindingDestination,
    DtypePlaneSpec,
    ModuleStateKind,
    ModuleUnitBinding,
    PlaneId,
    TargetModulePath,
    TensorBindingSpec,
    TensorPlacement,
    TransferCatalog,
    TransferPlan,
    TransferPlanKind,
    TransferUnitSpec,
    UnitKind,
    compute_exact_coverage_digest,
    compute_transfer_catalog_digest,
)
from vllm_omni.host_weight_runtime import (
    AccessFeature,
    AccessRequirements,
    ArtifactSpec,
    ArtifactTopologyDescriptor,
    BackingKind,
    CapabilityGrant,
    ProducerDescriptor,
    Ready,
    TopologyCoordinate,
    WeightFormatDescriptor,
    create_default_host_weight_runtime,
    derive_weight_format_plan_digest,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _InjectedInterruption(BaseException):
    pass


class _ArtifactRegistrar:
    def __init__(self) -> None:
        self.artifact = None

    def adopt_artifact(self, artifact) -> None:
        if self.artifact is not None:
            raise AssertionError("test artifact registrar already owns an artifact")
        self.artifact = artifact


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _artifact_spec() -> ArtifactSpec:
    format_id = "test-format"
    adapter_abi = "test-format/v1"
    semantic_fingerprint = _digest("format")
    format_recipe_schema_version = 1
    target_module_type_id = "test-target/v1"
    normalized_config = {"dtype": "float32"}
    kernel_identity = {"kernel": "copy"}
    return ArtifactSpec(
        source_fingerprint=_digest("checkpoint"),
        producer=ProducerDescriptor(
            producer_id="test-producer",
            producer_abi="test-producer/v1",
            semantic_fingerprint=_digest("producer"),
        ),
        weight_format=WeightFormatDescriptor(
            format_id=format_id,
            adapter_abi=adapter_abi,
            semantic_fingerprint=semantic_fingerprint,
            format_plan_digest=derive_weight_format_plan_digest(
                format_id=format_id,
                adapter_abi=adapter_abi,
                semantic_fingerprint=semantic_fingerprint,
                format_recipe_schema_version=format_recipe_schema_version,
                target_module_type_id=target_module_type_id,
                normalized_config=normalized_config,
                kernel_identity=kernel_identity,
            ),
            format_recipe_schema_version=format_recipe_schema_version,
            target_module_type_id=target_module_type_id,
            normalized_config=normalized_config,
            kernel_identity=kernel_identity,
        ),
        topology=ArtifactTopologyDescriptor(
            (
                TopologyCoordinate("pp", 1, 0),
                TopologyCoordinate("tp", 1, 0),
            )
        ),
        layout_abi="test-layout/v1",
    )


def _catalog(source: torch.Tensor) -> TransferCatalog:
    binding = TensorBindingSpec(
        tensor_id="block.weight",
        destination=BindingDestination(
            module_path=TargetModulePath("blocks.0"),
            attribute_name="weight",
            state_kind=ModuleStateKind.PARAMETER,
        ),
    )

    def unit(unit_id: str, unit_kind: UnitKind) -> TransferUnitSpec:
        plane_id = PlaneId(f"{unit_id}/plane/float32")
        return TransferUnitSpec(
            unit_id=unit_id,
            unit_kind=unit_kind,
            bindings=(binding,),
            planes=(
                DtypePlaneSpec(
                    plane_id=plane_id,
                    dtype=source.dtype,
                    storage_numel=source.numel(),
                    placements=(
                        TensorPlacement(
                            tensor_id="block.weight",
                            offset_numel=0,
                            logical_shape=tuple(source.shape),
                            physical_stride=tuple(source.stride()),
                            storage_numel=source.numel(),
                        ),
                    ),
                ),
            ),
        )

    component = unit("component.transformer", UnitKind.COMPONENT)
    block = unit("block.0", UnitKind.BLOCK)
    units = (component, block)
    component_unit_ids = (component.unit_id,)
    component_execution = (
        ModuleUnitBinding(
            module_path=TargetModulePath("."),
            unit_id=component.unit_id,
        ),
    )
    component_plan = TransferPlan(
        plan_id="plan.component",
        plan_kind=TransferPlanKind.COMPONENT,
        unit_ids=component_unit_ids,
        execution_bindings=component_execution,
        exact_coverage_digest=compute_exact_coverage_digest(
            component_unit_ids,
            component_execution,
            units,
        ),
    )
    block_unit_ids = (block.unit_id,)
    block_execution = (
        ModuleUnitBinding(
            module_path=TargetModulePath("blocks.0"),
            unit_id=block.unit_id,
        ),
    )
    block_plan = TransferPlan(
        plan_id="plan.blocks_plus_resident",
        plan_kind=TransferPlanKind.BLOCKS_PLUS_RESIDENT,
        unit_ids=block_unit_ids,
        execution_bindings=block_execution,
        exact_coverage_digest=compute_exact_coverage_digest(
            block_unit_ids,
            block_execution,
            units,
        ),
    )
    plans = (component_plan, block_plan)
    return TransferCatalog(
        artifact_compatibility_digest="artifact",
        transfer_catalog_digest=compute_transfer_catalog_digest(units, plans),
        units=units,
        plans=plans,
    )


@dataclass
class _Controller:
    closed_with: DetachMode | None = None
    close_count: int = 0

    def bind_device(self, _unit_id, _buffers):  # pragma: no cover - rejected first
        raise AssertionError("test must reject invalid device buffers before binding")

    def restore_cpu(self) -> None:  # pragma: no cover - v1 never calls this
        raise AssertionError

    def close(self, mode: DetachMode) -> None:
        self.close_count += 1
        self.closed_with = mode


class _PreparedBinding:
    def __init__(self) -> None:
        self.controller = _Controller()
        self.retained_controller: _Controller | None = None
        self.commit_state = PreparedBindingCommitState.PREPARED
        self.commit_count = 0
        self.rollback_count = 0

    def commit(self) -> _Controller:
        self.commit_count += 1
        self.retained_controller = self.controller
        self.commit_state = PreparedBindingCommitState.RETIREMENT_COMMITTED
        return self.controller

    def rollback(self) -> None:
        self.rollback_count += 1
        if self.commit_state is PreparedBindingCommitState.PROVISIONAL_CONTROLLER:
            assert self.retained_controller is not None
            self.retained_controller.close(DetachMode.TERMINAL)
            self.retained_controller = None
        self.commit_state = PreparedBindingCommitState.ROLLED_BACK


def _prepared():
    source = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    runtime = create_default_host_weight_runtime(None)
    grant = runtime.negotiate(
        AccessRequirements(
            required_features=frozenset({AccessFeature.COMPLETE_TENSOR_READ}),
            accepted_backings=frozenset({BackingKind.LOADED_TENSOR}),
        )
    )
    assert isinstance(grant, CapabilityGrant)
    registrar = _ArtifactRegistrar()
    outcome = runtime.resolve_loaded(
        _artifact_spec(),
        grant,
        {"block.weight": source},
        registrar,
    )
    assert isinstance(outcome, Ready)
    assert registrar.artifact is outcome.artifact
    binding = _PreparedBinding()
    catalog = _catalog(source)
    prepared = PreparedWeightAccessSession(
        pipeline=object(),
        catalog=catalog,
        transfer_plan=catalog.plan("plan.blocks_plus_resident"),
        capabilities=SessionCapabilities(
            runtime_instance_id=outcome.access.runtime_instance_id,
            capability_grant_id=outcome.access.grant_id,
            access_features=frozenset({AccessFeature.COMPLETE_TENSOR_READ}),
            selected_transfer_plan_id="plan.blocks_plus_resident",
            selected_transfer_plan_kind=TransferPlanKind.BLOCKS_PLUS_RESIDENT,
            unit_kinds=frozenset({UnitKind.BLOCK}),
            weight_format_id="test-format",
            backing_kind=BackingKind.LOADED_TENSOR,
            provider_id=outcome.access.provider_id,
            provider_abi=outcome.access.provider_abi,
            host_copy_mode=HostCopyMode.SYNCHRONOUS,
        ),
        artifact=outcome.artifact,
        binding=binding,
        runtime=runtime,
    )
    return source, binding, prepared


def test_session_copies_exactly_once_and_requires_quiescent_close() -> None:
    source, binding, prepared = _prepared()
    session = prepared.commit()
    assert binding.commit_count == 1

    unit = session.open_unit(UnitReadRequest("block.0"))
    destination = {PlaneId("block.0/plane/float32"): torch.empty(12)}
    unit.copy_into(destination)
    torch.testing.assert_close(destination[next(iter(destination))].view(3, 4), source)
    with pytest.raises(BindingStateError, match="already copied"):
        unit.copy_into(destination)
    with pytest.raises(SessionBusy, match="weight reads"):
        session.suspend()

    unit.close()
    unit.close()
    session.suspend()
    with pytest.raises(SessionClosed, match="suspended"):
        session.open_unit(UnitReadRequest("block.0"))
    session.close(DetachMode.TERMINAL)
    assert binding.controller.closed_with is DetachMode.TERMINAL
    session.close(DetachMode.TERMINAL)


def test_active_session_reports_lock_consistent_idle_diagnostics() -> None:
    _source, _binding, prepared = _prepared()
    session = prepared.commit()

    assert session.idle_state() == {
        "state": "active",
        "outstanding_units": 0,
        "bindings": 0,
        "resident_bindings": 0,
        "total_bindings": 0,
    }
    unit = session.open_unit(UnitReadRequest("block.0"))
    assert session.idle_state()["outstanding_units"] == 1
    unit.close()
    session.suspend()
    assert session.idle_state() == {
        "state": "suspended",
        "outstanding_units": 0,
        "bindings": 0,
        "resident_bindings": 0,
        "total_bindings": 0,
    }
    session.close(DetachMode.TERMINAL)


def test_idle_diagnostics_separate_resident_from_transient_bindings() -> None:
    _source, _binding, prepared = _prepared()
    session = prepared.commit()

    class _DiagnosticBinding:
        def __init__(self, unit_id: str, lifetime: DeviceBindingLifetime) -> None:
            self._unit_id = unit_id
            self._lifetime = lifetime

    resident = _DiagnosticBinding("resident", DeviceBindingLifetime.RESIDENT)
    stationary_block = _DiagnosticBinding("stationary-block", DeviceBindingLifetime.RESIDENT)
    transient = _DiagnosticBinding("block", DeviceBindingLifetime.TRANSIENT)
    session._device_bindings.update({resident, stationary_block, transient})  # type: ignore[arg-type]

    assert session.idle_state() == {
        "state": "active",
        "outstanding_units": 0,
        "bindings": 1,
        "resident_bindings": 2,
        "total_bindings": 3,
    }

    session._device_bindings.clear()
    session.suspend()
    session.close(DetachMode.TERMINAL)


def test_prepared_and_active_sessions_enforce_selected_plan_scope(monkeypatch) -> None:
    _source, _binding, prepared = _prepared()

    assert prepared.transfer_plan.plan_id == "plan.blocks_plus_resident"
    assert prepared.unit_spec("block.0").unit_kind is UnitKind.BLOCK
    with pytest.raises(UnitOutsideSelectedPlan, match="outside selected plan"):
        prepared.unit_spec("component.transformer")

    session = prepared.commit()
    assert session.transfer_plan is prepared.transfer_plan
    assert session.unit_spec("block.0").unit_kind is UnitKind.BLOCK
    artifact = session._artifact
    assert artifact is not None
    open_calls = 0
    original_open = artifact.open

    def record_open(*args, **kwargs):
        nonlocal open_calls
        open_calls += 1
        return original_open(*args, **kwargs)

    monkeypatch.setattr(artifact, "open", record_open)
    with pytest.raises(UnitOutsideSelectedPlan, match="component.transformer"):
        session.open_unit(UnitReadRequest("component.transformer"))
    with pytest.raises(UnitOutsideSelectedPlan, match="component.transformer"):
        session.bind_device("component.transformer", {})
    assert open_calls == 0

    session.suspend()
    session.close(DetachMode.TERMINAL)


def test_session_rejects_meta_as_a_device_plane() -> None:
    _source, _binding, prepared = _prepared()
    session = prepared.commit()

    with pytest.raises(SessionError, match="physical non-CPU"):
        session.bind_device(
            "block.0",
            {
                PlaneId("block.0/plane/float32"): torch.empty(
                    12,
                    device="meta",
                )
            },
        )

    session.suspend()
    session.close(DetachMode.TERMINAL)


def test_prepared_rollback_releases_artifact_without_commit() -> None:
    _source, binding, prepared = _prepared()

    prepared.rollback()

    assert binding.rollback_count == 1
    with pytest.raises(SessionClosed, match="already closed"):
        prepared.commit()


def test_prepared_commit_constructs_active_shell_before_irreversible_binding(
    monkeypatch,
) -> None:
    _source, binding, prepared = _prepared()
    primary = MemoryError("injected active-session construction failure")

    def fail_construction(**_kwargs):
        raise primary

    monkeypatch.setattr(session_module, "WeightAccessSession", fail_construction)

    with pytest.raises(MemoryError, match="active-session construction failure") as exc_info:
        prepared.commit()

    assert exc_info.value is primary
    assert binding.commit_count == 0
    prepared.rollback()
    assert binding.rollback_count == 1


def test_prepared_commit_retains_active_owner_if_binding_return_is_interrupted(
    monkeypatch,
) -> None:
    _source, binding, prepared = _prepared()
    original_commit = binding.commit
    primary = _InjectedInterruption("injected post-binding-commit interruption")

    def commit_then_interrupt():
        original_commit()
        raise primary

    monkeypatch.setattr(binding, "commit", commit_then_interrupt)

    with pytest.raises(_InjectedInterruption) as exc_info:
        prepared.commit()

    assert exc_info.value is primary
    assert prepared._state == "committed"
    assert prepared._committed_session is not None
    prepared.rollback()
    assert binding.controller.closed_with is DetachMode.TERMINAL
    assert prepared._state == "closed"


def test_prepared_commit_uses_marker_not_provisional_controller_for_recovery(
    monkeypatch,
) -> None:
    _source, binding, prepared = _prepared()
    primary = _InjectedInterruption("injected pre-retirement-marker interruption")

    def retain_provisional_then_interrupt():
        binding.commit_count += 1
        binding.retained_controller = binding.controller
        binding.commit_state = PreparedBindingCommitState.PROVISIONAL_CONTROLLER
        raise primary

    monkeypatch.setattr(binding, "commit", retain_provisional_then_interrupt)

    with pytest.raises(_InjectedInterruption) as exc_info:
        prepared.commit()

    assert exc_info.value is primary
    assert prepared._state == "prepared"
    assert prepared._committed_session is None
    assert binding.retained_controller is binding.controller

    prepared.rollback()
    assert binding.rollback_count == 1
    assert binding.controller.closed_with is DetachMode.TERMINAL
    assert binding.controller.close_count == 1
    assert prepared._state == "closed"


def test_prepared_commit_retains_controller_if_active_assignment_fails(
    monkeypatch,
) -> None:
    original_active_type = session_module.WeightAccessSession
    primary = MemoryError("injected active controller assignment failure")

    class _FailControllerAssignment(original_active_type):
        def __setattr__(self, name, value) -> None:
            if (
                name == "_controller"
                and value is not None
                and not getattr(
                    self,
                    "_controller_assignment_failed",
                    False,
                )
            ):
                object.__setattr__(self, "_controller_assignment_failed", True)
                raise primary
            super().__setattr__(name, value)

    monkeypatch.setattr(session_module, "WeightAccessSession", _FailControllerAssignment)
    _source, binding, prepared = _prepared()

    with pytest.raises(MemoryError, match="active controller assignment failure") as exc_info:
        prepared.commit()

    assert exc_info.value is primary
    assert prepared._state == "committed"
    assert prepared._committed_session is not None
    assert prepared._committed_session._controller is binding.controller
    assert binding.retained_controller is binding.controller

    prepared.rollback()
    assert binding.controller.closed_with is DetachMode.TERMINAL
    assert binding.controller.close_count == 1
    assert prepared._state == "closed"


def test_open_unit_return_interruption_is_drained_by_suspend(monkeypatch) -> None:
    _source, _binding, prepared = _prepared()
    session = prepared.commit()
    primary = _InjectedInterruption("injected open-unit return interruption")
    captured = None

    def interrupt_return(unit):
        nonlocal captured
        captured = unit
        raise primary

    monkeypatch.setattr(session, "_return_open_unit", interrupt_return)

    with pytest.raises(_InjectedInterruption) as exc_info:
        session.open_unit(UnitReadRequest("block.0"))

    assert exc_info.value is primary
    assert captured is not None
    registration = session._unit_registrations[id(captured)]
    assert registration.unit is captured
    assert registration.state is session_module._UnitHandleState.UNPUBLISHED

    session.suspend()
    assert not session._unit_registrations
    session.close(DetachMode.TERMINAL)


def test_open_unit_registration_precedes_artifact_view_acquisition(monkeypatch) -> None:
    _source, _binding, prepared = _prepared()
    session = prepared.commit()
    artifact = session._artifact
    assert artifact is not None
    primary = MemoryError("injected exact-unit registration failure")
    open_calls = 0
    original_open = artifact.open

    def record_open(*args, **kwargs):
        nonlocal open_calls
        open_calls += 1
        return original_open(*args, **kwargs)

    original_register = session._register_unpublished_unit

    def register_then_interrupt(registration) -> None:
        original_register(registration)
        stored = session._unit_registrations[id(registration.unit)]
        assert stored is registration
        assert stored.state is session_module._UnitHandleState.UNPUBLISHED
        raise primary

    monkeypatch.setattr(artifact, "open", record_open)
    monkeypatch.setattr(session, "_register_unpublished_unit", register_then_interrupt)

    with pytest.raises(MemoryError, match="exact-unit registration failure") as exc_info:
        session.open_unit(UnitReadRequest("block.0"))

    assert exc_info.value is primary
    assert open_calls == 0
    assert not session._unit_registrations
    session.suspend()
    session.close(DetachMode.TERMINAL)


def test_open_unit_failed_cleanup_retains_exact_unpublished_owner(monkeypatch) -> None:
    _source, _binding, prepared = _prepared()
    session = prepared.commit()
    primary = _InjectedInterruption("injected post-view-open interruption")
    cleanup_failure = RuntimeError("injected unpublished-view close failure")
    original_open_view = session_module.PreparedWeightUnit._open_view
    captured = None
    close_calls = 0

    def open_then_interrupt(unit) -> None:
        nonlocal captured
        captured = unit
        original_open_view(unit)
        assert unit._view is not None
        original_close = unit._view.close

        def fail_close_once() -> None:
            nonlocal close_calls
            close_calls += 1
            if close_calls == 1:
                raise cleanup_failure
            original_close()

        monkeypatch.setattr(unit._view, "close", fail_close_once)
        raise primary

    monkeypatch.setattr(session_module.PreparedWeightUnit, "_open_view", open_then_interrupt)

    with pytest.raises(_InjectedInterruption) as exc_info:
        session.open_unit(UnitReadRequest("block.0"))

    assert exc_info.value is primary
    assert captured is not None
    registration = session._unit_registrations[id(captured)]
    assert registration.unit is captured
    assert registration.state is session_module._UnitHandleState.UNPUBLISHED
    assert close_calls == 1
    assert any("unpublished weight read" in note for note in primary.__notes__)

    session.suspend()
    assert close_calls == 2
    assert not session._unit_registrations
    session.close(DetachMode.TERMINAL)


def test_unit_registry_uses_exact_identity_for_equal_handles(monkeypatch) -> None:
    original_unit_type = session_module.PreparedWeightUnit

    class _EqualPreparedWeightUnit(original_unit_type):
        def __hash__(self) -> int:
            return 0

        def __eq__(self, _other: object) -> bool:
            return True

    monkeypatch.setattr(session_module, "PreparedWeightUnit", _EqualPreparedWeightUnit)
    _source, _binding, prepared = _prepared()
    session = prepared.commit()

    first = session.open_unit(UnitReadRequest("block.0"))
    second = session.open_unit(UnitReadRequest("block.0"))
    assert first == second
    assert len(session._unit_registrations) == 2
    assert session._unit_registrations[id(first)].unit is first
    assert session._unit_registrations[id(second)].unit is second

    first.publish()
    first.publish()
    assert session._unit_registrations[id(first)].state is session_module._UnitHandleState.PUBLISHED
    assert session._unit_registrations[id(second)].state is session_module._UnitHandleState.UNPUBLISHED
    with pytest.raises(SessionBusy, match="weight reads"):
        session.suspend()
    assert second._closed
    assert not first._closed
    assert len(session._unit_registrations) == 1

    first.close()
    session.suspend()
    session.close(DetachMode.TERMINAL)


def test_explicit_unit_publication_preserves_live_handle_busy_contract() -> None:
    _source, _binding, prepared = _prepared()
    session = prepared.commit()
    unit = session.open_unit(UnitReadRequest("block.0"))

    registration = session._unit_registrations[id(unit)]
    assert registration.unit is unit
    assert registration.state is session_module._UnitHandleState.UNPUBLISHED
    unit.publish()
    assert registration.state is session_module._UnitHandleState.PUBLISHED
    unit.publish()
    assert registration.state is session_module._UnitHandleState.PUBLISHED
    with pytest.raises(SessionBusy, match="weight reads"):
        session.suspend()

    unit.close()
    session.suspend()
    session.close(DetachMode.TERMINAL)


def test_suspend_is_idempotent_after_session_transition() -> None:
    _source, binding, prepared = _prepared()
    session = prepared.commit()

    session.suspend()
    session.suspend()
    session.close(DetachMode.TERMINAL)

    assert binding.controller.close_count == 1


@pytest.mark.parametrize("failure_point", ["construction", "registration"])
def test_device_binding_wrapper_precedes_controller_publication(
    monkeypatch,
    failure_point,
) -> None:
    _source, binding, prepared = _prepared()
    session = prepared.commit()
    primary = MemoryError(f"injected wrapper {failure_point} failure")
    monkeypatch.setattr(session_module, "validate_plane_buffers", lambda *_args, **_kwargs: None)
    buffers = {
        PlaneId("block.0/plane/float32"): SimpleNamespace(
            device=SimpleNamespace(type="cuda"),
            is_meta=False,
        )
    }

    if failure_point == "construction":
        monkeypatch.setattr(
            session_module,
            "DeviceBinding",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(primary),
        )
    else:

        class _FailingRegistrationSet(set):
            def add(self, _value) -> None:
                raise primary

        session._pending_device_bindings = _FailingRegistrationSet()

    with pytest.raises(MemoryError, match=f"wrapper {failure_point} failure") as exc_info:
        session.bind_device("block.0", buffers)  # type: ignore[arg-type]

    assert exc_info.value is primary
    session.suspend()
    session.close(DetachMode.TERMINAL)


def test_device_binding_attach_failure_retains_failed_release_for_suspend_retry(
    monkeypatch,
) -> None:
    _source, binding, prepared = _prepared()
    session = prepared.commit()
    primary = MemoryError("injected wrapper delegate-attach failure")

    class _Delegate:
        def __init__(self) -> None:
            self.release_calls = 0

        def release(self, _target) -> None:
            self.release_calls += 1
            if self.release_calls == 1:
                raise RuntimeError("injected unpublished delegate release failure")

    delegate = _Delegate()
    controller_active = False

    def bind_device(_unit_id, _buffers):
        nonlocal controller_active
        controller_active = True
        return delegate

    def release_device(_unit_id, target) -> bool:
        nonlocal controller_active
        if not controller_active:
            return False
        delegate.release(target)
        controller_active = False
        return True

    monkeypatch.setattr(binding.controller, "bind_device", bind_device)
    monkeypatch.setattr(binding.controller, "release_device", release_device, raising=False)
    monkeypatch.setattr(session_module, "validate_plane_buffers", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        session_module.DeviceBinding,
        "_attach_delegate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(primary),
    )
    buffers = {
        PlaneId("block.0/plane/float32"): SimpleNamespace(
            device=SimpleNamespace(type="cuda"),
            is_meta=False,
        )
    }

    with pytest.raises(MemoryError, match="delegate-attach failure") as exc_info:
        session.bind_device("block.0", buffers)  # type: ignore[arg-type]

    assert exc_info.value is primary
    assert controller_active
    assert len(session._pending_device_bindings) == 1
    assert any("unpublished" in note for note in primary.__notes__)

    session.suspend()
    assert delegate.release_calls == 2
    assert not controller_active
    assert not session._device_bindings
    session.close(DetachMode.TERMINAL)


def test_device_binding_return_interruption_is_drained_by_suspend(monkeypatch) -> None:
    _source, binding, prepared = _prepared()
    session = prepared.commit()
    primary = _InjectedInterruption("injected device-binding return interruption")
    controller_active = False

    class _Delegate:
        def __init__(self) -> None:
            self.release_calls = 0

        def release(self, _target) -> None:
            nonlocal controller_active
            self.release_calls += 1
            controller_active = False

    delegate = _Delegate()

    def bind_device(_unit_id, _buffers):
        nonlocal controller_active
        controller_active = True
        return delegate

    def release_device(_unit_id, target) -> bool:
        nonlocal controller_active
        if not controller_active:
            return False
        delegate.release(target)
        controller_active = False
        return True

    def interrupt_return(_device_binding):
        raise primary

    monkeypatch.setattr(binding.controller, "bind_device", bind_device)
    monkeypatch.setattr(binding.controller, "release_device", release_device, raising=False)
    monkeypatch.setattr(session_module, "validate_plane_buffers", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(session, "_return_device_binding", interrupt_return)
    buffers = {
        PlaneId("block.0/plane/float32"): SimpleNamespace(
            device=SimpleNamespace(type="cuda"),
            is_meta=False,
        )
    }

    with pytest.raises(_InjectedInterruption) as exc_info:
        session.bind_device("block.0", buffers)  # type: ignore[arg-type]

    assert exc_info.value is primary
    assert controller_active
    assert len(session._pending_device_bindings) == 1
    assert len(session._device_bindings) == 1

    session.suspend()
    assert delegate.release_calls == 1
    assert not controller_active
    assert not session._pending_device_bindings
    assert not session._device_bindings
    session.close(DetachMode.TERMINAL)


def test_explicit_binding_publication_preserves_live_handle_busy_contract(
    monkeypatch,
) -> None:
    _source, binding, prepared = _prepared()
    session = prepared.commit()

    class _Delegate:
        def __init__(self) -> None:
            self.release_calls = 0

        def release(self, _target) -> None:
            self.release_calls += 1

    delegate = _Delegate()
    monkeypatch.setattr(binding.controller, "bind_device", lambda *_args: delegate)
    monkeypatch.setattr(session_module, "validate_plane_buffers", lambda *_args, **_kwargs: None)
    buffers = {
        PlaneId("block.0/plane/float32"): SimpleNamespace(
            device=SimpleNamespace(type="cuda"),
            is_meta=False,
        )
    }

    device_binding = session.bind_device(  # type: ignore[arg-type]
        "block.0",
        buffers,
        lifetime=DeviceBindingLifetime.RESIDENT,
    )
    assert device_binding in session._pending_device_bindings
    device_binding.publish()
    assert device_binding not in session._pending_device_bindings
    assert session.idle_state() == {
        "state": "active",
        "outstanding_units": 0,
        "bindings": 0,
        "resident_bindings": 1,
        "total_bindings": 1,
    }
    with pytest.raises(SessionBusy, match="device bindings"):
        session.suspend()

    device_binding.release(session_module.ReleaseTarget.PLACEHOLDER)
    assert delegate.release_calls == 1
    session.suspend()
    session.close(DetachMode.TERMINAL)


def test_prepared_unit_close_retries_view_before_releasing_owner(monkeypatch) -> None:
    _source, _binding, prepared = _prepared()
    session = prepared.commit()
    unit = session.open_unit(UnitReadRequest("block.0"))
    assert unit._view is not None
    original_close = unit._view.close
    close_calls = 0
    failure = RuntimeError("injected view close failure")

    def fail_once() -> None:
        nonlocal close_calls
        close_calls += 1
        if close_calls == 1:
            raise failure
        original_close()

    monkeypatch.setattr(unit._view, "close", fail_once)

    with pytest.raises(RuntimeError, match="injected view close failure") as exc_info:
        unit.close()
    assert exc_info.value is failure
    with pytest.raises(SessionBusy, match="weight reads"):
        session.suspend()

    unit.close()
    unit.close()
    assert close_calls == 2
    session.suspend()
    session.close(DetachMode.TERMINAL)


def test_prepared_rollback_retries_only_unfinished_resources(monkeypatch) -> None:
    _source, binding, prepared = _prepared()
    artifact = prepared._artifact
    runtime = prepared._runtime
    assert artifact is not None and runtime is not None
    original_binding_rollback = binding.rollback
    original_artifact_close = artifact.close
    original_runtime_close = runtime.close
    calls = {"binding": 0, "artifact": 0, "runtime": 0}

    def binding_rollback() -> None:
        calls["binding"] += 1
        if calls["binding"] == 1:
            raise RuntimeError("binding rollback failed")
        original_binding_rollback()

    def artifact_close() -> None:
        calls["artifact"] += 1
        if calls["artifact"] == 1:
            raise RuntimeError("artifact close failed")
        original_artifact_close()

    def runtime_close() -> None:
        calls["runtime"] += 1
        if calls["runtime"] == 1:
            raise RuntimeError("runtime close failed")
        original_runtime_close()

    monkeypatch.setattr(binding, "rollback", binding_rollback)
    monkeypatch.setattr(artifact, "close", artifact_close)
    monkeypatch.setattr(runtime, "close", runtime_close)

    with pytest.raises(RuntimeError, match="binding rollback failed"):
        prepared.rollback()
    assert calls == {"binding": 1, "artifact": 0, "runtime": 0}
    with pytest.raises(RuntimeError, match="artifact close failed"):
        prepared.rollback()
    assert calls == {"binding": 2, "artifact": 1, "runtime": 0}
    with pytest.raises(RuntimeError, match="runtime close failed"):
        prepared.rollback()
    assert calls == {"binding": 2, "artifact": 2, "runtime": 1}

    prepared.rollback()
    prepared.rollback()
    assert calls == {"binding": 2, "artifact": 2, "runtime": 2}
    with pytest.raises(SessionClosed, match="already closed"):
        prepared.commit()


def test_prepared_terminal_rollback_does_not_reattach_retired_controller(monkeypatch) -> None:
    _source, binding, prepared = _prepared()
    active = prepared.commit()
    artifact = active._artifact
    assert artifact is not None
    original_artifact_close = artifact.close
    artifact_close_calls = 0

    def fail_artifact_close_once() -> None:
        nonlocal artifact_close_calls
        artifact_close_calls += 1
        if artifact_close_calls == 1:
            raise RuntimeError("injected artifact close failure")
        original_artifact_close()

    monkeypatch.setattr(artifact, "close", fail_artifact_close_once)

    with pytest.raises(RuntimeError, match="artifact close failure"):
        prepared.rollback()
    assert binding.controller.close_count == 1
    assert active._controller is None
    assert active._state == "closing"

    prepared.rollback()
    assert binding.controller.close_count == 1
    assert artifact_close_calls == 2
    assert active._state == "closed"
    assert prepared._state == "closed"


def test_active_session_close_retries_only_unfinished_resources(monkeypatch) -> None:
    _source, binding, prepared = _prepared()
    session = prepared.commit()
    artifact = session._artifact
    runtime = session._runtime
    assert artifact is not None and runtime is not None
    original_controller_close = binding.controller.close
    original_artifact_close = artifact.close
    original_runtime_close = runtime.close
    calls = {"controller": 0, "artifact": 0, "runtime": 0}

    def controller_close(mode: DetachMode) -> None:
        calls["controller"] += 1
        if calls["controller"] == 1:
            raise RuntimeError("controller close failed")
        original_controller_close(mode)

    def artifact_close() -> None:
        calls["artifact"] += 1
        if calls["artifact"] == 1:
            raise RuntimeError("artifact close failed")
        original_artifact_close()

    def runtime_close() -> None:
        calls["runtime"] += 1
        if calls["runtime"] == 1:
            raise RuntimeError("runtime close failed")
        original_runtime_close()

    monkeypatch.setattr(binding.controller, "close", controller_close)
    monkeypatch.setattr(artifact, "close", artifact_close)
    monkeypatch.setattr(runtime, "close", runtime_close)
    session.suspend()

    with pytest.raises(RuntimeError, match="controller close failed"):
        session.close(DetachMode.TERMINAL)
    assert calls == {"controller": 1, "artifact": 0, "runtime": 0}
    with pytest.raises(RuntimeError, match="artifact close failed"):
        session.close(DetachMode.TERMINAL)
    assert calls == {"controller": 2, "artifact": 1, "runtime": 0}
    with pytest.raises(RuntimeError, match="runtime close failed"):
        session.close(DetachMode.TERMINAL)
    assert calls == {"controller": 2, "artifact": 2, "runtime": 1}

    session.close(DetachMode.TERMINAL)
    session.close(DetachMode.TERMINAL)
    assert calls == {"controller": 2, "artifact": 2, "runtime": 2}
