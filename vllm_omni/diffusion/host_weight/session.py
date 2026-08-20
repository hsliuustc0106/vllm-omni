# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transactional offloader-facing Host Weight Runtime session.

The session is the only HWR object consumed by an offload backend.  It keeps
artifact reads, module binding, and lifetime accounting together while leaving
slot allocation, H2D, events, and collectives with the backend.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

import torch

from vllm_omni.host_weight_runtime import (
    AccessFeature,
    AccessRequirements,
    BackingKind,
    HostCopyMode,
    HostWeightRuntime,
    ResolvedArtifact,
    TensorSelection,
)

from .binding import PreparedBindingCommitState
from .transfer import (
    PlaneId,
    TransferCatalog,
    TransferPlan,
    TransferPlanError,
    TransferPlanKind,
    TransferUnitSpec,
    UnitKind,
    tensor_views_for_unit,
    validate_plane_buffers,
)


class SessionError(RuntimeError):
    pass


class SessionClosed(SessionError):  # noqa: N818 - public lifecycle state name
    pass


class SessionBusy(SessionError):  # noqa: N818 - public lifecycle state name
    pass


class BindingStateError(SessionError):
    pass


class UnitOutsideSelectedPlan(SessionError):  # noqa: N818 - public contract name
    pass


@dataclass(frozen=True)
class SessionRequirements:
    access: AccessRequirements
    required_transfer_plan_kind: TransferPlanKind
    required_weight_format_id: str
    host_copy_mode: HostCopyMode

    def __post_init__(self) -> None:
        try:
            object.__setattr__(
                self,
                "required_transfer_plan_kind",
                TransferPlanKind(self.required_transfer_plan_kind),
            )
        except ValueError as exc:
            raise SessionError(f"unknown required transfer plan kind {self.required_transfer_plan_kind!r}") from exc
        if not self.required_weight_format_id:
            raise SessionError("required weight format ID must not be empty")


@dataclass(frozen=True)
class SessionCapabilities:
    runtime_instance_id: str
    capability_grant_id: str
    access_features: frozenset[AccessFeature]
    selected_transfer_plan_id: str
    selected_transfer_plan_kind: TransferPlanKind
    unit_kinds: frozenset[UnitKind]
    weight_format_id: str
    backing_kind: BackingKind
    provider_id: str
    provider_abi: str
    host_copy_mode: HostCopyMode

    def __post_init__(self) -> None:
        if not self.selected_transfer_plan_id:
            raise SessionError("selected transfer plan ID must not be empty")
        try:
            object.__setattr__(
                self,
                "selected_transfer_plan_kind",
                TransferPlanKind(self.selected_transfer_plan_kind),
            )
        except ValueError as exc:
            raise SessionError(f"unknown selected transfer plan kind {self.selected_transfer_plan_kind!r}") from exc


class DetachMode(str, Enum):
    RESTORE_CPU = "restore_cpu"
    TERMINAL = "terminal"


class ReleaseTarget(str, Enum):
    PLACEHOLDER = "placeholder"


class DeviceBindingLifetime(str, Enum):
    """Execution lifetime of a device binding, independent of unit layout."""

    TRANSIENT = "transient"
    RESIDENT = "resident"


HostPlaneBuffers = Mapping[PlaneId, torch.Tensor]
DevicePlaneBuffers = Mapping[PlaneId, torch.Tensor]


@dataclass(frozen=True)
class UnitReadRequest:
    unit_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.unit_id, str) or not self.unit_id:
            raise SessionError("unit read request requires a non-empty unit ID")


def _select_transfer_plan(
    catalog: TransferCatalog,
    capabilities: SessionCapabilities,
) -> TransferPlan:
    try:
        plan = catalog.plan(capabilities.selected_transfer_plan_id)
    except TransferPlanError as exc:
        raise SessionError(f"selected transfer plan {capabilities.selected_transfer_plan_id!r} is unavailable") from exc
    if plan.plan_kind is not capabilities.selected_transfer_plan_kind:
        raise SessionError(
            f"selected transfer plan {plan.plan_id!r} has kind {plan.plan_kind.value!r}, not "
            f"{capabilities.selected_transfer_plan_kind.value!r}"
        )
    derived_kinds = frozenset(catalog.unit(unit_id).unit_kind for unit_id in plan.unit_ids)
    if capabilities.unit_kinds != derived_kinds:
        raise SessionError(f"session capability unit kinds do not match selected transfer plan {plan.plan_id!r}")
    return plan


def _plan_unit_spec(
    catalog: TransferCatalog,
    plan: TransferPlan,
    unit_id: str,
) -> TransferUnitSpec:
    if unit_id in plan.unit_ids:
        return catalog.unit(unit_id)
    # Preserve the catalog's typed unknown-unit error, but distinguish a real
    # catalog unit that this session has no authority to access.
    catalog.unit(unit_id)
    raise UnitOutsideSelectedPlan(f"transfer unit {unit_id!r} is outside selected plan {plan.plan_id!r}")


class _PreparedBinding(Protocol):
    @property
    def commit_state(self) -> PreparedBindingCommitState: ...

    @property
    def retained_controller(self) -> Any | None: ...

    def commit(self) -> Any: ...

    def rollback(self) -> None: ...


class _BindingController(Protocol):
    def bind_device(
        self,
        unit_id: str,
        buffers: DevicePlaneBuffers,
    ) -> Any: ...

    def restore_cpu(self) -> None: ...

    def release_device(self, unit_id: str, target: object) -> bool: ...

    def close(self, mode: DetachMode) -> None: ...


def _add_cleanup_note(
    primary_error: BaseException,
    action: str,
    cleanup_error: BaseException,
) -> None:
    try:
        detail = str(cleanup_error)
    except BaseException:
        detail = f"<{type(cleanup_error).__name__} detail unavailable>"
    try:
        primary_error.add_note(f"{action} also failed: {type(cleanup_error).__name__}: {detail}")
    except BaseException:
        pass


class _UnitHandleState(str, Enum):
    UNPUBLISHED = "unpublished"
    PUBLISHED = "published"


@dataclass
class _UnitRegistration:
    """Exact-identity session ownership for one prepared read handle."""

    unit: PreparedWeightUnit
    state: _UnitHandleState


class PreparedWeightUnit:
    """One lifetime-safe, exactly-once synchronous artifact read."""

    def __init__(self, owner: WeightAccessSession, layout: TransferUnitSpec) -> None:
        self._owner = owner
        self._layout = layout
        self._view: Any | None = None
        self._copied = False
        self._view_closed = False
        self._owner_released = False
        self._closed = False
        self._lock = threading.RLock()

    def _open_view(self) -> None:
        """Acquire the artifact view after session ownership is registered."""

        artifact = self._owner._artifact
        assert artifact is not None
        self._view = artifact.open(TensorSelection(self._layout.tensor_ids))

    @property
    def layout(self) -> TransferUnitSpec:
        self.publish()
        return self._layout

    def publish(self) -> None:
        """Acknowledge that the caller retained this returned read handle."""

        self._owner._publish_unit(self)

    def copy_into(self, destination: HostPlaneBuffers) -> None:
        self.publish()
        with self._lock:
            if self._closed or self._view_closed:
                raise SessionClosed(f"weight unit {self._layout.unit_id!r} is closed")
            if self._copied:
                raise BindingStateError(f"weight unit {self._layout.unit_id!r} was already copied")
            assert self._view is not None
            views = tensor_views_for_unit(
                self._layout,
                destination,
                device_type="cpu",
            )
            self._view.copy_into(views)
            self._copied = True

    def close(self) -> None:
        self._close(publish=True)

    def _close(self, *, publish: bool) -> None:
        if publish:
            self.publish()
        with self._lock:
            if self._closed:
                return
            if not self._view_closed and self._view is not None:
                self._view.close()
                self._view_closed = True
            if not self._owner_released:
                self._owner._release_unit(self)
                self._owner_released = True
            self._closed = True

    def __enter__(self) -> PreparedWeightUnit:
        self.publish()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class DeviceBinding:
    def __init__(
        self,
        owner: WeightAccessSession,
        delegate: Any | None = None,
        *,
        controller: _BindingController | None = None,
        unit_id: str | None = None,
        lifetime: DeviceBindingLifetime = DeviceBindingLifetime.TRANSIENT,
    ) -> None:
        self._owner = owner
        self._delegate = delegate
        self._controller = controller
        self._unit_id = unit_id
        self._lifetime = DeviceBindingLifetime(lifetime)
        self._released: ReleaseTarget | None = None
        self._lock = threading.RLock()

    def _attach_delegate(self, delegate: Any) -> None:
        if self._delegate is not None:
            raise BindingStateError("device binding delegate was already attached")
        self._delegate = delegate

    def publish(self) -> None:
        """Acknowledge that the caller retained this returned binding."""

        self._owner._publish_binding(self)

    def release(self, target: ReleaseTarget) -> None:
        self._release(target, publish=True)

    def _release(self, target: ReleaseTarget, *, publish: bool) -> None:
        if publish:
            self.publish()
        with self._lock:
            if self._released is not None:
                if self._released is not target:
                    raise BindingStateError(
                        f"device binding was released to {self._released.value}, not {target.value}"
                    )
            else:
                release = getattr(self._delegate, "release", None)
                if callable(release):
                    release(target)
                elif self._controller is not None and self._unit_id is not None:
                    self._controller.release_device(self._unit_id, target)
                self._released = target
        self._owner._release_binding(self)


class PreparedWeightAccessSession:
    """Prevalidated binding transaction transferred to one backend."""

    def __init__(
        self,
        *,
        pipeline: object,
        catalog: TransferCatalog,
        transfer_plan: TransferPlan,
        capabilities: SessionCapabilities,
        artifact: ResolvedArtifact,
        binding: _PreparedBinding,
        runtime: HostWeightRuntime,
    ) -> None:
        self.pipeline = pipeline
        self._catalog = catalog
        self._capabilities = capabilities
        selected_plan = _select_transfer_plan(catalog, capabilities)
        if transfer_plan != selected_plan:
            raise SessionError(
                f"prepared transfer plan {transfer_plan.plan_id!r} does not match selected catalog plan "
                f"{selected_plan.plan_id!r}"
            )
        self._transfer_plan = selected_plan
        self._artifact: ResolvedArtifact | None = artifact
        self._binding: _PreparedBinding | None = binding
        self._runtime: HostWeightRuntime | None = runtime
        self._committed_session: WeightAccessSession | None = None
        self._adopted_session_id: int | None = None
        self._terminal_active_controller_installed = False
        self._state = "prepared"
        self._lock = threading.RLock()

    @property
    def catalog(self) -> TransferCatalog:
        return self._catalog

    @property
    def transfer_plan(self) -> TransferPlan:
        return self._transfer_plan

    def unit_spec(self, unit_id: str) -> TransferUnitSpec:
        return _plan_unit_spec(self._catalog, self._transfer_plan, unit_id)

    @property
    def capabilities(self) -> SessionCapabilities:
        return self._capabilities

    def commit(self) -> WeightAccessSession:
        with self._lock:
            if self._state != "prepared":
                raise SessionClosed(f"prepared weight session is already {self._state}")
            assert self._binding is not None
            assert self._artifact is not None
            assert self._runtime is not None
            # Allocate and retain the active cleanup shell before the binding
            # can cross its irreversible retirement marker.
            active = WeightAccessSession(
                pipeline=self.pipeline,
                catalog=self._catalog,
                transfer_plan=self._transfer_plan,
                capabilities=self._capabilities,
                artifact=self._artifact,
                controller=None,
                runtime=self._runtime,
            )
            binding = self._binding
            self._committed_session = active
            self._state = "committing"
            try:
                controller = binding.commit()
            except BaseException:
                if binding.commit_state is not PreparedBindingCommitState.RETIREMENT_COMMITTED:
                    self._committed_session = None
                    self._state = "prepared"
                    raise
                self._retain_terminal_active_owner(active, binding)
                raise
            if binding.commit_state is not PreparedBindingCommitState.RETIREMENT_COMMITTED:
                self._committed_session = None
                self._state = "prepared"
                raise BindingStateError(
                    "prepared binding returned a controller without publishing RETIREMENT_COMMITTED"
                )
            retained_controller = binding.retained_controller
            if retained_controller is not controller:
                self._retain_terminal_active_owner(active, binding)
                raise BindingStateError("prepared binding returned a controller it did not retain")
            try:
                active._controller = controller
                self._terminal_active_controller_installed = True
            except BaseException:
                # Bypass an injected/subclassed STORE_ATTR failure.  Marker
                # state, never controller presence, selected this terminal
                # cleanup branch.
                object.__setattr__(active, "_controller", controller)
                self._terminal_active_controller_installed = True
                self._state = "committed"
                raise
            self._state = "committed"
            return active

    def _retain_terminal_active_owner(
        self,
        active: WeightAccessSession,
        binding: _PreparedBinding,
    ) -> None:
        """Attach the post-marker controller to the retained cleanup shell."""

        controller = binding.retained_controller
        if controller is None:
            self._state = "cleanup_required"
            raise BindingStateError("RETIREMENT_COMMITTED binding has no retained controller for terminal cleanup")
        object.__setattr__(active, "_controller", controller)
        self._terminal_active_controller_installed = True
        self._state = "committed"

    def adopt(self, active: WeightAccessSession) -> None:
        """Acknowledge that a backend published the committed active owner."""

        with self._lock:
            if self._state == "active" and self._adopted_session_id == id(active):
                self._committed_session = None
                self._binding = None
                self._artifact = None
                self._runtime = None
                return
            if self._state != "committed" or self._committed_session is not active:
                raise SessionClosed("prepared weight session has no matching committed session to adopt")
            self._adopted_session_id = id(active)
            self._state = "active"
            self._committed_session = None
            self._binding = None
            self._artifact = None
            self._runtime = None

    def rollback(self) -> None:
        with self._lock:
            if self._state == "closed":
                self._committed_session = None
                self._binding = None
                self._artifact = None
                self._runtime = None
                return
            binding = self._binding
            retirement_committed = (
                binding is not None and binding.commit_state is PreparedBindingCommitState.RETIREMENT_COMMITTED
            )
            if retirement_committed:
                active = self._committed_session
                assert active is not None
                if not self._terminal_active_controller_installed:
                    if active._controller is None:
                        self._retain_terminal_active_owner(active, binding)
                    else:
                        self._terminal_active_controller_installed = True
                active.abort(DetachMode.TERMINAL)
                self._state = "closed"
                if self._committed_session is active:
                    self._committed_session = None
                self._binding = None
                self._artifact = None
                self._runtime = None
                return
            if self._state not in {"prepared", "committing", "rolling_back", "cleanup_required"}:
                raise SessionClosed(f"prepared weight session is already {self._state}")
            self._state = "rolling_back"
            resources = (
                ("_binding", "rolling back the prepared binding", "rollback"),
                ("_artifact", "closing the prepared artifact", "close"),
                ("_runtime", "closing the prepared runtime", "close"),
            )
            for attribute, _action, method_name in resources:
                resource = getattr(self, attribute)
                if resource is None:
                    continue
                # Cleanup is dependency ordered.  In particular, the artifact
                # must remain live while binding rollback is unfinished, and
                # the runtime must remain live until its artifact is closed.
                getattr(resource, method_name)()
                setattr(self, attribute, None)
            if self._binding is None and self._artifact is None and self._runtime is None:
                self._state = "closed"


class WeightAccessSession:
    def __init__(
        self,
        *,
        pipeline: object,
        catalog: TransferCatalog,
        transfer_plan: TransferPlan,
        capabilities: SessionCapabilities,
        artifact: ResolvedArtifact,
        controller: _BindingController | None,
        runtime: HostWeightRuntime,
    ) -> None:
        self.pipeline = pipeline
        self._catalog = catalog
        self._transfer_plan = transfer_plan
        self._capabilities = capabilities
        self._artifact: ResolvedArtifact | None = artifact
        self._controller: _BindingController | None = controller
        self._runtime: HostWeightRuntime | None = runtime
        self._state = "active"
        self._close_mode: DetachMode | None = None
        self._unit_registrations: dict[int, _UnitRegistration] = {}
        self._device_bindings: set[DeviceBinding] = set()
        self._pending_device_bindings: set[DeviceBinding] = set()
        self._lock = threading.RLock()

    @property
    def catalog(self) -> TransferCatalog:
        return self._catalog

    @property
    def transfer_plan(self) -> TransferPlan:
        return self._transfer_plan

    def unit_spec(self, unit_id: str) -> TransferUnitSpec:
        return _plan_unit_spec(self._catalog, self._transfer_plan, unit_id)

    @property
    def capabilities(self) -> SessionCapabilities:
        return self._capabilities

    def idle_state(self) -> dict[str, int | str]:
        """Return in-flight work separately from intentional resident binds."""

        with self._lock:
            resident_bindings = 0
            for binding in self._device_bindings:
                if binding._lifetime is DeviceBindingLifetime.RESIDENT:  # noqa: SLF001
                    resident_bindings += 1
            total_bindings = len(self._device_bindings)
            return {
                "state": self._state,
                "outstanding_units": len(self._unit_registrations),
                # Qualification's idle gate covers transient block/component
                # bindings. Resident-lifetime bindings intentionally remain
                # attached for the enabled backend's lifetime and are reported
                # separately from their transfer-unit layout kind.
                "bindings": total_bindings - resident_bindings,
                "resident_bindings": resident_bindings,
                "total_bindings": total_bindings,
            }

    def _require_active(self) -> None:
        if self._state != "active":
            raise SessionClosed(f"weight access session is {self._state}")

    def open_unit(self, request: UnitReadRequest) -> PreparedWeightUnit:
        with self._lock:
            self._require_active()
            assert self._artifact is not None
            unit = PreparedWeightUnit(self, self.unit_spec(request.unit_id))
            registration = _UnitRegistration(unit, _UnitHandleState.UNPUBLISHED)
            try:
                self._register_unpublished_unit(registration)
                unit._open_view()
            except BaseException as primary_error:
                try:
                    unit._close(publish=False)
                except BaseException as cleanup_error:
                    _add_cleanup_note(
                        primary_error,
                        "closing a failed unpublished weight read",
                        cleanup_error,
                    )
                raise
            return self._return_open_unit(unit)

    def _register_unpublished_unit(self, registration: _UnitRegistration) -> None:
        """Install the sole exact-identity owner before artifact acquisition."""

        key = id(registration.unit)
        existing = self._unit_registrations.get(key)
        if existing is not None and existing.unit is not registration.unit:
            raise BindingStateError("prepared-unit identity collision")
        self._unit_registrations[key] = registration

    @staticmethod
    def _return_open_unit(unit: PreparedWeightUnit) -> PreparedWeightUnit:
        """Exact return-boundary hook; ownership remains unpublished here."""

        return unit

    def bind_device(
        self,
        unit_id: str,
        buffers: DevicePlaneBuffers,
        *,
        lifetime: DeviceBindingLifetime | None = None,
    ) -> DeviceBinding:
        with self._lock:
            self._require_active()
            layout = self.unit_spec(unit_id)
            validate_plane_buffers(layout, buffers, device_type=None)
            if any(tensor.device.type == "cpu" or tensor.is_meta for tensor in buffers.values()):
                raise SessionError("device binding requires physical non-CPU plane buffers")
            assert self._controller is not None
            binding_lifetime = (
                DeviceBindingLifetime.RESIDENT
                if lifetime is None and layout.unit_kind is UnitKind.RESIDENT
                else DeviceBindingLifetime.TRANSIENT
                if lifetime is None
                else DeviceBindingLifetime(lifetime)
            )
            binding = DeviceBinding(
                self,
                controller=self._controller,
                unit_id=unit_id,
                lifetime=binding_lifetime,
            )
            # Wrapper construction and registration happen before the
            # controller installs any device tensors.  If either allocation
            # fails, no controller-side state needs recovery.
            self._pending_device_bindings.add(binding)
            try:
                self._device_bindings.add(binding)
            except BaseException:
                self._pending_device_bindings.discard(binding)
                raise
            try:
                delegate = self._controller.bind_device(unit_id, buffers)
                binding._attach_delegate(delegate)
            except BaseException as primary_error:
                try:
                    binding._release(ReleaseTarget.PLACEHOLDER, publish=False)
                except BaseException as cleanup_error:
                    _add_cleanup_note(
                        primary_error,
                        "releasing an unpublished device binding",
                        cleanup_error,
                    )
                raise
            return self._return_device_binding(binding)

    @staticmethod
    def _return_device_binding(binding: DeviceBinding) -> DeviceBinding:
        """Exact return-boundary hook; ownership remains unpublished here."""

        return binding

    def _release_unit(self, unit: PreparedWeightUnit) -> None:
        with self._lock:
            key = id(unit)
            registration = self._unit_registrations.get(key)
            if registration is not None and registration.unit is unit:
                self._unit_registrations.pop(key, None)

    def _publish_unit(self, unit: PreparedWeightUnit) -> None:
        with self._lock:
            registration = self._unit_registrations.get(id(unit))
            if registration is not None and registration.unit is unit:
                registration.state = _UnitHandleState.PUBLISHED

    def _release_binding(self, binding: DeviceBinding) -> None:
        with self._lock:
            self._pending_device_bindings.discard(binding)
            self._device_bindings.discard(binding)

    def _publish_binding(self, binding: DeviceBinding) -> None:
        with self._lock:
            self._pending_device_bindings.discard(binding)

    def _drain_unpublished_units(self) -> None:
        for registration in tuple(self._unit_registrations.values()):
            if registration.state is _UnitHandleState.UNPUBLISHED:
                registration.unit._close(publish=False)

    def _drain_pending_bindings(self) -> None:
        for binding in tuple(self._pending_device_bindings):
            binding._release(ReleaseTarget.PLACEHOLDER, publish=False)

    def suspend(self) -> None:
        with self._lock:
            if self._state == "suspended":
                return
            self._require_active()
            self._drain_unpublished_units()
            self._drain_pending_bindings()
            if self._unit_registrations or self._device_bindings:
                raise SessionBusy("cannot suspend while weight reads or device bindings are live")
            self._state = "suspended"

    def resume(self) -> None:
        with self._lock:
            if self._state != "suspended":
                raise SessionClosed(f"only a suspended weight session can resume; state={self._state}")
            self._state = "active"

    def abort(self, mode: DetachMode) -> None:
        """Retryable terminal cleanup for a committed session not yet adopted."""

        with self._lock:
            if self._state == "closed":
                return
            if self._state == "active":
                if self._unit_registrations or self._device_bindings:
                    raise SessionBusy("cannot abort while weight reads or device bindings are live")
                self._state = "suspended"
            elif self._state not in {"suspended", "closing"}:
                raise SessionClosed(f"weight access session cannot abort from {self._state}")
            self.close(mode)

    def close(self, mode: DetachMode) -> None:
        with self._lock:
            if self._state == "closed":
                return
            if self._state not in {"suspended", "closing"}:
                raise SessionBusy("weight session must be suspended before it can close")
            if self._unit_registrations or self._device_bindings:
                raise SessionBusy("cannot close while weight reads or device bindings are live")
            if self._close_mode is not None and self._close_mode is not mode:
                raise BindingStateError(
                    f"weight session close already started with {self._close_mode.value}, not {mode.value}"
                )
            self._close_mode = mode
            self._state = "closing"
            resources = (
                ("_controller", "closing the binding controller", lambda resource: resource.close(mode)),
                ("_artifact", "closing the resolved artifact", lambda resource: resource.close()),
                ("_runtime", "closing the host-weight runtime", lambda resource: resource.close()),
            )
            for attribute, _action, closer in resources:
                resource = getattr(self, attribute)
                if resource is None:
                    continue
                closer(resource)
                setattr(self, attribute, None)
            if self._controller is None and self._artifact is None and self._runtime is None:
                self._state = "closed"


__all__ = [
    "BindingStateError",
    "DetachMode",
    "DeviceBinding",
    "DevicePlaneBuffers",
    "HostPlaneBuffers",
    "HostCopyMode",
    "PreparedWeightAccessSession",
    "PreparedWeightUnit",
    "ReleaseTarget",
    "SessionBusy",
    "SessionCapabilities",
    "SessionClosed",
    "SessionError",
    "SessionRequirements",
    "UnitOutsideSelectedPlan",
    "UnitReadRequest",
    "WeightAccessSession",
]
