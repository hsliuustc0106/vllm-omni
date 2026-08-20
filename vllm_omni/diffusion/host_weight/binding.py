# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transactional hydration of a diffusion consumer skeleton."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import torch
from torch import nn

from .formats.base import (
    ConsumerFormatAdapter,
    FormatBindingRecipe,
    FormatTensorRole,
    ModuleStateKind,
    TargetModulePath,
)
from .skeleton import PipelineSkeleton


class BindingError(RuntimeError):
    """Base class for structural or lifecycle binding failures."""


class BindingStateError(BindingError):
    pass


class BindingValidationError(BindingError):
    pass


class _FormatStateRetirement(Protocol):
    @property
    def retirement_committed(self) -> bool: ...

    def apply(self) -> None: ...

    def validate_quiesced(self) -> None: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...


class _BindingState(str, Enum):
    PREPARED = "prepared"
    HYDRATED = "hydrated"
    VALIDATED = "validated"
    ROLLING_BACK = "rolling_back"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"


class PreparedBindingCommitState(str, Enum):
    """Monotonic recovery marker for prepared-to-active binding handoff."""

    PREPARED = "prepared"
    PROVISIONAL_CONTROLLER = "provisional_controller"
    RETIREMENT_COMMITTED = "retirement_committed"
    ROLLED_BACK = "rolled_back"


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


def _value(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        if name not in value:
            raise BindingValidationError(f"value is missing {name!r}")
        return value[name]
    if not hasattr(value, name):
        raise BindingValidationError(f"{type(value).__name__} is missing {name!r}")
    return getattr(value, name)


def _enum_value(value: object) -> str:
    raw = getattr(value, "value", value)
    return str(raw)


def _resolve_module(root: nn.Module, path: str) -> nn.Module:
    if path == ".":
        path = ""
    try:
        return root.get_submodule(path) if path else root
    except AttributeError as exc:
        raise BindingValidationError(f"target module has no submodule {path!r}") from exc


def _resolve_pipeline_target(pipeline: object, path: str) -> object:
    current = pipeline
    for component in path.split("."):
        if not component or not hasattr(current, component):
            raise BindingValidationError(f"pipeline has no target path {path!r}")
        current = getattr(current, component)
    return current


def _storage_numel(shape: tuple[int, ...], stride: tuple[int, ...]) -> int:
    if len(shape) != len(stride):
        raise BindingValidationError("tensor shape and stride ranks differ")
    if any(size < 0 for size in shape):
        raise BindingValidationError("tensor shape has a negative dimension")
    if any(size == 0 for size in shape):
        return 0
    if any(axis_stride < 0 for axis_stride in stride):
        raise BindingValidationError("negative strides are unsupported")
    return 1 + sum((size - 1) * axis_stride for size, axis_stride in zip(shape, stride, strict=True))


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _manifest_specs(manifest: object) -> dict[str, object]:
    tensors = _value(manifest, "tensors")
    if not isinstance(tensors, (tuple, list)):
        raise BindingValidationError("artifact manifest tensors must be a sequence")
    result: dict[str, object] = {}
    for spec in tensors:
        tensor_id = str(_value(spec, "tensor_id"))
        if tensor_id in result:
            raise BindingValidationError(f"duplicate manifest tensor ID {tensor_id!r}")
        result[tensor_id] = spec
    return result


def _catalog_tensor_ids(catalog: object) -> set[str]:
    direct = getattr(catalog, "tensors", None)
    if isinstance(direct, (tuple, list)):
        return {str(_value(tensor, "tensor_id")) for tensor in direct}

    result: set[str] = set()
    units = getattr(catalog, "units", None)
    if isinstance(units, (tuple, list)):
        for unit in units:
            bindings = getattr(unit, "bindings", None)
            if isinstance(bindings, (tuple, list)):
                result.update(str(_value(binding, "tensor_id")) for binding in bindings)
    if result:
        return result

    transfer_units = getattr(catalog, "transfer_units", None)
    if isinstance(transfer_units, (tuple, list)):
        for unit in transfer_units:
            tensor_ids = getattr(unit, "tensor_ids", None)
            if isinstance(tensor_ids, (tuple, list)):
                result.update(str(tensor_id) for tensor_id in tensor_ids)
    if not result:
        raise BindingValidationError("transfer catalog does not expose tensor coverage")
    return result


def _catalog_unit(catalog: object, unit_id: str) -> object:
    getter = getattr(catalog, "unit", None)
    if callable(getter):
        try:
            return getter(unit_id)
        except Exception as exc:
            raise BindingValidationError(f"unknown transfer unit {unit_id!r}") from exc
    units = getattr(catalog, "units", None)
    if isinstance(units, (tuple, list)):
        for unit in units:
            if str(_value(unit, "unit_id")) == unit_id:
                return unit
    raise BindingValidationError(f"unknown transfer unit {unit_id!r}")


def _validate_transfer_catalog(
    catalog: object,
    expected_ids: set[str],
    specs: Mapping[str, object],
    destinations: Mapping[str, _Destination],
) -> None:
    if _catalog_tensor_ids(catalog) != expected_ids:
        actual = _catalog_tensor_ids(catalog)
        raise BindingValidationError(
            "transfer catalog and binding recipe tensor coverage differ: "
            f"missing={sorted(expected_ids - actual)}, extra={sorted(actual - expected_ids)}"
        )
    units = getattr(catalog, "units", None)
    if not isinstance(units, (tuple, list)) or not units:
        raise BindingValidationError("final transfer catalog must expose non-empty units")
    for unit in units:
        bindings = _value(unit, "bindings")
        planes = _value(unit, "planes")
        if not isinstance(bindings, (tuple, list)) or not isinstance(planes, (tuple, list)):
            raise BindingValidationError("transfer unit bindings and planes must be sequences")
        binding_ids = [str(_value(binding, "tensor_id")) for binding in bindings]
        placement_ids: list[str] = []
        for plane in planes:
            placements = _value(plane, "placements")
            if not isinstance(placements, (tuple, list)):
                raise BindingValidationError("transfer plane placements must be a sequence")
            plane_dtype = _value(plane, "dtype")
            plane_storage_numel = int(_value(plane, "storage_numel"))
            intervals: list[tuple[int, int]] = []
            for placement in placements:
                tensor_id = str(_value(placement, "tensor_id"))
                placement_ids.append(tensor_id)
                if tensor_id not in specs:
                    raise BindingValidationError(f"transfer placement references unknown tensor {tensor_id!r}")
                spec = specs[tensor_id]
                expected = (
                    str(_value(spec, "dtype")).removeprefix("torch."),
                    tuple(int(item) for item in _value(spec, "shape")),  # type: ignore[arg-type]
                    tuple(int(item) for item in _value(spec, "stride")),  # type: ignore[arg-type]
                    int(_value(spec, "storage_numel")),
                )
                actual = (
                    _dtype_name(plane_dtype),  # type: ignore[arg-type]
                    tuple(int(item) for item in _value(placement, "logical_shape")),  # type: ignore[arg-type]
                    tuple(int(item) for item in _value(placement, "physical_stride")),  # type: ignore[arg-type]
                    int(_value(placement, "storage_numel")),
                )
                if actual != expected:
                    raise BindingValidationError(
                        f"transfer placement for {tensor_id!r} differs from the manifest: "
                        f"expected={expected}, actual={actual}"
                    )
                start = int(_value(placement, "offset_numel"))
                end = start + actual[3]
                if start < 0 or end > plane_storage_numel:
                    raise BindingValidationError(f"transfer placement for {tensor_id!r} exceeds its dtype plane")
                intervals.append((start, end))
            intervals.sort()
            if any(current[0] < previous[1] for previous, current in zip(intervals, intervals[1:])):
                raise BindingValidationError("transfer plane contains overlapping tensor placements")
        if len(set(binding_ids)) != len(binding_ids):
            raise BindingValidationError("transfer unit contains duplicate tensor bindings")
        if len(set(placement_ids)) != len(placement_ids):
            raise BindingValidationError("transfer unit contains duplicate tensor placements")
        if set(binding_ids) != set(placement_ids):
            raise BindingValidationError("transfer unit bindings and placements differ")
        if not set(binding_ids) <= expected_ids:
            raise BindingValidationError("transfer unit references a tensor outside the recipe")
        for binding in bindings:
            tensor_id = str(_value(binding, "tensor_id"))
            destination = _value(binding, "destination")
            module_path = str(_value(destination, "module_path"))
            attribute_name = str(_value(destination, "attribute_name"))
            state_kind = _enum_value(_value(destination, "state_kind"))
            expected = destinations[tensor_id]
            actual_destination = (module_path, attribute_name, state_kind)
            expected_destination = (
                str(expected.module_path),
                expected.attribute,
                expected.kind.value,
            )
            if actual_destination != expected_destination:
                raise BindingValidationError(
                    f"transfer binding for {tensor_id!r} has destination {actual_destination!r}, "
                    f"expected {expected_destination!r}"
                )


@dataclass(slots=True)
class _StateSnapshot:
    owner: nn.Module
    attribute: str
    kind: ModuleStateKind
    existed: bool
    previous: object

    def restore(self) -> None:
        if self.kind is ModuleStateKind.PARAMETER:
            if self.existed:
                self.owner._parameters[self.attribute] = self.previous  # type: ignore[assignment]
            else:
                self.owner._parameters.pop(self.attribute, None)
        elif self.kind is ModuleStateKind.PERSISTENT_BUFFER:
            if self.existed:
                self.owner._buffers[self.attribute] = self.previous  # type: ignore[assignment]
            else:
                self.owner._buffers.pop(self.attribute, None)
                self.owner._non_persistent_buffers_set.discard(self.attribute)
        elif self.existed:
            setattr(self.owner, self.attribute, self.previous)
        elif self.attribute in self.owner.__dict__:
            delattr(self.owner, self.attribute)


@dataclass(slots=True)
class _ScalarSnapshot:
    owner: object
    attribute: str
    existed: bool
    previous: object

    def restore(self) -> None:
        if self.existed:
            setattr(self.owner, self.attribute, self.previous)
        elif self.attribute in getattr(self.owner, "__dict__", {}):
            delattr(self.owner, self.attribute)


@dataclass(frozen=True, slots=True)
class _Destination:
    tensor_id: str
    module_path: TargetModulePath
    owner: nn.Module
    attribute: str
    kind: ModuleStateKind
    format_role: FormatTensorRole | None


def _preflight_destination(destination: _Destination) -> None:
    owner = destination.owner
    attribute = destination.attribute
    if destination.kind is ModuleStateKind.PARAMETER:
        if attribute in owner._buffers:
            raise BindingValidationError(f"{destination.tensor_id!r} expects a parameter but destination is a buffer")
        existing = owner._parameters.get(attribute)
        if existing is None and hasattr(owner, attribute) and attribute not in owner._parameters:
            value = getattr(owner, attribute)
            if value is not None:
                raise BindingValidationError(
                    f"{destination.tensor_id!r} parameter destination conflicts with an attribute"
                )
    elif destination.kind is ModuleStateKind.PERSISTENT_BUFFER:
        if attribute in owner._parameters:
            raise BindingValidationError(f"{destination.tensor_id!r} expects a buffer but destination is a parameter")
        if attribute in owner._non_persistent_buffers_set:
            raise BindingValidationError(f"{destination.tensor_id!r} cannot replace a non-persistent buffer")
        if attribute not in owner._buffers and hasattr(owner, attribute):
            value = getattr(owner, attribute)
            if value is not None:
                raise BindingValidationError(
                    f"{destination.tensor_id!r} buffer destination conflicts with an attribute"
                )
    elif attribute in owner._parameters or attribute in owner._buffers:
        raise BindingValidationError(
            f"{destination.tensor_id!r} tensor attribute conflicts with registered module state"
        )


def _expected_manifest_role(destination: _Destination) -> set[str]:
    if destination.format_role is FormatTensorRole.WEIGHT_SCALE:
        return {"quant_metadata", "quantization_metadata"}
    if destination.kind is ModuleStateKind.PARAMETER:
        return {"parameter"}
    return {"persistent_buffer"}


def _validate_artifact_tensor(
    destination: _Destination,
    tensor: torch.Tensor,
    spec: object,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise BindingValidationError(f"placeholder for {destination.tensor_id!r} is not a tensor")
    if not tensor.is_meta:
        raise BindingValidationError(f"structural placeholder {destination.tensor_id!r} must not allocate storage")
    shape = tuple(int(value) for value in _value(spec, "shape"))  # type: ignore[arg-type]
    stride = tuple(int(value) for value in _value(spec, "stride"))  # type: ignore[arg-type]
    expected = (
        str(_value(spec, "dtype")).removeprefix("torch."),
        shape,
        stride,
    )
    actual = (_dtype_name(tensor.dtype), tuple(tensor.shape), tuple(tensor.stride()))
    if actual != expected:
        raise BindingValidationError(
            f"artifact tensor {destination.tensor_id!r} layout mismatch: expected={expected}, actual={actual}"
        )
    storage_numel = _storage_numel(shape, stride)
    if hasattr(spec, "storage_numel") or (isinstance(spec, Mapping) and "storage_numel" in spec):
        if int(_value(spec, "storage_numel")) != storage_numel:
            raise BindingValidationError(f"artifact tensor {destination.tensor_id!r} storage span is inconsistent")
    role = _enum_value(_value(spec, "role"))
    if role not in _expected_manifest_role(destination):
        raise BindingValidationError(
            f"artifact tensor {destination.tensor_id!r} role {role!r} conflicts with "
            f"destination {destination.kind.value!r}"
        )


def _placeholder_from_spec(destination: _Destination, spec: object) -> torch.Tensor:
    shape = tuple(int(value) for value in _value(spec, "shape"))  # type: ignore[arg-type]
    stride = tuple(int(value) for value in _value(spec, "stride"))  # type: ignore[arg-type]
    dtype_name = str(_value(spec, "dtype")).removeprefix("torch.")
    dtype = getattr(torch, dtype_name, None)
    if not isinstance(dtype, torch.dtype):
        raise BindingValidationError(f"artifact tensor {destination.tensor_id!r} has unsupported dtype {dtype_name!r}")
    placeholder = torch.empty_strided(
        shape,
        stride,
        dtype=dtype,
        device="meta",
    )
    _validate_artifact_tensor(destination, placeholder, spec)
    return placeholder


def _install_state(
    destination: _Destination,
    tensor: torch.Tensor,
    snapshots: list[_StateSnapshot],
) -> None:
    """Install one tensor only after its rollback owner is registered.

    Capturing or appending the snapshot is fallible.  Both operations must
    therefore complete before the first module mutation; once registered, the
    caller's existing reverse-order rollback owns every subsequent failure.
    """

    owner = destination.owner
    attribute = destination.attribute
    if destination.kind is ModuleStateKind.PARAMETER:
        existed = attribute in owner._parameters and owner._parameters[attribute] is not None
        previous = owner._parameters.get(attribute)
        requires_grad = bool(getattr(previous, "requires_grad", False))
        snapshots.append(_StateSnapshot(owner, attribute, destination.kind, existed, previous))
        owner._parameters[attribute] = nn.Parameter(tensor, requires_grad=requires_grad)
    elif destination.kind is ModuleStateKind.PERSISTENT_BUFFER:
        existed = attribute in owner._buffers and owner._buffers[attribute] is not None
        previous = owner._buffers.get(attribute)
        snapshots.append(_StateSnapshot(owner, attribute, destination.kind, existed, previous))
        owner._buffers[attribute] = tensor
        owner._non_persistent_buffers_set.discard(attribute)
    else:
        existed = attribute in owner.__dict__
        previous = owner.__dict__.get(attribute)
        snapshots.append(_StateSnapshot(owner, attribute, destination.kind, existed, previous))
        setattr(owner, attribute, tensor)


def _current_state(destination: _Destination) -> torch.Tensor:
    if destination.kind is ModuleStateKind.PARAMETER:
        value = destination.owner._parameters.get(destination.attribute)
    elif destination.kind is ModuleStateKind.PERSISTENT_BUFFER:
        value = destination.owner._buffers.get(destination.attribute)
    else:
        value = getattr(destination.owner, destination.attribute, None)
    if not isinstance(value, torch.Tensor):
        raise BindingValidationError(f"bound destination for {destination.tensor_id!r} is not a tensor")
    return value


class _DeviceBindingDelegate:
    def __init__(
        self,
        owner: BindingController,
        unit_id: str,
        tensor_ids: frozenset[str],
        snapshots: tuple[_StateSnapshot, ...],
    ) -> None:
        self._owner = owner
        self._unit_id = unit_id
        self._tensor_ids = tensor_ids
        self._snapshots = list(snapshots)
        self._released = False
        self._lock = threading.RLock()

    def release(self, target: object) -> None:
        target_value = _enum_value(target)
        if target_value != "placeholder":
            raise BindingStateError(f"device binding can only release to placeholder, got {target_value!r}")
        with self._lock:
            if self._released:
                return
            first_error: BaseException | None = None
            failed_in_reverse: list[_StateSnapshot] = []
            for snapshot in reversed(self._snapshots):
                try:
                    snapshot.restore()
                except BaseException as exc:
                    failed_in_reverse.append(snapshot)
                    if first_error is None:
                        first_error = exc
                    else:
                        _add_cleanup_note(
                            first_error,
                            "restoring another device-binding snapshot",
                            exc,
                        )
            self._snapshots = list(reversed(failed_in_reverse))
            if first_error is not None:
                raise first_error
            self._owner._released(self._unit_id, self._tensor_ids)
            self._released = True


class BindingController:
    """Committed structural binding owner retained by the active session."""

    def __init__(
        self,
        skeleton: PipelineSkeleton,
        catalog: object,
        destinations: Mapping[str, _Destination],
    ) -> None:
        self._skeleton: PipelineSkeleton | None = skeleton
        self._catalog = catalog
        self._destinations = dict(destinations)
        self._active_units: dict[str, _DeviceBindingDelegate] = {}
        self._active_tensor_ids: set[str] = set()
        self._pending_snapshots: list[_StateSnapshot] = []
        self._closed = False
        self._lock = threading.RLock()

    @property
    def pipeline(self) -> object:
        if self._closed or self._skeleton is None:
            raise BindingStateError("binding controller is closed")
        return self._skeleton.pipeline

    @property
    def target_module(self) -> nn.Module:
        if self._closed or self._skeleton is None:
            raise BindingStateError("binding controller is closed")
        return self._skeleton.target_module

    def restore_cpu(self) -> None:
        if self._closed:
            raise BindingStateError("binding controller is closed")
        raise BindingStateError(
            "structural placeholders cannot restore CPU bytes; the active "
            "weight-access session must perform bounded source restoration"
        )

    def bind_device(
        self,
        unit_id: str,
        buffers: Mapping[object, torch.Tensor],
    ) -> _DeviceBindingDelegate:
        with self._lock:
            if self._closed or self._skeleton is None:
                raise BindingStateError("binding controller is closed")
            if self._pending_snapshots:
                raise BindingStateError("a failed device bind must be restored before another bind")
            if unit_id in self._active_units:
                raise BindingStateError(f"transfer unit {unit_id!r} is already bound")
            unit = _catalog_unit(self._catalog, unit_id)
            try:
                from .transfer import tensor_views_for_unit

                views = tensor_views_for_unit(unit, buffers, device_type=None)
            except Exception as exc:
                raise BindingValidationError(f"device planes do not match transfer unit {unit_id!r}") from exc
            tensor_ids = frozenset(views)
            overlap = tensor_ids & self._active_tensor_ids
            if overlap:
                raise BindingStateError(f"device binding overlaps active tensors: {sorted(overlap)}")
            if not tensor_ids or not tensor_ids <= self._destinations.keys():
                raise BindingValidationError(f"transfer unit {unit_id!r} contains unknown tensor bindings")

            snapshots: list[_StateSnapshot] = []
            try:
                for tensor_id, view in views.items():
                    destination = self._destinations[tensor_id]
                    if not isinstance(view, torch.Tensor) or view.is_meta:
                        raise BindingValidationError(f"device view for {tensor_id!r} must have physical storage")
                    current = _current_state(destination)
                    if not current.is_meta:
                        raise BindingStateError(f"destination {tensor_id!r} is not a structural placeholder")
                    _install_state(destination, view, snapshots)
                delegate = _DeviceBindingDelegate(
                    self,
                    unit_id,
                    tensor_ids,
                    tuple(snapshots),
                )
                self._register_device_binding(unit_id, delegate, tensor_ids)
            except BaseException as primary_error:
                # Publication is part of the transaction.  A failure after a
                # partial dict/set update must make the controller quiescent
                # before restoring placeholders.
                self._active_units.pop(unit_id, None)
                self._active_tensor_ids.difference_update(tensor_ids)
                failed_in_reverse: list[_StateSnapshot] = []
                for snapshot in reversed(snapshots):
                    try:
                        snapshot.restore()
                    except BaseException as cleanup_error:
                        failed_in_reverse.append(snapshot)
                        _add_cleanup_note(
                            primary_error,
                            "restoring a partially installed device binding",
                            cleanup_error,
                        )
                self._pending_snapshots.extend(reversed(failed_in_reverse))
                raise
            return delegate

    def _register_device_binding(
        self,
        unit_id: str,
        delegate: _DeviceBindingDelegate,
        tensor_ids: frozenset[str],
    ) -> None:
        self._active_units[unit_id] = delegate
        self._active_tensor_ids.update(tensor_ids)

    def release_device(self, unit_id: str, target: object) -> bool:
        """Release a published delegate when its outer wrapper was not returned."""

        with self._lock:
            delegate = self._active_units.get(unit_id)
            if delegate is None:
                return False
            delegate.release(target)
            return True

    def _released(self, unit_id: str, tensor_ids: frozenset[str]) -> None:
        with self._lock:
            self._active_units.pop(unit_id, None)
            self._active_tensor_ids.difference_update(tensor_ids)

    def close(self, mode: object = "terminal") -> None:
        with self._lock:
            if self._closed:
                return
            if self._active_units:
                raise BindingStateError("cannot close while device bindings are active")
            if _enum_value(mode) == "restore_cpu":
                raise BindingStateError("restore_cpu requires the active weight-access session to materialize bytes")
            first_error: BaseException | None = None
            failed_in_reverse: list[_StateSnapshot] = []
            for snapshot in reversed(self._pending_snapshots):
                try:
                    snapshot.restore()
                except BaseException as exc:
                    failed_in_reverse.append(snapshot)
                    if first_error is None:
                        first_error = exc
                    else:
                        _add_cleanup_note(
                            first_error,
                            "restoring another failed device-bind snapshot",
                            exc,
                        )
            self._pending_snapshots = list(reversed(failed_in_reverse))
            if first_error is not None:
                raise first_error
            self._closed = True
            self._skeleton = None
            self._destinations.clear()


class PreparedModuleBinding:
    def __init__(
        self,
        *,
        skeleton: PipelineSkeleton,
        manifest: object,
        recipe: FormatBindingRecipe,
        adapter: ConsumerFormatAdapter,
        destinations: tuple[_Destination, ...],
        tensors: Mapping[str, torch.Tensor],
        specs: Mapping[str, object],
    ) -> None:
        self._skeleton = skeleton
        self._manifest = manifest
        self._recipe = recipe
        self._adapter = adapter
        self._destinations = destinations
        self._tensors = dict(tensors)
        self._specs = dict(specs)
        self._state = _BindingState.PREPARED
        self._transfer_catalog: object | None = None
        self._state_snapshots: list[_StateSnapshot] = []
        self._scalar_snapshots: list[_ScalarSnapshot] = []
        self._format_state_retirement: _FormatStateRetirement | None = None
        self._commit_state = PreparedBindingCommitState.PREPARED
        self._retained_controller: BindingController | None = None

    @property
    def state(self) -> str:
        return self._state.value

    @property
    def commit_state(self) -> PreparedBindingCommitState:
        """Explicit marker used to select rollback versus terminal cleanup."""

        return self._commit_state

    @property
    def retained_controller(self) -> BindingController | None:
        """Controller retained across the commit return/publication boundary."""

        return self._retained_controller

    def _restore(self) -> None:
        first_error: BaseException | None = None
        for attribute, action in (
            ("_scalar_snapshots", "restoring another scalar binding snapshot"),
            ("_state_snapshots", "restoring another tensor binding snapshot"),
        ):
            snapshots = getattr(self, attribute)
            failed_in_reverse: list[object] = []
            for snapshot in reversed(snapshots):
                try:
                    snapshot.restore()
                except BaseException as exc:
                    failed_in_reverse.append(snapshot)
                    if first_error is None:
                        first_error = exc
                    else:
                        _add_cleanup_note(first_error, action, exc)
            setattr(self, attribute, list(reversed(failed_in_reverse)))

        retirement = self._format_state_retirement
        if retirement is not None:
            try:
                retirement.rollback()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                else:
                    _add_cleanup_note(
                        first_error,
                        "rolling back online-loader state retirement",
                        exc,
                    )
            else:
                self._format_state_retirement = None
        if first_error is not None:
            raise first_error

    def hydrate(self) -> None:
        if self._state is not _BindingState.PREPARED:
            raise BindingStateError(f"hydrate requires prepared state, got {self._state.value}")
        assignments = self._adapter.scalar_assignments(
            self._skeleton.target_module,
            self._recipe,
        )
        try:
            retirement = self._adapter.retire_online_loader_state(
                self._skeleton.target_module,
                self._recipe,
            )
            # Retain the exact rollback owner before its first registry
            # mutation.  An interruption in apply() is therefore handled by
            # the same progress-preserving _restore() path as later hydration
            # failures.
            self._format_state_retirement = retirement
            retirement.apply()
            with torch.no_grad():
                for destination in self._destinations:
                    _install_state(
                        destination,
                        self._tensors[destination.tensor_id],
                        self._state_snapshots,
                    )
                for owner, attribute, value in assignments:
                    existed = attribute in getattr(owner, "__dict__", {})
                    previous = getattr(owner, attribute, None)
                    self._scalar_snapshots.append(_ScalarSnapshot(owner, attribute, existed, previous))
                    setattr(owner, attribute, value)
        except BaseException as primary_error:
            self._state = _BindingState.ROLLING_BACK
            try:
                self._restore()
            except BaseException as cleanup_error:
                _add_cleanup_note(
                    primary_error,
                    "rolling back a failed binding hydration",
                    cleanup_error,
                )
            else:
                self._state = _BindingState.ROLLED_BACK
            raise
        self._state = _BindingState.HYDRATED

    def validate(self) -> None:
        if self._state is not _BindingState.HYDRATED:
            raise BindingStateError(f"validate requires hydrated state, got {self._state.value}")
        if self._transfer_catalog is None:
            raise BindingValidationError("final transfer catalog must be attached after structural hydration")
        for destination in self._destinations:
            current = _current_state(destination)
            _validate_artifact_tensor(
                destination,
                current,
                self._specs[destination.tensor_id],
            )
        self._adapter.validate_hydrated(
            self._skeleton.target_module,
            self._recipe,
        )
        assert self._format_state_retirement is not None
        self._format_state_retirement.validate_quiesced()
        self._state = _BindingState.VALIDATED

    def set_transfer_catalog(self, catalog: object) -> None:
        if self._state is not _BindingState.HYDRATED:
            raise BindingStateError(f"set_transfer_catalog requires hydrated state, got {self._state.value}")
        if self._transfer_catalog is not None:
            raise BindingStateError("final transfer catalog was already attached")
        _validate_transfer_catalog(
            catalog,
            set(self._recipe.tensor_ids),
            self._specs,
            {destination.tensor_id: destination for destination in self._destinations},
        )
        self._transfer_catalog = catalog

    def commit(self) -> BindingController:
        if self._state is not _BindingState.VALIDATED:
            raise BindingStateError(f"commit requires validated state, got {self._state.value}")
        if self._commit_state is not PreparedBindingCommitState.PREPARED:
            raise BindingStateError(f"commit requires prepared marker state, got {self._commit_state.value}")
        # Construct the active owner before the format transaction forgets its
        # rollback snapshot.  Controller construction allocates Python
        # containers and a lock and can therefore fail (including MemoryError).
        assert self._transfer_catalog is not None
        controller = BindingController(
            self._skeleton,
            self._transfer_catalog,
            {destination.tensor_id: destination for destination in self._destinations},
        )
        # Controller presence is deliberately not a commit signal.  It is a
        # provisional rollback resource until the explicit retirement marker
        # below becomes monotonic.
        self._retained_controller = controller
        self._commit_state = PreparedBindingCommitState.PROVISIONAL_CONTROLLER
        assert self._format_state_retirement is not None
        retirement = self._format_state_retirement
        try:
            retirement.commit()
        except BaseException:
            if retirement.retirement_committed:
                self._publish_retirement_committed()
            raise
        if not retirement.retirement_committed:
            raise BindingStateError("format-state retirement returned without publishing its commit marker")
        self._publish_retirement_committed()
        return controller

    def _publish_retirement_committed(self) -> None:
        # This assignment is the prepared binding's recovery linearization
        # point.  It is first and monotonic: any interruption afterwards must
        # select terminal active cleanup and must never restore snapshots.
        self._commit_state = PreparedBindingCommitState.RETIREMENT_COMMITTED
        self._state = _BindingState.COMMITTED
        self._format_state_retirement = None
        self._state_snapshots.clear()
        self._scalar_snapshots.clear()

    def rollback(self) -> None:
        if self._commit_state is PreparedBindingCommitState.ROLLED_BACK:
            return
        if self._commit_state is PreparedBindingCommitState.RETIREMENT_COMMITTED:
            raise BindingStateError("a committed binding cannot be rolled back")
        self._state = _BindingState.ROLLING_BACK
        controller = self._retained_controller
        if controller is not None:
            # A provisional controller is a pre-marker rollback resource.  It
            # must be closed before restoring any tensor/loader snapshots, and
            # remains retained if close fails so the same rollback can retry.
            controller.close("terminal")
            self._retained_controller = None
        self._restore()
        self._state = _BindingState.ROLLED_BACK
        self._commit_state = PreparedBindingCommitState.ROLLED_BACK


class DiffusionConsumerBinder:
    """Prepare a reversible, exact-coverage consumer hydration transaction."""

    def __init__(
        self,
        *,
        target_module_type: type[nn.Module] | None = None,
    ) -> None:
        self._target_module_type = target_module_type

    def prepare(
        self,
        skeleton: PipelineSkeleton,
        manifest: object,
        catalog: object,
        format_adapter: ConsumerFormatAdapter,
        cleanup: object | None = None,
    ) -> PreparedModuleBinding:
        descriptor = getattr(format_adapter, "descriptor", None)
        if isinstance(descriptor, Mapping):
            adapter_target_type_id = descriptor.get("target_module_type_id")
        else:
            adapter_target_type_id = getattr(descriptor, "target_module_type_id", None)
        if not isinstance(adapter_target_type_id, str) or not adapter_target_type_id:
            raise BindingValidationError("format adapter descriptor has no target module type ID")
        if skeleton.target_module_type_id != adapter_target_type_id:
            raise BindingValidationError(
                f"skeleton target type ID {skeleton.target_module_type_id!r} differs from the selected format adapter"
            )
        resolved_target = _resolve_pipeline_target(
            skeleton.pipeline,
            skeleton.target_module_path,
        )
        if resolved_target is not skeleton.target_module:
            raise BindingValidationError("pipeline target identity differs from the skeleton target")
        if self._target_module_type is not None and not isinstance(
            skeleton.target_module,
            self._target_module_type,
        ):
            raise BindingValidationError(
                f"skeleton target has type {type(skeleton.target_module).__name__}, expected "
                f"{self._target_module_type.__name__}"
            )

        recipe = format_adapter.prepare_consumer_structure(
            skeleton.target_module,
            manifest,
        )
        if recipe.target_module_type_id != skeleton.target_module_type_id:
            raise BindingValidationError("recipe and skeleton target type IDs differ")
        specs = _manifest_specs(manifest)
        recipe_ids = set(recipe.tensor_ids)
        catalog_ids = _catalog_tensor_ids(catalog)
        if catalog_ids != recipe_ids:
            raise BindingValidationError(
                "transfer catalog and binding recipe tensor coverage differ: "
                f"missing={sorted(recipe_ids - catalog_ids)}, extra={sorted(catalog_ids - recipe_ids)}"
            )

        destinations: list[_Destination] = []
        destination_paths: set[tuple[str, str]] = set()
        for layer in recipe.layers:
            layer_path = str(layer.module_path)
            owner = _resolve_module(skeleton.target_module, layer_path)
            for binding in layer.tensor_bindings:
                if binding.tensor_id is None:
                    # The allowlisted format adapter has already proven that
                    # this declared optional destination is absent (or has its
                    # declared non-tensor default). It consumes no artifact
                    # tensor and therefore creates no installation target.
                    continue
                declared = binding.destination
                if str(declared.module_path) != layer_path:
                    raise BindingValidationError("layer binding destination is outside its enclosing layer")
                path = (layer_path, declared.attribute_name)
                if path in destination_paths:
                    raise BindingValidationError(f"duplicate binding destination {path!r}")
                destination_paths.add(path)
                destinations.append(
                    _Destination(
                        tensor_id=binding.tensor_id,
                        module_path=declared.module_path,
                        owner=owner,
                        attribute=declared.attribute_name,
                        kind=declared.state_kind,
                        format_role=binding.role,
                    )
                )
        for binding in recipe.non_layer_bindings:
            declared = binding.destination
            owner_path = str(declared.module_path)
            attribute = declared.attribute_name
            owner = _resolve_module(skeleton.target_module, owner_path)
            path = (owner_path, attribute)
            if path in destination_paths:
                raise BindingValidationError(f"duplicate binding destination {path!r}")
            destination_paths.add(path)
            destinations.append(
                _Destination(
                    tensor_id=binding.tensor_id,
                    module_path=declared.module_path,
                    owner=owner,
                    attribute=attribute,
                    kind=declared.state_kind,
                    format_role=None,
                )
            )
        if {destination.tensor_id for destination in destinations} != recipe_ids:
            raise BindingValidationError("prepared destinations do not cover the recipe exactly")

        placeholders: dict[str, torch.Tensor] = {}
        for destination in destinations:
            _preflight_destination(destination)
            spec = specs[destination.tensor_id]
            placeholders[destination.tensor_id] = _placeholder_from_spec(
                destination,
                spec,
            )

        prepared = PreparedModuleBinding(
            skeleton=skeleton,
            manifest=manifest,
            recipe=recipe,
            adapter=format_adapter,
            destinations=tuple(destinations),
            tensors=placeholders,
            specs=specs,
        )
        if cleanup is not None:
            retain = getattr(cleanup, "retain_binding", None)
            if not callable(retain):
                raise BindingValidationError("preparation cleanup registrar cannot retain a binding")
            try:
                retain(prepared)
            except BaseException:
                prepared.rollback()
                raise
        return prepared


__all__ = [
    "BindingController",
    "BindingError",
    "BindingStateError",
    "BindingValidationError",
    "DiffusionConsumerBinder",
    "PreparedBindingCommitState",
    "PreparedModuleBinding",
]
