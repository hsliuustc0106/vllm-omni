# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recording fakes for offloader-facing host-weight session tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import torch

from vllm_omni.diffusion.host_weight.session import (
    DetachMode,
    DeviceBindingLifetime,
    ReleaseTarget,
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


def _storage_numel(tensor: torch.Tensor) -> int:
    if tensor.numel() == 0:
        return 0
    return 1 + sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride(), strict=True))


def make_unit(
    unit_id: str,
    unit_kind: UnitKind,
    entries: Sequence[tuple[str, str, torch.Tensor, str]],
) -> TransferUnitSpec:
    def target_relative_path(module_path: str) -> TargetModulePath:
        if module_path == "transformer":
            return TargetModulePath(".")
        prefix = "transformer."
        if module_path.startswith(prefix):
            return TargetModulePath(module_path.removeprefix(prefix))
        return TargetModulePath(module_path)

    def local_attribute_name(tensor_id: str) -> str:
        name = tensor_id.rsplit(".", 1)[-1]
        return "bias" if name.endswith("_bias") else name

    bindings = tuple(
        TensorBindingSpec(
            tensor_id=tensor_id,
            destination=BindingDestination(
                module_path=target_relative_path(module_path),
                attribute_name=local_attribute_name(tensor_id),
                state_kind=ModuleStateKind(state_kind),
            ),
        )
        for tensor_id, module_path, _tensor, state_kind in entries
    )
    by_dtype: dict[torch.dtype, list[tuple[str, torch.Tensor]]] = {}
    for tensor_id, _module_path, tensor, _state_kind in entries:
        by_dtype.setdefault(tensor.dtype, []).append((tensor_id, tensor))

    planes: list[DtypePlaneSpec] = []
    for index, (dtype, tensors) in enumerate(by_dtype.items()):
        offset = 0
        placements: list[TensorPlacement] = []
        for tensor_id, tensor in tensors:
            storage_numel = _storage_numel(tensor)
            placements.append(
                TensorPlacement(
                    tensor_id=tensor_id,
                    offset_numel=offset,
                    logical_shape=tuple(tensor.shape),
                    physical_stride=tuple(tensor.stride()),
                    storage_numel=storage_numel,
                )
            )
            offset += storage_numel
        planes.append(
            DtypePlaneSpec(
                plane_id=PlaneId(f"{unit_id}/plane/{index}"),
                dtype=dtype,
                storage_numel=offset,
                placements=tuple(placements),
            )
        )
    return TransferUnitSpec(
        unit_id=unit_id,
        unit_kind=unit_kind,
        bindings=bindings,
        planes=tuple(planes),
    )


def make_catalog(units: Sequence[TransferUnitSpec]) -> TransferCatalog:
    materialized_units = tuple(units)
    execution_units = [unit for unit in materialized_units if unit.unit_kind is not UnitKind.RESIDENT]
    if not execution_units:
        raise ValueError("recording catalog requires at least one executable transfer unit")
    execution_kinds = {unit.unit_kind for unit in execution_units}
    if execution_kinds == {UnitKind.COMPONENT}:
        if len(execution_units) != 1:
            raise ValueError("recording component catalog supports one managed target")
        plan_kind = TransferPlanKind.COMPONENT
        plan_id = "plan.component"
        execution_bindings = (ModuleUnitBinding(module_path=TargetModulePath("."), unit_id=execution_units[0].unit_id),)
    elif execution_kinds == {UnitKind.BLOCK}:
        plan_kind = TransferPlanKind.BLOCKS_PLUS_RESIDENT
        plan_id = "plan.blocks_plus_resident"
        execution_bindings = tuple(
            ModuleUnitBinding(
                module_path=unit.bindings[0].destination.module_path,
                unit_id=unit.unit_id,
            )
            for unit in execution_units
        )
    else:
        raise ValueError(f"recording catalog has incompatible execution unit kinds: {execution_kinds}")

    unit_ids = tuple(unit.unit_id for unit in materialized_units)
    transfer_plan = TransferPlan(
        plan_id=plan_id,
        plan_kind=plan_kind,
        unit_ids=unit_ids,
        execution_bindings=execution_bindings,
        exact_coverage_digest=compute_exact_coverage_digest(
            unit_ids,
            execution_bindings,
            materialized_units,
        ),
    )
    plans = (transfer_plan,)
    return TransferCatalog(
        artifact_compatibility_digest="artifact",
        transfer_catalog_digest=compute_transfer_catalog_digest(materialized_units, plans),
        units=materialized_units,
        plans=plans,
    )


class RecordingPreparedUnit:
    def __init__(self, owner: RecordingSession, layout: TransferUnitSpec) -> None:
        self.owner = owner
        self.layout = layout
        self.closed = False
        self.copied = False
        self.published = False

    def publish(self) -> None:
        self.published = True

    def copy_into(self, destination: Mapping[PlaneId, torch.Tensor]) -> None:
        if self.closed or self.copied:
            raise RuntimeError("prepared unit is not readable")
        if set(destination) != {plane.plane_id for plane in self.layout.planes}:
            raise RuntimeError("destination plane keys are not exact")
        self.owner.destination_storage_ids.add(
            tuple(destination[plane.plane_id].data_ptr() for plane in self.layout.planes)
        )
        if self.owner.fail_unit_id == self.layout.unit_id:
            raise RuntimeError("injected session read failure")
        for plane in self.layout.planes:
            storage = destination[plane.plane_id]
            for placement in plane.placements:
                view = torch.as_strided(
                    storage[placement.offset_numel : placement.offset_numel + placement.storage_numel],
                    size=placement.logical_shape,
                    stride=placement.physical_stride,
                )
                view.copy_(self.owner.sources[placement.tensor_id])
        self.copied = True
        self.owner.events.append(("copy", self.layout.unit_id))

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.owner.open_units.discard(self)
        self.owner.events.append(("unit_close", self.layout.unit_id))


class RecordingDeviceBinding:
    def __init__(
        self,
        owner: RecordingSession,
        unit: TransferUnitSpec,
        lifetime: DeviceBindingLifetime,
    ) -> None:
        self.owner = owner
        self.unit = unit
        self.lifetime = lifetime
        self.released = False
        self.release_calls = 0
        self.published = False

    def publish(self) -> None:
        self.published = True

    def release(self, target: ReleaseTarget) -> None:
        self.release_calls += 1
        if target is not ReleaseTarget.PLACEHOLDER:
            raise RuntimeError("recording binding supports placeholders only")
        if self.released:
            return
        self.released = True
        for binding in self.unit.bindings:
            tensor = self.owner.targets[binding.tensor_id]
            tensor.data = torch.empty((0,), dtype=tensor.dtype, device=tensor.device)
        self.owner.active_bindings.discard(self)
        self.owner.events.append(("release", self.unit.unit_id))


class RecordingSession:
    def __init__(
        self,
        catalog: TransferCatalog,
        transfer_plan: TransferPlan,
        targets: Mapping[str, torch.Tensor],
        sources: Mapping[str, torch.Tensor],
        events: list[Any],
        *,
        fail_unit_id: str | None,
    ) -> None:
        self.catalog = catalog
        self.transfer_plan = transfer_plan
        self.targets = targets
        self.sources = sources
        self.events = events
        self.fail_unit_id = fail_unit_id
        self.open_units: set[RecordingPreparedUnit] = set()
        self.active_bindings: set[RecordingDeviceBinding] = set()
        self.destination_storage_ids: set[tuple[int, ...]] = set()
        self.suspended = False
        self.suspend_count = 0
        self.closed = False

    def open_unit(self, request: UnitReadRequest) -> RecordingPreparedUnit:
        if self.suspended or self.closed:
            raise RuntimeError("invalid unit read")
        unit = self.unit_spec(request.unit_id)
        prepared = RecordingPreparedUnit(self, unit)
        self.open_units.add(prepared)
        self.events.append(("open", unit.unit_id))
        return prepared

    def bind_device(
        self,
        unit_id: str,
        buffers: Mapping[PlaneId, torch.Tensor],
        *,
        lifetime: DeviceBindingLifetime | None = None,
    ) -> RecordingDeviceBinding:
        unit = self.unit_spec(unit_id)
        if set(buffers) != {plane.plane_id for plane in unit.planes}:
            raise RuntimeError("device plane keys are not exact")
        for plane in unit.planes:
            storage = buffers[plane.plane_id]
            for placement in plane.placements:
                target = self.targets[placement.tensor_id]
                target.data = torch.as_strided(
                    storage[placement.offset_numel : placement.offset_numel + placement.storage_numel],
                    size=placement.logical_shape,
                    stride=placement.physical_stride,
                )
        binding_lifetime = (
            DeviceBindingLifetime.RESIDENT
            if lifetime is None and unit.unit_kind is UnitKind.RESIDENT
            else DeviceBindingLifetime.TRANSIENT
            if lifetime is None
            else DeviceBindingLifetime(lifetime)
        )
        binding = RecordingDeviceBinding(self, unit, binding_lifetime)
        self.active_bindings.add(binding)
        self.events.append(("bind", unit_id))
        return binding

    def unit_spec(self, unit_id: str) -> TransferUnitSpec:
        if unit_id not in self.transfer_plan.unit_ids:
            raise RuntimeError(f"unit {unit_id!r} is outside the selected recording plan")
        return self.catalog.unit(unit_id)

    def suspend(self) -> None:
        if self.closed:
            raise RuntimeError("weight access session is closed")
        if self.suspended:
            return
        if self.open_units or self.active_bindings:
            raise RuntimeError("session is busy")
        self.suspend_count += 1
        self.suspended = True
        self.events.append("suspend")

    def resume(self) -> None:
        self.suspended = False
        self.events.append("resume")

    def close(self, mode: DetachMode) -> None:
        if self.closed:
            return
        if not self.suspended:
            raise RuntimeError("session must be suspended")
        self.closed = True
        self.events.append(("close", mode))


class RecordingPreparedSession:
    def __init__(
        self,
        catalog: TransferCatalog,
        targets: Mapping[str, torch.Tensor],
        *,
        fail_unit_id: str | None = None,
    ) -> None:
        self.catalog = catalog
        self.transfer_plan = catalog.plans[0]
        self.targets = dict(targets)
        self.sources = {tensor_id: tensor.detach().clone() for tensor_id, tensor in self.targets.items()}
        self.capabilities = SimpleNamespace(
            access_features=frozenset({"complete_tensor_read"}),
            selected_transfer_plan_id=self.transfer_plan.plan_id,
            selected_transfer_plan_kind=self.transfer_plan.plan_kind,
            unit_kinds=frozenset(self.unit_spec(unit_id).unit_kind for unit_id in self.transfer_plan.unit_ids),
            host_copy_mode="synchronous",
        )
        self.fail_unit_id = fail_unit_id
        self.events: list[Any] = []
        self.commit_count = 0
        self.rollback_count = 0
        self.active: RecordingSession | None = None
        self._adopted = False

    def unit_spec(self, unit_id: str) -> TransferUnitSpec:
        if unit_id not in self.transfer_plan.unit_ids:
            raise RuntimeError(f"unit {unit_id!r} is outside the selected recording plan")
        return self.catalog.unit(unit_id)

    def commit(self) -> RecordingSession:
        self.commit_count += 1
        self.events.append("commit")
        for target in self.targets.values():
            target.data = torch.empty((0,), dtype=target.dtype, device=target.device)
        self.active = RecordingSession(
            self.catalog,
            self.transfer_plan,
            self.targets,
            self.sources,
            self.events,
            fail_unit_id=self.fail_unit_id,
        )
        return self.active

    def adopt(self, active: RecordingSession) -> None:
        if active is not self.active:
            raise RuntimeError("cannot adopt an unknown recording session")
        self._adopted = True

    def rollback(self) -> None:
        self.rollback_count += 1
        self.events.append("rollback")
        if self.active is not None and not self._adopted and not self.active.closed:
            self.active.suspend()
            self.active.close(DetachMode.TERMINAL)


__all__ = [
    "RecordingPreparedSession",
    "make_catalog",
    "make_unit",
]
