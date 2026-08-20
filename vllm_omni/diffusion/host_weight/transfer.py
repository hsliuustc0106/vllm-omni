# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Complete-block transfer planning independent of an artifact backing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import NewType

import torch


class TransferPlanError(ValueError):
    pass


PlaneId = NewType("PlaneId", str)
TargetModulePath = NewType("TargetModulePath", str)


class ModuleStateKind(str, Enum):
    PARAMETER = "parameter"
    PERSISTENT_BUFFER = "persistent_buffer"
    TENSOR_ATTRIBUTE = "tensor_attribute"


class UnitKind(str, Enum):
    COMPONENT = "component"
    BLOCK = "block"
    RESIDENT = "resident"


class TransferPlanKind(str, Enum):
    COMPONENT = "component"
    BLOCKS_PLUS_RESIDENT = "blocks_plus_resident"


def _require_name(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TransferPlanError(f"{label} must be a string")
    if not value or value.strip() != value or "\x00" in value:
        raise TransferPlanError(f"{label} must be a non-empty canonical name")
    if value.startswith(".") or value.endswith(".") or ".." in value:
        raise TransferPlanError(f"{label} is not canonical: {value!r}")
    return value


def _target_module_path(value: object, label: str = "module_path") -> TargetModulePath:
    if value == ".":
        return TargetModulePath(".")
    return TargetModulePath(_require_name(value, label))


@dataclass(frozen=True, slots=True)
class BindingDestination:
    """Canonical tensor destination relative to the managed DiT target."""

    module_path: TargetModulePath
    attribute_name: str
    state_kind: ModuleStateKind

    def __post_init__(self) -> None:
        object.__setattr__(self, "module_path", _target_module_path(self.module_path))
        attribute_name = _require_name(self.attribute_name, "attribute_name")
        if "." in attribute_name:
            raise TransferPlanError("attribute_name must be local to its owning module")
        object.__setattr__(self, "attribute_name", attribute_name)
        try:
            object.__setattr__(self, "state_kind", ModuleStateKind(self.state_kind))
        except ValueError as exc:
            raise TransferPlanError(f"unknown module state kind {self.state_kind!r}") from exc

    def to_dict(self) -> dict[str, str]:
        return _destination_payload(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> BindingDestination:
        expected = {"module_path", "attribute_name", "state_kind"}
        if set(value) != expected:
            raise TransferPlanError(
                f"binding destination keys differ: missing={sorted(expected - set(value))}, "
                f"extra={sorted(set(value) - expected)}"
            )
        return cls(
            module_path=TargetModulePath(str(value["module_path"])),
            attribute_name=str(value["attribute_name"]),
            state_kind=str(value["state_kind"]),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class TensorBindingSpec:
    tensor_id: str
    destination: BindingDestination

    def __post_init__(self) -> None:
        _require_name(self.tensor_id, "tensor_id")
        if not isinstance(self.destination, BindingDestination):
            raise TransferPlanError("tensor binding destination must be a BindingDestination")

    def to_dict(self) -> dict[str, object]:
        return {
            "tensor_id": self.tensor_id,
            "destination": self.destination.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> TensorBindingSpec:
        expected = {"tensor_id", "destination"}
        if set(value) != expected:
            raise TransferPlanError(
                f"tensor binding keys differ: missing={sorted(expected - set(value))}, "
                f"extra={sorted(set(value) - expected)}"
            )
        destination = value["destination"]
        if not isinstance(destination, Mapping):
            raise TransferPlanError("tensor binding destination must be a mapping")
        return cls(
            tensor_id=str(value["tensor_id"]),
            destination=BindingDestination.from_dict(destination),
        )


@dataclass(frozen=True)
class TensorPlacement:
    tensor_id: str
    offset_numel: int
    logical_shape: tuple[int, ...]
    physical_stride: tuple[int, ...]
    storage_numel: int


@dataclass(frozen=True)
class DtypePlaneSpec:
    plane_id: PlaneId
    dtype: torch.dtype
    storage_numel: int
    placements: tuple[TensorPlacement, ...]


@dataclass(frozen=True)
class TransferUnitSpec:
    unit_id: str
    unit_kind: UnitKind
    bindings: tuple[TensorBindingSpec, ...]
    planes: tuple[DtypePlaneSpec, ...]

    def __post_init__(self) -> None:
        _require_name(self.unit_id, "transfer unit_id")
        try:
            object.__setattr__(self, "unit_kind", UnitKind(self.unit_kind))
        except ValueError as exc:
            raise TransferPlanError(f"unknown transfer unit kind {self.unit_kind!r}") from exc

    @property
    def tensor_ids(self) -> tuple[str, ...]:
        return tuple(binding.tensor_id for binding in self.bindings)


@dataclass(frozen=True, slots=True)
class ModuleUnitBinding:
    module_path: TargetModulePath
    unit_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "module_path", _target_module_path(self.module_path))
        _require_name(self.unit_id, "execution binding unit_id")


@dataclass(frozen=True, slots=True)
class TransferPlan:
    plan_id: str
    plan_kind: TransferPlanKind
    unit_ids: tuple[str, ...]
    execution_bindings: tuple[ModuleUnitBinding, ...]
    exact_coverage_digest: str

    def __post_init__(self) -> None:
        _require_name(self.plan_id, "plan_id")
        try:
            object.__setattr__(self, "plan_kind", TransferPlanKind(self.plan_kind))
        except ValueError as exc:
            raise TransferPlanError(f"unknown transfer plan kind {self.plan_kind!r}") from exc
        if not self.unit_ids:
            raise TransferPlanError(f"transfer plan {self.plan_id!r} contains no units")
        for unit_id in self.unit_ids:
            _require_name(unit_id, "plan unit_id")
        if len(self.unit_ids) != len(set(self.unit_ids)):
            raise TransferPlanError(f"transfer plan {self.plan_id!r} contains duplicate unit IDs")
        if not self.exact_coverage_digest:
            raise TransferPlanError(f"transfer plan {self.plan_id!r} has an empty exact-coverage digest")


def _destination_payload(destination: BindingDestination) -> dict[str, str]:
    return {
        "module_path": str(destination.module_path),
        "attribute_name": destination.attribute_name,
        "state_kind": destination.state_kind.value,
    }


def _binding_payload(binding: TensorBindingSpec) -> dict[str, object]:
    return {
        "tensor_id": binding.tensor_id,
        "destination": _destination_payload(binding.destination),
    }


def _unit_payload(unit: TransferUnitSpec) -> dict[str, object]:
    return {
        "unit_id": unit.unit_id,
        "unit_kind": unit.unit_kind.value,
        "bindings": [_binding_payload(binding) for binding in unit.bindings],
        "planes": [
            {
                "plane_id": str(plane.plane_id),
                "dtype": str(plane.dtype),
                "storage_numel": plane.storage_numel,
                "placements": [
                    {
                        "tensor_id": placement.tensor_id,
                        "offset_numel": placement.offset_numel,
                        "logical_shape": placement.logical_shape,
                        "physical_stride": placement.physical_stride,
                        "storage_numel": placement.storage_numel,
                    }
                    for placement in plane.placements
                ],
            }
            for plane in unit.planes
        ],
    }


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def compute_exact_coverage_digest(
    unit_ids: tuple[str, ...],
    execution_bindings: tuple[ModuleUnitBinding, ...],
    units: tuple[TransferUnitSpec, ...],
) -> str:
    """Commit to one plan's ordered units, tensor coverage, and execution map."""

    unit_by_id = {unit.unit_id: unit for unit in units}
    try:
        selected = [unit_by_id[unit_id] for unit_id in unit_ids]
    except KeyError as exc:
        raise TransferPlanError(f"transfer plan references unknown unit {exc.args[0]!r}") from exc
    return _canonical_digest(
        {
            "unit_ids": list(unit_ids),
            "coverage": [
                {
                    "unit_id": unit.unit_id,
                    "bindings": [_binding_payload(binding) for binding in unit.bindings],
                }
                for unit in selected
            ],
            "execution_bindings": [
                {
                    "module_path": str(binding.module_path),
                    "unit_id": binding.unit_id,
                }
                for binding in execution_bindings
            ],
        }
    )


def compute_transfer_catalog_digest(
    units: tuple[TransferUnitSpec, ...],
    plans: tuple[TransferPlan, ...],
) -> str:
    """Return the deterministic consumer-side digest for units and plans."""

    return _canonical_digest(
        {
            "units": [_unit_payload(unit) for unit in units],
            "plans": [
                {
                    "plan_id": plan.plan_id,
                    "plan_kind": plan.plan_kind.value,
                    "unit_ids": list(plan.unit_ids),
                    "execution_bindings": [
                        {
                            "module_path": str(binding.module_path),
                            "unit_id": binding.unit_id,
                        }
                        for binding in plan.execution_bindings
                    ],
                    "exact_coverage_digest": plan.exact_coverage_digest,
                }
                for plan in plans
            ],
        }
    )


def _module_owns_destination(
    module_path: TargetModulePath,
    destination: BindingDestination,
) -> bool:
    if module_path == ".":
        return True
    return destination.module_path == module_path or str(destination.module_path).startswith(f"{module_path}.")


@dataclass(frozen=True)
class TransferCatalog:
    artifact_compatibility_digest: str
    transfer_catalog_digest: str
    units: tuple[TransferUnitSpec, ...]
    plans: tuple[TransferPlan, ...]

    def __post_init__(self) -> None:
        if not self.artifact_compatibility_digest:
            raise TransferPlanError("artifact compatibility digest must not be empty")
        if not self.transfer_catalog_digest:
            raise TransferPlanError("transfer catalog digest must not be empty")
        unit_ids = [unit.unit_id for unit in self.units]
        if len(unit_ids) != len(set(unit_ids)):
            raise TransferPlanError("transfer catalog contains duplicate unit IDs")
        if not self.units:
            raise TransferPlanError("transfer catalog contains no units")
        for unit in self.units:
            if not unit.bindings:
                raise TransferPlanError(f"transfer unit {unit.unit_id!r} contains no tensor bindings")
            binding_ids = [binding.tensor_id for binding in unit.bindings]
            if len(binding_ids) != len(set(binding_ids)):
                raise TransferPlanError(f"transfer unit {unit.unit_id!r} contains duplicate tensor IDs")
            destinations = [binding.destination for binding in unit.bindings]
            if len(destinations) != len(set(destinations)):
                raise TransferPlanError(f"transfer unit {unit.unit_id!r} contains duplicate destinations")
            plane_ids = [str(plane.plane_id) for plane in unit.planes]
            if any(not plane_id for plane_id in plane_ids):
                raise TransferPlanError(f"transfer unit {unit.unit_id!r} has an empty plane ID")
            if len(plane_ids) != len(set(plane_ids)):
                raise TransferPlanError(f"transfer unit {unit.unit_id!r} contains duplicate plane IDs")
            placement_ids = [placement.tensor_id for plane in unit.planes for placement in plane.placements]
            if len(placement_ids) != len(set(placement_ids)):
                raise TransferPlanError(f"transfer unit {unit.unit_id!r} contains duplicate tensor placements")
            if set(placement_ids) != set(binding_ids):
                raise TransferPlanError(f"transfer unit {unit.unit_id!r} binding and placement tensor IDs differ")

        if not self.plans:
            raise TransferPlanError("transfer catalog contains no exact-coverage plans")
        plan_ids = [plan.plan_id for plan in self.plans]
        if len(plan_ids) != len(set(plan_ids)):
            raise TransferPlanError("transfer catalog contains duplicate plan IDs")
        plan_kinds = [plan.plan_kind for plan in self.plans]
        if len(plan_kinds) != len(set(plan_kinds)):
            raise TransferPlanError("transfer catalog contains duplicate transfer plan kinds")

        unit_by_id = {unit.unit_id: unit for unit in self.units}
        expected_coverage: dict[BindingDestination, str] | None = None
        for plan in self.plans:
            unknown = [unit_id for unit_id in plan.unit_ids if unit_id not in unit_by_id]
            if unknown:
                raise TransferPlanError(f"transfer plan {plan.plan_id!r} references unknown units {unknown}")
            coverage: dict[BindingDestination, str] = {}
            tensor_destinations: dict[str, BindingDestination] = {}
            for unit_id in plan.unit_ids:
                for binding in unit_by_id[unit_id].bindings:
                    if binding.destination in coverage:
                        raise TransferPlanError(
                            f"transfer plan {plan.plan_id!r} maps destination {binding.destination!r} more than once"
                        )
                    coverage[binding.destination] = binding.tensor_id
                    if binding.tensor_id in tensor_destinations:
                        raise TransferPlanError(
                            f"transfer plan {plan.plan_id!r} maps tensor {binding.tensor_id!r} more than once"
                        )
                    tensor_destinations[binding.tensor_id] = binding.destination
            if expected_coverage is None:
                expected_coverage = coverage
            elif coverage != expected_coverage:
                raise TransferPlanError(
                    f"transfer plan {plan.plan_id!r} does not provide the catalog's exact tensor/destination coverage"
                )

            execution_paths = [binding.module_path for binding in plan.execution_bindings]
            execution_unit_ids = [binding.unit_id for binding in plan.execution_bindings]
            if len(execution_paths) != len(set(execution_paths)):
                raise TransferPlanError(f"transfer plan {plan.plan_id!r} contains duplicate execution module paths")
            if len(execution_unit_ids) != len(set(execution_unit_ids)):
                raise TransferPlanError(f"transfer plan {plan.plan_id!r} contains duplicate execution unit IDs")
            if any(unit_id not in plan.unit_ids for unit_id in execution_unit_ids):
                raise TransferPlanError(f"transfer plan {plan.plan_id!r} binds a unit outside the plan")
            expected_execution_units = {
                unit_id for unit_id in plan.unit_ids if unit_by_id[unit_id].unit_kind is not UnitKind.RESIDENT
            }
            if set(execution_unit_ids) != expected_execution_units:
                raise TransferPlanError(
                    f"transfer plan {plan.plan_id!r} execution bindings do not exactly cover non-resident units"
                )
            for execution in plan.execution_bindings:
                unit = unit_by_id[execution.unit_id]
                if any(
                    not _module_owns_destination(execution.module_path, binding.destination)
                    for binding in unit.bindings
                ):
                    raise TransferPlanError(
                        f"execution module {execution.module_path!r} does not own every destination in "
                        f"unit {execution.unit_id!r}"
                    )
            expected_plan_digest = compute_exact_coverage_digest(
                plan.unit_ids,
                plan.execution_bindings,
                self.units,
            )
            if plan.exact_coverage_digest != expected_plan_digest:
                raise TransferPlanError(f"transfer plan {plan.plan_id!r} exact-coverage digest mismatch")

        expected_catalog_digest = compute_transfer_catalog_digest(self.units, self.plans)
        if self.transfer_catalog_digest != expected_catalog_digest:
            raise TransferPlanError("transfer catalog digest mismatch")

    def unit(self, unit_id: str) -> TransferUnitSpec:
        for unit in self.units:
            if unit.unit_id == unit_id:
                return unit
        raise TransferPlanError(f"unknown transfer unit {unit_id!r}")

    def plan(self, plan_id: str) -> TransferPlan:
        for plan in self.plans:
            if plan.plan_id == plan_id:
                return plan
        raise TransferPlanError(f"unknown transfer plan {plan_id!r}")

    def plan_for_kind(self, plan_kind: TransferPlanKind) -> TransferPlan:
        try:
            normalized = TransferPlanKind(plan_kind)
        except ValueError as exc:
            raise TransferPlanError(f"unknown transfer plan kind {plan_kind!r}") from exc
        matches = [plan for plan in self.plans if plan.plan_kind is normalized]
        if not matches:
            raise TransferPlanError(f"transfer plan kind {normalized.value!r} is unavailable")
        if len(matches) != 1:
            raise TransferPlanError(f"transfer plan kind {normalized.value!r} is ambiguous")
        return matches[0]


def allocate_host_planes(
    unit: TransferUnitSpec,
    *,
    pin_memory: bool = False,
) -> dict[PlaneId, torch.Tensor]:
    return {
        plane.plane_id: torch.empty(
            plane.storage_numel,
            dtype=plane.dtype,
            device="cpu",
            pin_memory=pin_memory,
        )
        for plane in unit.planes
    }


def validate_plane_buffers(
    unit: TransferUnitSpec,
    buffers: Mapping[PlaneId, torch.Tensor],
    *,
    device_type: str | None,
) -> None:
    expected = {plane.plane_id for plane in unit.planes}
    actual = set(buffers)
    if actual != expected:
        raise TransferPlanError(
            f"plane keys for {unit.unit_id!r} differ: "
            f"missing={sorted(map(str, expected - actual))}, "
            f"extra={sorted(map(str, actual - expected))}"
        )
    for plane in unit.planes:
        tensor = buffers[plane.plane_id]
        if device_type is not None and tensor.device.type != device_type:
            raise TransferPlanError(f"plane {plane.plane_id!r} must be on {device_type}, got {tensor.device}")
        if tensor.dtype is not plane.dtype:
            raise TransferPlanError(
                f"plane {plane.plane_id!r} dtype mismatch: expected {plane.dtype}, got {tensor.dtype}"
            )
        if tensor.ndim != 1 or not tensor.is_contiguous():
            raise TransferPlanError(f"plane {plane.plane_id!r} must be a contiguous one-dimensional tensor")
        if tensor.numel() < plane.storage_numel:
            raise TransferPlanError(
                f"plane {plane.plane_id!r} has {tensor.numel()} elements; requires {plane.storage_numel}"
            )


def tensor_views_for_unit(
    unit: TransferUnitSpec,
    buffers: Mapping[PlaneId, torch.Tensor],
    *,
    device_type: str | None,
) -> Mapping[str, torch.Tensor]:
    validate_plane_buffers(unit, buffers, device_type=device_type)
    views: dict[str, torch.Tensor] = {}
    for plane in unit.planes:
        storage = buffers[plane.plane_id]
        for placement in plane.placements:
            span = storage[placement.offset_numel : placement.offset_numel + placement.storage_numel]
            views[placement.tensor_id] = torch.as_strided(
                span,
                size=placement.logical_shape,
                stride=placement.physical_stride,
            )
    return MappingProxyType(views)


__all__ = [
    "BindingDestination",
    "DtypePlaneSpec",
    "ModuleStateKind",
    "ModuleUnitBinding",
    "PlaneId",
    "TargetModulePath",
    "TensorBindingSpec",
    "TensorPlacement",
    "TransferCatalog",
    "TransferPlan",
    "TransferPlanError",
    "TransferPlanKind",
    "TransferUnitSpec",
    "UnitKind",
    "allocate_host_planes",
    "compute_exact_coverage_digest",
    "compute_transfer_catalog_digest",
    "tensor_views_for_unit",
    "validate_plane_buffers",
]
