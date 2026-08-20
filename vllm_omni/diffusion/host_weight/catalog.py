# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-facing ownership catalog for immutable host weights.

An artifact unit is the atomic publication and fallback boundary.  A transfer
unit is a consumer-facing subset copied together (for example one DiT block).
The two are intentionally distinct: several model-, layer-, or DLO-level
transfer units may select tensors from the same immutable artifact.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType

import torch
from torch import nn


class CatalogError(ValueError):
    """Raised when module declarations do not form an unambiguous catalog."""


class TensorRole(str, Enum):
    PARAMETER = "parameter"
    PERSISTENT_BUFFER = "persistent_buffer"


def _join_path(prefix: str, suffix: str) -> str:
    if not prefix:
        return suffix
    if not suffix:
        return prefix
    return f"{prefix}.{suffix}"


def _validate_path(path: str, *, label: str, allow_root: bool = True) -> None:
    if not isinstance(path, str):
        raise CatalogError(f"{label} must be a string, got {type(path)!r}")
    if not path:
        if allow_root:
            return
        raise CatalogError(f"{label} must not be empty")
    if path.startswith(".") or path.endswith(".") or ".." in path:
        raise CatalogError(f"{label} is not a canonical module path: {path!r}")


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    to_local = getattr(tensor, "to_local", None)
    return to_local() if callable(to_local) else tensor


def storage_numel(shape: Sequence[int], stride: Sequence[int]) -> int:
    """Return the physical span required by a strided tensor at offset zero."""
    if len(shape) != len(stride):
        raise CatalogError(f"shape/stride rank mismatch: shape={tuple(shape)}, stride={tuple(stride)}")
    if any(size < 0 for size in shape):
        raise CatalogError(f"negative tensor dimension in shape {tuple(shape)}")
    if any(size == 0 for size in shape):
        return 0
    for size, axis_stride in zip(shape, stride, strict=True):
        if axis_stride < 0 or (size > 1 and axis_stride == 0):
            raise CatalogError(
                "overlapping or negative-stride tensors are outside the v1 host-weight contract: "
                f"shape={tuple(shape)}, stride={tuple(stride)}"
            )
    return 1 + sum((size - 1) * axis_stride for size, axis_stride in zip(shape, stride, strict=True))


@dataclass(frozen=True)
class ModuleScope:
    """A module together with its canonical path from the pipeline root."""

    module_path: str
    module: nn.Module = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_path(self.module_path, label="module_path")
        if not isinstance(self.module, nn.Module):
            raise CatalogError(f"module must be nn.Module, got {type(self.module)!r}")

    @classmethod
    def from_pipeline(cls, pipeline: nn.Module, module_path: str) -> ModuleScope:
        _validate_path(module_path, label="module_path")
        module = pipeline if not module_path else pipeline.get_submodule(module_path)
        return cls(module_path=module_path, module=module)


@dataclass(frozen=True)
class ArtifactUnitDeclaration:
    """Scopes whose tensors are atomically published as one artifact."""

    unit_id: str
    scopes: tuple[ModuleScope, ...]

    def __post_init__(self) -> None:
        _validate_path(self.unit_id, label="artifact unit_id", allow_root=False)
        if not self.scopes:
            raise CatalogError(f"artifact unit {self.unit_id!r} has no module scopes")


@dataclass(frozen=True)
class TransferUnitDeclaration:
    """Scopes copied together by one offloader operation."""

    unit_id: str
    artifact_unit_id: str
    scopes: tuple[ModuleScope, ...]

    def __post_init__(self) -> None:
        _validate_path(self.unit_id, label="transfer unit_id", allow_root=False)
        _validate_path(self.artifact_unit_id, label="artifact_unit_id", allow_root=False)
        if not self.scopes:
            raise CatalogError(f"transfer unit {self.unit_id!r} has no module scopes")


@dataclass(frozen=True)
class TensorTarget:
    """Stable runtime metadata for one parameter or persistent buffer."""

    tensor_id: str
    owner_module_path: str
    local_name: str
    role: TensorRole
    dtype: torch.dtype
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_numel: int


@dataclass(frozen=True)
class ArtifactUnit:
    unit_id: str
    scope_paths: tuple[str, ...]
    tensor_ids: tuple[str, ...]


@dataclass(frozen=True)
class TransferUnit:
    unit_id: str
    artifact_unit_id: str
    scope_paths: tuple[str, ...]
    tensor_ids: tuple[str, ...]


@dataclass(frozen=True)
class WeightCatalog:
    """Validated ownership graph shared by binders and offload consumers."""

    tensors: tuple[TensorTarget, ...]
    artifact_units: tuple[ArtifactUnit, ...]
    transfer_units: tuple[TransferUnit, ...]
    _tensor_by_id: Mapping[str, TensorTarget] = field(init=False, repr=False, compare=False)
    _artifact_by_id: Mapping[str, ArtifactUnit] = field(init=False, repr=False, compare=False)
    _transfer_by_id: Mapping[str, TransferUnit] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        tensor_by_id = {tensor.tensor_id: tensor for tensor in self.tensors}
        artifact_by_id = {unit.unit_id: unit for unit in self.artifact_units}
        transfer_by_id = {unit.unit_id: unit for unit in self.transfer_units}
        if len(tensor_by_id) != len(self.tensors):
            raise CatalogError("catalog contains duplicate tensor IDs")
        if len(artifact_by_id) != len(self.artifact_units):
            raise CatalogError("catalog contains duplicate artifact unit IDs")
        if len(transfer_by_id) != len(self.transfer_units):
            raise CatalogError("catalog contains duplicate transfer unit IDs")
        object.__setattr__(self, "_tensor_by_id", MappingProxyType(tensor_by_id))
        object.__setattr__(self, "_artifact_by_id", MappingProxyType(artifact_by_id))
        object.__setattr__(self, "_transfer_by_id", MappingProxyType(transfer_by_id))

    def tensor(self, tensor_id: str) -> TensorTarget:
        try:
            return self._tensor_by_id[tensor_id]
        except KeyError as exc:
            raise CatalogError(f"unknown tensor ID {tensor_id!r}") from exc

    def artifact(self, unit_id: str) -> ArtifactUnit:
        try:
            return self._artifact_by_id[unit_id]
        except KeyError as exc:
            raise CatalogError(f"unknown artifact unit {unit_id!r}") from exc

    def transfer(self, unit_id: str) -> TransferUnit:
        try:
            return self._transfer_by_id[unit_id]
        except KeyError as exc:
            raise CatalogError(f"unknown transfer unit {unit_id!r}") from exc


def _persistent_buffer(module: nn.Module, relative_name: str) -> bool:
    parts = relative_name.split(".")
    owner = module.get_submodule(".".join(parts[:-1])) if len(parts) > 1 else module
    return parts[-1] not in owner._non_persistent_buffers_set


def _named_parameters(module: nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
    # remove_duplicate=False is essential: v1 rejects aliases rather than
    # silently publishing one name and leaving another binding ambiguous.
    yield from module.named_parameters(recurse=True, remove_duplicate=False)


def _named_buffers(module: nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
    yield from module.named_buffers(recurse=True, remove_duplicate=False)


def _scope_tensors(scope: ModuleScope) -> Iterator[tuple[TensorTarget, torch.Tensor]]:
    entries: list[tuple[str, torch.Tensor, TensorRole]] = [
        (name, tensor, TensorRole.PARAMETER) for name, tensor in _named_parameters(scope.module)
    ]
    entries.extend(
        (name, tensor, TensorRole.PERSISTENT_BUFFER)
        for name, tensor in _named_buffers(scope.module)
        if _persistent_buffer(scope.module, name)
    )
    for relative_name, wrapped_tensor, role in entries:
        tensor = _local_tensor(wrapped_tensor)
        tensor_id = _join_path(scope.module_path, relative_name)
        owner_relative, _, local_name = relative_name.rpartition(".")
        owner_path = _join_path(scope.module_path, owner_relative)
        shape = tuple(tensor.shape)
        stride = tuple(tensor.stride())
        yield (
            TensorTarget(
                tensor_id=tensor_id,
                owner_module_path=owner_path,
                local_name=local_name,
                role=role,
                dtype=tensor.dtype,
                shape=shape,
                stride=stride,
                storage_numel=storage_numel(shape, stride),
            ),
            tensor,
        )


def _storage_identity(tensor: torch.Tensor) -> int | None:
    if tensor.is_meta or tensor.numel() == 0:
        return None
    try:
        return int(tensor.untyped_storage()._cdata)
    except (AttributeError, RuntimeError):
        return None


def _collect_declaration_tensors(
    *,
    declaration_name: str,
    scopes: Sequence[ModuleScope],
) -> tuple[list[TensorTarget], dict[str, torch.Tensor]]:
    targets: list[TensorTarget] = []
    tensors: dict[str, torch.Tensor] = {}
    object_owners: dict[int, str] = {}
    storage_owners: dict[int, str] = {}
    for scope in scopes:
        for target, tensor in _scope_tensors(scope):
            if target.tensor_id in tensors:
                raise CatalogError(
                    f"{declaration_name} selects tensor {target.tensor_id!r} more than once; "
                    "module scopes must not overlap"
                )
            object_id = id(tensor)
            if previous := object_owners.get(object_id):
                raise CatalogError(
                    f"{declaration_name} contains aliased tensor names {previous!r} and {target.tensor_id!r}"
                )
            storage_id = _storage_identity(tensor)
            if storage_id is not None and (previous := storage_owners.get(storage_id)):
                raise CatalogError(
                    f"{declaration_name} contains shared storage for {previous!r} and {target.tensor_id!r}; "
                    "storage aliases are outside the v1 contract"
                )
            object_owners[object_id] = target.tensor_id
            if storage_id is not None:
                storage_owners[storage_id] = target.tensor_id
            targets.append(target)
            tensors[target.tensor_id] = tensor
    return targets, tensors


def build_weight_catalog(
    *,
    artifact_units: Sequence[ArtifactUnitDeclaration],
    transfer_units: Sequence[TransferUnitDeclaration],
) -> WeightCatalog:
    """Compile explicit module scopes into an immutable ownership catalog."""
    if not artifact_units:
        raise CatalogError("at least one artifact unit is required")

    artifact_ids: set[str] = set()
    tensor_owners: dict[str, str] = {}
    tensor_by_id: dict[str, TensorTarget] = {}
    compiled_artifacts: list[ArtifactUnit] = []
    for declaration in artifact_units:
        if declaration.unit_id in artifact_ids:
            raise CatalogError(f"duplicate artifact unit ID {declaration.unit_id!r}")
        artifact_ids.add(declaration.unit_id)
        targets, _ = _collect_declaration_tensors(
            declaration_name=f"artifact unit {declaration.unit_id!r}",
            scopes=declaration.scopes,
        )
        if not targets:
            raise CatalogError(f"artifact unit {declaration.unit_id!r} contains no managed tensors")
        for target in targets:
            if previous_owner := tensor_owners.get(target.tensor_id):
                raise CatalogError(
                    f"tensor {target.tensor_id!r} is owned by both {previous_owner!r} and {declaration.unit_id!r}"
                )
            tensor_owners[target.tensor_id] = declaration.unit_id
            tensor_by_id[target.tensor_id] = target
        compiled_artifacts.append(
            ArtifactUnit(
                unit_id=declaration.unit_id,
                scope_paths=tuple(scope.module_path for scope in declaration.scopes),
                tensor_ids=tuple(target.tensor_id for target in targets),
            )
        )

    transfer_ids: set[str] = set()
    compiled_transfers: list[TransferUnit] = []
    for declaration in transfer_units:
        if declaration.unit_id in transfer_ids:
            raise CatalogError(f"duplicate transfer unit ID {declaration.unit_id!r}")
        transfer_ids.add(declaration.unit_id)
        if declaration.artifact_unit_id not in artifact_ids:
            raise CatalogError(
                f"transfer unit {declaration.unit_id!r} references unknown artifact {declaration.artifact_unit_id!r}"
            )
        targets, _ = _collect_declaration_tensors(
            declaration_name=f"transfer unit {declaration.unit_id!r}",
            scopes=declaration.scopes,
        )
        if not targets:
            raise CatalogError(f"transfer unit {declaration.unit_id!r} contains no managed tensors")
        for target in targets:
            owner = tensor_owners.get(target.tensor_id)
            if owner != declaration.artifact_unit_id:
                detail = "not managed by any artifact" if owner is None else f"owned by {owner!r}"
                raise CatalogError(
                    f"transfer tensor {target.tensor_id!r} is {detail}, not by {declaration.artifact_unit_id!r}"
                )
            if tensor_by_id[target.tensor_id] != target:
                raise CatalogError(f"runtime metadata changed while cataloging {target.tensor_id!r}")
        compiled_transfers.append(
            TransferUnit(
                unit_id=declaration.unit_id,
                artifact_unit_id=declaration.artifact_unit_id,
                scope_paths=tuple(scope.module_path for scope in declaration.scopes),
                tensor_ids=tuple(target.tensor_id for target in targets),
            )
        )

    return WeightCatalog(
        tensors=tuple(tensor_by_id.values()),
        artifact_units=tuple(compiled_artifacts),
        transfer_units=tuple(compiled_transfers),
    )


__all__ = [
    "ArtifactUnit",
    "ArtifactUnitDeclaration",
    "CatalogError",
    "ModuleScope",
    "TensorRole",
    "TensorTarget",
    "TransferUnit",
    "TransferUnitDeclaration",
    "WeightCatalog",
    "build_weight_catalog",
    "storage_numel",
]
