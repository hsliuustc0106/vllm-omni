# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Declarative weight-format contracts used by diffusion integration.

These types intentionally live outside :mod:`host_weight_runtime`.  The core
stores opaque format metadata; only the locally selected, allowlisted adapter
is allowed to interpret it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Literal, Protocol, TypeAlias

import torch

from ..transfer import (
    BindingDestination,
    ModuleStateKind,
    TargetModulePath,
    TensorBindingSpec,
)

JSONScalar: TypeAlias = None | bool | int | float | str
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
FinalizedTensorRole: TypeAlias = Literal["parameter", "persistent_buffer", "quant_metadata"]
FORMAT_BINDING_RECIPE_SCHEMA_VERSION = 2


class FormatContractError(ValueError):
    """Raised when trusted format metadata is malformed or incompatible."""


class FormatTensorRole(str, Enum):
    WEIGHT = "weight"
    WEIGHT_SCALE = "weight_scale"
    BIAS = "bias"
    IMMUTABLE_AUXILIARY = "immutable_auxiliary"


def _require_name(value: object, label: str, *, allow_root: bool = False) -> str:
    if not isinstance(value, str):
        raise FormatContractError(f"{label} must be a string")
    if not value:
        if allow_root:
            return value
        raise FormatContractError(f"{label} must not be empty")
    if value.strip() != value or value.startswith(".") or value.endswith(".") or ".." in value:
        raise FormatContractError(f"{label} is not canonical: {value!r}")
    if "\x00" in value:
        raise FormatContractError(f"{label} contains a NUL byte")
    return value


def _normalize_json(value: object, label: str = "JSON value") -> JSONValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
            raise FormatContractError(f"{label} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, JSONValue] = {}
        for key in sorted(value):
            if not isinstance(key, str) or not key:
                raise FormatContractError(f"{label} keys must be non-empty strings")
            normalized[key] = _normalize_json(value[key], f"{label}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item, f"{label}[]") for item in value]
    raise FormatContractError(f"{label} contains unsupported type {type(value).__name__}")


def _freeze_json(value: object, label: str = "JSON value") -> object:
    normalized = _normalize_json(value, label)
    if isinstance(normalized, dict):
        return MappingProxyType({key: _freeze_json(item, f"{label}.{key}") for key, item in normalized.items()})
    if isinstance(normalized, list):
        return tuple(_freeze_json(item, f"{label}[]") for item in normalized)
    return normalized


def _thaw_json(value: object) -> JSONValue:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value  # type: ignore[return-value]


def canonical_json(value: object) -> str:
    return json.dumps(
        _normalize_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _normalize_layer_tensor_binding(
    binding: object,
) -> tuple[FormatTensorRole, BindingDestination]:
    try:
        role = FormatTensorRole(getattr(binding, "role"))
    except (AttributeError, ValueError) as exc:
        raise FormatContractError("layer tensor binding contains an unknown enum value") from exc
    destination = getattr(binding, "destination", None)
    if not isinstance(destination, BindingDestination):
        raise FormatContractError("layer tensor binding requires one canonical BindingDestination")
    return role, destination


@dataclass(frozen=True, slots=True)
class RequiredLayerTensorBinding:
    """Required tensor destination relative to its layer module path."""

    role: FormatTensorRole
    destination: BindingDestination
    tensor_id: str

    def __post_init__(self) -> None:
        role, destination = _normalize_layer_tensor_binding(self)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "destination", destination)
        _require_name(self.tensor_id, "tensor_id")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "binding_kind": "required",
            "destination": self.destination.to_dict(),
            "role": self.role.value,
            "tensor_id": self.tensor_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> RequiredLayerTensorBinding:
        _require_exact_keys(
            value,
            {"binding_kind", "role", "destination", "tensor_id"},
            "required layer tensor binding",
        )
        if value["binding_kind"] != "required":
            raise FormatContractError("required layer tensor binding has the wrong binding_kind")
        destination = value["destination"]
        if not isinstance(destination, Mapping):
            raise FormatContractError("required layer destination must be an object")
        return cls(
            role=FormatTensorRole(str(value["role"])),
            destination=BindingDestination.from_dict(destination),
            tensor_id=_require_name(value["tensor_id"], "tensor_id"),
        )


@dataclass(frozen=True, slots=True)
class OptionalLayerTensorBinding:
    """Declared optional destination; ``None`` consumes no artifact tensor."""

    role: FormatTensorRole
    destination: BindingDestination
    tensor_id: str | None

    def __post_init__(self) -> None:
        role, destination = _normalize_layer_tensor_binding(self)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "destination", destination)
        if self.tensor_id is not None:
            _require_name(self.tensor_id, "tensor_id")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "binding_kind": "optional",
            "destination": self.destination.to_dict(),
            "role": self.role.value,
            "tensor_id": self.tensor_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> OptionalLayerTensorBinding:
        _require_exact_keys(
            value,
            {"binding_kind", "role", "destination", "tensor_id"},
            "optional layer tensor binding",
        )
        if value["binding_kind"] != "optional":
            raise FormatContractError("optional layer tensor binding has the wrong binding_kind")
        destination = value["destination"]
        if not isinstance(destination, Mapping):
            raise FormatContractError("optional layer destination must be an object")
        raw_tensor_id = value["tensor_id"]
        if raw_tensor_id is not None:
            raw_tensor_id = _require_name(raw_tensor_id, "tensor_id")
        return cls(
            role=FormatTensorRole(str(value["role"])),
            destination=BindingDestination.from_dict(destination),
            tensor_id=raw_tensor_id,
        )


LayerTensorBinding: TypeAlias = RequiredLayerTensorBinding | OptionalLayerTensorBinding


def _layer_tensor_binding_from_dict(value: Mapping[str, object]) -> LayerTensorBinding:
    binding_kind = value.get("binding_kind")
    if binding_kind == "required":
        return RequiredLayerTensorBinding.from_dict(value)
    if binding_kind == "optional":
        return OptionalLayerTensorBinding.from_dict(value)
    raise FormatContractError("layer tensor binding_kind must be exactly 'required' or 'optional'")


@dataclass(frozen=True, slots=True)
class LayerFormatSpec:
    module_path: TargetModulePath
    quant_method_id: str
    kernel_id: str
    tensor_bindings: tuple[LayerTensorBinding, ...]
    scalar_state: Mapping[str, JSONValue]

    def __post_init__(self) -> None:
        module_path = str(self.module_path)
        if module_path != ".":
            _require_name(module_path, "layer module_path")
        object.__setattr__(self, "module_path", TargetModulePath(module_path))
        _require_name(self.quant_method_id, "quant_method_id")
        _require_name(self.kernel_id, "kernel_id")
        bindings = tuple(self.tensor_bindings)
        if not bindings:
            raise FormatContractError(f"layer {self.module_path!r} has no tensor bindings")
        if not all(
            isinstance(binding, (RequiredLayerTensorBinding, OptionalLayerTensorBinding)) for binding in bindings
        ):
            raise FormatContractError(f"layer {self.module_path!r} contains an unsupported tensor binding variant")
        object.__setattr__(self, "tensor_bindings", bindings)
        roles = [binding.role for binding in bindings]
        for required in (FormatTensorRole.WEIGHT, FormatTensorRole.WEIGHT_SCALE):
            if roles.count(required) != 1:
                raise FormatContractError(f"layer {self.module_path!r} must bind {required.value!r} exactly once")
            binding = next(item for item in bindings if item.role is required)
            if not isinstance(binding, RequiredLayerTensorBinding):
                raise FormatContractError(f"layer {self.module_path!r} must declare {required.value!r} as required")
        if roles.count(FormatTensorRole.BIAS) != 1:
            raise FormatContractError(f"layer {self.module_path!r} must declare exactly one bias slot")
        if any(str(binding.destination.module_path) != module_path for binding in bindings):
            raise FormatContractError(f"layer {self.module_path!r} contains a destination owned by another module")
        attributes = [binding.destination.attribute_name for binding in bindings]
        tensor_ids = [binding.tensor_id for binding in bindings if binding.tensor_id is not None]
        if len(set(attributes)) != len(attributes):
            raise FormatContractError(f"layer {self.module_path!r} has duplicate destination attributes")
        if len(set(tensor_ids)) != len(tensor_ids):
            raise FormatContractError(f"layer {self.module_path!r} has duplicate tensor IDs")
        normalized = _freeze_json(self.scalar_state, "scalar_state")
        assert isinstance(normalized, Mapping)
        object.__setattr__(self, "scalar_state", normalized)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "kernel_id": self.kernel_id,
            "module_path": self.module_path,
            "quant_method_id": self.quant_method_id,
            "scalar_state": _thaw_json(self.scalar_state),
            "tensor_bindings": [binding.to_dict() for binding in self.tensor_bindings],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> LayerFormatSpec:
        _require_exact_keys(
            value,
            {"module_path", "quant_method_id", "kernel_id", "tensor_bindings", "scalar_state"},
            "layer format spec",
        )
        raw_bindings = value["tensor_bindings"]
        raw_state = value["scalar_state"]
        if not isinstance(raw_bindings, (list, tuple)) or not all(isinstance(item, Mapping) for item in raw_bindings):
            raise FormatContractError("tensor_bindings must be a JSON array of objects")
        if not isinstance(raw_state, Mapping):
            raise FormatContractError("scalar_state must be a JSON object")
        return cls(
            module_path=TargetModulePath(str(value["module_path"])),
            quant_method_id=str(value["quant_method_id"]),
            kernel_id=str(value["kernel_id"]),
            tensor_bindings=tuple(_layer_tensor_binding_from_dict(item) for item in raw_bindings),
            scalar_state=raw_state,
        )


@dataclass(frozen=True, slots=True)
class FormatBindingRecipe:
    schema_version: int
    format_plan_digest: str
    target_module_type_id: str
    layers: tuple[LayerFormatSpec, ...]
    non_layer_bindings: tuple[TensorBindingSpec, ...]

    def __post_init__(self) -> None:
        if self.schema_version != FORMAT_BINDING_RECIPE_SCHEMA_VERSION:
            raise FormatContractError(
                f"unsupported format recipe schema {self.schema_version}; "
                f"expected {FORMAT_BINDING_RECIPE_SCHEMA_VERSION}"
            )
        if len(self.format_plan_digest) != 64 or any(
            character not in "0123456789abcdef" for character in self.format_plan_digest
        ):
            raise FormatContractError("format_plan_digest must be a lowercase SHA-256 digest")
        _require_name(self.target_module_type_id, "target_module_type_id")
        layers = tuple(self.layers)
        non_layer = tuple(self.non_layer_bindings)
        object.__setattr__(self, "layers", layers)
        object.__setattr__(self, "non_layer_bindings", non_layer)
        layer_paths = [layer.module_path for layer in layers]
        if len(set(layer_paths)) != len(layer_paths):
            raise FormatContractError("format recipe contains duplicate layer paths")
        references = [
            binding.tensor_id for layer in layers for binding in layer.tensor_bindings if binding.tensor_id is not None
        ] + [binding.tensor_id for binding in non_layer]
        if not references:
            raise FormatContractError("format recipe contains no tensor bindings")
        if len(set(references)) != len(references):
            raise FormatContractError("format recipe references a tensor ID more than once")
        destinations = [
            (str(binding.destination.module_path), binding.destination.attribute_name) for binding in non_layer
        ]
        if len(set(destinations)) != len(destinations):
            raise FormatContractError("format recipe has duplicate non-layer destinations")

    @property
    def tensor_ids(self) -> tuple[str, ...]:
        return tuple(
            [
                binding.tensor_id
                for layer in self.layers
                for binding in layer.tensor_bindings
                if binding.tensor_id is not None
            ]
            + [binding.tensor_id for binding in self.non_layer_bindings]
        )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "format_plan_digest": self.format_plan_digest,
            "layers": [layer.to_dict() for layer in self.layers],
            "non_layer_bindings": [binding.to_dict() for binding in self.non_layer_bindings],
            "schema_version": self.schema_version,
            "target_module_type_id": self.target_module_type_id,
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> FormatBindingRecipe:
        _require_exact_keys(
            value,
            {"schema_version", "format_plan_digest", "target_module_type_id", "layers", "non_layer_bindings"},
            "format binding recipe",
        )
        raw_layers = value["layers"]
        raw_non_layer = value["non_layer_bindings"]
        if not isinstance(raw_layers, (list, tuple)) or not all(isinstance(item, Mapping) for item in raw_layers):
            raise FormatContractError("layers must be a JSON array of objects")
        if not isinstance(raw_non_layer, (list, tuple)) or not all(isinstance(item, Mapping) for item in raw_non_layer):
            raise FormatContractError("non_layer_bindings must be a JSON array of objects")
        return cls(
            schema_version=int(value["schema_version"]),
            format_plan_digest=str(value["format_plan_digest"]),
            target_module_type_id=str(value["target_module_type_id"]),
            layers=tuple(LayerFormatSpec.from_dict(item) for item in raw_layers),
            non_layer_bindings=tuple(TensorBindingSpec.from_dict(item) for item in raw_non_layer),
        )

    @classmethod
    def from_json(cls, payload: str) -> FormatBindingRecipe:
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise FormatContractError("format recipe is not valid JSON") from exc
        if not isinstance(value, Mapping):
            raise FormatContractError("format recipe must be a JSON object")
        recipe = cls.from_dict(value)
        if recipe.to_json() != payload:
            raise FormatContractError("format recipe JSON is not canonical")
        return recipe


@dataclass(frozen=True, slots=True)
class FinalizedTensor:
    tensor_id: str
    role: FinalizedTensorRole
    tensor: torch.Tensor

    def __post_init__(self) -> None:
        _require_name(self.tensor_id, "tensor_id")
        if self.role not in {"parameter", "persistent_buffer", "quant_metadata"}:
            raise FormatContractError(f"unsupported finalized tensor role {self.role!r}")
        if not isinstance(self.tensor, torch.Tensor):
            raise FormatContractError("finalized tensor payload must be a torch.Tensor")
        if self.tensor.is_meta or self.tensor.device.type != "cpu":
            raise FormatContractError(f"finalized tensor {self.tensor_id!r} must have CPU storage")


@dataclass(frozen=True, slots=True)
class FinalizedTensorSet:
    tensors: tuple[FinalizedTensor, ...]
    binding_recipe: FormatBindingRecipe

    def __post_init__(self) -> None:
        tensors = tuple(self.tensors)
        object.__setattr__(self, "tensors", tensors)
        tensor_ids = [tensor.tensor_id for tensor in tensors]
        if not tensors or len(set(tensor_ids)) != len(tensor_ids):
            raise FormatContractError("finalized tensor set must contain unique tensors")
        if set(tensor_ids) != set(self.binding_recipe.tensor_ids):
            missing = sorted(set(self.binding_recipe.tensor_ids) - set(tensor_ids))
            extra = sorted(set(tensor_ids) - set(self.binding_recipe.tensor_ids))
            raise FormatContractError(
                f"finalized tensors do not match binding recipe; missing={missing}, extra={extra}"
            )

    @property
    def format_metadata(self) -> Mapping[str, JSONValue]:
        frozen = _freeze_json(self.binding_recipe.to_dict(), "format_metadata")
        assert isinstance(frozen, Mapping)
        return frozen  # type: ignore[return-value]


class ArtifactFormatExporter(Protocol):
    """Builder-only port that finalizes one trusted runtime representation."""

    @property
    def descriptor(self) -> object: ...

    def finalize_for_artifact(
        self,
        loaded_module: torch.nn.Module,
    ) -> FinalizedTensorSet: ...


class ConsumerFormatAdapter(Protocol):
    """Warm-consumer port that restores declarative format structure only.

    The independent core stores ``format_metadata`` as inert JSON.  Only a
    locally selected adapter may export finalized tensors or interpret that
    metadata while hydrating a consumer skeleton.
    """

    @property
    def descriptor(self) -> object: ...

    def prepare_consumer_structure(
        self,
        target_module: torch.nn.Module,
        manifest: object,
    ) -> FormatBindingRecipe: ...


class WeightFormatAdapter(ArtifactFormatExporter, ConsumerFormatAdapter, Protocol):
    """Compatibility composition of the two narrow trusted format ports."""


def _require_exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise FormatContractError(
            f"{label} keys do not match schema; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


__all__ = [
    "ArtifactFormatExporter",
    "ConsumerFormatAdapter",
    "FinalizedTensor",
    "FinalizedTensorRole",
    "FinalizedTensorSet",
    "FORMAT_BINDING_RECIPE_SCHEMA_VERSION",
    "FormatBindingRecipe",
    "FormatContractError",
    "FormatTensorRole",
    "JSONValue",
    "LayerFormatSpec",
    "LayerTensorBinding",
    "BindingDestination",
    "ModuleStateKind",
    "OptionalLayerTensorBinding",
    "RequiredLayerTensorBinding",
    "TensorBindingSpec",
    "TargetModulePath",
    "WeightFormatAdapter",
    "canonical_digest",
    "canonical_json",
]
