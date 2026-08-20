# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backing-independent contracts for immutable host-weight artifacts.

This module intentionally contains no diffusion-model, quantizer, offloader,
or filesystem concepts.  The manifest describes logical runtime tensors;
``BackingIndex`` is the separate, provider-owned physical description.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeAlias

import torch

if TYPE_CHECKING:
    from .artifact import ResolvedArtifact

SCHEMA_VERSION = 1
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")

JSONScalar: TypeAlias = type(None) | bool | int | float | str
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


class ContractError(ValueError):
    """Raised when an immutable runtime contract is malformed."""


class AccessFeature(str, Enum):
    COMPLETE_TENSOR_READ = "complete_tensor_read"
    SHARED_PAGES = "shared_pages"


class HostCopyMode(str, Enum):
    """Lifetime contract for copying an opened host-weight view.

    V1 exposes only complete-tensor, synchronous copies.  Adding an
    asynchronous mode requires a separate request and lifetime ABI rather
    than another backing feature flag.
    """

    SYNCHRONOUS = "synchronous"


class BackingKind(str, Enum):
    LOADED_TENSOR = "loaded_tensor"
    RUNTIME_MMAP = "runtime_mmap"
    CHECKPOINT_MMAP = "checkpoint_mmap"


class TensorRole(str, Enum):
    PARAMETER = "parameter"
    PERSISTENT_BUFFER = "persistent_buffer"
    QUANT_METADATA = "quant_metadata"


_DTYPE_ITEM_SIZES: dict[str, int] = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "float8_e4m3fn": 1,
    "float8_e4m3fnuz": 1,
    "float8_e5m2": 1,
    "float8_e5m2fnuz": 1,
    "int16": 2,
    "float16": 2,
    "bfloat16": 2,
    "int32": 4,
    "float32": 4,
    "int64": 8,
    "float64": 8,
}


def dtype_item_size(dtype: str) -> int:
    try:
        return _DTYPE_ITEM_SIZES[dtype]
    except KeyError as exc:
        raise ContractError(f"unsupported runtime dtype: {dtype!r}") from exc


def torch_dtype_name(dtype: torch.dtype) -> str:
    name = str(dtype)
    if not name.startswith("torch."):
        raise ContractError(f"cannot canonicalize torch dtype {dtype!r}")
    result = name.removeprefix("torch.")
    dtype_item_size(result)
    return result


def _require_identifier(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value or value.strip() != value or "\x00" in value:
        raise ContractError(f"{field_name} must be a non-empty canonical string")


def _require_digest(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not _DIGEST_RE.fullmatch(value):
        raise ContractError(f"{field_name} must be a lowercase SHA-256 digest")


def _freeze_json(value: object, field_name: str) -> object:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError(f"{field_name} contains a non-finite float")
        return value
    if isinstance(value, list) or isinstance(value, tuple):
        return tuple(_freeze_json(item, field_name) for item in value)
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractError(f"{field_name} contains a non-string map key")
            normalized[key] = _freeze_json(item, field_name)
        return MappingProxyType(dict(sorted(normalized.items())))
    raise ContractError(f"{field_name} contains non-JSON value {type(value).__name__}")


def _thaw_json(value: object) -> JSONValue:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value  # type: ignore[return-value]


def canonical_json(value: object) -> str:
    return json.dumps(
        _thaw_json(_freeze_json(value, "canonical value")),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_digest(value: object) -> str:
    import hashlib

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{name} must be a JSON object")
    return value


def _require_sequence(value: object, name: str) -> list[Any] | tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise ContractError(f"{name} must be a JSON array")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    value = _require_mapping(value, name)
    actual = set(value)
    if actual != expected:
        raise ContractError(
            f"{name} fields do not match schema; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


@dataclass(frozen=True, slots=True)
class TopologyCoordinate:
    axis: Literal["pp", "tp"]
    size: int
    rank: int

    def __post_init__(self) -> None:
        if self.axis not in {"pp", "tp"}:
            raise ContractError("artifact topology axis must be 'pp' or 'tp'")
        if self.size < 1:
            raise ContractError("topology size must be positive")
        if not 0 <= self.rank < self.size:
            raise ContractError("topology rank must be in [0, size)")

    def to_dict(self) -> dict[str, JSONValue]:
        return {"axis": self.axis, "size": self.size, "rank": self.rank}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TopologyCoordinate:
        _require_exact_keys(value, {"axis", "size", "rank"}, "topology coordinate")
        return cls(axis=value["axis"], size=int(value["size"]), rank=int(value["rank"]))


@dataclass(frozen=True, slots=True)
class ArtifactTopologyDescriptor:
    coordinates: tuple[TopologyCoordinate, ...] = ()

    def __post_init__(self) -> None:
        coordinates = tuple(sorted(tuple(self.coordinates), key=lambda coordinate: coordinate.axis))
        if len({coordinate.axis for coordinate in coordinates}) != len(coordinates):
            raise ContractError("artifact topology axes must be unique")
        object.__setattr__(self, "coordinates", coordinates)

    def to_dict(self) -> dict[str, JSONValue]:
        return {"coordinates": [coordinate.to_dict() for coordinate in self.coordinates]}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ArtifactTopologyDescriptor:
        _require_exact_keys(value, {"coordinates"}, "artifact topology")
        coordinates = _require_sequence(value["coordinates"], "artifact topology coordinates")
        return cls(tuple(TopologyCoordinate.from_dict(item) for item in coordinates))


@dataclass(frozen=True, slots=True)
class ProducerDescriptor:
    producer_id: str
    producer_abi: str
    semantic_fingerprint: str

    def __post_init__(self) -> None:
        _require_identifier(self.producer_id, "producer_id")
        _require_identifier(self.producer_abi, "producer_abi")
        _require_digest(self.semantic_fingerprint, "producer semantic_fingerprint")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "producer_id": self.producer_id,
            "producer_abi": self.producer_abi,
            "semantic_fingerprint": self.semantic_fingerprint,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ProducerDescriptor:
        _require_exact_keys(
            value,
            {"producer_id", "producer_abi", "semantic_fingerprint"},
            "producer descriptor",
        )
        return cls(
            producer_id=str(value["producer_id"]),
            producer_abi=str(value["producer_abi"]),
            semantic_fingerprint=str(value["semantic_fingerprint"]),
        )


def derive_weight_format_plan_digest(
    *,
    format_id: str,
    adapter_abi: str,
    semantic_fingerprint: str,
    format_recipe_schema_version: int,
    target_module_type_id: str,
    normalized_config: Mapping[str, JSONValue],
    kernel_identity: Mapping[str, JSONValue],
) -> str:
    """Derive the canonical structural plan digest for a weight format.

    The payload deliberately excludes only ``format_plan_digest`` itself.
    It includes every field that can change the finalized byte layout or the
    process-local structure needed to bind those bytes.
    """

    _require_identifier(format_id, "format_id")
    _require_identifier(adapter_abi, "adapter_abi")
    _require_digest(semantic_fingerprint, "format semantic_fingerprint")
    if type(format_recipe_schema_version) is not int or format_recipe_schema_version < 1:
        raise ContractError("format_recipe_schema_version must be a positive integer")
    _require_identifier(target_module_type_id, "target_module_type_id")
    return canonical_digest(
        {
            "format_id": format_id,
            "adapter_abi": adapter_abi,
            "semantic_fingerprint": semantic_fingerprint,
            "format_recipe_schema_version": format_recipe_schema_version,
            "target_module_type_id": target_module_type_id,
            "normalized_config": normalized_config,
            "kernel_identity": kernel_identity,
        }
    )


@dataclass(frozen=True, slots=True)
class WeightFormatDescriptor:
    format_id: str
    adapter_abi: str
    semantic_fingerprint: str
    format_plan_digest: str
    format_recipe_schema_version: int
    target_module_type_id: str
    normalized_config: Mapping[str, JSONValue] = field(default_factory=dict)
    kernel_identity: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_identifier(self.format_id, "format_id")
        _require_identifier(self.adapter_abi, "adapter_abi")
        _require_digest(self.semantic_fingerprint, "format semantic_fingerprint")
        _require_digest(self.format_plan_digest, "format_plan_digest")
        if type(self.format_recipe_schema_version) is not int or self.format_recipe_schema_version < 1:
            raise ContractError("format_recipe_schema_version must be a positive integer")
        _require_identifier(self.target_module_type_id, "target_module_type_id")
        object.__setattr__(
            self,
            "normalized_config",
            _freeze_json(self.normalized_config, "normalized_config"),
        )
        object.__setattr__(
            self,
            "kernel_identity",
            _freeze_json(self.kernel_identity, "kernel_identity"),
        )
        expected_plan_digest = derive_weight_format_plan_digest(
            format_id=self.format_id,
            adapter_abi=self.adapter_abi,
            semantic_fingerprint=self.semantic_fingerprint,
            format_recipe_schema_version=self.format_recipe_schema_version,
            target_module_type_id=self.target_module_type_id,
            normalized_config=self.normalized_config,
            kernel_identity=self.kernel_identity,
        )
        if self.format_plan_digest != expected_plan_digest:
            raise ContractError("format_plan_digest does not match the canonical weight-format plan payload")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "format_id": self.format_id,
            "adapter_abi": self.adapter_abi,
            "semantic_fingerprint": self.semantic_fingerprint,
            "format_plan_digest": self.format_plan_digest,
            "format_recipe_schema_version": self.format_recipe_schema_version,
            "target_module_type_id": self.target_module_type_id,
            "normalized_config": _thaw_json(self.normalized_config),
            "kernel_identity": _thaw_json(self.kernel_identity),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> WeightFormatDescriptor:
        _require_exact_keys(
            value,
            {
                "format_id",
                "adapter_abi",
                "semantic_fingerprint",
                "format_plan_digest",
                "format_recipe_schema_version",
                "target_module_type_id",
                "normalized_config",
                "kernel_identity",
            },
            "weight-format descriptor",
        )
        recipe_schema_version = value["format_recipe_schema_version"]
        if type(recipe_schema_version) is not int:
            raise ContractError("format_recipe_schema_version must be a positive integer")
        return cls(
            format_id=str(value["format_id"]),
            adapter_abi=str(value["adapter_abi"]),
            semantic_fingerprint=str(value["semantic_fingerprint"]),
            format_plan_digest=str(value["format_plan_digest"]),
            format_recipe_schema_version=recipe_schema_version,
            target_module_type_id=str(value["target_module_type_id"]),
            normalized_config=_require_mapping(value["normalized_config"], "normalized_config"),
            kernel_identity=_require_mapping(value["kernel_identity"], "kernel_identity"),
        )


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    """Canonical pre-build identity of one immutable artifact."""

    source_fingerprint: str
    producer: ProducerDescriptor
    weight_format: WeightFormatDescriptor
    topology: ArtifactTopologyDescriptor
    layout_abi: str
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ContractError(f"unsupported artifact schema {self.schema_version}; expected {SCHEMA_VERSION}")
        _require_digest(self.source_fingerprint, "source_fingerprint")
        _require_identifier(self.layout_abi, "layout_abi")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "source_fingerprint": self.source_fingerprint,
            "producer": self.producer.to_dict(),
            "weight_format": self.weight_format.to_dict(),
            "topology": self.topology.to_dict(),
            "layout_abi": self.layout_abi,
        }

    @property
    def artifact_key(self) -> str:
        return canonical_digest(self.to_dict())


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """Logical tensor metadata and the digest of its normalized storage span."""

    tensor_id: str
    role: TensorRole
    dtype: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_numel: int
    content_digest: str

    def __post_init__(self) -> None:
        _require_identifier(self.tensor_id, "tensor_id")
        try:
            role = TensorRole(self.role)
        except ValueError as exc:
            raise ContractError(f"unsupported tensor role: {self.role!r}") from exc
        shape = tuple(int(item) for item in self.shape)
        stride = tuple(int(item) for item in self.stride)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "stride", stride)
        dtype_item_size(self.dtype)
        _require_digest(self.content_digest, "tensor content_digest")
        if len(shape) != len(stride):
            raise ContractError("tensor shape and stride ranks differ")
        if any(dimension < 0 for dimension in shape):
            raise ContractError("tensor dimensions must be non-negative")
        if any(value < 0 for value in stride):
            raise ContractError("negative tensor strides are unsupported")
        if any(dimension > 1 and value == 0 for dimension, value in zip(shape, stride, strict=True)):
            raise ContractError("broadcast/aliased tensor storage is unsupported")
        _validate_non_overlapping_strides(shape, stride)
        expected = required_storage_numel(shape, stride)
        if self.storage_numel != expected:
            raise ContractError(
                f"tensor storage span mismatch for {self.tensor_id!r}: "
                f"expected {expected} elements, got {self.storage_numel}"
            )

    @property
    def storage_nbytes(self) -> int:
        return self.storage_numel * dtype_item_size(self.dtype)

    @property
    def numel(self) -> int:
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "tensor_id": self.tensor_id,
            "role": self.role.value,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "stride": list(self.stride),
            "storage_numel": self.storage_numel,
            "content_digest": self.content_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TensorSpec:
        _require_exact_keys(
            value,
            {
                "tensor_id",
                "role",
                "dtype",
                "shape",
                "stride",
                "storage_numel",
                "content_digest",
            },
            "tensor spec",
        )
        shape = _require_sequence(value["shape"], "tensor shape")
        stride = _require_sequence(value["stride"], "tensor stride")
        return cls(
            tensor_id=str(value["tensor_id"]),
            role=TensorRole(value["role"]),
            dtype=str(value["dtype"]),
            shape=tuple(int(item) for item in shape),
            stride=tuple(int(item) for item in stride),
            storage_numel=int(value["storage_numel"]),
            content_digest=str(value["content_digest"]),
        )


def required_storage_numel(shape: tuple[int, ...], stride: tuple[int, ...]) -> int:
    if any(dimension == 0 for dimension in shape):
        return 0
    return 1 + sum(
        (dimension - 1) * dimension_stride for dimension, dimension_stride in zip(shape, stride, strict=True)
    )


def _validate_non_overlapping_strides(shape: tuple[int, ...], stride: tuple[int, ...]) -> None:
    dimensions = sorted(
        (
            (dimension_stride, dimension)
            for dimension, dimension_stride in zip(shape, stride, strict=True)
            if dimension > 1
        ),
        key=lambda item: item[0],
    )
    required_span = 1
    for dimension_stride, dimension in dimensions:
        if dimension_stride < required_span:
            raise ContractError("tensor strides describe overlapping storage")
        required_span += (dimension - 1) * dimension_stride


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    """Logical, backing-independent description of finalized runtime bytes."""

    artifact_key: str
    producer: ProducerDescriptor
    weight_format: WeightFormatDescriptor
    format_metadata: Mapping[str, JSONValue]
    compatibility_digest: str
    tensors: tuple[TensorSpec, ...]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ContractError(f"unsupported manifest schema {self.schema_version}; expected {SCHEMA_VERSION}")
        _require_digest(self.artifact_key, "artifact_key")
        _require_digest(self.compatibility_digest, "compatibility_digest")
        tensors = tuple(sorted(tuple(self.tensors), key=lambda tensor: tensor.tensor_id))
        if not tensors:
            raise ContractError("an artifact manifest must contain at least one tensor")
        if len({tensor.tensor_id for tensor in tensors}) != len(tensors):
            raise ContractError("artifact tensor ids must be unique")
        object.__setattr__(self, "tensors", tensors)
        object.__setattr__(
            self,
            "format_metadata",
            _freeze_json(self.format_metadata, "format_metadata"),
        )

    @classmethod
    def create(
        cls,
        spec: ArtifactSpec,
        tensors: tuple[TensorSpec, ...],
        format_metadata: Mapping[str, JSONValue] | None = None,
    ) -> ArtifactManifest:
        from .identity import derive_manifest_compatibility_digest

        normalized_tensors = tuple(sorted(tuple(tensors), key=lambda tensor: tensor.tensor_id))
        metadata = format_metadata or {}
        digest = derive_manifest_compatibility_digest(
            schema_version=spec.schema_version,
            artifact_key=spec.artifact_key,
            producer=spec.producer,
            weight_format=spec.weight_format,
            format_metadata=metadata,
            tensors=normalized_tensors,
        )
        return cls(
            schema_version=spec.schema_version,
            artifact_key=spec.artifact_key,
            producer=spec.producer,
            weight_format=spec.weight_format,
            format_metadata=metadata,
            compatibility_digest=digest,
            tensors=normalized_tensors,
        )

    @property
    def tensor_ids(self) -> tuple[str, ...]:
        return tuple(tensor.tensor_id for tensor in self.tensors)

    def tensor(self, tensor_id: str) -> TensorSpec:
        for tensor in self.tensors:
            if tensor.tensor_id == tensor_id:
                return tensor
        raise KeyError(tensor_id)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "artifact_key": self.artifact_key,
            "producer": self.producer.to_dict(),
            "weight_format": self.weight_format.to_dict(),
            "format_metadata": _thaw_json(self.format_metadata),
            "compatibility_digest": self.compatibility_digest,
            "tensors": [tensor.to_dict() for tensor in self.tensors],
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ArtifactManifest:
        _require_exact_keys(
            value,
            {
                "schema_version",
                "artifact_key",
                "producer",
                "weight_format",
                "format_metadata",
                "compatibility_digest",
                "tensors",
            },
            "artifact manifest",
        )
        tensors = _require_sequence(value["tensors"], "artifact manifest tensors")
        return cls(
            schema_version=int(value["schema_version"]),
            artifact_key=str(value["artifact_key"]),
            producer=ProducerDescriptor.from_dict(_require_mapping(value["producer"], "manifest producer")),
            weight_format=WeightFormatDescriptor.from_dict(
                _require_mapping(value["weight_format"], "manifest weight_format")
            ),
            format_metadata=_require_mapping(value["format_metadata"], "manifest format_metadata"),
            compatibility_digest=str(value["compatibility_digest"]),
            tensors=tuple(TensorSpec.from_dict(item) for item in tensors),
        )

    @classmethod
    def from_json(cls, payload: str) -> ArtifactManifest:
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ContractError("artifact manifest is not valid JSON") from exc
        if not isinstance(value, dict):
            raise ContractError("artifact manifest must be a JSON object")
        return cls.from_dict(value)


@dataclass(frozen=True, slots=True)
class TensorSelection:
    tensor_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        tensor_ids = tuple(self.tensor_ids)
        if not tensor_ids:
            raise ContractError("a tensor selection must not be empty")
        for tensor_id in tensor_ids:
            _require_identifier(tensor_id, "tensor_id")
        if len(set(tensor_ids)) != len(tensor_ids):
            raise ContractError("a tensor selection must not contain duplicates")
        object.__setattr__(self, "tensor_ids", tensor_ids)

    @classmethod
    def one(cls, tensor_id: str) -> TensorSelection:
        return cls((tensor_id,))


@dataclass(frozen=True, slots=True)
class AccessRequirements:
    required_features: frozenset[AccessFeature]
    accepted_backings: frozenset[BackingKind]

    def __post_init__(self) -> None:
        try:
            features = frozenset(AccessFeature(item) for item in self.required_features)
            backings = frozenset(BackingKind(item) for item in self.accepted_backings)
        except ValueError as exc:
            raise ContractError("access requirements contain an unknown enum value") from exc
        if not features:
            raise ContractError("at least one access feature must be required")
        if not backings:
            raise ContractError("at least one backing kind must be accepted")
        object.__setattr__(self, "required_features", features)
        object.__setattr__(self, "accepted_backings", backings)


@dataclass(frozen=True, slots=True)
class BackingCapabilities:
    kind: BackingKind
    provider_id: str
    provider_abi: str
    features: frozenset[AccessFeature]

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", BackingKind(self.kind))
        _require_identifier(self.provider_id, "provider_id")
        _require_identifier(self.provider_abi, "provider_abi")
        object.__setattr__(
            self,
            "features",
            frozenset(AccessFeature(item) for item in self.features),
        )


@dataclass(frozen=True, slots=True)
class CapabilityGrant:
    """Opaque authority to resolve through one negotiated provider.

    ``runtime_instance_id`` and ``grant_id`` are generated by the issuing
    runtime.  Callers may carry this value across integration layers, but a
    runtime accepts only an exact grant that it issued while open.
    """

    runtime_instance_id: str
    grant_id: str
    requirements: AccessRequirements
    backing_kind: BackingKind
    provider_id: str
    provider_abi: str
    features: frozenset[AccessFeature]

    def __post_init__(self) -> None:
        _require_identifier(self.runtime_instance_id, "runtime_instance_id")
        _require_identifier(self.grant_id, "grant_id")
        object.__setattr__(self, "backing_kind", BackingKind(self.backing_kind))
        _require_identifier(self.provider_id, "provider_id")
        _require_identifier(self.provider_abi, "provider_abi")
        object.__setattr__(
            self,
            "features",
            frozenset(AccessFeature(item) for item in self.features),
        )


@dataclass(frozen=True, slots=True)
class CapabilitiesUnavailable:
    missing_features_by_backing: Mapping[BackingKind, frozenset[AccessFeature]]
    reason_code: str

    def __post_init__(self) -> None:
        _require_identifier(self.reason_code, "reason_code")
        missing = {
            BackingKind(kind): frozenset(AccessFeature(item) for item in features)
            for kind, features in self.missing_features_by_backing.items()
        }
        object.__setattr__(
            self,
            "missing_features_by_backing",
            MappingProxyType(dict(sorted(missing.items(), key=lambda item: item[0].value))),
        )


CapabilityDecision: TypeAlias = CapabilityGrant | CapabilitiesUnavailable


class BuildFailureKind(str, Enum):
    RETRYABLE = "retryable"
    FATAL = "fatal"


class BuildFailureStage(str, Enum):
    OWNER_LOST = "owner_lost"
    INITIAL_SIGNAL = "initial_signal"
    SINK_CREATE = "sink_create"
    PRODUCER_OPEN = "producer_open"
    PRODUCER_BUILD = "producer_build"
    PRODUCER_CLEANUP = "producer_cleanup"
    SEMANTIC_VALIDATION = "semantic_validation"
    COMMIT = "commit"
    READY_OPEN = "ready_open"


@dataclass(frozen=True, slots=True)
class BuildFailureClassification:
    stage: BuildFailureStage
    code: str
    detail: str
    kind: BuildFailureKind
    retry_after_s: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", BuildFailureStage(self.stage))
        object.__setattr__(self, "kind", BuildFailureKind(self.kind))
        _require_identifier(self.code, "build failure code")
        if not isinstance(self.detail, str):
            raise ContractError("build failure detail must be a string")
        if self.retry_after_s is not None and (not math.isfinite(self.retry_after_s) or self.retry_after_s < 0):
            raise ContractError("build failure retry_after_s must be finite and non-negative")


class BuildRole(str, Enum):
    AUTHORIZED_BUILDER = "authorized_builder"
    ORDERED_WAITER = "ordered_waiter"
    READ_ONLY = "read_only"


class BuilderInitialSignalState(str, Enum):
    PENDING = "pending"
    STARTED = "started"
    READY = "ready"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class BuilderStarted:
    launch_id: str
    artifact_key: str
    lease_id: str
    builder_actor_id: str
    monotonic_time_ns: int

    def __post_init__(self) -> None:
        _require_identifier(self.launch_id, "launch_id")
        _require_digest(self.artifact_key, "artifact_key")
        _require_identifier(self.lease_id, "lease_id")
        _require_identifier(self.builder_actor_id, "builder_actor_id")
        if self.monotonic_time_ns < 0:
            raise ContractError("monotonic_time_ns must be non-negative")


@dataclass(frozen=True, slots=True)
class ArtifactAlreadyReady:
    launch_id: str
    artifact_key: str

    def __post_init__(self) -> None:
        _require_identifier(self.launch_id, "launch_id")
        _require_digest(self.artifact_key, "artifact_key")


@dataclass(frozen=True, slots=True)
class BuilderStartFailed:
    launch_id: str
    artifact_key: str
    code: str
    detail: str

    def __post_init__(self) -> None:
        _require_identifier(self.launch_id, "launch_id")
        _require_digest(self.artifact_key, "artifact_key")
        _require_identifier(self.code, "builder start failure code")
        if not isinstance(self.detail, str):
            raise ContractError("builder start failure detail must be a string")


@dataclass(frozen=True, slots=True)
class BuilderStartTimeout:
    launch_id: str
    artifact_key: str
    timeout_s: float

    def __post_init__(self) -> None:
        _require_identifier(self.launch_id, "launch_id")
        _require_digest(self.artifact_key, "artifact_key")
        if not math.isfinite(self.timeout_s) or self.timeout_s < 0:
            raise ContractError("builder start timeout must be finite and non-negative")


BuildGateOutcome: TypeAlias = BuilderStarted | ArtifactAlreadyReady | BuilderStartFailed | BuilderStartTimeout


@dataclass(frozen=True, slots=True)
class BuildAuthorization:
    role: BuildRole
    actor_id: str
    authorized_builder_actor_id: str
    launch_id: str
    observed_start: BuilderStarted | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", BuildRole(self.role))
        _require_identifier(self.actor_id, "actor_id")
        _require_identifier(self.authorized_builder_actor_id, "authorized_builder_actor_id")
        _require_identifier(self.launch_id, "launch_id")


@dataclass(frozen=True, slots=True)
class BuildIntent:
    producer: ProducerDescriptor
    owner_lost_failure: BuildFailureClassification


@dataclass(frozen=True, slots=True)
class StorageSpan:
    object_id: str
    offset_bytes: int
    length_bytes: int

    def __post_init__(self) -> None:
        _require_identifier(self.object_id, "storage object id")
        if self.offset_bytes < 0 or self.length_bytes < 0:
            raise ContractError("storage span offset and length must be non-negative")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "object_id": self.object_id,
            "offset_bytes": self.offset_bytes,
            "length_bytes": self.length_bytes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> StorageSpan:
        _require_exact_keys(value, {"object_id", "offset_bytes", "length_bytes"}, "storage span")
        return cls(
            object_id=str(value["object_id"]),
            offset_bytes=int(value["offset_bytes"]),
            length_bytes=int(value["length_bytes"]),
        )


@dataclass(frozen=True, slots=True)
class StorageObject:
    object_id: str
    nbytes: int
    alignment: int
    content_digest: str
    relative_path: str | None

    def __post_init__(self) -> None:
        _require_identifier(self.object_id, "storage object id")
        if self.nbytes < 0:
            raise ContractError("storage object size must be non-negative")
        if self.alignment < 1 or self.alignment & (self.alignment - 1):
            raise ContractError("storage object alignment must be a positive power of two")
        _require_digest(self.content_digest, "storage object content_digest")
        if self.relative_path is not None:
            path = PurePosixPath(self.relative_path)
            if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
                raise ContractError("storage object path must be a safe relative POSIX path")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "object_id": self.object_id,
            "nbytes": self.nbytes,
            "alignment": self.alignment,
            "content_digest": self.content_digest,
            "relative_path": self.relative_path,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> StorageObject:
        _require_exact_keys(
            value,
            {"object_id", "nbytes", "alignment", "content_digest", "relative_path"},
            "storage object",
        )
        relative_path = value["relative_path"]
        if relative_path is not None and not isinstance(relative_path, str):
            raise ContractError("storage object relative_path must be a string or null")
        return cls(
            object_id=str(value["object_id"]),
            nbytes=int(value["nbytes"]),
            alignment=int(value["alignment"]),
            content_digest=str(value["content_digest"]),
            relative_path=relative_path,
        )


@dataclass(frozen=True, slots=True)
class BackingIndex:
    artifact_key: str
    generation_id: str
    kind: BackingKind
    provider_name: str
    provider_version: str
    objects: tuple[StorageObject, ...]
    tensor_spans: Mapping[str, StorageSpan]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ContractError(f"unsupported backing-index schema {self.schema_version}; expected {SCHEMA_VERSION}")
        _require_digest(self.artifact_key, "backing-index artifact_key")
        _require_identifier(self.generation_id, "generation_id")
        _require_identifier(self.provider_name, "provider_name")
        _require_identifier(self.provider_version, "provider_version")
        kind = BackingKind(self.kind)
        objects = tuple(sorted(tuple(self.objects), key=lambda item: item.object_id))
        if not objects:
            raise ContractError("a backing index must contain at least one storage object")
        if len({item.object_id for item in objects}) != len(objects):
            raise ContractError("storage object ids must be unique")
        spans = dict(sorted(dict(self.tensor_spans).items()))
        if not spans:
            raise ContractError("a backing index must contain at least one tensor span")
        for tensor_id in spans:
            _require_identifier(tensor_id, "tensor span id")
        object_map = {item.object_id: item for item in objects}
        intervals: dict[str, list[tuple[int, int, str]]] = {}
        for tensor_id, span in spans.items():
            if span.object_id not in object_map:
                raise ContractError(f"tensor {tensor_id!r} references unknown storage object {span.object_id!r}")
            storage_object = object_map[span.object_id]
            if span.offset_bytes + span.length_bytes > storage_object.nbytes:
                raise ContractError(f"tensor {tensor_id!r} exceeds its storage object")
            if span.length_bytes:
                intervals.setdefault(span.object_id, []).append(
                    (span.offset_bytes, span.offset_bytes + span.length_bytes, tensor_id)
                )
        for values in intervals.values():
            values.sort()
            for previous, current in zip(values, values[1:], strict=False):
                if current[0] < previous[1]:
                    raise ContractError(f"tensor storage overlaps between {previous[2]!r} and {current[2]!r}")
        for storage_object in objects:
            if kind in {BackingKind.RUNTIME_MMAP, BackingKind.CHECKPOINT_MMAP}:
                if storage_object.relative_path is None:
                    raise ContractError("mmap storage objects require a relative path")
            elif storage_object.relative_path is not None:
                raise ContractError("loaded-tensor objects cannot refer to files")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "objects", objects)
        object.__setattr__(self, "tensor_spans", MappingProxyType(spans))

    def object(self, object_id: str) -> StorageObject:
        for item in self.objects:
            if item.object_id == object_id:
                return item
        raise KeyError(object_id)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "artifact_key": self.artifact_key,
            "generation_id": self.generation_id,
            "kind": self.kind.value,
            "provider_name": self.provider_name,
            "provider_version": self.provider_version,
            "objects": [item.to_dict() for item in self.objects],
            "tensor_spans": {tensor_id: span.to_dict() for tensor_id, span in self.tensor_spans.items()},
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BackingIndex:
        _require_exact_keys(
            value,
            {
                "schema_version",
                "artifact_key",
                "generation_id",
                "kind",
                "provider_name",
                "provider_version",
                "objects",
                "tensor_spans",
            },
            "backing index",
        )
        objects = _require_sequence(value["objects"], "backing index objects")
        tensor_spans = _require_mapping(value["tensor_spans"], "backing index tensor_spans")
        return cls(
            schema_version=int(value["schema_version"]),
            artifact_key=str(value["artifact_key"]),
            generation_id=str(value["generation_id"]),
            kind=BackingKind(value["kind"]),
            provider_name=str(value["provider_name"]),
            provider_version=str(value["provider_version"]),
            objects=tuple(StorageObject.from_dict(item) for item in objects),
            tensor_spans={str(tensor_id): StorageSpan.from_dict(span) for tensor_id, span in tensor_spans.items()},
        )

    @classmethod
    def from_json(cls, payload: str) -> BackingIndex:
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ContractError("backing index is not valid JSON") from exc
        if not isinstance(value, dict):
            raise ContractError("backing index must be a JSON object")
        return cls.from_dict(value)


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    manifest: ArtifactManifest
    backing_index: BackingIndex

    def __post_init__(self) -> None:
        if self.manifest.artifact_key != self.backing_index.artifact_key:
            raise ContractError("manifest and backing index artifact keys differ")
        if set(self.manifest.tensor_ids) != set(self.backing_index.tensor_spans):
            raise ContractError("manifest and backing index tensor ids differ")
        used_objects = {span.object_id for span in self.backing_index.tensor_spans.values()}
        if used_objects != {item.object_id for item in self.backing_index.objects}:
            raise ContractError("backing index contains an unreferenced storage object")
        for tensor in self.manifest.tensors:
            span = self.backing_index.tensor_spans[tensor.tensor_id]
            if span.length_bytes != tensor.storage_nbytes:
                raise ContractError(f"physical span length does not match tensor {tensor.tensor_id!r}")
            if span.offset_bytes % dtype_item_size(tensor.dtype):
                raise ContractError(f"physical span is not dtype-aligned for tensor {tensor.tensor_id!r}")

    @property
    def publication_lease_id(self) -> str:
        """Generation provenance used to validate an ordered builder gate."""

        return self.backing_index.generation_id


class ArtifactSink(Protocol):
    def write_tensor(
        self,
        tensor_id: str,
        tensor: torch.Tensor,
        *,
        role: TensorRole = TensorRole.PARAMETER,
    ) -> TensorSpec: ...


class ArtifactBuildSession(Protocol):
    def build(self, sink: ArtifactSink) -> ArtifactManifest: ...

    def close(self) -> None: ...


class BuildSessionCleanupRegistry(Protocol):
    """Runtime-owned exact-identity registry for producer sessions."""

    def register_before_return(self, session: ArtifactBuildSession) -> None: ...

    def close_and_release(self, session: ArtifactBuildSession) -> None: ...


class ResolvedArtifactRegistrar(Protocol):
    """Ownership handoff used before a resolved artifact can be returned.

    A valid implementation is preinstalled by the integration layer and its
    adoption method is non-raising for one empty artifact slot.  Successful
    adoption is the ownership linearization point; a subsequent ``Ready``
    value contains only a borrowed reference to the exact object adopted here.
    """

    def adopt_artifact(self, artifact: ResolvedArtifact) -> None: ...


class ArtifactProducer(Protocol):
    @property
    def descriptor(self) -> ProducerDescriptor: ...

    def open_build(
        self,
        cleanup_registry: BuildSessionCleanupRegistry,
    ) -> ArtifactBuildSession: ...


class BuilderStartPublisher(Protocol):
    @property
    def launch_id(self) -> str: ...

    @property
    def initial_signal_state(self) -> BuilderInitialSignalState: ...

    def publish_started_if_pending(self, event: BuilderStarted) -> bool: ...

    def publish_ready_if_pending(self, event: ArtifactAlreadyReady) -> bool: ...

    def publish_failed_if_pending(self, event: BuilderStartFailed) -> bool: ...


class BuilderStartGate(Protocol):
    def wait(
        self,
        launch_id: str,
        artifact_key: str,
        timeout_s: float,
    ) -> BuildGateOutcome: ...


__all__ = [
    "SCHEMA_VERSION",
    "AccessFeature",
    "AccessRequirements",
    "ArtifactAlreadyReady",
    "ArtifactBuildSession",
    "ArtifactManifest",
    "ArtifactProducer",
    "ArtifactRecord",
    "ArtifactSink",
    "ArtifactSpec",
    "ArtifactTopologyDescriptor",
    "BackingCapabilities",
    "BackingIndex",
    "BackingKind",
    "BuildAuthorization",
    "BuildFailureClassification",
    "BuildFailureKind",
    "BuildFailureStage",
    "BuildGateOutcome",
    "BuildIntent",
    "BuildRole",
    "BuildSessionCleanupRegistry",
    "BuilderInitialSignalState",
    "BuilderStartFailed",
    "BuilderStartGate",
    "BuilderStartPublisher",
    "BuilderStartTimeout",
    "BuilderStarted",
    "CapabilitiesUnavailable",
    "CapabilityDecision",
    "CapabilityGrant",
    "ContractError",
    "HostCopyMode",
    "JSONScalar",
    "JSONValue",
    "ProducerDescriptor",
    "ResolvedArtifactRegistrar",
    "StorageObject",
    "StorageSpan",
    "TensorRole",
    "TensorSelection",
    "TensorSpec",
    "TopologyCoordinate",
    "WeightFormatDescriptor",
    "canonical_digest",
    "canonical_json",
    "derive_weight_format_plan_digest",
    "dtype_item_size",
    "required_storage_numel",
    "torch_dtype_name",
]
