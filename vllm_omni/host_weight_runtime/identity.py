# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Canonical artifact and manifest identity functions."""

from __future__ import annotations

from collections.abc import Mapping

from .contracts import (
    ArtifactSpec,
    JSONValue,
    ProducerDescriptor,
    TensorSpec,
    WeightFormatDescriptor,
    canonical_digest,
)


def derive_artifact_key(spec: ArtifactSpec) -> str:
    return canonical_digest(spec.to_dict())


def derive_manifest_compatibility_digest(
    *,
    schema_version: int,
    artifact_key: str,
    producer: ProducerDescriptor,
    weight_format: WeightFormatDescriptor,
    format_metadata: Mapping[str, JSONValue],
    tensors: tuple[TensorSpec, ...],
) -> str:
    """Hash logical compatibility only; no provider path enters this digest."""
    return canonical_digest(
        {
            "schema_version": schema_version,
            "artifact_key": artifact_key,
            "producer": producer.to_dict(),
            "weight_format": weight_format.to_dict(),
            "format_metadata": format_metadata,
            "tensors": [tensor.to_dict() for tensor in sorted(tensors, key=lambda item: item.tensor_id)],
        }
    )


__all__ = ["derive_artifact_key", "derive_manifest_compatibility_digest"]
