# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backing-independent semantic validation before publication or binding."""

from __future__ import annotations

from .contracts import ArtifactManifest, ArtifactSpec, ContractError
from .identity import derive_artifact_key, derive_manifest_compatibility_digest


def validate_manifest_against_spec(
    spec: ArtifactSpec,
    manifest: ArtifactManifest,
) -> ArtifactManifest:
    """Return ``manifest`` only when it is semantically valid for ``spec``.

    This check must run before repository commit.  Physical object validation
    remains the repository/backing layer's independent responsibility.
    """
    expected_key = derive_artifact_key(spec)
    if manifest.schema_version != spec.schema_version:
        raise ContractError("manifest and artifact-spec schemas differ")
    if manifest.artifact_key != expected_key:
        raise ContractError("manifest artifact key does not match requested spec")
    if manifest.producer != spec.producer:
        raise ContractError("manifest producer does not match requested spec")
    if manifest.weight_format != spec.weight_format:
        raise ContractError("manifest weight format does not match requested spec")
    expected_compatibility = derive_manifest_compatibility_digest(
        schema_version=manifest.schema_version,
        artifact_key=manifest.artifact_key,
        producer=manifest.producer,
        weight_format=manifest.weight_format,
        format_metadata=manifest.format_metadata,
        tensors=manifest.tensors,
    )
    if manifest.compatibility_digest != expected_compatibility:
        raise ContractError("manifest compatibility digest is invalid")
    return manifest


__all__ = ["validate_manifest_against_spec"]
