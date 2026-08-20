# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Structured capability and artifact-resolution outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from .contracts import AccessFeature, BackingKind

if TYPE_CHECKING:
    from .artifact import ResolvedArtifact


class ResolutionPath(str, Enum):
    LOADED = "loaded"
    MMAP_HIT = "mmap_hit"
    MMAP_BUILT = "mmap_built"
    MMAP_WAIT_HIT = "mmap_wait_hit"


@dataclass(frozen=True, slots=True)
class ResolutionInfo:
    path: ResolutionPath
    artifact_key: str
    generation_id: str
    backing_kind: BackingKind


@dataclass(frozen=True, slots=True)
class ResolvedAccess:
    runtime_instance_id: str
    grant_id: str
    backing_kind: BackingKind
    provider_id: str
    provider_abi: str
    features: frozenset[AccessFeature]


@dataclass(frozen=True, slots=True)
class Ready:
    artifact: ResolvedArtifact
    info: ResolutionInfo
    access: ResolvedAccess


@dataclass(frozen=True, slots=True)
class RetryableFailure:
    code: str
    detail: str
    retry_after_s: float | None = None


@dataclass(frozen=True, slots=True)
class FatalFailure:
    code: str
    detail: str


ResolveOutcome = Ready | RetryableFailure | FatalFailure


__all__ = [
    "FatalFailure",
    "Ready",
    "ResolvedAccess",
    "ResolutionInfo",
    "ResolutionPath",
    "ResolveOutcome",
    "RetryableFailure",
]
