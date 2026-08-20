# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-neutral no-storage diffusion consumer skeleton contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from torch import nn

from .formats.base import FormatContractError


class SkeletonError(FormatContractError):
    """Raised before artifact tensors mutate a consumer module."""


@dataclass(frozen=True, slots=True)
class PipelineSkeleton:
    pipeline: object
    target_module_path: str
    target_module: nn.Module
    target_module_type_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.target_module, nn.Module):
            raise SkeletonError("pipeline skeleton target must be an nn.Module")
        if not self.target_module_path:
            raise SkeletonError("pipeline skeleton target path must not be empty")
        if not self.target_module_type_id:
            raise SkeletonError("pipeline skeleton target type ID must not be empty")


class ConsumerCandidateOrigin(str, Enum):
    BUILDER_REBIND = "builder_rebind"
    ORDERED_WAITER = "ordered_waiter"
    WARM_HIT = "warm_hit"


@dataclass(frozen=True, slots=True)
class ConsumerPipelineCandidate:
    """Exact locally constructed pipeline retained before factory return."""

    skeleton: PipelineSkeleton
    origin: ConsumerCandidateOrigin
    artifact_key: str

    def __post_init__(self) -> None:
        if not self.artifact_key:
            raise SkeletonError("consumer pipeline candidate artifact key must not be empty")


__all__ = [
    "ConsumerCandidateOrigin",
    "ConsumerPipelineCandidate",
    "PipelineSkeleton",
    "SkeletonError",
]
