# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""JSON-safe Host Weight Runtime resolution evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from vllm_omni.host_weight_runtime import (
    FatalFailure,
    Ready,
    ResolutionPath,
    RetryableFailure,
)

HWR_RESOLUTION_EVIDENCE_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class HostWeightResolutionEvidence:
    """One process-local record of an HWR artifact resolution attempt."""

    runtime_mode: str
    outcome: str
    events: tuple[str, ...]
    artifact_key: str | None = None
    artifact_compatibility_digest: str | None = None
    resolution_path: str | None = None
    claim_role: str | None = None
    cache_hit: bool | None = None
    generation_id: str | None = None
    backing_kind: str | None = None
    runtime_instance_id: str | None = None
    capability_grant_id: str | None = None
    provider_id: str | None = None
    provider_abi: str | None = None
    access_features: tuple[str, ...] | None = None
    negotiated_capability_grant_id: str | None = None
    selected_transfer_plan_id: str | None = None
    selected_transfer_plan_kind: str | None = None
    exact_coverage_digest: str | None = None
    unit_kinds: tuple[str, ...] | None = None
    pre_resolve: Mapping[str, int] | None = None
    builder_started: Mapping[str, object] | None = None
    observed_builder_started: Mapping[str, object] | None = None
    producer_present: bool | None = None
    code: str | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": HWR_RESOLUTION_EVIDENCE_SCHEMA_VERSION,
            "runtime_mode": self.runtime_mode,
            "outcome": self.outcome,
            "events": list(self.events),
            "artifact_key": self.artifact_key,
            "artifact_compatibility_digest": self.artifact_compatibility_digest,
            "resolution_path": self.resolution_path,
            "claim_role": self.claim_role,
            "cache_hit": self.cache_hit,
            "generation_id": self.generation_id,
            "backing_kind": self.backing_kind,
            "runtime_instance_id": self.runtime_instance_id,
            "capability_grant_id": self.capability_grant_id,
            "provider_id": self.provider_id,
            "provider_abi": self.provider_abi,
            "access_features": (None if self.access_features is None else list(self.access_features)),
            "negotiated_capability_grant_id": self.negotiated_capability_grant_id,
            "selected_transfer_plan_id": self.selected_transfer_plan_id,
            "selected_transfer_plan_kind": self.selected_transfer_plan_kind,
            "exact_coverage_digest": self.exact_coverage_digest,
            "unit_kinds": None if self.unit_kinds is None else list(self.unit_kinds),
            "pre_resolve": None if self.pre_resolve is None else dict(self.pre_resolve),
            "builder_started": None if self.builder_started is None else dict(self.builder_started),
            "observed_builder_started": (
                None if self.observed_builder_started is None else dict(self.observed_builder_started)
            ),
            "producer_present": self.producer_present,
            "code": self.code,
            "detail": self.detail,
        }


def evidence_from_outcome(
    outcome: Ready | RetryableFailure | FatalFailure,
    *,
    runtime_mode: str,
    expected_artifact_key: str,
    artifact_compatibility_digest: str | None = None,
) -> HostWeightResolutionEvidence:
    """Normalize a core resolution outcome without losing its election path."""

    if isinstance(outcome, Ready):
        path = outcome.info.path
        if path is ResolutionPath.MMAP_BUILT:
            claim_role = "builder"
            events = ("builder", "ready")
        elif path is ResolutionPath.MMAP_WAIT_HIT:
            claim_role = "waiter"
            events = ("waiter", "ready")
        elif path is ResolutionPath.MMAP_HIT:
            claim_role = "cache_hit"
            events = ("cache_hit", "ready")
        else:
            claim_role = "loaded"
            events = ("ready",)
        return HostWeightResolutionEvidence(
            runtime_mode=runtime_mode,
            outcome="ready",
            events=events,
            artifact_key=outcome.info.artifact_key,
            artifact_compatibility_digest=artifact_compatibility_digest,
            resolution_path=path.value,
            claim_role=claim_role,
            cache_hit=path is ResolutionPath.MMAP_HIT,
            generation_id=outcome.info.generation_id,
            backing_kind=outcome.info.backing_kind.value,
            runtime_instance_id=outcome.access.runtime_instance_id,
            capability_grant_id=outcome.access.grant_id,
            provider_id=outcome.access.provider_id,
            provider_abi=outcome.access.provider_abi,
            access_features=tuple(sorted(feature.value for feature in outcome.access.features)),
        )

    if isinstance(outcome, FatalFailure):
        normalized_outcome = "fatal"
        events = ("fatal",)
    elif isinstance(outcome, RetryableFailure):
        normalized_outcome = "retryable_failure"
        events = ("retryable_failure",)
    else:
        raise TypeError(f"unknown Host Weight Runtime outcome {type(outcome).__name__}")
    return HostWeightResolutionEvidence(
        runtime_mode=runtime_mode,
        outcome=normalized_outcome,
        events=events,
        artifact_key=expected_artifact_key,
        code=outcome.code,
        detail=outcome.detail,
    )


def preparation_failure_evidence(
    error: BaseException,
    *,
    runtime_mode: str,
    fallback: bool,
) -> HostWeightResolutionEvidence:
    """Represent a preparation failure that occurred before core resolution."""

    message = str(error)
    code, separator, detail = message.partition(": ")
    if not separator:
        code = "preparation_fallback" if fallback else "preparation_failed"
        detail = message
    outcome = "fallback" if fallback else "fatal"
    return HostWeightResolutionEvidence(
        runtime_mode=runtime_mode,
        outcome=outcome,
        events=(outcome,),
        code=code,
        detail=detail,
    )


def not_requested_evidence() -> HostWeightResolutionEvidence:
    return HostWeightResolutionEvidence(
        runtime_mode="disabled",
        outcome="not_requested",
        events=(),
    )


__all__ = [
    "HWR_RESOLUTION_EVIDENCE_SCHEMA_VERSION",
    "HostWeightResolutionEvidence",
    "evidence_from_outcome",
    "not_requested_evidence",
    "preparation_failure_evidence",
]
