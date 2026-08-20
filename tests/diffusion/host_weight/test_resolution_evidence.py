# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import vllm_omni.diffusion.worker.diffusion_worker as worker_module
from vllm_omni.diffusion.host_weight.evidence import (
    HostWeightResolutionEvidence,
    evidence_from_outcome,
    preparation_failure_evidence,
)
from vllm_omni.diffusion.worker.diffusion_model_runner import (
    DiffusionModelRunner,
)
from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker
from vllm_omni.host_weight_runtime import (
    AccessFeature,
    BackingKind,
    FatalFailure,
    Ready,
    ResolutionInfo,
    ResolutionPath,
    ResolvedAccess,
    RetryableFailure,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_ARTIFACT_KEY = "a" * 64
_COMPATIBILITY_DIGEST = "b" * 64


@pytest.mark.parametrize(
    ("path", "claim_role", "events", "cache_hit"),
    [
        (ResolutionPath.MMAP_BUILT, "builder", ["builder", "ready"], False),
        (ResolutionPath.MMAP_WAIT_HIT, "waiter", ["waiter", "ready"], False),
        (ResolutionPath.MMAP_HIT, "cache_hit", ["cache_hit", "ready"], True),
        (ResolutionPath.LOADED, "loaded", ["ready"], False),
    ],
)
def test_ready_evidence_preserves_resolution_path(
    path: ResolutionPath,
    claim_role: str,
    events: list[str],
    cache_hit: bool,
) -> None:
    outcome = Ready(
        artifact=object(),  # type: ignore[arg-type]
        info=ResolutionInfo(
            path=path,
            artifact_key=_ARTIFACT_KEY,
            generation_id="generation-1",
            backing_kind=BackingKind.RUNTIME_MMAP,
        ),
        access=ResolvedAccess(
            runtime_instance_id="runtime-1",
            grant_id="grant-1",
            backing_kind=BackingKind.RUNTIME_MMAP,
            provider_id="local-runtime-mmap",
            provider_abi="1",
            features=frozenset(
                {
                    AccessFeature.COMPLETE_TENSOR_READ,
                    AccessFeature.SHARED_PAGES,
                }
            ),
        ),
    )

    result = evidence_from_outcome(
        outcome,
        runtime_mode="read_write",
        expected_artifact_key=_ARTIFACT_KEY,
        artifact_compatibility_digest=_COMPATIBILITY_DIGEST,
    ).to_dict()

    assert result == {
        "schema_version": 2,
        "runtime_mode": "read_write",
        "outcome": "ready",
        "events": events,
        "artifact_key": _ARTIFACT_KEY,
        "artifact_compatibility_digest": _COMPATIBILITY_DIGEST,
        "resolution_path": path.value,
        "claim_role": claim_role,
        "cache_hit": cache_hit,
        "generation_id": "generation-1",
        "backing_kind": "runtime_mmap",
        "runtime_instance_id": "runtime-1",
        "capability_grant_id": "grant-1",
        "provider_id": "local-runtime-mmap",
        "provider_abi": "1",
        "access_features": ["complete_tensor_read", "shared_pages"],
        "negotiated_capability_grant_id": None,
        "selected_transfer_plan_id": None,
        "selected_transfer_plan_kind": None,
        "exact_coverage_digest": None,
        "unit_kinds": None,
        "pre_resolve": None,
        "builder_started": None,
        "observed_builder_started": None,
        "producer_present": None,
        "code": None,
        "detail": None,
    }
    json.dumps(result)


@pytest.mark.parametrize(
    ("outcome", "expected_outcome", "expected_events"),
    [
        (
            RetryableFailure("publication_not_ready", "builder active"),
            "retryable_failure",
            ["retryable_failure"],
        ),
        (FatalFailure("corrupt_store", "digest mismatch"), "fatal", ["fatal"]),
    ],
)
def test_failure_evidence_keeps_core_outcome(
    outcome: object,
    expected_outcome: str,
    expected_events: list[str],
) -> None:
    result = evidence_from_outcome(
        outcome,  # type: ignore[arg-type]
        runtime_mode="read_only",
        expected_artifact_key=_ARTIFACT_KEY,
    ).to_dict()

    assert result["outcome"] == expected_outcome
    assert result["events"] == expected_events
    assert result["artifact_key"] == _ARTIFACT_KEY
    assert result["code"] == outcome.code  # type: ignore[attr-defined]
    json.dumps(result)


def test_pre_resolution_failure_is_explicit_and_json_safe() -> None:
    result = preparation_failure_evidence(
        RuntimeError("repository_unavailable: no mount"),
        runtime_mode="read_only",
        fallback=True,
    ).to_dict()

    assert result["outcome"] == "fallback"
    assert result["events"] == ["fallback"]
    assert result["code"] == "repository_unavailable"
    assert result["detail"] == "no mount"
    assert result["artifact_key"] is None
    json.dumps(result)


def test_runner_returns_a_copy_with_fallback_policy() -> None:
    runner = object.__new__(DiffusionModelRunner)
    runner.host_weight_runtime_evidence = HostWeightResolutionEvidence(
        runtime_mode="read_only",
        outcome="ready",
        events=("cache_hit", "ready"),
        artifact_key=_ARTIFACT_KEY,
    ).to_dict()
    runner.host_weight_runtime_fell_back = True

    result = runner.get_host_weight_runtime_evidence()

    assert result["ordinary_loader_fallback"] is True
    assert "ordinary_loader_fallback" not in runner.host_weight_runtime_evidence


def test_worker_rpc_collects_rank_identity(monkeypatch) -> None:
    local_base = HostWeightResolutionEvidence(
        runtime_mode="read_write",
        outcome="ready",
        events=("builder", "ready"),
        artifact_key=_ARTIFACT_KEY,
    ).to_dict()
    worker = object.__new__(DiffusionWorker)
    worker.rank = 1
    worker.local_rank = 1
    worker.stage_id = 3
    worker.od_config = SimpleNamespace(num_gpus=2)
    worker.model_runner = SimpleNamespace(get_host_weight_runtime_evidence=lambda: dict(local_base))

    def gather(local_result):
        ok, local = local_result
        assert ok is True
        peer = dict(local)
        peer["rank"] = {
            "global_rank": 0,
            "local_rank": 0,
            "world_size": 2,
            "stage_id": 3,
            "pid": 100,
        }
        return [(True, peer), local_result]

    monkeypatch.setattr(worker_module, "_all_gather_rank_values", gather)

    result = worker.get_host_weight_runtime_evidence()

    assert [item["rank"]["global_rank"] for item in result] == [0, 1]  # type: ignore[index]
    assert result[1]["artifact_key"] == _ARTIFACT_KEY
    assert "rank" not in local_base
    json.dumps(result)
