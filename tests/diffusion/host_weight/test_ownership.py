# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import pytest

from vllm_omni.diffusion.host_weight.ownership import (
    FatalPreparationFailure,
    LegacyReason,
    PreparedSessionReady,
    UseLegacy,
    WeightConsumerOwner,
    WeightConsumerOwnerPhase,
)


class _Closeable:
    def __init__(self, *, fail_once: bool = False) -> None:
        self.calls = 0
        self.fail_once = fail_once

    def close(self) -> None:
        self.calls += 1
        if self.fail_once and self.calls == 1:
            raise RuntimeError("injected close failure")


class _Binding:
    def __init__(self) -> None:
        self.calls = 0

    def rollback(self) -> None:
        self.calls += 1


@dataclass
class _Skeleton:
    pipeline: object


@dataclass
class _Candidate:
    skeleton: _Skeleton


class _Prepared:
    def __init__(self, pipeline: object, runtime: object, artifact: object, binding: object) -> None:
        self.pipeline = pipeline
        self._runtime = runtime
        self._artifact = artifact
        self._binding = binding
        self.rollback_calls = 0

    def rollback(self) -> None:
        self.rollback_calls += 1


class _Consumer:
    def __init__(self, *, fail_disable_once: bool = False) -> None:
        self.prepared = None
        self.enable_calls = 0
        self.disable_calls = 0
        self.fail_disable_once = fail_disable_once

    def adopt_prepared_session(self, prepared) -> None:
        assert self.prepared is None
        self.prepared = prepared

    def enable_transactionally(self) -> None:
        self.enable_calls += 1

    def disable(self) -> None:
        self.disable_calls += 1
        if self.fail_disable_once and self.disable_calls == 1:
            raise RuntimeError("injected disable failure")
        if self.enable_calls == 0 and self.prepared is not None:
            self.prepared.rollback()
        self.prepared = None


class _FailingAdoptConsumer(_Consumer):
    def adopt_prepared_session(self, prepared) -> None:
        self.prepared = prepared
        raise RuntimeError("injected adopt failure")


def _ready_owner(*, consumer: _Consumer | None = None):
    owner = WeightConsumerOwner()
    runtime = _Closeable()
    artifact = _Closeable()
    binding = _Binding()
    pipeline = object()
    candidate = _Candidate(_Skeleton(pipeline))
    handle = owner.begin_preparation(runtime)  # type: ignore[arg-type]
    handle.adopt_artifact(artifact)  # type: ignore[arg-type]
    handle.retain_candidate(candidate)
    handle.retain_binding(binding)  # type: ignore[arg-type]
    prepared = _Prepared(pipeline, runtime, artifact, binding)
    handle.transfer_to_prepared_session(prepared)  # type: ignore[arg-type]
    owner.publish_preparation_result(PreparedSessionReady(prepared))  # type: ignore[arg-type]
    if consumer is not None:
        owner.publish_consumer(consumer)
    return owner, prepared, runtime, artifact, binding


def test_owner_retains_prepared_session_without_take_window() -> None:
    owner, prepared, runtime, artifact, binding = _ready_owner()

    assert owner.phase is WeightConsumerOwnerPhase.PREPARATION_RESULT
    assert owner.prepared_session is prepared
    owner.close()

    assert prepared.rollback_calls == 1
    assert runtime.calls == 0
    assert artifact.calls == 0
    assert binding.calls == 0
    assert owner.phase is WeightConsumerOwnerPhase.CLOSED


def test_consumer_handoff_keeps_owner_as_pre_enable_retry_target() -> None:
    consumer = _Consumer()
    owner, prepared, *_ = _ready_owner(consumer=consumer)

    assert owner.phase is WeightConsumerOwnerPhase.CONSUMER
    assert owner.consumer is consumer
    assert consumer.prepared is prepared
    owner.close()

    assert consumer.disable_calls == 1
    assert prepared.rollback_calls == 1
    assert owner.phase is WeightConsumerOwnerPhase.CLOSED


def test_consumer_disable_failure_preserves_exact_retry_owner() -> None:
    consumer = _Consumer(fail_disable_once=True)
    owner, prepared, *_ = _ready_owner(consumer=consumer)

    with pytest.raises(RuntimeError, match="injected disable failure"):
        owner.close()
    assert owner.phase is WeightConsumerOwnerPhase.CONSUMER
    assert owner.consumer is consumer
    assert consumer.prepared is prepared

    owner.close()
    assert owner.phase is WeightConsumerOwnerPhase.CLOSED
    assert consumer.disable_calls == 2


def test_consumer_adoption_failure_retains_owner_cleanup_authority() -> None:
    consumer = _FailingAdoptConsumer()
    owner, prepared, *_ = _ready_owner()

    with pytest.raises(RuntimeError, match="injected adopt failure"):
        owner.publish_consumer(consumer)

    assert owner.phase is WeightConsumerOwnerPhase.CONSUMER_PREPARING
    assert consumer.prepared is prepared
    owner.close()
    assert prepared.rollback_calls == 1
    assert owner.phase is WeightConsumerOwnerPhase.CLOSED


def test_preparation_cleanup_stops_at_failed_dependency_and_retries() -> None:
    owner = WeightConsumerOwner()
    runtime = _Closeable()
    artifact = _Closeable(fail_once=True)
    binding = _Binding()
    handle = owner.begin_preparation(runtime)  # type: ignore[arg-type]
    handle.adopt_artifact(artifact)  # type: ignore[arg-type]
    handle.retain_binding(binding)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="injected close failure"):
        handle.close()
    assert binding.calls == 1
    assert artifact.calls == 1
    assert runtime.calls == 0

    owner.publish_preparation_result(
        FatalPreparationFailure("cleanup_failed", "artifact close failed", cleanup_required=True)
    )
    owner.close()
    assert binding.calls == 1
    assert artifact.calls == 2
    assert runtime.calls == 1
    assert owner.phase is WeightConsumerOwnerPhase.CLOSED


def test_resource_free_legacy_result_needs_no_cleanup_handle() -> None:
    owner = WeightConsumerOwner()
    result = UseLegacy(LegacyReason.HWR_DISABLED, "disabled by configuration")
    owner.publish_preparation_result(result)

    assert owner.preparation_result is result
    owner.close()
    assert owner.phase is WeightConsumerOwnerPhase.CLOSED


def test_owner_rejects_legacy_while_resources_are_live() -> None:
    owner = WeightConsumerOwner()
    owner.begin_preparation(_Closeable())  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="cleanup remains"):
        owner.publish_preparation_result(UseLegacy(LegacyReason.OPTIONAL_ARTIFACT_UNAVAILABLE, "miss"))
