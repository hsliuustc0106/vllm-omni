# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Composition-owned Host Weight Runtime lifecycle envelope.

The owner is installed before preparation starts and remains the only outward
cleanup authority until the selected consumer has been disabled.  Borrowed
properties expose objects for validation and execution without transferring
their ownership through a Python return/assignment window.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, TypeAlias

from vllm_omni.host_weight_runtime import HostWeightRuntime, ResolvedArtifact

from .binding import PreparedModuleBinding
from .session import PreparedWeightAccessSession


class WeightConsumer(Protocol):
    """Narrow lifecycle implemented by every HWR weight consumer."""

    def adopt_prepared_session(self, prepared: PreparedWeightAccessSession) -> None:
        """Store the exact session; valid input must make this non-raising."""

    def enable_transactionally(self) -> None: ...

    def disable(self) -> None: ...


class LegacyReason(str, Enum):
    HWR_DISABLED = "hwr_disabled"
    OPTIONAL_CAPABILITY_UNAVAILABLE = "optional_capability_unavailable"
    OPTIONAL_ARTIFACT_UNAVAILABLE = "optional_artifact_unavailable"


@dataclass(frozen=True, slots=True)
class PreparedSessionReady:
    prepared_session: PreparedWeightAccessSession


@dataclass(frozen=True, slots=True)
class UseLegacy:
    reason: LegacyReason
    detail: str


@dataclass(frozen=True, slots=True)
class RetryablePreparationFailure:
    code: str
    detail: str
    retry_after_s: float | None = None
    cleanup_required: bool = False


@dataclass(frozen=True, slots=True)
class FatalPreparationFailure:
    code: str
    detail: str
    cleanup_required: bool = False


WeightAccessSessionFactoryResult: TypeAlias = (
    PreparedSessionReady | UseLegacy | RetryablePreparationFailure | FatalPreparationFailure
)


class WeightConsumerOwnerPhase(str, Enum):
    EMPTY = "empty"
    PREPARING = "preparing"
    PREPARATION_RESULT = "preparation_result"
    CONSUMER_PREPARING = "consumer_preparing"
    CONSUMER = "consumer"
    CLOSED = "closed"


class PreparationCleanupHandle:
    """Owner-retained, dependency-ordered preparation cleanup registrar."""

    def __init__(self, runtime: HostWeightRuntime) -> None:
        self._runtime: HostWeightRuntime | None = runtime
        self._artifact: ResolvedArtifact | None = None
        self._candidate: object | None = None
        self._binding: PreparedModuleBinding | None = None
        self._prepared: PreparedWeightAccessSession | None = None
        self._lock = threading.RLock()

    @property
    def closed(self) -> bool:
        with self._lock:
            return all(
                item is None
                for item in (
                    self._runtime,
                    self._artifact,
                    self._candidate,
                    self._binding,
                    self._prepared,
                )
            )

    @property
    def prepared_session(self) -> PreparedWeightAccessSession | None:
        with self._lock:
            return self._prepared

    def retain_candidate(self, candidate: object) -> None:
        """Register an exact completed skeleton candidate before callee return."""

        with self._lock:
            if self._candidate is not None:
                raise RuntimeError("preparation cleanup already owns a pipeline candidate")
            if self._prepared is not None:
                raise RuntimeError("prepared-session handoff is already complete")
            self._candidate = candidate

    def adopt_artifact(self, artifact: ResolvedArtifact) -> None:
        """Linearize core-to-integration ownership before ``Ready`` returns."""

        with self._lock:
            if self._artifact is not None:
                raise RuntimeError("preparation cleanup already owns an artifact")
            if self._prepared is not None:
                raise RuntimeError("prepared-session handoff is already complete")
            self._artifact = artifact

    def retain_binding(self, binding: PreparedModuleBinding) -> None:
        with self._lock:
            if self._binding is not None:
                raise RuntimeError("preparation cleanup already owns a binding")
            if self._prepared is not None:
                raise RuntimeError("prepared-session handoff is already complete")
            self._binding = binding

    def transfer_to_prepared_session(self, prepared: PreparedWeightAccessSession) -> None:
        """Replace individual resources with one exact rollback authority."""

        with self._lock:
            if self._prepared is not None:
                raise RuntimeError("preparation cleanup already owns a prepared session")
            if prepared._runtime is not self._runtime:  # noqa: SLF001 - exact ownership contract
                raise RuntimeError("prepared session does not own the registered runtime")
            if prepared._artifact is not self._artifact:  # noqa: SLF001
                raise RuntimeError("prepared session does not own the registered artifact")
            if prepared._binding is not self._binding:  # noqa: SLF001
                raise RuntimeError("prepared session does not own the registered binding")
            candidate = self._candidate
            if candidate is not None:
                skeleton = getattr(candidate, "skeleton", None)
                pipeline = getattr(skeleton, "pipeline", None)
                if pipeline is not prepared.pipeline:
                    raise RuntimeError("prepared session does not own the registered pipeline candidate")
            self._prepared = prepared
            self._binding = None
            self._candidate = None
            self._artifact = None
            self._runtime = None

    def release_prepared_to_consumer(self, prepared: PreparedWeightAccessSession) -> None:
        """Disarm only after the owner publishes consumer authority."""

        with self._lock:
            if self._prepared is not prepared:
                raise RuntimeError("consumer adopted a different prepared session")
            self._prepared = None

    @staticmethod
    def _close_candidate(candidate: object) -> None:
        close = getattr(candidate, "close", None)
        if callable(close):
            close()

    def close(self) -> None:
        """Retry cleanup from the first unfinished dependency."""

        with self._lock:
            prepared = self._prepared
            if prepared is not None:
                prepared.rollback()
                if self._prepared is prepared:
                    self._prepared = None
                return

            binding = self._binding
            if binding is not None:
                binding.rollback()
                if self._binding is binding:
                    self._binding = None

            candidate = self._candidate
            if candidate is not None:
                self._close_candidate(candidate)
                if self._candidate is candidate:
                    self._candidate = None

            artifact = self._artifact
            if artifact is not None:
                artifact.close()
                if self._artifact is artifact:
                    self._artifact = None

            runtime = self._runtime
            if runtime is not None:
                runtime.close()
                if self._runtime is runtime:
                    self._runtime = None


class WeightConsumerOwner:
    """Runner-retained authority for preparation, consumer enable, and close."""

    def __init__(self) -> None:
        self._phase = WeightConsumerOwnerPhase.EMPTY
        self._cleanup: PreparationCleanupHandle | None = None
        self._result: WeightAccessSessionFactoryResult | None = None
        self._consumer: WeightConsumer | None = None
        self._lock = threading.RLock()

    @property
    def phase(self) -> WeightConsumerOwnerPhase:
        with self._lock:
            return self._phase

    @property
    def preparation_result(self) -> WeightAccessSessionFactoryResult:
        with self._lock:
            if self._phase is not WeightConsumerOwnerPhase.PREPARATION_RESULT or self._result is None:
                raise RuntimeError(f"preparation result is unavailable in owner phase {self._phase.value}")
            return self._result

    @property
    def prepared_session(self) -> PreparedWeightAccessSession:
        with self._lock:
            if self._phase not in {
                WeightConsumerOwnerPhase.PREPARATION_RESULT,
                WeightConsumerOwnerPhase.CONSUMER_PREPARING,
            }:
                raise RuntimeError(f"prepared session is unavailable in owner phase {self._phase.value}")
            result = self._result
            if not isinstance(result, PreparedSessionReady):
                raise RuntimeError("owner preparation did not produce a ready session")
            return result.prepared_session

    @property
    def consumer(self) -> WeightConsumer:
        with self._lock:
            if self._phase is not WeightConsumerOwnerPhase.CONSUMER or self._consumer is None:
                raise RuntimeError(f"consumer is unavailable in owner phase {self._phase.value}")
            return self._consumer

    def begin_preparation(self, runtime: HostWeightRuntime) -> PreparationCleanupHandle:
        with self._lock:
            if self._phase is not WeightConsumerOwnerPhase.EMPTY:
                raise RuntimeError(f"cannot begin preparation in owner phase {self._phase.value}")
            handle = PreparationCleanupHandle(runtime)
            self._cleanup = handle
            self._phase = WeightConsumerOwnerPhase.PREPARING
            return handle

    def publish_preparation_result(self, result: WeightAccessSessionFactoryResult) -> None:
        with self._lock:
            if self._phase is WeightConsumerOwnerPhase.EMPTY:
                if isinstance(result, PreparedSessionReady):
                    raise RuntimeError("a ready result requires an owner-held preparation")
                cleanup_required = bool(getattr(result, "cleanup_required", False))
                if cleanup_required:
                    raise RuntimeError("resource-free preparation result cannot require cleanup")
            elif self._phase is WeightConsumerOwnerPhase.PREPARING:
                handle = self._cleanup
                assert handle is not None
                if isinstance(result, PreparedSessionReady):
                    if handle.prepared_session is not result.prepared_session:
                        raise RuntimeError("ready result does not match the owner-held prepared session")
                elif isinstance(result, UseLegacy):
                    if not handle.closed:
                        raise RuntimeError("legacy fallback cannot be published while cleanup remains")
                else:
                    expected = not handle.closed
                    if result.cleanup_required is not expected:
                        raise RuntimeError("preparation failure cleanup_required does not match retained ownership")
            else:
                raise RuntimeError(f"cannot publish preparation result in owner phase {self._phase.value}")
            self._result = result
            self._phase = WeightConsumerOwnerPhase.PREPARATION_RESULT

    def publish_consumer(self, candidate: WeightConsumer) -> None:
        with self._lock:
            if self._phase is not WeightConsumerOwnerPhase.PREPARATION_RESULT:
                raise RuntimeError(f"cannot publish consumer in owner phase {self._phase.value}")
            prepared = self.prepared_session
            handle = self._cleanup
            if handle is None or handle.prepared_session is not prepared:
                raise RuntimeError("owner has no exact prepared-session cleanup authority")
            self._phase = WeightConsumerOwnerPhase.CONSUMER_PREPARING
            self._consumer = candidate
            candidate.adopt_prepared_session(prepared)
            # This monotonic phase marker is the cleanup-authority handoff.
            self._phase = WeightConsumerOwnerPhase.CONSUMER
            self._result = None
            handle.release_prepared_to_consumer(prepared)
            if handle.closed:
                self._cleanup = None

    def close(self) -> None:
        with self._lock:
            if self._phase is WeightConsumerOwnerPhase.CLOSED:
                return
            if self._phase is WeightConsumerOwnerPhase.CONSUMER:
                consumer = self._consumer
                assert consumer is not None
                consumer.disable()
                if self._consumer is consumer:
                    self._consumer = None
                self._cleanup = None
                self._result = None
                self._phase = WeightConsumerOwnerPhase.CLOSED
                return

            handle = self._cleanup
            if handle is not None and not handle.closed:
                handle.close()
            if handle is not None and handle.closed:
                self._cleanup = None
            if self._phase is WeightConsumerOwnerPhase.CONSUMER_PREPARING:
                self._consumer = None
            self._result = None
            self._phase = WeightConsumerOwnerPhase.CLOSED


__all__ = [
    "FatalPreparationFailure",
    "LegacyReason",
    "PreparationCleanupHandle",
    "PreparedSessionReady",
    "RetryablePreparationFailure",
    "UseLegacy",
    "WeightAccessSessionFactoryResult",
    "WeightConsumer",
    "WeightConsumerOwner",
    "WeightConsumerOwnerPhase",
]
