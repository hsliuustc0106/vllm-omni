# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capability negotiation and producer-safe artifact resolution."""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass

import torch

from ._exceptions import (
    safe_add_exception_note,
    safe_exception_detail,
    safe_exception_summary,
)
from .artifact import ResolvedArtifact
from .backings import (
    BackingError,
    BackingProvider,
    BackingProviderRegistry,
    WeightBacking,
)
from .contracts import (
    AccessRequirements,
    ArtifactAlreadyReady,
    ArtifactManifest,
    ArtifactProducer,
    ArtifactRecord,
    ArtifactSink,
    ArtifactSpec,
    BackingCapabilities,
    BackingKind,
    BuildAuthorization,
    BuilderStarted,
    BuilderStartFailed,
    BuilderStartPublisher,
    BuildFailureClassification,
    BuildFailureKind,
    BuildFailureStage,
    BuildIntent,
    BuildRole,
    BuildSessionCleanupRegistry,
    CapabilitiesUnavailable,
    CapabilityDecision,
    CapabilityGrant,
    ContractError,
    JSONValue,
    ResolvedArtifactRegistrar,
    TensorRole,
)
from .outcomes import (
    FatalFailure,
    Ready,
    ResolutionInfo,
    ResolutionPath,
    ResolvedAccess,
    ResolveOutcome,
    RetryableFailure,
)
from .store import (
    ArtifactRepository,
    Builder,
    BuilderFailed,
    Existing,
    NoBuilder,
    ReservationError,
    StoreCorruptionError,
    StoreError,
    StorePublicationDurabilityError,
    Waiter,
    WaitTimeout,
)
from .validation import validate_manifest_against_spec


class RuntimeClosed(RuntimeError):  # noqa: N818 - public lifecycle state name
    """A new operation was attempted after runtime close."""


class RuntimeBusy(RuntimeError):  # noqa: N818 - public lifecycle state name
    """Runtime close was requested while a resolved artifact is live."""


class _ClassifiedBuildError(RuntimeError):
    def __init__(self, failure: BuildFailureClassification) -> None:
        super().__init__(failure.code)
        self.failure = failure


class _ProducerPhaseError(RuntimeError):
    def __init__(
        self,
        stage: BuildFailureStage,
        primary: BaseException,
    ) -> None:
        super().__init__(stage.value)
        self.stage = stage
        self.primary = primary


class _RuntimeBuildSessionCleanupRegistry(BuildSessionCleanupRegistry):
    """Exact-identity ownership for sessions crossing ``open_build``."""

    def __init__(self) -> None:
        self._sessions: dict[int, object] = {}
        self._lock = threading.RLock()

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def owns(self, session: object) -> bool:
        with self._lock:
            return self._sessions.get(id(session)) is session

    def register_before_return(self, session: object) -> None:
        with self._lock:
            identity = id(session)
            current = self._sessions.get(identity)
            if current is session:
                return
            if current is not None:
                raise RuntimeError("producer build-session identity collision")
            self._sessions[identity] = session

    def close_and_release(self, session: object) -> None:
        with self._lock:
            if self._sessions.get(id(session)) is not session:
                raise RuntimeError("producer build session is not registered with this runtime")
        close = getattr(session, "close", None)
        if not callable(close):
            raise RuntimeError("registered producer build session has no close()")
        close()
        with self._lock:
            if self._sessions.get(id(session)) is session:
                del self._sessions[id(session)]

    def close_pending(self) -> None:
        primary: BaseException | None = None
        with self._lock:
            sessions = tuple(self._sessions.values())
        for session in sessions:
            try:
                self.close_and_release(session)
            except BaseException as exc:
                if primary is None:
                    primary = exc
                else:
                    safe_add_exception_note(
                        primary,
                        "another retained producer session cleanup also failed",
                        exc,
                    )
        if primary is not None:
            raise primary


def _build_manifest(
    producer: ArtifactProducer,
    sink: ArtifactSink,
    cleanup_registry: _RuntimeBuildSessionCleanupRegistry,
) -> ArtifactManifest:
    """Build through a runtime-owned session and release it before publish."""

    try:
        build_session = producer.open_build(cleanup_registry)
    except BaseException as exc:
        raise _ProducerPhaseError(BuildFailureStage.PRODUCER_OPEN, exc) from exc

    if not cleanup_registry.owns(build_session):
        # Adopt before attempting cleanup so a failing close cannot orphan the
        # exact object returned by a contract-violating producer.
        cleanup_registry.register_before_return(build_session)
        try:
            cleanup_registry.close_and_release(build_session)
        except BaseException as cleanup_exc:
            raise _ProducerPhaseError(
                BuildFailureStage.PRODUCER_CLEANUP,
                cleanup_exc,
            ) from cleanup_exc
        contract_error = ContractError("artifact producer returned an unregistered build session")
        raise _ProducerPhaseError(
            BuildFailureStage.PRODUCER_OPEN,
            contract_error,
        ) from contract_error

    try:
        manifest = build_session.build(sink)
    except BaseException as primary:
        try:
            cleanup_registry.close_and_release(build_session)
        except BaseException as cleanup_exc:
            safe_add_exception_note(
                primary,
                "artifact producer cleanup after build failure also failed",
                cleanup_exc,
            )
            raise _ProducerPhaseError(
                BuildFailureStage.PRODUCER_CLEANUP,
                cleanup_exc,
            ) from primary
        raise _ProducerPhaseError(
            BuildFailureStage.PRODUCER_BUILD,
            primary,
        ) from primary
    try:
        cleanup_registry.close_and_release(build_session)
    except BaseException as cleanup_exc:
        raise _ProducerPhaseError(
            BuildFailureStage.PRODUCER_CLEANUP,
            cleanup_exc,
        ) from cleanup_exc
    return manifest


def _classify_build_failure(
    stage: BuildFailureStage,
    error: BaseException,
) -> BuildFailureClassification:
    detail = safe_exception_detail(error)
    if stage is BuildFailureStage.PRODUCER_CLEANUP:
        code = "artifact_builder_cleanup_failed"
        kind = BuildFailureKind.FATAL
    elif stage is BuildFailureStage.PRODUCER_OPEN:
        code = "artifact_producer_open_failed"
        kind = BuildFailureKind.FATAL
    elif stage is BuildFailureStage.PRODUCER_BUILD:
        code = "artifact_producer_build_failed"
        kind = BuildFailureKind.FATAL
    elif stage is BuildFailureStage.SEMANTIC_VALIDATION:
        code = "artifact_build_rejected"
        kind = BuildFailureKind.FATAL
    elif stage is BuildFailureStage.INITIAL_SIGNAL:
        code = "builder_initial_signal_failed"
        kind = BuildFailureKind.FATAL
    elif stage is BuildFailureStage.READY_OPEN:
        if isinstance(error, OSError):
            code = "ready_open_io_failed"
            kind = BuildFailureKind.RETRYABLE
        else:
            code = "invalid_published_artifact"
            kind = BuildFailureKind.FATAL
    elif isinstance(error, StorePublicationDurabilityError):
        code = "artifact_publication_durability_failed"
        kind = BuildFailureKind.FATAL
    elif isinstance(error, StoreCorruptionError):
        code = "corrupt_store"
        kind = BuildFailureKind.FATAL
    elif isinstance(error, (ContractError, BackingError, ReservationError, TypeError, ValueError)):
        code = "artifact_build_rejected"
        kind = BuildFailureKind.FATAL
    elif isinstance(error, (StoreError, OSError)):
        code = "artifact_publication_failed"
        kind = BuildFailureKind.RETRYABLE
    else:
        code = "artifact_builder_failed"
        kind = BuildFailureKind.FATAL
    return BuildFailureClassification(stage, code, detail, kind)


def _outcome_from_build_failure(
    failure: BuildFailureClassification,
) -> RetryableFailure | FatalFailure:
    if failure.kind is BuildFailureKind.RETRYABLE:
        return RetryableFailure(
            failure.code,
            failure.detail,
            failure.retry_after_s,
        )
    return FatalFailure(failure.code, failure.detail)


def _owner_lost_failure(artifact_key: str) -> BuildFailureClassification:
    return BuildFailureClassification(
        BuildFailureStage.OWNER_LOST,
        "builder_owner_lost",
        artifact_key,
        BuildFailureKind.RETRYABLE,
    )


@dataclass(frozen=True, slots=True)
class RuntimeCapabilities:
    backings: tuple[BackingCapabilities, ...]

    def __post_init__(self) -> None:
        backings = tuple(self.backings)
        if len({item.kind for item in backings}) != len(backings):
            raise ContractError("runtime capability backing kinds must be unique")
        object.__setattr__(self, "backings", backings)


class HostWeightRuntime:
    """Independent single-node runtime for immutable host weights.

    ``writable=False`` is the read-only mode.  Such a runtime passes
    ``BuildIntent=None`` even when a producer object is supplied, so it can
    hit or wait but can never elect itself as builder.
    """

    def __init__(
        self,
        repository: ArtifactRepository | None,
        provider_registry: BackingProviderRegistry,
        *,
        writable: bool = True,
    ) -> None:
        self.repository = repository
        self.writable = bool(writable)
        self._provider_registry = provider_registry
        self._capabilities = self._read_provider_registry(provider_registry)
        self._runtime_instance_id = f"runtime-{uuid.uuid4().hex}"
        self._grants: dict[str, CapabilityGrant] = {}
        self._build_session_cleanup_registry = _RuntimeBuildSessionCleanupRegistry()
        self._closed = False
        self._artifact_registrations: dict[object, bool] = {}
        # Strong references held only while a freshly opened backing is being
        # assembled into a ResolvedArtifact and handed to the external owner.
        # A failed cleanup remains here so runtime.close() can retry the exact
        # object instead of relying on a best-effort destructor.
        self._artifact_assembly_guards: dict[object, object] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _read_provider_registry(
        registry: BackingProviderRegistry,
    ) -> RuntimeCapabilities:
        capabilities = RuntimeCapabilities(tuple(registry.capabilities()))
        for capability in capabilities.backings:
            provider = registry.provider_for(capability.kind)
            if provider is None:
                raise ContractError(f"provider registry advertises {capability.kind.value!r} without a provider")
            if provider.capabilities() != capability:
                raise ContractError(f"provider registry capability differs from provider {capability.kind.value!r}")
        return capabilities

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return self._capabilities

    @property
    def provider_registry(self) -> BackingProviderRegistry:
        return self._provider_registry

    @property
    def active_artifacts(self) -> int:
        with self._lock:
            return len(self._artifact_registrations)

    @property
    def pending_artifact_assemblies(self) -> int:
        with self._lock:
            return len(self._artifact_assembly_guards)

    @property
    def runtime_instance_id(self) -> str:
        return self._runtime_instance_id

    @property
    def build_session_cleanup_registry(self) -> BuildSessionCleanupRegistry:
        return self._build_session_cleanup_registry

    def _require_open(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeClosed("host-weight runtime is closed")

    def _require_stable_provider_registry(self) -> None:
        current = self._read_provider_registry(self._provider_registry)
        if current != self._capabilities:
            raise ContractError("backing provider registry changed after runtime construction")

    def _provider_for_grant(self, grant: CapabilityGrant) -> BackingProvider:
        provider = self._provider_registry.provider_for(grant.backing_kind)
        if provider is None:
            raise BackingError(f"no provider is registered for {grant.backing_kind.value!r}")
        capability = provider.capabilities()
        if (
            capability.kind is not grant.backing_kind
            or capability.provider_id != grant.provider_id
            or capability.provider_abi != grant.provider_abi
            or capability.features != grant.features
        ):
            raise BackingError("registered provider no longer matches the negotiated capability grant")
        return provider

    def negotiate(self, requirements: AccessRequirements) -> CapabilityDecision:
        self._require_open()
        self._require_stable_provider_registry()
        preference = (
            BackingKind.RUNTIME_MMAP,
            BackingKind.LOADED_TENSOR,
            BackingKind.CHECKPOINT_MMAP,
        )
        by_kind = {item.kind: item for item in self.capabilities.backings}
        for kind in preference:
            if kind not in requirements.accepted_backings or kind not in by_kind:
                continue
            candidate = by_kind[kind]
            if requirements.required_features <= candidate.features:
                grant = CapabilityGrant(
                    runtime_instance_id=self._runtime_instance_id,
                    grant_id=f"grant-{uuid.uuid4().hex}",
                    requirements=requirements,
                    backing_kind=candidate.kind,
                    provider_id=candidate.provider_id,
                    provider_abi=candidate.provider_abi,
                    features=candidate.features,
                )
                with self._lock:
                    if self._closed:
                        raise RuntimeClosed("host-weight runtime is closed")
                    self._grants[grant.grant_id] = grant
                return grant
        missing = {
            kind: requirements.required_features
            - by_kind.get(
                kind,
                BackingCapabilities(kind, "unavailable", "unavailable", frozenset()),
            ).features
            for kind in requirements.accepted_backings
        }
        return CapabilitiesUnavailable(
            missing_features_by_backing=missing,
            reason_code="unsupported_access",
        )

    def _validate_grant(
        self,
        grant: CapabilityGrant,
    ) -> FatalFailure | None:
        with self._lock:
            issued = self._grants.get(grant.grant_id)
        if grant.runtime_instance_id != self._runtime_instance_id or issued != grant:
            return FatalFailure(
                "invalid_capability_grant",
                "capability grant was not issued by this runtime instance",
            )
        if grant.backing_kind not in grant.requirements.accepted_backings:
            return FatalFailure(
                "invalid_capability_grant",
                "grant backing is outside its captured access requirements",
            )
        if not grant.requirements.required_features <= grant.features:
            return FatalFailure(
                "invalid_capability_grant",
                "grant features no longer satisfy its captured requirements",
            )
        try:
            current_capabilities = self._read_provider_registry(self._provider_registry)
        except Exception:
            current_capabilities = RuntimeCapabilities(())
        current = {item.kind: item for item in current_capabilities.backings}.get(grant.backing_kind)
        if current is None or (
            current_capabilities != self._capabilities
            or current.provider_id != grant.provider_id
            or current.provider_abi != grant.provider_abi
            or current.features != grant.features
        ):
            return FatalFailure(
                "stale_capability_grant",
                "the negotiated provider, ABI, or feature set changed before resolution",
            )
        return None

    def _require_resolution_ready(self) -> None:
        self._require_open()
        if self._build_session_cleanup_registry.pending_count:
            raise RuntimeBusy("cannot resolve while producer build-session cleanup remains pending")
        if self.pending_artifact_assemblies:
            raise RuntimeBusy("cannot resolve while resolved-artifact cleanup remains pending")

    @staticmethod
    def _validate_artifact_registrar(
        artifact_registrar: ResolvedArtifactRegistrar,
    ) -> FatalFailure | None:
        try:
            adopt = getattr(artifact_registrar, "adopt_artifact")
        except BaseException as exc:
            return FatalFailure(
                "invalid_artifact_registrar",
                safe_exception_detail(exc),
            )
        if not callable(adopt):
            return FatalFailure(
                "invalid_artifact_registrar",
                "artifact registrar must expose a callable adopt_artifact()",
            )
        return None

    def _release_artifact_assembly_guard(
        self,
        registration: object,
        target: object,
    ) -> None:
        with self._lock:
            if self._artifact_assembly_guards.get(registration) is target:
                del self._artifact_assembly_guards[registration]

    def _close_artifact_assembly_guard(
        self,
        registration: object,
        target: object,
    ) -> None:
        close = getattr(target, "close", None)
        if not callable(close):
            raise RuntimeError("artifact assembly cleanup target has no close()")
        close()
        self._release_artifact_assembly_guard(registration, target)
        # A ResolvedArtifact normally removes the liveness token through its
        # close callback.  A backing retained before wrapper construction has
        # no such callback; the idempotent pop covers both cases.
        with self._lock:
            self._artifact_registrations.pop(registration, None)

    def _close_pending_artifact_assemblies(self) -> None:
        primary: BaseException | None = None
        with self._lock:
            pending = tuple(self._artifact_assembly_guards.items())
        for registration, target in pending:
            try:
                self._close_artifact_assembly_guard(registration, target)
            except BaseException as exc:
                if primary is None:
                    primary = exc
                else:
                    safe_add_exception_note(
                        primary,
                        "another retained artifact assembly cleanup also failed",
                        exc,
                    )
        if primary is not None:
            raise primary

    def _register(
        self,
        backing: object,
        artifact_registrar: ResolvedArtifactRegistrar,
    ) -> ResolvedArtifact:
        """Assemble and externally adopt one artifact before returning it.

        The runtime-local guard is installed before wrapper construction and
        disarmed only after the registrar has adopted the exact artifact.
        Therefore a caller interruption in the method-return/assignment
        window cannot orphan the backing.
        """

        registration = object()

        def release() -> None:
            with self._lock:
                self._artifact_registrations.pop(registration, None)

        with self._lock:
            if self._closed:
                closed_error = RuntimeClosed("host-weight runtime is closed")
                close = getattr(backing, "close")
                try:
                    close()
                except BaseException as cleanup_exc:
                    safe_add_exception_note(
                        closed_error,
                        "new backing cleanup after concurrent runtime close also failed",
                        cleanup_exc,
                    )
                raise closed_error
            self._artifact_registrations[registration] = False
            self._artifact_assembly_guards[registration] = backing

        try:
            artifact = ResolvedArtifact(backing, on_closed=release)  # type: ignore[arg-type]
            with self._lock:
                if registration not in self._artifact_registrations:
                    raise RuntimeError("resolved artifact closed before registration activation")
                self._artifact_registrations[registration] = True
                self._artifact_assembly_guards[registration] = artifact
        except BaseException as construction_exc:
            with self._lock:
                wrapper_cleanup_completed = registration not in self._artifact_registrations
            if wrapper_cleanup_completed:
                # A fully constructed but unpublished wrapper can be destroyed
                # by an injected return-window interruption.  Its callback
                # proves backing cleanup already completed.
                self._release_artifact_assembly_guard(registration, backing)
            else:
                try:
                    self._close_artifact_assembly_guard(registration, backing)
                except BaseException as cleanup_exc:
                    safe_add_exception_note(
                        construction_exc,
                        "backing cleanup after resolved-artifact construction failure also failed",
                        cleanup_exc,
                    )
            raise

        try:
            # The registrar contract is a non-raising exact-reference store.
            # Keep runtime close serialized across adoption and guard disarm
            # so no thread can close the artifact between those two steps.
            with self._lock:
                if self._artifact_assembly_guards.get(registration) is not artifact:
                    raise RuntimeError("resolved-artifact assembly guard was lost before adoption")
                artifact_registrar.adopt_artifact(artifact)
                del self._artifact_assembly_guards[registration]
        except BaseException as adoption_exc:
            try:
                self._close_artifact_assembly_guard(registration, artifact)
            except BaseException as cleanup_exc:
                safe_add_exception_note(
                    adoption_exc,
                    "artifact cleanup after registrar adoption failure also failed",
                    cleanup_exc,
                )
            raise
        return artifact

    @staticmethod
    def _validate_opened_backing(
        backing: WeightBacking,
        record: ArtifactRecord,
    ) -> None:
        if backing.manifest != record.manifest or backing.backing_index != record.backing_index:
            raise BackingError("provider opened a backing that differs from the requested artifact record")

    def _open_record_backing(
        self,
        record: ArtifactRecord,
        grant: CapabilityGrant,
    ) -> WeightBacking:
        provider = self._provider_for_grant(grant)
        backing: WeightBacking | None = None
        try:
            backing = provider.open(record.manifest, record.backing_index)
            self._validate_opened_backing(backing, record)
            return backing
        except BaseException as primary:
            if backing is not None:
                try:
                    backing.close()
                except BaseException as cleanup_exc:
                    safe_add_exception_note(
                        primary,
                        "invalid provider backing cleanup also failed",
                        cleanup_exc,
                    )
            raise

    def _ready_from_record(
        self,
        spec: ArtifactSpec,
        record: ArtifactRecord,
        path: ResolutionPath,
        grant: CapabilityGrant,
        artifact_registrar: ResolvedArtifactRegistrar,
    ) -> ResolveOutcome:
        try:
            validate_manifest_against_spec(spec, record.manifest)
            index = record.backing_index
            if (
                index.kind is not grant.backing_kind
                or index.provider_name != grant.provider_id
                or index.provider_version != grant.provider_abi
            ):
                return FatalFailure(
                    "resolved_provider_mismatch",
                    "published backing does not match the negotiated capability grant",
                )
            backing = self._open_record_backing(record, grant)
            artifact = self._register(backing, artifact_registrar)
        except RuntimeClosed:
            raise
        except BaseException as exc:
            return _outcome_from_build_failure(_classify_build_failure(BuildFailureStage.READY_OPEN, exc))
        return Ready(
            artifact,
            ResolutionInfo(
                path=path,
                artifact_key=record.manifest.artifact_key,
                generation_id=record.backing_index.generation_id,
                backing_kind=record.backing_index.kind,
            ),
            ResolvedAccess(
                runtime_instance_id=grant.runtime_instance_id,
                grant_id=grant.grant_id,
                backing_kind=grant.backing_kind,
                provider_id=grant.provider_id,
                provider_abi=grant.provider_abi,
                features=grant.features,
            ),
        )

    @staticmethod
    def _validate_build_authorization(
        spec: ArtifactSpec,
        authorization: BuildAuthorization,
        producer: ArtifactProducer | None,
        build_events: BuilderStartPublisher | None,
        *,
        writable: bool,
    ) -> FatalFailure | None:
        role = authorization.role
        observed = authorization.observed_start
        if role is BuildRole.AUTHORIZED_BUILDER:
            if (
                authorization.actor_id != authorization.authorized_builder_actor_id
                or observed is not None
                or producer is None
                or build_events is None
                or not writable
            ):
                return FatalFailure(
                    "invalid_builder_authorization",
                    "AUTHORIZED_BUILDER requires the authorized actor, a producer, a publisher, and writable mode",
                )
            if build_events.launch_id != authorization.launch_id:
                return FatalFailure(
                    "builder_publisher_launch_mismatch",
                    "builder publisher and authorization launch IDs differ",
                )
            try:
                descriptor = producer.descriptor
            except Exception as exc:
                return FatalFailure(
                    "producer_descriptor_failed",
                    f"producer descriptor raised {safe_exception_summary(exc)}",
                )
            if descriptor != spec.producer:
                return FatalFailure(
                    "producer_descriptor_mismatch",
                    "supplied producer does not match the artifact spec",
                )
        elif role is BuildRole.ORDERED_WAITER:
            if (
                authorization.actor_id == authorization.authorized_builder_actor_id
                or producer is not None
                or build_events is not None
                or observed is None
            ):
                return FatalFailure(
                    "invalid_ordered_waiter_authorization",
                    "ORDERED_WAITER requires a distinct producerless actor and one observed start",
                )
            if (
                observed.launch_id != authorization.launch_id
                or observed.artifact_key != spec.artifact_key
                or observed.builder_actor_id != authorization.authorized_builder_actor_id
            ):
                return FatalFailure(
                    "builder_start_identity_mismatch",
                    "observed builder start does not match the launch, artifact, and authorized actor",
                )
        elif role is BuildRole.READ_ONLY:
            if producer is not None or build_events is not None or observed is not None:
                return FatalFailure(
                    "invalid_read_only_authorization",
                    "READ_ONLY cannot carry a producer, publisher, or observed start",
                )
        else:  # pragma: no cover - BuildRole construction rejects this shape.
            return FatalFailure("unknown_build_role", repr(role))
        return None

    def resolve(
        self,
        spec: ArtifactSpec,
        grant: CapabilityGrant,
        producer: ArtifactProducer | None,
        build_authorization: BuildAuthorization,
        artifact_registrar: ResolvedArtifactRegistrar,
        build_events: BuilderStartPublisher | None = None,
        *,
        wait_timeout_s: float = 30.0,
    ) -> ResolveOutcome:
        self._require_resolution_ready()

        def signal_failed_if_pending(outcome: ResolveOutcome) -> ResolveOutcome:
            if build_events is not None and not isinstance(outcome, Ready):
                build_events.publish_failed_if_pending(
                    BuilderStartFailed(
                        build_events.launch_id,
                        spec.artifact_key,
                        outcome.code,
                        outcome.detail,
                    )
                )
            return outcome

        def open_ready(
            existing: Existing,
            path: ResolutionPath,
        ) -> ResolveOutcome:
            outcome = self._ready_from_record(
                spec,
                existing.record,
                path,
                grant,
                artifact_registrar,
            )
            if isinstance(outcome, Ready):
                if build_events is not None:
                    build_events.publish_ready_if_pending(
                        ArtifactAlreadyReady(
                            build_events.launch_id,
                            spec.artifact_key,
                        )
                    )
                return outcome
            return signal_failed_if_pending(outcome)

        if failure := self._validate_grant(grant):
            return signal_failed_if_pending(failure)
        if failure := self._validate_artifact_registrar(artifact_registrar):
            return signal_failed_if_pending(failure)
        if grant.backing_kind is not BackingKind.RUNTIME_MMAP:
            return signal_failed_if_pending(
                FatalFailure(
                    "capability_grant_backing_mismatch",
                    "runtime-mmap resolve requires a runtime-mmap capability grant",
                )
            )
        if failure := self._validate_build_authorization(
            spec,
            build_authorization,
            producer,
            build_events,
            writable=self.writable,
        ):
            return signal_failed_if_pending(failure)
        repository = self.repository
        if repository is None:
            return signal_failed_if_pending(
                FatalFailure(
                    "repository_unavailable",
                    "the negotiated runtime-mmap provider has no artifact repository",
                )
            )

        try:
            existing = repository.lookup(spec.artifact_key)
        except StoreCorruptionError as exc:
            return signal_failed_if_pending(FatalFailure("corrupt_store", safe_exception_detail(exc)))
        except (StoreError, OSError) as exc:
            return signal_failed_if_pending(RetryableFailure("store_lookup_failed", safe_exception_detail(exc)))
        if existing is not None:
            observed = build_authorization.observed_start
            if observed is not None and existing.record.publication_lease_id != observed.lease_id:
                return RetryableFailure(
                    "builder_generation_changed",
                    spec.artifact_key,
                )
            return open_ready(existing, ResolutionPath.MMAP_HIT)

        build_intent = (
            BuildIntent(spec.producer, _owner_lost_failure(spec.artifact_key))
            if build_authorization.role is BuildRole.AUTHORIZED_BUILDER
            else None
        )

        try:
            decision = repository.claim(spec.artifact_key, build_intent)
        except StoreCorruptionError as exc:
            return signal_failed_if_pending(FatalFailure("corrupt_store", safe_exception_detail(exc)))
        except (StoreError, OSError) as exc:
            return signal_failed_if_pending(RetryableFailure("store_claim_failed", safe_exception_detail(exc)))

        if isinstance(decision, Existing):
            observed = build_authorization.observed_start
            if observed is not None and decision.record.publication_lease_id != observed.lease_id:
                return RetryableFailure(
                    "builder_generation_changed",
                    spec.artifact_key,
                )
            return open_ready(decision, ResolutionPath.MMAP_HIT)
        if isinstance(decision, Waiter):
            observed = build_authorization.observed_start
            if observed is not None and decision.observed_lease_id != observed.lease_id:
                return signal_failed_if_pending(
                    RetryableFailure(
                        "builder_generation_changed",
                        spec.artifact_key,
                    )
                )
            try:
                waited = repository.wait(decision, wait_timeout_s)
            except StoreCorruptionError as exc:
                return signal_failed_if_pending(FatalFailure("corrupt_store", safe_exception_detail(exc)))
            except (StoreError, OSError) as exc:
                return signal_failed_if_pending(
                    RetryableFailure(
                        "publication_wait_failed",
                        safe_exception_detail(exc),
                    )
                )
            if isinstance(waited, BuilderFailed):
                return signal_failed_if_pending(_outcome_from_build_failure(waited.failure))
            if isinstance(waited, WaitTimeout):
                reason = "timed out" if waited.builder_active else "builder exited"
                return signal_failed_if_pending(
                    RetryableFailure(
                        "publication_not_ready",
                        f"artifact publication {reason} before a generation became visible",
                    )
                )
            if observed is not None and waited.record.publication_lease_id != observed.lease_id:
                return signal_failed_if_pending(
                    RetryableFailure(
                        "builder_generation_changed",
                        spec.artifact_key,
                    )
                )
            return open_ready(waited, ResolutionPath.MMAP_WAIT_HIT)
        if isinstance(decision, NoBuilder):
            observed = build_authorization.observed_start
            if observed is not None:
                terminal = repository.wait(
                    Waiter(repository, spec.artifact_key, observed.lease_id),
                    0,
                )
                if isinstance(terminal, BuilderFailed):
                    return _outcome_from_build_failure(terminal.failure)
                if isinstance(terminal, Existing):
                    if terminal.record.publication_lease_id != observed.lease_id:
                        return RetryableFailure(
                            "builder_generation_changed",
                            spec.artifact_key,
                        )
                    return open_ready(terminal, ResolutionPath.MMAP_WAIT_HIT)
                return RetryableFailure(
                    "builder_disappeared_after_start",
                    spec.artifact_key,
                )
            return signal_failed_if_pending(
                RetryableFailure(
                    "no_builder",
                    spec.artifact_key,
                )
            )

        assert isinstance(decision, Builder)
        if producer is None or build_intent is None:
            decision.lease.abort()
            return FatalFailure(
                "invalid_builder_election",
                "repository elected a builder without a matching build intent",
            )
        try:
            with decision.lease:
                try:
                    stage = BuildFailureStage.INITIAL_SIGNAL
                    if build_events is not None:
                        published = build_events.publish_started_if_pending(
                            BuilderStarted(
                                launch_id=build_events.launch_id,
                                artifact_key=spec.artifact_key,
                                lease_id=decision.lease.lease_id,
                                builder_actor_id=build_authorization.actor_id,
                                monotonic_time_ns=time.monotonic_ns(),
                            )
                        )
                        if not published:
                            raise RuntimeError("builder initial gate signal was already published")
                    stage = BuildFailureStage.SINK_CREATE
                    sink = repository.create_sink(decision.lease)
                    try:
                        manifest = _build_manifest(
                            producer,
                            sink,
                            self._build_session_cleanup_registry,
                        )
                    except _ProducerPhaseError as phase_failure:
                        failure = _classify_build_failure(
                            phase_failure.stage,
                            phase_failure.primary,
                        )
                        decision.lease.record_failure(failure)
                        raise _ClassifiedBuildError(failure) from phase_failure.primary
                    stage = BuildFailureStage.SEMANTIC_VALIDATION
                    validated = validate_manifest_against_spec(spec, manifest)
                    stage = BuildFailureStage.COMMIT
                    record = repository.commit(
                        decision.lease,
                        sink,
                        validated,
                    )
                except BaseException as exc:
                    if isinstance(exc, _ClassifiedBuildError):
                        raise
                    failure = _classify_build_failure(stage, exc)
                    decision.lease.record_failure(failure)
                    raise _ClassifiedBuildError(failure) from exc
        except _ClassifiedBuildError as classified:
            return _outcome_from_build_failure(classified.failure)

        # The runtime registry released every producer-private object before
        # validation and publication.
        return self._ready_from_record(
            spec,
            record,
            ResolutionPath.MMAP_BUILT,
            grant,
            artifact_registrar,
        )

    def resolve_loaded(
        self,
        spec: ArtifactSpec,
        grant: CapabilityGrant,
        tensors: Mapping[str, torch.Tensor],
        artifact_registrar: ResolvedArtifactRegistrar,
        *,
        roles: Mapping[str, TensorRole] | None = None,
        format_metadata: Mapping[str, JSONValue] | None = None,
    ) -> ResolveOutcome:
        self._require_resolution_ready()
        if failure := self._validate_grant(grant):
            return failure
        if failure := self._validate_artifact_registrar(artifact_registrar):
            return failure
        if grant.backing_kind is not BackingKind.LOADED_TENSOR:
            return FatalFailure(
                "capability_grant_backing_mismatch",
                "loaded resolution requires a loaded-tensor capability grant",
            )
        try:
            provider = self._provider_for_grant(grant)
            open_ephemeral = getattr(provider, "open_ephemeral", None)
            if not callable(open_ephemeral):
                raise BackingError("negotiated provider cannot open an ephemeral tensor source")
            backing = open_ephemeral(
                spec,
                tensors,
                roles=roles,
                format_metadata=format_metadata,
            )
            try:
                record = ArtifactRecord(backing.manifest, backing.backing_index)
                validate_manifest_against_spec(spec, record.manifest)
                index = record.backing_index
                if (
                    index.kind is not grant.backing_kind
                    or index.provider_name != grant.provider_id
                    or index.provider_version != grant.provider_abi
                ):
                    raise BackingError("ephemeral backing does not match the negotiated capability grant")
                self._validate_opened_backing(backing, record)
            except BaseException as primary:
                try:
                    backing.close()
                except BaseException as cleanup_exc:
                    safe_add_exception_note(
                        primary,
                        "invalid ephemeral backing cleanup also failed",
                        cleanup_exc,
                    )
                raise
            artifact = self._register(backing, artifact_registrar)
        except RuntimeClosed:
            raise
        except BaseException as exc:
            return FatalFailure("invalid_loaded_artifact", safe_exception_detail(exc))
        return Ready(
            artifact,
            ResolutionInfo(
                path=ResolutionPath.LOADED,
                artifact_key=artifact.manifest.artifact_key,
                generation_id=record.backing_index.generation_id,
                backing_kind=BackingKind.LOADED_TENSOR,
            ),
            ResolvedAccess(
                runtime_instance_id=grant.runtime_instance_id,
                grant_id=grant.grant_id,
                backing_kind=grant.backing_kind,
                provider_id=grant.provider_id,
                provider_abi=grant.provider_abi,
                features=grant.features,
            ),
        )

    def close(self) -> None:
        self._build_session_cleanup_registry.close_pending()
        self._close_pending_artifact_assemblies()
        with self._lock:
            if self._closed:
                return
            if self._artifact_registrations:
                raise RuntimeBusy(
                    "cannot close host-weight runtime with "
                    f"{len(self._artifact_registrations)} resolved artifact(s) live"
                )
            self._closed = True

    def __enter__(self) -> HostWeightRuntime:
        self._require_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = [
    "HostWeightRuntime",
    "RuntimeBusy",
    "RuntimeCapabilities",
    "RuntimeClosed",
]
