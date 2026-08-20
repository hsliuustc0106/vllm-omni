# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Authority-boundary tests for capability grants and build roles."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from conftest import ArtifactRegistrar

from vllm_omni.host_weight_runtime import (
    AccessFeature,
    AccessRequirements,
    ArtifactSpec,
    BackingCapabilities,
    BackingKind,
    BuildAuthorization,
    BuilderInitialSignalState,
    BuilderStarted,
    BuildRole,
    CapabilityGrant,
    FatalFailure,
    LocalArtifactRepository,
)
from vllm_omni.host_weight_runtime import (
    HostWeightRuntime as InjectedHostWeightRuntime,
)
from vllm_omni.host_weight_runtime import (
    create_default_host_weight_runtime as HostWeightRuntime,
)


class _NoLookupRepository(LocalArtifactRepository):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.lookup_calls = 0

    def lookup(self, _artifact_key: str):
        self.lookup_calls += 1
        raise AssertionError("invalid authority must fail before repository lookup")


class _DescriptorOnlyProducer:
    def __init__(self, spec: ArtifactSpec) -> None:
        self._descriptor = spec.producer
        self.open_calls = 0

    @property
    def descriptor(self):
        return self._descriptor

    def open_build(self, _cleanup_registry):
        self.open_calls += 1
        raise AssertionError("invalid authority must not open a producer")


class _Publisher:
    def __init__(self, launch_id: str = "launch-1") -> None:
        self.launch_id = launch_id
        self.initial_signal_state = BuilderInitialSignalState.PENDING

    def publish_started_if_pending(self, _event) -> bool:
        if self.initial_signal_state is not BuilderInitialSignalState.PENDING:
            return False
        self.initial_signal_state = BuilderInitialSignalState.STARTED
        return True

    def publish_ready_if_pending(self, _event) -> bool:
        if self.initial_signal_state is not BuilderInitialSignalState.PENDING:
            return False
        self.initial_signal_state = BuilderInitialSignalState.READY
        return True

    def publish_failed_if_pending(self, _event) -> bool:
        if self.initial_signal_state is not BuilderInitialSignalState.PENDING:
            return False
        self.initial_signal_state = BuilderInitialSignalState.FAILED
        return True


def _grant(runtime: InjectedHostWeightRuntime) -> CapabilityGrant:
    grant = runtime.negotiate(
        AccessRequirements(
            required_features=frozenset(
                {
                    AccessFeature.COMPLETE_TENSOR_READ,
                    AccessFeature.SHARED_PAGES,
                }
            ),
            accepted_backings=frozenset({BackingKind.RUNTIME_MMAP}),
        )
    )
    assert isinstance(grant, CapabilityGrant)
    return grant


def _read_only_authorization() -> BuildAuthorization:
    return BuildAuthorization(
        BuildRole.READ_ONLY,
        "reader",
        "builder",
        "launch-1",
    )


def test_forged_grant_is_rejected_before_lookup(
    tmp_path: Path,
    artifact_spec: ArtifactSpec,
) -> None:
    repository = _NoLookupRepository(tmp_path)
    runtime = HostWeightRuntime(repository)
    forged = replace(_grant(runtime), grant_id="forged-grant")

    outcome = runtime.resolve(
        artifact_spec,
        forged,
        None,
        _read_only_authorization(),
        ArtifactRegistrar(),
    )

    assert isinstance(outcome, FatalFailure)
    assert outcome.code == "invalid_capability_grant"
    assert repository.lookup_calls == 0
    runtime.close()


def test_invalid_artifact_registrar_is_rejected_before_lookup(
    tmp_path: Path,
    artifact_spec: ArtifactSpec,
) -> None:
    repository = _NoLookupRepository(tmp_path)
    runtime = HostWeightRuntime(repository)

    outcome = runtime.resolve(
        artifact_spec,
        _grant(runtime),
        None,
        _read_only_authorization(),
        object(),  # type: ignore[arg-type]
    )

    assert isinstance(outcome, FatalFailure)
    assert outcome.code == "invalid_artifact_registrar"
    assert repository.lookup_calls == 0
    runtime.close()


def test_cross_runtime_grant_is_rejected_before_lookup(
    tmp_path: Path,
    artifact_spec: ArtifactSpec,
) -> None:
    issuer = HostWeightRuntime(LocalArtifactRepository(tmp_path / "issuer"))
    repository = _NoLookupRepository(tmp_path / "resolver")
    resolver = HostWeightRuntime(repository)

    outcome = resolver.resolve(
        artifact_spec,
        _grant(issuer),
        None,
        _read_only_authorization(),
        ArtifactRegistrar(),
    )

    assert isinstance(outcome, FatalFailure)
    assert outcome.code == "invalid_capability_grant"
    assert repository.lookup_calls == 0
    issuer.close()
    resolver.close()


def test_stale_grant_is_rejected_before_lookup(
    tmp_path: Path,
    artifact_spec: ArtifactSpec,
) -> None:
    class MutableProvider:
        provider_abi = "1"

        def capabilities(self):
            return BackingCapabilities(
                BackingKind.RUNTIME_MMAP,
                "local-runtime-mmap",
                self.provider_abi,
                frozenset(
                    {
                        AccessFeature.COMPLETE_TENSOR_READ,
                        AccessFeature.SHARED_PAGES,
                    }
                ),
            )

        def open(self, _manifest, _index):
            raise AssertionError("stale grant must fail before provider open")

    class MutableRegistry:
        def __init__(self) -> None:
            self.provider = MutableProvider()

        def capabilities(self):
            return (self.provider.capabilities(),)

        def provider_for(self, kind):
            return self.provider if kind is BackingKind.RUNTIME_MMAP else None

    repository = _NoLookupRepository(tmp_path)
    registry = MutableRegistry()
    runtime = InjectedHostWeightRuntime(repository, registry)
    grant = _grant(runtime)
    registry.provider.provider_abi = "2"

    outcome = runtime.resolve(
        artifact_spec,
        grant,
        None,
        _read_only_authorization(),
        ArtifactRegistrar(),
    )

    assert isinstance(outcome, FatalFailure)
    assert outcome.code == "stale_capability_grant"
    assert repository.lookup_calls == 0
    runtime.close()


def test_repository_disappearing_after_runtime_mmap_grant_is_fatal(
    tmp_path: Path,
    artifact_spec: ArtifactSpec,
) -> None:
    runtime = HostWeightRuntime(LocalArtifactRepository(tmp_path))
    grant = _grant(runtime)
    runtime.repository = None

    outcome = runtime.resolve(
        artifact_spec,
        grant,
        None,
        _read_only_authorization(),
        ArtifactRegistrar(),
    )

    assert outcome == FatalFailure(
        "repository_unavailable",
        "the negotiated runtime-mmap provider has no artifact repository",
    )
    runtime.close()


def test_invalid_build_authorization_shapes_fail_before_lookup(
    tmp_path: Path,
    artifact_spec: ArtifactSpec,
) -> None:
    repository = _NoLookupRepository(tmp_path)
    runtime = HostWeightRuntime(repository)
    grant = _grant(runtime)
    producer = _DescriptorOnlyProducer(artifact_spec)
    observed = BuilderStarted(
        "launch-1",
        artifact_spec.artifact_key,
        "lease-1",
        "builder",
        1,
    )
    cases = (
        (
            BuildAuthorization(
                BuildRole.AUTHORIZED_BUILDER,
                "not-builder",
                "builder",
                "launch-1",
            ),
            producer,
            _Publisher(),
            "invalid_builder_authorization",
        ),
        (
            BuildAuthorization(
                BuildRole.AUTHORIZED_BUILDER,
                "builder",
                "builder",
                "launch-1",
            ),
            producer,
            None,
            "invalid_builder_authorization",
        ),
        (
            BuildAuthorization(
                BuildRole.AUTHORIZED_BUILDER,
                "builder",
                "builder",
                "launch-1",
            ),
            producer,
            _Publisher("another-launch"),
            "builder_publisher_launch_mismatch",
        ),
        (
            BuildAuthorization(
                BuildRole.ORDERED_WAITER,
                "builder",
                "builder",
                "launch-1",
                observed,
            ),
            None,
            None,
            "invalid_ordered_waiter_authorization",
        ),
        (
            BuildAuthorization(
                BuildRole.ORDERED_WAITER,
                "waiter",
                "builder",
                "launch-1",
            ),
            None,
            None,
            "invalid_ordered_waiter_authorization",
        ),
        (
            BuildAuthorization(
                BuildRole.ORDERED_WAITER,
                "waiter",
                "builder",
                "launch-2",
                observed,
            ),
            None,
            None,
            "builder_start_identity_mismatch",
        ),
        (
            _read_only_authorization(),
            producer,
            None,
            "invalid_read_only_authorization",
        ),
    )

    for authorization, supplied_producer, publisher, expected_code in cases:
        outcome = runtime.resolve(
            artifact_spec,
            grant,
            supplied_producer,
            authorization,
            ArtifactRegistrar(),
            publisher,
        )
        assert isinstance(outcome, FatalFailure)
        assert outcome.code == expected_code

    assert repository.lookup_calls == 0
    assert producer.open_calls == 0
    runtime.close()
