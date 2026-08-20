# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import json
import multiprocessing
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from conftest import ArtifactRegistrar, make_weight_format

from vllm_omni.host_weight_runtime import (
    AccessFeature,
    AccessRequirements,
    ArtifactManifest,
    ArtifactSpec,
    ArtifactTopologyDescriptor,
    BackingIntegrityError,
    BackingKind,
    BuildAuthorization,
    Builder,
    BuilderInitialSignalState,
    BuilderStarted,
    BuildFailureClassification,
    BuildFailureKind,
    BuildFailureStage,
    BuildIntent,
    BuildRole,
    CapabilityGrant,
    Existing,
    FatalFailure,
    LocalArtifactRepository,
    NoBuilder,
    ProducerDescriptor,
    Ready,
    ResolutionPath,
    RetryableFailure,
    RuntimeBusy,
    RuntimeMmapArtifactSink,
    RuntimeMmapBacking,
    StoreCorruptionError,
    StoreError,
    StorePublicationDurabilityError,
    TensorSelection,
    TopologyCoordinate,
    Waiter,
    create_default_backing_provider_registry,
)
from vllm_omni.host_weight_runtime import (
    HostWeightRuntime as InjectedHostWeightRuntime,
)
from vllm_omni.host_weight_runtime import (
    create_default_host_weight_runtime as HostWeightRuntime,
)
from vllm_omni.host_weight_runtime.store import (
    BACKING_INDEX_FILENAME,
    MANIFEST_FILENAME,
    WEIGHTS_FILENAME,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class _HostileMessageError(RuntimeError):
    def __str__(self) -> str:
        raise AssertionError("exception text must not be trusted")


class _HostileMessageOSError(OSError):
    def __str__(self) -> str:
        raise AssertionError("exception text must not be trusted")


class _HostileMessageStoreError(StoreError):
    def __str__(self) -> str:
        raise AssertionError("exception text must not be trusted")


def _make_spec() -> ArtifactSpec:
    producer = ProducerDescriptor("test-producer", "producer/v1", _digest("producer"))
    weight_format = make_weight_format(
        format_id="test-format",
        adapter_abi="format/v1",
        semantic_fingerprint=_digest("format"),
        target_module_type_id="test.transformer",
        normalized_config={"dtype": "float32"},
        kernel_identity={"kernel": "copy"},
    )
    return ArtifactSpec(
        source_fingerprint=_digest("source"),
        producer=producer,
        weight_format=weight_format,
        topology=ArtifactTopologyDescriptor((TopologyCoordinate("pp", 1, 0), TopologyCoordinate("tp", 1, 0))),
        layout_abi="test-layout/v1",
    )


class _Publisher:
    launch_id = "test-launch"

    def __init__(self) -> None:
        self.initial_signal_state = BuilderInitialSignalState.PENDING
        self.event = None

    def publish_started_if_pending(self, event):
        if self.initial_signal_state is not BuilderInitialSignalState.PENDING:
            return False
        self.initial_signal_state = BuilderInitialSignalState.STARTED
        self.event = event
        return True

    def publish_ready_if_pending(self, event):
        if self.initial_signal_state is not BuilderInitialSignalState.PENDING:
            return False
        self.initial_signal_state = BuilderInitialSignalState.READY
        self.event = event
        return True

    def publish_failed_if_pending(self, event):
        if self.initial_signal_state is not BuilderInitialSignalState.PENDING:
            return False
        self.initial_signal_state = BuilderInitialSignalState.FAILED
        self.event = event
        return True


def _owner_lost(spec: ArtifactSpec) -> BuildFailureClassification:
    return BuildFailureClassification(
        BuildFailureStage.OWNER_LOST,
        "builder_owner_lost",
        spec.artifact_key,
        BuildFailureKind.RETRYABLE,
    )


def _build_intent(spec: ArtifactSpec) -> BuildIntent:
    return BuildIntent(spec.producer, _owner_lost(spec))


def _grant(runtime: InjectedHostWeightRuntime) -> CapabilityGrant:
    grant = runtime.negotiate(
        AccessRequirements(
            frozenset(
                {
                    AccessFeature.COMPLETE_TENSOR_READ,
                    AccessFeature.SHARED_PAGES,
                }
            ),
            frozenset({BackingKind.RUNTIME_MMAP}),
        )
    )
    assert isinstance(grant, CapabilityGrant)
    return grant


def _resolve(
    runtime: InjectedHostWeightRuntime,
    spec: ArtifactSpec,
    producer=None,
    *,
    wait_timeout_s: float = 30.0,
    authorization: BuildAuthorization | None = None,
    publisher: _Publisher | None = None,
):
    if authorization is None:
        if producer is None:
            authorization = BuildAuthorization(
                BuildRole.READ_ONLY,
                "reader",
                "builder",
                "test-launch",
            )
        else:
            authorization = BuildAuthorization(
                BuildRole.AUTHORIZED_BUILDER,
                "builder",
                "builder",
                "test-launch",
            )
            publisher = publisher or _Publisher()
    registrar = ArtifactRegistrar()
    outcome = runtime.resolve(
        spec,
        _grant(runtime),
        producer,
        authorization,
        registrar,
        publisher,
        wait_timeout_s=wait_timeout_s,
    )
    if isinstance(outcome, Ready):
        assert registrar.artifact is outcome.artifact
    return outcome


class _BuildSession:
    def __init__(
        self,
        spec: ArtifactSpec,
        *,
        calls=None,
        events: list[str] | None = None,
        bad_manifest: bool = False,
        fail_enter: bool = False,
        fail_build: bool = False,
        fail_close: bool = False,
    ) -> None:
        self.spec = spec
        self.calls = calls
        self.events = events
        self.bad_manifest = bad_manifest
        self.fail_enter = fail_enter
        self.fail_build = fail_build
        self.fail_close = fail_close
        self.closed = False

    def __enter__(self):
        if self.events is not None:
            self.events.append("enter")
        if self.fail_enter:
            raise RuntimeError("enter failed")
        return self

    def __exit__(self, *_):
        self.close()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.events is not None:
            self.events.append("close")
        if self.fail_close:
            raise RuntimeError("cleanup failed")

    def build(self, sink):
        if self.calls is not None:
            with self.calls.get_lock():
                self.calls.value += 1
        if self.events is not None:
            self.events.append("build")
        if self.fail_build:
            raise RuntimeError("build failed")
        weight = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
        receipt = sink.write_tensor("transformer.weight", weight)
        manifest = ArtifactManifest.create(
            self.spec,
            (receipt,),
            format_metadata={"recipe_schema": 1},
        )
        if self.bad_manifest:
            manifest = replace(manifest, artifact_key=_digest("wrong artifact"))
        return manifest


class _Producer:
    def __init__(
        self,
        spec: ArtifactSpec,
        *,
        calls=None,
        events: list[str] | None = None,
        bad_manifest: bool = False,
        fail_open: bool = False,
        fail_enter: bool = False,
        fail_build: bool = False,
        fail_close: bool = False,
    ) -> None:
        self._spec = spec
        self._calls = calls
        self._events = events
        self._bad_manifest = bad_manifest
        self._fail_open = fail_open
        self._fail_enter = fail_enter
        self._fail_build = fail_build
        self._fail_close = fail_close
        self.last_session: _BuildSession | None = None

    @property
    def descriptor(self):
        return self._spec.producer

    def open_build(self, cleanup_registry):
        if self._events is not None:
            self._events.append("open_build")
        if self._fail_open:
            raise RuntimeError("open failed")
        self.last_session = _BuildSession(
            self._spec,
            calls=self._calls,
            events=self._events,
            bad_manifest=self._bad_manifest,
            fail_enter=self._fail_enter,
            fail_build=self._fail_build,
            fail_close=self._fail_close,
        )
        cleanup_registry.register_before_return(self.last_session)
        return self.last_session


class _PoisonProducer:
    def __init__(self, spec: ArtifactSpec, *, poison_descriptor: bool = False) -> None:
        self._spec = spec
        self.poison_descriptor = poison_descriptor
        self.open_calls = 0

    @property
    def descriptor(self):
        if self.poison_descriptor:
            raise AssertionError("descriptor must not be inspected")
        return self._spec.producer

    def open_build(self, _cleanup_registry):
        self.open_calls += 1
        raise AssertionError("warm/wait path must not open the producer")


def _resolve_worker(root: str, start, calls, results) -> None:
    spec = _make_spec()
    runtime = HostWeightRuntime(LocalArtifactRepository(root), writable=True)
    start.wait()
    outcome = _resolve(
        runtime,
        spec,
        _Producer(spec, calls=calls),
        wait_timeout_s=30,
    )
    if isinstance(outcome, Ready):
        results.put(("ready", outcome.info.path.value))
        outcome.artifact.close()
    else:
        results.put((type(outcome).__name__, outcome.code))
    runtime.close()


def _publish_direct(repository, decision: Builder, spec: ArtifactSpec) -> Existing:
    with decision.lease:
        sink = repository.create_sink(decision.lease)
        receipt = sink.write_tensor(
            "transformer.weight",
            torch.arange(4, dtype=torch.float32),
        )
        manifest = ArtifactManifest.create(spec, (receipt,), {"recipe_schema": 1})
        record = repository.commit(decision.lease, sink, manifest)
    return Existing(record, repository._artifact_dir(spec.artifact_key))


def test_runtime_mmap_backing_close_retains_only_failed_mappings_for_retry() -> None:
    class Mapping:
        def __init__(self, failure: BaseException | None = None) -> None:
            self.failure = failure
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            if self.failure is not None and self.close_calls == 1:
                raise self.failure

    primary = _HostileMessageError()
    first = Mapping()
    retry = Mapping(primary)
    last = Mapping()
    backing = object.__new__(RuntimeMmapBacking)
    backing._mappings = {"first": first, "retry": retry, "last": last}
    backing._closed = False
    backing._lock = threading.RLock()

    with pytest.raises(_HostileMessageError) as caught:
        backing.close()

    assert caught.value is primary
    assert first.close_calls == 1
    assert retry.close_calls == 1
    assert last.close_calls == 1
    assert tuple(backing._mappings) == ("retry",)
    assert backing._closed is False

    backing.close()
    assert retry.close_calls == 2
    assert backing._mappings == {}
    assert backing._closed is True
    backing.close()
    assert retry.close_calls == 2


def test_runtime_mmap_sink_abort_retries_failed_handle_close() -> None:
    class Handle:
        def __init__(self) -> None:
            self.close_calls = 0
            self.failure = _HostileMessageError()

        def close(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                raise self.failure

    handle = Handle()
    sink = object.__new__(RuntimeMmapArtifactSink)
    sink._finished = False
    sink._aborted = False
    sink._handle = handle
    sink._lock = threading.RLock()

    with pytest.raises(_HostileMessageError) as caught:
        sink.abort()

    assert caught.value is handle.failure
    assert sink._aborted is False
    sink.abort()
    assert sink._aborted is True
    assert handle.close_calls == 2
    sink.abort()
    assert handle.close_calls == 2


def test_integrity_scan_close_failure_does_not_mask_digest_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import vllm_omni.host_weight_runtime.backings as backings_module

    spec = _make_spec()
    repository = LocalArtifactRepository(tmp_path)
    decision = repository.claim(spec.artifact_key, _build_intent(spec))
    assert isinstance(decision, Builder)
    existing = _publish_direct(repository, decision, spec)
    weights_path = existing.artifact_dir / WEIGHTS_FILENAME
    with weights_path.open("r+b") as handle:
        first = handle.read(1)
        handle.seek(0)
        handle.write(bytes([first[0] ^ 0xFF]))

    original_open = backings_module.os.open
    original_close = backings_module.os.close
    target_fd: int | None = None
    close_failure = _HostileMessageOSError()

    def record_target_open(path, flags, *args, **kwargs):
        nonlocal target_fd
        fd = original_open(path, flags, *args, **kwargs)
        if Path(path) == weights_path:
            target_fd = fd
        return fd

    def close_target_then_fail(fd: int) -> None:
        original_close(fd)
        if fd == target_fd:
            raise close_failure

    monkeypatch.setattr(backings_module.os, "open", record_target_open)
    monkeypatch.setattr(backings_module.os, "close", close_target_then_fail)

    with pytest.raises(BackingIntegrityError, match="digest mismatch") as caught:
        backings_module.validate_runtime_record_files(existing.artifact_dir, existing.record)

    assert caught.value is not close_failure
    assert any("integrity descriptor close" in note for note in caught.value.__notes__)


def test_mmap_constructor_close_failure_does_not_mask_mapping_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import vllm_omni.host_weight_runtime.backings as backings_module

    spec = _make_spec()
    repository = LocalArtifactRepository(tmp_path)
    decision = repository.claim(spec.artifact_key, _build_intent(spec))
    assert isinstance(decision, Builder)
    existing = _publish_direct(repository, decision, spec)
    weights_path = existing.artifact_dir / WEIGHTS_FILENAME
    original_open = backings_module.os.open
    original_close = backings_module.os.close
    target_fd: int | None = None
    primary = _HostileMessageOSError()
    close_failure = _HostileMessageOSError()

    def record_target_open(path, flags, *args, **kwargs):
        nonlocal target_fd
        fd = original_open(path, flags, *args, **kwargs)
        if Path(path) == weights_path:
            target_fd = fd
        return fd

    def fail_mapping(*_args, **_kwargs):
        raise primary

    def close_target_then_fail(fd: int) -> None:
        original_close(fd)
        if fd == target_fd:
            raise close_failure

    monkeypatch.setattr(backings_module.os, "open", record_target_open)
    monkeypatch.setattr(backings_module.mmap, "mmap", fail_mapping)
    monkeypatch.setattr(backings_module.os, "close", close_target_then_fail)

    with pytest.raises(_HostileMessageOSError) as caught:
        RuntimeMmapBacking(
            existing.artifact_dir,
            existing.record,
            verify_integrity=False,
        )

    assert caught.value is primary
    assert any("source descriptor close" in note for note in primary.__notes__)


def test_staging_fsync_close_failure_preserves_exact_fsync_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import vllm_omni.host_weight_runtime.store as store_module

    spec = _make_spec()
    repository = LocalArtifactRepository(tmp_path)
    decision = repository.claim(spec.artifact_key, _build_intent(spec))
    assert isinstance(decision, Builder)
    sink = repository.create_sink(decision.lease)
    receipt = sink.write_tensor("transformer.weight", torch.arange(4, dtype=torch.float32))
    manifest = ArtifactManifest.create(spec, (receipt,), {"recipe_schema": 1})
    original_open = store_module.os.open
    original_fsync = store_module.os.fsync
    original_close = store_module.os.close
    directory_fd: int | None = None
    primary = _HostileMessageOSError()
    close_failure = _HostileMessageOSError()

    def record_directory_open(path, flags, *args, **kwargs):
        nonlocal directory_fd
        fd = original_open(path, flags, *args, **kwargs)
        if Path(path) == sink._staging_dir:
            directory_fd = fd
        return fd

    def fail_directory_fsync(fd: int) -> None:
        if fd == directory_fd:
            raise primary
        original_fsync(fd)

    def close_directory_then_fail(fd: int) -> None:
        original_close(fd)
        if fd == directory_fd:
            raise close_failure

    monkeypatch.setattr(store_module.os, "open", record_directory_open)
    monkeypatch.setattr(store_module.os, "fsync", fail_directory_fsync)
    monkeypatch.setattr(store_module.os, "close", close_directory_then_fail)

    with pytest.raises(_HostileMessageOSError) as caught:
        sink.finish(manifest)

    assert caught.value is primary
    assert any("staging-directory descriptor close" in note for note in primary.__notes__)
    monkeypatch.setattr(store_module.os, "close", original_close)
    decision.lease.abort()


def test_publication_fsync_close_failure_keeps_fsync_error_as_cause(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import vllm_omni.host_weight_runtime.store as store_module

    spec = _make_spec()
    repository = LocalArtifactRepository(tmp_path)
    decision = repository.claim(spec.artifact_key, _build_intent(spec))
    assert isinstance(decision, Builder)
    sink = repository.create_sink(decision.lease)
    receipt = sink.write_tensor("transformer.weight", torch.arange(4, dtype=torch.float32))
    manifest = ArtifactManifest.create(spec, (receipt,), {"recipe_schema": 1})
    record = sink.finish(manifest)
    assert decision.lease._staging_dir is not None
    original_open = store_module.os.open
    original_fsync = store_module.os.fsync
    original_close = store_module.os.close
    directory_fd: int | None = None
    primary = _HostileMessageOSError()
    close_failure = _HostileMessageOSError()

    def record_directory_open(path, flags, *args, **kwargs):
        nonlocal directory_fd
        fd = original_open(path, flags, *args, **kwargs)
        if Path(path) == repository._artifacts_dir:
            directory_fd = fd
        return fd

    def fail_directory_fsync(fd: int) -> None:
        if fd == directory_fd:
            raise primary
        original_fsync(fd)

    def close_directory_then_fail(fd: int) -> None:
        original_close(fd)
        if fd == directory_fd:
            raise close_failure

    monkeypatch.setattr(store_module.os, "open", record_directory_open)
    monkeypatch.setattr(store_module.os, "fsync", fail_directory_fsync)
    monkeypatch.setattr(store_module.os, "close", close_directory_then_fail)

    with pytest.raises(StorePublicationDurabilityError) as caught:
        repository._publish(
            decision.lease,
            decision.lease._staging_dir,
            record,
        )

    assert caught.value.__cause__ is primary
    assert any("publication-directory descriptor close" in note for note in primary.__notes__)
    assert repository._artifact_dir(spec.artifact_key).exists()
    assert decision.lease.active is True
    decision.lease.abort()
    assert decision.lease.active is False


def test_waiter_probe_close_failure_preserves_exact_unlock_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import vllm_omni.host_weight_runtime.store as store_module

    spec = _make_spec()
    repository = LocalArtifactRepository(tmp_path)
    original_open = store_module.os.open
    original_flock = store_module.fcntl.flock
    original_close = store_module.os.close
    target_fd: int | None = None
    primary = _HostileMessageOSError()
    close_failure = _HostileMessageOSError()

    def record_lock_open(path, flags, *args, **kwargs):
        nonlocal target_fd
        fd = original_open(path, flags, *args, **kwargs)
        if Path(path) == repository._lock_path(spec.artifact_key):
            target_fd = fd
        return fd

    def fail_unlock(fd: int, operation: int) -> None:
        if fd == target_fd and operation == store_module.fcntl.LOCK_UN:
            raise primary
        original_flock(fd, operation)

    def close_target_then_fail(fd: int) -> None:
        original_close(fd)
        if fd == target_fd:
            raise close_failure

    monkeypatch.setattr(store_module.os, "open", record_lock_open)
    monkeypatch.setattr(store_module.fcntl, "flock", fail_unlock)
    monkeypatch.setattr(store_module.os, "close", close_target_then_fail)

    with pytest.raises(_HostileMessageOSError) as caught:
        repository._publication_active(spec.artifact_key)

    assert caught.value is primary
    assert any("waiter-probe descriptor close" in note for note in primary.__notes__)


def test_lease_cleanup_preserves_body_failure_and_retries_pending_resources(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import vllm_omni.host_weight_runtime.store as store_module

    spec = _make_spec()
    repository = LocalArtifactRepository(tmp_path)
    decision = repository.claim(spec.artifact_key, _build_intent(spec))
    assert isinstance(decision, Builder)
    lease = decision.lease
    sink = repository.create_sink(lease)
    original_abort = sink.abort
    original_rmtree = store_module.shutil.rmtree
    sink_failure = _HostileMessageError()
    directory_failure = _HostileMessageError()
    sink_calls = 0
    rmtree_calls = 0

    def fail_sink_once() -> None:
        nonlocal sink_calls
        sink_calls += 1
        if sink_calls == 1:
            raise sink_failure
        original_abort()

    def fail_rmtree_once(path) -> None:
        nonlocal rmtree_calls
        rmtree_calls += 1
        if rmtree_calls == 1:
            raise directory_failure
        original_rmtree(path)

    monkeypatch.setattr(sink, "abort", fail_sink_once)
    monkeypatch.setattr(store_module.shutil, "rmtree", fail_rmtree_once)
    body_failure = _HostileMessageError()

    with pytest.raises(_HostileMessageError) as caught:
        with lease:
            raise body_failure

    assert caught.value is body_failure
    assert lease.active is False
    assert lease._fd_closed is True
    assert lease._lock_released is True
    assert lease._cleanup_complete is False
    assert lease._staging_dir is not None and lease._staging_dir.exists()
    assert any("artifact lease cleanup" in note for note in body_failure.__notes__)

    lease.abort()
    assert lease._cleanup_complete is True
    assert sink._aborted is True
    assert lease._staging_dir is not None and not lease._staging_dir.exists()
    assert (sink_calls, rmtree_calls) == (2, 2)
    lease.abort()
    assert (sink_calls, rmtree_calls) == (2, 2)


def test_claim_cleanup_preserves_primary_and_continues_after_unlock_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import vllm_omni.host_weight_runtime.store as store_module

    spec = _make_spec()
    repository = LocalArtifactRepository(tmp_path)
    primary = _HostileMessageStoreError()
    secondary = _HostileMessageError()
    lookup_calls = 0
    unlock_calls = 0
    close_calls = 0
    original_flock = store_module.fcntl.flock
    original_close = store_module.os.close
    lock_fd: int | None = None

    def fail_second_lookup(_artifact_key: str):
        nonlocal lookup_calls
        lookup_calls += 1
        if lookup_calls == 1:
            return None
        raise primary

    def fail_unlock_once(fd: int, operation: int) -> None:
        nonlocal unlock_calls, lock_fd
        if operation == store_module.fcntl.LOCK_UN:
            unlock_calls += 1
            lock_fd = fd
            raise secondary
        original_flock(fd, operation)

    def record_close(fd: int) -> None:
        nonlocal close_calls
        if fd == lock_fd:
            close_calls += 1
        original_close(fd)

    monkeypatch.setattr(repository, "lookup", fail_second_lookup)
    monkeypatch.setattr(store_module.fcntl, "flock", fail_unlock_once)
    monkeypatch.setattr(store_module.os, "close", record_close)

    with pytest.raises(_HostileMessageStoreError) as caught:
        repository.claim(spec.artifact_key, _build_intent(spec))

    assert caught.value is primary
    assert unlock_calls == 1
    assert close_calls == 1
    assert any("artifact claim lock release" in note for note in primary.__notes__)


def test_hostile_store_error_is_normalized_to_typed_resolution_detail(
    artifact_spec,
) -> None:
    class Repository:
        def lookup(self, _artifact_key: str):
            raise _HostileMessageStoreError()

    runtime = HostWeightRuntime(Repository())  # type: ignore[arg-type]

    outcome = _resolve(runtime, artifact_spec)

    assert isinstance(outcome, RetryableFailure)
    assert outcome.code == "store_lookup_failed"
    assert outcome.detail == "<_HostileMessageStoreError detail unavailable>"
    runtime.close()


def test_runtime_mmap_cold_build_and_poisoned_warm_hit_preserve_layout(
    tmp_path: Path,
) -> None:
    spec = _make_spec()
    runtime = HostWeightRuntime(LocalArtifactRepository(tmp_path))
    producer = _Producer(spec)
    first = _resolve(runtime, spec, producer)

    assert isinstance(first, Ready)
    assert first.info.path is ResolutionPath.MMAP_BUILT
    assert producer.last_session is not None and producer.last_session.closed
    assert first.artifact.manifest.tensor("transformer.weight").stride == (1, 4)
    destination = torch.empty(4, 3, dtype=torch.float32)
    with first.artifact.open(TensorSelection.one("transformer.weight")) as view:
        view.copy_into({"transformer.weight": destination})
    torch.testing.assert_close(
        destination,
        torch.arange(12, dtype=torch.float32).reshape(3, 4).t(),
    )
    generation = first.info.generation_id
    first.artifact.close()

    poison = _PoisonProducer(spec)
    second = _resolve(runtime, spec, poison)
    assert isinstance(second, Ready)
    assert second.info.path is ResolutionPath.MMAP_HIT
    assert second.info.generation_id == generation
    assert poison.open_calls == 0
    second.artifact.close()
    runtime.close()


def test_read_only_absent_artifact_reports_retryable_no_builder_without_producer(
    tmp_path: Path,
) -> None:
    spec = _make_spec()
    runtime = HostWeightRuntime(LocalArtifactRepository(tmp_path), writable=False)
    poison = _PoisonProducer(spec, poison_descriptor=True)

    outcome = _resolve(runtime, spec)

    assert isinstance(outcome, RetryableFailure)
    assert outcome.code == "no_builder"
    assert outcome.detail == spec.artifact_key
    assert poison.open_calls == 0
    runtime.close()


def test_producerless_claim_can_never_become_builder(tmp_path: Path) -> None:
    spec = _make_spec()
    repository = LocalArtifactRepository(tmp_path)

    decision = repository.claim(spec.artifact_key, None)

    assert isinstance(decision, NoBuilder)
    assert list((tmp_path / "staging").iterdir()) == []


def test_producerless_waiter_observes_active_builder_publish(tmp_path: Path) -> None:
    spec = _make_spec()
    repository = LocalArtifactRepository(tmp_path)
    decision = repository.claim(spec.artifact_key, _build_intent(spec))
    assert isinstance(decision, Builder)
    runtime = HostWeightRuntime(repository, writable=False)

    def publish() -> None:
        time.sleep(0.05)
        _publish_direct(repository, decision, spec)

    thread = threading.Thread(target=publish)
    thread.start()
    outcome = _resolve(runtime, spec, wait_timeout_s=2)
    thread.join()

    assert isinstance(outcome, Ready)
    assert outcome.info.path is ResolutionPath.MMAP_WAIT_HIT
    outcome.artifact.close()
    runtime.close()


def test_wait_rechecks_lookup_after_builder_becomes_inactive(
    tmp_path: Path,
    monkeypatch,
) -> None:
    spec = _make_spec()
    repository = LocalArtifactRepository(tmp_path)
    builder = repository.claim(spec.artifact_key, _build_intent(spec))
    assert isinstance(builder, Builder)
    waiter = Waiter(repository, spec.artifact_key)
    original_publication_active = repository._publication_active
    publication_checks = 0

    def publish_before_lock_check(artifact_key: str) -> bool:
        nonlocal publication_checks
        publication_checks += 1
        _publish_direct(repository, builder, spec)
        return original_publication_active(artifact_key)

    monkeypatch.setattr(repository, "_publication_active", publish_before_lock_check)

    outcome = repository.wait(waiter, timeout_s=0)

    assert isinstance(outcome, Existing)
    assert publication_checks == 1
    assert outcome.record.manifest.artifact_key == spec.artifact_key


def test_poison_open_build_is_not_called_for_waiter(tmp_path: Path) -> None:
    spec = _make_spec()
    repository = LocalArtifactRepository(tmp_path)
    decision = repository.claim(spec.artifact_key, _build_intent(spec))
    assert isinstance(decision, Builder)
    runtime = HostWeightRuntime(repository, writable=True)
    poison = _PoisonProducer(spec)

    thread = threading.Thread(target=lambda: (time.sleep(0.05), _publish_direct(repository, decision, spec)))
    thread.start()
    outcome = _resolve(runtime, spec, poison, wait_timeout_s=2)
    thread.join()

    assert isinstance(outcome, Ready)
    assert outcome.info.path is ResolutionPath.MMAP_WAIT_HIT
    assert poison.open_calls == 0
    outcome.artifact.close()
    runtime.close()


def test_only_authorized_builder_role_can_open_lazy_producer(
    tmp_path: Path,
) -> None:
    spec = _make_spec()
    runtime = HostWeightRuntime(LocalArtifactRepository(tmp_path), writable=True)
    poison = _PoisonProducer(spec)

    read_only = _resolve(
        runtime,
        spec,
        poison,
        authorization=BuildAuthorization(
            BuildRole.READ_ONLY,
            "reader",
            "builder",
            "test-launch",
        ),
    )
    ordered = _resolve(
        runtime,
        spec,
        poison,
        authorization=BuildAuthorization(
            BuildRole.ORDERED_WAITER,
            "waiter",
            "builder",
            "test-launch",
            BuilderStarted(
                "test-launch",
                spec.artifact_key,
                "observed-lease",
                "builder",
                1,
            ),
        ),
    )

    publisher = _Publisher()
    authorized = _resolve(
        runtime,
        spec,
        poison,
        authorization=BuildAuthorization(
            BuildRole.AUTHORIZED_BUILDER,
            "builder",
            "builder",
            "test-launch",
        ),
        publisher=publisher,
    )

    assert isinstance(read_only, FatalFailure)
    assert read_only.code == "invalid_read_only_authorization"
    assert isinstance(ordered, FatalFailure)
    assert ordered.code == "invalid_ordered_waiter_authorization"
    assert isinstance(authorized, FatalFailure)
    assert authorized.code == "artifact_producer_open_failed"
    assert poison.open_calls == 1
    assert publisher.initial_signal_state is BuilderInitialSignalState.STARTED
    runtime.close()


def test_ordered_waiter_requires_and_returns_exact_started_generation(
    tmp_path: Path,
) -> None:
    spec = _make_spec()
    repository = LocalArtifactRepository(tmp_path)
    decision = repository.claim(spec.artifact_key, _build_intent(spec))
    assert isinstance(decision, Builder)
    runtime = HostWeightRuntime(repository)
    observed = BuilderStarted(
        "test-launch",
        spec.artifact_key,
        decision.lease.lease_id,
        "builder",
        1,
    )
    authorization = BuildAuthorization(
        BuildRole.ORDERED_WAITER,
        "waiter",
        "builder",
        "test-launch",
        observed,
    )

    thread = threading.Thread(
        target=lambda: (
            time.sleep(0.05),
            _publish_direct(repository, decision, spec),
        )
    )
    thread.start()
    registrar = ArtifactRegistrar()
    outcome = runtime.resolve(
        spec,
        _grant(runtime),
        None,
        authorization,
        registrar,
        wait_timeout_s=2,
    )
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert isinstance(outcome, Ready)
    assert outcome.info.path is ResolutionPath.MMAP_WAIT_HIT
    assert outcome.info.generation_id == observed.lease_id
    assert registrar.artifact is outcome.artifact
    outcome.artifact.close()

    mismatched = replace(observed, lease_id="another-lease")
    changed = runtime.resolve(
        spec,
        _grant(runtime),
        None,
        replace(authorization, observed_start=mismatched),
        ArtifactRegistrar(),
    )
    assert isinstance(changed, RetryableFailure)
    assert changed.code == "builder_generation_changed"
    runtime.close()


def test_ordered_waiter_recovers_exact_terminal_failure_after_builder_exit(
    tmp_path: Path,
) -> None:
    spec = _make_spec()
    repository = LocalArtifactRepository(tmp_path)
    decision = repository.claim(spec.artifact_key, _build_intent(spec))
    assert isinstance(decision, Builder)
    observed = BuilderStarted(
        "test-launch",
        spec.artifact_key,
        decision.lease.lease_id,
        "builder",
        1,
    )
    failure = BuildFailureClassification(
        BuildFailureStage.COMMIT,
        "exact_terminal_failure",
        "preserve this detail",
        BuildFailureKind.RETRYABLE,
        retry_after_s=1.25,
    )
    decision.lease.record_failure(failure)
    decision.lease.record_failure(
        BuildFailureClassification(
            BuildFailureStage.OWNER_LOST,
            "must_not_replace_primary",
            "secondary envelope",
            BuildFailureKind.FATAL,
        )
    )
    decision.lease.abort()
    runtime = HostWeightRuntime(repository)

    outcome = runtime.resolve(
        spec,
        _grant(runtime),
        None,
        BuildAuthorization(
            BuildRole.ORDERED_WAITER,
            "waiter",
            "builder",
            "test-launch",
            observed,
        ),
        ArtifactRegistrar(),
        wait_timeout_s=0,
    )

    assert outcome == RetryableFailure(
        failure.code,
        failure.detail,
        failure.retry_after_s,
    )
    runtime.close()


def test_authorized_rank_waiting_on_another_builder_fails_pending_gate(
    tmp_path: Path,
) -> None:
    spec = _make_spec()
    repository = LocalArtifactRepository(tmp_path)
    active = repository.claim(spec.artifact_key, _build_intent(spec))
    assert isinstance(active, Builder)
    failure = BuildFailureClassification(
        BuildFailureStage.PRODUCER_BUILD,
        "other_builder_failed",
        "exact peer failure",
        BuildFailureKind.FATAL,
    )
    runtime = HostWeightRuntime(repository)
    poison = _PoisonProducer(spec)
    publisher = _Publisher()

    def release_failed_builder() -> None:
        time.sleep(0.05)
        active.lease.record_failure(failure)
        active.lease.abort()

    thread = threading.Thread(target=release_failed_builder)
    thread.start()
    outcome = _resolve(
        runtime,
        spec,
        poison,
        wait_timeout_s=2,
        authorization=BuildAuthorization(
            BuildRole.AUTHORIZED_BUILDER,
            "builder",
            "builder",
            "test-launch",
        ),
        publisher=publisher,
    )
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert outcome == FatalFailure(failure.code, failure.detail)
    assert poison.open_calls == 0
    assert publisher.initial_signal_state is BuilderInitialSignalState.FAILED
    assert publisher.event.code == failure.code
    runtime.close()


def test_runtime_retains_failed_build_session_cleanup_until_retry(
    tmp_path: Path,
) -> None:
    spec = _make_spec()
    repository = LocalArtifactRepository(tmp_path)
    runtime = HostWeightRuntime(repository)

    class RetryCloseSession(_BuildSession):
        def __init__(self) -> None:
            super().__init__(spec)
            self.close_calls = 0

        def close(self) -> None:
            if self.closed:
                return
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("injected one-shot cleanup failure")
            self.closed = True

    session = RetryCloseSession()

    class RetryCloseProducer:
        @property
        def descriptor(self):
            return spec.producer

        def open_build(self, cleanup_registry):
            cleanup_registry.register_before_return(session)
            return session

    outcome = _resolve(runtime, spec, RetryCloseProducer())

    assert isinstance(outcome, FatalFailure)
    assert outcome.code == "artifact_builder_cleanup_failed"
    assert session.close_calls == 1
    assert repository.lookup(spec.artifact_key) is None
    with pytest.raises(RuntimeBusy, match="cleanup remains pending"):
        _resolve(runtime, spec)

    runtime.close()
    assert session.close_calls == 2
    assert session.closed


def test_four_process_contenders_invoke_exactly_one_lazy_producer(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    calls = context.Value("i", 0)
    results = context.Queue()
    processes = [
        context.Process(
            target=_resolve_worker,
            args=(str(tmp_path), start, calls, results),
        )
        for _ in range(4)
    ]
    for process in processes:
        process.start()
    start.set()
    observed = [results.get(timeout=90) for _ in processes]
    for process in processes:
        process.join(timeout=30)

    assert calls.value == 1
    assert all(item[0] == "ready" for item in observed), observed
    assert sum(item[1] == ResolutionPath.MMAP_BUILT.value for item in observed) == 1
    assert all(process.exitcode == 0 for process in processes)


def test_build_session_closes_before_runtime_opens_ready(tmp_path: Path) -> None:
    spec = _make_spec()
    events: list[str] = []
    producer = _Producer(spec, events=events)

    class CheckingRuntime(InjectedHostWeightRuntime):
        def _ready_from_record(self, spec, record, path, grant, artifact_registrar):
            if path is ResolutionPath.MMAP_BUILT:
                assert producer.last_session is not None
                assert producer.last_session.closed
                events.append("open_ready")
            return super()._ready_from_record(
                spec,
                record,
                path,
                grant,
                artifact_registrar,
            )

    repository = LocalArtifactRepository(tmp_path)
    runtime = CheckingRuntime(
        repository,
        create_default_backing_provider_registry(repository),
    )
    outcome = _resolve(runtime, spec, producer)

    assert isinstance(outcome, Ready)
    assert events == ["open_build", "build", "close", "open_ready"]
    outcome.artifact.close()
    runtime.close()


def test_artifact_is_not_visible_until_build_session_close_succeeds(tmp_path: Path) -> None:
    spec = _make_spec()
    repository = LocalArtifactRepository(tmp_path)
    runtime = HostWeightRuntime(repository)
    close_started = threading.Event()
    allow_close = threading.Event()
    outcomes: list[object] = []

    class BlockingCloseSession(_BuildSession):
        def close(self) -> None:
            if self.closed:
                return
            close_started.set()
            if not allow_close.wait(timeout=5):
                raise RuntimeError("timed out waiting to close")
            super().close()

    class BlockingCloseProducer(_Producer):
        def open_build(self, cleanup_registry):
            self.last_session = BlockingCloseSession(self._spec)
            cleanup_registry.register_before_return(self.last_session)
            return self.last_session

    thread = threading.Thread(target=lambda: outcomes.append(_resolve(runtime, spec, BlockingCloseProducer(spec))))
    thread.start()
    try:
        assert close_started.wait(timeout=5)
        assert repository.lookup(spec.artifact_key) is None
        assert isinstance(repository.claim(spec.artifact_key, None), Waiter)
    finally:
        allow_close.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert len(outcomes) == 1
    assert isinstance(outcomes[0], Ready)
    assert repository.lookup(spec.artifact_key) is not None
    outcomes[0].artifact.close()
    runtime.close()


def test_build_session_cleanup_failure_aborts_before_publication(tmp_path: Path) -> None:
    spec = _make_spec()
    repository = LocalArtifactRepository(tmp_path)
    runtime = HostWeightRuntime(repository)

    outcome = _resolve(runtime, spec, _Producer(spec, fail_close=True))

    assert isinstance(outcome, FatalFailure)
    assert outcome.code == "artifact_builder_cleanup_failed"
    assert repository.lookup(spec.artifact_key) is None
    assert list((tmp_path / "staging").iterdir()) == []
    runtime.close()


def test_post_rename_durability_failure_is_fatal_and_preserves_target(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import vllm_omni.host_weight_runtime.store as store_module

    spec = _make_spec()
    repository = LocalArtifactRepository(tmp_path)
    runtime = HostWeightRuntime(repository)
    original_open = store_module.os.open

    def fail_publication_directory_open(path, flags, *args, **kwargs):
        if Path(path) == repository._artifacts_dir:
            raise OSError("injected post-rename directory open failure")
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(store_module.os, "open", fail_publication_directory_open)

    outcome = _resolve(runtime, spec, _Producer(spec))

    assert isinstance(outcome, FatalFailure)
    assert outcome.code == "artifact_publication_durability_failed"
    assert "publication is visible" in outcome.detail
    assert repository.lookup(spec.artifact_key) is not None
    assert repository._publication_active(spec.artifact_key) is False
    assert list((tmp_path / "staging").iterdir()) == []
    runtime.close()


def test_post_rename_hostile_error_reporting_remains_typed_and_releases_lease(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import vllm_omni.host_weight_runtime.store as store_module

    spec = _make_spec()
    repository = LocalArtifactRepository(tmp_path)
    runtime = HostWeightRuntime(repository)
    original_open = store_module.os.open

    def fail_publication_directory_open(path, flags, *args, **kwargs):
        if Path(path) == repository._artifacts_dir:
            raise _HostileMessageOSError()
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(store_module.os, "open", fail_publication_directory_open)

    outcome = _resolve(runtime, spec, _Producer(spec))

    assert isinstance(outcome, FatalFailure)
    assert outcome.code == "artifact_publication_durability_failed"
    assert "<_HostileMessageOSError detail unavailable>" in outcome.detail
    assert repository.lookup(spec.artifact_key) is not None
    assert repository._publication_active(spec.artifact_key) is False
    assert list((tmp_path / "staging").iterdir()) == []
    runtime.close()


@pytest.mark.parametrize(
    ("failure_phase", "expected_code"),
    [
        ("open", "artifact_producer_open_failed"),
        ("build", "artifact_producer_build_failed"),
    ],
)
def test_generic_isolated_build_failure_is_fatal_and_aborts_staging(
    tmp_path: Path,
    failure_phase: str,
    expected_code: str,
) -> None:
    spec = _make_spec()
    repository = LocalArtifactRepository(tmp_path)
    runtime = HostWeightRuntime(repository)
    producer = _Producer(
        spec,
        fail_open=failure_phase == "open",
        fail_build=failure_phase == "build",
    )

    outcome = _resolve(runtime, spec, producer)

    assert isinstance(outcome, FatalFailure)
    assert outcome.code == expected_code
    if producer.last_session is not None:
        assert producer.last_session.closed
    assert repository.lookup(spec.artifact_key) is None
    assert list((tmp_path / "staging").iterdir()) == []
    runtime.close()


def test_semantic_rejection_happens_before_publication(tmp_path: Path) -> None:
    spec = _make_spec()
    repository = LocalArtifactRepository(tmp_path)
    runtime = HostWeightRuntime(repository)

    outcome = _resolve(runtime, spec, _Producer(spec, bad_manifest=True))

    assert isinstance(outcome, FatalFailure)
    assert outcome.code == "artifact_build_rejected"
    assert repository.lookup(spec.artifact_key) is None
    assert list((tmp_path / "staging").iterdir()) == []
    runtime.close()


def test_builder_crash_releases_lease_and_waiter_is_retryable(tmp_path: Path) -> None:
    spec = _make_spec()
    repository = LocalArtifactRepository(tmp_path)
    builder = repository.claim(spec.artifact_key, _build_intent(spec))
    assert isinstance(builder, Builder)
    runtime = HostWeightRuntime(repository, writable=False)

    result: list[object] = []
    thread = threading.Thread(target=lambda: result.append(_resolve(runtime, spec, wait_timeout_s=2)))
    thread.start()
    time.sleep(0.05)
    builder.lease.abort()
    thread.join()

    assert len(result) == 1
    assert isinstance(result[0], RetryableFailure)
    assert result[0].code == "builder_owner_lost"
    next_builder = repository.claim(spec.artifact_key, _build_intent(spec))
    assert isinstance(next_builder, Builder)
    next_builder.lease.abort()
    runtime.close()


def _build_one(repository: LocalArtifactRepository, spec: ArtifactSpec) -> Ready:
    runtime = HostWeightRuntime(repository)
    outcome = _resolve(runtime, spec, _Producer(spec))
    assert isinstance(outcome, Ready)
    outcome.artifact.close()
    runtime.close()
    return outcome


def test_warm_reader_rejects_corrupt_object(tmp_path: Path) -> None:
    spec = _make_spec()
    repository = LocalArtifactRepository(tmp_path)
    _build_one(repository, spec)
    existing = repository.lookup(spec.artifact_key)
    assert existing is not None
    weights = existing.artifact_dir / WEIGHTS_FILENAME
    with weights.open("r+b") as handle:
        first = handle.read(1)
        handle.seek(0)
        handle.write(bytes([first[0] ^ 0xFF]))

    runtime = HostWeightRuntime(repository, writable=False)
    outcome = _resolve(runtime, spec)
    assert isinstance(outcome, FatalFailure)
    assert outcome.code == "corrupt_store"
    assert "digest mismatch" in outcome.detail
    runtime.close()


@pytest.mark.parametrize(
    ("metadata_filename", "field", "malformed"),
    [
        (BACKING_INDEX_FILENAME, "tensor_spans", []),
        (MANIFEST_FILENAME, "tensors", {}),
    ],
)
def test_warm_reader_maps_malformed_nested_metadata_to_corrupt_store(
    tmp_path: Path,
    metadata_filename: str,
    field: str,
    malformed: object,
) -> None:
    spec = _make_spec()
    repository = LocalArtifactRepository(tmp_path)
    _build_one(repository, spec)
    metadata_path = repository._artifact_dir(spec.artifact_key) / metadata_filename
    value = json.loads(metadata_path.read_text())
    value[field] = malformed
    metadata_path.write_text(json.dumps(value))

    runtime = HostWeightRuntime(repository, writable=False)
    outcome = _resolve(runtime, spec)

    assert isinstance(outcome, FatalFailure)
    assert outcome.code == "corrupt_store"
    assert "must be a JSON" in outcome.detail
    runtime.close()


def test_repository_rejects_path_traversal_in_backing_index(tmp_path: Path) -> None:
    spec = _make_spec()
    repository = LocalArtifactRepository(tmp_path)
    _build_one(repository, spec)
    artifact_dir = repository._artifact_dir(spec.artifact_key)
    index_path = artifact_dir / BACKING_INDEX_FILENAME
    value = json.loads(index_path.read_text())
    value["objects"][0]["relative_path"] = "../outside.bin"
    index_path.write_text(json.dumps(value))

    with pytest.raises(StoreCorruptionError):
        repository.lookup(spec.artifact_key)


def test_repository_rejects_symlinked_storage_object(tmp_path: Path) -> None:
    spec = _make_spec()
    repository = LocalArtifactRepository(tmp_path)
    _build_one(repository, spec)
    artifact_dir = repository._artifact_dir(spec.artifact_key)
    weights = artifact_dir / WEIGHTS_FILENAME
    outside = tmp_path / "outside.bin"
    outside.write_bytes(weights.read_bytes())
    weights.unlink()
    weights.symlink_to(outside)

    with pytest.raises(StoreCorruptionError, match="symlink"):
        repository.lookup(spec.artifact_key)
