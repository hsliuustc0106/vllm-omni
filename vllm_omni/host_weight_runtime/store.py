# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Crash-safe single-node POSIX artifact repository."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import stat
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch

from ._exceptions import (
    safe_add_exception_note,
    safe_close_fd,
    safe_exception_detail,
    safe_exception_summary,
)
from .backings import (
    BackingError,
    RuntimeMmapBacking,
    WeightBacking,
    normalized_storage_view,
    validate_runtime_record_files,
)
from .contracts import (
    AccessFeature,
    ArtifactManifest,
    ArtifactRecord,
    ArtifactSink,
    BackingCapabilities,
    BackingIndex,
    BackingKind,
    BuildFailureClassification,
    BuildFailureKind,
    BuildFailureStage,
    BuildIntent,
    ContractError,
    StorageObject,
    StorageSpan,
    TensorRole,
    TensorSpec,
    torch_dtype_name,
)

MANIFEST_FILENAME = "manifest.json"
BACKING_INDEX_FILENAME = "backing-index.json"
WEIGHTS_FILENAME = "weights.bin"
_OBJECT_ID = "runtime-weights"
_ALIGNMENT = 64
_MAX_METADATA_BYTES = 64 * 1024 * 1024


class StoreError(RuntimeError):
    """Base class for repository coordination and publication failures."""


class StoreCorruptionError(StoreError):
    """A path marked as published is malformed or physically inconsistent."""


class StorePublicationDurabilityError(StoreError):
    """An artifact is visible, but publication durability is uncertain."""


class ReservationError(StoreError):
    """A build lease or staging sink was used outside its authority."""


@dataclass(frozen=True, slots=True)
class Existing:
    record: ArtifactRecord
    artifact_dir: Path


@dataclass(frozen=True, slots=True)
class Builder:
    lease: BuildLease


@dataclass(frozen=True, slots=True)
class NoBuilder:
    artifact_key: str


@dataclass(frozen=True, slots=True)
class WaitTimeout:
    artifact_key: str
    builder_active: bool


@dataclass(frozen=True, slots=True)
class BuilderFailed:
    artifact_key: str
    observed_lease_id: str
    failure: BuildFailureClassification


class Waiter:
    """A non-owning observation of another process's active build."""

    def __init__(
        self,
        repository: LocalArtifactRepository,
        artifact_key: str,
        observed_lease_id: str | None = None,
    ) -> None:
        self._repository = repository
        self.artifact_key = artifact_key
        self.observed_lease_id = observed_lease_id


class ArtifactRepository(Protocol):
    def lookup(self, key: str) -> Existing | None: ...

    def claim(
        self,
        key: str,
        build_intent: BuildIntent | None,
    ) -> Existing | Builder | Waiter | NoBuilder: ...

    def wait(
        self,
        waiter: Waiter,
        timeout_s: float,
    ) -> Existing | BuilderFailed | WaitTimeout: ...

    def create_sink(self, lease: BuildLease) -> ArtifactSink: ...

    def commit(
        self,
        lease: BuildLease,
        sink: ArtifactSink,
        manifest: ArtifactManifest,
    ) -> ArtifactRecord: ...


class RuntimeMmapArtifactSink:
    """A staging-only sink that never retains producer tensors."""

    def __init__(
        self,
        staging_dir: Path,
        artifact_key: str,
        publication_lease_id: str,
    ) -> None:
        self._staging_dir = staging_dir
        self._artifact_key = artifact_key
        self._publication_lease_id = publication_lease_id
        self._weights_path = staging_dir / WEIGHTS_FILENAME
        self._handle = self._weights_path.open("xb")
        self._object_digest = hashlib.sha256()
        self._receipts: dict[str, TensorSpec] = {}
        self._spans: dict[str, StorageSpan] = {}
        self._finished = False
        self._aborted = False
        self._lock = threading.RLock()
        self._record: ArtifactRecord | None = None

    @property
    def receipts(self) -> tuple[TensorSpec, ...]:
        with self._lock:
            return tuple(self._receipts[tensor_id] for tensor_id in sorted(self._receipts))

    def write_tensor(
        self,
        tensor_id: str,
        tensor: torch.Tensor,
        *,
        role: TensorRole = TensorRole.PARAMETER,
    ) -> TensorSpec:
        """Synchronously copy one final tensor and return its logical receipt."""
        with self._lock:
            if self._finished or self._aborted:
                raise ReservationError("cannot write to a closed artifact sink")
            if tensor_id in self._receipts:
                raise ReservationError(f"duplicate runtime tensor id: {tensor_id!r}")
            raw = normalized_storage_view(tensor_id, tensor)
            current_offset = self._handle.tell()
            padding = (-current_offset) % _ALIGNMENT
            if padding:
                padding_bytes = b"\x00" * padding
                self._handle.write(padding_bytes)
                self._object_digest.update(padding_bytes)
            storage_offset = current_offset + padding
            if raw.numel():
                raw_view = memoryview(raw.numpy())
                self._handle.write(raw_view)
                self._object_digest.update(raw_view)
                content_digest = hashlib.sha256(raw_view).hexdigest()
            else:
                content_digest = hashlib.sha256(b"").hexdigest()
            receipt = TensorSpec(
                tensor_id=tensor_id,
                role=TensorRole(role),
                dtype=torch_dtype_name(tensor.dtype),
                shape=tuple(tensor.shape),
                stride=tuple(tensor.stride()),
                storage_numel=raw.numel() // tensor.element_size(),
                content_digest=content_digest,
            )
            self._receipts[tensor_id] = receipt
            self._spans[tensor_id] = StorageSpan(
                object_id=_OBJECT_ID,
                offset_bytes=storage_offset,
                length_bytes=receipt.storage_nbytes,
            )
            return receipt

    def finish(self, manifest: ArtifactManifest) -> ArtifactRecord:
        with self._lock:
            if self._aborted:
                raise ReservationError("artifact sink was aborted")
            if self._finished:
                if self._record is None or self._record.manifest != manifest:
                    raise ReservationError("artifact sink was finished with a different manifest")
                return self._record
            if not self._receipts:
                raise ReservationError("a runtime-mmap artifact must not be empty")
            if manifest.artifact_key != self._artifact_key:
                raise ReservationError("manifest does not match the staging artifact key")
            receipt_map = self._receipts
            if set(receipt_map) != set(manifest.tensor_ids):
                raise ReservationError("manifest tensors do not exactly match sink writes")
            for tensor_spec in manifest.tensors:
                if receipt_map[tensor_spec.tensor_id] != tensor_spec:
                    raise ReservationError(f"manifest tensor does not match written bytes: {tensor_spec.tensor_id!r}")

            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
            nbytes = self._weights_path.stat().st_size
            index = BackingIndex(
                artifact_key=manifest.artifact_key,
                generation_id=self._publication_lease_id,
                kind=BackingKind.RUNTIME_MMAP,
                provider_name="local-runtime-mmap",
                provider_version="1",
                objects=(
                    StorageObject(
                        object_id=_OBJECT_ID,
                        nbytes=nbytes,
                        alignment=_ALIGNMENT,
                        content_digest=self._object_digest.hexdigest(),
                        relative_path=WEIGHTS_FILENAME,
                    ),
                ),
                tensor_spans=self._spans,
            )
            record = ArtifactRecord(manifest, index)
            self._write_metadata(MANIFEST_FILENAME, manifest.to_json())
            self._write_metadata(BACKING_INDEX_FILENAME, index.to_json())
            directory_descriptor = os.open(self._staging_dir, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                safe_close_fd(
                    directory_descriptor,
                    "artifact staging-directory descriptor close also failed",
                )
            self._record = record
            self._finished = True
            return record

    def _write_metadata(self, filename: str, payload: str) -> None:
        encoded = payload.encode("utf-8")
        if len(encoded) > _MAX_METADATA_BYTES:
            raise ReservationError(f"{filename} exceeds the metadata size limit")
        path = self._staging_dir / filename
        with path.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())

    def abort(self) -> None:
        with self._lock:
            if self._finished or self._aborted:
                return
            self._handle.close()
            self._aborted = True


class BuildLease:
    """Exclusive process-safe authority to build one key for one intent."""

    def __init__(
        self,
        repository: LocalArtifactRepository,
        artifact_key: str,
        intent: BuildIntent,
        lock_fd: int,
        lease_id: str,
    ) -> None:
        self._repository = repository
        self.artifact_key = artifact_key
        self.intent = intent
        self.lease_id = lease_id
        self._lock_fd = lock_fd
        self._active = True
        self._release_published: bool | None = None
        self._sink_cleanup_complete = False
        self._staging_cleanup_complete = False
        self._lock_released = False
        self._fd_closed = False
        self._cleanup_complete = False
        self._staging_dir: Path | None = None
        self._sink: RuntimeMmapArtifactSink | None = None
        self._lock = threading.RLock()
        self._recorded_failure: BuildFailureClassification | None = None

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    def _create_sink(self) -> RuntimeMmapArtifactSink:
        with self._lock:
            if not self._active:
                raise ReservationError("build lease is inactive")
            if self._sink is not None:
                raise ReservationError("build lease already owns a sink")
            staging_path = tempfile.mkdtemp(
                prefix=f"{self.artifact_key}.",
                dir=self._repository._staging_dir,
            )
            self._staging_dir = Path(staging_path)
            self._sink = RuntimeMmapArtifactSink(
                self._staging_dir,
                self.artifact_key,
                self.lease_id,
            )
            return self._sink

    def record_failure(self, failure: BuildFailureClassification) -> None:
        """Retain the core-owned classification for exact waiter transport."""

        with self._lock:
            if self._recorded_failure is None:
                self._recorded_failure = failure
            # Classification is deliberately write-once and non-failing: this
            # method is called while propagating the primary build exception,
            # so a duplicate cleanup/error path must never replace it.  The
            # first core-owned envelope remains the exact waiter result.

    def _mark_published(self) -> None:
        """Record the rename linearization point without releasing the lease."""

        with self._lock:
            if self._release_published is False:
                raise ReservationError("cannot publish an aborting build lease")
            self._release_published = True

    def _release(self, *, published: bool) -> None:
        if self._cleanup_complete:
            return

        if self._release_published is None:
            self._release_published = published
        # Authority ends when release begins, independently of how many
        # best-effort resource cleanup steps still need retrying.
        self._active = False
        published = self._release_published
        primary: BaseException | None = None

        def record_failure(context: str, exc: BaseException) -> None:
            nonlocal primary
            if primary is None:
                primary = exc
            else:
                safe_add_exception_note(primary, context, exc)

        if published:
            self._sink_cleanup_complete = True
            self._staging_cleanup_complete = True
        else:
            if not self._sink_cleanup_complete:
                try:
                    if self._sink is not None:
                        self._sink.abort()
                except BaseException as exc:
                    record_failure("artifact sink cleanup also failed", exc)
                else:
                    self._sink_cleanup_complete = True
            if not self._staging_cleanup_complete:
                try:
                    if self._staging_dir is not None and self._staging_dir.exists():
                        # The target is an explicit mkdtemp child of this
                        # repository's staging directory, never a
                        # caller-provided recursive path.
                        shutil.rmtree(self._staging_dir)
                except BaseException as exc:
                    record_failure("artifact staging-directory cleanup also failed", exc)
                else:
                    self._staging_cleanup_complete = True

        if not self._lock_released:
            if not published:
                try:
                    self._repository._write_lease_metadata(
                        self._lock_fd,
                        artifact_key=self.artifact_key,
                        lease_id=self.lease_id,
                        owner_lost_failure=self.intent.owner_lost_failure,
                        failure=self._recorded_failure,
                    )
                except BaseException as exc:
                    record_failure("artifact build failure publication also failed", exc)
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            except BaseException as exc:
                record_failure("artifact build-lock release also failed", exc)
            else:
                self._lock_released = True
        if not self._fd_closed:
            try:
                os.close(self._lock_fd)
            except BaseException as exc:
                record_failure("artifact build-lock descriptor close also failed", exc)
            else:
                self._fd_closed = True
                # POSIX close releases any remaining process lock even when an
                # explicit unlock attempt failed.
                self._lock_released = True

        self._cleanup_complete = (
            self._sink_cleanup_complete and self._staging_cleanup_complete and self._lock_released and self._fd_closed
        )
        if primary is not None:
            raise primary

    def abort(self) -> None:
        with self._lock:
            self._release(published=False)

    def __enter__(self) -> BuildLease:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        exc: BaseException | None,
        _traceback: object,
    ) -> None:
        with self._lock:
            try:
                self._release(published=False)
            except BaseException as cleanup_exc:
                if exc is None:
                    raise
                safe_add_exception_note(
                    exc,
                    "artifact lease cleanup while propagating the build failure also failed",
                    cleanup_exc,
                )

    def __del__(self) -> None:
        try:
            self.abort()
        except Exception:
            pass


ClaimDecision = Existing | Builder | Waiter | NoBuilder


class LocalArtifactRepository:
    """One-node immutable repository coordinated with POSIX ``flock``."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self._artifacts_dir = self.root / "artifacts"
        self._locks_dir = self.root / "locks"
        self._staging_dir = self.root / "staging"

    def _ensure_directories(self) -> None:
        """Materialize local storage only when an owned operation starts."""

        for directory in (
            self.root,
            self._artifacts_dir,
            self._locks_dir,
            self._staging_dir,
        ):
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)

    @staticmethod
    def _validate_key(artifact_key: str) -> None:
        if len(artifact_key) != 64 or any(character not in "0123456789abcdef" for character in artifact_key):
            raise ValueError("artifact key must be a lowercase SHA-256 digest")

    def _artifact_dir(self, artifact_key: str) -> Path:
        self._validate_key(artifact_key)
        return self._artifacts_dir / artifact_key

    def artifact_directory(self, artifact_key: str) -> Path:
        """Return the provider-facing directory for one validated key."""

        return self._artifact_dir(artifact_key)

    def _lock_path(self, artifact_key: str) -> Path:
        self._validate_key(artifact_key)
        return self._locks_dir / f"{artifact_key}.lock"

    @staticmethod
    def _failure_to_dict(
        failure: BuildFailureClassification,
    ) -> dict[str, object]:
        return {
            "stage": failure.stage.value,
            "code": failure.code,
            "detail": failure.detail,
            "kind": failure.kind.value,
            "retry_after_s": failure.retry_after_s,
        }

    @staticmethod
    def _failure_from_dict(value: object) -> BuildFailureClassification:
        if not isinstance(value, dict):
            raise StoreError("build lease failure metadata is malformed")
        try:
            return BuildFailureClassification(
                stage=BuildFailureStage(value["stage"]),
                code=str(value["code"]),
                detail=str(value["detail"]),
                kind=BuildFailureKind(value["kind"]),
                retry_after_s=(None if value.get("retry_after_s") is None else float(value["retry_after_s"])),
            )
        except (KeyError, TypeError, ValueError, ContractError) as exc:
            raise StoreError("build lease failure metadata is malformed") from exc

    @classmethod
    def _write_lease_metadata(
        cls,
        lock_fd: int,
        *,
        artifact_key: str,
        lease_id: str,
        owner_lost_failure: BuildFailureClassification,
        failure: BuildFailureClassification | None = None,
    ) -> None:
        payload = json.dumps(
            {
                "artifact_key": artifact_key,
                "lease_id": lease_id,
                "owner_lost_failure": cls._failure_to_dict(owner_lost_failure),
                "failure": None if failure is None else cls._failure_to_dict(failure),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        os.lseek(lock_fd, 0, os.SEEK_SET)
        os.ftruncate(lock_fd, 0)
        written = 0
        while written < len(payload):
            count = os.write(lock_fd, payload[written:])
            if count <= 0:
                raise OSError("short write while publishing build lease metadata")
            written += count
        os.fsync(lock_fd)

    @classmethod
    def _read_lease_metadata(
        cls,
        lock_fd: int,
    ) -> tuple[str, str, BuildFailureClassification, BuildFailureClassification | None]:
        os.lseek(lock_fd, 0, os.SEEK_SET)
        payload = os.read(lock_fd, 64 * 1024 + 1)
        if not payload or len(payload) > 64 * 1024:
            raise StoreError("active build lease metadata is unavailable")
        try:
            value = json.loads(payload)
            artifact_key = str(value["artifact_key"])
            lease_id = str(value["lease_id"])
            owner_lost = cls._failure_from_dict(value["owner_lost_failure"])
            failure_value = value.get("failure")
            failure = None if failure_value is None else cls._failure_from_dict(failure_value)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise StoreError("active build lease metadata is malformed") from exc
        cls._validate_key(artifact_key)
        if not lease_id:
            raise StoreError("active build lease ID is empty")
        return artifact_key, lease_id, owner_lost, failure

    def _read_lease_metadata_for_key(
        self,
        lock_fd: int,
        artifact_key: str,
    ) -> tuple[str, BuildFailureClassification, BuildFailureClassification | None]:
        observed_key, lease_id, owner_lost, failure = self._read_lease_metadata(lock_fd)
        if observed_key != artifact_key:
            raise StoreError("active build lease metadata names another artifact")
        return lease_id, owner_lost, failure

    @staticmethod
    def _read_metadata(path: Path) -> str:
        try:
            path_stat = path.stat(follow_symlinks=False)
        except FileNotFoundError as exc:
            raise StoreCorruptionError(f"artifact metadata is missing: {path}") from exc
        if not stat.S_ISREG(path_stat.st_mode):
            raise StoreCorruptionError(f"artifact metadata is not a regular file: {path}")
        if path_stat.st_size > _MAX_METADATA_BYTES:
            raise StoreCorruptionError(f"artifact metadata exceeds size limit: {path}")
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            return handle.read(_MAX_METADATA_BYTES + 1)

    def lookup(self, artifact_key: str) -> Existing | None:
        artifact_dir = self._artifact_dir(artifact_key)
        try:
            directory_stat = artifact_dir.stat(follow_symlinks=False)
        except FileNotFoundError:
            return None
        if not stat.S_ISDIR(directory_stat.st_mode):
            raise StoreCorruptionError(f"published artifact path is not a directory: {artifact_dir}")
        try:
            manifest = ArtifactManifest.from_json(self._read_metadata(artifact_dir / MANIFEST_FILENAME))
            index = BackingIndex.from_json(self._read_metadata(artifact_dir / BACKING_INDEX_FILENAME))
            record = ArtifactRecord(manifest, index)
            if manifest.artifact_key != artifact_key:
                raise ContractError("published manifest key does not match directory key")
            if index.kind is not BackingKind.RUNTIME_MMAP:
                raise ContractError("local repository contains a non-runtime-mmap index")
            self._validate_record_files(artifact_dir, record)
        except StoreCorruptionError:
            raise
        except (BackingError, OSError, ValueError, KeyError, TypeError, ContractError) as exc:
            raise StoreCorruptionError(
                f"failed to validate published artifact {artifact_dir}: {safe_exception_detail(exc)}"
            ) from exc
        return Existing(record, artifact_dir)

    @staticmethod
    def _validate_record_files(artifact_dir: Path, record: ArtifactRecord) -> None:
        validate_runtime_record_files(artifact_dir, record)

    def claim(
        self,
        artifact_key: str,
        build_intent: BuildIntent | None,
    ) -> ClaimDecision:
        self._ensure_directories()
        if hit := self.lookup(artifact_key):
            return hit
        lock_fd = os.open(self._lock_path(artifact_key), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            try:
                # Ordered callers reach this path only after BuilderStarted,
                # so the elected builder has already persisted this exact ID.
                metadata_error: StoreError | None = None
                for _ in range(1000):
                    try:
                        _, lease_id, _, _ = self._read_lease_metadata(lock_fd)
                        break
                    except StoreError as exc:
                        metadata_error = exc
                        time.sleep(0.001)
                else:
                    assert metadata_error is not None
                    raise metadata_error
            finally:
                os.close(lock_fd)
            return Waiter(self, artifact_key, lease_id)
        try:
            if hit := self.lookup(artifact_key):
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)
                return hit
            if build_intent is None:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)
                return NoBuilder(artifact_key)
            lease_id = f"lease-{uuid.uuid4().hex}"
            self._write_lease_metadata(
                lock_fd,
                artifact_key=artifact_key,
                lease_id=lease_id,
                owner_lost_failure=build_intent.owner_lost_failure,
            )
            return Builder(
                BuildLease(
                    self,
                    artifact_key,
                    build_intent,
                    lock_fd,
                    lease_id,
                )
            )
        except BaseException as exc:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            except BaseException as cleanup_exc:
                safe_add_exception_note(
                    exc,
                    "artifact claim lock release also failed",
                    cleanup_exc,
                )
            try:
                os.close(lock_fd)
            except BaseException as cleanup_exc:
                safe_add_exception_note(
                    exc,
                    "artifact claim lock descriptor close also failed",
                    cleanup_exc,
                )
            raise

    def _publication_active(self, artifact_key: str) -> bool:
        self._ensure_directories()
        lock_fd = os.open(self._lock_path(artifact_key), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            return False
        finally:
            safe_close_fd(
                lock_fd,
                "artifact waiter-probe descriptor close also failed",
            )

    def wait(
        self,
        waiter: Waiter,
        timeout_s: float,
        *,
        poll_interval_s: float = 0.01,
    ) -> Existing | BuilderFailed | WaitTimeout:
        if waiter._repository is not self:
            raise ReservationError("waiter belongs to a different repository")
        if timeout_s < 0:
            raise ValueError("timeout_s must be non-negative")
        self._ensure_directories()
        deadline = time.monotonic() + timeout_s
        while True:
            if hit := self.lookup(waiter.artifact_key):
                return hit
            active = self._publication_active(waiter.artifact_key)
            remaining = deadline - time.monotonic()
            if not active:
                # Publication may have linearized after the first lookup but
                # before the builder released its lock.  Recheck after
                # observing inactivity before declaring that no generation is
                # available.
                if hit := self.lookup(waiter.artifact_key):
                    return hit
                if waiter.observed_lease_id is not None:
                    lock_fd = os.open(
                        self._lock_path(waiter.artifact_key),
                        os.O_RDONLY,
                    )
                    try:
                        lease_id, owner_lost, failure = self._read_lease_metadata_for_key(
                            lock_fd,
                            waiter.artifact_key,
                        )
                    finally:
                        os.close(lock_fd)
                    if lease_id == waiter.observed_lease_id:
                        return BuilderFailed(
                            waiter.artifact_key,
                            lease_id,
                            failure or owner_lost,
                        )
                return WaitTimeout(waiter.artifact_key, builder_active=False)
            if remaining <= 0:
                return WaitTimeout(waiter.artifact_key, builder_active=active)
            time.sleep(min(poll_interval_s, remaining))

    def create_sink(self, lease: BuildLease) -> RuntimeMmapArtifactSink:
        if lease._repository is not self:
            raise ReservationError("build lease belongs to a different repository")
        return lease._create_sink()

    def commit(
        self,
        lease: BuildLease,
        sink: ArtifactSink,
        manifest: ArtifactManifest,
    ) -> ArtifactRecord:
        if lease._repository is not self or not lease.active:
            raise ReservationError("build lease is not active in this repository")
        if sink is not lease._sink or not isinstance(sink, RuntimeMmapArtifactSink):
            raise ReservationError("artifact sink does not belong to this lease")
        if manifest.producer != lease.intent.producer:
            raise ReservationError("manifest producer does not match elected build intent")
        if lease._staging_dir is None:
            raise ReservationError("build lease has no staging directory")
        record = sink.finish(manifest)
        self._publish(lease, lease._staging_dir, record)
        return record

    def _publish(
        self,
        lease: BuildLease,
        staging_dir: Path,
        record: ArtifactRecord,
    ) -> None:
        if lease.artifact_key != record.manifest.artifact_key:
            raise ReservationError("build lease does not own this artifact")
        target = self._artifact_dir(record.manifest.artifact_key)
        if target.exists():
            raise ReservationError("immutable artifact target already exists")
        published = False
        try:
            os.rename(staging_dir, target)
            # Rename is the publication linearization point.  Every failure
            # after this assignment must preserve the immutable target and
            # release the lease as published.
            published = True
            lease._mark_published()
            directory_descriptor = os.open(self._artifacts_dir, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                safe_close_fd(
                    directory_descriptor,
                    "artifact publication-directory descriptor close also failed",
                )
        except BaseException as exc:
            if published and isinstance(exc, Exception):
                raise StorePublicationDurabilityError(
                    "artifact publication is visible, but directory durability "
                    f"could not be confirmed: {safe_exception_summary(exc)}"
                ) from exc
            raise


class LocalRuntimeMmapBackingProvider:
    """Open LocalArtifactRepository records through RuntimeMmapBacking."""

    def __init__(
        self,
        repository: LocalArtifactRepository,
        *,
        verify_integrity: bool = True,
    ) -> None:
        self._repository = repository
        self._verify_integrity = bool(verify_integrity)

    def capabilities(self) -> BackingCapabilities:
        return BackingCapabilities(
            BackingKind.RUNTIME_MMAP,
            "local-runtime-mmap",
            "1",
            frozenset(
                {
                    AccessFeature.COMPLETE_TENSOR_READ,
                    AccessFeature.SHARED_PAGES,
                }
            ),
        )

    def open(self, manifest: ArtifactManifest, index: BackingIndex) -> WeightBacking:
        record = ArtifactRecord(manifest, index)
        return RuntimeMmapBacking(
            self._repository.artifact_directory(index.artifact_key),
            record,
            verify_integrity=self._verify_integrity,
        )


__all__ = [
    "BACKING_INDEX_FILENAME",
    "MANIFEST_FILENAME",
    "WEIGHTS_FILENAME",
    "ArtifactRepository",
    "BuildLease",
    "Builder",
    "BuilderFailed",
    "ClaimDecision",
    "Existing",
    "LocalArtifactRepository",
    "LocalRuntimeMmapBackingProvider",
    "NoBuilder",
    "ReservationError",
    "RuntimeMmapArtifactSink",
    "StoreCorruptionError",
    "StoreError",
    "StorePublicationDurabilityError",
    "WaitTimeout",
    "Waiter",
]
