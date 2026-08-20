# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Immutable process-local and POSIX mmap host-weight backings."""

from __future__ import annotations

import hashlib
import mmap
import os
import stat
import threading
import uuid
import warnings
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

import torch

from ._exceptions import safe_add_exception_note, safe_close_fd
from .contracts import (
    AccessFeature,
    ArtifactManifest,
    ArtifactRecord,
    ArtifactSpec,
    BackingCapabilities,
    BackingIndex,
    BackingKind,
    JSONValue,
    StorageObject,
    StorageSpan,
    TensorRole,
    TensorSpec,
    required_storage_numel,
    torch_dtype_name,
)
from .validation import validate_manifest_against_spec


class BackingError(RuntimeError):
    """Base class for backing validation and access failures."""


class BackingIntegrityError(BackingError):
    """Published or loaded bytes do not match their immutable record."""


class BackingClosedError(BackingError):
    """A backing was accessed after its lifetime ended."""


class WeightBacking(Protocol):
    manifest: ArtifactManifest
    backing_index: BackingIndex

    def copy_into(self, tensor_id: str, destination: torch.Tensor) -> None: ...

    def close(self) -> None: ...


class BackingProvider(Protocol):
    def capabilities(self) -> BackingCapabilities: ...

    def open(self, manifest: ArtifactManifest, index: BackingIndex) -> WeightBacking: ...


class EphemeralBackingProvider(BackingProvider, Protocol):
    """Provider that can materialize a process-local backing from live tensors."""

    def open_ephemeral(
        self,
        spec: ArtifactSpec,
        tensors: Mapping[str, torch.Tensor],
        *,
        roles: Mapping[str, TensorRole] | None = None,
        format_metadata: Mapping[str, JSONValue] | None = None,
    ) -> WeightBacking: ...


class BackingProviderRegistry(Protocol):
    """Lookup boundary injected into the backing-independent runtime."""

    def capabilities(self) -> tuple[BackingCapabilities, ...]: ...

    def provider_for(self, kind: BackingKind) -> BackingProvider | None: ...


class StaticBackingProviderRegistry:
    """Immutable registry for an explicitly composed set of providers."""

    def __init__(self, providers: Iterable[BackingProvider]) -> None:
        by_kind: dict[BackingKind, BackingProvider] = {}
        capabilities: dict[BackingKind, BackingCapabilities] = {}
        for provider in providers:
            capability = provider.capabilities()
            if not isinstance(capability, BackingCapabilities):
                raise TypeError("backing provider capabilities() must return BackingCapabilities")
            if capability.kind in by_kind:
                raise BackingError(f"duplicate backing provider for {capability.kind.value!r}")
            by_kind[capability.kind] = provider
            capabilities[capability.kind] = capability
        self._providers: Mapping[BackingKind, BackingProvider] = MappingProxyType(by_kind)
        self._capabilities = tuple(capabilities[kind] for kind in sorted(capabilities, key=lambda item: item.value))

    def capabilities(self) -> tuple[BackingCapabilities, ...]:
        return self._capabilities

    def provider_for(self, kind: BackingKind) -> BackingProvider | None:
        return self._providers.get(BackingKind(kind))


def _torch_dtype(dtype_name: str) -> torch.dtype:
    value = getattr(torch, dtype_name, None)
    if not isinstance(value, torch.dtype):
        raise BackingIntegrityError(f"torch does not provide manifest dtype {dtype_name!r}")
    return value


def _validate_source_tensor(tensor_id: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise BackingError(f"host object {tensor_id!r} is not a torch.Tensor")
    if tensor.device.type != "cpu":
        raise BackingError(f"host tensor {tensor_id!r} must be on CPU, got {tensor.device}")
    if tensor.layout is not torch.strided:
        raise BackingError(f"host tensor {tensor_id!r} must use strided layout")
    if any(value < 0 for value in tensor.stride()):
        raise BackingError(f"host tensor {tensor_id!r} has an unsupported negative stride")


def normalized_storage_view(tensor_id: str, tensor: torch.Tensor) -> torch.Tensor:
    """Return the exact normalized uint8 span written to an artifact sink.

    Contiguous tensors use their existing storage without a second full-size
    allocation.  Non-contiguous tensors need a normalized temporary so holes
    contain deterministic zero bytes rather than unrelated source storage.
    """
    _validate_source_tensor(tensor_id, tensor)
    tensor = tensor.detach()
    storage_numel = required_storage_numel(tuple(tensor.shape), tuple(tensor.stride()))
    span_nbytes = storage_numel * tensor.element_size()
    if span_nbytes == 0:
        return torch.empty(0, dtype=torch.uint8)
    if tensor.is_contiguous() and storage_numel == tensor.numel():
        return torch.empty(0, dtype=torch.uint8).set_(
            tensor.untyped_storage(),
            tensor.storage_offset() * tensor.element_size(),
            (span_nbytes,),
            (1,),
        )
    raw = torch.zeros(span_nbytes, dtype=torch.uint8)
    typed = raw.view(tensor.dtype)
    normalized = torch.as_strided(typed, size=tensor.shape, stride=tensor.stride())
    with torch.no_grad():
        normalized.copy_(tensor, non_blocking=False)
    return raw


def _digest_raw(raw: torch.Tensor) -> str:
    digest = hashlib.sha256()
    if raw.numel():
        digest.update(memoryview(raw.numpy()))
    return digest.hexdigest()


def tensor_spec_from_tensor(
    tensor_id: str,
    tensor: torch.Tensor,
    *,
    role: TensorRole = TensorRole.PARAMETER,
) -> TensorSpec:
    raw = normalized_storage_view(tensor_id, tensor)
    return TensorSpec(
        tensor_id=tensor_id,
        role=TensorRole(role),
        dtype=torch_dtype_name(tensor.dtype),
        shape=tuple(tensor.shape),
        stride=tuple(tensor.stride()),
        storage_numel=raw.numel() // tensor.element_size(),
        content_digest=_digest_raw(raw),
    )


def _validate_destination(spec: TensorSpec, destination: torch.Tensor) -> None:
    if not isinstance(destination, torch.Tensor):
        raise BackingError(f"destination for {spec.tensor_id!r} is not a torch.Tensor")
    if destination.device.type != "cpu":
        raise BackingError(f"destination for {spec.tensor_id!r} must be on CPU")
    if tuple(destination.shape) != spec.shape:
        raise BackingError(
            f"destination shape mismatch for {spec.tensor_id!r}: expected {spec.shape}, got {tuple(destination.shape)}"
        )
    if torch_dtype_name(destination.dtype) != spec.dtype:
        raise BackingError(
            f"destination dtype mismatch for {spec.tensor_id!r}: "
            f"expected {spec.dtype}, got {torch_dtype_name(destination.dtype)}"
        )


def _tensor_version(tensor: torch.Tensor) -> int | None:
    try:
        return tensor._version
    except RuntimeError:
        # Inference tensors do not expose a version counter.  Their producer
        # still owns the immutable-transfer promise.
        return None


class LoadedTensorBacking:
    """Read-only facade over already-finalized process-local CPU tensors."""

    def __init__(self, record: ArtifactRecord, tensors: Mapping[str, torch.Tensor]) -> None:
        if record.backing_index.kind is not BackingKind.LOADED_TENSOR:
            raise BackingError("loaded-tensor backing requires a loaded-tensor index")
        if set(tensors) != set(record.manifest.tensor_ids):
            raise BackingError("loaded tensors must exactly match manifest tensor ids")

        owned = {
            tensor_id: tensor.detach() if isinstance(tensor, torch.Tensor) else tensor
            for tensor_id, tensor in tensors.items()
        }
        storage_pointers: dict[int, str] = {}
        versions: dict[str, int | None] = {}
        for spec in record.manifest.tensors:
            tensor = owned[spec.tensor_id]
            _validate_source_tensor(spec.tensor_id, tensor)
            actual = tensor_spec_from_tensor(spec.tensor_id, tensor, role=spec.role)
            if actual != spec:
                raise BackingIntegrityError(f"loaded tensor metadata or content mismatch for {spec.tensor_id!r}")
            if tensor.numel():
                pointer = tensor.untyped_storage().data_ptr()
                if previous := storage_pointers.get(pointer):
                    raise BackingError(
                        f"loaded tensors {previous!r} and {spec.tensor_id!r} share storage; aliases are unsupported"
                    )
                storage_pointers[pointer] = spec.tensor_id
            versions[spec.tensor_id] = _tensor_version(tensor)

        self.manifest = record.manifest
        self.backing_index = record.backing_index
        self._tensors: Mapping[str, torch.Tensor] = MappingProxyType(owned)
        self._versions = versions
        self._closed = False
        self._lock = threading.RLock()

    @classmethod
    def from_tensors(
        cls,
        spec: ArtifactSpec,
        tensors: Mapping[str, torch.Tensor],
        *,
        roles: Mapping[str, TensorRole] | None = None,
        format_metadata: Mapping[str, JSONValue] | None = None,
    ) -> LoadedTensorBacking:
        if not tensors:
            raise BackingError("a loaded-tensor artifact must not be empty")
        roles = roles or {}
        if unknown := set(roles) - set(tensors):
            raise BackingError(f"roles reference unknown tensors: {sorted(unknown)}")

        tensor_specs: list[TensorSpec] = []
        objects: list[StorageObject] = []
        spans: dict[str, StorageSpan] = {}
        for index, tensor_id in enumerate(sorted(tensors)):
            tensor = tensors[tensor_id]
            tensor_spec = tensor_spec_from_tensor(
                tensor_id,
                tensor,
                role=roles.get(tensor_id, TensorRole.PARAMETER),
            )
            object_id = f"loaded-{index}"
            tensor_specs.append(tensor_spec)
            objects.append(
                StorageObject(
                    object_id=object_id,
                    nbytes=tensor_spec.storage_nbytes,
                    alignment=max(1, tensor.element_size()),
                    content_digest=tensor_spec.content_digest,
                    relative_path=None,
                )
            )
            spans[tensor_id] = StorageSpan(object_id, 0, tensor_spec.storage_nbytes)
        manifest = ArtifactManifest.create(
            spec,
            tuple(tensor_specs),
            format_metadata=format_metadata,
        )
        validate_manifest_against_spec(spec, manifest)
        index = BackingIndex(
            artifact_key=spec.artifact_key,
            generation_id=f"loaded-{uuid.uuid4().hex}",
            kind=BackingKind.LOADED_TENSOR,
            provider_name="loaded-tensor",
            provider_version="1",
            objects=tuple(objects),
            tensor_spans=spans,
        )
        return cls(ArtifactRecord(manifest, index), tensors)

    def copy_into(self, tensor_id: str, destination: torch.Tensor) -> None:
        with self._lock:
            if self._closed:
                raise BackingClosedError("loaded-tensor backing is closed")
            try:
                source = self._tensors[tensor_id]
                spec = self.manifest.tensor(tensor_id)
            except KeyError as exc:
                raise BackingError(f"unknown tensor id: {tensor_id!r}") from exc
            expected_version = self._versions[tensor_id]
            if expected_version is not None and _tensor_version(source) != expected_version:
                raise BackingIntegrityError(f"loaded tensor {tensor_id!r} was mutated after publication")
            _validate_destination(spec, destination)
            with torch.no_grad():
                destination.copy_(source, non_blocking=False)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._tensors = MappingProxyType({})
            self._versions.clear()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def safe_storage_path(artifact_dir: Path, relative_path: str) -> Path:
    """Resolve a record-owned relative path without following symlink escapes."""
    root = artifact_dir.resolve(strict=True)
    current = artifact_dir
    parts = relative_path.split("/")
    for index, part in enumerate(parts):
        current = current / part
        try:
            item_stat = current.stat(follow_symlinks=False)
        except FileNotFoundError as exc:
            raise BackingIntegrityError(f"storage path does not exist: {current}") from exc
        if stat.S_ISLNK(item_stat.st_mode):
            raise BackingIntegrityError(f"storage path must not contain a symlink: {current}")
        if index < len(parts) - 1 and not stat.S_ISDIR(item_stat.st_mode):
            raise BackingIntegrityError(f"storage path parent is not a directory: {current}")
    try:
        current.resolve(strict=True).relative_to(root)
    except ValueError as exc:
        raise BackingIntegrityError(f"storage path escapes artifact directory: {current}") from exc
    return current


def validate_runtime_record_files(
    artifact_dir: Path,
    record: ArtifactRecord,
) -> dict[str, Path]:
    """Validate sizes plus object/tensor SHA-256 values in one file scan."""
    paths: dict[str, Path] = {}
    tensor_by_id = {item.tensor_id: item for item in record.manifest.tensors}
    for storage_object in record.backing_index.objects:
        assert storage_object.relative_path is not None
        object_path = safe_storage_path(artifact_dir, storage_object.relative_path)
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(object_path, flags)
        try:
            object_stat = os.fstat(descriptor)
            if not stat.S_ISREG(object_stat.st_mode):
                raise BackingIntegrityError(f"storage object is not a regular file: {object_path}")
            if object_stat.st_size != storage_object.nbytes:
                raise BackingIntegrityError(
                    f"storage size mismatch for {storage_object.object_id!r}: "
                    f"expected {storage_object.nbytes}, got {object_stat.st_size}"
                )
            intervals = sorted(
                (
                    span.offset_bytes,
                    span.offset_bytes + span.length_bytes,
                    tensor_by_id[tensor_id],
                )
                for tensor_id, span in record.backing_index.tensor_spans.items()
                if span.object_id == storage_object.object_id
            )
            tensor_digests = {tensor_spec.tensor_id: hashlib.sha256() for _, _, tensor_spec in intervals}
            object_digest = hashlib.sha256()
            offset = 0
            interval_index = 0
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                while chunk := handle.read(4 * 1024 * 1024):
                    object_digest.update(chunk)
                    chunk_end = offset + len(chunk)
                    while interval_index < len(intervals) and intervals[interval_index][1] <= offset:
                        interval_index += 1
                    scan_index = interval_index
                    while scan_index < len(intervals) and intervals[scan_index][0] < chunk_end:
                        start, end, tensor_spec = intervals[scan_index]
                        overlap_start = max(start, offset)
                        overlap_end = min(end, chunk_end)
                        if overlap_start < overlap_end:
                            tensor_digests[tensor_spec.tensor_id].update(
                                chunk[overlap_start - offset : overlap_end - offset]
                            )
                        if end <= chunk_end:
                            scan_index += 1
                        else:
                            break
                    offset = chunk_end
            if object_digest.hexdigest() != storage_object.content_digest:
                raise BackingIntegrityError(f"storage digest mismatch for {storage_object.object_id!r}")
            for _, _, tensor_spec in intervals:
                if tensor_digests[tensor_spec.tensor_id].hexdigest() != tensor_spec.content_digest:
                    raise BackingIntegrityError(f"tensor digest mismatch for {tensor_spec.tensor_id!r}")
        finally:
            safe_close_fd(
                descriptor,
                "runtime-mmap integrity descriptor close also failed",
            )
        paths[storage_object.object_id] = object_path
    return paths


class RuntimeMmapBacking:
    """Read-only complete-tensor views over one verified runtime artifact."""

    def __init__(
        self,
        artifact_dir: str | os.PathLike[str],
        record: ArtifactRecord,
        *,
        verify_integrity: bool = True,
    ) -> None:
        if record.backing_index.kind is not BackingKind.RUNTIME_MMAP:
            raise BackingError("runtime-mmap backing requires a runtime-mmap index")
        artifact_path = Path(artifact_dir)
        try:
            directory_stat = artifact_path.stat(follow_symlinks=False)
        except FileNotFoundError as exc:
            raise BackingIntegrityError(f"artifact directory does not exist: {artifact_path}") from exc
        if not stat.S_ISDIR(directory_stat.st_mode):
            raise BackingIntegrityError(f"artifact path is not a directory: {artifact_path}")

        if verify_integrity:
            validate_runtime_record_files(artifact_path, record)

        mappings: dict[str, mmap.mmap | None] = {}
        try:
            for storage_object in record.backing_index.objects:
                assert storage_object.relative_path is not None
                object_path = safe_storage_path(artifact_path, storage_object.relative_path)
                object_stat = object_path.stat(follow_symlinks=False)
                if not stat.S_ISREG(object_stat.st_mode):
                    raise BackingIntegrityError(f"storage object is not a regular file: {object_path}")
                if object_stat.st_size != storage_object.nbytes:
                    raise BackingIntegrityError(
                        f"storage size mismatch for {storage_object.object_id!r}: "
                        f"expected {storage_object.nbytes}, got {object_stat.st_size}"
                    )
                if storage_object.nbytes == 0:
                    mappings[storage_object.object_id] = None
                    continue
                flags = os.O_RDONLY
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                file_descriptor = os.open(object_path, flags)
                try:
                    mappings[storage_object.object_id] = mmap.mmap(
                        file_descriptor,
                        length=0,
                        access=mmap.ACCESS_READ,
                    )
                finally:
                    safe_close_fd(
                        file_descriptor,
                        "runtime-mmap source descriptor close also failed",
                    )
        except BaseException as exc:
            for mapping in mappings.values():
                if mapping is not None:
                    try:
                        mapping.close()
                    except BaseException as cleanup_exc:
                        safe_add_exception_note(
                            exc,
                            "runtime-mmap constructor mapping cleanup also failed",
                            cleanup_exc,
                        )
            raise

        self.manifest = record.manifest
        self.backing_index = record.backing_index
        self._mappings = mappings
        self._closed = False
        self._lock = threading.RLock()

    def _source_tensor(self, spec: TensorSpec) -> torch.Tensor:
        span = self.backing_index.tensor_spans[spec.tensor_id]
        if span.length_bytes == 0:
            return torch.empty_strided(
                spec.shape,
                spec.stride,
                dtype=_torch_dtype(spec.dtype),
                device="cpu",
            )
        mapping = self._mappings[span.object_id]
        assert mapping is not None
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given buffer is not writable")
            raw = torch.frombuffer(
                mapping,
                dtype=torch.uint8,
                count=span.length_bytes,
                offset=span.offset_bytes,
            )
        typed = raw.view(_torch_dtype(spec.dtype))
        return torch.as_strided(typed, size=spec.shape, stride=spec.stride)

    def copy_into(self, tensor_id: str, destination: torch.Tensor) -> None:
        with self._lock:
            if self._closed:
                raise BackingClosedError("runtime-mmap backing is closed")
            try:
                spec = self.manifest.tensor(tensor_id)
            except KeyError as exc:
                raise BackingError(f"unknown tensor id: {tensor_id!r}") from exc
            _validate_destination(spec, destination)
            source = self._source_tensor(spec)
            with torch.no_grad():
                destination.copy_(source, non_blocking=False)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            primary: BaseException | None = None
            for object_id, mapping in tuple(self._mappings.items()):
                if mapping is not None:
                    try:
                        mapping.close()
                    except BaseException as exc:
                        if primary is None:
                            primary = exc
                        else:
                            safe_add_exception_note(
                                primary,
                                f"runtime-mmap mapping {object_id!r} cleanup also failed",
                                exc,
                            )
                        continue
                del self._mappings[object_id]
            if not self._mappings:
                self._closed = True
            if primary is not None:
                raise primary


class LoadedTensorBackingProvider:
    def capabilities(self) -> BackingCapabilities:
        return BackingCapabilities(
            BackingKind.LOADED_TENSOR,
            "loaded-tensor",
            "1",
            frozenset({AccessFeature.COMPLETE_TENSOR_READ}),
        )

    def open(self, manifest: ArtifactManifest, index: BackingIndex) -> WeightBacking:
        del manifest, index
        raise BackingError("loaded-tensor provider requires an ephemeral tensor source")

    def open_ephemeral(
        self,
        spec: ArtifactSpec,
        tensors: Mapping[str, torch.Tensor],
        *,
        roles: Mapping[str, TensorRole] | None = None,
        format_metadata: Mapping[str, JSONValue] | None = None,
    ) -> WeightBacking:
        return LoadedTensorBacking.from_tensors(
            spec,
            tensors,
            roles=roles,
            format_metadata=format_metadata,
        )


__all__ = [
    "BackingClosedError",
    "BackingError",
    "BackingIntegrityError",
    "BackingProvider",
    "BackingProviderRegistry",
    "EphemeralBackingProvider",
    "LoadedTensorBacking",
    "LoadedTensorBackingProvider",
    "RuntimeMmapBacking",
    "StaticBackingProviderRegistry",
    "WeightBacking",
    "normalized_storage_view",
    "safe_storage_path",
    "sha256_file",
    "tensor_spec_from_tensor",
    "torch_dtype_name",
    "validate_runtime_record_files",
]
