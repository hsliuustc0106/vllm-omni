# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer-3 dependency-injection tests for repositories and backing providers."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from conftest import ArtifactRegistrar

from vllm_omni.host_weight_runtime import (
    AccessFeature,
    AccessRequirements,
    ArtifactManifest,
    ArtifactRecord,
    BackingCapabilities,
    BackingError,
    BackingIndex,
    BackingKind,
    BuildAuthorization,
    BuildRole,
    CapabilityGrant,
    Existing,
    HostWeightRuntime,
    LocalArtifactRepository,
    Ready,
    ResolutionPath,
    StaticBackingProviderRegistry,
    StorageObject,
    StorageSpan,
    TensorSelection,
    create_default_host_weight_runtime,
    tensor_spec_from_tensor,
)


class _InMemoryBacking:
    def __init__(self, record: ArtifactRecord, tensors: dict[str, torch.Tensor]) -> None:
        self.manifest = record.manifest
        self.backing_index = record.backing_index
        self._tensors = tensors
        self.closed = False

    def copy_into(self, tensor_id: str, destination: torch.Tensor) -> None:
        if self.closed:
            raise RuntimeError("backing is closed")
        destination.copy_(self._tensors[tensor_id])

    def close(self) -> None:
        self.closed = True


class _InjectedProvider:
    def __init__(self, record: ArtifactRecord, tensors: dict[str, torch.Tensor]) -> None:
        self.record = record
        self.tensors = tensors
        self.open_calls = 0

    def capabilities(self) -> BackingCapabilities:
        return BackingCapabilities(
            BackingKind.RUNTIME_MMAP,
            "injected-mmap",
            "test/v1",
            frozenset(
                {
                    AccessFeature.COMPLETE_TENSOR_READ,
                    AccessFeature.SHARED_PAGES,
                }
            ),
        )

    def open(self, manifest: ArtifactManifest, index: BackingIndex) -> _InMemoryBacking:
        self.open_calls += 1
        assert manifest == self.record.manifest
        assert index == self.record.backing_index
        return _InMemoryBacking(self.record, self.tensors)


class _HitRepository:
    def __init__(self, existing: Existing) -> None:
        self.existing = existing
        self.lookup_calls = 0

    def lookup(self, key: str) -> Existing | None:
        self.lookup_calls += 1
        return self.existing if key == self.existing.record.manifest.artifact_key else None

    def claim(self, *_args, **_kwargs):
        raise AssertionError("repository hit must not claim")

    def wait(self, *_args, **_kwargs):
        raise AssertionError("repository hit must not wait")

    def create_sink(self, *_args, **_kwargs):
        raise AssertionError("repository hit must not create a sink")

    def commit(self, *_args, **_kwargs):
        raise AssertionError("repository hit must not commit")


def _record(artifact_spec, source: torch.Tensor) -> ArtifactRecord:
    tensor = tensor_spec_from_tensor("weight", source)
    manifest = ArtifactManifest.create(artifact_spec, (tensor,), {})
    index = BackingIndex(
        artifact_key=artifact_spec.artifact_key,
        generation_id="injected-generation",
        kind=BackingKind.RUNTIME_MMAP,
        provider_name="injected-mmap",
        provider_version="test/v1",
        objects=(
            StorageObject(
                "in-memory",
                tensor.storage_nbytes,
                max(1, source.element_size()),
                tensor.content_digest,
                "provider-owned.bin",
            ),
        ),
        tensor_spans={"weight": StorageSpan("in-memory", 0, tensor.storage_nbytes)},
    )
    return ArtifactRecord(manifest, index)


def test_runtime_resolves_through_injected_repository_and_provider(
    artifact_spec,
) -> None:
    source = torch.arange(4, dtype=torch.float32)
    record = _record(artifact_spec, source)
    provider = _InjectedProvider(record, {"weight": source})
    registry = StaticBackingProviderRegistry((provider,))
    repository = _HitRepository(Existing(record, Path("provider-owned-location")))
    runtime = HostWeightRuntime(repository, registry)  # type: ignore[arg-type]
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
    registrar = ArtifactRegistrar()

    outcome = runtime.resolve(
        artifact_spec,
        grant,
        None,
        BuildAuthorization(
            BuildRole.READ_ONLY,
            "reader",
            "builder",
            "launch",
        ),
        registrar,
    )

    assert isinstance(outcome, Ready)
    assert outcome.info.path is ResolutionPath.MMAP_HIT
    assert provider.open_calls == 1
    assert repository.lookup_calls == 1
    destination = torch.empty_like(source)
    with outcome.artifact.open(TensorSelection.one("weight")) as view:
        view.copy_into({"weight": destination})
    torch.testing.assert_close(destination, source)
    outcome.artifact.close()
    runtime.close()


def test_registry_rejects_duplicate_backing_kinds(artifact_spec) -> None:
    source = torch.arange(1, dtype=torch.float32)
    record = _record(artifact_spec, source)

    with pytest.raises(BackingError, match="duplicate backing provider"):
        StaticBackingProviderRegistry(
            (
                _InjectedProvider(record, {"weight": source}),
                _InjectedProvider(record, {"weight": source}),
            )
        )


def test_explicit_default_factory_composes_loaded_and_local_mmap(tmp_path) -> None:
    loaded_only = create_default_host_weight_runtime()
    assert {item.kind for item in loaded_only.capabilities.backings} == {BackingKind.LOADED_TENSOR}
    loaded_only.close()

    repository = LocalArtifactRepository(tmp_path)
    local = create_default_host_weight_runtime(repository)
    assert {item.kind for item in local.capabilities.backings} == {
        BackingKind.LOADED_TENSOR,
        BackingKind.RUNTIME_MMAP,
    }
    local.close()


def test_local_composition_is_side_effect_free_until_repository_use(tmp_path) -> None:
    root = tmp_path / "host-weight-cache"

    repository = LocalArtifactRepository(root)
    runtime = create_default_host_weight_runtime(repository)

    assert not root.exists()
    runtime.close()
