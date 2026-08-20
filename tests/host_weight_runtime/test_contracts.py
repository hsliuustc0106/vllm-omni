# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace

import pytest
import torch
from conftest import digest, make_weight_format

from vllm_omni.host_weight_runtime import (
    AccessFeature,
    AccessRequirements,
    ArtifactManifest,
    ArtifactRecord,
    ArtifactSpec,
    ArtifactTopologyDescriptor,
    BackingIndex,
    BackingKind,
    CapabilitiesUnavailable,
    CapabilityGrant,
    ContractError,
    HostCopyMode,
    LocalArtifactRepository,
    ProducerDescriptor,
    StorageObject,
    StorageSpan,
    TensorRole,
    TensorSpec,
    TopologyCoordinate,
    canonical_digest,
    derive_artifact_key,
    tensor_spec_from_tensor,
    validate_manifest_against_spec,
)
from vllm_omni.host_weight_runtime import (
    create_default_host_weight_runtime as HostWeightRuntime,
)


def test_artifact_key_is_canonical_across_topology_input_order(
    producer_descriptor,
    weight_format,
) -> None:
    first = ArtifactSpec(
        source_fingerprint=digest("source"),
        producer=producer_descriptor,
        weight_format=weight_format,
        topology=ArtifactTopologyDescriptor((TopologyCoordinate("tp", 2, 1), TopologyCoordinate("pp", 3, 0))),
        layout_abi="layout/v1",
    )
    second = replace(
        first,
        topology=ArtifactTopologyDescriptor((TopologyCoordinate("pp", 3, 0), TopologyCoordinate("tp", 2, 1))),
    )

    assert first == second
    assert first.artifact_key == second.artifact_key == derive_artifact_key(first)


def test_artifact_topology_rejects_transfer_only_dp_axis() -> None:
    with pytest.raises(ContractError, match="'pp' or 'tp'"):
        TopologyCoordinate("dp", 4, 1)  # type: ignore[arg-type]


def test_format_config_is_deeply_immutable_and_canonical() -> None:
    source = {"activation": {"dynamic": False}, "widths": [32, 64]}
    descriptor = make_weight_format(
        format_id="fp8-per-tensor",
        adapter_abi="adapter/v1",
        semantic_fingerprint=digest("format"),
        normalized_config=source,
        kernel_identity={"kernel": "cutlass"},
    )
    before = canonical_digest(descriptor.to_dict())
    source["widths"].append(128)  # type: ignore[union-attr]

    assert descriptor.to_dict()["normalized_config"]["widths"] == [32, 64]  # type: ignore[index]
    assert canonical_digest(descriptor.to_dict()) == before
    with pytest.raises(TypeError):
        descriptor.normalized_config["new"] = True  # type: ignore[index]


def test_weight_format_plan_digest_covers_structural_import_fields() -> None:
    descriptor = make_weight_format()

    with pytest.raises(ContractError, match="format_plan_digest"):
        replace(descriptor, target_module_type_id="another.transformer")
    with pytest.raises(ContractError, match="format_plan_digest"):
        replace(descriptor, format_recipe_schema_version=2)
    with pytest.raises(ContractError, match="format_plan_digest"):
        replace(descriptor, normalized_config={"dtype": "bfloat16"})


def test_v1_core_exposes_only_complete_sync_host_access() -> None:
    assert tuple(AccessFeature) == (
        AccessFeature.COMPLETE_TENSOR_READ,
        AccessFeature.SHARED_PAGES,
    )
    assert tuple(HostCopyMode) == (HostCopyMode.SYNCHRONOUS,)


def test_tensor_spec_rejects_internally_overlapping_stride() -> None:
    with pytest.raises(ContractError, match="overlapping storage"):
        TensorSpec(
            tensor_id="weight",
            role=TensorRole.PARAMETER,
            dtype="float32",
            shape=(2, 2),
            stride=(1, 1),
            storage_numel=3,
            content_digest=digest("bytes"),
        )


def test_manifest_is_logical_path_independent_and_round_trips(artifact_spec) -> None:
    tensor_spec = tensor_spec_from_tensor(
        "block.weight",
        torch.arange(6, dtype=torch.float32).reshape(2, 3),
    )
    manifest = ArtifactManifest.create(
        artifact_spec,
        (tensor_spec,),
        format_metadata={"recipe_schema": 1},
    )

    assert ArtifactManifest.from_json(manifest.to_json()) == manifest
    assert "weights.bin" not in manifest.to_json()
    assert validate_manifest_against_spec(artifact_spec, manifest) is manifest


def test_artifact_record_keeps_physical_index_separate(artifact_spec) -> None:
    tensor_spec = tensor_spec_from_tensor("weight", torch.arange(4, dtype=torch.float32))
    manifest = ArtifactManifest.create(artifact_spec, (tensor_spec,), {})
    index = BackingIndex(
        artifact_key=artifact_spec.artifact_key,
        generation_id="generation",
        kind=BackingKind.RUNTIME_MMAP,
        provider_name="test-provider",
        provider_version="1",
        objects=(
            StorageObject(
                "weights",
                tensor_spec.storage_nbytes,
                64,
                tensor_spec.content_digest,
                "weights.bin",
            ),
        ),
        tensor_spans={"weight": StorageSpan("weights", 0, tensor_spec.storage_nbytes)},
    )

    record = ArtifactRecord(manifest, index)
    assert record.manifest.compatibility_digest == manifest.compatibility_digest
    assert record.backing_index.object("weights").relative_path == "weights.bin"


def test_backing_index_rejects_cross_tensor_aliases() -> None:
    with pytest.raises(ContractError, match="storage overlaps"):
        BackingIndex(
            artifact_key=digest("artifact"),
            generation_id="generation",
            kind=BackingKind.RUNTIME_MMAP,
            provider_name="test",
            provider_version="1",
            objects=(StorageObject("weights", 16, 8, digest("object"), "weights.bin"),),
            tensor_spans={
                "first": StorageSpan("weights", 0, 8),
                "second": StorageSpan("weights", 4, 8),
            },
        )


@pytest.mark.parametrize("path", ["../escape", "/absolute", "nested/../../escape"])
def test_backing_index_rejects_unsafe_paths(path: str) -> None:
    with pytest.raises(ContractError, match="safe relative"):
        StorageObject("weights", 4, 4, digest("object"), path)


def test_semantic_validation_rejects_wrong_spec_and_digest(artifact_spec) -> None:
    tensor_spec = tensor_spec_from_tensor("weight", torch.arange(4, dtype=torch.float32))
    manifest = ArtifactManifest.create(artifact_spec, (tensor_spec,), {})
    wrong_producer = ProducerDescriptor("other", "other/v1", digest("other"))
    wrong_spec = replace(artifact_spec, producer=wrong_producer)

    with pytest.raises(ContractError, match="artifact key"):
        validate_manifest_against_spec(wrong_spec, manifest)
    with pytest.raises(ContractError, match="compatibility digest"):
        validate_manifest_against_spec(
            artifact_spec,
            replace(manifest, compatibility_digest=digest("invalid")),
        )


def test_capability_negotiation_requires_one_backing_to_satisfy_all_features(
    tmp_path,
) -> None:
    runtime = HostWeightRuntime(LocalArtifactRepository(tmp_path))
    loaded_only = runtime.negotiate(
        AccessRequirements(
            required_features=frozenset({AccessFeature.COMPLETE_TENSOR_READ, AccessFeature.SHARED_PAGES}),
            accepted_backings=frozenset({BackingKind.LOADED_TENSOR}),
        )
    )
    mmap = runtime.negotiate(
        AccessRequirements(
            required_features=frozenset({AccessFeature.COMPLETE_TENSOR_READ, AccessFeature.SHARED_PAGES}),
            accepted_backings=frozenset({BackingKind.RUNTIME_MMAP}),
        )
    )

    assert isinstance(loaded_only, CapabilitiesUnavailable)
    assert loaded_only.missing_features_by_backing[BackingKind.LOADED_TENSOR] == frozenset({AccessFeature.SHARED_PAGES})
    assert isinstance(mmap, CapabilityGrant)
    assert mmap.backing_kind is BackingKind.RUNTIME_MMAP
    assert mmap.provider_id == "local-runtime-mmap"
    runtime.close()


def test_manifest_decoder_rejects_unknown_schema_fields(artifact_spec) -> None:
    tensor_spec = tensor_spec_from_tensor("weight", torch.arange(1, dtype=torch.float32))
    value = ArtifactManifest.create(artifact_spec, (tensor_spec,), {}).to_dict()
    value["executable_import"] = "evil.module"  # type: ignore[assignment]

    with pytest.raises(ContractError, match="extra"):
        ArtifactManifest.from_dict(value)
