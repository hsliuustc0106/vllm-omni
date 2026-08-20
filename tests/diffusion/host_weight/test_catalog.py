# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.host_weight import (
    ArtifactUnitDeclaration,
    CatalogError,
    ModuleScope,
    TensorRole,
    TransferUnitDeclaration,
    build_weight_catalog,
)

from .conftest import TinyPipeline

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _build_catalog(pipeline: TinyPipeline):
    transformer = ModuleScope.from_pipeline(pipeline, "transformer")
    blocks = [
        ModuleScope.from_pipeline(pipeline, f"transformer.blocks.{index}")
        for index in range(len(pipeline.transformer.blocks))
    ]
    return build_weight_catalog(
        artifact_units=(ArtifactUnitDeclaration("fl2va.dit", (transformer,)),),
        transfer_units=(
            TransferUnitDeclaration("fl2va.dit.complete", "fl2va.dit", (transformer,)),
            *(
                TransferUnitDeclaration(f"fl2va.dit.block.{index}", "fl2va.dit", (block,))
                for index, block in enumerate(blocks)
            ),
        ),
    )


def test_catalog_separates_artifact_and_transfer_units() -> None:
    pipeline = TinyPipeline()
    catalog = _build_catalog(pipeline)

    artifact = catalog.artifact("fl2va.dit")
    complete = catalog.transfer("fl2va.dit.complete")
    block = catalog.transfer("fl2va.dit.block.0")

    assert artifact.tensor_ids == complete.tensor_ids
    assert set(block.tensor_ids) < set(artifact.tensor_ids)
    assert block.artifact_unit_id == artifact.unit_id
    assert "transformer.blocks.0.scratch" not in artifact.tensor_ids

    scale = catalog.tensor("transformer.blocks.0.weight_scale")
    assert scale.role is TensorRole.PERSISTENT_BUFFER
    assert scale.dtype is torch.float32
    weight = catalog.tensor("transformer.blocks.0.weight")
    assert weight.role is TensorRole.PARAMETER
    assert weight.shape == (4, 3)
    assert weight.stride == (1, 4)
    assert weight.storage_numel == 12


def test_transfer_scope_must_be_owned_by_declared_artifact() -> None:
    pipeline = TinyPipeline()
    transformer = ModuleScope.from_pipeline(pipeline, "transformer")
    encoder = ModuleScope.from_pipeline(pipeline, "text_encoder")

    with pytest.raises(CatalogError, match="not managed by any artifact"):
        build_weight_catalog(
            artifact_units=(ArtifactUnitDeclaration("dit", (transformer,)),),
            transfer_units=(TransferUnitDeclaration("encoder", "dit", (encoder,)),),
        )


def test_artifact_units_cannot_overlap() -> None:
    pipeline = TinyPipeline()
    transformer = ModuleScope.from_pipeline(pipeline, "transformer")

    with pytest.raises(CatalogError, match="owned by both"):
        build_weight_catalog(
            artifact_units=(
                ArtifactUnitDeclaration("dit.a", (transformer,)),
                ArtifactUnitDeclaration("dit.b", (transformer,)),
            ),
            transfer_units=(),
        )


def test_tensor_aliases_fail_closed() -> None:
    module = nn.Module()
    shared = nn.Parameter(torch.ones(2), requires_grad=False)
    module.register_parameter("first", shared)
    module.register_parameter("second", shared)

    with pytest.raises(CatalogError, match="aliased tensor names"):
        build_weight_catalog(
            artifact_units=(ArtifactUnitDeclaration("aliased", (ModuleScope("", module),)),),
            transfer_units=(),
        )
