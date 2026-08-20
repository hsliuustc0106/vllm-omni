# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace

import pytest
from torch import nn

from vllm_omni.diffusion.host_weight.pipeline_catalog import (
    compile_pipeline_weight_catalog,
)
from vllm_omni.diffusion.host_weight.transfer import (
    TransferCatalog,
    TransferPlan,
    TransferPlanError,
    TransferPlanKind,
    UnitKind,
    compute_exact_coverage_digest,
    compute_transfer_catalog_digest,
)
from vllm_omni.diffusion.offloader.offload_plan import OffloadPlan

from .conftest import TinyPipeline, TinyTransformer

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_pipeline_catalog_reuses_one_artifact_across_component_and_blocks(monkeypatch) -> None:
    monkeypatch.setattr(
        TinyTransformer,
        "_layerwise_offload_blocks_attrs",
        ["blocks"],
        raising=False,
    )
    compiled = compile_pipeline_weight_catalog(TinyPipeline())

    assert compiled.artifact_unit_by_dit == {"transformer": "dit.transformer"}
    assert compiled.component_unit_by_dit == {"transformer": "component.transformer"}
    assert compiled.block_unit_by_path == {
        "transformer.blocks.0": "block.transformer.blocks.0",
        "transformer.blocks.1": "block.transformer.blocks.1",
    }
    assert compiled.resident_unit_by_dit == {"transformer": "resident.transformer"}

    artifact = compiled.catalog.artifact("dit.transformer")
    component = compiled.catalog.transfer("component.transformer")
    first_block = compiled.catalog.transfer("block.transformer.blocks.0")
    assert component.tensor_ids == artifact.tensor_ids
    assert set(first_block.tensor_ids) < set(artifact.tensor_ids)

    public_units = compiled.transfer_catalog.units
    assert [unit.unit_kind for unit in public_units] == [
        UnitKind.COMPONENT,
        UnitKind.BLOCK,
        UnitKind.BLOCK,
        UnitKind.RESIDENT,
    ]
    assert public_units[0].tensor_ids == tuple(
        tensor_id.removeprefix("transformer.") for tensor_id in artifact.tensor_ids
    )
    assert [plan.plan_kind for plan in compiled.transfer_catalog.plans] == [
        TransferPlanKind.COMPONENT,
        TransferPlanKind.BLOCKS_PLUS_RESIDENT,
    ]
    component_plan, blocks_plan = compiled.transfer_catalog.plans
    assert component_plan.unit_ids == ("component.transformer",)
    assert [(str(item.module_path), item.unit_id) for item in component_plan.execution_bindings] == [
        (".", "component.transformer")
    ]
    assert blocks_plan.unit_ids == (
        "block.transformer.blocks.0",
        "block.transformer.blocks.1",
        "resident.transformer",
    )
    assert [(str(item.module_path), item.unit_id) for item in blocks_plan.execution_bindings] == [
        ("blocks.0", "block.transformer.blocks.0"),
        ("blocks.1", "block.transformer.blocks.1"),
    ]
    component_coverage = {
        (binding.tensor_id, binding.destination)
        for unit_id in component_plan.unit_ids
        for binding in compiled.transfer_catalog.unit(unit_id).bindings
    }
    block_coverage = {
        (binding.tensor_id, binding.destination)
        for unit_id in blocks_plan.unit_ids
        for binding in compiled.transfer_catalog.unit(unit_id).bindings
    }
    assert component_coverage == block_coverage
    assert all(
        str(plane.plane_id).startswith(f"{unit.unit_id}/plane/") for unit in public_units for plane in unit.planes
    )
    assert (
        len(
            {
                compiled.transfer_catalog.transfer_catalog_digest,
                compiled.transfer_catalog.artifact_compatibility_digest,
            }
        )
        == 2
    )


def test_pipeline_catalog_ignores_aliases_outside_managed_dits(monkeypatch) -> None:
    monkeypatch.setattr(
        TinyTransformer,
        "_layerwise_offload_blocks_attrs",
        ["blocks"],
        raising=False,
    )
    pipeline = TinyPipeline()
    baseline = compile_pipeline_weight_catalog(pipeline)
    pipeline.video_vae = nn.Module()
    pipeline.video_vae.remote = nn.Module()
    pipeline.video_vae.remote.model = nn.Linear(2, 2, bias=False)
    pipeline.video_vae.model = pipeline.video_vae.remote.model
    pipeline.vae = pipeline.video_vae

    compiled = compile_pipeline_weight_catalog(pipeline)

    assert compiled.catalog == baseline.catalog
    assert compiled.transfer_catalog == baseline.transfer_catalog


def test_pipeline_catalog_rejects_aliases_involving_managed_dits(monkeypatch) -> None:
    monkeypatch.setattr(
        TinyTransformer,
        "_layerwise_offload_blocks_attrs",
        ["blocks"],
        raising=False,
    )
    pipeline = TinyPipeline()
    pipeline.transformer_alias = pipeline.transformer

    with pytest.raises(ValueError, match="module aliases are outside"):
        compile_pipeline_weight_catalog(pipeline)


def test_pipeline_catalog_rejects_external_alias_to_managed_descendant(monkeypatch) -> None:
    monkeypatch.setattr(
        TinyTransformer,
        "_layerwise_offload_blocks_attrs",
        ["blocks"],
        raising=False,
    )
    pipeline = TinyPipeline()
    pipeline.external_block_alias = pipeline.transformer.blocks[0]

    with pytest.raises(ValueError, match="module aliases are outside"):
        compile_pipeline_weight_catalog(pipeline)


def test_pipeline_catalog_includes_plan_declared_auxiliary_block_ring(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        TinyTransformer,
        "_layerwise_offload_blocks_attrs",
        ["blocks"],
        raising=False,
    )

    class Auxiliary(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(4, 4, bias=False)
            self.layers = nn.ModuleList([nn.Linear(4, 4, bias=False), nn.Linear(4, 4, bias=False)])

    class PlannedPipeline(TinyPipeline):
        _offload_plan = OffloadPlan(offload_submodules={"token_refiner": "layers"})

        def __init__(self) -> None:
            super().__init__()
            self.transformer.token_refiner = Auxiliary()

    compiled = compile_pipeline_weight_catalog(PlannedPipeline())

    assert {
        "transformer.blocks.0",
        "transformer.blocks.1",
        "transformer.token_refiner.layers.0",
        "transformer.token_refiner.layers.1",
    } == set(compiled.block_unit_by_path)
    resident = compiled.transfer_catalog.unit("resident.transformer")
    assert any(binding.destination.module_path == "token_refiner.proj" for binding in resident.bindings)
    assert all("token_refiner.layers." not in binding.destination.module_path for binding in resident.bindings)


def test_pipeline_catalog_rejects_multiple_outer_dit_targets(monkeypatch) -> None:
    monkeypatch.setattr(
        TinyTransformer,
        "_layerwise_offload_blocks_attrs",
        ["blocks"],
        raising=False,
    )

    class MultiDitPipeline(TinyPipeline):
        _dit_modules = ["transformer", "second_transformer"]
        _encoder_modules = []
        _vae_modules = []
        _resident_modules = []

        def __init__(self) -> None:
            super().__init__()
            self.second_transformer = TinyTransformer()

    with pytest.raises(ValueError, match="exactly one managed DiT target"):
        compile_pipeline_weight_catalog(MultiDitPipeline())


def test_transfer_catalog_rejects_inexact_and_duplicate_plan_views(monkeypatch) -> None:
    monkeypatch.setattr(
        TinyTransformer,
        "_layerwise_offload_blocks_attrs",
        ["blocks"],
        raising=False,
    )
    catalog = compile_pipeline_weight_catalog(TinyPipeline()).transfer_catalog
    component_plan, blocks_plan = catalog.plans

    duplicate_kind = replace(
        blocks_plan,
        plan_kind=TransferPlanKind.COMPONENT,
    )
    with pytest.raises(TransferPlanError, match="duplicate transfer plan kinds"):
        TransferCatalog(
            artifact_compatibility_digest=catalog.artifact_compatibility_digest,
            transfer_catalog_digest=catalog.transfer_catalog_digest,
            units=catalog.units,
            plans=(component_plan, duplicate_kind),
        )

    incomplete_unit_ids = blocks_plan.unit_ids[:-1]
    incomplete_plan = TransferPlan(
        plan_id=blocks_plan.plan_id,
        plan_kind=blocks_plan.plan_kind,
        unit_ids=incomplete_unit_ids,
        execution_bindings=blocks_plan.execution_bindings,
        exact_coverage_digest=compute_exact_coverage_digest(
            incomplete_unit_ids,
            blocks_plan.execution_bindings,
            catalog.units,
        ),
    )
    incomplete_plans = (component_plan, incomplete_plan)
    with pytest.raises(TransferPlanError, match="exact tensor/destination coverage"):
        TransferCatalog(
            artifact_compatibility_digest=catalog.artifact_compatibility_digest,
            transfer_catalog_digest=compute_transfer_catalog_digest(catalog.units, incomplete_plans),
            units=catalog.units,
            plans=incomplete_plans,
        )
