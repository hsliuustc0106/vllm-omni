# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile diffusion model topology into backing-neutral weight units."""

from __future__ import annotations

from dataclasses import dataclass
from operator import attrgetter

from torch import nn

from vllm_omni.diffusion.offloader.block_discovery import get_blocks_from_dit
from vllm_omni.diffusion.offloader.module_collector import ModuleDiscovery
from vllm_omni.diffusion.offloader.offload_plan import get_offload_plan

from .catalog import (
    ArtifactUnitDeclaration,
    CatalogError,
    ModuleScope,
    TransferUnitDeclaration,
    WeightCatalog,
    build_weight_catalog,
)
from .transfer import (
    BindingDestination,
    DtypePlaneSpec,
    ModuleStateKind,
    ModuleUnitBinding,
    PlaneId,
    TargetModulePath,
    TensorBindingSpec,
    TensorPlacement,
    TransferCatalog,
    TransferPlan,
    TransferPlanKind,
    TransferUnitSpec,
    UnitKind,
    compute_exact_coverage_digest,
    compute_transfer_catalog_digest,
)


@dataclass(frozen=True)
class PipelineWeightCatalog:
    """Catalog plus stable model-path mappings used by offloader adapters."""

    catalog: WeightCatalog
    artifact_unit_by_dit: dict[str, str]
    component_unit_by_dit: dict[str, str]
    block_unit_by_path: dict[str, str]
    resident_unit_by_dit: dict[str, str]
    transfer_catalog: TransferCatalog


def _canonical_module_paths(
    pipeline: nn.Module,
    managed_roots: tuple[nn.Module, ...],
) -> dict[int, str]:
    managed_ids = {id(module) for managed_root in managed_roots for module in managed_root.modules()}
    paths: dict[int, str] = {}
    for path, module in pipeline.named_modules(remove_duplicate=False):
        if id(module) not in managed_ids:
            continue
        previous = paths.setdefault(id(module), path)
        if previous != path:
            raise CatalogError(
                f"module object is reachable through both {previous!r} and {path!r}; "
                "module aliases are outside the v1 host-weight contract"
            )
    return paths


def _plane_id(unit_id: str, dtype: object) -> PlaneId:
    dtype_name = str(dtype).removeprefix("torch.").replace(".", "_")
    return PlaneId(f"{unit_id}/plane/{dtype_name}")


def _target_relative_path(pipeline_path: str, target_path: str) -> TargetModulePath:
    if pipeline_path == target_path:
        return TargetModulePath(".")
    prefix = f"{target_path}."
    if not pipeline_path.startswith(prefix):
        raise CatalogError(f"module path {pipeline_path!r} is outside managed target {target_path!r}")
    return TargetModulePath(pipeline_path.removeprefix(prefix))


def _compile_unit_spec(
    catalog: WeightCatalog,
    *,
    unit_id: str,
    unit_kind: UnitKind,
    tensor_ids: tuple[str, ...],
    artifact_prefix: str,
    target_module_path: str,
) -> TransferUnitSpec:
    bindings: list[TensorBindingSpec] = []
    by_dtype: dict[object, list[object]] = {}
    for tensor_id in tensor_ids:
        target = catalog.tensor(tensor_id)
        if not tensor_id.startswith(artifact_prefix):
            raise CatalogError(f"tensor {tensor_id!r} is outside artifact prefix {artifact_prefix!r}")
        artifact_tensor_id = tensor_id.removeprefix(artifact_prefix)
        bindings.append(
            TensorBindingSpec(
                tensor_id=artifact_tensor_id,
                destination=BindingDestination(
                    module_path=_target_relative_path(target.owner_module_path, target_module_path),
                    attribute_name=target.local_name,
                    state_kind=ModuleStateKind(target.role.value),
                ),
            )
        )
        by_dtype.setdefault(target.dtype, []).append(target)

    planes: list[DtypePlaneSpec] = []
    for dtype, targets in by_dtype.items():
        offset = 0
        placements: list[TensorPlacement] = []
        for target in targets:
            placements.append(
                TensorPlacement(
                    tensor_id=target.tensor_id.removeprefix(artifact_prefix),
                    offset_numel=offset,
                    logical_shape=target.shape,
                    physical_stride=target.stride,
                    storage_numel=target.storage_numel,
                )
            )
            offset += target.storage_numel
        planes.append(
            DtypePlaneSpec(
                plane_id=_plane_id(unit_id, dtype),
                dtype=dtype,
                storage_numel=offset,
                placements=tuple(placements),
            )
        )
    return TransferUnitSpec(
        unit_id=unit_id,
        unit_kind=unit_kind,
        bindings=tuple(bindings),
        planes=tuple(planes),
    )


def compile_pipeline_weight_catalog(
    pipeline: nn.Module,
    *,
    artifact_compatibility_digest: str = "unresolved",
) -> PipelineWeightCatalog:
    """Declare the complete managed DiT and its alternative transfer views.

    Artifact units intentionally do not depend on the selected offloader. The
    same complete-DiT artifact can serve a component transfer for model-level
    offload or block transfers for layerwise and DLO no-AllGather.
    """
    discovered = ModuleDiscovery.discover(pipeline)
    if not discovered.dits:
        raise CatalogError("no DiT modules were discovered for Host Weight Runtime")
    if len(discovered.dits) != 1:
        raise CatalogError(
            f"v1 Host Weight Runtime requires exactly one managed DiT target; discovered {list(discovered.dit_names)}"
        )
    outer_names, _ = discovered.outermost_dits()
    if len(outer_names) != len(discovered.dit_names):
        nested = sorted(set(discovered.dit_names) - set(outer_names))
        raise CatalogError(
            f"nested discovered DiTs are outside the v1 Host Weight Runtime ownership contract: {nested}"
        )

    paths = _canonical_module_paths(pipeline, tuple(discovered.dits))
    artifact_declarations: list[ArtifactUnitDeclaration] = []
    transfer_declarations: list[TransferUnitDeclaration] = []
    artifact_unit_by_dit: dict[str, str] = {}
    component_unit_by_dit: dict[str, str] = {}
    block_unit_by_path: dict[str, str] = {}
    resident_unit_by_dit: dict[str, str] = {}
    offload_plan = get_offload_plan(pipeline)

    for dit_name, dit_module in zip(discovered.dit_names, discovered.dits, strict=True):
        actual_path = paths.get(id(dit_module))
        if actual_path != dit_name:
            raise CatalogError(f"discovered DiT path mismatch: declared={dit_name!r}, actual={actual_path!r}")

        artifact_unit_id = f"dit.{dit_name}"
        component_unit_id = f"component.{dit_name}"
        dit_scope = ModuleScope(module_path=dit_name, module=dit_module)
        artifact_declarations.append(ArtifactUnitDeclaration(unit_id=artifact_unit_id, scopes=(dit_scope,)))
        transfer_declarations.append(
            TransferUnitDeclaration(
                unit_id=component_unit_id,
                artifact_unit_id=artifact_unit_id,
                scopes=(dit_scope,),
            )
        )
        artifact_unit_by_dit[dit_name] = artifact_unit_id
        component_unit_by_dit[dit_name] = component_unit_id

        _, main_blocks = get_blocks_from_dit(dit_module)
        declared_blocks = list(main_blocks)
        if offload_plan is not None:
            for child_name, blocks_attr in sorted(offload_plan.offload_submodules.items()):
                child = getattr(dit_module, child_name, None)
                if not isinstance(child, nn.Module):
                    continue
                try:
                    auxiliary_blocks = attrgetter(blocks_attr)(child)
                except AttributeError as exc:
                    raise CatalogError(
                        f"OffloadPlan path {child_name}.{blocks_attr} is missing under DiT {dit_name!r}"
                    ) from exc
                if not isinstance(auxiliary_blocks, nn.ModuleList):
                    raise CatalogError(
                        f"OffloadPlan path {child_name}.{blocks_attr} under DiT {dit_name!r} is not an nn.ModuleList"
                    )
                declared_blocks.extend(auxiliary_blocks)

        seen_blocks: set[int] = set()
        for block in declared_blocks:
            if id(block) in seen_blocks:
                raise CatalogError(
                    f"streaming block {type(block).__name__} was declared more than once under DiT {dit_name!r}"
                )
            seen_blocks.add(id(block))
            block_path = paths.get(id(block))
            if block_path is None or not block_path.startswith(f"{dit_name}."):
                raise CatalogError(f"streaming block {type(block).__name__} is not a descendant of DiT {dit_name!r}")
            transfer_unit_id = f"block.{block_path}"
            if block_path in block_unit_by_path:
                raise CatalogError(f"streaming block path {block_path!r} was declared more than once")
            transfer_declarations.append(
                TransferUnitDeclaration(
                    unit_id=transfer_unit_id,
                    artifact_unit_id=artifact_unit_id,
                    scopes=(ModuleScope(module_path=block_path, module=block),),
                )
            )
            block_unit_by_path[block_path] = transfer_unit_id

    catalog = build_weight_catalog(
        artifact_units=tuple(artifact_declarations),
        transfer_units=tuple(transfer_declarations),
    )

    unit_specs: list[TransferUnitSpec] = []
    for dit_name, artifact_unit_id in artifact_unit_by_dit.items():
        artifact_prefix = f"{dit_name}."
        component_unit_id = component_unit_by_dit[dit_name]
        component = catalog.transfer(component_unit_id)
        unit_specs.append(
            _compile_unit_spec(
                catalog,
                unit_id=component.unit_id,
                unit_kind=UnitKind.COMPONENT,
                tensor_ids=component.tensor_ids,
                artifact_prefix=artifact_prefix,
                target_module_path=dit_name,
            )
        )
        block_ids: set[str] = set()
        for block_path, block_unit_id in block_unit_by_path.items():
            if not block_path.startswith(f"{dit_name}."):
                continue
            block = catalog.transfer(block_unit_id)
            block_ids.update(block.tensor_ids)
            unit_specs.append(
                _compile_unit_spec(
                    catalog,
                    unit_id=block.unit_id,
                    unit_kind=UnitKind.BLOCK,
                    tensor_ids=block.tensor_ids,
                    artifact_prefix=artifact_prefix,
                    target_module_path=dit_name,
                )
            )
        artifact = catalog.artifact(artifact_unit_id)
        resident_ids = tuple(tensor_id for tensor_id in artifact.tensor_ids if tensor_id not in block_ids)
        if resident_ids:
            resident_unit_id = f"resident.{dit_name}"
            resident_unit_by_dit[dit_name] = resident_unit_id
            unit_specs.append(
                _compile_unit_spec(
                    catalog,
                    unit_id=resident_unit_id,
                    unit_kind=UnitKind.RESIDENT,
                    tensor_ids=resident_ids,
                    artifact_prefix=artifact_prefix,
                    target_module_path=dit_name,
                )
            )

    units = tuple(unit_specs)
    dit_name = discovered.dit_names[0]
    component_unit_id = component_unit_by_dit[dit_name]
    component_execution = (
        ModuleUnitBinding(
            module_path=TargetModulePath("."),
            unit_id=component_unit_id,
        ),
    )
    component_unit_ids = (component_unit_id,)
    component_plan = TransferPlan(
        plan_id="plan.component",
        plan_kind=TransferPlanKind.COMPONENT,
        unit_ids=component_unit_ids,
        execution_bindings=component_execution,
        exact_coverage_digest=compute_exact_coverage_digest(
            component_unit_ids,
            component_execution,
            units,
        ),
    )

    block_execution = tuple(
        ModuleUnitBinding(
            module_path=_target_relative_path(block_path, dit_name),
            unit_id=unit_id,
        )
        for block_path, unit_id in block_unit_by_path.items()
    )
    block_plan_unit_ids = tuple(binding.unit_id for binding in block_execution)
    resident_unit_id = resident_unit_by_dit.get(dit_name)
    if resident_unit_id is not None:
        block_plan_unit_ids += (resident_unit_id,)
    blocks_plus_resident_plan = TransferPlan(
        plan_id="plan.blocks_plus_resident",
        plan_kind=TransferPlanKind.BLOCKS_PLUS_RESIDENT,
        unit_ids=block_plan_unit_ids,
        execution_bindings=block_execution,
        exact_coverage_digest=compute_exact_coverage_digest(
            block_plan_unit_ids,
            block_execution,
            units,
        ),
    )
    plans = (component_plan, blocks_plus_resident_plan)
    transfer_catalog = TransferCatalog(
        artifact_compatibility_digest=artifact_compatibility_digest,
        transfer_catalog_digest=compute_transfer_catalog_digest(units, plans),
        units=units,
        plans=plans,
    )
    return PipelineWeightCatalog(
        catalog=catalog,
        artifact_unit_by_dit=artifact_unit_by_dit,
        component_unit_by_dit=component_unit_by_dit,
        block_unit_by_path=block_unit_by_path,
        resident_unit_by_dit=resident_unit_by_dit,
        transfer_catalog=transfer_catalog,
    )


__all__ = ["PipelineWeightCatalog", "compile_pipeline_weight_catalog"]
