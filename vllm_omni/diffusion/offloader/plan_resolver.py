# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Central, read-only resolver for diffusion CPU-offload topology."""

from __future__ import annotations

from dataclasses import dataclass, replace
from operator import attrgetter

from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery

from .base import OffloadConfig
from .block_discovery import get_blocks_attr_names
from .config import DIT_COMPONENT, TEXT_ENCODER_COMPONENT, OffloadStrategy
from .module_collector import ModuleDiscovery, PipelineModules
from .offload_plan import (
    HostResidentTableSpec,
    OffloadComponentSpec,
    OffloadPlan,
    OffloadPlanSource,
    OffloadSelectionMode,
    OffloadWeightLayout,
    ResolvedBlockGroup,
    ResolvedHostResidentTable,
    ResolvedOffloadComponent,
    ResolvedOffloadPhase,
    ResolvedOffloadPlan,
    get_offload_plan,
)

logger = init_logger(__name__)

_ENCODER_COMPONENT = "encoder"
_VAE_COMPONENT = "vae"


@dataclass
class _ComponentDraft:
    spec: OffloadComponentSpec
    source: OffloadPlanSource
    block_source: OffloadPlanSource | None = None


def _ordered_unique(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _join_path(prefix: str, relative: str) -> str:
    return f"{prefix}.{relative}" if prefix else relative


def _require_name(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{label} must not have leading or trailing whitespace: {value!r}")
    return value


def _resolve_module(root: nn.Module, path: str, label: str) -> nn.Module:
    _require_name(path, label)
    try:
        module = attrgetter(path)(root)
    except AttributeError as exc:
        raise ValueError(f"{label} {path!r} does not exist") from exc
    if not isinstance(module, nn.Module):
        raise ValueError(f"{label} {path!r} must resolve to nn.Module, got {type(module).__name__}")
    return module


def _infer_encoder_type(path: str, declaration: OffloadPlan) -> str:
    declared = declaration.encoder_component_types.get(path)
    if declared is not None:
        return declared
    leaf = path.rsplit(".", 1)[-1]
    if leaf.startswith(TEXT_ENCODER_COMPONENT) or leaf.endswith(TEXT_ENCODER_COMPONENT):
        return TEXT_ENCODER_COMPONENT
    return _ENCODER_COMPONENT


def _legacy_host_tables(declaration: OffloadPlan, component_path: str) -> tuple[HostResidentTableSpec, ...]:
    return tuple(
        HostResidentTableSpec(forward_module_path=path)
        for path in declaration.encoder_host_resident_table_attrs.get(component_path, ())
    )


def _merge_compatibility_fields(
    spec: OffloadComponentSpec,
    declaration: OffloadPlan,
) -> tuple[OffloadComponentSpec, OffloadPlanSource | None]:
    """Fill universal fields from the pre-J1 declaration without overriding."""
    legacy_component_type = declaration.encoder_component_types.get(spec.module_path)
    if legacy_component_type is not None and legacy_component_type != spec.component_type:
        raise ValueError(
            f"Offload component {spec.component_id!r} declares type {spec.component_type!r}, "
            f"but encoder_component_types declares {legacy_component_type!r}"
        )

    legacy_blocks = declaration.block_attrs.get(spec.module_path)
    if legacy_blocks is None:
        legacy_blocks = declaration.encoder_block_attrs.get(spec.module_path)
    block_source = None
    if legacy_blocks:
        normalized_legacy_blocks = tuple(legacy_blocks)
        if spec.block_paths and spec.block_paths != normalized_legacy_blocks:
            raise ValueError(
                f"Offload component {spec.component_id!r} declares conflicting block paths: "
                f"{spec.block_paths!r} and {normalized_legacy_blocks!r}"
            )
        if not spec.block_paths:
            spec = replace(spec, block_paths=normalized_legacy_blocks)
        block_source = OffloadPlanSource.EXPLICIT

    legacy_tables = _legacy_host_tables(declaration, spec.module_path)
    if legacy_tables:
        if spec.host_resident_tables and spec.host_resident_tables != legacy_tables:
            raise ValueError(f"Offload component {spec.component_id!r} declares conflicting host-resident tables")
        if not spec.host_resident_tables:
            spec = replace(spec, host_resident_tables=legacy_tables)

    if spec.module_path in declaration.on_demand_component_paths and not spec.on_demand:
        spec = replace(spec, on_demand=True)
    if spec.module_path in declaration.resident_dit_paths and not spec.resident_prefix_eligible:
        spec = replace(spec, resident_prefix_eligible=True)
    if (
        spec.module_path in declaration.encoder_dlo_weight_replication
        and spec.weight_layout is OffloadWeightLayout.UNKNOWN
    ):
        spec = replace(spec, weight_layout=OffloadWeightLayout.REPLICATED)
    return spec, block_source


def _add_draft(
    drafts_by_id: dict[str, _ComponentDraft],
    ids_by_path: dict[str, str],
    spec: OffloadComponentSpec,
    source: OffloadPlanSource,
) -> _ComponentDraft:
    component_id = _require_name(spec.component_id, "Offload component id")
    component_type = _require_name(spec.component_type, f"Offload component {component_id!r} type")
    module_path = _require_name(spec.module_path, f"Offload component {component_id!r} module path")
    if len(set(spec.block_paths)) != len(spec.block_paths):
        raise ValueError(f"Offload component {component_id!r} declares duplicate block paths")

    existing = drafts_by_id.get(component_id)
    if existing is not None:
        if existing.spec != spec:
            raise ValueError(f"Conflicting declarations for offload component id {component_id!r}")
        return existing

    existing_id = ids_by_path.get(module_path)
    if existing_id is not None:
        existing = drafts_by_id[existing_id]
        if existing.spec.component_type != component_type:
            raise ValueError(
                f"Offload module path {module_path!r} is declared as both "
                f"{existing.spec.component_type!r} and {component_type!r}"
            )
        return existing

    normalized_spec = replace(
        spec,
        component_id=component_id,
        component_type=component_type,
        module_path=module_path,
        block_paths=tuple(spec.block_paths),
        host_resident_tables=tuple(spec.host_resident_tables),
        weight_layout=OffloadWeightLayout(spec.weight_layout),
    )
    draft = _ComponentDraft(spec=normalized_spec, source=source)
    drafts_by_id[component_id] = draft
    ids_by_path[module_path] = component_id
    return draft


def _add_discovered_component(
    drafts_by_id: dict[str, _ComponentDraft],
    ids_by_path: dict[str, str],
    declaration: OffloadPlan,
    *,
    path: str,
    component_type: str,
    source: OffloadPlanSource,
) -> None:
    existing_id = ids_by_path.get(path)
    if existing_id is not None:
        existing = drafts_by_id[existing_id]
        compatible_types = {existing.spec.component_type, component_type}
        if compatible_types <= {_ENCODER_COMPONENT, TEXT_ENCODER_COMPONENT}:
            return
        if existing.spec.component_type != component_type:
            raise ValueError(
                f"Explicit offload component {existing.spec.component_id!r} has type "
                f"{existing.spec.component_type!r}, but component discovery classifies {path!r} "
                f"as {component_type!r}"
            )
        return

    spec = OffloadComponentSpec(
        component_id=path,
        component_type=component_type,
        module_path=path,
    )
    spec, block_source = _merge_compatibility_fields(spec, declaration)
    draft = _add_draft(drafts_by_id, ids_by_path, spec, source)
    draft.block_source = block_source


def _add_compatibility_plan_components(
    drafts_by_id: dict[str, _ComponentDraft],
    ids_by_path: dict[str, str],
    declaration: OffloadPlan,
) -> None:
    """Treat path-keyed pre-J1 plan fields as explicit component producers."""
    dit_paths = _ordered_unique(
        (
            *declaration.block_attrs,
            *sorted(declaration.resident_dit_paths),
        )
    )
    for path in dit_paths:
        spec, block_source = _merge_compatibility_fields(
            OffloadComponentSpec(
                component_id=path,
                component_type=DIT_COMPONENT,
                module_path=path,
            ),
            declaration,
        )
        draft = _add_draft(
            drafts_by_id,
            ids_by_path,
            spec,
            OffloadPlanSource.EXPLICIT,
        )
        draft.block_source = block_source

    encoder_paths = _ordered_unique(
        (
            *declaration.encoder_component_types,
            *declaration.encoder_block_attrs,
            *sorted(declaration.encoder_dlo_weight_replication),
            *declaration.encoder_host_resident_table_attrs,
        )
    )
    for path in encoder_paths:
        spec, block_source = _merge_compatibility_fields(
            OffloadComponentSpec(
                component_id=path,
                component_type=_infer_encoder_type(path, declaration),
                module_path=path,
            ),
            declaration,
        )
        draft = _add_draft(
            drafts_by_id,
            ids_by_path,
            spec,
            OffloadPlanSource.EXPLICIT,
        )
        draft.block_source = block_source


def _collect_drafts(
    pipeline: nn.Module,
    declaration: OffloadPlan,
) -> tuple[list[_ComponentDraft], tuple[str, ...], bool]:
    drafts_by_id: dict[str, _ComponentDraft] = {}
    ids_by_path: dict[str, str] = {}

    for declared_spec in declaration.components:
        spec, block_source = _merge_compatibility_fields(declared_spec, declaration)
        draft = _add_draft(
            drafts_by_id,
            ids_by_path,
            spec,
            OffloadPlanSource.EXPLICIT,
        )
        draft.block_source = block_source or (OffloadPlanSource.EXPLICIT if spec.block_paths else None)

    uses_component_protocol = isinstance(pipeline, SupportsComponentDiscovery)
    uses_legacy_discovery = not uses_component_protocol and not declaration.components
    if uses_component_protocol or uses_legacy_discovery:
        modules = ModuleDiscovery.discover(pipeline)
    else:
        # A universal component declaration is complete by contract and does
        # not need well-known-name discovery merely because the pipeline does
        # not also implement SupportsComponentDiscovery.
        modules = PipelineModules(
            dits=[],
            dit_names=[],
            encoders=[],
            encoder_names=[],
            vaes=[],
        )
    component_source = (
        OffloadPlanSource.COMPONENT_PROTOCOL if uses_component_protocol else OffloadPlanSource.LEGACY_DISCOVERY
    )
    if uses_legacy_discovery:
        logger.warning(
            "Pipeline %s uses deprecated offload component-name discovery; "
            "declare SupportsComponentDiscovery or OffloadPlan.components",
            pipeline.__class__.__name__,
        )

    for path in modules.dit_names:
        _add_discovered_component(
            drafts_by_id,
            ids_by_path,
            declaration,
            path=path,
            component_type=DIT_COMPONENT,
            source=component_source,
        )
    for path in modules.encoder_names:
        _add_discovered_component(
            drafts_by_id,
            ids_by_path,
            declaration,
            path=path,
            component_type=_infer_encoder_type(path, declaration),
            source=component_source,
        )
    for path in modules.vae_names:
        _add_discovered_component(
            drafts_by_id,
            ids_by_path,
            declaration,
            path=path,
            component_type=_VAE_COMPONENT,
            source=component_source,
        )

    # Add explicit compatibility paths that were not already supplied by the
    # component protocol. Running this after discovery preserves the existing
    # component order for migrated pipelines.
    _add_compatibility_plan_components(drafts_by_id, ids_by_path, declaration)

    unknown_on_demand_paths = sorted(declaration.on_demand_component_paths - frozenset(ids_by_path))
    if unknown_on_demand_paths:
        raise ValueError(
            "OffloadPlan.on_demand_component_paths references paths that are not declared "
            f"as pipeline components: {', '.join(unknown_on_demand_paths)}"
        )

    resident_paths = _ordered_unique((*declaration.resident_paths, *modules.resident_names))
    return list(drafts_by_id.values()), resident_paths, uses_legacy_discovery


def _coerce_blocks(value: object, path: str) -> tuple[nn.Module, ...]:
    if isinstance(value, nn.ModuleList | nn.Sequential):
        blocks = tuple(value)
    elif isinstance(value, nn.Module):
        blocks = (value,)
    else:
        try:
            blocks = tuple(value)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError(f"Offload block path {path!r} is not an iterable of modules") from exc
    if not blocks:
        raise ValueError(f"Offload block path {path!r} resolves to an empty block group")
    invalid = next((block for block in blocks if not isinstance(block, nn.Module)), None)
    if invalid is not None:
        raise ValueError(f"Offload block path {path!r} contains {type(invalid).__name__}, expected nn.Module")
    return blocks


def _resolve_block_group(
    component: nn.Module,
    component_path: str,
    block_path: str,
    source: OffloadPlanSource,
) -> ResolvedBlockGroup:
    _require_name(block_path, f"Block path for component {component_path!r}")
    full_path = _join_path(component_path, block_path)
    try:
        value = attrgetter(block_path)(component)
    except AttributeError as exc:
        raise ValueError(f"Offload block path {full_path!r} does not exist") from exc
    return ResolvedBlockGroup(
        path=full_path,
        blocks=_coerce_blocks(value, full_path),
        source=source,
    )


def _resolve_host_table(
    component: nn.Module,
    component_path: str,
    spec: HostResidentTableSpec,
) -> ResolvedHostResidentTable:
    forward_relative_path = _require_name(
        spec.forward_module_path,
        f"Host-resident table path for component {component_path!r}",
    )
    forward_path = _join_path(component_path, forward_relative_path)
    forward_module = _resolve_module(component, forward_relative_path, "Host-resident table")

    if len(set(spec.tied_alias_paths)) != len(spec.tied_alias_paths):
        raise ValueError(f"Host-resident table {forward_path!r} declares duplicate tied aliases")
    alias_paths: list[str] = []
    alias_modules: list[nn.Module] = []
    forward_weight = getattr(forward_module, "weight", None)
    for relative_alias_path in spec.tied_alias_paths:
        _require_name(relative_alias_path, f"Tied alias for host-resident table {forward_path!r}")
        if relative_alias_path == forward_relative_path:
            raise ValueError(f"Host-resident table {forward_path!r} lists itself as a tied alias")
        alias_path = _join_path(component_path, relative_alias_path)
        alias_module = _resolve_module(component, relative_alias_path, "Host-resident table alias")
        alias_weight = getattr(alias_module, "weight", None)
        if forward_weight is None or alias_weight is None or alias_weight is not forward_weight:
            raise ValueError(
                f"Host-resident table alias {alias_path!r} does not share the exact weight "
                f"parameter with executed module {forward_path!r}"
            )
        alias_paths.append(alias_path)
        alias_modules.append(alias_module)

    if forward_weight is not None:
        actual_tied_paths = {
            path
            for path, module in component.named_modules()
            if path and getattr(module, "weight", None) is forward_weight
        }
        declared_tied_paths = {forward_relative_path, *spec.tied_alias_paths}
        missing_tied_paths = sorted(actual_tied_paths - declared_tied_paths)
        if missing_tied_paths:
            raise ValueError(
                f"Host-resident table {forward_path!r} has undeclared tied module aliases: "
                f"{', '.join(_join_path(component_path, path) for path in missing_tied_paths)}"
            )
    return ResolvedHostResidentTable(
        forward_module_path=forward_path,
        forward_module=forward_module,
        tied_alias_paths=tuple(alias_paths),
        tied_alias_modules=tuple(alias_modules),
    )


def _resolve_components(
    pipeline: nn.Module,
    drafts: list[_ComponentDraft],
    declaration: OffloadPlan,
) -> tuple[ResolvedOffloadComponent, ...]:
    resolved: list[ResolvedOffloadComponent] = []
    modules_by_id: dict[int, str] = {}

    for draft in drafts:
        spec = draft.spec
        module = _resolve_module(
            pipeline,
            spec.module_path,
            f"Offload component {spec.component_id!r}",
        )
        existing_path = modules_by_id.get(id(module))
        if existing_path is not None:
            raise ValueError(
                f"Offload component paths {existing_path!r} and {spec.module_path!r} "
                "resolve to the same module; declare one canonical path"
            )
        modules_by_id[id(module)] = spec.module_path

        block_paths = spec.block_paths
        block_source = draft.block_source
        if not block_paths:
            block_paths = tuple(get_blocks_attr_names(module))
            if block_paths:
                block_source = OffloadPlanSource.BLOCK_ATTRIBUTE
        groups = tuple(
            _resolve_block_group(
                module,
                spec.module_path,
                path,
                block_source or OffloadPlanSource.EXPLICIT,
            )
            for path in block_paths
        )
        tables = tuple(
            _resolve_host_table(module, spec.module_path, table_spec) for table_spec in spec.host_resident_tables
        )
        resolved.append(
            ResolvedOffloadComponent(
                component_id=spec.component_id,
                component_type=spec.component_type,
                module_path=spec.module_path,
                module=module,
                block_groups=groups,
                on_demand=spec.on_demand,
                weight_layout=spec.weight_layout,
                resident_prefix_eligible=spec.resident_prefix_eligible,
                host_resident_tables=tables,
                source=draft.source,
            )
        )

    resolved = _add_legacy_submodule_groups(resolved, declaration)
    _validate_unique_block_ownership(resolved)
    return tuple(resolved)


def _add_legacy_submodule_groups(
    components: list[ResolvedOffloadComponent],
    declaration: OffloadPlan,
) -> list[ResolvedOffloadComponent]:
    if not declaration.offload_submodules:
        return components

    component_module_ids = {id(component.module) for component in components}
    matched_names: set[str] = set()
    updated: list[ResolvedOffloadComponent] = []
    for component in components:
        groups = list(component.block_groups)
        if component.component_type == DIT_COMPONENT:
            for child_name, child in component.module.named_children():
                block_path = declaration.offload_submodules.get(child_name)
                if block_path is None:
                    continue
                matched_names.add(child_name)
                if id(child) in component_module_ids:
                    continue
                groups.append(
                    _resolve_block_group(
                        child,
                        _join_path(component.module_path, child_name),
                        block_path,
                        OffloadPlanSource.EXPLICIT,
                    )
                )
        updated.append(replace(component, block_groups=tuple(groups)))

    missing_names = sorted(set(declaration.offload_submodules) - matched_names)
    if missing_names:
        raise ValueError(
            f"OffloadPlan.offload_submodules entries do not match a direct child of any DiT: {', '.join(missing_names)}"
        )
    return updated


def _validate_unique_block_ownership(components: list[ResolvedOffloadComponent]) -> None:
    owners: dict[int, str] = {}
    for component in components:
        for group in component.block_groups:
            for block in group.blocks:
                existing = owners.get(id(block))
                if existing is not None:
                    raise ValueError(f"Offload block module is owned by both {existing!r} and {group.path!r}")
                owners[id(block)] = group.path


def _resolve_legacy_selection_overrides(
    declaration: OffloadPlan,
    component_ids: frozenset[str],
) -> dict[str, frozenset[str]]:
    overrides: dict[str, frozenset[str]] = {}
    for selection in declaration.legacy_selections:
        try:
            policy = OffloadStrategy(selection.policy).value
        except ValueError as exc:
            choices = ", ".join(policy.value for policy in OffloadStrategy)
            raise ValueError(
                f"Unknown legacy offload selection policy {selection.policy!r}; choose from: {choices}"
            ) from exc
        if policy in overrides:
            raise ValueError(f"OffloadPlan declares duplicate legacy selection for policy {policy!r}")
        selected = frozenset(selection.component_ids)
        unknown = sorted(selected - component_ids)
        if unknown:
            raise ValueError(
                f"Legacy selection for policy {policy!r} references unknown component ids: {', '.join(unknown)}"
            )
        overrides[policy] = selected
    return overrides


def _resolve_selection(
    components: tuple[ResolvedOffloadComponent, ...],
    declaration: OffloadPlan,
    config: OffloadConfig,
) -> tuple[OffloadSelectionMode, frozenset[str]]:
    all_ids = frozenset(component.component_id for component in components)
    if config.components_explicit:
        selected = frozenset(
            component.component_id for component in components if component.component_type in config.components
        )
        missing_types = sorted(
            component_type
            for component_type in config.components
            if not any(component.component_type == component_type for component in components)
        )
        if missing_types:
            raise ValueError(
                f"Selected offload component type(s) are not declared by this model: {', '.join(missing_types)}"
            )
        return OffloadSelectionMode.EXPLICIT, selected

    policy = config.strategy.value
    overrides = _resolve_legacy_selection_overrides(declaration, all_ids)
    if policy in overrides:
        return OffloadSelectionMode.LEGACY_OMITTED, overrides[policy]

    if config.strategy is OffloadStrategy.MODEL_LEVEL:
        selected = frozenset(
            component.component_id
            for component in components
            if component.component_type in {DIT_COMPONENT, TEXT_ENCODER_COMPONENT, _ENCODER_COMPONENT}
        )
    elif config.strategy in {
        OffloadStrategy.LAYER_WISE,
        OffloadStrategy.DISTRIBUTED_LAYER_WISE,
    }:
        selected = frozenset(
            component.component_id
            for component in components
            if component.component_type == DIT_COMPONENT or component.on_demand
        )
    else:
        selected = frozenset()
    return OffloadSelectionMode.LEGACY_OMITTED, selected


def _resolve_phases(
    declaration: OffloadPlan,
    components: tuple[ResolvedOffloadComponent, ...],
) -> tuple[ResolvedOffloadPhase, ...]:
    components_by_id = {component.component_id: component for component in components}
    phases: list[ResolvedOffloadPhase] = []
    seen_names: set[str] = set()
    for phase in declaration.phases:
        name = _require_name(phase.name, "Offload phase name")
        if name in seen_names:
            raise ValueError(f"OffloadPlan declares duplicate phase {name!r}")
        seen_names.add(name)
        if not phase.component_ids:
            raise ValueError(f"Offload phase {name!r} must reference at least one component")
        if len(set(phase.component_ids)) != len(phase.component_ids):
            raise ValueError(f"Offload phase {name!r} contains duplicate component ids")
        unknown = [component_id for component_id in phase.component_ids if component_id not in components_by_id]
        if unknown:
            raise ValueError(f"Offload phase {name!r} references unknown component ids: {', '.join(unknown)}")
        phases.append(
            ResolvedOffloadPhase(
                name=name,
                component_ids=tuple(phase.component_ids),
                components=tuple(components_by_id[component_id] for component_id in phase.component_ids),
            )
        )
    return tuple(phases)


def _resolve_residents(
    pipeline: nn.Module,
    resident_paths: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[nn.Module, ...]]:
    paths: list[str] = []
    modules: list[nn.Module] = []
    seen_modules: set[int] = set()
    for path in resident_paths:
        module = _resolve_module(pipeline, path, "Offload resident path")
        if id(module) in seen_modules:
            continue
        seen_modules.add(id(module))
        paths.append(path)
        modules.append(module)
    return tuple(paths), tuple(modules)


def _validate_selected_topology(
    plan: ResolvedOffloadPlan,
    config: OffloadConfig,
) -> None:
    if plan.selection_mode is OffloadSelectionMode.EXPLICIT and config.strategy in {
        OffloadStrategy.LAYER_WISE,
        OffloadStrategy.DISTRIBUTED_LAYER_WISE,
    }:
        for component_type in config.components:
            selected = plan.components_for(component_type, selected_only=True)
            if not any(component.block_groups or component.on_demand for component in selected):
                raise ValueError(
                    f"Selected component {component_type!r} has no streamable blocks or on-demand lifecycle"
                )

    if config.strategy in {
        OffloadStrategy.LAYER_WISE,
        OffloadStrategy.DISTRIBUTED_LAYER_WISE,
    }:
        for component in plan.selected_components:
            if not component.on_demand:
                continue
            if not callable(getattr(component.module, "load_to_device", None)) or not callable(
                getattr(component.module, "offload_to_cpu", None)
            ):
                raise ValueError(
                    f"Component {component.module_path!r} declares on-demand offload but must implement "
                    "load_to_device() and offload_to_cpu()"
                )

    if config.strategy is OffloadStrategy.DISTRIBUTED_LAYER_WISE:
        for component in plan.components_for(TEXT_ENCODER_COMPONENT, selected_only=True):
            if (
                component.block_groups
                and config.uses_allgather(TEXT_ENCODER_COMPONENT)
                and component.weight_layout is not OffloadWeightLayout.REPLICATED
            ):
                raise ValueError(
                    f"Text encoder {component.module_path!r} cannot use DLO AllGather: "
                    "its loader-produced weights are not declared replicated"
                )
        if config.dlo_resident_layers and not any(
            component.resident_prefix_eligible for component in plan.components_for(DIT_COMPONENT, selected_only=True)
        ):
            raise ValueError(
                "dlo_resident_layers was requested, but the resolved plan has no selected "
                "DiT target eligible for resident leading blocks"
            )


def resolve_offload_plan(
    pipeline: nn.Module,
    config: OffloadConfig,
) -> ResolvedOffloadPlan:
    """Resolve and validate one pipeline's topology without mutating it.

    User-only configuration validation belongs to :class:`OffloadConfig` and
    runs before model loading. This resolver owns model-dependent validation
    and must complete before a backend installs hooks, moves tensors, or takes
    ownership of loader resources.
    """
    declaration = get_offload_plan(pipeline) or OffloadPlan()
    if not isinstance(declaration, OffloadPlan):
        raise TypeError(
            f"{pipeline.__class__.__name__}._offload_plan must be OffloadPlan, got {type(declaration).__name__}"
        )

    drafts, resident_paths, uses_legacy_discovery = _collect_drafts(pipeline, declaration)
    components = _resolve_components(pipeline, drafts, declaration)
    selection_mode, selected_component_ids = _resolve_selection(
        components,
        declaration,
        config,
    )
    phases = _resolve_phases(declaration, components)
    resolved_resident_paths, resident_modules = _resolve_residents(pipeline, resident_paths)
    resolved = ResolvedOffloadPlan(
        policy=config.strategy.value,
        selection_mode=selection_mode,
        components=components,
        selected_component_ids=selected_component_ids,
        phases=phases,
        resident_paths=resolved_resident_paths,
        resident_modules=resident_modules,
        uses_legacy_discovery=uses_legacy_discovery,
    )
    _validate_selected_topology(resolved, config)
    return resolved


__all__ = ["resolve_offload_plan"]
