# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Declarative and resolved CPU-offload topology.

Models may declare an :class:`OffloadPlan` as ``_offload_plan`` on the
pipeline. :func:`resolve_offload_plan` (implemented in ``plan_resolver.py``)
combines that declaration with the component-discovery compatibility
protocols and returns one immutable :class:`ResolvedOffloadPlan` per pipeline
instance.

The declaration contains paths and policy-neutral capabilities. The resolved
plan binds those paths to concrete modules, records the active compatibility
selection, and is the artifact that offload backends will consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from torch import nn


class OffloadWeightLayout(str, Enum):
    """Loader-produced weight layout relevant to offload transports."""

    UNKNOWN = "unknown"
    REPLICATED = "replicated"
    RANK_LOCAL = "rank-local"


class OffloadPlanSource(str, Enum):
    """Origin of one normalized topology declaration."""

    EXPLICIT = "explicit"
    COMPONENT_PROTOCOL = "component-protocol"
    BLOCK_ATTRIBUTE = "block-attribute"
    LEGACY_DISCOVERY = "legacy-discovery"


class OffloadSelectionMode(str, Enum):
    """Whether component selection was explicit or compatibility-derived."""

    EXPLICIT = "explicit"
    LEGACY_OMITTED = "legacy-omitted"


@dataclass(frozen=True)
class HostResidentTableSpec:
    """A lookup module that may remain on the host during block streaming.

    Paths are relative to the owning component. ``forward_module_path`` must
    name the module whose ``forward`` is actually executed. If its weight is
    tied, every known module alias must be listed so the resolver can reject a
    declaration that hooks only the storage owner instead of the executed
    module.
    """

    forward_module_path: str
    tied_alias_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class OffloadComponentSpec:
    """Policy-neutral topology for one pipeline component."""

    component_id: str
    component_type: str
    module_path: str
    block_paths: tuple[str, ...] = ()
    on_demand: bool = False
    weight_layout: OffloadWeightLayout = OffloadWeightLayout.UNKNOWN
    resident_prefix_eligible: bool = False
    host_resident_tables: tuple[HostResidentTableSpec, ...] = ()


@dataclass(frozen=True)
class OffloadPhaseSpec:
    """Named execution phase and the components active in that phase."""

    name: str
    component_ids: tuple[str, ...]


@dataclass(frozen=True)
class OffloadLegacySelection:
    """Policy-specific targets used when ``offload_components`` is omitted."""

    policy: str
    component_ids: tuple[str, ...]


@dataclass(frozen=True)
class OffloadPlan:
    """Optional model declaration consumed by the central plan resolver.

    ``components``, ``phases``, ``resident_paths``, and
    ``legacy_selections`` are the universal declaration. The remaining fields
    are compatibility inputs used by models added before the universal plan;
    the resolver normalizes them into the same resolved representation.

    If not declared, the offloader falls back to:
    1. ``SupportsComponentDiscovery`` component paths.
    2. ``_layerwise_offload_blocks_attrs`` on component module classes.
    3. Legacy well-known component-name discovery, with a warning.

    Attributes:
        components: Universal component declarations. When non-empty on a
            pipeline that does not implement ``SupportsComponentDiscovery``,
            this tuple is treated as complete and the legacy component-name
            scan is not run.
        phases: Named execution phases referencing ``component_id`` values.
        resident_paths: Pipeline-relative modules that remain device-resident.
        legacy_selections: Policy-specific compatibility targets used only
            when the public component selector is omitted.
        block_attrs: Maps DiT path → tuple of block-list attribute names.
            e.g. ``{"transformer": ("gen_layers",),
                    "transformer.language_model": ("layers",)}``
        offload_submodules: Maps child name → block-list attribute name,
            for large non-DiT submodules within a DiT that should be
            independently offloaded with their own hooks.
            e.g. ``{"context_encoder": "layers"}``
        resident_dit_paths: DiT paths whose leading blocks may be kept on the
            device when ``dlo_resident_layers`` is nonzero. Keeping this
            model-declared avoids applying a consumer-GPU tuning knob to
            auxiliary or dual DiTs unintentionally.
        encoder_component_types: Maps encoder paths to public selector types
            (currently text_encoder). This declaration is used before the
            compatibility name heuristic.
        encoder_block_attrs: Maps encoder paths to streamable block-list paths.
        encoder_dlo_weight_replication: Encoder paths whose loader-produced
            block tensors are identical across the DiT DLO group. Only these
            encoders may use multi-rank AllGather transfer; this must not be
            declared for encoder-TP shards.
        encoder_host_resident_table_attrs: Compatibility form of host-resident
            table declarations. New declarations should use
            ``HostResidentTableSpec`` so tied aliases and the executed module
            are explicit.
    """

    components: tuple[OffloadComponentSpec, ...] = ()
    phases: tuple[OffloadPhaseSpec, ...] = ()
    resident_paths: tuple[str, ...] = ()
    legacy_selections: tuple[OffloadLegacySelection, ...] = ()

    on_demand_component_paths: frozenset[str] = field(default_factory=frozenset)

    block_attrs: dict[str, tuple[str, ...]] = field(default_factory=dict)
    offload_submodules: dict[str, str] = field(default_factory=dict)
    resident_dit_paths: frozenset[str] = field(default_factory=frozenset)
    encoder_component_types: dict[str, str] = field(default_factory=dict)
    encoder_block_attrs: dict[str, tuple[str, ...]] = field(default_factory=dict)
    encoder_dlo_weight_replication: frozenset[str] = field(default_factory=frozenset)
    encoder_host_resident_table_attrs: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedBlockGroup:
    """A block-list path bound to its concrete modules."""

    path: str
    blocks: tuple[nn.Module, ...]
    source: OffloadPlanSource


@dataclass(frozen=True)
class ResolvedHostResidentTable:
    """A host-resident table and all explicitly declared module aliases."""

    forward_module_path: str
    forward_module: nn.Module
    tied_alias_paths: tuple[str, ...]
    tied_alias_modules: tuple[nn.Module, ...]


@dataclass(frozen=True)
class ResolvedOffloadComponent:
    """One normalized component bound to a pipeline instance."""

    component_id: str
    component_type: str
    module_path: str
    module: nn.Module
    block_groups: tuple[ResolvedBlockGroup, ...]
    on_demand: bool
    weight_layout: OffloadWeightLayout
    resident_prefix_eligible: bool
    host_resident_tables: tuple[ResolvedHostResidentTable, ...]
    source: OffloadPlanSource


@dataclass(frozen=True)
class ResolvedOffloadPhase:
    """A phase whose component references have been validated and bound."""

    name: str
    component_ids: tuple[str, ...]
    components: tuple[ResolvedOffloadComponent, ...]


@dataclass(frozen=True)
class ResolvedOffloadPlan:
    """Immutable, read-only topology produced once per pipeline instance."""

    policy: str
    selection_mode: OffloadSelectionMode
    components: tuple[ResolvedOffloadComponent, ...]
    selected_component_ids: frozenset[str]
    phases: tuple[ResolvedOffloadPhase, ...]
    resident_paths: tuple[str, ...]
    resident_modules: tuple[nn.Module, ...]
    uses_legacy_discovery: bool = False

    @property
    def selected_components(self) -> tuple[ResolvedOffloadComponent, ...]:
        """Return selected components in deterministic declaration order."""
        return tuple(
            component for component in self.components if component.component_id in self.selected_component_ids
        )

    def components_for(
        self,
        component_type: str,
        *,
        selected_only: bool = False,
    ) -> tuple[ResolvedOffloadComponent, ...]:
        """Return components of one type in deterministic declaration order."""
        components = self.selected_components if selected_only else self.components
        return tuple(component for component in components if component.component_type == component_type)

    def component_by_id(self, component_id: str) -> ResolvedOffloadComponent:
        """Return one component or raise an actionable topology error."""
        for component in self.components:
            if component.component_id == component_id:
                return component
        raise KeyError(f"Unknown offload component id {component_id!r}")


def get_offload_plan(pipeline: nn.Module) -> OffloadPlan | None:
    """Retrieve the OffloadPlan declared by the pipeline, if any."""
    return getattr(pipeline, "_offload_plan", None)
