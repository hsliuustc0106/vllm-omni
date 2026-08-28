# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU-only contract tests for the central offload-plan resolver."""

from __future__ import annotations

from typing import ClassVar

import pytest
import torch
from pytest_mock import MockerFixture
from torch import nn

import vllm_omni.diffusion.offloader as offloader_module
import vllm_omni.diffusion.offloader.plan_resolver as plan_resolver_module
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.offloader import (
    HostResidentTableSpec,
    OffloadBackend,
    OffloadComponentSpec,
    OffloadConfig,
    OffloadLegacySelection,
    OffloadPhaseSpec,
    OffloadPlan,
    OffloadPlanSource,
    OffloadSelectionMode,
    OffloadStrategy,
    OffloadWeightLayout,
    resolve_offload_plan,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


class _BlockStack(nn.Module):
    _layerwise_offload_blocks_attrs = ["blocks"]

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])


class _TextEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.block = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])


class _StagedModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(2, 2)

    def load_to_device(self) -> None:
        pass

    def offload_to_cpu(self) -> None:
        pass


class _ProtocolPipeline(nn.Module, SupportsComponentDiscovery):
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]
    _resident_modules: ClassVar[list[str]] = ["conditioner"]
    _offload_plan = OffloadPlan(
        encoder_component_types={"text_encoder": "text_encoder"},
        encoder_block_attrs={"text_encoder": ("encoder.block",)},
    )

    def __init__(self) -> None:
        super().__init__()
        self.transformer = _BlockStack()
        self.text_encoder = _TextEncoder()
        self.vae = nn.Linear(2, 2)
        self.conditioner = nn.Linear(2, 2)


def _config(
    strategy: OffloadStrategy = OffloadStrategy.LAYER_WISE,
    *,
    components: frozenset[str] = frozenset({"dit"}),
    explicit: bool = False,
    use_allgather: bool = True,
    resident_layers: int = 0,
) -> OffloadConfig:
    return OffloadConfig(
        strategy=strategy,
        pin_cpu_memory=False,
        components=components,
        components_explicit=explicit,
        dlo_use_allgather=use_allgather,
        dlo_resident_layers=resident_layers,
    )


def test_protocol_and_block_attribute_producers_share_one_plan() -> None:
    pipeline = _ProtocolPipeline()

    plan = resolve_offload_plan(pipeline, _config())

    assert not plan.uses_legacy_discovery
    assert plan.selection_mode is OffloadSelectionMode.LEGACY_OMITTED
    assert [component.module_path for component in plan.components] == [
        "transformer",
        "text_encoder",
        "vae",
    ]
    assert [component.module_path for component in plan.selected_components] == ["transformer"]
    transformer = plan.component_by_id("transformer")
    assert transformer.source is OffloadPlanSource.COMPONENT_PROTOCOL
    assert transformer.block_groups[0].source is OffloadPlanSource.BLOCK_ATTRIBUTE
    assert transformer.block_groups[0].blocks == tuple(pipeline.transformer.blocks)
    encoder = plan.component_by_id("text_encoder")
    assert encoder.block_groups[0].source is OffloadPlanSource.EXPLICIT
    assert plan.resident_paths == ("conditioner",)
    assert plan.resident_modules == (pipeline.conditioner,)


def test_fallback_component_scan_warns_and_is_visible_in_provenance(
    mocker: MockerFixture,
) -> None:
    class Pipeline(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer = _BlockStack()

    warning = mocker.spy(plan_resolver_module.logger, "warning")
    plan = resolve_offload_plan(Pipeline(), _config())

    assert plan.uses_legacy_discovery
    assert plan.component_by_id("transformer").source is OffloadPlanSource.LEGACY_DISCOVERY
    warning.assert_called_once()
    assert "uses deprecated offload component-name discovery" in warning.call_args.args[0]


def test_path_keyed_compatibility_plan_produces_components_and_nested_blocks() -> None:
    class Transformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
            self.token_refiner = nn.Module()
            self.token_refiner.blocks = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])

    class Pipeline(nn.Module):
        _offload_plan = OffloadPlan(
            block_attrs={"denoiser": ("blocks",)},
            offload_submodules={"token_refiner": "blocks"},
        )

        def __init__(self) -> None:
            super().__init__()
            self.denoiser = Transformer()

    plan = resolve_offload_plan(Pipeline(), _config())
    transformer = plan.component_by_id("denoiser")

    assert transformer.source is OffloadPlanSource.EXPLICIT
    assert [group.path for group in transformer.block_groups] == [
        "denoiser.blocks",
        "denoiser.token_refiner.blocks",
    ]


def test_unknown_on_demand_compatibility_path_is_rejected() -> None:
    class Pipeline(nn.Module):
        _offload_plan = OffloadPlan(
            on_demand_component_paths=frozenset({"missing"}),
        )

    with pytest.raises(ValueError, match="not declared as pipeline components: missing"):
        resolve_offload_plan(Pipeline(), _config())


def test_explicit_plan_resolves_phases_residents_and_executed_table_aliases() -> None:
    class TiedEncoder(_TextEncoder):
        def __init__(self) -> None:
            super().__init__()
            self.shared = nn.Embedding(8, 2)
            self.encoder.embed_tokens = nn.Embedding(8, 2)
            self.encoder.embed_tokens.weight = self.shared.weight

    class Pipeline(nn.Module):
        _offload_plan = OffloadPlan(
            components=(
                OffloadComponentSpec(
                    component_id="reasoner",
                    component_type="dit",
                    module_path="reasoner",
                    block_paths=("blocks",),
                    resident_prefix_eligible=True,
                ),
                OffloadComponentSpec(
                    component_id="generator",
                    component_type="dit",
                    module_path="generator",
                    block_paths=("blocks",),
                ),
                OffloadComponentSpec(
                    component_id="prompt",
                    component_type="text_encoder",
                    module_path="text_encoder",
                    block_paths=("encoder.block",),
                    weight_layout=OffloadWeightLayout.REPLICATED,
                    host_resident_tables=(
                        HostResidentTableSpec(
                            forward_module_path="encoder.embed_tokens",
                            tied_alias_paths=("shared",),
                        ),
                    ),
                ),
            ),
            phases=(
                OffloadPhaseSpec(name="reasoner", component_ids=("reasoner",)),
                OffloadPhaseSpec(name="generator", component_ids=("generator",)),
            ),
            resident_paths=("conditioner",),
        )

        def __init__(self) -> None:
            super().__init__()
            self.reasoner = _BlockStack()
            self.generator = _BlockStack()
            self.text_encoder = TiedEncoder()
            self.conditioner = nn.Linear(2, 2)

    pipeline = Pipeline()
    original_weight = pipeline.text_encoder.shared.weight

    plan = resolve_offload_plan(
        pipeline,
        _config(
            OffloadStrategy.DISTRIBUTED_LAYER_WISE,
            components=frozenset({"dit", "text_encoder"}),
            explicit=True,
            use_allgather=True,
        ),
    )

    assert not plan.uses_legacy_discovery
    assert [phase.name for phase in plan.phases] == ["reasoner", "generator"]
    assert plan.phases[0].components == (plan.component_by_id("reasoner"),)
    prompt = plan.component_by_id("prompt")
    table = prompt.host_resident_tables[0]
    assert table.forward_module is pipeline.text_encoder.encoder.embed_tokens
    assert table.tied_alias_modules == (pipeline.text_encoder.shared,)
    assert table.forward_module.weight is original_weight
    assert pipeline.text_encoder.shared.weight is original_weight
    assert not hasattr(pipeline.reasoner.blocks[0], "_hook_registry")


def test_host_resident_table_rejects_undeclared_tied_module_alias() -> None:
    class Encoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared = nn.Embedding(8, 2)
            self.encoder = nn.Module()
            self.encoder.embed_tokens = nn.Embedding(8, 2)
            self.encoder.embed_tokens.weight = self.shared.weight

    class Pipeline(nn.Module):
        _offload_plan = OffloadPlan(
            components=(
                OffloadComponentSpec(
                    component_id="prompt",
                    component_type="text_encoder",
                    module_path="text_encoder",
                    host_resident_tables=(HostResidentTableSpec(forward_module_path="shared"),),
                ),
            )
        )

        def __init__(self) -> None:
            super().__init__()
            self.text_encoder = Encoder()

    with pytest.raises(ValueError, match="undeclared tied module aliases.*encoder.embed_tokens"):
        resolve_offload_plan(
            Pipeline(),
            _config(
                OffloadStrategy.MODEL_LEVEL,
                components=frozenset({"text_encoder"}),
                explicit=True,
            ),
        )


def test_conflicting_explicit_and_component_protocol_types_fail() -> None:
    class Pipeline(nn.Module, SupportsComponentDiscovery):
        _dit_modules: ClassVar[list[str]] = ["transformer"]
        _encoder_modules: ClassVar[list[str]] = []
        _vae_modules: ClassVar[list[str]] = []
        _offload_plan = OffloadPlan(
            components=(
                OffloadComponentSpec(
                    component_id="wrong",
                    component_type="vae",
                    module_path="transformer",
                ),
            )
        )

        def __init__(self) -> None:
            super().__init__()
            self.transformer = _BlockStack()

    with pytest.raises(ValueError, match="component discovery classifies.*as 'dit'"):
        resolve_offload_plan(Pipeline(), _config())


def test_explicit_selection_fails_when_model_does_not_declare_component() -> None:
    class Pipeline(nn.Module, SupportsComponentDiscovery):
        _dit_modules: ClassVar[list[str]] = ["transformer"]
        _encoder_modules: ClassVar[list[str]] = []
        _vae_modules: ClassVar[list[str]] = []

        def __init__(self) -> None:
            super().__init__()
            self.transformer = _BlockStack()

    with pytest.raises(ValueError, match="not declared by this model: text_encoder"):
        resolve_offload_plan(
            Pipeline(),
            _config(components=frozenset({"text_encoder"}), explicit=True),
        )


def test_explicit_layerwise_selection_requires_supported_lifecycle() -> None:
    class Pipeline(nn.Module, SupportsComponentDiscovery):
        _dit_modules: ClassVar[list[str]] = []
        _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
        _vae_modules: ClassVar[list[str]] = []

        def __init__(self) -> None:
            super().__init__()
            self.text_encoder = nn.Linear(2, 2)

    with pytest.raises(ValueError, match="no streamable blocks or on-demand lifecycle"):
        resolve_offload_plan(
            Pipeline(),
            _config(components=frozenset({"text_encoder"}), explicit=True),
        )


def test_on_demand_contract_fails_before_mutation() -> None:
    class IncompleteStage(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))

        def offload_to_cpu(self) -> None:
            pass

    class Pipeline(nn.Module, SupportsComponentDiscovery):
        _dit_modules: ClassVar[list[str]] = []
        _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
        _vae_modules: ClassVar[list[str]] = []
        _offload_plan = OffloadPlan(
            components=(
                OffloadComponentSpec(
                    component_id="prompt",
                    component_type="text_encoder",
                    module_path="text_encoder",
                    on_demand=True,
                ),
            )
        )

        def __init__(self) -> None:
            super().__init__()
            self.text_encoder = IncompleteStage()

    pipeline = Pipeline()
    original_weight = pipeline.text_encoder.weight

    with pytest.raises(ValueError, match=r"must implement load_to_device\(\) and offload_to_cpu\(\)"):
        resolve_offload_plan(
            pipeline,
            _config(components=frozenset({"text_encoder"}), explicit=True),
        )

    assert pipeline.text_encoder.weight is original_weight
    assert pipeline.text_encoder.weight.device.type == "cpu"


def test_omission_can_preserve_private_legacy_lifecycle() -> None:
    class Pipeline(nn.Module, SupportsComponentDiscovery):
        _dit_modules: ClassVar[list[str]] = ["transformer"]
        _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
        _vae_modules: ClassVar[list[str]] = ["vae"]
        _offload_plan = OffloadPlan(
            on_demand_component_paths=frozenset({"text_encoder", "vae"}),
        )

        def __init__(self) -> None:
            super().__init__()
            self.transformer = _BlockStack()
            self.text_encoder = _StagedModule()
            self.vae = _StagedModule()

    pipeline = Pipeline()
    omitted = resolve_offload_plan(pipeline, _config())
    explicit = resolve_offload_plan(
        pipeline,
        _config(components=frozenset({"dit"}), explicit=True),
    )

    assert omitted.selected_component_ids == frozenset({"transformer", "text_encoder", "vae"})
    assert explicit.selected_component_ids == frozenset({"transformer"})


def test_policy_specific_legacy_selection_overrides_generic_default() -> None:
    class Pipeline(nn.Module, SupportsComponentDiscovery):
        _dit_modules: ClassVar[list[str]] = []
        _encoder_modules: ClassVar[list[str]] = []
        _vae_modules: ClassVar[list[str]] = []
        _offload_plan = OffloadPlan(
            components=(
                OffloadComponentSpec(
                    component_id="transformer",
                    component_type="dit",
                    module_path="transformer",
                    block_paths=("blocks",),
                ),
                OffloadComponentSpec(
                    component_id="prompt",
                    component_type="text_encoder",
                    module_path="text_encoder",
                    on_demand=True,
                ),
            ),
            legacy_selections=(OffloadLegacySelection(policy="layerwise", component_ids=("prompt",)),),
        )

        def __init__(self) -> None:
            super().__init__()
            self.transformer = _BlockStack()
            self.text_encoder = _StagedModule()

    plan = resolve_offload_plan(Pipeline(), _config())

    assert plan.selected_component_ids == frozenset({"prompt"})


def test_dlo_encoder_allgather_requires_replicated_weight_capability() -> None:
    pipeline = _ProtocolPipeline()
    config = _config(
        OffloadStrategy.DISTRIBUTED_LAYER_WISE,
        components=frozenset({"text_encoder"}),
        explicit=True,
        use_allgather=True,
    )

    with pytest.raises(ValueError, match="cannot use DLO AllGather"):
        resolve_offload_plan(pipeline, config)

    rank_local = resolve_offload_plan(
        pipeline,
        _config(
            OffloadStrategy.DISTRIBUTED_LAYER_WISE,
            components=frozenset({"text_encoder"}),
            explicit=True,
            use_allgather=False,
        ),
    )
    assert rank_local.selected_component_ids == frozenset({"text_encoder"})


def test_duplicate_block_ownership_is_rejected() -> None:
    class Pipeline(nn.Module, SupportsComponentDiscovery):
        _dit_modules: ClassVar[list[str]] = []
        _encoder_modules: ClassVar[list[str]] = []
        _vae_modules: ClassVar[list[str]] = []
        _offload_plan = OffloadPlan(
            components=(
                OffloadComponentSpec("first", "dit", "first", block_paths=("blocks",)),
                OffloadComponentSpec("second", "dit", "second", block_paths=("blocks",)),
            )
        )

        def __init__(self) -> None:
            super().__init__()
            shared = nn.Linear(2, 2)
            self.first = nn.Module()
            self.first.blocks = nn.ModuleList([shared])
            self.second = nn.Module()
            self.second.blocks = nn.ModuleList([shared])

    with pytest.raises(ValueError, match="owned by both"):
        resolve_offload_plan(Pipeline(), _config())


def test_phase_references_must_resolve() -> None:
    class Pipeline(nn.Module, SupportsComponentDiscovery):
        _dit_modules: ClassVar[list[str]] = []
        _encoder_modules: ClassVar[list[str]] = []
        _vae_modules: ClassVar[list[str]] = []
        _offload_plan = OffloadPlan(
            phases=(OffloadPhaseSpec(name="reasoner", component_ids=("missing",)),),
        )

    with pytest.raises(ValueError, match="references unknown component ids: missing"):
        resolve_offload_plan(Pipeline(), _config())


def test_central_enable_boundary_attaches_plan_before_backend_mutation(mocker: MockerFixture) -> None:
    pipeline = _ProtocolPipeline()

    class FakeBackend(OffloadBackend):
        def __init__(self) -> None:
            super().__init__(_config(), torch.device("cpu"))
            self.enabled_with = None

        def enable(self, model: nn.Module) -> None:
            assert self.resolved_plan is not None
            self.enabled_with = model

        def disable(self) -> None:
            pass

    backend = FakeBackend()
    mocker.patch.object(offloader_module, "get_offload_backend", return_value=backend)

    resolved_pipeline, resolved_backend = offloader_module.enable_offload_backend(
        mocker.Mock(spec=OmniDiffusionConfig),
        pipeline,
        device=torch.device("cpu"),
    )

    assert resolved_pipeline is pipeline
    assert resolved_backend is backend
    assert backend.enabled_with is pipeline
    assert backend.resolved_plan.component_by_id("transformer").module is pipeline.transformer
