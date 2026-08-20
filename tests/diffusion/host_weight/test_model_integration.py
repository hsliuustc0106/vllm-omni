# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Conformance tests for the pluggable model-integration boundary."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.config.load import LoadConfig

import vllm_omni.diffusion.host_weight as host_weight_api
from vllm_omni.diffusion.host_weight import session_factory
from vllm_omni.diffusion.host_weight.integrations.minimax_h3_fp8 import (
    MINIMAX_H3_FP8_INTEGRATION,
)
from vllm_omni.diffusion.host_weight.model_integration import (
    HOST_WEIGHT_MODEL_INTEGRATION_ABI,
    DiffusionPipelineSpec,
    HostWeightModelIntegrationBundle,
    ModelIntegrationCapabilities,
    ModelIntegrationError,
    ModelIntegrationLegacyClassification,
    ModelIntegrationRegistry,
    ModelIntegrationSelectionError,
    ModelIntegrationSupportDecision,
    ModelIntegrationUnavailableError,
    ProducerFactoryContext,
    SkeletonFactoryContext,
)
from vllm_omni.diffusion.host_weight.transfer import TransferPlanKind
from vllm_omni.diffusion.model_loader.diffusers_loader import (
    DiffusersPipelineLoader,
)
from vllm_omni.host_weight_runtime import (
    ArtifactSpec,
    ArtifactTopologyDescriptor,
    ProducerDescriptor,
    TopologyCoordinate,
    WeightFormatDescriptor,
    canonical_digest,
    derive_weight_format_plan_digest,
)


class _Target(nn.Module):
    pass


class _Source:
    source_fingerprint = canonical_digest("fake-source")


class _Adapter:
    def __init__(self, descriptor: WeightFormatDescriptor) -> None:
        self.descriptor = descriptor

    def finalize_for_artifact(self, _loaded_module: nn.Module) -> object:
        raise AssertionError("not exercised by bundle conformance")

    def prepare_consumer_structure(self, _target_module: nn.Module, _manifest: object) -> object:
        raise AssertionError("not exercised by bundle conformance")


class _Producer:
    def __init__(self, descriptor: ProducerDescriptor) -> None:
        self.descriptor = descriptor

    def open_build(self, _cleanup_registry: object) -> object:
        raise AssertionError("not exercised by bundle conformance")


class _SkeletonFactory:
    target_module_type = _Target

    def __init__(self, target_module_type_id: str) -> None:
        self.target_module_type_id = target_module_type_id

    def create_candidate(self, *_args: object) -> object:
        raise AssertionError("not exercised by bundle conformance")


def _weight_format(format_id: str, target_module_type_id: str) -> WeightFormatDescriptor:
    semantic = canonical_digest({"format": format_id})
    plan = derive_weight_format_plan_digest(
        format_id=format_id,
        adapter_abi="fake-adapter/v1",
        semantic_fingerprint=semantic,
        format_recipe_schema_version=1,
        target_module_type_id=target_module_type_id,
        normalized_config={},
        kernel_identity={},
    )
    return WeightFormatDescriptor(
        format_id=format_id,
        adapter_abi="fake-adapter/v1",
        semantic_fingerprint=semantic,
        format_plan_digest=plan,
        format_recipe_schema_version=1,
        target_module_type_id=target_module_type_id,
        normalized_config={},
        kernel_identity={},
    )


def _fake_bundle(
    *,
    integration_id: str = "test.second_model.dense",
    model_class_name: str = "SecondPipeline",
    format_id: str = "test_dense",
    target_module_type_id: str = "test_dit/v1",
    transfer_plans: frozenset[TransferPlanKind] = frozenset({TransferPlanKind.COMPONENT}),
    source_resolver: Callable[[object], _Source] | None = None,
    support_probe: Callable[[object], ModelIntegrationSupportDecision] | None = None,
) -> HostWeightModelIntegrationBundle:
    descriptor = _weight_format(format_id, target_module_type_id)
    producer_descriptor = ProducerDescriptor(
        producer_id=f"{integration_id}.producer",
        producer_abi="1",
        semantic_fingerprint=canonical_digest({"producer": integration_id}),
    )
    return HostWeightModelIntegrationBundle(
        integration_id=integration_id,
        integration_abi=HOST_WEIGHT_MODEL_INTEGRATION_ABI,
        capabilities=ModelIntegrationCapabilities(
            model_class_names=frozenset({model_class_name}),
            pipeline_family_id="second_family",
            weight_format_id=format_id,
            target_module_type_id=target_module_type_id,
            artifact_layout_abi="second_layout/v1",
            supported_transfer_plan_kinds=transfer_plans,
        ),
        support_probe=support_probe or (lambda _config: ModelIntegrationSupportDecision.accepted()),
        source_resolver=source_resolver or (lambda _config: _Source()),  # type: ignore[arg-type]
        pipeline_spec_factory=lambda _config: DiffusionPipelineSpec(
            pipeline_family_id="second_family",
            model_config_digest=canonical_digest("second-model-config"),
            normalized_init_config={"component": "transformer"},
        ),
        quantization_preparer=lambda _config: SimpleNamespace(method="dense"),
        weight_format_factory=lambda _config, _device, _quant_config: descriptor,
        format_adapter_factory=_Adapter,
        producer_descriptor_factory=lambda: producer_descriptor,
        producer_factory=lambda context: _Producer(context.spec.producer),
        skeleton_factory=lambda _context: _SkeletonFactory(target_module_type_id),
    )


def test_fake_second_bundle_conforms_to_all_composition_ports() -> None:
    bundle = _fake_bundle()
    config = SimpleNamespace(model_class_name="SecondPipeline")
    prepared = bundle.prepare(config, torch.device("cpu"))  # type: ignore[arg-type]
    assert prepared.source.source_fingerprint == _Source.source_fingerprint
    assert prepared.pipeline_spec.pipeline_family_id == "second_family"
    assert prepared.weight_format.format_id == "test_dense"
    assert prepared.format_adapter.descriptor == prepared.weight_format

    spec = ArtifactSpec(
        source_fingerprint=prepared.source.source_fingerprint,
        producer=prepared.producer_descriptor,
        weight_format=prepared.weight_format,
        topology=ArtifactTopologyDescriptor((TopologyCoordinate("tp", 1, 0),)),
        layout_abi=bundle.capabilities.artifact_layout_abi,
    )
    producer = bundle.create_producer(
        ProducerFactoryContext(
            spec=spec,
            od_config=config,  # type: ignore[arg-type]
            source=prepared.source,
            device=torch.device("cpu"),
            quant_config=prepared.quant_config,
            format_adapter=prepared.format_adapter,
            load_config=LoadConfig(),
        )
    )
    assert producer.descriptor == prepared.producer_descriptor
    skeleton_factory = bundle.create_skeleton_factory(
        SkeletonFactoryContext(
            od_config=config,  # type: ignore[arg-type]
            quant_config=prepared.quant_config,
            pipeline_builder=lambda module: module,
        )
    )
    assert skeleton_factory.target_module_type is _Target


def test_registry_selects_exact_model_format_pair_without_probing() -> None:
    second = _fake_bundle()
    alternate = _fake_bundle(
        integration_id="test.second_model.alternate",
        format_id="test_alternate",
    )
    registry = ModelIntegrationRegistry((MINIMAX_H3_FP8_INTEGRATION, second, alternate))
    config = SimpleNamespace(model_class_name="SecondPipeline")

    assert (
        registry.resolve(config, "test_dense", TransferPlanKind.COMPONENT)  # type: ignore[arg-type]
        is second
    )
    assert (
        registry.resolve(config, "test_alternate", TransferPlanKind.COMPONENT)  # type: ignore[arg-type]
        is alternate
    )
    with pytest.raises(ModelIntegrationUnavailableError) as raised:
        registry.resolve(config, "missing", TransferPlanKind.COMPONENT)  # type: ignore[arg-type]
    assert raised.value.code == "model_weight_format_integration_not_registered"
    assert raised.value.legacy_classification is ModelIntegrationLegacyClassification.CAPABILITY


def test_registry_probe_selects_format_from_quantization_config_without_source_resolution() -> None:
    source_calls = 0

    def source_resolver(_config: object) -> _Source:
        nonlocal source_calls
        source_calls += 1
        return _Source()

    def probe_for(method: str) -> Callable[[object], ModelIntegrationSupportDecision]:
        def probe(config: object) -> ModelIntegrationSupportDecision:
            if getattr(config, "quantization_config", None) == method:
                return ModelIntegrationSupportDecision.accepted()
            return ModelIntegrationSupportDecision.rejected(
                "weight_format_unsupported",
                f"requires quantization method {method!r}",
            )

        return probe

    dense = _fake_bundle(
        source_resolver=source_resolver,
        support_probe=probe_for("dense"),
    )
    alternate = _fake_bundle(
        integration_id="test.second_model.alternate",
        format_id="test_alternate",
        source_resolver=source_resolver,
        support_probe=probe_for("alternate"),
    )
    registry = ModelIntegrationRegistry((dense, alternate))

    selected = registry.select(
        SimpleNamespace(
            model_class_name="SecondPipeline",
            quantization_config="alternate",
        )  # type: ignore[arg-type]
    )

    assert selected is alternate
    assert selected.capabilities.weight_format_id == "test_alternate"
    assert source_calls == 0


def test_registry_probe_no_match_is_typed_and_ambiguity_is_fatal_typed() -> None:
    rejected = _fake_bundle(
        support_probe=lambda _config: ModelIntegrationSupportDecision.rejected(
            "test_format_unsupported",
            "test quantization is unavailable",
        )
    )
    no_match_registry = ModelIntegrationRegistry((rejected,))
    with pytest.raises(ModelIntegrationUnavailableError) as no_match:
        no_match_registry.select(  # type: ignore[arg-type]
            SimpleNamespace(model_class_name="SecondPipeline")
        )
    assert no_match.value.code == "test_format_unsupported"
    assert no_match.value.legacy_classification is ModelIntegrationLegacyClassification.CAPABILITY

    first = _fake_bundle(integration_id="test.first")
    second = _fake_bundle(
        integration_id="test.second",
        format_id="test_alternate",
    )
    ambiguous_registry = ModelIntegrationRegistry((first, second))
    with pytest.raises(ModelIntegrationSelectionError) as ambiguous:
        ambiguous_registry.select(  # type: ignore[arg-type]
            SimpleNamespace(model_class_name="SecondPipeline")
        )
    assert ambiguous.value.code == "model_integration_ambiguous"


def test_probe_exception_is_fatal_but_base_exception_is_not_normalized() -> None:
    probe_error = OSError("injected probe failure")

    def fail_probe(_config: object) -> ModelIntegrationSupportDecision:
        raise probe_error

    broken = _fake_bundle(support_probe=fail_probe)
    with pytest.raises(ModelIntegrationSelectionError) as raised:
        broken.probe(SimpleNamespace())  # type: ignore[arg-type]
    assert raised.value.code == "model_integration_probe_failed"
    assert raised.value.__cause__ is probe_error

    def interrupt_probe(_config: object) -> ModelIntegrationSupportDecision:
        raise SystemExit("injected interrupt")

    interrupted = _fake_bundle(support_probe=interrupt_probe)
    with pytest.raises(SystemExit, match="injected interrupt"):
        interrupted.probe(SimpleNamespace())  # type: ignore[arg-type]


@pytest.mark.parametrize("task_type", ["auto", "combined", "ref2va"])
def test_minimax_probe_rejects_non_fl2va_partition_before_format_probe(
    task_type: str,
) -> None:
    decision = MINIMAX_H3_FP8_INTEGRATION.probe(  # type: ignore[arg-type]
        SimpleNamespace(
            task_type=task_type,
            model="/snapshot/root",
            quantization_config=None,
        )
    )
    assert decision.supported is False
    assert decision.code == "minimax_h3_partition_unsupported"


def test_registry_rejects_duplicate_composite_key_and_plan_before_prepare() -> None:
    first = _fake_bundle(integration_id="test.first")
    duplicate = _fake_bundle(integration_id="test.duplicate")
    with pytest.raises(ModelIntegrationError, match="model/format key"):
        ModelIntegrationRegistry((first, duplicate))

    source_calls = 0

    def source_resolver(_config: object) -> _Source:
        nonlocal source_calls
        source_calls += 1
        return _Source()

    block_only = _fake_bundle(
        integration_id="test.block_only",
        model_class_name="BlockOnlyPipeline",
        transfer_plans=frozenset({TransferPlanKind.BLOCKS_PLUS_RESIDENT}),
        source_resolver=source_resolver,
    )
    registry = ModelIntegrationRegistry((block_only,))
    with pytest.raises(ModelIntegrationUnavailableError) as raised:
        registry.resolve(
            SimpleNamespace(model_class_name="BlockOnlyPipeline"),  # type: ignore[arg-type]
            "test_dense",
            TransferPlanKind.COMPONENT,
        )
    assert raised.value.code == "transfer_plan_unsupported"
    assert source_calls == 0


def test_generic_session_factory_has_no_concrete_model_or_format_imports() -> None:
    source = inspect.getsource(session_factory)
    for concrete_token in (
        "MiniMaxH3",
        "Fp8Config",
        "Fp8PerTensorFormatAdapter",
        "minimax_h3_fp8",
        "formats.fp8_per_tensor",
    ):
        assert concrete_token not in source

    loader_source = inspect.getsource(DiffusersPipelineLoader._load_model_with_host_weight_runtime)
    assert "host_weight.formats.fp8_per_tensor" not in loader_source
    assert "FORMAT_ID" not in loader_source


def test_public_integration_surface_excludes_builtin_model_and_format_types() -> None:
    assert "MiniMaxH3TransformerSkeletonFactory" not in host_weight_api.__all__
    assert "Fp8PerTensorFormatAdapter" not in host_weight_api.__all__
    assert "HostWeightModelIntegrationBundle" in host_weight_api.__all__
