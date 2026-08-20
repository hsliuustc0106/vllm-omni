# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Typed model-integration boundary for diffusion Host Weight Runtime.

The session composition root owns runtime policy and lifecycle.  A model
integration bundle owns every model- or weight-format-specific decision used
to describe, build, and hydrate one artifact.  Keeping those two authorities
separate lets another model integration be registered without adding a branch
to the generic session factory.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import torch
from torch import nn
from vllm.config.load import LoadConfig

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.host_weight_runtime import (
    ArtifactProducer,
    ArtifactSpec,
    ProducerDescriptor,
    WeightFormatDescriptor,
)

from .formats.base import WeightFormatAdapter
from .skeleton import (
    ConsumerCandidateOrigin,
    ConsumerPipelineCandidate,
)
from .transfer import TransferPlanKind

HOST_WEIGHT_MODEL_INTEGRATION_ABI = "vllm_omni.diffusion.host_weight.model_integration/v1"


class ModelIntegrationError(RuntimeError):
    """A selected integration violated its trusted local contract."""


class ModelIntegrationLegacyClassification(str, Enum):
    CAPABILITY = "capability"
    ARTIFACT = "artifact"


class ModelIntegrationUnavailableError(ModelIntegrationError):
    """No exact supported integration is available for the requested model."""

    def __init__(
        self,
        code: str,
        detail: str,
        *,
        legacy_classification: ModelIntegrationLegacyClassification,
    ) -> None:
        if not code:
            raise ValueError("model integration unavailable code must not be empty")
        if not detail:
            raise ValueError("model integration unavailable detail must not be empty")
        self.code = code
        self.detail = detail
        self.legacy_classification = ModelIntegrationLegacyClassification(legacy_classification)
        super().__init__(f"{code}: {detail}")


class ModelIntegrationSelectionError(ModelIntegrationError):
    """The registry could not choose one unambiguous trusted integration."""

    def __init__(self, code: str, detail: str) -> None:
        if not code:
            raise ValueError("model integration selection code must not be empty")
        if not detail:
            raise ValueError("model integration selection detail must not be empty")
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class ModelIntegrationSupportDecision:
    """Side-effect-free result of probing one bundle against a request."""

    supported: bool
    code: str | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.supported, bool):
            raise ModelIntegrationError("model integration support status must be a bool")
        if self.supported:
            if self.code is not None or self.detail is not None:
                raise ModelIntegrationError("a supported integration probe must not include rejection detail")
            return
        if not isinstance(self.code, str) or not self.code:
            raise ModelIntegrationError("an unsupported integration probe requires a non-empty code")
        if not isinstance(self.detail, str) or not self.detail:
            raise ModelIntegrationError("an unsupported integration probe requires non-empty detail")

    @classmethod
    def accepted(cls) -> ModelIntegrationSupportDecision:
        return cls(supported=True)

    @classmethod
    def rejected(cls, code: str, detail: str) -> ModelIntegrationSupportDecision:
        return cls(supported=False, code=code, detail=detail)


@dataclass(frozen=True, slots=True)
class DiffusionPipelineSpec:
    pipeline_family_id: str
    model_config_digest: str
    normalized_init_config: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.pipeline_family_id:
            raise ModelIntegrationError("pipeline family ID must not be empty")
        if not self.model_config_digest:
            raise ModelIntegrationError("model config digest must not be empty")
        if not isinstance(self.normalized_init_config, Mapping):
            raise ModelIntegrationError("normalized pipeline init config must be a mapping")


class ModelWeightSource(Protocol):
    """Immutable source identity resolved without constructing model weights."""

    @property
    def source_fingerprint(self) -> str: ...


class ConsumerSkeletonFactory(Protocol):
    """Model-owned factory for a no-storage consumer candidate."""

    @property
    def target_module_type(self) -> type[nn.Module]: ...

    target_module_type_id: str

    def create_candidate(
        self,
        pipeline_spec: DiffusionPipelineSpec,
        weight_format: WeightFormatDescriptor,
        origin: ConsumerCandidateOrigin,
        artifact_key: str,
        cleanup: object,
    ) -> ConsumerPipelineCandidate: ...


@dataclass(frozen=True, slots=True)
class ModelIntegrationCapabilities:
    """Declarative compatibility envelope for one trusted bundle."""

    model_class_names: frozenset[str]
    pipeline_family_id: str
    weight_format_id: str
    target_module_type_id: str
    artifact_layout_abi: str
    supported_transfer_plan_kinds: frozenset[TransferPlanKind]

    def __post_init__(self) -> None:
        model_class_names = frozenset(self.model_class_names)
        transfer_plan_kinds = frozenset(TransferPlanKind(kind) for kind in self.supported_transfer_plan_kinds)
        object.__setattr__(self, "model_class_names", model_class_names)
        object.__setattr__(self, "supported_transfer_plan_kinds", transfer_plan_kinds)
        if not model_class_names or any(not isinstance(name, str) or not name for name in model_class_names):
            raise ModelIntegrationError("model integration requires non-empty model class names")
        for label, value in (
            ("pipeline family ID", self.pipeline_family_id),
            ("weight format ID", self.weight_format_id),
            ("target module type ID", self.target_module_type_id),
            ("artifact layout ABI", self.artifact_layout_abi),
        ):
            if not isinstance(value, str) or not value:
                raise ModelIntegrationError(f"{label} must not be empty")
        if not transfer_plan_kinds:
            raise ModelIntegrationError("model integration must support at least one transfer plan kind")


@dataclass(frozen=True, slots=True)
class PreparedModelIntegration:
    source: ModelWeightSource
    pipeline_spec: DiffusionPipelineSpec
    quant_config: object
    weight_format: WeightFormatDescriptor
    format_adapter: WeightFormatAdapter
    producer_descriptor: ProducerDescriptor


@dataclass(frozen=True, slots=True)
class ProducerFactoryContext:
    spec: ArtifactSpec
    od_config: OmniDiffusionConfig
    source: ModelWeightSource
    device: torch.device
    quant_config: object
    format_adapter: WeightFormatAdapter
    load_config: LoadConfig


@dataclass(frozen=True, slots=True)
class SkeletonFactoryContext:
    od_config: OmniDiffusionConfig
    quant_config: object
    pipeline_builder: Callable[[nn.Module], nn.Module]


SourceResolver = Callable[[OmniDiffusionConfig], ModelWeightSource]
SupportProbe = Callable[[OmniDiffusionConfig], ModelIntegrationSupportDecision]
PipelineSpecFactory = Callable[[OmniDiffusionConfig], DiffusionPipelineSpec]
QuantizationPreparer = Callable[[OmniDiffusionConfig], object]
WeightFormatFactory = Callable[[OmniDiffusionConfig, torch.device, object], WeightFormatDescriptor]
FormatAdapterFactory = Callable[[WeightFormatDescriptor], WeightFormatAdapter]
ProducerDescriptorFactory = Callable[[], ProducerDescriptor]
ProducerFactory = Callable[[ProducerFactoryContext], ArtifactProducer]
SkeletonFactory = Callable[[SkeletonFactoryContext], ConsumerSkeletonFactory]


@dataclass(frozen=True, slots=True)
class HostWeightModelIntegrationBundle:
    """All trusted model-specific ports consumed by the generic composition."""

    integration_id: str
    integration_abi: str
    capabilities: ModelIntegrationCapabilities
    support_probe: SupportProbe
    source_resolver: SourceResolver
    pipeline_spec_factory: PipelineSpecFactory
    quantization_preparer: QuantizationPreparer
    weight_format_factory: WeightFormatFactory
    format_adapter_factory: FormatAdapterFactory
    producer_descriptor_factory: ProducerDescriptorFactory
    producer_factory: ProducerFactory
    skeleton_factory: SkeletonFactory

    def __post_init__(self) -> None:
        if not isinstance(self.integration_id, str) or not self.integration_id:
            raise ModelIntegrationError("model integration ID must not be empty")
        if self.integration_abi != HOST_WEIGHT_MODEL_INTEGRATION_ABI:
            raise ModelIntegrationError(
                f"model integration {self.integration_id!r} declares unsupported ABI {self.integration_abi!r}"
            )
        for field_name in (
            "support_probe",
            "source_resolver",
            "pipeline_spec_factory",
            "quantization_preparer",
            "weight_format_factory",
            "format_adapter_factory",
            "producer_descriptor_factory",
            "producer_factory",
            "skeleton_factory",
        ):
            if not callable(getattr(self, field_name)):
                raise ModelIntegrationError(f"model integration port {field_name!r} must be callable")

    def probe(self, od_config: OmniDiffusionConfig) -> ModelIntegrationSupportDecision:
        """Check request compatibility without resolving sources or building models."""

        try:
            decision = self.support_probe(od_config)
        except Exception as exc:
            raise ModelIntegrationSelectionError(
                "model_integration_probe_failed",
                f"support probe for {self.integration_id!r} raised {type(exc).__name__}: {exc}",
            ) from exc
        if not isinstance(decision, ModelIntegrationSupportDecision):
            raise ModelIntegrationError(
                f"model integration {self.integration_id!r} support probe returned an unsupported value"
            )
        return decision

    def prepare(
        self,
        od_config: OmniDiffusionConfig,
        device: torch.device,
    ) -> PreparedModelIntegration:
        """Resolve immutable identity and local format ports without loading weights."""

        source = self.source_resolver(od_config)
        source_fingerprint = getattr(source, "source_fingerprint", None)
        if not isinstance(source_fingerprint, str) or not source_fingerprint:
            raise ModelIntegrationError("model source must expose a non-empty source_fingerprint")
        pipeline_spec = self.pipeline_spec_factory(od_config)
        if not isinstance(pipeline_spec, DiffusionPipelineSpec):
            raise ModelIntegrationError("pipeline spec factory returned an unsupported value")
        if pipeline_spec.pipeline_family_id != self.capabilities.pipeline_family_id:
            raise ModelIntegrationError("pipeline spec family differs from the bundle capability")
        quant_config = self.quantization_preparer(od_config)
        weight_format = self.weight_format_factory(od_config, device, quant_config)
        if not isinstance(weight_format, WeightFormatDescriptor):
            raise ModelIntegrationError("weight format factory returned an unsupported value")
        if weight_format.format_id != self.capabilities.weight_format_id:
            raise ModelIntegrationError("weight format ID differs from the bundle capability")
        if weight_format.target_module_type_id != self.capabilities.target_module_type_id:
            raise ModelIntegrationError("weight format target type differs from the bundle capability")
        format_adapter = self.format_adapter_factory(weight_format)
        if getattr(format_adapter, "descriptor", None) != weight_format:
            raise ModelIntegrationError("format adapter descriptor differs from the prepared weight format")
        producer_descriptor = self.producer_descriptor_factory()
        if not isinstance(producer_descriptor, ProducerDescriptor):
            raise ModelIntegrationError("producer descriptor factory returned an unsupported value")
        return PreparedModelIntegration(
            source=source,
            pipeline_spec=pipeline_spec,
            quant_config=quant_config,
            weight_format=weight_format,
            format_adapter=format_adapter,
            producer_descriptor=producer_descriptor,
        )

    def create_producer(self, context: ProducerFactoryContext) -> ArtifactProducer:
        producer = self.producer_factory(context)
        if getattr(producer, "descriptor", None) != context.spec.producer:
            raise ModelIntegrationError("artifact producer descriptor differs from the artifact spec")
        if not callable(getattr(producer, "open_build", None)):
            raise ModelIntegrationError("artifact producer does not implement open_build")
        return producer

    def create_skeleton_factory(
        self,
        context: SkeletonFactoryContext,
    ) -> ConsumerSkeletonFactory:
        factory = self.skeleton_factory(context)
        if getattr(factory, "target_module_type_id", None) != self.capabilities.target_module_type_id:
            raise ModelIntegrationError("skeleton factory target type differs from the bundle capability")
        target_module_type = getattr(factory, "target_module_type", None)
        if not isinstance(target_module_type, type) or not issubclass(target_module_type, nn.Module):
            raise ModelIntegrationError("skeleton factory must declare one nn.Module target type")
        if not callable(getattr(factory, "create_candidate", None)):
            raise ModelIntegrationError("skeleton factory does not implement create_candidate")
        return factory


class ModelIntegrationRegistry:
    """Immutable exact-key registry; selection never probes model paths."""

    def __init__(self, bundles: Iterable[HostWeightModelIntegrationBundle] = ()) -> None:
        by_selection_key: dict[tuple[str, str], HostWeightModelIntegrationBundle] = {}
        ordered: list[HostWeightModelIntegrationBundle] = []
        integration_ids: set[str] = set()
        for bundle in bundles:
            if not isinstance(bundle, HostWeightModelIntegrationBundle):
                raise TypeError("model integration registry accepts only typed bundles")
            if bundle.integration_id in integration_ids:
                raise ModelIntegrationError(f"duplicate model integration ID {bundle.integration_id!r}")
            integration_ids.add(bundle.integration_id)
            ordered.append(bundle)
            for model_class_name in bundle.capabilities.model_class_names:
                selection_key = (model_class_name, bundle.capabilities.weight_format_id)
                existing = by_selection_key.get(selection_key)
                if existing is not None:
                    raise ModelIntegrationError(
                        f"model/format key {selection_key!r} is claimed by both "
                        f"{existing.integration_id!r} and {bundle.integration_id!r}"
                    )
                by_selection_key[selection_key] = bundle
        self._bundles = tuple(ordered)
        self._by_selection_key = by_selection_key

    @property
    def bundles(self) -> tuple[HostWeightModelIntegrationBundle, ...]:
        return self._bundles

    def select(
        self,
        od_config: OmniDiffusionConfig,
    ) -> HostWeightModelIntegrationBundle:
        """Select exactly one bundle from model identity and quantization config."""

        model_class_name = self._model_class_name(od_config)
        candidates = tuple(
            bundle for bundle in self._bundles if model_class_name in bundle.capabilities.model_class_names
        )
        if not candidates:
            raise ModelIntegrationUnavailableError(
                "model_integration_not_registered",
                f"no Host Weight Runtime integration is registered for model class {model_class_name!r}",
                legacy_classification=ModelIntegrationLegacyClassification.CAPABILITY,
            )

        supported: list[HostWeightModelIntegrationBundle] = []
        rejected: list[tuple[HostWeightModelIntegrationBundle, ModelIntegrationSupportDecision]] = []
        for bundle in candidates:
            decision = bundle.probe(od_config)
            if decision.supported:
                supported.append(bundle)
            else:
                rejected.append((bundle, decision))

        if len(supported) == 1:
            return supported[0]
        if len(supported) > 1:
            integration_ids = ", ".join(repr(bundle.integration_id) for bundle in supported)
            raise ModelIntegrationSelectionError(
                "model_integration_ambiguous",
                f"multiple Host Weight Runtime integrations support model class "
                f"{model_class_name!r} and its quantization configuration: {integration_ids}",
            )

        if len(rejected) == 1:
            decision = rejected[0][1]
            assert decision.code is not None
            assert decision.detail is not None
            raise ModelIntegrationUnavailableError(
                decision.code,
                decision.detail,
                legacy_classification=ModelIntegrationLegacyClassification.CAPABILITY,
            )
        rejection_detail = "; ".join(
            f"{bundle.integration_id}: {decision.code}: {decision.detail}" for bundle, decision in rejected
        )
        raise ModelIntegrationUnavailableError(
            "model_integration_probe_no_match",
            f"no Host Weight Runtime integration supports model class {model_class_name!r} "
            f"and its quantization configuration ({rejection_detail})",
            legacy_classification=ModelIntegrationLegacyClassification.CAPABILITY,
        )

    @staticmethod
    def _model_class_name(od_config: OmniDiffusionConfig) -> str:
        model_class_name = getattr(od_config, "model_class_name", None)
        if not isinstance(model_class_name, str) or not model_class_name:
            raise ModelIntegrationUnavailableError(
                "model_class_name_unavailable",
                "diffusion model_class_name is unavailable for exact HWR selection",
                legacy_classification=ModelIntegrationLegacyClassification.CAPABILITY,
            )
        return model_class_name

    def validate_transfer_plan(
        self,
        bundle: HostWeightModelIntegrationBundle,
        required_transfer_plan_kind: TransferPlanKind,
    ) -> TransferPlanKind:
        """Validate a consumer plan against the already selected bundle."""

        if not any(registered is bundle for registered in self._bundles):
            raise ModelIntegrationSelectionError(
                "model_integration_not_registered",
                f"selected integration {bundle.integration_id!r} does not belong to this registry",
            )
        try:
            transfer_plan_kind = TransferPlanKind(required_transfer_plan_kind)
        except ValueError as exc:
            raise ModelIntegrationUnavailableError(
                "transfer_plan_unsupported",
                f"consumer requested unknown transfer plan kind {required_transfer_plan_kind!r}",
                legacy_classification=ModelIntegrationLegacyClassification.CAPABILITY,
            ) from exc
        if transfer_plan_kind not in bundle.capabilities.supported_transfer_plan_kinds:
            raise ModelIntegrationUnavailableError(
                "transfer_plan_unsupported",
                f"integration {bundle.integration_id!r} does not support transfer plan {transfer_plan_kind.value!r}",
                legacy_classification=ModelIntegrationLegacyClassification.CAPABILITY,
            )
        return transfer_plan_kind

    def resolve(
        self,
        od_config: OmniDiffusionConfig,
        required_weight_format_id: str,
        required_transfer_plan_kind: TransferPlanKind,
    ) -> HostWeightModelIntegrationBundle:
        model_class_name = self._model_class_name(od_config)
        if not isinstance(required_weight_format_id, str) or not required_weight_format_id:
            raise ModelIntegrationUnavailableError(
                "weight_format_id_unavailable",
                "consumer did not provide an exact required weight format ID",
                legacy_classification=ModelIntegrationLegacyClassification.CAPABILITY,
            )
        selection_key = (model_class_name, required_weight_format_id)
        bundle = self._by_selection_key.get(selection_key)
        if bundle is None:
            raise ModelIntegrationUnavailableError(
                "model_weight_format_integration_not_registered",
                f"no Host Weight Runtime integration is registered for model/format key {selection_key!r}",
                legacy_classification=ModelIntegrationLegacyClassification.CAPABILITY,
            )
        self.validate_transfer_plan(bundle, required_transfer_plan_kind)
        return bundle


__all__ = [
    "HOST_WEIGHT_MODEL_INTEGRATION_ABI",
    "ConsumerSkeletonFactory",
    "DiffusionPipelineSpec",
    "HostWeightModelIntegrationBundle",
    "ModelIntegrationCapabilities",
    "ModelIntegrationError",
    "ModelIntegrationLegacyClassification",
    "ModelIntegrationRegistry",
    "ModelIntegrationSelectionError",
    "ModelIntegrationSupportDecision",
    "ModelIntegrationUnavailableError",
    "ModelWeightSource",
    "PreparedModelIntegration",
    "ProducerFactoryContext",
    "SkeletonFactoryContext",
]
