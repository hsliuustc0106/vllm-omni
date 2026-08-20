# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-H3 FL2VA online-FP8 Host Weight Runtime integration."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import nn
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    cutlass_fp8_supported,
)

from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
    MiniMaxH3Pipeline,
)
from vllm_omni.diffusion.registry import _prepare_diffusion_quant_config
from vllm_omni.host_weight_runtime import (
    ProducerDescriptor,
    WeightFormatDescriptor,
    canonical_digest,
    derive_weight_format_plan_digest,
)
from vllm_omni.quantization import resolve_component_quant_config

from ..formats.fp8_per_tensor import (
    FORMAT_ADAPTER_ABI,
    FORMAT_ID,
    FORMAT_RECIPE_SCHEMA_VERSION,
    KERNEL_ID,
    TARGET_MODULE_TYPE_ID,
    Fp8PerTensorFormatAdapter,
)
from ..model_integration import (
    HOST_WEIGHT_MODEL_INTEGRATION_ABI,
    DiffusionPipelineSpec,
    HostWeightModelIntegrationBundle,
    ModelIntegrationCapabilities,
    ModelIntegrationError,
    ModelIntegrationLegacyClassification,
    ModelIntegrationSupportDecision,
    ModelIntegrationUnavailableError,
    ProducerFactoryContext,
    SkeletonFactoryContext,
)
from ..producer import (
    DiffusionArtifactProducer,
    DiffusionArtifactProducerError,
    MiniMaxH3TransformerSource,
    resolve_minimax_h3_transformer_source,
)
from ..skeleton import (
    ConsumerCandidateOrigin,
    ConsumerPipelineCandidate,
    PipelineSkeleton,
    SkeletonError,
)
from ..transfer import TransferPlanKind

MINIMAX_H3_PIPELINE_FAMILY_ID = "minimax_h3"


def _stable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, Mapping):
        return {str(key): _stable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_stable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_stable(item) for item in value), key=repr)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _stable(to_dict())
    if hasattr(value, "__dict__"):
        return {
            key: _stable(item)
            for key, item in sorted(vars(value).items())
            if not key.startswith("_") and not callable(item)
        }
    return f"{type(value).__module__}.{type(value).__qualname__}:{value}"


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "source"


def _resolved_fp8_config(od_config: OmniDiffusionConfig) -> Fp8Config:
    resolved = resolve_component_quant_config(
        od_config.quantization_config,
        "transformer",
    )
    if not isinstance(resolved, Fp8Config):
        raise ModelIntegrationUnavailableError(
            "weight_format_unsupported",
            "MiniMax-H3 HWR supports only transformer FP8 quantization",
            legacy_classification=ModelIntegrationLegacyClassification.CAPABILITY,
        )
    if resolved.is_checkpoint_fp8_serialized:
        raise ModelIntegrationUnavailableError(
            "serialized_fp8_unsupported",
            "MiniMax-H3 HWR requires online FP8 from a dense checkpoint",
            legacy_classification=ModelIntegrationLegacyClassification.CAPABILITY,
        )
    if resolved.weight_block_size is not None:
        raise ModelIntegrationUnavailableError(
            "blockwise_fp8_unsupported",
            "MiniMax-H3 HWR excludes blockwise FP8",
            legacy_classification=ModelIntegrationLegacyClassification.CAPABILITY,
        )
    if resolved.activation_scheme != "dynamic":
        raise ModelIntegrationUnavailableError(
            "fp8_activation_scheme_unsupported",
            "MiniMax-H3 HWR requires dynamic activation scaling",
            legacy_classification=ModelIntegrationLegacyClassification.CAPABILITY,
        )
    if not cutlass_fp8_supported():
        raise ModelIntegrationUnavailableError(
            "fp8_kernel_unavailable",
            "MiniMax-H3 HWR requires the Cutlass FP8 kernel",
            legacy_classification=ModelIntegrationLegacyClassification.CAPABILITY,
        )
    return resolved


def _prepare_quantization(od_config: OmniDiffusionConfig) -> Fp8Config:
    _prepare_diffusion_quant_config(od_config, MiniMaxH3Pipeline)
    return _resolved_fp8_config(od_config)


def _requested_partition_is_fl2va(od_config: OmniDiffusionConfig) -> bool:
    task = str(getattr(od_config, "task_type", None) or "auto").lower()
    if task in {"t2va", "fl2va"}:
        return True
    if task != "auto":
        return False
    model = str(getattr(od_config, "model", "")).rstrip("/\\")
    partition_name = model.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    return bool(model) and partition_name.lower() == "fl2va"


def _support_probe(od_config: OmniDiffusionConfig) -> ModelIntegrationSupportDecision:
    if not _requested_partition_is_fl2va(od_config):
        return ModelIntegrationSupportDecision.rejected(
            "minimax_h3_partition_unsupported",
            "MiniMax-H3 HWR v1 supports only the FL2VA partition; set "
            "task_type to t2va/fl2va or pass the FL2VA partition path",
        )
    try:
        _resolved_fp8_config(od_config)
    except ModelIntegrationUnavailableError as exc:
        return ModelIntegrationSupportDecision.rejected(exc.code, exc.detail)
    return ModelIntegrationSupportDecision.accepted()


def _weight_format_descriptor(
    quant_config: Fp8Config,
    device: torch.device,
    runtime_dtype: torch.dtype,
) -> WeightFormatDescriptor:
    if not isinstance(runtime_dtype, torch.dtype):
        raise ModelIntegrationError(f"MiniMax-H3 HWR requires a resolved torch dtype, got {runtime_dtype!r}")
    runtime_dtype_name = str(runtime_dtype).removeprefix("torch.")
    index = device.index
    if index is None:
        index = torch.cuda.current_device()  # noqa: TID251
    major, minor = torch.cuda.get_device_capability(index)
    normalized_config = {
        "method": "fp8",
        "activation_scheme": quant_config.activation_scheme,
        "ignored_layers": sorted(quant_config.ignored_layers or []),
        "is_checkpoint_fp8_serialized": False,
        "store_dtype": quant_config.store_dtype,
        "weight_block_size": None,
        "input_dtype": runtime_dtype_name,
        "runtime_dtype": runtime_dtype_name,
    }
    kernel_identity = {
        "kernel_id": KERNEL_ID,
        "alignment": 16,
        "cuda_runtime": torch.version.cuda,
        "compute_capability": [major, minor],
        "torch_version": torch.__version__,
        "vllm_version": _package_version("vllm"),
    }
    semantic = canonical_digest(
        {
            "format_id": FORMAT_ID,
            "adapter_abi": FORMAT_ADAPTER_ABI,
            "normalized_config": normalized_config,
            "kernel_identity": kernel_identity,
            "target_module_type_id": TARGET_MODULE_TYPE_ID,
        }
    )
    format_plan_digest = derive_weight_format_plan_digest(
        format_id=FORMAT_ID,
        adapter_abi=FORMAT_ADAPTER_ABI,
        semantic_fingerprint=semantic,
        format_recipe_schema_version=FORMAT_RECIPE_SCHEMA_VERSION,
        target_module_type_id=TARGET_MODULE_TYPE_ID,
        normalized_config=normalized_config,
        kernel_identity=kernel_identity,
    )
    return WeightFormatDescriptor(
        format_id=FORMAT_ID,
        adapter_abi=FORMAT_ADAPTER_ABI,
        semantic_fingerprint=semantic,
        format_plan_digest=format_plan_digest,
        format_recipe_schema_version=FORMAT_RECIPE_SCHEMA_VERSION,
        target_module_type_id=TARGET_MODULE_TYPE_ID,
        normalized_config=normalized_config,
        kernel_identity=kernel_identity,
    )


def _weight_format_factory(
    od_config: OmniDiffusionConfig,
    device: torch.device,
    quant_config: object,
) -> WeightFormatDescriptor:
    if not isinstance(quant_config, Fp8Config):
        raise ModelIntegrationError("MiniMax-H3 bundle received a non-FP8 quantization config")
    return _weight_format_descriptor(quant_config, device, od_config.dtype)


def _producer_descriptor() -> ProducerDescriptor:
    fingerprint = canonical_digest(
        {
            "producer": "minimax_h3_fl2va_transformer",
            "producer_abi": "1",
            "target_module_type_id": TARGET_MODULE_TYPE_ID,
            "loader": "DiffusersPipelineLoader/ordinary-stream-online-fp8",
            "minimax_transforms": ["grouped_qkv_to_qkv", "fc1_gate_up"],
            "torch": torch.__version__,
            "vllm": _package_version("vllm"),
            "vllm_omni": _package_version("vllm-omni"),
        }
    )
    return ProducerDescriptor(
        producer_id="vllm_omni.minimax_h3.fp8_transformer",
        producer_abi="1",
        semantic_fingerprint=fingerprint,
    )


def _pipeline_spec(od_config: OmniDiffusionConfig) -> DiffusionPipelineSpec:
    config = od_config.tf_model_config
    config_value = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    if not isinstance(od_config.dtype, torch.dtype):
        raise ModelIntegrationError(f"MiniMax-H3 HWR requires a resolved torch dtype, got {od_config.dtype!r}")
    runtime_dtype = str(od_config.dtype).removeprefix("torch.")
    model_digest = canonical_digest(
        {
            "runtime_dtype": runtime_dtype,
            "transformer_config": _stable(config_value),
        }
    )
    return DiffusionPipelineSpec(
        pipeline_family_id=MINIMAX_H3_PIPELINE_FAMILY_ID,
        model_config_digest=model_digest,
        normalized_init_config={
            "partition": "fl2va",
            "task_type": str(getattr(od_config, "task_type", "fl2va")),
            "model_config_digest": model_digest,
            "runtime_dtype": runtime_dtype,
        },
    )


def _resolve_source(od_config: OmniDiffusionConfig) -> MiniMaxH3TransformerSource:
    try:
        return resolve_minimax_h3_transformer_source(od_config)
    except DiffusionArtifactProducerError as exc:
        raise ModelIntegrationUnavailableError(
            "model_source_unavailable",
            str(exc),
            legacy_classification=ModelIntegrationLegacyClassification.ARTIFACT,
        ) from exc


class _TargetOnlyPipeline(nn.Module):
    """Reference envelope when a caller does not request a full pipeline."""

    def __init__(self, transformer: nn.Module) -> None:
        super().__init__()
        self.transformer = transformer


def _value(value: object, name: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _validate_weight_format(weight_format: object) -> None:
    if _value(weight_format, "format_id") != FORMAT_ID:
        raise SkeletonError(
            f"MiniMax-H3 skeleton requires weight format {FORMAT_ID!r}, got {_value(weight_format, 'format_id')!r}"
        )
    if str(_value(weight_format, "adapter_abi")) != FORMAT_ADAPTER_ABI:
        raise SkeletonError(f"MiniMax-H3 skeleton requires adapter ABI {FORMAT_ADAPTER_ABI!r}")
    if _value(weight_format, "target_module_type_id") != TARGET_MODULE_TYPE_ID:
        raise SkeletonError(f"MiniMax-H3 skeleton requires target type {TARGET_MODULE_TYPE_ID!r}")
    if int(_value(weight_format, "format_recipe_schema_version", -1)) != FORMAT_RECIPE_SCHEMA_VERSION:
        raise SkeletonError(f"MiniMax-H3 skeleton requires recipe schema {FORMAT_RECIPE_SCHEMA_VERSION}")
    kernel_identity = _value(weight_format, "kernel_identity")
    if not isinstance(kernel_identity, Mapping) or kernel_identity.get("kernel_id") != KERNEL_ID:
        raise SkeletonError(f"MiniMax-H3 skeleton requires kernel {KERNEL_ID!r}")


def _fp8_types() -> tuple[type[Any], type[Any]]:
    from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
        CutlassFP8ScaledMMLinearKernel,
    )
    from vllm.model_executor.layers.quantization.online.fp8 import (
        Fp8PerTensorOnlineLinearMethod,
    )

    return Fp8PerTensorOnlineLinearMethod, CutlassFP8ScaledMMLinearKernel


def _validate_meta_fp8_target(target: nn.Module) -> None:
    materialized = [
        name
        for name, tensor in list(target.named_parameters(recurse=True, remove_duplicate=False))
        + list(target.named_buffers(recurse=True, remove_duplicate=False))
        if not tensor.is_meta and tensor.numel() != 0
    ]
    if materialized:
        raise SkeletonError(f"MiniMax-H3 skeleton construction allocated tensor storage: {materialized[:8]}")

    method_type, kernel_type = _fp8_types()
    found = 0
    for module_path, module in target.named_modules():
        method = getattr(module, "quant_method", None)
        if isinstance(method, method_type):
            found += 1
            kernel = getattr(method, "fp8_linear", None)
            if not isinstance(kernel, kernel_type):
                raise SkeletonError(f"{module_path!r} selected {type(kernel).__name__}, not {kernel_type.__name__}")
            if bool(getattr(method, "use_marlin", False)):
                raise SkeletonError(f"{module_path!r} selected unsupported Marlin FP8")
        elif method is not None and ("Fp8" in type(method).__name__ or "FP8" in type(method).__name__):
            raise SkeletonError(f"{module_path!r} selected unsupported FP8 method {type(method).__name__}")
    if found == 0:
        raise SkeletonError("MiniMax-H3 target contains no online per-tensor FP8 layers")


class MiniMaxH3TransformerSkeletonFactory:
    """Build only the FL2VA transformer under a meta-device context."""

    target_module_path = "transformer"
    target_module_type_id = TARGET_MODULE_TYPE_ID

    def __init__(
        self,
        od_config: object,
        *,
        quant_config: object | None = None,
        pipeline_factory: Callable[..., object] | None = None,
        pipeline: object | None = None,
        transformer_type: type[nn.Module] | None = None,
    ) -> None:
        if pipeline_factory is not None and pipeline is not None:
            raise SkeletonError("pipeline_factory and pipeline are mutually exclusive")
        self._od_config = od_config
        self._quant_config = quant_config
        self._pipeline_factory = pipeline_factory
        self._pipeline = pipeline
        self._transformer_type = transformer_type

    @property
    def target_module_type(self) -> type[nn.Module]:
        if self._transformer_type is not None:
            return self._transformer_type
        from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
            MiniMaxH3DiTModel,
        )

        return MiniMaxH3DiTModel

    def _resolved_quant_config(self) -> object:
        if self._quant_config is not None:
            return self._quant_config
        resolved = resolve_component_quant_config(
            getattr(self._od_config, "quantization_config", None),
            "transformer",
        )
        if resolved is None:
            raise SkeletonError("MiniMax-H3 warm skeleton requires transformer online FP8 configuration")
        return resolved

    @staticmethod
    def attach_transformer(pipeline: object, transformer: nn.Module) -> object:
        try:
            setattr(pipeline, "transformer", transformer)
        except Exception as exc:
            raise SkeletonError("pipeline does not accept a transformer override") from exc
        if getattr(pipeline, "transformer", None) is not transformer:
            raise SkeletonError("pipeline transformer override did not preserve target identity")
        return pipeline

    def resolve_target(self, pipeline: object, target_module_path: str) -> nn.Module:
        current = pipeline
        for component in target_module_path.split("."):
            if not component or not hasattr(current, component):
                raise SkeletonError(f"pipeline has no target path {target_module_path!r}")
            current = getattr(current, component)
        if not isinstance(current, nn.Module):
            raise SkeletonError(f"pipeline target {target_module_path!r} is not an nn.Module")
        return current

    def create(self, pipeline_spec: object, weight_format: object) -> PipelineSkeleton:
        family_id = _value(pipeline_spec, "pipeline_family_id")
        if family_id != MINIMAX_H3_PIPELINE_FAMILY_ID:
            raise SkeletonError(
                f"unsupported pipeline family {family_id!r}; expected {MINIMAX_H3_PIPELINE_FAMILY_ID!r}"
            )
        normalized = _value(pipeline_spec, "normalized_init_config", {})
        if not isinstance(normalized, Mapping):
            raise SkeletonError("pipeline normalized_init_config must be a mapping")
        partition = normalized.get("partition")
        if partition is not None and str(partition).lower() != "fl2va":
            raise SkeletonError(f"v1 MiniMax-H3 skeleton supports FL2VA only, got {partition!r}")
        task_type = normalized.get("task_type", getattr(self._od_config, "task_type", None))
        if isinstance(task_type, str) and task_type.lower() == "ref2va":
            raise SkeletonError("v1 MiniMax-H3 skeleton does not support the Ref2VA transformer")
        _validate_weight_format(weight_format)

        quant_config = self._resolved_quant_config()
        with torch.device("meta"), set_current_diffusion_config(self._od_config):
            transformer = self.target_module_type(
                self._od_config,
                quant_config=quant_config,
            )
        if not isinstance(transformer, self.target_module_type):
            raise SkeletonError("transformer constructor returned the wrong module type")
        _validate_meta_fp8_target(transformer)

        if self._pipeline_factory is not None:
            try:
                pipeline = self._pipeline_factory(
                    od_config=self._od_config,
                    transformer=transformer,
                )
            except Exception as exc:
                raise SkeletonError("pipeline factory rejected the transformer override") from exc
        elif self._pipeline is not None:
            pipeline = self.attach_transformer(self._pipeline, transformer)
        else:
            pipeline = _TargetOnlyPipeline(transformer)

        resolved = self.resolve_target(pipeline, self.target_module_path)
        if resolved is not transformer:
            raise SkeletonError("pipeline target is not the constructed transformer")
        if not isinstance(resolved, self.target_module_type):
            raise SkeletonError("pipeline target type does not match the allowlisted factory")
        return PipelineSkeleton(
            pipeline=pipeline,
            target_module_path=self.target_module_path,
            target_module=transformer,
            target_module_type_id=self.target_module_type_id,
        )

    def create_candidate(
        self,
        pipeline_spec: object,
        weight_format: object,
        origin: ConsumerCandidateOrigin,
        artifact_key: str,
        cleanup: object,
    ) -> ConsumerPipelineCandidate:
        skeleton = self.create(pipeline_spec, weight_format)
        candidate = ConsumerPipelineCandidate(
            skeleton=skeleton,
            origin=origin,
            artifact_key=artifact_key,
        )
        retain = getattr(cleanup, "retain_candidate", None)
        if not callable(retain):
            raise SkeletonError("preparation cleanup registrar cannot retain a pipeline candidate")
        retain(candidate)
        return candidate


def _producer_factory(context: ProducerFactoryContext) -> DiffusionArtifactProducer:
    if not isinstance(context.source, MiniMaxH3TransformerSource):
        raise ModelIntegrationError("MiniMax-H3 producer received an incompatible source")
    if not isinstance(context.quant_config, Fp8Config):
        raise ModelIntegrationError("MiniMax-H3 producer received an incompatible quantization config")
    return DiffusionArtifactProducer(
        spec=context.spec,
        od_config=context.od_config,
        source=context.source,
        device=context.device,
        quant_config=context.quant_config,
        format_adapter=context.format_adapter,
        load_config=context.load_config,
    )


def _skeleton_factory(context: SkeletonFactoryContext) -> MiniMaxH3TransformerSkeletonFactory:
    return MiniMaxH3TransformerSkeletonFactory(
        context.od_config,
        quant_config=context.quant_config,
        pipeline_factory=lambda **kwargs: context.pipeline_builder(kwargs["transformer"]),
    )


MINIMAX_H3_FP8_INTEGRATION = HostWeightModelIntegrationBundle(
    integration_id="vllm_omni.minimax_h3.fl2va.fp8",
    integration_abi=HOST_WEIGHT_MODEL_INTEGRATION_ABI,
    capabilities=ModelIntegrationCapabilities(
        model_class_names=frozenset({"MiniMaxH3Pipeline", "MiniMaxH3ModularPipeline"}),
        pipeline_family_id=MINIMAX_H3_PIPELINE_FAMILY_ID,
        weight_format_id=FORMAT_ID,
        target_module_type_id=TARGET_MODULE_TYPE_ID,
        artifact_layout_abi="minimax_h3_dit_runtime/v1",
        supported_transfer_plan_kinds=frozenset(
            {
                TransferPlanKind.COMPONENT,
                TransferPlanKind.BLOCKS_PLUS_RESIDENT,
            }
        ),
    ),
    support_probe=_support_probe,
    source_resolver=_resolve_source,
    pipeline_spec_factory=_pipeline_spec,
    quantization_preparer=_prepare_quantization,
    weight_format_factory=_weight_format_factory,
    format_adapter_factory=Fp8PerTensorFormatAdapter,
    producer_descriptor_factory=_producer_descriptor,
    producer_factory=_producer_factory,
    skeleton_factory=_skeleton_factory,
)


__all__ = [
    "MINIMAX_H3_FP8_INTEGRATION",
    "MINIMAX_H3_PIPELINE_FAMILY_ID",
    "MiniMaxH3TransformerSkeletonFactory",
]
