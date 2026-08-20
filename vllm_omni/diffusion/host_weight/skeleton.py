# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""No-storage MiniMax-H3 consumer skeleton construction."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
from torch import nn

from vllm_omni.diffusion.config import set_current_diffusion_config

from .formats.base import FormatContractError
from .formats.fp8_per_tensor import (
    FORMAT_ADAPTER_ABI,
    FORMAT_ID,
    FORMAT_RECIPE_SCHEMA_VERSION,
    KERNEL_ID,
    TARGET_MODULE_TYPE_ID,
)

MINIMAX_H3_PIPELINE_FAMILY_ID = "minimax_h3"


class SkeletonError(FormatContractError):
    """Raised before artifact tensors mutate a consumer module."""


@dataclass(frozen=True, slots=True)
class PipelineSkeleton:
    pipeline: object
    target_module_path: str
    target_module: nn.Module
    target_module_type_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.target_module, nn.Module):
            raise SkeletonError("pipeline skeleton target must be an nn.Module")
        if not self.target_module_path:
            raise SkeletonError("pipeline skeleton target path must not be empty")
        if self.target_module_type_id != TARGET_MODULE_TYPE_ID:
            raise SkeletonError(f"unsupported target module type ID {self.target_module_type_id!r}")


class ConsumerCandidateOrigin(str, Enum):
    BUILDER_REBIND = "builder_rebind"
    ORDERED_WAITER = "ordered_waiter"
    WARM_HIT = "warm_hit"


@dataclass(frozen=True, slots=True)
class ConsumerPipelineCandidate:
    """Exact locally constructed pipeline retained before factory return."""

    skeleton: PipelineSkeleton
    origin: ConsumerCandidateOrigin
    artifact_key: str

    def __post_init__(self) -> None:
        if not self.artifact_key:
            raise SkeletonError("consumer pipeline candidate artifact key must not be empty")


class _TargetOnlyPipeline(nn.Module):
    """Integration test/reference envelope when no full pipeline is requested."""

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
    """Build only the FL2VA transformer under a meta-device context.

    A full pipeline constructor may be supplied as ``pipeline_factory``.  Its
    narrow contract is ``factory(od_config=..., transformer=...)`` and it must
    honor that override without constructing or loading another transformer.
    This class never reads ``weights_sources`` and never invokes a loader or a
    post-load finalizer.
    """

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
        quant_config = getattr(self._od_config, "quantization_config", None)
        from vllm_omni.quantization.component_config import (
            resolve_component_quant_config,
        )

        resolved = resolve_component_quant_config(quant_config, "transformer")
        if resolved is None:
            raise SkeletonError("MiniMax-H3 warm skeleton requires transformer online FP8 configuration")
        return resolved

    @staticmethod
    def attach_transformer(pipeline: object, transformer: nn.Module) -> object:
        """Inject the target before binding without importing a pipeline type."""

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
        """Construct and owner-register a candidate before returning it."""

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


__all__ = [
    "ConsumerCandidateOrigin",
    "ConsumerPipelineCandidate",
    "MINIMAX_H3_PIPELINE_FAMILY_ID",
    "MiniMaxH3TransformerSkeletonFactory",
    "PipelineSkeleton",
    "SkeletonError",
]
