# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Trusted import/export adapter for finalized online per-tensor FP8 weights."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from types import MappingProxyType
from typing import Any
from weakref import WeakKeyDictionary

import torch
from torch import nn

from .base import (
    FORMAT_BINDING_RECIPE_SCHEMA_VERSION,
    BindingDestination,
    FinalizedTensor,
    FinalizedTensorSet,
    FormatBindingRecipe,
    FormatContractError,
    FormatTensorRole,
    JSONValue,
    LayerFormatSpec,
    LayerTensorBinding,
    ModuleStateKind,
    OptionalLayerTensorBinding,
    RequiredLayerTensorBinding,
    TargetModulePath,
    TensorBindingSpec,
    canonical_digest,
)

TARGET_MODULE_TYPE_ID = "minimax_h3_dit/v1"
FORMAT_ID = "vllm.online_fp8.per_tensor.cutlass"
QUANT_METHOD_ID = "Fp8PerTensorOnlineLinearMethod"
KERNEL_ID = "CutlassFP8ScaledMMLinearKernel"
FORMAT_RECIPE_SCHEMA_VERSION = FORMAT_BINDING_RECIPE_SCHEMA_VERSION
FORMAT_ADAPTER_ABI = "2"
_CUTLASS_ALIGNMENT = 16

_SCALAR_KEYS = frozenset(
    {
        "activation_quant_key",
        "finalization_complete",
        "input_dtype",
        "input_scale",
        "input_size_per_partition",
        "kernel_logical_output_size",
        "logical_widths",
        "orig_dtype",
        "out_dtype",
        "output_size_per_partition",
        "use_marlin",
        "weight_block_size",
        "weight_quant_key",
    }
)


class Fp8FormatError(FormatContractError):
    """Raised when the first FP8 format cannot be exported or imported."""


@dataclass(frozen=True, slots=True)
class _OnlineLayerwiseStateSnapshot:
    module_path: str
    module: nn.Module
    info: object


class _OnlineLayerwiseStateRetirement:
    """Reversible retirement of upstream state owned by hydrated FP8 layers."""

    def __init__(
        self,
        registry: WeakKeyDictionary[nn.Module, object],
        snapshots: tuple[_OnlineLayerwiseStateSnapshot, ...],
    ) -> None:
        self._registry: WeakKeyDictionary[nn.Module, object] | None = registry
        self._snapshots = list(snapshots)
        self._restore_stack: list[_OnlineLayerwiseStateSnapshot] = []
        self._apply_started = False
        self._applied = False
        self._retirement_committed = False

    @property
    def retirement_committed(self) -> bool:
        return self._retirement_committed

    def apply(self) -> None:
        """Remove captured entries while retaining an exact rollback owner."""

        registry = self._registry
        if registry is None:
            raise Fp8FormatError("online-layerwise state retirement is no longer active")
        if self._apply_started:
            raise Fp8FormatError("online-layerwise state retirement was already applied")

        # Publish that application started before its first registry mutation.
        # The binder already owns this transaction at this point.  Each
        # snapshot is then pushed before deletion so an exception immediately
        # after ``del`` cannot orphan the exact object that must be restored.
        self._apply_started = True
        for snapshot in self._snapshots:
            if registry.get(snapshot.module) is not snapshot.info:
                raise Fp8FormatError(f"{snapshot.module_path!r} upstream layerwise state changed during retirement")
            self._restore_stack.append(snapshot)
            del registry[snapshot.module]
        self._applied = True

    def validate_quiesced(self) -> None:
        registry = self._registry
        if registry is None:
            raise Fp8FormatError("online-layerwise state retirement is no longer active")
        if not self._applied:
            raise Fp8FormatError("online-layerwise state retirement was not fully applied")
        rebound: list[str] = []
        for snapshot in self._snapshots:
            if snapshot.module not in registry:
                continue
            info = registry[snapshot.module]
            can_load = getattr(info, "can_load", None)
            is_inert = (
                callable(can_load)
                and not can_load()
                and getattr(info, "load_numel", None) == 0
                and getattr(info, "load_numel_total", object()) is None
                and getattr(info, "kernel_tensors", object()) is None
                and getattr(info, "loaded_weights", object()) == []
            )
            if not is_inert:
                rebound.append(snapshot.module_path)
        if rebound:
            raise Fp8FormatError(
                "active upstream online-layerwise loader state was recreated before consumer "
                f"binding completed for layers {rebound}"
            )

    def commit(self) -> None:
        if not self._applied:
            raise Fp8FormatError("online-layerwise state retirement was not fully applied")
        self.validate_quiesced()
        # The original Parameters (and their wrapped loaders) are retained by
        # the binder's structural snapshots until this same commit. Dropping
        # these references makes the retirement permanent without disturbing
        # any blank entries an ordinary finalizer may have since created.
        # Publish the irreversible outcome before dropping the rollback data so
        # an async interruption in the return path remains distinguishable
        # from a commit that never started.
        self._retirement_committed = True
        self._restore_stack.clear()
        self._snapshots.clear()
        self._registry = None

    def rollback(self) -> None:
        registry = self._registry
        if registry is None:
            return
        while self._restore_stack:
            snapshot = self._restore_stack[-1]
            # Replace a blank entry that finalize_layerwise_processing may have
            # created after hydration with the exact original active object.
            # Pop only after the assignment returns: if restoration raises (or
            # is interrupted after mutating the registry), retrying the same
            # idempotent assignment preserves ownership and makes progress.
            registry[snapshot.module] = snapshot.info
            self._restore_stack.pop()
        self._snapshots.clear()
        self._registry = None
        self._applied = False


def _fp8_method_type() -> type[Any]:
    from vllm.model_executor.layers.quantization.online.fp8 import (
        Fp8PerTensorOnlineLinearMethod,
    )

    return Fp8PerTensorOnlineLinearMethod


def _cutlass_kernel_type() -> type[Any]:
    from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
        CutlassFP8ScaledMMLinearKernel,
    )

    return CutlassFP8ScaledMMLinearKernel


def _online_layerwise_api() -> tuple[
    WeakKeyDictionary[nn.Module, object],
    type[Any],
]:
    """Resolve the exact upstream state API this adapter transaction requires."""

    try:
        from vllm.model_executor.model_loader.reload import layerwise
        from vllm.model_executor.model_loader.reload.types import (
            LayerReloadingInfo,
        )
    except ImportError as exc:
        raise Fp8FormatError("online FP8 consumer binding requires vLLM's layerwise reload state API") from exc

    registry = getattr(layerwise, "LAYERWISE_INFO", None)
    if not isinstance(registry, WeakKeyDictionary):
        raise Fp8FormatError("vLLM layerwise reload state registry is incompatible with this adapter")
    if not isinstance(LayerReloadingInfo, type) or not callable(getattr(LayerReloadingInfo, "can_load", None)):
        raise Fp8FormatError("vLLM LayerReloadingInfo is incompatible with this adapter")
    return registry, LayerReloadingInfo


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _dtype_from_name(name: object) -> torch.dtype:
    if not isinstance(name, str):
        raise Fp8FormatError("dtype name must be a string")
    normalized = name.removeprefix("torch.")
    dtype = getattr(torch, normalized, None)
    if not isinstance(dtype, torch.dtype):
        raise Fp8FormatError(f"unsupported torch dtype {name!r}")
    return dtype


def _join_path(prefix: str, suffix: str) -> str:
    return f"{prefix}.{suffix}" if prefix else suffix


def _split_state_path(path: str) -> tuple[str, str]:
    owner, separator, attribute = path.rpartition(".")
    return (owner, attribute) if separator else ("", path)


def _target_path(path: str) -> TargetModulePath:
    return TargetModulePath(path or ".")


def _module_path(path: object) -> str:
    normalized = str(path)
    return "" if normalized == "." else normalized


def _descriptor_mapping(descriptor: object) -> dict[str, JSONValue]:
    if isinstance(descriptor, Mapping):
        raw = dict(descriptor)
    elif callable(getattr(descriptor, "to_dict", None)):
        raw = descriptor.to_dict()
    elif is_dataclass(descriptor):
        raw = asdict(descriptor)
    else:
        fields = (
            "format_id",
            "adapter_abi",
            "semantic_fingerprint",
            "format_plan_digest",
            "format_recipe_schema_version",
            "target_module_type_id",
            "normalized_config",
            "kernel_identity",
        )
        if not all(hasattr(descriptor, field) for field in fields):
            raise Fp8FormatError(f"unsupported weight-format descriptor {type(descriptor).__name__}")
        raw = {field: getattr(descriptor, field) for field in fields}
    required = {
        "format_id",
        "adapter_abi",
        "semantic_fingerprint",
        "format_plan_digest",
        "format_recipe_schema_version",
        "target_module_type_id",
        "normalized_config",
        "kernel_identity",
    }
    if set(raw) != required:
        raise Fp8FormatError(
            "weight-format descriptor fields do not match the v3 contract: "
            f"missing={sorted(required - set(raw))}, extra={sorted(set(raw) - required)}"
        )
    # canonical_digest performs the recursive JSON type validation.
    canonical_digest(raw)
    return raw  # type: ignore[return-value]


def _default_descriptor() -> object:
    normalized_config: dict[str, JSONValue] = {
        "method": "fp8_per_tensor",
        "scale_dtype": "float32",
        "weight_block_size": None,
        "weight_dtype": "float8_e4m3fn",
    }
    kernel_identity: dict[str, JSONValue] = {
        "alignment": _CUTLASS_ALIGNMENT,
        "kernel_id": KERNEL_ID,
    }
    semantic_fingerprint = canonical_digest(
        {
            "normalized_config": normalized_config,
            "kernel_identity": kernel_identity,
            "target_module_type_id": TARGET_MODULE_TYPE_ID,
        }
    )
    format_plan_digest = canonical_digest(
        {
            "format_id": FORMAT_ID,
            "adapter_abi": FORMAT_ADAPTER_ABI,
            "semantic_fingerprint": semantic_fingerprint,
            "format_recipe_schema_version": FORMAT_RECIPE_SCHEMA_VERSION,
            "target_module_type_id": TARGET_MODULE_TYPE_ID,
            "normalized_config": normalized_config,
            "kernel_identity": kernel_identity,
        }
    )
    kwargs = {
        "format_id": FORMAT_ID,
        "adapter_abi": FORMAT_ADAPTER_ABI,
        "semantic_fingerprint": semantic_fingerprint,
        "format_plan_digest": format_plan_digest,
        "format_recipe_schema_version": FORMAT_RECIPE_SCHEMA_VERSION,
        "target_module_type_id": TARGET_MODULE_TYPE_ID,
        "normalized_config": normalized_config,
        "kernel_identity": kernel_identity,
    }
    try:
        from vllm_omni.host_weight_runtime.contracts import WeightFormatDescriptor
    except ImportError:
        return MappingProxyType(kwargs)
    try:
        return WeightFormatDescriptor(**kwargs)
    except TypeError:
        # Keep this module importable while the core contract is upgraded in a
        # separate change; callers may always supply the exact core descriptor.
        return MappingProxyType(kwargs)


def _manifest_value(manifest: object, name: str) -> object:
    if isinstance(manifest, Mapping):
        if name not in manifest:
            raise Fp8FormatError(f"artifact manifest is missing {name!r}")
        return manifest[name]
    if not hasattr(manifest, name):
        raise Fp8FormatError(f"artifact manifest is missing {name!r}")
    return getattr(manifest, name)


def _manifest_tensor_specs(manifest: object) -> tuple[object, ...]:
    tensors = _manifest_value(manifest, "tensors")
    if not isinstance(tensors, (tuple, list)):
        raise Fp8FormatError("artifact manifest tensors must be a sequence")
    return tuple(tensors)


def _spec_value(spec: object, name: str) -> object:
    if isinstance(spec, Mapping):
        if name not in spec:
            raise Fp8FormatError(f"tensor spec is missing {name!r}")
        return spec[name]
    if not hasattr(spec, name):
        raise Fp8FormatError(f"tensor spec is missing {name!r}")
    return getattr(spec, name)


def _state_kind(module: nn.Module, attribute: str) -> ModuleStateKind:
    if attribute in module._parameters:
        return ModuleStateKind.PARAMETER
    if attribute in module._buffers:
        if attribute in module._non_persistent_buffers_set:
            raise Fp8FormatError(f"non-persistent buffer {attribute!r} cannot enter an immutable artifact")
        return ModuleStateKind.PERSISTENT_BUFFER
    value = getattr(module, attribute, None)
    if isinstance(value, torch.Tensor):
        return ModuleStateKind.TENSOR_ATTRIBUTE
    raise Fp8FormatError(f"{type(module).__name__}.{attribute} is not registered tensor state")


def _tensor_role(kind: ModuleStateKind, *, quant_metadata: bool = False) -> str:
    if quant_metadata:
        return "quant_metadata"
    if kind is ModuleStateKind.PARAMETER:
        return "parameter"
    return "persistent_buffer"


def _persistent_named_buffers(module: nn.Module) -> list[tuple[str, torch.Tensor]]:
    result: list[tuple[str, torch.Tensor]] = []
    for path, tensor in module.named_buffers(recurse=True, remove_duplicate=False):
        owner_path, attribute = _split_state_path(path)
        owner = module.get_submodule(owner_path) if owner_path else module
        if attribute not in owner._non_persistent_buffers_set:
            result.append((path, tensor))
    return result


def _check_no_aliases(entries: list[tuple[str, torch.Tensor]]) -> None:
    names_by_object: dict[int, list[str]] = {}
    for name, tensor in entries:
        names_by_object.setdefault(id(tensor), []).append(name)
    aliases = [names for names in names_by_object.values() if len(names) > 1]
    names_by_storage: dict[tuple[str, int], list[str]] = {}
    for name, tensor in entries:
        if tensor.numel() == 0 or tensor.is_meta:
            continue
        key = (str(tensor.device), tensor.untyped_storage().data_ptr())
        names_by_storage.setdefault(key, []).append(name)
    aliases.extend(names for names in names_by_storage.values() if len(names) > 1)
    if aliases:
        raise Fp8FormatError(f"aliased transformer state is unsupported in v2: {aliases}")


def _expected_padded_size(logical: int) -> int:
    return ((logical + _CUTLASS_ALIGNMENT - 1) // _CUTLASS_ALIGNMENT) * _CUTLASS_ALIGNMENT


def _validate_final_layer(module_path: str, module: nn.Module) -> None:
    method = getattr(module, "quant_method", None)
    if not isinstance(method, _fp8_method_type()):
        raise Fp8FormatError(f"{module_path!r} does not use trusted {QUANT_METHOD_ID}; got {type(method).__name__}")
    kernel = getattr(method, "fp8_linear", None)
    if not isinstance(kernel, _cutlass_kernel_type()):
        raise Fp8FormatError(f"{module_path!r} does not use trusted {KERNEL_ID}; got {type(kernel).__name__}")
    if bool(getattr(method, "use_marlin", False)):
        raise Fp8FormatError(f"{module_path!r} selected Marlin; v2 supports Cutlass only")
    if getattr(module, "weight_block_size", None) is not None:
        raise Fp8FormatError(f"{module_path!r} is block-quantized, not per-tensor FP8")
    if getattr(module, "input_scale", None) is not None:
        raise Fp8FormatError(f"{module_path!r} has static input_scale; online dynamic activation is required")
    if not bool(getattr(module, "_already_called_process_weights_after_loading", False)):
        raise Fp8FormatError(f"{module_path!r} has not completed online FP8 finalization")

    weight = getattr(module, "weight", None)
    scale = getattr(module, "weight_scale", None)
    if not isinstance(weight, torch.Tensor) or not isinstance(scale, torch.Tensor):
        raise Fp8FormatError(f"{module_path!r} must contain weight and weight_scale tensors")
    if weight.device.type != "cpu" or weight.is_meta or scale.device.type != "cpu" or scale.is_meta:
        raise Fp8FormatError(f"{module_path!r} finalized tensors must be on CPU")
    if weight.dtype is not torch.float8_e4m3fn:
        raise Fp8FormatError(f"{module_path!r} weight dtype must be float8_e4m3fn, got {weight.dtype}")
    if scale.dtype is not torch.float32 or scale.numel() != 1:
        raise Fp8FormatError(f"{module_path!r} weight_scale must be one FP32 value")

    input_size = getattr(module, "input_size_per_partition", None)
    output_size = getattr(module, "output_size_per_partition", None)
    widths = getattr(module, "logical_widths", None)
    if not isinstance(input_size, int) or input_size <= 0:
        raise Fp8FormatError(f"{module_path!r} has invalid input_size_per_partition")
    if not isinstance(output_size, int) or output_size <= 0:
        raise Fp8FormatError(f"{module_path!r} has invalid output_size_per_partition")
    if (
        not isinstance(widths, (list, tuple))
        or not widths
        or any(not isinstance(width, int) or width <= 0 for width in widths)
    ):
        raise Fp8FormatError(f"{module_path!r} has invalid logical_widths")
    if sum(widths) != output_size:
        raise Fp8FormatError(f"{module_path!r} logical_widths do not sum to output_size_per_partition")
    expected_shape = (_expected_padded_size(input_size), _expected_padded_size(output_size))
    expected_stride = (1, expected_shape[0])
    if tuple(weight.shape) != expected_shape or tuple(weight.stride()) != expected_stride:
        raise Fp8FormatError(
            f"{module_path!r} finalized Cutlass layout mismatch: expected "
            f"shape/stride={expected_shape}/{expected_stride}, got "
            f"{tuple(weight.shape)}/{tuple(weight.stride())}"
        )
    if getattr(kernel, "logical_output_size", None) != output_size:
        raise Fp8FormatError(f"{module_path!r} Cutlass logical_output_size does not match the logical layer width")


def _validate_consumer_layer(module_path: str, module: nn.Module, spec: LayerFormatSpec) -> None:
    if spec.quant_method_id != QUANT_METHOD_ID or spec.kernel_id != KERNEL_ID:
        raise Fp8FormatError(
            f"{module_path!r} recipe selects unsupported method/kernel {spec.quant_method_id}/{spec.kernel_id}"
        )
    method = getattr(module, "quant_method", None)
    if not isinstance(method, _fp8_method_type()):
        raise Fp8FormatError(
            f"{module_path!r} consumer did not construct {QUANT_METHOD_ID}; got {type(method).__name__}"
        )
    kernel = getattr(method, "fp8_linear", None)
    if not isinstance(kernel, _cutlass_kernel_type()):
        raise Fp8FormatError(f"{module_path!r} consumer selected {type(kernel).__name__}, not {KERNEL_ID}")

    bindings_by_role = {
        role: [binding for binding in spec.tensor_bindings if binding.role is role] for role in FormatTensorRole
    }
    weight_binding = bindings_by_role[FormatTensorRole.WEIGHT][0]
    scale_binding = bindings_by_role[FormatTensorRole.WEIGHT_SCALE][0]
    bias_binding = bindings_by_role[FormatTensorRole.BIAS][0]
    expected_bindings = (
        (weight_binding, "weight"),
        (scale_binding, "weight_scale"),
        (bias_binding, "bias"),
    )
    for binding, expected_attribute in expected_bindings:
        destination = binding.destination
        if (
            destination.attribute_name != expected_attribute
            or destination.state_kind is not ModuleStateKind.PARAMETER
            or str(destination.module_path) != str(spec.module_path)
        ):
            raise Fp8FormatError(
                f"{module_path!r} {binding.role.value} binding must target the {expected_attribute!r} parameter"
            )
        if binding.tensor_id is not None and binding.tensor_id != _join_path(module_path, expected_attribute):
            raise Fp8FormatError(f"{module_path!r} {binding.role.value} tensor ID does not match its destination")
    auxiliary_bindings = bindings_by_role[FormatTensorRole.IMMUTABLE_AUXILIARY]
    if auxiliary_bindings:
        raise Fp8FormatError(
            f"{module_path!r} immutable auxiliary tensor destinations are not allowlisted by adapter ABI v2"
        )

    if set(spec.scalar_state) != _SCALAR_KEYS:
        raise Fp8FormatError(
            f"{module_path!r} scalar-state schema mismatch: "
            f"missing={sorted(_SCALAR_KEYS - set(spec.scalar_state))}, "
            f"extra={sorted(set(spec.scalar_state) - _SCALAR_KEYS)}"
        )
    state = spec.scalar_state
    if bool(getattr(method, "use_marlin", False)) or bool(state["use_marlin"]):
        raise Fp8FormatError(f"{module_path!r} Marlin state is forbidden by the v2 recipe")
    if state["weight_block_size"] is not None or state["input_scale"] is not None:
        raise Fp8FormatError(f"{module_path!r} is not dynamic per-tensor FP8")
    if state["finalization_complete"] is not True:
        raise Fp8FormatError(f"{module_path!r} recipe is not finalized")
    for name in ("input_size_per_partition", "output_size_per_partition", "kernel_logical_output_size"):
        if not isinstance(state[name], int) or state[name] <= 0:
            raise Fp8FormatError(f"{module_path!r} has invalid scalar {name}")
    if state["kernel_logical_output_size"] != state["output_size_per_partition"]:
        raise Fp8FormatError(f"{module_path!r} kernel and module output widths differ")
    widths = state["logical_widths"]
    if (
        not isinstance(widths, (list, tuple))
        or not widths
        or any(not isinstance(width, int) or width <= 0 for width in widths)
    ):
        raise Fp8FormatError(f"{module_path!r} has invalid logical_widths metadata")
    if sum(widths) != state["output_size_per_partition"]:
        raise Fp8FormatError(f"{module_path!r} logical_widths metadata is inconsistent")

    expected_local = {
        "activation_quant_key": str(getattr(method, "activation_quant_key", None)),
        "input_dtype": _dtype_name(getattr(method, "input_dtype", None)),
        "input_size_per_partition": getattr(module, "input_size_per_partition", None),
        "logical_widths": tuple(getattr(module, "logical_widths", ())),
        "orig_dtype": _dtype_name(getattr(module, "orig_dtype", None)),
        "out_dtype": _dtype_name(getattr(method, "out_dtype", None)),
        "output_size_per_partition": getattr(module, "output_size_per_partition", None),
        "weight_quant_key": str(getattr(method, "weight_quant_key", None)),
    }
    for key, local_value in expected_local.items():
        if state[key] != local_value:
            raise Fp8FormatError(
                f"{module_path!r} consumer structure mismatch for {key}: local={local_value!r}, recipe={state[key]!r}"
            )

    has_recipe_bias = bias_binding.tensor_id is not None
    has_local_bias = module._parameters.get("bias") is not None
    if has_recipe_bias != has_local_bias:
        raise Fp8FormatError(
            f"{module_path!r} bias schema mismatch: consumer={has_local_bias}, recipe={has_recipe_bias}"
        )
    if has_recipe_bias and not isinstance(bias_binding, RequiredLayerTensorBinding):
        raise Fp8FormatError(f"{module_path!r} present bias must use a required tensor binding")
    if not has_recipe_bias and not isinstance(bias_binding, OptionalLayerTensorBinding):
        raise Fp8FormatError(f"{module_path!r} absent bias must use an optional tensor binding")
    if not has_recipe_bias and getattr(module, bias_binding.destination.attribute_name, None) is not None:
        raise Fp8FormatError(
            f"{module_path!r} optional absent bias destination does not have its declared None default"
        )


def _validate_hydrated_layer(module_path: str, module: nn.Module, spec: LayerFormatSpec) -> None:
    """Validate final structure without requiring materialized CPU bytes."""

    _validate_consumer_layer(module_path, module, spec)
    state = spec.scalar_state
    weight = getattr(module, "weight", None)
    scale = getattr(module, "weight_scale", None)
    if not isinstance(weight, torch.Tensor) or not isinstance(scale, torch.Tensor):
        raise Fp8FormatError(f"{module_path!r} did not hydrate weight and weight_scale")
    if weight.dtype is not torch.float8_e4m3fn:
        raise Fp8FormatError(f"{module_path!r} hydrated weight is not float8_e4m3fn")
    if scale.dtype is not torch.float32 or scale.numel() != 1:
        raise Fp8FormatError(f"{module_path!r} hydrated scale is not scalar FP32")
    input_size = int(state["input_size_per_partition"])
    output_size = int(state["output_size_per_partition"])
    expected_shape = (_expected_padded_size(input_size), _expected_padded_size(output_size))
    expected_stride = (1, expected_shape[0])
    if tuple(weight.shape) != expected_shape or tuple(weight.stride()) != expected_stride:
        raise Fp8FormatError(
            f"{module_path!r} hydrated Cutlass layout mismatch: {tuple(weight.shape)}/{tuple(weight.stride())}"
        )
    method = module.quant_method
    if method.fp8_linear.logical_output_size != output_size:
        raise Fp8FormatError(f"{module_path!r} did not restore Cutlass logical_output_size")
    if not bool(getattr(module, "_already_called_process_weights_after_loading", False)):
        raise Fp8FormatError(f"{module_path!r} did not restore the finalization marker")


class Fp8PerTensorFormatAdapter:
    """Export and restore the exact finalized Cutlass online-FP8 layout.

    Artifact metadata is never used to import Python objects.  The consumer
    must already contain the locally selected upstream method and kernel; this
    adapter only validates them and restores allowlisted scalar state.
    """

    def __init__(self, descriptor: object | None = None) -> None:
        self._descriptor = descriptor if descriptor is not None else _default_descriptor()
        descriptor_mapping = _descriptor_mapping(self._descriptor)
        if descriptor_mapping["format_id"] != FORMAT_ID:
            raise Fp8FormatError(f"unsupported format_id {descriptor_mapping['format_id']!r}")
        if str(descriptor_mapping["adapter_abi"]) != FORMAT_ADAPTER_ABI:
            raise Fp8FormatError(f"unsupported adapter ABI {descriptor_mapping['adapter_abi']!r}")
        if int(descriptor_mapping["format_recipe_schema_version"]) != FORMAT_RECIPE_SCHEMA_VERSION:
            raise Fp8FormatError("weight-format descriptor recipe schema does not match the FP8 adapter")
        if str(descriptor_mapping["target_module_type_id"]) != TARGET_MODULE_TYPE_ID:
            raise Fp8FormatError("weight-format descriptor target module type does not match the FP8 adapter")
        self._descriptor_digest = canonical_digest(descriptor_mapping)
        self._format_plan_digest = str(descriptor_mapping["format_plan_digest"])

    @property
    def descriptor(self) -> object:
        return self._descriptor

    @property
    def descriptor_digest(self) -> str:
        """Diagnostic digest of the complete descriptor, not recipe identity."""

        return self._descriptor_digest

    @property
    def format_plan_digest(self) -> str:
        return self._format_plan_digest

    def finalize_for_artifact(self, loaded_module: nn.Module) -> FinalizedTensorSet:
        if not isinstance(loaded_module, nn.Module):
            raise Fp8FormatError("loaded_module must be an nn.Module")

        all_state = list(loaded_module.named_parameters(recurse=True, remove_duplicate=False))
        all_state.extend(_persistent_named_buffers(loaded_module))
        _check_no_aliases(all_state)
        state_by_path = dict(all_state)

        layers: list[LayerFormatSpec] = []
        finalized: list[FinalizedTensor] = []
        owned_tensor_ids: set[str] = set()
        method_type = _fp8_method_type()
        for module_path, module in sorted(loaded_module.named_modules(), key=lambda item: item[0]):
            method = getattr(module, "quant_method", None)
            if not isinstance(method, method_type):
                if method is not None and ("Fp8" in type(method).__name__ or "FP8" in type(method).__name__):
                    raise Fp8FormatError(f"{module_path!r} uses unsupported FP8 method {type(method).__name__}")
                continue
            _validate_final_layer(module_path, module)
            destination_module_path = _target_path(module_path)
            bindings: list[LayerTensorBinding] = []
            direct_state: list[tuple[str, torch.Tensor]] = [
                (name, tensor) for name, tensor in module._parameters.items() if tensor is not None
            ]
            direct_state.extend(
                (name, tensor)
                for name, tensor in module._buffers.items()
                if tensor is not None and name not in module._non_persistent_buffers_set
            )
            for attribute, tensor in sorted(direct_state):
                tensor_id = _join_path(module_path, attribute)
                if attribute == "weight":
                    role = FormatTensorRole.WEIGHT
                elif attribute == "weight_scale":
                    role = FormatTensorRole.WEIGHT_SCALE
                elif attribute == "bias":
                    role = FormatTensorRole.BIAS
                else:
                    raise Fp8FormatError(
                        f"{module_path!r} tensor destination {attribute!r} is not allowlisted by adapter ABI v2"
                    )
                kind = _state_kind(module, attribute)
                bindings.append(
                    RequiredLayerTensorBinding(
                        role=role,
                        destination=BindingDestination(
                            module_path=destination_module_path,
                            attribute_name=attribute,
                            state_kind=kind,
                        ),
                        tensor_id=tensor_id,
                    )
                )
                finalized.append(
                    FinalizedTensor(
                        tensor_id=tensor_id,
                        role=_tensor_role(kind, quant_metadata=role is FormatTensorRole.WEIGHT_SCALE),  # type: ignore[arg-type]
                        tensor=tensor.detach(),
                    )
                )
                owned_tensor_ids.add(tensor_id)

            if not any(binding.role is FormatTensorRole.BIAS for binding in bindings):
                bindings.append(
                    OptionalLayerTensorBinding(
                        role=FormatTensorRole.BIAS,
                        destination=BindingDestination(
                            module_path=destination_module_path,
                            attribute_name="bias",
                            state_kind=ModuleStateKind.PARAMETER,
                        ),
                        tensor_id=None,
                    )
                )

            scalar_state: dict[str, JSONValue] = {
                "activation_quant_key": str(method.activation_quant_key),
                "finalization_complete": True,
                "input_dtype": _dtype_name(method.input_dtype),
                "input_scale": None,
                "input_size_per_partition": int(module.input_size_per_partition),
                "kernel_logical_output_size": int(method.fp8_linear.logical_output_size),
                "logical_widths": [int(width) for width in module.logical_widths],
                "orig_dtype": _dtype_name(module.orig_dtype),
                "out_dtype": _dtype_name(method.out_dtype),
                "output_size_per_partition": int(module.output_size_per_partition),
                "use_marlin": False,
                "weight_block_size": None,
                "weight_quant_key": str(method.weight_quant_key),
            }
            layers.append(
                LayerFormatSpec(
                    module_path=destination_module_path,
                    quant_method_id=QUANT_METHOD_ID,
                    kernel_id=KERNEL_ID,
                    tensor_bindings=tuple(bindings),
                    scalar_state=scalar_state,
                )
            )

        if not layers:
            raise Fp8FormatError("target module contains no online per-tensor FP8 linear layers")

        non_layer: list[TensorBindingSpec] = []
        for tensor_id, tensor in sorted(state_by_path.items()):
            if tensor_id in owned_tensor_ids:
                continue
            owner_path, attribute = _split_state_path(tensor_id)
            owner = loaded_module.get_submodule(owner_path) if owner_path else loaded_module
            kind = _state_kind(owner, attribute)
            non_layer.append(
                TensorBindingSpec(
                    tensor_id=tensor_id,
                    destination=BindingDestination(
                        module_path=_target_path(owner_path),
                        attribute_name=attribute,
                        state_kind=kind,
                    ),
                )
            )
            finalized.append(
                FinalizedTensor(
                    tensor_id=tensor_id,
                    role=_tensor_role(kind),  # type: ignore[arg-type]
                    tensor=tensor.detach(),
                )
            )

        recipe = FormatBindingRecipe(
            schema_version=FORMAT_RECIPE_SCHEMA_VERSION,
            format_plan_digest=self.format_plan_digest,
            target_module_type_id=TARGET_MODULE_TYPE_ID,
            layers=tuple(layers),
            non_layer_bindings=tuple(non_layer),
        )
        return FinalizedTensorSet(tensors=tuple(finalized), binding_recipe=recipe)

    def parse_recipe(self, format_metadata: object) -> FormatBindingRecipe:
        if isinstance(format_metadata, str):
            recipe = FormatBindingRecipe.from_json(format_metadata)
        elif isinstance(format_metadata, Mapping):
            recipe = FormatBindingRecipe.from_dict(format_metadata)
        else:
            raise Fp8FormatError("format_metadata must be a JSON mapping or canonical JSON string")
        if recipe.format_plan_digest != self.format_plan_digest:
            raise Fp8FormatError("format recipe plan digest does not match the requested adapter")
        if recipe.schema_version != int(_descriptor_mapping(self._descriptor)["format_recipe_schema_version"]):
            raise Fp8FormatError("format recipe schema does not match the requested adapter")
        if recipe.target_module_type_id != TARGET_MODULE_TYPE_ID:
            raise Fp8FormatError(
                f"format recipe targets {recipe.target_module_type_id!r}, expected {TARGET_MODULE_TYPE_ID!r}"
            )
        return recipe

    def prepare_consumer_structure(
        self,
        target_module: nn.Module,
        manifest: object,
    ) -> FormatBindingRecipe:
        manifest_descriptor = _manifest_value(manifest, "weight_format")
        if canonical_digest(_descriptor_mapping(manifest_descriptor)) != self.descriptor_digest:
            raise Fp8FormatError("artifact weight-format descriptor does not match the requested adapter")
        recipe = self.parse_recipe(_manifest_value(manifest, "format_metadata"))

        tensor_specs = _manifest_tensor_specs(manifest)
        tensor_ids = [str(_spec_value(spec, "tensor_id")) for spec in tensor_specs]
        if len(set(tensor_ids)) != len(tensor_ids):
            raise Fp8FormatError("artifact manifest contains duplicate tensor IDs")
        if set(tensor_ids) != set(recipe.tensor_ids):
            missing = sorted(set(recipe.tensor_ids) - set(tensor_ids))
            extra = sorted(set(tensor_ids) - set(recipe.tensor_ids))
            raise Fp8FormatError(
                f"artifact tensors do not have one-to-one recipe coverage; missing={missing}, extra={extra}"
            )

        spec_by_id = {str(_spec_value(spec, "tensor_id")): spec for spec in tensor_specs}
        for layer in recipe.layers:
            module_path = _module_path(layer.module_path)
            module = target_module.get_submodule(module_path) if module_path else target_module
            _validate_consumer_layer(module_path, module, layer)
            state = layer.scalar_state
            weight_binding = next(
                binding for binding in layer.tensor_bindings if binding.role is FormatTensorRole.WEIGHT
            )
            scale_binding = next(
                binding for binding in layer.tensor_bindings if binding.role is FormatTensorRole.WEIGHT_SCALE
            )
            assert weight_binding.tensor_id is not None
            assert scale_binding.tensor_id is not None
            weight_spec = spec_by_id[weight_binding.tensor_id]
            scale_spec = spec_by_id[scale_binding.tensor_id]
            input_size = int(state["input_size_per_partition"])
            output_size = int(state["output_size_per_partition"])
            expected_shape = (_expected_padded_size(input_size), _expected_padded_size(output_size))
            expected_stride = (1, expected_shape[0])
            actual_weight = (
                str(_spec_value(weight_spec, "dtype")).removeprefix("torch."),
                tuple(int(value) for value in _spec_value(weight_spec, "shape")),  # type: ignore[arg-type]
                tuple(int(value) for value in _spec_value(weight_spec, "stride")),  # type: ignore[arg-type]
            )
            if actual_weight != ("float8_e4m3fn", expected_shape, expected_stride):
                raise Fp8FormatError(f"{layer.module_path!r} artifact Cutlass weight layout mismatch: {actual_weight}")
            scale_dtype = str(_spec_value(scale_spec, "dtype")).removeprefix("torch.")
            scale_shape = tuple(int(value) for value in _spec_value(scale_spec, "shape"))  # type: ignore[arg-type]
            scale_numel = 1
            for dimension in scale_shape:
                scale_numel *= dimension
            if scale_dtype != "float32" or scale_numel != 1:
                raise Fp8FormatError(f"{layer.module_path!r} artifact scale is not scalar FP32")

            bias_binding = next(binding for binding in layer.tensor_bindings if binding.role is FormatTensorRole.BIAS)
            if bias_binding.tensor_id is not None:
                local_bias = module._parameters.get("bias")
                assert isinstance(local_bias, torch.Tensor)
                bias_spec = spec_by_id[bias_binding.tensor_id]
                artifact_bias_layout = (
                    str(_spec_value(bias_spec, "dtype")).removeprefix("torch."),
                    tuple(int(value) for value in _spec_value(bias_spec, "shape")),  # type: ignore[arg-type]
                    tuple(int(value) for value in _spec_value(bias_spec, "stride")),  # type: ignore[arg-type]
                )
                local_bias_layout = (
                    _dtype_name(local_bias.dtype),
                    tuple(local_bias.shape),
                    tuple(local_bias.stride()),
                )
                if artifact_bias_layout != local_bias_layout:
                    raise Fp8FormatError(
                        f"{layer.module_path!r} artifact bias layout differs from the local consumer schema: "
                        f"artifact={artifact_bias_layout}, local={local_bias_layout}"
                    )
                raw_bias_role = _spec_value(bias_spec, "role")
                bias_role = getattr(raw_bias_role, "value", raw_bias_role)
                if str(bias_role) != "parameter":
                    raise Fp8FormatError(f"{layer.module_path!r} artifact bias must have parameter role")
        return recipe

    def scalar_assignments(
        self,
        target_module: nn.Module,
        recipe: FormatBindingRecipe,
    ) -> tuple[tuple[object, str, object], ...]:
        """Return allowlisted process-private state mutations for a binder."""

        assignments: list[tuple[object, str, object]] = []
        for layer in recipe.layers:
            module_path = _module_path(layer.module_path)
            module = target_module.get_submodule(module_path) if module_path else target_module
            _validate_consumer_layer(module_path, module, layer)
            method = module.quant_method
            state = layer.scalar_state
            # QuantKey objects are selected locally.  Metadata only proves the
            # selection matches; reassigning the local values restores method
            # ownership without deserializing executable objects.
            assignments.extend(
                [
                    (module, "logical_widths", list(state["logical_widths"])),
                    (module, "input_size_per_partition", int(state["input_size_per_partition"])),
                    (module, "output_size_per_partition", int(state["output_size_per_partition"])),
                    (module, "orig_dtype", _dtype_from_name(state["orig_dtype"])),
                    (module, "weight_block_size", None),
                    (module, "input_scale", None),
                    (method, "activation_quant_key", method.activation_quant_key),
                    (method, "weight_quant_key", method.weight_quant_key),
                    (method, "input_dtype", _dtype_from_name(state["input_dtype"])),
                    (method, "out_dtype", _dtype_from_name(state["out_dtype"])),
                    (method, "use_marlin", False),
                    (method.fp8_linear, "logical_output_size", int(state["kernel_logical_output_size"])),
                    (module, "_already_called_process_weights_after_loading", True),
                ]
            )
        return tuple(assignments)

    def retire_online_loader_state(
        self,
        target_module: nn.Module,
        recipe: FormatBindingRecipe,
    ) -> _OnlineLayerwiseStateRetirement:
        """Quiesce upstream online loading for recipe-owned finalized layers.

        Online FP8 construction registers each unfinalized layer in vLLM's
        global layerwise loader registry. HWR replaces those source-layout
        Parameters with finalized artifact-layout placeholders, so leaving the
        original entry active would make an ordinary component finalizer try to
        quantize the finalized layout a second time.
        """

        registry, info_type = _online_layerwise_api()
        snapshots: list[_OnlineLayerwiseStateSnapshot] = []
        seen_modules: set[int] = set()
        for layer in recipe.layers:
            module_path = _module_path(layer.module_path)
            module = target_module.get_submodule(module_path) if module_path else target_module
            if id(module) in seen_modules:
                raise Fp8FormatError(f"recipe paths alias the same online FP8 layer at {layer.module_path!r}")
            seen_modules.add(id(module))
            if module not in registry:
                raise Fp8FormatError(f"{layer.module_path!r} has no active upstream online-layerwise loader state")
            info = registry[module]
            if not isinstance(info, info_type):
                raise Fp8FormatError(
                    f"{layer.module_path!r} has incompatible upstream layerwise state {type(info).__name__}"
                )
            required_fields = (
                "kernel_tensors",
                "loaded_weights",
                "load_numel",
                "load_numel_total",
            )
            if any(not hasattr(info, field) for field in required_fields):
                raise Fp8FormatError(f"{layer.module_path!r} upstream layerwise state schema is incompatible")
            if (
                not info.can_load()
                or info.load_numel != 0
                or info.kernel_tensors is not None
                or not isinstance(info.loaded_weights, list)
                or info.loaded_weights
            ):
                raise Fp8FormatError(
                    f"{layer.module_path!r} upstream online-layerwise state is not a pristine first-load transaction"
                )
            snapshots.append(
                _OnlineLayerwiseStateSnapshot(
                    module_path=layer.module_path,
                    module=module,
                    info=info,
                )
            )

        # Capture only. The binder stores this exact transaction before
        # invoking apply(), so every subsequent registry mutation has a
        # progress-preserving rollback owner.
        return _OnlineLayerwiseStateRetirement(registry, tuple(snapshots))

    def validate_hydrated(
        self,
        target_module: nn.Module,
        recipe: FormatBindingRecipe,
    ) -> None:
        for layer in recipe.layers:
            module_path = _module_path(layer.module_path)
            module = target_module.get_submodule(module_path) if module_path else target_module
            _validate_hydrated_layer(module_path, module, layer)


__all__ = [
    "FORMAT_ADAPTER_ABI",
    "FORMAT_ID",
    "FORMAT_RECIPE_SCHEMA_VERSION",
    "Fp8FormatError",
    "Fp8PerTensorFormatAdapter",
    "KERNEL_ID",
    "QUANT_METHOD_ID",
    "TARGET_MODULE_TYPE_ID",
]
