# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
    CutlassFP8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.marlin import (
    MarlinFP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PerTensorOnlineLinearMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTokenSym,
    kFp8StaticTensorSym,
)
from vllm.model_executor.model_loader.reload.layerwise import (
    LAYERWISE_INFO,
    finalize_layerwise_processing,
    initialize_online_processing,
)

from vllm_omni.diffusion.config import get_current_diffusion_config_or_none
from vllm_omni.diffusion.host_weight import binding as binding_module
from vllm_omni.diffusion.host_weight.binding import (
    BindingStateError,
    BindingValidationError,
    DiffusionConsumerBinder,
)
from vllm_omni.diffusion.host_weight.formats import (
    FormatBindingRecipe,
    FormatContractError,
    Fp8FormatError,
    Fp8PerTensorFormatAdapter,
    OptionalLayerTensorBinding,
    RequiredLayerTensorBinding,
    canonical_json,
)
from vllm_omni.diffusion.host_weight.skeleton import (
    MiniMaxH3TransformerSkeletonFactory,
    PipelineSkeleton,
    SkeletonError,
)
from vllm_omni.diffusion.host_weight.transfer import (
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
    allocate_host_planes,
    compute_exact_coverage_digest,
    compute_transfer_catalog_digest,
)


def _storage_numel(tensor: torch.Tensor) -> int:
    if tensor.numel() == 0:
        return 0
    return 1 + sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride(), strict=True))


def _method(*, cutlass: bool = True) -> Fp8PerTensorOnlineLinearMethod:
    method = Fp8PerTensorOnlineLinearMethod.__new__(Fp8PerTensorOnlineLinearMethod)
    method.block_quant = False
    method.use_deep_gemm = False
    method.use_marlin = not cutlass
    method.marlin_input_dtype = None
    method.weight_quant_key = kFp8StaticTensorSym
    method.activation_quant_key = kFp8DynamicTokenSym
    method.input_dtype = torch.bfloat16
    method.out_dtype = torch.bfloat16
    kernel_type = CutlassFP8ScaledMMLinearKernel if cutlass else MarlinFP8ScaledMMLinearKernel
    method.fp8_linear = kernel_type.__new__(kernel_type)
    method.fp8_linear.logical_output_size = None
    return method


class _Fp8Layer(nn.Module):
    def __init__(
        self,
        *,
        finalized: bool,
        bias: bool = True,
        cutlass: bool = True,
    ) -> None:
        super().__init__()
        self.logical_widths = [9]
        self.input_size_per_partition = 7
        self.output_size_per_partition = 9
        self.orig_dtype = torch.bfloat16
        self.weight_block_size = None
        self.input_scale = None
        self.quant_method = _method(cutlass=cutlass)
        if finalized:
            # Transpose a row-major [N, K] allocation to get Cutlass [K, N]
            # column-major stride, including the v1 16-element padding.
            weight = torch.zeros((16, 16), dtype=torch.float8_e4m3fn).t()
            self.weight = nn.Parameter(weight, requires_grad=False)
            self.weight_scale = nn.Parameter(torch.tensor(0.25), requires_grad=False)
            if bias:
                self.bias = nn.Parameter(torch.zeros(9, dtype=torch.bfloat16), requires_grad=False)
            self.quant_method.fp8_linear.logical_output_size = 9
            self._already_called_process_weights_after_loading = True
        else:
            self.weight = nn.Parameter(
                torch.empty((9, 7), device="meta", dtype=torch.bfloat16),
                requires_grad=False,
            )
            # Mirror _Fp8OnlineLinearBase.create_weights(): online loading is
            # initialized immediately after weight registration and before a
            # parent LinearBase may add bias.
            initialize_online_processing(self)
            if bias:
                self.bias = nn.Parameter(
                    torch.empty(9, device="meta", dtype=torch.bfloat16),
                    requires_grad=False,
                )


class _Transformer(nn.Module):
    def __init__(
        self,
        *,
        finalized: bool,
        bias: bool = True,
        cutlass: bool = True,
    ) -> None:
        super().__init__()
        self.proj = _Fp8Layer(
            finalized=finalized,
            bias=bias,
            cutlass=cutlass,
        )
        device = "cpu" if finalized else "meta"
        self.dense = nn.Linear(3, 2, bias=False, dtype=torch.bfloat16, device=device)
        self.register_buffer(
            "persistent",
            torch.arange(2, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "scratch",
            torch.empty(1, device=device),
            persistent=False,
        )


class _ConstructedTransformer(_Transformer):
    def __init__(self, od_config: object, quant_config: object) -> None:
        super().__init__(finalized=False)
        self.constructor_inputs = (od_config, quant_config)
        self.constructor_config_context = get_current_diffusion_config_or_none()


class _Pipeline(nn.Module):
    def __init__(self, transformer: nn.Module) -> None:
        super().__init__()
        self.transformer = transformer


@dataclass(frozen=True)
class _TensorSpec:
    tensor_id: str
    role: str
    dtype: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_numel: int
    content_digest: str = "0" * 64


@dataclass(frozen=True)
class _Manifest:
    weight_format: object
    format_metadata: object
    tensors: tuple[_TensorSpec, ...]


def _manifest(adapter: Fp8PerTensorFormatAdapter, finalized: object) -> _Manifest:
    specs = tuple(
        _TensorSpec(
            tensor_id=item.tensor_id,
            role=item.role,
            dtype=str(item.tensor.dtype).removeprefix("torch."),
            shape=tuple(item.tensor.shape),
            stride=tuple(item.tensor.stride()),
            storage_numel=_storage_numel(item.tensor),
        )
        for item in finalized.tensors
    )
    return _Manifest(
        weight_format=adapter.descriptor,
        format_metadata=finalized.format_metadata,
        tensors=specs,
    )


def _skeleton(transformer: _Transformer) -> PipelineSkeleton:
    pipeline = _Pipeline(transformer)
    return PipelineSkeleton(
        pipeline=pipeline,
        target_module_path="transformer",
        target_module=transformer,
        target_module_type_id="minimax_h3_dit/v1",
    )


def _transfer_catalog(manifest: _Manifest) -> TransferCatalog:
    by_dtype: dict[torch.dtype, list[_TensorSpec]] = {}
    for spec in manifest.tensors:
        by_dtype.setdefault(getattr(torch, spec.dtype), []).append(spec)
    planes: list[DtypePlaneSpec] = []
    for dtype, specs in by_dtype.items():
        offset = 0
        placements: list[TensorPlacement] = []
        for spec in specs:
            placements.append(
                TensorPlacement(
                    tensor_id=spec.tensor_id,
                    offset_numel=offset,
                    logical_shape=spec.shape,
                    physical_stride=spec.stride,
                    storage_numel=spec.storage_numel,
                )
            )
            offset += spec.storage_numel
        planes.append(
            DtypePlaneSpec(
                plane_id=PlaneId(f"all/{str(dtype).removeprefix('torch.')}"),
                dtype=dtype,
                storage_numel=offset,
                placements=tuple(placements),
            )
        )
    unit = TransferUnitSpec(
        unit_id="all",
        unit_kind=UnitKind.COMPONENT,
        bindings=tuple(
            TensorBindingSpec(
                tensor_id=spec.tensor_id,
                destination=BindingDestination(
                    module_path=TargetModulePath(spec.tensor_id.rpartition(".")[0] or "."),
                    attribute_name=spec.tensor_id.rpartition(".")[2],
                    state_kind=(
                        ModuleStateKind.PERSISTENT_BUFFER
                        if spec.role == "persistent_buffer"
                        else ModuleStateKind.PARAMETER
                    ),
                ),
            )
            for spec in manifest.tensors
        ),
        planes=tuple(planes),
    )
    units = (unit,)
    execution = (ModuleUnitBinding(module_path=TargetModulePath("."), unit_id=unit.unit_id),)
    unit_ids = (unit.unit_id,)
    plan = TransferPlan(
        plan_id="plan.component",
        plan_kind=TransferPlanKind.COMPONENT,
        unit_ids=unit_ids,
        execution_bindings=execution,
        exact_coverage_digest=compute_exact_coverage_digest(unit_ids, execution, units),
    )
    plans = (plan,)
    return TransferCatalog(
        artifact_compatibility_digest="compatibility",
        transfer_catalog_digest=compute_transfer_catalog_digest(units, plans),
        units=units,
        plans=plans,
    )


def _export() -> tuple[Fp8PerTensorFormatAdapter, object, _Manifest]:
    adapter = Fp8PerTensorFormatAdapter()
    finalized = adapter.finalize_for_artifact(_Transformer(finalized=True))
    return adapter, finalized, _manifest(adapter, finalized)


def test_export_is_canonical_and_has_exact_state_coverage() -> None:
    adapter, finalized, manifest = _export()

    assert {tensor.tensor_id for tensor in finalized.tensors} == {
        "dense.weight",
        "persistent",
        "proj.bias",
        "proj.weight",
        "proj.weight_scale",
    }
    assert canonical_json(finalized.format_metadata) == finalized.binding_recipe.to_json()
    assert finalized.binding_recipe.to_json() == finalized.binding_recipe.to_json()
    assert finalized.binding_recipe.format_plan_digest == adapter.format_plan_digest
    assert set(finalized.binding_recipe.tensor_ids) == {spec.tensor_id for spec in manifest.tensors}


def test_export_declares_absent_bias_without_consuming_an_artifact_tensor() -> None:
    adapter = Fp8PerTensorFormatAdapter()
    finalized = adapter.finalize_for_artifact(_Transformer(finalized=True, bias=False))
    manifest = _manifest(adapter, finalized)
    layer = finalized.binding_recipe.layers[0]
    bias_binding = next(binding for binding in layer.tensor_bindings if binding.role.value == "bias")

    assert isinstance(bias_binding, OptionalLayerTensorBinding)
    assert bias_binding.tensor_id is None
    assert "proj.bias" not in finalized.binding_recipe.tensor_ids
    assert "proj.bias" not in {tensor.tensor_id for tensor in finalized.tensors}
    assert all(
        isinstance(binding, RequiredLayerTensorBinding)
        for binding in layer.tensor_bindings
        if binding.role.value in {"weight", "weight_scale"}
    )

    recipe = adapter.prepare_consumer_structure(
        _Transformer(finalized=False, bias=False),
        manifest,
    )
    assert recipe == finalized.binding_recipe
    with pytest.raises(Fp8FormatError, match="bias schema mismatch"):
        adapter.prepare_consumer_structure(
            _Transformer(finalized=False, bias=True),
            manifest,
        )


def test_optional_absent_bias_binding_requires_the_exact_local_destination() -> None:
    adapter = Fp8PerTensorFormatAdapter()
    finalized = adapter.finalize_for_artifact(_Transformer(finalized=True, bias=False))
    manifest = _manifest(adapter, finalized)
    metadata = finalized.binding_recipe.to_dict()
    bias_binding = next(binding for binding in metadata["layers"][0]["tensor_bindings"] if binding["role"] == "bias")
    bias_binding["destination"]["attribute_name"] = "not_bias"

    with pytest.raises(Fp8FormatError, match="bias binding must target"):
        adapter.prepare_consumer_structure(
            _Transformer(finalized=False, bias=False),
            replace(manifest, format_metadata=metadata),
        )


def test_present_bias_manifest_must_match_the_local_consumer_layout() -> None:
    adapter, _, manifest = _export()
    bias_index = next(index for index, spec in enumerate(manifest.tensors) if spec.tensor_id == "proj.bias")
    tensors = list(manifest.tensors)
    tensors[bias_index] = replace(
        tensors[bias_index],
        shape=(1,),
        stride=(1,),
        storage_numel=1,
    )

    with pytest.raises(Fp8FormatError, match="bias layout differs"):
        adapter.prepare_consumer_structure(
            _Transformer(finalized=False),
            replace(manifest, tensors=tuple(tensors)),
        )


def test_recipe_rejects_unknown_fields_and_duplicate_coverage() -> None:
    _, finalized, _ = _export()
    metadata = finalized.binding_recipe.to_dict()
    metadata["unexpected"] = True
    with pytest.raises(FormatContractError, match="keys do not match"):
        FormatBindingRecipe.from_dict(metadata)

    metadata = finalized.binding_recipe.to_dict()
    metadata["non_layer_bindings"][0]["tensor_id"] = metadata["layers"][0]["tensor_bindings"][0]["tensor_id"]
    with pytest.raises(FormatContractError, match="more than once"):
        FormatBindingRecipe.from_dict(metadata)


def test_binder_hydrates_only_meta_structure_then_commits() -> None:
    adapter, finalized, manifest = _export()
    consumer = _Transformer(finalized=False)
    original_weight = consumer.proj.weight
    assert consumer.proj in LAYERWISE_INFO
    prepared = DiffusionConsumerBinder(target_module_type=_Transformer).prepare(
        _skeleton(consumer),
        manifest,
        manifest,
        adapter,
    )

    assert prepared.state == "prepared"
    assert consumer.proj.weight is original_weight
    assert not hasattr(consumer.proj, "weight_scale")
    prepared.hydrate()

    assert prepared.state == "hydrated"
    assert all(parameter.is_meta for parameter in consumer.parameters())
    assert all(buffer.is_meta for buffer in consumer.buffers())
    assert consumer.proj.weight.dtype is torch.float8_e4m3fn
    assert tuple(consumer.proj.weight.shape) == (16, 16)
    assert tuple(consumer.proj.weight.stride()) == (1, 16)
    assert consumer.proj.weight_scale.is_meta
    assert consumer.proj.quant_method.fp8_linear.logical_output_size == 9
    assert consumer.proj._already_called_process_weights_after_loading is True
    assert consumer.proj not in LAYERWISE_INFO

    catalog = _transfer_catalog(manifest)
    prepared.set_transfer_catalog(catalog)
    prepared.validate()
    controller = prepared.commit()
    assert controller.target_module is consumer
    device_planes = allocate_host_planes(catalog.unit("all"))
    device_binding = controller.bind_device("all", device_planes)
    assert all(not parameter.is_meta for parameter in consumer.parameters())
    device_binding.release(SimpleNamespace(value="placeholder"))
    assert all(parameter.is_meta for parameter in consumer.parameters())
    with pytest.raises(BindingStateError, match="committed"):
        prepared.rollback()
    with pytest.raises(BindingStateError, match="active weight-access session"):
        controller.restore_cpu()
    controller.close()


def test_hydrate_registers_snapshot_before_first_module_mutation() -> None:
    adapter, _, manifest = _export()
    consumer = _Transformer(finalized=False)
    original_weight = consumer.proj.weight
    prepared = DiffusionConsumerBinder(target_module_type=_Transformer).prepare(
        _skeleton(consumer),
        manifest,
        manifest,
        adapter,
    )
    primary = MemoryError("injected snapshot registration failure")

    class FailingSnapshotList(list):
        def append(self, _value) -> None:
            raise primary

    prepared._state_snapshots = FailingSnapshotList()

    with pytest.raises(MemoryError, match="snapshot registration failure") as exc_info:
        prepared.hydrate()

    assert exc_info.value is primary
    assert prepared.state == "rolled_back"
    assert consumer.proj.weight is original_weight
    assert not hasattr(consumer.proj, "weight_scale")


def test_device_bind_captures_snapshot_before_first_module_mutation(
    monkeypatch,
) -> None:
    adapter, _, manifest = _export()
    consumer = _Transformer(finalized=False)
    prepared = DiffusionConsumerBinder(target_module_type=_Transformer).prepare(
        _skeleton(consumer),
        manifest,
        manifest,
        adapter,
    )
    prepared.hydrate()
    prepared.set_transfer_catalog(_transfer_catalog(manifest))
    prepared.validate()
    controller = prepared.commit()
    placeholder_weight = consumer.proj.weight
    primary = MemoryError("injected snapshot construction failure")
    monkeypatch.setattr(
        binding_module,
        "_StateSnapshot",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(primary),
    )

    with pytest.raises(MemoryError, match="snapshot construction failure") as exc_info:
        controller.bind_device(
            "all",
            allocate_host_planes(_transfer_catalog(manifest).unit("all")),
        )

    assert exc_info.value is primary
    assert consumer.proj.weight is placeholder_weight
    assert consumer.proj.weight.is_meta
    assert not controller._active_units
    assert not controller._pending_snapshots
    controller.close()


def test_binding_commit_constructs_controller_before_retiring_rollback_state(
    monkeypatch,
) -> None:
    adapter, _, manifest = _export()
    consumer = _Transformer(finalized=False)
    original_weight = consumer.proj.weight
    prepared = DiffusionConsumerBinder(target_module_type=_Transformer).prepare(
        _skeleton(consumer),
        manifest,
        manifest,
        adapter,
    )
    prepared.hydrate()
    prepared.set_transfer_catalog(_transfer_catalog(manifest))
    prepared.validate()
    primary = MemoryError("injected binding-controller construction failure")
    monkeypatch.setattr(
        binding_module,
        "BindingController",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(primary),
    )

    with pytest.raises(MemoryError, match="binding-controller construction failure") as exc_info:
        prepared.commit()

    assert exc_info.value is primary
    assert prepared.state == "validated"
    prepared.rollback()
    assert prepared.state == "rolled_back"
    assert consumer.proj.weight is original_weight


def test_binding_pre_marker_failure_closes_provisional_owner_before_snapshot_rollback(
    monkeypatch,
) -> None:
    adapter, _, manifest = _export()
    consumer = _Transformer(finalized=False)
    original_weight = consumer.proj.weight
    prepared = DiffusionConsumerBinder(target_module_type=_Transformer).prepare(
        _skeleton(consumer),
        manifest,
        manifest,
        adapter,
    )
    prepared.hydrate()
    prepared.set_transfer_catalog(_transfer_catalog(manifest))
    prepared.validate()
    retirement = prepared._format_state_retirement
    assert retirement is not None
    primary = MemoryError("injected pre-retirement-marker failure")
    monkeypatch.setattr(retirement, "commit", lambda: (_ for _ in ()).throw(primary))

    with pytest.raises(MemoryError, match="pre-retirement-marker failure") as exc_info:
        prepared.commit()

    assert exc_info.value is primary
    assert prepared.commit_state is binding_module.PreparedBindingCommitState.PROVISIONAL_CONTROLLER
    controller = prepared.retained_controller
    assert controller is not None
    with pytest.raises(BindingStateError, match="prepared marker state"):
        prepared.commit()
    assert prepared.retained_controller is controller
    close_failure = RuntimeError("injected provisional-controller close failure")
    original_close = controller.close
    close_calls = 0

    def fail_close_once(mode="terminal") -> None:
        nonlocal close_calls
        close_calls += 1
        if close_calls == 1:
            raise close_failure
        original_close(mode)

    monkeypatch.setattr(controller, "close", fail_close_once)

    with pytest.raises(RuntimeError, match="provisional-controller close failure") as cleanup_exc:
        prepared.rollback()

    assert cleanup_exc.value is close_failure
    assert prepared.retained_controller is controller
    assert prepared.commit_state is binding_module.PreparedBindingCommitState.PROVISIONAL_CONTROLLER
    assert consumer.proj.weight is not original_weight

    prepared.rollback()
    prepared.rollback()
    assert close_calls == 2
    assert prepared.retained_controller is None
    assert prepared.commit_state is binding_module.PreparedBindingCommitState.ROLLED_BACK
    assert prepared.state == "rolled_back"
    assert consumer.proj.weight is original_weight


def test_binding_commit_retains_controller_if_format_commit_return_is_interrupted(
    monkeypatch,
) -> None:
    adapter, _, manifest = _export()
    consumer = _Transformer(finalized=False)
    prepared = DiffusionConsumerBinder(target_module_type=_Transformer).prepare(
        _skeleton(consumer),
        manifest,
        manifest,
        adapter,
    )
    prepared.hydrate()
    prepared.set_transfer_catalog(_transfer_catalog(manifest))
    prepared.validate()
    retirement = prepared._format_state_retirement
    assert retirement is not None
    original_commit = retirement.commit
    primary = MemoryError("injected post-format-commit interruption")

    def commit_then_interrupt() -> None:
        original_commit()
        raise primary

    monkeypatch.setattr(retirement, "commit", commit_then_interrupt)

    with pytest.raises(MemoryError, match="post-format-commit interruption") as exc_info:
        prepared.commit()

    assert exc_info.value is primary
    assert prepared.state == "committed"
    assert prepared.commit_state is binding_module.PreparedBindingCommitState.RETIREMENT_COMMITTED
    controller = prepared.retained_controller
    assert controller is not None
    controller.close()


@pytest.mark.parametrize("failure_point", ["delegate", "registration"])
def test_device_bind_publication_failure_restores_placeholders(
    monkeypatch,
    failure_point,
) -> None:
    adapter, _, manifest = _export()
    consumer = _Transformer(finalized=False)
    prepared = DiffusionConsumerBinder(target_module_type=_Transformer).prepare(
        _skeleton(consumer),
        manifest,
        manifest,
        adapter,
    )
    prepared.hydrate()
    catalog = _transfer_catalog(manifest)
    prepared.set_transfer_catalog(catalog)
    prepared.validate()
    controller = prepared.commit()
    primary = MemoryError(f"injected device-binding {failure_point} failure")

    if failure_point == "delegate":
        monkeypatch.setattr(
            binding_module,
            "_DeviceBindingDelegate",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(primary),
        )
    else:
        original_register = controller._register_device_binding

        def fail_after_registration(unit_id, delegate, tensor_ids) -> None:
            original_register(unit_id, delegate, tensor_ids)
            raise primary

        monkeypatch.setattr(controller, "_register_device_binding", fail_after_registration)

    with pytest.raises(MemoryError, match=f"device-binding {failure_point} failure") as exc_info:
        controller.bind_device("all", allocate_host_planes(catalog.unit("all")))

    assert exc_info.value is primary
    assert not controller._active_units
    assert not controller._active_tensor_ids
    assert all(parameter.is_meta for parameter in consumer.parameters())
    assert all(buffer.is_meta for buffer in consumer.buffers())
    controller.close()


def test_real_upstream_finalizer_does_not_reenter_hwr_quantization() -> None:
    adapter, _, manifest = _export()
    consumer = _Transformer(finalized=False)
    skeleton = _skeleton(consumer)
    original_info = LAYERWISE_INFO[consumer.proj]
    prepared = DiffusionConsumerBinder(target_module_type=_Transformer).prepare(
        skeleton,
        manifest,
        manifest,
        adapter,
    )
    prepared.hydrate()
    prepared.set_transfer_catalog(_transfer_catalog(manifest))
    prepared.validate()

    # This is the same real finalizer called by the ordinary component loader.
    # With the constructor's stale online entry it tries to quantize the HWR
    # FP8 placeholder again (and fails on CPU); retirement makes it see a new,
    # inactive entry instead.
    finalize_layerwise_processing(skeleton.pipeline, model_config=None)

    assert consumer.proj._already_called_process_weights_after_loading is True
    assert consumer.proj.weight.is_meta
    assert consumer.proj.weight.dtype is torch.float8_e4m3fn
    replacement_info = LAYERWISE_INFO[consumer.proj]
    assert replacement_info is not original_info
    assert not replacement_info.can_load()

    controller = prepared.commit()
    assert LAYERWISE_INFO[consumer.proj] is replacement_info
    controller.close()


def test_rollback_restores_prefinalized_meta_skeleton() -> None:
    adapter, _, manifest = _export()
    consumer = _Transformer(finalized=False)
    original_weight = consumer.proj.weight
    original_info = LAYERWISE_INFO[consumer.proj]
    skeleton = _skeleton(consumer)
    prepared = DiffusionConsumerBinder(target_module_type=_Transformer).prepare(
        skeleton,
        manifest,
        manifest,
        adapter,
    )
    prepared.hydrate()
    prepared.set_transfer_catalog(_transfer_catalog(manifest))
    prepared.validate()
    finalize_layerwise_processing(skeleton.pipeline, model_config=None)
    assert LAYERWISE_INFO[consumer.proj] is not original_info
    prepared.rollback()
    prepared.rollback()

    assert prepared.state == "rolled_back"
    assert consumer.proj.weight is original_weight
    assert not hasattr(consumer.proj, "weight_scale")
    assert consumer.proj.quant_method.fp8_linear.logical_output_size is None
    assert not hasattr(consumer.proj, "_already_called_process_weights_after_loading")
    assert LAYERWISE_INFO[consumer.proj] is original_info


def test_hydrate_fails_closed_without_constructor_layerwise_state() -> None:
    adapter, _, manifest = _export()
    consumer = _Transformer(finalized=False)
    original_weight = consumer.proj.weight
    original_info = LAYERWISE_INFO.pop(consumer.proj)
    prepared = DiffusionConsumerBinder(target_module_type=_Transformer).prepare(
        _skeleton(consumer),
        manifest,
        manifest,
        adapter,
    )
    try:
        with pytest.raises(Fp8FormatError, match="no active upstream online-layerwise"):
            prepared.hydrate()
        assert prepared.state == "rolled_back"
        assert consumer.proj.weight is original_weight
        assert not hasattr(consumer.proj, "weight_scale")
    finally:
        LAYERWISE_INFO[consumer.proj] = original_info


def test_hydrate_fails_closed_for_incompatible_layerwise_api(monkeypatch) -> None:
    from vllm.model_executor.model_loader.reload import layerwise

    adapter, _, manifest = _export()
    consumer = _Transformer(finalized=False)
    original_weight = consumer.proj.weight
    prepared = DiffusionConsumerBinder(target_module_type=_Transformer).prepare(
        _skeleton(consumer),
        manifest,
        manifest,
        adapter,
    )
    monkeypatch.setattr(layerwise, "LAYERWISE_INFO", {})

    with pytest.raises(Fp8FormatError, match="state registry is incompatible"):
        prepared.hydrate()
    assert prepared.state == "rolled_back"
    assert consumer.proj.weight is original_weight


def test_hydrate_preserves_primary_and_retries_failed_retirement(monkeypatch) -> None:
    class PrimaryError(RuntimeError):
        def add_note(self, _note: str) -> None:
            raise SystemExit("injected binding add_note failure")

    class CleanupError(RuntimeError):
        def __str__(self) -> str:
            raise SystemExit("injected retirement __str__ failure")

    adapter, _, manifest = _export()
    consumer = _Transformer(finalized=False)
    original_weight = consumer.proj.weight
    prepared = DiffusionConsumerBinder(target_module_type=_Transformer).prepare(
        _skeleton(consumer),
        manifest,
        manifest,
        adapter,
    )
    primary = PrimaryError("primary state installation failure")
    cleanup = CleanupError("retirement rollback failed")
    original_retire = adapter.retire_online_loader_state
    rollback_calls = 0

    def retire_with_fail_once(*args, **kwargs):
        retirement = original_retire(*args, **kwargs)
        original_rollback = retirement.rollback

        def fail_once() -> None:
            nonlocal rollback_calls
            rollback_calls += 1
            if rollback_calls == 1:
                raise cleanup
            original_rollback()

        retirement.rollback = fail_once
        return retirement

    monkeypatch.setattr(adapter, "retire_online_loader_state", retire_with_fail_once)
    monkeypatch.setattr(
        binding_module,
        "_install_state",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(primary),
    )

    with pytest.raises(PrimaryError) as exc_info:
        prepared.hydrate()

    assert exc_info.value is primary
    assert prepared.state == "rolling_back"
    assert prepared._format_state_retirement is not None
    prepared.rollback()
    assert rollback_calls == 2
    assert prepared.state == "rolled_back"
    assert consumer.proj.weight is original_weight


def test_prepare_fails_closed_for_manifest_layout_and_catalog_mismatch() -> None:
    adapter, _, manifest = _export()
    weight_index = next(index for index, spec in enumerate(manifest.tensors) if spec.tensor_id == "proj.weight")
    bad_weight = replace(manifest.tensors[weight_index], stride=(16, 1))
    tensors = list(manifest.tensors)
    tensors[weight_index] = bad_weight
    bad_manifest = replace(manifest, tensors=tuple(tensors))
    with pytest.raises(Fp8FormatError, match="layout mismatch"):
        DiffusionConsumerBinder(target_module_type=_Transformer).prepare(
            _skeleton(_Transformer(finalized=False)),
            bad_manifest,
            bad_manifest,
            adapter,
        )

    with pytest.raises(BindingValidationError, match="coverage differ"):
        DiffusionConsumerBinder(target_module_type=_Transformer).prepare(
            _skeleton(_Transformer(finalized=False)),
            manifest,
            SimpleNamespace(tensors=manifest.tensors[:-1]),
            adapter,
        )


def test_adapter_rejects_marlin_and_nonfinalized_exports() -> None:
    adapter = Fp8PerTensorFormatAdapter()
    with pytest.raises(Fp8FormatError, match="unsupported FP8 method|trusted Cutlass|does not use trusted"):
        adapter.finalize_for_artifact(_Transformer(finalized=True, cutlass=False))
    with pytest.raises(Fp8FormatError, match="not completed"):
        adapter.finalize_for_artifact(_Transformer(finalized=False))


def test_minimax_factory_builds_meta_target_and_injects_by_identity() -> None:
    adapter = Fp8PerTensorFormatAdapter()
    od_config = SimpleNamespace(task_type="t2va", quantization_config=object())
    built: dict[str, object] = {}

    def pipeline_factory(*, od_config: object, transformer: nn.Module) -> _Pipeline:
        built["od_config"] = od_config
        built["transformer"] = transformer
        return _Pipeline(transformer)

    factory = MiniMaxH3TransformerSkeletonFactory(
        od_config,
        quant_config=object(),
        pipeline_factory=pipeline_factory,
        transformer_type=_ConstructedTransformer,
    )
    skeleton = factory.create(
        SimpleNamespace(
            pipeline_family_id="minimax_h3",
            normalized_init_config={"partition": "fl2va", "task_type": "t2va"},
        ),
        adapter.descriptor,
    )

    assert skeleton.target_module is built["transformer"]
    assert skeleton.target_module.constructor_config_context is od_config
    assert factory.resolve_target(skeleton.pipeline, "transformer") is skeleton.target_module
    assert all(parameter.is_meta for parameter in skeleton.target_module.parameters())
    assert not hasattr(skeleton.target_module.proj, "_already_called_process_weights_after_loading")


def test_minimax_factory_fails_closed_for_ref2va_and_wrong_format() -> None:
    adapter = Fp8PerTensorFormatAdapter()
    od_config = SimpleNamespace(task_type="t2va", quantization_config=object())
    factory = MiniMaxH3TransformerSkeletonFactory(
        od_config,
        quant_config=object(),
        transformer_type=_ConstructedTransformer,
    )
    with pytest.raises(SkeletonError, match="FL2VA only"):
        factory.create(
            SimpleNamespace(
                pipeline_family_id="minimax_h3",
                normalized_init_config={"partition": "ref2va"},
            ),
            adapter.descriptor,
        )
    descriptor = adapter.descriptor
    wrong_format = descriptor.to_dict() if callable(getattr(descriptor, "to_dict", None)) else dict(descriptor)
    wrong_format["format_id"] = "dense"
    with pytest.raises(SkeletonError, match="requires weight format"):
        factory.create(
            SimpleNamespace(
                pipeline_family_id="minimax_h3",
                normalized_init_config={"partition": "fl2va"},
            ),
            wrong_format,
        )
