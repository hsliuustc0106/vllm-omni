# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Unit tests for LayerwiseOffloadHook and LayerWiseOffloadBackend utilities."""

import gc
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DeviceMesh, DTensor, Replicate

import vllm_omni.diffusion.offloader.layerwise_backend as layerwise_backend_module
from tests.helpers.runtime import get_distributed_init_method
from vllm_omni.diffusion.offloader.base import OffloadConfig, OffloadStrategy
from vllm_omni.diffusion.offloader.layerwise_backend import (
    LayerWiseOffloadBackend,
    LayerwiseOffloadHook,
)
from vllm_omni.diffusion.offloader.offload_plan import OffloadPlan
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


class DummyStream:
    def wait_stream(self, _stream) -> None:
        return None

    def wait_event(self, _event) -> None:
        return None


class DummyEvent:
    def record(self, _stream) -> None:
        return None


@contextmanager
def dummy_stream(_stream):
    yield None


def _cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()

    gc.collect()
    if current_omni_platform.is_available():
        current_omni_platform.empty_cache()
        current_omni_platform.synchronize()


@pytest.fixture(scope="module")
def dist_group():
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=get_distributed_init_method())
    try:
        yield
    finally:
        _cleanup_distributed()


@pytest.fixture
def patched_offload_runtime(monkeypatch):
    monkeypatch.setattr(layerwise_backend_module.current_omni_platform, "Stream", DummyStream)
    monkeypatch.setattr(layerwise_backend_module.current_omni_platform, "Event", DummyEvent)
    monkeypatch.setattr(layerwise_backend_module.current_omni_platform, "current_stream", lambda: DummyStream())
    monkeypatch.setattr(layerwise_backend_module.current_omni_platform, "stream", dummy_stream)


class TinyBlock(nn.Module):
    def __init__(self, values: torch.Tensor):
        super().__init__()
        mesh = DeviceMesh("cpu", [0])
        dtensor = DTensor.from_local(values, mesh, [Replicate()])
        self.weight = nn.Parameter(dtensor)


def _make_values(start: float) -> torch.Tensor:
    return torch.arange(start, start + 4, dtype=torch.float32)


class TestLayerwiseOffloadHook:
    def test_dtensor_wrapper_is_preserved_across_prefetch_and_offload(self, dist_group, patched_offload_runtime):
        current_block = TinyBlock(_make_values(1.0))
        next_block = TinyBlock(_make_values(10.0))

        hook = LayerwiseOffloadHook(
            next_block=next_block,
            device=torch.device("cpu"),
            stream=DummyStream(),
            pin_memory=False,
        )

        hook.initialize_hook(current_block)

        assert isinstance(next_block.weight, DTensor)
        assert next_block.weight.to_local().is_meta
        assert next_block.weight.to_local().shape == torch.Size([4])
        assert hook.dtype_metadata[next_block.weight.dtype][0]["shape"] == torch.Size([4])

        hook.prefetch_layer(non_blocking=False)
        assert isinstance(next_block.weight, DTensor)
        assert torch.equal(next_block.weight.to_local(), _make_values(10.0))
        assert next_block.weight.to_local().shape == torch.Size([4])

        hook.offload_layer()
        assert isinstance(current_block.weight, DTensor)
        assert current_block.weight.to_local().is_meta
        assert current_block.weight.to_local().shape == torch.Size([4])
        assert not hook.is_materialized

    def test_prefetch_preserves_transposed_weight_stride(self, patched_offload_runtime):
        """Online-FP8 Cutlass weights must retain their transposed layout."""

        class StridedBlock(nn.Module):
            def __init__(self, start: float):
                super().__init__()
                base = torch.arange(start, start + 12).reshape(3, 4)
                self.weight = nn.Parameter(base.t(), requires_grad=False)

        current_block = StridedBlock(1.0)
        next_block = StridedBlock(20.0)
        expected = next_block.weight.detach().clone()
        expected_stride = next_block.weight.stride()
        assert expected_stride == (1, 4)

        hook = LayerwiseOffloadHook(
            next_block=next_block,
            device=torch.device("cpu"),
            stream=DummyStream(),
            pin_memory=False,
        )
        hook.initialize_hook(current_block)
        hook.prefetch_layer(non_blocking=False)

        assert next_block.weight.stride() == expected_stride
        assert torch.equal(next_block.weight, expected)


class _DummyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(10, 10))


class _SingleBlockModel(nn.Module):
    _layerwise_offload_blocks_attrs = ["blocks"]

    def __init__(self, num_blocks: int = 3):
        super().__init__()
        self.blocks = nn.ModuleList([_DummyBlock() for _ in range(num_blocks)])


class _MultiBlockModel(nn.Module):
    _layerwise_offload_blocks_attrs = ["transformer_blocks", "single_transformer_blocks"]

    def __init__(self, num_transformer: int = 2, num_single: int = 2):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_DummyBlock() for _ in range(num_transformer)])
        self.single_transformer_blocks = nn.ModuleList([_DummyBlock() for _ in range(num_single)])


class _EmptyBlocksModel(nn.Module):
    _layerwise_offload_blocks_attrs = ["blocks"]

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([])


class _InvalidAttrModel(nn.Module):
    _layerwise_offload_blocks_attrs = ["nonexistent_blocks", "blocks"]

    def __init__(self, num_blocks: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([_DummyBlock() for _ in range(num_blocks)])


class _DeprecatedSingleAttrModel(nn.Module):
    _layerwise_offload_blocks_attr = "blocks"

    def __init__(self, num_blocks: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([_DummyBlock() for _ in range(num_blocks)])


class _NoAttrsModel(nn.Module):
    def __init__(self, num_blocks: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([_DummyBlock() for _ in range(num_blocks)])


class TestGetBlocksFromDit:
    def test_get_blocks_from_dit_single_block_attr(self):
        model = _SingleBlockModel(num_blocks=3)
        attr_names, blocks = LayerWiseOffloadBackend.get_blocks_from_dit(model)
        assert attr_names == ["blocks"]
        assert len(blocks) == 3
        assert all(isinstance(b, _DummyBlock) for b in blocks)

    def test_get_blocks_from_dit_multi_block_attrs(self):
        model = _MultiBlockModel(num_transformer=2, num_single=3)
        attr_names, blocks = LayerWiseOffloadBackend.get_blocks_from_dit(model)
        assert set(attr_names) == {"transformer_blocks", "single_transformer_blocks"}
        assert len(blocks) == 5
        assert all(isinstance(b, _DummyBlock) for b in blocks)

    def test_get_blocks_from_dit_empty_blocks(self):
        model = _EmptyBlocksModel()
        attr_names, blocks = LayerWiseOffloadBackend.get_blocks_from_dit(model)
        assert attr_names == []
        assert blocks == []

    def test_get_blocks_from_dit_invalid_attr_name(self):
        model = _InvalidAttrModel(num_blocks=2)
        with pytest.raises(
            AttributeError,
            match="Attribute 'nonexistent_blocks' declared in _layerwise_offload_blocks_attrs does not exist",
        ):
            LayerWiseOffloadBackend.get_blocks_from_dit(model)

    def test_get_blocks_from_dit_no_attrs_defined(self):
        model = _NoAttrsModel(num_blocks=3)
        attr_names, blocks = LayerWiseOffloadBackend.get_blocks_from_dit(model)
        assert attr_names == []
        assert blocks == []

    def test_get_blocks_from_dit_deprecated_single_attr(self):
        model = _DeprecatedSingleAttrModel(num_blocks=2)
        attr_names, blocks = LayerWiseOffloadBackend.get_blocks_from_dit(model)
        assert attr_names == ["blocks"]
        assert len(blocks) == 2


class TestGetBlocksAttrNames:
    def test_get_blocks_attr_names_new_format(self):
        model = _MultiBlockModel()
        attrs = LayerWiseOffloadBackend.get_blocks_attr_names(model)
        assert attrs == ["transformer_blocks", "single_transformer_blocks"]

    def test_get_blocks_attr_names_no_attrs(self):
        model = _NoAttrsModel()
        attrs = LayerWiseOffloadBackend.get_blocks_attr_names(model)
        assert attrs == []

    def test_set_blocks_attr_names(self):
        model = _NoAttrsModel()
        LayerWiseOffloadBackend.set_blocks_attr_names(model, ["new_blocks"])
        assert hasattr(model.__class__, "_layerwise_offload_blocks_attrs")
        assert model.__class__._layerwise_offload_blocks_attrs == ["new_blocks"]


class _StagedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = nn.Module()
        self.vision.blocks = nn.ModuleList([_DummyBlock(), _DummyBlock()])
        self.text_model = nn.Module()
        self.text_model.layers = nn.ModuleList([_DummyBlock(), _DummyBlock()])
        self.load_calls = 0
        self.offload_calls = 0
        self.to_calls = 0

    def load_to_device(self):
        self.load_calls += 1

    def offload_to_cpu(self):
        self.offload_calls += 1
        for hook in getattr(self, "_omni_layerwise_hooks", []):
            hook.offload_layer()

    def to(self, *args, **kwargs):
        self.to_calls += 1
        return super().to(*args, **kwargs)


class _StagedVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2)
        self.offload_calls = 0
        self.to_calls = 0

    def offload_to_cpu(self):
        self.offload_calls += 1
        return self.to("cpu")

    def to(self, *args, **kwargs):
        self.to_calls += 1
        return super().to(*args, **kwargs)


class _PlainEncoder(nn.Module):
    """Standard encoder with no offload-specific lifecycle methods."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.block = nn.ModuleList([_DummyBlock(), _DummyBlock()])
        self.final_norm = nn.Linear(2, 2)


class _HostTableEncoder(_PlainEncoder):
    def __init__(self):
        super().__init__()
        self.shared = nn.Embedding(32, 4)


class _PlainImageEncoder(nn.Module):
    """Standard vision encoder using the Hugging Face CLIP module layout."""

    def __init__(self):
        super().__init__()
        self.vision_model = nn.Module()
        self.vision_model.encoder = nn.Module()
        self.vision_model.encoder.layers = nn.ModuleList([_DummyBlock(), _DummyBlock()])
        self.final_norm = nn.Linear(2, 2)


class _ComponentPipeline(nn.Module):
    _offload_plan = OffloadPlan(
        encoder_component_types={"text_encoder": "text_encoder"},
        encoder_block_attrs={"text_encoder": ("vision.blocks", "text_model.layers")},
        on_demand_component_paths=frozenset({"text_encoder", "vae"}),
    )

    def __init__(self):
        super().__init__()
        self.transformer = _SingleBlockModel()
        self.text_encoder = _StagedEncoder()
        self.vae = _StagedVAE()


class _LegacyComponentPipeline(nn.Module):
    """Pipeline with DiT block metadata but no auxiliary OffloadPlan."""

    def __init__(self):
        super().__init__()
        self.transformer = _SingleBlockModel()
        self.text_encoder = _StagedEncoder()
        self.vae = _StagedVAE()


class _GenericEncoderPipeline(nn.Module):
    _offload_plan = OffloadPlan(
        encoder_component_types={"text_encoder": "text_encoder"},
        encoder_block_attrs={"text_encoder": ("encoder.block",)},
    )

    def __init__(self):
        super().__init__()
        self.transformer = _SingleBlockModel()
        self.text_encoder = _PlainEncoder()


class _HostTableEncoderPipeline(nn.Module):
    _offload_plan = OffloadPlan(
        encoder_component_types={"text_encoder": "text_encoder"},
        encoder_block_attrs={"text_encoder": ("encoder.block",)},
        encoder_host_resident_table_attrs={"text_encoder": ("shared",)},
    )

    def __init__(self):
        super().__init__()
        self.transformer = _SingleBlockModel()
        self.text_encoder = _HostTableEncoder()


class _DualEncoderPipeline(nn.Module):
    _offload_plan = OffloadPlan(
        encoder_component_types={
            "text_encoder": "text_encoder",
            "image_encoder": "image_encoder",
        },
        encoder_block_attrs={
            "text_encoder": ("encoder.block",),
            "image_encoder": ("vision_model.encoder.layers",),
        },
    )

    def __init__(self):
        super().__init__()
        self.transformer = _SingleBlockModel()
        self.text_encoder = _PlainEncoder()
        self.image_encoder = _PlainImageEncoder()


class TestLayerwiseComponentSelection:
    def test_encoder_only_streams_planned_blocks(self, patched_offload_runtime):
        pipeline = _ComponentPipeline()
        backend = LayerWiseOffloadBackend(
            OffloadConfig(
                strategy=OffloadStrategy.LAYER_WISE,
                pin_cpu_memory=False,
                components=frozenset({"text_encoder"}),
            ),
            torch.device("cpu"),
        )

        backend.enable(pipeline)

        assert pipeline.text_encoder._omni_layerwise_enabled
        assert len(pipeline.text_encoder._omni_layerwise_hooks) == 4
        assert pipeline.text_encoder.offload_calls == 1
        assert pipeline.vae.offload_calls == 0
        assert pipeline.vae.to_calls == 1
        assert not hasattr(pipeline.transformer.blocks[0], "_hook_registry")
        assert backend.enabled

        backend.disable()
        assert not pipeline.text_encoder._omni_layerwise_enabled

    def test_single_gpu_dit_only_keeps_encoder_and_vae_resident(self, patched_offload_runtime):
        pipeline = _ComponentPipeline()
        backend = LayerWiseOffloadBackend(
            OffloadConfig(
                strategy=OffloadStrategy.LAYER_WISE,
                pin_cpu_memory=False,
                components=frozenset({"dit"}),
            ),
            torch.device("cpu"),
        )

        backend.enable(pipeline)

        assert not hasattr(pipeline.text_encoder, "_omni_layerwise_enabled")
        assert pipeline.text_encoder.to_calls == 1
        assert pipeline.text_encoder.offload_calls == 0
        assert pipeline.vae.to_calls == 1
        assert pipeline.vae.offload_calls == 0
        assert hasattr(pipeline.transformer.blocks[0], "_hook_registry")

        backend.disable()

    def test_encoder_only_requires_streamable_offload_plan(self, patched_offload_runtime):
        pipeline = nn.Module()
        pipeline.transformer = _SingleBlockModel()
        pipeline.text_encoder = _StagedEncoder()
        backend = LayerWiseOffloadBackend(
            OffloadConfig(
                strategy=OffloadStrategy.LAYER_WISE,
                pin_cpu_memory=False,
                components=frozenset({"text_encoder"}),
            ),
            torch.device("cpu"),
        )

        with pytest.raises(ValueError, match="None of the selected layerwise offload components"):
            backend.enable(pipeline)

    def test_declared_on_demand_component_requires_lifecycle(self, patched_offload_runtime):
        pipeline = _ComponentPipeline()
        backend = LayerWiseOffloadBackend(
            OffloadConfig(
                strategy=OffloadStrategy.LAYER_WISE,
                pin_cpu_memory=False,
                components=frozenset({"vae"}),
            ),
            torch.device("cpu"),
        )

        with pytest.raises(ValueError, match="implement load_to_device"):
            backend.enable(pipeline)

    def test_default_selection_preserves_unplanned_auxiliaries(self, patched_offload_runtime):
        pipeline = _LegacyComponentPipeline()
        backend = LayerWiseOffloadBackend(
            OffloadConfig(
                strategy=OffloadStrategy.LAYER_WISE,
                pin_cpu_memory=False,
            ),
            torch.device("cpu"),
        )

        backend.enable(pipeline)

        assert hasattr(pipeline.transformer.blocks[0], "_hook_registry")
        assert not hasattr(pipeline.text_encoder, "_omni_layerwise_enabled")
        assert pipeline.text_encoder.to_calls == 1
        assert pipeline.text_encoder.offload_calls == 0
        assert pipeline.vae.to_calls == 1
        assert pipeline.vae.offload_calls == 0

        backend.disable()

    def test_image_encoder_selector_does_not_capture_text_encoder(self, patched_offload_runtime):
        pipeline = _DualEncoderPipeline()
        backend = LayerWiseOffloadBackend(
            OffloadConfig(
                strategy=OffloadStrategy.LAYER_WISE,
                pin_cpu_memory=False,
                components=frozenset({"image_encoder"}),
            ),
            torch.device("cpu"),
        )

        backend.enable(pipeline)

        assert not hasattr(pipeline.text_encoder, "_omni_layerwise_enabled")
        assert pipeline.image_encoder._omni_layerwise_enabled
        assert all(
            block._hook_registry.get_hook("layerwise_offload") is not None
            for block in pipeline.image_encoder.vision_model.encoder.layers
        )

        backend.disable()

    def test_standard_encoder_needs_only_declared_block_paths(self, patched_offload_runtime):
        pipeline = _GenericEncoderPipeline()
        backend = LayerWiseOffloadBackend(
            OffloadConfig(
                strategy=OffloadStrategy.LAYER_WISE,
                pin_cpu_memory=False,
                components=frozenset({"text_encoder"}),
            ),
            torch.device("cpu"),
        )

        backend.enable(pipeline)

        blocks = pipeline.text_encoder.encoder.block
        assert pipeline.text_encoder._omni_layerwise_enabled
        assert all(block._hook_registry.get_hook("layerwise_offload") is not None for block in blocks)
        assert pipeline.text_encoder.final_norm.weight.numel() == 4

        backend.disable()

        assert not pipeline.text_encoder._omni_layerwise_enabled
        assert all(block._hook_registry.get_hook("layerwise_offload") is None for block in blocks)
        assert all(block.weight.numel() == 100 for block in blocks)

    def test_declared_vocab_table_stays_on_host_and_hooks_are_reentrant(self, patched_offload_runtime):
        pipeline = _HostTableEncoderPipeline()
        expected_weight = pipeline.text_encoder.shared.weight.detach().clone()
        backend = LayerWiseOffloadBackend(
            OffloadConfig(
                strategy=OffloadStrategy.LAYER_WISE,
                pin_cpu_memory=False,
                components=frozenset({"text_encoder"}),
            ),
            torch.device("cpu"),
        )

        backend.enable(pipeline)

        table = pipeline.text_encoder.shared
        assert table.weight.device.type == "cpu"
        assert len(pipeline.text_encoder._omni_host_resident_table_handles) == 2
        token_ids = torch.tensor([[1, 3, 5]])
        assert torch.equal(table(token_ids), torch.nn.functional.embedding(token_ids, expected_weight))

        backend.disable()
        assert not table._forward_pre_hooks
        assert not table._forward_hooks

        backend.enable(pipeline)
        assert len(pipeline.text_encoder._omni_host_resident_table_handles) == 2
        backend.disable()


def _offload_od_config(**overrides):
    values = {
        "enable_cpu_offload": False,
        "enable_layerwise_offload": False,
        "enable_distributed_layerwise_offload": False,
        "layerwise_offload_components": None,
        "dlo_use_allgather": True,
        "dlo_resident_layers": 0,
        "pin_cpu_memory": True,
        "parallel_config": None,
        "model": "/fake/model",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TestLayerwiseComponentConfig:
    def test_csv_and_sequence_values_are_normalized(self):
        csv_config = OffloadConfig.from_od_config(
            _offload_od_config(
                enable_layerwise_offload=True,
                layerwise_offload_components="dit, text_encoder",
            )
        )
        sequence_config = OffloadConfig.from_od_config(
            _offload_od_config(
                enable_layerwise_offload=True,
                layerwise_offload_components=["vae", "image-encoder"],
            )
        )

        assert csv_config.components == frozenset({"dit", "text_encoder"})
        assert sequence_config.components == frozenset({"vae", "image_encoder"})

    def test_omitted_selector_preserves_legacy_dit_only_behavior(self):
        config = OffloadConfig.from_od_config(_offload_od_config(enable_layerwise_offload=True))

        assert config.components == frozenset({"dit"})
        assert config.offloads("dit")
        assert not config.offloads("text_encoder")

    def test_sglang_all_and_default_aliases(self):
        all_config = OffloadConfig.from_od_config(
            _offload_od_config(
                enable_layerwise_offload=True,
                layerwise_offload_components="all,dit",
            )
        )
        default_config = OffloadConfig.from_od_config(
            _offload_od_config(
                enable_layerwise_offload=True,
                layerwise_offload_components="default",
            )
        )

        assert all_config.components == frozenset({"dit", "text_encoder", "image_encoder", "vae"})
        assert default_config.components == frozenset({"text_encoder", "image_encoder", "vae"})
        assert all_config.offloads("dit")
        plan = OffloadPlan(encoder_component_types={"mllm": "text_encoder"})
        assert all_config.offloads_encoder("mllm", plan)
        assert default_config.offloads_encoder("text_encoder_2")
        assert default_config.offloads_encoder("condition_text_encoder")
        assert default_config.offloads_encoder("image_encoder")
        assert not default_config.offloads_encoder("vision_encoder")
        vision_plan = OffloadPlan(encoder_component_types={"vision_encoder": "image_encoder"})
        assert default_config.offloads_encoder("vision_encoder", vision_plan)

    def test_unknown_component_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown layerwise offload"):
            OffloadConfig.from_od_config(
                _offload_od_config(
                    enable_layerwise_offload=True,
                    layerwise_offload_components="dit,scheduler",
                )
            )

    def test_component_filter_requires_layerwise_strategy(self):
        with pytest.raises(ValueError, match="requires layerwise"):
            OffloadConfig.from_od_config(_offload_od_config(layerwise_offload_components="text_encoder"))

    def test_distributed_offload_requires_dit(self):
        with pytest.raises(ValueError, match="requires the 'dit' component"):
            OffloadConfig.from_od_config(
                _offload_od_config(
                    enable_distributed_layerwise_offload=True,
                    layerwise_offload_components="text_encoder",
                )
            )

    def test_distributed_offload_accepts_all(self):
        config = OffloadConfig.from_od_config(
            _offload_od_config(
                enable_distributed_layerwise_offload=True,
                layerwise_offload_components="all",
            )
        )

        assert config.components == frozenset({"dit", "text_encoder", "image_encoder", "vae"})
        assert config.offloads("dit")

    def test_single_gpu_encoder_only_is_allowed(self):
        config = OffloadConfig.from_od_config(
            _offload_od_config(
                enable_layerwise_offload=True,
                layerwise_offload_components="text_encoder",
            )
        )

        assert config.strategy is OffloadStrategy.LAYER_WISE
        assert config.components == frozenset({"text_encoder"})
