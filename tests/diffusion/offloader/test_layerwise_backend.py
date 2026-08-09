# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for LayerwiseOffloadHook and LayerWiseOffloadBackend utilities."""

import gc
import os
import socket
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DeviceMesh, DTensor, Replicate

import vllm_omni.diffusion.offloader.layerwise_backend as layerwise_backend_module
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


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _set_dist_env(*, rank: int, world_size: int, master_port: int) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)


def _cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()

    for key in ["MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"]:
        os.environ.pop(key, None)

    gc.collect()
    if current_omni_platform.is_available():
        current_omni_platform.empty_cache()
        current_omni_platform.synchronize()


@pytest.fixture(scope="module")
def dist_group():
    master_port = _find_free_port()
    _set_dist_env(rank=0, world_size=1, master_port=master_port)

    dist.init_process_group("gloo", rank=0, world_size=1)
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


def test_layerwise_hook_preserves_packed_stride(patched_offload_runtime):
    current_block = nn.Linear(2, 2)
    next_block = nn.Module()
    expected = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
    next_block.weight = nn.Parameter(expected.clone(memory_format=torch.preserve_format))
    hook = LayerwiseOffloadHook(
        next_block=next_block,
        device=torch.device("cpu"),
        stream=DummyStream(),
        pin_memory=False,
    )

    hook.initialize_hook(current_block)
    hook.prefetch_layer(non_blocking=False)

    assert next_block.weight.stride() == expected.stride() == (1, 4)
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


class _ComponentPipeline(nn.Module):
    _offload_plan = OffloadPlan(
        encoder_block_attrs={"text_encoder": ("vision.blocks", "text_model.layers")},
        on_demand_component_paths=frozenset({"text_encoder", "vae"}),
    )

    def __init__(self):
        super().__init__()
        self.transformer = _SingleBlockModel()
        self.text_encoder = _StagedEncoder()
        self.vae = _StagedVAE()


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
                layerwise_offload_components=["vae", "text_encoder"],
            )
        )

        assert csv_config.components == frozenset({"dit", "text_encoder"})
        assert sequence_config.components == frozenset({"vae", "text_encoder"})

    def test_default_keeps_all_components(self):
        config = OffloadConfig.from_od_config(_offload_od_config(enable_layerwise_offload=True))

        assert config.components == frozenset({"dit", "text_encoder", "vae"})

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

    def test_single_gpu_encoder_only_is_allowed(self):
        config = OffloadConfig.from_od_config(
            _offload_od_config(
                enable_layerwise_offload=True,
                layerwise_offload_components="text_encoder",
            )
        )

        assert config.strategy is OffloadStrategy.LAYER_WISE
        assert config.components == frozenset({"text_encoder"})
