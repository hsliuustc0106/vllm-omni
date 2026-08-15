# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from collections.abc import Collection
from dataclasses import dataclass
from enum import Enum

import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig

from .offload_plan import OffloadPlan

logger = init_logger(__name__)

DIT_COMPONENT = "dit"
TEXT_ENCODER_COMPONENT = "text_encoder"
IMAGE_ENCODER_COMPONENT = "image_encoder"
VAE_COMPONENT = "vae"
ALL_COMPONENT = "all"
DEFAULT_COMPONENT = "default"
LAYERWISE_OFFLOAD_COMPONENTS = frozenset(
    {DIT_COMPONENT, TEXT_ENCODER_COMPONENT, IMAGE_ENCODER_COMPONENT, VAE_COMPONENT}
)
LAYERWISE_OFFLOAD_SELECTORS = LAYERWISE_OFFLOAD_COMPONENTS | {ALL_COMPONENT, DEFAULT_COMPONENT}
DEFAULT_LAYERWISE_OFFLOAD_COMPONENTS = frozenset({TEXT_ENCODER_COMPONENT, IMAGE_ENCODER_COMPONENT, VAE_COMPONENT})


def parse_layerwise_offload_components(value: str | Collection[str] | None) -> frozenset[str]:
    """Normalize the public component selection into validated names."""
    if value is None:
        return LAYERWISE_OFFLOAD_COMPONENTS
    if isinstance(value, str):
        values = value.split(",")
    elif isinstance(value, Collection):
        values = list(value)
    else:
        raise TypeError(
            "layerwise_offload_components must be a comma-separated string "
            f"or a sequence of strings, got {type(value).__name__}"
        )

    if any(not isinstance(item, str) for item in values):
        raise TypeError("layerwise_offload_components entries must be strings")
    components = [item.strip().lower().replace("-", "_") for item in values]
    if not components or any(not item for item in components):
        raise ValueError("layerwise_offload_components must not be empty")
    unknown = sorted(set(components) - LAYERWISE_OFFLOAD_SELECTORS)
    if unknown:
        raise ValueError(
            "Unknown layerwise offload component(s): "
            f"{', '.join(unknown)}. Choose from: "
            f"{', '.join(sorted(LAYERWISE_OFFLOAD_SELECTORS))}"
        )
    if ALL_COMPONENT in components:
        return LAYERWISE_OFFLOAD_COMPONENTS
    normalized = set(components)
    if DEFAULT_COMPONENT in normalized:
        normalized.remove(DEFAULT_COMPONENT)
        normalized.update(DEFAULT_LAYERWISE_OFFLOAD_COMPONENTS)
    return frozenset(normalized)


def should_offload_component(od_config: OmniDiffusionConfig, component: str) -> bool:
    """Return whether an active layerwise strategy selected ``component``."""
    if component not in LAYERWISE_OFFLOAD_COMPONENTS:
        raise ValueError(f"Unknown layerwise offload component: {component}")
    active = bool(
        getattr(od_config, "enable_layerwise_offload", False)
        or getattr(od_config, "enable_distributed_layerwise_offload", False)
    )
    if not active:
        return False
    components = parse_layerwise_offload_components(getattr(od_config, "layerwise_offload_components", None))
    return component in components


class OffloadStrategy(Enum):
    """Weight-residency granularity within one diffusion stage."""

    NONE = "none"
    MODEL_LEVEL = "model_level"  # Sequential offloading between DiT and encoders
    LAYERWISE = "layerwise"  # Block-level
    DISTRIBUTED_LAYERWISE = "distributed_layerwise"  # Block-level with DP sharding + H2D/AllGather overlap


@dataclass
class OffloadConfig:
    strategy: OffloadStrategy
    pin_cpu_memory: bool = True
    use_hsdp: bool = False
    dp_size: int = 1  # derived from parallel_config, not user-configurable
    # True: add DP sharding + AllGather. False: stream the standard loader's
    # rank-local tensors (including TP-local shards) with H2D only.
    dlo_use_allgather: bool = True
    dlo_resident_layers: int = 0  # leading DiT layers kept on device
    model_path: str | None = None  # checkpoint path for mmap weight loading
    components: frozenset[str] = LAYERWISE_OFFLOAD_COMPONENTS

    def __post_init__(self) -> None:
        self.components = parse_layerwise_offload_components(self.components)

    def offloads(self, component: str) -> bool:
        return component in self.components

    def offloads_encoder(self, name: str, plan: OffloadPlan | None = None) -> bool:
        """Return whether the selector covers a discovered encoder path.

        Plans declare non-standard encoder names explicitly. The name-based
        fallback preserves compatibility with pipelines that predate OffloadPlan.
        """
        declared_component = None if plan is None else plan.encoder_component_types.get(name)
        if declared_component is not None:
            if declared_component not in {TEXT_ENCODER_COMPONENT, IMAGE_ENCODER_COMPONENT}:
                raise ValueError(f"OffloadPlan maps encoder {name!r} to unknown component {declared_component!r}")
            return self.offloads(declared_component)

        leaf_name = name.rsplit(".", 1)[-1]
        if leaf_name.startswith(TEXT_ENCODER_COMPONENT) or leaf_name.endswith(TEXT_ENCODER_COMPONENT):
            return self.offloads(TEXT_ENCODER_COMPONENT)
        if leaf_name == IMAGE_ENCODER_COMPONENT:
            return self.offloads(IMAGE_ENCODER_COMPONENT)
        return False

    @classmethod
    def from_od_config(cls, od_config: OmniDiffusionConfig) -> "OffloadConfig":
        """Extract and validate offload settings from OmniDiffusionConfig.

        Enforces mutual exclusion among the three offload strategies.
        Distributed layer-wise takes the highest priority, then layer-wise,
        then model-level.

        The ``dp_size`` is automatically derived from ``parallel_config`` —
        it is NOT a user-configurable parameter. The distributed layerwise
        offload works with whatever DP/SP parallelism is already set up.

        Args:
            od_config: OmniDiffusionConfig with offload settings

        Returns:
            OffloadConfig with validated settings
        """
        enable_cpu_offload = getattr(od_config, "enable_cpu_offload", False)
        enable_layerwise_offload = getattr(od_config, "enable_layerwise_offload", False)
        enable_distributed_layerwise_offload = getattr(od_config, "enable_distributed_layerwise_offload", False)
        pin_cpu_memory = getattr(od_config, "pin_cpu_memory", True)

        parallel_config = getattr(od_config, "parallel_config", None)
        use_hsdp = getattr(parallel_config, "use_hsdp", False) if parallel_config else False

        # Derive dp_size from parallel_config — not user-configurable.
        # The offload adapts to whatever DP/SP is already configured.
        dp_size = 1
        if parallel_config is not None:
            dp_size = getattr(parallel_config, "data_parallel_size", 1)
            # HSDP's fully_shard_degree also contributes to effective DP
            hsdp_shard_size = getattr(parallel_config, "hsdp_shard_size", -1) if use_hsdp else -1
            hsdp_replicate_size = getattr(parallel_config, "hsdp_replicate_size", 1) if use_hsdp else 1
            if use_hsdp and hsdp_shard_size > 0:
                dp_size = hsdp_shard_size * hsdp_replicate_size

            # When there is no DP but SP > 1, shard weights across SP ranks.
            # AllGather reconstructs full weights per layer; each rank then
            # computes on its SP portion of the sequence.  This gives N×
            # compute parallelism with 1/N H2D transfer, reusing the exact
            # same AllGather code path — only the process group changes.
            if dp_size <= 1:
                sp_size = getattr(parallel_config, "sequence_parallel_size", 1)
                if sp_size and sp_size > 1:
                    dp_size = sp_size

        # Determine strategy (mutual exclusion, distributed layer-wise takes priority)
        if enable_distributed_layerwise_offload:
            strategy = OffloadStrategy.DISTRIBUTED_LAYERWISE
            if enable_layerwise_offload or enable_cpu_offload:
                logger.info("Distributed layer-wise offloading takes priority, disabling other offloading strategies.")
        elif enable_layerwise_offload:
            strategy = OffloadStrategy.LAYERWISE
            if enable_cpu_offload:
                logger.info(
                    "Both model-level and layer-wise offloading enabled. "
                    "Layer-wise takes priority, disabling model-level offloading."
                )
        elif enable_cpu_offload:
            strategy = OffloadStrategy.MODEL_LEVEL
        else:
            strategy = OffloadStrategy.NONE

        raw_components = getattr(od_config, "layerwise_offload_components", None)
        components = parse_layerwise_offload_components(raw_components)
        if raw_components is not None and strategy not in {
            OffloadStrategy.LAYERWISE,
            OffloadStrategy.DISTRIBUTED_LAYERWISE,
        }:
            raise ValueError(
                "layerwise_offload_components requires layerwise or distributed layerwise offload to be enabled"
            )
        if strategy == OffloadStrategy.DISTRIBUTED_LAYERWISE and DIT_COMPONENT not in components:
            raise ValueError(
                "Distributed layerwise offload requires the 'dit' component. "
                "Use ordinary layerwise offload for encoder-only or VAE-only staging."
            )

        # With dlo_use_allgather=False, do not add another DP shard. Each rank
        # streams the tensors produced by the standard loader, which may
        # already be TP-local shards. This avoids AllGather synchronization
        # requirements (concurrent requests, dummy run skip).
        dlo_use_allgather = getattr(od_config, "dlo_use_allgather", True)
        dlo_resident_layers = int(getattr(od_config, "dlo_resident_layers", 0))
        if dlo_resident_layers < 0:
            raise ValueError(f"dlo_resident_layers must be >= 0, got {dlo_resident_layers}")
        if dlo_resident_layers and dlo_use_allgather:
            raise ValueError(
                "dlo_resident_layers currently requires --dlo-no-use-allgather so "
                "resident blocks use weights prepared by the standard TP-aware loader"
            )

        # If dlo_use_allgather=False, force dp_size=1 (each rank independent)
        if enable_distributed_layerwise_offload and not dlo_use_allgather:
            dp_size = 1
            logger.info(
                "Distributed layerwise offload: dlo_use_allgather=False, "
                "streaming standard-loader rank-local weights (no DP shard or AllGather)"
            )

        # HSDP already shards parameters into DTensors.  Running distributed
        # layerwise offload on top would shard each to_local() again, producing
        # incorrect reconstruction after AllGather.  Reject this combination.
        if enable_distributed_layerwise_offload and use_hsdp and dlo_use_allgather:
            raise ValueError(
                "Distributed layerwise offload with AllGather is incompatible with "
                "HSDP: HSDP parameters are already sharded DTensors, and the offloader "
                "would double-shard them. Use --dlo-no-use-allgather (standard-loader "
                "rank-local weights) or disable HSDP."
            )

        return cls(
            strategy=strategy,
            pin_cpu_memory=pin_cpu_memory,
            use_hsdp=use_hsdp,
            dp_size=dp_size,
            dlo_use_allgather=dlo_use_allgather,
            dlo_resident_layers=dlo_resident_layers,
            model_path=getattr(od_config, "model", None),
            components=components,
        )


class OffloadBackend(ABC):
    """Base class for CPU offload backends"""

    def __init__(self, config: OffloadConfig, device: torch.device):
        self.config = config
        self.device = device
        self.enabled = False

    @abstractmethod
    def enable(self, pipeline: nn.Module) -> None:
        """Enable offloading on the pipeline.

        Discovers modules, moves them to appropriate devices, and
        registers forward hooks for swapping/prefetching.

        Args:
            pipeline: Diffusion pipeline model (e.g., Wan22Pipeline)
        """
        raise NotImplementedError

    @abstractmethod
    def disable(self) -> None:
        """Disable offloading and cleanup resources.

        Removes all registered hooks. Does NOT move modules back to
        original devices (caller responsible for that).
        """
        raise NotImplementedError

    def is_enabled(self) -> bool:
        return self.enabled
