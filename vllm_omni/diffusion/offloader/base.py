# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from abc import ABC, abstractmethod
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig, validate_dlo_host_registration_options

from .config import (
    DIT_COMPONENT,
    DLO_COMPONENTS,
    TEXT_ENCODER_COMPONENT,
    DLOTransfer,
    parse_dlo_transfer,
)
from .offload_plan import OffloadPlan

logger = init_logger(__name__)

ALL_COMPONENT = "all"
LAYERWISE_OFFLOAD_COMPONENTS = DLO_COMPONENTS
LAYERWISE_OFFLOAD_SELECTORS = LAYERWISE_OFFLOAD_COMPONENTS | {ALL_COMPONENT}
OMITTED_LAYERWISE_OFFLOAD_COMPONENTS = frozenset({DIT_COMPONENT})


def parse_layerwise_offload_components(value: str | Collection[str] | None) -> frozenset[str]:
    """Normalize the public component selection into validated names."""
    if value is None:
        return OMITTED_LAYERWISE_OFFLOAD_COMPONENTS
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


@runtime_checkable
class SupportsModelCpuOffload(Protocol):
    """Pipeline-owned lifecycle for model-level CPU offload.

    Pipelines with non-forward component entry points (for example VAE
    ``decode_latent`` methods) need to activate those stages explicitly, so
    generic forward-hook discovery cannot manage their full lifecycle.
    """

    def enable_omni_model_cpu_offload(
        self,
        *,
        device: torch.device,
        pin_memory: bool,
        use_hsdp: bool,
    ) -> None: ...

    def disable_omni_model_cpu_offload(self) -> None: ...


class OffloadStrategy(Enum):
    NONE = "none"
    MODEL_LEVEL = "model_level"  # Sequential offloading between DiT and encoders
    LAYER_WISE = "layer_wise"  # Block-level
    DISTRIBUTED_LAYER_WISE = "distributed_layer_wise"  # Block-level with DP sharding + H2D/AllGather overlap


@dataclass
class OffloadConfig:
    strategy: OffloadStrategy
    pin_cpu_memory: bool = True
    use_hsdp: bool = False
    dp_size: int = 1  # derived from parallel_config, not user-configurable
    # True: add DP sharding + AllGather. False: stream complete rank-local
    # blocks from the loader-selected host backing with H2D only.
    dlo_use_allgather: bool = True
    dlo_resident_layers: int = 0  # leading DiT layers kept on device
    # Optional per-worker ceiling for registering an HWR mmap. Zero means no
    # additional ceiling; pin_cpu_memory controls whether registration is tried.
    dlo_host_registration_limit_gib: float = 0.0
    components: frozenset[str] = OMITTED_LAYERWISE_OFFLOAD_COMPONENTS
    dlo_transfers: dict[str, DLOTransfer] | None = None

    def __post_init__(self) -> None:
        self.components = parse_layerwise_offload_components(self.components)
        self.dlo_transfers = parse_dlo_transfer(
            self.dlo_transfers,
            legacy_use_allgather=self.dlo_use_allgather,
        )
        # Preserve the old field as the DiT transfer compatibility view.
        self.dlo_use_allgather = self.uses_allgather(DIT_COMPONENT)

    def offloads(self, component: str) -> bool:
        return component in self.components

    def transfer_for(self, component: str) -> DLOTransfer:
        if self.dlo_transfers is None:
            raise RuntimeError("DLO transfers were not initialized")
        try:
            return self.dlo_transfers[component]
        except KeyError as exc:
            raise ValueError(f"Unknown DLO component {component!r}") from exc

    def uses_allgather(self, component: str) -> bool:
        return self.transfer_for(component) is DLOTransfer.ALLGATHER

    def offloads_encoder(self, name: str, plan: OffloadPlan | None = None) -> bool:
        """Return whether the selector covers a discovered encoder path.

        Plans declare non-standard encoder names explicitly. The name-based
        fallback preserves compatibility with pipelines that predate OffloadPlan.
        """
        declared_component = None if plan is None else plan.encoder_component_types.get(name)
        if declared_component is not None:
            if declared_component != TEXT_ENCODER_COMPONENT:
                raise ValueError(f"OffloadPlan maps encoder {name!r} to unknown component {declared_component!r}")
            return self.offloads(declared_component)

        leaf_name = name.rsplit(".", 1)[-1]
        if leaf_name.startswith(TEXT_ENCODER_COMPONENT) or leaf_name.endswith(TEXT_ENCODER_COMPONENT):
            return self.offloads(TEXT_ENCODER_COMPONENT)
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
            # HSDP shard and replica sizes determine the effective group size.
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
            strategy = OffloadStrategy.DISTRIBUTED_LAYER_WISE
            if enable_layerwise_offload or enable_cpu_offload:
                logger.info("Distributed layer-wise offloading takes priority, disabling other offloading strategies.")
        elif enable_layerwise_offload:
            strategy = OffloadStrategy.LAYER_WISE
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
            OffloadStrategy.LAYER_WISE,
            OffloadStrategy.DISTRIBUTED_LAYER_WISE,
        }:
            raise ValueError(
                "layerwise_offload_components requires layerwise or distributed layerwise offload to be enabled"
            )
        dlo_use_allgather = getattr(od_config, "dlo_use_allgather", True)
        raw_dlo_transfer = getattr(od_config, "dlo_transfer", None)
        dlo_transfers = parse_dlo_transfer(
            raw_dlo_transfer,
            legacy_use_allgather=dlo_use_allgather,
        )
        if raw_dlo_transfer is not None and strategy != OffloadStrategy.DISTRIBUTED_LAYER_WISE:
            raise ValueError("dlo_transfer requires distributed layerwise offload to be enabled")
        if isinstance(raw_dlo_transfer, Mapping):
            explicitly_configured = {str(component).strip().lower().replace("-", "_") for component in raw_dlo_transfer}
        elif isinstance(raw_dlo_transfer, str) and "=" in raw_dlo_transfer:
            explicitly_configured = {
                item.partition("=")[0].strip().lower().replace("-", "_") for item in raw_dlo_transfer.split(",")
            }
        else:
            explicitly_configured = set()
        unused_transfer_components = sorted((explicitly_configured - {ALL_COMPONENT}) - components)
        if unused_transfer_components:
            raise ValueError(
                "dlo_transfer configures component(s) not selected for offload: "
                f"{', '.join(unused_transfer_components)}"
            )

        dlo_resident_layers = int(getattr(od_config, "dlo_resident_layers", 0))
        dit_uses_allgather = dlo_transfers[DIT_COMPONENT] is DLOTransfer.ALLGATHER
        dlo_host_registration_limit_gib = validate_dlo_host_registration_options(
            limit_gib=getattr(od_config, "dlo_host_registration_limit_gib", 0.0),
            enable_dlo=enable_distributed_layerwise_offload,
            use_allgather=dit_uses_allgather,
            hwr_mode=getattr(od_config, "host_weight_runtime_mode", "disabled"),
        )
        if dlo_resident_layers < 0:
            raise ValueError(f"dlo_resident_layers must be >= 0, got {dlo_resident_layers}")
        if dlo_resident_layers and DIT_COMPONENT not in components:
            raise ValueError("dlo_resident_layers requires the 'dit' component to be selected")
        if dlo_resident_layers and dit_uses_allgather:
            raise ValueError(
                "dlo_resident_layers requires the DiT DLO transfer to be rank-local so "
                "resident blocks use weights prepared by the standard TP-aware loader"
            )

        if enable_distributed_layerwise_offload and all(
            transfer is DLOTransfer.RANK_LOCAL
            for component, transfer in dlo_transfers.items()
            if component in components
        ):
            logger.info(
                "Distributed layerwise offload: all selected components use "
                "rank-local transfer (no DLO shard or AllGather)"
            )

        # HSDP already shards parameters into DTensors.  Running distributed
        # layerwise offload on top would shard each to_local() again, producing
        # incorrect reconstruction after AllGather.  Reject this combination.
        if (
            enable_distributed_layerwise_offload
            and use_hsdp
            and any(dlo_transfers[component] is DLOTransfer.ALLGATHER for component in components)
        ):
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
            dlo_use_allgather=dit_uses_allgather,
            dlo_resident_layers=dlo_resident_layers,
            dlo_host_registration_limit_gib=dlo_host_registration_limit_gib,
            components=components,
            dlo_transfers=dlo_transfers,
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
