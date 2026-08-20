# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Translation from diffusion parallelism into artifact and transfer topology."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class TopologyError(ValueError):
    pass


class TransferMode(str, Enum):
    COMPLETE = "complete"
    SHARD_ALLGATHER = "shard_allgather"


@dataclass(frozen=True)
class ParallelRanks:
    pipeline_rank: int = 0
    data_rank: int = 0
    tensor_rank: int = 0
    sequence_rank: int = 0
    cfg_rank: int = 0


@dataclass(frozen=True)
class ArtifactTopology:
    """Only coordinates that can change final runtime bytes or ownership."""

    pipeline_size: int
    pipeline_rank: int
    tensor_size: int
    tensor_rank: int
    hsdp_enabled: bool
    hsdp_shard_size: int | None
    hsdp_replicate_size: int | None

    @property
    def identity(self) -> tuple[object, ...]:
        return (
            "pp",
            self.pipeline_size,
            self.pipeline_rank,
            "tp",
            self.tensor_size,
            self.tensor_rank,
            "hsdp",
            self.hsdp_enabled,
            self.hsdp_shard_size,
            self.hsdp_replicate_size,
        )


@dataclass(frozen=True)
class TransferTopology:
    """Replica and collective geometry used only when compiling copy plans."""

    mode: TransferMode
    data_size: int
    data_rank: int
    sequence_size: int
    sequence_rank: int
    collective_size: int
    collective_rank: int


@dataclass(frozen=True)
class RuntimeTopology:
    artifact: ArtifactTopology
    transfer: TransferTopology
    single_node: bool
    cfg_size: int
    cfg_rank: int
    expert_parallel: bool


@dataclass(frozen=True)
class TopologySupportDecision:
    reasons: tuple[str, ...] = ()

    @property
    def supported(self) -> bool:
        return not self.reasons


def _positive_size(config: Any, name: str, *, allow_none: bool = False) -> int:
    value = getattr(config, name, None)
    if value is None and allow_none:
        return 1
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise TopologyError(f"{name} must be a resolved positive integer, got {value!r}")
    return value


def _validate_rank(name: str, rank: int, size: int) -> None:
    if not isinstance(rank, int) or isinstance(rank, bool) or not 0 <= rank < size:
        raise TopologyError(f"{name} must be in [0, {size}), got {rank!r}")


def adapt_diffusion_parallel_config(
    config: Any,
    *,
    ranks: ParallelRanks,
    transfer_mode: TransferMode,
    single_node: bool,
) -> RuntimeTopology:
    """Split vLLM-Omni configuration into semantic and transfer topology.

    ``config`` is intentionally duck-typed so the generic host-weight package
    does not become coupled to ``DiffusionParallelConfig`` imports.
    """
    pipeline_size = _positive_size(config, "pipeline_parallel_size")
    data_size = _positive_size(config, "data_parallel_size")
    tensor_size = _positive_size(config, "tensor_parallel_size")
    sequence_size = _positive_size(config, "sequence_parallel_size")
    cfg_size = _positive_size(config, "cfg_parallel_size", allow_none=True)

    _validate_rank("pipeline_rank", ranks.pipeline_rank, pipeline_size)
    _validate_rank("data_rank", ranks.data_rank, data_size)
    _validate_rank("tensor_rank", ranks.tensor_rank, tensor_size)
    _validate_rank("sequence_rank", ranks.sequence_rank, sequence_size)
    _validate_rank("cfg_rank", ranks.cfg_rank, cfg_size)

    hsdp_enabled = bool(getattr(config, "use_hsdp", False))
    hsdp_shard_size = _positive_size(config, "hsdp_shard_size") if hsdp_enabled else None
    hsdp_replicate_size = _positive_size(config, "hsdp_replicate_size") if hsdp_enabled else None

    if transfer_mode is TransferMode.SHARD_ALLGATHER:
        # Match today's DLO rule: DP is the weight-sharing group when present;
        # SP is used only for the standalone-SP case.
        if data_size > 1:
            collective_size, collective_rank = data_size, ranks.data_rank
        else:
            collective_size, collective_rank = sequence_size, ranks.sequence_rank
    else:
        collective_size, collective_rank = 1, 0

    return RuntimeTopology(
        artifact=ArtifactTopology(
            pipeline_size=pipeline_size,
            pipeline_rank=ranks.pipeline_rank,
            tensor_size=tensor_size,
            tensor_rank=ranks.tensor_rank,
            hsdp_enabled=hsdp_enabled,
            hsdp_shard_size=hsdp_shard_size,
            hsdp_replicate_size=hsdp_replicate_size,
        ),
        transfer=TransferTopology(
            mode=transfer_mode,
            data_size=data_size,
            data_rank=ranks.data_rank,
            sequence_size=sequence_size,
            sequence_rank=ranks.sequence_rank,
            collective_size=collective_size,
            collective_rank=collective_rank,
        ),
        single_node=single_node,
        cfg_size=cfg_size,
        cfg_rank=ranks.cfg_rank,
        expert_parallel=bool(getattr(config, "enable_expert_parallel", False)),
    )


def evaluate_single_node_complete_v1(topology: RuntimeTopology) -> TopologySupportDecision:
    """Fail-closed capability envelope for the first host-weight use case."""
    reasons: list[str] = []
    if not topology.single_node:
        reasons.append("v1 host weights are restricted to one node")
    if topology.artifact.pipeline_size != 1:
        reasons.append("pipeline parallelism is not supported in v1")
    if topology.artifact.tensor_size != 1:
        reasons.append("tensor parallelism is not supported in v1")
    if topology.transfer.sequence_size != 1:
        reasons.append("sequence parallelism is not supported in v1")
    if topology.artifact.hsdp_enabled:
        reasons.append("HSDP/DTensor ownership is not supported in v1")
    if topology.transfer.mode is not TransferMode.COMPLETE:
        reasons.append("v1 supports complete-block copies only; DLO AllGather is deferred")
    if topology.cfg_size != 1:
        reasons.append("CFG parallelism is not supported in v1")
    if topology.expert_parallel:
        reasons.append("expert parallelism is not supported in v1")
    return TopologySupportDecision(reasons=tuple(reasons))


__all__ = [
    "ArtifactTopology",
    "ParallelRanks",
    "RuntimeTopology",
    "TopologyError",
    "TopologySupportDecision",
    "TransferMode",
    "TransferTopology",
    "adapt_diffusion_parallel_config",
    "evaluate_single_node_complete_v1",
]
