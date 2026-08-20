# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.host_weight import (
    ParallelRanks,
    TopologyError,
    TransferMode,
    adapt_diffusion_parallel_config,
    evaluate_single_node_complete_v1,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _config(**overrides):
    values = {
        "pipeline_parallel_size": 1,
        "data_parallel_size": 4,
        "tensor_parallel_size": 1,
        "sequence_parallel_size": 1,
        "cfg_parallel_size": 1,
        "enable_expert_parallel": False,
        "use_hsdp": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_dp_replicas_share_artifact_identity_but_keep_transfer_rank() -> None:
    rank_zero = adapt_diffusion_parallel_config(
        _config(),
        ranks=ParallelRanks(data_rank=0),
        transfer_mode=TransferMode.COMPLETE,
        single_node=True,
    )
    rank_three = adapt_diffusion_parallel_config(
        _config(),
        ranks=ParallelRanks(data_rank=3),
        transfer_mode=TransferMode.COMPLETE,
        single_node=True,
    )

    assert rank_zero.artifact.identity == rank_three.artifact.identity
    assert rank_zero.transfer.data_rank == 0
    assert rank_three.transfer.data_rank == 3
    assert evaluate_single_node_complete_v1(rank_zero).supported
    assert evaluate_single_node_complete_v1(rank_three).supported


def test_tp_coordinate_changes_artifact_identity_and_is_rejected_by_v1() -> None:
    rank_zero = adapt_diffusion_parallel_config(
        _config(tensor_parallel_size=2, data_parallel_size=1),
        ranks=ParallelRanks(tensor_rank=0),
        transfer_mode=TransferMode.COMPLETE,
        single_node=True,
    )
    rank_one = adapt_diffusion_parallel_config(
        _config(tensor_parallel_size=2, data_parallel_size=1),
        ranks=ParallelRanks(tensor_rank=1),
        transfer_mode=TransferMode.COMPLETE,
        single_node=True,
    )

    assert rank_zero.artifact.identity != rank_one.artifact.identity
    decision = evaluate_single_node_complete_v1(rank_zero)
    assert not decision.supported
    assert "tensor parallelism is not supported in v1" in decision.reasons


def test_allgather_geometry_is_transfer_only_and_deferred() -> None:
    topology = adapt_diffusion_parallel_config(
        _config(),
        ranks=ParallelRanks(data_rank=2),
        transfer_mode=TransferMode.SHARD_ALLGATHER,
        single_node=True,
    )

    assert topology.transfer.collective_size == 4
    assert topology.transfer.collective_rank == 2
    assert not evaluate_single_node_complete_v1(topology).supported


def test_config_must_have_resolved_dp_size() -> None:
    with pytest.raises(TopologyError, match="data_parallel_size"):
        adapt_diffusion_parallel_config(
            _config(data_parallel_size=None),
            ranks=ParallelRanks(),
            transfer_mode=TransferMode.COMPLETE,
            single_node=True,
        )
