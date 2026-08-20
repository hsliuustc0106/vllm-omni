# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-node rank-policy tests for the ordered cold-build gate."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from vllm_omni.diffusion.host_weight.build_coordination import (
    BuildCoordinationError,
    FileBuilderCoordinator,
    compose_first_c4_build,
)
from vllm_omni.host_weight_runtime import (
    ArtifactAlreadyReady,
    BuilderInitialSignalState,
    BuilderStarted,
    BuildRole,
)

_ARTIFACT_KEY = "a" * 64


def test_only_dp_rank_zero_receives_builder_capability(tmp_path: Path) -> None:
    coordinator = FileBuilderCoordinator(tmp_path)

    composition = compose_first_c4_build(
        mode="read_write",
        dp_rank=0,
        dp_size=4,
        launch_id="launch-1",
        artifact_key=_ARTIFACT_KEY,
        wait_timeout_s=1,
        coordinator=coordinator,
    )

    assert composition.authorization.role is BuildRole.AUTHORIZED_BUILDER
    assert composition.authorization.actor_id == "dp:0"
    assert composition.authorization.authorized_builder_actor_id == "dp:0"
    assert composition.authorization.observed_start is None
    assert composition.producer_allowed
    assert composition.publisher is not None
    assert composition.publisher.initial_signal_state is BuilderInitialSignalState.PENDING


def test_nonzero_rank_waits_for_exact_builder_started_event(tmp_path: Path) -> None:
    coordinator = FileBuilderCoordinator(tmp_path)
    builder = compose_first_c4_build(
        mode="read_write",
        dp_rank=0,
        dp_size=2,
        launch_id="launch-1",
        artifact_key=_ARTIFACT_KEY,
        wait_timeout_s=1,
        coordinator=coordinator,
    )
    assert builder.publisher is not None
    entered = threading.Event()
    result: list[object] = []

    def compose_waiter() -> None:
        entered.set()
        result.append(
            compose_first_c4_build(
                mode="read_write",
                dp_rank=1,
                dp_size=2,
                launch_id="launch-1",
                artifact_key=_ARTIFACT_KEY,
                wait_timeout_s=1,
                coordinator=coordinator,
            )
        )

    thread = threading.Thread(target=compose_waiter)
    thread.start()
    assert entered.wait(timeout=1)
    time.sleep(0.03)
    assert result == []
    started = BuilderStarted(
        "launch-1",
        _ARTIFACT_KEY,
        "lease-exact",
        "dp:0",
        1,
    )
    assert builder.publisher.publish_started_if_pending(started)
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert len(result) == 1
    waiter = result[0]
    assert waiter.authorization.role is BuildRole.ORDERED_WAITER
    assert waiter.authorization.observed_start == started
    assert waiter.publisher is None
    assert not waiter.producer_allowed


def test_ready_gate_releases_nonzero_rank_as_read_only(tmp_path: Path) -> None:
    coordinator = FileBuilderCoordinator(tmp_path)
    builder = compose_first_c4_build(
        mode="read_write",
        dp_rank=0,
        dp_size=2,
        launch_id="launch-ready",
        artifact_key=_ARTIFACT_KEY,
        wait_timeout_s=1,
        coordinator=coordinator,
    )
    assert builder.publisher is not None
    assert builder.publisher.publish_ready_if_pending(ArtifactAlreadyReady("launch-ready", _ARTIFACT_KEY))

    reader = compose_first_c4_build(
        mode="read_write",
        dp_rank=1,
        dp_size=2,
        launch_id="launch-ready",
        artifact_key=_ARTIFACT_KEY,
        wait_timeout_s=1,
        coordinator=coordinator,
    )

    assert reader.authorization.role is BuildRole.READ_ONLY
    assert reader.authorization.observed_start is None
    assert reader.publisher is None
    assert not reader.producer_allowed


def test_waiter_timeout_is_a_typed_coordination_failure(tmp_path: Path) -> None:
    coordinator = FileBuilderCoordinator(tmp_path)

    with pytest.raises(BuildCoordinationError) as raised:
        compose_first_c4_build(
            mode="read_write",
            dp_rank=1,
            dp_size=2,
            launch_id="launch-timeout",
            artifact_key=_ARTIFACT_KEY,
            wait_timeout_s=0,
            coordinator=coordinator,
        )

    assert raised.value.code == "builder_start_timeout"


def test_read_only_mode_never_constructs_a_gate_publisher(tmp_path: Path) -> None:
    coordinator = FileBuilderCoordinator(tmp_path)

    for rank in range(4):
        composition = compose_first_c4_build(
            mode="read_only",
            dp_rank=rank,
            dp_size=4,
            launch_id="launch-read-only",
            artifact_key=_ARTIFACT_KEY,
            wait_timeout_s=0,
            coordinator=coordinator,
        )
        assert composition.authorization.role is BuildRole.READ_ONLY
        assert composition.publisher is None
        assert not composition.producer_allowed
