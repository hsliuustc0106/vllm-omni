# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-node ordered cold-build coordination for diffusion HWR.

Layer 2 owns the DP-rank policy.  The independent runtime sees only opaque
actor and launch identities through :class:`BuildAuthorization`.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from vllm_omni.host_weight_runtime import (
    ArtifactAlreadyReady,
    BuildAuthorization,
    BuilderInitialSignalState,
    BuilderStarted,
    BuilderStartFailed,
    BuilderStartPublisher,
    BuilderStartTimeout,
    BuildGateOutcome,
    BuildRole,
    canonical_digest,
)


class BuildCoordinationError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class BuilderCoordinator(Protocol):
    def publisher(
        self,
        launch_id: str,
        artifact_key: str,
    ) -> BuilderStartPublisher: ...

    def wait(
        self,
        launch_id: str,
        artifact_key: str,
        timeout_s: float,
    ) -> BuildGateOutcome: ...


def _event_to_dict(event: BuildGateOutcome) -> dict[str, object]:
    return {"type": type(event).__name__, **asdict(event)}


def _event_from_dict(value: object) -> BuildGateOutcome:
    if not isinstance(value, dict):
        raise ValueError("builder gate payload is not an object")
    event_type = value.get("type")
    fields = {key: item for key, item in value.items() if key != "type"}
    if event_type == "BuilderStarted":
        return BuilderStarted(**fields)  # type: ignore[arg-type]
    if event_type == "ArtifactAlreadyReady":
        return ArtifactAlreadyReady(**fields)  # type: ignore[arg-type]
    if event_type == "BuilderStartFailed":
        return BuilderStartFailed(**fields)  # type: ignore[arg-type]
    if event_type == "BuilderStartTimeout":
        return BuilderStartTimeout(**fields)  # type: ignore[arg-type]
    raise ValueError(f"unknown builder gate event type {event_type!r}")


class _FileBuilderStartPublisher:
    def __init__(self, launch_id: str, artifact_key: str, path: Path) -> None:
        self._launch_id = launch_id
        self._artifact_key = artifact_key
        self._path = path
        self._state = BuilderInitialSignalState.PENDING
        self._lock = threading.RLock()

    @property
    def launch_id(self) -> str:
        return self._launch_id

    @property
    def initial_signal_state(self) -> BuilderInitialSignalState:
        with self._lock:
            return self._state

    def _publish(
        self,
        event: BuildGateOutcome,
        state: BuilderInitialSignalState,
    ) -> bool:
        with self._lock:
            if self._state is not BuilderInitialSignalState.PENDING:
                return False
            if event.launch_id != self._launch_id or event.artifact_key != self._artifact_key:
                raise BuildCoordinationError(
                    "builder_signal_identity_mismatch",
                    "publisher event does not match its launch and artifact",
                )
            self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            payload = json.dumps(
                _event_to_dict(event),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{self._path.name}.",
                dir=self._path.parent,
            )
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_name, self._path)
            except BaseException:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass
                raise
            self._state = state
            return True

    def publish_started_if_pending(self, event: BuilderStarted) -> bool:
        return self._publish(event, BuilderInitialSignalState.STARTED)

    def publish_ready_if_pending(self, event: ArtifactAlreadyReady) -> bool:
        return self._publish(event, BuilderInitialSignalState.READY)

    def publish_failed_if_pending(self, event: BuilderStartFailed) -> bool:
        return self._publish(event, BuilderInitialSignalState.FAILED)


class FileBuilderCoordinator:
    """One-node gate backed by an atomically published JSON event."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self._root = Path(root) / "coordination"

    def _path(self, launch_id: str, artifact_key: str) -> Path:
        return self._root / launch_id / f"{artifact_key}.json"

    def publisher(
        self,
        launch_id: str,
        artifact_key: str,
    ) -> BuilderStartPublisher:
        return _FileBuilderStartPublisher(
            launch_id,
            artifact_key,
            self._path(launch_id, artifact_key),
        )

    def wait(
        self,
        launch_id: str,
        artifact_key: str,
        timeout_s: float,
    ) -> BuildGateOutcome:
        deadline = time.monotonic() + timeout_s
        path = self._path(launch_id, artifact_key)
        while True:
            try:
                payload = path.read_text(encoding="utf-8")
            except FileNotFoundError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return BuilderStartTimeout(launch_id, artifact_key, timeout_s)
                time.sleep(min(0.01, remaining))
                continue
            try:
                event = _event_from_dict(json.loads(payload))
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                return BuilderStartFailed(
                    launch_id,
                    artifact_key,
                    "builder_gate_corrupt",
                    f"builder gate payload is invalid: {type(exc).__name__}",
                )
            if event.launch_id != launch_id or event.artifact_key != artifact_key:
                return BuilderStartFailed(
                    launch_id,
                    artifact_key,
                    "builder_gate_identity_mismatch",
                    "builder gate returned another launch or artifact",
                )
            return event


@dataclass(frozen=True, slots=True)
class BuildComposition:
    authorization: BuildAuthorization
    publisher: BuilderStartPublisher | None
    producer_allowed: bool


def launch_id_for_first_c4(
    *,
    artifact_key: str,
    stage_id: int,
    master_port: int | None,
) -> str:
    """Derive the launch-local ID shared by v1's local MP workers."""

    parent_pid = os.getppid()
    parent_start = "unknown"
    try:
        parent_start = Path(f"/proc/{parent_pid}/stat").read_text().split()[21]
    except (OSError, IndexError):
        pass
    return canonical_digest(
        {
            "schema": 1,
            "artifact_key": artifact_key,
            "stage_id": int(stage_id),
            "master_port": master_port,
            "parent_pid": parent_pid,
            "parent_start": parent_start,
        }
    )


def compose_first_c4_build(
    *,
    mode: str,
    dp_rank: int,
    dp_size: int,
    launch_id: str,
    artifact_key: str,
    wait_timeout_s: float,
    coordinator: BuilderCoordinator,
) -> BuildComposition:
    """Map only DP rank 0 to build authority for the first C4 topology."""

    if dp_size < 1 or not 0 <= dp_rank < dp_size:
        raise BuildCoordinationError(
            "invalid_dp_coordinate",
            f"dp rank {dp_rank} is outside size {dp_size}",
        )
    actor_id = f"dp:{dp_rank}"
    builder_actor_id = "dp:0"
    if mode == "read_only":
        return BuildComposition(
            BuildAuthorization(
                BuildRole.READ_ONLY,
                actor_id,
                builder_actor_id,
                launch_id,
            ),
            None,
            False,
        )
    if mode != "read_write":
        raise BuildCoordinationError(
            "invalid_runtime_mode",
            f"ordered build does not support mode {mode!r}",
        )
    if dp_rank == 0:
        publisher = coordinator.publisher(launch_id, artifact_key)
        return BuildComposition(
            BuildAuthorization(
                BuildRole.AUTHORIZED_BUILDER,
                actor_id,
                builder_actor_id,
                launch_id,
            ),
            publisher,
            True,
        )

    gate = coordinator.wait(launch_id, artifact_key, wait_timeout_s)
    if isinstance(gate, BuilderStarted):
        return BuildComposition(
            BuildAuthorization(
                BuildRole.ORDERED_WAITER,
                actor_id,
                builder_actor_id,
                launch_id,
                gate,
            ),
            None,
            False,
        )
    if isinstance(gate, ArtifactAlreadyReady):
        return BuildComposition(
            BuildAuthorization(
                BuildRole.READ_ONLY,
                actor_id,
                builder_actor_id,
                launch_id,
            ),
            None,
            False,
        )
    if isinstance(gate, BuilderStartTimeout):
        raise BuildCoordinationError(
            "builder_start_timeout",
            f"rank 0 did not publish an initial signal within {gate.timeout_s}s",
        )
    assert isinstance(gate, BuilderStartFailed)
    raise BuildCoordinationError(gate.code, gate.detail)


__all__ = [
    "BuildComposition",
    "BuildCoordinationError",
    "BuilderCoordinator",
    "FileBuilderCoordinator",
    "compose_first_c4_build",
    "launch_id_for_first_c4",
]
