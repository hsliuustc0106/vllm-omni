# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-launch MiniMax-H3 Host Weight Runtime qualification recorder.

Use ``host_weight_runtime_qualify.py`` for the complete alternating L4/C4/W4
suite.  This child recorder deliberately owns no cache-eviction, cgroup
creation, or artifact-generation policy: it verifies the launch controls
provided by the suite, records exact worker-scoped memory samples, performs
warmups, and executes measured requests sequentially.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import multiprocessing
import os
import re
import statistics
import subprocess
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from vllm_omni.entrypoints.async_omni import AsyncOmni

PROMPT = (
    "At night, three cats march into a bedroom playing tiny brass instruments, "
    "then abruptly file out, with synchronized room ambience."
)

# Standalone evaluator copy of the integration wire-schema version. A stale
# worker payload must fail before any field-level qualification is attempted.
HWR_RESOLUTION_EVIDENCE_SCHEMA_VERSION = 2


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _minimax_steps(value: str) -> int:
    parsed = int(value)
    if parsed < 2:
        raise argparse.ArgumentTypeError("MiniMax-H3 requires at least 2 inference steps")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return parsed


def _minimax_duration(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or not 4.0 <= parsed <= 15.0:
        raise argparse.ArgumentTypeError("MiniMax-H3 duration must be in [4, 15] seconds")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--hwr-root", required=True, type=Path)
    parser.add_argument(
        "--hwr-mode",
        required=True,
        choices=("disabled", "read_write", "read_only"),
    )
    parser.add_argument(
        "--offloader",
        required=True,
        choices=("model", "layerwise", "dlo-no-allgather"),
    )
    parser.add_argument("--dp-size", type=_positive, default=4)
    parser.add_argument(
        "--client-requests",
        type=_positive,
        default=None,
        help=("Deprecated alias for --measured-requests. Requests are always submitted sequentially."),
    )
    parser.add_argument("--warmup-requests", type=_positive, default=2)
    parser.add_argument("--measured-requests", type=_positive, default=10)
    parser.add_argument("--settle-s", type=_nonnegative_float, default=5.0)
    parser.add_argument("--idle-window-s", type=_positive_float, default=20.0)
    parser.add_argument("--sample-interval-s", type=_positive_float, default=0.1)
    parser.add_argument("--steps", type=_minimax_steps, default=2)
    parser.add_argument("--duration", type=_minimax_duration, default=4.0)
    parser.add_argument("--width", type=_positive, default=448)
    parser.add_argument("--height", type=_positive, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--hwr-wait-timeout-s",
        type=_positive_float,
        default=3600.0,
        help="Maximum cold-build wait per non-builder worker.",
    )
    parser.add_argument(
        "--process-exit-timeout-s",
        type=_positive_float,
        default=5.0,
        help=(
            "Maximum post-close wait for every observed engine process to exit. Survivors still fail the cleanup gate."
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--tensor-output-dir",
        type=Path,
        help="Archive measured video/audio arrays for cross-scenario tolerance checks.",
    )
    parser.add_argument(
        "--scenario",
        choices=("L4", "C4", "W4"),
        help="Suite scenario identity; when set, mode/state invariants are strict.",
    )
    parser.add_argument(
        "--launch-id",
        help="Opaque suite launch ID recorded in cache/cgroup attestations.",
    )
    parser.add_argument(
        "--measurement-cgroup",
        type=Path,
        help="Fresh cgroup created by the suite and already joined by this process.",
    )
    parser.add_argument(
        "--cache-attestation",
        type=Path,
        help="Suite-written JSON attestation for the immediately preceding eviction.",
    )
    parser.add_argument(
        "--formal-runtime-contract",
        action="store_true",
        help="Require ordered-build, grant-bound, PRE_RESOLVE, and idle-state evidence.",
    )
    parser.add_argument(
        "--poison-warm-build",
        action="store_true",
        help="Fail if a warm run opens a producer or transformer ComponentSource",
    )
    parser.add_argument(
        "--external-cold-cache-confirmed",
        action="store_true",
        help="Assert an external controller evicted this artifact's pages",
    )
    parser.add_argument(
        "--external-isolated-cgroup-confirmed",
        action="store_true",
        help="Assert this process runs in a fresh writable measurement cgroup",
    )
    parser.add_argument(
        "--allow-nonexclusive-gpus",
        action="store_true",
        help=(
            "Allow a diagnostic run when another compute process uses a selected "
            "GPU. Whole-device memory metrics are then disqualified."
        ),
    )
    return parser.parse_args()


def _engine_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": args.model,
        # Pin the first use case to the FL2VA transformer.  A repository or
        # snapshot root otherwise selects MiniMax's combined partition, which
        # is deliberately outside the v1 artifact contract.
        "task_type": "t2va",
        "trust_remote_code": True,
        "num_gpus": args.dp_size,
        # MiniMax-H3 does not support request-level batching in these modes.
        # Pin the scheduler contract instead of presenting client submission as
        # DP request concurrency.
        "max_num_seqs": 1,
        "tensor_parallel_size": 1,
        "data_parallel_size": args.dp_size,
        "pipeline_parallel_size": 1,
        "usp": 1,
        "ring": 1,
        "text_encoder_tp_size": 1,
        "vae_patch_parallel_size": 1,
        "vae_parallel_mode": "tile",
        "vae_use_tiling": True,
        "diffusion_attention_backend": "CUDNN_ATTN",
        "diffusion_quantization_config": {
            "transformer": {"method": "fp8"},
        },
        "linear_backend": "cutlass",
        "enforce_eager": True,
        "stage_init_timeout": 3600.0,
        "init_timeout": 3600.0,
        "host_weight_runtime_mode": args.hwr_mode,
        "host_weight_runtime_root": str(args.hwr_root),
        # An evaluator must never report a legacy fallback as an HWR result.
        "host_weight_runtime_required": args.hwr_mode != "disabled",
        "host_weight_runtime_wait_timeout_s": args.hwr_wait_timeout_s,
    }
    if args.offloader == "model":
        kwargs["enable_cpu_offload"] = True
    elif args.offloader == "layerwise":
        kwargs["enable_layerwise_offload"] = True
    else:
        kwargs.update(
            enable_distributed_layerwise_offload=True,
            dlo_use_allgather=False,
            dlo_resident_layers=0,
        )
    return kwargs


def _descendants(root_pid: int) -> list[int]:
    parents: dict[int, int] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            for line in (entry / "status").read_text().splitlines():
                if line.startswith("PPid:"):
                    parents[int(entry.name)] = int(line.split()[1])
                    break
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    selected = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in selected and pid not in selected:
                selected.add(pid)
                changed = True
    return sorted(selected)


def _proc_stat_fields(pid: int) -> list[str]:
    raw = Path(f"/proc/{pid}/stat").read_text()
    comm_end = raw.rfind(")")
    if comm_end < 0:
        raise ValueError(f"invalid /proc/{pid}/stat")
    # Field 3 (state) is index 0 in the returned suffix. This avoids shifting
    # every later field when the parenthesized process name contains spaces.
    return raw[comm_end + 2 :].split()


def _pid_identity(pid: int) -> dict[str, Any] | None:
    try:
        fields = _proc_stat_fields(pid)
        cmdline = [
            item.decode("utf-8", errors="replace")
            for item in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            if item
        ]
        return {
            "pid": pid,
            # Linux /proc/<pid>/stat field 4, relative to field 3 above.
            "ppid": int(fields[1]),
            "comm": Path(f"/proc/{pid}/comm").read_text().strip(),
            "cmdline": cmdline,
            # Linux /proc/<pid>/stat field 22, relative to field 3 above.
            "start_time_ticks": int(fields[19]),
        }
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, IndexError):
        return None


_MAP_HEADER = re.compile(r"^([0-9a-fA-F]+)-([0-9a-fA-F]+)\s+\S+\s+\S+\s+\S+\s+\S+(?:\s+(.*))?$")


def _path_is_within(path: str, roots: tuple[Path, ...]) -> bool:
    normalized = path.removesuffix(" (deleted)")
    if not normalized.startswith("/"):
        return False
    candidate = Path(normalized).resolve(strict=False)
    for root in roots:
        try:
            candidate.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _mapped_artifact_kib(pid: int, artifact_roots: tuple[Path, ...]) -> int:
    if not artifact_roots:
        return 0
    total_bytes = 0
    for line in Path(f"/proc/{pid}/maps").read_text().splitlines():
        match = _MAP_HEADER.match(line)
        if match is None or not match.group(3):
            continue
        if _path_is_within(match.group(3), artifact_roots):
            total_bytes += int(match.group(2), 16) - int(match.group(1), 16)
    return (total_bytes + 1023) // 1024


def _proc_metrics(pid: int, *, artifact_roots: tuple[Path, ...] = ()) -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
            if line.startswith(("Pss:", "Rss:", "Private_", "Shared_", "Locked:")):
                key, value, *_ = line.replace(":", "").split()
                result[f"{key.lower()}_kib"] = int(value)
        fields = _proc_stat_fields(pid)
        result["minor_faults"] = int(fields[7])
        result["major_faults"] = int(fields[9])
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith(("VmHWM:", "VmPeak:")):
                key, value, *_ = line.replace(":", "").split()
                result[f"{key.lower()}_kib"] = int(value)
        result["private_kib"] = result.get("private_clean_kib", 0) + result.get("private_dirty_kib", 0)
        result["private_and_locked_kib"] = result["private_kib"] + result.get("locked_kib", 0)
        result["mapped_artifact_kib"] = _mapped_artifact_kib(pid, artifact_roots)
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, IndexError):
        result["unavailable"] = 1
    return result


def _read_optional_int(path: Path) -> int | None:
    try:
        value = path.read_text().strip()
        return int(value) if value not in {"", "max"} else None
    except (FileNotFoundError, PermissionError, OSError, ValueError):
        return None


def _cgroup_metrics(cgroup_path: Path | None) -> dict[str, Any]:
    if cgroup_path is None:
        return {"path": None, "current_bytes": None, "peak_bytes": None, "available": False}
    current = _read_optional_int(cgroup_path / "memory.current")
    peak = _read_optional_int(cgroup_path / "memory.peak")
    version = 2
    if current is None:
        current = _read_optional_int(cgroup_path / "memory.usage_in_bytes")
        peak = _read_optional_int(cgroup_path / "memory.max_usage_in_bytes")
        version = 1
    return {
        "path": str(cgroup_path),
        "version": version,
        "current_bytes": current,
        "peak_bytes": peak,
        "available": current is not None and peak is not None,
    }


def _process_snapshot(
    *,
    pids: list[int] | None = None,
    artifact_roots: tuple[Path, ...] = (),
    cgroup_path: Path | None = None,
) -> dict[str, Any]:
    if pids is None:
        pids = _descendants(os.getpid())
    resolved_roots = tuple(root.resolve(strict=False) for root in artifact_roots)
    per_process = {str(pid): _proc_metrics(pid, artifact_roots=resolved_roots) for pid in pids}
    identities = {str(pid): identity for pid in pids if (identity := _pid_identity(pid)) is not None}
    totals = {
        key: sum(item.get(key, 0) for item in per_process.values())
        for key in (
            "pss_kib",
            "rss_kib",
            "private_clean_kib",
            "private_dirty_kib",
            "private_kib",
            "locked_kib",
            "private_and_locked_kib",
            "mapped_artifact_kib",
            "major_faults",
            "minor_faults",
        )
    }
    return {
        "monotonic_time_ns": time.monotonic_ns(),
        "pids": pids,
        "identities": identities,
        "per_process": per_process,
        "totals": totals,
        # Keep the original summary names for compatibility with prior
        # diagnostic records.
        "total_pss_kib": totals["pss_kib"],
        "total_rss_kib": totals["rss_kib"],
        "total_major_faults": totals["major_faults"],
        "total_minor_faults": totals["minor_faults"],
        "cgroup": _cgroup_metrics(cgroup_path),
    }


def _nvidia_smi(query: str) -> list[str]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--query-{query}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _optional_int(value: str) -> int | None:
    normalized = value.strip()
    if not normalized or normalized.startswith("[") or normalized.upper() == "N/A":
        return None
    return int(normalized)


def _gpu_inventory() -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for line in _nvidia_smi("gpu=index,uuid,name,memory.total,memory.used"):
        fields = [field.strip() for field in line.split(",", 4)]
        if len(fields) != 5:
            raise RuntimeError(f"unexpected nvidia-smi GPU row: {line!r}")
        index, uuid, name, total_mib, used_mib = fields
        inventory.append(
            {
                "index": int(index),
                "uuid": uuid,
                "name": name,
                "memory_total_mib": _optional_int(total_mib),
                "memory_used_mib": _optional_int(used_mib),
            }
        )
    return inventory


def _visible_gpu_inventory(inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return list(inventory)
    tokens = [token.strip() for token in visible.split(",") if token.strip()]
    selected: list[dict[str, Any]] = []
    for token in tokens:
        matches = [
            gpu
            for gpu in inventory
            if str(gpu["index"]) == token or str(gpu["uuid"]) == token or str(gpu["uuid"]).startswith(token)
        ]
        if len(matches) != 1:
            raise RuntimeError(f"cannot resolve CUDA_VISIBLE_DEVICES entry {token!r}")
        selected.append(matches[0])
    return selected


def _gpu_compute_processes() -> list[dict[str, Any]]:
    processes: list[dict[str, Any]] = []
    for line in _nvidia_smi("compute-apps=gpu_uuid,pid,process_name,used_memory"):
        fields = [field.strip() for field in line.split(",", 3)]
        if len(fields) != 4:
            raise RuntimeError(f"unexpected nvidia-smi process row: {line!r}")
        gpu_uuid, pid, process_name, used_mib = fields
        processes.append(
            {
                "gpu_uuid": gpu_uuid,
                "pid": int(pid),
                "process_name": process_name,
                "used_memory_mib": _optional_int(used_mib),
            }
        )
    return processes


def _gpu_preflight(dp_size: int) -> dict[str, Any]:
    inventory = _gpu_inventory()
    visible = _visible_gpu_inventory(inventory)
    selected = visible[:dp_size]
    selected_uuids = {str(gpu["uuid"]) for gpu in selected}
    owned_pids = set(_descendants(os.getpid()))
    foreign = [
        process
        for process in _gpu_compute_processes()
        if process["gpu_uuid"] in selected_uuids and process["pid"] not in owned_pids
    ]
    enough_visible_gpus = len(selected) == dp_size
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "requested_dp_size": dp_size,
        "visible_gpu_count": len(visible),
        "selected_gpus": selected,
        "foreign_compute_processes": foreign,
        "enough_visible_gpus": enough_visible_gpus,
        "exclusive_selected_gpus": enough_visible_gpus and not foreign,
        "memory_measurement_scope": "whole selected device",
    }


def _device_memory(dp_size: int) -> list[int]:
    selected = _visible_gpu_inventory(_gpu_inventory())[:dp_size]
    values = [gpu["memory_used_mib"] for gpu in selected]
    if len(values) != dp_size or any(value is None for value in values):
        raise RuntimeError("nvidia-smi did not report memory.used for every selected GPU")
    return [int(value) for value in values]


def _cold_cache_report(args: argparse.Namespace) -> dict[str, Any]:
    drop_caches = Path("/proc/sys/vm/drop_caches")
    artifacts = args.hwr_root / "artifacts"
    artifact_count: int | None = None
    artifact_scan_error: str | None = None
    try:
        artifact_count = sum(1 for path in artifacts.iterdir() if path.is_dir()) if artifacts.is_dir() else 0
    except OSError as exc:
        artifact_scan_error = f"{type(exc).__name__}: {exc}"
    try:
        cgroup_membership = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError as exc:
        cgroup_membership = [f"unavailable: {type(exc).__name__}: {exc}"]
    return {
        "status": "reported",
        "external_eviction_confirmed": bool(args.external_cold_cache_confirmed),
        "evaluator_performs_eviction": False,
        "global_drop_caches_exists": drop_caches.exists(),
        "global_drop_caches_writable": os.access(drop_caches, os.W_OK),
        "posix_fadvise_dontneed_available": bool(hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED")),
        "hwr_root_exists_before_run": args.hwr_root.exists(),
        "published_artifact_count_before_run": artifact_count,
        "artifact_scan_error": artifact_scan_error,
        "cgroup_membership": cgroup_membership,
        "note": (
            "The evaluator only reports capability and external confirmation; "
            "it never drops or advises away artifact pages itself."
        ),
    }


def _same_process_is_alive(identity: dict[str, Any]) -> bool:
    current = _pid_identity(int(identity["pid"]))
    return current is not None and current["start_time_ticks"] == identity["start_time_ticks"]


_RESOURCE_TRACKER_CODE = re.compile(r"^from multiprocessing\.resource_tracker import main;main\(\d+\)$")


def _is_parent_scoped_resource_tracker(identity: dict[str, Any]) -> bool:
    """Identify only CPython's exact parent-lifetime resource tracker.

    The tracker is not returned by ``multiprocessing.active_children()`` and
    intentionally remains until this evaluator interpreter exits. The suite's
    outer child-process wait proves that interpreter (and therefore this exact
    helper) exits. Treating it as an engine worker makes every successful spawn
    run fail its in-process cleanup check.
    """

    cmdline = identity.get("cmdline")
    if identity.get("ppid") != os.getpid() or not isinstance(cmdline, list):
        return False
    if cmdline.count("-c") != 1:
        return False
    program_index = cmdline.index("-c") + 1
    return (
        program_index == len(cmdline) - 1
        and isinstance(cmdline[program_index], str)
        and _RESOURCE_TRACKER_CODE.fullmatch(cmdline[program_index]) is not None
    )


def _wait_for_process_exit(
    identities: dict[str, dict[str, Any]],
    *,
    timeout_s: float,
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout_s
    # ``active_children`` also performs nonblocking waitpid calls for direct
    # children that have exited. Without this refresh, a reaped-late worker can
    # remain visible in /proc as a zombie for the entire polling window.
    multiprocessing.active_children()
    survivors = [identity for identity in identities.values() if _same_process_is_alive(identity)]
    while survivors and time.monotonic() < deadline:
        time.sleep(0.1)
        multiprocessing.active_children()
        survivors = [identity for identity in survivors if _same_process_is_alive(identity)]
    return survivors


def _active_child_pids() -> list[int]:
    return sorted(child.pid for child in multiprocessing.active_children() if child.pid is not None)


@dataclass
class _DevicePeakMonitor:
    dp_size: int
    interval_s: float = 0.2

    def __post_init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_mib: list[int] = []

    def start(self) -> None:
        initial = _device_memory(self.dp_size)
        self.peak_mib = list(initial)

        def sample() -> None:
            while not self._stop.wait(self.interval_s):
                try:
                    values = _device_memory(self.dp_size)
                except Exception:
                    continue
                if len(values) == len(self.peak_mib):
                    self.peak_mib = [max(old, new) for old, new in zip(self.peak_mib, values, strict=True)]

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def _distribution(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "median": float(statistics.median(values)) if values else None,
        "p95": _percentile(values, 95.0),
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "values": list(values),
    }


@dataclass
class _HostMemoryMonitor:
    interval_s: float
    artifact_roots: tuple[Path, ...]
    cgroup_path: Path | None

    def __post_init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._phase = "startup"
        self._phase_transitions: list[dict[str, Any]] = [
            {
                "phase": self._phase,
                "monotonic_time_ns": time.monotonic_ns(),
            }
        ]
        self._worker_pids: list[int] | None = None
        self.samples: list[dict[str, Any]] = []
        self.errors: list[str] = []

    def set_phase(self, phase: str) -> None:
        with self._lock:
            if phase == self._phase:
                return
            self._phase = phase
            self._phase_transitions.append(
                {
                    "phase": phase,
                    "monotonic_time_ns": time.monotonic_ns(),
                }
            )

    def set_worker_pids(self, pids: list[int]) -> None:
        with self._lock:
            self._worker_pids = list(pids)

    def _capture(self) -> None:
        try:
            with self._lock:
                pids = None if self._worker_pids is None else list(self._worker_pids)
            sample = _process_snapshot(
                pids=pids,
                artifact_roots=self.artifact_roots,
                cgroup_path=self.cgroup_path,
            )
            with self._lock:
                sample["phase"] = self._phase
                self.samples.append(sample)
        except BaseException as exc:
            with self._lock:
                self.errors.append(f"{type(exc).__name__}: {exc}")

    def start(self) -> None:
        self._capture()

        def sample() -> None:
            next_sample = time.monotonic() + self.interval_s
            while not self._stop.wait(max(0.0, next_sample - time.monotonic())):
                self._capture()
                next_sample += self.interval_s
                if next_sample < time.monotonic():
                    next_sample = time.monotonic() + self.interval_s

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_s * 2))
        self._capture()

    def worker_report(self, rank_to_pid: dict[int, int]) -> dict[str, Any]:
        worker_pids = set(rank_to_pid.values())
        selected_samples: list[dict[str, Any]] = []
        first_by_pid: dict[str, dict[str, Any]] = {}
        peaks_by_pid: dict[str, dict[str, int]] = {}
        metric_keys = (
            "pss_kib",
            "rss_kib",
            "private_clean_kib",
            "private_dirty_kib",
            "private_kib",
            "locked_kib",
            "private_and_locked_kib",
            "mapped_artifact_kib",
            "major_faults",
            "minor_faults",
        )
        for sample in self.samples:
            per_process = {
                pid: metrics for pid, metrics in sample.get("per_process", {}).items() if int(pid) in worker_pids
            }
            if not per_process:
                continue
            totals = {key: sum(int(metrics.get(key, 0)) for metrics in per_process.values()) for key in metric_keys}
            worker_sample = {
                "monotonic_time_ns": sample["monotonic_time_ns"],
                "phase": sample.get("phase"),
                "per_process": per_process,
                "totals": totals,
                "cgroup": sample.get("cgroup"),
            }
            selected_samples.append(worker_sample)
            for pid, metrics in per_process.items():
                first_by_pid.setdefault(
                    pid,
                    {
                        "monotonic_time_ns": sample["monotonic_time_ns"],
                        "phase": sample.get("phase"),
                        "metrics": dict(metrics),
                    },
                )
                peaks = peaks_by_pid.setdefault(pid, {})
                for key in metric_keys:
                    peaks[key] = max(peaks.get(key, 0), int(metrics.get(key, 0)))

        idle_samples = [sample for sample in selected_samples if sample["phase"] == "idle_steady_state"]
        idle_distributions = {
            key: _distribution([float(sample["totals"][key]) for sample in idle_samples]) for key in metric_keys
        }
        cgroup_current = [
            float(sample["cgroup"]["current_bytes"])
            for sample in idle_samples
            if sample.get("cgroup") and sample["cgroup"].get("current_bytes") is not None
        ]
        cgroup_peaks = [
            int(sample["cgroup"]["peak_bytes"])
            for sample in selected_samples
            if sample.get("cgroup") and sample["cgroup"].get("peak_bytes") is not None
        ]
        idle_sampling_coverage = self._phase_sampling_coverage(
            "idle_steady_state",
            idle_samples,
        )
        return {
            "rank_to_pid": {str(rank): pid for rank, pid in sorted(rank_to_pid.items())},
            "sampling_interval_s": self.interval_s,
            "sample_count": len(selected_samples),
            "idle_sample_count": len(idle_samples),
            "phase_transitions": list(self._phase_transitions),
            "idle_sampling_coverage": idle_sampling_coverage,
            "samples": selected_samples,
            "first_observed_by_pid": first_by_pid,
            "first_observed_is_pre_resolve": False,
            "worker_peaks": peaks_by_pid,
            "idle_steady_state": idle_distributions,
            "idle_cgroup_current_bytes": _distribution(cgroup_current),
            "cgroup_peak_bytes": max(cgroup_peaks) if cgroup_peaks else None,
            "sampling_errors": list(self.errors),
        }

    def _phase_sampling_coverage(
        self,
        phase: str,
        samples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Describe elapsed-time coverage and cadence for one monitor phase.

        A raw ``window / interval`` sample-count gate assumes a sample exactly
        on both phase boundaries, which the periodic sampler cannot guarantee.
        Instead, verify the observed wall-time span (with one nominal cadence
        interval of edge allowance) and fail closed on long edge or interior
        gaps.
        """

        transitions = list(self._phase_transitions)
        start_index = next(
            (index for index, item in enumerate(transitions) if item["phase"] == phase),
            None,
        )
        if start_index is None or start_index + 1 >= len(transitions):
            return {
                "passed": False,
                "reason": "phase boundary timestamps are unavailable",
                "phase": phase,
                "sample_count": len(samples),
            }

        start_ns = int(transitions[start_index]["monotonic_time_ns"])
        end_ns = int(transitions[start_index + 1]["monotonic_time_ns"])
        duration_s = max(0.0, (end_ns - start_ns) / 1e9)
        timestamps = sorted(
            int(sample["monotonic_time_ns"])
            for sample in samples
            if start_ns <= int(sample["monotonic_time_ns"]) <= end_ns
        )
        cadence_limit_s = self.interval_s * 2.5
        if not timestamps or duration_s <= 0:
            return {
                "passed": False,
                "reason": "no samples within a positive phase window",
                "phase": phase,
                "phase_duration_s": duration_s,
                "sample_count": len(timestamps),
                "nominal_interval_s": self.interval_s,
                "maximum_allowed_gap_s": cadence_limit_s,
            }

        leading_gap_s = (timestamps[0] - start_ns) / 1e9
        trailing_gap_s = (end_ns - timestamps[-1]) / 1e9
        inter_sample_gaps_s = [(right - left) / 1e9 for left, right in zip(timestamps, timestamps[1:])]
        maximum_interior_gap_s = max(inter_sample_gaps_s, default=0.0)
        observed_span_s = (timestamps[-1] - timestamps[0]) / 1e9
        covered_wall_time_s = min(duration_s, observed_span_s + self.interval_s)
        coverage_fraction = covered_wall_time_s / duration_s
        cadence_passed = (
            max(
                leading_gap_s,
                trailing_gap_s,
                maximum_interior_gap_s,
            )
            <= cadence_limit_s
        )
        passed = coverage_fraction >= 0.95 and cadence_passed and not self.errors
        return {
            "passed": passed,
            "phase": phase,
            "phase_start_monotonic_ns": start_ns,
            "phase_end_monotonic_ns": end_ns,
            "phase_duration_s": duration_s,
            "sample_count": len(timestamps),
            "nominal_interval_s": self.interval_s,
            "observed_span_s": observed_span_s,
            "covered_wall_time_s": covered_wall_time_s,
            "coverage_fraction": coverage_fraction,
            "leading_gap_s": leading_gap_s,
            "trailing_gap_s": trailing_gap_s,
            "maximum_interior_gap_s": maximum_interior_gap_s,
            "maximum_allowed_gap_s": cadence_limit_s,
            "cadence_passed": cadence_passed,
            "sampling_errors": list(self.errors),
        }


def _sampling_params(
    engine: AsyncOmni,
    args: argparse.Namespace,
    *,
    seed: int,
) -> list[Any]:
    params = copy.deepcopy(engine.default_sampling_params_list)
    diffusion = params[0]
    diffusion.width = args.width
    diffusion.height = args.height
    diffusion.fps = 24
    diffusion.num_inference_steps = args.steps
    diffusion.seed = seed
    diffusion.extra_args = {
        "task": "t2va",
        "duration": args.duration,
        "aspect_ratio": "16:9",
        "flow_shift": 12.0,
        "audio_flow_shift": 3.0,
    }
    return params


async def _generate_one(
    engine: AsyncOmni,
    args: argparse.Namespace,
    *,
    request_index: int,
    phase: str,
) -> Any:
    seed = args.seed + request_index
    final = None
    async for output in engine.generate(
        prompt=PROMPT,
        request_id=f"hwr-{phase}-seed-{seed}-request-{request_index}",
        sampling_params_list=_sampling_params(engine, args, seed=seed),
    ):
        if output.finished:
            final = output
    if final is None:
        raise RuntimeError("request completed without a final output")
    return final


async def _generate_sequential(
    engine: AsyncOmni,
    args: argparse.Namespace,
    *,
    count: int,
    phase: str,
) -> tuple[list[Any], list[float]]:
    outputs: list[Any] = []
    latencies: list[float] = []
    for request_index in range(count):
        started = time.perf_counter()
        outputs.append(
            await _generate_one(
                engine,
                args,
                request_index=request_index,
                phase=phase,
            )
        )
        latencies.append(time.perf_counter() - started)
    return outputs, latencies


def _array_digest(value: object) -> tuple[list[int], str]:
    array = np.ascontiguousarray(np.asarray(value))
    return list(array.shape), hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def _archive_array(value: object, path: Path) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(value))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array, allow_pickle=False)
    return {
        "path": str(path.resolve()),
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": hashlib.sha256(memoryview(array).cast("B")).hexdigest(),
        "bytes": int(array.nbytes),
    }


def _error_record(phase: str, exc: BaseException) -> dict[str, str]:
    return {
        "phase": phase,
        "type": type(exc).__name__,
        "message": str(exc),
    }


def _diffusion_stage_ids(engine: AsyncOmni) -> list[int]:
    stage_clients = getattr(engine.engine, "stage_clients", ())
    return [
        stage_id
        for stage_id, stage_client in enumerate(stage_clients)
        if getattr(stage_client, "stage_type", None) == "diffusion"
    ]


def _worker_resolution_records(value: object) -> list[dict[str, Any]]:
    """Flatten the stage/replica envelope without assuming its nesting."""

    records: list[dict[str, Any]] = []

    def visit(item: object) -> None:
        if isinstance(item, dict):
            if isinstance(item.get("rank"), dict) and "outcome" in item and "runtime_mode" in item:
                records.append(dict(item))
                return
            for nested in item.values():
                visit(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)

    visit(value)
    return records


def _resolution_summary(
    records: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if len(records) != args.dp_size:
        raise RuntimeError(f"expected HWR evidence from {args.dp_size} workers, got {len(records)}")
    ranks = [int(record["rank"]["global_rank"]) for record in records]
    expected_ranks = list(range(args.dp_size))
    if sorted(ranks) != expected_ranks:
        raise RuntimeError(f"HWR evidence rank set differs: expected={expected_ranks}, actual={sorted(ranks)}")

    schema_versions = {record.get("schema_version") for record in records}
    if schema_versions != {HWR_RESOLUTION_EVIDENCE_SCHEMA_VERSION}:
        raise RuntimeError(
            "HWR evidence schema differs: "
            f"expected={HWR_RESOLUTION_EVIDENCE_SCHEMA_VERSION}, actual={sorted(map(str, schema_versions))}"
        )

    modes = {str(record.get("runtime_mode")) for record in records}
    if modes != {args.hwr_mode}:
        raise RuntimeError(f"HWR evidence mode differs: expected={args.hwr_mode!r}, actual={sorted(modes)}")
    outcomes = Counter(str(record.get("outcome")) for record in records)
    roles = Counter(str(record.get("claim_role")) for record in records)
    summary: dict[str, Any] = {
        "worker_count": len(records),
        "global_ranks": sorted(ranks),
        "outcomes": dict(sorted(outcomes.items())),
        "claim_roles": dict(sorted(roles.items())),
    }
    if args.hwr_mode == "disabled":
        if outcomes != Counter({"not_requested": args.dp_size}):
            raise RuntimeError(f"disabled HWR produced unexpected evidence outcomes: {dict(outcomes)}")
        summary["observed_artifact_state"] = "not_requested"
        return summary

    if outcomes != Counter({"ready": args.dp_size}):
        raise RuntimeError(f"required HWR did not resolve ready on every worker: {dict(outcomes)}")
    if any(bool(record.get("ordinary_loader_fallback")) for record in records):
        raise RuntimeError("required HWR evidence reports an ordinary-loader fallback")
    artifact_keys = {str(record.get("artifact_key")) for record in records if record.get("artifact_key")}
    compatibility_digests = {
        str(record.get("artifact_compatibility_digest"))
        for record in records
        if record.get("artifact_compatibility_digest")
    }
    if len(artifact_keys) != 1 or len(compatibility_digests) != 1:
        raise RuntimeError(
            "workers did not resolve one artifact identity: "
            f"keys={sorted(artifact_keys)}, compatibility_digests={sorted(compatibility_digests)}"
        )
    generation_ids = {str(record.get("generation_id")) for record in records if record.get("generation_id")}
    if len(generation_ids) != 1:
        raise RuntimeError(f"workers did not resolve one artifact generation: {sorted(generation_ids)}")
    summary["artifact_key"] = next(iter(artifact_keys))
    summary["artifact_compatibility_digest"] = next(iter(compatibility_digests))
    summary["generation_id"] = next(iter(generation_ids))

    builder_count = roles.get("builder", 0)
    waiter_count = roles.get("waiter", 0)
    cache_hit_count = roles.get("cache_hit", 0)
    if args.hwr_mode == "read_only":
        if cache_hit_count != args.dp_size:
            raise RuntimeError(f"read-only HWR was not a cache hit on every worker: roles={dict(roles)}")
        summary["observed_artifact_state"] = "warm_cache_hit"
    elif builder_count == 1 and builder_count + waiter_count + cache_hit_count == args.dp_size:
        summary["observed_artifact_state"] = "cold_build"
    elif builder_count == 0 and cache_hit_count == args.dp_size:
        summary["observed_artifact_state"] = "preexisting_cache_hit"
    else:
        raise RuntimeError(f"read-write HWR produced an invalid election pattern: roles={dict(roles)}")

    grant_fields = (
        "runtime_instance_id",
        "capability_grant_id",
        "provider_id",
        "provider_abi",
        "backing_kind",
        "access_features",
    )
    grant_records = [{field: record.get(field) for field in grant_fields} for record in records]
    if any(any(value is not None for value in grant.values()) for grant in grant_records):
        incomplete = [
            index for index, grant in enumerate(grant_records) if any(grant[field] is None for field in grant_fields)
        ]
        if incomplete:
            raise RuntimeError(f"workers emitted incomplete capability-grant evidence: record_indexes={incomplete}")
        provider_tuples = {
            (
                grant["provider_id"],
                grant["provider_abi"],
                grant["backing_kind"],
                tuple(sorted(str(value) for value in grant["access_features"])),
            )
            for grant in grant_records
        }
        if len(provider_tuples) != 1:
            raise RuntimeError(f"workers resolved incompatible capability grants: {provider_tuples}")
        runtime_ids = {str(grant["runtime_instance_id"]) for grant in grant_records}
        grant_ids = {str(grant["capability_grant_id"]) for grant in grant_records}
        if len(runtime_ids) != args.dp_size or len(grant_ids) != args.dp_size:
            raise RuntimeError(
                "capability grants are not bound one-to-one to worker runtimes: "
                f"runtime_ids={sorted(runtime_ids)}, grant_ids={sorted(grant_ids)}"
            )
        summary["capability_grants"] = grant_records
    return summary


def _formal_runtime_contract(
    records: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    missing: list[str] = []
    violations: list[str] = []
    if args.hwr_mode != "disabled":
        grant_fields = (
            "runtime_instance_id",
            "capability_grant_id",
            "provider_id",
            "provider_abi",
            "backing_kind",
            "access_features",
        )
        for record in records:
            rank = record.get("rank", {}).get("global_rank", "unknown")
            if record.get("schema_version") != HWR_RESOLUTION_EVIDENCE_SCHEMA_VERSION:
                violations.append(
                    f"rank {rank} evidence schema_version is not {HWR_RESOLUTION_EVIDENCE_SCHEMA_VERSION}"
                )
            for field in grant_fields:
                if record.get(field) is None:
                    missing.append(f"rank {rank} capability-grant field {field}")
            if record.get("negotiated_capability_grant_id") is None:
                missing.append(f"rank {rank} negotiated_capability_grant_id")
            elif record.get("negotiated_capability_grant_id") != record.get("capability_grant_id"):
                violations.append(f"rank {rank} Ready access differs from negotiated grant")
            if record.get("backing_kind") != "runtime_mmap":
                violations.append(f"rank {rank} did not resolve the runtime_mmap backing")
            features = {str(feature).lower() for feature in record.get("access_features") or ()}
            if "complete_tensor_read" not in features:
                violations.append(f"rank {rank} grant lacks complete_tensor_read")
            pinned_budget = record.get("pinned_slot_budget_bytes")
            if not isinstance(pinned_budget, int) or isinstance(pinned_budget, bool) or pinned_budget < 0:
                missing.append(f"rank {rank} canonical pinned_slot_budget_bytes")

        expected_plan_kind = "component" if args.offloader == "model" else "blocks_plus_resident"
        plan_ids: set[str] = set()
        plan_kinds: set[str] = set()
        coverage_digests: set[str] = set()
        for record in records:
            rank = record.get("rank", {}).get("global_rank", "unknown")
            plan_id = record.get("selected_transfer_plan_id")
            if not isinstance(plan_id, str) or not plan_id:
                missing.append(f"rank {rank} selected_transfer_plan_id")
            else:
                plan_ids.add(plan_id)
            plan_kind = record.get("selected_transfer_plan_kind")
            if not isinstance(plan_kind, str) or not plan_kind:
                missing.append(f"rank {rank} selected_transfer_plan_kind")
            else:
                plan_kinds.add(plan_kind)
                if plan_kind != expected_plan_kind:
                    violations.append(
                        f"rank {rank} selected transfer plan kind differs: "
                        f"expected={expected_plan_kind}, actual={plan_kind}"
                    )
            coverage_digest = record.get("exact_coverage_digest")
            if not isinstance(coverage_digest, str) or not coverage_digest:
                missing.append(f"rank {rank} exact_coverage_digest")
            else:
                coverage_digests.add(coverage_digest)
                if len(coverage_digest) != 64 or any(
                    character not in "0123456789abcdef" for character in coverage_digest
                ):
                    violations.append(f"rank {rank} exact_coverage_digest is not a lowercase SHA-256 digest")
        if len(plan_ids) > 1:
            violations.append(f"workers selected different transfer plan IDs: {sorted(plan_ids)}")
        if len(plan_kinds) > 1:
            violations.append(f"workers selected different transfer plan kinds: {sorted(plan_kinds)}")
        if len(coverage_digests) > 1:
            violations.append(f"workers selected different exact transfer coverage: {sorted(coverage_digests)}")

        required_kinds = {"component"} if args.offloader == "model" else {"block", "resident"}
        for record in records:
            rank = record.get("rank", {}).get("global_rank", "unknown")
            reported = {str(value).lower() for value in record.get("unit_kinds", ())}
            if not reported:
                missing.append(f"rank {rank} unit_kinds")
            elif not required_kinds.issubset(reported):
                violations.append(
                    f"rank {rank} lacks required unit kinds: required={sorted(required_kinds)}, actual={sorted(reported)}"
                )

    for record in records:
        rank = record.get("rank", {}).get("global_rank", "unknown")
        if record.get("pre_resolve") is None:
            missing.append(f"rank {rank} PRE_RESOLVE evidence")
        idle_state = record.get("idle_state")
        if idle_state is None:
            missing.append(f"rank {rank} idle-state evidence")
        elif any(int(idle_state.get(field, -1)) != 0 for field in ("outstanding_units", "bindings", "events")):
            violations.append(f"rank {rank} has nonzero idle state: {idle_state}")
        elif args.hwr_mode != "disabled":
            resident = idle_state.get("resident_bindings")
            total = idle_state.get("total_bindings")
            transient = idle_state.get("bindings")
            if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in (resident, total)):
                missing.append(f"rank {rank} resident/total binding evidence")
            elif total != transient + resident:
                violations.append(
                    f"rank {rank} binding accounting differs: transient={transient}, resident={resident}, total={total}"
                )

    if args.scenario == "C4":
        builders = [record for record in records if record.get("claim_role") == "builder"]
        waiters = [record for record in records if record.get("claim_role") == "waiter"]
        if len(builders) != 1:
            violations.append(f"C4 requires exactly one builder, observed {len(builders)}")
        if len(waiters) != len(records) - 1:
            violations.append(f"C4 requires {len(records) - 1} ordered waiters, observed {len(waiters)}")
        if len(builders) == 1:
            builder = builders[0]
            if int(builder["rank"]["global_rank"]) != 0:
                violations.append("C4 builder must be DP rank 0")
            started = builder.get("builder_started")
            if not isinstance(started, dict):
                missing.append("rank 0 BuilderStarted event")
            else:
                for field in ("launch_id", "artifact_key", "lease_id", "builder_actor_id"):
                    if started.get(field) is None:
                        missing.append(f"BuilderStarted.{field}")
                if started.get("launch_id") != args.launch_id:
                    violations.append("BuilderStarted launch_id differs from suite launch")
                if started.get("artifact_key") != builder.get("artifact_key"):
                    violations.append("BuilderStarted artifact_key differs from resolved artifact")
                if started.get("builder_actor_id") != "dp:0":
                    violations.append("BuilderStarted builder_actor_id is not dp:0")
                if started.get("lease_id") != builder.get("generation_id"):
                    violations.append("BuilderStarted lease_id differs from resolved generation")
                expected_identity = {
                    "launch_id": args.launch_id,
                    "artifact_key": builder.get("artifact_key"),
                    "lease_id": started.get("lease_id"),
                    "builder_actor_id": "dp:0",
                }
                for record in waiters:
                    rank = record["rank"]["global_rank"]
                    observation = record.get("observed_builder_started")
                    if not isinstance(observation, dict):
                        missing.append(f"rank {rank} observed BuilderStarted")
                        continue
                    for field, expected in expected_identity.items():
                        if observation.get(field) != expected:
                            violations.append(f"rank {rank} observed different BuilderStarted.{field}")
            for record in waiters:
                if record.get("producer_present") is not False:
                    missing.append(f"rank {record['rank']['global_rank']} producer_present=false evidence")
    elif args.scenario == "W4":
        for record in records:
            rank = record["rank"]["global_rank"]
            if record.get("producer_present") is not False:
                missing.append(f"rank {rank} producer_present=false evidence")
            if record.get("claim_role") != "cache_hit":
                violations.append(f"rank {rank} is not a read-only cache hit")

    return {
        "passed": not missing and not violations,
        "missing": sorted(set(missing)),
        "violations": sorted(set(violations)),
    }


def _load_cache_attestation(args: argparse.Namespace) -> dict[str, Any]:
    if args.cache_attestation is None:
        return {"verified": False, "reason": "cache attestation was not provided"}
    try:
        attestation = json.loads(args.cache_attestation.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"verified": False, "reason": f"cannot read cache attestation: {type(exc).__name__}: {exc}"}
    reasons: list[str] = []
    if attestation.get("launch_id") != args.launch_id:
        reasons.append("launch_id differs")
    if attestation.get("scenario") != args.scenario:
        reasons.append("scenario differs")
    if not attestation.get("verified_zero_resident_bytes"):
        reasons.append("resident file pages were not verified as zero")
    if attestation.get("scope") != "global_then_file_verified":
        reasons.append("cache eviction scope is not formal global_then_file_verified")
    return {"verified": not reasons, "reasons": reasons, "attestation": attestation}


def _measurement_cgroup_report(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"verified": False, "reason": "measurement cgroup was not provided"}
    try:
        members = {int(value) for value in (path / "cgroup.procs").read_text().split()}
    except (OSError, ValueError) as exc:
        return {"verified": False, "reason": f"cannot read cgroup.procs: {type(exc).__name__}: {exc}"}
    metrics = _cgroup_metrics(path)
    reasons = []
    if members != {os.getpid()}:
        reasons.append(f"fresh cgroup initially contains PIDs {sorted(members)}, expected only {os.getpid()}")
    if not metrics["available"]:
        reasons.append("cgroup current/peak memory metrics are unavailable")
    return {
        "verified": not reasons,
        "initial_members": sorted(members),
        "metrics": metrics,
        "reasons": reasons,
    }


_ENVIRONMENT_KEYS = frozenset(
    {
        "CUDA_VISIBLE_DEVICES",
        "CUDA_HOME",
        "CUDA_PATH",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_HUB_OFFLINE",
        "LD_LIBRARY_PATH",
        "NCCL_DEBUG",
        "NCCL_IB_DISABLE",
        "NCCL_P2P_DISABLE",
        "PATH",
        "PYTHONHASHSEED",
        "PYTHONPATH",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "VLLM_WORKER_MULTIPROC_METHOD",
    }
)
_ENVIRONMENT_PREFIXES = ("CUDA_", "NCCL_", "TORCH_", "VLLM_", "VLLM_OMNI_")
_SECRET_FRAGMENTS = ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "API_KEY", "ACCESS_KEY")


def _environment_snapshot() -> dict[str, Any]:
    selected: dict[str, str] = {}
    redacted: list[str] = []
    for key, value in sorted(os.environ.items()):
        if key not in _ENVIRONMENT_KEYS and not key.startswith(_ENVIRONMENT_PREFIXES):
            continue
        if any(fragment in key.upper() for fragment in _SECRET_FRAGMENTS):
            redacted.append(key)
            continue
        selected[key] = value
    return {
        "allowlist": sorted(_ENVIRONMENT_KEYS),
        "prefixes": list(_ENVIRONMENT_PREFIXES),
        "values": selected,
        "redacted_keys": redacted,
    }


async def run(args: argparse.Namespace) -> tuple[dict[str, Any], BaseException | None]:
    measured_requests = args.client_requests or args.measured_requests
    result: dict[str, Any] = {
        "schema_version": 3,
        "status": "running",
        "launch": {
            "scenario": args.scenario,
            "launch_id": args.launch_id,
            "argv": list(os.sys.argv),
            "executable": os.sys.executable,
            "environment": _environment_snapshot(),
        },
        "candidate": {
            "hwr_mode": args.hwr_mode,
            "offloader": args.offloader,
            "engine_kwargs": _engine_kwargs(args),
        },
        "execution_semantics": {
            "client_submission": "strictly_sequential",
            "engine_execution": "serial_broadcast",
            "scheduler_max_num_seqs": 1,
            "dp_size": args.dp_size,
            "rank_dispatch": "each admitted request executes on every DP rank",
            "effective_rank_executions_per_client_request": args.dp_size,
            "response_collection": "only DP rank 0 returns the response for each client request",
            "dp_throughput_measurement": False,
            "warmup_requests": args.warmup_requests,
            "settle_s": args.settle_s,
            "idle_window_s": args.idle_window_s,
            "sample_interval_s": args.sample_interval_s,
            "measured_requests": measured_requests,
        },
        "request": {
            "width": args.width,
            "height": args.height,
            "duration": args.duration,
            "fps": 24,
            "steps": args.steps,
            "measured_requests": measured_requests,
            "seeds": [args.seed + index for index in range(measured_requests)],
            "prompt": PROMPT,
            "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        },
        "qualification_preconditions": {
            "legacy_external_cold_cache_assertion": args.external_cold_cache_confirmed,
            "legacy_external_isolated_cgroup_assertion": args.external_isolated_cgroup_confirmed,
            "exclusive_selected_gpus": False,
            "formal_qualification": False,
            "formal_qualification_missing": ["preflight not completed"],
        },
        "cold_cache": {"status": "not_run"},
        "gpu_preflight": {"status": "not_run"},
        "host_weight_runtime_evidence": {"status": "not_collected"},
        "timings": {},
        "outputs": [],
        "output_capture": {
            "status": "not_collected",
            "scope": None,
        },
        "memory_measurement": {"status": "not_collected"},
        "errors": [],
        "process_diagnostics": {
            "process_exit_timeout_s": args.process_exit_timeout_s,
            "verification_scope": (
                "PID/start-time identities are checked for descendants observed at "
                "ready, steady, or immediately before teardown. "
                "multiprocessing.active_children is supplemental direct-child "
                "diagnostic data, not proof that no grandchildren exist."
            ),
        },
    }

    process_diagnostics = result["process_diagnostics"]
    observed_engine_identities: dict[str, dict[str, Any]] = {}
    parent_scoped_resource_trackers: dict[str, dict[str, Any]] = {}
    baseline_identity_keys: set[tuple[int, int]] = set()
    baseline_active_children: set[int] = set()

    def remember_engine_processes(snapshot: dict[str, Any]) -> None:
        for pid, identity in snapshot.get("identities", {}).items():
            identity_key = (int(identity["pid"]), int(identity["start_time_ticks"]))
            if int(identity["pid"]) != os.getpid() and identity_key not in baseline_identity_keys:
                if _is_parent_scoped_resource_tracker(identity):
                    parent_scoped_resource_trackers[pid] = identity
                else:
                    observed_engine_identities[pid] = identity

    poison_keys = (
        "VLLM_OMNI_HWR_POISON_PRODUCER",
        "VLLM_OMNI_HWR_POISON_TRANSFORMER_LOAD",
    )
    prior_poison_env = {key: os.environ.get(key) for key in poison_keys}

    device_monitor = _DevicePeakMonitor(args.dp_size)
    device_monitor_started = False
    host_monitor = _HostMemoryMonitor(
        interval_s=args.sample_interval_s,
        artifact_roots=(args.hwr_root,),
        cgroup_path=args.measurement_cgroup,
    )
    host_monitor_started = False
    engine: AsyncOmni | None = None
    rank_to_pid: dict[int, int] = {}
    primary_error: BaseException | None = None
    phase = "process_baseline"
    try:
        baseline = _process_snapshot(
            artifact_roots=(args.hwr_root,),
            cgroup_path=args.measurement_cgroup,
        )
        process_diagnostics["baseline"] = baseline
        baseline_identity_keys = {
            (int(identity["pid"]), int(identity["start_time_ticks"])) for identity in baseline["identities"].values()
        }
        baseline_active_children = set(_active_child_pids())
        process_diagnostics["active_children_baseline"] = sorted(baseline_active_children)

        phase = "cold_cache_capability"
        result["cold_cache"] = _cold_cache_report(args)

        phase = "suite_control_validation"
        expected_modes = {"L4": "disabled", "C4": "read_write", "W4": "read_only"}
        if args.scenario is not None and expected_modes[args.scenario] != args.hwr_mode:
            raise ValueError(
                f"scenario {args.scenario} requires HWR mode {expected_modes[args.scenario]}, got {args.hwr_mode}"
            )
        cache_report = _load_cache_attestation(args)
        cgroup_report = _measurement_cgroup_report(args.measurement_cgroup)
        result["qualification_preconditions"]["cache_control"] = cache_report
        result["qualification_preconditions"]["cgroup_control"] = cgroup_report
        if args.formal_runtime_contract:
            if not cache_report["verified"]:
                raise RuntimeError(f"formal cache-control attestation failed: {cache_report}")
            if not cgroup_report["verified"]:
                raise RuntimeError(f"formal cgroup-control verification failed: {cgroup_report}")

        phase = "argument_validation"
        if args.poison_warm_build:
            if args.hwr_mode != "read_only":
                raise ValueError("--poison-warm-build requires --hwr-mode read_only")
            os.environ["VLLM_OMNI_HWR_POISON_PRODUCER"] = "1"
            os.environ["VLLM_OMNI_HWR_POISON_TRANSFORMER_LOAD"] = "1"

        phase = "gpu_preflight"
        gpu_preflight = _gpu_preflight(args.dp_size)
        gpu_preflight["allow_nonexclusive_gpus"] = bool(args.allow_nonexclusive_gpus)
        result["gpu_preflight"] = gpu_preflight
        qualification = result["qualification_preconditions"]
        qualification["exclusive_selected_gpus"] = gpu_preflight["exclusive_selected_gpus"]
        missing: list[str] = []
        if not cache_report["verified"]:
            missing.append("verified global cold-cache eviction")
        if not cgroup_report["verified"]:
            missing.append("fresh isolated measurement cgroup")
        if not gpu_preflight["exclusive_selected_gpus"]:
            missing.append("exclusive selected GPUs")
        qualification["formal_qualification_missing"] = missing
        qualification["formal_qualification"] = not missing

        if not gpu_preflight["enough_visible_gpus"]:
            gpu_preflight["status"] = "blocked"
            raise RuntimeError(
                f"DP size {args.dp_size} requires at least {args.dp_size} visible GPUs; "
                f"found {gpu_preflight['visible_gpu_count']}"
            )
        if gpu_preflight["foreign_compute_processes"]:
            if not args.allow_nonexclusive_gpus:
                gpu_preflight["status"] = "blocked"
                raise RuntimeError(
                    "selected GPUs are not exclusive; use --allow-nonexclusive-gpus "
                    "only for a non-qualifying diagnostic run"
                )
            gpu_preflight["status"] = "allowed_nonexclusive_diagnostic"
        else:
            gpu_preflight["status"] = "passed"

        phase = "host_monitor_start"
        host_monitor.start()
        host_monitor_started = True

        phase = "device_monitor_start"
        device_monitor.start()
        device_monitor_started = True

        phase = "engine_startup"
        from vllm_omni.entrypoints.async_omni import AsyncOmni

        started = time.perf_counter()
        engine = AsyncOmni(**_engine_kwargs(args))
        result["timings"]["startup_s"] = time.perf_counter() - started
        result["timings"]["time_to_ready_s"] = result["timings"]["startup_s"]
        result["timings"]["time_to_ready_scope"] = (
            "process constructor; exact PRE_RESOLVE release timing requires runtime gate evidence"
        )
        ready = _process_snapshot(
            artifact_roots=(args.hwr_root,),
            cgroup_path=args.measurement_cgroup,
        )
        process_diagnostics["ready"] = ready
        remember_engine_processes(ready)
        result["ready_device_used_mib"] = _device_memory(args.dp_size)

        phase = "host_weight_runtime_evidence"
        stage_ids = _diffusion_stage_ids(engine)
        if len(stage_ids) != 1:
            raise RuntimeError(f"expected exactly one diffusion stage, got {stage_ids}")
        raw_evidence = await engine.collective_rpc(
            method="get_host_weight_runtime_evidence",
            timeout=60.0,
            stage_ids=stage_ids,
        )
        records = _worker_resolution_records(raw_evidence)
        rank_to_pid = {int(record["rank"]["global_rank"]): int(record["rank"]["pid"]) for record in records}
        host_monitor.set_worker_pids(list(rank_to_pid.values()))
        evidence_result = {
            "status": "collected",
            "stage_ids": stage_ids,
            "raw_stage_results": raw_evidence,
            "worker_records": records,
        }
        result["host_weight_runtime_evidence"] = evidence_result
        evidence_result["summary"] = _resolution_summary(records, args=args)
        runtime_contract = _formal_runtime_contract(records, args=args)
        evidence_result["formal_runtime_contract"] = runtime_contract
        if args.formal_runtime_contract and not runtime_contract["passed"]:
            raise RuntimeError(f"formal runtime evidence is incomplete: {runtime_contract}")

        phase = "warmup_requests"
        host_monitor.set_phase("warmup")
        _, warmup_latencies = await _generate_sequential(
            engine,
            args,
            count=args.warmup_requests,
            phase="warmup",
        )
        result["timings"]["warmup_request_latency_s"] = _distribution(warmup_latencies)

        phase = "settle"
        host_monitor.set_phase("settle")
        if args.settle_s:
            await asyncio.sleep(args.settle_s)

        phase = "idle_steady_state"
        host_monitor.set_phase("idle_steady_state")
        await asyncio.sleep(args.idle_window_s)

        phase = "measured_requests"
        host_monitor.set_phase("measured_requests")
        outputs, request_latencies = await _generate_sequential(
            engine,
            args,
            count=measured_requests,
            phase="measured",
        )
        result["timings"]["request_latency_s"] = {
            **_distribution(request_latencies),
            "values": request_latencies,
        }
        result["timings"]["sequential_measured_wall_s"] = sum(request_latencies)

        phase = "output_digest"
        for request_index, output in enumerate(outputs):
            video_shape, video_digest = _array_digest(output.images[0])
            audio_shape, audio_digest = _array_digest(output.multimodal_output.get("audio"))
            output_record = {
                "request_index": request_index,
                "seed": args.seed + request_index,
                "video_shape": video_shape,
                "video_sha256": video_digest,
                "audio_shape": audio_shape,
                "audio_sha256": audio_digest,
                "peak_memory_mb": output.peak_memory_mb,
                "stage_durations": output.stage_durations,
            }
            if args.tensor_output_dir is not None:
                output_record["tensor_archives"] = {
                    "video": _archive_array(
                        output.images[0],
                        args.tensor_output_dir / f"request-{request_index:02d}-video.npy",
                    ),
                    "audio": _archive_array(
                        output.multimodal_output.get("audio"),
                        args.tensor_output_dir / f"request-{request_index:02d}-audio.npy",
                    ),
                }
            result["outputs"].append(output_record)
        result["output_capture"] = {
            "status": "captured",
            "scope": "pre_encoding",
            "request_count": len(result["outputs"]),
            "media": ["video", "audio"],
            "source": (
                "final AsyncOmni output.images[0] and multimodal_output['audio'] "
                "arrays, captured before any MP4/WAV or other external media encoding"
            ),
            "tensor_archives_written": args.tensor_output_dir is not None,
        }
        steady = _process_snapshot(
            pids=list(rank_to_pid.values()),
            artifact_roots=(args.hwr_root,),
            cgroup_path=args.measurement_cgroup,
        )
        process_diagnostics["steady"] = steady
        remember_engine_processes(steady)
    except BaseException as exc:
        primary_error = exc
        result["errors"].append(_error_record(phase, exc))
    finally:
        diagnostic_errors: list[BaseException] = []
        try:
            if host_monitor_started:
                host_monitor.set_phase("teardown")
            before_teardown = _process_snapshot(
                artifact_roots=(args.hwr_root,),
                cgroup_path=args.measurement_cgroup,
            )
            process_diagnostics["before_teardown"] = before_teardown
            remember_engine_processes(before_teardown)
            process_diagnostics["active_children_before_teardown"] = _active_child_pids()
        except BaseException as exc:
            diagnostic_errors.append(exc)
            result["errors"].append(_error_record("before_teardown_diagnostics", exc))

        try:
            if engine is not None:
                started = time.perf_counter()
                engine.close()
                result["timings"]["shutdown_s"] = time.perf_counter() - started
        except BaseException as exc:
            if primary_error is None:
                primary_error = exc
            result["errors"].append(_error_record("engine_close", exc))
        finally:
            try:
                if device_monitor_started:
                    device_monitor.stop()
            except BaseException as exc:
                if primary_error is None:
                    primary_error = exc
                result["errors"].append(_error_record("device_monitor_stop", exc))
            try:
                if host_monitor_started:
                    host_monitor.stop()
            except BaseException as exc:
                if primary_error is None:
                    primary_error = exc
                result["errors"].append(_error_record("host_monitor_stop", exc))
            result["sampled_device_peak_mib"] = device_monitor.peak_mib
            if rank_to_pid:
                result["memory_measurement"] = {
                    "status": "collected",
                    **host_monitor.worker_report(rank_to_pid),
                }
            else:
                result["memory_measurement"] = {
                    "status": "worker_pids_unavailable",
                    "sampling_errors": list(host_monitor.errors),
                }
            for key, previous in prior_poison_env.items():
                if previous is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = previous

        try:
            survivors = _wait_for_process_exit(
                observed_engine_identities,
                timeout_s=args.process_exit_timeout_s,
            )
        except BaseException as exc:
            diagnostic_errors.append(exc)
            survivors = []
            result["errors"].append(_error_record("tracked_process_exit_check", exc))
        process_diagnostics["observed_engine_processes"] = list(observed_engine_identities.values())
        process_diagnostics["parent_scoped_resource_trackers"] = list(parent_scoped_resource_trackers.values())
        process_diagnostics["surviving_observed_processes"] = survivors
        try:
            after_teardown = _process_snapshot(
                artifact_roots=(args.hwr_root,),
                cgroup_path=args.measurement_cgroup,
            )
            process_diagnostics["after_teardown"] = after_teardown
            active_children_after = _active_child_pids()
            process_diagnostics["active_children_after_teardown"] = active_children_after
            unexpected_active_children = sorted(set(active_children_after) - baseline_active_children)
            process_diagnostics["unexpected_active_children_after_teardown"] = unexpected_active_children
        except BaseException as exc:
            diagnostic_errors.append(exc)
            unexpected_active_children = []
            result["errors"].append(_error_record("after_teardown_diagnostics", exc))

        if diagnostic_errors:
            process_diagnostics["cleanup_status"] = "unverified"
        elif survivors or unexpected_active_children:
            process_diagnostics["cleanup_status"] = "failed"
        else:
            process_diagnostics["cleanup_status"] = "passed_within_observed_scope"
        if survivors or unexpected_active_children:
            cleanup_error = RuntimeError(
                "observed engine processes remain after shutdown: "
                f"tracked={survivors}, active_children={unexpected_active_children}"
            )
            if primary_error is None:
                primary_error = cleanup_error
            result["errors"].append(_error_record("process_cleanup", cleanup_error))
        elif diagnostic_errors and primary_error is None:
            primary_error = diagnostic_errors[0]

        qualification = result["qualification_preconditions"]
        runtime_contract = result.get("host_weight_runtime_evidence", {}).get("formal_runtime_contract", {})
        missing = list(qualification.get("formal_qualification_missing", ()))
        if not runtime_contract.get("passed"):
            missing.append("formal runtime contract evidence")
        idle_coverage = result.get("memory_measurement", {}).get("idle_sampling_coverage", {})
        idle_duration_s = idle_coverage.get("phase_duration_s")
        if (
            not idle_coverage.get("passed")
            or not isinstance(idle_duration_s, (int, float))
            or float(idle_duration_s) < args.idle_window_s * 0.95
        ):
            missing.append(f"idle steady-state wall-time sampling coverage ({idle_coverage})")
        output_capture_scope = result.get("output_capture", {}).get("scope")
        if output_capture_scope != "pre_encoding":
            missing.append("pre-encoding output capture hook")
        qualification["formal_qualification_missing"] = sorted(set(missing))
        qualification["formal_qualification"] = not missing
        result["status"] = "failed" if primary_error is not None else "success"
    return result, primary_error


def main() -> None:
    args = parse_args()
    result, failure = asyncio.run(run(args))
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(f"HWR_EVAL_RESULT {rendered}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    if failure is not None:
        raise failure


if __name__ == "__main__":
    main()
