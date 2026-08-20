# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Controlled L4/C4/W4 qualification suite for MiniMax-H3 HWR.

The suite creates a unique empty C4 namespace, preserves that exact generation
for W4, alternates the legacy and cold/warm blocks, and launches every trial in
a fresh process.  Formal mode fails before GPU launch unless a delegated cgroup
v2 parent and explicit global page-cache control are both available.  File-
scoped ``posix_fadvise(DONTNEED)`` plus ``fincore`` is available only as
diagnostic evidence because it cannot prove a globally controlled cold cache.

``--protocol lightweight`` is deterministic, uses no GPU, and exists solely to
exercise suite orchestration and report math in CPU unit tests.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCENARIO_MODES = {"L4": "disabled", "C4": "read_write", "W4": "read_only"}
OFFLOADERS = ("model", "layerwise", "dlo-no-allgather")
FORMAL_REPETITIONS = 5
FORMAL_WARMUPS = 2
FORMAL_MEASURED_REQUESTS = 10
FORMAL_SETTLE_S = 5.0
FORMAL_IDLE_WINDOW_S = 20.0
FORMAL_SAMPLE_INTERVAL_S = 0.1
FORMAL_PROCESS_EXIT_TIMEOUT_S = 5.0
MINIMUM_PSS_SAVING_BYTES = 75 * 1024**3
CEILING_ALLOWANCE_BYTES = 2 * 1024**3
SYNTHETIC_PINNED_SLOT_BUDGET_BYTES = 1024**2
MODEL_SNAPSHOT = "42ed227ee7df40d41602854ae760620d6eb651fe"
EXPECTED_RUNTIME = {
    "vllm": "0.27.0",
    "torch": "2.13.0+cu129",
    "torch_cuda": "12.9",
    "diffusers": "0.38.0",
}


class QualificationBlockedError(RuntimeError):
    """Formal controls are unavailable; no candidate launch was attempted."""


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


def _nonnegative(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return parsed


def _minimax_duration(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 4.0 <= parsed <= 15.0:
        raise argparse.ArgumentTypeError("MiniMax-H3 duration must be in [4, 15] seconds")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--protocol",
        choices=("formal", "diagnostic", "lightweight"),
        default="formal",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--evaluator",
        type=Path,
        default=Path(__file__).with_name("host_weight_runtime_eval.py"),
    )
    parser.add_argument(
        "--offloaders",
        nargs="+",
        choices=OFFLOADERS,
        default=list(OFFLOADERS),
    )
    parser.add_argument("--repetitions", type=_positive, default=FORMAL_REPETITIONS)
    parser.add_argument("--warmup-requests", type=_positive, default=FORMAL_WARMUPS)
    parser.add_argument("--measured-requests", type=_positive, default=FORMAL_MEASURED_REQUESTS)
    parser.add_argument("--settle-s", type=_nonnegative_float, default=FORMAL_SETTLE_S)
    parser.add_argument("--idle-window-s", type=_positive_float, default=FORMAL_IDLE_WINDOW_S)
    parser.add_argument("--sample-interval-s", type=_positive_float, default=FORMAL_SAMPLE_INTERVAL_S)
    parser.add_argument("--dp-size", type=_positive, default=4)
    parser.add_argument("--steps", type=_minimax_steps, default=2)
    parser.add_argument("--duration", type=_minimax_duration, default=4.0)
    parser.add_argument("--width", type=_positive, default=448)
    parser.add_argument("--height", type=_positive, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hwr-wait-timeout-s", type=_positive_float, default=3600.0)
    parser.add_argument(
        "--process-exit-timeout-s",
        type=_positive_float,
        default=FORMAL_PROCESS_EXIT_TIMEOUT_S,
        help=(
            "Maximum post-close wait for observed engine processes; survivors still fail the evaluator cleanup gate."
        ),
    )
    parser.add_argument("--launch-timeout-s", type=_positive_float, default=7200.0)
    parser.add_argument("--device-idle-timeout-s", type=_positive_float, default=300.0)
    parser.add_argument("--device-idle-settle-s", type=_nonnegative_float, default=5.0)
    parser.add_argument("--cuda-visible-devices", default="0,1,2,3")
    parser.add_argument(
        "--cgroup-parent",
        type=Path,
        help="Writable delegated cgroup v2 parent; required in formal mode.",
    )
    parser.add_argument(
        "--allow-global-drop-caches",
        action="store_true",
        help="Explicitly authorize /proc/sys/vm/drop_caches before each formal launch.",
    )
    parser.add_argument(
        "--diagnostic-file-cache-eviction",
        action="store_true",
        help="Use file-scoped fadvise+fincore in diagnostic mode; never qualifies as global control.",
    )
    parser.add_argument(
        "--pinned-slot-budget-bytes",
        type=_nonnegative,
        help=(
            "Optional assertion for the process-local pinned-slot budget P. "
            "Formal ceilings always use the canonical value reported by every "
            "runtime worker and fail closed when that evidence is absent or differs."
        ),
    )
    parser.add_argument("--rtol", type=_nonnegative_float, default=1e-4)
    parser.add_argument("--atol", type=_nonnegative_float, default=1e-4)
    parser.add_argument(
        "--allow-nonexclusive-gpus",
        action="store_true",
        help="Diagnostic only; formal mode rejects this option.",
    )
    return parser.parse_args(argv)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_text(argv: Sequence[str], *, cwd: Path | None = None) -> str:
    return subprocess.run(
        list(argv),
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repository_root(start: Path) -> Path:
    output = _run_text(("git", "-C", str(start), "rev-parse", "--show-toplevel"))
    return Path(output).resolve()


def _git_provenance(repository: Path) -> dict[str, Any]:
    head = _run_text(("git", "-C", str(repository), "rev-parse", "HEAD"))
    status = subprocess.run(
        ("git", "-C", str(repository), "status", "--porcelain=v1", "-z", "--untracked-files=all"),
        check=True,
        capture_output=True,
    ).stdout
    tracked_diff = subprocess.run(
        ("git", "-C", str(repository), "diff", "--binary", "HEAD", "--"),
        check=True,
        capture_output=True,
    ).stdout
    untracked: list[dict[str, Any]] = []
    for entry in status.split(b"\0"):
        if not entry.startswith(b"?? "):
            continue
        relative = entry[3:].decode(errors="surrogateescape")
        path = repository / relative
        if path.is_file():
            untracked.append(
                {
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    return {
        "repository": str(repository),
        "commit": head,
        "clean": not status,
        "status_porcelain_sha256": hashlib.sha256(status).hexdigest(),
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "status_entries": [entry.decode(errors="backslashreplace") for entry in status.split(b"\0") if entry],
        "untracked_files": untracked,
    }


def _runtime_versions(python: Path) -> dict[str, Any]:
    script = r"""
import importlib.metadata, json, platform, sys
def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
result = {
    "python": sys.version,
    "python_executable": sys.executable,
    "platform": platform.platform(),
    "vllm": version("vllm"),
    "vllm_omni": version("vllm-omni"),
    "torch": version("torch"),
    "diffusers": version("diffusers"),
    "numpy": version("numpy"),
}
try:
    import torch
    result["torch_cuda"] = torch.version.cuda
except Exception as exc:
    result["torch_import_error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(result, sort_keys=True))
"""
    completed = subprocess.run(
        (str(python), "-c", script),
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _nvidia_smi(query: str) -> list[str]:
    completed = subprocess.run(
        ("nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"),
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _gpu_inventory() -> list[dict[str, Any]]:
    inventory = []
    for line in _nvidia_smi("gpu=index,uuid,name,driver_version,memory.total"):
        fields = [field.strip() for field in line.split(",", 4)]
        if len(fields) != 5:
            raise RuntimeError(f"unexpected nvidia-smi inventory row: {line!r}")
        index, gpu_uuid, name, driver, memory_mib = fields
        inventory.append(
            {
                "index": int(index),
                "uuid": gpu_uuid,
                "name": name,
                "driver_version": driver,
                "memory_total_mib": int(memory_mib),
            }
        )
    return inventory


def _selected_gpus(inventory: Sequence[dict[str, Any]], visible: str) -> list[dict[str, Any]]:
    selected = []
    for token in (value.strip() for value in visible.split(",") if value.strip()):
        matches = [
            gpu
            for gpu in inventory
            if str(gpu["index"]) == token or str(gpu["uuid"]) == token or str(gpu["uuid"]).startswith(token)
        ]
        if len(matches) != 1:
            raise RuntimeError(f"cannot resolve CUDA_VISIBLE_DEVICES token {token!r}")
        selected.append(matches[0])
    return selected


def _selected_compute_processes(selected: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    selected_uuids = {gpu["uuid"] for gpu in selected}
    processes = []
    for line in _nvidia_smi("compute-apps=gpu_uuid,pid,process_name,used_memory"):
        fields = [field.strip() for field in line.split(",", 3)]
        if len(fields) != 4:
            raise RuntimeError(f"unexpected nvidia-smi process row: {line!r}")
        gpu_uuid, pid, name, memory_mib = fields
        if gpu_uuid in selected_uuids:
            processes.append(
                {
                    "gpu_uuid": gpu_uuid,
                    "pid": int(pid),
                    "process_name": name,
                    "used_memory_mib": None if memory_mib in {"N/A", "[N/A]"} else int(memory_mib),
                }
            )
    return processes


def _wait_for_idle_devices(
    selected: Sequence[dict[str, Any]],
    *,
    timeout_s: float,
    settle_s: float,
) -> dict[str, Any]:
    started = time.monotonic()
    idle_since: float | None = None
    observations = 0
    last_processes: list[dict[str, Any]] = []
    while time.monotonic() - started <= timeout_s:
        observations += 1
        last_processes = _selected_compute_processes(selected)
        now = time.monotonic()
        if not last_processes:
            idle_since = now if idle_since is None else idle_since
            if now - idle_since >= settle_s:
                return {
                    "passed": True,
                    "wait_s": now - started,
                    "settle_s": settle_s,
                    "observations": observations,
                    "selected_gpu_uuids": [gpu["uuid"] for gpu in selected],
                }
        else:
            idle_since = None
        time.sleep(0.5)
    return {
        "passed": False,
        "wait_s": time.monotonic() - started,
        "settle_s": settle_s,
        "observations": observations,
        "selected_gpu_uuids": [gpu["uuid"] for gpu in selected],
        "remaining_compute_processes": last_processes,
    }


_ENV_KEYS = frozenset(
    {
        "CUDA_VISIBLE_DEVICES",
        "CUDA_HOME",
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
_ENV_PREFIXES = ("CUDA_", "NCCL_", "TORCH_", "VLLM_", "VLLM_OMNI_")
_SECRET_FRAGMENTS = ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "API_KEY", "ACCESS_KEY")


def _archived_environment(environment: dict[str, str]) -> dict[str, Any]:
    values: dict[str, str] = {}
    redacted: list[str] = []
    for key, value in sorted(environment.items()):
        if key not in _ENV_KEYS and not key.startswith(_ENV_PREFIXES):
            continue
        if any(fragment in key.upper() for fragment in _SECRET_FRAGMENTS):
            redacted.append(key)
        else:
            values[key] = value
    return {
        "allowlist": sorted(_ENV_KEYS),
        "prefixes": list(_ENV_PREFIXES),
        "values": values,
        "redacted_keys": redacted,
    }


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "median": float(statistics.median(numeric)) if numeric else None,
        "p95": _percentile(numeric, 95.0),
        "minimum": min(numeric) if numeric else None,
        "maximum": max(numeric) if numeric else None,
        "values": numeric,
    }


def scenario_order(repetition: int) -> tuple[str, str, str]:
    return ("L4", "C4", "W4") if repetition % 2 == 0 else ("C4", "W4", "L4")


def _regular_files(roots: Iterable[Path]) -> list[Path]:
    selected: dict[Path, None] = {}
    for root in roots:
        if root.is_file():
            selected[root.resolve()] = None
        elif root.is_dir():
            for path in root.rglob("*"):
                if path.is_file():
                    selected[path.resolve()] = None
    return sorted(selected)


def _mapped_processes(files: Sequence[Path]) -> list[dict[str, Any]]:
    targets = {str(path.resolve()) for path in files}
    matches: list[dict[str, Any]] = []
    if not targets:
        return matches
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            mapped = []
            for line in (entry / "maps").read_text().splitlines():
                fields = line.split(maxsplit=5)
                if len(fields) == 6 and fields[5].removesuffix(" (deleted)") in targets:
                    mapped.append(fields[5])
            if mapped:
                matches.append({"pid": int(entry.name), "files": sorted(set(mapped))})
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return matches


def _fincore(files: Sequence[Path]) -> dict[str, Any]:
    if not files:
        return {"available": shutil.which("fincore") is not None, "files": [], "resident_bytes": 0}
    executable = shutil.which("fincore")
    if executable is None:
        return {"available": False, "files": [], "resident_bytes": None, "error": "fincore not found"}
    records: list[dict[str, Any]] = []
    for start in range(0, len(files), 128):
        batch = files[start : start + 128]
        completed = subprocess.run(
            (executable, "--json", "--bytes", *(str(path) for path in batch)),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            return {
                "available": True,
                "files": records,
                "resident_bytes": None,
                "error": completed.stderr.strip() or f"fincore exited {completed.returncode}",
            }
        payload = json.loads(completed.stdout)
        records.extend(payload.get("fincore", ()))
    resident = sum(int(record.get("res", 0)) for record in records)
    return {"available": True, "files": records, "resident_bytes": resident}


def _file_scoped_evict(files: Sequence[Path]) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return [{"path": "*", "error": "POSIX_FADV_DONTNEED unavailable"}]
    os.sync()
    for path in files:
        try:
            with path.open("rb", buffering=0) as source:
                os.posix_fadvise(source.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError as exc:
            errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    return errors


def _cache_attestation(
    *,
    files: Sequence[Path],
    launch_id: str,
    scenario: str,
    protocol: str,
    allow_global_drop_caches: bool,
    diagnostic_file_eviction: bool,
) -> dict[str, Any]:
    before = _fincore(files)
    mapped = _mapped_processes(files)
    errors: list[dict[str, str]] = []
    scope = "not_controlled"
    global_drop_performed = False
    if protocol == "formal":
        scope = "global_then_file_verified"
        if not allow_global_drop_caches:
            errors.append({"path": "/proc/sys/vm/drop_caches", "error": "explicit authorization missing"})
        elif not os.access("/proc/sys/vm/drop_caches", os.W_OK):
            errors.append({"path": "/proc/sys/vm/drop_caches", "error": "not writable"})
        elif mapped:
            errors.append({"path": "/proc/*/maps", "error": "target files remain mapped by another process"})
        else:
            os.sync()
            Path("/proc/sys/vm/drop_caches").write_text("3\n", encoding="ascii")
            global_drop_performed = True
            errors.extend(_file_scoped_evict(files))
    elif diagnostic_file_eviction:
        scope = "file_scoped_advice_only"
        errors.extend(_file_scoped_evict(files))
    after = _fincore(files)
    zero = after.get("resident_bytes") == 0
    return {
        "schema_version": 1,
        "launch_id": launch_id,
        "scenario": scenario,
        "protocol": protocol,
        "scope": scope,
        "created_unix_ns": time.time_ns(),
        "target_file_count": len(files),
        "target_bytes": sum(path.stat().st_size for path in files),
        "mapped_processes_before": mapped,
        "before": before,
        "after": after,
        "global_drop_performed": global_drop_performed,
        "errors": errors,
        "verified_zero_resident_bytes": zero and not errors,
        "formal_control_verified": scope == "global_then_file_verified" and zero and not errors,
    }


def _cgroup_v2_probe(parent: Path | None) -> dict[str, Any]:
    if parent is None:
        return {"supported": False, "reason": "--cgroup-parent was not provided"}
    reasons = []
    if not (parent / "cgroup.controllers").is_file():
        reasons.append("not a cgroup v2 directory")
    if not os.access(parent, os.W_OK | os.X_OK):
        reasons.append("parent is not writable/delegated")
    subtree = ""
    try:
        subtree = (parent / "cgroup.subtree_control").read_text()
    except OSError as exc:
        reasons.append(f"cannot read cgroup.subtree_control: {exc}")
    if "memory" not in subtree.split():
        reasons.append("memory controller is not enabled for child cgroups")
    return {
        "supported": not reasons,
        "path": str(parent),
        "subtree_control": subtree.split(),
        "reasons": reasons,
    }


def _current_memory_cgroup_path() -> Path | None:
    try:
        entries = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        return None
    for entry in entries:
        hierarchy, controllers, relative = entry.split(":", 2)
        if hierarchy == "0" and not controllers:
            candidate = Path("/sys/fs/cgroup") / relative.lstrip("/")
            return candidate if candidate.is_dir() else Path("/sys/fs/cgroup")
        if "memory" in controllers.split(","):
            root = Path("/sys/fs/cgroup/memory")
            candidate = root / relative.lstrip("/")
            return candidate if candidate.is_dir() else root if root.is_dir() else None
    return None


@dataclass
class _CgroupLease:
    path: Path

    @classmethod
    def create(cls, parent: Path, name: str) -> _CgroupLease:
        path = parent / name
        path.mkdir(mode=0o755)
        required = ("cgroup.procs", "cgroup.kill", "memory.current", "memory.peak")
        missing = [file for file in required if not (path / file).exists()]
        if missing:
            try:
                path.rmdir()
            except OSError:
                pass
            raise QualificationBlockedError(f"fresh cgroup lacks required files: {missing}")
        if (path / "cgroup.procs").read_text().split():
            raise QualificationBlockedError("fresh cgroup is unexpectedly populated")
        return cls(path)

    def metrics(self) -> dict[str, Any]:
        result: dict[str, Any] = {"path": str(self.path)}
        for name in ("memory.current", "memory.peak"):
            try:
                result[name.replace(".", "_") + "_bytes"] = int((self.path / name).read_text().strip())
            except (OSError, ValueError):
                result[name.replace(".", "_") + "_bytes"] = None
        try:
            result["remaining_pids"] = [int(value) for value in (self.path / "cgroup.procs").read_text().split()]
        except (OSError, ValueError):
            result["remaining_pids"] = None
        return result

    def close(self) -> dict[str, Any]:
        metrics = self.metrics()
        if metrics.get("remaining_pids"):
            metrics["forced_kill"] = True
            try:
                (self.path / "cgroup.kill").write_text("1\n", encoding="ascii")
            except OSError as exc:
                metrics["cleanup"] = f"kill_failed: {type(exc).__name__}: {exc}"
                return metrics
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                try:
                    remaining = [int(value) for value in (self.path / "cgroup.procs").read_text().split()]
                except (OSError, ValueError):
                    remaining = [-1]
                if not remaining:
                    break
                time.sleep(0.1)
            metrics["remaining_pids_after_forced_kill"] = remaining
            if remaining:
                metrics["cleanup"] = "retained_after_forced_kill"
                return metrics
        else:
            metrics["forced_kill"] = False
        deadline = time.monotonic() + 5.0
        while True:
            try:
                self.path.rmdir()
                metrics["cleanup"] = "removed"
                break
            except OSError as exc:
                if time.monotonic() >= deadline:
                    metrics["cleanup"] = f"remove_failed: {type(exc).__name__}: {exc}"
                    break
                time.sleep(0.1)
        return metrics


def _artifact_fingerprint(root: Path) -> dict[str, Any]:
    artifacts = root / "artifacts"
    records: list[dict[str, Any]] = []
    if artifacts.is_dir():
        for path in _regular_files((artifacts,)):
            stat = path.stat()
            records.append(
                {
                    "path": str(path.relative_to(root.resolve())),
                    "size": stat.st_size,
                    "mode": stat.st_mode,
                    "sha256": _sha256_file(path),
                }
            )
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return {
        "file_count": len(records),
        "storage_span_bytes": sum((record["size"] + 4095) // 4096 * 4096 for record in records),
        "tree_sha256": hashlib.sha256(canonical).hexdigest(),
        "files": records,
    }


def _common_child_argv(args: argparse.Namespace) -> list[str]:
    return [
        str(args.python),
        str(args.evaluator),
        "--model",
        str(args.model),
        "--dp-size",
        str(args.dp_size),
        "--warmup-requests",
        str(args.warmup_requests),
        "--measured-requests",
        str(args.measured_requests),
        "--settle-s",
        str(args.settle_s),
        "--idle-window-s",
        str(args.idle_window_s),
        "--sample-interval-s",
        str(args.sample_interval_s),
        "--steps",
        str(args.steps),
        "--duration",
        str(args.duration),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--seed",
        str(args.seed),
        "--hwr-wait-timeout-s",
        str(args.hwr_wait_timeout_s),
        "--process-exit-timeout-s",
        str(args.process_exit_timeout_s),
    ]


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=15)
        return
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=15)


def _launch_child(
    *,
    args: argparse.Namespace,
    argv: list[str],
    environment: dict[str, str],
    launch_dir: Path,
    cgroup: _CgroupLease | None,
) -> dict[str, Any]:
    stdout_path = launch_dir / "stdout.log"
    stderr_path = launch_dir / "stderr.log"
    actual_argv = list(argv)
    child_setup = None
    if cgroup is not None:
        cgroup_procs = os.fsencode(cgroup.path / "cgroup.procs")

        def enter_measurement_cgroup() -> None:
            # Popen invokes this after fork and before exec, so the candidate
            # interpreter and every worker are born inside the fresh cgroup.
            # Keep the child-side operation to async-signal-safe syscalls.
            descriptor = os.open(cgroup_procs, os.O_WRONLY)
            try:
                os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
            finally:
                os.close(descriptor)

        child_setup = enter_measurement_cgroup
    started_ns = time.time_ns()
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            actual_argv,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            preexec_fn=child_setup,
        )
        try:
            returncode = process.wait(timeout=args.launch_timeout_s)
            timeout = False
        except subprocess.TimeoutExpired:
            timeout = True
            _terminate_process_group(process)
            returncode = process.returncode
    return {
        "candidate_argv": argv,
        "actual_argv": actual_argv,
        "environment": _archived_environment(environment),
        "pid": process.pid,
        "returncode": returncode,
        "timed_out": timeout,
        "started_unix_ns": started_ns,
        "finished_unix_ns": time.time_ns(),
        "stdout": str(stdout_path.resolve()),
        "stderr": str(stderr_path.resolve()),
    }


def _output_archives(run: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    archives: dict[tuple[int, str], dict[str, Any]] = {}
    for output in run.get("outputs", ()):
        for media, record in output.get("tensor_archives", {}).items():
            archives[(int(output["request_index"]), media)] = record
    return archives


def compare_outputs(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    reference_archives = _output_archives(reference)
    candidate_archives = _output_archives(candidate)
    errors: list[str] = []
    comparisons: list[dict[str, Any]] = []
    reference_scope = reference.get("output_capture", {}).get("scope")
    candidate_scope = candidate.get("output_capture", {}).get("scope")
    if reference_scope != "pre_encoding" or candidate_scope != "pre_encoding":
        errors.append(
            f"output capture is not verified pre-encoding: reference={reference_scope!r}, candidate={candidate_scope!r}"
        )
    if set(reference_archives) != set(candidate_archives):
        errors.append(
            f"archive keys differ: reference={sorted(reference_archives)}, candidate={sorted(candidate_archives)}"
        )
    for key in sorted(set(reference_archives) & set(candidate_archives)):
        expected = np.load(reference_archives[key]["path"], mmap_mode="r", allow_pickle=False)
        actual = np.load(candidate_archives[key]["path"], mmap_mode="r", allow_pickle=False)
        shape_equal = expected.shape == actual.shape
        dtype_equal = expected.dtype == actual.dtype
        finite_pattern_equal = bool(np.array_equal(np.isfinite(expected), np.isfinite(actual)))
        close = (
            shape_equal
            and dtype_equal
            and finite_pattern_equal
            and bool(np.allclose(expected, actual, rtol=rtol, atol=atol, equal_nan=True))
        )
        maximum_absolute = None
        maximum_relative = None
        if shape_equal and expected.size:
            expected_float = expected.astype(np.float64, copy=False)
            actual_float = actual.astype(np.float64, copy=False)
            absolute = np.abs(actual_float - expected_float)
            finite = np.isfinite(absolute)
            if finite.any():
                maximum_absolute = float(absolute[finite].max())
                denominator = np.maximum(np.abs(expected_float), atol)
                relative = np.divide(absolute, denominator, where=np.isfinite(denominator))
                finite_relative = np.isfinite(relative)
                if finite_relative.any():
                    maximum_relative = float(relative[finite_relative].max())
        comparison = {
            "request_index": key[0],
            "media": key[1],
            "shape_equal": shape_equal,
            "dtype_equal": dtype_equal,
            "finite_pattern_equal": finite_pattern_equal,
            "allclose": close,
            "max_absolute_error": maximum_absolute,
            "max_relative_error": maximum_relative,
        }
        comparisons.append(comparison)
        if not close:
            errors.append(f"output differs for request={key[0]} media={key[1]}: {comparison}")
    return {
        "passed": bool(comparisons) and not errors,
        "rtol": rtol,
        "atol": atol,
        "scope": "pre_encoding",
        "reference_capture": reference.get("output_capture"),
        "candidate_capture": candidate.get("output_capture"),
        "comparisons": comparisons,
        "errors": errors,
    }


def _paired_configuration_check(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    normalized = {}
    for scenario, run in runs.items():
        engine = dict(run.get("candidate", {}).get("engine_kwargs", {}))
        for key in (
            "host_weight_runtime_mode",
            "host_weight_runtime_root",
            "host_weight_runtime_required",
        ):
            engine.pop(key, None)
        normalized[scenario] = {
            "offloader": run.get("candidate", {}).get("offloader"),
            "engine_kwargs": engine,
            "request": run.get("request"),
            "execution_semantics": run.get("execution_semantics"),
        }
    canonical = {
        scenario: json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
        for scenario, value in normalized.items()
    }
    passed = len(set(canonical.values())) == 1
    return {
        "passed": passed,
        "normalized_sha256": {
            scenario: hashlib.sha256(value.encode()).hexdigest() for scenario, value in canonical.items()
        },
        "allowed_differences": [
            "host_weight_runtime_mode",
            "host_weight_runtime_root",
            "host_weight_runtime_required",
            "producer role",
            "measurement instrumentation",
        ],
    }


def _steady_pss_bytes(run: dict[str, Any]) -> float | None:
    value = run.get("memory_measurement", {}).get("idle_steady_state", {}).get("pss_kib", {}).get("median")
    return None if value is None else float(value) * 1024


def _latency_median(run: dict[str, Any]) -> float | None:
    value = run.get("timings", {}).get("request_latency_s", {}).get("median")
    return None if value is None else float(value)


def _device_peak_bytes(run: dict[str, Any]) -> int | None:
    values = run.get("sampled_device_peak_mib", ())
    return max(int(value) for value in values) * 1024**2 if values else None


def _worker_baseline(run: dict[str, Any], pid: int) -> tuple[int | None, int | None, bool]:
    records = run.get("host_weight_runtime_evidence", {}).get("worker_records", ())
    for record in records:
        if int(record.get("rank", {}).get("pid", -1)) != pid:
            continue
        pre_resolve = record.get("pre_resolve")
        if isinstance(pre_resolve, dict):
            pss = pre_resolve.get("pss_kib")
            private = pre_resolve.get("private_and_locked_kib")
            if pss is not None and private is not None:
                return int(pss), int(private), True
    first = run.get("memory_measurement", {}).get("first_observed_by_pid", {}).get(str(pid), {})
    metrics = first.get("metrics", {})
    pss = metrics.get("pss_kib")
    private = metrics.get("private_and_locked_kib")
    return (
        None if pss is None else int(pss),
        None if private is None else int(private),
        False,
    )


def _legacy_delta_bytes(runs: Sequence[dict[str, Any]]) -> tuple[int | None, bool]:
    deltas: list[int] = []
    exact = True
    for run in runs:
        for pid_text, peaks in run.get("memory_measurement", {}).get("worker_peaks", {}).items():
            baseline_pss, _, is_exact = _worker_baseline(run, int(pid_text))
            if baseline_pss is None:
                continue
            exact = exact and is_exact
            deltas.append(max(0, int(peaks.get("pss_kib", 0)) - baseline_pss) * 1024)
    return (max(deltas) if deltas else None, exact)


def _role_for_pid(run: dict[str, Any], pid: int) -> str | None:
    for record in run.get("host_weight_runtime_evidence", {}).get("worker_records", ()):
        if int(record.get("rank", {}).get("pid", -1)) == pid:
            return record.get("claim_role")
    return None


def _runtime_pinned_slot_budget(
    candidate_runs: Sequence[dict[str, Any]],
    *,
    asserted_budget_bytes: int | None,
) -> dict[str, Any]:
    """Resolve P only from canonical per-worker runtime evidence.

    The command-line value is an equality assertion, never the source of a
    ceiling. This prevents a rerun from making a failed host-memory gate pass
    merely by supplying a larger budget.
    """

    observations: list[dict[str, Any]] = []
    errors: list[str] = []
    for run in candidate_runs:
        scenario = run.get("launch", {}).get("scenario")
        repetition = run.get("suite", {}).get("repetition")
        records = run.get("host_weight_runtime_evidence", {}).get("worker_records", ())
        if not records:
            errors.append(f"{scenario} repetition {repetition} has no worker records")
            continue
        for record in records:
            rank = record.get("rank", {}).get("global_rank")
            value = record.get("pinned_slot_budget_bytes")
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                errors.append(
                    f"{scenario} repetition {repetition} rank {rank} lacks canonical pinned_slot_budget_bytes"
                )
                continue
            observations.append(
                {
                    "scenario": scenario,
                    "repetition": repetition,
                    "rank": rank,
                    "value_bytes": value,
                }
            )

    values = {int(item["value_bytes"]) for item in observations}
    if len(values) > 1:
        errors.append(f"runtime workers reported inconsistent pinned-slot budgets: {sorted(values)}")
    derived = next(iter(values)) if len(values) == 1 else None
    if asserted_budget_bytes is not None and derived is not None and asserted_budget_bytes != derived:
        errors.append(
            "--pinned-slot-budget-bytes assertion differs from runtime evidence: "
            f"asserted={asserted_budget_bytes}, runtime={derived}"
        )
    return {
        "passed": derived is not None and not errors,
        "source": "host_weight_runtime_evidence.worker_records[].pinned_slot_budget_bytes",
        "value_bytes": derived,
        "asserted_value_bytes": asserted_budget_bytes,
        "observations": observations,
        "errors": errors,
    }


def _ceiling_checks(
    *,
    candidate_runs: Sequence[dict[str, Any]],
    legacy_delta_bytes: int | None,
    artifact_span_bytes: int,
    pinned_budget_bytes: int | None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    exact_pre_resolve = True
    if legacy_delta_bytes is None or pinned_budget_bytes is None:
        return {
            "passed": False,
            "reason": "legacy delta or canonical runtime pinned-slot budget unavailable",
            "checks": checks,
        }
    for run in candidate_runs:
        scenario = run.get("launch", {}).get("scenario")
        for pid_text, peaks in run.get("memory_measurement", {}).get("worker_peaks", {}).items():
            pid = int(pid_text)
            baseline_pss, baseline_private, is_exact = _worker_baseline(run, pid)
            exact_pre_resolve = exact_pre_resolve and is_exact
            if baseline_pss is None or baseline_private is None:
                checks.append({"scenario": scenario, "pid": pid, "passed": False, "reason": "P0 unavailable"})
                continue
            role = _role_for_pid(run, pid)
            pss_growth = max(0, int(peaks.get("pss_kib", 0)) - baseline_pss) * 1024
            if scenario == "C4" and role == "builder":
                pss_ceiling = legacy_delta_bytes + artifact_span_bytes + CEILING_ALLOWANCE_BYTES
            else:
                pss_ceiling = artifact_span_bytes + pinned_budget_bytes + CEILING_ALLOWANCE_BYTES
            private_growth = (
                max(
                    0,
                    int(peaks.get("private_and_locked_kib", 0)) - baseline_private,
                )
                * 1024
            )
            private_ceiling = pinned_budget_bytes + CEILING_ALLOWANCE_BYTES
            checks.append(
                {
                    "scenario": scenario,
                    "pid": pid,
                    "role": role,
                    "pss_growth_bytes": pss_growth,
                    "pss_ceiling_bytes": pss_ceiling,
                    "pss_passed": pss_growth <= pss_ceiling,
                    "private_growth_bytes": private_growth,
                    "private_ceiling_bytes": private_ceiling,
                    "private_passed": private_growth <= private_ceiling,
                    "passed": pss_growth <= pss_ceiling and private_growth <= private_ceiling,
                }
            )
    return {
        "passed": bool(checks) and all(check["passed"] for check in checks) and exact_pre_resolve,
        "exact_pre_resolve": exact_pre_resolve,
        "legacy_delta_bytes": legacy_delta_bytes,
        "artifact_span_bytes": artifact_span_bytes,
        "pinned_slot_budget_bytes": pinned_budget_bytes,
        "allowance_bytes": CEILING_ALLOWANCE_BYTES,
        "checks": checks,
    }


def aggregate_results(
    runs: Sequence[dict[str, Any]],
    *,
    pinned_budget_bytes: int | None,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for offloader in OFFLOADERS:
        relevant = [run for run in runs if run.get("candidate", {}).get("offloader") == offloader]
        if not relevant:
            continue
        by_scenario = {
            scenario: [run for run in relevant if run.get("launch", {}).get("scenario") == scenario]
            for scenario in SCENARIO_MODES
        }
        scenario_stats: dict[str, Any] = {}
        for scenario, scenario_runs in by_scenario.items():
            scenario_stats[scenario] = {
                "startup_s": _distribution(
                    [
                        float(run["timings"]["startup_s"])
                        for run in scenario_runs
                        if "startup_s" in run.get("timings", {})
                    ]
                ),
                "time_to_ready_s": _distribution(
                    [
                        float(run["timings"]["time_to_ready_s"])
                        for run in scenario_runs
                        if "time_to_ready_s" in run.get("timings", {})
                    ]
                ),
                "request_latency_per_run_median_s": _distribution(
                    [value for run in scenario_runs if (value := _latency_median(run)) is not None]
                ),
                "steady_pss_per_run_median_bytes": _distribution(
                    [value for run in scenario_runs if (value := _steady_pss_bytes(run)) is not None]
                ),
                "device_peak_bytes": _distribution(
                    [float(value) for run in scenario_runs if (value := _device_peak_bytes(run)) is not None]
                ),
                "cgroup_peak_bytes": _distribution(
                    [
                        float(value)
                        for run in scenario_runs
                        if (value := run.get("memory_measurement", {}).get("cgroup_peak_bytes")) is not None
                    ]
                ),
            }
        paired: list[dict[str, Any]] = []
        repetitions = sorted({int(run["suite"]["repetition"]) for run in relevant})
        artifact_spans: list[int] = []
        for repetition in repetitions:
            selected = {
                run["launch"]["scenario"]: run for run in relevant if int(run["suite"]["repetition"]) == repetition
            }
            if set(selected) != set(SCENARIO_MODES):
                paired.append({"repetition": repetition, "complete": False})
                continue
            l4_pss = _steady_pss_bytes(selected["L4"])
            c4_pss = _steady_pss_bytes(selected["C4"])
            w4_pss = _steady_pss_bytes(selected["W4"])
            l4_latency = _latency_median(selected["L4"])
            l4_device = _device_peak_bytes(selected["L4"])
            l4_ready = selected["L4"].get("timings", {}).get("time_to_ready_s")
            candidate_values = {}
            for scenario, pss in (("C4", c4_pss), ("W4", w4_pss)):
                latency = _latency_median(selected[scenario])
                device = _device_peak_bytes(selected[scenario])
                candidate_ready = selected[scenario].get("timings", {}).get("time_to_ready_s")
                candidate_values[scenario] = {
                    "pss_saving_bytes": None if l4_pss is None or pss is None else l4_pss - pss,
                    "latency_regression_fraction": (
                        None if l4_latency in (None, 0) or latency is None else latency / l4_latency - 1.0
                    ),
                    "device_peak_delta_bytes": (None if l4_device is None or device is None else device - l4_device),
                    "time_to_ready_regression_fraction": (
                        None
                        if l4_ready in (None, 0) or candidate_ready is None
                        else float(candidate_ready) / float(l4_ready) - 1.0
                    ),
                    "output_comparison": selected[scenario].get("suite", {}).get("output_comparison"),
                }
            artifact_spans.append(int(selected["C4"]["suite"]["artifact_fingerprint"]["storage_span_bytes"]))
            paired.append({"repetition": repetition, "complete": True, **candidate_values})

        savings = {
            scenario: _distribution(
                [
                    float(pair[scenario]["pss_saving_bytes"])
                    for pair in paired
                    if pair.get("complete") and pair[scenario]["pss_saving_bytes"] is not None
                ]
            )
            for scenario in ("C4", "W4")
        }
        legacy_delta, exact_delta = _legacy_delta_bytes(by_scenario["L4"])
        pinned_budget = _runtime_pinned_slot_budget(
            [*by_scenario["C4"], *by_scenario["W4"]],
            asserted_budget_bytes=pinned_budget_bytes,
        )
        ceilings = _ceiling_checks(
            candidate_runs=[*by_scenario["C4"], *by_scenario["W4"]],
            legacy_delta_bytes=legacy_delta,
            artifact_span_bytes=max(artifact_spans) if artifact_spans else 0,
            pinned_budget_bytes=(pinned_budget["value_bytes"] if pinned_budget["passed"] else None),
        )
        ceilings["pinned_slot_budget_derivation"] = pinned_budget
        latency_regressions = {
            scenario: [
                float(pair[scenario]["latency_regression_fraction"])
                for pair in paired
                if pair.get("complete") and pair[scenario]["latency_regression_fraction"] is not None
            ]
            for scenario in ("C4", "W4")
        }
        latency_passed = {
            scenario: bool(values) and statistics.median(values) <= 0.10
            for scenario, values in latency_regressions.items()
        }
        warm_ready_regressions = [
            float(pair["W4"]["time_to_ready_regression_fraction"])
            for pair in paired
            if pair.get("complete") and pair["W4"]["time_to_ready_regression_fraction"] is not None
        ]
        warm_ready_passed = bool(warm_ready_regressions) and statistics.median(warm_ready_regressions) <= 0.10
        device_passed = all(
            pair[scenario]["device_peak_delta_bytes"] is not None
            and pair[scenario]["device_peak_delta_bytes"]
            <= max(
                1024**3,
                0.05
                * float(
                    _device_peak_bytes(
                        next(
                            run
                            for run in relevant
                            if run["suite"]["repetition"] == pair["repetition"] and run["launch"]["scenario"] == "L4"
                        )
                    )
                    or 0
                ),
            )
            for pair in paired
            if pair.get("complete")
            for scenario in ("C4", "W4")
        )
        output_passed = all(
            bool(pair[scenario]["output_comparison"] and pair[scenario]["output_comparison"].get("passed"))
            for pair in paired
            if pair.get("complete")
            for scenario in ("C4", "W4")
        )
        saving_passed = all(
            savings[scenario]["median"] is not None and savings[scenario]["median"] >= MINIMUM_PSS_SAVING_BYTES
            for scenario in ("C4", "W4")
        )
        report[offloader] = {
            "scenario_statistics": scenario_stats,
            "paired_repetitions": paired,
            "paired_pss_savings_bytes": savings,
            "legacy_delta_exact_pre_resolve": exact_delta,
            "pinned_slot_budget": pinned_budget,
            "latency_regressions_by_scenario": {
                scenario: _distribution(values) for scenario, values in latency_regressions.items()
            },
            "host_peak_ceilings": ceilings,
            "gates": {
                "paired_75_gib_pss_savings": saving_passed,
                "c4_latency_regression_at_most_10_percent": latency_passed["C4"],
                "w4_latency_regression_at_most_10_percent": latency_passed["W4"],
                "warm_time_to_ready_regression_at_most_10_percent": warm_ready_passed,
                "device_peak_delta_within_limit": device_passed,
                "output_parity": output_passed,
                "host_peak_ceilings": ceilings["passed"],
            },
        }
        report[offloader]["passed"] = all(report[offloader]["gates"].values())
    return report


def _formal_preflight(
    args: argparse.Namespace,
    provenance: dict[str, Any],
    runtime: dict[str, Any],
    selected_gpus: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    failures: list[str] = []
    cgroup = _cgroup_v2_probe(args.cgroup_parent)
    if not cgroup["supported"]:
        failures.append(f"fresh cgroup v2 unavailable: {cgroup.get('reasons') or cgroup.get('reason')}")
    if not args.allow_global_drop_caches:
        failures.append("--allow-global-drop-caches is required")
    elif not os.access("/proc/sys/vm/drop_caches", os.W_OK):
        failures.append("/proc/sys/vm/drop_caches is not writable")
    if shutil.which("fincore") is None:
        failures.append("fincore is unavailable")
    if not provenance["clean"]:
        failures.append("candidate worktree is dirty")
    if args.repetitions != FORMAL_REPETITIONS:
        failures.append(f"formal protocol requires exactly {FORMAL_REPETITIONS} repetitions")
    expected_measurement = {
        "warmup_requests": FORMAL_WARMUPS,
        "measured_requests": FORMAL_MEASURED_REQUESTS,
        "settle_s": FORMAL_SETTLE_S,
        "idle_window_s": FORMAL_IDLE_WINDOW_S,
        "sample_interval_s": FORMAL_SAMPLE_INTERVAL_S,
        "process_exit_timeout_s": FORMAL_PROCESS_EXIT_TIMEOUT_S,
    }
    for field, expected in expected_measurement.items():
        if getattr(args, field) != expected:
            failures.append(f"formal protocol requires {field}={expected}")
    if args.dp_size != 4:
        failures.append("formal protocol requires DP=4")
    if len(selected_gpus) != 4:
        failures.append(f"formal protocol requires exactly four selected GPUs, got {len(selected_gpus)}")
    if any(gpu.get("name") != "NVIDIA L20X" for gpu in selected_gpus):
        failures.append(f"selected GPU models differ from NVIDIA L20X: {[gpu.get('name') for gpu in selected_gpus]}")
    if any(gpu.get("driver_version") != "570.133.20" for gpu in selected_gpus):
        failures.append(
            f"selected GPU driver differs from 570.133.20: {[gpu.get('driver_version') for gpu in selected_gpus]}"
        )
    if args.allow_nonexclusive_gpus:
        failures.append("formal protocol forbids --allow-nonexclusive-gpus")
    model = args.model.resolve()
    if model.name != "FL2VA" or MODEL_SNAPSHOT not in model.parts:
        failures.append(f"model path is not the pinned {MODEL_SNAPSHOT}/FL2VA snapshot")
    for package, expected in EXPECTED_RUNTIME.items():
        if runtime.get(package) != expected:
            failures.append(f"runtime {package} differs: expected={expected!r}, actual={runtime.get(package)!r}")
    return {
        "passed": not failures,
        "failures": failures,
        "cgroup": cgroup,
        "drop_caches_writable": os.access("/proc/sys/vm/drop_caches", os.W_OK),
        "fincore": shutil.which("fincore"),
    }


def _synthetic_worker_records(scenario: str, launch_id: str, generation_id: str) -> list[dict[str, Any]]:
    records = []
    pid_base = {"L4": 3000, "C4": 1000, "W4": 2000}[scenario]
    for rank in range(4):
        if scenario == "L4":
            role = None
            outcome = "not_requested"
            mode = "disabled"
        elif scenario == "C4":
            role = "builder" if rank == 0 else "waiter"
            outcome = "ready"
            mode = "read_write"
        else:
            role = "cache_hit"
            outcome = "ready"
            mode = "read_only"
        records.append(
            {
                "rank": {"global_rank": rank, "pid": pid_base + rank},
                "runtime_mode": mode,
                "outcome": outcome,
                "claim_role": role,
                "artifact_key": None if scenario == "L4" else "synthetic-artifact",
                "generation_id": None if scenario == "L4" else generation_id,
                "pre_resolve": {"pss_kib": 1000, "private_and_locked_kib": 500},
                "pinned_slot_budget_bytes": (None if scenario == "L4" else SYNTHETIC_PINNED_SLOT_BUDGET_BYTES),
                "builder_started": (
                    {
                        "launch_id": launch_id,
                        "artifact_key": "synthetic-artifact",
                        "lease_id": "lease-0",
                        "builder_actor_id": "dp:0",
                    }
                    if scenario == "C4" and rank == 0
                    else None
                ),
            }
        )
    return records


def _synthetic_run(
    *,
    args: argparse.Namespace,
    offloader: str,
    scenario: str,
    repetition: int,
    launch_id: str,
    launch_dir: Path,
    generation_id: str,
) -> dict[str, Any]:
    pss_gib = {"L4": 120, "C4": 40, "W4": 38}[scenario]
    latency = {"L4": 1.0, "C4": 1.04, "W4": 1.02}[scenario]
    outputs = []
    for index in range(args.measured_requests):
        video = np.full((2, 2), index, dtype=np.float32)
        audio = np.full((4,), index / 10, dtype=np.float32)
        tensor_dir = launch_dir / "tensors"
        tensor_dir.mkdir(parents=True, exist_ok=True)
        video_path = tensor_dir / f"request-{index:02d}-video.npy"
        audio_path = tensor_dir / f"request-{index:02d}-audio.npy"
        np.save(video_path, video, allow_pickle=False)
        np.save(audio_path, audio, allow_pickle=False)
        outputs.append(
            {
                "request_index": index,
                "tensor_archives": {
                    "video": {"path": str(video_path), "dtype": "float32", "shape": [2, 2]},
                    "audio": {"path": str(audio_path), "dtype": "float32", "shape": [4]},
                },
            }
        )
    records = _synthetic_worker_records(scenario, launch_id, generation_id)
    worker_peaks = {
        str(record["rank"]["pid"]): {
            "pss_kib": 1100,
            "private_and_locked_kib": 600,
        }
        for record in records
    }
    return {
        "schema_version": 3,
        "status": "success",
        "launch": {"scenario": scenario, "launch_id": launch_id},
        "candidate": {"hwr_mode": SCENARIO_MODES[scenario], "offloader": offloader},
        "timings": {
            "startup_s": 2.0,
            "time_to_ready_s": 2.0,
            "request_latency_s": {**_distribution([latency] * args.measured_requests)},
        },
        "outputs": outputs,
        "output_capture": {
            "status": "captured",
            "scope": "pre_encoding",
            "request_count": len(outputs),
            "media": ["video", "audio"],
            "source": "lightweight synthetic pre-encoding arrays",
            "tensor_archives_written": True,
        },
        "host_weight_runtime_evidence": {"worker_records": records},
        "memory_measurement": {
            "idle_steady_state": {"pss_kib": {**_distribution([pss_gib * 1024**2] * 3)}},
            "worker_peaks": worker_peaks,
            "first_observed_by_pid": {},
            "cgroup_peak_bytes": int(pss_gib * 1024**3),
        },
        "sampled_device_peak_mib": [1000] * 4,
        "suite": {"repetition": repetition},
    }


def _run_multi_offloader_suite(args: argparse.Namespace) -> tuple[dict[str, Any], BaseException | None]:
    """Run each offloader as an isolated sub-suite and always attempt all.

    A failure in one offloader must not erase qualification evidence for the
    others. Each child uses the existing single-offloader protocol unchanged,
    while this wrapper merges their self-contained reports.
    """

    combined_root = args.work_root / (
        f"hwr-multi-offloader-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex}"
    )
    combined_root.mkdir(parents=True)
    child_results: list[tuple[str, Path, dict[str, Any], BaseException | None]] = []
    first_failure: BaseException | None = None
    for index, offloader in enumerate(args.offloaders):
        child_root = combined_root / f"{index:02d}-{offloader}"
        child_args = argparse.Namespace(**vars(args))
        child_args.offloaders = [offloader]
        child_args.work_root = child_root / "work"
        child_args.output = child_root / "suite.json"
        try:
            child_result, child_failure = _run_suite(child_args)
        except BaseException as exc:
            child_failure = exc
            child_result = {
                "schema_version": 1,
                "status": "blocked" if isinstance(exc, QualificationBlockedError) else "failed",
                "protocol": args.protocol,
                "qualification": False,
                "candidate": {},
                "environment": {},
                "configuration": {"offloaders": [offloader]},
                "formal_preflight": None,
                "launches": [],
                "aggregate": {},
                "errors": [{"type": type(exc).__name__, "message": str(exc)}],
            }
        _atomic_json(child_args.output, child_result)
        child_results.append((offloader, child_args.output, child_result, child_failure))
        if first_failure is None and child_failure is not None:
            first_failure = child_failure

    template = next(
        (child for _, _, child, _ in child_results if child.get("candidate")),
        child_results[0][2],
    )
    result = {
        "schema_version": template.get("schema_version", 1),
        "status": "running",
        "protocol": args.protocol,
        "qualification": False,
        "suite_argv": list(sys.argv),
        "candidate": template.get("candidate", {}),
        "environment": template.get("environment", {}),
        "configuration": {
            **template.get("configuration", {}),
            "offloaders": list(args.offloaders),
        },
        "formal_preflight": template.get("formal_preflight"),
        "formal_preflight_by_offloader": {
            offloader: child.get("formal_preflight") for offloader, _, child, _ in child_results
        },
        "run_root": str(combined_root.resolve()),
        "launches": [],
        "aggregate": {},
        "errors": [],
        "offloader_suites": {},
    }
    statuses: list[str] = []
    for offloader, output, child, child_failure in child_results:
        status = str(child.get("status", "failed"))
        statuses.append(status)
        result["launches"].extend(child.get("launches", ()))
        result["aggregate"].update(child.get("aggregate", {}))
        result["offloader_suites"][offloader] = {
            "status": status,
            "qualification": bool(child.get("qualification")),
            "output": str(output.resolve()),
            "run_root": child.get("run_root"),
            "launch_count": len(child.get("launches", ())),
        }
        for error in child.get("errors", ()):
            result["errors"].append({"offloader": offloader, **dict(error)})
        if child_failure is None and status not in {"success", "blocked"} and first_failure is None:
            first_failure = RuntimeError(f"{offloader} sub-suite reported {status}")

    all_children_success = bool(statuses) and all(status == "success" for status in statuses)
    result["qualification"] = (
        args.protocol == "formal"
        and all_children_success
        and all(child.get("qualification") for _, _, child, _ in child_results)
    )
    if all_children_success:
        result["status"] = "success"
    elif statuses and all(status == "blocked" for status in statuses):
        result["status"] = "blocked"
    else:
        result["status"] = "failed"
    return result, first_failure


def _run_suite(args: argparse.Namespace) -> tuple[dict[str, Any], BaseException | None]:
    if len(args.offloaders) > 1:
        return _run_multi_offloader_suite(args)

    repository = _repository_root(Path(__file__).resolve().parent)
    provenance = _git_provenance(repository)
    runtime = (
        {"lightweight": True, "numpy": importlib.metadata.version("numpy")}
        if args.protocol == "lightweight"
        else _runtime_versions(args.python)
    )
    hardware = [] if args.protocol == "lightweight" else _gpu_inventory()
    selected_gpus = [] if args.protocol == "lightweight" else _selected_gpus(hardware, args.cuda_visible_devices)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    environment_archive = _archived_environment(environment)
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "protocol": args.protocol,
        "qualification": False,
        "suite_argv": list(sys.argv),
        "candidate": {
            "git": provenance,
            "runtime": runtime,
            "hardware_inventory": hardware,
            "selected_gpus": selected_gpus,
        },
        "environment": environment_archive,
        "configuration": {
            "model": str(args.model.resolve()),
            "offloaders": list(args.offloaders),
            "repetitions": args.repetitions,
            "warmup_requests": args.warmup_requests,
            "measured_requests": args.measured_requests,
            "settle_s": args.settle_s,
            "idle_window_s": args.idle_window_s,
            "sample_interval_s": args.sample_interval_s,
            "pinned_slot_budget_assertion_bytes": args.pinned_slot_budget_bytes,
            "rtol": args.rtol,
            "atol": args.atol,
        },
        "formal_preflight": None,
        "launches": [],
        "aggregate": {},
        "errors": [],
    }
    failure: BaseException | None = None
    if args.protocol == "formal":
        preflight = _formal_preflight(args, provenance, runtime, selected_gpus)
        result["formal_preflight"] = preflight
        if not preflight["passed"]:
            failure = QualificationBlockedError("; ".join(preflight["failures"]))
            result["status"] = "blocked"
            result["errors"].append({"type": type(failure).__name__, "message": str(failure)})
            return result, failure

    run_root = args.work_root / f"hwr-qualification-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex}"
    run_root.mkdir(parents=True)
    result["run_root"] = str(run_root.resolve())
    completed_runs: list[dict[str, Any]] = []
    common_argv = _common_child_argv(args)
    diagnostic_cgroup = _current_memory_cgroup_path() if args.protocol == "diagnostic" else None
    result["diagnostic_cgroup"] = None if diagnostic_cgroup is None else str(diagnostic_cgroup)

    try:
        for offloader in args.offloaders:
            for repetition in range(args.repetitions):
                roots = run_root / offloader / f"repetition-{repetition:02d}"
                l4_root = roots / "l4-unused-empty"
                candidate_root = roots / "c4-w4-artifact"
                l4_root.mkdir(parents=True)
                candidate_root.mkdir(parents=True)
                if any(l4_root.iterdir()) or any(candidate_root.iterdir()):
                    raise RuntimeError("new scenario namespace is not empty")
                generation_id = uuid.uuid4().hex
                per_scenario: dict[str, dict[str, Any]] = {}
                c4_fingerprint: dict[str, Any] | None = None
                for scenario in scenario_order(repetition):
                    hwr_root = l4_root if scenario == "L4" else candidate_root
                    launch_id = f"{offloader}-r{repetition}-{scenario}-{uuid.uuid4().hex}"
                    launch_dir = roots / scenario.lower()
                    launch_dir.mkdir(parents=True)
                    output_path = launch_dir / "result.json"
                    tensor_dir = launch_dir / "tensors"
                    cache_path = launch_dir / "cache-attestation.json"

                    if scenario == "C4" and any(candidate_root.iterdir()):
                        raise RuntimeError("C4 namespace is not uniquely empty")
                    if scenario == "W4":
                        current = _artifact_fingerprint(candidate_root)
                        if c4_fingerprint is None or current["tree_sha256"] != c4_fingerprint["tree_sha256"]:
                            raise RuntimeError("W4 did not start from the exact C4 generation")

                    idle_report: dict[str, Any] | None = None
                    if args.protocol != "lightweight":
                        if args.allow_nonexclusive_gpus:
                            processes = _selected_compute_processes(selected_gpus)
                            idle_report = {
                                "passed": not processes,
                                "bypassed_for_diagnostic": bool(processes),
                                "remaining_compute_processes": processes,
                            }
                        else:
                            idle_report = _wait_for_idle_devices(
                                selected_gpus,
                                timeout_s=args.device_idle_timeout_s,
                                settle_s=args.device_idle_settle_s,
                            )
                            if not idle_report["passed"]:
                                raise RuntimeError(f"selected GPUs did not become idle: {idle_report}")

                    target_roots = [args.model]
                    if scenario == "W4":
                        target_roots.append(candidate_root / "artifacts")
                    files = _regular_files(target_roots)
                    cache = _cache_attestation(
                        files=files,
                        launch_id=launch_id,
                        scenario=scenario,
                        protocol=args.protocol,
                        allow_global_drop_caches=args.allow_global_drop_caches,
                        diagnostic_file_eviction=args.diagnostic_file_cache_eviction,
                    )
                    _atomic_json(cache_path, cache)
                    if args.protocol == "formal" and not cache["formal_control_verified"]:
                        raise QualificationBlockedError(f"formal cache control failed: {cache}")

                    if args.protocol == "lightweight":
                        if scenario == "C4":
                            artifact = candidate_root / "artifacts" / "synthetic" / "generation.bin"
                            artifact.parent.mkdir(parents=True)
                            artifact.write_bytes(b"synthetic-generation")
                        run = _synthetic_run(
                            args=args,
                            offloader=offloader,
                            scenario=scenario,
                            repetition=repetition,
                            launch_id=launch_id,
                            launch_dir=launch_dir,
                            generation_id=generation_id,
                        )
                        invocation = {"synthetic": True, "candidate_argv": []}
                        cgroup_record = None
                    else:
                        cgroup = None
                        if args.protocol == "formal":
                            assert args.cgroup_parent is not None
                            cgroup = _CgroupLease.create(
                                args.cgroup_parent,
                                f"hwr-{os.getpid()}-{uuid.uuid4().hex}",
                            )
                        argv = [
                            *common_argv,
                            "--offloader",
                            offloader,
                            "--scenario",
                            scenario,
                            "--launch-id",
                            launch_id,
                            "--hwr-mode",
                            SCENARIO_MODES[scenario],
                            "--hwr-root",
                            str(hwr_root),
                            "--output",
                            str(output_path),
                            "--tensor-output-dir",
                            str(tensor_dir),
                            "--cache-attestation",
                            str(cache_path),
                        ]
                        if cgroup is not None:
                            argv.extend(("--measurement-cgroup", str(cgroup.path), "--formal-runtime-contract"))
                        elif diagnostic_cgroup is not None:
                            argv.extend(("--measurement-cgroup", str(diagnostic_cgroup)))
                        if scenario == "W4":
                            argv.append("--poison-warm-build")
                        if args.allow_nonexclusive_gpus:
                            argv.append("--allow-nonexclusive-gpus")
                        try:
                            invocation = _launch_child(
                                args=args,
                                argv=argv,
                                environment=environment,
                                launch_dir=launch_dir,
                                cgroup=cgroup,
                            )
                        finally:
                            cgroup_record = cgroup.close() if cgroup is not None else None
                        if cgroup_record is not None and (
                            cgroup_record.get("remaining_pids") or cgroup_record.get("cleanup") != "removed"
                        ):
                            raise RuntimeError(f"measurement cgroup cleanup failed: {cgroup_record}")
                        if invocation["returncode"] != 0 or not output_path.is_file():
                            raise RuntimeError(f"{launch_id} failed: {invocation}")
                        run = json.loads(output_path.read_text())
                        if run.get("status") != "success":
                            raise RuntimeError(f"{launch_id} reported {run.get('status')}: {run.get('errors')}")
                        if args.protocol == "formal" and not run.get("qualification_preconditions", {}).get(
                            "formal_qualification"
                        ):
                            raise QualificationBlockedError(
                                f"{launch_id} lacks formal evidence: "
                                f"{run.get('qualification_preconditions', {}).get('formal_qualification_missing')}"
                            )

                    run.setdefault("suite", {})
                    run["suite"].update(
                        {
                            "repetition": repetition,
                            "order": list(scenario_order(repetition)),
                            "invocation": invocation,
                            "cache_attestation": cache,
                            "cgroup": cgroup_record,
                            "device_idle_preflight": idle_report,
                        }
                    )
                    if scenario == "C4":
                        summary = run.get("host_weight_runtime_evidence", {}).get("summary", {})
                        if args.protocol != "lightweight" and summary.get("observed_artifact_state") != "cold_build":
                            raise RuntimeError("C4 was not an observed cold_build")
                        c4_fingerprint = _artifact_fingerprint(candidate_root)
                        if c4_fingerprint["file_count"] == 0:
                            raise RuntimeError("C4 produced no published artifact generation")
                        run["suite"]["artifact_fingerprint"] = c4_fingerprint
                    elif scenario == "W4":
                        after = _artifact_fingerprint(candidate_root)
                        assert c4_fingerprint is not None
                        run["suite"]["artifact_fingerprint"] = after
                        run["suite"]["exact_c4_generation_retained"] = (
                            after["tree_sha256"] == c4_fingerprint["tree_sha256"]
                        )
                        if not run["suite"]["exact_c4_generation_retained"]:
                            raise RuntimeError("W4 mutated the C4 generation")
                        c4_generations = {
                            record.get("generation_id")
                            for record in per_scenario["C4"]
                            .get("host_weight_runtime_evidence", {})
                            .get("worker_records", ())
                            if record.get("generation_id")
                        }
                        w4_generations = {
                            record.get("generation_id")
                            for record in run.get("host_weight_runtime_evidence", {}).get("worker_records", ())
                            if record.get("generation_id")
                        }
                        if c4_generations != w4_generations:
                            raise RuntimeError(
                                f"W4 generation IDs differ from C4: C4={c4_generations}, W4={w4_generations}"
                            )
                        c4_pids = {
                            int(record["rank"]["pid"])
                            for record in per_scenario["C4"]
                            .get("host_weight_runtime_evidence", {})
                            .get("worker_records", ())
                        }
                        w4_pids = {
                            int(record["rank"]["pid"])
                            for record in run.get("host_weight_runtime_evidence", {}).get("worker_records", ())
                        }
                        if c4_pids & w4_pids:
                            raise RuntimeError(f"W4 did not use fresh worker PIDs: {sorted(c4_pids & w4_pids)}")
                    else:
                        if any(l4_root.iterdir()):
                            raise RuntimeError("L4 disabled mode touched its unused HWR namespace")
                        run["suite"]["artifact_fingerprint"] = {
                            "file_count": 0,
                            "storage_span_bytes": 0,
                            "tree_sha256": hashlib.sha256(b"[]").hexdigest(),
                            "files": [],
                        }

                    per_scenario[scenario] = run
                    completed_runs.append(run)
                    result["launches"].append(run)
                    _atomic_json(args.output, result)

                baseline = per_scenario["L4"]
                configuration_check = _paired_configuration_check(per_scenario)
                for run in per_scenario.values():
                    run["suite"]["paired_configuration_check"] = configuration_check
                if not configuration_check["passed"]:
                    raise RuntimeError(
                        f"{offloader} repetition {repetition} scenarios differ outside declared HWR controls"
                    )
                for scenario in ("C4", "W4"):
                    comparison = compare_outputs(
                        baseline,
                        per_scenario[scenario],
                        rtol=args.rtol,
                        atol=args.atol,
                    )
                    per_scenario[scenario]["suite"]["output_comparison"] = comparison
                    if not comparison["passed"]:
                        raise RuntimeError(f"{offloader} repetition {repetition} {scenario} output parity failed")

        result["aggregate"] = aggregate_results(
            completed_runs,
            pinned_budget_bytes=args.pinned_slot_budget_bytes,
        )
        all_launches_success = all(run.get("status") == "success" for run in completed_runs)
        all_aggregates_pass = bool(result["aggregate"]) and all(
            value.get("passed") for value in result["aggregate"].values()
        )
        result["qualification"] = (
            args.protocol == "formal"
            and all_launches_success
            and all_aggregates_pass
            and len(completed_runs) == len(args.offloaders) * args.repetitions * 3
        )
        if args.protocol == "formal" and not result["qualification"]:
            failure = RuntimeError("formal qualification acceptance gates did not all pass")
            result["errors"].append({"type": type(failure).__name__, "message": str(failure)})
            result["status"] = "failed"
        else:
            result["status"] = "success" if all_launches_success else "failed"
    except BaseException as exc:
        failure = exc
        result["status"] = "blocked" if isinstance(exc, QualificationBlockedError) else "failed"
        result["qualification"] = False
        result["errors"].append({"type": type(exc).__name__, "message": str(exc)})
        if completed_runs:
            result["aggregate"] = aggregate_results(
                completed_runs,
                pinned_budget_bytes=args.pinned_slot_budget_bytes,
            )
    return result, failure


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result, failure = _run_suite(args)
    _atomic_json(args.output, result)
    summary = {
        "output": str(args.output.resolve()),
        "status": result["status"],
        "qualification": result["qualification"],
        "launch_count": len(result.get("launches", ())),
    }
    print(f"HWR_QUALIFICATION_RESULT {json.dumps(summary, sort_keys=True)}", flush=True)
    if failure is not None:
        raise failure


if __name__ == "__main__":
    main()
