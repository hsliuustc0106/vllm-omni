# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import mmap
import multiprocessing
import os
import sys
import time
from argparse import Namespace
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

pytestmark = pytest.mark.core_model

_REPOSITORY = Path(__file__).resolve().parents[3]


def _load_example(name: str) -> ModuleType:
    path = _REPOSITORY / "examples" / "offline_inference" / "minimax_h3" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def evaluator() -> ModuleType:
    return _load_example("host_weight_runtime_eval")


@pytest.fixture(scope="module")
def qualifier() -> ModuleType:
    return _load_example("host_weight_runtime_qualify")


def test_scenario_order_alternates_whole_cold_warm_block(qualifier: ModuleType) -> None:
    assert qualifier.scenario_order(0) == ("L4", "C4", "W4")
    assert qualifier.scenario_order(1) == ("C4", "W4", "L4")
    assert qualifier.scenario_order(2) == ("L4", "C4", "W4")


@pytest.mark.parametrize("value", ["1", "3.99", "15.01", "nan"])
def test_minimax_duration_is_rejected_before_gpu_launch(
    evaluator: ModuleType,
    qualifier: ModuleType,
    value: str,
) -> None:
    with pytest.raises(evaluator.argparse.ArgumentTypeError, match=r"\[4, 15\]"):
        evaluator._minimax_duration(value)
    with pytest.raises(qualifier.argparse.ArgumentTypeError, match=r"\[4, 15\]"):
        qualifier._minimax_duration(value)


@pytest.mark.parametrize("value", ["0", "1"])
def test_minimax_step_count_is_rejected_before_gpu_launch(
    evaluator: ModuleType,
    qualifier: ModuleType,
    value: str,
) -> None:
    with pytest.raises(evaluator.argparse.ArgumentTypeError, match="at least 2"):
        evaluator._minimax_steps(value)
    with pytest.raises(qualifier.argparse.ArgumentTypeError, match="at least 2"):
        qualifier._minimax_steps(value)


def test_formal_preflight_fails_closed_without_host_controls(
    qualifier: ModuleType,
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    args = qualifier.parse_args(
        [
            "--model",
            str(model),
            "--work-root",
            str(tmp_path / "work"),
            "--output",
            str(tmp_path / "out.json"),
            "--protocol",
            "formal",
        ]
    )

    report = qualifier._formal_preflight(
        args,
        {"clean": False},
        {"vllm": "wrong", "torch": "wrong", "torch_cuda": None, "diffusers": "wrong"},
        [],
    )

    assert report["passed"] is False
    assert any("cgroup" in failure for failure in report["failures"])
    assert "--allow-global-drop-caches is required" in report["failures"]
    assert "candidate worktree is dirty" in report["failures"]
    assert not any("pinned-slot-budget" in failure for failure in report["failures"])


def test_file_scoped_eviction_is_never_formal_control(
    qualifier: ModuleType,
    tmp_path: Path,
) -> None:
    target = tmp_path / "artifact.bin"
    target.write_bytes(b"x" * 8192)

    report = qualifier._cache_attestation(
        files=[target],
        launch_id="launch",
        scenario="W4",
        protocol="diagnostic",
        allow_global_drop_caches=False,
        diagnostic_file_eviction=True,
    )

    assert report["scope"] == "file_scoped_advice_only"
    assert report["formal_control_verified"] is False
    assert report["target_file_count"] == 1


def test_artifact_fingerprint_detects_generation_mutation(
    qualifier: ModuleType,
    tmp_path: Path,
) -> None:
    generation = tmp_path / "artifacts" / "key" / "generation.bin"
    generation.parent.mkdir(parents=True)
    generation.write_bytes(b"first")
    before = qualifier._artifact_fingerprint(tmp_path)

    generation.write_bytes(b"second")
    after = qualifier._artifact_fingerprint(tmp_path)

    assert before["file_count"] == 1
    assert before["tree_sha256"] != after["tree_sha256"]
    assert before["storage_span_bytes"] == 4096


def _output_record(path: Path, value: np.ndarray) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value, allow_pickle=False)
    return {
        "output_capture": {
            "status": "captured",
            "scope": "pre_encoding",
        },
        "outputs": [
            {
                "request_index": 0,
                "tensor_archives": {
                    "video": {"path": str(path)},
                },
            }
        ],
    }


def test_output_comparison_enforces_shape_dtype_finite_pattern_and_tolerance(
    qualifier: ModuleType,
    tmp_path: Path,
) -> None:
    reference = _output_record(tmp_path / "reference.npy", np.array([1.0, 2.0], dtype=np.float32))
    close = _output_record(tmp_path / "close.npy", np.array([1.0, 2.00001], dtype=np.float32))
    wrong_dtype = _output_record(tmp_path / "wrong.npy", np.array([1.0, 2.0], dtype=np.float64))

    close_report = qualifier.compare_outputs(reference, close, rtol=1e-4, atol=1e-4)
    wrong_report = qualifier.compare_outputs(reference, wrong_dtype, rtol=1e-4, atol=1e-4)

    assert close_report["passed"] is True
    assert wrong_report["passed"] is False
    assert wrong_report["comparisons"][0]["dtype_equal"] is False


def _resolution_records(*, builder_rank: int = 0) -> list[dict[str, object]]:
    records = []
    for rank in range(4):
        role = "builder" if rank == builder_rank else "waiter"
        records.append(
            {
                "rank": {"global_rank": rank, "pid": 100 + rank},
                "schema_version": 2,
                "runtime_mode": "read_write",
                "outcome": "ready",
                "claim_role": role,
                "artifact_key": "artifact",
                "artifact_compatibility_digest": "compatibility",
                "generation_id": "generation",
                "runtime_instance_id": f"runtime-{rank}",
                "capability_grant_id": f"grant-{rank}",
                "provider_id": "runtime-mmap",
                "provider_abi": "1",
                "backing_kind": "runtime_mmap",
                "access_features": ["complete_tensor_read"],
                "pinned_slot_budget_bytes": 1048576,
            }
        )
    return records


def _formal_records(*, offloader: str = "dlo-no-allgather") -> list[dict[str, object]]:
    records = _resolution_records()
    plan_kind = "component" if offloader == "model" else "blocks_plus_resident"
    unit_kinds = ["component"] if offloader == "model" else ["block", "resident"]
    artifact_key = "a" * 64
    lease_id = "generation"
    event = {
        "launch_id": "launch",
        "artifact_key": artifact_key,
        "lease_id": lease_id,
        "builder_actor_id": "dp:0",
        "monotonic_time_ns": 1,
    }
    for record in records:
        rank = record["rank"]["global_rank"]  # type: ignore[index]
        record.update(
            artifact_key=artifact_key,
            negotiated_capability_grant_id=record["capability_grant_id"],
            selected_transfer_plan_id=f"plan.{plan_kind}",
            selected_transfer_plan_kind=plan_kind,
            exact_coverage_digest="c" * 64,
            unit_kinds=unit_kinds,
            pre_resolve={},
            idle_state={
                "outstanding_units": 0,
                "bindings": 0,
                "resident_bindings": 2,
                "total_bindings": 2,
                "events": 0,
            },
            producer_present=rank == 0,
            builder_started=event if rank == 0 else None,
            observed_builder_started=event if rank != 0 else None,
        )
    return records


def test_resolution_summary_validates_complete_grant_evidence(evaluator: ModuleType) -> None:
    args = Namespace(dp_size=4, hwr_mode="read_write")
    summary = evaluator._resolution_summary(_resolution_records(), args=args)
    assert summary["observed_artifact_state"] == "cold_build"
    assert len(summary["capability_grants"]) == 4

    incomplete = _resolution_records()
    incomplete[2]["provider_abi"] = None
    with pytest.raises(RuntimeError, match="incomplete capability-grant evidence"):
        evaluator._resolution_summary(incomplete, args=args)

    stale = _resolution_records()
    stale[0]["schema_version"] = 1
    with pytest.raises(RuntimeError, match="evidence schema differs"):
        evaluator._resolution_summary(stale, args=args)


def test_formal_runtime_contract_rejects_scheduler_race_and_missing_barriers(evaluator: ModuleType) -> None:
    args = Namespace(
        hwr_mode="read_write",
        offloader="dlo-no-allgather",
        scenario="C4",
        launch_id="launch",
    )

    report = evaluator._formal_runtime_contract(_resolution_records(builder_rank=2), args=args)

    assert report["passed"] is False
    assert "C4 builder must be DP rank 0" in report["violations"]
    assert any("PRE_RESOLVE" in item for item in report["missing"])
    assert any("unit_kinds" in item for item in report["missing"])


def test_formal_runtime_contract_uses_canonical_complete_tensor_feature(evaluator: ModuleType) -> None:
    args = Namespace(
        hwr_mode="read_write",
        offloader="model",
        scenario=None,
        launch_id="launch",
    )
    records = _resolution_records()
    for record in records:
        record.update(
            negotiated_capability_grant_id=record["capability_grant_id"],
            unit_kinds=["component"],
            pre_resolve={},
            idle_state={
                "outstanding_units": 0,
                "bindings": 0,
                "resident_bindings": 0,
                "total_bindings": 0,
                "events": 0,
            },
        )

    report = evaluator._formal_runtime_contract(records, args=args)
    assert not any("complete_tensor_read" in violation for violation in report["violations"])

    records[0]["access_features"] = ["complete_unit_read"]
    report = evaluator._formal_runtime_contract(records, args=args)
    assert "rank 0 grant lacks complete_tensor_read" in report["violations"]


@pytest.mark.parametrize(
    ("field", "replacement", "expected_fragment"),
    [
        ("selected_transfer_plan_id", None, "selected_transfer_plan_id"),
        (
            "selected_transfer_plan_kind",
            "component",
            "selected transfer plan kind differs",
        ),
        ("exact_coverage_digest", None, "exact_coverage_digest"),
    ],
)
def test_formal_runtime_contract_requires_exact_selected_plan_evidence(
    evaluator: ModuleType,
    field: str,
    replacement: object,
    expected_fragment: str,
) -> None:
    args = Namespace(
        hwr_mode="read_write",
        offloader="dlo-no-allgather",
        scenario="C4",
        launch_id="launch",
    )
    records = _formal_records()
    records[1][field] = replacement

    report = evaluator._formal_runtime_contract(records, args=args)

    assert report["passed"] is False
    assert any(expected_fragment in item for item in [*report["missing"], *report["violations"]])


@pytest.mark.parametrize(
    ("field", "replacement", "expected_fragment"),
    [
        (
            "selected_transfer_plan_id",
            "plan.other",
            "different transfer plan IDs",
        ),
        (
            "exact_coverage_digest",
            "d" * 64,
            "different exact transfer coverage",
        ),
    ],
)
def test_formal_runtime_contract_requires_cross_rank_plan_agreement(
    evaluator: ModuleType,
    field: str,
    replacement: object,
    expected_fragment: str,
) -> None:
    args = Namespace(
        hwr_mode="read_write",
        offloader="dlo-no-allgather",
        scenario="C4",
        launch_id="launch",
    )
    records = _formal_records()
    records[2][field] = replacement

    report = evaluator._formal_runtime_contract(records, args=args)

    assert report["passed"] is False
    assert any(expected_fragment in item for item in report["violations"])


def test_formal_runtime_contract_handles_zero_builders_without_indexing(
    evaluator: ModuleType,
) -> None:
    args = Namespace(
        hwr_mode="read_write",
        offloader="dlo-no-allgather",
        scenario="C4",
        launch_id="launch",
    )
    records = _formal_records()
    for record in records:
        record["claim_role"] = "waiter"
        record["builder_started"] = None

    report = evaluator._formal_runtime_contract(records, args=args)

    assert report["passed"] is False
    assert "C4 requires exactly one builder, observed 0" in report["violations"]


@pytest.mark.parametrize(
    "field",
    ["launch_id", "artifact_key", "lease_id", "builder_actor_id"],
)
def test_formal_runtime_contract_matches_full_builder_event_identity(
    evaluator: ModuleType,
    field: str,
) -> None:
    args = Namespace(
        hwr_mode="read_write",
        offloader="dlo-no-allgather",
        scenario="C4",
        launch_id="launch",
    )
    records = _formal_records()
    observation = dict(records[3]["observed_builder_started"])  # type: ignore[arg-type]
    observation[field] = "different"
    records[3]["observed_builder_started"] = observation

    report = evaluator._formal_runtime_contract(records, args=args)

    assert report["passed"] is False
    assert any(f"BuilderStarted.{field}" in violation for violation in report["violations"])


def test_formal_runtime_contract_rejects_stale_evidence_schema(
    evaluator: ModuleType,
) -> None:
    args = Namespace(
        hwr_mode="read_write",
        offloader="dlo-no-allgather",
        scenario="C4",
        launch_id="launch",
    )
    records = _formal_records()
    records[2]["schema_version"] = 1

    report = evaluator._formal_runtime_contract(records, args=args)

    assert report["passed"] is False
    assert any("schema_version" in violation for violation in report["violations"])


def test_process_snapshot_reports_mapped_locked_private_and_fault_metrics(
    evaluator: ModuleType,
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "mapped.bin"
    artifact.write_bytes(b"x" * 8192)
    with artifact.open("rb") as source, mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ):
        snapshot = evaluator._process_snapshot(
            pids=[os.getpid()],
            artifact_roots=(tmp_path,),
        )

    metrics = snapshot["per_process"][str(os.getpid())]
    assert metrics["mapped_artifact_kib"] >= 8
    assert "private_clean_kib" in metrics
    assert "private_dirty_kib" in metrics
    assert "locked_kib" in metrics
    assert "minor_faults" in metrics
    assert snapshot["totals"]["pss_kib"] > 0


def test_only_exact_parent_scoped_resource_tracker_is_excluded_from_workers(
    evaluator: ModuleType,
) -> None:
    tracker = {
        "pid": 123,
        "ppid": os.getpid(),
        "comm": "python",
        "start_time_ticks": 456,
        "cmdline": [
            "/venv/bin/python",
            "-c",
            "from multiprocessing.resource_tracker import main;main(17)",
        ],
    }

    assert evaluator._is_parent_scoped_resource_tracker(tracker) is True
    assert evaluator._is_parent_scoped_resource_tracker({**tracker, "ppid": os.getpid() + 1}) is False
    assert (
        evaluator._is_parent_scoped_resource_tracker({**tracker, "cmdline": ["/venv/bin/python", "worker.py"]}) is False
    )
    assert (
        evaluator._is_parent_scoped_resource_tracker(
            {
                **tracker,
                "cmdline": [
                    "/venv/bin/python",
                    "from multiprocessing.resource_tracker import main;main(17)",
                ],
            }
        )
        is False
    )
    assert (
        evaluator._is_parent_scoped_resource_tracker(
            {
                **tracker,
                "cmdline": [
                    "/venv/bin/python",
                    "-c",
                    "from multiprocessing.resource_tracker import main;main(17)",
                    "extra",
                ],
            }
        )
        is False
    )
    assert (
        evaluator._is_parent_scoped_resource_tracker(
            {
                **tracker,
                "cmdline": [
                    "/venv/bin/python",
                    "-c",
                    "from multiprocessing.resource_tracker import main;main(evil)",
                ],
            }
        )
        is False
    )


def test_process_exit_wait_allows_delayed_exit_but_remains_fail_closed(
    evaluator: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = {"pid": 123, "start_time_ticks": 456}
    alive = iter((True, True, False))
    monotonic = iter((0.0, 0.1, 0.2))
    sleeps: list[float] = []
    monkeypatch.setattr(evaluator, "_same_process_is_alive", lambda _identity: next(alive))
    monkeypatch.setattr(evaluator.time, "monotonic", lambda: next(monotonic))
    monkeypatch.setattr(evaluator.time, "sleep", sleeps.append)

    assert evaluator._wait_for_process_exit({"123": identity}, timeout_s=30.0) == []
    assert sleeps == [0.1, 0.1]

    monkeypatch.setattr(evaluator, "_same_process_is_alive", lambda _identity: True)
    monotonic = iter((0.0, 0.0, 0.02))
    monkeypatch.setattr(evaluator.time, "monotonic", lambda: next(monotonic))
    sleeps.clear()

    assert evaluator._wait_for_process_exit({"123": identity}, timeout_s=0.01) == [identity]
    assert sleeps == [0.1]


def test_process_exit_wait_reaps_a_delayed_direct_child(evaluator: ModuleType) -> None:
    process = multiprocessing.get_context("spawn").Process(target=time.sleep, args=(0.1,))
    process.start()
    try:
        identity = evaluator._pid_identity(process.pid)
        assert identity is not None

        assert evaluator._wait_for_process_exit({str(process.pid): identity}, timeout_s=2.0) == []
        assert process not in multiprocessing.active_children()
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=2.0)


def test_qualifier_pins_process_exit_timeout_in_child_argv(
    qualifier: ModuleType,
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    args = qualifier.parse_args(
        [
            "--model",
            str(model),
            "--work-root",
            str(tmp_path / "work"),
            "--output",
            str(tmp_path / "out.json"),
            "--process-exit-timeout-s",
            "42",
        ]
    )

    argv = qualifier._common_child_argv(args)

    assert argv[argv.index("--process-exit-timeout-s") + 1] == "42.0"


def test_host_monitor_sums_each_tick_before_taking_median(evaluator: ModuleType) -> None:
    monitor = evaluator._HostMemoryMonitor(0.1, (), None)
    monitor.samples = [
        {
            "monotonic_time_ns": 1,
            "phase": "idle_steady_state",
            "per_process": {
                "10": {"pss_kib": 100},
                "11": {"pss_kib": 200},
            },
            "cgroup": {"current_bytes": 1000, "peak_bytes": 2000},
        },
        {
            "monotonic_time_ns": 2,
            "phase": "idle_steady_state",
            "per_process": {
                "10": {"pss_kib": 300},
                "11": {"pss_kib": 400},
            },
            "cgroup": {"current_bytes": 2000, "peak_bytes": 3000},
        },
    ]

    report = monitor.worker_report({0: 10, 1: 11})

    assert report["idle_steady_state"]["pss_kib"]["values"] == [300.0, 700.0]
    assert report["idle_steady_state"]["pss_kib"]["median"] == 500.0
    assert report["cgroup_peak_bytes"] == 3000


def test_host_monitor_uses_wall_time_and_cadence_not_impossible_sample_count(
    evaluator: ModuleType,
) -> None:
    monitor = evaluator._HostMemoryMonitor(0.1, (), None)
    start_ns = 1_000_000_000
    end_ns = 21_000_000_000
    monitor._phase_transitions = [
        {"phase": "idle_steady_state", "monotonic_time_ns": start_ns},
        {"phase": "measured_requests", "monotonic_time_ns": end_ns},
    ]
    samples = [{"monotonic_time_ns": start_ns + 50_000_000 + index * 100_000_000} for index in range(200)]

    coverage = monitor._phase_sampling_coverage("idle_steady_state", samples)

    assert coverage["passed"] is True
    assert coverage["phase_duration_s"] == 20.0
    assert coverage["coverage_fraction"] >= 0.95

    clustered = samples[:20]
    coverage = monitor._phase_sampling_coverage("idle_steady_state", clustered)
    assert coverage["passed"] is False
    assert coverage["coverage_fraction"] < 0.95


def test_lightweight_suite_exercises_all_three_offloaders_without_gpu(
    qualifier: ModuleType,
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.bin").write_bytes(b"model")
    args = qualifier.parse_args(
        [
            "--model",
            str(model),
            "--work-root",
            str(tmp_path / "work"),
            "--output",
            str(tmp_path / "result.json"),
            "--protocol",
            "lightweight",
            "--repetitions",
            "1",
            "--measured-requests",
            "1",
            "--pinned-slot-budget-bytes",
            "1048576",
        ]
    )

    result, failure = qualifier._run_suite(args)

    assert failure is None
    assert result["status"] == "success"
    assert result["qualification"] is False
    assert len(result["launches"]) == 9
    assert set(result["aggregate"]) == set(qualifier.OFFLOADERS)
    for offloader in qualifier.OFFLOADERS:
        launches = [run for run in result["launches"] if run["candidate"]["offloader"] == offloader]
        assert [run["launch"]["scenario"] for run in launches] == ["L4", "C4", "W4"]
        assert launches[2]["suite"]["exact_c4_generation_retained"] is True
        assert result["aggregate"][offloader]["paired_repetitions"][0]["C4"]["output_comparison"]["passed"]
        budget = result["aggregate"][offloader]["pinned_slot_budget"]
        assert budget["passed"] is True
        assert budget["value_bytes"] == qualifier.SYNTHETIC_PINNED_SLOT_BUDGET_BYTES


def test_output_comparison_fails_closed_without_pre_encoding_capture(
    qualifier: ModuleType,
    tmp_path: Path,
) -> None:
    reference = _output_record(tmp_path / "reference.npy", np.array([1.0], dtype=np.float32))
    candidate = _output_record(tmp_path / "candidate.npy", np.array([1.0], dtype=np.float32))
    candidate["output_capture"]["scope"] = "post_encoding"

    report = qualifier.compare_outputs(reference, candidate, rtol=0.0, atol=0.0)

    assert report["passed"] is False
    assert any("not verified pre-encoding" in error for error in report["errors"])


def test_runtime_budget_is_canonical_and_cli_value_is_only_an_assertion(
    qualifier: ModuleType,
) -> None:
    runs = [
        {
            "launch": {"scenario": "C4"},
            "suite": {"repetition": 0},
            "host_weight_runtime_evidence": {
                "worker_records": [
                    {
                        "rank": {"global_rank": rank},
                        "pinned_slot_budget_bytes": 1024,
                    }
                    for rank in range(4)
                ]
            },
        }
    ]

    matching = qualifier._runtime_pinned_slot_budget(runs, asserted_budget_bytes=1024)
    loosened = qualifier._runtime_pinned_slot_budget(runs, asserted_budget_bytes=4096)
    missing = qualifier._runtime_pinned_slot_budget(
        [
            {
                "launch": {"scenario": "C4"},
                "suite": {"repetition": 0},
                "host_weight_runtime_evidence": {
                    "worker_records": [{"rank": {"global_rank": 0}}],
                },
            }
        ],
        asserted_budget_bytes=None,
    )

    assert matching["passed"] is True
    assert matching["value_bytes"] == 1024
    assert loosened["passed"] is False
    assert loosened["value_bytes"] == 1024
    assert missing["passed"] is False


def test_latency_gates_c4_and_w4_independently(
    qualifier: ModuleType,
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.bin").write_bytes(b"model")
    args = qualifier.parse_args(
        [
            "--model",
            str(model),
            "--work-root",
            str(tmp_path / "work"),
            "--output",
            str(tmp_path / "result.json"),
            "--protocol",
            "lightweight",
            "--offloaders",
            "model",
            "--repetitions",
            "1",
            "--measured-requests",
            "1",
        ]
    )
    result, failure = qualifier._run_suite(args)
    assert failure is None
    for run in result["launches"]:
        scenario = run["launch"]["scenario"]
        latency = {"L4": 1.0, "C4": 1.20, "W4": 1.0}[scenario]
        run["timings"]["request_latency_s"] = qualifier._distribution([latency])

    aggregate = qualifier.aggregate_results(result["launches"], pinned_budget_bytes=None)["model"]

    assert aggregate["gates"]["c4_latency_regression_at_most_10_percent"] is False
    assert aggregate["gates"]["w4_latency_regression_at_most_10_percent"] is True


def test_multi_offloader_suite_continues_after_first_offloader_failure(
    qualifier: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.bin").write_bytes(b"model")
    args = qualifier.parse_args(
        [
            "--model",
            str(model),
            "--work-root",
            str(tmp_path / "work"),
            "--output",
            str(tmp_path / "result.json"),
            "--protocol",
            "lightweight",
            "--repetitions",
            "1",
            "--measured-requests",
            "1",
        ]
    )
    original = qualifier._synthetic_run

    def fail_model(*, offloader: str, scenario: str, **kwargs: object) -> dict[str, object]:
        if offloader == "model" and scenario == "C4":
            raise RuntimeError("injected model failure")
        return original(offloader=offloader, scenario=scenario, **kwargs)

    monkeypatch.setattr(qualifier, "_synthetic_run", fail_model)

    result, failure = qualifier._run_suite(args)

    assert isinstance(failure, RuntimeError)
    assert result["status"] == "failed"
    assert result["offloader_suites"]["model"]["status"] == "failed"
    assert result["offloader_suites"]["layerwise"]["status"] == "success"
    assert result["offloader_suites"]["dlo-no-allgather"]["status"] == "success"
    completed_offloaders = {run["candidate"]["offloader"] for run in result["launches"]}
    assert {"layerwise", "dlo-no-allgather"}.issubset(completed_offloaders)
