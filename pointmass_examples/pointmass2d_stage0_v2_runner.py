#!/usr/bin/env python3
"""Guarded, isolated runner for the frozen point-mass Stage-0 v2 contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import shlex
import signal
import subprocess
import sys
import time

import numpy as np
import scipy


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "pointmass_src")]

from gato_pointmass.pointmass2d import (  # noqa: E402
    DEVELOPMENT_TASK_SEEDS as V1_DEVELOPMENT_TASK_SEEDS,
    HELDOUT_TASK_SEEDS as V1_HELDOUT_TASK_SEEDS,
    PROTOCOL,
    generate_task,
    solver_task_identity_hash,
)
from pointmass_examples.pointmass2d_stage0_oracle import (  # noqa: E402
    FIRST_PAIR_WATCHDOG_SECONDS,
    FULL_RUN_WATCHDOG_SECONDS,
    OracleRuntimeWatchdogStop,
    Stage0Problem,
    _atomic_write_json,
    _atomic_write_npz,
    _first_pair_alarm_handler,
    execute_ordered_pairs,
    partial_generation_paths,
    partial_latest_path,
    sha256_file,
    sha256_json,
    solve_restart,
    watchdog_assessment,
    write_partial_checkpoint,
    write_runtime_rejection,
)
from pointmass_examples.pointmass2d_stage0_v2 import (  # noqa: E402
    STAGE0_V2_DEVELOPMENT_TASK_SEEDS,
    STAGE0_V2_HELDOUT_TASK_SEEDS,
    STAGE0_V2_VERSION,
    stage0_v2_campaign_aggregate,
    stage0_v2_expected_identities,
    stage0_v2_task_aggregate,
)


V1_PILOT_TASK_SEEDS = V1_DEVELOPMENT_TASK_SEEDS + V1_HELDOUT_TASK_SEEDS
V2_TASK_SEEDS = STAGE0_V2_DEVELOPMENT_TASK_SEEDS + STAGE0_V2_HELDOUT_TASK_SEEDS


def _source_paths():
    return {
        "task_schema": REPO_ROOT / "pointmass_src/gato_pointmass/pointmass2d.py",
        "v1_solver": REPO_ROOT / "pointmass_examples/pointmass2d_stage0_oracle.py",
        "v2_contract": REPO_ROOT / "pointmass_examples/pointmass2d_stage0_v2.py",
        "v2_runner": Path(__file__).resolve(),
        "v1_tests": REPO_ROOT / "tests/python/test_pointmass2d_stage0.py",
        "v2_contract_tests": REPO_ROOT / "tests/python/test_pointmass2d_stage0_v2.py",
        "v2_runner_tests": REPO_ROOT / "tests/python/test_pointmass2d_stage0_v2_runner.py",
    }


def _git_status():
    return subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()


def _git_head():
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _tracked_dirty(status):
    return any(not line.startswith("??") for line in status)


def _retain_row_arrays(arrays, identity, row):
    prefix = identity["key"]
    arrays[f"{prefix}_acquisition_initial_z"] = row["acquisition_initial_z"]
    arrays[f"{prefix}_unanchored_polish_initial_z"] = row[
        "unanchored_polish_initial_z"
    ]
    for phase in ("acquisition", "unanchored_polish"):
        arrays[f"{prefix}_{phase}_z"] = row[phase]["z"]
        arrays[f"{prefix}_{phase}_controls"] = row[phase]["controls"]
        arrays[f"{prefix}_{phase}_dense_states"] = row[phase]["dense_states"]
        for name, value in row[phase]["independent_kkt"].items():
            if isinstance(value, np.ndarray):
                arrays[f"{prefix}_{phase}_kkt_{name}"] = value


def run_stage0_v2(output, dependency_overrides=None):
    """Run exactly one v2 oracle campaign; tests may inject inert fake dependencies."""
    dependency_overrides_for_test_only = bool(dependency_overrides)
    dependency_overrides = dependency_overrides or {}
    task_factory = dependency_overrides.get("task_factory", generate_task)
    problem_factory = dependency_overrides.get("problem_factory", Stage0Problem)
    solve_function = dependency_overrides.get("solve_function", solve_restart)

    output = Path(output)
    final_npz = output.with_suffix(".npz")
    final_manifest = output.with_suffix(".sha256.json")
    runtime_rejection = output.with_name(f"{output.stem}.runtime_rejected.json")
    latest_checkpoint = partial_latest_path(output)
    existing_checkpoint_files = list(
        output.parent.glob(f"{output.stem}.partial.*")
    )
    forbidden_existing = [
        path
        for path in (output, final_npz, final_manifest, runtime_rejection)
        if path.exists()
    ] + existing_checkpoint_files
    if forbidden_existing:
        raise RuntimeError(
            "Stage0-v2 is single-run only; refusing existing artifacts: "
            + ", ".join(str(path) for path in forbidden_existing)
        )

    source_paths = _source_paths()
    source_hashes_at_start = {
        name: sha256_file(path) for name, path in source_paths.items()
    }
    status_at_start = _git_status()
    head_at_start = _git_head()
    if not dependency_overrides_for_test_only and _tracked_dirty(status_at_start):
        raise RuntimeError(
            "Stage0-v2 default execution requires a tracked-clean provenance commit"
        )
    if set(V1_PILOT_TASK_SEEDS) & set(V2_TASK_SEEDS):
        raise RuntimeError("v1 pilot and v2 fresh task identities must be disjoint")

    # This guarded body is the only v2 path allowed to instantiate fresh tasks.
    tasks = {task_seed: task_factory(task_seed) for task_seed in V2_TASK_SEEDS}
    expected_identities = stage0_v2_expected_identities()
    if len(expected_identities) != 80:
        raise RuntimeError("Stage0-v2 requires exactly 80 ordered restart pairs")
    task_identity_hashes = {
        str(seed): tasks[seed]["solver_task_identity_sha256"] for seed in V2_TASK_SEEDS
    }
    identity_provenance = {
        "stage0_version": STAGE0_V2_VERSION,
        "git_head_at_start": head_at_start,
        "git_status_porcelain_full_at_start": status_at_start,
        "git_tracked_clean_at_start": not _tracked_dirty(status_at_start),
        "protocol_sha256": sha256_json(PROTOCOL.__dict__),
        "task_identity_sha256_by_seed": task_identity_hashes,
        "source_hashes_at_start": source_hashes_at_start,
        "exact_command_argv": list(sys.argv),
        "exact_command_shell": shlex.join(sys.argv),
        "v1_pilot_exclusion": {
            "excluded_task_seeds": V1_PILOT_TASK_SEEDS,
            "v1_task_seed_intersection": [],
            "v1_rows_or_aggregates_consumed": 0,
            "v1_artifacts_consumed": 0,
            "v1_pilot_status": "formally_rejected_excluded_from_v2",
        },
        "dependency_overrides_for_test_only": dependency_overrides_for_test_only,
    }
    arrays = {}
    for task_seed, task in tasks.items():
        arrays[f"task{task_seed}_solver_x0_float32"] = task["x0"]
        arrays[f"task{task_seed}_solver_reference_float32"] = task["reference"]
    problems = {}
    completed_records = []
    run_started = time.monotonic()
    first_pair_seconds = None

    def execute_pair(identity):
        nonlocal first_pair_seconds
        task_seed = identity["task_seed"]
        if task_seed not in tasks or task_seed in V1_PILOT_TASK_SEEDS:
            raise RuntimeError("restart identity is not an authorized fresh v2 task")
        if task_seed not in problems:
            problems[task_seed] = problem_factory(tasks[task_seed])
        pair_started = time.monotonic()
        first_pair = len(completed_records) == 0
        previous_handler = None
        previous_timer = None
        if first_pair:
            previous_handler = signal.signal(signal.SIGALRM, _first_pair_alarm_handler)
            previous_timer = signal.setitimer(
                signal.ITIMER_REAL, FIRST_PAIR_WATCHDOG_SECONDS
            )
        try:
            row = solve_function(
                problems[task_seed], identity["mode_side"], identity["restart_index"]
            )
        except OracleRuntimeWatchdogStop:
            elapsed = time.monotonic() - run_started
            first_pair_seconds = max(
                time.monotonic() - pair_started,
                np.nextafter(FIRST_PAIR_WATCHDOG_SECONDS, np.inf),
            )
            assessment = watchdog_assessment(first_pair_seconds, elapsed, 0)
            write_runtime_rejection(output, assessment, identity_provenance)
            raise
        finally:
            if first_pair:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                signal.signal(signal.SIGALRM, previous_handler)
                if previous_timer and previous_timer[0] > 0.0:
                    signal.setitimer(
                        signal.ITIMER_REAL, previous_timer[0], previous_timer[1]
                    )
        pair_seconds = time.monotonic() - pair_started
        if first_pair:
            first_pair_seconds = pair_seconds
        row["runtime_pair_wall_seconds"] = pair_seconds
        return row

    def retain_completed_pair(identity, row):
        if (
            row["expected_mode"] != identity["expected_mode"]
            or row["restart_index"] != identity["restart_index"]
            or row["mode_side"] != identity["mode_side"]
        ):
            raise RuntimeError("v2 solver row identity differs from frozen ledger")
        row["identity_key"] = identity["key"]
        assessment = watchdog_assessment(
            first_pair_seconds,
            time.monotonic() - run_started,
            len(completed_records) + 1,
        )
        row["runtime_watchdog_assessment_after_pair"] = assessment
        _retain_row_arrays(arrays, identity, row)
        completed_records.append(
            {"identity": identity, "task": tasks[identity["task_seed"]], "row": row}
        )
        current_provenance = dict(identity_provenance)
        current_provenance["source_hashes_current"] = {
            name: sha256_file(path) for name, path in source_paths.items()
        }
        write_partial_checkpoint(
            output,
            expected_identities,
            completed_records,
            arrays,
            current_provenance,
        )
        if assessment["stop"]:
            write_runtime_rejection(output, assessment, current_provenance)
            raise OracleRuntimeWatchdogStop(
                "Stage0-v2 runtime watchdog stopped the single authorized run"
            )

    execute_ordered_pairs(expected_identities, execute_pair, retain_completed_pair)
    if len(completed_records) != 80:
        raise RuntimeError("Stage0-v2 final publication requires exactly 80 pairs")
    completed_identities = [record["identity"] for record in completed_records]
    if completed_identities != expected_identities:
        raise RuntimeError("Stage0-v2 completed ledger differs from expected order")

    latest = json.loads(latest_checkpoint.read_text())
    if latest["latest_complete_generation"] != 80:
        raise RuntimeError("latest v2 checkpoint must reference generation 80")
    generation_json, generation_npz = partial_generation_paths(output, 80)
    if (
        latest["generation_json"] != str(generation_json)
        or latest["generation_npz"] != str(generation_npz)
        or sha256_file(generation_json) != latest["generation_json_sha256"]
        or sha256_file(generation_npz) != latest["generation_npz_sha256"]
    ):
        raise RuntimeError("latest v2 checkpoint pointer/hash mismatch")

    task_rows = []
    for task_seed in V2_TASK_SEEDS:
        rows = [
            record["row"]
            for record in completed_records
            if record["identity"]["task_seed"] == task_seed
        ]
        task = tasks[task_seed]
        task_rows.append(
            {
                "task_seed": task_seed,
                "task": task,
                "rows": rows,
                "aggregate": stage0_v2_task_aggregate(task, rows),
            }
        )
    campaign = stage0_v2_campaign_aggregate(task_rows)
    source_hashes_at_end = {
        name: sha256_file(path) for name, path in source_paths.items()
    }
    status_at_end = _git_status()
    head_at_end = _git_head()
    source_stable = source_hashes_at_start == source_hashes_at_end
    head_stable = head_at_start == head_at_end
    exact_task_identities = bool(
        len(set(task_identity_hashes.values())) == 8
        and tuple(tasks) == V2_TASK_SEEDS
        and all(
            tasks[seed]["canonical_numeric_source"]
            == "exact_solver_bound_float32_bytes"
            and tasks[seed]["solver_task_identity_sha256"]
            == solver_task_identity_hash(tasks[seed]["x0"], tasks[seed]["reference"])
            for seed in V2_TASK_SEEDS
        )
    )
    tracked_clean = not _tracked_dirty(status_at_start) and not _tracked_dirty(
        status_at_end
    )
    all_runner_gates = bool(
        campaign["all_stage0_v2_gates_pass"]
        and len(completed_records) == 80
        and completed_identities == expected_identities
        and exact_task_identities
        and source_stable
        and head_stable
        and tracked_clean
        and not dependency_overrides_for_test_only
        and not (set(V1_PILOT_TASK_SEEDS) & set(V2_TASK_SEEDS))
    )
    provenance = {
        **identity_provenance,
        "git_head_at_end": head_at_end,
        "git_head_stable_during_execution": head_stable,
        "git_status_porcelain_full_at_end": status_at_end,
        "git_tracked_clean_at_end": not _tracked_dirty(status_at_end),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "source_hashes_at_end": source_hashes_at_end,
        "source_hashes_stable": source_stable,
    }
    summary = {
        "schema_version": 1,
        "diagnostic": "pointmass2d_stage0_v2_quarantined_oracle",
        "stage0_version": STAGE0_V2_VERSION,
        "incomplete": False,
        "oracle_only": True,
        "benchmark_seed_eligible": False,
        "gato_sqp_cuda_or_build_calls": 0,
        "v1_pilot_excluded": True,
        "v1_rows_or_aggregates_consumed": 0,
        "v1_artifacts_consumed": 0,
        "expected_restart_count": 80,
        "expected_ordered_identities": expected_identities,
        "completed_restart_count": len(completed_records),
        "completed_ordered_identities": completed_identities,
        "pending_ordered_identities": [],
        "task_rows": task_rows,
        "campaign_aggregate": campaign,
        "exact_eight_canonical_v2_task_identities": exact_task_identities,
        "source_hashes_stable_during_execution": source_stable,
        "git_head_stable_during_execution": head_stable,
        "tracked_provenance_clean": tracked_clean,
        "all_stage0_v2_runner_gates_pass": all_runner_gates,
        "provenance": provenance,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_npz(final_npz, arrays)
    summary["artifacts"] = {
        "npz": str(final_npz),
        "npz_sha256": sha256_file(final_npz),
        "supersedes_retained_latest_checkpoint_pointer": str(latest_checkpoint),
        "supersedes_retained_checkpoint_generation": 80,
        "supersedes_retained_checkpoint_json": str(generation_json),
        "supersedes_retained_checkpoint_npz": str(generation_npz),
    }
    _atomic_write_json(output, summary)
    _atomic_write_json(
        final_manifest,
        {
            "json": str(output),
            "json_sha256": sha256_file(output),
            "npz": str(final_npz),
            "npz_sha256": sha256_file(final_npz),
            "stage0_version": STAGE0_V2_VERSION,
            "incomplete": False,
            "completed_restart_count": 80,
            "source_hashes_stable": source_stable,
            "supersedes_retained_checkpoint_generation": 80,
            **provenance,
        },
    )
    print(output)
    print(json.dumps({"all_stage0_v2_runner_gates_pass": all_runner_gates}, indent=2))
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-stage0-v2-oracle", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.execute_stage0_v2_oracle:
        parser.error("execution is blocked without --execute-stage0-v2-oracle")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    run_stage0_v2(arguments.output)
