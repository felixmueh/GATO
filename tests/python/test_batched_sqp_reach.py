import importlib.util
from pathlib import Path
import sys
from copy import deepcopy
import os
from types import SimpleNamespace

import numpy as np
import pinocchio as pin
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "examples/benchmark_batched_sqp_reach.py"
SPEC = importlib.util.spec_from_file_location("benchmark_batched_sqp_reach", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_instance_generation_is_reproducible_and_uses_bounded_start_jitter():
    model = pin.buildModelFromUrdf(str(MODULE.MODEL_PATH))
    first = MODULE.generate_instance(model, MODULE.LANES["near"], 40)
    repeated = MODULE.generate_instance(model, MODULE.LANES["near"], 40)
    different = MODULE.generate_instance(model, MODULE.LANES["near"], 41)

    for left, right in zip(first, repeated):
        np.testing.assert_array_equal(left, right)
    assert not np.array_equal(first[0], different[0])
    assert np.max(np.abs(first[0])) <= 0.10
    np.testing.assert_allclose(
        first[2] - first[1], MODULE.LANES["near"].goal_offset, atol=1e-7
    )


def test_seed_batches_are_nested_cheap_bounded_and_keep_common_x0():
    model = pin.buildModelFromUrdf(str(MODULE.MODEL_PATH))
    q0, _, _ = MODULE.generate_instance(model, MODULE.LANES["near"], 40)
    seeds4, metadata4 = MODULE.generate_seeds(model, q0, method_seed=1000, budget=4)
    seeds8, metadata8 = MODULE.generate_seeds(model, q0, method_seed=1000, budget=8)

    np.testing.assert_array_equal(seeds4, seeds8[:4])
    assert metadata4["state_amplitude_rad"] == 0.020
    assert metadata8["control_effort_fraction"] == 0.010
    for candidate, seed in enumerate(seeds8):
        q, qd, controls = MODULE.unpack(seed, model)
        np.testing.assert_array_equal(q[0], q0)
        np.testing.assert_array_equal(qd, np.zeros_like(qd))
        assert np.max(np.abs(q - q0)) <= 0.020 + 1e-7
        assert np.max(np.abs(controls) / model.effortLimit) <= 0.010 + 1e-7
        np.testing.assert_array_equal(q[-1], q0)
        if candidate == 0:
            np.testing.assert_array_equal(q, np.tile(q0, (MODULE.KNOTS, 1)))
            np.testing.assert_array_equal(controls, np.zeros_like(controls))
        else:
            assert np.any(q[1:-1] != q0)
            assert np.any(controls != 0.0)


def test_method_seed_changes_only_perturbed_candidates():
    model = pin.buildModelFromUrdf(str(MODULE.MODEL_PATH))
    q0, _, _ = MODULE.generate_instance(model, MODULE.LANES["far"], 42)
    first, _ = MODULE.generate_seeds(model, q0, method_seed=1000, budget=8)
    second, _ = MODULE.generate_seeds(model, q0, method_seed=1001, budget=8)

    np.testing.assert_array_equal(first[0], second[0])
    assert not np.array_equal(first[1:], second[1:])


def test_solver_merit_gate_requires_finite_one_percent_decrease():
    passing = MODULE.solver_merit_improvement(100.0, 99.0)
    too_small = MODULE.solver_merit_improvement(100.0, 99.0001)
    nonfinite = MODULE.solver_merit_improvement(np.inf, 0.0)

    assert passing["passes_solver_merit_decrease_gate"]
    assert passing["solver_merit_required_decrease"] == 1.0
    assert not too_small["passes_solver_merit_decrease_gate"]
    assert not nonfinite["passes_solver_merit_decrease_gate"]
    assert MODULE.json_safe(nonfinite)["solver_merit_reduction_fraction"] is None


def test_repeatability_uses_relative_merit_spread():
    raw = [
        {
            "batched_initial_merit": [100.0, 200.0],
            "batched_final_merit": [10.0, 20.0],
        },
        {
            "batched_initial_merit": [100.00001, 199.99999],
            "batched_final_merit": [10.000001, 19.999999],
        },
    ]
    result = MODULE.merit_repeatability(raw, "batched")

    assert result["initial_merit"]["maximum_relative_span"] <= 1e-6
    assert result["final_merit"]["maximum_relative_span"] <= 1e-6


def test_limit_noninferiority_and_multistart_gain_helpers():
    metrics = {
        "finite": True,
        "terminal_error_m": 0.06,
        "nonnegative_task_motion_cost": 0.27,
        "joint_violation_rad": 0.0,
        "velocity_ratio": 0.2,
        "control_ratio": 0.1,
    }
    cold = {
        "candidate_index": 0,
        "optimizer_improved_success": False,
        "final_cuda": deepcopy(metrics),
        "final_pinocchio": deepcopy(metrics),
        "final_cuda_one_step_pinocchio_defect": 1e-6,
    }
    best = deepcopy(cold)
    best["candidate_index"] = 1
    best["optimizer_improved_success"] = True
    best["final_cuda"]["terminal_error_m"] = 0.02
    best["final_cuda"]["nonnegative_task_motion_cost"] = 0.05

    assert MODULE.final_rollouts_finite_and_within_limits(cold)
    invalid = deepcopy(cold)
    invalid["final_pinocchio"]["velocity_ratio"] = 1.01
    assert not MODULE.final_rollouts_finite_and_within_limits(invalid)
    gain = MODULE.multistart_quality_vs_cold([cold, best])
    assert gain["best_is_non_cold"]
    assert gain["best_successful_candidate_index"] == 1
    np.testing.assert_allclose(gain["terminal_error_reduction_vs_cold_fraction"], 2 / 3)


@pytest.mark.skipif(
    os.environ.get("GATO_RUN_CUDA_TESTS") != "1",
    reason="set GATO_RUN_CUDA_TESTS=1 for the build-backed smoke test",
)
def test_cuda_model_pair_and_productive_solve_smoke(tmp_path):
    output = tmp_path / "smoke.json"
    MODULE.run(
        SimpleNamespace(
            lane="near",
            instance_seed=40,
            method_seed=1000,
            budget=8,
            warmups=0,
            repeats=1,
            output=output,
        )
    )
    summary = __import__("json").loads(output.read_text(encoding="utf-8"))

    assert summary["model_pair_preflight"]["passes_1e-3_gate"]
    assert summary["optimizer_improved_success_count"] >= 1
    assert summary["pcg_cap_lane_count"] == 0
    assert summary["all_output_quality_noninferiority_checks_pass"]
