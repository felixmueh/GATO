import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
KNOTS = 32
MODULE = f"bsqp.bsqpN{KNOTS}_tiago_right_multimodal"


@pytest.mark.cuda
@pytest.mark.tracking
def test_easy_pillar_instance_covers_two_ranked_winding_modes(tmp_path):
    if os.environ.get("GATO_RUN_MULTIMODAL_TESTS") != "1":
        pytest.skip("set GATO_RUN_MULTIMODAL_TESTS=1 to run the multimodal CUDA test")
    if importlib.util.find_spec(MODULE) is None:
        pytest.skip(f"built solver extension is not available: {MODULE}")

    command = [
        sys.executable,
        "tiago_examples/tiago_multimodal_pillar.py",
        "run",
        "--difficulty",
        "easy",
        "--seed",
        "30",
        "--candidate-budget",
        "8",
        "--require-ranked",
        "--output-root",
        str(tmp_path),
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    summary_path = tmp_path / "easy" / "seed_000030" / "budget_08" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    certificate = summary["certificate"]
    sensitivity = summary["initialization_sensitivity"]
    assert sensitivity["same_initial_state_reference_and_objective"]
    assert sensitivity["only_full_trajectory_initialization_varies"]
    assert sensitivity["all_certifiable_candidates_unchanged_by_sqp"]
    assert sensitivity["maximum_solution_delta_from_seed"] == 0.0
    assert sensitivity["maximum_reconstructed_objective_improvement"] == 0.0
    assert certificate["ranked_finite_budget_benchmark"]
    assert certificate["certifiable_support"]["clockwise"] >= 2
    assert certificate["certifiable_support"]["counterclockwise"] >= 2
    assert certificate["cross_mode_midpoint_min_clearance_m"] < 0.0
    assert certificate["mode_objective_gap_fraction"] >= 0.02
    certifiable = [row for row in summary["results"] if row["certifiable"]]
    assert certifiable
    assert all(row["rollout_finite"] for row in certifiable)
    assert all(row["max_cuda_planned_dynamics_defect"] <= 1e-4 for row in certifiable)
    assert all(row["physical_feasible"] for row in certifiable)
    assert all(abs(row["final_solver_merit"] - row["objective"]) <= 1e-3 for row in certifiable)
    assert all(row["planned_initial_state_error"] <= 1e-6 for row in certifiable)
    assert all(row["max_planned_state_to_rollout_error"] <= 1e-3 for row in certifiable)
    assert all(
        row["terminal_error_m"] <= summary["difficulty"]["terminal_tolerance_m"]
        for row in certifiable
    )
    assert all(
        row["min_dense_clearance_m"]
        >= summary["difficulty"]["minimum_clearance_margin_m"]
        for row in certifiable
    )
    assert all(row["max_joint_violation_rad"] == 0.0 for row in certifiable)
    assert all(row["max_velocity_ratio"] <= 1.0 for row in certifiable)
    assert all(row["max_torque_ratio"] <= 1.0 for row in certifiable)
    assert all("solution_l2_delta_from_calibrated_seed" in row for row in summary["results"])
    assert all("reconstructed_objective_improvement" in row for row in summary["results"])

    trajectory_path = summary_path.with_name("trajectories.npz")
    with np.load(trajectory_path, allow_pickle=False) as archive:
        assert "planned_packed_trajectories" in archive
        assert "exact_rollout_packed_trajectories" in archive
        planned = archive["planned_packed_trajectories"]
        rollout = archive["exact_rollout_packed_trajectories"]
        candidate_names = archive["candidate_names"]
        x0 = np.asarray(summary["instance"]["start_q"] + [0.0] * 7, dtype=np.float32)
        for row in certifiable:
            index = row["candidate_index"]
            np.testing.assert_array_equal(planned[index, :14], x0)
            np.testing.assert_allclose(planned[index], rollout[index], atol=1e-6, rtol=0.0)

    evaluation_dir = tmp_path / "submission_evaluation"
    evaluation_command = [
        sys.executable,
        "tiago_examples/tiago_multimodal_pillar.py",
        "evaluate",
        "--instance-file",
        str(summary_path.with_name("instance.json")),
        "--trajectories",
        str(trajectory_path),
        "--output-dir",
        str(evaluation_dir),
        "--method-name",
        "builtin-roundtrip",
        "--method-runtime-ms",
        "0",
        "--proposal-budget",
        "8",
        "--oracle-summary",
        str(summary_path),
    ]
    subprocess.run(
        evaluation_command, cwd=REPO_ROOT, check=True, capture_output=True, text=True
    )
    evaluation = json.loads(
        (evaluation_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert evaluation["certificate"] == certificate
    assert evaluation["evaluation_metadata"]["proposal_budget"] == 8
    assert evaluation["evaluation_metadata"]["normalized_regret_to_oracle"] == 0.0

    tampered = planned.copy()
    tampered[1, 0] += 0.02
    tampered_path = tmp_path / "tampered_trajectories.npz"
    np.savez_compressed(
        tampered_path,
        planned_packed_trajectories=tampered,
        candidate_names=candidate_names,
    )
    tampered_dir = tmp_path / "tampered_evaluation"
    tampered_command = [
        sys.executable,
        "tiago_examples/tiago_multimodal_pillar.py",
        "evaluate",
        "--instance-file",
        str(summary_path.with_name("instance.json")),
        "--trajectories",
        str(tampered_path),
        "--output-dir",
        str(tampered_dir),
        "--method-name",
        "tampered",
        "--method-runtime-ms",
        "1",
        "--proposal-budget",
        "8",
    ]
    subprocess.run(
        tampered_command, cwd=REPO_ROOT, check=True, capture_output=True, text=True
    )
    tampered_summary = json.loads(
        (tampered_dir / "summary.json").read_text(encoding="utf-8")
    )
    tampered_row = tampered_summary["results"][1]
    assert not tampered_row["certifiable"]
    assert tampered_row["planned_initial_state_error"] > 0.01
