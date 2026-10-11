"""Exercise production fault handling without ROS or CUDA.

The fake ROS client records calls; it cannot establish physical stopping.
The controller loop, torque validation, error handling, and cleanup are real.
"""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tiago_src"), str(ROOT / "tiago_examples/fault_campaign")]
from hooks import transform_trajectory
from gato_tiago import tiago_controller_process as execution
from test_tiago_controller_validation import run_controller


@pytest.mark.parametrize("case,error", [
    ("trajectory_nan", "non-finite"), ("trajectory_shape", "shape"),
    ("trajectory_dt_zero", "positive and finite"),
])
def test_injected_invalid_horizon_is_rejected_by_production_validator(case, error):
    torques, dt, start = transform_trajectory(case, np.ones((4, 7)), 0.01, None, 10)
    with pytest.raises(ValueError, match=error):
        execution._validate_trajectory(execution.TorqueTrajectory(torques, dt, start), 7, 30, True)


def test_spike_is_clamped_by_production_validator():
    torques, dt, start = transform_trajectory("torque_spike", np.ones((4, 7)), 0.01, None, 10)
    with pytest.warns(RuntimeWarning, match="clamping"):
        result = execution._validate_trajectory(execution.TorqueTrajectory(torques, dt, start), 7, 30, True)
    assert np.max(result) == 30


def test_controller_rejects_nan_before_effort_activation(monkeypatch, tmp_path):
    calls = run_controller(monkeypatch, tmp_path, "nan")
    assert "activate" not in calls and "effort" not in calls
    assert calls.count("restore") == 2  # startup plus error cleanup


def test_expired_horizon_restores_after_zero_effort(monkeypatch, tmp_path):
    calls = run_controller(monkeypatch, tmp_path, "expired")
    assert "activate" in calls and "zero" in calls
    assert calls[-1] == "restore"


def test_publication_failure_currently_prevents_cleanup_switch(monkeypatch, tmp_path):
    calls = run_controller(monkeypatch, tmp_path, "publish")
    assert calls.count("restore") == 1  # startup only: characterize the gap
    assert calls[-2:] == ["publish_error", "publish_error"]


def test_report_does_not_call_missing_fault_evidence_success(tmp_path):
    spec = importlib.util.spec_from_file_location("campaign", ROOT / "tiago_examples/fault_campaign/run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "trial.json").write_text(json.dumps({"case": "planner_exit", "worker_exit": 86}))
    (tmp_path / "observer.jsonl").write_text("")
    (tmp_path / "worker.log").write_text("")
    result = module.summarize(tmp_path)
    assert result["outcome"] == "inconclusive"
    assert result["max_post_fault_joint_displacement_rad"] is None


def test_report_requires_stable_positions_as_well_as_low_reported_velocity(tmp_path):
    spec = importlib.util.spec_from_file_location("campaign", ROOT / "tiago_examples/fault_campaign/run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    controllers = [{"name": "arm_right_controller", "state": "active"},
                   {"name": "arm_right_gravity_compensation_controller", "state": "active"}]
    (tmp_path / "trial.json").write_text(json.dumps({"case": "planner_exit", "worker_exit": 86,
        "finished_wall": 3.1, "controller_snapshots": [{"wall": 3.0, "controllers": controllers}]}))
    (tmp_path / "events-1.jsonl").write_text('\n'.join(json.dumps(e) for e in [
        {"kind": "effort_activated", "wall": 0.0}, {"kind": "injected", "wall": 1.0}]) + '\n')
    (tmp_path / "observer.jsonl").write_text('\n'.join(json.dumps({
        "wall": float(t), "q": [float(t)] * 7, "qd": [0.0] * 7})
        for t in np.arange(0.98, 3.02, 0.02)) + '\n')
    (tmp_path / "worker.log").write_text("")
    result = module.summarize(tmp_path)
    assert result["settled_final_half_second"] is False
