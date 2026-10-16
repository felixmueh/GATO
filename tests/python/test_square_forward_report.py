"""CPU-only checks of recorded-sample square reporting."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

SCRIPT = Path(__file__).resolve().parents[2] / "tiago_examples/physical_arm/report_square_forward.py"
spec = importlib.util.spec_from_file_location("square_report", SCRIPT)
reporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporter)


def config():
    return {"GOALS_POSES_M_RAD": [[i, 0., 0., 0., 0., 0.] for i in range(4)],
            "GOAL_NAMES": ["a", "b", "c", "d"], "ROUGH_POSITION_TOLERANCE_M": .05,
            "ROUGH_TOOL_AXIS_TOLERANCE_DEG": 30., "ROUGH_SPEED_L1_RAD_S": 1., "ROUGH_DWELL_S": 0.}


def fk(q):
    return q[:3], Rotation.from_euler("z", q[3]).as_matrix()


def q(x, roll=0.):
    return [x, 0., 0., roll, 0., 0., 0.]


def stats():
    return {"timestamps": [1., 2., 3.], "goal_indices": [0, 1, 2],
            "joint_positions": [q(0), q(1), q(2)], "joint_velocities": [[0.] * 7] * 3,
            "goal_outcomes": ["reached"] * 4, "goal_reached_times": [1., 2., 3., 4.],
            "terminal_state": q(3) + [0.] * 7, "terminal_time_s": 4.}


def test_terminal_fourth_corner_and_strict_fields_are_recorded():
    raw = stats()
    result = reporter.build_report(config(), raw, fk)
    assert result["status"] == "complete_recorded_sequence"
    final = result["goals"][3]
    assert final["last_observed"]["scope"] == "terminal_boundary"
    assert final["last_observed"]["time_s"] == 4.
    assert final["rough_first_observed_entry"]["time_s"] == 4.
    assert final["strict_reached_time_s_recorded"] == 4.
    assert final["last_observation_at_strict_arrival"] is True
    assert result["strict_goal_outcomes_recorded"] == raw["goal_outcomes"]


def test_axial_roll_is_separate_from_axis_and_rough_entry():
    raw = stats()
    raw["joint_positions"][0] = q(0., np.pi / 2)
    sample = reporter.build_report(config(), raw, fk)["goals"][0]["rough_first_observed_entry"]
    assert sample["tool_axis_error_deg"] == pytest.approx(0.)
    assert sample["full_so3_error_deg"] == pytest.approx(90.)


def test_rough_entry_uses_goal_identity_and_all_three_conditions():
    raw = stats()
    raw.update(timestamps=[.1, .2, .3, .4], goal_indices=[0, 0, 0, 1],
               joint_positions=[q(.06), q(0.), q(0.), q(0.)],
               joint_velocities=[[0.] * 7, [1.1] + [0.] * 6, [0.] * 7, [0.] * 7])
    result = reporter.build_report(config(), raw, fk)
    goals = result["goals"]
    assert goals[0]["rough_first_observed_entry"]["time_s"] == .3
    assert goals[1]["rough_first_observed_entry"] is None
    assert goals[2]["last_observed"] is None
    assert result["status"] == "incomplete"
    assert result["recorded_outcomes_complete"] is True
    assert goals[0]["last_observation_at_strict_arrival"] is False
    assert goals[2]["last_observation_at_strict_arrival"] is None


def test_partial_error_does_not_assign_terminal_or_unvisited_goals():
    raw = stats()
    raw.update(timestamps=[1.], goal_indices=[0], joint_positions=[q(0)], joint_velocities=[[0.] * 7],
               goal_outcomes=["reached", "not_reached", "not_reached", "not_reached"],
               goal_reached_times=[1., None, None, None])
    result = reporter.build_report(config(), raw, fk)
    assert result["status"] == "incomplete"
    assert "Terminal boundary not assigned" in result["issues"][0]
    for goal in result["goals"][1:]:
        assert goal["observation_status"] == "missing_observations"
        assert goal["rough_first_observed_entry"] is None
        assert goal["last_observed"] is None


def test_missing_stats_and_initial_boundary():
    missing = reporter.build_report(config(), {}, fk)
    assert missing["status"] == "incomplete"
    assert all(g["last_observed"] is None for g in missing["goals"])
    initial = reporter.build_report(config(), {"initial_state": q(0) + [0.] * 7}, fk)
    assert initial["goals"][0]["rough_first_observed_entry"]["scope"] == "initial_boundary"
    assert initial["goals"][1]["last_observed"] is None


def test_misaligned_or_nonfinite_samples_remain_incomplete():
    raw = stats()
    raw["joint_positions"][0] = q(float("nan"))
    raw["goal_indices"].append(3)
    result = reporter.build_report(config(), raw, fk)
    assert result["status"] == "incomplete"
    assert result["goals"][0]["last_observed"] is None
    assert len(result["issues"]) == 2
