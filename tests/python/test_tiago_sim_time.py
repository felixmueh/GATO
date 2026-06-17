import json
from types import SimpleNamespace
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TIAGO_SRC = REPO_ROOT / "tiago_src"

if str(TIAGO_SRC) not in sys.path:
    sys.path.insert(0, str(TIAGO_SRC))

from gato_tiago.tiago_controller_process import (
    DEFAULT_COLLISION_BLACKLIST_PATH,
    CollisionSafetySettings,
    TiagoControllerOrchestrator,
    elapsed_sim_time_from_stamp,
    _format_full_state_history_row,
    _sample_trajectory,
)


def test_elapsed_sim_time_uses_initial_stamp_as_zero():
    assert elapsed_sim_time_from_stamp(100.0, 100.0) == pytest.approx(0.0)
    assert elapsed_sim_time_from_stamp(100.02, 100.0, 0.0) == pytest.approx(0.02)


def test_elapsed_sim_time_allows_repeated_robot_state_stamps():
    stamps = [100.0, 100.02, 100.02, 100.04]
    elapsed = []
    previous = None
    for stamp in stamps:
        current = elapsed_sim_time_from_stamp(stamp, stamps[0], previous)
        elapsed.append(current)
        previous = current

    assert elapsed == pytest.approx([0.0, 0.02, 0.02, 0.04])


def test_elapsed_sim_time_rejects_backward_robot_state_stamps():
    with pytest.raises(RuntimeError, match="moved backward"):
        elapsed_sim_time_from_stamp(100.01, 100.0, 0.02)


def test_collision_safety_defaults_to_40mm_clearance():
    settings = CollisionSafetySettings()

    assert settings.enabled is True
    assert settings.min_distance_m == pytest.approx(0.04)
    assert settings.max_body_speed_m_s == pytest.approx(1.0)
    assert settings.joint_position_margin == pytest.approx(0.0)
    assert settings.joint_velocity_scale == pytest.approx(1.0)
    assert settings.blacklist_path == DEFAULT_COLLISION_BLACKLIST_PATH


def test_orchestrator_uses_default_collision_blacklist_when_unspecified():
    controller = TiagoControllerOrchestrator(collision_blacklist_path=None)

    assert controller.collision_safety.blacklist_path == DEFAULT_COLLISION_BLACKLIST_PATH


def test_orchestrator_validates_joint_limit_settings():
    with pytest.raises(ValueError, match="joint_position_margin"):
        TiagoControllerOrchestrator(joint_position_margin=-0.01)
    with pytest.raises(ValueError, match="joint_velocity_scale"):
        TiagoControllerOrchestrator(joint_velocity_scale=0.0)


def test_sample_trajectory_returns_in_horizon_rows():
    torques = np.asarray([[1.0], [2.0], [3.0]], dtype=np.float64)

    tau, in_horizon = _sample_trajectory(
        torques,
        dt=0.1,
        start_monotonic_sec=10.0,
        now=10.15,
    )

    assert tau == pytest.approx([2.0])
    assert in_horizon is True


def test_full_state_history_row_contains_named_joint_state():
    row = _format_full_state_history_row(
        SimpleNamespace(
            seq=7,
            stamp_sec=12.5,
            received_monotonic_sec=99.0,
            joint_positions={"joint_a": 1.25},
            joint_velocities={"joint_a": -0.5, "wheel_joint": float("nan")},
        ),
        "RUNNING",
    )

    data = json.loads(row)
    assert data["source_seq"] == 7
    assert data["controller_mode"] == "RUNNING"
    assert data["positions_by_name"] == {"joint_a": 1.25}
    assert data["velocities_by_name"] == {"joint_a": -0.5, "wheel_joint": None}
