import json
from types import SimpleNamespace
import sys
from pathlib import Path
import time

import numpy as np
import pinocchio as pin
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TIAGO_SRC = REPO_ROOT / "tiago_src"

if str(TIAGO_SRC) not in sys.path:
    sys.path.insert(0, str(TIAGO_SRC))

from gato_tiago.tiago_controller_process import (
    TiagoControllerOrchestrator,
    elapsed_sim_time_from_stamp,
    _format_full_state_history_row,
    _sample_trajectory,
)
from gato_tiago.safety_monitor import (
    AsyncSafetyMonitor,
    CollisionSafetySettings,
    DEFAULT_COLLISION_BLACKLIST_PATH,
    DEFAULT_COLLISION_URDF_PATH,
    SafetyCheckState,
)
from gato_tiago.config import TIAGO_RIGHT_DEFAULT_START_CONFIG, TIAGO_RIGHT_START_CONFIGS
from gato_tiago.ros_tiago import RIGHT_ARM_JOINTS


PASSIVE_LEFT_ARM_COMFORTABLE_FORWARD_OUTWARD = np.array(
    [0.39, -1.73, 0.38, -2.35, 0.0, -1.21, -0.04],
    dtype=np.float64,
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
    assert settings.check_timeout_sec == pytest.approx(0.05)
    assert settings.max_monitored_geometry_speed_m_s == pytest.approx(1.0)
    assert settings.joint_position_margin_rad == pytest.approx(0.0)
    assert settings.joint_velocity_scale == pytest.approx(1.0)
    assert settings.blacklist_path == DEFAULT_COLLISION_BLACKLIST_PATH
    assert settings.urdf_path == DEFAULT_COLLISION_URDF_PATH
    assert settings.urdf_path.is_file()


def test_orchestrator_uses_default_collision_blacklist_when_unspecified():
    controller = TiagoControllerOrchestrator(collision_blacklist_path=None)

    assert controller.collision_safety.blacklist_path == DEFAULT_COLLISION_BLACKLIST_PATH


def test_orchestrator_validates_joint_limit_settings():
    with pytest.raises(ValueError, match="collision_check_timeout_sec"):
        TiagoControllerOrchestrator(collision_check_timeout_sec=0.0)
    with pytest.raises(ValueError, match="joint_position_margin_rad"):
        TiagoControllerOrchestrator(joint_position_margin_rad=-0.01)
    with pytest.raises(ValueError, match="joint_velocity_scale"):
        TiagoControllerOrchestrator(joint_velocity_scale=0.0)


def test_async_collision_safety_worker_reports_for_comfortable_state():
    full_model = pin.buildModelFromUrdf(str(DEFAULT_COLLISION_URDF_PATH))
    positions = {name: 0.0 for name in full_model.names if name != "universe"}
    velocities = {name: 0.0 for name in positions}
    positions["torso_lift_joint"] = 0.0999909568
    start_q = TIAGO_RIGHT_START_CONFIGS[TIAGO_RIGHT_DEFAULT_START_CONFIG]
    for idx, value in enumerate(start_q, start=1):
        positions[f"arm_right_{idx}_joint"] = float(value)
    for idx, value in enumerate(PASSIVE_LEFT_ARM_COMFORTABLE_FORWARD_OUTWARD, start=1):
        positions[f"arm_left_{idx}_joint"] = float(value)
    positions["gripper_left_finger_joint"] = 0.15
    positions["gripper_right_finger_joint"] = 0.15
    state = SafetyCheckState(
        q=np.asarray(start_q, dtype=np.float64),
        qd=np.zeros(7, dtype=np.float64),
        stamp_sec=0.0,
        received_monotonic_sec=0.0,
        seq=1,
        joint_positions=positions,
        joint_velocities=velocities,
    )
    settings = CollisionSafetySettings(
        controlled_joint_names=tuple(RIGHT_ARM_JOINTS),
        min_distance_m=0.0,
        check_timeout_sec=0.5,
        blacklist_path=DEFAULT_COLLISION_BLACKLIST_PATH,
    )
    monitor = AsyncSafetyMonitor(settings=settings, initial_state=state)
    try:
        start = time.perf_counter()
        monitor.check(state)
        assert time.perf_counter() - start < 0.05

        deadline = time.perf_counter() + settings.check_timeout_sec
        while monitor._last_checked_seq != state.seq and time.perf_counter() < deadline:
            monitor.check(state)
            time.sleep(0.001)
        assert monitor._last_checked_seq == state.seq
    finally:
        monitor.close()


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
        safety_status="ok",
    )

    data = json.loads(row)
    assert data["source_seq"] == 7
    assert data["controller_mode"] == "RUNNING"
    assert data["safety_status"] == "ok"
    assert data["safety_fault"] == ""
    assert data["positions_by_name"] == {"joint_a": 1.25}
    assert data["velocities_by_name"] == {"joint_a": -0.5, "wheel_joint": None}
