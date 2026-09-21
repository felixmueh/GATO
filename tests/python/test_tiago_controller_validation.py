"""Validate torque-horizon timing through the production execution loop.

The fake ROS client records commands and restoration; it does not model motion.
"""
import queue
from types import SimpleNamespace

import numpy as np
import pytest

from gato_tiago import ros_tiago as ros
from gato_tiago import tiago_controller_process as execution
from gato_tiago.safety_monitor import CollisionSafetySettings


@pytest.mark.parametrize("start", [40.0, np.nan, np.inf, -np.inf])
def test_invalid_start_is_rejected_by_production_validator(monkeypatch, start):
    monkeypatch.setattr(execution.time, "perf_counter", lambda: 10.0)
    with pytest.raises(ValueError, match="trajectory start"):
        execution._validate_trajectory(
            execution.TorqueTrajectory(np.ones((4, 7)), 0.01, start), 7, 26, True)



@pytest.mark.parametrize("start", [None, 10.0, 9.99])
def test_immediate_and_past_starts_remain_valid(monkeypatch, start):
    monkeypatch.setattr(execution.time, "perf_counter", lambda: 10.0)
    torques = np.arange(28).reshape(4, 7) / 10
    result = execution._validate_trajectory(
        execution.TorqueTrajectory(torques, 0.01, start), 7, 26, True)
    np.testing.assert_array_equal(result, torques)



def run_controller(monkeypatch, tmp_path, fault, *, invalid_start=None, match=None):
    clock = SimpleNamespace(now=0.0)
    class Time:
        @staticmethod
        def monotonic(): return clock.now
        @staticmethod
        def perf_counter(): return clock.now
        @staticmethod
        def sleep(seconds): clock.now += max(seconds, 0.001)
    monkeypatch.setattr(execution, "time", Time)
    ctx = execution.mp.get_context("spawn")
    shared = execution._SharedState(ctx)
    trajectories, statuses = queue.Queue(), queue.Queue()
    stop = SimpleNamespace(is_set=lambda: False)
    calls = []
    state = SimpleNamespace(q=np.zeros(7), qd=np.zeros(7), stamp_sec=1.0,
                            received_monotonic_sec=0.0, seq=1, age_sec=0.0,
                            joint_positions={n: 0.0 for n in ros.RIGHT_ARM_JOINTS},
                            joint_velocities={n: 0.0 for n in ros.RIGHT_ARM_JOINTS})
    class Arm:
        joint_names = ros.RIGHT_ARM_JOINTS
        def __init__(self, **kwargs):
            self.active = False
            self.injected = False
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def spin_once(self, timeout_sec=0): clock.now += max(timeout_sec, 0.001)
        def read_state(self, **kwargs): return state
        def latest_state(self): return state
        def switch_to_default_control(self, **kwargs): calls.append("restore")
        def publish_position_trajectory(self, *args, **kwargs): calls.append("reset")
        def configure_runtime_effort_controller(self, **kwargs):
            torques = np.ones((4, 7))
            if fault == "nan": torques[0, 0] = np.nan
            start = invalid_start if fault == "start_initial" else None
            trajectories.put(execution.TorqueTrajectory(torques, 0.01, start))
        def switch_to_effort_control(self, **kwargs):
            self.active = True
            calls.append("activate")
        def publish_effort(self, torques):
            if fault == "start_active" and self.active and not self.injected:
                self.injected = True
                trajectories.put(execution.TorqueTrajectory(
                    np.full((4, 7), 2.0), 0.01, invalid_start))
            if np.all(np.asarray(torques) == 2.0):
                calls.append("invalid_start_effort")
            if self.active and fault == "publish":
                calls.append("publish_error")
                raise RuntimeError("publication failed")
            calls.append("zero" if np.all(np.asarray(torques) == 0) else "effort")
    monkeypatch.setattr(ros, "TiagoRightArmClient", Arm)
    with pytest.raises((ValueError, RuntimeError), match=match):
        execution._controller_main(
            target_hz=100, reset_q=np.zeros(7), reset_duration_sec=0.01,
            stale_timeout_sec=0.1, max_abs_torque=30, clamp_torque=True,
            collision_safety=CollisionSafetySettings(enabled=False), history_max_records=20,
            shared_state=shared, trajectory_q=trajectories, status_q=statuses,
            history_path=tmp_path / "history.csv", history_metadata_path=tmp_path / "meta.json",
            stop_event=stop)
    return calls



@pytest.mark.parametrize("start", [30.0, np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("fault", ["start_initial", "start_active"])
def test_bad_start_cannot_activate_or_replace_running_horizon(
        monkeypatch, tmp_path, start, fault):
    calls = run_controller(monkeypatch, tmp_path, fault,
                           invalid_start=start, match="trajectory start")
    assert "invalid_start_effort" not in calls
    if fault == "start_initial":
        assert "activate" not in calls and "effort" not in calls
    else:
        assert "activate" in calls and "zero" in calls
    assert calls.count("restore") == 2
