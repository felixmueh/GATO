"""Measured startup arrival and reset history, with no ROS or CUDA execution."""
import json
import queue
from types import SimpleNamespace as NS

import numpy as np
import pytest

from gato_tiago import ros_tiago as ros
from gato_tiago import tiago_controller_process as execution
from gato_tiago.safety_monitor import CollisionSafetySettings


@pytest.mark.parametrize("name", [
    "reset_position_tolerance_rad", "reset_velocity_l1_tolerance_rad_s",
    "reset_arrival_timeout_sec", "reset_duration_sec", "stale_timeout_sec",
])
@pytest.mark.parametrize("value", [0., -1., np.nan, np.inf])
def test_reset_thresholds_reject_unbounded_or_invalid_values(name, value):
    with pytest.raises(ValueError, match=f"{name} must be finite and positive"):
        execution.TiagoControllerOrchestrator(**{name: value})


def test_gate_is_opt_in_and_reset_target_must_be_finite():
    controller = execution.TiagoControllerOrchestrator()
    assert controller.reset_position_tolerance_rad is None
    with pytest.raises(ValueError, match="reset_q must be finite"):
        execution.TiagoControllerOrchestrator(reset_q=[np.nan] * 7)


def test_gate_configuration_reaches_worker(monkeypatch):
    controller = execution.TiagoControllerOrchestrator(
        reset_position_tolerance_rad=.04, reset_velocity_l1_tolerance_rad_s=.08,
        reset_arrival_timeout_sec=3.)
    controller._status_q = queue.Queue()
    monkeypatch.setattr(ros, "ensure_ros_environment", lambda **kwargs: None)
    monkeypatch.setattr(execution.atexit, "register", lambda *args: None)
    captured = {}
    def process(**kwargs):
        captured.update(kwargs["kwargs"])
        return NS(start=lambda: controller._status_q.put(execution.ControllerStatus("READY")))
    controller._ctx = NS(Process=process)
    controller.initialize(timeout_sec=1.)
    assert captured["reset_position_tolerance_rad"] == .04
    assert captured["reset_velocity_l1_tolerance_rad_s"] == .08
    assert captured["reset_arrival_timeout_sec"] == 3.


def sample(q=0., qd=0., received=0., seq=1):
    return NS(q=np.full(7, q), qd=np.full(7, qd), received_monotonic_sec=received,
              stamp_sec=received, seq=seq, age_sec=0.,
              joint_positions={name: q for name in ros.RIGHT_ARM_JOINTS},
              joint_velocities={name: qd for name in ros.RIGHT_ARM_JOINTS})


@pytest.mark.parametrize("fault", ["position", "velocity", "stale", "before_reset", "absent"])
def test_arrival_rejects_bad_or_old_measurements_until_bounded_timeout(monkeypatch, fault):
    clock = NS(now=1.)
    monkeypatch.setattr(execution, "time", NS(monotonic=lambda: clock.now))
    state = sample(q=.06 if fault == "position" else 0.,
                   qd=.02 if fault == "velocity" else 0.,
                   received=.5 if fault == "stale" else .99 if fault == "before_reset" else 1.)
    arm = NS(latest_state=lambda: None if fault == "absent" else state,
             spin_once=lambda timeout_sec: setattr(clock, "now", clock.now + timeout_sec))
    with pytest.raises(TimeoutError, match="reset arrival not verified"):
        execution._wait_for_reset_arrival(
            arm, reset_q=np.zeros(7), position_tolerance_rad=.05,
            velocity_l1_tolerance_rad_s=.1, timeout_sec=.15, stale_timeout_sec=.1,
            not_before_monotonic_sec=1., stop_event=NS(is_set=lambda: False))
    assert 1.15 <= clock.now <= 1.151


def test_arrival_waits_for_new_settled_sample_and_reports_observations(monkeypatch):
    clock = NS(now=1.)
    monkeypatch.setattr(execution, "time", NS(monotonic=lambda: clock.now))
    states = iter([sample(q=.2, received=1.), sample(qd=.03, received=1.05),
                   sample(q=.04, qd=.01, received=1.1)])
    observed = []
    arm = NS(latest_state=lambda: next(states),
             spin_once=lambda timeout_sec: setattr(clock, "now", clock.now + timeout_sec))
    result = execution._wait_for_reset_arrival(
        arm, reset_q=np.zeros(7), position_tolerance_rad=.05,
        velocity_l1_tolerance_rad_s=.1, timeout_sec=.5, stale_timeout_sec=.1,
        not_before_monotonic_sec=1., stop_event=NS(is_set=lambda: False),
        on_state=observed.append)
    assert len(observed) == 3 and result is observed[-1]
    assert clock.now == pytest.approx(1.1)


@pytest.mark.parametrize("field", ["q", "qd", "received_monotonic_sec"])
def test_nonfinite_measurements_cannot_pass_arrival(monkeypatch, field):
    state = sample()
    setattr(state, field, np.full(7, np.nan) if field in ("q", "qd") else np.nan)
    monkeypatch.setattr(execution, "time", NS(monotonic=lambda: 0.))
    with pytest.raises(RuntimeError, match="invalid joint state"):
        execution._wait_for_reset_arrival(
            NS(latest_state=lambda: state), reset_q=np.zeros(7),
            position_tolerance_rad=.05, velocity_l1_tolerance_rad_s=.1,
            timeout_sec=.5, stale_timeout_sec=.1, not_before_monotonic_sec=0.,
            stop_event=NS(is_set=lambda: False))


@pytest.mark.parametrize("outcome", ["arrived", "timeout", "cancel_reset", "cancel_arrival"])
def test_worker_keeps_default_control_until_arrival_and_retains_reset_path(
        monkeypatch, tmp_path, outcome):
    clock = NS(now=0.)
    monkeypatch.setattr(execution, "time", NS(
        monotonic=lambda: clock.now, perf_counter=lambda: clock.now,
        sleep=lambda dt: setattr(clock, "now", clock.now + dt)))
    statuses = queue.Queue()
    calls, observed_modes = [], []
    original_put = execution._put_latest
    def record_status(q, status):
        observed_modes.append(status.mode)
        original_put(q, status)
    monkeypatch.setattr(execution, "_put_latest", record_status)
    latest = NS(state=sample(q=.3))
    class Arm:
        joint_names = ros.RIGHT_ARM_JOINTS
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def switch_to_default_control(self, **kwargs): calls.append("restore")
        def publish_position_trajectory(self, *args, **kwargs): calls.append("reset")
        def latest_state(self): return latest.state
        def spin_once(self, timeout_sec=0.):
            clock.now += timeout_sec
            latest.state = sample(q=0. if clock.now >= .15 and outcome == "arrived" else .2,
                                  received=clock.now, seq=latest.state.seq + 1)
        def configure_runtime_effort_controller(self, **kwargs): calls.append("configure")
    monkeypatch.setattr(ros, "TiagoRightArmClient", Arm)
    stop = NS(is_set=lambda: ("READY" in observed_modes
              or (outcome == "cancel_reset" and clock.now >= .05)
              or (outcome == "cancel_arrival" and clock.now >= .15)))
    def run():
        execution._controller_main(
            target_hz=100., reset_q=np.zeros(7), reset_duration_sec=.1,
            stale_timeout_sec=.1, reset_position_tolerance_rad=.05,
            reset_velocity_l1_tolerance_rad_s=.1, reset_arrival_timeout_sec=.2,
            max_abs_torque=26., clamp_torque=False,
            collision_safety=CollisionSafetySettings(enabled=False), history_max_records=100,
            shared_state=execution._SharedState(execution.mp.get_context("spawn")),
            trajectory_q=queue.Queue(), status_q=statuses,
            history_path=tmp_path / "history.csv", history_metadata_path=tmp_path / "meta.json",
            stop_event=stop)
    if outcome == "timeout":
        with pytest.raises(TimeoutError, match="reset arrival not verified"):
            run()
        assert execution._get_latest(statuses).mode == "ERROR_RESTORE"
    else:
        run()
        assert execution._get_latest(statuses).mode == "RESTORED"
    assert calls.count("restore") == 2
    assert ("configure" in calls) == (outcome == "arrived")
    assert ("READY" in observed_modes) == (outcome == "arrived")
    records = [json.loads(line) for line in
               (tmp_path / "full_joint_state_history.jsonl").read_text().splitlines()]
    assert records[0]["q"] == [.3] * 7  # Captured before publishing the reset.
    assert all(record["controller_mode"] == "RESETTING" for record in records)
    assert len(records) >= 2
    assert len({record["source_seq"] for record in records}) == len(records)
    if outcome == "arrived":
        assert records[-1]["q"] == [0.] * 7
