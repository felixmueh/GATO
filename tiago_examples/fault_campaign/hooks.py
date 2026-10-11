"""Experiment-only fault adapters. Never imported by normal GATO experiments.

The independent runner writes the specification into the worker environment.
Multiprocessing children import these adapters too, so the injection occurs in
the real controller process, not just in the planner's view of it.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import time

import numpy as np


CASES = {
    "baseline": "Unmodified planner, controller, and feedback",
    "trajectory_nan": "One outgoing torque horizon contains NaN",
    "trajectory_shape": "One outgoing horizon has six joint columns",
    "trajectory_dt_zero": "One outgoing horizon has dt=0",
    "trajectory_expired": "One outgoing horizon starts ten seconds in the past",
    "trajectory_future": "One outgoing horizon starts thirty seconds in the future",
    "trajectory_start_nan": "One outgoing horizon has a NaN start timestamp",
    "torque_spike": "One outgoing horizon requests 60 Nm on every joint; clamp remains enabled",
    "planner_exception": "Planner raises while submitting the next horizon",
    "planner_stall": "Planner sleeps for two wall-clock seconds before submitting",
    "planner_exit": "Planner exits abruptly without Python cleanup",
    "feedback_drop": "Controller callback discards all subsequent joint states",
    "feedback_nan": "Controller receives NaN for one right-arm position",
    "feedback_frozen": "Controller receives frozen positions/velocities with fresh timestamps",
    "feedback_stamp_backwards": "Controller receives timestamps ten seconds earlier",
    "controller_exit": "Controller exits abruptly while publishing effort",
    "controller_stall": "Controller sleeps for two seconds while publishing effort",
    "safety_exit": "Controller kills its own safety worker",
    "publish_exception": "Effort publication raises, including during cleanup",
    "restore_exception": "Planner raises, then default-controller restoration raises",
}


def transform_trajectory(case, torques, dt, start, now):
    """Perturb the submitted horizon; production validation remains unchanged."""
    result = np.array(torques, dtype=float, copy=True)
    if case == "trajectory_nan":
        result[0, 0] = np.nan
    elif case == "trajectory_shape":
        result = result[:, :-1]
    elif case == "trajectory_dt_zero":
        dt = 0.0
    elif case == "trajectory_expired":
        start = now - 10.0
    elif case == "trajectory_future":
        start = now + 30.0
    elif case == "trajectory_start_nan":
        start = np.nan
    elif case == "torque_spike":
        result[:] = 60.0
    return result, dt, start


def event(directory, kind, **fields):
    record = {"wall": time.monotonic(), "pid": os.getpid(), "kind": kind, **fields}
    # Per-process files survive os._exit and avoid shared logging locks.
    path = Path(directory) / f"events-{os.getpid()}.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
        stream.flush()


def install(spec):
    import gato_tiago.ros_tiago as ros
    import gato_tiago.tiago_controller_process as execution
    from gato_tiago.safety_monitor import AsyncSafetyMonitor

    case = spec["case"]
    directory = Path(spec["directory"])
    armed = directory / "armed.json"
    fired = False
    injection_wall = None
    frozen = None

    def emit(kind, **fields):
        event(directory, kind, **fields)

    def due():
        if case == "baseline" or not armed.exists():
            return False
        with armed.open() as stream:
            t0 = json.load(stream)["wall"]
        return time.monotonic() >= t0 + spec["inject_after"]

    def trigger():
        nonlocal fired, injection_wall
        if fired or not due():
            return False
        fired = True
        injection_wall = time.monotonic()
        emit("injected", case=case)
        return True

    original_activate = ros.TiagoRightArmClient.switch_to_effort_control

    def activate(self, *args, **kwargs):
        result = original_activate(self, *args, **kwargs)
        temporary = directory / f"armed-{os.getpid()}.tmp"
        temporary.write_text(json.dumps({"wall": time.monotonic()}))
        temporary.replace(armed)
        emit("effort_activated")
        return result

    ros.TiagoRightArmClient.switch_to_effort_control = activate
    original_restore = ros.TiagoRightArmClient.switch_to_default_control

    def restore(self, *args, **kwargs):
        emit("restore_requested")
        if case == "restore_exception" and due():
            emit("restore_failed", error="injected restore exception")
            raise RuntimeError("FAULT: default controller restoration unavailable")
        try:
            result = original_restore(self, *args, **kwargs)
        except BaseException as exc:
            emit("restore_failed", error=str(exc))
            raise
        emit("restore_returned")
        return result

    ros.TiagoRightArmClient.switch_to_default_control = restore
    original_send = execution.TiagoControllerOrchestrator.send_trajectory
    planner_cases = {"planner_exception", "planner_stall", "planner_exit", "restore_exception"}

    def send(self, torques, dt, *, start_monotonic_sec=None):
        horizon_case = case.startswith("trajectory_") or case == "torque_spike"
        first = (horizon_case or case in planner_cases) and trigger()
        repeat = horizon_case and fired and time.monotonic() < injection_wall + spec.get("horizon_fault_duration", 0.0)
        if first or repeat:
            if case in {"planner_exception", "restore_exception"}:
                raise RuntimeError("FAULT: planner exception")
            if case == "planner_exit":
                os._exit(86)
            if case == "planner_stall":
                time.sleep(2.0)
            torques, dt, start_monotonic_sec = transform_trajectory(
                case, torques, dt, start_monotonic_sec, time.monotonic()
            )
        return original_send(self, torques, dt, start_monotonic_sec=start_monotonic_sec)

    execution.TiagoControllerOrchestrator.send_trajectory = send
    original_validate = execution._validate_trajectory

    def validate(traj, n_joints, max_abs_torque, clamp_torque):
        values = np.asarray(traj.torques)
        anomalous = (values.ndim != 2 or values.shape[-1] != n_joints or
                     not np.isfinite(values).all() or not np.isfinite(traj.dt) or
                     traj.dt <= 0 or (values.size and np.max(np.abs(values)) > max_abs_torque) or
                     (traj.start_monotonic_sec is not None and
                      (not np.isfinite(traj.start_monotonic_sec) or
                       abs(traj.start_monotonic_sec - time.monotonic()) > 5)))
        if anomalous:
            emit("anomalous_horizon_consumed")
        return original_validate(traj, n_joints, max_abs_torque, clamp_torque)

    execution._validate_trajectory = validate
    original_put_latest = execution._put_latest
    last_status = None

    def put_latest(target_queue, item):
        nonlocal last_status
        if isinstance(item, execution.ControllerStatus) and item.mode != last_status:
            last_status = item.mode
            emit("controller_status", mode=item.mode, error=item.error)
        return original_put_latest(target_queue, item)

    execution._put_latest = put_latest
    original_publish = ros.TiagoRightArmClient.publish_effort

    def publish(self, torques):
        if case in {"controller_exit", "controller_stall", "publish_exception"}:
            if trigger():
                if case == "controller_exit":
                    os._exit(87)
                if case == "controller_stall":
                    time.sleep(2.0)
            if fired and case == "publish_exception":
                emit("publish_failed")
                raise RuntimeError("FAULT: effort publication unavailable")
        result = original_publish(self, torques)
        emit("effort_published", max_abs=float(np.max(np.abs(torques))))
        return result

    ros.TiagoRightArmClient.publish_effort = publish
    original_callback = ros.TiagoRightArmClient._on_joint_state

    def callback(self, msg):
        nonlocal frozen
        if case.startswith("feedback_") and (fired or trigger()):
            if case == "feedback_drop":
                return
            msg = copy.deepcopy(msg)
            if case == "feedback_nan" and "arm_right_1_joint" in msg.name:
                msg.position[msg.name.index("arm_right_1_joint")] = np.nan
            elif case == "feedback_frozen":
                if frozen is None:
                    frozen = copy.deepcopy(msg)
                msg.name = frozen.name
                msg.position = frozen.position
                msg.velocity = frozen.velocity
            elif case == "feedback_stamp_backwards":
                msg.header.stamp.sec = max(0, msg.header.stamp.sec - 10)
        return original_callback(self, msg)

    ros.TiagoRightArmClient._on_joint_state = callback
    original_check = AsyncSafetyMonitor.check

    def check(self, state):
        if case == "safety_exit" and trigger():
            self._process.kill()
            self._process.join(timeout=1.0)
        return original_check(self, state)

    AsyncSafetyMonitor.check = check
    emit("hooks_installed", case=case)
