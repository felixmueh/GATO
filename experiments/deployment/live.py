"""Experiment-local recording proxies; all control remains in public APIs."""
from dataclasses import asdict
import copy
import json
from pathlib import Path
import threading
import time

import numpy as np


def record_cleanup_event(recorder, kind, record, errors):
    """A failed evidence write must never prevent shutdown or replace its error."""
    try:
        recorder(kind, record)
    except BaseException as exc:
        errors.append(dict(stage='recorder', event=kind,
                           type=type(exc).__name__, message=str(exc)))


class RecordedSolver:
    def __init__(self, solver, recorder, clock=time.perf_counter):
        self.delegate = solver
        self.recorder = recorder
        self.clock = clock
        self.wall_times = []
        self.native_times_us = []
        self.recording_times = []
        self.recording_errors = []

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def solve(self, xcur_B, eepos_goals_B, XU_B=None):
        # BSQP overwrites the seed's first state: preserve the public arguments
        # before that mutation and before starting the wrapper timer.
        recording_start = time.perf_counter()
        record = dict(state=copy.deepcopy(xcur_B), reference=copy.deepcopy(eepos_goals_B),
                      warm_start=copy.deepcopy(XU_B),
                      implicit_warm_start=copy.deepcopy(getattr(self.delegate, 'XU_B', None))
                      if XU_B is None else None,
                      started_monotonic_s=time.monotonic())
        input_copy_s = time.perf_counter() - recording_start
        start = self.clock()
        try:
            result = self.delegate.solve(xcur_B, eepos_goals_B, XU_B)
        except BaseException as exc:
            elapsed = self.clock() - start
            recording_start = time.perf_counter()
            record.update(wrapper_wall_s=elapsed, exception_type=type(exc).__name__,
                          exception=str(exc))
            self.wall_times.append(elapsed)
            record_cleanup_event(self.recorder, 'solve_failure', record, self.recording_errors)
            self.recording_times.append(input_copy_s + time.perf_counter() - recording_start)
            raise
        elapsed = self.clock() - start
        recording_start = time.perf_counter()
        # No get_stats call or device synchronization is added. BSQP.solve
        # already populated its public host stats and returned host output.
        record.update(wrapper_wall_s=elapsed, output=copy.deepcopy(result[0]),
                      returned_native_time_us=copy.deepcopy(result[1]),
                      available_stats=copy.deepcopy(getattr(self.delegate, 'stats', {})))
        self.wall_times.append(elapsed)
        self.native_times_us.append(copy.deepcopy(result[1]))
        self.recorder('solve', record)
        self.recording_times.append(input_copy_s + time.perf_counter() - recording_start)
        return result

    def get_stats(self):
        # Forward the task loop's existing call exactly; no duplicate fetch.
        return self.delegate.get_stats()


class RecordedController:
    def __init__(self, controller, recorder):
        self.delegate = controller
        self.recorder = recorder
        self.states = []
        self.handback_confirmed = False
        self.close_attempted = False

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def initialize(self, **kwargs):
        self.recorder('initialize_request', kwargs)
        result = self.delegate.initialize(**kwargs)
        self.recorder('initialized', {})
        return result

    def read_state(self, **kwargs):
        state = self.delegate.read_state(**kwargs)
        if state is not None:
            snapshot = asdict(state)
            self.states.append(snapshot)
            self.recorder('feedback_state', snapshot)
        return state

    def send_trajectory(self, torques, dt, *args, **kwargs):
        # Retain the requested full horizon and its publication request time.
        self.recorder('planned_published_horizon', dict(torques=copy.deepcopy(torques),
            dt_s=dt, request_monotonic_s=time.monotonic(), arguments=args, options=kwargs))
        return self.delegate.send_trajectory(torques, dt, *args, **kwargs)

    def finish_control(self, **kwargs):
        self.recorder('finish_control_request', kwargs)
        result = self.delegate.finish_control(**kwargs)
        self.recorder('finish_control_confirmed', {})
        return result

    def close(self, **kwargs):
        self.close_attempted = True
        return self.delegate.close(**kwargs)


def execute_with_handback(controller, task, output, recorder, close_timeout_s=8.0):
    """Save histories on success, task failure and unconfirmed restoration."""
    errors = []
    stats = {}
    try:
        _, stats = task(controller)
    except BaseException as exc:
        import traceback
        errors.append(dict(stage='task', type=type(exc).__name__, message=str(exc),
                           traceback=traceback.format_exc()))
        record_cleanup_event(recorder, 'execution_failure', errors[-1], errors)
    finally:
        record_cleanup_event(recorder, 'close_request', dict(timeout_sec=close_timeout_s), errors)
        try:
            controller.close(timeout_sec=close_timeout_s)
            controller.handback_confirmed = True
        except BaseException as exc:
            errors.append(dict(stage='handback', type=type(exc).__name__, message=str(exc)))
            record_cleanup_event(recorder, 'handback_failure', errors[-1], errors)
        else:
            record_cleanup_event(recorder, 'close_confirmed_restored', {}, errors)
        # Separate attempts ensure a CSV failure does not discard full histories.
        for method, filename in (('write_state_history_csv', 'controller_state_history.csv'),
                                ('write_full_state_history_jsonl', 'full_joint_state_history.jsonl')):
            try:
                summary = getattr(controller, method)(Path(output) / filename)
            except BaseException as exc:
                errors.append(dict(stage=method, type=type(exc).__name__, message=str(exc)))
                record_cleanup_event(recorder, 'history_failure', errors[-1], errors)
            else:
                if summary is not None:
                    record_cleanup_event(recorder, 'history_summary', summary, errors)
    return stats, errors


class GazeboObserver:
    """Read-only subscriptions, including effort publications without a header.

    ROS imports are lazy. The environment must already be sourced. Reception
    time and latest observed clock are labels, not invented publication stamps.
    """
    EFFORT_TOPIC = '/gato_arm_effort_forward_runtime/commands'

    def __init__(self, output):
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray
        self.rclpy = rclpy
        self.owns_context = not rclpy.ok()
        if self.owns_context:
            rclpy.init()
        self.node = rclpy.create_node('gato_d01_read_only_observer')
        self.stream = (Path(output) / 'ros_observations.jsonl').open('x')
        self.clocks = []
        self.joint_states = []
        self.efforts = []
        self.error = None
        self.stop = threading.Event()
        self.node.create_subscription(Clock, '/clock', self._clock, qos_profile_sensor_data)
        self.node.create_subscription(JointState, '/joint_states', self._joints, qos_profile_sensor_data)
        self.node.create_subscription(Float64MultiArray, self.EFFORT_TOPIC,
                                      self._effort, qos_profile_sensor_data)
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

    def _write(self, topic, payload):
        row = dict(topic=topic, received_monotonic_s=time.monotonic(),
                   observed_clock=self.clocks[-1] if self.clocks else None, **payload)
        # Retain ROS nonfinite values literally in this raw JSONL; metrics use
        # strict JSON. No observations are dropped to make summaries finite.
        self.stream.write(json.dumps(row) + '\n')
        self.stream.flush()
        return row

    def _clock(self, msg):
        stamp = dict(sec=msg.clock.sec, nanosec=msg.clock.nanosec)
        self.clocks.append(stamp)
        self._write('/clock', dict(stamp=stamp))

    def _joints(self, msg):
        self.joint_states.append(self._write('/joint_states', dict(
            stamp=dict(sec=msg.header.stamp.sec, nanosec=msg.header.stamp.nanosec),
            frame_id=msg.header.frame_id, names=list(msg.name),
            position=list(msg.position), velocity=list(msg.velocity), effort=list(msg.effort))))

    def _effort(self, msg):
        self.efforts.append(self._write(self.EFFORT_TOPIC, dict(
            data=list(msg.data), layout=dict(data_offset=msg.layout.data_offset,
                dim=[dict(label=d.label, size=d.size, stride=d.stride) for d in msg.layout.dim]),
            source_stamp=None, meaning='observed controller-published effort, not actuator torque')))

    def _spin(self):
        try:
            while not self.stop.is_set() and self.rclpy.ok():
                self.rclpy.spin_once(self.node, timeout_sec=0.05)
        except BaseException as exc:
            self.error = repr(exc)

    def require_advancing_clock(self, timeout_s):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.error:
                raise RuntimeError(self.error)
            clocks = list(self.clocks)
            if (len(clocks) > 1 and self.joint_states
                    and any((c['sec'], c['nanosec']) > (clocks[0]['sec'], clocks[0]['nanosec'])
                            for c in clocks[1:])):
                return
            self.stop.wait(0.02)
        raise RuntimeError('No advancing Gazebo /clock and /joint_states; motion prohibited')

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2.0)
        if self.thread.is_alive():
            raise RuntimeError('ROS observer failed to stop')
        self.node.destroy_node()
        if self.owns_context:
            self.rclpy.shutdown()
        self.stream.close()
        if self.error:
            raise RuntimeError('ROS observation failed: ' + self.error)
