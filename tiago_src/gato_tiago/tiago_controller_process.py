"""Simple subprocess controller for TIAGo right-arm torque experiments.

The orchestrator side intentionally does not import ROS. The child process owns
all ROS interaction through :class:`gato_tiago.ros_tiago.TiagoRightArmClient`.
"""

from __future__ import annotations

import atexit
from dataclasses import dataclass
import multiprocessing as mp
from pathlib import Path
import queue
import time
from typing import Any, Sequence

import numpy as np

from gato_tiago.config import TIAGO_RIGHT_START_CONFIGS


@dataclass(frozen=True)
class RobotState:
    q: np.ndarray
    qd: np.ndarray
    stamp_sec: float
    received_monotonic_sec: float
    controller_mode: str
    command_count: int = 0
    command_rate_hz: float = 0.0
    max_period_sec: float = 0.0
    seq: int = 0
    source_seq: int = 0
    error: str | None = None

    @property
    def age_sec(self) -> float:
        return time.monotonic() - self.received_monotonic_sec


@dataclass(frozen=True)
class TorqueTrajectory:
    torques: np.ndarray
    dt: float
    start_monotonic_sec: float | None = None


@dataclass(frozen=True)
class ControllerStatus:
    mode: str
    error: str | None = None


@dataclass(frozen=True)
class StateHistorySample:
    source_seq: int
    stamp_sec: float
    received_monotonic_sec: float
    controller_mode: str
    q: np.ndarray
    qd: np.ndarray


def elapsed_sim_time_from_stamp(
    stamp_sec: float,
    origin_stamp_sec: float,
    previous_elapsed_sec: float | None = None,
    *,
    backward_tolerance_sec: float = 1e-9,
) -> float:
    """Return elapsed simulation time from a ROS message stamp."""
    stamp = float(stamp_sec)
    origin = float(origin_stamp_sec)
    elapsed = stamp - origin
    if not np.isfinite(elapsed):
        raise ValueError(
            f"non-finite simulation timestamp: stamp={stamp_sec!r} origin={origin_stamp_sec!r}"
        )
    if elapsed < 0.0:
        if abs(elapsed) <= backward_tolerance_sec:
            elapsed = 0.0
        else:
            raise RuntimeError(
                "robot state simulation timestamp is before the initial stamp: "
                f"stamp={stamp:.9f}s origin={origin:.9f}s"
            )
    if previous_elapsed_sec is not None:
        previous = float(previous_elapsed_sec)
        if elapsed < previous:
            if previous - elapsed <= backward_tolerance_sec:
                return previous
            raise RuntimeError(
                "robot state simulation timestamp moved backward: "
                f"previous_elapsed={previous:.9f}s current_elapsed={elapsed:.9f}s"
            )
    return elapsed


class _SharedState:
    def __init__(self, ctx: mp.context.BaseContext) -> None:
        self.lock = ctx.Lock()
        self.q = ctx.Array("d", 7)
        self.qd = ctx.Array("d", 7)
        self.stamp_sec = ctx.Value("d", 0.0)
        self.received_monotonic_sec = ctx.Value("d", 0.0)
        self.mode = ctx.Array("c", 32)
        self.command_count = ctx.Value("q", 0)
        self.command_rate_hz = ctx.Value("d", 0.0)
        self.max_period_sec = ctx.Value("d", 0.0)
        self.seq = ctx.Value("q", 0)
        self.source_seq = ctx.Value("q", 0)


def _put_latest(q: mp.Queue, item: Any) -> None:
    while True:
        try:
            q.put_nowait(item)
            return
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                time.sleep(0.001)


def _get_latest(q: mp.Queue) -> Any | None:
    latest = None
    while True:
        try:
            latest = q.get_nowait()
        except queue.Empty:
            return latest


def _write_state(
    shared: _SharedState,
    *,
    q: np.ndarray,
    qd: np.ndarray,
    stamp_sec: float,
    received_monotonic_sec: float,
    mode: str,
    command_count: int,
    command_rate_hz: float,
    max_period_sec: float,
    source_seq: int,
) -> None:
    encoded_mode = mode.encode("ascii", errors="replace")[:31]
    with shared.lock:
        shared.q[:] = [float(v) for v in q]
        shared.qd[:] = [float(v) for v in qd]
        shared.stamp_sec.value = float(stamp_sec)
        shared.received_monotonic_sec.value = float(received_monotonic_sec)
        shared.mode[:] = b"\0" * len(shared.mode)
        shared.mode[: len(encoded_mode)] = encoded_mode
        shared.command_count.value = int(command_count)
        shared.command_rate_hz.value = float(command_rate_hz)
        shared.max_period_sec.value = float(max_period_sec)
        shared.source_seq.value = int(source_seq)
        shared.seq.value += 1


def _read_state(shared: _SharedState) -> RobotState | None:
    with shared.lock:
        seq = int(shared.seq.value)
        if seq == 0:
            return None
        mode = bytes(shared.mode[:]).split(b"\0", 1)[0].decode("ascii")
        return RobotState(
            q=np.asarray(shared.q[:], dtype=np.float64),
            qd=np.asarray(shared.qd[:], dtype=np.float64),
            stamp_sec=float(shared.stamp_sec.value),
            received_monotonic_sec=float(shared.received_monotonic_sec.value),
            controller_mode=mode,
            command_count=int(shared.command_count.value),
            command_rate_hz=float(shared.command_rate_hz.value),
            max_period_sec=float(shared.max_period_sec.value),
            seq=seq,
            source_seq=int(shared.source_seq.value),
        )


def _validate_trajectory(
    traj: TorqueTrajectory,
    n_joints: int,
    max_abs_torque: float,
    clamp_torque: bool,
) -> np.ndarray:
    torques = np.asarray(traj.torques, dtype=np.float64)
    if torques.ndim != 2 or torques.shape[1] != n_joints:
        raise ValueError(
            f"torques must have shape (N, {n_joints}), got {torques.shape}"
        )
    if torques.shape[0] < 1:
        raise ValueError("torques must contain at least one row")
    if not np.isfinite(torques).all():
        raise ValueError("torques contain non-finite values")
    if traj.dt <= 0.0 or not np.isfinite(traj.dt):
        raise ValueError("trajectory dt must be positive and finite")
    max_abs_observed = float(np.max(np.abs(torques)))
    if max_abs_observed > max_abs_torque and not clamp_torque:
        raise ValueError(
            f"trajectory torque exceeds max_abs_torque={max_abs_torque:.3f}"
        )
    if max_abs_observed > max_abs_torque:
        print(
            "clamping torque trajectory: "
            f"max_abs={max_abs_observed:.3f} max_abs_torque={max_abs_torque:.3f}",
            flush=True,
        )
        torques = np.clip(torques, -max_abs_torque, max_abs_torque)
    return torques


def _sample_trajectory(
    torques: np.ndarray,
    dt: float,
    start_monotonic_sec: float,
    now: float,
) -> tuple[np.ndarray, bool]:
    idx = int((now - start_monotonic_sec) / dt)
    if idx < 0:
        idx = 0
    if idx >= torques.shape[0]:
        return torques[-1], False
    return torques[idx], True


class TiagoControllerOrchestrator:
    """Minimal process wrapper with initialize/send_trajectory/read_state."""

    def __init__(
        self,
        *,
        target_hz: float = 100.0,
        reset_q: Sequence[float] | None = None,
        reset_duration_sec: float = 2.0,
        stale_timeout_sec: float = 0.25,
        # TODO: read from urdf
        max_abs_torque: float = 30.0,
        clamp_torque: bool = False,
        restore_on_exit: bool = True,
        start_method: str = "spawn",
    ) -> None:
        self.target_hz = float(target_hz)
        if reset_q is None:
            reset_q = TIAGO_RIGHT_START_CONFIGS["comfortable"]
        self.reset_q = np.asarray(reset_q, dtype=np.float64)
        self.reset_duration_sec = float(reset_duration_sec)
        self.stale_timeout_sec = float(stale_timeout_sec)
        self.max_abs_torque = float(max_abs_torque)
        self.clamp_torque = bool(clamp_torque)
        self.restore_on_exit = restore_on_exit
        if self.target_hz <= 0.0:
            raise ValueError("target_hz must be positive")
        if self.reset_q.shape != (7,):
            raise ValueError(f"reset_q must have shape (7,), got {self.reset_q.shape}")
        if self.reset_duration_sec <= 0.0:
            raise ValueError("reset_duration_sec must be positive")
        if self.stale_timeout_sec <= 0.0:
            raise ValueError("stale_timeout_sec must be positive")
        if self.max_abs_torque <= 0.0:
            raise ValueError("max_abs_torque must be positive")
        self._ctx = mp.get_context(start_method)
        self._state = _SharedState(self._ctx)
        self._trajectory_q: mp.Queue = self._ctx.Queue(maxsize=1)
        self._status_q: mp.Queue = self._ctx.Queue(maxsize=8)
        self._history_q: mp.Queue = self._ctx.Queue(maxsize=1)
        self._stop_event = self._ctx.Event()
        self._process: mp.Process | None = None
        self._latest_state: RobotState | None = None
        self._state_history: list[StateHistorySample] = []
        self._closed = False
        self._cleanup_registered = False

    def initialize(self, timeout_sec: float = 10.0) -> None:
        if self._process is not None and self._process.is_alive():
            return
        self._closed = False
        from gato_tiago.ros_tiago import ensure_ros_environment

        ensure_ros_environment(allow_reexec=False)
        self._process = self._ctx.Process(
            target=_controller_main,
            kwargs={
                "target_hz": self.target_hz,
                "reset_q": self.reset_q,
                "reset_duration_sec": self.reset_duration_sec,
                "stale_timeout_sec": self.stale_timeout_sec,
                "max_abs_torque": self.max_abs_torque,
                "clamp_torque": self.clamp_torque,
                "restore_on_exit": self.restore_on_exit,
                "shared_state": self._state,
                "trajectory_q": self._trajectory_q,
                "status_q": self._status_q,
                "history_q": self._history_q,
                "stop_event": self._stop_event,
            },
        )
        self._process.start()
        if not self._cleanup_registered:
            atexit.register(self.close)
            self._cleanup_registered = True

        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            status = self._latest_status()
            if status is not None and status.mode == "READY":
                return
            self._raise_if_failed(status)
            time.sleep(0.05)
        raise TimeoutError("controller process did not become ready")

    def _latest_status(self) -> ControllerStatus | None:
        return _get_latest(self._status_q)

    def _raise_if_failed(self, status: ControllerStatus | None = None) -> None:
        if status is None:
            status = self._latest_status()
        if status is not None and status.mode.startswith("ERROR"):
            raise RuntimeError(status.error or status.mode)
        if self._process is not None and self._process.exitcode is not None:
            raise RuntimeError(f"controller process exited: {self._process.exitcode}")

    def send_trajectory(
        self,
        torques: Sequence[Sequence[float]],
        dt: float,
        *,
        start_monotonic_sec: float | None = None,
    ) -> None:
        self._raise_if_failed()
        if self._process is None or not self._process.is_alive():
            raise RuntimeError("controller process is not running")
        traj = TorqueTrajectory(
            torques=np.asarray(torques, dtype=np.float64),
            dt=float(dt),
            start_monotonic_sec=start_monotonic_sec,
        )
        _put_latest(self._trajectory_q, traj)

    def read_state(self, timeout_sec: float = 0.0) -> RobotState | None:
        deadline = time.monotonic() + timeout_sec
        while True:
            self._raise_if_failed()
            latest = _read_state(self._state)
            if latest is not None:
                self._latest_state = latest
                return latest
            if timeout_sec <= 0.0 or time.monotonic() >= deadline:
                return self._latest_state
            time.sleep(0.01)

    def state_history(self) -> list[StateHistorySample]:
        self._drain_state_history()
        return list(self._state_history)

    def state_history_frequency_summary(
        self,
        *,
        change_atol: float = 0.0,
        running_only: bool = True,
    ) -> dict[str, float | int | bool | None]:
        return summarize_state_history_frequency(
            self.state_history(),
            change_atol=change_atol,
            running_only=running_only,
        )

    def write_state_history_csv(
        self,
        path: str | Path,
        *,
        change_atol: float = 0.0,
    ) -> dict[str, float | int | bool | None]:
        history = self.state_history()
        rows = _state_history_rows(history)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = (
            "source_seq,stamp_sec,received_monotonic_sec,controller_mode_code,"
            "q0,q1,q2,q3,q4,q5,q6,qd0,qd1,qd2,qd3,qd4,qd5,qd6"
        )
        np.savetxt(path, rows, delimiter=",", header=header, comments="", fmt="%.10g")
        return summarize_state_history_frequency(
            history,
            change_atol=change_atol,
            running_only=True,
        )

    def _drain_state_history(self) -> None:
        latest = _get_latest(self._history_q)
        if latest is not None:
            self._state_history = list(latest)

    def close(self, timeout_sec: float = 5.0) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        if self._process is not None:
            self._process.join(timeout=timeout_sec)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)
        self._drain_state_history()

    def __enter__(self) -> "TiagoControllerOrchestrator":
        self.initialize()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def _controller_main(
    *,
    target_hz: float,
    reset_q: np.ndarray,
    reset_duration_sec: float,
    stale_timeout_sec: float,
    max_abs_torque: float,
    clamp_torque: bool,
    restore_on_exit: bool,
    shared_state: _SharedState,
    trajectory_q: mp.Queue,
    status_q: mp.Queue,
    history_q: mp.Queue,
    stop_event: mp.Event,
) -> None:
    from gato_tiago.ros_tiago import TiagoRightArmClient

    period = 1.0 / target_hz
    mode = "INIT"
    current_torques: np.ndarray | None = None
    current_dt = 0.0
    current_start = 0.0
    effort_active = False
    had_error = False
    command_count = 0
    command_window_start = time.perf_counter()
    last_publish_time: float | None = None
    command_rate_hz = 0.0
    max_period_sec = 0.0
    state_history: list[StateHistorySample] = []
    last_history_source_seq = 0

    def set_status(new_mode: str, error: str | None = None) -> None:
        nonlocal mode
        mode = new_mode
        _put_latest(status_q, ControllerStatus(mode=new_mode, error=error))

    with TiagoRightArmClient(node_name="gato_tiago_controller_process") as arm:
        try:
            set_status("RESETTING")
            arm.switch_to_default_control(timeout_sec=5.0)
            arm.publish_zero_base_velocity()
            arm.publish_position_trajectory(reset_q, duration_sec=reset_duration_sec)
            reset_deadline = time.monotonic() + reset_duration_sec
            while time.monotonic() < reset_deadline and not stop_event.is_set():
                arm.publish_zero_base_velocity()
                arm.spin_once(timeout_sec=0.05)
            arm.read_state(timeout_sec=8.0)
            arm.configure_runtime_effort_controller(timeout_sec=5.0)
            set_status("READY")

            next_time = time.perf_counter()
            while not stop_event.is_set():
                now = time.perf_counter()
                arm.spin_once(timeout_sec=0.0)
                arm.publish_zero_base_velocity()

                state = arm.latest_state()
                if state is not None:
                    _write_state(
                        shared_state,
                        q=state.q,
                        qd=state.qd,
                        stamp_sec=state.stamp_sec,
                        received_monotonic_sec=state.received_monotonic_sec,
                        mode=mode,
                        command_count=command_count,
                        command_rate_hz=command_rate_hz,
                        max_period_sec=max_period_sec,
                        source_seq=state.seq,
                    )
                    if state.seq != last_history_source_seq:
                        state_history.append(
                            StateHistorySample(
                                source_seq=state.seq,
                                stamp_sec=state.stamp_sec,
                                received_monotonic_sec=state.received_monotonic_sec,
                                controller_mode=mode,
                                q=state.q.astype(np.float64).copy(),
                                qd=state.qd.astype(np.float64).copy(),
                            )
                        )
                        last_history_source_seq = state.seq

                if effort_active and (state is None or state.age_sec > stale_timeout_sec):
                    age = state.age_sec if state is not None else float("inf")
                    raise RuntimeError(f"joint state stale during effort control: age={age:.3f}s")

                traj = _get_latest(trajectory_q)
                if traj is not None:
                    current_torques = _validate_trajectory(
                        traj,
                        len(arm.joint_names),
                        max_abs_torque,
                        clamp_torque,
                    )
                    current_dt = traj.dt
                    current_start = (
                        traj.start_monotonic_sec
                        if traj.start_monotonic_sec is not None
                        else now
                    )
                    command_count = 0
                    command_window_start = now
                    last_publish_time = None
                    command_rate_hz = 0.0
                    max_period_sec = 0.0
                    if not effort_active:
                        arm.publish_effort(current_torques[0])
                        arm.spin_once(timeout_sec=0.01)
                        arm.switch_to_effort_control(timeout_sec=5.0)
                        effort_active = True
                    set_status("RUNNING")

                if effort_active:
                    if (
                        current_torques is None
                        or now
                        > current_start
                        + current_dt * (current_torques.shape[0] - 1)
                        + stale_timeout_sec
                    ):
                        raise RuntimeError("torque trajectory stale")
                    tau, still_in_horizon = _sample_trajectory(
                        current_torques,
                        current_dt,
                        current_start,
                        now,
                    )
                    arm.publish_effort(tau)
                    command_count += 1
                    if last_publish_time is not None:
                        max_period_sec = max(max_period_sec, now - last_publish_time)
                    last_publish_time = now
                    window_elapsed = now - command_window_start
                    if window_elapsed >= 0.25:
                        command_rate_hz = command_count / window_elapsed

                next_time += period
                sleep_time = next_time - time.perf_counter()
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
                else:
                    next_time = time.perf_counter()

        except BaseException as exc:
            had_error = True
            set_status("ERROR_RESTORE", str(exc))
            if restore_on_exit:
                try:
                    arm.switch_to_default_control(timeout_sec=5.0)
                except BaseException as restore_exc:
                    set_status("ERROR_RESTORE_FAILED", str(restore_exc))
                    raise
            raise
        finally:
            _put_latest(history_q, list(state_history))
            if restore_on_exit:
                try:
                    arm.switch_to_default_control(timeout_sec=5.0)
                    if not had_error:
                        set_status("RESTORED")
                except BaseException as exc:
                    set_status("ERROR_RESTORE_FAILED", str(exc))


# =======================================================
# State history helpers
# =======================================================


def _state_history_rows(history: Sequence[StateHistorySample]) -> np.ndarray:
    rows = []
    for sample in history:
        mode_code = {
            "READY": 1.0,
            "RUNNING": 2.0,
            "RESTORED": 3.0,
        }.get(sample.controller_mode, 0.0)
        rows.append(
            [
                float(sample.source_seq),
                float(sample.stamp_sec),
                float(sample.received_monotonic_sec),
                mode_code,
                *[float(v) for v in sample.q],
                *[float(v) for v in sample.qd],
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def summarize_state_history_frequency(
    history: Sequence[StateHistorySample],
    *,
    change_atol: float = 0.0,
    running_only: bool = True,
) -> dict[str, float | int | bool | None]:
    samples = list(history)
    if running_only:
        samples = [sample for sample in samples if sample.controller_mode == "RUNNING"]
    if len(samples) < 2:
        return {
            "running_only": bool(running_only),
            "samples": len(samples),
            "changed_samples": 0,
            "changed_transitions": 0,
            "lower_bound_hz": None,
            "elapsed_stamp_sec": None,
            "elapsed_wall_receive_sec": None,
        }

    values = np.asarray(
        [np.concatenate([sample.q, sample.qd]).astype(np.float64) for sample in samples],
        dtype=np.float64,
    )
    deltas = np.abs(np.diff(values, axis=0))
    changed = np.any(deltas > change_atol, axis=1)
    changed_transitions = np.flatnonzero(changed)
    if changed_transitions.size == 0:
        return {
            "running_only": bool(running_only),
            "samples": len(samples),
            "changed_samples": 0,
            "changed_transitions": 0,
            "lower_bound_hz": 0.0,
            "elapsed_stamp_sec": 0.0,
            "elapsed_wall_receive_sec": 0.0,
        }

    first = samples[int(changed_transitions[0])]
    last = samples[int(changed_transitions[-1] + 1)]
    changed_transition_count = int(changed_transitions.size)
    elapsed_stamp = float(last.stamp_sec - first.stamp_sec)
    elapsed_wall = float(last.received_monotonic_sec - first.received_monotonic_sec)
    if elapsed_stamp > 0.0:
        lower_bound_hz = float(changed_transition_count / elapsed_stamp)
    elif elapsed_wall > 0.0:
        lower_bound_hz = float(changed_transition_count / elapsed_wall)
    else:
        lower_bound_hz = None
    return {
        "running_only": bool(running_only),
        "samples": len(samples),
        "changed_samples": changed_transition_count + 1,
        "changed_transitions": changed_transition_count,
        "lower_bound_hz": lower_bound_hz,
        "elapsed_stamp_sec": elapsed_stamp,
        "elapsed_wall_receive_sec": elapsed_wall,
        "change_atol": float(change_atol),
    }
