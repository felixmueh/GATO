"""End-of-run lifecycle tests; no ROS, GPU or robot required."""

import queue
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from gato_tiago import tiago_controller_process as execution


@pytest.mark.parametrize("experiment", ["fig8", "goals"])
def test_tracking_restores_control_before_slow_result_saving(monkeypatch, experiment):
    # Load the real MPC loop without its unused GPU constructor dependency.
    path = Path(execution.__file__).with_name("tiago_mpc_controller.py")
    spec = importlib.util.spec_from_file_location("shutdown_test_mpc", path)
    module = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as isolated:
        isolated.setitem(sys.modules, "bsqp.interface", SimpleNamespace(BSQP=object))
        spec.loader.exec_module(module)
    mpc = module.MPC_GATO.__new__(module.MPC_GATO)
    mpc.N, mpc.nq, mpc.nx, mpc.nu = 3, 1, 2, 1
    mpc.dt, mpc.batch_size, mpc.track_full_stats = .1, 1, False
    mpc.constant_f_ext_world = np.zeros(6)
    mpc.has_pendulum = False
    mpc.update_force_batch = lambda q: None
    mpc.evaluate_best_trajectory = lambda *args: 0
    mpc.solver = SimpleNamespace(
        reset_dual=lambda: None, reset_rho=lambda: None,
        solve=lambda x, ref, plan: (plan.copy(), 100.),
        ee_pos=lambda q: np.zeros(3),
        ee_tool_axis_error=lambda q, goal: 0.,
    )
    clock = SimpleNamespace(now=0., restored=False)
    class Controller:
        target_hz = 10.
        def initialize(self): pass
        def read_state(self, **kwargs):
            clock.now += .1
            return SimpleNamespace(q=np.zeros(1), qd=np.zeros(1), stamp_sec=clock.now,
                                   age_sec=0., command_rate_hz=10., max_period_sec=.1)
        def send_trajectory(self, *args): pass
        def finish_control(self, **kwargs): clock.restored = True

    if experiment == "fig8":
        _, stats = mpc.run_mpc_fig8(np.zeros(2), np.zeros(600), sim_time=.2,
                                   controller=Controller())
    else:
        _, stats = mpc.run_mpc_goals(np.zeros(2), [[1., 0., 0.]], goal_timeout=.2,
                                    controller=Controller())
    # Model result writing taking longer than a horizon plus watchdog grace.
    clock.now += 1.
    assert clock.restored, "controller still executing expired torques during result saving"
    assert len(stats["timestamps"]) > 0


def orchestrator_with_exited_child(tmp_path, *, exitcode, status):
    controller = execution.TiagoControllerOrchestrator()
    controller._status_q = queue.Queue()
    if status is not None:
        controller._status_q.put(status)
    controller._process = SimpleNamespace(
        join=lambda **kwargs: None, is_alive=lambda: False, exitcode=exitcode)
    return controller


@pytest.mark.parametrize("exitcode,status,match", [
    (1, execution.ControllerStatus("ERROR_RESTORE", "torque trajectory stale"), "torque trajectory stale"),
    (0, execution.ControllerStatus("ERROR_RESTORE_FAILED", "switch rejected"), "switch rejected"),
    (1, None, "exited"),
    (0, None, "restoration"),
])
def test_close_reports_failed_or_unconfirmed_shutdown(tmp_path, exitcode, status, match):
    controller = orchestrator_with_exited_child(tmp_path, exitcode=exitcode, status=status)
    with pytest.raises(RuntimeError, match=match):
        controller.close()
    assert controller._stop_event.is_set()
    controller.close()  # Cleanup is idempotent after the first reported failure.


def test_close_accepts_confirmed_restoration(tmp_path):
    controller = orchestrator_with_exited_child(
        tmp_path, exitcode=0, status=execution.ControllerStatus("RESTORED"))
    controller.close()
    controller.close()


def test_close_reports_forced_termination(tmp_path):
    controller = execution.TiagoControllerOrchestrator()
    calls = []
    controller._process = SimpleNamespace(
        join=lambda **kwargs: None, is_alive=lambda: True, exitcode=None,
        terminate=lambda: calls.append("terminate"))
    with pytest.raises(TimeoutError, match="shutdown"):
        controller.close(timeout_sec=.01)
    assert calls == ["terminate"]


@pytest.mark.parametrize("finish,restore_fails", [(True, False), (False, False), (True, True)])
def test_worker_finish_restores_before_idle_and_supports_next_batch(
        monkeypatch, tmp_path, capsys, finish, restore_fails):
    from gato_tiago import ros_tiago as ros
    from gato_tiago.safety_monitor import CollisionSafetySettings

    clock = SimpleNamespace(now=0.)
    monkeypatch.setattr(execution, "time", SimpleNamespace(
        monotonic=lambda: clock.now, perf_counter=lambda: clock.now,
        sleep=lambda dt: setattr(clock, "now", clock.now + max(dt, .001))))
    trajectories, statuses = queue.Queue(), queue.Queue()
    calls, acknowledgements, observed_statuses = [], [], []
    state = SimpleNamespace(q=np.zeros(7), qd=np.zeros(7), stamp_sec=1.,
        received_monotonic_sec=0., seq=1, age_sec=0.,
        joint_positions={n: 0. for n in ros.RIGHT_ARM_JOINTS},
        joint_velocities={n: 0. for n in ros.RIGHT_ARM_JOINTS})
    arm_state = SimpleNamespace(active=False, batches=0, restored_at=None)

    class Arm:
        joint_names = ros.RIGHT_ARM_JOINTS
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def spin_once(self, timeout_sec=0):
            clock.now += max(timeout_sec, .001)
            status = execution._get_latest(statuses)
            if status is not None:
                observed_statuses.append(status)
            if status and status.finished_request_id is not None:
                assert not arm_state.active  # Ack must follow successful restoration.
                acknowledgements.append(status.finished_request_id)
            if (arm_state.restored_at is not None and arm_state.batches == 1
                    and clock.now - arm_state.restored_at > 1.):
                # Idle longer than the .13 s horizon expiry, then reuse the worker.
                arm_state.restored_at = None
                trajectories.put(execution.TorqueTrajectory(np.ones((4, 7)), .01))
        def read_state(self, **kwargs): return state
        def latest_state(self): return state
        def switch_to_default_control(self, **kwargs):
            if arm_state.active and restore_fails:
                raise RuntimeError("switch rejected")
            if arm_state.active:
                arm_state.restored_at = clock.now
            arm_state.active = False
            calls.append("restore")
        def publish_position_trajectory(self, *args, **kwargs): pass
        def configure_runtime_effort_controller(self, **kwargs):
            trajectories.put(execution.TorqueTrajectory(np.ones((4, 7)), .01))
        def switch_to_effort_control(self, **kwargs):
            arm_state.active = True
            arm_state.batches += 1
            calls.append("activate")
            if finish:
                trajectories.put(execution.FinishControl(arm_state.batches))
        def publish_effort(self, torques):
            calls.append("zero" if np.all(np.asarray(torques) == 0) else "effort")

    monkeypatch.setattr(ros, "TiagoRightArmClient", Arm)
    def run():
        execution._controller_main(
            target_hz=100, reset_q=np.zeros(7), reset_duration_sec=.01,
            stale_timeout_sec=.1, max_abs_torque=26, clamp_torque=True,
            collision_safety=CollisionSafetySettings(enabled=False), history_max_records=20,
            shared_state=execution._SharedState(execution.mp.get_context("spawn")),
            trajectory_q=trajectories, status_q=statuses,
            history_path=tmp_path / "history.csv", history_metadata_path=tmp_path / "meta.json",
            stop_event=SimpleNamespace(is_set=lambda: clock.now > 2.5))

    if not finish or restore_fails:
        with pytest.raises(RuntimeError, match="switch rejected" if restore_fails else "torque trajectory stale"):
            run()
        assert acknowledgements == []
        observed_statuses.append(execution._get_latest(statuses))
        assert any(s is not None and s.mode.startswith("ERROR") for s in observed_statuses)
    else:
        run()
        assert acknowledgements == [1, 2]
        assert calls.count("activate") == 2
        assert calls.count("restore") == 4  # Startup, both finishes, final cleanup.
        assert not arm_state.active
        assert execution._get_latest(statuses).mode == "RESTORED"

    output = capsys.readouterr().err
    stages = ["creating ROS client", "switching to position control",
              "publishing reset trajectory", "waiting for reset duration",
              "reading initial joint state", "configuring GATO effort controller",
              "setup complete"]
    offsets = [output.index(stage) for stage in stages]
    assert offsets == sorted(offsets)
    assert "previous stage took" in output and "max reset error=" in output


@pytest.mark.parametrize("reply", ["current", "old", "error"])
def test_finish_waits_for_matching_restoration(monkeypatch, reply):
    controller = execution.TiagoControllerOrchestrator()
    controller._trajectory_q, controller._status_q = queue.Queue(), queue.Queue()
    controller._process = SimpleNamespace(is_alive=lambda: True, exitcode=None)
    controller._finish_request_id = 1
    clock = SimpleNamespace(now=0.)
    def sleep(dt):
        clock.now += dt
        request = execution._get_latest(controller._trajectory_q)
        if request is not None:
            assert isinstance(request, execution.FinishControl)
            assert request.request_id == 2
            controller._status_q.put(execution.ControllerStatus(
                "ERROR_RESTORE_FAILED" if reply == "error" else "RESTORED",
                error="switch rejected" if reply == "error" else None,
                finished_request_id=2 if reply == "current" else 1))
    monkeypatch.setattr(execution, "time", SimpleNamespace(monotonic=lambda: clock.now, sleep=sleep))
    if reply == "current":
        controller.finish_control(timeout_sec=.1)
    else:
        with pytest.raises((TimeoutError, RuntimeError), match="restoration|switch rejected"):
            controller.finish_control(timeout_sec=.1)


def test_finish_does_not_discard_ack_arriving_between_polls(monkeypatch):
    controller = execution.TiagoControllerOrchestrator()
    controller._trajectory_q = queue.Queue()
    controller._process = SimpleNamespace(is_alive=lambda: True, exitcode=None)
    replies = iter([None, None, execution.ControllerStatus("RESTORED", finished_request_id=1)])
    monkeypatch.setattr(controller, "_latest_status", lambda: next(replies, None))
    controller.finish_control(timeout_sec=.05)
