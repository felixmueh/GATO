"""CPU call-path checks for the single-controller manual square launcher."""
import builtins
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from gato_tiago import live_example
from gato_tiago.final_approach import FinalApproachSwitch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tiago_examples/physical_arm/square_forward.py"


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(monkeypatch, tmp_path, *, sea=False, architecture="120"):
    script = load_file("square_forward_test", SCRIPT)
    example = load_file("square_forward_example_test", SCRIPT.parents[1] / "tiago_reach_target.py")
    script.RAW_OUTPUT_ROOT = tmp_path
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), *(["--ros-sea-absolute"] if sea else [])])
    events = []
    receipt = {"config": {"cuda_architecture": architecture}}
    monkeypatch.setattr(live_example, "check_live_backend",
                        lambda backend, **kw: events.append(("backend", backend, kw)))
    monkeypatch.setitem(sys.modules, "native_build", NS(
        verified_receipt=lambda profile: (events.append(("verify", profile)) or receipt),
        load_profile=lambda profile: events.append(("load", profile))))
    monkeypatch.setitem(sys.modules, "tiago_reach_target", example)
    return script, example, events, receipt


def test_preview_has_no_ros_native_or_numeric_imports(tmp_path):
    # Block dependencies explicitly so this verifies preview remains standard-library only.
    driver = """
import builtins, runpy, sys
original_import = builtins.__import__
def checked(name, *args, **kwargs):
    if name.split('.')[0] in {'numpy', 'pinocchio', 'rclpy', 'bsqp', 'native_build', 'gato_tiago', 'tiago_reach_target'}:
        raise AssertionError('Preview imported ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = checked
sys.argv = [sys.argv[1], '--preview', '--ros-sea-absolute']
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-c", driver, str(SCRIPT)],
                            cwd=tmp_path, check=True, capture_output=True, text=True)
    settings = json.loads(result.stdout)
    assert settings["BACKEND"] == "gazebo"
    assert settings["PHYSICAL_PATH_AND_WORKSPACE_REVIEWED"] is False
    assert settings["SEA_ABSOLUTE"] is True
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("sea", [False, True])
def test_launcher_resolves_exact_four_pose_task_in_one_call(monkeypatch, tmp_path, sea):
    script, example, events, receipt = prepare(monkeypatch, tmp_path, sea=sea)
    calls = []
    monkeypatch.setattr(example, "run_experiment", lambda args, **kw: calls.append((args, kw)))
    script.main()
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert events == [("backend", "gazebo", {"physical_reviewed": False}),
                      ("verify", "reaching"), ("load", "reaching")]
    assert args.native_build is receipt
    assert args.ros_tiago and args.ros_record_commands and args.ros_confirm_before_motion
    assert args.tiago_start_config == "comfortable_high_clearance"
    assert args.N == 32 and args.dt == .03 and args.sim_dt == .003
    assert args.goal_timeout == 15.0 and args.goal_threshold == .01
    assert args.goal_axis_threshold_deg == 10.0
    assert args.velocity_threshold == .2 and args.goal_dwell_time == .2
    assert args.ros_sea_absolute is sea
    assert args.ros_joint_states_topic == "/joint_states"
    assert args.ros_sea_state_topic == script.SEA_STATE_TOPIC
    assert args.ros_sea_velocity_filter_tau == .01
    assert args.ros_target_hz == 125.0
    assert args.ros_reset_position_tolerance == .05
    assert args.ros_reset_speed_l1 == .1 and args.ros_reset_arrival_timeout == 5.0
    assert args.ros_max_abs_torque == 26.0 and args.ros_clamp_torque is False
    assert args.ros_collision_min_distance == .04 and args.ros_disable_collision_safety is False
    assert args.ros_joint_position_margin_rad == 0 and args.ros_joint_velocity_scale == 1
    expected_goals = np.array([[.60, -.40, -.20, np.pi/2, 0, np.pi/2],
                               [.60, -.40, -.10, np.pi/2, 0, np.pi/2],
                               [.60, -.30, -.10, np.pi/2, 0, np.pi/2],
                               [.60, -.30, -.20, np.pi/2, 0, np.pi/2]])
    np.testing.assert_allclose(kwargs["goals"], expected_goals)
    # Verify full rotation, including axial roll; the columns are [+Y,+Z,+X].
    for goal in kwargs["goals"]:
        np.testing.assert_allclose(example.pin.rpy.rpyToMatrix(*goal[3:]),
                                   [[0, 0, 1], [1, 0, 0], [0, 1, 0]], atol=1e-14)
    approach = kwargs["solver_params"]
    assert approach == {"max_sqp_iters": 5, "kkt_tol": .001, "max_pcg_iters": 120,
                        "pcg_tol": .001, "solve_ratio": 1.0, "mu": 10.0,
                        "q_cost": 4.0, "N_cost": 80.0, "qd_cost": .05, "u_cost": .0001,
                        "ee_orient_cost": .1, "ee_orient_N_cost": 2.0,
                        "q_lim_cost": .02125, "vel_lim_cost": 1.275, "ctrl_lim_cost": .02,
                        "rho": .01}
    final = example.final_approach_kwargs(args)
    switch = FinalApproachSwitch(approach, enabled=final["final_approach"],
                                params=final["final_approach_params"],
                                trigger=final["final_approach_trigger"])
    assert {**approach, **switch.after} == {**approach, "q_cost": 800.0, "N_cost": 2400.0,
                            "qd_cost": 1.0, "u_cost": .0003, "ee_orient_cost": 10.0,
                            "ee_orient_N_cost": 10.0, "max_sqp_iters": 20, "mu": 30.0}
    assert switch.trigger == {"distance_m": .075, "max_joint_speed_rad_s": .1}
    assert args.final_approach_params.parent == args.expr_dir
    saved = json.loads((args.expr_dir / "launcher_config.json").read_text())
    assert saved == args.square_forward_config
    assert saved["ROUGH_POSITION_TOLERANCE_M"] == .05
    assert saved["ROUGH_TOOL_AXIS_TOLERANCE_DEG"] == 30.0
    assert saved["ROUGH_SPEED_L1_RAD_S"] == 1.0 and saved["ROUGH_DWELL_S"] == 0.0
    assert args.manual_run_index == tmp_path / "gazebo/manual-run-index.jsonl"


def test_single_controller_loop_confirmation_and_partial_recording(monkeypatch, tmp_path):
    script, example, events, receipt = prepare(monkeypatch, tmp_path)
    from gato_tiago import tiago_controller_process, tiago_mpc_controller
    constructed, ros_options, loops = [], [], []
    partial = {"goal_outcomes": ["reached", "timeout", "not_reached", "not_reached"]}
    failure = RuntimeError("guard rejection")

    class MPC:
        def __init__(self, **kwargs):
            constructed.append(kwargs)
            self.last_task_stats = partial

        def run_mpc_goals(self, **kwargs):
            events.append("loop")
            loops.append(kwargs)
            raise failure

    monkeypatch.setattr(tiago_mpc_controller, "MPC_GATO", MPC)
    monkeypatch.setattr(tiago_controller_process, "TiagoControllerOrchestrator",
                        lambda **kwargs: (ros_options.append(kwargs) or NS()))
    monkeypatch.setattr(example, "load_model", lambda: NS(nv=7, createData=lambda: None))
    monkeypatch.setattr(example, "tool_position", lambda *args: np.zeros(3))
    monkeypatch.setattr(example, "preflight", lambda *args, **kwargs: {})
    monkeypatch.setattr(builtins, "input", lambda prompt: (events.append("MOVE") or "MOVE"))

    def finish(controller, output, stats, error, timeout, **kwargs):
        events.append("finish")
        assert stats is partial and error is failure
        assert kwargs["context"]["args"]["native_build"] is receipt
        assert kwargs["context"]["args"]["square_forward_config"]["GOAL_NAMES"] == script.GOAL_NAMES
        return {}

    monkeypatch.setattr(live_example, "finish_live_recording", finish)
    with pytest.raises(RuntimeError, match="guard rejection"):
        script.main()
    assert len(constructed) == len(ros_options) == len(loops) == 1
    assert events[-3:] == ["MOVE", "loop", "finish"]
    assert constructed[0]["batch_size"] == 1 and constructed[0]["N"] == 32
    assert constructed[0]["solver_params"] == script.APPROACH_SOLVER_PARAMS  # no rectangle ctrl=.05
    assert constructed[0]["final_approach_params"] == script.FINAL_APPROACH_PARAMS
    loop = loops[0]
    np.testing.assert_allclose(loop["goals"], script.GOALS_POSES_M_RAD)
    np.testing.assert_allclose(loop["x_start"], script.START_Q_RAD + script.START_QD_RAD_S)
    assert loop["goal_threshold"] == .01 and loop["velocity_threshold"] == .2
    assert loop["goal_dwell_time"] == .2 and loop["goal_axis_threshold"] == np.deg2rad(10)
    np.testing.assert_allclose(ros_options[0]["reset_q"], script.START_Q_RAD)
    assert ros_options[0]["record_command_events"] is True
    assert ros_options[0]["clamp_torque"] is False


def test_wrong_native_architecture_holds_before_call_or_output(monkeypatch, tmp_path):
    script, example, events, _ = prepare(monkeypatch, tmp_path, architecture="87")
    monkeypatch.setattr(example, "run_experiment", lambda *a, **kw: pytest.fail("must hold"))
    with pytest.raises(RuntimeError, match="selected backend"):
        script.main()
    assert ("load", "reaching") not in events
    assert not list(tmp_path.iterdir())
