"""Small shared recording/cleanup helpers for the manually launched scripts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT / "python", ROOT / "tiago_src"):
    sys.path.insert(0, str(directory))


def json_value(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, default=json_value, indent=2) + "\n")


def digest(value):
    return hashlib.sha256(json.dumps(value, default=json_value, sort_keys=True).encode()).hexdigest()


def public_settings(namespace):
    return {key: value for key, value in namespace.items() if key.isupper() and key != "ROOT"}


def check_backend(backend):
    expected = {"gazebo": ("simulation", "1"), "physical_robot": ("tiago", "2")}
    if backend not in expected:
        raise ValueError("BACKEND must be gazebo or physical_robot")
    profile, domain = expected[backend]
    if (os.environ.get("GATO_ROS_PROFILE"), os.environ.get("ROS_DOMAIN_ID")) != (profile, domain):
        setup = "ros_simulation.sh" if backend == "gazebo" else "ros_tiago.sh"
        raise RuntimeError(f"Source tiago_tools/{setup} in this container first ({profile}, domain {domain}).")
    if os.environ.get("RMW_IMPLEMENTATION") != "rmw_cyclonedds_cpp":
        raise RuntimeError("The existing TIAGo ROS profile requires rmw_cyclonedds_cpp")


def family_identity(cfg, receipt):
    keys = ("BACKEND", "NATIVE_PROFILE", "START_Q_RAD", "N_STATE_KNOTS", "KNOT_DT_S",
            "SOLVER_PARAMS", "CONTROLLER_HZ", "STALE_TIMEOUT_S", "MAX_TORQUE_NM",
            "COLLISION_MARGIN_M", "COLLISION_TIMEOUT_S", "MAX_GEOMETRY_SPEED_M_S",
            "JOINT_POSITION_MARGIN_RAD", "JOINT_VELOCITY_SCALE", "RESET_DURATION_S",
            "RESET_POSITION_TOLERANCE_RAD", "RESET_SPEED_L1_RAD_S", "RESET_ARRIVAL_TIMEOUT_S")
    sources = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in [Path(__file__), ROOT / "python/bsqp/common.py",
                            *sorted((ROOT / "tiago_src/gato_tiago").glob("*.py"))]}
    geometry = {name: hashlib.sha256((ROOT / cfg[name]).read_bytes()).hexdigest()
                for name in ("COLLISION_URDF_PATH", "COLLISION_BLACKLIST_PATH")}
    return digest({"settings": {key: cfg[key] for key in keys}, "native": receipt,
                   "runtime_sources": sources, "geometry": geometry})


class RecordedSolver:
    """Observe the public solve boundary; retain compact per-call work/timing."""
    def __init__(self, solver, output):
        self.delegate = solver
        self.output = output
        self.stream = (output / "solves.jsonl").open("x")
        self.count = 0

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def solve(self, state, reference, seed=None):
        import numpy as np
        actual_state = np.array(state, copy=True)
        seed_copy = np.array(seed if seed is not None else self.delegate.XU_B, copy=True)
        started = time.perf_counter()
        result = None
        error = None
        try:
            result = self.delegate.solve(state, reference, seed)
            elapsed = time.perf_counter() - started
            stats = getattr(self.delegate, "stats", {})
            if not np.isfinite(result[0]).all() or not np.isfinite(stats.get("final_merit", [])).all():
                raise FloatingPointError("Nonfinite returned plan or merit")
            return result
        except BaseException as exc:
            elapsed = time.perf_counter() - started
            error = f"{type(exc).__name__}: {exc}"
            np.savez_compressed(self.output / f"failed_solve_{self.count:06d}.npz",
                                measured_state=actual_state, reference=reference, seed=seed_copy,
                                output=result[0] if result is not None else np.array([]))
            raise
        finally:
            stats = getattr(self.delegate, "stats", {})
            row = {"solve_index": self.count, "started_monotonic_s": started,
                   "wrapper_wall_s": elapsed, "feedback_state": actual_state,
                   "native_time_us": result[1] if result is not None else None,
                   "error": error}
            for key in ("sqp_iters", "pcg_iters", "pcg_times_us", "final_merit", "ls_num_iters"):
                row[key] = stats.get(key)
            self.stream.write(json.dumps(row, default=json_value) + "\n")
            self.stream.flush()
            self.count += 1


def reference(cfg, model):
    """Same horizontal geometry/phase as the retained tracking references."""
    import numpy as np
    import pinocchio as pin
    data = model.createData()
    pin.framesForwardKinematics(model, data, np.array(cfg["START_Q_RAD"]))
    anchor = data.oMf[model.getFrameId("arm_right_tool_link")].translation.copy()
    dt, period = cfg["KNOT_DT_S"], cfg["PERIOD_S"]
    count = int(np.ceil((cfg["DURATION_S"] + (cfg["N_STATE_KNOTS"] - 1) * dt) / dt)) + 1
    # The retained public helper samples floor(period/dt) points per cycle.
    phase = (np.arange(count) % int(period / dt)) * (2 * np.pi / int(period / dt))
    xyzrpy = np.zeros((count, 6))
    xyzrpy[:, :3] = anchor
    xyzrpy[:, 0] += cfg["EDGE_CLEARANCE_M"] + .5 * cfg["X_SPAN_M"] * (1 - np.cos(phase))
    xyzrpy[:, 1] += cfg["Y_AMPLITUDE_M"] * np.sin(2 * phase)
    xyzrpy[:, 3:] = cfg["REFERENCE_RPY_RAD"]
    return xyzrpy.reshape(-1)


def execute(cfg, output, goal=None):
    try:
        return _execute(cfg, output, goal)
    except BaseException as exc:
        # Setup/analysis failures also leave a result in the already-created run.
        result = dict(task_success=False, physical_pass=False, handback_confirmed=False,
                      interrupted=isinstance(exc, KeyboardInterrupt), observations={},
                      errors=[{"stage": "setup_or_finalization", "error": str(exc),
                               "traceback": traceback.format_exc()}])
        write_json(output / "result.json", result)
        return result


def _execute(cfg, output, goal=None):
    import numpy as np
    import pinocchio as pin
    from gato_tiago.tiago_controller_process import TiagoControllerOrchestrator
    from gato_tiago.tiago_mpc_controller import MPC_GATO
    from gato_tiago.safety_monitor import DEFAULT_COLLISION_URDF_PATH

    if (ROOT / cfg["COLLISION_URDF_PATH"]).resolve() != DEFAULT_COLLISION_URDF_PATH.resolve():
        raise ValueError("Use the runtime collision URDF declared by the existing safety monitor")

    model_path = ROOT / "gato/dynamics/tiago_right/tiago_right_arm.urdf"
    model = pin.buildModelFromUrdf(str(model_path))
    controller = TiagoControllerOrchestrator(
        target_hz=cfg["CONTROLLER_HZ"], reset_q=cfg["START_Q_RAD"],
        reset_duration_sec=cfg["RESET_DURATION_S"], stale_timeout_sec=cfg["STALE_TIMEOUT_S"],
        reset_position_tolerance_rad=cfg["RESET_POSITION_TOLERANCE_RAD"],
        reset_velocity_l1_tolerance_rad_s=cfg["RESET_SPEED_L1_RAD_S"],
        reset_arrival_timeout_sec=cfg["RESET_ARRIVAL_TIMEOUT_S"],
        max_abs_torque=cfg["MAX_TORQUE_NM"], clamp_torque=False,
        disable_collision_safety_for_sim_debug=False,
        collision_min_distance_m=cfg["COLLISION_MARGIN_M"],
        collision_check_timeout_sec=cfg["COLLISION_TIMEOUT_S"],
        collision_max_monitored_geometry_speed_m_s=cfg["MAX_GEOMETRY_SPEED_M_S"],
        collision_blacklist_path=ROOT / cfg["COLLISION_BLACKLIST_PATH"],
        joint_position_margin_rad=cfg["JOINT_POSITION_MARGIN_RAD"],
        joint_velocity_scale=cfg["JOINT_VELOCITY_SCALE"],
        history_max_records=cfg["HISTORY_MAX_RECORDS"], start_method="spawn")
    stats, errors, recorder = {}, [], None
    handback = False
    interrupted = False
    try:
        mpc = MPC_GATO(model, str(model_path), N=cfg["N_STATE_KNOTS"], dt=cfg["KNOT_DT_S"],
                       batch_size=1, plant_type="tiago_right", solver_params=cfg["SOLVER_PARAMS"],
                       track_full_stats=True, final_approach=False)
        recorder = RecordedSolver(mpc.solver, output)
        mpc.solver = recorder
        x = np.r_[cfg["START_Q_RAD"], np.zeros(7)].astype(np.float32)
        common = dict(x_start=x, sim_dt=cfg["SIM_DT_S"], controller=controller,
                      controller_timeout=cfg["CONTROLLER_TIMEOUT_S"],
                      controller_startup_timeout=cfg["STARTUP_TIMEOUT_S"])
        if goal is None:
            ref = reference(cfg, model)
            np.save(output / "reference.npy", ref)
            _, stats = mpc.run_mpc_fig8(fig8_traj=ref, sim_time=cfg["DURATION_S"], **common)
        else:
            _, stats = mpc.run_mpc_goals(goals=[np.array(goal, dtype=np.float32)],
                goal_timeout=cfg["GOAL_TIMEOUT_S"], goal_threshold=cfg["POSITION_TOLERANCE_M"],
                goal_axis_threshold=np.deg2rad(cfg["TOOL_AXIS_TOLERANCE_DEG"]),
                velocity_threshold=cfg["ARRIVAL_SPEED_L1_RAD_S"], goal_dwell_time=cfg["DWELL_S"],
                reset_rho_each_solve=False, initial_guess=None, **common)
    except BaseException as exc:
        interrupted = isinstance(exc, KeyboardInterrupt)
        errors.append({"stage": "task", "error": f"{type(exc).__name__}: {exc}",
                       "traceback": traceback.format_exc()})
    finally:
        # Cleanup is attempted before any stats conversion or disk finalization.
        try:
            worker_started = controller._process is not None
            controller.close(timeout_sec=cfg["CONTROLLER_TIMEOUT_S"])
            handback = worker_started
        except BaseException as exc:
            errors.append({"stage": "handback", "error": str(exc)})
        if recorder is not None:
            recorder.stream.close()
        for method, name in (("write_state_history_csv", "controller_state_history.csv"),
                             ("write_full_state_history_jsonl", "full_joint_state_history.jsonl")):
            try:
                result = getattr(controller, method)(output / name)
                if result is not None:
                    write_json(output / "history_summary.json", result)
            except BaseException as exc:
                errors.append({"stage": method, "error": str(exc)})
        metadata_path = getattr(controller, "_history_metadata_path", None)
        try:
            if metadata_path is None or not Path(metadata_path).is_file():
                raise RuntimeError("worker history completeness receipt unavailable")
            (output / "history_metadata.json").write_bytes(Path(metadata_path).read_bytes())
        except BaseException as exc:
            write_json(output / "history_metadata.json", {"history_complete": False, "error": str(exc)})
            errors.append({"stage": "history_metadata", "error": str(exc)})
        write_json(output / "task_stats.json", stats)
    try:
        physical, observed = observed_physical(output, model, cfg)
    except Exception as exc:
        physical, observed = False, {"history_complete": False}
        errors.append({"stage": "observed_physical", "error": str(exc)})
    if goal is None:
        distances = np.asarray(stats.get("goal_distances", []), dtype=float)
        rms = float(np.sqrt(np.mean(distances ** 2))) if len(distances) else None
        maximum = float(np.max(distances)) if len(distances) else None
        success = bool(rms is not None and np.isfinite(distances).all() and
                       rms <= cfg["TRACKING_RMS_GATE_M"] and maximum <= cfg["TRACKING_MAX_GATE_M"])
        observed.update(tracking_feedback_rms_m=rms, tracking_feedback_max_m=maximum)
    else:
        success = stats.get("goal_outcomes") == ["reached"]
    result = dict(task_success=success, physical_pass=physical and not errors,
                  handback_confirmed=handback, errors=errors, interrupted=interrupted,
                  observations=observed, collision_scope="runtime sampled robot-self monitor; offline path review separate")
    write_json(output / "result.json", result)
    return result


def observed_physical(output, model, cfg):
    """Use controller publications/observations, never planner CSV torques."""
    import numpy as np
    rows = [json.loads(line) for line in (output / "full_joint_state_history.jsonl").read_text().splitlines()]
    if not rows:
        return False, {"history_complete": False}
    q, v = np.array([r["q"] for r in rows]), np.array([r["qd"] for r in rows])
    finite = bool(np.isfinite(q).all() and np.isfinite(v).all())
    q_excess = float(max(0, np.max(model.lowerPositionLimit - q), np.max(q - model.upperPositionLimit)))
    v_excess = float(max(0, np.max(np.abs(v) - model.velocityLimit)))
    torques = [r["applied_tau"] for r in rows if r.get("applied_tau") is not None]
    u_max = float(np.max(np.abs(torques))) if torques else None
    faults = [r for r in rows if r.get("safety_fault")]
    summary = json.loads((output / "history_summary.json").read_text())
    metadata = json.loads((output / "history_metadata.json").read_text())
    complete = (metadata.get("dropped_history_records") == 0 and
                metadata.get("record_count") == len(rows))
    physical = finite and complete and q_excess <= 1e-4 and v_excess <= 1e-3 and u_max is not None and np.isfinite(u_max) and u_max <= cfg["MAX_TORQUE_NM"] + 1e-3 and not faults
    return bool(physical), dict(history_complete=complete, recorded_states=len(rows),
        q_excess_rad=q_excess, speed_excess_rad_s=v_excess, max_published_torque_Nm=u_max,
        recorded_safety_faults=len(faults), state_frequency=summary,
        measured_actuator_torque_available=False)


def main(namespace, *, tracking=False):
    cfg = public_settings(namespace)
    parser = argparse.ArgumentParser(description="Review constants in this file before motion.")
    parser.add_argument("--preview", action="store_true", help="Print settings only; no ROS/native load or motion")
    args = parser.parse_args()
    print(json.dumps(cfg, default=json_value, indent=2))
    if args.preview:
        return 0
    check_backend(cfg["BACKEND"])
    from native_build import load_profile
    receipt = load_profile(cfg["NATIVE_PROFILE"])
    if cfg["N_STATE_KNOTS"] != receipt["config"]["knots"]:
        raise ValueError("This prepared native module supports N32; prepare a matching module before changing N")
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    print(json.dumps(dict(source_commit=source_commit, native_module=receipt["module"],
                          native_config=receipt["config"]), indent=2))
    if cfg["BACKEND"] == "physical_robot" and (
            receipt["config"]["cuda_architecture"] != "87" or
            receipt["platform"]["machine"] not in {"aarch64", "arm64"}):
        raise RuntimeError("Physical runs require the reviewed Jetson ARM64/sm87 build")
    raw_root = ROOT / cfg["RAW_OUTPUT_ROOT"] / cfg["BACKEND"]
    raw_root.mkdir(parents=True, exist_ok=True)
    temp_root = raw_root / "_tmp"
    temp_root.mkdir(exist_ok=True)
    tempfile.tempdir = str(temp_root)
    identity = family_identity(cfg, receipt)
    preflight_file = raw_root / f"preflight_{cfg['NATIVE_PROFILE']}.json"
    if not cfg["IS_PREFLIGHT"]:
        preflight = json.loads(preflight_file.read_text()) if preflight_file.exists() else {}
        if preflight.get("identity") != identity or preflight.get("passed") is not True:
            raise RuntimeError(f"Run preflight_{cfg['NATIVE_PROFILE']}.py first; matching passed evidence is required.")
    if cfg["BACKEND"] == "physical_robot" and not cfg["PHYSICAL_PATH_AND_WORKSPACE_REVIEWED"]:
        raise RuntimeError("Review Gazebo paths/reset collision results and actual workspace before enabling physical execution.")
    goals = [None] if tracking else cfg["GOALS_XYZ_M"]
    for index, goal in enumerate(goals):
        print(f"{cfg['BACKEND']}: reset to comfortable, then {cfg['TASK_NAME']} {index + 1}; target={goal}")
        phrase = f"RUN {cfg['TASK_NAME']} {index + 1} ON {cfg['BACKEND']}"
        if input(f"Type {phrase!r} before ANY motion: ").strip() != phrase:
            print("Motion declined.")
            return 1
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        output = raw_root / f"{stamp}-{cfg['TASK_NAME']}-{index + 1}"
        output.mkdir()
        write_json(output / "resolved_config.json", dict(cfg, goal_xyz_m=goal))
        write_json(output / "provenance.json", dict(
            source_commit=source_commit,
            config_sha256=digest(cfg), script_sha256=hashlib.sha256(Path(namespace["__file__"]).read_bytes()).hexdigest(),
            native_build=receipt, family_identity=identity,
            power_mode="MODE_40W user-confirmed, not measured" if cfg["BACKEND"] == "physical_robot" else None,
            worktree_changes=subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)))
        result = execute(cfg, output, goal)
        write_json(output / "result.json", result)
        passed = result["task_success"] and result["physical_pass"] and result["handback_confirmed"] and not result["errors"]
        if cfg["IS_PREFLIGHT"]:
            write_json(preflight_file, dict(identity=identity, passed=bool(passed), result_directory=str(output)))
        with (raw_root / "manual-run-index.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(task=cfg["TASK_NAME"], path=str(output), **result), default=json_value) + "\n")
        print(json.dumps(dict(path=str(output), **result), default=json_value, indent=2))
        if not result["physical_pass"] or not result["handback_confirmed"] or result["errors"] or result["interrupted"]:
            return 1
        if cfg["IS_PREFLIGHT"] and not passed:
            return 1
    return 0
