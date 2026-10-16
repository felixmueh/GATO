"""Small opt-in live-example confirmation and close-before-export helpers."""
import json
import hashlib
import math
import os
from pathlib import Path
import platform
import subprocess


def _json_value(value):
    return value.tolist() if hasattr(value, "tolist") else str(value)


def _positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError("must be finite and positive")
    return number


def _nonnegative(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError("must be finite and non-negative")
    return number


def add_live_arguments(parser):
    parser.add_argument("--ros-sea-absolute", action="store_true",
                        help="Use SEA output-encoder absolute positions and estimated arm velocity.")
    parser.add_argument("--ros-sea-state-topic",
                        default="/arm_right_sea_state_broadcaster_controller/sea_joint_states")
    parser.add_argument("--ros-sea-velocity-filter-tau", type=_nonnegative, default=.01,
                        help="Derivative low-pass time constant [s]; zero disables filtering.")
    parser.add_argument("--ros-joint-states-topic", default="/joint_states",
                        help="JointState source for measured feedback and full-state safety checks.")
    parser.add_argument("--ros-reset-position-tolerance", type=_positive, default=None,
                        help="Opt in to measured reset arrival; max joint error [rad].")
    parser.add_argument("--ros-reset-speed-l1", type=_positive, default=.1,
                        help="Reset arrival joint-speed L1 tolerance [rad/s].")
    parser.add_argument("--ros-reset-arrival-timeout", type=_positive, default=5.,
                        help="Additional reset-arrival wait [s].")
    parser.add_argument("--ros-record-commands", action="store_true")
    parser.add_argument("--ros-confirm-before-motion", action="store_true")


def confirm_live_motion(args):
    if args.ros_confirm_before_motion:
        print(json.dumps(vars(args), default=_json_value, indent=2))
        if input("Type MOVE before reset and live motion: ").strip() != "MOVE":
            raise RuntimeError("Live motion declined")


def check_live_backend(backend, *, physical_reviewed=False):
    expected = {"gazebo": ("simulation", "1"), "physical_robot": ("tiago", "2")}
    if backend not in expected:
        raise ValueError("backend must be gazebo or physical_robot")
    if (os.environ.get("GATO_ROS_PROFILE"), os.environ.get("ROS_DOMAIN_ID")) != expected[backend]:
        raise RuntimeError(f"Source the matching {backend} ROS profile inside the container")
    if os.environ.get("RMW_IMPLEMENTATION") != "rmw_cyclonedds_cpp":
        raise RuntimeError("The TIAGo ROS profiles require rmw_cyclonedds_cpp")
    if backend == "physical_robot" and (not physical_reviewed or platform.machine() not in {"aarch64", "arm64"}):
        raise RuntimeError("Physical motion requires reviewed paths/workspace and Jetson ARM64")


def finish_live_recording(controller, output, stats, error, timeout, *, context=None):
    """Close first, export independently, and return the saved result.

    Pass the active task exception as ``error`` and re-raise it in the caller's
    except block. This helper will preserve it even if cleanup/export also fails.
    With no task exception, cleanup/export errors raise after evidence is saved.
    ``context`` retains example args/settings and optional goals/native receipt.
    """
    errors = [] if error is None else [{"stage": "task", "error": f"{type(error).__name__}: {error}"}]
    handback = False
    controller_state_history = None
    interrupted = isinstance(error, KeyboardInterrupt)
    try:
        started = controller._process is not None
        controller.close(timeout_sec=timeout)
        handback = started
    except BaseException as exc:
        errors.append({"stage": "handback", "error": f"{type(exc).__name__}: {exc}"})
        interrupted |= isinstance(exc, KeyboardInterrupt)
    output = Path(output)
    try:
        output.mkdir(parents=True, exist_ok=True)
    except BaseException:
        if error is None:
            raise
        return {"handback_confirmed": handback, "errors": errors, "interrupted": interrupted}

    def save(name, value):
        try:
            (output / name).write_text(json.dumps(value, default=_json_value, indent=2) + "\n")
        except BaseException as exc:
            errors.append({"stage": name, "error": str(exc)})

    exports = [("write_state_history_csv", "controller_state_history.csv"),
               ("write_full_state_history_jsonl", "full_joint_state_history.jsonl")]
    if controller.record_command_events:
        exports += [("write_command_history_jsonl", "command_history.jsonl"),
                    ("write_lifecycle_events_json", "lifecycle_events.json")]
    for method, name in exports:
        try:
            saved = getattr(controller, method)(output / name)
            if method == "write_state_history_csv":
                controller_state_history = saved
                save("history_summary.json", saved)
            elif isinstance(saved, dict) and (saved.get("recording_complete") is not True or saved.get("errors")):
                raise RuntimeError(f"Incomplete {name}: {saved}")
        except BaseException as exc:
            errors.append({"stage": method, "error": str(exc)})
    try:
        (output / "history_metadata.json").write_bytes(controller._history_metadata_path.read_bytes())
    except BaseException as exc:
        errors.append({"stage": "history_metadata", "error": str(exc)})
    save("task_stats.json", stats)
    context = dict(context or {})
    commit = None
    try:
        safety = controller.collision_safety
        goals = context.get("goals")
        config = dict(context, COLLISION_MARGIN_M=safety.min_distance_m,
                      JOINT_POSITION_MARGIN_RAD=safety.joint_position_margin_rad,
                      JOINT_VELOCITY_SCALE=safety.joint_velocity_scale,
                      COLLISION_URDF_PATH=safety.urdf_path,
                      COLLISION_BLACKLIST_PATH=safety.blacklist_path,
                      goal_xyz_m=goals[0] if goals is not None and len(goals) else None)
        if goals is None and context.get("args", {}).get("sim_time") is not None:
            config["DURATION_S"] = context["args"]["sim_time"]
        save("resolved_config.json", config)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         cwd=Path(__file__).resolve().parents[2], text=True).strip()
        config_sha256 = hashlib.sha256(json.dumps(context, default=_json_value, sort_keys=True).encode()).hexdigest()
        save("provenance.json", dict(source_commit=commit, config_sha256=config_sha256, **context))
    except BaseException as exc:
        errors.append({"stage": "provenance", "error": str(exc)})
    outcomes = stats.get("goal_outcomes", [])
    result = {"handback_confirmed": handback, "errors": errors, "interrupted": interrupted,
              "task_success": all(value == "reached" for value in outcomes) if len(outcomes) else None,
              "controller_state_history": controller_state_history}
    save("result.json", result)
    index = context.get("args", {}).get("manual_run_index")
    if index is not None:
        try:
            index = Path(index)
            index.parent.mkdir(parents=True, exist_ok=True)
            with index.open("a") as stream:
                stream.write(json.dumps(dict(output_directory=output, source_commit=commit, **result),
                                        default=_json_value) + "\n")
        except BaseException as exc:
            errors.append({"stage": "manual_run_index", "error": str(exc)})
            save("result.json", result)
    if errors and error is None:
        raise RuntimeError(f"Live cleanup/recording failed: {errors}")
    return result
