#!/usr/bin/env python3
"""Offline sampled task report; no interpolation, rollout or physical certification."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def build_report(config, stats, fk):
    """fk(q) returns (XYZ, rotation) in torso_lift_link for arm_right_tool_link."""
    goals = config["GOALS_POSES_M_RAD"]
    for key, expected in (("GOAL_FRAME", "torso_lift_link"), ("TOOL_FRAME", "arm_right_tool_link")):
        if config.get(key, expected) != expected:
            raise ValueError(f"Unsupported {key}: {config[key]}")
    outcomes, reached = stats.get("goal_outcomes", []), stats.get("goal_reached_times", [])
    report = {"scope": "Recorded task samples and identified boundaries only; no physical compliance claim",
              "strict_goal_outcomes_recorded": outcomes, "strict_goal_reached_times_s_recorded": reached,
              "time_to_all_reached_s_recorded": stats.get("time_to_all_reached"),
              "rough_gate": {k: config[k] for k in (
                  "ROUGH_POSITION_TOLERANCE_M", "ROUGH_TOOL_AXIS_TOLERANCE_DEG",
                  "ROUGH_SPEED_L1_RAD_S", "ROUGH_DWELL_S")}, "issues": [], "goals": []}
    if config["ROUGH_DWELL_S"] != 0:
        raise ValueError("This report supports the selected zero-dwell rough gate only")
    rows = []
    keys = ("timestamps", "goal_indices", "joint_positions", "joint_velocities")
    lengths = [len(stats.get(k, [])) for k in keys]
    if len(set(lengths)) > 1:
        report["issues"].append("Mismatched sample arrays; only aligned prefix examined")
    for t, index, q, v in zip(*(stats.get(k, []) for k in keys)):
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(goals):
            report["issues"].append(f"Invalid recorded goal index: {index}")
            continue
        rows.append((t, index, q, v, "per_solve_observation"))
    initial = stats.get("initial_state")
    if initial is not None and len(initial) == 14:
        rows.insert(0, (0., 0, initial[:7], initial[7:], "initial_boundary"))
    completed = len(outcomes) == len(goals) and all(o in ("reached", "timeout") for o in outcomes)
    report["recorded_outcomes_complete"] = completed
    terminal, terminal_t = stats.get("terminal_state"), stats.get("terminal_time_s")
    terminal_assigned = False
    if terminal is not None and terminal_t is not None:
        # Normal final completion breaks before appending its final per-solve row.
        # A partial/error state without explicit goal identity cannot identify a new goal.
        if completed and len(terminal) == 14:
            rows.append((terminal_t, len(goals) - 1, terminal[:7], terminal[7:], "terminal_boundary"))
            terminal_assigned = True
        else:
            report["issues"].append("Terminal boundary not assigned: incomplete goal sequence or invalid state")
    else:
        report["issues"].append("Terminal boundary missing")
    for i, goal in enumerate(goals):
        desired = Rotation.from_euler("xyz", goal[3:]).as_matrix()
        samples = []
        for t, index, q, v, scope in rows:
            if index != i:
                continue
            try:
                q, v = np.asarray(q, dtype=float), np.asarray(v, dtype=float)
                if q.shape != (7,) or v.shape != (7,) or not np.isfinite(np.r_[q, v, t]).all():
                    raise ValueError("invalid q/v/time")
                xyz, rotation = fk(q)
                xyz, rotation = np.asarray(xyz), np.asarray(rotation)
                if xyz.shape != (3,) or rotation.shape != (3, 3) or not np.isfinite(np.r_[xyz, rotation.ravel()]).all():
                    raise ValueError("invalid FK pose")
                sample = {"time_s": float(t), "scope": scope, "xyz_m": xyz.tolist(),
                          "position_error_m": float(np.linalg.norm(xyz - goal[:3])),
                          "tool_axis_error_deg": float(np.degrees(np.arccos(np.clip(rotation[:, 2] @ desired[:, 2], -1, 1)))),
                          "full_so3_error_deg": float(np.degrees(Rotation.from_matrix(desired.T @ rotation).magnitude())),
                          "speed_l1_rad_s": float(np.abs(v).sum())}
                samples.append(sample)
            except (ValueError, TypeError) as exc:
                report["issues"].append(f"Goal {i}, {scope}: {exc}")
        samples.sort(key=lambda sample: sample["time_s"])
        rough = next((s for s in samples if
                      s["position_error_m"] <= config["ROUGH_POSITION_TOLERANCE_M"] and
                      s["tool_axis_error_deg"] <= config["ROUGH_TOOL_AXIS_TOLERANCE_DEG"] and
                      s["speed_l1_rad_s"] <= config["ROUGH_SPEED_L1_RAD_S"]), None)
        report["goals"].append({"goal_index": i, "name": config["GOAL_NAMES"][i], "target_pose_m_rad": goal,
                                "strict_outcome_recorded": outcomes[i] if i < len(outcomes) else None,
                                "strict_reached_time_s_recorded": reached[i] if i < len(reached) else None,
                                "observation_status": "observed" if samples else "missing_observations",
                                "last_observation_at_strict_arrival": (
                                    abs(samples[-1]["time_s"] - reached[i]) <= 1e-6
                                    if samples and i < len(reached) and reached[i] is not None else None),
                                "sample_count": len(samples), "rough_first_observed_entry": rough,
                                "last_observed": samples[-1] if samples else None})
    covered = all(goal["last_observed"] is not None for goal in report["goals"])
    report["status"] = "complete_recorded_sequence" if completed and covered and terminal_assigned and not report["issues"] else "incomplete"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runroot", type=Path)
    parser.add_argument("--model", type=Path, help="Explicit right-arm URDF override (recorded in report)")
    options = parser.parse_args()
    root = options.runroot.resolve()
    output = root / "square_report.json"
    if output.exists():
        parser.error(f"Refusing to overwrite {output}")
    config = json.loads((root / "launcher_config.json").read_text())
    stats_path = root / "data/task_stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    metadata_path = root / "data/metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    checkout = Path(__file__).resolve().parents[2]
    model_path = options.model or Path(metadata.get("model_path", "gato/dynamics/tiago_right/tiago_right_arm.urdf"))
    if not model_path.is_absolute():
        model_path = checkout / model_path
    import pinocchio as pin  # offline kinematics only; no ROS/native solver imports
    model = pin.buildModelFromUrdf(str(model_path))
    if model.nq != 7:
        raise ValueError("Expected the seven-joint right-arm model")
    torso, tool = model.getFrameId("torso_lift_link"), model.getFrameId("arm_right_tool_link")
    if torso >= model.nframes or tool >= model.nframes:
        raise ValueError("Required torso/tool frames missing from model")
    data = model.createData()

    def fk(q):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        pose = data.oMf[torso].inverse() * data.oMf[tool]
        return pose.translation.copy(), pose.rotation.copy()

    report = build_report(config, stats, fk)
    report["model"] = {"path": str(model_path.resolve()), "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                       "selection": "override" if options.model else "recorded metadata" if "model_path" in metadata else "checkout fallback"}
    report["sources"] = {"config": str(root / "launcher_config.json"), "stats": str(stats_path),
                         "stats_present": stats_path.exists()}
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(output)


if __name__ == "__main__":
    main()
