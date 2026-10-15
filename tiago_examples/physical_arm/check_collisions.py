"""Replay recorded full named states against the runtime self-collision model.

Run with: python tiago_examples/physical_arm/check_collisions.py RUN_DIRECTORY
This CPU-only check includes reset observations and linearly interpolates joint
positions/velocities at <=3 ms source-time spacing. It is sampled geometry review,
not a certificate of the physical path or of clearance from lab objects.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tiago_src")]
MAX_STEP_S = .003


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_observations(path):
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        for key in ("positions_by_name", "velocities_by_name"):
            values = row[key]
            if not values or any(v is None or not math.isfinite(float(v)) for v in values.values()):
                raise ValueError(f"line {line_number}: empty/nonfinite {key}")
        if not math.isfinite(float(row["stamp_sec"])):
            raise ValueError(f"line {line_number}: nonfinite stamp_sec")
        if "controller_mode" not in row:
            raise ValueError(f"line {line_number}: missing controller_mode")
        if rows:
            previous = rows[-1]
            if row["stamp_sec"] < previous["stamp_sec"]:
                raise ValueError(f"line {line_number}: source time goes backwards")
            for key in ("positions_by_name", "velocities_by_name"):
                if row[key].keys() != previous[key].keys():
                    raise ValueError(f"line {line_number}: incomplete/changing named state")
                if row["stamp_sec"] == previous["stamp_sec"] and row[key] != previous[key]:
                    raise ValueError(f"line {line_number}: changed state with repeated timestamp")
        rows.append(row)
    if not rows:
        raise ValueError("empty full joint-state history")
    for mode in ("RESETTING", "RUNNING"):
        stamps = [float(r["stamp_sec"]) for r in rows if r["controller_mode"] == mode]
        if len(stamps) < 2 or max(stamps) <= min(stamps):
            raise ValueError(f"missing positive-duration {mode} observation coverage")
    return rows


def interpolated_observations(rows):
    """Yield every source observation and interior linear samples in time order."""
    yield rows[0], False
    for left, right in zip(rows, rows[1:]):
        duration = float(right["stamp_sec"]) - float(left["stamp_sec"])
        steps = max(1, math.ceil(duration / MAX_STEP_S))
        for index in range(1, steps):
            fraction = index / steps
            row = {"stamp_sec": float(left["stamp_sec"]) + duration * fraction,
                   "controller_mode": left["controller_mode"]}
            for key in ("positions_by_name", "velocities_by_name"):
                row[key] = {name: float(a) + fraction * (float(right[key][name]) - float(a))
                            for name, a in left[key].items()}
            yield row, True
        yield right, False


class GeometryReview:
    def __init__(self, config, initial):
        from gato_tiago import safety_monitor as safety
        from gato_tiago.ros_tiago import RIGHT_ARM_JOINTS
        from gato_tiago.tiago_tools import compute_pair_distances

        self.safety = safety
        self.distances = compute_pair_distances
        self.joints = RIGHT_ARM_JOINTS
        self.position_margin = float(config["JOINT_POSITION_MARGIN_RAD"])
        self.velocity_scale = float(config["JOINT_VELOCITY_SCALE"])
        if not math.isfinite(self.position_margin) or self.position_margin < 0:
            raise ValueError("invalid joint position margin")
        if not math.isfinite(self.velocity_scale) or self.velocity_scale <= 0:
            raise ValueError("invalid joint velocity scale")
        urdf = source_path(config["COLLISION_URDF_PATH"])
        blacklist = source_path(config["COLLISION_BLACKLIST_PATH"])
        if not blacklist.is_file():
            raise FileNotFoundError(f"collision blacklist missing: {blacklist}")
        self.model = safety.build_tiago_collision_model(
            urdf_path=urdf, blacklist_path=blacklist,
            reference_positions=initial["positions_by_name"], controlled_joint_names=self.joints,
            state_fixed_joint_names=safety.TIAGO_RUNTIME_STATE_FIXED_JOINT_NAMES,
            monitor_only_collision_pairs=True)
        if not self.model.geometry_model.collisionPairs:
            raise ValueError("collision model has no monitored pairs")
        files = {str(urdf): sha256(urdf), str(blacklist): sha256(blacklist)}
        for obj in self.model.geometry_model.geometryObjects:
            mesh = str(getattr(obj, "meshPath", ""))
            # Pinocchio also stores primitive labels such as BOX here.
            if mesh and type(obj.geometry).__name__ not in {"Box", "Cylinder", "Sphere", "Capsule", "Cone"}:
                files[mesh] = sha256(mesh)
        self.fixed = {name: initial["positions_by_name"].get(name, 0.)
                      for name in safety.TIAGO_RUNTIME_STATE_FIXED_JOINT_NAMES}
        self.provenance = {
            "geometry_files_sha256": files,
            "safety_monitor_sha256": sha256(Path(safety.__file__)),
            "distance_helper_sha256": sha256(ROOT / "tiago_src/gato_tiago/tiago_tools.py"),
            "monitored_pair_count": len(self.model.geometry_model.collisionPairs),
            "fixed_joint_positions_rad": self.fixed,
            "fixed_joint_assumption": "Runtime wheel/finger joints locked at first recorded position; missing locked joints use URDF neutral. Changing recorded locked joints fail this review.",
        }

    def check(self, row):
        for name, value in self.fixed.items():
            recorded = row["positions_by_name"].get(name, value)
            if abs(float(recorded) - float(value)) > 1e-6:
                raise ValueError(f"runtime fixed joint moved: {name}; reduced geometry is insufficient")
        state = self.safety.NamedJointState(row["positions_by_name"], row["velocities_by_name"], row["stamp_sec"])
        q, _ = self.safety.state_to_qv(self.model.model, state)
        violations = self.safety.check_joint_limits(
            self.model.model, state, position_margin=self.position_margin,
            velocity_scale=self.velocity_scale, joint_names=self.joints)
        pairs = self.distances(self.model, q)
        if not pairs or any(not math.isfinite(p.distance_m) for p in pairs):
            raise ValueError("missing/nonfinite mesh pair distances")
        nearest = min(pairs, key=lambda p: p.distance_m)
        pair = nearest.to_json()
        # Nearest-point coordinates are unnecessary for this minimum-clearance
        # report and may be unspecified by the backend for intersecting meshes.
        pair.pop("nearest_point_a", None)
        pair.pop("nearest_point_b", None)
        return pair, [asdict(v) for v in violations]


def review(run_directory, geometry_factory=GeometryReview):
    run_directory = Path(run_directory)
    report = {"passed": False, "errors": [], "minimum_checked_clearance": None,
              "checked_samples": 0, "interpolated_samples": 0,
              "collision_violation_samples": 0, "joint_position_violation_samples": 0,
              "joint_velocity_violation_samples": 0, "first_joint_violations": [],
              "maximum_interpolation_step_s": MAX_STEP_S,
              "distance_kind": "Coal mesh distance queries for every retained monitored pair; not thresholded collision lower bounds",
              "scope": "Sampled robot self-geometry touching the right-arm subtree, with configured blacklist. No lab/environment obstacles, swept-volume/continuous-motion guarantee, or sensor uncertainty. Manual workspace/trajectory review remains necessary.",
              "interpolation": "Linear named joint positions and separately linear recorded velocities; does not reconstruct unobserved motion or certify between-sample clearance."}
    try:
        history = run_directory / "full_joint_state_history.jsonl"
        config_path = run_directory / "resolved_config.json"
        report["input_sha256"] = {history.name: sha256(history), config_path.name: sha256(config_path)}
        config = json.loads(config_path.read_text())
        margin = float(config["COLLISION_MARGIN_M"])
        if not math.isfinite(margin) or margin < 0:
            raise ValueError("invalid collision margin")
        report["collision_margin_m"] = margin
        rows = load_observations(history)
        report["source_observations"] = len(rows)
        report["source_modes"] = {mode: sum(r["controller_mode"] == mode for r in rows)
                                  for mode in sorted({r["controller_mode"] for r in rows})}
        metadata_path = next((run_directory / name for name in
                              ("history_metadata.json", "history_summary.json")
                              if (run_directory / name).is_file()), None)
        if metadata_path is not None:
            metadata = json.loads(metadata_path.read_text())
            report["input_sha256"][metadata_path.name] = sha256(metadata_path)
        else:
            metadata = config
        if "dropped_history_records" not in metadata:
            raise ValueError("missing dropped_history_records: history completeness unknown")
        report["dropped_history_records"] = int(metadata["dropped_history_records"])
        if report["dropped_history_records"] != 0:
            raise ValueError("history contains dropped records")
        if "record_count" in metadata and int(metadata["record_count"]) != len(rows):
            raise ValueError("history record_count disagrees with JSONL")
        geometry = geometry_factory(config, rows[0])
        report.update(geometry.provenance)
        for row, interpolated in interpolated_observations(rows):
            pair, violations = geometry.check(row)
            report["checked_samples"] += 1
            report["interpolated_samples"] += int(interpolated)
            sample = {**pair, "stamp_sec": row["stamp_sec"],
                      "controller_mode": row["controller_mode"], "interpolated": interpolated}
            minimum = report["minimum_checked_clearance"]
            if minimum is None or pair["distance_m"] < minimum["distance_m"]:
                report["minimum_checked_clearance"] = sample
            if pair["distance_m"] <= margin:
                report["collision_violation_samples"] += 1
            for category, prefix in (("position", "joint_position"), ("velocity", "joint_velocity")):
                matching = [v for v in violations if v["kind"].startswith(category)]
                report[prefix + "_violation_samples"] += int(bool(matching))
                if matching and len(report["first_joint_violations"]) < 20:
                    report["first_joint_violations"].append({"stamp_sec": row["stamp_sec"],
                                                           "controller_mode": row["controller_mode"],
                                                           "violations": matching})
        report["passed"] = not any(report[k] for k in (
            "collision_violation_samples", "joint_position_violation_samples", "joint_velocity_violation_samples"))
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    output = run_directory / "collision_review.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    result = review(args.run_directory)
    print(json.dumps({"passed": result["passed"], "checked_samples": result["checked_samples"],
                      "minimum_checked_clearance": result["minimum_checked_clearance"],
                      "errors": result["errors"]}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
