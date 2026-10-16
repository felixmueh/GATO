"""Replay recorded full named states against the runtime self-collision model.

Run with: python tiago_examples/physical_arm/check_collisions.py RUN_DIRECTORY
This CPU-only check includes reset observations and linearly interpolates joint
positions/velocities at <=3 ms interpolation spacing only across supported source
gaps (<=100 ms). Interpolation does not turn sparse observations into 3 ms data.
It is sampled geometry review,
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
MAX_SOURCE_GAP_S = .10
STAMP_ROUNDOFF_S = 1e-6  # Epoch float64 subtraction, not an extra sensor budget.


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_observations(path, excluded_velocity_names=None):
    from gato_tiago.safety_monitor import TIAGO_RUNTIME_STATE_FIXED_JOINT_NAMES

    fixed_names = set(TIAGO_RUNTIME_STATE_FIXED_JOINT_NAMES)
    rows = []
    excluded = set()
    # Normalize only unavailable velocities of joints removed by the runtime
    # reduced model. Drop the same names throughout this in-memory replay so
    # interpolation never fabricates a velocity or sees changing key sets.
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        for name, value in row["velocities_by_name"].items():
            if name in fixed_names and (value is None or not math.isfinite(float(value))):
                excluded.add(name)
        rows.append((line_number, row))
    if excluded_velocity_names is not None:
        excluded_velocity_names.extend(sorted(excluded))
    observations = rows
    rows = []
    for line_number, row in observations:
        row["velocities_by_name"] = {name: value for name, value in row["velocities_by_name"].items()
                                     if name not in excluded}
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
    return rows


def interpolated_observations(rows):
    """Yield every source observation and interior linear samples in time order."""
    yield rows[0], False
    for left, right in zip(rows, rows[1:]):
        duration = float(right["stamp_sec"]) - float(left["stamp_sec"])
        if duration > MAX_SOURCE_GAP_S + STAMP_ROUNDOFF_S:
            # Keep both observations; never manufacture a path through an
            # unsupported gap, even when its interpolated geometry looks clear.
            yield right, False
            continue
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


def assess_coverage(run_directory, config, rows, report):
    """Bound observation coverage using recorded motion events, not timeouts."""
    reasons = report["coverage_reasons"]
    gaps = [float(b["stamp_sec"]) - float(a["stamp_sec"]) for a, b in zip(rows, rows[1:])]
    report["maximum_original_source_gap_s"] = max(gaps, default=0.)
    report["unsupported_source_gap_count"] = sum(g > MAX_SOURCE_GAP_S + STAMP_ROUNDOFF_S for g in gaps)
    if report["unsupported_source_gap_count"]:
        reasons.append("original observation gap exceeds MAX_SOURCE_GAP_S; no interpolation across these gaps")
    for mode in ("RESETTING", "RUNNING"):
        stamps = [float(r["stamp_sec"]) for r in rows if r["controller_mode"] == mode]
        if len(stamps) < 2 or max(stamps) <= min(stamps):
            reasons.append(f"missing positive-duration {mode} coverage")

    def read_record(name):
        path = run_directory / name
        if not path.is_file():
            raise ValueError(f"missing {name}; coverage unknown")
        report["input_sha256"][name] = sha256(path)
        return json.loads(path.read_text())

    try:
        metadata_path = next((run_directory / name for name in
                              ("history_metadata.json", "history_summary.json")
                              if (run_directory / name).is_file()), None)
        metadata = read_record(metadata_path.name) if metadata_path else config
        if "dropped_history_records" not in metadata:
            raise ValueError("missing dropped_history_records; completeness unknown")
        report["dropped_history_records"] = int(metadata["dropped_history_records"])
        if report["dropped_history_records"] != 0:
            reasons.append("history contains dropped records")
        if "record_count" in metadata and int(metadata["record_count"]) != len(rows):
            reasons.append("history record_count disagrees with JSONL")
    except Exception as exc:
        reasons.append(str(exc))

    try:
        result = read_record("result.json")
        report["recorded_task_result"] = {k: result.get(k) for k in
                                         ("task_success", "handback_confirmed", "interrupted", "errors")}
        if (result.get("handback_confirmed") is not True or result.get("interrupted") is not False
                or result.get("errors") != []):
            reasons.append("task result lacks error-free, uninterrupted execution with confirmed handback")
    except Exception as exc:
        reasons.append(str(exc))

    try:
        lifecycle = read_record("lifecycle_events.json")
        if lifecycle.get("recording_complete") is not True or lifecycle.get("errors") != []:
            raise ValueError("lifecycle recording incomplete or contains errors")
        events = lifecycle["events"]
        previous = -math.inf
        for event in events:
            moment = float(event["monotonic_sec"])
            if not math.isfinite(moment) or moment < previous:
                raise ValueError("invalid/out-of-order lifecycle monotonic time")
            previous = moment

        def marker(name, after=-math.inf, active=False, last=False):
            matches = [e for e in events if e["event"] == name and float(e["monotonic_sec"]) >= after
                       and (not active or e.get("effort_was_active") is True)]
            if not matches:
                raise ValueError(f"missing lifecycle boundary {name}")
            event = matches[-1] if last else matches[0]
            stamp = float(event["source_stamp_sec"])
            received = float(event["source_received_monotonic_sec"])
            if not isinstance(event.get("source_seq"), int):
                raise ValueError(f"missing source sequence at lifecycle boundary {name}")
            age = float(event["monotonic_sec"]) - received
            if not math.isfinite(stamp) or not math.isfinite(received) or not -STAMP_ROUNDOFF_S <= age <= MAX_SOURCE_GAP_S + STAMP_ROUNDOFF_S:
                raise ValueError(f"stale/missing observed source at lifecycle boundary {name}")
            if name not in {"reset_published", "handback_confirmed"} and not any(abs(float(r["stamp_sec"]) - stamp) <= 1e-6
                       and r.get("source_seq") == event["source_seq"]
                       and abs(float(r.get("received_monotonic_sec", -math.inf)) - received) <= 1e-6
                       for r in rows):
                raise ValueError(f"lifecycle boundary {name} has no matching source observation")
            return event

        reset = marker("reset_command")
        published = marker("reset_published", reset["monotonic_sec"])
        verified = marker("reset_verified", published["monotonic_sec"])
        if verified["source_received_monotonic_sec"] < published["monotonic_sec"]:
            raise ValueError("reset verification has no fresh observation after reset publication")
        first = marker("first_task_effort", verified["monotonic_sec"])
        request = marker("handback_request", first["monotonic_sec"], active=True)
        confirmed = marker("handback_confirmed", request["monotonic_sec"])
        observed = marker("handback_observed", confirmed["monotonic_sec"])
        final = marker("handback_observed", observed["monotonic_sec"], last=True)
        final_confirmations = [e for e in events if e["event"] == "handback_confirmed"
                               and e["monotonic_sec"] <= final["monotonic_sec"]]
        for endpoint, confirmation in ((observed, confirmed), (final, final_confirmations[-1])):
            if endpoint["source_received_monotonic_sec"] < confirmation["monotonic_sec"]:
                raise ValueError("RESTORED endpoint was not observed after handback confirmation")
            if not any(r["controller_mode"] == "RESTORED"
                       and r.get("source_seq") == endpoint.get("source_seq")
                       and abs(float(r["stamp_sec"]) - float(endpoint["source_stamp_sec"])) <= 1e-6 for r in rows):
                raise ValueError("missing fresh RESTORED history endpoint")
        for start, end, mode in ((reset, verified, "RESETTING"), (first, request, "RUNNING")):
            stamps = [float(r["stamp_sec"]) for r in rows if r["controller_mode"] == mode]
            if (not stamps or min(stamps) > float(start["source_stamp_sec"]) + MAX_SOURCE_GAP_S
                    or max(stamps) < float(end["source_stamp_sec"]) - MAX_SOURCE_GAP_S):
                raise ValueError(f"{mode} observations do not cover recorded motion boundaries")
        if float(rows[0]["stamp_sec"]) > float(reset["source_stamp_sec"]) + 1e-6:
            raise ValueError("history begins after reset command")
        if float(rows[-1]["stamp_sec"]) < float(final["source_stamp_sec"]) - 1e-6:
            raise ValueError("history ends before confirmed handback observation")
        task_duration = float(request["source_stamp_sec"]) - float(first["source_stamp_sec"])
        if task_duration <= 0:
            raise ValueError("nonpositive observed task interval")
        report["observed_task_interval_s"] = task_duration
        report["motion_boundaries"] = {e["event"]: e for e in
                                       (reset, published, verified, first, request, confirmed, final)}
        if config.get("goal_xyz_m") is None and "DURATION_S" in config:
            planned = float(config["DURATION_S"])
            report["planned_tracking_duration_s"] = planned
            stats = read_record("task_stats.json")
            origin = float(stats["source_time_origin_sec"])
            terminal_relative = float(stats["timestamps"][-1])
            if not math.isfinite(origin) or not math.isfinite(terminal_relative) or terminal_relative < 0:
                raise ValueError("invalid tracking source-time origin/terminal feedback")
            terminal = origin + terminal_relative
            report["tracking_source_time_origin_sec"] = origin
            report["tracking_terminal_source_stamp_sec"] = terminal
            report["tracking_terminal_relative_s"] = terminal_relative
            for name, stamp in (("origin", origin), ("terminal", terminal)):
                if (not float(rows[0]["stamp_sec"]) <= stamp <= float(rows[-1]["stamp_sec"])
                        or min(abs(float(r["stamp_sec"]) - stamp) for r in rows) > MAX_SOURCE_GAP_S):
                    raise ValueError(f"tracking {name} source instant is outside supported observed history")
            if (origin > float(first["source_stamp_sec"]) + MAX_SOURCE_GAP_S
                    or terminal > float(request["source_stamp_sec"]) + MAX_SOURCE_GAP_S):
                raise ValueError("tracking feedback source instants disagree with actual task boundaries")
            if not math.isfinite(planned) or planned <= 0 or terminal_relative < planned - MAX_SOURCE_GAP_S:
                reasons.append("recorded tracking feedback interval is shorter than planned duration minus .10s tolerance")
    except Exception as exc:
        reasons.append(f"lifecycle coverage: {exc}")
    report["coverage_complete"] = not reasons


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
    report = {"checker_source_sha256": sha256(Path(__file__)),
              "passed": False, "geometry_clearance_pass": False,
              "joint_limits_pass": False, "coverage_complete": False,
              "coverage_reasons": [], "errors": [], "minimum_checked_clearance": None,
              "excluded_unavailable_fixed_joint_velocities": [],
              "velocity_scope": "Unavailable velocities are omitted only for explicitly fixed joints in the runtime reduced model. Their recorded positions must remain fixed; all required model velocities remain mandatory. Raw history is unchanged.",
              "checked_samples": 0, "interpolated_samples": 0,
              "collision_violation_samples": 0, "joint_position_violation_samples": 0,
              "joint_velocity_violation_samples": 0, "first_joint_violations": [],
              "maximum_interpolation_step_s": MAX_STEP_S,
              "maximum_supported_source_gap_s": MAX_SOURCE_GAP_S,
              "source_timestamp_roundoff_s": STAMP_ROUNDOFF_S,
              "distance_kind": "Coal mesh distance queries for every retained monitored pair; not thresholded collision lower bounds",
              "scope": "Sampled robot self-geometry touching the right-arm subtree, with configured blacklist. No lab/environment obstacles, swept-volume/continuous-motion guarantee, or sensor uncertainty. Manual workspace/trajectory review remains necessary.",
              "interpolation": "Linear named joint positions and separately linear recorded velocities only across original gaps <=.10s; interpolation is not 3ms measurement data and does not reconstruct unobserved motion or certify between-sample clearance."}
    try:
        history = run_directory / "full_joint_state_history.jsonl"
        config_path = run_directory / "resolved_config.json"
        report["input_sha256"] = {history.name: sha256(history), config_path.name: sha256(config_path)}
        config = json.loads(config_path.read_text())
        margin = float(config["COLLISION_MARGIN_M"])
        if not math.isfinite(margin) or margin < 0:
            raise ValueError("invalid collision margin")
        report["collision_margin_m"] = margin
        rows = load_observations(history, report["excluded_unavailable_fixed_joint_velocities"])
        report["source_observations"] = len(rows)
        report["source_modes"] = {mode: sum(r["controller_mode"] == mode for r in rows)
                                  for mode in sorted({r["controller_mode"] for r in rows})}
        assess_coverage(run_directory, config, rows, report)
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
        report["geometry_clearance_pass"] = report["collision_violation_samples"] == 0
        report["joint_limits_pass"] = not any(report[k] for k in (
            "joint_position_violation_samples", "joint_velocity_violation_samples"))
        report["passed"] = (report["geometry_clearance_pass"] and report["joint_limits_pass"]
                            and report["coverage_complete"])
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
                      "geometry_clearance_pass": result["geometry_clearance_pass"],
                      "coverage_complete": result["coverage_complete"],
                      "coverage_reasons": result["coverage_reasons"],
                      "minimum_checked_clearance": result["minimum_checked_clearance"],
                      "errors": result["errors"]}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
