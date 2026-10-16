"""Offline replay checks with geometry standing in for the expensive mesh model."""
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "tiago_examples/physical_arm/check_collisions.py"
spec = importlib.util.spec_from_file_location("physical_collision_replay", SCRIPT)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def make_run(tmp_path):
    rows = [{"stamp_sec": t, "received_monotonic_sec": t, "source_seq": index,
             "controller_mode": mode,
             "positions_by_name": {"joint": q}, "velocities_by_name": {"joint": 0.}}
            for index, (t, mode, q) in enumerate([(0., "RESETTING", -1.), (.006, "RESETTING", 1.),
                                (.012, "RUNNING", 1.), (.018, "RUNNING", 1.),
                                (.024, "RESTORED", 1.)])]
    (tmp_path / "full_joint_state_history.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "resolved_config.json").write_text(json.dumps({"COLLISION_MARGIN_M": .04}))
    (tmp_path / "history_metadata.json").write_text(json.dumps(
        {"record_count": len(rows), "dropped_history_records": 0}))
    (tmp_path / "result.json").write_text(json.dumps(
        {"task_success": True, "handback_confirmed": True, "interrupted": False, "errors": []}))
    (tmp_path / "task_stats.json").write_text(json.dumps(
        {"source_time_origin_sec": .012, "timestamps": [0., .006]}))
    events = []
    for name, moment, index in [("reset_command", 0., 0), ("reset_published", .001, 0),
                               ("reset_verified", .006, 1), ("first_task_effort", .012, 2),
                               ("handback_request", .018, 3), ("handback_confirmed", .019, 3),
                               ("handback_observed", .024, 4)]:
        events.append(dict(event=name, monotonic_sec=moment, source_stamp_sec=rows[index]["stamp_sec"],
                           source_received_monotonic_sec=rows[index]["received_monotonic_sec"],
                           source_seq=index, effort_was_active=name == "handback_request"))
    (tmp_path / "lifecycle_events.json").write_text(json.dumps(
        {"schema_version": 1, "events": events, "recording_complete": True, "errors": []}))
    return rows


class FakeGeometry:
    provenance = {"geometry_files_sha256": {"fake": "recorded"}}

    def __init__(self, config, first):
        pass

    def check(self, row):
        # Endpoints clear; the midpoint of the reset passes close to the body.
        distance = .02 + .1 * abs(row["positions_by_name"]["joint"])
        return {"geometry_a": "arm", "geometry_b": "body", "distance_m": distance}, []


class ClearGeometry(FakeGeometry):
    def check(self, row):
        return {"geometry_a": "arm", "geometry_b": "body", "distance_m": .2}, []


def test_interpolation_catches_reset_collision_and_preserves_raw(tmp_path):
    rows = make_run(tmp_path)
    raw = (tmp_path / "full_joint_state_history.jsonl").read_bytes()
    report = replay.review(tmp_path, FakeGeometry)
    assert not report["passed"]
    assert report["errors"] == []
    assert report["collision_violation_samples"] >= 1
    minimum = report["minimum_checked_clearance"]
    assert minimum["distance_m"] == pytest.approx(.02)
    assert minimum["stamp_sec"] == pytest.approx(.003)
    assert minimum["controller_mode"] == "RESETTING"
    assert minimum["interpolated"]
    samples = list(replay.interpolated_observations(rows))
    assert max(b[0]["stamp_sec"] - a[0]["stamp_sec"] for a, b in zip(samples, samples[1:])) <= .003 + 1e-15
    assert (tmp_path / "full_joint_state_history.jsonl").read_bytes() == raw
    assert json.loads((tmp_path / "collision_review.json").read_text()) == report


@pytest.mark.parametrize("problem", ["missing_name", "nonfinite"])
def test_invalid_coverage_fails_before_geometry(tmp_path, problem):
    rows = make_run(tmp_path)
    if problem == "missing_name":
        rows[-1]["positions_by_name"] = {}
    else:
        rows[-1]["velocities_by_name"]["joint"] = None
    (tmp_path / "full_joint_state_history.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))

    def unexpected_geometry(*args):
        pytest.fail("incomplete/invalid data must fail before geometry construction")

    report = replay.review(tmp_path, unexpected_geometry)
    assert not report["passed"] and report["errors"]
    assert report["checked_samples"] == 0


def test_joint_limits_report_separately_from_clear_collision(tmp_path):
    make_run(tmp_path)

    class JointViolationGeometry(FakeGeometry):
        def check(self, row):
            return {"geometry_a": "arm", "geometry_b": "body", "distance_m": .2}, [
                {"joint": "joint", "kind": "velocity", "value": 3.}]

    report = replay.review(tmp_path, JointViolationGeometry)
    assert not report["passed"] and not report["errors"]
    assert report["collision_violation_samples"] == 0
    assert report["joint_position_violation_samples"] == 0
    assert report["joint_velocity_violation_samples"] == report["checked_samples"]


def test_complete_clear_run_accepts_live_history_summary(tmp_path):
    make_run(tmp_path)
    (tmp_path / "history_metadata.json").rename(tmp_path / "history_summary.json")

    report = replay.review(tmp_path, ClearGeometry)
    assert report["passed"] and not report["errors"]
    assert report["source_modes"] == {"RESETTING": 2, "RUNNING": 2, "RESTORED": 1}
    assert "history_summary.json" in report["input_sha256"]


def test_long_original_gap_is_not_interpolated_or_qualified(tmp_path):
    rows = make_run(tmp_path)
    # Supported observations before/after a nearly two-second missing interval.
    rows[-1]["stamp_sec"] = 2.008
    rows[-1]["received_monotonic_sec"] = 2.008
    (tmp_path / "full_joint_state_history.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    report = replay.review(tmp_path, ClearGeometry)
    assert report["geometry_clearance_pass"] and not report["coverage_complete"]
    assert not report["passed"]
    assert report["maximum_original_source_gap_s"] == pytest.approx(1.99)
    assert report["unsupported_source_gap_count"] == 1
    checked = list(replay.interpolated_observations(rows))
    assert not any(.018 < r["stamp_sec"] < 2.008 for r, _ in checked)


def test_truncated_tracking_cannot_pass_even_with_successful_result(tmp_path):
    make_run(tmp_path)
    (tmp_path / "resolved_config.json").write_text(json.dumps(
        {"COLLISION_MARGIN_M": .04, "DURATION_S": 24., "goal_xyz_m": None}))
    report = replay.review(tmp_path, ClearGeometry)
    assert report["geometry_clearance_pass"] and not report["passed"]
    assert any("tracking feedback interval" in reason for reason in report["coverage_reasons"])


def test_successful_early_reach_uses_actual_boundaries_not_timeout(tmp_path):
    make_run(tmp_path)
    (tmp_path / "resolved_config.json").write_text(json.dumps(
        {"COLLISION_MARGIN_M": .04, "GOAL_TIMEOUT_S": 8., "goal_xyz_m": [.4, -.2, -.3]}))
    report = replay.review(tmp_path, ClearGeometry)
    assert report["passed"] and report["coverage_complete"]
    assert report["observed_task_interval_s"] == pytest.approx(.006)


def make_timed_run(tmp_path, *, duration=24., startup_delay=2., tracking=True):
    make_run(tmp_path)
    first_effort = .012 + startup_delay
    terminal = .012 + duration
    # Keep actual READY observations during bootstrap rather than inventing them.
    source_times = [0., .006] + [.012 + .05 * i for i in range(round(duration / .05) + 1)] + [terminal + .006]
    rows = [{"stamp_sec": t, "received_monotonic_sec": t, "source_seq": index,
             "controller_mode": ("RESETTING" if t < .012 else "READY" if t < first_effort
                                 else "RESTORED" if t > terminal else "RUNNING"),
             "positions_by_name": {"joint": 1.}, "velocities_by_name": {"joint": 0.}}
            for index, t in enumerate(source_times)]
    (tmp_path / "full_joint_state_history.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "history_metadata.json").write_text(json.dumps(
        {"record_count": len(rows), "dropped_history_records": 0}))
    task_config = ({"DURATION_S": duration, "goal_xyz_m": None} if tracking else
                   {"GOAL_TIMEOUT_S": duration, "goal_xyz_m": [.4, -.2, -.3]})
    (tmp_path / "resolved_config.json").write_text(json.dumps({"COLLISION_MARGIN_M": .04, **task_config}))
    (tmp_path / "task_stats.json").write_text(json.dumps(
        {"source_time_origin_sec": .012, "timestamps": [0., duration]}))
    events = []
    for name, moment, source_time in [("reset_command", 0., 0.), ("reset_published", .001, 0.),
                                     ("reset_verified", .006, .006), ("first_task_effort", first_effort, first_effort),
                                     ("handback_request", terminal, terminal),
                                     ("handback_confirmed", terminal + .001, terminal),
                                     ("handback_observed", terminal + .006, terminal + .006)]:
        index = min(range(len(rows)), key=lambda i: abs(rows[i]["stamp_sec"] - source_time))
        row = rows[index]
        events.append(dict(event=name, monotonic_sec=moment, source_stamp_sec=row["stamp_sec"],
                           source_received_monotonic_sec=row["received_monotonic_sec"],
                           source_seq=index, effort_was_active=name == "handback_request"))
    (tmp_path / "lifecycle_events.json").write_text(json.dumps(
        {"schema_version": 1, "events": events, "recording_complete": True, "errors": []}))


def test_tracking_preserves_existing_bootstrap_clock_with_shorter_effort_interval(tmp_path):
    make_timed_run(tmp_path)
    report = replay.review(tmp_path, ClearGeometry)
    assert report["coverage_reasons"] == []
    assert report["passed"]
    assert report["observed_task_interval_s"] == pytest.approx(22.)
    assert report["tracking_terminal_relative_s"] == 24.


def test_complete_tracking_error_gate_failure_keeps_collision_coverage(tmp_path):
    make_timed_run(tmp_path)
    result_path = tmp_path / "result.json"
    result = json.loads(result_path.read_text())
    result.update(task_success=False, observations={"tracking_feedback_rms_m": .057})
    result_path.write_text(json.dumps(result))
    report = replay.review(tmp_path, ClearGeometry)
    assert report["coverage_complete"] and report["geometry_clearance_pass"] and report["passed"]
    assert report["recorded_task_result"]["task_success"] is False
    assert json.loads(result_path.read_text())["task_success"] is False


def test_complete_ordinary_reach_timeout_keeps_collision_coverage(tmp_path):
    make_timed_run(tmp_path, duration=8., startup_delay=0., tracking=False)
    result_path = tmp_path / "result.json"
    result = json.loads(result_path.read_text())
    result.update(task_success=False, observations={"goal_outcomes": ["timeout"]})
    result_path.write_text(json.dumps(result))
    report = replay.review(tmp_path, ClearGeometry)
    assert report["coverage_complete"] and report["geometry_clearance_pass"] and report["passed"]
    assert report["observed_task_interval_s"] == pytest.approx(8.)
    assert report["recorded_task_result"]["task_success"] is False


@pytest.mark.parametrize("problem", ["missing_terminal", "missing_lifecycle", "missing_result",
                                     "old_incomplete", "drops", "failed_task", "stale_boundary"])
def test_incomplete_paths_retain_geometry_without_full_qualification(tmp_path, problem):
    rows = make_run(tmp_path)
    if problem == "missing_terminal":
        rows[-1]["controller_mode"] = "RUNNING"
    elif problem == "missing_lifecycle":
        (tmp_path / "lifecycle_events.json").unlink()
    elif problem == "missing_result":
        (tmp_path / "result.json").unlink()
    elif problem == "old_incomplete":
        for row in rows:
            row["controller_mode"] = "RUNNING"
        (tmp_path / "lifecycle_events.json").unlink()
        (tmp_path / "history_metadata.json").unlink()
    elif problem == "drops":
        (tmp_path / "history_metadata.json").write_text('{"dropped_history_records": 2}')
    elif problem == "failed_task":
        (tmp_path / "result.json").write_text(json.dumps(
            {"task_success": False, "handback_confirmed": True, "interrupted": True, "errors": []}))
    else:
        path = tmp_path / "lifecycle_events.json"
        lifecycle = json.loads(path.read_text())
        lifecycle["events"][3]["source_received_monotonic_sec"] -= .2
        path.write_text(json.dumps(lifecycle))
    (tmp_path / "full_joint_state_history.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    report = replay.review(tmp_path, ClearGeometry)
    assert report["geometry_clearance_pass"] and report["checked_samples"] >= len(rows)
    assert not report["coverage_complete"] and not report["passed"]
    assert report["coverage_reasons"]


def test_unavailable_fixed_wheel_velocities_are_omitted_only_in_replay(tmp_path):
    rows = make_run(tmp_path)
    wheels = ["wheel_front_left_joint", "wheel_front_right_joint",
              "wheel_rear_left_joint", "wheel_rear_right_joint"]
    for index, row in enumerate(rows):
        row["positions_by_name"].update(dict.fromkeys(wheels, .25))
        row["velocities_by_name"].update(dict.fromkeys(wheels, None if index else 0.))
    history = tmp_path / "full_joint_state_history.jsonl"
    history.write_text("".join(json.dumps(row) + "\n" for row in rows))
    original = history.read_bytes()

    class FixedWheelGeometry(ClearGeometry):
        def __init__(self, config, first):
            self.fixed = dict.fromkeys(wheels, .25)

        def check(self, row):
            assert all(name not in row["velocities_by_name"] for name in wheels)
            # Exercise the unchanged production fixed-position guard before
            # substituting the expensive mesh query for this focused test.
            if any(row["positions_by_name"][name] != .25 for name in wheels):
                replay.GeometryReview.check(self, row)
            return super().check(row)

    report = replay.review(tmp_path, FixedWheelGeometry)
    assert report["passed"]
    assert report["excluded_unavailable_fixed_joint_velocities"] == sorted(wheels)
    assert history.read_bytes() == original
    rows[-1]["positions_by_name"][wheels[0]] += .01
    history.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = replay.review(tmp_path, FixedWheelGeometry)
    assert not report["passed"]
    assert any("runtime fixed joint moved" in error for error in report["errors"])


@pytest.mark.parametrize("name", ["arm_right_1_joint", "unknown_joint"])
def test_unavailable_required_or_unknown_velocity_still_fails(tmp_path, name):
    rows = make_run(tmp_path)
    for row in rows:
        row["velocities_by_name"][name] = None
    (tmp_path / "full_joint_state_history.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows))
    report = replay.review(tmp_path, ClearGeometry)
    assert not report["passed"]
    assert report["checked_samples"] == 0
    assert report["excluded_unavailable_fixed_joint_velocities"] == []
    assert any("nonfinite velocities_by_name" in error for error in report["errors"])
