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
    rows = [{"stamp_sec": t, "controller_mode": mode,
             "positions_by_name": {"joint": q}, "velocities_by_name": {"joint": 0.}}
            for t, mode, q in [(0., "RESETTING", -1.), (.006, "RESETTING", 1.),
                                (.012, "RUNNING", 1.), (.018, "RUNNING", 1.)]]
    (tmp_path / "full_joint_state_history.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "resolved_config.json").write_text(json.dumps({"COLLISION_MARGIN_M": .04}))
    (tmp_path / "history_metadata.json").write_text(json.dumps(
        {"record_count": len(rows), "dropped_history_records": 0}))
    return rows


class FakeGeometry:
    provenance = {"geometry_files_sha256": {"fake": "recorded"}}

    def __init__(self, config, first):
        pass

    def check(self, row):
        # Endpoints clear; the midpoint of the reset passes close to the body.
        distance = .02 + .1 * abs(row["positions_by_name"]["joint"])
        return {"geometry_a": "arm", "geometry_b": "body", "distance_m": distance}, []


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


@pytest.mark.parametrize("problem", ["missing_name", "missing_reset", "nonfinite", "drops", "missing_metadata"])
def test_invalid_coverage_fails_before_geometry(tmp_path, problem):
    rows = make_run(tmp_path)
    if problem == "missing_name":
        rows[-1]["positions_by_name"] = {}
    elif problem == "missing_reset":
        for row in rows:
            row["controller_mode"] = "RUNNING"
    elif problem == "nonfinite":
        rows[-1]["velocities_by_name"]["joint"] = None
    elif problem == "drops":
        (tmp_path / "history_metadata.json").write_text('{"dropped_history_records": 2}')
    else:
        (tmp_path / "history_metadata.json").unlink()
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

    class ClearGeometry(FakeGeometry):
        def check(self, row):
            return {"geometry_a": "arm", "geometry_b": "body", "distance_m": .2}, []

    report = replay.review(tmp_path, ClearGeometry)
    assert report["passed"] and not report["errors"]
    assert report["source_modes"] == {"RESETTING": 2, "RUNNING": 2}
    assert "history_summary.json" in report["input_sha256"]
