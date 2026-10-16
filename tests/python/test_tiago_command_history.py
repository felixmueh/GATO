"""Publication events and measured handback coverage; no ROS, GPU or robot."""
import json
import queue
from types import SimpleNamespace as NS

import numpy as np
import pytest

from gato_tiago import ros_tiago as ros
from gato_tiago import tiago_controller_process as execution
from gato_tiago.safety_monitor import CollisionSafetySettings


def run_worker(monkeypatch, *, fault=None, record=True):
    controller = execution.TiagoControllerOrchestrator(record_command_events=record)
    clock = NS(now=0., last_update=0., seq=1, stopped=False, active=False,
               reset_at=None, default_after_task=False, regular=0)
    actual_publications, states, calls = [], [], []
    statuses, trajectories = queue.Queue(), queue.Queue()
    monkeypatch.setattr(execution, "time", NS(
        monotonic=lambda: clock.now, perf_counter=lambda: clock.now,
        sleep=lambda dt: setattr(clock, "now", clock.now + max(0., dt))))
    original_put = execution._put_latest
    def put(q, item):
        original_put(q, item)
        if isinstance(item, execution.ControllerStatus) and item.finished_request_id is not None:
            clock.stopped = True
    monkeypatch.setattr(execution, "_put_latest", put)

    def state(q):
        return NS(q=np.full(7, q), qd=np.zeros(7), seq=clock.seq,
                  stamp_sec=100. + clock.now, received_monotonic_sec=clock.now,
                  age_sec=0., joint_positions={name: q for name in ros.RIGHT_ARM_JOINTS},
                  joint_velocities={name: 0. for name in ros.RIGHT_ARM_JOINTS})
    latest = NS(value=state(.2))
    states.append(latest.value)
    class Arm:
        joint_names = ros.RIGHT_ARM_JOINTS
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def latest_state(self): return latest.value
        def spin_once(self, timeout_sec=0.):
            clock.now += max(timeout_sec, .001)
            if clock.default_after_task:
                assert not clock.active  # Endpoint wait happens after effort handback.
                if fault == "missing_endpoint":
                    return
            # Deliberately fewer state observations than effort publications.
            if clock.now - clock.last_update >= .04:
                clock.last_update = clock.now
                clock.seq += 1
                at_reset = clock.reset_at is not None and clock.now - clock.reset_at >= .08
                latest.value = state(0. if at_reset else .2)
                states.append(latest.value)
        def read_state(self, **kwargs):
            self.spin_once(timeout_sec=.01)
            return latest.value
        def switch_to_default_control(self, **kwargs):
            calls.append("default")
            if clock.active and fault == "switch":
                raise RuntimeError("default switch failed")
            if clock.active:
                clock.default_after_task = True
            clock.active = False
        def publish_position_trajectory(self, *args, **kwargs):
            clock.reset_at = clock.now
        def configure_runtime_effort_controller(self, **kwargs):
            trajectories.put(execution.TorqueTrajectory(np.ones((100, 7)), .01))
        def switch_to_effort_control(self, **kwargs):
            clock.active = True
            calls.append("activate")
        def publish_effort(self, tau):
            assert not controller._command_history_path.exists()
            assert not controller._command_history_metadata_path.exists()
            assert not controller._lifecycle_events_path.exists()
            actual_publications.append(np.array(tau, copy=True))
            clock.now += .0001
            zero = bool(np.all(np.asarray(tau) == 0))
            if fault == "prime" and not clock.active and len(actual_publications) == 1:
                raise RuntimeError("prime publication failed")
            if zero and fault == "zero":
                raise RuntimeError("zero publication failed")
            if clock.active and not zero:
                clock.regular += 1
                if clock.regular == 3 and fault == "publication":
                    raise RuntimeError("task publication failed")
                if clock.regular == 10:
                    trajectories.put(execution.FinishControl(1))
    monkeypatch.setattr(ros, "TiagoRightArmClient", Arm)
    error = None
    try:
        execution._controller_main(
            target_hz=125., reset_q=np.zeros(7), reset_duration_sec=.08,
            reset_position_tolerance_rad=.05, reset_velocity_l1_tolerance_rad_s=.1,
            reset_arrival_timeout_sec=.2, stale_timeout_sec=.1,
            max_abs_torque=26., clamp_torque=False, record_command_events=record,
            collision_safety=CollisionSafetySettings(enabled=False), history_max_records=1000,
            shared_state=controller._state, trajectory_q=trajectories, status_q=statuses,
            history_path=controller._history_path, history_metadata_path=controller._history_metadata_path,
            stop_event=NS(is_set=lambda: clock.stopped))
    except RuntimeError as exc:
        error = exc
    return NS(controller=controller, actual=actual_publications, observed=states,
              calls=calls, clock=clock, error=error, statuses=statuses)


def export(result, tmp_path):
    metadata = result.controller.write_command_history_jsonl(tmp_path / "commands.jsonl")
    lifecycle = result.controller.write_lifecycle_events_json(tmp_path / "lifecycle.json")
    rows = [json.loads(line) for line in (tmp_path / "commands.jsonl").read_text().splitlines()]
    history = [json.loads(line) for line in result.controller._full_state_history_path.read_text().splitlines()]
    assert json.loads((tmp_path / "command_history_metadata.json").read_text()) == metadata
    return rows, metadata, lifecycle, history


def test_every_publication_survives_repeated_source_stamps(monkeypatch, tmp_path):
    result = run_worker(monkeypatch)
    rows, metadata, lifecycle, history = export(result, tmp_path)
    assert result.error is None
    assert metadata["recording_complete"] is True and metadata["recording_enabled"] is True
    assert metadata["record_count"] == len(result.actual) == len(rows) == 16
    assert metadata["failed_publications"] == 0 and metadata["errors"] == []
    assert [row["sequence"] for row in rows] == list(range(1, len(rows) + 1))
    assert [row["phase"] for row in rows] == ["prime"] + ["task"] * 10 + ["handback_zero"] * 5
    assert len({row["source_seq"] for row in rows}) < len(rows)
    for row, tau in zip(rows, result.actual):
        np.testing.assert_array_equal(row["tau_Nm"], tau)
        assert row["publication_succeeded"] and row["error"] is None
        assert row["publish_begin_monotonic_sec"] <= row["publish_end_monotonic_sec"]
    assert rows[0]["knot_index"] == 0
    assert all(row["trajectory_id"] == 0 for row in rows)
    assert all(row["knot_index"] is None for row in rows[-5:])
    assert lifecycle["errors"] == [] and lifecycle["recording_complete"] is True
    events = lifecycle["events"]
    names = [event["event"] for event in events]
    assert names[:4] == ["reset_command", "reset_published", "reset_verified", "first_task_effort"]
    assert names.count("handback_request") == names.count("handback_confirmed") == names.count("handback_observed") == 2
    for event in events:
        matches = [row for row in history if row["source_seq"] == event["source_seq"]
                   and row["stamp_sec"] == event["source_stamp_sec"]
                   and row["received_monotonic_sec"] == event["source_received_monotonic_sec"]]
        assert matches, f"marker has no actual source observation: {event}"
        assert 0. <= event["monotonic_sec"] - event["source_received_monotonic_sec"] <= .1
        if event["event"] == "first_task_effort":
            assert any(row["controller_mode"] == "RUNNING" for row in matches)
        if event["event"] == "handback_observed":
            confirmed = max(e["monotonic_sec"] for e in events if e["event"] == "handback_confirmed"
                            and e["monotonic_sec"] <= event["monotonic_sec"])
            assert event["source_received_monotonic_sec"] >= confirmed
            assert any(row["controller_mode"] == "RESTORED" for row in matches)
    assert history[-1]["controller_mode"] == "RESTORED"
    shared = execution._read_state(result.controller._state)
    assert shared.controller_mode == "RESTORED"
    assert shared.source_seq == history[-1]["source_seq"]


@pytest.mark.parametrize("fault", ["prime", "publication", "zero", "switch"])
def test_failed_publications_and_shutdown_preserve_original_error(monkeypatch, tmp_path, fault):
    result = run_worker(monkeypatch, fault=fault)
    rows, metadata, lifecycle, history = export(result, tmp_path)
    assert result.error is not None
    expected = {"prime": "prime publication failed", "publication": "task publication failed",
                "zero": "zero publication failed", "switch": "default switch failed"}[fault]
    assert str(result.error) == expected
    assert metadata["recording_complete"] is True
    assert metadata["record_count"] == len(result.actual) == len(rows)
    failures = [row for row in rows if not row["publication_succeeded"]]
    assert bool(failures) == (fault != "switch")
    if failures:
        assert all(expected in row["error"] for row in failures)
        assert metadata["errors"]
    if fault in ("zero", "switch"):
        assert lifecycle["errors"]
        assert execution._get_latest(result.statuses).mode == "ERROR_RESTORE_FAILED"
    else:
        assert lifecycle["events"][-1]["event"] == "handback_observed"
        assert execution._get_latest(result.statuses).mode == "ERROR_RESTORE"
    if fault == "publication":
        assert rows[-5:][0]["phase"] == "handback_zero"
        assert all(row["publication_succeeded"] for row in rows[-5:])


def test_missing_post_handback_state_is_explicit_unknown_coverage(monkeypatch, tmp_path):
    result = run_worker(monkeypatch, fault="missing_endpoint")
    rows, metadata, lifecycle, history = export(result, tmp_path)
    assert result.error is None  # Default-control restoration itself succeeded.
    assert execution._get_latest(result.statuses).mode == "RESTORED"
    assert lifecycle["errors"] and all("no fresh measured state" in error for error in lifecycle["errors"])
    assert sum(event["event"] == "handback_observation_failed" for event in lifecycle["events"]) == 2
    assert not any(event["event"] == "handback_observed" for event in lifecycle["events"])
    assert not any(row["controller_mode"] == "RESTORED" for row in history)
    assert metadata["recording_complete"] and len(rows) == len(result.actual)


def test_disabled_recording_keeps_existing_control_path(monkeypatch, tmp_path):
    result = run_worker(monkeypatch, record=False)
    rows, metadata, lifecycle, history = export(result, tmp_path)
    assert result.error is None and result.actual
    assert rows == [] and lifecycle["events"] == []
    assert metadata["recording_enabled"] is False and metadata["recording_complete"] is False
    assert lifecycle["recording_enabled"] is False and lifecycle["recording_complete"] is False


def test_export_does_not_fabricate_missing_worker_receipts(tmp_path):
    controller = execution.TiagoControllerOrchestrator(record_command_events=True)
    with pytest.raises(FileNotFoundError):
        controller.write_command_history_jsonl(tmp_path / "commands.jsonl")
    with pytest.raises(FileNotFoundError):
        controller.write_lifecycle_events_json(tmp_path / "lifecycle.json")
    assert not (tmp_path / "commands.jsonl").exists()
    assert not (tmp_path / "lifecycle.json").exists()
