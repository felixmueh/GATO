"""Direct-example cleanup and reference options; no ROS or GATO native loads."""
import argparse
import json
from pathlib import Path
import runpy
from types import SimpleNamespace as NS
import sys

import numpy as np
import pytest

from gato_tiago import live_example as live
from gato_tiago.references import tiago_horizontal_figure8


def controller(tmp_path, *, close_error=False, state_error=False, started=True):
    calls = []
    metadata = tmp_path / "worker_metadata.json"
    metadata.write_text('{"record_count": 3, "dropped_history_records": 0}')
    def close(**kwargs):
        calls.append("close")
        assert not (tmp_path / "output").exists()
        if close_error:
            raise RuntimeError("restoration failed")
    def export(name, result=None, fail=False):
        def write(path):
            assert calls[0] == "close"
            calls.append(name)
            if fail:
                raise RuntimeError("state export failed")
            path.write_text("{}\n")
            return result
        return write
    value = NS(_process=object() if started else None, close=close,
               record_command_events=True, _history_metadata_path=metadata,
               collision_safety=NS(min_distance_m=.04, joint_position_margin_rad=0.,
                    joint_velocity_scale=1., urdf_path=Path("collision.urdf"),
                    blacklist_path=Path("blacklist.json")),
               write_state_history_csv=export("state", {"samples": 3}, state_error),
               write_full_state_history_jsonl=export("full"),
               write_command_history_jsonl=export("commands", {"recording_complete": True, "errors": []}),
               write_lifecycle_events_json=export("lifecycle", {"recording_complete": True, "errors": []}))
    return value, calls


def test_close_precedes_independent_exports_and_numpy_stats_survive(monkeypatch, tmp_path):
    monkeypatch.setattr(live.subprocess, "check_output", lambda *args, **kwargs: "commit\n")
    ctl, calls = controller(tmp_path)
    result = live.finish_live_recording(
        ctl, tmp_path / "output", {"timestamps": np.array([0., .03]), "native_us": np.int32(5)},
        None, 8., context={"args": {"sim_time": 12., "manual_run_index": tmp_path / "index.jsonl"},
                          "solver_params": {"qd_cost": .01}})
    assert calls == ["close", "state", "full", "commands", "lifecycle"]
    assert result == {"handback_confirmed": True, "errors": [], "interrupted": False,
                      "task_success": None, "controller_state_history": {"samples": 3}}
    out = tmp_path / "output"
    assert json.loads((out / "task_stats.json").read_text()) == {"timestamps": [0., .03], "native_us": 5}
    assert json.loads((out / "history_metadata.json").read_text())["record_count"] == 3
    cfg = json.loads((out / "resolved_config.json").read_text())
    assert cfg["DURATION_S"] == 12. and cfg["goal_xyz_m"] is None
    assert cfg["COLLISION_URDF_PATH"] == "collision.urdf"
    assert cfg["COLLISION_MARGIN_M"] == .04
    assert json.loads((tmp_path / "index.jsonl").read_text())["handback_confirmed"] is True
    assert len(json.loads((out / "provenance.json").read_text())["config_sha256"]) == 64


def test_secondary_cleanup_export_errors_do_not_replace_task_exception(monkeypatch, tmp_path):
    monkeypatch.setattr(live.subprocess, "check_output", lambda *args, **kwargs: "commit\n")
    ctl, calls = controller(tmp_path, close_error=True, state_error=True)
    error = None
    with pytest.raises(ValueError, match="original task failure"):
        try:
            try:
                raise ValueError("original task failure")
            except BaseException as exc:
                error = exc
                raise
        finally:
            live.finish_live_recording(ctl, tmp_path / "output", {"samples": np.array([1])}, error, 8.)
    assert calls == ["close", "state", "full", "commands", "lifecycle"]
    result = json.loads((tmp_path / "output/result.json").read_text())
    assert result["handback_confirmed"] is False
    assert {item["stage"] for item in result["errors"]} == {"task", "handback", "write_state_history_csv"}
    assert (tmp_path / "output/task_stats.json").is_file()


def test_cleanup_failure_raises_after_other_artifacts_are_saved(monkeypatch, tmp_path):
    monkeypatch.setattr(live.subprocess, "check_output", lambda *args, **kwargs: "commit\n")
    ctl, calls = controller(tmp_path, state_error=True)
    with pytest.raises(RuntimeError, match="state export failed"):
        live.finish_live_recording(ctl, tmp_path / "output", {}, None, 8.)
    assert calls[-2:] == ["commands", "lifecycle"]
    assert (tmp_path / "output/result.json").is_file()


def test_no_worker_cannot_claim_confirmed_handback(monkeypatch, tmp_path):
    monkeypatch.setattr(live.subprocess, "check_output", lambda *args, **kwargs: "commit\n")
    ctl, _ = controller(tmp_path, started=False)
    result = live.finish_live_recording(ctl, tmp_path / "output", {"goal_outcomes": ["reached"]},
                                        None, 8., context={"goals": np.array([[1., 2., 3.]])})
    assert result["handback_confirmed"] is False and result["task_success"] is True
    cfg = json.loads((tmp_path / "output/resolved_config.json").read_text())
    assert cfg["goal_xyz_m"] == [1., 2., 3.] and "DURATION_S" not in cfg


def test_confirmation_defaults_off_and_decline_prevents_motion(monkeypatch, capsys):
    parser = argparse.ArgumentParser()
    live.add_live_arguments(parser)
    args = parser.parse_args([])
    assert args.ros_reset_position_tolerance is None and args.ros_record_commands is False
    monkeypatch.setattr("builtins.input", lambda *args: pytest.fail("ordinary callers must not prompt"))
    live.confirm_live_motion(args)
    args = parser.parse_args(["--ros-confirm-before-motion", "--ros-reset-position-tolerance", ".05"])
    monkeypatch.setattr("builtins.input", lambda *args: "decline")
    with pytest.raises(RuntimeError, match="declined"):
        live.confirm_live_motion(args)
    assert '"ros_reset_position_tolerance": 0.05' in capsys.readouterr().out


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_live_arrival_options_must_be_bounded(value):
    parser = argparse.ArgumentParser()
    live.add_live_arguments(parser)
    with pytest.raises(SystemExit):
        parser.parse_args([f"--ros-reset-arrival-timeout={value}"])


def test_backend_checks_profile_before_native_load(monkeypatch):
    monkeypatch.setenv("GATO_ROS_PROFILE", "simulation")
    monkeypatch.setenv("ROS_DOMAIN_ID", "1")
    monkeypatch.setenv("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp")
    live.check_live_backend("gazebo")
    with pytest.raises(RuntimeError, match="matching"):
        live.check_live_backend("physical_robot", physical_reviewed=True)
    monkeypatch.setenv("GATO_ROS_PROFILE", "tiago")
    monkeypatch.setenv("ROS_DOMAIN_ID", "2")
    monkeypatch.setattr(live.platform, "machine", lambda: "aarch64")
    with pytest.raises(RuntimeError, match="reviewed"):
        live.check_live_backend("physical_robot")
    live.check_live_backend("physical_robot", physical_reviewed=True)


def test_period_changes_speed_without_changing_geometry_or_defaults():
    slow, slow_meta = tiago_horizontal_figure8(.03, [0., 0., 0.], cycles=1)
    fast, fast_meta = tiago_horizontal_figure8(.03, [0., 0., 0.], cycles=1, period=12.)
    np.testing.assert_allclose(slow.reshape(-1, 6)[::2], fast.reshape(-1, 6), atol=1e-14)
    assert slow_meta["period_sec"] == 24. and fast_meta["period_sec"] == 12.


@pytest.mark.parametrize("period", [0., -1., np.nan, np.inf])
def test_invalid_reference_period_is_rejected(period):
    with pytest.raises(ValueError, match="period must be finite and positive"):
        tiago_horizontal_figure8(.03, [0., 0., 0.], period=period)


def test_figure8_parse_args_accepts_direct_wrapper_options(monkeypatch):
    monkeypatch.setitem(sys.modules, "bsqp.interface", NS(BSQP=object))
    path = Path(__file__).resolve().parents[2] / "tiago_examples/tiago_fig8_tracking.py"
    module = runpy.run_path(str(path), run_name="fig8_live_options_test")
    args = module["parse_args"](["--plant", "tiago_right", "--figure8-period", "12",
        "--ros-reset-position-tolerance", ".05", "--ros-record-commands", "--ros-confirm-before-motion"])
    assert args.figure8_period == 12. and args.ros_reset_position_tolerance == .05
    assert args.ros_record_commands and args.ros_confirm_before_motion
