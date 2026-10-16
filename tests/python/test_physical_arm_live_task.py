"""CPU checks of manual-task failure retention and motion boundaries."""
import importlib.util
import json
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

DIRECTORY = Path(__file__).resolve().parents[2] / "tiago_examples/physical_arm"
SPEC = importlib.util.spec_from_file_location("manual_live_task_test", DIRECTORY / "live_task.py")
live = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(live)


def settings():
    return runpy.run_path(str(DIRECTORY / "preflight_reaching.py"))


def test_preview_needs_no_ros_profile_or_native_artifact(monkeypatch):
    monkeypatch.delenv("GATO_ROS_PROFILE", raising=False)
    monkeypatch.delenv("ROS_DOMAIN_ID", raising=False)
    for name in ("preflight_tracking", "preflight_reaching", "figure8_24s", "figure8_12s", "reach_goals"):
        result = subprocess.run([sys.executable, str(DIRECTORY / (name + ".py")), "--preview"],
                                check=True, capture_output=True, text=True)
        cfg = json.loads(result.stdout)
        assert cfg["BACKEND"] == "gazebo"
        assert cfg["PHYSICAL_PATH_AND_WORKSPACE_REVIEWED"] is False


@pytest.mark.parametrize("backend,profile,domain", [
    ("gazebo", "tiago", "2"), ("physical_robot", "simulation", "1")])
def test_wrong_ros_profile_cannot_reach_motion(backend, profile, domain, monkeypatch):
    monkeypatch.setenv("GATO_ROS_PROFILE", profile)
    monkeypatch.setenv("ROS_DOMAIN_ID", domain)
    with pytest.raises(RuntimeError, match="Source tiago_tools"):
        live.check_backend(backend)


def test_missing_preflight_stops_before_confirmation_or_controller(tmp_path, monkeypatch):
    namespace = settings()
    namespace["IS_PREFLIGHT"] = False
    monkeypatch.setattr(live, "ROOT", tmp_path)
    monkeypatch.setattr(live, "check_backend", lambda _: None)
    monkeypatch.setattr(live, "family_identity", lambda *_: "current")
    monkeypatch.setitem(sys.modules, "native_build", NS(load_profile=lambda _: {"module": {}, "config": {"knots": 32}}))
    monkeypatch.setattr(live.subprocess, "check_output", lambda *args, **kwargs: "test-source\n")
    monkeypatch.setattr(sys, "argv", ["task.py"])
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("must not confirm motion"))
    monkeypatch.setattr(live, "execute", lambda *_: pytest.fail("must not construct controller"))
    with pytest.raises(RuntimeError, match="matching passed evidence"):
        live.main(namespace)


def test_preflight_source_identity_changes_with_target_or_criteria(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "__file__", str(tmp_path / "live_task.py"))
    preflight = tmp_path / "preflight_reaching.py"
    preflight.write_text("GOALS_XYZ_M = [[.4, -.2, -.3]]\nPOSITION_TOLERANCE_M = .05\n")
    original = live.preflight_script_identity({"NATIVE_PROFILE": "reaching"})
    preflight.write_text(preflight.read_text().replace(".05", ".15"))
    assert live.preflight_script_identity({"NATIVE_PROFILE": "reaching"}) != original
    preflight.write_text(preflight.read_text().replace(".4", ".5"))
    assert live.preflight_script_identity({"NATIVE_PROFILE": "reaching"}) != original


def test_nonfinite_solve_is_retained_and_not_returned_for_publication(tmp_path):
    delegate = NS(XU_B=np.zeros((1, 4)), stats={"final_merit": [np.nan]},
                  solve=lambda *args: (np.ones((1, 4)), 15))
    recorder = live.RecordedSolver(delegate, tmp_path)
    with pytest.raises(FloatingPointError, match="Nonfinite returned"):
        recorder.solve(np.array([[.3]]), np.array([1.]), np.zeros((1, 4)))
    recorder.stream.close()
    failed = np.load(tmp_path / "failed_solve_000000.npz")
    np.testing.assert_array_equal(failed["measured_state"], [[.3]])
    row = json.loads((tmp_path / "solves.jsonl").read_text())
    assert row["error"].startswith("FloatingPointError")
    assert row["native_time_us"] == 15 and row["wrapper_wall_s"] >= 0


def test_latest_successful_plan_can_be_retained_after_controller_rejection(tmp_path):
    plan = np.array([[0., 27., 2., 3.]])
    delegate = NS(XU_B=np.zeros((1, 4)), stats={"final_merit": [1.]},
                  solve=lambda *args: (plan, 15))
    recorder = live.RecordedSolver(delegate, tmp_path)
    recorder.solve(np.array([[.3]]), np.array([1.]), np.zeros((1, 4)))
    plan[:] = -1  # The retained last call must survive later warm-start mutations.
    recorder.save_last_call("last_solve_before_failure.npz")
    recorder.stream.close()
    retained = np.load(tmp_path / "last_solve_before_failure.npz")
    np.testing.assert_array_equal(retained["output"], [[0., 27., 2., 3.]])
    np.testing.assert_array_equal(retained["measured_state"], [[.3]])


def test_import_model_failure_still_writes_result(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pinocchio", NS(buildModelFromUrdf=lambda _: (_ for _ in ()).throw(RuntimeError("model unavailable"))))
    result = live.execute(live.public_settings(settings()), tmp_path, [0, 0, 0])
    assert not result["task_success"] and not result["handback_confirmed"]
    assert "model unavailable" in result["errors"][0]["error"]
    assert json.loads((tmp_path / "result.json").read_text()) == result


def test_frequency_summary_cannot_replace_missing_worker_completeness_receipt(tmp_path):
    rows = [{"q": [0.] * 7, "qd": [0.] * 7, "applied_tau": [0.] * 7} for _ in range(2)]
    (tmp_path / "full_joint_state_history.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    live.write_json(tmp_path / "history_summary.json", {"dropped_history_records": 0})
    live.write_json(tmp_path / "history_metadata.json", {"history_complete": False})
    commands = [{"sequence": i + 1, "tau_Nm": [12.] * 7, "publication_succeeded": True}
                for i in range(3)]
    (tmp_path / "command_history.jsonl").write_text("\n".join(json.dumps(row) for row in commands))
    live.write_json(tmp_path / "command_history_metadata.json", {"recording_complete": True, "record_count": 3})
    model = NS(lowerPositionLimit=-np.ones(7), upperPositionLimit=np.ones(7), velocityLimit=np.ones(7))
    physical, observed = live.observed_physical(tmp_path, model, {"MAX_TORQUE_NM": 26.})
    assert not physical and not observed["history_complete"]
    live.write_json(tmp_path / "history_metadata.json", {"record_count": 2, "dropped_history_records": 0})
    physical, observed = live.observed_physical(tmp_path, model, {"MAX_TORQUE_NM": 26.})
    assert physical and observed["history_complete"]
    assert observed["max_published_torque_Nm"] == 12.  # Snapshot torques above were zero.
    assert observed["recorded_publications"] == 3


@pytest.mark.parametrize("started", [False, True])
def test_exception_closes_before_exports_and_never_invents_handback(tmp_path, monkeypatch, started):
    calls = []
    class Controller:
        def __init__(self, **kwargs): self._process = None
        def initialize(self, **kwargs): self._process = object()
        def close(self, **kwargs): calls.append("close")
        def write_state_history_csv(self, path):
            calls.append("csv")
            path.write_text("")
            return {"record_count": 0, "dropped_history_records": 0}
        def write_full_state_history_jsonl(self, path):
            calls.append("jsonl")
            path.write_text("")
    class MPC:
        def __init__(self, *args, **kwargs):
            if not started: raise RuntimeError("constructor failed")
            self.solver = NS(XU_B=np.zeros((1, 4)))
        def run_mpc_goals(self, **kwargs):
            kwargs["controller"].initialize()
            self.last_task_stats = {"timestamps": [0., .1], "goal_distances": [.1, .08]}
            raise KeyboardInterrupt()
    monkeypatch.setitem(sys.modules, "pinocchio", NS(buildModelFromUrdf=lambda _: NS()))
    monkeypatch.setitem(sys.modules, "gato_tiago.tiago_controller_process", NS(TiagoControllerOrchestrator=Controller))
    monkeypatch.setitem(sys.modules, "gato_tiago.tiago_mpc_controller", NS(MPC_GATO=MPC))
    result = live.execute(live.public_settings(settings()), tmp_path, [0, 0, 0])
    assert calls == ["close", "csv", "jsonl"]
    assert result["handback_confirmed"] is started
    assert result["interrupted"] is started
    assert not result["physical_pass"]
    if started:
        assert json.loads((tmp_path / "task_stats.json").read_text())["timestamps"] == [0., .1]
