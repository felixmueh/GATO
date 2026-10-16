"""Manual launchers use the public examples, without ROS or native imports."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from gato_tiago import live_example
from gato_tiago.config import TIAGO_REACHING_SOLVER_PARAMS, TIAGO_TRACKING_SOLVER_PARAMS

DIRECTORY = Path(__file__).resolve().parents[2] / "tiago_examples/physical_arm"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, DIRECTORY / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["figure8_24s", "figure8_12s", "reach_goals", "preflight_tracking", "preflight_reaching"])
def test_preview_without_ros_native_or_build(name):
    result = subprocess.run([sys.executable, str(DIRECTORY / (name + ".py")), "--preview"],
                            check=True, capture_output=True, text=True)
    settings = json.loads(result.stdout)
    assert settings["BACKEND"] == "gazebo"
    assert settings["PHYSICAL_PATH_AND_WORKSPACE_REVIEWED"] is False
    assert settings["N_STATE_KNOTS"] == 32


def launcher_dependencies(monkeypatch, tmp_path, name):
    script = load_script(name)
    script.RAW_OUTPUT_ROOT = tmp_path
    calls = []
    receipt = {"config": {"cuda_architecture": "120"}}
    monkeypatch.setattr(sys, "argv", [name + ".py"])
    monkeypatch.setattr(live_example, "check_live_backend", lambda *args, **kwargs: calls.append("backend"))
    monkeypatch.setitem(sys.modules, "native_build", NS(
        verified_receipt=lambda profile: receipt,
        load_profile=lambda profile: calls.append(("native", profile))))
    def parse(options):
        calls.append(("options", options))
        return NS()
    def reach(args, **kwargs):
        calls.append(("reach", args, kwargs))
        return {"goal_outcomes": ["reached"]}
    monkeypatch.setitem(sys.modules, "tiago_reach_target", NS(parse_args=parse, run_experiment=reach))
    monkeypatch.setitem(sys.modules, "tiago_fig8_tracking", NS(
        parse_args=parse, run=lambda args: calls.append(("tracking", args))))
    return script, calls


def test_three_separate_public_reaching_calls(monkeypatch, tmp_path):
    script, calls = launcher_dependencies(monkeypatch, tmp_path, "reach_goals")
    script.JOINT_STATES_TOPIC = "/test/full_body_joint_states"
    script.main()
    reaches = [call for call in calls if call[0] == "reach"]
    assert len(reaches) == 3
    assert len({id(call[1]) for call in reaches}) == 3
    assert [call[2]["goals"] for call in reaches] == [[goal] for goal in script.GOALS_XYZ_M]
    assert all("solver_params" not in call[2] for call in reaches)  # actual reaching defaults
    for _, options in [c for c in calls if c[0] == "options"]:
        assert "--ros-confirm-before-motion" in options and "--ros-record-commands" in options
        assert "--ros-sea-absolute" not in options
        assert options[options.index("--ros-joint-states-topic") + 1] == "/test/full_body_joint_states"
        assert options[options.index("--tiago-start-config") + 1] == "comfortable_high_clearance"


@pytest.mark.parametrize("name,profile,period", [("figure8_24s", "tracking", "24.0"), ("figure8_12s", "tracking", "12.0")])
def test_tracking_calls_public_example_with_real_period(monkeypatch, tmp_path, name, profile, period):
    script, calls = launcher_dependencies(monkeypatch, tmp_path, name)
    script.main()
    assert ("native", profile) in calls
    assert len([c for c in calls if c[0] == "tracking"]) == 1
    options = next(c[1] for c in calls if c[0] == "options")
    assert options[options.index("--figure8-period") + 1] == period
    assert options[options.index("--sim-time") + 1] == period


@pytest.mark.parametrize("name,profile", [("preflight_tracking", "tracking"), ("preflight_reaching", "reaching")])
def test_manual_preflight_uses_its_family_defaults(monkeypatch, tmp_path, name, profile):
    script, calls = launcher_dependencies(monkeypatch, tmp_path, name)
    script.main()
    assert ("native", profile) in calls
    reach = next(c for c in calls if c[0] == "reach")
    assert len(reach[2]["goals"]) == 1
    if profile == "tracking":
        assert reach[2]["solver_params"] == TIAGO_TRACKING_SOLVER_PARAMS
    else:
        assert "solver_params" not in reach[2]


def test_wrong_architecture_does_not_load_native_or_call_example(monkeypatch, tmp_path):
    script, calls = launcher_dependencies(monkeypatch, tmp_path, "reach_goals")
    monkeypatch.setitem(sys.modules, "native_build", NS(
        verified_receipt=lambda profile: {"config": {"cuda_architecture": "87"}},
        load_profile=lambda profile: pytest.fail("must not load mismatched native module")))
    with pytest.raises(RuntimeError, match="selected backend"):
        script.main()
    assert not any(c[0] == "reach" for c in calls)


@pytest.mark.parametrize("sea_absolute", [False, True])
def test_reaching_entrypoint_confirms_before_run_and_preserves_partial_failure(monkeypatch, tmp_path, sea_absolute):
    path = DIRECTORY.parent / "tiago_reach_target.py"
    spec = importlib.util.spec_from_file_location("direct_reach_test", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    from gato_tiago import tiago_controller_process, tiago_mpc_controller
    events, constructed, controller_options = [], [], []
    partial = {"timestamps": [.02], "goal_distances": [.2]}
    failure = RuntimeError("finite plan rejected by controller")
    class MPC:
        def __init__(self, **kwargs):
            constructed.append(kwargs)
            self.last_task_stats = partial
        def run_mpc_goals(self, **kwargs):
            events.append("run")
            assert len(kwargs["goals"]) == 1
            raise failure
    monkeypatch.setattr(tiago_mpc_controller, "MPC_GATO", MPC)
    def controller(**kwargs):
        controller_options.append(kwargs)
        return NS()
    monkeypatch.setattr(tiago_controller_process, "TiagoControllerOrchestrator", controller)
    monkeypatch.setattr(example, "load_model", lambda: NS(nv=7, createData=lambda: None))
    monkeypatch.setattr(example, "tool_position", lambda *args: np.zeros(3))
    monkeypatch.setattr(example, "preflight", lambda *args, **kwargs: {})
    monkeypatch.setattr(live_example, "confirm_live_motion", lambda args: events.append("confirm"))
    def finish(controller, output, stats, error, timeout, **kwargs):
        events.append("finish")
        assert stats is partial and error is failure
        np.testing.assert_array_equal(kwargs["context"]["goals"], [[.4, -.2, -.3]])
        return {}
    monkeypatch.setattr(live_example, "finish_live_recording", finish)
    feedback = (["--ros-sea-absolute", "--ros-sea-state-topic", "/test/sea",
                 "--ros-sea-velocity-filter-tau", ".02"] if sea_absolute else [])
    args = example.parse_args(["run", "--ros-tiago", "--expr-dir", str(tmp_path), *feedback])
    with pytest.raises(RuntimeError) as exc:
        example.run_experiment(args, goals=[[.4, -.2, -.3]])
    assert exc.value is failure
    assert events == ["confirm", "run", "finish"]
    assert controller_options[0]["sea_absolute"] is sea_absolute
    assert controller_options[0]["sea_state_topic"] == ("/test/sea" if sea_absolute else
            "/arm_right_sea_state_broadcaster_controller/sea_joint_states")
    assert controller_options[0]["sea_velocity_filter_tau_sec"] == (.02 if sea_absolute else .01)
    assert constructed[0]["solver_params"] == TIAGO_REACHING_SOLVER_PARAMS
    assert constructed[0]["final_approach"] is False


@pytest.mark.parametrize("name", ["figure8_24s", "figure8_12s", "reach_goals", "preflight_tracking", "preflight_reaching"])
def test_manual_launchers_forward_sea_opt_in(monkeypatch, tmp_path, name):
    script, calls = launcher_dependencies(monkeypatch, tmp_path, name)
    monkeypatch.setattr(sys, "argv", [name + ".py", "--ros-sea-absolute"])
    script.main()
    for _, options in [c for c in calls if c[0] == "options"]:
        assert "--ros-sea-absolute" in options
        assert options[options.index("--ros-sea-state-topic") + 1] == script.SEA_STATE_TOPIC
        assert options[options.index("--ros-sea-velocity-filter-tau") + 1] == "0.01"
