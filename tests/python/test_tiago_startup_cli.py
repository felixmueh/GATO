"""Parse actual experiment CLIs without ROS connections or solver construction."""
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("script,base_args", [
    ("tiago_fig8_tracking.py", ["--plant", "tiago_right", "--N", "32"]),
    ("tiago_reach_target.py", ["run", "--N", "32"]),
])
@pytest.mark.parametrize("value", [None, "45", "0", "-1", "nan", "inf"])
def test_startup_timeout_cli(monkeypatch, script, base_args, value):
    monkeypatch.setitem(sys.modules, "bsqp.interface", SimpleNamespace(BSQP=object))
    path = Path(__file__).resolve().parents[2] / "tiago_examples" / script
    module = runpy.run_path(str(path), run_name="startup_cli_test")
    args = [script, *base_args]
    if value is not None:
        args.append(f"--ros-startup-timeout={value}")
    monkeypatch.setattr(sys, "argv", args)
    if value not in {None, "45"}:
        with pytest.raises(SystemExit) as exc:
            module["parse_args"]()
        assert exc.value.code == 2
    else:
        parsed = module["parse_args"]()
        assert parsed.ros_startup_timeout == (30. if value is None else 45.)
        assert parsed.ros_controller_timeout == 8.  # Separate runtime/shutdown budget.
