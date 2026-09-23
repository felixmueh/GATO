"""Startup diagnostics retain the last observed worker stage across empty polls."""
import queue
from types import SimpleNamespace as NS

import pytest

from gato_tiago import ros_tiago
from gato_tiago import tiago_controller_process as execution


@pytest.mark.parametrize("stage", [None, "starting collision safety worker"])
def test_timeout_names_last_observed_stage(monkeypatch, capsys, stage):
    controller = execution.TiagoControllerOrchestrator()
    controller._status_q = queue.Queue()
    clock = NS(now=0.)
    monkeypatch.setattr(execution, "time", NS(
        monotonic=lambda: clock.now,
        sleep=lambda dt: setattr(clock, "now", clock.now + dt)))
    monkeypatch.setattr(ros_tiago, "ensure_ros_environment", lambda **kwargs: None)
    monkeypatch.setattr(execution.atexit, "register", lambda *args: None)

    def start():
        if stage is not None:
            controller._status_q.put(execution.ControllerStatus(
                "RESETTING", startup_stage=stage, startup_stage_started=0.))

    controller._ctx = NS(Process=lambda **kwargs: NS(start=start, exitcode=None))
    with pytest.raises(TimeoutError) as exc:
        controller.initialize(timeout_sec=.2)
    expected = stage or "worker spawn/imports"
    assert expected in str(exc.value)
    assert "0.2s" in str(exc.value)
    output = capsys.readouterr().err
    assert "loading ROS environment" in output
    assert "ready timeout=0.2s" in output
    assert "TIMEOUT" in output and expected in output


def test_setup_logging_flushes_to_stderr(monkeypatch):
    writes = []
    monkeypatch.setattr(execution, "print", lambda *args, **kwargs: writes.append((args, kwargs)),
                        raising=False)
    execution._log_startup(execution.time.monotonic(), "test stage")
    assert writes[0][1] == {"file": execution.sys.stderr, "flush": True}
    assert "test stage" in writes[0][0][0]
