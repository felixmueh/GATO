"""Exercise the passive recorder with a fake clock and subscription-only ROS node."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np

from gato_tiago.ros_tiago import RIGHT_ARM_JOINTS


def test_passive_observer_records_50hz_without_command_interfaces(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[2] / "tiago_examples/physical_arm/observe_sea_feedback.py"
    spec = importlib.util.spec_from_file_location("passive_observer", path)
    observer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(observer)
    clock = NS(now=1., next_arrival=1., callback=None, destroyed=False, shutdown=False)
    monkeypatch.setattr(observer.time, "monotonic", lambda: clock.now)

    def subscribe(message_type, topic, callback, qos):
        assert topic.endswith("/sea_joint_states")
        assert qos.depth == 1
        clock.callback = callback
        return object()

    # Deliberately has no publisher, service client or action client methods.
    node = NS(create_subscription=subscribe,
              destroy_node=lambda: setattr(clock, "destroyed", True))

    def create_node(name, **kwargs):
        assert kwargs == dict(start_parameter_services=False, enable_rosout=False)
        return node

    def spin_once(node, timeout_sec):
        until = clock.now + timeout_sec
        if clock.next_arrival <= until + 1e-12:
            clock.now = clock.next_arrival
            stamp = 100. + clock.now - 1.
            sec = int(stamp)
            clock.callback(NS(name=list(RIGHT_ARM_JOINTS),
                position=[.2 + clock.now - 1.] * 7,
                absolute_position=[clock.now - 1.] * 7,
                header=NS(stamp=NS(sec=sec, nanosec=round((stamp - sec) * 1e9)))))
            clock.next_arrival += .02
        else:
            clock.now = until

    monkeypatch.setitem(sys.modules, "rclpy", NS(init=lambda: None,
        create_node=create_node, ok=lambda: True, spin_once=spin_once,
        shutdown=lambda: setattr(clock, "shutdown", True)))
    monkeypatch.setitem(sys.modules, "rclpy.qos", NS(
        QoSProfile=lambda **kwargs: NS(**kwargs),
        ReliabilityPolicy=NS(BEST_EFFORT="best_effort")))
    monkeypatch.setitem(sys.modules, "sea_state_msgs.msg", NS(SeaState=object))
    output = tmp_path / "samples.jsonl"
    monkeypatch.setattr(sys, "argv", [str(path), "--duration", ".119", "--output", str(output)])
    assert observer.main() == 0
    assert clock.destroyed and clock.shutdown
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 6
    assert rows[0]["qd_estimated"] is None
    np.testing.assert_allclose([row["qd_estimated"] for row in rows[1:]], 1., atol=1e-10)
    np.testing.assert_allclose(np.diff([row["source_stamp_sec"] for row in rows]), .02)
    summary = json.loads(output.with_suffix(".summary.json").read_text())
    assert summary["callbacks"] == 6 and summary["estimates"] == 5
    np.testing.assert_allclose(summary["receive_intervals"]["median_ms"], 20.)
