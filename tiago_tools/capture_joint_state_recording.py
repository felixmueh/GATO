#!/usr/bin/env python3
"""Capture ROS joint states as full_joint_state_history.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
TIAGO_SRC_DIR = REPO_ROOT / "tiago_src"
if str(TIAGO_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(TIAGO_SRC_DIR))

from gato_tiago.ros_tiago import DEFAULT_JOINT_STATES_TOPIC, ensure_ros_environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("full_joint_state_history.jsonl"),
    )
    parser.add_argument("--joint-states-topic", default=DEFAULT_JOINT_STATES_TOPIC)
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Number of distinct JointState messages to record.",
    )
    args = parser.parse_args()
    if args.timeout <= 0.0:
        parser.error("--timeout must be positive")
    if args.samples <= 0:
        parser.error("--samples must be positive")
    return args


class _JointStateRecorder:
    def __init__(self, topic: str) -> None:
        ensure_ros_environment(allow_reexec=False)
        import rclpy
        from sensor_msgs.msg import JointState

        self.rclpy = rclpy
        self.owns_rclpy = False
        if not self.rclpy.ok():
            self.rclpy.init()
            self.owns_rclpy = True
        self.node = self.rclpy.create_node("gato_joint_state_recording_capture")
        self.messages = []
        self.node.create_subscription(JointState, topic, self._on_joint_state, 10)

    def close(self) -> None:
        self.node.destroy_node()
        if self.owns_rclpy:
            self.rclpy.shutdown()

    def _on_joint_state(self, msg) -> None:
        if not self.messages or _stamp_sec(msg) != _stamp_sec(self.messages[-1]):
            self.messages.append(msg)

    def wait(self, *, samples: int, timeout_sec: float) -> list:
        deadline = time.monotonic() + timeout_sec
        while self.rclpy.ok() and time.monotonic() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
            if len(self.messages) >= samples:
                return self.messages[:samples]
        raise TimeoutError(
            f"received {len(self.messages)} JointState messages before "
            f"{timeout_sec:.1f}s timeout; expected {samples}"
        )


def _stamp_sec(msg) -> float:
    return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9


def _json_float(value) -> float | None:
    numeric = float(value)
    return numeric if numeric == numeric and abs(numeric) != float("inf") else None


def _row(msg, seq: int) -> dict[str, object]:
    return {
        "source_seq": int(seq),
        "stamp_sec": _stamp_sec(msg),
        "received_monotonic_sec": time.monotonic(),
        "controller_mode": "RECORDED",
        "safety_status": "unchecked",
        "safety_fault": "",
        "safety_message": "",
        "positions_by_name": {
            str(name): _json_float(value)
            for name, value in zip(msg.name, msg.position)
        },
        "velocities_by_name": {
            str(name): _json_float(value)
            for name, value in zip(msg.name, msg.velocity)
        },
    }


def main() -> int:
    args = parse_args()
    recorder = _JointStateRecorder(args.joint_states_topic)
    try:
        messages = recorder.wait(samples=args.samples, timeout_sec=args.timeout)
    finally:
        recorder.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for seq, msg in enumerate(messages, start=1):
            handle.write(json.dumps(_row(msg, seq), sort_keys=True) + "\n")
    print(f"wrote {len(messages)} samples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
