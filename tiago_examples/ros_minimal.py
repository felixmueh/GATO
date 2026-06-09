#!/usr/bin/env python3
"""Minimal ROS 2 preflight for TIAGo right-arm experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
TIAGO_SRC_DIR = REPO_ROOT / "tiago_src"
for path in (TIAGO_SRC_DIR, PYTHON_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gato_tiago.ros_tiago import TiagoRightArmClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read the PAL TIAGo right-arm joint state and publish a tiny "
            "position trajectory command."
        )
    )
    parser.add_argument(
        "--nudge",
        type=float,
        default=0.02,
        help="Small radian offset applied to arm_right_7_joint.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Trajectory duration in seconds.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=8.0,
        help="Seconds to wait for ROS discovery and joint state messages.",
    )
    args = parser.parse_args()
    if args.duration <= 0.0:
        parser.error("--duration must be positive")
    if args.timeout <= 0.0:
        parser.error("--timeout must be positive")
    return args


def main() -> int:
    args = parse_args()
    with TiagoRightArmClient(node_name="gato_ros_minimal") as arm:
        state = arm.read_state(timeout_sec=args.timeout)
        target = state.q.copy()
        target[-1] += args.nudge

        arm.node.get_logger().info(
            "current right-arm positions: "
            + ", ".join(f"{value:+.3f}" for value in state.q)
        )
        arm.node.get_logger().info(
            f"commanding arm_right_7_joint by {args.nudge:+.3f} rad"
        )
        arm.publish_position_trajectory(target, duration_sec=args.duration)
        arm.spin_once(timeout_sec=0.1)
        arm.node.get_logger().info("minimal joint command published")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
