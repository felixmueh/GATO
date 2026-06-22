#!/usr/bin/env python3
"""Restore the PAL TIAGo right arm to the local default simulator posture."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
TIAGO_SRC_DIR = REPO_ROOT / "tiago_src"
for path in (TIAGO_SRC_DIR, PYTHON_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gato_tiago.config import TIAGO_RIGHT_DEFAULT_START_CONFIG, TIAGO_RIGHT_START_CONFIGS
from gato_tiago.ros_tiago import TiagoRightArmClient


def wait_for_position(
    arm: TiagoRightArmClient,
    target: np.ndarray,
    *,
    timeout_sec: float,
    tolerance_rad: float,
) -> tuple[bool, float]:
    deadline = time.monotonic() + timeout_sec
    last_error = float("inf")
    while time.monotonic() < deadline:
        state = arm.read_state(timeout_sec=1.0)
        last_error = float(np.max(np.abs(state.q.astype(np.float64) - target)))
        if last_error <= tolerance_rad:
            return True, last_error
        time.sleep(0.05)
    return False, last_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--tolerance", type=float, default=0.03)
    parser.add_argument(
        "--target",
        nargs=7,
        type=float,
        default=TIAGO_RIGHT_START_CONFIGS[TIAGO_RIGHT_DEFAULT_START_CONFIG].tolist(),
    )
    args = parser.parse_args()
    if args.duration <= 0.0:
        parser.error("--duration must be positive")
    if args.timeout <= 0.0:
        parser.error("--timeout must be positive")
    if args.tolerance <= 0.0:
        parser.error("--tolerance must be positive")
    return args


def main() -> int:
    args = parse_args()
    target = np.asarray(args.target, dtype=np.float64)
    with TiagoRightArmClient(node_name="gato_tiago_restore_default") as arm:
        arm.node.get_logger().info("switching to default PAL right-arm controllers")
        arm.switch_to_default_control(timeout_sec=args.timeout)
        arm.node.get_logger().info(
            "commanding default posture: "
            + ", ".join(f"{value:+.3f}" for value in target)
        )
        arm.publish_position_trajectory(target, duration_sec=args.duration)
        arm.spin_once(timeout_sec=0.1)
        reached, error = wait_for_position(
            arm,
            target,
            timeout_sec=args.timeout,
            tolerance_rad=args.tolerance,
        )

    print(f"target: {' '.join(f'{value:+.6f}' for value in target)}")
    print(f"max_error_rad: {error:.6f}")
    print(f"reached: {reached}")
    return 0 if reached else 1


if __name__ == "__main__":
    raise SystemExit(main())
