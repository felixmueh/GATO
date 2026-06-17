#!/usr/bin/env python3
"""Capture Tiago collision-pair review data from a live ROS joint state."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
TIAGO_SRC_DIR = REPO_ROOT / "tiago_src"
if str(TIAGO_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(TIAGO_SRC_DIR))

import pinocchio as pin

from gato_tiago.ros_tiago import DEFAULT_JOINT_STATES_TOPIC, ensure_ros_environment
from gato_tiago.safety_monitor import (
    DEFAULT_LOCKED_JOINTS,
    DEFAULT_URDF_PATH,
    NamedJointState,
    build_tiago_collision_model,
    collision_model_metadata,
    compute_collision_body_speeds,
    compute_pair_distances,
    geometry_objects_json,
    gripper_joint_names,
    state_to_qv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture all SafetyMonitor collision pairs and distances from a "
            "live Tiago /joint_states sample."
        )
    )
    parser.add_argument("--output", type=Path, default=Path("collision_review_data.json"))
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF_PATH)
    parser.add_argument(
        "--package-dir",
        action="append",
        dest="package_dirs",
        default=None,
        help="Package directory for URDF mesh resolution. Defaults to the URDF parent.",
    )
    parser.add_argument("--joint-states-topic", default=DEFAULT_JOINT_STATES_TOPIC)
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument(
        "--lock-grippers",
        action="store_true",
        help="Lock gripper joints at the captured state or neutral if absent.",
    )
    args = parser.parse_args()
    if args.timeout <= 0.0:
        parser.error("--timeout must be positive")
    return args


class _JointStateCapture:
    def __init__(self, topic: str) -> None:
        ensure_ros_environment(allow_reexec=False)
        import rclpy
        from sensor_msgs.msg import JointState

        self.rclpy = rclpy
        self.JointState = JointState
        self.owns_rclpy = False
        if not self.rclpy.ok():
            self.rclpy.init()
            self.owns_rclpy = True
        self.node = self.rclpy.create_node("gato_collision_review_capture")
        self.latest = None
        self.node.create_subscription(JointState, topic, self._on_joint_state, 10)

    def close(self) -> None:
        self.node.destroy_node()
        if self.owns_rclpy:
            self.rclpy.shutdown()

    def _on_joint_state(self, msg) -> None:
        self.latest = msg

    def wait(self, timeout_sec: float):
        deadline = time.monotonic() + timeout_sec
        while self.rclpy.ok() and time.monotonic() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
            if self.latest is not None:
                return self.latest
        raise TimeoutError(f"did not receive JointState within {timeout_sec:.1f}s")


def _stamp_sec(msg) -> float:
    return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9


def _locked_joints(urdf: Path, lock_grippers: bool) -> tuple[str, ...]:
    names = list(DEFAULT_LOCKED_JOINTS)
    if lock_grippers:
        model = pin.buildModelFromUrdf(str(urdf))
        names.extend(gripper_joint_names(model))
    return tuple(names)


def _state_from_msg(msg) -> NamedJointState:
    return NamedJointState.from_sequences(
        msg.name,
        msg.position,
        msg.velocity,
        stamp_sec=_stamp_sec(msg),
    )


def main() -> int:
    args = parse_args()
    package_dirs = (
        tuple(Path(path) for path in args.package_dirs)
        if args.package_dirs is not None
        else (args.urdf.parent,)
    )

    capture = _JointStateCapture(args.joint_states_topic)
    try:
        msg = capture.wait(args.timeout)
    finally:
        capture.close()

    state = _state_from_msg(msg)
    collision_model = build_tiago_collision_model(
        urdf_path=args.urdf,
        package_dirs=package_dirs,
        locked_joint_names=_locked_joints(args.urdf, args.lock_grippers),
        reference_positions=state.position,
    )
    q, qd = state_to_qv(collision_model.model, state)
    pairs = compute_pair_distances(collision_model, q)
    speeds = compute_collision_body_speeds(collision_model, q, qd)

    output = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "urdf_path": str(args.urdf),
        "package_dirs": [str(path) for path in package_dirs],
        "joint_states_topic": args.joint_states_topic,
        "lock_grippers": bool(args.lock_grippers),
        "model": collision_model_metadata(collision_model),
        "state": {
            "source": args.joint_states_topic,
            "stamp_sec": state.stamp_sec,
            "q": [float(value) for value in q],
            "qd": [float(value) for value in qd],
            "positions_by_name": state.position,
            "velocities_by_name": state.velocity,
        },
        "geometry_objects": geometry_objects_json(collision_model),
        "collision_pairs": [report.to_json() for report in pairs],
        "collision_body_speeds": [
            {
                "geometry": report.geometry,
                "link": report.link,
                "parent_joint": report.parent_joint,
                "linear_speed_m_s": report.linear_speed_m_s,
                "angular_speed_rad_s": report.angular_speed_rad_s,
                "radius_m": report.radius_m,
                "speed_bound_m_s": report.speed_bound_m_s,
            }
            for report in speeds
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"collision pairs: {len(pairs)}")
    print(f"minimum distance: {min(report.distance_m for report in pairs):+.6f} m")
    print(f"maximum collision-body speed bound: {max(report.speed_bound_m_s for report in speeds):.6f} m/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
