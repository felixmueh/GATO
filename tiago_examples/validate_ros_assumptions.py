#!/usr/bin/env python3
"""Read-only hardware preflight for the Tiago ROS controller path."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any, Callable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TIAGO_SRC_DIR = REPO_ROOT / "tiago_src"
if str(TIAGO_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(TIAGO_SRC_DIR))

from gato_tiago.ros_tiago import (  # noqa: E402
    RIGHT_ARM_JOINTS,
    TiagoRightArmClient,
    ensure_ros_environment,
    validate_controller_manager_interfaces,
)


STALE_TIMEOUT_SEC = 0.1
FORWARD_CONTROLLER_TYPE = "forward_command_controller/ForwardCommandController"
DEFAULT_CONTROLLERS = (
    "arm_right_controller",
    "arm_right_gravity_compensation_controller",
)


class Checks:
    def __init__(self) -> None:
        self.failures = 0

    def run(self, description: str, check: Callable[[], str | None]) -> bool:
        print(f"- Checking {description}...", flush=True)
        try:
            detail = check()
        except Exception as exc:
            self.failures += 1
            print(f"  Fail: {exc}", flush=True)
            return False
        print(f"  OK{f' ({detail})' if detail else ''}", flush=True)
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--samples", type=int, default=25)
    args = parser.parse_args()
    if args.timeout <= 0.0:
        parser.error("--timeout must be positive")
    if args.samples < 2:
        parser.error("--samples must be at least 2")
    return args


def _load_ros() -> SimpleNamespace:
    ensure_ros_environment(allow_reexec=False)
    try:
        validate_controller_manager_interfaces()
        from controller_manager_msgs.srv import (
            ListControllers,
            ListControllerTypes,
            ListHardwareInterfaces,
        )
        from sensor_msgs.msg import JointState
    except ImportError as exc:
        raise RuntimeError(
            "required ROS 2 Python packages are not importable; source the "
            "robot ROS environment or set GATO_ROS_SETUP"
        ) from exc
    return SimpleNamespace(
        JointState=JointState,
        ListControllers=ListControllers,
        ListControllerTypes=ListControllerTypes,
        ListHardwareInterfaces=ListHardwareInterfaces,
    )


def _collect_joint_states(
    arm: TiagoRightArmClient,
    joint_state_type: type,
    *,
    count: int,
    timeout_sec: float,
) -> list[tuple[Any, float]]:
    samples: list[tuple[Any, float]] = []

    def callback(msg: Any) -> None:
        samples.append((msg, time.monotonic()))

    subscription = arm.node.create_subscription(
        joint_state_type,
        arm.joint_states_topic,
        callback,
        10,
    )
    try:
        deadline = time.monotonic() + timeout_sec
        while arm.rclpy.ok() and time.monotonic() < deadline:
            arm.spin_once(timeout_sec=0.05)
            if len(samples) >= count:
                return samples[:count]
    finally:
        arm.node.destroy_subscription(subscription)
    raise TimeoutError(
        f"received {len(samples)}/{count} messages on "
        f"{arm.joint_states_topic} within {timeout_sec:.1f}s"
    )


def _message_maps(msg: Any) -> tuple[dict[str, float], dict[str, float]]:
    names = list(map(str, msg.name))
    if not names:
        raise ValueError("JointState.name is empty")
    if len(set(names)) != len(names):
        raise ValueError("JointState contains duplicate joint names")
    if len(msg.position) != len(names):
        raise ValueError(
            f"JointState has {len(names)} names but {len(msg.position)} positions"
        )
    if len(msg.velocity) != len(names):
        raise ValueError(
            f"JointState has {len(names)} names but {len(msg.velocity)} velocities; "
            "the safety monitor requires named velocities for the full robot state"
        )
    return (
        dict(zip(names, map(float, msg.position))),
        dict(zip(names, map(float, msg.velocity))),
    )


def _validate_joint_messages(
    samples: Sequence[tuple[Any, float]],
    required_joints: Sequence[str],
) -> str:
    for index, (msg, _) in enumerate(samples):
        positions, velocities = _message_maps(msg)
        missing = sorted(set(required_joints) - positions.keys())
        if missing:
            raise ValueError(f"sample {index} is missing right-arm joints: {missing}")
        values = [positions[name] for name in required_joints]
        values += [velocities[name] for name in required_joints]
        if not np.isfinite(values).all():
            raise ValueError(f"sample {index} has non-finite right-arm q or qd")
    return "7 expected joints with finite q and qd"


def _stamp_sec(msg: Any) -> float:
    return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9


def _validate_timing(
    samples: Sequence[tuple[Any, float]],
    stale_timeout_sec: float = STALE_TIMEOUT_SEC,
) -> str:
    stamps = np.asarray([_stamp_sec(msg) for msg, _ in samples])
    received = np.asarray([received_at for _, received_at in samples])
    if not np.isfinite(stamps).all() or np.any(stamps <= 0.0):
        raise ValueError("joint-state timestamps must be finite and positive")
    stamp_deltas = np.diff(stamps)
    if np.any(stamp_deltas < 0.0):
        raise ValueError("joint-state timestamp moved backward")
    if not np.any(stamp_deltas > 0.0):
        raise ValueError("joint-state timestamps did not advance")
    gaps = np.diff(received)
    max_gap = float(np.max(gaps))
    if max_gap >= stale_timeout_sec:
        raise ValueError(
            f"maximum receive gap {max_gap:.4f}s reaches/exceeds controller "
            f"stale timeout {stale_timeout_sec:.4f}s"
        )
    return f"median {1.0 / float(np.median(gaps)):.1f} Hz, max gap {max_gap:.4f}s"


def _service_call(
    arm: TiagoRightArmClient,
    service_name: str,
    service_type: type,
    timeout_sec: float,
) -> Any:
    return arm._service_call(  # noqa: SLF001 - exercise the wrapper's call path
        service_name,
        service_type,
        service_type.Request(),
        timeout_sec,
    )


def _check_services(arm: TiagoRightArmClient) -> str:
    manager = arm.controller_manager
    expected = {
        f"{manager}/{suffix}"
        for suffix in (
            "list_controllers",
            "list_controller_types",
            "list_hardware_interfaces",
            "load_controller",
            "configure_controller",
            "switch_controller",
            "set_parameters",
        )
    }
    discovered = {name for name, _ in arm.node.get_service_names_and_types()}
    missing = sorted(expected - discovered)
    if missing:
        raise RuntimeError(f"required services are missing: {missing}")
    return "7 required services available"


def _controller_map(response: Any) -> dict[str, Any]:
    return {str(controller.name): controller for controller in response.controller}


def _check_default_controllers(arm: TiagoRightArmClient, response: Any) -> str:
    controllers = _controller_map(response)
    missing = sorted(set(DEFAULT_CONTROLLERS) - controllers.keys())
    if missing:
        raise RuntimeError(f"required default controllers are missing: {missing}")
    inactive = [
        name
        for name in DEFAULT_CONTROLLERS
        if str(controllers[name].state).lower() != "active"
    ]
    if inactive:
        raise RuntimeError(f"default controllers are not active: {inactive}")
    effort = controllers.get(arm.effort_controller)
    if effort is not None and str(effort.state).lower() == "active":
        raise RuntimeError("runtime effort controller is already active")
    if not arm.node.get_subscriptions_info_by_topic(arm.trajectory_topic):
        raise RuntimeError(f"no subscriber on {arm.trajectory_topic}")
    return "position and gravity-compensation controllers active"


def _check_effort_controller(arm: TiagoRightArmClient, controllers: Any, types: Any) -> str:
    if FORWARD_CONTROLLER_TYPE not in set(map(str, types.types)):
        raise RuntimeError(f"controller plugin unavailable: {FORWARD_CONTROLLER_TYPE}")
    existing = _controller_map(controllers).get(arm.effort_controller)
    if existing is not None and str(existing.type) != FORWARD_CONTROLLER_TYPE:
        raise RuntimeError(
            f"{arm.effort_controller} has type {existing.type!r}, "
            f"expected {FORWARD_CONTROLLER_TYPE!r}"
        )
    state = "not loaded" if existing is None else existing.state
    return f"plugin available; runtime controller state: {state}"


def _check_effort_interfaces(response: Any) -> str:
    interfaces = {str(value.name): value for value in response.command_interfaces}
    required = {f"{joint}/effort" for joint in RIGHT_ARM_JOINTS}
    missing = sorted(required - interfaces.keys())
    if missing:
        raise RuntimeError(f"right-arm effort command interfaces are missing: {missing}")
    unavailable = sorted(name for name in required if not interfaces[name].is_available)
    if unavailable:
        raise RuntimeError(f"right-arm effort interfaces are unavailable: {unavailable}")
    claimed = sum(bool(interfaces[name].is_claimed) for name in required)
    return f"7 available interfaces; {claimed} currently claimed"


def _check_safety(arm: TiagoRightArmClient) -> str:
    from gato_tiago.safety_monitor import (
        AsyncSafetyMonitor,
        CollisionSafetySettings,
    )
    from gato_tiago.tiago_controller_process import _safety_check_state

    state = arm.latest_state()
    if state is None:
        raise RuntimeError("joint-state stream unavailable")
    safety_state = _safety_check_state(state)
    settings = CollisionSafetySettings(controlled_joint_names=tuple(RIGHT_ARM_JOINTS))
    monitor = AsyncSafetyMonitor(settings=settings, initial_state=safety_state)
    try:
        monitor.wait_until_checked(safety_state)
    finally:
        monitor.close()
    return "full state, limits, 40mm clearance, and geometry speed pass"


def _fail(reason: str) -> None:
    raise RuntimeError(reason)


def main() -> int:
    args = parse_args()
    checks = Checks()
    loaded: dict[str, Any] = {}

    def load_ros() -> str:
        loaded["ros"] = _load_ros()
        return "rclpy and controller-manager messages importable"

    if not checks.run("ROS 2 Python environment", load_ros):
        print("\nPreflight failed; remaining checks require ROS 2.")
        return 1

    try:
        arm = TiagoRightArmClient(node_name="gato_tiago_hardware_preflight")
    except Exception as exc:
        reason = str(exc)
        checks.run("ROS node creation and graph discovery", lambda: _fail(reason))
        print("\nPreflight failed; remaining checks require a ROS node.")
        return 1
    checks.run(
        "ROS node creation and graph discovery",
        lambda: f"node created in namespace {arm.node.get_namespace()}",
    )

    samples: list[tuple[Any, float]] = []
    responses: dict[str, Any] = {}
    ros = loaded["ros"]

    def require(value: str, reason: str) -> Any:
        if value not in responses:
            raise RuntimeError(reason)
        return responses[value]

    try:
        def receive_states() -> str:
            samples.extend(
                _collect_joint_states(
                    arm,
                    ros.JointState,
                    count=args.samples,
                    timeout_sec=args.timeout,
                )
            )
            return f"{len(samples)} messages on {arm.joint_states_topic}"

        checks.run("live joint-state stream", receive_states)
        checks.run(
            "right-arm names and finite position/velocity data",
            lambda: _validate_joint_messages(samples, RIGHT_ARM_JOINTS)
            if samples
            else _fail("joint-state stream unavailable"),
        )
        checks.run(
            "joint-state timestamps and freshness",
            lambda: _validate_timing(samples)
            if samples
            else _fail("joint-state stream unavailable"),
        )
        checks.run("controller-manager service surface", lambda: _check_services(arm))

        def read_inventories() -> str:
            manager = arm.controller_manager
            for key, suffix, service_type in (
                ("controllers", "list_controllers", ros.ListControllers),
                ("types", "list_controller_types", ros.ListControllerTypes),
                ("interfaces", "list_hardware_interfaces", ros.ListHardwareInterfaces),
            ):
                responses[key] = _service_call(
                    arm, f"{manager}/{suffix}", service_type, args.timeout
                )
            return "controller, plugin, and hardware inventories readable"

        checks.run("controller-manager read-only inventory calls", read_inventories)
        checks.run(
            "default right-arm controller state",
            lambda: _check_default_controllers(
                arm, require("controllers", "controller inventory unavailable")
            ),
        )
        checks.run(
            "runtime forward-effort controller support",
            lambda: _check_effort_controller(
                arm,
                require("controllers", "controller inventory unavailable"),
                require("types", "controller-type inventory unavailable"),
            ),
        )
        checks.run(
            "right-arm effort command interfaces",
            lambda: _check_effort_interfaces(
                require("interfaces", "hardware-interface inventory unavailable")
            ),
        )
        checks.run("runtime collision/limit safety path", lambda: _check_safety(arm))
    finally:
        arm.close()

    if checks.failures:
        print(f"\nPreflight failed: {checks.failures} check(s) failed.")
        return 1
    print("\nPreflight passed: ROS wrapper/controller assumptions hold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
