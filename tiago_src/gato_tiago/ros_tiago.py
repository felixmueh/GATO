"""Small ROS 2 helpers for controlling a TIAGo right arm in simulation.

The module avoids importing ROS at import time so non-ROS tests and experiments
can still import :mod:`gato_tiago.ros_tiago`. ROS packages are loaded lazily when a
client is constructed.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence

import numpy as np

# TODO: consider generating this from the robot's URDF or ROS parameters.
# Blocker: currently we use an incomplete urdf in GRiD
# ATTENTION: modifying this directly modifies collision safety behavior and should be tested rigorously in simulation.
RIGHT_ARM_JOINTS = (
    "arm_right_1_joint",
    "arm_right_2_joint",
    "arm_right_3_joint",
    "arm_right_4_joint",
    "arm_right_5_joint",
    "arm_right_6_joint",
    "arm_right_7_joint",
)

DEFAULT_JOINT_STATES_TOPIC = "/joint_states"
DEFAULT_TRAJECTORY_TOPIC = "/arm_right_controller/joint_trajectory"
DEFAULT_CONTROLLER_MANAGER = "/controller_manager"
DEFAULT_EFFORT_CONTROLLER = "gato_arm_effort_forward_runtime"
DEFAULT_ROS_SETUP = "/opt/ros/humble/setup.bash"


@dataclass(frozen=True)
class ArmState:
    """Latest joint state for the right arm."""

    q: np.ndarray
    qd: np.ndarray
    stamp_sec: float
    received_monotonic_sec: float
    seq: int = 0
    joint_positions: dict[str, float] | None = None
    joint_velocities: dict[str, float] | None = None

    @property
    def age_sec(self) -> float:
        return time.monotonic() - self.received_monotonic_sec

def ensure_ros_environment(
    setup_path: str | None = None,
    *,
    allow_reexec: bool = True,
) -> None:
    """Load ROS setup variables into this Python process if rclpy is missing."""
    # TODO: consider usinga more canonical way to load ROS env variables.
    try:
        import rclpy  # noqa: F401
    except ImportError:
        pass
    else:
        return

    if os.environ.get("GATO_ROS_SETUP_ATTEMPTED") == "1":
        return

    setup = Path(setup_path or os.environ.get("GATO_ROS_SETUP", DEFAULT_ROS_SETUP))
    if not setup.is_file():
        return

    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1" >/dev/null 2>&1 && env -0',
            "gato_ros_setup",
            str(setup),
        ],
        check=True,
        stdout=subprocess.PIPE,
    )
    for entry in result.stdout.decode("utf-8", errors="surrogateescape").split("\0"):
        if not entry or "=" not in entry:
            continue
        key, value = entry.split("=", 1)
        os.environ[key] = value
    os.environ["GATO_ROS_SETUP_ATTEMPTED"] = "1"

    for path in reversed(os.environ.get("PYTHONPATH", "").split(os.pathsep)):
        if path and path not in sys.path:
            sys.path.insert(0, path)

    try:
        import rclpy  # noqa: F401
    except ImportError:
        if allow_reexec and sys.argv and sys.argv[0] not in {"-", "-c"}:
            os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)


def _ros_imports(setup_path: str | None = None) -> dict[str, Any]:
    # TODO: if this runs in its own container with ROS already sourced import directly.
    ensure_ros_environment(setup_path)
    try:
        import rclpy
        from builtin_interfaces.msg import Duration
        from controller_manager_msgs.srv import (
            ConfigureController,
            LoadController,
            SwitchController,
        )
        from rcl_interfaces.msg import Parameter as RosParameter
        from rcl_interfaces.msg import ParameterType, ParameterValue
        from rcl_interfaces.srv import SetParameters
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    except ImportError as exc:
        raise RuntimeError(
            "ROS 2 Python packages are not importable. Source ROS first, or set "
            "GATO_ROS_SETUP to the setup.bash path for your ROS installation."
        ) from exc

    return locals()


def _duration_msg(ros: dict[str, Any], seconds: float) -> Any:
    if seconds < 0.0:
        raise ValueError("duration must be non-negative")
    duration = ros["Duration"]()
    duration.sec = int(seconds)
    duration.nanosec = int(round((seconds - int(seconds)) * 1e9))
    if duration.nanosec >= 1_000_000_000:
        duration.sec += 1
        duration.nanosec -= 1_000_000_000
    return duration


def _vector(values: Sequence[float], *, size: int, name: str) -> list[float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size != size:
        raise ValueError(f"{name} must contain {size} values, got {arr.size}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return [float(value) for value in arr]


class TiagoRightArmClient:
    """ROS 2 client for TIAGo right-arm state, position, and effort commands."""

    def __init__(
        self,
        *,
        node_name: str = "gato_tiago_right_arm_client",
        joint_names: Sequence[str] = RIGHT_ARM_JOINTS,
        joint_states_topic: str = DEFAULT_JOINT_STATES_TOPIC,
        trajectory_topic: str = DEFAULT_TRAJECTORY_TOPIC,
        controller_manager: str = DEFAULT_CONTROLLER_MANAGER,
        effort_controller: str = DEFAULT_EFFORT_CONTROLLER,
        setup_path: str | None = None,
    ) -> None:
        self.ros = _ros_imports(setup_path)
        self.rclpy = self.ros["rclpy"]
        self._owns_rclpy = False
        if not self.rclpy.ok():
            self.rclpy.init()
            self._owns_rclpy = True

        self.joint_names = tuple(joint_names)
        self.joint_states_topic = joint_states_topic
        self.trajectory_topic = trajectory_topic
        self.controller_manager = controller_manager.rstrip("/")
        self.effort_controller = effort_controller
        self.effort_command_topic = f"/{effort_controller}/commands"
        self._latest_state: ArmState | None = None
        self._state_seq = 0

        self.node = self.rclpy.create_node(node_name)
        self.node.create_subscription(
            self.ros["JointState"],
            self.joint_states_topic,
            self._on_joint_state,
            10,
        )
        self._trajectory_pub = self.node.create_publisher(
            self.ros["JointTrajectory"],
            self.trajectory_topic,
            10,
        )
        self._effort_pub = self.node.create_publisher(
            self.ros["Float64MultiArray"],
            self.effort_command_topic,
            10,
        )

    def close(self) -> None:
        self.node.destroy_node()
        if self._owns_rclpy:
            self.rclpy.shutdown()

    def __enter__(self) -> "TiagoRightArmClient":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def _on_joint_state(self, msg: Any) -> None:
        positions = {name: float(value) for name, value in zip(msg.name, msg.position)}
        velocities = {name: float(value) for name, value in zip(msg.name, msg.velocity)}
        if not all(name in positions for name in self.joint_names):
            return

        q = np.asarray([positions[name] for name in self.joint_names], dtype=np.float32)
        qd = np.asarray(
            [velocities.get(name, 0.0) for name in self.joint_names],
            dtype=np.float32,
        )
        stamp_sec = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        self._state_seq += 1
        self._latest_state = ArmState(
            q=q,
            qd=qd,
            stamp_sec=stamp_sec,
            received_monotonic_sec=time.monotonic(),
            seq=self._state_seq,
            joint_positions=positions,
            joint_velocities=velocities,
        )

    def spin_once(self, timeout_sec: float = 0.0) -> None:
        self.rclpy.spin_once(self.node, timeout_sec=timeout_sec)

    def read_state(self, timeout_sec: float = 2.0) -> ArmState:
        deadline = time.monotonic() + timeout_sec
        while self.rclpy.ok() and time.monotonic() < deadline:
            self.spin_once(timeout_sec=0.05)
            if self._latest_state is not None:
                return self._latest_state
        raise TimeoutError(
            f"did not receive {len(self.joint_names)} right-arm joints on "
            f"{self.joint_states_topic} within {timeout_sec:.1f}s"
        )

    def latest_state(self) -> ArmState | None:
        return self._latest_state

    def publish_position_trajectory(
        self,
        positions: Sequence[float],
        *,
        duration_sec: float = 2.0,
        velocities: Sequence[float] | None = None,
    ) -> None:
        point = self.ros["JointTrajectoryPoint"]()
        point.positions = _vector(positions, size=len(self.joint_names), name="positions")
        if velocities is not None:
            point.velocities = _vector(
                velocities,
                size=len(self.joint_names),
                name="velocities",
            )
        point.time_from_start = _duration_msg(self.ros, duration_sec)

        msg = self.ros["JointTrajectory"]()
        msg.joint_names = list(self.joint_names)
        msg.points = [point]
        self._trajectory_pub.publish(msg)

    def publish_position_trajectory_points(
        self,
        positions: Sequence[Sequence[float]],
        *,
        dt: float,
    ) -> None:
        """Publish a multi-point joint position trajectory.

        The first row is sent at ``dt`` seconds, not zero. This avoids asking
        the controller to jump if the first IK point is slightly different from
        the current measured state.
        """
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        rows = np.asarray(positions, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != len(self.joint_names):
            raise ValueError(
                f"positions must have shape (N, {len(self.joint_names)}), "
                f"got {rows.shape}"
            )
        if not np.isfinite(rows).all():
            raise ValueError("positions contains non-finite values")

        msg = self.ros["JointTrajectory"]()
        msg.joint_names = list(self.joint_names)
        for idx, row in enumerate(rows):
            point = self.ros["JointTrajectoryPoint"]()
            point.positions = [float(value) for value in row]
            point.time_from_start = _duration_msg(self.ros, (idx + 1) * dt)
            msg.points.append(point)
        self._trajectory_pub.publish(msg)

    def publish_effort(self, torques: Sequence[float]) -> None:
        msg = self.ros["Float64MultiArray"]()
        msg.data = _vector(torques, size=len(self.joint_names), name="torques")
        self._effort_pub.publish(msg)

    def configure_runtime_effort_controller(self, timeout_sec: float = 5.0) -> None:
        """Load and configure a forward effort controller through public ROS APIs."""
        if self._topic_has_subscription(self.effort_command_topic):
            return

        self._set_controller_manager_type(
            self.effort_controller,
            "forward_command_controller/ForwardCommandController",
            timeout_sec,
        )
        self._load_controller(self.effort_controller, timeout_sec)
        self._set_remote_parameters(
            f"/{self.effort_controller}",
            {
                "joints": list(self.joint_names),
                "interface_name": "effort",
            },
            timeout_sec,
        )
        self._configure_controller(self.effort_controller, timeout_sec)

    def switch_to_effort_control(self, timeout_sec: float = 5.0) -> None:
        self._switch_controllers(
            activate=[self.effort_controller],
            deactivate=[
                "arm_right_controller",
                "arm_right_gravity_compensation_controller",
            ],
            strictness=2,
            timeout_sec=timeout_sec,
        )
        self._wait_for_topic_subscription(self.effort_command_topic, timeout_sec)

    def switch_to_default_control(self, timeout_sec: float = 5.0) -> None:
        effort_active = self._topic_has_subscription(self.effort_command_topic)
        if self._topic_has_subscription(self.trajectory_topic) and not effort_active:
            return
        # Start the reactivated position controller from the measured joint
        # state, not from a stale command-interface value left by a previous
        # trajectory/controller mode.
        self._set_remote_parameters(
            "/arm_right_controller",
            {"set_last_command_interface_value_as_state_on_activation": False},
            timeout_sec,
        )
        self._switch_controllers(
            activate=["arm_right_controller"],
            deactivate=[self.effort_controller] if effort_active else [],
            strictness=1,
            timeout_sec=timeout_sec,
        )
        self._switch_controllers(
            activate=["arm_right_gravity_compensation_controller"],
            deactivate=[],
            strictness=1,
            timeout_sec=timeout_sec,
        )

    def _service_call(
        self,
        service_name: str,
        service_type: type,
        request: Any,
        timeout_sec: float,
    ) -> Any:
        client = self.node.create_client(service_type, service_name)
        try:
            if not client.wait_for_service(timeout_sec=timeout_sec):
                raise TimeoutError(f"service not available: {service_name}")
            future = client.call_async(request)
            self.rclpy.spin_until_future_complete(
                self.node,
                future,
                timeout_sec=timeout_sec,
            )
            if not future.done():
                raise TimeoutError(f"service call timed out: {service_name}")
            result = future.result()
            if result is None:
                raise RuntimeError(f"service call returned no result: {service_name}")
            return result
        finally:
            self.node.destroy_client(client)

    def _set_controller_manager_type(
        self,
        controller_name: str,
        controller_type: str,
        timeout_sec: float,
    ) -> None:
        self._set_remote_parameters(
            self.controller_manager,
            {f"{controller_name}.type": controller_type},
            timeout_sec,
        )

    def _set_remote_parameters(
        self,
        node_name: str,
        values: dict[str, str | bool | list[str]],
        timeout_sec: float,
    ) -> None:
        request = self.ros["SetParameters"].Request()
        for name, value in values.items():
            request.parameters.append(self._parameter_msg(name, value))
        response = self._service_call(
            f"{node_name}/set_parameters",
            self.ros["SetParameters"],
            request,
            timeout_sec,
        )
        failed = [result.reason for result in response.results if not result.successful]
        if failed:
            raise RuntimeError(f"failed setting parameters on {node_name}: {failed}")

    def _parameter_msg(self, name: str, value: str | bool | list[str]) -> Any:
        msg = self.ros["RosParameter"]()
        msg.name = name
        msg.value = self.ros["ParameterValue"]()
        if isinstance(value, bool):
            msg.value.type = self.ros["ParameterType"].PARAMETER_BOOL
            msg.value.bool_value = value
        elif isinstance(value, str):
            msg.value.type = self.ros["ParameterType"].PARAMETER_STRING
            msg.value.string_value = value
        else:
            msg.value.type = self.ros["ParameterType"].PARAMETER_STRING_ARRAY
            msg.value.string_array_value = list(value)
        return msg

    def _load_controller(self, name: str, timeout_sec: float) -> None:
        request = self.ros["LoadController"].Request()
        request.name = name
        response = self._service_call(
            f"{self.controller_manager}/load_controller",
            self.ros["LoadController"],
            request,
            timeout_sec,
        )
        if not response.ok and not self._node_exists(f"/{name}"):
            raise RuntimeError(f"failed to load controller: {name}")

    def _configure_controller(self, name: str, timeout_sec: float) -> None:
        request = self.ros["ConfigureController"].Request()
        request.name = name
        response = self._service_call(
            f"{self.controller_manager}/configure_controller",
            self.ros["ConfigureController"],
            request,
            timeout_sec,
        )
        if not response.ok and not self._node_exists(f"/{name}"):
            raise RuntimeError(f"failed to configure controller: {name}")

    def _switch_controllers(
        self,
        *,
        activate: Sequence[str],
        deactivate: Sequence[str],
        strictness: int,
        timeout_sec: float,
    ) -> None:
        request = self.ros["SwitchController"].Request()
        request.activate_controllers = list(activate)
        request.deactivate_controllers = list(deactivate)
        request.strictness = strictness
        request.start_asap = True
        request.timeout = _duration_msg(self.ros, timeout_sec)
        response = self._service_call(
            f"{self.controller_manager}/switch_controller",
            self.ros["SwitchController"],
            request,
            timeout_sec,
        )
        if not response.ok:
            raise RuntimeError(
                "failed to switch controllers: "
                f"activate={list(activate)} deactivate={list(deactivate)}"
            )

    def _node_exists(self, full_name: str) -> bool:
        for name, namespace in self.node.get_node_names_and_namespaces():
            candidate = f"{namespace.rstrip('/')}/{name}" if namespace != "/" else f"/{name}"
            if candidate == full_name:
                return True
        return False

    def _topic_has_subscription(self, topic_name: str) -> bool:
        return bool(self.node.get_subscriptions_info_by_topic(topic_name))

    def _wait_for_topic_subscription(self, topic_name: str, timeout_sec: float) -> None:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            self.spin_once(timeout_sec=0.05)
            if self._topic_has_subscription(topic_name):
                return
        raise TimeoutError(f"no subscriber appeared on {topic_name}")
