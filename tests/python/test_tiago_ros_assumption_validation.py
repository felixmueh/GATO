from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TIAGO_EXAMPLES = REPO_ROOT / "tiago_examples"
if str(TIAGO_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(TIAGO_EXAMPLES))

from validate_ros_assumptions import (
    RIGHT_ARM_JOINTS,
    _check_effort_interfaces,
    _validate_joint_messages,
    _validate_timing,
)


def _message(*, stamp: float, velocities: bool = True):
    sec = int(stamp)
    nanosec = int(round((stamp - sec) * 1e9))
    names = list(RIGHT_ARM_JOINTS)
    return SimpleNamespace(
        name=names,
        position=[0.0] * len(names),
        velocity=[0.0] * len(names) if velocities else [],
        header=SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=nanosec)),
    )


def test_joint_validation_rejects_missing_velocity_array():
    samples = [(_message(stamp=1.0, velocities=False), 10.0)]

    with pytest.raises(ValueError, match="requires named velocities"):
        _validate_joint_messages(samples, RIGHT_ARM_JOINTS)


def test_timing_validation_accepts_fresh_monotonic_samples():
    samples = [
        (_message(stamp=1.00), 10.00),
        (_message(stamp=1.01), 10.01),
        (_message(stamp=1.02), 10.02),
    ]

    detail = _validate_timing(samples, stale_timeout_sec=0.1)

    assert "100.0 Hz" in detail


def test_timing_validation_rejects_controller_stale_gap():
    samples = [
        (_message(stamp=1.00), 10.00),
        (_message(stamp=1.20), 10.20),
    ]

    with pytest.raises(ValueError, match="stale timeout"):
        _validate_timing(samples, stale_timeout_sec=0.1)


def test_effort_interface_validation_requires_all_right_arm_interfaces():
    response = SimpleNamespace(
        command_interfaces=[
            SimpleNamespace(
                name=f"{joint}/effort",
                is_available=True,
                is_claimed=False,
            )
            for joint in RIGHT_ARM_JOINTS[:-1]
        ]
    )

    with pytest.raises(RuntimeError, match="arm_right_7_joint/effort"):
        _check_effort_interfaces(response)
