"""SEA source selection and estimator safety, without ROS, GPU, or robot access."""
import builtins
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from gato_tiago import ros_tiago as ros
from gato_tiago.tiago_history_writer import make_history_record


def message(stamp, q=0., *, names=ros.RIGHT_ARM_JOINTS):
    q = np.broadcast_to(q, (len(names),)).astype(float)
    sec = int(stamp)
    return NS(name=list(names), absolute_position=q.tolist(), position=(q + .2).tolist(),
              velocity=np.zeros(len(names)).tolist(),
              header=NS(stamp=NS(sec=sec, nanosec=round((stamp - sec) * 1e9))))


@pytest.fixture
def client(monkeypatch):
    clock = NS(now=1.)
    monkeypatch.setattr(ros.time, "monotonic", lambda: clock.now)
    subscriptions, publications = [], []
    node = NS(create_subscription=lambda *args: subscriptions.append(args),
              create_publisher=lambda *args: publications.append(args), destroy_node=lambda: None)
    fake_ros = dict(rclpy=NS(ok=lambda: True, create_node=lambda name: node),
                    JointState=object, JointTrajectory=object, Float64MultiArray=object)
    monkeypatch.setattr(ros, "_ros_imports", lambda setup: fake_ros)
    monkeypatch.setattr(ros, "_sea_state_type", lambda: NS())
    monkeypatch.setitem(sys.modules, "rclpy.qos", NS(
        QoSProfile=lambda **kwargs: NS(**kwargs), ReliabilityPolicy=NS(BEST_EFFORT="best_effort")))
    arm = ros.TiagoRightArmClient(sea_absolute=True)
    return NS(arm=arm, clock=clock, subscriptions=subscriptions, publications=publications)


def test_irregular_source_dt_name_order_and_unfiltered_position():
    estimator = ros.SeaAbsoluteVelocityEstimator(tau_sec=.01)
    names = tuple(reversed(ros.RIGHT_ARM_JOINTS))
    base = np.arange(7, dtype=float)
    speed = np.arange(1, 8, dtype=float)
    assert estimator.update(message(10., base, names=names), 1.) is None
    for stamp, received in [(10.007, 1.02), (10.026, 1.03), (10.030, 1.08)]:
        raw = base + speed * (stamp - 10.)
        q, qd, returned_stamp = estimator.update(message(stamp, raw, names=names), received)
        np.testing.assert_allclose(q, raw[::-1], rtol=0, atol=1e-12)
        np.testing.assert_allclose(qd, speed[::-1], rtol=0, atol=1e-9)
        assert returned_stamp == stamp


def test_time_based_filter_only_changes_velocity():
    estimator = ros.SeaAbsoluteVelocityEstimator(tau_sec=.01)
    estimator.update(message(1., 0.), 1.)
    estimator.update(message(1.01, .01), 1.01)
    q, qd, _ = estimator.update(message(1.03, .07), 1.03)
    np.testing.assert_allclose(q, .07)
    np.testing.assert_allclose(qd, 1. + (1 - np.exp(-2.)) * 2.)


def test_duplicate_reversed_and_invalid_samples_do_not_refresh_estimator():
    estimator = ros.SeaAbsoluteVelocityEstimator()
    estimator.update(message(1., 0.), 1.)
    estimator.update(message(1.01, .01), 1.01)
    for stamp in [1.01, 1.005]:
        assert estimator.update(message(stamp, 100.), 1.05) is None
    q, qd, _ = estimator.update(message(1.02, .02), 1.06)
    np.testing.assert_allclose(q, .02)
    np.testing.assert_allclose(qd, 1.)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "absolute_short", "motor_short", "absolute_nan", "motor_nan", "stamp_nan"])
def test_reject_malformed_sea_samples(client, fault):
    arm = client.arm
    arm._on_joint_state(message(1.))
    arm._on_sea_state(message(1.))
    bad = message(1.01)
    if fault == "missing": bad.name[0] = "other_joint"
    elif fault == "duplicate": bad.name[0] = bad.name[1]
    elif fault == "absolute_short": bad.absolute_position.pop()
    elif fault == "motor_short": bad.position.pop()
    elif fault == "absolute_nan": bad.absolute_position[0] = float("nan")
    elif fault == "motor_nan": bad.position[0] = float("nan")
    elif fault == "stamp_nan": bad.header.stamp.sec = float("nan")
    arm._on_sea_state(bad)
    assert arm.latest_state() is None
    client.clock.now = 1.02
    arm._on_sea_state(message(1.02, .02))
    np.testing.assert_allclose(arm.latest_state().qd, 1.)


@pytest.mark.parametrize("gap_kind", ["source", "receive"])
def test_gap_requires_two_new_samples(client, gap_kind):
    arm = client.arm
    arm._on_joint_state(message(1.))
    arm._on_sea_state(message(1.))
    client.clock.now = 1.01
    arm._on_sea_state(message(1.01, .01))
    assert arm.latest_state() is not None
    client.clock.now = 1.2
    stamp = 1.2 if gap_kind == "source" else 1.02
    arm._on_joint_state(message(1.2))
    arm._on_sea_state(message(stamp, 1.))
    assert arm.latest_state() is None
    client.clock.now += .01
    arm._on_sea_state(message(stamp + .01, 1.02))
    np.testing.assert_allclose(arm.latest_state().qd, 2.)


def test_full_body_merge_and_raw_channels_in_history(client):
    arm = client.arm
    body_names = (*ros.RIGHT_ARM_JOINTS, "torso_lift_joint", "wheel_right_joint")
    body = message(2., .4, names=body_names)
    body.velocity[-1] = float("nan")  # Normal unused wheel state on this robot.
    arm._on_joint_state(body)
    arm._on_sea_state(message(3., 1.))
    assert arm.latest_state() is None  # No fabricated startup velocity.
    client.clock.now = 1.02
    arm._on_sea_state(message(3.02, 1.04, names=tuple(reversed(ros.RIGHT_ARM_JOINTS))))
    state = arm.latest_state()
    np.testing.assert_allclose(state.q, 1.04)
    np.testing.assert_allclose(state.qd, 2.)
    assert state.stamp_sec == 3.02 and state.body_stamp_sec == 2.
    assert state.received_monotonic_sec == 1.
    assert state.joint_positions["torso_lift_joint"] == pytest.approx(.6)
    assert np.isnan(state.joint_velocities["wheel_right_joint"])
    assert all(state.joint_positions[name] == 1.04 for name in ros.RIGHT_ARM_JOINTS)
    record = make_history_record(state, "READY").to_json_dict()
    metadata = record["feedback_metadata"]
    np.testing.assert_allclose(metadata["sea_motor_position"], 1.24)
    np.testing.assert_allclose(metadata["sea_absolute_position"], state.q)
    assert metadata["velocity_filter_tau_sec"] == .01
    assert metadata["body_received_monotonic_sec"] == 1.
    assert metadata["arm_received_monotonic_sec"] == 1.02
    assert client.subscriptions[0][3].depth == 1
    assert client.subscriptions[1][1] == ros.DEFAULT_SEA_STATE_TOPIC


@pytest.mark.parametrize("frozen_source", ["body", "arm"])
def test_fresh_other_source_cannot_mask_stalled_stamp(client, frozen_source):
    arm = client.arm
    arm._on_joint_state(message(1.))
    arm._on_sea_state(message(1.))
    client.clock.now = 1.01
    arm._on_sea_state(message(1.01))
    assert arm.latest_state() is not None
    for now in [1.04, 1.08, 1.12]:
        client.clock.now = now
        arm._on_joint_state(message(1. if frozen_source == "body" else now))
        arm._on_sea_state(message(1.01 if frozen_source == "arm" else now))
    assert arm.latest_state() is None


def test_default_does_not_resolve_sea_dependency_or_change_jointstate_path(client, monkeypatch):
    monkeypatch.setattr(ros, "_sea_state_type", lambda: pytest.fail("optional SEA import"))
    arm = ros.TiagoRightArmClient()
    msg = message(5., .4)
    msg.velocity = [.3] * 7
    arm._on_joint_state(msg)
    state = arm.latest_state()
    np.testing.assert_allclose(state.q, .6)
    np.testing.assert_allclose(state.qd, .3)
    assert state.feedback_metadata is None
    assert "feedback_metadata" not in make_history_record(state, "READY").to_json_dict()
    assert client.subscriptions[-1][3] == 10


def test_missing_optional_interface_gives_clear_error(monkeypatch):
    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name == "sea_state_msgs.msg":
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(RuntimeError, match="sea_state_msgs/msg/SeaState"):
        ros._sea_state_type()


@pytest.mark.parametrize("tau", [-1., float("nan"), float("inf")])
def test_bad_tau_rejected(tau):
    with pytest.raises(ValueError, match="finite and non-negative"):
        ros.SeaAbsoluteVelocityEstimator(tau_sec=tau)


def test_abs_position_is_sufficient_without_optional_motor_channel(client):
    arm = client.arm
    arm._on_joint_state(message(1.))
    first, second = message(1.), message(1.01, .02)
    first.position = second.position = []
    arm._on_sea_state(first)
    client.clock.now = 1.01
    arm._on_sea_state(second)
    np.testing.assert_allclose(arm.latest_state().qd, 2.)
    assert arm.latest_state().feedback_metadata["sea_motor_position"] is None
