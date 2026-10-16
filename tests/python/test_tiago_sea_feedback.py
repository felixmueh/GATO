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


def test_sustained_50hz_feedback_holds_between_125hz_polls(client):
    from sea_feedback_synthetic import delivery_schedule

    arm = client.arm
    source_times, ticks, selected = delivery_schedule(50., duration_sec=3.)
    body_times = .004 + np.arange(150) / 50.  # Independent 50 Hz body phase.
    body_index, last_body = 0, None
    base, speed = np.arange(7) * .05, np.linspace(-.3, .3, 7)
    last_sea, last_state, delivered = None, None, 0
    held_polls = 0
    for tick, selected_index in zip(ticks, selected):
        client.clock.now = 1. + tick
        if body_index < len(body_times) and body_times[body_index] <= tick:
            while body_index < len(body_times) and body_times[body_index] <= tick:
                last_body = body_times[body_index]
                body_index += 1
            body = message(50. + last_body, .4,
                           names=(*ros.RIGHT_ARM_JOINTS, "torso_lift_joint", "wheel_right_joint"))
            body.velocity[-1] = float("nan")
            arm._on_joint_state(body)
        if selected_index >= 0:
            source_t = source_times[selected_index]
            last_sea = message(100. + source_t, base + speed * source_t)
            arm._on_sea_state(last_sea)
            delivered += 1
        elif last_sea is not None:
            # Even a duplicate callback at an extra poll must not refresh source state.
            arm._on_sea_state(last_sea)
        state = arm.latest_state()
        if delivered < 2 or last_body is None:
            assert state is None
            continue
        assert state is not None
        source_t = state.stamp_sec - 100.
        np.testing.assert_allclose(state.q, base + speed * source_t, rtol=0, atol=1e-12)
        np.testing.assert_allclose(state.qd, speed, rtol=0, atol=1e-10)
        assert state.body_stamp_sec == pytest.approx(50. + last_body)
        assert state.age_sec < .04
        assert state.joint_positions["torso_lift_joint"] == pytest.approx(.6)
        if selected_index < 0 and last_state is not None:
            held_polls += 1
            assert state.seq == last_state.seq
            assert state.stamp_sec == last_state.stamp_sec
            assert state.arm_received_monotonic_sec == last_state.arm_received_monotonic_sec
            np.testing.assert_array_equal(state.qd, last_state.qd)
        last_state = state
    assert delivered == len(source_times) == 151
    assert state.seq == 150  # One startup sample, then one estimate per advancing source.
    assert held_polls > 200


@pytest.mark.parametrize("rate", [50., 125.])
def test_irregular_rate_and_delivery_jitter_keep_source_time_velocity(rate):
    from sea_feedback_synthetic import delivery_schedule, sea_message

    source_times, ticks, selected = delivery_schedule(rate, jitter=True, duration_sec=3.)
    estimator = ros.SeaAbsoluteVelocityEstimator()
    speed = np.linspace(-.25, .4, 7)
    estimates, delivered_times, receive_times = [], [], []
    for tick, index in zip(ticks, selected):
        if index < 0:
            continue
        t = source_times[index]
        q = .2 + speed * t
        result = estimator.update(sea_message(t, q), 1. + tick)
        delivered_times.append(t)
        receive_times.append(tick)
        if result is not None:
            measured_q, qd, stamp = result
            np.testing.assert_array_equal(measured_q, q)
            np.testing.assert_allclose(qd, speed, rtol=0, atol=1e-10)
            assert stamp == pytest.approx(100. + t)
            estimates.append(result)
    assert len(estimates) == len(delivered_times) - 1
    assert max(np.diff(delivered_times)) < .1
    assert max(np.diff(receive_times)) < .1
    assert max(np.diff(delivered_times)) > 2. / rate  # Includes short losses/burst.


@pytest.mark.parametrize("rate", [1000., 125., 50.])
@pytest.mark.parametrize("jitter", [False, True])
@pytest.mark.parametrize("noise_std", [.0001, .001])
def test_seeded_stationary_noise_attenuation_without_position_filter(rate, jitter, noise_std):
    from sea_feedback_synthetic import run_case

    metrics, rows = run_case(rate, noise_std, "stationary", jitter=jitter)
    # This is a chosen iid-noise protocol, not a universal safety/noise bound.
    assert metrics["qd_error_rms_rad_s"] < metrics["raw_derivative_error_rms_rad_s"]
    assert metrics["delivered_callback_rate_hz"] <= 125.
    assert metrics["max_delivered_source_gap_sec"] < .1
    assert metrics["max_callback_gap_sec"] < .1
    assert metrics["estimator_unavailable_callbacks"] == 1
    # A raw noisy position is returned, including its noise; no position smoothing.
    assert metrics["q_raw_error_rms_rad"] > .9 * noise_std
    assert metrics["q_raw_error_rms_rad"] < 1.1 * noise_std
    assert metrics["q_matches_input_exactly"] is True
    for row in rows:
        np.testing.assert_array_equal(row["q"], row["observed_q"])
