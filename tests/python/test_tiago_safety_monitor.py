import json
import os
from pathlib import Path
import secrets

import numpy as np
import pytest

pin = pytest.importorskip("pinocchio")

from gato_tiago.safety_monitor import (
    TIAGO_RUNTIME_STATE_FIXED_JOINT_NAMES,
    NamedJointState,
    build_tiago_collision_model,
    check_joint_limits,
    compute_collision_body_speeds,
    compute_collision_margin_violation,
    compute_collision_margin_violations,
    state_to_qv,
)
from gato_tiago.tiago_tools import (
    compute_minimum_pair_distance,
    compute_pair_distances,
)
from gato_tiago.ros_tiago import RIGHT_ARM_JOINTS


URDF_PATH = Path("/workspace/GATO/TiagoProURDF/tiago_pro.urdf")


def _skip_without_urdf() -> None:
    if not URDF_PATH.is_file():
        pytest.skip(f"missing Tiago URDF at {URDF_PATH}")


@pytest.fixture(scope="module")
def collision_model():
    _skip_without_urdf()
    return build_tiago_collision_model(
        urdf_path=URDF_PATH,
        monitor_only_collision_pairs=False,
    )


@pytest.fixture(scope="module")
def runtime_collision_model():
    _skip_without_urdf()
    return build_tiago_collision_model(urdf_path=URDF_PATH)


def _zero_state(collision_model):
    return NamedJointState(
        position={name: 0.0 for name in collision_model.state_updated_joint_names},
        velocity={name: 0.0 for name in collision_model.state_updated_joint_names},
    )


def _state_with(collision_model, *, positions=None, velocities=None):
    state = _zero_state(collision_model)
    if positions:
        state.position.update(positions)
    if velocities:
        state.velocity.update(velocities)
    return state


def _random_state(collision_model, rng):
    positions = {}
    velocities = {}
    for idx, name in enumerate(collision_model.model.names):
        if idx == 0:
            continue
        joint = collision_model.model.joints[idx]
        q_idx = joint.idx_q
        v_idx = joint.idx_v

        lower = float(collision_model.model.lowerPositionLimit[q_idx])
        upper = float(collision_model.model.upperPositionLimit[q_idx])
        if np.isfinite(lower) and np.isfinite(upper) and upper > lower:
            margin = min(0.05, 0.1 * (upper - lower))
            positions[name] = float(rng.uniform(lower + margin, upper - margin))
        else:
            positions[name] = 0.0

        velocity_limit = float(collision_model.model.velocityLimit[v_idx])
        velocity_bound = velocity_limit if np.isfinite(velocity_limit) else 1.0
        velocity_bound = min(max(velocity_bound, 0.1), 1.0)
        velocities[name] = float(rng.uniform(-velocity_bound, velocity_bound))

    return NamedJointState(position=positions, velocity=velocities)


def _geometry_placements(collision_model, q):
    data = collision_model.model.createData()
    geometry_data = pin.GeometryData(collision_model.geometry_model)
    pin.framesForwardKinematics(collision_model.model, data, q)
    pin.updateGeometryPlacements(
        collision_model.model,
        data,
        collision_model.geometry_model,
        geometry_data,
        q,
    )
    return geometry_data.oMg


def _finite_difference_geometry_velocity(collision_model, geometry_index, q, v):
    dt = 1e-7
    placements = _geometry_placements(collision_model, q)

    q_next = pin.integrate(collision_model.model, q, v * dt)
    next_placements = _geometry_placements(collision_model, q_next)
    return (
        next_placements[geometry_index].translation
        - placements[geometry_index].translation
    ) / dt


def _world_point(placement, local_point):
    return placement.translation + placement.rotation @ local_point


def _random_unit_vector(rng):
    vector = rng.normal(size=3)
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        return np.array([1.0, 0.0, 0.0])
    return vector / norm


def _fuzz_seed():
    raw_seed = os.environ.get("GATO_COLLISION_SPEED_FUZZ_SEED")
    if raw_seed:
        return int(raw_seed, 0)
    return secrets.randbits(32)


def test_reduced_model_fixes_runtime_state_fixed_joints(collision_model):
    assert collision_model.state_fixed_joint_names == TIAGO_RUNTIME_STATE_FIXED_JOINT_NAMES
    assert not set(TIAGO_RUNTIME_STATE_FIXED_JOINT_NAMES).intersection(
        collision_model.state_updated_joint_names
    )
    assert "torso_lift_joint" in collision_model.state_updated_joint_names
    assert len(collision_model.geometry_model.geometryObjects) == len(
        collision_model.geometry_radii_m
    )
    assert len(collision_model.geometry_model.collisionPairs) > 0


def test_state_to_qv_requires_every_state_updated_joint(collision_model):
    state = _zero_state(collision_model)
    missing = collision_model.state_updated_joint_names[0]
    del state.position[missing]

    with pytest.raises(ValueError, match=missing):
        state_to_qv(collision_model.model, state)


def test_distance_and_speed_reports_cover_collision_model(collision_model):
    q, qd = state_to_qv(collision_model.model, _zero_state(collision_model))

    distances = compute_pair_distances(collision_model, q)
    speeds = compute_collision_body_speeds(collision_model, q, qd)

    assert len(distances) == len(collision_model.geometry_model.collisionPairs)
    assert len(speeds) == len(collision_model.monitored_geometry_indices)
    assert all(report.geometry_a for report in distances)
    assert all(report.geometry_b for report in distances)
    assert all(report.speed_bound_m_s == pytest.approx(0.0) for report in speeds)
    assert all(report.radius_m >= 0.0 for report in speeds)


def test_collision_body_speed_uses_current_geometry_velocity(collision_model):
    geometry_index = next(
        idx
        for idx in collision_model.monitored_geometry_indices
        if collision_model.geometry_model.geometryObjects[idx].name.startswith(
            "gripper_right_fingertip_right_link"
        )
    )
    geometry_name = collision_model.geometry_model.geometryObjects[geometry_index].name

    cases = [
        {},
        {
            "arm_right_2_joint": -0.8,
            "arm_right_3_joint": 1.1,
            "arm_right_4_joint": 0.7,
            "arm_right_5_joint": -1.2,
            "arm_right_6_joint": 0.6,
            "arm_right_7_joint": -0.4,
        },
    ]
    linear_speeds = []
    for positions in cases:
        q, v = state_to_qv(
            collision_model.model,
            _state_with(
                collision_model,
                positions=positions,
                velocities={"arm_right_1_joint": 0.5},
            ),
        )

        reports = compute_collision_body_speeds(
            collision_model,
            q,
            v,
            geometry_indices=[geometry_index],
        )
        report = reports[0]
        finite_difference_velocity = _finite_difference_geometry_velocity(
            collision_model,
            geometry_index,
            q,
            v,
        )

        assert report.geometry == geometry_name
        assert report.linear_speed_m_s == pytest.approx(
            np.linalg.norm(finite_difference_velocity),
            rel=1e-4,
            abs=1e-5,
        )
        assert report.speed_bound_m_s >= report.linear_speed_m_s
        linear_speeds.append(report.linear_speed_m_s)

    assert abs(linear_speeds[1] - linear_speeds[0]) > 0.1


def test_collision_body_speed_fuzz_matches_finite_difference(collision_model):
    seed = _fuzz_seed()
    print(f"GATO_COLLISION_SPEED_FUZZ_SEED={seed}")
    rng = np.random.default_rng(seed)
    dt = 1e-7

    for _ in range(24):
        q, v = state_to_qv(collision_model.model, _random_state(collision_model, rng))
        next_q = pin.integrate(collision_model.model, q, v * dt)
        placements = _geometry_placements(collision_model, q)
        next_placements = _geometry_placements(collision_model, next_q)
        reports = {
            report.geometry: report
            for report in compute_collision_body_speeds(collision_model, q, v)
        }

        for geometry_index in collision_model.monitored_geometry_indices:
            obj = collision_model.geometry_model.geometryObjects[geometry_index]
            report = reports[obj.name]
            origin_velocity = (
                next_placements[geometry_index].translation
                - placements[geometry_index].translation
            ) / dt
            assert report.linear_speed_m_s == pytest.approx(
                np.linalg.norm(origin_velocity),
                rel=2e-4,
                abs=2e-5,
            ), f"seed={seed} geometry={obj.name}"

            radius = float(collision_model.geometry_radii_m[geometry_index])
            for scale in (0.25, 1.0):
                local_point = _random_unit_vector(rng) * radius * scale
                point_velocity = (
                    _world_point(next_placements[geometry_index], local_point)
                    - _world_point(placements[geometry_index], local_point)
                ) / dt
                assert (
                    np.linalg.norm(point_velocity) <= report.speed_bound_m_s + 2e-4
                ), f"seed={seed} geometry={obj.name} scale={scale}"


def test_runtime_collision_model_only_keeps_pairs_touching_monitored_geometry(
    collision_model,
    runtime_collision_model,
):
    monitored = set(runtime_collision_model.monitored_geometry_indices)
    monitored_parent_joints = set(
        runtime_collision_model.monitored_geometry_parent_joint_names
    )
    monitored_geometry_names = {
        runtime_collision_model.geometry_model.geometryObjects[idx].name
        for idx in monitored
    }

    assert set(RIGHT_ARM_JOINTS).issubset(monitored_parent_joints)
    assert any(name.startswith("gripper_right_") for name in monitored_geometry_names)
    assert "arm_left_1_joint" not in monitored_parent_joints
    assert len(runtime_collision_model.geometry_model.collisionPairs) < len(
        collision_model.geometry_model.collisionPairs
    )
    for pair in runtime_collision_model.geometry_model.collisionPairs:
        assert int(pair.first) in monitored or int(pair.second) in monitored


def test_collision_margin_violation_matches_minimum_distance():
    _skip_without_urdf()
    collision_model = build_tiago_collision_model(
        urdf_path=URDF_PATH,
        monitor_only_collision_pairs=False,
    )
    q, _ = state_to_qv(collision_model.model, _zero_state(collision_model))
    minimum = compute_minimum_pair_distance(collision_model, q)

    violation = compute_collision_margin_violation(
        collision_model,
        q,
        margin_m=0.0,
    )

    assert (violation is not None) == (minimum.distance_m <= 0.0)
    if violation is not None:
        assert violation.margin_m == pytest.approx(0.0)
        violations = compute_collision_margin_violations(
            collision_model,
            q,
            margin_m=0.0,
        )
        assert violations
        assert violations[0].margin_m == pytest.approx(0.0)
    with pytest.raises(ValueError, match="margin_m"):
        compute_collision_margin_violation(collision_model, q, margin_m=-1e-3)


def test_joint_limit_check_accepts_state_inside_limits(collision_model):
    state = _zero_state(collision_model)

    assert check_joint_limits(collision_model.model, state) == []


def test_joint_limit_check_reports_position_and_velocity_violations(collision_model):
    state = _zero_state(collision_model)
    joint_name = "arm_right_1_joint"
    joint_id = collision_model.model.getJointId(joint_name)
    joint = collision_model.model.joints[joint_id]
    upper = float(collision_model.model.upperPositionLimit[joint.idx_q])
    velocity_limit = float(collision_model.model.velocityLimit[joint.idx_v])
    state.position[joint_name] = upper + 0.01
    state.velocity[joint_name] = velocity_limit + 0.01

    violations = check_joint_limits(collision_model.model, state)

    assert [violation.kind for violation in violations] == ["position_upper", "velocity"]
    assert all(violation.joint == joint_name for violation in violations)


def test_joint_limit_check_honors_position_margin(collision_model):
    state = _zero_state(collision_model)
    joint_name = "arm_right_1_joint"
    joint_id = collision_model.model.getJointId(joint_name)
    joint = collision_model.model.joints[joint_id]
    upper = float(collision_model.model.upperPositionLimit[joint.idx_q])
    state.position[joint_name] = upper - 0.005

    violations = check_joint_limits(
        collision_model.model,
        state,
        position_margin=0.01,
    )

    assert any(
        violation.joint == joint_name and violation.kind == "position_upper"
        for violation in violations
    )


def test_joint_limit_check_can_be_restricted_to_controlled_joints(collision_model):
    state = _zero_state(collision_model)
    left_joint_name = "arm_left_1_joint"
    left_joint_id = collision_model.model.getJointId(left_joint_name)
    left_joint = collision_model.model.joints[left_joint_id]
    left_upper = float(collision_model.model.upperPositionLimit[left_joint.idx_q])
    state.position[left_joint_name] = left_upper + 0.01

    assert (
        check_joint_limits(
            collision_model.model,
            state,
            joint_names=RIGHT_ARM_JOINTS,
        )
        == []
    )
    assert check_joint_limits(collision_model.model, state)


def test_blacklist_removes_named_collision_pair(tmp_path, collision_model):
    pair = collision_model.geometry_model.collisionPairs[0]
    object_a = collision_model.geometry_model.geometryObjects[pair.first]
    object_b = collision_model.geometry_model.geometryObjects[pair.second]
    blacklist_path = tmp_path / "blacklist.json"
    blacklist_path.write_text(
        json.dumps(
            {
                "ignored_geometry_pairs": [
                    {
                        "geometry_a": object_a.name,
                        "geometry_b": object_b.name,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    filtered = build_tiago_collision_model(
        urdf_path=URDF_PATH,
        blacklist_path=blacklist_path,
        monitor_only_collision_pairs=False,
    )

    assert len(filtered.geometry_model.collisionPairs) == (
        len(collision_model.geometry_model.collisionPairs) - 1
    )
    remaining = {
        tuple(
            sorted(
                (
                    filtered.geometry_model.geometryObjects[pair.first].name,
                    filtered.geometry_model.geometryObjects[pair.second].name,
                )
            )
        )
        for pair in filtered.geometry_model.collisionPairs
    }
    assert tuple(sorted((object_a.name, object_b.name))) not in remaining
