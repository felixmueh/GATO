import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pinocchio")

from gato_tiago.safety_monitor import (
    TIAGO_RUNTIME_STATE_FIXED_JOINT_NAMES,
    NamedJointState,
    build_tiago_collision_model,
    check_joint_limits,
    compute_collision_body_speeds,
    compute_collision_margin_violation,
    compute_collision_margin_violations,
    compute_minimum_pair_distance,
    compute_pair_distances,
    state_to_qv,
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
