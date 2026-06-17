import json
from pathlib import Path

import pytest

pytest.importorskip("pinocchio")

from gato_tiago.safety_monitor import (
    DEFAULT_LOCKED_JOINTS,
    NamedJointState,
    build_tiago_collision_model,
    check_joint_limits,
    compute_collision_body_speeds,
    compute_pair_distances,
    state_to_qv,
)


URDF_PATH = Path("/workspace/GATO/TiagoProURDF/tiago_pro.urdf")


def _skip_without_urdf() -> None:
    if not URDF_PATH.is_file():
        pytest.skip(f"missing Tiago URDF at {URDF_PATH}")


@pytest.fixture(scope="module")
def collision_model():
    _skip_without_urdf()
    return build_tiago_collision_model(urdf_path=URDF_PATH)


def _zero_state(collision_model):
    return NamedJointState(
        position={name: 0.0 for name in collision_model.unlocked_joint_names},
        velocity={name: 0.0 for name in collision_model.unlocked_joint_names},
    )


def test_reduced_model_locks_base_wheel_joints(collision_model):
    assert collision_model.locked_joint_names == DEFAULT_LOCKED_JOINTS
    assert not set(DEFAULT_LOCKED_JOINTS).intersection(collision_model.unlocked_joint_names)
    assert "torso_lift_joint" in collision_model.unlocked_joint_names
    assert len(collision_model.geometry_model.geometryObjects) == len(
        collision_model.geometry_radii_m
    )
    assert len(collision_model.geometry_model.collisionPairs) > 0


def test_state_to_qv_requires_every_unlocked_joint(collision_model):
    state = _zero_state(collision_model)
    missing = collision_model.unlocked_joint_names[0]
    del state.position[missing]

    with pytest.raises(ValueError, match=missing):
        state_to_qv(collision_model.model, state)


def test_distance_and_speed_reports_cover_collision_model(collision_model):
    q, qd = state_to_qv(collision_model.model, _zero_state(collision_model))

    distances = compute_pair_distances(collision_model, q)
    speeds = compute_collision_body_speeds(collision_model, q, qd)

    assert len(distances) == len(collision_model.geometry_model.collisionPairs)
    assert len(speeds) == len(collision_model.geometry_model.geometryObjects)
    assert all(report.geometry_a for report in distances)
    assert all(report.geometry_b for report in distances)
    assert all(report.speed_bound_m_s == pytest.approx(0.0) for report in speeds)
    assert all(report.radius_m >= 0.0 for report in speeds)


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
