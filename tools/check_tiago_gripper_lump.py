#!/usr/bin/env python3
"""Small Pinocchio check for Tiago gripper inertia lumping."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pinocchio as pin


REPO_ROOT = Path(__file__).resolve().parents[1]
ARM_JOINTS = [f"arm_right_{idx}_joint" for idx in range(1, 8)]


def set_joints(model, names, values):
    q = pin.neutral(model)
    for name, value in zip(names, values):
        joint = model.joints[model.getJointId(name)]
        q[joint.idx_q] = value
    return q


def set_velocities(model, names, values):
    v = np.zeros(model.nv)
    for name, value in zip(names, values):
        joint = model.joints[model.getJointId(name)]
        v[joint.idx_v] = value
    return v


def arm_limits(model):
    lower = []
    upper = []
    for name in ARM_JOINTS:
        joint = model.joints[model.getJointId(name)]
        lower.append(model.lowerPositionLimit[joint.idx_q])
        upper.append(model.upperPositionLimit[joint.idx_q])
    return np.array(lower), np.array(upper)


def arm_rows(model):
    return [model.joints[model.getJointId(name)].idx_v for name in ARM_JOINTS]


def total_mass(model):
    return sum(float(inertia.mass) for inertia in model.inertias)


def frame_vector(model, data, q, name):
    pin.framesForwardKinematics(model, data, q)
    pose = data.oMf[model.getFrameId(name)]
    return np.concatenate([pose.translation, pose.rotation.reshape(-1)])


def check_case(sum_model, include_model, q_arm, v_arm):
    sum_data = sum_model.createData()
    include_data = include_model.createData()
    q_sum = set_joints(sum_model, ARM_JOINTS, q_arm)
    q_include = set_joints(include_model, ARM_JOINTS, q_arm)
    v_sum = set_velocities(sum_model, ARM_JOINTS, v_arm)
    v_include = set_velocities(include_model, ARM_JOINTS, v_arm)
    rows = arm_rows(include_model)

    m_sum = pin.crba(sum_model, sum_data, q_sum)
    m_include = pin.crba(include_model, include_data, q_include)
    g_sum = pin.computeGeneralizedGravity(sum_model, sum_data, q_sum)
    g_include = pin.computeGeneralizedGravity(include_model, include_data, q_include)
    nle_sum = pin.nonLinearEffects(sum_model, sum_data, q_sum, v_sum)
    nle_include = pin.nonLinearEffects(include_model, include_data, q_include, v_include)
    com_sum = pin.centerOfMass(sum_model, sum_data, q_sum)
    com_include = pin.centerOfMass(include_model, include_data, q_include)

    return {
        "mass_matrix_arm": float(np.max(np.abs(m_sum - m_include[np.ix_(rows, rows)]))),
        "gravity_arm": float(np.max(np.abs(g_sum - g_include[rows]))),
        "nle_arm": float(np.max(np.abs(nle_sum - nle_include[rows]))),
        "com": float(np.linalg.norm(com_sum - com_include)),
        "tool_frame": float(
            np.linalg.norm(
                frame_vector(sum_model, sum_data, q_sum, "arm_right_tool_link")
                - frame_vector(include_model, include_data, q_include, "arm_right_tool_link")
            )
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sum", type=Path, default=REPO_ROOT / "gato/dynamics/tiago_right/tiago_right_arm.urdf")
    parser.add_argument(
        "--include",
        type=Path,
        default=REPO_ROOT / "gato/dynamics/tiago_right/tiago_right_arm_include_gripper.urdf",
    )
    parser.add_argument("--tolerance", type=float, default=1e-8)
    args = parser.parse_args()

    sum_model = pin.buildModelFromUrdf(str(args.sum))
    include_model = pin.buildModelFromUrdf(str(args.include))
    lower, upper = arm_limits(sum_model)
    cases = [
        (np.clip(np.zeros(7), lower, upper), np.zeros(7)),
        (lower + 0.33 * (upper - lower), np.linspace(-0.2, 0.2, 7)),
        (lower + 0.71 * (upper - lower), np.linspace(0.3, -0.1, 7)),
    ]

    print(f"sum nq/nv={sum_model.nq}/{sum_model.nv} include nq/nv={include_model.nq}/{include_model.nv}")
    print(f"total_mass_diff={abs(total_mass(sum_model) - total_mass(include_model)):.3e}")
    failures = []
    for index, (q_arm, v_arm) in enumerate(cases):
        errors = check_case(sum_model, include_model, q_arm, v_arm)
        print(f"case {index}: " + " ".join(f"{name}={value:.3e}" for name, value in errors.items()))
        failures.extend((name, value) for name, value in errors.items() if value > args.tolerance)

    if failures:
        print(f"FAILED tolerance={args.tolerance:g}: {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
