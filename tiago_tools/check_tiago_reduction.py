#!/usr/bin/env python3
"""Compare the extracted/lumped arm with the original URDF using Pinocchio.

Run: python tiago_tools/check_tiago_reduction.py
All non-arm joints are locked at neutral (gripper joints at zero). This checks
the native URDF, not the GRiD axis conversion or generated CUDA dynamics.
"""

from pathlib import Path

import numpy as np
import pinocchio as pin


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_URDF = ROOT / "tiago_tools/sim_tiago_urdf.xml"
ARM_URDF = ROOT / "gato/dynamics/tiago_right/tiago_right_arm.urdf"
ARM_JOINTS = [f"arm_right_{i}_joint" for i in range(1, 8)]
ARM_FRAMES = [f"arm_right_{i}_link" for i in range(1, 8)] + ["arm_right_tool_link"]
SAMPLES = 1000
SEED = 42
ATOL = 1e-8
RTOL = 1e-9
DT = 0.001
ROLLOUT_STEPS = 100


def quantities(model, data, q, v, a, torque):
    # Copy results: Pinocchio reuses the arrays in data on subsequent calls.
    result = {
        "mass matrix": pin.crba(model, data, q).copy(),
        "gravity": pin.computeGeneralizedGravity(model, data, q).copy(),
        "nonlinear effects": pin.nonLinearEffects(model, data, q, v).copy(),
        "inverse dynamics": pin.rnea(model, data, q, v, a).copy(),
        "forward dynamics": pin.aba(model, data, q, v, torque).copy(),
    }
    pin.framesForwardKinematics(model, data, q)
    torso_inverse = data.oMf[model.getFrameId("torso_lift_link")].inverse()
    result["arm/tool poses"] = np.stack([
        (torso_inverse * data.oMf[model.getFrameId(name)]).homogeneous
        for name in ARM_FRAMES
    ])
    return result


def main():
    full = pin.buildModelFromUrdf(str(ORIGINAL_URDF))
    locked = [j for j in range(1, full.njoints) if full.names[j] not in ARM_JOINTS]
    reference = pin.buildReducedModel(full, locked, pin.neutral(full))
    arm = pin.buildModelFromUrdf(str(ARM_URDF))
    reference_data, arm_data = reference.createData(), arm.createData()

    # These models use the same named, scalar joint order. Fail clearly if that
    # changes, rather than silently comparing different coordinates.
    for model in (reference, arm):
        assert model.nq == model.nv == 7
        assert list(model.names)[1:] == ARM_JOINTS
        for name in ["torso_lift_link", *ARM_FRAMES]:
            assert model.existFrame(name), name

    # The full model is rooted at the robot base; the extracted model at torso.
    # Express the same physical gravity in the extracted model's root frame.
    pin.framesForwardKinematics(reference, reference_data, pin.neutral(reference))
    torso_rotation = reference_data.oMf[reference.getFrameId("torso_lift_link")].rotation
    arm.gravity.linear = torso_rotation.T @ reference.gravity.linear

    worst = {}

    def compare(name, actual, expected, case):
        np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL,
                                   equal_nan=False, err_msg=f"{case}: {name}")
        assert np.all(np.isfinite(actual)) and np.all(np.isfinite(expected)), case
        worst[name] = max(worst.get(name, 0.0), float(np.max(np.abs(actual - expected))))

    for name in ("lowerPositionLimit", "upperPositionLimit", "velocityLimit", "effortLimit"):
        compare(name, getattr(arm, name), getattr(reference, name), "joint limits")

    rng = np.random.default_rng(SEED)
    lower, upper = reference.lowerPositionLimit, reference.upperPositionLimit
    middle = (lower + upper) / 2
    configurations = [np.clip(np.zeros(7), lower, upper), middle,
                      lower + 1e-6, upper - 1e-6]
    # Explicitly exercise joint 6's native negative axis in both directions.
    for angle in (-1.0, 0.0, 1.0):
        q = middle.copy()
        q[5] = angle
        configurations.append(q)
    configurations.extend(rng.uniform(lower, upper, size=(SAMPLES, 7)))

    for i, q in enumerate(configurations):
        v = rng.uniform(-0.5, 0.5, 7)
        a = rng.uniform(-2.0, 2.0, 7)
        torque = rng.uniform(-1.0, 1.0, 7)
        expected = quantities(reference, reference_data, q, v, a, torque)
        actual = quantities(arm, arm_data, q, v, a, torque)
        for name in expected:
            compare(name, actual[name], expected[name], f"sample {i}")

    # Each model advances its own state. Both receive the same torque sequence
    # and use semi-implicit Euler; small excitation keeps these rollouts short
    # and bounded. Gravity compensation is computed from the reference only.
    for rollout in range(3):
        q_ref = rng.uniform(lower + 0.2 * (upper - lower), upper - 0.2 * (upper - lower))
        v_ref = rng.uniform(-0.1, 0.1, 7)
        q_arm, v_arm = q_ref.copy(), v_ref.copy()
        phase = rng.uniform(-np.pi, np.pi, 7)
        for step in range(ROLLOUT_STEPS):
            torque = pin.computeGeneralizedGravity(reference, reference_data, q_ref).copy()
            torque += 0.05 * np.sin(10 * step * DT + phase)
            a_ref = pin.aba(reference, reference_data, q_ref, v_ref, torque).copy()
            a_arm = pin.aba(arm, arm_data, q_arm, v_arm, torque).copy()
            v_ref += DT * a_ref
            v_arm += DT * a_arm
            q_ref = pin.integrate(reference, q_ref, DT * v_ref)
            q_arm = pin.integrate(arm, q_arm, DT * v_arm)
            case = f"rollout {rollout}, step {step}"
            compare("rollout position", q_arm, q_ref, case)
            compare("rollout velocity", v_arm, v_ref, case)
            pin.framesForwardKinematics(reference, reference_data, q_ref)
            pin.framesForwardKinematics(arm, arm_data, q_arm)
            poses = []
            for model, data in ((reference, reference_data), (arm, arm_data)):
                torso_inverse = data.oMf[model.getFrameId("torso_lift_link")].inverse()
                poses.append((torso_inverse * data.oMf[model.getFrameId("arm_right_tool_link")]).homogeneous)
            compare("rollout tool pose", poses[1], poses[0], case)

    print(f"PASS: {len(configurations)} states and 3 x {ROLLOUT_STEPS} rollout steps (seed {SEED})")
    print(f"Tolerance: atol={ATOL:g}, rtol={RTOL:g}. Maximum absolute differences:")
    for name, error in worst.items():
        print(f"  {name}: {error:.3e}")


if __name__ == "__main__":
    main()
