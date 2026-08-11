"""Fail-closed Stage V0 schema for Tiago tool-center/model validation.

This module contains deterministic input generation and pure certification
only.  It does not import the task-construction oracle, an extension module,
or any SQP runner.  Frozen task seeds are represented by identities alone;
their construction remains blocked until a later authorization.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Callable, Mapping, Sequence

import numpy as np

from gato_tiago import multimodal_toll as toll


V0_PROTOCOL_VERSION = "tiago_tool_center_toll_v0_frame_model_preflight_1"
V0_RANDOM_SEED = 20260811
V0_SAMPLE_COUNT = 32
FK_JOINT_MARGIN_RAD = 0.08
DYNAMICS_Q_JITTER_RAD = 0.05
DYNAMICS_QD_BOUND_RAD_S = 0.25
DYNAMICS_CONTROL_FRACTION = 0.20
ONE_STEP_DT = 0.0125

FK_POSITION_TOLERANCE_M = 1e-4
Q_GOAL_TARGET_TOLERANCE_M = 5e-4
ONE_STEP_STATE_L2_TOLERANCE = 1e-3
ONE_STEP_STATE_MAXABS_TOLERANCE = 1e-3
DENSE_TOOL_TOLERANCE_M = 1e-3
DENSE_SUBSTEPS = 64
DENSE_SAMPLE_COUNT = (toll.KNOTS - 1) * DENSE_SUBSTEPS + 1

V0_EXECUTION_AUTHORIZATION = None

EXPECTED_TASK_IDENTITIES = tuple(
    [("development", seed) for seed in toll.DEVELOPMENT_TASK_SEEDS]
    + [("heldout", seed) for seed in toll.HELDOUT_TASK_SEEDS]
)

EXPECTED_EXTENSION_MANIFEST = {
    "bsqpN64_tiago_right_multimodal_toll": {
        "knots": 64,
        "reference_size": 10,
        "tool_position": True,
        "tool_position_frame": "arm_right_tool_joint_origin",
    },
    "bsqpN64_tiago_right": {
        "knots": 64,
        "reference_size": 6,
        "tool_position": True,
        "tool_position_frame": "arm_right_tool_joint_origin",
    },
    "bsqpN64_tiago_right_multimodal": {
        "knots": 64,
        "reference_size": 6,
        "tool_position": True,
        "tool_position_frame": "arm_right_tool_joint_origin",
    },
}

REQUIRED_SOURCE_PATHS = {
    "bindings": "python/bindings.cu",
    "bsqp": "gato/bsqp/bsqp.cuh",
    "build_script": "tools/build.sh",
    "cmake": "CMakeLists.txt",
    "integrator": "gato/dynamics/integrator.cuh",
    "sim_kernel": "gato/bsqp/kernels/sim.cuh",
    "stage0_cli": "tiago_examples/tiago_multimodal_toll_stage0.py",
    "tiago_grid": "gato/dynamics/tiago_right/tiago_right_grid.cuh",
    "tiago_limits": "gato/dynamics/tiago_right/tiago_right_limits.cuh",
    "tiago_plant": "gato/dynamics/tiago_right/tiago_right_plant.cuh",
    "toll_oracle_schema": "tiago_src/gato_tiago/multimodal_toll_oracle_schema.py",
    "toll_schema": "tiago_src/gato_tiago/multimodal_toll.py",
    "toll_tests": "tests/python/test_tiago_multimodal_toll.py",
    "tool_position_kernel": "gato/bsqp/kernels/tool_position.cuh",
    "v0_schema": "tiago_src/gato_tiago/multimodal_toll_v0.py",
    "v0_tests": "tests/python/test_tiago_multimodal_toll_v0.py",
}
REQUIRED_SOURCE_HASH_LABELS = frozenset(REQUIRED_SOURCE_PATHS)


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = f"{array.dtype.str}|{array.shape}|".encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def _is_sha256(value) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _exact_array(value, shape, dtype):
    array = np.asarray(value)
    if array.shape != shape or array.dtype != np.dtype(dtype):
        raise ValueError(f"array must have exact shape {shape} and dtype {np.dtype(dtype)}")
    if not np.all(np.isfinite(array)):
        raise ValueError("array must be finite")
    return array


@dataclass(frozen=True)
class V0Draws:
    lower: np.ndarray
    upper: np.ndarray
    effort: np.ndarray
    comfortable_q: np.ndarray
    fk_q_float64: np.ndarray
    dynamics_q_offsets_float64: np.ndarray
    dynamics_q_float64: np.ndarray
    dynamics_qd_float64: np.ndarray
    dynamics_control_fractions_float64: np.ndarray
    dynamics_u_float64: np.ndarray
    fk_q_float32: np.ndarray
    dynamics_x_float32: np.ndarray
    dynamics_u_float32: np.ndarray

    def array_hashes(self) -> dict[str, str]:
        return {
            name: _array_hash(value)
            for name, value in self.__dict__.items()
            if isinstance(value, np.ndarray)
        }


def generate_v0_draws(lower, upper, effort, comfortable_q) -> V0Draws:
    """Generate the one frozen V0 draw family without constructing tasks."""

    lower = _exact_array(np.asarray(lower, dtype=np.float64), (7,), np.float64).copy()
    upper = _exact_array(np.asarray(upper, dtype=np.float64), (7,), np.float64).copy()
    effort = _exact_array(np.asarray(effort, dtype=np.float64), (7,), np.float64).copy()
    comfortable_q = _exact_array(
        np.asarray(comfortable_q, dtype=np.float64), (7,), np.float64
    ).copy()
    if np.any(lower + 2 * FK_JOINT_MARGIN_RAD >= upper):
        raise ValueError("joint range cannot support the frozen 0.08-rad margin")
    if np.any(effort <= 0):
        raise ValueError("effort limits must be positive")

    rng = np.random.default_rng(V0_RANDOM_SEED)
    fk_q = rng.uniform(
        lower + FK_JOINT_MARGIN_RAD,
        upper - FK_JOINT_MARGIN_RAD,
        size=(V0_SAMPLE_COUNT, 7),
    )
    q_offsets = rng.uniform(
        -DYNAMICS_Q_JITTER_RAD,
        DYNAMICS_Q_JITTER_RAD,
        size=(V0_SAMPLE_COUNT, 7),
    )
    dynamics_q = comfortable_q[None, :] + q_offsets
    # No clipping or replacement is permitted. A model/configuration mismatch
    # is a hard pre-data failure.
    if np.any(dynamics_q < lower + FK_JOINT_MARGIN_RAD) or np.any(
        dynamics_q > upper - FK_JOINT_MARGIN_RAD
    ):
        raise ValueError("comfortable_high_clearance jitter violates frozen joint margins")
    dynamics_qd = rng.uniform(
        -DYNAMICS_QD_BOUND_RAD_S,
        DYNAMICS_QD_BOUND_RAD_S,
        size=(V0_SAMPLE_COUNT, 7),
    )
    control_fractions = rng.uniform(
        -DYNAMICS_CONTROL_FRACTION,
        DYNAMICS_CONTROL_FRACTION,
        size=(V0_SAMPLE_COUNT, 7),
    )
    dynamics_u = control_fractions * effort[None, :]
    dynamics_x = np.concatenate([dynamics_q, dynamics_qd], axis=1)

    return V0Draws(
        lower=lower,
        upper=upper,
        effort=effort,
        comfortable_q=comfortable_q,
        fk_q_float64=fk_q,
        dynamics_q_offsets_float64=q_offsets,
        dynamics_q_float64=dynamics_q,
        dynamics_qd_float64=dynamics_qd,
        dynamics_control_fractions_float64=control_fractions,
        dynamics_u_float64=dynamics_u,
        fk_q_float32=fk_q.astype(np.float32),
        dynamics_x_float32=dynamics_x.astype(np.float32),
        dynamics_u_float32=dynamics_u.astype(np.float32),
    )


def frozen_v0_metadata() -> dict:
    """Static metadata only; this function never constructs a frozen task."""

    return {
        "protocol_version": V0_PROTOCOL_VERSION,
        "execution_authorized": V0_EXECUTION_AUTHORIZATION is not None,
        "rng": "numpy.random.default_rng",
        "random_seed": V0_RANDOM_SEED,
        "sample_count": V0_SAMPLE_COUNT,
        "fk_joint_margin_rad": FK_JOINT_MARGIN_RAD,
        "dynamics_q_jitter_rad": [-DYNAMICS_Q_JITTER_RAD, DYNAMICS_Q_JITTER_RAD],
        "dynamics_qd_rad_s": [-DYNAMICS_QD_BOUND_RAD_S, DYNAMICS_QD_BOUND_RAD_S],
        "dynamics_control_effort_fraction": [
            -DYNAMICS_CONTROL_FRACTION,
            DYNAMICS_CONTROL_FRACTION,
        ],
        "one_step_dt": ONE_STEP_DT,
        "independent_pin_one_step": {
            "acceleration": "pinocchio.aba(model,data,q,qd,u)",
            "external_wrench": "zero",
            "position": "q_next=q+dt*qd+0.5*dt^2*qdd",
            "velocity": "qd_next=qd+dt*qdd",
        },
        "expected_task_identities": [list(row) for row in EXPECTED_TASK_IDENTITIES],
        "task_rows_status": "identity_only_not_instantiated",
        "expected_extension_manifest": EXPECTED_EXTENSION_MANIFEST,
        "collision_scope": toll.COLLISION_SCOPE,
        "pinocchio_tool_frame": toll.TOOL_FRAME,
        "pinocchio_base_frame": toll.TORSO_FRAME,
        "cuda_tool_query_frame": "arm_right_tool_joint_origin",
    }


def constant_acceleration_step(x, qdd, dt=ONE_STEP_DT):
    """Independent Pin replay step matching GATO integrator type 2."""

    state = _exact_array(np.asarray(x), (14,), np.float64)
    acceleration = _exact_array(np.asarray(qdd), (7,), np.float64)
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    q = state[:7]
    qd = state[7:]
    return np.concatenate(
        [q + dt * qd + 0.5 * dt * dt * acceleration, qd + dt * acceleration]
    )


def pin_one_step_from_aba(aba: Callable, x, u, dt=ONE_STEP_DT):
    """Call one zero-wrench ABA function and apply the frozen integrator."""

    state = _exact_array(np.asarray(x), (14,), np.float64)
    control = _exact_array(np.asarray(u), (7,), np.float64)
    qdd = _exact_array(
        np.asarray(aba(state[:7], state[7:], control), dtype=np.float64),
        (7,),
        np.float64,
    )
    return constant_acceleration_step(state, qdd, dt)


def _draws_are_authentic(draws):
    if not isinstance(draws, V0Draws):
        return False, None
    try:
        regenerated = generate_v0_draws(
            draws.lower, draws.upper, draws.effort, draws.comfortable_q
        )
    except (TypeError, ValueError):
        return False, None
    names = tuple(V0Draws.__dataclass_fields__)
    exact = all(
        isinstance(getattr(draws, name), np.ndarray)
        and np.array_equal(getattr(draws, name), getattr(regenerated, name))
        for name in names
    )
    exact = exact and draws.array_hashes() == regenerated.array_hashes()
    return exact, regenerated


def _task_gate(row, lower, upper, comfortable_q):
    if not isinstance(row, Mapping):
        return {"identity": None, "passes": False, "input_valid": False}
    identity = (row.get("phase"), row.get("task_seed"))
    try:
        x0 = _exact_array(row["solver_x0"], (14,), np.float32)
        reference = _exact_array(row["solver_reference"], (10,), np.float32)
        q_goal = _exact_array(row["q_goal_float32"], (7,), np.float32)
        q_goal64_retained = _exact_array(row["q_goal_float64"], (7,), np.float64)
        dq_retained = _exact_array(row["dq_float64"], (7,), np.float64)
        requested_delta_retained = _exact_array(
            row["requested_delta_xyz_float64"], (3,), np.float64
        )
        jacobian = _exact_array(row["pin_position_jacobian_float64"], (3, 7), np.float64)
        q0_jitter = _exact_array(row["q0_jitter_float64"], (7,), np.float64)
        phi = float(row["phi_float64"])
        offset_sign = int(row["offset_sign"])
        pin_start = _exact_array(row["pin_start_tool_xyz"], (3,), np.float64)
        cuda_start = _exact_array(row["cuda_start_tool_xyz"], (3,), np.float32)
        pin_goal = _exact_array(row["pin_q_goal_tool_xyz"], (3,), np.float64)
        cuda_goal = _exact_array(row["cuda_q_goal_tool_xyz"], (3,), np.float32)
        toll.validate_reference_row(reference)
    except (KeyError, TypeError, ValueError):
        return {"identity": identity, "passes": False, "input_valid": False}

    try:
        task_seed = int(row["task_seed"])
        rng = np.random.default_rng(task_seed)
        expected_jitter = rng.uniform(-toll.Q0_JITTER_RAD, toll.Q0_JITTER_RAD, size=7)
        expected_phi = float(
            rng.uniform(-toll.PLANAR_ANGLE_RAD, toll.PLANAR_ANGLE_RAD)
        )
        expected_sign = int(rng.choice((-1, 1)))
        expected_q0 = np.asarray(comfortable_q, dtype=np.float64) + expected_jitter
        expected_delta = toll.REQUESTED_TRAVEL_M * np.asarray(
            [np.cos(expected_phi), np.sin(expected_phi), 0.0], dtype=np.float64
        )
        expected_dq = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + toll.DAMPING**2 * np.eye(3), expected_delta
        )
        expected_q_goal64 = expected_q0 + expected_dq
        expected_x0 = np.concatenate([expected_q0, np.zeros(7)]).astype(np.float32)
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
        return {"identity": identity, "passes": False, "input_valid": False}

    planar = pin_goal[:2] - pin_start[:2]
    planar_norm = float(np.linalg.norm(planar))
    if planar_norm > 0:
        direction = planar / planar_norm
        normal = np.asarray([-direction[1], direction[0]])
        expected_center = (
            0.5 * (pin_start[:2] + pin_goal[:2])
            + expected_sign * toll.CYLINDER_OFFSET_M * normal
        )
        expected_default_side = -expected_sign
        expected_toll = (
            expected_center
            + expected_default_side * toll.TOLL_OFFSET_FROM_CYLINDER_M * normal
        )
        expected_reference = toll.TollReference(
            goal_xyz=tuple(pin_goal),
            cylinder_xy=tuple(expected_center),
            physical_radius_m=toll.PHYSICAL_RADIUS_M,
            toll_xy=tuple(expected_toll),
            toll_sigma_m=toll.TOLL_SIGMA_M,
            clearance_margin_m=toll.CLEARANCE_MARGIN_M,
        ).as_float32()
    else:
        expected_center = np.full(2, np.nan)
        expected_default_side = 0
        expected_reference = np.full(10, np.nan, dtype=np.float32)

    ref = toll.TollReference.from_solver_bytes(reference)
    goal = np.asarray(ref.goal_xyz, dtype=np.float64)
    q0 = x0[:7].astype(np.float64)
    q_goal64 = q_goal.astype(np.float64)
    start_frame_error = float(np.linalg.norm(pin_start - cuda_start.astype(np.float64)))
    goal_frame_error = float(np.linalg.norm(pin_goal - cuda_goal.astype(np.float64)))
    pin_target_error = float(np.linalg.norm(pin_goal - goal))
    cuda_target_error = float(np.linalg.norm(cuda_goal.astype(np.float64) - goal))
    full_travel = float(np.linalg.norm(pin_goal - pin_start))
    planar_travel = float(np.linalg.norm(pin_goal[:2] - pin_start[:2]))
    vertical_travel = float(abs(pin_goal[2] - pin_start[2]))

    center = np.asarray(ref.cylinder_xy, dtype=np.float64)
    physical_clearances = np.asarray(
        [np.linalg.norm(pin_start[:2] - center), np.linalg.norm(pin_goal[:2] - center)]
    ) - ref.physical_radius_m
    chord = pin_goal[:2] - pin_start[:2]
    chord_norm_sq = float(chord @ chord)
    if chord_norm_sq > 0:
        alpha = float(np.clip((center - pin_start[:2]) @ chord / chord_norm_sq, 0.0, 1.0))
        chord_distance = float(np.linalg.norm(pin_start[:2] + alpha * chord - center))
    else:
        chord_distance = float("inf")

    numeric = np.asarray(
        [
            start_frame_error,
            goal_frame_error,
            pin_target_error,
            cuda_target_error,
            full_travel,
            planar_travel,
            vertical_travel,
            *physical_clearances,
            chord_distance,
        ]
    )
    gates = {
        "identity": identity,
        "input_valid": True,
        "zero_retry_replacement": row.get("retry_count") == 0
        and row.get("replacement_count") == 0,
        "solver_byte_hashes_match": row.get("solver_x0_sha256") == _array_hash(x0)
        and row.get("solver_reference_sha256") == _array_hash(reference)
        and row.get("q_goal_sha256") == _array_hash(q_goal)
        and row.get("construction_draws_sha256")
        == _array_hash(np.concatenate([q0_jitter, [phi, float(offset_sign)]])),
        "construction_draws_in_frozen_domain": bool(
            np.all(np.abs(q0_jitter) <= toll.Q0_JITTER_RAD)
            and -toll.PLANAR_ANGLE_RAD <= phi <= toll.PLANAR_ANGLE_RAD
            and offset_sign in (-1, 1)
        ),
        "exact_task_rng_draws": bool(
            np.array_equal(q0_jitter, expected_jitter)
            and phi == expected_phi
            and offset_sign == expected_sign
        ),
        "exact_dls_witness": bool(
            np.array_equal(requested_delta_retained, expected_delta)
            and np.array_equal(dq_retained, expected_dq)
            and np.array_equal(q_goal64_retained, expected_q_goal64)
            and np.array_equal(q_goal, expected_q_goal64.astype(np.float32))
            and np.array_equal(x0, expected_x0)
        ),
        "exact_reconstructed_reference": bool(
            np.array_equal(reference, expected_reference)
            and row.get("default_side") == expected_default_side
        ),
        "zero_initial_velocity": bool(np.array_equal(x0[7:], np.zeros(7, dtype=np.float32))),
        "joint_margins": bool(
            np.all(q0 >= lower + toll.GOAL_JOINT_MARGIN_RAD)
            and np.all(q0 <= upper - toll.GOAL_JOINT_MARGIN_RAD)
            and np.all(q_goal64 >= lower + toll.GOAL_JOINT_MARGIN_RAD)
            and np.all(q_goal64 <= upper - toll.GOAL_JOINT_MARGIN_RAD)
        ),
        "tool_frame_agreement": start_frame_error <= FK_POSITION_TOLERANCE_M
        and goal_frame_error <= FK_POSITION_TOLERANCE_M,
        "q_goal_hits_target": pin_target_error <= Q_GOAL_TARGET_TOLERANCE_M
        and cuda_target_error <= Q_GOAL_TARGET_TOLERANCE_M,
        "full_tool_travel_in_range": toll.ACTUAL_TRAVEL_RANGE_M[0]
        <= full_travel
        <= toll.ACTUAL_TRAVEL_RANGE_M[1],
        "vertical_travel_in_range": vertical_travel <= toll.MAX_VERTICAL_TRAVEL_M,
        "endpoint_physical_clearance": bool(
            np.min(physical_clearances) >= toll.MIN_ENDPOINT_PHYSICAL_CLEARANCE_M
        ),
        "direct_chord_intersects_physical_cylinder": chord_distance
        < ref.physical_radius_m,
        "direct_chord_intersects_inflated_keepout": chord_distance
        < ref.optimizer_radius_m,
        "finite": bool(np.all(np.isfinite(numeric))),
    }
    gates.update(
        {
            "start_frame_error_m": start_frame_error,
            "q_goal_frame_error_m": goal_frame_error,
            "pin_q_goal_target_error_m": pin_target_error,
            "cuda_q_goal_target_error_m": cuda_target_error,
            "full_tool_travel_m": full_travel,
            "planar_tool_travel_m": planar_travel,
            "vertical_tool_travel_m": vertical_travel,
            "endpoint_physical_clearance_m": physical_clearances.tolist(),
            "direct_chord_center_distance_m": chord_distance,
        }
    )
    gate_names = (
        "zero_retry_replacement",
        "solver_byte_hashes_match",
        "construction_draws_in_frozen_domain",
        "exact_task_rng_draws",
        "exact_dls_witness",
        "exact_reconstructed_reference",
        "zero_initial_velocity",
        "joint_margins",
        "tool_frame_agreement",
        "q_goal_hits_target",
        "full_tool_travel_in_range",
        "vertical_travel_in_range",
        "endpoint_physical_clearance",
        "direct_chord_intersects_physical_cylinder",
        "direct_chord_intersects_inflated_keepout",
        "finite",
    )
    gates["passes"] = all(gates[name] for name in gate_names)
    return gates


def _extension_manifest_gate(extension_manifest, toll_extension_sha256, source_commit):
    if not isinstance(extension_manifest, Mapping) or set(extension_manifest) != set(
        EXPECTED_EXTENSION_MANIFEST
    ):
        return False
    for module_name, expected in EXPECTED_EXTENSION_MANIFEST.items():
        row = extension_manifest.get(module_name)
        if not isinstance(row, Mapping):
            return False
        if any(row.get(key) != value for key, value in expected.items()):
            return False
        if row.get("import_smoke_pass") is not True or not _is_sha256(
            row.get("extension_sha256")
        ):
            return False
        try:
            pin_fk_max_error = float(row.get("pin_fk_max_error_m", np.inf))
        except (TypeError, ValueError):
            return False
        if (
            row.get("native_reference_shape_accepted") is not True
            or row.get("wrong_reference_shape_rejected") is not True
            or row.get("accepted_reference_width") != expected["reference_size"]
            or row.get("rejected_reference_width")
            != (6 if expected["reference_size"] == 10 else 10)
            or row.get("b1_reference_cost_output_parity") is not True
            or row.get("b16_reference_cost_output_parity") is not True
            or row.get("b1_broadcast_fk_parity") is not True
            or row.get("b16_broadcast_fk_parity") is not True
            or row.get("b16_per_lane_fk_parity") is not True
            or not np.isfinite(pin_fk_max_error)
            or pin_fk_max_error < 0
            or pin_fk_max_error > FK_POSITION_TOLERANCE_M
            or row.get("cuda_reference_smoke_pass") is not True
            or not isinstance(row.get("build_command"), str)
            or not row.get("build_command")
            or not isinstance(row.get("test_command"), str)
            or not row.get("test_command")
            or not isinstance(row.get("extension_path"), str)
            or not row.get("extension_path")
            or row.get("source_commit") != source_commit
        ):
            return False
    return (
        extension_manifest["bsqpN64_tiago_right_multimodal_toll"].get(
            "extension_sha256"
        )
        == toll_extension_sha256
    )


def certify_v0(
    *,
    draws: V0Draws,
    captured_fk_q_float32,
    captured_dynamics_x_float32,
    captured_dynamics_u_float32,
    pin_fk_positions,
    cuda_fk_positions,
    pin_qdd_float64,
    pin_next_states,
    cuda_next_states,
    task_rows: Sequence[Mapping],
    extension_manifest: Mapping,
    provenance: Mapping,
) -> dict:
    """Pure V0 certificate over retained future execution outputs."""

    authentic_draws, _ = _draws_are_authentic(draws)

    try:
        captured_fk = _exact_array(captured_fk_q_float32, (32, 7), np.float32)
        captured_x = _exact_array(captured_dynamics_x_float32, (32, 14), np.float32)
        captured_u = _exact_array(captured_dynamics_u_float32, (32, 7), np.float32)
        pin_fk = _exact_array(pin_fk_positions, (32, 3), np.float64)
        cuda_fk = _exact_array(cuda_fk_positions, (32, 3), np.float32)
        pin_qdd = _exact_array(pin_qdd_float64, (32, 7), np.float64)
        pin_next = _exact_array(pin_next_states, (32, 14), np.float64)
        cuda_next = _exact_array(cuda_next_states, (32, 14), np.float32)
        arrays_valid = True
    except (TypeError, ValueError):
        arrays_valid = False
        captured_fk = captured_x = captured_u = np.empty((0,))
        pin_fk = cuda_fk = pin_qdd = pin_next = cuda_next = np.empty((0,))

    exact_inputs = arrays_valid and authentic_draws and all(
        np.array_equal(left, right)
        for left, right in (
            (captured_fk, draws.fk_q_float32),
            (captured_x, draws.dynamics_x_float32),
            (captured_u, draws.dynamics_u_float32),
        )
    )
    if arrays_valid:
        fk_errors = np.linalg.norm(pin_fk - cuda_fk.astype(np.float64), axis=1)
        state_delta = pin_next - cuda_next.astype(np.float64)
        state_l2 = np.linalg.norm(state_delta, axis=1)
        state_maxabs = np.max(np.abs(state_delta), axis=1)
        formula_states = np.stack(
            [
                constant_acceleration_step(
                    np.concatenate(
                        [draws.dynamics_q_float64[index], draws.dynamics_qd_float64[index]]
                    ),
                    pin_qdd[index],
                )
                for index in range(V0_SAMPLE_COUNT)
            ]
        )
        pin_formula_exact = bool(np.array_equal(pin_next, formula_states))
    else:
        fk_errors = state_l2 = state_maxabs = np.asarray([np.inf])
        pin_formula_exact = False

    identities = [
        (row.get("phase"), row.get("task_seed"))
        if isinstance(row, Mapping)
        else None
        for row in task_rows
    ]
    task_gates = [
        _task_gate(row, draws.lower, draws.upper, draws.comfortable_q)
        for row in task_rows
    ]
    tasks_exact = identities == list(EXPECTED_TASK_IDENTITIES)
    all_tasks_pass = tasks_exact and len(task_gates) == 12 and all(
        row.get("passes") is True for row in task_gates
    )

    source_rows = provenance.get("source_hashes", {})
    source_hashes_valid = (
        isinstance(source_rows, Mapping)
        and set(source_rows) == REQUIRED_SOURCE_HASH_LABELS
        and all(
            isinstance(source_rows[label], Mapping)
            and source_rows[label].get("path") == path
            and _is_sha256(source_rows[label].get("sha256"))
            for label, path in REQUIRED_SOURCE_PATHS.items()
        )
    )
    provenance_pass = (
        provenance.get("tracked_tree_clean_at_start") is True
        and provenance.get("tracked_tree_clean_at_end") is True
        and _is_sha256(provenance.get("git_head_at_start"))
        and provenance.get("git_head_at_end") == provenance.get("git_head_at_start")
        and _is_sha256(provenance.get("model_sha256"))
        and provenance.get("model_path") == str(toll.MODEL_PATH)
        and _is_sha256(provenance.get("extension_sha256"))
        and isinstance(provenance.get("exact_command"), str)
        and bool(provenance.get("exact_command"))
        and all(
            isinstance(provenance.get(name), Sequence)
            and not isinstance(provenance.get(name), (str, bytes))
            for name in ("full_git_status_at_start", "full_git_status_at_end")
        )
        and all(
            isinstance(provenance.get(name), str) and bool(provenance.get(name))
            for name in ("python_version", "numpy_version", "pinocchio_version")
        )
        and source_hashes_valid
        and provenance.get("retry_count") == 0
        and provenance.get("replacement_count") == 0
    )
    manifest_pass = _extension_manifest_gate(
        extension_manifest,
        provenance.get("extension_sha256"),
        provenance.get("git_head_at_start"),
    )
    retained_array_hashes = {
        "captured_fk_q_float32": _array_hash(captured_fk),
        "captured_dynamics_x_float32": _array_hash(captured_x),
        "captured_dynamics_u_float32": _array_hash(captured_u),
        "pin_fk_positions_float64": _array_hash(pin_fk),
        "cuda_fk_positions_float32": _array_hash(cuda_fk),
        "pin_qdd_float64": _array_hash(pin_qdd),
        "pin_next_states_float64": _array_hash(pin_next),
        "cuda_next_states_float32": _array_hash(cuda_next),
    }
    task_solver_byte_hashes = [
        {
            "identity": list(identity) if identity is not None else None,
            "solver_x0_sha256": row.get("solver_x0_sha256")
            if isinstance(row, Mapping)
            else None,
            "solver_reference_sha256": row.get("solver_reference_sha256")
            if isinstance(row, Mapping)
            else None,
            "q_goal_sha256": row.get("q_goal_sha256")
            if isinstance(row, Mapping)
            else None,
            "construction_draws_sha256": row.get("construction_draws_sha256")
            if isinstance(row, Mapping)
            else None,
            "q_goal_float64_sha256": _array_hash(row["q_goal_float64"])
            if isinstance(row, Mapping)
            and isinstance(row.get("q_goal_float64"), np.ndarray)
            else None,
            "dq_float64_sha256": _array_hash(row["dq_float64"])
            if isinstance(row, Mapping)
            and isinstance(row.get("dq_float64"), np.ndarray)
            else None,
            "pin_position_jacobian_float64_sha256": _array_hash(
                row["pin_position_jacobian_float64"]
            )
            if isinstance(row, Mapping)
            and isinstance(row.get("pin_position_jacobian_float64"), np.ndarray)
            else None,
        }
        for identity, row in zip(identities, task_rows)
    ]

    gates = {
        "arrays_valid": arrays_valid,
        "draws_independently_regenerated": authentic_draws,
        "exact_solver_bound_input_bytes": exact_inputs,
        "fk_sample_count": int(fk_errors.size if arrays_valid else 0),
        "max_fk_position_error_m": float(np.max(fk_errors)),
        "all_fk_position_gates_pass": bool(
            arrays_valid and np.max(fk_errors) <= FK_POSITION_TOLERANCE_M
        ),
        "max_one_step_state_l2": float(np.max(state_l2)),
        "max_one_step_state_maxabs": float(np.max(state_maxabs)),
        "all_one_step_model_gates_pass": bool(
            arrays_valid
            and pin_formula_exact
            and np.max(state_l2) <= ONE_STEP_STATE_L2_TOLERANCE
            and np.max(state_maxabs) <= ONE_STEP_STATE_MAXABS_TOLERANCE
        ),
        "pin_one_step_formula_exact": pin_formula_exact,
        "task_identities_exact": tasks_exact,
        "task_gates": task_gates,
        "all_task_gates_pass": all_tasks_pass,
        "extension_and_ref6_manifest_pass": manifest_pass,
        "extension_hashes": {
            name: row.get("extension_sha256") if isinstance(row, Mapping) else None
            for name, row in extension_manifest.items()
        }
        if isinstance(extension_manifest, Mapping)
        else {},
        "provenance_pass": provenance_pass,
        "source_hashes_valid": source_hashes_valid,
        "draw_array_hashes": draws.array_hashes(),
        "retained_array_hashes": retained_array_hashes,
        "task_solver_byte_hashes": task_solver_byte_hashes,
        "task_feasibility_independent_of_toll_cost": all_tasks_pass,
        "collision_scope": toll.COLLISION_SCOPE,
        "optimization_solve_calls": 0,
        "oracle_solve_calls": 0,
    }
    gates["all_v0_gates_pass"] = all(
        gates[name]
        for name in (
            "arrays_valid",
            "draws_independently_regenerated",
            "exact_solver_bound_input_bytes",
            "all_fk_position_gates_pass",
            "all_one_step_model_gates_pass",
            "all_task_gates_pass",
            "extension_and_ref6_manifest_pass",
            "provenance_pass",
        )
    )
    return gates


def certify_dense_model_replay(
    *,
    common_x0_float32,
    coarse_controls_float32,
    dense_controls_float32,
    dense_time_float64,
    cuda_states_float32,
    pin_states_float64,
    cuda_tool_xyz_float32,
    pin_tool_xyz_float64,
    reference_float32,
    lower_float64,
    upper_float64,
    velocity_limit_float64,
    effort_limit_float64,
) -> dict:
    """Reusable fail-closed dense CUDA/Pin witness certificate.

    The state disagreement is retained for diagnosis but deliberately is not
    an acceptance gate. The frozen model gate is dense tool disagreement.
    """

    try:
        x0 = _exact_array(common_x0_float32, (14,), np.float32)
        coarse_controls = np.asarray(coarse_controls_float32)
        controls = np.asarray(dense_controls_float32)
        time = np.asarray(dense_time_float64)
        cuda_states = np.asarray(cuda_states_float32)
        pin_states = np.asarray(pin_states_float64)
        cuda_tool = np.asarray(cuda_tool_xyz_float32)
        pin_tool = np.asarray(pin_tool_xyz_float64)
        reference = _exact_array(reference_float32, (10,), np.float32)
        lower = _exact_array(lower_float64, (7,), np.float64)
        upper = _exact_array(upper_float64, (7,), np.float64)
        velocity_limit = _exact_array(velocity_limit_float64, (7,), np.float64)
        effort_limit = _exact_array(effort_limit_float64, (7,), np.float64)
        sample_count = int(time.size)
        shapes = (
            time.shape == (DENSE_SAMPLE_COUNT,)
            and sample_count == DENSE_SAMPLE_COUNT
            and time.dtype == np.float64
            and coarse_controls.shape == (toll.KNOTS - 1, 7)
            and coarse_controls.dtype == np.float32
            and controls.shape == (sample_count - 1, 7)
            and controls.dtype == np.float32
            and cuda_states.shape == (sample_count, 14)
            and cuda_states.dtype == np.float32
            and pin_states.shape == (sample_count, 14)
            and pin_states.dtype == np.float64
            and cuda_tool.shape == (sample_count, 3)
            and cuda_tool.dtype == np.float32
            and pin_tool.shape == (sample_count, 3)
            and pin_tool.dtype == np.float64
        )
        if not shapes:
            raise ValueError("dense replay arrays have invalid exact shapes or dtypes")
        arrays = (
            coarse_controls,
            controls,
            time,
            cuda_states,
            pin_states,
            cuda_tool,
            pin_tool,
        )
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("dense replay arrays must be finite")
        steps = np.diff(time)
        expected_time = np.arange(DENSE_SAMPLE_COUNT, dtype=np.float64) * (
            toll.DT / DENSE_SUBSTEPS
        )
        if not np.array_equal(time, expected_time):
            raise ValueError("dense replay time must equal the frozen 64-substep grid")
        if not np.array_equal(
            controls, np.repeat(coarse_controls, DENSE_SUBSTEPS, axis=0)
        ):
            raise ValueError("dense controls must be exact repeats of retained controls")
        toll.validate_reference_row(reference)
        if np.any(velocity_limit <= 0) or np.any(effort_limit <= 0):
            raise ValueError("modeled velocity and effort limits must be positive")
        arrays_valid = True
    except (TypeError, ValueError):
        arrays_valid = False
        sample_count = 0
        state_l2 = tool_errors = np.asarray([np.inf])
        result = {
            "arrays_valid": False,
            "all_dense_model_gates_pass": False,
            "sample_count": 0,
        }
        return result

    goal = reference[:3].astype(np.float64)
    ref = toll.TollReference.from_solver_bytes(reference)
    state_l2 = np.linalg.norm(pin_states - cuda_states.astype(np.float64), axis=1)
    tool_errors = np.linalg.norm(pin_tool - cuda_tool.astype(np.float64), axis=1)
    cuda_clearance = np.linalg.norm(
        cuda_tool[:, :2].astype(np.float64) - np.asarray(ref.cylinder_xy), axis=1
    ) - ref.physical_radius_m
    pin_clearance = np.linalg.norm(
        pin_tool[:, :2] - np.asarray(ref.cylinder_xy), axis=1
    ) - ref.physical_radius_m
    cuda_speed = np.linalg.norm(np.diff(cuda_tool.astype(np.float64), axis=0), axis=1) / np.diff(time)
    pin_speed = np.linalg.norm(np.diff(pin_tool, axis=0), axis=1) / np.diff(time)
    control_ratio = float(np.max(np.abs(controls.astype(np.float64)) / effort_limit))

    def model_gates(states, tool, clearance, speed, terminal_tolerance):
        q = states[:, :7].astype(np.float64)
        qd = states[:, 7:].astype(np.float64)
        return {
            "common_x0": bool(np.array_equal(states[0], x0.astype(states.dtype))),
            "joint_limits": bool(np.all(q >= lower) and np.all(q <= upper)),
            "velocity_limits": bool(np.max(np.abs(qd) / velocity_limit) <= 1.0),
            "terminal_error_m": float(np.linalg.norm(tool[-1].astype(np.float64) - goal)),
            "terminal": bool(
                np.linalg.norm(tool[-1].astype(np.float64) - goal) <= terminal_tolerance
            ),
            "final_tool_speed_mps": float(speed[-1]),
            "final_speed": bool(speed[-1] <= toll.FINAL_TOOL_SPEED_TOLERANCE_MPS),
            "min_physical_clearance_m": float(np.min(clearance)),
            "physical_clearance": bool(np.min(clearance) >= toll.CLEARANCE_MARGIN_M),
        }

    cuda = model_gates(
        cuda_states,
        cuda_tool,
        cuda_clearance,
        cuda_speed,
        toll.TERMINAL_CUDA_TOLERANCE_M,
    )
    pin = model_gates(
        pin_states,
        pin_tool,
        pin_clearance,
        pin_speed,
        toll.TERMINAL_PINOCCHIO_TOLERANCE_M,
    )
    model_gate_names = (
        "common_x0",
        "joint_limits",
        "velocity_limits",
        "terminal",
        "final_speed",
        "physical_clearance",
    )
    cuda["all_model_gates_pass"] = all(cuda[name] for name in model_gate_names)
    pin["all_model_gates_pass"] = all(pin[name] for name in model_gate_names)
    result = {
        "arrays_valid": arrays_valid,
        "sample_count": sample_count,
        "same_controls_both_models": True,
        "dense_substeps_per_interval": DENSE_SUBSTEPS,
        "collision_scope": toll.COLLISION_SCOPE,
        "control_ratio": control_ratio,
        "control_limits": control_ratio <= 1.0,
        "max_dense_tool_disagreement_m": float(np.max(tool_errors)),
        "dense_tool_agreement": bool(np.max(tool_errors) <= DENSE_TOOL_TOLERANCE_M),
        "max_dense_state_l2_report_only": float(np.max(state_l2)),
        "state_l2_is_acceptance_gate": False,
        "cuda": cuda,
        "pin": pin,
        "retained_array_hashes": {
            "common_x0_float32": _array_hash(x0),
            "coarse_controls_float32": _array_hash(coarse_controls),
            "dense_controls_float32": _array_hash(controls),
            "dense_time_float64": _array_hash(time),
            "cuda_states_float32": _array_hash(cuda_states),
            "pin_states_float64": _array_hash(pin_states),
            "cuda_tool_xyz_float32": _array_hash(cuda_tool),
            "pin_tool_xyz_float64": _array_hash(pin_tool),
        },
    }
    result["all_dense_model_gates_pass"] = bool(
        arrays_valid
        and result["control_limits"]
        and result["dense_tool_agreement"]
        and cuda["all_model_gates_pass"]
        and pin["all_model_gates_pass"]
    )
    return result


def execute_v0(*, model_factory: Callable, extension_factory: Callable):
    """Fail before model/task/extension construction in this checkpoint."""

    if V0_EXECUTION_AUTHORIZATION is None:
        raise RuntimeError("Stage V0 execution is blocked pending verifier authorization")
    # Kept unreachable deliberately so a future authorization patch must be
    # reviewed together with actual retention and provenance handling.
    model_factory()
    extension_factory()
    raise RuntimeError("Stage V0 execution runner is not implemented")
