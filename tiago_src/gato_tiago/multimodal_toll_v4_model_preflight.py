"""Static contract for the V4 artifact-only Tiago model preflight."""

from __future__ import annotations

from pathlib import Path

import numpy as np


MODEL_PREFLIGHT_PROTOCOL_VERSION = "tiago_tool_center_toll_v4_model_preflight_1"
MODEL_PREFLIGHT_EXECUTION_AUTHORIZATION = None
AUTHORIZED_OUTPUT_PATH = Path(
    "/tmp/tiago-tool-center-toll-v4-model-preflight-authorized-once/preflight.json"
)
V4_ARTIFACT_ROOT = Path(
    "/tmp/tiago-tool-center-toll-v4-task-construction-authorized-once"
)
V4_ARTIFACT_PATHS_AND_HASHES = {
    "final_json": (
        V4_ARTIFACT_ROOT / "v4.json",
        "6695c10a048ed07284422871201c458ed8f3a37fbd965591a3897d90ef682aac",
    ),
    "final_npz": (
        V4_ARTIFACT_ROOT / "v4.npz",
        "4d343d9dc75bf51987c9a18f61239207d1959f59caab64e492ab711ff9bfc7e8",
    ),
    "final_manifest": (
        V4_ARTIFACT_ROOT / "v4.manifest.json",
        "72cef090d49c4f76efe3398b2dfcdd5b72c91cecfe28efc910f4d67a18a01007",
    ),
    "generation13_pointer": (
        V4_ARTIFACT_ROOT / "v4.partial.latest.json",
        "3d37c6851f0e92a8b541585aa0f6e61b4f9b4f402c7bd9b038cac2dbdde6d460",
    ),
}
EXPECTED_V4_ARRAY_COUNT = 603
EXPECTED_TASK_COUNT = 12
EXPECTED_TASK_IDENTITIES = tuple(
    [("development", seed) for seed in range(12600, 12604)]
    + [("heldout", seed) for seed in range(12700, 12708)]
)
EXPECTED_WORKER_COUNT = 3
EXPECTED_NONWORKER_ARRAY_NAMES = frozenset(
    {
        "public_solver_x0_float32",
        "public_reference_float32",
        "public_default_side_int8",
        "quarantined_q8_float64",
        "quarantined_q8_float32",
        "quarantined_q8_oracle_only_bool",
        "quarantined_q8_initializer_eligible_bool",
        "quarantined_q8_solver_seed_eligible_bool",
        "v4_artifact_authentication_gate_bool",
        "model_lower_float64",
        "model_upper_float64",
        "model_velocity_float64",
        "model_effort_float64",
        "pin_q0_tool_float64",
        "pin_q8_tool_float64",
        "one_step_x_float32",
        "one_step_u_float32",
        "pin_one_step_float64",
        "broad_q_codes_int8",
        "broad_qd_codes_int8",
        "broad_u_codes_int8",
        "broad_x_float64",
        "broad_u_float64",
        "broad_x_float32",
        "broad_u_float32",
        "pin_broad_one_step_float64",
        "pin_broad_tool_float64",
        "dense_x0_float32",
        "dense_u_float32",
        "pin_dense_states_float64",
        "pin_dense_tool_positions_float64",
        "public_handoff_field_names_unicode",
        "forbidden_call_counts_int64",
    }
)
ONE_STEP_DT = 0.0125
DENSE_SUBSTEPS = 64
FK_PIN_CUDA_TOLERANCE_M = 1e-4
Q8_PUBLIC_GOAL_TOLERANCE_M = 5e-4
DENSE_TOOL_POSITION_TOLERANCE_M = 1e-3
ONE_STEP_STATE_L2_TOLERANCE = 1e-3
ONE_STEP_STATE_MAXABS_TOLERANCE = 1e-3
DENSE_STATE_L2_IS_REPORT_ONLY = True
REFERENCE_SMOKE_MAX_SQP_ITERS = 0
EXPECTED_DIAGNOSTIC_SOLVE_CALLS_PER_MODULE = 2
EXPECTED_SQP_OPTIMIZATION_CALLS = 0
COLLISION_SCOPE = "tool_center_only"
BROAD_ROW_COUNT = 32
BROAD_Q_RESERVE_RAD = 0.08
BROAD_Q_SPAN_FRACTION = 0.875
BROAD_VELOCITY_FRACTION = 0.20
BROAD_CONTROL_FRACTION = 0.20
BROAD_CODE_MODULUS = 67
BROAD_CODE_HALF_RANGE = 33
FROZEN_BUILD_COMMIT = "9bb1ceaf64787597f1ea7df5566f84d0635c4b7f"
FROZEN_CUDA_ARCH = "61-real"
FROZEN_BUILD_SOURCE_HASHES = {
    "python/bindings.cu": "13d909ff8e6f498e435d0c5314a435808fa886fd9990a97bf3a87ea6903a24d0",
    "gato/bsqp/kernels/tool_position.cuh": "99fe1c402663bbc4e2a73f9ed5bae0550420b2c4d3ef29980bce9f047607ae25",
    "gato/dynamics/integrator.cuh": "55da35cd8c1cddb6a13483739a99614f80f61a4b5bfb0b629fc02e03c60d252c",
    "gato/bsqp/bsqp.cuh": "ff6374b974420219dd5a46d4d2dd73f4fa4ea14b1b95f3ed5a51668d178fd628",
    "gato/dynamics/tiago_right/tiago_right_plant.cuh": "9941ebbf5260fb3b9b25f0f2189df8f49d275e26916b15837fc3a5085b846af2",
    "gato/dynamics/tiago_right/tiago_right_grid.cuh": "22eb6dcb37cca46e84a9c839f459479d63668d401eac7cbafe004e4c8692fcf4",
    "CMakeLists.txt": "29001140df02e2a5b7311e4f748b8e8cdfb546806da669f1447bc514795c3cdb",
    "tools/build.sh": "5fa16321e426c5903e0b26626877bed6ecb806b7d31ebc25e8ddbb2de107fcc1",
}

FROZEN_EXTENSIONS = (
    {
        "module_name": "bsqp.bsqpN64_tiago_right_multimodal_toll",
        "extension_path": "python/bsqp/bsqpN64_tiago_right_multimodal_toll.cpython-310-x86_64-linux-gnu.so",
        "extension_sha256": "3d3c1e0808a57b6189a0267e701a64d0b11ed13cda37b65873781954ffaa8551",
        "reference_size": 10,
        "variant": "toll",
    },
    {
        "module_name": "bsqp.bsqpN64_tiago_right",
        "extension_path": "python/bsqp/bsqpN64_tiago_right.cpython-310-x86_64-linux-gnu.so",
        "extension_sha256": "f0944080523a212edce0af72d323f5eeffd587adde2f0bea6063a31fabff2f41",
        "reference_size": 6,
        "variant": "plain_ref6",
    },
    {
        "module_name": "bsqp.bsqpN64_tiago_right_multimodal",
        "extension_path": "python/bsqp/bsqpN64_tiago_right_multimodal.cpython-310-x86_64-linux-gnu.so",
        "extension_sha256": "75650c37b3cd0e729922cdd82e65dd81214bb29f9591d0a7466aac197908b1fe",
        "reference_size": 6,
        "variant": "pillar_ref6",
    },
)
REQUIRED_SOURCE_PATHS = {
    "preflight_schema": "tiago_src/gato_tiago/multimodal_toll_v4_model_preflight.py",
    "preflight_runner": "tiago_src/gato_tiago/multimodal_toll_v4_model_preflight_runner.py",
    "preflight_worker": "tiago_src/gato_tiago/multimodal_toll_v4_model_preflight_worker.py",
    "preflight_tests": "tests/python/test_tiago_multimodal_toll_v4_model_preflight.py",
    "v4_contract": "tiago_src/gato_tiago/multimodal_toll_v4.py",
    "v4_oracle_schema": "tiago_src/gato_tiago/multimodal_toll_v4_oracle_schema.py",
    "v4_task_runner": "tiago_src/gato_tiago/multimodal_toll_v4_runner.py",
    "v4_tests": "tests/python/test_tiago_multimodal_toll_v4.py",
    "public_toll_schema": "tiago_src/gato_tiago/multimodal_toll.py",
    "tool_fk_source": "tiago_src/gato_tiago/multimodal_pillar.py",
    "integrator_source": "gato/dynamics/integrator.cuh",
    "solver_source": "gato/bsqp/bsqp.cuh",
    "tiago_plant_source": "gato/dynamics/tiago_right/tiago_right_plant.cuh",
    "tiago_grid": "gato/dynamics/tiago_right/tiago_right_grid.cuh",
    "cmake_wiring": "CMakeLists.txt",
    "model": "gato/dynamics/tiago_right/tiago_right_arm.urdf",
    "build_script": "tools/build.sh",
    "python_bindings": "python/bindings.cu",
    "tool_position_kernel": "gato/bsqp/kernels/tool_position.cuh",
}


def array_hash(value: np.ndarray) -> str:
    import hashlib

    array = np.ascontiguousarray(value)
    return hashlib.sha256(
        f"{array.dtype.str}|{array.shape}|".encode() + array.tobytes()
    ).hexdigest()


def is_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def generate_broad_algebraic_rows(lower, upper, velocity, effort):
    """Return the frozen 32-row task-neutral, deterministic model grid."""

    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    effort = np.asarray(effort, dtype=np.float64)
    if any(value.shape != (7,) for value in (lower, upper, velocity, effort)):
        raise ValueError("broad algebraic model grid expects four seven-wide limits")
    if not all(np.all(np.isfinite(value)) for value in (lower, upper, velocity, effort)):
        raise ValueError("broad algebraic model grid limits must be finite")
    lo = lower + BROAD_Q_RESERVE_RAD
    hi = upper - BROAD_Q_RESERVE_RAD
    if np.any(lo >= hi) or np.any(velocity <= 0) or np.any(effort <= 0):
        raise ValueError("broad algebraic model grid limits are invalid")
    rows = np.arange(BROAD_ROW_COUNT, dtype=np.int64)[:, None]
    joints = np.arange(7, dtype=np.int64)[None, :]

    def codes(row_factor, joint_factor, offset):
        raw = (row_factor * rows + joint_factor * joints + offset) % BROAD_CODE_MODULUS
        return raw.astype(np.int8) - BROAD_CODE_HALF_RANGE

    q_codes = codes(5, 3, 1)
    qd_codes = codes(7, 5, 4)
    u_codes = codes(11, 7, 2)
    normalized_q = q_codes.astype(np.float64) / BROAD_CODE_HALF_RANGE
    normalized_qd = qd_codes.astype(np.float64) / BROAD_CODE_HALF_RANGE
    normalized_u = u_codes.astype(np.float64) / BROAD_CODE_HALF_RANGE
    midpoint = 0.5 * (lo + hi)
    halfspan = 0.5 * (hi - lo)
    q = midpoint + BROAD_Q_SPAN_FRACTION * halfspan * normalized_q
    qd = BROAD_VELOCITY_FRACTION * velocity * normalized_qd
    u = BROAD_CONTROL_FRACTION * effort * normalized_u
    x = np.concatenate([q, qd], axis=1)
    return {
        "broad_q_codes_int8": q_codes,
        "broad_qd_codes_int8": qd_codes,
        "broad_u_codes_int8": u_codes,
        "broad_x_float64": x,
        "broad_u_float64": u,
        "broad_x_float32": x.astype(np.float32),
        "broad_u_float32": u.astype(np.float32),
    }


def frozen_model_preflight_metadata() -> dict:
    return {
        "protocol_version": MODEL_PREFLIGHT_PROTOCOL_VERSION,
        "execution_authorized": MODEL_PREFLIGHT_EXECUTION_AUTHORIZATION is not None,
        "authorized_output_path": str(AUTHORIZED_OUTPUT_PATH),
        "artifact_paths_and_hashes": {
            name: {"path": str(path), "sha256": digest}
            for name, (path, digest) in V4_ARTIFACT_PATHS_AND_HASHES.items()
        },
        "expected_v4_array_count": EXPECTED_V4_ARRAY_COUNT,
        "expected_task_identities": [list(row) for row in EXPECTED_TASK_IDENTITIES],
        "task_rng_calls": 0,
        "task_construction_calls": 0,
        "predecessor_artifact_loads": 0,
        "q8_scope": "quarantined_tool_position_model_preflight_only",
        "q8_public_handoff": False,
        "q8_initializer_eligible": False,
        "q8_solver_seed_eligible": False,
        "public_handoff_fields": ["x0", "reference", "default_side"],
        "q0_q8_pin_cuda_tool_tolerance_m": FK_PIN_CUDA_TOLERANCE_M,
        "q8_cuda_public_goal_tolerance_m": Q8_PUBLIC_GOAL_TOLERANCE_M,
        "one_step_dt": ONE_STEP_DT,
        "one_step_controls": ["zero", "stationary_gravity_rnea"],
        "pin_dynamics_algorithm": "pinocchio.aba_zero_external_wrench",
        "pin_one_step_integrator": (
            "q_next=q+dt*qd+0.5*dt^2*qdd; qd_next=qd+dt*qdd"
        ),
        "one_step_state_l2_tolerance": ONE_STEP_STATE_L2_TOLERANCE,
        "one_step_state_maxabs_tolerance": ONE_STEP_STATE_MAXABS_TOLERANCE,
        "dense_substeps": DENSE_SUBSTEPS,
        "dense_tool_tolerance_m": DENSE_TOOL_POSITION_TOLERANCE_M,
        "dense_state_l2_report_only": DENSE_STATE_L2_IS_REPORT_ONLY,
        "reference_smoke_max_sqp_iters": REFERENCE_SMOKE_MAX_SQP_ITERS,
        "diagnostic_solve_calls_per_module": EXPECTED_DIAGNOSTIC_SOLVE_CALLS_PER_MODULE,
        "sqp_optimization_calls": EXPECTED_SQP_OPTIMIZATION_CALLS,
        "optimization_evidence": False,
        "timing_evidence": False,
        "extensions": [dict(row) for row in FROZEN_EXTENSIONS],
        "broad_algebraic_grid": {
            "task_neutral": True,
            "row_count": BROAD_ROW_COUNT,
            "q_reserve_rad": BROAD_Q_RESERVE_RAD,
            "q_span_fraction": BROAD_Q_SPAN_FRACTION,
            "velocity_fraction": BROAD_VELOCITY_FRACTION,
            "control_fraction": BROAD_CONTROL_FRACTION,
            "code_modulus": BROAD_CODE_MODULUS,
            "code_half_range": BROAD_CODE_HALF_RANGE,
            "q_code": "((5*i+3*j+1)%67)-33",
            "qd_code": "((7*i+5*j+4)%67)-33",
            "u_code": "((11*i+7*j+2)%67)-33",
            "rng_calls": 0,
        },
        "frozen_build_commit": FROZEN_BUILD_COMMIT,
        "frozen_cuda_arch": FROZEN_CUDA_ARCH,
        "frozen_build_source_hashes": dict(FROZEN_BUILD_SOURCE_HASHES),
        "required_source_paths": dict(REQUIRED_SOURCE_PATHS),
    }
