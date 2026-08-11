"""Static-only Stage V1 draw schema for Tiago tool-center preflight.

V1 preserves the rejected V0 protocol byte-for-byte except for the second RNG
draw: diagnostic dynamics configurations are sampled directly inside the
modeled joint limits instead of jittering the comfortable configuration.
Nothing in this module constructs a task, loads a model, imports CUDA, or
authorizes execution.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping, Sequence

import numpy as np

from gato_tiago.multimodal_toll_v0 import (
    DYNAMICS_CONTROL_FRACTION,
    DYNAMICS_QD_BOUND_RAD_S,
    FK_JOINT_MARGIN_RAD,
    ONE_STEP_DT,
    V0_RANDOM_SEED,
    V0_SAMPLE_COUNT,
    EXPECTED_TASK_IDENTITIES,
    EXPECTED_EXTENSION_MANIFEST,
    FK_POSITION_TOLERANCE_M,
    ONE_STEP_STATE_L2_TOLERANCE,
    ONE_STEP_STATE_MAXABS_TOLERANCE,
    REQUIRED_SOURCE_PATHS as V0_REQUIRED_SOURCE_PATHS,
    _exact_array,
    _extension_manifest_gate,
    _is_git_commit,
    _is_sha256,
    _task_gate,
    constant_acceleration_step,
)


V1_PROTOCOL_VERSION = "tiago_tool_center_toll_v1_frame_model_preflight_1"
V1_EXECUTION_AUTHORIZATION = None
V1_RANDOM_SEED = V0_RANDOM_SEED
V1_SAMPLE_COUNT = V0_SAMPLE_COUNT
V1_OUTPUT_PATH = "/tmp/tiago-tool-center-toll-v1-authorized-once/v1.json"

V0_STATUS = "rejected_closed_pre_task_margin_gate"
V0_ARTIFACTS_EXCLUDED = True
V0_REJECTED_ARTIFACT_HASHES = {
    "generation0_json": "28f5253235a02b8d451b7500a8000aed57fa24a31911cf5ccbc04464cfa2bbdf",
    "generation0_npz": "8739c76e681f900923b900c9df0ef75cf421d39cabb54650c4b9ad19b6a76d85",
    "generation1_json": "568f8e2fbcdbe13a3198e0bab56f5bc150c8bd84367cc170a16f34434abb27a5",
    "generation1_npz": "70e0c3ff79e53c6435d8057a47f582939ced4799c6a573d4164913d4158340e8",
    "latest_pointer": "87eb0fc4d238afce55b7439c236506c82b26c833d9d04850729c88dd05beb522",
}
REQUIRED_SOURCE_PATHS = {
    **{
        name: path
        for name, path in V0_REQUIRED_SOURCE_PATHS.items()
        if name not in {"v0_schema", "v0_runner", "v0_worker", "v0_tests", "v0_runner_tests"}
    },
    "v1_schema": "tiago_src/gato_tiago/multimodal_toll_v1.py",
    "v1_runner": "tiago_src/gato_tiago/multimodal_toll_v1_runner.py",
    "v1_worker": "tiago_src/gato_tiago/multimodal_toll_v1_worker.py",
    "v1_tests": "tests/python/test_tiago_multimodal_toll_v1.py",
    "shared_preflight_gate_source": "tiago_src/gato_tiago/multimodal_toll_v0.py",
}
REQUIRED_SOURCE_HASH_LABELS = frozenset(REQUIRED_SOURCE_PATHS)


def _array_hash(value):
    array = np.ascontiguousarray(value)
    header = f"{array.dtype.str}|{array.shape}|".encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def _exact_vector(value, name):
    array = np.asarray(value)
    if array.shape != (7,) or array.dtype != np.float64 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be one finite float64 seven-vector")
    return array.copy()


@dataclass(frozen=True)
class V1Draws:
    lower_float64: np.ndarray
    upper_float64: np.ndarray
    effort_float64: np.ndarray
    comfortable_q_float64: np.ndarray
    fk_q_float64: np.ndarray
    dynamics_q_float64: np.ndarray
    dynamics_qd_float64: np.ndarray
    dynamics_control_fractions_float64: np.ndarray
    dynamics_u_float64: np.ndarray
    fk_q_float32: np.ndarray
    dynamics_x_float32: np.ndarray
    dynamics_u_float32: np.ndarray

    def array_hashes(self):
        return {
            name: _array_hash(value)
            for name, value in self.__dict__.items()
        }


def generate_v1_draws(lower, upper, effort, comfortable_q):
    """Generate the frozen V1 diagnostic stream without task construction."""

    lower = _exact_vector(lower, "lower")
    upper = _exact_vector(upper, "upper")
    effort = _exact_vector(effort, "effort")
    comfortable = _exact_vector(comfortable_q, "comfortable_q")
    if np.any(lower + 2 * FK_JOINT_MARGIN_RAD >= upper):
        raise ValueError("joint range cannot support the frozen 0.08-rad margin")
    if np.any(effort <= 0):
        raise ValueError("effort limits must be positive")

    rng = np.random.default_rng(V1_RANDOM_SEED)
    fk_q = rng.uniform(
        lower + FK_JOINT_MARGIN_RAD,
        upper - FK_JOINT_MARGIN_RAD,
        size=(V1_SAMPLE_COUNT, 7),
    )
    dynamics_q = rng.uniform(
        lower + FK_JOINT_MARGIN_RAD,
        upper - FK_JOINT_MARGIN_RAD,
        size=(V1_SAMPLE_COUNT, 7),
    )
    dynamics_qd = rng.uniform(
        -DYNAMICS_QD_BOUND_RAD_S,
        DYNAMICS_QD_BOUND_RAD_S,
        size=(V1_SAMPLE_COUNT, 7),
    )
    control_fractions = rng.uniform(
        -DYNAMICS_CONTROL_FRACTION,
        DYNAMICS_CONTROL_FRACTION,
        size=(V1_SAMPLE_COUNT, 7),
    )
    dynamics_u = control_fractions * effort[None, :]
    dynamics_x = np.concatenate([dynamics_q, dynamics_qd], axis=1)
    return V1Draws(
        lower_float64=lower,
        upper_float64=upper,
        effort_float64=effort,
        comfortable_q_float64=comfortable,
        fk_q_float64=fk_q,
        dynamics_q_float64=dynamics_q,
        dynamics_qd_float64=dynamics_qd,
        dynamics_control_fractions_float64=control_fractions,
        dynamics_u_float64=dynamics_u,
        fk_q_float32=fk_q.astype(np.float32),
        dynamics_x_float32=dynamics_x.astype(np.float32),
        dynamics_u_float32=dynamics_u.astype(np.float32),
    )


def _draws_are_authentic_v1(draws):
    if not isinstance(draws, V1Draws):
        return False, None
    try:
        regenerated = generate_v1_draws(
            draws.lower_float64,
            draws.upper_float64,
            draws.effort_float64,
            draws.comfortable_q_float64,
        )
    except (TypeError, ValueError):
        return False, None
    names = tuple(regenerated.__dict__)
    exact = tuple(draws.__dict__) == names and all(
        np.array_equal(getattr(draws, name), getattr(regenerated, name))
        for name in names
    )
    return exact and draws.array_hashes() == regenerated.array_hashes(), regenerated


def certify_v1(
    *,
    draws: V1Draws,
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
    """Pure V1 certificate over retained future execution outputs."""

    authentic_draws, _ = _draws_are_authentic_v1(draws)

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
                for index in range(V1_SAMPLE_COUNT)
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
        _task_gate(row, draws.lower_float64, draws.upper_float64, draws.comfortable_q_float64)
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
        and _is_git_commit(provenance.get("git_head_at_start"))
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
        provenance.get(
            "extension_source_commits", provenance.get("git_head_at_start")
        ),
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
    gates["all_v1_gates_pass"] = all(
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


def frozen_v1_metadata():
    return {
        "protocol_version": V1_PROTOCOL_VERSION,
        "execution_authorized": V1_EXECUTION_AUTHORIZATION is not None,
        "output_path": V1_OUTPUT_PATH,
        "rng": "numpy.random.default_rng",
        "random_seed": V1_RANDOM_SEED,
        "sample_count": V1_SAMPLE_COUNT,
        "draw_order": [
            "fk_q_direct_interior",
            "dynamics_q_direct_interior",
            "dynamics_qd_uniform",
            "dynamics_control_fraction_uniform",
        ],
        "v0_to_v1_exact_diff": {
            "only_changed_draw": "dynamics_q",
            "v0_transform": "comfortable_q + U[-0.05,0.05]",
            "v1_transform": "U[lower+0.08,upper-0.08]",
            "shape": [32, 7],
            "rng_position": 2,
            "unchanged": [
                "seed",
                "sample_count",
                "draw_order_and_sizes",
                "fk_q",
                "dynamics_qd",
                "dynamics_control_fractions",
                "dynamics_u",
                "dt",
                "all_model_and_certificate_thresholds",
                "task_method_and_benchmark_protocol",
            ],
        },
        "dt": ONE_STEP_DT,
        "joint_margin_rad": FK_JOINT_MARGIN_RAD,
        "dynamics_qd_rad_s": [-DYNAMICS_QD_BOUND_RAD_S, DYNAMICS_QD_BOUND_RAD_S],
        "dynamics_control_effort_fraction": [
            -DYNAMICS_CONTROL_FRACTION,
            DYNAMICS_CONTROL_FRACTION,
        ],
        "v0_status": V0_STATUS,
        "v0_artifacts_consumed": False,
        "v0_rejected_artifact_hashes_report_only": dict(V0_REJECTED_ARTIFACT_HASHES),
        "task_construction_authorized": False,
        "model_or_cuda_execution_authorized": False,
    }
