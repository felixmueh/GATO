"""Static schema for deterministic box-constrained Tiago task construction V4."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from gato_tiago import multimodal_toll as toll


V4_PROTOCOL_VERSION = "tiago_tool_center_toll_task_construction_v4_1"
V4_EXECUTION_AUTHORIZATION = None
DEVELOPMENT_TASK_SEEDS = tuple(range(12600, 12604))
HELDOUT_TASK_SEEDS = tuple(range(12700, 12708))
EXPECTED_TASK_IDENTITIES = tuple(
    [("development", seed) for seed in DEVELOPMENT_TASK_SEEDS]
    + [("heldout", seed) for seed in HELDOUT_TASK_SEEDS]
)
V4_OUTPUT_PATH = Path(
    "/tmp/tiago-tool-center-toll-v4-task-construction-authorized-once/v4.json"
)

DLS_ITERATIONS = 8
DLS_DAMPING = toll.DAMPING
DLS_UNIT_STEP = 1.0
DLS_RESIDUAL_IS_REPORT_ONLY = True
FACE_STATUS_ORDER = (-1, 0, 1)
FACE_STATUS_MEANING = {"-1": "lower", "0": "free", "1": "upper"}
FACE_COUNT = 3**7
FACE_FEASIBILITY_TOLERANCE = 1e-12
FREE_NORMAL_RESIDUAL_TOLERANCE = 1e-10
OBJECTIVE_TIE_ABSOLUTE_TOLERANCE = 1e-12
VERIFY_DQ_MAXABS_TOLERANCE = 1e-10
VERIFY_OBJECTIVE_TOLERANCE = 1e-12
VERIFY_GLOBAL_DOMINANCE_TOLERANCE = 1e-12
SELECTED_KKT_TOLERANCE = 1e-9
REJECTED_PREDECESSOR_ARTIFACT_HASHES = {
    "v0_latest_pointer": "87eb0fc4d238afce55b7439c236506c82b26c833d9d04850729c88dd05beb522",
    "v1_latest_pointer": "728f802236336f9f5fea937dd729460f0af8663ed1ffa983b6c8210f589d28e3",
    "v1_generation4_json": "5fc52d8138dccfa07fa25fcd2a174348d5f3351317fe608080b3e7c3999409dc",
    "v1_generation4_npz": "e1915a02fabd35fb63fc6401aede3f0a09ed55d539ec6c048414020184216717",
    "v2_latest_pointer": "cea28572a97f62f02fe9ac23e04d63dafed4b69a8300527b829e264f0039ca39",
    "v2_generation13_json": "c03c23dba9fc7103b8f86893f0b284cad05006ff97228a44b7201067022b8dc4",
    "v2_generation13_npz": "c6626dd260b58b26e5aa1bd048e00cc357d7021847eb31e282e90d0ca85fe05d",
    "v3_latest_pointer": "4fb4053fc9696e557e15162cf5a7d02c54b2b9d1d53d55bf9d07fbc78f855cd5",
    "v3_generation13_json": "7f3c154dae6eb00ad2b009b92ad3c17bf612469a62efc4768b075ac39b5363a0",
    "v3_generation13_npz": "a53ab862440e788ad48ceb4d7927b51f3a4fad26e1599880d6d350357d5363ae",
}
REQUIRED_SOURCE_PATHS = {
    "v4_schema": "tiago_src/gato_tiago/multimodal_toll_v4.py",
    "v4_oracle_schema": "tiago_src/gato_tiago/multimodal_toll_v4_oracle_schema.py",
    "v4_runner": "tiago_src/gato_tiago/multimodal_toll_v4_runner.py",
    "v4_tests": "tests/python/test_tiago_multimodal_toll_v4.py",
    "public_toll_schema": "tiago_src/gato_tiago/multimodal_toll.py",
    "tool_fk_source": "tiago_src/gato_tiago/multimodal_pillar.py",
    "start_configuration_source": "tiago_src/gato_tiago/config.py",
    "model": "gato/dynamics/tiago_right/tiago_right_arm.urdf",
}
NON_SEED_PUBLIC_PROTOCOL_FIELDS = (
    "knots",
    "dt",
    "horizon_s",
    "budget",
    "reference_size",
    "frames",
    "collision_scope",
    "model_path",
)


def unchanged_non_seed_protocol_snapshot() -> dict:
    base = toll.frozen_protocol_metadata()
    return {name: base[name] for name in NON_SEED_PUBLIC_PROTOCOL_FIELDS}


def _canonical_json_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def frozen_v4_metadata() -> dict:
    """Describe the exact V4 construction-only delta without opening RNG."""

    return {
        "protocol_version": V4_PROTOCOL_VERSION,
        "execution_authorized": V4_EXECUTION_AUTHORIZATION is not None,
        "development_task_seeds": list(DEVELOPMENT_TASK_SEEDS),
        "heldout_task_seeds": list(HELDOUT_TASK_SEEDS),
        "expected_task_identities": [list(row) for row in EXPECTED_TASK_IDENTITIES],
        "output_path": str(V4_OUTPUT_PATH),
        "construction": {
            "iterations": DLS_ITERATIONS,
            "damping": DLS_DAMPING,
            "step": DLS_UNIT_STEP,
            "residual": "full_3d_to_fixed_planar_target_report_only",
            "target": "p0 + 0.15*[cos(phi),sin(phi),0]",
            "jacobian": "LOCAL_WORLD_ALIGNED translational tool-frame Jacobian",
            "early_exit": False,
            "step_cap": False,
            "clip": False,
            "projection": False,
            "retry": False,
            "replacement": False,
            "line_search": False,
            "method": "exhaustive_strictly_convex_box_qp_active_set",
            "objective": "0.5*||J*d-r||^2 + 0.5*0.05^2*||d||^2",
            "increment_lower": "joint_lower+0.08-q_k",
            "increment_upper": "joint_upper-0.08-q_k",
            "status_order": list(FACE_STATUS_ORDER),
            "status_meaning": dict(FACE_STATUS_MEANING),
            "faces_per_step": FACE_COUNT,
            "exhaust_all_faces_without_pruning": True,
            "free_solver": "float64_spd_solve_no_inverse",
            "face_feasibility_tolerance": FACE_FEASIBILITY_TOLERANCE,
            "free_normal_residual_tolerance": FREE_NORMAL_RESIDUAL_TOLERANCE,
            "objective_tie_absolute_tolerance": OBJECTIVE_TIE_ABSOLUTE_TOLERANCE,
            "tie_break": "lexicographically_first_lower_free_upper",
            "independent_verifier_calls_constructor_enumerator": False,
            "verify_dq_maxabs_tolerance": VERIFY_DQ_MAXABS_TOLERANCE,
            "verify_objective_tolerance": VERIFY_OBJECTIVE_TOLERANCE,
            "verify_global_dominance_tolerance": VERIFY_GLOBAL_DOMINANCE_TOLERANCE,
            "selected_kkt_tolerance": SELECTED_KKT_TOLERANCE,
            "all_q0_through_q8_reserve_margin_is_acceptance_gate": True,
            "all_numerical_ik_iterates_are_oracle_only": True,
            "all_numerical_ik_iterates_are_public_task_data": False,
            "all_numerical_ik_iterates_are_initializer_inputs": False,
            "all_numerical_ik_iterates_are_sqp_inputs": False,
        },
        "v4_task_seed_override": {
            "development": list(DEVELOPMENT_TASK_SEEDS),
            "heldout": list(HELDOUT_TASK_SEEDS),
        },
        "unchanged_non_seed_protocol": unchanged_non_seed_protocol_snapshot(),
        "unchanged_non_seed_protocol_sha256": _canonical_json_hash(
            unchanged_non_seed_protocol_snapshot()
        ),
        "unchanged_non_seed_protocol_matches_base": bool(
            unchanged_non_seed_protocol_snapshot()
            == {
                name: toll.frozen_protocol_metadata()[name]
                for name in NON_SEED_PUBLIC_PROTOCOL_FIELDS
            }
        ),
        "excluded_predecessor_identity_fields": [
            "protocol",
            "task_instantiation_authorized",
            "optimizer_execution_authorized",
        ],
        "predecessor_artifacts_consumed": False,
        "rejected_predecessor_artifact_hashes_report_only": dict(
            REJECTED_PREDECESSOR_ARTIFACT_HASHES
        ),
        "required_source_paths": dict(REQUIRED_SOURCE_PATHS),
        "benchmark_initializer_eligible": False,
        "forbidden_public_or_initializer_fields": [
            "dls_history",
            "dls_intermediate_q",
            "ik_iterates",
            "intermediate_joint_margins",
            "box_dls_history",
            "box_dls_face_table",
            "box_dls_active_set",
        ],
    }


def validate_v4_task_seed(task_seed: int) -> tuple[str, int]:
    identity = next(
        (row for row in EXPECTED_TASK_IDENTITIES if row[1] == int(task_seed)),
        None,
    )
    if identity is None:
        raise ValueError("task_seed is not in the frozen V4 development/held-out sets")
    return identity
