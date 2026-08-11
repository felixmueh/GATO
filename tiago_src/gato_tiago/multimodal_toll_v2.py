"""Static schema for the quarantined Tiago toll task-construction V2.

V2 changes only how the hidden reachable Cartesian goal is constructed.  It
does not change the public task geometry, objective, initializer, solver, or
certificates.  Runtime construction remains blocked in this checkpoint.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from gato_tiago import multimodal_toll as toll


V2_PROTOCOL_VERSION = "tiago_tool_center_toll_task_construction_v2_1"
V2_EXECUTION_AUTHORIZATION = None
DEVELOPMENT_TASK_SEEDS = tuple(range(12200, 12204))
HELDOUT_TASK_SEEDS = tuple(range(12300, 12308))
EXPECTED_TASK_IDENTITIES = tuple(
    [("development", seed) for seed in DEVELOPMENT_TASK_SEEDS]
    + [("heldout", seed) for seed in HELDOUT_TASK_SEEDS]
)
V2_OUTPUT_PATH = Path(
    "/tmp/tiago-tool-center-toll-v2-authorized-once/v2.json"
)

DLS_ITERATIONS = 8
DLS_DAMPING = toll.DAMPING
DLS_UNIT_STEP = 1.0
DLS_RESIDUAL_IS_REPORT_ONLY = True
REJECTED_PREDECESSOR_ARTIFACT_HASHES = {
    "v0_latest_pointer": "87eb0fc4d238afce55b7439c236506c82b26c833d9d04850729c88dd05beb522",
    "v1_latest_pointer": "728f802236336f9f5fea937dd729460f0af8663ed1ffa983b6c8210f589d28e3",
    "v1_generation4_json": "5fc52d8138dccfa07fa25fcd2a174348d5f3351317fe608080b3e7c3999409dc",
    "v1_generation4_npz": "e1915a02fabd35fb63fc6401aede3f0a09ed55d539ec6c048414020184216717",
}
REQUIRED_SOURCE_PATHS = {
    "v2_schema": "tiago_src/gato_tiago/multimodal_toll_v2.py",
    "v2_oracle_schema": "tiago_src/gato_tiago/multimodal_toll_v2_oracle_schema.py",
    "v2_runner": "tiago_src/gato_tiago/multimodal_toll_v2_runner.py",
    "v2_tests": "tests/python/test_tiago_multimodal_toll_v2.py",
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


def frozen_v2_metadata() -> dict:
    """Describe the exact V2 construction-only delta without opening RNG."""

    return {
        "protocol_version": V2_PROTOCOL_VERSION,
        "execution_authorized": V2_EXECUTION_AUTHORIZATION is not None,
        "development_task_seeds": list(DEVELOPMENT_TASK_SEEDS),
        "heldout_task_seeds": list(HELDOUT_TASK_SEEDS),
        "expected_task_identities": [list(row) for row in EXPECTED_TASK_IDENTITIES],
        "output_path": str(V2_OUTPUT_PATH),
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
        },
        "v2_task_seed_override": {
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
        "v0_v1_artifacts_consumed": False,
        "rejected_predecessor_artifact_hashes_report_only": dict(
            REJECTED_PREDECESSOR_ARTIFACT_HASHES
        ),
        "required_source_paths": dict(REQUIRED_SOURCE_PATHS),
        "benchmark_initializer_eligible": False,
    }


def validate_v2_task_seed(task_seed: int) -> tuple[str, int]:
    identity = next(
        (row for row in EXPECTED_TASK_IDENTITIES if row[1] == int(task_seed)),
        None,
    )
    if identity is None:
        raise ValueError("task_seed is not in the frozen V2 development/held-out sets")
    return identity
