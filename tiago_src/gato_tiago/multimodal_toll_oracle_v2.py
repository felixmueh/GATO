"""Frozen Oracle V2 prerequisite-authentication contract.

This module intentionally contains no task construction, model, CUDA, or
optimizer entry point.  The prerequisite runner is a separately authorized
CPU-only stage.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


PROTOCOL = "tiago_tool_center_toll_oracle_v2_prerequisite_1"
AUTHORIZED_CWD = Path("/workspace/GATO")
AUTHORIZED_OUTPUT = Path(
    "/tmp/tiago-tool-center-toll-oracle-v2-prerequisite-authorized-once/"
    "prerequisite.json"
)
AUTHORIZED_ORIG_ARGV = (
    "python",
    "-B",
    "-m",
    "gato_tiago.multimodal_toll_oracle_v2_prerequisite_runner",
    "--execute",
    "--output",
    str(AUTHORIZED_OUTPUT),
)

TASK_ROOT = Path(
    "/tmp/tiago-tool-center-toll-v4-task-construction-authorized-once"
)
MODEL_ROOT = Path(
    "/tmp/tiago-tool-center-toll-v4-model-preflight-v4-authorized-once"
)
TASK_PINS = {
    "final_json": (TASK_ROOT / "v4.json", "6695c10a048ed07284422871201c458ed8f3a37fbd965591a3897d90ef682aac"),
    "final_npz": (TASK_ROOT / "v4.npz", "4d343d9dc75bf51987c9a18f61239207d1959f59caab64e492ab711ff9bfc7e8"),
    "final_manifest": (TASK_ROOT / "v4.manifest.json", "72cef090d49c4f76efe3398b2dfcdd5b72c91cecfe28efc910f4d67a18a01007"),
    "latest_pointer": (TASK_ROOT / "v4.partial.latest.json", "3d37c6851f0e92a8b541585aa0f6e61b4f9b4f402c7bd9b038cac2dbdde6d460"),
}
MODEL_PINS = {
    "final_json": (MODEL_ROOT / "model.json", "3493feae03b7b6368a0dc506aa1f0b5067c63c16f90e08b3455c2314f6e7cc16"),
    "final_npz": (MODEL_ROOT / "model.npz", "fe29896b3c5cacbfb15be2a66ddc222a88f8e2e5c2646e183cdbaac34bd1fb6e"),
    "final_manifest": (MODEL_ROOT / "model.manifest.json", "438df87e388352ebe5762ca6e9cc3163019d1d9a9fdb616d5d4b027e8bc1a92c"),
    "latest_pointer": (MODEL_ROOT / "model.partial.latest.json", "d0eddf81f3a7072fda795d92ad7f0e75d869ff62cb282c3eb1b453b1af98539e"),
    "worker_input": (MODEL_ROOT / "model.worker-input.npz", "2e0953180dc9661a4740e57a6f94d9a8c4e438e7f4e3b34273bf5c07a154f426"),
}
TASK_ARRAY_COUNT = 603
MODEL_ARRAY_COUNT = 150
EXPECTED_TASK_IDENTITIES = tuple(
    [("development", seed) for seed in range(12600, 12604)]
    + [("heldout", seed) for seed in range(12700, 12708)]
)
PRODUCER_HASH_CONVENTION = (
    'sha256(f"{a.dtype.str}|{a.shape}|".encode()+a.tobytes())'
)

_V1_COUNT_KEYS = (
    "task_artifact_loads",
    "task_pure_recert_calls",
    "model_artifact_loads",
    "model_pure_recert_calls",
    "pinocchio_model_calls",
    "task_rng_calls",
    "task_construction_calls",
    "route_template_calls",
    "dls_calls",
    "optimizer_calls",
    "bootstrap_rng_calls",
    "worker_subprocess_calls",
    "cuda_calls",
    "sqp_calls",
    "initializer_calls",
)
V1_RETAINED_GEN0_COUNTS = {name: 0 for name in _V1_COUNT_KEYS}
V1_RECONSTRUCTED_EXECUTION_COUNTS = {
    **V1_RETAINED_GEN0_COUNTS,
    "task_artifact_loads": 1,
    "task_pure_recert_calls": 1,
}
V1_REJECTED_REPORT_ONLY = {
    "classification": "launch_invalid_predecessor_hash_format",
    "commit": "c51997874db51aff3bc1ddcf08f70f930d545edb",
    "command": (
        "PYTHONPATH=tiago_src:python python -B -m "
        "gato_tiago.multimodal_toll_oracle_v1_runner --execute --output "
        "/tmp/tiago-tool-center-toll-oracle-v1-authorized-once/oracle.json"
    ),
    "exit_code": 1,
    "wall_seconds": 0.965679851,
    "gen0_json_sha256": "9d545e2db2cae7aedbf3d202b21972985d78112969d9664da0c97ce781816b2b",
    "gen0_npz_sha256": "c415c0df2b17deb46d60f9daf4f0c65a87e56f47e81ca751a8d348566e7622a9",
    "latest_pointer_sha256": "5888efa3a2e7803499bf011dab9e0217632fe5b8414a6bfb89258b4222600636",
    "completed_pairs": 0,
    "pending_pairs": 120,
    "retained_gen0_counts": V1_RETAINED_GEN0_COUNTS,
    "independently_reconstructed_execution_counts": (
        V1_RECONSTRUCTED_EXECUTION_COUNTS
    ),
    "artifact_loads_in_v2": 0,
}

FORBIDDEN_CALL_COUNTS = (
    "pinocchio_model_calls",
    "task_rng_calls",
    "task_construction_calls",
    "route_template_calls",
    "dls_calls",
    "optimizer_calls",
    "bootstrap_rng_calls",
    "worker_subprocess_calls",
    "cuda_calls",
    "sqp_calls",
    "initializer_calls",
)


def producer_array_hash(value: np.ndarray) -> str:
    """Use the exact task-V4/model-V4 producer serialization convention."""

    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(
        f"{array.dtype.str}|{array.shape}|".encode() + array.tobytes()
    ).hexdigest()


def frozen_metadata() -> dict:
    return {
        "protocol": PROTOCOL,
        "authorized_output": str(AUTHORIZED_OUTPUT),
        "task_pins": {
            key: {"path": str(path), "sha256": digest}
            for key, (path, digest) in TASK_PINS.items()
        },
        "model_pins": {
            key: {"path": str(path), "sha256": digest}
            for key, (path, digest) in MODEL_PINS.items()
        },
        "task_array_count": TASK_ARRAY_COUNT,
        "model_array_count": MODEL_ARRAY_COUNT,
        "producer_hash_convention": PRODUCER_HASH_CONVENTION,
        "v1_rejected_report_only": dict(V1_REJECTED_REPORT_ONLY),
        "rejected_v1_artifact_loads": 0,
        "prerequisite_auth_only": True,
        "oracle_evidence": False,
        "benchmark_evidence": False,
    }
