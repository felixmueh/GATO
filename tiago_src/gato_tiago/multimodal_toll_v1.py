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

import numpy as np

from gato_tiago.multimodal_toll_v0 import (
    DYNAMICS_CONTROL_FRACTION,
    DYNAMICS_QD_BOUND_RAD_S,
    FK_JOINT_MARGIN_RAD,
    ONE_STEP_DT,
    V0_RANDOM_SEED,
    V0_SAMPLE_COUNT,
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
