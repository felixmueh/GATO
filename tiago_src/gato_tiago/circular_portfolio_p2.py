"""Pure reduced-coordinate schema for circular-portfolio P2.

P2 changes only the acquisition coordinates.  The original P1 objective,
inequalities, integration, inverse dynamics, references, and evidence gates
remain the definitions evaluated after expanding 70 independent coordinates
spanning the endpoint-projected family of 84 raw spline coefficients.
"""

from __future__ import annotations

import hashlib
from typing import Mapping

import numpy as np

from gato_tiago.circular_portfolio import DT, INTERVALS, KNOTS
from gato_tiago.circular_portfolio_constructor import (
    integrate_acceleration, recurrence_sensitivities,
)


PROTOCOL = "tiago_tool_center_circular_portfolio_p2_1"
PRECHECK_IDENTITY = ("development", 12600, "short", 0)
SPLINE_DEGREE = 3
SPLINE_COEFFICIENTS_PER_JOINT = 12
RAW_SPLINE_VARIABLES = SPLINE_COEFFICIENTS_PER_JOINT * 7
EFFECTIVE_REDUCED_RANK = (SPLINE_COEFFICIENTS_PER_JOINT - 2) * 7
OPTIMIZER_VARIABLES = EFFECTIVE_REDUCED_RANK
FULL_ACCELERATION_VARIABLES = INTERVALS * 7
PREFLIGHT_WALL_LIMIT_S = 30.0
RANK_TOLERANCE = 1e-10

P1_REJECTED_REPORT = {
    "protocol": "tiago_tool_center_circular_portfolio_p1_1",
    "source_head": "a32f1d4091e0db1876882583fa2e9f0502c91538",
    "exit_code": 1,
    "campaign_elapsed_s": 121.57387311197817,
    "identity": ["development", 12600, "short", 0],
    "stage": "profile_failed",
    "error_type": "TimeoutError",
    "error_message": "circular portfolio profile wall limit",
    "completed": 0,
    "pending": 192,
    "worker_calls": 0,
    "cuda_calls": 0,
    "replay_calls": 0,
    "sqp_calls": 0,
    "final_artifacts": 0,
    "rejected_p1_artifact_loads": 0,
    "hashes": {
        "rejection_json": "1a4d428b5c23da776734f6c3742de80a256b7c4b29179b844ac434af60f6e60b",
        "rejection_npz": "8739c76e681f900923b900c9df0ef75cf421d39cabb54650c4b9ad19b6a76d85",
        "latest": "2bf5556f2276f7b02e6bffe22a685fde948eb372aefbd13a115434e1697bd308",
        "gen0_json": "7ec0973a38784a4d9a66795514ff5b05b3286e17c66d46c796d6d4d43cc5b504",
        "gen0_npz": "a130edf0e36ef4fa4fd6ddfffa6d571b8b0d9648399aacf7820407b8ab3899c0",
        "gen1_json": "7729dfceeafc6c18a5577c738acb93a17303b3c404cb691572df6b427fa0fdbb",
        "gen1_npz": "6cd9b0bb6c10a113ea8d6178b5d39ca1ff02ca7478a1c2c8490d028a7f9ccb41",
        "gen2_json": "b95092133f76aaa31e3a450a6fe974d6ebb5866939d1cfc38fd94fd566f4f351",
        "gen2_npz": "d1d8a097dfca6d9b697f17c514e1e58d0af392a6fae904bc657aad365289ab95",
        "gen3_json": "21ea125b5cc82bb01aa8c2ae72b5f84e720eb967966c68fd5bfde57c7441320f",
        "gen3_npz": "fc72c0baa98123ea12c74f9b70aef845b55f09cec5f6d7b10b74ee60d08a67f8",
    },
}


def open_uniform_knots():
    interior = np.arange(1, 9, dtype=np.float64) / 9.0
    return np.r_[np.zeros(4), interior, np.ones(4)]


def cubic_bspline_basis():
    """Evaluate the fixed open-uniform cubic basis at 95 interval midpoints."""
    knots = open_uniform_knots()
    points = (np.arange(INTERVALS, dtype=np.float64) + 0.5) / INTERVALS
    count = SPLINE_COEFFICIENTS_PER_JOINT
    values = np.zeros((len(points), count + SPLINE_DEGREE), np.float64)
    for column in range(count + SPLINE_DEGREE):
        values[:, column] = (
            (points >= knots[column]) & (points < knots[column + 1])
        )
    for degree in range(1, SPLINE_DEGREE + 1):
        next_values = np.zeros((len(points), count + SPLINE_DEGREE - degree))
        for column in range(next_values.shape[1]):
            left_denominator = knots[column + degree] - knots[column]
            right_denominator = knots[column + degree + 1] - knots[column + 1]
            if left_denominator:
                next_values[:, column] += (
                    (points - knots[column]) / left_denominator * values[:, column]
                )
            if right_denominator:
                next_values[:, column] += (
                    (knots[column + degree + 1] - points) / right_denominator
                    * values[:, column + 1]
                )
        values = next_values
    return values[:, :count]


def endpoint_map():
    dq, dv = recurrence_sensitivities()
    return np.vstack((dq[-1], dv[-1]))


def reduced_affine_map(q0, q_goal):
    q0 = np.asarray(q0, np.float64); q_goal = np.asarray(q_goal, np.float64)
    if q0.shape != (7,) or q_goal.shape != (7,):
        raise ValueError("P2 endpoint vectors must be seven-dimensional")
    basis = cubic_bspline_basis()
    endpoint = endpoint_map()
    scalar_endpoint=endpoint[np.ix_((0,7),np.arange(0,FULL_ACCELERATION_VARIABLES,7))]
    scalar_gram=scalar_endpoint@scalar_endpoint.T
    scalar_right_inverse=scalar_endpoint.T@np.linalg.solve(scalar_gram,np.eye(2))
    temporal_projection=np.eye(INTERVALS)-scalar_right_inverse@scalar_endpoint
    temporal_reduced=temporal_projection@basis
    temporal_left,temporal_singular,temporal_right_transpose=np.linalg.svd(
        temporal_reduced,full_matrices=False
    )
    # Factor only the scalar temporal problem.  Expanding its simple-spectrum
    # basis by I7 avoids arbitrary choices in seven-fold repeated subspaces.
    temporal_independent=temporal_left[:,:10].copy()
    for column in range(10):
        pivot=int(np.argmax(np.abs(temporal_independent[:,column])))
        if temporal_independent[pivot,column]<0: temporal_independent[:,column]*=-1
    raw_map = np.kron(basis, np.eye(7, dtype=np.float64))
    right_inverse=np.kron(scalar_right_inverse,np.eye(7))
    projection=np.kron(temporal_projection,np.eye(7))
    target = np.r_[q_goal - q0, np.zeros(7)]
    particular = right_inverse @ target
    reduced = projection @ raw_map
    independent=np.kron(temporal_independent,np.eye(7))
    compression=independent.T@reduced
    return {
        "basis_float64": basis,
        "scalar_endpoint_map_float64":scalar_endpoint,
        "temporal_projection_float64":temporal_projection,
        "temporal_reduced_float64":temporal_reduced,
        "temporal_independent_float64":temporal_independent,
        "temporal_singular_values_float64":temporal_singular,
        "raw_map_float64": raw_map,
        "endpoint_map_float64": endpoint,
        "endpoint_right_inverse_float64": right_inverse,
        "projection_float64": projection,
        "particular_acceleration_float64": particular.reshape(INTERVALS, 7),
        "reduced_map_float64": reduced,
        "independent_map_float64":independent,
        "raw_to_independent_float64":compression,
        "reduced_singular_values_float64":np.repeat(temporal_singular,7),
    }


def expand_coefficients(coefficients, affine):
    coefficients = np.asarray(coefficients, np.float64)
    if coefficients.shape != (OPTIMIZER_VARIABLES,) or not np.isfinite(coefficients).all():
        raise ValueError("P2 optimizer coordinates must be 70 finite float64 values")
    return (
        np.asarray(affine["particular_acceleration_float64"]).ravel()
        + np.asarray(affine["independent_map_float64"]) @ coefficients
    ).reshape(INTERVALS, 7)


def least_squares_initial_coefficients(proxy_acceleration, affine):
    target = np.asarray(proxy_acceleration, np.float64).reshape(FULL_ACCELERATION_VARIABLES)
    defect = target - np.asarray(affine["particular_acceleration_float64"]).ravel()
    coefficients, residuals, rank, singular = np.linalg.lstsq(
        np.asarray(affine["independent_map_float64"]), defect, rcond=None
    )
    return {
        "initial_coefficients_float64": coefficients,
        "least_squares_residual_float64": np.asarray(
            np.linalg.norm(affine["independent_map_float64"] @ coefficients - defect),
            np.float64,
        ),
        "least_squares_rank_int64": np.asarray(rank, np.int64),
        "least_squares_singular_float64": singular,
        "least_squares_reported_residual_float64": np.asarray(residuals, np.float64),
    }


def chain_gradient(full_gradient, affine):
    gradient = np.asarray(full_gradient, np.float64)
    return np.asarray(affine["independent_map_float64"]).T @ gradient


def chain_jacobian(full_jacobian, affine):
    return np.asarray(full_jacobian, np.float64) @ np.asarray(
        affine["independent_map_float64"]
    )


def certify_affine_map(affine: Mapping, q0, q_goal):
    fresh = reduced_affine_map(q0, q_goal)
    exact = set(affine) == set(fresh) and all(
        np.array_equal(np.asarray(affine[name]), value) for name, value in fresh.items()
    )
    endpoint = fresh["endpoint_map_float64"]
    reduced = fresh["reduced_map_float64"]
    particular = fresh["particular_acceleration_float64"].ravel()
    target = np.r_[np.asarray(q_goal) - np.asarray(q0), np.zeros(7)]
    rank = lambda value: int(np.count_nonzero(
        np.linalg.svd(value, compute_uv=False) > RANK_TOLERANCE
    ))
    gates = {
        "exact_regeneration": exact,
        "basis_rank": rank(fresh["basis_float64"]) == 12,
        "scalar_endpoint_rank":rank(fresh["scalar_endpoint_map_float64"])==2,
        "temporal_rank":rank(fresh["temporal_reduced_float64"])==10,
        "temporal_orthonormal":np.max(np.abs(fresh["temporal_independent_float64"].T@
            fresh["temporal_independent_float64"]-np.eye(10)))<=1e-12,
        "temporal_null":np.max(np.abs(fresh["scalar_endpoint_map_float64"]@
            fresh["temporal_independent_float64"]),initial=0.0)<=1e-12,
        "temporal_span":np.max(np.abs(fresh["temporal_independent_float64"]@
            fresh["temporal_independent_float64"].T-fresh["temporal_reduced_float64"]@
            np.linalg.pinv(fresh["temporal_reduced_float64"],rcond=RANK_TOLERANCE)),initial=0.0)<=1e-11,
        "temporal_simple_spectrum":bool(np.min(np.abs(np.diff(
            fresh["temporal_singular_values_float64"][:10])))>RANK_TOLERANCE),
        "time_major_joint_minor":np.array_equal(fresh["independent_map_float64"],
            np.kron(fresh["temporal_independent_float64"],np.eye(7))),
        "endpoint_rank": rank(endpoint) == 14,
        "projection_rank": rank(fresh["projection_float64"]) == 651,
        # Endpoint projection removes exactly position/velocity modes per joint;
        # The raw 84-coefficient family has a predeclared 14-dimensional
        # endpoint nullity; trust-constr sees only the 70 independent modes.
        "reduced_rank": rank(reduced) == EFFECTIVE_REDUCED_RANK,
        "coefficient_nullity": RAW_SPLINE_VARIABLES - rank(reduced) == 14,
        "independent_rank":rank(fresh["independent_map_float64"])==OPTIMIZER_VARIABLES,
        "singular_gap":fresh["temporal_singular_values_float64"][9]>RANK_TOLERANCE
        and fresh["temporal_singular_values_float64"][10]<RANK_TOLERANCE,
        "orthonormal":np.max(np.abs(fresh["independent_map_float64"].T@
            fresh["independent_map_float64"]-np.eye(OPTIMIZER_VARIABLES)))<=1e-12,
        "independent_null":np.max(np.abs(endpoint@fresh["independent_map_float64"]),initial=0.0)<=1e-12,
        "span_projector":np.max(np.abs(
            fresh["independent_map_float64"]@fresh["independent_map_float64"].T
            -reduced@np.linalg.pinv(reduced,rcond=1e-12)),initial=0.0)<=1e-11,
        "compression":np.max(np.abs(
            fresh["independent_map_float64"]@fresh["raw_to_independent_float64"]-reduced
        ),initial=0.0)<=1e-12,
        "right_inverse": np.max(np.abs(endpoint @ fresh["endpoint_right_inverse_float64"] - np.eye(14))) <= 1e-12,
        "null_map": np.max(np.abs(endpoint @ reduced), initial=0.0) <= 1e-12,
        "particular_endpoint": np.max(np.abs(endpoint @ particular - target), initial=0.0) <= 1e-12,
    }
    return {"gates": gates, "passes": bool(all(gates.values()))}


def certify_expansion(coefficients, affine, q0, q_goal):
    acceleration = expand_coefficients(coefficients, affine)
    q, qd = integrate_acceleration(q0, np.zeros(7), acceleration)
    endpoint_residual = np.r_[q[-1] - q_goal, qd[-1]]
    return {
        "acceleration_float64": acceleration,
        "q_float64": q, "qd_float64": qd,
        "endpoint_residual_float64": endpoint_residual,
        "passes": bool(np.max(np.abs(endpoint_residual), initial=0.0) <= 1e-12),
    }


def array_hash(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(
        f"{value.dtype.str}|{value.shape}|".encode() + value.tobytes()
    ).hexdigest()


def campaign_adapter_declaration():
    """Static handoff for the later all-192 P2 campaign audit."""
    return {
        "ledger_count":192,"attempts_per_identity":1,"retries":0,
        "changed_component":"acquisition_coordinate_parameterization_only",
        "raw_spline_coefficients":RAW_SPLINE_VARIABLES,
        "independent_optimizer_variables":OPTIMIZER_VARIABLES,
        "unchanged": ["task_artifacts","model_artifacts","geometry","profiles",
            "objective","inequalities","Pin_replay","CUDA_worker","topology",
            "reversal","B1_B16_handoff","campaign_watchdog","final_certification"],
        "required_preflight_artifact":"accepted_P2_CPU_preflight_full_hash_set",
        "preflight_artifact_loads":1,"p1_artifact_loads":0,
        "campaign_tokens_enabled":False,
    }
