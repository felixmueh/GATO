"""Frozen pure schema for the Tiago circular route-seeded portfolio.

This module contains no model, artifact, CUDA, or optimizer imports.  The
benchmark is explicitly route/mode/pillar/q_goal-informed at construction
time; it is not an obstacle-blind homotopy-discovery experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


PROTOCOL_VERSION = "tiago_tool_center_circular_portfolio_p1_1"
PLANT_TYPE = "tiago_right_circular_portfolio_toll"
COLLISION_SCOPE = "tool_center_only"
KNOTS = 96
INTERVALS = 95
DT = 0.0125
NQ = NV = NU = 7
NX = 14
REFERENCE_SIZE = 10
BATCH_SIZE = 16
SHOOTING_VARIABLES = INTERVALS * NQ
ENDPOINT_EQUALITIES = 14
SHOOTING_INEQUALITIES = KNOTS * 14 + KNOTS * 14 + INTERVALS * 14 + KNOTS

DEVELOPMENT_TASK_SEEDS = tuple(range(12600, 12604))
HELDOUT_TASK_SEEDS = tuple(range(12700, 12708))
TASK_IDENTITIES = tuple(
    [("development", seed) for seed in DEVELOPMENT_TASK_SEEDS]
    + [("heldout", seed) for seed in HELDOUT_TASK_SEEDS]
)

# Ordered exact rational values.  Lane zero is the standalone B1 seed.
PROFILE_BETA_NUMERATOR = (0, -3, -1, -1, 1, 1, 3, 1)
PROFILE_BETA_DENOMINATOR = (1, 8, 4, 8, 8, 4, 8, 16)
PROFILE_BETAS = tuple(
    numerator / denominator
    for numerator, denominator in zip(
        PROFILE_BETA_NUMERATOR, PROFILE_BETA_DENOMINATOR
    )
)
ROUTES = ("short", "long")
EXPECTED_LEDGER = tuple(
    (split, seed, route, profile)
    for split, seed in TASK_IDENTITIES
    for route in ROUTES
    for profile in range(8)
)
assert len(EXPECTED_LEDGER) == 192

SHORT_ANGLE_RAD = 8.0 * np.pi / 9.0
LONG_ANGLE_RAD = 10.0 * np.pi / 9.0
PHYSICAL_RADIUS_M = 0.030
CLEARANCE_MARGIN_M = 0.005
KEEP_OUT_RADIUS_M = PHYSICAL_RADIUS_M + CLEARANCE_MARGIN_M
MIN_NOMINAL_PHYSICAL_CLEARANCE_M = 0.010
TOLL_WEIGHT = 0.5
CONSTRUCTION_Q_RESERVE_RAD = 0.01
CONSTRUCTION_LIMIT_RATIO = 0.95
CONSTRUCTION_CLEARANCE_M = 0.010

PATH_SCALE_M = 0.002
ACCELERATION_SCALE_RAD_S2 = 4.0
ACCELERATION_WEIGHT = 1e-4
JERK_WEIGHT = 1e-3
CONTROL_WEIGHT = 1e-4
DLS_STEPS = 8
DLS_DAMPING = 0.03
LINEAR_PROXY_REGULARIZATION = 1e-6

BACKEND = "scipy.optimize.minimize:trust-constr"
BACKEND_OPTIONS = {
    "maxiter": 300,
    "gtol": 1e-8,
    "xtol": 1e-10,
    "barrier_tol": 1e-10,
    "sparse_jacobian": True,
    "factorization_method": "AugmentedSystem",
    "initial_tr_radius": 1.0,
    "initial_constr_penalty": 1.0,
    "initial_barrier_parameter": 0.1,
    "initial_barrier_tolerance": 0.1,
    "verbose": 0,
}
BACKEND_HESSIAN = "scipy.optimize.BFGS()"
PROFILE_WALL_LIMIT_S = 120.0
CAMPAIGN_WALL_LIMIT_S = 21600.0
PROFILE_ATTEMPTS = 1

TERMINAL_CUDA_TOLERANCE_M = 0.015
TERMINAL_PIN_TOLERANCE_M = 0.020
FINAL_TOOL_SPEED_TOLERANCE_MPS = 0.05
MODEL_TOOL_TOLERANCE_M = 0.001
DENSE_SUBSTEPS = 64
DENSE_SAMPLES = INTERVALS * DENSE_SUBSTEPS + 1
PATH_RMS_TOLERANCE_M = 0.002
PATH_MAX_TOLERANCE_M = 0.004
RECURRENCE_TOLERANCE = 1e-12
ABA_RNEA_TOLERANCE = 1e-10
ENDPOINT_TOLERANCE = 1e-9
CLOSED_WINDING_RESIDUAL = 0.10

Q_COST = 2.0
N_COST = 260.0
QD_COST = 0.15
U_COST = 7.5e-5
CYLINDER_WEIGHT = 800.0
Q_LIMIT_COST = 0.01
VELOCITY_LIMIT_COST = 0.001
CONTROL_LIMIT_COST = 0.003
MAX_SQP_ITERS = 60
KKT_TOL = 1e-3
MAX_PCG_ITERS = 500
PCG_TOL = 8e-4
MU = 20.0
RHO = 0.01
SOLVE_RATIO = 1.0

FORBIDDEN_PUBLIC_FIELDS = frozenset(
    {
        "q_goal", "quarantined_q8", "pillar", "pillar_xy", "short_unit",
        "route", "mode", "profile", "arc", "constructor", "dls_proxy",
        "shooting_acceleration", "oracle", "witness",
    }
)

# Every runtime capability remains closed in P0/P1 static source.
CONSTRUCTOR_EXECUTION_AUTHORIZATION = None
OPTIMIZER_EXECUTION_AUTHORIZATION = None

TASK_ARTIFACT_ROOT = "/tmp/tiago-tool-center-toll-v4-task-construction-authorized-once"
TASK_ARTIFACT_PINS = {
    "v4.json": "6695c10a048ed07284422871201c458ed8f3a37fbd965591a3897d90ef682aac",
    "v4.npz": "4d343d9dc75bf51987c9a18f61239207d1959f59caab64e492ab711ff9bfc7e8",
    "v4.manifest.json": "72cef090d49c4f76efe3398b2dfcdd5b72c91cecfe28efc910f4d67a18a01007",
    "v4.partial.latest.json": "3d37c6851f0e92a8b541585aa0f6e61b4f9b4f402c7bd9b038cac2dbdde6d460",
}
MODEL_ARTIFACT_ROOT = "/tmp/tiago-tool-center-toll-v4-model-preflight-v4-authorized-once"
MODEL_ARTIFACT_PINS = {
    "model.json": "3493feae03b7b6368a0dc506aa1f0b5067c63c16f90e08b3455c2314f6e7cc16",
    "model.npz": "fe29896b3c5cacbfb15be2a66ddc222a88f8e2e5c2646e183cdbaac34bd1fb6e",
    "model.manifest.json": "438df87e388352ebe5762ca6e9cc3163019d1d9a9fdb616d5d4b027e8bc1a92c",
    "model.partial.latest.json": "d0eddf81f3a7072fda795d92ad7f0e75d869ff62cb282c3eb1b453b1af98539e",
    "model.worker-input.npz": "2e0953180dc9661a4740e57a6f94d9a8c4e438e7f4e3b34273bf5c07a154f426",
}
TASK_ARRAY_COUNT = 603
MODEL_ARRAY_COUNT = 150


def quintic_progress(tau):
    value = np.asarray(tau, dtype=np.float64)
    return 10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5


def profile_progress(tau, profile_index: int):
    if profile_index not in range(8):
        raise ValueError("profile index must be in [0,7]")
    base = quintic_progress(tau)
    beta = PROFILE_BETAS[profile_index]
    return base + beta * np.sin(2.0 * np.pi * base) / (2.0 * np.pi)


def profile_derivative_multiplier(progress, profile_index: int):
    if profile_index not in range(8):
        raise ValueError("profile index must be in [0,7]")
    return 1.0 + PROFILE_BETAS[profile_index] * np.cos(2.0 * np.pi * progress)


@dataclass(frozen=True)
class CircularGeometry:
    start_xyz: np.ndarray
    goal_xyz: np.ndarray
    midpoint_xy: np.ndarray
    chord_unit_xy: np.ndarray
    normal_xy: np.ndarray
    short_unit_xy: np.ndarray
    pillar_xy: np.ndarray
    circle_radius_m: float
    chord_offset_m: float
    toll_length_m: float
    orientation_sign: int


def construct_geometry(start_xyz, goal_xyz, default_side: int) -> CircularGeometry:
    start = np.asarray(start_xyz, dtype=np.float64)
    goal = np.asarray(goal_xyz, dtype=np.float64)
    if start.shape != (3,) or goal.shape != (3,) or not (
        np.isfinite(start).all() and np.isfinite(goal).all()
    ):
        raise ValueError("start and goal must be finite xyz vectors")
    if default_side not in (-1, 1):
        raise ValueError("default side must be +/-1")
    chord = goal[:2] - start[:2]
    distance = float(np.linalg.norm(chord))
    if distance <= 0.0:
        raise ValueError("planar chord must be nonzero")
    midpoint = 0.5 * (start[:2] + goal[:2])
    e = chord / distance
    n = np.asarray((-e[1], e[0]), dtype=np.float64)
    short = float(default_side) * n
    radius = distance / (2.0 * np.sin(4.0 * np.pi / 9.0))
    offset = distance / (2.0 * np.tan(4.0 * np.pi / 9.0))
    pillar = midpoint - offset * short
    ell = 0.5 * (KEEP_OUT_RADIUS_M - offset)
    orientation = int(np.sign(e[0] * short[1] - e[1] * short[0]))
    return CircularGeometry(
        start, goal, midpoint, e, n, short, pillar,
        float(radius), float(offset), float(ell), orientation,
    )


def certify_geometry(geometry: CircularGeometry) -> dict:
    chord_distance = geometry.chord_offset_m
    endpoint_radii = np.linalg.norm(
        np.stack((geometry.start_xyz[:2], geometry.goal_xyz[:2]))
        - geometry.pillar_xy[None, :], axis=1,
    )
    gates = {
        "finite": all(
            np.isfinite(value).all()
            for value in (
                geometry.start_xyz, geometry.goal_xyz, geometry.midpoint_xy,
                geometry.chord_unit_xy, geometry.short_unit_xy,
                geometry.pillar_xy,
            )
        ),
        "unit_vectors": bool(
            abs(np.linalg.norm(geometry.chord_unit_xy) - 1.0) <= 1e-12
            and abs(np.linalg.norm(geometry.short_unit_xy) - 1.0) <= 1e-12
            and abs(geometry.chord_unit_xy @ geometry.short_unit_xy) <= 1e-12
        ),
        "chord_intersects_physical": chord_distance < PHYSICAL_RADIUS_M,
        "chord_intersects_keepout": chord_distance < KEEP_OUT_RADIUS_M,
        "positive_toll_length": geometry.toll_length_m > 0.0,
        "nominal_clearance": (
            geometry.circle_radius_m - PHYSICAL_RADIUS_M
            >= MIN_NOMINAL_PHYSICAL_CLEARANCE_M
        ),
        "circle_endpoint_identity": np.max(
            np.abs(endpoint_radii - geometry.circle_radius_m), initial=0.0
        ) <= 1e-12,
        "vertical_travel": abs(
            float(geometry.goal_xyz[2] - geometry.start_xyz[2])
        ) <= 0.020,
        "orientation_sign": geometry.orientation_sign in (-1, 1),
    }
    return {"gates": gates, "passes": bool(all(gates.values()))}


def _rotate(vector, angle):
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray(
        (cosine * vector[0] - sine * vector[1],
         sine * vector[0] + cosine * vector[1]),
        dtype=np.float64,
    )


def circular_reference(geometry: CircularGeometry, route: str, profile_index: int):
    if route not in ROUTES:
        raise ValueError("route must be short or long")
    tau = np.arange(KNOTS, dtype=np.float64) / INTERVALS
    progress = profile_progress(tau, profile_index)
    delta = (
        -geometry.orientation_sign * SHORT_ANGLE_RAD
        if route == "short"
        else geometry.orientation_sign * LONG_ANGLE_RAD
    )
    radial = geometry.start_xyz[:2] - geometry.pillar_xy
    xy = np.stack(
        [geometry.pillar_xy + _rotate(radial, delta * value) for value in progress]
    )
    z = geometry.start_xyz[2] + (
        geometry.goal_xyz[2] - geometry.start_xyz[2]
    ) * progress
    return np.column_stack((xy, z))


@dataclass(frozen=True)
class CircularReference:
    goal_xyz: tuple[float, float, float]
    pillar_xy: tuple[float, float]
    physical_radius_m: float
    short_unit_xy: tuple[float, float]
    toll_length_m: float
    clearance_margin_m: float

    def as_float32(self):
        row = np.asarray(
            [*self.goal_xyz, *self.pillar_xy, self.physical_radius_m,
             *self.short_unit_xy, self.toll_length_m,
             self.clearance_margin_m],
            dtype=np.float32,
        )
        return validate_reference_row(row)


def reference_from_geometry(geometry: CircularGeometry) -> CircularReference:
    return CircularReference(
        tuple(geometry.goal_xyz), tuple(geometry.pillar_xy), PHYSICAL_RADIUS_M,
        tuple(geometry.short_unit_xy), geometry.toll_length_m,
        CLEARANCE_MARGIN_M,
    )


def validate_reference_row(row):
    value = np.asarray(row)
    if value.shape != (REFERENCE_SIZE,) or value.dtype != np.float32:
        raise ValueError("circular portfolio reference must be float32 shape (10,)")
    if not np.isfinite(value).all():
        raise ValueError("reference must be finite")
    if value[5] != np.float32(PHYSICAL_RADIUS_M) or value[9] != np.float32(
        CLEARANCE_MARGIN_M
    ):
        raise ValueError("reference radius and margin differ from frozen values")
    if value[8] <= 0 or abs(float(np.linalg.norm(value[6:8])) - 1.0) > 1e-6:
        raise ValueError("reference short unit/toll length invalid")
    return value


def smoothstep_toll_residual_gradient(position_xy, reference):
    p = np.asarray(position_xy, dtype=np.float64)
    ref = validate_reference_row(np.asarray(reference))
    center = ref[3:5].astype(np.float64)
    short = ref[6:8].astype(np.float64)
    ell = float(ref[8])
    offset = float(ref[5] + ref[9] - 2.0 * ref[8])
    midpoint = center + offset * short
    z = float(short @ (p - midpoint) / ell)
    if z <= 0.0:
        return 0.0, np.zeros(2), np.zeros((2, 2))
    if z >= 1.0:
        return 1.0, np.zeros(2), np.zeros((2, 2))
    residual = 3.0 * z**2 - 2.0 * z**3
    derivative = (6.0 * z - 6.0 * z**2) * short / ell
    gn = TOLL_WEIGHT * np.outer(derivative, derivative)
    return float(residual), derivative, gn


def toll_cost_gradient_gn(position_xy, reference, *, terminal=False):
    if terminal:
        return 0.0, np.zeros(2), np.zeros((2, 2))
    residual, derivative, gn = smoothstep_toll_residual_gradient(
        position_xy, reference
    )
    return (
        0.5 * TOLL_WEIGHT * residual**2,
        TOLL_WEIGHT * residual * derivative,
        gn,
    )


def validate_public_handoff(payload: Mapping, canonical: Mapping):
    tainted = sorted(FORBIDDEN_PUBLIC_FIELDS.intersection(payload))
    if tainted:
        raise ValueError(f"constructor-only fields crossed public boundary: {tainted}")
    required = {
        "identity", "default_side", "x0_float32", "reference_float32",
        "seed_xu_float32",
    }
    if set(payload) != required:
        raise ValueError("public handoff schema mismatch")
    if set(canonical) != {"identity", "default_side", "x0_float32", "reference_float32"}:
        raise ValueError("canonical task binding schema mismatch")
    x0 = np.asarray(payload["x0_float32"])
    reference = np.asarray(payload["reference_float32"])
    seed = np.asarray(payload["seed_xu_float32"])
    if not (
        x0.shape == (14,) and reference.shape == (KNOTS * REFERENCE_SIZE,)
        and seed.shape == (KNOTS * NX + INTERVALS * NU,)
        and x0.dtype == reference.dtype == seed.dtype == np.float32
        and np.isfinite(x0).all() and np.isfinite(reference).all()
        and np.isfinite(seed).all()
    ):
        raise ValueError("public handoff arrays must be exact finite float32 shapes")
    identity = tuple(payload["identity"])
    if identity not in TASK_IDENTITIES or identity != tuple(canonical["identity"]):
        raise ValueError("public task identity is not the accepted canonical identity")
    if payload["default_side"] not in (-1, 1) or payload["default_side"] != canonical["default_side"]:
        raise ValueError("public default side differs from accepted task")
    canonical_x0 = np.asarray(canonical["x0_float32"])
    canonical_reference = np.asarray(canonical["reference_float32"])
    if canonical_x0.shape != (14,) or canonical_x0.dtype != np.float32:
        raise ValueError("canonical x0 invalid")
    if canonical_reference.shape != (10,) or canonical_reference.dtype != np.float32:
        raise ValueError("canonical task reference invalid")
    if not np.array_equal(x0, canonical_x0):
        raise ValueError("public x0 differs from accepted task bytes")
    if not np.all(reference.reshape(KNOTS, REFERENCE_SIZE)[:, :3] == canonical_reference[:3]):
        raise ValueError("portfolio goal differs from accepted task goal")
    return True


def frozen_metadata():
    return {
        "protocol": PROTOCOL_VERSION,
        "task_identities": TASK_IDENTITIES,
        "ledger_count": len(EXPECTED_LEDGER),
        "route_seeded": True,
        "discovery_claim": False,
        "local_optimum_claim": False,
        "collision_scope": COLLISION_SCOPE,
        "knots": KNOTS,
        "dt": DT,
        "variables": SHOOTING_VARIABLES,
        "equalities": ENDPOINT_EQUALITIES,
        "inequalities": SHOOTING_INEQUALITIES,
        "backend": BACKEND,
        "backend_options": BACKEND_OPTIONS,
        "backend_hessian": BACKEND_HESSIAN,
        "task_artifact_pins": TASK_ARTIFACT_PINS,
        "model_artifact_pins": MODEL_ARTIFACT_PINS,
        "profile_attempts": PROFILE_ATTEMPTS,
        "campaign_wall_limit_s": CAMPAIGN_WALL_LIMIT_S,
        "execution_authorized": False,
    }
