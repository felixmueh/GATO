"""Frozen schema and pure helpers for the Tiago multimodal-toll prototype.

This module deliberately contains no oracle runner and no CUDA/SQP import.
Task instantiation and optimizer execution remain disabled until a later,
explicitly reviewed stage.  The initializer boundary is obstacle- and
mode-neutral: it may receive the public Cartesian goal, but never cylinder,
toll, construction-witness, oracle, or mode information.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from gato_tiago.multimodal_pillar import MODEL_PATH, TOOL_FRAME, TORSO_FRAME


PROTOCOL_NAME = "tiago_tool_center_reachable_toll_pillar_v1"
COLLISION_SCOPE = "tool_center_only"
KNOTS = 64
DT = 0.0125
HORIZON_S = (KNOTS - 1) * DT
BUDGET = 16
REFERENCE_SIZE = 10

DEVELOPMENT_TASK_SEEDS = tuple(range(12000, 12004))
HELDOUT_TASK_SEEDS = tuple(range(12100, 12108))
CENTRAL_TASK_SEED = 12104
DEVELOPMENT_METHOD_SEEDS = tuple(range(22000, 22010))
HELDOUT_METHOD_SEEDS = tuple(range(22100, 22130))

Q0_JITTER_RAD = 0.015
PLANAR_ANGLE_RAD = 0.16
REQUESTED_TRAVEL_M = 0.15
DAMPING = 0.05
GOAL_JOINT_MARGIN_RAD = 0.08
ACTUAL_TRAVEL_RANGE_M = (0.13, 0.17)
MAX_VERTICAL_TRAVEL_M = 0.02

PHYSICAL_RADIUS_M = 0.030
CLEARANCE_MARGIN_M = 0.005
CYLINDER_OFFSET_M = 0.008
MIN_ENDPOINT_PHYSICAL_CLEARANCE_M = 0.040
TOLL_OFFSET_FROM_CYLINDER_M = PHYSICAL_RADIUS_M + 0.025
TOLL_SIGMA_M = 0.020
TOLL_WEIGHT = 0.5
CYLINDER_WEIGHT = 800.0

Q_COST = 2.0
N_COST = 260.0
QD_COST = 0.15
U_COST = 7.5e-5
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

INITIALIZER_PROGRESS = 0.55
INITIALIZER_BUMP_RAD = 0.04
OPEN_WINDING_THRESHOLD_RAD = 2.0
CLOSED_WINDING_INTEGER_RESIDUAL = 0.10

TERMINAL_CUDA_TOLERANCE_M = 0.015
TERMINAL_PINOCCHIO_TOLERANCE_M = 0.020
FINAL_TOOL_SPEED_TOLERANCE_MPS = 0.05
MODEL_TOOL_TOLERANCE_M = 0.001
MIN_CONTROL_CHANGE = 1e-3
MIN_TOOL_PATH_CHANGE_M = 0.005
MIN_OBJECTIVE_REDUCTION_FRACTION = 0.10
MIN_OBJECTIVE_REDUCTION_ABSOLUTE = 0.01
MIN_TERMINAL_REDUCTION_FRACTION = 0.50
MIN_TERMINAL_REDUCTION_M = 0.030

FORBIDDEN_INITIALIZER_FIELDS = frozenset(
    {
        "cylinder",
        "cylinder_xy",
        "physical_radius",
        "clearance_margin",
        "toll",
        "toll_xy",
        "toll_sigma",
        "task_seed",
        "default_sign",
        "default_side",
        "default_label",
        "mode",
        "mode_label",
        "q_goal",
        "oracle",
        "oracle_path",
        "oracle_witness",
        "construction_witness",
        "construction_q8",
        "oracle_endpoint_q",
        "quarantined_q8",
        "dls_history",
        "dls_intermediate_q",
        "ik_iterates",
        "intermediate_joint_margins",
        "box_dls_history",
        "box_dls_face_table",
        "box_dls_active_set",
        "collision",
        "winding",
    }
)

# Execution is intentionally impossible in this static checkpoint.  A later
# audited patch must replace these with explicit, stage-specific capabilities.
OPTIMIZER_EXECUTION_AUTHORIZATION = None


def _sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    header = f"{value.dtype.str}|{value.shape}|".encode()
    return hashlib.sha256(header + value.tobytes()).hexdigest()


def source_hashes(paths: Mapping[str, str | Path]):
    """Hash exact source bytes for later immutable artifact provenance."""

    result = {}
    for label, raw_path in sorted(paths.items()):
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        result[str(label)] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return result


@dataclass(frozen=True)
class TollReference:
    goal_xyz: tuple[float, float, float]
    cylinder_xy: tuple[float, float]
    physical_radius_m: float
    toll_xy: tuple[float, float]
    toll_sigma_m: float
    clearance_margin_m: float

    def as_float32(self) -> np.ndarray:
        row = np.asarray(
            [
                *self.goal_xyz,
                *self.cylinder_xy,
                self.physical_radius_m,
                *self.toll_xy,
                self.toll_sigma_m,
                self.clearance_margin_m,
            ],
            dtype=np.float32,
        )
        validate_reference_row(row)
        return row

    @classmethod
    def from_solver_bytes(cls, row: np.ndarray) -> "TollReference":
        canonical = validate_reference_row(row)
        promoted = canonical.astype(np.float64)
        return cls(
            goal_xyz=tuple(promoted[:3]),
            cylinder_xy=tuple(promoted[3:5]),
            physical_radius_m=float(promoted[5]),
            toll_xy=tuple(promoted[6:8]),
            toll_sigma_m=float(promoted[8]),
            clearance_margin_m=float(promoted[9]),
        )

    @property
    def optimizer_radius_m(self) -> float:
        return self.physical_radius_m + self.clearance_margin_m


@dataclass(frozen=True)
class TollTask:
    task_seed: int
    x0: tuple[float, ...]
    reference: TollReference
    default_side: int
    solver_x0_sha256: str
    solver_reference_sha256: str

    def public_metadata(self) -> dict:
        return asdict(self)


def validate_reference_row(row: np.ndarray) -> np.ndarray:
    value = np.asarray(row)
    if value.shape != (REFERENCE_SIZE,) or value.dtype != np.float32:
        raise ValueError("toll reference must have exact shape (10,) and dtype float32")
    if not np.all(np.isfinite(value)):
        raise ValueError("toll reference must be finite")
    frozen = {
        5: np.float32(PHYSICAL_RADIUS_M),
        8: np.float32(TOLL_SIGMA_M),
        9: np.float32(CLEARANCE_MARGIN_M),
    }
    if any(value[index] != expected for index, expected in frozen.items()):
        raise ValueError("radius, toll sigma, and clearance margin must equal frozen values")
    return value


def reference_batch(reference: TollReference, batch_size: int = BUDGET) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    flat = np.tile(reference.as_float32(), KNOTS)
    return np.tile(flat, (batch_size, 1)).astype(np.float32, copy=False)


def _positive_hinge(position_xy, center_xy, radius):
    delta = np.asarray(position_xy, dtype=np.float64) - np.asarray(center_xy, dtype=np.float64)
    radius = float(radius)
    if radius <= 0:
        raise ValueError("radius must be positive")
    residual = 1.0 - float(delta @ delta) / (radius * radius)
    if residual <= 0:
        return 0.0, np.zeros(2, dtype=np.float64)
    return residual, -2.0 * delta / (radius * radius)


def toll_residual_gradient(position_xy, toll_xy, sigma):
    delta = np.asarray(position_xy, dtype=np.float64) - np.asarray(toll_xy, dtype=np.float64)
    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    residual = float(np.exp(-float(delta @ delta) / (2.0 * sigma * sigma)))
    return residual, -residual * delta / (sigma * sigma)


def workspace_cost_gradient_gn(position, reference, *, terminal=False):
    """Independent target/cylinder/toll cost, gradient, and PSD GN matrix."""

    p = np.asarray(position, dtype=np.float64)
    if p.shape != (3,) or not np.all(np.isfinite(p)):
        raise ValueError("position must be one finite xyz vector")
    ref = reference if isinstance(reference, TollReference) else TollReference.from_solver_bytes(reference)
    goal = np.asarray(ref.goal_xyz, dtype=np.float64)
    target_weight = N_COST if terminal else Q_COST
    error = p - goal
    cost = 0.5 * target_weight * float(error @ error)
    gradient = target_weight * error
    gn = target_weight * np.eye(3, dtype=np.float64)

    cylinder, cylinder_grad_xy = _positive_hinge(
        p[:2], ref.cylinder_xy, ref.optimizer_radius_m
    )
    cost += 0.5 * CYLINDER_WEIGHT * cylinder * cylinder
    gradient[:2] += CYLINDER_WEIGHT * cylinder * cylinder_grad_xy
    gn[:2, :2] += CYLINDER_WEIGHT * np.outer(cylinder_grad_xy, cylinder_grad_xy)

    toll = 0.0
    if not terminal:
        toll, toll_grad_xy = toll_residual_gradient(p[:2], ref.toll_xy, ref.toll_sigma_m)
        cost += 0.5 * TOLL_WEIGHT * toll * toll
        gradient[:2] += TOLL_WEIGHT * toll * toll_grad_xy
        gn[:2, :2] += TOLL_WEIGHT * np.outer(toll_grad_xy, toll_grad_xy)

    return {
        "cost": float(cost),
        "gradient": gradient,
        "gauss_newton": gn,
        "target_cost": float(0.5 * target_weight * (error @ error)),
        "cylinder_cost": float(0.5 * CYLINDER_WEIGHT * cylinder * cylinder),
        "toll_cost": float(0.5 * TOLL_WEIGHT * toll * toll),
    }


def physical_clearance(position_xy, reference) -> float:
    ref = reference if isinstance(reference, TollReference) else TollReference.from_solver_bytes(reference)
    return float(
        np.linalg.norm(np.asarray(position_xy, dtype=np.float64) - np.asarray(ref.cylinder_xy))
        - ref.physical_radius_m
    )


def minimum_jerk_progress(knots=KNOTS):
    t = np.linspace(0.0, 1.0, int(knots), dtype=np.float64)
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def initializer_nominal_dq(position_jacobian, start_xyz, goal_xyz):
    """The initializer's sole goal-aware operation: one frozen DLS step."""

    jacobian = np.asarray(position_jacobian, dtype=np.float64)
    start = np.asarray(start_xyz, dtype=np.float64)
    goal = np.asarray(goal_xyz, dtype=np.float64)
    if jacobian.shape != (3, 7) or start.shape != (3,) or goal.shape != (3,):
        raise ValueError("initializer DLS expects J(3,7), start xyz, and goal xyz")
    error = goal - start
    return jacobian.T @ np.linalg.solve(
        jacobian @ jacobian.T + DAMPING**2 * np.eye(3), error
    )


def initializer_directions(method_seed, nq=7):
    rng = np.random.default_rng(int(method_seed))
    draws = rng.normal(size=(8, int(nq)))
    norms = np.linalg.norm(draws, axis=1)
    if np.any(norms == 0):
        raise RuntimeError("degenerate initializer direction")
    draws /= norms[:, None]
    result = [np.zeros(nq, dtype=np.float64)]
    for direction in draws[:7]:
        result.extend((direction, -direction))
    result.append(draws[7])
    return np.asarray(result, dtype=np.float64)


def validate_initializer_inputs(payload: Mapping):
    keys = set(payload)
    tainted = sorted(keys.intersection(FORBIDDEN_INITIALIZER_FIELDS))
    if tainted:
        raise ValueError(f"initializer input is tainted by forbidden fields: {tainted}")
    required = {"robot_model", "x0", "goal_xyz", "limits", "knots", "dt", "method_seed"}
    if keys != required:
        raise ValueError(f"initializer input fields must be exactly {sorted(required)}")
    if int(payload["knots"]) != KNOTS or float(payload["dt"]) != DT:
        raise ValueError("initializer time grid differs from the frozen protocol")
    return True


def planned_initializer_paths(q0, nominal_dq, method_seed):
    """Pure planned-q schema; RNEA and one rollout belong to a later stage."""

    q0 = np.asarray(q0, dtype=np.float64)
    nominal_dq = np.asarray(nominal_dq, dtype=np.float64)
    if q0.shape != (7,) or nominal_dq.shape != (7,):
        raise ValueError("q0 and nominal_dq must be seven-vectors")
    progress = minimum_jerk_progress()
    envelope = np.sin(np.pi * np.linspace(0.0, 1.0, KNOTS)) ** 2
    base = q0[None, :] + INITIALIZER_PROGRESS * progress[:, None] * nominal_dq[None, :]
    directions = initializer_directions(method_seed)
    paths = base[None, :, :] + INITIALIZER_BUMP_RAD * envelope[None, :, None] * directions[:, None, :]
    paths[:, 0] = q0
    return paths.astype(np.float32), {
        "candidate_count": BUDGET,
        "direction_family": "goal55_nominal_plus_7_antithetic_pairs_plus_1_independent",
        "candidate_zero": "obvious_default_partial_dls_nominal",
        "goal_progress_fraction": INITIALIZER_PROGRESS,
        "joint_bump_amplitude_rad": INITIALIZER_BUMP_RAD,
        "rnea_calls_per_candidate": KNOTS - 1,
        "cuda_rollouts_per_candidate": 1,
        "forbidden_input_fields": sorted(FORBIDDEN_INITIALIZER_FIELDS),
        "feedback_iterations": 0,
        "calibration_iterations": 0,
        "rejection_attempts": 0,
    }


def signed_open_turn(path_xy, center_xy):
    path = np.asarray(path_xy, dtype=np.float64)
    vectors = path[:, :2] - np.asarray(center_xy, dtype=np.float64)
    angles = np.unwrap(np.arctan2(vectors[:, 1], vectors[:, 0]))
    angle = float(angles[-1] - angles[0])
    if angle >= OPEN_WINDING_THRESHOLD_RAD:
        return "counterclockwise", angle
    if angle <= -OPEN_WINDING_THRESHOLD_RAD:
        return "clockwise", angle
    return "neutral", angle


def closed_loop_winding(path_a_xy, path_b_xy, center_xy):
    a = np.asarray(path_a_xy, dtype=np.float64)[:, :2]
    b = np.asarray(path_b_xy, dtype=np.float64)[:, :2]
    loop = np.vstack([a, b[-2::-1]])
    vectors = loop - np.asarray(center_xy, dtype=np.float64)
    following = np.roll(vectors, -1, axis=0)
    increments = np.arctan2(
        vectors[:, 0] * following[:, 1] - vectors[:, 1] * following[:, 0],
        np.sum(vectors * following, axis=1),
    )
    winding = float(np.sum(increments) / (2 * np.pi))
    nearest = int(np.rint(winding))
    return {
        "winding": winding,
        "nearest_integer": nearest,
        "integer_residual": abs(winding - nearest),
        "valid_integer": abs(winding - nearest) <= CLOSED_WINDING_INTEGER_RESIDUAL,
    }


def _rms_difference(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("RMS inputs must have identical shapes")
    difference = left - right
    return float(np.sqrt(np.mean(difference * difference)))


def topology_certificate(
    rows: Sequence[Mapping],
    center_xy,
    *,
    path_key="tool_path",
    control_key="normalized_controls",
):
    """Fail-closed keyed topology over every productive supported path."""

    classified = {"clockwise": [], "counterclockwise": []}
    distinct_supported = {"clockwise": [], "counterclockwise": []}
    productive_ids = [
        int(row["candidate_index"])
        for row in rows
        if row.get("productive_certified", False)
    ]
    unique_ids = len(productive_ids) == len(set(productive_ids))
    candidate_ids_in_range = all(0 <= value < BUDGET for value in productive_ids)
    productive_rows = [row for row in rows if row.get("productive_certified", False)]
    path_shapes = [np.shape(row[path_key]) for row in productive_rows]
    control_shapes = [np.shape(row.get(control_key)) for row in productive_rows]
    common_path_shape = bool(path_shapes) and len(set(path_shapes)) == 1
    common_path_shape &= len(path_shapes[0]) == 2 and path_shapes[0][0] > 1 and path_shapes[0][1] >= 2
    common_control_shape = bool(control_shapes) and len(set(control_shapes)) == 1
    common_control_shape &= len(control_shapes[0]) == 2 and all(value > 0 for value in control_shapes[0])
    finite_paths = True
    finite_controls = True
    for row in rows:
        if not row.get("productive_certified", False):
            continue
        path_array = np.asarray(row[path_key])
        control_array = np.asarray(row.get(control_key))
        finite_paths &= bool(
            np.issubdtype(path_array.dtype, np.number)
            and np.all(np.isfinite(path_array))
        )
        finite_controls &= bool(
            np.issubdtype(control_array.dtype, np.number)
            and np.all(np.isfinite(control_array))
        )
        label, angle = signed_open_turn(row[path_key], center_xy)
        if label in classified:
            candidate = (int(row["candidate_index"]), row[path_key], angle, row.get(control_key))
            classified[label].append(candidate)

    shapes_valid = common_path_shape and common_control_shape
    if shapes_valid:
        for mode, candidates in classified.items():
            for candidate in candidates:
                distinct = all(
                    _rms_difference(candidate[1], previous[1]) >= MIN_TOOL_PATH_CHANGE_M
                    or _rms_difference(candidate[3], previous[3]) >= MIN_CONTROL_CHANGE
                    for previous in distinct_supported[mode]
                )
                if distinct:
                    distinct_supported[mode].append(candidate)

    same, cross = [], []
    for mode, values in classified.items():
        for left in range(len(values)):
            for right in range(left + 1, len(values)):
                result = closed_loop_winding(values[left][1], values[right][1], center_xy)
                same.append({"mode": mode, "left": values[left][0], "right": values[right][0], **result})
    for clockwise in classified["clockwise"]:
        for counterclockwise in classified["counterclockwise"]:
            result = closed_loop_winding(clockwise[1], counterclockwise[1], center_xy)
            cross.append({"clockwise": clockwise[0], "counterclockwise": counterclockwise[0], **result})

    same_pass = all(row["valid_integer"] and row["nearest_integer"] == 0 for row in same)
    cross_pass = bool(cross) and all(
        row["valid_integer"] and abs(row["nearest_integer"]) == 1 for row in cross
    )
    classified_counts = {mode: len(values) for mode, values in classified.items()}
    support = {mode: len(values) for mode, values in distinct_supported.items()}
    return {
        "support": support,
        "classified_productive_counts": classified_counts,
        "classified_candidate_indices": {
            mode: [value[0] for value in values] for mode, values in classified.items()
        },
        "supported_candidate_indices": {
            mode: [value[0] for value in values]
            for mode, values in distinct_supported.items()
        },
        "same_mode_pairs": same,
        "cross_mode_pairs": cross,
        "all_same_mode_class_zero": same_pass,
        "all_cross_mode_class_one": cross_pass,
        "unique_productive_candidate_ids": unique_ids,
        "candidate_ids_in_range": candidate_ids_in_range,
        "common_nonempty_path_shape": common_path_shape,
        "common_nonempty_control_shape": common_control_shape,
        "finite_paths": finite_paths,
        "finite_controls": finite_controls,
        "two_mode_two_support": all(value >= 2 for value in support.values())
        and same_pass
        and cross_pass
        and unique_ids
        and candidate_ids_in_range
        and shapes_valid
        and finite_paths
        and finite_controls,
    }


def model_topology_agreement(rows: Sequence[Mapping], center_xy):
    cuda = topology_certificate(rows, center_xy, path_key="cuda_tool_path")
    pin = topology_certificate(rows, center_xy, path_key="pin_tool_path")

    def keyed(certificate, key, identity_fields):
        return sorted(
            tuple(row[field] for field in identity_fields)
            + (row["nearest_integer"], row["valid_integer"])
            for row in certificate[key]
        )

    same_fields = ("mode", "left", "right")
    cross_fields = ("clockwise", "counterclockwise")
    gates = {
        "candidate_path_shapes": all(
            np.shape(row["cuda_tool_path"]) == np.shape(row["pin_tool_path"])
            for row in rows
            if row.get("productive_certified", False)
        ),
        "supported_candidate_ids": cuda["supported_candidate_indices"]
        == pin["supported_candidate_indices"],
        "classified_candidate_ids": cuda["classified_candidate_indices"]
        == pin["classified_candidate_indices"],
        "same_mode_keyed_classes": keyed(cuda, "same_mode_pairs", same_fields)
        == keyed(pin, "same_mode_pairs", same_fields),
        "cross_mode_keyed_classes": keyed(cuda, "cross_mode_pairs", cross_fields)
        == keyed(pin, "cross_mode_pairs", cross_fields),
        "cuda_two_mode_two_support": cuda["two_mode_two_support"],
        "pin_two_mode_two_support": pin["two_mode_two_support"],
    }
    return {
        "cuda": cuda,
        "pinocchio": pin,
        "gates": gates,
        "all_model_topology_gates_pass": all(gates.values()),
    }


def certify_productive_replay(metrics: Mapping):
    """Apply the frozen external replay gates; toll never implies feasibility."""

    required = {
        "finite",
        "cuda_seed_objective",
        "cuda_final_objective",
        "pin_seed_objective",
        "pin_final_objective",
        "seed_solver_merit",
        "final_solver_merit",
        "cuda_seed_terminal_error_m",
        "cuda_final_terminal_error_m",
        "pin_seed_terminal_error_m",
        "pin_final_terminal_error_m",
        "cuda_final_tool_speed_mps",
        "pin_final_tool_speed_mps",
        "cuda_physical_clearance_margin_m",
        "pin_physical_clearance_margin_m",
        "cuda_joint_ratio",
        "pin_joint_ratio",
        "cuda_velocity_ratio",
        "pin_velocity_ratio",
        "control_ratio",
        "model_tool_error_m",
        "normalized_control_change",
        "tool_path_rms_change_m",
        "cuda_initial_collision_violation_m",
        "cuda_final_collision_violation_m",
        "pin_initial_collision_violation_m",
        "pin_final_collision_violation_m",
        "pcg_cap_hits",
        "sqp_hit_max",
    }
    if set(metrics) != required:
        raise ValueError(f"replay metric fields must be exactly {sorted(required)}")
    cuda_seed_cost = float(metrics["cuda_seed_objective"])
    cuda_final_cost = float(metrics["cuda_final_objective"])
    pin_seed_cost = float(metrics["pin_seed_objective"])
    pin_final_cost = float(metrics["pin_final_objective"])
    seed_merit = float(metrics["seed_solver_merit"])
    final_merit = float(metrics["final_solver_merit"])
    cuda_seed_terminal = float(metrics["cuda_seed_terminal_error_m"])
    pin_seed_terminal = float(metrics["pin_seed_terminal_error_m"])
    cuda_terminal_reduction = cuda_seed_terminal - float(
        metrics["cuda_final_terminal_error_m"]
    )
    pin_terminal_reduction = pin_seed_terminal - float(
        metrics["pin_final_terminal_error_m"]
    )
    cuda_objective_reduction = cuda_seed_cost - cuda_final_cost
    pin_objective_reduction = pin_seed_cost - pin_final_cost
    merit_reduction = seed_merit - final_merit

    numeric_keys = required.difference({"finite", "sqp_hit_max"})
    numeric_finite = all(np.isfinite(float(metrics[key])) for key in numeric_keys)
    nonnegative_keys = numeric_keys.difference(
        {
            "seed_solver_merit",
            "final_solver_merit",
            "cuda_physical_clearance_margin_m",
            "pin_physical_clearance_margin_m",
        }
    )
    numeric_nonnegative = all(float(metrics[key]) >= 0 for key in nonnegative_keys)

    def collision_reduced(prefix):
        initial = float(metrics[f"{prefix}_initial_collision_violation_m"])
        final = float(metrics[f"{prefix}_final_collision_violation_m"])
        return final < initial if initial > 0 else final <= 0

    gates = {
        "finite": bool(metrics["finite"]) and numeric_finite,
        "numeric_nonnegative_domain": numeric_nonnegative,
        "solver_merit_reduction": merit_reduction
        >= max(
            MIN_OBJECTIVE_REDUCTION_ABSOLUTE,
            MIN_OBJECTIVE_REDUCTION_FRACTION * abs(seed_merit),
        ),
        "cuda_objective_reduction": cuda_objective_reduction
        >= max(
            MIN_OBJECTIVE_REDUCTION_ABSOLUTE,
            MIN_OBJECTIVE_REDUCTION_FRACTION * abs(cuda_seed_cost),
        ),
        "pin_objective_reduction": pin_objective_reduction
        >= max(
            MIN_OBJECTIVE_REDUCTION_ABSOLUTE,
            MIN_OBJECTIVE_REDUCTION_FRACTION * abs(pin_seed_cost),
        ),
        "cuda_terminal_reduction": cuda_terminal_reduction
        >= max(
            MIN_TERMINAL_REDUCTION_M,
            MIN_TERMINAL_REDUCTION_FRACTION * cuda_seed_terminal,
        ),
        "pin_terminal_reduction": pin_terminal_reduction
        >= max(
            MIN_TERMINAL_REDUCTION_M,
            MIN_TERMINAL_REDUCTION_FRACTION * pin_seed_terminal,
        ),
        "terminal_cuda": float(metrics["cuda_final_terminal_error_m"])
        <= TERMINAL_CUDA_TOLERANCE_M,
        "terminal_pinocchio": float(metrics["pin_final_terminal_error_m"])
        <= TERMINAL_PINOCCHIO_TOLERANCE_M,
        "cuda_tool_speed": float(metrics["cuda_final_tool_speed_mps"])
        <= FINAL_TOOL_SPEED_TOLERANCE_MPS,
        "pin_tool_speed": float(metrics["pin_final_tool_speed_mps"])
        <= FINAL_TOOL_SPEED_TOLERANCE_MPS,
        "cuda_physical_clearance": float(
            metrics["cuda_physical_clearance_margin_m"]
        )
        >= CLEARANCE_MARGIN_M,
        "pin_physical_clearance": float(
            metrics["pin_physical_clearance_margin_m"]
        )
        >= CLEARANCE_MARGIN_M,
        "cuda_limits": max(
            float(metrics["cuda_joint_ratio"]),
            float(metrics["cuda_velocity_ratio"]),
            float(metrics["control_ratio"]),
        )
        <= 1.0,
        "pin_limits": max(
            float(metrics["pin_joint_ratio"]),
            float(metrics["pin_velocity_ratio"]),
            float(metrics["control_ratio"]),
        )
        <= 1.0,
        "model_tool_agreement": float(metrics["model_tool_error_m"])
        <= MODEL_TOOL_TOLERANCE_M,
        "material_control_change": float(metrics["normalized_control_change"]) >= MIN_CONTROL_CHANGE,
        "material_tool_change": float(metrics["tool_path_rms_change_m"]) >= MIN_TOOL_PATH_CHANGE_M,
        "cuda_collision_violation_reduced": collision_reduced("cuda"),
        "pin_collision_violation_reduced": collision_reduced("pin"),
        "no_caps": float(metrics["pcg_cap_hits"]) == 0
        and not bool(metrics["sqp_hit_max"]),
    }
    return {
        "gates": gates,
        "cuda_objective_reduction": cuda_objective_reduction,
        "pin_objective_reduction": pin_objective_reduction,
        "solver_merit_reduction": merit_reduction,
        "cuda_terminal_reduction_m": cuda_terminal_reduction,
        "pin_terminal_reduction_m": pin_terminal_reduction,
        "productive_certified": all(gates.values()),
        "toll_used_as_feasibility": False,
        "collision_scope": COLLISION_SCOPE,
    }


def frozen_protocol_metadata():
    return {
        "protocol": PROTOCOL_NAME,
        "knots": KNOTS,
        "dt": DT,
        "horizon_s": HORIZON_S,
        "budget": BUDGET,
        "reference_size": REFERENCE_SIZE,
        "frames": {"base": TORSO_FRAME, "tool": TOOL_FRAME},
        "collision_scope": COLLISION_SCOPE,
        "model_path": str(MODEL_PATH),
        "task_instantiation_authorized": False,
        "optimizer_execution_authorized": OPTIMIZER_EXECUTION_AUTHORIZATION is not None,
    }
