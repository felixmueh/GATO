#!/usr/bin/env python3
"""Independent CPU endpoint-validity audit for frozen IIWA task 170.

This is a task-aware feasibility oracle.  It is deliberately quarantined from
benchmark seed generation and does not run GATO, CUDA, route planning, or a
dynamics solve.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pinocchio as pin
from scipy.optimize import differential_evolution, least_squares


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tiago_src"))

from gato_tiago.iiwa_multimodal_sqp import (  # noqa: E402
    DEVELOPMENT_PROTOCOL,
    MODEL_PATH,
    generate_instance,
    load_model,
    tool_position,
)


TASK_SEED = 170
LS_RANDOM_SEED = 20260812
LS_RANDOM_START_COUNT = 100
LS_SETTINGS = {
    "jac": "3-point",
    "max_nfev": 2000,
    "xtol": 1e-12,
    "ftol": 1e-12,
    "gtol": 1e-12,
}
DE_SEED = 20260813
DE_SETTINGS = {
    "workers": 1,
    "updating": "immediate",
    "polish": False,
    "popsize": 20,
    "maxiter": 2000,
    "tol": 1e-10,
    "atol": 1e-12,
    "mutation": (0.5, 1.0),
    "recombination": 0.7,
}
AGREEMENT_TOLERANCE_M = 1e-5


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def benchmark_position(model, q):
    """Exact benchmark FK: world position of the final A7 joint origin."""
    return tool_position(model, model.createData(), q)


def contact_position(model, q):
    """Report-only URDF contact frame, which is not the benchmark position."""
    data = model.createData()
    pin.forwardKinematics(model, data, np.asarray(q, dtype=np.float64))
    pin.updateFramePlacements(model, data)
    return data.oMf[model.getFrameId("contact")].translation.copy()


def residual(model, goal, q):
    return benchmark_position(model, q) - goal


def result_row(model, goal, start, result, method, index):
    q = np.asarray(result.x, dtype=np.float64)
    position = benchmark_position(model, q)
    contact = contact_position(model, q)
    vector = position - goal
    return {
        "method": method,
        "index": int(index),
        "start_q": np.asarray(start, dtype=np.float64),
        "start_sha256": sha256_array(np.asarray(start, dtype=np.float64)),
        "q": q,
        "benchmark_position_m": position,
        "benchmark_residual_vector_m": vector,
        "benchmark_residual_norm_m": float(np.linalg.norm(vector)),
        "contact_position_m_report_only": contact,
        "contact_residual_norm_m_report_only": float(np.linalg.norm(contact - goal)),
        "within_joint_limits": bool(
            np.all(q >= model.lowerPositionLimit)
            and np.all(q <= model.upperPositionLimit)
        ),
        "finite": bool(
            np.all(np.isfinite(q))
            and np.all(np.isfinite(position))
            and np.all(np.isfinite(vector))
        ),
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "nfev": int(result.nfev),
        "njev": None if result.njev is None else int(result.njev),
    }


def solve_bounded_ls(model, goal, start, method, index):
    result = least_squares(
        lambda q: residual(model, goal, q),
        np.asarray(start, dtype=np.float64),
        bounds=(model.lowerPositionLimit, model.upperPositionLimit),
        **LS_SETTINGS,
    )
    return result, result_row(model, goal, start, result, method, index)


def classify_endpoint(ls_best, de_best):
    agreement = abs(float(ls_best) - float(de_best))
    some_within_020 = min(ls_best, de_best) <= 0.020
    some_in_ambiguous_band = 0.020 < min(ls_best, de_best) <= 0.022
    both_above_022 = ls_best > 0.022 and de_best > 0.022
    agree = agreement <= AGREEMENT_TOLERANCE_M
    if both_above_022 and agree:
        decision = "empirically_unreachable_invalid_frozen_endpoint"
    elif some_within_020:
        decision = "endpoint_reachable_route_search_may_resume"
    else:
        decision = "requires_verifier_ruling"
    return {
        "decision": decision,
        "least_squares_best_residual_m": float(ls_best),
        "differential_evolution_polished_residual_m": float(de_best),
        "independent_best_residual_agreement_m": agreement,
        "agreement_within_1e_5_m": agree,
        "some_witness_within_0_020_m": some_within_020,
        "some_witness_in_0_020_to_0_022_m": some_in_ambiguous_band,
        "both_independent_best_residuals_above_0_022_m": both_above_022,
        "abort_route_and_dynamics_search": decision
        != "endpoint_reachable_route_search_may_resume",
    }


def run(output):
    model = load_model()
    instance = generate_instance(TASK_SEED, model=model)
    goal = np.asarray(instance["goal_position_m"], dtype=np.float64)
    q0 = np.asarray(instance["q0"], dtype=np.float64)
    q0_benchmark_position = benchmark_position(model, q0)
    rng = np.random.default_rng(LS_RANDOM_SEED)
    random_starts = rng.uniform(
        model.lowerPositionLimit,
        model.upperPositionLimit,
        size=(LS_RANDOM_START_COUNT, model.nq),
    )
    starts = np.vstack([q0, random_starts])

    ls_rows = []
    for index, start in enumerate(starts):
        _, row = solve_bounded_ls(model, goal, start, "bounded_least_squares", index)
        ls_rows.append(row)
    ls_best_index = int(
        np.argmin([row["benchmark_residual_norm_m"] for row in ls_rows])
    )

    bounds = list(zip(model.lowerPositionLimit, model.upperPositionLimit))
    de_result = differential_evolution(
        lambda q: float(residual(model, goal, q) @ residual(model, goal, q)),
        bounds,
        seed=DE_SEED,
        **DE_SETTINGS,
    )
    de_q = np.asarray(de_result.x, dtype=np.float64)
    de_position = benchmark_position(model, de_q)
    de_raw = {
        "q": de_q,
        "benchmark_position_m": de_position,
        "benchmark_residual_vector_m": de_position - goal,
        "benchmark_residual_norm_m": float(np.linalg.norm(de_position - goal)),
        "objective_squared_error_m2": float(de_result.fun),
        "success": bool(de_result.success),
        "message": str(de_result.message),
        "nit": int(de_result.nit),
        "nfev": int(de_result.nfev),
        "population_shape": list(np.asarray(de_result.population).shape),
        "population_energies_shape": list(
            np.asarray(de_result.population_energies).shape
        ),
    }
    _, de_polished = solve_bounded_ls(
        model, goal, de_q, "differential_evolution_then_bounded_ls", 0
    )

    decision = classify_endpoint(
        ls_rows[ls_best_index]["benchmark_residual_norm_m"],
        de_polished["benchmark_residual_norm_m"],
    )
    source = Path(__file__).resolve()
    task_identity = np.hstack(
        [
            q0,
            instance["start_position_m"],
            q0_benchmark_position,
            goal,
            instance["pillar_xy_m"],
            [instance["pillar_radius_m"]],
        ]
    )
    summary = {
        "schema_version": 1,
        "diagnostic": "iiwa_multimodal_frozen_endpoint_validity",
        "oracle_only": True,
        "benchmark_seed_eligible": False,
        "gato_or_cuda_calls": 0,
        "route_planner_calls": 0,
        "dynamics_solve_calls": 0,
        "task_seed": TASK_SEED,
        "task": {
            "q0": q0,
            "start_position_m": instance["start_position_m"],
            "q0_benchmark_position_m": q0_benchmark_position,
            "q0_fk_minus_generated_start_m": (
                q0_benchmark_position - instance["start_position_m"]
            ),
            "goal_position_m": goal,
            "start_to_goal_distance_m": float(
                np.linalg.norm(goal - instance["start_position_m"])
            ),
            "pillar_xy_m": instance["pillar_xy_m"],
            "pillar_radius_m": instance["pillar_radius_m"],
            "protocol": instance["metadata"]["protocol"],
            "identity_sha256": sha256_array(task_identity),
        },
        "frame_definition": {
            "validity_frame": "world position of data.oMi[model.njoints - 1] (A7 joint origin)",
            "validity_joint_name": str(model.names[model.njoints - 1]),
            "validity_joint_id": int(model.njoints - 1),
            "urdf_contact_frame_is_report_only": True,
            "contact_frame_id": int(model.getFrameId("contact")),
            "contact_frame_parent_joint": int(
                model.frames[model.getFrameId("contact")].parentJoint
            ),
            "contact_frame_placement_translation_m": model.frames[
                model.getFrameId("contact")
            ].placement.translation,
            "known_frame_mismatch_m": 0.04,
        },
        "bounds": {
            "lower_q": model.lowerPositionLimit,
            "upper_q": model.upperPositionLimit,
        },
        "least_squares": {
            "random_seed": LS_RANDOM_SEED,
            "q0_start_count": 1,
            "random_start_count": LS_RANDOM_START_COUNT,
            "total_start_count": len(starts),
            "random_distribution": "independent uniform over each full joint-limit interval",
            "settings": LS_SETTINGS,
            "best_index": ls_best_index,
            "best_row": ls_rows[ls_best_index],
            "rows": ls_rows,
        },
        "differential_evolution": {
            "random_seed": DE_SEED,
            "objective": "squared 3D benchmark-position residual in m^2",
            "settings": DE_SETTINGS,
            "raw": de_raw,
            "bounded_least_squares_polish": de_polished,
        },
        "decision": decision,
        "provenance": {
            "git_head": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "git_tracked_dirty": bool(
                subprocess.run(
                    ["git", "status", "--porcelain", "--untracked-files=no"],
                    cwd=REPO_ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            ),
            "model_path": str(MODEL_PATH),
            "model_sha256": sha256_file(MODEL_PATH),
            "source_path": str(source),
            "source_sha256": sha256_file(source),
        },
    }

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    npz = output.with_suffix(".npz")
    np.savez_compressed(
        npz,
        starts=starts,
        least_squares_q=np.asarray([row["q"] for row in ls_rows]),
        least_squares_fk=np.asarray(
            [row["benchmark_position_m"] for row in ls_rows]
        ),
        least_squares_residual=np.asarray(
            [row["benchmark_residual_vector_m"] for row in ls_rows]
        ),
        de_raw_q=de_q,
        de_raw_fk=de_position,
        de_population=np.asarray(de_result.population),
        de_population_energies=np.asarray(de_result.population_energies),
        de_polished_q=np.asarray(de_polished["q"]),
        de_polished_fk=np.asarray(de_polished["benchmark_position_m"]),
    )
    summary["artifacts"] = {"npz": str(npz), "npz_sha256": sha256_file(npz)}
    output.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n")
    manifest = output.with_suffix(".sha256.json")
    manifest.write_text(
        json.dumps(
            {
                "json": str(output),
                "json_sha256": sha256_file(output),
                "npz": str(npz),
                "npz_sha256": sha256_file(npz),
                "source_sha256": summary["provenance"]["source_sha256"],
                "model_sha256": summary["provenance"]["model_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(output)
    print(json.dumps(decision, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args().output)
