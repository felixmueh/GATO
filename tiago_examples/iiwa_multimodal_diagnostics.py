#!/usr/bin/env python3
"""Opt-in diagnostics for the isolated IIWA multimodal plant prototype."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pinocchio as pin


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "python"), str(REPO_ROOT / "tiago_src")]

from bsqp.interface import BSQP
from gato_tiago.iiwa_multimodal_sqp import (
    BUDGET,
    DEVELOPMENT_METHOD_SEEDS,
    DEVELOPMENT_PROTOCOL,
    DEVELOPMENT_TASK_SEEDS,
    FINAL_PROTOCOL,
    MODEL_PATH,
    batch_lane_repeatability,
    cpu_dense_pin_seed_screen,
    cuda_rollout,
    dense_cuda_rollout,
    dense_pinocchio_rollout,
    generate_instance,
    generate_planned_seeds,
    independent_pillar_cost,
    load_model,
    model_pair_preflight,
    pinocchio_rollout,
    planned_seed_screen,
    seed_kinematic_screen,
    seed_model_screen,
    tool_path,
    unpack_planned_seeds,
)


def sha256_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_artifacts(output, summary, arrays):
    """Write the retained NPZ, JSON, and an external hash manifest."""
    output.parent.mkdir(parents=True, exist_ok=True)
    npz_path = output.with_suffix(".npz")
    np.savez_compressed(npz_path, **arrays)
    summary["artifacts"] = {
        "npz": str(npz_path),
        "npz_sha256": sha256_file(npz_path),
    }
    output.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n")
    manifest = {
        "json": str(output),
        "json_sha256": sha256_file(output),
        "npz": str(npz_path),
        "npz_sha256": sha256_file(npz_path),
        "source_sha256": summary["provenance"]["source_sha256"],
    }
    manifest_path = output.with_suffix(".sha256.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest_path


def solver_parameters(max_sqp_iters, pillar_weight):
    protocol = DEVELOPMENT_PROTOCOL
    return {
        "max_sqp_iters": max_sqp_iters,
        "kkt_tol": 1e-3,
        "max_pcg_iters": protocol.max_pcg_iters,
        "pcg_tol": protocol.pcg_tol,
        "solve_ratio": 1.0,
        "mu": protocol.mu,
        "q_cost": protocol.q_cost,
        "qd_cost": protocol.qd_cost,
        "u_cost": protocol.u_cost,
        "N_cost": protocol.terminal_cost,
        "ee_orient_cost": pillar_weight,
        "ee_orient_N_cost": pillar_weight,
        "q_lim_cost": 0.01,
        "vel_lim_cost": 0.0,
        "ctrl_lim_cost": 0.0,
        "rho": protocol.rho,
    }


def make_solver(batch_size, max_sqp_iters, pillar_weight):
    return BSQP(
        model_path=str(MODEL_PATH),
        batch_size=batch_size,
        N=DEVELOPMENT_PROTOCOL.knots,
        dt=DEVELOPMENT_PROTOCOL.dt,
        plant_type="iiwa14_multimodal",
        **solver_parameters(max_sqp_iters, pillar_weight),
    )


def provenance(solver, harness_path=None):
    extension = Path(solver.lib.__file__)
    harness_path = Path(__file__).resolve() if harness_path is None else Path(harness_path).resolve()
    return {
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "git_tracked_dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=REPO_ROOT, check=True, capture_output=True, text=True,
            ).stdout.strip()
        ),
        "model_sha256": sha256_file(MODEL_PATH),
        "extension": str(extension),
        "extension_sha256": sha256_file(extension),
        "source_sha256": {
            "plant": sha256_file(
                REPO_ROOT / "gato/dynamics/iiwa14/iiwa14_plant.cuh"
            ),
            "module": sha256_file(
                REPO_ROOT / "tiago_src/gato_tiago/iiwa_multimodal_sqp.py"
            ),
            "harness": sha256_file(Path(__file__).resolve()),
            "executing_harness": sha256_file(harness_path),
        },
    }


def stats_snapshot(solver, max_sqp_iters):
    stats = solver.get_stats()
    pcg = np.asarray(stats["pcg_iters"], dtype=np.int32)
    return {
        "initial_merit": np.asarray(stats["initial_merit"], dtype=np.float64),
        "final_merit": np.asarray(stats["final_merit"], dtype=np.float64),
        "sqp_iterations": np.asarray(stats["sqp_iters"], dtype=np.int32),
        "total_pcg_iterations": np.sum(pcg, axis=0, dtype=np.int64),
        "pcg_cap_hits": np.sum(pcg >= DEVELOPMENT_PROTOCOL.max_pcg_iters, axis=0),
        "reached_requested_sqp_iteration_limit": (
            np.asarray(stats["sqp_iters"], dtype=np.int32) >= max_sqp_iters
        ),
    }


def solve_once(solver, x0, references, seeds, max_sqp_iters, capture=None):
    solver.reset_dual()
    solver.reset_rho()
    seed_input = seeds.copy()
    if capture is not None:
        capture.update(
            {
                "x0": np.asarray(x0).copy(),
                "reference": np.asarray(references).copy(),
                "seeds": seed_input.copy(),
            }
        )
    output, _ = solver.solve(x0, references, seed_input)
    return np.asarray(output), stats_snapshot(solver, max_sqp_iters)


def run_seed_screen(output):
    if FINAL_PROTOCOL is not None:
        raise RuntimeError("seed diagnostic expects final protocol to remain blocked")
    model = load_model()
    instance = generate_instance(DEVELOPMENT_TASK_SEEDS[0], model=model)
    seeds, controls, generation = generate_planned_seeds(
        model, instance["q0"], DEVELOPMENT_METHOD_SEEDS[0]
    )
    planned = planned_seed_screen(model, instance["q0"], seeds)
    cpu_rows, cpu_aggregate = seed_kinematic_screen(model, instance, controls)
    cpu_dense_rows, cpu_dense_aggregate = cpu_dense_pin_seed_screen(
        model, instance, controls
    )
    solver = make_solver(BUDGET, 1, DEVELOPMENT_PROTOCOL.pillar_cost)
    preflight = model_pair_preflight(
        solver.sim_forward, model, instance["q0"], DEVELOPMENT_PROTOCOL.dt
    )
    x0 = np.hstack([instance["q0"], np.zeros(model.nv)]).astype(np.float32)
    cuda_states = cuda_rollout(
        solver.sim_forward, x0, controls, DEVELOPMENT_PROTOCOL.dt
    )
    pin_states = pinocchio_rollout(model, x0, controls, DEVELOPMENT_PROTOCOL.dt)
    model_rows, model_aggregate = seed_model_screen(
        model, instance, controls, cuda_states, pin_states, preflight
    )
    dense_cuda_states = dense_cuda_rollout(
        solver.sim_forward, x0, controls, DEVELOPMENT_PROTOCOL.dt
    )
    dense_pin_states = dense_pinocchio_rollout(
        model, x0, controls, DEVELOPMENT_PROTOCOL.dt
    )
    dense_model_rows, dense_model_aggregate = seed_model_screen(
        model,
        instance,
        controls,
        dense_cuda_states,
        dense_pin_states,
        preflight,
    )
    summary = {
        "schema_version": 1,
        "diagnostic": "iiwa_multimodal_seed_only",
        "sqps_or_optimization_calls": 0,
        "task_seed": DEVELOPMENT_TASK_SEEDS[0],
        "method_seed": DEVELOPMENT_METHOD_SEEDS[0],
        "generation": generation,
        "instance": instance["metadata"],
        "planned_seed_screen": planned,
        "cpu_seed_replay": cpu_rows,
        "cpu_seed_replay_aggregate": cpu_aggregate,
        "cpu_dense4_seed_replay": cpu_dense_rows,
        "cpu_dense4_seed_replay_aggregate": cpu_dense_aggregate,
        "model_pair_preflight": preflight,
        "cuda_pinocchio_seed_replay": model_rows,
        "cuda_pinocchio_seed_replay_aggregate": model_aggregate,
        "cuda_pinocchio_dense4_seed_replay": dense_model_rows,
        "cuda_pinocchio_dense4_seed_replay_aggregate": dense_model_aggregate,
        "all_gates_pass": bool(
            planned["passes"]
            and cpu_aggregate["passes"]
            and cpu_dense_aggregate["passes"]
            and model_aggregate["passes"]
            and dense_model_aggregate["passes"]
        ),
        "input_hashes": {
            "common_x0": sha256_array(x0),
            "planned_seeds": sha256_array(seeds),
            "seed_controls": sha256_array(controls),
            "reference": sha256_array(instance["reference"]),
        },
        "provenance": provenance(solver),
    }
    write_artifacts(
        output,
        summary,
        {
            "planned_seeds": seeds,
            "controls": controls,
            "cuda_seed_rollout": cuda_states,
            "pinocchio_seed_rollout": pin_states,
            "cuda_dense4_seed_rollout": dense_cuda_states,
            "pinocchio_dense4_seed_rollout": dense_pin_states,
        },
    )
    print(output)
    print(json.dumps(json_safe(model_aggregate), indent=2, sort_keys=True))


def synthetic_seed(model):
    protocol = DEVELOPMENT_PROTOCOL
    q0 = np.zeros(model.nq, dtype=np.float32)
    progress = np.linspace(0.0, 1.0, protocol.knots)
    q = q0 + 0.04 * np.sin(np.pi * progress)[:, None] ** 2 * np.eye(model.nq)[1]
    gravity = pin.rnea(
        model, model.createData(), q0.astype(np.float64),
        np.zeros(model.nv), np.zeros(model.nv),
    ).astype(np.float32)
    nx, nu = model.nq + model.nv, model.nv
    seed = np.zeros(protocol.knots * (nx + nu) - nu, dtype=np.float32)
    for knot in range(protocol.knots):
        offset = knot * (nx + nu)
        seed[offset : offset + model.nq] = q[knot]
        if knot + 1 < protocol.knots:
            seed[offset + nx : offset + nx + nu] = gravity
    return q0, seed


def run_core_smoke(output):
    model = load_model()
    q0, seed = synthetic_seed(model)
    q, _, controls = unpack_planned_seeds(model, seed[None, :])
    start = tool_path(model, q[0])[0]
    pillar = start[:2] + np.asarray([0.005, 0.0])
    radius = 0.025
    reference = np.asarray([*start, *pillar, radius], dtype=np.float32)
    x0 = np.hstack([q0, np.zeros(model.nv)]).astype(np.float32)
    expected = independent_pillar_cost(
        model, np.hstack([q[0], np.zeros_like(q[0])]), pillar, radius, 800.0
    )

    zero_solver = make_solver(1, 2, 0.0)
    one_solver = make_solver(1, 2, 800.0)
    batch_solver = make_solver(BUDGET, 2, 800.0)
    ref1 = np.tile(reference, DEVELOPMENT_PROTOCOL.knots)[None, :]
    out0, stats0 = solve_once(zero_solver, x0[None, :], ref1, seed[None, :], 2)
    out1, stats1 = solve_once(one_solver, x0[None, :], ref1, seed[None, :], 2)
    batch_seed = np.tile(seed, (BUDGET, 1))
    batch_ref = np.tile(ref1, (BUDGET, 1))
    batch_x0 = np.tile(x0, (BUDGET, 1))
    outb, statsb = solve_once(batch_solver, batch_x0, batch_ref, batch_seed, 2)
    _, _, seed_u = unpack_planned_seeds(model, seed[None, :])
    _, _, out1_u = unpack_planned_seeds(model, out1)
    _, _, outb_u = unpack_planned_seeds(model, outb)
    seed_states = cuda_rollout(one_solver.sim_forward, x0, seed_u, DEVELOPMENT_PROTOCOL.dt)[0]
    serial_states = cuda_rollout(one_solver.sim_forward, x0, out1_u, DEVELOPMENT_PROTOCOL.dt)[0]
    batch_states = cuda_rollout(
        batch_solver.sim_forward, x0, outb_u, DEVELOPMENT_PROTOCOL.dt
    )
    seed_obstacle = independent_pillar_cost(model, seed_states, pillar, radius, 800.0)
    serial_obstacle = independent_pillar_cost(model, serial_states, pillar, radius, 800.0)
    batch_obstacles = [
        independent_pillar_cost(model, states, pillar, radius, 800.0)
        for states in batch_states
    ]
    batch_repeatability = batch_lane_repeatability(outb, statsb)
    merit_delta = float(stats1["initial_merit"][0] - stats0["initial_merit"][0])
    summary = {
        "schema_version": 1,
        "diagnostic": "iiwa_multimodal_core_cost_gradient_smoke",
        "benchmark_evidence": False,
        "sqp_solve_calls": 3,
        "synthetic": {
            "q_direction_joint": 1, "amplitude_rad": 0.04,
            "pillar_inside_offset_m": [0.005, 0.0], "radius_m": radius,
            "goal_equals_common_start": True, "no_ik": True,
            "immutable_initial_collision": True,
            "integrated_hinge_reduction_is_synthetic_smoke_only": True,
        },
        "input_hashes": {
            "common_x0": sha256_array(x0),
            "planned_seed": sha256_array(seed),
            "reference": sha256_array(ref1),
        },
        "initial_merit_delta": merit_delta,
        "independent_expected_pillar_cost": expected["weighted_cost"],
        "initial_merit_delta_matches": bool(
            np.isclose(merit_delta, expected["weighted_cost"], rtol=1e-4, atol=1e-3)
        ),
        "seed_replay_obstacle": seed_obstacle,
        "serial_replay_obstacle": serial_obstacle,
        "batch_replay_obstacle_by_lane": batch_obstacles,
        "serial_reduces_obstacle": bool(
            serial_obstacle["weighted_cost"] < seed_obstacle["weighted_cost"]
            and serial_obstacle["sum_positive_hinge"]
            < seed_obstacle["sum_positive_hinge"]
        ),
        "batch_reduces_obstacle": bool(
            all(
                row["weighted_cost"] < seed_obstacle["weighted_cost"]
                and row["sum_positive_hinge"] < seed_obstacle["sum_positive_hinge"]
                for row in batch_obstacles
            )
        ),
        "all_batch_lanes_noninferior": bool(
            all(
                row["weighted_cost"]
                <= serial_obstacle["weighted_cost"]
                + max(1e-3, 0.01 * serial_obstacle["weighted_cost"])
                and row["sum_positive_hinge"]
                <= serial_obstacle["sum_positive_hinge"] + 1e-6
                for row in batch_obstacles
            )
        ),
        "identical_input_batch_lane_repeatability": batch_repeatability,
        "stats": {"weight0": stats0, "serial_weight800": stats1, "batch_weight800": statsb},
        "no_pcg_caps_or_nonfinite": bool(
            all(np.all(np.isfinite(stats["initial_merit"])) and np.all(np.isfinite(stats["final_merit"]))
                and not np.any(stats["pcg_cap_hits"])
                for stats in (stats0, stats1, statsb))
            and all(
                np.all(np.isfinite(value))
                for value in (out0, out1, outb, seed_states, serial_states, batch_states)
            )
        ),
        "within_requested_one_or_two_sqp_iterations": bool(
            all(
                np.all((stats["sqp_iterations"] >= 1) & (stats["sqp_iterations"] <= 2))
                for stats in (stats0, stats1, statsb)
            )
        ),
        "provenance": provenance(batch_solver),
    }
    summary["all_gates_pass"] = bool(
        summary["initial_merit_delta_matches"]
        and summary["serial_reduces_obstacle"]
        and summary["batch_reduces_obstacle"]
        and summary["all_batch_lanes_noninferior"]
        and summary["identical_input_batch_lane_repeatability"]["passes"]
        and summary["no_pcg_caps_or_nonfinite"]
        and summary["within_requested_one_or_two_sqp_iterations"]
    )
    write_artifacts(
        output,
        summary,
        {
            "seed": seed,
            "weight0_output": out0,
            "serial_weight800_output": out1,
            "batch_weight800_output": outb,
            "seed_rollout": seed_states,
            "serial_rollout": serial_states,
            "batch_rollouts": batch_states,
        },
    )
    print(output)
    print(json.dumps({"all_gates_pass": summary["all_gates_pass"]}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", choices=("seed-screen", "core-smoke"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.diagnostic == "seed-screen":
        run_seed_screen(args.output)
    else:
        run_core_smoke(args.output)
