#!/usr/bin/env python3
"""Mechanism-only IIWA target solve with the pillar objective disabled."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(REPO_ROOT),
    str(REPO_ROOT / "python"),
    str(REPO_ROOT / "tiago_src"),
]

from gato_tiago.iiwa_multimodal_sqp import (
    BUDGET,
    DEVELOPMENT_METHOD_SEEDS,
    DEVELOPMENT_PROTOCOL,
    DEVELOPMENT_TASK_SEEDS,
    FINAL_PROTOCOL,
    captured_solve_input_identity,
    dense_cuda_rollout,
    dense_pinocchio_rollout,
    dense_replay_metrics,
    execute_after_seed_gate,
    generate_discrete_rnea_seeds,
    generate_instance,
    load_model,
    solve_call_audit_passes,
    tool_path,
    unpack_planned_seeds,
)
from tiago_examples.iiwa_multimodal_diagnostics import (
    make_solver,
    provenance,
    sha256_array,
    solver_parameters,
    write_artifacts,
)
from tiago_examples.iiwa_multimodal_real_dev_probe import (
    seed_prerequisite,
    solve_exact_candidates,
)


EXPECTED_PLANNED_SEEDS_SHA256 = (
    "0897ea2fed948d0a469420d3d0e44af574b7c1dc03617a0e5f276ffc1a20205e"
)
EXPECTED_CONTROLS_SHA256 = (
    "bd6c278f68f2e3a2317b7c6f4805e7e19cb62b40cb84bac1f317f4fb1351c91c"
)


def exact_solver_parameter_ablation():
    baseline = solver_parameters(
        DEVELOPMENT_PROTOCOL.max_sqp_iters, DEVELOPMENT_PROTOCOL.pillar_cost
    )
    ablation = solver_parameters(DEVELOPMENT_PROTOCOL.max_sqp_iters, 0.0)
    changed = {
        key: {"baseline": baseline[key], "ablation": ablation[key]}
        for key in baseline
        if baseline[key] != ablation[key]
    }
    return {
        "baseline": baseline,
        "ablation": ablation,
        "changed_fields": changed,
        "passes": changed
        == {
            "ee_orient_cost": {
                "baseline": DEVELOPMENT_PROTOCOL.pillar_cost,
                "ablation": 0.0,
            },
            "ee_orient_N_cost": {
                "baseline": DEVELOPMENT_PROTOCOL.pillar_cost,
                "ablation": 0.0,
            },
        },
    }


def dense_seed_screen(model, instance, controls, sim_forward):
    x0 = np.hstack([instance["q0"], np.zeros(model.nv)])
    cuda = dense_cuda_rollout(
        sim_forward, x0, controls, DEVELOPMENT_PROTOCOL.dt
    )
    pinocchio = dense_pinocchio_rollout(
        model, x0, controls, DEVELOPMENT_PROTOCOL.dt
    )
    rows = []
    for candidate in range(BUDGET):
        cuda_metrics = dense_replay_metrics(
            model, instance, cuda[candidate], controls[candidate]
        )
        pin_metrics = dense_replay_metrics(
            model, instance, pinocchio[candidate], controls[candidate]
        )
        difference = np.asarray(cuda[candidate], dtype=np.float64) - pinocchio[
            candidate
        ]
        state_l2 = np.linalg.norm(difference, axis=1)
        cuda_tool = tool_path(model, cuda[candidate, :, : model.nq])
        pin_tool = tool_path(model, pinocchio[candidate, :, : model.nq])
        ee = np.linalg.norm(cuda_tool - pin_tool, axis=1)
        finite = bool(cuda_metrics["finite"] and pin_metrics["finite"])
        q_control_limits = bool(
            cuda_metrics["joint_violation_rad"] <= 0.0
            and pin_metrics["joint_violation_rad"] <= 0.0
            and cuda_metrics["control_ratio"] <= 1.0
            and pin_metrics["control_ratio"] <= 1.0
        )
        velocity_limits = bool(
            cuda_metrics["velocity_ratio"] <= 1.0
            and pin_metrics["velocity_ratio"] <= 1.0
        )
        model_agreement = bool(
            np.max(state_l2) <= 0.002 and np.max(ee) <= 0.001
        )
        rows.append(
            {
                "candidate_index": candidate,
                "finite": finite,
                "q_and_control_limits_pass": q_control_limits,
                "velocity_limits_pass": velocity_limits,
                "maximum_cuda_pin_state_l2": float(np.max(state_l2)),
                "maximum_cuda_pin_ee_disagreement_m": float(np.max(ee)),
                "model_agreement_pass": model_agreement,
                "cuda": cuda_metrics,
                "pinocchio": pin_metrics,
            }
        )
    aggregate = {
        "candidate_count": len(rows),
        "all_finite": all(row["finite"] for row in rows),
        "all_q_and_control_limits_pass": all(
            row["q_and_control_limits_pass"] for row in rows
        ),
        "all_velocity_limits_pass": all(
            row["velocity_limits_pass"] for row in rows
        ),
        "all_model_agreement_pass": all(
            row["model_agreement_pass"] for row in rows
        ),
    }
    aggregate["dense_seed_gate_pass"] = bool(
        len(rows) == BUDGET
        and aggregate["all_finite"]
        and aggregate["all_q_and_control_limits_pass"]
        and aggregate["all_velocity_limits_pass"]
        and aggregate["all_model_agreement_pass"]
    )
    aggregate["mechanism_only_velocity_bypass_prerequisites_pass"] = bool(
        len(rows) == BUDGET
        and aggregate["all_finite"]
        and aggregate["all_q_and_control_limits_pass"]
        and aggregate["all_model_agreement_pass"]
        and not aggregate["all_velocity_limits_pass"]
    )
    return rows, aggregate, cuda, pinocchio


def execution_report(
    model,
    instance,
    seeds,
    outputs,
    stats,
    sim_forward,
    target_only_protocol,
):
    _, _, seed_controls = unpack_planned_seeds(model, seeds)
    _, _, final_controls = unpack_planned_seeds(model, outputs)
    x0 = np.hstack([instance["q0"], np.zeros(model.nv)])
    seed_cuda = dense_cuda_rollout(
        sim_forward, x0, seed_controls, DEVELOPMENT_PROTOCOL.dt
    )
    seed_pin = dense_pinocchio_rollout(
        model, x0, seed_controls, DEVELOPMENT_PROTOCOL.dt
    )
    final_cuda = dense_cuda_rollout(
        sim_forward, x0, final_controls, DEVELOPMENT_PROTOCOL.dt
    )
    final_pin = dense_pinocchio_rollout(
        model, x0, final_controls, DEVELOPMENT_PROTOCOL.dt
    )
    rows = []
    for candidate in range(BUDGET):
        seed_cuda_metrics = dense_replay_metrics(
            model,
            instance,
            seed_cuda[candidate],
            seed_controls[candidate],
            target_only_protocol,
        )
        seed_pin_metrics = dense_replay_metrics(
            model,
            instance,
            seed_pin[candidate],
            seed_controls[candidate],
            target_only_protocol,
        )
        final_cuda_metrics = dense_replay_metrics(
            model,
            instance,
            final_cuda[candidate],
            final_controls[candidate],
            target_only_protocol,
        )
        final_pin_metrics = dense_replay_metrics(
            model,
            instance,
            final_pin[candidate],
            final_controls[candidate],
            target_only_protocol,
        )
        state_difference = (
            np.asarray(final_cuda[candidate], dtype=np.float64)
            - final_pin[candidate]
        )
        cuda_tool = final_cuda_metrics["dense_tool_path"]
        pin_tool = final_pin_metrics["dense_tool_path"]
        rows.append(
            {
                "candidate_index": candidate,
                "initial_solver_merit": float(stats["initial_merit"][candidate]),
                "final_solver_merit": float(stats["final_merit"][candidate]),
                "solver_merit_decrease": float(
                    stats["initial_merit"][candidate]
                    - stats["final_merit"][candidate]
                ),
                "packed_trajectory_rms_change": float(
                    np.linalg.norm(outputs[candidate] - seeds[candidate])
                    / np.sqrt(seeds.shape[1])
                ),
                "cuda_target_only_cost_decrease": float(
                    seed_cuda_metrics["external_cost"]
                    - final_cuda_metrics["external_cost"]
                ),
                "pinocchio_target_only_cost_decrease": float(
                    seed_pin_metrics["external_cost"]
                    - final_pin_metrics["external_cost"]
                ),
                "cuda_terminal_error_decrease_m": float(
                    seed_cuda_metrics["terminal_error_m"]
                    - final_cuda_metrics["terminal_error_m"]
                ),
                "pinocchio_terminal_error_decrease_m": float(
                    seed_pin_metrics["terminal_error_m"]
                    - final_pin_metrics["terminal_error_m"]
                ),
                "maximum_cuda_pin_state_l2": float(
                    np.max(np.linalg.norm(state_difference, axis=1))
                ),
                "maximum_cuda_pin_ee_disagreement_m": float(
                    np.max(np.linalg.norm(cuda_tool - pin_tool, axis=1))
                ),
                "pcg_cap_hits": int(stats["pcg_cap_hits"][candidate]),
                "sqp_iterations": int(stats["sqp_iterations"][candidate]),
                "seed_cuda": seed_cuda_metrics,
                "seed_pinocchio": seed_pin_metrics,
                "final_cuda": final_cuda_metrics,
                "final_pinocchio": final_pin_metrics,
            }
        )
    return rows, {
        "seed_controls": seed_controls,
        "final_controls": final_controls,
        "seed_cuda_dense_rollout": seed_cuda,
        "seed_pinocchio_dense_rollout": seed_pin,
        "final_cuda_dense_rollout": final_cuda,
        "final_pinocchio_dense_rollout": final_pin,
    }


def run(output):
    if FINAL_PROTOCOL is not None:
        raise RuntimeError("mechanism ablation requires final seed embargo")
    model = load_model()
    instance = generate_instance(DEVELOPMENT_TASK_SEEDS[0], model=model)
    seeds, controls, generation = generate_discrete_rnea_seeds(
        model, instance["q0"], DEVELOPMENT_METHOD_SEEDS[0]
    )
    exact_hashes = {
        "planned_seeds": sha256_array(seeds),
        "controls": sha256_array(controls),
    }
    exact_rejected_inputs = bool(
        exact_hashes["planned_seeds"] == EXPECTED_PLANNED_SEEDS_SHA256
        and exact_hashes["controls"] == EXPECTED_CONTROLS_SHA256
    )
    parameter_ablation = exact_solver_parameter_ablation()
    serial_solver = make_solver(1, DEVELOPMENT_PROTOCOL.max_sqp_iters, 0.0)
    batch_solver = make_solver(BUDGET, DEVELOPMENT_PROTOCOL.max_sqp_iters, 0.0)
    coarse = seed_prerequisite(
        model,
        instance,
        seeds,
        controls,
        batch_solver,
        initializer_kind="discrete_rnea",
    )
    dense_rows, dense_aggregate, dense_cuda, dense_pin = dense_seed_screen(
        model, instance, controls, batch_solver.sim_forward
    )
    bypass = bool(
        coarse["passes"]
        and dense_aggregate[
            "mechanism_only_velocity_bypass_prerequisites_pass"
        ]
        and exact_rejected_inputs
        and parameter_ablation["passes"]
    )
    solve_result, solve_call_audit = execute_after_seed_gate(
        bypass,
        lambda: solve_exact_candidates(
            model, instance, seeds, serial_solver, batch_solver
        ),
    )
    summary = {
        "schema_version": 1,
        "diagnostic": "iiwa_multimodal_target_only_mechanism_ablation",
        "benchmark_evidence": False,
        "multimodal_evidence": False,
        "all_gates_pass": False,
        "topology_and_support": "not_applicable",
        "known_seed_invalid_for_benchmark": True,
        "mechanism_only_limit_bypass": True,
        "dense_seed_gate_pass": dense_aggregate["dense_seed_gate_pass"],
        "warmups": 0,
        "repeats": 1,
        "task_seed": DEVELOPMENT_TASK_SEEDS[0],
        "method_seed": DEVELOPMENT_METHOD_SEEDS[0],
        "generation": generation,
        "exact_rejected_input_hashes": exact_hashes,
        "exact_rejected_inputs_pass": exact_rejected_inputs,
        "solver_parameter_ablation": parameter_ablation,
        "coarse_seed_prerequisite": {
            key: value
            for key, value in coarse.items()
            if key not in ("cuda_states", "pinocchio_states")
        },
        "dense_seed_rows": dense_rows,
        "dense_seed_aggregate": dense_aggregate,
        "mechanism_only_bypass_prerequisites_pass": bypass,
        "solve_call_audit": solve_call_audit,
        "solve_call_audit_passes": solve_call_audit_passes(solve_call_audit),
        "completed_solve_once_calls": sum(solve_call_audit.values()),
        "provenance": provenance(batch_solver, Path(__file__)),
    }
    arrays = {
        "planned_seeds": seeds,
        "seed_controls": controls,
        "coarse_seed_cuda_rollout": coarse["cuda_states"],
        "coarse_seed_pinocchio_rollout": coarse["pinocchio_states"],
        "dense_seed_cuda_rollout": dense_cuda,
        "dense_seed_pinocchio_rollout": dense_pin,
    }
    if solve_result is not None:
        input_identity = captured_solve_input_identity(
            solve_result["serial_captured_inputs"],
            solve_result["batch_captured_inputs"],
        )
        target_protocol = replace(DEVELOPMENT_PROTOCOL, pillar_cost=0.0)
        serial_rows, serial_arrays = execution_report(
            model,
            instance,
            seeds,
            solve_result["serial_outputs"],
            solve_result["serial_stats"],
            batch_solver.sim_forward,
            target_protocol,
        )
        batch_rows, batch_arrays = execution_report(
            model,
            instance,
            seeds,
            solve_result["batch_outputs"],
            solve_result["batch_stats"],
            batch_solver.sim_forward,
            target_protocol,
        )
        summary.update(
            {
                "captured_input_identity": input_identity,
                "serial_raw_stats": solve_result["serial_stats"],
                "batch_raw_stats": solve_result["batch_stats"],
                "serial_rows": serial_rows,
                "batch_rows": batch_rows,
                "mechanism_execution_integrity_pass": bool(
                    input_identity["passes"]
                    and solve_call_audit_passes(solve_call_audit)
                ),
            }
        )
        arrays.update(
            {
                "serial_outputs": solve_result["serial_outputs"],
                "batch_outputs": solve_result["batch_outputs"],
            }
        )
        for execution in ("serial", "batch"):
            captured = solve_result[f"{execution}_captured_inputs"]
            arrays.update(
                {
                    f"{execution}_solve_input_{field}": captured[field]
                    for field in ("x0", "reference", "seeds")
                }
            )
        for prefix, values in (
            ("serial", serial_arrays),
            ("batch", batch_arrays),
        ):
            arrays.update(
                {f"{prefix}_{key}": value for key, value in values.items()}
            )
    write_artifacts(output, summary, arrays)
    print(output)
    print(
        {
            "all_gates_pass": False,
            "dense_seed_gate_pass": summary["dense_seed_gate_pass"],
            "completed_solve_once_calls": summary["completed_solve_once_calls"],
        }
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args().output)
