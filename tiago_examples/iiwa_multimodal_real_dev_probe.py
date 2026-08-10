#!/usr/bin/env python3
"""Opt-in real-task IIWA multimodal development probe (final seeds blocked)."""

from __future__ import annotations

import argparse
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
    all_rows_cap_free,
    captured_solve_input_identity,
    certify_real_execution,
    compare_serial_batch,
    cpu_dense_pin_seed_screen,
    cuda_rollout,
    dense_cuda_rollout,
    dense_pinocchio_rollout,
    execute_after_seed_gate,
    generate_instance,
    generate_planned_seeds,
    load_model,
    mode_certificates_agree,
    mode_support_certificate,
    model_pair_preflight,
    pinocchio_rollout,
    planned_seed_screen,
    seed_kinematic_screen,
    seed_model_screen,
    solve_call_audit_passes,
)
from tiago_examples.iiwa_multimodal_diagnostics import (
    json_safe,
    make_solver,
    provenance,
    sha256_array,
    solve_once,
    write_artifacts,
)


def concatenate_stats(rows):
    return {
        key: np.concatenate([np.atleast_1d(row[key]) for row in rows])
        for key in rows[0]
    }


def seed_prerequisite_gate(
    planned,
    cpu_aggregate,
    cpu_dense_aggregate,
    model_aggregate,
    dense_model_aggregate,
):
    """Fail closed unless every frozen coarse and dense seed gate passes."""
    return bool(
        planned["passes"]
        and cpu_aggregate["passes"]
        and cpu_dense_aggregate["passes"]
        and model_aggregate["passes"]
        and dense_model_aggregate["passes"]
    )


def seed_prerequisite(
    model,
    instance,
    seeds,
    controls,
    solver,
    initializer_kind="final_conditioning",
):
    planned = planned_seed_screen(
        model, instance["q0"], seeds, initializer_kind=initializer_kind
    )
    cpu_rows, cpu_aggregate = seed_kinematic_screen(model, instance, controls)
    cpu_dense_rows, cpu_dense_aggregate = cpu_dense_pin_seed_screen(
        model, instance, controls
    )
    preflight = model_pair_preflight(
        solver.sim_forward, model, instance["q0"], DEVELOPMENT_PROTOCOL.dt
    )
    x0 = np.hstack([instance["q0"], np.zeros(model.nv)]).astype(np.float32)
    cuda_states = cuda_rollout(
        solver.sim_forward, x0, controls, DEVELOPMENT_PROTOCOL.dt
    )
    pin_states = pinocchio_rollout(
        model, x0, controls, DEVELOPMENT_PROTOCOL.dt
    )
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
    passes = seed_prerequisite_gate(
        planned,
        cpu_aggregate,
        cpu_dense_aggregate,
        model_aggregate,
        dense_model_aggregate,
    )
    return {
        "passes": passes,
        "planned": planned,
        "cpu_rows": cpu_rows,
        "cpu_aggregate": cpu_aggregate,
        "cpu_dense4_rows": cpu_dense_rows,
        "cpu_dense4_aggregate": cpu_dense_aggregate,
        "preflight": preflight,
        "model_rows": model_rows,
        "model_aggregate": model_aggregate,
        "dense4_model_rows": dense_model_rows,
        "dense4_model_aggregate": dense_model_aggregate,
        "cuda_states": cuda_states,
        "pinocchio_states": pin_states,
        "dense4_cuda_states": dense_cuda_states,
        "dense4_pinocchio_states": dense_pin_states,
    }


def solve_exact_candidates(model, instance, seeds, serial_solver, batch_solver):
    x0 = np.hstack([instance["q0"], np.zeros(model.nv)]).astype(np.float32)
    reference = np.tile(
        instance["reference"], DEVELOPMENT_PROTOCOL.knots
    ).astype(np.float32)
    serial_outputs = []
    serial_stats = []
    serial_captures = []
    solve_call_audit = {
        "completed_b1_solve_once_calls": 0,
        "completed_b16_solve_once_calls": 0,
    }
    for candidate in range(BUDGET):
        capture = {}
        output, stats = solve_once(
            serial_solver,
            x0[None, :],
            reference[None, :],
            seeds[candidate : candidate + 1],
            DEVELOPMENT_PROTOCOL.max_sqp_iters,
            capture,
        )
        solve_call_audit["completed_b1_solve_once_calls"] += 1
        serial_outputs.append(output[0].copy())
        serial_stats.append(stats)
        serial_captures.append(capture)
    serial_outputs = np.asarray(serial_outputs)
    serial_stats = concatenate_stats(serial_stats)
    batch_x0 = np.tile(x0, (BUDGET, 1))
    batch_reference = np.tile(reference, (BUDGET, 1))
    batch_capture = {}
    batch_outputs, batch_stats = solve_once(
        batch_solver,
        batch_x0,
        batch_reference,
        seeds,
        DEVELOPMENT_PROTOCOL.max_sqp_iters,
        batch_capture,
    )
    solve_call_audit["completed_b16_solve_once_calls"] += 1
    return {
        "x0": x0,
        "reference": reference,
        "serial_outputs": serial_outputs,
        "serial_stats": serial_stats,
        "batch_outputs": batch_outputs.copy(),
        "batch_stats": batch_stats,
        "serial_captured_inputs": {
            key: np.stack([capture[key] for capture in serial_captures])
            for key in ("x0", "reference", "seeds")
        },
        "batch_captured_inputs": batch_capture,
        "solve_call_audit": solve_call_audit,
        "completed_solve_once_calls": sum(solve_call_audit.values()),
    }


def build_topology(rows, instance, model):
    cuda = mode_support_certificate(
        rows,
        instance["pillar_xy_m"],
        model.effortLimit,
        "cuda_dense_tool_path",
        "cuda_external_cost",
    )
    pinocchio = mode_support_certificate(
        rows,
        instance["pillar_xy_m"],
        model.effortLimit,
        "pinocchio_dense_tool_path",
        "pinocchio_external_cost",
    )
    return {
        "cuda": cuda,
        "pinocchio": pinocchio,
        "cuda_pinocchio_exact_agreement": mode_certificates_agree(
            cuda, pinocchio
        ),
        "two_mode_support_both_models": bool(
            cuda["two_mode_support"] and pinocchio["two_mode_support"]
        ),
    }


def run(output):
    if FINAL_PROTOCOL is not None:
        raise RuntimeError("development probe requires final protocol embargo")
    model = load_model()
    task_seed = DEVELOPMENT_TASK_SEEDS[0]
    method_seed = DEVELOPMENT_METHOD_SEEDS[0]
    instance = generate_instance(task_seed, model=model)
    seeds, controls, generation = generate_planned_seeds(
        model, instance["q0"], method_seed
    )
    serial_solver = make_solver(
        1, DEVELOPMENT_PROTOCOL.max_sqp_iters, DEVELOPMENT_PROTOCOL.pillar_cost
    )
    batch_solver = make_solver(
        BUDGET,
        DEVELOPMENT_PROTOCOL.max_sqp_iters,
        DEVELOPMENT_PROTOCOL.pillar_cost,
    )
    screen = seed_prerequisite(model, instance, seeds, controls, batch_solver)

    solve_result, solve_call_audit = execute_after_seed_gate(
        screen["passes"],
        lambda: solve_exact_candidates(
            model, instance, seeds, serial_solver, batch_solver
        ),
    )
    base = {
        "schema_version": 1,
        "diagnostic": "iiwa_multimodal_real_development_probe",
        "benchmark_evidence": False,
        "timing_evidence": False,
        "warmups": 0,
        "repeats": 1,
        "task_seed": task_seed,
        "method_seed": method_seed,
        "final_protocol_embargoed": FINAL_PROTOCOL is None,
        "ordered_candidate_count": BUDGET,
        "solve_call_audit": solve_call_audit,
        "solve_call_audit_passes": solve_call_audit_passes(solve_call_audit),
        "expected_completed_b1_calls_if_seed_passes": BUDGET,
        "expected_completed_b16_calls_if_seed_passes": 1,
        "seed_generation": generation,
        "instance": instance["metadata"],
        "seed_prerequisite": {
            key: value
            for key, value in screen.items()
            if key
            not in (
                "cuda_states",
                "pinocchio_states",
                "dense4_cuda_states",
                "dense4_pinocchio_states",
            )
        },
        "input_hashes": {
            "seeds": sha256_array(seeds),
            "seed_controls": sha256_array(controls),
            "reference": sha256_array(instance["reference"]),
            "common_x0": sha256_array(instance["q0"]),
        },
        "provenance": provenance(batch_solver, Path(__file__)),
    }
    if solve_result is None:
        base["all_gates_pass"] = False
        base["failure"] = "seed_prerequisite_failed_before_any_solve"
        write_artifacts(
            output,
            base,
            {
                "planned_seeds": seeds,
                "seed_controls": controls,
                "seed_cuda_rollout": screen["cuda_states"],
                "seed_pinocchio_rollout": screen["pinocchio_states"],
                "seed_cuda_dense4_rollout": screen["dense4_cuda_states"],
                "seed_pinocchio_dense4_rollout": screen[
                    "dense4_pinocchio_states"
                ],
            },
        )
        print(output)
        print("seed prerequisite failed; solver_call_count=0")
        return

    serial_rows, serial_arrays = certify_real_execution(
        model,
        instance,
        seeds,
        solve_result["serial_outputs"],
        solve_result["serial_stats"],
        batch_solver.sim_forward,
    )
    batch_rows, batch_arrays = certify_real_execution(
        model,
        instance,
        seeds,
        solve_result["batch_outputs"],
        solve_result["batch_stats"],
        batch_solver.sim_forward,
    )
    serial_topology = build_topology(serial_rows, instance, model)
    batch_topology = build_topology(batch_rows, instance, model)
    comparison_rows, comparison = compare_serial_batch(serial_rows, batch_rows)
    topology_execution_agreement = bool(
        mode_certificates_agree(
            serial_topology["cuda"], batch_topology["cuda"]
        )
        and mode_certificates_agree(
            serial_topology["pinocchio"], batch_topology["pinocchio"]
        )
    )
    all_cap_free = all_rows_cap_free(serial_rows + batch_rows)
    input_identity = captured_solve_input_identity(
        solve_result["serial_captured_inputs"],
        solve_result["batch_captured_inputs"],
    )
    base.update(
        {
            "execution_input_identity": input_identity,
            "serial_raw_stats": solve_result["serial_stats"],
            "batch_raw_stats": solve_result["batch_stats"],
            "serial_rows": serial_rows,
            "batch_rows": batch_rows,
            "serial_topology": serial_topology,
            "batch_topology": batch_topology,
            "serial_batch_comparison_rows": comparison_rows,
            "serial_batch_comparison": comparison,
            "serial_batch_exact_topology_agreement": topology_execution_agreement,
            "all_serial_and_batch_lanes_cap_free": all_cap_free,
            "all_gates_pass": bool(
                screen["passes"]
                and solve_call_audit_passes(solve_call_audit)
                and input_identity["passes"]
                and serial_topology["cuda_pinocchio_exact_agreement"]
                and batch_topology["cuda_pinocchio_exact_agreement"]
                and serial_topology["two_mode_support_both_models"]
                and batch_topology["two_mode_support_both_models"]
                and topology_execution_agreement
                and comparison["passes"]
                and all_cap_free
            ),
        }
    )
    arrays = {
        "planned_seeds": seeds,
        "seed_controls": controls,
        "serial_outputs": solve_result["serial_outputs"],
        "batch_outputs": solve_result["batch_outputs"],
        "seed_cuda_rollout": screen["cuda_states"],
        "seed_pinocchio_rollout": screen["pinocchio_states"],
        "seed_cuda_dense4_rollout": screen["dense4_cuda_states"],
        "seed_pinocchio_dense4_rollout": screen["dense4_pinocchio_states"],
    }
    for execution in ("serial", "batch"):
        captured = solve_result[f"{execution}_captured_inputs"]
        arrays.update(
            {
                f"{execution}_solve_input_{field}": captured[field]
                for field in ("x0", "reference", "seeds")
            }
        )
    for prefix, values in (("serial", serial_arrays), ("batch", batch_arrays)):
        arrays.update({f"{prefix}_{key}": value for key, value in values.items()})
    write_artifacts(output, base, arrays)
    print(output)
    print(
        json_safe(
            {
                "all_gates_pass": base["all_gates_pass"],
                "serial_support": serial_topology["cuda"]["support"],
                "batch_support": batch_topology["cuda"]["support"],
            }
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args().output)
