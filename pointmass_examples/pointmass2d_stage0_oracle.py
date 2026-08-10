#!/usr/bin/env python3
"""Quarantined Stage-0 feasibility oracle for the frozen point-mass tasks.

This task-aware direct-collocation oracle is never a benchmark initializer.
Execution is opt-in and remains subject to a separate verification ruling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import os
import time

import numpy as np
import scipy
from scipy import sparse
from scipy.optimize import LinearConstraint, NonlinearConstraint, minimize


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pointmass_src"))

from gato_pointmass.pointmass2d import (  # noqa: E402
    DEVELOPMENT_TASK_SEEDS,
    HELDOUT_TASK_SEEDS,
    PROTOCOL,
    affine_state_map,
    generate_task,
    minimum_energy_controls,
    objective_control_derivatives,
    rollout_controls,
    signed_turns,
    terminal_nullspace,
    terminal_target,
)


ORACLE_RESTART_MARGINS_M = (0.0, 0.015, 0.030, 0.045, 0.060)
MODE_SIDES = (-1, 1)
TRUST_CONSTR_OPTIONS = {
    "maxiter": 2000,
    "gtol": 1e-10,
    "xtol": 1e-12,
    "barrier_tol": 1e-12,
    "initial_constr_penalty": 1.0,
    "initial_barrier_parameter": 0.1,
    "initial_barrier_tolerance": 0.1,
    "sparse_jacobian": True,
    "verbose": 0,
}
FIRST_PAIR_WATCHDOG_SECONDS = 5.0 * 60.0
FULL_RUN_WATCHDOG_SECONDS = 6.0 * 60.0 * 60.0


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_json(value):
    payload = json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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


def restart_identity(task_seed, mode_side, restart_index):
    mode = "clockwise" if int(mode_side) > 0 else "counterclockwise"
    return {
        "key": f"task{int(task_seed)}_{mode}_restart{int(restart_index)}",
        "task_seed": int(task_seed),
        "mode_side": int(mode_side),
        "expected_mode": mode,
        "restart_index": int(restart_index),
    }


def expected_restart_identities():
    return [
        restart_identity(task_seed, mode_side, restart_index)
        for task_seed in DEVELOPMENT_TASK_SEEDS + HELDOUT_TASK_SEEDS
        for mode_side in MODE_SIDES
        for restart_index in range(len(ORACLE_RESTART_MARGINS_M))
    ]


def _atomic_write_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_json(path, value):
    payload = (
        json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(path, payload)


def _atomic_write_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def partial_latest_path(output):
    output = Path(output)
    return output.with_name(f"{output.stem}.partial.latest.json")


def partial_generation_paths(output, completed_count):
    output = Path(output)
    generation = f"{int(completed_count):04d}"
    return (
        output.with_name(f"{output.stem}.partial.{generation}.json"),
        output.with_name(f"{output.stem}.partial.{generation}.npz"),
    )


def write_partial_checkpoint(
    output,
    expected_identities,
    completed_records,
    arrays,
    identity_provenance,
    after_npz_published=None,
):
    """Publish an immutable generation, then atomically advance latest."""
    completed_identities = [record["identity"] for record in completed_records]
    completed_keys = [identity["key"] for identity in completed_identities]
    expected_keys = [identity["key"] for identity in expected_identities]
    if completed_keys != expected_keys[: len(completed_keys)]:
        raise RuntimeError("completed restart identities are not an ordered prefix")
    completed_count = len(completed_records)
    if completed_count <= 0:
        raise RuntimeError("checkpoint generation count must be positive")
    requested_final_artifacts_exist = {
        "json": Path(output).exists(),
        "npz": Path(output).with_suffix(".npz").exists(),
        "manifest": Path(output).with_suffix(".sha256.json").exists(),
    }
    if any(requested_final_artifacts_exist.values()):
        raise RuntimeError("final artifacts must not exist while checkpoint is incomplete")
    generation_json, generation_npz = partial_generation_paths(
        output, completed_count
    )
    if generation_json.exists() or generation_npz.exists():
        raise RuntimeError("checkpoint generations are immutable")
    _atomic_write_npz(generation_npz, arrays)
    if after_npz_published is not None:
        after_npz_published(generation_npz)
    checkpoint = {
        "schema_version": 1,
        "diagnostic": "pointmass2d_stage0_quarantined_oracle_checkpoint_generation",
        "oracle_only": True,
        "benchmark_seed_eligible": False,
        "incomplete": True,
        "all_stage0_gates_pass": False,
        "all_stage0_gates_status": "not_applicable_while_incomplete",
        "expected_restart_count": len(expected_identities),
        "expected_ordered_identities": expected_identities,
        "completed_restart_count": len(completed_records),
        "completed_ordered_identities": completed_identities,
        "pending_ordered_identities": expected_identities[len(completed_records) :],
        "completed_records": completed_records,
        "identity_provenance": identity_provenance,
        "checkpoint_generation": completed_count,
        "generation_json": str(generation_json),
        "generation_npz": str(generation_npz),
        "generation_npz_sha256": sha256_file(generation_npz),
        "requested_final_json": str(Path(output)),
        "requested_final_artifacts_exist": requested_final_artifacts_exist,
    }
    _atomic_write_json(generation_json, checkpoint)
    latest = {
        "schema_version": 1,
        "diagnostic": "pointmass2d_stage0_quarantined_oracle_checkpoint_pointer",
        "oracle_only": True,
        "benchmark_seed_eligible": False,
        "incomplete": True,
        "all_stage0_gates_pass": False,
        "all_stage0_gates_status": "not_applicable_while_incomplete",
        "latest_complete_generation": completed_count,
        "generation_json": str(generation_json),
        "generation_json_sha256": sha256_file(generation_json),
        "generation_npz": str(generation_npz),
        "generation_npz_sha256": sha256_file(generation_npz),
        "unreferenced_generation_files_are_orphaned_non_evidence": True,
    }
    _atomic_write_json(partial_latest_path(output), latest)
    return checkpoint, latest


def execute_ordered_pairs(expected_identities, execute_pair, on_completed):
    """Execute an immutable order; an in-flight exception is never checkpointed."""
    for identity in expected_identities:
        payload = execute_pair(identity)
        on_completed(identity, payload)


class OracleRuntimeWatchdogStop(RuntimeError):
    pass


def watchdog_assessment(first_pair_seconds, elapsed_seconds, completed_count):
    completed_count = int(completed_count)
    projected = (
        None
        if completed_count <= 0
        else float(elapsed_seconds) * 80.0 / completed_count
    )
    first_exceeded = bool(
        first_pair_seconds is not None
        and float(first_pair_seconds) > FIRST_PAIR_WATCHDOG_SECONDS
    )
    projected_exceeded = bool(
        projected is not None and projected > FULL_RUN_WATCHDOG_SECONDS
    )
    reasons = []
    if first_exceeded:
        reasons.append("first_pair_exceeded_5_minutes")
    if projected_exceeded:
        reasons.append("projected_full_run_exceeded_6_hours")
    return {
        "stop": bool(reasons),
        "reasons": reasons,
        "first_pair_seconds": (
            None if first_pair_seconds is None else float(first_pair_seconds)
        ),
        "elapsed_seconds": float(elapsed_seconds),
        "completed_restart_count": completed_count,
        "projected_full_run_seconds": projected,
        "first_pair_limit_seconds": FIRST_PAIR_WATCHDOG_SECONDS,
        "full_run_limit_seconds": FULL_RUN_WATCHDOG_SECONDS,
    }


def write_runtime_rejection(output, assessment, identity_provenance):
    output = Path(output)
    rejection = output.with_name(f"{output.stem}.runtime_rejected.json")
    latest = partial_latest_path(output)
    payload = {
        "schema_version": 1,
        "diagnostic": "pointmass2d_stage0_runtime_watchdog_rejection",
        "oracle_only": True,
        "benchmark_seed_eligible": False,
        "runtime_rejected": True,
        "incomplete": True,
        "all_stage0_gates_pass": False,
        "all_stage0_gates_status": "not_applicable_runtime_rejected",
        "watchdog": assessment,
        "latest_complete_checkpoint_pointer": (
            str(latest) if latest.exists() else None
        ),
        "identity_provenance": identity_provenance,
        "automatic_resume_or_rerun_authorized": False,
    }
    _atomic_write_json(rejection, payload)
    return rejection


def _first_pair_alarm_handler(_signal_number, _frame):
    raise OracleRuntimeWatchdogStop("first pair exceeded 5-minute wall watchdog")


class Stage0Problem:
    """Exact terminal-eliminated oracle problem for one frozen task."""

    def __init__(self, task):
        self.task = task
        self.protocol = PROTOCOL
        self.intervals = self.protocol.knots - 1
        self.control_width = 2 * self.intervals
        terminal_offset, self.terminal_map = (
            affine_state_map(task, self.protocol, substeps=1)[0][-1],
            affine_state_map(task, self.protocol, substeps=1)[1][-1],
        )
        _, self.nullspace, self.terminal_singular_values = terminal_nullspace(
            task, self.protocol
        )
        terminal_rhs = terminal_target(task, 1.0) - terminal_offset
        self.particular = self.terminal_map.T @ np.linalg.solve(
            self.terminal_map @ self.terminal_map.T, terminal_rhs
        )
        self.dense_offset, self.dense_map = affine_state_map(
            task, self.protocol, substeps=self.protocol.dense_substeps
        )
        self.dense_particular = self.dense_offset + np.einsum(
            "sij,j->si", self.dense_map, self.particular
        )
        self.dense_z_map = np.einsum(
            "sij,jk->sik", self.dense_map, self.nullspace
        )
        self.original_linear = self._linear_constraint(anchor_side=None)

    @property
    def variable_count(self):
        return self.nullspace.shape[1]

    def controls(self, z):
        return (self.particular + self.nullspace @ np.asarray(z)).reshape(
            self.intervals, 2
        )

    def dense_states(self, z):
        return self.dense_particular + np.einsum(
            "sik,k->si", self.dense_z_map, np.asarray(z)
        )

    def objective(self, z):
        value, _, _, _ = objective_control_derivatives(
            self.task, self.controls(z), self.protocol
        )
        return value

    def objective_jacobian(self, z):
        _, gradient, _, _ = objective_control_derivatives(
            self.task, self.controls(z), self.protocol
        )
        return self.nullspace.T @ gradient

    def objective_hessian(self, z):
        return sparse.csc_matrix(self.objective_hessian_dense(z))

    def objective_hessian_dense(self, z):
        _, _, hessian, _ = objective_control_derivatives(
            self.task, self.controls(z), self.protocol
        )
        return self.nullspace.T @ hessian @ self.nullspace

    def _linear_constraint_dense_components(self, anchor_side):
        control_matrix = self.nullspace
        control_offset = self.particular
        state_matrix = self.dense_z_map.reshape(-1, self.variable_count)
        state_offset = self.dense_particular.reshape(-1)
        matrices = [control_matrix, state_matrix]
        lower = [
            np.full(self.control_width, -self.protocol.acceleration_limit)
            - control_offset,
            np.tile(
                [
                    -self.protocol.position_limit,
                    -self.protocol.position_limit,
                    -self.protocol.velocity_limit,
                    -self.protocol.velocity_limit,
                ],
                len(self.dense_particular),
            )
            - state_offset,
        ]
        upper = [
            np.full(self.control_width, self.protocol.acceleration_limit)
            - control_offset,
            np.tile(
                [
                    self.protocol.position_limit,
                    self.protocol.position_limit,
                    self.protocol.velocity_limit,
                    self.protocol.velocity_limit,
                ],
                len(self.dense_particular),
            )
            - state_offset,
        ]
        if anchor_side is not None:
            midpoint = (self.protocol.knots - 1) * self.protocol.dense_substeps // 2
            side = float(anchor_side)
            normal = np.asarray(self.task["normal"], dtype=np.float64)
            center = np.asarray(self.task["disk_center_m"], dtype=np.float64)
            required = float(self.task["required_distance_m"])
            row = side * normal @ self.dense_z_map[midpoint, :2]
            offset = side * normal @ (
                self.dense_particular[midpoint, :2] - center
            )
            matrices.append(row[None, :])
            lower.append(np.asarray([required - offset]))
            upper.append(np.asarray([np.inf]))
        return np.vstack(matrices), np.concatenate(lower), np.concatenate(upper)

    def _linear_constraint(self, anchor_side):
        matrix, lower, upper = self._linear_constraint_dense_components(anchor_side)
        return LinearConstraint(sparse.csr_matrix(matrix), lower, upper)

    def clearance(self, z):
        positions = self.dense_states(z)[:, :2]
        delta = positions - np.asarray(self.task["disk_center_m"])[None, :]
        required = float(self.task["required_distance_m"])
        return np.sum(delta * delta, axis=1) - required * required

    def clearance_jacobian(self, z):
        return sparse.csr_matrix(self.clearance_jacobian_dense(z))

    def clearance_jacobian_dense(self, z):
        positions = self.dense_states(z)[:, :2]
        delta = positions - np.asarray(self.task["disk_center_m"])[None, :]
        return 2.0 * np.einsum(
            "si,sik->sk", delta, self.dense_z_map[:, :2]
        )

    def clearance_hessian(self, z, multipliers):
        return sparse.csc_matrix(
            self.clearance_hessian_dense(z, multipliers)
        )

    def clearance_hessian_dense(self, z, multipliers):
        del z
        return 2.0 * np.einsum(
            "s,sia,sib->ab",
            np.asarray(multipliers),
            self.dense_z_map[:, :2],
            self.dense_z_map[:, :2],
        )

    def clearance_constraint(self):
        return NonlinearConstraint(
            self.clearance,
            0.0,
            np.inf,
            jac=self.clearance_jacobian,
            hess=self.clearance_hessian,
        )

    def initial_z(self, mode_side, restart_index):
        side = float(mode_side)
        intervals = self.intervals
        total_time = intervals * self.protocol.dt
        tau = (np.arange(intervals) + 0.5) / intervals
        displacement = (
            np.asarray(self.task["goal_position_m"])
            - np.asarray(self.task["start_position_m"])
        )
        smoothstep_second = 60.0 * tau - 180.0 * tau**2 + 120.0 * tau**3
        amplitude = (
            self.task["disk_radius_m"]
            + self.protocol.clearance_margin
            + side * self.task["offset_sign"] * self.task["disk_offset_m"]
            + ORACLE_RESTART_MARGINS_M[int(restart_index)]
        )
        acceleration = (
            smoothstep_second[:, None] * displacement[None, :] / total_time**2
            + side
            * amplitude
            * (2.0 * np.pi**2 / total_time**2)
            * np.cos(2.0 * np.pi * tau)[:, None]
            * np.asarray(self.task["normal"])[None, :]
        )
        raw = acceleration.reshape(-1)
        projected = self.particular + self.nullspace @ (
            self.nullspace.T @ (raw - self.particular)
        )
        return self.nullspace.T @ (projected - self.particular)

    def independent_kkt(self, result, linear_constraint):
        """Recompute KKT residuals from exact functions and raw multipliers.

        SciPy's interval multiplier convention is positive at an upper bound
        and negative at a lower bound. The retained split duals make sign and
        complementarity independently reproducible.
        """
        if len(result.v) != 2:
            raise RuntimeError("trust-constr result must have two multiplier groups")
        gradient = self.objective_jacobian(result.x)
        linear_multiplier = np.asarray(result.v[0], dtype=np.float64)
        clearance_multiplier = np.asarray(result.v[1], dtype=np.float64)
        stationarity_vector = (
            gradient
            + linear_constraint.A.T @ linear_multiplier
            + self.clearance_jacobian_dense(result.x).T
            @ clearance_multiplier
        )
        linear_value = linear_constraint.A @ np.asarray(result.x)
        finite_lower = np.isfinite(linear_constraint.lb)
        finite_upper = np.isfinite(linear_constraint.ub)
        linear_lower_slack = np.where(
            finite_lower, linear_value - linear_constraint.lb, 0.0
        )
        linear_upper_slack = np.where(
            finite_upper, linear_constraint.ub - linear_value, 0.0
        )
        linear_lower_dual = np.maximum(-linear_multiplier, 0.0)
        linear_upper_dual = np.maximum(linear_multiplier, 0.0)
        linear_dual_sign_residual = np.zeros_like(linear_multiplier)
        lower_only = finite_lower & ~finite_upper
        upper_only = ~finite_lower & finite_upper
        linear_dual_sign_residual[lower_only] = np.maximum(
            linear_multiplier[lower_only], 0.0
        )
        linear_dual_sign_residual[upper_only] = np.maximum(
            -linear_multiplier[upper_only], 0.0
        )
        linear_lower_complementarity = np.where(
            finite_lower, linear_lower_dual * linear_lower_slack, 0.0
        )
        linear_upper_complementarity = np.where(
            finite_upper, linear_upper_dual * linear_upper_slack, 0.0
        )

        clearance_value = self.clearance(result.x)
        clearance_lower_slack = clearance_value.copy()
        clearance_lower_dual = np.maximum(-clearance_multiplier, 0.0)
        clearance_dual_sign_residual = np.maximum(clearance_multiplier, 0.0)
        clearance_complementarity = (
            clearance_lower_dual * clearance_lower_slack
        )
        dual_sign_vector = np.concatenate(
            [linear_dual_sign_residual, clearance_dual_sign_residual]
        )
        complementarity_vector = np.concatenate(
            [
                linear_lower_complementarity,
                linear_upper_complementarity,
                clearance_complementarity,
            ]
        )
        return {
            "objective_gradient": gradient,
            "linear_multiplier": linear_multiplier,
            "clearance_multiplier": clearance_multiplier,
            "linear_value": linear_value,
            "linear_lower_finite": finite_lower,
            "linear_upper_finite": finite_upper,
            "linear_lower_slack": linear_lower_slack,
            "linear_upper_slack": linear_upper_slack,
            "linear_lower_dual": linear_lower_dual,
            "linear_upper_dual": linear_upper_dual,
            "clearance_lower_slack": clearance_lower_slack,
            "clearance_lower_dual": clearance_lower_dual,
            "stationarity_vector": stationarity_vector,
            "dual_sign_residual_vector": dual_sign_vector,
            "complementarity_residual_vector": complementarity_vector,
            "stationarity_inf": float(np.max(np.abs(stationarity_vector))),
            "dual_sign_inf": float(np.max(np.abs(dual_sign_vector))),
            "complementarity_inf": float(
                np.max(np.abs(complementarity_vector))
            ),
        }

    def independent_violation(self, z, linear_constraint=None):
        linear = (
            self.original_linear
            if linear_constraint is None
            else linear_constraint
        )
        linear_value = linear.A @ np.asarray(z)
        lower = np.max(linear.lb - linear_value, initial=0.0)
        upper = np.max(linear_value - linear.ub, initial=0.0)
        clearance = np.max(-self.clearance(z), initial=0.0)
        return float(max(lower, upper, clearance, 0.0))


def solve_phase(problem, initial_z, anchor_side):
    linear = (
        problem.original_linear
        if anchor_side is None
        else problem._linear_constraint(anchor_side)
    )
    return minimize(
        problem.objective,
        np.asarray(initial_z, dtype=np.float64),
        method="trust-constr",
        jac=problem.objective_jacobian,
        hess=problem.objective_hessian,
        constraints=(linear, problem.clearance_constraint()),
        options=TRUST_CONSTR_OPTIONS,
    )


def phase_metrics(problem, result, linear_constraint):
    controls = problem.controls(result.x)
    dense = problem.dense_states(result.x)
    position = dense[:, :2]
    velocity = dense[:, 2:]
    center = np.asarray(problem.task["disk_center_m"])
    physical_clearance = float(
        np.min(np.linalg.norm(position - center[None, :], axis=1))
        - problem.task["disk_radius_m"]
    )
    label, turns = signed_turns(position, center)
    terminal = terminal_target(problem.task, 1.0)
    kkt = problem.independent_kkt(result, linear_constraint)
    def optional_int(value):
        return None if value is None else int(value)

    return {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "nit": optional_int(result.nit),
        "nfev": optional_int(result.nfev),
        "njev": optional_int(result.njev),
        "nhev": optional_int(result.nhev),
        "reported_optimality": float(result.optimality),
        "reported_constraint_violation": float(result.constr_violation),
        "objective": float(problem.objective(result.x)),
        "terminal_position_error_m": float(
            np.linalg.norm(dense[-1, :2] - terminal[:2])
        ),
        "terminal_velocity_error_m_s": float(
            np.linalg.norm(dense[-1, 2:] - terminal[2:])
        ),
        "minimum_physical_clearance_m": physical_clearance,
        "maximum_position_component_m": float(np.max(np.abs(position))),
        "maximum_velocity_component_m_s": float(np.max(np.abs(velocity))),
        "maximum_control_component_m_s2": float(np.max(np.abs(controls))),
        "mode": label,
        "signed_turns": turns,
        "finite": bool(np.all(np.isfinite(controls)) and np.all(np.isfinite(dense))),
        "independent_kkt": kkt,
        "z": np.asarray(result.x),
        "controls": controls,
        "dense_states": dense,
    }


def solve_restart(problem, mode_side, restart_index):
    acquisition_linear = problem._linear_constraint(mode_side)
    acquisition_initial_z = problem.initial_z(mode_side, restart_index)
    acquisition = solve_phase(
        problem, acquisition_initial_z, mode_side
    )
    acquisition_metrics = phase_metrics(problem, acquisition, acquisition_linear)
    acquisition_metrics["anchor_multiplier_report_only"] = acquisition_metrics[
        "independent_kkt"
    ]["linear_multiplier"][-1]
    acquisition_metrics["anchor_lower_slack_report_only_m"] = acquisition_metrics[
        "independent_kkt"
    ]["linear_lower_slack"][-1]
    polished = solve_phase(problem, acquisition.x, anchor_side=None)
    polished_metrics = phase_metrics(problem, polished, problem.original_linear)
    polished_kkt = polished_metrics["independent_kkt"]
    violation = problem.independent_violation(polished.x)
    acquisition_violation = problem.independent_violation(
        acquisition.x, acquisition_linear
    )
    path_rms = float(
        np.sqrt(
            np.mean(
                np.sum(
                    (
                        polished_metrics["dense_states"][:, :2]
                        - acquisition_metrics["dense_states"][:, :2]
                    )
                    ** 2,
                    axis=1,
                )
            )
        )
    )
    cost_change_fraction = abs(
        polished_metrics["objective"] - acquisition_metrics["objective"]
    ) / max(abs(acquisition_metrics["objective"]), 1e-12)
    expected_mode = "clockwise" if mode_side > 0 else "counterclockwise"
    acquisition_converged = bool(
        acquisition_metrics["success"]
        and acquisition_metrics["finite"]
        and acquisition_violation <= 1e-6
        and acquisition_metrics["mode"] == expected_mode
    )
    unanchored_polish_converged = bool(
        polished_metrics["success"]
        and polished_metrics["finite"]
        and violation <= 1e-6
        and polished_kkt["stationarity_inf"] <= 1e-5
        and polished_kkt["dual_sign_inf"] <= 1e-5
        and polished_kkt["complementarity_inf"] <= 1e-5
    )
    passes = bool(
        acquisition_converged
        and unanchored_polish_converged
        and polished_metrics["terminal_position_error_m"] <= 0.01
        and polished_metrics["terminal_velocity_error_m_s"] <= 0.02
        and polished_metrics["minimum_physical_clearance_m"] >= 0.02
        and polished_metrics["maximum_position_component_m"]
        <= PROTOCOL.position_limit
        and polished_metrics["maximum_velocity_component_m_s"]
        <= PROTOCOL.velocity_limit
        and polished_metrics["maximum_control_component_m_s2"]
        <= PROTOCOL.acceleration_limit
        and acquisition_metrics["mode"] == expected_mode
        and polished_metrics["mode"] == expected_mode
        and cost_change_fraction <= 0.01
        and path_rms <= 0.01
    )
    return {
        "mode_side": int(mode_side),
        "expected_mode": expected_mode,
        "restart_index": int(restart_index),
        "restart_margin_m": ORACLE_RESTART_MARGINS_M[int(restart_index)],
        "anchor_used_for_acquisition_only": True,
        "anchor_present_in_certifying_polish": False,
        "acquisition_initial_z": acquisition_initial_z,
        "unanchored_polish_initial_z": np.asarray(acquisition.x),
        "acquisition_converged": acquisition_converged,
        "unanchored_polish_converged": unanchored_polish_converged,
        "independent_acquisition_violation": acquisition_violation,
        "acquisition": acquisition_metrics,
        "unanchored_polish": polished_metrics,
        "independent_original_task_stationarity_inf": polished_kkt[
            "stationarity_inf"
        ],
        "independent_original_task_dual_sign_inf": polished_kkt[
            "dual_sign_inf"
        ],
        "independent_original_task_complementarity_inf": polished_kkt[
            "complementarity_inf"
        ],
        "independent_original_task_violation": violation,
        "acquisition_to_polish_cost_change_fraction": cost_change_fraction,
        "acquisition_to_polish_dense_path_rms_m": path_rms,
        "passes": passes,
    }


def task_aggregate(task, rows):
    by_mode = {
        mode: [row for row in rows if row["expected_mode"] == mode]
        for mode in ("clockwise", "counterclockwise")
    }
    exact_row_identities = bool(
        len(rows) == 10
        and all(len(by_mode[mode]) == 5 for mode in by_mode)
        and all(
            sorted(row["restart_index"] for row in by_mode[mode])
            == list(range(5))
            for mode in by_mode
        )
    )
    mode_costs = {
        mode: [row["unanchored_polish"]["objective"] for row in values]
        for mode, values in by_mode.items()
    }
    spreads = {
        mode: (
            (max(values) - min(values)) / max(abs(np.mean(values)), 1e-12)
            if len(values) == 5
            else np.inf
        )
        for mode, values in mode_costs.items()
    }
    predicted_worse = task["predicted_worse_mode"]
    better = (
        "counterclockwise" if predicted_worse == "clockwise" else "clockwise"
    )
    worse_cost = float(np.mean(mode_costs[predicted_worse]))
    better_cost = float(np.mean(mode_costs[better]))
    gap = worse_cost - better_cost
    gap_fraction = gap / max(abs(better_cost), 1e-12)
    clockwise_turns = [
        row["unanchored_polish"]["signed_turns"]
        for row in by_mode["clockwise"]
    ]
    counterclockwise_turns = [
        row["unanchored_polish"]["signed_turns"]
        for row in by_mode["counterclockwise"]
    ]
    cross_pair_classes = {}
    for clockwise in by_mode["clockwise"]:
        for counterclockwise in by_mode["counterclockwise"]:
            delta = (
                counterclockwise["unanchored_polish"]["signed_turns"]
                - clockwise["unanchored_polish"]["signed_turns"]
            )
            nearest = int(np.rint(delta))
            key = (
                f"clockwise_{clockwise['restart_index']}__"
                f"counterclockwise_{counterclockwise['restart_index']}"
            )
            cross_pair_classes[key] = {
                "delta_turns": float(delta),
                "nearest_integer": nearest,
                "nearest_integer_residual": float(abs(delta - nearest)),
                "absolute_class_one": bool(abs(nearest) == 1),
            }
    same_pair_classes = {}
    same_pair_count_by_mode = {}
    for mode, mode_rows in by_mode.items():
        before = len(same_pair_classes)
        for first_index, first in enumerate(mode_rows):
            for second in mode_rows[first_index + 1 :]:
                delta = (
                    second["unanchored_polish"]["signed_turns"]
                    - first["unanchored_polish"]["signed_turns"]
                )
                nearest = int(np.rint(delta))
                key = (
                    f"{mode}_{first['restart_index']}__"
                    f"{mode}_{second['restart_index']}"
                )
                same_pair_classes[key] = {
                    "delta_turns": float(delta),
                    "nearest_integer": nearest,
                    "nearest_integer_residual": float(abs(delta - nearest)),
                    "class_zero": bool(nearest == 0),
                }
        same_pair_count_by_mode[mode] = len(same_pair_classes) - before
    actual_mode_thresholds_pass = bool(
        len(by_mode["clockwise"]) == 5
        and len(by_mode["counterclockwise"]) == 5
        and all(
            row["unanchored_polish"]["mode"] == "clockwise"
            and row["unanchored_polish"]["signed_turns"] <= -0.25
            for row in by_mode["clockwise"]
        )
        and all(
            row["unanchored_polish"]["mode"] == "counterclockwise"
            and row["unanchored_polish"]["signed_turns"] >= 0.25
            for row in by_mode["counterclockwise"]
        )
    )
    same_pair_gate = bool(
        same_pair_count_by_mode
        == {"clockwise": 10, "counterclockwise": 10}
        and all(
            pair["class_zero"]
            and pair["nearest_integer_residual"] <= 0.10
            for pair in same_pair_classes.values()
        )
    )
    cross_pair_gate = bool(
        len(cross_pair_classes) == 25
        and all(
            pair["absolute_class_one"]
            and pair["nearest_integer_residual"] <= 0.10
            for pair in cross_pair_classes.values()
        )
    )
    clockwise_turn = np.mean(clockwise_turns)
    counterclockwise_turn = np.mean(counterclockwise_turns)
    all_acquisitions = bool(
        exact_row_identities and all(
            row["acquisition_converged"] for row in rows
        )
    )
    all_polishes = bool(
        exact_row_identities and all(
            row["unanchored_polish_converged"] for row in rows
        )
    )
    all_restarts = bool(
        exact_row_identities and all(row["passes"] for row in rows)
    )
    aggregate = {
        "all_ten_restarts_retained": len(rows) == 10,
        "exact_five_restarts_per_mode_with_unique_identities": exact_row_identities,
        "all_acquisitions_converged": all_acquisitions,
        "all_unanchored_polishes_converged": all_polishes,
        "all_restarts_pass": all_restarts,
        "within_mode_cost_spread_fraction": spreads,
        "all_within_mode_cost_spreads_at_most_1_percent": all(
            value <= 0.01 for value in spreads.values()
        ),
        "predicted_worse_mode": predicted_worse,
        "better_mode": better,
        "worse_mode_mean_cost": worse_cost,
        "better_mode_mean_cost": better_cost,
        "worse_mode_absolute_gap": gap,
        "worse_mode_gap_fraction": gap_fraction,
        "predicted_worse_cost_strictly_greater": bool(gap > 0.0),
        "worse_mode_gap_at_least_5_percent_and_0_005": bool(
            gap_fraction >= 0.05 and gap >= 0.005
        ),
        "mean_clockwise_turns": float(clockwise_turn),
        "mean_counterclockwise_turns": float(counterclockwise_turn),
        "actual_mode_thresholds_pass": actual_mode_thresholds_pass,
        "same_mode_pair_classes": same_pair_classes,
        "same_mode_pair_count_by_mode": same_pair_count_by_mode,
        "all_10_same_mode_pairs_per_mode_retained": same_pair_count_by_mode
        == {"clockwise": 10, "counterclockwise": 10},
        "same_mode_class_zero_with_integer_residual_at_most_0_10": same_pair_gate,
        "cross_mode_pair_classes": cross_pair_classes,
        "all_25_cross_mode_pairs_retained": len(cross_pair_classes) == 25,
        "cross_mode_class_one": cross_pair_gate,
    }
    aggregate["all_required_task_gates_pass"] = bool(
        aggregate["all_ten_restarts_retained"]
        and aggregate["exact_five_restarts_per_mode_with_unique_identities"]
        and aggregate["all_acquisitions_converged"]
        and aggregate["all_unanchored_polishes_converged"]
        and aggregate["all_restarts_pass"]
        and aggregate["all_within_mode_cost_spreads_at_most_1_percent"]
        and aggregate["actual_mode_thresholds_pass"]
        and aggregate[
            "same_mode_class_zero_with_integer_residual_at_most_0_10"
        ]
        and aggregate["cross_mode_class_one"]
        and aggregate["predicted_worse_cost_strictly_greater"]
    )
    return aggregate


def run_stage0(output):
    output = Path(output)
    final_npz = output.with_suffix(".npz")
    final_manifest = output.with_suffix(".sha256.json")
    runtime_rejection = output.with_name(f"{output.stem}.runtime_rejected.json")
    latest_checkpoint = partial_latest_path(output)
    existing_checkpoint_files = list(
        output.parent.glob(f"{output.stem}.partial.*")
    )
    forbidden_existing = [
        path
        for path in (output, final_npz, final_manifest, runtime_rejection)
        if path.exists()
    ] + existing_checkpoint_files
    if forbidden_existing:
        raise RuntimeError(
            "Stage0 is single-run only; refusing existing artifacts: "
            + ", ".join(str(path) for path in forbidden_existing)
        )
    source_paths = {
        "oracle": Path(__file__).resolve(),
        "task_schema": REPO_ROOT / "pointmass_src/gato_pointmass/pointmass2d.py",
        "tests": REPO_ROOT / "tests/python/test_pointmass2d_stage0.py",
    }
    source_hashes_at_start = {
        name: sha256_file(path) for name, path in source_paths.items()
    }
    task_seeds = DEVELOPMENT_TASK_SEEDS + HELDOUT_TASK_SEEDS
    tasks = {task_seed: generate_task(task_seed) for task_seed in task_seeds}
    expected_identities = expected_restart_identities()
    if len(expected_identities) != 80:
        raise RuntimeError("frozen Stage0 must contain exactly 80 restart pairs")
    full_status_at_start = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    identity_provenance = {
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "git_status_porcelain_full_at_start": full_status_at_start,
        "protocol_sha256": sha256_json(PROTOCOL.__dict__),
        "source_hashes_at_start": source_hashes_at_start,
        "task_identity_sha256_by_seed": {
            str(task_seed): tasks[task_seed]["solver_task_identity_sha256"]
            for task_seed in task_seeds
        },
        "exact_command_argv": list(sys.argv),
        "exact_command_shell": shlex.join(sys.argv),
    }
    arrays = {}
    for task_seed, task in tasks.items():
        arrays[f"task{task_seed}_solver_x0_float32"] = task["x0"]
        arrays[f"task{task_seed}_solver_reference_float32"] = task["reference"]
    problems = {}
    completed_records = []
    run_started = time.monotonic()
    first_pair_seconds = None

    def execute_pair(identity):
        nonlocal first_pair_seconds
        task_seed = identity["task_seed"]
        if task_seed not in problems:
            problems[task_seed] = Stage0Problem(tasks[task_seed])
        pair_started = time.monotonic()
        first_pair = len(completed_records) == 0
        previous_handler = None
        previous_timer = None
        if first_pair:
            previous_handler = signal.signal(
                signal.SIGALRM, _first_pair_alarm_handler
            )
            previous_timer = signal.setitimer(
                signal.ITIMER_REAL, FIRST_PAIR_WATCHDOG_SECONDS
            )
        try:
            row = solve_restart(
                problems[task_seed],
                identity["mode_side"],
                identity["restart_index"],
            )
        except OracleRuntimeWatchdogStop:
            elapsed = time.monotonic() - run_started
            first_pair_seconds = max(
                time.monotonic() - pair_started,
                np.nextafter(FIRST_PAIR_WATCHDOG_SECONDS, np.inf),
            )
            assessment = watchdog_assessment(first_pair_seconds, elapsed, 0)
            write_runtime_rejection(output, assessment, identity_provenance)
            raise
        finally:
            if first_pair:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                signal.signal(signal.SIGALRM, previous_handler)
                if previous_timer and previous_timer[0] > 0.0:
                    signal.setitimer(
                        signal.ITIMER_REAL, previous_timer[0], previous_timer[1]
                    )
        pair_seconds = time.monotonic() - pair_started
        if first_pair:
            first_pair_seconds = pair_seconds
        row["runtime_pair_wall_seconds"] = pair_seconds
        return row

    def retain_completed_pair(identity, row):
        if (
            row["expected_mode"] != identity["expected_mode"]
            or row["restart_index"] != identity["restart_index"]
            or row["mode_side"] != identity["mode_side"]
        ):
            raise RuntimeError("solver row identity differs from frozen order")
        assessment = watchdog_assessment(
            first_pair_seconds,
            time.monotonic() - run_started,
            len(completed_records) + 1,
        )
        row["runtime_watchdog_assessment_after_pair"] = assessment
        prefix = identity["key"]
        arrays[f"{prefix}_acquisition_initial_z"] = row[
            "acquisition_initial_z"
        ]
        arrays[f"{prefix}_unanchored_polish_initial_z"] = row[
            "unanchored_polish_initial_z"
        ]
        for phase in ("acquisition", "unanchored_polish"):
            arrays[f"{prefix}_{phase}_z"] = row[phase]["z"]
            arrays[f"{prefix}_{phase}_controls"] = row[phase]["controls"]
            arrays[f"{prefix}_{phase}_dense_states"] = row[phase][
                "dense_states"
            ]
            for name, value in row[phase]["independent_kkt"].items():
                if isinstance(value, np.ndarray):
                    arrays[f"{prefix}_{phase}_kkt_{name}"] = value
        completed_records.append(
            {"identity": identity, "task": tasks[identity["task_seed"]], "row": row}
        )
        current_provenance = dict(identity_provenance)
        current_provenance["source_hashes_current"] = {
            name: sha256_file(path) for name, path in source_paths.items()
        }
        write_partial_checkpoint(
            output,
            expected_identities,
            completed_records,
            arrays,
            current_provenance,
        )
        if assessment["stop"]:
            write_runtime_rejection(output, assessment, current_provenance)
            raise OracleRuntimeWatchdogStop(
                "Stage0 runtime watchdog stopped the single authorized run"
            )

    execute_ordered_pairs(expected_identities, execute_pair, retain_completed_pair)
    if len(completed_records) != 80:
        raise RuntimeError("final Stage0 publication requires exactly 80 pairs")
    latest = json.loads(latest_checkpoint.read_text())
    if latest["latest_complete_generation"] != 80:
        raise RuntimeError("latest checkpoint must reference complete generation 80")
    latest_generation_json = Path(latest["generation_json"])
    latest_generation_npz = Path(latest["generation_npz"])
    if (
        sha256_file(latest_generation_json) != latest["generation_json_sha256"]
        or sha256_file(latest_generation_npz) != latest["generation_npz_sha256"]
    ):
        raise RuntimeError("latest checkpoint pointer hash mismatch")

    task_rows = []
    for task_seed in task_seeds:
        rows = [
            record["row"]
            for record in completed_records
            if record["identity"]["task_seed"] == task_seed
        ]
        task = tasks[task_seed]
        aggregate = task_aggregate(task, rows)
        task_rows.append(
            {"task_seed": task_seed, "task": task, "rows": rows, "aggregate": aggregate}
        )
    heldout = [row for row in task_rows if row["task_seed"] in HELDOUT_TASK_SEEDS]
    source_hashes_at_end = {
        name: sha256_file(path) for name, path in source_paths.items()
    }
    exact_task_seed_set = bool(
        len(task_rows) == 8
        and tuple(row["task_seed"] for row in task_rows) == task_seeds
    )
    canonical_task_identities = bool(
        len({row["task"]["solver_task_identity_sha256"] for row in task_rows})
        == 8
        and all(
            row["task"]["canonical_numeric_source"]
            == "exact_solver_bound_float32_bytes"
            for row in task_rows
        )
    )
    source_hashes_stable = source_hashes_at_start == source_hashes_at_end
    predicted_worse_never_reverses = all(
        row["aggregate"]["predicted_worse_cost_strictly_greater"]
        for row in task_rows
    )
    heldout_gap_count = sum(
            row["aggregate"]["worse_mode_gap_at_least_5_percent_and_0_005"]
            for row in heldout
        )
    all_gates = bool(
        exact_task_seed_set
        and canonical_task_identities
        and source_hashes_stable
        and all(
            row["aggregate"]["all_required_task_gates_pass"]
            for row in task_rows
        )
        and predicted_worse_never_reverses
        and len(heldout) == 5
        and heldout_gap_count >= 4
    )
    full_status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    summary = {
        "schema_version": 1,
        "diagnostic": "pointmass2d_stage0_quarantined_oracle",
        "incomplete": False,
        "oracle_only": True,
        "benchmark_seed_eligible": False,
        "gato_sqp_cuda_or_build_calls": 0,
        "task_seeds": task_seeds,
        "trust_constr_options": TRUST_CONSTR_OPTIONS,
        "restart_margins_m": ORACLE_RESTART_MARGINS_M,
        "task_rows": task_rows,
        "expected_restart_count": 80,
        "expected_ordered_identities": expected_identities,
        "completed_restart_count": len(completed_records),
        "completed_ordered_identities": [
            record["identity"] for record in completed_records
        ],
        "pending_ordered_identities": [],
        "exact_eight_task_seed_identities": exact_task_seed_set,
        "canonical_solver_float32_task_identities": canonical_task_identities,
        "predicted_worse_cost_greater_on_all_eight_tasks": (
            predicted_worse_never_reverses
        ),
        "heldout_task_count": len(heldout),
        "heldout_tasks_with_predeclared_cost_gap": int(heldout_gap_count),
        "source_hashes_stable_during_execution": source_hashes_stable,
        "all_stage0_gates_pass": all_gates,
        "provenance": {
            "git_head": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
                capture_output=True, text=True,
            ).stdout.strip(),
            "git_tracked_dirty": any(
                not line.startswith("??") for line in full_status
            ),
            "git_status_porcelain_full": full_status,
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "exact_command_argv": list(sys.argv),
            "exact_command_shell": shlex.join(sys.argv),
            "protocol_sha256": sha256_json(PROTOCOL.__dict__),
            "task_identity_sha256_by_seed": {
                str(row["task_seed"]): row["task"][
                    "solver_task_identity_sha256"
                ]
                for row in task_rows
            },
            "source_hashes_at_start": source_hashes_at_start,
            "source_hashes_at_end": source_hashes_at_end,
            "source_hashes_stable": source_hashes_stable,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_npz(final_npz, arrays)
    summary["artifacts"] = {
        "npz": str(final_npz),
        "npz_sha256": sha256_file(final_npz),
        "supersedes_retained_latest_checkpoint_pointer": str(latest_checkpoint),
        "supersedes_retained_checkpoint_generation": 80,
        "supersedes_retained_checkpoint_json": str(latest_generation_json),
        "supersedes_retained_checkpoint_npz": str(latest_generation_npz),
    }
    _atomic_write_json(output, summary)
    _atomic_write_json(
        final_manifest,
        {
            "json": str(output),
            "json_sha256": sha256_file(output),
            "npz": str(final_npz),
            "npz_sha256": sha256_file(final_npz),
            "incomplete": False,
            "completed_restart_count": 80,
            "supersedes_retained_latest_checkpoint_pointer": str(
                latest_checkpoint
            ),
            "supersedes_retained_checkpoint_generation": 80,
            "supersedes_retained_checkpoint_json": str(
                latest_generation_json
            ),
            "supersedes_retained_checkpoint_npz": str(
                latest_generation_npz
            ),
            **summary["provenance"],
        },
    )
    print(output)
    print(json.dumps({"all_stage0_gates_pass": all_gates}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-stage0-oracle", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.execute_stage0_oracle:
        parser.error("execution is blocked without --execute-stage0-oracle")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    run_stage0(arguments.output)
