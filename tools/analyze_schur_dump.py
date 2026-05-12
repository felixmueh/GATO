#!/usr/bin/env python3
"""Analyze a raw Schur-system dump from GATO BSQP."""

import argparse
import json
from pathlib import Path

import numpy as np


def load_manifest(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def raw_array(path, dtype=np.float32):
    return np.fromfile(path, dtype=dtype)


def block_row_value(raw, block_row, block_col, row, col, state_size, block_row_r_dim, block_row_size):
    block = raw[block_row * block_row_size : (block_row + 1) * block_row_size]
    if block_col + 1 == block_row:
        return block[row * block_row_r_dim + col]
    if block_col == block_row:
        return block[row * block_row_r_dim + state_size + col]
    if block_col == block_row + 1:
        return block[row * block_row_r_dim + 2 * state_size + col]
    return 0.0


def dense_from_block_rows(raw, knot_points, state_size, block_row_r_dim, block_row_size):
    n = knot_points * state_size
    dense = np.zeros((n, n), dtype=np.float64)
    for block_row in range(knot_points):
        for block_col in range(max(0, block_row - 1), min(knot_points, block_row + 2)):
            for row in range(state_size):
                global_row = block_row * state_size + row
                for col in range(state_size):
                    global_col = block_col * state_size + col
                    dense[global_row, global_col] = block_row_value(
                        raw,
                        block_row,
                        block_col,
                        row,
                        col,
                        state_size,
                        block_row_r_dim,
                        block_row_size,
                    )
    return dense


def unpadded_vector(raw, knot_points, state_size):
    out = np.zeros(knot_points * state_size, dtype=np.float64)
    for knot in range(knot_points):
        out[knot * state_size : (knot + 1) * state_size] = raw[(knot + 1) * state_size : (knot + 2) * state_size]
    return out


def norm_report(name, values):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    report = {
        f"{name}_finite": bool(np.all(finite)),
        f"{name}_size": int(values.size),
    }
    if finite.any():
        finite_values = values[finite]
        report.update(
            {
                f"{name}_min": float(np.min(finite_values)),
                f"{name}_max": float(np.max(finite_values)),
                f"{name}_linf": float(np.max(np.abs(finite_values))),
                f"{name}_l2": float(np.linalg.norm(finite_values)),
            }
        )
    return report


def cg(A, b, x0=None, tol=1e-6, max_iters=5000, M=None, dtype=np.float64):
    A = A.astype(dtype, copy=False)
    b = b.astype(dtype, copy=False)
    if M is not None:
        M = M.astype(dtype, copy=False)
    x = np.zeros_like(b, dtype=dtype) if x0 is None else x0.astype(dtype).copy()
    r = b - A @ x
    z = M @ r if M is not None else r.copy()
    p = z.copy()
    rz = float(r @ z)
    r0 = np.linalg.norm(r)
    history = [r0]
    if r0 == 0.0:
        return x, 0, history

    for iteration in range(1, max_iters + 1):
        Ap = A @ p
        denom = float(p @ Ap)
        if not np.isfinite(denom) or denom == 0.0:
            history.append(float("nan"))
            return x, iteration, history
        alpha = rz / denom
        x += alpha * p
        r -= alpha * Ap
        rnorm = np.linalg.norm(r)
        history.append(float(rnorm))
        if rnorm <= tol * r0 + 1e-6:
            return x, iteration, history
        z = M @ r if M is not None else r.copy()
        rz_new = float(r @ z)
        if not np.isfinite(rz_new) or rz == 0.0:
            history.append(float("nan"))
            return x, iteration, history
        beta = rz_new / rz
        p = z + beta * p
        rz = rz_new
    return x, max_iters, history


def gpu_style_pcg(A, b, x0=None, epsilon=1e-3, max_iters=1000, M=None, dtype=np.float32):
    A = A.astype(dtype, copy=False)
    b = b.astype(dtype, copy=False)
    M = M.astype(dtype, copy=False)
    x = np.zeros_like(b, dtype=dtype) if x0 is None else x0.astype(dtype).copy()
    r = b - A @ x
    z = M @ r
    p = z.copy()
    rho = np.asarray(r @ z, dtype=dtype).item()
    abs_tol = np.asarray(1e-6, dtype=dtype).item()
    rho_init = abs(rho)
    residual_history = [float(np.linalg.norm(r.astype(np.float64)))]
    rho_history = [float(rho)]

    if abs(rho) < abs_tol:
        return x, 0, residual_history, rho_history, "initial_abs_rho"

    for iteration in range(1, max_iters + 1):
        Ap = A @ p
        denom = np.asarray(p @ Ap, dtype=dtype).item()
        if not np.isfinite(denom) or denom == 0.0:
            return x, iteration, residual_history, rho_history, "bad_denom"
        alpha = np.asarray(rho / denom, dtype=dtype).item()
        x = x + alpha * p
        r = r - alpha * Ap
        z = M @ r
        rho_new = np.asarray(r @ z, dtype=dtype).item()
        residual_history.append(float(np.linalg.norm(r.astype(np.float64))))
        rho_history.append(float(rho_new))
        if abs(rho_new) < (abs_tol + np.asarray(epsilon, dtype=dtype).item() * rho_init):
            return x, iteration, residual_history, rho_history, "rho_tol"
        beta = np.asarray(rho_new / rho, dtype=dtype).item()
        rho = rho_new
        p = z + beta * p
    return x, max_iters, residual_history, rho_history, "max_iters"


def analyze(manifest_path):
    manifest = load_manifest(manifest_path)
    files = manifest["files"]
    kp = int(manifest["knot_points"])
    ss = int(manifest["state_size"])
    brd = int(manifest["block_row_r_dim"])
    brs = int(manifest["block_row_size"])

    S_raw = raw_array(files["S"])
    P_raw = raw_array(files["P_inv"])
    gamma_raw = raw_array(files["gamma"])
    lambda_before_raw = raw_array(files["lambda_before"])
    lambda_after_raw = raw_array(files["lambda_after"])

    batch_size = int(manifest["batch_size"])
    if batch_size != 1:
        raise NotImplementedError("analyzer currently expects batch_size=1")

    S = dense_from_block_rows(S_raw, kp, ss, brd, brs)
    P = dense_from_block_rows(P_raw, kp, ss, brd, brs)
    gamma = unpadded_vector(gamma_raw, kp, ss)
    lambda_before = unpadded_vector(lambda_before_raw, kp, ss)
    lambda_after = unpadded_vector(lambda_after_raw, kp, ss)

    sym = 0.5 * (S + S.T)
    eig_sym = np.linalg.eigvalsh(sym)
    abs_eig = np.abs(eig_sym)
    pos_abs = abs_eig[abs_eig > 0]
    cond_abs = float(np.max(pos_abs) / np.min(pos_abs)) if pos_abs.size else float("inf")
    min_diag = float(np.min(np.diag(S)))
    max_diag = float(np.max(np.diag(S)))

    residual_before = gamma - S @ lambda_before
    residual_after = gamma - S @ lambda_after

    report = {
        "manifest": str(manifest_path),
        "pcg_iterations": manifest.get("pcg_iterations", []),
        "matrix_dim": int(S.shape[0]),
        "symmetry_linf": float(np.max(np.abs(S - S.T))),
        "symmetry_l2": float(np.linalg.norm(S - S.T)),
        "diag_min": min_diag,
        "diag_max": max_diag,
        "eig_min_sym": float(np.min(eig_sym)),
        "eig_max_sym": float(np.max(eig_sym)),
        "eig_negative_count_sym": int(np.sum(eig_sym < -1e-8)),
        "eig_near_zero_count_sym": int(np.sum(np.abs(eig_sym) < 1e-8)),
        "cond_abs_sym": cond_abs,
        "residual_before_l2": float(np.linalg.norm(residual_before)),
        "residual_after_l2": float(np.linalg.norm(residual_after)),
        "residual_before_linf": float(np.max(np.abs(residual_before))),
        "residual_after_linf": float(np.max(np.abs(residual_after))),
        "lambda_step_l2": float(np.linalg.norm(lambda_after - lambda_before)),
        "lambda_after_l2": float(np.linalg.norm(lambda_after)),
        "gamma_l2": float(np.linalg.norm(gamma)),
    }
    report.update(norm_report("S_raw", S_raw))
    report.update(norm_report("P_inv_raw", P_raw))
    report.update(norm_report("gamma_raw", gamma_raw))

    try:
        direct = np.linalg.solve(S, gamma)
        direct_residual = gamma - S @ direct
        report["direct_solve_ok"] = True
        report["direct_residual_l2"] = float(np.linalg.norm(direct_residual))
        report["direct_lambda_l2"] = float(np.linalg.norm(direct))
        report["pcg_after_vs_direct_l2"] = float(np.linalg.norm(lambda_after - direct))
    except np.linalg.LinAlgError as exc:
        report["direct_solve_ok"] = False
        report["direct_solve_error"] = str(exc)

    cg_solution, cg_iters, cg_history = cg(S, gamma, x0=lambda_before, tol=1e-3, max_iters=5000)
    report["cpu_cg_iters"] = int(cg_iters)
    report["cpu_cg_initial_residual_l2"] = float(cg_history[0])
    report["cpu_cg_final_residual_l2"] = float(cg_history[-1])
    report["cpu_cg_residual_first_10"] = [float(v) for v in cg_history[:10]]
    report["cpu_cg_solution_l2"] = float(np.linalg.norm(cg_solution))

    pcg_solution, pcg_iters, pcg_history = cg(S, gamma, x0=lambda_before, tol=1e-3, max_iters=5000, M=P)
    report["cpu_pcg_iters_with_dumped_P_inv"] = int(pcg_iters)
    report["cpu_pcg_initial_residual_l2"] = float(pcg_history[0])
    report["cpu_pcg_final_residual_l2"] = float(pcg_history[-1])
    report["cpu_pcg_residual_first_10"] = [float(v) for v in pcg_history[:10]]
    report["cpu_pcg_solution_l2"] = float(np.linalg.norm(pcg_solution))

    cg32_solution, cg32_iters, cg32_history = cg(S, gamma, x0=lambda_before, tol=1e-3, max_iters=5000, dtype=np.float32)
    report["cpu_float32_cg_iters"] = int(cg32_iters)
    report["cpu_float32_cg_initial_residual_l2"] = float(cg32_history[0])
    report["cpu_float32_cg_final_residual_l2"] = float(cg32_history[-1])
    report["cpu_float32_cg_residual_first_10"] = [float(v) for v in cg32_history[:10]]
    report["cpu_float32_cg_solution_l2"] = float(np.linalg.norm(cg32_solution.astype(np.float64)))

    pcg32_solution, pcg32_iters, pcg32_history = cg(S, gamma, x0=lambda_before, tol=1e-3, max_iters=5000, M=P, dtype=np.float32)
    report["cpu_float32_pcg_iters_with_dumped_P_inv"] = int(pcg32_iters)
    report["cpu_float32_pcg_initial_residual_l2"] = float(pcg32_history[0])
    report["cpu_float32_pcg_final_residual_l2"] = float(pcg32_history[-1])
    report["cpu_float32_pcg_residual_first_10"] = [float(v) for v in pcg32_history[:10]]
    report["cpu_float32_pcg_solution_l2"] = float(np.linalg.norm(pcg32_solution.astype(np.float64)))

    gpu_pcg_solution, gpu_pcg_iters, gpu_residual_history, gpu_rho_history, gpu_exit = gpu_style_pcg(
        S,
        gamma,
        x0=lambda_before,
        epsilon=1e-3,
        max_iters=int(manifest["max_pcg_iters"]),
        M=P,
        dtype=np.float32,
    )
    report["cpu_gpu_style_float32_pcg_iters"] = int(gpu_pcg_iters)
    report["cpu_gpu_style_float32_pcg_exit"] = gpu_exit
    report["cpu_gpu_style_float32_pcg_initial_residual_l2"] = float(gpu_residual_history[0])
    report["cpu_gpu_style_float32_pcg_final_residual_l2"] = float(gpu_residual_history[-1])
    report["cpu_gpu_style_float32_pcg_initial_rho"] = float(gpu_rho_history[0])
    report["cpu_gpu_style_float32_pcg_final_rho"] = float(gpu_rho_history[-1])
    report["cpu_gpu_style_float32_pcg_residual_first_10"] = [float(v) for v in gpu_residual_history[:10]]
    report["cpu_gpu_style_float32_pcg_rho_first_10"] = [float(v) for v in gpu_rho_history[:10]]
    report["cpu_gpu_style_float32_pcg_solution_l2"] = float(np.linalg.norm(gpu_pcg_solution.astype(np.float64)))

    if "trace_rho" in files:
        trace_stride = int(manifest.get("trace_stride", 0))
        iterations = int(manifest.get("pcg_iterations", [0])[0])
        trace_count = min(trace_stride, iterations + 1)
        trace_rho = raw_array(files["trace_rho"])[:trace_count].astype(np.float64)
        trace_denom = raw_array(files["trace_denom"])[: min(trace_stride, iterations)].astype(np.float64)
        trace_alpha = raw_array(files["trace_alpha"])[: min(trace_stride, iterations)].astype(np.float64)
        threshold = 1e-6 + 1e-3 * abs(trace_rho[0]) if trace_rho.size else float("nan")
        first_below = np.flatnonzero(np.abs(trace_rho[1:]) < threshold) + 1 if trace_rho.size > 1 else np.asarray([], dtype=np.int64)
        report["gpu_trace_rho_initial"] = float(trace_rho[0]) if trace_rho.size else float("nan")
        report["gpu_trace_rho_final"] = float(trace_rho[-1]) if trace_rho.size else float("nan")
        report["gpu_trace_rho_threshold"] = float(threshold)
        report["gpu_trace_first_rho_below_threshold_iter"] = int(first_below[0]) if first_below.size else -1
        report["gpu_trace_rho_first_10"] = [float(v) for v in trace_rho[:10]]
        report["gpu_trace_rho_last_10"] = [float(v) for v in trace_rho[-10:]]
        report["gpu_trace_denom_first_10"] = [float(v) for v in trace_denom[:10]]
        report["gpu_trace_denom_last_10"] = [float(v) for v in trace_denom[-10:]]
        report["gpu_trace_alpha_first_10"] = [float(v) for v in trace_alpha[:10]]
        report["gpu_trace_alpha_last_10"] = [float(v) for v in trace_alpha[-10:]]

    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="Path to *_manifest.json")
    parser.add_argument("--output", help="Optional JSON output path")
    return parser.parse_args()


def main():
    args = parse_args()
    report = analyze(args.manifest)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
