#!/usr/bin/env python3
"""Export the complete saved B1 pilot; no native imports or solver calls."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "savefig.dpi": 170})
MODEL = "gato/dynamics/tiago_right/tiago_right_arm.urdf"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def finite_max(values):
    finite = np.asarray(values)[np.isfinite(values)]
    return float(finite.max()) if finite.size else None


def number(value, spec=".2f"):
    return "null" if value is None or not np.isfinite(value) else format(value, spec)


def penalty_description(protocol):
    penalty = protocol.get("limit_penalty", {})
    if penalty.get("kind") == "C2_quadratic_continuation":
        return ("The reviewed C2 quadratic limit continuation is active, candidate "
                + penalty["reviewed_candidate"] + ", with relaxation width "
                + str(100 * penalty["relaxation_fraction_of_limit_interval"]) + "% of each limit interval. "
                "This is a soft cost and does not guarantee physical bounds.")
    return "The original clamped limit helper is active; the quadratic continuation candidate is excluded."


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def csv_write(path, rows):
    with Path(path).open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def csv_read(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def trajectory_limits(source, provenance):
    path = source / MODEL
    if sha(path) != provenance["source_sha256"][MODEL]:
        raise ValueError("Analysis URDF differs from measured source")
    joints = [j for j in ET.parse(path).getroot().findall("joint") if j.get("type") != "fixed"]
    if len(joints) != 7:
        raise ValueError("Expected exactly seven reduced-model joints")
    return {k: np.asarray([float(j.find("limit").get(k)) for j in joints])
            for k in ("lower", "upper", "effort", "velocity")}


def collect(args):
    raw = args.raw.resolve()
    provenance, protocol = read(raw / "provenance.json"), read(raw / "protocol.json")
    planned = provenance["pairs"]
    if planned != protocol["development_task_ids"] or len(planned) != 12:
        raise ValueError("Expected the complete fixed ordered twelve-task development pilot")
    missing = [p for p in planned if not (raw / f"pair_{p:05d}/metrics.json").is_file()]
    if missing:
        raise ValueError("Pilot not complete; final export withheld. Missing metrics: " + str(missing))
    summary = read(raw / "summary.json")
    by_id = {r["pair_id"]: r for r in summary}
    if len(summary) != len(planned) or set(by_id) != set(planned):
        raise ValueError("Summary does not contain all assigned tasks exactly once")
    limits = trajectory_limits(args.source.resolve(), provenance)
    cases, traces, solves, scope_rows, originals = [], [], [], [], []
    evidence = dict(raw_root=str(raw), source_commit=provenance["source_commit"],
                    numerical_source_base=protocol["source_base"], module_sha256=provenance["module_sha256"],
                    wrapper_sha256=provenance["wrapper_sha256"], protocol_sha256=sha(raw / "protocol.json"),
                    summary_sha256=sha(raw / "summary.json"), provenance_sha256=sha(raw / "provenance.json"),
                    input_sha256=provenance["inputs_sha256"], summary_matches_case_metrics=True, cases={})
    for pair in planned:
        directory = raw / f"pair_{pair:05d}"
        result = read(directory / "metrics.json")
        if result != by_id[pair]:
            raise ValueError("Summary/case metric mismatch: " + str(pair))
        originals.append(result)
        rows = [json.loads(line) for line in (directory / "solves.jsonl").read_text().splitlines()]
        if (len(rows) != result["feedback_solves"] or
                sum(r["work"]["actual_sqp"] for r in rows) != result["actual_sqp_sum"] or
                sum(r["work"]["pcg_iteration_sum"] for r in rows) != result["actual_pcg_iteration_sum"] or
                sum(r["work"]["zero_pcg_outer_exit"] for r in rows) != result["zero_pcg_exit_solves"]):
            raise ValueError("Saved work log does not match case summary: " + str(pair))
        with np.load(directory / "executed.npz", allow_pickle=False) as z:
            t, ct, x, u = z["times"], z["control_times"], z["states"], z["controls"]
            q_excess = np.maximum(np.maximum(limits["lower"]-x[:, :7], x[:, :7]-limits["upper"]), 0.).max(axis=1)
            vr = (np.abs(x[:, 7:])/limits["velocity"]).max(axis=1)
            er = (np.abs(u)/limits["effort"]).max(axis=1) if len(u) else np.array([])
            # Preserve every early-transient sample, regular30ms observations,
            # all global extrema, first violation times and final observation.
            sample = set(range(0, len(t), 10)) | set(np.flatnonzero(t <= .12)) | {len(t)-1, int(np.argmax(q_excess)), int(np.argmax(vr))}
            if len(er):
                sample.add(int(np.argmax(er)))
            for name in ("joint", "velocity", "effort"):
                first = result["physical"][name]["first_violation"]
                if first:
                    sample.add(int(first["sample_index"]))
            for i in sorted(sample):
                traces.append(dict(pair_id=pair, time_s=float(t[i]), position_error_mm=float(z["position_error_m"][i]*1000),
                    tool_axis_error_deg=float(z["tool_axis_error_deg"][i]), orientation_error_deg=float(z["orientation_error_deg"][i]),
                    speed_L1_rad_s=float(z["velocity_l1_rad_s"][i]), joint_excess_rad=float(q_excess[i]),
                    maximum_velocity_ratio=float(vr[i]),
                    maximum_effort_ratio=float(er[min(i, len(er)-1)]) if len(er) else 0.))
            final = result["final"]
            first = {name: result["physical"][name]["first_violation"] for name in ("joint", "velocity", "effort")}
            late = t >= max(0., t[-1]-1.)
            late_u = ct >= max(0., t[-1]-1.)
            first_u = ct < .12
            cap = sum(r["work"]["stop_reason"] == "budget_exhausted" for r in rows)
            case = dict(pair_id=pair, outcome=result["outcome"], task_success=int(result["task_success"]),
                physical_pass=int(result["physical_pass"]), execution_error=int(bool(result["execution_error"])),
                simulated_seconds=result["simulated_seconds"], arrival_seconds=result["reached_seconds"],
                final_position_error_mm=final["position_error_m"]*1000, final_axis_deg=final["tool_axis_error_deg"],
                final_SO3_deg=final["orientation_error_deg"], final_speed_L1_rad_s=final["velocity_l1_rad_s"],
                maximum_joint_excess_rad=result["physical"]["joint"]["maximum_excess"],
                maximum_velocity_ratio=result["physical"]["maximum_velocity_ratio"],
                maximum_effort_ratio=result["physical"]["maximum_effort_ratio"],
                first_joint_violation_s=first["joint"]["time_s"] if first["joint"] else None,
                first_joint_violation_joint=first["joint"]["joint_index"]+1 if first["joint"] else None,
                first_velocity_violation_s=first["velocity"]["time_s"] if first["velocity"] else None,
                first_velocity_violation_joint=first["velocity"]["joint_index"]+1 if first["velocity"] else None,
                first_effort_violation_s=first["effort"]["time_s"] if first["effort"] else None,
                first_effort_violation_joint=first["effort"]["joint_index"]+1 if first["effort"] else None,
                first_120ms_maximum_effort_ratio=float(er[first_u].max()) if first_u.any() else 0.,
                last_second_joint_excess_rad=float(q_excess[late].max()),
                last_second_maximum_velocity_ratio=float(vr[late].max()),
                last_second_maximum_effort_ratio=float(er[late_u].max()) if late_u.any() else 0.,
                last_second_position_min_mm=result["last_second"]["position_error_min_m"]*1000,
                last_second_position_max_mm=result["last_second"]["position_error_max_m"]*1000,
                last_second_position_progress_mm=result["last_second"]["net_position_error_reduction_m"]*1000,
                last_second_joint_variation_L1_rad=result["last_second"]["joint_total_variation_L1_rad"],
                feedback_solves=len(rows), actual_sqp_sum=result["actual_sqp_sum"], actual_pcg_sum=result["actual_pcg_iteration_sum"],
                sqp_min=min(r["work"]["actual_sqp"] for r in rows), sqp_median=float(np.median([r["work"]["actual_sqp"] for r in rows])),
                sqp_max=max(r["work"]["actual_sqp"] for r in rows), zero_pcg_exit_solves=result["zero_pcg_exit_solves"],
                budget_exhausted_solves=cap, other_exit_solves=len(rows)-cap-result["zero_pcg_exit_solves"],
                solve_wall_s=result["solve_wall_s"], native_solve_s=result["native_solve_us"]/1e6,
                metrics_sha256=sha(directory / "metrics.json"))
            cases.append(case)
        for row in rows:
            w = row["work"]
            solves.append(dict(pair_id=pair, time_s=row["feedback_time_s"], actual_sqp=w["actual_sqp"],
                pcg_iteration_sum=w["pcg_iteration_sum"], last_pcg_iterations=w["last_pcg_iterations"],
                stop_reason=w["stop_reason"], zero_pcg_outer_exit=int(w["zero_pcg_outer_exit"]),
                accepted_steps=w["accepted_steps"], native_solve_us=row["native_solve_us"], solve_wall_s=row["solve_wall_s"]))
        for scope in ("seed_limits", "predicted_limits", "executed_prefix_limits"):
            scope_rows.append(dict(pair_id=pair, scope=scope, assessed_solves=len(rows),
                physical_pass_solves=sum(r[scope]["physical_pass"] for r in rows),
                maximum_joint_excess_rad=max(r[scope]["joint"]["maximum_excess"] for r in rows),
                maximum_velocity_ratio=max(r[scope]["maximum_velocity_ratio"] for r in rows),
                maximum_effort_ratio=max(r[scope]["maximum_effort_ratio"] for r in rows)))
        evidence["cases"][str(pair)] = {name: sha(directory / name) for name in ("metrics.json", "executed.npz", "solves.jsonl", "inputs.npz")}
    preflight = read(args.preflight)
    evidence["preflight_metrics_sha256"] = sha(args.preflight)
    build_receipt = raw.parent.parent / "build_receipt.json"
    if build_receipt.is_file():
        evidence["build_receipt"] = read(build_receipt)
        evidence["build_receipt_sha256"] = sha(build_receipt)
    return cases, traces, solves, scope_rows, originals, protocol, provenance, evidence, preflight


def collect_comparison(raw, cases, source, primary_protocol, primary_provenance, primary_raw):
    raw = raw.resolve();provenance = read(raw / "provenance.json");protocol = read(raw / "protocol.json")
    planned = [c["pair_id"] for c in cases]
    if provenance["pairs"] != planned:
        raise ValueError("Comparison must preserve the exact ordered twelve task IDs")
    for key in ("source_sha256", "module_sha256", "wrapper_sha256", "inputs_sha256"):
        if provenance[key] != primary_provenance[key]:
            raise ValueError("Unmatched comparison identity: " + key)
    primary_config = dict(primary_protocol["solver_params"])
    primary_config["max_sqp_iters"] = 1
    if protocol["solver_params"] != primary_config:
        raise ValueError("SQP1 comparison changes more than the SQP cap")
    summary = read(raw / "summary.json")
    if len(summary) != len(planned) or [r["pair_id"] for r in summary] != planned:
        raise ValueError("Comparison incomplete; final export withheld")
    limits = trajectory_limits(source.resolve(), provenance)
    rows, originals, hashes = [], [], {}
    for pair, saved in zip(planned, summary):
        directory = raw / f"pair_{pair:05d}";m = read(directory / "metrics.json")
        preparation = read(directory / "preparation.json")
        primary_preparation = read(primary_raw / f"pair_{pair:05d}/preparation.json")
        for key in ("initial_state_sha256", "goal_pose_sha256"):
            if preparation[key] != primary_preparation[key]:
                raise ValueError("Comparison changed initial state or goal: " + str(pair))
        if m != saved:
            raise ValueError("Comparison summary/case mismatch: " + str(pair))
        originals.append(m)
        log = [json.loads(line) for line in (directory / "solves.jsonl").read_text().splitlines()]
        if len(log) != m["feedback_solves"] or sum(r["work"]["actual_sqp"] for r in log) != m["actual_sqp_sum"]:
            raise ValueError("Comparison work log mismatch: " + str(pair))
        with np.load(directory / "executed.npz", allow_pickle=False) as z:
            x, u = z["states"], z["controls"]
            qe = np.maximum(np.maximum(limits["lower"]-x[:, :7], x[:, :7]-limits["upper"]), 0.)
            vr = np.abs(x[:, 7:])/limits["velocity"]
            er = np.abs(u)/limits["effort"]
            row = dict(pair_id=pair, outcome=m["outcome"], task_success=int(m["task_success"]), physical_pass=int(m["physical_pass"]),
                execution_error=int(bool(m["execution_error"])), simulated_seconds=m["simulated_seconds"],
                failure_class=(m["execution_error"] or "").splitlines()[-1] if m["execution_error"] else "none",
                final_position_error_m=m["final"]["position_error_m"], final_axis_deg=m["final"]["tool_axis_error_deg"],
                final_speed_L1_rad_s=m["final"]["velocity_l1_rad_s"],
                recorded_maximum_joint_excess_rad=m["physical"]["joint"]["maximum_excess"],
                recorded_maximum_velocity_ratio=m["physical"]["maximum_velocity_ratio"],
                recorded_maximum_effort_ratio=m["physical"]["maximum_effort_ratio"],
                largest_finite_joint_excess_rad=finite_max(qe), largest_finite_velocity_ratio=finite_max(vr),
                largest_finite_effort_ratio=finite_max(er),
                nonfinite_executed_state_rows=int(np.count_nonzero(~np.isfinite(x).all(axis=1))),
                feedback_solves=m["feedback_solves"], actual_sqp_sum=m["actual_sqp_sum"],
                actual_pcg_sum=m["actual_pcg_iteration_sum"], zero_pcg_exit_solves=m["zero_pcg_exit_solves"],
                budget_exhausted_solves=sum(r["work"]["stop_reason"] == "budget_exhausted" for r in log),
                metrics_sha256=sha(directory / "metrics.json"))
            rows.append(row)
        hashes[str(pair)] = {name: sha(directory / name) for name in ("metrics.json", "executed.npz", "solves.jsonl", "inputs.npz")}
    return rows, originals, dict(raw_root=str(raw), provenance=provenance, protocol=protocol,
                protocol_sha256=sha(raw / "protocol.json"), summary_sha256=sha(raw / "summary.json"),
                summary_matches_case_metrics=True, cases=hashes)


def reference_comparison(root, current_raw, current_low_raw, current_protocol, current_provenance):
    """Two fixed caps, same twelve inputs, one explicitly declared helper change."""
    rows, evidence, counts = [], {}, []
    for cap, name, now in ((1000, "run-001", current_raw), (1, "sqp1-run-001", current_low_raw)):
        if now is None:
            raise ValueError("Reference comparison requires both completed current caps")
        old = root / "pilot" / name
        old_info, now_info = read(old / "provenance.json"), read(now / "provenance.json")
        old_protocol, now_protocol = read(old / "protocol.json"), read(now / "protocol.json")
        if old_info["pairs"] != now_info["pairs"] or old_info["inputs_sha256"] != now_info["inputs_sha256"]:
            raise ValueError("Old/new comparison has different task/input identities")
        if old_protocol["solver_params"] != now_protocol["solver_params"]:
            raise ValueError("Old/new comparison changes solver parameters")
        for key in ("N", "batch", "timing_mode", "offline_timing", "knot_dt_s", "feedback_s", "sim_dt_s", "timeout_s",
                    "position_tolerance_m", "tool_axis_tolerance_deg", "velocity_l1_tolerance_rad_s",
                    "goal_dwell_s", "final_approach", "reset_rho_each_solve"):
            if old_protocol[key] != now_protocol[key]:
                raise ValueError("Old/new protocol setting differs: " + key)
        old_summary, now_summary = read(old / "summary.json"), read(now / "summary.json")
        ids = now_info["pairs"]
        if [m["pair_id"] for m in old_summary] != ids or [m["pair_id"] for m in now_summary] != ids:
            raise ValueError("Old/new fixed task set is incomplete or reordered")
        cohort = dict(cap=cap, assigned=len(ids))
        for label, saved in (("old", old_summary), ("new", now_summary)):
            cohort.update({label + "_task_pass": sum(m["task_success"] for m in saved),
                label + "_physical_pass": sum(m["physical_pass"] for m in saved),
                label + "_both_pass": sum(m["task_success"] and m["physical_pass"] for m in saved),
                label + "_execution_errors": sum(bool(m["execution_error"]) for m in saved),
                label + "_timeouts": sum(m["outcome"] == "timeout" for m in saved)})
            for category, field in (("joint", "maximum_excess"), ("velocity", "maximum_velocity_ratio"),
                                    ("effort", "maximum_effort_ratio")):
                cohort[label + "_" + category + "_failures"] = sum(not m["physical"][category]["pass_check"] for m in saved)
                values = [m["physical"]["joint"][field] if category == "joint" else m["physical"][field] for m in saved]
                cohort[label + "_" + category + "_maximum"] = None if any(v is None for v in values) else max(values)
        for key, predicate in (("task", lambda m: m["task_success"]), ("physical", lambda m: m["physical_pass"]),
                               ("both", lambda m: m["task_success"] and m["physical_pass"])):
            cohort[key + "_rescues"] = sum(not predicate(a) and predicate(b) for a, b in zip(old_summary, now_summary))
            cohort[key + "_regressions"] = sum(predicate(a) and not predicate(b) for a, b in zip(old_summary, now_summary))
        counts.append(cohort)
        hashes = {}
        for a, b in zip(old_summary, now_summary):
            pair = a["pair_id"];old_dir = old / f"pair_{pair:05d}";new_dir = now / f"pair_{pair:05d}"
            if read(old_dir / "metrics.json") != a or read(new_dir / "metrics.json") != b:
                raise ValueError("Reference per-case metrics differ from saved summary")
            pa, pb = read(old_dir / "preparation.json"), read(new_dir / "preparation.json")
            for key in ("initial_state_sha256", "goal_pose_sha256"):
                if pa[key] != pb[key]:
                    raise ValueError("Old/new case changed initial state or goal")
            row = dict(cap=cap, pair_id=pair)
            for label, metric in (("old", a), ("new", b)):
                row.update({label + "_outcome": metric["outcome"], label + "_task_success": int(metric["task_success"]),
                    label + "_physical_pass": int(metric["physical_pass"]), label + "_execution_error": int(bool(metric["execution_error"])),
                    label + "_final_position_mm": None if metric["final"]["position_error_m"] is None else metric["final"]["position_error_m"]*1000,
                    label + "_final_axis_deg": metric["final"]["tool_axis_error_deg"], label + "_final_speed_L1_rad_s": metric["final"]["velocity_l1_rad_s"],
                    label + "_joint_excess_rad": metric["physical"]["joint"]["maximum_excess"],
                    label + "_velocity_ratio": metric["physical"]["maximum_velocity_ratio"],
                    label + "_effort_ratio": metric["physical"]["maximum_effort_ratio"],
                    label + "_simulated_seconds": metric["simulated_seconds"], label + "_actual_sqp_sum": metric["actual_sqp_sum"]})
            rows.append(row)
            hashes[str(pair)] = dict(old_metrics_sha256=sha(old_dir / "metrics.json"), new_metrics_sha256=sha(new_dir / "metrics.json"),
                                      initial_state_sha256=pa["initial_state_sha256"], goal_pose_sha256=pa["goal_pose_sha256"])
        evidence[str(cap)] = dict(old_raw=str(old), new_raw=str(now), old_summary_sha256=sha(old / "summary.json"),
            new_summary_sha256=sha(now / "summary.json"), old_module_sha256=old_info["module_sha256"], new_module_sha256=now_info["module_sha256"],
            old_source_base=old_protocol["source_base"], new_source_base=now_protocol["source_base"], cases=hashes)
    return rows, counts, evidence


def plots(output):
    cases = csv_read(output / "case_metrics.csv")
    trace = csv_read(output / "sampled_traces.csv")
    work = csv_read(output / "solve_work.csv")
    ids = [int(r["pair_id"]) for r in cases]
    colors = {p: "#27865d" if int(r["physical_pass"]) else "#c05a39" for p, r in zip(ids, cases)}
    for filename, late in (("position_progress.png", False), ("late_position_progress.png", True)):
        fig, axes = plt.subplots(3, 4, figsize=(12, 7.6), constrained_layout=True)
        for ax, pair, case in zip(axes.flat, ids, cases):
            data = [r for r in trace if int(r["pair_id"]) == pair]
            t = np.asarray([float(r["time_s"]) for r in data])
            y = np.asarray([float(r["position_error_mm"]) for r in data])
            if late:
                selected = t >= max(0., t[-1]-1.);t, y = t[selected], y[selected]
            ax.plot(t, y, color=colors[pair], lw=1.6)
            if not late:
                ax.set_yscale("log");ax.axhline(50, color=".5", ls="--", lw=.8)
                ax.set_ylim(bottom=min(20, float(y.min())*.8))
            ax.set_title(f"{pair}: {case['outcome']}; limits {'pass' if int(case['physical_pass']) else 'fail'}", fontsize=9)
            ax.set_xlabel("Simulated time (s)");ax.set_ylabel("Position error (mm)");ax.grid(alpha=.2)
        fig.suptitle("Last observed second; observation ends at arrival or timeout" if late else "All twelve assigned tasks: position progress in mode A")
        fig.savefig(output / filename);plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.8), constrained_layout=True, sharex=True)
    x = np.arange(len(ids))
    axes[0].bar(x, [float(r["maximum_joint_excess_rad"]) for r in cases], color=[colors[p] for p in ids])
    axes[0].set_ylabel("Maximum q excess (rad)")
    axes[1].bar(x-.18, [float(r["maximum_velocity_ratio"]) for r in cases], .36, label="Executed speed / URDF limit", color="#476fa5")
    axes[1].bar(x+.18, [float(r["maximum_effort_ratio"]) for r in cases], .36, label="Applied torque / URDF limit", color="#c0843d")
    axes[1].axhline(1, color="black", ls="--", lw=.9);axes[1].set_ylabel("Maximum physical-limit ratio")
    axes[1].set_xticks(x, [str(p) for p in ids], rotation=35);axes[1].set_xlabel("Frozen task ID in assigned order");axes[1].legend(fontsize=9)
    for ax in axes:ax.grid(axis="y", alpha=.2)
    fig.suptitle("Sampled executed limits over each complete motion");fig.savefig(output / "physical_limits.png");plt.close(fig)
    fig, axes = plt.subplots(3, 4, figsize=(12, 7.6), constrained_layout=True)
    for ax, pair in zip(axes.flat, ids):
        data = [r for r in work if int(r["pair_id"]) == pair]
        t = np.asarray([float(r["time_s"]) for r in data]);sqp = np.asarray([int(r["actual_sqp"]) for r in data])
        zero = np.asarray([bool(int(r["zero_pcg_outer_exit"])) for r in data])
        ax.plot(t, sqp, color="#476fa5", lw=.9);ax.scatter(t[zero], sqp[zero], s=9, color="#27865d", label="Zero-PCG exit")
        ax.set_yscale("log");ax.set_ylim(.8, 1500);ax.axhline(1000, color=".6", ls="--", lw=.7)
        ax.set_title(str(pair));ax.set_xlabel("Feedback time (s)");ax.set_ylabel("Actual SQP iterations");ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Cap1000 is an allowance: actual work and practical zero-PCG exits")
    fig.savefig(output / "actual_sqp.png");plt.close(fig)
    if (output / "sqp1_comparison.csv").is_file():
        comparison = csv_read(output / "sqp1_comparison.csv")
        fig, axes = plt.subplots(2, 1, figsize=(11.5, 7), constrained_layout=True, sharex=True)
        for ax, key, label in zip(axes, ("largest_finite_joint_excess_rad", "largest_finite_velocity_ratio"),
                ("Largest finite q excess (rad)", "Largest finite speed / limit")):
            values = [float(c[key]) if c[key] else np.nan for c in comparison]
            # Exponent coordinates avoid overflowing the log tick generator
            # over300 decades while preserving the actual finite observations.
            exponents = [np.log10(v) if np.isfinite(v) and v > 0 else np.nan for v in values]
            ax.scatter(np.arange(len(ids)), exponents, color="#b23e3e", s=45)
            zeros = np.asarray([np.isfinite(v) and v == 0 for v in values])
            if zeros.any():
                positive = np.asarray(exponents)[np.isfinite(exponents)]
                baseline = float(positive.min()-1) if positive.size else -1.
                ax.scatter(np.arange(len(ids))[zeros], np.full(zeros.sum(), baseline),
                           facecolors="none", edgecolors="#27865d", s=55, label="Exact zero (placed below positive values)")
                ax.legend(fontsize=8)
            ax.set_ylabel("log10 positive " + label);ax.grid(alpha=.2)
        axes[-1].set_xticks(np.arange(len(ids)), [str(p) for p in ids], rotation=35)
        failures = sum(int(c["execution_error"]) for c in comparison)
        timeouts = sum(c["outcome"] == "timeout" for c in comparison)
        axes[-1].set_xlabel(f"All twelve assigned task IDs; {failures} execution failures, {timeouts} timeouts")
        fig.suptitle("SQP1 diagnostic: finite observed limit magnitudes\nNonfinite full-history maxima remain null; points are finite components only", fontsize=12)
        fig.savefig(output / "sqp1_severe_limits.png");plt.close(fig)


def report(output, cases, protocol, provenance, evidence, preflight, comparison=None, comparison_evidence=None, reference=None):
    n = len(cases);reached = sum(c["task_success"] for c in cases);physical = sum(c["physical_pass"] for c in cases)
    both = sum(c["task_success"] and c["physical_pass"] for c in cases);errors = sum(c["execution_error"] for c in cases)
    total_solves = sum(c["feedback_solves"] for c in cases);zero = sum(c["zero_pcg_exit_solves"] for c in cases);caps = sum(c["budget_exhausted_solves"] for c in cases)
    title = "Random-reaching B1 with quadratic continuation" if protocol.get("limit_penalty") else "Random-reaching B1 baseline"
    assessment = ("All assigned tasks remain within the sampled physical limits in this bounded run."
                  if physical == n else "This configuration does not establish limits-respecting behavior across the development set.")
    lines = ["# " + title + ": twelve-task development pilot", "",
        f"All {n} assigned tasks are retained: {reached} rough arrivals, {physical} sampled physical passes, {both} both, and {errors} execution errors. "
        f"There are {n-reached-errors} finite timeouts. " + assessment, "",
        "These are the first12 tasks in the frozen historical100-task order, selected before this pilot's outcomes. The remaining88 were reserved from new tuning; they are not historically unseen. No cases were filtered, retried or rescued.", "",
        "## Setup and provenance", "",
        f"Numerical source: `{protocol['source_base']}`. Measured wrapper/source commit: `{provenance['source_commit']}`. Native N32 module SHA256: `{provenance['module_sha256']}`. Wrapper bytes SHA256: `{provenance['wrapper_sha256']}`.", "",
        "Mode A advances the plant by30 ms per feedback update independently of solve latency. N32 with30 ms knot spacing gives0.93 s prediction; RK4 plant samples are3 ms and each episode stops at rough arrival or8 s. The first solve is recorded and applied. Cold rest-state/zero-control seeds, production warm shifts, dual/rho reset once per episode and retained adaptive rho are explicit. Final approach is off.", "",
        "Costs: running position/velocity/effort4/.1/.0001; terminal position80; orientation running/terminal.1/2; q/velocity/control limit penalties.01/.1/.003; initial rho.01. SQP cap1000, PCG120, existing tolerances.001. Historical BA used control-limit.05 and reset rho each solve. " + penalty_description(protocol), "",
        f"The matching canonical preflight {'passed' if preflight.get('preflight_pass') else 'failed'}: final position error {preflight['final']['position_error_m']*1000:.2f} mm, sampled physical pass {preflight['physical_pass']}. This modest preflight does not certify the random task set.", "",
        "## Per-task outcomes", "",
        "Rough arrival: position<=50 mm, tool-axis<=30 deg and speed L1<=1 rad/s at a feedback boundary, zero dwell. Full SO(3) error is additional data. Physical checks use raw URDF q/v/u bounds, including43 Nm on joints1/2 and26 Nm on joints3-7; historical numerical tolerances are q<=1e-6 rad excess and ratios<=1+1e-6.", "",
        "| ID | Outcome | Limits | End mm | Axis deg | Speed L1 |", "|---|---|---|---:|---:|---:|"]
    validation = evidence.get("build_receipt", {}).get("validation")
    if validation:
        index = lines.index("## Per-task outcomes")
        lines[index:index] = ["Matching-source validation: `" + validation["command"] + "`. Recorded outcome: "
            + validation["outcome"] + ". Validation log SHA256: `" + validation["log_sha256"]
            + "`. The matching regression passes establish bounded readiness; the random-task outcomes below remain separate.", ""]
    for c in cases:
        lines.append(f"| {c['pair_id']} | {c['outcome']} | {'pass' if c['physical_pass'] else 'FAIL'} | {c['final_position_error_mm']:.2f} | {c['final_axis_deg']:.2f} | {c['final_speed_L1_rad_s']:.3f} |")
    lines += ["", "![Every assigned case is shown. Dashed line is50 mm position region only; arrival also requires orientation and speed. Green denotes sampled physical pass and orange a physical failure.](position_progress.png)", "",
        "| ID | q excess rad | Speed ratio | Torque ratio | First torque s / joint |", "|---|---:|---:|---:|---|"]
    for c in cases:
        first = "none" if c['first_effort_violation_s'] is None else f"{c['first_effort_violation_s']:.3f} / J{c['first_effort_violation_joint']}"
        lines.append(f"| {c['pair_id']} | {c['maximum_joint_excess_rad']:.4f} | {c['maximum_velocity_ratio']:.3f} | {c['maximum_effort_ratio']:.3f} | {first} |")
    lines += ["", "![Complete executed-motion q excess and speed/torque ratios. A late clean state does not erase an earlier violation; the first two torque limits are43 Nm.](physical_limits.png)", "",
        "## Early transients and late behavior", "",
        "The following contrasts the first120 ms torque peak with the last observed second. These are measured descriptions, not a cause attribution. Every violation remains a failure regardless of its duration. Late position progress is error at the start of that second minus final error; negative values indicate worsening.", "",
        "| ID | Early torque ratio | Late q rad | Late v ratio | Late u ratio | Late progress mm |", "|---|---:|---:|---:|---:|---:|"]
    for c in cases:
        lines.append(f"| {c['pair_id']} | {c['first_120ms_maximum_effort_ratio']:.3f} | {c['last_second_joint_excess_rad']:.4f} | {c['last_second_maximum_velocity_ratio']:.3f} | {c['last_second_maximum_effort_ratio']:.3f} | {c['last_second_position_progress_mm']:.3f} |")
    lines += ["", "![Last observed second for every case, with each panel scaled to reveal its actual error range. A flat position trace may coexist with joint motion, orientation error or physical violations. Arrival-stopped traces do not show subsequent settling.](late_position_progress.png)", "",
        "## Actual optimizer work", "",
        f"Across {total_solves} recorded feedback solves, {zero} end through the practical zero-PCG outer shortcut and {caps} exhaust the1000-SQP cap. The other {total_solves-zero-caps} use another existing exit. The high cap is not necessarily spent and does not certify stationarity or guarantee lower endpoint error. Existing KKT flags are descriptive and impose no acceptance gate.", "",
        "| ID | Solves | SQP min/median/max | Zero-PCG | Cap exits | Total SQP |", "|---|---:|---|---:|---:|---:|"]
    for c in cases:
        lines.append(f"| {c['pair_id']} | {c['feedback_solves']} | {c['sqp_min']}/{c['sqp_median']:.0f}/{c['sqp_max']} | {c['zero_pcg_exit_solves']} | {c['budget_exhausted_solves']} | {c['actual_sqp_sum']} |")
    lines += ["", "![Actual SQP work at every feedback solve; green points mark zero-PCG outer exits and the dashed line marks the1000 cap. A nonzero-iteration PCG solve is distinct from this shortcut.](actual_sqp.png)", "",
        "## Artifacts and limits", "",
        "`case_metrics.csv` retains all outcomes, timing, first violation joints/times, late behavior and work counts. `sampled_traces.csv` contains30 ms samples plus all early120 ms samples, extrema, first violations and endpoints. `solve_work.csv` retains one row per solve; `prediction_scope.csv` keeps seed, returned prediction and executed-prefix checks separate. Full3 ms histories, exact input/output horizons and per-iteration arrays remain in raw storage, with hashes in `evidence.json`. No whole-arm collision or continuous-time physical certificate is claimed.", "",
        f"Raw root: `{evidence['raw_root']}`. All twelve per-case metrics equal the saved run summary; metric/history/log/input hashes and source/protocol identities are exported. No GPU solve or rerun occurs during this analysis.", "",
        "Regenerate figures on another machine using the compact exported data:", "", "```sh", "python analyze.py --plot-only .", "python report_to_pdf.py --report REPORT.md", "```", "",
        "The next parameter-development choice belongs to a separate recorded protocol. These results provide no isolated solver-defect attribution, reliability guarantee, verified local-minimum claim or basis for silently replacing failed tasks."]
    valid_timeouts = [str(c["pair_id"]) for c in cases if c["outcome"] == "timeout" and c["physical_pass"]]
    low_progress = [str(c["pair_id"]) for c in cases if c["outcome"] == "timeout" and abs(c["last_second_position_progress_mm"]) <= 1.]
    description = ("Physically valid timeouts: " + (", ".join(valid_timeouts) or "none") + ". "
        "Timeouts with at most 1 mm absolute net position-error change during the last observed second: "
        + (", ".join(low_progress) or "none") + ". These are descriptive observations, without establishing a local minimum "
        "or a solver defect. Early torque peaks and late joint-bound violations remain separate measurements.")
    location = lines.index("## Actual optimizer work")
    lines[location:location] = [description, ""]
    if comparison:
        canonical = comparison_evidence.get("canonical_preflight")
        canonical_note = (f" Its canonical preflight {'passed' if canonical['preflight_pass'] else 'failed'} at "
                          f"{canonical['reached_seconds']:.2f} s, with physical pass {canonical['physical_pass']}." if canonical else "")
        block = ["## Separate SQP1 diagnostic on the same twelve tasks", "",
            "After the high-cap pilot, the explicitly requested cap1 diagnostic used the same source, module, tasks, costs, warm starts, rho policy and mode-A clock. Only the SQP cap changed." + canonical_note + " All twelve diagnostic outcomes are retained.", "",
            "| Cohort | Rough arrivals | Physical passes | Both | Timeouts | Execution failures |",
            "|---|---:|---:|---:|---:|---:|",
            f"| SQP cap1000 | {reached} | {physical} | {both} | {n-reached-errors} | {errors} |",
            f"| SQP cap1 | {sum(c['task_success'] for c in comparison)} | {sum(c['physical_pass'] for c in comparison)} | {sum(c['task_success'] and c['physical_pass'] for c in comparison)} | {sum(c['outcome']=='timeout' for c in comparison)} | {sum(c['execution_error'] for c in comparison)} |", "",
            "Saved diagnostic failure classes: " + "; ".join(f"{name}: {count}" for name, count in Counter(c["failure_class"] for c in comparison if c["execution_error"]).items())
            + ("." if any(c["execution_error"] for c in comparison) else "none.")
            + " Counts describe cap sensitivity for this recorded setup. No universal reaching or limits guarantee follows. No solver investigation, tuning, clipping or extra retries were performed during this saved-data analysis.", "",
            "The table shows largest finite observed components, including partial episodes when failures occur. Original metrics retain null whenever their full-history quantity is nonfinite. Finite values are preserved in scientific notation and in log10 coordinates; they are not clipped or relabeled.", "",
            "| ID | Outcome | Stop s | Finite q excess rad | Finite speed ratio |",
            "|---|---|---:|---:|---:|"]
        for c in comparison:
            block.append(f"| {c['pair_id']} | {c['outcome']} | {c['simulated_seconds']:.3f} | {c['largest_finite_joint_excess_rad']:.3e} | {c['largest_finite_velocity_ratio']:.3e} |")
        block += ["", "![All SQP1 cases are shown using the largest finite observed components. Log10 coordinates retain the severe magnitudes; nonfinite state rows and full-history nulls remain separately reported and preserved in raw data.](sqp1_severe_limits.png)", "",
            "`sqp1_comparison.csv` includes failure class, retained execution duration, finite component extrema, original nullable maxima and work. `sqp1_case_metrics.json` preserves the saved case summaries; `comparison_evidence.json` records source/module/protocol and hashes. Comparison raw root: `" + comparison_evidence["raw_root"] + "`.", ""]
        location = lines.index("## Artifacts and limits")
        lines[location:location] = block
    if reference:
        paired, counts, reference_evidence = reference
        block = ["## Original helper versus reviewed quadratic continuation", "",
            "Both caps use the same ordered twelve tasks, initial states, FK pose goals, costs, horizon, clock, warm starts, stopping and retained-rho policy. Initial/goal hashes and configuration identities match. Only the reviewed limit-helper replacement defines the numerical intervention. The original failed cohorts and original report are preserved. These are bounded paired observations, with no universal reliability claim.", "",
            "| Cap | Helper | Arrivals | Physical | Both | Timeout | Errors |", "|---|---|---:|---:|---:|---:|---:|"]
        for row in counts:
            for label in ("old", "new"):
                block.append(f"| {row['cap']} | {label} | {row[label+'_task_pass']} | {row[label+'_physical_pass']} | {row[label+'_both_pass']} | {row[label+'_timeouts']} | {row[label+'_execution_errors']} |")
        block += ["", "Physical categories overlap. Each cell below is violating episodes / maximum recorded magnitude; q uses radians of excess and v/u use physical-limit ratios. A null aggregate means at least one full-history maximum is nonfinite, not that violations disappeared.", "",
            "| Cap | Helper | q count/max rad | v count/max ratio | u count/max ratio |", "|---|---|---:|---:|---:|"]
        for row in counts:
            for label in ("old", "new"):
                values = [str(row[label+'_'+category+'_failures']) + "/" + number(row[label+'_'+category+'_maximum'], '.5g')
                          for category in ("joint", "velocity", "effort")]
                block.append(f"| {row['cap']} | {label} | " + " | ".join(values) + " |")
        high = next(row for row in counts if row["cap"] == 1000)
        if all(high[label+'_'+category+'_maximum'] is not None for label in ('old','new') for category in ('joint','velocity','effort')):
            block += ["", "At cap 1000, maximum q excess changes "
                + number(high['old_joint_maximum'], '.6g') + " to " + number(high['new_joint_maximum'], '.6g') + " rad; "
                "maximum torque ratio changes " + number(high['old_effort_maximum'], '.6g') + " to " + number(high['new_effort_maximum'], '.6g')
                + ". Maximum speed ratio changes " + number(high['old_velocity_maximum'], '.6g') + " to "
                + number(high['new_velocity_maximum'], '.6g') + ". Improvements in one limit category do not imply improvement or compliance in every category."]
        block += ["", "Old means the original clamped helper; new means the reviewed C2 continuation. Paired gains and losses count all twelve assigned tasks:", "",
            "| Cap | Task gains/losses | Physical gains/losses | Valid-arrival gains/losses |", "|---|---:|---:|---:|"]
        for row in counts:
            block.append(f"| {row['cap']} | {row['task_rescues']}/{row['task_regressions']} | {row['physical_rescues']}/{row['physical_regressions']} | {row['both_rescues']}/{row['both_regressions']} |")
        magnitude_examples = [row for row in paired if row["cap"] == 1000 and row["old_task_success"] and not row["new_task_success"]
            and all(row[label+key] is not None for label in ("old_", "new_") for key in ("joint_excess_rad", "effort_ratio"))
            and row["new_joint_excess_rad"] < row["old_joint_excess_rad"] and row["new_effort_ratio"] < row["old_effort_ratio"]]
        for row in magnitude_examples[:2]:
            block += ["", f"Case {row['pair_id']} loses its rough-arrival label while measured limit magnitudes improve: "
                f"maximum q excess changes {row['old_joint_excess_rad']:.6g} to {row['new_joint_excess_rad']:.6g} rad, "
                f"and torque ratio {row['old_effort_ratio']:.5g} to {row['new_effort_ratio']:.5g}. "
                "The original success and physical gates remain unchanged; any remaining q/v/u exceedance still fails the declared physical check."]
        block += ["", "Per-case entries below are old/new. Outcome R=rough arrival, T=timeout, E=execution failure; P/F is sampled physical pass/fail. Position is final error in mm, q is maximum excess in rad, v/u are maximum speed/torque ratios. Null means the saved full-history quantity was nonfinite; no value is replaced with zero. Unrounded values, axis/speed changes and metric hashes are in `old_new_cases.csv`.", ""]
        symbols = {"reached": "R", "timeout": "T", "execution_failed": "E"}
        for cap in (1000, 1):
            block += [f"### Paired cases at SQP cap {cap}", "", "| ID | Result | Limits | End mm | q rad | v ratio | u ratio |", "|---|---|---|---:|---:|---:|---:|"]
            for row in [r for r in paired if r["cap"] == cap]:
                result = symbols[row['old_outcome']] + "/" + symbols[row['new_outcome']]
                physical_pair = ("P" if row['old_physical_pass'] else "F") + "/" + ("P" if row['new_physical_pass'] else "F")
                values = [number(row['old_'+key], '.3g') + "/" + number(row['new_'+key], '.3g') for key in
                          ('final_position_mm', 'joint_excess_rad', 'velocity_ratio', 'effort_ratio')]
                block.append(f"| {row['pair_id']} | {result} | {physical_pair} | " + " | ".join(values) + " |")
            block += [""]
        block += ["`old_new_counts.csv`, `old_new_cases.csv` and `reference_evidence.json` retain this paired comparison and both source/module identities. The complete original raw recordings remain at `" + str(reference_evidence['1000']['old_raw'].rsplit('/pilot/',1)[0]) + "`.", ""]
        location = lines.index("## Artifacts and limits")
        lines[location:location] = block
    replacements = {"first12": "first 12", "remaining88": "remaining 88", "by30": "by 30", "with30": "with 30",
        "gives0.93": "gives 0.93", "are3 ms": "are 3 ms", "effort4/": "effort 4/", "position80": "position 80",
        "running/terminal.1": "running/terminal .1", "penalties.01": "penalties .01", "rho.01": "rho .01",
        "cap1000": "cap 1000", "PCG120": "PCG 120", "tolerances.001": "tolerances .001", "control-limit.05": "control-limit .05",
        "50 mm": "50 mm", "including43": "including 43", "joints1/2": "joints 1/2", "and26": "and 26",
        "joints3-7": "joints 3-7", "first120": "first 120", "the1000": "the 1000", "cap1 diagnostic": "cap 1 diagnostic",
        "and9651": "and 9651", "Case1265": "Case 1265", "the8 s": "the 8 s", "Full3 ms": "Full 3 ms",
        "contains30 ms": "contains 30 ms", "early120 ms": "early 120 ms", "all twelve": "all twelve"}
    text = "\n".join(lines) + "\n"
    for old, new in replacements.items():
        text = text.replace(old, new)
    (output / "REPORT.md").write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pdf-helper", type=Path)
    parser.add_argument("--previews", type=Path)
    parser.add_argument("--plot-only", type=Path)
    parser.add_argument("--comparison-raw", type=Path, help="Complete same-task SQP1 diagnostic")
    parser.add_argument("--comparison-preflight", type=Path)
    parser.add_argument("--reference-root", type=Path, help="Original cohort root containing pilot/run-001 and pilot/sqp1-run-001")
    args = parser.parse_args()
    if args.plot_only:
        plots(args.plot_only.resolve());return
    if any(getattr(args, name) is None for name in ("raw", "preflight", "source", "output", "pdf_helper")):
        parser.error("Provide --raw --preflight --source --output --pdf-helper, or --plot-only")
    if args.output.resolve().is_relative_to(args.raw.resolve()):
        raise ValueError("Final export must be outside retained raw data")
    cases, traces, solves, scopes, originals, protocol, provenance, evidence, preflight = collect(args)
    output = args.output.resolve();output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("case_metrics.csv", cases), ("sampled_traces.csv", traces), ("solve_work.csv", solves), ("prediction_scope.csv", scopes)):
        csv_write(output / name, rows)
    write(output / "evidence.json", evidence);write(output / "source_provenance.json", provenance)
    if "build_receipt" in evidence:
        write(output / "build_receipt.json", evidence["build_receipt"])
    write(output / "protocol.json", protocol);write(output / "case_metrics.json", originals)
    comparison = comparison_evidence = None
    if args.comparison_raw:
        comparison, comparison_originals, comparison_evidence = collect_comparison(args.comparison_raw, cases, args.source,
                protocol, provenance, args.raw)
        if args.comparison_preflight:
            comparison_evidence["canonical_preflight"] = read(args.comparison_preflight)
            comparison_evidence["canonical_preflight_metrics_sha256"] = sha(args.comparison_preflight)
        csv_write(output / "sqp1_comparison.csv", comparison)
        write(output / "sqp1_case_metrics.json", comparison_originals)
        write(output / "comparison_evidence.json", comparison_evidence)
    reference = None
    if args.reference_root:
        reference = reference_comparison(args.reference_root, args.raw, args.comparison_raw, protocol, provenance)
        csv_write(output / "old_new_cases.csv", reference[0]);csv_write(output / "old_new_counts.csv", reference[1])
        write(output / "reference_evidence.json", reference[2])
    plots(output);report(output, cases, protocol, provenance, evidence, preflight, comparison, comparison_evidence, reference)
    if Path(__file__).resolve() != output / "analyze.py":
        shutil.copy2(__file__, output / "analyze.py")
    shutil.copy2(args.pdf_helper, output / "report_to_pdf.py")
    spec = importlib.util.spec_from_file_location("report_to_pdf", args.pdf_helper)
    helper = importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    pages = helper.render(output / "REPORT.md", args.previews)
    print(json.dumps(dict(cases=len(cases), task_pass=sum(c['task_success'] for c in cases),
        physical_pass=sum(c['physical_pass'] for c in cases), both=sum(c['task_success'] and c['physical_pass'] for c in cases),
        pdf_pages=pages, output=str(output)), indent=2))


if __name__ == "__main__":
    main()
