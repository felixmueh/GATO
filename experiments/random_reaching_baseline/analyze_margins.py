#!/usr/bin/env python3
"""Export saved margin-study cohorts without native imports or GPU calls.

Only complete, ordered twelve-case cohorts enter comparisons. Partial cohorts,
failed episodes and missing future cohorts remain explicitly recorded.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
MODEL = "gato/dynamics/tiago_right/tiago_right_arm.urdf"
REFERENCE = "Reference-continuation-public-mu1"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 170})


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def csv_write(path, rows, fields=None):
    with Path(path).open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def csv_read(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def finite(value):
    return value is not None and np.isfinite(value)


def number(value, spec=".5g"):
    return format(float(value), spec) if finite(value) else "null"


def maximum(values):
    """Preserve nullable full-history maxima; never discard nonfinite episodes."""
    return max(values) if values and all(finite(v) for v in values) else None


def validate_protocol(protocol, reference, name):
    """Only interior bounds, limit weights, qd cost and mu may vary."""
    fixed = ("timing_mode", "offline_timing", "N", "knot_dt_s", "feedback_s", "sim_dt_s",
             "timeout_s", "position_tolerance_m", "tool_axis_tolerance_deg",
             "velocity_l1_tolerance_rad_s", "goal_dwell_s", "final_approach", "batch",
             "reset_rho_each_solve", "initialization", "development_task_ids",
             "task_ids", "task_ids_sha256", "input_sha256")
    for key in fixed:
        if key not in protocol or protocol[key] != reference[key]:
            raise ValueError(f"Fixed protocol setting changed: {name}/{key}")
    allowed = {"q_lim_cost", "vel_lim_cost", "ctrl_lim_cost", "qd_cost", "mu"}
    measured = {k: v for k, v in protocol["solver_params"].items() if k not in allowed}
    original = {k: v for k, v in reference["solver_params"].items() if k not in allowed}
    if measured != original:
        raise ValueError(f"Unapproved solver/cost configuration difference: {name}")
    for key in ("kind", "reviewed_candidate", "relaxation_fraction_of_limit_interval"):
        if protocol["limit_penalty"].get(key) != reference["limit_penalty"].get(key):
            raise ValueError(f"Continuation configuration changed: {name}/{key}")


def find_run(root, role="pilot"):
    candidates = (root, root / role, root / role / "run-001")
    for path in candidates:
        if (path / "provenance.json").is_file():
            return path
    return root / role


def prefix_mismatches(name, pair, directory, protocol):
    """Compare returned first future state with the same-time executed endpoint."""
    log = directory / "solves.jsonl"
    history = directory / "executed.npz"
    if not log.is_file() or not history.is_file():
        return []
    with np.load(history, allow_pickle=False) as data:
        times, states = data["times"], data["states"]
    dt = protocol["knot_dt_s"]
    rows = []
    for line in log.read_text().splitlines():
        item = json.loads(line)
        step = item["step"]
        returned = directory / f"step_{step:04d}_returned.npz"
        end = item["feedback_time_s"] + dt
        row = dict(cohort=name, pair_id=pair, step=step,
                   feedback_time_s=item["feedback_time_s"], endpoint_time_s=end,
                   recorded_prefix_duration_s=item["executed_duration_s"],
                   status=None, q_max_abs_difference_rad=None,
                   v_max_abs_difference_rad_s=None,
                   returned_sha256=sha(returned) if returned.is_file() else None)
        if not np.isclose(item["executed_duration_s"], dt, rtol=0., atol=1e-10):
            row["status"] = "short_prefix_not_same_time"
        elif not returned.is_file():
            row["status"] = "missing_returned_file"
        else:
            index = int(np.searchsorted(times, end - 1e-9))
            if index >= len(times) or not np.isclose(times[index], end, rtol=0., atol=1e-8):
                row["status"] = "executed_prefix_incomplete"
            else:
                with np.load(returned, allow_pickle=False) as result:
                    packed = result["optimized"].reshape(-1)
                    expected = protocol["N"] * 14 + (protocol["N"] - 1) * 7
                    if packed.size != expected:
                        raise ValueError(f"Unexpected packed horizon: {returned}")
                    predicted = packed[21:35]  # [x0,u0,x1,u1,...,xN-1]
                actual = states[index]
                if not np.isfinite(predicted).all():
                    row["status"] = "nonfinite_prediction"
                elif not np.isfinite(actual).all():
                    row["status"] = "nonfinite_executed_endpoint"
                else:
                    row["status"] = "matched"
                    row["q_max_abs_difference_rad"] = float(np.max(np.abs(predicted[:7] - actual[:7])))
                    row["v_max_abs_difference_rad_s"] = float(np.max(np.abs(predicted[7:] - actual[7:])))
        rows.append(row)
    return rows


def collect_cohort(name, root, fallback_protocol, expected, reference_inputs,
                   assigned_pairs=None, run_role="pilot", include_prefix_mismatches=True):
    raw = find_run(root, role=run_role)
    provenance = read(raw / "provenance.json") if (raw / "provenance.json").is_file() else None
    preflight_run = find_run(root / "preflight", role="preflight")
    preflight_path = preflight_run / "preflight/metrics.json"
    preflight = read(preflight_path) if preflight_path.is_file() else None
    preflight_provenance_path = preflight_run / "provenance.json"
    preflight_provenance = read(preflight_provenance_path) if preflight_provenance_path.is_file() else None
    protocol = (read(raw / "protocol.json") if (raw / "protocol.json").is_file() else
                preflight_provenance["protocol"] if preflight_provenance else fallback_protocol)
    hold_path = root / "pilot_held.json"
    hold = read(hold_path) if hold_path.is_file() else None
    evaluation_hold_path = root / "evaluation_held.json"
    evaluation_hold = read(evaluation_hold_path) if evaluation_hold_path.is_file() else None
    planned = protocol.get("development_task_ids", expected) if protocol else expected
    if planned != expected or len(planned) != 12:
        raise ValueError(f"Different ordered development task set: {name}")
    assigned = expected if assigned_pairs is None else assigned_pairs
    if provenance and provenance["pairs"] != assigned:
        raise ValueError(f"Different measured task assignment: {name}")
    summary_path = raw / "summary.json"
    summary = read(summary_path) if summary_path.is_file() else []
    ids = [entry["pair_id"] for entry in summary]
    if ids != assigned[:len(ids)] or len(ids) != len(set(ids)):
        raise ValueError(f"Summary is not a unique ordered prefix: {name}")
    metrics, status_rows, inputs, case_hashes = [], [], {}, {}
    for pair in assigned:
        directory = raw / f"pair_{pair:05d}"
        metric_path = directory / "metrics.json"
        m = read(metric_path) if metric_path.is_file() else None
        saved = next((entry for entry in summary if entry["pair_id"] == pair), None)
        if saved and saved != m:
            raise ValueError(f"Summary/case metrics differ: {name}/{pair}")
        status = (m["outcome"] if m else "in_progress" if directory.exists() else "not_recorded")
        status_rows.append(dict(cohort=name, pair_id=pair, status=status,
            in_summary=int(saved is not None), complete=int(bool(m and m.get("complete"))),
            physical_pass=int(bool(m and m.get("physical_pass"))),
            task_success=int(bool(m and m.get("task_success"))),
            execution_error=m.get("execution_error") if m else None,
            interrupted=int(bool(m and m.get("interrupted"))),
            shared_setup_failure=int(bool(m and m.get("shared_setup_failure")))))
        if m:
            metrics.append(m)
        case_hashes[str(pair)] = {p.name: sha(p) for p in
            (metric_path, directory / "inputs.npz", directory / "executed.npz", directory / "solves.jsonl") if p.is_file()}
        if (directory / "inputs.npz").is_file():
            with np.load(directory / "inputs.npz", allow_pickle=False) as z:
                inputs[pair] = {key: hashlib.sha256(np.ascontiguousarray(z[key]).tobytes()).hexdigest()
                    for key in ("initial_state", "target_xyz", "target_rotation")}
            if reference_inputs is not None and inputs[pair] != reference_inputs[pair]:
                raise ValueError(f"Paired physical input changed: {name}/{pair}")
    complete = len(summary) == len(assigned) and all(m.get("complete") for m in summary)
    interrupted = any(m.get("interrupted") for m in metrics)
    shared_failure = any(m.get("shared_setup_failure") for m in metrics) or (raw / "setup_failure.json").is_file()
    status = "complete" if complete else "shared_setup_failure" if shared_failure else "interrupted" if interrupted else "partial" if provenance else "not_recorded"
    if hold and preflight and not preflight.get("preflight_pass") and not provenance and not summary:
        status = "held_preflight_failure"
        for row in status_rows:
            if row["status"] == "not_recorded":
                row["status"] = status
    info = dict(name=name, raw_root=str(raw), status=status, recorded_cases=len(summary), assigned=len(assigned),
        protocol=protocol, provenance=provenance, preflight=preflight,
        preflight_provenance=preflight_provenance, preflight_raw_root=str(preflight_run), pilot_hold=hold,
        pilot_hold_sha256=sha(hold_path) if hold else None,
        evaluation_hold=evaluation_hold, evaluation_hold_sha256=sha(evaluation_hold_path) if evaluation_hold else None,
        hashes={p.name: sha(p) for p in (summary_path, raw / "protocol.json", raw / "provenance.json") if p.is_file()},
        preflight_hashes={p.name: sha(p) for p in (preflight_path, preflight_run / "protocol.json", preflight_run / "provenance.json") if p.is_file()},
        case_hashes=case_hashes, physical_input_array_sha256=inputs)
    cases, mismatches = [], []
    if complete:
        for m in metrics:
            p = m["physical"]
            row = dict(cohort=name, pair_id=m["pair_id"], outcome=m["outcome"],
                task_success=int(m["task_success"]), physical_pass=int(m["physical_pass"]),
                both_pass=int(m["task_success"] and m["physical_pass"]),
                execution_failure=int(bool(m["execution_error"])), failure_class=m["execution_error"],
                arrival_seconds=m["reached_seconds"], observed_seconds=m["simulated_seconds"],
                final_position_mm=m["final"]["position_error_m"] * 1000 if m["final"]["position_error_m"] is not None else None,
                final_tool_axis_deg=m["final"]["tool_axis_error_deg"],
                final_speed_L1_rad_s=m["final"]["velocity_l1_rad_s"],
                joint_violation=int(not p["joint"]["pass_check"]),
                velocity_violation=int(not p["velocity"]["pass_check"]),
                effort_violation=int(not p["effort"]["pass_check"]),
                maximum_joint_excess_rad=p["joint"]["maximum_excess"],
                maximum_velocity_ratio=p["maximum_velocity_ratio"], maximum_effort_ratio=p["maximum_effort_ratio"],
                feedback_solves=m["feedback_solves"], actual_sqp_sum=m["actual_sqp_sum"],
                zero_pcg_exit_solves=m["zero_pcg_exit_solves"],
                metrics_sha256=case_hashes[str(m["pair_id"])]["metrics.json"])
            cases.append(row)
            if include_prefix_mismatches:
                mismatches.extend(prefix_mismatches(name, m["pair_id"], raw / f"pair_{m['pair_id']:05d}", protocol))
    return info, status_rows, cases, mismatches, inputs


def aggregate(cases, mismatches):
    results = []
    for name in dict.fromkeys(c["cohort"] for c in cases):
        cohort = [c for c in cases if c["cohort"] == name]
        mm = [r for r in mismatches if r["cohort"] == name]
        matched = [r for r in mm if r["status"] == "matched"]
        row = dict(cohort=name, assigned=len(cohort), rough_arrivals=sum(c["task_success"] for c in cohort),
            physical_passes=sum(c["physical_pass"] for c in cohort), both_passes=sum(c["both_pass"] for c in cohort),
            execution_failures=sum(c["execution_failure"] for c in cohort),
            timeouts=sum(c["outcome"] == "timeout" for c in cohort),
            joint_violation_cases=sum(c["joint_violation"] for c in cohort),
            velocity_violation_cases=sum(c["velocity_violation"] for c in cohort),
            effort_violation_cases=sum(c["effort_violation"] for c in cohort),
            maximum_joint_excess_rad=maximum([c["maximum_joint_excess_rad"] for c in cohort]),
            maximum_velocity_ratio=maximum([c["maximum_velocity_ratio"] for c in cohort]),
            maximum_effort_ratio=maximum([c["maximum_effort_ratio"] for c in cohort]),
            feedback_solves=sum(c["feedback_solves"] for c in cohort),
            actual_sqp_sum=sum(c["actual_sqp_sum"] for c in cohort),
            zero_pcg_exit_solves=sum(c["zero_pcg_exit_solves"] for c in cohort),
            mismatch_matched=len(matched), mismatch_unmatched=len(mm)-len(matched),
            q_prefix_mismatch_max_rad=maximum([r["q_max_abs_difference_rad"] for r in matched]),
            v_prefix_mismatch_max_rad_s=maximum([r["v_max_abs_difference_rad_s"] for r in matched]))
        for key, suffix in (("q_max_abs_difference_rad", "q_prefix_mismatch_p95_rad"),
                            ("v_max_abs_difference_rad_s", "v_prefix_mismatch_p95_rad_s")):
            row[suffix] = float(np.quantile([r[key] for r in matched], .95)) if matched else None
        results.append(row)
    return results


def collect_reserved_evaluation(raw_root, info, cases, status_rows, expected, selected=None):
    """Optional frozen selected-cohort evaluation on the remaining 88 tasks."""
    if selected is None:
        recorded = [item["name"] for item in info if (raw_root / item["name"] / "evaluation/provenance.json").is_file()]
        held = [item["name"] for item in info if (raw_root / item["name"] / "evaluation_held.json").is_file()]
        if len(recorded) > 1:
            raise ValueError("Multiple recorded evaluations; select one with --evaluation-cohort")
        selected = recorded[0] if recorded else held[-1] if held else None
    pilot = next((item for item in info if item["name"] == selected), None)
    if pilot is None:
        if selected:
            raise ValueError("Unknown reserved-evaluation cohort: " + selected)
        return None
    protocol = pilot["protocol"]
    roster = protocol["task_ids"]
    if len(roster) != 100 or len(set(roster)) != 100 or roster[:12] != expected:
        raise ValueError("Reserved evaluation requires the unchanged frozen 100-task order")
    result = collect_cohort(selected + "-reserved88", raw_root / selected, protocol,
        expected, None, assigned_pairs=roster[12:], run_role="evaluation", include_prefix_mismatches=False)
    evaluation, evaluation_status, evaluation_cases = result[:3]
    if evaluation["protocol"] != protocol:
        raise ValueError("Reserved evaluation protocol differs from selected development cohort")
    if evaluation["provenance"]:
        if pilot["status"] != "complete":
            raise ValueError("Reserved evaluation recorded before complete selected development cohort")
        for key in ("source_sha256", "module_sha256", "wrapper_sha256", "inputs_sha256"):
            if evaluation["provenance"][key] != pilot["provenance"][key]:
                raise ValueError("Reserved/development identity changed: " + key)
    populations, all_cases = [], []
    if evaluation["status"] == "complete":
        development = [dict(row, cohort="Development12") for row in cases if row["cohort"] == pilot["name"]]
        reserved = [dict(row, cohort="Reserved88") for row in evaluation_cases]
        all_cases = development + reserved
        combined = [dict(row, cohort="Combined100") for row in all_cases]
        populations = aggregate(development + reserved + combined, [])
        if [row["pair_id"] for row in all_cases] != roster:
            raise ValueError("Combined case order differs from frozen roster")
    hold_path = raw_root / selected / "evaluation_held.json"
    hold = read(hold_path) if hold_path.is_file() else None
    if hold and evaluation["status"] == "not_recorded":
        evaluation["status"] = "held_development_failure"
        for row in evaluation_status:
            row["status"] = "held_development_failure"
    statuses = [row for row in status_rows if row["cohort"] == pilot["name"]] + evaluation_status
    return dict(info=evaluation, populations=populations, cases=all_cases, case_status=statuses,
        selected_cohort=selected, hold_record=hold, hold_sha256=sha(hold_path) if hold else None,
        development_summary_sha256=pilot["hashes"].get("summary.json"),
        assigned_order=roster, reserved_order=roster[12:])


def preflight_rows(info):
    rows = []
    for item in info:
        m = item["preflight"]
        if not m:
            continue
        rows.append(dict(cohort=item["name"], mu=item["protocol"]["solver_params"]["mu"],
            outcome=m["outcome"], preflight_pass=int(m["preflight_pass"]),
            task_success=int(m["task_success"]), physical_pass=int(m["physical_pass"]),
            finite=int(m["physical"]["finite"]), execution_error=m["execution_error"],
            initial_position_mm=m["initial"]["position_error_m"] * 1000,
            final_position_mm=m["final"]["position_error_m"] * 1000 if m["final"]["position_error_m"] is not None else None,
            endpoint_displacement_mm=m["executed_endpoint_displacement_m"] * 1000 if m["executed_endpoint_displacement_m"] is not None else None,
            arrival_seconds=m["reached_seconds"], observed_seconds=m["simulated_seconds"],
            feedback_solves=m["feedback_solves"], zero_pcg_exit_solves=m["zero_pcg_exit_solves"],
            actual_sqp_sum=m["actual_sqp_sum"], pilot_status=item["status"],
            metrics_sha256=item["preflight_hashes"]["metrics.json"]))
    return rows


def short(name):
    return "Reference" if name == REFERENCE else name.split("-", 1)[0]


def plots(output):
    cases = csv_read(output / "cases.csv")
    counts = csv_read(output / "cohorts.csv")
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    for row in counts:
        name = row["cohort"]
        arrivals = sorted(float(c["arrival_seconds"]) for c in cases
            if c["cohort"] == name and c["task_success"] == "1")
        ax.step([0.] + arrivals + [8.], [0.] + [(i+1)/12 for i in range(len(arrivals))] + [len(arrivals)/12],
            where="post", label=f"{short(name)} ({len(arrivals)}/12)")
    ax.set(xlim=(0, 8), ylim=(0, 1), xlabel="Simulated episode time (s)",
        ylabel="Cumulative rough arrivals / 12 assigned", title="Arrival fraction across every assigned task")
    ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output / "arrival_ecdf.png"); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))
    quantities = (("maximum_joint_excess_rad", "Joint-position excess (rad)", 0.),
                  ("maximum_velocity_ratio", "Speed / physical limit", 1.),
                  ("maximum_effort_ratio", "Torque / physical limit", 1.))
    for ax, (key, label, bound) in zip(axes, quantities):
        for i, row in enumerate(counts):
            per_case = [float(c[key]) for c in cases if c["cohort"] == row["cohort"] and c[key]]
            ax.scatter([i] * len(per_case), per_case, s=22, alpha=.45, color="C0", label="Episode maxima" if i == 0 else None)
            if row[key]:
                ax.scatter([i], [float(row[key])], marker="D", s=55, color="C3", label="Cohort maximum" if i == 0 else None)
            else:
                ax.text(i, .98, "null", transform=ax.get_xaxis_transform(), ha="center", va="top", color="C3")
        ax.axhline(bound, color="black", ls="--", lw=.8)
        ax.set_xticks(range(len(counts)), [short(r["cohort"]) for r in counts], rotation=35, ha="right")
        ax.set_ylabel(label); ax.grid(axis="y", alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8)
    fig.suptitle("Executed motion: per-episode and cohort physical maxima")
    fig.tight_layout(rect=(0, .07, 1, 1)); fig.savefig(output / "physical_maxima.png"); plt.close(fig)
    population_path = output / "evaluation_population_counts.csv"
    if population_path.is_file():
        populations = csv_read(population_path)
        population_cases = csv_read(output / "evaluation_all_cases.csv")
        if populations:
            fig, ax = plt.subplots(figsize=(7.8, 4.4))
            for population in populations:
                name, denominator = population["cohort"], int(population["assigned"])
                assigned = population_cases if name == "Combined100" else [row for row in population_cases if row["cohort"] == name]
                arrivals = sorted(float(row["arrival_seconds"]) for row in assigned if row["task_success"] == "1")
                ax.step([0.] + arrivals + [8.], [0.] + [(i+1)/denominator for i in range(len(arrivals))] + [len(arrivals)/denominator],
                    where="post", label=f"{name} ({len(arrivals)}/{denominator})")
            ax.set(xlim=(0, 8), ylim=(0, 1), xlabel="Simulated episode time (s)",
                ylabel="Cumulative rough arrivals / assigned tasks", title="Frozen configuration: development and reserved evaluation")
            ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=.2)
            fig.tight_layout(); fig.savefig(output / "evaluation_population_arrivals.png"); plt.close(fig)


def cost_scale_table(info, rows):
    """Use matching recorded settings; never substitute a proposed grid cell."""
    chosen = {}
    specs = (("velocity", "all", "speed=2", "vel_lim_cost", "qd_cost", "velocity_fraction_of_physical_limit"),
             ("effort", "arm_right_2_joint", "start_gravity_max_abs_rounded", "ctrl_lim_cost", "u_cost", "effort_fraction_of_physical_limit"),
             ("joint", "arm_right_2_joint", "physical_upper", "q_lim_cost", None, "joint_margin_rad"))
    for item in info:
        p = item["protocol"]
        bounds = p.get("limit_penalty", {}).get("cost_bounds") if p else None
        if not bounds:
            continue
        cfg = p["solver_params"]
        for category, joint, point, weight, ordinary, margin in specs:
            matches = []
            for row in rows:
                if (row["category"], row["joint"], row["point"]) != (category, joint, point):
                    continue
                upper = float(row["physical_upper"])
                target = upper - bounds[margin] if category == "joint" else upper * bounds[margin]
                if (np.isclose(float(row["cost_upper"]), target, rtol=0., atol=1e-10)
                        and np.isclose(float(row["penalty_weight"]), cfg[weight], rtol=0., atol=1e-12)
                        and (ordinary is None or np.isclose(float(row["regularization_weight"]), cfg[ordinary], rtol=0., atol=1e-12))):
                    matches.append(row)
            if len(matches) > 1:
                raise ValueError("Ambiguous cost-scale table row")
            if matches:
                row = matches[0]
                identity = tuple(row.items())
                if identity not in chosen:
                    chosen[identity] = dict(row, cohorts=[])
                chosen[identity]["cohorts"].append(short(item["name"]))
    return list(chosen.values())


def collect_rationale(args, raw_root, source, output):
    roots = dict(cost_scaling=args.cost_scaling_root or source / "experiments/random_reaching_baseline",
                 physical_scales=args.physics_root or raw_root.with_name(raw_root.name.replace("random-reaching-margins-", "random-reaching-margins-physics-", 1)),
                 joint_tradeoffs=args.joint_tradeoffs_root or raw_root.with_name(raw_root.name.replace("random-reaching-margins-", "random-reaching-joint-margin-tradeoffs-", 1)))
    names = dict(cost_scaling=("cost_scaling.json", "cost_scaling.csv", "cost_scaling.py", "COST_SCALING.md",
                              "velocity_cost_curve.png", "velocity_cost_curve.pdf", "velocity_cost_curve.csv",
                              "velocity_cost_curve.json", "velocity_cost_curve.py"),
                 physical_scales=("physical_scales.json", "acceleration_capacities.csv", "gravity_headroom.csv", "PHYSICAL_SCALES.md"),
                 joint_tradeoffs=("joint_margin_tradeoffs.json", "joint_margin_summary.csv", "joint_margin_scalar_points.csv", "JOINT_MARGIN_TRADEOFFS.md"))
    evidence = {}
    for kind, root in roots.items():
        root = root.resolve()
        primary = root / names[kind][0]
        if not primary.is_file():
            continue
        copied = {}
        paths = [root / name for name in names[kind]]
        if kind in ("physical_scales", "joint_tradeoffs"):
            script_name = "physical_scales.py" if kind == "physical_scales" else "joint_margin_tradeoffs.py"
            saved_script = root / script_name
            paths.append(saved_script if saved_script.is_file() else source / "experiments/random_reaching_baseline" / script_name)
        for path in paths:
            if path.is_file():
                if path.resolve() != output / path.name:
                    shutil.copy2(path, output / path.name)
                copied[path.name] = dict(source=str(path), sha256=sha(path))
        evidence[kind] = dict(root=str(root), summary=read(primary), files=copied)
    return evidence


def collect_search(raw_root, source, output):
    from search_results import collect
    stages, cases, evidence = collect(raw_root, source)
    if not stages:
        return None
    csv_write(output / "SEARCH_STAGES.csv", stages)
    csv_write(output / "SEARCH_CASES.csv", cases)
    write(output / "SEARCH_EVIDENCE.json", dict(stages=evidence))
    files = [source / "experiments/random_reaching_baseline/search_results.py"]
    files += [raw_root / name for name in ("SEARCH_TRADEOFF_2342.png", "SEARCH_TRADEOFF_2342.pdf",
        "SEARCH_TRADEOFF_2342.csv", "SEARCH_TRADEOFF_2342.json", "plot_search_tradeoff.py")]
    copied = {}
    for path in files:
        if path.is_file():
            if path.resolve() != output / path.name:
                shutil.copy2(path, output / path.name)
            copied[path.name] = dict(source=str(path), sha256=sha(path))
    for name in ("SEARCH_STAGES.csv", "SEARCH_CASES.csv", "SEARCH_EVIDENCE.json"):
        copied[name] = dict(sha256=sha(output / name))
    return dict(stages=stages, cases=cases, files=copied)


def endpoint_progress(selected):
    """Describe saved selected-cohort endpoints and final recorded work only."""
    if selected is None:
        return []
    root = Path(selected['raw_root']); protocol = selected['protocol']; rows = []
    for pair in protocol['development_task_ids']:
        directory = root / f'pair_{pair:05d}'; m = read(directory / 'metrics.json')
        last = m['last_second']; final = m['final']
        work = [json.loads(line)['work'] for line in (directory / 'solves.jsonl').read_text().splitlines()][-10:]
        axis_reduction = None
        with np.load(directory / 'executed.npz', allow_pickle=False) as z:
            index = np.flatnonzero(z['times'] >= max(0., z['times'][-1]-1.))
            axis = z['tool_axis_error_deg'][index]
            if len(axis) and np.isfinite(axis).all():
                axis_reduction = float(axis[0]-axis[-1])
        rows.append(dict(cohort=selected['name'], pair_id=pair, outcome=m['outcome'],
            task_success=int(m['task_success']), physical_pass=int(m['physical_pass']),
            execution_error=m['execution_error'], observed_seconds=m['simulated_seconds'],
            final_position_mm=final['position_error_m']*1000 if finite(final['position_error_m']) else None,
            final_tool_axis_deg=final['tool_axis_error_deg'], final_speed_L1_rad_s=final['velocity_l1_rad_s'],
            endpoint_position_gate=int(finite(final['position_error_m']) and final['position_error_m'] <= protocol['position_tolerance_m']),
            endpoint_axis_gate=int(finite(final['tool_axis_error_deg']) and final['tool_axis_error_deg'] <= protocol['tool_axis_tolerance_deg']),
            endpoint_speed_gate=int(finite(final['velocity_l1_rad_s']) and final['velocity_l1_rad_s'] <= protocol['velocity_l1_tolerance_rad_s']),
            last_second_observation_s=last['observation_seconds'],
            last_second_position_error_reduction_mm=last['net_position_error_reduction_m']*1000 if finite(last['net_position_error_reduction_m']) else None,
            last_second_axis_error_reduction_deg=axis_reduction,
            last_second_joint_total_variation_L1_rad=last['joint_total_variation_L1_rad'],
            tail_solves=len(work), tail_zero_pcg_exits=sum(bool(w['zero_pcg_outer_exit']) for w in work),
            tail_budget_exhausted=sum(w['stop_reason']=='budget_exhausted' for w in work),
            tail_sqp_min=min((w['actual_sqp'] for w in work), default=None),
            tail_sqp_max=max((w['actual_sqp'] for w in work), default=None)))
    return rows


def report(output, info, counts, cases, rationale, reserved, selected_name, search):
    lines = ["# TIAGo reaching: interior penalty bounds", "",
        "This report retains every assigned task. Main pilot comparisons use complete twelve-case cohorts; canonical episodes and one-case physical-search screens are labeled separately. Partial, interrupted, failed-setup and unrecorded cohorts remain listed below. Completed execution failures count as failures, not missing cases. These are sequential parameter-development observations, not an independent reliability test.", "",
        "## Cohort status", "", "| Cohort | State | Recorded / assigned | Canonical preflight |", "|---|---|---:|---|"]
    final = next((row for row in counts if row["cohort"] == selected_name), None)
    if final:
        failed_q = [str(row["pair_id"]) for row in cases if row["cohort"] == final["cohort"] and row["joint_violation"]]
        conclusion = (f"Selected configuration {selected_name}: {final['rough_arrivals']}/12 rough arrivals, {final['physical_passes']}/12 sampled physical passes, "
            f"{final['both_passes']}/12 both and {final['execution_failures']} execution errors. "
            f"{final['timeouts']}/12 episodes timed out, so rough task performance remains weak. "
            f"Joint-position violating cases: {', '.join(failed_q) or 'none'}; maximum excess {number(final['maximum_joint_excess_rad'])} rad. "
            f"Speed/torque violating episodes are {final['velocity_violation_cases']}/{final['effort_violation_cases']}. "
            "These bounded, sampled results do not establish general physical qualification. The study froze this configuration before a separate 100-additional-task validation, retaining any known development failures. That validation is separate from these development comparisons and will not feed back into parameter selection.")
        if not reserved or not reserved["info"]["recorded_cases"]:
            conclusion += " No reserved-88 evaluation was executed."
        lines[2:2] = [conclusion, ""]
        selected_info = next(item for item in info if item['name'] == selected_name)
        if selected_info.get('final_configuration_freeze'):
            freeze = selected_info['final_configuration_freeze']['record']
            lines[4:4] = [f"The selection was frozen at {freeze['created_utc']}, before additional validation. "
                f"Minimum sampled joint-bound clearance was {number(freeze['minimum_actual_joint_clearance_rad'])} rad; "
                f"maximum physical speed and torque ratios were {number(freeze['max_velocity_ratio'])} and {number(freeze['max_effort_ratio'])}. "
                "No additional-100 validation outcomes are included here. final_configuration_freeze.json retains the selection record and its limitations.", ""]
    for item in info:
        pre = item["preflight"]
        prelabel = "pass" if pre and pre.get("preflight_pass") else "FAIL" if pre else "not included"
        lines.append(f"| {short(item['name'])} | {item['status']} | {item['recorded_cases']}/12 | {prelabel} |")
    for item in info:
        if item["evaluation_hold"]:
            lines += ["", short(item["name"]) + " reserved-88 evaluation held: " + item["evaluation_hold"]["reason"] + "."]
    preflights = preflight_rows(info)
    if preflights:
        lines += ["", "## Canonical preflight outcomes", "",
            "These single canonical episodes are separate from the twelve-task development pilots. Arrival and sampled physical compliance remain independent checks. Motion is the Cartesian distance between the initial and final executed end-effector positions, not traveled path length or reduction in goal error. The exact arrival times, work and metric hashes are retained in preflights.csv.", "",
            "| Config | mu | Outcome | Arrival | Physical | Final error mm | Motion mm |", "|---|---:|---|---|---|---:|---:|"]
        for row in preflights:
            lines.append(f"| {short(row['cohort'])} | {row['mu']:g} | {row['outcome']} | {'yes' if row['task_success'] else 'no'} | {'pass' if row['physical_pass'] else 'FAIL'} | {number(row['final_position_mm'])} | {number(row['endpoint_displacement_mm'])} |")
        lines += ["", "| Config | Observed s | Feedback solves | Zero-PCG outer exits | Total SQP iterations | Pilot |", "|---|---:|---:|---:|---:|---|"]
        for row in preflights:
            pilot = "HELD; no pilot calls" if row["pilot_status"] == "held_preflight_failure" else row["pilot_status"]
            lines.append(f"| {short(row['cohort'])} | {row['observed_seconds']:g} | {row['feedback_solves']} | {row['zero_pcg_exit_solves']} | {row['actual_sqp_sum']} | {pilot} |")
        held = [r for r in preflights if r["pilot_status"] == "held_preflight_failure"]
        if held:
            lines += ["", "Failed matching preflights hold the corresponding development pilots: no twelve-task pilot solves were launched for " + ", ".join(short(r["cohort"]) for r in held) + ". The finite timeout trajectories, physical passes and original failure labels remain retained. For these configurations, stronger interior penalties can compete with soft pose-goal costs throughout the admissible region, favoring motion toward a lower-penalty pose or an offset goal error. The scalar cost scales explain this possible tradeoff; these combined parameter changes do not isolate one cause of the observed motion. A larger mu changes the merit tradeoff and did not restore canonical arrival in the recorded M2/M3 contrast. Natural zero-PCG stopping is preserved; these task failures do not call for a KKT gate or automatic iteration extension."]
    lines += ["", "## Complete-cohort outcomes", "",
        "Rough arrival requires position error <=50 mm, tool-axis error <=30 degrees and joint-speed L1 <=1 rad/s at a feedback boundary, with zero dwell. Full orientation about the tool axis is not constrained by this arrival test. A physical pass independently requires sampled executed motion to respect raw URDF joint, speed and torque bounds with the original numerical tolerances (q excess <=1e-6 rad, ratios <=1+1e-6). Torque limits remain 43 Nm for joints 1/2 and 26 Nm for joints 3-7.", "",
        "| Cohort | Arrival /12 | Physical /12 | Both /12 | Timeout | Exec. failures |", "|---|---:|---:|---:|---:|---:|"]
    for r in counts:
        lines.append(f"| {short(r['cohort'])} | {r['rough_arrivals']} | {r['physical_passes']} | {r['both_passes']} | {r['timeouts']} | {r['execution_failures']} |")
    lines += ["", "![Cumulative arrivals use all twelve assigned cases as the denominator. Timeouts and execution failures remain in the denominator and never appear as arrivals. Solve latency is unpaid in mode A.](arrival_ecdf.png)", "",
        "| Cohort | q fail / max rad | Speed fail / max ratio | Torque fail / max ratio |", "|---|---:|---:|---:|"]
    for r in counts:
        lines.append(f"| {short(r['cohort'])} | {r['joint_violation_cases']}/{number(r['maximum_joint_excess_rad'])} | {r['velocity_violation_cases']}/{number(r['maximum_velocity_ratio'])} | {r['effort_violation_cases']}/{number(r['maximum_effort_ratio'])} |")
    lines += ["", "Counts are episodes violating each category; categories overlap. Null means a full-history maximum is nonfinite or unavailable, never zero. Original per-case nulls and failure classes remain in cases.csv.", "",
        "![Episode maxima and cohort maxima use physical bounds, including all early transients. Interior cost bounds do not replace the physical validation limits. Null cohort maxima remain labeled rather than being computed by discarding nonfinite episodes.](physical_maxima.png)", "",
        "## Protocol and changed parameters", "",
        "The same frozen first twelve historical tasks, initial states and FK pose goals are paired across cohorts. This is B1/N32 with 30 ms knot spacing and feedback, 0.93 s prediction, a 3 ms RK4 plant, an 8 s timeout, and mode A (controller_dt): plant time advances independently of solve latency. Cold rest-state/zero-control seeds, production warm shifts, once-per-episode dual/rho reset, retained adaptive rho and final approach off remain explicit. The first solve is recorded and applied. The public-mu1 continuation reference is an isolated candidate and not a production merge.", "",
        "| Cohort | mu | qd cost | q/v/u limit costs | Interior q/v/u bounds |", "|---|---:|---:|---|---|"]
    progress_path = output / 'selected_endpoint_progress.csv'
    if selected_name and progress_path.is_file():
        progress = csv_read(progress_path); timeouts = [r for r in progress if r['outcome']=='timeout']
        failures = [sum(r[key]=='0' for r in timeouts) for key in ('endpoint_position_gate','endpoint_axis_gate','endpoint_speed_gate')]
        zero = sum(r['tail_zero_pcg_exits']==r['tail_solves'] and int(r['tail_solves'])>0 for r in timeouts)
        caps = [r['pair_id'] for r in timeouts if r['tail_budget_exhausted']==r['tail_solves'] and int(r['tail_solves'])>0]
        block = ['## Selected development endpoints and progress', '',
            f"All {len(timeouts)} selected-cohort timeouts are retained: {failures[0]} ended outside the position tolerance, {failures[1]} outside the tool-axis tolerance and {failures[2]} outside the speed-L1 tolerance. These endpoint categories overlap. "
            'They describe why the saved endpoints missed the rough arrival test; they do not prove that no earlier dense sample touched a goal region.', '',
            f"The last ten feedback solves all used the natural zero-PCG outer exit in {zero}/{len(timeouts)} timeouts. "
            f"All ten instead exhausted the SQP budget for IDs {', '.join(caps) or 'none'}. "
            'Zero-PCG is the practical stopping heuristic, not a formal stationarity certificate. These are different work patterns; the data do not establish that every timeout is locally settled.', '',
            '| ID | Result | End mm | Axis deg | Speed L1 | Last1s mm/deg | Zero/last10 | Cap/last10 |', '|---|---|---:|---:|---:|---:|---:|---:|']
        for r in progress:
            cell = lambda key: number(float(r[key])) if r[key] else 'null'
            block.append(f"| {r['pair_id']} | {r['outcome']} | {cell('final_position_mm')} | {cell('final_tool_axis_deg')} | {cell('final_speed_L1_rad_s')} | "
                f"{cell('last_second_position_error_reduction_mm')}/{cell('last_second_axis_error_reduction_deg')} | {r['tail_zero_pcg_exits']}/{r['tail_solves']} | {r['tail_budget_exhausted']}/{r['tail_solves']} |")
        block += ['', 'Last1s entries are net position/tool-axis error reductions over the saved final observation window (approximately one second); positive means improvement. The CSV retains exact observation durations and joint total variation. Arrival ends recording, so the arrived episode supplies no post-arrival settling evidence.', '',
            'The eight-second cutoff is not the sole plausible explanation. For example, ID7093 ends far from both pose tolerances with very low speed and little final-second progress, whereas ID9651 still reduces position and axis error. More time could help some episodes, but eventual arrival does not follow from these records.', '',
            'Interior limit penalties act throughout the interval and can bias posture, while velocity regularization discourages motion. They compete with the soft pose goal rather than enforce arrival. The finite 0.93 s prediction window can also affect local progress and braking tradeoffs. These are plausible contributors, not isolated causal results; no new horizon comparison, solver change, precision gate or tuning decision is inferred here. selected_endpoint_progress.csv contains all twelve selected episodes and uses only saved metrics, dense axis-error observations and last-ten-solve work.', '']
        index = lines.index('## Protocol and changed parameters'); lines[index:index] = block
    for item in info:
        p = item["protocol"]
        if not p:
            continue
        s = p["solver_params"]; b = p.get("limit_penalty", {}).get("cost_bounds")
        bounds = (f"{b['joint_margin_rad']} rad / {b['velocity_fraction_of_physical_limit']:.0%} / {b['effort_fraction_of_physical_limit']:.0%}" if b else "raw physical bounds")
        lines.append(f"| {short(item['name'])} | {s['mu']} | {s['qd_cost']} | {s['q_lim_cost']}/{s['vel_lim_cost']}/{s['ctrl_lim_cost']} | {bounds} |")
    lines += ["", "The following contrasts describe recorded configurations where measured; pending configurations remain proposals identified in the status table."]
    previous = None
    for item in info:
        p = item["protocol"]
        if not p:
            continue
        if previous:
            old = previous["protocol"]
            changes = [f"{key}: {old['solver_params'].get(key)} to {value}"
                for key, value in p["solver_params"].items() if value != old["solver_params"].get(key)]
            if p["limit_penalty"].get("cost_bounds") != old["limit_penalty"].get("cost_bounds"):
                changes.append("interior cost bounds changed")
            else:
                changes.append("interior cost bounds unchanged")
            lines += ["", short(item["name"]) + " versus " + short(previous["name"]) + ": " + "; ".join(changes) + "."]
        previous = item
    lines += ["", "Interior q bounds subtract the recorded joint margin from each side; speed and torque fractions scale the physical bounds for penalty evaluation only. Initial states and targets are retained without clipping or filtering. Soft penalties do not guarantee either interior or physical compliance. Exact complete configurations and numerical/module/source identities are embedded in metadata.json.", "",
        "The SQP cap remains 1000 and PCG cap 120. Natural zero-PCG outer exits remain active as a practical stopping heuristic. An ordinary PCG solve ending after nonzero iterations is a different event. No KKT gate or high-precision residual requirement is applied; no stationarity certificate is claimed.", "",
        "## First future knot versus executed prefix", "",
        "For every saved solve with a full 30 ms executed prefix, the returned first future state x[1] is compared with the saved RK4 plant endpoint at the same time. Each sample records max_j |q_pred[j]-q_exec[j]| in rad and max_j |v_pred[j]-v_exec[j]| in rad/s. A shortened final 20 ms timeout prefix is excluded because its endpoint time differs from the first future knot. Missing endpoints and nonfinite predictions/executions remain labeled in prefix_mismatch.csv.", "",
        "This measures a composite of model representation, optimizer dynamics feasibility and numerical integration. It does not isolate a dynamics defect or demonstrate a cause of reaching failure. The following percentiles pool matched solves descriptively; episodes with longer durations contribute more solves.", "",
        "| Cohort | Matched / other | q p95 / max rad | v p95 / max rad/s |", "|---|---:|---:|---:|"]
    for r in counts:
        lines.append(f"| {short(r['cohort'])} | {r['mismatch_matched']}/{r['mismatch_unmatched']} | {number(r['q_prefix_mismatch_p95_rad'])}/{number(r['q_prefix_mismatch_max_rad'])} | {number(r['v_prefix_mismatch_p95_rad_s'])}/{number(r['v_prefix_mismatch_max_rad_s'])} |")
    lines += ["", "## Per-case outcomes", "",
        "All cases in each complete cohort follow. P/F denotes the original sampled physical check; q is maximum executed joint excess, v/u are speed/torque ratios. Saved execution failures retain partial trajectories and nullable extrema. Arrival stops observation and therefore gives no post-arrival settling evidence."]
    for r in counts:
        name = r["cohort"]
        lines += ["", "### " + name, "", "| ID | Outcome | Phys. | End mm | q rad | v ratio | u ratio |", "|---|---|---|---:|---:|---:|---:|"]
        for c in (c for c in cases if c["cohort"] == name):
            lines.append(f"| {c['pair_id']} | {c['outcome']} | {'P' if c['physical_pass'] else 'F'} | {number(c['final_position_mm'], '.3g')} | {number(c['maximum_joint_excess_rad'], '.3g')} | {number(c['maximum_velocity_ratio'], '.3g')} | {number(c['maximum_effort_ratio'], '.3g')} |")
    lines += ["", "## Provenance and reproduction", "",
        "This export analyzes saved records only and imports no native solver. It does not alter raw data, rescue failures or run additional experiments. Whole-arm collision clearance, continuous-time physical safety, deployment timing and universal reaching reliability are outside this comparison. Compact editable data are cohorts.csv, cases.csv, case_status.csv and prefix_mismatch.csv; all full histories and horizons remain under the retained raw roots.", ""]
    campaign = next((item for item in info if item["name"] != REFERENCE), None)
    if campaign:
        lines += ["Campaign raw root: `" + str(Path(campaign["raw_root"]).parent.parent) + "`.", ""]
    lines += ["Continuation reference raw root: `" + info[0]["raw_root"] + "`.", ""]
    lines += ["The following compact identities show twelve-character hash prefixes. A row refers to its measured pilot when available, otherwise its canonical preflight. Full protocol IDs, all full hashes and exact raw roots are retained in metadata.json; SEARCH_EVIDENCE.json additionally records screen stages.", "",
        "| Cohort | Stage | Numerical base | Source commit | Native SHA256 | Protocol SHA256 |", "|---|---|---|---|---|---|"]
    for item in info:
        p = item["protocol"]
        v = item["provenance"] or item["preflight_provenance"]
        hashes = item["hashes"] if item["provenance"] else item["preflight_hashes"]
        if p and v:
            phase = "pilot" if item["provenance"] else "preflight"
            lines.append(f"| {short(item['name'])} | {phase} | {p['source_base'][:12]} | {v['source_commit'][:12]} | {v['module_sha256'][:12]} | {hashes.get('protocol.json', '')[:12]} |")
    lines += ["Exact analysis invocation and any retained execution-command records are in metadata.json. Command files are copied verbatim when present; absent launch commands are explicitly reported as absent rather than inferred.", "",
        "Regenerate figures from the compact exported data, then rebuild this self-contained PDF:", "", "```sh",
        "python analyze_margins.py --plot-only .", "python report_to_pdf.py --report REPORT.md", "```", ""]
    if reserved:
        evaluation = reserved["info"]
        block = ["## " + short(reserved["selected_cohort"]) + " reserved-task evaluation", "",
            "The remaining 88 tasks follow the exact frozen historical 100-task order after the twelve development tasks. They were reserved from new tuning but were historically seen in BA. Any recorded evaluation is checked against the selected measured development configuration, native module, numerical source bytes, wrapper, input-pool hashes, physical gates and stopping behavior. No tuning on these 88 outcomes is part of this comparison. The combined 100-task rate is not a matched comparison against the twelve-case reference.", "",
            f"Reserved evaluation status: {evaluation['status']}; {evaluation['recorded_cases']}/88 recorded outcomes. Development and reserved case-status labels remain in evaluation_case_status.csv, including failures and unrecorded tasks. Raw evaluation root: `{evaluation['raw_root']}`.", ""]
        if reserved["hold_record"]:
            block += ["The reserved evaluation is held: " + reserved["hold_record"]["reason"] + ". No reserved-task outcomes are claimed.", ""]
        if reserved["populations"]:
            block += ["| Population | Assigned | Arrivals | Physical | Both | Timeouts | Exec. failures |", "|---|---:|---:|---:|---:|---:|---:|"]
            for row in reserved["populations"]:
                block.append(f"| {row['cohort']} | {row['assigned']} | {row['rough_arrivals']} | {row['physical_passes']} | {row['both_passes']} | {row['timeouts']} | {row['execution_failures']} |")
            block += ["", "| Population | q fail / max rad | Speed fail / max ratio | Torque fail / max ratio |", "|---|---:|---:|---:|"]
            for row in reserved["populations"]:
                block.append(f"| {row['cohort']} | {row['joint_violation_cases']}/{number(row['maximum_joint_excess_rad'])} | {row['velocity_violation_cases']}/{number(row['maximum_velocity_ratio'])} | {row['effort_violation_cases']}/{number(row['maximum_effort_ratio'])} |")
            block += ["", "![Each curve uses its fixed assigned denominator: development 12, reserved 88, combined 100. All timeouts and execution failures remain in that denominator. The 88-task evaluation uses the already frozen selected settings and does not support a matched full-100 comparison with the twelve-case reference.](evaluation_population_arrivals.png)", "",
                "evaluation_all_cases.csv retains every one of the 100 case outcomes in the exact frozen order, including physical extrema, arrival times and failure labels. evaluation_population_counts.csv contains the editable population summaries. No incomplete episode or execution failure is filtered, retried or converted into success; nonfinite full-history extrema remain null. Native/module/protocol and per-record hashes are retained in metadata.json. These observations do not establish a universal reaching or physical-compliance guarantee.", ""]
        else:
            block += ["Population outcome summaries and the 100-case arrival plot are exported only after the complete 88-task summary is available. This preserves a visible distinction between progress status and completed evidence.", ""]
        if not evaluation['recorded_cases']:
            block = ["## Historical unused 88-task plan", "",
                f"The earlier {short(reserved['selected_cohort'])} reserved evaluation remains {evaluation['status']}, with 0/88 outcomes. "
                "These tasks were the remainder of the historically studied 100, after its twelve development cases. No reserved-88 experiment was launched. "
                "The original hold record, exact roster, raw path and case-status labels remain in metadata.json and evaluation_case_status.csv. "
                "This historical plan is separate from the fresh 100 additional validation tasks selected after development tuning; no fresh-100 outcomes enter this report.", ""]
        index = lines.index("## Protocol and changed parameters")
        lines[index:index] = block
    if search:
        stages = [row for row in search["stages"] if int(row["configuration"].split("-", 1)[0][1:]) >= 6
            and row["stage"] in ("screen", "pilot")]
        if stages:
            canonical = {row["configuration"]: row for row in search["stages"] if row["stage"] == "preflight"}
            block = ["## Physical-first penalty search", "",
                "The later search first sought physical compliance on the fixed development task2342, allowing poor or stalled reaching during the screen. A finite, physically compliant canonical timeout was permitted only for this explicitly requested one-case screen. It remains a failed canonical gate; this exception does not qualify a full twelve-task baseline. Main pilot counts above contain no substituted screen outcomes.", "",
                "The table retains actual stage scope: screens assign one task, full pilots assign twelve. Counts refer to recorded outcomes; a partial or held stage does not turn its unexecuted cases into failures. Penalty response need not be monotone, so these observations are not a guaranteed mathematical bracket.", "",
                "| Config | Stage/status | Done/N | Inset rad | q/v weights | Canonical | Arrivals | Physical | Both |", "|---|---|---:|---:|---:|---|---:|---:|---:|"]
            for row in stages:
                preflight = canonical.get(row["configuration"])
                gate = "pass" if preflight and preflight["preflight_pass"] else "FAIL" if preflight else "unknown"
                block.append(f"| {short(row['configuration'])} | {row['stage']}/{row['status']} | {row['recorded']}/{row['assigned']} | {row['q_margin_rad']:g} | {row['q_limit_weight']:g}/{row['velocity_limit_weight']:g} | {gate} | {row['rough_arrivals']} | {row['physical_passes']} | {row['both']} |")
            block += ["", "All-limit compliance includes joint position, speed and torque. A q-only pass with a speed failure is retained as a physical failure. The following extrema are for single-case screens; their raw histories and any execution errors remain preserved.", "",
                "| Screen | q excess rad | Speed ratio | Torque ratio | First recorded failure |", "|---|---:|---:|---:|---|"]
            for case in search["cases"]:
                if case["stage"] != "screen":
                    continue
                failures = []
                for category, label in (("joint", "q"), ("velocity", "v"), ("effort", "u")):
                    if not case[category + "_pass"]:
                        failures.append(label + ": " + str(case[category + "_first_joint"]) + " @ " + number(case[category + "_first_time_s"]) + " s")
                block.append(f"| {short(case['configuration'])}, ID{case['pair_id']} | {number(case['maximum_q_excess_rad'])} | {number(case['maximum_speed_ratio'])} | {number(case['maximum_torque_ratio'])} | {'; '.join(failures) or 'none'} |")
            if "SEARCH_TRADEOFF_2342.png" in search["files"]:
                block += ["", "![Fixed representative traces: M8 is task2342 within a twelve-task pilot; M9/M10/M11 are one-task screens. Every saved 3 ms sample is retained. J1 signed clearance uses actual physical bounds; the speed ratio is the maximum over seven joints. Both penalty weights and stage scope are labeled. These observations establish no solver-cause claim.](SEARCH_TRADEOFF_2342.png)", ""]
            block += ["SEARCH_STAGES.csv and SEARCH_CASES.csv retain all canonical/screen/pilot rows and original failure labels. SEARCH_EVIDENCE.json preserves exact protocol/source/module/wrapper/input identities and summary/case hashes. The dense trace CSV/JSON and plotting script remain editable companions; the figure is a fixed four-example illustration rather than every search point.", ""]
            index = lines.index("## Protocol and changed parameters")
            lines[index:index] = block
    if "cost_scaling" in rationale:
        scale_rows = cost_scale_table(info, csv_read(output / "cost_scaling.csv"))
        block = ["## Physical meaning of the penalty scales", "",
            "The C2 helper continues each logarithmic bound penalty quadratically when its distance to the cost bound falls below 1% of that cost interval. It has finite values and derivatives beyond the cost bound. This continuation width is distinct from the joint inset or speed/torque fraction separating cost bounds from physical bounds; neither parameter turns a soft cost into a hard constraint.", "",
            "The following CPU scalar calculations evaluate the actual C2 helper at representative points using matching interior bounds and weights. Costs are for one scalar at one running knot; the native objective sums knots without dt scaling. Limit costs are measured relative to the same interval center, removing the arbitrary log constant. Derivatives are with respect to that row's variable: rad/s for velocity, Nm for torque and rad for joint position. Their numerical sizes cannot be compared across those different units.", "",
            "| Settings | Variable / point | Limit cost | Ordinary cost | Limit gradient | Curvature |", "|---|---|---:|---:|---:|---:|"]
        for row in scale_rows:
            label = "/".join(row["cohorts"])
            point = {"velocity": "v=2 rad/s", "effort": "J2 initial max abs gravity", "joint": "J2 raw upper q"}[row["category"]]
            values = [number(float(row[key])) if row[key] else "n/a" for key in
                ("relative_limit_cost", "ordinary_cost", "limit_gradient", "limit_curvature")]
            block.append(f"| {label} | {point} | " + " | ".join(values) + " |")
        block += ["", "Velocity regularization is 0.5*qd_cost*v^2 and effort regularization is 0.5*u_cost*u^2. The parameter q_cost weights Cartesian position error in meters, so no corresponding ordinary joint-angle quadratic is shown. The complete editable table includes finite cost increments, additional points and proposed wider margins in cost_scaling.csv; those unexecuted proposals are not experimental outcomes.", "",
            "Increasing mu changes the merit weight on the sum of absolute, unnormalized dynamics and initial-state defects, rather than the running objective weights in the table. For example, a single 0.1 rad/s velocity defect contributes 0.1 at mu=1 and 1 at mu=10. A 0.001 rad position defect contributes 0.001 and 0.01, respectively. The merit sum mixes units and accumulates across knots; these examples illustrate scale, not an exact tradeoff or enforcement guarantee. M3's running-cost rows match M2 because its isolated configured change is mu.", "",
            "These are arithmetic comparisons for understanding the chosen settings, not a native solver qualification or physical certificate. Exact helper/URDF hashes and calculation definitions are retained in cost_scaling.json; the editable CPU script is cost_scaling.py.", ""]
        if "velocity_cost_curve.png" in rationale["cost_scaling"]["files"]:
            block += ["![One-joint running velocity cost combines the relative C2 limit penalty and ordinary quadratic regularization at the actual M0/M1/M4/M5 settings. Physical and interior cost bounds are marked, with a low-speed inset showing that the penalty also changes the cost inside the limits. These are scalar cost curves, not trajectories or physical guarantees.](velocity_cost_curve.png)", ""]
        lines[lines.index("## First future knot versus executed prefix"):lines.index("## First future knot versus executed prefix")] = block
    if "physical_scales" in rationale:
        physics = rationale["physical_scales"]["summary"]
        acceleration = csv_read(output / "acceleration_capacities.csv")
        headroom = csv_read(output / "gravity_headroom.csv")
        block = ["## Back-of-envelope physical scales", "",
            "The unchanged URDF mass matrices and gravity torques were evaluated on the CPU at the twelve frozen starts and twelve goal-witness postures. Each frozen-posture acceleration range sets qdd=a*e_j, holds all other joint accelerations at zero and intersects all seven inequalities |M(q)*qdd+g(q)| <= effort_fraction*physical_limit. The weaker acceleration direction gives the constant-deceleration estimate t_stop=v/a and d_stop=v^2/(2*a). This includes inertial coupling at that posture but excludes changing posture, velocity/Coriolis effects, friction, actuator dynamics and simultaneous joint motion. It is a scale estimate, not a full-path feasibility or safety certificate.", "",
            "The following ranges span all 24 postures. Acceleration is rad/s^2; estimated stopping distance starts at 2.5 rad/s and is in rad. The 85% effort rows describe a wider-margin proposal unless its cohort is explicitly recorded above.", "",
            "| Joint | a range, 90% effort | Stop range, 90% | a range, 85% effort | Stop range, 85% |", "|---|---:|---:|---:|---:|"]
        clearances = physics.get("sampling_bound_clearances")
        if clearances:
            clearance_lines = ["The frozen pool's sampling bounds are inset 0.12 rad from the actual URDF bounds. Small clearances to that sampling box must not be called small physical clearances. The following minima use raw physical URDF bounds; every listed start and goal witness also lies inside a 0.10 rad URDF cost inset. This input screening does not certify the optimized or executed motion.", "",
                "| Assigned population | Min start clearance rad | Min goal-witness clearance rad |", "|---|---:|---:|"]
            for name, population in clearances["populations"].items():
                start = population["start"]["physical_URDF"]["minimum_inset_rad"]
                goal = population["goal_witness"]["physical_URDF"]["minimum_inset_rad"]
                clearance_lines.append(f"| {name} | {start:.6g} | {goal:.6g} |")
            clearance_lines += [""]
            block[2:2] = clearance_lines
        for joint in range(1, 8):
            values = []
            for fraction in (.9, .85):
                rows = [r for r in acceleration if int(r["joint"]) == joint and np.isclose(float(r["effort_fraction"]), fraction)]
                for key in ("weaker_direction_acceleration_rad_s2", "braking_distance_rad_at_2p5"):
                    numbers = [float(r[key]) for r in rows]
                    values.append(number(min(numbers), ".3g") + " to " + number(max(numbers), ".3g"))
            block.append(f"| J{joint} | " + " | ".join(values) + " |")
        margin = physics["back_of_envelope"]
        minimum_headroom = min(float(r["gravity_only_headroom_Nm"]) for r in headroom if np.isclose(float(r["effort_fraction"]), .85))
        block += ["", "At 2.5 rad/s, one 30 ms prefix covers 0.075 rad even before acceleration is considered. The following back-of-envelope inset scales compare travel time and the constant deceleration needed to stop within the inset. These are not generic stopping-distance bounds: posture-dependent weaker-direction capacity can be smaller, and the assumptions above omit moving-state effects.", "",
            "| q inset rad | Travel ms at 2.5 rad/s | Required stop a, rad/s^2 | Assigned configurations |", "|---|---:|---:|---|"]
        for inset, travel, required in zip(margin["qmargin_rad"], margin["travel_time_ms_at_2p5"], margin["required_stop_acceleration_at_2p5_rad_s2"]):
            labels = [short(item["name"]) for item in info if item["protocol"] and np.isclose(
                item["protocol"].get("limit_penalty", {}).get("cost_bounds", {}).get("joint_margin_rad", -1), inset)]
            block.append(f"| {inset:g} | {travel:g} | {required:.4g} | {('/'.join(labels) or 'scale only; no assigned cohort')} |")
        block += ["", "Assigned configurations may still be pending; their execution status remains in the first table.", "",
            f"Speed headroom of {margin['velocity_headroom_rad_s'][0]} or {margin['velocity_headroom_rad_s'][1]} rad/s corresponds to only {margin['acceleration_over_30ms_rad_s2'][0]:.3g} or {margin['acceleration_over_30ms_rad_s2'][1]:.3g} rad/s^2 sustained for 30 ms. Interior speed limits therefore create an incentive to leave room, rather than guaranteeing that the next executed prefix respects the physical speed bound. At 85% effort, the smallest gravity-only headroom across the retained postures/joints is {minimum_headroom:.3g} Nm; this does not bound dynamic torque demand.", "",
            "For two selected saved M0 prefixes, the following CPU frozen-acceleration analogue uses ABA at the actual feedback state. It is compared with the returned first future knot and the already saved RK4 endpoint; no native kernel evaluation or new plant replay is performed. These examples describe component sizes and do not establish a cause for all observed errors.", "",
            "| Task / step | Returned vs CPU map, max v | CPU map vs saved RK4, max v | Returned vs RK4, max v |", "|---|---:|---:|---:|"]
        for prefix in physics["saved_prefixes"]:
            differences = prefix["differences"]
            vals = [number(differences[key]["maximum_v_difference_rad_s"]) for key in
                ("returned_vs_frozen_ABA_map", "frozen_ABA_map_vs_saved_RK4", "returned_vs_saved_RK4")]
            block.append(f"| {prefix['pair_id']} / {prefix['step']} | " + " | ".join(vals) + " |")
        block += ["", "All velocity differences in this table are rad/s. Full posture matrices, gravity torques, assumptions, raw-record hashes and software identities remain in physical_scales.json. Editable per-joint capacity/headroom data and the CPU calculation script are exported beside this report. These estimates motivate bounded penalty/margin comparisons without replacing the unchanged physical checks.", ""]
        index = lines.index("## First future knot versus executed prefix")
        lines[index:index] = block
    if "joint_tradeoffs" in rationale:
        data = rationale["joint_tradeoffs"]["summary"]
        block = ["## Larger joint insets with lower joint-limit weights", "",
            "This focused follow-up varies the joint cost inset and q-limit weight while keeping the other M5 settings fixed. The table is CPU scalar/local pose-gradient accounting on existing development inputs, not rollout outcomes or a scheduled grid. Cohort labels below identify measured or prepared configurations; unmatched rows remain conditional proposals.", "",
            "Gradients are objective cost per radian, not mechanical torque. Counts outside the soft interval refer to the twelve frozen random starts and twelve FK goal-witness postures. These unchanged inputs are retained without clipping or filtering. Being outside a soft cost interval neither changes the raw physical bounds nor makes the soft-cost problem infeasible.", "",
            "| Inset / weight | Cohort / state | Starts/goals outside | J1 bound gradient | Canonical J4 | Task2342 J4 |", "|---|---|---:|---:|---:|---:|"]
        for row in data["summary"]:
            labels = []
            for item in info:
                p = item["protocol"]
                if not p:
                    continue
                inset = p.get("limit_penalty", {}).get("cost_bounds", {}).get("joint_margin_rad", -1)
                if np.isclose(inset, row["joint_margin_rad"]) and np.isclose(p["solver_params"]["q_lim_cost"], row["q_lim_cost"]):
                    state = {"complete": "complete", "partial": "partial", "not_recorded": "prepared", "held_preflight_failure": "held"}.get(item["status"], item["status"])
                    labels.append(short(item["name"]) + "/" + state)
            block.append(f"| {row['joint_margin_rad']:g}/{row['q_lim_cost']:g} | {', '.join(labels) or 'conditional only'} | "
                f"{row['start_outside_cost_interval_postures']}/{row['goal_witness_outside_cost_interval_postures']} | "
                f"{number(row['J1_physical_upper_gradient'])} | {number(row['canonical_q4_gradient'])} | {number(row['task2342_start_q4_gradient'])} |")
        block += ["", "A larger inset starts discouraging boundary proximity earlier; a smaller weight reduces the penalty throughout the interval. Those changes can counteract or reinforce each other locally. For example, the canonical J4 starts only 0.093461 rad above its actual lower bound, so increasing the inset from 0.10 to 0.15 rad places it farther outside the soft interval. Its joint-limit gradient grows in magnitude from about 0.970 to 1.133 despite lowering the weight from 0.015 to 0.010. At J1's physical upper bound, the same change reduces the gradient from about 1.184 to 1.019. Earlier onset therefore does not imply a stronger slope at every physically relevant point.", "",
            "The canonical running-pose gradient component for J4 is about 0.01420 cost/rad. This local comparison omits terminal weighting, dynamics, feasible directions and the rest of the trajectory; it does not identify a cause of a timeout or prove a safe choice. The summary/scalar CSVs retain bulk gradients, C2 widths, all-joint points and full-vector comparisons. joint_margin_tradeoffs.json and the exported CPU script retain exact source/helper/URDF/protocol/module identities. No validation outcomes enter these calculations.", ""]
        index = lines.index("## First future knot versus executed prefix")
        lines[index:index] = block
    (output / "REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pdf-helper", type=Path)
    parser.add_argument("--previews", type=Path)
    parser.add_argument("--physics-root", type=Path, help="Optional saved CPU physical-scale calculation directory")
    parser.add_argument("--cost-scaling-root", type=Path, help="Optional scalar cost-table directory; defaults to source experiment directory")
    parser.add_argument("--joint-tradeoffs-root", type=Path, help="Optional saved development-only joint-margin arithmetic")
    parser.add_argument("--evaluation-cohort", help="Optional selected M* cohort for the reserved 88; otherwise discover its evaluation directory")
    parser.add_argument("--selected-cohort", help="Explicit final configuration chosen for this study; completed historical M5 data do not imply selection")
    parser.add_argument("--plot-only", type=Path)
    args = parser.parse_args()
    if args.plot_only:
        plots(args.plot_only.resolve()); return
    if any(getattr(args, key) is None for key in ("raw_root", "reference_root", "source", "output")):
        parser.error("Provide --raw-root --reference-root --source --output, or --plot-only")
    raw_root, reference_root, source, output = (getattr(args, key).resolve() for key in ("raw_root", "reference_root", "source", "output"))
    if output.is_relative_to(raw_root) or output.is_relative_to(reference_root):
        raise ValueError("Compact export must be outside retained raw data")
    reference_raw = find_run(reference_root)
    protocol = read(reference_raw / "protocol.json")
    expected = protocol["development_task_ids"]
    ref = collect_cohort(REFERENCE, reference_root, protocol, expected, None)
    if ref[0]["status"] != "complete":
        raise ValueError("Reference cohort must be complete")
    if sha(source / MODEL) != ref[0]["provenance"]["source_sha256"][MODEL]:
        raise ValueError("Analysis URDF differs from measured reference")
    info, status_rows, cases, mismatches = [ref[0]], ref[1], ref[2], ref[3]
    candidates = {p.stem: read(p) for p in (source / "experiments/random_reaching_baseline/protocols").glob("M*.json")}
    if raw_root.exists():
        for root in raw_root.iterdir():
            if root.is_dir() and root.name.startswith("M"):
                candidates.setdefault(root.name, None)
    for name, fallback in sorted(candidates.items(), key=lambda item: int(item[0].split("-", 1)[0][1:])):
        result = collect_cohort(name, raw_root / name, fallback, expected, ref[4])
        item = result[0]
        if item["protocol"]:
            validate_protocol(item["protocol"], protocol, name)
        if item["preflight_provenance"]:
            validate_protocol(item["preflight_provenance"]["protocol"], protocol, name + "/preflight")
        measured = item["provenance"] or item["preflight_provenance"]
        if measured:
            measured_model = measured["source_sha256"][MODEL]
            if measured_model != ref[0]["provenance"]["source_sha256"][MODEL] or sha(source / MODEL) != measured_model:
                raise ValueError("Physical URDF identity changed: " + name)
            if measured["inputs_sha256"] != ref[0]["provenance"]["inputs_sha256"]:
                raise ValueError("Measured task input pool identity changed: " + name)
            if item["provenance"] and item["preflight_provenance"]:
                for key in ("source_sha256", "module_sha256", "wrapper_sha256"):
                    if item["preflight_provenance"][key] != item["provenance"][key]:
                        raise ValueError(f"Preflight/pilot identity changed: {name}/{key}")
        info.append(item); status_rows.extend(result[1]); cases.extend(result[2]); mismatches.extend(result[3])
    counts = aggregate(cases, mismatches)
    reserved = collect_reserved_evaluation(raw_root, info, cases, status_rows, expected, args.evaluation_cohort)
    output.mkdir(parents=True, exist_ok=True)
    csv_write(output / "cohorts.csv", counts)
    csv_write(output / "cases.csv", cases)
    csv_write(output / "case_status.csv", status_rows)
    preflights = preflight_rows(info)
    csv_write(output / "preflights.csv", preflights, list(preflights[0]) if preflights else ["cohort", "outcome"])
    csv_write(output / "prefix_mismatch.csv", mismatches, list(mismatches[0]) if mismatches else ["cohort", "pair_id", "step", "status"])
    if reserved:
        csv_write(output / "evaluation_population_counts.csv", reserved["populations"], list(counts[0]))
        csv_write(output / "evaluation_all_cases.csv", reserved["cases"], list(cases[0]))
        csv_write(output / "evaluation_case_status.csv", reserved["case_status"])
    rationale = collect_rationale(args, raw_root, source, output)
    search = collect_search(raw_root, source, output)
    if "cost_scaling" in rationale:
        scales = rationale["cost_scaling"]["summary"]
        if sha(output / "cost_scaling.csv") != scales["csv_sha256"]:
            raise ValueError("Cost-scale CSV differs from its retained identity")
    if "physical_scales" in rationale:
        physics = rationale["physical_scales"]["summary"]
        if (physics["assigned_task_ids"] != expected or
                physics["provenance"]["model_sha256"] != ref[0]["provenance"]["source_sha256"][MODEL]):
            raise ValueError("Physical-scale inputs differ from this paired study")
        if sha(output / "physical_scales.py") != physics["provenance"]["script_sha256"]:
            raise ValueError("Physical-scale script differs from its saved identity")
    if "joint_tradeoffs" in rationale:
        joint = rationale["joint_tradeoffs"]["summary"]
        if joint["provenance"]["model_sha256"] != ref[0]["provenance"]["source_sha256"][MODEL]:
            raise ValueError("Joint-margin arithmetic uses a different physical model")
        if joint["provenance"]["input_sha256"] != ref[0]["provenance"]["inputs_sha256"]:
            raise ValueError("Joint-margin arithmetic uses different frozen input data")
        if sha(output / "joint_margin_tradeoffs.py") != joint["provenance"]["script_sha256"]:
            raise ValueError("Joint-margin script differs from its saved identity")
    commands = {}
    for root in (raw_root, reference_root):
        for pattern in ("**/commands.txt", "**/launch_commands.json", "**/launch_commands.jsonl", "**/command*.txt", "**/command*.json"):
            for path in root.glob(pattern):
                commands[str(path)] = dict(sha256=sha(path), text=path.read_text())
    metadata = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        analysis_command=shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]),
        analysis_sha256=sha(__file__), raw_root=str(raw_root), reference_root=str(reference_root),
        source=str(source), launch_command_records=commands,
        launch_command_status="verbatim retained records included" if commands else "exact execution argv not recorded in available inputs",
        cohorts=info, physical_rationale=rationale, reserved_evaluation=reserved,
        search_artifacts=search["files"] if search else None)
    selected = next((item for item in info if item["name"] == args.selected_cohort and item["status"] == "complete"), None)
    if args.selected_cohort and selected is None:
        raise ValueError("Selected configuration does not have a complete development cohort")
    if selected:
        freeze_path = raw_root / selected['name'] / 'final_configuration_freeze.json'
        if freeze_path.is_file():
            recorded = read(freeze_path)
            for key, expected_value in (
                    ('selected_cohort', selected['name']),
                    ('protocol_sha256', selected['hashes']['protocol.json']),
                    ('development_summary_sha256', selected['hashes']['summary.json']),
                    ('module_sha256', selected['provenance']['module_sha256'])):
                if recorded[key] != expected_value:
                    raise ValueError('Final selection freeze differs from measured development: ' + key)
            shutil.copy2(freeze_path, output / freeze_path.name)
            selected['final_configuration_freeze'] = dict(source=str(freeze_path), sha256=sha(freeze_path), record=recorded)
        freeze = dict(selected_cohort=selected["name"], frozen_before_additional_validation=True,
            validation_role="100 additional random tasks; validation only, no parameter selection or retuning",
            protocol=selected["protocol"], provenance=selected["provenance"],
            development_summary_sha256=selected["hashes"]["summary.json"],
            development_outcomes=next(row for row in counts if row["cohort"] == selected["name"]),
            known_physical_failures=[row for row in cases if row["cohort"] == selected["name"] and not row["physical_pass"]])
        if selected.get('final_configuration_freeze'):
            freeze['recorded_selection_freeze'] = selected['final_configuration_freeze']
        write(output / "selected_configuration.json", freeze)
        metadata["selected_configuration"] = freeze
        progress = endpoint_progress(selected)
        csv_write(output / 'selected_endpoint_progress.csv', progress)
        metadata['selected_endpoint_progress'] = dict(sha256=sha(output / 'selected_endpoint_progress.csv'),
            source='Saved metrics/last_second, dense executed axis errors, final ten solves; original case raw hashes retained above.')
    else:
        metadata["selection_status"] = "No final configuration selected in this export; historical completed M5 is not an automatic choice."
    write(output / "metadata.json", metadata)
    plots(output); report(output, info, counts, cases, rationale, reserved, args.selected_cohort, search)
    if Path(__file__).resolve() != output / "analyze_margins.py":
        shutil.copy2(__file__, output / "analyze_margins.py")
    helper = args.pdf_helper or source / "experiments/random_reaching_baseline/report_to_pdf.py"
    if not helper.is_file():
        helper = Path("/workspace/GATO/thesis_results/reaching-landscape/report_to_pdf.py")
    if not helper.is_file():
        raise FileNotFoundError("Provide --pdf-helper for Markdown/PDF rendering")
    if helper.resolve() != output / "report_to_pdf.py":
        shutil.copy2(helper, output / "report_to_pdf.py")
    spec = importlib.util.spec_from_file_location("report_to_pdf", helper)
    renderer = importlib.util.module_from_spec(spec); spec.loader.exec_module(renderer)
    pages = renderer.render(output / "REPORT.md", args.previews)
    for note in ("COST_SCALING.md", "PHYSICAL_SCALES.md", "JOINT_MARGIN_TRADEOFFS.md"):
        if any(note in item["files"] for item in rationale.values()):
            renderer.render(output / note)
    print(json.dumps(dict(complete_cohorts=[r["cohort"] for r in counts], pdf_pages=pages,
        cohort_status={item["name"]: item["status"] for item in info}, output=str(output))))


if __name__ == "__main__":
    main()
