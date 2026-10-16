#!/usr/bin/env python3
"""Analyze the frozen additional-100 validation from saved records; CPU only.

No solver imports or execution. Incomplete runs require --allow-partial and an
output outside thesis_results. Plot-only regeneration uses compact CSV/JSON.
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
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COUNT = 100
MODEL = "gato/dynamics/tiago_right/tiago_right_arm.urdf"
RUNTIME = ("source_base", "timing_mode", "offline_timing", "N", "knot_dt_s",
    "feedback_s", "sim_dt_s", "timeout_s", "position_tolerance_m",
    "tool_axis_tolerance_deg", "velocity_l1_tolerance_rad_s", "goal_dwell_s",
    "final_approach", "batch", "solver_params", "reset_rho_each_solve",
    "initialization", "input_sha256", "limit_penalty", "native_module_sha256")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def value(x):
    return float(x) if x is not None and np.isfinite(x) else None


def clean(data):
    if isinstance(data, dict): return {k: clean(v) for k, v in data.items()}
    if isinstance(data, list): return [clean(v) for v in data]
    if isinstance(data, float): return value(data)
    return data


def write(path, data):
    Path(path).write_text(json.dumps(clean(data), indent=2, allow_nan=False) + "\n")


def csv_write(path, rows, fields=None):
    with Path(path).open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def csv_read(path):
    with Path(path).open() as handle: return list(csv.DictReader(handle))


def number(x, spec=".5g"):
    return format(float(x), spec) if value(x) is not None else "null"


def ids_sha(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def verify(raw, prep, source):
    assignment = read(prep / "assignment.json")
    protocol = read(prep / "validation_protocol.json")
    freeze = read(prep / "final_configuration_freeze.json")
    receipt = read(prep / "selection_receipt.json")
    launch = read(raw / "launch_receipt.json")
    prepared = read(prep / "PREPARED.json")
    provenance = read(raw / "pilot/provenance.json")
    ids = assignment["task_ids"]
    if (len(ids) != COUNT or len(set(ids)) != COUNT or
            set(ids).intersection(assignment["excluded_historical_task_ids"]) or
            len(assignment["excluded_historical_task_ids"]) != COUNT or
            ids_sha(ids) != assignment["task_ids_sha256"] or
            ids_sha(assignment["excluded_historical_task_ids"]) != assignment["excluded_historical_task_ids_sha256"]):
        raise ValueError("Assignment is not the frozen disjoint additional100")
    for key, filename in (("assignment_sha256", "assignment.json"),
            ("validation_protocol_sha256", "validation_protocol.json"),
            ("final_configuration_freeze_sha256", "final_configuration_freeze.json")):
        if receipt[key] != sha(prep / filename): raise ValueError("Selection receipt mismatch: " + key)
    for key, filename in (("assignment_sha256", "assignment.json"),
            ("protocol_sha256", "validation_protocol.json"),
            ("final_configuration_freeze_sha256", "final_configuration_freeze.json")):
        if launch[key] != sha(prep / filename): raise ValueError("Launch identity mismatch: " + key)
    if protocol != read(raw / "pilot/protocol.json") or protocol != provenance["protocol"]:
        raise ValueError("Measured validation protocol differs from its freeze")
    if (protocol["task_ids"] != ids or protocol["validation_task_ids"] != ids or
            provenance["pairs"] != ids or protocol["task_ids_sha256"] != ids_sha(ids) or
            protocol["validation_assignment_sha256"] != sha(prep / "assignment.json")):
        raise ValueError("Measured ordered assignment differs")
    for key in RUNTIME:
        if protocol[key] != freeze["protocol"][key]: raise ValueError("Selected runtime changed: " + key)
    if (receipt["selected_protocol_sha256"] != freeze["protocol_sha256"] or
            receipt["native_module_sha256"] != freeze["module_sha256"] or
            prepared["wrapper_sha256"] != provenance["wrapper_sha256"] or
            launch["wrapper_sha256"] != provenance["wrapper_sha256"] or
            launch["task_ids_sha256"] != ids_sha(ids) or
            provenance["module_sha256"] != freeze["module_sha256"] or
            launch["native_module_sha256"] != freeze["module_sha256"] or
            prepared["source_commit"] != provenance["source_commit"] or
            launch["source_commit_at_launch"] != provenance["source_commit"]):
        raise ValueError("Frozen source/module/worker identity mismatch")
    if sha(source / "experiments/random_reaching_baseline/run.py") != provenance["wrapper_sha256"]:
        raise ValueError("Saved worker differs from current read-only source")
    for name, digest in provenance["source_sha256"].items():
        if sha(source / name) != digest: raise ValueError("Measured numerical source changed: " + name)
    if sha(Path(provenance["module"])) != provenance["module_sha256"]:
        raise ValueError("Native artifact differs from its retained identity")
    if datetime.fromisoformat(freeze["created_utc"]) >= datetime.fromisoformat(launch["launch_utc"]):
        raise ValueError("Configuration freeze must precede validation launch")
    pre = protocol["reused_canonical_preflight"]
    prepath = Path(pre["path"]); prem = read(prepath)
    if sha(prepath) != pre["metrics_sha256"] or not prem["preflight_pass"]:
        raise ValueError("Reused selected canonical preflight differs or failed")
    preprov = read(prepath.parents[1] / "provenance.json")
    if preprov["module_sha256"] != provenance["module_sha256"] or preprov["source_sha256"] != provenance["source_sha256"]:
        raise ValueError("Reused canonical source/module differs")
    for key in RUNTIME:
        if preprov["protocol"][key] != protocol[key]: raise ValueError("Reused preflight runtime differs: " + key)
    pool = Path(provenance["pool"])
    for name, digest in assignment["input_sha256"].items():
        if sha(pool / name) != digest or provenance["inputs_sha256"][name] != digest:
            raise ValueError("Frozen input archive changed: " + name)
    with np.load(pool / "pairs.npz", allow_pickle=False) as pairs:
        records = [dict(order=i+1, pair_id=pair, start_index=int(pairs["start_index"][pair]),
            goal_index=int(pairs["goal_index"][pair])) for i, pair in enumerate(ids)]
    if records != assignment["records"]: raise ValueError("Assignment pair indices changed")
    joints = [j for j in ET.parse(source / MODEL).getroot().findall("joint") if j.get("type") != "fixed"]
    if len(joints) != 7: raise ValueError("Expected the measured seven-joint physical model")
    limits = {key: np.array([float(j.find("limit").get(key)) for j in joints])
        for key in ("lower", "upper", "velocity", "effort")}
    limits["names"] = [j.get("name") for j in joints]
    return assignment, protocol, freeze, provenance, launch, limits


def history(directory, limits, usable):
    row = dict(history_status="missing", state_samples=0, control_samples=0,
        observed_prefix_minimum_q_clearance_rad=None, observed_prefix_maximum_speed_ratio=None,
        observed_prefix_maximum_torque_ratio=None, minimum_q_clearance_rad=None,
        maximum_speed_ratio=None, maximum_torque_ratio=None)
    path = directory / "executed.npz"
    if not path.is_file(): return row
    with np.load(path, allow_pickle=False) as z:
        states, controls, times = z["states"], z["controls"], z["times"]
        if states.shape != (len(times), 14) or controls.shape != (len(z["control_times"]), 7):
            raise ValueError("Unexpected dense history shape: " + str(directory))
        row.update(state_samples=len(states), control_samples=len(controls))
        finite = bool(np.isfinite(states).all() and np.isfinite(controls).all())
        row["history_status"] = "finite" if finite else "nonfinite"
        if finite and len(states):
            row["observed_prefix_minimum_q_clearance_rad"] = float(np.min(np.minimum(states[:, :7]-limits["lower"], limits["upper"]-states[:, :7])))
            row["observed_prefix_maximum_speed_ratio"] = float(np.max(np.abs(states[:, 7:])/limits["velocity"]))
        if finite and len(controls):
            row["observed_prefix_maximum_torque_ratio"] = float(np.max(np.abs(controls)/limits["effort"]))
        if usable and finite and len(states) and len(controls):
            for suffix in ("minimum_q_clearance_rad", "maximum_speed_ratio", "maximum_torque_ratio"):
                row[suffix] = row["observed_prefix_" + suffix]
    return row


def collect(raw, assignment, protocol, provenance, limits):
    pilot = raw / "pilot"; ids = assignment["task_ids"]
    summary_path = pilot / "summary.json"
    summary = read(summary_path) if summary_path.is_file() else []
    if [m["pair_id"] for m in summary] != ids[:len(summary)]:
        raise ValueError("Summary must be the exact ordered assignment prefix")
    saved = {m["pair_id"]: m for m in summary}; rows = []; hashes = {}
    with np.load(Path(provenance["pool"]) / "pool.npz", allow_pickle=False) as archive:
        configurations = archive["q"]
    for record in assignment["records"]:
        pair = record["pair_id"]; directory = pilot / f"pair_{pair:05d}"
        path = directory / "metrics.json"; m = read(path) if path.is_file() else None
        if pair in saved and saved[pair] != m: raise ValueError("Summary/case metrics differ: " + str(pair))
        if m and m["pair_id"] != pair: raise ValueError("Case ID differs: " + str(pair))
        accepted = pair in saved and bool(m and m.get("complete"))
        row = dict(**record, status=m["outcome"] if accepted else "pending_summary" if m else "in_progress" if directory.exists() else "not_recorded",
            recorded=int(pair in saved), complete=int(accepted), outcome=m["outcome"] if m else None,
            task_success=int(m["task_success"]) if accepted else None,
            physical_pass=int(m["physical_pass"]) if accepted else None,
            both_pass=int(m["task_success"] and m["physical_pass"]) if accepted else None,
            execution_failure=int(bool(m["execution_error"])) if accepted else None,
            execution_error=m.get("execution_error") if m else None,
            failure_phase=m.get("failure_phase") if m else None,
            interrupted=int(bool(m and m.get("interrupted"))),
            shared_setup_failure=int(bool(m and m.get("shared_setup_failure"))),
            arrival_seconds=value(m["reached_seconds"]) if accepted else None,
            observed_seconds=value(m["simulated_seconds"]) if m else None,
            final_position_mm=value(m["final"]["position_error_m"]*1000) if m and value(m["final"]["position_error_m"]) is not None else None,
            final_tool_axis_deg=value(m["final"]["tool_axis_error_deg"]) if m else None,
            final_orientation_deg=value(m["final"]["orientation_error_deg"]) if m else None,
            final_speed_L1_rad_s=value(m["final"]["velocity_l1_rad_s"]) if m else None,
            feedback_solves=m["feedback_solves"] if m else None,
            actual_sqp_sum=m["actual_sqp_sum"] if m else None,
            zero_pcg_exit_solves=m["zero_pcg_exit_solves"] if m else None,
            episode_wall_s=value(m["episode_wall_s"]) if m else None,
            metrics_sha256=sha(path) if m else None)
        p = m["physical"] if m else None
        row["physical_finite"] = int(p["finite"]) if p else None
        row["completed_episode_physical_pass"] = int(not m['execution_error'] and p['finite'] and m['physical_pass']) if accepted else None
        row["both_pass"] = int(row['task_success'] and row['completed_episode_physical_pass']) if accepted else None
        last = m.get('last_second', {}) if m else {}
        for key in ('observation_seconds', 'position_error_min_m', 'position_error_max_m',
                'net_position_error_reduction_m', 'joint_total_variation_L1_rad',
                'maximum_joint_speed_rad_s', 'maximum_speed_L1_rad_s'):
            row['last_second_' + key] = value(last.get(key))
        for category in ("joint", "velocity", "effort"):
            c = p[category] if p else None; first = c["first_violation"] if c else None
            row[category + "_pass"] = int(c["pass_check"]) if c else None
            row[category + "_recorded_maximum_excess"] = value(c["maximum_excess"]) if c else None
            row[category + "_first_joint"] = first["joint_name"] if first else None
            row[category + "_first_time_s"] = value(first["time_s"]) if first else None
        usable = accepted and not m["execution_error"] and p["finite"]
        row.update(history(directory, limits, usable))
        row["maximum_q_excess_rad"] = max(0., -row["minimum_q_clearance_rad"]) if row["minimum_q_clearance_rad"] is not None else None
        if usable and row["history_status"] == "finite":
            for actual, expected in ((row["maximum_q_excess_rad"], p["joint"]["maximum_excess"]),
                    (row["maximum_speed_ratio"], p["maximum_velocity_ratio"]),
                    (row["maximum_torque_ratio"], p["maximum_effort_ratio"])):
                if actual is not None and not np.isclose(actual, expected, rtol=1e-6, atol=1e-8):
                    raise ValueError("Dense saved extrema differ from metrics: " + str(pair))
        inputs = directory / "inputs.npz"
        if inputs.is_file():
            with np.load(inputs, allow_pickle=False) as z:
                if not np.allclose(z["initial_state"][:7], configurations[record["start_index"]], rtol=0., atol=1e-7) or not np.allclose(z["source_goal_q"], configurations[record["goal_index"]], rtol=0., atol=1e-7):
                    raise ValueError("Executed start/goal witness differs from frozen pair: " + str(pair))
        hashes[str(pair)] = {p.name: sha(p) for p in (path, inputs, directory / "executed.npz", directory / "solves.jsonl") if p.is_file()}
        rows.append(row)
    complete = len(summary) == COUNT and all(m.get("complete") for m in summary)
    recorded = [r for r in rows if r["complete"]]
    counts = dict(assigned=COUNT, complete=complete, recorded=len(recorded), unrecorded=COUNT-len(recorded),
        rough_arrivals=sum(r["task_success"] for r in recorded), physical_passes=sum(r["completed_episode_physical_pass"] for r in recorded),
        raw_sampled_physical_passes=sum(r["physical_pass"] for r in recorded),
        both_passes=sum(r["both_pass"] for r in recorded), timeouts=sum(r["outcome"] == "timeout" for r in recorded),
        execution_failures=sum(r["execution_failure"] for r in recorded),
        physical_timeouts=sum(r["outcome"] == "timeout" and r["completed_episode_physical_pass"] for r in recorded),
        arrivals_with_physical_failure=sum(r["task_success"] and not r["completed_episode_physical_pass"] for r in recorded))
    for category in ("joint", "velocity", "effort"):
        counts[category + "_failed_checks"] = sum(r[category + "_pass"] == 0 for r in recorded)
    return rows, counts, hashes


def plot(output):
    rows = csv_read(output / "cases.csv"); counts = read(output / "counts.json")
    timeout = read(output / "protocol.json")["timeout_s"]
    arrivals = sorted(float(r["arrival_seconds"]) for r in rows if r["complete"] == "1" and r["task_success"] == "1")
    valid = sorted(float(r["arrival_seconds"]) for r in rows if r["complete"] == "1" and r["both_pass"] == "1")
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3))
    for times, label in ((arrivals, "Rough arrival"), (valid, "Arrival + physical pass")):
        axes[0].step([0, *times, timeout], [0, *np.arange(1, len(times)+1)/COUNT, len(times)/COUNT], where="post", label=label)
    axes[0].set(xlim=(0, timeout), ylim=(0, 1), xlabel="Simulated time (s)", ylabel="Cumulative arrivals / assigned 100")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=.2)
    axes[1].hist(arrivals, bins=np.linspace(0, timeout, 17), color="#277da8", edgecolor="white")
    axes[1].set(xlabel="Arrival time (s), arrivals only", ylabel="Number of arrivals", xlim=(0, timeout))
    axes[1].text(.02, .98, f"Assigned: 100; recorded: {counts['recorded']}\nTimeouts: {counts['timeouts']}; execution failures: {counts['execution_failures']}\nNot completed: {counts['unrecorded']}", transform=axes[1].transAxes, va="top", fontsize=8)
    fig.suptitle("Additional validation: fixed population and conditional arrival times" + (" (INCOMPLETE)" if not counts["complete"] else ""))
    fig.tight_layout(); fig.savefig(output / "arrivals.png", dpi=180); fig.savefig(output / "arrivals.pdf"); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.1))
    for ax, key, label, boundary in zip(axes,
            ("minimum_q_clearance_rad", "maximum_speed_ratio", "maximum_torque_ratio"),
            ("Minimum q clearance (rad)", "Maximum speed / physical limit", "Maximum torque / physical limit"), (0, 1, 1)):
        observations = [r for r in rows if r[key] != ""]
        for passed, marker, color, name in ((True, "o", "#277da8", "Physical pass"), (False, "x", "#d62828", "Physical failure")):
            group = [r for r in observations if (r["completed_episode_physical_pass"] == "1") == passed]
            ax.scatter([int(r["order"]) for r in group], [float(r[key]) for r in group], marker=marker, color=color, s=20, alpha=.75, label=name)
        ax.axhline(boundary, color="black", ls="--", lw=.8); ax.set(xlabel="Frozen assignment order", ylabel=label, xlim=(0, 101))
        ax.text(.02, .98, f"Finite full histories: {len(observations)}/100", transform=ax.transAxes, va="top", fontsize=8)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Sampled physical extrema; missing/error/nonfinite full histories remain null")
    fig.tight_layout(); fig.savefig(output / "physical_extrema.png", dpi=180); fig.savefig(output / "physical_extrema.pdf"); plt.close(fig)


def examples(rows):
    completed = [r for r in rows if r["complete"]]; chosen = {}
    for key, reverse, label in (("minimum_q_clearance_rad", False, "smallest q clearance"),
            ("maximum_speed_ratio", True, "largest speed ratio"), ("maximum_torque_ratio", True, "largest torque ratio")):
        available = [r for r in completed if r[key] is not None]
        if available:
            r = sorted(available, key=lambda r: r[key], reverse=reverse)[0]
            chosen.setdefault(r["pair_id"], [r, []])[1].append(label)
    categories = ((lambda r:r["execution_failure"], "execution failure"),
        (lambda r:r["task_success"] and not r["completed_episode_physical_pass"], "arrival with physical failure"),
        (lambda r:r["outcome"] == "timeout" and r["completed_episode_physical_pass"], "physical-pass timeout"))
    for predicate, label in categories:
        candidates = sorted([r for r in completed if predicate(r)], key=lambda r: r["final_position_mm"] if r["final_position_mm"] is not None else float("inf"))
        for r in candidates[:2]: chosen.setdefault(r["pair_id"], [r, []])[1].append(label)
    return list(chosen.values())


def report(output, rows, counts, assignment, protocol, freeze, provenance):
    state = "FINAL" if counts["complete"] else "INCOMPLETE PREVIEW"
    lines = ["# TIAGo: 100 additional random reaching tasks", "", f"Status: {state}; {counts['recorded']}/100 complete outcomes. "
        "The configuration was chosen before validation, and these results do not guide tuning, retries or further parameter decisions.", "",
        f"Rough arrivals: {counts['rough_arrivals']}/100; sampled physical passes: {counts['physical_passes']}/100; both: {counts['both_passes']}/100. "
        f"Timeouts: {counts['timeouts']}; execution failures: {counts['execution_failures']}; not completed: {counts['unrecorded']}. "
        "Incomplete assignments remain unrecorded rather than being relabeled failures or successes. In a preview, numerators and fixed100 cumulative curves are observed lower bounds until all assigned cases finish.", "",
        "## Population and frozen selection", "",
        f"The stdlib sampler used seed {assignment['seed']} to sample 100 ordered pair IDs without replacement from the remaining {assignment['remaining_population_size']} of {assignment['population_size']} frozen pairs, excluding every historical 100-task ID. "
        "No geometry or outcome filtering was applied. assignment.json retains all IDs, exclusions, start/goal indices and archive hashes. These are 100 additional tasks, separate from development; no combined 112-task success rate is reported.", "",
        f"Selected development configuration: {freeze['selected_cohort']}, frozen at {freeze['created_utc']}. "
        f"Its development results were {freeze['rough_arrivals']}/12 rough arrivals and {freeze['physical_passes']}/12 physical passes. "
        "The canonical preflight passed and is reused with unchanged source/module/runtime parameters. Development performance is tuning evidence, not an independent comparison against these 100 tasks.", "",
        "## Arrival and failure counts", "", "| Outcome | Count / assigned 100 |", "|---|---:|"]
    for key in ("rough_arrivals", "physical_passes", "both_passes", "timeouts", "execution_failures", "physical_timeouts", "arrivals_with_physical_failure", "unrecorded"):
        lines.append(f"| {key.replace('_',' ')} | {counts[key]}/100 |")
    lines += ["",
        "![The cumulative curves always divide by the assigned 100 tasks and are observed lower bounds in an incomplete preview. The histogram is conditional on observed arrivals and is not a success-rate estimate. Non-arriving timeouts are right-censored at their observed eight-second endpoint; execution failures are separate failed episodes. Unrecorded tasks have no inferred event/censoring time. No survival extrapolation is applied.](arrivals.png)", "",
        "## Sampled physical limits", "",
        "Physical checks use the unchanged raw URDF limits, not the interior soft-cost bounds: q excess <=1e-6 rad and speed/torque ratios <=1+1e-6. States are recorded at every RK4 plant sample (normally 3 ms; shorter final steps are retained), and applied controls at their original sample times. Finite sampled compliance is not a continuous-time safety, collision-clearance or physical-robot certificate.", "",
        "Headline physical passes require a complete, error-free, finite episode and its recorded physical-pass flag. cases.csv preserves the original physical_pass flag, which can describe a short prefix before an execution error, alongside completed_episode_physical_pass. Such an error prefix never earns a headline physical pass or a both-pass outcome.", "",
        f"Failed category checks among completed episodes: joint position {counts['joint_failed_checks']}, speed {counts['velocity_failed_checks']}, torque {counts['effort_failed_checks']}. "
        "Each episode's signed q clearance, speed/torque peaks, per-category flags and first-violation joint/time remain in cases.csv. Negative clearance means a physical joint bound was exceeded. Error/nonfinite full histories have null extrema, never zero; finite observed-prefix extrema are separately labeled and do not certify a complete episode.", "",
        "![Every available finite full-history episode contributes one sample per panel; none is removed because it failed a physical check. Null/error/nonfinite histories are counted separately in cases.csv. Horizontal lines show physical boundaries; colors show error-free completed-episode physical compliance. Original prefix flags remain in the CSV.](physical_extrema.png)", "",
        "## Representative outcomes and worst sampled limits", "",
        "The following deterministic examples include each worst finite q/speed/torque episode, up to two execution failures, up to two arrivals with physical failure and two physical-pass timeouts closest to the goal. Categories can select the same episode. This is descriptive failure identification, with no rerun or rescue.", "",
        "| ID | Selection | Result | Arrival / physical | Final mm | q clearance rad | Speed ratio | Torque ratio |", "|---|---|---|---|---:|---:|---:|---:|"]
    for row, labels in examples(rows):
        lines.append(f"| {row['pair_id']} | {', '.join(labels)} | {row['outcome']} | {row['task_success']}/{row['completed_episode_physical_pass']} | {number(row['final_position_mm'])} | {number(row['minimum_q_clearance_rad'])} | {number(row['maximum_speed_ratio'])} | {number(row['maximum_torque_ratio'])} |")
    lines += ['', 'Timeout does not establish that more time would produce arrival: a soft goal can approach an offset pose or stall. The saved last-second observations are descriptive, without a new progress gate.']
    for row, labels in examples(rows):
        if 'physical-pass timeout' in labels:
            lines += ['', f"Timeout ID{row['pair_id']}: final axis error {number(row['final_tool_axis_deg'])} degrees and speed L1 {number(row['final_speed_L1_rad_s'])} rad/s. "
                f"Over its final {number(row['last_second_observation_seconds'])} s, position error decreased by {number(row['last_second_net_position_error_reduction_m'])} m "
                f"and joint total variation L1 was {number(row['last_second_joint_total_variation_L1_rad'])} rad. Numeric last_second fields for every recorded episode remain in cases.csv."]
    failed = [r for r in rows if r["complete"] and (not r["completed_episode_physical_pass"] or r["execution_failure"])]
    lines += ["", "Physical-failure or execution-error IDs: " + (", ".join(str(r["pair_id"]) for r in failed) or "none recorded") + ". "
        "The complete failure_cases.csv also includes every task timeout. All 100 assignment rows remain in cases.csv, preserving the original outcome and null/missing labels.", "",
        "## Protocol and scope", "",
        "Rough arrival requires position error <=50 mm, tool-axis error <=30 degrees and joint-speed L1 <=1 rad/s at a 30 ms feedback boundary, with zero dwell. Tool-axis error does not constrain rotation around that axis; full orientation error is recorded separately. A task may arrive while violating physical limits, and a physically compliant trajectory may time out.", "",
        f"Mode {protocol['timing_mode']}, offline_timing={protocol['offline_timing']}: no plant time advances during solve. "
        f"B{protocol['batch']}, N{protocol['N']}, {protocol['knot_dt_s']} s knots and feedback, {protocol['sim_dt_s']} s RK4 plant, {protocol['timeout_s']} s maximum episode. "
        f"Prediction duration is {(protocol['N']-1)*protocol['knot_dt_s']:.2f} s; receding-horizon task duration can exceed that window. Final approach is disabled. Arrival ends observation; no post-arrival settling claim is possible.", "",
        protocol["initialization"], "",
        "| Solver/objective parameter | Frozen value |", "|---|---:|"]
    for key, val in protocol["solver_params"].items(): lines.append(f"| {key} | {val} |")
    bounds = protocol["limit_penalty"]["cost_bounds"]
    lines += ["", f"Interior C2 cost bounds use a {bounds['joint_margin_rad']} rad joint inset, speed fraction {bounds['velocity_fraction_of_physical_limit']} and effort fraction {bounds['effort_fraction_of_physical_limit']}. "
        "The quadratic continuation width is 1% of the interior interval. These are soft costs and do not guarantee constraint enforcement. Original starts outside soft intervals are retained without clipping. The C2 candidate remains isolated experiment source, not a production merge.", "",
        "The existing natural zero-PCG outer exit remains the practical stopping heuristic, distinct from an ordinary nonzero PCG solve. Recorded KKT settings/residuals are descriptive and are not acceptance gates or reasons to extend runs. No solver modifications or new convergence instrumentation are part of this analysis.", "",
        "## Identities, data and reproduction", "", "metadata.json records the absolute raw validation root and exact launch argv/cwd/UTC. "
        "Large histories, horizons and per-solve data remain in raw storage. Compact exports contain all100 case rows, failures, assignment/protocol/freeze/launch records and full source/module/input hashes.", "",
        "| Identity | SHA256 |", "|---|---|"]
    meta = read(output / "metadata.json")
    for key in ("assignment.json", "validation_protocol.json", "final_configuration_freeze.json", "launch_receipt.json"):
        lines.append(f"| {key} | {meta['records'][key]['sha256']} |")
    lines += [f"| Native module | {provenance['module_sha256']} |", f"| Worker | {provenance['wrapper_sha256']} |", "",
        f"Measured source commit: `{provenance['source_commit']}`; numerical source base: `{protocol['source_base']}`. "
        "metadata.json contains every measured source hash, reused preflight receipt, per-case raw-file hashes and the exact analysis command. The raw summary is checked against every case metric; a final export requires all100 complete records in exact frozen order.", "",
        "Regenerate only the figures and self-contained PDF from exported compact inputs:", "", "```sh", "python analyze_validation.py --plot-only .", "python report_to_pdf.py --report REPORT.md", "```", "",
        "This simulated, single-machine, one-pass validation describes this fixed parameterization and sampled task population. It does not establish universal reachability, reliability, continuous safety, collision clearance, paid-computation performance or physical-robot behavior. Validation outcomes are not used to select new parameters.", ""]
    (output / "REPORT.md").write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("raw-root", "preparation", "source", "output", "previews"):
        p.add_argument("--" + key, type=Path)
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--plot-only", type=Path)
    args = p.parse_args()
    if args.plot_only: plot(args.plot_only.resolve()); return
    if any(getattr(args, k) is None for k in ("raw_root", "source", "output")):
        p.error("Provide --raw-root --source --output, optionally --preparation")
    raw, source, output = (getattr(args, k).resolve() for k in ("raw_root", "source", "output"))
    prep = (args.preparation or raw / "preparation-M14-v2").resolve()
    if output.is_relative_to(raw): raise ValueError("Compact export must be outside raw storage")
    assignment, protocol, freeze, provenance, launch, limits = verify(raw, prep, source)
    rows, counts, hashes = collect(raw, assignment, protocol, provenance, limits)
    if not counts["complete"] and (not args.allow_partial or "thesis_results" in output.parts):
        raise ValueError("Incomplete100: use --allow-partial with a temporary output outside thesis_results")
    output.mkdir(parents=True, exist_ok=True)
    csv_write(output / "cases.csv", rows)
    failures = [r for r in rows if r["complete"] and (not r["task_success"] or not r["completed_episode_physical_pass"] or r["execution_failure"])]
    csv_write(output / "failure_cases.csv", failures, list(rows[0]))
    write(output / "counts.json", counts); write(output / "protocol.json", protocol)
    paths = [prep / name for name in ("assignment.json", "validation_protocol.json", "final_configuration_freeze.json", "selection_receipt.json", "PREPARED.json")]
    paths += [raw / "launch_receipt.json"]
    copied = {}
    for path in paths:
        shutil.copy2(path, output / path.name)
        copied[path.name] = dict(source=str(path), sha256=sha(path))
    metadata = dict(created_utc=datetime.now(timezone.utc).isoformat(), status="final" if counts["complete"] else "INCOMPLETE PREVIEW",
        analysis_command=shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]),
        analysis_sha256=sha(__file__), raw_root=str(raw), preparation=str(prep), source=str(source),
        records=copied, provenance=provenance, launch=launch, case_raw_sha256=hashes,
        summary_sha256=sha(raw / "pilot/summary.json"), fixed_denominator=COUNT,
        physical_limits={key: val.tolist() if isinstance(val, np.ndarray) else val for key, val in limits.items()})
    write(output / "metadata.json", metadata)
    plot(output); report(output, rows, counts, assignment, protocol, freeze, provenance)
    script = Path(__file__).resolve()
    if script != output / script.name: shutil.copy2(script, output / script.name)
    helper = source / "experiments/random_reaching_baseline/report_to_pdf.py"
    shutil.copy2(helper, output / helper.name)
    spec = importlib.util.spec_from_file_location("report_to_pdf", helper)
    renderer = importlib.util.module_from_spec(spec); spec.loader.exec_module(renderer)
    pages = renderer.render(output / "REPORT.md", args.previews)
    print(json.dumps(dict(counts=counts, pdf_pages=pages, output=str(output))))


if __name__ == "__main__": main()
