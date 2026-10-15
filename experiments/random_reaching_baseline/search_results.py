#!/usr/bin/env python3
"""Summarize saved parameter-search stages; stdlib only, no solver calls."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import sys


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def maximum(values):
    return max(values) if values and all(value is not None for value in values) else None


def number(value):
    return "null" if value is None else f"{value:.7g}"


def collect(raw_root, source):
    plans = {path.stem: read(path) for path in (source / "experiments/random_reaching_baseline/protocols").glob("M*.json")}
    for path in raw_root.glob("M*"):
        if path.is_dir():
            plans.setdefault(path.name, None)
    stages, cases, evidence = [], [], {}
    for name in sorted(plans, key=lambda value: int(value.split("-", 1)[0][1:])):
        previous = None
        for stage in ("preflight", "screen", "pilot"):
            root = raw_root / name / stage
            provenance_path = root / "provenance.json"
            held_path = raw_root / name / (stage + "_held.json")
            if not provenance_path.is_file() and not held_path.is_file():
                continue
            provenance = read(provenance_path) if provenance_path.is_file() else None
            protocol_path = root / "protocol.json"
            protocol = read(protocol_path) if protocol_path.is_file() else plans[name]
            if protocol is None:
                raise ValueError("No retained protocol: " + str(root))
            if provenance:
                if provenance["protocol"] != protocol:
                    raise ValueError("Protocol/provenance mismatch: " + str(root))
                identity = {key: provenance[key] for key in ("source_sha256", "module_sha256", "wrapper_sha256", "inputs_sha256")}
                identity["protocol"] = protocol
                if previous is not None and identity != previous:
                    raise ValueError("Same-candidate measured stage identities differ: " + name)
                previous = identity
            assigned = provenance["pairs"] if provenance else protocol["development_task_ids"]
            summary_path = root / "summary.json"
            summary = read(summary_path) if summary_path.is_file() else []
            ids = [item["pair_id"] for item in summary]
            if ids != assigned[:len(ids)] or len(ids) != len(set(ids)):
                raise ValueError("Summary is not an exact ordered assignment prefix: " + str(root))
            hashes = {}
            for m in summary:
                directory = root / ("preflight" if m["pair_id"] == -1 else f"pair_{m['pair_id']:05d}")
                metric_path = directory / "metrics.json"
                if read(metric_path) != m:
                    raise ValueError("Summary/case metrics mismatch: " + str(metric_path))
                hashes[str(m["pair_id"])] = sha(metric_path)
                physical = m["physical"]
                case = dict(configuration=name, stage=stage, pair_id=m["pair_id"], complete=m["complete"],
                    outcome=m["outcome"], rough_arrival=m["task_success"], physical_pass=m["physical_pass"],
                    finite=physical["finite"], execution_error=m["execution_error"],
                    preflight_pass=m.get("preflight_pass"), observed_s=m["simulated_seconds"],
                    arrival_s=m["reached_seconds"], maximum_q_excess_rad=physical["joint"]["maximum_excess"],
                    maximum_speed_ratio=physical["maximum_velocity_ratio"], maximum_torque_ratio=physical["maximum_effort_ratio"],
                    feedback_solves=m["feedback_solves"], actual_sqp_sum=m["actual_sqp_sum"],
                    zero_pcg_exits=m["zero_pcg_exit_solves"], metrics_sha256=hashes[str(m["pair_id"])])
                for category in ("joint", "velocity", "effort"):
                    first = physical[category]["first_violation"]
                    case[category + "_pass"] = physical[category]["pass_check"]
                    case[category + "_first_joint"] = first["joint_name"] if first else None
                    case[category + "_first_time_s"] = first["time_s"] if first else None
                    case[category + "_first_excess"] = first["excess"] if first else None
                cases.append(case)
            held = read(held_path) if held_path.is_file() else None
            complete = len(summary) == len(assigned) and all(item["complete"] for item in summary)
            state = "held" if held and not provenance else "complete" if complete else "partial"
            if (root / "setup_failure.json").is_file():
                state = "shared_setup_failure"
            cfg = protocol["solver_params"]
            bounds = protocol["limit_penalty"].get("cost_bounds", {})
            row = dict(configuration=name, stage=stage, status=state, assigned=len(assigned), recorded=len(summary),
                q_margin_rad=bounds.get("joint_margin_rad", 0.), q_limit_weight=cfg["q_lim_cost"],
                velocity_limit_weight=cfg["vel_lim_cost"], torque_limit_weight=cfg["ctrl_lim_cost"],
                qd_weight=cfg["qd_cost"], mu=cfg["mu"], rough_arrivals=sum(item["task_success"] for item in summary),
                preflight_pass=summary[0].get("preflight_pass") if stage == "preflight" and summary else None,
                physical_passes=sum(item["physical_pass"] for item in summary),
                both=sum(item["task_success"] and item["physical_pass"] for item in summary),
                execution_failures=sum(bool(item["execution_error"]) for item in summary),
                timeouts=sum(item["outcome"] == "timeout" for item in summary),
                q_failures=sum(not item["physical"]["joint"]["pass_check"] for item in summary),
                velocity_failures=sum(not item["physical"]["velocity"]["pass_check"] for item in summary),
                torque_failures=sum(not item["physical"]["effort"]["pass_check"] for item in summary),
                maximum_q_excess_rad=maximum([item["physical"]["joint"]["maximum_excess"] for item in summary]),
                maximum_speed_ratio=maximum([item["physical"]["maximum_velocity_ratio"] for item in summary]),
                maximum_torque_ratio=maximum([item["physical"]["maximum_effort_ratio"] for item in summary]),
                protocol_id=protocol["id"], protocol_sha256=sha(protocol_path) if protocol_path.is_file() else None,
                numerical_source=protocol["source_base"], module_sha256=provenance["module_sha256"] if provenance else None)
            stages.append(row)
            evidence[name + "/" + stage] = dict(raw_root=str(root), protocol=protocol, provenance=provenance,
                hold_record=held, hashes={p.name: sha(p) for p in
                    (summary_path, protocol_path, provenance_path, held_path, root / "setup_failure.json") if p.is_file()},
                case_metric_sha256=hashes)
    return stages, cases, evidence


def report(stages, cases, evidence):
    lines = ["# Saved parameter-search status", "",
        "Canonical preflights, one-case screens and full development pilots are separate evidence. Counts below refer only to recorded outcomes; unexecuted assigned cases are not failures. Screen success is not a twelve-task qualification. Rough arrival and physical compliance are independent. A finite, physically compliant canonical timeout may precede the explicitly requested one-case screen, but its preflight_pass remains false and it does not qualify a final baseline.", "",
        "Physical checks retain raw URDF bounds and their original tolerances. In particular, a joint-position pass alone is not an all-limit pass: speed and torque must also pass. No monotonicity of optimizer/task behavior versus penalty weight is assumed. This summary chooses no next parameter, changes no protocol and launches no solver. Fresh100 validation and horizon studies remain separate held work.", "",
        "## Recorded stage outcomes", "", "A complete stage means the assigned recording finished; it does not imply arrival, a passed canonical gate or physical compliance.", "",
        "| Config | Stage | State | Recorded/assigned | Inset / q weight | Canonical gate | Arrivals | Physical | Both | Errors |", "|---|---|---|---:|---:|---|---:|---:|---:|---:|"]
    for row in stages:
        gate = "pass" if row["preflight_pass"] else "FAIL" if row["preflight_pass"] is False else "n/a"
        lines.append(f"| {row['configuration'].split('-')[0]} | {row['stage']} | {row['status']} | {row['recorded']}/{row['assigned']} | {row['q_margin_rad']:g}/{row['q_limit_weight']:g} | {gate} | {row['rough_arrivals']} | {row['physical_passes']} | {row['both']} | {row['execution_failures']} |")
    lines += ["", "## Executed physical magnitudes", "", "Null preserves an unavailable/nonfinite full-history maximum; it is never converted to zero. Violation categories overlap.", "",
        "| Config | Stage | q fail / max rad | Speed fail / max ratio | Torque fail / max ratio |", "|---|---|---:|---:|---:|"]
    for row in stages:
        lines.append(f"| {row['configuration'].split('-')[0]} | {row['stage']} | {row['q_failures']}/{number(row['maximum_q_excess_rad'])} | {row['velocity_failures']}/{number(row['maximum_speed_ratio'])} | {row['torque_failures']}/{number(row['maximum_torque_ratio'])} |")
    lines += ["", "## Task2342 observations at a 0.20 rad inset", "",
        "These observed pass/fail points provide a search record, not a guaranteed monotone bracket. A one-case screen and task2342 inside a twelve-case pilot retain their different scopes. Physically compliant stalled/poor reaches remain distinct from rough arrival.", "",
        "| Config | Scope | q weight | Speed/torque weights | Arrival | q pass | Speed pass | Torque pass | All limits | q rad | Speed ratio | Torque ratio |", "|---|---|---:|---|---|---|---|---|---|---:|---:|---:|"]
    for case in cases:
        stage = next(row for row in stages if row["configuration"] == case["configuration"] and row["stage"] == case["stage"])
        if case["pair_id"] != 2342 or abs(stage["q_margin_rad"]-.2) > 1e-12:
            continue
        flags = ["yes" if case[key] else "NO" for key in ("rough_arrival", "joint_pass", "velocity_pass", "effort_pass", "physical_pass")]
        lines.append(f"| {case['configuration'].split('-')[0]} | {case['stage']} ({stage['assigned']} assigned) | {stage['q_limit_weight']:g} | {stage['velocity_limit_weight']:g}/{stage['torque_limit_weight']:g} | " + " | ".join(flags) + f" | {number(case['maximum_q_excess_rad'])} | {number(case['maximum_speed_ratio'])} | {number(case['maximum_torque_ratio'])} |")
    lines += ["", "First violations below identify the earliest recorded joint/time in each category; the episode-wide peak ratio can occur at another joint/time. Full names, times and excesses are retained in SEARCH_CASES.csv."]
    for case in cases:
        if case["pair_id"] != 2342:
            continue
        violations = [f"{category}: {case[category+'_first_joint']} at {number(case[category+'_first_time_s'])} s"
            for category in ("joint", "velocity", "effort") if not case[category+"_pass"]]
        if violations:
            lines += ["", case["configuration"] + "/" + case["stage"] + ": " + "; ".join(violations) + "."]
    lines += ["", "## Exact retained identities", "",
        "SEARCH_STAGES.csv retains every numeric/configuration row; SEARCH_EVIDENCE.json embeds all saved protocols, source/module/wrapper/input hashes, metric/summary hashes, hold records and the exact analysis command. Every recorded summary entry equals its saved per-case metrics. Natural zero-PCG outer stopping and original SQP/PCG caps remain unchanged; existing KKT flags impose no gate."]
    for key, item in evidence.items():
        protocol, provenance = item["protocol"], item["provenance"]
        lines += ["", key + ": `" + protocol["id"] + "`; numerical source `" + protocol["source_base"] + "`."]
        if provenance:
            lines += ["Native SHA256: `" + provenance["module_sha256"] + "`.", "Protocol SHA256: `" + item["hashes"]["protocol.json"] + "`."]
        if item["hold_record"]:
            lines += ["Held: " + item["hold_record"].get("reason", str(item["hold_record"])) + "."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Defaults to raw root; writes SEARCH_* analysis files only")
    args = parser.parse_args()
    raw_root, source = args.raw_root.resolve(), args.source.resolve()
    output = (args.output or raw_root).resolve(); output.mkdir(parents=True, exist_ok=True)
    stages, cases, evidence = collect(raw_root, source)
    if not stages:
        parser.error("No recorded stages or retained hold decisions")
    write_csv(output / "SEARCH_STAGES.csv", stages)
    write_csv(output / "SEARCH_CASES.csv", cases)
    record = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        command=shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]),
        script_sha256=sha(__file__), raw_root=str(raw_root), stages=evidence)
    (output / "SEARCH_EVIDENCE.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    status = report(stages, cases, evidence)
    if (output / "SEARCH_TRADEOFF_2342.png").is_file():
        status += ("\n## Dense executed task2342 comparison\n\n"
            "![J1 signed clearance to its raw physical interval and maximum speed ratio over seven joints. Every saved 3 ms sample is retained. Curves label both penalty weights and recording scope; endpoints stop at the original arrival, timeout or failure. This descriptive comparison establishes no solver-cause claim.](SEARCH_TRADEOFF_2342.png)\n\n"
            "SEARCH_TRADEOFF_2342.csv retains all plotted samples; its JSON records source/module/protocol/input identities. "
            "Regenerate from compact data using `python plot_search_tradeoff.py --plot-only .`.\n")
    (output / "SEARCH_STATUS.md").write_text(status)
    print(json.dumps(dict(stages=len(stages), recorded_stage_outcomes=len(cases), output=str(output))))


if __name__ == "__main__":
    main()
