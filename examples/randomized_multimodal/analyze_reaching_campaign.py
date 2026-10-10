"""Incrementally analyze completed paired MPC campaign episodes (CPU only)."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
URDF = ROOT / "gato/dynamics/tiago_right/tiago_right_arm.urdf"
COLORS = {1: "#087e97", 16: "#d4642e"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [safe(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return "NaN" if np.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    return value


def write_json(path, value):
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(safe(value), indent=2, allow_nan=False)+"\n")
    tmp.replace(path)


def number(value):
    return np.nan if value is None else float(value)


def distribution(values):
    values = np.asarray([number(v) for v in values], dtype=float)
    finite = values[np.isfinite(values)]
    return dict(total=len(values), finite=len(finite), nan_or_missing=int(np.isnan(values).sum()),
                positive_infinity=int(np.isposinf(values).sum()), negative_infinity=int(np.isneginf(values).sum()),
                minimum=float(finite.min()) if len(finite) else None,
                median=float(np.median(finite)) if len(finite) else None,
                maximum=float(finite.max()) if len(finite) else None)


def limits():
    joints = {j.attrib["name"]: j for j in ET.parse(URDF).getroot().findall("joint")}
    rows = [joints[f"arm_right_{i}_joint"].find("limit").attrib for i in range(1, 8)]
    return tuple(np.array([float(r[key]) for r in rows]) for key in ("lower", "upper", "velocity", "effort"))


def validate_prediction_duration(manifest):
    """Allow only binary64 representation rounding of the derived horizon."""
    N = manifest.get("N")
    dt = manifest.get("knot_dt")
    duration = manifest.get("prediction_seconds")
    if type(N) is not int or N < 2:
        raise ValueError("Invalid/missing prediction knot count")
    for name, value in (("knot_dt", dt), ("prediction_seconds", duration)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid/missing prediction {name}")
    derived = (N - 1) * dt
    if not math.isfinite(derived) or abs(duration - derived) > 2 * math.ulp(derived):
        raise ValueError("Prediction duration differs from (N-1)*knot_dt")


def validate_campaign_episode(manifest, campaign_manifest):
    if any(key in campaign_manifest for key in ("N", "knot_dt", "prediction_seconds")):
        validate_prediction_duration(campaign_manifest)
        validate_prediction_duration(manifest)
    for key in ("N", "knot_dt", "feedback_seconds", "simulation_step", "simulation_timeout", "solver_params", "binary_sha256", "cohort_id", "timing_mode", "offline_timing", "protocol_sha256"):
        if key in campaign_manifest and manifest.get(key) != campaign_manifest[key]:
            raise ValueError(f"Episode differs from campaign {key}")
    if "episode_seed" in campaign_manifest and manifest["args"]["seed"] != campaign_manifest["episode_seed"]:
        raise ValueError("Episode RNG seed differs from campaign")
    if "input_sha256" in campaign_manifest and manifest.get("source_input_sha256") != campaign_manifest["input_sha256"]:
        raise ValueError("Episode source-input hashes differ from campaign")
    sources = campaign_manifest.get("source_sha256", {})
    for name, actual in manifest.get("source_sha256", {}).items():
        if name in sources and actual != sources[name]:
            raise ValueError(f"Episode source hash differs from campaign: {name}")
    fixed = campaign_manifest.get("fixed_runner_args", [])
    for i in range(0, len(fixed), 2):
        key = fixed[i].removeprefix("--").replace("-", "_")
        actual = manifest["args"].get(key)
        expected = float(fixed[i+1])
        if key == "knot_dts":
            actual = actual[0] if isinstance(actual, list) and len(actual) == 1 else None
        if actual != expected:
            raise ValueError(f"Episode differs from fixed runner argument {key}")
    selected = next((p for p in campaign_manifest.get("pairs", []) if p["pair_id"] == manifest["pair_id"]), None)
    if selected and (manifest["start_pool_index"] != selected["start_index"] or manifest["goal_pool_index"] != selected["goal_index"]):
        raise ValueError("Episode start/goal pool indices differ from campaign")


def episode(campaign, record, robot_limits, campaign_manifest):
    completed = [a for a in record.get("attempts", []) if a.get("state") == "completed"]
    if not completed:
        return None
    attempt = completed[-1]
    relative = attempt.get("summary_path")
    folder = (campaign/relative).parent if relative else campaign/attempt["relative_output"]
    summary = json.loads((folder/"summary.json").read_text())
    if not summary["complete"]:
        raise ValueError(f"Completed attempt has incomplete summary: {folder}")
    if (int(summary["pair_id"]), int(summary["batch"])) != (int(record["pair_id"]), int(record["batch"])):
        raise ValueError(f"Episode identity mismatch: {folder}")
    receipt = json.loads((folder/"receipt.json").read_text())
    hashes = {}
    for name in ("summary.json", "manifest.json", "inputs.npz", "executed.npz"):
        hashes[name] = digest(folder/name)
        if hashes[name] != receipt[name]:
            raise ValueError(f"Receipt mismatch: {folder/name}")
    manifest = json.loads((folder/"manifest.json").read_text())
    validate_campaign_episode(manifest, campaign_manifest)
    model_hash = manifest.get("source_sha256", {}).get("gato/dynamics/tiago_right/tiago_right_arm.urdf")
    if model_hash is not None and model_hash != digest(URDF):
        raise ValueError("Local URDF differs from recorded episode; cannot audit duration fractions")
    with np.load(folder/"executed.npz") as a:
        states, controls, times = (a[k].copy() for k in ("states", "controls", "times"))
    if states.shape != (len(times), 14) or controls.shape != (len(times)-1, 7):
        raise ValueError(f"Invalid executed arrays: {folder}")
    steps = np.diff(times)
    if not np.isfinite(times).all() or np.any(steps <= 0):
        raise ValueError(f"Invalid recorded times: {folder}")
    lower, upper, velocity, effort = robot_limits
    q, v = states[:, :7], states[:, 7:]
    position_bad = np.any(np.maximum(lower-q, q-upper) > 1e-6, axis=1)
    speed_bad = np.any(np.abs(v)/velocity > 1+1e-6, axis=1)
    torque_bad = np.any(np.abs(controls)/effort > 1+1e-6, axis=1)
    interval_masks = dict(position=position_bad[:-1] | position_bad[1:],
                          speed=speed_bad[:-1] | speed_bad[1:], torque=torque_bad,
                          nonfinite=(~np.isfinite(states).all(axis=1))[:-1] |
                          (~np.isfinite(states).all(axis=1))[1:] | ~np.isfinite(controls).all(axis=1))
    duration = float(steps.sum())
    fractions = {name: float(steps[mask].sum()/duration) if duration else None for name, mask in interval_masks.items()}
    p = summary["physical"]
    result = dict(pair_id=int(record["pair_id"]), batch=int(record["batch"]),
                  episode_path=str(folder.resolve()), outcome=summary["outcome"],
                  reached=summary["outcome"] == "reached", physical_valid=bool(p["physical_valid"]),
                  valid_reach=summary["outcome"] == "reached" and bool(p["physical_valid"]),
                  reached_seconds=summary.get("reached_seconds"), simulated_seconds=summary["simulated_seconds"],
                  controller_failure=summary.get("controller_failure", summary.get("failure")),
                  selection_failure=summary.get("selection_failure"),
                  timing_mode=manifest.get("timing_mode"), timing_scope=manifest.get("timing_scope"),
                  overheads=summary.get("overheads"),execution_intervals=summary.get("execution_intervals"),
                  executed_quadratic_cost=summary.get("quadratic_executed_cost", summary.get("total_integrated_cost")),
                  max_position_excess_deg=float(np.rad2deg(number(p.get("max_joint_bound_excess_rad")))) if np.isfinite(q).all() else np.nan,
                  max_speed_ratio=p.get("max_velocity_ratio"), max_torque_ratio=p.get("max_effort_ratio"),
                  limit_validity=dict(position=bool(np.isfinite(q).all() and p["joint_bounds_valid"]),
                                      speed=bool(np.isfinite(v).all() and p["velocity_valid"]),
                                      torque=bool(np.isfinite(controls).all() and p["effort_valid"])),
                  sampled_violation_interval_fractions=fractions,
                  provenance=dict(attempt=attempt.get("attempt"), receipt_sha256=digest(folder/"receipt.json"),
                                  verified_file_sha256=hashes, source_input_sha256=manifest.get("source_input_sha256")))
    with np.load(folder/"inputs.npz") as a:
        identity = tuple(a[k].copy() for k in ("initial_state", "target_xyz", "target_rotation"))
    config = dict(solver_params=manifest["solver_params"],timing_mode=manifest.get("timing_mode"),
                  offline_timing=manifest.get("offline_timing"),protocol_sha256=manifest.get("protocol_sha256"),knot_dt=manifest["knot_dt"],
                  prediction_seconds=manifest["prediction_seconds"], feedback_seconds=manifest["feedback_seconds"],
                  simulation_step=manifest["simulation_step"], simulation_timeout=manifest["simulation_timeout"],
                  success=manifest["success"], prefix_guard=manifest["args"].get("prefix_guard", False))
    config.update({key: manifest.get(key) for key in ("N", "binary_sha256", "source_sha256", "source_input_sha256",
                   "initialization", "proposal", "dual_policy", "rho_policy", "selection", "physical_limits", "cost",
                   "python", "numpy", "pinocchio")})
    config["episode_seed"] = manifest["args"].get("seed")
    return result, identity, config


def gain(a, b):
    x, y = number(a), number(b)
    return float((x-y)/x*100) if np.isfinite(x) and np.isfinite(y) and x > 0 else None


def gain_omission(a, b):
    x, y = number(a), number(b)
    if not np.isfinite(x) or not np.isfinite(y):
        return "nonfinite_or_missing_value"
    if x == 0:
        return "zero_B1_denominator"
    if x < 0:
        return "negative_B1_denominator"
    return None


def draw_hist(ax, values, title, xlabel, color, threshold=None, shared_bins=None, logarithmic=False):
    counts = distribution(values)
    finite = np.asarray([number(v) for v in values]); finite = finite[np.isfinite(finite)]
    if len(finite):
        bins = shared_bins if shared_bins is not None else min(20, max(5, int(np.sqrt(len(finite)))*2))
        ax.hist(finite, bins=bins, color=color, edgecolor="white")
    else:
        ax.text(.5, .5, "No eligible finite observations", ha="center", va="center", transform=ax.transAxes)
    if threshold is not None:
        ax.axvline(threshold, color="#475467", lw=1, ls="--")
    if logarithmic:
        ax.set_xscale("symlog", linthresh=1)
        xlabel += "\nlog scale beyond 1; log-spaced bins"
    if shared_bins is not None:
        low, high = shared_bins[0], shared_bins[-1]
        if threshold is not None:
            low, high = min(low, threshold), max(high, threshold)
        ax.set_xlim(low, high)
    ax.set_title(title)
    ax.set_xlabel(xlabel); ax.set_ylabel("Episodes" if "B" in title else "Pairs")
    ax.grid(axis="y", alpha=.2)
    ax.text(.98, .97, f"n={counts['total']}; finite={counts['finite']}\n"
            f"NaN/missing={counts['nan_or_missing']}; +∞={counts['positive_infinity']}; −∞={counts['negative_infinity']}",
            ha="right", va="top", transform=ax.transAxes, fontsize=8)


def figure_outputs(output, pairs, episodes):
    categories = [("reached", True, "Reached within checked bounds", "#198754"),
                  ("reached", False, "Reached with limit breaches", "#e09b32"),
                  ("timeout", True, "Timed out within checked bounds", "#8c9ba5"),
                  ("timeout", False, "Timed out with limit breaches", "#c84d54"),
                  ("failed", None, "Numerical failure", "#8465a7")]
    fig, ax = plt.subplots(figsize=(11, 4.4))
    left = np.zeros(2)
    for outcome, valid, label, color in categories:
        counts = np.array([sum(e["batch"] == b and e["outcome"] == outcome and
                               (valid is None or e["physical_valid"] == valid) for e in episodes)
                           for b in (1, 16)])
        ax.barh([0, 1], counts, left=left, color=color, label=label, height=.5)
        for y, count in enumerate(counts):
            if count:
                ax.text(left[y]+count/2, y, str(count), ha="center", va="center", color="white", fontweight="bold")
        left += counts
    ax.set_yticks([0, 1], ["B1", "B16"]); ax.invert_yaxis()
    ax.set_xlim(0, max(1, left.max())); ax.set_xlabel("Completed episodes")
    ax.set_title("Goal reaching and physical bounds are separate outcomes\n"
                 "Position, speed and torque checks; no collision checks", fontsize=12)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.2), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(output/f"outcomes.{ext}", dpi=160)
    plt.close(fig)
    eligible = [p for p in pairs if p["both_valid_reach"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, key, title in zip(axes, ("cost_gain_percent", "reach_time_gain_percent"),
                            ("Executed quadratic cost reduction", "Time-to-reach reduction")):
        draw_hist(ax, [p[key] for p in eligible], title, "(B1 − B16) / B1 × 100 [%]", COLORS[16], 0.)
    fig.suptitle(f"Both methods reached and satisfied checked bounds: {len(eligible)} eligible pairs\n"
                 "Cost integrated until each termination; positive gain favors B16", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .9))
    for ext in ("png", "pdf"):
        fig.savefig(output/f"paired_performance.{ext}", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(12, 11))
    for row, (key, label, threshold) in enumerate((("max_position_excess_deg", "Maximum position excess [degrees]", 0.),
                                                 ("max_speed_ratio", "Maximum speed / URDF limit", 1.),
                                                 ("max_torque_ratio", "Maximum torque / URDF limit", 1.))):
        pooled = np.asarray([number(e[key]) for e in episodes]); pooled = pooled[np.isfinite(pooled)]
        logarithmic = bool(len(pooled) and pooled.min() >= 0 and pooled.max() > 10 and pooled.max() > pooled.min())
        bins = np.histogram_bin_edges(pooled, bins=min(20, max(5, int(np.sqrt(len(pooled)))*2))) if len(pooled) else np.array([0., 1.])
        if logarithmic:
            bins = np.expm1(np.linspace(np.log1p(pooled.min()), np.log1p(pooled.max()), 21))
            bins[0], bins[-1] = pooled.min(), pooled.max()
        for col, batch in enumerate((1, 16)):
            values = [e[key] for e in episodes if e["batch"] == batch]
            draw_hist(axes[row, col], values, f"B{batch} · {label}", label, COLORS[batch], threshold, bins, logarithmic)
            limit_name = ("position", "speed", "torque")[row]
            within = sum(e["limit_validity"][limit_name] for e in episodes if e["batch"] == batch)
            axes[row, col].text(.98, .83, f"Within this limit: {within}/{len(values)}",
                               ha="right", va="top", transform=axes[row, col].transAxes, fontsize=9)
        maximum_count = max(axes[row, col].get_ylim()[1] for col in range(2))
        for ax in axes[row]:
            ax.set_ylim(0, maximum_count)
    fig.suptitle("All completed episodes per method, including timeouts and numerical failures\n"
                 "Technical failures have no completed trajectory and are counted separately", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94))
    for ext in ("png", "pdf"):
        fig.savefig(output/f"limit_distributions.{ext}", dpi=160)
    plt.close(fig)


def candidate(pair, a, b, tier, reason):
    return dict(pair_id=pair["pair_id"], tier=tier, reason=reason,
                b1_episode_path=a["episode_path"], b16_episode_path=b["episode_path"],
                cost_gain_percent=pair["cost_gain_percent"], reach_time_gain_percent=pair["reach_time_gain_percent"],
                b1_outcome=a["outcome"], b16_outcome=b["outcome"],
                b1_physical_valid=a["physical_valid"], b16_physical_valid=b["physical_valid"],
                position_excess_reduction_deg=number(a["max_position_excess_deg"])-number(b["max_position_excess_deg"]),
                speed_ratio_reduction=number(a["max_speed_ratio"])-number(b["max_speed_ratio"]),
                torque_ratio_reduction=number(a["max_torque_ratio"])-number(b["max_torque_ratio"]))


def fmt(value):
    return "—" if value is None or not np.isfinite(number(value)) else f"{float(value):.3f}"


def analyze(campaign, output):
    manifest_bytes = (campaign/"manifest.json").read_bytes()
    status_bytes = (campaign/"status.json").read_bytes()
    manifest, status = json.loads(manifest_bytes), json.loads(status_bytes)
    if status.get("manifest_sha256") != hashlib.sha256(manifest_bytes).hexdigest():
        raise ValueError("Campaign status manifest hash does not match loaded manifest")
    pair_ids = [int(x) for x in manifest["pair_ids"]]
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError("Duplicate planned pair IDs")
    batches = manifest.get("batches", [1, 16])
    if sorted(batches) != [1, 16]:
        raise ValueError("Analyzer expects planned B1 and B16")
    records = status["episodes"]
    keys = [(int(r["pair_id"]), int(r["batch"])) for r in records]
    if len(set(keys)) != len(keys) or any(p not in pair_ids or b not in batches for p, b in keys):
        raise ValueError("Duplicate or unplanned status episodes")
    record_map = dict(zip(keys, records))
    loaded, errors = {}, []
    robot_limits = limits()
    for record in records:
        try:
            item = episode(campaign, record, robot_limits, manifest)
            if item:
                loaded[(int(record["pair_id"]), int(record["batch"]))] = item
        except (OSError, ValueError, KeyError) as error:
            errors.append(dict(pair_id=record["pair_id"], batch=record["batch"], error=str(error)))
    episodes = [v[0] for v in loaded.values()]
    pairs, wins, regressions = [], [], []
    contingency = Counter({k: 0 for k in ("both_valid_reach", "B16_only_valid_reach", "B1_only_valid_reach", "neither_valid_reach")})
    goal_contingency = Counter({k: 0 for k in ("both_reached", "B16_only_reached", "B1_only_reached", "neither_reached")})
    paired_available = 0
    for pair_id in pair_ids:
        row = dict(pair_id=pair_id, paired_completed=False, matched_protocol=None, both_valid_reach=False,
                   cost_gain_percent=None, reach_time_gain_percent=None,
                   cost_gain_omission_reason="pair_not_eligible", reach_time_gain_omission_reason="pair_not_eligible")
        for batch in (1, 16):
            e = loaded.get((pair_id, batch))
            current_record = record_map.get((pair_id, batch), {})
            row[f"B{batch}_state"] = current_record.get("state", "unlisted")
            row[f"B{batch}_attempts"] = len(current_record.get("attempts", []))
            row.update({f"B{batch}_{k}": e[0][k] if e else None for k in
                        ("outcome", "reached", "physical_valid", "valid_reach", "executed_quadratic_cost",
                         "reached_seconds", "max_position_excess_deg", "max_speed_ratio", "max_torque_ratio", "episode_path")})
        if (pair_id, 1) not in loaded or (pair_id, 16) not in loaded:
            pairs.append(row); continue
        paired_available += 1
        a, ai, ac = loaded[(pair_id, 1)]; b, bi, bc = loaded[(pair_id, 16)]
        row["paired_completed"] = True
        row["matched_protocol"] = ac == bc and all(np.array_equal(x, y) for x, y in zip(ai, bi))
        if not row["matched_protocol"]:
            errors.append(dict(pair_id=pair_id, error="Paired inputs or protocol differ; excluded from paired statistics"))
            pairs.append(row); continue
        row["both_valid_reach"] = a["valid_reach"] and b["valid_reach"]
        outcome = ("both_valid_reach" if row["both_valid_reach"] else "B16_only_valid_reach" if b["valid_reach"]
                   else "B1_only_valid_reach" if a["valid_reach"] else "neither_valid_reach")
        contingency[outcome] += 1
        goal_outcome = ("both_reached" if a["reached"] and b["reached"] else "B16_only_reached" if b["reached"]
                        else "B1_only_reached" if a["reached"] else "neither_reached")
        goal_contingency[goal_outcome] += 1
        if row["both_valid_reach"]:
            row["cost_gain_percent"] = gain(a["executed_quadratic_cost"], b["executed_quadratic_cost"])
            row["reach_time_gain_percent"] = gain(a["reached_seconds"], b["reached_seconds"])
            row["cost_gain_omission_reason"] = gain_omission(a["executed_quadratic_cost"], b["executed_quadratic_cost"])
            row["reach_time_gain_omission_reason"] = gain_omission(a["reached_seconds"], b["reached_seconds"])
        if outcome == "B16_only_valid_reach":
            wins.append(candidate(row, a, b, 1, "B16 reached within checked bounds; B1 did not"))
        elif outcome == "B1_only_valid_reach":
            regressions.append(candidate(row, a, b, 1, "B1 reached within checked bounds; B16 did not"))
        elif row["both_valid_reach"]:
            if row["cost_gain_percent"] is not None and row["cost_gain_percent"] >= 10:
                wins.append(candidate(row, a, b, 2, "Both valid reaches; B16 executed cost at least 10% lower"))
            elif row["reach_time_gain_percent"] is not None and row["reach_time_gain_percent"] >= 10:
                wins.append(candidate(row, a, b, 3, "Both valid reaches; B16 reaches at least 10% sooner"))
            if ((row["cost_gain_percent"] is not None and row["cost_gain_percent"] < 0) or
                    (row["reach_time_gain_percent"] is not None and row["reach_time_gain_percent"] < 0)):
                regressions.append(candidate(row, a, b, 2, "Both valid reaches; B16 has higher cost or takes longer"))
        pairs.append(row)
    def score(c):
        metrics = ([c["position_excess_reduction_deg"], c["speed_ratio_reduction"], c["torque_ratio_reduction"]]
                   if c["tier"] == 1 else [c["reach_time_gain_percent"], c["cost_gain_percent"]] if c["tier"] == 3
                   else [c["cost_gain_percent"], c["reach_time_gain_percent"]])
        return (c["tier"], *[-number(x) if np.isfinite(number(x)) else float("inf") for x in metrics], c["pair_id"])
    wins.sort(key=score)
    regressions.sort(key=lambda c: (c["tier"], number(c["cost_gain_percent"]) if c["cost_gain_percent"] is not None else 0, c["pair_id"]))
    attempts = [a for r in records for a in r.get("attempts", [])]
    counts = dict(planned_pairs=len(pair_ids), planned_episodes=2*len(pair_ids),
                  completed_episodes=len(episodes), completed_by_batch={f"B{b}": sum(e["batch"] == b for e in episodes) for b in batches},
                  paired_completed=paired_available, paired_matched=sum(contingency.values()),
                  partial_pairs=sum(((p, 1) in loaded) != ((p, 16) in loaded) for p in pair_ids),
                  pairs_without_completed_episode=sum((p, 1) not in loaded and (p, 16) not in loaded for p in pair_ids),
                  status_episode_states=dict(Counter(r["state"] for r in records)), attempt_states=dict(Counter(a["state"] for a in attempts)),
                  current_technical_failure_episodes=sum(r["state"] == "technical_failure" for r in records),
                  historical_technical_failure_attempts=sum(a["state"] == "technical_failure" for a in attempts),
                  data_integrity_errors=len(errors), both_valid_reach_pairs=contingency["both_valid_reach"])
    eligible = [p for p in pairs if p["both_valid_reach"]]
    summary = dict(schema_version=1, campaign=str(campaign.resolve()), denominators=counts,
                   outcome_contingency=dict(contingency), goal_only_contingency=dict(goal_contingency),
                   eligible_performance={key: distribution([p[key] for p in eligible]) for key in ("cost_gain_percent", "reach_time_gain_percent")},
                   eligible_gain_omissions={key: dict(Counter(p[key] for p in eligible if p[key] is not None)) for key in
                       ("cost_gain_omission_reason", "reach_time_gain_omission_reason")},
                   method_outcomes={f"B{b}": dict(Counter(e["outcome"] for e in episodes if e["batch"] == b)) for b in batches},
                   limit_distributions={f"B{b}": {key: distribution([e[key] for e in episodes if e["batch"] == b]) for key in
                       ("max_position_excess_deg", "max_speed_ratio", "max_torque_ratio")} for b in batches},
                   episodes=episodes, pair_ids=pair_ids, data_integrity_errors=errors,
                   definitions=dict(eligible="Both completed matched episodes reached and physical_valid=true",
                       gain="100*(B1-B16)/B1; finite positive B1 denominator required; negative and zero gains retained",
                       cost="Running quadratic executed cost until each episode terminates; excludes barriers and terminal costs; not fixed-duration cost or global optimality",
                       physical="All recorded executed states and controls satisfy finite, raw URDF position/speed/effort checks within existing tolerances; no collision checks",
                       goal="Recorded runner outcome using manifest goal checks; current-cohort strict position<.05m, tool-z-axis<30deg, velocity L1<1rad/s, zero dwell",
                       duration_fractions="Fraction of executed duration in sampled intervals with either endpoint position/speed violation, or applied interval torque violation; nonfinite intervals separate; not continuous-time proof",
                       nonfinite="Histograms plot finite bins and explicitly count NaN/missing and +/-Infinity separately; technical failures excluded from numeric distributions",
                       completion="Latest completed attempt per episode; ordinary solver/numerical failure with complete summary remains a completed outcome, not a technical failure"),
                   provenance=dict(manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(), status_sha256=hashlib.sha256(status_bytes).hexdigest(),
                                   analyzer_sha256=digest(Path(__file__)), urdf_sha256=digest(URDF),
                                   campaign_manifest=manifest, verification="Receipt-verified summary, manifest, inputs and executed arrays; full per-step artifact receipts preserved by campaign"))
    output.mkdir(parents=True, exist_ok=True)
    (output/"campaign_manifest_snapshot.json").write_bytes(manifest_bytes)
    (output/"campaign_status_snapshot.json").write_bytes(status_bytes)
    write_json(output/"summary.json", summary)
    write_json(output/"candidate_cases.json", dict(wins=wins, regressions=regressions,
        ranking="Wins: tier1 B16-only valid reach, ordered by position/speed/torque excess reductions; tier2 both valid >=10% cost gain; tier3 both valid >=10% time gain. Regressions include B1-only valid reaches and any negative eligible cost/time gain."))
    with (output/"pairs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]) if pairs else ["pair_id"], lineterminator="\n")
        writer.writeheader(); writer.writerows(safe(pairs))
    figure_outputs(output, pairs, episodes)
    report = ["# Reaching MPC campaign snapshot", "", f"Planned **{counts['planned_pairs']} pairs / {counts['planned_episodes']} episodes**. "
              f"Completed **{counts['completed_episodes']} episodes**; **{counts['paired_completed']} complete pairs**, "
              f"**{counts['partial_pairs']} partial pairs**, **{counts['pairs_without_completed_episode']} pairs without a completed episode**.", "",
              f"Current technical failures: **{counts['current_technical_failure_episodes']} episodes**; "
              f"historical technical failures: **{counts['historical_technical_failure_attempts']} attempts**. "
              f"Data-integrity exclusions: **{counts['data_integrity_errors']}**. Pending/running/interrupted states and all attempt counts are in summary.json.", "",
              f"Matched completed-pair denominator: **{counts['paired_matched']}**. Both reached within checked bounds: **{counts['both_valid_reach_pairs']}**.", "",
              "| Paired outcome | Count |", "|---|---:|", *[f"| {k} | {v} |" for k, v in contingency.items()], "",
              "Goal-only reaches, ignoring physical validity:", "", "| Outcome | Count |", "|---|---:|",
              *[f"| {k} | {v} |" for k, v in goal_contingency.items()], "",
              "![Executed MPC outcomes](outcomes.png)", "",
              "![Eligible paired performance](paired_performance.png)", "",
              "Only pairs where **both methods reached and satisfied all checked bounds** enter performance histograms. "
              "Gain is `100*(B1-B16)/B1`; positive favors B16, zero and negative values are retained. "
              "A finite positive B1 denominator is required. Timeouts and failed episodes never become cheap performance wins. "
              "Executed quadratic cost is integrated until each run terminates; durations can differ. "
              "It excludes barrier and terminal charges and is neither a fixed-duration comparison nor a global-optimality measurement.", "",
              "Eligible gain omissions (zero/negative denominator or nonfinite values): "
              +json.dumps(summary["eligible_gain_omissions"])+".", "",
              "![Limits for all completed episodes](limit_distributions.png)", "",
              "Limit histograms include every completed episode independently for each method, including unpaired episodes, timeouts, "
              "and completed numerical failures. Position excess is in degrees; speed and torque are ratios to raw URDF limits. "
              "Nonfinite/missing observations have explicit counts beside finite bins. Technical failures remain separate. "
              "The physical-validity check includes finite states/controls and position, speed and effort bounds; **no collisions are checked**.", "",
              "Per-episode sampled violation interval fractions are saved in summary.json: position/speed use either recorded interval endpoint, "
              "torque uses the applied interval control. Nonfinite interval duration is counted separately. These sampled fractions do not prove continuous-time validity.", "",
              "## Cases to inspect", "", "Candidates are illustrative selected cases, not additional statistical evidence. "
              "Tier 1 prioritizes B16-only valid reaches; tier 2 requires at least 10% eligible cost gain; tier 3 highlights at least 10% time gain.", "",
              "| Pair | Tier | Reason | Cost gain % | Time gain % |", "|---|---:|---|---:|---:|",
              *[f"| {c['pair_id']} | {c['tier']} | {c['reason']} | {fmt(c['cost_gain_percent'])} | {fmt(c['reach_time_gain_percent'])} |" for c in wins[:12]], "",
              "## Regressions", "", "| Pair | Reason | Cost gain % | Time gain % |", "|---|---|---:|---:|",
              *[f"| {c['pair_id']} | {c['reason']} | {fmt(c['cost_gain_percent'])} | {fmt(c['reach_time_gain_percent'])} |" for c in regressions[:12]], "",
              "## Reproduction and provenance", "", "summary.json preserves the ordered planned pair list, campaign manifest, selected episode paths, "
              "source-input hashes, verified compact artifact hashes, and the manifest/status/analyzer/URDF hashes. "
              "candidate_cases.json contains all ranked candidates and regressions with episode paths; pairs.csv has one row per planned pair.", "",
              "```bash", f"python examples/randomized_multimodal/analyze_reaching_campaign.py --campaign {campaign} --output {output}", "```", ""]
    (output/"REPORT.md").write_text("\n".join(report))
    print(json.dumps(safe(counts), indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    analyze(args.campaign, args.output)


if __name__ == "__main__":
    main()
