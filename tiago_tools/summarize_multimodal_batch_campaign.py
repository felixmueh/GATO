#!/usr/bin/env python3
"""Aggregate multimodal ``batch-benchmark`` JSON without dropping failures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _input_paths(values):
    paths = []
    for value in values:
        value = Path(value)
        if value.is_dir():
            paths.extend(sorted(value.rglob("*.json")))
        else:
            paths.append(value)
    return paths


def _case_row(path, payload):
    if payload.get("benchmark") != "tiago_multimodal_exact_candidate_batch_ab":
        raise ValueError(f"not a batch-benchmark artifact: {path}")
    raw = payload.get("raw_repeats", [])
    if not raw:
        raise ValueError(f"artifact has no measured repeats: {path}")
    batched = raw[0]["runs"]["batched"]
    first_certificate = batched["certificate"]
    first_signature = json.dumps(first_certificate, sort_keys=True, allow_nan=False)
    for repeat in raw:
        candidate = repeat["runs"]["batched"]
        if json.dumps(candidate["certificate"], sort_keys=True, allow_nan=False) != first_signature:
            raise ValueError(f"quality changed across repeats: {path}")
    support = first_certificate["certifiable_support"]
    total_support = int(sum(support.values()))
    mode_count = int(sum(count > 0 for count in support.values()))
    cap_hits = batched.get("pcg_cap_hits")
    return {
        "artifact": str(path),
        "difficulty": payload["difficulty"],
        "instance_seed": int(payload["instance"]["seed"]),
        "proposal_seed": int(payload["proposal_seed"]),
        "proposal_strategy": payload["proposal_strategy"],
        "candidate_budget": int(payload["candidate_budget"]),
        "all_parity_checks_pass": bool(
            payload["all_repeats_candidate_and_certificate_parity"]
            and payload["all_repeats_trajectory_allclose"]
        ),
        "certifiable_support": support,
        "certifiable_candidate_count": total_support,
        "covered_mode_count": mode_count,
        "at_least_one_candidate_per_mode": bool(
            support.get("clockwise", 0) >= 1
            and support.get("counterclockwise", 0) >= 1
        ),
        "strict_two_per_mode_certificate": bool(
            first_certificate["finite_budget_two_mode_coverage"]
        ),
        "ranked_certificate": bool(first_certificate["ranked_finite_budget_benchmark"]),
        "best_known_solver_objective": first_certificate[
            "best_known_solver_objective"
        ],
        "total_pcg_cap_hits": int(sum(cap_hits)) if cap_hits is not None else None,
        "independent_median_end_to_end_ms": payload["timings"]["independent"][
            "end_to_end"
        ]["median_ms"],
        "batched_median_end_to_end_ms": payload["timings"]["batched"][
            "end_to_end"
        ]["median_ms"],
        "median_end_to_end_reduction_fraction": payload[
            "paired_end_to_end_improvement"
        ]["median_reduction_fraction"],
        "bootstrap_95_ci_reduction_fraction": payload[
            "paired_end_to_end_improvement"
        ]["bootstrap_95_ci_median_reduction_fraction"],
    }


def aggregate(paths):
    rows = []
    keys = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = _case_row(path, payload)
        key = (
            row["difficulty"],
            row["instance_seed"],
            row["proposal_seed"],
            row["proposal_strategy"],
            row["candidate_budget"],
        )
        if key in keys:
            raise ValueError(f"duplicate campaign case: {key}")
        keys.add(key)
        rows.append(row)
    if not rows:
        raise ValueError("campaign contains no benchmark artifacts")
    rows.sort(key=lambda row: (row["difficulty"], row["instance_seed"], row["proposal_seed"]))
    groups = {}
    for difficulty in sorted({row["difficulty"] for row in rows}):
        group = [row for row in rows if row["difficulty"] == difficulty]
        reductions = [row["median_end_to_end_reduction_fraction"] for row in group]
        groups[difficulty] = {
            "case_count": len(group),
            "parity_pass_count": sum(row["all_parity_checks_pass"] for row in group),
            "at_least_one_per_mode_count": sum(
                row["at_least_one_candidate_per_mode"] for row in group
            ),
            "strict_two_per_mode_count": sum(
                row["strict_two_per_mode_certificate"] for row in group
            ),
            "median_end_to_end_reduction_fraction": float(np.median(reductions)),
            "minimum_end_to_end_reduction_fraction": float(np.min(reductions)),
        }
    reductions = [row["median_end_to_end_reduction_fraction"] for row in rows]
    return {
        "schema_version": 1,
        "campaign": "tiago_multimodal_batched_held_out_aggregate",
        "policy": "Every supplied case is retained, including failed coverage and PCG-cap cases.",
        "case_count": len(rows),
        "all_cases_parity_pass": all(row["all_parity_checks_pass"] for row in rows),
        "at_least_one_per_mode_rate": float(
            np.mean([row["at_least_one_candidate_per_mode"] for row in rows])
        ),
        "strict_two_per_mode_rate": float(
            np.mean([row["strict_two_per_mode_certificate"] for row in rows])
        ),
        "median_end_to_end_reduction_fraction": float(np.median(reductions)),
        "minimum_end_to_end_reduction_fraction": float(np.min(reductions)),
        "by_difficulty": groups,
        "cases": rows,
    }


def _markdown(summary):
    lines = [
        "# Multimodal batched held-out campaign",
        "",
        summary["policy"],
        "",
        f"Cases: {summary['case_count']}",
        f"All parity checks pass: {summary['all_cases_parity_pass']}",
        f"At least one candidate per mode: {summary['at_least_one_per_mode_rate']:.1%}",
        f"Strict two-per-mode certificate: {summary['strict_two_per_mode_rate']:.1%}",
        f"Median end-to-end reduction: {summary['median_end_to_end_reduction_fraction']:.1%}",
        f"Worst case end-to-end reduction: {summary['minimum_end_to_end_reduction_fraction']:.1%}",
        "",
        "| difficulty | instance | proposal | support CW/CCW | strict | cap hits | reduction |",
        "|---|---:|---:|---:|:---:|---:|---:|",
    ]
    for row in summary["cases"]:
        support = row["certifiable_support"]
        cap_hits = row["total_pcg_cap_hits"]
        lines.append(
            f"| {row['difficulty']} | {row['instance_seed']} | {row['proposal_seed']} | "
            f"{support.get('clockwise', 0)}/{support.get('counterclockwise', 0)} | "
            f"{row['strict_two_per_mode_certificate']} | "
            f"{cap_hits if cap_hits is not None else 'not recorded'} | "
            f"{row['median_end_to_end_reduction_fraction']:.1%} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    paths = _input_paths(args.inputs)
    if not paths:
        raise SystemExit("no JSON inputs found")
    summary = aggregate(paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(_markdown(summary), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
