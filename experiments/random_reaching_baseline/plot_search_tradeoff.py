#!/usr/bin/env python3
"""Plot saved task2342 joint/speed traces without native imports or reruns."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["path.simplify"] = False

STEM = "SEARCH_TRADEOFF_2342"
MODEL = "gato/dynamics/tiago_right/tiago_right_arm.urdf"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def finite(value):
    return float(value) if np.isfinite(value) else None


def collect(raw_root, source, prefixes):
    model_path = source / MODEL
    joints = [joint for joint in ET.parse(model_path).getroot().findall("joint") if joint.get("type") != "fixed"]
    if len(joints) != 7:
        raise ValueError("Expected the unchanged seven-joint reduced model")
    lower = float(joints[0].find("limit").get("lower"))
    upper = float(joints[0].find("limit").get("upper"))
    velocity = np.array([float(joint.find("limit").get("velocity")) for joint in joints])
    rows, evidence, skipped = [], [], []
    for prefix in prefixes:
        matches = [path for path in raw_root.iterdir() if path.is_dir() and (path.name == prefix or path.name.startswith(prefix + "-"))]
        if not matches:
            skipped.append(dict(configuration=prefix, reason="not recorded")); continue
        if len(matches) != 1:
            raise ValueError("Ambiguous configuration prefix: " + prefix)
        root = matches[0]
        selected = None
        for stage in ("screen", "pilot"):
            metrics = root / stage / "pair_02342/metrics.json"
            if metrics.is_file() and read(metrics).get("complete"):
                selected = stage; break
        if selected is None:
            skipped.append(dict(configuration=root.name, reason="task2342 recording not complete")); continue
        stage_root = root / selected
        directory = stage_root / "pair_02342"
        protocol = read(stage_root / "protocol.json")
        provenance = read(stage_root / "provenance.json")
        metrics = read(directory / "metrics.json")
        summary = read(stage_root / "summary.json")
        if next((item for item in summary if item["pair_id"] == 2342), None) != metrics:
            raise ValueError("Summary/case mismatch: " + str(directory))
        if provenance["source_sha256"][MODEL] != sha(model_path):
            raise ValueError("Plot URDF differs from the measured physical model")
        cfg = protocol["solver_params"]
        with np.load(directory / "executed.npz", allow_pickle=False) as z:
            times, states = z["times"], z["states"]
            if states.shape != (len(times), 14):
                raise ValueError("Unexpected dense executed-state shape")
            clearance = np.minimum(states[:, 0]-lower, upper-states[:, 0])
            speed_ratio = np.max(np.abs(states[:, 7:])/velocity, axis=1)
            for i in range(len(times)):
                rows.append(dict(configuration=root.name, stage=selected, pair_id=2342,
                    q_limit_weight=cfg["q_lim_cost"], velocity_limit_weight=cfg["vel_lim_cost"],
                    simulated_time_s=float(times[i]), q1_rad=finite(states[i, 0]),
                    q1_physical_clearance_rad=finite(clearance[i]), maximum_speed_ratio=finite(speed_ratio[i]),
                    state_finite=int(np.isfinite(states[i]).all())))
        with np.load(directory / "inputs.npz", allow_pickle=False) as z:
            inputs = {key: hashlib.sha256(np.ascontiguousarray(z[key]).tobytes()).hexdigest()
                for key in ("initial_state", "source_goal_q", "target_xyz", "target_rotation")}
        evidence.append(dict(configuration=root.name, stage=selected, assigned_case_ids=provenance["pairs"],
            samples=len(times), first_time_s=float(times[0]), final_time_s=float(times[-1]),
            q_limit_weight=cfg["q_lim_cost"], velocity_limit_weight=cfg["vel_lim_cost"],
            outcome=metrics["outcome"], rough_arrival=metrics["task_success"], physical_pass=metrics["physical_pass"],
            raw_root=str(directory), numerical_source=protocol["source_base"], source_commit=provenance["source_commit"],
            module_sha256=provenance["module_sha256"], source_sha256=provenance["source_sha256"],
            wrapper_sha256=provenance["wrapper_sha256"], pool_inputs_sha256=provenance["inputs_sha256"],
            physical_input_array_sha256=inputs,
            hashes={name: sha(directory/name) for name in ("executed.npz", "inputs.npz", "metrics.json")},
            protocol_sha256=sha(stage_root/"protocol.json"), summary_sha256=sha(stage_root/"summary.json")))
    return rows, dict(role="Saved dense executed motion; descriptive comparison, no solver-cause attribution",
        sampling="Every saved state/time row retained; no decimation or peak filtering", model_sha256=sha(model_path),
        q1_physical_lower_rad=lower, q1_physical_upper_rad=upper, speed_limits_rad_s=velocity.tolist(),
        input_policy="Input identities are recorded per trace; stage scope and arrival stopping are retained",
        included=evidence, skipped=skipped, script_sha256=sha(__file__))


def plot(output):
    with (output / (STEM + ".csv")).open() as handle:
        rows = list(csv.DictReader(handle))
    metadata = read(output / (STEM + ".json"))
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for i, item in enumerate(metadata["included"]):
        trace = [row for row in rows if row["configuration"] == item["configuration"] and row["stage"] == item["stage"]]
        times = [float(row["simulated_time_s"]) for row in trace]
        label = f"{item['configuration'].split('-')[0]} {item['stage']}: q={item['q_limit_weight']:g}, v={item['velocity_limit_weight']:g}"
        for ax, key in zip(axes, ("q1_physical_clearance_rad", "maximum_speed_ratio")):
            values = [float(row[key]) if row[key] else np.nan for row in trace]
            ax.plot(times, values, label=label, lw=1.25, color=f"C{i}")
            ax.plot(times[-1], values[-1], marker="o", ms=3, color=f"C{i}")
    axes[0].axhline(0, color="black", ls="--", lw=.9)
    axes[1].axhline(1, color="black", ls="--", lw=.9)
    axes[0].set_ylabel("J1 physical clearance\n(rad)")
    axes[1].set_ylabel("Max speed / physical limit")
    axes[1].set_xlabel("Simulated time (s)")
    axes[1].set_xlim(0, 8)
    for ax in axes:
        ax.grid(alpha=.2); ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(.5, .94), fontsize=9)
    fig.suptitle("Task 2342: recorded joint-position and speed behavior", y=.99)
    fig.text(.08, .012, "All saved 3 ms samples retained. Negative clearance exceeds J1's physical bound.\nSpeed ratio is the maximum over seven joints. End dots mark recording stops.", fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, .84))
    for suffix in (".png", ".pdf"):
        fig.savefig(output / (STEM + suffix), dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cohorts", nargs="+", default=["M8", "M9", "M10", "M11"])
    parser.add_argument("--plot-only", type=Path)
    args = parser.parse_args()
    if args.plot_only:
        plot(args.plot_only.resolve()); return
    if args.raw_root is None or args.source is None:
        parser.error("Provide --raw-root --source, or --plot-only")
    output = (args.output or args.raw_root).resolve(); output.mkdir(parents=True, exist_ok=True)
    rows, metadata = collect(args.raw_root.resolve(), args.source.resolve(), args.cohorts)
    if not rows:
        parser.error("No complete task2342 recordings available")
    with (output / (STEM + ".csv")).open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output / (STEM + ".json")).write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    if Path(__file__).resolve() != output / Path(__file__).name:
        shutil.copy2(__file__, output / Path(__file__).name)
    plot(output)
    print(json.dumps(dict(included=[item["configuration"] for item in metadata["included"]],
        dense_samples=len(rows), skipped=metadata["skipped"], output=str(output))))


if __name__ == "__main__":
    main()
