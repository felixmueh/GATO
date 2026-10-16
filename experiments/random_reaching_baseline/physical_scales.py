#!/usr/bin/env python3
"""CPU physical scales from the retained twelve-task inputs and two saved prefixes.

No native solver is loaded. Frozen-posture torque capacities are estimates, not
trajectory feasibility, braking guarantees or physical-robot qualification.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pinocchio as pin

MODEL = "gato/dynamics/tiago_right/tiago_right_arm.urdf"
SPEEDS = (2.5, 2.25, 2.0)
FRACTIONS = (1., .9, .85)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def csv_write(path, rows):
    with Path(path).open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def acceleration_interval(column, gravity, limits):
    """Intersect all seven inequalities -limits <= gravity + column*a <= limits."""
    lower, upper = -np.inf, np.inf
    lower_joint = upper_joint = None
    for i, (coefficient, g, limit) in enumerate(zip(column, gravity, limits)):
        if abs(coefficient) < 1e-14:
            if abs(g) > limit:
                return None, None, None, None
            continue
        a, b = (-limit-g)/coefficient, (limit-g)/coefficient
        lo, hi = min(a, b), max(a, b)
        if lo > lower:
            lower, lower_joint = float(lo), i+1
        if hi < upper:
            upper, upper_joint = float(hi), i+1
    if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
        return None, None, lower_joint, upper_joint
    return lower, upper, lower_joint, upper_joint


def frozen_scales(model, pool, pairs, ids):
    data = model.createData();postures, capacities, headroom = [], [], []
    for pair in ids:
        for label, key in (("start", "start_index"), ("goal_witness", "goal_index")):
            index = int(pairs[key][pair]);q = np.asarray(pool["q"][index], dtype=float)
            upper = np.asarray(pin.crba(model, data, q)).copy()
            mass = np.triu(upper) + np.triu(upper, 1).T
            gravity = np.asarray(pin.computeGeneralizedGravity(model, data, q)).copy()
            if not np.isfinite(mass).all() or not np.isfinite(gravity).all():
                raise ValueError("Nonfinite CPU model scales")
            postures.append(dict(pair_id=pair, posture=label, pool_index=index, q_rad=q.tolist(),
                M_kg_m2=mass.tolist(), gravity_Nm=gravity.tolist(),
                minimum_mass_eigenvalue=float(np.linalg.eigvalsh(mass).min()),
                maximum_gravity_physical_effort_ratio=float(np.max(np.abs(gravity)/model.effortLimit))))
            for fraction in FRACTIONS:
                limits = fraction*model.effortLimit
                for joint in range(7):
                    headroom.append(dict(pair_id=pair, posture=label, pool_index=index, joint=joint+1,
                        joint_name=str(model.names[joint+1]), effort_fraction=fraction,
                        physical_effort_limit_Nm=float(model.effortLimit[joint]), cost_effort_limit_Nm=float(limits[joint]),
                        gravity_Nm=float(gravity[joint]), gravity_absolute_Nm=float(abs(gravity[joint])),
                        gravity_only_headroom_Nm=float(limits[joint]-abs(gravity[joint])),
                        M_diagonal_kg_m2=float(mass[joint, joint])))
                    amin, amax, lower_joint, upper_joint = acceleration_interval(mass[:, joint], gravity, limits)
                    at_rest_feasible = bool(np.all(np.abs(gravity) <= limits))
                    weaker = min(-amin, amax) if at_rest_feasible and amin is not None and amin <= 0 <= amax else None
                    row = dict(pair_id=pair, posture=label, pool_index=index, joint=joint+1,
                        effort_fraction=fraction, gravity_feasible=at_rest_feasible,
                        acceleration_min_rad_s2=amin, acceleration_max_rad_s2=amax,
                        limiting_joint_negative=lower_joint, limiting_joint_positive=upper_joint,
                        weaker_direction_acceleration_rad_s2=weaker)
                    for speed in SPEEDS:
                        name = str(speed).replace(".", "p")
                        row["braking_time_s_at_"+name] = speed/weaker if weaker and weaker > 0 else None
                        row["braking_distance_rad_at_"+name] = speed*speed/(2*weaker) if weaker and weaker > 0 else None
                    capacities.append(row)
    return postures, capacities, headroom


def sampling_clearances(model, pool, pairs, development_ids, all_ids):
    """Sampling bounds are retained pool metadata, not physical robot limits."""
    sampling_lower, sampling_upper = pool["joint_lower"], pool["joint_upper"]
    result = dict(physical_lower_rad=model.lowerPositionLimit.tolist(), physical_upper_rad=model.upperPositionLimit.tolist(),
        pool_sampling_lower_rad=sampling_lower.tolist(), pool_sampling_upper_rad=sampling_upper.tolist(),
        sampling_lower_inset_from_URDF_rad=(sampling_lower-model.lowerPositionLimit).tolist(),
        sampling_upper_inset_from_URDF_rad=(model.upperPositionLimit-sampling_upper).tolist(), populations={})
    for label, ids in (("development12", development_ids), ("frozen100", all_ids)):
        result["populations"][label] = {}
        for role, index_key in (("start", "start_index"), ("goal_witness", "goal_index")):
            indices = [int(pairs[index_key][pair]) for pair in ids]
            q = np.asarray(pool["q"][indices], dtype=float)
            row = dict(assigned=len(ids))
            for source, lower, upper in (("physical_URDF", model.lowerPositionLimit, model.upperPositionLimit),
                                         ("pool_sampling", sampling_lower, sampling_upper)):
                inset = np.minimum(q-lower, upper-q)
                i, j = np.unravel_index(int(np.argmin(inset)), inset.shape)
                row[source] = dict(minimum_inset_rad=float(inset[i, j]), pair_id=int(ids[i]), pool_index=indices[i],
                    joint=int(j+1), joint_name=str(model.names[int(j)+1]),
                    outside_postures=int((inset < 0).any(axis=1).sum()),
                    postures_outside_0p10_rad_URDF_cost_margin=int((inset < .1).any(axis=1).sum()) if source == "physical_URDF" else None)
            result["populations"][label][role] = row
    result["interpretation"] = "Pool sampling bounds are inset 0.12rad from actual URDF limits. Small margins to the sampling box must not be called physical near-limit starts. Existing inventory files remain unchanged."
    return result


def saved_prefix_scales(model, raw):
    """Two requested saved solves only; no solver call or new replay."""
    result = []
    for pair, step in ((7093, 1), (2342, 15)):
        directory = raw / f"pair_{pair:05d}"
        inputs, returned = directory/f"step_{step:04d}_input.npz", directory/f"step_{step:04d}_returned.npz"
        log = [json.loads(s) for s in (directory/"solves.jsonl").read_text().splitlines()]
        row = next(r for r in log if r["step"] == step)
        with np.load(inputs, allow_pickle=False) as z:
            actual = z["actual_state"].reshape(-1).astype(float)
        with np.load(returned, allow_pickle=False) as z:
            packed = z["optimized"].reshape(-1).astype(float)
        u, predicted = packed[14:21], packed[21:35]
        dt = row["executed_duration_s"];t0 = row["feedback_time_s"]
        with np.load(directory/"executed.npz", allow_pickle=False) as z:
            index = int(np.argmin(np.abs(z["times"]-(t0+dt))))
            if not np.isclose(z["times"][index], t0+dt, atol=1e-9, rtol=0):
                raise ValueError("Saved same-time RK4 endpoint missing")
            endpoint = z["states"][index].copy()
            ui = int(np.argmin(np.abs(z["control_times"]-t0)))
            if not np.array_equal(z["controls"][ui], u):
                raise ValueError("Saved applied control differs from returned first control")
        data = model.createData()
        acceleration = np.asarray(pin.aba(model, data, actual[:7], actual[7:], u)).copy()
        frozen_q = pin.integrate(model, actual[:7], dt*actual[7:] + .5*dt*dt*acceleration)
        frozen = np.r_[frozen_q, actual[7:]+dt*acceleration]
        metrics = {}
        for label, left, right in (("returned_vs_frozen_ABA_map", predicted, frozen),
                                   ("frozen_ABA_map_vs_saved_RK4", frozen, endpoint),
                                   ("returned_vs_saved_RK4", predicted, endpoint),
                                   ("returned_x0_vs_actual_feedback", packed[:14], actual)):
            metrics[label] = dict(maximum_q_difference_rad=float(np.max(np.abs(left[:7]-right[:7]))),
                maximum_v_difference_rad_s=float(np.max(np.abs(left[7:]-right[7:]))))
        result.append(dict(pair_id=pair, step=step, feedback_time_s=t0, endpoint_time_s=t0+dt,
            actual_feedback_state=actual.tolist(), applied_control_Nm=u.tolist(),
            CPU_ABA_acceleration_rad_s2=acceleration.tolist(), frozen_acceleration_map=frozen.tolist(),
            returned_first_future_state=predicted.tolist(), saved_RK4_endpoint=endpoint.tolist(), differences=metrics,
            input_sha256=sha(inputs), returned_sha256=sha(returned), executed_sha256=sha(directory/"executed.npz"),
            interpretation="CPU analogue of default type2 constant-acceleration map using Pinocchio ABA. This is not a native kernel evaluation or a cause attribution; differences show sizes only."))
    return result


def report(output, data, capacities, headroom):
    rows = ["# Physical scales for the retained random-reaching tasks", "",
        "These CPU estimates use the unchanged reduced URDF and the exact twelve development start configurations and twelve FK goal witnesses. They are frozen-posture scales, not full-path feasibility, braking guarantees or robot safety qualification.", "",
        "At rest, tau = g(q) + M(q) qdd. For qdd = a e_j, all other joint accelerations are held at zero; the allowable a interval is the intersection of all seven actuator torque inequalities. The weaker positive/negative capacity is used for the simple braking estimates t=v/a and d=v²/(2a). Changing configuration, Coriolis effects at speed, friction, actuator dynamics, simultaneous joint acceleration and path-dependent limits are excluded from those estimates. A goal witness is a posture sample, not an executed endpoint or joint target.", "",
        "## Gravity and torque headroom", "",
        "| Joint | Start median / max abs g (Nm) | Minimum 0.90 headroom (Nm) | Minimum 0.85 headroom (Nm) |", "|---|---:|---:|---:|"]
    for joint in range(1, 8):
        g = [r["gravity_absolute_Nm"] for r in headroom if r["joint"] == joint and r["posture"] == "start" and r["effort_fraction"] == 1.]
        h = [min(r["gravity_only_headroom_Nm"] for r in headroom if r["joint"] == joint and r["effort_fraction"] == fraction) for fraction in (.9, .85)]
        rows.append(f"| {joint} | {np.median(g):.4g} / {max(g):.4g} | {h[0]:.4g} | {h[1]:.4g} |")
    rows += ["", "Headroom minima above include both starts and goal witnesses. Physical effort limits remain43 Nm on joints1/2 and26 Nm on joints3–7. The 0.90 cost limits are38.7/23.4 Nm; 0.85 gives36.55/22.1 Nm. These are interior cost thresholds, not altered robot limits.", "",
        "## Coupled frozen-posture acceleration and braking", "",
        "The following ranges include all24 retained postures, using0.90 effort fractions and the weaker acceleration direction. Braking distance is evaluated at2.25 rad/s. All seven torque bounds are enforced in each single-joint estimate.", "",
        "| Joint | Weak acceleration min / median / max (rad/s²) | Braking distance min / median / max (rad) |", "|---|---:|---:|"]
    for joint in range(1, 8):
        selected = [r for r in capacities if r["joint"] == joint and r["effort_fraction"] == .9]
        aa = [r["weaker_direction_acceleration_rad_s2"] for r in selected if r["weaker_direction_acceleration_rad_s2"] is not None]
        dd = [r["braking_distance_rad_at_2p25"] for r in selected if r["braking_distance_rad_at_2p25"] is not None]
        rows.append(f"| {joint} | {min(aa):.4g} / {np.median(aa):.4g} / {max(aa):.4g} | {min(dd):.4g} / {np.median(dd):.4g} / {max(dd):.4g} |")
    rows += ["", "The CSV also retains physical,0.90 and0.85 effort fractions and braking times/distances at2.5,2.25 and2.0 rad/s, with the torque-limiting joint for either acceleration direction. These generous frozen-posture capacities are not achievable whole-motion acceleration guarantees.", "",
        "## Margin and feedback scales", "",
        "At2.5 rad/s, traversing0.02/0.04 rad takes8/16 ms, less than the30 ms feedback interval. Stopping from2.5 rad/s inside those margins under constant deceleration would require156.25/78.125 rad/s². At2.25 rad/s the requirements are126.56/63.28 rad/s². Thus a small q cost margin should not be interpreted as a full stopping-distance guarantee.", "",
        "Velocity headroom of0.25/0.50 rad/s corresponds to8.33/16.67 rad/s² over30 ms. Fractions0.90/0.85 of2.5 rad/s give2.25/2.125 rad/s cost thresholds and0.25/0.375 rad/s headroom (8.33/12.5 rad/s² over30 ms). A2.0 rad/s threshold would be fraction0.80.", "",
        "## Two saved prefix comparisons", "",
        "CPU ABA is evaluated once at each saved actual feedback state and applied first control. F holds that acceleration fixed for30 ms: q1=q0+v0 dt+0.5 a0 dt², v1=v0+a0 dt, matching the algebra of the native default type2 integrator. This is a CPU analogue, not a native dynamics evaluation. The RK4 endpoints are retained measurements, not new replays.", "",
        "| Case / step | Returned vs F q / v | F vs saved RK4 q / v | Returned vs RK4 q / v |", "|---|---:|---:|---:|"]
    for row in data["saved_prefixes"]:
        cells = [f"{row['differences'][key]['maximum_q_difference_rad']:.5g} / {row['differences'][key]['maximum_v_difference_rad_s']:.5g}" for key in
            ("returned_vs_frozen_ABA_map", "frozen_ABA_map_vs_saved_RK4", "returned_vs_saved_RK4")]
        rows.append(f"| {row['pair_id']} / {row['step']} | " + " | ".join(cells) + " |")
    rows += ["", "q differences are in rad and v differences in rad/s. These size comparisons distinguish the frozen-acceleration approximation scale from the returned-knot difference without diagnosing a solver defect or proving a remedy. No residual or convergence gate is added.", "",
        "## Practical interpretation", "",
        "Interior torque buffers are physically interpretable through gravity headroom and available inertial torque. The twelve starts leave substantial static gravity headroom even at0.85 effort fraction, but dynamic coupled motion can consume it. Prefer keeping effort regularization and assessing actual q/v/u histories separately. Joint margins and speed headroom operate on different time/distance scales; changing penalty weights cannot turn them into hard physical constraints. Any selected margin/weight change still requires its own bounded paired task results.", "",
        "Exact URDF, protocol, pool/pairs and saved-prefix hashes, complete M/g arrays and assumptions are in `physical_scales.json`. `gravity_headroom.csv` and `acceleration_capacities.csv` contain all cases and fractions. No GPU calls, native solver imports, new solves or plant replays were used."]
    inset = data["joint_margin_scales"]
    wide = next(r for r in inset["margins"] if r["margin_rad"] == .1)
    where = rows.index("## Two saved prefix comparisons")
    extra = ["A 0.10 rad interior cost buffer spans40 ms at the physical2.5 rad/s limit, compared with30 ms feedback and0.075 rad travel per update. At a2.0 rad/s cost threshold it spans50 ms. This is a physically motivated earlier-cost scale;0.04 rad spans only16 ms at2.5 rad/s.", "",
        f"For0.10 rad per side, the smallest cost interval remains {wide['minimum_remaining_interval_rad']:.6g} rad wide. The margin is {min(wide['per_side_fraction_of_span'])*100:.3g}–{max(wide['per_side_fraction_of_span'])*100:.3g}% of the raw joint span per side. "
        f"None of the12 retained starts or12 goal witnesses lies outside these interior bounds. Their minimum raw insets are {inset['minimum_start_inset_rad']:.6g}/{inset['minimum_goal_witness_inset_rad']:.6g} rad; no state change or task filtering is required.", ""]
    for joint in (1, 2):
        selected = [r for r in capacities if r["joint"] == joint and r["effort_fraction"] == .85]
        dd = [r["braking_distance_rad_at_2p0"] for r in selected]
        extra += [f"At0.85 effort fraction and2.0 rad/s, J{joint} frozen braking distance has median {np.median(dd):.5g} rad and maximum {max(dd):.5g} rad; {sum(d>.1 for d in dd)}/24 postures exceed0.10 rad. "
                  "The buffer therefore matches a typical proximal-joint scale rather than a worst-case braking guarantee.", ""]
    rows[where:where] = extra
    bounds = data["sampling_bound_clearances"]
    where = rows.index("## Gravity and torque headroom")
    extra = ["## Sampling bounds versus physical limits", "",
        "The retained pool's joint_lower/joint_upper arrays are sampling bounds inset 0.12 rad from each actual URDF limit. A small inset to that sampling box is not a small physical clearance. Original inventory artifacts are preserved; the following labels distinguish the two definitions.", "",
        "| Task set / role | Minimum sampling-box inset (rad) | Minimum actual-URDF inset (rad) |", "|---|---:|---:|"]
    for population, roles in bounds["populations"].items():
        for role, row in roles.items():
            extra.append(f"| {population} / {role} | {row['pool_sampling']['minimum_inset_rad']:.8g} | {row['physical_URDF']['minimum_inset_rad']:.8g} |")
    extra += ["", "None of these retained starts or goal witnesses lies outside a 0.10 rad interior cost interval defined from the actual URDF bounds. These counts concern only saved postures, not intervening trajectories.", ""]
    rows[where:where] = extra
    text = "\n".join(rows)+"\n"
    for old, new in {"remain43": "remain 43", "joints1/2": "joints 1/2", "and26": "and 26", "joints3–7": "joints 3–7",
        "are38.7": "are 38.7", "all24": "all 24", "using0.90": "using 0.90", "at2.25": "at 2.25",
        "physical,0.90": "physical, 0.90", "and0.85": "and 0.85", "at2.5": "at 2.5", "at2.0": "at 2.0",
        "At2.5": "At 2.5", "traversing0.02": "traversing 0.02", "takes8/16": "takes 8/16", "the30": "the 30",
        "require156.25": "require 156.25", "At2.25": "At 2.25", "are126.56": "are 126.56", "of0.25": "of 0.25",
        "to8.33": "to 8.33", "over30": "over 30", "Fractions0.90": "Fractions 0.90", "of2.5": "of 2.5", "give2.25": "give 2.25",
        "and0.25": "and 0.25", "A2.0": "A 2.0", "fraction0.80": "fraction 0.80", "for30": "for 30", "at0.85": "at 0.85",
        "spans40": "spans 40", "physical2.5": "physical 2.5", "with30": "with 30", "and0.075": "and 0.075",
        "a2.0": "a 2.0", "spans50": "spans 50", "scale;0.04": "scale; 0.04", "only16": "only 16",
        "For0.10": "For 0.10", "the12": "the 12", "or12": "or 12", "At0.85": "At 0.85", "and2.0": "and 2.0",
        "exceed0.10": "exceed 0.10"}.items():
        text = text.replace(old, new)
    (output/"PHYSICAL_SCALES.md").write_text(text)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "pool", "protocol", "margin-raw", "output"):
        p.add_argument("--"+key, type=Path, required=True)
    args = p.parse_args()
    protocol = json.loads(args.protocol.read_text());ids = protocol["development_task_ids"]
    if len(ids) != 12:
        raise ValueError("Expected the exact twelve-task development set")
    for name, expected in protocol["input_sha256"].items():
        if sha(args.pool/name) != expected:
            raise ValueError("Frozen task input hash differs: " + name)
    model_path = args.source/MODEL
    info = json.loads((args.margin_raw/"provenance.json").read_text())
    if sha(model_path) != info["source_sha256"][MODEL] or info["pairs"] != ids:
        raise ValueError("Physics model or assigned task identities differ from measured cohort")
    model = pin.buildModelFromUrdf(str(model_path));model.gravity.linear = np.array([0.,0.,-9.81])
    with np.load(args.pool/"pool.npz", allow_pickle=False) as pool, np.load(args.pool/"pairs.npz", allow_pickle=False) as pairs:
        postures, capacities, headroom = frozen_scales(model, pool, pairs, ids)
        pool_bounds = sampling_clearances(model, pool, pairs, ids, protocol["task_ids"])
    prefixes = saved_prefix_scales(model, args.margin_raw)
    spans = model.upperPositionLimit-model.lowerPositionLimit
    q_by_role = {label: np.asarray([r["q_rad"] for r in postures if r["posture"] == label])
                 for label in ("start", "goal_witness")}
    insets = {label: np.minimum(q-model.lowerPositionLimit, model.upperPositionLimit-q)
              for label, q in q_by_role.items()}
    margin_scales = dict(raw_joint_spans_rad=spans.tolist(),
        minimum_start_inset_rad=float(insets["start"].min()),
        minimum_goal_witness_inset_rad=float(insets["goal_witness"].min()), margins=[])
    for margin in (.02, .04, .1):
        row = dict(margin_rad=margin, minimum_remaining_interval_rad=float((spans-2*margin).min()),
                   per_side_fraction_of_span=(margin/spans).tolist())
        for label, inset in insets.items():
            row[label+"_outside_postures"] = int((inset < margin).any(axis=1).sum())
            row[label+"_outside_joint_values"] = int((inset < margin).sum())
        margin_scales["margins"].append(row)
    output = args.output.resolve();output.mkdir(parents=True, exist_ok=True)
    data = dict(created_utc=datetime.now(timezone.utc).isoformat(), provenance=dict(source=str(args.source.resolve()),
        model_sha256=sha(model_path), script_sha256=sha(__file__), protocol_sha256=sha(args.protocol),
        measured_source_commit=info["source_commit"], measured_module_sha256=info["module_sha256"],
        measured_cohort_provenance_sha256=sha(args.margin_raw/"provenance.json"), input_sha256=protocol["input_sha256"],
        integrator_source_sha256=sha(args.source/"gato/dynamics/integrator.cuh"),
        Pinocchio_version=pin.__version__, NumPy_version=np.__version__, python=sys.version),
        assigned_task_ids=ids, effort_fractions=list(FRACTIONS), braking_speeds_rad_s=list(SPEEDS),
        assumptions=["Gravity vector [0,0,-9.81] m/s²; unchanged reduced URDF",
            "Rest-posture capacities impose qdd=a*ej and every other joint acceleration zero; all seven torque inequalities intersected",
            "No changing posture, velocity/Coriolis, friction, actuator dynamics or simultaneous acceleration in capacity/braking estimates",
            "Weaker positive/negative acceleration direction used; capacities are not full-path feasibility or safety certificates",
            "CPU ABA at actual moving state includes velocity effects for the two saved-prefix comparisons only",
            "No native solver/GPU calls or new RK4 replay; returned horizons and executed endpoints loaded from retained M0 recordings"],
        postures=postures, saved_prefixes=prefixes, joint_margin_scales=margin_scales, sampling_bound_clearances=pool_bounds,
        back_of_envelope=dict(qmargin_rad=[.02,.04,.1], travel_time_ms_at_2p5=[8.,16.,40.],
            required_stop_acceleration_at_2p5_rad_s2=[156.25,78.125,31.25],
            velocity_headroom_rad_s=[.25,.5], acceleration_over_30ms_rad_s2=[.25/.03,.5/.03]))
    write(output/"physical_scales.json", data)
    csv_write(output/"acceleration_capacities.csv", capacities);csv_write(output/"gravity_headroom.csv", headroom)
    report(output, data, capacities, headroom)
    shutil.copy2(__file__, output/"physical_scales.py")
    print(json.dumps(dict(postures=len(postures), capacity_rows=len(capacities), output=str(output)), indent=2))


if __name__ == "__main__":
    main()
