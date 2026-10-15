#!/usr/bin/env python3
"""Bounded CPU cost accounting for larger joint margins and lower q penalties."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pinocchio as pin

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODEL = ROOT / "gato/dynamics/tiago_right/tiago_right_arm.urdf"
CONFIGS = [(.10, .015), (.15, .01), (.20, .005), (.15, .005), (.20, .01)]
LABELS = ["M5", "M6", "M7", "unexecuted alternative", "M8"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pose(model, data, q):
    pin.forwardKinematics(model, data, q);pin.updateFramePlacements(model, data)
    return (data.oMf[model.getFrameId("torso_lift_link")].inverse()*data.oMf[model.getFrameId("arm_right_tool_link")]).copy()


def task_gradient(model, q, target, cfg):
    data = model.createData()
    def cost(q):
        actual = pose(model, data, q)
        error = (pin.rpy.matrixToRpy(actual.rotation)-target[3:]+np.pi)%(2*np.pi)-np.pi
        return .5*cfg["q_cost"]*float(np.sum((actual.translation-target[:3])**2)) + .5*cfg["ee_orient_cost"]*float(np.sum(error**2))
    h = 1e-6
    return np.asarray([(cost(q+np.eye(7)[j]*h)-cost(q-np.eye(7)[j]*h))/(2*h) for j in range(7)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("pool", "canonical", "protocol", "output"):
        parser.add_argument("--"+key, type=Path, required=True)
    parser.add_argument("--m6-protocol", type=Path,
        help="Record the exact selected M6 source/module/protocol identity without loading its native module")
    parser.add_argument("--m7-protocol", type=Path)
    parser.add_argument("--m8-protocol", type=Path)
    parser.add_argument("--cohort-root", type=Path,
        help="Read saved summary counts only to label measured versus prepared configurations")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("cost_scaling", HERE/"cost_scaling.py")
    costs = importlib.util.module_from_spec(spec);spec.loader.exec_module(costs)
    text = (MODEL.parent/"tiago_right_plant.cuh").read_text()
    start = text.index("        template<class T>\n        __host__ __device__ constexpr T jointBarrierRelaxationDistance")
    end = text.index("        template<typename T>\n        __device__ void forwardDynamics(", start)
    if hashlib.sha256(text[start:end].encode()).hexdigest() != costs.HELPER_SHA256:
        raise ValueError("Reviewed C2 helper differs from the scalar accounting")
    protocol = json.loads(args.protocol.read_text());cfg = protocol["solver_params"]
    identities = {}
    statuses = {label:"prepared; no saved-cohort status supplied" for label in LABELS}
    statuses["unexecuted alternative"] = "unexecuted alternative; no cohort scheduled"
    for label, path, expected in (("M5", args.protocol, CONFIGS[0]), ("M6", args.m6_protocol, CONFIGS[1]),
            ("M7", args.m7_protocol, CONFIGS[2]), ("M8", args.m8_protocol, CONFIGS[4])):
        if path is None:
            continue
        selected = json.loads(path.read_text())
        if (selected["limit_penalty"]["cost_bounds"]["joint_margin_rad"], selected["solver_params"]["q_lim_cost"]) != expected:
            raise ValueError(f"{label} protocol does not match its declared margin/weight comparison")
        if {k:v for k,v in cfg.items() if k != "q_lim_cost"} != {k:v for k,v in selected["solver_params"].items() if k != "q_lim_cost"}:
            raise ValueError(f"{label} changes other solver parameters relative to baseline accounting")
        identity = dict(path=str(path.resolve()),sha256=sha(path),id=selected["id"],
            source_base=selected["source_base"],native_module_sha256=selected["native_module_sha256"],
            native_module_path=selected["native_module_path"])
        if args.cohort_root:
            summary_path = args.cohort_root/path.stem/"pilot/summary.json"
            if summary_path.exists():
                summary_bytes = summary_path.read_bytes()
                count = len(json.loads(summary_bytes))
                total = len(selected["development_task_ids"])
                statuses[label] = f"measured {count}/{total} cases saved" + ("; complete" if count == total else "; in progress")
                identity.update(saved_summary_sha256=hashlib.sha256(summary_bytes).hexdigest(),
                    saved_summary_path=str(summary_path.resolve()),saved_case_count=count)
        identity["status"] = statuses[label]
        identities[label] = identity
    for name in ("pool.npz", "pairs.npz"):
        if sha(args.pool/name) != protocol["input_sha256"][name]:
            raise ValueError("Frozen task archive differs")
    model = pin.buildModelFromUrdf(str(MODEL));data = model.createData()
    fixture = json.loads(args.canonical.read_text())
    cq, ct = np.asarray(fixture["initial_state"][:7]), np.asarray(fixture["goal"])
    with np.load(args.pool/"pool.npz", allow_pickle=False) as pool, np.load(args.pool/"pairs.npz", allow_pickle=False) as pairs:
        si, gi = int(pairs["start_index"][2342]), int(pairs["goal_index"][2342])
        q = pool["q"][si].copy();goal_q = pool["q"][gi].copy()
        target_pose = pose(model, data, goal_q)
        target = np.r_[target_pose.translation, pin.rpy.matrixToRpy(target_pose.rotation)]
        dev_q = {role: np.asarray([pool["q"][int(pairs[key][pair])] for pair in protocol["development_task_ids"]])
                 for role, key in (("start", "start_index"), ("goal_witness", "goal_index"))}
    gradients = {"canonical": task_gradient(model, cq, ct, cfg), "task2342_start": task_gradient(model, q, target, cfg)}
    summary, scalar = [], []
    for label, (margin, weight) in zip(LABELS, CONFIGS):
        lo, hi = model.lowerPositionLimit+margin, model.upperPositionLimit-margin
        bulk = np.asarray([abs(weight*costs.barrier(l+.25*(h-l),lc,hc)[1])
            for l,h,lc,hc in zip(model.lowerPositionLimit,model.upperPositionLimit,lo,hi)])
        row = dict(configuration=label, measurement_status=statuses[label], joint_margin_rad=margin, q_lim_cost=weight,
            J1_continuation_delta_rad=.01*float(hi[0]-lo[0]),
            J4_continuation_delta_rad=.01*float(hi[3]-lo[3]),
            quarter_span_bulk_absolute_gradient_min=float(bulk.min()),
            quarter_span_bulk_absolute_gradient_max=float(bulk.max()))
        for label, state in (("canonical", cq), ("task2342_start", q)):
            gradient = np.asarray([weight*costs.barrier(x,l,h)[1] for x,l,h in zip(state,lo,hi)])
            row[label+"_q4_gradient"] = float(gradient[3])
            row[label+"_q1_gradient"] = float(gradient[0])
            row[label+"_q_gradient_norm"] = float(np.linalg.norm(gradient))
            row[label+"_ratio_to_running_pose_gradient_norm"] = float(np.linalg.norm(gradient)/np.linalg.norm(gradients[label]))
        for point, angle in (("physical_upper", model.upperPositionLimit[0]),
                ("one_degree_inside_physical_upper", model.upperPositionLimit[0]-math.pi/180),
                ("five_degrees_inside_physical_upper", model.upperPositionLimit[0]-5*math.pi/180)):
            row["J1_"+point+"_gradient"] = weight*costs.barrier(angle,lo[0],hi[0])[1]
        for role, qq in dev_q.items():
            inset = np.minimum(qq-model.lowerPositionLimit, model.upperPositionLimit-qq)
            row[role+"_outside_cost_interval_postures"] = int((inset<margin).any(axis=1).sum())
        summary.append(row)
        for j in range(7):
            points = [("physical_lower", model.lowerPositionLimit[j]), ("physical_upper", model.upperPositionLimit[j]),
                ("raw_quarter_span", model.lowerPositionLimit[j]+.25*(model.upperPositionLimit[j]-model.lowerPositionLimit[j])),
                ("canonical", cq[j]), ("task2342_start", q[j])]
            if j == 0:
                points += [("one_degree_inside_physical_upper", model.upperPositionLimit[0]-math.pi/180),
                           ("five_degrees_inside_physical_upper", model.upperPositionLimit[0]-5*math.pi/180)]
            for point, x in points:
                value, gradient, curvature = costs.barrier(x,lo[j],hi[j])
                center = costs.barrier(.5*(lo[j]+hi[j]),lo[j],hi[j])[0]
                scalar.append(dict(joint=j+1, joint_margin_rad=margin,q_lim_cost=weight,point=point,q_rad=float(x),
                    physical_lower_rad=float(model.lowerPositionLimit[j]),physical_upper_rad=float(model.upperPositionLimit[j]),
                    cost_lower_rad=float(lo[j]),cost_upper_rad=float(hi[j]),
                    limit_cost_relative_to_interval_center=weight*(value-center),
                    weighted_gradient_cost_per_rad=weight*gradient,weighted_curvature_cost_per_rad2=weight*curvature))
    output = args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    for filename, rows in (("joint_margin_summary.csv",summary),("joint_margin_scalar_points.csv",scalar)):
        with (output/filename).open("w") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    meta = dict(role="development-only CPU scalar accounting; validation100 remains unexecuted and uninspected",
        provenance=dict(script_sha256=sha(__file__),cost_scaling_script_sha256=sha(HERE/"cost_scaling.py"),
            reviewed_helper_sha256=costs.HELPER_SHA256,model_sha256=sha(MODEL),
            canonical_sha256=sha(args.canonical),protocol_path=str(args.protocol.resolve()),
            protocol_sha256=sha(args.protocol),source_base=protocol["source_base"],
            input_sha256=protocol["input_sha256"],configuration_identities=identities,
            status_snapshot_utc=datetime.now(timezone.utc).isoformat()),
        canonical_q=cq.tolist(),canonical_running_pose_gradient=gradients["canonical"].tolist(),
        task2342=dict(start_pool_index=si,goal_pool_index=gi,actual_start_q=q.tolist(),
            goal_q_is_FK_witness_only=goal_q.tolist(),running_pose_gradient=gradients["task2342_start"].tolist()),
        summary=summary,units="weighted scalar objective gradients: cost/rad, not mechanical torque; q_cost=4 multiplies Cartesian position squared, not joint q squared",
        limitations=["CPU double scalar/local FK accounting, not native solver results or a new reaching experiment",
            "Raw physical limits and all initial states are unchanged; some fixed starts/canonical q4 may lie outside the soft cost interval",
            "Terminal pose gradient uses20 times the running pose weights; dynamics, feasible directions and later behavior are excluded",
            "Rows compare plausible parameter choices; no parameter grid is scheduled or chosen from validation outcomes"])
    (output/"joint_margin_tradeoffs.json").write_text(json.dumps(meta,indent=2)+"\n")
    lines=["# Larger joint margins versus reduced joint-limit weights", "",
        "CPU scalar/local FK accounting on existing development inputs only. All other M5 costs, clocks and dynamics are unchanged. No validation tasks, GPU solve or new rollout is used. Gradients below are objective cost/rad, not Nm. Cartesian q_cost=4 is not a joint-q quadratic.","",
        "The five rows retain M5/M6/M7, the unexecuted .15/.005 alternative, and M8 (.20/.01). M6→M8 changes only margin at fixed .01 weight; M7→M8 changes only weight at fixed .20 margin. Other M5 speed/torque/qd/mu settings stay fixed. Saved-cohort availability below is a timestamped snapshot; the scalar table contains no rollout outcome claims.","",
        "| Margin rad / weight | J1 physical upper | J1 1deg inside | J1 5deg inside | Canonical J4 | Task2342 start J4 |", "|---|---:|---:|---:|---:|---:|"]
    for r in summary:
        vals=[r[k] for k in ("J1_physical_upper_gradient","J1_one_degree_inside_physical_upper_gradient","J1_five_degrees_inside_physical_upper_gradient","canonical_q4_gradient","task2342_start_q4_gradient")]
        lines.append(f"| {r['joint_margin_rad']:.2f} / {r['q_lim_cost']:.3f} | "+" | ".join(f"{v:.6g}" for v in vals)+" |")
    lines += ["",f"Canonical q4={cq[3]:.8g} rad is only {cq[3]-model.lowerPositionLimit[3]:.8g} rad above its actual lower limit. Its running-pose gradient component is {gradients['canonical'][3]:.8g} cost/rad; this local comparison excludes dynamics and is not a causal timeout diagnosis.","",
        "Increasing the cost margin moves canonical q4 farther outside that soft interval. Reducing lambda can therefore still increase its gradient. The summary CSV retains full-vector norms/pose-gradient ratios and interior-start counts; scalar CSV includes quarter-span bulk gradients and curvatures for all joints. Earlier onset and smaller global bias trade against physical-boundary slope; no combination is asserted safe from this accounting.","",
        "The original100 validation assignment and old M5 preparation remain frozen and unexecuted. Any future validation configuration requires a separate reviewed preparation with the same sampled IDs."]
    lines += ["", "| Margin / weight | C2 delta J1 / J4 (rad) | Quarter-span abs gradient min / max | Task2342 start J1 |", "|---|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['joint_margin_rad']:.2f} / {r['q_lim_cost']:.3f} | {r['J1_continuation_delta_rad']:.6g} / {r['J4_continuation_delta_rad']:.6g} | {r['quarter_span_bulk_absolute_gradient_min']:.6g} / {r['quarter_span_bulk_absolute_gradient_max']:.6g} | {r['task2342_start_q1_gradient']:.6g} |")
    for label, identity in identities.items():
        lines += ["", f"{label}: {identity['status']}. Protocol `{identity['id']}` SHA256 `{identity['sha256']}`; numeric source `{identity['source_base']}`; immutable native module SHA256 `{identity['native_module_sha256']}`. No native module is loaded by this calculator."]
    (output/"JOINT_MARGIN_TRADEOFFS.md").write_text("\n".join(lines)+"\n")
    shutil.copy2(__file__,output/"joint_margin_tradeoffs.py")
    print(json.dumps(dict(output=str(output),configurations=len(summary),scalar_rows=len(scalar))))


if __name__ == "__main__":
    main()
