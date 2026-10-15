#!/usr/bin/env python3
"""CPU arithmetic for the reviewed C2 cost, with no native solver imports."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PLANT = ROOT / "gato/dynamics/tiago_right"
HELPER_SHA256 = "effd36439784b4bd17931833a5e5d6e7085b55f0d7473b30a8f9baf06a693979"
# Rounded CPU Pinocchio results supplied by the independent physical-scale
# calculation for the unchanged twelve initial postures, gravity [0,0,-9.81].
# These are representative points, not independently sampled torque demands.
GRAVITY_MEDIAN_NM = [5.473, 10.461, 2.174, 6.072, .542, .667, .00104]
GRAVITY_MAX_NM = [11.395, 22.954, 8.385, 9.799, 1.400, 1.419, .00308]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def distance_cost(distance, delta):
    if distance >= delta:
        return -math.log(distance), -1 / distance, 1 / distance**2
    t = (delta - distance) / delta
    return -math.log(delta) + t + .5*t*t, -(1+t)/delta, 1/delta**2


def barrier(x, lower, upper):
    delta = .01*(upper-lower)
    lo, hi = distance_cost(x-lower, delta), distance_cost(upper-x, delta)
    return lo[0]+hi[0], lo[1]-hi[1], lo[2]+hi[2]


def row(category, joint, raw_lower, raw_upper, lower, upper, policy, weight,
        regularization, point, x, increment, unit):
    value, gradient, curvature = barrier(x, lower, upper)
    center_value = barrier(.5*(lower+upper), lower, upper)[0]
    increment_value = barrier(x+increment, lower, upper)[0]
    return dict(category=category, joint=joint, physical_lower=raw_lower,
        physical_upper=raw_upper, cost_lower=lower, cost_upper=upper,
        margin_policy=policy, penalty_weight=weight,
        regularization_weight=regularization, point=point, x=x, variable_unit=unit,
        continuation_delta=.01*(upper-lower),
        relative_limit_cost=weight*(value-center_value),
        limit_gradient=weight*gradient, limit_curvature=weight*curvature,
        ordinary_cost=.5*regularization*x*x if regularization is not None else None,
        ordinary_gradient=regularization*x if regularization is not None else None,
        ordinary_curvature=regularization,
        increment=increment, limit_cost_change=weight*(increment_value-value),
        ordinary_cost_change=.5*regularization*((x+increment)**2-x*x)
            if regularization is not None else None)


def canonical_accounting(path):
    """Optional bounded CPU FK/RPY accounting; does not construct a controller."""
    import numpy as np
    import pinocchio as pin
    fixture = json.loads(path.read_text())
    model = pin.buildModelFromUrdf(str(PLANT/"tiago_right_arm.urdf"))
    data = model.createData()
    q, target = np.asarray(fixture["initial_state"])[:7], np.asarray(fixture["goal"])
    def cost(q):
        pin.forwardKinematics(model,data,q)
        pin.updateFramePlacements(model,data)
        pose = data.oMf[model.getFrameId("torso_lift_link")].inverse()*data.oMf[model.getFrameId("arm_right_tool_link")]
        rpy_error = (pin.rpy.matrixToRpy(pose.rotation)-target[3:]+np.pi)%(2*np.pi)-np.pi
        return .5*4*float(np.sum((pose.translation-target[:3])**2))+.5*.1*float(np.sum(rpy_error**2))
    step = 1e-6
    task_gradient = np.asarray([(cost(q+np.eye(7)[j]*step)-cost(q-np.eye(7)[j]*step))/(2*step) for j in range(7)])
    groups = []
    for margin,weight in [(.02,.01),(.02,.1),(.10,.01),(.10,.015),(.10,.02),
                          (.15,.01),(.15,.005),(.20,.005),(.20,.01),(.20,.02125)]:
        bounds = list(zip(model.lowerPositionLimit+margin,model.upperPositionLimit-margin))
        gradient = np.asarray([weight*barrier(x,lo,hi)[1] for x,(lo,hi) in zip(q,bounds)])
        groups.append(dict(joint_margin_rad=margin,q_lim_cost=weight,
            q_limit_gradient_cost_per_rad=gradient.tolist(),
            norm_ratio_to_running_pose_gradient=float(np.linalg.norm(gradient)/np.linalg.norm(task_gradient)),
            gradient_cosine_with_running_pose=float(gradient@task_gradient/(np.linalg.norm(gradient)*np.linalg.norm(task_gradient)))))
    return dict(fixture=str(path),fixture_sha256=sha(path),pinocchio_version=pin.__version__,
        q=q.tolist(),raw_minimum_clearance_rad=float(np.min(np.minimum(q-model.lowerPositionLimit,model.upperPositionLimit-q))),
        running_pose_cost=cost(q),running_pose_gradient_cost_per_rad=task_gradient.tolist(),
        terminal_pose_gradient_cost_per_rad=(20*task_gradient).tolist(),finite_difference_step_rad=step,
        q_limit_comparisons=groups,
        limitation="Local scalar objective accounting at canonical start; excludes dynamics, feasible directions and later trajectory behavior; not a causal explanation for a timeout")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--canonical-preflight", type=Path,
                        help="Optionally add local canonical FK/RPY gradients using CPU Pinocchio")
    args = parser.parse_args()
    source = (PLANT / "tiago_right_plant.cuh").read_text()
    start = source.index("        template<class T>\n        __host__ __device__ constexpr T jointBarrierRelaxationDistance")
    end = source.index("        template<typename T>\n        __device__ void forwardDynamics(", start)
    if hashlib.sha256(source[start:end].encode()).hexdigest() != HELPER_SHA256:
        raise ValueError("Reviewed C2 helper changed; check this arithmetic before reuse")
    joints = [j for j in ET.parse(PLANT / "tiago_right_arm.urdf").getroot().findall("joint")
              if j.get("type") != "fixed"]
    assert len(joints) == 7
    rows = []
    for bound in [2.25, 2.0]:
        for weight in [.1, .2, 1.]:
            for qd in [.1, .05]:
                for speed in [0., 1., 2., 2.5]:
                    rows.append(row("velocity", "all", -2.5, 2.5, -bound, bound,
                        f"fraction={bound/2.5:g}", weight, qd, f"speed={speed:g}", speed, .01, "rad/s"))
    # Exact M5 additions only, rather than extending every weight/margin pairing.
    for speed in [0.,1.,2.,2.5]:
        rows.append(row("velocity","all",-2.5,2.5,-2.,2.,"fraction=0.8",
                        .6,.05,f"speed={speed:g}",speed,.01,"rad/s"))
    for index, joint in enumerate(joints):
        limits = joint.find("limit")
        lo, hi, effort = [float(limits.get(k)) for k in ["lower", "upper", "effort"]]
        q_pairs = [(margin,weight) for margin in [.02,.04,.10]
                   for weight in [.01,.1]+([.015] if margin == .10 else [])]
        # Selected joint follow-ups and the retained inactive alternative only.
        q_pairs += [(.15,.01),(.15,.005),(.20,.005),(.20,.01)]
        for margin,weight in q_pairs:
            for point, q in [("physical_lower", lo), ("physical_upper", hi)]:
                rows.append(row("joint", joint.get("name"), lo, hi, lo+margin, hi-margin,
                    f"margin={margin:g}rad", weight, None, point, q, .01, "rad"))
        for fraction in [.90, .85]:
            for weight in [.003, .01, .03]+([.02] if fraction == .85 else []):
                for point, torque in [("zero", 0.),
                        ("start_gravity_median_abs_rounded", GRAVITY_MEDIAN_NM[index]),
                        ("start_gravity_max_abs_rounded", GRAVITY_MAX_NM[index]),
                        ("physical_effort", effort)]:
                    rows.append(row("effort", joint.get("name"), -effort, effort,
                        -fraction*effort, fraction*effort, f"fraction={fraction:g}",
                        weight, .0001, point, torque, 1., "Nm"))
    # One selected M14 configuration only, not another weight/margin grid.
    m14_path = HERE / "protocols/M14-joint-speed-mid02125.json"
    m14 = json.loads(m14_path.read_text())
    m14_params = m14["solver_params"]
    m14_margin = m14["limit_penalty"]["cost_bounds"]["joint_margin_rad"]
    assert (m14_margin, m14_params["q_lim_cost"], m14_params["vel_lim_cost"]) == (.20,.02125,1.275)
    m14_rows = []
    for speed in [0.,1.,2.,2.5]:
        m14_rows.append(row("velocity","all",-2.5,2.5,-2.,2.,"fraction=0.8",
            m14_params["vel_lim_cost"],m14_params["qd_cost"],f"speed={speed:g}",speed,.01,"rad/s"))
    canonical_path = ROOT.parent / "GATO-experiment-runner-simple/experiments/inputs/batching/ba_preflight.json"
    canonical_q4 = json.loads(canonical_path.read_text())["initial_state"][3]
    # Exact actual start q4 from frozen task2342 pool entry897, not goal witness75.
    task2342_q4 = 1.0028412172525591
    for index,point,x in [(0,"physical_upper",None),(3,"canonical_start",canonical_q4),
                          (3,"task2342_actual_start_pool897",task2342_q4)]:
        limits = joints[index].find("limit")
        lo,hi = [float(limits.get(k)) for k in ["lower","upper"]]
        m14_rows.append(row("joint",joints[index].get("name"),lo,hi,lo+m14_margin,hi-m14_margin,
            "margin=0.2rad",m14_params["q_lim_cost"],None,point,hi if x is None else x,.01,"rad"))
    rows.extend(m14_rows)
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "cost_scaling.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    metadata = dict(schema="reviewed-c2-cost-scales-v1", rows=len(rows),
        numerical_source_base="85ef2ea5cf5b96726be6f4b5441f4a7c0ca041b4",
        script_sha256=sha(__file__),
        helper_sha256=HELPER_SHA256,
        source_sha256={p.name:sha(p) for p in [PLANT/"tiago_right_plant.cuh", PLANT/"tiago_right_arm.urdf",
                                             PLANT/"tiago_right_cost_limits.cuh"]},
        csv_sha256=sha(csv_path),
        relative_cost_reference="same-bound interval center; removes arbitrary log constant",
        derivative_units="cost per variable_unit and cost per variable_unit squared; never compare categories directly",
        stage_scope="one scalar at one running knot; native objective sums knots without dt scaling",
        increment_scope="finite increase x -> x+increment, including outside raw physical bounds as descriptive arithmetic",
        gravity_points=dict(source="Independent CPU physical-scale calculation, same12 frozen initial postures; rounded representative inputs",
                            absolute_median_Nm=GRAVITY_MEDIAN_NM, absolute_max_Nm=GRAVITY_MAX_NM),
        balanced_wide_candidate=dict(status="M4 combined retuning; measured task outcomes are retained separately, not a margin-only effect",
            joint_margin_rad=.10, velocity_fraction=.80, effort_fraction=.85,
            q_lim_cost=.01, vel_lim_cost=.2, ctrl_lim_cost=.01, qd_cost=.05, u_cost=.0001,
            mu=10,
            zero_speed_total_curvature=.05+.2*barrier(0.,-2.,2.)[2],
            M0_zero_speed_total_curvature=.1+.1*barrier(0.,-2.25,2.25)[2],
            physical_speed_limit_gradient=.2*barrier(2.5,-2.,2.)[1],
            M0_physical_speed_limit_gradient=.1*barrier(2.5,-2.25,2.25)[1]),
        firmer_wide_candidate=dict(status="M5 checkpoint before later joint-only follow-up; task outcomes retained separately",
            joint_margin_rad=.10,velocity_fraction=.80,effort_fraction=.85,
            q_lim_cost=.015,vel_lim_cost=.6,ctrl_lim_cost=.02,qd_cost=.05,u_cost=.0001,mu=10,
            zero_speed_total_curvature=.05+.6*barrier(0.,-2.,2.)[2],
            M1_zero_speed_total_curvature=.1+barrier(0.,-2.25,2.25)[2],
            physical_speed_limit_gradient=.6*barrier(2.5,-2.,2.)[1],
            added_scalar_rows=dict(velocity=4,joint=14,effort=28)),
        joint_margin_followup=dict(status="Later explicit user priority supersedes the M5 stopping point before validation; no 100-task validation or horizon runs launched",
            fixed_other_M5_parameters=dict(vel_lim_cost=.6,ctrl_lim_cost=.02,qd_cost=.05,u_cost=.0001,mu=10,
                                          velocity_fraction=.80,effort_fraction=.85),
            selected_first=dict(label="M6",joint_margin_rad=.15,q_lim_cost=.01),
            conditional_physical_failure=dict(label="M7",joint_margin_rad=.20,q_lim_cost=.005,status="Selected and executed after M6 physical failure; outcome remains recorded separately"),
            alternative_canonical_progress_failure=dict(joint_margin_rad=.15,q_lim_cost=.005,status="Unexecuted preparation; alternative second branch, not another queued case"),
            one_factor_followup=dict(label="M8",joint_margin_rad=.20,q_lim_cost=.01,
                comparison="M7/M8 changes only weight; M6/M8 changes only joint margin; selected after M7 physical failure"),
            added_scalar_rows=56,velocity_curve="M6/M7/M8 unchanged from M5; no duplicate curve"),
        selected_M14_accounting=dict(
            status="Selected development candidate; full twelve-task pilot pending at accounting snapshot, not a final freeze or qualification",
            protocol_path=str(m14_path),protocol_sha256=sha(m14_path),
            protocol_id=m14["id"],source_base=m14["source_base"],
            native_module_sha256=m14["native_module_sha256"],
            joint_margin_rad=m14_margin,solver_params=m14_params,
            canonical_fixture=str(canonical_path),canonical_fixture_sha256=sha(canonical_path),
            task2342_start_pool_index=897,task2342_actual_start_q4_rad=task2342_q4,
            frozen_input_sha256=m14["input_sha256"],added_scalar_rows=len(m14_rows),
            J1_physical_upper_gradient=m14_rows[4]["limit_gradient"],
            canonical_J4_gradient=m14_rows[5]["limit_gradient"],
            task2342_start_J4_gradient=m14_rows[6]["limit_gradient"],
            zero_speed_total_curvature=m14_rows[0]["limit_curvature"]+m14_rows[0]["ordinary_curvature"],
            physical_speed_limit_gradient=m14_rows[3]["limit_gradient"],
            physical_speed_total_gradient=m14_rows[3]["limit_gradient"]+m14_rows[3]["ordinary_gradient"],
            J1_C2_transition_physical_clearance_rad=m14_margin+m14_rows[4]["continuation_delta"],
            sampled_J1_braking_plus_feedback_travel_rad=.132+.03*2.,
            physical_scale_limitation="Sampled frozen-posture braking plus one 30ms prefix at 2rad/s is motivation, not a coupled-dynamics guarantee",
            velocity_figure="Original M0/M1/M4/M5 checkpoint figure retained; M14 represented numerically only"),
        merit_defect_examples=[dict(component=unit, absolute_defect=delta,
            merit_at_mu1=delta, merit_at_mu10=10*delta, merit_at_mu100=100*delta)
            for unit,delta in [("one joint q [rad]", .001), ("one joint velocity [rad/s]", .01),
                               ("one joint velocity [rad/s]", .1)]],
        limitations=["CPU scalar arithmetic, not native numerical qualification or a physical guarantee",
                     "mu mixes unnormalized position and velocity defects and sums them across knots",
                     "joint position cost weight q_cost acts on Cartesian meters; there is no matching ordinary joint q quadratic",
                     "Representative gravity points are rounded; use the separate physical-scale records for their full provenance"])
    if args.canonical_preflight:
        metadata["canonical_accounting"] = canonical_accounting(args.canonical_preflight.resolve())
    (args.output/"cost_scaling.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps(dict(rows=len(rows), csv=str(csv_path), helper_sha256=HELPER_SHA256)))


if __name__ == "__main__":
    main()
