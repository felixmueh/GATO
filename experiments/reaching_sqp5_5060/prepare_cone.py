#!/usr/bin/env python3
"""Freeze Cartesian cone targets using CPU position IK; never load a solver."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET


SOURCE = Path(__file__).resolve().parents[2]
MODEL = "gato/dynamics/tiago_right/tiago_right_arm.urdf"
ROOT_FRAME = "torso_lift_link"
TOOL_FRAME = "arm_right_tool_link"
INITIAL_Q = [-.5, -1.4, -.3, -1.7, 0., -.9, 0.]
TARGET_COUNT = 100
SAMPLER_SEED = 20261016
HALF_ANGLE_DEG = 45.
MAX_IK_CANDIDATES = 10000
MAX_CARTESIAN_PROPOSALS = 1000000
MAX_CPU_WALL_S = 3600.


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def enclosing_ball(urdf):
    """A serial-chain triangle-inequality bound, independent of joint angles.

    The first actuated joint origin is fixed in the root frame. Every subsequent
    joint-origin translation contributes its norm. Fixed tool offsets contribute
    too. All intervening rotations preserve these norms, so no FK endpoint can
    lie outside this ball. No sampled FK maximum is used as a workspace bound.
    """
    import numpy as np

    robot = ET.parse(urdf).getroot()
    by_child = {j.find("child").attrib["link"]: j for j in robot.findall("joint")}
    chain = []
    child = TOOL_FRAME
    while child != ROOT_FRAME:
        if child not in by_child:
            raise ValueError("Tool is not connected to the declared root")
        joint = by_child[child]
        chain.append(joint)
        child = joint.find("parent").attrib["link"]
    chain.reverse()
    first = next(i for i, j in enumerate(chain) if j.attrib["type"] != "fixed")
    rotation, center = np.eye(3), np.zeros(3)
    terms = []
    for index, joint in enumerate(chain):
        if joint.attrib["type"] not in ("fixed", "revolute", "continuous"):
            raise ValueError("Enclosing-ball proof requires a rotation-only serial chain")
        origin = joint.find("origin")
        attrs = origin.attrib if origin is not None else {}
        xyz = np.fromstring(attrs.get("xyz", "0 0 0"), sep=" ")
        rpy = np.fromstring(attrs.get("rpy", "0 0 0"), sep=" ")
        if xyz.shape != (3,) or rpy.shape != (3,) or not np.isfinite(np.r_[xyz, rpy]).all():
            raise ValueError("Invalid URDF origin")
        if index <= first:
            center += rotation @ xyz
            r, p, y = rpy
            cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
            rz = np.array([[cy, -sy, 0.], [sy, cy, 0.], [0., 0., 1.]])
            ry = np.array([[cp, 0., sp], [0., 1., 0.], [-sp, 0., cp]])
            rx = np.array([[1., 0., 0.], [0., cr, -sr], [0., sr, cr]])
            rotation = rotation @ rz @ ry @ rx
        terms.append(dict(joint=joint.attrib["name"], type=joint.attrib["type"],
                          parent=joint.find("parent").attrib["link"],
                          child=joint.find("child").attrib["link"],
                          origin_xyz_m=xyz.tolist(), origin_rpy_rad=rpy.tolist(),
                          translation_norm_m=float(np.linalg.norm(xyz)),
                          contributes_to_radius=index > first))
    radius = math.fsum(t["translation_norm_m"] for t in terms if t["contributes_to_radius"])
    if not radius > 0:
        raise ValueError("Empty arm reach bound")
    return dict(center_torso_m=center.tolist(), radius_m=radius, chain=terms,
                first_actuated_joint=chain[first].attrib["name"],
                proof="The first actuated joint origin is fixed in torso coordinates. The tool displacement from it is a sum of subsequent joint-origin translations transformed by rotations. Triangle inequality and rotation invariance of Euclidean norm bound every FK endpoint by the sum of their norms, including the fixed tool offset.",
                artificial_radial_cutoff=False,
                limitation="A conservative containing volume, not a characterization of reachable positions.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--half-angle-deg", type=float, default=HALF_ANGLE_DEG,
                        help="Explicit cone half-angle; full opening is twice this value")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SAMPLER_SEED)
    parser.add_argument("--count", type=int, default=TARGET_COUNT)
    parser.add_argument("--max-ik-candidates", type=int, default=MAX_IK_CANDIDATES)
    parser.add_argument("--max-cartesian-proposals", type=int, default=MAX_CARTESIAN_PROPOSALS)
    parser.add_argument("--max-cpu-wall-s", type=float, default=MAX_CPU_WALL_S)
    args = parser.parse_args()
    if not (0 < args.half_angle_deg < 90 and args.count > 0 and args.max_ik_candidates > 0
            and args.max_cartesian_proposals > 0 and args.max_cpu_wall_s > 0):
        parser.error("Require angle in (0,90), positive counts and a positive wall allowance")
    # CPU geometry dependencies are imported only after command-line validation.
    import numpy as np
    import pinocchio as pin
    from scipy.optimize import least_squares

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    ball = enclosing_ball(SOURCE / MODEL)
    center, radius = np.asarray(ball["center_torso_m"]), ball["radius_m"]
    model = pin.buildModelFromUrdf(str(SOURCE / MODEL))
    data = model.createData()
    torso, tool = model.getFrameId(ROOT_FRAME), model.getFrameId(TOOL_FRAME)
    lo, hi = model.lowerPositionLimit.copy(), model.upperPositionLimit.copy()
    q0 = np.asarray(INITIAL_Q)
    if model.nq != 7 or not (np.all(q0 >= lo) and np.all(q0 <= hi)):
        raise ValueError("Unexpected arm or invalid comfortable posture")

    def pose(q):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        return (data.oMf[torso].inverse() * data.oMf[tool]).copy()

    initial_pose = pose(q0)
    apex = initial_pose.translation.copy()
    target_rng = np.random.default_rng(np.random.SeedSequence([args.seed, 0]))
    ik_rng = np.random.default_rng(np.random.SeedSequence([args.seed, 1]))
    cosine = math.cos(math.radians(args.half_angle_deg))
    config = dict(seed=args.seed, half_angle_deg=args.half_angle_deg,
                  full_opening_deg=2 * args.half_angle_deg, count=args.count,
                  axis_torso=[1., 0., 0.], apex_xyz_torso_m=apex.tolist(),
                  initial_q_rad=INITIAL_Q, initial_velocity_rad_s=[0.] * 7,
                  frame=ROOT_FRAME, enclosing_ball=ball,
                  proposal_measure="Uniform Cartesian volume in the URDF enclosing ball intersected with the cone; implemented by uniform bounding-cube proposals and geometric rejection.",
                  accepted_measure="Proposal measure conditioned on a bounded numerical position-IK witness; not certified uniform sampling of the entire reachable workspace.",
                  acceptance="Position residual <=1 mm and actual raw URDF joint bounds. No orientation, soft-margin, collision, gravity, dynamics or MPC-outcome filter.",
                  already_inside_arrival_region="Retain and flag position distance <=50 mm; never reject based on distance.",
                  IK_random_restarts=5, IK_max_nfev_per_start=200,
                  IK_solver_tolerance=1e-6, IK_position_tolerance_m=.001,
                  max_ik_candidates=args.max_ik_candidates,
                  max_cartesian_proposals=args.max_cartesian_proposals,
                  max_cpu_wall_s=args.max_cpu_wall_s,
                  wall_check="Checked between proposals and bounded IK calls; the final bounded call may finish after the allowance.",
                  radius_min_m=None, radius_max_m=None,
                  source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True).strip(),
                  model_sha256=hashlib.sha256((SOURCE / MODEL).read_bytes()).hexdigest(),
                  no_native_solver_or_GPU_calls=True)
    write(output / "SAMPLER_PLAN.json", config)
    targets, ik_count, starts_count = [], 0, 0
    stop_reason = "cartesian_proposal_budget_exhausted"
    proposals = 0
    with (output / "all_cartesian_proposals.jsonl").open("w") as geometry_log, (output / "all_IK_attempts.jsonl").open("w") as ik_log:
        for proposal_id in range(args.max_cartesian_proposals):
            if time.perf_counter() - began >= args.max_cpu_wall_s:
                stop_reason = "cpu_wall_allowance_exhausted"
                break
            xyz = center + radius * (2 * target_rng.random(3) - 1)
            displacement = xyz - apex
            distance = float(np.linalg.norm(displacement))
            inside_ball = bool(np.linalg.norm(xyz - center) <= radius)
            inside_cone = bool(distance > 0 and displacement[0] >= distance * cosine)
            proposal = dict(proposal_id=proposal_id, xyz_torso_m=xyz.tolist(),
                            inside_enclosing_ball=inside_ball, inside_cone=inside_cone)
            geometry_log.write(json.dumps(proposal, allow_nan=False) + "\n")
            proposals += 1
            if not (inside_ball and inside_cone):
                continue
            if ik_count == args.max_ik_candidates:
                stop_reason = "IK_candidate_budget_exhausted"
                break
            ik_count += 1
            seeds = [q0.copy(), *ik_rng.uniform(lo, hi, (5, 7))]
            attempts, chosen = [], None
            for restart, seed in enumerate(seeds):
                if time.perf_counter() - began >= args.max_cpu_wall_s:
                    stop_reason = "cpu_wall_allowance_exhausted"
                    break
                sol = least_squares(lambda q: pose(q).translation - xyz, seed,
                                    bounds=(lo, hi), max_nfev=200,
                                    ftol=1e-6, xtol=1e-6, gtol=1e-6)
                q = sol.x.copy()
                error = float(np.linalg.norm(pose(q).translation - xyz))
                valid = bool(np.isfinite(q).all() and np.all(q >= lo) and np.all(q <= hi) and error <= .001)
                starts_count += 1
                attempts.append(dict(restart=restart, seed_q_rad=seed.tolist(),
                                     solution_q_rad=q.tolist(), position_residual_m=error,
                                     nfev=int(sol.nfev), solver_status=int(sol.status),
                                     solver_success=bool(sol.success), message=str(sol.message),
                                     accepted_position_witness=valid))
                if valid:
                    chosen = q
                    break
            record = dict(proposal_id=proposal_id, IK_candidate_id=ik_count - 1,
                          xyz_torso_m=xyz.tolist(), seed_bank_q_rad=[q.tolist() for q in seeds],
                          attempts=attempts, accepted=chosen is not None,
                          rejection=None if chosen is not None else "No witness found within declared CPU/IK allowance; not proof of unreachability")
            if chosen is not None:
                target = dict(id=len(targets), name=f"cone_{len(targets):03d}",
                              proposal_id=proposal_id, xyz_torso_m=xyz.tolist(),
                              displacement_from_initial_tool_m=distance,
                              cone_angle_deg=math.degrees(math.acos(float(np.clip(displacement[0] / distance, -1, 1)))),
                              already_inside_arrival_region=distance <= .05,
                              IK_witness_q_rad=chosen.tolist(),
                              IK_position_residual_m=attempts[-1]["position_residual_m"],
                              IK_selected_restart=attempts[-1]["restart"],
                              minimum_raw_joint_clearance_rad=float(np.minimum(chosen - lo, hi - chosen).min()))
                targets.append(target)
                record["accepted_target_id"] = target["id"]
            ik_log.write(json.dumps(record, allow_nan=False) + "\n")
            ik_log.flush()
            if len(targets) == args.count:
                stop_reason = "requested_target_count_complete"
                break
            if stop_reason == "cpu_wall_allowance_exhausted":
                break
    complete = len(targets) == args.count
    manifest = dict(config, targets=targets, complete=complete, stop_reason=stop_reason,
                    cartesian_proposals=proposals, IK_candidates=ik_count,
                    IK_starts_attempted=starts_count, CPU_wall_s=time.perf_counter() - began,
                    reference_rotation=initial_pose.rotation.tolist(),
                    raw_joint_lower_rad=lo.tolist(), raw_joint_upper_rad=hi.tolist(),
                    raw_speed_limits_rad_s=model.velocityLimit.tolist(),
                    raw_effort_limits_Nm=model.effortLimit.tolist(),
                    IK_witness_is_MPC_seed=False,
                    no_collision_or_dynamic_feasibility_certificate=True)
    write(output / ("targets.json" if complete else "SAMPLING_EXHAUSTED.json"), manifest)
    if not complete:
        raise RuntimeError(f"Only {len(targets)} targets; {stop_reason}. Preserve partial evidence; do not silently extend the allowance.")
    np.savez_compressed(output / "targets.npz",
                        target_xyz_torso_m=np.asarray([t["xyz_torso_m"] for t in targets]),
                        IK_witness_q_rad=np.asarray([t["IK_witness_q_rad"] for t in targets]),
                        initial_state=np.r_[q0, np.zeros(7)],
                        proposal_ids=np.asarray([t["proposal_id"] for t in targets]),
                        reference_rotation=initial_pose.rotation)
    write(output / "CPU_RECEIPT.json", dict(created_utc=datetime.now(timezone.utc).isoformat(),
          targets_sha256=hashlib.sha256((output / "targets.json").read_bytes()).hexdigest(),
          arrays_sha256=hashlib.sha256((output / "targets.npz").read_bytes()).hexdigest(),
          sampler_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          accepted=len(targets), native_solver_calls=0, controller_calls=0, GPU_calls=0))
    print(json.dumps(dict(output=str(output), accepted=len(targets), IK_candidates=ik_count,
                          cartesian_proposals=proposals, CPU_wall_s=time.perf_counter() - began)))


if __name__ == "__main__":
    main()
