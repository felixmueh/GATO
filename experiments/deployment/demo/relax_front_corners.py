#!/usr/bin/env python3
"""Bounded simultaneous four-corner posture selection; no MPC solver call."""
import json
import argparse
from functools import lru_cache
from pathlib import Path
import sys
import time
import numpy as np
import pinocchio as pin
from scipy.optimize import minimize
from screen_front_axis_free import ROOT, build_tiago_collision_model, NamedJointState, state_to_qv, compute_pair_distances, DEFAULT_COLLISION_URDF_PATH, DEFAULT_COLLISION_BLACKLIST_PATH


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('source',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--prefer-clearance',type=float,default=0.)
    parser.add_argument('--coarse-paths',action='store_true')
    parser.add_argument('--optimization-clearance',type=float,default=.04)
    args=parser.parse_args();source=args.source;output=args.output;output.mkdir(parents=True,exist_ok=False)
    (output/'optimization_source.py').write_bytes(Path(__file__).read_bytes())
    s=json.loads((source/'screen.json').read_text());base=s['center_outcomes'][0]['goals'][:4];assert len(base)==4
    model=pin.buildModelFromUrdf(str(ROOT/'gato/dynamics/tiago_right/tiago_right_arm.urdf'));data=model.createData();tool=model.getFrameId('arm_right_tool_link')
    col=build_tiago_collision_model(urdf_path=DEFAULT_COLLISION_URDF_PATH,blacklist_path=DEFAULT_COLLISION_BLACKLIST_PATH,monitor_only_collision_pairs=True)
    start=np.asarray(s['initial_q_rad']);initial=np.asarray([r['q_rad'] for r in base]).reshape(-1)
    targets=np.asarray([r['target_m'] for r in base]);axes=np.asarray([r['tool_axis'] for r in base])
    # Unit axes fixed in torso YZ. +X and its perpendicular within YZ span
    # the two transverse directions; axis sign is checked after optimization.
    transverse=np.asarray([np.stack((np.array((1.,0.,0.)),np.cross(axis,(1.,0.,0.)))) for axis in axes])
    lo=model.lowerPositionLimit+.001;hi=model.upperPositionLimit-.001
    weight=np.array((1.,1.,1.,1.,1.35,1.35,1.35))
    cd,gd=col.make_data()
    def q_full(q):
        body=dict(s['body_positions'],**{f'arm_right_{j+1}_joint':float(v) for j,v in enumerate(q)})
        return state_to_qv(col.model,NamedJointState(position=body,velocity={k:0. for k in body}))[0]
    @lru_cache(maxsize=96)
    def sampled_clearances(key):
        q=np.frombuffer(key,dtype=np.float64).reshape(4,7);out=[];previous=start
        for after in q:
            for alpha in ((.25,.5,.75,1.) if args.coarse_paths else (1.,)):
                state=previous+alpha*(after-previous)
                pin.computeDistances(col.model,cd,col.geometry_model,gd,q_full(state))
                out.append(min(float(r.min_distance) for r in gd.distanceResults))
            previous=after
        return np.asarray(out)
    def distances(flat):return sampled_clearances(np.asarray(flat,dtype=np.float64).tobytes())
    def objective(flat):
        q=flat.reshape(4,7);delta=np.diff(np.vstack((start,q)),axis=0)
        travel=np.sum(delta**2*weight)
        near=.01*np.sum(1/(q-model.lowerPositionLimit+.15)**2+1/(model.upperPositionLimit-q+.15)**2)
        preference=1e4*np.sum(np.maximum(args.prefer_clearance-distances(flat),0.)**2) if args.prefer_clearance else 0.
        return float(travel+near+preference)
    def pose_constraint(flat):
        out=[]
        for i,q in enumerate(flat.reshape(4,7)):
            pin.framesForwardKinematics(model,data,q);p=data.oMf[tool]
            out.extend(p.translation-targets[i]);out.extend(transverse[i]@p.rotation[:,2])
        return np.asarray(out)
    def clearance(q):
        body=dict(s['body_positions'],**{f'arm_right_{j+1}_joint':float(v) for j,v in enumerate(q)})
        cq,_=state_to_qv(col.model,NamedJointState(position=body,velocity={k:0. for k in body}))
        return min(compute_pair_distances(col,cq),key=lambda p:p.distance_m)
    def collision_constraint(flat):return distances(flat)-args.optimization_clearance
    iterations=[];feasible_snapshots=[];began=time.monotonic()
    def callback(flat):
        row=dict(iteration=len(iterations)+1,objective=objective(flat),max_pose_equality_abs=float(np.max(np.abs(pose_constraint(flat)))),minimum_clearance_m=float(np.min(distances(flat))))
        if row['max_pose_equality_abs']<.0004 and row['minimum_clearance_m']>=.04:feasible_snapshots.append((row['objective'],flat.copy()))
        iterations.append(row);print(json.dumps(row),flush=True)
        (output/'iterations.json').write_text(json.dumps(iterations,indent=2)+'\n')
    result=minimize(objective,initial,method='SLSQP',bounds=list(zip(np.tile(lo,4),np.tile(hi,4))),
        constraints=[dict(type='eq',fun=pose_constraint),dict(type='ineq',fun=collision_constraint)],
        callback=callback,options=dict(maxiter=30,ftol=1e-6,disp=False))
    def validate(flat):
        selected=[];previous=start.copy()
        for original,q in zip(base,flat.reshape(4,7)):
            pin.framesForwardKinematics(model,data,q);p=data.oMf[tool].copy();axis=np.asarray(original['tool_axis'])
            pe=float(np.linalg.norm(p.translation-np.asarray(original['target_m'])));oe=float(np.arccos(np.clip(p.rotation[:,2]@axis,-1,1)))
            v=np.cross(p.rotation[:,2],axis);norm=np.linalg.norm(v);R=pin.exp3(v/norm*oe)@p.rotation if norm>1e-12 else p.rotation
            pair=clearance(q);margin=np.minimum(q-model.lowerPositionLimit,model.upperPositionLimit-q);delta=q-previous
            path=[clearance(previous+a*delta).distance_m for a in np.linspace(0,1,21)]
            row=dict(original,seed='joint-four-corner-SLSQP',q_rad=q.tolist(),selected_rotation=R.tolist(),
                pose_m_rad=original['target_m']+pin.rpy.matrixToRpy(R).tolist(),position_error_m=pe,orientation_error_rad=oe,
                endpoint_min_m=pair.distance_m,closest_pair=pair.to_json(),joint_travel_rad=float(np.linalg.norm(delta)),
                max_per_joint_step_rad=float(np.max(np.abs(delta))),wrist_travel_rad=float(np.linalg.norm(delta[4:])),
                joint_margin_rad=float(np.min(margin)),wrist_margin_rad=float(np.min(margin[4:])),
                joint_interpolation_clearance_m=path,joint_interpolation_min_m=min(path),
                accepted_endpoint=pe<.001 and oe<.02 and pair.distance_m>=.04)
            selected.append(row);previous=q.copy()
        return selected
    candidates=[('optimizer-final',result.x)]+[('feasible-iteration',flat) for score,flat in sorted(feasible_snapshots,key=lambda item:item[0])[:2]]+[('original-valid-screen',initial)]
    checks=[];valid=[]
    for label,flat in candidates:
        rows=validate(flat);passed=all(r['accepted_endpoint'] and r['joint_interpolation_min_m']>=.04 for r in rows)
        record=dict(kind=label,objective=objective(flat),accepted=passed,poses=rows);checks.append(record)
        if passed:valid.append(record)
    chosen=min(valid,key=lambda r:r['objective']) if valid else checks[0];selected=chosen['poses'];accepted=chosen['accepted']
    summary=dict(source_screen=str(source),method='One joint SLSQP run, maximum30iterations, 28 joint variables; continuous axial roll unconstrained in CPU selection only',
        objective='sum_j [1 for arm1-4,1.35 for wrist5-7]*(q_i-q_previous)^2 + .01*sum((q-lower+.15)^-2+(upper-q+.15)^-2), including initial-to-first transition',
        objective_before=objective(initial),objective_after=chosen['objective'],optimizer_final_objective=objective(result.x),optimizer_success=bool(result.success),optimizer_message=str(result.message),
        iterations=len(iterations),elapsed_wall_s=time.monotonic()-began,accepted_four_corner_witness=accepted,
        limitation='Local bounded posture optimization, not globally most relaxed; strict center held separately. No MPC/Gazebo/physical run.',
        clearance_preference_m=args.prefer_clearance,optimization_clearance_m=args.optimization_clearance,coarse_paths=args.coarse_paths,
        selected_candidate_kind=chosen['kind'],candidate_checks=checks,before=base,after=selected)
    (output/'optimization.json').write_text(json.dumps(summary,indent=2)+'\n')
    corners_only=s.get('corner_only',False)
    final=dict(s,attempts=selected,selected=selected if corners_only and accepted else None,center_outcomes=[dict(center_m=s['centers_m'][0],goals=selected if accepted else base,complete=corners_only and accepted)],
        four_corner_pass=True if accepted else all(r['accepted_endpoint'] for r in base),center_status='excluded by explicit user scope' if corners_only else 'held: fifthcenter failed IK/4cmclearance',optimization=summary,
        orientation_selection='Full rotations frozen from bounded jointly relaxed position+tool-axis CPU IK; all inward directions unchanged')
    (output/'screen.json').write_text(json.dumps(final,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k not in ('before','after','candidate_checks')}),flush=True)


if __name__=='__main__':main()
