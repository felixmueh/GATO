#!/usr/bin/env python3
"""Bounded center-only posture optimization with collision directly constrained."""
import json
import argparse
from pathlib import Path
import sys
import time
import numpy as np
import pinocchio as pin
from scipy.optimize import minimize
from screen_front_axis_free import ROOT, build_tiago_collision_model, NamedJointState, state_to_qv, compute_pair_distances, DEFAULT_COLLISION_URDF_PATH, DEFAULT_COLLISION_BLACKLIST_PATH


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('base',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--refine-exact',type=Path,help='One approved refinement from the preserved exact center candidate')
    parser.add_argument('--screen',type=Path,help='Explicit fixed-placement corner/center CPU screen')
    parser.add_argument('--only-best',action='store_true',help='One bounded refinement from its closest center candidate')
    args=parser.parse_args();base=args.base;output=args.output;output.mkdir(parents=True,exist_ok=False)
    (output/'optimization_source.py').write_bytes(Path(__file__).read_bytes())
    frozen_path=args.screen or base/'screen-007-joint-relaxation/screen.json'
    original_path=args.screen or base/'screen-005-right-shoulder/screen.json'
    frozen=json.loads(frozen_path.read_text());original=json.loads(original_path.read_text())
    previous=np.asarray(frozen['center_outcomes'][0]['goals'][-1]['q_rad'])
    target=np.asarray(frozen['centers_m'][0]);axis=np.array((1.,0.,0.))
    old=json.loads((ROOT/'experiments/deployment/demo/proposal.json').read_text())
    old_center=np.asarray(next(g for g in old['goals'] if g['id']=='center')['ik_witness_q_rad'])
    exact=min([r for r in original['attempts'] if r['goal']=='center'],key=lambda r:r['position_error_m']+.15*r['orientation_error_rad'])
    seeds=[('exact-31mm-clearance',np.asarray(exact['q_rad'])),('fixed-bottom-left',previous),('accepted-v5-center',old_center)]
    if args.only_best:seeds=[('best-new-center',np.asarray(exact['q_rad']))]
    if args.refine_exact:
        prior=json.loads(args.refine_exact.read_text());seeds=[('buffered-exact-center',np.asarray(prior['attempts'][0]['q_rad']))]
    model=pin.buildModelFromUrdf(str(ROOT/'gato/dynamics/tiago_right/tiago_right_arm.urdf'));data=model.createData();tool=model.getFrameId('arm_right_tool_link')
    col=build_tiago_collision_model(urdf_path=DEFAULT_COLLISION_URDF_PATH,blacklist_path=DEFAULT_COLLISION_BLACKLIST_PATH,monitor_only_collision_pairs=True)
    weight=np.asarray((1,1,1,1,1.35,1.35,1.35));lo=model.lowerPositionLimit+.001;hi=model.upperPositionLimit-.001
    def objective(q):
        return float(np.sum(weight*(q-previous)**2)+.01*np.sum(1/(q-model.lowerPositionLimit+.15)**2+1/(model.upperPositionLimit-q+.15)**2))
    def pose_constraint(q):
        pin.framesForwardKinematics(model,data,q);p=data.oMf[tool]
        return np.r_[p.translation-target,p.rotation[1:,2]]
    def axis_sign(q):
        pin.framesForwardKinematics(model,data,q);return data.oMf[tool].rotation[0,2]
    def clearance(q):
        body=dict(frozen['body_positions'],**{f'arm_right_{j+1}_joint':float(v) for j,v in enumerate(q)})
        cq,_=state_to_qv(col.model,NamedJointState(position=body,velocity={k:0. for k in body}))
        return min(compute_pair_distances(col,cq),key=lambda p:p.distance_m)
    attempts=[];selected=None
    for label,seed in seeds:
        history=[];began=time.monotonic()
        def callback(q):
            row=dict(iteration=len(history)+1,objective=objective(q),pose_equality_max=float(np.max(np.abs(pose_constraint(q)))),clearance_m=clearance(q).distance_m)
            history.append(row)
            (output/(label+'-iterations.json')).write_text(json.dumps(history,indent=2)+'\n')
        result=minimize(objective,np.clip(seed,lo,hi),method='SLSQP',bounds=list(zip(lo,hi)),
            constraints=[dict(type='eq',fun=pose_constraint),dict(type='ineq',fun=lambda q:clearance(q).distance_m-.0401),dict(type='ineq',fun=axis_sign)],
            callback=callback,options=dict(maxiter=30,ftol=1e-7,disp=False))
        q=result.x;pin.framesForwardKinematics(model,data,q);p=data.oMf[tool].copy()
        pe=float(np.linalg.norm(p.translation-target));oe=float(np.arccos(np.clip(p.rotation[:,2]@axis,-1,1)))
        cross=np.cross(p.rotation[:,2],axis);norm=np.linalg.norm(cross);R=pin.exp3(cross/norm*oe)@p.rotation if norm>1e-12 else p.rotation
        pair=clearance(q);margin=np.minimum(q-model.lowerPositionLimit,model.upperPositionLimit-q);delta=q-previous
        row=dict(center_m=target.tolist(),goal='center',target_m=target.tolist(),tool_axis=axis.tolist(),seed=label,
            q_rad=q.tolist(),position_error_m=pe,orientation_error_rad=oe,endpoint_min_m=pair.distance_m,closest_pair=pair.to_json(),
            pose_m_rad=target.tolist()+pin.rpy.matrixToRpy(R).tolist(),selected_rotation=R.tolist(),joint_travel_rad=float(np.linalg.norm(delta)),
            max_per_joint_step_rad=float(np.max(np.abs(delta))),wrist_travel_rad=float(np.linalg.norm(delta[4:])),joint_margin_rad=float(np.min(margin)),
            wrist_margin_rad=float(np.min(margin[4:])),objective=objective(q),optimizer_success=bool(result.success),message=str(result.message),
            iterations=len(history),elapsed_wall_s=time.monotonic()-began,accepted_endpoint=pe<.001 and oe<.02 and pair.distance_m>=.04)
        if row['accepted_endpoint']:
            distances=[clearance(previous+a*delta).distance_m for a in np.linspace(0,1,21)]
            row.update(joint_interpolation_clearance_m=distances,joint_interpolation_min_m=min(distances))
        row['accepted_sequence_witness']=row['accepted_endpoint'] and row.get('joint_interpolation_min_m',0)>=.04
        attempts.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ('selected_rotation','q_rad','closest_pair','joint_interpolation_clearance_m')}),flush=True)
        (output/'center_optimization.json').write_text(json.dumps(dict(attempts=attempts,selected=selected),indent=2)+'\n')
        if row['accepted_sequence_witness']:
            selected=row;break
    receipt=dict(id='square-front20cm-center-constrained-cpu-v1',source_frozen_corners=str(frozen_path),
        corners_fixed=True,center_axis=[1,0,0],objective='Joint transition squared (wrist weight1.35) plus .01 inverse-square joint-limit proximity penalty with .15rad offset',
        method='Center-only SLSQP, predefined seeds only,30iterations maximum each; continuous axial roll free, physical joint bounds and+Xaxis sign;40.1mm optimizationbuffer, unchanged40mm validation',attempts=attempts,selected=selected)
    (output/'center_optimization.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(accepted=selected is not None,attempts=len(attempts))),flush=True)


if __name__=='__main__':main()
