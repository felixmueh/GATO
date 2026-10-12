#!/usr/bin/env python3
"""One center-only CPU IK check from the accepted v5 center witness."""
import json
from pathlib import Path
import sys
import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares
from screen_front_axis_free import ROOT, build_tiago_collision_model, NamedJointState, state_to_qv, compute_pair_distances, DEFAULT_COLLISION_URDF_PATH, DEFAULT_COLLISION_BLACKLIST_PATH


def main():
    base=Path(sys.argv[1]);output=Path(sys.argv[2]);output.mkdir(parents=True,exist_ok=False)
    (output/'screen_source.py').write_bytes(Path(__file__).read_bytes())
    screen=json.loads((base/'screen.json').read_text());target=np.asarray(screen['centers_m'][0]);axis=np.array((1.,0.,0.))
    old=json.loads((ROOT/'experiments/deployment/demo/proposal.json').read_text());seed=np.asarray(next(g for g in old['goals'] if g['id']=='center')['ik_witness_q_rad'])
    previous=np.asarray(screen['center_outcomes'][0]['goals'][-1]['q_rad'])
    model=pin.buildModelFromUrdf(str(ROOT/'gato/dynamics/tiago_right/tiago_right_arm.urdf'));data=model.createData();tool=model.getFrameId('arm_right_tool_link')
    collision=build_tiago_collision_model(urdf_path=DEFAULT_COLLISION_URDF_PATH,blacklist_path=DEFAULT_COLLISION_BLACKLIST_PATH,monitor_only_collision_pairs=True)
    def clearance(q):
        body=dict(screen['body_positions'],**{f'arm_right_{i+1}_joint':float(v) for i,v in enumerate(q)})
        cq,_=state_to_qv(collision.model,NamedJointState(position=body,velocity={k:0. for k in body}))
        return min(compute_pair_distances(collision,cq),key=lambda p:p.distance_m)
    def residual(q):
        pin.framesForwardKinematics(model,data,q);pose=data.oMf[tool]
        return np.r_[pose.translation-target,.15*(pose.rotation[:,2]-axis)]
    result=least_squares(residual,np.clip(seed,model.lowerPositionLimit+.001,model.upperPositionLimit-.001),bounds=(model.lowerPositionLimit+.001,model.upperPositionLimit-.001),max_nfev=220,ftol=1e-8,xtol=1e-8,gtol=1e-8)
    q=result.x;pin.framesForwardKinematics(model,data,q);pose=data.oMf[tool].copy();pe=float(np.linalg.norm(pose.translation-target));oe=float(np.arccos(np.clip(pose.rotation[:,2]@axis,-1,1)))
    cross=np.cross(pose.rotation[:,2],axis);norm=np.linalg.norm(cross);R=pin.exp3(cross/norm*oe)@pose.rotation if norm>1e-12 else pose.rotation
    pair=clearance(q);delta=q-previous;margin=np.minimum(q-model.lowerPositionLimit,model.upperPositionLimit-q)
    row=dict(center_m=target.tolist(),goal='center',target_m=target.tolist(),tool_axis=axis.tolist(),seed='accepted-v5-center',q_rad=q.tolist(),position_error_m=pe,orientation_error_rad=oe,endpoint_min_m=pair.distance_m,closest_pair=pair.to_json(),nfev=result.nfev,pose_m_rad=target.tolist()+pin.rpy.matrixToRpy(R).tolist(),selected_rotation=R.tolist(),joint_travel_rad=float(np.linalg.norm(delta)),max_per_joint_step_rad=float(np.max(np.abs(delta))),wrist_travel_rad=float(np.linalg.norm(delta[4:])),joint_margin_rad=float(np.min(margin)),wrist_margin_rad=float(np.min(margin[4:])),accepted_endpoint=pe<.001 and oe<.02 and pair.distance_m>=.04)
    if row['accepted_endpoint']:
        distances=[clearance(previous+a*delta).distance_m for a in np.linspace(0,1,21)];row.update(joint_interpolation_clearance_m=distances,joint_interpolation_min_m=min(distances))
    row['accepted_sequence_witness']=row['accepted_endpoint'] and row.get('joint_interpolation_min_m',0)>=.04
    receipt=dict(source_screen=str(base),seed_source='accepted v5 center IK witness in historical proposal',attempt=row,scope='One CPU center IK call; prior corners and all previous failures unchanged')
    (output/'center_check.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(row),flush=True)


if __name__=='__main__':main()
