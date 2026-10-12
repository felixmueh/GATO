#!/usr/bin/env python3
"""Four fixed axial choices, using a short sequential Cartesian/SO(3) IK proxy."""
import json
from pathlib import Path
import sys
import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares
from screen_front_square import (ROOT, BODY_POSITIONS, build_tiago_collision_model,
    NamedJointState, state_to_qv, compute_pair_distances,
    DEFAULT_COLLISION_URDF_PATH, DEFAULT_COLLISION_BLACKLIST_PATH)


def main():
    output=Path(sys.argv[1]);output.mkdir(parents=True,exist_ok=False)
    (output/'screen_source.py').write_bytes(Path(__file__).read_bytes())
    proposal=json.loads((ROOT/'experiments/deployment/demo/proposal_right10.json').read_text())
    prior=Path('/workspace/GATO/_expr_results/square-right10cm-v8-internal-20261012/analysis')
    rows=json.loads((prior/'per_goal.json').read_text());q0=np.asarray(rows[2]['achieved_q_rad'])
    model=pin.buildModelFromUrdf(str(ROOT/'gato/dynamics/tiago_right/tiago_right_arm.urdf'))
    data=model.createData();tool=model.getFrameId('arm_right_tool_link')
    pin.framesForwardKinematics(model,data,q0);pose0=data.oMf[tool].copy()
    goal=proposal['goals'][3];target=np.asarray(goal['pose_m_rad'][:3]);axis=np.asarray(goal['tool_axis']);baseline=pin.rpy.rpyToMatrix(*goal['pose_m_rad'][3:])
    # Minimal axis transport preserves the starting transverse tool frame.
    a=pose0.rotation[:,2];cross=np.cross(a,axis);angle=np.arccos(np.clip(a@axis,-1,1))
    transported=pin.exp3(cross/np.linalg.norm(cross)*angle)@pose0.rotation
    phi=float(pin.log3(transported@baseline.T)@axis)
    choices=[('baseline',0.),('axis-transport-minus30',phi-np.pi/6),('axis-transport',phi),('axis-transport-plus30',phi+np.pi/6)]
    collision=build_tiago_collision_model(urdf_path=DEFAULT_COLLISION_URDF_PATH,blacklist_path=DEFAULT_COLLISION_BLACKLIST_PATH,monitor_only_collision_pairs=True)
    def clearance(q):
        body=dict(BODY_POSITIONS,**{f'arm_right_{i+1}_joint':float(v) for i,v in enumerate(q)})
        cq,_=state_to_qv(collision.model,NamedJointState(position=body,velocity={k:0. for k in body}))
        return min(compute_pair_distances(collision,cq),key=lambda p:p.distance_m)
    candidates=[]
    for name,twist in choices:
        R=pin.exp3(axis*twist)@baseline;relative=pin.log3(pose0.rotation.T@R);q=q0.copy();samples=[]
        for alpha in np.linspace(0,1,11):
            p=(1-alpha)*pose0.translation+alpha*target;ri=pose0.rotation@pin.exp3(alpha*relative)
            def residual(x):
                pin.framesForwardKinematics(model,data,x);fk=data.oMf[tool]
                return np.r_[fk.translation-p,.15*pin.log3(ri.T@fk.rotation)]
            fit=least_squares(residual,np.clip(q,model.lowerPositionLimit+.001,model.upperPositionLimit-.001),bounds=(model.lowerPositionLimit+.001,model.upperPositionLimit-.001),max_nfev=120,ftol=1e-8,xtol=1e-8,gtol=1e-8)
            q=fit.x;pin.framesForwardKinematics(model,data,q);fk=data.oMf[tool].copy();pair=clearance(q);margin=np.minimum(q-model.lowerPositionLimit,model.upperPositionLimit-q)
            sample=dict(alpha=float(alpha),q_rad=q.tolist(),position_error_m=float(np.linalg.norm(fk.translation-p)),full_rotation_error_rad=float(np.linalg.norm(pin.log3(ri.T@fk.rotation))),tool_axis=fk.rotation[:,2].tolist(),clearance_m=pair.distance_m,wrist_margin_rad=float(min(margin[4:])),closest_pair=pair.to_json(),nfev=fit.nfev)
            samples.append(sample)
        minimum=min(s['clearance_m'] for s in samples);valid_pose=all(s['position_error_m']<.001 and s['full_rotation_error_rad']<.02 for s in samples)
        path=np.asarray([s['q_rad'] for s in samples]);steps=np.diff(path,axis=0);travel=float(np.linalg.norm(steps,axis=1).sum());wristtravel=float(np.linalg.norm(steps[:,4:],axis=1).sum());margin=min(s['wrist_margin_rad'] for s in samples)
        candidate=dict(name=name,axial_twist_from_v8_rad=float(twist),relative_rotation_deg=float(np.rad2deg(np.linalg.norm(relative))),pose_m_rad=target.tolist()+pin.rpy.matrixToRpy(R).tolist(),tool_axis=axis.tolist(),q_rad=q.tolist(),minimum_route_clearance_m=minimum,endpoint_clearance_m=samples[-1]['clearance_m'],route_pose_pass=valid_pose,route_clearance_pass=minimum>=.04,accepted=valid_pose and minimum>=.04,route_joint_travel_rad=travel,route_wrist_travel_rad=wristtravel,wrist_margin_rad=margin,score=travel+.35*wristtravel+.05/max(margin,.02)+100.*max(.05-minimum,0.),samples=samples)
        candidates.append(candidate);print(json.dumps({k:v for k,v in candidate.items() if k!='samples'}),flush=True)
    valid=[c for c in candidates if c['accepted']];selected=min(valid,key=lambda c:c['score']) if valid else None
    receipt=dict(proposal=str(ROOT/'experiments/deployment/demo/proposal_right10.json'),center_m=[.5,-.35,-.15],side_m=.1,first_three_fixed=True,start_actual_top_left_q_rad=q0.tolist(),body_positions=BODY_POSITIONS,method='Four fixed axial target choices; 11 sequential linear-position/SO(3)-geodesic samples using previous IK as seed; 1 mm/.02 rad pose check, unchanged 40 mm monitor gate.',scope='Heuristic endpoint/path screen only, no MPC, dynamic or continuous swept certificate. Actual prior top-left posture is a seed, not guaranteed in a new episode.',candidates=candidates,selected=selected)
    (output/'screen.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
