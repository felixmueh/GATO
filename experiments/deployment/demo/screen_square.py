#!/usr/bin/env python3
"""Bounded CPU IK/endpoint collision screen; never executes a controller."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tiago_src'))
from gato_tiago.safety_monitor import (build_tiago_collision_model,NamedJointState,state_to_qv,
    DEFAULT_COLLISION_URDF_PATH,DEFAULT_COLLISION_BLACKLIST_PATH)
from gato_tiago.tiago_tools import compute_pair_distances

SIDE_M=.30
CENTERS_M=((.45,-.35,-.10),(.45,-.45,-.10),(.40,-.40,0.),(.50,-.40,0.))
BODY_POSITIONS=dict(torso_lift_joint=.10,head_1_joint=0.,head_2_joint=0.,
 **{f'arm_left_{i+1}_joint':v for i,v in enumerate((.36,-1.83,.47,-2.35,0.,-1.2,0.))})
START_Q=np.array((-.39,-1.73,-.38,-2.35,0.,-1.21,.04))


def rotation(axis, twist=0.):
    z=np.asarray(axis,dtype=float);z/=np.linalg.norm(z)
    reference=np.array((1.,0.,0.)) if abs(z[0])<.9 else np.array((0.,1.,0.))
    x=reference-z*(reference@z);x/=np.linalg.norm(x);y=np.cross(z,x)
    R=np.column_stack((x,y,z))
    return R@pin.AngleAxis(twist,np.array((0.,0.,1.))).matrix()


def main():
    output=Path(sys.argv[1])
    output.mkdir(parents=True,exist_ok=False)
    model=pin.buildModelFromUrdf(str(ROOT/'gato/dynamics/tiago_right/tiago_right_arm.urdf'))
    data=model.createData();tool=model.getFrameId('arm_right_tool_link')
    collision=build_tiago_collision_model(urdf_path=DEFAULT_COLLISION_URDF_PATH,
        blacklist_path=DEFAULT_COLLISION_BLACKLIST_PATH,monitor_only_collision_pairs=True)
    records=[];accepted=None
    for center in CENTERS_M:
        center=np.asarray(center);previous=START_Q.copy();selected=[]
        for label, offset in zip(('bottom-right','top-right','top-left','bottom-left','center'),
                ((0.,-.15,-.15),(0.,-.15,.15),(0.,.15,.15),(0.,.15,-.15),(0.,0.,0.))):
            target=center+offset;axis=-np.asarray(offset) if label!='center' else np.array((-1.,0.,0.))
            if label!='center':axis/=np.linalg.norm(axis)
            chosen=None
            for twist in (0.,np.pi/2.,-np.pi/2.):
                goalR=rotation(axis,twist)
                def residual(q):
                    pin.framesForwardKinematics(model,data,q);pose=data.oMf[tool]
                    return np.r_[pose.translation-target,.15*pin.log3(goalR.T@pose.rotation)]
                for seed in (previous,START_Q):
                    result=least_squares(residual,np.clip(seed,model.lowerPositionLimit+.001,model.upperPositionLimit-.001),
                        bounds=(model.lowerPositionLimit+.001,model.upperPositionLimit-.001),max_nfev=180,
                        ftol=1e-8,xtol=1e-8,gtol=1e-8)
                    q=result.x;pin.framesForwardKinematics(model,data,q);pose=data.oMf[tool]
                    position=float(np.linalg.norm(pose.translation-target));angle=float(np.linalg.norm(pin.log3(goalR.T@pose.rotation)))
                    row=dict(center_m=center.tolist(),goal=label,target_m=target.tolist(),tool_axis=axis.tolist(),twist_rad=twist,
                        q=q.tolist(),position_error_m=position,full_orientation_error_rad=angle,nfev=result.nfev,
                        endpoint_collision_min_m=None,collision_pass=None)
                    if position<.001 and angle<.02:
                        body=dict(BODY_POSITIONS,**{f'arm_right_{i+1}_joint':float(v) for i,v in enumerate(q)})
                        state=NamedJointState(position=body,velocity={k:0. for k in body})
                        cq,_=state_to_qv(collision.model,state)
                        pairs=compute_pair_distances(collision,cq)
                        closest=min(pairs,key=lambda p:p.distance_m)
                        row.update(endpoint_collision_min_m=closest.distance_m,collision_pass=closest.distance_m>=.04,
                            closest_pair=closest.to_json())
                    row['accepted']=position<.001 and angle<.02 and row['collision_pass'] is True
                    records.append(row)
                    if row['accepted']:chosen=row;break
                if chosen:break
            if not chosen:break
            previous=np.asarray(chosen['q']);selected.append(chosen)
        if len(selected)==5:
            accepted=selected;break
    protocol=dict(side_m=SIDE_M,plane='torso-frame yz at fixed x; viewed from robot right=-y and up=+z',
        center_tool_axis=[-1.,0.,0.],corner_tool_axes='in-plane diagonals toward square center',
        body_positions=BODY_POSITIONS,body_assumption='Torso0.10m, left-arm comfortable, headzero; wheels/grippers fixedneutral exactly as runtime reduced monitor. Not current robot observation.',
        collision_urdf=str(DEFAULT_COLLISION_URDF_PATH),blacklist=str(DEFAULT_COLLISION_BLACKLIST_PATH),
        collision_threshold_m=.04,collision_scope='same runtime monitored full-body pairs at IK endpoints only; no swept-path certificate',
        attempts=records,selected=accepted,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'screen.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps(dict(centers_tried=len({tuple(r['center_m']) for r in records}),ik_attempts=len(records),
        accepted=bool(accepted),selected_center=accepted[0]['center_m'] if accepted else None)))


if __name__=='__main__':main()
