#!/usr/bin/env python3
"""At most two translated public-geometry candidates; CPU IK endpoints only."""
import json
from pathlib import Path
import sys
import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares

from screen_square import ROOT,BODY_POSITIONS,START_Q
sys.path.insert(0,str(ROOT/'tiago_src'))
from gato_tiago.safety_monitor import build_tiago_collision_model,NamedJointState,state_to_qv,DEFAULT_COLLISION_URDF_PATH,DEFAULT_COLLISION_BLACKLIST_PATH
from gato_tiago.tiago_tools import compute_pair_distances

SHIFTS_M=((0.,0.,0.),(.08,0.,0.))


def main():
    output=Path(sys.argv[1]);output.mkdir(parents=True,exist_ok=False)
    frozen=json.loads((ROOT/'experiments/deployment/inputs/t01.json').read_text());task=next(t for t in frozen['tasks'] if t['id']=='figure8-24s')
    reference=np.asarray(task['reference_packed']).reshape(-1,6)
    m=pin.buildModelFromUrdf(str(ROOT/'gato/dynamics/tiago_right/tiago_right_arm.urdf'));d=m.createData();t=m.getFrameId('arm_right_tool_link')
    pin.framesForwardKinematics(m,d,START_Q)
    start_x=d.oMf[t].rotation[:,0].copy();start_x[2]=0.;start_x/=np.linalg.norm(start_x)
    up=np.array((0.,0.,1.));goalR=np.column_stack((start_x,np.cross(up,start_x),up))
    yaw=float(np.arctan2(start_x[1],start_x[0]))
    c=build_tiago_collision_model(urdf_path=DEFAULT_COLLISION_URDF_PATH,blacklist_path=DEFAULT_COLLISION_BLACKLIST_PATH)
    records=[];selected=None
    for shift in SHIFTS_M:
        previous=START_Q.copy();points=[]
        for phase in range(8):
            xyz=reference[int(phase*task['period_s']/task['reference_dt_s']/8),:3]+shift
            def residual(q):
                pin.framesForwardKinematics(m,d,q);T=d.oMf[t];return np.r_[T.translation-xyz,.15*pin.log3(goalR.T@T.rotation)]
            best=None
            for seed in (previous,START_Q):
                r=least_squares(residual,np.clip(seed,m.lowerPositionLimit+.001,m.upperPositionLimit-.001),
                    bounds=(m.lowerPositionLimit+.001,m.upperPositionLimit-.001),max_nfev=180)
                q=r.x;error=residual(q)
                row=dict(shift_m=shift,phase=phase/8,xyz_m=xyz.tolist(),q=q.tolist(),position_error_m=float(np.linalg.norm(error[:3])),
                    orientation_error_rad=float(np.linalg.norm(error[3:])/.15),endpoint_collision_min_m=None,pass_=False)
                if row['position_error_m']<.001 and row['orientation_error_rad']<.02:
                    body=dict(BODY_POSITIONS,**{f'arm_right_{i+1}_joint':float(v) for i,v in enumerate(q)})
                    cq,_=state_to_qv(c.model,NamedJointState(body,{k:0. for k in body}));pair=min(compute_pair_distances(c,cq),key=lambda p:p.distance_m)
                    row.update(endpoint_collision_min_m=pair.distance_m,closest_pair=pair.to_json(),pass_=pair.distance_m>=.04)
                records.append(row)
                if best is None or row['pass_'] or (row['position_error_m']+.15*row['orientation_error_rad']<best['position_error_m']+.15*best['orientation_error_rad']):best=row
                if row['pass_']:break
            points.append(best)
            if best['pass_']:previous=np.array(best['q'])
        if all(p['pass_'] for p in points):selected=dict(shift_m=shift,samples=points);break
    result=dict(geometry='publicXYfigure8 translated; amplitudes32cmfullx/16cmfully unchanged',tool_axis=[0.,0.,1.],rpy_rad=[0.,0.,yaw],orientation_convention='toolX projected from comfortable start onto horizontal plane; toolZup',
        periods_s=[24.,18.,12.],attempts=records,selected=selected,body_positions=BODY_POSITIONS,
        scope='8phases/candidate fullposeIK/runtime4cm endpointclearance; no swept/dynamic/MPC certificate')
    (output/'screen.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(shift_candidates=len({tuple(r['shift_m']) for r in records}),ik_attempts=len(records),selected_shift=selected['shift_m'] if selected else None)))


if __name__=='__main__':main()
