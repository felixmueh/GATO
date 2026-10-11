#!/usr/bin/env python3
"""CPU witness dynamics, drawing and sparse upward-reference endpoint screening."""
import json
from pathlib import Path
import sys
import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares

from screen_square import ROOT, BODY_POSITIONS, START_Q, rotation
sys.path.insert(0,str(ROOT/'tiago_src'))
from gato_tiago.safety_monitor import build_tiago_collision_model,NamedJointState,state_to_qv,DEFAULT_COLLISION_URDF_PATH,DEFAULT_COLLISION_BLACKLIST_PATH
from gato_tiago.tiago_tools import compute_pair_distances


def main():
    directory=Path(sys.argv[1]);proposal=json.loads((directory/'screen.json').read_text())
    selected=proposal['selected']
    variants_path=directory/'center_variants.json'
    if variants_path.exists():
        variants=json.loads(variants_path.read_text())
        viable=[r for r in variants['variants'] if r['accepted']]
        if viable:
            chosen=max(viable,key=lambda r:r['joint_margin_rad'])
            proposal['original_screen_selected']=list(selected)
            selected=list(selected[:-1])+[chosen]
            proposal['selected']=selected
            proposal['center_witness_selection']='Among3bounded accepted alternatives choose largest minimum joint-limit margin; original witness retained'
            proposal['center_variants']=variants

    model=pin.buildModelFromUrdf(str(ROOT/'gato/dynamics/tiago_right/tiago_right_arm.urdf'));data=model.createData()
    previous=START_Q.copy();dynamics=[]
    for row in selected:
        delta=np.asarray(row['q'])-previous;duration=15.
        samples=[]
        for fraction in np.linspace(0.,1.,31):
            q=previous+delta*(10*fraction**3-15*fraction**4+6*fraction**5)
            v=delta*(30*fraction**2-60*fraction**3+30*fraction**4)/duration
            acc=delta*(60*fraction-180*fraction**2+120*fraction**3)/duration**2
            torque=pin.rnea(model,data,q,v,acc)
            samples.append((q.copy(),v.copy(),torque.copy()))
        q,v,tau=map(np.asarray,zip(*samples))
        dynamics.append(dict(goal=row['goal'],duration_s=duration,necessary_velocity_time_s=float(np.max(abs(delta)/model.velocityLimit)),
            sampled_velocity_margin_rad_s=float(np.min(model.velocityLimit-abs(v))),
            sampled_effort_margin_Nm=float(np.min(model.effortLimit-abs(tau))),
            joint_position_margin_rad=float(min(np.min(q-model.lowerPositionLimit),np.min(model.upperPositionLimit-q))),
            sampled_dynamic_witness_pass=bool(np.all(abs(v)<=model.velocityLimit) and np.all(abs(tau)<=model.effortLimit))))
        previous=np.asarray(row['q'])
    proposal['quintic_15s_witness']=dynamics
    proposal['dynamic_scope']='31 inverse-dynamics samples per sequential joint interpolation; no MPC behavior, collision sweep or between-sample certificate'
    collision=build_tiago_collision_model(urdf_path=DEFAULT_COLLISION_URDF_PATH,blacklist_path=DEFAULT_COLLISION_BLACKLIST_PATH)
    inputs=json.loads((ROOT/'experiments/deployment/inputs/t01.json').read_text())
    task=next(t for t in inputs['tasks'] if t['id']=='figure8-24s')
    reference=np.asarray(task['reference_packed']).reshape(-1,6)
    previous=START_Q.copy();points=[];tool=model.getFrameId('arm_right_tool_link')
    for phase in range(8):
        xyz=reference[int(phase*task['period_s']/task['reference_dt_s']/8),:3]
        def residual(q):
            pin.framesForwardKinematics(model,data,q);pose=data.oMf[tool]
            return np.r_[pose.translation-xyz,.15*pin.log3(pose.rotation)]
        best=None
        for seed in (previous,START_Q):
            result=least_squares(residual,np.clip(seed,model.lowerPositionLimit+.001,model.upperPositionLimit-.001),
                bounds=(model.lowerPositionLimit+.001,model.upperPositionLimit-.001),max_nfev=180)
            q=result.x;pin.framesForwardKinematics(model,data,q);pose=data.oMf[tool]
            error=float(np.linalg.norm(pose.translation-xyz));angle=float(np.linalg.norm(pin.log3(pose.rotation)))
            row=dict(phase=phase/8,xyz_m=xyz.tolist(),q=q.tolist(),position_error_m=error,orientation_error_rad=angle,
                endpoint_collision_min_m=None,pass_=False)
            if error<.001 and angle<.02:
                body=dict(BODY_POSITIONS,**{f'arm_right_{i+1}_joint':float(v) for i,v in enumerate(q)})
                cq,_=state_to_qv(collision.model,NamedJointState(body,{k:0. for k in body}))
                minimum=min(compute_pair_distances(collision,cq),key=lambda p:p.distance_m)
                row.update(endpoint_collision_min_m=minimum.distance_m,closest_pair=minimum.to_json(),pass_=minimum.distance_m>=.04)
            if best is None or row['pass_'] or (error+.15*angle<best['position_error_m']+.15*best['orientation_error_rad']):best=row
            if row['pass_']:break
        points.append(best)
        if best['pass_']:previous=np.asarray(best['q'])
    proposal['upward_figure8_sparse_screen']=dict(tool_axis=[0.,0.,1.],rotation_rpy_rad=[0.,0.,0.],periods_s=[24.,18.,12.],
        samples=points,scope='8 phase endpoints at shared public geometry; fullposeIK and runtime endpointclearance only, not tracking dynamic/swept qualification')
    (directory/'proposal.json').write_text(json.dumps(proposal,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(10,7));ax=fig.add_subplot(111,projection='3d')
    xyz=np.array([r['target_m'] for r in selected]);ax.plot(xyz[:4,0],xyz[:4,1],xyz[:4,2],'o-',label='BR→TR→TL→BL→center')
    ax.plot(xyz[[3,0],0],xyz[[3,0],1],xyz[[3,0],2],':',color='gray')
    ax.plot(xyz[[3,4],0],xyz[[3,4],1],xyz[[3,4],2],'o-',color='blue')
    for i,row in enumerate(selected):
        axis=np.asarray(row['tool_axis']);ax.quiver(*xyz[i],*axis,length=.08,color='red')
        ax.text(*xyz[i],str(i+1)+' '+row['goal'],fontsize=8)
    ax.plot(reference[:800,0],reference[:800,1],reference[:800,2],color='green',label='Public figure8 geometry; tool+Z up')
    ax.set(xlabel='Torso x (m)',ylabel='Torso y (m)',zlabel='Torso z (m)',title='30cm visual demonstration proposal; arrows show tool+Z\nIK endpoint screen only; full motions await internal/Gazebo review')
    ax.legend();fig.tight_layout();fig.savefig(directory/'proposal.png',dpi=150)
    print(json.dumps(dict(square_dynamic_witness_all_pass=all(r['sampled_dynamic_witness_pass'] for r in dynamics),
        upward_figure8_endpoint_pass=sum(r['pass_'] for r in points),upward_figure8_endpoints=len(points))))


if __name__=='__main__':main()
