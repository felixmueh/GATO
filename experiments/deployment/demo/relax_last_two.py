#!/usr/bin/env python3
"""One bounded bottom-left/center posture refinement; first three corners fixed."""
import json
from pathlib import Path
import sys
import time
import numpy as np
import pinocchio as pin
from scipy.optimize import minimize
from screen_front_axis_free import ROOT, build_tiago_collision_model, NamedJointState, state_to_qv, compute_pair_distances, DEFAULT_COLLISION_URDF_PATH, DEFAULT_COLLISION_BLACKLIST_PATH


def main():
    base=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=False)
    (out/'optimization_source.py').write_bytes(Path(__file__).read_bytes())
    frozen=json.loads((base/'screen-007-joint-relaxation/screen.json').read_text());corners=frozen['center_outcomes'][0]['goals']
    center=json.loads((base/'screen-009-center-buffer/center_optimization.json').read_text())['attempts'][0]
    original=[corners[-1],center];previous=np.asarray(corners[-2]['q_rad']);initial=np.asarray([r['q_rad'] for r in original]).reshape(-1)
    model=pin.buildModelFromUrdf(str(ROOT/'gato/dynamics/tiago_right/tiago_right_arm.urdf'));data=model.createData();tool=model.getFrameId('arm_right_tool_link')
    col=build_tiago_collision_model(urdf_path=DEFAULT_COLLISION_URDF_PATH,blacklist_path=DEFAULT_COLLISION_BLACKLIST_PATH,monitor_only_collision_pairs=True)
    cd,gd=col.make_data()
    # Signed distances give local optimization a direction inside a collision;
    # only this CPU query uses them. Final acceptance uses the unchanged monitor.
    for request in gd.distanceRequests:request.enable_signed_distance=True
    axes=np.asarray([r['tool_axis'] for r in original]);targets=np.asarray([r['target_m'] for r in original]);transverse=[]
    for axis in axes:
        reference=np.array((1.,0.,0.)) if abs(axis[0])<.9 else np.array((0.,1.,0.))
        e=reference-axis*(axis@reference);e/=np.linalg.norm(e);transverse.append(np.stack((e,np.cross(axis,e))))
    transverse=np.asarray(transverse);weight=np.asarray((1,1,1,1,1.35,1.35,1.35));lo=model.lowerPositionLimit+.001;hi=model.upperPositionLimit-.001
    def cq(q):
        body=dict(frozen['body_positions'],**{f'arm_right_{j+1}_joint':float(v) for j,v in enumerate(q)})
        return state_to_qv(col.model,NamedJointState(position=body,velocity={k:0. for k in body}))[0]
    def clearance(q):return min(compute_pair_distances(col,cq(q)),key=lambda p:p.distance_m)
    def signed(q):
        pin.computeDistances(col.model,cd,col.geometry_model,gd,cq(q));return min(float(r.min_distance) for r in gd.distanceResults)
    def objective(flat):
        q=flat.reshape(2,7);delta=np.diff(np.vstack((previous,q)),axis=0)
        return float(np.sum(delta**2*weight)+.01*np.sum(1/(q-model.lowerPositionLimit+.15)**2+1/(model.upperPositionLimit-q+.15)**2))
    def pose(flat):
        results=[]
        for i,q in enumerate(flat.reshape(2,7)):
            pin.framesForwardKinematics(model,data,q);p=data.oMf[tool];results.extend(p.translation-targets[i]);results.extend(transverse[i]@p.rotation[:,2])
        return np.asarray(results)
    def constraints(flat):
        q=flat.reshape(2,7);values=[];before=previous
        for i,after in enumerate(q):
            values.extend(signed(before+a*(after-before))-.0401 for a in (.25,.50,.75,1.))
            pin.framesForwardKinematics(model,data,after);values.append(float(data.oMf[tool].rotation[:,2]@axes[i]));before=after
        return np.asarray(values)
    iterations=[];began=time.monotonic()
    def callback(flat):
        row=dict(iteration=len(iterations)+1,objective=objective(flat),pose_equality_max=float(np.max(np.abs(pose(flat)))),constraint_min=float(np.min(constraints(flat))))
        iterations.append(row);print(json.dumps(row),flush=True);(out/'iterations.json').write_text(json.dumps(iterations,indent=2)+'\n')
    result=minimize(objective,initial,method='SLSQP',bounds=list(zip(np.tile(lo,2),np.tile(hi,2))),
        constraints=[dict(type='eq',fun=pose),dict(type='ineq',fun=constraints)],callback=callback,options=dict(maxiter=30,ftol=1e-6,disp=False))
    rows=[];before=previous
    for old,q,axis,target in zip(original,result.x.reshape(2,7),axes,targets):
        pin.framesForwardKinematics(model,data,q);p=data.oMf[tool].copy();pe=float(np.linalg.norm(p.translation-target));oe=float(np.arccos(np.clip(p.rotation[:,2]@axis,-1,1)))
        cross=np.cross(p.rotation[:,2],axis);norm=np.linalg.norm(cross);R=pin.exp3(cross/norm*oe)@p.rotation if norm>1e-12 else p.rotation
        pair=clearance(q);delta=q-before;margin=np.minimum(q-model.lowerPositionLimit,model.upperPositionLimit-q)
        distances=[clearance(before+a*delta).distance_m for a in np.linspace(0,1,21)]
        row=dict(old,seed='joint-last-two-SLSQP',q_rad=q.tolist(),pose_m_rad=target.tolist()+pin.rpy.matrixToRpy(R).tolist(),selected_rotation=R.tolist(),
            position_error_m=pe,orientation_error_rad=oe,endpoint_min_m=pair.distance_m,closest_pair=pair.to_json(),
            joint_travel_rad=float(np.linalg.norm(delta)),max_per_joint_step_rad=float(np.max(np.abs(delta))),wrist_travel_rad=float(np.linalg.norm(delta[4:])),
            joint_margin_rad=float(np.min(margin)),wrist_margin_rad=float(np.min(margin[4:])),joint_interpolation_clearance_m=distances,joint_interpolation_min_m=min(distances),
            accepted_endpoint=pe<.001 and oe<.02 and pair.distance_m>=.04)
        row['accepted_sequence_witness']=row['accepted_endpoint'] and min(distances)>=.04;rows.append(row);before=q.copy()
    accepted=all(r['accepted_sequence_witness'] for r in rows)
    receipt=dict(source_fixed_first_three=str(base/'screen-007-joint-relaxation'),first_three_fixed=True,objective='Squared joint changes (wrist weight1.35) plus .01 inverse-square joint-limit proximity penalty/.15rad offset',
        method='One14variableSLSQP max30iterations, fixedlast2positions/axes; axialrollfree; physicaljointbounds; signedCPUclearance40.1mm at4fractions/transition; unchangedmonitor40mm validation at21fractions',
        objective_before=objective(initial),objective_after=objective(result.x),optimizer_success=bool(result.success),message=str(result.message),iterations=len(iterations),elapsed_wall_s=time.monotonic()-began,
        before=original,after=rows,accepted=accepted,limitations='Joint-interpolation screening is a witness, not the actual MPC trajectory; local optimization only.')
    (out/'optimization.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k not in ('before','after')}),flush=True)


if __name__=='__main__':main()
