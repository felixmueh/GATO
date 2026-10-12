#!/usr/bin/env python3
"""Select full corner orientations using continuous axial roll in CPU IK only."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares
from screen_front_square import (ROOT, BODY_POSITIONS, START_Q, OFFSETS_M, LABELS,
    build_tiago_collision_model, NamedJointState, state_to_qv, compute_pair_distances,
    DEFAULT_COLLISION_URDF_PATH, DEFAULT_COLLISION_BLACKLIST_PATH)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    parser.add_argument('--center',default='.5,0,-.1');args=parser.parse_args()
    center=np.fromstring(args.center,sep=',');assert center.shape==(3,)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'screen_source.py').write_bytes(Path(__file__).read_bytes())
    model=pin.buildModelFromUrdf(str(ROOT/'gato/dynamics/tiago_right/tiago_right_arm.urdf'))
    data=model.createData();tool=model.getFrameId('arm_right_tool_link')
    collision=build_tiago_collision_model(urdf_path=DEFAULT_COLLISION_URDF_PATH,
        blacklist_path=DEFAULT_COLLISION_BLACKLIST_PATH,monitor_only_collision_pairs=True)
    old=json.loads((ROOT/'experiments/deployment/demo/proposal.json').read_text())
    known={g['id']:np.asarray(g['ik_witness_q_rad']) for g in old['goals']}
    def clearance(q):
        body=dict(BODY_POSITIONS,**{f'arm_right_{i+1}_joint':float(v) for i,v in enumerate(q)})
        cq,_=state_to_qv(collision.model,NamedJointState(position=body,velocity={k:0. for k in body}))
        return min(compute_pair_distances(collision,cq),key=lambda p:p.distance_m)
    previous=START_Q.copy();selected=[];attempts=[]
    for label,offset in zip(LABELS,np.asarray(OFFSETS_M)*(2/3)):
        target=center+offset;axis=-offset if label!='center' else np.array((1.,0.,0.));axis=axis/np.linalg.norm(axis)
        candidates=[]
        for seedlabel,seed in [('previous',previous),('initial',START_Q),
                              ('v5-top-left',known['top-left']),('v5-bottom-left',known['bottom-left'])]:
            def residual(q):
                pin.framesForwardKinematics(model,data,q);pose=data.oMf[tool]
                return np.r_[pose.translation-target,.15*(pose.rotation[:,2]-axis)]
            result=least_squares(residual,np.clip(seed,model.lowerPositionLimit+.001,model.upperPositionLimit-.001),
                bounds=(model.lowerPositionLimit+.001,model.upperPositionLimit-.001),max_nfev=220,
                ftol=1e-8,xtol=1e-8,gtol=1e-8)
            q=result.x;pin.framesForwardKinematics(model,data,q);pose=data.oMf[tool].copy()
            pe=float(np.linalg.norm(pose.translation-target));oe=float(np.arccos(np.clip(pose.rotation[:,2]@axis,-1,1)))
            delta=q-previous;minimum=clearance(q)
            margin=np.minimum(q-model.lowerPositionLimit,model.upperPositionLimit-q)
            # Align the solved tool Z exactly to the declared axis; its axial roll
            # remains the continuously selected IK result. Freeze this full R.
            cross=np.cross(pose.rotation[:,2],axis);crossnorm=np.linalg.norm(cross)
            R=pin.exp3(cross/crossnorm*oe)@pose.rotation if crossnorm>1e-12 else pose.rotation
            row=dict(center_m=center.tolist(),goal=label,target_m=target.tolist(),tool_axis=axis.tolist(),
                seed=seedlabel,q_rad=q.tolist(),position_error_m=pe,orientation_error_rad=oe,
                endpoint_min_m=minimum.distance_m,closest_pair=minimum.to_json(),nfev=result.nfev,
                pose_m_rad=target.tolist()+pin.rpy.matrixToRpy(R).tolist(),
                selected_rotation=R.tolist(),joint_travel_rad=float(np.linalg.norm(delta)),
                max_per_joint_step_rad=float(np.max(np.abs(delta))),wrist_travel_rad=float(np.linalg.norm(delta[4:])),
                joint_margin_rad=float(np.min(margin)),wrist_margin_rad=float(np.min(margin[4:])),
                score=float(np.linalg.norm(delta)+.35*np.linalg.norm(delta[4:])+.05/max(np.min(margin),.02)),
                accepted_endpoint=pe<.001 and oe<.02 and minimum.distance_m>=.04)
            attempts.append(row)
            if row['accepted_endpoint']:candidates.append(row)
        chosen=None
        for row in sorted(candidates,key=lambda r:r['score'])[:3]:
            distances=[clearance(previous+alpha*(np.asarray(row['q_rad'])-previous)).distance_m for alpha in np.linspace(0,1,21)]
            row['joint_interpolation_clearance_m']=distances;row['joint_interpolation_min_m']=min(distances)
            if min(distances)>=.04:chosen=row;break
        if chosen is None:
            print(json.dumps(dict(center=center.tolist(),failed_goal=label,endpoint_candidates=len(candidates))),flush=True);break
        selected.append(chosen);previous=np.asarray(chosen['q_rad'])
    receipt=dict(side_m=.20,centers_m=[center.tolist()],body_positions=BODY_POSITIONS,initial_q_rad=START_Q.tolist(),
        orientation_selection='Continuous axial roll free in CPU position+tool-Z IK; all full rotations then frozen, same inward corner axes and center+X. Runtime orientation costs unchanged.',
        scope='Endpoint IK/4cm monitored clearance and 21-sample joint interpolation; no MPC/dynamic/swept/physical qualification',
        attempts=attempts,center_outcomes=[dict(center_m=center.tolist(),goals=selected,complete=len(selected)==5)],
        selected=selected if len(selected)==5 else None,collision_urdf=str(DEFAULT_COLLISION_URDF_PATH),blacklist=str(DEFAULT_COLLISION_BLACKLIST_PATH))
    (args.output/'screen.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(accepted=len(selected)==5,center=center.tolist(),ik_attempts=len(attempts),
        goals=[dict(goal=g['goal'],clearance=g['endpoint_min_m'],travel=g['joint_travel_rad'],wrist_margin=g['wrist_margin_rad']) for g in selected])),flush=True)


if __name__=='__main__':main()
