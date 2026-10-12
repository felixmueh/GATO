#!/usr/bin/env python3
"""Bounded CPU setup screen for a front-facing 30 cm square; no controller."""
import json
from pathlib import Path
import sys

import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tiago_src'))
from gato_tiago.safety_monitor import (build_tiago_collision_model, NamedJointState,
    state_to_qv, DEFAULT_COLLISION_URDF_PATH, DEFAULT_COLLISION_BLACKLIST_PATH)
from gato_tiago.tiago_tools import compute_pair_distances
from screen_square import BODY_POSITIONS, START_Q, rotation

CENTERS_M = ((.35, 0., -.10), (.40, 0., -.10), (.55, 0., -.10), (.60, 0., -.10))
TWISTS_DEG = tuple(range(-180, 180, 30))
OFFSETS_M = ((0., -.15, -.15), (0., -.15, .15), (0., .15, .15),
             (0., .15, -.15), (0., 0., 0.))
LABELS = ('bottom-right', 'top-right', 'top-left', 'bottom-left', 'center')


def main():
    output = Path(sys.argv[1]); output.mkdir(parents=True, exist_ok=False)
    (output/'screen_source.py').write_bytes(Path(__file__).read_bytes())
    model = pin.buildModelFromUrdf(str(ROOT / 'gato/dynamics/tiago_right/tiago_right_arm.urdf'))
    data = model.createData(); tool = model.getFrameId('arm_right_tool_link')
    collision = build_tiago_collision_model(urdf_path=DEFAULT_COLLISION_URDF_PATH,
        blacklist_path=DEFAULT_COLLISION_BLACKLIST_PATH, monitor_only_collision_pairs=True)
    cdata = collision.model.createData()
    def collision_state(q):
        body = dict(BODY_POSITIONS, **{f'arm_right_{i+1}_joint': float(v) for i,v in enumerate(q)})
        return state_to_qv(collision.model, NamedJointState(position=body, velocity={k:0. for k in body}))[0]
    def clearance(q):
        pairs = compute_pair_distances(collision, collision_state(q))
        return min(pairs, key=lambda p:p.distance_m)
    cq = collision_state(START_Q); pin.framesForwardKinematics(collision.model, cdata, cq)
    torso = cdata.oMf[collision.model.getFrameId('torso_lift_link')].copy()
    full_tool = torso.inverse() * cdata.oMf[collision.model.getFrameId('arm_right_tool_link')]
    pin.framesForwardKinematics(model, data, START_Q); arm_tool=data.oMf[tool].copy()
    frame_check = dict(torso_in_base_translation_m=torso.translation.tolist(),
        torso_in_base_rotation=torso.rotation.tolist(), full_body_tool_in_torso_m=full_tool.translation.tolist(),
        arm_only_tool_m=arm_tool.translation.tolist(), fk_position_difference_m=float(np.linalg.norm(full_tool.translation-arm_tool.translation)),
        fk_rotation_difference_rad=float(np.linalg.norm(pin.log3(full_tool.rotation.T@arm_tool.rotation))),
        initial_collision_min_m=clearance(START_Q).distance_m)
    records=[]; centers=[]; selected=None
    for center in CENTERS_M:
        previous=START_Q.copy(); goals=[]
        for label, offset in zip(LABELS, OFFSETS_M):
            target=np.asarray(center)+offset
            axis=-np.asarray(offset) if label!='center' else np.array((1.,0.,0.))
            axis=axis/np.linalg.norm(axis)
            candidates=[]
            pin.framesForwardKinematics(model,data,previous); previous_R=data.oMf[tool].rotation.copy()
            for twist in TWISTS_DEG:
                R=rotation(axis,np.deg2rad(twist))
                def residual(q):
                    pin.framesForwardKinematics(model,data,q); pose=data.oMf[tool]
                    return np.r_[pose.translation-target, .15*pin.log3(R.T@pose.rotation)]
                for seed_label, seed in [('previous',previous),('initial',START_Q)]:
                    result=least_squares(residual,np.clip(seed,model.lowerPositionLimit+.001,model.upperPositionLimit-.001),
                        bounds=(model.lowerPositionLimit+.001,model.upperPositionLimit-.001),max_nfev=150,
                        ftol=1e-7,xtol=1e-7,gtol=1e-7)
                    q=result.x; pin.framesForwardKinematics(model,data,q)
                    pe=float(np.linalg.norm(data.oMf[tool].translation-target))
                    oe=float(np.linalg.norm(pin.log3(R.T@data.oMf[tool].rotation)))
                    row=dict(center_m=list(center),goal=label,target_m=target.tolist(),tool_axis=axis.tolist(),
                        twist_deg=twist,seed=seed_label,q_rad=q.tolist(),position_error_m=pe,
                        orientation_error_rad=oe,nfev=result.nfev,accepted_endpoint=False)
                    if pe<.001 and oe<.02:
                        closest=clearance(q)
                        margin=float(np.min(np.minimum(q-model.lowerPositionLimit,model.upperPositionLimit-q)))
                        travel=float(np.linalg.norm(q-previous)); wrist=float(np.linalg.norm(q[4:]-previous[4:]))
                        orient_change=float(np.linalg.norm(pin.log3(previous_R.T@R)))
                        row.update(endpoint_min_m=closest.distance_m,closest_pair=closest.to_json(),
                            joint_margin_rad=margin,joint_travel_rad=travel,wrist_travel_rad=wrist,
                            full_rotation_change_rad=orient_change,
                            score=travel+.35*wrist+.10*orient_change+.05/max(margin,.02),
                            pose_m_rad=target.tolist()+pin.rpy.matrixToRpy(R).tolist(),
                            accepted_endpoint=closest.distance_m>=.04)
                        if row['accepted_endpoint']:candidates.append(row)
                    records.append(row)
            chosen=None
            # At most three alternatives, with 21 fixed joint-interpolation samples each.
            # This is a screening witness, never an MPC path/collision certificate.
            for row in sorted(candidates,key=lambda x:x['score'])[:3]:
                distances=[clearance(previous+alpha*(np.asarray(row['q_rad'])-previous)).distance_m
                           for alpha in np.linspace(0.,1.,21)]
                row['joint_interpolation_min_m']=min(distances)
                row['joint_interpolation_clearance_m']=distances
                if min(distances)>=.04:chosen=row;break
            if chosen is None:
                print(json.dumps(dict(center=center,failed_goal=label,endpoint_candidates=len(candidates))),flush=True)
                break
            goals.append(chosen);previous=np.asarray(chosen['q_rad'])
        centers.append(dict(center_m=list(center),goals=goals,complete=len(goals)==5))
        if len(goals)==5:
            selected=goals;break
    receipt=dict(side_m=.30,centers_m=CENTERS_M,twists_deg=TWISTS_DEG,body_positions=BODY_POSITIONS,
        initial_q_rad=START_Q.tolist(),frame_check=frame_check,attempts=records,center_outcomes=centers,
        selected=selected,scope='CPU full-pose IK endpoints and 21-sample joint-interpolation witnesses only; no controller or dynamic/swept collision certificate',
        collision_urdf=str(DEFAULT_COLLISION_URDF_PATH),blacklist=str(DEFAULT_COLLISION_BLACKLIST_PATH))
    (output/'screen.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(accepted=bool(selected),center=selected[0]['center_m'] if selected else None,
        ik_attempts=len(records),frame_check=frame_check)),flush=True)


if __name__=='__main__':main()
