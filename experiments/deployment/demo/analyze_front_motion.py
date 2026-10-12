#!/usr/bin/env python3
"""Saved-motion analysis and bounded full-body collision sampling; no solver."""
import csv
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
import pinocchio as pin

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'tiago_src'),'/workspace/GATO-experiment-runner-simple/experiments']
from recording import load_controller_stats, iter_events, write_json
from gato_tiago.safety_monitor import build_tiago_collision_model, NamedJointState, state_to_qv, DEFAULT_COLLISION_URDF_PATH, DEFAULT_COLLISION_BLACKLIST_PATH
from gato_tiago.tiago_tools import compute_pair_distance_values, compute_pair_distances


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('raw',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--frames',type=int,default=180);args=parser.parse_args()
    if args.frames<2:raise ValueError('At least two actual-state frames required')
    raw=args.raw;out=args.output;out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((raw/'protocol.json').read_text());metrics=json.loads((raw/'metrics.json').read_text());stats=load_controller_stats(raw)
    model=pin.buildModelFromUrdf(str(ROOT/protocol['urdf']['path']));data=model.createData();tool=model.getFrameId('arm_right_tool_link')
    times=np.asarray(stats['timestamps']);q=np.asarray(stats['joint_positions']);v=np.asarray(stats['joint_velocities']);goals=np.asarray(stats['goal_indices'],dtype=int)
    if stats.get('terminal_state') is not None:
        terminal=np.asarray(stats['terminal_state']);times=np.r_[times,stats['terminal_time_s']];q=np.vstack((q,terminal[:7]));v=np.vstack((v,terminal[7:]));goals=np.r_[goals,min(len(protocol['goals'])-1,max(goals))]
    finite=np.isfinite(times)&np.isfinite(q).all(axis=1)&np.isfinite(v).all(axis=1)
    bad=np.flatnonzero(~finite);valid_end=int(bad[0]) if len(bad) else len(times)
    rows=[]
    for i,goal in enumerate(protocol['goals']):
        indices=np.flatnonzero((goals==i)&finite);row=dict(goal=goal['id'],rough_pass=metrics['rough_goal_success'][i],strict_outcome=metrics['settling_goal_outcomes'][i])
        if len(indices):
            index=int(indices[-1]);pin.framesForwardKinematics(model,data,q[index]);pose=data.oMf[tool];desired=pin.rpy.rpyToMatrix(*goal['pose_m_rad'][3:])
            row.update(last_recorded_time_s=float(times[index]),terminal_position_m=float(np.linalg.norm(pose.translation-np.asarray(goal['pose_m_rad'][:3]))),
                terminal_axis_deg=float(np.rad2deg(np.arccos(np.clip(pose.rotation[:,2]@desired[:,2],-1,1)))),terminal_full_rotation_deg=float(np.rad2deg(np.linalg.norm(pin.log3(pose.rotation.T@desired)))),terminal_speed_L1_rad_s=float(np.abs(v[index]).sum()))
            margin=np.minimum(q[index]-model.lowerPositionLimit,model.upperPositionLimit-q[index])
            row.update(achieved_q_rad=q[index].tolist(),achieved_joint_limit_margin_rad=float(np.min(margin)),achieved_wrist_limit_margin_rad=float(np.min(margin[4:])))
            reference=goal.get('ik_witness_q_rad',goal.get('orientation_selection_candidate_q_rad'))
            row['reference_configuration_kind']='qualified endpoint IK witness' if goal.get('IK_endpoint_qualified',True) else 'unqualified axial-roll selection candidate'
            row['distance_from_reference_configuration_rad']=float(np.linalg.norm(q[index]-np.asarray(reference)))
            if row['reference_configuration_kind']=='qualified endpoint IK witness':row['distance_from_ik_witness_rad']=row['distance_from_reference_configuration_rad']
        rows.append(row)
    # Actual saved feedback at <=10ms target spacing, retaining real samples,
    # plus every goal transition and the final finite endpoint. No synthetic path.
    collision_ids=np.unique(np.r_[np.clip(np.searchsorted(times[:valid_end],np.arange(times[0],times[valid_end-1]+.01,.01)),0,valid_end-1),
        np.flatnonzero(np.diff(goals[:valid_end])!=0),np.flatnonzero(np.diff(goals[:valid_end])!=0)+1,valid_end-1]).astype(int)
    collision=build_tiago_collision_model(urdf_path=DEFAULT_COLLISION_URDF_PATH,blacklist_path=DEFAULT_COLLISION_BLACKLIST_PATH,monitor_only_collision_pairs=True)
    body=protocol['CPU_selection']['body_positions'];collision_rows=[];worst=None
    for index in collision_ids:
        positions=dict(body,**{f'arm_right_{j+1}_joint':float(value) for j,value in enumerate(q[index])})
        cq,_=state_to_qv(collision.model,NamedJointState(position=positions,velocity={k:0. for k in positions}))
        distances=compute_pair_distance_values(collision,cq);minimum=float(np.min(distances))
        row=dict(time_s=float(times[index]),goal_index=int(goals[index]),feedback_index=int(index),minimum_clearance_m=minimum,closest_pair_index=int(np.argmin(distances)))
        collision_rows.append(row)
        if worst is None or minimum<worst['minimum_clearance_m']:
            closest=min(compute_pair_distances(collision,cq),key=lambda p:p.distance_m);worst=dict(row,closest_pair=closest.to_json())
    for i,row in enumerate(rows):
        selected=[r for r in collision_rows if r['goal_index']==i];row.update(collision_samples=len(selected),sampled_minimum_clearance_m=min((r['minimum_clearance_m'] for r in selected),default=None),sampled_below_40mm=sum(r['minimum_clearance_m']<.04 for r in selected),sampled_contacts=sum(r['minimum_clearance_m']<=0 for r in selected))
    collision_summary=dict(scope='Actual saved feedback samples selected at10ms targetspacing plusgoalboundaries; fixedbodyassumption; no continuoussweptcertificate or runtimecollisionenforcement inarm-onlyinternalplant',
        finite_feedback_prefix_samples=valid_end,nonfinite_feedback_samples=int((~finite).sum()),collision_samples=len(collision_rows),below_40mm=sum(r['minimum_clearance_m']<.04 for r in collision_rows),contacts=sum(r['minimum_clearance_m']<=0 for r in collision_rows),minimum=worst,all_sampled_at_least_40mm=all(r['minimum_clearance_m']>=.04 for r in collision_rows),body_positions=body)
    counts=Counter(r['closest_pair_index'] for r in collision_rows if r['minimum_clearance_m']<.04);pairs=[]
    for index,count in counts.most_common():
        pair=collision.geometry_model.collisionPairs[index];a=collision.geometry_model.geometryObjects[pair.first];b=collision.geometry_model.geometryObjects[pair.second]
        pairs.append(dict(closest_pair_index=index,geometry_a=a.name,geometry_b=b.name,samples_below_40mm=count,contact_samples=sum(r['closest_pair_index']==index and r['minimum_clearance_m']<=0 for r in collision_rows)))
    collision_summary.update(violating_closest_pairs=pairs,limitation='Counts identify the closest monitored pair per sampled state; simultaneous additional violating pairs are not exhaustively counted.')
    np.savez_compressed(out/'executed_feedback.npz',timestamps=times,joint_positions=q,joint_velocities=v,goal_indices=goals)
    with (out/'collision_samples.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(collision_rows[0]));writer.writeheader();writer.writerows(collision_rows)
    write_json(out/'collision_metrics.json',collision_summary);write_json(out/'per_goal.json',rows)
    # Compact editable actual frames; never interpolate IK configurations.
    frame_ids=np.unique(np.clip(np.searchsorted(times[:valid_end],np.linspace(times[0],times[valid_end-1],args.frames)),0,valid_end-1))
    with (out/'motion_frames.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['time_s','goal_index',*[f'q{i+1}_rad' for i in range(7)]])
        for index in frame_ids:writer.writerow([times[index],goals[index],*q[index]])
    animation_failure=dict(first_invalid_feedback_index=valid_end,first_invalid_time_s=float(times[valid_end]),last_finite_time_s=float(times[valid_end-1])) if len(bad) else None
    ct=np.asarray([r['time_s'] for r in collision_rows]);frame_collision=[]
    for index in frame_ids:
        sample=collision_rows[int(np.argmin(np.abs(ct-times[index])))];frame_collision.append(dict(sampled_time_s=sample['time_s'],sampled_clearance_m=sample['minimum_clearance_m']))
    write_json(out/'plot_inputs.json',dict(protocol=protocol,metrics=metrics,per_goal=rows,collision_metrics=collision_summary,animation_failure=animation_failure,
        feedback_samples=len(times),sampled_frames=len(frame_ids),terminal_time_s=float(times[-1]),frame_collision_samples=frame_collision))
    print(json.dumps(dict(task_success=metrics['task_success'],physical_pass=metrics['physical_pass'],per_goal=rows,collision=collision_summary),indent=2))


if __name__=='__main__':main()
