#!/usr/bin/env python3
"""Saved visual-demo feedback paths, errors and existing arm-animation helper.

Read recorded outcomes/selection without native solver calls or robot motion.
Every attempt and backend remains separate; animation frames are a display subset
only. No swept collision or physical compliance certificate is inferred.
"""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recording import sha,write_json
from analysis.f01ab import csv_write


def targets(protocol,stats,time):
    if protocol['kind']=='tracking':
        reference=np.asarray(protocol['reference_packed']).reshape(-1,6)
        clock=np.arange(len(reference))*protocol['reference_dt_s']
        return np.column_stack([np.interp(time,clock,reference[:,i]) for i in range(6)])
    goals=np.asarray([g['pose_m_rad'] for g in protocol['goals']])
    indices=np.asarray(stats.get('goal_indices',[]),dtype=int)
    if indices.shape!=(len(time),) or np.any(indices<0) or np.any(indices>=len(goals)):
        raise ValueError('Recorded measured-goal labels missing or invalid; no guessed goal switching')
    return goals[indices]


def analyze(raw,source,output,gifs=True):
    raw,source,output=map(lambda p:Path(p).resolve(),(raw,source,output))
    if output.is_relative_to(raw):raise ValueError('Derived output must be outside retained raw')
    output.mkdir(parents=True,exist_ok=False)
    import pinocchio as pin
    model_path=source/'gato/dynamics/tiago_right/tiago_right_arm.urdf'
    model=pin.buildModelFromUrdf(str(model_path));data=model.createData();tool=model.getFrameId('arm_right_tool_link')
    helper_path=source/'tiago_examples/tiago_reach_target.py'
    spec=importlib.util.spec_from_file_location('saved_demo_arm_render',helper_path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    attempts=[];selection=[];errors=[]
    for path in sorted(raw.rglob('index.json')):
        try:
            index=json.loads(path.read_text())
            if not isinstance(index,list):continue
            for row in index:
                if isinstance(row,dict) and row.get('task_id'):
                    selection.append(dict(index=str(path),index_sha256=sha(path),task_id=row['task_id'],backend=row.get('backend'),status=row.get('status'),reason=row.get('reason'),raw_path=row.get('raw_path')))
        except (OSError,ValueError,TypeError) as error:errors.append(dict(path=str(path),error=str(error)))
    for metrics_path in sorted(raw.rglob('metrics.json')):
        row=None
        try:
            metrics=json.loads(metrics_path.read_text())
            if metrics.get('campaign_id')!='demo_visual_square':continue
            attempt=metrics_path.parent
            row=dict(attempt=str(attempt),metrics_sha256=sha(metrics_path),task_id=metrics.get('task_id','sequence'),backend=metrics.get('backend'),machine=metrics.get('machine'),
                evidence_kind=metrics.get('evidence_kind','recorded'),task_success=metrics.get('task_success'),settling_success=metrics.get('settling_success'),physical_pass=metrics.get('physical_pass'),execution_error=metrics.get('execution_error'),handback_confirmed=metrics.get('handback_confirmed'))
            attempts.append(row)
            if not (attempt/'protocol.json').exists():row['visual_unavailable']='Sequence aggregate; subtask artifacts rendered separately';continue
            protocol=json.loads((attempt/'protocol.json').read_text());stats=json.loads((attempt/'controller_stats.json').read_text())
            provenance=json.loads((attempt/'provenance.json').read_text())
            row.update(runtime_source=provenance.get('runtime_source_commit'),module_sha256=provenance.get('module',{}).get('sha256'),input_sha256=protocol.get('input_sha256'))
            if sha(model_path)!=protocol['urdf']['sha256']:raise ValueError('Visualization model differs from recorded task asset')
            time=np.asarray(stats.get('timestamps',[]),dtype=float);q=np.asarray(stats.get('joint_positions',[]),dtype=float)
            if q.shape!=(len(time),7) or not len(time):row['visual_unavailable']='No complete recorded feedback trace';continue
            reference=targets(protocol,stats,time);actual=[];rotations=[]
            for joints in q:
                if np.isfinite(joints).all():
                    pin.framesForwardKinematics(model,data,joints);actual.append(data.oMf[tool].translation.copy());rotations.append(data.oMf[tool].rotation.copy())
                else:actual.append(np.full(3,np.nan));rotations.append(np.full((3,3),np.nan))
            actual=np.asarray(actual);rotations=np.asarray(rotations)
            position=np.linalg.norm(actual-reference[:,:3],axis=1)
            reference_axes=np.asarray([pin.rpy.rpyToMatrix(*r[3:])[:,2] for r in reference])
            axis=np.rad2deg(np.arccos(np.clip(np.sum(rotations[:,:,2]*reference_axes,axis=1),-1.,1.)))
            derived=output/f'attempt-{len(attempts):03d}';derived.mkdir()
            np.savetxt(derived/'feedback_trace.csv',np.column_stack((time,q,actual,reference,position,axis)),delimiter=',',
                header='time_s,'+','.join(f'q{i}_rad' for i in range(1,8))+',x_m,y_m,z_m,target_x_m,target_y_m,target_z_m,target_roll_rad,target_pitch_rad,target_yaw_rad,position_error_m,tool_axis_error_deg',comments='')
            fig=plt.figure(figsize=(12,5));path_axis=fig.add_subplot(121,projection='3d');error_axis=fig.add_subplot(222);angle_axis=fig.add_subplot(224)
            path_axis.plot(*actual.T,label='Executed feedback path');path_axis.plot(*reference[:,:3].T,':',label='Active reference')
            goals=np.asarray([g['pose_m_rad'] for g in protocol['goals']]) if protocol['kind']=='reaching' else reference[np.linspace(0,len(reference)-1,min(8,len(reference))).astype(int)]
            for goal in goals:
                direction=pin.rpy.rpyToMatrix(*goal[3:])[:,2];path_axis.quiver(*goal[:3],*direction,length=.06,color='red')
            path_axis.set(xlabel='Torso x (m)',ylabel='Torso y (m)',zlabel='Torso z (m)');path_axis.legend(fontsize=8)
            error_axis.plot(time,position);error_axis.set(ylabel='Position error (m)')
            angle_axis.plot(time,axis);angle_axis.set(xlabel='Recorded plant/ROS time (s)',ylabel='Tool-axis error (deg)')
            fig.suptitle(f"{row['backend']} / {row['task_id']} / {row['machine']} / {row['evidence_kind']}\nrough={row['task_success']}, settling={row['settling_success']}, physical={row['physical_pass']}; recorded feedback, no swept collision claim",fontsize=9)
            fig.tight_layout();fig.savefig(derived/'path-and-errors.png',dpi=130);plt.close(fig)
            row.update(derived_path=str(derived),feedback_samples=len(time),nonfinite_feedback_samples=int(np.count_nonzero(~np.isfinite(q).all(axis=1))))
            if gifs:
                if not np.isfinite(q).all() or not np.isfinite(time).all():row['animation_unavailable']='Nonfinite feedback retained in trace; no filtered replacement animation'
                else:
                    ids=np.linspace(0,len(time)-1,min(220,len(time))).astype(int)
                    arm=np.asarray([helper.arm_link_positions(model,data,joints) for joints in q[ids]])
                    joint_axes=[helper.joint_axis_overlays(model,data,joints) for joints in q[ids]]
                    tool_axes=np.asarray([helper.tool_axis_segment(model,data,joints) for joints in q[ids]])
                    helper.save_gif(derived,time[ids],actual[ids],reference[ids],goals,arm,joint_axes,tool_axes,interactive=False,write_gif=True)
                    row['animation_scope']='Existing arm skeleton/tool/joint-axis helper; at most220 display frames, all feedback exported; timestamp labels retained'
        except (OSError,ValueError,KeyError,TypeError,RuntimeError) as error:
            errors.append(dict(path=str(metrics_path),error=str(error)))
            if row is not None:row['visual_error']=str(error)
    csv_write(output/'attempts.csv',attempts);csv_write(output/'selection.csv',selection)
    write_json(output/'summary.json',dict(attempts=attempts,selection=selection,errors=errors,model_sha256=sha(model_path),helper_sha256=sha(helper_path),analysis_sha256=sha(__file__),
        scope='Saved recorded feedback paths, active target axes and error traces. Actual backend/outcomes/skips remain explicit. Endpoint screens and drawings do not certify swept clearance, motor torque or task reliability.'))
    return bool(errors)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('raw','source','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--no-gifs',action='store_true')
    args=parser.parse_args();raise SystemExit(analyze(args.raw,args.source,args.output,not args.no_gifs))
