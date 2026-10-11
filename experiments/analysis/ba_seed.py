#!/usr/bin/env python3
"""Saved BA seed-to-own-output diagnostics; never run another solver call."""
import argparse
import csv
import json
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recording import sha, write_json


def unpack(plan):
    body=np.asarray(plan)[:651].reshape(31,21)
    return np.vstack((body[:,:14],np.asarray(plan)[-14:])),body[:,14:]


def bounds(model,states,controls):
    return bool(np.isfinite(states).all() and np.isfinite(controls).all() and
        np.all(states[:,:7]>=model.lowerPositionLimit-1e-4) and np.all(states[:,:7]<=model.upperPositionLimit+1e-4) and
        np.all(np.abs(states[:,7:])<=model.velocityLimit+1e-3) and np.all(np.abs(controls)<=model.effortLimit+1e-3))


def diagnostic(model,data,pin,rk4,plan,state,xyz,rotation):
    states,controls=unpack(plan)
    if not np.isfinite(plan).all():return dict(finite=False)
    pin.forwardKinematics(model,data,states[-1,:7].astype(float));pin.updateFramePlacements(model,data)
    pose=data.oMf[model.getFrameId('torso_lift_link')].inverse()*data.oMf[model.getFrameId('arm_right_tool_link')]
    result=dict(finite=True,position_error_m=float(np.linalg.norm(pose.translation-xyz)),
        tool_axis_error_deg=float(np.rad2deg(np.arccos(np.clip(pose.rotation[:,2]@rotation[:,2],-1.,1.)))),
        orientation_error_deg=float(np.rad2deg(np.linalg.norm(pin.log3(rotation.T@pose.rotation)))),
        speed_l1=float(np.abs(states[-1,7:]).sum()),predicted_knots_physical=bounds(model,states,controls))
    dense=[np.asarray(state,dtype=float).copy()]
    for _ in range(10):
        q,v=rk4(model,data,dense[-1][:7],dense[-1][7:],controls[0].astype(float),.003)
        dense.append(np.r_[q,v])
        if not np.isfinite(dense[-1]).all():break
    result['nominal_prefix_physical']=bounds(model,np.asarray(dense),controls[:1])
    result['modeled_endpoint_in_rough_region']=bool(result['position_error_m']<.05 and result['tool_axis_error_deg']<30 and result['speed_l1']<1)
    return result


def progress(initial,final,seed,output,steps):
    initial,final=float(initial),float(final)
    return dict(initial_merit=initial,final_merit=final,merit_decrease=initial-final,
        merit_available=bool(np.isfinite(initial) and np.isfinite(final)),
        accepted_steps=int(np.count_nonzero(np.asarray(steps)>0)),
        own_output_change_linf=float(np.max(np.abs(np.asarray(output)-np.asarray(seed)))))


def analyze(raw,source,output,limit=None):
    raw,source,output=map(lambda p:Path(p).resolve(),(raw,source,output))
    if output.is_relative_to(raw):raise ValueError('Derived output must be outside retained raw cohort')
    sys.path.insert(0,str(source/'python'))
    import pinocchio as pin
    from bsqp.common import rk4
    model_path=source/'gato/dynamics/tiago_right/tiago_right_arm.urdf'
    model=pin.buildModelFromUrdf(str(model_path));data=model.createData()
    output.mkdir(parents=True,exist_ok=True)
    fields=['path','cohort_id','protocol_sha256','binary_sha256','pair_id','batch','step','lane','winner','error',
        'initial_merit','final_merit','merit_decrease','merit_available','accepted_steps','own_output_change_linf']
    diagnostics=['finite','position_error_m','tool_axis_error_deg','orientation_error_deg','speed_l1',
        'predicted_knots_physical','nominal_prefix_physical','modeled_endpoint_in_rough_region']
    fields += [prefix+'_'+key for prefix in ('seed','output') for key in diagnostics]
    files=rows=errors=0
    with (output/'seed_to_output.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for path in sorted(raw.rglob('step_*.npz')):
            if not re.fullmatch(r'step_\d+\.npz',path.name):continue
            if limit is not None and files>=limit:break
            files+=1
            try:
                manifest=json.loads((path.parent/'manifest.json').read_text())
                with np.load(path,allow_pickle=False) as plans, np.load(path.with_name(path.stem+'_stats.npz'),allow_pickle=False) as stats, np.load(path.parent/'inputs.npz',allow_pickle=False) as goal:
                    count=manifest['batch']
                    if (plans['candidates'].shape!=(count,665) or plans['optimized'].shape!=(count,665) or
                        stats['stats__initial_merit'].shape!=(count,) or stats['stats__final_merit'].shape!=(count,) or
                        stats['stats__step_size'].ndim!=2 or stats['stats__step_size'].shape[1]!=count):
                        raise ValueError('Raw BA lane dimensions differ; no lane filtering')
                    for lane,(seed,optimized) in enumerate(zip(plans['candidates'],plans['optimized'])):
                        row={k:manifest.get(k) for k in ('cohort_id','protocol_sha256','binary_sha256','pair_id','batch')}
                        row.update(path=str(path),step=int(path.stem.split('_')[1]),lane=lane,winner=int(plans['winner']))
                        row.update(progress(stats['stats__initial_merit'][lane],stats['stats__final_merit'][lane],seed,optimized,stats['stats__step_size'][:,lane]))
                        for name,plan in (('seed',seed),('output',optimized)):
                            details=diagnostic(model,data,pin,rk4,plan,plans['actual_state'],goal['target_xyz'],goal['target_rotation'])
                            row.update({name+'_'+k:v for k,v in details.items()})
                        writer.writerow(row);rows+=1
            except (OSError,ValueError,KeyError,RuntimeError) as error:
                writer.writerow(dict(path=str(path),error=str(error)));errors+=1
    write_json(output/'summary.json',dict(files=files,lane_rows=rows,errors=errors,limited=limit is not None,
        model_sha256=sha(model_path),analysis_sha256=sha(__file__),raw=str(raw),source=str(source),
        scope='Matched own seed/output native merits and modeled endpoints/predicted knots/nominal30ms prefixes; no unoptimized closed-loop task success, optimizer stationarity, or collision certificate. No solver calls.'))
    return errors


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('raw','source','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--limit-files',type=int,help='Explicit development subset; saved summary labels limited scope')
    args=parser.parse_args()
    sys.exit(bool(analyze(args.raw,args.source,args.output,args.limit_files)))
