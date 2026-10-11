#!/usr/bin/env python3
"""Saved cylinder attempts, including failures and retries without pooling."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recording import iter_events,sha,write_json
from analysis.f01ab import csv_write,read,fingerprint


def mean(values):
    x=np.asarray(values,dtype=float)
    return float(np.mean(x)) if x.size and np.isfinite(x).all() else None


def native_progress(seed,output,stats,lane):
    delta=np.asarray(output)-np.asarray(seed)
    result=dict(own_output_change_linf=float(np.max(np.abs(delta))) if np.isfinite(delta).all() else None,
        own_output_array_unchanged=bool(np.array_equal(seed,output,equal_nan=True)),seed_finite=bool(np.isfinite(seed).all()),output_finite=bool(np.isfinite(output).all()))
    for name in ('initial_merit','final_merit'):
        values=stats.get('stats__'+name)
        result[name]=float(values[lane]) if values is not None else None
    steps=stats.get('stats__step_size')
    result['accepted_steps']=int(np.count_nonzero(np.asarray(steps)[:,lane]>0)) if steps is not None else None
    return result


def model_limits(source,commit):
    if source is None:return None
    path=source/'gato/dynamics/tiago_right/tiago_right_arm.urdf'
    original=subprocess.check_output(['git','-C',str(source),'show',commit+':gato/dynamics/tiago_right/tiago_right_arm.urdf'])
    if path.read_bytes()!=original:raise ValueError('Analysis model differs from recorded source commit')
    import pinocchio as pin
    model=pin.buildModelFromUrdf(str(path))
    return dict(lower=model.lowerPositionLimit,upper=model.upperPositionLimit,velocity=model.velocityLimit,effort=model.effortLimit,sha256=sha(path))


def attempt(path,root,source=None):
    receipt,metrics,provenance,protocol=(read(path/(name+'.json')) for name in ('receipt','metrics','provenance','protocol'))
    case=provenance.get('case',receipt.get('case',{}));params=case.get('parameters',{})
    scene=protocol.get('scene');inputs={};returns={};stats={};assessment={};tool=None;parse_error=None
    if (path/'events.jsonl').exists():
        try:
            for event in iter_events(path):
                kind,r=event['event'],event['record']
                if kind=='initial_seeds':scene=scene or r.get('scene')
                elif kind=='solve_input':inputs[r['step']]=r
                elif kind=='solve_returned':returns[r['step']]=r
                elif kind=='solve_stats':stats[r['step']]=r
                elif kind=='candidate_assessment':assessment[r['step']]=r
                elif kind=='executed_clearance':tool=r
        except (OSError,ValueError,KeyError,TypeError) as error:parse_error=str(error)
    study=metrics.get('campaign_id',params.get('study'))
    scene_name=params.get('scene','easy30' if study=='X03' else 'isolated-cylinder-checks')
    spec=receipt.get('identity',{})
    if isinstance(spec.get('experiment'),dict):spec=spec['experiment']
    source_commit=provenance.get('source_commit',spec.get('source_commit',receipt.get('source_commit')))
    identity=dict(machine=receipt.get('machine',metrics.get('machine')),study=study,scene=scene_name,B=params.get('batch'),
        source_commit=source_commit,module_sha256=provenance.get('module_sha256'),
        science_identity=fingerprint(provenance.get('science_sha256',{})),input_identity=fingerprint(provenance.get('inputs_sha256',{})),
        adapter_sha256=provenance.get('adapter_sha256'),timing_mode=metrics.get('timing_mode',spec.get('timing_mode',receipt.get('timing_mode'))),backend=metrics.get('execution_backend',spec.get('backend',receipt.get('backend'))),
        evidence_kind=metrics.get('evidence_kind','measurement'),protocol_id=spec.get('protocol',receipt.get('protocol')),protocol_identity=fingerprint(protocol) if protocol else None)
    base=dict(identity,attempt=str(path.relative_to(root)),attempt_status=receipt.get('status'),outcome=metrics.get('outcome'),
        execution_error=metrics.get('execution_error',receipt.get('error')),raw_read_error=parse_error)
    summary=dict(base,task_success=metrics.get('task_success'),physical_pass=metrics.get('physical_pass'),
        collision_samples_pass=metrics.get('collision_samples_pass'),solver_numerical_pass=metrics.get('solver_numerical_pass'),
        simulated_seconds=metrics.get('simulated_seconds'),reached_seconds=metrics.get('reached_seconds'),
        min_tool_clearance_m=metrics.get('min_executed_tool_clearance_m'),controller_failure=metrics.get('controller_failure'),
        mean_wrapper_ms=1000*mean(metrics['wall_solve_times_s']) if mean(metrics.get('wall_solve_times_s',[])) is not None else None,
        mean_native_ms=mean(metrics.get('native_solve_times_ms',[])),retained_inputs=len(inputs),retained_returns=len(returns))
    candidates=[]
    for step,record in inputs.items():
        seeds=np.asarray(record['candidates']);reply=returns.get(step);stat=stats.get(step,{})
        optimized=np.asarray(reply['optimized']) if reply is not None else None
        for lane,seed in enumerate(seeds):
            row=dict(base,step=step,lane=lane,seed_origin='supplied-route prefix' if step==0 and study=='X03' else 'own-lane previous output' if study=='X03' else 'retained early_smooth',
                winner=assessment.get(step,{}).get('winner'),output_available=optimized is not None)
            if optimized is not None:
                if optimized.shape!=seeds.shape:raise ValueError('Saved candidate/output dimensions differ')
                row.update(native_progress(seed,optimized[lane],stat,lane))
            candidates.append(row)
    trajectory=[];limit_error=None;limits=None
    if source is not None and source_commit:
        try:limits=model_limits(source,source_commit)
        except (OSError,ValueError,RuntimeError,subprocess.CalledProcessError) as error:limit_error=str(error)
    summary.update(physical_trace_unavailable_reason=limit_error or ('Supply matching --source for model-limit traces' if limits is None else None),model_sha256=limits['sha256'] if limits else None)
    if (path/'executed.npz').exists():
        with np.load(path/'executed.npz',allow_pickle=False) as data:states=data['states'];controls=data['controls'];times=data['times']
        if states.shape!=(len(times),14) or controls.shape!=(max(0,len(times)-1),7):raise ValueError('Saved executed trajectory dimensions differ')
        if tool is not None and len(tool['tool_xyz'])!=len(times):raise ValueError('Saved tool trace is not aligned with executed states')
        for i,(state,t) in enumerate(zip(states,times)):
            row=dict(base,sample=i,time_s=float(t),speed_l1_rad_s=float(np.abs(state[7:]).sum()),position_error_m=None,tool_clearance_m=None,
                joint_bound_excess_rad=None,velocity_limit_ratio=None,effort_limit_ratio=None)
            if tool is not None:
                row['tool_clearance_m']=float(tool['clearance_m'][i])
                if scene:row['position_error_m']=float(np.linalg.norm(np.asarray(tool['tool_xyz'][i])-scene['goal_xyz']))
            if limits:
                row['joint_bound_excess_rad']=float(max(0.,np.max(np.maximum(limits['lower']-state[:7],state[:7]-limits['upper']))))
                row['velocity_limit_ratio']=float(np.max(np.abs(state[7:])/limits['velocity']))
                if i:row['effort_limit_ratio']=float(np.max(np.abs(controls[i-1])/limits['effort']))
            trajectory.append(row)
    return summary,candidates,trajectory


def plots(summaries,trajectories,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for number,summary in enumerate(summaries):
        rows=[r for r in trajectories if r['attempt']==summary['attempt']]
        if not rows:continue
        fig,axes=plt.subplots(2,2,figsize=(10,7));axes=axes.ravel();t=[r['time_s'] for r in rows]
        for ax,field,label in zip(axes[:3],('position_error_m','speed_l1_rad_s','tool_clearance_m'),('Position error (m)','Speed L1 (rad/s)','Tool clearance (m)')):
            ax.plot(t,[np.nan if r[field] is None else r[field] for r in rows]);ax.set_ylabel(label);ax.set_xlabel('Simulated time (s)')
        axes[0].axhline(.05,color='gray',linestyle=':');axes[1].axhline(1,color='gray',linestyle=':');axes[2].axhline(0,color='gray',linestyle=':')
        for field in ('velocity_limit_ratio','effort_limit_ratio'):
            axes[3].plot(t,[np.nan if r[field] is None else r[field] for r in rows],label=field)
        axes[3].axhline(1,color='gray',linestyle=':');axes[3].legend(fontsize=7);axes[3].set_ylabel('Recorded model limit ratios');axes[3].set_xlabel('Simulated time (s)')
        fig.suptitle(f"{summary['machine']} {summary['scene']} B{summary['B']} / {summary['attempt']}\nTask={summary['task_success']}, physical={summary['physical_pass']}, outcome={summary['outcome']}",fontsize=9)
        fig.tight_layout();fig.savefig(output/f'attempt-{number:03d}.png',dpi=120);plt.close(fig)
    groups={}
    for row in summaries:
        if row['B'] is None:continue
        keys=('machine','study','scene','source_commit','module_sha256','science_identity','input_identity','adapter_sha256','timing_mode','backend','protocol_id','evidence_kind')
        key=tuple(row.get(k) for k in keys);groups.setdefault(key,[]).append(row)
    for number,rows in enumerate(groups.values()):
        fig,axes=plt.subplots(1,3,figsize=(12,4));labels=[f"B{r['B']} / {Path(r['attempt']).name}" for r in rows];x=range(len(rows))
        for ax,field in zip(axes,('task_success','physical_pass','min_tool_clearance_m')):
            ax.scatter(x,[np.nan if r[field] is None else r[field] for r in rows]);ax.set_ylabel(field);ax.set_xticks(list(x),labels,rotation=30)
        fig.suptitle(f"{rows[0]['machine']} {rows[0]['scene']}: one point per attempt, failures/retries retained\nDescriptive task comparison; no equal-quality speedup or convergence claim",fontsize=9)
        fig.tight_layout();fig.savefig(output/f'comparison-{number:03d}.png',dpi=120);plt.close(fig)


def analyze(raw,output,source=None,make_plots=True):
    raw,output=Path(raw).resolve(),Path(output).resolve();source=Path(source).resolve() if source else None
    if output.is_relative_to(raw):raise ValueError('Derived output must be outside retained raw')
    output.mkdir(parents=True,exist_ok=True);summaries=[];candidates=[];trajectories=[];errors=[]
    for receipt_path in sorted(raw.rglob('receipt.json')):
        path=receipt_path.parent
        try:
            receipt=read(receipt_path);metrics=read(path/'metrics.json')
            eid=receipt.get('experiment',receipt.get('identity',{}).get('experiment'))
            if isinstance(eid,dict):eid=eid.get('id')
            if receipt.get('kind')=='preflight' or not (metrics.get('campaign_id') in ('X02','X03','X04') or eid in ('CylinderChecks','CylinderTasks')):continue
            summary,cs,ts=attempt(path,raw,source);summaries.append(summary);candidates.extend(cs);trajectories.extend(ts)
        except (OSError,ValueError,KeyError,TypeError) as error:
            errors.append(dict(attempt=str(path.relative_to(raw)),error=str(error)))
            summaries.append(dict(attempt=str(path.relative_to(raw)),B=None,raw_read_error=str(error),outcome='analysis_read_failed'))
    csv_write(output/'attempts.csv',summaries);csv_write(output/'seed_to_own_output.csv',candidates);csv_write(output/'trajectory.csv',trajectories);csv_write(output/'errors.csv',errors)
    write_json(output/'summary.json',dict(attempts=summaries,errors=errors,analysis_sha256=sha(__file__),raw=str(raw),
        scope='All saved attempts, no retry pooling or new measurements. Existing task/physical outcomes unchanged; model-limit traces descriptive. Native own seed/output comparison is not convergence certification. Tool-center sampled cylinder clearance is not full-arm or continuous safety.'))
    if make_plots:plots(summaries,trajectories,output)
    return len(errors)+sum(bool(s['raw_read_error']) for s in summaries if s.get('outcome')!='analysis_read_failed')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--source',type=Path,help='Matching pinned source for optional physical model-limit time traces');p.add_argument('--no-plots',action='store_true');a=p.parse_args()
    raise SystemExit(bool(analyze(a.raw,a.output,a.source,not a.no_plots)))
