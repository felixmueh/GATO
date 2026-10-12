#!/usr/bin/env python3
"""Bounded saved-data validation of production recording projections; no solves.

Original attempts are read-only. Derived raw belongs outside thesis exports.
Deterministic selection is declared before outcomes are read: current GTX1070
successor cohort, T01 figure8-24s; adapted force 10N lowest repeat; F01c B16
middle budget including its declared correspondence diagnostics.
"""
import argparse
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile
import time
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recording import iter_events, json_safe, load_controller_stats, sha, write_json
from adapters.task_recording import TaskRecorder, project_task_record
from adapters.force_recording import ForceRecorder, project_force_record

COHORT = Path('/workspace/GATO/_expr_results/gtx1070-zero-pcg-20261011T214000Z/gtx1070')
SELECTIONS = {
    'tracking': ('T01', 'figure8-24s*'),
    'force': ('E02-adapted', 'figure8-10N-r0*'),
    'batching': ('F01c', 'N32-B16-SQP5-PCG100*'),
}


def digest_value(value, digest):
    """Stable decoded-value digest, independent of chunk packing/references."""
    if isinstance(value, np.ndarray):
        digest.update(str(value.dtype).encode()); digest.update(str(value.shape).encode())
        digest.update(value.tobytes(order='C'))
    elif isinstance(value, np.generic):
        digest_value(np.asarray(value), digest)
    elif isinstance(value, dict):
        for k in sorted(value):
            digest.update(k.encode()); digest_value(value[k], digest)
    elif isinstance(value, (list, tuple)):
        digest.update(str(len(value)).encode())
        for v in value: digest_value(v, digest)
    else:
        digest.update(json.dumps(value, sort_keys=True, allow_nan=True).encode())


def semantic_digest_value(value, digest):
    """Numeric lists and arrays retain the same values and nested shape."""
    if isinstance(value,np.ndarray):value=value.tolist()
    elif isinstance(value,np.generic):value=value.item()
    if isinstance(value,dict):
        for k in sorted(value):digest.update(k.encode());semantic_digest_value(value[k],digest)
    elif isinstance(value,(list,tuple)):
        digest.update(str(len(value)).encode())
        for v in value:semantic_digest_value(v,digest)
    else:digest.update(json.dumps(value,sort_keys=True,allow_nan=True).encode())


def storage(path):
    files = sorted(p for p in path.rglob('*') if p.is_file())
    result = dict(total_bytes=sum(p.stat().st_size for p in files), files=len(files),
                  event_metadata_bytes=(path/'events.jsonl').stat().st_size,
                  raw_npz_bytes=sum(p.stat().st_size for p in (path/'raw').glob('*.npz')))
    # ZIP metadata reads are bounded; no array expansion needed for payload size.
    result['raw_npy_bytes'] = 0
    for p in (path/'raw').glob('*.npz'):
        with zipfile.ZipFile(p) as z:
            result['raw_npy_bytes'] += sum(i.file_size for i in z.infolist())
    return result


def source_events_without_duplicate_history(source):
    """Skip discarded aggregate before NPZ decoding; historical layout only."""
    if (source/'recording.json').exists():
        yield from iter_events(source)
        return
    archive=None;raw_file=None
    try:
        with (source/'events.jsonl').open() as stream:
            for line in stream:
                if '"event": "final_raw_solves"' in line[:200]:continue
                event=json.loads(line)
                if event['raw_file']!=raw_file:
                    if archive is not None:archive.close()
                    raw_file=event['raw_file'];archive=np.load(source/raw_file,allow_pickle=False)
                def unpack(v):
                    if isinstance(v,dict):
                        if 'npz_key' in v:
                            a=archive[v['npz_key']].copy()
                            return a[()] if v.get('scalar') else a
                        return {k:unpack(x) for k,x in v.items()}
                    if isinstance(v,list):return [unpack(x) for x in v]
                    return v
                yield dict(event,record=unpack(event['record']))
    finally:
        if archive is not None:archive.close()


def iter_validation_events(source):
    """Same saved chunk/reference decoding, with one bounded physical chunk read."""
    archive=None;raw_file=None;recent={};references=(source/'recording.json').exists()
    try:
        with (source/'events.jsonl').open() as stream:
            for line in stream:
                if not line.endswith('\n'):break
                if '"event": "final_raw_solves"' in line[:200]:continue
                event=json.loads(line)
                if event['raw_file']!=raw_file:
                    if archive is not None:archive.close()
                    raw_file=event['raw_file'];target=source/raw_file
                    # Current immutable adapter chunks are bounded at16MiB plus
                    # one oversized observation. Never buffer giant aggregate histories.
                    if target.stat().st_size>64*1024*1024:raise ValueError('Unexpected oversized validation chunk')
                    archive=np.load(io.BytesIO(target.read_bytes()),allow_pickle=False)
                def unpack(v):
                    if isinstance(v,dict):
                        if 'npz_key' in v:
                            a=archive[v['npz_key']].copy()
                            return a[()] if v.get('scalar') or 'nonfinite_scalar' in v else a
                        return {k:unpack(x) for k,x in v.items()}
                    if isinstance(v,list):return [unpack(x) for x in v]
                    return v
                r=unpack(event['record'])
                if 'record_ref' in event:
                    ref=event['record_ref']
                    if ref not in recent or not 0<event['event_index']-ref<=16:raise ValueError('Invalid reference')
                    base=deepcopy(recent[ref])
                    for k in event.get('record_removed',[]):del base[k]
                    base.update(r);r=base
                if references:
                    recent[event['event_index']]=deepcopy(r);recent.pop(event['event_index']-16,None)
                yield dict(event,record=r)
    finally:
        if archive is not None:archive.close()


class Observations:
    def __init__(self):
        self.counts = Counter(); self.motion = []; self.solves = []; self.intervals = []
        self.force = []; self.native_work = hashlib.sha256(); self.motion_hash = hashlib.sha256()

    def add(self, kind, r):
        self.counts[kind] += 1
        if kind == 'step_completed':
            digest_value({k:r[k] for k in ('state_before','state_after','control','time_s','step_dt_s')}, self.motion_hash)
            self.motion.append((r['time_s'], r['step_dt_s'], r['state_before'], r['state_after'], r['control']))
        elif kind == 'solve_completed':
            self.solves.append((r.get('wall_solve_time_s'), r.get('native_solve_time_ms')))
            semantic_digest_value(r.get('solver_stats',{}), self.native_work)
        elif kind == 'interval_completed':
            self.intervals.append((r['time_s'], r['update_interval_s'], r['requested_interval_s']))
        elif kind in ('force_tick','force_bootstrap'):
            self.force.append((r.get('estimated_force_error_N'),r.get('estimated_torque_error_Nm'),
                               r.get('selected_index'),r.get('selected_force',r.get('force'))))

    def summary(self):
        a = np.asarray(self.solves, dtype=float)
        intervals = np.asarray(self.intervals, dtype=float)
        return dict(event_counts=dict(self.counts), plant_substeps=len(self.motion),
                    motion_sha256=self.motion_hash.hexdigest(), native_work_sha256=self.native_work.hexdigest(),
                    solve_wall_ms=dict(count=len(a), mean=float(np.mean(a[:,0])*1000), p95=float(np.percentile(a[:,0],95)*1000)) if len(a) else None,
                    solve_native_ms=dict(count=len(a),mean=float(np.mean(a[:,1])),p95=float(np.percentile(a[:,1],95))) if len(a) else None,
                    above_nominal_share=float(np.mean(intervals[:,1]>.03)) if len(intervals) else None,
                    force_ticks_and_bootstrap=len(self.force))


def motion_metrics(observations, source, kind, source_root):
    import pinocchio as pin
    sys.path.insert(0,str(source_root/'python'))
    from bsqp.common import sample_reference
    rows=observations.motion
    x=np.asarray([rows[0][2]]+[r[3] for r in rows]);u=np.asarray([r[4] for r in rows])
    t=np.asarray([rows[0][0]]+[r[0]+r[1] for r in rows])
    model=pin.buildModelFromUrdf(str(source_root/'gato/dynamics/tiago_right/tiago_right_arm.urdf'))
    data=model.createData();frame=model.getFrameId('arm_right_tool_link')
    protocol=json.loads((source/'protocol.json').read_text())
    config=protocol['task'] if kind=='force' else protocol
    reference=np.asarray(config['reference_packed'],dtype=np.float32)
    targets=sample_reference(reference,t,config['reference_dt_s'])
    xyz=[]
    for state in x:
        pin.framesForwardKinematics(model,data,state[:7].astype(float));xyz.append(data.oMf[frame].translation.copy())
    xyz=np.asarray(xyz); errors=np.linalg.norm(xyz-targets[:,:3],axis=1)
    aggregate_matches=None
    if (source/'execution.npz').exists():
        with np.load(source/'execution.npz',allow_pickle=False) as a:
            aggregate_matches=all(np.array_equal(v,a[k]) for k,v in [('x',x),('u',u),('t',t)])
    return dict(states=len(x),controls=len(u),duration_s=float(t[-1]),aggregate_execution_match=aggregate_matches,
        finite=bool(np.isfinite(x).all() and np.isfinite(u).all()),
        time_weighted_RMS_m=float(np.sqrt(np.trapezoid(errors**2,t)/(t[-1]-t[0]))), maximum_error_m=float(np.max(errors)),
        q_excess_rad=float(np.max(np.maximum(np.maximum(model.lowerPositionLimit-x[:,:7],x[:,:7]-model.upperPositionLimit),0))),
        v_excess_rad_s=float(np.max(np.maximum(abs(x[:,7:])-model.velocityLimit,0))),
        u_excess_Nm=float(np.max(np.maximum(abs(u)-model.effortLimit,0)))),(t,xyz,targets[:,:3],errors)


def validate(kind,args):
    family,pattern=SELECTIONS[kind]
    pool=sorted((COHORT/family).glob('*/attempt-*'))
    selected=sorted((COHORT/family).glob(pattern))
    if len(selected)!=1:raise ValueError('Representative selection is ambiguous')
    source=selected[0]/'attempt-001';output=args.raw_output/kind
    output.mkdir(parents=True,exist_ok=False)
    original=Observations(); reduced=Observations(); expected=hashlib.sha256();actual=hashlib.sha256()
    count=0; started=time.monotonic()
    projector=project_force_record if kind=='force' else project_task_record
    cls=ForceRecorder if kind=='force' else TaskRecorder
    if kind=='batching':
        # Explicit matched-input exemption: retain complete I/O, including warmup.
        from recording import RawRecorder
        cls=RawRecorder;projector=lambda k,r:r
    with cls(output) as recorder:
        for event in source_events_without_duplicate_history(source):
            k,r=event['event'],event['record']
            if k=='final_raw_solves':continue # duplicate of preceding lifecycle events
            original.add(k,r)
            p=projector(k,r)
            digest_value((k,p),expected)
            recorder(k,r)
            if kind=='force' and k=='solve_started':recorder.flush() # pinned worker durability seam
            count+=1
            if count%20000==0:print(kind,count,round(time.monotonic()-started,1),'seconds',flush=True)
        if (source/'controller_stats.json').exists():
            stats=json.loads((source/'controller_stats.json').read_text())
            if kind=='force':
                # The exact pinned worker writes ordinary numeric JSON unchanged.
                shutil.copy2(source/'controller_stats.json',output/'controller_stats.json')
            else:recorder.write_controller_stats(stats)
    for name in ('case.json','protocol.json','provenance.json','metrics.json','eligibility.json','receipt.json','execution.npz'):
        if (source/name).exists():shutil.copy2(source/name,output/name)
    recorder.manifest()
    for event in iter_events(output):
        k,r=event['event'],event['record']
        if k=='controller_stats':continue
        digest_value((k,r),actual);reduced.add(k,r)
    result=dict(kind=kind,source=str(source),pool=[str(p) for p in pool],selection_rule=pattern+' / attempt-001',
                original=storage(source), reduced=storage(output), original_observations=original.summary(),
                reduced_observations=reduced.summary(), projected_decoded_values_match=expected.hexdigest()==actual.hexdigest(),
                motion_match=original.motion_hash.hexdigest()==reduced.motion_hash.hexdigest(),
                native_work_match=original.native_work.hexdigest()==reduced.native_work.hexdigest(),
                original_history_duplicate_excluded=kind=='tracking')
    if not result['projected_decoded_values_match']:raise AssertionError('Recorded production projection differs on readback')
    if kind!='batching':
        metrics,traces=motion_metrics(reduced,output,kind,args.source_root)
        result['independent_motion_metrics']=metrics
        saved=json.loads((source/'metrics.json').read_text())
        stats=load_controller_stats(output)
        if kind=='force':
            feedback_error=np.asarray(stats['goal_distances']); result['feedback_RMS_m']=float(np.sqrt(np.mean(feedback_error**2)))
            result['feedback_maximum_m']=float(np.max(feedback_error))
            result['original_reported_RMS_m']=saved['tracking_RMS_error_m']
            result['RMS_check_pass']=bool(np.isclose(result['feedback_RMS_m'],saved['tracking_RMS_error_m'],rtol=.01,atol=.0005))
        else:
            result['original_reported_RMS_m']=saved['tracking_RMS_m']
            result['RMS_check_pass']=bool(np.isclose(metrics['time_weighted_RMS_m'],saved['tracking_RMS_m'],rtol=.01,atol=.0005))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        t,xyz,target,error=traces
        fig,axes=plt.subplots(1,3,figsize=(12,3.5))
        axes[0].plot(target[:,1],target[:,2],label='reference');axes[0].plot(xyz[:,1],xyz[:,2],label='saved execution');axes[0].legend();axes[0].set_xlabel('Y (m)');axes[0].set_ylabel('Z (m)')
        axes[1].plot(t,error*1000);axes[1].set_xlabel('Executed time (s)');axes[1].set_ylabel('Position error (mm)')
        wall=np.asarray(reduced.solves)[:,0]*1000
        axes[2].hist(wall,bins=60);axes[2].set_xlabel('Wrapper wall solve time (ms)');axes[2].set_ylabel('All calls including startup')
        fig.tight_layout();fig.savefig(args.report_output/(kind+'.png'),dpi=160);plt.close(fig)
        np.savez_compressed(args.report_output/(kind+'-plot-inputs.npz'),t=t,xyz=xyz,target=target,error=error,wall_ms=wall)
    else:
        from analysis.f01c import load_attempt,summarize,plot
        r,c,_,missing=load_attempt(dict(attempt=str(output)),{},output.parent)
        summaries=summarize(r);result['normal_consumer']=dict(calls=len(r),collections=len(c),missing=missing,summaries=summaries)
        plot(summaries,args.report_output)
    archive=args.raw_output/(kind+'.tar.gz')
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:tar.add(output,arcname=kind)
    result['reduced_compressed_archive_bytes']=archive.stat().st_size
    result['validation_wall_s']=time.monotonic()-started
    write_json(args.report_output/(kind+'-validation.json'),result)
    print(json.dumps({k:result[k] for k in ('kind','original','reduced','reduced_compressed_archive_bytes','projected_decoded_values_match')}),flush=True)


def reproject(kind,args):
    """Apply the final production adapter to earlier derived data, preserving it."""
    source=args.reproject/kind;output=args.raw_output/(kind+'-final')
    output.mkdir(parents=True,exist_ok=False)
    cls=ForceRecorder if kind=='force' else TaskRecorder
    project=project_force_record if kind=='force' else project_task_record
    original=Observations();reduced=Observations();expected=hashlib.sha256();actual=hashlib.sha256()
    with cls(output) as recorder:
        for event in iter_validation_events(source):
            k,r=event['event'],event['record']
            if k=='controller_stats':continue
            original.add(k,r);semantic_digest_value((k,project(k,r)),expected);recorder(k,r)
            if kind=='force' and k=='solve_started':recorder.flush()
        if kind=='tracking':recorder.write_controller_stats(load_controller_stats(source))
        elif (source/'controller_stats.json').exists():shutil.copy2(source/'controller_stats.json',output/'controller_stats.json')
    for name in ('case.json','protocol.json','provenance.json','metrics.json','eligibility.json','receipt.json','execution.npz'):
        if (source/name).exists():shutil.copy2(source/name,output/name)
    # First conversion was launched before the aggregate execution copy fix.
    if kind=='tracking' and not (output/'execution.npz').exists():
        _,pattern=SELECTIONS[kind];historical=sorted((COHORT/'T01').glob(pattern))[0]/'attempt-001'
        shutil.copy2(historical/'execution.npz',output/'execution.npz')
    recorder.manifest()
    for event in iter_validation_events(output):
        k,r=event['event'],event['record']
        if k=='controller_stats':continue
        semantic_digest_value((k,r),actual);reduced.add(k,r)
    archive=args.raw_output/(kind+'-final.tar.gz')
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:tar.add(output,arcname=output.name)
    result=dict(kind=kind,source=str(source),final_output=str(output),
                contract_implementation_commit='a83bf0c',
                initial_projection_storage=storage(source),final_storage=storage(output),
                archive_gzip_level=1,final_archive_bytes=archive.stat().st_size,
                normalized_values_match=expected.hexdigest()==actual.hexdigest(),
                native_work_match=original.native_work.hexdigest()==reduced.native_work.hexdigest(),
                motion_match=original.motion_hash.hexdigest()==reduced.motion_hash.hexdigest(),
                original_observations=original.summary(),final_observations=reduced.summary(),
                adapter_sha256={name:sha(Path(__file__).resolve().parents[1]/'adapters'/name)
                    for name in ('task_recording.py','force_recording.py')})
    if not all(result[k] for k in ('normalized_values_match','native_work_match','motion_match')):
        raise AssertionError('Final representation changed retained observations or work')
    metrics,traces=motion_metrics(reduced,output,kind,args.source_root)
    result['independent_final_motion_metrics']=metrics
    stats=load_controller_stats(output);saved=json.loads((output/'metrics.json').read_text())
    if kind=='force':
        result['final_feedback_RMS_m']=float(np.sqrt(np.mean(np.asarray(stats['goal_distances'])**2)))
        result['RMS_check_pass']=bool(np.isclose(result['final_feedback_RMS_m'],saved['tracking_RMS_error_m'],rtol=.01,atol=.0005))
    else:
        result['RMS_check_pass']=bool(np.isclose(metrics['time_weighted_RMS_m'],saved['tracking_RMS_m'],rtol=.01,atol=.0005))
    write_json(args.report_output/(kind+'-final-validation.json'),result)
    print(json.dumps({k:result[k] for k in ('kind','final_storage','final_archive_bytes','normalized_values_match','native_work_match','motion_match','RMS_check_pass')}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=SELECTIONS,required=True)
    p.add_argument('--raw-output',type=Path,required=True)
    p.add_argument('--report-output',type=Path,required=True)
    p.add_argument('--source-root',type=Path,default=Path('/workspace/GATO'))
    p.add_argument('--reproject',type=Path,help='Earlier derived recordings, never original historical attempts')
    a=p.parse_args();a.raw_output.mkdir(parents=True,exist_ok=True);a.report_output.mkdir(parents=True,exist_ok=True)
    if a.reproject:reproject(a.kind,a)
    else:validate(a.kind,a)

if __name__=='__main__':main()
