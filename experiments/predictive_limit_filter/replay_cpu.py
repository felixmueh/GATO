#!/usr/bin/env python3
"""Saved Jetson finite-unsafe samples, evaluated by the CPU filter only.

These are 30ms constant-control counterfactual forecasts, not closed-loop rescues.
"""
import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pinocchio as pin

SOURCE=Path(__file__).resolve().parents[2]
RAW=Path('/workspace/GATO/_expr_results/random-reaching-jetson-fresh100-20261015/jetson-import/random-reaching-jetson-fresh100-20261015')
OUT=Path('/workspace/GATO/_expr_results/random-reaching-predictive-filter-20261015/phase1-cpu')

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=OUT)
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    impl=load('predictive_filter',Path(__file__).with_name('filter.py'))
    common=load('frozen_common',SOURCE/'python/bsqp/common.py')
    model_path=SOURCE/'gato/dynamics/tiago_right/tiago_right_arm.urdf'
    model=pin.buildModelFromUrdf(str(model_path));model.gravity.linear=np.array([0.,0.,-9.81])
    filt=impl.PredictiveLimitFilter(model,common.rk4)
    functional=subprocess.run([sys.executable,str(Path(__file__).with_name('check_cpu.py'))],capture_output=True,text=True,check=True)
    (out/'functional_checks.json').write_text(functional.stdout)
    rows=[];full=[];excluded=[];summary_hashes={}
    for arm,expected in [('J1',16),('J2',13)]:
        summary_path=RAW/arm/'pilot/summary.json';summaries=json.loads(summary_path.read_text());summary_hashes[arm]=sha(summary_path)
        assert len(summaries)==100
        eligible=[r for r in summaries if not r['execution_error'] and r['complete'] and r['physical']['finite'] and not r['physical_pass']]
        assert len(eligible)==expected
        for r in summaries:
            if r['execution_error']:
                excluded.append(dict(arm=arm,pair_id=r['pair_id'],reason='Original execution error; prefix evidence not a complete finite unsafe episode',failure_phase=r.get('failure_phase')))
        for row in eligible:
            pair=row['pair_id'];case=RAW/arm/'pilot'/f'pair_{pair:05d}';executed=case/'executed.npz'
            with np.load(executed,allow_pickle=False) as z:
                x=z['states'].copy();u=z['controls'].copy();t=z['times'].copy();ut=z['control_times'].copy()
            assert len(u)==len(x)-1 and np.isfinite(x).all() and np.isfinite(u).all()
            q,v=x[:,:7],x[:,7:]
            state_guard=np.all((q>=filt.qlow)&(q<=filt.qhigh)&(np.abs(v)<=filt.vguard),axis=1)
            u_guard=np.all(np.abs(u)<=filt.uguard,axis=1)
            state_raw=np.all((q>=filt.qlo-1e-6)&(q<=filt.qhi+1e-6)&(np.abs(v)<=filt.vmax*(1+1e-6)),axis=1)
            u_raw=np.all(np.abs(u)<=filt.umax*(1+1e-6),axis=1)
            def preceding(bad_state,bad_control):
                choices=[]
                if np.any(bad_state):
                    j=int(np.flatnonzero(bad_state)[0]);choices.append((float(t[j]),max(0,j-1),'state'))
                if np.any(bad_control):
                    j=int(np.flatnonzero(bad_control)[0]);choices.append((float(ut[j]),j,'control'))
                if not choices:raise ValueError('Recorded unsafe episode has no aligned violation')
                return min(choices,key=lambda a:(a[0],a[1]))
            for label,(event_time,index,event_type) in [
                    ('before_first_guard_entry',preceding(~state_guard,~u_guard)),
                    ('before_first_raw_violation',preceding(~state_raw,~u_raw))]:
                result=filt.filter(x[index],u[index])
                chosen=result.get('selected_forecast',{})
                original=next((c.get('forecast',{}) for c in result['candidates'] if c['name']=='proposed'),{})
                record=dict(arm=arm,pair_id=pair,anchor=label,actual_sample_index=index,actual_time_s=float(t[index]),
                    first_event_time_s=event_time,first_event_kind=event_type,original_applied_segment_s=float(t[index+1]-t[index]),
                    initial_physical_pass=result['initial']['physical_state_pass'],initial_guard_pass=result['initial']['guard_state_pass'],
                    status=result['status'],guard_certified=result['guard_certified'],modified=result.get('modified'),
                    selected_candidate=result['selected_candidate'],proposed_30ms_max_speed_ratio=original.get('maximum_speed_ratio'),
                    proposed_30ms_q_clearance_rad=original.get('minimum_physical_q_clearance_rad'),
                    filtered_30ms_max_speed_ratio=chosen.get('maximum_speed_ratio'),filtered_30ms_q_clearance_rad=chosen.get('minimum_physical_q_clearance_rad'),
                    proposed_effort_ratio=float(np.max(np.abs(u[index])/model.effortLimit)),filtered_effort_ratio=chosen.get('maximum_effort_ratio'),
                    filter_wall_ms=1000*result['filter_wall_s'],wall_to_3ms_ratio=result['wall_to_nominal_step_ratio'],nominal_3ms_overrun=result['nominal_step_overrun'],
                    original_task_success=row['task_success'],original_physical_pass=row['physical_pass'],executed_sha256=sha(executed))
                rows.append(record);full.append(dict(**record,actual_state=x[index].tolist(),proposed_torque=u[index].tolist(),filter_result=result))
    with (out/'replay_samples.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(out/'replay_details.json',full);write(out/'excluded_error_episodes.json',excluded)
    counts=[]
    for arm in ['J1','J2']:
        for anchor in ['before_first_guard_entry','before_first_raw_violation']:
            rr=[r for r in rows if r['arm']==arm and r['anchor']==anchor]
            counts.append(dict(arm=arm,anchor=anchor,samples=len(rr),statuses=dict(Counter(r['status'] for r in rr)),
                modified=sum(r['modified'] is True for r in rr),initial_guard_pass=sum(r['initial_guard_pass'] for r in rr),
                median_filter_wall_ms=float(np.median([r['filter_wall_ms'] for r in rr])),p95_filter_wall_ms=float(np.percentile([r['filter_wall_ms'] for r in rr],95)),
                max_filter_wall_ms=max(r['filter_wall_ms'] for r in rr),over_nominal3ms=sum(r['nominal_3ms_overrun'] for r in rr)))
    write(out/'counts.json',counts)
    write(out/'metadata.json',dict(status='CPU_FUNCTIONAL_ONLY_NO_NATIVE',source=str(SOURCE),
        source_commit=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip(),
        native_source_base='85ef2ea5cf5b96726be6f4b5441f4a7c0ca041b4',Jetson_frontend_base='72533b237a55732aaa11b7747f204eee0ff16725',
        source_sha256={str(p.relative_to(SOURCE)):sha(p) for p in [model_path,SOURCE/'python/bsqp/common.py',Path(__file__),Path(__file__).with_name('filter.py'),Path(__file__).with_name('check_cpu.py')]},
        original_summary_sha256=summary_hashes,filter_settings=impl.DEFAULTS,forecast='Constant proposed/filtered torque for30ms, sampledRK4max3ms; source actual command was held for only its recorded <=3ms segment',
        selection='All16J1+13J2 finite error-free unsafe episodes, two anchors per episode; original errors excluded separately, never credited as passes',
        timing='Desktop CPU measurements only.3ms overruns descriptive, no acceptance gate. No native/GPU, concurrent scheduling or Jetson feasibility measurement.',
        no_controller_rollout=True,no_physical_safety_guarantee=True))
    print(json.dumps(counts,indent=2))

if __name__=='__main__':main()
