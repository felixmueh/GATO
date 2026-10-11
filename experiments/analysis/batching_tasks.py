#!/usr/bin/env python3
"""Export saved BA/BB task outcomes per attempt, without legacy manager metadata."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collect import records
from analysis.f01ab import csv_write,read
from recording import write_json,sha


def analyze(raw,output):
    raw,output=Path(raw).resolve(),Path(output).resolve()
    if output.is_relative_to(raw):raise ValueError('Derived output must be outside raw')
    output.mkdir(parents=True,exist_ok=False);rows=[]
    for entry in records(raw):
        if entry.get('experiment') not in ('BA','BB') or entry['status']=='running':continue
        path=raw/entry['attempt'];receipt=read(path/'receipt.json')
        if receipt.get('kind')=='preflight':continue
        m=entry['metrics'];p=read(path/'provenance.json')
        params=receipt.get('case',{}).get('parameters',m.get('parameters',{}));bounds=m.get('physical',{})
        row={k:entry.get(k) for k in ('attempt','machine','source_commit','protocol','timing_mode','execution_backend','status')}
        row['timing_mode']=row['timing_mode'] or receipt.get('timing_mode',receipt.get('identity',{}).get('timing_mode'))
        row.update(study=entry['experiment'],pair_id=params.get('pair_id',m.get('pair_id')),B=params.get('batch',m.get('batch')),
            arm=params.get('arm','default'),cohort=m.get('cohort_id',p.get('cohort_id')),module_sha256=p.get('module_sha256'),
            protocol_sha256=p.get('protocol_sha256',p.get('frozen_protocol_sha256')),outcome=m.get('outcome'),
            execution_error=m.get('execution_error',entry.get('error')),task_success=m.get('task_success'),physical_pass=m.get('physical_pass'),
            reached_seconds=m.get('reached_seconds'),simulated_seconds=m.get('simulated_seconds'),
            controller_failure=m.get('controller_failure',m.get('failure')),selected_nonzero_lanes=m.get('selected_nonzero_lanes'),
            accepted_sqp_steps_selected=m.get('accepted_sqp_steps_selected'),solve_wall_seconds=m.get('solve_wall_seconds'),
            elapsed_wall_seconds=m.get('elapsed_wall_seconds'),above_nominal_share=m.get('above_nominal_share'),
            max_joint_bound_excess_rad=bounds.get('max_joint_bound_excess_rad'),max_velocity_ratio=bounds.get('max_velocity_ratio'),max_effort_ratio=bounds.get('max_effort_ratio'))
        rows.append(row)
    csv_write(output/'attempts.csv',rows)
    groups=defaultdict(list)
    for row in rows:
        key=tuple(row[k] for k in ('study','machine','source_commit','protocol','timing_mode','execution_backend','cohort','module_sha256','arm','B'))
        groups[key].append(row)
    counts=[]
    for selected in groups.values():
        first=selected[0];counts.append(dict({k:first[k] for k in ('study','machine','source_commit','protocol','timing_mode','execution_backend','cohort','module_sha256','arm','B')},
            attempts=len(selected),receipt_statuses=dict(Counter(r['status'] for r in selected)),
            task_pass=sum(r['task_success'] is True for r in selected),task_fail=sum(r['task_success'] is False for r in selected),task_unavailable=sum(r['task_success'] is None for r in selected),
            physical_pass=sum(r['physical_pass'] is True for r in selected),physical_fail=sum(r['physical_pass'] is False for r in selected),physical_unavailable=sum(r['physical_pass'] is None for r in selected)))
    csv_write(output/'attempt_counts.csv',counts)
    write_json(output/'summary.json',dict(raw=str(raw),attempts=rows,counts=counts,analysis_sha256=sha(__file__),
        scope='Terminal saved BA/BB attempts only. Every retry remains a distinct row; counts are attempts, not independent task success rates. Current continuation does not replace historical23completed+1failed BB attempts or retained BA baseline. Source/protocol/mode/module/arm cohorts separated. No solver calls or tighter task/physical gates.'))
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(len(analyze(a.raw,a.output)),'saved attempts')
