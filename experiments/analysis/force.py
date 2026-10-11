#!/usr/bin/env python3
"""Saved force time/memory, correspondence and disturbance-task tables.

E01 is derived from first-input measured E03 records; no extra executions.
Descriptive differences preserve accepted practical-validation policy.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recording import iter_events,sha,write_json
from analysis.f01ab import csv_write,fingerprint


def load(index_path):
    index_path=Path(index_path).resolve();root=index_path.parent
    index=json.loads(index_path.read_text());attempts=[];ticks=[];tasks=[];pairs=[]
    entries=index if isinstance(index,list) else index.get('attempts',[])
    index_metadata={} if isinstance(index,list) else index
    for entry in entries:
        if not entry.get('experiment','').startswith(('E02','E03')):continue
        relative=entry.get('attempt',entry.get('path'))
        if relative is None:attempts.append(dict(entry,status='missing_raw'));continue
        raw=(root/relative).resolve()
        if not Path(relative).is_absolute() and not raw.is_relative_to(root):raise ValueError('Attempt escapes collection')
        def read(name):
            target=raw/(name+'.json');return json.loads(target.read_text()) if target.exists() else {}
        protocol,provenance,metrics=read('protocol'),read('provenance'),read('metrics')
        case=read('receipt').get('case',entry.get('case',{}))
        case=case if isinstance(case,dict) else {}
        p=metrics.get('parameters',protocol.get('case',case.get('parameters',{})))
        identity=dict(attempt=str(raw),machine=provenance.get('machine') or entry.get('machine') or index_metadata.get('machine'),
            source_commit=provenance.get('source_commit',entry.get('source_commit')),module_sha256=provenance.get('module_sha256'),
            bank_manifest_sha256=provenance.get('bank_manifest_sha256'),evidence_kind=metrics.get('evidence_kind'),
            mode=p.get('mode'),truth_index=p.get('truth_index'),input_index=p.get('input_index'),experiment=entry.get('experiment'))
        identity['cohort']=fingerprint({k:identity[k] for k in ('machine','source_commit','module_sha256','bank_manifest_sha256','evidence_kind')})
        attempts.append(dict(identity,status=entry.get('status'),outcome=metrics.get('outcome'),execution_error=metrics.get('execution_error',entry.get('error'))))
        pin=entry.get('manifest_sha256')
        if pin and sha(raw/'SHA256SUMS.json')!=pin:raise ValueError('Pinned manifest differs')
        if pin:
            for name,digest in json.loads((raw/'SHA256SUMS.json').read_text()).items():
                target=(raw/name).resolve()
                if not target.is_relative_to(raw) or sha(target)!=digest:raise ValueError('Pinned raw bytes/path differ')
        if entry['experiment'].startswith('E02'):
            tasks.append(dict(identity,task_success=metrics.get('task_success'),physical_pass=metrics.get('physical_pass'),
                tracking_RMS_error_m=metrics.get('tracking_RMS_error_m'),tracking_max_error_m=metrics.get('tracking_max_error_m'),
                update_above_nominal_share=metrics.get('above_nominal_share'),force_estimation_errors_SI=metrics.get('force_estimation_errors_SI'),
                execution_error=metrics.get('execution_error',entry.get('error'))))
        if not (raw/'events.jsonl').exists():continue
        for event in iter_events(raw):
            r=event['record']
            if event['event']!='force_tick':continue
            row=dict(identity,category=r.get('category','task'),status=r.get('status'),selected_index=r.get('selected_index'),
                     error=r.get('error'),timings_s=r.get('timings_s',{}),backend_measurements=r.get('backend_measurements',[]),
                     predictions=r.get('predictions'),errors=r.get('errors'),hypotheses=r.get('hypotheses'))
            measurements=row['backend_measurements'];native=[m.get('native_solve_us') for m in measurements]
            row['native_solve_s']=sum(native)/1e6 if native and all(x is not None and np.isfinite(x) for x in native) else None
            total=row['timings_s'].get('total');row['algorithm_wall_s']=total
            row['native_reciprocal_Hz']=1/row['native_solve_s'] if row['native_solve_s'] and row['native_solve_s']>0 else None
            row['algorithm_reciprocal_Hz']=1/total if total and total>0 else None
            memory=[m.get('memory') for m in measurements if m.get('memory')]
            row['observed_additional_device_bytes']=max((m.get('maximum_observed_additional_device_bytes',0) for m in memory),default=None)
            row['memory_errors']=[m.get('memory_error') for m in measurements if m.get('memory_error')]
            ticks.append(row)
    grouped=defaultdict(dict)
    for row in ticks:
        if row['experiment']=='E03' and row['category']=='measured' and row['status']=='complete':
            grouped[(row['cohort'],row['input_index'],row['truth_index'])].setdefault(row['mode'],[]).append(row)
    for (cohort,input_index,truth),methods in grouped.items():
        # Duplicate completed attempts remain visible; do not pick a convenient repeat.
        if any(len(methods.get(mode,[]))!=1 for mode in ('original','adapted')):continue
        a,b=methods['original'][0],methods['adapted'][0]
        if not np.array_equal(a['hypotheses'],b['hypotheses']):continue
        delta=float(np.max(np.abs(np.asarray(a['predictions'])-np.asarray(b['predictions']))))
        pairs.append(dict(cohort=cohort,input_index=input_index,truth_index=truth,machine=a['machine'],
            original_attempt=a['attempt'],adapted_attempt=b['attempt'],E01_correspondence=input_index==0,
            predictions_max_abs_difference=delta,selected_indices_equal=a['selected_index']==b['selected_index'],
            original_wall_s=a['algorithm_wall_s'],adapted_wall_s=b['algorithm_wall_s'],
            original_over_adapted_wall=a['algorithm_wall_s']/b['algorithm_wall_s'] if b['algorithm_wall_s'] else None,
            original_native_s=a['native_solve_s'],adapted_native_s=b['native_solve_s']))
    return dict(index_sha256=sha(index_path),attempts=attempts,ticks=ticks,tasks=tasks,pairs=pairs)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--index',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();loaded=load(args.index);output=args.output.resolve()
    if any(output.is_relative_to(Path(r['attempt']).resolve()) for r in loaded['attempts'] if r.get('attempt')):raise ValueError('Derived output inside raw')
    output.mkdir(parents=True,exist_ok=False)
    for name in ('attempts','ticks','tasks','pairs'):
        rows=[{k:v for k,v in row.items() if k not in ('predictions','errors','hypotheses')} for row in loaded[name]]
        csv_write(output/(name+'.csv'),rows)
    write_json(output/'summary.json',dict(index_sha256=loaded['index_sha256'],attempts=len(loaded['attempts']),ticks=len(loaded['ticks']),
        tasks=len(loaded['tasks']),pairs=len(loaded['pairs']),E01_pairs=sum(p['E01_correspondence'] for p in loaded['pairs']),
        limitations='Body-fixed local force; full algorithm includes fresh owner/model/memory queries. Observed boundary device delta is not continuous peak. Failed attempts retained; no equal-quality acceleration or constant-world-force claim. Practical correspondence descriptive, no additional tight gate.'))
    return 0


if __name__=='__main__':raise SystemExit(main())
