#!/usr/bin/env python3
"""Saved force time/memory, correspondence and disturbance-task tables.

E01 is derived from corresponding measured E03 ticks; no extra executions.
Descriptive differences preserve accepted practical-validation policy.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recording import iter_events,json_safe,sha,write_json
from analysis.f01ab import csv_write,fingerprint


def finite(value):
    return isinstance(value,(int,float,np.number)) and np.isfinite(value)


def load(index_path):
    index_path=Path(index_path).resolve();root=index_path.parent
    index=json.loads(index_path.read_text());attempts=[];ticks=[];tasks=[];pairs=[];constructions=[];execution=[];unpaired=[]
    entries=index if isinstance(index,list) else index.get('attempts',[])
    index_metadata={} if isinstance(index,list) else index
    for entry in entries:
        if not entry.get('experiment','').startswith(('E02','E03','Force')):continue
        relative=entry.get('attempt',entry.get('path'))
        if relative is None:
            attempts.append(dict(entry,status='missing_raw'))
            continue
        raw=(root/relative).resolve()
        if not Path(relative).is_absolute() and not raw.is_relative_to(root):raise ValueError('Attempt escapes collection')
        def read(name):
            target=raw/(name+'.json');return json.loads(target.read_text()) if target.exists() else {}
        receipt=read('receipt')
        if receipt.get('kind')=='preflight':continue
        protocol,provenance,metrics=read('protocol'),read('provenance'),read('metrics')
        case=receipt.get('case',entry.get('case',{}))
        case=case if isinstance(case,dict) else {}
        p=metrics.get('parameters',protocol.get('case',case.get('parameters',{})))
        study=p.get('study',metrics.get('campaign_id'))
        if study not in ('E02','E03'):
            study=next((name for name in ('E02','E03') if entry.get('experiment','').startswith(name)),None)
        if study is None:continue
        identity=dict(attempt=str(raw),machine=provenance.get('machine') or entry.get('machine') or index_metadata.get('machine'),
            source_commit=provenance.get('source_commit',entry.get('source_commit')),module_sha256=provenance.get('module_sha256'),
            bank_manifest_sha256=provenance.get('bank_manifest_sha256'),evidence_kind=metrics.get('evidence_kind'),
            mode=p.get('mode'),truth_index=p.get('truth_index'),input_index=p.get('input_index'),experiment=entry.get('experiment'),study=study,
            force_N=p.get('force_N'),repeat=p.get('repeat'),stage=p.get('stage'),
            protocol_id=protocol.get('protocol_id') or protocol.get('protocol'),
            input_sha256=provenance.get('input_sha256'),solver_params=protocol.get('params'),
            timing_mode=p.get('timing_mode',metrics.get('timing_mode')),backend=p.get('backend',metrics.get('execution_backend')))
        identity['cohort']=fingerprint({k:identity[k] for k in ('machine','source_commit','module_sha256','bank_manifest_sha256',
            'evidence_kind','protocol_id','input_sha256','solver_params','timing_mode','backend')})
        attempts.append(dict(identity,status=entry.get('status'),outcome=metrics.get('outcome'),execution_error=metrics.get('execution_error',entry.get('error'))))
        pin=entry.get('manifest_sha256')
        if pin and sha(raw/'SHA256SUMS.json')!=pin:raise ValueError('Pinned manifest differs')
        if pin:
            for name,digest in json.loads((raw/'SHA256SUMS.json').read_text()).items():
                target=(raw/name).resolve()
                if not target.is_relative_to(raw) or sha(target)!=digest:raise ValueError('Pinned raw bytes/path differ')
        if study=='E02':
            tasks.append(dict(identity,task_success=metrics.get('task_success'),physical_pass=metrics.get('physical_pass'),
                tracking_RMS_error_m=metrics.get('tracking_RMS_error_m'),tracking_max_error_m=metrics.get('tracking_max_error_m'),
                update_above_nominal_share=metrics.get('above_nominal_share'),force_estimation_errors_SI=metrics.get('force_estimation_errors_SI'),
                execution_error=metrics.get('execution_error',entry.get('error')),status=entry.get('status'),
                maximum_sampled_effort_ratio=metrics.get('maximum_sampled_effort_ratio'),
                maximum_applied_torque_Nm=metrics.get('maximum_applied_torque_Nm'),
                maximum_selected_plan_torque_Nm=metrics.get('maximum_selected_plan_torque_Nm'),
                executed_duration_s=metrics.get('executed_duration_s'),requested_duration_s=metrics.get('requested_duration_s'),
                timing_scope=metrics.get('timing_scope'),construction=metrics.get('construction')))
        if not (raw/'events.jsonl').exists():continue
        last_step_time_s=None
        for event in iter_events(raw):
            r=event['record']
            if event['event']=='force_construction':
                constructions.append(dict(identity,**r));continue
            if event['event'] in ('step_completed','attached_force_direction'):
                if event['event']=='step_completed':last_step_time_s=r.get('time_s',0)+r.get('step_dt_s',0)
                execution.append(dict(identity,event=event['event'],time_s=r.get('time_s',last_step_time_s),step_dt_s=r.get('step_dt_s'),
                    sampled_effort_ratio=r.get('sampled_effort_ratio'),maximum_applied_abs_torque_Nm=r.get('maximum_applied_abs_torque_Nm'),
                    world_force_N=r.get('world_force_N'),world_force_direction=r.get('world_force_direction'),
                    deviation_from_world_down_deg=r.get('deviation_from_world_down_deg')))
            if event['event']!='force_tick':continue
            row=dict(identity,category=r.get('category','task'),status=r.get('status'),selected_index=r.get('selected_index'),
                     error=r.get('error'),timings_s=r.get('timings_s',{}),backend_measurements=r.get('backend_measurements',[]),
                     predictions=r.get('predictions'),errors=r.get('errors'),hypotheses=r.get('hypotheses'),
                     tick_number=r.get('tick_number',0),selected_force=r.get('selected_force'),workspace=r.get('workspace',{}),
                     prediction_observation=r.get('prediction_observation',{}),inputs=r.get('inputs'),
                     estimated_force_error_N=r.get('estimated_force_error_N'),estimated_torque_error_Nm=r.get('estimated_torque_error_Nm'),
                     selected_plan_max_abs_torque_Nm=r.get('selected_plan_max_abs_torque_Nm'),truth_direction=r.get('truth_direction'))
            for field in ('device_bytes','pinned_host_bytes'):
                row['requested_'+field]=row['workspace'].get(field)
            row['elapsed_observation_interval_s']=row['prediction_observation'].get('elapsed_interval_s')
            row['average_applied_control']=row['prediction_observation'].get('average_applied_control')
            for field,scope in (('prediction_wall_s','candidate_forward'),('TO_wrapper_wall_s','to_backend'),
                                ('selection_wall_s','selection'),('estimator_update_wall_s','estimator_update')):
                row[field]=row['timings_s'].get(scope)
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
        if row['study']=='E03' and row['category']=='measured' and row['status']=='complete':
            grouped[(row['cohort'],row['input_index'],row['truth_index'])].setdefault(row['mode'],[]).append(row)
    for (cohort,input_index,truth),methods in grouped.items():
        # Duplicate completed attempts remain visible; do not pick a convenient repeat.
        if any(len({r['attempt'] for r in methods.get(mode,[])})!=1 for mode in ('original','adapted')):
            unpaired.append(dict(cohort=cohort,input_index=input_index,truth_index=truth,reason='missing method or duplicate completed attempts'));continue
        by_tick=defaultdict(dict)
        for mode,rows in methods.items():
            for row in rows:by_tick[row['tick_number']].setdefault(mode,[]).append(row)
        for tick_number,rows in sorted(by_tick.items()):
            if any(len(rows.get(mode,[]))!=1 for mode in ('original','adapted')):
                unpaired.append(dict(cohort=cohort,input_index=input_index,truth_index=truth,tick_number=tick_number,reason='missing or duplicate tick'));continue
            a,b=rows['original'][0],rows['adapted'][0]
            if not np.array_equal(a['hypotheses'],b['hypotheses']):
                unpaired.append(dict(cohort=cohort,input_index=input_index,truth_index=truth,tick_number=tick_number,reason='hypotheses differ'));continue
            # Compare the measured transition, not method-dependent optimizer seeds.
            if a['inputs'] is not None or b['inputs'] is not None:
                keys=('previous_time_s','observed_time_s','previous_state','observed_state','applied_history')
                if fingerprint(json_safe({k:(a['inputs'] or {}).get(k) for k in keys}))!=fingerprint(json_safe({k:(b['inputs'] or {}).get(k) for k in keys})):
                    unpaired.append(dict(cohort=cohort,input_index=input_index,truth_index=truth,tick_number=tick_number,reason='observations or applied controls differ'));continue
            delta=float(np.max(np.abs(np.asarray(a['predictions'])-np.asarray(b['predictions']))))
            pairs.append(dict(cohort=cohort,input_index=input_index,truth_index=truth,tick_number=tick_number,machine=a['machine'],
                original_attempt=a['attempt'],adapted_attempt=b['attempt'],
                E01_correspondence=a['protocol_id']=='force-persistent-grid-native-v2' or input_index==0,
                predictions_max_abs_difference=delta,selected_indices_equal=a['selected_index']==b['selected_index'],
                original_wall_s=a['algorithm_wall_s'],adapted_wall_s=b['algorithm_wall_s'],
                original_over_adapted_wall=a['algorithm_wall_s']/b['algorithm_wall_s']
                    if finite(a['algorithm_wall_s']) and finite(b['algorithm_wall_s']) and b['algorithm_wall_s']>0 else None,
                original_native_s=a['native_solve_s'],adapted_native_s=b['native_solve_s'],
                original_requested_device_bytes=a['requested_device_bytes'],adapted_requested_device_bytes=b['requested_device_bytes'],
                original_requested_pinned_host_bytes=a['requested_pinned_host_bytes'],adapted_requested_pinned_host_bytes=b['requested_pinned_host_bytes']))
    baselines=defaultdict(list)
    for task in tasks:
        if task['force_N']==0 and not task['execution_error'] and task['status']=='completed':
            baselines[(task['cohort'],task['mode'],task['stage'])].append(task)
    for task in tasks:
        matches=baselines[(task['cohort'],task['mode'],task['stage'])]
        task['zero_baseline_attempt']=matches[0]['attempt'] if len(matches)==1 else None
        task['RMS_above_own_zero_m']=(task['tracking_RMS_error_m']-matches[0]['tracking_RMS_error_m']
            if len(matches)==1 and finite(task['tracking_RMS_error_m']) and finite(matches[0]['tracking_RMS_error_m']) else None)
        task['zero_baseline_note']='unique completed same-method/cohort baseline' if len(matches)==1 else 'missing or duplicate completed zero baselines; no selection'
    return dict(index_sha256=sha(index_path),attempts=attempts,ticks=ticks,tasks=tasks,pairs=pairs,
        constructions=constructions,execution=execution,unpaired=unpaired)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--index',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();loaded=load(args.index);output=args.output.resolve()
    if any(output.is_relative_to(Path(r['attempt']).resolve()) for r in loaded['attempts'] if r.get('attempt')):raise ValueError('Derived output inside raw')
    output.mkdir(parents=True,exist_ok=False)
    for name in ('attempts','ticks','tasks','pairs','constructions','execution','unpaired'):
        rows=[{k:v for k,v in row.items() if k not in ('predictions','errors','hypotheses','inputs')} for row in loaded[name]]
        rows=[{k:json.dumps(json_safe(v),sort_keys=True) if isinstance(v,(dict,list,tuple,np.ndarray)) else v
               for k,v in row.items()} for row in rows]
        csv_write(output/(name+'.csv'),rows)
    write_json(output/'summary.json',dict(index_sha256=loaded['index_sha256'],attempts=len(loaded['attempts']),ticks=len(loaded['ticks']),
        tasks=len(loaded['tasks']),pairs=len(loaded['pairs']),E01_pairs=sum(p['E01_correspondence'] for p in loaded['pairs']),
        limitations='Protocols/cohorts remain separate. Revised force runtime uses persistent owners and GPU prediction; construction/memory queries are outside algorithm ticks. Requested device/pinned allocations include model/simulation buffers; observed device availability is a coarse cross-check, not continuous peak. Legacy fresh-owner/CPU-predictor data retain their original timing scope. E01 pairs matched observations/hypotheses at corresponding tick numbers, not optimized plans; duplicate completed attempts are not selected. E02 differences use each method\'s unique completed same-cohort zero baseline; unavailable or duplicate baselines remain explicit. Attached body-fixed force turns with the joint; failures retained, no equal-quality acceleration or constant-world-force claim.'))
    return 0


if __name__=='__main__':raise SystemExit(main())
