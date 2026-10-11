#!/usr/bin/env python3
"""Saved resource/response/profile/sustained tables and original timing views.

Accept historical selection indexes and current collection indexes. Derived
figures keep machines, sources, adapters and evidence cohorts separate.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recording import iter_events, sha, json_safe
from analysis.f01ab import csv_write as write_csv, read as json_read, fingerprint
from adapters.resource_profiles_common import validate_profile


def load(index_path, pin=None):
    index_path=Path(index_path).resolve();root=index_path.parent
    if pin and sha(index_path)!=pin:raise ValueError('Selection index hash differs')
    index=json_read(index_path);attempts=[];calls=[];stages=[];resources=[]
    for entry in index.get('attempts',[]):
        path=entry.get('attempt',entry.get('path'))
        if path is None:
            attempts.append(dict(entry.get('case',{}),attempt_path=None,status='missing'));continue
        if entry.get('experiment') and not entry['experiment'].startswith(('F04','F05')):continue
        raw=(root/path).resolve()
        if not Path(path).is_absolute() and not raw.is_relative_to(root):raise ValueError('Attempt path escapes collection')
        if (raw/'resource-profile-raw').is_dir():raw=raw/'resource-profile-raw'
        protocol,provenance,metrics=(json_read(raw/(name+'.json')) for name in ('protocol','provenance','metrics'))
        case=metrics.get('case',entry.get('case',protocol))
        if case.get('study') not in ('F04','F05'):continue
        if protocol.get('timing_mode')!='A' or protocol.get('backend')!='internal':raise ValueError('Resource timing mode/backend differs')
        manifest_pin=entry.get('manifest_sha256')
        if manifest_pin:
            if sha(raw/'SHA256SUMS.json')!=manifest_pin:raise ValueError('Pinned attempt manifest differs')
            for relative,digest in json_read(raw/'SHA256SUMS.json').items():
                target=(raw/relative).resolve()
                if not target.is_relative_to(raw) or sha(target)!=digest:raise ValueError('Pinned raw bytes/path differ')
        expected=entry.get('case',case)
        if any(protocol.get(k)!=v for k,v in expected.items()):raise ValueError('Selected case differs')
        if provenance.get('module',{}).get('flags',{}).get('GATO_EXPERIMENT_STAGE_TIMING') != case.get('stage_timing'):
            raise ValueError('Selected stage build differs')
        identity=dict(case,attempt_path=str(raw),machine=provenance.get('machine',metrics.get('machine',entry.get('machine'))),
            source_commit=provenance.get('source_commit',entry.get('source_commit')),
            evidence_kind=metrics.get('evidence_kind',provenance.get('evidence_kind')),
            adapter_identity=fingerprint(provenance.get('adapter_sha256','historical-original')),
            cohort=protocol.get('cohort_id','historical-original'))
        if not (raw/'events.jsonl').is_file():
            attempts.append(dict(identity,status='no_raw',metrics=metrics,protocol=protocol,provenance=provenance));continue
        grouped={};current=None;query_payload=None
        for event in iter_events(raw):
            record,kind=event['record'],event['event']
            if kind=='query_returned':
                query_payload=json_safe(record['payload'])
                resources.extend(dict(identity,**r) for r in query_payload['kernels'])
            if 'slot' in record:
                current=record['slot']
                row=grouped.setdefault(current,dict(identity,slot=current,category=record['category'],status='started',native_ms=None,wrapper_ms=None))
                if kind in ('solve_returned','solve_completed','solve_failed','call_finished'):
                    for key,target,scale in (('native_solve_time_ms','native_ms',1.),('wall_solve_time_s','wrapper_ms',1000.)):
                        if key in record:
                            value=record[key]
                            row[target]=float(value)*scale if np.ndim(value)==0 and value is not None and np.isfinite(value) and value>0 else None
                if kind in ('solve_failed','execution_failed'):row.update(status='failed',error=record.get('error'))
                elif kind=='call_finished':row['status']='completed'
            if kind=='native_return' and current is not None:
                reply=record['native_reply'];value=reply.get('sqp_time_us')
                if value is not None and np.ndim(value)==0 and np.isfinite(value) and value>0:grouped[current]['native_ms']=float(value)/1000.
                profile=json_safe(reply.get('experiment_profile_stages'))
                if profile is not None:
                    if not case['stage_timing']:raise ValueError('Unexpected profile in OFF timing arm')
                    try:validate_profile(profile,case['SQP'])
                    except ValueError as error:
                        stages.append(dict(identity,slot=current,category=grouped[current]['category'],stage='invalid_profile',elapsed_us=None,unavailable_reason=str(error)))
                    else:stages.extend(dict(identity,slot=current,category=grouped[current]['category'],**window) for window in profile['windows'])
                elif case['stage_timing']:
                    stages.append(dict(identity,slot=current,category=grouped[current]['category'],stage='profile',elapsed_us=None,unavailable_reason='profile_missing'))
        calls.extend(grouped.values())
        attempts.append(dict(identity,protocol=protocol,provenance=provenance,metrics=metrics,resources=query_payload,
                             manifest_sha256=manifest_pin,status=metrics.get('status',entry.get('status'))))
    return dict(index=index,index_sha256=sha(index_path),attempts=attempts,calls=calls,stages=stages,resources=resources)


def render(loaded, output):
    output = Path(output).resolve()
    if any(entry['path'] is not None and output.is_relative_to(Path(entry['path']).resolve()) for entry in loaded['index']['attempts']):
        raise ValueError('Derived output in immutable raw')
    output.mkdir(parents=True, exist_ok=False)
    for name in ('calls', 'stages', 'resources', 'attempts'):
        write_csv(output / (name + '.csv'), loaded[name])
    (output / 'provenance.json').write_text(json.dumps(dict(loaded, plot_sha256=sha(__file__),
        pcg_times_us=None, pcg_times_us_reason='UNAVAILABLE: native literal-zero placeholder; never a measured stage',
        limitations='Stage ON overhead separate; stream-elapsed windows include idle gaps. Short loop is not thermal steady-state. No actuation/task-quality claim.'), indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    index = loaded['index']; label = 'SYNTHETIC TEST DATA — NOT MEASUREMENT' if index['evidence_kind'] == 'synthetic_test' else 'F04/F05 saved measurements'
    timing_present = any(r['study'] == 'F05' for r in loaded['attempts'])
    for scope in (('native', 'wrapper') if timing_present else ()):
        for reciprocal in (False, True):
            fig, axes = plt.subplots(1, 2, figsize=(13, 6))
            for axis, arm in zip(axes, ('first_response', 'sustained')):
                selected = [r for r in loaded['calls'] if r['arm'] == arm and r['category'] == 'measured']
                for B in (1, 16):
                    for S, P in ((1, 100), (5, 100), (10, 800)):
                        series = [r for r in selected if (r['B'], r['SQP'], r['PCG']) == (B, S, P)]
                        values = [r[scope + '_ms'] for r in series]
                        axis.plot(range(len(values)), [np.nan if v is None else 1000/v if reciprocal else v for v in values], 'o', markersize=3, label=f'B{B} S{S}/P{P}, n={len(values)}')
                axis.set_title(arm.replace('_', ' ') + ' / stage OFF')
                axis.set_xlabel('recorded call index (each series)')
                axis.set_ylabel('1/solve time (calls/s)' if reciprocal else 'solve time (ms)'); axis.legend(fontsize=7); axis.grid(alpha=.2)
            durations = [r.get('metrics', {}).get('sustained_wall_s') for r in loaded['attempts'] if r['arm'] == 'sustained']
            fig.suptitle(f"{label}\n{index['machine']} / A / internal / N32 / {scope} / source {index['source_commit']}\nactual saved sustained duration(s): {durations}\nSolve-only reciprocals; constructor/reset/record loop duration separately retained", fontsize=9)
            fig.tight_layout(rect=(0, 0, 1, .82)); fig.savefig(output / f"{scope}-{'reciprocal' if reciprocal else 'solve-time'}.png", dpi=130); plt.close(fig)
    if loaded['stages']:
        measured = [r for r in loaded['stages'] if r['category'] == 'measured']
        grouped = defaultdict(list)
        for row in measured:
            grouped[(row['B'], row['SQP'], row['PCG'], row['stage'])].append(row)
        stage_summary = []
        for (B, S, P, name), windows in grouped.items():
            values = [r['elapsed_us'] for r in windows if r['elapsed_us'] is not None]
            stage_summary.append(dict(B=B, SQP=S, PCG=P, stage=name, reached=len(values),
                unavailable=len(windows)-len(values),
                mean_stream_elapsed_us=float(np.mean(values)) if values else None))
        write_csv(output / 'stage_summary.csv', stage_summary)
        names = list(dict.fromkeys(r['stage'] for r in measured))
        fig, axis = plt.subplots(figsize=(13, 6))
        for B, S, P in dict.fromkeys((r['B'], r['SQP'], r['PCG']) for r in measured):
            series = [r for r in stage_summary if (r['B'], r['SQP'], r['PCG']) == (B, S, P)]
            axis.plot([names.index(r['stage']) for r in series],
                [np.nan if r['mean_stream_elapsed_us'] is None else r['mean_stream_elapsed_us'] for r in series],
                'o', markersize=4, label=f'B{B} S{S}/P{P}')
            for r in series:
                axis.annotate(f"{r['reached']}/{r['reached']+r['unavailable']}",
                    (names.index(r['stage']), r['mean_stream_elapsed_us'] or 0), fontsize=6)
        axis.set_xticks(range(len(names)), names, rotation=35, ha='right', fontsize=8)
        axis.legend(fontsize=8)
        axis.set_ylabel('mean reached window stream elapsed (µs), includes idle gaps'); axis.set_title(f"{label}\n{index['machine']} / A / internal / N32 / source {index['source_commit']}\nStage ON separate cohort; reached/all window counts; null/skipped retained; pcg_times_us UNAVAILABLE", fontsize=9)
        fig.tight_layout(); fig.savefig(output / 'stage-windows.png', dpi=130); plt.close(fig)
    return loaded



def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',type=Path,required=True);parser.add_argument('--index-sha256')
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    loaded=load(args.index,args.index_sha256);groups=defaultdict(list)
    # Stage ON/OFF remain named arms within a scientific cohort. No retry,
    # machine, source, evidence or adapter grouping crosses this key.
    for attempt in loaded['attempts']:
        key=tuple(attempt.get(k) for k in ('machine','source_commit','evidence_kind','adapter_identity','cohort'))
        groups[key].append(attempt)
    for i,(key,attempts) in enumerate(groups.items()):
        paths={a['attempt_path'] for a in attempts}
        subset={name:[r for r in loaded[name] if r['attempt_path'] in paths] for name in ('calls','stages','resources')}
        subset.update(attempts=attempts,index=dict(machine=key[0],source_commit=key[1],evidence_kind=key[2],
            attempts=[dict(path=p) for p in paths]),index_sha256=loaded['index_sha256'])
        render(subset,args.output/f'cohort-{i:03d}')
    print(json.dumps(dict(cohorts=len(groups),attempts=len(loaded['attempts']),calls=len(loaded['calls']),
                         stages=len(loaded['stages']),resources=len(loaded['resources']))))


if __name__=='__main__':main()
