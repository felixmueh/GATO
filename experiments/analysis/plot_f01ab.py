"""Integrity-pinned, CPU-only F01ab saved-event analysis; never executes a solve."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

MACHINES=('gtx1070','rtx5060','jetson')
CELLS=tuple(itertools.product((16,32,64,128),(1,2,5,10),(100,200,400,800)))
CATEGORIES=('cold','warm','goal_switch')
COLORS={'cold':'#0072B2','warm':'#009E73','goal_switch':'#D55E00'}
EXPECTED={'cold':12,'warm':222,'goal_switch':6,'process_cuda_warmup':1}
SCOPES=('native_ms','wrapper_ms')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_read(path):
    def bad(value):raise ValueError('Nonfinite literal in JSON: '+value)
    return json.loads(Path(path).read_text(),parse_constant=bad)


def checked_path(root,relative):
    relative=Path(relative)
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe relative input path')
    path=(root/relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():raise ValueError('Missing/escaping input file')
    return path


def verify_manifest(attempt,expected):
    path=checked_path(attempt,'SHA256SUMS.json')
    if sha(path)!=expected:raise ValueError('Independent attempt manifest hash mismatch')
    manifest=json_read(path)
    if not isinstance(manifest,dict):raise ValueError('Invalid attempt manifest')
    for name,digest in manifest.items():
        target=checked_path(attempt,name)
        if not isinstance(digest,str) or sha(target)!=digest:raise ValueError('Attempt file bytes differ: '+name)
    if not {'protocol.json','metrics.json','provenance.json'}.issubset(manifest):raise ValueError('Missing attempt identities')
    return manifest


def scalar(record,name,arrays):
    value=record.get(name)
    if isinstance(value,dict) and 'npz_key' in value:
        if value['npz_key'] not in arrays or arrays[value['npz_key']].shape!=():raise ValueError('Invalid timing scalar')
        value=arrays[value['npz_key']].item()
    if value is None:return None,'missing'
    if isinstance(value,bool) or not isinstance(value,(int,float)):raise ValueError('Invalid timing type')
    if not np.isfinite(value):return None,'nonfinite'
    if value<=0:return None,'nonpositive'
    return float(value),'finite'


def row_base(entry):
    return {key:entry[key] for key in ('machine','N','SQP','PCG','source_commit','cohort_id','attempt_id')}


def load_attempt(entry,root,schedule,index):
    relative=Path(entry['path'])
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe attempt path')
    attempt=(root/relative).resolve()
    if not attempt.is_relative_to(root.resolve()) or not attempt.is_dir():raise ValueError('Attempt path escape')
    manifest=verify_manifest(attempt,entry['manifest_sha256'])
    protocol=json_read(attempt/'protocol.json');provenance=json_read(attempt/'provenance.json');metrics=json_read(attempt/'metrics.json')
    expected=dict(N=entry['N'],SQP=entry['SQP'],PCG=entry['PCG'],variant='short12',timing_mode='A',
        offline_timing='controller_dt',backend='internal',batch_size=1,source_commit=entry['source_commit'],cohort_id=entry['cohort_id'])
    if any(protocol.get(k)!=v for k,v in expected.items()):raise ValueError('Attempt protocol/cell/source/mode mismatch')
    for k in ('machine','source_commit','cohort_id'):
        if provenance.get(k)!=entry[k]:raise ValueError('Attempt provenance mismatch')
    if provenance.get('evidence_kind', 'measurement')!=index['evidence_kind']:raise ValueError('Evidence kind mismatch')
    if provenance.get('bank_sha256')!=index['bank']['sha256']:raise ValueError('Attempt bank identity mismatch')
    outcomes={e['episode_id']:e for e in metrics.get('episodes',[])}
    if len(outcomes)!=len(metrics.get('episodes',[])) or any(k not in schedule for k in outcomes):raise ValueError('Duplicate/unknown episode outcome')
    grouped=defaultdict(list)
    if 'events.jsonl' not in manifest:
        if metrics.get('execution_error') is None:raise ValueError('Missing events without retained execution failure')
    else:
        for number,line in enumerate((attempt/'events.jsonl').read_text().splitlines()):
            event=json.loads(line,parse_constant=lambda value:(_ for _ in ()).throw(ValueError(value)))
            if event.get('event_index')!=number:raise ValueError('Duplicate/nonsequential event index')
            filename=event.get('raw_file');raw_path=checked_path(attempt,filename)
            if filename not in manifest or event.get('sha256')!=manifest[filename]:raise ValueError('Unpinned event NPZ')
            # All event files are hash-verified; decode only solve timing payloads.
            if event['event'] not in ('solve_started','solve_returned','solve_completed','solve_failed'):continue
            with np.load(raw_path,allow_pickle=False) as archive:
                arrays={name:archive[name] for name in archive.files}
                record=event.get('record',{})
                category=record.get('category');episode=record.get('episode_id');slot=record.get('slot')
                if category=='process_cuda_warmup':
                    if episode is not None or slot is not None or record.get('excluded') is not True:raise ValueError('Invalid unrelated warmup')
                    key=(None,None)
                else:
                    if episode not in schedule or type(slot) is not int or slot not in range(20):raise ValueError('Unknown episode/slot')
                    if category!=schedule[episode]['categories'][slot]:raise ValueError('Category does not match frozen schedule')
                    if record.get('timing_mode')!='A' or record.get('N')!=entry['N']:raise ValueError('Solve mode/N mismatch')
                    if record.get('caps')!={'SQP':entry['SQP'],'PCG':entry['PCG']}:raise ValueError('Solve cap mismatch')
                    key=(episode,slot)
                native,n_status=scalar(record,'native_solve_time_ms',arrays)
                wall,w_status=scalar(record,'wall_solve_time_s',arrays)
                grouped[key].append((event['event'],record,native,n_status,None if wall is None else wall*1000,w_status))
    samples=[]
    keys=[(None,None)]+[(episode,slot) for episode in schedule for slot in range(20)]
    for episode,slot in keys:
        category='process_cuda_warmup' if episode is None else schedule[episode]['categories'][slot]
        records=grouped.get((episode,slot),[]);names=[r[0] for r in records]
        if len(names)!=len(set(names)):raise ValueError('Duplicate solve-stage event')
        if records and names[0]!='solve_started':raise ValueError('Solve continuation without start')
        # Returned/completed/failed all belong to the same call; input+category stay fixed.
        for key in ('category','episode_id','slot','replay_input_hashes'):
            values=[r[1].get(key) for r in records]
            if values and any(value!=values[0] for value in values):raise ValueError('Solve identity changed across stages')
        last=records[-1] if records else None
        outcome=outcomes.get(episode,{})
        row=dict(row_base(entry),episode_id=episode,slot=slot,category=category,excluded=episode is None,
            call_status='missing' if not names else ('failed' if 'solve_failed' in names else 'completed' if 'solve_completed' in names else 'returned' if 'solve_returned' in names else 'started_only'),
            task_success=outcome.get('task_success'),physical_pass=outcome.get('physical_pass'),
            numerical_pass=outcome.get('solver_numerical_pass'),execution_error=outcome.get('execution_error'),
            native_ms=None,wrapper_ms=None,native_status='missing',wrapper_status='missing')
        # Keep last available timing when a later failure lacks native timing.
        for offset,scope,status_key in [(2,'native_ms','native_status'),(4,'wrapper_ms','wrapper_status')]:
            available=[r for r in records if r[offset+1]!='missing']
            if available:row[scope]=available[-1][offset];row[status_key]=available[-1][offset+1]
        samples.append(row)
    return samples,attempt,manifest


def load(index_path,index_sha256,input_root=None,synthetic_test=False):
    index_path=Path(index_path).resolve()
    if sha(index_path)!=index_sha256:raise ValueError('Independent index hash mismatch')
    index=json_read(index_path);root=Path(input_root or index_path.parent).resolve()
    if index.get('schema','f01ab-analysis-index-v1')!='f01ab-analysis-index-v1' or index.get('evidence_kind','measurement') not in ('measurement','synthetic_test'):
        raise ValueError('Unknown index schema/evidence kind')
    index.setdefault('evidence_kind','measurement')
    if (index['evidence_kind']=='synthetic_test')!=synthetic_test:raise ValueError('Synthetic test evidence must be explicitly labeled')
    bank_path=checked_path(root,index['bank']['path'])
    if sha(bank_path)!=index['bank']['sha256']:raise ValueError('Pinned bank JSON hash differs')
    bank=json_read(bank_path)
    if bank.get('N')!=[16,32,64,128] or bank.get('SQP')!=[1,2,5,10] or bank.get('PCG')!=[100,200,400,800]:
        raise ValueError('Frozen budget matrix differs')
    if bank.get('timing_mode')!='A' or bank.get('offline_timing')!='controller_dt' or bank.get('batch_size')!=1:
        raise ValueError('Bank mode/B1 differs')
    variant=bank['variants']['short12']
    schedule={e['id']:e for e in variant['episodes']}
    if len(schedule)!=12 or variant['counts']!={k:EXPECTED[k] for k in CATEGORIES} or variant['measured_calls']!=240:
        raise ValueError('short12 bank counts differ')
    for episode in schedule.values():
        expected=['cold']+['warm']*19
        if episode['kind']=='reaching':expected[10]='goal_switch'
        if episode['updates']!=20 or episode['categories']!=expected:raise ValueError('Invalid frozen episode schedule')
    if Counter(category for episode in schedule.values() for category in episode['categories'])!=Counter(variant['counts']):
        raise ValueError('Declared category counts do not match schedule')
    attempts=index.get('attempts',[]);seen=set();cohorts=set();samples=[];identities=[]
    for entry in attempts:
        key=(entry['machine'],entry['N'],entry['SQP'],entry['PCG'],entry['attempt_id'])
        if key in seen or entry['machine'] not in MACHINES or tuple(key[1:4]) not in CELLS:raise ValueError('Duplicate/invalid matrix attempt')
        if not isinstance(entry['source_commit'],str) or not entry['source_commit'] or not isinstance(entry['cohort_id'],str) or not entry['cohort_id']:
            raise ValueError('Source/cohort required')
        seen.add(key);cohorts.add((entry['source_commit'],entry['cohort_id']))
    if len(cohorts)>1:raise ValueError('Changed source/cohort must be analyzed separately')
    paths=set()
    for entry in attempts:
        rows,attempt,manifest=load_attempt(entry,root,schedule,index)
        if attempt in paths:raise ValueError('Attempt path reused under multiple cases')
        paths.add(attempt);samples.extend(rows);identities.append(dict(entry,files=manifest))
    return dict(index=index,index_path=str(index_path),index_sha256=index_sha256,bank_path=str(bank_path),
        attempts=identities,samples=samples,immutable_paths=[str(p) for p in paths]+[str(bank_path)])


def summarize(loaded):
    summaries=[];coverage=[]
    for entry in loaded['attempts']:
        base=row_base(entry);rows=[r for r in loaded['samples'] if all(r[k]==v for k,v in base.items())]
        for category,expected in EXPECTED.items():
            selected=[r for r in rows if r['category']==category]
            counted=Counter(r['call_status'] for r in selected)
            coverage.append(dict(base,category=category,expected=expected,observed=expected-counted['missing'],
                completed=counted['completed'],failed=counted['failed'],returned=counted['returned'],
                started_only=counted['started_only'],missing=counted['missing']))
            for scope,status in [('native_ms','native_status'),('wrapper_ms','wrapper_status')]:
                finite=[r[scope] for r in selected if r[status]=='finite']
                stats={'mean_ms':None,'median_ms':None,'p95_ms':None}
                if finite:stats=dict(mean_ms=float(np.mean(finite)),median_ms=float(np.median(finite)),p95_ms=float(np.percentile(finite,95)))
                summaries.append(dict(base,category=category,excluded=category=='process_cuda_warmup',scope=scope,
                    total_slots=len(selected),finite=len(finite),missing=sum(r[status]=='missing' for r in selected),
                    nonfinite=sum(r[status]=='nonfinite' for r in selected),nonpositive=sum(r[status]=='nonpositive' for r in selected),
                    **stats,**{f'reciprocal_{k}_Hz':None if v is None else 1000/v for k,v in stats.items()}))
    matrix=[]
    for machine in MACHINES:
        for N,SQP,PCG in CELLS:
            entries=[a for a in loaded['attempts'] if (a['machine'],a['N'],a['SQP'],a['PCG'])==(machine,N,SQP,PCG)]
            matrix.append(dict(machine=machine,N=N,SQP=SQP,PCG=PCG,status='attempted' if entries else 'missing',attempt_ids=';'.join(str(a['attempt_id']) for a in entries)))
    return summaries,coverage,matrix


def write_csv(path,rows):
    columns=list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(rows)


def render(loaded,output):
    output=Path(output).resolve()
    for raw in loaded['immutable_paths']:
        raw=Path(raw)
        if output==raw or output.is_relative_to(raw):raise ValueError('Derived output lies in immutable inputs')
    # Validation fully completed before output creation or matplotlib import.
    output.mkdir(parents=True,exist_ok=False)
    summaries,coverage,matrix=summarize(loaded)
    write_csv(output/'all_samples.csv',loaded['samples']);write_csv(output/'cell_summaries.csv',summaries)
    write_csv(output/'category_coverage.csv',coverage);write_csv(output/'matrix_coverage.csv',matrix)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    label='SYNTHETIC TEST DATA — NOT MEASUREMENT' if loaded['index']['evidence_kind']=='synthetic_test' else 'F01ab saved measurements'
    identity=next(iter(loaded['attempts']),{})
    label+=f"\nsource {identity.get('source_commit','none')} / cohort {identity.get('cohort_id','none')}"
    caveat='Mode A/B1 .6s subsets; cap cells evolve different inputs; no full-period T01 success claim'
    bins_receipt={}
    def save(fig,name):
        fig.savefig(output/(name+'.png'),dpi=130);fig.savefig(output/(name+'.pdf'));plt.close(fig)
    for scope in SCOPES:
        measured=[r for r in summaries if r['scope']==scope and not r['excluded']]
        for N in sorted({r['N'] for r in measured}):
            panel=[r for r in measured if r['N']==N]
            for reciprocal in (False,True):
                fig,axes=plt.subplots(3,1,figsize=(max(12,len(panel)*.085),10),sharex=True)
                for axis,stat in zip(axes,('mean_ms','median_ms','p95_ms')):
                    key='reciprocal_'+stat+'_Hz' if reciprocal else stat
                    for i,row in enumerate(panel):
                        if row[key] is None:axis.plot(i,0,marker='x',color='gray')
                        else:axis.plot(i,row[key],marker='o',color=COLORS[row['category']])
                    axis.set_ylabel(('1000/'+stat+' (Hz)' if reciprocal else stat+' (ms)'))
                    axis.grid(alpha=.2)
                machine_labels={'gtx1070':'1070','rtx5060':'5060','jetson':'Jetson'}
                axes[-1].set_xticks(range(len(panel)),[f"{machine_labels[r['machine']]} S{r['SQP']} P{r['PCG']} {r['category']} a{r['attempt_id']}" for r in panel],rotation=90,fontsize=6)
                totals=Counter(r['call_status'] for r in loaded['samples'] if not r['excluded'] and r['N']==N)
                missing=sum(r['status']=='missing' for r in matrix if r['N']==N)
                fig.suptitle(f'{label}\nN{N} {scope}: '+('reciprocal summary (not actuation rate)' if reciprocal else 'primary solve time')+f'\n{caveat}\ncall coverage {dict(totals)}; missing N{N} matrix cells {missing}/48')
                fig.tight_layout(rect=(0,0,1,.88));save(fig,f'{scope}-N{N}'+('-reciprocal' if reciprocal else '-solve-time'))
        # Common bin edges for a matched cap cell over all machines/attempts.
        present={tuple(r[k] for k in ('N','SQP','PCG')) for r in measured}
        for N,SQP,PCG in sorted(present):
            cellrows=[r for r in loaded['samples'] if (r['N'],r['SQP'],r['PCG'])==(N,SQP,PCG) and not r['excluded']]
            values=[r[scope] for r in cellrows if r[scope] is not None]
            high=max(values) if values else 1.;edges=np.linspace(0,high*1.05 if high>0 else 1.,21)
            bins_receipt[f'{scope}-N{N}-S{SQP}-P{PCG}']=edges.tolist()
            attempts={tuple(r[k] for k in ('machine','attempt_id')) for r in cellrows}
            for machine,attempt_id in sorted(attempts):
                rows=[r for r in cellrows if (r['machine'],r['attempt_id'])==(machine,attempt_id)]
                fig,axis=plt.subplots(figsize=(8,5))
                for category in CATEGORIES:
                    vals=[r[scope] for r in rows if r['category']==category and r[scope] is not None]
                    axis.hist(vals,bins=edges,histtype='step',color=COLORS[category],label=f'{category} finite={len(vals)}')
                counts=Counter(r['call_status'] for r in rows)
                invalid=sum(r[scope] is None for r in rows)
                axis.set(title=f'{label}\n{machine} N{N} SQP{SQP} PCG{PCG} attempt{attempt_id}\n{scope}; {dict(counts)}; missing/nonfinite/invalid timing={invalid}',xlabel=f'{scope} (ms)',ylabel='solve count')
                axis.legend();fig.tight_layout();save(fig,f'hist-{scope}-{machine}-N{N}-S{SQP}-P{PCG}-a{attempt_id}')
    provenance=dict(index_path=loaded['index_path'],index_sha256=loaded['index_sha256'],
        bank=loaded['index']['bank'],attempts=loaded['attempts'],script_sha256=sha(__file__),
        evidence_kind=loaded['index']['evidence_kind'],histogram_bins=bins_receipt,category_colors=COLORS,
        expected_counts=EXPECTED,statistic_definition='All finite positive recorded timing samples including failed calls and scientific negative episodes; warm-up exported/excluded from primary/histograms; reciprocal 1000/aggregate_ms, not mean instantaneous Hz',
        interpretation_limits=caveat,matrix=matrix)
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2,allow_nan=False)+'\n')
    return summaries,coverage,matrix


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--index',type=Path,required=True)
    parser.add_argument('--index-sha256',required=True);parser.add_argument('--input-root',type=Path)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--synthetic-test',action='store_true')
    args=parser.parse_args();render(load(args.index,args.index_sha256,args.input_root,args.synthetic_test),args.output)
