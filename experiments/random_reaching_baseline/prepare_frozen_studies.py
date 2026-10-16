#!/usr/bin/env python3
"""Prepare fresh100 validation and paired development SQP caps; CPU only."""
import argparse
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import random
import shlex
import subprocess
import sys
import numpy as np

SEED=20261015
COUNT=100
CAPS=(1,5,100)
BASELINE_SHA='511b99b2a0ed817f0bb881a84ca978e0de5d74d5e046321451db6231fe2c6b8f'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def id_sha(ids):return hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
def write(p,x):
    payload=json.dumps(x,indent=2,allow_nan=False)+'\n';p=Path(p)
    if p.exists() and p.read_text()!=payload:raise ValueError('Never overwrite a different frozen artifact: '+str(p))
    p.write_text(payload)
def pin(p):return dict(path=str(p),sha256=sha(p))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--pool',type=Path,required=True)
    p.add_argument('--baseline-root',type=Path,default=Path('/workspace/GATO/_expr_results/random-reaching-orientation-20261015/OFF_TORQUE'))
    p.add_argument('--old-assignment',type=Path,default=Path('/workspace/GATO/_expr_results/random-reaching-validation-100-20261015/preparation/assignment.json'))
    p.add_argument('--validation-root',type=Path,required=True);p.add_argument('--sqp-root',type=Path,required=True);args=p.parse_args()
    source,pool,baseline,oldpath,validation,sqp=(getattr(args,k).resolve() for k in ('source','pool','baseline_root','old_assignment','validation_root','sqp_root'))
    worker=source/'experiments/random_reaching_baseline/run_frozen_baseline.py';prior=worker.with_name('run_off_torque.py')
    oldfunc={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(prior.read_text()).body if isinstance(n,ast.FunctionDef)}
    functions={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(worker.read_text()).body if isinstance(n,ast.FunctionDef)}
    if set(functions)!=set(oldfunc) or any(functions[k]!=oldfunc[k] for k in functions if k!='prepare'):raise ValueError('Worker runtime differs beyond prepare')
    proto_path=baseline/'pilot/protocol.json';selected=read(proto_path)
    if sha(proto_path)!=BASELINE_SHA:raise ValueError('Reference protocol differs')
    freeze=baseline/'research_baseline_freeze.json';canonical=baseline/'preflight/preflight/metrics.json'
    frozen=read(freeze);canon=read(canonical)
    if frozen['protocol_sha256']!=BASELINE_SHA or frozen['canonical_metrics_sha256']!=sha(canonical) or canon['preflight_pass'] or not canon['physical_pass'] or canon['execution_error']:raise ValueError('Reference freeze/canonical differs')
    old=read(oldpath);historical=selected['task_ids'];oldids=old['task_ids'];excluded=sorted(set(historical)|set(oldids)|set(selected['development_task_ids']))
    if len(historical)!=100 or len(oldids)!=100 or len(excluded)!=200:raise ValueError('Expected two disjoint complete historical100 rosters')
    for name,digest in selected['input_sha256'].items():
        if sha(pool/name)!=digest:raise ValueError('Frozen pool input differs')
    with np.load(pool/'pairs.npz',allow_pickle=False) as pairs,np.load(pool/'pool.npz',allow_pickle=False) as configurations:
        size=len(pairs['start_index'])
        if size!=10000 or len(pairs['goal_index'])!=size:raise ValueError('Expected original10000-pair archive')
        population=[i for i in range(size) if i not in set(excluded)];ids=random.Random(SEED).sample(population,COUNT)
        records=[dict(order=i+1,pair_id=pair,start_index=int(pairs['start_index'][pair]),goal_index=int(pairs['goal_index'][pair])) for i,pair in enumerate(ids)]
        if any(not 0<=r[k]<len(configurations['q']) for r in records for k in ('start_index','goal_index')):raise ValueError('Pair references missing configuration')
    if set(ids)&set(excluded):raise ValueError('Fresh assignment overlaps seen/excluded tasks')
    now=datetime.now(timezone.utc).isoformat()
    for root in (validation,sqp):
        prep=root/'preparation'
        if prep.exists():raise ValueError('Use fresh preparation roots; retain previous frozen preparations')
        prep.mkdir(parents=True)
    vp=validation/'preparation';sp=sqp/'preparation'
    assignment=dict(schema_version=1,role='fresh100_validation_only',created_utc=now,seed=SEED,sampler='random.Random(seed).sample(population,100)',population_size=size,remaining_population_size=len(population),population_order='ascending original pair IDs excluding both historical100 rosters; no geometry/outcome filtering',
        task_ids=ids,task_ids_sha256=id_sha(ids),excluded_task_ids=excluded,excluded_task_ids_sha256=id_sha(excluded),excluded_historical_task_ids=historical,excluded_old_validation_task_ids=oldids,
        development_task_ids=selected['development_task_ids'],old_stopped_assignment=pin(oldpath),input_sha256=selected['input_sha256'],records=records,
        sampling='Original uniform-q configuration pool and original fixed pair table; targets use each recorded goal q FK. No IK/new sampler/easy-hard filter/clipping.',
        stopping_policy='One pass of all100; preserve all individual failures and continue; no tuning/selection/rescue/retry/filtering from validation outcomes.')
    write(vp/'assignment.json',assignment);(vp/'assignment.sha256').write_text(sha(vp/'assignment.json')+'  assignment.json\n')
    (vp/'task_ids.txt').write_text(' '.join(map(str,ids))+'\n')
    common=dict(baseline_protocol_sha256=BASELINE_SHA,research_baseline_freeze=pin(freeze),reused_canonical=dict(**pin(canonical),preflight_pass=False,physical_pass=True,final_position_m=canon['final']['position_error_m'],policy='Reuse unchanged failed OFF_TORQUE canonical openly as diagnostic context, never as a passed qualification. Latest user explicitly requests fresh100 and SQP study; no new canonical gate or redundant run.'),
        diagnostic_stage_policy='Shared setup/import/module/controller-constructor failures hold affected run; individual task/execution/nonfinite outcomes remain recorded and independent cases continue. No retry. Natural zero-PCG stopping unchanged; no KKT gate.',
        selection_policy='Frozen OFF_TORQUE selected only from original development study before either new study; fresh100 is validation only, no tuning or parameter selection from its outcomes.')
    jobs=[]
    v=dict(selected);v.update(common);v.update(id='OFF_TORQUE-fresh-validation100-A-v1',study_role='fresh100_validation_only',task_ids=ids,task_ids_sha256=id_sha(ids),run_task_ids=ids,validation_task_ids=ids,validation_assignment=pin(vp/'assignment.json'),development_task_ids_are_provenance_only=True,
        comparison='Fresh100 current OFF_TORQUE, separate from user-stopped M14 partial. No results from either validation may be merged into a112-task development population or used to retune the frozen configuration.')
    v.pop('historical_manifest_sha256_original_unavailable',None);write(vp/'validation_protocol.json',v);jobs.append(('VALIDATION100',vp/'validation_protocol.json',ids,validation/'pilot'))
    for cap in CAPS:
        q=dict(selected);q.update(common);q['solver_params']=dict(selected['solver_params'],max_sqp_iters=cap)
        q.update(id=f'OFF_TORQUE-development12-sqp{cap}-A-v1',study_role='development_sqp_cap',run_task_ids=selected['development_task_ids'],comparison='Only outer SQP cap changes: new1/5/100 on same12; reuse completed1000. Preserve practical zero-PCG exit; caps are allowances, not required work. Fresh100 is a separate fixed configuration validation.')
        path=sp/f'SQP{cap}_protocol.json';write(path,q);jobs.append((f'SQP{cap}',path,q['run_task_ids'],sqp/f'SQP{cap}/pilot'))
    summary=read(baseline/'pilot/summary.json');observed_seconds=sum(m['episode_wall_s'] for m in summary);casebytes=sum(f.stat().st_size for d in (baseline/'pilot').glob('pair_*') for f in d.rglob('*') if f.is_file())
    resource=dict(reference=str(baseline/'pilot'),reference_cases=len(summary),observed_episode_wall_s=observed_seconds,observed_case_bytes=casebytes,linear_validation100_wall_s=observed_seconds/12*100,linear_validation100_bytes=casebytes/12*100,
        planning='Fresh100 linear estimate about23min/193MB; allow30-120min and1GiB raw plus separate backup. New SQP36 episodes planning5-20min/0.5GiB raw plus backup. Difficulty/termination/CPU recording may change these estimates substantially; no duration/storage pruning authorized. Builds/modules reused, no new compile/native readiness run.',
        SQP_new_episodes=36,SQP_reused_episodes=12,validation_new_episodes=100)
    commands=[]
    for arm,path,tasks,output in jobs:
        argv=['/opt/gato-venv/bin/python3',str(worker),'--source',str(source),'--module',selected['native_module_path'],'--pool',str(pool),'--protocol',str(path),'--pairs',*map(str,tasks),'--output',str(output)]
        prep=vp if arm=='VALIDATION100' else sp
        dry=subprocess.run([*argv,'--dry-run'],capture_output=True,text=True)
        (prep/f'dryrun-{arm}.json').write_text(dry.stdout);(prep/f'dryrun-{arm}.stderr.log').write_text(dry.stderr)
        if dry.returncode:raise RuntimeError('Exact CPU dryrun failed: '+arm)
        commands.append(dict(arm=arm,argv=argv,cwd=str(source),protocol_sha256=sha(path),dryrun_sha256=sha(prep/f'dryrun-{arm}.json')))
    prepared=dict(created_utc=now,source=str(source),source_commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),worker=pin(worker),reference_worker=pin(prior),runtime_AST_unchanged=True,
        protocol_sha256=BASELINE_SHA,module_sha256=sha(selected['native_module_path']),baseline_freeze=pin(freeze),canonical=pin(canonical),GPU_execution_started=False,commands=commands,resource_estimate=resource)
    for prep in (vp,sp):
        write(prep/'PREPARED.json',prepared);write(prep/'launch_commands.json',commands);write(prep/'resource_estimate.json',resource)
        (prep/'launch_commands.txt').write_text('\n\n'.join(f"# {c['arm']}\n"+shlex.join(c['argv']) for c in commands)+'\n')
    print(json.dumps(dict(validation_preparation=str(vp),sqp_preparation=str(sp),fresh_ids_sha256=id_sha(ids),excluded=len(excluded),dryruns=4,GPU_execution_started=False),indent=2))

if __name__=='__main__':main()
