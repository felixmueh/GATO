#!/usr/bin/env python3
"""Prepare revised force smoke/screen, then final cohorts after selected F*.

No measurements occur here. Source must be a reviewed full commit. Screen
preparation requires passed actual smoke evidence. Final preparation requires
the saved bounded screen selection and a separately frozen matching v2 bank.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
from force_screen import smoke_evidence

ROOT=Path(__file__).resolve().parents[1]
METHODS=('no_estimator','original','adapted')
MACHINES=('gtx1070','rtx5060','jetson')
POLICY='zero_pcg_or_finite_kkt'

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def task_inputs(source):
    inputs={}
    for name in ('v01.json','t01.json','SHA256SUMS.json'):
        payload=subprocess.check_output(['git','-C',str(ROOT),'show',source+':experiments/force_adapter/task_inputs/'+name])
        relative='experiments/inputs/force_v2/task_inputs/'+name;target=ROOT/relative
        if target.exists() and target.read_bytes()!=payload:raise ValueError('Existing force-v2 task input differs; retain it as a separate cohort')
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(payload);inputs[relative]=sha(target)
    return inputs

def experiment(source,machine,eid,mode,inputs,folder):
    if not re.fullmatch('[0-9a-f]{40}',source):raise ValueError('Reviewed full source commit required')
    command=['{python}','{runner}/adapters/force.py','--source','{source}','--build','{build}',
        '--inputs','{inputs}/'+folder,'--output','{output}']
    build=dict(commands=[['cmake','-S','{source}/experiments/force_adapter/native','-B','{build}',
        '-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}-real','-DPython3_EXECUTABLE={python}'],
        ['cmake','--build','{build}','--target','bsqpN32_tiago_right','cuda_access_smoke','--parallel','2']],
        artifacts=['modules/bsqpN32_tiago_right*.so','cuda_access_smoke'])
    return dict(id=eid,research_question='RQ-force',protocol='force-persistent-grid-v2',cohort_id='force-grid-zero-pcg-v2',
        stopping_policy=POLICY,source_commit=source,machines=[machine],timing_mode=mode,backend='internal',
        inputs=inputs,build=build,command=command+['--case','{case}'],timeout_seconds=900,cases=[])

def canonical(e,method):
    base=e['command'][:-2]
    e['preflight']=dict(group='force-grid-B-'+method,command=base+['--preflight','--mode',method],
        scope='Undisturbed0N rough-reaching readiness, shared across0/F*; disturbance physical compliance measured in task runs')

def prepare_stage(source,stage,smoke=None,resume_after_approved_setup=False):
    stages=ROOT/'experiments/successor/force_stages.json'
    previous=json.loads(stages.read_text()) if stages.exists() else {}
    if previous.get('held') and not resume_after_approved_setup:
        raise ValueError('Force study held; discuss setup review before replacing the recorded hold')
    inputs=task_inputs(source);folder='experiments/inputs/force_v2/task_inputs'
    if stage=='smoke':
        e=experiment(source,'gtx1070','ForceOriginal0Smoke','B',inputs,folder);canonical(e,'original')
        e['cases']=[dict(id='original-0N-24s',parameters=dict(study='E02',stage='smoke',mode='original',force_N=0.,repeat=0,timing_mode='B',backend='internal',machine='gtx1070'))]
        evidence=None
    else:
        evidence=smoke_evidence(smoke)
        e=experiment(source,'gtx1070','ForceDisturbanceScreen','B',inputs,folder);canonical(e,'no_estimator')
        e['cases']=[dict(id=f'no-estimator-{force:g}N',parameters=dict(study='E02',stage='screen',mode='no_estimator',force_N=force,repeat=0,timing_mode='B',backend='internal',machine='gtx1070')) for force in (0.,5.,10.,20.)]
    plan=dict(version=1,experiments=[e],stage=stage,status='prepared_reviewed_source_target_checks_pending',
        original_smoke=evidence,screen_criterion=dict(RMS_degradation_m=.010,forces_N=[5,10,20],selection='smallest qualifying force within sampled effort and physical limits; no expansion'),
        note='One original0N tracking smoke first after canonical; crash/physical failure holds force study. Historical tracking task gate remains separately reported. No automatic screen or study continuation.')
    if previous.get('held'):plan['retained_hold']=previous
    path=ROOT/'experiments/successor'/f'force_{stage}.json';write(path,plan)
    inventory=ROOT/'experiments/successor/inventory.json';d=json.loads(inventory.read_text())
    d.setdefault('force_stage_plans',{})[stage]='successor/'+path.name;write(inventory,d)
    state=dict(version=1,experiments=[],source_commit=source,
        selected_force_N=None,status='Actual smoke/screen stage plan prepared; final F* and bank pending',stage_plans=d['force_stage_plans'],
        evidence_note='Bounded qualification/smoke/screen outcomes are retained separately; stage preparation launches no measurement.')
    if previous.get('held'):state['retained_hold']=previous
    write(ROOT/'experiments/successor/force_stages.json',state)
    return plan

def record_hold(source,canonical):
    canonical=Path(canonical).resolve();metrics=json.loads((canonical/'metrics.json').read_text())
    provenance=json.loads((canonical/'provenance.json').read_text())
    if provenance['source_commit']!=source:raise ValueError('Hold evidence source differs')
    if metrics.get('task_success') is True and metrics.get('physical_pass') is True and not metrics.get('execution_error'):
        raise ValueError('Canonical evidence does not contain a hold condition')
    status=dict(version=1,experiments=[],source_commit=source,selected_force_N=None,held=True,
        status='Held after original0N canonical physical failure; awaiting discussion of bounded setup-only review',
        canonical_evidence=dict(path=str(canonical),metrics_sha256=sha(canonical/'metrics.json'),
            provenance_sha256=sha(canonical/'provenance.json'),module_sha256=provenance['module_sha256'],
            task_success=metrics.get('task_success'),physical_pass=metrics.get('physical_pass'),
            physical_limit_excess=metrics.get('physical_limit_excess'),execution_error=metrics.get('execution_error')),
        pending=dict(original0N_tracking_smoke=True,bounded_disturbance_screen=True,selected_Fstar=True,
            E02_tasks_per_machine=12,E03_bank_inputs=12,E03_method_cases_per_machine=24,E01='Derived from future GPU bank'),
        note='No smoke/screen/final force campaign was run. No correction, tuning, extra GPU call or solver investigation authorized by this hold record.')
    write(ROOT/'experiments/successor/force_stages.json',status)
    smoke=ROOT/'experiments/successor/force_smoke.json'
    if smoke.exists():
        d=json.loads(smoke.read_text());d['status']=status['status']
        for e in d['experiments']:e.update(enabled=False,held_reason=status['status'],hold_evidence=status['canonical_evidence'])
        write(smoke,d)
    path=ROOT/'experiments/successor/inventory.json';d=json.loads(path.read_text());d['force_stage_status']=status['status']
    d['force_readiness']=status;write(path,d);return status

def prepare_final(source,selection,bank):
    stages=ROOT/'experiments/successor/force_stages.json'
    if stages.exists() and json.loads(stages.read_text()).get('held'):
        raise ValueError('Force study held; discuss setup review before final cohort preparation')
    selection=Path(selection);chosen=json.loads(selection.read_text());force=chosen['selected_force_N']
    if chosen['outcome']!='selected_smallest_qualifying_force' or force not in (5,10,20):raise ValueError('No bounded F* selected; report before study expansion')
    inputs=task_inputs(source);bank=Path(bank);protocol=json.loads((bank/'protocol.json').read_text())
    if protocol.get('protocol_id')!='force-persistent-grid-native-v2' or protocol.get('selected_force_N')!=force:raise ValueError('Frozen v2 bank does not match selected F*')
    manifests=json.loads((bank/'SHA256SUMS.json').read_text())
    if set(manifests)!= {'arrays.npz','protocol.json'}:raise ValueError('Unexpected bank manifest')
    bank_folder='experiments/inputs/force_v2/native_inputs'
    for name in ('arrays.npz','protocol.json','SHA256SUMS.json'):
        target=ROOT/bank_folder/name;target.parent.mkdir(parents=True,exist_ok=True)
        if name in manifests and sha(bank/name)!=manifests[name]:raise ValueError('Frozen bank hash mismatch')
        if target.exists() and target.read_bytes()!=(bank/name).read_bytes():raise ValueError('Existing force-v2 bank differs; retain original cohort')
        target.write_bytes((bank/name).read_bytes());inputs[str(target.relative_to(ROOT))]=sha(target)
    evidence=ROOT/'experiments/inputs/force_v2/selection.json';write(evidence,chosen);inputs[str(evidence.relative_to(ROOT))]=sha(evidence)
    files=[]
    for machine in MACHINES:
        native=experiment(source,machine,'E03','A',inputs,bank_folder)
        native['preflight']=dict(group='force-grid-native',command=['{build}/cuda_access_smoke'])
        native['cases']=[dict(id=f'input{i}-{method}-truth{truth}',parameters=dict(study='E03',stage='measurement',mode=method,input_index=i,truth_index=truth,timing_mode='A',backend='internal',machine=machine))
            for i in range(6) for truth in (0,1) for method in ('original','adapted')]
        tasks=[]
        for method in METHODS:
            e=experiment(source,machine,'E02-'+method,'B',inputs,'experiments/inputs/force_v2/task_inputs');canonical(e,method)
            e['cases']=[dict(id=f'figure8-{f:g}N-r{repeat}',parameters=dict(study='E02',stage='measurement',mode=method,force_N=f,repeat=repeat,timing_mode='B',backend='internal',machine=machine))
                for f,repeats in ((0.,(0,)),(force,(0,1,2))) for repeat in repeats]
            tasks.append(e)
        plan=dict(version=1,experiments=[native,*tasks],selected_force_N=force,selection_sha256=sha(selection),
            status='prepared_target_preflight_pending' if machine!='jetson' else 'proposed_duration_storage_discussion_pending',
            preparation=dict(E02_tasks=12,E03_input_truth_pairs=12,E03_method_cases=24,excluded_warmups_per_case=1,measured_ticks_per_case=20,
                E01='Derived from matched GPU prediction/selection bank; no separate runs',canonical='Three undisturbed0N method preflights; F* actual task physical outcomes separate',
                timing='Persistent estimator prediction/selection/update/TO wall; nativeTO separate; construction and memory query outside tick timing',
                force='Attached pure force at last-joint origin, initially world-Z downward; changing world direction retained',
                Jetson='Duration/storage discussion before execution; no silent pruning'))
        filename=f'force_{machine}.json';write(ROOT/'experiments/successor'/filename,plan);files.append('successor/'+filename)
    inventory=ROOT/'experiments/successor/inventory.json';d=json.loads(inventory.read_text())
    d['plan_files']=[f for f in d['plan_files'] if not f.startswith('successor/force_')]+files
    d['force']=dict(source_commit=source,selected_force_N=force,selection_sha256=sha(selection),status='Prepared revised full cohorts; target preflights pending')
    d['status']='All explicit cohorts prepared; no measurements launched by generation';write(inventory,d)
    write(ROOT/'experiments/successor/force_stages.json',dict(version=1,experiments=[],status='smoke_screen_selected_final_plans_prepared',
        source_commit=source,selected_force_N=force,selection_sha256=sha(selection),final_plan_files=files,
        note='Original smoke and screen remain separate evidence; old force plans and raw outcomes are historical.'))
    return files

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-commit',required=True)
    p.add_argument('--stage',choices=('smoke','screen','final','hold'),required=True)
    p.add_argument('--smoke-attempt',type=Path);p.add_argument('--selection',type=Path);p.add_argument('--bank',type=Path)
    p.add_argument('--canonical-attempt',type=Path)
    p.add_argument('--resume-after-approved-setup',action='store_true',help='Use only after explicit setup-change/retry authorization; preserves previous hold evidence')
    a=p.parse_args()
    if a.stage=='screen' and not a.smoke_attempt:p.error('Screen needs actual --smoke-attempt')
    if a.stage=='final' and not (a.selection and a.bank):p.error('Final needs selected --selection and frozen --bank')
    if a.stage=='hold' and not a.canonical_attempt:p.error('Hold needs actual --canonical-attempt')
    result=record_hold(a.source_commit,a.canonical_attempt) if a.stage=='hold' else prepare_final(a.source_commit,a.selection,a.bank) if a.stage=='final' else prepare_stage(a.source_commit,a.stage,a.smoke_attempt,a.resume_after_approved_setup)
    print(json.dumps(result,indent=2))
