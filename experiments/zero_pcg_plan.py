#!/usr/bin/env python3
"""Prepare distinct restored-zero-PCG cohorts; never execute measurements.

Old plans and scientific inputs remain unchanged. Run this, then use explicit
successor plans with run.py --dry-run. Jetson duration/storage and faster-task
simulation review remain pending; physical scripts are outside these queues.
"""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
COHORT='zero-pcg-20261011'
POLICY='zero_pcg_or_finite_kkt'
PINS={
 '502fa29c8b0d649dcb347f9534bf29c6935fa6dd':'fa6f6bede286c122cc41ac65f42a583da84290e0',
 '4e5301feaea5f2d06530922ba46aa543e1995d47':'852b0cbcfe6e56fe5337a3b6e5fe8e777f959b77',
 '4f6c8fd4be428c135fc16564c5dd56eb61ce5660':'84bad9ad3d88b1b50d01f4d7aa3721bbc31ea210',
 '63940f7a5c724c84c731a6a1edbc49a094f3b806':'8dbb8b8d6dc1d65e902373a9aa58693e33329c14',
 'd6c827842c6ed65d455472063bcf77f5498efd46':'b931bbdd41303f1456ce3242045ebe48b241600a',
 'a3a96ad3deabea5c0165da69addd1c55406710a9':'c887bea29d3ed8a24de339478a7390bd4bac1915'}
ORDINARY=PINS['502fa29c8b0d649dcb347f9534bf29c6935fa6dd']

def read(name):return json.loads((ROOT/'experiments'/name).read_text())
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def qualified(build):
    result=copy.deepcopy(build)
    for command in result['commands']:
        if command[:2]==['cmake','-S'] and command[2]=='{runner}/native':command[2]='{runner}/native/qualified'
    for relative in ('native/CMakeLists.txt','native/qualified/CMakeLists.txt','native/task_readiness/CMakeLists.txt'):
        result['inputs'][relative]=sha(ROOT/'experiments'/relative)
    if 'cuda_access_smoke' not in result['artifacts']:result['artifacts'].append('cuda_access_smoke')
    return result

def suite_then_canonical(e):
    canonical=['@canonical-output@' if x=='{output}' else x for x in e['preflight']['command']]
    e['preflight']['command']=['{python}','{runner}/adapters/successor_preflight.py',
        '--source','{source}','--build','{build}','--output','{output}','--',*canonical]
    e['build']=qualified(e['build'])

def successor(e,machine):
    e=copy.deepcopy(e);old=e['source_commit'];e['source_commit']=PINS[old];e['machines']=[machine]
    e.update(cohort_id=COHORT,stopping_policy=POLICY,original_runtime_source_commit=old,
        original_scientific_protocol=e['protocol'],protocol=e['protocol']+'-'+COHORT,
        preparation_status='fresh matching build and target preflight pending')
    if 'preflight' in e:e['preflight']['group']+='-'+COHORT
    e.pop('description',None)
    return e

def faster_inputs():
    old=read('inputs/t01/t01.json');base=old['tasks'][3]
    samples=np.asarray(base['reference_packed'],dtype='<f8').reshape(-1,6)
    old_times=np.arange(len(samples))*base['reference_dt_s']
    faster=copy.deepcopy(old);faster.update(name='T01-faster-8-6-4s-simulation-proposal',
        repeat_policy='One simulation attempt per proposed speed/machine; no live release',
        release_status='simulation_only_proposed_common_gate_pending_Felix_review',tasks=[],
        faster_proposal=dict(periods_s=[8,6,4],RMS_gate_m=.025,maximum_gate_m=.05,
            live_use='Held pending Felix review of bounded simulation results',
            initialization='Unchanged initial state, tracking costs, warm shift and feedback-rho reset'))
    for period in (8.,6.,4.):
        t=np.arange(int(np.ceil((period+.93)/.03))+2)*.03
        phase=np.mod(t*24./period,24.)
        ref=np.column_stack([np.interp(phase,old_times,samples[:,j]) for j in range(6)]).astype('<f8')
        task=copy.deepcopy(base);task.update(id=f'figure8-{int(period)}s-proposal',duration_s=period,period_s=period,
            reference_packed=ref.ravel().tolist(),reference_sha256=hashlib.sha256(ref.tobytes()).hexdigest(),
            provisional_RMS_gate_m=.025,provisional_maximum_gate_m=.05,
            maximum_reference_speed_m_s=base['maximum_reference_speed_m_s']*24./period,
            reference_scope='Same frozen24s public geometry, phase-resampled at30ms to faster period; periodic extension covers .93s preview. No dynamic-feasibility claim.')
        faster['tasks'].append(task)
    folder=ROOT/'experiments/inputs/t01_faster'
    write(folder/'t01.json',faster)
    (folder/'v01.json').write_bytes((ROOT/'experiments/inputs/t01/v01.json').read_bytes())
    return {str(p.relative_to(ROOT)):sha(p) for p in folder.iterdir()},faster

def prepare():
    output=ROOT/'experiments/successor';files=[]
    installed=json.loads((output/'inventory.json').read_text()) if (output/'inventory.json').exists() else {}
    installed_force=installed.get('force')
    force_files=[f for f in installed.get('plan_files',[]) if f.startswith('successor/force_')]
    if any(not (ROOT/'experiments'/f).is_file() for f in force_files):
        raise ValueError('Installed force successor plan is missing; do not silently drop it')
    faster_hashes,faster=faster_inputs()
    original_campaign=read('campaign.json')['experiments']
    templates=[]
    for filename in ('f01c.json','resource_profiles.json','bb_rtx5060.json','ba_rtx5060.json','control_studies.json','cylinder_studies.json','demo_internal.json'):
        for e in read(filename)['experiments']:
            if e['id'].startswith('F04') or e['id']=='CylinderChecks':continue
            templates.append(e)
    for machine in ('gtx1070','rtx5060','jetson'):
        entries=[]
        # Jetson viability is first, before timing and any later live work.
        for original in original_campaign:
            e=successor(original,machine);suite_then_canonical(e);entries.append(e)
            if e['id']=='T01':
                fast=copy.deepcopy(e);fast.update(id='T01-faster-proposal',inputs=faster_hashes,
                    cases=[dict(id=t['id'],parameters=dict(simulation_only=True,provisional_gate=True)) for t in faster['tasks']],
                    faster_tracking_proposal=True,live_release='Held pending Felix simulation review')
                for argv in (fast['command'],fast['preflight']['command']):
                    for i,arg in enumerate(argv):argv[i]=arg.replace('/inputs/t01','/inputs/t01_faster')
                entries.append(fast)
        for original in templates:
            if machine=='jetson' and original['id'] in ('BA','ControlChoices','demo_visual_square_internal'):continue
            e=successor(original,machine)
            if e['id']=='F01c':suite_then_canonical(e)
            if e['id'] in ('BA','BB'):
                # Always use fresh native source; historical native-copy/readiness
                # recipes belong only to old cohorts.
                e['build']=qualified(read('bb_rtx5060.json')['experiments'][0]['build'])
                obsolete=('source_reuse.json','gtx1070_readiness.json','gtx1070_A_preflight.json','continuation.json')
                e['inputs']={k:v for k,v in e['inputs'].items() if not k.endswith(obsolete)}
                suite_then_canonical(e)
            if e['id']=='CylinderTasks':
                if machine=='jetson':
                    approved=read('cylinder_studies_jetson_proposed.json')['experiments'][0]['cases']
                    e['cases']=copy.deepcopy(approved)
                e['dependency_source_commits']={'ordinary_canonical_runtime':ORDINARY}
                for argv in (e['command'],e['preflight']['command']):argv+=['--ordinary-runtime',ORDINARY]
            entries.append(e)
        plan=dict(version=1,cohort_id=COHORT,stopping_policy=POLICY,experiments=entries,
            status='prepared_target_build_preflight_pending' if machine!='jetson' else 'proposed_duration_storage_discussion_pending',
            note='New solver cohort, full frozen coverage; historical results/failed attempts and gates unchanged. Faster tracking simulation only; no physical robot queue.',
            omissions=dict(BA_seeds_alone='Dropped; seed-to-output progress remains saved analysis',
                Jetson_BA='No full BA task sweep; same-candidate speedup answered by F01c',
                force='Revised runtime, bounded smoke/screen and selected F* pending; see force_stages.json',
                independent_studies='No solver-change rerun of F04/Q02/X02/M02; pending M02/F04 target plans remain separate'))
        filename=machine+'_internal.json';write(output/filename,plan);files.append('successor/'+filename)
    live=[]
    for filename in ('deployment.json','demo_gazebo.json','simulated_faults.json'):
        live += [successor(e,'rtx5060') for e in read(filename)['experiments']]
    write(output/'rtx5060_gazebo.json',dict(version=1,cohort_id=COHORT,stopping_policy=POLICY,experiments=live,
        status='live_setup_and_prior_simulation_review_pending',prerequisites='Jetson V01/T01 B and Felix review before live; faster tracking not included. Manual D02 outside runner.'))
    files.append('successor/rtx5060_gazebo.json')
    if not (output/'force_stages.json').exists() or not json.loads((output/'force_stages.json').read_text()).get('source_commit'):
        write(output/'force_stages.json',dict(version=1,experiments=[],status='staged_prerequisites_not_execution_ready',
        stages=['Original-arm GTX1070 0N smoke; repeat crash holds force study',
            'New same-policy no-estimator0N baseline and5/10/20N modeB screen; frozen degradation criterion; choose smallest qualifying force within effort limits',
            'Freeze selectedF* and revised12input bank: sixstarts x{0,F*}',
            'E02 three methods x(one0N +threeF* repeats)=12tasks/machine; E03 bank one excluded warmup+20ticks/input/method; derive E01'],
        source_commit=None,selected_force_N=None,machines=['gtx1070','rtx5060','jetson'],stopping_policy=POLICY,
            limitation='This file cannot run old force adapters; revised source and F* must be supplied by force owner.'))
    independent=[]
    for filename in ('model_checks.json','collision_monitor.json','collision_monitor_jetson_proposed.json','resource_profiles.json','resource_profiles-jetson-proposed.json'):
        for e in read(filename)['experiments']:
            if filename.startswith('resource_profiles') and not e['id'].startswith('F04'):continue
            e=copy.deepcopy(e);e['machines']=[m for m in e['machines'] if m!='gtx1070']
            if not e['machines']:continue
            e['preparation_status']='Pending original target measurement, not a stopping-policy rerun'
            independent.append(e)
    write(output/'pending_independent.json',dict(version=1,experiments=independent,
        status='Original pending target checks; unchanged valid scientific source pins',
        note='Q02/M02/F04 already measured on GTX1070 are excluded; Jetson and RTX5060 unmeasured original studies retained. No pure-check rerun for solver change.'))
    files.append('successor/pending_independent.json')
    files+=force_files
    stage=json.loads((output/'force_stages.json').read_text())
    inventory=dict(cohort_id=COHORT,plan_files=files,source_map=PINS,
        status='Explicit successor plans prepared; target family preflights and campaign measurements pending',
        force_stage_status=stage.get('status'),
        retained='All old plans, input banks, raw attempts and scientific failures remain unchanged')
    if installed_force:inventory['force']=installed_force
    if stage.get('source_commit'):inventory['force_readiness']=stage
    if installed.get('force_stage_plans'):inventory['force_stage_plans']=installed['force_stage_plans']
    write(output/'inventory.json',inventory)
    return files

if __name__=='__main__':
    for path in prepare():print(path)
