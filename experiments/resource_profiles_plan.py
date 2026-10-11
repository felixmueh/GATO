#!/usr/bin/env python3
"""Prepare resource, response, profile and sustained timing plans; never execute."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from adapters import resource_profiles_common as C

ROOT=Path(__file__).resolve().parents[1]
SOURCE='4e5301feaea5f2d06530922ba46aa543e1995d47'
PROTOCOL='simple-resource-profiles-v1'


def prepare(output,machine='rtx5060'):
    if machine not in ('rtx5060','jetson'):raise ValueError('Only pending targets may be prepared')
    inputs={}
    bundles=[('resource_profile_campaign','f01c',name) for name in
             ('bank.json','arrays.npz','preparation.json','seed_diagnostics.json','SHA256SUMS.json')]
    bundles += [('campaign','t01',name) for name in ('v01.json','t01.json')]
    for original_family,family,name in bundles:
        data=subprocess.check_output(['git','-C',str(ROOT),'show',f'{SOURCE}:experiments/{original_family}/inputs/{name}'])
        relative=f'experiments/inputs/{family}/{name}'
        target=ROOT/relative
        if target.exists() and target.read_bytes()!=data:raise ValueError('Existing frozen input differs: '+str(target))
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        inputs[relative]=hashlib.sha256(data).hexdigest()
    builds={}
    for on in (False,True):
        knots=(32,) if on else (16,32,64,128)
        builds[on]=dict(inputs={relative:hashlib.sha256((ROOT/'experiments'/relative).read_bytes()).hexdigest()
                    for relative in ('native/CMakeLists.txt','native/resource_profiles/CMakeLists.txt','native/task_readiness/CMakeLists.txt')},
            commands=[['cmake','-S','{runner}/native/resource_profiles','-B','{build}',
                       '-DGATO_SOURCE_DIR={source}','-DKNOTS='+';'.join(map(str,knots)),
                       '-DCMAKE_BUILD_TYPE=Release','-DGATO_EXPERIMENT_TELEMETRY=ON',
                       '-DGATO_EXPERIMENT_STAGE_TIMING='+('ON' if on else 'OFF'),
                       '-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}-real','-DPython3_EXECUTABLE={python}'],
                      ['cmake','--build','{build}','--parallel','2']],
            artifacts=[f'modules/bsqpN{n}_tiago_right*.so' for n in knots]+['cuda_access_smoke'])
    experiments=[]
    for study,on in (('F04',False),('F05',False),('F05',True)):
        selected=[c for c in C.cases() if c['study']==study and c['stage_timing']==on]
        def name(c):
            result=f"{c['arm']}-N{c['N']}-B{c['B']}"
            if 'SQP' in c:result+=f"-SQP{c['SQP']}-PCG{c['PCG']}"
            if 'replicate' in c:result+=f"-r{c['replicate']}"
            return result
        preflight_command=(["{python}","{runner}/adapters/task_suite.py","--source","{source}","--build","{build}",
                            "--output","{output}"] if on else [
                '{python}','{runner}/adapters/task.py','--source','{source}','--build','{build}',
                '--inputs','{inputs}/experiments/inputs/t01','--output','{output}','--preflight','--timing-mode','A'])
        experiments.append(dict(id=study+('-stageON' if on else '-stageOFF'),research_question='RQ-timing',
            protocol=PROTOCOL,source_commit=SOURCE,machines=[machine],prepared_machines=[machine],
            preparation_status='prepared_no_measurements',
            timing_mode='A',backend='internal',
            description=('Compiled-kernel resource queries, zero solves' if study=='F04' else
                         'Stage ON stream profiles' if on else 'First response and six single120s sustained arms'),
            build=builds[on],inputs=inputs,
            preflight={'group':'resources-A-stage'+('ON' if on else 'OFF'),'command':preflight_command},
            command=['{python}','{runner}/adapters/resource_profiles.py','--source','{source}','--build','{build}',
                     '--inputs','{inputs}/experiments/inputs/f01c','--case','{case}','--output','{output}'],
            cases=[dict(id=name(c),parameters=dict(c,timing_mode='A',backend='internal')) for c in selected]))
    plan=dict(version=1,experiments=experiments,retained_results={
        'F04/F05':{'gtx1070':'All38 workers completed/reviewed; preserve/import old measurements'}},
        preparation=dict(machine=machine,native_calls='36 fixed plus variable sustained-loop calls',
            queries=8,workers=38,sustained_arms=6,sustained_seconds_per_arm=120,sustained_repeats=1,
            duration_approval_sha256=C.DURATION_APPROVAL_SHA256,
            execution_status='preparation_only_no_execution',
            readiness='Stage OFF canonical A. Stage ON unchanged original three-case task-suite launcher; reuse successful same-machine/module evidence.',
            jetson='Discuss measured bottlenecks, estimated duration/storage and shorter-protocol options with Felix before execution; no budget pruning'),
        note='OFF resource/response/sustained groups followed by ON profiles. New cohort/order recorded; no thermal-steady claim. Use resource_profiles_monitor.py during actual queue execution for passive nvidia-smi desktop or tegrastats Jetson temperature/clock/power recording outside worker timers.')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(plan,indent=2)+'\n')
    return plan


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--machine',choices=('rtx5060','jetson'),default='rtx5060')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    output=args.output or ROOT/'experiments'/('resource_profiles-jetson-proposed.json' if args.machine=='jetson' else 'resource_profiles.json')
    prepare(output,args.machine);print(output)
