#!/usr/bin/env python3
"""Prepare frozen force comparison plans; never build or measure."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
METHODS=('no_estimator','original','adapted')
MACHINES=('gtx1070','rtx5060','jetson')


def prepare(output,source_commit,machine):
    if not re.fullmatch('[0-9a-f]{40}',source_commit) or machine not in MACHINES:raise ValueError('Full source pin and named machine required')
    inputs={}
    for family,names in [('native_inputs',('arrays.npz','protocol.json','SHA256SUMS.json')),
                         ('task_inputs',('v01.json','t01.json','SHA256SUMS.json'))]:
        for name in names:
            relative=f'experiments/force_adapter/{family}/{name}'
            data=subprocess.check_output(['git','-C',str(ROOT),'show',source_commit+':'+relative])
            target=ROOT/relative
            if target.exists() and target.read_bytes()!=data:raise ValueError('Existing frozen force input differs')
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
            inputs[relative]=hashlib.sha256(data).hexdigest()
    build=dict(commands=[['cmake','-S','{source}/experiments/force_adapter/native','-B','{build}',
                '-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}-real','-DPython3_EXECUTABLE={python}'],
                ['cmake','--build','{build}','--target','bsqpN32_tiago_right','cuda_access_smoke','--parallel','2']],
               artifacts=['modules/bsqpN32_tiago_right*.so','cuda_access_smoke'])
    def command(worker):return ['{python}','-m','experiments.force_adapter.'+worker,'--build','{build}',
                               '--runner','{runner}','--inputs','{inputs}/experiments/force_adapter/'+('native_inputs' if worker=='run_native' else 'task_inputs'),
                               '--output','{output}']
    def experiment(eid,mode,description):
        return dict(id=eid,research_question='RQ-force',protocol='force-body-fixed-v1',source_commit=source_commit,
            machines=[machine],timing_mode=mode,backend='internal',description=description,build=build,inputs=inputs,
            preparation_status='prepared_no_measurements',cases=[])
    timing=experiment('E03','A','Force estimator time/memory: same frozen16 hypotheses, original B16 versus adapted selected B1')
    timing['command']=command('run_native')+['--case','{case}']
    timing['preflight']=dict(group='force-native-readiness',command=['{build}/cuda_access_smoke'])
    for index in range(6):
        for truth in range(2):
            for method in ('original','adapted'):
                timing['cases'].append(dict(id=f'input{index}-{method}-truth{truth}',parameters=dict(study='E03',input_index=index,truth_index=truth,mode=method,timing_mode='A',backend='internal',machine=machine)))
    experiments=[timing]
    for method in METHODS:
        for truth in range(2):
            task=experiment(f'E02-{method}-truth{truth}','B','Body-fixed known-disturbance tracking: unchanged24s figure-eight and gates')
            task['command']=command('run_task')+['--case','{case}']
            task['preflight']=dict(group=f'force-B-{method}-truth{truth}',command=command('run_task')+['--preflight','--mode',method,'--truth-index',str(truth)])
            task['cases']=[dict(id='figure8-24s',parameters=dict(study='E02',mode=method,truth_index=truth,timing_mode='B',backend='internal',machine=machine))]
            experiments.append(task)
    plan=dict(version=1,experiments=experiments,preparation=dict(unique_one_ticks=24,excluded_unrelated_warmups=24,
        disturbance_tasks=6,canonical_mode_B_preflights=6,nominal_tracking_seconds=144,
        E01='Saved-data correspondence uses four first-input measured E03 records; no duplicate native executions',
        force_scope='Attached/body-fixed arm_right_7_joint local x2N at joint origin; zero comparison. Not constant world force.',
        timing='Exact accepted modeB old-plan plant clock; fresh owner/model allocation and memory queries included in full algorithm wall. Native solve separately retained. Disk recording outside.',
        baseline='No estimator: zero-wrench B1 every tick. All methods start with zero/prior lane0 B1; no true-force access or estimator update before observed history.',
        memory='cudaMemGetInfo boundary availability delta; host peak RSS separate. Not continuous peak or per-solver allocation, no host/device summing on shared-memory Jetson.',
        readiness='Productionfloat force path and unchanged task-suite accepted at7d317fa. New method/truth canonicalB preflight per target. FailedB stays held; namedA diagnostic preserves evidence, no automatic rescue.',
        jetson='Execution requires duration/storage/bottleneck discussion with Felix; retain original protocol, no silent pruning.',
        execution_status='prepared_only_no_measurements'))
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(plan,indent=2)+'\n')
    return plan


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit',required=True);parser.add_argument('--machine',choices=MACHINES,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    prepare(args.output,args.source_commit,args.machine);print(args.output)
