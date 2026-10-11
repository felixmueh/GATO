#!/usr/bin/env python3
"""Write the remaining desktop F01ab/T01 plan from existing frozen protocols.

This prepares files only. Completed GTX1070 and Jetson F01ab runs are retained;
only the remaining RTX5060/internal measurements are selected by default.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = '502fa29c8b0d649dcb347f9534bf29c6935fa6dd'
PROTOCOL = 'simple-harness-v1'


def frozen(relative):
    return subprocess.check_output(['git', '-C', str(ROOT), 'show', SOURCE + ':' + relative])


def prepare(output):
    input_root = ROOT / 'experiments/inputs'
    input_hashes = {}
    paths = ['experiments/campaign/inputs/' + name for name in ('v01.json', 't01.json')]
    paths += ['experiments/frequency_campaign/inputs/' + name for name in ('bank.json', 'arrays.npz', 'model.urdf')]
    for original in paths:
        family = 'f01ab' if '/frequency_campaign/' in original else 't01'
        relative = 'experiments/inputs/' + family + '/' + Path(original).name
        payload = frozen(original)
        target = ROOT / relative
        if target.exists() and target.read_bytes() != payload:
            raise ValueError('Existing frozen input differs: ' + str(target))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        input_hashes[relative] = hashlib.sha256(payload).hexdigest()
    bank = json.loads((input_root / 'f01ab/bank.json').read_text())
    t01 = json.loads((input_root / 't01/t01.json').read_text())
    build = {'inputs': {'native/CMakeLists.txt': hashlib.sha256((ROOT / 'experiments/native/CMakeLists.txt').read_bytes()).hexdigest()}, 'commands': [
        ['cmake', '-S', '{runner}/native', '-B', '{build}', '-DGATO_SOURCE_DIR={source}',
         '-DKNOTS=16;32;64;128', '-DCMAKE_BUILD_TYPE=Release', '-DGATO_EXPERIMENT_TELEMETRY=ON',
         '-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}-real', '-DPython3_EXECUTABLE={python}'],
        ['cmake', '--build', '{build}', '--parallel', '2']],
        'artifacts': ['modules/bsqpN' + str(n) + '_tiago_right*.so' for n in (16,32,64,128)]}
    common = dict(source_commit=SOURCE, protocol=PROTOCOL, backend='internal',
                  machines=['rtx5060'], build=build, inputs=input_hashes)
    task_command = ['{python}', '{runner}/adapters/task.py', '--source', '{source}', '--build', '{build}', '--output', '{output}', '--inputs', '{inputs}/experiments/inputs/t01']
    f01 = dict(common, id='F01ab', research_question='RQ-timing', timing_mode='A',
        description='Frozen short12 cold/warm/goal-switch timings; completed 1070/Jetson are imported, not rerun.',
        preflight={'command': task_command + ['--preflight', '--timing-mode', 'A'], 'group':'F01ab-A'},
        command=['{python}', '{runner}/adapters/f01ab.py', '--source','{source}','--build','{build}',
                 '--case','{case}','--output','{output}','--inputs','{inputs}/experiments/inputs/f01ab'],
        cases=[dict(id=f'N{n}-SQP{sqp}-PCG{pcg}', parameters=dict(variant='short12',N=n,SQP=sqp,PCG=pcg,timing_mode='A',backend='internal'))
               for n,sqp,pcg in itertools.product(bank['N'],bank['SQP'],bank['PCG'])])
    task = dict(common, id='T01', research_question='RQ-control', timing_mode='B',
        description='Six existing frozen reaching/tracking tasks, computation time paid; desktop internal cohort.',
        preflight={'command': task_command + ['--preflight','--timing-mode','B'], 'group':'T01-B'},
        command=task_command + ['--case','{case}','--timing-mode','B'],
        cases=[dict(id=t['id'], parameters=dict(protocol='T01', offline_timing='solve_time')) for t in t01['tasks']])
    plan = dict(version=1, experiments=[task,f01], retained_results={
        'F01ab': {'gtx1070':'completed historical cohort; import existing results',
                  'jetson':'Felix reports completed; awaiting manual collection; do not rerun'},
        'T01': {'gtx1070':'six completed task/physical passes; preserve original cohort'}},
        note='No GPU execution during plan generation. Jetson T01 repeat/duration decisions remain separate. D02 remains manual.')
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(plan,indent=2)+'\n')
    return plan


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'experiments/campaign.json')
    args=parser.parse_args()
    prepare(args.output)
    print(args.output)
