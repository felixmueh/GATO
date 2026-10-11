#!/usr/bin/env python3
"""Prepare only pending RTX5060 F01c cells from the frozen approved study."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
NATIVE_SOURCE = '502fa29c8b0d649dcb347f9534bf29c6935fa6dd'
COLLECTOR_SOURCE = '9f2ab6f3dfd324e397fd4d6909f70284558c8d76'
PROTOCOL = 'simple-f01c-v1'


def prepare(output):
    hashes = {}
    bundles = [(COLLECTOR_SOURCE, 'batch_timing_campaign', 'f01c', name)
               for name in ('bank.json', 'arrays.npz', 'preparation.json', 'seed_diagnostics.json', 'SHA256SUMS.json')]
    bundles += [(NATIVE_SOURCE, 'campaign', 't01', name) for name in ('v01.json', 't01.json')]
    for commit, original_family, family, name in bundles:
        original = f'experiments/{original_family}/inputs/{name}'
        payload = subprocess.check_output(['git', '-C', str(ROOT), 'show', commit + ':' + original])
        relative = f'experiments/inputs/{family}/{name}'
        target = ROOT / relative
        if target.exists() and target.read_bytes() != payload:
            raise ValueError('Existing frozen input differs: ' + str(target))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        hashes[relative] = hashlib.sha256(payload).hexdigest()
    # Match the existing campaign build definition, allowing shared compatible
    # native modules rather than a second build for the unchanged N32 source.
    build = {'inputs': {'native/CMakeLists.txt': hashlib.sha256((ROOT / 'experiments/native/CMakeLists.txt').read_bytes()).hexdigest()},
        'commands': [
            ['cmake', '-S', '{runner}/native', '-B', '{build}', '-DGATO_SOURCE_DIR={source}',
             '-DKNOTS=16;32;64;128', '-DCMAKE_BUILD_TYPE=Release', '-DGATO_EXPERIMENT_TELEMETRY=ON',
             '-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}-real', '-DPython3_EXECUTABLE={python}'],
            ['cmake', '--build', '{build}', '--parallel', '2']],
        'artifacts': [f'modules/bsqpN{n}_tiago_right*.so' for n in (16, 32, 64, 128)]}
    experiment = dict(id='F01c', research_question='RQ-timing', source_commit=NATIVE_SOURCE,
        protocol=PROTOCOL, timing_mode='A', backend='internal', machines=['rtx5060'],
        original_collector_source_commit=COLLECTOR_SOURCE,
        description='Same 16 candidates across B1/2/4/8/16; retained1070 complete, Jetson execution separate.',
        build=build, inputs=hashes,
        preflight={'group':'F01c-A', 'command':[
            '{python}', '{runner}/adapters/task.py', '--source', '{source}', '--build', '{build}',
            '--output', '{output}', '--inputs', '{inputs}/experiments/inputs/t01', '--preflight', '--timing-mode', 'A']},
        command=['{python}', '{runner}/adapters/f01c.py', '--source', '{source}', '--build', '{build}',
                 '--case', '{case}', '--output', '{output}', '--inputs', '{inputs}/experiments/inputs/f01c'],
        cases=[dict(id=f'N32-B{B}-SQP{SQP}-PCG{PCG}', parameters=dict(N=32, B=B, SQP=SQP, PCG=PCG,
                diagnostics=(B, SQP, PCG)==(16, 5, 100), timing_mode='A', backend='internal'))
            for (SQP, PCG), B in itertools.product(((1, 100), (5, 100), (10, 800)), (1, 2, 4, 8, 16))])
    plan = dict(version=1, experiments=[experiment], retained_results={
        'F01c': {'gtx1070':'15 completed reviewed cells; preserve/import historical results',
                 'jetson':'Not selected here; retained duration estimate/protocol and target preparation remain separate'}},
        note='Development-only plan; no native build or GPU measurements performed by generation. Default campaign unchanged.')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2)+'\n')
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'experiments/f01c.json')
    args = parser.parse_args()
    prepare(args.output)
    print(args.output)
