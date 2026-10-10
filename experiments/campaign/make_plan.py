"""Generate explicit release plans after reviewed commits; never edit the legacy queue."""
import argparse
import json
from pathlib import Path
import re
import subprocess

from raw import sha, write_json

ROOT = Path(__file__).resolve().parents[2]
REQUIRED = ['experiments/campaign/run_v01.py', 'experiments/campaign/raw.py',
            'experiments/campaign/t01_protocol.py',
            'experiments/campaign/evaluate.py', 'experiments/campaign/build_isolated.py',
            'experiments/campaign/inputs/v01.json', 'experiments/campaign/mode_b_protocol.json',
            'python/bsqp/offline_mpc.py']


def make_plan(commit, machines=('gtx1070', 'rtx5060'), enabled=False, source=ROOT):
    if not re.fullmatch('[0-9a-f]{40}', commit) or commit.startswith('7187cf8'):
        raise ValueError('Use full reviewed campaign commit; legacy7187cf8 is prohibited')
    if not machines or set(machines)-{'gtx1070', 'rtx5060'}:
        raise ValueError('Jetson deployment held for duration/storage discussion; desktop only here')
    source = Path(source)
    if enabled and json.loads((source/'experiments/campaign/inputs/v01.json').read_text()).get('release_status') != 'released after reviewed clock decision':
        raise ValueError('B release held: implement/review/validate approved old-plan clock before enabling')
    input_paths = ['experiments/campaign/inputs/v01.json', 'experiments/campaign/mode_b_protocol.json',
                   'tests/python/tiago_task_cases/short_reach.json']
    for relative in REQUIRED+input_paths:
        committed = subprocess.check_output(['git', '-C', str(source), 'show', f'{commit}:{relative}'])
        if committed != (source/relative).read_bytes():
            raise ValueError(f'Reviewed commit bytes differ from current deployment input/script: {relative}')
    plan = {'version':1, 'experiments':[{
        'id':'campaign-V01-internal-B', 'source_commit':commit, 'enabled':bool(enabled),
        'machines':list(machines), 'repeats':1,
        'inputs':{relative:sha(source/relative) for relative in input_paths},
        'build':{'commands':[['{python}', '{source}/experiments/campaign/build_isolated.py',
                   '--source', '{source}', '--build', '{build}', '--python', '{python}', '--cuda-arch', '{cuda_arch}']],
                 'artifacts':['campaign-source/python/bsqp/bsqpN32_tiago_right*.so',
                              'campaign-native/cuda_access_smoke', 'build_receipt.json', 'build_source_receipt.json']},
        'command':['{python}', '{build}/campaign-source/experiments/campaign/run_v01.py',
                   '--case','{case}', '--output','{output}', '--inputs','{inputs}'],
        'cases':[{'id':'rough-reaching', 'parameters':{'protocol':'V01', 'offline_timing':'solve_time'}}]
    }], 'release_notes':{'task_suite':'required make test-tiago-tasks on exact reviewed source/module before enabling',
        'mode_B_semantics':'reviewed trueB executes old active plan during current measured solve then activates returned plan;103CPU and original-modeMake3/3 independently accepted; final V01 release preparation review/checkpoint/exactplan dryrun required before root measurement; explicit solve_time_legacy remains separate',
        'diagnostic_A':'not queued; generate a separate hashed same-machine failed-B evidence case only after completed B task failure',
        'T01':'not queued by this V01 plan; approved3ms-first goals/speeds with separate conditional1ms cohort; provisional gates frozen for simulation-result review beforeD01/D02; reviewed task-worker/protocol release and matching V01 preflight remain separate',
        'Jetson':'not queued; discuss retained bottlenecks, duration/storage and shorter protocol before execution',
        '5060':'internal protocol matches1070; Gazebo separate and not queued'}}
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--enable-reviewed', action='store_true')
    args = parser.parse_args()
    write_json(args.output, make_plan(args.source_commit, enabled=args.enable_reviewed))
