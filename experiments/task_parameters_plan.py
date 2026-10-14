#!/usr/bin/env python3
"""Generate the frozen task-parameter plans; this script does not execute studies."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
INTERNAL = 'fa6f6bede286c122cc41ac65f42a583da84290e0'
COHORT = 'task-parameters-20261014'
TASK_NAMES = {'goal-3': '317 mm reach', 'figure8-6s': '6-second figure-eight',
              'figure8-12s': '12-second figure-eight'}


def plans(collector, live):
    if any(not re.fullmatch('[0-9a-f]{40}', pin) for pin in (collector, live)):
        raise ValueError('Full immutable collector and live-runtime pins required')
    baseline = json.loads((HERE/'launch/rtx5060.json').read_text())
    recipe = copy.deepcopy(next(e for e in baseline['experiments'] if e['id'] == 'F01ab')['build'])
    inputs = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
        'experiments/inputs/t01/t01.json', 'experiments/inputs/t01/v01.json',
        'experiments/inputs/task_parameters/canonical_case.json')}
    result = {}
    for setting, machine, backend in (('gtx1070', 'gtx1070', 'internal'),
            ('rtx5060_internal', 'rtx5060', 'internal'), ('jetson', 'jetson', 'internal'),
            ('rtx5060_gazebo', 'rtx5060', 'gazebo')):
        tasks = ['goal-3', 'figure8-12s'] if backend == 'gazebo' else list(TASK_NAMES)
        experiments = []
        def experiment(eid, study, N, dt, sqp, selected_tasks, repeats=1):
            argv = ['{python}', '{runner}/adapters/task_parameters.py', '--source', '{source}',
                '--build', '{build}', '--inputs', '{inputs}/experiments/inputs/t01',
                '--case', '{case}', '--output', '{output}', '--backend', backend,
                '--collector-pin', collector]
            preflight = list(argv)
            preflight[preflight.index('{case}')] = '{inputs}/experiments/inputs/task_parameters/canonical_case.json'
            preflight += ['--preflight']
            e = dict(id=eid, name=study, source_commit=live if backend == 'gazebo' else INTERNAL,
                collector_source_commit=collector, protocol=COHORT, cohort_id=COHORT,
                research_question='RQ-control', timing_mode='live' if backend == 'gazebo' else 'B',
                backend=backend, machines=[machine], build=copy.deepcopy(recipe), inputs=dict(inputs),
                parameters={'collector_source_commit': collector, 'pcg_max_iter': 120},
                command=argv, preflight={'command': preflight, 'group': COHORT+'-'+backend},
                cases=[dict(id=t, name=TASK_NAMES[t], parameters=dict(N=N, dt_s=dt,
                    sqp_max_iter=sqp, pcg_max_iter=120)) for t in selected_tasks], repeats=repeats,
                timeout_seconds=1800)
            if backend == 'gazebo':
                e['allow_report_only_safety'] = True
            return e
        for N in (16, 32, 64, 128):
            for sqp in (1, 2, 5):
                eid = f'A-N{N}-SQP{sqp}'
                if backend == 'gazebo' and N == 32 and sqp == 5:
                    experiments += [experiment(eid+'-reach', 'Horizon length and solver budget', N, .03,
                        sqp, ['goal-3'], 3), experiment(eid+'-tracking',
                        'Horizon length and solver budget', N, .03, sqp, ['figure8-12s'], 2)]
                else:
                    experiments.append(experiment(eid, 'Horizon length and solver budget', N, .03,
                        sqp, tasks, 3 if N == 32 and sqp == 5 else 1))
        for N in (16, 32, 64, 128):
            for dt in (.010, .015, .020, .045, .060):
                experiments.append(experiment(f'B-N{N}-dt{round(dt*1000):02d}ms',
                    'Knot spacing and horizon length', N, dt, 5,
                    ['figure8-12s'] if backend == 'gazebo' else ['figure8-6s', 'figure8-12s']))
        plan = dict(version=1, status='prepared_not_executed', name='Task parameter studies',
            cohort_id=COHORT, setting=setting, scientific_episodes=sum(len(e['cases'])*e['repeats'] for e in experiments),
            readiness='Separate fixed N32/30 ms/SQP5/PCG120 canonical preflight; not a scientific grid cell',
            study_B_30ms_reuse='Use Study A N16/32/64/128, SQP5 tracking cases; no additional episodes',
            historical_Gazebo_anchor='Guarded D01 is historical context, not a fresh report-only repetition',
            failure_policy='Completed task/physical negatives remain evidence; independent cells continue. Shared setup or canonical task/execution/handback failure holds dependent cases.',
            experiments=experiments)
        pilot = copy.deepcopy(plan)
        pilot['name'] = 'Readiness and cost pilots (reuse identical science cells)'
        pilot['status'] = 'prepared_pilots_before_full_queue'
        if backend == 'gazebo':
            pilot['experiments'] = [copy.deepcopy(next(e for e in experiments if e['id'] == 'A-N128-SQP5'))]
            pilot['experiments'][0]['cases'] = [c for c in pilot['experiments'][0]['cases'] if c['id'] == 'figure8-12s']
        else:
            pilot['experiments'] = [copy.deepcopy(next(e for e in experiments if e['id'] == eid))
                for eid in ('A-N16-SQP1', 'A-N128-SQP5')]
            for e in pilot['experiments']:
                e['cases'] = [c for c in e['cases'] if c['id'] == 'figure8-6s']
        pilot['scientific_episodes'] = sum(len(e['cases'])*e['repeats'] for e in pilot['experiments'])
        result[setting] = plan
        result[setting+'_pilots'] = pilot
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collector-pin', required=True)
    parser.add_argument('--live-pin', required=True)
    args = parser.parse_args()
    destination = HERE/'launch/task_parameters'
    destination.mkdir(parents=True, exist_ok=True)
    for setting, plan in plans(args.collector_pin, args.live_pin).items():
        (destination/(setting+'.json')).write_text(json.dumps(plan, indent=2)+'\n')


if __name__ == '__main__':
    main()
