#!/usr/bin/env python3
"""Generate additive supported large-N plans without changing the base studies."""
import argparse
import copy
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
COHORT = 'task-parameters-large-optin-20261014'


def plans(collector, internal, live):
    if any(not re.fullmatch('[0-9a-f]{40}', p) for p in (collector, internal, live)):
        raise ValueError('Full immutable collector/internal/live pins required')
    result = {}
    for setting, Ns in (('rtx5060_internal', (256,)), ('rtx5060_gazebo', (256,)),
                        ('jetson', (256, 512))):
        base = json.loads((HERE/'launch/task_parameters'/(setting+'.json')).read_text())
        template = base['experiments'][0]
        gazebo = setting.endswith('gazebo')
        experiments = []
        for N in Ns:
            for study, dt, sqp in ([('A', .03, q) for q in (1, 2, 5)] +
                                 [('B', dt, 5) for dt in (.010, .015, .020, .045, .060)]):
                e = copy.deepcopy(template)
                e.update(id=f'{study}-N{N}-'+(f'SQP{sqp}' if study == 'A' else f'dt{round(dt*1000):02d}ms'),
                    name='Horizon length and solver budget' if study == 'A' else 'Knot spacing and horizon length',
                    source_commit=live if gazebo else internal, collector_source_commit=collector,
                    protocol=COHORT, cohort_id=COHORT, repeats=1)
                e['parameters']['collector_source_commit'] = collector
                e['parameters']['native_shared_memory_policy'] = 'conditional_opt_in_only_above_default_and_device_fit'
                argv = e['build']['commands'][0]
                argv[argv.index('-DKNOTS=16;32;64;128')] = f'-DKNOTS=32;{N}'
                e['build']['artifacts'] = ['modules/bsqpN32_tiago_right*.so',
                    f'modules/bsqpN{N}_tiago_right*.so', 'cuda_access_smoke']
                e['preflight']['group'] = COHORT+f'-{e["backend"]}-N{N}'
                for command in (e['command'], e['preflight']['command']):
                    command[command.index('--collector-pin')+1] = collector
                tasks = copy.deepcopy(template['cases'])
                if study == 'B':
                    tasks = [c for c in tasks if c['id'] != 'goal-3']
                for c in tasks:
                    c['parameters'] = dict(N=N, dt_s=dt, sqp_max_iter=sqp, pcg_max_iter=120)
                e['cases'] = tasks
                experiments.append(e)
        plan = dict(version=1, name='Additional supported large-horizon task cells',
            status='prepared_capacity_finite_readiness_and_pilots_pending', cohort_id=COHORT,
            setting=setting, scientific_episodes=sum(len(e['cases']) for e in experiments),
            base_study='The unchanged 293-episode N16–128 plans retain original pins/evidence; no campaign rerun',
            study_B_30ms_reuse='Use this extension Study A SQP5 tracking cases at each new N; no additional episodes',
            execution_policy='Manual capacity then finite native readiness then same-cell task pilot for each ascending N; review wall cost before full grid',
            experiments=experiments)
        pilot = copy.deepcopy(plan)
        pilot['name'] = 'Additional large-horizon task pilots (science cells reused)'
        pilot['experiments'] = [copy.deepcopy(e) for e in experiments
                               if e['id'].startswith('A-') and e['id'].endswith('SQP5')]
        target = 'figure8-12s' if gazebo else 'figure8-6s'
        for e in pilot['experiments']:
            e['cases'] = [c for c in e['cases'] if c['id'] == target]
        pilot['scientific_episodes'] = len(pilot['experiments'])
        readiness = copy.deepcopy(plan)
        readiness['name'] = 'Single-solve native readiness for additional horizons'
        readiness['scientific_episodes'] = 0
        readiness['readiness_observations'] = len(Ns)
        readiness['experiments'] = []
        for N in Ns:
            e = copy.deepcopy(next(e for e in experiments if e['id'] == f'A-N{N}-SQP1'))
            e.update(id=f'native-N{N}', name=f'{N}-knot single-solve native readiness', backend='internal', timing_mode='native_binding')
            e.pop('allow_report_only_safety', None)
            e.pop('preflight', None)
            e['command'][e['command'].index('--backend')+1] = 'internal'
            e['command'] += ['--native-readiness']
            e['cases'] = [dict(id='goal-3', name='Frozen canonical pose input, one native solve',
                parameters=dict(N=N, dt_s=.03, sqp_max_iter=1, pcg_max_iter=120))]
            readiness['experiments'].append(e)
        result[setting] = plan
        result[setting+'_pilots'] = pilot
        result[setting+'_native_readiness'] = readiness
    return result


def capacity_plans(internal):
    """Use the successor's safe resource-only probe, in independent single-N builds."""
    if not re.fullmatch('[0-9a-f]{40}', internal):
        raise ValueError('Full immutable internal opt-in source required')
    result = {}
    for machine, Ns in (('rtx5060', (256,)), ('jetson', (256, 512))):
        base = json.loads((HERE/'launch/task_parameter_capacity'/(machine+'.json')).read_text())
        plan = copy.deepcopy(base)
        plan.update(name='PCG opt-in capacity observations',
            status='prepared_optin_capacity_observations_not_executed',
            execution_policy='Repeat N256 after opt-in; Jetson N512 only after sensible N256 capacity/finite/pilot cost',
            historical_RTX_N256='Unchanged-source RTX and Jetson N256 rejections retained; these are separately pinned successor observations')
        plan['experiments'] = [e for e in plan['experiments'] if e['cases'][0]['parameters']['N'] in Ns]
        for e in plan['experiments']:
            e.update(source_commit=internal, protocol='task-parameter-capacity-optin-20261014',
                scope='Successor resource-only probe configures supported PCG opt-in and reports effective limits; zero kernel launches, no optimizer solve or motion')
        result[machine] = plan
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('collector-pin', 'internal-pin', 'live-pin'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    destination = HERE/'launch/task_parameters_large'
    destination.mkdir(parents=True, exist_ok=True)
    for setting, plan in plans(args.collector_pin, args.internal_pin, args.live_pin).items():
        (destination/(setting+'.json')).write_text(json.dumps(plan, indent=2)+'\n')
    capacity_destination = HERE/'launch/task_parameter_capacity_optin'
    capacity_destination.mkdir(parents=True, exist_ok=True)
    for machine, plan in capacity_plans(args.internal_pin).items():
        (capacity_destination/(machine+'.json')).write_text(json.dumps(plan, indent=2)+'\n')


if __name__ == '__main__':
    main()
