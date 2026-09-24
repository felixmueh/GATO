"""Collect the bounded local numerical pilot without dropping failed cells."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected(cell, batch):
    branch = cell['branches'][batch]
    lane = branch['lanes'][branch['winner']]
    return dict(batch=int(batch), winner=branch['winner'], coarse=lane['coarse'],
                fine=lane['fine'], planned_replay=lane['planned_replay'],
                initial_local_defect=lane.get('initial_local_defect'),
                planned_local_defect=lane.get('planned_local_defect'),
                initial_xu_sha256=branch.get('initial_xu_sha256'),
                telemetry=branch['telemetry'], solve_ms=branch['solve_ms'])


def collect(root):
    baseline=root/'tiago_horizon'
    result=dict(schema=1, scope='Bounded development pilot; three fixed arm tasks, '
        'sampled tool-cylinder and joint/effort checks, independent continuous simulation; '
        'no real-time, whole-body, global-optimum, or KKT-convergence claim.',
        raw_root=str(root), openloop=json.loads((baseline/'validation_table.json').read_text())['rows'],
        fixed_dt=[], rescue=[], mpc=[], artifacts=[])
    prefixes=json.loads((baseline/'prefix_validation.json').read_text())
    result['prefix_refinement']=[dict(cell=r['cell'],batch=int(b),
        step_halving_terminal=v['prefix'].get('step_halving_terminal'),
        clock=v['prefix'].get('clock')) for r in prefixes for b,v in r['branches'].items()]
    directories=[baseline,root/'tiago_horizon_fixed_dt_pilot',root/'tiago_horizon_rescue_sqp800',
                 root/'tiago_horizon_rescue_pcg4000',root/'tiago_horizon_mpc']
    for directory in directories:
        for path in sorted(directory.glob('*/summary.json')):
            data=json.loads(path.read_text())
            result['artifacts'].append(dict(path=str(path),sha256=digest(path),
                provenance=data.get('provenance'),machine=data.get('machine')))
            if directory==baseline:continue
            if directory.name=='tiago_horizon_mpc':
                for batch,b in data['branches'].items():
                    stages=[s for s in b['stages'] if 'winner' in s]
                    row=dict(case=path.parent.name,knots=data['config']['knots'],
                        duration=data['config']['duration'],dt=data['config']['dt'],
                        clock=data['clock'],initialization=data['initialization'],batch=int(batch),
                        executed=b['executed'],complete=b['complete'],failure=b['failure'],
                        simulated_seconds=b['simulated_seconds'],steps=len(b['stages']),
                        full_coarse_feasible=sum(s['lanes'][s['winner']]['coarse']['feasible'] for s in stages),
                        full_fine_feasible=sum(s['lanes'][s['winner']]['fine']['feasible'] for s in stages),
                        solve_ms=sum(s['solve_ms'] for s in stages),
                        kkt=sum(s['telemetry'][-1]['kkt'][s['winner']] for s in stages),
                        max_prefix_tool_error=max((s['applied_prefix']['tool_error_max'] for s in stages if 'applied_prefix' in s),default=None),
                        initial_local_defects=[s['lanes'][s['winner']].get('initial_local_defect') for s in stages],
                        initial_xu_sha256=[s.get('initial_xu_sha256') for s in stages])
                    result['mpc'].append(row)
            else:
                key='fixed_dt' if 'fixed_dt' in directory.name else 'rescue'
                for batch in data.get('branches',{}):
                    result[key].append(dict(case=directory.name,cell=data['cell'],config=data['config'],
                        failure=data.get('failure'),**selected(data,batch)))
    result['expected_counts']=dict(openloop=36,fixed_dt=3,rescue=2,mpc=6)
    result['observed_counts']={k:len(result[k]) for k in result['expected_counts']}
    result['all_expected_present']=result['observed_counts']==result['expected_counts']
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('example_artifacts/randomized_multimodal'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=collect(args.root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['observed_counts']))
