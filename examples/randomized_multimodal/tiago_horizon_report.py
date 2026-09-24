"""Build a reviewable per-cell table without hiding failed horizon cases."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from tiago_probe import json_safe


def report(root):
    prefixes={r['cell']:r for r in json.loads((root/'prefix_validation.json').read_text())} if (root/'prefix_validation.json').exists() else {}
    rows=[]
    for path in sorted(root.glob('*/summary.json')):
        cell=json.loads(path.read_text());cfg=cell.get('config',{})
        for batch,result in cell.get('branches',{'-':{}}).items():
            base=dict(cell=cell['cell'],task=cell.get('scene',{}).get('index'),knots=cfg.get('knots'),duration=cfg.get('duration'),dt=cfg.get('dt'),batch=batch,
                failure=cell.get('failure'),rho=cfg.get('rho'),sqp_budget=cfg.get('iters',0)*cfg.get('passes',0))
            if not result:rows.append(base);continue
            chosen=result['lanes'][result['winner']];coarse=chosen['coarse']
            validation=prefixes.get(cell['cell'],{}).get('branches',{}).get(batch,{})
            fine=validation.get('aligned_fine',chosen['fine']);prefix=validation.get('prefix',{})
            refinements=validation.get('aligned_fine_halving',{})
            reference=result.get('cpu_reference',{});winner=result['winner']
            base.update(winner=winner,coarse_feasible=coarse['feasible'],fine_full_feasible=fine.get('feasible',False),
                prefix_physical=prefix.get('physical'),prefix_tool_error_mm=1000*prefix.get('tool_error_max',float('nan')),
                prefix_velocity_error=prefix.get('coarse_fine',{}).get('v'),
                coarse_goal_mm=1000*coarse['terminal'] if coarse.get('terminal') is not None else None,
                fine_goal_mm=1000*fine['terminal'] if fine.get('terminal') is not None else None,
                fine_velocity_ratio=fine.get('velocity_ratio'),fine_joint_violation=fine.get('joint_violation'),
                coarse_objective=coarse['objective']['total'],fine_objective=fine.get('objective',{}).get('total'),
                planned_replay_q=chosen['planned_replay']['q'],planned_replay_v=chosen['planned_replay']['v'],
                fine_step_halving_q=refinements.get('q'),fine_step_halving_v=refinements.get('v'),
                kkt=result['telemetry'][-1]['kkt'][winner],
                accepted_steps=sum(t['accepted'][winner] for t in result['telemetry']),
                last_pass_accepted=result['telemetry'][-1]['accepted'][winner],
                pcg_caps=sum(t['pcg_cap'][winner] for t in result['telemetry']),
                pcg_breakdown=sum(t['pcg_breakdown'][winner] for t in result['telemetry']),
                cpu_coarse_feasible=reference.get('coarse',{}).get('feasible'),
                cpu_fine_feasible=reference.get('fine',{}).get('feasible'),cpu_gradient_inf=reference.get('gradient_inf'),
                cpu_objective=reference.get('final_cost'),solve_ms=result['solve_ms'])
            rows.append(base)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (root/'validation_table.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    summary=dict(scope='Open-loop fidelity and .06s applied-prefix validation are separate; neither alone proves closed-loop viability.',rows=rows)
    (root/'validation_table.json').write_text(json.dumps(json_safe(summary),indent=2,allow_nan=False))
    print(root/'validation_table.csv')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);report(p.parse_args().root)
