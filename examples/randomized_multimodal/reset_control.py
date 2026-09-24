"""Posthoc fresh-nominal B1 control on the frozen pointmass switch states.

Runs one deterministic reset per scene. No retuning, offline optimization, or
changes to the frozen campaign; this diagnoses an inexpensive alternative to
retaining the previous local solution after the prescribed obstacle jump.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np

from experiment import Gato
from model import Config, seeds, jsonable
from summarize import validate_grid, audit_common_state, finite_stats, wilson


def cost_comparison(reset, baseline):
    """Feasibility first; numerical cost ties use 1e-6 absolute/relative slack."""
    if reset['feasible'] != baseline['feasible']:
        outcome='win' if reset['feasible'] else 'loss'
    elif not reset['feasible']:
        outcome='both_infeasible'
    else:
        slack=1e-6*max(1.,abs(reset['cost']),abs(baseline['cost']))
        difference=baseline['cost']-reset['cost']
        outcome='win' if difference>slack else 'loss' if difference < -slack else 'tie'
    return dict(outcome=outcome, reset_cost=reset['cost'], baseline_cost=baseline['cost'],
                relative_saving=(baseline['cost']-reset['cost'])/baseline['cost']
                    if reset['feasible'] and baseline['feasible'] and baseline['cost'] else None)


def aggregate(comparisons):
    return dict(comparisons=len(comparisons),
        outcomes={key:sum(c['outcome']==key for c in comparisons) for key in ('win','loss','tie','both_infeasible')},
        relative_saving=finite_stats([c['relative_saving'] for c in comparisons]))


def run(args):
    source=args.campaign.resolve(); directory=args.output.resolve()
    if directory == source.parent:
        raise ValueError('Posthoc results must be written to a separate directory')
    payload=json.loads(source.read_text())
    manifest=json.loads(source.with_name('manifest.json').read_text())
    validate_grid(payload,manifest)
    parity=audit_common_state(payload['changed_state_runs'])
    cfg=Config(**payload['config'])
    tasks={t['scene_id']:t for t in payload['scenes']}
    ids=sorted(tasks)
    gato=Gato(cfg,args.extension_dir)
    results=[]
    for index,sid in enumerate(ids):
        rows=[r for r in payload['changed_state_runs'] if r['scene_id']==sid]
        inherited=sorted((r for r in rows if r['batch_size']==1),key=lambda r:r['proposal_seed'])
        batched=sorted((r for r in rows if r['batch_size']==16),key=lambda r:r['proposal_seed'])
        base=inherited[0]
        # Identical inputs do not imply bit-identical capped GPU outputs.
        # Prespecify the lowest proposal seed as comparator and retain all costs.
        task=dict(tasks[sid],x0=np.asarray(base['x0']),center=np.asarray(base['center']))
        _,initial=seeds(task['x0'],task['goal'],cfg.knots,cfg.dt,1,0,cfg.acceleration_limit)
        if index==0:
            gato.solve(task,initial)  # excluded warm-up; no measured outcome selection
        answer=gato.solve(task,initial)
        best=base['best_known_cost']
        answer['near_best']=bool(answer['feasible'] and best is not None and answer['cost']<=best*(1+cfg.near_best_relative))
        answer['best_known_cost']=best
        answer['relative_regret']=answer['cost']/best-1 if best else None
        answer['scene_id']=sid
        answer['x0']=task['x0']; answer['center']=task['center']
        answer['inherited_comparison']=cost_comparison(answer,base)
        answer['inherited_costs']=[r['cost'] for r in inherited]
        answer['inherited_near_best']=base['near_best']
        answer['batch16_comparisons']=[dict(proposal_seed=r['proposal_seed'],
            near_best=r['near_best'],**cost_comparison(answer,r)) for r in batched]
        results.append(answer)
        print(f"reset scene={sid} feasible={answer['feasible']} near_best={answer['near_best']} "
              f"cost={answer['cost']:.8f} inherited={answer['inherited_comparison']['outcome']}",flush=True)
    successes=sum(r['near_best'] for r in results)
    inherited_pairs=[r['inherited_comparison'] for r in results]
    batch_pairs=[c for r in results for c in r['batch16_comparisons']]
    scene_batch_savings=[np.mean([c['relative_saving'] for c in r['batch16_comparisons']
                         if c['relative_saving'] is not None]) for r in results]
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    summary=dict(scope='Posthoc diagnostic, not part of the frozen campaign: one fresh nominal B1 reset per scene at the identical frozen switch state.',
        scenes=len(results), reset_feasible=sum(r['feasible'] for r in results),
        reset_near_best=successes, reset_near_best_wilson95=wilson(successes,len(results)),
        inherited_near_best=sum(r['inherited_near_best'] for r in results),
        inherited_duplicate_cost_spread=finite_stats([max(r['inherited_costs'])-min(r['inherited_costs']) for r in results]),
        batch16_near_best=sum(c['near_best'] for c in batch_pairs),batch16_solves=len(batch_pairs),
        reset_vs_inherited=aggregate(inherited_pairs),reset_vs_batch16_each_draw=aggregate(batch_pairs),
        reset_vs_batch16_scene_mean_saving=finite_stats(scene_batch_savings),
        reset_relative_regret=finite_stats([r['relative_regret'] for r in results]),
        config=asdict(cfg),common_state_audit=parity,
        cost_tie_tolerance='1e-6 times max(1, absolute compared costs)',
        notes=['Positive saving and win mean the fresh reset is cheaper; negative saving means batching is cheaper.',
               'The 200 B16 comparisons reuse 40 reset outcomes; they are not 200 independent reset trials.',
               'The nominal reset is obstacle-blind and deterministic. No settings were selected using this diagnostic.',
               'One excluded warm-up precedes 40 recorded solves. Physical feasibility and objective use independent raw-control replay.'],
        provenance=dict(campaign=str(source),campaign_sha256=sha(source),manifest=manifest,
                        helper_sha256=sha(Path(__file__)),
                        loaded_extension_sha256=sha(Path(gato.module.__file__))),records=results)
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'summary.json').write_text(json.dumps(jsonable(summary),indent=2,allow_nan=False)+'\n')
    a,b=summary['reset_vs_inherited'],summary['reset_vs_batch16_each_draw']
    report=f"""# Posthoc fresh-nominal reset control

One fresh nominal B1 solve at each of the 40 frozen switch states, with the same objective, 80-SQP budget and 3% near-reference threshold. This diagnostic was added after inspecting the frozen campaign.

| Method | Near-best feasible outcomes | Independent scenes |
|---|---:|---:|
| Inherited warm B1 | {summary['inherited_near_best']}/{len(results)} | {len(results)} |
| Fresh nominal B1 reset | {successes}/{len(results)} | {len(results)} |
| Frozen B16 | {summary['batch16_near_best']}/{len(batch_pairs)} | {len(results)} |

Reset feasibility: {summary['reset_feasible']}/{len(results)}. Reset near-best Wilson 95% interval: {100*summary['reset_near_best_wilson95'][0]:.1f}–{100*summary['reset_near_best_wilson95'][1]:.1f}%.

Reset versus inherited B1: {a['outcomes']}; mean relative cost saving {100*a['relative_saving']['mean']:.3f}%.

Reset versus every frozen B16 draw: {b['outcomes']}; mean relative cost saving {100*summary['reset_vs_batch16_scene_mean_saving']['mean']:.3f}%. These 200 comparisons reuse 40 deterministic reset outputs.

Positive savings favor reset; negative savings favor batching. All raw outputs, comparisons, thresholds and provenance are in summary.json. The original frozen campaign is unchanged.
"""
    (directory/'summary.md').write_text(report)
    print(json.dumps(jsonable({k:v for k,v in summary.items() if k not in ('records','provenance')}),indent=2,allow_nan=False))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,default=Path('example_artifacts/randomized_multimodal/pointmass_holdout/campaign.json'))
    parser.add_argument('--output',type=Path,default=Path('example_artifacts/randomized_multimodal/pointmass_reset_posthoc'))
    parser.add_argument('--extension-dir',default='python/bsqp')
    run(parser.parse_args())
