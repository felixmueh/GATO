"""Offline summaries of frozen campaigns; no optimization or GPU access.

python examples/randomized_multimodal/summarize.py campaign.json --plot --animation
Use --development for incomplete/development records without a frozen manifest.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def wilson(successes, count):
    """95% score interval for an explicitly binary, independent-scene event."""
    if not count:
        return [None, None]
    z = 1.959963984540054
    p = successes / count
    scale = 1 + z*z/count
    center = (p + z*z/(2*count))/scale
    half = z*math.sqrt(p*(1-p)/count + z*z/(4*count*count))/scale
    return [max(0., center-half), min(1., center+half)]


def bounded_mean(values, bounds=(0., 1.)):
    """Distribution-free 95% Hoeffding interval over independent scene units.

    Dependence among repeats within a scene is unrestricted. Conservative but
    nondegenerate even when every observed scene has the same outcome.
    """
    values = np.asarray(values, float)
    if not len(values):
        return dict(mean=None, interval95=[None, None], scenes=0)
    lo, hi = bounds
    radius = (hi-lo)*math.sqrt(math.log(40)/(2*len(values)))
    mean = float(values.mean())
    return dict(mean=mean, interval95=[max(lo, mean-radius), min(hi, mean+radius)],
                scenes=len(values), interval_method='Hoeffding over independent scene means')


def finite_stats(values):
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    return dict(n=len(values), mean=float(values.mean()) if len(values) else None,
                median=float(np.median(values)) if len(values) else None,
                min=float(values.min()) if len(values) else None,
                max=float(values.max()) if len(values) else None)


def group_scenes(rows):
    return {sid: [r for r in rows if r['scene_id'] == sid]
            for sid in sorted({r['scene_id'] for r in rows})}


def success_stats(rows, key):
    groups = group_scenes(rows)
    proportions = [np.mean([bool(r.get(key, False)) for r in rs]) for rs in groups.values()]
    robust = sum(all(bool(r.get(key, False)) for r in rs) for rs in groups.values())
    return dict(**bounded_mean(proportions), successful_solves=sum(bool(r.get(key, False)) for r in rows),
                solves=len(rows), all_proposals_success=dict(successful_scenes=robust,
                    scenes=len(groups), proportion=robust/len(groups) if groups else None,
                    interval95=wilson(robust, len(groups)), interval_method='Wilson score',
                    event='Every complete solve across repeated RNG draws succeeds; this does not require every candidate lane to succeed'))


def optimization_stats(rows):
    lanes = [lane for r in rows for lane in r.get('lanes', [])]
    improvements = [float(l['initial_cost'])-float(l['cost']) for l in lanes
                    if l.get('initial_cost') is not None and l.get('cost') is not None
                    and abs(l['cost']) < 1e20 and abs(l['initial_cost']) < 1e20]
    polishes = [r['selected_polish'] for r in rows if r.get('selected_polish')]
    return dict(lanes=len(lanes), feasible_lanes=sum(bool(l.get('feasible')) for l in lanes),
                initial_feasible_lanes=sum(bool(l.get('initial_feasible')) for l in lanes),
                raw_kkt_converged_lanes=sum(bool(l.get('kkt_converged')) for l in lanes),
                objective_decrease=finite_stats(improvements),
                objective_decreased_lanes=sum(v > 0 for v in improvements),
                objective_comparable_lanes=len(improvements),
                planned_defect=finite_stats([l.get('planned_defect') for l in lanes]),
                selected_polish_count=len(polishes),
                selected_polish_projected_gradient=finite_stats([r.get('projected_gradient_inf') for r in polishes]),
                selected_polish_relative_improvement=finite_stats([
                    (r['unpolished_cost']-r['cost'])/max(abs(r['unpolished_cost']), 1e-12) for r in polishes]),
                selected_polish_mode_retained=sum(r['selected_polish']['mode'] == r.get('mode')
                    for r in rows if r.get('selected_polish')))


def section(rows):
    result = dict(by_batch={}, paired_gain_vs_b1={}, per_scene={}, failed_trials=[
        {key: r.get(key) for key in ('scene_id', 'proposal_seed', 'batch_size', 'failure',
            'terminal_error', 'min_clearance', 'max_acceleration', 'max_velocity')}
        for r in rows if not r.get('feasible', False)])
    for b in sorted({r['batch_size'] for r in rows}):
        selected = [r for r in rows if r['batch_size'] == b]
        both = [dict(r, both_modes={'cw', 'ccw'}.issubset(
            {l.get('mode') for l in r.get('lanes', []) if l.get('feasible')})) for r in selected]
        result['by_batch'][str(b)] = dict(feasible=success_stats(selected, 'feasible'),
            near_best=success_stats(selected, 'near_best'), both_modes=success_stats(both, 'both_modes'),
            optimization=optimization_stats(selected), solve_ms=finite_stats([r.get('solve_ms') for r in selected]))
        for sid, group in group_scenes(selected).items():
            regrets = [(r['cost']/r['best_known_cost']-1) for r in group
                       if r.get('feasible') and r.get('cost') is not None and r.get('best_known_cost')]
            result['per_scene'].setdefault(str(sid), {})[str(b)] = dict(solves=len(group),
                failures=sum(not r.get('feasible', False) for r in group),
                near_best_rate=float(np.mean([bool(r.get('near_best', False)) for r in group])),
                feasible_regret=finite_stats(regrets),
                modes=sorted({l.get('mode') for r in group for l in r.get('lanes', []) if l.get('feasible')}))
    baseline = {(r['scene_id'], r['proposal_seed']): r for r in rows if r['batch_size'] == 1}
    for b in sorted({r['batch_size'] for r in rows} - {1}):
        pairs = [(r, baseline[(r['scene_id'], r['proposal_seed'])]) for r in rows
                 if r['batch_size'] == b and (r['scene_id'], r['proposal_seed']) in baseline]
        gains = {}
        for key in ('feasible', 'near_best'):
            scene_gains = {}
            for r, base in pairs:
                scene_gains.setdefault(r['scene_id'], []).append(int(bool(r.get(key)))-int(bool(base.get(key))))
            gains[key] = dict(**bounded_mean([np.mean(v) for v in scene_gains.values()], (-1., 1.)),
                             paired_solves=len(pairs))
        result['paired_gain_vs_b1'][str(b)] = gains
    return result


def validate_grid(payload, manifest):
    args = manifest['arguments']
    scenes, proposals, batches = args['scene_seeds'], args['proposal_seeds'], args['batches']
    counts = {}
    for key in ('runs', 'random_single_runs', 'changed_state_runs', 'mpc'):
        if key == 'random_single_runs' and not args.get('random_single'): continue
        if key == 'changed_state_runs' and not args.get('changed_state'): continue
        if key == 'mpc' and not args.get('mpc'): continue
        bs = [1] if key == 'random_single_runs' else sorted({1, max(batches)}) if key == 'mpc' else batches
        ps = proposals[:1] if key == 'mpc' else proposals
        expected = {(s, p, b) for s in scenes for p in ps for b in bs}
        actual = [(r['scene_id'], r['proposal_seed'], r['batch_size']) for r in payload.get(key, [])]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError(f'{key}: incomplete/duplicate campaign: {len(actual)} rows; expected {len(expected)}; '
                             f'missing {len(expected-set(actual))}, extra {len(set(actual)-expected)}')
        counts[key] = len(actual)
    return counts


def audit_common_state(rows):
    checked = 0
    for sid, group in group_scenes(rows).items():
        available = [r for r in group if r.get('initial_trajectories') is not None]
        if not available:
            continue
        first = available[0]
        for row in available:
            for key in ('x0', 'center'):
                np.testing.assert_allclose(row[key], first[key], rtol=0, atol=1e-12,
                                           err_msg=f'scene {sid}: different {key}')
            np.testing.assert_allclose(row['initial_trajectories'][0], first['initial_trajectories'][0],
                                       rtol=0, atol=1e-12, err_msg=f'scene {sid}: different warm lane')
            if row.get('best_known_cost') != first.get('best_known_cost'):
                raise ValueError(f'scene {sid}: different common-state reference costs')
            checked += 1
    return dict(checked_rows=checked, unavailable_rows=len(rows)-checked,
                checks='Identical initial state, active obstacle center, warm lane 0, reference cost within scene')


def summarize(payload, manifest=None):
    counts = validate_grid(payload, manifest) if manifest else {k: len(payload.get(k, []))
        for k in ('runs', 'random_single_runs', 'changed_state_runs', 'mpc')}
    result = dict(status='Frozen held-out campaign' if manifest else 'Development observations',
        common_state_audit=audit_common_state(payload.get('changed_state_runs', [])),
        row_counts=counts, config=payload['config'],
        interpretation=[
            'Scene means weight each independently generated scene equally; proposal repeats are clustered.',
            'Inference concerns the specified scene distribution and fixed proposal-seed schedule, not arbitrary robot tasks.',
            'B1 warm and nominal baselines repeat the same deterministic solve; repeats add no independent scenes.',
            'The every-repeated-solve success event requires success of each complete solve across the tested RNG draws (five in the frozen campaign). It does not require all candidate lanes within a multistart solve to succeed.',
            'Intervals are pointwise, not simultaneous across batches. Hoeffding intervals are deliberately conservative.',
            'Near-best means feasible and within the configured relative tolerance of an offline best-known reference; no global-optimum proof.',
            'Raw GATO paths are finite-budget candidates, not certified stationary local optima. Some unselected lanes can improve substantially under offline polishing; selected-lane diagnostics do not establish stationarity of every lane.',
            'Common-state decisions share B1 switch state and warm lane 0. MPC policies evolve different states and are reported separately.',
            'Failure rows remain in success denominators; numerical cost/gradient summaries exclude unavailable values.',
            'MPC timing is simulated; solve milliseconds are observations, not a real-time performance claim.'],
        changed_state=section(payload.get('changed_state_runs', [])),
        static_control=section(payload.get('runs', [])), random_single=section(payload.get('random_single_runs', [])), mpc={})
    episodes = payload.get('mpc', [])
    for b in sorted({r['batch_size'] for r in episodes}):
        rows = [r for r in episodes if r['batch_size'] == b]
        decisions = [dict(r, near_best=bool((r.get('changed_decision') or {}).get('near_best'))) for r in rows]
        result['mpc'][str(b)] = dict(completed=success_stats(rows, 'completed'), feasible=success_stats(rows, 'feasible'),
            own_state_changed_near_best=success_stats(decisions, 'near_best'),
            feasible_realized_cost=finite_stats([r.get('cost') for r in rows if r.get('feasible')]),
            all_finite_realized_cost=finite_stats([r.get('cost') for r in rows]),
            per_scene=[dict(scene_id=r['scene_id'], completed=r.get('completed', False),
                feasible=r.get('feasible', False), cost=r.get('cost'), failure=r.get('failure'),
                min_clearance=r.get('min_clearance')) for r in rows])
    by_scene = {(r['scene_id'], r['batch_size']): r for r in episodes}
    maximum = max((r['batch_size'] for r in episodes), default=1)
    savings = []
    for (sid, b), r in by_scene.items():
        base = by_scene.get((sid, 1))
        if b == maximum and b != 1 and base and r.get('feasible') and base.get('feasible'):
            savings.append((base['cost']-r['cost'])/base['cost'])
    result['mpc_paired_relative_cost_saving_on_both_feasible'] = dict(batch=maximum, **finite_stats(savings))
    return result


def pct(value):
    return 'n/a' if value is None else f'{100*value:.1f}%'


def markdown(summary):
    lines = [f"# {summary['status']}", '',
        'Primary comparison: identical state and objective after the fixed obstacle change. Raw GATO outputs determine success; offline polishing is diagnostic only.', '']
    for name, title in [('changed_state', 'Common-state changed-obstacle decisions'),
                        ('static_control', 'Static nominal control'), ('random_single', 'Static random single-start control')]:
        if not summary[name]['by_batch']:
            continue
        lines += [f'## {title}', '']
        if name == 'random_single':
            lines += ['This control optimizes one obstacle-blind random initialization per RNG draw. The nominal B1 control above uses the deterministic minimum-energy initialization; these are different baselines.', '']
        lines += ['| B | Near-best solves | Independent scenes | Mean success [95% interval] | Every repeated solve succeeds: scenes [Wilson 95%] |',
                  '|---:|---:|---:|---|---|']
        for b, row in summary[name]['by_batch'].items():
            s=row['near_best']; robust=s['all_proposals_success']; lo, hi=s['interval95']; rlo, rhi=robust['interval95']
            lines.append(f"| {b} | {s['successful_solves']}/{s['solves']} | {s['scenes']} | {pct(s['mean'])} [{pct(lo)}, {pct(hi)}] | {robust['successful_scenes']}/{robust['scenes']} [{pct(rlo)}, {pct(rhi)}] |")
        lines += ['', 'Mean-success intervals use independent-scene Hoeffding bounds. The Wilson interval describes scenes where every complete solve across the repeated RNG draws succeeds, not success of every candidate lane.', '']
        for b, gains in summary[name]['paired_gain_vs_b1'].items():
            g=gains['near_best']; lo,hi=g['interval95']
            lines.append(f"B={b} minus B1 paired near-best gain: {100*g['mean']:.1f} percentage points, 95% interval [{100*lo:.1f}, {100*hi:.1f}] across {g['scenes']} scenes.")
        lines.append('')
    lines += ['## Common-state optimization diagnostics', '',
        '| B | Feasible outputs | Both route modes in batch | Raw KKT converged lanes | Maximum selected polish cost decrease | Maximum projected gradient after polish |',
        '|---:|---:|---:|---:|---:|---:|']
    for b, row in summary['changed_state']['by_batch'].items():
        f=row['feasible']; modes=row['both_modes']; opt=row['optimization']
        grad=opt['selected_polish_projected_gradient']['max']
        grad_text='n/a' if grad is None else f'{grad:.3g}'
        lines.append(f"| {b} | {f['successful_solves']}/{f['solves']} | {modes['successful_solves']}/{modes['solves']} | {opt['raw_kkt_converged_lanes']}/{opt['lanes']} | {pct(opt['selected_polish_relative_improvement']['max'])} | {grad_text} |")
    lines += ['', 'Both-mode coverage counts feasible cw and ccw candidates within a batch. Low polishing gradients diagnose approximate stationarity, not certified local or global optimality.', '']
    if summary['changed_state']['by_batch']:
        largest=max(summary['changed_state']['by_batch'], key=int)
        opt=summary['changed_state']['by_batch'][largest]['optimization']
        lines += [f"At B={largest}, objective decreased in {opt['objective_decreased_lanes']}/{opt['objective_comparable_lanes']} comparable lanes. "
            f"Selected-solution polishing retained the route mode in {opt['selected_polish_mode_retained']}/{opt['selected_polish_count']} cases. "
            "Full decrease distributions and per-scene worst feasible regret are in summary.json.", '']
    lines += ['## Executed MPC episodes', '', '| B | Feasible completed episodes | Mean feasible realized cost | Own-state switch near-best |', '|---:|---:|---:|---:|']
    for b, row in summary['mpc'].items():
        f=row['feasible']; d=row['own_state_changed_near_best']; cost=row['feasible_realized_cost']['mean']
        lines.append(f"| {b} | {f['successful_solves']}/{f['solves']} | {cost:.6g} | {d['successful_solves']}/{d['solves']} |" if cost is not None else f"| {b} | {f['successful_solves']}/{f['solves']} | n/a | {d['successful_solves']}/{d['solves']} |")
    lines += ['', 'MPC policies have different states after execution; these episode outcomes are separate from the common-state experiment.', '', '## Interpretation', '']
    lines += ['- '+s for s in summary['interpretation']]
    lines += ['', 'Per-scene means/worst feasible regrets, mode coverage, objective decreases, stationarity diagnostics and provenance are retained in summary.json.', '']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--development', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--animation', action='store_true')
    args=parser.parse_args()
    payload=json.loads(args.campaign.read_text())
    manifest_path=args.manifest or args.campaign.with_name('manifest.json')
    manifest=None if args.development else json.loads(manifest_path.read_text())
    summary=summarize(payload, manifest)
    sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    summary['provenance']=dict(campaign=str(args.campaign), campaign_sha256=sha(args.campaign),
        frozen_manifest=manifest, summary_script_sha256=sha(Path(__file__)),
        plotting_script_sha256=sha(Path(__file__).with_name('plotting.py')))
    output=args.output_dir or args.campaign.parent
    output.mkdir(parents=True, exist_ok=True)
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    (output/'summary.md').write_text(markdown(summary))
    if args.plot or args.animation:
        from plotting import plot_campaign
        payload['_summary']=summary
        plot_campaign(payload, output/'figures', make_animation=args.animation)
    print(json.dumps(dict(status=summary['status'], rows=summary['row_counts'], output=str(output))))


if __name__ == '__main__':
    main()
