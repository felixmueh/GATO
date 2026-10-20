#!/usr/bin/env python3
"""Freeze and run bounded public native/LDLT reaching and tracking verification."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from upstream_reach import ROOT, sha256, write_json


def native_path(checkout, knots=128):
    paths = list((checkout / 'python/gato').glob(f'bsqpN{knots}_indy7*.so'))
    if len(paths) != 1:
        raise RuntimeError(f'Expected exactly one prebuilt Indy7 N{knots} module, got {paths}')
    return paths[0]


def freeze(args):
    checkout = args.checkout.resolve()
    sys.path.insert(0, str(checkout / 'python'))
    import pinocchio as pin
    import gato
    from gato import tasks
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config_source = Path(__file__).with_name('upstream_reach_config.json')
    (output / 'config.json').write_bytes(config_source.read_bytes())
    urdf = Path(gato.robot_urdf('indy7'))
    model = pin.buildModelFromUrdf(str(urdf))
    knot_counts = [int(n) for n in args.knots.split(",")]
    modules = {str(n): native_path(checkout,n) for n in knot_counts}
    module = modules[str(knot_counts[0])]
    manifest = dict(protocol='public-ldlt-generalization-v1', checkout=str(checkout),
        binary_checkout_source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout, text=True).strip(),
        harness_source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        harness_dirty_status=subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True),
        worker_sha256=sha256(__file__), driver_sha256=sha256(Path(__file__).with_name('upstream_reach.py')),
        module_filename=module.name, module_sha256=sha256(module), urdf_sha256=sha256(urdf),
        public_tasks_sha256=sha256(checkout / 'python/gato/tasks.py'), config_sha256=sha256(output / 'config.json'),
        repeats=3, methods=['normal', 'block_ldlt'], cases=[], state='freezing',
        knot_counts=knot_counts, reference_padding_knots=max(knot_counts),
        module_sha256_by_knots={n:sha256(path) for n,path in modules.items()})
    write_json(output / 'manifest.json', manifest)
    try:
        specifications = [('pickplace', seed, None) for seed in range(2, 8)]
        specifications += [('fig8', seed, ratio) for seed, ratio in zip((11, 12, 13), (.6, .8, 1.0))]
        for kind, seed, ratio in specifications:
            case_id = ('reach' if kind == 'pickplace' else 'tracking') + f'-seed{seed}'
            if kind == 'pickplace':
                scenario = tasks.pickplace_scenarios(model, 'indy7', 1, seed=seed, n_goals=3, ramps=(0.0,))[0]
                case = dict(case_id=case_id, task=kind, seed=seed, scenario=scenario)
            else:
                scenario = tasks.fig8_scenarios(model, 'indy7', 1, seed=seed, dt=.01, ratio_range=(ratio, ratio))[0]
                reference, sim_time = tasks.fig8_reference(scenario, max(knot_counts), periods=2)
                reference_file = output / (case_id + '-reference.npz')
                np.savez_compressed(reference_file, reference=np.asarray(reference))
                case = dict(case_id=case_id, task=kind, seed=seed, ratio=ratio, periods=2,
                    scenario=scenario, sim_time_s=sim_time, reference_file=reference_file.name,
                    reference_sha256=sha256(reference_file))
            task_file = output / (case_id + '.json')
            write_json(task_file, case)
            manifest['cases'].append(dict(case_id=case_id, task=kind, seed=seed,
                                          file=task_file.name, sha256=sha256(task_file)))
            write_json(output / 'manifest.json', manifest)
        manifest['state'] = 'complete'
    except BaseException as exc:
        manifest.update(state='failed', error=repr(exc))
        write_json(output / 'manifest.json', manifest)
        raise
    write_json(output / 'manifest.json', manifest)
    print(json.dumps(manifest), flush=True)


def validate(tasks_dir, checkout, knots=128):
    manifest = json.loads((tasks_dir / 'manifest.json').read_text())
    if manifest['state'] != 'complete' or len(manifest['cases']) != 9:
        raise RuntimeError('Shared frozen-task setup incomplete; no cases may start')
    if sha256(native_path(checkout,knots)) != manifest.get('module_sha256_by_knots', {'128':manifest['module_sha256']}).get(str(knots)):
        raise RuntimeError('Prebuilt native/LDLT module differs from frozen identity; no rebuild/fallback')
    if sha256(tasks_dir / 'config.json') != manifest['config_sha256']:
        raise RuntimeError('Frozen task parameter file changed')
    for case in manifest['cases']:
        task_file = tasks_dir / case['file']
        if sha256(task_file) != case['sha256']:
            raise RuntimeError('Frozen task definition changed')
        task = json.loads(task_file.read_text())
        if task['task'] == 'fig8' and sha256(tasks_dir / task['reference_file']) != task['reference_sha256']:
            raise RuntimeError('Frozen tracking reference changed')
    return manifest


def command(checkout, tasks_dir, case, method, output, primary_timing, linsys=None, knots=128):
    cmd = [sys.executable, str(Path(__file__).resolve()), 'run', '--checkout', str(checkout),
            '--tasks', str(tasks_dir), '--case', case, '--method', method,
            '--primary-timing', primary_timing, '--knots', str(knots), '--output', str(output)]
    return cmd + ['--linsys', linsys] if linsys else cmd


def run(args):
    tasks_dir, checkout = args.tasks.resolve(), args.checkout.resolve()
    manifest = validate(tasks_dir, checkout,args.knots)
    case = next((c for c in manifest['cases'] if c['case_id'] == args.case), None)
    if case is None:
        raise ValueError(f'Unknown frozen case {args.case}')
    from upstream_reach import main
    cmd = ['--checkout', str(checkout), '--config', str(tasks_dir / 'config.json'),
                 '--task-file', str(tasks_dir / case['file']), '--method', args.method,
                 '--primary-timing', args.primary_timing, '--knots', str(args.knots),
                 '--output', str(args.output)]
    return main(cmd + ['--linsys', args.linsys] if args.linsys else cmd)


def campaign(args):
    tasks_dir, checkout, output = args.tasks.resolve(), args.checkout.resolve(), args.output.resolve()
    manifest = validate(tasks_dir, checkout,args.knots)
    output.mkdir(parents=True, exist_ok=False)
    requested = args.cases.split(',') if args.cases else [c['case_id'] for c in manifest['cases']]
    unknown = set(requested) - {c['case_id'] for c in manifest['cases']}
    if unknown:
        raise ValueError(f'Unknown frozen cases {sorted(unknown)}')
    selected = [c for c in manifest['cases'] if c['case_id'] in requested]
    repeats = args.repeats
    if repeats < 1:
        raise ValueError('--repeats must be positive')
    if args.preflight:
        selected = [next(c for c in selected if c['task'] == kind) for kind in ('pickplace', 'fig8')
                    if any(c['task'] == kind for c in selected)]
        repeats = 1
    methods = args.methods.split(',')
    if not methods or len(set(methods)) != len(methods):
        raise ValueError('--methods must be a nonempty list without duplicates')
    write_json(output / 'campaign_config.json', dict(cases=[c['case_id'] for c in selected], methods=methods,
        repeats=repeats, knots=args.knots, preflight=args.preflight, process_count=len(selected) * len(methods) * repeats,
        primary_timing=args.primary_timing, manifest_sha256=sha256(tasks_dir / 'manifest.json'),
        harness_source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()))
    status = []
    for case in selected:
        for repeat in range(repeats):
            order = methods if repeat % 2 == 0 else list(reversed(methods))
            for method in order:
                name = f"{case['case_id']}-{method}-r{repeat}"
                cmd = command(checkout, tasks_dir, case['case_id'], method, output / name, args.primary_timing, args.linsys,args.knots)
                with (output / (name + '.stdout')).open('w') as stdout, (output / (name + '.stderr')).open('w') as stderr:
                    begin = time.perf_counter_ns()
                    result = subprocess.run(cmd, stdout=stdout, stderr=stderr)
                    process_wall_us = (time.perf_counter_ns() - begin) / 1000
                status.append(dict(case_id=case['case_id'], method=method, repeat=repeat,
                                   returncode=result.returncode, command=cmd,
                                   process_wall_us=process_wall_us))
                write_json(output / 'status.json', status)
                print(json.dumps(status[-1]), flush=True)
    return int(any(row['returncode'] for row in status))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    for action in ('freeze', 'run', 'campaign'):
        s = sub.add_parser(action)
        s.add_argument('--checkout', type=Path, required=True)
        s.add_argument('--output', type=Path, required=True)
        s.add_argument('--knots', default='128' if action=='freeze' else 128, type=str if action=='freeze' else int)
        if action != 'freeze':
            s.add_argument('--tasks', type=Path, required=True)
            s.add_argument('--primary-timing', choices=('wall', 'compute_excluding_transfers'), default='compute_excluding_transfers')
            s.add_argument('--linsys', choices=('pcg', 'bdsv'))
        if action == 'run':
            s.add_argument('--case', required=True)
            s.add_argument('--method', required=True)
        if action == 'campaign':
            s.add_argument('--cases', help='comma-separated frozen case IDs; default all nine')
            s.add_argument('--methods', default='normal,block_ldlt')
            s.add_argument('--repeats', type=int, default=3)
            s.add_argument('--preflight', action='store_true', help='explicitly run first selected reach and tracking case, one repeat per method')
    args = p.parse_args()
    return {'freeze': freeze, 'run': run, 'campaign': campaign}[args.action](args) or 0


if __name__ == '__main__':
    raise SystemExit(main())
