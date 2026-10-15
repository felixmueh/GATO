#!/usr/bin/env python3
"""Prepare one explicitly chosen cost-search point and CPU-only dry run."""
import argparse
import json
import math
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--base', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--q-weight', type=float, required=True)
    parser.add_argument('--speed-weight', type=float, required=True)
    parser.add_argument('--reason', required=True)
    args = parser.parse_args()
    if any(not math.isfinite(value) or value <= 0 for value in (args.q_weight, args.speed_weight)):
        parser.error('Weights must be finite and positive')
    directory = args.source / 'experiments/random_reaching_baseline/protocols'
    protocol = json.loads((directory / (args.base + '.json')).read_text())
    target = directory / (args.name + '.json')
    if target.exists():
        parser.error('Refusing to overwrite an existing search protocol')
    preparation = args.raw_root / ('preparation-' + args.name.split('-')[0])
    if preparation.exists():
        parser.error('Refusing to overwrite an existing preparation directory')
    protocol['id'] = 'random-reaching-margins-' + args.name + '-A-b1-sqp1000-v1'
    protocol['solver_params']['q_lim_cost'] = args.q_weight
    protocol['solver_params']['vel_lim_cost'] = args.speed_weight
    protocol['comparison'] = args.reason
    target.write_text(json.dumps(protocol, indent=2) + '\n')
    preparation.mkdir(exist_ok=False)
    command = [
        '/opt/gato-venv/bin/python',
        str(args.source / 'experiments/random_reaching_baseline/run.py'),
        '--source', str(args.source), '--module', protocol['native_module_path'],
        '--pool', '/workspace/GATO-experiment-runner-simple/experiments/inputs/batching',
        '--protocol', str(target),
        '--output', str(args.raw_root / args.name / 'preflight'),
        '--preflight-input', '/workspace/GATO-experiment-runner-simple/experiments/inputs/batching/ba_preflight.json',
        '--dry-run',
    ]
    (preparation / 'argv.json').write_text(json.dumps(command, indent=2) + '\n')
    result = subprocess.run(command, cwd=args.source, text=True, capture_output=True)
    (preparation / 'dryrun.stdout.json').write_text(result.stdout)
    (preparation / 'dryrun.stderr.txt').write_text(result.stderr)
    print(json.dumps({'protocol': str(target), 'dry_run_returncode': result.returncode}))
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
