#!/usr/bin/env python3
"""Launch one frozen internal campaign locally, using the existing runner."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import collect
import package_plans
import run

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'experiments'


def source_ref(commit):
    return 'refs/heads/experiment/deployment-source-' + commit[:12]


def verify_sources(plan, package=None):
    pins = package_plans.sources(plan['experiments'])
    missing = [pin for pin in pins if subprocess.run(
        ['git', '-C', str(ROOT), 'cat-file', '-e', pin + '^{commit}'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode]
    if missing:
        if package:
            inventory = json.loads((package / 'inventory.json').read_text())
            bundle = package / 'sources.bundle'
            if run.file_hash(bundle) != inventory['package_sha256']['sources.bundle']:
                raise ValueError('Source bundle hash mismatch')
            subprocess.run(['git', '-C', str(ROOT), 'fetch', str(bundle),
                            'refs/heads/*:refs/experiment/package/*'], check=True)
        else:
            # Exact published refs make detached experiment pins available after a pull.
            subprocess.run(['git', '-C', str(ROOT), 'fetch', 'origin',
                            *[source_ref(pin) + ':refs/experiment/deployment/' + pin
                              for pin in missing]], check=True)
    for pin in pins:
        subprocess.run(['git', '-C', str(ROOT), 'cat-file', '-e', pin + '^{commit}'], check=True)
    for experiment in plan['experiments']:
        for value, expected in experiment.get('inputs', {}).items():
            if run.file_hash(ROOT / run.relative(value)) != expected:
                raise ValueError('Frozen input hash mismatch: ' + value)
        for value, expected in experiment.get('build', {}).get('inputs', {}).items():
            if run.file_hash(HERE / run.relative(value)) != expected:
                raise ValueError('Frozen build helper hash mismatch: ' + value)


def check_dependencies(python):
    missing = [name for name in ('git', 'cmake', 'nvcc', 'c++') if not shutil.which(name)]
    if missing:
        raise ValueError('Required tools unavailable: ' + ', '.join(missing) +
                         '. Use the prepared CUDA development environment; launcher does not alter system packages.')
    subprocess.run([python, '-c', 'import numpy, pinocchio, pybind11, pytest'], check=True)


def arguments(machine, plan, output, cache, python, dry_run=False, retry_failed=False):
    return SimpleNamespace(plan=plan, machine=machine, output=str(output), cache=str(cache),
                           python=python, dry_run=dry_run, retry_failed=retry_failed, only=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('machine', choices=('rtx5060', 'jetson'))
    parser.add_argument('--python', default='/opt/gato-venv/bin/python3' if Path('/opt/gato-venv/bin/python3').exists() else sys.executable,
                        help='Python in the existing CUDA/Pinocchio environment (defaults to this interpreter)')
    parser.add_argument('--package', type=Path, help='Self-contained transfer directory for offline exact-source provisioning')
    parser.add_argument('--dry-run', action='store_true', help='Validate sources/inputs and inspect the plan; no build or GPU execution')
    parser.add_argument('--collect-only', action='store_true', help='Index and archive existing local attempts; no GPU execution')
    parser.add_argument('--retry-failed', action='store_true', help='Explicitly create fresh attempts for failed/blocked cases')
    args = parser.parse_args()
    plan_path = HERE / 'launch' / (args.machine + '.json')
    plan = json.loads(plan_path.read_text())
    cohort = 'zero-pcg-20261011'
    output = ROOT / '.experiment-results' / (args.machine + '-' + cohort)
    cache = ROOT / '.experiment-cache' / (args.machine + '-' + cohort)
    archive = ROOT / '.experiment-work' / (args.machine + '-' + cohort + '-results.tar.gz')
    python = shutil.which(args.python) or str(Path(args.python).expanduser().absolute())
    print(f'Plan: {plan_path}\nSHA256: {run.file_hash(plan_path)}\nRaw: {output}\nArchive: {archive}', flush=True)
    print(f'Available storage beside checkout: {shutil.disk_usage(ROOT).free / (1 << 30):.1f} GiB', flush=True)
    if not args.collect_only:
        verify_sources(plan, args.package.resolve() if args.package else None)
        dry_args = arguments(args.machine, plan_path, output, cache, python, dry_run=True,
                             retry_failed=args.retry_failed)
        run.run(dry_args)
        if args.dry_run:
            return 0
        check_dependencies(python)
        output.mkdir(parents=True, exist_ok=True)
        deployment = dict(machine=args.machine, plan_sha256=run.file_hash(plan_path),
                          runner_commit=run.git('rev-parse', 'HEAD'), plan=plan,
                          note='Manual local internal simulation. Previous attempts and failures retained.')
        deployment_path = output / 'deployment.json'
        if deployment_path.exists():
            previous = json.loads(deployment_path.read_text())
            if previous['plan_sha256'] != deployment['plan_sha256']:
                raise ValueError('Plan changed for this result directory; preserve it and choose a fresh cohort.')
        else:
            deployment_path.write_text(json.dumps(deployment, indent=2) + '\n')
        status = 1
        try:
            status = run.run(arguments(args.machine, plan_path, output, cache, python,
                                       retry_failed=args.retry_failed))
        except KeyboardInterrupt:
            status = 130
        finally:
            collect.collect(output)
            collect.export_results(output, archive)
            print(f'All local raw attempts and failures archived: {archive}', flush=True)
        return status
    if not output.is_dir():
        raise ValueError('No local results yet: ' + str(output))
    collect.collect(output)
    collect.export_results(output, archive)
    print(f'All local raw attempts and failures archived: {archive}', flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
