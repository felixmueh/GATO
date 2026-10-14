#!/usr/bin/env python3
"""Bridge the existing runner to one immutable parameter-study worker."""
import argparse
from pathlib import Path
import re
import subprocess
import sys


def worker_path(repository, build, pin):
    if not re.fullmatch(r'[0-9a-f]{40}', pin):
        raise ValueError('Collector pin must be a full immutable commit')
    directory = Path(build)/'task-parameter-collectors'/pin
    for filename in ('worker.py', 'protocol.py'):
        payload = subprocess.check_output(['git', '-C', str(repository), 'show',
                                          pin+':experiments/task_parameters/'+filename])
        path = directory/filename
        if path.exists():
            if path.read_bytes() != payload:
                raise ValueError('Cached parameter collector changed; preserve the failed cache')
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.tmp')
            temporary.write_bytes(payload)
            temporary.replace(path)
    path = directory/'worker.py'
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'build', 'inputs', 'case', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--collector-pin', required=True)
    parser.add_argument('--backend', choices=('internal', 'gazebo'), required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--dangerously-disable-safety-stops', action='store_true')
    args = parser.parse_args()
    if args.dangerously_disable_safety_stops and args.backend != 'gazebo':
        parser.error('Report-only safety is restricted to Gazebo')
    worker = worker_path(args.source, args.build, args.collector_pin)
    argv = [sys.executable, str(worker), '--runner-root', str(Path(__file__).resolve().parents[2]),
            '--backend', args.backend, '--collector-pin', args.collector_pin]
    for name in ('source', 'build', 'inputs', 'case', 'output'):
        argv += ['--'+name, str(getattr(args, name))]
    if args.preflight:
        argv.append('--preflight')
    if args.dangerously_disable_safety_stops:
        argv.append('--dangerously-disable-safety-stops')
    return subprocess.run(argv).returncode


if __name__ == '__main__':
    raise SystemExit(main())
