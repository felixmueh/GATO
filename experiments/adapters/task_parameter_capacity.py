#!/usr/bin/env python3
"""Retain one existing native capacity probe; no optimizer task or ROS motion."""
import argparse
import json
from pathlib import Path
import subprocess

RESOURCE_ERRORS = {'invalid argument', 'invalid configuration argument',
                   'too many resources requested for launch'}


def classify(resource, code, N):
    if resource.get('plant') != 'tiago_right' or resource.get('N') != N:
        raise ValueError('Capacity probe returned a different plant or horizon')
    kernels = resource.get('kernels')
    if not kernels or not all(type(k.get('fits')) is bool for k in kernels):
        raise ValueError('Missing complete kernel resource outcomes')
    fits = all(k['fits'] for k in kernels)
    known_rejection = code in (0, 2) and not fits and resource.get('pcg_launch_error') in RESOURCE_ERRORS
    error = f'Capacity probe exited {code}' if code and not known_rejection else None
    return dict(outcome='execution_failed' if error else
                    ('resource_fit' if fits else 'unsupported_resources'),
        evidence_kind='resource_measurement', resource_fit=fits,
        resource_failures=[k['name'] for k in kernels if not k['fits']],
        probe_returncode=code, execution_error=error,
        task_success=None, physical_pass=None, solver_numerical_pass=None,
        optimizer_calls=0, motion_executed=False,
        scope='Configured resource/launch capacity only; no numerical or task qualification')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    N = json.loads(args.case.read_text())['parameters']['N']
    if N not in (256, 512):
        raise ValueError('Capacity extension probes are N256 or N512 only')
    try:
        result = subprocess.run([str(args.build/f'resources_N{N}')], capture_output=True,
                                text=True, timeout=30)
    except subprocess.TimeoutExpired as error:
        for filename, value in (('probe.stdout.json', error.stdout), ('probe.stderr.log', error.stderr)):
            (args.output/filename).write_text(value.decode(errors='replace') if isinstance(value, bytes) else value or '')
        (args.output/'metrics.json').write_text(json.dumps(dict(outcome='execution_failed',
            execution_error='Capacity probe timed out after 30 seconds', resource_fit=None))+'\n')
        return 1
    (args.output/'probe.stdout.json').write_text(result.stdout)
    (args.output/'probe.stderr.log').write_text(result.stderr)
    (args.output/'probe_returncode.json').write_text(json.dumps({'returncode': result.returncode})+'\n')
    try:
        resource = json.loads(result.stdout)
        metrics = classify(resource, result.returncode, N)
    except (ValueError, TypeError, KeyError) as error:
        metrics = dict(outcome='execution_failed', probe_returncode=result.returncode,
                       execution_error=str(error), resource_fit=None)
    (args.output/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    return int(metrics.get('execution_error') is not None)


if __name__ == '__main__':
    raise SystemExit(main())
