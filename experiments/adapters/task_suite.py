#!/usr/bin/env python3
"""Run the unchanged three-case task regression suite on one exact N32 build.

Equivalent task commands to check-tiago-tasks: original CUDA access smoke,
then original opt-in pytest fixtures. No controller or fixture transplant edits.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ET

EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS))
from recording import sha, write_json
from adapters.f01ab import resolve_module


def junit_result(path):
    root = ET.parse(path).getroot()
    cases = list(root.iter('testcase'))
    expected = ('short_reach', 'long_reach', 'figure8_tracking')
    names = [case.get('name', '') for case in cases]
    return dict(tests=len(cases), failures=sum(len(c.findall('failure')) for c in cases),
                errors=sum(len(c.findall('error')) for c in cases),
                skipped=sum(len(c.findall('skipped')) for c in cases),
                exact_cases=len(cases) == 3 and all(sum(f'[{name}]' in n for n in names) == 1 for name in expected))


def run(source, build, output, execute=subprocess.run):
    source, build, output = (Path(p).resolve() for p in (source, build, output))
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'metrics.json').exists():
        raise ValueError('Task-suite output already contains an attempt')
    metrics = dict(campaign_id='Q03', name='Frozen reaching and tracking regression suite',
                   execution_backend='internal', timing_mode='historical_replay',
                   task_success=None, physical_pass=None, execution_error=None,
                   outcome='execution_failed', evidence_kind='measurement')
    try:
        module = resolve_module(build, 32)
        smoke = build / 'cuda_access_smoke'
        if not smoke.is_file():
            raise FileNotFoundError('Build must include native/task_readiness CUDA access probe')
        test = source / 'tests/python/test_tiago_task_regression.py'
        names = ['tests/python/test_tiago_task_regression.py', 'tests/python/tiago_task_worker.py']
        names += [f'tests/python/tiago_task_cases/{name}.json' for name in ('short_reach', 'long_reach', 'figure8_tracking')]
        environment = dict(os.environ, GATO_RUN_TIAGO_TASK_TESTS='1',
            GATO_TEST_ARTIFACT_DIR=str(output / 'tasks'), GATO_TASK_SOURCE=str(source),
            GATO_TASK_NATIVE_MODULE=str(module), PYTHONDONTWRITEBYTECODE='1',
            PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
            PYTHONPATH=str(EXPERIMENTS / 'task_bootstrap') + os.pathsep + str(source / 'python') + os.pathsep + str(source / 'tiago_src'))
        commands = [[str(smoke)], [sys.executable, '-B', '-m', 'pytest', '-q', '-m', 'task_regression', str(test),
            '--junitxml', str(output / 'junit.xml'), '-o', 'cache_dir=' + str(output / 'pytest-cache'),
            '--basetemp', str(output / 'pytest-tmp')]]
        write_json(output / 'provenance.json', dict(source_commit=subprocess.check_output(
            ['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
            source_root=str(source), module_path=str(module), module_sha256=sha(module),
            smoke_sha256=sha(smoke), fixture_sha256={p: sha(source / p) for p in names},
            adapter_sha256=sha(__file__), bootstrap_sha256=sha(EXPERIMENTS / 'task_bootstrap/sitecustomize.py'),
            commands=commands, protocol='frozen-task-suite-adapter-v1'))
        for label, command in zip(('cuda-access', 'task-suite'), commands):
            with (output / (label + '.log')).open('w') as log:
                result = execute(command, cwd=source, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
            metrics[label + '_returncode'] = result.returncode
            if result.returncode:
                raise RuntimeError(f'{label} exited {result.returncode}; retained log and partial task artifacts')
        metrics.update(junit_result(output / 'junit.xml'))
        if not metrics['exact_cases'] or any(metrics[k] for k in ('failures', 'errors', 'skipped')):
            raise RuntimeError('Expected exactly all three unchanged task cases with no skips or failures')
        metrics.update(task_success=True, physical_pass=True, outcome='task_suite_passed')
    except BaseException:
        metrics['execution_error'] = traceback.format_exc()
    write_json(output / 'metrics.json', metrics)
    write_json(output / 'SHA256SUMS.json', {str(p.relative_to(output)): sha(p) for p in output.rglob('*')
        if p.is_file() and p.name != 'SHA256SUMS.json'})
    return 1 if metrics['execution_error'] else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'build', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.source, args.build, args.output))
