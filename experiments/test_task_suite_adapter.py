"""Readiness cannot silently skip tests or fall back to an installed module."""
import os
from pathlib import Path
import subprocess
import sys

from adapters.task_suite import junit_result, run


def test_all_original_tasks_required(tmp_path):
    path = tmp_path / 'junit.xml'
    names = ('short_reach', 'long_reach', 'figure8_tracking')
    path.write_text('<testsuite>' + ''.join(f'<testcase name="test_task[{n}]"/>' for n in names) + '</testsuite>')
    assert junit_result(path) == dict(tests=3, failures=0, errors=0, skipped=0, exact_cases=True)
    path.write_text('<testsuite><testcase name="test_task[short_reach]"><skipped/></testcase></testsuite>')
    assert not junit_result(path)['exact_cases']
    assert junit_result(path)['skipped'] == 1


def test_missing_selected_module_stops_python_before_fallback(tmp_path):
    bootstrap = Path(__file__).parent / 'task_bootstrap'
    env = dict(os.environ, PYTHONPATH=str(bootstrap), GATO_TASK_SOURCE=str(tmp_path),
               GATO_TASK_NATIVE_MODULE=str(tmp_path / 'missing.so'))
    child = subprocess.run([sys.executable, '-B', '-c', 'print("UNSAFE_FALLBACK")'], env=env,
                           capture_output=True, text=True)
    assert child.returncode == 86
    assert 'UNSAFE_FALLBACK' not in child.stdout


def test_missing_native_is_failure_with_saved_metrics(tmp_path):
    import json
    assert run(tmp_path / 'source', tmp_path / 'build', tmp_path / 'output') == 1
    metrics = json.loads((tmp_path / 'output/metrics.json').read_text())
    assert metrics['task_success'] is None and metrics['physical_pass'] is None
    assert metrics['execution_error']
