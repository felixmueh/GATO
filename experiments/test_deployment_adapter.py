"""Focused CPU packaging checks; no ROS/native imports or motion."""
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

from adapters import deployment
from recording import write_json, load_controller_stats


def test_compact_live_manifest_includes_observer_histories(tmp_path):
    recorder = deployment.LiveRecorder(tmp_path)
    recorder('state', dict(stamp_s=1., q=[0]*7))
    (tmp_path/'ros_observations.jsonl').write_text('{}\n')
    (tmp_path/'full_joint_state_history.jsonl').write_text('{}\n')
    recorder.manifest()
    manifest=json.loads((tmp_path/'SHA256SUMS.json').read_text())
    assert 'ros_observations.jsonl' in manifest and 'full_joint_state_history.jsonl' in manifest
    assert len(list((tmp_path/'raw').glob('*.npz'))) == 1


def test_preflight_receipt_resolution_and_missing_guard(tmp_path):
    output=tmp_path/'rtx5060/D01/goal1/attempt-001';output.mkdir(parents=True)
    preflight=tmp_path/'rtx5060/_preflight/D01/attempt-001';preflight.mkdir(parents=True)
    write_json(preflight/'metrics.json', {})
    write_json(output/'receipt.json', dict(preflight_path=str(preflight.relative_to(tmp_path))))
    assert deployment.preflight_path(output) == preflight
    write_json(output/'receipt.json', dict(preflight_path='../outside'))
    with pytest.raises(ValueError, match='relative preflight'):
        deployment.preflight_path(output)


def test_setup_failure_retained_without_native(tmp_path, monkeypatch):
    monkeypatch.setattr(deployment, '_run', lambda *args: (_ for _ in ()).throw(RuntimeError('retained setup failure')))
    assert deployment.run(tmp_path,tmp_path,{},tmp_path/'failed') == 1
    metrics=json.loads((tmp_path/'failed/metrics.json').read_text())
    assert 'retained setup failure' in metrics['execution_error']
    assert metrics['execution_backend'] == 'gazebo'


def test_worker_final_json_preserves_compact_summary_and_gate(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    module = tmp_path / 'module.so'
    module.write_bytes(b'fixture module')
    output = tmp_path / 'attempt'
    stats = dict(goal_outcomes=['reached'], timestamps=[.03])
    worker = SimpleNamespace(__file__=str(source / 'experiments/deployment/run_d01.py'),
                             write_json=write_json, gazebo_environment_guard=lambda: None)
    def run_case(case, path):
        with worker.RawRecorder(path) as recorder:
            recorder('controller_stats', stats)
            worker.write_json(path / 'controller_stats.json', stats)
            worker.write_json(path / 'metrics.json', dict(task_success=True, physical_pass=True,
                                                          handback_confirmed=True))
            recorder.manifest()
        assert stats == dict(goal_outcomes=['reached'], timestamps=[.03])
        return 0
    worker.run_case = run_case
    monkeypatch.setattr(deployment.importlib, 'import_module', lambda name: worker)
    monkeypatch.setattr(deployment, 'native_module', lambda *args: (object(), module))
    monkeypatch.setattr(deployment, 'adapt_force_argument', lambda worker: None)
    assert deployment._run(source, tmp_path, {}, output, True) == 0
    assert json.loads((output / 'controller_stats.json').read_text())['recording_format'] == 'task-observations-v1'
    assert load_controller_stats(output)['goal_outcomes'] == ['reached']


def test_pinned_force_seam_converts_argument_without_changing_protocol(tmp_path):
    path = tmp_path / 'worker.py'
    path.write_text("# Pinned worker fixture\n\ndef run_case(protocol):\n    return constructor(constant_f_ext=protocol['constant_f_ext'])\n")
    worker = types.ModuleType('force_fixture')
    worker.__file__ = str(path)
    def constructor(*, constant_f_ext):
        assert isinstance(constant_f_ext, np.ndarray)
        assert constant_f_ext.dtype == np.float32
        return constant_f_ext
    worker.constructor = constructor
    exec(compile(path.read_text(), str(path), 'exec'), worker.__dict__)
    before = path.read_bytes()
    original_line = worker.run_case.__code__.co_firstlineno
    protocol = dict(constant_f_ext=[0., 1., 2., 3., 4., 5.])
    deployment.adapt_force_argument(worker)
    assert worker.run_case.__code__.co_firstlineno == original_line
    np.testing.assert_array_equal(worker.run_case(protocol), protocol['constant_f_ext'])
    assert protocol['constant_f_ext'] == [0., 1., 2., 3., 4., 5.]
    assert path.read_bytes() == before
    deployment.adapt_force_argument(worker)  # Only one override per subprocess.


def test_changed_force_seam_is_rejected(tmp_path):
    path = tmp_path / 'worker.py'
    path.write_text('def run_case(protocol):\n    return protocol\n')
    worker = types.ModuleType('changed_force_fixture')
    worker.__file__ = str(path)
    exec(compile(path.read_text(), str(path), 'exec'), worker.__dict__)
    with pytest.raises(ValueError, match='seam changed'):
        deployment.adapt_force_argument(worker)
