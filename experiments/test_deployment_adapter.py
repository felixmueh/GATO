"""Focused CPU packaging checks; no ROS/native imports or motion."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from adapters import deployment
from recording import write_json


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
