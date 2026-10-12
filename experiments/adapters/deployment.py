#!/usr/bin/env python3
"""Compact recording wrapper around the reviewed live D01 Gazebo worker."""
import argparse
import importlib
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recording import sha, write_json
from adapters.task_recording import TaskRecorder, TASK_RECORDING_FORMAT
from adapters.task import native_module


class LiveRecorder(TaskRecorder):
    def manifest(self):
        super().manifest()
        path = self.output / 'SHA256SUMS.json'
        hashes = json.loads(path.read_text())
        for name in ('case.json', 'ros_observations.jsonl', 'controller_state_history.csv', 'full_joint_state_history.jsonl'):
            target = self.output / name
            if target.exists():
                hashes[name] = sha(target)
        write_json(path, hashes)


def preflight_path(output):
    receipt = json.loads((output / 'receipt.json').read_text())
    relative = receipt.get('preflight_path')
    if not relative or Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError('Runner receipt lacks a relative preflight path')
    matches = [root / relative for root in output.parents if (root / relative / 'metrics.json').is_file()]
    if len(matches) != 1:
        raise ValueError('Runner preflight path does not resolve uniquely')
    return matches[0]


def _run(source, build, case, output, preflight=False):
    source, build, output = map(lambda p: Path(p).resolve(), (source, build, output))
    sys.path.insert(0, str(source))
    worker = importlib.import_module('experiments.deployment.run_d01')
    if Path(worker.__file__).resolve() != source / 'experiments/deployment/run_d01.py':
        raise ValueError('D01 worker imported from a different checkout')
    worker.RawRecorder = LiveRecorder
    original_write = worker.write_json
    def compact_write(path, value):
        if Path(path).name == 'controller_stats.json':
            # The immediately preceding callback wrote its compact NPZ envelope.
            if not Path(path).is_file():
                raise ValueError('Missing compact controller summary')
            return
        return original_write(path, value)
    worker.write_json = compact_write
    # The public worker still resolves model, task and live controller defaults.
    case = dict(id='preflight-goal-1', repeat=0, parameters=dict(backend='gazebo')) if preflight else dict(case)
    if not preflight:
        case['parameters'] = dict(case.get('parameters', {}), preflight_output=str(preflight_path(output)))
    worker.gazebo_environment_guard()  # No native load when ROS simulation guard fails.
    _, module = native_module(source, build)
    status = worker.run_case(case, output)
    metrics_path = output / 'metrics.json'
    metrics = json.loads(metrics_path.read_text())
    metrics.update(collection_protocol='simple-d01-v1', research_question='RQ-control',
                   collector_sha256=sha(__file__), module_sha256=sha(module),
                   recording_format=TASK_RECORDING_FORMAT)
    write_json(metrics_path, metrics)
    manifest = json.loads((output / 'SHA256SUMS.json').read_text())
    manifest['metrics.json'] = sha(metrics_path)
    write_json(output / 'SHA256SUMS.json', manifest)
    if preflight and (metrics.get('family_held') or metrics.get('task_success') is not True
                     or metrics.get('physical_pass') is not True or metrics.get('handback_confirmed') is not True):
        return 1
    return status


def run(source, build, case, output, preflight=False):
    try:
        return _run(source, build, case, output, preflight)
    except BaseException:
        error = traceback.format_exc()
        output = Path(output).resolve()
        output.mkdir(parents=True, exist_ok=True)
        metrics_path = output / 'metrics.json'
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        metrics.update(campaign_id='D01', execution_backend='gazebo', timing_mode='live',
                       machine='rtx5060', execution_error=error, outcome='execution_failed')
        write_json(metrics_path, metrics)
        manifest_path = output / 'SHA256SUMS.json'
        if manifest_path.exists():
            hashes = json.loads(manifest_path.read_text())
            hashes['metrics.json'] = sha(metrics_path)
            write_json(manifest_path, hashes)
        print(error, file=sys.stderr)
        return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--build', required=True, type=Path)
    parser.add_argument('--case', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    raise SystemExit(run(args.source, args.build, json.loads(args.case.read_text()), args.output, args.preflight))
