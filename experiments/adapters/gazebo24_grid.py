#!/usr/bin/env python3
"""Period-only 24 s extension of the immutable 90bdb825 Gazebo collector."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapters.task_parameters import worker_path
from recording import sha, write_json

COLLECTOR = '90bdb825d059e22b709e2e53a785d7d034aad1b0'


def extend_protocol(original, periodic_reference, inputs, case, backend,
                    preflight=False, report_only=False):
    if preflight:
        return original(inputs, case, backend, True, report_only)
    if backend != 'gazebo' or case['id'] != 'figure8-24s':
        raise ValueError('Only the single N32 grid 24 s Gazebo extension is supported')
    anchor_case = copy.deepcopy(case)
    anchor_case['id'] = 'figure8-12s'
    p = original(inputs, anchor_case, backend, False, report_only)
    if (p['N'], p['dt_s'], p['solver_params']['max_sqp_iters'],
            p['solver_params']['max_pcg_iters']) != (32, .03, 5, 120):
        raise ValueError('Grid anchor parameters must remain unchanged')
    cohort = json.loads((Path(inputs) / 't01.json').read_text())
    task = next(t for t in cohort['tasks'] if t['id'] == 'figure8-12s')
    reference = periodic_reference(task, 24., 24., 32, .03)
    p.update(id='figure8-24s', case_id='figure8-24s', period_s=24., duration_s=24.,
             reference_packed=reference.tolist(),
             reference_sha256=hashlib.sha256(reference.tobytes()).hexdigest(),
             reference_coverage_s=(len(reference) // 6 - 1) * .03,
             maximum_reference_speed_m_s=task['maximum_reference_speed_m_s'] * task['period_s'] / 24.,
             period_extension='12 s to 24 s only; original 90bdb825 collector and tracking settings',
             joint_state_rate_policy='Existing PAL configuration retained; no rate changes')
    return p


def run(args):
    if args.collector_pin != COLLECTOR:
        raise ValueError('This extension requires the exact grid collector pin')
    path = worker_path(args.source, args.build, COLLECTOR)
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location('saved_grid24_worker', path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    from protocol import periodic_reference
    original = worker.resolve_protocol
    worker.resolve_protocol = lambda *a, **kw: extend_protocol(original, periodic_reference, *a, **kw)
    if not args.dry_run:
        # Reuse the passed identical grid canonical rather than adding a motion.
        # Its original require_preflight still checks setup hash, task/error/handback.
        from adapters import deployment
        evidence = args.preflight_evidence.resolve()
        for name in ('metrics.json', 'protocol.json', 'receipt.json'):
            if not (evidence / name).is_file():
                raise ValueError('Missing retained canonical evidence: ' + str(evidence / name))
        saved = json.loads((evidence / 'receipt.json').read_text())
        if saved.get('status') != 'completed':
            raise ValueError('Retained canonical receipt is not completed')
        destination = args.output / 'reused_preflight'
        destination.mkdir(parents=True, exist_ok=False)
        for name in ('metrics.json', 'protocol.json', 'receipt.json'):
            shutil.copy2(evidence / name, destination / name)
        write_json(destination / 'reuse.json', dict(original=str(evidence),
            hashes={name: sha(evidence / name) for name in ('metrics.json', 'protocol.json', 'receipt.json')},
            reason='Same immutable grid source/settings/policy; no extra canonical episode'))
        deployment.preflight_path = lambda output: destination
    status = worker.run(args)
    if not args.dry_run:
        metrics_path = args.output / 'metrics.json'
        metrics = json.loads(metrics_path.read_text())
        metrics['period_extension_adapter_sha256'] = sha(__file__)
        metrics['retained_collector_source_commit'] = COLLECTOR
        write_json(metrics_path, metrics)
        manifest_path = args.output / 'SHA256SUMS.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['metrics.json'] = sha(metrics_path)
        write_json(manifest_path, manifest)
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'build', 'inputs', 'case', 'output', 'preflight-evidence'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--collector-pin', default=COLLECTOR)
    parser.add_argument('--dangerously-disable-safety-stops', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    args.runner_root = Path(__file__).resolve().parents[2]
    args.backend, args.preflight = 'gazebo', False
    raise SystemExit(run(args))
