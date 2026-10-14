#!/usr/bin/env python3
"""Launch one unchanged public example, or export its small orientation-figure files."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import socket
import subprocess
import sys

SOURCE = 'b400bdf2bb6eea543b17cefcb0577d5de4d22ce2'
FILES = ('reference.csv', 'batch_1_actual.csv', 'batch_1_joints.csv',
         'summary.json', 'tracking_direction_vectors.png')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    # Raw summaries stay byte-identical; unavailable/nonfinite derived scalars use null.
    def finite(item):
        if isinstance(item, dict): return {k: finite(v) for k, v in item.items()}
        if isinstance(item, list): return [finite(v) for v in item]
        return None if isinstance(item, float) and not math.isfinite(item) else item
    path.write_text(json.dumps(finite(value), indent=2, allow_nan=False)+'\n')


def launch(args):
    source, build = args.source.resolve(), args.build.resolve()
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    if commit != SOURCE or dirty:
        raise ValueError('Exact clean b400 runtime required')
    if (build/'source_commit.txt').read_text().strip() != commit:
        raise ValueError('Matching source build receipt required; do not use the old main N64 binary')
    modules = list((build/'modules').glob('bsqpN64_tiago_right*.so'))
    installed = list((source/'python/bsqp').glob('bsqpN64_tiago_right*.so'))
    if len(modules) != 1 or len(installed) != 1 or sha(modules[0]) != sha(installed[0]):
        raise ValueError('Install the exact separately built N64 module first')
    output = args.root.resolve()/args.condition
    output.mkdir(parents=True, exist_ok=False)
    weights = ('0', '0') if args.condition == 'disabled' else ('0.2', '4')
    command = [sys.executable, str(source/'tiago_examples/tiago_fig8_tracking.py'),
        '--plant', 'tiago_right', '--N', '64', '--dt', '0.008', '--sim-dt', '0.001',
        '--sim-time', '16', '--batch-sizes', '1', '--tiago-start-config', 'comfortable_high_clearance',
        '--ee-orientation-rpy', '0', '0', '0', '--ee-orient-cost', weights[0],
        '--ee-orient-N-cost', weights[1], '--no-gif', '--output-root', str(output/'raw'),
        '--output-label', 'orientation_'+args.condition]
    receipt = dict(source_commit=commit, tracked_worktree_dirty=bool(dirty),
        collector_sha256=sha(__file__), module_path=str(modules[0]), module_sha256=sha(modules[0]),
        build_source_receipt_sha256=sha(build/'source_commit.txt'),
        cmake_cache_sha256=sha(build/'CMakeCache.txt'),
        machine=socket.gethostname(), command=command, offline_timing='controller_dt', timing_mode='A',
        orientation_target='tool +Z up in torso_lift_link frame; yaw about tool axis is not a success gate',
        started_utc=datetime.now(timezone.utc).isoformat())
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv,noheader'],
                         capture_output=True, text=True, timeout=10) if shutil.which('nvidia-smi') else None
    receipt['gpu'] = gpu.stdout.strip() if gpu else 'nvidia-smi unavailable'
    write(output/'receipt.json', receipt)
    with (output/'stdout.log').open('w') as stdout, (output/'stderr.log').open('w') as stderr:
        code = subprocess.run(command, cwd=source, stdout=stdout, stderr=stderr).returncode
    receipt.update(returncode=code, ended_utc=datetime.now(timezone.utc).isoformat())
    write(output/'receipt.json', receipt)
    return code


def export(args):
    import numpy as np
    root = args.root.resolve()
    destination = args.destination.resolve()/root.name
    destination.mkdir(parents=True, exist_ok=False)
    records = {}
    for condition, weights in (('disabled', (0., 0.)), ('enabled', (.2, 4.))):
        case = root/condition
        receipt = json.loads((case/'receipt.json').read_text())
        raw = case/'raw/tiago_right'/('orientation_'+condition)
        selected = destination/condition
        selected.mkdir()
        shutil.copy2(case/'receipt.json', selected/'receipt.json')
        checks = dict(process_exit_zero=receipt.get('returncode') == 0, missing_files=[],
                      finite_csvs=True, protocol_matches=False, duration_near_16s=False)
        hashes = {}
        for filename in FILES:
            path = raw/filename
            if not path.is_file():
                checks['missing_files'].append(filename)
                continue
            shutil.copy2(path, selected/filename)
            hashes[filename] = sha(path)
            if filename.endswith('.csv'):
                data = np.loadtxt(path, delimiter=',', skiprows=1)
                checks['finite_csvs'] &= bool(data.size and np.isfinite(data).all())
                if filename == 'batch_1_actual.csv' and data.ndim == 2 and data.shape[0]:
                    checks['last_recorded_time_s'] = float(data[-1, 0])
                    checks['duration_near_16s'] = bool(abs(data[-1, 0] - 16.) <= .05)
        for filename in ('batch_1_applied_controls.csv', 'batch_1_planned_controls.csv'):
            path = raw/filename
            if path.is_file():
                data = np.loadtxt(path, delimiter=',', skiprows=1)
                checks['finite_csvs'] &= bool(data.size and np.isfinite(data).all())
        summary_path = raw/'summary.json'
        summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
        params, ref = summary.get('solver_params', {}), summary.get('reference_metadata', {})
        checks['protocol_matches'] = bool(
            (summary.get('N'), summary.get('dt'), summary.get('sim_dt'), summary.get('sim_time'), summary.get('batch_sizes')) == (64, .008, .001, 16., [1])
            and summary.get('ros_tiago') is False
            and (params.get('ee_orient_cost'), params.get('ee_orient_N_cost')) == weights
            and params.get('max_pcg_iters') == 120 and params.get('max_sqp_iters') == 5
            and (ref.get('reference_mode'), ref.get('x_span_m'), ref.get('y_amplitude_m'), ref.get('period_sec'), ref.get('start_edge_clearance_m')) == ('tiago_horizontal', .32, .08, 24., .02)
            and ref.get('ee_orientation_rpy') == [0., 0., 0.])
        records[condition] = dict(receipt=receipt, checks=checks, sha256=hashes,
            summary_by_batch=summary.get('summary_by_batch'),
            note='KKT convergence fraction is descriptive; physical limit violations remain separate from execution checks')
    references_match = bool(records['disabled']['sha256'].get('reference.csv') and
        records['disabled']['sha256'].get('reference.csv') == records['enabled']['sha256'].get('reference.csv'))
    for record in records.values(): record['checks']['paired_reference_matches'] = references_match
    write(destination/'collection.json', records)
    lines = ['# Orientation-cost figure data', '',
        'New b400 current-public-example study; not a reproduction of June data. Timing: controller_dt (mode A), distinct from mode B campaign tasks.', '']
    for condition, record in records.items():
        lines += [f'## {condition}', '', f"Commit: {record['receipt']['source_commit']}; module SHA256: {record['receipt']['module_sha256']}",
            f"Machine/GPU: {record['receipt']['machine']} / {record['receipt']['gpu']}",
            f"Start/end: {record['receipt']['started_utc']} / {record['receipt'].get('ended_utc', 'unfinished')}",
            'Command: `'+ ' '.join(record['receipt']['command'])+'`',
            'Checks: `'+json.dumps(record['checks'])+'`',
            'Summary: `'+json.dumps(record['summary_by_batch'])+'`',
            'CSV/file SHA256: `'+json.dumps(record['sha256'])+'`', '']
    lines += ['Tool-axis mean/max angle comparison is computed by the existing thesis figure generator from these joint CSVs. Preserve either outcome; no tuning or retries.']
    (destination/'README.md').write_text('\n'.join(lines)+'\n')
    print(destination)
    return int(any(not r['checks']['process_exit_zero'] or r['checks']['missing_files']
                   or not r['checks']['finite_csvs'] or not r['checks']['protocol_matches']
                   or not r['checks']['duration_near_16s'] or not r['checks']['paired_reference_matches'] for r in records.values()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('run', 'export'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--condition', choices=('disabled', 'enabled'))
    p.add_argument('--source', type=Path)
    p.add_argument('--build', type=Path)
    p.add_argument('--destination', type=Path)
    args = p.parse_args()
    if args.action == 'run':
        if not all((args.condition, args.source, args.build)): p.error('run requires --condition, --source and --build')
        return launch(args)
    if args.destination is None: p.error('export requires --destination')
    return export(args)


if __name__ == '__main__':
    raise SystemExit(main())
