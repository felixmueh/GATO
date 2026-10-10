"""Prepare or resume a fixed-preset randomized reaching MPC campaign.

Each B1/B16 episode runs in an isolated child; partial attempts are immutable.
The sampling seed and the unchanged controller perturbation seed are distinct.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import numpy as np
import pinocchio as pin

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT/'examples/randomized_multimodal/tiago_reaching_batch_pilot.py'
sys.path.insert(0, str(ROOT/'tiago_src'))
from validate_reaching_mpc import load_protocol, COHORT_ID, PROTOCOL_PATH

FIXED_ARGS = ['--knot-dts', '.03', '--feedback', '.03', '--sim-dt', '.003',
              '--seconds', '8', '--position-tolerance', '.05',
              '--tool-axis-tolerance-deg', '30', '--velocity-l1-tolerance', '1',
              '--seed', '20260925', '--vel-lim-cost', '.1', '--qd-cost', '.1']
EXCLUDED = [0, 636, 1554]


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text())


def provenance(pool):
    sources = [Path(__file__), RUNNER,
               ROOT/'examples/randomized_multimodal/validate_reaching_mpc.py',
               ROOT/'tiago_src/gato_tiago/config.py',
               ROOT/'gato/dynamics/tiago_right/tiago_right_plant.cuh',
               ROOT/'gato/dynamics/tiago_right/tiago_right_arm.urdf',PROTOCOL_PATH]
    sources += sorted((ROOT/'python/bsqp').glob('*.py'))
    binaries = sorted((ROOT/'python/bsqp').glob('bsqpN32_tiago_right*.so'))
    if len(binaries) != 1:
        raise ValueError(f'expected one existing N32 Tiago binary, found {len(binaries)}')
    return dict(source_sha256={str(p.relative_to(ROOT)):digest(p) for p in sources},
                input_sha256={name:digest(pool/name) for name in ('pool.npz', 'pairs.npz')},
                binary_relative_path=str(binaries[0].relative_to(ROOT)),
                binary_sha256=digest(binaries[0]),
                versions=dict(python=platform.python_version(), numpy=np.__version__, pinocchio=pin.__version__),
                fixed_runner_args=FIXED_ARGS,
                solver_params=dict(load_protocol()['solver_params']))


def journal(root, event, **fields):
    with (root/'journal.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(time=now(), event=event, **fields), allow_nan=False)+'\n')
        stream.flush(); os.fsync(stream.fileno())


def save_status(root, status):
    status['updated'] = now()
    atomic_json(root/'status.json', status)


@contextmanager
def campaign_lock(root):
    with (root/'.manager.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('another manager holds this campaign lock') from None
        yield


def prepare(args, root):
    if (root/'manifest.json').exists() or (root/'status.json').exists():
        raise ValueError('campaign exists; use --resume or choose a new output')
    pool = args.pool.resolve()
    historical=read_json(args.pair_manifest)
    protocol=load_protocol()
    if digest(args.pair_manifest)!=protocol['input_sha256']['historical_manifest.json']:
        raise ValueError('Exact historical100 task manifest changed')
    if args.count!=100 or args.sample_seed!=historical['sample_seed']:
        raise ValueError('Current cohort retains exact100 frozen tasks and sample policy')
    for name in ('pool.npz','pairs.npz'):
        if digest(pool/name)!=historical['input_sha256'][name]:raise ValueError('Frozen task pool changed')
    ids=list(historical['pair_ids']);selected=list(historical['pairs'])
    if ids!=protocol['task_ids'] or len(ids)!=100 or len(set(ids))!=100:
        raise ValueError('Exact ordered100 task IDs required')
    manifest = dict(schema_version=1, cohort_id=COHORT_ID, timing_mode='A',
                    protocol_sha256=digest(PROTOCOL_PATH),frozen_pair_manifest_sha256=digest(args.pair_manifest),created=now(), pair_ids=ids, pairs=selected,
                    batches=[1,16], count=args.count, sample_seed=args.sample_seed,
                    episode_seed=20260925, excluded_pair_ids=EXCLUDED,
                    sampling='uniform permutation without replacement of existing pair IDs; no IK or task filtering',
                    source_pool=str(pool), source_root=str(ROOT), python_executable=sys.executable,
                    commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                    host=platform.node(), platform=platform.platform(),
                    wall_timeout_seconds=args.wall_timeout,
                    N=32, knot_dt=.03, prediction_seconds=.93, feedback_seconds=.03,
                    simulation_step=.003, simulation_timeout=8.,
                    goal='XYZ < .05 m, tool-Z-axis angle < 30 deg, joint velocity L1 < 1 rad/s; no dwell',
                    physical_limits='soft optimization penalties; independent executed-prefix checks; no collision checks',
                    initialization='existing pilot: identical constant-state/zero-control cold seed; shifted winner, fresh target-independent B16 perturbations thereafter',
                    **provenance(pool))
    atomic_json(root/'manifest.json', manifest)
    status = dict(schema_version=1, manifest_sha256=digest(root/'manifest.json'), created=now(),
                  episodes=[dict(pair_id=i, batch=b, state='pending', attempts=[]) for i in ids for b in [1,16]])
    save_status(root, status)
    journal(root, 'prepared', pairs=len(ids), episodes=len(status['episodes']))
    print(f'Prepared {len(ids)} pairs / {len(status["episodes"])} episodes: {root}', flush=True)
    return manifest, status


def verify_receipt(root, attempt):
    receipt_path = root/attempt['receipt_path']
    if digest(receipt_path) != attempt['receipt_sha256']:
        raise ValueError(f'attempt receipt changed: {receipt_path}')
    for relative, expected in read_json(receipt_path)['file_sha256'].items():
        if digest(root/relative) != expected:
            raise ValueError(f'artifact hash mismatch: {relative}')


def resume(args, root):
    manifest, status = read_json(root/'manifest.json'), read_json(root/'status.json')
    if manifest.get('cohort_id')!=COHORT_ID:raise ValueError('Historical cohort resume forbidden; choose fresh current output')
    if status.get('family_hold'):raise ValueError('Current family held after executionfailure')
    if digest(root/'manifest.json') != status['manifest_sha256']:
        raise ValueError('campaign manifest changed')
    pool = args.pool.resolve() if args.pool is not None else Path(manifest['source_pool'])
    current = provenance(pool)
    for key, value in current.items():
        if manifest.get(key) != value:
            raise ValueError(f'resume provenance mismatch: {key}')
    for episode in status['episodes']:
        for attempt in episode['attempts']:
            if 'receipt_path' in attempt:
                verify_receipt(root, attempt)
        if episode['state'] == 'running':
            attempt = episode['attempts'][-1]
            # A live orphan may still own the artifact files. Do not reuse them.
            pid = attempt.get('pid')
            if pid and Path(f'/proc/{pid}').exists():
                status['family_hold']=dict(pair_id=episode['pair_id'],batch=episode['batch'],reason='prior child still present')
                save_status(root,status)
                raise ValueError(f'prior child PID {pid} is still present; stop it before resuming')
            finish_attempt(root, status, episode, attempt, 'interrupted',
                           error='manager stopped before recording child completion')
    if status.get('family_hold'):raise ValueError('Current family held after interruption/orphan recovery')
    journal(root, 'resumed')
    return manifest, status, pool


def finish_attempt(root, status, episode, attempt, state, **fields):
    attempt.update(state=state, finished=now(), **fields)
    episode['state'] = state
    if state in ('technical_failure','interrupted'):
        status['family_hold']=dict(pair_id=episode['pair_id'],batch=episode['batch'],reason=fields.get('error',state))
        save_status(root,status)  # Hold before receipt hashing can fail/interruption propagates.
    directory = root/attempt['relative_output']
    receipt = directory/'attempt_receipt.json'
    atomic_json(receipt, dict(state=state, finished=attempt['finished'],
        file_sha256={str(p.relative_to(root)):digest(p) for p in sorted(directory.rglob('*'))
                     if p.is_file() and p not in (receipt, receipt.with_suffix('.json.tmp'))}))
    attempt['receipt_path'] = str(receipt.relative_to(root))
    attempt['receipt_sha256'] = digest(receipt)
    save_status(root, status)
    journal(root, 'attempt_finished', pair_id=episode['pair_id'], batch=episode['batch'], **attempt)


def terminate_child(child):
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def run_attempt(args, root, manifest, status, episode, pool):
    number = len(episode['attempts'])+1
    directory = root/'attempts'/f'pair_{episode["pair_id"]:05d}_B{episode["batch"]}'/f'attempt_{number:04d}'
    directory.mkdir(parents=True, exist_ok=False)
    child_output = directory/'output'
    command = [sys.executable, str(RUNNER), '--pool', str(pool), '--output', str(child_output),
               '--pairs', str(episode['pair_id']), '--batches', str(episode['batch']), *FIXED_ARGS]
    attempt = dict(attempt=number, relative_output=str(directory.relative_to(root)),
                   log_path=str((directory/'episode.log').relative_to(root)), state='running', started=now(),
                   command=command)
    episode['attempts'].append(attempt); episode['state']='running'
    save_status(root, status)
    journal(root, 'attempt_started', pair_id=episode['pair_id'], batch=episode['batch'], **attempt)
    child = None; started = time.monotonic()
    try:
        with (directory/'episode.log').open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            attempt['pid'] = child.pid; save_status(root, status)
            try:
                code = child.wait(timeout=manifest['wall_timeout_seconds'])
            except subprocess.TimeoutExpired:
                terminate_child(child)
                finish_attempt(root, status, episode, attempt, 'technical_failure', returncode=child.returncode,
                               wall_seconds=time.monotonic()-started, error='child wall-clock timeout; distinct from eight-second simulated timeout')
                return
        inner = child_output/f'pair_{episode["pair_id"]:05d}_dt_0.03_B{episode["batch"]}'
        if code != 0:
            raise RuntimeError(f'child exited with code {code}')
        summary = read_json(inner/'summary.json')
        if not summary.get('complete'):
            raise RuntimeError('child summary is incomplete')
        for name, expected in read_json(inner/'receipt.json').items():
            if digest(inner/name) != expected:
                raise RuntimeError(f'inner artifact hash mismatch: {name}')
        finish_attempt(root, status, episode, attempt, 'completed', returncode=code,
                       wall_seconds=time.monotonic()-started, outcome=summary['outcome'],
                       summary_path=str((inner/'summary.json').relative_to(root)),
                       inner_receipt_path=str((inner/'receipt.json').relative_to(root)))
    except (KeyboardInterrupt, InterruptedError):
        if child is not None: terminate_child(child)
        finish_attempt(root, status, episode, attempt, 'interrupted',
                       returncode=None if child is None else child.returncode,
                       wall_seconds=time.monotonic()-started, error='manager interrupted; partial attempt retained')
        raise
    except Exception as error:
        if child is not None: terminate_child(child)
        finish_attempt(root, status, episode, attempt, 'technical_failure',
                       returncode=None if child is None else child.returncode,
                       wall_seconds=time.monotonic()-started, error=str(error))
    finally:
        if child is not None: terminate_child(child)
        last = episode['attempts'][-1]
        print(f'Pair {episode["pair_id"]} B{episode["batch"]} attempt {number}: '
              f'{episode["state"]} {last.get("outcome", last.get("error", ""))} '
              f'({time.monotonic()-started:.1f} wall s)', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pool', type=Path, default=None, help='default: repository reaching pool; resume can relocate identical inputs')
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--sample-seed', type=int, default=20260926)
    parser.add_argument('--wall-timeout', type=float, default=120.)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retry-technical-failures', action='store_true', help='explicitly append new attempts for prior technical failures')
    parser.add_argument('--max-pairs', type=int, default=None, help='bounded execution prefix this invocation; manifest remains unchanged')
    parser.add_argument('--pair-manifest',type=Path,default=ROOT/'experiments/tiago_batching/historical_manifest.json')
    args = parser.parse_args()
    if args.retry_technical_failures:parser.error('Current cohort holds on executionfailure; no retries')
    if not np.isfinite(args.wall_timeout) or args.wall_timeout <= 0:
        parser.error('wall-timeout must be finite and positive')
    if args.max_pairs is not None and args.max_pairs < 1:
        parser.error('max-pairs must be positive')
    if args.retry_technical_failures and not args.resume:
        parser.error('retry-technical-failures requires --resume')
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=True)
    def interrupted(signum, frame):
        raise InterruptedError(f'received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        with campaign_lock(root):
            if args.resume:
                manifest, status, pool = resume(args, root)
            else:
                if args.pool is None: args.pool=ROOT/'experiments/tiago_batching/inputs'
                manifest, status = prepare(args, root); pool=args.pool.resolve()
            if args.prepare_only: return
            eligible_pairs = manifest['pair_ids'][:args.max_pairs] if args.max_pairs else manifest['pair_ids']
            for episode in status['episodes']:
                if episode['pair_id'] not in eligible_pairs: continue
                if episode['state'] == 'completed': continue
                if episode['state'] == 'technical_failure' and not args.retry_technical_failures: continue
                run_attempt(args, root, manifest, status, episode, pool)
                if episode['state'] in ('technical_failure','interrupted'):
                    status['family_hold']=dict(pair_id=episode['pair_id'],batch=episode['batch'],reason=episode['attempts'][-1].get('error'))
                    save_status(root,status);journal(root,'family_held',**status['family_hold'])
                    return 1
            counts = {state:sum(e['state']==state for e in status['episodes'])
                      for state in ('pending', 'completed', 'technical_failure', 'interrupted')}
            journal(root, 'invocation_finished', counts=counts)
            print(json.dumps(counts), flush=True)
    except (KeyboardInterrupt, InterruptedError):
        journal(root, 'manager_interrupted')
        return 130
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    return 0


if __name__=='__main__':
    sys.exit(main())
