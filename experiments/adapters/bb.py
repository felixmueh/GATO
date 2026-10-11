#!/usr/bin/env python3
"""One frozen BB episode, using the existing worker and compact recording hooks.

This adapter never resumes the historical campaign manager. A new reviewed plan
must explicitly map completed old episodes before continuing its remaining cases.
"""
import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback
from types import SimpleNamespace

EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS))
from recording import RawRecorder, sha, write_json


def load_worker(source, module):
    sys.path[:0] = [str(source / 'examples/randomized_multimodal'),
                    str(source / 'python'), str(source / 'tiago_src')]
    importlib.import_module('bsqp')
    name = 'bsqp.bsqpN32_tiago_right'
    spec = importlib.util.spec_from_file_location(name, module)
    native = importlib.util.module_from_spec(spec)
    sys.modules[name] = native
    spec.loader.exec_module(native)
    worker = importlib.import_module('tiago_reaching_batch_pilot')
    if Path(worker.__file__).resolve() != source / 'examples/randomized_multimodal/tiago_reaching_batch_pilot.py':
        raise ValueError('BB episode worker imported from a different source')
    return worker


def run_case(source, module, pool, protocol_path, case, output, *, machine=None,
             preflight_input=None, worker=None):
    source, module, pool, protocol_path, output = map(lambda p: Path(p).resolve(), (source, module, pool, protocol_path, output))
    result = dict(campaign_id='BB-readiness' if preflight_input else 'BB', research_question='RQ-batching',
                  execution_backend='internal', timing_mode='B', offline_timing='solve_time', machine=machine,
                  collection_protocol='compact-recording-v1',
                  evidence_kind='synthetic_test' if worker is not None else 'measurement',
                  task_success=None, physical_pass=None, solver_numerical_pass=None, execution_error=None)
    previous = {}

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'BB interrupted by signal {signum}')

    with RawRecorder(output) as recorder:
        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                previous[sig] = signal.signal(sig, interrupted)
            protocol = json.loads(protocol_path.read_text())
            if protocol.get('timing_mode') != 'B' or protocol.get('offline_timing') != 'solve_time':
                raise ValueError('BB requires frozen mode B protocol')
            pair = case['parameters']['pair_id']
            batch = case['parameters']['batch']
            if not isinstance(pair, int) or isinstance(pair, bool):
                raise ValueError('BB pair_id must be an integer')
            if batch not in (1, 16) or isinstance(batch, bool):
                raise ValueError('BB supports the original B1/B16 arms')
            if preflight_input is None and pair not in protocol['task_ids']:
                raise ValueError('Case pair is outside the frozen BB roster')
            write_json(output / 'protocol.json', dict(protocol, collection_protocol=result['collection_protocol']))
            write_json(output / 'provenance.json', dict(
                source_commit=subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
                module_path=str(module), module_sha256=sha(module), adapter_sha256=sha(__file__),
                frozen_protocol_path=str(protocol_path), frozen_protocol_sha256=sha(protocol_path),
                inputs={name: sha(pool / name) for name in ('pool.npz', 'pairs.npz')},
                machine=machine, historical_campaign_manager_used=False,
                recording='bounded raw event chunks; full executed history and physical summary only on finalization'))
            worker = worker or load_worker(source, module)
            args = SimpleNamespace(protocol=protocol_path, pool=pool, output=output / 'episode',
                preflight_input=Path(preflight_input) if preflight_input else None,
                evidence_kind=result['evidence_kind'], feedback=.03, sim_dt=.003, seconds=8.,
                position_tolerance=.05, tool_axis_tolerance_deg=30., velocity_l1_tolerance=1.,
                seed=20260925, vel_lim_cost=.1, qd_cost=.1, prefix_guard=False)
            summary = worker.run_episode(args, pair, batch, .03,
                                         recorder=recorder, final_checkpoint_only=True)
            if not summary.get('complete'):
                raise ValueError('BB episode returned an incomplete summary')
            result.update(summary)
            result.update(task_success=summary['task_success'],
                          physical_pass=summary['physical']['physical_valid'], execution_error=None)
        except BaseException:
            result['execution_error'] = traceback.format_exc()
            recorder('execution_failed', {'error': result['execution_error']})
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            result['outcome'] = 'execution_failed' if result['execution_error'] else result.get('outcome', 'unknown')
            write_json(output / 'metrics.json', result)
            recorder.manifest()
    # Scientific task failures are completed evidence; a failed preflight holds.
    return 1 if result['execution_error'] or (preflight_input and not
                    (result['task_success'] and result['physical_pass'])) else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=os.environ.get('GATO_SOURCE_DIR'), required=False)
    parser.add_argument('--build', type=Path, default=os.environ.get('GATO_BUILD_DIR'))
    parser.add_argument('--module', type=Path)
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--case', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight-input', type=Path)
    parser.add_argument('--batch', type=int, choices=(1, 16), help='preflight arm')
    args = parser.parse_args()
    if not args.source or not (args.module or args.build):
        parser.error('--source and --build (or --module) are required')
    if args.preflight_input:
        if args.batch is None:
            parser.error('--batch is required with --preflight-input')
        case = dict(id='preflight', parameters=dict(pair_id=-1, batch=args.batch))
    elif args.case:
        case = json.loads(args.case.read_text())
    else:
        parser.error('--case or --preflight-input is required')
    if args.module:
        module = args.module
    else:
        matches = list((args.build / 'modules').glob('bsqpN32_tiago_right*.so'))
        if len(matches) != 1:
            parser.error('Expected one N32 TIAGo module under the build/modules directory')
        module = matches[0]
    return run_case(args.source, module, args.pool, args.protocol, case, args.output,
                    machine=os.environ.get('GATO_MACHINE'), preflight_input=args.preflight_input)


if __name__ == '__main__':
    raise SystemExit(main())
