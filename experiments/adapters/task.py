#!/usr/bin/env python3
"""Run a frozen T01 task or canonical A/B preflight with the existing controller."""
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

import numpy as np

EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS))
from recording import sha, write_json
from adapters.task_recording import TaskRecorder, TASK_RECORDING_FORMAT


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_helpers(source):
    # Historical task helpers use sibling imports. Only import their pure functions;
    # the old run_case/release orchestration is never called.
    sys.path.insert(0, str(source / 'experiments/campaign'))
    sys.modules['raw'] = importlib.import_module('recording')
    return load('saved_task_helpers', source / 'experiments/campaign/run_t01.py')


def select_protocol(source, case, mode, preflight, inputs=None):
    inputs = Path(inputs) if inputs is not None else source / 'experiments/campaign/inputs'
    if preflight:
        protocol = json.loads((inputs / 'v01.json').read_text())
        protocol['timing_mode'] = mode
        protocol['offline_timing'] = 'controller_dt' if mode == 'A' else 'solve_time'
        if mode == 'A':
            protocol.update(timing_scope='native and wrapper solve time; computation ignored by plant',
                            implemented_clock_convention='fixed nominal feedback intervals',
                            approved_clock_convention='mode A fixed feedback; computation time ignored',
                            startup_convention='cold solve activated at0; bootstrap wall unpaid',
                            terminal_convention='rough reaching checked after nominal feedback intervals')
    else:
        if mode != 'B':
            raise ValueError('Frozen T01 tasks use mode B')
        cohort = json.loads((inputs / 't01.json').read_text())
        tasks = [task for task in cohort['tasks'] if task['id'] == case['id']]
        if len(tasks) != 1:
            raise ValueError('Case must name one frozen T01 task')
        protocol = {key: value for key, value in cohort.items() if key != 'tasks'}
        protocol.update(tasks[0])
        protocol['solver_params'] = cohort[protocol['kind'] + '_solver_params']
    # Historical release receipts remain in the original input; they are not
    # executable gates in this collector. Scientific parameters stay unchanged.
    historical = {key: protocol.pop(key) for key in list(protocol)
                  if key.startswith(('release_', 'reviewed_')) or key in ('family_hold_policy', 'clock_decision_record')}
    protocol['historical_release_metadata'] = historical
    protocol['collection_protocol'] = 'simple-harness-v1'
    protocol['repeat_policy'] = 'one planned attempt; explicit retries retain prior failures'
    return protocol


def native_module(source, build, N=32):
    matches = list((build / 'modules').glob(f'bsqpN{N}_tiago_right*.so'))
    if len(matches) != 1:
        raise ValueError(f'Expected one N{N} module in {build / "modules"}')
    sys.path[:0] = [str(source / 'python'), str(source / 'tiago_src')]
    importlib.import_module('bsqp')
    name = f'bsqp.bsqpN{N}_tiago_right'
    module = load(name, matches[0])
    sys.modules[name] = module
    return module, matches[0]


def execute(controller, protocol, emit, helpers):
    if protocol['kind'] == 'tracking':
        return helpers.execute_episode(controller, protocol, emit)
    return controller.run_mpc_goals(np.asarray(protocol['initial_state'], dtype=np.float32),
        [np.asarray(protocol['goal'], dtype=np.float32)], sim_dt=protocol['sim_dt_s'],
        goal_timeout=protocol['timeout_s'], goal_threshold=protocol['position_threshold_m'],
        goal_axis_threshold=np.deg2rad(protocol['axis_threshold_deg']),
        velocity_threshold=protocol['speed_L1_threshold_rad_s'], goal_dwell_time=0.,
        reset_rho_each_solve=False, initial_guess=None, offline_timing=protocol['offline_timing'],
        record_raw=True, record_callback=emit)


def run(source, build, case, output, mode='B', preflight=False, inputs=None):
    source, build, output = map(lambda p: Path(p).resolve(), (source, build, output))
    # Opening the recorder first rejects old attempts before overwriting evidence.
    with TaskRecorder(output) as recorder:
        return _run(source, build, case, output, mode, preflight, recorder, inputs)


def _run(source, build, case, output, mode, preflight, recorder, inputs):
    try:
        protocol = select_protocol(source, case, mode, preflight, inputs)
        write_json(output / 'protocol.json', protocol)
        helpers = load_helpers(source)
    except BaseException:
        error = traceback.format_exc()
        recorder('execution_failed', {'error': error})
        write_json(output / 'metrics.json', dict(execution_error=error, outcome='execution_failed',
                   task_success=None, physical_pass=None, timing_mode=mode))
        recorder.manifest()
        return 1
    controller = None
    stats, error = {}, None
    metrics = dict(research_question='RQ-control', campaign_id='V01' if preflight else 'T01',
                   execution_backend='internal', timing_mode=mode, offline_timing=protocol['offline_timing'],
                   machine=os.environ.get('GATO_MACHINE'), configuration_file='protocol.json',
                   recording_format=TASK_RECORDING_FORMAT,
                   task_success=None, physical_pass=None, solver_numerical_pass=None)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        native, module_path = native_module(source, build)
        import pinocchio as pin
        production = importlib.import_module('gato_tiago.tiago_mpc_controller')
        if Path(production.__file__).resolve() != source / 'tiago_src/gato_tiago/tiago_mpc_controller.py':
            raise ValueError('Controller imported from a different source')
        urdf = source / protocol['urdf']['path']
        if sha(urdf) != protocol['urdf']['sha256']:
            raise ValueError('Frozen URDF changed')
        model = pin.buildModelFromUrdf(str(urdf))
        model.gravity.linear = np.asarray(protocol['gravity_m_s2'])
        controller = production.MPC_GATO(model, str(urdf), N=32, dt=.03, batch_size=1,
            constant_f_ext=np.asarray(protocol['constant_f_ext'], dtype=np.float32), track_full_stats=True,
            plant_type='tiago_right', pendulum_config=None, solver_params=protocol['solver_params'], final_approach=False)
        if controller.solver.lib is not native or controller.solver_params != protocol['solver_params']:
            raise ValueError('Effective native module or solver parameters differ')
        source_commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        write_json(output / 'provenance.json', dict(source_commit=source_commit,
            module_path=str(module_path), module_sha256=sha(module_path),
            collector_sha256=sha(__file__), protocol_sha256=sha(output / 'protocol.json'),
            machine=os.environ.get('GATO_MACHINE')))
        _, stats = execute(controller, protocol, recorder, helpers)
    except BaseException:
        error = traceback.format_exc()
        stats = getattr(controller, 'last_run_stats', {}) if controller is not None else {}
        recorder('execution_failed', {'error': error})
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        complete = [step for step in stats.get('raw_execution', []) if 'state_after' in step]
        x = np.asarray([protocol['initial_state']] + [step['state_after'] for step in complete])
        u = np.asarray([step['control'] for step in complete]).reshape(-1, 7)
        t = np.asarray([0.] + [step['time_s'] + step['step_dt_s'] for step in complete])
        np.savez(output / 'execution.npz', x=x, u=u, t=t)
        recorder.write_controller_stats(stats)
        try:
            if controller is not None:
                metrics.update(helpers.trace_metrics(protocol, controller.model, x, u, t, stats, error, output))
        except Exception:
            error = (error or '') + '\nMetric evaluation failed:\n' + traceback.format_exc()
        metrics.update(execution_error=error, outcome='execution_failed' if error else
                       ('task_passed' if metrics['task_success'] else 'task_failed'))
        solves = stats.get('raw_solves', [])
        metrics['timing_distributions'] = {
            'wall_ms': helpers.distribution([s.get('wall_solve_time_s', np.nan) * 1000 for s in solves]),
            'native_ms': helpers.distribution([s.get('native_solve_time_ms', np.nan) for s in solves]),
            'update_s': helpers.distribution(stats.get('update_intervals_s', []))}
        metrics['timing_scope'] = protocol['timing_scope']
        intervals = np.asarray(stats.get('update_intervals_s', []))
        metrics['update_fraction_above_nominal'] = float(np.mean(intervals > .03)) if intervals.size and np.isfinite(intervals).all() else None
        write_json(output / 'metrics.json', metrics)
        recorder.manifest()
        recorder.close()
    return 1 if error or (preflight and not (metrics['task_success'] and metrics['physical_pass'])) else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--case', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, help='Declared frozen directory containing v01.json and t01.json')
    parser.add_argument('--timing-mode', choices=['A', 'B'], default='B')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if not args.preflight and not args.case:
        parser.error('--case is required for a task')
    case = json.loads(args.case.read_text()) if args.case else {}
    return run(args.source, args.build, case, args.output, args.timing_mode, args.preflight, args.inputs)


if __name__ == '__main__':
    raise SystemExit(main())
