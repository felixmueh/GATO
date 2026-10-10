"""Explicit frozen V01 runner worker; no ROS, robot, sweep, rescue or tuning."""
import argparse
import importlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import traceback

import numpy as np

from evaluate import distribution, evaluate
from raw import RawRecorder, sha, write_json
from t01_protocol import diagnostic_protocol

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / 'experiments/campaign/inputs/v01.json'


def load_protocol(case, inputs=None):
    parameters = case.get('parameters', {})
    expected = {'protocol': 'V01', 'offline_timing': 'solve_time'}
    mode = parameters.get('offline_timing', 'solve_time')
    if mode == 'controller_dt':
        expected['offline_timing'] = mode
        expected['diagnostic_after_failed_B'] = parameters.get('diagnostic_after_failed_B')
    if parameters != expected:
        raise ValueError('Case must match frozen V01 parameters; A requires hashed diagnostic_after_failed_B evidence')
    protocol = json.loads(PROTOCOL.read_text())
    if mode == 'controller_dt':
        evidence = parameters.get('diagnostic_after_failed_B')
        if not isinstance(evidence, dict) or set(evidence) != {'path', 'sha256'}:
            raise ValueError('A diagnostic requires hashed failed-B metrics evidence')
        relative = Path(evidence['path'])
        if relative.is_absolute() or '..' in relative.parts or inputs is None:
            raise ValueError('A diagnostic evidence must be a declared runner input')
        source = Path(inputs)/relative
        if sha(source) != evidence['sha256']:
            raise ValueError('Failed-B evidence hash mismatch')
        failure = json.loads(source.read_text())
        gates_known = all(type(failure.get(key)) is bool for key in ('task_success', 'physical_pass'))
        scientific_negative = (gates_known and (failure['task_success'] is False or failure['physical_pass'] is False))
        if (failure.get('timing_mode') != 'B' or failure.get('execution_backend') != 'internal'
            or failure.get('outcome') not in {'task_passed', 'task_failed'}
            or failure.get('execution_error') is not None or not scientific_negative
            or failure.get('configuration') != protocol
            or failure.get('machine') != os.environ.get('GATO_MACHINE')):
            raise ValueError('A diagnostic requires completed same-machine B rough-reaching or physical failure under this exact protocol')
        protocol = diagnostic_protocol(protocol)
    return protocol


def run_case(case, output, inputs=None):
    protocol = load_protocol(case, inputs)
    if protocol.get('release_status') != 'released after reviewed clock decision':
        raise ValueError('Campaign execution held by frozen release_status; no native/controller work authorized')
    if os.environ.get('GATO_MACHINE') not in protocol.get('release_machines', []):
        raise ValueError('Machine is outside frozen desktop release_machines; no native/controller work authorized')
    for relative, expected in protocol.get('reviewed_controller_sha256', {}).items():
        if sha(ROOT/relative) != expected:
            raise ValueError(f'Reviewed controller source changed: {relative}; no native/controller work authorized')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'events.jsonl').exists() or (output/'metrics.json').exists():
        raise ValueError('Output already contains a campaign attempt; use a fresh runner directory')
    write_json(output/'protocol.json', protocol)
    recorder = RawRecorder(output)
    stats, controller, model = {}, None, None
    error = None
    metrics = {'research_question': 'RQ-control', 'campaign_id': protocol['campaign_id'],
               'execution_backend': 'internal', 'timing_mode': protocol['timing_mode'],
               'offline_timing': protocol['offline_timing'], 'batch_size': 1,
               'configuration': protocol, 'known_limitations': protocol['known_limitations'],
               'machine': os.environ.get('GATO_MACHINE'),
               'timing_scope': protocol['timing_scope'],
               'startup_convention': protocol['startup_convention'],
               'interval_alignment': protocol['implemented_clock_convention'],
               'nominal_feedback_s': protocol['nominal_feedback_s']}
    old_handlers = {}

    def interrupted(signum, frame):
        raise InterruptedError(f'campaign interrupted by signal {signum}')

    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            old_handlers[sig] = signal.signal(sig, interrupted)
        # Explicit runner build source ensures CMake's source/python output is
        # isolated; local root qualification uses its own dedicated module.
        sys.path[:0] = [str(ROOT/'python'), str(ROOT/'tiago_src')]
        import pinocchio as pin
        import torch
        from gato_tiago import tiago_mpc_controller as production
        urdf = ROOT/protocol['urdf']['path']
        if sha(urdf) != protocol['urdf']['sha256']:
            raise RuntimeError('URDF differs from frozen protocol')
        fixture = ROOT/protocol['frozen_fixture']['path']
        if sha(fixture) != protocol['frozen_fixture']['sha256']:
            raise RuntimeError('Original task fixture differs from frozen receipt')
        model = pin.buildModelFromUrdf(str(urdf))
        initial = np.asarray(protocol['initial_state'], dtype=np.float32)
        endpoint = initial[:7].astype(np.float64)+np.asarray(protocol['endpoint_delta_rad'])
        goal = np.asarray(protocol['goal'], dtype=np.float32)
        data = model.createData()
        pin.framesForwardKinematics(model, data, initial[:7].astype(np.float64))
        start_pose = data.oMf[model.getFrameId('arm_right_tool_link')].copy()
        initial_distance = float(np.linalg.norm(start_pose.translation-goal[:3]))
        screen = {'initial_state':initial, 'goal':goal, 'initial_distance_m':initial_distance,
            'initial_pose_xyz_rpy':np.r_[start_pose.translation, pin.rpy.matrixToRpy(start_pose.rotation)],
            'initial_q_in_model_limits':bool(np.all(initial[:7]>=model.lowerPositionLimit)
                                           and np.all(initial[:7]<=model.upperPositionLimit)),
            'initial_velocity_in_model_limits':bool(np.all(abs(initial[7:])<=model.velocityLimit)),
            'endpoint_q_in_model_limits':bool(np.all(endpoint>=model.lowerPositionLimit)
                                            and np.all(endpoint<=model.upperPositionLimit)),
            'necessary_endpoint_velocity_time_bound_s':float(np.max(abs(endpoint-initial[:7])/model.velocityLimit)),
            'collision_scope':protocol['collision_scope'],
            'screening_scope':'FK joint endpoint witness and necessary velocity bound only; sampled8s inverse-dynamics witness is separately retained in T01 proposal'}
        pin.framesForwardKinematics(model, data, endpoint)
        endpoint_pose = data.oMf[model.getFrameId('arm_right_tool_link')]
        screen['endpoint_FK_matches_frozen_goal'] = bool(
            np.linalg.norm(endpoint_pose.translation-goal[:3])<1e-6
            and np.linalg.norm(pin.log3(pin.rpy.rpyToMatrix(*map(float, goal[3:])).T@endpoint_pose.rotation))<1e-6)
        write_json(output/'screening.json', screen)
        if (not np.isfinite(initial).all() or not np.isfinite(goal).all()
            or initial_distance <= protocol['position_threshold_m']
            or not all(screen[k] for k in ('initial_q_in_model_limits', 'initial_velocity_in_model_limits',
                                         'endpoint_q_in_model_limits', 'endpoint_FK_matches_frozen_goal'))
            or screen['necessary_endpoint_velocity_time_bound_s'] > protocol['timeout_s']):
            raise RuntimeError('Frozen initial-state/FK/time screening failed before solver construction')
        native = importlib.import_module('bsqp.bsqpN32_tiago_right')
        if Path(native.__file__).resolve().parent != (ROOT/'python/bsqp').resolve():
            raise RuntimeError('Native module is outside isolated experiment source')
        controller = production.MPC_GATO(model, str(urdf), N=32, dt=.03, batch_size=1,
            constant_f_ext=np.zeros(6, dtype=np.float32), track_full_stats=True,
            plant_type='tiago_right', pendulum_config=None,
            solver_params=protocol['solver_params'], final_approach=False)
        if controller.solver.lib is not native:
            raise RuntimeError('Production controller loaded unexpected native module')
        if controller.solver_params != protocol['solver_params']:
            raise RuntimeError('Resolved solver settings differ from frozen effective public values')
        runner_source = Path(os.environ.get('GATO_SOURCE_DIR', ROOT))
        source_commit = subprocess.check_output(['git', '-C', str(runner_source), 'rev-parse', 'HEAD'], text=True).strip()
        sources = ['python/bsqp/interface.py', 'python/bsqp/common.py', 'python/bsqp/offline_mpc.py',
                   'tiago_src/gato_tiago/tiago_mpc_controller.py', 'tiago_src/gato_tiago/config.py',
                   'experiments/campaign/run_v01.py', 'experiments/campaign/t01_protocol.py', 'experiments/campaign/raw.py', 'experiments/campaign/evaluate.py']
        build_receipt = Path(os.environ.get('GATO_CAMPAIGN_BUILD_RECEIPT', ROOT.parent/'build_receipt.json'))
        receipt_info = {'path':None, 'sha256':None, 'unmeasured_reason':'no explicit build receipt available for standalone invocation'}
        if build_receipt.is_file():
            shutil.copyfile(build_receipt, output/'build_receipt.json')
            receipt_info = {'path':str(build_receipt), 'sha256':sha(build_receipt),
                            'copied_artifact':'build_receipt.json'}
        write_json(output/'provenance.json', {'source_root':str(ROOT), 'source_commit':source_commit,
            'protocol_sha256':sha(PROTOCOL), 'module_path':native.__file__, 'module_sha256':sha(native.__file__),
            'task_config_sha256':sha(output/'protocol.json'), 'case':case,
            'source_sha256':{name:sha(ROOT/name) for name in sources}, 'urdf_sha256':sha(urdf),
            'python':sys.version, 'numpy':np.__version__, 'pinocchio':pin.__version__, 'torch':torch.__version__,
            'device':torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            'machine':os.environ.get('GATO_MACHINE'), 'build_receipt':receipt_info})
        # Readiness is independent of scientific task gates. The runner-isolated
        # executable must be run, not merely compiled, before task execution.
        readiness = ROOT.parent/'campaign-native/cuda_access_smoke'
        if os.environ.get('GATO_BUILD_DIR'):
            if not readiness.is_file():
                raise RuntimeError('Runner readiness smoke is missing')
            subprocess.run([str(readiness)], check=True)
            write_json(output/'readiness.json', {'command':[str(readiness)], 'exit_code':0,
                                                 'sha256':sha(readiness)})
        else:
            write_json(output/'readiness.json', {'executed':False,
                'unmeasured_reason':'standalone invocation; root must retain separately executed readiness evidence'})
        _, stats = controller.run_mpc_goals(initial,
            [goal], sim_dt=.003,
            goal_timeout=8., goal_threshold=.05, goal_axis_threshold=np.deg2rad(30.),
            velocity_threshold=1., goal_dwell_time=0., reset_rho_each_solve=False,
            initial_guess=None, offline_timing=protocol['offline_timing'], record_raw=True,
            record_callback=recorder)
    except BaseException:
        error = traceback.format_exc()
        if controller is not None:
            stats = getattr(controller, 'last_run_stats', {})
        recorder('execution_failed', {'error':error})
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        # Completed event observers precede lane selection inside controller;
        # retain the final/partial dictionaries too so selection metadata and
        # any native error buffers are not lost when the run raises.
        recorder('final_raw_solves', {'solves':stats.get('raw_solves', []),
                                     'execution_statuses':[s.get('status') for s in stats.get('raw_execution', [])]})
        initial = np.asarray(protocol['initial_state'], dtype=np.float32)
        steps = stats.get('raw_execution', [])
        complete = [step for step in steps if 'state_after' in step]
        # Raw includes started/failed steps too. Only returned RK4 states define
        # actual executed samples; no finite-value filtering is performed.
        x = np.asarray([initial] + [s['state_after'] for s in complete])
        u = np.asarray([s['control'] for s in complete]).reshape(-1, 7)
        t = np.asarray([0.] + [s['time_s']+s['step_dt_s'] for s in complete])
        np.savez_compressed(output/'execution.npz', x=x, u=u, t=t)
        write_json(output/'controller_stats.json', {k:v for k,v in stats.items()
                    if k not in {'raw_solves', 'raw_execution'}})
        try:
            if model is not None:
                metrics.update(evaluate(protocol, model, x, u, t, stats, error))
            else:
                metrics.update(task_success=None, physical_pass=None, solver_numerical_pass=None,
                    execution_error=error, unmeasured={'task_success':'setup failed before controller execution',
                    'physical_pass':'no executed plant samples', 'solver_numerical_pass':'SB-001/SB-002; no full certificate'})
        except BaseException:
            error = (error or '') + '\nmetric evaluation failed:\n' + traceback.format_exc()
            metrics.update(task_success=None, physical_pass=None, solver_numerical_pass=None,
                           execution_error=error, unmeasured={'task_success':'evaluator failed; raw data retained',
                           'physical_pass':'evaluator failed; raw data retained',
                           'solver_numerical_pass':'SB-001/SB-002; no full certificate'})
        feedback_solves = [s for s in stats.get('raw_solves', []) if s.get('category') != 'startup/cold']
        # Include timed failed calls, including failures before feedback stats
        # append; missing native values are NaN and make the summary unknown.
        wall = distribution([s.get('wall_solve_time_s', float('nan'))*1000 for s in feedback_solves])
        native = distribution([s.get('native_solve_time_ms', float('nan')) for s in feedback_solves])
        intervals = distribution(stats.get('update_intervals_s', []))
        updates = np.asarray(stats.get('update_intervals_s', []), dtype=float)
        metrics.update(samples=wall['count'], mean_wall_solve_ms=wall['mean'],
            median_wall_solve_ms=wall['median'], p95_wall_solve_ms=wall['p95'],
            mean_native_ms=native['mean'], median_native_ms=native['median'], p95_native_ms=native['p95'],
            startup_wall_solve_time_s=stats.get('startup_wall_solve_time_s'),
            startup_native_solve_time_ms=stats.get('startup_native_solve_time_ms'),
            mean_update_interval_s=intervals['mean'],
            update_fraction_above_nominal=float(np.mean(updates>protocol['nominal_feedback_s']))
            if updates.size and np.isfinite(updates).all() else None,
            timing_distributions={'wall_ms':wall, 'native_ms':native, 'update_s':intervals},
            raw_files=[e['raw_file'] for e in recorder.events]+['events.jsonl', 'execution.npz'],
            family_hold=bool(error or not metrics.get('task_success') or not metrics.get('physical_pass')),
            outcome='execution_failed' if error else ('task_passed' if metrics.get('task_success') else 'task_failed'))
        write_json(output/'metrics.json', metrics)
        recorder.manifest()
    # Negative scientific outcome completes normally; thrown exception is an
    # execution failure for explicit runner retry and preserved failed receipt.
    return 1 if error else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inputs', type=Path)
    args = parser.parse_args()
    raise SystemExit(run_case(json.loads(args.case.read_text()), args.output, args.inputs))
