#!/usr/bin/env python3
"""One frozen D01 Gazebo case, with CPU-only dry-run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback
import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.deployment.protocol import (ROOT, INPUT, PREFLIGHT_ID, resolve_case,
                                           gazebo_environment_guard, preflight_guard)
from experiments.deployment.raw import RawRecorder, sha, write_json
from experiments.deployment.live import record_cleanup_event


def provenance():
    paths = ['python/bsqp/interface.py', 'python/bsqp/common.py',
        'tiago_src/gato_tiago/tiago_mpc_controller.py',
        'tiago_src/gato_tiago/tiago_controller_process.py',
        'tiago_src/gato_tiago/ros_tiago.py', 'tiago_src/gato_tiago/config.py',
        'tiago_configs/comfortable_pose_collision_blacklist.json']
    paths += [str(p.relative_to(ROOT)) for p in Path(__file__).parent.glob('*.py')]
    return dict(source_commit=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        dirty_status=subprocess.check_output(['git', '-C', str(ROOT), 'status', '--short'], text=True),
        source_sha256={p: sha(ROOT / p) for p in paths}, input_sha256=sha(INPUT),
        environment={key: os.environ.get(key) for key in ('GATO_MACHINE', 'GATO_ROS_PROFILE',
            'ROS_DOMAIN_ID', 'RMW_IMPLEMENTATION', 'CYCLONEDDS_URI', 'ROS_LOCALHOST_ONLY')})


def execute_task(mpc, protocol, controller, recorder):
    import numpy as np
    initial = np.asarray(protocol['initial_state'], dtype=np.float32)
    common = dict(sim_dt=protocol['sim_dt_s'], controller=controller,
        controller_timeout=protocol['controller_timeout_s'],
        controller_startup_timeout=protocol['controller_startup_timeout_s'])
    if protocol['kind'] == 'reaching':
        return mpc.run_mpc_goals(initial, [np.asarray(protocol['goal'], dtype=np.float32)],
            goal_timeout=protocol['timeout_s'], goal_threshold=protocol['position_threshold_m'],
            goal_axis_threshold=np.deg2rad(protocol['axis_threshold_deg']),
            velocity_threshold=protocol['speed_L1_threshold_rad_s'], goal_dwell_time=0.,
            reset_rho_each_solve=False, initial_guess=None, **common)
    reference = np.asarray(protocol['reference_packed'], dtype=np.dtype(protocol['reference_dtype']))
    if hashlib.sha256(reference.tobytes()).hexdigest() != protocol['reference_sha256']:
        raise ValueError('Frozen tracking reference bytes changed')
    recorder('frozen_reference', dict(reference=reference, reference_dt_s=protocol['reference_dt_s']))
    return mpc.run_mpc_fig8(initial, reference, sim_time=protocol['duration_s'], **common)


def run_case(case, output):
    protocol = resolve_case(case)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ('protocol.json', 'events.jsonl', 'metrics.json')):
        raise ValueError('Fresh supplied output directory required')
    write_json(output / 'protocol.json', protocol)
    write_json(output / 'case.json', case)
    recorder = RawRecorder(output)
    identity = provenance()
    write_json(output / 'provenance.json', identity)
    metrics = dict(case_id=protocol['case_id'], campaign_id='D01', execution_backend='gazebo',
        machine='rtx5060', timing_mode='live', computation_paid=True,
        input_sha256=protocol['input_sha256'], config_sha256=sha(output / 'protocol.json'),
        preflight_setup_sha256=protocol['preflight_setup_sha256'],
        task_success=None, task_unavailable='task evidence not evaluated',
        physical_pass=None, execution_error=None,
        handback_confirmed=False, known_limitations=protocol['known_limitations'],
        solver_numerical_pass=None, measured_actuator_torque_available=False)
    observer = controller = mpc = model = None
    stats, errors = {}, []
    handlers = {}
    try:
        gazebo_environment_guard()
        preflight_guard(protocol, case)
        def interrupted(signum, frame):
            raise InterruptedError(f'D01 interrupted by signal {signum}')
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, interrupted)
        from experiments.deployment.live import (GazeboObserver, RecordedController,
                                                 RecordedSolver, execute_with_handback)
        observer = GazeboObserver(output)
        observer.require_advancing_clock(protocol['simulation_guard_timeout_s'])
        recorder('gazebo_guard_passed', dict(profile='simulation', domain=1, clock_advancing=True))
        # Import the public live path only after the simulation guard. Importing
        # rclpy above ensures ROS helpers cannot source a missing environment.
        for directory in (ROOT / 'python', ROOT / 'tiago_src'):
            sys.path.insert(0, str(directory))
        import pinocchio as pin
        from gato_tiago.tiago_mpc_controller import MPC_GATO
        from gato_tiago.tiago_controller_process import TiagoControllerOrchestrator
        model_path = ROOT / protocol['urdf']['path']
        if sha(model_path) != protocol['urdf']['sha256']:
            raise ValueError('Frozen model bytes changed')
        model = pin.buildModelFromUrdf(str(model_path))
        mpc = MPC_GATO(model, str(model_path), N=protocol['N'], dt=protocol['dt_s'],
            batch_size=1, constant_f_ext=np.asarray(protocol['constant_f_ext'], dtype=np.float32), track_full_stats=True,
            plant_type='tiago_right', pendulum_config=None,
            solver_params=protocol['solver_params'], final_approach=False)
        module = sys.modules.get(f'bsqp.bsqpN{protocol["N"]}_tiago_right')
        if module is not None and getattr(module, '__file__', None):
            identity['module'] = dict(path=module.__file__, sha256=sha(module.__file__))
        else:
            raise RuntimeError('Loaded native module identity unavailable')
        write_json(output / 'provenance.json', identity)
        preflight_guard(protocol, case, identity['module']['sha256'])
        mpc.solver = RecordedSolver(mpc.solver, recorder)
        controller = RecordedController(TiagoControllerOrchestrator(**protocol['live_controller']), recorder)
        stats, errors = execute_with_handback(controller,
            lambda live: execute_task(mpc, protocol, live, recorder), output, recorder,
            close_timeout_s=protocol['controller_timeout_s'])
    except BaseException as exc:
        errors.append(dict(stage='setup', type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
        record_cleanup_event(recorder, 'execution_failure', errors[-1], errors)
    finally:
        # Covers interruption between construction and entering the task helper.
        if controller is not None and not controller.close_attempted:
            _, cleanup_errors = execute_with_handback(controller, lambda _: (None, {}), output, recorder,
                                                      close_timeout_s=protocol['controller_timeout_s'])
            errors.extend(cleanup_errors)
        if observer is not None:
            try:
                observer.close()
            except BaseException as exc:
                errors.append(dict(stage='observer', type=type(exc).__name__, message=str(exc)))
                record_cleanup_event(recorder, 'observer_failure', errors[-1], errors)
        record_cleanup_event(recorder, 'controller_stats', stats, errors)
        try:
            write_json(output / 'controller_stats.json', stats)
        except BaseException as exc:
            errors.append(dict(stage='controller_stats_write', type=type(exc).__name__, message=str(exc)))
        if controller is not None:
            metrics['handback_confirmed'] = controller.handback_confirmed
        if model is not None and controller is not None and observer is not None:
            try:
                from experiments.deployment.evaluate import evaluate_live, distribution
                metrics.update(evaluate_live(protocol, model, controller, observer, stats, recorder, errors))
                if metrics['task_success'] is not None:
                    metrics.pop('task_unavailable', None)
                metrics['wrapper_wall_s'] = distribution(mpc.solver.wall_times)
                metrics['returned_native_time_us'] = distribution(mpc.solver.native_times_us)
                metrics['solve_recording_overhead_s'] = distribution(mpc.solver.recording_times)
                record_cleanup_event(recorder, 'recording_overhead', dict(solve_recording_overhead_s=mpc.solver.recording_times,
                    scope='input/output/stat copies and solve-record writes; other observer/controller recording excluded',
                    live_elapsed_includes_recording=True), errors)
            except BaseException as exc:
                errors.append(dict(stage='evaluation', type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
                record_cleanup_event(recorder, 'evaluation_failure', errors[-1], errors)
            errors.extend(mpc.solver.recording_errors)
        metrics['execution_error'] = errors or None
        metrics['physical_monitor_summary'] = (controller.physical_monitor_summary()
            if controller is not None else {'unavailable': 'controller not constructed'})
        if protocol['case_id'] == PREFLIGHT_ID:
            metrics['family_held'] = bool(errors or not metrics['task_success']
                or metrics['physical_pass'] is not True or not metrics['handback_confirmed'])
        try:
            try:
                write_json(output / 'metrics.json', metrics)
            except BaseException as exc:
                errors.append(dict(stage='metrics_write', type=type(exc).__name__, message=str(exc)))
            try:
                recorder.manifest()
            except BaseException as exc:
                errors.append(dict(stage='manifest_write', type=type(exc).__name__, message=str(exc)))
            # Preserve finalization errors if metrics remain writable. This is
            # one final write attempt, never recursive logging or retry-to-green.
            if errors:
                metrics.update(execution_error=errors)
                if protocol['case_id'] == PREFLIGHT_ID:
                    metrics['family_held'] = True
                try:
                    write_json(output / 'metrics.json', metrics)
                except BaseException as exc:
                    errors.append(dict(stage='final_metrics_write', type=type(exc).__name__, message=str(exc)))
                print(json.dumps({'D01_execution_error': errors}), file=sys.stderr)
        finally:
            for sig, old in handlers.items():
                signal.signal(sig, old)
    return 1 if errors else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True, help='JSON case, or path to case JSON')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--dry-run', action='store_true', help='Resolve setup without ROS/native imports or motion')
    args = parser.parse_args()
    case = json.loads(args.case) if args.case.lstrip().startswith('{') else json.loads(Path(args.case).read_text())
    if args.dry_run:
        print(json.dumps(resolve_case(case), indent=2))
        return 0
    return run_case(case, args.output)


if __name__ == '__main__':
    raise SystemExit(main())
