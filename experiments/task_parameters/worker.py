#!/usr/bin/env python3
"""Prepared parameter-study worker. No implicit safety override or ROS setup."""
import argparse
import importlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from protocol import resolve_protocol


def make_finite_recorder(parent):
    """Preserve the compact event, then stop invalid execution without limit gates."""
    class FiniteRecorder(parent):
        def __call__(self, kind, record):
            result = super().__call__(kind, record)
            if kind in ('solve', 'solve_started', 'solve_completed', 'step_started',
                        'step_completed', 'feedback_state'):
                for field in ('state', 'state_before', 'state_after', 'control',
                              'output', 'q', 'qd'):
                    value = record.get(field)
                    if value is not None and not np.isfinite(np.asarray(value, dtype=float)).all():
                        raise ValueError(f'nonfinite execution data: {kind}.{field}; event retained')
            return result
    return FiniteRecorder


def require_preflight(output, protocol, deployment):
    location = deployment.preflight_path(output)
    metrics = json.loads((location / 'metrics.json').read_text())
    previous = json.loads((location / 'protocol.json').read_text())
    if not (previous.get('case_id') == 'canonical-preflight'
            and previous.get('canonical_setup_sha256') == protocol['canonical_setup_sha256']
            and metrics.get('execution_error') is None
            and metrics.get('task_success') is True
            and (protocol['safety_policy']['report_only_physical_stops']
                 or metrics.get('physical_pass') is True)
            and metrics.get('handback_confirmed') is True):
        raise RuntimeError('Same-setup Gazebo canonical task/physical/handback preflight required')


def run_native_readiness(args, case):
    """One cold public binding call; no plant/controller/ROS or convergence gate."""
    if args.backend != 'internal' or args.preflight or args.dangerously_disable_safety_stops:
        raise ValueError('Native readiness requires internal backend, no preflight or unsafe flag')
    parameters = case.get('parameters', {})
    if (parameters.get('dt_s', .03) != .03 or parameters.get('sqp_max_iter', 1) != 1
            or parameters.get('pcg_max_iter', 120) != 120):
        raise ValueError('Native readiness is one SQP1/PCG120/30ms call')
    p = resolve_protocol(args.inputs, case, 'internal', preflight=True)
    # Validate the requested N through the ordinary resolver; canonical goal/state
    # still come exclusively from v01. Do not reset a new-horizon check to N32.
    selected = resolve_protocol(args.inputs, dict(id='goal-3', parameters=parameters), 'internal')
    p.update(N=selected['N'], prediction_duration_s=(selected['N'] - 1) * .03,
             solver_params=dict(p['solver_params'], max_sqp_iters=1, max_pcg_iters=120),
             execution_backend='native_readiness', case_id='native-readiness',
             timing_mode='native_binding', offline_timing=None,
             timing_scope='one cold public binding wrapper/native duration; no plant/controller clock',
             motion_executed=False, requested_native_calls=1,
             seed_policy='cold constant-current-state knots and zero controls; no warmup or retry')
    if args.dry_run:
        print(json.dumps(p, indent=2, allow_nan=False))
        return 0
    sys.path.insert(0, str(args.runner_root.resolve() / 'experiments'))
    from recording import sha, write_json
    from adapters.task import native_module
    source, build, output = (v.resolve() for v in (args.source, args.build, args.output))
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ('protocol.json', 'metrics.json', 'native_output.npz')):
        raise FileExistsError('Fresh native-readiness output required')
    write_json(output / 'protocol.json', p)
    metrics = dict(execution_backend='native_readiness', N=p['N'], motion_executed=False,
        optimizer_calls=0, task_success=None, physical_pass=None, execution_error=None,
        finite_readiness_pass=False, convergence_gate=False)
    provenance = dict(collector_source_commit=args.collector_pin,
        collector_sha256=sha(__file__), protocol_sha256=sha(output / 'protocol.json'))
    try:
        native, module = native_module(source, build, p['N'])
        interface = importlib.import_module('bsqp.interface')
        if Path(interface.__file__).resolve() != source / 'python/bsqp/interface.py':
            raise ValueError('Public binding wrapper imported from a different runtime')
        urdf = source / p['urdf']['path']
        if sha(urdf) != p['urdf']['sha256']:
            raise ValueError('Frozen URDF changed')
        provenance.update(runtime_commit=subprocess.check_output(
            ['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
            module_path=str(module), module_sha256=sha(module),
            wrapper_sha256=sha(interface.__file__), urdf_sha256=sha(urdf))
        solver = interface.BSQP(str(urdf), 1, p['N'], .03, plant_type='tiago_right', **p['solver_params'])
        if solver.lib is not native:
            raise ValueError('Effective native module differs')
        state = np.asarray(p['initial_state'], dtype=np.float32).reshape(1, 14)
        reference = np.tile(np.asarray(p['goal'], dtype=np.float32), (1, p['N']))
        seed = np.zeros((1, 21 * p['N'] - 7), dtype=np.float32)
        for knot in range(p['N']):
            seed[0, knot * 21:knot * 21 + 14] = state[0]
        metrics['input_finite'] = bool(all(np.isfinite(v).all() for v in (state, reference, seed)))
        np.savez(output / 'native_input.npz', state=state, reference=reference, seed=seed)
        if not metrics['input_finite']:
            raise ValueError('Nonfinite readiness input')
        metrics['optimizer_calls'] = 1
        tick = time.perf_counter()
        result, native_time = solver.solve(state, reference, seed)
        metrics['wrapper_wall_s'] = time.perf_counter() - tick
        metrics['native_time_us'] = np.asarray(native_time).tolist()
        np.savez(output / 'native_output.npz', XU=result)
        metrics['output_finite'] = bool(np.isfinite(result).all())
        metrics['output_shape'] = list(np.asarray(result).shape)
        metrics['kkt_converged_descriptive'] = np.asarray(solver.stats.get('kkt_converged', [])).tolist()
        if not metrics['output_finite'] or np.asarray(result).shape != seed.shape:
            raise ValueError('Nonfinite or malformed final readiness trajectory')
        metrics['finite_readiness_pass'] = True
    except BaseException:
        metrics['execution_error'] = traceback.format_exc()
    write_json(output / 'provenance.json', provenance)
    write_json(output / 'metrics.json', metrics)
    write_json(output / 'SHA256SUMS.json', {v.name: sha(v) for v in output.iterdir()
        if v.is_file() and v.name != 'SHA256SUMS.json'})
    return 0 if metrics['finite_readiness_pass'] else 1


def run(args):
    if not re.fullmatch('[0-9a-f]{40}', args.collector_pin):
        raise ValueError('--collector-pin must be the full pinned collector commit')
    case = json.loads(args.case.read_text()) if args.case else {}
    if args.native_readiness:
        return run_native_readiness(args, case)
    protocol = resolve_protocol(args.inputs, case, args.backend, args.preflight,
                                args.dangerously_disable_safety_stops)
    if args.dangerously_disable_safety_stops:
        print('DANGER: Gazebo physical safety STOPS bypassed; monitoring/reporting retained. '
              'No clamping. Disabled stops: ' + ', '.join(protocol['safety_policy']['disabled_stops']), flush=True)
    if args.dry_run:
        print(json.dumps(protocol, indent=2, allow_nan=False))
        return 0
    source, build, output = (p.resolve() for p in (args.source, args.build, args.output))
    sys.path.insert(0, str(args.runner_root.resolve() / 'experiments'))
    from recording import sha, write_json
    from adapters import task, deployment
    if args.backend == 'internal':
        task.select_protocol = lambda *unused, **ignored: protocol
        task.TaskRecorder = make_finite_recorder(task.TaskRecorder)
        status = task.run(source, build, case, output, 'B', args.preflight, args.inputs)
    else:
        sys.path.insert(0, str(source))
        # Reject a safe-only/hardware runtime before constructing or commanding ROS.
        sys.path[:0] = [str(source / 'python'), str(source / 'tiago_src')]
        controller_file = source / 'tiago_src/gato_tiago/tiago_controller_process.py'
        if args.dangerously_disable_safety_stops and 'REPORT_ONLY_PHYSICAL_STOPS' not in controller_file.read_text():
            raise ValueError('Runtime lacks reviewed Gazebo report-only support')
        worker = importlib.import_module('experiments.deployment.run_d01')
        if Path(worker.__file__).resolve() != source / 'experiments/deployment/run_d01.py':
            raise ValueError('Live worker imported from a different runtime')
        worker.gazebo_environment_guard()
        if not args.preflight:
            require_preflight(output, protocol, deployment)
        worker.resolve_case = lambda unused: protocol
        original_provenance = worker.provenance
        def study_provenance():
            identity = original_provenance()
            identity['input_sha256'] = protocol['input_sha256']
            identity['collector_source_commit'] = args.collector_pin
            return identity
        worker.provenance = study_provenance
        worker.preflight_guard = lambda *unused, **ignored: None  # Checked above; N varies by design.
        worker.PREFLIGHT_ID = 'canonical-preflight'
        protocol['live_controller']['collision_blacklist_path'] = str(source / 'tiago_configs/comfortable_pose_collision_blacklist.json')
        worker.RawRecorder = make_finite_recorder(deployment.LiveRecorder)
        original_write = worker.write_json
        def compact_write(path, value):
            if Path(path).name == 'controller_stats.json':
                if not Path(path).is_file():
                    raise ValueError('Missing compact controller summary')
                return
            return original_write(path, value)
        worker.write_json = compact_write
        _, module = task.native_module(source, build, protocol['N'])
        status = worker.run_case(case, output)
    metrics_path = output / 'metrics.json'
    metrics = json.loads(metrics_path.read_text())
    metrics.update(safety_policy=protocol['safety_policy'], N=protocol['N'], dt_s=protocol['dt_s'],
        prediction_duration_s=protocol['prediction_duration_s'],
        reference_dt_s=protocol.get('reference_dt_s'),
        canonical_setup_sha256=protocol['canonical_setup_sha256'],
        collector_source_commit=args.collector_pin,
        collector_protocol_sha256=sha(Path(__file__).with_name('protocol.py')),
        collector_sha256=sha(__file__))
    if args.backend == 'gazebo':
        summary = metrics.get('physical_monitor_summary', {})
        torques = summary.get('proposed_torque', {})
        monitor = summary.get('physical_monitor', {})
        checked = torques.get('accepted_trajectories', 0) > 0 and monitor.get('checked_states', 0) > 0
        metrics['reported_safety_criteria_pass'] = bool(
            metrics.get('physical_pass') is True and torques.get('violating_trajectories', 0) == 0
            and monitor.get('violating_states', 0) == 0) if checked else None
        # Production physical_pass reports URDF effort limits; preserve it separately.
        metrics['observed_urdf_q_v_effort_pass'] = metrics.get('physical_pass')
        metrics['physical_pass'] = metrics['reported_safety_criteria_pass']
        # Existing observer stream holds actual published effort messages.
        peak, values, rows, nonfinite = None, 0, 0, 0
        observer_path = output / 'ros_observations.jsonl'
        with observer_path.open() if observer_path.is_file() else open('/dev/null') as stream:
            for line in stream:
                event = json.loads(line)
                if event.get('topic') != '/gato_arm_effort_forward_runtime/commands':
                    continue
                tau = np.asarray(event['data'], dtype=float)
                if not np.isfinite(tau).all():
                    nonfinite += 1
                    continue
                absolute = np.abs(tau)
                peak = float(absolute.max()) if peak is None else max(peak, float(absolute.max()))
                values += int((absolute > 26.).sum())
                rows += int((absolute > 26.).any())
        metrics['published_effort_26Nm'] = dict(maximum_abs_Nm=peak,
            violating_messages=rows, violating_values=values, nonfinite_messages=nonfinite,
            available=peak is not None,
            measured_actuator_torque_available=False)
        if args.preflight:
            held = bool(metrics.get('execution_error') is not None
                or metrics.get('task_success') is not True
                or metrics.get('handback_confirmed') is not True
                or (not args.dangerously_disable_safety_stops and metrics['physical_pass'] is not True))
            metrics['family_held'] = held
            metrics['preflight_physical_criteria_enforced'] = not args.dangerously_disable_safety_stops
            status = 1 if held else 0
    write_json(metrics_path, metrics)
    manifest = json.loads((output / 'SHA256SUMS.json').read_text())
    manifest['metrics.json'] = sha(metrics_path)
    write_json(output / 'SHA256SUMS.json', manifest)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'build', 'runner-root', 'inputs', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--case', type=Path)
    parser.add_argument('--collector-pin', required=True,
        help='Full collector commit, supplied from the runner-verified source pin')
    parser.add_argument('--backend', choices=('internal', 'gazebo'), required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--native-readiness', action='store_true',
        help='One cold B1 SQP1/PCG120 binding solve; frozen canonical state/goal, no motion/ROS')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--dangerously-disable-safety-stops', action='store_true',
        help='Gazebo simulation only: report physical/torque violations without stopping; retain finite/stale/execution checks and handback')
    args = parser.parse_args()
    if not args.preflight and args.case is None:
        parser.error('--case is required unless --preflight')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
