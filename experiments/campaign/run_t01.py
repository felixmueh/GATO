"""Six fixed independent T01 episodes and separately authorized1ms preflight.

No native/controller import before release and hashed prerequisite validation.
"""
import argparse
import csv
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
from t01_protocol import ROOT, INPUTS, PROTOCOL_FILES, load_case, reference_bytes, episode_protocol, preflight, fallback


def trace_metrics(protocol, model, x, u, t, stats, error, output):
    """Every actual returned plant state; no missing controls manufactured."""
    import pinocchio as pin
    from bsqp.common import sample_reference
    tracking = protocol['kind'] == 'tracking'
    check_protocol = dict(protocol)
    if tracking:
        check_protocol['goal'] = [0.]*6
        check_protocol['timeout_s'] = protocol['duration_s']
    checked = evaluate(check_protocol, model, x, u, t, stats, error)
    if not tracking:
        result = checked
    else:
        result = {k:checked[k] for k in ('physical_pass','physical_limits','validation','solver_numerical_pass','execution_error','unmeasured') if k in checked}
    frame, data = model.getFrameId('arm_right_tool_link'), model.createData()
    reference = reference_bytes(protocol) if tracking else None
    targets = sample_reference(reference, t, protocol['reference_dt_s']) if tracking else np.tile(protocol['goal'], (len(t),1))
    rows, distances = [], []
    for index, (state, stamp, target) in enumerate(zip(x, t, targets)):
        if np.isfinite(state).all():
            pin.framesForwardKinematics(model, data, state[:7].astype(np.float64))
            placement = data.oMf[frame]
            xyz = placement.translation.copy()
            target_rotation = pin.rpy.rpyToMatrix(*target[3:])
            axis = np.rad2deg(np.arccos(np.clip(placement.rotation[:,2]@target_rotation[:,2], -1, 1)))
            full = np.rad2deg(np.linalg.norm(pin.log3(target_rotation.T@placement.rotation)))
        else:
            xyz, axis, full = np.full(3,np.nan), np.nan, np.nan
        distance = np.linalg.norm(xyz-target[:3]); distances.append(distance)
        control = u[index-1] if index and len(u) >= index else np.full(7,np.nan)
        q_violation = np.maximum(np.maximum(model.lowerPositionLimit-state[:7], state[:7]-model.upperPositionLimit),0)
        v_violation = np.maximum(abs(state[7:])-model.velocityLimit,0)
        u_violation = np.maximum(abs(control)-model.effortLimit,0)
        rows.append(np.r_[stamp, xyz, target[:3], distance, axis, full, np.linalg.norm(state[7:],ord=1), state, control, q_violation,v_violation,u_violation])
    header = ['time_s','fk_x_m','fk_y_m','fk_z_m','target_x_m','target_y_m','target_z_m','position_error_m','tool_axis_error_deg','full_SO3_error_deg','speed_L1_rad_s']
    header += [f'{prefix}{j}' for prefix in ('q_rad_','qd_rad_s_','applied_control_Nm_','q_violation_rad_','v_violation_rad_s_','u_violation_Nm_') for j in range(7)]
    with (Path(output)/'executed_trace.csv').open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(header); writer.writerows(rows)
    if tracking:
        e = np.asarray(distances)
        complete = bool(len(t)>1 and abs(t[-1]-protocol['duration_s'])<=1e-8)
        finite = bool(np.isfinite(e).all() and np.isfinite(t).all())
        # Time weighting prevents fast/slow feedback from changing sample weights.
        rms = float(np.sqrt(np.trapezoid(e*e,t)/(t[-1]-t[0]))) if finite and len(t)>1 and t[-1]>t[0] else None
        maximum = float(np.max(e)) if finite else None
        gates = {'full_period_completed':complete, 'finite_trace':finite,
                 'RMS':rms is not None and rms<=protocol['provisional_RMS_gate_m'],
                 'maximum':maximum is not None and maximum<=protocol['provisional_maximum_gate_m'],
                 'completed_without_exception':error is None,
                 'recording_clock':checked['validation'].get('recording_clock',False),
                 'initial_state_matches':checked['validation'].get('initial_state_matches',False)}
        result.update(task_success=bool(all(gates.values())),task_gates=gates,
            tracking_RMS_m=rms, tracking_maximum_m=maximum, RMS_definition='time-weighted trapezoidal position-error squared over all executed states; no startup exclusion',
            executed_duration_s=float(t[-1]), plant_steps=len(u),tracking_orientation_gate=None,
            tracking_feasibility_scope=protocol['tracking_feasibility_scope'])
    return result


def execute_episode(controller, protocol, recorder):
    initial = np.asarray(protocol['initial_state'], dtype=np.float32)
    if protocol['kind'] == 'reaching':
        return controller.run_mpc_goals(initial, [np.asarray(protocol['goal'],dtype=np.float32)],
            sim_dt=protocol['sim_dt_s'], goal_timeout=protocol['timeout_s'],
            goal_threshold=protocol['position_threshold_m'],goal_axis_threshold=np.deg2rad(protocol['axis_threshold_deg']),
            velocity_threshold=protocol['speed_L1_threshold_rad_s'],goal_dwell_time=0.,
            reset_rho_each_solve=False, initial_guess=None,offline_timing='solve_time',
            record_raw=True,record_callback=recorder)
    reference = reference_bytes(protocol)
    recorder('frozen_reference', {'reference':reference,'reference_dt_s':protocol['reference_dt_s'],
                                'reference_sha256':protocol['reference_sha256']})
    return controller.run_mpc_fig8(initial,reference,sim_dt=protocol['sim_dt_s'],
        sim_time=protocol['duration_s'],offline_timing='solve_time',record_raw=True,record_callback=recorder)


def current_identity():
    module = ROOT/'python/bsqp'
    modules = list(module.glob('bsqpN32_tiago_right*.so'))
    if len(modules)!=1:
        raise ValueError('Require one exact isolated N32 module before preflight validation')
    source = Path(os.environ.get('GATO_SOURCE_DIR',ROOT))
    commit = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    shared=['python/bsqp/interface.py','python/bsqp/common.py','python/bsqp/offline_mpc.py','tiago_src/gato_tiago/tiago_mpc_controller.py','tiago_src/gato_tiago/config.py','experiments/campaign/raw.py','experiments/campaign/evaluate.py']
    return {'machine':os.environ.get('GATO_MACHINE'),'source_commit':commit,'module_sha256':sha(modules[0]),'module_path':str(modules[0]),'source_sha256':{p:sha(ROOT/p) for p in shared}}


def release_guard(cohort):
    if cohort.get('release_machines') != ['gtx1070','rtx5060'] or os.environ.get('GATO_MACHINE') not in cohort['release_machines']:
        raise ValueError('Only frozen desktop internal machines are released')
    required={'python/bsqp/offline_mpc.py','python/bsqp/mpc_controller.py',
              'tiago_src/gato_tiago/tiago_mpc_controller.py'}
    reviewed=cohort.get('reviewed_controller_sha256',{})
    if set(reviewed)!=required or any(sha(ROOT/path)!=digest for path,digest in reviewed.items()):
        raise ValueError('Reviewed controller source bytes mismatch')
    scripts={'experiments/campaign/'+name for name in ('run_t01.py','t01_protocol.py','raw.py','evaluate.py','make_task_plan.py','build_isolated.py')}
    reviewed=cohort.get('reviewed_harness_sha256',{})
    if set(reviewed)!=scripts or any(sha(ROOT/path)!=digest for path,digest in reviewed.items()):
        raise ValueError('Reviewed task harness source bytes mismatch')


def build_identity(identity):
    """Validate all existing artifacts before any controller/native import."""
    receipt_path=Path(os.environ.get('GATO_CAMPAIGN_BUILD_RECEIPT',ROOT.parent/'build_receipt.json'))
    if not receipt_path.is_file(): raise ValueError('Exact isolated build receipt required')
    build=json.loads(receipt_path.read_text())
    if (build.get('module_sha256')!=identity['module_sha256']
        or Path(build.get('module_path','')).resolve()!=Path(identity['module_path']).resolve()
        or Path(build.get('source','')).resolve()!=ROOT.resolve()
        or sha(identity['module_path'])!=identity['module_sha256']):
        raise ValueError('Build receipt module/source mismatch')
    source_path=ROOT.parent/'build_source_receipt.json'
    if not source_path.is_file() or sha(source_path)!=build.get('build_source_receipt_sha256'):
        raise ValueError('Pinned build source receipt missing or hash mismatch')
    source=json.loads(source_path.read_text())
    if source.get('source_commit')!=identity['source_commit'] or Path(source.get('isolated_source','')).resolve()!=ROOT.resolve():
        raise ValueError('Build source receipt commit/path mismatch')
    for path,digest in identity['source_sha256'].items():
        if source.get('source_file_sha256',{}).get(path)!=digest or sha(ROOT/path)!=digest:
            raise ValueError('Pinned shared source bytes mismatch')
    smoke=ROOT.parent/'campaign-native/cuda_access_smoke'
    if not smoke.is_file() or sha(smoke)!=build.get('cuda_access_smoke_sha256'):
        raise ValueError('Exact isolated readiness executable required')
    return receipt_path,source_path,smoke


def source_module(name, relative):
    module=importlib.import_module(name)
    if Path(module.__file__).resolve()!=(ROOT/relative).resolve():
        raise ValueError('Imported shared source path mismatch: '+name)
    return module


def run_case(case, output, inputs=None):
    cohort = load_case(case)
    if cohort.get('release_status') != 'released after reviewed clock decision':
        raise ValueError('Execution held pending trueB review, required suite and preflight')
    release_guard(cohort)
    identity = current_identity()
    parameters = case['parameters']
    if cohort['campaign_id'] != 'T01':
        canonical = json.loads((INPUTS/'t01.json').read_text())
        fallback(parameters['fallback'],inputs,identity,canonical)
    if cohort['campaign_id'] == 'V01-1ms':
        protocol = cohort
    else:
        expected_name = 'v01.json' if cohort['campaign_id']=='T01' else 'v01_1ms.json'
        expected = json.loads((INPUTS/expected_name).read_text())
        if sha(INPUTS/expected_name) != cohort['preflight_protocol_sha256']:
            raise ValueError('Preflight protocol differs from frozen family input')
        preflight(parameters['preflight'],inputs,identity,expected)
        matches = [task for task in cohort['tasks'] if task['id']==case['id']]
        if len(matches)!=1:
            raise ValueError('Case must select exactly one frozen task')
        protocol = episode_protocol(cohort,matches[0])
    output = Path(output);output.mkdir(parents=True,exist_ok=True)
    if any((output/name).exists() for name in ('events.jsonl','metrics.json','protocol.json')):
        raise ValueError('Fresh attempt directory required')
    write_json(output/'protocol.json',protocol)
    recorder=RawRecorder(output);controller=None;stats={};model=None;error=None
    metrics=dict(research_question='RQ-control',campaign_id=cohort['campaign_id'],execution_backend='internal',
        timing_mode='B',offline_timing='solve_time',configuration=protocol,machine=identity['machine'],
        nominal_feedback_s=.03,known_limitations=protocol['known_limitations'])
    handlers={}
    try:
        def interrupted(signum, frame):
            raise InterruptedError(f'T01 interrupted by signal {signum}')
        for sig in (signal.SIGINT,signal.SIGTERM): handlers[sig]=signal.signal(sig,interrupted)
        build_receipt,build_source,smoke=build_identity(identity)
        shutil.copyfile(build_source,output/'build_source_receipt.json')
        shutil.copyfile(build_receipt,output/'build_receipt.json')
        urdf=ROOT/protocol['urdf']['path']
        if sha(urdf)!=protocol['urdf']['sha256']: raise ValueError('Frozen URDF hash mismatch')
        subprocess.run([str(smoke)],check=True)
        write_json(output/'readiness.json',{'command':[str(smoke)],'sha256':sha(smoke),'exit_code':0})
        sys.path[:0]=[str(ROOT/'python'),str(ROOT/'tiago_src')]
        import pinocchio as pin
        for name,relative in [('bsqp.common','python/bsqp/common.py'),('bsqp.interface','python/bsqp/interface.py'),
                              ('bsqp.offline_mpc','python/bsqp/offline_mpc.py'),('bsqp.mpc_controller','python/bsqp/mpc_controller.py'),
                              ('gato_tiago.config','tiago_src/gato_tiago/config.py')]:
            source_module(name,relative)
        MPC_GATO=source_module('gato_tiago.tiago_mpc_controller','tiago_src/gato_tiago/tiago_mpc_controller.py').MPC_GATO
        model=pin.buildModelFromUrdf(str(urdf));model.gravity.linear=np.asarray(protocol['gravity_m_s2'])
        native=importlib.import_module('bsqp.bsqpN32_tiago_right')
        if str(Path(native.__file__).resolve())!=str(Path(identity['module_path']).resolve()): raise ValueError('Native identity mismatch')
        controller=MPC_GATO(model,str(urdf),N=32,dt=.03,batch_size=1,constant_f_ext=np.asarray(protocol['constant_f_ext'],dtype=np.float32),track_full_stats=True,plant_type='tiago_right',pendulum_config=None,solver_params=protocol['solver_params'],final_approach=False)
        if controller.solver_params!=protocol['solver_params'] or controller.solver.lib is not native:
            raise ValueError('Effective public preset/native module mismatch')
        sources=['python/bsqp/offline_mpc.py','python/bsqp/common.py','python/bsqp/interface.py','tiago_src/gato_tiago/tiago_mpc_controller.py','experiments/campaign/run_t01.py','experiments/campaign/t01_protocol.py','experiments/campaign/raw.py','experiments/campaign/evaluate.py']
        write_json(output/'provenance.json',dict(identity,build_receipt_sha256=sha(build_receipt),python=sys.version,numpy=np.__version__,pinocchio=pin.__version__,protocol_sha256=sha(INPUTS/PROTOCOL_FILES[cohort['campaign_id']]),task_config_sha256=sha(output/'protocol.json'),source_sha256={s:sha(ROOT/s) for s in set(sources)|set(identity['source_sha256'])},case=case,preflight=parameters.get('preflight'),fallback=parameters.get('fallback')))
        _,stats=execute_episode(controller,protocol,recorder)
    except BaseException:
        error=traceback.format_exc()
        if controller is not None: stats=getattr(controller,'last_run_stats',{})
        recorder('execution_failed',{'error':error})
    finally:
        for sig,handler in handlers.items(): signal.signal(sig,handler)
        recorder('final_raw_solves',{'solves':stats.get('raw_solves',[]),'execution_statuses':[s.get('status') for s in stats.get('raw_execution',[])]})
        complete=[s for s in stats.get('raw_execution',[]) if 'state_after' in s]
        x=np.asarray([protocol['initial_state']]+[s['state_after'] for s in complete]);u=np.asarray([s['control'] for s in complete]).reshape(-1,7)
        t=np.asarray([0.]+[s['time_s']+s['step_dt_s'] for s in complete])
        np.savez_compressed(output/'execution.npz',x=x,u=u,t=t)
        write_json(output/'controller_stats.json',{k:v for k,v in stats.items() if k not in {'raw_solves','raw_execution'}})
        try:
            if model is None: raise RuntimeError('Setup failed before model construction')
            metrics.update(trace_metrics(protocol,model,x,u,t,stats,error,output))
        except BaseException:
            error=(error or '')+'\nmetric evaluation failed:\n'+traceback.format_exc()
            metrics.update(task_success=None,physical_pass=None,solver_numerical_pass=None,execution_error=error,unmeasured={'task_success':'setup/evaluator failed; raw retained','physical_pass':'setup/evaluator failed; raw retained','solver_numerical_pass':'SB-001/SB-002; no certificate'})
        solves=[s for s in stats.get('raw_solves',[]) if s.get('category')!='startup/cold']
        wall=distribution([s.get('wall_solve_time_s',np.nan)*1000 for s in solves]);native=distribution([s.get('native_solve_time_ms',np.nan) for s in solves]);interval=distribution(stats.get('update_intervals_s',[]));updates=np.asarray(stats.get('update_intervals_s',[]))
        metrics.update(samples=wall['count'],mean_wall_solve_ms=wall['mean'],median_wall_solve_ms=wall['median'],p95_wall_solve_ms=wall['p95'],mean_native_ms=native['mean'],median_native_ms=native['median'],p95_native_ms=native['p95'],timing_distributions={'wall_ms':wall,'native_ms':native,'update_s':interval},update_fraction_above_nominal=float(np.mean(updates>.03)) if updates.size and np.isfinite(updates).all() else None,startup_wall_solve_time_s=stats.get('startup_wall_solve_time_s'),startup_native_solve_time_ms=stats.get('startup_native_solve_time_ms'),timing_scope=protocol['timing_scope'],startup_convention=protocol['startup_convention'],interval_alignment=protocol['implemented_clock_convention'],family_hold=bool(error or not metrics.get('task_success') or not metrics.get('physical_pass')),outcome='execution_failed' if error else ('task_passed' if metrics.get('task_success') else 'task_failed'))
        write_json(output/'metrics.json',metrics);recorder.manifest()
        # Extra task CSV and prerequisite artifacts are immutable worker outputs.
        manifest=json.loads((output/'SHA256SUMS.json').read_text())
        for extra in ('executed_trace.csv','build_source_receipt.json'):
            if (output/extra).is_file(): manifest[extra]=sha(output/extra)
        write_json(output/'SHA256SUMS.json',manifest)
    return 1 if error else 0


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--inputs',type=Path)
    args=parser.parse_args();raise SystemExit(run_case(json.loads(args.case.read_text()),args.output,args.inputs))
