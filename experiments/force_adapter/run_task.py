"""Body-fixed disturbance task through the unchanged accepted mode B engine."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import pinocchio as pin

from .task_backend import ForceTask
from .backend import attached_force,force_world_direction
from .run_native import ROOT,serializable,source_commit
from bsqp.offline_mpc import run_mode_b


def run_case(case,output,inputs,module,recorder_class,sha,write_json,*,preflight=False,optimizer=None):
    p=case['parameters'];output=Path(output)
    mode=p.get('timing_mode','B')
    metrics=dict(campaign_id='E02',timing_mode=mode,offline_timing='solve_time' if mode=='B' else 'controller_dt',execution_backend='internal',
        parameters=p,evidence_kind='synthetic_test' if optimizer is not None else 'measurement',
        task_success=None,physical_pass=True,solver_numerical_pass=None,execution_error=None,outcome='execution_failed')
    force=None
    with recorder_class(output) as recorder:
        try:
            if p['mode'] not in ('no_estimator','original','adapted') or 'force_N' not in p:raise ValueError('Explicit method and force_N required')
            if mode!='B' or p.get('backend')!='internal':raise ValueError('Force tasks and preflight require B/internal')
            for name,digest in json.loads((Path(inputs)/'SHA256SUMS.json').read_text()).items():
                if name not in ('v01.json','t01.json') or sha(Path(inputs)/name)!=digest:raise ValueError('Frozen task input bytes differ')
            v01=json.loads((Path(inputs)/'v01.json').read_text());t01=json.loads((Path(inputs)/'t01.json').read_text())
            if preflight:
                task=v01;params=v01['solver_params'];state=np.asarray(v01['initial_state'],dtype=np.float32)
            else:
                task=next(t for t in t01['tasks'] if t['id']=='figure8-24s')
                params=t01['tracking_solver_params'];state=np.asarray(t01['initial_state'],dtype=np.float32)
            np.random.seed(20261011+int(p.get('repeat',0)))
            model=pin.buildModelFromUrdf(str(ROOT/'gato/dynamics/tiago_right/tiago_right_arm.urdf'))
            truth=attached_force(model,state,float(p['force_N']))
            if optimizer is None:
                import bsqp
                name='bsqp.bsqpN32_tiago_right';spec=importlib.util.spec_from_file_location(name,module)
                native=importlib.util.module_from_spec(spec);sys.modules[name]=native;spec.loader.exec_module(native)
            write_json(output/'protocol.json',dict(protocol='force-persistent-grid-task-v2',case=p,task=task,params=params,
                timing_mode=mode,backend='internal',truth=truth,force_frame='arm_right_7_joint',force_point='arm_right_7_joint origin',
                timing_scope='full persistent force facade prediction/selection/update/TO wrapper wall; construction and memory queries outside; native TO separate',
                prediction='previous measured state; time-weighted actually applied controls; one existing GRiD TYPE2 step over actual elapsed observation interval',
                rho_policy='reset before each tracking tick; adapt within solve',dual_policy='initialize once at episode start',
                estimator=dict(initial_radius=10.,min_radius=1.,max_radius=100.,smoothing_factor=.5,alpha=.6,beta=.5),
                bootstrap='Original persistent B16 prior lane0, adapted/no-estimator persistent B1; no true-force access or estimator update; bootstrap computation uncharged, then nominal30ms interval',
                seed_policy='Accepted modeB selected-plan warm shift; broadcast same shifted plan to16 force hypotheses',rng_seed=20261011+int(p.get('repeat',0))))
            write_json(output/'provenance.json',dict(source_root=str(ROOT),source_commit=source_commit(),machine=p.get('machine'),module_path=str(module),module_sha256=sha(module),
                input_sha256={name:sha(Path(inputs)/name) for name in ('v01.json','t01.json')},evidence_kind=metrics['evidence_kind']))
            physical=dict(q_rad=0.,v_rad_s=0.,u_Nm=0.)
            effort_ratio=0.;maximum_applied_torque=0.;maximum_planned_torque=0.
            def validate(state,control=None):
                nonlocal effort_ratio,maximum_applied_torque
                q,v=state[:7],state[7:]
                if not np.isfinite(state).all() or (control is not None and not np.isfinite(control).all()):
                    raise ValueError('Nonfinite plant/control retained')
                physical['q_rad']=max(physical['q_rad'],float(np.max(np.maximum(np.maximum(model.lowerPositionLimit-q,q-model.upperPositionLimit),0))))
                physical['v_rad_s']=max(physical['v_rad_s'],float(np.max(np.maximum(abs(v)-model.velocityLimit,0))))
                if control is not None:
                    physical['u_Nm']=max(physical['u_Nm'],float(np.max(np.maximum(abs(control)-model.effortLimit,0))))
                    effort_ratio=max(effort_ratio,float(np.max(abs(control)/model.effortLimit)))
                    maximum_applied_torque=max(maximum_applied_torque,float(np.max(abs(control))))
            validate(state)
            estimation_errors=[]
            def emit(event,record):
                nonlocal maximum_planned_torque
                if event=='step_completed':
                    ratio=float(np.max(abs(np.asarray(record['control']))/model.effortLimit))
                    record=dict(record,sampled_effort_ratio=ratio,
                        maximum_applied_abs_torque_Nm=float(np.max(abs(record['control']))))
                if event in ('force_tick','force_bootstrap','force_baseline') and record.get('selected_plan') is not None:
                    controls=np.asarray(record['selected_plan'])[:31*21].reshape(31,21)[:,14:]
                    planned=float(np.max(abs(controls)));maximum_planned_torque=max(maximum_planned_torque,planned)
                    record=dict(record,selected_plan_max_abs_torque_Nm=planned)
                if event=='force_tick' and record.get('selected_force') is not None:
                    estimation_errors.append(float(np.linalg.norm(np.asarray(record['selected_force'])-truth)))
                    record=dict(record,force_estimation_error_SI=estimation_errors[-1])
                    record['estimated_force_error_N']=float(np.linalg.norm(np.asarray(record['selected_force'])[:3]-truth[:3]))
                    record['estimated_torque_error_Nm']=float(np.linalg.norm(np.asarray(record['selected_force'])[3:]-truth[3:]))
                    record['truth_direction']=force_world_direction(model,np.asarray(record['inputs'].observed_state)[:7],truth)
                elif event in ('force_bootstrap','force_baseline') and record.get('force') is not None:
                    estimation_errors.append(float(np.linalg.norm(np.asarray(record['force'])-truth)))
                    record=dict(record,force_estimation_error_SI=estimation_errors[-1])
                    record['estimated_force_error_N']=float(np.linalg.norm(np.asarray(record['force'])[:3]-truth[:3]))
                    record['estimated_torque_error_Nm']=float(np.linalg.norm(np.asarray(record['force'])[3:]-truth[3:]))
                recorder(event,serializable(record))
                if event=='solve_started':recorder.flush() # Inputs survive an abrupt native exit, outside solve timing.
                if event=='step_completed':
                    validate(record['state_after'],record['control'])
                    recorder('attached_force_direction',force_world_direction(model,record['state_after'][:7],truth))
            force=ForceTask(model,ROOT/'gato/dynamics/tiago_right/tiago_right_arm.urdf',params,p['mode'],truth,emit,optimizer)
            if preflight:
                common=dict(record_raw=True,record_callback=force.record,goals=np.array([task['goal']],dtype=np.float32),
                    goal_timeout=8.,goal_threshold=.05,goal_axis_threshold=np.deg2rad(30),velocity_threshold=1.,goal_dwell_time=0.,sim_dt=.003)
                _,stats=run_mode_b(force,state,integrate=force.integrate,clock=time.perf_counter,check_axis=True,**common)
                success=stats['goal_outcomes']==['reached']
                distance=float(np.linalg.norm(force.ee_pos(stats['terminal_state'][:7])-np.asarray(task['goal'])[:3]))
                metrics['final_position_error_m']=distance
                data=model.createData();pin.framesForwardKinematics(model,data,stats['terminal_state'][:7])
                rotation=data.oMf[model.getFrameId('arm_right_tool_link')].rotation
                metrics['final_full_orientation_error_rad']=float(np.linalg.norm(pin.log3(rotation.T@pin.rpy.rpyToMatrix(*task['goal'][3:]))))
            else:
                _,stats=run_mode_b(force,state,integrate=force.integrate,clock=time.perf_counter,sim_dt=.003,
                    record_raw=True,record_callback=force.record,reference=np.asarray(task['reference_packed'],dtype=np.float32),sim_time=task['duration_s'])
                errors=np.asarray(stats['goal_distances'],dtype=float)
                rms=float(np.sqrt(np.mean(errors**2))) if len(errors) else None
                maximum=float(np.max(errors)) if len(errors) else None
                success=bool(rms is not None and rms<=task['provisional_RMS_gate_m'] and maximum<=task['provisional_maximum_gate_m'] and
                             stats['terminal_time_s']>=task['duration_s']-1e-6)
                metrics.update(tracking_RMS_error_m=rms,tracking_max_error_m=maximum)
            metrics.update(outcome='force_task_completed',task_success=bool(success),physical_pass=bool(physical['q_rad']<=1e-4 and
                physical['v_rad_s']<=1e-3 and physical['u_Nm']<=1e-3),physical_limit_excess=physical,
                update_intervals_s=np.asarray(stats['update_intervals_s']),
                above_nominal_share=float(np.mean(np.asarray(stats['requested_update_intervals_s'])>.03)),
                timing_scope='full persistent force-algorithm wall advances old plan; construction/memory queries excluded',
                maximum_applied_torque_Nm=maximum_applied_torque,maximum_selected_plan_torque_Nm=maximum_planned_torque,
                maximum_sampled_effort_ratio=effort_ratio,construction=force.construction)
            metrics['force_estimation_errors_SI']=estimation_errors
            metrics['force_estimation_error_note']='Euclidean norm of six declared SI wrench components; report force/torque components from raw separately, not a unitless accuracy score.'
            write_json(output/'controller_stats.json',stats)
        except BaseException:
            metrics['physical_pass']=None
            metrics['execution_error']=traceback.format_exc();recorder('execution_failed',dict(error=metrics['execution_error']))
        finally:
            if force is not None:force.flush()
            write_json(output/'metrics.json',metrics);recorder.manifest()
    return metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('output','inputs','build','runner'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--case',type=Path);parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--mode',choices=('no_estimator','original','adapted'));parser.add_argument('--force-N',type=float)
    parser.add_argument('--repeat',type=int,default=0)
    parser.add_argument('--timing-mode',choices=('B',),default='B')
    args=parser.parse_args();sys.path.insert(0,str(args.runner))
    from recording import RawRecorder,sha,write_json
    modules=list((args.build/'modules').glob('bsqpN32_tiago_right*.so'))
    if len(modules)!=1:raise RuntimeError('One exact force N32 module required')
    case=json.loads(args.case.read_text()) if args.case else dict(parameters=dict(mode=args.mode,force_N=args.force_N,repeat=args.repeat,timing_mode=args.timing_mode,backend='internal'))
    metrics=run_case(case,args.output,args.inputs,modules[0],RawRecorder,sha,write_json,preflight=args.preflight)
    return 1 if metrics['execution_error'] or (args.preflight and not (metrics['task_success'] and metrics['physical_pass'])) else 0


if __name__=='__main__':raise SystemExit(main())
