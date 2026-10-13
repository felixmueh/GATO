#!/usr/bin/env python3
"""Run the same visual-demo inputs internally, in Gazebo, or manually on robot."""
import argparse
import hashlib
from datetime import datetime,timezone
import importlib
import importlib.util
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[3]
INPUT=Path(__file__).with_name('proposal_torque24_limits_only.json')
RUNTIME_PINS={'internal':'fa6f6bede286c122cc41ac65f42a583da84290e0',
              'gazebo':'ab00875006741688a98e76fc3a56ad3103dab7ff',
              'physical_robot':'8dbb8b8d6dc1d65e902373a9aa58693e33329c14'}
TASKS=('square',)  # Accepted four-corner cohort; upward tracking keeps collector2574409
sys.path.insert(0,str(ROOT))
from experiments.deployment.raw import RawRecorder,sha,write_json
from experiments.deployment.live import GazeboObserver
import time


class RobotObserver(GazeboObserver):
    def require_feedback(self,timeout_s):
        deadline=time.monotonic()+timeout_s
        while time.monotonic()<deadline:
            if self.error:raise RuntimeError(self.error)
            if self.joint_states:return
            self.stop.wait(.02)
        raise RuntimeError('No observed physical joint states; motion prohibited')


def physical_environment_guard():
    expected=dict(GATO_MACHINE='jetson',GATO_ROS_PROFILE='tiago',ROS_DOMAIN_ID='2',RMW_IMPLEMENTATION='rmw_cyclonedds_cpp')
    if any(os.environ.get(k)!=v for k,v in expected.items()):
        raise RuntimeError('Source the Jetson tiago ROS profile/domain2/CycloneDDS before manual robot execution')


def resolve(backend,task):
    p=json.loads(INPUT.read_text())
    if backend not in RUNTIME_PINS or task not in TASKS:raise ValueError('Unknown backend/task')
    p.update(backend=backend,task_id=task,timing_mode='B' if backend=='internal' else 'live',
             runtime_pin=RUNTIME_PINS[backend],input_sha256=sha(INPUT))
    if task=='square':p.update(kind='reaching',solver_params=p['reaching_solver_params'])
    else:
        p.update(next(t for t in p['tracking_tasks'] if t['id']==task));p.update(kind='tracking',solver_params=p['tracking_solver_params'])
    return p


def runtime_identity(source,pin):
    # New collector commits can differ; production runtime bytes must match the reviewed pin.
    paths=('python/bsqp/interface.py','python/bsqp/common.py','python/bindings.cu',
        'tiago_src/gato_tiago/tiago_mpc_controller.py','tiago_src/gato_tiago/final_approach.py',
        'tiago_src/gato_tiago/tiago_controller_process.py','tiago_src/gato_tiago/config.py',
        'tiago_src/gato_tiago/safety_monitor.py')
    if pin==RUNTIME_PINS['internal']:paths+=('python/bsqp/offline_mpc.py',)
    hashes={}
    import hashlib
    for relative in paths:
        expected=subprocess.check_output(['git','-C',str(source),'show',pin+':'+relative])
        if (source/relative).read_bytes()!=expected:raise ValueError('Runtime bytes differ: '+relative)
        hashes[relative]=hashlib.sha256(expected).hexdigest()
    if subprocess.check_output(['git','-C',str(source),'diff','--name-only',pin,'--','gato'],text=True).strip():raise ValueError('Native source differs from reviewed runtime pin')
    native_source_tree=subprocess.check_output(['git','-C',str(source),'rev-parse',pin+':gato'],text=True).strip()
    return dict(native_source_tree=native_source_tree,runtime_source_commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
                reviewed_runtime_pin=pin,runtime_sha256=hashes,collector_sha256=sha(__file__))


def preflight_identity(p):
    keys=('N','dt_s','sim_dt_s','initial_state','reaching_solver_params','tracking_solver_params','live_controller','urdf')
    return hashlib.sha256(json.dumps({k:p[k] for k in keys},sort_keys=True).encode()).hexdigest()


def resolve_preflight(backend):
    p=resolve(backend,'square')
    p.update(task_id='canonical-preflight',goals=[dict(id='canonical-preflight',pose_m_rad=p['canonical_preflight_goal'])],
             goal_timeout_s=8.,position_threshold_m=.05,axis_threshold_deg=30.,speed_L1_threshold_rad_s=1.,goal_dwell_time_s=0.,final_approach=False,reaching_stopping_criteria=dict(position_m=.05,tool_axis_deg=30.,speed_L1_rad_s=1.,dwell_s=0.))
    return p


def preflight_guard(p,identity,path):
    if path is None:raise RuntimeError('Supply successful same-machine/backend canonical preflight evidence before demo motion')
    path=Path(path).resolve();metrics=json.loads((path/'metrics.json').read_text());previous=json.loads((path/'provenance.json').read_text())
    valid=(metrics.get('task_id')=='canonical-preflight' and metrics.get('backend')==p['backend']
        and metrics.get('machine')==os.environ.get('GATO_MACHINE') and metrics.get('input_sha256')==p['input_sha256']
        and metrics.get('preflight_setup_sha256')==preflight_identity(p) and metrics.get('task_success') is True
        and metrics.get('physical_pass') is True and not metrics.get('execution_error'))
    if p['backend']!='internal':valid=valid and metrics.get('handback_confirmed') is True
    valid=valid and all(previous.get(k)==identity.get(k) for k in ('runtime_sha256','native_source_tree','module','preflight_setup_sha256','collector_sha256'))
    if not valid:raise RuntimeError('Canonical preflight identity/task/physical/handback mismatch; affected family held')


def rough_reaching_metrics(p,stats):
    import numpy as np
    outcomes=list(stats.get('goal_outcomes',[]));count=len(p['goals'])
    if p['task_id']=='canonical-preflight':
        return dict(task_success=outcomes==['reached'],rough_goal_success=[outcomes==['reached']],settling_goal_outcomes=outcomes)
    indices=np.asarray(stats.get('goal_indices',[]));distance=np.asarray(stats.get('goal_distances',[]));axis=np.asarray(stats.get('tool_axis_errors',[]));v=np.asarray(stats.get('joint_velocities',[]))
    if not (len(indices)==len(distance)==len(axis)==len(v)) or (len(v) and v.shape!=(len(indices),7)):
        return dict(task_success=None,rough_unavailable='malformed feedbackgate arrays; raw retained',settling_goal_outcomes=outcomes)
    speed=np.abs(v).sum(axis=1) if len(v) else np.array([])
    rough=[]
    for i in range(count):
        hit=(indices==i)&np.isfinite(distance)&np.isfinite(axis)&np.isfinite(speed)&(distance<=p['position_threshold_m'])&(axis<=np.deg2rad(p['axis_threshold_deg']))&(speed<=p['speed_L1_threshold_rad_s'])
        # A stricter reached outcome implies the rough gate, including the final
        # terminal feedback that the unchanged public loop may omit from arrays.
        rough.append(bool(np.any(hit) or (i<len(outcomes) and outcomes[i]=='reached')))
    return dict(task_success=all(rough),rough_goal_success=rough,settling_goal_outcomes=outcomes,
                settling_success=outcomes==['reached']*count,rough_scope='Any recordedfeedback inroughregion pergoal, plus stricter reachedterminal observations; zero roughdwell')


def execute(mpc,p,controller,recorder):
    import numpy as np
    common=dict(sim_dt=p['sim_dt_s'],controller=controller,controller_timeout=8.,controller_startup_timeout=30.)
    if p['backend']=='internal':common.update(offline_timing='solve_time',record_raw=True,record_callback=recorder)
    initial=np.asarray(p['initial_state'],dtype=np.float32)
    if p['kind']=='reaching':
        return mpc.run_mpc_goals(initial,np.asarray([g['pose_m_rad'] for g in p['goals']],dtype=np.float32),
            goal_timeout=p['goal_timeout_s'],goal_threshold=p['reaching_stopping_criteria']['position_m'],
            goal_axis_threshold=np.deg2rad(p['reaching_stopping_criteria']['tool_axis_deg']),velocity_threshold=p['reaching_stopping_criteria']['speed_L1_rad_s'],
            goal_dwell_time=p['reaching_stopping_criteria']['dwell_s'],reset_rho_each_solve=False,initial_guess=None,**common)
    return mpc.run_mpc_fig8(initial,np.asarray(p['reference_packed'],dtype=np.float64),sim_time=p['duration_s'],**common)


def run_task(p,source,module,output,preflight_output=None):
    import numpy as np
    from experiments.deployment.live import RecordedSolver,RecordedController,execute_with_handback,GazeboObserver,record_cleanup_event
    from experiments.deployment.evaluate import physical_metrics,task_metrics,distribution,evaluate_live
    write_json(output/'protocol.json',p)
    identity={}
    recorder=RawRecorder(output);controller=observer=mpc=model=None;stats={};errors=[]
    handlers={}
    metrics=dict(task_id=p['task_id'],campaign_id='demo_visual_square',machine=os.environ.get('GATO_MACHINE'),input_sha256=p['input_sha256'],preflight_setup_sha256=preflight_identity(p),backend=p['backend'],timing_mode=p['timing_mode'],
                 task_success=None,physical_pass=None,solver_numerical_pass=None,execution_error=None)
    try:
        def interrupted(signum,frame):raise InterruptedError(f'Demo interrupted by signal{signum}')
        for sig in (signal.SIGINT,signal.SIGTERM):handlers[sig]=signal.signal(sig,interrupted)
        identity=runtime_identity(source,p['runtime_pin']);identity['preflight_setup_sha256']=preflight_identity(p);write_json(output/'provenance.json',identity)
        if p['backend']!='internal':
            if p['backend']=='gazebo':
                from experiments.deployment.protocol import gazebo_environment_guard
                gazebo_environment_guard();observer=GazeboObserver(output);observer.require_advancing_clock(8.)
            else:
                physical_environment_guard()
                observer=RobotObserver(output);observer.require_feedback(8.)
        sys.path[:0]=[str(source/'python'),str(source/'tiago_src')]
        if module is None:raise ValueError('Supply the exact reviewed N32module; installed fallback is disabled')
        if module:
            importlib.import_module('bsqp');name='bsqp.bsqpN32_tiago_right'
            spec=importlib.util.spec_from_file_location(name,module);native=importlib.util.module_from_spec(spec);sys.modules[name]=native;spec.loader.exec_module(native)
        import pinocchio as pin
        from gato_tiago.tiago_mpc_controller import MPC_GATO
        urdf=source/p['urdf']['path']
        if sha(urdf)!=p['urdf']['sha256']:raise ValueError('Model bytes differ')
        model=pin.buildModelFromUrdf(str(urdf));model.gravity.linear=np.asarray(p['gravity_m_s2'])
        mpc=MPC_GATO(model,str(urdf),N=p['N'],dt=p['dt_s'],batch_size=1,constant_f_ext=np.asarray(p['constant_f_ext'],dtype=np.float32),track_full_stats=True,
            plant_type='tiago_right',pendulum_config=None,solver_params=p['solver_params'],final_approach=p['kind']=='reaching' and p['final_approach'],
            final_approach_params=p['final_approach_params'],final_approach_trigger=p['final_approach_trigger'])
        native=sys.modules['bsqp.bsqpN32_tiago_right']
        if Path(native.__file__).resolve()!=Path(module).resolve() or mpc.solver.lib is not native:raise RuntimeError('Loaded solver module differs from explicit selection')
        identity['module']=dict(path=native.__file__,sha256=sha(native.__file__))
        write_json(output/'provenance.json',identity)
        if p['task_id']!='canonical-preflight':preflight_guard(p,identity,preflight_output)
        if p['backend']=='internal':_,stats=execute(mpc,p,None,recorder)
        else:
            mpc.solver=RecordedSolver(mpc.solver,recorder)
            from gato_tiago.tiago_controller_process import TiagoControllerOrchestrator
            settings=dict(p['live_controller']);settings['target_hz']=125. if p['kind']=='tracking' else 100.
            # Current orchestrator uses its unchanged wrapped collision URDF default.
            settings.pop('collision_urdf_path')
            settings['collision_blacklist_path']=str(source/settings['collision_blacklist_path'])
            if p['backend']=='physical_robot':
                print(json.dumps(p,indent=2))
                print('Physical motion: reset2s to',p['initial_state'][:7],'; task',p['task_id'])
                confirmation='RUN '+p['task_id']+' ON ROBOT'
                if input('Type '+repr(confirmation)+' to authorize motion: ').strip()!=confirmation:raise RuntimeError('Motion confirmation declined')
            controller=RecordedController(TiagoControllerOrchestrator(**settings),recorder)
            stats,errors=execute_with_handback(controller,lambda live:execute(mpc,p,live,recorder),output,recorder,8.)
    except BaseException as exc:
        errors.append(dict(stage='execution',message=str(exc),traceback=traceback.format_exc()))
        record_cleanup_event(recorder,'execution_failure',errors[-1],errors)
        if mpc is not None:stats=getattr(mpc,'last_run_stats',stats)
    finally:
        if controller is not None and not controller.close_attempted:
            _,cleanup=execute_with_handback(controller,lambda _:(None,{}),output,recorder,8.);errors.extend(cleanup)
        if observer is not None:
            try:observer.close()
            except BaseException as exc:errors.append(dict(stage='observer',message=str(exc)))
        recorder('controller_stats',stats)
        write_json(output/'controller_stats.json',stats)
        try:
            if model is not None:
                if p['backend']=='internal':
                    steps=[r for r in stats.get('raw_execution',[]) if 'state_after' in r]
                    states=np.asarray([p['initial_state']]+[r['state_after'] for r in steps]);controls=np.asarray([r['control'] for r in steps]).reshape(-1,7)
                    validation=dict(physical_validation=dict(position_tolerance_rad=1e-4,velocity_tolerance_rad_s=1e-3,effort_tolerance_Nm=1e-3),physical_scope='All recorded simulatedplant substeps/controls')
                    metrics.update(physical_metrics(validation,model,states[:,:7],states[:,7:],controls))
                    metrics['measured_actuator_torque_available']=False
                    metrics['simulated_applied_control_available']=True
                    if p['kind']=='reaching':metrics['task_success']=bool(list(stats.get('goal_outcomes',[]))==['reached']*len(p['goals']))
                    elif steps:
                        times=np.asarray([0.]+[r['time_s']+r['step_dt_s'] for r in steps])
                        positions=[];rotations=[];fd=model.createData();tool=model.getFrameId('arm_right_tool_link')
                        for state in states:
                            pin.framesForwardKinematics(model,fd,np.asarray(state[:7],dtype=float));positions.append(fd.oMf[tool].translation.copy());rotations.append(fd.oMf[tool].rotation.copy())
                        metrics.update(task_metrics(p,times,positions,rotations,states[:,7:],stats))
                        axes=np.asarray(rotations)[:,:,2]
                        metrics['upward_tool_axis_max_error_deg']=float(np.max(np.rad2deg(np.arccos(np.clip(axes[:,2],-1.,1.)))))
                else:
                    metrics.update(handback_confirmed=bool(controller and controller.handback_confirmed),measured_actuator_torque_available=False)
                    live_protocol=dict(p,physical_validation=dict(position_tolerance_rad=1e-4,velocity_tolerance_rad_s=1e-3,effort_tolerance_Nm=1e-3),
                        physical_scope='Observedjointpositions/velocities/publishedeffort; actuator torque unavailable',
                        nominal_feedback_s=p['feedback_s'][p['kind']])
                    live_stats=stats
                    if p['kind']=='reaching':
                        live_protocol.update(goal=p['goals'][-1]['pose_m_rad'],timeout_s=p['goal_timeout_s']*len(p['goals']))
                        live_stats=dict(stats,goal_outcomes=[list(stats.get('goal_outcomes',[]))[-1]] if stats.get('goal_outcomes') else [])
                    metrics.update(evaluate_live(live_protocol,model,controller,observer,live_stats,recorder,errors))
                    if p['kind']=='reaching':
                        metrics.update(square_goal_outcomes=list(stats.get('goal_outcomes',[])),
                            task_success=bool(metrics.get('task_success') and list(stats.get('goal_outcomes',[]))==['reached']*len(p['goals'])),
                            final_pose_gate_scope='Finalcenterpose checkedindependently; pergoaloutcomes from unchangedpubliccontroller')
                    elif controller and controller.states:
                        axis_errors=[];fd=model.createData();tool=model.getFrameId('arm_right_tool_link')
                        for state in controller.states:
                            pin.framesForwardKinematics(model,fd,np.asarray(state['q'],dtype=float));axis_errors.append(np.rad2deg(np.arccos(np.clip(fd.oMf[tool].rotation[2,2],-1.,1.))))
                        metrics['upward_tool_axis_max_error_deg']=float(np.max(axis_errors))
                if mpc is not None:
                    if p['backend']=='internal':
                        solves=stats.get('raw_solves',[])
                        metrics.update(wrapper_wall_s=distribution([r.get('wall_solve_time_s',np.nan) for r in solves]),
                            returned_native_time_us=distribution([r.get('native_solve_time_ms',np.nan)*1000 for r in solves]))
                    else:
                        metrics.update(wrapper_wall_s=distribution(mpc.solver.wall_times),returned_native_time_us=distribution(mpc.solver.native_times_us))
                        errors.extend(mpc.solver.recording_errors)
                    metrics['final_approach_switches']=stats.get('final_approach_switches',[])
        except BaseException as exc:errors.append(dict(stage='evaluation',message=str(exc)))
        if p['kind']=='reaching':metrics.update(rough_reaching_metrics(p,stats))
        metrics.update(execution_error=errors or None,unavailable={'full_motion_collision_certificate':'notclaimed; runtime monitoractive live, screensendpointonly',
            'tracking_axis_gate':'descriptiveaxiserror; positiontrackinggates provisional' if p['kind']=='tracking' else None})
        write_json(output/'metrics.json',metrics);recorder.manifest()
        if p['backend']=='physical_robot':
            manual_index=ROOT/'.manual-robot-results'/'jetson'/'manual-robot-index.jsonl'
            manual_index.parent.mkdir(parents=True,exist_ok=True)
            with manual_index.open('a') as stream:
                stream.write(json.dumps(dict(actual_path=str(output.resolve()),backend='physical_robot',task_id=p['task_id'],
                    config_sha256=sha(output/'protocol.json'),script_sha256=sha(__file__),input_sha256=sha(INPUT),
                    data_manifest_sha256=sha(output/'SHA256SUMS.json'),runtime_identity=identity,metrics=metrics))+'\n')
    for sig,old in handlers.items():signal.signal(sig,old)
    return metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=tuple(RUNTIME_PINS),required=True)
    parser.add_argument('--tasks',nargs='+',choices=TASKS,default=list(TASKS))
    parser.add_argument('--source',type=Path,default=ROOT,help='Reviewed runtime checkout; exact internal/live source pins are printed by --preview')
    parser.add_argument('--module',type=Path,help='Required exact reviewed N32module for execution; no installed fallback')
    parser.add_argument('--output',type=Path,help='Fresh cohort path; default timestamp beneath collector checkout')
    parser.add_argument('--preview',action='store_true')
    parser.add_argument('--preflight',action='store_true',help='Run only the canonical rough short reach')
    parser.add_argument('--preflight-output',type=Path)
    args=parser.parse_args();selected=set(args.tasks)
    index=[dict(task_id=t,backend=args.backend,status='selected' if t in selected else 'skipped',reason=None if t in selected else 'explicituserselection') for t in TASKS]
    if args.preflight:index=[dict(task_id='canonical-preflight',backend=args.backend,status='selected',reason=None)]+[dict(task_id=t,backend=args.backend,status='skipped',reason='preflightonly') for t in TASKS]
    if args.preview:
        print(json.dumps(dict(selection=index,protocols=[resolve_preflight(args.backend)] if args.preflight else [resolve(args.backend,t) for t in TASKS if t in selected]),indent=2));return 0
    output=args.output or (ROOT/'.manual-robot-results'/'jetson' if args.backend=='physical_robot' else ROOT/'.demo-results'/args.backend)/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output.mkdir(parents=True,exist_ok=False);write_json(output/'index.json',index)
    for row in index:
        if row['status']=='skipped':continue
        raw=output/row['task_id'];raw.mkdir()
        metrics=run_task(resolve_preflight(args.backend) if args.preflight else resolve(args.backend,row['task_id']),args.source.resolve(),args.module,raw,args.preflight_output)
        row.update(status='execution_failed' if metrics['execution_error'] else 'completed',raw_path=str(raw.resolve()),metrics=metrics)
        write_json(output/'index.json',index)
        if args.backend!='internal' and (metrics['execution_error'] or metrics.get('handback_confirmed') is not True):
            for pending in index:
                if pending['status']=='selected':pending.update(status='held',reason='liveexecutionorsafety/handbackfailure')
            write_json(output/'index.json',index);return 1
    return int(any(r['status']=='execution_failed' for r in index))


if __name__=='__main__':raise SystemExit(main())
