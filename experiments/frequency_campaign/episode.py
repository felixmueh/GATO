"""Fixed-slot modeA collector with injected solver/plant; no native imports."""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
# Load pure sampling helpers directly, bypass package __init__/native interfaces.
_spec=importlib.util.spec_from_file_location('f01_common',ROOT/'python/bsqp/common.py')
common=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(common)


def array_hash(a):
    a=np.asarray(a)
    return dict(sha256=hashlib.sha256(a.tobytes()).hexdigest(),dtype=a.dtype.str,shape=list(a.shape))


def cold_seed(state,N):
    seed=np.zeros((1,N*21-7))
    for k in range(N):seed[0,k*21:k*21+14]=state
    return seed


def timed_solve(owner,state,reference,warm,metadata,emit,clock=time.perf_counter):
    record=dict(metadata,state=state.copy(),reference=reference.copy(),warm_start=warm.copy(),
        replay_input_hashes={name:array_hash(a) for name,a in [('state',state),('reference',reference),('warm_start',warm)]},
        replay_scope='exact wrapper state/reference/shifted primal inputs only; not complete native persistent-state replay',
        unmeasured_internal_state={'rho':'persistent native rho/drho not exported; configured rho is not actual evolving rho',
            'duals':'persistent native dual buffers not exported; fresh-episode/reset history retained'},status='started')
    emit('solve_started',record)
    started=clock()
    try:output,native_us=owner.solver.solve(state,reference,warm)
    except BaseException as error:
        record.update(status='failed',wall_solve_time_s=float(clock()-started),error_type=type(error).__name__,error=str(error))
        emit('solve_failed',record);raise
    wall=float(clock()-started)
    record.update(status='completed',wall_solve_time_s=wall,native_solve_time_ms=float(native_us)/1000.,
        output=np.asarray(output).copy(),
        numerical_certificate=None,numerical_reason='native diagnostics are not a full nonlinear certificate')
    # Persist returned bytes before optional stats normalization can fail.
    emit('solve_returned',record)
    try:record['solver_stats']=deepcopy(owner.solver.get_stats())
    except BaseException as error:
        record.update(status='telemetry_failed',solver_stats=None,telemetry_error=str(error))
        emit('solve_failed',record);raise
    emit('solve_completed',record)
    output=np.asarray(output)
    if output.shape!=warm.shape or not np.isfinite(output).all():raise ValueError('Invalid returned plan; raw bytes retained')
    return output.copy(),record


def observe(model,state,goal,control=None):
    import pinocchio as pin
    q=state[:7].astype(np.float64);v=state[7:].astype(np.float64);data=model.createData()
    pin.framesForwardKinematics(model,data,q);placement=data.oMf[model.getFrameId('arm_right_tool_link')]
    distance=float(np.linalg.norm(placement.translation-goal[:3]));target=pin.rpy.rpyToMatrix(*goal[3:])
    axis=float(np.rad2deg(np.arccos(np.clip(placement.rotation[:,2]@target[:,2],-1,1))))
    full=float(np.rad2deg(np.linalg.norm(pin.log3(target.T@placement.rotation))))
    excess=dict(q_rad=float(np.max(np.maximum(np.maximum(model.lowerPositionLimit-q,q-model.upperPositionLimit),0))),
        v_rad_s=float(np.max(np.maximum(abs(v)-model.velocityLimit,0))),
        u_Nm=None if control is None else float(np.max(np.maximum(abs(control)-model.effortLimit,0))))
    return dict(position_error_m=distance,tool_axis_error_deg=axis,full_SO3_error_deg=full,
        speed_L1_rad_s=float(np.linalg.norm(v,ord=1)),rough_reached=bool(distance<.05 and axis<30 and np.linalg.norm(v,ord=1)<1),
        physical_pass=bool(excess['q_rad']<=1e-4 and excess['v_rad_s']<=1e-3 and (excess['u_Nm'] is None or excess['u_Nm']<=1e-3)),limit_excess=excess)


def collect_episode(owner,episode,arrays,cell,emit,integrate,observer=None,clock=time.perf_counter):
    """Twenty solves followed by twenty30ms intervals, including last interval.

    Scientific rough/physical failures do not truncate; exceptions/nonfinite plant
    stop with missing slots. Solver constructor belongs to injected factory.
    """
    N=cell['N'];state=arrays['starts'][episode['start_index']].copy();warm=cold_seed(state,N)
    owner.solver.reset_dual()
    if episode['kind']=='reaching':owner.solver.reset_rho()
    report=dict(episode_id=episode['id'],completed_updates=0,completed_solves=0,scheduled_updates=20,
        execution_error=None,attempted_solves=0,task_success=None,physical_pass=True,physical_observed=observer is not None,
        solver_numerical_pass=None,subset_scope='.6s timing subset; no full T01 qualification',
        reaching_segments_reached=[False,False] if episode['kind']=='reaching' else None,observations=[])
    active_slot=None
    try:
        if observer is not None:
            initial_goal=(arrays['goals'][episode['goal_indices'][0]] if episode['kind']=='reaching'
                else common.sample_reference(arrays[f"reference_{episode['reference_index']}"],0.,.03))
            initial=observer(state,initial_goal,None)
            report['initial_observation']=deepcopy(initial)
            report['physical_pass'] &= initial['physical_pass']
            emit('observation',dict(episode_id=episode['id'],slot=None,substep=None,time_s=0.,
                initial_state=True,state=state.copy(),control=None,control_unavailable_reason='no control applied before first solve',**initial))
            # Initial rough gate is retained separately; no motion/segment success inferred.
        for slot in range(20):
            active_slot=slot;stamp=slot*.03
            if episode['kind']=='reaching':
                goal_index=episode['goal_indices'][int(slot>=10)];goal=arrays['goals'][goal_index]
                reference=np.tile(goal,N)
                if slot==10:
                    owner.solver.reset_rho()
                    emit('rho_reset',dict(episode_id=episode['id'],slot=slot,time_s=stamp,reason='public reaching goal-switch policy; duals preserved'))
            else:
                goal_index=-1;ref=arrays[f"reference_{episode['reference_index']}"]
                reference=common.sample_reference_horizon(ref,stamp,.03,N)
                goal=reference[:6]
            if slot==10 and episode['kind']=='reaching':emit('goal_switch',dict(episode_id=episode['id'],slot=slot,time_s=stamp,goal_index=goal_index))
            owner.update_force_batch(state[:7])
            report['attempted_solves']+=1
            output,raw=timed_solve(owner,state[None,:],reference[None,:],warm,
                dict(episode_id=episode['id'],slot=slot,time_s=stamp,category=episode['categories'][slot],
                    goal_index=goal_index,warm_start_shift_s=0. if slot==0 else .03,
                    caps=dict(SQP=cell['SQP'],PCG=cell['PCG']),N=N,timing_mode='A',effective_solver_params=deepcopy(owner.solver_params)),emit,clock)
            report['completed_solves']+=1
            emit('plan_activated',dict(episode_id=episode['id'],slot=slot,time_s=stamp,plan_age_s=0.))
            q=state[:7].copy();dq=state[7:].copy()
            for substep in range(10):
                age=substep*.003
                control=common.sample_packed_trajectory_control(output[0],14,7,N,age+1e-12,.03).copy()
                step=dict(episode_id=episode['id'],slot=slot,substep=substep,time_s=stamp+age,step_dt_s=.003,
                    control=control.copy(),state_before=np.r_[q,dq].copy(),status='started')
                emit('step_started',step)
                try:
                    q,dq=integrate(q,dq,control,.003);after=np.r_[q,dq]
                    step.update(state_after=after.copy(),status='completed');emit('step_completed',step)
                    if not np.isfinite(after).all():raise ValueError('Nonfinite plant state; raw bytes retained')
                except BaseException as error:
                    step.update(status='failed',error_type=type(error).__name__,error=str(error));emit('step_failed',step);raise
                if observer is not None:
                    target=goal if episode['kind']=='reaching' else common.sample_reference(ref,stamp+(substep+1)*.003,.03)
                    observed=observer(after,target,control)
                    emit('observation',dict(episode_id=episode['id'],slot=slot,substep=substep,time_s=stamp+(substep+1)*.003,**observed))
                    report['physical_pass'] &= observed['physical_pass']
                    report['observations'].append(observed)
                    if episode['kind']=='reaching':report['reaching_segments_reached'][int(slot>=10)] |= observed['rough_reached']
            state=np.concatenate([q,dq])
            warm=common.shift_packed_trajectory_warm_start(output,state,14,7,N,.03,.03)
            report['completed_updates']+=1
            emit('interval_completed',dict(episode_id=episode['id'],slot=slot,time_s=stamp,actual_interval_s=.03,new_state=state.copy()))
        if episode['kind']=='reaching' and observer is not None:report['task_success']=all(report['reaching_segments_reached'])
    except BaseException as error:
        report['execution_error']=dict(type=type(error).__name__,error=str(error),slot=active_slot)
        emit('episode_failed',report)
    report['missing_solve_slots']=list(range(report['attempted_solves'],20))
    report['missing_update_slots']=list(range(report['completed_updates'],20))
    if observer is None:report['physical_pass']=None
    emit('episode_finished',report)
    return report


def collect_cell(factory,bank,arrays,cell,emit,integrator,observer=None,clock=time.perf_counter):
    """Disposable unrelated warmup then fresh full solver/controller per episode."""
    episodes=bank['variants'][cell['variant']]['episodes'];reports=[]
    # Factory must create a new solver with exact preset and two cap overrides.
    def make_owner(kind):
        created=factory(kind,cell)
        expected=dict(bank['presets'][kind],max_sqp_iters=cell['SQP'],max_pcg_iters=cell['PCG'])
        if created.N!=cell['N'] or created.dt!=.03 or created.batch_size!=1 or created.solver_params!=expected:
            if hasattr(created,'close'):created.close()
            raise ValueError('Effective owner/preset differs from frozen cell')
        return created
    owner=None
    try:
        owner=make_owner('reaching')
        state=arrays['warmup_state'].copy();owner.solver.reset_dual();owner.solver.reset_rho();owner.update_force_batch(state[:7])
        timed_solve(owner,state[None,:],np.tile(arrays['warmup_goal'],cell['N'])[None,:],cold_seed(state,cell['N']),
            dict(category='process_cuda_warmup',excluded=True,episode_id=None,N=cell['N'],caps=dict(SQP=cell['SQP'],PCG=cell['PCG'])),emit,clock)
    except BaseException as error:
        result=dict(execution_error=str(error),episodes=[],missing_episodes=[e['id'] for e in episodes]);emit('cell_failed',result);return result
    finally:
        if owner is not None and hasattr(owner,'close'):owner.close()
    for index,episode in enumerate(episodes):
        owner=None
        try:
            owner=make_owner(episode['kind'])
            report=collect_episode(owner,episode,arrays,cell,emit,integrator(owner),observer,clock)
        except BaseException as error:
            report=dict(episode_id=episode['id'],execution_error=str(error),missing_solve_slots=list(range(20)),missing_update_slots=list(range(20)))
            emit('episode_failed',report)
        finally:
            if owner is not None and hasattr(owner,'close'):owner.close()
        reports.append(report)
        if report['execution_error'] is not None:break
    result=dict(episodes=reports,missing_episodes=[e['id'] for e in episodes[len(reports):]],
        execution_error=next((r['execution_error'] for r in reports if r['execution_error'] is not None),None))
    emit('cell_finished',result);return result
