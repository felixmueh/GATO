"""Normal cold-start full-pose reaching: Tiago defaults or explicit cost overrides, B1 versus B16.

No IK or endpoint-conditioned initial trajectory. Pool goal joints are used only
for FK. Prediction spacing and the fixed simulated feedback interval are separate.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import numpy as np
import pinocchio as pin
from validate_reaching_mpc import (ROOT, MODEL, digest, safe, write_json, pose,
    metrics as full_pose_metrics, state_cost, physical, initialize_warm_start, rk4,
    shift_packed_trajectory_warm_start, TIAGO_TRACKING_SOLVER_PARAMS, MPC_GATO,
    load_protocol, COHORT_ID, PROTOCOL_PATH, NativeCapture, solve_recorded,
    protocol_path, cohort_id)

N, NX, NU = 32, 14, 7


def metrics(model, data, x, goal):
    result = full_pose_metrics(model, data, x, goal)
    actual = pose(model, data, x[:7])
    result['tool_axis_error_deg'] = float(np.rad2deg(np.arccos(np.clip(
        actual.rotation[:, 2] @ goal.rotation[:, 2], -1., 1.))))
    return result


def unpack(z):
    z = np.atleast_2d(z)
    body = z[:, :(N-1)*21].reshape(-1, N-1, 21)
    return np.concatenate((body[:,:,:14], z[:,None,-14:]), axis=1), body[:,:,14:]


def pack(x, u):
    return np.concatenate((np.concatenate((x[:,:-1],u), axis=2).reshape(len(x),-1), x[:,-1]), axis=1).astype(np.float32)


def make_controller(model, batch, dt, args):
    cfg = dict(load_protocol(protocol_path(args))['solver_params'])
    for key in ('vel_lim_cost', 'qd_cost'):
        value = getattr(args, key)
        if value is not None and value!=cfg[key]:raise ValueError('Frozen current cost override differs: '+key)
    controller = MPC_GATO(model, str(MODEL), N=N, dt=dt, batch_size=batch,
                          plant_type='tiago_right', solver_params=cfg, track_full_stats=True)
    # Batching means initial trajectories, not the generic force-hypothesis mode.
    controller.force_estimator = None
    controller.solver.set_f_ext_B(np.zeros((batch,6), dtype=np.float32))
    controller.solver.reset_dual()
    controller.solver.solver = NativeCapture(controller.solver.solver)
    return controller, cfg


def perturbations(model, nominal, batch, dt, rng):
    """Analytic smooth delta q/v/a; RNEA about the nominal ABA acceleration.

    The s^2 envelope has delta q(0)=delta v(0)=0. No terminal condition is
    imposed. All coefficients are fresh, independent of target configuration.
    Nominal dynamic defects are retained; perturbed seeds need not be feasible.
    """
    if batch == 1: return nominal[None].copy(), []
    xs, us = unpack(nominal)
    base_x, base_u = xs[0].astype(float), us[0].astype(float)
    data = model.createData()
    base_a = np.asarray([pin.aba(model,data,x[:7],x[7:],u).copy() for x,u in zip(base_x,base_u)])
    s = np.linspace(0,1,N)[:,None]
    omega = np.pi*np.arange(1,4)[None,:]
    b = s*s*np.cos(s*omega)
    db = 2*s*np.cos(s*omega)-s*s*omega*np.sin(s*omega)
    ddb = 2*np.cos(s*omega)-4*s*omega*np.sin(s*omega)-s*s*omega**2*np.cos(s*omega)
    duration = (N-1)*dt
    result, metadata = [nominal.copy()], []
    for lane in range(1,batch):
        amplitude = (.02,.05,.1)[(lane-1)//5]
        coeff = rng.uniform(-1,1,(3,7))
        normalizer = max(float(np.max(np.abs(b@coeff))),1e-12)
        dq = amplitude*(b@coeff)/normalizer
        dv = amplitude*(db@coeff)/(normalizer*duration)
        da = amplitude*(ddb@coeff)/(normalizer*duration**2)
        candidate_x = base_x + np.concatenate((dq,dv),axis=1)
        candidate_u = np.asarray([pin.rnea(model,data,x[:7],x[7:],a).copy()
                                  for x,a in zip(candidate_x,base_a+da[:-1])])
        result.append(pack(candidate_x[None],candidate_u[None])[0])
        metadata.append(dict(lane=lane, amplitude_rad=amplitude, coefficients=coeff,
                             peak_delta_q_rad=float(np.max(np.abs(dq)))))
    return np.asarray(result,dtype=np.float32), metadata


def prefix(controller, x, u, duration, sim_dt):
    states, steps = [x.copy()], []
    elapsed = 0.
    while elapsed < duration-1e-12:
        h = min(sim_dt,duration-elapsed)
        q,v = rk4(controller.model,controller.data,states[-1][:7],states[-1][7:],u,h,controller.actual_f_ext)
        states.append(np.r_[q,v]); steps.append(h); elapsed += h
        if not np.isfinite(states[-1]).all(): break
    return np.asarray(states), np.asarray(steps)


def assess(controller, optimized, x, duration, sim_dt, merits, dt, prefix_guard=False,timings=None):
    diagnostic_started=time.perf_counter() if timings is not None else None
    xs, us = unpack(optimized)
    residual = []
    for k in range(N-1):
        residual.append(controller.solver.sim_forward(xs[:,k],us[:,k],dt)-xs[:,k+1])
    residual = np.asarray(residual)
    if timings is not None:timings['diagnostics_wall_s']+=time.perf_counter()-diagnostic_started
    selection_started=time.perf_counter() if timings is not None else None
    rows, prefixes = [], []
    for lane in range(len(xs)):
        finite = bool(np.isfinite(optimized[lane]).all() and np.isfinite(merits[lane]))
        if finite:
            dense, steps = prefix(controller,x,us[lane,0].astype(float),duration,sim_dt)
        else:
            dense, steps = x[None].copy(), np.array([])
        bounds = physical(controller.model,dense,us[lane,:1])
        valid = finite and all(bounds[k] for k in ('finite','joint_bounds_valid','velocity_valid','effort_valid'))
        rows.append(dict(lane=lane, finite=finite, prefix_physical_valid=valid,
                         prefix_physical=bounds, final_merit=float(merits[lane]),
                         local_defect_q_max=float(np.max(np.abs(residual[:,lane,:7]))),
                         local_defect_v_max=float(np.max(np.abs(residual[:,lane,7:]))),
                         local_defect_l1=float(np.sum(np.abs(residual[:,lane])))))
        prefixes.append((dense,steps))
    eligible = [r['lane'] for r in rows if r['finite'] and r['prefix_physical']['finite']
                and (r['prefix_physical_valid'] or not prefix_guard)]
    winner = min(eligible,key=lambda k: (merits[k],k)) if eligible else None
    if timings is not None:timings['selection_wall_s']+=time.perf_counter()-selection_started
    return winner, rows, prefixes


def inputs(args,pair_id):
    protocol=load_protocol(protocol_path(args))
    if protocol['timing_mode']=='B':
        if digest(MODEL)!=protocol['source_pins']['ba_source_sha256'][str(MODEL.relative_to(ROOT))]:
            raise ValueError('Frozen BB model hash differs')
        for name in ('pool.npz','pairs.npz'):
            if digest(args.pool/name)!=protocol['input_sha256'][name]:raise ValueError('Frozen BB task input differs: '+name)
        if getattr(args,'preflight_input',None) and digest(args.preflight_input)!=protocol['preflight']['bb_input_sha256']:
            raise ValueError('Frozen BB preflight input differs')
    model = pin.buildModelFromUrdf(str(MODEL)); data = model.createData()
    if getattr(args,'preflight_input',None):
        task=json.loads(args.preflight_input.read_text())
        if task['solver_params']!=protocol['solver_params']:raise ValueError('Preflight preset differs')
        if task.get('timing_mode')!=protocol['timing_mode']:raise ValueError('Preflight timing mode differs')
        x=np.asarray(task['initial_state'],dtype=float);target=np.asarray(task['goal'],dtype=float)
        goal=pin.SE3(pin.rpy.rpyToMatrix(target[3:]),target[:3])
        source_goal_q=x[:7]+np.asarray(task['endpoint_delta_rad'])
        ref=np.tile(target,N).astype(np.float32)
        return model,data,x,source_goal_q,goal,ref,-1,-1
    pool,pairs = np.load(args.pool/'pool.npz'),np.load(args.pool/'pairs.npz')
    si,gi = int(pairs['start_index'][pair_id]),int(pairs['goal_index'][pair_id])
    x = np.r_[pool['q'][si],np.zeros(7)]
    source_goal_q = pool['q'][gi].copy()
    goal = pose(model,data,source_goal_q)
    ref = np.tile(np.r_[goal.translation,pin.rpy.matrixToRpy(goal.rotation)],N).astype(np.float32)
    return model,data,x,source_goal_q,goal,ref,si,gi


def run_episode(args,pair_id,batch,dt):
    selected_protocol_path=protocol_path(args)
    protocol=load_protocol(selected_protocol_path)
    mode_b=protocol['timing_mode']=='B'
    selected_cohort=cohort_id(protocol)
    if mode_b:
        if args.prefix_guard:raise ValueError('Frozen BB forbids a physical prefix veto')
        if getattr(args,'evidence_kind','measurement')!='synthetic_test':
            for key,value in dict(feedback=.03,sim_dt=.003,seconds=8.,position_tolerance=.05,
                                  tool_axis_tolerance_deg=30.,velocity_l1_tolerance=1.,seed=20260925).items():
                if getattr(args,key)!=value:raise ValueError('Frozen BB arguments differ: '+key)
            if dt!=.03 or batch not in (1,16):raise ValueError('Frozen BB knot spacing/batch differs')
            if not getattr(args,'preflight_input',None) and pair_id not in protocol['task_ids']:
                raise ValueError('Frozen BB task ID differs')
    name=(f'preflight_dt_{dt:g}_B{batch}' if getattr(args,'preflight_input',None) else f'pair_{pair_id:05d}_dt_{dt:g}_B{batch}')
    out = args.output/name
    out.mkdir(parents=True,exist_ok=False)
    model,data,x,goal_q,goal,ref,si,gi = inputs(args,pair_id)
    write_json(out/'preparation.json',dict(cohort_id=selected_cohort,protocol_sha256=digest(selected_protocol_path),pair_id=pair_id,batch=batch,N=N,
               initial_state=x,reference=ref,solver_params=protocol['solver_params']))
    controller,cfg = make_controller(model,batch,dt,args)
    rng_pair_id=0 if getattr(args,'preflight_input',None) else pair_id
    rng = np.random.default_rng(np.random.SeedSequence([args.seed,rng_pair_id,round(dt*1e6)]))
    warm = np.repeat(initialize_warm_start(x,N,NX,NU)[None],batch,axis=0).astype(np.float32)
    reference = np.repeat(ref[None],batch,axis=0)
    np.savez_compressed(out/'inputs.npz',initial_state=x,source_goal_q=goal_q,target_xyz=goal.translation,
                        target_rotation=goal.rotation,reference=reference,first_seed=warm)
    source_paths = [Path(__file__),ROOT/'examples/randomized_multimodal/validate_reaching_mpc.py',MODEL,
                    ROOT/'python/bsqp/mpc_controller.py',ROOT/'python/bsqp/interface.py',
                    ROOT/'python/bsqp/common.py',ROOT/'tiago_src/gato_tiago/config.py',
                    ROOT/'gato/dynamics/tiago_right/tiago_right_plant.cuh']
    if mode_b:source_paths.append(ROOT/'examples/randomized_multimodal/tiago_batching_execution.py')
    manifest = dict(cohort_id=selected_cohort,evidence_kind=getattr(args,'evidence_kind','measurement'),timing_mode=protocol['timing_mode'],offline_timing=protocol['offline_timing'],
        rng_entropy=[args.seed,rng_pair_id,round(dt*1e6)],
        protocol_sha256=digest(selected_protocol_path),preflight_input_sha256=(digest(args.preflight_input) if getattr(args,'preflight_input',None) else None),created=datetime.now(timezone.utc).isoformat(),pair_id=pair_id,batch=batch,
        start_pool_index=si,goal_pool_index=gi,args=vars(args),solver_params=cfg,
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in source_paths},
        source_input_sha256={name:digest(args.pool/name) for name in ('pool.npz','pairs.npz')},
        selected_input_sha256=digest(out/'inputs.npz'),binary_path=controller.solver.lib.__file__,
        binary_sha256=digest(controller.solver.lib.__file__),python=sys.version,numpy=np.__version__,
        pinocchio=pin.__version__,host=platform.node(),N=N,knot_dt=dt,prediction_seconds=(N-1)*dt,
        feedback_seconds=args.feedback,simulation_step=args.sim_dt,simulation_timeout=args.seconds,
        endpoint='soft full pose at every knot; no hard terminal goal; goal joints only used for FK',
        initialization='first solve repeated actual state and zero controls, identical in all lanes; then lane 0 shared warm shift of selected plan, lanes 1:16 fresh smooth perturbations',
        proposal='three random cosine modes times s^2; scales .02/.05/.1 rad with five candidates each; analytic q/v/a perturbations, nominal ABA acceleration plus delta acceleration mapped by RNEA; no endpoint condition, IK, clipping, or amplitude tuning',
        dual_policy='standard MPC: reset once at episode start, retained per lane across solves, including random replacement lanes; no dual permutation or winner replication',
        rho_policy='reset before every solve, adapted within solve',
        selection=('require finite full optimized plan, merit, and RK4 prefix; select lowest solver final_merit, tie by lane; physical bounds diagnostic only; no eligible finite lane terminates episode without execution; no goal-arrival gate' if not args.prefix_guard else 'diagnostic strict physical-prefix veto; lowest final_merit among physically valid finite candidates'),
        physical_limits=f'soft solver penalties and independent raw URDF bounds on executed prefix at {args.sim_dt}s; no hard bound constraints or clipping; no collisions',
        success=f'feedback XYZ < {args.position_tolerance} m, tool-Z-axis angle < {args.tool_axis_tolerance_deg} deg, velocity L1 < {args.velocity_l1_tolerance} rad/s; no dwell; full SO3 diagnostic only; physical validity separate',
        cost='trapezoidal actual-time integral .5*q*XYZ^2 + .5*orient*wrappedRPY^2 + .5*qd*velocity^2 and ZOH .5*u*torque^2 until success/timeout/failure; excludes terminal and bound penalties',
        runtime='fixed simulated feedback period independent of wall time; no real-time claim')
    if mode_b:
        manifest.update(runtime=protocol['clock']['claim'],timing_scope=protocol['clock']['paid_scope'],
            unpaid_timing_scope=protocol['clock']['excluded_scope'],
            selection=protocol['candidate_policy']['selection'],bootstrap=protocol['clock']['unpaid_bootstrap'])
    write_json(out/'manifest.json',manifest); shutil.copyfile(__file__,out/'runner_source.py')
    states,controls,times,solve_rows = [x.copy()],[],[0.],[]
    integrated=np.zeros(4); failure=None; reached=None; pose_touch=None; position_touch=None
    metadata=[]; wall_start=time.perf_counter()
    timings={key:0. for key in ('proposal_generation_wall_s','selection_wall_s','diagnostics_wall_s',
                              'raw_recording_wall_s','plant_replay_wall_s')}
    intervals=[]; executed_plan_ids=[]; executed_plan_ages=[]; current_row=None

    def persist(function,*values,**keywords):
        started=time.perf_counter()
        try:return function(*values,**keywords)
        finally:timings['raw_recording_wall_s']+=time.perf_counter()-started

    def checkpoint():
        extras=dict(plan_solve_indices=executed_plan_ids,active_control_ages_s=executed_plan_ages) if mode_b else {}
        writer=persist if mode_b else lambda f,*a,**kw:f(*a,**kw)
        writer(np.savez_compressed,out/'executed.npz',states=np.asarray(states),controls=np.asarray(controls).reshape(-1,7),times=times,**extras)
        bounds=physical(model,np.asarray(states),np.asarray(controls))
        bounds['physical_valid']=all(bounds[k] for k in ('finite','joint_bounds_valid','velocity_valid','effort_valid'))
        complete=failure is not None or reached is not None or times[-1]>=args.seconds-1e-10
        summary=dict(pair_id=pair_id,batch=batch,knot_dt=dt,prediction_seconds=(N-1)*dt,
            outcome='reached' if reached is not None else ('failed' if failure else ('timeout' if complete else 'running')),
            complete=complete,failure=failure,simulated_seconds=times[-1],feedback_solves=len(solve_rows),
            reached_seconds=reached,first_pose_touch_seconds=pose_touch,first_position_touch_seconds=position_touch,
            initial=metrics(model,data,states[0],goal),final=metrics(model,data,states[-1],goal),physical=bounds,
            selected_nonzero_lanes=sum(r['winner'] not in (None,0) for r in solve_rows),
            accepted_sqp_steps_selected=sum(r['selected_accepted_steps'] for r in solve_rows),
            solve_wall_seconds=sum(r['wall_seconds'] for r in solve_rows),elapsed_wall_seconds=time.perf_counter()-wall_start,
            integrated_cost=dict(zip(('position','orientation_wrapped_rpy','velocity','effort'),integrated)),
            total_integrated_cost=float(integrated.sum()),
            quadratic_executed_cost=float(integrated.sum()),
            quadratic_cost_scope='running position, wrapped RPY, velocity and effort only; excludes limit barriers and terminal cost')
        if mode_b:
            paid=[r for r in solve_rows if r['category']!='startup/cold']
            summary.update(cohort_id=selected_cohort,timing_mode='B',offline_timing='solve_time',
                task_success=reached is not None,controller_failure=failure,
                selection_failure=next((r['selection_failure'] for r in solve_rows if 'selection_failure' in r),None),
                timing_scope=protocol['clock']['paid_scope'],overheads=dict(timings),execution_intervals=intervals,
                startup_wall_solve_time_s=solve_rows[0]['wall_seconds'] if solve_rows else None,
                startup_native_solve_time_ms=solve_rows[0]['gpu_microseconds']/1000 if solve_rows else None,
                wall_solve_times_s=[r['wall_seconds'] for r in paid],
                native_solve_times_ms=[r['gpu_microseconds']/1000 for r in paid],
                above_nominal_share=sum(i['actual_s']>args.feedback for i in intervals if i['category']=='paid')/
                    max(1,sum(i['category']=='paid' for i in intervals)))
        writer(write_json,out/'summary.json',summary)
        return summary

    try:
        with (out/'progress.jsonl').open('w',buffering=1) as log:
            if mode_b:
                from tiago_batching_execution import execute_active_plan
                active_plan=None; active_index=None; activation_s=None
                previous_optimizer_start=0.; step=0
                with (out/'execution.jsonl').open('w',buffering=1) as execution:
                    def emit(event,record):
                        persist(execution.write,json.dumps(safe(dict(event=event,**record)),allow_nan=False)+'\n')

                    def completed_step(next_x,u,h,end_s,plan_index,age):
                        nonlocal x
                        prior=x.copy()
                        x=next_x.copy(); states.append(x.copy()); controls.append(u.copy()); times.append(end_s)
                        executed_plan_ids.append(plan_index); executed_plan_ages.append(age)
                        integrated[:3]+=.5*h*(state_cost(model,data,prior,goal,cfg)+state_cost(model,data,x,goal,cfg))
                        integrated[3]+=h*.5*cfg['u_cost']*float(u@u)

                    while times[-1]<args.seconds-1e-10:
                        feedback=times[-1]; feedback_state=x.copy()
                        shift_s=0. if step==0 else feedback-previous_optimizer_start
                        row=dict(step=step,category='startup/cold' if step==0 else 'warm',
                            feedback_time_s=feedback,optimizer_reference_time_s=feedback,
                            previous_optimizer_reference_time_s=previous_optimizer_start,warm_start_shift_s=shift_s,
                            start_simulated_seconds=feedback,activation_time_s=None,activation_status='pending',
                            winner=None,selected_accepted_steps=0,proposal=metadata,rho_reset=True)
                        current_row=row
                        persist(write_json,out/f'step_{step:04d}_clock.json',dict(row,status='started'))
                        controller.solver.reset_rho()
                        candidates=warm.copy()
                        optimized,gpu_us,wall,stats=solve_recorded(controller,np.repeat(feedback_state[None],batch,axis=0),
                            reference,warm,out,step,cfg,timings=timings)
                        row.update(wall_seconds=wall,gpu_microseconds=gpu_us,
                            solve_arrival_time_s=0. if step==0 else feedback+wall,stats=safe(stats))
                        # Save completed call and its clocks before assessment/replay can fail.
                        solve_rows.append(row)
                        persist(write_json,out/f'step_{step:04d}_clock.json',dict(row,status='completed'))
                        if step and (not np.isfinite(wall) or wall<=0):
                            raise ValueError('paid solver wall duration must be finite and positive')
                        nominal_duration=min(args.feedback,args.seconds-feedback)
                        winner,lanes,prefixes=assess(controller,optimized,feedback_state,nominal_duration,
                            args.sim_dt,stats['final_merit'],dt,False,timings=timings)
                        row.update(winner=winner,lanes=lanes,nominal_selection_prefix_s=nominal_duration)
                        payload={f'lane_{lane}_{kind}':array for lane,(dense,hs) in enumerate(prefixes)
                                 for kind,array in [('states',dense),('step_dt_s',hs)]}
                        persist(np.savez_compressed,out/f'step_{step:04d}_prefixes.npz',**payload)
                        persist(np.savez_compressed,out/f'step_{step:04d}.npz',candidates=candidates,optimized=optimized,
                            actual_state=feedback_state,merits=stats['final_merit'],winner=-1 if winner is None else winner)
                        if winner is not None:
                            row['selected_accepted_steps']=int(np.count_nonzero(np.asarray(stats['step_size'])[:,winner]>0))
                        else:
                            failure='no finite optimized plan, merit, and nominal prefix'
                            row['selection_failure']=failure
                        if step==0:
                            if winner is not None:
                                active_plan=optimized[winner].copy(); active_index=step; activation_s=0.
                                row.update(activation_time_s=0.,activation_status='activated_bootstrap')
                            else:row['activation_status']='discarded_no_eligible_bootstrap'
                        requested=args.feedback if step==0 else wall
                        if active_plan is not None:
                            replay_started=time.perf_counter();raw_before=timings['raw_recording_wall_s']
                            try:
                                actual,clipped=execute_active_plan(active_plan,x,feedback,requested,activation_s,
                                    timeout_s=args.seconds,sim_dt=args.sim_dt,knot_dt=dt,knots=N,nx=NX,nu=NU,
                                    integrate=lambda state,u,h:np.concatenate(rk4(model,controller.data,state[:7],state[7:],u,h,controller.actual_f_ext)),
                                    emit=emit,completed_step=completed_step,plan_index=active_index)
                            finally:
                                timings['plant_replay_wall_s']+=max(0.,time.perf_counter()-replay_started-(timings['raw_recording_wall_s']-raw_before))
                            interval=dict(start_s=feedback,actual_s=actual,requested_s=requested,timeout_clipped=clipped,
                                plan_solve_index=active_index,active_plan_activation_time_s=activation_s,
                                category='startup/unpaid' if step==0 else 'paid')
                            intervals.append(interval);emit('interval_completed',interval)
                            m=metrics(model,data,x,goal)
                            if m['xyz_error']<args.position_tolerance and position_touch is None:position_touch=times[-1]
                            touching=m['xyz_error']<args.position_tolerance and m['tool_axis_error_deg']<args.tool_axis_tolerance_deg
                            if touching and pose_touch is None:pose_touch=times[-1]
                            if touching and m['velocity_l1']<args.velocity_l1_tolerance:reached=times[-1]
                        terminal=reached is not None or times[-1]>=args.seconds-1e-10
                        if step:
                            if terminal:row['activation_status']='discarded_task_terminal'
                            elif winner is None:row['activation_status']='discarded_no_eligible'
                            else:
                                active_plan=optimized[winner].copy();active_index=step;activation_s=times[-1]
                                row.update(activation_time_s=activation_s,activation_status='activated')
                        row['final']=metrics(model,data,x,goal)
                        persist(write_json,out/f'step_{step:04d}_clock.json',dict(row,status='completed'))
                        persist(log.write,json.dumps(safe(row),allow_nan=False)+'\n')
                        summary=checkpoint()
                        if summary['complete']:break
                        generation_started=time.perf_counter()
                        nominal=shift_packed_trajectory_warm_start(optimized[winner],x,NX,NU,N,times[-1]-feedback,dt)
                        warm,metadata=perturbations(model,nominal,batch,dt,rng)
                        timings['proposal_generation_wall_s']+=time.perf_counter()-generation_started
                        previous_optimizer_start=feedback;step+=1
            else:
                step=0
                while times[-1]<args.seconds-1e-10:
                    duration=min(args.feedback,args.seconds-times[-1])
                    controller.solver.reset_rho()
                    candidates=warm.copy()
                    optimized,gpu_us,wall,stats=solve_recorded(controller,np.repeat(x[None],batch,axis=0),reference,warm,out,step,cfg)
                    try:
                        winner,lanes,prefixes=assess(controller,optimized,x,duration,args.sim_dt,stats['final_merit'],dt,args.prefix_guard)
                    except BaseException as error:
                        write_json(out/f'step_{step:04d}_assessment_failure.json',dict(error_type=type(error).__name__,error=str(error)))
                        checkpoint()
                        raise
                    # Preserve every solve immediately, including rejected outputs and all candidates.
                    np.savez_compressed(out/f'step_{step:04d}.npz',candidates=candidates,optimized=optimized,
                                        actual_state=x,merits=stats['final_merit'],winner=-1 if winner is None else winner)
                    row=dict(step=step,start_simulated_seconds=times[-1],duration=duration,wall_seconds=wall,gpu_microseconds=gpu_us,
                             winner=winner,lanes=lanes,proposal=metadata,stats=safe(stats),selected_accepted_steps=0)
                    if winner is None:
                        failure='no finite physically valid executed prefix' if args.prefix_guard else 'no finite optimized plan, merit, and executed prefix'
                    else:
                        row['selected_accepted_steps']=int(np.count_nonzero(np.asarray(stats['step_size'])[:,winner]>0))
                        u=optimized[winner,14:21].astype(float)
                        dense,hs=prefixes[winner]
                        for next_x,h in zip(dense[1:],hs):
                            integrated[:3]+=.5*h*(state_cost(model,data,x,goal,cfg)+state_cost(model,data,next_x,goal,cfg))
                            integrated[3]+=h*.5*cfg['u_cost']*float(u@u)
                            x=next_x.copy(); states.append(x.copy()); controls.append(u.copy()); times.append(times[-1]+h)
                        m=metrics(model,data,x,goal)
                        if m['xyz_error']<args.position_tolerance and position_touch is None: position_touch=times[-1]
                        touching=m['xyz_error']<args.position_tolerance and m['tool_axis_error_deg']<args.tool_axis_tolerance_deg
                        if touching and pose_touch is None: pose_touch=times[-1]
                        if touching and m['velocity_l1']<args.velocity_l1_tolerance: reached=times[-1]
                    row['final']=metrics(model,data,x,goal)
                    solve_rows.append(row); log.write(json.dumps(safe(row),allow_nan=False)+'\n')
            
                    summary=checkpoint()
                    if step%25==0 or summary['complete']:
                        print(json.dumps(safe(dict(pair_id=pair_id,batch=batch,dt=dt,step=step,sim_seconds=times[-1],
                                                  winner=winner,failure=failure,**row['final']))),flush=True)
                    if summary['complete']: break
                    nominal=shift_packed_trajectory_warm_start(optimized[winner],x,NX,NU,N,duration,dt)
                    warm,metadata=perturbations(model,nominal,batch,dt,rng)
                    step+=1
    except BaseException as error:
        failure='technical execution failure: '+str(error)
        write_json(out/'technical_failure.json',dict(error_type=type(error).__name__,error=str(error)))
        if mode_b and current_row is not None:
            row=current_row
            if row['activation_status']=='pending':row['activation_status']='discarded_execution_failure'
            write_json(out/f"step_{row['step']:04d}_clock.json",dict(row,status='failed',error_type=type(error).__name__,error=str(error)))
        checkpoint()
        write_json(out/'receipt.json',{p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='receipt.json'})
        raise

    summary=checkpoint()
    write_json(out/'receipt.json',{p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='receipt.json'})
    print(json.dumps(safe(summary)),flush=True)
    return summary


def gate(args):
    """Bounded repeat, duplicate, heterogeneous and permutation lane checks."""
    out=args.output/'agreement_gate'; out.mkdir(parents=True,exist_ok=False)
    model,data,x,goal_q,goal,ref,si,gi=inputs(args,args.pairs[0])
    dt=args.knot_dts[0]; cold=initialize_warm_start(x,N,NX,NU).astype(np.float32)
    records=[]; arrays={}
    def solve(name,batch,candidates):
        controller,cfg=make_controller(model,batch,dt,args)
        controller.solver.reset_rho()
        z,_=controller.solver.solve(np.repeat(x[None],batch,axis=0),np.repeat(ref[None],batch,axis=0),candidates.copy())
        stats=safe(controller.solver.get_stats())
        records.append(dict(name=name,batch=batch,stats=stats,solver_params=cfg))
        arrays[name+'_candidates']=candidates.copy(); arrays[name+'_optimized']=z.copy()
        return z.copy(),np.asarray(controller.solver.get_stats()['final_merit']).copy()
    one,mone=solve('cold_single',1,cold[None]); repeat,mrepeat=solve('cold_single_repeat',1,cold[None])
    duplicate,mdup=solve('cold_duplicate16',16,np.repeat(cold[None],16,axis=0))
    candidates,_=perturbations(model,cold,16,dt,np.random.default_rng(args.seed))
    batch,mbatch=solve('heterogeneous16',16,candidates)
    permutation=np.random.default_rng(args.seed+1).permutation(16)
    perm,mperm=solve('permuted16',16,candidates[permutation])
    unperm=perm[np.argsort(permutation)]; unmerit=mperm[np.argsort(permutation)]
    def compare(a,ma,b,mb):
        return dict(max_plan_abs=float(np.max(np.abs(a-b))),max_first_control_abs=float(np.max(np.abs(a[...,14:21]-b[...,14:21]))),
                    max_merit_abs=float(np.max(np.abs(ma-mb))),max_merit_relative=float(np.max(np.abs(ma-mb)/np.maximum(1.,np.abs(ma)))))
    comparisons=dict(repeat=compare(one,mone,repeat,mrepeat),duplicate=compare(one,mone,duplicate,mdup),
                     heterogeneous_lane0=compare(one,mone,batch[:1],mbatch[:1]),permutation=compare(batch,mbatch,unperm,unmerit))
    arrays['permutation']=permutation
    np.savez_compressed(out/'solves.npz',**arrays)
    report=dict(comparisons=comparisons,records=records,args=vars(args),
                solver_params=records[0]['solver_params'],
                interpretation='Compare corresponding lanes and repeat noise using practical merit/control differences; exact equality is not required. All gate solves reset dual and rho. Gate does not execute trajectories.')
    write_json(out/'report.json',report)
    write_json(out/'receipt.json',{p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='receipt.json'})
    print(json.dumps(safe(comparisons)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,default=PROTOCOL_PATH,help='explicit frozen A or B protocol; default preserves A')
    p.add_argument('--pool',type=Path,default=ROOT/'experiments/tiago_batching/inputs')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pairs',type=int,nargs='+',default=[0])
    p.add_argument('--batches',type=int,nargs='+',default=[1,16])
    p.add_argument('--knot-dts',type=float,nargs='+',default=[.03])
    p.add_argument('--feedback',type=float,default=.03)
    p.add_argument('--sim-dt',type=float,default=.003)
    p.add_argument('--seconds',type=float,default=8.)
    p.add_argument('--position-tolerance',type=float,default=.05)
    p.add_argument('--tool-axis-tolerance-deg',type=float,default=30.)
    p.add_argument('--velocity-l1-tolerance',type=float,default=1.)
    p.add_argument('--seed',type=int,default=20260925)
    p.add_argument('--vel-lim-cost',type=float,default=None,help='optional soft velocity-limit penalty override; preserves default when omitted')
    p.add_argument('--qd-cost',type=float,default=None,help='optional quadratic joint-speed cost override; preserves default when omitted')
    p.add_argument('--preflight-input',type=Path,help='canonical preflight through this same offline portfolio loop')
    p.add_argument('--prefix-guard',action='store_true',help='diagnostic protocol: veto physically invalid prefixes')
    p.add_argument('--gate',action='store_true')
    p.add_argument('--gate-only',action='store_true')
    args=p.parse_args()
    for key in ('vel_lim_cost', 'qd_cost'):
        value = getattr(args, key)
        if value is not None and (not np.isfinite(value) or value < 0):
            p.error(key.replace('_', '-') + ' must be finite and nonnegative')
    for key in ('position_tolerance', 'tool_axis_tolerance_deg', 'velocity_l1_tolerance'):
        value = getattr(args, key)
        if not np.isfinite(value) or value <= 0:
            p.error(key.replace('_', '-') + ' must be finite and positive')
    if not 0<args.seconds<=8 or not 0<args.sim_dt<=args.feedback or any(dt<args.feedback for dt in args.knot_dts):
        p.error('require 0 < seconds <= 8, 0 < sim-dt <= feedback <= knot-dt')
    if any(b not in (1,16) for b in args.batches): p.error('pilot supports B1 and B16')
    protocol=load_protocol(protocol_path(args))
    if (args.knot_dts!=[.03] or args.feedback!=.03 or args.sim_dt!=.003 or args.seconds!=8. or
            args.position_tolerance!=.05 or args.tool_axis_tolerance_deg!=30. or args.velocity_l1_tolerance!=1.):
        p.error('Current cohort requires exact frozen timing and success gates')
    if args.prefix_guard or args.gate or args.gate_only:p.error('Current cohort uses original no-veto policy, no extra gate campaign')
    if getattr(args,'preflight_input',None):args.pairs=[-1]
    elif protocol['timing_mode']=='B' and any(i not in protocol['task_ids'] for i in args.pairs):
        p.error('BB requires original frozen100 task IDs')
    args.output.mkdir(parents=True,exist_ok=True)
    if args.gate or args.gate_only: gate(args)
    if args.gate_only: return
    for pair_id in args.pairs:
        for dt in args.knot_dts:
            for batch in args.batches: run_episode(args,pair_id,batch,dt)

if __name__=='__main__': main()
