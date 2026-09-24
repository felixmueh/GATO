"""Run obstacle-blind randomized GATO optimization and receding-horizon trials.

Example: python examples/randomized_multimodal/experiment.py --extension-dir build/pointmass32/python --output /tmp/multimodal.json
"""
import argparse
from dataclasses import asdict, replace
import importlib
import json
from pathlib import Path
import sys
import time
import numpy as np

from model import Config, scene, seeds, rollout, objective, certify, reference, polish, pack, unpack, jsonable


class Gato:
    def __init__(self, cfg, extension_dir=None):
        self.cfg=cfg
        if extension_dir: sys.path.insert(0,str(Path(extension_dir).resolve()))
        self.module=importlib.import_module('bsqpN32_pointmass2d')
        self.solvers={}

    def solver(self,batch):
        if batch not in self.solvers:
            c=self.cfg
            self.solvers[batch]=getattr(self.module,f'BSQP_{batch}_float')(
                c.dt,c.max_sqp_iters,c.kkt_tol,c.max_pcg_iters,c.pcg_tol,1.,c.mu,
                c.q_cost,c.qd_cost,c.u_cost,c.N_cost,c.obstacle_cost,c.obstacle_cost,
                0.,0.,0.,c.rho)
        return self.solvers[batch]

    def solve(self, task, controls):
        c=self.cfg; batch=len(controls)
        initial=rollout(task['x0'],controls,c.dt)
        xu=pack(initial,controls)
        x0=np.repeat(np.asarray(task['x0'],dtype=np.float32)[None],batch,axis=0)
        ref=np.array([*task['goal'],*task['center'],task['radius']+task['buffer'],0.],dtype=np.float32)
        refs=np.tile(ref,(batch,c.knots))
        solver=self.solver(batch)
        solver.reset_dual(); solver.reset_rho()
        start=time.perf_counter()
        result=solver.solve(xu,c.dt,x0,refs)
        elapsed=(time.perf_counter()-start)*1000
        planned, optimized=unpack(result['XU'],c)
        replay=rollout(task['x0'],optimized,c.dt)
        rows=[]
        for lane,u in enumerate(optimized):
            if not np.isfinite(u).all():
                rows.append(dict(cost=1e30,feasible=False,mode='invalid',terminal_error=1e30,min_clearance=-1e30))
                continue
            row=certify(u,task,c); before=certify(controls[lane],task,c)
            row.update(initial_cost=before['cost'],initial_feasible=before['feasible'],
                planned_defect=float(np.max(np.abs(planned[lane]-replay[lane]))),
                control_change=float(np.linalg.norm(u-controls[lane])),
                sqp_iters=int(result['sqp_iters'][lane]),kkt_converged=bool(result['kkt_converged'][lane]),
                initial_merit=float(result['initial_merit'][lane]),final_merit=float(result['final_merit'][lane]),
                accepted_steps=int(np.count_nonzero(np.asarray(result['ls_step_size'])[:,lane]>0)))
            rows.append(row)
        feasible=[i for i,r in enumerate(rows) if r['feasible']]
        winner=min(feasible if feasible else range(batch),key=lambda i:rows[i]['cost'])
        return dict(**rows[winner],selected_lane=int(winner),lanes=rows,solve_ms=elapsed,
                    solver_ms=float(result['sqp_time_us'])/1000,controls=optimized,
                    trajectory=replay[winner],candidate_trajectories=replay,initial_trajectories=initial)


def mpc(gato,task,batch,proposal_seed,ticks=32,execute=2,switch_tick=None,reference_starts=16):
    cfg=gato.cfg; current=dict(task)
    states=[np.array(task['x0'])]; actions=[]; snapshots=[]
    retained=None; minimum_clearance=np.inf; realized_running_cost=0.; changed_decision=None
    direction=task['goal']-task['x0'][:2]; direction=direction/np.linalg.norm(direction)
    normal=np.array([-direction[1],direction[0]])
    reflected=task['center']-2*np.dot(task['center']-task['x0'][:2],normal)*normal
    for tick in range(ticks):
        if switch_tick is not None and tick>=switch_tick: current['center']=reflected
        _, initial=seeds(current['x0'],task['goal'],cfg.knots,cfg.dt,batch,proposal_seed+1009*tick,cfg.acceleration_limit)
        if retained is not None:
            count=min(len(retained),max(1,batch//4))
            initial[:count]=retained[:count]
        answer=gato.solve(current,initial)
        snapshot=dict(tick=tick,executed_steps=len(actions),x0=current['x0'].copy(),center=current['center'].copy(),**{k:v for k,v in answer.items() if k!='controls'})
        if switch_tick is not None and tick==switch_tick and reference_starts:
            independent=reference(current,cfg,reference_starts)
            best=independent['best_known_cost']
            changed_decision=dict(cost=answer['cost'],feasible=answer['feasible'],best_known_cost=best,
                near_best=bool(answer['feasible'] and best is not None and answer['cost']<=best*(1+cfg.near_best_relative)),
                relative_regret=(answer['cost']/best-1) if best else None,mode=answer['mode'],
                reference_lanes=independent['lanes'])
            selected_control=answer['controls'][answer['selected_lane']]
            changed_decision['selected_polish']=polish(selected_control,current,cfg)[0] if np.isfinite(selected_control).all() else None
            snapshot['best_known_cost']=best
        snapshots.append(snapshot)
        selected=answer['controls'][answer['selected_lane']]
        if not np.isfinite(selected).all():
            return dict(scene_id=task['scene_id'],batch_size=batch,proposal_seed=proposal_seed,
                        executed_states=np.asarray(states),executed_controls=np.asarray(actions),snapshots=snapshots,
                        feasible=False,completed=False,failure='all selected controls nonfinite',cost=None)
        executed=rollout(current['x0'],selected[:execute],cfg.dt)
        chunk_cfg=replace(cfg,knots=execute+1)
        chunk=certify(selected[:execute],current,chunk_cfg)
        minimum_clearance=min(minimum_clearance,chunk['min_clearance'])
        for x,u in zip(executed[:-1],selected[:execute]):
            residual=max(0.,1-np.sum((x[:2]-current['center'])**2)/(current['radius']+current['buffer'])**2)
            realized_running_cost+=.5*(cfg.q_cost*np.sum((x[:2]-task['goal'])**2)+cfg.qd_cost*np.sum(x[2:]**2)+cfg.u_cost*np.sum(u**2)+cfg.obstacle_cost*residual**2)
        states.extend(executed[1:]); actions.extend(selected[:execute])
        current['x0']=executed[-1]
        # Keep a small diverse set of past local solutions. This uses trajectory
        # distance, never an obstacle side or winding label.
        order=sorted(range(batch),key=lambda i:(not answer['lanes'][i]['feasible'],answer['lanes'][i]['cost']))
        keep=[answer['selected_lane']]
        for idx in order:
            if all(np.sqrt(np.mean((answer['candidate_trajectories'][idx,:,:2]-answer['candidate_trajectories'][j,:,:2])**2))>.04 for j in keep):
                keep.append(idx)
            if len(keep)>=max(1,batch//4): break
        retained=np.array([np.concatenate((answer['controls'][i,execute:],np.zeros((execute,2))),axis=0) for i in keep])
        if np.linalg.norm(current['x0'][:2]-task['goal'])<cfg.terminal_tolerance and np.linalg.norm(current['x0'][2:])<.08:
            break
    actual_cfg=replace(cfg,knots=len(actions)+1)
    certificate=certify(np.array(actions),task,actual_cfg)
    certificate['min_clearance']=float(minimum_clearance)
    certificate['executed_running_cost']=float(realized_running_cost)
    certificate['cost']=float(realized_running_cost+.5*cfg.N_cost*np.sum((states[-1][:2]-task['goal'])**2)+.5*cfg.qd_cost*np.sum(states[-1][2:]**2))
    certificate['completed']=bool(certificate['terminal_error']<cfg.terminal_tolerance and np.linalg.norm(states[-1][2:])<.08)
    certificate['feasible']=bool(certificate['completed'] and minimum_clearance>=cfg.clearance_tolerance and certificate['max_acceleration']<=cfg.acceleration_limit+1e-5 and certificate['max_velocity']<=cfg.velocity_limit+1e-5)
    certificate['terminal_speed']=float(np.linalg.norm(states[-1][2:]))
    return dict(scene_id=task['scene_id'],batch_size=batch,proposal_seed=proposal_seed,
                executed_states=np.asarray(states),executed_controls=np.asarray(actions),snapshots=snapshots,
                scenario='moving_cylinder' if switch_tick is not None else 'static',switch_tick=switch_tick,
                changed_decision=changed_decision,**certificate)


def campaign(args):
    cfg=replace(Config(),dt=args.dt,max_sqp_iters=args.iterations)
    tasks=[scene(s) for s in args.scene_seeds]
    payload=dict(config=asdict(cfg),scenes=tasks,runs=[],random_single_runs=[],references={},mpc=[],changed_state_runs=[],notes=[
        'Optimum means best-known independently optimized feasible reference, not certified global optimum.',
        'Raw GATO controls are independently replayed and checked for continuous obstacle clearance.',
        'All random seeds are obstacle-blind; offline references are never used as online proposals.'])
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    def save(): output.write_text(json.dumps(jsonable(payload),indent=2)+'\n')
    gato=None if args.cpu_only else Gato(cfg,args.extension_dir)
    for task in tasks:
        if args.reference_starts:
            ref=reference(task,cfg,args.reference_starts)
            payload['references'][str(task['scene_id'])]={k:v for k,v in ref.items() if k!='controls'}
            print('reference',task['scene_id'],ref['best_known_cost'],{r['mode'] for r in ref['lanes'] if r['feasible']},flush=True)
        else: ref={'best_known_cost':None}
        if gato is not None:
            # Warm all instantiated batch shapes before collecting observations.
            _, warm=seeds(task['x0'],task['goal'],cfg.knots,cfg.dt,max(args.batches),args.proposal_seeds[0],cfg.acceleration_limit)
            for batch in args.batches: gato.solve(task,warm[:batch])
            for proposal_seed in args.proposal_seeds:
                _, initial=seeds(task['x0'],task['goal'],cfg.knots,cfg.dt,max(args.batches),proposal_seed,cfg.acceleration_limit)
                for batch in args.batches:
                    row=gato.solve(task,initial[:batch])
                    row.pop('controls')
                    row.update(scene_id=task['scene_id'],proposal_seed=proposal_seed,batch_size=batch,best_known_cost=ref['best_known_cost'])
                    row['near_best']=bool(row['feasible'] and ref['best_known_cost'] is not None and row['cost']<=ref['best_known_cost']*(1+cfg.near_best_relative))
                    payload['runs'].append(row)
                    print('GATO',task['scene_id'],proposal_seed,batch,row['feasible'],round(row['cost'],5),row['mode'],round(row['solve_ms'],2),flush=True)
                if args.random_single:
                    row=gato.solve(task,initial[1:2]); row.pop('controls')
                    row.update(scene_id=task['scene_id'],proposal_seed=proposal_seed,batch_size=1,best_known_cost=ref['best_known_cost'])
                    row['near_best']=bool(row['feasible'] and ref['best_known_cost'] is not None and row['cost']<=ref['best_known_cost']*(1+cfg.near_best_relative))
                    payload['random_single_runs'].append(row)
                    print('random B1',task['scene_id'],proposal_seed,row['feasible'],row['cost'],row['near_best'],flush=True)
            b1_episode=None
            if args.mpc:
                for batch in (1,max(args.batches)):
                    episode=mpc(gato,task,batch,args.proposal_seeds[0],args.mpc_ticks,args.execute,args.switch_tick,args.reference_starts)
                    payload['mpc'].append(episode)
                    if batch==1: b1_episode=episode
                    decision=episode.get('changed_decision')
                    print('MPC',task['scene_id'],batch,episode['feasible'],episode['cost'],
                          {k:decision[k] for k in ('near_best','relative_regret','mode')} if decision else None,flush=True)
            if args.changed_state and b1_episode is not None and args.switch_tick is not None:
                if len(b1_episode['snapshots'])<=args.switch_tick:
                    for proposal_seed in args.proposal_seeds:
                        for batch in args.batches:
                            payload['changed_state_runs'].append(dict(scene_id=task['scene_id'],proposal_seed=proposal_seed,
                                batch_size=batch,feasible=False,near_best=False,cost=None,best_known_cost=None,
                                failure='controller stopped before scheduled change'))
                    save()
                    continue
                snapshot=b1_episode['snapshots'][args.switch_tick]
                fixed=dict(task,x0=np.array(snapshot['x0']),center=np.array(snapshot['center']))
                # Reconstruct exact controls stored in the dynamically exact seed
                # path; unlike the optimized path this came directly from rollout.
                warm=np.diff(np.array(snapshot['initial_trajectories'])[0,:,2:],axis=0)/cfg.dt
                best=snapshot.get('best_known_cost')
                for proposal_seed in args.proposal_seeds:
                    _,initial=seeds(fixed['x0'],fixed['goal'],cfg.knots,cfg.dt,max(args.batches),proposal_seed,cfg.acceleration_limit)
                    initial[0]=warm
                    for batch in args.batches:
                        row=gato.solve(fixed,initial[:batch])
                        selected_control=row['controls'][row['selected_lane']]
                        row['selected_polish']=polish(selected_control,fixed,cfg)[0] if np.isfinite(selected_control).all() else None
                        row.pop('controls')
                        row.update(scene_id=task['scene_id'],proposal_seed=proposal_seed,batch_size=batch,
                                   x0=fixed['x0'],center=fixed['center'],best_known_cost=best)
                        row['near_best']=bool(row['feasible'] and best is not None and row['cost']<=best*(1+cfg.near_best_relative))
                        row['relative_regret']=row['cost']/best-1 if best else None
                        payload['changed_state_runs'].append(row)
                        print('same-state change',task['scene_id'],proposal_seed,batch,row['feasible'],row['near_best'],row['relative_regret'],flush=True)
        save()
    if args.plot:
        from plotting import plot_campaign
        plot_campaign(jsonable(payload),output.with_suffix(''))
    return payload


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--extension-dir')
    p.add_argument('--output',default='/tmp/randomized-multimodal-development.json')
    p.add_argument('--scene-seeds',nargs='+',type=int,default=[24092400,24092401,24092402])
    p.add_argument('--proposal-seeds',nargs='+',type=int,default=[901,902,903])
    p.add_argument('--batches',nargs='+',type=int,default=[1,4,8,16])
    p.add_argument('--dt',type=float,default=.08)
    p.add_argument('--iterations',type=int,default=80)
    p.add_argument('--reference-starts',type=int,default=32)
    p.add_argument('--cpu-only',action='store_true')
    p.add_argument('--mpc',action='store_true')
    p.add_argument('--random-single',action='store_true')
    p.add_argument('--changed-state',action='store_true')
    p.add_argument('--switch-tick',type=int,default=None)
    p.add_argument('--mpc-ticks',type=int,default=32)
    p.add_argument('--execute',type=int,default=2)
    p.add_argument('--plot',action='store_true')
    campaign(p.parse_args())


if __name__=='__main__': main()
