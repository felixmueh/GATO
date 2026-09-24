"""Applied-prefix and fixed-clock MPC validation for the horizon study.

Independent ABA/RK4 executes ZOH torques, splitting exactly at both optimizer
control switches and .06 s controller events. Running-cost quadrature uses each
actual simulation interval; no partial control is charged a full optimizer dt.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pinocchio as pin
from tiago_horizon import (ROOT,task,solve,evaluate,fine_rollout,differences,fk,json_safe,machine_metadata)


def interpolate_coarse(states,times,dt):
    k=np.minimum(np.floor(np.asarray(times)/dt+1e-10).astype(int),len(states)-2)
    tau=(np.asarray(times)-k*dt)[:,None]
    a=(states[k+1,7:]-states[k,7:])/dt
    return np.c_[states[k,:7]+tau*states[k,7:]+.5*tau*tau*a,states[k,7:]+tau*a]


def interval_controls(controls,times,dt):
    idx=np.minimum(np.floor(np.asarray(times[:-1])/dt+1e-10).astype(int),len(controls)-1)
    return controls[idx]


def physical_summary(model,states,controls,scene,args):
    # evaluate's discrete objective is intentionally discarded for a variable-
    # interval simulated trajectory; physical checks use every dense sample.
    row,xyz=evaluate(model,states,controls,scene,args,dense=states)
    row.pop('objective',None)
    return row,xyz


def integrated_objective(states,controls,times,xyz,scene,args):
    h=np.diff(times);e=np.sum((xyz-scene['goal'])**2,axis=1)
    r=np.maximum(1-np.sum((xyz[:,:2]-scene['center'])**2,axis=1)/(scene['radius']+args.margin)**2,0.)
    # Trapezoidal state quadrature, exact interval effort integral under ZOH.
    velocity=np.sum(states[:,7:]**2,axis=1)
    parts=dict(position=float(.5*np.sum(h*(e[:-1]+e[1:]))/.03),
        velocity=float(.25*.01/.03*np.sum(h*(velocity[:-1]+velocity[1:]))),
        effort=float(.5*.0001/.03*np.sum(h*np.sum(controls**2,axis=1))),
        obstacle=float(.25/.03*np.sum(h*(r[:-1]**2+r[1:]**2))),
        terminal=float(100*e[-1]),terminal_obstacle=float(.5*r[-1]**2))
    return dict(total=sum(parts.values()),**parts)


def prefix(model,scene,args,controls,coarse,clock=.06):
    # Align the validation initial state exactly to CUDA, rather than mixing
    # original float64 literals with CUDA's float32 input representation.
    initial=np.asarray(coarse[0],float)
    nodes,dense,times,failure=fine_rollout(model,initial,controls,args.dt,args.fine_step,clock)
    half,hd,ht,hfail=fine_rollout(model,initial,controls,args.dt,args.fine_step/2,clock)
    if failure:return dict(failure=failure),{}
    per_interval=interval_controls(controls,times,args.dt)
    row,xyz=physical_summary(model,dense,per_interval,scene,args)
    predicted=interpolate_coarse(coarse,times,args.dt)
    predicted_xyz=np.array([fk(model,model.createData(),q) for q in predicted[:,:7]])
    row.update(coarse_fine=differences(predicted,dense),
        tool_error_max=float(np.max(np.linalg.norm(predicted_xyz-xyz,axis=1))),
        step_halving_terminal=differences(dense[-1:],hd[-1:]),
        fine_halving_failure=hfail,clock=float(times[-1]),
        integrated_objective=integrated_objective(dense,per_interval,times,xyz,scene,args))
    return row,dict(states=dense,controls=per_interval,times=times,xyz=xyz,predicted=predicted)


def validate_saved(args):
    model=pin.buildModelFromUrdf(str(ROOT/'gato/dynamics/tiago_right/tiago_right_arm.urdf'))
    records=[]
    for file in sorted(args.source.glob('*/summary.json')):
        saved=json.loads(file.read_text())
        if 'branches' not in saved:continue
        cfg=SimpleNamespace(**saved['config']);scene={k:np.asarray(v) if isinstance(v,list) else v for k,v in saved['scene'].items()}
        result=dict(cell=saved['cell'],branches={})
        for batch,branch in saved['branches'].items():
            a=np.load(file.parent/f'b{batch}.npz');winner=branch['winner'];u=a['controls'][winner];coarse=a['replay'][winner]
            pr,arrays=prefix(model,scene,cfg,u,coarse,args.clock)
            fn,dense,times,failure=fine_rollout(model,np.asarray(coarse[0],float),u,cfg.dt,cfg.fine_step)
            half,_,_,hfail=fine_rollout(model,np.asarray(coarse[0],float),u,cfg.dt,cfg.fine_step/2)
            fine,_=evaluate(model,fn,u,scene,cfg,dense=dense) if not failure else (dict(failure=failure,feasible=False),None)
            result['branches'][batch]=dict(prefix=pr,aligned_fine=fine,
                aligned_coarse_fine=differences(coarse,fn),aligned_fine_halving=differences(fn,half),
                original_initial_rounding=float(np.max(np.abs(scene['x0']-coarse[0]))))
            np.savez_compressed(file.parent/f'b{batch}_applied_prefix.npz',**arrays)
        records.append(result)
    path=args.source/'prefix_validation.json';path.write_text(json.dumps(json_safe(records),indent=2,allow_nan=False))
    print(path)


def shifted_controls(controls,dt,clock,tail):
    """Interval-average old ZOH torques on the shifted optimizer grid.

    This is a warm initialization only. Actual execution retains exact switch
    times, and the next solve starts from the measured fine-simulation state.
    """
    count=len(controls);out=[];duration=count*dt
    for i in range(count):
        left=clock+i*dt;right=left+dt;value=np.zeros(7);t=left
        while t<right-1e-12:
            k=int(np.floor(t/dt+1e-10))
            end=min(right,(k+1)*dt) if k<count else right
            value+=(end-t)*(controls[k] if k<count else tail);t=end
        out.append(value/dt)
    return np.asarray(out)


def mpc(args):
    source=args.source/f'N{args.knots}_T{args.duration:g}_task{args.task}'/'summary.json'
    raw=json.loads(source.read_text());cfg=SimpleNamespace(**raw['config'])
    cfg.reference=False;cfg.inputs_root=None
    if args.rho is not None:cfg.rho=args.rho
    if args.passes is not None:cfg.passes=args.passes
    if args.iters is not None:cfg.iters=args.iters
    if args.pcg_iters is not None:cfg.pcg_iters=args.pcg_iters
    model=pin.buildModelFromUrdf(str(ROOT/'gato/dynamics/tiago_right/tiago_right_arm.urdf'))
    scene={k:np.asarray(v) if isinstance(v,list) else v for k,v in raw['scene'].items()}
    args.output.mkdir(parents=True,exist_ok=True)
    output=dict(source=str(source),clock=args.clock,steps=args.steps,config=vars(cfg),branches={},
        selection='coarse executable feasibility then normalized objective; no rejection solely for full-horizon fine drift',
        scope='independent continuous ABA/RK4 simulation with fixed physical controller clock, not hardware or whole-body safety',
        initialization=args.initialization,warm_state_source=args.warm_state_source,machine=machine_metadata())
    output['config']['output']=str(output['config']['output'])
    data=model.createData();tail=pin.rnea(model,data,scene['qgoal'],np.zeros(7),np.zeros(7)).copy()
    for batch in args.batches:
        current=scene['x0'].copy();warm=None;warm_states=None;all_x=[current.copy()];all_u=[];all_t=[0.];stages=[];failure=None
        for step in range(args.steps):
            local={**scene,'x0':current.copy()}
            if args.initialization=='cold':warm,warm_states=None,None
            record,arrays=solve(cfg,model,local,batch,cfg.seed+7919*step,warm,warm_states)
            winner=record['winner'];controls=arrays['controls'][winner];coarse=arrays['replay'][winner]
            # Execute from actual continuous state; only the solver's measurement
            # is float32. Record this tiny quantization separately.
            nodes,dense,times,error=fine_rollout(model,current,controls,cfg.dt,cfg.fine_step,args.clock)
            hu=interval_controls(controls,times,cfg.dt)
            if error:
                failure=error;stages.append(dict(step=step,solve=record,failure=error));break
            physical,xyz=physical_summary(model,dense,hu,local,cfg)
            predicted=interpolate_coarse(coarse,times,cfg.dt)
            record['applied_prefix']=dict(physical=physical,coarse_fine=differences(predicted,dense),
                initial_quantization=float(np.max(np.abs(current-coarse[0]))),
                tool_error_max=float(np.max(np.linalg.norm(np.array([fk(model,data,q) for q in predicted[:,:7]])-xyz,axis=1))))
            stages.append(record)
            np.savez_compressed(args.output/f'b{batch}_stage{step}.npz',**arrays,
                executed=dense,executed_times=times,executed_controls=hu)
            offset=all_t[-1];all_x.extend(dense[1:]);all_u.extend(hu);all_t.extend(offset+times[1:]);current=dense[-1].copy()
            print(json.dumps(json_safe(dict(batch=batch,step=step,sim_time=all_t[-1],
                prefix_physical=physical['physical'],terminal=physical['terminal'],speed=physical['terminal_speed'],
                horizon_coarse=record['lanes'][winner]['coarse']['feasible'],
                horizon_fine=record['lanes'][winner]['fine']['feasible']))),flush=True)
            if not physical['physical']:
                failure='physical bound or collision violated during applied prefix';break
            last_q=np.asarray(arrays['planned'][winner,-1,:7],float)
            tail=pin.rnea(model,data,last_q,np.zeros(7),np.zeros(7)).copy()
            warm=shifted_controls(controls,cfg.dt,args.clock,tail)
            if args.warm_state_source=='planned':
                old_times=np.arange(cfg.knots)*cfg.dt
                new_times=args.clock+old_times
                warm_states=np.column_stack([np.interp(new_times,old_times,arrays['planned'][winner,:,j]) for j in range(14)])
                warm_states[new_times>=old_times[-1],7:]=0.
                warm_states[0]=current
            else:warm_states=None
        xx,uu,tt=np.asarray(all_x),np.asarray(all_u),np.asarray(all_t)
        if len(uu):
            metrics,xyz=physical_summary(model,xx,uu,scene,cfg)
            metrics['objective']=integrated_objective(xx,uu,tt,xyz,scene,cfg)
        else:metrics,xyz=dict(physical=False,settled=False),np.full((1,3),np.nan)
        output['branches'][str(batch)]=dict(stages=stages,executed=metrics,failure=failure,
            simulated_seconds=float(tt[-1]),complete=bool(not failure and len(stages)==args.steps and metrics['settled']))
        np.savez_compressed(args.output/f'b{batch}_executed.npz',states=xx,controls=uu,times=tt,xyz=xyz)
        (args.output/'summary.json').write_text(json.dumps(json_safe(output),indent=2,allow_nan=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'example_artifacts/randomized_multimodal/tiago_horizon')
    p.add_argument('--validate-saved',action='store_true');p.add_argument('--clock',type=float,default=.06)
    p.add_argument('--knots',type=int,default=64);p.add_argument('--duration',type=float,default=.9);p.add_argument('--task',type=int,default=1)
    p.add_argument('--batches',type=int,nargs='+',default=[1,16]);p.add_argument('--steps',type=int,default=20)
    p.add_argument('--rho',type=float);p.add_argument('--passes',type=int)
    p.add_argument('--iters',type=int);p.add_argument('--pcg-iters',type=int)
    p.add_argument('--initialization',choices=['warm','cold'],default='warm')
    p.add_argument('--warm-state-source',choices=['planned','replay'],default='planned')
    p.add_argument('--output',type=Path,default=ROOT/'example_artifacts/randomized_multimodal/tiago_horizon_mpc')
    args=p.parse_args()
    validate_saved(args) if args.validate_saved else mpc(args)
