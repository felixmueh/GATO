"""Duration/discretization study with separate coarse and continuous dynamics.

This new objective dt-normalizes running terms. Terminal position/cylinder
weights stay fixed; the endpoint velocity sample belongs to the dt-weighted
running quadrature, not an independent terminal-velocity penalty.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import platform
import scipy
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares, minimize
from tiago_probe import ROOT, fk, joint_seeds, pack, unpack, replay_cuda, json_safe, pinocchio_one_step_parity

MATRIX = [(16,.45),(32,.45),(32,.90),(64,.90),(64,1.80),(128,1.80)]
TASKS = [(0.,.12,.025,.008,0.), (.30,.22,.035,-.010,.015), (-.25,.30,.045,.012,.03)]


def task(model, index):
    angle, travel, radius, offset, height = TASKS[index]
    q0=np.array([-.39,-1.73,-.38,-2.35,0.,-1.21,.04])
    data=model.createData(); start=fk(model,data,q0)
    direction=np.array([np.cos(angle),np.sin(angle)])
    goal=start+np.r_[travel*direction,height]
    ik=least_squares(lambda q:np.r_[100*(fk(model,data,q)-goal),.01*(q-q0)],q0,
        bounds=(model.lowerPositionLimit+.03,model.upperPositionLimit-.03),max_nfev=300)
    center=.5*(start[:2]+goal[:2])+offset*np.array([-direction[1],direction[0]])
    return dict(index=index,x0=np.r_[q0,np.zeros(7)],qgoal=ik.x,goal=goal,center=center,
                radius=radius,start=start,ik_error=float(np.linalg.norm(fk(model,data,ik.x)-goal)))


def weights(args):
    scale=args.dt/.03
    return dict(position=2.*scale,velocity=.01*scale,effort=.0001*scale,
                obstacle=1.*scale,terminal=200.,terminal_obstacle=1.)


def make_solver(args,batch):
    module=importlib.import_module(f'bsqp.bsqpN{args.knots}_tiago_right_multimodal')
    w=weights(args)
    solver=getattr(module,f'BSQP_{batch}_float')(args.dt,args.iters,args.kkt,getattr(args,'pcg_iters',1000),args.pcg,
        1.,10.,w['position'],w['velocity'],w['effort'],w['terminal'],w['obstacle'],
        w['terminal_obstacle'],0.,0.,0.,args.rho)
    solver.set_f_ext_batch(np.zeros((batch,6),np.float32));solver.set_rho_adaptation(False)
    return solver,module


def objective(x,u,xyz,scene,args):
    if not (np.isfinite(x).all() and np.isfinite(u).all() and np.isfinite(xyz).all()):
        return dict(total=float('inf'))
    w=weights(args);e=np.sum((xyz-scene['goal'])**2,axis=1)
    radial=np.sum((xyz[:,:2]-scene['center'])**2,axis=1)
    residual=np.maximum(1-radial/(scene['radius']+args.margin)**2,0.)
    parts=dict(position=.5*w['position']*np.sum(e[:-1]),terminal=.5*w['terminal']*e[-1],
        velocity=.5*w['velocity']*np.sum(x[:,7:]**2),effort=.5*w['effort']*np.sum(u**2),
        obstacle=.5*w['obstacle']*np.sum(residual[:-1]**2),
        terminal_obstacle=.5*w['terminal_obstacle']*residual[-1]**2)
    return {**{k:float(v) for k,v in parts.items()},'total':float(sum(parts.values()))}


def evaluate(model,x,u,scene,args,dense=None):
    finite=bool(np.isfinite(x).all() and np.isfinite(u).all())
    if not finite:
        return dict(finite=False,feasible=False,settled=False,terminal=float('inf'),
            terminal_speed=float('inf'),clearance=float('-inf'),joint_violation=float('inf'),
            velocity_ratio=float('inf'),effort_ratio=float('inf'),objective=dict(total=float('inf'))),np.full((len(x),3),np.nan)
    data=model.createData();xyz=np.array([fk(model,data,q) for q in x[:,:7]])
    if dense is None:
        ts=np.linspace(0,args.dt,17)[:,None]
        dense=np.concatenate([np.c_[x[k,:7]+ts*x[k,7:]+.5*ts**2*(x[k+1,7:]-x[k,7:])/args.dt,
            x[k,7:]+ts*(x[k+1,7:]-x[k,7:])/args.dt] for k in range(len(x)-1)])
    dense_xyz=np.array([fk(model,data,q) for q in dense[:,:7]])
    clearance=float(np.min(np.linalg.norm(dense_xyz[:,:2]-scene['center'],axis=1))-scene['radius'])
    q,v=dense[:,:7],dense[:,7:]
    joint=max(float(np.max(model.lowerPositionLimit-q)),float(np.max(q-model.upperPositionLimit)),0.)
    velocity=float(np.max(np.abs(v)/model.velocityLimit));effort=float(np.max(np.abs(u)/model.effortLimit))
    jac=pin.computeFrameJacobian(model,data,x[-1,:7],model.getFrameId('arm_right_tool_link'),pin.LOCAL_WORLD_ALIGNED)
    speed=float(np.linalg.norm(jac[:3]@x[-1,7:]));terminal=float(np.linalg.norm(xyz[-1]-scene['goal']))
    winding=float(np.diff(np.unwrap(np.arctan2(xyz[:,1]-scene['center'][1],xyz[:,0]-scene['center'][0]))).sum())
    physical=clearance>0 and joint<1e-5 and velocity<=1 and effort<=1
    return dict(finite=True,feasible=bool(physical and terminal<.02),physical=bool(physical),
        settled=bool(terminal<.02 and speed<.05),terminal=terminal,terminal_speed=speed,
        clearance=clearance,margin_clearance=clearance-args.margin,joint_violation=joint,
        velocity_ratio=velocity,effort_ratio=effort,winding=winding,
        objective=objective(x,u,xyz,scene,args)),xyz


def rk4(model,data,x,u,h):
    def f(z):
        return np.r_[z[7:],pin.aba(model,data,z[:7],z[7:],u).copy()]
    a=f(x);b=f(x+.5*h*a);c=f(x+.5*h*b);d=f(x+h*c)
    return x+h*(a+2*b+2*c+d)/6


def fine_rollout(model,x0,controls,dt,max_step,stop_time=None):
    """Independent float64 continuous ODE with exact ZOH torque switch times.

    The final control interval is truncated at stop_time, so the physical MPC
    clock does not require dt to divide the controller interval.
    """
    x=np.array(x0,float);data=model.createData();dense=[x.copy()];times=[0.];nodes=[x.copy()]
    t=0.;failure=None
    end=len(controls)*dt if stop_time is None else min(stop_time,len(controls)*dt)
    for k,control in enumerate(controls):
        remaining=min((k+1)*dt,end)-t
        if remaining<=1e-12:break
        steps=max(1,int(np.ceil(remaining/max_step)));h=remaining/steps
        for _ in range(steps):
            if not (np.isfinite(x).all() and np.max(np.abs(x[:7]))<1e3 and np.max(np.abs(x[7:]))<1e4):
                failure='nonfinite or integration safety bound (q=1000 rad, v=10000 rad/s)';break
            with np.errstate(over='ignore',invalid='ignore'):
                x=rk4(model,data,x,np.asarray(control,float),h)
            t+=h;dense.append(x.copy());times.append(t)
        nodes.append(x.copy())
        if failure:break
        if t>=end-1e-12:break
    if failure or not np.isfinite(x).all():
        failure=failure or 'nonfinite integration result'
        missing=len(controls)+1-len(nodes)
        if stop_time is None and missing>0:nodes.extend([np.full(14,np.nan)]*missing)
    return np.asarray(nodes),np.asarray(dense),np.asarray(times),failure


def differences(a,b):
    if a.shape!=b.shape or not (np.isfinite(a).all() and np.isfinite(b).all()):
        return dict(q=float('inf'),v=float('inf'),all=float('inf'))
    return dict(q=float(np.max(np.abs(a[:,:7]-b[:,:7]))),v=float(np.max(np.abs(a[:,7:]-b[:,7:]))),all=float(np.max(np.abs(a-b))))


def solve(args,model,scene,batch,seed,inherited=None,inherited_states=None):
    solver,module=make_solver(args,batch)
    x,u=joint_seeds(model,scene['x0'],scene['qgoal'],args.knots,args.dt,batch,seed,args.amplitude)
    if inherited is not None:
        u[0]=inherited
        # Keep a fresh nominal in lane one; warm lane zero is identical in B1/B16.
        if batch>1:
            nx,nu=joint_seeds(model,scene['x0'],scene['qgoal'],args.knots,args.dt,1,seed,args.amplitude)
            x[1],u[1]=nx[0],nu[0]
        x[0]=inherited_states if inherited_states is not None else replay_cuda(solver,scene['x0'],u.astype(np.float32),args.dt)[0]
    u=u.astype(np.float32);initial=pack(x,u)
    inputs_root=getattr(args,'inputs_root',None)
    if inputs_root and inherited is None:
        name=f'N{args.knots}_T{args.duration:g}_task{scene["index"]}'
        source=np.load(Path(inputs_root)/name/f'b{batch}.npz')
        initial=np.asarray(source['initial_xu'],np.float32).copy()
        x,u=unpack(initial,args.knots)
        if initial.shape != (batch,(args.knots-1)*21+14) or not np.array_equal(x[:,0],np.tile(scene['x0'],(batch,1)).astype(np.float32)):
            raise ValueError('Stored initialization has incompatible dimensions or initial state')
    ref=np.tile(np.r_[scene['goal'],scene['center'],scene['radius']+args.margin],(batch,args.knots)).astype(np.float32)
    z=initial.copy();telemetry=[];outputs=[]
    wall=time.monotonic()
    for _ in range(args.passes):
        result=solver.solve(z,args.dt,np.tile(scene['x0'],(batch,1)).astype(np.float32),ref)
        z=np.asarray(result['XU']).copy();outputs.append(z)
        status=np.asarray(result['pcg_status']);alpha=np.asarray(result['ls_step_size'])
        telemetry.append(dict(solve_ms=float(result['sqp_time_us'])/1000,
            sqp_iters=np.asarray(result['sqp_iters']).tolist(),kkt=np.asarray(result['kkt_converged']).tolist(),
            pcg_cap=np.sum(status==0,axis=0).tolist(),pcg_breakdown=np.sum(status==2,axis=0).tolist(),
            pcg_max=int(np.max(result['pcg_iters'])),accepted=np.sum(alpha>0,axis=0).tolist(),
            rejected=np.sum(alpha<0,axis=0).tolist(),initial_merit=np.asarray(result['initial_merit']).tolist(),
            final_merit=np.asarray(result['final_merit']).tolist()))
    solve_wall=time.monotonic()-wall
    planned,controls=unpack(z,args.knots)
    replay=replay_cuda(solver,scene['x0'],controls.copy(),args.dt)
    seed_replay=replay_cuda(solver,scene['x0'],u.copy(),args.dt)
    rows=[];xyz=[];fine_nodes=[];fine_xyz=[]
    for b in range(batch):
        row,tool=evaluate(model,replay[b],controls[b],scene,args)
        before,_=evaluate(model,seed_replay[b],u[b],scene,args)
        fn,dense,times,failure=fine_rollout(model,scene['x0'],controls[b],args.dt,args.fine_step)
        fm,ftool=evaluate(model,fn,controls[b],scene,args,dense=dense) if not failure else (dict(finite=False,feasible=False,settled=False,objective=dict(total=float('inf')),failure=failure),np.full((args.knots,3),np.nan))
        row=dict(coarse=row,fine=fm,before=before,planned_replay=differences(planned[b],replay[b]),
                 coarse_fine=differences(replay[b],fn))
        rows.append(row);xyz.append(tool);fine_nodes.append(fn);fine_xyz.append(ftool)
    def rank(i,domain):
        r=rows[i][domain];return (not r['feasible'],r['objective']['total'])
    winner=min(range(batch),key=lambda i:rank(i,'coarse'))
    fine_winner=min(range(batch),key=lambda i:rank(i,'fine'))
    # Refine the independent simulator for both relevant selections.
    checks={}
    for b in sorted({winner,fine_winner}):
        half,hd,ht,hfail=fine_rollout(model,scene['x0'],controls[b],args.dt,args.fine_step/2)
        checks[str(b)]=dict(step_halving=differences(fine_nodes[b],half),failure=hfail)
    record=dict(batch=batch,seed=seed,winner=winner,fine_winner=fine_winner,lanes=rows,
        telemetry=telemetry,solve_wall_seconds=solve_wall,solve_ms=sum(t['solve_ms'] for t in telemetry),
        fine_refinement=checks,one_step_parity=pinocchio_one_step_parity(model,replay,controls,args.dt),
        module=str(module.__file__))
    arrays=dict(initial_xu=initial,reference=ref,initial_state=np.asarray(scene['x0'],np.float32),
        planned=planned,replay=replay,controls=controls,xyz=np.asarray(xyz),
        fine=np.asarray(fine_nodes),fine_xyz=np.asarray(fine_xyz),seed_controls=u,seed_replay=seed_replay,
        solve_outputs=np.asarray(outputs))
    return record,arrays


class Reference:
    """Same dt-normalized objective, acceleration coordinates and RNEA derivatives."""
    def __init__(self,model,scene,args):
        self.model,self.data,self.scene,self.args=model,model.createData(),scene,args
        k,j=np.arange(args.knots)[:,None],np.arange(args.knots-1)[None]
        self.Q=np.where(k>j,args.dt**2*(k-j-.5),0.);self.V=np.where(k>j,args.dt,0.)
        self.base=scene['x0'][:7]+np.arange(args.knots)[:,None]*args.dt*scene['x0'][7:]
    def rollout(self,flat):
        a=flat.reshape(-1,7);q=self.base+self.Q@a;v=self.scene['x0'][7:]+self.V@a
        u=np.array([pin.rnea(self.model,self.data,qi,vi,ai).copy() for qi,vi,ai in zip(q[:-1],v[:-1],a)])
        return np.c_[q,v],u
    def __call__(self,flat):
        a=flat.reshape(-1,7);x,u=self.rollout(flat);w=weights(self.args);q,v=x[:,:7],x[:,7:]
        dq=np.zeros_like(q);dv=w['velocity']*v;da=np.zeros_like(a)
        cost=.5*w['velocity']*np.sum(v*v)+.5*w['effort']*np.sum(u*u)
        for k in range(len(x)):
            xyz=fk(self.model,self.data,q[k]);jac=pin.computeFrameJacobian(self.model,self.data,q[k],self.model.getFrameId('arm_right_tool_link'),pin.LOCAL_WORLD_ALIGNED)[:3]
            weight=w['terminal'] if k==len(x)-1 else w['position'];error=xyz-self.scene['goal']
            cost+=.5*weight*(error@error);dxyz=weight*error
            delta=xyz[:2]-self.scene['center'];radius=self.scene['radius']+self.args.margin
            residual=max(1-delta@delta/radius**2,0.);ow=w['terminal_obstacle'] if k==len(x)-1 else w['obstacle']
            cost+=.5*ow*residual**2;dxyz[:2]-=2*ow*residual*delta/radius**2;dq[k]+=jac.T@dxyz
            if k<len(a):
                tq,tv,ta=pin.computeRNEADerivatives(self.model,self.data,q[k],v[k],a[k])
                dq[k]+=w['effort']*tq.T@u[k];dv[k]+=w['effort']*tv.T@u[k];da[k]+=w['effort']*ta.T@u[k]
        return float(cost),(self.Q.T@dq+self.V.T@dv+da).ravel()


def reference(args,model,scene,arrays,lane):
    obj=Reference(model,scene,args)
    x=arrays['replay'][lane]
    if not np.isfinite(x).all():return dict(failure='nonfinite starting trajectory'),{}
    initial=np.diff(x[:,7:],axis=0).ravel()/args.dt
    value,grad=obj(initial);direction=np.random.default_rng(921).normal(size=len(initial));direction/=np.linalg.norm(direction)
    fd=(obj(initial+1e-4*direction)[0]-obj(initial-1e-4*direction)[0])/2e-4
    wall=time.monotonic();r=minimize(obj,initial,jac=True,method='L-BFGS-B',options=dict(maxiter=args.reference_iters,gtol=1e-8,ftol=1e-13,maxls=50,maxcor=30))
    xx,uu=obj.rollout(r.x);coarse,xyz=evaluate(model,xx,uu,scene,args)
    fine,dense,times,failure=fine_rollout(model,scene['x0'],uu,args.dt,args.fine_step)
    fm,_=evaluate(model,fine,uu,scene,args,dense=dense) if not failure else (dict(finite=False,feasible=False,failure=failure),None)
    return dict(initial_cost=value,final_cost=float(r.fun),gradient_inf=float(np.max(np.abs(r.jac))),
        derivative_error=float(abs(fd-grad@direction)),success=bool(r.success),message=str(r.message),
        iterations=int(r.nit),wall_seconds=time.monotonic()-wall,coarse=coarse,fine=fm,
        coarse_fine=differences(xx,fine)),dict(states=xx,controls=uu,xyz=xyz,fine=fine)


def integration_probe(args,model,scene):
    solver,_=make_solver(args,1)
    x,u=joint_seeds(model,scene['x0'],scene['qgoal'],args.knots,args.dt,1,args.seed,args.amplitude)
    u=u.astype(np.float32);coarse=replay_cuda(solver,scene['x0'],u,args.dt)[0]
    fn,dense,times,fail=fine_rollout(model,scene['x0'],u[0],args.dt,args.fine_step)
    half,_,_,half_fail=fine_rollout(model,scene['x0'],u[0],args.dt,args.fine_step/2)
    cm,_=evaluate(model,coarse,u[0],scene,args);fm,_=evaluate(model,fn,u[0],scene,args,dense=dense) if not fail else (dict(finite=False,feasible=False,failure=fail),None)
    return dict(constructed_cuda=differences(x[0],coarse),coarse_fine=differences(coarse,fn),
        fine_halving=differences(fn,half),fine_failure=fail,half_failure=half_fail,
        coarse=cm,fine=fm,parity=pinocchio_one_step_parity(model,coarse[None],u,args.dt)[0])


def scene_json(scene):
    return {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in scene.items()}


def export_inputs(root):
    """Export exact inputs, reconstructing older artifacts only with checks."""
    model=pin.buildModelFromUrdf(str(ROOT/'gato/dynamics/tiago_right/tiago_right_arm.urdf'))
    manifest=[]
    for path in sorted(root.glob('*/summary.json')):
        record=json.loads(path.read_text())
        if 'branches' not in record:continue
        cfg=SimpleNamespace(**record['config'])
        scene={k:np.asarray(v) if isinstance(v,list) else v for k,v in record['scene'].items()}
        item=dict(cell=record['cell'],scene=record['scene'],batches={})
        for batch,branch in record['branches'].items():
            target=path.parent/f'b{batch}.npz';arrays=dict(np.load(target))
            reconstructed='initial_xu' not in arrays
            if reconstructed:
                x,u=joint_seeds(model,scene['x0'],scene['qgoal'],cfg.knots,cfg.dt,int(batch),branch['seed'],cfg.amplitude)
                if not np.array_equal(u.astype(np.float32),arrays['seed_controls']):
                    raise ValueError(f'Cannot exactly reconstruct original controls: {target}')
                arrays['initial_xu']=pack(x,u)
                arrays['initial_state']=np.asarray(scene['x0'],np.float32)
                arrays['reference']=np.tile(np.r_[scene['goal'],scene['center'],scene['radius']+cfg.margin],(int(batch),cfg.knots)).astype(np.float32)
                np.savez_compressed(target,**arrays)
            item['batches'][batch]=dict(reconstructed=reconstructed,
                initial_xu_sha256=hashlib.sha256(np.ascontiguousarray(arrays['initial_xu']).tobytes()).hexdigest(),
                reference_sha256=hashlib.sha256(np.ascontiguousarray(arrays['reference']).tobytes()).hexdigest(),
                shape=list(arrays['initial_xu'].shape),dtype=str(arrays['initial_xu'].dtype))
        manifest.append(item)
    (root/'input_manifest.json').write_text(json.dumps(dict(cells=manifest,machine=machine_metadata()),indent=2))
    print(root/'input_manifest.json')


def machine_metadata():
    def output(command):
        try:return subprocess.check_output(command,text=True,stderr=subprocess.STDOUT).strip()
        except (OSError,subprocess.CalledProcessError) as exc:return str(exc)
    return dict(platform=platform.platform(),python=sys.version,executable=sys.executable,
        numpy=np.__version__,scipy=scipy.__version__,pinocchio=pin.__version__,
        gpu=output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader']),
        cuda_compiler=output(['nvcc','--version']))


def config_json(args):
    return {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}


def run(args):
    args.output.mkdir(parents=True,exist_ok=True)
    model=pin.buildModelFromUrdf(str(ROOT/'gato/dynamics/tiago_right/tiago_right_arm.urdf'))
    cells=MATRIX if args.cells is None else [(int(c.split(':')[0]),float(c.split(':')[1])) for c in args.cells]
    index_file=args.output/'index.json';records=[]
    if not args.worker and len(cells)>1:
        # pybind registers BSQP C++ template names globally; different knot
        # modules therefore need separate interpreters.
        for n,duration in cells:
            command=[sys.executable,str(Path(__file__).resolve()),'--worker','--cells',f'{n}:{duration}',
                '--tasks',*[str(v) for v in args.tasks],'--batches',*[str(v) for v in args.batches],
                '--output',str(args.output.resolve())]
            for key in ('seed','iters','passes','rho','pcg','kkt','margin','amplitude','fine_step','reference_iters','pcg_iters'):
                command.extend(['--'+key.replace('_','-'),str(getattr(args,key))])
            if args.reference:command.append('--reference')
            if args.resume:command.append('--resume')
            if args.inputs_root:command.extend(['--inputs-root',str(Path(args.inputs_root).resolve())])
            result=subprocess.run(command,cwd=ROOT)
            for tid in args.tasks:
                name=f'N{n}_T{duration:g}_task{tid}'
                path=args.output/name/'summary.json'
                records.append(json.loads(path.read_text()) if path.exists() else dict(cell=name,failure=f'worker exited {result.returncode} without a result'))
            index_file.write_text(json.dumps(json_safe(records),indent=2,allow_nan=False))
        return
    for n,duration in cells:
        for tid in args.tasks:
            cell=SimpleNamespace(**vars(args));cell.knots=n;cell.duration=duration;cell.dt=duration/(n-1)
            name=f'N{n}_T{duration:g}_task{tid}'
            if args.inputs_root:
                saved=json.loads((Path(args.inputs_root)/name/'summary.json').read_text())['scene']
                scene={k:np.asarray(v) if isinstance(v,list) else v for k,v in saved.items()}
            else:scene=task(model,tid)
            dest=args.output/name;dest.mkdir(exist_ok=True)
            if args.resume and (dest/'summary.json').exists():
                saved=json.loads((dest/'summary.json').read_text())
                ignored={'output','resume','worker','inputs_root','cells','tasks','batches','export_inputs_only'}
                if any(saved['config'].get(k,1000 if k=='pcg_iters' else None)!=v for k,v in config_json(cell).items() if k not in ignored):
                    raise ValueError(f'Resume configuration mismatch for {name}')
                if 'failure' not in saved and set(saved.get('branches',{}))>=set(map(str,args.batches)):
                    records.append(saved);continue
            started=time.monotonic();print('START',name,flush=True)
            try:
                probe=integration_probe(cell,model,scene)
                out=dict(cell=name,config=config_json(cell),weights=weights(cell),
                    scene=scene_json(scene),integration=probe,branches={},machine=machine_metadata())
                for batch in args.batches:
                    result,arrays=solve(cell,model,scene,batch,args.seed)
                    out['branches'][str(batch)]=result;np.savez_compressed(dest/f'b{batch}.npz',**arrays)
                    if args.reference:
                        ref,ra=reference(cell,model,scene,arrays,result['winner'])
                        result['cpu_reference']=ref;np.savez_compressed(dest/f'b{batch}_cpu.npz',**ra)
                    chosen=result['lanes'][result['winner']]
                    print(json.dumps(json_safe(dict(cell=name,batch=batch,winner=result['winner'],
                        coarse=chosen['coarse'],fine=chosen['fine'],defect=chosen['planned_replay']))),flush=True)
                module=Path(out['branches'][str(args.batches[0])]['module'])
                paths=[Path(__file__),ROOT/'examples/randomized_multimodal/tiago_probe.py',module,
                    ROOT/'gato/dynamics/tiago_right/tiago_right_arm.urdf',ROOT/'gato/dynamics/tiago_right/tiago_right_plant.cuh']
                out['provenance']=dict(git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
            except Exception as exc:
                out=dict(cell=name,config=config_json(cell),scene=scene_json(scene),failure=repr(exc))
                print('FAIL',name,repr(exc),flush=True)
            out['wall_seconds']=time.monotonic()-started
            (dest/'summary.json').write_text(json.dumps(json_safe(out),indent=2,allow_nan=False));records.append(out)
            index_file.write_text(json.dumps(json_safe(records),indent=2,allow_nan=False))
    index_file.write_text(json.dumps(json_safe(records),indent=2,allow_nan=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cells',nargs='+',help='N:duration pairs; default six-cell duration/grid matrix')
    p.add_argument('--tasks',type=int,nargs='+',default=[0,1,2]);p.add_argument('--batches',type=int,nargs='+',default=[1,16])
    p.add_argument('--seed',type=int,default=1401);p.add_argument('--iters',type=int,default=200);p.add_argument('--passes',type=int,default=2)
    p.add_argument('--pcg-iters',type=int,default=1000)
    p.add_argument('--rho',type=float,default=.01);p.add_argument('--pcg',type=float,default=1e-4);p.add_argument('--kkt',type=float,default=1e-3)
    p.add_argument('--margin',type=float,default=.008);p.add_argument('--amplitude',type=float,default=.1)
    p.add_argument('--fine-step',type=float,default=.001);p.add_argument('--reference',action='store_true');p.add_argument('--reference-iters',type=int,default=1500)
    p.add_argument('--export-inputs-only',action='store_true',help='Export/verify exact task and initial-XU arrays in --output without GPU solves')
    p.add_argument('--inputs-root',type=Path,help='Load exact saved task and initial_xu arrays from a prior output root')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--resume',action='store_true');p.add_argument('--output',type=Path,default=ROOT/'example_artifacts/randomized_multimodal/tiago_horizon')
    options=p.parse_args()
    export_inputs(options.output) if options.export_inputs_only else run(options)
