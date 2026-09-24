"""Per-lane CUDA batch-equivalence audit, separate from scene-quality claims.

Export portable inputs with --prepare-only; pass --input inputs.npz to reuse
the exact float32 arrays on another GPU. Use one interpreter per knot count.
--quick is an arithmetic screen, not acceptance of the full production budget.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import numpy as np
import pinocchio as pin

from tiago_probe import ROOT, joint_seeds, pack, unpack, replay_cuda, json_safe
from tiago_horizon import task, evaluate

TOLERANCES = dict(packed_atol=1e-4, packed_rtol=2e-5,
    objective_atol=1e-6, objective_rtol=1e-4,
    material_objective_rtol=1e-3, material_q_atol=1e-3,
    material_v_atol=1e-2, material_control_atol=.05,
    material_terminal_atol=5e-4)
TELEMETRY = ('sqp_iters','kkt_converged','pcg_iters','pcg_status',
             'ls_step_size','ls_min_merit','initial_merit','final_merit')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(values):
    digest=hashlib.sha256()
    for name,array in sorted(values.items()):
        a=np.ascontiguousarray(array)
        digest.update(name.encode());digest.update(str(a.dtype).encode())
        digest.update(str(a.shape).encode());digest.update(a.tobytes())
    return digest.hexdigest()


def dump(path,value):
    Path(path).write_text(json.dumps(json_safe(value),indent=2,allow_nan=False)+'\n')


def inputs(args,model):
    if args.input:
        arrays=dict(np.load(args.input));meta=json.loads(args.input.with_suffix('.json').read_text())
        if meta['knots']!=args.knots or abs(meta['duration']-args.duration)>1e-12:
            raise ValueError('Serialized inputs and requested knots/duration differ')
        if array_hash(arrays)!=meta['content_sha256']:
            raise ValueError('Serialized input content hash mismatch')
    else:
        scene=task(model,args.task)
        dt=args.duration/(args.knots-1)
        x,u=joint_seeds(model,scene['x0'],scene['qgoal'],args.knots,dt,16,args.seed,.1)
        hard_x,hard_u=joint_seeds(model,scene['x0'],scene['qgoal'],args.knots,dt,2,args.seed+19,.4)
        x=np.concatenate((x,hard_x[1:]));u=np.concatenate((u,hard_u[1:]))
        # Reproduce horizon-study packing: analytic seed states, float32 torques.
        xu=pack(x,u.astype(np.float32))
        x0=np.tile(scene['x0'],(17,1)).astype(np.float32)
        ref=np.tile(np.r_[scene['goal'],scene['center'],scene['radius']+.008],(17,args.knots)).astype(np.float32)
        arrays=dict(XU=xu,x0=x0,reference=ref,f_ext=np.zeros((17,6),np.float32))
        if not all(np.isfinite(a).all() for a in arrays.values()):
            raise ValueError('Audit inputs must be finite')
        meta=dict(knots=args.knots,duration=args.duration,dt=dt,seed=args.seed,task=args.task,
            scene={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in scene.items()},
            margin=.008,lanes='0..15 nested obstacle-blind proposals;16 finite larger-amplitude diagnostic',
            generator_sha256=sha(ROOT/'examples/randomized_multimodal/tiago_probe.py'),
            content_sha256=array_hash(arrays))
    np.savez_compressed(args.output/'inputs.npz',**arrays);dump(args.output/'inputs.json',meta)
    return arrays,meta


def make_solver(module,batch,cfg,iterations,kkt):
    scale=cfg.dt/.03
    solver=getattr(module,f'BSQP_{batch}_float')(cfg.dt,iterations,kkt,1000,1e-4,1.,10.,
        2*scale,.01*scale,.0001*scale,200.,scale,1.,0.,0.,0.,.01)
    solver.set_rho_adaptation(False)
    return solver


def cpu_coarse(model,x0,controls,dt):
    """Independent float64 replay of the same frozen-acceleration discrete map."""
    x=np.asarray(x0,float).copy();states=[x.copy()];data=model.createData()
    for u in controls:
        if not np.isfinite(x).all() or np.max(np.abs(x))>1e6 or not np.isfinite(u).all():
            states.extend([np.full(14,np.nan)]*(len(controls)+1-len(states)));break
        with np.errstate(over='ignore',invalid='ignore'):
            acc=pin.aba(model,data,x[:7],x[7:],np.asarray(u,float)).copy()
            x=np.r_[x[:7]+dt*x[7:]+.5*dt*dt*acc,x[7:]+dt*acc]
        states.append(x.copy())
    return np.asarray(states)


def metrics(model,scene,cfg,replay,controls):
    row,xyz=evaluate(model,replay,controls,scene,cfg)
    row['mode']=('ccw' if row['winding']>0 else 'cw') if row.get('finite') else 'invalid'
    return row,xyz


def lane_telemetry(value,lane,batch):
    a=np.asarray(value)
    return a[:,lane].tolist() if a.ndim==2 else a[lane].item()


def max_delta(a,b):
    a=np.asarray(a);b=np.asarray(b)
    if a.shape!=b.shape or not np.array_equal(np.isfinite(a),np.isfinite(b)):
        return float('inf')
    finite=np.isfinite(a)&np.isfinite(b)
    return float(np.max(np.abs(a[finite]-b[finite]))) if np.any(finite) else 0.


def compare(actual,expected,aa,ea,lane):
    t=TOLERANCES
    def error(key):return max_delta(aa[key][lane],ea[key][0])
    a,b=actual['metric'],expected['metric']
    ac,bc=a['objective']['total'],b['objective']['total']
    costs_finite=ac is not None and bc is not None and np.isfinite(ac) and np.isfinite(bc)
    gap=abs(ac-bc) if costs_finite else 0. if ac==bc else float('inf')
    objective_close=gap<=t['objective_atol']+t['objective_rtol']*abs(bc) if costs_finite else ac==bc
    objective_material=gap<=t['objective_atol']+t['material_objective_rtol']*abs(bc) if costs_finite else ac==bc
    telemetry={key:all(np.array_equal(p[key],q[key]) for p,q in zip(actual['telemetry'],expected['telemetry']))
               and len(actual['telemetry'])==len(expected['telemetry']) for key in TELEMETRY}
    stopping_equal=telemetry['sqp_iters']
    q=max_delta(aa['planned'][lane,:,:7],ea['planned'][0,:,:7])
    v=max_delta(aa['planned'][lane,:,7:],ea['planned'][0,:,7:])
    identity=actual['input_sha256']==expected['input_sha256']
    numeric=bool(np.allclose(aa['XU'][lane],ea['XU'][0],atol=t['packed_atol'],rtol=t['packed_rtol'],equal_nan=True))
    replay_close=bool(np.allclose(aa['gpu_replay'][lane],ea['gpu_replay'][0],atol=t['packed_atol'],rtol=t['packed_rtol'],equal_nan=True))
    outcome_equal=a['feasible']==b['feasible'] and a['mode']==b['mode']
    terminal_delta=abs(a['terminal']-b['terminal']) if a['terminal'] is not None and b['terminal'] is not None else 0. if a['terminal']==b['terminal'] else float('inf')
    material=bool(identity and outcome_equal and objective_material and q<=t['material_q_atol'] and v<=t['material_v_atol']
        and error('controls')<=t['material_control_atol'] and terminal_delta<=t['material_terminal_atol'])
    return dict(source_lane=actual['source_lane'],input_identity=identity,
        strict_numeric_pass=bool(identity and numeric and replay_close and objective_close and outcome_equal),material_pass=material,
        stopping_equal=stopping_equal,telemetry_equal=telemetry,
        max_XU=error('XU'),max_planned_q=q,max_planned_v=v,max_control=error('controls'),
        max_gpu_replay=error('gpu_replay'),max_cpu_replay=error('cpu_replay'),
        objective_abs_difference=gap,objective_relative_difference=gap/max(abs(bc),1e-12) if costs_finite else None,
        objective_actual=ac,objective_serial=bc,feasible_actual=a['feasible'],feasible_serial=b['feasible'],
        mode_actual=a['mode'],mode_serial=b['mode'],terminal_abs_difference=terminal_delta)


def run_case(args,module,model,scene,cfg,pool,name,mapping,iterations,passes,kkt,reuse=None):
    path=args.output/'cases'/name
    if args.resume and path.with_suffix('.json').exists() and path.with_suffix('.npz').exists() and reuse is None:
        return json.loads(path.with_suffix('.json').read_text()),dict(np.load(path.with_suffix('.npz'))),None
    batch=len(mapping);solver=reuse or make_solver(module,batch,cfg,iterations,kkt)
    solver.reset_dual();solver.reset_rho()
    fed={key:np.ascontiguousarray(value[mapping]) for key,value in pool.items()}
    original={key:value.copy() for key,value in fed.items()}
    solver.set_f_ext_batch(fed['f_ext'])
    z=fed['XU'].copy();traces=[];outputs=[]
    for step in range(passes):
        result=solver.solve(z.copy(),cfg.dt,fed['x0'].copy(),fed['reference'].copy())
        z=np.asarray(result['XU']).copy();outputs.append(z.copy())
        traces.append({key:np.asarray(result[key]).copy() for key in TELEMETRY})
    planned,controls=unpack(z,cfg.knots)
    gpu=replay_cuda(solver,scene['x0'],controls,cfg.dt)
    cpu=np.asarray([cpu_coarse(model,fed['x0'][b],u,cfg.dt) for b,u in enumerate(controls)])
    lanes=[];tool=[]
    for b,source in enumerate(mapping):
        metric,xyz=metrics(model,scene,cfg,cpu[b],controls[b]);tool.append(xyz)
        gm,_=metrics(model,scene,cfg,gpu[b],controls[b])
        lanes.append(dict(source_lane=source,input_sha256=array_hash({key:v[b:b+1] for key,v in original.items()}),
            metric=metric,gpu_metric=gm,cpu_gpu_replay=max_delta(cpu[b],gpu[b]),
            telemetry=[{key:lane_telemetry(value,b,batch) for key,value in trace.items()} for trace in traces]))
    record=dict(name=name,mapping=mapping,iterations=iterations,passes=passes,kkt_tol=kkt,
        lanes=lanes,input_unmodified=all(np.array_equal(fed[k],original[k]) for k in fed))
    arrays=dict(XU=z,planned=planned,controls=controls,gpu_replay=gpu,cpu_replay=cpu,xyz=np.asarray(tool),
                pass_outputs=np.asarray(outputs))
    np.savez_compressed(path.with_suffix('.npz'),**arrays);dump(path.with_suffix('.json'),record)
    print(json.dumps(dict(case=name,batch=batch,finished=True)),flush=True)
    return json_safe(record),arrays,solver


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--knots',type=int,default=16);p.add_argument('--duration',type=float,default=.45)
    p.add_argument('--batches',type=int,nargs='+',default=[2,4,8,16]);p.add_argument('--seed',type=int,default=1401)
    p.add_argument('--task',type=int,default=0);p.add_argument('--iterations',type=int,default=200)
    p.add_argument('--passes',type=int,default=2);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--profiles',nargs='+',choices=['fixed','production'],default=['fixed','production'])
    p.add_argument('--quick',action='store_true');p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--resume',action='store_true');p.add_argument('--input',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.quick:args.iterations,args.passes,args.repeats,args.profiles=20,1,2,['fixed']
    if not set(args.batches)<=set((1,2,4,8,16)):raise ValueError('Supported audited batch sizes are1,2,4,8,16')
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'cases').mkdir(exist_ok=True)
    model=pin.buildModelFromUrdf(str(ROOT/'gato/dynamics/tiago_right/tiago_right_arm.urdf'))
    pool,meta=inputs(args,model)
    if args.prepare_only:
        print(json.dumps(dict(inputs=str(args.output/'inputs.npz'),sha256=meta['content_sha256'])));return
    module=importlib.import_module(f'bsqp.bsqpN{args.knots}_tiago_right_multimodal')
    scene={k:np.asarray(v) if isinstance(v,list) else v for k,v in meta['scene'].items()}
    # All audit replays start at exactly the submitted float32 measurement.
    scene['x0']=pool['x0'][0].astype(float)
    cfg=SimpleNamespace(knots=args.knots,dt=meta['dt'],margin=meta['margin'])
    source_paths=[Path(__file__),ROOT/'examples/randomized_multimodal/tiago_horizon.py',
        ROOT/'examples/randomized_multimodal/tiago_probe.py',ROOT/'python/bindings.cu',
        ROOT/'gato/bsqp/bsqp.cuh',ROOT/'gato/bsqp/kernels/pcg.cuh',
        ROOT/'gato/dynamics/tiago_right/tiago_right_plant.cuh',Path(module.__file__)]
    manifest=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        inputs=meta,tolerances=TOLERANCES,precision='float32 solver; float64 independent coarse replay',
        solver=dict(max_pcg_iters=1000,pcg_tol=1e-4,mu=10.,rho=.01,rho_adaptation=False,
            solve_ratio=1.,external_wrench='exact zero',duals='zero before first pass; continued between passes',
            running_scale=cfg.dt/.03,terminal_position=200.,terminal_obstacle=1.),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        sha256={str(q.relative_to(ROOT)):sha(q) for q in source_paths},
        note='Compare batching within each device. Do not require bit-identical trajectories across GPU architectures.')
    manifest_path=args.output/'manifest.json'
    if args.resume and manifest_path.exists():
        old=json.loads(manifest_path.read_text())
        for key in ('inputs','tolerances','solver','sha256'):
            if old[key]!=manifest[key]:raise ValueError(f'Cannot resume changed audit {key}')
        for key in ('iterations','passes','repeats','profiles','quick','batches'):
            if old['arguments'][key]!=manifest['arguments'][key]:raise ValueError(f'Cannot resume changed {key}')
    dump(manifest_path,manifest)  # Freeze tolerance and workload before any solve.
    profiles=[('prefix1',1,1,0.)]+[(s,args.iterations,args.passes,0. if s=='fixed' else 1e-3) for s in args.profiles]
    comparisons=[];repeat_rows=[]
    summary=dict(scope='Per-lane arithmetic/stopping audit; not a trajectory-quality success claim',
        quick=args.quick,profiles={},tolerances=TOLERANCES,input_sha256=meta['content_sha256'])
    for profile,iterations,passes,kkt in profiles:
        serial={}
        for lane in range(17):
            for repetition in range(args.repeats):
                name=f'{profile}_serial{lane}_repeat{repetition}'
                result,arrays,_=run_case(args,module,model,scene,cfg,pool,name,[lane],iterations,passes,kkt)
                if repetition==0:serial[lane]=(result,arrays)
                else:
                    ref,ra=serial[lane];row=compare(result['lanes'][0],ref['lanes'][0],arrays,ra,0)
                    repeat_rows.append(dict(profile=profile,case=name,**row))
        cases=[(f'nested{b}',list(range(b))) for b in sorted(set(args.batches)) if b>1]
        largest=max(args.batches)
        cases += [('reversed',list(reversed(range(largest)))),('duplicates',[0]*largest),
                  ('hard_mix',[0]*(largest-1)+[16]),('reset_after_hard',list(range(largest)))]
        reused=None
        for label,mapping in cases:
            name=f'{profile}_{label}'
            if label=='reset_after_hard' and reused is None:
                # A resumed hard_mix case may have come from disk. Recreate the
                # contaminated solver before testing its reset, never silently
                # substitute a fresh solver for this stateful test.
                warm=make_solver(module,largest,cfg,iterations,kkt)
                _,_,reused=run_case(args,module,model,scene,cfg,pool,f'{profile}_hard_mix',
                    [0]*(largest-1)+[16],iterations,passes,kkt,reuse=warm)
            result,arrays,solver=run_case(args,module,model,scene,cfg,pool,name,mapping,iterations,passes,kkt,
                reuse=reused if label=='reset_after_hard' else None)
            if label=='hard_mix':reused=solver
            for position,lane in enumerate(mapping):
                ref,ra=serial[lane]
                row=compare(result['lanes'][position],ref['lanes'][0],arrays,ra,position)
                comparisons.append(dict(profile=profile,case=label,position=position,batch=len(mapping),**row))
        selected=[r for r in comparisons if r['profile']==profile]
        repeats=[r for r in repeat_rows if r['profile']==profile]
        summary['profiles'][profile]=dict(comparisons=len(selected),strict_pass=sum(r['strict_numeric_pass'] for r in selected),
            material_pass=sum(r['material_pass'] for r in selected),input_identity=all(r['input_identity'] for r in selected),
            differing_stopping=sum(not r['stopping_equal'] for r in selected),
            max_XU=max(r['max_XU'] for r in selected),max_objective_relative=max(r['objective_relative_difference'] or 0 for r in selected),
            serial_repeat_max_XU=max((r['max_XU'] for r in repeats),default=0),
            serial_repeat_max_objective_relative=max((r['objective_relative_difference'] or 0 for r in repeats),default=0),
            by_batch={str(b):dict(comparisons=sum(r['batch']==b for r in selected),
                strict_pass=sum(r['batch']==b and r['strict_numeric_pass'] for r in selected),
                material_pass=sum(r['batch']==b and r['material_pass'] for r in selected)) for b in sorted({r['batch'] for r in selected})},
            verdict='PASS' if all(r['strict_numeric_pass'] for r in selected) else 'REVIEW_REQUIRED')
        summary['comparisons']=comparisons;summary['serial_repeats']=repeat_rows
        dump(args.output/'summary.json',summary)
        print(json.dumps(json_safe(dict(profile=profile,**summary['profiles'][profile]))),flush=True)
    dump(args.output/'summary.json',summary)


if __name__=='__main__':main()
