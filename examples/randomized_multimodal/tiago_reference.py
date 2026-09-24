"""Independent acceleration-coordinate CPU polish of the TIAGo objective.

This numerical comparator is never used to propose or select GATO candidates.
Its exact linear joint integration and Pinocchio inverse dynamics parameterize
the same discrete dynamics without unstable long-horizon torque shooting.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pinocchio as pin
from scipy.optimize import minimize
from tiago_probe import ROOT, fk, metrics, json_safe


class Objective:
    def __init__(self, model, x0, goal, center, radius, args):
        self.model, self.data = model, model.createData()
        self.x0, self.goal, self.center = x0, goal, center
        self.radius, self.args = radius, args
        n, dt = args.knots, args.dt
        k, j = np.arange(n)[:,None], np.arange(n-1)[None]
        self.Q = np.where(k>j, dt**2*(k-j-.5), 0.)
        self.V = np.where(k>j, dt, 0.)
        self.base_q = x0[:7] + np.arange(n)[:,None]*dt*x0[7:]
        self.frame = model.getFrameId('arm_right_tool_link')

    def rollout(self, acceleration):
        a = acceleration.reshape(-1,7)
        q = self.base_q + self.Q@a
        v = self.x0[7:] + self.V@a
        u = np.array([pin.rnea(self.model,self.data,qi,vi,ai).copy()
                      for qi,vi,ai in zip(q[:-1],v[:-1],a)])
        return np.c_[q,v], u

    def __call__(self, acceleration):
        args, model, data = self.args, self.model, self.data
        a = acceleration.reshape(-1,7)
        x,u = self.rollout(acceleration)
        q,v = x[:,:7],x[:,7:]
        dq = np.zeros_like(q)
        dv = args.qd*v
        da = np.zeros_like(a)
        cost = .5*args.qd*np.sum(v*v)+.5*args.u*np.sum(u*u)
        for k in range(args.knots):
            xyz = fk(model,data,q[k])
            torso_rotation = data.oMf[model.getFrameId('torso_lift_link')].rotation.copy()
            jac = torso_rotation.T @ pin.computeFrameJacobian(model,data,q[k],self.frame,pin.LOCAL_WORLD_ALIGNED)[:3]
            err = xyz-self.goal
            weight = args.terminal if k==args.knots-1 else 2.
            cost += .5*weight*(err@err)
            dxyz = weight*err
            delta = xyz[:2]-self.center
            residual = max(1.-delta@delta/self.radius**2,0.)
            cost += .5*args.obstacle*residual**2
            dxyz[:2] -= 2*args.obstacle*residual*delta/self.radius**2
            dq[k] += jac.T@dxyz
            if k<len(a):
                tau_q,tau_v,tau_a = pin.computeRNEADerivatives(model,data,q[k],v[k],a[k])
                dq[k] += args.u*tau_q.T@u[k]
                dv[k] += args.u*tau_v.T@u[k]
                da[k] += args.u*tau_a.T@u[k]
        gradient = self.Q.T@dq+self.V.T@dv+da
        return cost, gradient.ravel()


def run(args):
    summary = json.loads((args.input/'summary.json').read_text())
    config = SimpleNamespace(**summary['config'])
    model = pin.buildModelFromUrdf(str(ROOT/'gato/dynamics/tiago_right/tiago_right_arm.urdf'))
    results = json.loads((args.input/'cpu_reference.json').read_text()) if args.resume and (args.input/'cpu_reference.json').exists() else []
    completed = {(r['scene'],r['seed'],r['batch'],r['lane']) for r in results}
    for trial in summary['trials']:
        index,seed = trial['scene'],trial['seed']
        prefix = f'scene{index}_seed{seed}'
        goal,center = np.array(trial['goal']),np.array(trial['center_after'])
        x0 = np.array(trial['common_state'])
        obj = Objective(model,x0,goal,center,trial['radius']+config.margin,config)
        for batch in (1,config.batch):
            record = trial['branches'][str(batch)]['stages'][0]
            arrays = np.load(args.input/f'{prefix}_b{batch}_stage0.npz')
            lanes = range(batch) if args.all_lanes else [record['winner']]
            for lane in lanes:
                if (index,seed,batch,lane) in completed:
                    continue
                a0 = np.diff(arrays['replay'][lane,:,7:],axis=0).ravel()/config.dt
                value,g = obj(a0)
                rng = np.random.default_rng(29)
                direction = rng.normal(size=a0.size); direction/=np.linalg.norm(direction)
                fd = (obj(a0+1e-4*direction)[0]-obj(a0-1e-4*direction)[0])/2e-4
                result = minimize(obj,a0,jac=True,method='L-BFGS-B',
                    options=dict(maxiter=args.maxiter,gtol=1e-8,ftol=1e-13,maxls=50,maxcor=30))
                epsilon = 1e-3
                hessian = np.column_stack([(obj(result.x+epsilon*d)[1]-obj(result.x-epsilon*d)[1])/(2*epsilon)
                                            for d in np.eye(result.x.size)])
                minimum_curvature = float(np.linalg.eigvalsh((hessian+hessian.T)/2)[0])
                x,u = obj.rollout(result.x)
                rows,xyz = metrics(model,x[None],u[None],goal,center,trial['radius'],
                                   config.dt,margin=config.margin,args=config)
                row = dict(scene=index,seed=seed,batch=batch,lane=lane,
                    initial_cost=value,initial_gradient_inf=float(np.max(np.abs(g))),
                    final_cost=float(result.fun),gradient_inf=float(np.max(np.abs(result.jac))),
                    derivative_error=float(abs(fd-g@direction)),success=bool(result.success),
                    message=str(result.message),iterations=int(result.nit),
                    finite_difference_hessian_min_eigenvalue=minimum_curvature,metrics=rows[0])
                results.append(row)
                np.savez_compressed(args.input/f'{prefix}_b{batch}_lane{lane}_cpu.npz',
                                    states=x,controls=u,xyz=xyz[0],acceleration=result.x.reshape(-1,7))
                print(json.dumps(json_safe(row)),flush=True)
        (args.input/'cpu_reference.json').write_text(json.dumps(json_safe(results),indent=2,allow_nan=False))


def refine_curvature(directory):
    """Resolve a finite-difference stencil that crosses a residual hinge.

    Preserve the coarse diagnostic and verify stability at two smaller scales.
    This does not change the optimizer output or its objective value.
    """
    raw=json.loads((directory/'summary.json').read_text())
    rows=json.loads((directory/'cpu_reference.json').read_text())
    args=SimpleNamespace(**raw['config'])
    model=pin.buildModelFromUrdf(str(ROOT/'gato/dynamics/tiago_right/tiago_right_arm.urdf'))
    for row in rows:
        if row['finite_difference_hessian_min_eigenvalue']>=0 or 'curvature_checks' in row:
            continue
        trial=next(t for t in raw['trials'] if t['scene']==row['scene'] and t['seed']==row['seed'])
        obj=Objective(model,np.array(trial['common_state']),np.array(trial['goal']),
                      np.array(trial['center_after']),trial['radius']+args.margin,args)
        filename=f"scene{row['scene']}_seed{row['seed']}_b{row['batch']}_lane{row['lane']}_cpu.npz"
        acceleration=np.load(directory/filename)['acceleration'].ravel()
        checks=[dict(epsilon=.001,min_eigenvalue=row['finite_difference_hessian_min_eigenvalue'])]
        for epsilon in (1e-4,1e-5):
            h=np.column_stack([(obj(acceleration+epsilon*d)[1]-obj(acceleration-epsilon*d)[1])/(2*epsilon)
                               for d in np.eye(acceleration.size)])
            eig,vec=np.linalg.eigh((h+h.T)/2)
            checks.append(dict(epsilon=epsilon,min_eigenvalue=float(eig[0]),
                plus_direction_cost_change=float(obj(acceleration+.01*vec[:,0])[0]-row['final_cost']),
                minus_direction_cost_change=float(obj(acceleration-.01*vec[:,0])[0]-row['final_cost'])))
        row['curvature_checks']=checks
        stable=abs(checks[-1]['min_eigenvalue']-checks[-2]['min_eigenvalue'])<1e-8
        row['curvature_refinement_stable']=stable
        if stable:
            row['validated_hessian_min_eigenvalue']=checks[-1]['min_eigenvalue']
    (directory/'cpu_reference.json').write_text(json.dumps(json_safe(rows),indent=2,allow_nan=False))


def summarize(directory):
    """Scene-clustered paired statistics; CPU is a best-known local reference."""
    raw = json.loads((directory/'summary.json').read_text())
    cpu = json.loads((directory/'cpu_reference.json').read_text())
    expected = {(scene,seed) for scene in range(raw['config']['scenes']) for seed in raw['config']['seeds']}
    actual = {(t['scene'],t['seed']) for t in raw['trials']}
    if actual != expected or len(raw['trials']) != len(expected):
        raise ValueError('Statistics require the complete predefined scene/seed set')
    cpu_keys = {(r['scene'],r['seed'],r['batch'],r['lane']) for r in cpu}
    needed = {(t['scene'],t['seed'],int(b),t['branches'][b]['stages'][0]['winner'])
              for t in raw['trials'] for b in ('1',str(raw['config']['batch']))}
    if not needed <= cpu_keys:
        raise ValueError('Statistics require independent CPU references for both selected outputs in every trial')
    records, scene_gains = [], {}
    common_states = {}
    for t in raw['trials']:
        sid = t['scene']
        common_states.setdefault(sid, []).append(t['common_state']+t['goal']+t['center_after']+[t['radius']])
        available = [r for r in cpu if r['scene']==sid and r['metrics']['feasible']]
        reference = min((r['final_cost'] for r in available), default=float('nan'))
        selected = {}
        for batch in ('1', str(raw['config']['batch'])):
            stage = t['branches'][batch]['stages'][0]
            lane = stage['winner']
            after, before = stage['lanes'][lane], stage['before'][lane]
            cost = after['objective']['total'] if after['objective']['total'] is not None else float('inf')
            initial_cost = before['objective']['total'] if before['objective']['total'] is not None else float('inf')
            selected[batch] = dict(cost=cost,
                feasible=after['feasible'], winding=after['winding'], lane=lane,
                repaired=bool(not before['feasible'] and after['feasible']),
                initial_clearance=before['clearance'], clearance=after['clearance'],
                defect=stage['defect'][lane],
                reference_gap=cost/reference-1,
                optimization_reduction=1-cost/initial_cost)
        a,b = selected['1'],selected[str(raw['config']['batch'])]
        gain = 1-b['cost']/a['cost'] if np.isfinite(a['cost']) and np.isfinite(b['cost']) else float('nan')
        scene_gains.setdefault(sid,[]).append(gain)
        records.append(dict(scene=sid,seed=t['seed'],prefix_safe=t.get('prefix_safe'),
            reference=reference,gain=gain,opposite_winding=bool(a['winding'] is not None and b['winding'] is not None and a['winding']*b['winding']<0),
            methods=selected))
    max_state_spread = max(float(np.max(np.ptp(v,axis=0))) for v in common_states.values())
    if max_state_spread != 0:
        raise ValueError('Per-scene CPU pooling requires identical common states, goals, obstacle geometry and shared config across seeds')
    cluster_means = np.array([np.mean(v) for v in scene_gains.values()])
    rng = np.random.default_rng(20260924)
    boot = np.mean(rng.choice(cluster_means, size=(20000,len(cluster_means))),axis=1)
    methods = {}
    for batch in ('1',str(raw['config']['batch'])):
        values = [r['methods'][batch] for r in records]
        gaps = np.array([v['reference_gap'] for v in values])
        methods[batch] = dict(physical_feasible=sum(v['feasible'] for v in values),
            failures=sum(not v['feasible'] for v in values),
            repaired_selected_seeds=sum(v['repaired'] for v in values),
            fresh_selected=sum(v['lane']>0 for v in values),
            reference_gap_mean=float(np.mean(gaps)),reference_gap_median=float(np.median(gaps)),
            reference_gap_max=float(np.max(gaps)),
            within_3_percent=int(sum(v['feasible'] and g<=.03 for v,g in zip(values,gaps))),
            within_5_percent=int(sum(v['feasible'] and g<=.05 for v,g in zip(values,gaps))),
            within_10_percent=int(sum(v['feasible'] and g<=.10 for v,g in zip(values,gaps))),
            selected_defect_max=max(v['defect'] for v in values),
            physical_clearance_min=min(v['clearance'] for v in values),
            optimization_reduction_median=float(np.median([v['optimization_reduction'] for v in values])))
    out = dict(scope='Common-state first replan; tool center; best-known CPU local reference, no global or GATO KKT certificate.',
        split='heldout' if raw['config'].get('heldout') else 'development',
        scenes=len(scene_gains),trials=len(records),
        expected_trials=raw['config']['scenes']*len(raw['config']['seeds']),
        reference_coverage=len(cpu),expected_reference_solutions=2*len(records),
        finite_cost_pairs=int(sum(np.isfinite(r['gain']) for r in records)),prefix_safe=sum(r['prefix_safe'] is True for r in records),
        paired_mean_reduction=float(np.mean(cluster_means)),
        scene_cluster_bootstrap_95_percent=np.quantile(boot,[.025,.975]).tolist(),
        opposite_winding=sum(r['opposite_winding'] for r in records),
        paired_positive=sum(r['gain']>0 for r in records),methods=methods,
        cpu=dict(solutions=len(cpu),feasible=sum(r['metrics']['feasible'] for r in cpu),
            gradient_inf_max=max(r['gradient_inf'] for r in cpu),
            hessian_min_eigenvalue=min(r.get('validated_hessian_min_eigenvalue',r['finite_difference_hessian_min_eigenvalue']) for r in cpu),
            coarse_curvature_negative=sum(r['finite_difference_hessian_min_eigenvalue']<0 for r in cpu),
            refined_curvature_checks=sum('curvature_checks' in r for r in cpu),
            directional_derivative_error_max=max(r['derivative_error'] for r in cpu)),
        records=records)
    (directory/'statistics.json').write_text(json.dumps(json_safe(out),indent=2,allow_nan=False))
    print(json.dumps(json_safe({k:v for k,v in out.items() if k!='records'}),indent=2,allow_nan=False))
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--maxiter',type=int,default=3000)
    p.add_argument('--all-lanes',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--summarize-only',action='store_true')
    p.add_argument('--refine-curvature',action='store_true')
    options=p.parse_args()
    if not options.summarize_only:
        run(options)
    if options.refine_curvature:
        refine_curvature(options.input)
    summarize(options.input)
