"""Development transfer probe: obstacle-blind joint trajectories with TIAGo GATO.

This is a tool-center benchmark, not a whole-arm collision safety test.
"""
from __future__ import annotations

import argparse
import importlib
import hashlib
import subprocess
import json
from pathlib import Path
import sys

import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))


def fk(model, data, q):
    pin.framesForwardKinematics(model, data, q)
    return (data.oMf[model.getFrameId('torso_lift_link')].inverse()
            * data.oMf[model.getFrameId('arm_right_tool_link')]).translation.copy()


def joint_seeds(model, x0, qgoal, n, dt, batch, seed, amplitude):
    """Minimum-energy acceleration plus random smooth endpoint-null perturbations.

    Inputs contain no obstacle description or collision/mode feedback.
    Inverse dynamics converts the single analytic joint rollout to torques.
    """
    h = n - 1
    t = (np.arange(h) + .5) / h
    terminal = np.vstack([dt**2 * (h - np.arange(h) - .5), np.full(h, dt)])
    rhs = np.stack([qgoal - x0[:7] - h * dt * x0[7:], -x0[7:]])
    nominal = terminal.T @ np.linalg.solve(terminal @ terminal.T, rhs)
    basis = np.cos(np.pi * np.arange(1, 4)[:, None] * t)
    basis -= (terminal.T @ np.linalg.solve(terminal @ terminal.T, terminal @ basis.T)).T
    rng = np.random.default_rng(seed)
    controls, states = [], []
    for b in range(batch):
        coefficients = rng.uniform(-1, 1, (3, 7))
        perturb = basis.T @ coefficients
        # Scale by position excursion rather than acceleration magnitude.
        dv = np.vstack([np.zeros(7), dt * np.cumsum(perturb, axis=0)])
        dq = np.vstack([np.zeros(7), np.cumsum(dt * dv[:-1] + .5 * dt**2 * perturb, axis=0)])
        perturb *= amplitude / max(np.max(np.abs(dq)), 1e-12)
        acceleration = nominal + (perturb if b else 0)
        x = x0.copy()
        xs, us = [x.copy()], []
        data = model.createData()
        for a in acceleration:
            us.append(pin.rnea(model, data, x[:7], x[7:], a).copy())
            x = np.r_[x[:7] + dt * x[7:] + .5 * dt**2 * a, x[7:] + dt * a]
            xs.append(x.copy())
        states.append(xs)
        controls.append(us)
    return np.asarray(states), np.asarray(controls)


def pack(x, u):
    z = np.concatenate([x[:, :-1], u], axis=2).reshape(len(x), -1)
    return np.concatenate([z, x[:, -1]], axis=1).astype(np.float32)


def unpack(z, n):
    body = z[:, : (n-1)*21].reshape(-1, n-1, 21)
    return np.concatenate([body[:, :, :14], z[:, None, -14:]], axis=1), body[:, :, 14:]


def executed_objective(x, u, xyz, goal, center, effective_radius, args):
    """Independent nonnegative plant objective; sum once per knot, no dt factor."""
    error2 = np.sum((xyz - goal) ** 2, axis=-1)
    radial = np.sum((xyz[:, :2] - center) ** 2, axis=-1)
    residual = np.maximum(1. - radial / effective_radius**2, 0.)
    parts = dict(position=float(np.sum(error2[:-1])),
                 terminal=float(.5 * args.terminal * error2[-1]),
                 velocity=float(.5 * args.qd * np.sum(x[:, 7:]**2)),
                 effort=float(.5 * args.u * np.sum(u**2)),
                 obstacle=float(.5 * args.obstacle * np.sum(residual**2)))
    return dict(total=sum(parts.values()), **parts)


def metrics(model, x, u, goal, center, radius, dt, *, margin=0., args=None):
    data = model.createData()
    xyz = np.full((len(x), x.shape[1], 3), np.nan)
    rows = []
    for b in range(len(x)):
        finite = bool(np.isfinite(x[b]).all() and np.isfinite(u[b]).all())
        if not finite:
            rows.append(dict(finite=False, terminal=float('inf'), clearance=float('-inf'),
                             clearance_margin=float('-inf'), joint_violation=float('inf'),
                             velocity_ratio=float('inf'), effort_ratio=float('inf'),
                             terminal_tool_speed=float('inf'), winding=float('nan'),
                             feasible=False, margin_feasible=False, settled=False,
                             objective=dict(total=float('inf'))))
            continue
        q, v = x[b, :, :7], x[b, :, 7:]
        xyz[b] = [fk(model, data, qi) for qi in q]
        sample_times = np.linspace(0., dt, 17)[:, None]
        # Dense samples of the exact constant-acceleration interpolant of each
        # coarse CUDA step. This is not a fine-step nonlinear dynamics replay.
        dense_q = np.concatenate([
            q[k] + sample_times*v[k]
            + .5*sample_times**2*(v[k+1]-v[k])/dt
            for k in range(len(q)-1)])
        dense_xyz = np.array([fk(model,data,qi) for qi in dense_q])
        clearance = np.min(np.linalg.norm(dense_xyz[:, :2] - center, axis=1)) - radius
        terminal = np.linalg.norm(xyz[b, -1] - goal)
        joint = max(float(np.max(model.lowerPositionLimit - dense_q)), float(np.max(dense_q - model.upperPositionLimit)), 0.)
        velocity = np.max(np.abs(v) / model.velocityLimit)
        effort = np.max(np.abs(u[b]) / model.effortLimit)
        delta = xyz[b, :, :2] - center
        winding = np.diff(np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))).sum()
        jacobian = pin.computeFrameJacobian(model, data, q[-1], model.getFrameId('arm_right_tool_link'), pin.LOCAL_WORLD_ALIGNED)
        speed = np.linalg.norm(jacobian[:3] @ v[-1])
        feasible = bool(terminal < .02 and clearance > 0 and joint < 1e-5 and velocity <= 1 and effort <= 1)
        row = dict(finite=True, terminal=float(terminal), clearance=float(clearance),
                   clearance_margin=float(clearance-margin), joint_violation=joint,
                   velocity_ratio=float(velocity), effort_ratio=float(effort), winding=float(winding),
                   terminal_tool_speed=float(speed), feasible=feasible,
                   margin_feasible=bool(feasible and clearance >= margin), settled=bool(speed < .05))
        if args is not None:
            row['objective'] = executed_objective(x[b], u[b], xyz[b], goal, center, radius+margin, args)
        rows.append(row)
    return rows, xyz


def replay_cuda(solver, x0, controls, dt):
    states = np.zeros((len(controls), controls.shape[1]+1, len(x0)), dtype=np.float32)
    states[:, 0] = x0
    for k in range(controls.shape[1]):
        states[:, k+1] = solver.sim_forward(states[:, k].copy(), controls[:, k].copy(), dt)
    return states


def pinocchio_one_step_parity(model, states, controls, dt):
    """Evaluate independent dynamics at each CUDA state, without error accumulation."""
    data = model.createData()
    rows = []
    for xs, us in zip(states, controls):
        if not (np.isfinite(xs).all() and np.isfinite(us).all()):
            rows.append(dict(finite=False, max_abs=float('inf'), max_l2=float('inf')))
            continue
        errors = []
        for k, control in enumerate(us):
            q, v = np.asarray(xs[k, :7], float), np.asarray(xs[k, 7:], float)
            acceleration = pin.aba(model, data, q, v, np.asarray(control, float))
            predicted = np.r_[q+dt*v+.5*dt**2*acceleration, v+dt*acceleration]
            errors.append(predicted-xs[k+1])
        errors = np.asarray(errors)
        rows.append(dict(finite=bool(np.isfinite(errors).all()), max_abs=float(np.max(np.abs(errors))),
                         max_l2=float(np.max(np.linalg.norm(errors, axis=1)))))
    return rows


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run(args):
    model = pin.buildModelFromUrdf(str(ROOT / 'gato/dynamics/tiago_right/tiago_right_arm.urdf'))
    data = model.createData()
    q0 = np.array([-.39, -1.73, -.38, -2.35, 0., -1.21, .04])
    start = fk(model, data, q0)
    target = start + np.array([args.travel, 0., 0.])
    ik = least_squares(lambda q: np.r_[100*(fk(model, data, q)-target), .01*(q-q0)], q0,
                       bounds=(model.lowerPositionLimit+.03, model.upperPositionLimit-.03), max_nfev=200)
    goal = fk(model, data, ik.x)
    center = .5*(start[:2]+goal[:2]) + np.array([0., args.offset])
    x0 = np.r_[q0, np.zeros(7)]
    x, u = joint_seeds(model, x0, ik.x, args.knots, args.dt, args.batch, args.seed, args.amplitude)
    seeds = pack(x,u)
    u = u.astype(np.float32)  # exact controls submitted to the extension
    module = importlib.import_module(f'bsqp.bsqpN{args.knots}_tiago_right_multimodal')
    solver = getattr(module, f'BSQP_{args.batch}_float')(
        args.dt, args.iters, args.kkt, 1000, args.pcg, 1., args.mu, 2., args.qd, args.u,
        args.terminal, args.obstacle, args.obstacle, .0, .0, .0, args.rho)
    solver.set_f_ext_batch(np.zeros((args.batch,6),np.float32))
    solver.set_rho_adaptation(not args.fixed_rho)
    ref = np.tile(np.r_[goal,center,args.radius+args.margin], (args.batch,args.knots)).astype(np.float32)
    solve_inputs = seeds
    results = []
    for _ in range(1 + args.polish):
        result = solver.solve(solve_inputs, args.dt, np.tile(x0,(args.batch,1)).astype(np.float32), ref)
        results.append(result)
        solve_inputs = np.asarray(result['XU']).copy()
    # Retain every solve call: polishing reuses the decision trajectory and the
    # solver's dual/rho warm state, with the same original objective and budget.
    result = results[-1]
    pass_summary = [dict(initial_merit=np.asarray(r['initial_merit']).tolist(),
                         final_merit=np.asarray(r['final_merit']).tolist(),
                         sqp_iters=np.asarray(r['sqp_iters']).tolist(),
                         kkt=np.asarray(r['kkt_converged']).tolist(),
                         solve_ms=float(r['sqp_time_us'])/1000) for r in results]
    planned, optimized_u = unpack(np.asarray(result['XU']), args.knots)
    replay = replay_cuda(solver, x0, optimized_u, args.dt)
    seed_replay = replay_cuda(solver, x0, u.astype(np.float32), args.dt)
    common = dict(margin=args.margin, args=args)
    constructed, constructed_xyz = metrics(model,x,u,goal,center,args.radius,args.dt,**common)
    before, seed_xyz = metrics(model,seed_replay,u,goal,center,args.radius,args.dt,**common)
    after, xyz = metrics(model,replay,optimized_u,goal,center,args.radius,args.dt,**common)
    planned_metrics, planned_xyz = metrics(model,planned,optimized_u,goal,center,args.radius,args.dt,**common)
    status = np.concatenate([np.asarray(r['pcg_status']) for r in results],axis=0)
    pcg_iters = np.concatenate([np.asarray(r['pcg_iters']) for r in results],axis=0)
    source_paths = [Path(__file__), Path(module.__file__), ROOT/'gato/dynamics/tiago_right/tiago_right_arm.urdf',
                    ROOT/'gato/dynamics/tiago_right/tiago_right_plant.cuh']
    out = dict(config=vars(args).copy(),start=start.tolist(),goal=goal.tolist(),center=center.tolist(),
               certificate_scope='tool center, 17 samples per coarse constant-acceleration step; no whole-arm collision check',
               before=before,after=after,constructed=constructed,planned_metrics=planned_metrics,
               defect=np.max(np.abs(replay-planned),axis=(1,2)).tolist(),
               seed_defect=np.max(np.abs(seed_replay-x),axis=(1,2)).tolist(),
               pinocchio_one_step_before=pinocchio_one_step_parity(model,seed_replay,u,args.dt),
               pinocchio_one_step_after=pinocchio_one_step_parity(model,replay,optimized_u,args.dt),
               objective_improvement=[a['objective']['total']-b['objective']['total'] for a,b in zip(before,after)],
               clearance_improvement=[b['clearance']-a['clearance'] for a,b in zip(before,after)],
               replay_tool_path_rms_change=np.sqrt(np.mean(np.sum((xyz-seed_xyz)**2,axis=2),axis=1)).tolist(),
               change=np.max(np.abs(np.asarray(result['XU'])-seeds),axis=1).tolist(),
               initial_merit=np.asarray(results[0]['initial_merit']).tolist(),final_merit=np.asarray(result['final_merit']).tolist(),
               solve_passes=pass_summary,
               sqp_iters=np.sum([np.asarray(r['sqp_iters']) for r in results],axis=0).tolist(),kkt=np.asarray(result['kkt_converged']).tolist(),
               pcg_status=status.tolist(),pcg_iters=pcg_iters.tolist(),
               pcg_cap_counts=np.sum(status==0,axis=0).tolist(),
               pcg_breakdown_counts=np.sum(status==2,axis=0).tolist(),
               pcg_cap_rows=np.argwhere(status==0).tolist(),
               pcg_max=int(np.max(pcg_iters)),solve_ms=sum(float(r['sqp_time_us'])/1000 for r in results),
               provenance=dict(git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                   source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}))
    args.output.mkdir(parents=True,exist_ok=True)
    out['config']['output'] = str(args.output)
    (args.output/'summary.json').write_text(json.dumps(json_safe(out),indent=2,allow_nan=False))
    np.savez_compressed(args.output/'trajectories.npz',seed_xyz=seed_xyz,xyz=xyz,planned=planned,replay=replay,
                        seeds=seeds,seed_replay=seed_replay,seed_controls=u,constructed=x,constructed_xyz=constructed_xyz,
                        planned_xyz=planned_xyz,optimized=np.asarray(result['XU']),controls=optimized_u,
                        pcg_status=status,pcg_iters=pcg_iters,solve_outputs=np.asarray([r['XU'] for r in results]))
    print(json.dumps(json_safe(out),indent=2,allow_nan=False))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name,default,type_ in [('knots',16,int),('dt',.03,float),('batch',16,int),('seed',100,int),
                              ('amplitude',.1,float),('travel',.12,float),('radius',.025,float),
                              ('margin',.008,float),('offset',.008,float),('iters',60,int),
                              ('obstacle',50.,float),('rho',.01,float),('mu',10.,float),
                              ('qd',.01,float),('u',.0001,float),('terminal',200.,float),
                              ('pcg',1e-4,float),('kkt',1e-3,float),('polish',0,int)]:
        p.add_argument('--'+name,type=type_,default=default)
    p.add_argument('--fixed-rho',action='store_true',help='Disable adaptive regularization; retain --rho during each solve.')
    p.add_argument('--output',type=Path,default=ROOT/'example_artifacts/randomized_multimodal/tiago_dev')
    run(p.parse_args())
