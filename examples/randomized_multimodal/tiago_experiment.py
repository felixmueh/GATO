"""Fixed simulated clock TIAGo tool-center multistart transfer experiment.

A single nominal solve first acquires a route. After a fixed two-control prefix,
a cylinder's signed offset is reflected; B1 and B16 start at the identical state
and retain the identical inherited control sequence. Additional B16 proposals
use only current joints and goal-only IK. No robot commands are sent.
"""
from __future__ import annotations
import argparse
import importlib
import hashlib
import subprocess
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares
from tiago_probe import (ROOT, fk, joint_seeds, pack, unpack, metrics,
                         replay_cuda, json_safe, executed_objective)


def make_solver(args, batch):
    module = importlib.import_module(f'bsqp.bsqpN{args.knots}_tiago_right_multimodal')
    solver = getattr(module, f'BSQP_{batch}_float')(
        args.dt, args.iters, args.kkt, 1000, args.pcg, 1., args.mu,
        2., args.qd, args.u, args.terminal, args.obstacle, args.obstacle,
        0., 0., 0., args.rho)
    solver.set_f_ext_batch(np.zeros((batch, 6), np.float32))
    solver.set_rho_adaptation(False)
    return solver


def solve(args, model, x0, qgoal, goal, center, radius, batch, seed, inherited=None):
    solver = make_solver(args, batch)
    x, u = joint_seeds(model, x0, qgoal, args.knots, args.dt, batch, seed, args.amplitude)
    if inherited is not None:
        u[0] = inherited
        x = replay_cuda(solver, x0, u.astype(np.float32), args.dt)
    submitted = pack(x, u)
    ref = np.tile(np.r_[goal, center, radius+args.margin], (batch, args.knots)).astype(np.float32)
    telemetry = []
    z = submitted
    for _ in range(args.passes):
        result = solver.solve(z, args.dt, np.tile(x0, (batch, 1)).astype(np.float32), ref)
        z = np.asarray(result['XU']).copy()
        status = np.asarray(result['pcg_status'])
        telemetry.append(dict(solve_ms=float(result['sqp_time_us'])/1000,
            sqp_iters=np.asarray(result['sqp_iters']).tolist(),
            kkt=np.asarray(result['kkt_converged']).tolist(),
            pcg_cap=int(np.sum(status==0)), pcg_breakdown=int(np.sum(status==2))))
    planned, controls = unpack(z, args.knots)
    replay = replay_cuda(solver, x0, controls.copy(), args.dt)
    rows, xyz = metrics(model, replay, controls, goal, center, radius, args.dt,
                        margin=args.margin, args=args)
    # Only executable replay participates in selection. Physical feasibility
    # includes goal/joint/velocity/effort and densely sampled tool clearance.
    def rank(i):
        r = rows[i]
        return (not r['feasible'], r['objective']['total'])
    winner = min(range(batch), key=rank)
    seed_replay = replay_cuda(solver, x0, u.astype(np.float32), args.dt)
    before, seed_xyz = metrics(model, seed_replay, u, goal, center, radius,
                              args.dt, margin=args.margin, args=args)
    record = dict(batch=batch, seed=seed, winner=winner, lanes=rows, before=before,
                  selected_feasible=rows[winner]['feasible'], any_feasible=any(r['feasible'] for r in rows),
                  telemetry=telemetry, solve_ms=sum(t['solve_ms'] for t in telemetry),
                  defect=np.max(np.abs(replay-planned), axis=(1, 2)).tolist())
    arrays = dict(xyz=xyz, replay=replay, controls=controls, seed_xyz=seed_xyz,
                  seed_controls=u, planned=planned)
    return record, arrays


def shifted(model, solver, x0, controls, shift, dt):
    tail = controls[shift:].copy()
    end = replay_cuda(solver, x0, tail[None].astype(np.float32), dt)[0, -1]
    data = model.createData()
    extra = []
    for _ in range(shift):
        # Terminal joint hold, obstacle-independent; gentle braking is bounded
        # by the dynamics rather than inventing a geometric route.
        a = -end[7:]/max(.15, dt)
        u = pin.rnea(model, data, end[:7].astype(float), end[7:].astype(float), a.astype(float)).copy()
        extra.append(u)
        end = solver.sim_forward(end[None].astype(np.float32), u[None].astype(np.float32), dt)[0]
    return np.vstack([tail, extra])


def scene(model, index, seed, heldout):
    q0 = np.array([-.39, -1.73, -.38, -2.35, 0., -1.21, .04])
    rng = np.random.default_rng(80319+index)
    if heldout:
        q0 += rng.uniform(-.07, .07, 7)
        angle, travel, radius = rng.uniform(-.45, .45), rng.uniform(.10, .15), rng.uniform(.022, .029)
        offset = rng.uniform(.006, .012)*(-1 if index%2 else 1)
    else:
        angle, travel, radius, offset = [(0., .12, .025, .008), (.25, .14, .025, -.01),
                                         (-.3, .105, .022, .006)][index%3]
    data = model.createData()
    start = fk(model, data, q0)
    direction = np.array([np.cos(angle), np.sin(angle)])
    target = start+np.r_[travel*direction, 0.]
    ik = least_squares(lambda q: np.r_[100*(fk(model, data, q)-target), .01*(q-q0)],
                       np.clip(q0, model.lowerPositionLimit+.030001, model.upperPositionLimit-.030001),
                       bounds=(model.lowerPositionLimit+.03, model.upperPositionLimit-.03), max_nfev=200)
    goal = fk(model, data, ik.x)
    normal = np.array([-direction[1], direction[0]])
    middle = .5*(start[:2]+goal[:2])
    return np.r_[q0, np.zeros(7)], ik.x, goal, middle+offset*normal, middle-offset*normal, radius


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    model = pin.buildModelFromUrdf(str(ROOT/'gato/dynamics/tiago_right/tiago_right_arm.urdf'))
    output = dict(config={**vars(args), 'output':str(args.output)},
                  scope='Tool center with 17 interpolated samples/step; no whole-body check or real-time claim.',
                  stationarity='GATO iteration-limited; no KKT certificate is asserted.', trials=[],
                  provenance=dict(git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
                        Path(__file__),ROOT/'examples/randomized_multimodal/tiago_probe.py',
                        ROOT/'gato/dynamics/tiago_right/tiago_right_arm.urdf',
                        Path(importlib.import_module(f'bsqp.bsqpN{args.knots}_tiago_right_multimodal').__file__)]}))
    cached = {}
    if args.resume and (args.output/'summary.json').exists():
        prior = json.loads((args.output/'summary.json').read_text())
        for key,value in prior['config'].items():
            if key not in ('steps','scenes','seeds','output','resume') and getattr(args,key) != value:
                raise ValueError(f'Resume configuration mismatch: {key}')
        cached = {(t['scene'],t['seed']):t for t in prior['trials']}
        output['resumed_provenance'] = prior.get('provenance')
    simulator = make_solver(args, 1)
    for index in range(args.scenes):
        for seed in args.seeds:
            x0, qgoal, goal, before_center, center, radius = scene(model, index, seed, args.heldout)
            stem = f'scene{index}_seed{seed}'
            previous = cached.get((index,seed))
            if previous is None:
                prefix_record, prefix = solve(args, model, x0, qgoal, goal, before_center, radius, 1, seed)
            else:
                prefix_record = previous['prefix']
                prefix = dict(np.load(args.output/f'{stem}_prefix.npz'))
            prior_u = prefix['controls'][0]
            prefix_rows, _ = metrics(model, prefix['replay'][:,:args.advance+1],
                prefix['controls'][:,:args.advance], goal, before_center, radius, args.dt,
                margin=args.margin, args=args)
            pr = prefix_rows[0]
            prefix_safe = bool(pr['finite'] and pr['clearance']>0 and pr['joint_violation']<1e-5
                               and pr['velocity_ratio']<=1 and pr['effort_ratio']<=1)
            common = prefix['replay'][0, args.advance].copy()
            inherited = shifted(model, simulator, common, prior_u, args.advance, args.dt)
            trial = dict(scene=index, seed=seed, goal=goal.tolist(), radius=radius,
                center_before=before_center.tolist(), center_after=center.tolist(),
                common_state=common.tolist(), prefix=prefix_record, prefix_execution=pr,
                prefix_safe=prefix_safe, branches={})
            stem = f'scene{index}_seed{seed}'
            np.savez_compressed(args.output/f'{stem}_prefix.npz', **prefix)
            for batch in (1, args.batch):
                x = common.copy()
                warm = inherited.copy()
                history = [x.copy()]
                executed_u = []
                stages = []
                if previous is not None and str(batch) in previous['branches']:
                    stages = previous['branches'][str(batch)]['stages']
                    for rec in stages:
                        rec['selected_feasible'] = rec['lanes'][rec['winner']]['feasible']
                        rec['any_feasible'] = any(r['feasible'] for r in rec['lanes'])
                    archive = np.load(args.output/f'{stem}_b{batch}_executed.npz')
                    history, executed_u = list(archive['states']), list(archive['controls'])
                    x = history[-1].copy()
                    last = np.load(args.output/f'{stem}_b{batch}_stage{len(stages)-1}.npz')
                    warm = shifted(model, simulator, x, last['controls'][stages[-1]['winner']], args.advance, args.dt)
                for stage in range(len(stages), args.steps):
                    rec, arrays = solve(args, model, x, qgoal, goal, center, radius, batch,
                                        seed+1009*(stage+1), warm)
                    winner = rec['winner']
                    if not rec['lanes'][winner]['finite']:
                        rec['failure'] = 'No finite executable candidate'
                        stages.append(rec)
                        break
                    chosen_u = arrays['controls'][winner]
                    history.extend(arrays['replay'][winner, 1:args.advance+1])
                    executed_u.extend(chosen_u[:args.advance])
                    x = arrays['replay'][winner, args.advance].copy()
                    warm = shifted(model, simulator, x, chosen_u, args.advance, args.dt)
                    rec['stage'] = stage
                    rec['sim_time'] = (stage+1)*args.advance*args.dt
                    stages.append(rec)
                    np.savez_compressed(args.output/f'{stem}_b{batch}_stage{stage}.npz', **arrays)
                    print(json.dumps(dict(scene=index, seed=seed, batch=batch, stage=stage,
                        chosen=winner, feasible=rec['lanes'][winner]['feasible'],
                        cost=rec['lanes'][winner]['objective']['total'],
                        clearance=rec['lanes'][winner]['clearance'], defect=rec['defect'][winner])), flush=True)
                hx, hu = np.asarray(history), np.asarray(executed_u).reshape(-1, 7)
                if len(hu):
                    replay_rows, hxyz = metrics(model, hx[None], hu[None], goal, center, radius,
                        args.dt, margin=args.margin, args=args)
                else:
                    # Preserve a failed trial even if every lane is nonfinite at
                    # its first replan; there is then no executable interval.
                    replay_rows = [stages[-1]['lanes'][stages[-1]['winner']]]
                    hxyz = np.array([[fk(model, model.createData(), hx[0, :7])]])
                branch = dict(stages=stages, executed=replay_rows[0],
                              success=bool(prefix_safe and all(s['selected_feasible'] for s in stages) and replay_rows[0]['feasible'] and replay_rows[0]['settled']),
                              solve_ms=sum(s['solve_ms'] for s in stages))
                trial['branches'][str(batch)] = branch
                np.savez_compressed(args.output/f'{stem}_b{batch}_executed.npz', states=hx, controls=hu, xyz=hxyz[0])
            output['trials'].append(trial)
            (args.output/'summary.json').write_text(json.dumps(json_safe(output), indent=2, allow_nan=False))
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name, default, kind in [('knots',16,int),('dt',.03,float),('batch',16,int),
        ('scenes',3,int),('steps',8,int),('advance',2,int),('amplitude',.1,float),
        ('margin',.008,float),('iters',200,int),('passes',2,int),('obstacle',1.,float),
        ('rho',1.,float),('mu',10.,float),('qd',.01,float),('u',.0001,float),
        ('terminal',200.,float),('pcg',1e-4,float),('kkt',1e-3,float)]:
        p.add_argument('--'+name, type=kind, default=default)
    p.add_argument('--seeds', type=int, nargs='+', default=[100])
    p.add_argument('--heldout', action='store_true')
    p.add_argument('--resume', action='store_true', help='Continue saved episodes to a larger --steps budget.')
    p.add_argument('--output', type=Path, default=ROOT/'example_artifacts/randomized_multimodal/tiago_mpc_dev')
    run(p.parse_args())
