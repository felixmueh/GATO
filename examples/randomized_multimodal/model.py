"""Independent planar model and obstacle-blind multistart proposals.

The reference optimizer is an offline comparator, never an initializer for GATO.
"""
from dataclasses import asdict, dataclass
from functools import lru_cache
import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class Config:
    knots: int = 32
    dt: float = .08
    q_cost: float = .1
    qd_cost: float = .05
    u_cost: float = .01
    N_cost: float = 200.
    obstacle_cost: float = 100.
    max_sqp_iters: int = 80
    kkt_tol: float = 1e-5
    pcg_tol: float = 1e-6
    max_pcg_iters: int = 200
    mu: float = 20.
    rho: float = .01
    terminal_tolerance: float = .02
    clearance_tolerance: float = 0.
    acceleration_limit: float = 4.
    velocity_limit: float = 2.
    near_best_relative: float = .03


def scene(seed):
    """Fresh scenes vary world frame, travel, offset, size, and initial velocity."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-np.pi, np.pi)
    e = np.array([np.cos(theta), np.sin(theta)])
    n = np.array([-e[1], e[0]])
    midpoint = rng.uniform(-.15, .15, 2)
    distance = rng.uniform(.6, .9)
    start, goal = midpoint - .5 * distance * e, midpoint + .5 * distance * e
    offset = rng.choice([-1, 1]) * rng.uniform(.018, .05)
    center = midpoint + offset * n + rng.uniform(-.05, .05) * e
    velocity = rng.uniform(-.03, .03, 2)
    return dict(scene_id=int(seed), x0=np.r_[start, velocity], goal=goal,
                center=center, radius=float(rng.uniform(.075, .115)), buffer=.015)


@lru_cache(maxsize=32)
def matrices(knots, dt):
    """Position/velocity sensitivities to piecewise constant acceleration."""
    t = np.arange(knots)[:, None]
    j = np.arange(knots - 1)[None, :]
    P = np.where(j < t, (t - j - .5) * dt**2, 0.)
    V = np.where(j < t, dt, 0.)
    return P, V


def rollout(x0, controls, dt):
    u = np.asarray(controls)
    P, V = matrices(u.shape[-2] + 1, dt)
    times = np.arange(u.shape[-2] + 1) * dt
    p = np.asarray(x0)[:2] + times[:, None] * np.asarray(x0)[2:] + P @ u
    v = np.asarray(x0)[2:] + V @ u
    return np.concatenate((p, v), axis=-1)


def seeds(x0, goal, knots, dt, batch_size, proposal_seed, acceleration_limit=4.):
    """Only boundary conditions, dynamics, limits and RNG enter this function.

    Low-frequency uniform coefficients perturb the minimum-energy endpoint
    control in the endpoint map's nullspace. Nested prefixes are batch invariant.
    """
    rng = np.random.default_rng(proposal_seed)
    x0, goal = np.asarray(x0), np.asarray(goal)
    P, V = matrices(knots, dt)
    A = np.stack([P[-1], V[-1]])
    target = np.stack([goal - x0[:2] - (knots - 1)*dt*x0[2:], -x0[2:]])
    nominal = A.T @ np.linalg.solve(A @ A.T, target)
    projector = np.eye(knots - 1) - A.T @ np.linalg.solve(A @ A.T, A)
    time = (np.arange(knots - 1) + .5) / (knots - 1)
    basis = projector @ np.stack([np.sin(k*np.pi*time) for k in range(1, 5)], axis=1)
    controls = [nominal]
    distance = np.linalg.norm(goal - x0[:2])
    while len(controls) < batch_size:
        perturbation = basis @ rng.uniform(-1., 1., (4, 2))
        excursion = np.max(np.linalg.norm(P @ perturbation, axis=1))
        scale = (.2, .4, .7)[((len(controls)-1)//2) % 3] * distance
        perturbation *= scale / max(excursion, 1e-12)
        # Symmetric bound preserves the exact endpoint and antithetic pairing.
        available = np.maximum(acceleration_limit - np.abs(nominal), 0.)
        factor = min(1., float(np.min(available / np.maximum(np.abs(perturbation), 1e-12))))
        perturbation *= factor
        controls.extend([nominal + perturbation, nominal - perturbation])
    u = np.asarray(controls[:batch_size])
    return rollout(x0, u, dt), u


def objective(controls, task, cfg, gradient=False):
    u = np.asarray(controls).reshape(cfg.knots - 1, 2)
    x = rollout(task['x0'], u, cfg.dt)
    weights = np.full(cfg.knots, cfg.q_cost)
    weights[-1] = cfg.N_cost
    error = x[:, :2] - task['goal']
    delta = x[:, :2] - task['center']
    radius = task['radius'] + task['buffer']
    residual = np.maximum(0., 1. - np.sum(delta**2, axis=1)/radius**2)
    cost = .5 * (np.sum(weights[:, None]*error**2) + cfg.qd_cost*np.sum(x[:, 2:]**2)
                 + cfg.u_cost*np.sum(u**2) + cfg.obstacle_cost*np.sum(residual**2))
    if not gradient:
        return float(cost)
    dp = weights[:, None]*error - 2*cfg.obstacle_cost*residual[:, None]*delta/radius**2
    P, V = matrices(cfg.knots, cfg.dt)
    grad = P.T @ dp + cfg.qd_cost*(V.T @ x[:, 2:]) + cfg.u_cost*u
    return float(cost), grad.ravel()


def certify(controls, task, cfg):
    u = np.asarray(controls)
    x = rollout(task['x0'], u, cfg.dt)
    if not np.isfinite(x).all() or not np.isfinite(u).all():
        return dict(cost=1e30,feasible=False,terminal_error=1e30,min_clearance=-1e30,
                    mode='invalid',max_acceleration=1e30,max_velocity=1e30)
    # Exact minimum distance on each quadratic ZOH interval: stationary distance
    # is a cubic in time. Check every real root and both interval endpoints.
    minimum = np.inf
    for state, acc in zip(x[:-1], u):
        d, v = state[:2] - task['center'], state[2:]
        coeff = [.5*np.dot(acc, acc), 1.5*np.dot(v, acc), np.dot(v,v)+np.dot(d,acc), np.dot(d,v)]
        roots = np.roots(np.trim_zeros(coeff, 'f')) if np.any(coeff) else []
        times = [0., cfg.dt] + [float(r.real) for r in roots if abs(r.imag)<1e-9 and 0<r.real<cfg.dt]
        distances = [np.linalg.norm(d+t*v+.5*t*t*acc) for t in times]
        minimum = min(minimum, *distances)
    clearance = float(minimum-task['radius'])
    terminal = float(np.linalg.norm(x[-1,:2]-task['goal']))
    finite = bool(np.isfinite(x).all() and np.isfinite(u).all())
    limited = bool(np.max(np.abs(u)) <= cfg.acceleration_limit+1e-5 and np.max(np.abs(x[:,2:])) <= cfg.velocity_limit+1e-5)
    # Signed accumulated angle is an analysis label; proposals never use it.
    angles = np.unwrap(np.arctan2(x[:,1]-task['center'][1], x[:,0]-task['center'][0]))
    mode = 'ccw' if angles[-1]-angles[0] > 0 else 'cw'
    return dict(cost=objective(u,task,cfg), feasible=bool(finite and limited and clearance>=cfg.clearance_tolerance and terminal<=cfg.terminal_tolerance),
                terminal_error=terminal, min_clearance=clearance, mode=mode,
                max_acceleration=float(np.max(np.abs(u))), max_velocity=float(np.max(np.abs(x[:,2:]))))


def polish(controls, task, cfg):
    u=np.asarray(controls)
    result=minimize(lambda z:objective(z,task,cfg,True),u.ravel(),jac=True,method='L-BFGS-B',
                    options=dict(maxiter=1200,ftol=1e-13,gtol=1e-8),
                    bounds=[(-cfg.acceleration_limit,cfg.acceleration_limit)]*u.size)
    optimized=result.x.reshape(u.shape)
    projected=result.jac.copy()
    projected[(result.x<=-cfg.acceleration_limit+1e-7)&(projected>0)]=0
    projected[(result.x>=cfg.acceleration_limit-1e-7)&(projected<0)]=0
    row=certify(optimized,task,cfg)
    row.update(converged=bool(result.success),gradient_inf=float(np.max(np.abs(result.jac))),
               projected_gradient_inf=float(np.max(np.abs(projected))),iterations=int(result.nit),
               unpolished_cost=objective(u,task,cfg),control_change=float(np.linalg.norm(optimized-u)))
    return row,optimized


def reference(task, cfg, count=64):
    """Generous offline multistart; returns a best-known reference, not a proof."""
    _, initial = seeds(task['x0'], task['goal'], cfg.knots, cfg.dt, count, 170001, cfg.acceleration_limit)
    rows, controls = [], []
    for u in initial:
        row,optimized=polish(u,task,cfg)
        rows.append(row); controls.append(optimized)
    valid = [i for i,r in enumerate(rows) if r['feasible']]
    winner = min(valid,key=lambda i:rows[i]['cost']) if valid else int(np.argmin([r['cost'] for r in rows]))
    return dict(best_known_cost=rows[winner]['cost'] if valid else None, lanes=rows,
                trajectory=rollout(task['x0'],controls[winner],cfg.dt), controls=np.asarray(controls),
                candidate_trajectories=rollout(task['x0'],np.asarray(controls),cfg.dt))


def pack(states, controls):
    return np.concatenate((np.concatenate((states[:,:-1],controls),axis=2).reshape(len(states),-1),states[:,-1]),axis=1).astype(np.float32)


def unpack(packed, cfg):
    z=np.asarray(packed)
    interior=z[:,:-4].reshape(len(z),cfg.knots-1,6)
    return np.concatenate((interior[:,:,:4],z[:,-4:,None].transpose(0,2,1)),axis=1), interior[:,:,4:]


def jsonable(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,dict): return {k:jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [jsonable(v) for v in value]
    return value
