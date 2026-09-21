"""Solver parameters, reference sampling, and integration for both protocols.

Historical behavior comes from benchmark_fig8.py at 4173824. Fixed-duration
experiments share its reference and dynamics while varying prediction spacing
and normalizing running costs. The solver and URDF come from this checkout.
"""
import numpy as np
import pinocchio as pin

from settings import (CONTROL_DT, COST_ANCHOR_KNOTS, INTEGRATION_DT,
                      REFERENCE_DT, prediction_step)

REFERENCE_COMMIT = "4173824"


def solver_parameters(n, horizon_time=None):
    params = dict(max_sqp_iters=1, kkt_tol=.001, max_pcg_iters=100,
                pcg_tol=1e-6, solve_ratio=1., mu=10., q_cost=2.,
                qd_cost=1e-3, u_cost=1e-8 * n, N_cost=20.,
                q_lim_cost=0., vel_lim_cost=0., ctrl_lim_cost=0., rho=.1)
    if horizon_time is not None:
        # One physical running objective at every resolution. Anchor to the
        # historical N32 weights at 10 ms; terminal position cost is unchanged.
        scale = prediction_step(n, horizon_time) / REFERENCE_DT
        params.update(q_cost=2. * scale, qd_cost=1e-3 * scale,
                      u_cost=(1e-8 * COST_ANCHOR_KNOTS) * scale)
    return params


def periodic_reference(reference, times, sample_dt=REFERENCE_DT):
    """Interpolate a shared periodic reference without regenerating its phase."""
    reference = np.asarray(reference).reshape(-1, 6)
    samples = np.remainder(np.asarray(times) / sample_dt, len(reference))
    lower = np.floor(samples).astype(int)
    alpha = (samples - lower)[..., None]
    return (1. - alpha) * reference[lower] + alpha * reference[(lower + 1) % len(reference)]


def advance_plan(model, data, q, dq, plan, n, prediction_dt, duration=CONTROL_DT,
                 max_step=INTEGRATION_DT):
    """Apply piecewise-constant controls, splitting integration at knot boundaries."""
    nx, nu = model.nq + model.nv, model.nv
    elapsed = 0.
    while elapsed < duration - 1e-14:
        control = min(int(np.floor((elapsed + 1e-13) / prediction_dt)), n - 2)
        boundary = (control + 1) * prediction_dt
        step = min(max_step, duration - elapsed, boundary - elapsed)
        if step <= 0:
            raise ValueError('Prediction horizon must cover the control update')
        start = control * (nx + nu) + nx
        q, dq = rk4(model, data, q, dq, plan[start:start + nu], step)
        elapsed += step
        if not np.isfinite(q).all() or not np.isfinite(dq).all():
            break
    return q, dq


def rk4(model, data, q, dq, u, dt):
    """RK4 integration for forward dynamics."""
    # No external forces for this benchmark
    fext = pin.StdVec_Force()
    for _ in range(model.njoints):
        fext.append(pin.Force.Zero())

    # RK4 integration
    k1q = dq
    k1v = pin.aba(model, data, q, dq, u, fext)

    q2 = pin.integrate(model, q, k1q * dt / 2)
    k2q = dq + k1v * dt/2
    k2v = pin.aba(model, data, q2, k2q, u, fext)

    q3 = pin.integrate(model, q, k2q * dt / 2)
    k3q = dq + k2v * dt/2
    k3v = pin.aba(model, data, q3, k3q, u, fext)

    q4 = pin.integrate(model, q, k3q * dt)
    k4q = dq + k3v * dt
    k4v = pin.aba(model, data, q4, k4q, u, fext)

    dq_next = dq + (dt/6) * (k1v + 2*k2v + 2*k3v + k4v)
    avg_dq = (k1q + 2*k2q + 2*k3q + k4q) / 6
    q_next = pin.integrate(model, q, avg_dq * dt)

    return q_next, dq_next
