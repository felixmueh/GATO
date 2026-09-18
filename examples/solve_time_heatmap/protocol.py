"""Simulation protocol accompanying the unchanged reference PNG at 4173824.

Adapted from benchmark_fig8.py in that commit. Keep its capped clock, integer
reference offsets, residual integration and unshifted warm starts: changing
those choices changes the workload whose solve times are being measured.
The solver implementation and URDF are supplied by the current checkout.
"""
import numpy as np
import pinocchio as pin

REFERENCE_COMMIT = "4173824"


def solver_parameters(n):
    return dict(max_sqp_iters=1, kkt_tol=.001, max_pcg_iters=100,
                pcg_tol=1e-6, solve_ratio=1., mu=10., q_cost=2.,
                qd_cost=1e-3, u_cost=1e-8 * n, N_cost=20.,
                q_lim_cost=0., vel_lim_cost=0., ctrl_lim_cost=0., rho=.1)


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
