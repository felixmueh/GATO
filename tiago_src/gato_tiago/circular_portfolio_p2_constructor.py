"""Capability-blocked reduced shooting constructor for circular portfolio P2."""

from __future__ import annotations

import time

import numpy as np

from gato_tiago.circular_portfolio import BACKEND_OPTIONS, INTERVALS
from gato_tiago.circular_portfolio_constructor import (
    ShootingResult, certify_shooting_result, integrate_acceleration,
    linear_proxy_initial_acceleration, position_and_control_sensitivities,
    proxy_joint_path, reconstruct_controls, shooting_inequalities,
    shooting_inequality_jacobian, shooting_objective,
    shooting_objective_gradient,
)
from gato_tiago.circular_portfolio_p2 import (
    PREFLIGHT_WALL_LIMIT_S, chain_gradient, chain_jacobian,
    certify_affine_map, expand_coefficients, least_squares_initial_coefficients,
    reduced_affine_map,
)


CONSTRUCTOR_EXECUTION_AUTHORIZATION = object()


def execute_reduced_constructor(
    identity, q0, q_goal, tool_reference, lower, upper, velocity, effort,
    pillar, kinematics, rnea, rnea_derivatives, aba, *, wall_limit_s,
    campaign_deadline, monotonic=time.monotonic, authorization=None,
    operational_state=None,
):  # pragma: no cover - separately audited Pin/SciPy boundary
    if (CONSTRUCTOR_EXECUTION_AUTHORIZATION is None
            or authorization is not CONSTRUCTOR_EXECUTION_AUTHORIZATION):
        raise RuntimeError("circular portfolio P2 constructor is blocked")
    from scipy.optimize import BFGS, NonlinearConstraint, minimize
    from scipy.sparse import csr_matrix

    started = monotonic()
    deadline = min(started + float(wall_limit_s), float(campaign_deadline))
    counts = {name: 0 for name in (
        "objective", "gradient", "inequality", "inequality_jacobian",
        "kinematics", "rnea", "rnea_derivatives", "aba",
    )}
    phases = {"proxy_s": 0.0, "affine_s": 0.0, "least_squares_s": 0.0,
              "optimizer_s": 0.0, "postcheck_s": 0.0}
    if operational_state is not None:
        operational_state.clear()
        operational_state.update(phase_timings_s=phases,evaluation_counts=counts)
    def guard():
        if monotonic() >= deadline:
            raise TimeoutError("circular portfolio P2 profile wall limit")
    def counted(name, callback):
        def wrapped(*args):
            guard(); counts[name] += 1; value = callback(*args); guard(); return value
        return wrapped
    fk = counted("kinematics", kinematics)
    inverse_dynamics = counted("rnea", rnea)
    inverse_derivatives = counted("rnea_derivatives", rnea_derivatives)
    forward_dynamics = counted("aba", aba)
    q0 = np.asarray(q0, np.float64); q_goal = np.asarray(q_goal, np.float64)
    phase = monotonic(); proxy = proxy_joint_path(q0, q_goal, tool_reference, fk)
    phases["proxy_s"] = monotonic() - phase; guard()
    phase = monotonic(); affine = reduced_affine_map(q0, q_goal)
    if not certify_affine_map(affine, q0, q_goal)["passes"]:
        raise RuntimeError("P2 affine map failed independent certificate")
    phases["affine_s"] = monotonic() - phase; guard()
    phase = monotonic()
    proxy_initial = linear_proxy_initial_acceleration(q0, q_goal, proxy["proxy_q_float64"])
    initial = least_squares_initial_coefficients(
        proxy_initial["initial_acceleration_float64"], affine
    )
    phases["least_squares_s"] = monotonic() - phase; guard()

    def evaluate(coefficients):
        guard(); qdd = expand_coefficients(coefficients, affine)
        q, qd = integrate_acceleration(q0, np.zeros(7), qdd)
        controls = reconstruct_controls(q, qd, qdd, inverse_dynamics)
        positions, dp, du = position_and_control_sensitivities(
            q, qd, qdd, fk, inverse_derivatives
        )
        guard(); return qdd, q, qd, controls, positions, dp, du
    def objective(coefficients):
        counts["objective"] += 1
        qdd,q,_qd,u,p,_dp,_du=evaluate(coefficients)
        value=shooting_objective(q,qdd,u,p,tool_reference,effort); guard(); return value
    def gradient(coefficients):
        counts["gradient"] += 1
        qdd,_q,_qd,u,p,dp,du=evaluate(coefficients)
        full=shooting_objective_gradient(qdd,u,p,tool_reference,effort,dp,du)
        value=chain_gradient(full,affine); guard(); return value
    def inequality(coefficients):
        counts["inequality"] += 1
        _qdd,q,qd,u,p,_dp,_du=evaluate(coefficients)
        value=shooting_inequalities(q,qd,u,p,lower,upper,velocity,effort,pillar)
        guard(); return value
    def inequality_jacobian(coefficients):
        counts["inequality_jacobian"] += 1
        _qdd,q,qd,u,p,dp,du=evaluate(coefficients)
        full=shooting_inequality_jacobian(q,qd,u,p,pillar,dp,du)
        value=csr_matrix(chain_jacobian(full,affine)); guard(); return value
    def callback(_x,_state=None): guard(); return False

    phase = monotonic()
    result = minimize(
        objective, initial["initial_coefficients_float64"], method="trust-constr",
        jac=gradient, hess=BFGS(),
        constraints=(NonlinearConstraint(
            inequality, 0.0, np.inf, jac=inequality_jacobian
        ),), callback=callback, options=dict(BACKEND_OPTIONS),
    )
    phases["optimizer_s"] = monotonic() - phase; guard()
    phase = monotonic(); qdd,q,qd,u,p,_dp,_du=evaluate(result.x)
    endpoint = np.r_[q[-1]-q_goal, qd[-1]]
    inequalities = inequality(result.x)
    elapsed = monotonic() - started
    shooting = ShootingResult(
        tuple(identity),qdd,q,qd,u,p,float(result.fun),endpoint,inequalities,
        bool(result.success),int(result.status),int(result.niter),float(elapsed),
    )
    shooting_certificate = certify_shooting_result(
        shooting,q0,q_goal,tool_reference,lower,upper,velocity,effort,pillar,
        fk,inverse_dynamics,forward_dynamics,
    )
    phases["postcheck_s"] = monotonic() - phase; elapsed = monotonic()-started; guard()
    return {
        "identity": list(identity), "coefficients_float64": np.asarray(result.x,np.float64),
        "affine": affine, "proxy": proxy, "proxy_initial": proxy_initial,
        "least_squares": initial, "acceleration_float64": qdd,
        "q_float64": q, "qd_float64": qd, "controls_float64": u,
        "positions_float64": p, "endpoint_residual_float64": endpoint,
        "inequalities_float64": inequalities,
        "shooting_objective_float64":np.asarray(result.fun,np.float64),
        "optimizer_success_bool": bool(result.success),
        "optimizer_status_int64": int(result.status),
        "optimizer_iterations_int64": int(result.niter),
        "phase_timings_s": phases, "evaluation_counts": counts,
        "elapsed_s": float(elapsed), "shooting_certificate": shooting_certificate,
        "passes": bool(shooting_certificate["passes"] and elapsed <= wall_limit_s),
    }
