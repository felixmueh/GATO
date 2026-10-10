"""Small active-plan replay helper for the recovered BB portfolio.

Boundary/remainder/control-age logic is adapted from run_mode_b.execute in
python/bsqp/offline_mpc.py at12dc7cc (helper commit2ebd448). No solver or
selection is performed here. Optimizer age is managed by the calling pilot.
"""
import numpy as np
from bsqp.common import sample_packed_trajectory_control


def execute_active_plan(plan, state, start_s, requested_s, activation_s, *,
                        timeout_s, sim_dt, knot_dt, knots, nx, nu, integrate,
                        emit, completed_step, plan_index):
    if not np.isfinite(requested_s) or requested_s <= 0:
        raise ValueError('paid interval must be finite and positive')
    actual_s = min(float(requested_s), max(0., timeout_s-start_s))
    end_s = start_s+actual_s
    time_s = float(start_s)
    state = state.copy()
    while time_s < end_s-1e-12:
        age = max(0., time_s-activation_s)
        h = min(sim_dt, end_s-time_s)
        knot = int(np.floor(age/knot_dt+1e-10))
        if knot < knots-2:
            h = min(h, activation_s+(knot+1)*knot_dt-time_s)
        if h <= 0:
            raise ValueError('nonpositive control-boundary replay step')
        control = sample_packed_trajectory_control(plan,nx,nu,knots,age+1e-12,knot_dt).astype(float)
        record = dict(time_s=time_s, step_dt_s=h, plan_solve_index=plan_index,
                      active_control_age_s=age, state_before=state.copy(), control=control.copy())
        emit('step_started', record)
        try:
            next_state = integrate(state,control,h)
        except BaseException as error:
            emit('step_failed',dict(record,error_type=type(error).__name__,error=str(error)))
            raise
        # Update the owner's last completed trajectory before a recorder can fail.
        completed_step(next_state,control,h,time_s+h,plan_index,age)
        emit('step_completed',dict(record,state_after=next_state.copy()))
        state = next_state
        time_s += h
        if not np.isfinite(state).all():
            emit('step_failed',dict(record,state_after=state.copy(),error_type='FloatingPointError',
                                   error='nonfinite executed plant state'))
            raise FloatingPointError('nonfinite executed plant state; partial trajectory retained')
    return actual_s, actual_s < requested_s
