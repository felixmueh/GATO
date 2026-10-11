"""Partial predicted-knot diagnostics; not solver merit/full task qualification."""
import numpy as np
import pinocchio as pin


def assess(model,packed,goal,params):
    body=np.asarray(packed)[:31*21].reshape(31,21)
    states=np.vstack([body[:,:14],np.asarray(packed)[-14:]]);controls=body[:,14:]
    if not np.isfinite(packed).all():return dict(finite=False,physical_knots_pass=None)
    data=model.createData();frame=model.getFrameId('arm_right_tool_link');positions=[]
    for state in states:
        pin.framesForwardKinematics(model,data,state[:7].astype(np.float64));positions.append(data.oMf[frame].translation.copy())
    errors=np.linalg.norm(np.asarray(positions)-goal[:3],axis=1)
    q=states[:,:7];v=states[:,7:]
    excess=dict(q_rad=float(np.max(np.maximum(np.maximum(model.lowerPositionLimit-q,q-model.upperPositionLimit),0))),
        v_rad_s=float(np.max(np.maximum(abs(v)-model.velocityLimit,0))),u_Nm=float(np.max(np.maximum(abs(controls)-model.effortLimit,0))))
    return dict(finite=True,physical_knots_pass=bool(excess['q_rad']<=1e-4 and excess['v_rad_s']<=1e-3 and excess['u_Nm']<=1e-3),
        limit_excess=excess,first_control=controls[0].copy(),terminal_position_error_m=float(errors[-1]),
        partial_running_diagnostic=float(params['q_cost']*sum(errors[:-1]**2)+params['qd_cost']*np.sum(v[:-1]**2)+params['u_cost']*np.sum(controls**2)),
        objective_scope='weighted position/velocity/control squared knot sums only; orientation,terminal,bounds,dynamics,dual merit excluded; not native objective',task_success=None)
