"""Independent sampled executed-task checks; no solver-certificate inference."""
import numpy as np


def distribution(values):
    samples = np.asarray(values, dtype=float)
    if not samples.size or not np.isfinite(samples).all():
        return {'count': int(samples.size), 'mean': None, 'median': None, 'p95': None,
                'unmeasured_reason': 'no samples' if not samples.size else 'nonfinite samples retained; no filtered statistic'}
    return {'count': int(samples.size), 'mean': float(np.mean(samples)),
            'median': float(np.median(samples)), 'p95': float(np.percentile(samples, 95))}


def evaluate(protocol, model, x, u, t, stats, error=None):
    import pinocchio as pin
    x, u, t = map(np.asarray, (x, u, t))
    valid = (x.ndim == 2 and x.shape[1:] == (14,) and len(x) >= 1
             and u.shape == (len(x)-1, 7) and t.shape == (len(x),))
    results = {'task_success': False, 'physical_pass': None,
               'solver_numerical_pass': None, 'execution_error': error,
               'unmeasured': {'solver_numerical_pass': 'SB-001/SB-002 and frozen Q02 method-negative retained; no full nonlinear/KKT certificate'},
               'validation': {'recording_shapes': bool(valid)}}
    if not valid:
        results['unmeasured']['physical_pass'] = 'malformed raw execution; shapes retained'
        return results
    finite = bool(np.isfinite(x).all() and np.isfinite(u).all() and np.isfinite(t).all())
    clock = bool(finite and t[0] == 0 and np.all(np.diff(t) > 0)
                 and np.all(np.diff(t) <= protocol['sim_dt_s'] + 1e-9))
    initial = bool(np.allclose(x[0], protocol['initial_state'], atol=1e-6, rtol=0))
    results['validation'].update(finite_execution=finite, recording_clock=clock,
                                 initial_state_matches=initial, recorded_applied_control=bool(len(u)))
    if not finite:
        results['physical_pass'] = False
        results['unmeasured']['task_errors'] = 'nonfinite execution retained in raw NPZ'
        return results
    limits = {'position_violation_rad': float(max(0., np.max(model.lowerPositionLimit-x[:, :7]),
                                                np.max(x[:, :7]-model.upperPositionLimit))),
              'velocity_violation_rad_s': float(max(0., np.max(abs(x[:, 7:])-model.velocityLimit))),
              'torque_violation_Nm': float(max(0., np.max(abs(u)-model.effortLimit))) if len(u) else None}
    if len(u):
        tolerance = protocol['physical_validation']
        results['physical_pass'] = bool(clock and initial
            and limits['position_violation_rad'] <= tolerance['position_tolerance_rad']
            and limits['velocity_violation_rad_s'] <= tolerance['velocity_tolerance_rad_s']
            and limits['torque_violation_Nm'] <= tolerance['effort_tolerance_Nm'])
    else:
        results['unmeasured']['physical_pass'] = 'no executed plant interval'
    data = model.createData()
    frame = model.getFrameId('arm_right_tool_link')

    def pose(q):
        pin.framesForwardKinematics(model, data, q.astype(np.float64))
        placement = data.oMf[frame]
        return placement.translation.copy(), placement.rotation.copy()

    start_p, _ = pose(x[0, :7])
    final_p, final_r = pose(x[-1, :7])
    target = np.asarray(protocol['goal'])
    target_r = pin.rpy.rpyToMatrix(*target[3:])
    distance = float(np.linalg.norm(final_p-target[:3]))
    axis = float(np.rad2deg(np.arccos(np.clip(final_r[:, 2] @ target_r[:, 2], -1, 1))))
    speed = float(np.linalg.norm(x[-1, 7:], ord=1))
    gates = {'position': distance <= protocol['position_threshold_m'],
             'tool_axis': axis <= protocol['axis_threshold_deg'],
             'speed_L1': speed <= protocol['speed_L1_threshold_rad_s'],
             'initial_motion_required': float(np.linalg.norm(start_p-target[:3])) > protocol['position_threshold_m'],
             'positive_bounded_duration': 0 < t[-1] <= protocol['timeout_s']+protocol['dt_s']+1e-8,
             'controller_reached': stats.get('goal_outcomes') == ['reached'],
             'completed_without_exception': error is None,
             'recording_clock': clock, 'initial_state_matches': initial, 'executed_control': bool(len(u))}
    results.update(task_success=bool(all(gates.values())), task_gates=gates,
                   physical_limits=limits, initial_distance_m=float(np.linalg.norm(start_p-target[:3])),
                   final_distance_m=distance, final_axis_error_deg=axis,
                   final_full_SO3_error_deg=float(np.rad2deg(np.linalg.norm(pin.log3(target_r.T@final_r)))),
                   final_L1_speed_rad_s=speed, executed_duration_s=float(t[-1]), plant_steps=len(u))
    return results
