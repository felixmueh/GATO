"""Live task and sampled limit summaries; no solver convergence certificate."""
import numpy as np


def distribution(values, nominal_s=None):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        return dict(count=len(values), mean=None, p95=None,
                    unavailable='empty or nonfinite samples; all raw samples retained')
    result = dict(count=len(values), mean=float(np.mean(values)),
                  median=float(np.median(values)), p95=float(np.percentile(values, 95)),
                  maximum=float(np.max(values)))
    if nominal_s is not None:
        result['above_nominal_share'] = float(np.mean(values > nominal_s))
    return result


def task_metrics(protocol, times, positions, rotations, velocities, stats):
    """Compute errors on all observed task states; repeated stamps have zero weight."""
    times, positions, rotations, velocities = map(np.asarray,
        (times, positions, rotations, velocities))
    if (not len(times) or not all(np.isfinite(x).all()
            for x in (times, positions, rotations, velocities)) or np.any(np.diff(times) < 0)):
        return dict(task_success=False, task_unavailable='missing/nonfinite/backward task trace')
    if protocol['kind'] == 'reaching':
        # Pinocchio computes the target rotation in runtime evaluate_live;
        # keep the gate arithmetic usable in CPU-only tests.
        target = np.asarray(protocol['goal'])
        rotation = np.asarray(protocol['target_rotation'])
        position = float(np.linalg.norm(positions[-1] - target[:3]))
        axis = float(np.rad2deg(np.arccos(np.clip(rotations[-1, :, 2] @ rotation[:, 2], -1, 1))))
        # SO(3) angle trace formula records the unconstrained full error too.
        full = float(np.rad2deg(np.arccos(np.clip((np.trace(rotation.T @ rotations[-1]) - 1) / 2, -1, 1))))
        speed = float(np.linalg.norm(velocities[-1], ord=1))
        gates = dict(position=position <= protocol['position_threshold_m'],
            tool_axis=axis <= protocol['axis_threshold_deg'],
            speed_L1=speed <= protocol['speed_L1_threshold_rad_s'],
            initial_outside_success_region=bool(np.linalg.norm(positions[0]-target[:3]) > protocol['position_threshold_m']),
            controller_reached=bool(list(stats.get('goal_outcomes', [])) == ['reached']))
        return dict(task_success=bool(all(gates.values())), task_gates=gates,
            final_distance_m=position, final_axis_error_deg=axis,
            final_full_SO3_error_deg=full, final_L1_speed_rad_s=speed,
            executed_duration_s=float(times[-1]), timeout_lateness_s=max(0., float(times[-1])-protocol['timeout_s']))
    reference = np.asarray(protocol['reference_packed'], dtype=np.dtype(protocol['reference_dtype'])).reshape(-1, 6)
    ref_times = np.arange(len(reference)) * protocol['reference_dt_s']
    target = np.column_stack([np.interp(times, ref_times, reference[:, i]) for i in range(3)])
    errors = np.linalg.norm(positions-target, axis=1)
    duration = float(times[-1] - times[0])
    trapezoid = getattr(np, 'trapezoid', np.trapz)
    rms = float(np.sqrt(trapezoid(errors**2, times) / duration)) if duration > 0 else None
    maximum = float(np.max(errors))
    gates = dict(full_period_completed=bool(times[-1] >= protocol['duration_s']),
        RMS=bool(rms is not None and rms <= protocol['provisional_RMS_gate_m']),
        maximum=maximum <= protocol['provisional_maximum_gate_m'])
    return dict(task_success=bool(all(gates.values())), task_gates=gates,
        tracking_RMS_m=rms, tracking_maximum_m=maximum,
        RMS_definition='time-weighted trapezoidal squared error on all observed task states '
                       'from first feedback to terminal feedback; no startup exclusion; live overshoot retained',
        executed_duration_s=float(times[-1]), end_lateness_s=max(0., float(times[-1])-protocol['duration_s']))


def physical_metrics(protocol, model, positions, velocities, efforts):
    q, qd, tau = map(np.asarray, (positions, velocities, efforts))
    result = dict(physical_pass=None, physical_scope=protocol['physical_scope'],
                  measured_actuator_torque_available=False, full_physical_compliance=None)
    if (not len(q) or not len(qd) or not len(tau)
            or q.shape != qd.shape or q.ndim != 2 or q.shape[1] != 7
            or tau.ndim != 2 or tau.shape[1] != 7):
        result['physical_unavailable'] = 'missing complete observed q/qd or effort-command messages'
        return result
    if not all(np.isfinite(x).all() for x in (q, qd, tau)):
        result['physical_pass'] = False
        result['physical_unavailable'] = 'nonfinite observed state/command retained'
        return result
    limits = dict(position_violation_rad=float(max(0., np.max(model.lowerPositionLimit-q),
                                                    np.max(q-model.upperPositionLimit))),
                  velocity_violation_rad_s=float(max(0., np.max(abs(qd)-model.velocityLimit))),
                  published_effort_violation_Nm=float(max(0., np.max(abs(tau)-model.effortLimit))))
    tol = protocol['physical_validation']
    result.update(physical_limits=limits, physical_pass=bool(
        limits['position_violation_rad'] <= tol['position_tolerance_rad']
        and limits['velocity_violation_rad_s'] <= tol['velocity_tolerance_rad_s']
        and limits['published_effort_violation_Nm'] <= tol['effort_tolerance_Nm']))
    return result


def evaluate_live(protocol, model, controller, observer, stats, recorder, errors=None):
    import pinocchio as pin
    names = [f'arm_right_{i}_joint' for i in range(1, 8)]
    rows, missing = [], 0
    for row in observer.joint_states:
        lookup = {name: i for i, name in enumerate(row['names'])}
        try:
            q = np.asarray([row['position'][lookup[name]] for name in names])
            qd = np.asarray([row['velocity'][lookup[name]] for name in names])
        except (KeyError, IndexError):
            missing += 1
            continue
        stamp = row['stamp']['sec'] + row['stamp']['nanosec'] * 1e-9
        rows.append((stamp, q, qd))
    result = physical_metrics(protocol, model, [r[1] for r in rows], [r[2] for r in rows],
                              [r['data'] for r in observer.efforts])
    result['incomplete_joint_messages'] = missing
    if missing:
        result.update(physical_pass=None, physical_unavailable='incomplete observed joint messages; raw retained')
    if not controller.states:
        result.update(task_success=None, task_unavailable='no public feedback state')
        return result
    origin = controller.states[0]['stamp_sec']
    terminal = controller.states[-1]['stamp_sec']
    # Boundary samples are the actual public feedback observations. Add only
    # independently observed messages strictly inside that task interval.
    task_rows = [(origin, controller.states[0]['q'], controller.states[0]['qd'])]
    task_rows += [r for r in rows if origin < r[0] < terminal]
    if terminal > origin:
        task_rows.append((terminal, controller.states[-1]['q'], controller.states[-1]['qd']))
    data = model.createData()
    tool = model.getFrameId('arm_right_tool_link')
    positions, rotations = [], []
    for _, q, _ in task_rows:
        if np.isfinite(q).all():
            pin.framesForwardKinematics(model, data, np.asarray(q, dtype=np.float64))
            positions.append(data.oMf[tool].translation.copy())
            rotations.append(data.oMf[tool].rotation.copy())
        else:
            positions.append(np.full(3, np.nan)); rotations.append(np.full((3, 3), np.nan))
    times = np.asarray([r[0] - origin for r in task_rows])
    velocities = np.asarray([r[2] for r in task_rows])
    evaluated = dict(protocol)
    if protocol['kind'] == 'reaching':
        evaluated['target_rotation'] = pin.rpy.rpyToMatrix(*protocol['goal'][3:])
    result.update(task_metrics(evaluated, times, positions, rotations, velocities, stats))
    from experiments.deployment.live import record_cleanup_event
    record_cleanup_event(recorder, 'observed_task_trace', dict(time_s=times,
        q=np.asarray([r[1] for r in task_rows]), qd=velocities,
        tool_position=np.asarray(positions), tool_rotation=np.asarray(rotations)),
        errors if errors is not None else [])
    result.update(observed_joint_intervals_s=distribution(np.diff([r[0] for r in rows]), protocol['nominal_feedback_s']),
                  public_feedback_intervals_s=distribution(np.diff([s['stamp_sec'] for s in controller.states]), protocol['nominal_feedback_s']))
    return result
