"""CPU-only FK and sampled inverse-dynamics screening; never runs an optimizer."""
import argparse
import json
from pathlib import Path

import numpy as np

from raw import sha, write_json

ROOT = Path(__file__).resolve().parents[2]


def figure8_speed(period_s):
    # x=.16(1-cos phase), y=.08 sin(2 phase), same public geometry.
    # speed^2=(.16*w)^2 [sin^2 phase + cos^2(2 phase)]. Its maximum
    # occurs at sin^2 phase=1, so maximum speed=.16*w*sqrt(2).
    omega = 2 * np.pi / period_s
    return {'maximum_reference_speed_m_s': float(.16 * omega * np.sqrt(2)),
            'maximum_x_speed_m_s': float(.16 * omega),
            'maximum_y_speed_m_s': float(.16 * omega)}


def propose(output):
    import pinocchio as pin
    urdf = ROOT / 'gato/dynamics/tiago_right/tiago_right_arm.urdf'
    model = pin.buildModelFromUrdf(str(urdf))
    model.gravity.linear = np.array([0., 0., -9.81])
    data = model.createData()
    frame = model.getFrameId('arm_right_tool_link')
    q0 = np.array([-.39, -1.73, -.38, -2.35, 0., -1.21, .04])
    delta = np.array([.12, .4, 0., .15, 0., .05, 0.])
    duration = 8.
    # Include both endpoints; actual spacing <= canonical maximum3ms.
    times = np.linspace(0., duration, int(np.ceil(duration/.003)) + 1)
    s = times/duration
    h = 10*s**3 - 15*s**4 + 6*s**5
    hd = (30*s**2 - 60*s**3 + 30*s**4)/duration
    hdd = (60*s - 180*s**2 + 120*s**3)/duration**2

    def pose(q):
        pin.framesForwardKinematics(model, data, q)
        p = data.oMf[frame]
        return np.r_[p.translation, pin.rpy.matrixToRpy(p.rotation)]

    start_pose = pose(q0)
    goals, witness = [], {}
    for scale in (1, 2, 3):
        displacement = scale*delta
        q = q0 + h[:, None]*displacement
        v = hd[:, None]*displacement
        a = hdd[:, None]*displacement
        u = np.array([pin.rnea(model, data, qi, vi, ai).copy()
                      for qi, vi, ai in zip(q, v, a)])
        target = pose(q0+displacement)
        margins = {'minimum_position_margin_rad': float(min(
            np.min(q-model.lowerPositionLimit), np.min(model.upperPositionLimit-q))),
            'minimum_velocity_margin_rad_s': float(np.min(model.velocityLimit-abs(v))),
            'minimum_effort_margin_Nm': float(np.min(model.effortLimit-abs(u)))}
        goals.append({'id': f'goal-{scale}', 'endpoint_q_rad': (q0+displacement).tolist(),
                      'endpoint_delta_rad': displacement.tolist(),
                      'goal_pose_xyz_rpy': target.tolist(),
                      'initial_distance_m': float(np.linalg.norm(target[:3]-start_pose[:3])),
                      'necessary_endpoint_velocity_time_bound_s': float(np.max(abs(displacement)/model.velocityLimit)),
                      'quintic_witness_duration_s': duration,
                      'sample_count': len(times), 'maximum_sample_dt_s': float(np.max(np.diff(times))),
                      'physical_margins': margins,
                      'sampled_inverse_dynamics_pass': bool(all(vv >= 0 for vv in margins.values())),
                      'maximum_joint_speed_rad_s': np.max(abs(v), axis=0).tolist(),
                      'maximum_abs_torque_Nm': np.max(abs(u), axis=0).tolist(),
                      'outside_5cm_start_region': bool(np.linalg.norm(target[:3]-start_pose[:3]) > .05)})
        witness.update({f'goal_{scale}_q': q, f'goal_{scale}_v': v,
                        f'goal_{scale}_a': a, f'goal_{scale}_u': u})
    witness['time_s'] = times
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/'t01_witness.npz', **witness)
    tracking = []
    for period, rms, peak in ((24., .02, .04), (18., .025, .05), (12., .03, .06)):
        tracking.append({'period_s': period, 'duration_s': period,
                         **figure8_speed(period), 'provisional_RMS_gate_m': rms,
                         'provisional_maximum_gate_m': peak})
    proposal = {'id': 'T01-proposal', 'status': 'goals/speeds approved; true old-plan controller implemented; execution held pending independent controller review/required suite',
                'clock_decision_record': {'speaker':'Felix', 'date':'2026-10-10',
                    'exact_text':'Implement old-plan execution during solve',
                    'execution_hold':'new true old-plan controller implemented; pending independent review and required frozen task suite'},
                'approval_record': {'speaker':'Felix', 'date':'2026-10-10',
                    'exact_text':'Yes but try this first. If this does not work also try with 1ms.',
                    'context':'Three FK-derived goals100.636/207.933/317.052mm with8s allowance; public figure-eight geometry at24/18/12s periods; N32/.03s knots/feedback/.003s plant first; proposed one desktop episode perGPU.',
                    'scope':'goals/speeds and3ms-first protocol approved; conditional1ms is a separate preserved cohort if3ms does not work; Jetson repetitions/discussion still pending'},
                'conditional_plant_variant': {'campaign_id':'T01-conditional-1ms', 'sim_dt_s':.001,
                    'status':'conditional authorization recorded; implementation deferred to next batch',
                    'trigger':'3ms task protocol does not work; preserve/report3ms outcomes first',
                    'separate_protocol_hash_required':True, 'qualifies_canonical_3ms':False},
                'execution_backend': 'internal', 'timing_mode': 'B',
                'offline_timing': 'solve_time', 'machines': ['gtx1070', 'rtx5060', 'jetson'],
                'initial_state': np.r_[q0, np.zeros(7)].tolist(),
                'start_pose_xyz_rpy': start_pose.tolist(), 'reaching': goals,
                'figure8': {'x_span_m': .32, 'y_amplitude_m': .08, 'edge_clearance_m': .02,
                            'anchor_m': start_pose[:3].tolist(), 'speeds': tracking,
                            'orientation_costs': [0., 0.], 'orientation_reference': 'public zero RPY with orientation costs disabled',
                            'solver_params': {'max_sqp_iters': 5, 'kkt_tol': .001, 'max_pcg_iters':120,
                                              'pcg_tol':.001, 'solve_ratio':1., 'mu':1., 'q_cost':4.,
                                              'qd_cost':.01, 'u_cost':.0001, 'N_cost':80.,
                                              'ee_orient_cost':0., 'ee_orient_N_cost':0.,
                                              'q_lim_cost':.01, 'vel_lim_cost':.001,
                                              'ctrl_lim_cost':.003, 'rho':.01}},
                'tracking_gate_rationale': '24s:20mm RMS/40mm maximum leaves margin above historical14.323mm RMS/26.193mm max from one24s fixed-clock1ms/qd.01 run.18s:25/50mm;12s:30/60mm are provisional higher-speed gates, respectively31.25%/37.5% of80mm y amplitude for RMS at18s/12s. These are proposals, not measured B results or claims of physical safety. Felix reviews simulation results before D01/D02.',
                'provisional_tracking_gates_frozen':True,
                'witness_scope': 'sampled quintic q/v/a with Pinocchio inverse-dynamics torques, gravity9.81; no between-sample certificate, collision clearance or tracking feasibility assessment',
                'seed_policy': 'public constant-current-state zero-control seed retained despite screened witness; witness proves promising task parameterization but is not the optimizer seed or tracking reference',
                'seed_justification': 'explicit current campaign compares public task preset behavior, with no initialization tuning; using the witness would change the selected public protocol',
                'protocol': 'N32/.03s knots and feedback/.003s maximum plant/8s per reaching goal/.05m/30deg/speedL1<=1/zero dwell/final approach disabled; all reaching solver settings from inputs/v01.json',
                'required_before_execution': ['Independent review and required frozen suite of new true old-plan-during-solve controller; current execution held', 'Agent freezes provisional tracking gates; Felix reviews gates on simulation results before D01/D02', 'V01 preflight on each family machine/parameterization in B', 'Felix discusses Jetson duration/storage/shorter protocol and repeats before execution'],
                'proposed_repeats': {'desktop': 1, 'jetson': None, 'D02': None},
                'urdf_sha256':sha(urdf), 'generator_sha256':sha(__file__),
                'witness_sha256':sha(output/'t01_witness.npz'),
                'numpy_version':np.__version__, 'pinocchio_version':pin.__version__}
    write_json(output/'t01_proposal.json', proposal)
    return proposal


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(propose(args.output), indent=2))
