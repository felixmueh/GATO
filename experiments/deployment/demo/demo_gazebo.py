#!/usr/bin/env python3
"""Inspect top-level constants, choose tasks, then run this standalone script."""
from pathlib import Path
import sys

BACKEND = 'gazebo'
TASK_SELECTION = ('square','figure8-24s','figure8-18s','figure8-12s')  # omitted tasks recorded as skipped
REPEATS = 1  # no retry-to-pass
RUNTIME_SOURCE = None  # this deployment checkout
N32_MODULE_PATH = None  # required exact reviewed module path before execution
PREFLIGHT_ONLY = False  # True runs only the unchanged canonical short reach
PREFLIGHT_OUTPUT = None  # required successful same-machine/backend/runtime/module/config attempt
PREVIEW_ONLY = True  # prints configuration/selection; set False for reviewed execution
# Units: xyz[m], rpy/joint angles[rad], velocities[rad/s], duration[s], published torque[Nm].
# Weights are public solver costs/penalties, not hard constraints.
N_STATE_KNOTS = 32
KNOT_DT_S = 0.03
PLANT_SIM_DT_S = 0.003
GOAL_TIMEOUT_S = 15.0
GOALS_POSES_M_RAD = [[0.5, -0.55, -0.15, -4.329780281177467e-17, 0.7853981633974483, 1.5707963267948966],
 [0.5, -0.55, 0.15, -2.356194490192345, -0.0, 0.0],
 [0.5, -0.25, 0.15, 2.356194490192345, -0.0, 0.0],
 [0.5, -0.25, -0.15, 0.7853981633974483, -0.0, 0.0],
 [0.5, -0.4, 0.0, -1.5707963267948966, -0.0, 1.5707963267948966]]
INITIAL_STATE_RAD_RAD_S = [-0.38999998569488525,
 -1.7300000190734863,
 -0.3799999952316284,
 -2.3499999046325684,
 0.0,
 -1.2100000381469727,
 0.03999999910593033,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0]
POSITION_THRESHOLD_M = 0.05
TOOL_AXIS_THRESHOLD_DEG = 30.0
JOINT_SPEED_L1_THRESHOLD_RAD_S = 1.0
DWELL_TIME_S = 0.0  # roughsuccess is zerodwell
SETTLING_STOP = {'position_m': .01, 'tool_axis_deg': 10., 'speed_L1_rad_s': .2, 'dwell_s': .2}  # execution stopping, separately reported
FIGURE8_PERIODS_S = [24.0, 18.0, 12.0]
FIGURE8_SHIFT_M = [0.08, 0.0, 0.0]
FIGURE8_RPY_RAD = [0.0, 0.0, -2.995275773743368]
REACHING_SOLVER_PARAMS = {'max_sqp_iters': 5,
 'kkt_tol': 0.001,
 'max_pcg_iters': 120,
 'pcg_tol': 0.001,
 'solve_ratio': 1.0,
 'mu': 1.0,
 'q_cost': 4.0,
 'qd_cost': 0.1,
 'u_cost': 0.0001,
 'N_cost': 80.0,
 'ee_orient_cost': 0.1,
 'ee_orient_N_cost': 2.0,
 'q_lim_cost': 0.01,
 'vel_lim_cost': 0.1,
 'ctrl_lim_cost': 0.05,
 'rho': 0.01}
TRACKING_SOLVER_PARAMS = {'max_sqp_iters': 5,
 'kkt_tol': 0.001,
 'max_pcg_iters': 120,
 'pcg_tol': 0.001,
 'solve_ratio': 1.0,
 'mu': 1.0,
 'q_cost': 4.0,
 'qd_cost': 0.01,
 'u_cost': 0.0001,
 'N_cost': 80.0,
 'ee_orient_cost': 0.1,
 'ee_orient_N_cost': 2.0,
 'q_lim_cost': 0.01,
 'vel_lim_cost': 0.001,
 'ctrl_lim_cost': 0.003,
 'rho': 0.01}
FINAL_APPROACH_PARAMS = {'q_cost': 400.0,
 'N_cost': 1200.0,
 'qd_cost': 1.0,
 'u_cost': 0.0003,
 'ee_orient_cost': 10.0,
 'ee_orient_N_cost': 10.0,
 'max_sqp_iters': 20,
 'mu': 30.0}
FINAL_APPROACH_TRIGGER = {'distance_m': 0.075, 'max_joint_speed_rad_s': 0.1}
LIVE_CONTROLLER_SETTINGS = {'target_hz': 100.0,
 'reset_duration_sec': 2.0,
 'stale_timeout_sec': 0.1,
 'max_abs_torque': 26.0,
 'clamp_torque': False,
 'disable_collision_safety_for_sim_debug': False,
 'collision_min_distance_m': 0.04,
 'collision_check_timeout_sec': 0.05,
 'collision_max_monitored_geometry_speed_m_s': 1.0,
 'joint_position_margin_rad': 0.0,
 'joint_velocity_scale': 1.0,
 'history_max_records': 20000,
 'start_method': 'spawn',
 'collision_urdf_path': 'tiago_configs/collision_alpha_wrap_a3_o25_cap001/tiago_pro_collision_alpha_wrap_a3_o25_cap001.urdf',
 'collision_blacklist_path': 'tiago_configs/comfortable_pose_collision_blacklist.json',
 'reset_q': [-0.38999998569488525,
             -1.7300000190734863,
             -0.3799999952316284,
             -2.3499999046325684,
             0.0,
             -1.2100000381469727,
             0.03999999910593033]}


def main():
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    import run_demo
    source=Path(RUNTIME_SOURCE) if RUNTIME_SOURCE else run_demo.ROOT
    # The standalone constants and shared frozen input describe exactly the same experiment.
    p=run_demo.resolve(BACKEND,'square')
    assert N_STATE_KNOTS==p['N'] and KNOT_DT_S==p['dt_s'] and GOAL_TIMEOUT_S==p['goal_timeout_s']
    assert GOALS_POSES_M_RAD==[g['pose_m_rad'] for g in p['goals']]
    assert REACHING_SOLVER_PARAMS==p['reaching_solver_params'] and TRACKING_SOLVER_PARAMS==p['tracking_solver_params']
    assert FINAL_APPROACH_PARAMS==p['final_approach_params'] and FINAL_APPROACH_TRIGGER==p['final_approach_trigger']
    assert PLANT_SIM_DT_S==p['sim_dt_s'] and INITIAL_STATE_RAD_RAD_S==p['initial_state']
    assert POSITION_THRESHOLD_M==p['position_threshold_m'] and TOOL_AXIS_THRESHOLD_DEG==p['axis_threshold_deg']
    assert JOINT_SPEED_L1_THRESHOLD_RAD_S==p['speed_L1_threshold_rad_s'] and DWELL_TIME_S==p['goal_dwell_time_s']
    assert SETTLING_STOP==p['reaching_stopping_criteria']
    assert FIGURE8_SHIFT_M==p['tracking_translation_m'] and FIGURE8_RPY_RAD==p['tracking_rpy_rad']
    assert FIGURE8_PERIODS_S==[t['period_s'] for t in p['tracking_tasks']] and LIVE_CONTROLLER_SETTINGS==p['live_controller']
    sys.argv=[__file__,'--backend',BACKEND,'--source',str(source),'--tasks',*TASK_SELECTION]
    if N32_MODULE_PATH:sys.argv+=['--module',str(N32_MODULE_PATH)]
    if PREFLIGHT_ONLY:sys.argv+=['--preflight']
    if PREFLIGHT_OUTPUT:sys.argv+=['--preflight-output',str(PREFLIGHT_OUTPUT)]
    if PREVIEW_ONLY:sys.argv+=['--preview']
    if REPEATS!=1:raise ValueError('This proposed cohort has one pass; discuss changed repeats before execution')
    raise SystemExit(run_demo.main())


if __name__=='__main__':main()
