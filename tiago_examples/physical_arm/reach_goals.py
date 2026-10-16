#!/usr/bin/env python3
"""Three independent comfortable-start reaches. Review these constants, then run manually inside the container."""

# Change only after reviewing the matching Gazebo paths and physical workspace.
BACKEND = "gazebo"  # "gazebo" or "physical_robot"; corresponding ROS profile required
PHYSICAL_PATH_AND_WORKSPACE_REVIEWED = False
TASK_NAME = "reach_goals"
NATIVE_PROFILE = "reaching"
IS_PREFLIGHT = False
RAW_OUTPUT_ROOT = "_expr_results/physical-arm"

# State [rad], time [s], velocity [rad/s], torque [Nm], Cartesian positions [m].
START_Q_RAD = [-0.39, -1.73, -0.38, -2.35, 0.0, -1.21, 0.04]
N_STATE_KNOTS = 32
KNOT_DT_S = 0.03  # 0.93 s prediction window; live update intervals are measured
SIM_DT_S = 0.003  # API parameter; live execution uses observed ROS time
SOLVER_PARAMS = {
    "max_sqp_iters": 5, "max_pcg_iters": 120,
    "kkt_tol": 1e-3, "pcg_tol": 1e-3, "solve_ratio": 1.0,
    "mu": 10.0, "rho": 0.01,
    "q_cost": 4.0, "N_cost": 80.0, "qd_cost": 0.05, "u_cost": 0.0001,
    "ee_orient_cost": 0.0, "ee_orient_N_cost": 0.0,
    "q_lim_cost": 0.02125, "vel_lim_cost": 1.275, "ctrl_lim_cost": 0.02,
}

CONTROLLER_HZ = 125.0  # torque publication rate, separate from joint-state rate
RESET_DURATION_S = 2.0
RESET_POSITION_TOLERANCE_RAD = 0.05
RESET_SPEED_L1_RAD_S = 0.1
RESET_ARRIVAL_TIMEOUT_S = 5.0
STALE_TIMEOUT_S = 0.1
STARTUP_TIMEOUT_S = 30.0
CONTROLLER_TIMEOUT_S = 8.0
MAX_TORQUE_NM = 26.0  # rejection, no torque clamping
COLLISION_MARGIN_M = 0.04
COLLISION_TIMEOUT_S = 0.05
MAX_GEOMETRY_SPEED_M_S = 1.0
JOINT_POSITION_MARGIN_RAD = 0.0
JOINT_VELOCITY_SCALE = 1.0
HISTORY_MAX_RECORDS = 100000
COLLISION_URDF_PATH = "tiago_configs/collision_alpha_wrap_a3_o25_cap001/tiago_pro_collision_alpha_wrap_a3_o25_cap001.urdf"
COLLISION_BLACKLIST_PATH = "tiago_configs/comfortable_pose_collision_blacklist.json"
# Effort-task collision/stale/nonfinite stops and handback stay enabled.
# PAL position reset is recorded; review its path before physical use.
# Final approach off; B1; cold current-state/zero-control seed, then normal warm starts.

GOAL_FRAME = "torso_lift_link"
# T01 FK-derived candidates; qualification with the new costs remains pending.
# No witness warm start is supplied. Each goal gets a NEW comfortable reset.
GOALS_XYZ_M = [[0.4228671666527549, -0.21343649437968426, -0.36151277558663436], [0.5128658736104542, -0.20762798444888964, -0.2941445638456398], [0.5741897462360566, -0.21693072089055046, -0.18550261376089472]]
GOAL_SOURCE_IDS = ['T01-goal-1', 'T01-goal-2', 'T01-goal-3']
GOAL_TIMEOUT_S = 8.0
POSITION_TOLERANCE_M = 0.05
TOOL_AXIS_TOLERANCE_DEG = 30.0  # descriptive only: 3D goals have no axis gate
ARRIVAL_SPEED_L1_RAD_S = 1.0
DWELL_S = 0.0


if __name__ == "__main__":
    from live_task import main
    raise SystemExit(main(globals(), tracking=False))
