#!/usr/bin/env python3
"""Manual forward-facing square: four full poses in one sequential reach call."""
import argparse
from datetime import datetime, timezone
import json
from math import pi
from pathlib import Path
import sys

# Review before motion. Run from the checkout root; preview imports no ROS/native code.
BACKEND = "gazebo"  # physical_robot requires actual path/body/tool/workspace review
PHYSICAL_PATH_AND_WORKSPACE_REVIEWED = False
NATIVE_PROFILE = "reaching"  # C2 continuation, .20 rad/.80 velocity/.85 effort insets
N_STATE_KNOTS = 32
BATCH_SIZE = 1
KNOT_DT_S = 0.03  # prediction spacing; live control uses observed time
SIM_DT_S = 0.003  # offline API parameter, not a physical sensing interval
START_CONFIG = "comfortable_high_clearance"
START_Q_RAD = [-0.39, -1.73, -0.38, -2.35, 0.0, -1.21, 0.04]
START_QD_RAD_S = [0.0] * 7
GOAL_FRAME = "torso_lift_link"
TOOL_FRAME = "arm_right_tool_link"
GOAL_NAMES = ["bottom-right", "top-right", "top-left", "bottom-left"]
GOALS_POSES_M_RAD = [
    [0.60, -0.40, -0.20, pi / 2, 0.0, pi / 2],
    [0.60, -0.40, -0.10, pi / 2, 0.0, pi / 2],
    [0.60, -0.30, -0.10, pi / 2, 0.0, pi / 2],
    [0.60, -0.30, -0.20, pi / 2, 0.0, pi / 2],
]  # no center goal, closing reach, or automatic repeats
APPROACH_SOLVER_PARAMS = {
    "max_sqp_iters": 5, "kkt_tol": 0.001, "max_pcg_iters": 120, "pcg_tol": 0.001,
    "solve_ratio": 1.0, "mu": 10.0, "q_cost": 4.0, "N_cost": 80.0,
    "qd_cost": 0.05, "u_cost": 0.0001, "ee_orient_cost": 0.1, "ee_orient_N_cost": 2.0,
    "q_lim_cost": 0.02125, "vel_lim_cost": 1.275, "ctrl_lim_cost": 0.02, "rho": 0.01,
}  # iteration caps; natural zero-PCG stopping, no KKT acceptance gate
FINAL_APPROACH_PARAMS = {
    "q_cost": 800.0, "N_cost": 2400.0, "qd_cost": 1.0, "u_cost": 0.0003,
    "ee_orient_cost": 10.0, "ee_orient_N_cost": 10.0, "max_sqp_iters": 20, "mu": 30.0,
}  # all other approach parameters inherited by the existing switch
FINAL_APPROACH_DISTANCE_M = 0.075
FINAL_APPROACH_MAX_JOINT_SPEED_RAD_S = 0.1
GOAL_TIMEOUT_S = 15.0
POSITION_TOLERANCE_M = 0.01
TOOL_AXIS_TOLERANCE_DEG = 10.0  # full SO(3) error is reported separately
ARRIVAL_SPEED_L1_RAD_S = 0.2
ARRIVAL_DWELL_S = 0.2
ROUGH_POSITION_TOLERANCE_M = 0.05  # descriptive posthoc entry, never the transition gate
ROUGH_TOOL_AXIS_TOLERANCE_DEG = 30.0
ROUGH_SPEED_L1_RAD_S = 1.0
ROUGH_DWELL_S = 0.0
CONTROLLER_HZ = 125.0  # existing effort publication scheduling, not MPC solve rate
JOINT_STATES_TOPIC = "/joint_states"
SEA_STATE_TOPIC = "/arm_right_sea_state_broadcaster_controller/sea_joint_states"
SEA_VELOCITY_FILTER_TAU_S = 0.01  # optional estimated velocity; absolute q is unfiltered
RESET_DURATION_S = 2.0
RESET_POSITION_TOLERANCE_RAD = 0.05
RESET_SPEED_L1_RAD_S = 0.1
RESET_ARRIVAL_TIMEOUT_S = 5.0
STALE_TIMEOUT_S = 0.1
STARTUP_TIMEOUT_S = 30.0
CONTROLLER_TIMEOUT_S = 8.0
MAX_TORQUE_NM = 26.0  # reject; no clamping
COLLISION_MARGIN_M = 0.04
COLLISION_CHECK_TIMEOUT_S = 0.05
COLLISION_MAX_GEOMETRY_SPEED_M_S = 1.0
COLLISION_BLACKLIST = "tiago_configs/comfortable_pose_collision_blacklist.json"
JOINT_POSITION_MARGIN_RAD = 0.0
JOINT_VELOCITY_SCALE = 1.0
RAW_OUTPUT_ROOT = Path("_expr_results/physical-arm")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true", help="show settings without ROS/native imports")
    parser.add_argument("--ros-sea-absolute", action="store_true",
                        help="Use SEA absolute position with estimated arm velocity")
    options = parser.parse_args()
    settings = {key: str(value) if isinstance(value, Path) else value
                for key, value in globals().items() if key.isupper()}
    settings["SEA_ABSOLUTE"] = options.ros_sea_absolute
    print(json.dumps(settings, indent=2))
    if options.preview:
        return
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root / "tiago_src"), str(root / "tiago_examples")]
    from gato_tiago.live_example import check_live_backend
    from gato_tiago.config import TIAGO_RIGHT_START_CONFIGS
    from native_build import load_profile, verified_receipt
    check_live_backend(BACKEND, physical_reviewed=PHYSICAL_PATH_AND_WORKSPACE_REVIEWED)
    # Catch a changed named pose instead of silently changing the reset cohort.
    if any(abs(float(actual) - expected) > 1e-6
           for actual, expected in zip(TIAGO_RIGHT_START_CONFIGS[START_CONFIG], START_Q_RAD)):
        raise RuntimeError("Named start configuration differs from the reviewed square reset")
    receipt = verified_receipt(NATIVE_PROFILE)
    if receipt["config"]["cuda_architecture"] != ("87" if BACKEND == "physical_robot" else "120"):
        raise RuntimeError("Prepare the native module for the selected backend first")
    load_profile(NATIVE_PROFILE)
    print(json.dumps(receipt, indent=2))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = RAW_OUTPUT_ROOT / BACKEND / (stamp + "-square_forward")
    output.mkdir(parents=True, exist_ok=False)
    (output / "launcher_config.json").write_text(json.dumps(settings, indent=2) + "\n")
    final_params_path = output / "final_approach_params.json"
    final_params_path.write_text(json.dumps(FINAL_APPROACH_PARAMS, indent=2) + "\n")
    live_options = [
        "--N", str(N_STATE_KNOTS), "--dt", str(KNOT_DT_S), "--sim-dt", str(SIM_DT_S),
        "--tiago-start-config", START_CONFIG, "--ros-tiago",
        "--ros-joint-states-topic", JOINT_STATES_TOPIC,
        "--ros-target-hz", str(CONTROLLER_HZ), "--ros-reset-duration", str(RESET_DURATION_S),
        "--ros-reset-position-tolerance", str(RESET_POSITION_TOLERANCE_RAD),
        "--ros-reset-speed-l1", str(RESET_SPEED_L1_RAD_S),
        "--ros-reset-arrival-timeout", str(RESET_ARRIVAL_TIMEOUT_S),
        "--ros-stale-timeout", str(STALE_TIMEOUT_S),
        "--ros-startup-timeout", str(STARTUP_TIMEOUT_S),
        "--ros-controller-timeout", str(CONTROLLER_TIMEOUT_S),
        "--ros-max-abs-torque", str(MAX_TORQUE_NM),
        "--ros-collision-min-distance", str(COLLISION_MARGIN_M),
        "--ros-collision-check-timeout", str(COLLISION_CHECK_TIMEOUT_S),
        "--ros-collision-max-monitored-geometry-speed", str(COLLISION_MAX_GEOMETRY_SPEED_M_S),
        "--ros-collision-blacklist", COLLISION_BLACKLIST,
        "--ros-joint-position-margin-rad", str(JOINT_POSITION_MARGIN_RAD),
        "--ros-joint-velocity-scale", str(JOINT_VELOCITY_SCALE),
        "--ros-record-commands", "--ros-confirm-before-motion",
        "--goal-timeout", str(GOAL_TIMEOUT_S), "--goal-threshold", str(POSITION_TOLERANCE_M),
        "--goal-axis-threshold-deg", str(TOOL_AXIS_TOLERANCE_DEG),
        "--velocity-threshold", str(ARRIVAL_SPEED_L1_RAD_S),
        "--goal-dwell-time", str(ARRIVAL_DWELL_S),
        "--final-approach", "--final-approach-params", str(final_params_path),
        "--final-approach-distance", str(FINAL_APPROACH_DISTANCE_M),
        "--final-approach-speed", str(FINAL_APPROACH_MAX_JOINT_SPEED_RAD_S),
        "--expr-dir", str(output),
    ]
    if options.ros_sea_absolute:
        live_options += ["--ros-sea-absolute", "--ros-sea-state-topic", SEA_STATE_TOPIC,
                         "--ros-sea-velocity-filter-tau", str(SEA_VELOCITY_FILTER_TAU_S)]
    import tiago_reach_target as example
    args = example.parse_args(["run", *live_options])
    args.native_build = receipt
    args.manual_run_index = RAW_OUTPUT_ROOT / BACKEND / "manual-run-index.jsonl"
    args.square_forward_config = settings
    # One controller/reset and sequential call; existing goal loop retains warm starts.
    # Existing timeout handling, MOVE, partial recording and handback remain in the example.
    return example.run_experiment(args, goals=GOALS_POSES_M_RAD,
                                  solver_params=APPROACH_SOLVER_PARAMS)


if __name__ == "__main__":
    main()
