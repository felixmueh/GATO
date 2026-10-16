#!/usr/bin/env python3
"""Manual experiment: call the existing example with the settings below."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

# Review these settings before motion. All commands run from the checkout root.
BACKEND = "gazebo"  # or "physical_robot" after Gazebo/path/workspace review
PHYSICAL_PATH_AND_WORKSPACE_REVIEWED = False
N_STATE_KNOTS = 32
KNOT_DT_S = 0.03  # prediction spacing; the live loop uses observed time
SIM_DT_S = 0.003  # offline API parameter, not a live feedback scheduler
START_CONFIG = "comfortable_high_clearance"
CONTROLLER_HZ = 125.0
JOINT_STATES_TOPIC = "/joint_states"  # Gazebo default; select measured robot source after inspection
SEA_STATE_TOPIC = "/arm_right_sea_state_broadcaster_controller/sea_joint_states"
SEA_VELOCITY_FILTER_TAU_S = 0.01  # derivative filtering only; absolute position is unfiltered
RESET_DURATION_S = 2.0
RESET_POSITION_TOLERANCE_RAD = 0.05
RESET_SPEED_L1_RAD_S = 0.1
RESET_ARRIVAL_TIMEOUT_S = 5.0
MAX_TORQUE_NM = 26.0  # reject; no clamping
COLLISION_MARGIN_M = 0.04
COLLISION_BLACKLIST = "tiago_configs/comfortable_pose_collision_blacklist.json"
# Existing stale .1 s, startup 30 s, close 8 s and collision-check .05 s defaults.
# Existing joint guards: zero position margin, velocity scale 1; no safety bypass.
RAW_OUTPUT_ROOT = Path("_expr_results/physical-arm")
GOALS_XYZ_M = [[0.4228671666527549, -0.21343649437968426, -0.36151277558663436]]  # torso_lift_link
GOAL_TIMEOUT_S = 8.0
POSITION_TOLERANCE_M = 0.05
ARRIVAL_SPEED_L1_RAD_S = 1.0
# Existing zero dwell/final approach off; orientation off for XYZ goals.
# TIAGO_TRACKING_SOLVER_PARAMS; SQP5/PCG120; B1.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true", help="show settings without ROS/native imports")
    parser.add_argument("--ros-sea-absolute", action="store_true",
                        help="Use SEA absolute position with estimated arm velocity")
    options = parser.parse_args()
    if options.preview:
        settings = {key: str(value) if isinstance(value, Path) else value
                    for key, value in globals().items() if key.isupper()}
        settings["SEA_ABSOLUTE"] = options.ros_sea_absolute
        print(json.dumps(settings, indent=2))
        return
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root / "tiago_src"), str(root / "tiago_examples")]
    from gato_tiago.live_example import check_live_backend
    from native_build import load_profile, verified_receipt
    check_live_backend(BACKEND, physical_reviewed=PHYSICAL_PATH_AND_WORKSPACE_REVIEWED)
    receipt = verified_receipt("tracking")
    if receipt["config"]["cuda_architecture"] != ("87" if BACKEND == "physical_robot" else "120"):
        raise RuntimeError("Prepare the native module for the selected backend first")
    load_profile("tracking")
    print(json.dumps(receipt, indent=2))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = RAW_OUTPUT_ROOT / BACKEND / (stamp + "-preflight_tracking")
    live_options = [
        "--N", str(N_STATE_KNOTS), "--dt", str(KNOT_DT_S), "--sim-dt", str(SIM_DT_S),
        "--tiago-start-config", START_CONFIG, "--ros-tiago",
        "--ros-joint-states-topic", JOINT_STATES_TOPIC,
        "--ros-target-hz", str(CONTROLLER_HZ), "--ros-reset-duration", str(RESET_DURATION_S),
        "--ros-reset-position-tolerance", str(RESET_POSITION_TOLERANCE_RAD),
        "--ros-reset-speed-l1", str(RESET_SPEED_L1_RAD_S),
        "--ros-reset-arrival-timeout", str(RESET_ARRIVAL_TIMEOUT_S),
        "--ros-max-abs-torque", str(MAX_TORQUE_NM),
        "--ros-collision-min-distance", str(COLLISION_MARGIN_M),
        "--ros-collision-blacklist", COLLISION_BLACKLIST,
        "--ros-record-commands", "--ros-confirm-before-motion",
    ]
    if options.ros_sea_absolute:
        live_options += ["--ros-sea-absolute", "--ros-sea-state-topic", SEA_STATE_TOPIC,
                         "--ros-sea-velocity-filter-tau", str(SEA_VELOCITY_FILTER_TAU_S)]
    import tiago_reach_target as example
    from gato_tiago.config import TIAGO_TRACKING_SOLVER_PARAMS
    for index, goal in enumerate(GOALS_XYZ_M, 1):
        # A separate example call creates a NEW controller and comfortable reset.
        args = example.parse_args(["run", *live_options,
            "--goal-timeout", str(GOAL_TIMEOUT_S),
            "--goal-threshold", str(POSITION_TOLERANCE_M),
            "--velocity-threshold", str(ARRIVAL_SPEED_L1_RAD_S),
            "--expr-dir", str(output / f"goal-{index}"),
        ])
        args.native_build = receipt
        args.manual_run_index = RAW_OUTPUT_ROOT / BACKEND / "manual-run-index.jsonl"
        stats = example.run_experiment(args, goals=[goal], solver_params=TIAGO_TRACKING_SOLVER_PARAMS)
        if stats.get("goal_outcomes") != ["reached"]:
            raise RuntimeError("Family preflight did not reach its target; preserve the data and review before tasks")


if __name__ == "__main__":
    main()
