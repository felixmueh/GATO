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
PERIOD_S = 24.0
DURATION_S = 24.0  # one cycle; repeat with a separate manual invocation
# Existing horizontal reference: x span .32 m, y amplitude .08 m, edge .02 m.
# TIAGO_TRACKING_SOLVER_PARAMS; orientation off; SQP5/PCG120; B1.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true", help="show settings without ROS/native imports")
    if parser.parse_args().preview:
        print(json.dumps({key: str(value) if isinstance(value, Path) else value
                          for key, value in globals().items() if key.isupper()}, indent=2))
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
    output = RAW_OUTPUT_ROOT / BACKEND / (stamp + "-figure8_24s")
    live_options = [
        "--N", str(N_STATE_KNOTS), "--dt", str(KNOT_DT_S), "--sim-dt", str(SIM_DT_S),
        "--tiago-start-config", START_CONFIG, "--ros-tiago",
        "--ros-target-hz", str(CONTROLLER_HZ), "--ros-reset-duration", str(RESET_DURATION_S),
        "--ros-reset-position-tolerance", str(RESET_POSITION_TOLERANCE_RAD),
        "--ros-reset-speed-l1", str(RESET_SPEED_L1_RAD_S),
        "--ros-reset-arrival-timeout", str(RESET_ARRIVAL_TIMEOUT_S),
        "--ros-max-abs-torque", str(MAX_TORQUE_NM),
        "--ros-collision-min-distance", str(COLLISION_MARGIN_M),
        "--ros-collision-blacklist", COLLISION_BLACKLIST,
        "--ros-record-commands", "--ros-confirm-before-motion",
    ]
    import tiago_fig8_tracking as example
    args = example.parse_args(live_options + [
        "--plant", "tiago_right", "--batch-sizes", "1", "--no-gif",
        "--figure8-period", str(PERIOD_S), "--sim-time", str(DURATION_S),
        "--output-root", str(output),
    ])
    args.native_build = receipt
    args.manual_run_index = RAW_OUTPUT_ROOT / BACKEND / "manual-run-index.jsonl"
    example.run(args)


if __name__ == "__main__":
    main()
