#!/usr/bin/env python3
"""Freeze original random100 and CPU-prepared cone100 for the SQP5 benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from prepare_cone import SOURCE


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cone-preparation", type=Path, required=True)
    parser.add_argument("--original-study", type=Path, default=Path("/workspace/GATO/_expr_results/random-reaching-off-torque-validation-100-20261015"))
    parser.add_argument("--canonical", type=Path, default=Path("/workspace/GATO/_expr_results/random-reaching-orientation-20261015/OFF_TORQUE/preflight/preflight_effective.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "inputs")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    original_protocol = read(args.original_study / "preparation/validation_protocol.json")
    assignment = read(args.original_study / "preparation/assignment.json")
    random = []
    for pair in assignment["task_ids"]:
        path = args.original_study / f"pilot/pair_{pair:05d}/inputs.npz"
        old_metric = read(path.parent / "metrics.json")
        with np.load(path, allow_pickle=False) as saved:
            row = dict(pair_id=pair, initial_state=saved["initial_state"].tolist(),
                       source_goal_q=saved["source_goal_q"].tolist(),
                       target_xyz=saved["target_xyz"].tolist(),
                       target_rotation=saved["target_rotation"].tolist(),
                       original_reference_float32=saved["reference"].tolist(),
                       source="Exact original frozen random100 input, in its original order; goal q is descriptive FK witness only.",
                       source_input_sha256=sha(path),
                       already_inside_arrival_region=old_metric["initial"]["position_error_m"] <= .05)
            for key in ("initial_state", "source_goal_q", "target_xyz", "target_rotation"):
                if not np.array_equal(np.asarray(row[key]), saved[key]):
                    raise ValueError("JSON conversion changed an original input array")
            if not np.array_equal(np.asarray(row["original_reference_float32"], dtype=np.float32), saved["reference"]):
                raise ValueError("JSON conversion changed the original solver reference")
        random.append(row)
    prepared = read(args.cone_preparation / "targets.json")
    if not (prepared["complete"] and len(prepared["targets"]) == 100 and prepared["half_angle_deg"] == 45.
            and prepared["radius_min_m"] is None and prepared["radius_max_m"] is None):
        raise ValueError("Expected the reviewed45degree radius-free frozen100")
    cone = [dict(pair_id=t["id"], initial_state=prepared["initial_q_rad"] + [0.] * 7,
                 source_goal_q=t["IK_witness_q_rad"], target_xyz=t["xyz_torso_m"],
                 target_rotation=prepared["reference_rotation"],
                 source="Frozen Cartesian-volume cone proposal with bounded position-IK witness; witness is not a seed or joint target.",
                 already_inside_arrival_region=t["already_inside_arrival_region"],
                 source_proposal_id=t["proposal_id"]) for t in prepared["targets"]]
    tasks = dict(schema_version=1, families=dict(random=random, cone=cone),
                 original_random_assignment_sha256=sha(args.original_study / "preparation/assignment.json"),
                 original_pool_sha256=original_protocol["input_sha256"],
                 cone_targets_sha256=sha(args.cone_preparation / "targets.json"),
                 paired_variants=True, trials_per_family_per_arm=100,
                 no_controller_outcome_target_selection=True)
    write(args.output / "tasks.json", tasks)
    shutil.copy2(args.cone_preparation / "targets.json", args.output / "cone_targets.json")
    shutil.copy2(args.cone_preparation / "targets.npz", args.output / "cone_targets.npz")
    shutil.copy2(args.canonical, args.output / "canonical.json")
    names = ["gato/dynamics/tiago_right/tiago_right_plant.cuh",
             "gato/dynamics/tiago_right/tiago_right_barrier.cuh", "python/bindings.cu",
             "gato/dynamics/tiago_right/tiago_right_arm.urdf",
             "gato/dynamics/tiago_right/tiago_right_limits.cuh",
             "gato/dynamics/tiago_right/tiago_right_cost_limits.cuh",
             "gato/bsqp/bsqp.cuh", "python/bsqp/interface.py", "python/bsqp/mpc_controller.py",
             "python/bsqp/common.py", "python/bsqp/config.py", "tiago_src/gato_tiago/config.py",
             "tiago_src/gato_tiago/tiago_mpc_controller.py", "experiments/reaching_sqp5_5060/run_case.py"]
    from run_case import ARMS, FIXED
    for arm, (mode, terminal) in ARMS.items():
        cfg = dict(original_protocol["solver_params"], max_sqp_iters=5, N_cost=terminal)
        protocol = dict(FIXED, id="reaching-sqp5-5060-20261016-" + arm, arm=arm,
                        solver_params=cfg, joint_cost_mode=mode,
                        source_base="b0c1ce13c16a47cd47d8a9838df2350b12b05010",
                        source_sha256={name: sha(SOURCE / name) for name in names},
                        tasks_sha256=sha(args.output / "tasks.json"),
                        canonical_sha256=sha(args.output / "canonical.json"),
                        canonical_timeout_s=30.,
                        canonical_policy="One diagnostic canonical per variant, shared across task families. Preserve arrival and physical negatives; hold dependent work only on incomplete/nonfinite/setup failures.",
                        limit_penalty=original_protocol["limit_penalty"],
                        arrival_policy=original_protocol["arrival_policy"],
                        arrival_orientation_gate=False, orientation_required_for_arrival=False,
                        initialization="Cold repeated actual state/zero controls. Reset dual/rho once at episode start; then retain across feedback and shift the selected ordinary plan.",
                        interpretation="Four paired B1 variants, each100 original random and100 new cone reaches, max30s, SQP5/PCG120, natural zero-PCG stopping. No stationarity certificate or KKT acceptance gate.")
        write(args.output / (arm + ".json"), protocol)
    write(args.output / "FREEZE_RECEIPT.json", dict(
        random_count=len(random), cone_count=len(cone), arms=list(ARMS), scientific_episodes=800,
        diagnostic_canonicals=4, files={p.name: sha(p) for p in sorted(args.output.iterdir())},
        native_solver_calls=0, controller_calls=0, GPU_calls=0))
    print(json.dumps(dict(output=str(args.output), random_count=len(random), cone_count=len(cone), native_solver_calls=0)))


if __name__ == "__main__":
    main()
