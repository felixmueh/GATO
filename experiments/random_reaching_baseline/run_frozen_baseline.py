#!/usr/bin/env python3
"""Frozen OFF_TORQUE validation100 and development SQP-cap studies.

Runtime functions are unchanged from the completed OFF_TORQUE worker. Only
strict preparation accepts the predeclared fresh assignment or SQP caps.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
BASE = "85ef2ea5cf5b96726be6f4b5441f4a7c0ca041b4"
BASELINE_PROTOCOL_SHA256 = "511b99b2a0ed817f0bb881a84ca978e0de5d74d5e046321451db6231fe2c6b8f"
MODEL = "gato/dynamics/tiago_right/tiago_right_arm.urdf"
HISTORICAL_WORKER = "84bad9ad3d88b1b50d01f4d7aa3721bbc31ea210"
FIXED = dict(timing_mode="A", offline_timing="controller_dt", N=32,
             knot_dt_s=.03, feedback_s=.03, sim_dt_s=.003, timeout_s=8.,
             position_tolerance_m=.05, tool_axis_tolerance_deg=30.,
             velocity_l1_tolerance_rad_s=1., goal_dwell_s=0., final_approach=False, batch=1,
             reset_rho_each_solve=False)
PARAMETERS = {"max_sqp_iters", "kkt_tol", "max_pcg_iters", "pcg_tol", "solve_ratio",
              "mu", "q_cost", "qd_cost", "u_cost", "N_cost", "ee_orient_cost",
              "ee_orient_N_cost", "q_lim_cost", "vel_lim_cost", "ctrl_lim_cost", "rho"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe(v) for v in value]
    if hasattr(value, "tolist"):
        return safe(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write(path, value):
    Path(path).write_text(json.dumps(safe(value), indent=2, allow_nan=False) + "\n")


def git(source, *args):
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def prepare(args):
    """Stdlib-only validation: this function never loads a native extension."""
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("source_base") != BASE:
        raise ValueError("This study is pinned to the reviewed interior-cost-bound numerical source")
    for key, value in FIXED.items():
        if protocol.get(key) != value:
            raise ValueError("Fixed initial study setting differs: " + key)
    baseline_path = HERE / 'protocols/OFF_TORQUE-orientation.json'
    if sha(baseline_path) != BASELINE_PROTOCOL_SHA256:
        raise ValueError('Frozen OFF_TORQUE baseline protocol differs')
    baseline = json.loads(baseline_path.read_text())
    if (protocol.get('baseline_protocol_sha256') != BASELINE_PROTOCOL_SHA256 or
            protocol.get('arrival_orientation_gate') is not False or
            protocol.get('orientation_required_for_arrival') is not False or
            protocol.get('arrival_policy') != 'position_and_speed_only'):
        raise ValueError('Frozen study must retain the position+speed arrival gate')
    if protocol.get('orientation_variant') != 'OFF_TORQUE':
        raise ValueError('Only the frozen OFF_TORQUE variant is authorized')
    role = protocol.get('study_role')
    cap = protocol['solver_params']['max_sqp_iters']
    if role not in ('fresh100_validation_only', 'development_sqp_cap'):
        raise ValueError('Only the two declared frozen-baseline studies are supported')
    expected_cfg = dict(baseline['solver_params'])
    if role == 'development_sqp_cap':
        if cap not in (1, 5, 100):
            raise ValueError('Only SQP caps1/5/100 are new; reuse the completed1000 baseline')
        expected_cfg['max_sqp_iters'] = cap
        if protocol['task_ids'] != baseline['task_ids'] or protocol['task_ids_sha256'] != baseline['task_ids_sha256']:
            raise ValueError('SQP study must retain original historical task roster')
        if protocol['run_task_ids'] != baseline['development_task_ids']:
            raise ValueError('SQP study must retain exact ordered development12')
    else:
        assignment_path = Path(protocol['validation_assignment']['path'])
        if sha(assignment_path) != protocol['validation_assignment']['sha256']:
            raise ValueError('Fresh validation assignment hash differs')
        assignment = json.loads(assignment_path.read_text())
        if (protocol['task_ids'] != assignment['task_ids'] or protocol['run_task_ids'] != assignment['task_ids']
                or protocol['task_ids_sha256'] != assignment['task_ids_sha256']
                or len(set(assignment['task_ids'])) != 100
                or set(assignment['task_ids']).intersection(assignment['excluded_task_ids'])
                or assignment['input_sha256'] != baseline['input_sha256']
                or not set(baseline['task_ids']).issubset(assignment['excluded_task_ids'])):
            raise ValueError('Fresh validation assignment differs from declared exclusions/input')
    if protocol['solver_params'] != expected_cfg:
        raise ValueError('Frozen OFF_TORQUE costs/settings differ beyond the declared SQP cap')
    for key in ('source_base', 'input_sha256', 'development_task_ids', 'initialization',
            'limit_penalty', 'native_module_sha256'):
        if protocol.get(key) != baseline[key]:
            raise ValueError('Frozen OFF_TORQUE setting differs: ' + key)
    for key in ('research_baseline_freeze', 'reused_canonical'):
        receipt = protocol[key]
        if sha(receipt['path']) != receipt['sha256']:
            raise ValueError('Frozen reference differs: ' + key)
    if args.preflight_input:
        raise ValueError('These studies openly reuse the recorded failed canonical; no new canonical launch')
    if args.max_sqp_iters is not None:
        raise ValueError('Frozen study uses its explicit SQP protocol; no override')
    cfg = protocol["solver_params"]
    if set(cfg) != PARAMETERS:
        raise ValueError("Provide the complete explicit solver_params dictionary")
    if args.max_sqp_iters is not None:
        cfg["max_sqp_iters"] = args.max_sqp_iters
        protocol["id"] += "-cap" + str(args.max_sqp_iters)
    for key, value in cfg.items():
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError("Invalid solver parameter: " + key)
    for key in ("max_sqp_iters", "max_pcg_iters"):
        if not isinstance(cfg[key], int) or cfg[key] <= 0:
            raise ValueError(key + " must be a positive integer")
    if not 0 < cfg["solve_ratio"] <= 1 or cfg["pcg_tol"] <= 0 or cfg["kkt_tol"] <= 0:
        raise ValueError("Invalid solve ratio or existing solver tolerances")
    if args.output.exists():
        raise ValueError("Output must be a fresh directory: " + str(args.output))
    inputs = {name: sha(args.pool / name) for name in protocol["input_sha256"]}
    if inputs != protocol["input_sha256"]:
        raise ValueError("Frozen historical input hash mismatch")
    roster = protocol["task_ids"]
    if len(roster) != 100 or len(set(roster)) != 100:
        raise ValueError("Protocol must retain the complete frozen100 roster")
    roster_bytes = json.dumps(roster, separators=(",", ":")).encode()
    if hashlib.sha256(roster_bytes).hexdigest() != protocol["task_ids_sha256"]:
        raise ValueError("Frozen ordered task IDs differ")
    pairs = [-1] if args.preflight_input else args.pairs
    if not pairs or len(pairs) != len(set(pairs)) or (not args.preflight_input and any(p not in roster for p in pairs)):
        raise ValueError("Supply distinct explicit --pairs from the frozen100 roster, or --preflight-input")
    if pairs != protocol['run_task_ids']:
        raise ValueError('Run must retain exactly the declared ordered study assignment')
    # Verify actual loaded Python and numerical files against the chosen baseline,
    # allowing experiment-only commits in this dedicated source worktree.
    files = [MODEL, "python/bsqp/interface.py", "python/bsqp/mpc_controller.py", "python/bsqp/common.py",
             "python/bsqp/config.py", "tiago_src/gato_tiago/config.py", "gato/bsqp/bsqp.cuh",
             "tiago_src/gato_tiago/tiago_mpc_controller.py",
             "gato/dynamics/tiago_right/tiago_right_plant.cuh", "gato/dynamics/tiago_right/tiago_right_limits.cuh",
             "gato/dynamics/tiago_right/tiago_right_cost_limits.cuh"]
    source_hashes = {}
    for name in files:
        expected = subprocess.check_output(["git", "-C", str(args.source), "show", BASE + ":" + name])
        source_hashes[name] = sha(args.source / name)
        if source_hashes[name] != hashlib.sha256(expected).hexdigest():
            raise ValueError("Selected baseline source bytes differ: " + name)
    if sha(args.module) != protocol.get("native_module_sha256"):
        raise ValueError("Margin protocol native-module hash mismatch")
    old_helpers = {}
    for name in ("tiago_reaching_batch_pilot.py", "validate_reaching_mpc.py"):
        content = subprocess.check_output(["git", "-C", str(args.source), "show",
                                          HISTORICAL_WORKER + ":examples/randomized_multimodal/" + name])
        old_helpers[name] = hashlib.sha256(content).hexdigest()
    info = dict(protocol=protocol, source=str(args.source), source_commit=git(args.source, "rev-parse", "HEAD"),
                source_sha256=source_hashes, module=str(args.module),
                module_sha256=sha(args.module) if args.module.is_file() else None,
                pool=str(args.pool), inputs_sha256=inputs, pairs=pairs,
                preflight_input=str(args.preflight_input) if args.preflight_input else None,
                preflight_input_sha256=sha(args.preflight_input) if args.preflight_input else None,
                wrapper_sha256=sha(__file__), unchanged_baseline_worker_sha256=sha(HERE / 'run.py'),
                reused_OFF_TORQUE_worker_sha256=sha(HERE / 'run_off_torque.py'),
                study_role=role, research_baseline_freeze=protocol['research_baseline_freeze'],
                reused_canonical=protocol['reused_canonical'],
                arrival_policy='Position <=50mm and speed L1 <=1rad/s at feedback boundary; orientation descriptive only',
                orientation_required_for_arrival=False,
                tool_axis_tolerance_role='30degrees used only for descriptive first_pose_touch_seconds',
                original_modeA_worker_commit=HISTORICAL_WORKER,
                original_modeA_helpers_sha256=old_helpers,
                execution="No plant time advances during solve; first control held for30ms and RK4 integrated at3ms",
                recording="Full executed history; one seed/returned horizon and typed work arrays per solve, no duplicated per-iteration JSON",
                physical_tolerance="historical checks: q excess<=1e-6rad; velocity and effort ratio<=1+1e-6; raw excesses also retained")
    return protocol, info


def load_runtime(source, module):
    """Bind the explicit fresh module before constructing production MPC_GATO."""
    sys.path[:0] = [str(source / "python"), str(source / "tiago_src")]
    import numpy as np
    import pinocchio as pin
    importlib.import_module("bsqp")
    name = "bsqp.bsqpN32_tiago_right"
    spec = importlib.util.spec_from_file_location(name, module)
    native = importlib.util.module_from_spec(spec)
    sys.modules[name] = native
    spec.loader.exec_module(native)
    controller_module = importlib.import_module("gato_tiago.tiago_mpc_controller")
    common = importlib.import_module("bsqp.common")
    for loaded, relative in ((controller_module, "tiago_src/gato_tiago/tiago_mpc_controller.py"), (common, "python/bsqp/common.py")):
        if Path(loaded.__file__).resolve() != source / relative:
            raise ValueError("Imported Python implementation from a different source")
    return np, pin, controller_module.MPC_GATO, common


def unpack(np, packed):
    body = np.asarray(packed)[:31 * 21].reshape(31, 21)
    return np.vstack((body[:, :14], np.asarray(packed)[-14:])), body[:, 14:]


def pose(pin, model, data, q):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return (data.oMf[model.getFrameId("torso_lift_link")].inverse()
            * data.oMf[model.getFrameId("arm_right_tool_link")]).copy()


def metrics(np, pin, model, data, x, goal):
    if not np.isfinite(x).all():
        return dict(position_error_m=float("nan"), tool_axis_error_deg=float("nan"),
                    orientation_error_deg=float("nan"), velocity_l1_rad_s=float("nan"))
    actual = pose(pin, model, data, np.asarray(x[:7], dtype=float))
    return dict(position_error_m=float(np.linalg.norm(actual.translation - goal.translation)),
                tool_axis_error_deg=float(np.rad2deg(np.arccos(np.clip(actual.rotation[:, 2] @ goal.rotation[:, 2], -1., 1.)))),
                orientation_error_deg=float(np.rad2deg(np.linalg.norm(pin.log3(goal.rotation.T @ actual.rotation)))),
                velocity_l1_rad_s=float(np.abs(x[7:]).sum()))


def arrival(observation, protocol):
    return (observation['position_error_m'] <= protocol['position_tolerance_m'] and
            observation['velocity_l1_rad_s'] <= protocol['velocity_l1_tolerance_rad_s'])


def limit_checks(np, model, states, controls, state_times, control_times):
    states, controls = np.asarray(states).reshape(-1, 14), np.asarray(controls).reshape(-1, 7)
    finite = bool(np.isfinite(states).all() and np.isfinite(controls).all())
    result = dict(finite=finite, physical_pass=finite)
    q = states[:, :7]
    excesses = dict(joint=np.maximum(np.maximum(model.lowerPositionLimit-q, q-model.upperPositionLimit), 0.),
                    velocity=np.maximum(np.abs(states[:, 7:])-model.velocityLimit, 0.),
                    effort=np.maximum(np.abs(controls)-model.effortLimit, 0.))
    for name, array in excesses.items():
        times = np.asarray(control_times if name == "effort" else state_times)
        tolerance = 1e-6 if name == "joint" else 1e-6 * (model.velocityLimit if name == "velocity" else model.effortLimit)
        valid = bool(np.isfinite(array).all() and np.all(array <= tolerance))
        result["physical_pass"] &= valid
        violations = np.argwhere(array > tolerance)
        first = None
        if len(violations):
            index, joint = map(int, violations[0])
            first = dict(sample_index=index, time_s=float(times[index]), joint_index=joint,
                         joint_name=str(model.names[joint+1]), excess=float(array[index, joint]))
        maximum = float(np.max(array)) if array.size and np.isfinite(array).all() else (0. if not array.size else float("nan"))
        result[name] = dict(pass_check=valid, maximum_excess=maximum, first_violation=first)
    result["maximum_velocity_ratio"] = float(np.max(np.abs(states[:, 7:])/model.velocityLimit))
    result["maximum_effort_ratio"] = float(np.max(np.abs(controls)/model.effortLimit)) if controls.size else 0.
    return result


def work_summary(np, stats, cap):
    pcg = np.asarray(stats["pcg_iters"]).reshape(-1)
    sqp = int(np.asarray(stats["sqp_iters"]).reshape(-1)[0])
    ls = int(stats["ls_num_iters"])
    zero_exit = bool(pcg.size and pcg[-1] == 0 and ls < sqp)
    # Existing KKT flags are recorded but are never a study acceptance gate.
    reason = "zero_pcg_outer_exit" if zero_exit else "existing_residual_exit" if ls < sqp else "budget_exhausted" if sqp >= cap else "unclassified_existing_exit"
    return dict(actual_sqp=sqp, pcg_iteration_sum=int(pcg.sum()), pcg_count_entries=int(pcg.size),
                last_pcg_iterations=int(pcg[-1]) if pcg.size else None, line_search_count=ls,
                accepted_steps=int(np.count_nonzero(np.asarray(stats["step_size"]) > 0)),
                zero_pcg_outer_exit=zero_exit, stop_reason=reason,
                existing_kkt_flag=int(np.asarray(stats["kkt_converged"]).reshape(-1)[0]),
                initial_merit=float(np.asarray(stats["initial_merit"]).reshape(-1)[0]),
                final_merit=float(np.asarray(stats["final_merit"]).reshape(-1)[0]))


def task(np, pin, model, data, args, pair):
    if args.preflight_input:
        old = json.loads(args.preflight_input.read_text())
        initial = np.asarray(old["initial_state"], dtype=float)
        target = np.asarray(old["goal"], dtype=float)
        goal = pin.SE3(pin.rpy.rpyToMatrix(target[3:]), target[:3])
        witness = initial[:7] + np.asarray(old["endpoint_delta_rad"])
        indices = dict(start_pool_index=None, goal_pool_index=None, source="retained canonical geometry only; effective settings belong to new protocol")
    else:
        with np.load(args.pool / "pool.npz", allow_pickle=False) as pool, np.load(args.pool / "pairs.npz", allow_pickle=False) as pairs:
            si, gi = int(pairs["start_index"][pair]), int(pairs["goal_index"][pair])
            initial = np.r_[pool["q"][si], np.zeros(7)].astype(float)
            witness = pool["q"][gi].astype(float)
        goal = pose(pin, model, data, witness)
        target = np.r_[goal.translation, pin.rpy.matrixToRpy(goal.rotation)]
        indices = dict(start_pool_index=si, goal_pool_index=gi, source="frozen historical pool; goal joint configuration is FK witness only")
    reference = np.tile(target, 32).astype(np.float32)[None]
    return initial, witness, goal, reference, indices


def run_episode(runtime, args, protocol, pair):
    np, pin, MPC, common = runtime
    out = args.output / ("preflight" if pair == -1 else f"pair_{pair:05d}")
    out.mkdir()
    cfg = protocol["solver_params"]
    model = pin.buildModelFromUrdf(str(args.source / MODEL))
    model.gravity.linear = np.array([0., 0., -9.81])
    data = model.createData()
    x, witness, goal, reference, indices = task(np, pin, model, data, args, pair)
    states, controls, times, control_times = [x.copy()], [], [0.], []
    rows, observations = [], [metrics(np, pin, model, data, x, goal)]
    reached = position_touch = pose_touch = None
    failure = None
    failure_phase = None
    interrupted = False
    phase = "controller_setup"
    started = time.perf_counter()
    warm = common.initialize_warm_start(x, 32, 14, 7)[None].astype(np.float32)
    np.savez_compressed(out / "inputs.npz", initial_state=x, source_goal_q=witness,
                        target_xyz=goal.translation, target_rotation=goal.rotation,
                        reference=reference, first_seed=warm)
    write(out / "preparation.json", dict(pair_id=pair, **indices, protocol_id=protocol["id"],
          selected_input_sha256=sha(out / "inputs.npz"), initial_state_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
          goal_pose_sha256=hashlib.sha256(np.r_[goal.translation, goal.rotation.ravel()].tobytes()).hexdigest(),
          initial=observations[0], solver_params=cfg))
    try:
        controller = MPC(model, str(args.source / MODEL), N=32, dt=.03, batch_size=1,
                         plant_type="tiago_right", solver_params=cfg, track_full_stats=True)
        controller.force_estimator = None
        controller.solver.set_f_ext_B(np.zeros((1, 6), dtype=np.float32))
        controller.solver.reset_dual()
        controller.solver.reset_rho()
        phase = "episode"
        step = 0
        with (out / "solves.jsonl").open("w") as log:
            while times[-1] < 8. - 1e-10:
                now, duration = times[-1], min(.03, 8. - times[-1])
                seed = warm.copy()
                sx, su = unpack(np, seed[0])
                grid = now + np.arange(32) * .03
                seed_limits = limit_checks(np, model, sx, su, grid, grid[:-1])
                np.savez_compressed(out / f"step_{step:04d}_input.npz", actual_state=x,
                                    candidates=seed, reference=reference)
                solve_started = time.perf_counter()
                optimized, native_us = controller.solver.solve(x[None], reference, warm.copy())
                wall = time.perf_counter() - solve_started
                stats = controller.solver.get_stats()
                arrays = {name: np.asarray(stats[name]) for name in
                          ("pcg_iters", "sqp_iters", "step_size", "min_merit", "kkt_converged", "initial_merit", "final_merit")}
                np.savez_compressed(out / f"step_{step:04d}_returned.npz", optimized=optimized,
                                    native_solve_us=native_us, solve_wall_s=wall, **arrays)
                predicted_x, predicted_u = unpack(np, optimized[0])
                row = dict(step=step, feedback_time_s=now, executed_duration_s=duration,
                           native_solve_us=float(native_us), solve_wall_s=wall,
                           seed_limits=seed_limits,
                           predicted_limits=limit_checks(np, model, predicted_x, predicted_u, grid, grid[:-1]),
                           work=work_summary(np, stats, cfg["max_sqp_iters"]))
                if not np.isfinite(optimized).all() or not math.isfinite(row["work"]["final_merit"]):
                    rows.append(row); log.write(json.dumps(safe(row), allow_nan=False) + "\n"); log.flush()
                    raise RuntimeError("No finite optimized plan and merit; invalid output retained")
                u = predicted_u[0].astype(float)
                dense, prefix_times = [x.copy()], [now]
                elapsed = 0.
                while elapsed < duration - 1e-12:
                    h = min(.003, duration-elapsed)
                    q, v = common.rk4(model, controller.data, x[:7], x[7:], u, h, controller.actual_f_ext)
                    x = np.r_[q, v]
                    control_times.append(times[-1]); controls.append(u.copy())
                    times.append(times[-1]+h); states.append(x.copy())
                    observations.append(metrics(np, pin, model, data, x, goal))
                    observation = observations[-1]
                    position = observation["position_error_m"] <= .05
                    touching = position and observation["tool_axis_error_deg"] <= 30.
                    if position and position_touch is None:
                        position_touch = times[-1]
                    if touching and pose_touch is None:
                        pose_touch = times[-1]
                    elapsed += h; dense.append(x.copy()); prefix_times.append(times[-1])
                    if not np.isfinite(x).all():
                        break
                row["executed_prefix_limits"] = limit_checks(np, model, dense, [u], prefix_times, [now])
                row["final"] = observations[-1]
                rows.append(row); log.write(json.dumps(safe(row), allow_nan=False) + "\n"); log.flush()
                if not np.isfinite(x).all():
                    raise RuntimeError("Nonfinite executed plant state retained")
                m = observations[-1]
                position = m["position_error_m"] <= .05
                touching = position and m["tool_axis_error_deg"] <= 30.
                if arrival(m, protocol):
                    reached = times[-1]
                    break
                warm = common.shift_packed_trajectory_warm_start(optimized[0], x, 14, 7, 32, duration, .03)[None]
                step += 1
    except BaseException as error:
        failure = traceback.format_exc()
        failure_phase = phase
        interrupted = isinstance(error, (KeyboardInterrupt, SystemExit))
        write(out / "execution_failure.json", dict(error=failure, failure_phase=phase, interrupted=interrupted,
              simulated_seconds=times[-1], completed_solves=len(rows)))
    finally:
        obs = {key: np.asarray([row[key] for row in observations]) for key in observations[0]}
        np.savez_compressed(out / "executed.npz", states=np.asarray(states), controls=np.asarray(controls).reshape(-1, 7),
                            times=np.asarray(times), control_times=np.asarray(control_times), **obs)
        checks = limit_checks(np, model, states, controls, times, control_times)
        indices_last = np.flatnonzero(np.asarray(times) >= max(0., times[-1]-1.))
        last_q = np.asarray(states)[indices_last, :7]
        errors = obs["position_error_m"][indices_last]
        hover = dict(observation_seconds=float(times[-1]-times[indices_last[0]]),
                     position_error_min_m=float(np.min(errors)), position_error_max_m=float(np.max(errors)),
                     net_position_error_reduction_m=float(errors[0]-errors[-1]),
                     joint_total_variation_L1_rad=float(np.abs(np.diff(last_q, axis=0)).sum()),
                     maximum_joint_speed_rad_s=float(np.max(np.abs(np.asarray(states)[indices_last, 7:]))),
                     maximum_speed_L1_rad_s=float(np.max(obs["velocity_l1_rad_s"][indices_last])),
                     limitation="Observation ends at rough arrival or timeout; no post-arrival settling inference")
        initial_xyz = pose(pin, model, data, np.asarray(states[0][:7], dtype=float)).translation
        final_xyz = pose(pin, model, data, np.asarray(states[-1][:7], dtype=float)).translation if np.isfinite(states[-1]).all() else np.full(3, np.nan)
        result = dict(pair_id=pair, outcome="execution_failed" if failure else "reached" if reached is not None else "timeout",
                      complete=True, task_success=reached is not None and failure is None,
                      physical_pass=checks["physical_pass"], execution_error=failure,
                      failure_phase=failure_phase, shared_setup_failure=bool(failure and failure_phase == "controller_setup"),
                      interrupted=interrupted,
                      simulated_seconds=times[-1], reached_seconds=reached,
                      arrival_policy=protocol['arrival_policy'], arrival_orientation_gate=False,
                      orientation_required_for_arrival=False,
                      tool_axis_tolerance_role='30degrees used only for descriptive first_pose_touch_seconds',
                      first_position_touch_seconds=position_touch, first_pose_touch_seconds=pose_touch,
                      touch_sampling="first dense3ms observation; rough arrival checked at30ms feedback boundary",
                      initial=observations[0], final=observations[-1], physical=checks, last_second=hover,
                      executed_endpoint_displacement_m=float(np.linalg.norm(final_xyz-initial_xyz)),
                      feedback_solves=len(rows), actual_sqp_sum=sum(r["work"]["actual_sqp"] for r in rows),
                      actual_pcg_iteration_sum=sum(r["work"]["pcg_iteration_sum"] for r in rows),
                      zero_pcg_exit_solves=sum(r["work"]["zero_pcg_outer_exit"] for r in rows),
                      solve_wall_s=sum(r["solve_wall_s"] for r in rows), native_solve_us=sum(r["native_solve_us"] for r in rows),
                      episode_wall_s=time.perf_counter()-started, timing_mode="A", offline_timing="controller_dt")
        if pair == -1:
            old = json.loads(args.preflight_input.read_text())
            result["canonical_geometry_check"] = observations[0]["position_error_m"] >= old.get("minimum_initial_distance_m", .1)
            result["canonical_motion_check"] = result["executed_endpoint_displacement_m"] >= old.get("minimum_executed_displacement_m", .05)
            result["preflight_pass"] = bool(result["task_success"] and result["physical_pass"] and result["canonical_geometry_check"] and result["canonical_motion_check"])
        write(out / "metrics.json", result)
    print(json.dumps(safe(result), allow_nan=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=HERE / "protocol.json")
    parser.add_argument("--max-sqp-iters", type=int)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pairs", type=int, nargs="+")
    group.add_argument("--preflight-input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate source/configuration/input hashes; no native imports, GPU calls or output writes")
    args = parser.parse_args()
    for key in ("source", "module", "pool", "protocol", "output", "preflight_input"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.resolve())
    protocol, info = prepare(args)
    if args.dry_run:
        print(json.dumps(safe(info), indent=2, allow_nan=False))
        return 0
    if not args.module.is_file():
        parser.error("Native module missing: " + str(args.module))
    args.output.mkdir(parents=True, exist_ok=False)
    info["created_utc"] = datetime.now(timezone.utc).isoformat()
    write(args.output / "provenance.json", info)
    write(args.output / "protocol.json", protocol)
    if args.preflight_input:
        old = json.loads(args.preflight_input.read_text())
        write(args.output / "preflight_effective.json", dict(protocol, initial_state=old["initial_state"],
              goal=old["goal"], endpoint_delta_rad=old["endpoint_delta_rad"],
              reset_rho_each_solve=False, name=protocol["id"] + "-canonical",
              geometry_source_sha256=sha(args.preflight_input), status="new run; original file used for geometry only"))
    try:
        runtime = load_runtime(args.source, args.module)
    except BaseException:
        write(args.output / "setup_failure.json", dict(error=traceback.format_exc()))
        raise
    results = []
    for pair in info["pairs"]:
        result = run_episode(runtime, args, protocol, pair)
        results.append(result)
        write(args.output / "summary.json", results)
        # Constructor/setup failures are shared failures, holding dependent cases.
        if result["shared_setup_failure"] or result["interrupted"]:
            break
    return int(any(r["execution_error"] or (args.preflight_input and not r["preflight_pass"]) for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
