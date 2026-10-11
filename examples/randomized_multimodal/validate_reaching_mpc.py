"""Shared audited model/raw helpers for current-default portfolio experiments.

Uses MPC_GATO's constructor/BSQP and the shared shift/RK4 implementations.
This offline adapter is necessary to supply full poses, measure SO(3) success,
retain the final actual state, and preserve every solve and executed interval.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import pinocchio as pin

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))
sys.path.insert(0, str(ROOT / 'tiago_src'))
from bsqp.common import initialize_warm_start, rk4, shift_packed_trajectory_warm_start
from gato_tiago.config import TIAGO_TRACKING_SOLVER_PARAMS
from bsqp.mpc_controller import MPC_GATO

MODEL = ROOT / 'gato/dynamics/tiago_right/tiago_right_arm.urdf'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [safe(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(safe(value), indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def pose(model, data, q):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return (data.oMf[model.getFrameId('torso_lift_link')].inverse()
            * data.oMf[model.getFrameId('arm_right_tool_link')]).copy()


def metrics(model, data, x, goal):
    actual = pose(model, data, x[:7])
    angle = float(np.linalg.norm(pin.log3(goal.rotation.T @ actual.rotation)))
    return dict(xyz_error=float(np.linalg.norm(actual.translation-goal.translation)),
                orientation_error_deg=float(np.rad2deg(angle)),
                velocity_l1=float(np.linalg.norm(x[7:], ord=1)))


def state_cost(model, data, x, goal, cfg):
    actual = pose(model, data, x[:7])
    delta_rpy = pin.rpy.matrixToRpy(actual.rotation)-pin.rpy.matrixToRpy(goal.rotation)
    delta_rpy = (delta_rpy+np.pi) % (2*np.pi)-np.pi
    # Physical-time integral of running terms, without horizon terminal charges.
    return np.array([.5*cfg['q_cost']*np.sum((actual.translation-goal.translation)**2),
                     .5*cfg['ee_orient_cost']*np.sum(delta_rpy**2),
                     .5*cfg['qd_cost']*np.sum(x[7:]**2)])


def physical(model, states, controls):
    q, v = states[:, :7], states[:, 7:]
    excess = np.maximum(model.lowerPositionLimit-q, q-model.upperPositionLimit)
    return dict(finite=bool(np.isfinite(states).all() and np.isfinite(controls).all()),
                max_joint_bound_excess_rad=float(max(0., np.max(excess))),
                max_velocity_ratio=float(np.max(np.abs(v)/model.velocityLimit)),
                max_effort_ratio=float(np.max(np.abs(controls)/model.effortLimit)) if controls.size else 0.,
                joint_bounds_valid=bool(np.max(excess) <= 1e-6),
                velocity_valid=bool(np.max(np.abs(v)/model.velocityLimit) <= 1.+1e-6),
                effort_valid=bool(not controls.size or np.max(np.abs(controls)/model.effortLimit) <= 1.+1e-6))



PROTOCOL_PATH = ROOT/'experiments/tiago_batching/protocol.json'
COHORT_ID = 'current-default-batching-A-v1'


def protocol_path(args):
    return Path(getattr(args, 'protocol', None) or PROTOCOL_PATH).resolve()


def cohort_id(protocol):
    return COHORT_ID if protocol['timing_mode'] == 'A' else 'current-default-batching-B-v1'


def load_protocol(path=None):
    protocol = json.loads(Path(path or PROTOCOL_PATH).read_text())
    expected = dict(TIAGO_TRACKING_SOLVER_PARAMS, max_sqp_iters=5, max_pcg_iters=120,
                    qd_cost=.1, vel_lim_cost=.1, ctrl_lim_cost=.05,
                    ee_orient_cost=.1, ee_orient_N_cost=2.)
    if protocol['solver_params'] != expected:
        raise ValueError('Frozen current public pose preset changed')
    for key,value in dict(N=32,knot_dt_s=.03,feedback_s=.03,sim_dt_s=.003,
                          timeout_s=8.,position_tolerance_m=.05,tool_axis_tolerance_deg=30.,
                          velocity_l1_tolerance_rad_s=1.,goal_dwell_s=0.,final_approach=False,
                          orientation_enabled=True).items():
        if protocol.get(key) != value:raise ValueError('Frozen current protocol changed: '+key)
    mode = protocol.get('timing_mode')
    if mode not in ('A', 'B') or protocol.get('offline_timing') != {'A':'controller_dt', 'B':'solve_time'}[mode]:
        raise ValueError('Frozen timing mode differs')
    if mode == 'B':
        original = json.loads(PROTOCOL_PATH.read_text())
        for key in ('task_ids', 'task_count', 'arms', 'repeats', 'input_sha256', 'episode_seed', 'sample_seed', 'solver_core_base'):
            if protocol.get(key) != original[key]:raise ValueError('Frozen BB identity differs: '+key)
        roster = protocol['roster']
        encoded = json.dumps(roster['pair_records'], sort_keys=True, separators=(',', ':')).encode()
        if hashlib.sha256(encoded).hexdigest() != roster['full_ordered_pair_records_sha256']:
            raise ValueError('Frozen BB ordered roster hash differs')
    return protocol


class NativeCapture:
    """References only inside the public solve timer; raw serialization follows."""
    def __init__(self,native):self.native=native;self.reply=None
    def __getattr__(self,name):return getattr(self.native,name)
    def solve(self,*args):
        self.reply=None
        self.reply=self.native.solve(*args)
        return self.reply


def numeric_payload(value,prefix='record'):
    """Retain typed arrays/scalars, including NaN/Inf, without object pickle."""
    result={}
    if isinstance(value,dict):
        for key,item in value.items():result.update(numeric_payload(item,prefix+'__'+str(key)))
    elif value is not None:
        try:array=np.asarray(value)
        except (ValueError,TypeError):return result
        if array.dtype.kind in 'biufcUS':result[prefix]=array
    return result


def solve_recorded(controller,state,reference,candidates,out,step,cfg,timings=None,recorder=None):
    path=out/f'step_{step:04d}'
    def save(kind,**payload):
        if recorder is None:np.savez_compressed(str(path)+'_'+kind+'.npz',**payload)
        else:recorder('solve_'+kind,dict(step=step,**payload))
    raw_started=time.perf_counter() if timings is not None else None
    save('input',actual_state=state,reference=reference,candidates=candidates,
                        caps=np.array([controller.N,len(candidates),cfg['max_sqp_iters'],cfg['max_pcg_iters']]))
    if timings is not None:timings['raw_recording_wall_s'] += time.perf_counter()-raw_started
    started=time.perf_counter();wall=None;returned_saved=False
    try:
        optimized,gpu_us=controller.solver.solve(state,reference,candidates)
        wall=time.perf_counter()-started
        reply=controller.solver.solver.reply
        raw_started=time.perf_counter() if timings is not None else None
        save('returned',optimized=optimized,wall_seconds=wall,gpu_microseconds=gpu_us,
                            **numeric_payload(reply,'native'))
        returned_saved=True
        if timings is not None:timings['raw_recording_wall_s'] += time.perf_counter()-raw_started
        stats_started=time.perf_counter() if timings is not None else None
        stats=controller.solver.get_stats()
        if timings is not None:timings['diagnostics_wall_s'] += time.perf_counter()-stats_started
        raw_started=time.perf_counter() if timings is not None else None
        save('stats',**numeric_payload(stats,'stats'))
        if timings is not None:timings['raw_recording_wall_s'] += time.perf_counter()-raw_started
        return optimized,gpu_us,wall,stats
    except BaseException as error:
        if wall is None:wall=time.perf_counter()-started
        reply=controller.solver.solver.reply
        if reply is not None and not returned_saved:
            save('returned',wall_seconds=wall,**numeric_payload(reply,'native'))
        write_json(Path(str(path)+'_failure.json'),dict(error_type=type(error).__name__,error=str(error),wall_seconds=wall,
                     native_reply_available=reply is not None))
        raise

# Only the audited helpers are transplanted; the old N64/10ms standalone
# validation experiment is deliberately not adopted by this portfolio harness.
