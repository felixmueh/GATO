"""Freeze actual TIAGo one-tick comparisons from the retained timing bank."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import numpy as np
import pinocchio as pin

from .backend import PinocchioPredictor,transform_local,LOCAL_FRAME,LOCAL_POINT
from .estimator import FrozenBatchForceEstimator
from .orchestration import AppliedSegment,TickInputs,WrenchConvention,PredictionRequest

ROOT=Path(__file__).resolve().parents[2]
INPUT_SOURCE='9f2ab6f3dfd324e397fd4d6909f70284558c8d76'


def prepare(output,force_N,previous=ROOT/'experiments/force_adapter/native_inputs'):
    """Regenerate only truth/direction/plant observations; preserve old bank."""
    from .backend import attached_force
    if not np.isfinite(force_N) or force_N<=0:raise ValueError('Explicit selected positive force_N required')
    output=Path(output);output.mkdir(parents=True,exist_ok=False);previous=Path(previous)
    old=json.loads((previous/'protocol.json').read_text())
    with np.load(previous/'arrays.npz',allow_pickle=False) as saved:arrays={k:saved[k].copy() for k in saved.files}
    model=pin.buildModelFromUrdf(str(ROOT/'gato/dynamics/tiago_right/tiago_right_arm.urdf'))
    convention=WrenchConvention(LOCAL_FRAME,LOCAL_POINT,LOCAL_FRAME,LOCAL_POINT)
    intervals=old['control_intervals_s'];history=tuple(AppliedSegment(a,b,u) for (a,b),u in zip(intervals,arrays['controls']))
    truths=[];observed=[];seeds=[]
    for state,reference in zip(arrays['previous_states'],arrays['references']):
        forces=np.stack([np.zeros(6),attached_force(model,state,force_N)])
        tick=TickInputs(old['previous_time_s'],old['observed_time_s'],state,state,history,np.tile(state,(2,1)),np.tile(reference,(2,1)),np.zeros((2,665)))
        measured=PinocchioPredictor(model)(PredictionRequest(tick,forces,convention,transform_local))
        seed=np.zeros((2,16,665),dtype=np.float32)
        for truth_index in range(2):
            for knot in range(32):seed[truth_index,:,knot*21:knot*21+14]=measured[truth_index]
        truths.append(forces);observed.append(measured);seeds.append(seed)
    arrays.update(truths=np.asarray(truths),observed_states=np.asarray(observed),warm_seeds=np.asarray(seeds))
    np.savez(output/'arrays.npz',**arrays)
    task=json.loads((ROOT/'experiments/force_adapter/task_inputs/t01.json').read_text())
    protocol=dict(old,protocol_id='force-persistent-grid-native-v2',params=task['tracking_solver_params'],
        status='new selected-force bank; target measurements pending',selected_force_N=float(force_N),
        truth_direction='attached last-joint force initially aligned world-Z downward, per preserved input state',
        modes=['original','adapted'],truths=['zero','selected_attached_force'],
        estimator=dict(initial_radius=10.,min_radius=1.,max_radius=100.,smoothing_factor=.5,alpha=.6,beta=.5),
        seed_policy='same shifted selected-plan warm seed broadcast to all hypotheses per tick',
        rho_policy='reset each tick; within-solve adaptation',dual_policy='reset after excluded warmup, once before20 measured ticks',
        prediction='existing parallel GRiD TYPE2 simulation, previous measured state and actual-interval time-weighted control',
        E01=dict(derived_from='matched E03 prediction/selection records across both arms; no separate measurements'),
        E03=dict(input_truth_pairs=12,excluded_warmups_per_owner=1,measured_ticks_per_input_method=20,timing_mode='A',
            construction_memory_queries='outside per-tick timers',memory='exact requested persistent device and pinned-host bytes, model/simulation included; boundary snapshots cross-check'),
        original_bank_sha256=hashlib.sha256((previous/'arrays.npz').read_bytes()).hexdigest(),
        limitations='Matched fixed observation bank; optimizer outputs/warm histories can differ between methods; no same-quality or task-outcome claim')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (output/'SHA256SUMS.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (output/'arrays.npz',output/'protocol.json')},indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--force-N',type=float,required=True)
    parser.add_argument('--previous',type=Path,default=ROOT/'experiments/force_adapter/native_inputs');args=parser.parse_args()
    prepare(args.output,args.force_N,args.previous)
