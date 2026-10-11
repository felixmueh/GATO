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


def prepare(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    def frozen(name):return subprocess.check_output(['git','-C',str(ROOT),'show',INPUT_SOURCE+':experiments/batch_timing_campaign/inputs/'+name])
    metadata=json.loads(frozen('bank.json'))
    with np.load(io.BytesIO(frozen('arrays.npz')),allow_pickle=False) as saved:
        starts=saved['starts'].copy();references=saved['references'].copy()
        warmup_state=saved['warmup_state'].copy();warmup_reference=saved['warmup_reference'].copy()
    model=pin.buildModelFromUrdf(str(ROOT/'gato/dynamics/tiago_right/tiago_right_arm.urdf'))
    convention=WrenchConvention(LOCAL_FRAME,LOCAL_POINT,LOCAL_FRAME,LOCAL_POINT)
    controls=np.stack([np.zeros(7),np.ones(7)*.05])
    history=(AppliedSegment(0.,.015,controls[0]),AppliedSegment(.015,.03,controls[1]))
    estimator=FrozenBatchForceEstimator(16,initial_radius=5.,min_radius=2.,max_radius=20.,smoothing_factor=.5)
    hypotheses=estimator.generate_batch().copy()
    truths=np.array([np.zeros(6),np.array([2.,0.,0.,0.,0.,0.])])
    observed=[];previous=[];goals=[];seeds=[]
    for entry,reference in zip(metadata['inputs'],references):
        state=starts[entry['start_index']].astype(float)
        tick=TickInputs(0.,.03,state,state,history,np.tile(state,(2,1)),np.tile(reference,(2,1)),np.zeros((2,665)))
        observed.append(PinocchioPredictor(model)(PredictionRequest(tick,truths,convention,transform_local)))
        previous.append(state);goals.append(reference)
        # Public cold seed policy: current state at every knot, zero controls.
        nominal=np.zeros((2,16,665),dtype=np.float32)
        for truth_index in range(2):
            for knot in range(32):nominal[truth_index,:,knot*21:knot*21+14]=observed[-1][truth_index]
        seeds.append(nominal)
    warmup_tick=TickInputs(0.,.03,warmup_state,warmup_state,history,np.tile(warmup_state,(1,1)),
                          warmup_reference[None,:],np.zeros((1,665)))
    warmup_observed=PinocchioPredictor(model)(PredictionRequest(warmup_tick,truths[:1],convention,transform_local))[0]
    warmup_seed=np.zeros((16,665),dtype=np.float32)
    for knot in range(32):warmup_seed[:,knot*21:knot*21+14]=warmup_observed
    np.savez(output/'arrays.npz',previous_states=np.asarray(previous),observed_states=np.asarray(observed),
             references=np.asarray(goals),warm_seeds=np.asarray(seeds),controls=controls,hypotheses=hypotheses,truths=truths,
             warmup_previous_state=warmup_state,warmup_observed_state=warmup_observed,warmup_reference=warmup_reference,warmup_seed=warmup_seed)
    protocol=dict(protocol_id='force-body-fixed-native-v1',status='proposed_no_measurements',
        input_source_commit=INPUT_SOURCE,input_source_bank_sha256=hashlib.sha256(frozen('bank.json')).hexdigest(),
        inputs=metadata['inputs'],params=metadata['preset'],N=32,dt_s=.03,sim_dt_s=.003,batch_size=16,
        previous_time_s=0.,observed_time_s=.03,control_intervals_s=[[0.,.015],[.015,.03]],
        modes=['original','adapted'],truths=['zero','local_x_2N'],
        frame=LOCAL_FRAME,point=LOCAL_POINT,components='Fx,Fy,Fz,Tx,Ty,Tz',
        estimator=dict(initial_radius=5.,min_radius=2.,max_radius=20.,smoothing_factor=.5,alpha=.6,beta=.5),
        seed_policy='Same public constant-current-state/zero-control cold seed for all force hypotheses; matching selected seed for adapted B1',
        comparison='Original B16 TO then predict/select/update; adapted predict/select/update then selected B1 TO. Correspondence of predictions/selection separate from optimizer output quality.',
        E01=dict(inputs=[0],derived_from='Four measured first-input E03 records, two methods x two truths; no separate GPU cases',timing_mode='A'),
        E03=dict(inputs=list(range(6)),repeats_per_mode=1,excluded_unrelated_warmups_per_worker=1,timing_mode='A',
                 scopes='full callback, constructor, public solve, native solve, observed CUDA free-total boundary deltas and host RSS separately'),
        E02=dict(status='task_protocol_preparation',timing_mode='B',goals='Reuse canonical modest goal and frozen T01 inputs; body-fixed2N disturbance; current rough/physical gates; no actual task run'),
        model_sha256=hashlib.sha256((ROOT/'gato/dynamics/tiago_right/tiago_right_arm.urdf').read_bytes()).hexdigest(),
        limitations='New attached/body-fixed physical scope; Pinocchio CPU forward predictor. No reconstructed historical adapted result, no equal-quality acceleration or constant-world-force claim.')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (output/'SHA256SUMS.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir()) if p.is_file()},indent=2)+'\n')
    return protocol


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();prepare(args.output);print(args.output)
