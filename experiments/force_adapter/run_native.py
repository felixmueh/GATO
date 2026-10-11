"""One actual one-tick force correspondence or time/memory comparison case."""
import argparse
from dataclasses import fields,is_dataclass
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import traceback
import numpy as np
import pinocchio as pin

from .backend import NativeOptimizer,GPUPredictor,transform_local,LOCAL_FRAME,LOCAL_POINT
from .estimator import FrozenBatchForceEstimator
from .orchestration import AppliedSegment,TickInputs,WrenchConvention,run_tick

ROOT=Path(__file__).resolve().parents[2]


def source_commit():
    return subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()


def serializable(value):
    if is_dataclass(value):return {field.name:serializable(getattr(value,field.name)) for field in fields(value)}
    if callable(value):return dict(callable=value.__module__+'.'+value.__qualname__)
    if isinstance(value,dict):return {k:serializable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serializable(v) for v in value]
    return value


def load_inputs(path,sha):
    path=Path(path).resolve();manifest=json.loads((path/'SHA256SUMS.json').read_text())
    for relative,digest in manifest.items():
        target=(path/relative).resolve()
        if not target.is_relative_to(path) or sha(target)!=digest:raise ValueError('Frozen force input bytes differ')
    protocol=json.loads((path/'protocol.json').read_text())
    with np.load(path/'arrays.npz',allow_pickle=False) as saved:arrays={k:saved[k].copy() for k in saved.files}
    return protocol,arrays


def tick_inputs(protocol,arrays,index,truth,warmup=False):
    observed=arrays['warmup_observed_state'] if warmup else arrays['observed_states'][index,truth]
    previous=arrays['warmup_previous_state'] if warmup else arrays['previous_states'][index]
    reference=arrays['warmup_reference'] if warmup else arrays['references'][index]
    seed=arrays['warmup_seed'] if warmup else arrays['warm_seeds'][index,truth]
    intervals=protocol['control_intervals_s']
    return TickInputs(protocol['previous_time_s'],protocol['observed_time_s'],previous,observed,
        tuple(AppliedSegment(a,b,control) for (a,b),control in zip(intervals,arrays['controls'])),
        np.tile(observed,(16,1)),np.tile(reference,(16,1)),seed)


def run_case(case,output,inputs,module,recorder_class,sha,write_json,*,optimizer_factory=None):
    from bsqp.common import shift_packed_trajectory_warm_start
    from .memory import CudaMemory,summarize
    output=Path(output);p=case['parameters']
    metrics=dict(campaign_id='E03',timing_mode='A',execution_backend='internal',parameters=p,
        task_success=None,physical_pass=None,solver_numerical_pass=None,execution_error=None,outcome='execution_failed')
    with recorder_class(output) as recorder:
        try:
            protocol,arrays=load_inputs(inputs,sha)
            if (protocol['protocol_id']!='force-persistent-grid-native-v2' or p['mode'] not in ('original','adapted')
                    or p.get('study')!='E03' or p.get('timing_mode')!='A' or p.get('backend')!='internal'):
                raise ValueError('New persistent GPU force bank and explicit estimator mode required')
            index,truth=int(p['input_index']),int(p['truth_index']);batch=16 if p['mode']=='original' else 1
            write_json(output/'protocol.json',dict(protocol,case=p))
            write_json(output/'provenance.json',dict(source_root=str(ROOT),source_commit=source_commit(),machine=p.get('machine'),
                module_path=str(module),module_sha256=sha(module),bank_manifest_sha256=sha(Path(inputs)/'SHA256SUMS.json')))
            if optimizer_factory is None:
                import bsqp
                name='bsqp.bsqpN32_tiago_right';spec=importlib.util.spec_from_file_location(name,module)
                native=importlib.util.module_from_spec(spec);sys.modules[name]=native;spec.loader.exec_module(native)
            memory=CudaMemory();snapshots=[memory('before_owner')]
            optimizer=(optimizer_factory or NativeOptimizer)(ROOT/'gato/dynamics/tiago_right/tiago_right_arm.urdf',protocol['params'],batch_size=batch)
            predictor=GPUPredictor(optimizer,p['mode']);snapshots.append(memory('after_owner'))
            workspace=dict(device_bytes=int(optimizer.workspace['device_bytes'])+int(predictor.workspace.get('device_bytes',0)),
                pinned_host_bytes=int(optimizer.workspace['pinned_host_bytes'])+int(predictor.workspace.get('pinned_host_bytes',0)),
                optimizer=optimizer.workspace,prediction=predictor.workspace)
            recorder('force_construction',dict(optimizer_wall_s=optimizer.construction_wall_s,prediction_wall_s=predictor.construction_wall_s,
                workspace=workspace,snapshots=snapshots,timing_scope='one-time setup outside all measured algorithm ticks'))
            convention=WrenchConvention(LOCAL_FRAME,LOCAL_POINT,LOCAL_FRAME,LOCAL_POINT)
            class PairEstimator(FrozenBatchForceEstimator):
                def generate_batch(self):
                    cached=getattr(self,'_tick_candidates',None)
                    return (cached if cached is not None else arrays['hypotheses']).copy()
            settings={k:v for k,v in protocol['estimator'].items() if k in ('initial_radius','min_radius','max_radius','smoothing_factor')}
            records=[]
            def execute(category,tick_number,estimator,warm):
                tick=tick_inputs(protocol,arrays,index,truth,warmup=category=='excluded_warmup')
                tick=TickInputs(tick.previous_time_s,tick.observed_time_s,tick.previous_state,tick.observed_state,tick.applied_history,
                    tick.candidate_states,tick.candidate_references,warm)
                recorder('force_tick_inputs',serializable(dict(category=category,tick_number=tick_number,mode=p['mode'],inputs=tick,
                    hypotheses=arrays['hypotheses'],rng_state_before=np.random.get_state())))
                recorder.flush()
                # This saved-input tracking study resets rho for every tick,
                # including the excluded warm-up; duals persist between measured ticks.
                optimizer.reset_rho()
                before=len(optimizer.native_records);measured_before=len(optimizer.measurements)
                def observe(record):
                    payload=dict(record,category=category,tick_number=tick_number,
                        native_records=list(optimizer.native_records[before:]),backend_measurements=list(optimizer.measurements[measured_before:]),
                        rng_state_after=np.random.get_state(),workspace=workspace)
                    actual_truth=np.zeros(6) if category=='excluded_warmup' else arrays['truths'][index,truth]
                    if record.get('selected_force') is not None:
                        payload['estimated_force_error_N']=float(np.linalg.norm(np.asarray(record['selected_force'])[:3]-actual_truth[:3]))
                        payload['estimated_torque_error_Nm']=float(np.linalg.norm(np.asarray(record['selected_force'])[3:]-actual_truth[3:]))
                    recorder('force_tick',serializable(payload))
                    if category=='measured':records.append(record)
                result=run_tick(p['mode'],tick,estimator=estimator,predictor=predictor,optimizer=optimizer,
                    convention=convention,transform_wrench=transform_local,observer=observe)
                optimizer.native_records.clear();optimizer.measurements.clear()
                return result
            np.random.seed(20261011)
            execute('excluded_warmup',-1,PairEstimator(16,**settings),arrays['warmup_seed'])
            optimizer.reset_dual();np.random.seed(20261011);estimator=PairEstimator(16,**settings)
            warm=arrays['warm_seeds'][index,truth].copy()
            for tick_number in range(20):
                result=execute('measured',tick_number,estimator,warm)
                shifted=shift_packed_trajectory_warm_start(result['selected_plan'][None,:],arrays['observed_states'][index,truth],14,7,32,
                    protocol['dt_s'],protocol['dt_s'])
                warm=np.repeat(shifted,16,axis=0)
            snapshots.append(memory('after_measured_ticks'))
            metrics.update(outcome='force_bank_completed',measured_ticks=len(records),excluded_warmups=1,
                selected_indices=[r['selected_index'] for r in records],timings_s=[r['timings_s'] for r in records],
                workspace=workspace,memory=summarize(snapshots),construction_wall_s=optimizer.construction_wall_s+predictor.construction_wall_s,
                comparison_scope='matched observations/hypotheses; native outputs and retained dual/warm histories may differ; no equal quality claim')
        except BaseException:
            metrics['execution_error']=traceback.format_exc();recorder('execution_failed',dict(error=metrics['execution_error']))
        finally:
            write_json(output/'metrics.json',metrics);recorder.manifest()
    return metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('case','output','inputs','build','runner'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();sys.path.insert(0,str(args.runner))
    from recording import RawRecorder,sha,write_json
    modules=list((args.build/'modules').glob('bsqpN32_tiago_right*.so'))
    if len(modules)!=1:raise RuntimeError('One exact force-runtime N32 module required')
    metrics=run_case(json.loads(args.case.read_text()),args.output,args.inputs,modules[0],RawRecorder,sha,write_json)
    return 1 if metrics['execution_error'] else 0


if __name__=='__main__':raise SystemExit(main())
