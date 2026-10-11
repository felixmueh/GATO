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

from .backend import NativeOptimizer,PinocchioPredictor,transform_local,LOCAL_FRAME,LOCAL_POINT
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
    output=Path(output);p=case['parameters']
    metrics=dict(campaign_id=p['study'],timing_mode='A',execution_backend='internal',
        evidence_kind='synthetic_test' if optimizer_factory is not None else 'measurement',
        task_success=None,physical_pass=None,solver_numerical_pass=None,execution_error=None,
        outcome='execution_failed',parameters=p)
    with recorder_class(output) as recorder:
        try:
            protocol,arrays=load_inputs(inputs,sha)
            if p['study'] not in ('E01','E03') or p['mode'] not in protocol['modes'] or p['timing_mode']!='A' or p['backend']!='internal':
                raise ValueError('Explicit mode A correspondence/time/memory case required')
            index,truth=int(p['input_index']),int(p['truth_index'])
            write_json(output/'protocol.json',dict(protocol,case=p))
            write_json(output/'provenance.json',dict(source_root=str(ROOT),source_commit=source_commit(),machine=p.get('machine'),module_path=str(module),module_sha256=sha(module),
                bank_manifest_sha256=sha(Path(inputs)/'SHA256SUMS.json'),evidence_kind=metrics['evidence_kind']))
            if optimizer_factory is None:
                import bsqp
                name='bsqp.bsqpN32_tiago_right';spec=importlib.util.spec_from_file_location(name,module)
                native=importlib.util.module_from_spec(spec);sys.modules[name]=native;spec.loader.exec_module(native)
            model=pin.buildModelFromUrdf(str(ROOT/'gato/dynamics/tiago_right/tiago_right_arm.urdf'))
            predictor=PinocchioPredictor(model)
            optimizer=(optimizer_factory or NativeOptimizer)(ROOT/'gato/dynamics/tiago_right/tiago_right_arm.urdf',protocol['params'])
            convention=WrenchConvention(LOCAL_FRAME,LOCAL_POINT,LOCAL_FRAME,LOCAL_POINT)
            def execute(category):
                np.random.seed(20261011)
                class PairEstimator(FrozenBatchForceEstimator):
                    def generate_batch(self):
                        cached=getattr(self,'_tick_candidates',None)
                        return (cached if cached is not None else arrays['hypotheses']).copy()
                estimator=PairEstimator(16,**{k:v for k,v in protocol['estimator'].items() if k in
                    ('initial_radius','min_radius','max_radius','smoothing_factor')})
                def observe(record):
                    record=dict(record,category=category,backend_measurements=optimizer.measurements,
                                native_records=optimizer.native_records)
                    if optimizer.measurements:
                        memory=optimizer.measurements[-1].get('memory')
                        record['memory_bytes']=None if not memory else memory['maximum_observed_additional_device_bytes']
                        record['memory_reason']='Observed device availability boundary delta; no continuous peak. Host RSS separate.'
                    record['rng_state_after']=np.random.get_state()
                    recorder('force_tick',serializable(record))
                # Hypotheses and candidate inputs remain frozen; estimator is fresh
                # per isolated comparison, matching both method workers.
                tick=tick_inputs(protocol,arrays,index,truth,warmup=category=='excluded_warmup')
                recorder('force_tick_inputs',serializable(dict(category=category,mode=p['mode'],inputs=tick,
                    hypotheses=arrays['hypotheses'],rng_state_before=np.random.get_state())))
                recorder.flush()
                return run_tick(p['mode'],tick,estimator=estimator,
                    predictor=predictor,optimizer=optimizer,convention=convention,transform_wrench=transform_local,observer=observe)
            if p['study']=='E03':
                execute('excluded_warmup')
                optimizer.native_records.clear();optimizer.measurements.clear()
            result=execute('measured')
            metrics.update(outcome='force_tick_completed',selected_index=result['selected_index'],
                           timings_s=result['timings_s'],backend_measurements=optimizer.measurements,
                           task_success=None,physical_pass=None)
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
