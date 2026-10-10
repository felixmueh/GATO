#!/usr/bin/env python3
"""CPU reference preparation, or explicitly reviewed standalone Q02 native run."""
from __future__ import annotations
import argparse
import importlib.util
import json
import platform
import sys
import time
import traceback
from pathlib import Path
import numpy as np
from protocol import (HERE, ROOT, WEIGHTS, PinReference, read_protocol, sha256, compare,
                      pose_difference, one_step, decode, gn_reference)

def json_write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def verify_manifest():
    manifest=json.loads((HERE/'source_manifest.json').read_text())
    for name,digest in manifest['files'].items():
        if sha256(ROOT/name)!=digest: raise ValueError(f'Frozen source hash mismatch: {name}')
    return manifest

def prepare(output):
    protocol=read_protocol(); manifest=verify_manifest(); ref=PinReference()
    json_write(output/'protocol.json',protocol); json_write(output/'source_manifest.json',manifest)
    inputs=np.array([s['xu'] for s in protocol['samples']],dtype=np.float32)
    goals=np.array([s['goal'] for s in protocol['samples']],dtype=np.float32)
    np.savez(output/'inputs.npz',xu=inputs,goal=goals,gravity=np.array(protocol['gravity_m_s2']))
    # Preserve the physical model bytes, not only a path to a mutable checkout.
    urdf=ROOT/'gato/dynamics/tiago_right/tiago_right_arm.urdf'
    (output/urdf.name).write_bytes(urdf.read_bytes())
    references=[]; setup=[]
    for i,(xu,goal) in enumerate(zip(inputs,goals)):
        xu=xu.astype(float); goal=goal.astype(float)
        start=time.perf_counter(); expected=ref.references(xu,goal,protocol['limits'])
        elapsed=time.perf_counter()-start
        np.savez(output/f'reference-{i:02}.npz',**expected,wall_s=np.array(elapsed))
        finite=all(np.isfinite(a).all() for a in expected.values())
        margins=[]
        for offset,key in [(0,'q'),(7,'v'),(14,'u')]:
            lim=np.array(protocol['limits'][key]); z=xu[offset:offset+7]
            margin=float(np.min(np.minimum(z-lim[:,0],lim[:,1]-z)))
            margins.append(margin)
        qlim=np.array(protocol['limits']['q']); vlim=np.array(protocol['limits']['v']); step=expected['step']
        step_margin=min(float(np.min(np.minimum(step[:7]-qlim[:,0],qlim[:,1]-step[:7]))),
                        float(np.min(np.minimum(step[7:]-vlim[:,0],vlim[:,1]-step[7:]))))
        nonsingular=float(abs(np.cos(expected['pose'][4])))
        setup.append(dict(sample=i,finite=bool(finite),min_input_margin=min(margins),min_step_qv_margin=step_margin,
                          abs_cos_pitch=nonsingular,pass_=bool(finite and min(margins)>protocol['native_fd_step'] and step_margin>0 and nonsingular>.1),
                          reference_wall_s=elapsed))
        references.append(expected)
    json_write(output/'cpu_setup.json',dict(status='PASS' if all(s['pass_'] for s in setup) else 'FAIL',samples=setup,
        python=sys.version,numpy=np.__version__,pinocchio=ref.pin.__version__,platform=platform.platform(),
        excluded_model_terms='URDF damping/friction; both ABA and generated rigid-body dynamics omit them',
        claim='Independent CPU references only; native qualification pending'))
    return protocol,inputs,goals,references,all(s['pass_'] for s in setup)

def append_event(path,record):
    with path.open('a') as f:
        f.write(json.dumps(record,allow_nan=False)+'\n')
        f.flush()


def timing_record(value,unavailable_reason='not_returned'):
    if value is None: return None,unavailable_reason
    try:
        scalar=float(value)
    except (TypeError,ValueError):
        return None,'invalid_scalar'
    if not np.isfinite(scalar):
        return None,'nonfinite_nan' if np.isnan(scalar) else 'nonfinite_infinity'
    return scalar,None


def run_native(output,module_path,protocol,inputs,goals,references):
    # Archive the complete frozen cohort before loading native code or creating
    # device resources. These are attempted inputs, never fabricated outputs.
    np.savez(output/'native_attempted_inputs.npz',xu=np.asarray(inputs,dtype=np.float32),
             goal=np.asarray(goals,dtype=np.float32))
    setup_log=output/'native_setup.jsonl'
    setup_start=time.perf_counter()
    setup=dict(module_path=str(module_path),input_raw='native_attempted_inputs.npz',
               input_sha256=sha256(output/'native_attempted_inputs.npz'),
               protocol_sha256=sha256(HERE/'protocol.json'),
               source_manifest_sha256=sha256(HERE/'source_manifest.json'))
    if module_path.is_file():
        setup['module_sha256']=sha256(module_path)
        setup['module_sha256_reason']=None
    else:
        setup['module_sha256']=None
        setup['module_sha256_reason']='module_path_not_a_file'
    stage='module_import'
    append_event(setup_log,dict(event='started',stage=stage,**setup))
    try:
        spec=importlib.util.spec_from_file_location('q02_probe',module_path)
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    except BaseException as error:
        elapsed,reason=timing_record(time.perf_counter()-setup_start)
        append_event(setup_log,dict(event='failed',stage=stage,error_type=type(error).__name__,error=str(error),
            wrapper_elapsed_s=elapsed,wrapper_elapsed_s_reason=reason,kernel_ms=None,
            kernel_ms_reason='module_import_failed_before_native_call',output=None,
            output_reason='module_import_failed_before_native_call',**setup))
        raise
    append_event(setup_log,dict(event='completed',stage=stage,**setup))
    receipt={'module_path':str(module_path),'module_sha256':sha256(module_path),'scope':module.scope,
             'protocol_sha256':sha256(HERE/'protocol.json'),'source_manifest_sha256':sha256(HERE/'source_manifest.json')}
    # Root reviews cache/compile log before this call; keep its bytes when present.
    build_root=module_path.parent.parent
    for name in ['CMakeCache.txt','compile_commands.json']:
        path=build_root/name
        if path.exists():
            (output/name).write_bytes(path.read_bytes()); receipt[name+'_sha256']=sha256(path)
    json_write(output/'native_receipt.json',receipt)
    stage='probe_constructor'
    constructor_start=time.perf_counter()
    append_event(setup_log,dict(event='started',stage=stage,**setup))
    try:
        probe=module.Probe()
    except BaseException as error:
        elapsed,reason=timing_record(time.perf_counter()-constructor_start)
        append_event(setup_log,dict(event='failed',stage=stage,error_type=type(error).__name__,error=str(error),
            wrapper_elapsed_s=elapsed,wrapper_elapsed_s_reason=reason,kernel_ms=None,
            kernel_ms_reason='constructor_failed_before_evaluate',output=None,
            output_reason='constructor_failed_before_evaluate',**setup))
        raise
    append_event(setup_log,dict(event='completed',stage=stage,**setup))
    calls=output/'native_calls'; calls.mkdir()
    records=[]; checks=[]; counter=0
    def call(xu,goal,category):
        nonlocal counter
        xu=np.array(xu,dtype=np.float32,order='C',copy=True)
        goal=np.array(goal,dtype=np.float32,order='C',copy=True)
        name=f'{counter:04}.npz'; input_name=f'{counter:04}.input.npz'
        np.savez(calls/input_name,xu=xu,goal=goal,call=np.array(counter),category=np.array(category))
        attempt=dict(call=counter,category=category,batch=len(xu),
                     input_raw=str(Path('native_calls')/input_name),input_sha256=sha256(calls/input_name))
        append_event(output/'native_calls.jsonl',dict(event='started',**attempt))
        start=time.perf_counter()
        try:
            result=probe.evaluate(xu,goal)
        except BaseException as error:
            elapsed,reason=timing_record(time.perf_counter()-start)
            append_event(output/'native_calls.jsonl',dict(event='failed',error_type=type(error).__name__,error=str(error),
                wrapper_elapsed_s=elapsed,wrapper_elapsed_s_reason=reason,output=None,
                output_reason='evaluate_raised_no_output_returned',kernel_ms=None,
                kernel_ms_reason='evaluate_raised_native_clock_unavailable',wall_s=None,
                wall_s_reason='evaluate_raised_native_wrapper_clock_unavailable',**attempt))
            raise
        elapsed,elapsed_reason=timing_record(time.perf_counter()-start)
        # Preserve original scalar dtype/bytes, including NaN/Inf. JSON timing
        # fields are null+reason and never replace or filter these raw scalars.
        raw={key:np.asarray(result[key]) for key in ['flat','kernel_ms','wall_s'] if key in result}
        np.savez(calls/name,xu=xu,goal=goal,**raw)
        kernel,kernel_reason=timing_record(result.get('kernel_ms'))
        wall,wall_reason=timing_record(result.get('wall_s'))
        record=dict(event='completed',raw=str(Path('native_calls')/name),sha256=sha256(calls/name),
                    kernel_ms=kernel,kernel_ms_reason=kernel_reason,wall_s=wall,wall_s_reason=wall_reason,
                    wrapper_elapsed_s=elapsed,wrapper_elapsed_s_reason=elapsed_reason,
                    output='returned' if 'flat' in result else None,
                    output_reason=None if 'flat' in result else 'flat_not_returned',**attempt)
        append_event(output/'native_calls.jsonl',record)
        records.append(record); counter+=1
        return decode(result['flat'])
    tol=protocol['tolerances']
    def check(label,key,actual,expected,tolerance,angular=False):
        result=compare(actual,expected,*tol[tolerance],angular=angular)
        checks.append(dict(label=label,field=key,tolerance=tolerance,**result))
    call(inputs[:1],goals[:1],'warmup-retained')
    scalar=[]
    value_tols={'qdd':'value_qdd','qdd_gradient_path':'value_qdd','step':'value_step','pose':'value_pose',
                'cost':'value_cost','terminal_cost':'value_cost','dynamics_jac':'reference_dynamics_jac',
                'A':'reference_step_jac','B':'reference_step_jac','pose_jac':'reference_pose_jac',
                'gradient':'reference_gradient','terminal_gradient':'reference_gradient',
                'Q':'reference_gn','R':'reference_gn','terminal_Q':'reference_gn'}
    for i in range(8):
        result=call(inputs[i:i+1],goals[i:i+1],f'scalar-{i}')
        scalar.append({k:v[0] for k,v in result.items()})
        for key,t in value_tols.items(): check(f'sample-{i}-independent',key,scalar[i][key],references[i][key],t,key=='pose')
    for group_id,group in enumerate(protocol['batch_groups']):
        result=call(inputs[group],goals[group],f'batch-group-{group_id}')
        for lane,index in enumerate(group):
            for key in value_tols:
                check(f'group-{group_id}-lane-{lane}-scalar-{index}',key,result[key][lane],scalar[index][key],'scalar_batch',key=='pose')
    h=protocol['native_fd_step']
    for i in range(8):
        derivatives={k:[] for k in ['qdd','step','pose','cost','terminal_cost']}
        for j in range(21):
            plus=inputs[i:i+1].copy(); minus=inputs[i:i+1].copy()
            plus[0,j]+=h; minus[0,j]-=h
            p=call(plus,goals[i:i+1],f'fd-{i}-{j}-plus')
            m=call(minus,goals[i:i+1],f'fd-{i}-{j}-minus')
            denom=float(plus[0,j])-float(minus[0,j])
            for key in derivatives:
                difference=pose_difference(p[key][0],m[key][0]) if key=='pose' else p[key][0]-m[key][0]
                derivatives[key].append(difference/denom)
        derivatives={k:np.stack(v,axis=-1).astype(float) for k,v in derivatives.items()}
        np.savez(output/f'native-fd-{i:02}.npz',**derivatives)
        check(f'sample-{i}-native-fd','dynamics_jac',scalar[i]['dynamics_jac'],derivatives['qdd'],'native_fd_dynamics_jac')
        check(f'sample-{i}-native-fd','A',scalar[i]['A'],derivatives['step'][:,:14],'native_fd_step_jac')
        check(f'sample-{i}-native-fd','B',scalar[i]['B'],derivatives['step'][:,14:],'native_fd_step_jac')
        check(f'sample-{i}-native-fd','pose_jac',scalar[i]['pose_jac'],derivatives['pose'][:,:7],'native_fd_pose_jac')
        check(f'sample-{i}-native-fd','gradient',scalar[i]['gradient'],derivatives['cost'][0],'native_fd_gradient')
        check(f'sample-{i}-native-fd','terminal_gradient',scalar[i]['terminal_gradient'],derivatives['terminal_cost'][0,:14],'native_fd_gradient')
        check(f'sample-{i}-native-fd','terminal_control_gradient',np.zeros(7),derivatives['terminal_cost'][0,14:],'native_fd_gradient')
        Q,R=gn_reference(inputs[i].astype(float),derivatives['pose'][:,:7],protocol['limits'])
        tQ,_=gn_reference(inputs[i].astype(float),derivatives['pose'][:,:7],protocol['limits'],True)
        np.savez(output/f'native-fd-gn-{i:02}.npz',Q=Q,R=R,terminal_Q=tQ)
        for key,expected in [('Q',Q),('R',R),('terminal_Q',tQ)]:
            check(f'sample-{i}-native-fd-gn',key,scalar[i][key],expected,'native_fd_gradient')
    if counter!=protocol['native_evaluate_calls']: raise ValueError(f'Frozen call count mismatch: {counter}')
    status='PASS' if all(c['pass'] for c in checks) else 'FAIL'
    json_write(output/'checks.json',checks)
    json_write(output/'summary.json',dict(status=status,checks=len(checks),failed=sum(not c['pass'] for c in checks),
        native_calls=counter,timing_scope='kernel CUDA event; wrapper wall includes allocation/copies; initialization excluded; all calls and warmup retained',
        claim='Bounded current-source Q02 output agreement only; not task/solver reliability; historical debug pack still unavailable'))
    return status=='PASS'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='Fresh nonexisting raw directory')
    parser.add_argument('--module',type=Path,help='Exact reviewed isolated q02_probe extension; omitted means CPU-only')
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    ok=False
    try:
        protocol,inputs,goals,references,setup_ok=prepare(args.output)
        if not setup_ok:
            json_write(args.output/'summary.json',dict(status='SETUP_FAIL',native_executed=False,action='Hold Q02; preserve/report setup failure before changing protocol'))
        elif args.module:
            ok=run_native(args.output,args.module.resolve(),protocol,inputs,goals,references)
        else:
            json_write(args.output/'summary.json',dict(status='CPU_REFERENCE_READY',native_executed=False,qualification='PENDING_NATIVE'))
            ok=True
    except BaseException:
        (args.output/'execution_error.txt').write_text(traceback.format_exc())
        json_write(args.output/'summary.json',dict(status='EXECUTION_ERROR',action='Preserved partial raw; no retry or scientific pass'))
        raise
    finally:
        manifest={str(p.relative_to(args.output)):sha256(p) for p in sorted(args.output.rglob('*')) if p.is_file() and p.name!='raw_manifest.json'}
        json_write(args.output/'raw_manifest.json',manifest)
    return 0 if ok else 1
if __name__=='__main__': raise SystemExit(main())
