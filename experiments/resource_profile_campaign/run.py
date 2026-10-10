"""Held resource/profile entry; all recording occurs outside public solve timer."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.resource_profile_campaign import common as C, release as R


def run_case(case, output, *, manifest_path, manifest_sha256, machine, family_state, inputs=None,
             builder=None, query=None, clock=time.perf_counter, metrics_mirror=None):
    case = case.get('parameters', case)
    if case not in C.cases(): raise ValueError('Exact held protocol case required')
    output=Path(output)
    if output.exists(): raise ValueError('Fresh raw directory required')
    output.mkdir(parents=True)
    synthetic=builder is not None or query is not None or clock is not time.perf_counter
    kind='synthetic_test' if synthetic else 'measurement'
    C.write_json(output/'protocol.json',dict(case,timing_mode='A',offline_timing='controller_dt',backend='internal',
        cohort_id='F04-resource' if case['study']=='F04' else 'F05-'+case['arm'],evidence_kind=kind,
        pcg_times_us_is_placeholder=True,pcg_times_us_unavailable_reason='Existing native field is literalzero; not measured PCG cost',
        native_timing_scope='Host BSQP chrono includes kernels/copies/waits/CPU predicate; binding inputs/outputs excluded; publicwrapper wall primary',
        scope='stream-elapsed profiles distinct from OFF first/sustained; no thermal steady-state/saturation claim'))
    recorder=C.RAW.RawRecorder(output); context=None; events=C.F.Events(recorder)
    result=dict(status='held',case=case,evidence_kind=kind,completed_calls=0,measured_calls=0,native_calls=0,
        execution_error=None,task_success=None,physical_pass=None,solver_numerical_pass=None)
    try:
        context=R.validate(manifest_path,manifest_sha256,machine,case,family_state)
        R.begin(family_state,context,case,output)
        C.write_json(output/'provenance.json',dict(machine=machine,source_commit=context['source_commit'],
            module=context['selected_module'],manifest_sha256=manifest_sha256,evidence_kind=kind,
            persistent_state='fresh owner percall; dual/rho resets recorded; actual dual/rho/drho not exported'))
        if case['study']=='F04':
            from experiments.resource_profile_campaign.factory import query as production_query
            recorder('query_started',dict(case=case))
            started=clock()
            payload=(query or production_query)(context,case)
            wall=clock()-started
            recorder('query_returned',dict(payload=payload,query_wall_s=wall,scope='resource query host wall; zero solves'))
            if payload['N']!=case['N'] or payload['batch_size']!=case['B']: raise ValueError('Resource query target differs')
            if any(payload[x]['code']!=0 for x in ('device_status','properties_status','driver_status','runtime_status')):
                raise RuntimeError('Device/version resource query failed; raw status retained')
            if [row['kernel'] for row in payload['kernels']]!=['setup_kkt','merit','pcg','schur1','schur2','dz','line_search'] or any(row['attributes_status']['code']!=0 for row in payload['kernels']):
                raise RuntimeError('Compiled-kernel attribute query missing/failed')
            # Occupancy/fit unsupported cells are retained scientific resource evidence.
            result.update(status='completed',query_wall_s=wall,solver_calls=0)
        else:
            source=Path(inputs) if inputs is not None else Path(context['bank']['path'])
            meta,arrays=C.bank(source,context['bank']['sha256'])
            from experiments.resource_profile_campaign.factory import Factory
            factory=(builder or Factory)(context,meta,events)
            B=case['B']; SQP=case['SQP']; PCG=case['PCG']; serial=0; coverage=[]
            def call(request, warmup=False):
                nonlocal serial
                owner=None; metadata=dict(episode_id=case['arm'],slot=serial,category='warmup' if warmup else 'measured',
                    input_id=request['input_id'],input_index=request['input_index'],candidate_ids=request['candidate_ids'],
                    B=B,N=32,caps=dict(SQP=SQP,PCG=PCG),timing_mode='A',fresh_owner=True)
                recorder('owner_started',dict(metadata,state=request['state'],reference=request['reference'],warm_start=request['warm']))
                started=clock()
                try:
                    owner=factory(B,SQP,PCG); constructor_end=clock()
                    recorder('owner_created',dict(metadata,factory_wall_s=constructor_end-started,effective_solver_params=owner.solver_params))
                    owner.solver.reset_dual();owner.solver.reset_rho()
                    recorder('state_reset',dict(metadata,dual_reset=True,rho_reset=True,reset_wall_s=clock()-constructor_end))
                    _,raw=C.E.timed_solve(owner,request['state'],request['reference'],request['warm'],metadata,events,clock)
                    # Native profile payload already durably captured by Events.
                    profile=events.capture.reply.get('experiment_profile_stages')
                    if case['stage_timing']:
                        if profile is None: raise RuntimeError('Profile fields missing from ON module')
                        C.validate_profile(profile,SQP)
                    elif profile is not None: raise RuntimeError('Timing arm unexpectedly instrumented')
                    result['completed_calls']+=1
                    if not warmup:
                        result['measured_calls']+=1;coverage.append(dict(input_index=request['input_index'],candidate_ids=request['candidate_ids']))
                    recorder('call_finished',dict(metadata,wall_solve_time_s=raw['wall_solve_time_s'],native_solve_time_ms=raw['native_solve_time_ms']))
                except BaseException as error:
                    events('execution_failed',dict(metadata,error=str(error),error_type=type(error).__name__))
                    raise
                finally:
                    if owner is not None: owner.close()
                    serial+=1
            if case['arm']!='first_response': call(C.request(meta,arrays,B,warmup=True),warmup=True)
            if case['arm']=='sustained':
                start=clock(); i=0
                while clock()-start < case['duration_s']:
                    # Boundary precedes request preparation AND next owner creation.
                    index,candidates=C.schedule(B,i)
                    call(C.request(meta,arrays,B,index,candidates));i+=1
                elapsed=clock()-start
                result.update(sustained_wall_s=elapsed,boundary_overrun_s=max(0.,elapsed-case['duration_s']),
                    scope='whole fresh-owner/reset/publicsolve/record loop; passiveconditions; no thermalsteady/saturation claim')
            else: call(C.request(meta,arrays,B))
            result.update(status='completed',coverage=coverage,coverage_comparable_between_B=False)
    except Exception as error:
        result.update(status='held' if context is None else 'execution_failed',execution_error=dict(type=type(error).__name__,error=str(error)))
        events('execution_failed',dict(error=str(error),error_type=type(error).__name__))
        if context is not None: R.hold(family_state,context,error,output)
    result['native_calls']=events.native_calls
    try:
        C.write_json(output/'metrics.json',result);recorder.manifest()
        if metrics_mirror is not None:
            try: shutil.copyfile(output/'metrics.json',Path(metrics_mirror))
            except Exception as error:
                result.update(status='execution_failed',failure_stage='runner_metrics_mirror',
                    execution_error=dict(type=type(error).__name__,error=str(error)))
                events('execution_failed',dict(stage='runner_metrics_mirror',error=str(error),error_type=type(error).__name__))
                C.write_json(output/'metrics.json',result);recorder.manifest()
                raise
        if context is not None and result['execution_error'] is None: R.complete(family_state,context,output)
    except Exception as error:
        if context is not None: R.hold(family_state,context,error,output)
        raise
    if result['execution_error'] is not None: raise RuntimeError(result['execution_error']['error'])
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--manifest',type=Path,default=Path(__file__).with_name('release_manifest.json'))
    p.add_argument('--manifest-sha256');p.add_argument('--machine',default=os.environ.get('GATO_MACHINE'))
    p.add_argument('--family-state',type=Path);p.add_argument('--inputs',type=Path);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    if a.dry_run:
        print(json.dumps(dict(status='held_planning_only',cases=C.cases(),queries=8,first_response_calls=18,
            profile_calls_including_warmups=12,sustained_warmups=6,approved_sustained_wall_s=720,duration_approval_sha256=C.DURATION_APPROVAL_SHA256,
            measured_sustained_calls='variable; allraw retained',release=json.loads(a.manifest.read_text())),indent=2));return 0
    if not all((a.case,a.output,a.manifest_sha256,a.machine,a.family_state)):p.error('case/output/pinned manifest/machine/family-state required')
    a.output.mkdir(parents=True,exist_ok=True);raw=a.output/'resource-profile-raw'
    run_case(json.loads(a.case.read_text()),raw,manifest_path=a.manifest,manifest_sha256=a.manifest_sha256,
        machine=a.machine,family_state=a.family_state,inputs=a.inputs,metrics_mirror=a.output/'metrics.json')
    return 0


if __name__=='__main__':raise SystemExit(main())
