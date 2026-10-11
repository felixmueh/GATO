"""Scientific F04/F05 loop reused from4e5301f; current runner owns failures."""
from pathlib import Path
import time
from adapters import resource_profiles_common as C

def execute(case, context, inputs, events, recorder, builder, query, clock=time.perf_counter):
    result=dict(status='started',case=case,completed_calls=0,measured_calls=0,native_calls=0,
        execution_error=None,task_success=None,physical_pass=None,solver_numerical_pass=None)
    try:
        if case['study']=='F04':
            recorder('query_started',dict(case=case))
            started=clock()
            payload=query(context,case)
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
            factory=builder(context,meta,events)
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
    except BaseException as error:
        result.update(status='execution_failed', execution_error=dict(type=type(error).__name__, error=str(error)))
        events('execution_failed', dict(error=str(error),error_type=type(error).__name__))
    result['native_calls']=events.native_calls
    return result
