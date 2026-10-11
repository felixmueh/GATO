"""Injected F01c collector; fresh complete owner per call, modeA/static solve."""
from copy import deepcopy
import time
import numpy as np

# Scientific loop transplanted from 9f2ab6f; dependencies supplied by adapter.
E = None


def partitions(B):
    if type(B) is not int or B not in (1, 2, 4, 8, 16):
        raise ValueError('B must divide fixed16 pool')
    return [list(range(i, i+B)) for i in range(0, 16, B)]


def jobs(bank,B,SQP,PCG,diagnostics):
    scheduled=[dict(category='process_cuda_warmup',excluded=True,input_index=None,repeat=None,group_index=0,
        candidate_ids=list(range(B)))]
    for repeat in range(3):
        order=list(range(6)) if repeat%2==0 else list(reversed(range(6)))
        for t in order:
            for group,ids in enumerate(partitions(B)):
                scheduled.append(dict(category='primary',excluded=False,input_index=t,input_id=bank['inputs'][t]['id'],
                    repeat=repeat,group_index=group,candidate_ids=ids))
    if diagnostics:
        if (B,SQP,PCG)!=(16,5,100):raise ValueError('Correspondence belongs only to middle-budgetB16worker')
        for category,ids in [('duplicate',[0,0]+list(range(2,16))),('permutation',list(reversed(range(16))))]:
            for repeat in range(3):scheduled.append(dict(category=category,excluded=True,input_index=0,input_id=bank['inputs'][0]['id'],
                repeat=repeat,group_index=0,candidate_ids=ids))
    return scheduled


def collect(factory,bank,arrays,case,emit,assessor=None,clock=time.perf_counter,evidence_kind='synthetic_test'):
    B,SQP,PCG=(case[k] for k in ('B','SQP','PCG'))
    scheduled=jobs(bank,B,SQP,PCG,case.get('diagnostics',False))
    result=dict(evidence_kind=evidence_kind,outcome='timing_completed',task_success=None,physical_pass=None,solver_numerical_pass=None,
        execution_error=None,calls=[],scheduled_calls=len(scheduled),attempted_calls=0,
        timing_scope='native/us and public solve wrapper wall; setup/teardown,diagnostics,recording outside solve timer',
        collections=[],candidate_correspondence='same fixed pool allB; no equality/feasibility or fullpersistentstate certificate')
    collections={};collection_starts={};owner=None
    for call_index,job in enumerate(scheduled):
        start=clock();setup_done=False;row=dict(job,call_index=call_index,status='started');before_io=getattr(emit,'recording_s',None)
        if job['category']=='primary':
            key=(job['input_index'],job['repeat'])
            collection_starts.setdefault(key,start)
        try:
            if job['category']=='process_cuda_warmup':
                state=arrays['warmup_state'];pool=arrays['warmup_pool'];reference=arrays['warmup_reference']
            else:
                task=bank['inputs'][job['input_index']];state=arrays['starts'][task['start_index']]
                pool=arrays['pools'][task['start_index']];reference=arrays['references'][job['input_index']]
            owner=factory(B,SQP,PCG)
            expected=dict(bank['preset'],max_sqp_iters=SQP,max_pcg_iters=PCG)
            if owner.N!=32 or owner.dt!=.03 or owner.batch_size!=B or owner.solver_params!=expected:
                raise ValueError('Effective public owner/preset differs')
            owner.solver.reset_dual();owner.solver.reset_rho()
            owner.solver.set_f_ext_B(np.zeros((B,6),dtype=np.float32))
            setup=clock()-start;setup_done=True;row['setup_s']=setup
            output,raw=E.timed_solve(owner,np.tile(state,(B,1)),np.tile(reference,(B,1)),pool[job['candidate_ids']].copy(),
                dict(job,call_index=call_index,B=B,N=32,caps=dict(SQP=SQP,PCG=PCG),timing_mode='A'),emit,clock)
            row.update(status='completed',setup_s=setup,native_ms=raw['native_solve_time_ms'],wrapper_ms=raw['wall_solve_time_s']*1000)
            if assessor is not None:
                checks=[assessor(owner,z,reference[:6]) for z in output]
                emit('lane_diagnostics',dict(job,call_index=call_index,lanes=checks))
            if job['category']=='primary':
                key=(job['input_index'],job['repeat'])
                entry=collections.setdefault(key,dict(input_id=job['input_id'],repeat=job['repeat'],B=B,expected_calls=16//B,
                    completed_calls=0,native_ms=0.,wrapper_ms=0.,setup_s=0.,collection_wall_s=0.))
                entry['completed_calls']+=1
                for field in ('native_ms','wrapper_ms','setup_s'):entry[field]+=row[field]
        except BaseException as error:
            row.update(status='failed',error_type=type(error).__name__,error=str(error))
            if not setup_done:row['setup_s']=clock()-start
            result.update(outcome='execution_failed',execution_error=str(error))
            emit('execution_failed',dict(job,call_index=call_index,error=str(error),status='failed'))
        finally:
            teardown_start=clock()
            if owner is not None and hasattr(owner,'close'):owner.close()
            owner=None;row['teardown_s']=clock()-teardown_start
            row['call_pipeline_wall_s']=clock()-start
            row['recording_s']=None if before_io is None else emit.recording_s-before_io
            result['calls'].append(row);result['attempted_calls']+=1
            emit('call_finished',row)
            if job['category']=='primary':
                key=(job['input_index'],job['repeat'])
                if key in collections:collections[key]['collection_wall_s']=clock()-collection_starts[key]
        if result['execution_error'] is not None:break
    result['missing_calls']=scheduled[result['attempted_calls']:]
    # Every planned primary collection exists even if never reached.
    for task in bank['inputs']:
        for repeat in range(3):
            index=bank['inputs'].index(task);key=(index,repeat)
            entry=collections.get(key,dict(input_id=task['id'],repeat=repeat,B=B,expected_calls=16//B,
                completed_calls=0,native_ms=None,wrapper_ms=None,setup_s=None,collection_wall_s=None))
            entry['complete']=entry['completed_calls']==entry['expected_calls'];result['collections'].append(entry)
    emit('worker_finished',result)
    return result
