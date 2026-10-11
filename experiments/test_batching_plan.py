"""CPU-only queue coverage and BB retained-readiness wiring checks."""
import json
from pathlib import Path
import subprocess

import pytest

import batching_plan
import bb_prepare
from adapters import ba
from analysis.ba_seed import progress,unpack
import numpy as np


def test_pending_only_plans_and_frozen_source_reuse(tmp_path):
    plans=batching_plan.prepare(tmp_path)
    specs={m:p['experiments'][0] for m,p in plans.items()}
    pending=[(c['parameters']['pair_id'],c['parameters']['batch']) for c in specs['gtx1070']['cases']]
    full=[(c['parameters']['pair_id'],c['parameters']['batch']) for c in specs['rtx5060']['cases']]
    assert len(pending)==176 and len(set(pending))==176 and pending==full[24:]
    assert full==[(p,b) for p in json.loads((batching_plan.ROOT/'experiments/inputs/batching/bb_protocol.json').read_text())['task_ids'] for b in (1,16)]
    assert len(full)==200 and specs['jetson']['cases']==specs['rtx5060']['cases']
    assert (2342,16) not in pending and (2342,1) not in pending
    assert all(e['machines']==[m] and e['source_commit']==batching_plan.SOURCE and e['timeout_seconds']==600 for m,e in specs.items())
    assert plans['jetson']['status']=='proposed_not_executed_target_duration_discussion_pending'
    assert '--reuse-gtx1070' in specs['gtx1070']['preflight']['command']
    assert '--reuse-gtx1070' not in specs['rtx5060']['preflight']['command']
    retained=plans['gtx1070']['retained_results']['BB']['gtx1070']
    assert len(retained['retained'])==24 and retained['failed_episode']['technical_failure']['error_type']=='FloatingPointError'
    changed=subprocess.check_output(['git','-C',str(batching_plan.ROOT),'diff','--name-only',batching_plan.ORIGINAL_SOURCE,batching_plan.SOURCE],text=True).splitlines()
    assert set(changed)=={'examples/randomized_multimodal/tiago_reaching_batch_pilot.py',
        'examples/randomized_multimodal/validate_reaching_mpc.py','tests/python/test_tiago_batching_recording.py'}
    a1070=json.loads((tmp_path/'ba_gtx1070.json').read_text())['experiments'][0]
    a5060=json.loads((tmp_path/'ba_rtx5060.json').read_text())['experiments'][0]
    assert len(a1070['cases'])==100 and len(a5060['cases'])==300
    assert all(c['parameters']['arm']=='nominal16x' for c in a1070['cases'])
    assert a1070['build']==specs['gtx1070']['build'] and a5060['build']==specs['rtx5060']['build']


def test_reuse_module_and_ready_evidence_without_gpu(tmp_path,monkeypatch):
    source=tmp_path/'source'; inputs=tmp_path/'inputs'; build=tmp_path/'build'
    source.mkdir();inputs.mkdir()
    native=tmp_path/'bsqpN32_tiago_right.synthetic.so';native.write_bytes(b'synthetic native bytes')
    module=bb_prepare.copy_module(native,bb_prepare.sha(native),build)
    assert module.read_bytes()==native.read_bytes()
    with pytest.raises(ValueError,match='native module differs'):
        bb_prepare.copy_module(native,'wrong',build)
    (source/'runtime.py').write_text('synthetic_runtime=True\n')
    readiness=dict(native_module=dict(sha256=bb_prepare.sha(native)),required_task_suite=dict(passed=True),
        canonical_preflight=dict(passed=True,arms=[dict(batch_size=b,task_success=True,physical_pass=True) for b in (1,16)]))
    (inputs/'gtx1070_readiness.json').write_text(json.dumps(readiness))
    (inputs/'source_reuse.json').write_text(json.dumps(dict(unchanged_runtime_sha256={'runtime.py':bb_prepare.sha(source/'runtime.py')})))
    assert bb_prepare.preflight(source,build,inputs,tmp_path/'output',True)==0
    assert json.loads((tmp_path/'output/metrics.json').read_text())['reused'] is True
    (inputs/'gtx1070_A_preflight.json').write_text(json.dumps(dict(returncode=0,summaries={
        'B1':dict(task_success=True,physical=dict(physical_valid=True)),
        'B16':dict(task_success=True,physical=dict(physical_valid=True))})))
    monkeypatch.setattr(bb_prepare.subprocess,'run',lambda command:type('Result',(),{'returncode':1})())
    assert bb_prepare.preflight(source,build,inputs,tmp_path/'nominal-failed',True,'A',True)==1
    failed=json.loads((tmp_path/'nominal-failed/metrics.json').read_text())
    assert failed['task_success'] is failed['physical_pass'] is None and failed['failed_arm']=='nominal16x-B1'
    assert 'source_reuse' in failed and 'original_preflight' in failed
    assert json.loads((tmp_path/'nominal-failed/reused_canonical.json').read_text())['task_success'] is True
    (source/'runtime.py').write_text('changed_runtime=True\n')
    with pytest.raises(ValueError,match='runtime differs'):
        bb_prepare.preflight(source,build,inputs,tmp_path/'changed',True)


def test_target_preflight_is_sequential_and_holds_shared_failure(tmp_path,monkeypatch):
    build=tmp_path/'build';(build/'modules').mkdir(parents=True)
    (build/'modules/bsqpN32_tiago_right.synthetic.so').write_bytes(b'synthetic')
    calls=[]
    def run(command):
        calls.append(command)
        return type('Result',(),{'returncode':0})()
    monkeypatch.setattr(bb_prepare.subprocess,'run',run)
    assert bb_prepare.preflight(tmp_path,build,tmp_path,tmp_path/'success')==0
    assert [c[c.index('--batch')+1] for c in calls]==['1','16']
    calls.clear()
    def failed(command):
        calls.append(command)
        return type('Result',(),{'returncode':1})()
    monkeypatch.setattr(bb_prepare.subprocess,'run',failed)
    assert bb_prepare.preflight(tmp_path,build,tmp_path,tmp_path/'failed')==1
    assert len(calls)==1
    metrics=json.loads((tmp_path/'failed/metrics.json').read_text())
    assert metrics['task_success'] is metrics['physical_pass'] is None and metrics['failed_arm']=='default-B1'


def test_ba_declared_budget_only_and_worker_wiring(tmp_path):
    inputs=batching_plan.ROOT/'experiments/inputs/batching'
    nominal=ba.load_protocol(inputs/'ba_nominal16x.json',inputs/'ba_protocol.json')
    assert nominal['solver_params']['max_sqp_iters']==80
    modified=dict(nominal,solver_params=dict(nominal['solver_params'],qd_cost=.01))
    bad=tmp_path/'bad.json';bad.write_text(json.dumps(modified))
    with pytest.raises(ValueError,match='protocol differs'):ba.load_protocol(bad,inputs/'ba_protocol.json')
    build=tmp_path/'build';(build/'modules').mkdir(parents=True)
    (build/'modules/bsqpN32_tiago_right.synthetic.so').write_bytes(b'fake')
    class Worker:
        def run_episode(self,args,pair,batch,dt):
            assert self.load_protocol(args.protocol)['solver_params']['max_sqp_iters']==80
            assert pair==3626 and batch==1 and dt==.03 and args.evidence_kind=='synthetic_test'
            return dict(complete=True,task_success=False,physical=dict(physical_valid=False),outcome='timeout')
    case=dict(parameters=dict(pair_id=3626,batch=1,arm='nominal16x'))
    assert ba.run_case(tmp_path,build,inputs,case,tmp_path/'output',worker=Worker())==0
    result=json.loads((tmp_path/'output/metrics.json').read_text())
    assert result['evidence_kind']=='synthetic_test' and result['task_success'] is False


def test_seed_progress_is_own_lane_and_not_task_success():
    unchanged=progress(2.,2.,np.zeros(665),np.zeros(665),[0,0,0])
    assert unchanged['merit_decrease']==0 and unchanged['own_output_change_linf']==0 and unchanged['accepted_steps']==0
    failed=progress(np.nan,1.,np.zeros(665),np.ones(665),[0,1,0])
    assert failed['merit_available'] is False and failed['own_output_change_linf']==1
    assert [a.shape for a in unpack(np.zeros(665))]==[(32,14),(31,7)]


@pytest.mark.parametrize('outcome,physical_valid,success,returncode',[
    ('reached',True,True,0),('timeout',True,False,1),('reached',False,True,1),
])
def test_ba_mode_a_original_outcome_drives_preflight(tmp_path,outcome,physical_valid,success,returncode):
    inputs=batching_plan.ROOT/'experiments/inputs/batching'
    build=tmp_path/'build';(build/'modules').mkdir(parents=True)
    (build/'modules/bsqpN32_tiago_right.synthetic.so').write_bytes(b'fake')
    summary=dict(complete=True,outcome=outcome,reached_seconds=.9 if outcome=='reached' else None,
                 physical=dict(physical_valid=physical_valid,q_violation_max_rad=0.))
    class Worker:
        def run_episode(self,args,pair,batch,dt):
            assert pair==-1 and batch==1 and args.preflight_input.name=='ba_nominal16x_preflight.json'
            return summary.copy()
    output=tmp_path/'output'
    assert ba.run_case(tmp_path,build,inputs,None,output,preflight=True,batch=1,
                       worker=Worker(),preflight_arm='nominal16x')==returncode
    metrics=json.loads((output/'metrics.json').read_text())
    assert metrics['task_success'] is success and metrics['physical_pass'] is physical_valid
    assert metrics['outcome']==outcome and metrics['physical']==summary['physical']
    assert metrics['reached_seconds']==summary['reached_seconds'] and metrics['execution_error'] is None
