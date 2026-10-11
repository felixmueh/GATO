"""Simulator trial gates and original-roster bookkeeping, no ROS or GPU."""
import json
from pathlib import Path

import pytest

from adapters import simulated_fault_trial as adapter
import simulated_fault_plan


def config(tmp_path):
    source=tmp_path/'source';source.mkdir()
    for name in adapter.RUNTIME_FILES:
        file=source/name;file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(name.encode())
    receipt=tmp_path/'metrics.json';receipt.write_text(json.dumps(dict(task_success=True,physical_pass=True,
        execution_backend='gazebo',handback_confirmed=True)))
    (tmp_path/'protocol.json').write_text(json.dumps(dict(N=32,dt_s=.03,solver_params=dict(qd_cost=.1,vel_lim_cost=.1))))
    (tmp_path/'provenance.json').write_text(json.dumps(dict(source_sha256={p:adapter.sha(source/p) for p in adapter.RUNTIME_FILES})))
    log=tmp_path/'pal.jsonl';log.write_text(json.dumps(dict(wall=9.9,controllers=[dict(name=n,state='active') for n in adapter.DEFAULTS]))+'\n')
    return source,dict(simulation_only=True,domain=91,canonical_live_preflight_receipt=str(receipt),pal_controller_log=str(log))


def test_missing_or_stale_simulator_evidence_prevents_trial(tmp_path):
    source,c=config(tmp_path)
    assert adapter.prerequisites(c,source,10.)['pal_age_s']==pytest.approx(.1)
    with pytest.raises(ValueError,match='Fresh observed'):adapter.prerequisites(c,source,13.)
    with pytest.raises(ValueError,match='Gazebo-only'):adapter.prerequisites(dict(c,simulation_only=False),source,10.)
    (source/adapter.RUNTIME_FILES[0]).write_text('changed runtime')
    with pytest.raises(ValueError,match='runtime differs'):adapter.prerequisites(c,source,10.)


def test_exact_original_trials_preserved_with_leading_baseline_as_preflight(tmp_path):
    plan=simulated_fault_plan.prepare(tmp_path/'plan.json');cases=plan['experiments'][0]['cases']
    assert len(plan['original_order'])==21 and len(cases)==20
    assert plan['original_leading_baseline_is_preflight'] and plan['original_order'][0]=='baseline'
    assert [c['parameters']['fault'] for c in cases]==plan['original_order'][1:]
    assert cases[-1]['parameters']['fault']=='baseline'
    assert plan['experiments'][0]['timeout_seconds']==600


def test_pal_freshness_is_rechecked_after_target_suite(tmp_path,monkeypatch):
    source,c=config(tmp_path);build=tmp_path/'build';(build/'modules').mkdir(parents=True)
    (build/'modules/bsqpN64_tiago_right.fake.so').write_bytes(b'fake')
    path=tmp_path/'local.json';path.write_text(json.dumps(c))
    times=iter((10.,13.))
    monkeypatch.setattr(adapter.time,'monotonic',lambda:next(times))
    monkeypatch.setattr(adapter.task_suite,'run',lambda *a:0)
    monkeypatch.setattr(adapter.subprocess,'run',lambda *a,**kw: (_ for _ in ()).throw(AssertionError('stale observation must prevent motion')))
    assert adapter.run(source,build,path,None,tmp_path/'output',True)==1
    metrics=json.loads((tmp_path/'output/metrics.json').read_text())
    assert 'Fresh observed' in metrics['execution_error']
