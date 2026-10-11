"""Focused no-native/no-ROS checks for shared visual-demo task configuration."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
DEMO=ROOT/'experiments/deployment/demo/run_demo.py'
spec=importlib.util.spec_from_file_location('visual_demo_run',DEMO)
run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)


def test_shared_geometry_backend_labels_and_final_switch():
    a=run.resolve('internal','square');b=run.resolve('gazebo','square');c=run.resolve('physical_robot','square')
    assert a['goals']==b['goals']==c['goals'] and len(a['goals'])==5
    assert a['timing_mode']=='B' and b['timing_mode']==c['timing_mode']=='live'
    assert a['runtime_pin']=='502fa29' and b['runtime_pin']==c['runtime_pin']=='63940f7'
    assert a['final_approach'] and a['final_approach_trigger']==dict(distance_m=.075,max_joint_speed_rad_s=.1)
    assert [g['id'] for g in a['goals']]==['bottom-right','top-right','top-left','bottom-left','center']
    xyz=np.array([g['pose_m_rad'][:3] for g in a['goals']]);assert np.isclose(abs(xyz[0,2]-xyz[1,2]),.30)


def test_execute_square_and_tracking_preserves_backend_clock():
    class Controller:
        def run_mpc_goals(self,*args,**kwargs):self.call=(args,kwargs);return None,{}
        def run_mpc_fig8(self,*args,**kwargs):self.call=(args,kwargs);return None,{}
    c=Controller();callback=lambda *args:None
    run.execute(c,run.resolve('internal','square'),None,callback)
    assert c.call[0][1].shape==(5,6) and c.call[1]['goal_timeout']==15.
    assert c.call[1]['offline_timing']=='solve_time' and c.call[1]['record_callback'] is callback
    live=object();run.execute(c,run.resolve('gazebo','square'),live,callback)
    assert c.call[1]['controller'] is live and 'offline_timing' not in c.call[1]
    run.execute(c,run.resolve('internal','figure8-12s'),None,callback)
    assert c.call[1]['sim_time']==12. and c.call[1]['offline_timing']=='solve_time'
    reference=c.call[0][1].reshape(-1,6)
    assert np.allclose(reference[:,3:],[0.,0.,-2.995275773743368])


def test_preview_records_explicit_skips_without_writes(tmp_path,monkeypatch,capsys):
    output=tmp_path/'not-created'
    monkeypatch.setattr(sys,'argv',['run','--backend','physical_robot','--tasks','square','--output',str(output),'--preview'])
    assert run.main()==0 and not output.exists()
    data=json.loads(capsys.readouterr().out)
    assert [r['status'] for r in data['selection']]==['selected','skipped','skipped','skipped']
    assert all(r['backend']=='physical_robot' for r in data['selection'])


def test_runtime_failure_keeps_raw_error(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'runtime_identity',lambda *args:(_ for _ in ()).throw(RuntimeError('retained identity failure')))
    output=tmp_path/'failed';output.mkdir()
    metrics=run.run_task(run.resolve('internal','square'),tmp_path,None,output)
    assert metrics['execution_error'] and 'retained identity failure' in metrics['execution_error'][0]['message']
    assert (output/'metrics.json').is_file() and (output/'SHA256SUMS.json').is_file()


def test_preflight_reuses_only_matching_runtime_module_setup(tmp_path,monkeypatch):
    monkeypatch.setenv('GATO_MACHINE','test-machine')
    p=run.resolve('internal','square')
    identity=dict(runtime_sha256={'controller':'same'},native_source_tree='core',module=dict(path='/exact/module',sha256='binary'),
                  preflight_setup_sha256=run.preflight_identity(p),collector_sha256='collector')
    metrics=dict(task_id='canonical-preflight',backend='internal',machine='test-machine',input_sha256=p['input_sha256'],
                 preflight_setup_sha256=run.preflight_identity(p),task_success=True,physical_pass=True,execution_error=None)
    (tmp_path/'metrics.json').write_text(json.dumps(metrics));(tmp_path/'provenance.json').write_text(json.dumps(identity))
    run.preflight_guard(p,identity,tmp_path)
    import pytest
    changed=dict(identity,module=dict(path='/exact/module',sha256='other'))
    with pytest.raises(RuntimeError,match='mismatch'):run.preflight_guard(p,changed,tmp_path)
    metrics['physical_pass']=False;(tmp_path/'metrics.json').write_text(json.dumps(metrics))
    with pytest.raises(RuntimeError,match='mismatch'):run.preflight_guard(p,identity,tmp_path)
    canonical=run.resolve_preflight('physical_robot')
    assert len(canonical['goals'])==1 and canonical['goal_timeout_s']==8. and canonical['final_approach'] is False
    assert canonical['position_threshold_m']==.05 and canonical['axis_threshold_deg']==30.


def test_strict_settling_timeout_is_separate_from_rough_success():
    p=run.resolve('internal','square')
    stats=dict(goal_outcomes=['timeout']*5,goal_indices=np.arange(5),goal_distances=np.full(5,.03),
               tool_axis_errors=np.full(5,np.deg2rad(15.)),joint_velocities=np.full((5,7),.05))
    metrics=run.rough_reaching_metrics(p,stats)
    assert metrics['task_success'] is True and metrics['settling_success'] is False
    class Controller:
        def run_mpc_goals(self,*args,**kwargs):self.options=kwargs;return None,{}
    c=Controller();run.execute(c,p,None,lambda *a:None)
    assert c.options['goal_threshold']==.01 and c.options['goal_dwell_time']==.2
