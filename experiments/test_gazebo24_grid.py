"""Bounded CPU review of period-only Gazebo adapter; no native or ROS imports."""
import copy,importlib.util,json,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
from adapters import gazebo24_grid as adapter
PIN=adapter.COLLECTOR

def load_protocol():
 namespace={};exec(subprocess.check_output(['git','show',PIN+':experiments/task_parameters/protocol.py'],cwd=ROOT),namespace);return namespace

def case():return dict(id='figure8-24s',parameters=dict(N=32,dt_s=.03,sqp_max_iter=5,pcg_max_iter=120))

def test_only_period_reference_extension(monkeypatch):
 monkeypatch.setenv('GATO_ROS_PROFILE','simulation');monkeypatch.setenv('ROS_DOMAIN_ID','1');monkeypatch.setenv('RMW_IMPLEMENTATION','rmw_cyclonedds_cpp')
 ns=load_protocol();inputs=ROOT/'experiments/inputs/t01';c=case();anchor=copy.deepcopy(c);anchor['id']='figure8-12s'
 old=ns['resolve_protocol'](inputs,anchor,'gazebo',False,True)
 new=adapter.extend_protocol(ns['resolve_protocol'],ns['periodic_reference'],inputs,c,'gazebo',False,True)
 changed={k for k in set(old)|set(new) if old.get(k)!=new.get(k)}
 assert changed=={'id','case_id','period_s','duration_s','reference_packed','reference_sha256','reference_coverage_s','maximum_reference_speed_m_s','period_extension','joint_state_rate_policy'}
 assert new['solver_params']==old['solver_params'] and new['initial_state']==old['initial_state'] and new['live_controller']==old['live_controller']
 assert (new['N'],new['dt_s'],new['solver_params']['max_sqp_iters'],new['solver_params']['max_pcg_iters'])==(32,.03,5,120)
 assert new['reference_coverage_s']>=24.+31*.03
 ref=np.asarray(new['reference_packed']).reshape(-1,6)
 old_ref=np.asarray(old['reference_packed']).reshape(-1,6)
 # At even new-grid indices, 24s phase equals integer old 12s-grid samples.
 np.testing.assert_allclose(ref[::2][:len(old_ref)],old_ref[:len(ref[::2])],rtol=0,atol=1e-14)
 assert new['maximum_reference_speed_m_s']==old['maximum_reference_speed_m_s']/2
 assert new['canonical_setup_sha256']==old['canonical_setup_sha256']

def test_plan_one_attempt_original_build():
 p=json.loads((ROOT/'experiments/launch/state_rate/gazebo24.json').read_text());assert p['scientific_episodes']==1 and len(p['experiments'])==1
 e=p['experiments'][0];assert len(e['cases'])==1 and e['repeats']==1 and 'preflight' not in e
 old=json.loads((ROOT/'experiments/launch/task_parameters/rtx5060_gazebo.json').read_text())['experiments'][0]
 assert e['build']==old['build'];assert e['source_commit']==PIN and e['collector_source_commit']==PIN
 assert e['allow_report_only_safety'] is True
 assert '--preflight-evidence' in e['command']

def test_retained_worker_preflight_check(tmp_path,monkeypatch):
 # Load the immutable worker with its exact protocol module; no native import.
 ns=load_protocol();module=type(sys)('protocol');module.__dict__.update(ns);monkeypatch.setitem(sys.modules,'protocol',module)
 worker=dict(__file__=str(tmp_path/'worker.py'));exec(subprocess.check_output(['git','show',PIN+':experiments/task_parameters/worker.py'],cwd=ROOT),worker)
 monkeypatch.setenv('GATO_ROS_PROFILE','simulation');monkeypatch.setenv('ROS_DOMAIN_ID','1');monkeypatch.setenv('RMW_IMPLEMENTATION','rmw_cyclonedds_cpp')
 p=ns['resolve_protocol'](ROOT/'experiments/inputs/t01',case(),'gazebo',True,True)
 previous={'case_id':'canonical-preflight','canonical_setup_sha256':p['canonical_setup_sha256']};metrics=dict(execution_error=None,task_success=True,physical_pass=False,handback_confirmed=True)
 (tmp_path/'protocol.json').write_text(json.dumps(previous));(tmp_path/'metrics.json').write_text(json.dumps(metrics))
 deployment=type('Deployment',(),{'preflight_path':staticmethod(lambda _:tmp_path)})
 worker['require_preflight'](tmp_path,p,deployment)
 metrics['handback_confirmed']=False;(tmp_path/'metrics.json').write_text(json.dumps(metrics))
 with pytest.raises(RuntimeError):worker['require_preflight'](tmp_path,p,deployment)

def test_export_observed_aliases(tmp_path,monkeypatch):
 spec=importlib.util.spec_from_file_location('export_gazebo24',ROOT/'experiments/launch/state_rate/export_gazebo24.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 raw=tmp_path/'raw';attempt=raw/'rtx5060/Gazebo24-N32-grid/figure8-24s/attempt-001';attempt.mkdir(parents=True)
 (attempt/'protocol.json').write_text(json.dumps(dict(reference_packed=[0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.],reference_dt_s=.03)))
 (attempt/'metrics.json').write_text(json.dumps(dict(task_success=False,physical_pass=True,handback_confirmed=True,execution_error=None)))
 trace=dict(time_s=np.array([0.,.03]),tool_position=np.array([[.1,0,0],[1.2,0,0]]),q=np.ones((2,7)),qd=np.zeros((2,7)))
 monkeypatch.setattr(m,'iter_events',lambda _:iter([dict(event='observed_task_trace',record=trace)]))
 out=tmp_path/'out';rows=m.export(raw,out);assert len(rows)==1 and rows[0]['task_success'] is False
 import csv
 data=list(csv.DictReader((out/'attempt-001/observed_task_trace.csv').open()));assert float(data[1]['t'])==.03 and float(data[1]['x'])==1.2 and float(data[1]['fk_x_m'])==1.2 and float(data[1]['q0_rad'])==1.
 assert float(data[1]['error'])==pytest.approx(.2)
 assert (out/'REPORT.pdf').is_file()
