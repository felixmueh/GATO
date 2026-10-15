"""Portable preparation/export contracts, CPU only."""
from pathlib import Path
import importlib.util,json,types
import pytest
ROOT=Path(__file__).resolve().parents[1]
def prep():
 s=importlib.util.spec_from_file_location('fresh100_prepare',ROOT/'experiments/jetson_random_reaching/prepare.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_bundle_exact_ordered100_and_arms(tmp_path):
 m=prep();manifest,assignment=m.verify_bundle();assert len(assignment['task_ids'])==100 and assignment['task_ids'][0]==3546
 for arm,(mode,cap) in m.ARMS.items():
  p=m.protocol(arm,tmp_path/'missing.so',tmp_path/'receipt.json');assert p['timing_mode']==mode and p['solver_params']['max_sqp_iters']==cap and p['source_base']==m.BASE
  assert p['run_task_ids']==assignment['task_ids'] and p['solver_params']['q_lim_cost']==.02125
  assert p['native_module_sha256'] is None and p['desktop_native_module_sha256'].startswith('d9bee23f')

def test_dry_run_pending_no_native_import(tmp_path):
 m=prep();p=m.protocol('J1',tmp_path/'missing.so',tmp_path/'receipt.json');a=types.SimpleNamespace(source=ROOT,module=tmp_path/'missing.so',module_receipt=tmp_path/'receipt.json',pool=m.INPUTS,protocol=None,output=tmp_path/'new',pairs=p['run_task_ids'],preflight_input=None,smoke=False,timing_mode='B',max_sqp_iters=None,dry_run=True,save_plans=False)
 _,info=m.validate_worker(a,p);assert info['module_status'].startswith('pending') and len(info['pairs'])==100
 a.dry_run=False
 with pytest.raises(ValueError,match='requires rebuilt'):m.validate_worker(a,p)
 p['solver_params']['qd_cost']=.01
 with pytest.raises(ValueError,match='costs'):m.validate_worker(a,p)

def analysis():
 s=importlib.util.spec_from_file_location('fresh100_export',ROOT/'experiments/jetson_random_reaching/export.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_wilson_zero_and_count_denominator():
 m=analysis();assert m.wilson(0,100)[0]==0. and m.wilson(0,100)[1]==pytest.approx(.03699349820698568)
 low,high=m.wilson(55,100);assert low<.55<high

def test_tied_first_limits_and_actual_plan_association():
 m=analysis();row={k+'_first_time_s':v for k,v in [('joint',.2),('velocity',.1),('effort',.1)]};row.update({k+'_first_joint':'arm_right_1_joint' for k in ['joint','velocity','effort']})
 assert [e['category'] for e in m.order_events(row)]==['velocity','effort','joint']
 assert m.order_label(m.order_events(row))=='velocity + effort -> joint'
 raw=dict(step=4,feedback_time_s=.2,executed_duration_s=.01,native_solve_us=1000.,solve_wall_s=.002,charged_solve_s=.002,actual_charged_elapsed_s=.001,executed_prefix_plan_step=3,work=dict(actual_sqp=1,pcg_iteration_sum=0,zero_pcg_outer_exit=True,accepted_steps=0,line_search_count=0,stop_reason='zero_pcg'))
 r=m.solve_row('J2',3546,{'timing_mode':'B','solver_params':{'max_sqp_iters':1}},raw)
 assert r['executed_prefix_plan_step']==3 and r['actual_charged_elapsed_s']==.001 and r['zero_PCG_exit']==1 and r['SQP_cap_hit']==1

def test_hundred_case_export_real_collector_with_error(tmp_path):
 """Exercise original collect_run schema, denominator and all final export plumbing."""
 import csv,copy,numpy as np
 m=analysis();prepared=prep();assignment=prepared.read(prepared.INPUTS/'assignment.json');pilot=tmp_path/'raw/J1/pilot';pilot.mkdir(parents=True)
 p=prepared.protocol('J1',tmp_path/'test.so',tmp_path/'receipt.json');p['native_module_sha256']='synthetic-module'
 provenance=dict(protocol=p,pairs=p['run_task_ids'],pool=str(prepared.INPUTS),inputs_sha256=p['input_sha256'],source_sha256=prepared.read(prepared.INPUTS/'native_source_sha256.json')['native_files'],module_sha256='synthetic-module',module_receipt_data=dict(module_sha256='synthetic-module'),power_mode='MODE_40W synthetic fixture')
 def write(path,value):path.write_text(json.dumps(value))
 write(pilot/'protocol.json',p);write(pilot/'provenance.json',provenance)
 with np.load(prepared.INPUTS/'pool.npz') as z:configs=z['q'].copy()
 results=[]
 for index,record in enumerate(assignment['records']):
  pair=record['pair_id'];folder=pilot/f'pair_{pair:05d}';folder.mkdir();bad=index==0;state=np.r_[configs[record['start_index']],np.zeros(7)];states=np.vstack([state,state]);states[-1,0]=np.nan if bad else states[-1,0]
  np.savez(folder/'executed.npz',states=states,controls=np.zeros((1,7)),times=np.array([0.,8.]),control_times=np.array([0.]))
  np.savez(folder/'inputs.npz',initial_state=state,source_goal_q=configs[record['goal_index']])
  physical=dict(finite=not bad,maximum_velocity_ratio=None if bad else 0.,maximum_effort_ratio=None if bad else 0.)
  for category in ['joint','velocity','effort']:physical[category]=dict(pass_check=not bad,maximum_excess=None if bad else 0.,first_violation=None)
  if bad:physical['velocity']['first_violation']=dict(joint_name='arm_right_1_joint',time_s=.01)
  metric=dict(pair_id=pair,outcome='numerical_failure' if bad else 'timeout',complete=True,task_success=False,physical_pass=not bad,execution_error='synthetic nonfinite state' if bad else None,failure_phase='episode' if bad else None,interrupted=False,shared_setup_failure=False,reached_seconds=None,simulated_seconds=8.,final=dict(position_error_m=None if bad else .1,tool_axis_error_deg=0.,orientation_error_deg=0.,velocity_l1_rad_s=0.),initial=dict(position_error_m=.1,velocity_l1_rad_s=0.),feedback_solves=1,actual_sqp_sum=1,zero_pcg_exit_solves=1,actual_pcg_iteration_sum=0,episode_wall_s=.01,solve_wall_s=.002,native_solve_us=1000.,physical=physical)
  work=dict(actual_sqp=1,pcg_iteration_sum=0,zero_pcg_outer_exit=True,accepted_steps=0,line_search_count=0,stop_reason='zero_pcg',initial_merit=0.,final_merit=0.)
  solve=dict(step=0,feedback_time_s=0.,executed_duration_s=.03,native_solve_us=1000.,solve_wall_s=.002,charged_solve_s=0.,actual_charged_elapsed_s=0.,executed_prefix_plan_step=0,work=work)
  if bad:
   work.update(actual_sqp=None,pcg_iteration_sum=None,zero_pcg_outer_exit=None,accepted_steps=None,line_search_count=None,initial_merit=None,final_merit=None)
   solve.update(native_solve_us=None,work_available=False);metric.update(actual_sqp_sum=0,actual_pcg_iteration_sum=0,zero_pcg_exit_solves=0,native_solve_us=0.,work_complete=False)
  (folder/'solves.jsonl').write_text(json.dumps(solve)+'\n');write(folder/'metrics.json',metric);results.append(metric)
 write(pilot/'summary.json',results)
 # J1 checkpoint works before a J2 directory exists.
 counts=m.export(tmp_path/'raw',None,['J1'],counts_only=True);assert counts[0]['execution_failures']==1 and counts[0]['assigned']==100 and counts[0]['physical_passes']==99
 out=tmp_path/'out';m.export(tmp_path/'raw',out,['J1']);rows=list(csv.DictReader((out/'cases.csv').open()));assert len(rows)==100 and rows[0]['minimum_q_clearance_rad']=='' and rows[0]['history_status']=='nonfinite'
 assert json.loads((out/'counts.json').read_text())['arms'][0]['wilson95']['execution_failures']['denominator']==100
 assert rows[0]['work_complete']=='False' and rows[0]['native_median_ms']==''
 assert 'arrival median/Q1/Q3' in (out/'WRITING_HANDOFF.md').read_text()
 assert (out/'REPORT.pdf').is_file() and (out/'solve_samples.csv.gz').is_file()
 assert json.loads((out/'metadata.json').read_text())['arm_provenance']['J1']['power_mode']=='MODE_40W synthetic fixture'
