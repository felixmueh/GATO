import importlib.util,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def module():
 s=importlib.util.spec_from_file_location('state_rate_prepare',ROOT/'experiments/launch/state_rate/prepare.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_six_frozen_protocols():
 m=module();ps=m.protocols();assert set(ps)=={'sep-exact','sep-50hz','grid12-exact','grid12-50hz','grid24-exact','grid24-50hz'}
 for name,p in ps.items():
  assert p['N']==32 and p['sim_dt_s']==.001 and p['tasks'][0]['start_state']=='nominal'
  assert p['state_period_s']==(.02 if name.endswith('50hz') else 0.)
  assert p['tracking_solver_params']['max_sqp_iters']==5 and p['tracking_solver_params']['max_pcg_iters']==120
 for prefix in ('sep','grid12','grid24'):
  a,b=ps[prefix+'-exact'],ps[prefix+'-50hz'];assert a['initial_state']==b['initial_state'];assert a['tasks'][0]['reference_packed']==b['tasks'][0]['reference_packed']
 saved=json.loads((ROOT/'experiments/fig8_sep23_counterpart/inputs/t01.json').read_text());assert ps['sep-exact']['tasks'][0]['reference_packed']==saved['tasks'][0]['reference_packed']

def test_plans_reuse_module_and_review_boundary():
 m=module();p=json.loads((m.HERE/'internal.json').read_text());assert p['preparation']['stop_after']=='sep-exact' and not p['preparation']['execution_authorized']
 assert len(p['experiments'])==6
 for e in p['experiments']:
  assert e['repeats']==1 and e['build']['commands'][0][2]=='copy-module' and m.SHA in e['build']['commands'][0]

def test_exact_call_unchanged_positive_forward(monkeypatch):
 import sys
 sys.path.insert(0,str(ROOT/'experiments'))
 from adapters import state_rate as a
 sentinel=object();calls=[]
 monkeypatch.setattr(a,'task_execute',lambda *args:sentinel)
 p=module().protocols()['sep-exact'];p.update(p['tasks'][0])
 assert a.execute(None,p,None,None) is sentinel
 class Controller:
  def run_mpc_fig8(self,*args,**kwargs):calls.append(kwargs);return sentinel
 p['state_period_s']=.02
 assert a.execute(Controller(),p,lambda *args:None,None) is sentinel
 assert calls[0]['state_period_s']==.02 and calls[0]['sim_dt']==.001 and calls[0]['offline_timing']=='solve_time'

def test_actual_motion_columns_and_sampled_preservation():
 import sys
 sys.path.insert(0,str(ROOT/'experiments'))
 from adapters.state_rate import actual_feedback_row
 row={**{f'q{j}_rad':0. for j in range(7)},**{f'v{j}_rad_s':0. for j in range(7)},'fk_x_m':0.,'fk_y_m':0.,'fk_z_m':0.,'error_m':0.,'reference_x_m':0.,'reference_y_m':0.,'reference_z_m':0.}
 actual_feedback_row(row,np.ones(14),[1.,0.,0.]);assert row['q0_rad']==1. and row['sampled_q0_rad']==0.;assert row['fk_x_m']==1. and row['sampled_fk_x_m']==0.;assert row['error_m']==1.

def test_runs_timestep_is_knot_spacing(tmp_path):
 s=importlib.util.spec_from_file_location('state_rate_export',ROOT/'experiments/launch/state_rate/export.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 p=module().protocols()['grid12-exact'];p.update(p['tasks'][0]);(tmp_path/'protocol.json').write_text(json.dumps(p));(tmp_path/'metrics.json').write_text(json.dumps({'outcome':'task_failed'}))
 assert m.summarize(tmp_path)['timestep_s']==.03
