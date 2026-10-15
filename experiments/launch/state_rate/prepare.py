"""CPU-only frozen six-case preparation; no native import or execution."""
import copy,hashlib,json,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
SOURCE='1b517fce5bd4ddcbabd200fa120d72f012f40771'
COLLECTOR='90bdb825d059e22b709e2e53a785d7d034aad1b0'
SHA='0c2cbc7cd217561cdba2ebc82cfa9e9eb12ddb3b442da7d9b4bd2ff0828c1666'
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def protocols():
 sep=json.loads((ROOT/'experiments/fig8_sep23_counterpart/inputs/t01.json').read_text()); sep_task=sep['tasks'][0]
 grid=json.loads((ROOT/'experiments/inputs/t01/t01.json').read_text())
 assert digest(ROOT/'experiments/inputs/t01/t01.json')=='aec47cfa3819de5adbb8633b637f5bd9bcd327cb3afc5043a856b323af5aac41'
 namespace={};exec(subprocess.check_output(['git','show',COLLECTOR+':experiments/task_parameters/protocol.py'],cwd=ROOT),namespace)
 template=next(t for t in grid['tasks'] if t['id']=='figure8-12s')
 result={}
 for family,period in [('sep',24.),('grid12',12.),('grid24',24.)]:
  for suffix,state_period in [('exact',0.),('50hz',.02)]:
   name=family+'-'+suffix;cohort=copy.deepcopy(sep if family=='sep' else grid);task=copy.deepcopy(sep_task if family=='sep' else template)
   if family!='sep':
    reference=namespace['periodic_reference'](template,period,period,32,.03)
    task.update(reference_packed=reference.tolist(),reference_sha256=hashlib.sha256(reference.tobytes()).hexdigest(),reference_dtype='<f8',reference_dt_s=.03,duration_s=period,period_s=period,provisional_RMS_gate_m=.03,provisional_maximum_gate_m=.06,reference_scope='frozen T01 sampled period regridded by collector90bdb periodic_reference through full preview',reference_coverage_s=(len(reference)//6-1)*.03,maximum_reference_speed_m_s=template['maximum_reference_speed_m_s']*template['period_s']/period)
   task.update(id=name,state_period_s=state_period,cost_preset='saved-sep23' if family=='sep' else 'task-grid-anchor',start_state='nominal')
   cohort.update(sim_dt_s=.001,state_period_s=state_period,tasks=[task],repeat_policy='one planned attempt; no automatic retry',campaign_id='state-rate-50hz',timing_mode='B',offline_timing='solve_time')
   result[name]=cohort
 return result

def prepare():
 base=json.loads((ROOT/'experiments/fig8_sep23_counterpart/plan.json').read_text());experiments=[]
 for name,cohort in protocols().items():
  directory=HERE/'inputs'/name;directory.mkdir(parents=True,exist_ok=True);path=directory/'t01.json';path.write_text(json.dumps(cohort,indent=2)+'\n')
  e=copy.deepcopy(base['experiments'][0]);e.update(id=name,protocol='state-rate-50hz-v1',source_commit=SOURCE,repeats=1)
  e['inputs']={str(path.relative_to(ROOT)):digest(path)}
  e['command']=['{python}','{runner}/adapters/state_rate.py','--source','{source}','--build','{build}','--inputs','{inputs}/'+str(directory.relative_to(ROOT)),'--case','{case}','--output','{output}']
  e['cases']=[dict(id=name,parameters=dict(state_period_s=cohort['state_period_s']))];experiments.append(e)
  (HERE/(name+'.json')).write_text(json.dumps(dict(version=1,experiments=[e]),indent=2)+'\n')
 (HERE/'internal.json').write_text(json.dumps(dict(version=1,experiments=experiments,preparation=dict(execution_authorized=False,stop_after='sep-exact',review='Review sep-exact before launching remaining five; no automatic tolerance gate',no_gpu_build=True)),indent=2)+'\n')
if __name__=='__main__':prepare()
