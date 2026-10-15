#!/usr/bin/env python3
"""One frozen state-rate tracking episode; preserve existing mode-B charging."""
import argparse,csv,json,sys,traceback
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from adapters import task,fig8_sep23
from recording import iter_events,write_json

SAMPLE_FIELDS=('sample_timestamp_s','sample_age_s','repeated_sample','sample_index','sample_delta_s','effective_warm_start_shift_s','optimizer_reference_time_s')
def sampling_rows(output):
 records=[e['record'] for e in iter_events(output) if e['event']=='solve_completed']
 result={}
 for r in records:
  stamp=float(r['feedback_time_s'])
  # Exact path intentionally emits no sampler metadata: exact state is fresh per call.
  result[r['solve_index']]={k:r.get(k,stamp if k in ('sample_timestamp_s','optimizer_reference_time_s') else 0. if k in ('sample_age_s','sample_delta_s') else False if k=='repeated_sample' else r['solve_index'] if k=='sample_index' else None) for k in SAMPLE_FIELDS}
 return result

def actual_feedback_row(row,state,xyz):
 """Main motion columns use actual plant state; preserve held solver-input columns."""
 state=np.asarray(state).reshape(-1,14)[0]
 for key in [f'q{j}_rad' for j in range(7)]+[f'v{j}_rad_s' for j in range(7)]+['fk_x_m','fk_y_m','fk_z_m','error_m']:
  row['sampled_'+key]=row[key]
 row.update({f'q{j}_rad':float(state[j]) for j in range(7)})
 row.update({f'v{j}_rad_s':float(state[7+j]) for j in range(7)})
 row.update(dict(zip(('fk_x_m','fk_y_m','fk_z_m'),map(float,xyz))))
 target=np.asarray([float(row['reference_'+axis+'_m']) for axis in 'xyz'])
 row['error_m']=float(np.linalg.norm(np.asarray(xyz)-target))
 return row

def export_feedback(source,output):
 output=Path(output);fig8_sep23.export_feedback(source,output);samples=sampling_rows(output)
 path=output/'feedback.csv'
 if path.exists():
  with path.open() as f:rows=list(csv.DictReader(f))
  for row in rows:row.update(samples[int(row['solve_index'])])
  actual={e['record']['solve_index']:e['record'].get('actual_feedback_state',e['record']['state']) for e in iter_events(output) if e['event']=='solve_completed'}
  import pinocchio as pin
  protocol=json.loads((output/'protocol.json').read_text());model=pin.buildModelFromUrdf(str(Path(source)/protocol['urdf']['path']));data=model.createData();frame=model.getFrameId('arm_right_tool_link')
  for row in rows:
   state=np.asarray(actual[int(row['solve_index'])]).reshape(-1,14)[0]
   pin.framesForwardKinematics(model,data,state[:7].astype(np.float64))
   actual_feedback_row(row,state,data.oMf[frame].translation.copy())
  with path.open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 schema=json.loads((output/'feedback_schema.json').read_text());schema.update(state_and_fk='main q/v/FK/error use actual plant state at solve start; sampled_* columns preserve held optimizer state/FK/error against actual-time reference',sample_timestamp='held measurement time; actual feedback_time/time_s still solve start; exact path inferred fresh timestamp',sample_age='actual solve start minus sample timestamp',sampling_fields=list(SAMPLE_FIELDS),physical_and_tracking_metrics='metrics.json uses every actual plant state; feedback sampled-state error is separate')
 write_json(output/'feedback_schema.json',schema)
 p=json.loads((output/'protocol.json').read_text());values=list(samples.values());indices=[v['sample_index'] for v in values]
 summary=dict(state_period_s=p.get('state_period_s',0.),solves=len(values),unique_samples=len(set(indices)),solves_per_sample_mean=len(values)/len(set(indices)) if indices else None,sample_age_mean_ms=1000*float(np.mean([v['sample_age_s'] for v in values])) if values else None,repeated_sample_share=float(np.mean([v['repeated_sample'] for v in values])) if values else None)
 write_json(output/'sampling_metrics.json',summary)
 metrics=json.loads((output/'metrics.json').read_text());metrics['state_sampling']=summary;metrics['actual_plant_tracking_RMS_m']=metrics.get('tracking_RMS_m');write_json(output/'metrics.json',metrics)

def execute(controller,protocol,recorder,helpers):
 if protocol.get('state_period_s',0.)==0.:return task_execute(controller,protocol,recorder,helpers)
 reference=np.asarray(protocol['reference_packed'],dtype=protocol['reference_dtype'])
 recorder('frozen_reference',dict(reference=reference,reference_dt_s=protocol['reference_dt_s'],reference_sha256=protocol['reference_sha256']))
 return controller.run_mpc_fig8(np.asarray(protocol['initial_state'],dtype=np.float32),reference,sim_dt=protocol['sim_dt_s'],sim_time=protocol['duration_s'],offline_timing='solve_time',record_raw=True,record_callback=recorder,state_period_s=protocol['state_period_s'])
task_execute=task.execute

def run(source,build,inputs,case,output):
 task.execute=execute
 try:code=task.run(source,build,case,output,mode='B',inputs=inputs)
 finally:task.execute=task_execute
 export_feedback(source,output)
 return code
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('source','build','inputs','case','output'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--export-only',action='store_true');a=p.parse_args()
 if a.export_only:export_feedback(a.source,a.output)
 else:raise SystemExit(run(a.source,a.build,a.inputs,json.loads(a.case.read_text()),a.output))
