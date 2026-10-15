"""CPU-only state-rate export. Usage: python3 export.py --raw ROOT --output ROOT."""
import argparse,csv,json,shutil,hashlib
from pathlib import Path
FIELDS='run_id,period_s,N,timestep_s,cost_preset,state_period_s,start_state,rms_time_weighted_mm,max_mm,solves,solves_per_sample_mean,sample_age_mean_ms,repeated_sample_share,native_solve_mean_ms,native_solve_median_ms,source_commit,module_sha256,status'.split(',')
def summarize(path):
 p=json.loads((path/'protocol.json').read_text());m=json.loads((path/'metrics.json').read_text());s=m.get('state_sampling',{});identity=json.loads((path/'provenance.json').read_text()) if (path/'provenance.json').exists() else {};native=m.get('timing_distributions',{}).get('native_ms',{})
 row=dict.fromkeys(FIELDS);row.update(run_id=p['id'],period_s=p['period_s'],N=p['N'],timestep_s=p['dt_s'],cost_preset=p['cost_preset'],state_period_s=p['state_period_s'],start_state=p['start_state'],rms_time_weighted_mm=1000*m['tracking_RMS_m'] if m.get('tracking_RMS_m') is not None else None,max_mm=1000*m['tracking_maximum_m'] if m.get('tracking_maximum_m') is not None else None,solves=s.get('solves'),solves_per_sample_mean=s.get('solves_per_sample_mean'),sample_age_mean_ms=s.get('sample_age_mean_ms'),repeated_sample_share=s.get('repeated_sample_share'),native_solve_mean_ms=native.get('mean'),native_solve_median_ms=native.get('median'),source_commit=identity.get('source_commit'),module_sha256=identity.get('module_sha256'),status='execution_failed' if m.get('execution_error') else m.get('outcome'))
 return row

def export(raw,output):
 raw,output=Path(raw),Path(output);output.mkdir(parents=True,exist_ok=True);rows=[]
 for path in sorted(raw.rglob('metrics.json')):
  attempt=path.parent
  if not(attempt/'protocol.json').exists():continue
  p=json.loads((attempt/'protocol.json').read_text())
  if p.get('campaign_id')!='state-rate-50hz':continue
  row=summarize(attempt);rows.append(row);destination=output/row['run_id']/attempt.name;destination.mkdir(parents=True,exist_ok=True)
  for name in ('metrics.json','protocol.json','provenance.json','feedback_schema.json','sampling_metrics.json'):
   if (attempt/name).exists():shutil.copy2(attempt/name,destination/name)
  if (attempt/'feedback.csv').exists():shutil.copy2(attempt/'feedback.csv',destination/'executed_trace.csv')
  # Reference is immutable plotting input, independent of sampled-feedback cadence.
  import numpy as np
  np.save(destination/'reference.npy',np.asarray(p['reference_packed'],dtype=p['reference_dtype']).reshape(-1,6))
 with (output/'runs.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
 lines=['# State-rate tracking comparison','','Six planned one-attempt runs; sep-exact must be reviewed before launching the remaining cases. All outcomes and source/native identities remain separate.','',
 'Native source b400 and exact retained N32 module remain unchanged. Frontend extends reviewed7cf modeB only for sampled-state option. The plant integrates at1ms, while state sampling, knot spacing, actual solve-start time and charged solve wall time remain separate. Exact state_period=0 uses the unchanged call path;20ms holds optimizer measurements. Bootstrap wall remains unpaid and old active plan executes during each measured solve.','',
 'runs.csv uses time-weighted RMS and maximum over every actual plant state, including startup; sampled-feedback errors are separate. Each run’s executed_trace.csv is the per-feedback export with sample time/age/repeated flags. Full plant states and applied-control history remain raw; feedback torque is the first executed substep at that actual solve-start timestamp. Terminal plans may be discarded.','',
 'Positive sampling also reproduces historical3aa warm-start shift max(sample timestamp delta,sim_dt); repeated samples therefore shift1ms. Exact0 retains current frontend warm-shift behavior. Optimizer reference origin is sample timestamp; returned first knot activates on solve arrival, while old active plan executes during solve.', '',
 'The September reference/settings and grid anchor reference/settings are separate presets. No causal GPU/model claim, full KKT certificate, safety qualification or retries are introduced. Task/physical negatives remain in metrics.json.','',
 '| Run | Plant RMS mm | Plant max mm | Solves | Mean sample age ms | Repeated share | Status |','|---|---:|---:|---:|---:|---:|---|']
 for r in rows:lines.append('| '+' | '.join(str(r[k]) for k in ('run_id','rms_time_weighted_mm','max_mm','solves','sample_age_mean_ms','repeated_sample_share','status'))+' |')
 (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt,textwrap
 from matplotlib.backends.backend_pdf import PdfPages
 text=[]
 for line in lines:text.extend(textwrap.wrap(line,width=100) or [''])
 with PdfPages(output/'REPORT.pdf') as pdf:
  for i in range(0,len(text),60):
   fig=plt.figure(figsize=(8.27,11.69));block=fig.text(.07,.95,'\n'.join(text[i:i+60]),va='top',fontsize=8,fontfamily='monospace');fig.canvas.draw();b=block.get_window_extent(fig.canvas.get_renderer());assert b.x1<fig.bbox.x1 and b.y0>0
   pdf.savefig(fig);plt.close(fig)
 return rows
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(len(export(a.raw,a.output)),'retained attempts')
