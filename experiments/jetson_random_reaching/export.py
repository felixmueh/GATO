"""CPU-only Jetson fresh100 exports, reusing the frozen desktop collectors.
Run --counts-only after J1/J2, before any optional J3. No native imports.
"""
import argparse,csv,gzip,hashlib,json,sys,xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'random_reaching_baseline'))
sys.path.insert(0,str(HERE))
import analyze_frozen_validation as shared
import importlib.util
_spec=importlib.util.spec_from_file_location('portable_fresh100_prepare',HERE/'prepare.py');_prep=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_prep)
verify_bundle,INPUTS,ARMS=_prep.verify_bundle,_prep.INPUTS,_prep.ARMS
read=shared.read;write=shared.write

def wilson(k,n,z=1.959963984540054):
 if not n:return [None,None]
 p=k/n;den=1+z*z/n;center=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return [0. if k==0 else float(max(0,center-half)),1. if k==n else float(min(1,center+half))]
def dump_csv(path,rows,fields=None):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
def finite(x):return isinstance(x,(int,float)) and np.isfinite(x)
def summary(values):
 a=np.asarray([x for x in values if finite(x)],float)
 return dict(count=len(a),mean=float(a.mean()) if len(a) else None,median=float(np.median(a)) if len(a) else None,q25=float(np.percentile(a,25)) if len(a) else None,q75=float(np.percentile(a,75)) if len(a) else None,p95=float(np.percentile(a,95)) if len(a) else None)
def limits():
 joints=[j for j in ET.parse(INPUTS/'tiago_right_arm.urdf').getroot().findall('joint') if j.get('type')!='fixed']
 return dict(**{k:np.asarray([float(j.find('limit').get(k)) for j in joints]) for k in ['lower','upper','velocity','effort']},names=[j.get('name') for j in joints])
def read_solves(path):
 if path.exists():
  with path.open() as f:
   for line in f:
    if line.strip():yield json.loads(line)
def order_events(row):
 events=[dict(category=k,joint=row[k+'_first_joint'],time_s=row[k+'_first_time_s']) for k in ['joint','velocity','effort'] if row[k+'_first_time_s'] is not None]
 return sorted(events,key=lambda e:e['time_s'])
def order_label(events):
 groups=[]
 for event in events:
  if groups and abs(event['time_s']-groups[-1][0])<=1e-9:groups[-1][1].append(event['category'])
  else:groups.append((event['time_s'],[event['category']]))
 return ' -> '.join(' + '.join(categories) for _,categories in groups)
def extract_arm(root,arm,assignment,physical_limits):
 pilot=root/arm/'pilot';p=read(pilot/'protocol.json');prov=read(pilot/'provenance.json')
 if p.get('arm')!=arm or p['run_task_ids']!=assignment['task_ids'] or p['task_ids_sha256']!=assignment['task_ids_sha256']:raise ValueError('Frozen arm/order differs: '+arm)
 mode,cap=ARMS[arm]
 if p['timing_mode']!=mode or p['solver_params']['max_sqp_iters']!=cap:raise ValueError('Arm timing/cap differs')
 if prov['inputs_sha256']!=assignment['input_sha256']:raise ValueError('Saved input hashes differ')
 baseline=read(INPUTS/'desktop_protocol.json')
 for key in shared.prior.RUNTIME:
  if key in ('timing_mode','offline_timing','native_module_sha256'):continue
  expected=dict(baseline[key],max_sqp_iters=cap) if key=='solver_params' else baseline[key]
  if p.get(key)!=expected:raise ValueError('Saved frozen parameter differs: '+key)
 if prov['source_sha256']!=read(INPUTS/'native_source_sha256.json')['native_files']:raise ValueError('Captured native source hashes differ')
 receipt=prov.get('module_receipt_data',{})
 if receipt.get('module_sha256')!=prov['module_sha256'] or p.get('native_module_sha256')!=prov['module_sha256']:raise ValueError('Saved loaded module/receipt identity differs')
 # Resolve the SAME hashed pool locally; preserve original target provenance separately.
 local_prov=dict(prov,pool=str(INPUTS));rows,counts,hashes=shared.collect_run(root/arm,assignment,p,local_prov,physical_limits)
 counts.update(arm=arm,timing_mode=mode,SQP_cap=cap)
 counts['wilson95']={key:dict(numerator=counts[key],denominator=100,interval=wilson(counts[key],100)) for key in ['rough_arrivals','physical_passes','both_passes','execution_failures']}
 counts['arrival_time_s']=summary([r['arrival_seconds'] for r in rows if r['task_success']==1]);counts['nonfinite_saved_histories']=sum(r['history_status']=='nonfinite' for r in rows)
 for row in rows:
  row.update(arm=arm,timing_mode=mode,SQP_cap=cap);events=order_events(row);row['recorded_first_limit_order']=json.dumps(events,separators=(',',':'));row['first_limit_category']=' + '.join(e['category'] for e in events if abs(e['time_s']-events[0]['time_s'])<=1e-9) if events else 'unknown after execution error' if row['execution_failure'] else 'none recorded'
 counts['finite_violation_episodes']=sum(bool(order_events(r)) and r['physical_finite']==1 for r in rows)
 counts['first_limit_category_counts']=dict(Counter(r['first_limit_category'] for r in rows))
 counts['recorded_violation_order_counts']=dict(Counter(order_label(order_events(r)) for r in rows if order_events(r)))
 return rows,counts,p,prov,hashes
SOLVE_FIELDS='arm,pair_id,SQP_cap,timing_mode,step,timing_category,feedback_time_s,executed_duration_s,charged_solve_s,actual_charged_elapsed_s,charged_interval_clipped,native_ms,wrapper_ms,actual_SQP,actual_PCG,zero_PCG_exit,SQP_cap_hit,accepted_steps,line_search_count,accepted_step_sizes,stop_reason,executed_prefix_plan_step,returned_plan_status,warm_shift_s'.split(',')
def solve_row(arm,pair,p,r):
 work=r['work'];mode=p['timing_mode'];cap=p['solver_params']['max_sqp_iters']
 return dict(arm=arm,pair_id=pair,SQP_cap=cap,timing_mode=mode,step=r['step'],timing_category=r.get('timing_category','startup/cold' if r['step']==0 else 'warm'),feedback_time_s=r['feedback_time_s'],executed_duration_s=r.get('executed_duration_s'),charged_solve_s=r.get('charged_solve_s',0. if mode=='A' else None),actual_charged_elapsed_s=r.get('actual_charged_elapsed_s',0. if mode=='A' else None),charged_interval_clipped=r.get('charged_interval_clipped'),native_ms=r['native_solve_us']/1000 if r.get('native_solve_us') is not None else None,wrapper_ms=1000*r['solve_wall_s'] if r.get('solve_wall_s') is not None else None,actual_SQP=work['actual_sqp'],actual_PCG=work['pcg_iteration_sum'],zero_PCG_exit=int(work['zero_pcg_outer_exit']) if work.get('zero_pcg_outer_exit') is not None else None,SQP_cap_hit=int(work['actual_sqp']>=cap) if work.get('actual_sqp') is not None else None,accepted_steps=work.get('accepted_steps'),line_search_count=work.get('line_search_count'),accepted_step_sizes=json.dumps(r.get('accepted_step_sizes'),separators=(',',':')),stop_reason=work['stop_reason'],executed_prefix_plan_step=r.get('executed_prefix_plan_step',r['step'] if mode=='A' else None),returned_plan_status=r.get('returned_plan_status'),warm_shift_s=r.get('warm_shift_s'))
def plots(output,cases,counts,solve_times,desktop):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 arms=[c['arm'] for c in counts];x=np.arange(len(arms));fig,ax=plt.subplots(figsize=(7.5,4.3))
 ax.bar(x-.23,[c['rough_arrivals'] for c in counts],.23,label='Arrivals');ax.bar(x,[100-c['physical_passes'] for c in counts],.23,label='Physical non-passes (includes errors)');ax.bar(x+.23,[c['execution_failures'] for c in counts],.23,label='Execution errors')
 ax.set(xticks=x,xticklabels=[f"{c['arm']} / SQP{c['SQP_cap']} / {c['timing_mode']}" for c in counts],ylim=(0,105),ylabel='Cases / assigned 100');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(output/'arm_outcomes.png',dpi=160);fig.savefig(output/'arm_outcomes.pdf');plt.close(fig)
 fig,ax=plt.subplots(figsize=(7.5,4.3))
 for arm,group in [(a,[r for r in cases if r['arm']==a]) for a in arms]+[('GTX desktop A/SQP1000 (juxtaposition)',desktop)]:
  times=sorted(float(r['arrival_seconds']) for r in group if str(r['complete'])=='1' and str(r['task_success'])=='1')
  ax.step([0,*times,8],[0,*np.arange(1,len(times)+1)/100,len(times)/100],where='post',label=arm)
 ax.set(xlim=(0,8),ylim=(0,1),xlabel='Simulated arrival time (s)',ylabel='Cumulative arrivals / assigned 100');ax.legend(fontsize=7);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(output/'arrival_ecdf.png',dpi=160);fig.savefig(output/'arrival_ecdf.pdf');plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(10,4))
 for c in counts:
  times=solve_times[c['arm']]
  for ax,key,label in zip(axes,['native_ms','wrapper_ms'],['Native solve (ms)','Wrapper solve (ms)']):
   a=np.asarray([v for v in times[key] if finite(v)],float);ax.hist(a,bins=60,histtype='step',density=True,label=f"{c['arm']} cap{c['SQP_cap']} (n={len(a)})");ax.set(xlabel=label,ylabel='Density of recorded solves');ax.set_xscale('log');ax.legend(fontsize=7)
 fig.tight_layout();fig.savefig(output/'solve_times.png',dpi=160);fig.savefig(output/'solve_times.pdf');plt.close(fig)
def export(raw,output,arms,counts_only=False):
 _,assignment=verify_bundle();physical_limits=limits();cases=[];counts=[];metadata={};protocols={};hashes={}
 for arm in arms:
  rows,c,p,prov,casehash=extract_arm(raw,arm,assignment,physical_limits);cases.extend(rows);counts.append(c);metadata[arm]=prov;protocols[arm]=p;hashes[arm]=casehash
 print(json.dumps(dict(checkpoint='J1/J2 counts before optional J3',arms=counts),indent=2),flush=True)
 if counts_only:return counts
 if not all(c['complete'] for c in counts):raise ValueError('Final export requires all100 outcomes per selected arm; missing outcomes are not failures or passes')
 output.mkdir(parents=True,exist_ok=True);solve_times={};first_events=[]
 with gzip.open(output/'solve_samples.csv.gz','wt',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=SOLVE_FIELDS);writer.writeheader()
  for arm in arms:
   p=protocols[arm];solve_times[arm]={'native_ms':[],'wrapper_ms':[]};all_charged=[];all_elapsed=[]
   for row in [r for r in cases if r['arm']==arm]:
    path=raw/arm/'pilot'/f"pair_{row['pair_id']:05d}"/'solves.jsonl';solves=list(read_solves(path));converted=[solve_row(arm,row['pair_id'],p,r) for r in solves];writer.writerows(converted)
    row.update(wrapper_median_ms=summary([r['wrapper_ms'] for r in converted])['median'],wrapper_p95_ms=summary([r['wrapper_ms'] for r in converted])['p95'],native_median_ms=summary([r['native_ms'] for r in converted])['median'],native_p95_ms=summary([r['native_ms'] for r in converted])['p95'],accepted_steps_sum=sum(r['accepted_steps'] or 0 for r in converted),SQP_cap_hit_solves=sum(r['SQP_cap_hit'] or 0 for r in converted),charged_requested_total_s=sum(r['charged_solve_s'] or 0 for r in converted),actual_charged_total_s=sum(r['actual_charged_elapsed_s'] or 0 for r in converted),work_complete=all(r['actual_SQP'] is not None and r['actual_PCG'] is not None for r in converted),work_unavailable_calls=sum(r['actual_SQP'] is None for r in converted))
    for key in solve_times[arm]:solve_times[arm][key].extend(r[key] for r in converted)
    for event in order_events(row):
     when=event['time_s'];prior=[r for r in solves if r['feedback_time_s']<when-1e-9];interval=prior[-1] if prior else solves[0] if solves and abs(when)<1e-9 else None
     if interval is not None and when>interval['feedback_time_s']+interval.get('executed_duration_s',0.)+1e-8:interval=None
     first_events.append(dict(arm=arm,pair_id=row['pair_id'],**event,interval_solve_step=interval['step'] if interval else None,actual_plan_source_step=interval.get('executed_prefix_plan_step',interval['step'] if p['timing_mode']=='A' else None) if interval else None,execution_error=row['execution_failure']))
   c=next(c for c in counts if c['arm']==arm);c['native_solve_ms']=summary(solve_times[arm]['native_ms']);c['wrapper_solve_ms']=summary(solve_times[arm]['wrapper_ms'])
 desktop=list(csv.DictReader((INPUTS/'desktop_cases.csv').open()));byid={int(r['pair_id']):r for r in desktop};paired=[]
 for row in cases:
  old=byid[row['pair_id']];paired.append(dict(arm=row['arm'],pair_id=row['pair_id'],jetson_mode=row['timing_mode'],jetson_SQP=row['SQP_cap'],jetson_arrival=row['task_success'],jetson_physical=row['completed_episode_physical_pass'],jetson_execution_error=row['execution_failure'],desktop_mode='A',desktop_SQP=1000,desktop_arrival=int(old['task_success']),desktop_physical=int(old['completed_episode_physical_pass']),desktop_execution_error=int(old['execution_failure']),arrival_agreement=int(row['task_success']==int(old['task_success'])),physical_agreement=int(row['completed_episode_physical_pass']==int(old['completed_episode_physical_pass'])),execution_error_agreement=int(row['execution_failure']==int(old['execution_failure'])),scope='J3 paired same-mode/cap; trajectories may differ' if row['arm']=='J3' else 'juxtaposition; platform/timing/cap differ'))
 for c in counts:
  group=[r for r in paired if r['arm']==c['arm']];c['desktop_per_task_agreement']={k:sum(r[k] for r in group) for k in ['arrival_agreement','physical_agreement','execution_error_agreement']}
 dump_csv(output/'cases.csv',cases);dump_csv(output/'desktop_per_task.csv',paired)
 if first_events:dump_csv(output/'first_limit_events.csv',first_events)
 write(output/'counts.json',dict(denominator=100,Wilson95='score intervals per marginal A/P/B/E, not paired-difference intervals',arms=counts,desktop_reference=read(INPUTS/'desktop_counts.json'),required_counts_precede_optional_J3=True))
 condition_records={}
 for folder in [raw/'_host_conditions',raw/'conditions']:
  if folder.is_dir():
   for path in folder.iterdir():
    if path.is_file() and path.stat().st_size<1000000:condition_records[str(path.relative_to(raw))]=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),text=path.read_text(errors='replace'))
 write(output/'metadata.json',dict(host_conditions=condition_records,host_conditions_scope='Saved nvpmodel/jetson_clocks/telemetry text only; no query/change during export; absent files remain unavailable',raw_root=str(raw),portable_manifest=read(INPUTS/'manifest.json'),protocols=protocols,arm_provenance=metadata,case_hashes=hashes,analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),physical_limits_source='unchanged raw URDF; interior cost bounds are not physical limits',recording_scope='all episode states/controls and per-solve telemetry retained raw; routine modeB horizons optional, failed calls retained; no sampling filter',timing_statistics='all saved solves including bootstrap; bootstrap modeB charge0 is separate from measured native/wrapper duration'))
 canonical_records={}
 for arm in arms:
  location=raw/arm/'canonical'/'preflight';path=location/'metrics.json'
  if path.exists():canonical_records[arm]=dict(metrics=read(path),provenance=read(location.parent/'provenance.json') if (location.parent/'provenance.json').exists() else None)
 write(output/'canonical_outcomes.json',canonical_records)
 plots(output,cases,counts,solve_times,desktop)
 lines=['# Jetson frozen OFF_TORQUE fresh100','', 'All selected arms have100 recorded outcomes, including execution errors. J1/J2 counts are reported before optional J3; no missing-case inference, tuning, retries or changes to frozen gates.','', '| Arm | Mode / SQP | Arrivals | Physical passes | Both | Errors | Arrival median s | Wrapper median / p95 ms |','|---|---|---:|---:|---:|---:|---:|---:|']
 for c in counts:lines.append(f"| {c['arm']} | {c['timing_mode']} / {c['SQP_cap']} | {c['rough_arrivals']} | {c['physical_passes']} | {c['both_passes']} | {c['execution_failures']} | {c['arrival_time_s']['median']} | {c['wrapper_solve_ms']['median']} / {c['wrapper_solve_ms']['p95']} |")
 writer_details=[]
 for c in counts:
  arrival=c['arrival_time_s'];native=c['native_solve_ms'];wall=c['wrapper_solve_ms']
  writer_details.append(f"{c['arm']}: arrival median/Q1/Q3 = {arrival['median']} / {arrival['q25']} / {arrival['q75']} s (arrivals only); native median/p95 = {native['median']} / {native['p95']} ms; wrapper median/p95 = {wall['median']} / {wall['p95']} ms. Finite-violation episodes = {c['finite_violation_episodes']}; nonfinite saved histories = {c['nonfinite_saved_histories']}; first-limit category counts = {c['first_limit_category_counts']}; recorded violation-order counts = {c['recorded_violation_order_counts']}.")
  if c['arm']=='J3':writer_details.append('J3 per-task agreement counts /100: '+str(c['desktop_per_task_agreement']))
 lines+=['',*writer_details,'','Wilson95 marginal intervals for A/P/B/E and conditional arrival quartiles are in counts.json; every denominator stays100 including numerical execution failures. Arrival is position<=5cm and speedL1<=1rad/s, without orientation. Physical pass additionally requires complete finite error-free raw-URDF position/speed/effort compliance. Cost bounds q inset.20rad / velocity.80 / effort.85 are interior penalties, not relaxed physical checks.','',
 '![Arrivals and physical non-passes per assigned100; errors remain separately shown and also prevent physical qualification.](arm_outcomes.png)','![Arrival ECDF uses all100 assigned cases; timeouts and execution errors never become arrivals. The desktop A/SQP1000 is a juxtaposition for J1/J2.](arrival_ecdf.png)','![Per-solve native and wrapper distributions include bootstrap timing; solve counts vary with each episode. These are not equal-work or independent-task latency estimates.](solve_times.png)','',
 'Native source is exact85ef2ea numerical bytes, published as a tree-equivalent native pin; e1bc38b original modeA functions/protocol are preserved. J1 uses modeB/SQP5, J2 modeB/SQP1, optional J3 modeA/SQP1000. N32/B1,30ms knots,3ms RK4,8s maximum, weights q4/terminal80/qd.05/u1e-5/limits.02125/1.275/.02/mu10/rho.01/PCG120/tolerances1e-3 stay frozen. Cold rest/zero-control seed and once-per-episode dual/rho resets are retained.','',
 'ModeB charges the BSQP.solve wrapper wall, not native-only time; get_stats/recorder/NPZ/plant overhead stays outside the charged interval. The previous active plan executes during that interval, then returned first knot activates on arrival; bootstrap supplies the first nominal prefix with solve wall unpaid. Charged request and actual clipped elapsed are distinct fields. ModeA leaves computation unpaid. Different caps affect realized solve/update intervals and evolving inputs, so J1/J2 versus desktop is not an isolated platform, latency or iteration effect.','',
 'Fresh canonical uses the unchanged geometry and gates infrastructure only. Its recorded task/physical/error labels remain in raw; the historical desktop canonical remains a physical-pass8s timeout at55.740448mm with preflight_pass=false, never relabeled as a passed qualification. New sm87 module receipt and SHA256 are separate from the retained desktopd9bee receipt. Native source/loaded module, power mode/clock telemetry and all hashes are in metadata/target provenance; no power or clock changes are implied.','',
 'cases.csv retains original per-episode schema plus arm/timing/SQP/charged timing/work fields. solve_samples.csv.gz preserves all solve durations, actual SQP/PCG, zero-PCG exit, cap flag, accepted-step data and active-plan association. These flags are not forced mutually exclusive and zero-PCG exit is not a KKT certificate. first_limit_events.csv sorts recorded position/speed/effort first times with ties; active plan source uses executed_prefix_plan_step, not the just-returned plan. Nonfinite/error full-history extrema remain unavailable; observed prefix fields are descriptive. No divergence cause is inferred.','',
 'desktop_per_task.csv compares matching pair IDs, with J1/J2 labeled juxtaposition; J3, if included, reports per-task agreement at the same mode/cap without claiming identical trajectories. Assignment100 seed20261015 and exact pool/pairs hashes are frozen. Large executed.npz, failed solve arrays, raw logs and backups stay under _expr_results. Regenerate using export.py; report_to_pdf.py embeds the figures into a standalone PDF.']
 (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
 (output/'WRITING_HANDOFF.md').write_text('\n'.join(lines[:len(counts)+8])+"\n\n"+'\n\n'.join(writer_details)+"\n\nFull Wilson95 marginal intervals and counts:\n"+json.dumps(counts,indent=2)+"\n\nUse counts.json arrival quartiles/Wilson intervals and cases.csv first-limit category/order for cap1. Interpret J1/J2 results before optional J3. No outcome is tuned toward the thesis claim.\n")
 # Reuse the existing Markdown/figure PDF renderer, no new framework.
 import importlib.util,shutil
 renderer=HERE.parent/'random_reaching_baseline/report_to_pdf.py';shutil.copy2(renderer,output/'report_to_pdf.py')
 spec=importlib.util.spec_from_file_location('fresh100_pdf',renderer);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.render(output/'REPORT.md',output/'pdf-preview')
 return counts
if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--raw-root',type=Path,required=True);ap.add_argument('--output',type=Path);ap.add_argument('--arms',nargs='+',choices=ARMS,default=['J1','J2']);ap.add_argument('--counts-only',action='store_true');a=ap.parse_args()
 if not a.counts_only and not a.output:ap.error('--output required except counts-only')
 export(a.raw_root,a.output,a.arms,a.counts_only)
