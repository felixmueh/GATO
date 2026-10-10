"""CPU summaries of the already validated plot exports; no raw NPZ reparse."""
import csv,json,hashlib
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
BASE=Path('/workspace/gato-campaign-recovery-20261010')
PLOTS=BASE/'analysis/f04-f05-gtx1070'
INDEX=BASE/'f04-f05-gtx1070-analysis-index.json'
INDEX_SHA='260ed100e64ee1849552c331b0d83d19c331adaa3f5f00c73fa3017bb4170fcf'
SOURCE='4e5301feaea5f2d06530922ba46aa543e1995d47'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(name,data):
 p=OUT/name
 if p.exists():raise FileExistsError(p)
 p.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def table(name,rows):
 p=OUT/name
 if p.exists():raise FileExistsError(p)
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def stat(values):
 a=np.asarray([v for v in values if v is not None],dtype=float);a=a[np.isfinite(a)]
 return dict(count=len(values),available=len(a),unavailable=len(values)-len(a),mean=None if not len(a) else float(a.mean()),median=None if not len(a) else float(np.median(a)),p95=None if not len(a) else float(np.percentile(a,95)),minimum=None if not len(a) else float(a.min()),maximum=None if not len(a) else float(a.max()))
assert sha(INDEX)==INDEX_SHA
export=read(PLOTS/'provenance.json');coverage=read(BASE/'f04-f05-gtx1070-measured-coverage.json');release=read(BASE/'f04-f05-gtx1070-release.json')
assert export['index_sha256']==INDEX_SHA and export['index']==read(INDEX)
assert export['index']['source_commit']==SOURCE and export['index']['machine']=='gtx1070' and export['index']['evidence_kind']=='measurement'
assert sha(BASE/'f04-f05-gtx1070-plan.json')==coverage['plan_sha256']=='32480d8e9664937591cee40035f14439fba42229bc1f344d035162436934066c'
assert release['bank']['sha256']=='f8a8217982f3fa2b1b4fd2629e8cb45949d372f6d43c6137bbdaa7e52acc7007'
assert sha(BASE/'f04-f05-gtx1070-plan-review.json')=='b8b38e8bf3112647a478e61c40aa0a2a9ea326b6f3b1844b2779e714a62986a4'
# Parse CSV already written by validated plot adapter; compare compact identifiers
# against its in-memory saved provenance, not a second full raw audit.
with (PLOTS/'calls.csv').open() as f:csvcalls=list(csv.DictReader(f))
calls=export['calls'];assert len(csvcalls)==len(calls)==coverage['total_native_calls']==44385
assert len(export['attempts'])==len(coverage['cases'])==38
assert all(r['status']=='completed' for r in calls)
counts=Counter((r['arm'],r['category']) for r in calls)
casepaths={c['raw']:c for c in coverage['cases']};selected={e['path']:e for e in export['index']['attempts']}
attempts=[];checks=[];modules={}
for a in export['attempts']:
 p=Path(a['attempt_path']);m=a['metrics'];c=casepaths[str(p)];entry=selected[str(p)];case=entry['case']
 assert all(a['protocol'][k]==v for k,v in case.items())
 assert a['protocol']['timing_mode']=='A' and a['protocol']['backend']=='internal'
 assert a['provenance']['source_commit']==SOURCE and a['provenance']['module']['flags']['GATO_EXPERIMENT_STAGE_TIMING']==case['stage_timing']
 assert a['manifest_sha256']==entry['manifest_sha256'] and m['execution_error'] is None and m['native_calls']==c['native_calls']
 module=a['provenance']['module'];modules[(a['N'],case['stage_timing'])]=module
 attempted=dict(case,raw=str(p),manifest_sha256=entry['manifest_sha256'],native_calls=m['native_calls'],measured_calls=m['measured_calls'],worker_wall_s=c['end']-c['start'],module_sha256=module['sha256'],module_path=module['path'])
 attempts.append(attempted)
 # Eight bounded metadata checks across query, first-response, profile and sustained arms.
 if len(checks)<8 and (a['arm'] in ['profile','sustained'] or not any(x['arm']==a['arm'] for x in checks)):
  for name in ['protocol','metrics','provenance']:assert read(p/(name+'.json'))==a[name]
  checks.append(dict(arm=a['arm'],raw=str(p),protocol_sha256=sha(p/'protocol.json'),metrics_sha256=sha(p/'metrics.json'),provenance_sha256=sha(p/'provenance.json'),checks='case/mode/source/flags/count/status against validated export and explicit selected coverage'))
groups=defaultdict(list)
for r in calls:groups[(r['arm'],r['B'],r['SQP'],r['PCG'],r['category'])].append(r)
latency=[];first=[]
for (arm,B,S,P,category),rs in sorted(groups.items()):
 for scope in ['native_ms','wrapper_ms']:
  s=stat([r[scope] for r in rs]);latency.append(dict(arm=arm,B=B,SQP=S,PCG=P,category=category,stage_timing=arm=='profile',scope=scope,**s,reciprocal_mean_calls_per_s=None if not s['mean'] else 1000/s['mean'],reciprocal_median_calls_per_s=None if not s['median'] else 1000/s['median'],reciprocal_p95_calls_per_s=None if not s['p95'] else 1000/s['p95']))
 if arm=='first_response':first.extend(rs)
loops=[]
for a in export['attempts']:
 if a['arm']!='sustained':continue
 m=a['metrics'];rs=groups[('sustained',a['B'],a['SQP'],a['PCG'],'measured')]
 assert len(rs)==m['measured_calls']
 n=stat([r['native_ms'] for r in rs]);w=stat([r['wrapper_ms'] for r in rs])
 loops.append(dict(B=a['B'],SQP=a['SQP'],PCG=a['PCG'],measured_calls=len(rs),excluded_warmups=m['native_calls']-len(rs),duration_s=m['sustained_wall_s'],boundary_overrun_s=m['boundary_overrun_s'],whole_loop_calls_per_s=len(rs)/m['sustained_wall_s'],native_mean_ms=n['mean'],native_median_ms=n['median'],native_p95_ms=n['p95'],wrapper_mean_ms=w['mean'],wrapper_median_ms=w['median'],wrapper_p95_ms=w['p95'],solve_only_reciprocal_wrapper_mean_calls_per_s=1000/w['mean'],scope=m['scope']))
loops.sort(key=lambda r:(r['SQP'],r['B']))
resources=[]
for r in export['resources']:
 attr=r.get('attributes') or {};resources.append(dict(N=r['N'],B=r['B'],kernel=r['kernel'],threads=r['threads'],dynamic_shared_bytes=r['dynamic_shared_bytes'],**attr,attributes_status=r['attributes_status']['code'],occupancy_status=r['occupancy_status']['code'],theoretical_active_blocks_per_SM=r['theoretical_active_blocks_per_SM'],fits_default_limits_prediction=r['fits_default_limits_prediction'],observed_launch=r['observed_launch'],observed_launch_unavailable_reason=r['observed_launch_unavailable_reason']))
assert len(resources)==56
stagegroups=defaultdict(list)
for r in export['stages']:stagegroups[(r['B'],r['SQP'],r['PCG'],r['category'],r['stage'])].append(r)
stages=[]
for (B,S,P,category,name),rs in sorted(stagegroups.items()):stages.append(dict(B=B,SQP=S,PCG=P,category=category,stage=name,scope='stream elapsed us includes idle host submission gaps',**stat([r['elapsed_us'] for r in rs]),unavailable_reasons=dict(Counter(r['unavailable_reason'] for r in rs if r['elapsed_us'] is None))))
assert len(export['stages'])==472 and sum(r['category']=='measured' for r in export['stages'])==236
queries=[]
for a in export['attempts']:
 if a['study']!='F04':continue
 payload=a['resources'];queries.append(dict(N=a['N'],B=a['B'],kernel_rows=len(payload['kernels']),query_wall_s=a['metrics']['query_wall_s'],driver_version=payload['driver_version'],runtime_version=payload['runtime_version'],device=payload['device'],API_statuses={k:v for k,v in payload.items() if k.endswith('_status')},scope=payload['scope']))
profiletotals=[]
for a in export['attempts']:
 if a['arm']!='profile':continue
 rs=[r for r in export['stages'] if r['attempt_path']==a['attempt_path'] and r['category']=='measured'];call=next(r for r in calls if r['attempt_path']==a['attempt_path'] and r['category']=='measured')
 profiletotals.append(dict(B=a['B'],SQP=a['SQP'],PCG=a['PCG'],windows=len(rs),unavailable=sum(r['elapsed_us'] is None for r in rs),sum_observed_window_elapsed_us=sum(r['elapsed_us'] for r in rs if r['elapsed_us'] is not None),native_ms=call['native_ms'],wrapper_ms=call['wrapper_ms'],scope='sum is diagnostic, not an additive decomposition/native clock identity'))
passiveitem=coverage['passive_monitor'];assert sha(passiveitem['path'])==passiveitem['sha256'];passreceipt=read(passiveitem['path']);monitor=Path(passiveitem['path']).parent/'gpu.csv'
with monitor.open() as f:passrows=list(csv.DictReader(f))
passivestats=[]
for name in list(passrows[0]):
 if name=='timestamp':continue
 values=[];missing=Counter()
 for r in passrows:
  text=r[name].strip()
  try:values.append(float(text.split()[0]))
  except (ValueError,IndexError):values.append(None);missing[text]+=1
 passivestats.append(dict(source_field=name,**stat(values),missing_value_tokens=dict(missing)))
metadata=dict(machine='gtx1070',source_commit=SOURCE,core_commit=release['production_core_commit'],bank=release['bank'],index=dict(path=str(INDEX),sha256=INDEX_SHA),
 validated_plot_exports=dict(path=str(PLOTS),files_sha256={f.name:sha(f) for f in PLOTS.iterdir() if f.suffix in ['.csv','.json']},plot_sha256=export['plot_sha256']),
 plan=dict(path=str(BASE/'f04-f05-gtx1070-plan.json'),sha256=coverage['plan_sha256']),release=dict(path=str(BASE/'f04-f05-gtx1070-release.json'),sha256=sha(BASE/'f04-f05-gtx1070-release.json')),review=dict(path=str(BASE/'f04-f05-gtx1070-plan-review.json'),sha256=sha(BASE/'f04-f05-gtx1070-plan-review.json')),
 coverage=dict(path=str(BASE/'f04-f05-gtx1070-measured-coverage.json'),sha256=sha(BASE/'f04-f05-gtx1070-measured-coverage.json')),whole_queue_wall_s=coverage['whole_queue_wall_s'],attempts=attempts,modules=[dict(N=k[0],stage_timing=k[1],**v) for k,v in sorted(modules.items())],representative_metadata_checks=checks,
 passive=dict(receipt=passiveitem,csv=dict(path=str(monitor),sha256=sha(monitor)),command=passreceipt['monitor'],rows=len(passrows),scope=passreceipt['scope']),
 raw_storage=dict(total_bytes=None,total_file_count=None,unavailable_reason='total raw bytes/file count not indexed in consumed validated exports/coverage; no second manifest parse or heavy filesystem walk performed'),
 pcg_placeholder=dict(value=None,reason=export['pcg_times_us_reason']),limits='stageON separate cohort, stream gaps/nonadditive; all resource fit/occupancy predictive; passive queue measurements not thermalsteady or saturation; no equalquality/numericalcertificate/actuation/crossGPU claims')
summary=dict(total_native_calls=len(calls),counts=[dict(arm=k[0],category=k[1],calls=v) for k,v in sorted(counts.items())],execution_failures=coverage['execution_failures'],missing_timings=dict(native=sum(r['native_ms'] is None for r in calls),wrapper=sum(r['wrapper_ms'] is None for r in calls)),query_count=8,kernel_query_rows=56,
 resource_api_errors=sum(r['attributes_status']!=0 or r['occupancy_status']!=0 for r in resources),predicted_fit_false=sum(r['fits_default_limits_prediction'] is False for r in resources),actual_launch_observations=sum(r['observed_launch'] is not None for r in resources),
 profile_measured_windows=236,profile_measured_missing=sum(r['category']=='measured' and r['elapsed_us'] is None for r in export['stages']),passive_rows=len(passrows))
for name,data in [('provenance.json',metadata),('summary.json',summary)]:write(name,data)
for name,data in [('latency_summary.csv',latency),('first_response_samples.csv',first),('sustained_summary.csv',loops),('kernel_resources.csv',resources),('stage_windows_summary.csv',stages),('profile_totals.csv',profiletotals),('passive_conditions.csv',passivestats),('attempts.csv',attempts),('query_devices.csv',queries)]:table(name,data)
print(json.dumps(summary,indent=2));print('SUSTAINED',json.dumps(loops,indent=2));print('PASSIVE',json.dumps(passivestats,indent=2))
