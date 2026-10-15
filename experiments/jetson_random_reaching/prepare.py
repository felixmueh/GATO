"""Portable frozen fresh100 preparation/validation. Stdlib only; no native loads."""
import argparse,copy,hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;INPUTS=HERE/'inputs'
BASE='85ef2ea5cf5b96726be6f4b5441f4a7c0ca041b4'
IDS='438d9ac51ec264ffcd2bb0dfa40b9caaccfc579990cc3b1b502cd40e48a6a96e'
ARMS={'J1':('B',5),'J2':('B',1),'J3':('A',1000)}
BOUNDS={'TIAGO_COST_JOINT_MARGIN_RAD':.20,'TIAGO_COST_VELOCITY_LIMIT_FRACTION':.80,'TIAGO_COST_CONTROL_LIMIT_FRACTION':.85}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def verify_bundle():
 manifest=read(INPUTS/'manifest.json')
 for name,entry in manifest['files'].items():
  if sha(INPUTS/name)!=entry['sha256']:raise ValueError('Portable frozen input changed: '+name)
 assignment=read(INPUTS/'assignment.json')
 if len(assignment['task_ids'])!=100 or len(set(assignment['task_ids']))!=100 or assignment['task_ids_sha256']!=IDS or hashlib.sha256(json.dumps(assignment['task_ids'],separators=(',',':')).encode()).hexdigest()!=IDS:raise ValueError('Ordered fresh100 assignment differs')
 if set(assignment['task_ids'])&set(assignment['excluded_task_ids']):raise ValueError('Fresh100 exclusions differ')
 return manifest,assignment

def protocol(arm,module,module_receipt,published_source_pin=None):
 verify_bundle();p=copy.deepcopy(read(INPUTS/'desktop_protocol.json'));mode,cap=ARMS[arm]
 p.update(id='Jetson-OFF_TORQUE-fresh100-'+arm,study_role='jetson_fresh100',arm=arm,timing_mode=mode,offline_timing='solve_time' if mode=='B' else 'controller_dt',native_module_path=str(Path(module).resolve()),native_module_sha256=sha(module) if Path(module).is_file() else None,native_build_receipt=str(Path(module_receipt).resolve()),desktop_native_module_sha256=read(INPUTS/'desktop_protocol.json')['native_module_sha256'],published_source_pin=published_source_pin,portable_input_manifest_sha256=sha(INPUTS/'manifest.json'))
 p['solver_params']['max_sqp_iters']=cap
 for key,name in [('research_baseline_freeze','research_baseline_freeze.json'),('reused_canonical','canonical_metrics.json'),('validation_assignment','assignment.json')]:
  p[key]=dict(p[key],path=str((INPUTS/name).resolve()),sha256=sha(INPUTS/name))
 p['canonical_policy']='Fresh target canonical gates infrastructure only; retain actual arrival/physical/error labels. Desktop55.740448mm timeout remains failed canonical context.'
 p['charged_time_policy']='BSQP.solve wrapper wall only; recorder/stat access outside timing; old plan executes during charged interval; bootstrap wall unpaid' if mode=='B' else 'Fixed30ms control; computation unpaid, original modeA path'
 return p

def validate_worker(args,p):
 manifest,assignment=verify_bundle();arm=p.get('arm')
 if arm not in ARMS or p.get('study_role')!='jetson_fresh100':raise ValueError('Only declared J1/J2/J3 fresh100 arms')
 mode,cap=ARMS[arm]
 if (p.get('timing_mode'),p.get('offline_timing'),p['solver_params']['max_sqp_iters'])!=(mode,'solve_time' if mode=='B' else 'controller_dt',cap):raise ValueError('Arm timing/cap differs')
 if getattr(args,'timing_mode',mode)!=mode:raise ValueError('CLI timing mode differs from frozen arm')
 baseline=read(INPUTS/'desktop_protocol.json');exceptions={'id','study_role','arm','timing_mode','offline_timing','solver_params','native_module_path','native_module_sha256','native_build_receipt','desktop_native_module_sha256','published_source_pin','portable_input_manifest_sha256','canonical_policy','charged_time_policy','research_baseline_freeze','reused_canonical','validation_assignment'}
 for key,value in baseline.items():
  if key not in exceptions and p.get(key)!=value:raise ValueError('Frozen setting differs: '+key)
 expected=dict(baseline['solver_params'],max_sqp_iters=cap)
 if p['solver_params']!=expected or p['source_base']!=BASE:raise ValueError('Frozen numerical source/costs differ')
 for key in ('task_ids','run_task_ids','validation_task_ids'):
  if p[key]!=assignment['task_ids']:raise ValueError('Ordered fresh100 differs: '+key)
 for key,name in [('research_baseline_freeze','research_baseline_freeze.json'),('reused_canonical','canonical_metrics.json'),('validation_assignment','assignment.json')]:
  if p[key]['sha256']!=sha(INPUTS/name) or sha(p[key]['path'])!=sha(INPUTS/name):raise ValueError('Portable historical context differs: '+key)
 if p['portable_input_manifest_sha256']!=sha(INPUTS/'manifest.json'):raise ValueError('Portable manifest differs')
 source=Path(args.source).resolve();captured=read(INPUTS/'native_source_sha256.json')['native_files']
 for name,digest in captured.items():
  if sha(source/name)!=digest:raise ValueError('Native85ef baseline file differs: '+name)
 inputs={name:sha(Path(args.pool)/name) for name in p['input_sha256']}
 if inputs!=p['input_sha256']:raise ValueError('Frozen pool bytes differ')
 is_preflight=bool(getattr(args,'preflight_input',None));smoke=bool(getattr(args,'smoke',False));pairs=[-1] if is_preflight else assignment['task_ids'][:1] if smoke else assignment['task_ids']
 if is_preflight and smoke:raise ValueError('Canonical and smoke are separate episodes')
 if is_preflight and sha(args.preflight_input)!=sha(INPUTS/'canonical_geometry.json'):raise ValueError('Canonical geometry differs')
 if not is_preflight and list(args.pairs or [])!=pairs:raise ValueError('Explicit ordered task list differs (only smoke may use first3546)')
 if getattr(args,'max_sqp_iters',None) is not None:raise ValueError('No per-command cap override')
 if Path(args.output).exists():raise ValueError('Output must be fresh; preserve failed attempts')
 module=Path(args.module);receipt_path=Path(args.module_receipt);dry=bool(getattr(args,'dry_run',False));module_hash=sha(module) if module.is_file() else None;receipt=read(receipt_path) if receipt_path.is_file() else None
 if not dry and (module_hash is None or receipt is None):raise ValueError('Actual run requires rebuilt native module and receipt')
 if module_hash is not None:
  if p['native_module_sha256']!=module_hash:raise ValueError('Protocol/module hash differs; regenerate preparation after build')
  if receipt is None and not dry:raise ValueError('Native build receipt missing')
 if receipt:
  if receipt.get('module') and Path(receipt['module']).resolve()!=module.resolve():raise ValueError('Receipt/module path differs')
  if receipt.get('module_sha256')!=module_hash or receipt.get('numerical_source_base')!=BASE or receipt.get('cost_bound_defines')!=BOUNDS:raise ValueError('Module receipt source/hash/interior-bound defines differ')
  if not smoke and receipt.get('cuda_arch') not in (87,'87'):raise ValueError('Jetson module must be sm87')
  if not dry and not smoke and receipt.get('host_machine') not in ('aarch64','arm64'):raise ValueError('Actual Jetson run requires ARM-built module, not desktop cross-target compile')
  if smoke and module_hash!=baseline['native_module_sha256']:raise ValueError('Smoke must reuse exact retained desktopq020 module')
  for name,digest in receipt.get('source_sha256',{}).items():
   if name in captured and captured[name]!=digest:raise ValueError('Receipt/native source hash differs: '+name)
 try:commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
 except subprocess.CalledProcessError:raise ValueError('Source must be a pinned Git checkout')
 return p,dict(protocol=p,source=str(source),source_commit=commit,source_sha256=captured,module=str(module.resolve()),module_sha256=module_hash,module_receipt=str(receipt_path.resolve()),module_receipt_sha256=sha(receipt_path) if receipt else None,module_receipt_data=receipt,module_status='present' if module_hash and receipt else 'pending build (dry run only)',pool=str(Path(args.pool).resolve()),inputs_sha256=inputs,pairs=pairs,preflight_input=str(args.preflight_input) if is_preflight else None,preflight_input_sha256=sha(args.preflight_input) if is_preflight else None,wrapper_sha256=sha(HERE.parent/'random_reaching_baseline/run_frozen_baseline.py'),runtime_source=str(HERE.parents[1]),runtime_commit=subprocess.check_output(['git','-C',str(HERE.parents[1]),'rev-parse','HEAD'],text=True).strip(),recording_policy=dict(full_routine_horizons=bool(getattr(args,'save_plans',False)) if mode=='B' else True,all_executed_states_and_controls=True,all_per_solve_work=True,failed_call_inputs_outputs_retained=True),study_role='jetson_fresh100',arm='SMOKE-'+arm if smoke else arm,original_desktop_context=dict(protocol_sha256=sha(INPUTS/'desktop_protocol.json'),native_build_receipt=read(INPUTS/'desktop_native_build_receipt.json'),canonical_metrics=read(INPUTS/'canonical_metrics.json'),assignment_sha256=sha(INPUTS/'assignment.json')),input_bundle_manifest_sha256=sha(INPUTS/'manifest.json'))
if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--module',type=Path,required=True);ap.add_argument('--module-receipt',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--published-source-pin');a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 for arm in ARMS:write(a.output/(arm+'.json'),protocol(arm,a.module,a.module_receipt,a.published_source_pin))
 write(a.output/'preparation.json',dict(arms={'J1':'requiredB5','J2':'requiredB1','J3':'optionalA1000 only after J1/J2 counts reviewed'},bundle_manifest_sha256=sha(INPUTS/'manifest.json'),source_base=BASE,published_source_pin=a.published_source_pin,pairs=read(INPUTS/'assignment.json')['task_ids'],no_runs_performed=True))
 print('Prepared J1/J2 and optionalJ3; actual module pending unless built. No GPU operation.')
