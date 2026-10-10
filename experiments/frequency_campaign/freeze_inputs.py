"""Freeze held F01 timing inputs and sampled witnesses using CPU Pinocchio only.

No solver/controller imports, optimizer, GPU call or experiment execution.
Fresh directories preserve both successful and failed preparations.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TASK = 'experiments/campaign/inputs/t01.json'
ENDPOINTS = 'experiments/campaign/inputs/t01_proposal.json'
MODEL = 'gato/dynamics/tiago_right/tiago_right_arm.urdf'
TASK_HASH = 'aec47cfa3819de5adbb8633b637f5bd9bcd327cb3afc5043a856b323af5aac41'
ENDPOINT_HASH = '48728c3e4ea9c3b67f4202ebda7ea943811d9c6cd904c32f75616326fc80bbe9'
MODEL_HASH = '8e1fd973a1d1c3fbddd55039e6eb17b355ebb29f46e1bd8c774b7879c2cdc5bb'
FRACTIONS = (0., .05, .10, .15)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def source_inputs(root=ROOT):
    for path, expected in ((TASK,TASK_HASH),(ENDPOINTS,ENDPOINT_HASH),(MODEL,MODEL_HASH)):
        if sha(root/path) != expected:
            raise ValueError('Frozen source input/model changed: '+path)
    task=json.loads((root/TASK).read_text()); endpoints=json.loads((root/ENDPOINTS).read_text())
    if len(task['tasks'])!=6 or [t['kind'] for t in task['tasks']]!=['reaching']*3+['tracking']*3:
        raise ValueError('Expected exact approved six tasks')
    return task,endpoints


def arrays_and_schedules(task, endpoints):
    initial=np.asarray(task['initial_state'],dtype='<f4')
    delta=np.asarray(task['tasks'][0]['endpoint_delta_rad'],dtype='<f8')
    starts=np.asarray([np.r_[initial[:7].astype('<f8')+fraction*delta,np.zeros(7)]
                       for fraction in FRACTIONS],dtype='<f4')
    goals=np.asarray([t['goal'] for t in task['tasks'][:3]],dtype='<f8')
    endpoint=np.asarray([t['endpoint_q_rad'] for t in endpoints['reaching']],dtype='<f8')
    # Warm-up shares approved goal3 but a distinct start outside all episode inputs.
    warmup=np.asarray(np.r_[initial[:7].astype('<f8')+.025*delta,np.zeros(7)],dtype='<f4')
    arrays={'starts':starts,'goals':goals,'endpoint_q_rad':endpoint,'warmup_state':warmup,
            'warmup_goal':goals[2].copy()}
    for i,t in enumerate(task['tasks'][3:]):
        ref=np.asarray(t['reference_packed'],dtype='<f8')
        if hashlib.sha256(ref.tobytes()).hexdigest()!=t['reference_sha256']:
            raise ValueError('Reference bytes changed')
        if ref.size%6 or ref.size//6<149:
            raise ValueError('Reference window cannot cover N128 preview')
        arrays[f'reference_{i}']=ref
    episodes=[]
    for start in range(4):
        for goal in range(3):
            episodes.append(dict(id=f'start-{start}-goal-{goal+1}',kind='reaching',start_index=start,
                goal_indices=[goal,(goal+1)%3],goal_switch_before_update=10,updates=20,
                categories=['cold']+['warm']*9+['goal_switch']+['warm']*9))
        for speed in range(3):
            episodes.append(dict(id=f'start-{start}-tracking-{speed}',kind='tracking',start_index=start,
                reference_index=speed,source_task_id=task['tasks'][3+speed]['id'],reference_origin_s=0.,
                global_anchor='original T01 reference; never reanchored to new start',updates=20,
                categories=['cold']+['warm']*19))
    variants={}
    for name,count in [('full24',4),('short12',2)]:
        selected=[e for e in episodes if e['start_index']<count]
        variants[name]=dict(episodes=selected,start_indices=list(range(count)),
            counts={kind:sum(e['categories'].count(kind) for e in selected) for kind in ['cold','warm','goal_switch']},
            measured_calls=len(selected)*20,unrelated_warmup_calls_per_cell=1,
            status='held input bank; selection and Jetson duration discussion before execution')
    if any(np.array_equal(warmup,s) for s in starts):
        raise ValueError('Warm-up duplicates episode cold state')
    return arrays,variants


def screen(arrays, urdf):
    import pinocchio as pin
    model=pin.buildModelFromUrdf(str(urdf)); model.gravity.linear=np.array([0.,0.,-9.81])
    if model.nq!=7 or model.nv!=7: raise ValueError('Require original7DOF arm model')
    data=model.createData(); frame=model.getFrameId('arm_right_tool_link')
    def pose(q):
        pin.framesForwardKinematics(model,data,np.asarray(q,dtype=np.float64))
        placement=data.oMf[frame]
        return placement.translation.copy(),placement.rotation.copy()
    distances=[]
    for state in arrays['starts']:
        if not np.isfinite(state).all() or np.any(state[:7]<model.lowerPositionLimit) or np.any(state[:7]>model.upperPositionLimit) or np.any(abs(state[7:])>model.velocityLimit):
            raise ValueError('Invalid start state/limit')
        xyz,_=pose(state[:7]); row=np.linalg.norm(arrays['goals'][:,:3]-xyz,axis=1)
        if np.any(row<=.05): raise ValueError('Start is inside rough5cm region')
        distances.append(row.tolist())
    for q,goal in zip(arrays['endpoint_q_rad'],arrays['goals']):
        xyz,rotation=pose(q)
        if np.max(abs(xyz-goal[:3]))>1e-12 or np.max(abs(rotation-pin.rpy.rpyToMatrix(*goal[3:])))>1e-12:
            raise ValueError('Approved endpoint FK does not match unchanged goal')
    duration=8.; times=np.linspace(0.,duration,int(np.ceil(duration/.003))+1)
    z=times/duration; h=10*z**3-15*z**4+6*z**5
    hd=(30*z**2-60*z**3+30*z**4)/duration
    hdd=(60*z-180*z**2+120*z**3)/duration**2
    curves={}; results=[]
    routes=[(f'start_{s}_goal_{g}',state[:7].astype(np.float64),q)
            for s,state in enumerate(arrays['starts']) for g,q in enumerate(arrays['endpoint_q_rad'])]
    routes += [(f'switch_goal_{g}_to_{(g+1)%3}',q,arrays['endpoint_q_rad'][(g+1)%3])
               for g,q in enumerate(arrays['endpoint_q_rad'])]
    for name,q0,q1 in routes:
        d=q1-q0; q=q0+h[:,None]*d; v=hd[:,None]*d; a=hdd[:,None]*d
        u=np.asarray([pin.rnea(model,data,qi,vi,ai).copy() for qi,vi,ai in zip(q,v,a)])
        margins={'q_rad':float(min(np.min(q-model.lowerPositionLimit),np.min(model.upperPositionLimit-q))),
                 'v_rad_s':float(np.min(model.velocityLimit-abs(v))),
                 'u_Nm':float(np.min(model.effortLimit-abs(u)))}
        passed=bool(np.isfinite(u).all() and all(m>0 for m in margins.values()))
        results.append(dict(id=name,positive_sampled_margins=passed,margins=margins,
            necessary_velocity_time_bound_s=float(np.max(abs(d)/model.velocityLimit)),duration_s=duration))
        for suffix,value in [('q',q),('v',v),('a',a),('u',u)]:curves[name+'_'+suffix]=value
    curves['time_s']=times
    report=dict(scope='CPU sampled8s quintic/RNEA witnesses only; not optimizer seeds or executed prefixes',
        start_goal_distance_m=distances,all_starts_outside5cm=True,all_start_q_qd_limits_pass=True,
        full_T01_minimum_initial_distance_m=.1,qualifies_T01_or_canonical=False,
        routes=results,sample_count=len(times),maximum_sample_step_s=float(np.max(np.diff(times))),
        all_routes_pass=all(r['positive_sampled_margins'] for r in results),
        unqualified=['collision','acceleration-limit compliance (no model acceleration bound)',
                     'between-sample compliance','tracking transient/dynamic feasibility',
                     'short .3s/.6s closed-loop performance'],numpy_version=np.__version__,pinocchio_version=pin.__version__)
    return report,curves


def freeze(output, root=ROOT):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    try:
        task,endpoints=source_inputs(root); arrays,variants=arrays_and_schedules(task,endpoints)
        for name,source in [('source_t01.json',root/TASK),('source_endpoints.json',root/ENDPOINTS),
                            ('model.urdf',root/MODEL),('freeze_inputs.py',Path(__file__))]:
            shutil.copyfile(source,output/name)
        np.savez_compressed(output/'arrays.npz',**arrays)
        report,curves=screen(arrays,output/'model.urdf')
        np.savez_compressed(output/'witnesses.npz',**curves); write_json(output/'screening.json',report)
        write_json(output/'bank.json',dict(schema='held-F01-input-bank-v1',variants=variants,
            N=[16,32,64,128],SQP=[1,2,5,10],PCG=[100,200,400,800],batch_size=1,
            timing_mode='A',offline_timing='controller_dt',dt_s=.03,sim_dt_s=.003,
            warmup=dict(category='process_cuda_warmup',excluded_from_measured_categories=True,
                state_array='warmup_state',goal_array='warmup_goal',standard_initial_guess=True),
            initial_guess_policy='public constant-current-state zero-control; shifted previous output thereafter',
            warm_start_shift_s=.03,goal_switch_preserves_warm_start=True,
            presets=dict(reaching=task['reaching_solver_params'],tracking=task['tracking_solver_params']),
            reference_sha256=[t['reference_sha256'] for t in task['tasks'][3:]],
            reference_periods_s=[t['period_s'] for t in task['tasks'][3:]],
            array_dtypes={'starts':'<f4','goals':'<f8','endpoint_q_rad':'<f8','references':'<f8','warmup_state':'<f4'},
            release_status='held: CPU preparation only; independent review/implementation/plan and Jetson discussion pending'))
        commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
        write_json(output/'provenance.json',dict(source_commit=commit,production_core='16b33719b4f0fd463eec7262c99b16b0b324366f',
            source_sha256={TASK:TASK_HASH,ENDPOINTS:ENDPOINT_HASH,MODEL:MODEL_HASH},generator_sha256=sha(__file__),
            generator_command='python3 experiments/frequency_campaign/freeze_inputs.py --output <fresh-directory>',
            cpu_only=True,gpu_calls=0))
        if not report['all_routes_pass']:raise ValueError('Sampled dynamic screen failed; evidence held')
    except BaseException as error:
        write_json(output/'preparation_failure.json',dict(type=type(error).__name__,error=str(error),gpu_calls=0))
        write_json(output/'SHA256SUMS.json',{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})
        raise
    write_json(output/'SHA256SUMS.json',{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    validate(output,root)
    return report


def validate(output,root=ROOT):
    output=Path(output); task,endpoints=source_inputs(root)
    manifest=json.loads((output/'SHA256SUMS.json').read_text())
    expected={'arrays.npz','bank.json','freeze_inputs.py','model.urdf','provenance.json','screening.json','source_endpoints.json','source_t01.json','witnesses.npz'}
    if set(manifest)!=expected:raise ValueError('Incomplete/failed or extra input artifacts')
    for name,digest in manifest.items():
        if sha(output/name)!=digest:raise ValueError('Bank artifact bytes changed: '+name)
    arrays,variants=arrays_and_schedules(task,endpoints)
    with np.load(output/'arrays.npz',allow_pickle=False) as saved:
        if set(saved.files)!=set(arrays):raise ValueError('Array keys changed')
        for name,a in arrays.items():
            b=saved[name]
            if b.dtype!=a.dtype or b.shape!=a.shape or b.tobytes()!=a.tobytes():raise ValueError('Array definition changed: '+name)
    bank=json.loads((output/'bank.json').read_text())
    if bank['variants']!=variants or bank['reference_sha256']!=[t['reference_sha256'] for t in task['tasks'][3:]]:
        raise ValueError('Episode/reference definition changed')
    fixed=dict(N=[16,32,64,128],SQP=[1,2,5,10],PCG=[100,200,400,800],batch_size=1,
        timing_mode='A',offline_timing='controller_dt',dt_s=.03,sim_dt_s=.003,
        warm_start_shift_s=.03,goal_switch_preserves_warm_start=True,
        presets=dict(reaching=task['reaching_solver_params'],tracking=task['tracking_solver_params']),
        reference_periods_s=[t['period_s'] for t in task['tasks'][3:]],
        warmup=dict(category='process_cuda_warmup',excluded_from_measured_categories=True,
            state_array='warmup_state',goal_array='warmup_goal',standard_initial_guess=True))
    if any(bank.get(key)!=value for key,value in fixed.items()):raise ValueError('Bank configuration changed')
    for name,digest in [('model.urdf',MODEL_HASH),('source_t01.json',TASK_HASH),
                        ('source_endpoints.json',ENDPOINT_HASH),('freeze_inputs.py',sha(__file__))]:
        if sha(output/name)!=digest:raise ValueError('Frozen source/model/generator changed: '+name)
    report,curves=screen(arrays,output/'model.urdf')
    if json.loads((output/'screening.json').read_text())!=report or not report['all_routes_pass']:
        raise ValueError('Screening data changed or failed')
    with np.load(output/'witnesses.npz',allow_pickle=False) as saved:
        if set(saved.files)!=set(curves):raise ValueError('Witness keys changed')
        for name,value in curves.items():
            if saved[name].dtype!=value.dtype or saved[name].shape!=value.shape or not np.array_equal(saved[name],value):
                raise ValueError('Witness values changed: '+name)
    provenance=json.loads((output/'provenance.json').read_text())
    if provenance.get('generator_sha256')!=sha(__file__) or provenance.get('source_sha256')!={TASK:TASK_HASH,ENDPOINTS:ENDPOINT_HASH,MODEL:MODEL_HASH}:
        raise ValueError('Provenance source identity changed')
    return bank


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); result=freeze(args.output)
    print(json.dumps(dict(all_routes_pass=result['all_routes_pass'],routes=len(result['routes']),gpu_calls=0)))
