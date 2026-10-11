#!/usr/bin/env python3
"""Prepare bounded current control choices; historical sweeps remain evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
SOURCE='502fa29c8b0d649dcb347f9534bf29c6935fa6dd'
PROTOCOL='bounded-control-choices-B-v2'


def variants():
    return [
        dict(id='current-default',study='shared-baseline',name='Current defaults',weights={}),
        dict(id='position-half',study='C01',name='Half running position weight',weights={'q_cost':2.}),
        dict(id='orientation-off',study='C01',name='Orientation costs off',weights={'ee_orient_cost':0.,'ee_orient_N_cost':0.}),
        dict(id='switch-20mm',study='C03',name='Historical factor10 switch at20mm',switch_distance_m=.02),
        dict(id='switch-75mm',study='C03',name='Historical factor10 switch at75mm',switch_distance_m=.075),
        dict(id='final-approach',study='C04',name='Existing final approach',final_approach=True),
        dict(id='final-approach-perturbed',study='C04',name='Final approach with +.01rad joint1 start',final_approach=True,initial_delta_rad=.01),
        dict(id='default-perturbed',study='C04',name='Defaults with +.01rad joint1 start',initial_delta_rad=.01),
        dict(id='damping-low',study='C05',name='Damping .01',weights={'qd_cost':.01}),
        dict(id='damping-high',study='C05',name='Damping1',weights={'qd_cost':1.}),
        dict(id='speed-limit-low',study='C05',name='Speed limit penalty .001',weights={'vel_lim_cost':.001}),
        dict(id='speed-limit-high',study='C05',name='Speed limit penalty1',weights={'vel_lim_cost':1.}),
        dict(id='terminal-half',study='C05',name='Half terminal position weight',terminal_factor=.5),
        dict(id='terminal-double',study='C05',name='Double terminal position weight',terminal_factor=2.),
        dict(id='holding-default',study='C06',name='Nominal holding with current effort cost',holding=True),
        dict(id='holding-gripper-upper',study='C06',name='Holding with narrower-upper gripper plant, unchanged nominal optimizer',holding=True,gripper_angle_rad=.8028514559173916),
        dict(id='holding-no-effort-cost',study='C06',name='Nominal holding with effort cost0',holding=True,weights={'u_cost':0.}),
        dict(id='warm-unshifted',study='C08',name='Unshifted previous output reanchored',seed_policy='unshifted'),
        dict(id='warm-cold',study='C08',name='Fresh constant state seed each solve',seed_policy='cold')]


def prepare(output=None,jetson=False):
    hashes={}
    for name in ('v01.json','t01.json'):
        payload=subprocess.check_output(['git','-C',str(ROOT),'show',SOURCE+':experiments/campaign/inputs/'+name])
        relative='experiments/inputs/control_studies/'+name;target=ROOT/relative
        if target.exists() and target.read_bytes()!=payload:raise ValueError('Frozen control task input differs')
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(payload)
        hashes[relative]=hashlib.sha256(payload).hexdigest()
    relative='experiments/inputs/control_studies/tiago_right_arm_include_gripper.urdf'
    payload=subprocess.check_output(['git','-C',str(ROOT),'show',SOURCE+':gato/dynamics/tiago_right/tiago_right_arm_include_gripper.urdf'])
    target=ROOT/relative;target.write_bytes(payload);hashes[relative]=hashlib.sha256(payload).hexdigest()
    t01=json.loads((ROOT/'experiments/inputs/control_studies/t01.json').read_text())
    selected=variants()
    questions={
        'C01':'Do running position/orientation choices change bounded current reaching behavior?',
        'C03':'Does the retained factor10 latched20/75mm switch change progress during a fixed8s observation?',
        'C04':'What changes under existing final approach, including one declared small start perturbation?',
        'C05':'Separate damping, speed-limit penalty and terminal position weight on fixed tasks.',
        'C06':'Finite8s holding offset: current versus zero effort regularization, and nominal-zero wider plant versus actual URDF narrower-upper gripper inertia with unchanged nominal native optimizer; no equilibrium/physical-drive claim.',
        'C08':'Does correctly shifted warm start affect task outcomes relative to unshifted/reanchored or cold seeds?'}
    protocol=dict(id=PROTOCOL,source_commit=SOURCE,variants=selected,reaching_tasks=['goal-1','goal-2'],
        tracking_task='figure8-24s',fixed_reaching_observation_s=8.,observation_dwell_s=8.03,
        rough_success=dict(position_m=.05,axis_deg=30.,speed_L1_rad_s=1.,dwell_s=0.),
        questions=questions,baseline_reuse='Shared current-default goal1/goal2 and figure8 baseline, no per-study reruns',
        C03_policy=dict(factor=10.,speed_max_rad_s=.1,scaled=['q_cost','qd_cost','ee_orient_cost'],
            terminal_equalized_to_scaled_running=['N_cost','ee_orient_N_cost']),
        C04_final_params=dict(q_cost=400.,N_cost=1200.,qd_cost=1.,u_cost=.0003,ee_orient_cost=10.,ee_orient_N_cost=10.,max_sqp_iters=20,mu=30.),
        C04_trigger=dict(distance_m=.075,max_joint_speed_rad_s=.1),
        C06_holding='Pose reference = FK of approved initial state; public goal regulation with explicit full-window dwell stop convention.',
        C06_gripper=dict(asset='tiago_right_arm_include_gripper.urdf',asset_sha256=hashes['experiments/inputs/control_studies/tiago_right_arm_include_gripper.urdf'],
            source_asset_commit='3303eafcd393a74dcb26fb031d0815e84511e540',baseline='holding-default uses current nominal-zero wider lumped plant',
            variation='holding-gripper-upper locks actual six finger joints at driver upper0.8028514559173916rad with URDF mimic multipliers/offsets; narrower-upper plant only',
            optimizer='unchanged nominal-zero native solver/model, caps, costs and modeB',model_mismatch_intentional=True,
            scope='Fixed finger configuration alters rigid-body gripper inertial distribution only; no added payload, friction, drive response, gripper motion or collision claim',
            reused_evidence='Historical three-case zero-gripper lumping check passed; reused without numerical-equivalence rerun',
            CPU_difference_at_holding_start='About0.005Nm maximum gravity-vector difference; not measured closed-loop improvement'),
        unsupported={},
        historical_evidence=['/workspace/gato-tiago-reaching-cost-sweep','/workspace/gato-tiago-reaching-vel01',
            '/workspace/gato-tiago-reaching-weight-switch','/workspace/gato-tiago-reaching-weight-switch-75mm-far',
            '/workspace/gato-tiago-final-approach-tuning','/workspace/gato-tiago-mpc-final-weights'],
        stopping='One planned attempt per case. Scientific failures retained, independent cases continue; no tuning/retry-green.',
        qualifications='CPU preparation only. Genuine unchanged target task suite + canonical modeB preflight precede execution. New behavior is not GPU-qualified.',
        C08_scope='Same task/source/caps/cold start; subsequent states and optimized seeds differ. No identical-input speedup or determinism claim.')
    relative='experiments/inputs/control_studies/protocol.json';target=ROOT/relative
    payload=(json.dumps(protocol,indent=2)+'\n').encode();target.write_bytes(payload);hashes[relative]=hashlib.sha256(payload).hexdigest()
    cases=[]
    for variant in selected:
        if jetson and variant['study'] not in ('C08','shared-baseline'):continue
        tasks=['goal-1'] if variant.get('holding') else ['goal-1','goal-2']
        if not jetson and variant['id'] in ('current-default','terminal-half','terminal-double'):tasks+=['figure8-24s']
        for task in tasks:
            cases.append(dict(id=variant['id']+'-'+task,parameters=dict(variant=variant['id'],base_task=task)))
    helper_paths=('native/CMakeLists.txt','native/control_studies/CMakeLists.txt','native/task_readiness/CMakeLists.txt')
    build=dict(inputs={p:hashlib.sha256((ROOT/'experiments'/p).read_bytes()).hexdigest() for p in helper_paths},commands=[
        ['cmake','-S','{runner}/native/control_studies','-B','{build}','-DGATO_SOURCE_DIR={source}','-DKNOTS=32',
         '-DCMAKE_BUILD_TYPE=Release','-DGATO_EXPERIMENT_TELEMETRY=ON','-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}-real','-DPython3_EXECUTABLE={python}'],
        ['cmake','--build','{build}','--parallel','2']],artifacts=['modules/bsqpN32_tiago_right*.so','cuda_access_smoke'])
    command=['{python}','{runner}/adapters/control_studies.py','--source','{source}','--build','{build}',
        '--inputs','{inputs}/experiments/inputs/control_studies','--output','{output}']
    plan=dict(version=1,experiments=[dict(id='ControlChoices',research_question='RQ-control-choices',protocol=PROTOCOL,
        source_commit=SOURCE,machines=['jetson'] if jetson else ['gtx1070','rtx5060'],timing_mode='B',backend='internal',inputs=hashes,build=build,
        command=command+['--case','{case}'],preflight=dict(group='ControlChoices-B',command=command+['--preflight']),cases=cases,
        timeout_seconds=600)],status='proposed_target_duration_discussion_pending' if jetson else 'prepared_target_validation_pending',
        unsupported=protocol['unsupported'],note='Historical sweeps and failures preserved. Fixed8s observations report rough success separately from public timeout. No full parameter optimization.')
    output=Path(output) if output else ROOT/'experiments'/('control_studies_jetson_proposed.json' if jetson else 'control_studies.json')
    output.write_text(json.dumps(plan,indent=2)+'\n');return plan


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path);parser.add_argument('--jetson-proposed',action='store_true')
    args=parser.parse_args();p=prepare(args.output,args.jetson_proposed);print(len(p['experiments'][0]['cases']),'prepared cases')
