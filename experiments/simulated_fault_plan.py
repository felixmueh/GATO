#!/usr/bin/env python3
"""Prepare the reviewed source's exact fault cases, with live setup held pending."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
SOURCE='a3a96ad3deabea5c0165da69addd1c55406710a9'


def prepare(output=None):
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse',SOURCE],text=True).strip()
    hooks=subprocess.check_output(['git','-C',str(ROOT),'show',commit+':tiago_examples/fault_campaign/hooks.py'],text=True)
    faults=next(ast.literal_eval(n.value) for n in ast.parse(hooks).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CASES' for t in n.targets))
    c3po=subprocess.check_output(['git','-C',str(ROOT),'show',commit+':tiago_examples/fault_campaign/c3po.py'],text=True)
    call=next(n for n in ast.walk(ast.parse(c3po)) if isinstance(n,ast.Call) and any(isinstance(a,ast.Constant) and a.value=='--cases' for a in n.args))
    order=next(ast.literal_eval(k.value) for k in call.keywords if k.arg=='default')
    command=['{python}','{runner}/adapters/simulated_fault_trial.py','--source','{source}','--build','{build}',
        '--simulator-config','{runner}/collision_local.json','--output','{output}']
    plan=dict(version=1,experiments=[dict(id='M03',research_question='RQ-monitor',source_commit=commit,
        protocol='unchanged-live-fault-campaign-v1',machines=['rtx5060'],timing_mode='live',backend='Gazebo',timeout_seconds=600,
        build=dict(inputs={p:hashlib.sha256((ROOT/'experiments'/p).read_bytes()).hexdigest() for p in (
            'native/CMakeLists.txt','native/control_studies/CMakeLists.txt','native/task_readiness/CMakeLists.txt')},commands=[
            ['cmake','-S','{runner}/native/control_studies','-B','{build}','-DGATO_SOURCE_DIR={source}','-DKNOTS=32;64',
             '-DCMAKE_BUILD_TYPE=Release','-DGATO_EXPERIMENT_TELEMETRY=ON','-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}-real','-DPython3_EXECUTABLE={python}'],
            ['cmake','--build','{build}','--parallel','2']],artifacts=['modules/bsqpN32_tiago_right*.so','modules/bsqpN64_tiago_right*.so','cuda_access_smoke']),
        preflight=dict(group='M03-live-simulator',command=command+['--preflight']),command=command+['--case','{case}'],
        cases=[dict(id=f'{i:02d}-'+fault,parameters=dict(fault=fault,description=faults[fault])) for i,fault in enumerate(order) if i>0])],
        status='prepared_reviewed_source_target_native_canonical_live_and_simulator_setup_pending',
        prerequisites=dict(config='Ignored experiments/collision_local.json: simulation_only true, domain1or91, canonical_live_preflight_receipt, pal_controller_log',
            canonical_receipt='Existing D01 metrics.json with task/physical/handback pass and backendgazebo; sibling protocol.json records N32/.03 qd=.1/vel_lim=.1 and provenance.json source_sha256 matches relevant runtime bytes. No invented receipt conversion.',
            reset='Host must perform separately logged assisted simulator restoration before each trial, as original c3po orchestration. Fresh PAL default-controller observations are required for every case.',
            host='Existing PAL simulator, GATO runtime/ROS dependencies, loopback-only DDS and current PAL-side observation log. This runner does not create Docker containers or infer host mount paths.',
            baseline='Original leading baseline is the family preflight, followed by the original20 remaining trials including trailing baseline.21 original ordered trials total; no extra baseline repeat.',
            safety='Simulation only. Original controller/executor/publication/restoration negative behavior preserved; no ROS or cleanup repair.'),
        original_order=order,original_leading_baseline_is_preflight=True,
        historical='External robot_faults/2026-09-21 records recovery pending; no historical result overwritten')
    output=Path(output) if output else ROOT/'experiments/simulated_faults.json';output.write_text(json.dumps(plan,indent=2)+'\n');return plan


if __name__=='__main__':print(len(prepare()['experiments'][0]['cases']),'unchanged simulator cases prepared')
