#!/usr/bin/env python3
"""Prepare frozen BB plans; never build or run experiments during generation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = '4f6c8fd4be428c135fc16564c5dd56eb61ce5660'
ORIGINAL_SOURCE = 'aa241da65fce3e062e62168846b8d5d589b5cd90'
HISTORICAL = Path('/workspace/GATO-eval-batching/example_artifacts/bb_campaign/20261010T211939.429686Z')
VALIDATION = Path('/workspace/GATO-eval-batching/experiments/tiago_batching/BB_VALIDATION.json')
PROTOCOL = 'simple-bb-recording-v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def freeze(relative, payload, hashes):
    target = ROOT / relative
    if target.exists() and target.read_bytes() != payload:
        raise ValueError('Existing frozen BB input differs: '+str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    hashes[relative] = sha(target)


def prepare(output=None):
    hashes = {}
    for name in ('inputs/pool.npz', 'inputs/pairs.npz', 'bb_protocol.json', 'bb_preflight.json'):
        payload = subprocess.check_output(['git', '-C', str(ROOT), 'show', SOURCE+':experiments/tiago_batching/'+name])
        freeze('experiments/inputs/batching/'+Path(name).name, payload, hashes)
    bank = json.loads((ROOT/'experiments/inputs/batching/bb_protocol.json').read_text())
    proposal = json.loads((HISTORICAL/'continuation-proposal/proposal.json').read_text())
    status = json.loads((HISTORICAL/'raw/status.json').read_text())
    full = [(p,b) for p in bank['task_ids'] for b in (1,16)]
    pending = [(e['pair_id'],e['batch']) for e in status['episodes'] if e['state']=='pending']
    proposed = [(e['pair_id'],e['batch']) for e in proposal['remaining_episodes']]
    retained = [dict(pair_id=e['pair_id'],batch=e['batch'],state=e['state'],attempts=e['attempts'])
                for e in status['episodes'] if e['state']!='pending']
    if (pending != proposed or len(pending)!=176 or len(retained)!=24 or
            [ (e['pair_id'],e['batch']) for e in status['episodes']] != full or
            sum(e['state']=='completed' for e in retained)!=23 or
            [(e['pair_id'],e['batch']) for e in retained if e['state']!='completed']!=[(2342,16)]):
        raise ValueError('BB retained/pending coverage differs; preserve evidence and review the queue')
    continuation = dict(original_source_commit=ORIGINAL_SOURCE,
        original_cohort=str(HISTORICAL/'raw'), original_status_sha256=sha(HISTORICAL/'raw/status.json'),
        retained=retained, failed_episode=proposal['failed_episode'], pending=[dict(pair_id=p,batch=b) for p,b in pending],
        accounting='23 completed + 1 retained execution failure + 176 pending = 200; no reruns',
        policy='Current independent-case policy supersedes historical family hold; execution errors remain execution errors.')
    freeze('experiments/inputs/batching/continuation.json', (json.dumps(continuation,indent=2)+'\n').encode(),hashes)
    validation = json.loads(VALIDATION.read_text())
    freeze('experiments/inputs/batching/gtx1070_readiness.json', VALIDATION.read_bytes(), hashes)
    unchanged = ('python/bindings.cu', 'python/bsqp/common.py', 'python/bsqp/interface.py',
        'python/bsqp/mpc_controller.py',
        'tiago_src/gato_tiago/config.py', 'examples/randomized_multimodal/tiago_batching_execution.py')
    runtime_hashes = {}
    for relative in unchanged:
        before = subprocess.check_output(['git','-C',str(ROOT),'show',ORIGINAL_SOURCE+':'+relative])
        after = subprocess.check_output(['git','-C',str(ROOT),'show',SOURCE+':'+relative])
        if before!=after:raise ValueError('BB readiness runtime changed: '+relative)
        runtime_hashes[relative] = hashlib.sha256(after).hexdigest()
    core_ids = [subprocess.check_output(['git','-C',str(ROOT),'rev-parse',c+':gato'],text=True).strip()
                for c in (ORIGINAL_SOURCE,SOURCE)]
    if core_ids[0]!=core_ids[1]:raise ValueError('BB native source changed')
    reuse = dict(original_source_commit=ORIGINAL_SOURCE, recording_source_commit=SOURCE,
        unchanged_runtime_sha256=runtime_hashes, gato_tree=core_ids[0],
        reviewed_recording_only_changes=['examples/randomized_multimodal/tiago_reaching_batch_pilot.py',
            'examples/randomized_multimodal/validate_reaching_mpc.py'],
        scope='Recording hooks only; solve, timing, selection, task settings and native source unchanged')
    freeze('experiments/inputs/batching/source_reuse.json',(json.dumps(reuse,indent=2)+'\n').encode(),hashes)
    helper = 'bb_prepare.py'
    command = ['{python}','{runner}/adapters/bb.py','--source','{source}','--build','{build}',
        '--pool','{inputs}/experiments/inputs/batching','--protocol','{inputs}/experiments/inputs/batching/bb_protocol.json',
        '--case','{case}','--output','{output}']
    plans = {}
    for machine, roster in (('gtx1070',pending),('rtx5060',full),('jetson',full)):
        recipe_inputs = {helper:sha(ROOT/'experiments'/helper)}
        if machine=='gtx1070':
            native = validation['native_module']
            build = dict(inputs=recipe_inputs,commands=[['{python}','{runner}/'+helper,'copy-module',
                '--module',native['path'],'--sha256',native['sha256'],'--build','{build}']],
                artifacts=['modules/bsqpN32_tiago_right*.so'])
        else:
            recipe_inputs['native/CMakeLists.txt']=sha(ROOT/'experiments/native/CMakeLists.txt')
            build = dict(inputs=recipe_inputs, commands=[
                ['cmake','-S','{runner}/native','-B','{build}','-DGATO_SOURCE_DIR={source}',
                 '-DKNOTS=32','-DCMAKE_BUILD_TYPE=Release','-DGATO_EXPERIMENT_TELEMETRY=ON',
                 '-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}-real','-DPython3_EXECUTABLE={python}'],
                ['cmake','--build','{build}','--parallel','2']],artifacts=['modules/bsqpN32_tiago_right*.so'])
        preflight_command = ['{python}','{runner}/'+helper,'preflight','--source','{source}','--build','{build}',
            '--inputs','{inputs}/experiments/inputs/batching','--output','{output}']
        if machine=='gtx1070':preflight_command+=['--reuse-gtx1070']
        experiment = dict(id='BB',research_question='RQ-batching',protocol=PROTOCOL,source_commit=SOURCE,
            timing_mode='B',backend='internal',machines=[machine],inputs=hashes,build=build,
            preflight=dict(group='BB-B',command=preflight_command),command=command,timeout_seconds=600,
            original_source_commit=ORIGINAL_SOURCE,
            cases=[dict(id=f'pair-{p:05d}-B{b}',parameters=dict(pair_id=p,batch=b,timing_mode='B',backend='internal')) for p,b in roster])
        plan=dict(version=1,experiments=[experiment],
            status='proposed_not_executed_target_duration_discussion_pending' if machine=='jetson' else 'prepared_not_executed',
            retained_results={'BB':dict(gtx1070=continuation)},
            note='Frozen100-pair one-pass coverage; GPU use sequential. Jetson target readiness and duration/storage discussion precede execution; this plan authorizes no runs.')
        filename='bb_jetson_proposed.json' if machine=='jetson' else 'bb_'+machine+'.json'
        target=Path(output) / filename if output else ROOT/'experiments'/filename
        write(target,plan)
        plans[machine]=plan
    ba_hashes={k:v for k,v in hashes.items() if Path(k).name in ('pool.npz','pairs.npz','gtx1070_readiness.json','source_reuse.json')}
    for original,name in (('protocol.json','ba_protocol.json'),('preflight.json','ba_preflight.json')):
        payload=subprocess.check_output(['git','-C',str(ROOT),'show',SOURCE+':experiments/tiago_batching/'+original])
        freeze('experiments/inputs/batching/'+name,payload,ba_hashes)
    base=json.loads((ROOT/'experiments/inputs/batching/ba_protocol.json').read_text())
    nominal=dict(base,id='BA-nominal16x-A-v1',cohort_id='BA-nominal16x-A-v1',
        status='Prepared nominal16xSQP arm; no realized-equal-work claim',
        arms=[dict(batch_size=1,max_sqp_iters=80,max_pcg_iters=120)],solver_params=dict(base['solver_params'],max_sqp_iters=80))
    freeze('experiments/inputs/batching/ba_nominal16x.json',(json.dumps(nominal,indent=2)+'\n').encode(),ba_hashes)
    nominal_preflight=json.loads((ROOT/'experiments/inputs/batching/ba_preflight.json').read_text())
    nominal_preflight['solver_params']=nominal['solver_params']
    freeze('experiments/inputs/batching/ba_nominal16x_preflight.json',(json.dumps(nominal_preflight,indent=2)+'\n').encode(),ba_hashes)
    a_receipt=Path('/workspace/gato-campaign-recovery-20261010/ba-recovery/preflight-execution.json')
    freeze('experiments/inputs/batching/gtx1070_A_preflight.json',a_receipt.read_bytes(),ba_hashes)
    for machine in ('gtx1070','rtx5060'):
        bb=plans[machine]['experiments'][0]
        pf=list(bb['preflight']['command'])+['--mode','A','--ba-nominal-preflight']
        arms=[('nominal16x',1)] if machine=='gtx1070' else [('default',1),('default',16),('nominal16x',1)]
        experiment=dict(id='BA',research_question='RQ-batching',protocol='simple-ba-nominal16x-v1',source_commit=SOURCE,
            timing_mode='A',backend='internal',machines=[machine],inputs=ba_hashes,build=bb['build'],
            preflight=dict(group='BA-A',command=pf),timeout_seconds=600,
            command=['{python}','{runner}/adapters/ba.py','--source','{source}','--build','{build}',
                '--inputs','{inputs}/experiments/inputs/batching','--case','{case}','--output','{output}'],
            cases=[dict(id=f'pair-{p:05d}-B{b}-{arm}',parameters=dict(pair_id=p,batch=b,arm=arm))
                   for p in base['task_ids'] for arm,b in arms])
        plan=dict(version=1,experiments=[experiment],status='prepared_not_executed',retained_results=dict(
            BA={'gtx1070':'200 defaultA episodes retained at /home/felix/gato-results/gtx1070/BA-current-defaults-100-A-20261010T192610Z'}),
            note='Nominal16xSQP allocation is not matched realized compute. Existing default seeds/own outputs support saved-data diagnostics; no seed-only closed-loop arm.')
        target=Path(output)/('ba_'+machine+'.json') if output else ROOT/'experiments'/('ba_'+machine+'.json')
        write(target,plan)
    return plans


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory',type=Path)
    args=parser.parse_args()
    plans=prepare(args.output_directory)
    print(', '.join(f'{m}: {len(p["experiments"][0]["cases"])} cases' for m,p in plans.items()))
