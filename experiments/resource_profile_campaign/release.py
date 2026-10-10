"""Small receipt guard; checked-in release stays held and native import is lazy."""
import json
from datetime import datetime, timezone
from pathlib import Path

from experiments.resource_profile_campaign.common import sha, write_json, F01, BUDGETS, DURATION_APPROVAL_SHA256


def artifact(item):
    path = Path(item['path']).resolve()
    if sha(path) != item['sha256']: raise ValueError('Pinned artifact differs: ' + str(path))
    return path


def read(item): return json.loads(artifact(item).read_text())


def key(N, profile): return f'N{N}-stage{"ON" if profile else "OFF"}'


def validate(path, digest, machine, case, family_state):
    path = Path(path)
    if sha(path) != digest: raise ValueError('Reviewed manifest hash differs')
    context = json.loads(path.read_text())
    if context.get('status') != 'released_F04_F05': raise RuntimeError('F04/F05 execution held: ' + context.get('reason', 'unreleased'))
    if len(context['source_commit'])!=40: raise ValueError('Full pinned source commit required')
    if context['machine'] != machine: raise ValueError('Released machine differs')
    review = read(context['implementation_review'])
    if review.get('verdict') != 'approved': raise ValueError('Implementation review missing/negative')
    acceptance = read(context['observer_acceptance'])
    if acceptance.get('decision') != 'accept_existing_first_instrumentation_without_equivalence_gate':
        raise ValueError('Accepted stop observer receipt differs')
    info = context['modules'][key(case['N'], case['stage_timing'])]; build = read(info['build_receipt'])
    expected = dict(GATO_EXPERIMENT_STAGE_TIMING=case['stage_timing'], GATO_EXPERIMENT_RESOURCE_QUERY=True, GATO_EXPERIMENT_TELEMETRY=True)
    if build['N'] != case['N'] or build['flags'] != expected or build['source_commit'] != context['source_commit']:
        raise ValueError('Compiled N/flags/source differ')
    source = Path(build['source_root']).resolve(); module = artifact(build['module'])
    if module != Path(info['path']).resolve() or sha(module) != info['sha256'] or module.parent != source / 'python/bsqp':
        raise ValueError('Actual module identity/path differs')
    cache = artifact(build['cmake_cache']).read_text()
    for name, value in expected.items():
        if f'{name}:BOOL={"ON" if value else "OFF"}' not in cache: raise ValueError('Compiled cache flag differs')
    if f'KNOTS:STRING={case["N"]}' not in cache or 'PLANT:STRING=tiago_right' not in cache: raise ValueError('Compiled plant/horizon differs')
    archive = read(build['source_receipt'])
    if archive['source_commit'] != context['source_commit']: raise ValueError('Archived source differs')
    for name, file_digest in archive['source_file_sha256'].items():
        if sha(source / name) != file_digest: raise ValueError('Archived file differs: ' + name)
    # Workers run from that source-owned archive, not an independently changed wrapper.
    for name in ('python/bsqp/interface.py','gato/bsqp/bsqp.cuh','experiments/resource_profile_campaign/run.py'):
        if sha(source / name) != sha(Path(__file__).resolve().parents[2] / name): raise ValueError('Worker/runtime source differs')
    state = json.loads(Path(family_state).read_text())
    if state.get('status') != 'ready': raise RuntimeError('Family held or interrupted prior worker remains running')
    if state.get('machine') != machine or state.get('release_manifest_sha256') != digest: raise ValueError('Family state identity differs')
    if case['study'] == 'F05':
        if context['duration_approval']['sha256'] != DURATION_APPROVAL_SHA256:
            raise ValueError('Pinned Felix duration/coverage approval differs')
        approval=read(context['duration_approval'])
        if (approval['decision']!='use_120_seconds_single_sustained_block_per_arm'
            or approval['sustained_seconds_per_arm']!=120 or approval['sustained_repeats']!=1
            or approval['sustained_blocks_per_machine']!=6 or approval['budgets']!=[list(x) for x in BUDGETS]
            or approval['B']!=[1,16]):
            raise ValueError('Approved duration/coverage differs')
        readiness = read(context['canonical_A_readiness'])
        off_module = context['modules'][key(32, False)]
        if (readiness.get('status') != 'ready' or readiness.get('task_success') is not True
            or readiness.get('physical_pass') is not True or readiness.get('execution_error') is not None
            or readiness.get('machine') != machine or readiness.get('source_commit') != context['source_commit']
            or readiness.get('module_sha256') != off_module['sha256'] or readiness.get('timing_mode') != 'A'
            or readiness.get('canonical_config_sha256') != F01.canonical_hash()):
            raise RuntimeError('Canonical same-source/machine stage-OFF A readiness absent/negative')
        if case['arm'] == 'sustained' and context.get('sustained_duration_s')!=case['duration_s']:
            raise RuntimeError('Released duration differs from approved120s case')
    return dict(context, manifest_sha256=digest, selected_module=dict(info, source_root=str(source), flags=expected))


def hold(path, context, error, output):
    state=json.loads(Path(path).read_text())
    state.update(status='held',machine=context['machine'],release_manifest_sha256=context['manifest_sha256'],reason=str(error),failed_attempt=str(output))
    write_json(path,state)


def begin(path, context, case, output):
    state=json.loads(Path(path).read_text())
    if state.get('status')!='ready': raise RuntimeError('Family not ready before import')
    state.update(status='running',current_attempt=dict(case=case,attempt=str(output),current_raw_path=str(output),
        start_utc=datetime.now(timezone.utc).isoformat(),status='running'))
    write_json(path,state)


def complete(path, context, output):
    state=json.loads(Path(path).read_text())
    if (state.get('status')!='running' or state['current_attempt']['attempt']!=str(output)
        or state.get('release_manifest_sha256')!=context['manifest_sha256']):
        raise RuntimeError('Running family marker differs during finalization')
    state['current_attempt'].update(status='completed',completed_utc=datetime.now(timezone.utc).isoformat())
    state['status']='ready'
    write_json(path,state)
