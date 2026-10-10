"""Small pinned receipt adapter. Validation precedes native imports."""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CORE = '16b33719b4f0fd463eec7262c99b16b0b324366f'
RUNTIME = ('python/bsqp/interface.py', 'python/bsqp/common.py', 'python/bsqp/offline_mpc.py',
           'tiago_src/gato_tiago/tiago_mpc_controller.py', 'tiago_src/gato_tiago/config.py',
           'gato/dynamics/tiago_right/tiago_right_arm.urdf')
COLLECTOR = tuple('experiments/frequency_campaign/' + name for name in
                  ('episode.py', 'record_bank.py', 'protocol.py', 'run.py', 'factory.py', 'readiness.py', 'release.py', 'build_isolated.py'))


def load(name):
    spec = importlib.util.spec_from_file_location('f01_' + name, HERE / (name + '.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


P = load('protocol')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path): return json.loads(Path(path).read_text())


def artifact(item):
    path = Path(item['path']).resolve()
    if sha(path) != item['sha256']: raise ValueError('Pinned evidence bytes changed: ' + str(path))
    return path


def evidence(item): return read(artifact(item))


def selected_cases():
    bank, _ = P.load_bank()
    selection = read(HERE / 'selected_variant.json')
    review = read(HERE / 'selection_review.json')
    if review['verdict'] != 'approved' or sha(HERE / 'selected_variant.json') != review['selection']['sha256']:
        raise ValueError('Reviewed selection bytes differ')
    if selection['selected_variant'] != 'short12' or selection['input_manifest_sha256'] != P.BANK_MANIFEST_SHA256:
        raise ValueError('Frozen short12 selection differs')
    return [dict(variant='short12', N=N, SQP=S, PCG=C, timing_mode='A', backend='internal')
            for N in bank['N'] for S in bank['SQP'] for C in bank['PCG']]


def canonical_config():
    p = P.preflight_protocol()
    # Only physical/task/solver values bind readiness, not evolving documentation.
    keys = ('initial_state', 'goal', 'N', 'dt_s', 'sim_dt_s', 'timeout_s', 'solver_params',
            'position_threshold_m', 'axis_threshold_deg', 'speed_L1_threshold_rad_s',
            'goal_dwell_time_s', 'final_approach', 'timing_mode', 'offline_timing')
    return {key: p[key] for key in keys}


def canonical_hash():
    return hashlib.sha256(json.dumps(canonical_config(), sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate(manifest_path, expected_hash, machine, N):
    if sha(manifest_path) != expected_hash: raise ValueError('Reviewed release manifest differs')
    manifest = read(manifest_path)
    if manifest.get('status') != 'released_F01_A': raise RuntimeError('F01 release held: ' + manifest.get('reason', 'no reviewed release'))
    if machine not in P.MACHINES or manifest['machine'] != machine: raise ValueError('Release machine differs')
    if manifest['production_core_commit'] != CORE: raise ValueError('Approved production core differs')
    commit = manifest['collector_source_commit']
    if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit): raise ValueError('Pinned full source commit required')
    if set(manifest['collector_source_sha256']) != set(COLLECTOR): raise ValueError('Collector source receipt incomplete')
    for name, digest in manifest['collector_source_sha256'].items():
        if sha(ROOT / name) != digest: raise ValueError('Collector source bytes differ: ' + name)
    if manifest['bank_manifest_sha256'] != P.BANK_MANIFEST_SHA256 or manifest['selection_sha256'] != sha(HERE / 'selected_variant.json'):
        raise ValueError('Bank/selection release differs')
    acceptance = evidence(manifest['telemetry_acceptance'])
    if acceptance['decision'] != 'accept_existing_first_instrumentation_without_equivalence_gate' or acceptance['numerical_equivalence_required'] is not False:
        raise ValueError('First telemetry acceptance is missing')
    if manifest['observer_patch_sha256'] != acceptance['accepted_native_patch_sha256']:
        raise ValueError('Observer version differs from Felix acceptance')
    review = evidence(manifest['build_review'])
    if review.get('verdict') != 'approved': raise ValueError('Native build/source review unapproved')
    reason = evidence(manifest['reason_review'])
    if (manifest['reason_review']['sha256'] != acceptance['comparison_audit']['sha256']
            or not reason.get('telemetry_reason_validation')): raise ValueError('Actual accepted native reason audit absent')
    suite = evidence(manifest['task_suite'])
    if (manifest['task_suite'].get('machine') != machine or suite.get('status') != 'passed'
            or suite.get('exit_code') != 0 or suite.get('fixed_task_processes') != 3 or suite.get('error') is not None
            or suite.get('restored_patch_sha256') != manifest['observer_patch_sha256']
            or not suite.get('module_before_sha256') or suite.get('module_after_sha256') != suite.get('module_before_sha256')):
        raise ValueError('Required same-machine ON task suite missing/negative/stale')
    modules = {}
    for key in {str(N), '32'}:
        info = manifest['modules'][key]
        build = evidence(info['build_receipt'])
        if build.get('variant') != 'ON' or build['source_commit'] != manifest['collector_source_commit']:
            raise ValueError('Compiled observer identity differs')
        if build['flags'].get('GATO_EXPERIMENT_TELEMETRY') is not True or build['flags'].get('N') != int(key) or build['flags'].get('plant') != 'tiago_right':
            raise ValueError('Compiled target differs')
        source = Path(build['source_root']).resolve()
        module = Path(info['path']).resolve()
        if module != Path(build['module']['path']).resolve() or sha(module) != info['sha256'] or build['module']['sha256'] != info['sha256']:
            raise ValueError('Actual native module differs')
        if module.parent != source / 'python/bsqp': raise ValueError('Native module is outside isolated source')
        cache = artifact(build['cmake_cache']).read_text()
        if ('GATO_EXPERIMENT_TELEMETRY:BOOL=ON' not in cache or f'KNOTS:STRING={key}' not in cache
                or 'PLANT:STRING=tiago_right' not in cache): raise ValueError('Actual compiled target/cache differs')
        deps = evidence(build['source_receipt'])
        if deps['source_commit'] != commit: raise ValueError('Archived source receipt commit differs')
        deps = dict(local_source_files=list(deps['source_file_sha256']), external_dependency_sha256={}, depfile_sha256={})
        for name in deps['local_source_files']:
            if sha(source / name) != build['source_file_sha256'][name]: raise ValueError('Compiled source dependency differs')
        for name, digest in {**deps.get('external_dependency_sha256', {}), **deps.get('depfile_sha256', {})}.items():
            if sha(name) != digest: raise ValueError('Compiled external/depfile identity differs')
        for name in RUNTIME:
            if sha(source / name) != build['source_file_sha256'][name] or sha(source / name) != sha(ROOT / name):
                raise ValueError('Runtime source bytes differ: ' + name)
        modules[key] = dict(info, source_root=str(source), runtime_source_sha256={name: sha(source / name) for name in RUNTIME})
    return dict(manifest, manifest_sha256=expected_hash, modules=modules)


def readiness_guard(context, family_state):
    state = read(family_state)
    if state.get('status') != 'ready' or state.get('machine') != context['machine'] or state.get('release_manifest_sha256') != context['manifest_sha256']:
        raise RuntimeError('F01 family held/not ready on this machine')
    metrics = evidence(state['readiness_metrics'])
    if (metrics.get('machine') != context['machine'] or metrics.get('canonical_config_sha256') != canonical_hash()
            or metrics.get('module_sha256') != context['modules']['32']['sha256']
            or metrics.get('observer_patch_sha256') != context['observer_patch_sha256']
            or metrics.get('runtime_source_sha256') != context['modules']['32']['runtime_source_sha256']
            or metrics.get('timing_mode') != 'A' or metrics.get('task_success') is not True
            or metrics.get('physical_pass') is not True or metrics.get('execution_error') is not None):
        raise RuntimeError('Canonical same-source/module/machine A preflight failed/stale')
    return metrics


def write_state(path, context, status, **fields):
    raw = load('record_bank').raw
    raw.write_json(path, dict(status=status, machine=context['machine'],
                             release_manifest_sha256=context['manifest_sha256'], **fields))
