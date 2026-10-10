"""Held F01 input/case/preflight guards. No controller/native imports."""
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BANK=ROOT/'experiments/frequency_campaign/inputs'
MACHINES=('gtx1070','rtx5060','jetson')
SOURCE_FILES=('experiments/frequency_campaign/episode.py','experiments/frequency_campaign/protocol.py',
    'experiments/frequency_campaign/run.py','experiments/frequency_campaign/record_bank.py',
    'python/bsqp/common.py','python/bsqp/interface.py','python/bsqp/offline_mpc.py',
    'tiago_src/gato_tiago/tiago_mpc_controller.py','tiago_src/gato_tiago/config.py')
EXECUTION_RELEASED=False
BANK_MANIFEST_SHA256='29f8ccdff73d36c339e50ea7153a46fb6ecd22b32da786b74f16cb9e1a61f609'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bank():
    if sha(BANK/'SHA256SUMS.json')!=BANK_MANIFEST_SHA256:raise ValueError('Pinned bank manifest differs')
    spec=importlib.util.spec_from_file_location('f01_freeze',ROOT/'experiments/frequency_campaign/runtime_bank.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    bank=module.validate(BANK)
    import numpy as np
    with np.load(BANK/'arrays.npz',allow_pickle=False) as saved: arrays={k:saved[k].copy() for k in saved.files}
    return bank,arrays


def validate_case(case,bank):
    params=case.get('parameters',{})
    if set(params)!={'variant','N','SQP','PCG','timing_mode','backend'}:raise ValueError('Exact frozen case keys required')
    for key in ['N','SQP','PCG']:
        if type(params[key]) is not int or params[key] not in bank[key]:raise ValueError('Invalid matrix cell '+key)
    if params['variant'] not in bank['variants'] or params['timing_mode']!='A' or params['backend']!='internal':
        raise ValueError('Only frozen internal modeA variants')
    return params


def preflight_protocol():
    p=json.loads((ROOT/'experiments/campaign/inputs/v01.json').read_text())
    # Keep task inputs, strip previous B deployment/qualification metadata.
    previous={key:p.pop(key) for key in list(p) if key.startswith('release_') or key in
              {'clock_decision_record','family_hold_policy','reviewed_controller_sha256'}}
    p['source_provenance']={'source_protocol':'experiments/campaign/inputs/v01.json',
        'source_protocol_sha256':sha(ROOT/'experiments/campaign/inputs/v01.json'),
        'previous_B_context':'historical source metadata only; neither A qualification nor A release',
        'previous_B_source_commit':previous.get('release_evidence',{}).get('source_commit')}
    p.update(campaign_id='F01-A-preflight',name='F01-canonical-rough-preflight-A',
        provenance='New held F01 modeA canonical readiness; historical B receipt is source provenance only, not A qualification or release',
        timing_mode='A',offline_timing='controller_dt',
        timing_scope='wrapper wall/native recorded separately; computation ignored by plant',
        startup_convention='cold solve activated at0; fixed30ms updates',
        terminal_convention='rough reaching checked after nominal intervals',
        approved_clock_convention='modeA fixedfeedback',implemented_clock_convention='modeA fixedfeedback',
        release_status='held: own F01-A readiness worker/review/plan required',
        release_machines=[],release_scope='none; preparation only',release_authorization=None,
        clock_decision_record={'scope':'independent fixed-clock F01 readiness; prior B approval does not release A'},
        family_hold_policy='failed canonical A rough/physical check holds F01 timing family on that machine')
    return p


def read_evidence(item,inputs):
    if set(item)!= {'path','sha256'}:raise ValueError('Exact hashed evidence required')
    relative=Path(item['path']);root=Path(inputs).resolve();path=(root/relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root) or sha(path)!=item['sha256']:
        raise ValueError('Evidence path/hash mismatch')
    return json.loads(path.read_text())


def guard(case,bank,release,inputs,identity):
    """Validate source/build/input and canonical same-machine A receipt before import.

    This batch never enables execution even with valid synthetic release evidence.
    """
    p=validate_case(case,bank)
    required={'machine','source_commit','source_sha256','module_sha256_by_N','module_path_by_N'}
    if set(identity)!=required or identity['machine'] not in MACHINES:raise ValueError('Exact identity required')
    if set(release)!= {'source_commit','source_sha256','bank_manifest_sha256','preflight','build'}:
        raise ValueError('Exact release evidence keys required')
    if release['source_commit']!=identity['source_commit'] or not identity['source_sha256'] or release['source_sha256']!=identity['source_sha256']:
        raise ValueError('Source identity differs')
    if set(identity['source_sha256'])!=set(SOURCE_FILES):raise ValueError('Incomplete source hash set')
    # Optional worktree provenance only; normal archived workers have no .git.
    if (ROOT/'.git').exists():
        actual_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
        if identity['source_commit']!=actual_commit:raise ValueError('Actual source commit differs')
    for relative,digest in identity['source_sha256'].items():
        path=(ROOT/relative).resolve()
        if not path.is_relative_to(ROOT) or sha(path)!=digest:raise ValueError('Actual source bytes differ')
    if release['bank_manifest_sha256']!=BANK_MANIFEST_SHA256:raise ValueError('Release bank differs')
    for N in {32,p['N']}:
        key=str(N)
        if key not in identity['module_path_by_N'] or key not in identity['module_sha256_by_N']:
            raise ValueError('Missing preflight/cell module identity')
        if sha(identity['module_path_by_N'][key])!=identity['module_sha256_by_N'][key]:raise ValueError('Actual module bytes differ')
    build=read_evidence(release['build'],inputs)
    if build.get('source_commit')!=identity['source_commit'] or build.get('source_sha256')!=identity['source_sha256'] or build.get('module_sha256_by_N')!=identity['module_sha256_by_N']:
        raise ValueError('Build identity differs')
    bundle=release['preflight']
    if set(bundle)!= {'metrics','provenance'}:raise ValueError('Complete preflight receipt required')
    metrics=read_evidence(bundle['metrics'],inputs);provenance=read_evidence(bundle['provenance'],inputs)
    expected=preflight_protocol()
    config_hash=hashlib.sha256((json.dumps(expected,indent=2,allow_nan=False)+'\n').encode()).hexdigest()
    if (metrics.get('configuration')!=expected or metrics.get('machine')!=identity['machine'] or
        metrics.get('execution_backend')!='internal' or metrics.get('timing_mode')!='A' or
        metrics.get('task_success') is not True or metrics.get('physical_pass') is not True or
        metrics.get('execution_error') is not None or metrics.get('outcome')!='task_passed'):
        raise ValueError('Canonical same-machine A preflight missing/failed')
    if (provenance.get('source_commit')!=identity['source_commit'] or provenance.get('machine')!=identity['machine'] or
        provenance.get('module_sha256')!=identity['module_sha256_by_N']['32'] or
        provenance.get('source_sha256')!=identity['source_sha256'] or provenance.get('task_config_sha256')!=config_hash):
        raise ValueError('Preflight provenance differs')
    if not EXECUTION_RELEASED:raise RuntimeError('F01 execution held: independent harness/telemetry release, selected variant, per-machine preflight and Jetson duration discussion pending')
    return p
