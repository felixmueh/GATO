"""Frozen T01 inputs and evidence gates; CPU only, no controller imports."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from raw import sha

ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT/'experiments/campaign/inputs'
PROTOCOL_FILES = {'T01':'t01.json', 'T01-1ms':'t01_1ms.json', 'V01-1ms':'v01_1ms.json'}


def diagnostic_protocol(canonical):
    """Derive descriptive fixed-feedback A metadata without changing task settings."""
    return dict(canonical,timing_mode='A',offline_timing='controller_dt',name='V01-A-diagnostic',
        campaign_id='V01-A-diagnostic',diagnostic_only='separate cohort, eligible only after B fails on this machine',
        timing_scope='BSQP.solve wrapper wall and native times recorded separately; computation time ignored by the simulated plant clock',
        implemented_clock_convention='new plan activates at the current feedback timestamp and executes one nominal feedback interval; solve wall time is unpaid',
        approved_clock_convention='mode A fixed feedback clock; computation time ignored',
        startup_convention='initial solve retained separately and activated at0; bootstrap wall is unpaid; execute one nominal30ms first interval',
        terminal_convention='execute full nominal feedback intervals, then check goal success and timeout at interval end; nominal overshoot is possible; solve wall time is unpaid',
        clock_decision_record={'scope':'A diagnostic fixed-feedback clock; B old-plan timing decision does not apply'})


def reference_bytes(task):
    reference = np.asarray(task['reference_packed'], dtype='<f8')
    if hashlib.sha256(reference.tobytes()).hexdigest() != task['reference_sha256']:
        raise ValueError('Frozen reference byte hash mismatch')
    return reference


def declared_json(evidence, inputs):
    if not isinstance(evidence, dict) or set(evidence) != {'path', 'sha256'}:
        raise ValueError('Evidence requires declared path and sha256')
    relative = Path(evidence['path'])
    if inputs is None or relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Evidence must be a declared runner input')
    path = Path(inputs)/relative
    if sha(path) != evidence['sha256']:
        raise ValueError('Evidence hash mismatch')
    return json.loads(path.read_text())


def receipt(bundle, inputs, identity, expected):
    if not isinstance(bundle, dict) or set(bundle) != {'metrics', 'provenance'}:
        raise ValueError('Require hashed metrics and provenance, not a readiness flag')
    metrics = declared_json(bundle['metrics'], inputs)
    provenance = declared_json(bundle['provenance'], inputs)
    if (metrics.get('machine') != identity['machine'] or metrics.get('execution_backend') != 'internal'
        or metrics.get('timing_mode') != expected['timing_mode']
        or metrics.get('configuration') != expected or metrics.get('execution_error') is not None
        or metrics.get('outcome') not in {'task_passed', 'task_failed'}
        or any(type(metrics.get(k)) is not bool for k in ('task_success', 'physical_pass'))):
        raise ValueError('Receipt must be completed same-machine exact-protocol scientific evidence')
    for key in ('machine', 'source_commit', 'module_sha256'):
        if not identity.get(key) or provenance.get(key) != identity[key]:
            raise ValueError('Receipt source/module/machine identity mismatch')
    for path, digest in identity.get('source_sha256', {}).items():
        if provenance.get('source_sha256', {}).get(path) != digest:
            raise ValueError('Receipt actual harness source bytes mismatch')
    # Config hash is the actual serialized protocol artifact, not an asserted flag.
    expected_hash = hashlib.sha256((json.dumps(expected, indent=2, allow_nan=False)+'\n').encode()).hexdigest()
    if provenance.get('task_config_sha256') != expected_hash:
        raise ValueError('Receipt configuration byte hash mismatch')
    return metrics


def preflight(bundle, inputs, identity, expected):
    result = receipt(bundle, inputs, identity, expected)
    if not result['task_success'] or not result['physical_pass']:
        raise ValueError('Failed preflight holds this family; no task execution')
    return result


def fallback(bundle, inputs, identity, canonical):
    """Authorize matched all-six 1ms, never setup errors/selective rescues."""
    if not isinstance(bundle, dict) or set(bundle) not in ({'failed_preflight', 'A_diagnostic'}, {'tasks'}):
        raise ValueError('Fallback requires failed3ms preflight+A diagnostic or all six completed3ms tasks')
    v01 = json.loads((INPUTS/'v01.json').read_text())
    if 'failed_preflight' in bundle:
        failed = receipt(bundle['failed_preflight'], inputs, identity, v01)
        if failed['task_success'] and failed['physical_pass']:
            raise ValueError('Passing3ms preflight cannot trigger fallback')
        a = diagnostic_protocol(v01)
        receipt(bundle['A_diagnostic'], inputs, identity, a)
        return
    entries = bundle['tasks']
    if not isinstance(entries, dict) or set(entries) != {task['id'] for task in canonical['tasks']}:
        raise ValueError('Require all six completed3ms tasks; no failed-only rescue')
    negative = False
    for task in canonical['tasks']:
        expected = episode_protocol(canonical, task)
        result = receipt(entries[task['id']], inputs, identity, expected)
        negative |= not result['task_success'] or not result['physical_pass']
    if not negative:
        raise ValueError('All3ms tasks passed; fallback is unauthorized')


def episode_protocol(protocol, task):
    p = {k:v for k,v in protocol.items() if k != 'tasks'}
    p.update(task)
    if task['kind'] == 'reaching':
        p['solver_params'] = protocol['reaching_solver_params']
    else:
        p['solver_params'] = protocol['tracking_solver_params']
    return p


def load_case(case):
    parameters = case.get('parameters', {})
    cohort = parameters.get('protocol')
    if cohort not in PROTOCOL_FILES or parameters.get('offline_timing') != 'solve_time':
        raise ValueError('Only frozen internal B T01 or authorized1ms preflight cases')
    allowed = {'protocol', 'offline_timing', 'preflight', 'fallback'} if cohort != 'V01-1ms' else {'protocol', 'offline_timing', 'fallback'}
    if set(parameters) != allowed:
        raise ValueError('Case evidence keys must match frozen cohort')
    protocol = json.loads((INPUTS/PROTOCOL_FILES[cohort]).read_text())
    if cohort == 'T01' and parameters['fallback'] is not None:
        raise ValueError('Canonical3ms case cannot specify fallback')
    return protocol
