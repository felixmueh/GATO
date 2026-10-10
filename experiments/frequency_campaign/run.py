"""Single-cell or canonical-A readiness entry. Defaults held without reviewed receipts."""
import argparse
import importlib.util
import json
from pathlib import Path
import os
import shutil

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('f01_release', HERE / 'release.py')
R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
protocol = R.P


def run_case(case, output, release=None, inputs=None, identity=None, *, manifest_path=None,
             manifest_sha256=None, machine=None, family_state=None, builder=None, integrator=None, observer=None):
    # Retain old held-call signature used by the previously reviewed CPU tests.
    output = Path(output)
    if output.exists() and any(output.iterdir()): raise ValueError('Fresh worker output required')
    cell = case.get('parameters', {})
    recorder = R.load('record_bank').Recorder(output, dict(cell, batch_size=1, offline_timing='controller_dt',
        source_commit=None, cohort_id='F01ab-short12', evidence_kind='synthetic_test' if any((builder, integrator, observer)) else 'measurement'))
    context = None; events = None
    result = dict(campaign_id='F01ab', timing_mode='A', execution_backend='internal', machine=machine,
                  outcome='execution_held', task_success=None, physical_pass=None, solver_numerical_pass=None, execution_error=None, episodes=[], native_calls=0, evidence_kind='synthetic_test' if any((builder, integrator, observer)) else 'measurement')
    try:
        bank, arrays = protocol.load_bank(); protocol.validate_case(case, bank)
        if cell not in R.selected_cases(): raise ValueError('Exact selected short12 matrix cell required')
        if not all((manifest_path, manifest_sha256, machine, family_state)):
            raise RuntimeError('F01 execution held; pinned acceptance/build/suite/A-readiness release absent')
        context = R.validate(manifest_path, manifest_sha256, machine, cell['N'])
        R.readiness_guard(context, family_state)
        source_commit = context['collector_source_commit']
        raw = R.load('record_bank').raw
        raw.write_json(output / 'protocol.json', dict(cell, batch_size=1, offline_timing='controller_dt', source_commit=source_commit, cohort_id='F01ab-short12', evidence_kind=result['evidence_kind']))
        raw.write_json(output / 'provenance.json', dict(machine=machine, source_commit=source_commit, cohort_id='F01ab-short12',
            bank_sha256=R.sha(protocol.BANK / 'bank.json'), release_manifest_sha256=manifest_sha256,
            module=context['modules'][str(cell['N'])], observer_acceptance=context['telemetry_acceptance'],
            numerical_equivalence='not required; earlier comparison variation retained descriptively',
            evidence_kind=result['evidence_kind'], persistent_state='rho/drho/duals unavailable; full-episode constructor/reset history retained'))
        F = R.load('factory'); E = R.load('episode'); events = F.Events(recorder)
        factory = F.Factory(context, bank, events, builder)
        model_observer = observer
        if model_observer is None:
            # Factory constructs each fresh owner; lazy observer follows that owner model.
            model_observer = lambda state, goal, u: E.observe(factory.events.capture_model, state, goal, u)
        def plant(owner):
            events.capture_model = owner.model
            return (integrator or F.integrator)(owner)
        result.update(E.collect_cell(factory, bank, arrays, cell, events, plant, model_observer))
        result.update(outcome='execution_failed' if result['execution_error'] else 'timing_cell_completed',
                      source_commit=source_commit, cohort_id='F01ab-short12')
    except BaseException as error:
        result.update(execution_error=str(error), outcome='execution_held' if context is None else 'execution_failed')
    if events is not None: result['native_calls'] = events.native_calls
    recorder.finish(result)
    if context is not None and result['execution_error'] is not None:
        R.write_state(family_state, context, 'held', reason=result['execution_error'], failed_attempt=str(output))
    if result['execution_error'] is not None: raise RuntimeError(result['execution_error'])
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case', type=Path); p.add_argument('--output', type=Path)
    p.add_argument('--manifest', type=Path, default=HERE / 'release_manifest.json')
    p.add_argument('--manifest-sha256'); p.add_argument('--machine', default=os.environ.get('GATO_MACHINE'))
    p.add_argument('--family-state', type=Path); p.add_argument('--readiness', action='store_true')
    p.add_argument('--inputs', type=Path, help='Normal runner frozen input root; relative manifest paths resolve here')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if a.inputs and not a.manifest.is_absolute(): a.manifest = a.inputs / a.manifest
    if a.dry_run:
        print(json.dumps(dict(status='planning_only', cells=R.selected_cases(), episodes_per_cell=12,
              calls_per_cell=241, calls_per_machine=15424, release=R.read(a.manifest)), indent=2)); return 0
    if not a.output: p.error('--output is required')
    if not all((a.manifest_sha256, a.machine, a.family_state)): p.error('Pinned manifest hash/machine/family-state are required')
    # Generic runner attempt already contains its case, receipt and logs.
    payload = a.output / 'f01-raw'
    a.output.mkdir(parents=True, exist_ok=True)
    try:
        if a.readiness:
            result = R.load('readiness').run(payload, a.manifest, a.manifest_sha256, a.machine, a.family_state)
            return 0 if result['execution_error'] is None else 1
        if not a.case: p.error('--case is required for one-cell worker')
        run_case(R.read(a.case), payload, manifest_path=a.manifest, manifest_sha256=a.manifest_sha256,
                 machine=a.machine, family_state=a.family_state)
        return 0
    finally:
        if (payload / 'metrics.json').exists(): shutil.copyfile(payload / 'metrics.json', a.output / 'metrics.json')


if __name__ == '__main__': raise SystemExit(main())
