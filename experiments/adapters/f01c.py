"""One frozen F01c timing cell, using the reviewed same-candidate collector.

Native code stays at 502fa29. The original 9f2ab6f collector loop/diagnostics
are reused locally; the runner owns preflight and independent-case scheduling.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recording import RawRecorder, sha, write_json
from adapters import f01ab, f01c_collect as science

COLLECTOR_SOURCE = '9f2ab6f3dfd324e397fd4d6909f70284558c8d76'
NATIVE_SOURCE = '502fa29c8b0d649dcb347f9534bf29c6935fa6dd'
ORIGINAL_SCIENCE_SHA256 = {
    'collect.py': '24c854f9bcbadfd399a492220d57c7a900ce634c11ec71176b5d1a568609437d',
    'diagnostics.py': 'd71b2e7b93687312c8c19d3a94af97189b0d045c2ba5abdb633b3a970a12de2a',
}
BUDGETS = ((1, 100), (5, 100), (10, 800))
PROTOCOL = 'simple-f01c-v1'


def validate_case(case):
    p = case['parameters']
    if (p.get('N') != 32 or type(p.get('B')) is not int or p['B'] not in (1, 2, 4, 8, 16)
            or type(p.get('SQP')) is not int or type(p.get('PCG')) is not int
            or (p['SQP'], p['PCG']) not in BUDGETS):
        raise ValueError('F01c requires the approved N32 batch/budget cell')
    if (p.get('timing_mode') != 'A' or p.get('backend') != 'internal'
            or type(p.get('diagnostics')) is not bool
            or p['diagnostics'] != ((p['B'], p['SQP'], p['PCG']) == (16, 5, 100))):
        raise ValueError('F01c requires internal mode A and the frozen correspondence schedule')
    return p


def load_bank(path):
    """Validate frozen bytes and the layout used here, without regenerating seeds."""
    path = Path(path).resolve()
    manifest = json.loads((path / 'SHA256SUMS.json').read_text())
    required = {'bank.json', 'arrays.npz', 'preparation.json', 'seed_diagnostics.json'}
    if set(manifest) != required:
        raise ValueError('Frozen F01c manifest files differ')
    for relative, digest in manifest.items():
        target = (path / relative).resolve()
        if not target.is_relative_to(path) or sha(target) != digest:
            raise ValueError('Frozen F01c bank bytes/path differ')
    bank = json.loads((path / 'bank.json').read_text())
    if (bank['N'], bank['dt_s'], bank['repeats'], bank['candidates']) != (32, .03, 3, 16):
        raise ValueError('Frozen F01c scientific configuration differs')
    if [tuple(v) for v in bank['approved_budgets']] != list(BUDGETS) or len(bank['inputs']) != 6:
        raise ValueError('Frozen F01c budgets/inputs differ')
    with np.load(path / 'arrays.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    shapes = dict(starts=(2, 14), pools=(2, 16, 665), references=(6, 192),
                  warmup_state=(14,), warmup_pool=(16, 665), warmup_reference=(192,))
    for key, shape in shapes.items():
        if arrays[key].shape != shape or arrays[key].dtype != np.dtype('<f4'):
            raise ValueError('Frozen F01c array shape/dtype differs: ' + key)
    for task in bank['inputs']:
        start = arrays['starts'][task['start_index']]
        if not np.array_equal(arrays['pools'][task['start_index'], :, :14], np.tile(start, (16, 1))):
            raise ValueError('Frozen candidate initial states differ')
    return bank, arrays


class Events:
    """Keep the original native-return capture and separate recording clock."""
    def __init__(self, F, recorder):
        self.recording_s = 0.
        def timed(event, record):
            start = time.perf_counter()
            try:
                recorder(event, record)
            finally:
                self.recording_s += time.perf_counter() - start
        self.events = F.Events(timed)

    @property
    def native_calls(self):
        return self.events.native_calls

    def bind(self, capture):
        self.events.bind(capture)

    def __call__(self, event, record):
        self.events(event, record)


def production_factory(source, module, bank, events, F):
    f01ab.preload(source, module, 32)
    interface = importlib.import_module('bsqp.interface')
    native = sys.modules['bsqp.bsqpN32_tiago_right']
    if Path(interface.__file__).resolve() != source / 'python/bsqp/interface.py':
        raise RuntimeError('Actual public wrapper import differs')

    def create(B, SQP, PCG):
        params = dict(bank['preset'], max_sqp_iters=SQP, max_pcg_iters=PCG)
        solver = interface.BSQP(str(source / 'gato/dynamics/tiago_right/tiago_right_arm.urdf'),
                                B, 32, .03, plant_type='tiago_right', **params)
        if solver.lib is not native:
            raise RuntimeError('Resolved native module differs')
        return F.Owned(SimpleNamespace(solver=solver, N=32, dt=.03,
                                      batch_size=B, solver_params=params), events)
    return create


def run_case(case, output, source, module, inputs, machine=None, *, builder=None, assessor=None):
    source, module, output, inputs = Path(source).resolve(), Path(module).resolve(), Path(output), Path(inputs)
    result = dict(campaign_id='F01c', research_question='RQ-timing', timing_mode='A',
                  offline_timing='controller_dt', execution_backend='internal', machine=machine,
                  native_source_commit=NATIVE_SOURCE, original_collector_source_commit=COLLECTOR_SOURCE,
                  collector_source_commit=COLLECTOR_SOURCE, cohort_id=PROTOCOL,
                  evidence_kind='synthetic_test' if builder is not None else 'measurement',
                  task_success=None, physical_pass=None, solver_numerical_pass=None,
                  execution_error=None, native_calls=0)
    events, handlers = None, {}

    def interrupt(signum, frame):
        raise InterruptedError(f'F01c interrupted by signal {signum}')

    with RawRecorder(output) as recorder:
        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                handlers[sig] = signal.signal(sig, interrupt)
            cell = validate_case(case)
            bank, arrays = load_bank(inputs)
            bank_pin = sha(inputs / 'SHA256SUMS.json')
            result['bank_manifest_sha256'] = bank_pin
            scheduled = science.jobs(bank, cell['B'], cell['SQP'], cell['PCG'], cell['diagnostics'])
            result.update(parameters=cell, scheduled_calls=len(scheduled), attempted_calls=0,
                          missing_calls=scheduled, calls=[], collections=[])
            write_json(output / 'protocol.json', dict(parameters=cell, preset=bank['preset'],
                candidate_pool_size=16, input_count=6, fixed_repeats=3, timing_mode='A',
                native_source_commit=NATIVE_SOURCE, collector_source_commit=COLLECTOR_SOURCE,
                bank_manifest_sha256=bank_pin, cohort_id=PROTOCOL,
                offline_timing='controller_dt', execution_backend='internal',
                comparison='same fixed 16 candidates; fresh owner/reset per call; B1 is serial baseline',
                scope='Static solve timing; task success and complete nonlinear qualification unavailable'))
            helpers = ('f01c.py', 'f01c_collect.py', 'f01c_diagnostics.py')
            write_json(output / 'provenance.json', dict(native_source_commit=NATIVE_SOURCE,
                original_collector_source_commit=COLLECTOR_SOURCE, source_root=str(source),
                collector_source_commit=COLLECTOR_SOURCE, bank_manifest_sha256=bank_pin, cohort_id=PROTOCOL,
                original_scientific_source_sha256=ORIGINAL_SCIENCE_SHA256,
                module_path=str(module), module_sha256=sha(module),
                input_sha256={name: sha(inputs / name) for name in ('bank.json', 'arrays.npz', 'SHA256SUMS.json')},
                adapter_sha256={name: sha(Path(__file__).with_name(name)) for name in helpers},
                scientific_source_sha256={name: sha(source / name) for name in (
                    'experiments/frequency_campaign/episode.py', 'experiments/frequency_campaign/factory.py')},
                evidence_kind=result['evidence_kind'], persistent_state='Native rho/drho/duals unavailable; fresh owner/reset history retained'))
            F = f01ab.load(source, 'experiments/frequency_campaign/factory.py', 'f01c_native_capture')
            science.E = f01ab.load(source, 'experiments/frequency_campaign/episode.py', 'f01c_episode')
            events = Events(F, recorder)
            factory = (builder or production_factory)(source, module, bank, events, F)
            if assessor is None and builder is None:
                from adapters.f01c_diagnostics import assess
                import pinocchio as pin
                model = pin.buildModelFromUrdf(str(source / 'gato/dynamics/tiago_right/tiago_right_arm.urdf'))
                assessor = lambda owner, z, goal: assess(model, z, goal, owner.solver_params)
            result.update(science.collect(factory, bank, arrays, cell, events, assessor,
                                        evidence_kind=result['evidence_kind']))
        except BaseException:
            result.update(execution_error=traceback.format_exc(), outcome='execution_failed')
            recorder('execution_failed', {'error': result['execution_error']})
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            if events is not None:
                result['native_calls'] = events.native_calls
            write_json(output / 'metrics.json', result)
            recorder.manifest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=os.environ.get('GATO_SOURCE_DIR'))
    parser.add_argument('--build', type=Path, default=os.environ.get('GATO_BUILD_DIR'))
    parser.add_argument('--module', type=Path)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--machine', default=os.environ.get('GATO_MACHINE'))
    args = parser.parse_args()
    if not args.source or not (args.build or args.module):
        parser.error('--source and --build (or --module) are required')
    case = json.loads(args.case.read_text())
    module = f01ab.resolve_module(args.build, 32, args.module)
    result = run_case(case, args.output, args.source, module, args.inputs, args.machine)
    return 1 if result['execution_error'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
