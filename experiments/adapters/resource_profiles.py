"""F04 resources and F05 response/profile/sustained cells, mode A/internal."""
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recording import RawRecorder, sha, write_json
from adapters import f01ab
from adapters.task_recording import project_task_record, NATIVE_DIAGNOSTIC_REPRESENTATION
from adapters import resource_profiles_common as C, resource_profiles_science as science

SOURCE = '4e5301feaea5f2d06530922ba46aa543e1995d47'
PROTOCOL = 'simple-resource-profiles-v1'
BANK_SHA256 = 'f8a8217982f3fa2b1b4fd2629e8cb45949d372f6d43c6137bbdaa7e52acc7007'


def project_performance_record(kind, record):
    """Keep timings/work/quality; the frozen input bank is retained once."""
    compact = project_task_record(kind, record)
    # All inputs here are static-bank requests, unlike ordinary measured task states.
    if kind.startswith(('owner_', 'solve')):
        compact = {key: value for key, value in compact.items() if key != 'state'}
    return compact


def validate_case(case):
    parameters = dict(case['parameters'])
    if parameters.pop('timing_mode', None) != 'A' or parameters.pop('backend', None) != 'internal':
        raise ValueError('F04/F05 require internal mode A')
    if parameters not in C.cases():
        raise ValueError('F04/F05 require the exact approved 38-worker protocol')
    return parameters


def preload(context, N):
    info = context['selected_module']
    source, path = Path(info['source_root']), Path(info['path'])
    if any(name.startswith('bsqp.bsqpN') for name in sys.modules):
        raise RuntimeError('Fresh one-N/stage-flag worker required')
    f01ab.preload(source, path, N)
    native = sys.modules[f'bsqp.bsqpN{N}_tiago_right']
    if dict(native.experiment_build_flags) != info['flags']:
        raise RuntimeError('Imported resource/profile compiled flags differ')
    return native, source


class Factory:
    def __init__(self, context, bank, events):
        self.context, self.bank, self.events = context, bank, events
        self.interface = None

    def __call__(self, B, SQP, PCG):
        if self.interface is None:
            started = time.perf_counter()
            self.native, self.source = preload(self.context, 32)
            self.interface = importlib.import_module('bsqp.interface')
            if Path(self.interface.__file__).resolve() != self.source / 'python/bsqp/interface.py':
                raise RuntimeError('Actual public wrapper differs')
            self.events('native_wrapper_imported', dict(import_wall_s=time.perf_counter()-started,
                scope='native/public-wrapper import before first constructor'))
        params = dict(self.bank['preset'], max_sqp_iters=SQP, max_pcg_iters=PCG)
        started = time.perf_counter()
        solver = self.interface.BSQP(str(self.source/'gato/dynamics/tiago_right/tiago_right_arm.urdf'),
                                    B, 32, .03, plant_type='tiago_right', **params)
        self.events('native_owner_constructed', dict(constructor_wall_s=time.perf_counter()-started,
                    B=B, N=32, effective_solver_params=params))
        if solver.lib is not self.native:
            raise RuntimeError('Resolved native differs')
        return C.F.Owned(SimpleNamespace(solver=solver, N=32, dt=.03,
                                       batch_size=B, solver_params=params), self.events)


def query(context, case):
    native, _ = preload(context, case['N'])
    return native.experiment_resources(case['B'])


def run_case(case, output, source, module, inputs, machine=None, *, builder=None, resource_query=None, clock=time.perf_counter):
    source, module, inputs, output = Path(source).resolve(), Path(module).resolve(), Path(inputs), Path(output)
    synthetic = builder is not None or resource_query is not None or clock is not time.perf_counter
    result = dict(evidence_kind='synthetic_test' if synthetic else 'measurement',
        source_commit=SOURCE, machine=machine, timing_mode='A', execution_backend='internal',
        task_success=None, physical_pass=None, solver_numerical_pass=None, execution_error=None,
        native_calls=0, completed_calls=0, measured_calls=0, status='started')
    handlers, events = {}, None

    def interrupt(signum, frame):
        raise InterruptedError(f'F04/F05 interrupted by signal {signum}')

    with RawRecorder(output) as recorder:
        try:
            for sig in (signal.SIGTERM, signal.SIGINT):
                handlers[sig] = signal.signal(sig, interrupt)
            parameters = validate_case(case)
            def emit(kind, record):
                return recorder(kind, project_performance_record(kind, record)
                                if parameters['study'] == 'F05' else record)
            write_json(output/'recording.json', dict(
                recording_format='performance-observations-v1' if parameters['study']=='F05' else 'resource-query-v1',
                projection_sha256=sha(Path(__file__).with_name('task_recording.py')),
                native_diagnostic_representation=NATIVE_DIAGNOSTIC_REPRESENTATION,
                retained='Every timing, stage profile, native work/quality, warmup, failure and input/candidate identity',
                omitted='F05 repeated frozen-bank inputs and solver matrices; array quality summaries retained',
                input_bank='Unchanged frozen bank retained once in runner family _inputs, identified by provenance hashes',
                timing_scope='Recording remains in sustained wall loop; changed overhead can change call count. No timing equivalence claim.'))
            flags = dict(GATO_EXPERIMENT_STAGE_TIMING=parameters['stage_timing'],
                         GATO_EXPERIMENT_RESOURCE_QUERY=True, GATO_EXPERIMENT_TELEMETRY=True)
            info = dict(path=str(module), source_root=str(source), sha256=sha(module), flags=flags)
            context = dict(source_commit=SOURCE, selected_module=info, bank=dict(path=str(inputs),sha256=BANK_SHA256))
            result.update(campaign_id=parameters['study'], case=parameters, cohort_id=PROTOCOL)
            write_json(output/'protocol.json',dict(parameters, timing_mode='A', offline_timing='controller_dt',
                backend='internal', execution_backend='internal', cohort_id=PROTOCOL,
                evidence_kind=result['evidence_kind'], pcg_times_us_is_placeholder=True,
                pcg_times_us_unavailable_reason='Native literal-zero placeholder; not measured PCG cost',
                native_timing_scope='Native host chrono includes kernels/copies/waits/CPU predicate; binding I/O excluded',
                scope='Stage ON profiles separate from OFF response/sustained; no thermal steady-state/saturation claim'))
            write_json(output/'provenance.json',dict(machine=machine, source_commit=SOURCE, module=info,
                bank_manifest_sha256=BANK_SHA256, cohort_id=PROTOCOL, evidence_kind=result['evidence_kind'],
                adapter_sha256={name:sha(Path(__file__).with_name(name)) for name in
                    ('resource_profiles.py','resource_profiles_common.py','resource_profiles_science.py')},
                scientific_source_sha256={name:sha(source/name) for name in
                    ('experiments/frequency_campaign/factory.py','experiments/frequency_campaign/episode.py')},
                persistent_state='fresh owner eachcall; native dual/rho/drho unavailable; resets retained'))
            C.F = f01ab.load(source,'experiments/frequency_campaign/factory.py','profiles_capture')
            C.E = f01ab.load(source,'experiments/frequency_campaign/episode.py','profiles_episode')
            events = C.F.Events(emit)
            result.update(science.execute(parameters, context, inputs, events, emit,
                                          builder or Factory, resource_query or query, clock))
        except BaseException:
            result.update(status='execution_failed',execution_error=dict(type='AdapterFailure',error=traceback.format_exc()))
            recorder('execution_failed',dict(error=result['execution_error']['error']))
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig,handler)
            result['outcome']='execution_failed' if result['execution_error'] else 'timing_completed'
            if events is not None:result['native_calls']=events.native_calls
            write_json(output/'metrics.json',result)
            recorder.manifest()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source',type=Path,default=os.environ.get('GATO_SOURCE_DIR'))
    parser.add_argument('--build',type=Path,default=os.environ.get('GATO_BUILD_DIR'))
    parser.add_argument('--module',type=Path)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--machine',default=os.environ.get('GATO_MACHINE'))
    args=parser.parse_args()
    if not args.source or not (args.build or args.module):
        parser.error('--source and --build (or --module) required')
    case=json.loads(args.case.read_text())
    parameters=validate_case(case)
    module=f01ab.resolve_module(args.build,parameters['N'],args.module)
    result=run_case(case,args.output,args.source,module,args.inputs,args.machine)
    return 1 if result['execution_error'] else 0


if __name__=='__main__':raise SystemExit(main())
