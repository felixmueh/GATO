"""One frozen F01ab cell: original episode science, compact raw recording.

The runner owns the canonical A preflight and independent-cell scheduling.
No release graph, family-state file, input regeneration, or package comparison.
"""
import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import traceback

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recording import RawRecorder, sha, write_json


def load(source, relative, name):
    spec = importlib.util.spec_from_file_location(name, Path(source) / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_bank(path, cell):
    """Read retained input bytes; only check dimensions/configuration used here."""
    path = Path(path)
    bank = json.loads((path / 'bank.json').read_text())
    with np.load(path / 'arrays.npz', allow_pickle=False) as saved:
        arrays = {k: saved[k].copy() for k in saved.files}
    for key in ('N', 'SQP', 'PCG'):
        if type(cell.get(key)) is not int or cell[key] not in bank[key]:
            raise ValueError('Invalid frozen cell ' + key)
    if cell.get('variant') not in bank['variants'] or cell.get('timing_mode') != 'A' or cell.get('backend') != 'internal':
        raise ValueError('F01ab requires frozen internal mode A cell')
    if (bank['batch_size'], bank['dt_s'], bank['sim_dt_s'], bank['offline_timing']) != (1, .03, .003, 'controller_dt'):
        raise ValueError('Bank timing/batch configuration differs from episode implementation')
    for key, width in [('starts', 14), ('goals', 6)]:
        if arrays[key].ndim != 2 or arrays[key].shape[1] != width:
            raise ValueError('Invalid bank shape: ' + key)
    if arrays['warmup_state'].shape != (14,) or arrays['warmup_goal'].shape != (6,):
        raise ValueError('Invalid warmup input shapes')
    for episode in bank['variants'][cell['variant']]['episodes']:
        if episode['updates'] != 20 or len(episode['categories']) != 20:
            raise ValueError('Episode must retain twenty fixed updates/categories')
        arrays['starts'][episode['start_index']]
        if episode['kind'] == 'reaching':
            if episode['goal_switch_before_update'] != 10 or len(episode['goal_indices']) != 2:
                raise ValueError('Frozen reaching goal-switch schedule differs')
            arrays['goals'][episode['goal_indices']]
        elif episode['kind'] == 'tracking':
            reference = arrays[f"reference_{episode['reference_index']}"]
            if reference.ndim != 1 or reference.size < 6 or reference.size % 6:
                raise ValueError('Invalid tracking reference shape')
        else:
            raise ValueError('Unknown episode kind')
    return bank, arrays


def resolve_module(build, N, explicit=None):
    if explicit:
        module = Path(explicit).resolve()
        if not module.is_file():
            raise FileNotFoundError(module)
        return module
    matches = list((Path(build) / 'modules').glob(f'bsqpN{N}_tiago_right*.so'))
    if len(matches) != 1:
        raise ValueError(f'Expected one N{N} module under {build}/modules, found {len(matches)}')
    return matches[0].resolve()


def preload(source, module, N):
    """Load the exact build extension while keeping source files untouched."""
    sys.path[:0] = [str(source / 'python'), str(source / 'tiago_src')]
    importlib.import_module('bsqp')
    name = f'bsqp.bsqpN{N}_tiago_right'
    spec = importlib.util.spec_from_file_location(name, module)
    native = importlib.util.module_from_spec(spec)
    sys.modules[name] = native
    spec.loader.exec_module(native)


def collect(E, factory, bank, arrays, cell, emit, integrate, observer, cancelled):
    """Reuse exact warmup/episode primitives; fresh episodes survive other errors."""
    episodes = bank['variants'][cell['variant']]['episodes']
    reports = []

    def make_owner(kind):
        owner = factory(kind, cell)
        expected = dict(bank['presets'][kind], max_sqp_iters=cell['SQP'], max_pcg_iters=cell['PCG'])
        if owner.N != cell['N'] or owner.dt != .03 or owner.batch_size != 1 or owner.solver_params != expected:
            owner.close()
            raise ValueError('Effective owner/preset differs from frozen cell')
        return owner

    owner = None
    try:
        owner = make_owner('reaching')
        state = arrays['warmup_state'].copy()
        owner.solver.reset_dual(); owner.solver.reset_rho(); owner.update_force_batch(state[:7])
        E.timed_solve(owner, state[None, :], np.tile(arrays['warmup_goal'], cell['N'])[None, :],
                      E.cold_seed(state, cell['N']),
                      dict(category='process_cuda_warmup', excluded=True, episode_id=None,
                           N=cell['N'], caps=dict(SQP=cell['SQP'], PCG=cell['PCG'])), emit)
    finally:
        if owner is not None:
            owner.close()
    for episode in episodes:
        if cancelled():
            break
        owner = None
        try:
            owner = make_owner(episode['kind'])
            observe = observer or (lambda state, goal, u: E.observe(owner.model, state, goal, u))
            report = E.collect_episode(owner, episode, arrays, cell, emit, integrate(owner), observe)
        except BaseException as error:
            report = dict(episode_id=episode['id'], execution_error=str(error),
                          missing_solve_slots=list(range(20)), missing_update_slots=list(range(20)))
            emit('episode_failed', report)
        finally:
            if owner is not None:
                owner.close()
        reports.append(report)
    result = dict(episodes=reports, missing_episodes=[e['id'] for e in episodes[len(reports):]],
                  execution_error=next((r['execution_error'] for r in reports if r['execution_error'] is not None),
                                       'Interrupted' if cancelled() else None))
    emit('cell_finished', result)
    return result


def run_case(case, output, source, module, inputs=None, machine=None, *, builder=None, integrator=None, observer=None):
    source, output, module = Path(source).resolve(), Path(output), Path(module).resolve()
    cell = case['parameters']
    bank_path = Path(inputs) if inputs else source / 'experiments/frequency_campaign/inputs'
    result = dict(campaign_id='F01ab', research_question='RQ-timing', timing_mode='A',
                  offline_timing='controller_dt', execution_backend='internal', machine=machine,
                  evidence_kind='synthetic_test' if any((builder, integrator, observer)) else 'measurement',
                  task_success=None, physical_pass=None, solver_numerical_pass=None,
                  execution_error=None, episodes=[], native_calls=0)
    interrupted, handlers, events = [], {}, None

    def interrupt(signum, frame):
        interrupted.append(signum)
        raise InterruptedError(f'F01ab interrupted by signal {signum}')

    with RawRecorder(output) as recorder:
        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                handlers[sig] = signal.signal(sig, interrupt)
            bank, arrays = load_bank(bank_path, cell)
            write_json(output / 'protocol.json', dict(cell, batch_size=1, offline_timing='controller_dt',
                       bank_variant=bank['variants'][cell['variant']], presets=bank['presets'],
                       initial_guess_policy=bank['initial_guess_policy'], warm_start_shift_s=bank['warm_start_shift_s']))
            write_json(output / 'provenance.json', dict(source_root=str(source), module_path=str(module),
                       module_sha256=sha(module), bank_path=str(bank_path),
                       input_sha256={name: sha(bank_path / name) for name in ('bank.json', 'arrays.npz')},
                       scientific_source_sha256={name: sha(source / name) for name in (
                           'experiments/frequency_campaign/episode.py', 'experiments/frequency_campaign/factory.py',
                           'python/bsqp/common.py')}, evidence_kind=result['evidence_kind'],
                       persistent_state='rho/drho/duals unavailable; episode constructor/reset history retained'))
            F = load(source, 'experiments/frequency_campaign/factory.py', 'f01_factory')
            E = load(source, 'experiments/frequency_campaign/episode.py', 'f01_episode')
            if builder is None:
                preload(source, module, cell['N'])
            events = F.Events(recorder)
            factory = F.Factory({'modules': {str(cell['N']): {'path': str(module), 'source_root': str(source)}}},
                                bank, events, builder)
            result.update(collect(E, factory, bank, arrays, cell, events, integrator or F.integrator,
                                  observer, lambda: bool(interrupted)))
        except BaseException:
            result['execution_error'] = traceback.format_exc()
            recorder('execution_failed', {'error': result['execution_error']})
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            if events is not None:
                result['native_calls'] = events.native_calls
            result['outcome'] = 'execution_failed' if result['execution_error'] else 'timing_cell_completed'
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
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--machine', default=os.environ.get('GATO_MACHINE'))
    args = parser.parse_args()
    if not args.source or not (args.build or args.module):
        parser.error('--source and --build (or --module) are required')
    case = json.loads(args.case.read_text())
    module = resolve_module(args.build, case['parameters']['N'], args.module)
    result = run_case(case, args.output, args.source, module, args.inputs, args.machine)
    return 1 if result['execution_error'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
