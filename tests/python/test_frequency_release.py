"""Small CPU-only boundary/complete-cell checks; all evidence/native owners fake."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
def load(name):
    spec = importlib.util.spec_from_file_location('release_test_' + name, ROOT / f'experiments/frequency_campaign/{name}.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
W = load('run'); R = W.R; F = load('factory'); E = load('episode'); Ready = load('readiness')


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(data, indent=2) + '\n')
    return dict(path=str(path), sha256=R.sha(path))


def evidence(tmp):
    # Synthetic evidence is only used with injected CPU owners, never production imports.
    source = tmp / 'source'; source.mkdir()
    hashes = {}
    for name in R.RUNTIME:
        out = source / name; out.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / name, out)
        hashes[name] = R.sha(out)
    module = source / 'python/bsqp/bsqpN16_tiago_right.fake'; module.write_text('CPU artifact only')
    module32 = source / 'python/bsqp/bsqpN32_tiago_right.fake'; module32.write_text('CPU N32 artifact only')
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    src = write(tmp / 'source.json', dict(source_file_sha256=hashes, source_commit=commit))
    reason = write(tmp / 'reason.json', dict(telemetry_reason_validation='synthetic_test actual-path metadata'))
    acceptance = write(tmp / 'accept.json', dict(decision='accept_existing_first_instrumentation_without_equivalence_gate',
        numerical_equivalence_required=False, accepted_native_patch_sha256='observer', comparison_audit=reason))
    suite = write(tmp / 'suite.json', dict(status='passed', exit_code=0, fixed_task_processes=3, error=None,
        module_before_sha256='qualified-first-module', module_after_sha256='qualified-first-module', restored_patch_sha256='observer'))
    suite['machine'] = 'gtx1070'
    modules = {}
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    for N, target in [(16, module), (32, module32)]:
        cache = tmp / f'cache{N}.txt'; cache.write_text(f'GATO_EXPERIMENT_TELEMETRY:BOOL=ON\nKNOTS:STRING={N}\nPLANT:STRING=tiago_right\n')
        build = write(tmp / f'build{N}.json', dict(variant='ON', source_commit=commit, source_root=str(source),
            flags=dict(GATO_EXPERIMENT_TELEMETRY=True, N=N, plant='tiago_right'),
            module=dict(path=str(target), sha256=R.sha(target)), cmake_cache=dict(path=str(cache), sha256=R.sha(cache)),
            source_receipt=src, source_file_sha256=hashes))
        modules[str(N)] = dict(path=str(target), sha256=R.sha(target), build_receipt=build)
    m = dict(status='released_F01_A', evidence_kind='synthetic_test', machine='gtx1070', production_core_commit=R.CORE,
        collector_source_commit=commit, collector_source_sha256={name:R.sha(ROOT / name) for name in R.COLLECTOR},
        bank_manifest_sha256=R.P.BANK_MANIFEST_SHA256, selection_sha256=R.sha(R.HERE / 'selected_variant.json'),
        telemetry_acceptance=acceptance, observer_patch_sha256='observer', reason_review=reason,
        build_review=write(tmp / 'review.json', dict(verdict='approved', evidence_kind='synthetic_test')),
        task_suite=suite, modules=modules)
    manifest = tmp / 'manifest.json'; write(manifest, m)
    context = R.validate(manifest, R.sha(manifest), 'gtx1070', 16)
    metrics = write(tmp / 'readiness.json', dict(machine='gtx1070', canonical_config_sha256=R.canonical_hash(),
        module_sha256=R.sha(module32), observer_patch_sha256='observer', timing_mode='A', task_success=True,
        physical_pass=True, execution_error=None, runtime_source_sha256=context['modules']['32']['runtime_source_sha256']))
    family = tmp / 'family.json'; R.write_state(family, context, 'ready', readiness_metrics=metrics)
    return manifest, family


class Native:
    def solve(self, xu, dt, state, reference):
        return dict(XU=xu.astype(np.float32), sqp_time_us=42., sqp_iters=np.array([1]), pcg_iters=np.array([[1]]),
                    experiment_telemetry=dict(pcg_stop_reason_codes=np.array([[2]], np.uint32)))
class Public:
    def __init__(self, fail=False): self.solver = Native(); self.stats = {}; self.fail = fail
    def reset_dual(self): pass
    def reset_rho(self): pass
    def solve(self, state, reference, warm):
        reply = self.solver.solve(warm, .03, state, reference)
        if self.fail: raise RuntimeError('synthetic public normalization failure')
        self.stats = reply; return reply['XU'], reply['sqp_time_us']
    def get_stats(self): return self.stats
class Controller:
    def __init__(self, N, params, fail=False):
        self.N=N; self.dt=.03; self.batch_size=1; self.solver_params=params; self.solver=Public(fail); self.model=None
    def update_force_batch(self,q): pass


def builder(log, fail=False):
    def prepare(context, N):
        log.append(('import_hook', N))
        def make(kind, params):
            log.append((kind, params.copy())); return Controller(N, params, fail)
        return make
    return prepare

def plant(owner): return lambda q,v,u,dt: (q.astype(np.float64), v.astype(np.float64))
def negative(state, goal, u): return dict(rough_reached=False, physical_pass=False, position_error_m=.1)
def cell(): return dict(parameters=dict(variant='short12', N=16, SQP=1, PCG=100, timing_mode='A', backend='internal'))


def test_complete_selected_cell_keeps_scientific_negatives_and_exact_counts(tmp_path):
    manifest, family = evidence(tmp_path); log=[]
    result = W.run_case(cell(), tmp_path/'run', manifest_path=manifest, manifest_sha256=R.sha(manifest), machine='gtx1070',
        family_state=family, builder=builder(log), integrator=plant, observer=negative)
    assert len(R.selected_cases()) == 64
    assert result['outcome']=='timing_cell_completed' and len(result['episodes'])==12
    assert result['evidence_kind']=='synthetic_test'
    assert R.read(tmp_path/'run/provenance.json')['evidence_kind']=='synthetic_test'
    assert all(e['completed_solves']==20 and e['completed_updates']==20 for e in result['episodes'])
    assert R.read(family)['status']=='ready'  # Timing science negatives retained, not execution hold.
    events=[json.loads(l) for l in (tmp_path/'run/events.jsonl').read_text().splitlines()]
    assert sum(e['event']=='native_return' for e in events)==241
    assert sum(e['event']=='step_completed' for e in events)==2400
    assert len(log)==14  # One lazy import, one warmup +12 fresh full owners.


@pytest.mark.parametrize('change', ['held','source','machine','target','negative_suite'])
def test_bad_or_unqualified_release_rejected_before_import(tmp_path, change):
    manifest, family=evidence(tmp_path); m=R.read(manifest); log=[]
    if change=='held':m['status']='held'
    elif change=='source':m['collector_source_sha256']['experiments/frequency_campaign/factory.py']='stale'
    elif change=='machine':m['machine']='rtx5060'
    elif change=='target':
        b=R.read(m['modules']['16']['build_receipt']['path']);b['flags']['N']=32
        m['modules']['16']['build_receipt']=write(Path(m['modules']['16']['build_receipt']['path']),b)
    else:
        s=R.read(m['task_suite']['path']);s['status']='failed'
        m['task_suite']=dict(write(Path(m['task_suite']['path']),s),machine='gtx1070')
    write(manifest,m)
    with pytest.raises(RuntimeError):W.run_case(cell(),tmp_path/'run',manifest_path=manifest,manifest_sha256=R.sha(manifest),
        machine='gtx1070',family_state=family,builder=builder(log),integrator=plant,observer=negative)
    assert log==[] and (tmp_path/'run/metrics.json').exists()


def test_normalization_failure_durable_native_reply_and_cross_cell_hold(tmp_path):
    manifest,family=evidence(tmp_path);log=[]
    with pytest.raises(RuntimeError,match='normalization'):
        W.run_case(cell(),tmp_path/'failed',manifest_path=manifest,manifest_sha256=R.sha(manifest),machine='gtx1070',
            family_state=family,builder=builder(log,True),integrator=plant,observer=negative)
    events=[json.loads(l) for l in (tmp_path/'failed/events.jsonl').read_text().splitlines()]
    returned=next(e for e in events if e['event']=='native_return')
    with np.load(tmp_path/'failed'/returned['raw_file']) as a: assert any('native_reply__XU' in k for k in a.files)
    assert R.read(family)['status']=='held'
    before=len(log)
    with pytest.raises(RuntimeError,match='held'):
        W.run_case(cell(),tmp_path/'next',manifest_path=manifest,manifest_sha256=R.sha(manifest),machine='gtx1070',
            family_state=family,builder=builder(log),integrator=plant,observer=negative)
    assert len(log)==before


def test_own_canonical_A_readiness_negative_holds_before_cell(tmp_path):
    manifest,family=evidence(tmp_path);family.unlink();log=[]
    def task(owner,events,config):
        assert config['offline_timing']=='controller_dt' and config['solver_params']['max_pcg_iters']==120
        assert config['sim_dt_s']==.003 and config['solver_params']['qd_cost']==.1
        return dict(task_success=False,physical_pass=True,execution_error=None)
    result=Ready.run(tmp_path/'preflight',manifest,R.sha(manifest),'gtx1070',family,builder=builder(log),execute_task=task)
    assert result['outcome']=='task_failed' and R.read(family)['status']=='held'
    before=len(log)
    with pytest.raises(RuntimeError,match='held'):W.run_case(cell(),tmp_path/'cell',manifest_path=manifest,manifest_sha256=R.sha(manifest),
        machine='gtx1070',family_state=family,builder=builder(log),integrator=plant,observer=negative)
    assert len(log)==before


def test_native_exception_retains_normalized_arguments_before_owner_close_and_holds(tmp_path):
    manifest,family=evidence(tmp_path);log=[];calls=[]
    class FailingNative:
        def reset_dual(self): pass
        def reset_rho(self): pass
        def solve(self,xu,dt,state,reference):
            calls.append((xu.copy(),state.copy(),reference.copy()))
            raise RuntimeError('synthetic native exception')
    def prepare(context,N):
        log.append(('import_hook',N))
        def make(kind,params):
            controller=Controller(N,params)
            # Use the actual public normalizer, without invoking any native constructor.
            from bsqp.interface import BSQP
            public=BSQP.__new__(BSQP);public.batch_size=1;public.nx=14;public.dt=.03
            public.XU_B=np.zeros((1,N*21-7),np.float32);public.stats={};public.solver=FailingNative()
            controller.solver=public
            return controller
        return make
    with pytest.raises(RuntimeError,match='native exception'):
        W.run_case(cell(),tmp_path/'failed_native',manifest_path=manifest,manifest_sha256=R.sha(manifest),
            machine='gtx1070',family_state=family,builder=prepare,integrator=plant,observer=negative)
    assert len(calls)==1 and calls[0][0].dtype==np.float32
    events=[json.loads(l) for l in (tmp_path/'failed_native/events.jsonl').read_text().splitlines()]
    failure=next(e for e in events if e['event']=='native_call_failed')
    assert failure['record']['native_reply'] is None
    with np.load(tmp_path/'failed_native'/failure['raw_file']) as saved:
        for key,expected in [('XU',calls[0][0]),('state',calls[0][1]),('reference',calls[0][2])]:
            np.testing.assert_array_equal(saved['record__native_argument_arrays_at_failure__'+key],expected)
    assert R.read(family)['status']=='held'
    with pytest.raises(RuntimeError,match='held'):
        W.run_case(cell(),tmp_path/'after_native_failure',manifest_path=manifest,manifest_sha256=R.sha(manifest),
            machine='gtx1070',family_state=family,builder=prepare,integrator=plant,observer=negative)
    assert len(calls)==1 and len(log)==1
