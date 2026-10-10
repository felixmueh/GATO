"""CPU qualification of frozen Q02 setup/oracles; no native pass implied."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
SPEC=importlib.util.spec_from_file_location('q02_protocol',ROOT/'experiments/campaign/q02/protocol.py')
p=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(p)

@pytest.fixture(scope='module')
def frozen(): return p.read_protocol()

@pytest.fixture(scope='module')
def pin_reference(): return p.PinReference()

def test_frozen_budget_and_scope(frozen):
    assert len(frozen['samples'])==8
    assert frozen['knots']==32 and frozen['knot_dt_s']==.03
    assert frozen['timing_mode']=='A'
    assert frozen['dtype']=='float32'
    assert frozen['native_evaluate_calls']==1+8+len(frozen['batch_groups'])+8*21*2
    assert frozen['native_fd_step']==.004 and frozen['reference_fd_step']==1e-5
    assert frozen['samples'][1]['xu'][5]<0<frozen['samples'][2]['xu'][5]
    assert all(len(group)==4 for group in frozen['batch_groups'])
    assert any(len(set(g))<4 for g in frozen['batch_groups'])
    assert sorted(frozen['batch_groups'][0])==sorted(frozen['batch_groups'][2])
    assert frozen['weights']==p.WEIGHTS

@pytest.mark.parametrize('index',range(8))
def test_interior_inputs_all_fd_offsets_and_reference_step(frozen,pin_reference,index):
    xu=np.array(frozen['samples'][index]['xu']); h=frozen['native_fd_step']
    for offset,key in [(0,'q'),(7,'v'),(14,'u')]:
        lim=np.array(frozen['limits'][key]); z=xu[offset:offset+7]
        assert np.all(z-h>lim[:,0]) and np.all(z+h<lim[:,1])
    # Every planned finite difference remains inside the cost domain. Inspect
    # independent one-step q/v too, without using any native implementation.
    for j in range(21):
        for sign in [-1,1]:
            z=xu.copy(); z[j]+=sign*h
            y=p.one_step(z,pin_reference.acceleration(z))
            for offset,key in [(0,'q'),(7,'v')]:
                lim=np.array(frozen['limits'][key]); state=y[offset:offset+7]
                assert np.all(state>lim[:,0]) and np.all(state<lim[:,1])
    assert abs(np.cos(pin_reference.pose(xu[:7])[4]))>.1

@pytest.mark.parametrize('index',range(8))
def test_independent_reference_arrays_finite_and_gn_psd(frozen,pin_reference,index):
    s=frozen['samples'][index]; xu=np.array(s['xu']); goal=np.array(s['goal'])
    ref=pin_reference.references(xu,goal,frozen['limits'])
    assert all(np.isfinite(x).all() for x in ref.values())
    assert ref['dynamics_jac'].shape==(7,21) and ref['pose_jac'].shape==(6,7)
    for key in ['Q','R','terminal_Q']:
        assert np.allclose(ref[key],ref[key].T,atol=1e-12)
        assert np.linalg.eigvalsh(ref[key]).min()>0
    # Terminal ignores u; R remains a running-only block.
    altered=xu.copy(); altered[14:]+=.1
    assert p.cost_reference(altered,ref['pose'],goal,frozen['limits'],True)==p.cost_reference(xu,ref['pose'],goal,frozen['limits'],True)


def test_finite_difference_oracle_known_function():
    z=np.array([.4,-.7])
    jac=p.central_jacobian(lambda x:np.array([x[0]**3+x[1],np.sin(x[0])*x[1]]),z,1e-5)
    expected=np.array([[3*z[0]**2,1],[np.cos(z[0])*z[1],np.sin(z[0])]])
    np.testing.assert_allclose(jac,expected,atol=2e-10,rtol=1e-9)


def test_gauss_newton_reference_is_not_exact_nonlinear_hessian(frozen):
    xu=np.array(frozen['samples'][0]['xu']); jac=np.zeros((6,7)); jac[0,0]=2*xu[0]
    Q,_=p.gn_reference(xu,jac,frozen['limits'])
    regularizer=p.WEIGHTS['q_barrier']*p.barrier_hessian(xu[:7],frozen['limits']['q'])[0]
    assert Q[0,0]==pytest.approx(p.WEIGHTS['position']*(2*xu[0])**2+regularizer)
    # For residual q0^2, the exact nonlinear cost curvature adds 2*w*q0^2.
    assert Q[0,0]!=pytest.approx(p.WEIGHTS['position']*6*xu[0]**2+regularizer)


def test_native_decode_preserves_column_major_contract():
    flat=np.arange(995,dtype=float)[None,:]
    decoded=p.decode(flat)
    assert decoded['dynamics_jac'][0,2,3]==7+3*7+2
    assert decoded['A'][0,3,9]==168+9*14+3
    assert decoded['pose_jac'][0,2,5]==468+5*6+2
    assert decoded['terminal_Q'][0,3,9]==792+9*14+3
    with pytest.raises(ValueError): p.decode(np.zeros((1,988)))


def test_comparison_rejects_nonfinite_and_shape_mismatch():
    assert not p.compare(np.array([np.nan]),np.array([1.]),1e-3,1e-3)['pass']
    assert not p.compare(np.array([1.]),np.array([np.inf]),1e-3,1e-3)['pass']
    assert p.compare(np.zeros(3),np.full(3,1e-6),2e-6,0)['pass']
    assert not p.compare(np.zeros(3),np.full(3,1e-6),1e-7,0)['pass']
    with pytest.raises(ValueError): p.compare(np.zeros(3),np.zeros((1,3)),1,1)


def test_pose_comparison_handles_rpy_branch_without_erasing_translation():
    a=np.array([0.,0.,0.,np.pi-1e-4,0.,0.])
    b=np.array([0.,0.,0.,-np.pi+1e-4,0.,0.])
    assert p.compare(a,b,3e-4,0,True)['pass']
    a[0]=.01
    assert not p.compare(a,b,3e-4,0,True)['pass']


def test_cost_reference_refuses_outside_domain(frozen):
    z=np.array([lo-.1 for lo,hi in frozen['limits']['q']])
    with pytest.raises(ValueError,match='strict interior'):
        p.barrier(z,frozen['limits']['q'])

@pytest.fixture
def q02_runner(monkeypatch):
    import sys
    monkeypatch.syspath_prepend(str(ROOT/'experiments/campaign/q02'))
    spec=importlib.util.spec_from_file_location('q02_runner',ROOT/'experiments/campaign/q02/run.py')
    runner=importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    return runner


def fake_main_inputs(monkeypatch,runner,frozen):
    inputs=np.array([s['xu'] for s in frozen['samples']],dtype=np.float32)
    goals=np.array([s['goal'] for s in frozen['samples']],dtype=np.float32)
    zero=p.decode(np.zeros((1,995),dtype=np.float32))
    references=[{key:value[0].copy() for key,value in zero.items()} for _ in range(8)]
    def prepare(output):
        # Only native recording is exercised by this fake CPU-only test. The
        # independent mathematical CPU oracles have their separate tests above.
        runner.json_write(output/'protocol.json',frozen)
        np.savez(output/'inputs.npz',xu=inputs,goal=goals)
        return frozen,inputs,goals,references,True
    monkeypatch.setattr(runner,'prepare',prepare)
    return inputs,goals


def strict_json_lines(path):
    import json
    def reject(token): raise AssertionError(f'Invalid JSON scalar {token}')
    return [json.loads(line,parse_constant=reject) for line in path.read_text().splitlines()]


def assert_final_raw_manifest(output,runner):
    import json
    manifest=json.loads((output/'raw_manifest.json').read_text())
    actual={str(f.relative_to(output)) for f in output.rglob('*') if f.is_file() and f.name!='raw_manifest.json'}
    assert set(manifest)==actual
    assert all(runner.sha256(output/name)==digest for name,digest in manifest.items())


def fake_module(tmp_path,output,fail_at=None,nonfinite=False):
    module=tmp_path/'build'/'modules'/'q02_probe.py'; module.parent.mkdir(parents=True)
    code="""import json
from pathlib import Path
import numpy as np
scope='FAKE CPU RECORDING TEST; NO NATIVE QUALIFICATION'
class Probe:
    def __init__(self): self.count=0
    def evaluate(self,xu,goal):
        i=self.count; self.count+=1
        out=Path(OUTPUT)
        event=json.loads((out/'native_calls.jsonl').read_text().splitlines()[-1])
        assert event['event']=='started' and event['call']==i
        # Validate started-call bytes exist before entry into evaluate.
        with np.load(out/event['input_raw']) as saved:
            assert saved['xu'].tobytes()==xu.tobytes()
            assert saved['goal'].tobytes()==goal.tobytes()
            assert saved['call'].item()==i
            assert saved['category'].item()==event['category']
        if i==FAIL_AT: raise RuntimeError('injected FD failure')
        kernel=np.array(0x7fc01234,dtype=np.uint32).view(np.float32) if NONFINITE else np.float32(.25)
        wall=np.float64(np.inf) if NONFINITE else np.float64(.001)
        return dict(flat=np.zeros((len(xu),995),dtype=np.float32),kernel_ms=kernel,wall_s=wall)
"""
    code=code.replace('OUTPUT',repr(str(output))).replace('FAIL_AT',repr(fail_at)).replace('NONFINITE',repr(nonfinite))
    module.write_text(code)
    return module


def test_full_native_run_archives_failed_fd_attempt_before_evaluate(tmp_path,monkeypatch,q02_runner,frozen):
    import sys
    runner=q02_runner; output=tmp_path/'failed-raw'
    inputs,goals=fake_main_inputs(monkeypatch,runner,frozen)
    module=fake_module(tmp_path,output,fail_at=17)
    monkeypatch.setattr(sys,'argv',['run.py','--output',str(output),'--module',str(module)])
    with pytest.raises(RuntimeError,match='injected FD failure'): runner.main()
    events=strict_json_lines(output/'native_calls.jsonl')
    starts=[e for e in events if e['event']=='started']; completed=[e for e in events if e['event']=='completed']
    assert len(starts)==18 and len(completed)==17
    failed=events[-1]
    assert failed['event']=='failed' and failed['call']==17 and failed['category']=='fd-0-1-plus'
    assert failed['error_type']=='RuntimeError' and 'injected FD failure' in failed['error']
    assert failed['wrapper_elapsed_s']>=0
    assert failed['output'] is None and failed['output_reason']=='evaluate_raised_no_output_returned'
    assert failed['kernel_ms'] is None and failed['kernel_ms_reason']=='evaluate_raised_native_clock_unavailable'
    assert failed['wall_s'] is None and failed['wall_s_reason']=='evaluate_raised_native_wrapper_clock_unavailable'
    expected=inputs[:1].copy(); expected[0,1]+=frozen['native_fd_step']
    with np.load(output/failed['input_raw']) as raw:
        np.testing.assert_array_equal(raw['xu'],expected)
        np.testing.assert_array_equal(raw['goal'],goals[:1])
    assert (output/'native_calls/0016.npz').exists()
    assert not (output/'native_calls/0017.npz').exists()
    assert 'injected FD failure' in (output/'execution_error.txt').read_text()
    assert_final_raw_manifest(output,runner)


def test_full_native_run_retains_nonfinite_timing_bytes_and_valid_json(tmp_path,monkeypatch,q02_runner,frozen):
    import json
    import sys
    runner=q02_runner; output=tmp_path/'nonfinite-timing-raw'
    fake_main_inputs(monkeypatch,runner,frozen)
    module=fake_module(tmp_path,output,nonfinite=True)
    monkeypatch.setattr(sys,'argv',['run.py','--output',str(output),'--module',str(module)])
    # Fake zero outputs fail the scientific GN comparisons; all 351 calls still
    # finish. This return is a recording test, never native qualification.
    assert runner.main()==1
    events=strict_json_lines(output/'native_calls.jsonl')
    completed=[e for e in events if e['event']=='completed']
    assert len(completed)==frozen['native_evaluate_calls']==351
    assert len([e for e in events if e['event']=='started'])==351
    assert not any(e['event']=='failed' for e in events)
    for e in completed:
        assert e['kernel_ms'] is None and e['kernel_ms_reason']=='nonfinite_nan'
        assert e['wall_s'] is None and e['wall_s_reason']=='nonfinite_infinity'
        with np.load(output/e['raw']) as raw:
            assert raw['kernel_ms'].dtype==np.dtype('float32')
            assert raw['kernel_ms'].view(np.uint32).item()==0x7fc01234
            assert raw['wall_s'].dtype==np.dtype('float64') and np.isposinf(raw['wall_s'])
    summary=json.loads((output/'summary.json').read_text())
    assert summary['native_calls']==351 and summary['status']=='FAIL'
    assert_final_raw_manifest(output,runner)


@pytest.mark.parametrize('stage',['module_import','probe_constructor'])
def test_native_setup_failure_preserves_frozen_attempts_and_manifest(tmp_path,monkeypatch,q02_runner,frozen,stage):
    import sys
    runner=q02_runner; output=tmp_path/'setup-failed-raw'
    inputs,goals=fake_main_inputs(monkeypatch,runner,frozen)
    module=tmp_path/'q02_probe.py'
    if stage=='module_import': module.write_text("raise RuntimeError('injected import failure')\n")
    else: module.write_text("scope='FAKE CPU TEST'\nclass Probe:\n def __init__(self): raise RuntimeError('injected constructor failure')\n")
    monkeypatch.setattr(sys,'argv',['run.py','--output',str(output),'--module',str(module)])
    with pytest.raises(RuntimeError,match='injected'): runner.main()
    events=strict_json_lines(output/'native_setup.jsonl')
    assert events[-1]['event']=='failed' and events[-1]['stage']==stage
    assert any(e['event']=='started' and e['stage']==stage for e in events)
    assert events[-1]['wrapper_elapsed_s']>=0
    assert events[-1]['kernel_ms'] is None and events[-1]['kernel_ms_reason']
    assert events[-1]['output'] is None and events[-1]['output_reason']
    with np.load(output/'native_attempted_inputs.npz') as raw:
        np.testing.assert_array_equal(raw['xu'],inputs); np.testing.assert_array_equal(raw['goal'],goals)
    assert not (output/'native_calls.jsonl').exists()
    assert 'injected' in (output/'execution_error.txt').read_text()
    assert_final_raw_manifest(output,runner)


def test_manifest_refresh_excludes_mutable_review_receipt(tmp_path,monkeypatch,q02_runner):
    import json
    import sys
    spec=importlib.util.spec_from_file_location('q02_freeze',ROOT/'experiments/campaign/q02/freeze.py')
    freeze=importlib.util.module_from_spec(spec); spec.loader.exec_module(freeze)
    here=tmp_path/'experiments/campaign/q02'; here.mkdir(parents=True)
    for name in freeze.SOURCE_FILES:
        path=tmp_path/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text('fake frozen source')
    for name in ['protocol.json','probe.cu','run.py','review.json','source_manifest.json','cpu_validation.json']:
        (here/name).write_text('{}')
    monkeypatch.setattr(freeze,'HERE',here); monkeypatch.setattr(freeze,'ROOT',tmp_path)
    monkeypatch.setattr(freeze.subprocess,'check_output',lambda *a,**k:'test-source-head\n')
    monkeypatch.setattr(sys,'argv',['freeze.py','--refresh-manifest-only'])
    freeze.main()
    manifest=json.loads((here/'source_manifest.json').read_text())
    assert 'experiments/campaign/q02/review.json' not in manifest['files']
    assert 'experiments/campaign/q02/source_manifest.json' not in manifest['files']
    assert 'experiments/campaign/q02/protocol.json' in manifest['files']
    (here/'review.json').write_text('mutable independent review')
    monkeypatch.setattr(q02_runner,'HERE',here); monkeypatch.setattr(q02_runner,'ROOT',tmp_path)
    assert q02_runner.verify_manifest()==manifest
