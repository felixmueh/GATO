"""Bounded current-cylinder preparation checks; no native solve or target gate."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
SOURCE=Path('/workspace/GATO-eval-cylinder')
sys.path[:0]=[str(SOURCE/'python'),str(SOURCE/'tiago_src'),str(SOURCE/'examples/randomized_multimodal'),str(SOURCE/'experiments/cylinder_current')]

def worker():
    spec=importlib.util.spec_from_file_location('cylinder_worker_test',SOURCE/'experiments/cylinder_current/run_tasks.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_physically_timed_route_prefix():
    m=worker();x=np.zeros((1,32,14));u=np.zeros((1,31,7));x[0,:,0]=np.arange(32)*.06;u[0,:,0]=np.arange(31)
    z=m.prefix_seed(m.pack(x,u)[0]);xs,us=m.unpack(z)
    assert xs[0,-1,0]==pytest.approx(.93)
    np.testing.assert_allclose(xs[0,:,0],np.arange(32)*.03,atol=1e-7)
    assert us[0,-1,0]==15 # Original control at physical time.90, not index30.

def test_stage_terminal_cylinder_oracle_scope():
    spec=importlib.util.spec_from_file_location('cylinder_checks_test',SOURCE/'experiments/cylinder_current/checks.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    f=np.zeros(995);f[462:464]=[.05,0];j=np.zeros((6,7));j[0,0]=1;f[468:510]=j.reshape(-1,order='F')
    cost,g,Q=m.expected(f,np.array([0,0,0,0,0,.1]))
    assert cost==pytest.approx(.28125);assert g[0]==pytest.approx(-7.5);assert Q[0,0]==pytest.approx(100)
    f[462]=.15;cost,g,Q=m.expected(f,np.array([0,0,0,0,0,.1]));assert cost==0 and not g.any() and not Q.any()

def test_paid_old_plan_and_terminal_discard(tmp_path,monkeypatch):
    m=worker();x=np.zeros(14);scene=dict(initial_state=x.tolist(),goal_xyz=[10,10,10],seconds=8,obstacle_running_cost=1,obstacle_terminal_cost=1,
        cylinder=dict(initial_center_xy=[10,10],final_center_xy=[10,10],radius=.1,motion_start_seconds=0,motion_duration_seconds=1))
    class Solver:
        solver=SimpleNamespace()
        def set_f_ext_B(self,x):pass
        def reset_dual(self):pass
        def reset_rho(self):pass
    controller=SimpleNamespace(solver=Solver(),actual_f_ext=np.zeros(6))
    calls=[]
    def solve(c,state,reference,warm,out,step,cfg,**kwargs):
        optimized=warm.copy();optimized[:,14:21]=step+1
        return optimized,1000.,4.,dict(final_merit=np.array([1.]),step_size=np.ones((1,1)))
    monkeypatch.setattr(m,'solve_recorded',solve)
    monkeypatch.setattr(m,'assess',lambda *a:(0,[dict(finite=True)],[(x[None],np.zeros(0))]))
    def execute(plan,state,start,requested,activation,**kwargs):
        actual=min(requested,8-start);calls.append((start,requested,activation,float(plan[14])))
        kwargs['completed_step'](state.copy(),np.zeros(7),actual,start+actual,kwargs['plan_index'],start-activation)
        return actual,actual<requested
    monkeypatch.setattr(m,'execute_active_plan',execute)
    events=[];result=m.run(SimpleNamespace(output=str(tmp_path),scene_id=4),scene,1,lambda kind,row:events.append((kind,row)),lambda model,batch,scene:(controller,{}))
    assert calls==[(0.,.03,0.,1.),(.03,4.,0.,1.),(4.03,4.,4.03,2.)]
    assert result['simulated_seconds']==8 and result['task_success'] is False
    rows=[r for kind,r in events if kind=='feedback_completed']
    assert rows[-1]['activation_status']=='discarded_terminal_or_selection_failure'
    assert rows[-1]['optimizer_origin_s']==4.03
    assert result['physical_pass'] is True

def test_seed_sources_and_current_source_unchanged():
    import ast,subprocess
    saved=ast.parse((SOURCE/'experiments/cylinder_current/helpers.py').read_text())
    old=ast.parse(subprocess.check_output(['git','-C',str(SOURCE),'show','c084033b4147d267e094dd7699e0f31dd0b7ba6a:examples/randomized_multimodal/tiago_moving_pillar.py'],text=True))
    functions={n.name:ast.dump(n) for n in old.body if isinstance(n,ast.FunctionDef)}
    assert all(ast.dump(n)==functions[n.name] for n in saved.body if isinstance(n,ast.FunctionDef))
    assert not subprocess.check_output(['git','-C',str(SOURCE),'diff','--name-only','4f6c8fd','--','gato','python','tiago_src','tests'],text=True)
    import json
    root=Path(__file__).resolve().parent
    desktop=json.loads((root/'cylinder_studies.json').read_text())['experiments']
    jetson=json.loads((root/'cylinder_studies_jetson_proposed.json').read_text())['experiments']
    assert sum(len(e['cases']) for e in desktop)==9
    assert desktop[0]['timing_mode']=='A' and 'preflight' not in desktop[0]
    assert desktop[1]['timing_mode']=='B' and 'preflight' in desktop[1]
    assert sum(len(e['cases']) for e in jetson)==6 and all(e['timing_mode']=='B' for e in jetson)


def test_ordinary_mode_b_dependency_isolated_from_cylinder_worker(tmp_path,monkeypatch):
    import json
    import run as runner
    from adapters import cylinder_studies as adapter
    source=tmp_path/'cylinder';runtime=tmp_path/'ordinary';inputs=tmp_path/'inputs';inputs.mkdir()
    build=tmp_path/'build';(build/'modules').mkdir(parents=True)
    ordinary=build/'modules/bsqpN32_tiago_right.synthetic.so';ordinary.write_bytes(b'synthetic')
    (build/'modules/bsqpN32_tiago_right_multimodal.synthetic.so').write_bytes(b'synthetic cylinder')
    (inputs/'v01.json').write_text(json.dumps(dict(initial_state=[0.]*14,goal=[0.]*6)))
    assert adapter.exact_module(build,'bsqpN32_tiago_right.')==ordinary
    original_native=adapter.task.native_module
    def checkout(pin,cache):
        assert pin==adapter.ORDINARY_RUNTIME and cache==build/'prerequisite-cache';return runtime
    monkeypatch.setattr(runner,'worktree',checkout)
    def canonical(src,bld,case,out,mode,preflight,inp):
        assert (src,bld,case,mode,preflight,inp)==(runtime,build,None,'B',True,inputs)
        assert adapter.task.native_module is not original_native
        return 0
    monkeypatch.setattr(adapter.task,'run',canonical)
    assert adapter.ordinary_preflight(source,build,inputs,tmp_path/'ordinary-canonical')==0
    assert adapter.task.native_module is original_native
    calls=[]
    def subprocess_call(command):
        assert '--ordinary-preflight' in command and command[command.index('--source')+1]==str(source)
        calls.append(command);return SimpleNamespace(returncode=0)
    monkeypatch.setattr(adapter.subprocess,'run',subprocess_call)
    monkeypatch.setattr(adapter.subprocess,'check_output',lambda *args,**kwargs:'d6c8278\n')
    monkeypatch.setattr(adapter.control_studies,'ensure_suite',lambda *args:0)
    def cylinder(src,bld,inp,out,*args):
        assert src==source
        out.mkdir(parents=True)
        adapter.write_json(out/'metrics.json',dict(task_success=True,physical_pass=True,execution_error=None))
        return 0
    monkeypatch.setattr(adapter,'collect',cylinder)
    assert adapter.run(source,build,inputs,tmp_path/'preflight',preflight=True)==0
    assert len(calls)==1
    metrics=json.loads((tmp_path/'preflight/metrics.json').read_text())
    assert metrics['ordinary_runtime_source_commit']==adapter.ORDINARY_RUNTIME
    assert metrics['cylinder_runtime_source_commit']=='d6c8278'
    assert metrics['ordinary_native_source_commit']=='d6c8278'
