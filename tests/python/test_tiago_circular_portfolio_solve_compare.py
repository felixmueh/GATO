import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gato_tiago import circular_portfolio_p8 as schema
from gato_tiago import circular_portfolio_solve_compare as compare


def _stats(xu,batch):
    return {"XU":xu.copy(),"sqp_time_us":123,"sqp_iters":np.ones(batch,np.int32),
        "kkt_converged":np.ones(batch,np.int32),"initial_merit":np.full(batch,10.,np.float32),
        "final_merit":np.arange(batch,dtype=np.float32)/10,
        "ls_num_iters":1,"pcg_times_us":np.asarray([2.],np.float32),
        "pcg_iters":np.ones((1,batch),np.int32),"ls_min_merit":np.zeros((1,batch),np.float32),
        "ls_step_size":np.ones((1,batch),np.float32)}


class _Solver:
    instances=[]
    def __init__(self,*args):self.args=args;self.calls=[];self.__class__.instances.append(self)
    def solve(self,xu,dt,x0,reference):
        self.calls.append((xu.copy(),dt,x0.copy(),reference.copy()))
        return _stats(xu,len(xu))
    def sim_forward(self,state,control,dt):return np.asarray(state,np.float32)
    def tool_position(self,q):
        q=np.atleast_2d(np.asarray(q,np.float32));return np.c_[q[:,:2],np.full(len(q),.5,np.float32)]


def _seed(value):
    q=np.full((schema.KNOTS,7),value,np.float32);qd=np.zeros_like(q)
    u=np.zeros((schema.INTERVALS,7),np.float32);return schema.pack_seed(q,qd,u)


def _rows(batch):
    rows=[]
    for lane in range(batch):
        family="short" if lane<8 else "long";cost=5. if family=="short" else 1.
        rows.append({"lane":lane,"seed_family":family,"metrics":{"full_cost":cost},
            "gates":{"ok":True},"topology_preserved":True,"cost":{},"passes":True})
    return rows,np.zeros((batch,1,3),np.float32),np.zeros((batch,1,3)),np.zeros((batch,3,7))


def test_solver_options_unpack_and_exposed_stationarity_contract():
    assert compare.ROOT==Path("/tmp/tiago-circular-solve-compare-v4")
    assert compare.SOLVER_EXTENSION=={
        "module":"bsqp.bsqpN260_tiago_right_constructed_route_portfolio_toll_pcg_compact",
        "relative_path":"python/bsqp/bsqpN260_tiago_right_constructed_route_portfolio_toll_pcg_compact.cpython-310-x86_64-linux-gnu.so",
        "sha256":"464a72cf57440f948d6b40adf0a61caaa27071657721a8b7a488e55dd3e05432",
        "size":6678192,"build_head":"3aaf757dd2e0e0534bd1498aa57b3a4c5324a0de",
        "arch":"61-real","KNOT_POINTS":260,"REFERENCE_SIZE":10,
        "TOOL_POSITION_FRAME":"arm_right_tool_joint_origin","TOOL_POSITION_SIZE":3,
        "pcg_source_sha256":"38631d3716b0ea96c6bade40ecc659c83e51d73d5649ae05549df6a240c1291f"}
    assert "gato/bsqp/kernels/pcg.cuh" in compare.SOURCE_PATHS
    assert compare.solver_args()==(.05,60,1e-3,500,8e-4,1.,20.,2.,.15,7.5e-5,
        260.,800.,800.,.01,.001,.003,.01)
    seed=_seed(.25)[None];q,qd,u=compare.unpack_xu(seed,1)
    assert np.array_equal(q,np.full((1,schema.KNOTS,7),.25,np.float32))
    assert not qd.any() and not u.any()
    stats=compare.normalize_solve(_stats(seed,1),1)
    cert=compare.solver_certificate(stats)
    assert cert["passes"] and cert["lanes"][0]["kkt_converged"]==1
    bad=_stats(seed,1);bad["kkt_converged"][0]=0
    assert not compare.solver_certificate(compare.normalize_solve(bad,1))["passes"]
    for key,value in (("sqp_time_us",-1),("pcg_iters",np.asarray([[-1]],np.int32)),
                      ("pcg_times_us",np.asarray([-1.],np.float32)),
                      ("ls_step_size",np.asarray([[1.1]],np.float32)),
                      ("sqp_iters",np.asarray([2],np.int32))):
        bad=_stats(seed,1);bad[key]=value
        assert not compare.solver_certificate(compare.normalize_solve(bad,1))["passes"]
    sentinel=_stats(seed,1);sentinel["ls_step_size"][:]=-1
    assert compare.solver_certificate(compare.normalize_solve(sentinel,1))["passes"]
    bad=_stats(seed,1);bad["ls_step_size"][:]=-1.01
    assert not compare.solver_certificate(compare.normalize_solve(bad,1))["passes"]
    bad=_stats(seed,1);bad["ls_step_size"][:]=0
    assert not compare.solver_certificate(compare.normalize_solve(bad,1))["passes"]


def test_nonfinite_science_is_retained_as_json_null_instead_of_crashing():
    value=compare.builtin({"scalar":float("nan"),"nested":[np.float32("inf")],
        "array":np.asarray([1.,np.nan],np.float64)})
    assert value=={"scalar":None,"nested":[None],"array":[1.,None]}
    assert json.loads(json.dumps(value,allow_nan=False))==value
    b1,b16=_rows(1)[0],_rows(16)[0]
    b1[0]["passes"]=False
    for row in b16:row["passes"]=False
    failed=compare.comparison_certificate(b1,b16,{"passes":False},{"passes":False})
    assert failed["passes"] is False and failed["improvement"] is None
    assert json.loads(json.dumps(compare.builtin(failed),allow_nan=False))["improvement"] is None
    retained=np.asarray([1.,np.nan],np.float32)
    assert compare.exact_array(retained,retained.copy())
    assert not compare.exact_array(retained,np.asarray([1.,2.],np.float32))
    assert not compare.exact_array(retained,retained.astype(np.float64))


def test_exact_lane_mapping_and_cost_selection_requires_long_winner():
    b1=_rows(1)[0];b16=_rows(16)[0]
    stats={"passes":True}
    result=compare.comparison_certificate(b1,b16,stats,stats)
    assert result["passes"] and result["best_feasible_lane"]==8
    assert result["best_feasible_family"]=="long" and result["improvement"]==4.
    for row in b16[8:]:row["metrics"]["full_cost"]=4.5
    assert not compare.comparison_certificate(b1,b16,stats,stats)["passes"]
    b16=_rows(16)[0];b16[3]["passes"]=False
    assert not compare.comparison_certificate(b1,b16,stats,stats)["passes"]
    b16=_rows(16)[0];b16[3]["lane"]=4
    assert not compare.comparison_certificate(b1,b16,stats,stats)["passes"]


def test_execute_calls_exactly_one_b1_then_one_b16_and_recertifies_disk(tmp_path,monkeypatch):
    short=_seed(0.);long=_seed(.01);seed16=np.stack([short]*8+[long]*8)
    cached_arrays={"b1_seed_float32":short,"b16_seed_float32":seed16}
    construction={"x0_float32":np.zeros(14,np.float32),
        "reference_float32":np.zeros(schema.KNOTS*10,np.float32)}
    monkeypatch.setattr(compare,"load_inputs",lambda:({"passes":True},cached_arrays,construction))
    extension={};snap={"head":"a"*40,"clean":True,"cwd":compare.AUTHORIZED_CWD,
        "orig_argv":list(compare.AUTHORIZED_ORIG_ARGV),"thread_environment":compare.THREAD_ENV,
        "sources":{},"extension":extension}
    monkeypatch.setattr(compare,"snapshot",lambda:snap)
    monkeypatch.setattr(compare,"certify_module",lambda module:True)
    monkeypatch.setattr(compare,"frozen_extension",lambda:extension)
    monkeypatch.setattr(compare,"worker",SimpleNamespace(
        cuda_diagnostics=lambda:{"ok":True},certify_cuda_diagnostics=lambda value:value=={"ok":True}))
    prerequisite={};monkeypatch.setattr(compare,"authenticate_cpu_prerequisite",
        lambda:({"passes":True},prerequisite))
    monkeypatch.setattr(compare,"_production_pin_context",lambda:(None,lambda q:(q[:3],np.zeros((3,7))),
        None,None,None,None,None,None))
    monkeypatch.setattr(compare,"evaluate_lanes",lambda planned,*args:_rows(len(planned)))
    _Solver.instances=[];module=SimpleNamespace(BSQP_1_float=_Solver,BSQP_16_float=_Solver)
    times=iter(np.linspace(0.,1.,100));root=tmp_path/"result"
    result=compare.execute(root,lambda name:module,lambda:next(times),
        compare.authenticate_cpu_prerequisite,compare._production_pin_context,
        lambda:{"ok":True})
    assert len(_Solver.instances)==2
    assert [len(item.calls) for item in _Solver.instances]==[1,1]
    call1,call16=_Solver.instances[0].calls[0],_Solver.instances[1].calls[0]
    assert call1[1]==call16[1]==.05 and np.array_equal(call1[0][0],call16[0][0])
    assert np.array_equal(call16[0][:8],np.tile(short,(8,1)))
    assert np.array_equal(call16[0][8:],np.tile(long,(8,1)))
    assert call1[0].dtype==call16[0].dtype==np.float32
    assert result["counts"]=={"b1_constructors":1,"b16_constructors":1,
        "b1_solve_calls":1,"b16_solve_calls":1} and result["passes"]
    assert compare.recertify(root,lambda name:module,compare.authenticate_cpu_prerequisite,
        compare._production_pin_context)["passes"]
    document=json.loads((root/"result.json").read_text())
    with np.load(root/"result.npz",allow_pickle=False) as archive:arrays={key:archive[key] for key in archive.files}
    arrays["raw_initial_merit_b16"][8]+=1
    with (root/"result.npz").open("wb") as stream:np.savez(stream,**arrays)
    document["npz_sha256"]=compare.sha(root/"result.npz")
    document["array_hashes"]["raw_initial_merit_b16"]=schema.array_hash(arrays["raw_initial_merit_b16"])
    (root/"result.json").write_text(json.dumps(document,sort_keys=True,separators=(",",":")))
    assert not compare.recertify(root,lambda name:module,compare.authenticate_cpu_prerequisite,
        compare._production_pin_context)["passes"]


def test_refuses_wrong_pins_and_existing_root(tmp_path):
    root=tmp_path/"exists";root.mkdir()
    with pytest.raises(FileExistsError):compare.execute(root)
    with pytest.raises(RuntimeError,match="pin mismatch"):
        compare.load_inputs(tmp_path/"missing.json",tmp_path/"missing.npz")
