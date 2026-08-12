import inspect,json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gato_tiago import circular_portfolio_cached_replay as replay
from gato_tiago import circular_portfolio_p8 as schema
from gato_tiago import circular_portfolio_p8_worker as worker
from gato_tiago.circular_portfolio_p5_worker import cuda_diagnostics
from gato_tiago import circular_portfolio_p5_worker as p5_worker


def _construction():
    arrays={key:np.zeros(shape,dtype) for key,(shape,dtype) in schema.CONSTRUCTION_SPECS.items()}
    arrays["proxy_q_float64"][1,:,0]=np.linspace(0.,.01,schema.KNOTS)
    arrays["proxy_qd_float64"][1,1:-1,0]=np.diff(
        arrays["proxy_q_float64"][1,:-1,0])/schema.DT
    arrays["proxy_qdd_float64"][:]=np.diff(arrays["proxy_qd_float64"],axis=1)/schema.DT
    return arrays


class _B1:
    def __init__(self,*args):pass
    def sim_forward(self,state,control,dt):
        out=np.atleast_2d(np.asarray(state,np.float32)).copy()
        out[:,:7]+=dt*np.atleast_2d(np.asarray(control,np.float32));return out
    def tool_position(self,q):return np.atleast_2d(np.asarray(q,np.float32))[:,:3]


class _B16(_B1):pass


def test_actual_cuda_diagnostics_signature_and_corrected_call_source(monkeypatch):
    assert tuple(inspect.signature(cuda_diagnostics).parameters)==()
    source=inspect.getsource(worker.execute)
    assert "diagnostics=cuda_diagnostics()" in source
    assert source.index("certify_module(module)")<source.index("cuda_diagnostics()") \
        <source.index("_production_pin_context()")
    class Library:
        @staticmethod
        def cudaRuntimeGetVersion(pointer):pointer._obj.value=12090;return 0
        @staticmethod
        def cudaDriverGetVersion(pointer):pointer._obj.value=13000;return 0
    monkeypatch.setattr(p5_worker.ctypes,"CDLL",lambda name:Library())
    monkeypatch.setattr(p5_worker.subprocess,"run",lambda *args,**kwargs:
        SimpleNamespace(stdout="uuid, gpu, driver, 6.1\n"))
    assert worker.cuda_diagnostics()["cuda_runtime_version"]==12090


def test_real_source_snapshot_has_cli_and_runtime_critical_diagnostics_source():
    value=replay.source_snapshot()
    assert isinstance(value["clean"],bool) and len(value["head"])==40
    assert value["cwd"]==str(Path.cwd().resolve()) and isinstance(value["argv"],list)
    assert "tiago_src/gato_tiago/circular_portfolio_p5_worker.py" in value["sources"]
    assert value["extension"]==worker.frozen_extension()


def test_cached_replay_pins_and_atomic_refusal(tmp_path,monkeypatch):
    source=tmp_path/"p9.worker.input.npz"
    with source.open("wb") as stream:np.savez(stream,**_construction())
    digest=replay.sha(source);monkeypatch.setattr(replay,"P9_INPUT",source.resolve())
    monkeypatch.setattr(replay,"P9_INPUT_SHA256",digest)
    monkeypatch.setattr(replay,"P9_INPUT_SIZE",source.stat().st_size)
    gen1=tmp_path/"p9.gen1.json";gen1.write_text(json.dumps({"protocol":
        "tiago_tool_center_circular_route_feasibility_p9_1","generation":1,
        "stage":"cpu_authenticated","incomplete":True,"counts":{},
        "detail":{"construction":{"passes":True}},"provenance":{},
        "operational_limits":{},"evidence":False}));monkeypatch.setattr(
        replay,"P9_GEN1",gen1.resolve());monkeypatch.setattr(replay,"P9_GEN1_SHA256",replay.sha(gen1))
    monkeypatch.setattr(replay,"P9_GEN1_SIZE",gen1.stat().st_size)
    assert schema.exact_arrays(replay.load_cached_construction(source,gen1),schema.CONSTRUCTION_SPECS)
    source.write_bytes(source.read_bytes()+b"x")
    with pytest.raises(RuntimeError,match="pin mismatch"):replay.load_cached_construction(source,None)
    root=tmp_path/"existing";root.mkdir()
    with pytest.raises(FileExistsError,match="root exists"):
        replay.run_cached_replay(source,None,root)


def test_cached_replay_runs_real_generate_and_output_certificate_with_mock_kernels(tmp_path,monkeypatch):
    def kin(q):
        J=np.zeros((3,7));J[:3,:3]=np.eye(3)
        return np.asarray([q[0],q[1],.5]),J
    monkeypatch.setattr(replay.science,"exact_box_dls",lambda J,res,q,lo,hi:
        (np.r_[res[:2],np.zeros(5)],np.zeros(7,np.int8)))
    q0=np.zeros(7);qgoal=np.zeros(7);qgoal[0]=.08
    prerequisite={"public_x0_float32":np.r_[q0,np.zeros(7)][None].astype(np.float32),
        "quarantined_q8_float64":qgoal[None],"public_default_side_int8":np.asarray([1],np.int8),
        "joint_lower_float64":np.full(7,-100.),"joint_upper_float64":np.full(7,100.),
        "velocity_limit_float64":np.full(7,100.),"effort_limit_float64":np.full(7,10000.)}
    construction,_=replay.science.construct_cpu(prerequisite,kin,float("inf"))
    source=tmp_path/"p9.worker.input.npz"
    with source.open("wb") as stream:np.savez(stream,**construction)
    gen1=tmp_path/"p9.gen1.json";gen1.write_text(json.dumps({"protocol":
        "tiago_tool_center_circular_route_feasibility_p9_1","generation":1,
        "stage":"cpu_authenticated","incomplete":True,"counts":{},
        "detail":{"construction":{"passes":True}},"provenance":{},
        "operational_limits":{},"evidence":False}))
    monkeypatch.setattr(replay,"P9_INPUT",source.resolve());monkeypatch.setattr(
        replay,"P9_INPUT_SHA256",replay.sha(source))
    monkeypatch.setattr(replay,"P9_INPUT_SIZE",source.stat().st_size)
    monkeypatch.setattr(replay,"P9_GEN1",gen1.resolve());monkeypatch.setattr(
        replay,"P9_GEN1_SHA256",replay.sha(gen1))
    monkeypatch.setattr(replay,"P9_GEN1_SIZE",gen1.stat().st_size)
    extension=worker.frozen_extension()
    run_snapshot={"head":"a"*40,"clean":True,"cwd":replay.AUTHORIZED_CWD,
        "argv":list(replay.AUTHORIZED_ORIG_ARGV),"sources":{},"extension":extension}
    monkeypatch.setattr(replay,"source_snapshot",lambda:run_snapshot)
    monkeypatch.setattr(worker,"certify_module",lambda module:True)
    monkeypatch.setattr(worker,"module_measurement",lambda module:extension)
    monkeypatch.setattr(replay,"authenticate_cpu_prerequisite",lambda:({"passes":True},prerequisite))
    monkeypatch.setattr(replay,"_production_pin_context",lambda:(None,kin,
        lambda q,v,a:a,None,None,None,None))
    class B1:
        def __init__(self,*args):self.call=0
        def sim_forward(self,state,control,dt):
            route=(self.call//schema.INTERVALS)%2;knot=self.call%schema.INTERVALS
            self.call+=1
            return np.concatenate((construction["proxy_q_float64"][route,knot+1],
                construction["proxy_qd_float64"][route,knot+1]))[None].astype(np.float32)
        def tool_position(self,q):
            q=np.atleast_2d(np.asarray(q));return np.c_[q[:,:2],np.full(len(q),.5)].astype(np.float32)
    class B16(B1):
        def sim_forward(self,state,control,dt):
            knot=self.call%schema.INTERVALS;self.call+=1
            return np.stack([np.concatenate((construction["proxy_q_float64"][0 if lane<8 else 1,knot+1],
                construction["proxy_qd_float64"][0 if lane<8 else 1,knot+1])) for lane in range(16)]).astype(np.float32)
    module=SimpleNamespace(BSQP_1_float=B1,BSQP_16_float=B16)
    diagnostics={"uuid":"u","name":"gpu","driver_version":"1","compute_capability":"6.1",
        "cuda_runtime_version":1,"cuda_driver_api_version":1}
    clock=iter(np.linspace(0.,1.,50000));root=tmp_path/"result"
    result=replay.run_cached_replay(source,gen1,root,lambda:next(clock),
        lambda name:module,lambda:diagnostics)
    with np.load(root/"result.npz",allow_pickle=False) as archive:
        arrays={key:archive[key] for key in archive.files}
    assert worker.certify_output(arrays,construction)
    assert result["output_certificate"] is True and result["campaign_certificate"]["passes"] is True
    assert result["counts"]==worker.SUCCESS_COUNTS
    assert result["optimizer_calls"]==result["solve_calls"]==result["sqp_calls"]==0
    assert json.loads((root/"result.json").read_text())==result
    audit_snapshot={**result["source"]["start"],"argv":["python","-m","independent_auditor"]}
    monkeypatch.setattr(replay,"source_snapshot",lambda:audit_snapshot)
    assert replay.recertify_cached_replay(root)["passes"]
    arrays["rnea_qdd_float64"][0,0,0]+=1.
    with (root/"result.npz").open("wb") as stream:np.savez(stream,**arrays)
    result["output_npz_sha256"]=replay.sha(root/"result.npz")
    result["array_hashes"]["rnea_qdd_float64"]=schema.array_hash(arrays["rnea_qdd_float64"])
    (root/"result.json").write_text(json.dumps(result,sort_keys=True,separators=(",",":")))
    assert not replay.recertify_cached_replay(root)["passes"]


def test_cached_replay_failure_is_one_atomic_builtin_document(tmp_path,monkeypatch):
    source=tmp_path/"bad.npz";source.write_bytes(b"bad")
    monkeypatch.setattr(replay,"P9_INPUT",source.resolve())
    monkeypatch.setattr(replay,"P9_INPUT_SHA256","0"*64)
    monkeypatch.setattr(replay,"P9_INPUT_SIZE",source.stat().st_size)
    root=tmp_path/"failure"
    with pytest.raises(RuntimeError,match="pin mismatch"):
        replay.run_cached_replay(source,None,root,lambda:0.)
    assert sorted(path.name for path in root.iterdir())==["failure.json"]
    value=json.loads((root/"failure.json").read_text())
    assert value["evidence"] is False and value["error_type"]=="RuntimeError"


def test_cli_exits_nonzero_after_retaining_false_science(monkeypatch,tmp_path):
    monkeypatch.setattr(replay,"run_cached_replay",lambda *args,**kwargs:{"passes":False})
    monkeypatch.setattr(replay.sys,"orig_argv",list(replay.AUTHORIZED_ORIG_ARGV))
    with pytest.raises(SystemExit) as error:
        replay.main(["--input",str(replay.P9_INPUT),"--gen1",str(replay.P9_GEN1),
            "--output-root",str(replay.ROOT)])
    assert error.value.code==1
