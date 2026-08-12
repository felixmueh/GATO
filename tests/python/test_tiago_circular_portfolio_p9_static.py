import inspect,json,re
from pathlib import Path

import numpy as np
import pytest

from gato_tiago import circular_portfolio_p8 as p8_schema
from gato_tiago import circular_portfolio_p8_runner as p8_runner
from gato_tiago import circular_portfolio_p8_worker as p8_worker
from gato_tiago import circular_portfolio_p9 as schema
from gato_tiago import circular_portfolio_p9_runner as runner
from gato_tiago import circular_portfolio_p9_worker as worker


def test_p9_isolated_closed_scope_exact_p8_report_and_fresh_root():
    assert p8_runner.RUNNER_EXECUTION_AUTHORIZATION is None
    assert p8_worker.WORKER_EXECUTION_AUTHORIZATION is None
    assert schema.RUNNER_EXECUTION_AUTHORIZATION is None
    assert schema.WORKER_EXECUTION_AUTHORIZATION is None
    assert runner.RUNNER_EXECUTION_AUTHORIZATION is not None
    assert worker.WORKER_EXECUTION_AUTHORIZATION is not None
    root=Path(__file__).resolve().parents[2]
    pattern=re.compile(r"^[A-Z][A-Z0-9_]*AUTHORIZATION\s*=\s*object\(\)$")
    assert sorted((path.name,line) for path in (root/"tiago_src/gato_tiago").glob("*.py")
        for line in path.read_text().splitlines() if pattern.fullmatch(line))==[
            ("circular_portfolio_p9_runner.py","RUNNER_EXECUTION_AUTHORIZATION=object()"),
            ("circular_portfolio_p9_worker.py","WORKER_EXECUTION_AUTHORIZATION=object()")]
    report=schema.P8_REJECTED_REPORT
    assert report["classification"]=="operational_cpu_preflight_timeout_only"
    assert report["trigger_elapsed_s"]==30.06871920998674
    assert report["cleanup_finish_elapsed_s"]==30.069459119986277
    assert report["completed"]==0 and report["pending"]==2
    assert report["nonzero_counts"]=={"prerequisite_loads":1,"parent_pin_contexts":1,
        "primary_cpu_endpoint_fk_calls":2,"primary_dls_kinematics_calls":214}
    assert {key:value for key,value in report["retained_counts"].items() if value} \
        ==report["nonzero_counts"]
    assert set(report["retained_counts"])==set(schema._P8_COUNT_KEYS)
    assert report["worker_cuda_calls"]==report["optimizer_calls"]==report["rng_calls"]==0
    assert report["rejected_p8_artifact_loads"]==0
    assert not schema.OUTPUT.parent.exists()


def test_p9_science_is_shared_and_only_operational_constants_change():
    assert schema.CONSTRUCTION_SPECS is p8_schema.CONSTRUCTION_SPECS
    assert schema.WORKER_OUTPUT_SPECS is p8_schema.WORKER_OUTPUT_SPECS
    assert schema.EXTENSION is p8_schema.EXTENSION
    assert (schema.KNOTS,schema.DT,schema.IK_ITERATIONS,schema.DLS_DAMPING,
        schema.KP,schema.KD,schema.AFFINE_SUBSTEPS)==(260,.05,8,.03,25.,10.,16)
    assert (schema.CPU_WALL_LIMIT_S,schema.RUNNER_WALL_LIMIT_S,
        schema.WORKER_WALL_LIMIT_S)==(1500.,2100.,300.)
    assert worker.generate is p8_worker.generate
    assert worker.certify_output is p8_worker.certify_output
    assert "optimizer" not in inspect.getsource(runner.execute).lower()
    with runner.p9_context():
        assert p8_runner.operational_limits()=={"cpu_wall_limit_s":1500.,
            "worker_wall_limit_s":300.,"campaign_wall_limit_s":2100.}
        assert p8_runner.PROTOCOL==schema.PROTOCOL and p8_runner.OUTPUT==schema.OUTPUT
        assert p8_worker.WORKER_PROTOCOL==schema.WORKER_PROTOCOL
    assert p8_runner.PROTOCOL==p8_schema.PROTOCOL and p8_runner.OUTPUT==p8_schema.OUTPUT
    source=inspect.getsource(p8_runner.execute)+inspect.getsource(p8_runner.recertify_retained_p8)
    assert "start+30." not in source and "start+600." not in source
    with pytest.raises(RuntimeError,match="injected"):
        with runner.p9_context():raise RuntimeError("injected")
    assert p8_runner.PROTOCOL==p8_schema.PROTOCOL and p8_worker.WORKER_PROTOCOL==p8_schema.WORKER_PROTOCOL


def test_p9_exact_argv_paths_and_disabled_boundaries():
    assert runner.AUTHORIZED_ORIG_ARGV==("python","-B","-m",
        "gato_tiago.circular_portfolio_p9_runner","--execute","--output",str(schema.OUTPUT))
    paths=schema.worker_paths()
    assert worker.expected_argv()==("python","-B","-m","gato_tiago.circular_portfolio_p9_worker",
        "--execute","--request",str(paths["request"]),"--output",str(paths["json"]))
    with pytest.raises(RuntimeError,match="wrong P8 output"):
        runner.execute(schema.OUTPUT.with_name("wrong.json"),runner.RUNNER_EXECUTION_AUTHORIZATION)
    with pytest.raises(RuntimeError,match="wrong P8 worker request"):
        worker.execute(paths["request"].with_name("wrong.request.json"),
            worker.WORKER_EXECUTION_AUTHORIZATION)
    assert "if OUTPUT.parent.exists()" in inspect.getsource(p8_runner.execute)
    assert 'Path(str(path)+".candidate").exists()' in inspect.getsource(p8_worker.execute)


def test_p9_fake_clock_cpu_1500_rejection_and_public_recert(tmp_path,monkeypatch):
    output=tmp_path/"fresh"/"p9.json";monkeypatch.setattr(schema,"OUTPUT",output)
    snap={"head":"a"*40,"clean":True,"sources":{"x":"b"*64},
        "extension":p8_runner.frozen_extension()}
    monkeypatch.setattr(p8_runner,"snapshot",lambda:snap)
    monkeypatch.setattr(p8_runner.sys,"orig_argv",list((*runner.AUTHORIZED_ORIG_ARGV[:-1],str(output))))
    q0=np.zeros(7);qg=np.zeros(7);qg[0]=.08
    prerequisite={"public_x0_float32":np.r_[q0,np.zeros(7)][None].astype(np.float32),
        "quarantined_q8_float64":qg[None],"public_default_side_int8":np.asarray([1],np.int8),
        "joint_lower_float64":np.full(7,-10.),"joint_upper_float64":np.full(7,10.)}
    monkeypatch.setattr(p8_runner,"authenticate_cpu_prerequisite",
        lambda:({"passes":True},prerequisite))
    def kin(q):
        J=np.zeros((3,7));J[:3,:3]=np.eye(3);return np.asarray([q[0],q[1],.5]),J
    monkeypatch.setattr(p8_runner,"_production_pin_context",
        lambda:(None,kin,lambda q,v,a:a,None,None,None,None))
    times=iter((0.,1500.1,1500.1,1500.2))
    token=object();monkeypatch.setattr(runner,"RUNNER_EXECUTION_AUTHORIZATION",token)
    with pytest.raises(TimeoutError,match="CPU preflight"):
        runner.execute(output,token,lambda:next(times))
    rejected=json.loads(schema.rejection_path().read_text())
    assert rejected["stage"]=="runtime_watchdog_rejected"
    assert rejected["operational_limits"]=={"cpu_wall_limit_s":1500.,
        "worker_wall_limit_s":300.,"campaign_wall_limit_s":2100.}
    assert rejected["counts"]["primary_cpu_endpoint_fk_calls"]==2
    assert rejected["counts"]["primary_dls_kinematics_calls"]==0
    assert rejected["counts"]["worker_attempts"]==0
    assert rejected["wall_limit_s"]==2100.
    assert runner.recertify_retained_failure(output)["passes"]


def test_p9_no_p8_artifact_access_or_science_override_in_adapters():
    source=inspect.getsource(runner)+inspect.getsource(worker)
    assert "/tmp/tiago-tool-center-circular-route-feasibility-p8" not in source
    assert "rejected_p8_artifact_loads" not in inspect.getsource(runner.execute)
    assert all(token not in source for token in ("minimize(","default_rng","sqp("))
