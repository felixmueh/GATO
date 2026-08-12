import copy,json,re
from pathlib import Path

import numpy as np
import pytest

from gato_tiago import circular_portfolio_p5 as v1_schema
from gato_tiago import circular_portfolio_p5_runner as v1_runner
from gato_tiago import circular_portfolio_p5_worker as v1_worker
from gato_tiago import circular_portfolio_p5_v2 as schema
from gato_tiago import circular_portfolio_p5_v2_runner as runner
from gato_tiago import circular_portfolio_p5_v2_worker as worker


def test_v1_and_v2_pilots_are_closed_for_p6_static_work():
    report=schema.P5_V1_REJECTED_REPORT
    assert report["classification"]=="launch_invalid_serialization_failure"
    assert report["hashes"]=={
        "gen0_json":"24acf0830bcc434811c72dbaa3f1ad410c9750c684c6bb33e87249e70be81e5b",
        "latest":"6c8b9e96193b84c04cb9f19b5b7a0db5c9a1da0a94f0cf21c9feac141bf388"}
    assert len(report["retained_counts"])==31 and set(report["retained_counts"].values())=={0}
    assert report["independently_reconstructed_execution"]["prerequisite_internal_fk_calls"]==24
    assert report["independently_reconstructed_execution"]["prerequisite_artifact_loads"]==1
    assert len(report["bad_paths"])==12 and all("circle_endpoint_identity" in x for x in report["bad_paths"])
    assert report["rejected_p5_v1_artifact_loads"]==0
    assert v1_runner.RUNNER_EXECUTION_AUTHORIZATION is None
    assert v1_worker.WORKER_EXECUTION_AUTHORIZATION is None
    assert runner.RUNNER_EXECUTION_AUTHORIZATION is None
    assert worker.WORKER_EXECUTION_AUTHORIZATION is None
    root=Path(__file__).resolve().parents[2]
    pattern=re.compile(r"^[A-Z][A-Z0-9_]*AUTHORIZATION\s*=\s*object\(\)$")
    enabled=sorted((path.name,line) for path in (root/"tiago_src/gato_tiago").glob("*.py")
        for line in path.read_text().splitlines() if pattern.fullmatch(line))
    assert enabled==[("circular_portfolio_p9_runner.py","RUNNER_EXECUTION_AUTHORIZATION=object()"),
        ("circular_portfolio_p9_worker.py","WORKER_EXECUTION_AUTHORIZATION=object()")]
    assert schema.PROTOCOL!=v1_schema.PROTOCOL and schema.OUTPUT!=v1_schema.OUTPUT
    assert schema.OUTPUT.parent.exists() and not schema.OUTPUT.exists()
    assert runner.AUTHORIZED_ORIG_ARGV==("python","-B","-m",
        "gato_tiago.circular_portfolio_p5_v2_runner","--execute","--output",str(schema.OUTPUT))
    with pytest.raises(RuntimeError,match="P5 V2 runner blocked"):
        runner.execute(schema.OUTPUT.with_name("wrong.json"),runner.RUNNER_EXECUTION_AUTHORIZATION)
    with worker.v2_context():
        assert v1_worker.expected_argv()==("python","-B","-m",
            "gato_tiago.circular_portfolio_p5_v2_worker","--execute","--request",
            str(schema.worker_paths()["request"]),"--output",str(schema.worker_paths()["json"]))
    source=Path(runner.__file__).read_text()
    assert source.index("if Path(output).resolve()!=schema.OUTPUT") \
        <source.index("with v2_context()")
    assert "if OUTPUT.parent.exists()" in Path(v1_runner.__file__).read_text()
    assert "if path.exists() or candidate.exists()" in Path(v1_runner.__file__).read_text()


def test_recursive_canonicalization_exact_bool_paths_and_nested_types():
    detail={"recertification":{"final":{"semantic":{"geometry_details":[
        {"certificate":{"gates":{"circle_endpoint_identity":np.bool_(True)}}}
        for _ in range(12)]}}},"future":{"i":np.int64(7),"f":np.float32(.5),
        "scalar":np.asarray(True),"vector":np.asarray([1,2],np.int16),
        "matrix":np.asarray([[1.],[2.]],np.float64),"nested":({"x":np.uint8(3)},)}}
    value=runner.canonical_authentication(detail)
    gates=value["recertification"]["final"]["semantic"]["geometry_details"]
    assert [row["certificate"]["gates"]["circle_endpoint_identity"] for row in gates]==[True]*12
    assert value["future"]=={"i":7,"f":.5,"scalar":True,"vector":[1,2],
        "matrix":[[1.0],[2.0]],"nested":[{"x":3}]}
    assert json.loads(json.dumps(value,allow_nan=False))==value


@pytest.mark.parametrize("value",(float("nan"),float("inf"),np.float64(-np.inf),{1:"bad"},object()))
def test_canonicalization_fails_closed_on_nonfinite_or_unsupported(value):
    with pytest.raises((TypeError,ValueError)):runner.canonical_authentication({"value":value})


def test_scoped_context_canonicalizes_every_auth_and_restores_globals(monkeypatch):
    import gato_tiago.circular_portfolio_runner as predecessor
    original=predecessor.authenticate_cpu_prerequisite
    detail={"value":np.bool_(True)};arrays={"array":np.asarray([1.])}
    monkeypatch.setattr(predecessor,"authenticate_cpu_prerequisite",lambda:(detail,arrays))
    patched=predecessor.authenticate_cpu_prerequisite
    with runner.v2_context():
        got_detail,got_arrays=predecessor.authenticate_cpu_prerequisite()
        assert got_detail=={"value":True} and got_arrays is arrays
        assert v1_runner.OUTPUT==schema.OUTPUT and v1_schema.OUTPUT==schema.OUTPUT
    assert predecessor.authenticate_cpu_prerequisite is patched
    assert v1_runner.OUTPUT!=schema.OUTPUT and v1_schema.OUTPUT!=schema.OUTPUT
    monkeypatch.setattr(predecessor,"authenticate_cpu_prerequisite",original)


def test_worker_scope_binds_v2_protocol_paths_argv_and_restores():
    old=(v1_worker.WORKER_PROTOCOL,v1_worker.expected_argv())
    with worker.v2_context():
        assert v1_worker.WORKER_PROTOCOL==schema.WORKER_PROTOCOL
        assert v1_worker.expected_argv()==("python","-B","-m",
            "gato_tiago.circular_portfolio_p5_v2_worker","--execute","--request",
            str(schema.worker_paths()["request"]),"--output",str(schema.worker_paths()["json"]))
    assert (v1_worker.WORKER_PROTOCOL,v1_worker.expected_argv())==old


def test_v2_worker_cli_delegates_full_v1_transaction_once_and_restores(monkeypatch):
    calls=[];old=(v1_worker.WORKER_PROTOCOL,v1_worker.expected_argv)
    def delegated(argv):
        calls.append((argv,v1_worker.WORKER_PROTOCOL,v1_worker.expected_argv()))
        return {"json_published":True,"npz_published":True}
    monkeypatch.setattr(v1_worker,"main",delegated)
    argv=["--execute","--request",str(schema.worker_paths()["request"]),
        "--output",str(schema.worker_paths()["json"])]
    assert worker.main(argv)=={"json_published":True,"npz_published":True}
    assert calls==[(argv,schema.WORKER_PROTOCOL,("python","-B","-m",
        "gato_tiago.circular_portfolio_p5_v2_worker","--execute","--request",
        str(schema.worker_paths()["request"]),"--output",str(schema.worker_paths()["json"])))]
    assert v1_worker.WORKER_PROTOCOL==old[0] and v1_worker.expected_argv is old[1]


def test_v2_worker_cli_failure_uses_v1_transaction_boundary_and_restores(monkeypatch):
    old=(v1_worker.WORKER_PROTOCOL,v1_worker.expected_argv);calls=[]
    def rejected(argv):
        calls.append((argv,v1_worker.WORKER_PROTOCOL));raise TimeoutError("published rejection")
    monkeypatch.setattr(v1_worker,"main",rejected)
    argv=["--execute","--request",str(schema.worker_paths()["request"]),
        "--output",str(schema.worker_paths()["json"])]
    with pytest.raises(TimeoutError,match="published rejection"):worker.main(argv)
    assert calls==[(argv,schema.WORKER_PROTOCOL)]
    assert v1_worker.WORKER_PROTOCOL==old[0] and v1_worker.expected_argv is old[1]


def test_gen1_preflight_rejects_before_publish_and_fallback_is_builtin(tmp_path,monkeypatch):
    monkeypatch.setattr(schema,"OUTPUT",tmp_path/"p5.json")
    observed=[]
    def original(*args,**kwargs):observed.append((args,kwargs))
    monkeypatch.setattr(v1_runner,"publish_checkpoint",original)
    with runner.v2_context():
        bad={"authentication":{"unsupported":object()}}
        with pytest.raises(TypeError):v1_runner.publish_checkpoint(
            1,"prerequisite_authenticated",v1_runner.initial_counts(),{},bad)
    assert observed==[]
    fallback=runner.publish_fallback(TypeError("primary"))
    assert json.loads(schema.OUTPUT.with_name("p5.rejected.json").read_text())==fallback
    assert runner.recertify_retained_failure()["passes"]
    bad=copy.deepcopy(fallback);bad["oracle_evidence"]=True
    schema.OUTPUT.with_name("p5.rejected.json").write_text(json.dumps(bad))
    assert not runner.recertify_retained_failure()["passes"]


def test_v2_science_constants_and_binary_are_exact_v1_parity():
    for name in ("KNOTS","INTERVALS","DENSE_SUBSTEPS","DENSE_SAMPLES","DT","KP","KD",
            "PILOT_IDENTITY","PILOT_ROUTES","LANES","ACTIVE_LANES","PILLAR_RADIUS",
            "CLEARANCE_MARGIN","TOLL_LENGTH","CAMPAIGN_WALL_LIMIT_S","WORKER_WALL_LIMIT_S",
            "BUILD_COMMAND","EXTENSION"):
        assert getattr(schema,name)==getattr(v1_schema,name)
    assert runner.AUTHORIZED_ORIG_ARGV==("python","-B","-m",
        "gato_tiago.circular_portfolio_p5_v2_runner","--execute","--output",str(schema.OUTPUT))
    assert runner.RUNNER_EXECUTION_AUTHORIZATION is None
    assert worker.WORKER_EXECUTION_AUTHORIZATION is None


def test_public_success_adapter_disk_roundtrip_canonicalizes_fresh_auth(tmp_path,monkeypatch):
    monkeypatch.setattr(schema,"OUTPUT",tmp_path/"p5.json")
    schema.OUTPUT.write_text(json.dumps({"authentication":{"gate":True}}))
    import gato_tiago.circular_portfolio_runner as predecessor
    monkeypatch.setattr(predecessor,"authenticate_cpu_prerequisite",
        lambda:({"gate":np.bool_(True)},{"array":np.asarray([1.])}))
    def disk_recert(path,**_kwargs):
        retained=json.loads(Path(path).read_text())
        fresh,_arrays=predecessor.authenticate_cpu_prerequisite()
        return {"passes":retained=={"authentication":fresh}}
    monkeypatch.setattr(v1_runner,"recertify_retained_p5",disk_recert)
    assert runner.recertify_retained_p5()["passes"]
    schema.OUTPUT.write_text(json.dumps({"authentication":{"gate":False}}))
    assert not runner.recertify_retained_p5()["passes"]
