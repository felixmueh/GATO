import copy,inspect,re
from pathlib import Path

import pytest

from gato_tiago import circular_portfolio_p4_runner as v1_runner
from gato_tiago import circular_portfolio_p4_worker as v1_worker
from gato_tiago import circular_portfolio_p4_v2 as schema
from gato_tiago import circular_portfolio_p4_v2_runner as runner
from gato_tiago import circular_portfolio_p4_v2_worker as worker


def test_v1_closed_v2_tokens_none_and_fresh_root():
    assert v1_runner.RUNNER_EXECUTION_AUTHORIZATION is None
    assert v1_worker.WORKER_EXECUTION_AUTHORIZATION is None
    assert runner.RUNNER_EXECUTION_AUTHORIZATION is None
    assert worker.WORKER_EXECUTION_AUTHORIZATION is None
    assert schema.OUTPUT.parent.exists()
    root=Path(__file__).resolve().parents[2]
    pattern=re.compile(r"^[A-Z][A-Z0-9_]*AUTHORIZATION\s*=\s*object\(\)$")
    enabled={(path.name,line) for path in (root/"tiago_src/gato_tiago").glob("*.py")
        for line in path.read_text().splitlines() if pattern.fullmatch(line)}
    assert enabled==set()


def test_v1_launch_invalid_report_is_exact_and_downstream_zero():
    report=schema.P4_V1_REJECTED_REPORT
    assert report["classification"]=="launch_invalid_provenance_only"
    assert report["completed"]==0 and report["pending"]==12
    assert report["rejected_v1_artifact_loads"]==0
    assert all(value==0 for value in report["retained_counts"].values())
    assert report["hashes"]=={
        "gen0_json":"8d1913a2402b5a80ab434b6ce8c555c968df6aa363d69c068648df9ed4122b5e",
        "gen0_pointer":"99439e162bbbb7d823be7b1b894600d7f266b0d432d877bd17348326597842a0",
        "rejection_json":"96b9841a3d42606201463585a9f8b9116178de0e841655ad9da749275524d632",
        "rejection_pointer":"7e1242e1f4d7a636e157b81ade8097c2be55767952ae1fcb254756ff7650a49d"}


def test_current_p3_science_matches_producer_with_only_two_token_normalizations():
    detail=schema.scientific_source_parity()
    # P4 V2 is permanently closed.  Subsequent isolated P5 build wiring is an
    # intentional scientific-source change and must make live V2 auth fail.
    assert not schema.certify_scientific_source_parity(detail)
    normalized={path for path,row in detail["sources"].items()
        if row["authorization_normalization_allowed"]}
    assert normalized=={"tiago_src/gato_tiago/circular_portfolio_p3_runner.py",
        "tiago_src/gato_tiago/circular_portfolio_p3_constructor.py"}
    assert all(detail["sources"][path]["authorization_transition"]["passes"] for path in normalized)
    assert any(row["current_sha256"]!=row["producer_sha256"]
        for path,row in detail["sources"].items() if path not in normalized)
    bad=copy.deepcopy(detail);first=next(iter(bad["sources"]));bad["sources"][first]["current_sha256"]="0"*64
    assert not schema.certify_scientific_source_parity(bad)


def test_authorization_transition_is_exact_one_way_and_rejects_other_mutations():
    path="tiago_src/gato_tiago/circular_portfolio_p3_runner.py"
    producer=schema.producer_bytes(path)
    closed=producer.replace(b"RUNNER_EXECUTION_AUTHORIZATION=object()",
        b"RUNNER_EXECUTION_AUTHORIZATION=None")
    detail=schema.certify_authorization_transition(path,closed,producer)
    assert detail=={"path":path,"authorization_transition":True,
        "name":"RUNNER_EXECUTION_AUTHORIZATION",
        "from":"RUNNER_EXECUTION_AUTHORIZATION=object()",
        "to":"RUNNER_EXECUTION_AUTHORIZATION=None","producer_from_count":1,
        "producer_to_count":0,"current_from_count":0,"current_to_count":1,"passes":True}
    mutated=closed.replace(b"CAMPAIGN_WALL_LIMIT_S",b"CAMPAIGN_WALL_LIMIT_X",1)
    assert not schema.certify_authorization_transition(path,mutated,producer)["passes"]
    whitespace=closed.replace(b"RUNNER_EXECUTION_AUTHORIZATION=None",
        b"RUNNER_EXECUTION_AUTHORIZATION = None")
    assert not schema.certify_authorization_transition(path,whitespace,producer)["passes"]
    assert not schema.certify_authorization_transition(path,producer,closed)["passes"]
    duplicate=producer+ b"\nRUNNER_EXECUTION_AUTHORIZATION=object()\n"
    assert not schema.certify_authorization_transition(path,closed,duplicate)["passes"]


def test_historical_snapshot_is_exact_producer_git_object():
    snapshot=schema.historical_snapshot()
    assert schema.P3_PRODUCER_HEAD=="bc7c89da2ebdc3dcf38d38bf0f0380590a022641"
    assert re.fullmatch(r"[0-9a-f]{40}",schema.P3_PRODUCER_HEAD)
    assert snapshot["head"]=="bc7c89da2ebdc3dcf38d38bf0f0380590a022641"
    assert snapshot["clean"] is True and len(snapshot["sources"])==33
    assert all(len(value)==64 for value in snapshot["sources"].values())


def test_authentication_detail_is_exact_not_passes_only(monkeypatch):
    parity=schema.scientific_source_parity();snapshot=schema.historical_snapshot()
    value={"pins":schema.P3_PINS,"producer_snapshot":snapshot,
        "scientific_source_parity":parity,"retained_provenance_exact":True,
        "recertification":{"passes":True},"passes":True}
    assert not schema.certify_p3_authentication(value)
    for bad in ({"passes":True},{**value,"extra":1},{**value,"retained_provenance_exact":False}):
        assert not schema.certify_p3_authentication(bad)


def test_scoped_runner_and_worker_restore_every_v1_global_on_exception():
    runner_names=("OUTPUT","PROTOCOL","WORKER_PROTOCOL","worker_paths","authenticate_p3",
        "certify_p3_authentication","SOURCE_PATHS","AUTHORIZED_ORIG_ARGV","run_worker",
        "certify_output","certify_worker_rejection","start_provenance","certify_provenance")
    before={key:getattr(v1_runner,key) for key in runner_names}
    with pytest.raises(RuntimeError):
        with runner.v2_runner_scope():
            assert v1_runner.OUTPUT==schema.OUTPUT
            raise RuntimeError("injected")
    assert all(getattr(v1_runner,key) is value for key,value in before.items())
    worker_names=("WORKER_PROTOCOL","worker_paths","expected_argv","SOURCE_PATHS",
        "WORKER_EXECUTION_AUTHORIZATION")
    before={key:getattr(v1_worker,key) for key in worker_names}
    with pytest.raises(RuntimeError):
        with worker.v2_worker_scope():raise RuntimeError("injected")
    assert all(getattr(v1_worker,key) is value for key,value in before.items())


def test_v2_runner_authenticates_historical_p3_before_any_model_or_worker():
    source=inspect.getsource(v1_runner.execute)
    assert source.index("authenticate_p3()")<source.index("authenticate_cpu_prerequisite()")
    assert source.index("authenticate_p3()")<source.index("run_worker(")
    adapter=inspect.getsource(runner.v2_runner_scope)
    assert '"authenticate_p3":authenticate_p3_historical' in adapter
    assert '"run_worker":run_worker' in adapter


def test_v2_worker_and_runner_are_blocked_and_use_fresh_exact_commands(tmp_path):
    with pytest.raises(RuntimeError,match="blocked"):
        worker.execute_worker(tmp_path/"request",tmp_path/"output")
    with pytest.raises(RuntimeError,match="blocked"):
        runner.execute(schema.OUTPUT)
    assert runner.AUTHORIZED_ORIG_ARGV==("python","-B","-m",
        "gato_tiago.circular_portfolio_p4_v2_runner","--execute","--output",str(schema.OUTPUT))
    assert worker.expected_argv(0)[3]=="gato_tiago.circular_portfolio_p4_v2_worker"


def test_v2_science_delegates_exact_v1_pipeline_without_optimizer_or_sqp():
    source=inspect.getsource(runner.execute)+inspect.getsource(runner.v2_runner_scope)
    assert "_v1.execute" in source and "solve(" not in source and "np.random" not in source
    worker_source=inspect.getsource(runner.run_worker)
    assert ".sim_forward(" not in worker_source and ".solve(" not in worker_source
    assert "circular_portfolio_p4_v2_worker" in worker_source
