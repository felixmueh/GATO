import copy
import hashlib
import inspect
from pathlib import Path
import re

import numpy as np
import pytest

from gato_tiago import circular_portfolio_constructor as p1_constructor
from gato_tiago import circular_portfolio_runner as p1_runner
from gato_tiago import circular_portfolio_worker as p1_worker
from gato_tiago import circular_portfolio_p2 as schema
from gato_tiago import circular_portfolio_p2_constructor as constructor
from gato_tiago import circular_portfolio_p2_preflight_runner as runner


@pytest.fixture(autouse=True)
def _stable_owned_repository_snapshot(monkeypatch):
    sources={name:hashlib.sha256(name.encode()).hexdigest() for name in runner.SOURCE_PATHS}
    monkeypatch.setattr(runner,"repository_snapshot",lambda:{"head":"1"*40,
        "tracked_clean":True,"source_hashes":sources})


def test_p1_closed_p2_static_tokens_and_exact_report_only_closure():
    assert p1_constructor.RUNNER_EXECUTION_AUTHORIZATION is None
    assert p1_runner.RUNNER_EXECUTION_AUTHORIZATION is None
    assert p1_worker.WORKER_EXECUTION_AUTHORIZATION is None
    assert constructor.CONSTRUCTOR_EXECUTION_AUTHORIZATION is not None
    assert runner.RUNNER_EXECUTION_AUTHORIZATION is not None
    report=schema.P1_REJECTED_REPORT
    assert report["exit_code"]==1 and report["completed"]==0 and report["pending"]==192
    assert report["identity"]==["development",12600,"short",0]
    assert report["stage"]=="profile_failed"
    assert report["campaign_elapsed_s"]==121.57387311197817
    assert report["rejected_p1_artifact_loads"]==0
    assert set(report["hashes"])=={"rejection_json","rejection_npz","latest",
        "gen0_json","gen0_npz","gen1_json","gen1_npz","gen2_json","gen2_npz",
        "gen3_json","gen3_npz"}
    assert all(len(value)==64 for value in report["hashes"].values())
    root=Path(__file__).resolve().parents[2]
    pattern=re.compile(r"^[A-Z][A-Z0-9_]*AUTHORIZATION = object\(\)$")
    enabled = {(path.name, line) for path in (root/"tiago_src/gato_tiago").glob("*.py")
               for line in path.read_text().splitlines() if pattern.fullmatch(line)}
    assert enabled == {
        ("circular_portfolio_p2_constructor.py",
         "CONSTRUCTOR_EXECUTION_AUTHORIZATION = object()"),
        ("circular_portfolio_p2_preflight_runner.py",
         "RUNNER_EXECUTION_AUTHORIZATION = object()"),
    }


def test_open_uniform_cubic_basis_and_affine_endpoint_map_are_exact():
    knots=schema.open_uniform_knots(); basis=schema.cubic_bspline_basis()
    assert knots.shape==(16,) and np.array_equal(knots[:4],np.zeros(4))
    assert np.array_equal(knots[-4:],np.ones(4))
    assert basis.shape==(95,12) and np.linalg.matrix_rank(basis)==12
    assert np.max(np.abs(np.sum(basis,axis=1)-1.0))<1e-14
    q0=np.linspace(-.2,.2,7); q_goal=q0+np.linspace(.01,.07,7)
    affine=schema.reduced_affine_map(q0,q_goal)
    detail=schema.certify_affine_map(affine,q0,q_goal)
    assert detail["passes"] and all(detail["gates"].values())
    assert affine["raw_map_float64"].shape==(665,84)
    assert np.linalg.matrix_rank(affine["raw_map_float64"])==84
    assert np.linalg.matrix_rank(affine["reduced_map_float64"])==70
    assert affine["independent_map_float64"].shape==(665,70)
    bad={name:value.copy() for name,value in affine.items()}
    bad["projection_float64"][0,0]+=1e-7
    assert not schema.certify_affine_map(bad,q0,q_goal)["passes"]


def test_every_reduced_vector_integrates_to_exact_terminal_state_and_mutates_fail():
    q0=np.linspace(-.1,.1,7); q_goal=q0+.02
    affine=schema.reduced_affine_map(q0,q_goal)
    coefficients=np.sin(np.arange(70,dtype=np.float64))
    detail=schema.certify_expansion(coefficients,affine,q0,q_goal)
    assert detail["passes"]
    assert np.max(np.abs(detail["q_float64"][-1]-q_goal))<=1e-12
    assert np.max(np.abs(detail["qd_float64"][-1]))<=1e-12
    bad={name:value.copy() for name,value in affine.items()}
    bad["particular_acceleration_float64"][0,0]+=1e-3
    assert not schema.certify_expansion(coefficients,bad,q0,q_goal)["passes"]


def test_deterministic_least_squares_initial_fit_and_reduced_derivative_chain():
    q0=np.zeros(7); q_goal=np.linspace(.01,.07,7)
    affine=schema.reduced_affine_map(q0,q_goal)
    proxy=np.cos(np.arange(665,dtype=np.float64)/31.).reshape(95,7)
    first=schema.least_squares_initial_coefficients(proxy,affine)
    second=schema.least_squares_initial_coefficients(proxy,affine)
    assert all(np.array_equal(first[name],second[name]) for name in first)
    assert first["least_squares_rank_int64"]==70
    coefficients=np.linspace(-.2,.2,70); direction=np.sin(np.arange(70))
    full=schema.expand_coefficients(coefficients,affine).ravel()
    gradient=2*full
    reduced=schema.chain_gradient(gradient,affine)
    h=1e-6
    fd=(np.sum(schema.expand_coefficients(coefficients+h*direction,affine)**2)
        -np.sum(schema.expand_coefficients(coefficients-h*direction,affine)**2))/(2*h)
    assert abs(fd-reduced@direction)<=1e-7*max(1.,abs(fd))
    jacobian=np.arange(5*665,dtype=np.float64).reshape(5,665)/1000.
    assert np.array_equal(schema.chain_jacobian(jacobian,affine),
                          jacobian@affine["independent_map_float64"])


def _synthetic_arrays():
    return {name:np.zeros(shape,dtype) for name,(shape,dtype) in runner.ARRAY_SPECS.items()}


def _replays(generation=3):
    return {"primary_pin_replay_calls":int(generation>=2),
        "independent_pin_recert_replay_calls":int(generation>=3),
        "total_pin_replay_calls":int(generation>=2)+int(generation>=3)}


def _provenance(final=False):
    sources={name:hashlib.sha256(name.encode()).hexdigest() for name in runner.SOURCE_PATHS}
    return {"cwd":runner.AUTHORIZED_CWD,"orig_argv":list(runner.AUTHORIZED_ORIG_ARGV),
        "exact_command":runner.shlex.join(runner.AUTHORIZED_ORIG_ARGV),
        "git_head_at_start":"1"*40,"git_head_at_end":"1"*40 if final else None,
        "tracked_clean_at_start":True,"tracked_clean_at_end":True if final else None,
        "source_hashes_at_start":sources,"source_hashes_at_end":sources if final else None,
        "thread_environment":runner.THREAD_ENV,"runtime_versions":runner.runtime_versions(),
        "prerequisite_pins":runner.p1_prerequisite_pins()}


def test_preflight_transaction_is_single_identity_pin_only_and_fail_closed(tmp_path):
    transaction=runner.transaction_schema()
    assert transaction["identity"]==["development",12600,"short",0]
    assert transaction["profile_attempts"]==transaction["optimizer_calls"]==1
    assert transaction["primary_pin_replay_calls"]==1
    assert transaction["independent_pin_recert_replay_calls"]==1
    assert transaction["total_pin_replay_calls"]==2
    assert transaction["prerequisite_artifact_loads"]==1
    assert transaction["prerequisite_independent_recert_calls"]==1
    assert all(transaction[name]==0 for name in ("p1_artifact_loads","cuda_calls",
        "worker_calls","task_rng_calls","task_construction_calls","sqp_calls","initializer_calls"))
    assert transaction["wall_limit_s"]==30.0 and not transaction["oracle_evidence"]
    for generation in (0,1):
        document=runner.checkpoint_document(generation,{},_provenance(),_replays(generation))
        assert runner.certify_checkpoint(document,{},generation)
    arrays=_synthetic_arrays()
    for generation in (2,3):
        document=runner.checkpoint_document(generation,arrays,_provenance(generation==3),
                                              _replays(generation))
        assert runner.certify_checkpoint(document,arrays,generation)
        bad=copy.deepcopy(document); bad["p1_rejected_report"]["completed"]=1
        assert not runner.certify_checkpoint(bad,arrays,generation)
    output=tmp_path/"preflight.json"; output.write_text("occupied")
    with pytest.raises(FileExistsError): runner.refuse_existing(output)
    with pytest.raises(RuntimeError,match="blocked"):
        runner.execute(tmp_path/"new.json",authorization=object())


def test_preflight_final_certificate_binds_arrays_timings_counts_and_mutations():
    phases={name:0.0 for name in runner.PHASE_KEYS}; phases["optimizer_s"]=1.0
    counts={name:0 for name in runner.EVALUATION_KEYS}; counts["objective"]=2
    arrays=_synthetic_arrays(); certificate={"passes":True,"gates":{"synthetic":True},
        "phase_timings_s":phases,"evaluation_counts":counts,
        "elapsed_s":2.0}
    summary={"protocol":schema.PROTOCOL,"incomplete":False,"overall_pass":True,
        "identity":list(schema.PRECHECK_IDENTITY),"array_names":sorted(arrays),
        "array_hashes":{name:runner.array_hash(arrays[name]) for name in sorted(arrays)},
        "certificate":certificate,"phase_timings_s":phases,
        "evaluation_counts":counts,"elapsed_s":2.0,"wall_limit_s":30.0,
        "checkpoint_count":4,"transaction":runner.transaction_schema(),
        "replay_call_counts":_replays(),
        "p1_rejected_report":schema.P1_REJECTED_REPORT,"oracle_evidence":False,
        "benchmark_evidence":False,"provenance":_provenance(True)}
    assert runner.certify_final(summary,arrays,certificate)["passes"]
    bad=copy.deepcopy(summary); bad["elapsed_s"]=30.0001
    assert not runner.certify_final(bad,arrays,certificate)["passes"]
    bad=copy.deepcopy(summary); bad["evaluation_counts"]["objective"]+=1
    assert not runner.certify_final(bad,arrays,certificate)["passes"]
    mutated={name:value.copy() for name,value in arrays.items()}
    mutated["coefficients_float64"][0]=1
    assert not runner.certify_final(summary,mutated,certificate)["passes"]
    bad=copy.deepcopy(summary); bad["phase_timings_s"]["optimizer_s"]=True
    assert not runner.certify_final(bad,arrays,certificate)["passes"]


def test_atomic_failure_document_is_permanent_honest_and_disk_recertified(tmp_path):
    phases={name:0.0 for name in runner.PHASE_KEYS}
    counts={name:0 for name in runner.EVALUATION_KEYS}
    document=runner.rejection_document("runtime_watchdog_rejected",
        TimeoutError("frozen deadline"),31.0,_provenance(True),phases,counts,(0,),
        _replays(0))
    assert runner.certify_rejection(document)
    output=tmp_path/"preflight.json"
    runner.publish_checkpoint(output,runner.checkpoint_document(
        0,{},_provenance(),_replays(0)),{})
    runner.publish_rejection(output,document)
    assert runner.recertify_retained_preflight(output)["passes"]
    assert not output.exists() and not output.with_suffix(".npz").exists()
    bad=copy.deepcopy(document); bad["completed"]=1
    assert not runner.certify_rejection(bad)
    bad=copy.deepcopy(document)
    bad["replay_call_counts"]={"primary_pin_replay_calls":0,
        "independent_pin_recert_replay_calls":1,"total_pin_replay_calls":1}
    assert not runner.certify_rejection(bad)
    rejected=output.with_name("preflight.rejected.json")
    rejected.write_text(runner.json.dumps(bad,sort_keys=True,separators=(",",":"))+"\n")
    pointer=output.with_name("preflight.partial.latest.json")
    pointer_value=runner.json.loads(pointer.read_text())
    pointer_value["json_sha256"]=runner._file_hash(rejected)
    pointer.write_text(runner.json.dumps(pointer_value,sort_keys=True,separators=(",",":"))+"\n")
    assert not runner.recertify_retained_preflight(output)["passes"]


def test_provenance_is_exact_and_rejects_head_source_and_environment_mutations():
    value=_provenance(True)
    assert runner.certify_provenance(value,True)
    assert "tiago_src/gato_tiago/circular_portfolio_runner.py" in runner.SOURCE_PATHS
    assert "gato/dynamics/tiago_right/tiago_right_arm.urdf" in runner.SOURCE_PATHS
    for source_path in runner.SOURCE_PATHS:
        bad=copy.deepcopy(value); bad["source_hashes_at_start"][source_path]="2"*64
        assert not runner.certify_provenance(bad,True)
    bad=copy.deepcopy(value); bad["git_head_at_start"]="2"*40
    assert not runner.certify_provenance(bad,True)
    bad=copy.deepcopy(value); bad["thread_environment"]["OMP_NUM_THREADS"]="2"
    assert not runner.certify_provenance(bad,True)


def test_operational_elapsed_is_bound_after_single_semantic_certificate():
    certificate={"gates":{"semantic":True,"elapsed":True},"elapsed_s":1.0,
                 "passes":True}
    assert runner.bind_operational_elapsed(certificate,30.0)["passes"]
    assert not runner.bind_operational_elapsed(certificate,30.000001)["passes"]
    source=inspect.getsource(runner.execute)
    semantic=source.index("certificate=certify_pin_preflight(")
    final_sample=source.index("elapsed=monotonic()-started",semantic)
    finish=source.index("finish_provenance(provenance)",final_sample)
    assert semantic<final_sample<finish
    assert source.count("certificate=certify_pin_preflight(")==1


def test_primary_and_independent_pin_replay_calls_are_measured_exactly():
    observed=[]
    def replay(x0,controls):
        observed.append((np.asarray(x0).copy(),np.asarray(controls).copy()))
        return np.zeros((2,14)),np.zeros((2,3))
    ledger=_replays(0); x0=np.zeros(14); controls=np.zeros((95,7))
    runner.measured_pin_replay(replay,ledger,"primary_pin_replay_calls",x0,controls)
    runner.measured_pin_replay(replay,ledger,"independent_pin_recert_replay_calls",x0,controls)
    assert len(observed)==2 and ledger==_replays(3)
    with pytest.raises(ValueError,match="unknown"):
        runner.measured_pin_replay(replay,ledger,"prerequisite_recert",x0,controls)
    source=inspect.getsource(runner.execute)
    assert source.count("measured_pin_replay(")==2


def test_scalar_temporal_basis_is_canonical_before_joint_kronecker_expansion():
    affine=schema.reduced_affine_map(np.zeros(7),np.ones(7)*.01)
    temporal=affine["temporal_independent_float64"]
    assert temporal.shape==(95,10)
    assert np.array_equal(affine["independent_map_float64"],np.kron(temporal,np.eye(7)))
    for column in range(10):
        pivot=np.argmax(np.abs(temporal[:,column]))
        assert temporal[pivot,column]>0
    source=inspect.getsource(schema.reduced_affine_map)
    assert "np.linalg.svd(\n        temporal_reduced" in source
    assert "np.linalg.svd(reduced" not in source


def test_preflight_production_source_has_real_pin_path_and_no_cuda_worker_or_retry():
    source=inspect.getsource(runner.execute)
    assert "authenticate_cpu_prerequisite" in source
    assert source.index("publish_checkpoint(output,checkpoint_document(\n            0") < source.index("authenticate_cpu_prerequisite()")
    assert "_production_pin_context" in source and "dense_replay" in source
    assert "execute_reduced_constructor(" in source
    assert "certify_pin_preflight(" in source
    assert source.index("started=monotonic()") < source.index("publish_checkpoint(output,checkpoint_document(\n            0")
    assert "publish_rejection(" in source and "runtime_watchdog_rejected" in source
    assert "_run_cuda_worker" not in source and "subprocess" not in source
    assert source.count("execute_reduced_constructor(")==1
    constructor_source=inspect.getsource(constructor.execute_reduced_constructor)
    assert "hess=BFGS()" in constructor_source and "method=\"trust-constr\"" in constructor_source
    assert "NonlinearConstraint" in constructor_source
    assert "retry" not in constructor_source.lower()
    adapter=schema.campaign_adapter_declaration()
    assert adapter["ledger_count"]==192 and adapter["attempts_per_identity"]==1
    assert adapter["raw_spline_coefficients"]==84
    assert adapter["independent_optimizer_variables"]==70
    assert adapter["changed_component"]=="acquisition_coordinate_parameterization_only"
    assert adapter["campaign_tokens_enabled"] is False
