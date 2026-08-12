import copy,inspect

import numpy as np
import pytest

from gato_tiago import circular_portfolio_constructor as p1_constructor
from gato_tiago import circular_portfolio_runner as p1_runner
from gato_tiago import circular_portfolio_worker as p1_worker
from gato_tiago import circular_portfolio_p2_constructor as p2_constructor
from gato_tiago import circular_portfolio_p2_preflight_runner as p2_runner
from gato_tiago import circular_portfolio_p3 as schema
from gato_tiago import circular_portfolio_p3_constructor as constructor
from gato_tiago import circular_portfolio_p3_runner as runner


def test_all_predecessor_and_p3_capabilities_are_closed_and_reports_are_exact():
    assert all(token is None for token in (p1_constructor.RUNNER_EXECUTION_AUTHORIZATION,
        p1_runner.RUNNER_EXECUTION_AUTHORIZATION,p1_worker.WORKER_EXECUTION_AUTHORIZATION,
        p2_constructor.CONSTRUCTOR_EXECUTION_AUTHORIZATION,p2_runner.RUNNER_EXECUTION_AUTHORIZATION,
        constructor.CONSTRUCTOR_EXECUTION_AUTHORIZATION,runner.RUNNER_EXECUTION_AUTHORIZATION))
    report=schema.P2_REJECTED_REPORT
    assert report["stage"]=="runtime_watchdog_rejected" and report["completed"]==0
    assert report["elapsed_s"]==30.015096527989954
    assert report["evaluation_counts"]=={"objective":235,"gradient":235,
        "inequality":234,"inequality_jacobian":235,"kinematics":90866,
        "rnea":89205,"rnea_derivatives":89110,"aba":0}
    assert report["rejected_p2_artifact_loads"]==0
    assert all(len(value)==64 for value in report["hashes"].values())


def test_frozen_amplitude_gain_ledger_and_no_optimizer_contract():
    assert np.array_equal(schema.SHORT_AMPLITUDES,np.asarray([.15+.01*i for i in range(8)]))
    assert np.array_equal(schema.LONG_AMPLITUDES,np.asarray([.20+.01*i for i in range(8)]))
    assert schema.KP==400 and schema.KD==40 and schema.TOLL_LENGTH_M==.01
    assert len(runner.LEDGER)==192 and len(set(runner.LEDGER))==192
    transaction=runner.transaction_schema()
    assert transaction["computed_torque_rollouts"]==192
    assert transaction["immediate_independent_pin_replays"]==192
    assert transaction["owned_artifact_independent_pin_replays"]==192
    assert transaction["total_independent_pin_replays"]==384
    assert transaction["primary_rollout_aba_calls"]==192*6080
    assert transaction["immediate_recert_rollout_aba_calls"]==192*6080
    assert transaction["owned_artifact_recert_rollout_aba_calls"]==192*6080
    assert transaction["direction_fk_jacobian_calls"]==24
    assert transaction["immediate_semantic_certificate_fk_calls"]==192*97
    assert transaction["owned_artifact_semantic_certificate_fk_calls"]==192*97
    assert all(transaction[name]==0 for name in ("optimizer_calls","scipy_calls",
        "rng_calls","cuda_calls","worker_calls","sqp_calls","retries",
        "p1_artifact_loads","p2_artifact_loads"))


def test_endpoint_exact_scalar_bases_and_all_sixteen_planned_lanes():
    q0=np.linspace(-.2,.2,7);qg=q0+np.linspace(.01,.07,7)
    direction=np.arange(1.,8.);direction/=np.linalg.norm(direction)
    basis=schema.endpoint_exact_scalar_bases()
    assert np.max(np.abs(basis["scalar_endpoint_float64"]@
        basis["base_acceleration_float64"]-np.array([1.,0.])))<=1e-12
    assert np.max(np.abs(basis["scalar_endpoint_float64"]@
        basis["perturbation_acceleration_float64"]))<=1e-12
    for route in ("short","long"):
        for index in range(8):
            proxy=schema.planned_proxy(q0,qg,direction,route,index)
            assert schema.certify_planned_proxy(proxy,q0,qg,direction,route,index)["passes"]
            assert np.max(np.abs(proxy["planned_q_float64"][-1]-qg))<=1e-12
            assert np.max(np.abs(proxy["planned_qd_float64"][-1]))<=1e-12
    bad=schema.planned_proxy(q0,qg,direction,"short",0);bad["planned_q_float64"][1,0]+=1e-3
    assert not schema.certify_planned_proxy(bad,q0,qg,direction,"short",0)["passes"]


def test_direction_is_task_deterministic_dls_normalized_and_default_signed():
    q0=np.zeros(7);qg=np.ones(7)*.1;J=np.zeros((3,7));J[:3,:3]=np.eye(3)
    kinematics=lambda _q:(np.zeros(3),J)
    start=np.array([0.,0.,0.]);goal=np.array([.15,0.,0.])
    plus=schema.perturbation_direction(q0,qg,start,goal,kinematics,1)
    minus=schema.perturbation_direction(q0,qg,start,goal,kinematics,-1)
    assert np.array_equal(plus,-minus) and np.linalg.norm(plus)==pytest.approx(1.)
    assert np.sign((goal[0]-start[0])*(J@plus)[1])==1


def test_computed_torque_rollout_records_float32_and_measured_calls():
    q0=np.zeros(7);qg=np.ones(7)*.01;direction=np.ones(7)/np.sqrt(7)
    proxy=schema.planned_proxy(q0,qg,direction,"short",0)
    rnea=lambda q,qd,qdd:qdd
    aba=lambda q,qd,u:u
    kinematics=lambda q:(np.r_[q[:2],0.],np.zeros((3,7)))
    ticks=iter([0.0]*1000)
    row=constructor.computed_torque_rollout(proxy,q0,rnea,aba,kinematics,
        deadline=1.,monotonic=lambda:next(ticks))
    assert row["applied_controls_float32"].dtype==np.float32
    assert row["measured_call_counts"]=={"rnea_calls":95,"aba_calls":6080,"fk_calls":6081}
    ticks=iter([0.0]*1000)
    state,tool,counts=constructor.independent_replay(q0,row["applied_controls_float32"],
        aba,kinematics,deadline=1.,monotonic=lambda:next(ticks))
    assert np.array_equal(state,row["pin_dense_state_float64"])
    assert np.array_equal(tool,row["pin_dense_tool_float64"])
    assert counts=={"aba_calls":6080,"fk_calls":6081}


def test_geometry_pair_reversal_and_mutations_are_fail_closed():
    short_tool=np.zeros((6081,3));long_tool=np.zeros((6081,3))
    short_tool[:,0]=.05;long_tool[:,0]=-.05
    row_s={"pin_dense_tool_float64":short_tool};row_l={"pin_dense_tool_float64":long_tool}
    geometry=runner.geometry_from_lane_zero(row_s,row_l,np.array([.1,0.,0.]),1)
    assert geometry["passes"] and geometry["separation_m"]==pytest.approx(.1)
    residual=np.zeros(96);residual[1:10]=1
    short={"passes":True,"certificate":{"base_cost":1.,"full_cost":2.,
        "toll_contribution":1.,"tool_length_m":.1,"turn":2.5},
        "toll_residual_float64":residual}
    long={"passes":True,"certificate":{"base_cost":1.1,"full_cost":1.1,
        "toll_contribution":.005,"tool_length_m":.11,"turn":-3.1},
        "toll_residual_float64":np.zeros(96)}
    assert runner.pair_certificate(short,long)["passes"]
    bad=copy.deepcopy(short);bad["toll_residual_float64"][:]=0
    assert not runner.pair_certificate(bad,long)["passes"]


def _provenance(final=False):
    sources={name:"1"*64 for name in runner.SOURCE_PATHS}
    return {"cwd":runner.AUTHORIZED_CWD,"orig_argv":list(runner.AUTHORIZED_ORIG_ARGV),
        "command":runner.shlex.join(runner.AUTHORIZED_ORIG_ARGV),"head_start":"1"*40,
        "head_end":"1"*40 if final else None,"clean_start":True,
        "clean_end":True if final else None,"sources_start":sources,
        "sources_end":sources if final else None,"thread_environment":runner.THREAD_ENV,
        "runtime_versions":{"python":runner.sys.version,"numpy":np.__version__,
            "pinocchio":runner.importlib.metadata.version("pin")},
        "prerequisite_pins":runner.prerequisite_pins(),
        "benchmark_class":schema.BENCHMARK_CLASS,
        "task_geometry_derived_from_predeclared_canonical_lane0_paths_before_sqp":True,
        "solver_outcomes_accessed_for_task_geometry":False}


def test_transaction_checkpoints_are_exact_and_no_overwrite(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,"snapshot",lambda:{"head":"1"*40,"clean":True,
        "sources":{name:"1"*64 for name in runner.SOURCE_PATHS}})
    for generation,stage,completed,final in ((0,"gen0",0,False),
        (1,"prerequisite_authenticated",0,False),(2,"profile_completed",1,False),
        (193,"profile_completed",192,False),(194,"honest_end",192,True)):
        artifacts=[{"identity":list(runner.LEDGER[i]),"json_path":f"j{i}",
            "json_sha256":"1"*64,"npz_path":f"n{i}","npz_sha256":"2"*64}
            for i in range(completed)]
        document=runner.checkpoint_document(generation,stage,completed,artifacts,
            _provenance(final),runner.final_execution_counts() if final
            else runner.expected_checkpoint_counts(completed))
        assert runner.certify_checkpoint(document,final)
        bad=copy.deepcopy(document);bad["call_counts"]["optimizer_calls"]=1
        assert not runner.certify_checkpoint(bad,final)
    occupied=tmp_path/"p3.json";occupied.write_text("x")
    with pytest.raises(FileExistsError):runner.refuse_existing(occupied)


def test_public_failure_recert_rejects_refreshed_huge_counter_lie(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,"snapshot",lambda:{"head":"1"*40,"clean":True,
        "sources":{name:"1"*64 for name in runner.SOURCE_PATHS}})
    output=tmp_path/"p3.json";provenance=_provenance()
    runner.publish_checkpoint(output,runner.checkpoint_document(
        0,"gen0",0,[],provenance,runner.new_execution_counts()))
    final_provenance=_provenance(True)
    rejection={"protocol":schema.PROTOCOL,"stage":"screen_failed","incomplete":True,
        "completed":0,"pending":192,"error_type":"RuntimeError","error_message":"injected",
        "elapsed_s":.1,"checkpoint_count":1,"call_counts":runner.new_execution_counts(),
        "provenance":final_provenance,"p2_rejected_report":schema.P2_REJECTED_REPORT,
        "oracle_evidence":False,"benchmark_evidence":False}
    rejected=output.with_name("p3.rejected.json");runner._atomic_json(rejected,rejection)
    pointer=output.with_name("p3.partial.latest.json")
    runner._atomic_json(pointer,{"protocol":schema.PROTOCOL,"incomplete":True,
        "stage":"screen_failed","json_path":str(rejected),"json_sha256":runner.sha(rejected)})
    assert runner.recertify_retained_failure(output)["passes"]
    rejection["call_counts"]["primary_rnea_calls"]=10**12
    rejected.unlink();runner._atomic_json(rejected,rejection)
    pointer.unlink();runner._atomic_json(pointer,{"protocol":schema.PROTOCOL,"incomplete":True,
        "stage":"screen_failed","json_path":str(rejected),"json_sha256":runner.sha(rejected)})
    assert not runner.recertify_retained_failure(output)["passes"]


def test_final_document_binds_exact_192_ledger_96_pairs_geometry_and_provenance(monkeypatch):
    monkeypatch.setattr(runner,"snapshot",lambda:{"head":"1"*40,"clean":True,
        "sources":{name:"1"*64 for name in runner.SOURCE_PATHS}})
    rows=[{"identity":list(identity),"passes":True,"certificate":{"passes":True}}
          for identity in runner.LEDGER]
    pairs=[{"passes":True} for _ in range(96)]
    geometries=[{"passes":True} for _ in range(12)]
    document={"protocol":schema.PROTOCOL,"incomplete":False,"overall_pass":True,
        "rows":rows,"pair_certificates":pairs,
        "task_portfolio_certificates":[{"passes":True} for _ in range(12)],
        "owned_semantic_recertification":{"pair_certificates":pairs,
            "task_portfolio_certificates":[{"passes":True} for _ in range(12)],
            "gates":{"artifact_rows":True,"pairs":True,"portfolios":True},
            "passes":True},"task_geometries":geometries,
        "elapsed_s":100.,"transaction":runner.transaction_schema(),
        "execution_counts":runner.final_execution_counts(),
        "checkpoint_count":195,"provenance":_provenance(True),
        "p2_rejected_report":schema.P2_REJECTED_REPORT,"oracle_evidence":False,
        "benchmark_evidence":False}
    assert runner.certify_final_document(document)["passes"]
    for key in ("rows","pair_certificates","task_portfolio_certificates","task_geometries"):
        bad=copy.deepcopy(document);bad[key].pop()
        assert not runner.certify_final_document(bad)["passes"]
    bad=copy.deepcopy(document);bad["transaction"]["optimizer_calls"]=1
    assert not runner.certify_final_document(bad)["passes"]
    bad=copy.deepcopy(document);bad["pair_certificates"][0]["passes"]=False
    assert not runner.certify_final_document(bad)["passes"]


def test_all_family_distinctness_and_cross_topology_are_exactly_gated():
    rows=[];arrays=[]
    for index in range(16):
        route="short" if index<8 else "long"
        rows.append({"identity":["development",12600,route,index%8],
            "certificate":{"turn":3. if route=="short" else -3.2}})
        controls=np.full((95,7),index*.001,np.float32)
        tool=np.zeros((6081,3));tool[:,0]=index*.002
        if route=="long":tool[:,1]=.08
        arrays.append({"applied_controls_float32":controls,"pin_dense_tool_float64":tool})
    detail=runner.task_portfolio_certificate(rows,arrays)
    assert detail["passes"]
    bad=copy.deepcopy(arrays);bad[1]["applied_controls_float32"]=bad[0]["applied_controls_float32"]
    bad[1]["pin_dense_tool_float64"]=bad[0]["pin_dense_tool_float64"]
    assert not runner.task_portfolio_certificate(rows,bad)["passes"]
    bad=copy.deepcopy(arrays);bad[1]["pin_dense_tool_float64"]=(
        bad[0]["pin_dense_tool_float64"]+1e-5)
    assert not runner.task_portfolio_certificate(rows,bad)["passes"]


def test_measured_ledger_exactly_distinguishes_primary_immediate_and_owned_replays():
    prefix=runner.new_execution_counts()
    assert prefix["primary_rollouts"]==prefix["immediate_replays"]==0
    assert prefix["owned_replays"]==0
    final=runner.final_execution_counts()
    assert final["primary_aba_calls"]==192*6080
    assert final["immediate_aba_calls"]==192*6080
    assert final["owned_aba_calls"]==192*6080
    assert final["primary_fk_calls"]==192*6081
    assert final["immediate_fk_calls"]==192*6081
    assert final["owned_fk_calls"]==192*6081
    assert runner.certify_execution_counts(final,True,192)
    for completed in (0,1,8,9,16,17,192):
        expected=runner.expected_checkpoint_counts(completed)
        assert runner.certify_execution_counts(expected,False,completed,True)
    for key in ("primary_rnea_calls","primary_aba_calls","primary_fk_calls",
                "immediate_aba_calls","immediate_fk_calls",
                "immediate_semantic_fk_calls"):
        bad=runner.expected_checkpoint_counts(7);bad[key]=10**12
        assert not runner.certify_execution_counts(bad,False,7,False)


def test_measured_failure_counters_include_two_lane0_rollouts_before_geometry_acceptance():
    counts=runner.new_execution_counts();q0=np.zeros(7);qg=np.ones(7)*.01
    direction=np.ones(7)/np.sqrt(7)
    kinematics=lambda q:(np.r_[q[:2],0.],np.zeros((3,7)))
    rnea=lambda q,qd,qdd:qdd;aba=lambda q,qd,u:u
    for route in ("short","long"):
        proxy=schema.planned_proxy(q0,qg,direction,route,0)
        constructor.computed_torque_rollout(proxy,q0,rnea,aba,kinematics,
            deadline=1.,monotonic=lambda:0.,execution_counts=counts,phase="primary")
    # A geometry rejection here has completed no accepted profile, but the two
    # actual lane-zero rollouts and all their callbacks must remain counted.
    assert counts["primary_rollouts"]==2 and counts["primary_rnea_calls"]==190
    assert counts["primary_aba_calls"]==2*6080 and counts["primary_fk_calls"]==2*6081
    assert counts["immediate_replays"]==counts["owned_replays"]==0
    counts["primary_direction_calls"]=1
    assert runner.certify_execution_counts(counts,False,0,False)

    # The two-primary exception is local to the next task's lane-zero geometry.
    completed1=runner.expected_checkpoint_counts(1)
    impossible=copy.deepcopy(completed1)
    impossible["primary_rollouts"]=4
    impossible["primary_rnea_calls"]=4*95
    impossible["primary_aba_calls"]=4*6080
    impossible["primary_fk_calls"]=4*6081
    assert not runner.certify_execution_counts(impossible,False,1,False)
    completed8=runner.expected_checkpoint_counts(8)
    impossible=copy.deepcopy(completed8)
    impossible["primary_rollouts"]+=1
    impossible["primary_rnea_calls"]+=95
    impossible["primary_aba_calls"]+=6080
    impossible["primary_fk_calls"]+=6081
    assert not runner.certify_execution_counts(impossible,False,8,False)


def test_measured_mid_owned_failure_retains_partial_callback_counts():
    counts=runner.new_execution_counts();calls={"aba":0}
    def failing_aba(q,qd,u):
        calls["aba"]+=1
        if calls["aba"]==11:raise RuntimeError("injected owned replay failure")
        return u
    kinematics=lambda q:(np.r_[q[:2],0.],np.zeros((3,7)))
    with pytest.raises(RuntimeError,match="injected"):
        constructor.independent_replay(np.zeros(7),np.zeros((95,7),np.float32),
            failing_aba,kinematics,deadline=1.,monotonic=lambda:0.,
            execution_counts=counts,phase="owned")
    assert counts["owned_aba_calls"]==10 and counts["owned_fk_calls"]==11
    assert counts["owned_replays"]==0
    counts.update(runner.expected_checkpoint_counts(192))
    counts["owned_direction_calls"]=12;counts["owned_aba_calls"]=10
    counts["owned_fk_calls"]=11
    assert runner.certify_execution_counts(counts,False,192,False)
    bad=copy.deepcopy(counts);bad["owned_semantic_fk_calls"]=10**12
    assert not runner.certify_execution_counts(bad,False,192,False)


def test_owned_eager_direction_precompute_is_a_reachable_failure_boundary():
    counts=runner.expected_checkpoint_counts(192)
    counts["owned_direction_calls"]=12
    assert runner.certify_execution_counts(counts,False,192,False)
    bad=copy.deepcopy(counts);bad["owned_direction_calls"]=13
    assert not runner.certify_execution_counts(bad,False,192,False)
    # Owned artifact work cannot begin until all eager directions exist.
    bad=copy.deepcopy(counts);bad["owned_direction_calls"]=11
    bad["owned_fk_calls"]=1
    assert not runner.certify_execution_counts(bad,False,192,False)


def test_production_sources_have_no_optimizer_rng_retry_cuda_or_subprocess_worker():
    source=inspect.getsource(runner.execute)
    assert source.count("execute_direct_constructor(")==1
    assert "authenticate_cpu_prerequisite" in source and "_production_pin_context" in source
    assert "computed_torque_rollout(" in source and "certify_rollout(" in source
    assert "owned_campaign_semantic_recert(" in source
    owned_source=inspect.getsource(runner.owned_campaign_semantic_recert)
    assert "json.loads(expected_json.read_text())" in owned_source
    assert "np.load(expected_npz" in owned_source and "sha(expected_npz)" in owned_source
    assert "recertify_retained_p3" in inspect.getsource(runner)
    assert "publish_checkpoint(" in source and "certify_final_document(" in source
    assert "_atomic_json(output,final)" in source and "runtime_watchdog_rejected" in source
    assert all(token not in source for token in ("scipy","minimize(","np.random","_run_cuda_worker"))
    constructor_source=inspect.getsource(constructor)
    assert all(token not in constructor_source for token in ("scipy","minimize(","np.random","retry"))
    assert schema.BENCHMARK_CLASS=="constructed_opposite_side_route_portfolio"
    assert "circular_rms" not in constructor_source
    with pytest.raises(RuntimeError,match="blocked"):
        runner.execute(runner.OUTPUT,authorization=object())
