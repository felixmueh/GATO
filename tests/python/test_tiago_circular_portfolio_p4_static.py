import copy,inspect,json
from pathlib import Path

import numpy as np
import pytest

from gato_tiago import circular_portfolio_p4 as schema
from gato_tiago import circular_portfolio_p4_runner as runner
from gato_tiago import circular_portfolio_p4_worker as worker
from gato_tiago.circular_portfolio import validate_reference_row


def test_all_execution_capabilities_are_closed_and_p3_is_closed():
    assert runner.RUNNER_EXECUTION_AUTHORIZATION is None
    assert worker.WORKER_EXECUTION_AUTHORIZATION is None
    from gato_tiago import circular_portfolio_p3_runner,circular_portfolio_p3_constructor
    assert circular_portfolio_p3_runner.RUNNER_EXECUTION_AUTHORIZATION is None
    assert circular_portfolio_p3_constructor.CONSTRUCTOR_EXECUTION_AUTHORIZATION is None


def test_exact_p3_and_n96_pins_and_fresh_namespace():
    assert schema.P3_PINS=={"json":"5a63d1383f4de5d95b5d0af7556b99eb23f6b1f2b4ba5b0a7a7ee6fe5a69be47",
        "manifest":"9b6fb89fd8d741bad89fba283f24ef4cd498b57e625230f0a90d2836d6647880",
        "pointer":"77db605dfecbf1438cef67422cab8acb1093283cbf242b9b147003dddfc516f0",
        "gen194":"3d5e70d15aa2223e6c336efc98f449ced6a7614ee9e6c41253ea2a653b64fd25"}
    assert schema.EXTENSION["sha256"]=="b079410ade9e7de19ed3d4b7ed6f6ace27172cd442ccea0d5a46bb7277067a2f"
    assert schema.EXTENSION["size"]==6690480 and schema.EXTENSION["arch"]=="61-real"


def synthetic_inputs():
    x0=np.zeros((16,14),np.float32);controls=np.zeros((16,95,7),np.float32)
    seeds=np.zeros((16,2009),np.float32)
    return {"x0_float32":x0,"controls_float32":controls,
        "b1_seed_xu_float32":seeds[0].copy(),"b16_seed_xu_float32":seeds}


def test_worker_input_schema_binds_b1_lane0_bytes_dtype_shape_and_finite():
    value=synthetic_inputs();assert schema.input_schema(value)
    for name in value:
        bad=copy.deepcopy(value);bad[name]=bad[name].astype(np.float64)
        assert not schema.input_schema(bad)
    bad=synthetic_inputs();bad["b1_seed_xu_float32"][3]=1
    assert not schema.input_schema(bad)
    bad=synthetic_inputs();bad["controls_float32"][0,0,0]=np.nan
    assert not schema.input_schema(bad)


def test_worker_paths_are_task_identity_derived_and_wrong_paths_block(tmp_path):
    assert schema.worker_paths(0)["json"].name=="p4.worker.00.json"
    assert schema.worker_paths(11)["npz"].name=="p4.worker.11.npz"
    for bad in (-1,12,True):
        with pytest.raises(ValueError):schema.worker_paths(bad)
    with pytest.raises(RuntimeError,match="blocked"):
        worker.execute_worker(tmp_path/"request",tmp_path/"output")


def test_worker_source_is_exact_batch16_zero_sqp_replay():
    source=inspect.getsource(worker._execute_worker_body)
    assert source.count("module.BSQP_16_float(")==1
    assert "BSQP_1_float(" not in source
    assert source.count("solver.sim_forward(")==1 and source.count("solver.tool_position(")==1
    assert ".solve(" not in source and "solve_sqp(" not in source
    assert schema.SIM_FORWARD_CALLS==6080 and schema.TOOL_POSITION_CALLS==6081


def _lane(route="short"):
    angle=np.linspace(0,2.6 if route=="short" else -3.2,6081)
    tool=np.c_[.04*np.cos(angle),.04*np.sin(angle),np.zeros_like(angle)]
    state=np.zeros((6081,14));controls=np.zeros((95,7),np.float32)
    goal=tool[-1];pillar=np.zeros(2)
    row=validate_reference_row(np.asarray([*goal,*pillar,.03,1.,0.,.01,.005],np.float32))
    reference=np.tile(row,(96,1)).ravel()
    args=(state,tool,state.astype(np.float32),tool.astype(np.float32),
        controls,goal,pillar,reference,np.full(7,-10.),np.full(7,10.),np.full(7,10.),
        np.full(7,10.),lambda q,qd:np.zeros(3),route,-1)
    first=schema.certify_lane(*args,{"passes":True,"base_cost":0.,"full_cost":0.,"turn":0.})
    accepted={"passes":True,"base_cost":first["pin_base_cost"],
        "full_cost":first["pin_full_cost"],"turn":first["pin_turn"]}
    return schema.certify_lane(*args,accepted)


def test_lane_certificate_reconstructs_feasibility_cost_topology_and_model_agreement():
    lane=_lane();assert lane["passes"]
    assert lane["minimum_clearance_m"]>=.005 and lane["maximum_tool_disagreement_m"]<1e-6
    angle=np.linspace(0,2.6,6081);tool=np.c_[.04*np.cos(angle),.04*np.sin(angle),np.zeros_like(angle)]
    tool[100,0]+=.002
    state=np.zeros((6081,14));controls=np.zeros((95,7),np.float32);goal=tool[-1]
    row=validate_reference_row(np.asarray([*goal,0.,0.,.03,1.,0.,.01,.005],np.float32))
    pin_tool=np.c_[.04*np.cos(angle),.04*np.sin(angle),np.zeros_like(angle)]
    bad=schema.certify_lane(state,pin_tool,state.astype(np.float32),tool.astype(np.float32),
        controls,goal,np.zeros(2),
        np.tile(row,(96,1)).ravel(),np.full(7,-10.),np.full(7,10.),np.full(7,10.),
        np.full(7,10.),lambda q,qd:np.zeros(3),"short",-1,
        {"passes":True,"base_cost":0.,"full_cost":0.,"turn":0.})
    assert not bad["passes"] and not bad["gates"]["model_tool_agreement"]


def test_task_aggregate_requires_both_models_reversal_and_opposite_topology():
    lanes=[]
    for index in range(16):
        short=index<8
        lanes.append({"passes":True,"pin_base_cost":1. if short else 2.,
            "cuda_base_cost":1. if short else 2.,"pin_full_cost":2. if short else 1.,
            "cuda_full_cost":2. if short else 1.,"pin_turn":2.6 if short else -3.2,
            "cuda_turn":2.6 if short else -3.2})
    controls=np.stack([np.full((95,7),i*.001,np.float32) for i in range(16)])
    tools=np.empty((16,6081,3),np.float32)
    for i in range(16):tools[i]=np.asarray([(.002*i)+(0. if i<8 else .1),0.,0.],np.float32)
    assert schema.certify_task(lanes,controls,tools)["passes"]
    bad=copy.deepcopy(lanes);bad[8]["cuda_full_cost"]=3
    assert not schema.certify_task(bad,controls,tools)["passes"]


def test_checkpoint_exact_counts_artifacts_and_provenance(monkeypatch):
    monkeypatch.setattr(runner,"certify_provenance",lambda _v,final:True)
    provenance={};artifacts=[]
    gen0=runner.checkpoint(0,"gen0",0,artifacts,provenance)
    assert runner.certify_checkpoint(gen0)
    assert gen0["call_counts"]["p3_artifact_loads"]==0
    paths=schema.worker_paths(0)
    artifacts=[{"task_index":0,**{f"{k}_path":str(v) for k,v in paths.items()},
        **{f"{k}_sha256":"0"*64 for k in paths}}]
    auth={"passes":True,"pins":schema.P3_PINS,"recertification":{"passes":True}}
    row=runner.checkpoint(2,"task_completed",1,artifacts,provenance,auth)
    assert runner.certify_checkpoint(row)
    bad=copy.deepcopy(row);bad["call_counts"]["sim_forward_calls"]+=1
    assert not runner.certify_checkpoint(bad)
    bad=copy.deepcopy(row);bad["worker_artifacts"][0]["json_path"]="/tmp/wrong"
    assert not runner.certify_checkpoint(bad)


def test_p3_authentication_is_exact_not_passes_only_or_extensible():
    auth={"passes":True,"pins":schema.P3_PINS,"recertification":{"passes":True}}
    assert runner.certify_p3_authentication(auth)
    for mutation in ({**auth,"extra":1},{"passes":True,"pins":schema.P3_PINS},
                     {**auth,"recertification":{"passes":True,"extra":1}}):
        assert not runner.certify_p3_authentication(mutation)


def test_partial_failure_counts_allow_only_one_measured_active_worker():
    value=runner.checkpoint_counts(3,"task_completed")
    value["worker_subprocess_attempts"]+=1
    assert runner.certify_counts(value,3,"replay_failed",True)
    succeeded=copy.deepcopy(value);succeeded["worker_subprocess_successes"]+=1
    succeeded["b16_constructors"]+=1;succeeded["sim_forward_calls"]+=6080
    succeeded["tool_position_calls"]+=6081
    assert runner.certify_counts(succeeded,3,"replay_failed",True)
    bad=copy.deepcopy(succeeded);bad["sim_forward_calls"]+=1
    assert not runner.certify_counts(bad,3,"replay_failed",True)
    impossible=runner.new_counts();impossible["worker_subprocess_attempts"]=1
    assert not runner.certify_counts(impossible,0,"replay_failed",True)
    half=runner.new_counts();half.update({"p3_artifact_loads":1,"p3_disk_recertifications":1,
        "p3_independent_pin_replays":192})
    assert not runner.certify_counts(half,0,"replay_failed",True)


def test_rejection_document_is_fail_closed_and_non_evidence(monkeypatch):
    monkeypatch.setattr(runner,"certify_provenance",lambda _v,final:True)
    error=TimeoutError("deadline")
    value=runner.rejection_document("runtime_watchdog_rejected",0,[],[],{},
        runner.new_counts(),None,error,1,0.,0.,runner.active_artifact_map(0),
        runner.publication_artifact_map(schema.OUTPUT))
    assert runner.certify_rejection(value)
    for key in ("sqp_evidence","benchmark_evidence"):
        bad=copy.deepcopy(value);bad[key]=True;assert not runner.certify_rejection(bad)
    bad=copy.deepcopy(value);bad["call_counts"]["worker_subprocess_attempts"]=2
    assert not runner.certify_rejection(bad)
    killed=copy.deepcopy(value);killed["call_counts"].update({"p3_artifact_loads":1,
        "p3_disk_recertifications":1,"p3_independent_pin_replays":192,
        "prerequisite_artifact_loads":2,"pin_model_contexts":2,
        "worker_subprocess_attempts":1})
    killed["p3_authentication"]={"pins":schema.P3_PINS,
        "recertification":{"passes":True},"passes":True}
    killed["active_worker_execution"]={"task_index":0,"status":"killed_internal_unknown",
        "internal_counts_known":False,"constructor_calls_lower_bound":0,
        "sim_forward_calls_lower_bound":0,"tool_position_calls_lower_bound":0}
    assert runner.certify_rejection(killed)
    bad=copy.deepcopy(killed);bad["p3_authentication"]=None
    assert not runner.certify_rejection(bad)


def test_trigger_elapsed_is_bounded_while_cleanup_finish_may_overrun():
    value=runner.operational(runner.CAMPAIGN_WALL_LIMIT_S,
        runner.CAMPAIGN_WALL_LIMIT_S+17.)
    assert runner.certify_operational(value)
    bad=copy.deepcopy(value);bad["trigger_elapsed_s"]+=.1
    assert not runner.certify_operational(bad)
    bad=copy.deepcopy(value);bad["observed_finish_elapsed_s"]-=18
    assert not runner.certify_operational(bad)


def test_publication_is_candidate_first_pointer_last_and_late_failure_is_representable():
    source=inspect.getsource(runner.execute)
    assert source.index("write_json_candidate(output,final)")<source.index("os.replace(output_candidate,output)")
    assert source.index("os.replace(manifest_candidate,manifest)")<source.index(
        'atomic_json(output.with_name("p4.partial.latest.json")')
    assert 'failure_stage="publication_failed"' in source
    assert 'p4.rejected.latest.json' in source
    assert '"pointer_candidate"' in inspect.getsource(runner.publication_artifact_map)


def test_worker_timeout_watchdog_and_failure_retention_are_source_bound():
    source=inspect.getsource(runner.run_worker)
    assert "timeout=min(WORKER_WALL_LIMIT_S,remaining)" in source
    execute_source=inspect.getsource(runner.execute)
    assert "monotonic()>=deadline" in execute_source
    assert "rejection_document(" in execute_source and "p4.rejected.json" in execute_source
    worker_source=inspect.getsource(worker.execute_worker)
    assert "worker_failed" in worker_source and "provenance" in worker_source
    body=inspect.getsource(worker._execute_worker_body)
    assert 'counters["constructor_calls"]+=1' in body
    assert 'counters["sim_forward_calls"]+=1' in body
    assert 'counters["tool_position_calls"]+=1' in body
    assert "cuda_diagnostics()" in body and "operational(" in body
    assert "known_complete" in execute_source
    assert "error.worker_execution=known_complete" in execute_source


def test_post_worker_success_failure_requires_known_complete_execution_record(monkeypatch):
    monkeypatch.setattr(runner,"certify_provenance",lambda _v,final:True)
    counts=runner.checkpoint_counts(0,"p3_authenticated")
    counts.update({"prerequisite_artifact_loads":2,"pin_model_contexts":2,
        "worker_subprocess_attempts":1,"worker_subprocess_successes":1,
        "b16_constructors":1,"sim_forward_calls":6080,"tool_position_calls":6081})
    error=RuntimeError("semantic failure")
    error.worker_execution={"task_index":0,"status":"known_complete","internal_counts_known":True,
        "constructor_calls_lower_bound":1,"sim_forward_calls_lower_bound":6080,
        "tool_position_calls_lower_bound":6081}
    auth={"pins":schema.P3_PINS,"recertification":{"passes":True},"passes":True}
    value=runner.rejection_document("replay_failed",0,[],[],{},counts,auth,error,2,1.,1.,
        runner.active_artifact_map(0),runner.publication_artifact_map(schema.OUTPUT))
    assert runner.certify_rejection(value)
    bad=copy.deepcopy(value);bad["active_worker_execution"]=None
    assert not runner.certify_rejection(bad)
    for status in ("abnormal_exit_internal_unknown","exit0_internal_unknown"):
        unknown=copy.deepcopy(value)
        unknown["call_counts"]["worker_subprocess_successes"]=0
        unknown["call_counts"]["b16_constructors"]=0
        unknown["call_counts"]["sim_forward_calls"]=0
        unknown["call_counts"]["tool_position_calls"]=0
        unknown["active_worker_execution"]={"task_index":0,"status":status,
            "internal_counts_known":False,"constructor_calls_lower_bound":0,
            "sim_forward_calls_lower_bound":0,"tool_position_calls_lower_bound":0}
        assert runner.certify_rejection(unknown)


def test_worker_rejection_cuda_diagnostics_are_substantive():
    valid={"uuid":"GPU-x","name":"P100","driver_version":"999.1",
        "compute_capability":"6.1","cuda_runtime_version":12000,
        "cuda_driver_api_version":12000}
    assert worker.certify_cuda_diagnostics(valid)
    for key,value in (("compute_capability","8.0"),("uuid",""),
                      ("cuda_runtime_version",0)):
        bad=copy.deepcopy(valid);bad[key]=value
        assert not worker.certify_cuda_diagnostics(bad)


def test_exit_zero_parse_and_artifact_hash_failures_are_inside_known_transition():
    source=inspect.getsource(runner.run_worker)
    assert 'except Exception as source:' in source
    assert '"exit0_internal_unknown"' in source
    execute_source=inspect.getsource(runner.execute)
    assert execute_source.index('"request_sha256":sha(paths["request"])') \
        < execute_source.index("except Exception as error:")


def test_final_requires_exact_owned_semantic_task_documents(monkeypatch):
    monkeypatch.setattr(runner,"certify_provenance",lambda _v,final:True)
    source=inspect.getsource(runner.execute)
    assert "owned_semantic_recert(" in source
    assert 'owned["tasks"]!=tasks' in source and 'owned["worker_artifacts"]!=artifacts' in source


def test_runner_authenticates_p3_before_model_and_worker_and_never_solves():
    source=inspect.getsource(runner.execute)
    assert source.index("authenticate_p3()")<source.index("authenticate_cpu_prerequisite()")
    assert source.index("authenticate_p3()")<source.index("run_worker(")
    assert ".solve(" not in source and "solve_sqp(" not in source and "np.random" not in source
    assert "run_worker(task_index,inputs,deadline,monotonic,execution_counts)" in source
    assert "for task_index in range(12)" in source


def test_final_document_is_non_sqp_non_benchmark_evidence(monkeypatch):
    monkeypatch.setattr(runner,"certify_provenance",lambda _v,final:True)
    gates={key:True for key in ("finite","initial","q_limits","velocity_limits",
        "effort_limits","terminal","speed","clearance","model_tool_agreement",
        "turn_agreement","cost_agreement","accepted_p3_binding")}
    lane={"gates":gates,"speed_m_s":0.,"minimum_clearance_m":.01,
        "maximum_tool_disagreement_m":0.,"maximum_q_violation":0.,
        "maximum_velocity_ratio":0.,"maximum_effort_ratio":0.,"terminal_error_m":0.,
        "pin_base_cost":1.,"pin_full_cost":2.,"cuda_base_cost":1.,"cuda_full_cost":2.,
        "pin_turn":2.5,"cuda_turn":2.5,"passes":True}
    pair={"gates":{"pin_base":True,"cuda_base":True,"pin_reversal":True,
        "cuda_reversal":True,"topology":True},"passes":True}
    portfolio={key:True for key in ("unique_control_bytes","unique_cuda_path_bytes",
        "within_family_path_separation","cross_family_separation","cross_family_topology")}
    tasks=[]
    for i in range(12):tasks.append({"task_index":i,"identity":list(runner.TASK_IDENTITIES[i]),
        "lanes":[copy.deepcopy(lane) for _ in range(16)],"aggregate":{"pairs":[copy.deepcopy(pair)
        for _ in range(8)],"portfolio_gates":portfolio,"minimum_within_family_tool_rms_m":.001,
        "minimum_cross_family_max_separation_m":.075,"passes":True},"passes":True})
    artifacts=[{} for _ in range(12)]
    value={"protocol":schema.PROTOCOL,"incomplete":False,"overall_pass":True,
        "p3_authentication":{"passes":True,"pins":schema.P3_PINS,
            "recertification":{"passes":True}},"tasks":tasks,"worker_artifacts":artifacts,
        "owned_semantic_recertification":{"tasks":tasks,"worker_artifacts":artifacts,"passes":True},
        "call_counts":runner.checkpoint_counts(12,"honest_end"),
        "checkpoint_count":15,"provenance":{},"operational":runner.operational(1.),
        "sqp_evidence":False,"benchmark_evidence":False}
    assert runner.certify_final(value)
    bad=copy.deepcopy(value);bad["sqp_evidence"]=True
    assert not runner.certify_final(bad)
    bad=copy.deepcopy(value);bad["tasks"][0]["aggregate"]["pairs"][0]["gates"]["extra"]=True
    assert not runner.certify_final(bad)
    bad=copy.deepcopy(value);bad["tasks"][0]["aggregate"]["minimum_within_family_tool_rms_m"]=np.nan
    assert not runner.certify_final(bad)
    bad=copy.deepcopy(value);bad["owned_semantic_recertification"]["tasks"][0]["passes"]=False
    assert not runner.certify_final(bad)
