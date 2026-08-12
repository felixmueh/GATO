import copy,inspect,json,re
from pathlib import Path

import numpy as np
import pytest

from gato_tiago import circular_portfolio_p5_v2_runner as p5_runner
from gato_tiago import circular_portfolio_p5_v2_worker as p5_worker
from gato_tiago import circular_portfolio_p6 as schema
from gato_tiago import circular_portfolio_p6_runner as runner
from gato_tiago import circular_portfolio_p6_worker as worker


def test_p6_isolated_closed_boundary_and_exact_rejected_lineage():
    assert p5_runner.RUNNER_EXECUTION_AUTHORIZATION is None
    assert p5_worker.WORKER_EXECUTION_AUTHORIZATION is None
    assert runner.RUNNER_EXECUTION_AUTHORIZATION is None
    assert worker.WORKER_EXECUTION_AUTHORIZATION is None
    root=Path(__file__).resolve().parents[2]
    pattern=re.compile(r"^[A-Z][A-Z0-9_]*AUTHORIZATION\s*=\s*object\(\)$")
    assert sorted((p.name,line) for p in (root/"tiago_src/gato_tiago").glob("*.py")
        for line in p.read_text().splitlines() if pattern.fullmatch(line))==[
            ("circular_portfolio_p9_runner.py","RUNNER_EXECUTION_AUTHORIZATION=object()"),
            ("circular_portfolio_p9_worker.py","WORKER_EXECUTION_AUTHORIZATION=object()")]
    report=schema.P5_V2_REJECTED_REPORT
    assert report["rejected_p5_v2_artifact_loads"]==0
    assert report["classification"]==["parent_child_worker_provenance_adapter_mismatch",
        "scientific_cuda_open_loop_infeasible"]
    assert report["hashes"]["route0_npz"] \
        =="64305385b598a2146a7841b2715680805957ca18e869f897e11c46c90540da7b"
    assert set(report["actual_worker_counts"])=={"b1_constructors","b16_constructors",
        "b1_sim_forward_calls","b16_sim_forward_calls","b1_tool_position_calls",
        "b16_tool_position_calls","solve_calls","sqp_calls"}
    assert schema.P5_V1_REJECTED_REPORT["rejected_p5_v1_artifact_loads"]==0
    assert schema.OUTPUT.parent.name=="tiago-tool-center-cuda-authoritative-route-seed-p6-authorized-once"
    assert schema.OUTPUT.parent.exists()
    assert runner.AUTHORIZED_ORIG_ARGV==("python","-B","-m",
        "gato_tiago.circular_portfolio_p6_runner","--execute","--output",str(schema.OUTPUT))
    assert worker.expected_argv()==("python","-B","-m",
        "gato_tiago.circular_portfolio_p6_worker","--execute","--request",
        str(schema.worker_paths()["request"]),"--output",str(schema.worker_paths()["json"]))


def test_fixed_gain_is_lowest_source_declared_passing_screen():
    assert (schema.KP,schema.KD)==(25.,10.)
    short=schema.SCREEN_RESULTS[25]["short"];long=schema.SCREEN_RESULTS[25]["long"]
    for route in (short,long):
        assert route["terminal_m"]<=.015 and route["speed_mps"]<=.05
        assert max(route[k] for k in ("q_ratio","v_ratio","u_ratio"))<=1
        assert route["clearance_m"]>=.005 and route["fk_max_m"]<=.001
    assert short["turn_rad"]<=-2.4 and long["turn_rad"]>=3.
    assert long["base_cost"]-short["base_cost"]>=.10
    assert short["full_cost"]-long["full_cost"]>=max(1.,.05*max(1.,abs(long["full_cost"])))
    design=schema.static_design()
    assert design["optimizer_calls"]==design["solve_calls"]==design["sqp_calls"]==0
    assert design["cuda_authoritative_dynamics"] and not design["pin_dynamic_replay"]
    assert design["pin_rnea_calls"]==518 and design["pin_rnea_quarantined_nominal_constructor"]
    assert not design["pin_rnea_feasibility_evidence"]
    source=inspect.getsource(runner.certify_campaign)
    assert ('>=.005' in source and '>=.10' in source
        and 'full_threshold=max(1.,.05*max(1.' in source
        and 'cuda_full_difference>=full_threshold' in source)


def test_feedback_is_actual_state_policy_and_rejects_malformed_metrics():
    proxy={"planned_q_float64":np.ones((schema.KNOTS,7)),
        "planned_qd_float64":np.full((schema.KNOTS,7),2.),
        "planned_qdd_float64":np.full((schema.INTERVALS,7),3.)}
    state=np.r_[np.full(7,.25),np.full(7,.5)]
    assert np.array_equal(schema.feedback_acceleration(proxy,0,state),
        np.full(7,3+25*.75+10*1.5))
    result=schema.authoritative_metrics(np.zeros((1,14)),np.zeros((1,3)),
        np.zeros((1,3)),np.zeros((schema.INTERVALS,7),np.float32),np.zeros(3),
        np.zeros(2),np.zeros(schema.KNOTS*10),np.zeros(7),np.ones(7),np.ones(7),
        np.ones(7),np.zeros((3,7)),"short",1,False)
    assert result=={"passes":False}


def test_cost_reconstruction_retains_exact_components_residual_and_identity():
    state=np.zeros((schema.KNOTS,14));controls=np.zeros((schema.INTERVALS,7))
    tool=np.tile(np.asarray([0.,0.,.5]),(schema.KNOTS,1))
    row=np.asarray([0.,0.,.5,2.,2.,.03,1.,0.,.01,.005],np.float32)
    reference=np.tile(row,schema.KNOTS)
    result=schema.reconstruct_cost_components(state,controls,tool,reference,
        np.full(7,-10.),np.full(7,10.),np.full(7,10.),np.full(7,100.))
    assert set(result["non_toll_components"])=={"tool_tracking","cylinder",
        "joint_velocity","position_barrier","velocity_barrier","control","control_barrier"}
    assert result["toll_residual_float64"].shape==(schema.KNOTS,)
    assert result["toll_contribution"]>=0
    assert result["full_cost"]==result["base_cost"]+result["toll_contribution"]


def test_worker_output_binds_b1_lane0_and_full_b16_distinct_streams():
    states=np.zeros((2,schema.KNOTS,14),np.float32);states[1,1:,0]=.01
    tools=np.zeros((2,schema.KNOTS,3),np.float32);tools[1,1:,0]=.01
    dense=np.zeros((2,schema.DENSE_SAMPLES,14),np.float32);dense[1,1:,0]=.01
    dense_tool=np.zeros((2,schema.DENSE_SAMPLES,3),np.float32);dense_tool[1,1:,0]=.01
    controls=np.zeros((2,schema.INTERVALS,7),np.float32);controls[1,:,0]=.02
    seeds=np.stack([schema.pack_seed(states[lane%2,:,:7],states[lane%2,:,7:],controls[lane%2])
        for lane in range(schema.LANES)])
    arrays={"captured_x0_float32":np.zeros(14,np.float32),
        "generated_knot_state_float32":states,"generated_knot_tool_float32":tools,
        "rnea_q_float64":states[:,:-1,:7].astype(np.float64),
        "rnea_qd_float64":states[:,:-1,7:].astype(np.float64),
        "rnea_qdd_float64":np.zeros((2,schema.INTERVALS,7),np.float64),
        "rnea_u_float64":controls.astype(np.float64),
        "recorded_controls_float32":controls,
        "independent_b1_knot_state_float32":states.copy(),
        "independent_b1_knot_tool_float32":tools.copy(),
        "b1_knot_transition_defect_float32":np.zeros((2,schema.INTERVALS,14),np.float32),
        "b1_dense_state_float32":dense,"b1_dense_tool_float32":dense_tool,
        "fresh_b16_knot_state_float32":np.stack([states[lane%2] for lane in range(schema.LANES)]),
        "fresh_b16_knot_tool_float32":np.stack([tools[lane%2] for lane in range(schema.LANES)]),
        "b16_knot_transition_defect_float32":np.zeros(
            (schema.LANES,schema.INTERVALS,14),np.float32),
        "fresh_b16_state_float32":np.stack([dense[lane%2] for lane in range(schema.LANES)]),
        "fresh_b16_tool_float32":np.stack([dense_tool[lane%2] for lane in range(schema.LANES)]),
        "b1_seed_xu_float32":seeds[0],"b16_seed_xu_float32":seeds}
    assert worker.certify_output(arrays)
    broken={**arrays,"fresh_b16_state_float32":arrays["fresh_b16_state_float32"].copy()}
    broken["fresh_b16_state_float32"][1]=broken["fresh_b16_state_float32"][0]
    assert not worker.certify_output(broken)
    wrong_lane={**arrays,"fresh_b16_knot_state_float32":
        arrays["fresh_b16_knot_state_float32"].copy()}
    wrong_lane["fresh_b16_knot_state_float32"][14]=states[1]
    assert not worker.certify_output(wrong_lane)
    defect={**arrays,"b1_knot_transition_defect_float32":
        arrays["b1_knot_transition_defect_float32"].copy()}
    defect["b1_knot_transition_defect_float32"][0,0,0]=np.finfo(np.float32).eps
    assert not worker.certify_output(defect)


def test_worker_source_uses_full_b16_results_and_no_optimizer():
    source=inspect.getsource(worker.generate)
    assert "fresh[:,index+1]=value" in source
    assert "fresh[:,index+1]=value[0]" not in source
    assert "feedback_acceleration(proxy,knot,states[route,knot])" in source
    assert "sim1(states[route,knot],controls[route,knot],DT)" in source
    assert "DT/DENSE_SUBSTEPS" in source
    assert ".solve(" not in source and "minimize(" not in source
    with pytest.raises(RuntimeError,match="blocked"):
        runner.execute(Path("/tmp/not-the-authorized-p6.json"),runner.RUNNER_EXECUTION_AUTHORIZATION)
    with pytest.raises(RuntimeError,match="blocked"):
        worker.execute(Path("/tmp/wrong.request.json"),worker.WORKER_EXECUTION_AUTHORIZATION)


def test_execution_ledgers_are_exact_bounded_and_causal():
    assert runner.certify_counts(runner.SUCCESS_COUNTS,True)
    for key in ("constructor_rnea_calls","b1_sim_forward_calls","b16_tool_position_calls"):
        bad=dict(runner.SUCCESS_COUNTS);bad[key]+=1
        assert not runner.certify_counts(bad,True)
    partial=runner.zero_counts();partial.update({"prerequisite_artifact_loads":1,
        "parent_pin_model_contexts":1,"worker_subprocess_attempts":1,
        "worker_pin_model_contexts":1,"total_pin_model_contexts":2,
        "parent_input_start_kinematics_calls":1,
        "parent_input_direction_kinematics_calls":1,
        "total_parent_input_kinematics_calls":2,"total_parent_kinematics_calls":2,
        "b1_constructors":1,"b16_constructors":1,"constructor_rnea_calls":17,
        "total_rnea_calls":17,"b1_sim_forward_calls":17,
        "b1_tool_position_calls":18})
    assert runner.certify_counts(partial)
    impossible=dict(partial);impossible["constructor_rnea_calls"]=10**12
    assert not runner.certify_counts(impossible)
    no_context=dict(partial);no_context["parent_pin_model_contexts"]=0
    assert not runner.certify_counts(no_context)
    attempted=runner.zero_counts();attempted.update({"prerequisite_artifact_loads":1,
        "parent_pin_model_contexts":1,"worker_subprocess_attempts":1,
        "parent_input_start_kinematics_calls":1,
        "parent_input_direction_kinematics_calls":1,
        "total_parent_input_kinematics_calls":2,"total_parent_kinematics_calls":2})
    unknown={"status":"exit0_internal_unknown","returncode":0,"counts":None,
        "killed":False,"internal_counts_known":False}
    assert runner.certify_worker_execution(unknown,attempted)
    assert worker.certify_counts(worker.SUCCESS_COUNTS,True)
    huge=dict(worker.SUCCESS_COUNTS);huge["b1_sim_forward_calls"]+=1
    assert not worker.certify_counts(huge)


def test_worker_input_measures_start_and_midpoint_kinematics_separately():
    calls=[]
    def kinematics(q):
        calls.append(np.asarray(q).copy());jacobian=np.zeros((3,7));jacobian[:3,:3]=np.eye(3)
        return np.asarray([q[0],q[1],.5]),jacobian
    q0=np.zeros(7);qgoal=np.zeros(7);qgoal[0]=.1
    prerequisite={"public_x0_float32":np.r_[q0,np.zeros(7)][None].astype(np.float32),
        "quarantined_q8_float64":qgoal[None],"q8_tool_float64":np.asarray([[.1,0.,.5]]),
        "public_default_side_int8":np.asarray([1],np.int8)}
    assert runner.worker_input(prerequisite,kinematics)["x0_float32"].shape==(14,)
    assert len(calls)==2 and np.array_equal(calls[0],q0)
    assert np.array_equal(calls[1],.5*(q0+qgoal))
    assert runner.SUCCESS_COUNTS["parent_input_start_kinematics_calls"]==1
    assert runner.SUCCESS_COUNTS["parent_input_direction_kinematics_calls"]==1
    assert runner.SUCCESS_COUNTS["total_parent_input_kinematics_calls"]==2
    assert runner.SUCCESS_COUNTS["total_parent_kinematics_calls"]==33676


def test_pointer_deadline_is_sampled_immediately_before_publish(monkeypatch):
    observed=[]
    monkeypatch.setattr(runner,"atomic_json",lambda path,value,pointer=False:
        observed.append((path,copy.deepcopy(value),pointer)))
    values=iter((599.0,599.5))
    pointer={"publication_finish_elapsed_s":None}
    runner.publish_final_pointer(pointer,0.,600.,lambda:next(values))
    assert observed[0][1]["publication_finish_elapsed_s"]==599.
    observed.clear();values=iter((599.9,600.1))
    with pytest.raises(TimeoutError):
        runner.publish_final_pointer({"publication_finish_elapsed_s":None},0.,600.,lambda:next(values))
    assert observed==[]


def test_cuda_only_cost_claim_and_quarantined_rnea_source_boundary():
    source=inspect.getsource(runner.certify_campaign)
    assert "pin_base_cost" not in source and "pin_full_cost" not in source
    assert "pin_base_shorter" not in source and "pin_full_reversal" not in source
    assert "cuda_base_shorter" in source and "cuda_full_reversal" in source
    rnea_source=inspect.getsource(runner.certify_quarantined_rnea)
    assert 'range(259)' in rnea_source and 'rnea_u_float64' in rnea_source
    assert schema.SOLVER_TRANSITION=={"solver_knot_dt":schema.DT,
        "dense_substeps":64,"dense_substep_dt":schema.DT/64,"held_control":True,
        "integrator_source":"gato/dynamics/integrator.cuh","integration":"trapezoidal"}


def test_atomic_input_publication_refuses_overwrite_and_cleans_failed_candidate(tmp_path,monkeypatch):
    target=tmp_path/"input.npz"
    runner.atomic_npz(target,{"x":np.asarray([1.],np.float64)})
    with pytest.raises(FileExistsError):runner.atomic_npz(target,{"x":np.asarray([2.])})
    failed=tmp_path/"failed.npz"
    original=np.savez
    def explode(*args,**kwargs):raise OSError("injected")
    monkeypatch.setattr(np,"savez",explode)
    with pytest.raises(OSError):runner.atomic_npz(failed,{"x":np.asarray([1.])})
    assert not failed.exists() and not Path(str(failed)+".candidate").exists()
    monkeypatch.setattr(np,"savez",original)


def _fake_provenance(snapshot,final):
    argv=(*runner.AUTHORIZED_ORIG_ARGV[:-1],str(runner.OUTPUT))
    return {"cwd":"/workspace/GATO","orig_argv":list(argv),
        "command":" ".join(argv),"head_start":snapshot["head"],
        "head_end":snapshot["head"] if final else None,"clean_start":True,
        "clean_end":True if final else None,"sources_start":snapshot["sources"],
        "sources_end":snapshot["sources"] if final else None,
        "extension_start":snapshot["extension"],
        "extension_end":snapshot["extension"] if final else None,
        "runtime_versions":{"python":runner.sys.version,"numpy":runner.np.__version__,
            "pinocchio":runner.importlib.metadata.version("pin")},
        "thread_environment":runner.THREAD_ENV,
        "p5_v1_report":runner.canonical_authentication(schema.P5_V1_REJECTED_REPORT),
        "p5_v2_report":runner.canonical_authentication(schema.P5_V2_REJECTED_REPORT)}


def test_real_failure_disk_roundtrip_and_refreshed_counter_mutation(tmp_path,monkeypatch):
    output=tmp_path/"p6.json";monkeypatch.setattr(schema,"OUTPUT",output)
    monkeypatch.setattr(runner,"OUTPUT",output)
    snap={"head":"a"*40,"clean":True,"sources":{"x":"b"*64},
        "extension":{"path":"/x","sha256":"c"*64,"size":1}}
    monkeypatch.setattr(runner,"snapshot",lambda:snap)
    output.parent.mkdir(exist_ok=True)
    provenance=_fake_provenance(snap,False);counts=runner.zero_counts()
    runner.publish_checkpoint(0,"generation_zero",counts,provenance,{})
    rejection={"protocol":schema.PROTOCOL,"stage":"pilot_failed","error_type":"RuntimeError",
        "error_message":"injected","incomplete":True,"completed":0,"pending":2,
        "counts":counts,"worker_execution":{"status":"not_started","returncode":None,
            "counts":None,"killed":False,"internal_counts_known":False},
        "provenance":provenance,"active_artifacts":runner.active_artifact_map(),
        "candidate_classifications":runner.classify_candidates(),
        "trigger_elapsed_s":.1,"cleanup_finish_elapsed_s":.2,"campaign_wall_limit_s":600.,
        "oracle_evidence":False,"benchmark_evidence":False,
        "sqp_evidence":False,"seed_evidence":False}
    runner.atomic_json(runner.rejection_path(),rejection)
    assert runner.recertify_retained_failure(output)["passes"]
    rejection["counts"]["constructor_rnea_calls"]=10**12
    runner.rejection_path().write_text(json.dumps(rejection,sort_keys=True,separators=(",",":")))
    assert not runner.recertify_retained_failure(output)["passes"]


def test_real_success_document_roundtrip_and_refreshed_checkpoint_mutation(tmp_path,monkeypatch):
    output=tmp_path/"p6.json";monkeypatch.setattr(schema,"OUTPUT",output)
    monkeypatch.setattr(runner,"OUTPUT",output);output.parent.mkdir(exist_ok=True)
    snap={"head":"a"*40,"clean":True,"sources":{"x":"b"*64},
        "extension":{"path":"/x","sha256":"c"*64,"size":1}}
    monkeypatch.setattr(runner,"snapshot",lambda:snap)
    arrays,inputs,prerequisite,kinematics=_pure_semantic_fixture()
    certificate=runner.canonical_authentication(runner.certify_campaign(arrays,prerequisite,
        None,kinematics,None,lambda _q,_v,acceleration:acceleration,inputs))
    assert certificate["passes"]
    auth={"passes":True,"fixture":"owned synthetic prerequisite"}
    monkeypatch.setattr(runner,"authenticate_cpu_prerequisite",lambda:(auth,prerequisite))
    monkeypatch.setattr(runner,"_production_pin_context",lambda:(None,kinematics,
        lambda _q,_v,acceleration:acceleration,None,None,None,None))
    paths=schema.worker_paths()
    request={"protocol":schema.WORKER_PROTOCOL,"identity":["development",12600],
        "input_path":str(paths["input"]),"input_sha256":None,
        "output_json_path":str(paths["json"]),"output_npz_path":str(paths["npz"]),
        "extension":schema.EXTENSION,"wall_limit_s":300.0}
    runner.atomic_npz(paths["input"],inputs);request["input_sha256"]=runner.sha(paths["input"])
    runner.atomic_json(paths["request"],request);runner.atomic_npz(paths["npz"],arrays)
    worker_counts={key:runner.SUCCESS_COUNTS[key] for key in ("b1_constructors",
        "b16_constructors","constructor_rnea_calls","b1_sim_forward_calls","b16_sim_forward_calls",
        "b1_tool_position_calls","b16_tool_position_calls","solve_calls","sqp_calls")}
    worker_counts["pin_model_contexts"]=worker_counts.pop("constructor_rnea_calls")*0+1
    worker_counts["rnea_calls"]=518
    worker_snap={"head":"d"*40,"clean":True,"sources":{"worker":"e"*64},
        "extension":{"path":"/synthetic/n260.so","sha256":schema.EXTENSION["sha256"],
            "size":schema.EXTENSION["size"],"build_head":schema.EXTENSION["build_head"],
            "arch":schema.EXTENSION["arch"]}}
    monkeypatch.setattr(worker,"snapshot",lambda:worker_snap)
    monkeypatch.setattr(runner,"worker_snapshot",lambda:worker_snap)
    worker_provenance={"cwd":"/workspace/GATO","orig_argv":list(worker.expected_argv()),
        "thread_environment":worker.THREAD_ENV,
        "runtime_versions":{"python":worker.sys.version,"numpy":worker.np.__version__,
            "pinocchio":worker.importlib.metadata.version("pin")},
        "start_snapshot":worker_snap,"end_snapshot":worker_snap}
    worker_summary={**worker_counts,"protocol":schema.WORKER_PROTOCOL,
        "identity":["development",12600],"request_path":str(paths["request"]),
        "request_sha256":runner.sha(paths["request"]),"input_path":str(paths["input"]),
        "input_sha256":runner.sha(paths["input"]),"npz_path":str(paths["npz"]),
        "npz_sha256":runner.sha(paths["npz"]),"array_names":sorted(arrays),
        "array_hashes":{key:schema.array_hash(arrays[key]) for key in sorted(arrays)},
        "elapsed_s":1.,"certificate":True,"phase_timings":{"constructor_rnea_elapsed_s":.1},
        "provenance":worker_provenance,"imported_extension":{"path":"/synthetic/n260.so",
            "sha256":schema.EXTENSION["sha256"],"size":schema.EXTENSION["size"]},
        "cuda_diagnostics":{"uuid":"GPU-fixture","name":"fixture","driver_version":"1",
            "compute_capability":"6.1","cuda_runtime_version":1,
            "cuda_driver_api_version":1}}
    runner.atomic_json(paths["json"],worker_summary)
    p0=_fake_provenance(snap,False);p3=_fake_provenance(snap,True)
    runner.publish_checkpoint(0,"generation_zero",runner.zero_counts(),p0,{})
    runner.publish_checkpoint(1,"prerequisite_authenticated",runner.generation_one_counts(),p0,
        {"authentication":auth})
    execution={"status":"known_complete","returncode":0,"counts":worker_counts,
        "killed":False,"internal_counts_known":True}
    post_worker=dict(runner.SUCCESS_COUNTS);post_worker.update({"completed":0,
        "independent_rnea_recert_calls":0,"pin_identical_state_fk_calls":0,
        "total_rnea_calls":518,"total_parent_kinematics_calls":2})
    rejection={"protocol":schema.PROTOCOL,"stage":"pilot_failed",
        "error_type":"RuntimeError","error_message":"post-worker injected",
        "incomplete":True,"completed":0,"pending":2,"counts":post_worker,
        "worker_execution":execution,"provenance":p0,
        "active_artifacts":runner.active_artifact_map(),
        "candidate_classifications":runner.classify_candidates(),
        "trigger_elapsed_s":1.,"cleanup_finish_elapsed_s":1.1,
        "campaign_wall_limit_s":600.,"oracle_evidence":False,
        "benchmark_evidence":False,"sqp_evidence":False,"seed_evidence":False}
    runner.atomic_json(runner.rejection_path(),rejection)
    assert runner.recertify_retained_failure(output)["passes"]
    runner.rejection_path().unlink()
    runner.publish_checkpoint(2,"routes_certified",runner.SUCCESS_COUNTS,p0,
        {"certificate":certificate})
    runner.publish_checkpoint(3,"honest_end",runner.SUCCESS_COUNTS,p3,
        {"certificate":certificate})
    for index,route in enumerate(("short","long")):
        runner.atomic_json(runner.route_path(index),{"protocol":schema.PROTOCOL,
            "identity":["development",12600,route,0],"worker_npz_path":str(paths["npz"]),
            "worker_npz_sha256":runner.sha(paths["npz"]),"route_index":index,
            "seed_certificate":certificate["seed_routes"][index],
            "dense_certificate":certificate["dense_physical_routes"][index],"passes":True})
    final={"protocol":schema.PROTOCOL,"incomplete":False,"completed":2,"pending":0,
        "counts":runner.SUCCESS_COUNTS,"authentication":auth,"worker":worker_summary,
        "worker_execution":execution,"certificate":certificate,"provenance":p3,
        "semantic_finish_elapsed_s":1.,
        "oracle_evidence":False,"benchmark_evidence":False,"sqp_evidence":False,
        "seed_evidence":True}
    runner.atomic_json(output,final)
    side=[runner.checkpoint_path(i) for i in range(4)]+[paths["request"],paths["input"],
        paths["json"],paths["npz"],runner.route_path(0),runner.route_path(1)]
    manifest={"protocol":schema.PROTOCOL,"json_path":str(output),"json_sha256":runner.sha(output),
        "worker_json_path":str(paths["json"]),"worker_json_sha256":runner.sha(paths["json"]),
        "worker_npz_path":str(paths["npz"]),"worker_npz_sha256":runner.sha(paths["npz"]),
        "side_artifacts":[{"path":str(p),"sha256":runner.sha(p),"size":p.stat().st_size}
            for p in side],"overall_pass":True}
    runner.atomic_json(runner.manifest_path(),manifest)
    runner.atomic_json(runner.latest_path(),{"protocol":schema.PROTOCOL,"incomplete":False,
        "json_path":str(output),"json_sha256":runner.sha(output),
        "manifest_path":str(runner.manifest_path()),
        "manifest_sha256":runner.sha(runner.manifest_path()),
        "publication_finish_elapsed_s":1.},True)
    assert runner.recertify_retained_p6(output)["passes"]
    checkpoint=json.loads(runner.checkpoint_path(2).read_text());checkpoint["counts"]["constructor_rnea_calls"]-=1
    runner.checkpoint_path(2).write_text(json.dumps(checkpoint,sort_keys=True,separators=(",",":")))
    manifest["side_artifacts"][2]["sha256"]=runner.sha(runner.checkpoint_path(2))
    manifest["side_artifacts"][2]["size"]=runner.checkpoint_path(2).stat().st_size
    runner.manifest_path().write_text(json.dumps(manifest,sort_keys=True,separators=(",",":")))
    pointer=json.loads(runner.latest_path().read_text())
    pointer["manifest_sha256"]=runner.sha(runner.manifest_path())
    runner.latest_path().write_text(json.dumps(pointer,sort_keys=True,separators=(",",":")))
    assert not runner.recertify_retained_p6(output)["passes"]


def _pure_semantic_fixture():
    radius=.05;start_angle=np.deg2rad(170.)
    def kinematics(q):
        q=np.asarray(q);progress=np.clip(q[0]/.1,0.,1.)
        sweep=-np.deg2rad(160.) if q[1]>=0 else np.deg2rad(200.)
        angle=start_angle+sweep*progress
        position=np.asarray([radius*np.cos(angle),radius*np.sin(angle),.5])
        jacobian=np.zeros((3,7));jacobian[:3,:3]=np.eye(3)
        return position,jacobian
    q0=np.zeros(7);qgoal=np.zeros(7);qgoal[0]=.1
    prerequisite={"public_x0_float32":np.r_[q0,np.zeros(7)][None].astype(np.float32),
        "quarantined_q8_float64":qgoal[None],"q8_tool_float64":kinematics(qgoal)[0][None],
        "public_default_side_int8":np.asarray([1],np.int8),
        "joint_lower_float64":np.full(7,-10.),"joint_upper_float64":np.full(7,10.),
        "velocity_limit_float64":np.full(7,10.),"effort_limit_float64":np.full(7,100.)}
    inputs=runner.worker_input(prerequisite,kinematics)
    knot_state=[];knot_tool=[];dense_state=[];dense_tool=[];controls=[];accelerations=[]
    for route in ("short","long"):
        q=inputs[f"{route}_proxy_q_float64"];qd=inputs[f"{route}_proxy_qd_float64"]
        state=np.c_[q,qd].astype(np.float32);knot_state.append(state)
        knot_tool.append(np.asarray([kinematics(row)[0] for row in state[:,:7]],np.float32))
        proxy={"planned_q_float64":q,"planned_qd_float64":qd,
            "planned_qdd_float64":inputs[f"{route}_proxy_qdd_float64"]}
        acceleration=np.asarray([schema.feedback_acceleration(proxy,k,state[k])
                                 for k in range(schema.INTERVALS)])
        accelerations.append(acceleration);controls.append(acceleration.astype(np.float32))
        dense=[]
        for knot in range(schema.INTERVALS):
            dense.extend((1-alpha/64.)*state[knot]+alpha/64.*state[knot+1]
                         for alpha in range(64))
        dense.append(state[-1]);dense=np.asarray(dense,np.float32);dense_state.append(dense)
        dense_tool.append(np.asarray([kinematics(row[:7])[0] for row in dense],np.float32))
    knot_state=np.asarray(knot_state);knot_tool=np.asarray(knot_tool)
    dense_state=np.asarray(dense_state);dense_tool=np.asarray(dense_tool)
    controls=np.asarray(controls);accelerations=np.asarray(accelerations)
    seeds=np.stack([schema.pack_seed(knot_state[lane%2,:,:7],knot_state[lane%2,:,7:],
        controls[lane%2]) for lane in range(schema.LANES)])
    arrays={"captured_x0_float32":knot_state[0,0].copy(),
        "generated_knot_state_float32":knot_state,
        "generated_knot_tool_float32":knot_tool,
        "rnea_q_float64":knot_state[:,:-1,:7].astype(np.float64),
        "rnea_qd_float64":knot_state[:,:-1,7:].astype(np.float64),
        "rnea_qdd_float64":accelerations,"rnea_u_float64":accelerations,
        "recorded_controls_float32":controls,
        "independent_b1_knot_state_float32":knot_state.copy(),
        "independent_b1_knot_tool_float32":knot_tool.copy(),
        "b1_knot_transition_defect_float32":np.zeros((2,schema.INTERVALS,14),np.float32),
        "b1_dense_state_float32":dense_state,"b1_dense_tool_float32":dense_tool,
        "fresh_b16_knot_state_float32":np.stack([knot_state[i%2] for i in range(schema.LANES)]),
        "fresh_b16_knot_tool_float32":np.stack([knot_tool[i%2] for i in range(schema.LANES)]),
        "b16_knot_transition_defect_float32":np.zeros((schema.LANES,schema.INTERVALS,14),np.float32),
        "fresh_b16_state_float32":np.stack([dense_state[i%2] for i in range(schema.LANES)]),
        "fresh_b16_tool_float32":np.stack([dense_tool[i%2] for i in range(schema.LANES)]),
        "b1_seed_xu_float32":seeds[0],"b16_seed_xu_float32":seeds}
    return arrays,inputs,prerequisite,kinematics


def test_real_pure_worker_and_campaign_semantics_reject_coherent_route_mutation():
    arrays,inputs,prerequisite,kinematics=_pure_semantic_fixture()
    assert worker.certify_output(arrays)
    certificate=runner.certify_campaign(arrays,prerequisite,None,kinematics,None,
        lambda _q,_v,acceleration:acceleration,inputs)
    assert certificate["passes"]
    assert all(row["passes"] for row in certificate["seed_routes"])
    assert all(row["passes"] for row in certificate["dense_physical_routes"])
    mutated={key:value.copy() for key,value in arrays.items()}
    mutated["generated_knot_tool_float32"][0,100,0]+=.002
    mutated["independent_b1_knot_tool_float32"][0,100,0]+=.002
    mutated["fresh_b16_knot_tool_float32"][::2,100,0]+=.002
    assert worker.certify_output(mutated)
    assert not runner.certify_campaign(mutated,prerequisite,None,kinematics,None,
        lambda _q,_v,acceleration:acceleration,inputs)["passes"]
