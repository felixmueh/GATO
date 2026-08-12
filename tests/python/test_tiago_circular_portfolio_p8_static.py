import inspect,json,re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gato_tiago import circular_portfolio_p6_runner as p6_runner
from gato_tiago import circular_portfolio_p6_worker as p6_worker
from gato_tiago import circular_portfolio_p8 as schema
from gato_tiago import circular_portfolio_p8_runner as runner
from gato_tiago import circular_portfolio_p8_worker as worker
from gato_tiago import circular_portfolio_runner as base_runner


def test_p8_exact_frozen_scope_tokens_and_reports():
    assert p6_runner.RUNNER_EXECUTION_AUTHORIZATION is None
    assert p6_worker.WORKER_EXECUTION_AUTHORIZATION is None
    assert schema.RUNNER_EXECUTION_AUTHORIZATION is None
    assert schema.WORKER_EXECUTION_AUTHORIZATION is None
    assert runner.RUNNER_EXECUTION_AUTHORIZATION is None
    assert worker.WORKER_EXECUTION_AUTHORIZATION is None
    root=Path(__file__).resolve().parents[2]
    pattern=re.compile(r"^[A-Z][A-Z0-9_]*AUTHORIZATION\s*=\s*object\(\)$")
    assert sorted((path.name,line) for path in (root/"tiago_src/gato_tiago").glob("*.py")
        for line in path.read_text().splitlines() if pattern.fullmatch(line))==[
            ("circular_portfolio_p9_runner.py","RUNNER_EXECUTION_AUTHORIZATION=object()"),
            ("circular_portfolio_p9_worker.py","WORKER_EXECUTION_AUTHORIZATION=object()")]
    design=schema.static_design()
    assert (design["ik_iterations"],design["dls_damping"],design["kp"],design["kd"]) \
        ==(8,.03,25.,10.)
    assert (design["knots"],design["dt"])==(260,.05)
    assert design["horizon_s"]==pytest.approx(12.95)
    assert design["attempts"]==2 and design["retries"]==0
    assert design["optimizer_calls"]==design["solve_calls"]==design["sqp_calls"]==0
    assert len(schema.P7_EXPLORATORY_GRID)==18
    assert sum(row["overall_pass"] for row in schema.P7_EXPLORATORY_GRID)==0
    assert schema.P6_REJECTED_REPORT["rejected_p6_artifact_loads"]==0
    assert schema.P6_REJECTED_REPORT["hashes"]["worker_npz"] \
        =="49f784d37cbff1d124d9eda30c1ea9f90c176027dab17d46686fe7e33a962a51"
    assert schema.P6_REJECTED_REPORT["classification"]=="scientific_full_dt_and_dense_infeasible"
    assert runner.AUTHORIZED_ORIG_ARGV==("python","-B","-m",
        "gato_tiago.circular_portfolio_p8_runner","--execute","--output",str(schema.OUTPUT))
    paths=schema.worker_paths()
    assert worker.expected_argv()==("python","-B","-m","gato_tiago.circular_portfolio_p8_worker",
        "--execute","--request",str(paths["request"]),"--output",str(paths["json"]))
    assert not runner.refuse_existing()
    assert "if OUTPUT.parent.exists()" in inspect.getsource(runner.execute)
    assert 'Path(str(path)+".candidate").exists()' in inspect.getsource(worker.execute)


def test_p8_worker_constructor_uses_dt_point05_and_full_lane_blocks():
    args=worker.constructor_args();assert args[0]==.05
    source=inspect.getsource(worker.generate)
    assert "0 if lane<8 else 1" in source
    assert "sim1(state[route,k],u[route,k])" in source
    assert "sim16(state16[:,k],lane_u[:,k])" in source
    assert "AFFINE_SUBSTEPS" in source and "sim_forward" not in source.split("affine=np.empty",1)[1]
    with pytest.raises(RuntimeError,match="blocked"):
        runner.execute(Path("/tmp/wrong.json"),runner.RUNNER_EXECUTION_AUTHORIZATION)
    with pytest.raises(RuntimeError,match="blocked"):
        worker.execute(Path("/tmp/wrong.request.json"),worker.WORKER_EXECUTION_AUTHORIZATION)


def test_worker_wrong_resolved_module_fails_before_pin_or_cuda(tmp_path,monkeypatch):
    output=tmp_path/"p8.json";monkeypatch.setattr(schema,"OUTPUT",output)
    paths=schema.worker_paths();output.parent.mkdir(exist_ok=True)
    construction={key:np.zeros(shape,dtype) for key,(shape,dtype) in schema.CONSTRUCTION_SPECS.items()}
    with paths["input"].open("wb") as stream:np.savez(stream,**construction)
    request={"protocol":schema.WORKER_PROTOCOL,"identity":["development",12600],
        "input_path":str(paths["input"]),"input_sha256":worker.sha(paths["input"]),
        "output_json_path":str(paths["json"]),"output_npz_path":str(paths["npz"]),
        "extension":schema.EXTENSION,"wall_limit_s":300.}
    paths["request"].write_text(json.dumps(request,sort_keys=True,separators=(",",":")))
    snap={"head":"a"*40,"clean":True,"sources":{"x":"b"*64},
        "extension":worker.frozen_extension()}
    monkeypatch.setattr(worker,"snapshot",lambda:snap)
    monkeypatch.setattr(worker.sys,"orig_argv",list(worker.expected_argv()))
    bad=tmp_path/"wrong.so";bad.write_bytes(b"wrong")
    monkeypatch.setattr(worker.importlib,"import_module",lambda name:SimpleNamespace(__file__=bad))
    calls={"pin":0,"cuda":0}
    monkeypatch.setattr(base_runner,"_production_pin_context",lambda: calls.__setitem__("pin",1))
    monkeypatch.setattr(worker,"cuda_diagnostics",lambda module:calls.__setitem__("cuda",1))
    token=object();monkeypatch.setattr(worker,"WORKER_EXECUTION_AUTHORIZATION",token)
    with pytest.raises(RuntimeError,match="resolved extension"):
        worker.execute(paths["request"],token,lambda:0.)
    assert calls=={"pin":0,"cuda":0}


def _worker_arrays():
    state=np.zeros((2,schema.KNOTS,14),np.float32);state[1,1:,0]=.01
    tool=np.zeros((2,schema.KNOTS,3),np.float32);tool[1,1:,0]=.01
    controls=np.zeros((2,schema.INTERVALS,7),np.float32);controls[1,:,0]=.01
    lane_state=np.stack([state[0 if lane<8 else 1] for lane in range(schema.LANES)])
    lane_tool=np.stack([tool[0 if lane<8 else 1] for lane in range(schema.LANES)])
    seeds=np.stack([schema.pack_seed(state[0 if lane<8 else 1,:,:7],
        state[0 if lane<8 else 1,:,7:],controls[0 if lane<8 else 1])
        for lane in range(schema.LANES)])
    return {"captured_x0_float32":np.zeros(14,np.float32),
        "generated_state_float32":state,"generated_tool_float32":tool,
        "rnea_q_float64":state[:,:-1,:7].astype(np.float64),
        "rnea_qd_float64":state[:,:-1,7:].astype(np.float64),
        "rnea_qdd_float64":np.zeros((2,schema.INTERVALS,7)),
        "rnea_u_float64":controls.astype(np.float64),"controls_float32":controls,
        "fresh_b1_state_float32":state.copy(),"fresh_b1_tool_float32":tool.copy(),
        "b1_defect_float32":np.zeros((2,schema.INTERVALS,14),np.float32),
        "fresh_b16_state_float32":lane_state,"fresh_b16_tool_float32":lane_tool,
        "b16_defect_float32":np.zeros((schema.LANES,schema.INTERVALS,14),np.float32),
        "b1_seed_float32":seeds[0],"b16_seed_float32":seeds,
        "affine_cuda_tool_float32":np.zeros((2,schema.AFFINE_SAMPLES,3),np.float32)}


def _construction_for(arrays):
    result={key:np.zeros(shape,dtype) for key,(shape,dtype) in schema.CONSTRUCTION_SPECS.items()}
    result["proxy_q_float64"][:]=arrays["generated_state_float32"][:,:,:7]
    result["proxy_qd_float64"][:]=arrays["generated_state_float32"][:,:,7:]
    return result


def test_worker_output_exact_mapping_bytes_and_mutations():
    arrays=_worker_arrays();construction=_construction_for(arrays)
    assert worker.certify_output(arrays,construction)
    broken={**arrays,"fresh_b16_state_float32":arrays["fresh_b16_state_float32"].copy()}
    broken["fresh_b16_state_float32"][8]=broken["fresh_b16_state_float32"][0]
    assert not worker.certify_output(broken,construction)
    defect={**arrays,"b1_defect_float32":arrays["b1_defect_float32"].copy()}
    defect["b1_defect_float32"][0,0,0]=np.finfo(np.float32).eps
    assert not worker.certify_output(defect,construction)
    controller={**arrays,"rnea_qdd_float64":arrays["rnea_qdd_float64"].copy()}
    controller["rnea_qdd_float64"][0,0,0]=1.
    assert not worker.certify_output(controller,construction)
    wrong_start={**arrays,"captured_x0_float32":arrays["captured_x0_float32"].copy()}
    wrong_start["captured_x0_float32"][0]=1.
    assert not worker.certify_output(wrong_start,construction)


def test_cost_reconstruction_terminal_toll_off_and_components():
    state=np.zeros((schema.KNOTS,14));u=np.zeros((schema.INTERVALS,7))
    tool=np.tile(np.asarray([0.,0.,.5]),(schema.KNOTS,1))
    row=np.asarray([0.,0.,.5,2.,2.,.03,1.,0.,.01,.005],np.float32)
    result=schema.reconstruct_cost(state,u,tool,np.tile(row,schema.KNOTS),
        np.full(7,-10.),np.full(7,10.),np.full(7,10.),np.full(7,100.))
    assert set(result)=={"components","base","toll","full","residual_float64"}
    assert result["residual_float64"][-1]==0 and result["full"]==result["base"]+result["toll"]


def test_only_time_scaling_changes_p7_proxy_derivatives():
    q=np.zeros((2,schema.KNOTS,7));q[:,:,0]=np.linspace(0.,1.,schema.KNOTS)
    p8_v,p8_a=runner.time_scale_proxy(q,.05)
    p7_v,p7_a=runner.time_scale_proxy(q,.0125)
    assert np.array_equal(p8_v[:,0],np.zeros((2,7)))
    assert np.array_equal(p8_v[:,-1],np.zeros((2,7)))
    assert np.allclose(p8_v,.25*p7_v,rtol=0,atol=1e-15)
    assert np.allclose(p8_a,.0625*p7_a,rtol=0,atol=1e-13)


def test_exact_counter_schema_and_causal_bounds():
    final=runner.success_counts();assert runner.certify_counts(final,True)
    bad=dict(final);bad["worker_rnea_calls"]+=1
    assert not runner.certify_counts(bad)
    assert final["primary_dls_kinematics_calls"]==2*258*8
    assert final["independent_dls_kinematics_calls"]==2*258*8
    assert final["parent_affine_fk_calls"]==2*(259*16+1)
    assert worker.certify_counts(worker.SUCCESS_COUNTS,True)
    partial=worker.initial_counts();partial.update(pin_model_contexts=1,b1_constructors=1,
        b16_constructors=1,b16_sim_forward_calls=1)
    assert not worker.certify_counts(partial,False)


def test_parent_science_timeout_retains_exact_causal_prefix():
    arrays=_worker_arrays();construction=_construction_for(arrays)
    counts=runner.cpu_counts();counts.update(worker_attempts=1,worker_successes=1,
        worker_pin_contexts=1,b1_constructors=1,b16_constructors=1,worker_rnea_calls=518,
        b1_sim_forward_calls=1036,b16_sim_forward_calls=259,b1_tool_position_calls=9329,
        b16_tool_position_calls=260)
    times=iter((0.,0.,2.))
    prerequisite={"joint_lower_float64":np.full(7,-10.),"joint_upper_float64":np.full(7,10.),
        "velocity_limit_float64":np.full(7,10.),"effort_limit_float64":np.full(7,10.),
        "public_default_side_int8":np.asarray([1],np.int8)}
    with pytest.raises(TimeoutError,match="campaign wall"):
        runner.certify_science(arrays,construction,prerequisite,
            lambda q:(np.zeros(3),np.zeros((3,7))),lambda q,v,a:a,
            deadline=1.,monotonic=lambda:next(times),counts=counts)
    assert counts["science_routes_attempted"]==1
    assert counts["parent_identical_state_fk_calls"]==1
    assert counts["science_routes_completed"]==counts["completed"]==0
    assert runner.certify_counts(counts)


def test_cpu_constructor_exact_history_with_pure_dls_stub(monkeypatch):
    calls=[]
    def kin(q):
        calls.append(np.asarray(q).copy());J=np.zeros((3,7));J[:3,:3]=np.eye(3)
        return np.asarray([q[0],q[1],.5]),J
    def dls(J,res,q,lower,upper):return np.r_[res[:2],np.zeros(5)],np.zeros(7,np.int8)
    monkeypatch.setattr(runner,"exact_box_dls",dls)
    q0=np.zeros(7);qg=np.zeros(7);qg[0]=.08
    pre={"public_x0_float32":np.r_[q0,np.zeros(7)][None].astype(np.float32),
        "quarantined_q8_float64":qg[None],"public_default_side_int8":np.asarray([1],np.int8),
        "joint_lower_float64":np.full(7,-10.),"joint_upper_float64":np.full(7,10.)}
    arrays,detail=runner.construct_cpu(pre,kin,float("inf"))
    certificate=runner.certify_construction(arrays,pre,kin)
    assert schema.exact_arrays(arrays,schema.CONSTRUCTION_SPECS)
    json.dumps(certificate,allow_nan=False)
    assert arrays["proxy_qd_float64"][:,0].tobytes()==np.zeros((2,7)).tobytes()
    assert arrays["proxy_qd_float64"][:,-1].tobytes()==np.zeros((2,7)).tobytes()
    assert detail["dls_iterations"]==8 and detail["dls_damping"]==.03
    assert set(detail["history_hashes"])=={k for k in arrays if k.startswith("dls_")}


def test_atomic_boundaries_and_failure_checkpoint_roundtrip(tmp_path,monkeypatch):
    output=tmp_path/"p8.json";monkeypatch.setattr(schema,"OUTPUT",output)
    monkeypatch.setattr(runner,"OUTPUT",output);output.parent.mkdir(exist_ok=True)
    snap={"head":"a"*40,"clean":True,"sources":{"x":"b"*64},
        "extension":runner.frozen_extension()}
    monkeypatch.setattr(runner,"snapshot",lambda:snap)
    argv=(*runner.AUTHORIZED_ORIG_ARGV[:-1],str(output));monkeypatch.setattr(runner.sys,"orig_argv",list(argv))
    prov=runner.provenance(snap);counts=runner.zero_counts()
    runner.publish_checkpoint(0,"generation_zero",counts,prov,{})
    rejected={"protocol":schema.PROTOCOL,"stage":"pilot_failed","error_type":"RuntimeError",
        "error_message":"injected","incomplete":True,"completed":0,"pending":2,
        "counts":counts,"trigger_elapsed_s":.1,"cleanup_finish_elapsed_s":.2,
        "wall_limit_s":schema.RUNNER_WALL_LIMIT_S,"active_artifacts":runner.active_artifacts(),
        "operational_limits":runner.operational_limits(),
        "artifact_classifications":runner.classify_artifacts(),
        "diagnostic_certificate":None,"worker_execution":{"status":"not_started",
            "returncode":None,"internal_counts_known":False,"counts":None},"evidence":False}
    runner.atomic_json(runner.rejection_path(),rejected)
    assert runner.recertify_retained_failure(output)["passes"]
    rejected["counts"]["worker_rnea_calls"]=10**9
    runner.rejection_path().write_text(json.dumps(rejected,sort_keys=True,separators=(",",":")))
    assert not runner.recertify_retained_failure(output)["passes"]


def test_real_synthetic_success_disk_roundtrip_and_refreshed_science_lie(tmp_path,monkeypatch):
    output=tmp_path/"p8.json";output.parent.mkdir(exist_ok=True)
    monkeypatch.setattr(schema,"OUTPUT",output);monkeypatch.setattr(runner,"OUTPUT",output)
    snap={"head":"a"*40,"clean":True,"sources":{"x":"b"*64},
        "extension":runner.frozen_extension()}
    monkeypatch.setattr(runner,"snapshot",lambda:snap)
    monkeypatch.setattr(worker,"snapshot",lambda:snap)
    monkeypatch.setattr(runner.sys,"orig_argv",list((*runner.AUTHORIZED_ORIG_ARGV[:-1],str(output))))
    monkeypatch.setattr(worker.sys,"orig_argv",list(worker.expected_argv()))
    def kin(q):
        J=np.zeros((3,7));J[:3,:3]=np.eye(3)
        return np.asarray([q[0],q[1],.5]),J
    monkeypatch.setattr(runner,"exact_box_dls",lambda J,res,q,lo,hi:
        (np.r_[res[:2],np.zeros(5)],np.zeros(7,np.int8)))
    q0=np.zeros(7);qg=np.zeros(7);qg[0]=.08
    prerequisite={"public_x0_float32":np.r_[q0,np.zeros(7)][None].astype(np.float32),
        "quarantined_q8_float64":qg[None],"public_default_side_int8":np.asarray([1],np.int8),
        "joint_lower_float64":np.full(7,-100.),"joint_upper_float64":np.full(7,100.),
        "velocity_limit_float64":np.full(7,100.),"effort_limit_float64":np.full(7,10000.)}
    authentication={"passes":True,"synthetic":True}
    monkeypatch.setattr(runner,"authenticate_cpu_prerequisite",lambda:(authentication,prerequisite))
    monkeypatch.setattr(runner,"_production_pin_context",lambda:(None,kin,
        lambda q,v,a:a,None,None,None,None))
    construction,_=runner.construct_cpu(prerequisite,kin,float("inf"))
    construction_cert=runner.certify_construction(construction,prerequisite,kin)
    state=np.concatenate((construction["proxy_q_float64"],
        construction["proxy_qd_float64"]),axis=2).astype(np.float32)
    tool=np.asarray([[kin(q)[0] for q in route[:,:7]] for route in state],np.float32)
    promoted=state[:,:-1].astype(np.float64)
    commanded=construction["proxy_qdd_float64"] \
        +schema.KP*(construction["proxy_q_float64"][:,:-1]-promoted[:,:,:7]) \
        +schema.KD*(construction["proxy_qd_float64"][:,:-1]-promoted[:,:,7:])
    controls=commanded.astype(np.float32)
    affine=np.empty((2,schema.AFFINE_SAMPLES,3),np.float32)
    for route in range(2):
        index=0
        for knot in range(schema.INTERVALS):
            for substep in range(schema.AFFINE_SUBSTEPS):
                alpha=substep/schema.AFFINE_SUBSTEPS
                affine[route,index]=kin((1-alpha)*state[route,knot,:7]
                    +alpha*state[route,knot+1,:7])[0];index+=1
        affine[route,-1]=kin(state[route,-1,:7])[0]
    seeds=np.stack([schema.pack_seed(state[0 if lane<8 else 1,:,:7],
        state[0 if lane<8 else 1,:,7:],controls[0 if lane<8 else 1])
        for lane in range(schema.LANES)])
    arrays={"captured_x0_float32":construction["x0_float32"],
        "generated_state_float32":state,"generated_tool_float32":tool,
        "rnea_q_float64":state[:,:-1,:7].astype(np.float64),
        "rnea_qd_float64":state[:,:-1,7:].astype(np.float64),
        "rnea_qdd_float64":commanded,"rnea_u_float64":commanded,"controls_float32":controls,
        "fresh_b1_state_float32":state.copy(),"fresh_b1_tool_float32":tool.copy(),
        "b1_defect_float32":np.zeros((2,schema.INTERVALS,14),np.float32),
        "fresh_b16_state_float32":np.stack([state[0 if lane<8 else 1] for lane in range(16)]),
        "fresh_b16_tool_float32":np.stack([tool[0 if lane<8 else 1] for lane in range(16)]),
        "b16_defect_float32":np.zeros((16,schema.INTERVALS,14),np.float32),
        "b1_seed_float32":seeds[0],"b16_seed_float32":seeds,
        "affine_cuda_tool_float32":affine}
    assert worker.certify_output(arrays,construction)
    paths=schema.worker_paths();runner.atomic_npz(paths["input"],construction)
    request={"protocol":schema.WORKER_PROTOCOL,"identity":["development",12600],
        "input_path":str(paths["input"]),"input_sha256":runner.sha(paths["input"]),
        "output_json_path":str(paths["json"]),"output_npz_path":str(paths["npz"]),
        "extension":schema.EXTENSION,"wall_limit_s":300.}
    runner.atomic_json(paths["request"],request);runner.atomic_npz(paths["npz"],arrays)
    cuda={"uuid":"u","name":"gpu","driver_version":"1","compute_capability":"6.1",
        "cuda_runtime_version":1,"cuda_driver_api_version":1}
    worker_prov=worker.provenance(snap,snap)
    worker_summary={"protocol":schema.WORKER_PROTOCOL,"identity":["development",12600],
        "request_path":str(paths["request"]),"request_sha256":runner.sha(paths["request"]),
        "input_path":str(paths["input"]),"input_sha256":runner.sha(paths["input"]),
        "npz_path":str(paths["npz"]),"npz_sha256":runner.sha(paths["npz"]),
        "array_names":sorted(arrays),"array_hashes":{k:schema.array_hash(arrays[k]) for k in sorted(arrays)},
        "elapsed_s":1.,"counts":worker.SUCCESS_COUNTS,"provenance":worker_prov,
        "extension":snap["extension"],"resolved_module":worker.frozen_extension(),
        "cuda_diagnostics":cuda,"certificate":True}
    runner.atomic_json(paths["json"],worker_summary)
    monkeypatch.setattr(runner.sys,"orig_argv",list((*runner.AUTHORIZED_ORIG_ARGV[:-1],str(output))))
    science=runner.certify_science(arrays,construction,prerequisite,kin,lambda q,v,a:a)
    assert science["passes"]
    prov0=runner.provenance(snap);prov1=runner.provenance(snap,snap)
    runner.publish_checkpoint(0,"generation_zero",runner.zero_counts(),prov0,{})
    runner.publish_checkpoint(1,"cpu_authenticated",runner.cpu_counts(),prov0,
        {"authentication":authentication,"construction":construction_cert})
    runner.publish_checkpoint(2,"cuda_certified",runner.success_counts(),prov0,{"certificate":science})
    runner.publish_checkpoint(3,"honest_end",runner.success_counts(),prov1,{"certificate":science})
    execution={"status":"known_complete","returncode":0,"internal_counts_known":True,
        "counts":worker.SUCCESS_COUNTS}
    final={"protocol":schema.PROTOCOL,"incomplete":False,"completed":2,"pending":0,
        "counts":runner.success_counts(),"authentication":authentication,
        "construction_certificate":construction_cert,"worker":worker_summary,
        "worker_execution":execution,"certificate":science,"provenance":prov1,
        "semantic_finish_elapsed_s":2.,"evidence":True,"oracle_evidence":False,
        "benchmark_evidence":False,"sqp_evidence":False,
        "operational_limits":runner.operational_limits()}
    runner.atomic_json(output,final)
    side=[*(schema.checkpoint_path(i) for i in range(4)),paths["request"],paths["input"],
        paths["json"],paths["npz"]]
    manifest={"protocol":schema.PROTOCOL,"json_path":str(output),
        "json_sha256":runner.sha(output),"side_artifacts":[{"path":str(path),
        "sha256":runner.sha(path),"size":path.stat().st_size} for path in side],"overall_pass":True}
    runner.atomic_json(schema.manifest_path(),manifest)
    pointer={"protocol":schema.PROTOCOL,"incomplete":False,"json_path":str(output),
        "json_sha256":runner.sha(output),"manifest_path":str(schema.manifest_path()),
        "manifest_sha256":runner.sha(schema.manifest_path()),"publication_finish_elapsed_s":3.}
    runner.atomic_json(schema.latest_path(),pointer,True)
    assert runner.recertify_retained_p8(output)["passes"]
    # Canonical final+manifest with the still-incomplete gen3 pointer is a late
    # publication failure, never success evidence.
    gen3=schema.checkpoint_path(3)
    runner.atomic_json(schema.latest_path(),{"protocol":schema.PROTOCOL,"generation":3,
        "path":str(gen3),"sha256":runner.sha(gen3),"incomplete":True},True)
    late={"protocol":schema.PROTOCOL,"stage":"pilot_failed","error_type":"RuntimeError",
        "error_message":"injected pointer publication failure","incomplete":True,
        "completed":2,"pending":0,"counts":runner.success_counts(),"trigger_elapsed_s":3.,
        "cleanup_finish_elapsed_s":4.,"wall_limit_s":schema.RUNNER_WALL_LIMIT_S,
        "operational_limits":runner.operational_limits(),
        "active_artifacts":runner.active_artifacts(),
        "artifact_classifications":runner.classify_artifacts(construction),
        "diagnostic_certificate":science,"worker_execution":execution,"evidence":False}
    runner.atomic_json(schema.rejection_path(),late)
    assert runner.recertify_retained_failure(output)["passes"]
    schema.rejection_path().unlink()
    # Turn the same fully materialized worker/checkpoint chain into a genuine
    # known-complete terminal record, then recertify the failing boundary.
    output.unlink();schema.manifest_path().unlink()
    runner.atomic_json(schema.latest_path(),{"protocol":schema.PROTOCOL,"generation":3,
        "path":str(gen3),"sha256":runner.sha(gen3),"incomplete":True},True)
    rejection={"protocol":schema.PROTOCOL,"stage":"pilot_failed","error_type":"RuntimeError",
        "error_message":"injected post-worker failure","incomplete":True,"completed":2,"pending":0,
        "counts":runner.success_counts(),"trigger_elapsed_s":4.,"cleanup_finish_elapsed_s":5.,
        "wall_limit_s":schema.RUNNER_WALL_LIMIT_S,"active_artifacts":runner.active_artifacts(),
        "operational_limits":runner.operational_limits(),
        "artifact_classifications":runner.classify_artifacts(construction),
        "diagnostic_certificate":science,"worker_execution":execution,"evidence":False}
    runner.atomic_json(schema.rejection_path(),rejection)
    assert runner.recertify_retained_failure(output)["passes"]
    # A parent timeout after one independently measured FK call retains the
    # known-complete worker but certifies only the actual semantic prefix.
    schema.checkpoint_path(2).unlink();schema.checkpoint_path(3).unlink()
    gen1=schema.checkpoint_path(1)
    runner.atomic_json(schema.latest_path(),{"protocol":schema.PROTOCOL,"generation":1,
        "path":str(gen1),"sha256":runner.sha(gen1),"incomplete":True},True)
    prefix=runner.cpu_counts();prefix.update(worker_attempts=1,worker_successes=1,
        worker_pin_contexts=1,b1_constructors=1,b16_constructors=1,worker_rnea_calls=518,
        b1_sim_forward_calls=1036,b16_sim_forward_calls=259,b1_tool_position_calls=9329,
        b16_tool_position_calls=260,science_routes_attempted=1,
        parent_identical_state_fk_calls=1)
    rejection.update(stage="runtime_watchdog_rejected",error_type="TimeoutError",
        error_message="P8 campaign wall limit",completed=0,pending=2,counts=prefix,
        diagnostic_certificate=None)
    rejection["active_artifacts"]=runner.active_artifacts()
    rejection["artifact_classifications"]=runner.classify_artifacts(construction)
    schema.rejection_path().write_text(json.dumps(rejection,sort_keys=True,separators=(",",":")))
    assert runner.recertify_retained_failure(output)["passes"]
    arrays["rnea_qdd_float64"][0,0,0]+=1.
    with paths["npz"].open("wb") as stream:np.savez(stream,**arrays)
    worker_summary["npz_sha256"]=runner.sha(paths["npz"])
    worker_summary["array_hashes"]["rnea_qdd_float64"]=schema.array_hash(
        arrays["rnea_qdd_float64"])
    paths["json"].write_text(json.dumps(worker_summary,sort_keys=True,separators=(",",":")))
    rejection["active_artifacts"]=runner.active_artifacts()
    rejection["artifact_classifications"]=runner.classify_artifacts(construction)
    schema.rejection_path().write_text(json.dumps(rejection,sort_keys=True,separators=(",",":")))
    assert not runner.recertify_retained_failure(output)["passes"]
    # The same malformed NPZ is acceptable only as explicitly hashed non-evidence
    # when JSON parsing never established a known-complete worker boundary.
    paths["json"].unlink();schema.checkpoint_path(2).unlink(missing_ok=True)
    schema.checkpoint_path(3).unlink(missing_ok=True);gen1=schema.checkpoint_path(1)
    schema.latest_path().write_text(json.dumps({"protocol":schema.PROTOCOL,"generation":1,
        "path":str(gen1),"sha256":runner.sha(gen1),"incomplete":True},sort_keys=True,
        separators=(",",":")))
    partial=runner.cpu_counts();partial["worker_attempts"]=1
    rejection.update(completed=0,pending=2,counts=partial,diagnostic_certificate=None,
        worker_execution={"status":"internal_unknown","returncode":0,
            "internal_counts_known":False,"counts":None})
    rejection["active_artifacts"]=runner.active_artifacts()
    rejection["artifact_classifications"]=runner.classify_artifacts(construction)
    schema.rejection_path().write_text(json.dumps(rejection,sort_keys=True,separators=(",",":")))
    assert runner.recertify_retained_failure(output)["passes"]
    checkpoint=json.loads(gen1.read_text());checkpoint["detail"]["construction"]["passes"]=False
    gen1.write_text(json.dumps(checkpoint,sort_keys=True,separators=(",",":")))
    schema.latest_path().write_text(json.dumps({"protocol":schema.PROTOCOL,"generation":1,
        "path":str(gen1),"sha256":runner.sha(gen1),"incomplete":True},sort_keys=True,
        separators=(",",":")))
    rejection["active_artifacts"]=runner.active_artifacts()
    rejection["artifact_classifications"]=runner.classify_artifacts(construction)
    schema.rejection_path().write_text(json.dumps(rejection,sort_keys=True,separators=(",",":")))
    assert not runner.recertify_retained_failure(output)["passes"]


def test_no_sqp_optimizer_rng_or_p6_artifact_access_in_execution_sources():
    text=inspect.getsource(runner.execute)+inspect.getsource(worker.generate)
    assert "minimize(" not in text and ".solve(" not in text and "default_rng" not in text
    assert "/tmp/tiago-tool-center-cuda-authoritative-route-seed-p6" not in text
    assert runner.success_counts()["rejected_p6_artifact_loads"]==0
