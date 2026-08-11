import inspect
import json
import re
from pathlib import Path

import numpy as np
import pytest

from gato_tiago import multimodal_toll as toll
from gato_tiago import multimodal_toll_oracle_v1 as oracle
from gato_tiago import multimodal_toll_oracle_v1_runner as runner
from gato_tiago import multimodal_toll_oracle_v1_worker as worker
from gato_tiago import multimodal_toll_v4_model_preflight_v4_runner as model_runner
from gato_tiago import multimodal_toll_v4_model_preflight_v4_worker as model_worker


def test_static_contract_tokens_pins_ledger_and_grid_are_exact():
    assert oracle.ORACLE_EXECUTION_AUTHORIZATION is None
    assert runner.RUNNER_EXECUTION_AUTHORIZATION is None
    assert worker.WORKER_EXECUTION_AUTHORIZATION is None
    assert model_runner.RUNNER_EXECUTION_AUTHORIZATION is None
    assert model_worker.WORKER_EXECUTION_AUTHORIZATION is None
    assert oracle.Z_WIDTH == 1337
    assert oracle.DENSE_SAMPLES == 4033
    assert len(oracle.EXPECTED_TASK_IDENTITIES) == 12
    assert len(oracle.EXPECTED_LEDGER) == 120
    assert oracle.EXPECTED_LEDGER[:5] == tuple(
        ("development", 12600, "default", i) for i in range(5)
    )
    assert oracle.EXPECTED_LEDGER[5:10] == tuple(
        ("development", 12600, "alternate", i) for i in range(5)
    )
    assert oracle.TEMPLATE_RADII_M == pytest.approx((.038,.041,.044,.047,.050))
    assert oracle.TRUST_CONSTR_OPTIONS == {
        "method":"trust-constr","gtol":1e-10,"xtol":1e-12,
        "barrier_tol":1e-12,"sparse_jacobian":True,"curvature":"BFGS",
    }
    assert {name:digest for name,(_path,digest) in oracle.MODEL_ARTIFACT_PINS.items()} == {
        "final_json":"3493feae03b7b6368a0dc506aa1f0b5067c63c16f90e08b3455c2314f6e7cc16",
        "final_npz":"fe29896b3c5cacbfb15be2a66ddc222a88f8e2e5c2646e183cdbaac34bd1fb6e",
        "final_manifest":"438df87e388352ebe5762ca6e9cc3163019d1d9a9fdb616d5d4b027e8bc1a92c",
        "latest_pointer":"d0eddf81f3a7072fda795d92ad7f0e75d869ff62cb282c3eb1b453b1af98539e",
        "worker_input":"2e0953180dc9661a4740e57a6f94d9a8c4e438e7f4e3b34273bf5c07a154f426",
    }


def test_all_execution_tokens_are_disabled_repo_wide():
    root=Path(__file__).resolve().parents[2]; enabled=[]
    pattern=re.compile(r"^[A-Z][A-Z0-9_]*AUTHORIZATION = object\(\)$")
    for path in sorted((root/"tiago_src/gato_tiago").glob("*.py")):
        for line in path.read_text().splitlines():
            if pattern.fullmatch(line): enabled.append((path.name,line))
    assert enabled == []


def test_template_endpoints_progress_and_opposite_turns():
    start=np.array([-.10,0.,.5]); goal=np.array([.10,0.,.52]); center=np.zeros(2)
    left=oracle.tangent_arc_tangent_template(start,goal,center,.05,1)
    right=oracle.tangent_arc_tangent_template(start,goal,center,.05,-1)
    assert left.shape==right.shape==(64,3)
    assert np.array_equal(left[0],start) and np.array_equal(left[-1],goal)
    assert np.array_equal(right[0],start) and np.array_equal(right[-1],goal)
    assert np.min(np.linalg.norm(left[:,:2],axis=1)) >= .05-1e-12
    assert np.min(np.linalg.norm(right[:,:2],axis=1)) >= .05-1e-12
    assert np.sign(toll.signed_open_turn(left[:,:2],center)[1]) == 1
    assert np.sign(toll.signed_open_turn(right[:,:2],center)[1]) == -1


def test_box_dls_exhausts_faces_and_uses_lexicographic_tie():
    J=np.zeros((3,7)); residual=np.zeros(3); lower=-np.ones(7); upper=np.ones(7)
    result=oracle.enumerate_box_dls(J,residual,lower,upper)
    assert result["status"].shape==(2187,7)
    assert result["dq"].shape==(2187,7)
    assert result["selected"] == 1093
    assert np.array_equal(result["status"][result["selected"]],np.zeros(7,dtype=np.int8))
    assert np.array_equal(result["dq"][result["selected"]],np.zeros(7))
    assert np.all(result["feasible"])
    source=inspect.getsource(oracle.enumerate_box_dls)
    assert "itertools.product(FACE_STATUS_ORDER, repeat=7)" in source
    assert "DLS_DAMPING**2" in source
    independent=oracle.certify_box_dls_independently(J,residual,lower,upper,result)
    assert independent["passes"] and not independent["verifier_calls_constructor_enumerator"]
    independent_source=inspect.getsource(oracle.independently_enumerate_box_dls)
    assert "return enumerate_box_dls(" not in independent_source
    bad=dict(result); bad["selected"]=0
    assert not oracle.certify_box_dls_independently(J,residual,lower,upper,bad)["passes"]


def test_frozen_qd_qdd_rnea_recurrence_calls_exactly_63():
    t=np.arange(64)[:,None]; q=np.tile(t,(1,7))*1e-5; calls=[]
    def rnea(qk,qdk,qddk):
        calls.append((qk.copy(),qdk.copy(),qddk.copy())); return qk+qdk+qddk
    qd,qdd,u=oracle.knot_derivatives_and_rnea(q,rnea)
    assert len(calls)==63 and qd.shape==(64,7) and qdd.shape==u.shape==(63,7)
    assert np.allclose(qd[1],2*(q[1]-q[0])/oracle.DT)
    assert np.allclose(qdd[0],(qd[1]-qd[0])/oracle.DT)


def test_acquisition_builder_uses_exact_endpoints_iterations_and_rnea(monkeypatch):
    dls=[]
    def fake_dls(_J,_r,_lo,_hi):
        dls.append(1); return {"dq":np.zeros((1,7)),"selected":0,"status":np.zeros((1,7),np.int8),
                               "objective":np.zeros(1),"primal":np.zeros(1),"free_residual":np.zeros(1),"feasible":np.ones(1,bool)}
    monkeypatch.setattr(oracle,"enumerate_box_dls",fake_dls)
    def kin(q):
        J=np.zeros((3,7)); J[:3,:3]=np.eye(3); return np.array([-.1,0,.5]),J
    rnea=[]
    result=oracle.build_acquisition_initial(np.zeros(14),_reference(),-1,np.full(7,.1),
        -np.ones(7),np.ones(7),"default",0,kin,lambda *_:rnea.append(1) or np.zeros(7))
    assert len(dls)==62*12 and len(rnea)==63
    assert np.array_equal(result["q_float64"][0],np.zeros(7))
    assert np.array_equal(result["q_float64"][-1],np.full(7,.1))
    assert result["canonical_mode_sign"]==-1 and not result["benchmark_seed_eligible"]


def test_compact_dls_history_reenumerates_without_retaining_face_tables(monkeypatch):
    arrays={name:np.zeros(shape,dtype) for name,(shape,dtype) in {
        "dls_q_before":((744,7),np.float64),"dls_position":((744,3),np.float64),
        "dls_jacobian":((744,3,7),np.float64),"dls_residual":((744,3),np.float64),
        "dls_lower":((744,7),np.float64),"dls_upper":((744,7),np.float64),
        "dls_selected_dq":((744,7),np.float64),"dls_selected_status":((744,7),np.int8),
        "dls_selected_objective":((744,),np.float64),"dls_selected_primal":((744,),np.float64),
        "dls_selected_free_residual":((744,),np.float64),"dls_selected_face_index":((744,),np.int64)}.items()}
    arrays["dls_lower"][:]=-1.; arrays["dls_upper"][:]=1.; arrays["dls_selected_face_index"][:]=1
    verified={"selected":1,"dq":np.zeros((2,7)),"objective":np.zeros(2),"status":np.zeros((2,7),np.int8),
              "primal":np.zeros(2),"free_residual":np.zeros(2)}
    calls=[]; monkeypatch.setattr(oracle,"independently_enumerate_box_dls",lambda *_:calls.append(1) or verified)
    assert oracle.certify_compact_dls_history(arrays)["passes"] and len(calls)==744
    bad={**arrays,"dls_selected_dq":arrays["dls_selected_dq"].copy()}; bad["dls_selected_dq"][3,0]=1e-4
    assert not oracle.certify_compact_dls_history(bad)["passes"]
    assert not any(np.asarray(value).shape[:1]==(oracle.FACE_COUNT,) for value in arrays.values())


def test_pack_unpack_exact_width_and_shapes():
    x=np.arange(64*14,dtype=np.float64).reshape(64,14)
    u=np.arange(63*7,dtype=np.float64).reshape(63,7)
    z=oracle.pack_z(x,u); xr,ur=oracle.unpack_z(z)
    assert z.shape==(1337,) and np.array_equal(x,xr) and np.array_equal(u,ur)
    with pytest.raises(ValueError): oracle.unpack_z(z[:-1])


def _reference():
    return np.asarray([.2,0,.5, 0,0, .03, 0,.055, .02,.005],dtype=np.float32)


def test_full_and_base_objective_differ_only_by_running_toll():
    x=np.zeros((64,14)); u=np.zeros((63,7)); p=np.tile([0,.055,.5],(64,1))
    lower=-np.ones(7); upper=np.ones(7)
    full=oracle.reconstruct_objective(x,u,p,_reference(),lower,upper,np.ones(7),np.ones(7),include_toll=True)
    base=oracle.reconstruct_objective(x,u,p,_reference(),lower,upper,np.ones(7),np.ones(7),include_toll=False)
    assert full>base and full-base==pytest.approx(63*.25)


def test_original_kkt_uses_frozen_h_sign_and_explicit_bounds():
    result=oracle.original_kkt(np.zeros(2),np.zeros((1,2)),np.zeros(1),np.eye(2),np.ones(2),
                               np.zeros(1),np.zeros(2),np.ones(2),np.ones(2),np.zeros(2),np.zeros(2))
    assert result["passes"]
    bad=oracle.original_kkt(np.zeros(2),np.zeros((1,2)),np.zeros(1),np.eye(2),np.ones(2),
                            np.zeros(1),-np.ones(2),np.ones(2),np.ones(2),np.zeros(2),np.zeros(2))
    assert bad["dual_sign_inf"]==1 and not bad["passes"]


def test_retained_kkt_schema_recomputes_and_rejects_multiplier_mutation():
    arrays={"objective_gradient":np.zeros(oracle.Z_WIDTH),
            "equality_jacobian":np.zeros((1,oracle.Z_WIDTH)),"equality_residual":np.zeros(1),
            "inequality_jacobian":np.zeros((1,oracle.Z_WIDTH)),"inequality_slack":np.ones(1),
            "equality_multipliers":np.zeros(1),"inequality_multipliers":np.zeros(1),
            "lower_slack":np.ones(oracle.Z_WIDTH),"upper_slack":np.ones(oracle.Z_WIDTH),
            "lower_multipliers":np.zeros(oracle.Z_WIDTH),"upper_multipliers":np.zeros(oracle.Z_WIDTH),
            "scipy_raw_constraint_multipliers":np.zeros(2),"scipy_raw_bound_multipliers":np.zeros(oracle.Z_WIDTH)}
    assert oracle.certify_retained_kkt(arrays)["passes"]
    bad={**arrays,"inequality_multipliers":-np.ones(1)}
    assert not oracle.certify_retained_kkt(bad)["passes"]
    assert not oracle.certify_retained_kkt({k:v for k,v in arrays.items() if k!="lower_slack"})["passes"]


def _stationary_original_problem(template=None):
    reference=_reference(); reference[:3]=0.; reference[3:5]=[.3,.3]; reference[6:8]=[.4,.4]
    def kin(q):
        J=np.zeros((3,7)); J[:3,:3]=np.eye(3); return np.asarray(q[:3]),J
    def aba(_q,_v,u): return np.asarray(u),np.zeros((7,7)),np.zeros((7,7)),np.eye(7)
    def speed(_q,v):
        J=np.zeros((3,7)); J[:3,:3]=np.eye(3); return np.asarray(v[:3]),np.zeros((3,7)),J
    def replay(x0,controls):
        state=np.repeat(np.asarray(x0,dtype=np.float64)[None,:],oracle.DENSE_SAMPLES,axis=0)
        tool=np.asarray([kin(x[:7])[0] for x in state])
        return state,tool
    return oracle.OriginalOracleNLP(np.zeros(14),reference,-np.ones(7),np.ones(7),np.ones(7),np.ones(7),kin,aba,speed,
                                    template=template,rnea=lambda *_:np.zeros(7),dense_replay=replay)


def test_bound_compact_history_rejects_q_position_template_and_recurrence_mutations(monkeypatch):
    problem=_stationary_original_problem(); identity=oracle.EXPECTED_LEDGER[0]
    ref=toll.TollReference.from_solver_bytes(problem.reference)
    template=oracle.tangent_arc_tangent_template(np.zeros(3),ref.goal_xyz,ref.cylinder_xy,
                                                 oracle.TEMPLATE_RADII_M[0],-1)
    retained={name:np.zeros(shape,dtype) for name,(shape,dtype) in oracle.ROW_ARRAY_SPECS.items()}
    retained["template_float64"][:]=template
    retained["dls_jacobian"][:,:3,:3]=np.eye(3)
    retained["dls_lower"][:]=-1+oracle.DLS_MARGIN_RAD
    retained["dls_upper"][:]=1-oracle.DLS_MARGIN_RAD
    for knot in range(1,oracle.KNOTS-1):
        retained["dls_residual"][(knot-1)*12:knot*12]=template[knot]
    retained["seed_primal_float64"][:]=oracle.pack_z(
        np.c_[retained["seed_q_float64"],retained["seed_qd_float64"]],retained["seed_u_float64"])
    verified={"selected":0,"dq":np.zeros((1,7)),"objective":np.zeros(1),"status":np.zeros((1,7),np.int8),
              "primal":np.zeros(1),"free_residual":np.zeros(1)}
    monkeypatch.setattr(oracle,"independently_enumerate_box_dls",lambda *_:verified)
    assert runner.certify_bound_acquisition_history(retained,identity,np.zeros(7),-1,problem,reenumerate=True)["passes"]
    for name,index,value in (("dls_q_before",(4,0),123.),("dls_position",(8,1),-456.),
                             ("dls_position",(9,2),1e-13),
                             ("template_float64",(2,0),.2),("seed_qd_float64",(1,0),.1)):
        bad={key:array.copy() for key,array in retained.items()}; bad[name][index]=value
        assert not runner.certify_bound_acquisition_history(bad,identity,np.zeros(7),-1,problem,reenumerate=True)["passes"]


def test_primal_kkt_rebuilds_exact_problem_and_raw_scipy_sign_mapping():
    problem=_stationary_original_problem(); z=np.zeros(oracle.Z_WIDTH)
    retained={"raw_equality_multipliers_float64":np.zeros(oracle.EQUALITY_COUNT),
              "raw_inequality_multipliers_float64":np.zeros(oracle.INEQUALITY_COUNT),
              "raw_bound_multipliers_float64":np.zeros(oracle.Z_WIDTH),
              "canonical_equality_multipliers_float64":np.zeros(oracle.EQUALITY_COUNT),
              "canonical_inequality_multipliers_float64":np.zeros(oracle.INEQUALITY_COUNT),
              "canonical_lower_multipliers_float64":np.zeros(oracle.Z_WIDTH),
              "canonical_upper_multipliers_float64":np.zeros(oracle.Z_WIDTH)}
    assert oracle.certify_primal_kkt(problem,z,retained)["passes"]
    bad={**retained,"canonical_inequality_multipliers_float64":np.ones(oracle.INEQUALITY_COUNT)}
    assert not oracle.certify_primal_kkt(problem,z,bad)["passes"]
    assert not oracle.certify_primal_kkt(_stationary_original_problem(np.zeros((64,3))),z,retained)["passes"]


def test_unanchored_problem_binding_rejects_anchored_or_changed_public_input():
    problem=_stationary_original_problem(); payload={"acquisition_primal_float64":np.zeros(oracle.Z_WIDTH),
        "public_x0_float32":np.zeros(14,np.float32),"public_reference_float32":problem.reference,
        "joint_lower_float64":-np.ones(7),"joint_upper_float64":np.ones(7),
        "velocity_limit_float64":np.ones(7),"effort_limit_float64":np.ones(7),
        "solver_options":{**oracle.TRUST_CONSTR_OPTIONS,"maxiter":oracle.POLISH_MAXITER}}
    assert oracle.certify_polish_problem_binding(problem,payload)
    assert not oracle.certify_polish_problem_binding(_stationary_original_problem(np.zeros((64,3))),payload)
    changed={**payload,"effort_limit_float64":np.full(7,2.)}
    assert not oracle.certify_polish_problem_binding(problem,changed)


def test_directional_derivative_gate_passes_and_rejects():
    value=lambda z:.5*float(z@z); good=lambda z:z; bad=lambda z:2*z
    z=np.array([.2,-.3]); d=np.array([.7,.4])
    assert runner.directional_derivative_gate(value,good,z,d)["passes"]
    assert not runner.directional_derivative_gate(value,bad,z,d)["passes"]


def _linear_original_problem(template=None):
    def kin(q):
        J=np.zeros((3,7)); J[:3,:3]=np.eye(3); return np.asarray(q[:3]),J
    def aba(_q,_v,_u): return np.asarray(_u),np.zeros((7,7)),np.zeros((7,7)),np.eye(7)
    def speed(_q,v):
        J=np.zeros((3,7)); J[:3,:3]=np.eye(3); return np.asarray(v[:3]),np.zeros((3,7)),J
    reference=_reference().copy(); reference[3:5]=[.3,.3]
    return oracle.OriginalOracleNLP(np.zeros(14),reference,-2*np.ones(7),2*np.ones(7),
                                    2*np.ones(7),2*np.ones(7),kin,aba,speed,template=template)


@pytest.mark.parametrize("anchored",[False,True])
def test_original_nlp_exact_objective_and_constraint_directional_derivatives(anchored):
    template=np.zeros((64,3)) if anchored else None; problem=_linear_original_problem(template)
    z=np.zeros(oracle.Z_WIDTH); d=np.linspace(-.2,.2,oracle.Z_WIDTH)
    assert runner.directional_derivative_gate(problem.objective,problem.objective_gradient,z,d)["passes"]
    assert runner.directional_jacobian_gate(problem.equality,problem.equality_jacobian,z,d)["passes"]
    assert runner.directional_jacobian_gate(problem.inequality,problem.inequality_jacobian,z,d)["passes"]
    lower,upper=problem.bounds_arrays()
    assert lower.shape==upper.shape==(oracle.Z_WIDTH,) and np.all(lower<upper)


def test_fixed_three_point_production_derivative_diagnostic_is_fail_closed():
    result=runner.production_pin_derivative_diagnostics(_stationary_original_problem(),np.zeros(7),np.full(7,.1),lambda *_:np.zeros(7))
    assert result["exact_labels"] and result["all_pin_derivative_diagnostics_pass"]
    assert [row["label"] for row in result["rows"]]==["interior","near_cylinder","terminal"]


def test_pin_acquisition_and_polish_replays_are_independently_recomputed_and_speed_is_kinematic():
    problem=_stationary_original_problem(); controls=np.zeros((63,7))
    state,tool=problem.dense_replay(problem.x0,controls)
    result=runner.certify_independent_pin_replays(problem,controls,controls,state,tool,state,tool)
    assert result["passes"] and runner.kinematic_tool_speed(problem,state[-1])==0.
    bad_state=state.copy(); bad_state[17,0]=1e-4
    assert not runner.certify_independent_pin_replays(problem,controls,controls,bad_state,tool,state,tool)["passes"]
    bad_tool=tool.copy(); bad_tool[21,1]=1e-4
    assert not runner.certify_independent_pin_replays(problem,controls,controls,state,tool,state,bad_tool)["passes"]
    # A finite-difference endpoint jump cannot masquerade as J(q) qd speed.
    jumped=tool.copy(); jumped[-1,0]=1.
    assert runner.kinematic_tool_speed(problem,state[-1])==0.
    assert np.linalg.norm(jumped[-1]-jumped[-2])/(oracle.DT/oracle.DENSE_SUBSTEPS)>1.


def _worker_arrays():
    reference=_reference(); reference[:3]=0.; reference[3:5]=[.1,.1]
    return {
        "captured_x0_float32":np.zeros(14,dtype=np.float32),
        "captured_reference_float32":reference,
        "captured_controls_float32":np.zeros((63,7),dtype=np.float32),
        "captured_joint_lower_float64":-np.ones(7),"captured_joint_upper_float64":np.ones(7),
        "captured_velocity_limit_float64":np.ones(7),"captured_effort_limit_float64":np.ones(7),
        "cuda_knot_states_float32":np.zeros((64,14),dtype=np.float32),
        "cuda_dense_states_float32":np.zeros((4033,14),dtype=np.float32),
        "cuda_dense_tool_float32":np.zeros((4033,3),dtype=np.float32),
        "dense_time_float64":np.arange(4033,dtype=np.float64)*oracle.DT/64,
    }


def _worker_summary():
    return {"protocol":worker.WORKER_PROTOCOL_VERSION,"module_attributes":oracle.CUDA_MODULE_ATTRIBUTES,
            "constructor_calls":2,"sim_forward_calls":4032,"tool_position_calls":253,"solve_calls":0,"sqp_calls":0}


@pytest.mark.parametrize("mutation",["extra","dtype","shape","nan","terminal","clearance","calls"])
def test_worker_replay_schema_is_fail_closed(mutation):
    arrays=_worker_arrays(); summary=_worker_summary()
    if mutation=="extra": arrays["extra"]=np.zeros(1)
    elif mutation=="dtype": arrays["captured_x0_float32"]=arrays["captured_x0_float32"].astype(np.float64)
    elif mutation=="shape": arrays["cuda_dense_tool_float32"]=arrays["cuda_dense_tool_float32"][:-1]
    elif mutation=="nan": arrays["cuda_dense_states_float32"][0,0]=np.nan
    elif mutation=="terminal": arrays["cuda_dense_tool_float32"][-1]=1.
    elif mutation=="clearance": arrays["captured_reference_float32"][3:5]=0.
    else: summary["sim_forward_calls"]=1
    assert not worker.certify_replay(summary,arrays)["passes"]


def test_worker_replay_exact_synthetic_boundary_passes():
    assert worker.certify_replay(_worker_summary(),_worker_arrays())["passes"]
    fabricated={**_worker_summary(),"terminal_error_m":999.,"q_ratio":999.,"physical_clearance_m":-999.}
    assert worker.certify_replay(fabricated,_worker_arrays())["passes"]
    assert worker.certify_replay(fabricated,_worker_arrays())["derived_metrics"]["terminal_error_m"]==0.


def test_worker_request_is_hard_bound_to_ledger_module_and_paths():
    identity=oracle.EXPECTED_LEDGER[0]; root=oracle.ORACLE_OUTPUT_PATH.parent
    request={"identity":list(identity),"request_json":str(root/"oracle.worker.0000.request.json"),
             "input_npz":str(root/"oracle.worker.0000.input.npz"),
             "input_npz_sha256":"a"*64,"extension_module":oracle.CUDA_EXTENSION_MODULE,
             "extension_path":str(oracle.CUDA_EXTENSION_PATH),"extension_sha256":oracle.CUDA_EXTENSION_SHA256,
             "worker_protocol":worker.WORKER_PROTOCOL_VERSION,
             "output_json":str(root/"oracle.worker.0000.json"),"output_npz":str(root/"oracle.worker.0000.npz")}
    assert worker.validate_request(request)
    for key,value in (("identity",list(oracle.EXPECTED_LEDGER[1])),
                      ("extension_sha256","b"*64),("request_json",str(root/"wrong.request.json")),
                      ("output_json",str(root/"wrong.json"))):
        assert not worker.validate_request({**request,key:value})


def test_worker_uses_accepted_exact_classes_and_refuses_outputs_before_import():
    source=inspect.getsource(worker._run_authorized_worker)
    assert "module.BSQP_1_float()" in source and "module.BSQP_16_float()" in source
    assert "BSQPBatch1" not in source and "BSQPBatch16" not in source
    assert source.index('output already exists') < source.index('importlib.import_module')
    assert source.index('actual_attributes') < source.index('module.BSQP_1_float()')
    assert 'Path(request_path)!=Path(request["request_json"])' in source


def test_worker_atomic_publication_never_overwrites(tmp_path):
    js=tmp_path/"worker.json"; npz=tmp_path/"worker.npz"
    worker._atomic_json(js,{"first":True}); worker._atomic_npz(npz,{"x":np.zeros(1)})
    with pytest.raises(FileExistsError): worker._atomic_json(js,{"second":True})
    with pytest.raises(FileExistsError): worker._atomic_npz(npz,{"x":np.ones(1)})


def test_final_worker_boundary_binds_every_captured_input_and_output_hash():
    retained={name:np.zeros(shape,dtype) for name,(shape,dtype) in oracle.ROW_ARRAY_SPECS.items()}
    retained["dense_time_float64"]=np.arange(oracle.DENSE_SAMPLES)*oracle.DT/oracle.DENSE_SUBSTEPS
    public={"public_x0_float32":np.zeros(14,np.float32),"public_reference_float32":_reference(),
            "joint_lower_float64":-np.ones(7),"joint_upper_float64":np.ones(7),
            "velocity_limit_float64":np.ones(7),"effort_limit_float64":np.ones(7)}
    expected={"captured_x0_float32":public["public_x0_float32"],"captured_reference_float32":public["public_reference_float32"],
        "captured_controls_float32":retained["controls_float64"].astype(np.float32),
        "captured_joint_lower_float64":public["joint_lower_float64"],"captured_joint_upper_float64":public["joint_upper_float64"],
        "captured_velocity_limit_float64":public["velocity_limit_float64"],"captured_effort_limit_float64":public["effort_limit_float64"],
        "cuda_knot_states_float32":retained["cuda_dense_state_float32"][::64],
        "cuda_dense_states_float32":retained["cuda_dense_state_float32"],"cuda_dense_tool_float32":retained["cuda_dense_tool_float32"],
        "dense_time_float64":retained["dense_time_float64"]}
    artifact={name:"x" for name in ("request_path","request_sha256","input_path","input_sha256","json_path","json_sha256","npz_path","npz_sha256")}
    summary={"protocol":worker.WORKER_PROTOCOL_VERSION,"constructor_calls":2,"sim_forward_calls":4032,
             "tool_position_calls":253,"solve_calls":0,"sqp_calls":0,"artifact":artifact,
             "command":["python","-B","-m","gato_tiago.multimodal_toll_oracle_v1_worker","--request","x","--output","x"],
             "cwd":str(oracle.AUTHORIZED_CWD),"exit_code":0,
             "array_hashes":{name:oracle.array_hash(value) for name,value in expected.items()}}
    assert runner.certify_worker_row_boundary(summary,retained,public)
    summary["array_hashes"]={**summary["array_hashes"],"cuda_dense_tool_float32":"0"*64}
    assert not runner.certify_worker_row_boundary(summary,retained,public)


def _dense_route(side,index=0):
    theta=np.linspace(np.pi,0,oracle.DENSE_SAMPLES) if side=="default" else np.linspace(-np.pi,0,oracle.DENSE_SAMPLES)
    radius=.10+index*1e-5
    return np.c_[radius*np.cos(theta),radius*np.sin(theta),np.full(theta.shape,.5)]


def _certified_row(side="default",index=0):
    path=_dense_route(side,index); sign=-1 if side=="default" else 1
    row={"identity":["development",12600,side,index],"requested_side":side,"actual_side":side,
         "margin_index":index,"actual_mode_sign":sign,"public_default_side":-1,"cylinder_xy":np.zeros(2),
         "acquisition_attempted":True,"polish_attempted":True,"acquisition_finite":True,
         "acquisition_success":True,"acquisition_original_primal_inf":0.,
         "acquisition_intended_mode":True,"acquisition_template_tool_rms_m":0.,
         "polish_finite":True,"polish_success":True,"polish_limit_reached":False,
         "polish_equality_inf":0.,"polish_inequality_violation_inf":0.,
         "polish_stationarity_inf":0.,"polish_dual_sign_inf":0.,"polish_complementarity_inf":0.,
         "stability_cost_relative":0.,"stability_dense_tool_rms_m":0.,
         "acquisition_original_full_cost":1.,"polish_full_cost":1.,
         "cuda_terminal_error_m":0.,"pin_terminal_error_m":0.,
         "cuda_final_tool_speed_mps":0.,"pin_final_tool_speed_mps":0.,
         "cuda_physical_clearance_m":.05,"pin_physical_clearance_m":.05,
         "cuda_q_ratio":0.,"cuda_v_ratio":0.,"cuda_u_ratio":0.,
         "pin_q_ratio":0.,"pin_v_ratio":0.,"pin_u_ratio":0.,
         "cuda_full_cost":0.,"pin_full_cost":0.,"cuda_base_cost":0.,"pin_base_cost":0.,
         "cuda_path_length":1.,"pin_path_length":1.,
         "cuda_dense_tool":path,"pin_dense_tool":path,
         "cuda_controls":np.full((63,7),index*1e-3),"pin_controls":np.full((63,7),index*1e-3)}
    arrays={"common_x0":np.zeros(14),"common_reference":_reference(),
            "joint_lower":-2*np.ones(7),"joint_upper":2*np.ones(7),
            "velocity_limit":2*np.ones(7),"effort_limit":2*np.ones(7),
            "cuda_dense_tool":path,"pin_dense_tool":path,
            "acquisition_tool":path[::64],"acquisition_template":path[::64],
            "acquisition_dense_tool":path,"polish_dense_tool":path,
            "cuda_dense_state":np.zeros((oracle.DENSE_SAMPLES,14)),
            "pin_dense_state":np.zeros((oracle.DENSE_SAMPLES,14)),
            "controls":row["cuda_controls"]}
    for model in ("cuda","pin"):
        for flavor,include in (("full",True),("base",False)):
            row[f"{model}_{flavor}_cost"]=oracle.reconstruct_objective(
                arrays[f"{model}_dense_state"][::64],arrays["controls"],path[::64],arrays["common_reference"],
                arrays["joint_lower"],arrays["joint_upper"],arrays["velocity_limit"],arrays["effort_limit"],include_toll=include)
        row[f"{model}_path_length"]=oracle.path_length(path)
    return row,arrays


def test_pure_row_certificate_checks_full_domains_and_both_models():
    row,arrays=_certified_row()
    assert runner.certify_oracle_row(row,arrays)["passes"]
    for key,value in (("pin_terminal_error_m",-.1),("polish_dual_sign_inf",np.nan),
                      ("cuda_full_cost",np.nan),("cuda_physical_clearance_m",.004)):
        bad=dict(row); bad[key]=value
        assert not runner.certify_oracle_row(bad,arrays)["passes"]


def test_all_pair_topology_and_distinctness_are_fail_closed():
    rows=[]
    for side in oracle.SIDES:
        for index in range(5):
            row,arrays=_certified_row(side,index)
            row["certificate_pass"]=runner.certify_oracle_row(row,arrays)["passes"]
            rows.append(row)
    result=runner.certify_task_rows(rows,np.ones(7))
    assert result["passes"] and len(result["topology"]["cuda"])==45
    bad=[dict(row) for row in rows]
    bad[-1]["cuda_dense_tool"]=bad[-1]["cuda_dense_tool"][:-1]
    assert not runner.certify_task_rows(bad,np.ones(7))["passes"]
    extra=[dict(row) for row in rows]
    extra[1]["cuda_dense_tool"]=_dense_route("alternate",1)
    assert not runner.certify_task_rows(extra,np.ones(7))["passes"]


def test_exact_ten_attempts_allows_one_uncertified_restart_per_requested_side():
    rows=[]
    for side in oracle.SIDES:
        for index in range(5):
            row,arrays=_certified_row(side,index); row["certificate_pass"]=runner.certify_oracle_row(row,arrays)["passes"]
            row["certified"]=row["certificate_pass"]; rows.append(row)
    rows[4]["certificate_pass"]=rows[4]["certified"]=False
    rows[9]["certificate_pass"]=rows[9]["certified"]=False
    result=runner.certify_task_rows(rows,np.ones(7))
    assert result["passes"] and result["gates"]["at_least_eight_rows_certified"]
    rows[3]["certificate_pass"]=rows[3]["certified"]=False
    assert not runner.certify_task_rows(rows,np.ones(7))["passes"]


def _campaign_rows():
    rows=[]
    for phase,seed,side,index in oracle.EXPECTED_LEDGER:
        full=1.2 if side=="default" else 1.0; base=.9 if side=="default" else 1.0
        length=1.0 if side=="default" else 1.01
        rows.append({"identity":[phase,seed,side,index],"requested_side":side,"actual_side":side,
                     "margin_index":index,"acquisition_attempted":True,"polish_attempted":True,
                     "certified":True,"certificate_pass":True,
                     "cuda_full_cost":full,"pin_full_cost":full,"cuda_base_cost":base,"pin_base_cost":base,
                     "cuda_path_length":length,"pin_path_length":length})
    return rows


def test_campaign_aggregation_exact120_gap_reversal_and_bootstrap():
    gap=(1.2-1.0)/1.0; gaps={"cuda":np.full(8,gap),"pin":np.full(8,gap)}
    bootstrap={"seed":oracle.BOOTSTRAP_SEED,"resamples":oracle.BOOTSTRAP_RESAMPLES,
               "task_order":list(oracle.HELDOUT_TASK_SEEDS),"lower95_relative":{"cuda":.1,"pin":.1},
               "input_gaps":gaps,"input_gap_hashes":{k:oracle.array_hash(v) for k,v in gaps.items()}}
    result=runner.aggregate_campaign(_campaign_rows(),bootstrap)
    assert result["all_oracle_gates_pass"] and len(result["tasks"])==12
    bad=_campaign_rows(); bad.pop()
    assert not runner.aggregate_campaign(bad,bootstrap)["all_oracle_gates_pass"]


def test_bootstrap_certificate_rebuilds_samples_and_rejects_any_mutation(monkeypatch):
    gaps={"cuda":np.linspace(.05,.12,8),"pin":np.linspace(.06,.13,8)}
    samples=np.tile(np.array([[.08],[.09]]),(1,oracle.BOOTSTRAP_RESAMPLES))
    fake={"seed":oracle.BOOTSTRAP_SEED,"resamples":oracle.BOOTSTRAP_RESAMPLES,
          "task_order":list(oracle.HELDOUT_TASK_SEEDS),"input_gaps":gaps,
          "input_gap_hashes":{m:oracle.array_hash(v) for m,v in gaps.items()},
          "lower95_relative":{"cuda":.08,"pin":.09},"samples":samples}
    monkeypatch.setattr(runner,"bootstrap_heldout_gaps",lambda _g:fake)
    inputs=np.stack([gaps["cuda"],gaps["pin"]])
    assert runner.certify_retained_bootstrap(gaps,inputs,samples)["passes"]
    bad=samples.copy(); bad[0,0]+=1e-12
    assert not runner.certify_retained_bootstrap(gaps,inputs,bad)["passes"]


def test_representative_tie_breaks_by_lowest_margin_index():
    rows=_campaign_rows()[:10]
    assert runner.representative(rows,"cuda","default")["margin_index"]==0


def test_transaction_gen0_and_exact120_synthetic_pairs(tmp_path,monkeypatch):
    output=tmp_path/"oracle.json"; stages=[]
    def capture(*args,**kwargs):
        stages.append((args[1],args[2],len(args[3]))); return {"mock":True}
    monkeypatch.setattr(runner,"_checkpoint",capture)
    monkeypatch.setattr(
        runner,"publish_acquisition",
        lambda _output,index,identity,_summary,_arrays:{"json_path":f"acq{index}.json","identity":list(identity)},
    )
    monkeypatch.setattr(runner,"publish_pair",lambda _o,i,identity,_a:{"json_path":f"pair{i}.json","identity":list(identity)})
    monkeypatch.setattr(runner,"authenticate_summary",lambda *_args,**_kwargs:True)
    monkeypatch.setattr(runner,"cross_bind_prerequisites",lambda *_:True)
    def acquisition(identity,*_):
        return {"passes":True,"finite":True},{name:np.zeros(shape,dtype) for name,(shape,dtype) in oracle.ROW_ARRAY_SPECS.items() if name in oracle.ACQUISITION_ARRAY_NAMES}
    def polish(payload):
        assert set(payload)==oracle.POLISH_INPUT_FIELDS
        return {"passes":True},{name:np.zeros(shape,dtype) for name,(shape,dtype) in oracle.ROW_ARRAY_SPECS.items() if name in oracle.POLISH_ARRAY_NAMES}
    def replay(*_):
        return {"artifact":{}},{name:np.zeros(shape,dtype) for name,(shape,dtype) in oracle.ROW_ARRAY_SPECS.items() if name in oracle.REPLAY_ARRAY_NAMES}
    model_arrays={
        "public_solver_x0_float32":np.zeros((12,14),dtype=np.float32),
        "public_reference_float32":np.zeros((12,10),dtype=np.float32),
        "public_default_side_int8":np.ones(12,dtype=np.int8),"quarantined_q8_float64":np.zeros((12,7)),
        "model_lower_float64":-np.ones(7),"model_upper_float64":np.ones(7),
        "model_velocity_float64":np.ones(7),"model_effort_float64":np.ones(7),
    }
    result=runner._run_pipeline(output,provenance={},task_loader=lambda:({},[],{}),model_loader=lambda:({},model_arrays),
                                acquisition_solver=acquisition,polish_solver=polish,
                                replay_solver=replay,
                                finish_provenance=lambda p:p.update(done=True),test_override=True)
    assert result["incomplete"] and len(result["rows"])==120
    assert stages[0]==(0,"before_prerequisite_loads",0)
    assert stages[-1]==(123,"end_provenance",120)
    assert len(stages)==124


@pytest.mark.parametrize("elapsed,trigger",[(181.,"projected_total"),(21600.,"total_deadline")])
def test_campaign_watchdog_rejects_after_first_pair_and_retains_partial(tmp_path,monkeypatch,elapsed,trigger):
    output=tmp_path/"oracle.json"; calls={"acquisition":0,"polish":0,"worker":0}; times=iter((0.,elapsed))
    monkeypatch.setattr(runner,"authenticate_summary",lambda *_a,**_k:True)
    monkeypatch.setattr(runner,"cross_bind_prerequisites",lambda *_:True)
    model={"public_solver_x0_float32":np.zeros((12,14),np.float32),"public_reference_float32":np.zeros((12,10),np.float32),
        "public_default_side_int8":np.ones(12,np.int8),"quarantined_q8_float64":np.zeros((12,7)),
        "model_lower_float64":-np.ones(7),"model_upper_float64":np.ones(7),
        "model_velocity_float64":np.ones(7),"model_effort_float64":np.ones(7)}
    def acquisition(*_):
        calls["acquisition"]+=1
        return {"finite":True,"passes":True},{name:np.zeros(shape,dtype) for name,(shape,dtype) in oracle.ROW_ARRAY_SPECS.items() if name in oracle.ACQUISITION_ARRAY_NAMES}
    def polish(_):
        calls["polish"]+=1
        return {"passes":True},{name:np.zeros(shape,dtype) for name,(shape,dtype) in oracle.ROW_ARRAY_SPECS.items() if name in oracle.POLISH_ARRAY_NAMES}
    def replay(*_):
        calls["worker"]+=1
        return {"artifact":{}},{name:np.zeros(shape,dtype) for name,(shape,dtype) in oracle.ROW_ARRAY_SPECS.items() if name in oracle.REPLAY_ARRAY_NAMES}
    deadlines=[]
    with pytest.raises(RuntimeError,match="permanently rejected"):
        runner._run_pipeline(output,provenance={},task_loader=lambda:({},[],{}),model_loader=lambda:({},model),
            acquisition_solver=acquisition,polish_solver=polish,replay_solver=replay,
            finish_provenance=lambda _p:None,test_override=True,monotonic=lambda:next(times),
            campaign_deadline_setter=deadlines.append)
    assert calls=={"acquisition":1,"polish":1,"worker":1} and deadlines==[oracle.CAMPAIGN_WALL_LIMIT_S]
    pointer=json.loads((tmp_path/"oracle.partial.latest.json").read_text())
    retained=json.loads(Path(pointer["json_path"]).read_text())
    assert retained["stage"]=="runtime_watchdog_rejected" and retained["completed_count"]==1
    assert len(retained["pending_identities"])==119 and retained["runtime_watchdog"]["trigger"]==trigger
    assert retained["runtime_watchdog"]["threshold_seconds"]==21600.
    assert retained["all_oracle_gates_pass"] is False and retained["runtime_watchdog"]["oracle_evidence"] is False
    assert not output.exists() and not output.with_suffix(".npz").exists()


def test_polish_boundary_rejects_every_oracle_taint_field():
    payload={"acquisition_primal_float64":np.zeros(oracle.Z_WIDTH),
             "public_x0_float32":np.zeros(14,dtype=np.float32),
             "public_reference_float32":np.zeros(10,dtype=np.float32),
             "joint_lower_float64":-np.ones(7),"joint_upper_float64":np.ones(7),
             "velocity_limit_float64":np.ones(7),"effort_limit_float64":np.ones(7),
             "solver_options":{**oracle.TRUST_CONSTR_OPTIONS,"maxiter":oracle.POLISH_MAXITER}}
    assert oracle.validate_unanchored_polish_input(payload)
    for field in ("q8","template","requested_side","default_side","anchor_weight","cylinder_center"):
        with pytest.raises(ValueError,match="tainted"):
            oracle.validate_unanchored_polish_input({**payload,field:0})


def test_failed_polish_retains_acquisition_side_and_no_final(tmp_path,monkeypatch):
    output=tmp_path/"oracle.json"; monkeypatch.setattr(runner,"authenticate_summary",lambda *_a,**_k:True)
    monkeypatch.setattr(runner,"cross_bind_prerequisites",lambda *_:True)
    monkeypatch.setattr(runner,"_checkpoint",lambda *_a,**_k:{})
    model={"public_solver_x0_float32":np.zeros((12,14),np.float32),
           "public_reference_float32":np.zeros((12,10),np.float32),
           "public_default_side_int8":np.ones(12,dtype=np.int8),"quarantined_q8_float64":np.zeros((12,7)),
           "model_lower_float64":-np.ones(7),"model_upper_float64":np.ones(7),
           "model_velocity_float64":np.ones(7),"model_effort_float64":np.ones(7)}
    def acquisition(*_): return {"finite":True,"passes":True},{name:np.zeros(shape,dtype) for name,(shape,dtype) in oracle.ROW_ARRAY_SPECS.items() if name in oracle.ACQUISITION_ARRAY_NAMES}
    calls=[]
    def polish(_): calls.append(1); raise RuntimeError("synthetic polish failure")
    result=runner._run_pipeline(output,provenance={},task_loader=lambda:({},[],{}),
        model_loader=lambda:({},model),acquisition_solver=acquisition,polish_solver=polish,
        replay_solver=lambda *_:pytest.fail("replay after failed polish"),
        finish_provenance=lambda _p:None,test_override=True)
    assert len(calls)==120 and all(row["polish_attempted"] for row in result["rows"])
    assert all(Path(row["acquisition_side_artifact"]["json_path"]).is_file() for row in result["rows"])
    assert not output.exists() and not output.with_suffix(".npz").exists()


def test_false_certificate_never_publishes_final(tmp_path):
    output=tmp_path/"oracle.json"
    with pytest.raises(RuntimeError,match="certificate failed"):
        runner._finalize(output,[],{}, {}, {"all_oracle_gates_pass":False})
    assert not output.exists() and not output.with_suffix(".npz").exists()


def test_final_summary_manifest_pointer_boundary_is_exact_and_canonical(tmp_path):
    summary_path=tmp_path/"oracle.json"; npz_path=tmp_path/"oracle.npz"; manifest_path=tmp_path/"oracle.manifest.json"
    npz_path.write_bytes(b"npz")
    canonical=[{"identity":["synthetic"]}]
    summary={"protocol":oracle.ORACLE_PROTOCOL_VERSION,"incomplete":False,"all_oracle_gates_pass":True,
             "canonical_rows":canonical,"npz_path":str(npz_path),"npz_sha256":runner.sha256_file(npz_path)}
    summary_path.write_text(json.dumps(summary))
    manifest={"protocol":oracle.ORACLE_PROTOCOL_VERSION,"incomplete":False,"all_oracle_gates_pass":True,
              "json_path":str(summary_path),"json_sha256":runner.sha256_file(summary_path),
              "npz_path":str(npz_path),"npz_sha256":runner.sha256_file(npz_path)}
    manifest_path.write_text(json.dumps(manifest))
    pointer={"generation":123,"incomplete":False,"superseded_by":str(summary_path),"json_path":str(summary_path),
             "json_sha256":runner.sha256_file(summary_path),"npz_path":str(npz_path),"npz_sha256":runner.sha256_file(npz_path),
             "manifest_path":str(manifest_path),"manifest_sha256":runner.sha256_file(manifest_path)}
    recomputed={"canonical_rows":canonical}
    assert runner.certify_final_document_boundary(summary,manifest,pointer,summary_path,npz_path,manifest_path,recomputed)["passes"]
    mutations=((summary,"protocol","wrong"),(summary,"canonical_rows",[]),(manifest,"npz_path","wrong"),
               (summary,"npz_path","wrong"),(summary,"npz_sha256","0"*64),
               (pointer,"superseded_by","wrong"),(pointer,"json_sha256","0"*64))
    for target,key,value in mutations:
        changed=dict(target); changed[key]=value
        args=(changed,manifest,pointer) if target is summary else (summary,changed,pointer) if target is manifest else (summary,manifest,changed)
        assert not runner.certify_final_document_boundary(*args,summary_path,npz_path,manifest_path,recomputed)["passes"]


def test_disk_recert_owns_pinned_prerequisite_authentication_and_problem_factory(monkeypatch):
    final={name:np.zeros(shape,dtype) for name,(shape,dtype) in oracle.BASE_ARRAY_SPECS.items()}
    final["joint_lower_float64"][:]=-1; final["joint_upper_float64"][:]=1
    final["velocity_limit_float64"][:]=1; final["effort_limit_float64"][:]=1
    model={"public_solver_x0_float32":final["public_x0_float32"],"public_reference_float32":final["public_reference_float32"],
        "public_default_side_int8":final["public_default_side_int8"],"quarantined_q8_float64":final["quarantined_q8_float64"],
        "model_lower_float64":final["joint_lower_float64"],"model_upper_float64":final["joint_upper_float64"],
        "model_velocity_float64":final["velocity_limit_float64"],"model_effort_float64":final["effort_limit_float64"]}
    calls=[]
    monkeypatch.setattr(runner,"load_authenticated_task_artifact",lambda:calls.append("task") or ({},[],{}))
    monkeypatch.setattr(runner,"load_authenticated_model_artifact",lambda:calls.append("model") or ({},model))
    monkeypatch.setattr(runner,"cross_bind_prerequisites",lambda *_:calls.append("cross") or True)
    class Pin:
        kinematics=staticmethod(lambda q:(np.zeros(3),np.zeros((3,7))))
        aba_derivatives=staticmethod(lambda q,v,u:(np.zeros(7),np.zeros((7,7)),np.zeros((7,7)),np.eye(7)))
        tool_velocity_derivatives=staticmethod(lambda q,v:(np.zeros(3),np.zeros((3,7)),np.zeros((3,7))))
        rnea=staticmethod(lambda *_:np.zeros(7))
        dense_replay=staticmethod(lambda x0,u:(np.repeat(np.asarray(x0)[None,:],oracle.DENSE_SAMPLES,axis=0),np.zeros((oracle.DENSE_SAMPLES,3))))
    monkeypatch.setattr(runner,"PinOracleModel",Pin)
    factory=runner._authenticated_recertification_factory(final)
    assert type(factory(0)) is oracle.OriginalOracleNLP and calls==["task","model","cross"]
    bad={**final,"quarantined_q8_float64":final["quarantined_q8_float64"].copy()}; bad["quarantined_q8_float64"][0,0]=1.
    with pytest.raises(RuntimeError,match="final arrays"): runner._authenticated_recertification_factory(bad)
    monkeypatch.setattr(runner,"cross_bind_prerequisites",lambda *_:False)
    with pytest.raises(RuntimeError,match="cross-bind"): runner._authenticated_recertification_factory(final)
    assert "problem_factory" not in inspect.signature(runner.recertify_retained_oracle).parameters


def test_provenance_gate_rejects_head_command_source_and_count_mutations():
    hashes={label:{"path":path,"sha256":"a"*64} for label,path in oracle.REQUIRED_SOURCE_PATHS.items()}
    p={"git_head_at_start":"b"*40,"git_head_at_end":"b"*40,
       "tracked_tree_clean_at_start":True,"tracked_tree_clean_at_end":True,
       "cwd":str(oracle.AUTHORIZED_CWD),"orig_argv":list(oracle.AUTHORIZED_ORIG_ARGV),
       "exact_command":runner.shlex.join(oracle.AUTHORIZED_ORIG_ARGV),
       "source_hashes_at_start":hashes,"source_hashes_at_end":hashes,
       "python_version":"3","numpy_version":"1","scipy_version":"1","pinocchio_version":"1",
       "extension_path":str(oracle.CUDA_EXTENSION_PATH),"extension_sha256":oracle.CUDA_EXTENSION_SHA256,
       "extension_size_bytes":oracle.CUDA_EXTENSION_SIZE_BYTES,"extension_build_head":oracle.CUDA_BUILD_HEAD,
       "extension_sha256_at_end":oracle.CUDA_EXTENSION_SHA256,"extension_size_bytes_at_end":oracle.CUDA_EXTENSION_SIZE_BYTES,
       "cuda_arch":oracle.CUDA_ARCH,"expected_module_attributes":oracle.CUDA_MODULE_ATTRIBUTES,
       "task_artifact_pins":{k:{"path":str(path),"sha256":digest} for k,(path,digest) in oracle.TASK_ARTIFACT_PINS.items()},
       "model_artifact_pins":{k:{"path":str(path),"sha256":digest} for k,(path,digest) in oracle.MODEL_ARTIFACT_PINS.items()},
       "task_artifact_loads":1,"model_artifact_loads":1,"task_rng_calls":0,
       "task_construction_calls":0,"initializer_calls":0,
       "acquisition_optimizer_calls":120,"polish_optimizer_calls":120,
       "worker_subprocess_calls":120,"cuda_replay_calls":120,
       "sqp_optimization_calls":0,"bootstrap_rng_calls":1,
       "all_compact_dls_binding_precheck_pass":True,
       "pin_derivative_diagnostics":{"all_pin_derivative_diagnostics_pass":True}}
    assert runner.certify_provenance(p)
    for key,value in (("git_head_at_end","c"*40),("cwd","/tmp"),
                      ("bootstrap_rng_calls",2),("task_rng_calls",1)):
        bad={**p,key:value}
        assert not runner.certify_provenance(bad)


def test_atomic_checkpoint_retains_expected_pending_ledger_and_false_gate(tmp_path):
    output=tmp_path/"oracle.json"
    payload=runner._checkpoint(output,0,"before_prerequisite_loads",[],{"huge":np.zeros((100,100))}, {})
    pointer=json.loads((tmp_path/"oracle.partial.latest.json").read_text())
    assert payload["incomplete"] and not payload["all_oracle_gates_pass"]
    assert payload["completed_count"]==0
    assert len(payload["pending_identities"])==120
    assert pointer["generation"]==0 and pointer["incomplete"]
    assert Path(pointer["json_path"]).is_file() and Path(pointer["npz_path"]).is_file()
    with np.load(pointer["npz_path"],allow_pickle=False) as archive:
        assert archive.files==["completed_count_int64"] and archive["completed_count_int64"].shape==()


def test_pair_side_artifact_requires_exact_canonical_row_array_map(tmp_path):
    arrays={name:np.zeros(shape,dtype) for name,(shape,dtype) in oracle.ROW_ARRAY_SPECS.items()}
    artifact=runner.publish_pair(tmp_path/"oracle.json",0,oracle.EXPECTED_LEDGER[0],arrays)
    assert Path(artifact["json_path"]).is_file() and Path(artifact["npz_path"]).is_file()
    for mutation in ("missing","extra","dtype"):
        bad={name:value.copy() for name,value in arrays.items()}
        if mutation=="missing": bad.pop("polish_primal_float64")
        elif mutation=="extra": bad["extra"]=np.zeros(1)
        else: bad["polish_primal_float64"]=bad["polish_primal_float64"].astype(np.float32)
        with pytest.raises(RuntimeError,match="schema"):
            runner.publish_pair(tmp_path/f"{mutation}.json",0,oracle.EXPECTED_LEDGER[0],bad)


def test_disk_side_chain_reloads_semantics_and_rejects_hash_consistent_pair_mutation(tmp_path,monkeypatch):
    identity=oracle.EXPECTED_LEDGER[0]; monkeypatch.setattr(runner,"EXPECTED_LEDGER",(identity,))
    monkeypatch.setattr(runner,"validate_request",lambda _request:True)
    output=tmp_path/"oracle.json"
    final_arrays={name:np.zeros(shape,dtype) for name,(shape,dtype) in oracle.BASE_ARRAY_SPECS.items()}
    reference=_reference(); reference[:3]=0.; reference[3:5]=[.1,.1]
    final_arrays["public_reference_float32"][0]=reference
    final_arrays["joint_lower_float64"][:]=-1; final_arrays["joint_upper_float64"][:]=1
    final_arrays["velocity_limit_float64"][:]=1; final_arrays["effort_limit_float64"][:]=1
    retained={name:np.zeros(shape,dtype) for name,(shape,dtype) in oracle.ROW_ARRAY_SPECS.items()}
    retained["dense_time_float64"][:]=np.arange(oracle.DENSE_SAMPLES)*oracle.DT/oracle.DENSE_SUBSTEPS
    for name,value in retained.items(): final_arrays[f"row_000_{name}"]=value
    acquisition={name:retained[name] for name in oracle.ACQUISITION_ARRAY_NAMES}
    acq_artifact=runner.publish_acquisition(output,0,identity,{"finite":True,"success":True},acquisition)
    pair_artifact=runner.publish_pair(output,0,identity,retained)
    request_path=tmp_path/"oracle.worker.0000.request.json"; input_path=tmp_path/"oracle.worker.0000.input.npz"
    worker_json=tmp_path/"oracle.worker.0000.json"; worker_npz=tmp_path/"oracle.worker.0000.npz"
    worker_input={"x0_float32":final_arrays["public_x0_float32"][0],"reference_float32":final_arrays["public_reference_float32"][0],
        "controls_float32":retained["controls_float64"].astype(np.float32),"joint_lower_float64":final_arrays["joint_lower_float64"],
        "joint_upper_float64":final_arrays["joint_upper_float64"],"velocity_limit_float64":final_arrays["velocity_limit_float64"],
        "effort_limit_float64":final_arrays["effort_limit_float64"]}
    runner._atomic_npz(input_path,worker_input)
    request={"identity":list(identity),"request_json":str(request_path),"input_npz":str(input_path),
        "input_npz_sha256":runner.sha256_file(input_path),"extension_module":oracle.CUDA_EXTENSION_MODULE,
        "extension_path":str(oracle.CUDA_EXTENSION_PATH),"extension_sha256":oracle.CUDA_EXTENSION_SHA256,
        "worker_protocol":worker.WORKER_PROTOCOL_VERSION,"output_json":str(worker_json),"output_npz":str(worker_npz)}
    runner._atomic_json(request_path,request)
    worker_arrays=_worker_arrays(); worker_arrays["captured_reference_float32"][:]=final_arrays["public_reference_float32"][0]
    runner._atomic_npz(worker_npz,worker_arrays)
    worker_summary={**_worker_summary(),"identity":list(identity),"request_path":str(request_path),
        "request_sha256":runner.sha256_file(request_path),"input_npz":str(input_path),
        "input_npz_sha256":runner.sha256_file(input_path),"extension_path":str(oracle.CUDA_EXTENSION_PATH),
        "extension_sha256":oracle.CUDA_EXTENSION_SHA256,"array_names":sorted(worker_arrays),
        "array_hashes":{k:oracle.array_hash(v) for k,v in worker_arrays.items()},"npz_sha256":runner.sha256_file(worker_npz)}
    worker_summary["certificate"]=worker.certify_replay(worker_summary,worker_arrays)
    runner._atomic_json(worker_json,worker_summary)
    artifact={"request_path":str(request_path),"request_sha256":runner.sha256_file(request_path),"input_path":str(input_path),
        "input_sha256":runner.sha256_file(input_path),"json_path":str(worker_json),"json_sha256":runner.sha256_file(worker_json),
        "npz_path":str(worker_npz),"npz_sha256":runner.sha256_file(worker_npz)}
    row={"identity":list(identity),"acquisition":{"finite":True,"success":True},
         "acquisition_side_artifact":acq_artifact,"pair_side_artifact":pair_artifact,
         "worker_artifact":artifact,
         "worker":{**worker_summary,"artifact":artifact,"command":["python","-B","-m","gato_tiago.multimodal_toll_oracle_v1_worker",
                    "--request",str(request_path),"--output",str(worker_json)],"cwd":str(oracle.AUTHORIZED_CWD),"exit_code":0}}
    base_counts={"task_artifact_loads":0,"model_artifact_loads":0,"acquisition_optimizer_calls":0,"polish_optimizer_calls":0,
                 "worker_subprocess_calls":0,"cuda_replay_calls":0,"bootstrap_rng_calls":0,
                 "git_head_at_end":None,"tracked_tree_clean_at_end":None,"full_git_status_at_end":None,
                 "source_hashes_at_end":None,"extension_sha256_at_end":None,"extension_size_bytes_at_end":None}
    runner._checkpoint(output,0,"before_prerequisite_loads",[],{},dict(base_counts))
    p={**base_counts,"task_artifact_loads":1}; runner._checkpoint(output,1,"task_authenticated",[],{},p)
    p={**p,"model_artifact_loads":1}; runner._checkpoint(output,2,"model_authenticated",[],{},p)
    p={**p,"acquisition_optimizer_calls":1,"polish_optimizer_calls":1,"worker_subprocess_calls":1,"cuda_replay_calls":1}
    runner._checkpoint(output,3,"pair_completed",[row],{},p)
    p={**p,"bootstrap_rng_calls":1}; runner._checkpoint(output,4,"end_provenance",[row],{},p)
    side_paths=sorted([*tmp_path.glob("oracle.partial.[0-9][0-9][0-9][0-9].json"),
                       *tmp_path.glob("oracle.partial.[0-9][0-9][0-9][0-9].npz"),*tmp_path.glob("oracle.acquisition.*"),
                       *tmp_path.glob("oracle.pair.*"),*tmp_path.glob("oracle.worker.*")])
    manifest={"all_side_artifacts":[{"path":str(path),"sha256":runner.sha256_file(path)} for path in side_paths]}
    summary={"rows":[row],"provenance":p}
    detail=runner._recertify_side_chain(output,summary,final_arrays,manifest)
    assert detail["passes"], detail["worker_diagnostics"]
    swapped=json.loads(json.dumps(summary)); swapped["rows"][0]["acquisition_side_artifact"]=pair_artifact
    assert not runner._recertify_side_chain(output,swapped,final_arrays,manifest)["passes"]
    duplicated=json.loads(json.dumps(summary)); duplicated["rows"][0]["worker_artifact"]["json_sha256"]="0"*64
    assert not runner._recertify_side_chain(output,duplicated,final_arrays,manifest)["passes"]
    original_worker_json=worker_json.read_bytes(); lying=json.loads(original_worker_json); lying["extension_path"]="wrong"
    worker_json.write_text(json.dumps(lying,sort_keys=True)); refreshed_summary=json.loads(json.dumps(summary))
    refreshed_summary["rows"][0]["worker"]["extension_path"]="wrong"
    refreshed_hash=runner.sha256_file(worker_json)
    refreshed_summary["rows"][0]["worker"]["artifact"]["json_sha256"]=refreshed_hash
    refreshed_summary["rows"][0]["worker_artifact"]["json_sha256"]=refreshed_hash
    for item in manifest["all_side_artifacts"]:
        if item["path"]==str(worker_json): item["sha256"]=refreshed_hash
    assert not runner._recertify_side_chain(output,refreshed_summary,final_arrays,manifest)["passes"]
    worker_json.write_bytes(original_worker_json)
    for item in manifest["all_side_artifacts"]:
        if item["path"]==str(worker_json): item["sha256"]=runner.sha256_file(worker_json)
    for path,field,value in ((Path(acq_artifact["json_path"]),"row_index",9),
                             (request_path,"identity",list(oracle.EXPECTED_LEDGER[1])),
                             (tmp_path/"oracle.partial.0003.json","completed_count",0)):
        original=path.read_bytes(); payload=json.loads(original); payload[field]=value
        path.write_text(json.dumps(payload,sort_keys=True))
        for item in manifest["all_side_artifacts"]:
            if item["path"]==str(path): item["sha256"]=runner.sha256_file(path)
        assert not runner._recertify_side_chain(output,summary,final_arrays,manifest)["passes"]
        path.write_bytes(original)
        for item in manifest["all_side_artifacts"]:
            if item["path"]==str(path): item["sha256"]=runner.sha256_file(path)
    checkpoint=tmp_path/"oracle.partial.0003.json"; original=checkpoint.read_bytes()
    for key,value in (("exact_command","mutated command"),("task_artifact_authentication_pass",False),("extra",1)):
        payload=json.loads(original); payload["provenance"][key]=value; checkpoint.write_text(json.dumps(payload,sort_keys=True))
        for item in manifest["all_side_artifacts"]:
            if item["path"]==str(checkpoint): item["sha256"]=runner.sha256_file(checkpoint)
        assert not runner._recertify_side_chain(output,summary,final_arrays,manifest)["passes"]
        checkpoint.write_bytes(original)
        for item in manifest["all_side_artifacts"]:
            if item["path"]==str(checkpoint): item["sha256"]=runner.sha256_file(checkpoint)
    # Exercise the public disk recertifier, not only its component helpers.
    fake_certificate={"canonical_rows":[{"identity":list(identity)}],"all_oracle_gates_pass":True}
    def authenticated_factory(arrays):
        if float(arrays["quarantined_q8_float64"][0,0])!=0.: raise RuntimeError("synthetic prerequisite mismatch")
        return lambda _index:None
    monkeypatch.setattr(runner,"_authenticated_recertification_factory",authenticated_factory)
    monkeypatch.setattr(runner,"certify_final_oracle",lambda *_a,**_k:fake_certificate)
    monkeypatch.setattr(runner,"exact_final_array_schema",lambda arrays:set(arrays)==set(final_arrays))
    runner._atomic_npz(output.with_suffix(".npz"),final_arrays)
    summary.update({"protocol":oracle.ORACLE_PROTOCOL_VERSION,"incomplete":False,"all_oracle_gates_pass":True,
        "canonical_rows":fake_certificate["canonical_rows"],"certificate":fake_certificate,
        "array_names":sorted(final_arrays),"array_hashes":{k:oracle.array_hash(v) for k,v in final_arrays.items()}})
    summary["npz_path"]=str(output.with_suffix(".npz")); summary["npz_sha256"]=runner.sha256_file(summary["npz_path"])
    runner._atomic_json(output,summary)
    manifest.update({"protocol":oracle.ORACLE_PROTOCOL_VERSION,"incomplete":False,"all_oracle_gates_pass":True,
        "json_path":str(output),"json_sha256":runner.sha256_file(output),"npz_path":summary["npz_path"],
        "npz_sha256":summary["npz_sha256"],"side_artifact_count":len(manifest["all_side_artifacts"]),
        "acquisition_side_artifacts":[acq_artifact]})
    manifest_path=output.with_suffix(".manifest.json"); runner._atomic_json(manifest_path,manifest)
    pointer_path=tmp_path/"oracle.final.latest.json"
    runner._atomic_json(pointer_path,{"generation":123,"incomplete":False,"superseded_by":str(output),
        "json_path":str(output),"json_sha256":runner.sha256_file(output),"npz_path":summary["npz_path"],
        "npz_sha256":summary["npz_sha256"],"manifest_path":str(manifest_path),"manifest_sha256":runner.sha256_file(manifest_path)})
    monkeypatch.setattr(runner,"EXPECTED_SIDE_ARTIFACT_COUNT",len(manifest["all_side_artifacts"]))
    recert=runner.recertify_retained_oracle(output,summary["npz_path"],manifest_path,pointer_path)
    assert recert["all_disk_recertification_gates_pass"], recert
    original_summary=output.read_bytes(); original_npz=Path(summary["npz_path"]).read_bytes()
    original_manifest=manifest_path.read_bytes(); original_pointer=pointer_path.read_bytes()
    mutated_arrays={k:v.copy() for k,v in final_arrays.items()}; mutated_arrays["quarantined_q8_float64"][0,0]=1.
    runner._atomic_npz(summary["npz_path"],mutated_arrays)
    mutated_summary=json.loads(original_summary); mutated_summary["npz_sha256"]=runner.sha256_file(summary["npz_path"])
    mutated_summary["array_hashes"]["quarantined_q8_float64"]=oracle.array_hash(mutated_arrays["quarantined_q8_float64"])
    output.write_text(json.dumps(mutated_summary,sort_keys=True))
    mutated_manifest=json.loads(original_manifest); mutated_manifest["json_sha256"]=runner.sha256_file(output)
    mutated_manifest["npz_sha256"]=runner.sha256_file(summary["npz_path"]); manifest_path.write_text(json.dumps(mutated_manifest,sort_keys=True))
    mutated_pointer=json.loads(original_pointer); mutated_pointer["json_sha256"]=runner.sha256_file(output)
    mutated_pointer["npz_sha256"]=runner.sha256_file(summary["npz_path"]); mutated_pointer["manifest_sha256"]=runner.sha256_file(manifest_path)
    pointer_path.write_text(json.dumps(mutated_pointer,sort_keys=True))
    assert not runner.recertify_retained_oracle(output,summary["npz_path"],manifest_path,pointer_path)["all_disk_recertification_gates_pass"]
    output.write_bytes(original_summary); Path(summary["npz_path"]).write_bytes(original_npz)
    manifest_path.write_bytes(original_manifest); pointer_path.write_bytes(original_pointer)
    mutated=json.loads(original_summary); mutated["canonical_rows"]=[]; output.write_text(json.dumps(mutated,sort_keys=True))
    refreshed=json.loads(original_manifest); refreshed["json_sha256"]=runner.sha256_file(output); manifest_path.write_text(json.dumps(refreshed,sort_keys=True))
    refreshed_pointer=json.loads(original_pointer); refreshed_pointer["json_sha256"]=runner.sha256_file(output)
    refreshed_pointer["manifest_sha256"]=runner.sha256_file(manifest_path); pointer_path.write_text(json.dumps(refreshed_pointer,sort_keys=True))
    assert not runner.recertify_retained_oracle(output,summary["npz_path"],manifest_path,pointer_path)["all_disk_recertification_gates_pass"]
    output.write_bytes(original_summary); manifest_path.write_bytes(original_manifest); pointer_path.write_bytes(original_pointer)
    mutated=json.loads(original_summary); mutated["array_names"].append(mutated["array_names"][0]); output.write_text(json.dumps(mutated,sort_keys=True))
    refreshed=json.loads(original_manifest); refreshed["json_sha256"]=runner.sha256_file(output); manifest_path.write_text(json.dumps(refreshed,sort_keys=True))
    refreshed_pointer=json.loads(original_pointer); refreshed_pointer["json_sha256"]=runner.sha256_file(output)
    refreshed_pointer["manifest_sha256"]=runner.sha256_file(manifest_path); pointer_path.write_text(json.dumps(refreshed_pointer,sort_keys=True))
    assert not runner.recertify_retained_oracle(output,summary["npz_path"],manifest_path,pointer_path)["all_disk_recertification_gates_pass"]
    output.write_bytes(original_summary); manifest_path.write_bytes(original_manifest); pointer_path.write_bytes(original_pointer)
    mutated=json.loads(original_summary); mutated["array_hashes"]["extra"]="0"*64; output.write_text(json.dumps(mutated,sort_keys=True))
    refreshed=json.loads(original_manifest); refreshed["json_sha256"]=runner.sha256_file(output); manifest_path.write_text(json.dumps(refreshed,sort_keys=True))
    refreshed_pointer=json.loads(original_pointer); refreshed_pointer["json_sha256"]=runner.sha256_file(output)
    refreshed_pointer["manifest_sha256"]=runner.sha256_file(manifest_path); pointer_path.write_text(json.dumps(refreshed_pointer,sort_keys=True))
    assert not runner.recertify_retained_oracle(output,summary["npz_path"],manifest_path,pointer_path)["all_disk_recertification_gates_pass"]
    output.write_bytes(original_summary); manifest_path.write_bytes(original_manifest); pointer_path.write_bytes(original_pointer)
    mutated_manifest=json.loads(original_manifest); mutated_manifest["protocol"]="wrong"; manifest_path.write_text(json.dumps(mutated_manifest,sort_keys=True))
    refreshed_pointer=json.loads(original_pointer); refreshed_pointer["manifest_sha256"]=runner.sha256_file(manifest_path)
    pointer_path.write_text(json.dumps(refreshed_pointer,sort_keys=True))
    assert not runner.recertify_retained_oracle(output,summary["npz_path"],manifest_path,pointer_path)["all_disk_recertification_gates_pass"]
    manifest_path.write_bytes(original_manifest); pointer_path.write_bytes(original_pointer)
    mutated_pointer=json.loads(original_pointer); mutated_pointer["superseded_by"]="wrong"; pointer_path.write_text(json.dumps(mutated_pointer,sort_keys=True))
    assert not runner.recertify_retained_oracle(output,summary["npz_path"],manifest_path,pointer_path)["all_disk_recertification_gates_pass"]
    pointer_path.write_bytes(original_pointer)
    pair_arrays=runner._load_npz_map(pair_artifact["npz_path"]); pair_arrays["polish_primal_float64"][0]=1.
    Path(pair_artifact["npz_path"]).unlink(); runner._atomic_npz(pair_artifact["npz_path"],pair_arrays)
    for item in manifest["all_side_artifacts"]:
        if item["path"]==pair_artifact["npz_path"]: item["sha256"]=runner.sha256_file(item["path"])
    assert not runner._recertify_side_chain(output,summary,final_arrays,manifest)["passes"]


def test_exact_final_array_name_set_is_closed_and_has_no_embedded_row_arrays():
    expected=len(oracle.BASE_ARRAY_SPECS)+oracle.EXPECTED_PAIR_COUNT*len(oracle.ROW_ARRAY_SPECS)
    assert len(runner.expected_final_array_names())==expected
    assert "row_000_cuda_dense_tool_float32" in runner.expected_final_array_names()
    assert "row_120_cuda_dense_tool_float32" not in runner.expected_final_array_names()


def test_execution_boundaries_are_blocked_without_touching_files(tmp_path,monkeypatch):
    calls=[]; monkeypatch.setattr(runner,"_production_pipeline",lambda *_a,**_k:calls.append("runner"))
    monkeypatch.setattr(worker,"_run_authorized_worker",lambda *_a:calls.append("worker"))
    with pytest.raises(RuntimeError,match="blocked"): runner.execute_oracle(tmp_path/"oracle.json",authorization=object())
    with pytest.raises(RuntimeError,match="blocked"): worker.execute_worker(tmp_path/"r.json",tmp_path/"o.json",authorization=object())
    assert calls==[] and list(tmp_path.iterdir())==[]


def test_rejected_v1_runner_stays_closed_before_private_pipeline(tmp_path,monkeypatch):
    calls=[]; monkeypatch.setattr(runner,"_production_pipeline",lambda path,**_k:calls.append(Path(path)) or {"ok":True})
    with pytest.raises(RuntimeError,match="blocked"):
        runner.execute_oracle(tmp_path/"oracle.json",authorization=object())
    assert calls==[]
    source=inspect.getsource(runner._no_existing)
    assert "refuses resume, overwrite, or rerun" in source and "worker.*" in source




def test_oracle_sources_do_not_open_task_rng_or_generate_tasks():
    sources="\n".join(Path(module.__file__).read_text() for module in (oracle,runner,worker))
    assert "generate_task(" not in sources
    assert sources.count("np.random.default_rng(BOOTSTRAP_SEED)")==1
    assert "TASK_CONSTRUCTION_AUTHORIZATION" not in sources
    assert "q8_initializer_eligible" in inspect.getsource(runner.cross_bind_prerequisites)
    public=Path(toll.__file__).read_text()
    assert "multimodal_toll_oracle_v1" not in public


def test_production_boundaries_are_concrete_but_statically_disabled():
    runner_source=inspect.getsource(runner._production_pipeline)
    worker_source=inspect.getsource(worker._run_authorized_worker)
    assert "ProductionOracleDependencies" in runner_source and "_run_pipeline" in runner_source
    assert "sim_forward" in worker_source and "tool_position" in worker_source
    assert "runtime is unavailable" not in runner_source+worker_source
    assert worker.SIM_FORWARD_CALLS==4032 and worker.TOOL_POSITION_CALLS==253


def test_dls_enumeration_is_not_repeated_in_prebootstrap_pass():
    assert "reenumerate=False" in inspect.getsource(runner.ProductionOracleDependencies.before_finish)
    assert "reenumerate=True" in inspect.getsource(runner._production_pipeline)
    assert "reenumerate=True" in inspect.getsource(runner.recertify_retained_oracle)


def test_solver_callbacks_share_campaign_deadline_without_changing_per_call_limits():
    source=inspect.getsource(runner._trust_constr)
    assert "now>=campaign_deadline" in source and "now-start > wall_limit" in source
    assert oracle.ACQUISITION_WALL_LIMIT_S==900. and oracle.POLISH_WALL_LIMIT_S==1800.
    assert oracle.CAMPAIGN_WALL_LIMIT_S==21600.
    assert "campaign_deadline=self.campaign_deadline" in inspect.getsource(runner.ProductionOracleDependencies.acquisition)
    assert "campaign_deadline=self.campaign_deadline" in inspect.getsource(runner.ProductionOracleDependencies.polish)


def test_unanchored_solver_source_has_no_template_q8_side_or_anchor_input():
    source=inspect.getsource(runner.solve_unanchored_polish)
    assert "q8" not in source and "template" not in source and "side" not in source
    assert "POLISH_MAXITER" in source and "validate_unanchored_polish_input" in source
    common=inspect.getsource(runner._trust_constr)
    assert "BFGS()" in common and '"sparse_jacobian":True' in common


def test_required_sources_and_generated_grids_are_not_oracle_outputs():
    root=Path(__file__).resolve().parents[2]
    assert all((root/path).is_file() for path in oracle.REQUIRED_SOURCE_PATHS.values())
    assert oracle.REQUIRED_SOURCE_PATHS["tiago_grid"]=="gato/dynamics/tiago_right/tiago_right_grid.cuh"
    assert not any("grid.cuh" in Path(module.__file__).name for module in (oracle,runner,worker))
