"""Independent Pinocchio wrench oracles and CPU force-backend contracts."""
import ast
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pinocchio as pin
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'python'))
from bsqp.wrench import WorldToolWrench,linear_to_angular_first,world_tool_to_joint
from bsqp.common import rk4
from experiments.force_adapter.backend import PinocchioPredictor,NativeOptimizer,transform_local,LOCAL_FRAME,LOCAL_POINT
from experiments.force_adapter.orchestration import AppliedSegment,TickInputs,WrenchConvention,PredictionRequest,TORequest


@pytest.fixture
def model():
    return pin.buildModelFromUrdf(str(ROOT/'gato/dynamics/tiago_right/tiago_right_arm.urdf'))


def method(relative,name):
    parsed=ast.parse((ROOT/relative).read_text())
    owner=next(n for n in parsed.body if isinstance(n,ast.ClassDef))
    return next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name==name)


def test_world_tool_force_matches_jacobian_and_rnea_at_distinct_poses(model):
    wrench=np.array([1.2,-.7,.4,.1,-.2,.3])
    center=(model.lowerPositionLimit+model.upperPositionLimit)/2
    for q in (center,center+np.linspace(-.1,.1,7)):
        jid,local=world_tool_to_joint(model,q,wrench)
        force=pin.StdVec_Force()
        for _ in range(model.njoints):force.append(pin.Force.Zero())
        force[jid]=pin.Force(local[:3],local[3:])
        tool=model.getFrameId('arm_right_tool_link')
        J=pin.computeFrameJacobian(model,model.createData(),q,tool,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        expected=J.T@wrench
        zeros=np.zeros(7)
        tau=pin.rnea(model,model.createData(),q,zeros,zeros)-pin.rnea(model,model.createData(),q,zeros,zeros,force)
        np.testing.assert_allclose(tau,expected,atol=2e-12,rtol=2e-12)
        assert jid==model.njoints-1
        data=model.createData();pin.forwardKinematics(model,data,q)
        unshifted=np.r_[data.oMi[jid].rotation.T@wrench[:3],data.oMi[jid].rotation.T@wrench[3:]]
        assert np.max(abs(local-unshifted))>1e-4


@pytest.mark.parametrize('model_path',['examples/indy7_description/indy7.urdf','examples/iiwa_description/iiwa14.urdf'])
def test_generic_model_world_tool_transform_oracle(model_path):
    model=pin.buildModelFromUrdf(str(ROOT/model_path))
    assert model.existFrame('EE')
    q=pin.neutral(model);w=np.array([2.,3.,4.,.2,.3,.4])
    jid,local=world_tool_to_joint(model,q,w)
    forces=pin.StdVec_Force()
    for _ in range(model.njoints):forces.append(pin.Force.Zero())
    forces[jid]=pin.Force(local[:3],local[3:])
    zeros=np.zeros(model.nv)
    actual=pin.rnea(model,model.createData(),q,zeros,zeros)-pin.rnea(model,model.createData(),q,zeros,zeros,forces)
    jac=pin.computeFrameJacobian(model,model.createData(),q,model.getFrameId('EE'),pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    np.testing.assert_allclose(actual,jac.T@w,atol=2e-12,rtol=2e-12)


@pytest.mark.parametrize('relative',['python/bsqp/mpc_controller.py','tiago_src/gato_tiago/tiago_mpc_controller.py'])
def test_existing_controller_transform_uses_repaired_frame_point(model,relative):
    node=method(relative,'transform_force_to_gato_frame')
    namespace=dict(world_tool_to_joint=world_tool_to_joint)
    exec(compile(ast.Module(body=[node],type_ignores=[]),relative,'exec'),namespace)
    controller=SimpleNamespace(solver_model=model,nq_robot=7)
    q=(model.lowerPositionLimit+model.upperPositionLimit)/2;w=np.array([2.,3.,4.,.2,.3,.4])
    np.testing.assert_allclose(namespace[node.name](controller,q,w),world_tool_to_joint(model,q,w)[1])


def test_public_interface_explicit_component_order_and_invalid_batch():
    node=method('python/bsqp/interface.py','set_f_ext_B')
    namespace=dict(np=np,linear_to_angular_first=linear_to_angular_first)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'interface','exec'),namespace)
    calls=[];owner=SimpleNamespace(batch_size=2,solver=SimpleNamespace(set_f_ext_batch=calls.append))
    w=np.arange(12,dtype=np.float32).reshape(2,6)
    namespace[node.name](owner,w)
    np.testing.assert_array_equal(calls[0],w[:,[3,4,5,0,1,2]])
    assert calls[0].flags.c_contiguous
    np.testing.assert_array_equal(owner.f_ext_B,w)
    for invalid in (np.zeros((1,6)),np.full((2,6),np.nan)):
        with pytest.raises(ValueError):namespace[node.name](owner,invalid)


def test_body_fixed_pinocchio_derivatives_and_zero_path(model):
    q=(model.lowerPositionLimit+model.upperPositionLimit)/2
    v=np.linspace(-.08,.08,7);u=np.linspace(-1.,1.,7)
    force=pin.StdVec_Force()
    for _ in range(model.njoints):force.append(pin.Force.Zero())
    force[-1]=pin.Force(np.array([1.2,-.7,.4]),np.array([.1,-.2,.3]))
    data=model.createData();derivatives=[x.copy() for x in pin.computeABADerivatives(model,data,q,v,u,force)]
    for idx,(value,expected) in enumerate(zip((q,v,u),derivatives)):
        columns=[]
        for j in range(7):
            delta=np.eye(7)[j]*1e-6
            plus=[q,v,u];minus=[q,v,u];plus[idx]=value+delta;minus[idx]=value-delta
            columns.append((pin.aba(model,model.createData(),*plus,force)-pin.aba(model,model.createData(),*minus,force))/2e-6)
        np.testing.assert_allclose(np.column_stack(columns),expected,rtol=2e-6,atol=3e-6)
    explicit=pin.StdVec_Force()
    for _ in range(model.njoints):explicit.append(pin.Force.Zero())
    a=rk4(model,model.createData(),q,v,u,.003)
    b=rk4(model,model.createData(),q,v,u,.003,explicit)
    for actual,expected in zip(a,b):np.testing.assert_array_equal(actual,expected)


def test_world_wrench_reexpressed_at_every_rk_stage(model):
    q=(model.lowerPositionLimit+model.upperPositionLimit)/2
    v=np.linspace(-.08,.08,7);u=np.zeros(7);seen=[]
    transform=WorldToolWrench(model,np.array([1.2,-.7,.4,.1,-.2,.3]))
    def observe(state):seen.append(state.copy());return transform(state)
    result=rk4(model,model.createData(),q,v,u,.003,observe)
    assert len(seen)==4 and np.max(abs(seen[-1]-seen[0]))>0
    assert all(np.isfinite(x).all() for x in result)


def test_predictor_replays_all_applied_segments_with_body_fixed_force(model):
    convention=WrenchConvention(LOCAL_FRAME,LOCAL_POINT,LOCAL_FRAME,LOCAL_POINT)
    q=(model.lowerPositionLimit+model.upperPositionLimit)/2;state=np.r_[q,np.zeros(7)]
    controls=(AppliedSegment(0.,.003,np.zeros(7)),AppliedSegment(.003,.006,np.ones(7)*.05))
    forces=np.array([[1.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.]])
    tick=TickInputs(0.,.006,state,state,controls,np.tile(state,(2,1)),np.zeros((2,192)),np.zeros((2,665)))
    request=PredictionRequest(tick,forces,convention,transform_local)
    predictions=PinocchioPredictor(model)(request)
    assert predictions.shape==(2,14) and np.max(abs(predictions[0]-predictions[1]))>1e-6
    assert np.isfinite(predictions).all()
    invalid=WrenchConvention('world','tool origin',LOCAL_FRAME,LOCAL_POINT)
    with pytest.raises(ValueError):transform_local(state,forces[0],invalid)


def test_native_owner_persists_duals_and_keeps_setup_memory_outside_ticks(monkeypatch):
    import bsqp.interface as interface
    log=[]
    class Owner:
        def __init__(self,*args,**kwargs):
            log.append('construct')
            self.solver=SimpleNamespace(solve=lambda warm,dt,state,ref:dict(XU=warm.copy(),sqp_time_us=123.),
                workspace_bytes=lambda:dict(device_bytes=1234,pinned_host_bytes=44))
        def reset_dual(self):log.append('dual')
        def reset_rho(self):log.append('rho')
        def set_f_ext_B(self,forces):log.append(forces.copy())
        def solve(self,state,ref,warm):
            reply=self.solver.solve(warm,.03,state,ref)
            return reply['XU'],reply['sqp_time_us']
        def get_stats(self):return {'sqp_iters':np.array([2])}
    monkeypatch.setattr(interface,'BSQP',Owner)
    def memory(stage):
        return dict(stage=stage,cuda_status=0,free_bytes=1000 if stage in ('before_owner','after_close') else 600,
                    total_bytes=2000,query_wall_s=.001,host_peak_RSS_bytes=50)
    convention=WrenchConvention(LOCAL_FRAME,LOCAL_POINT,LOCAL_FRAME,LOCAL_POINT)
    request=TORequest((3,),np.zeros((1,14)),np.zeros((1,192)),np.zeros((1,665)),
                      np.array([[1.,2.,3.,4.,5.,6.]]),np.array([[1.,2.,3.,4.,5.,6.]]),convention)
    backend=NativeOptimizer('unused',{},memory_query=memory)
    backend.reset_dual()
    assert backend(request).shape==(1,665)
    assert backend(request).shape==(1,665)
    record=backend.measurements[0]
    assert record['native_solve_us']==123.
    assert backend.construction_wall_s>=0 and record['public_solve_wall_s']>=0
    counts={name:sum(isinstance(entry,str) and entry==name for entry in log) for name in ('construct','dual','rho')}
    assert counts==dict(construct=1,dual=1,rho=2)
    assert backend.workspace==dict(device_bytes=1234,pinned_host_bytes=44)
    assert 'memory' not in record and 'constructor_wall_s' not in record
    assert backend.native_records[0]['native_reply']['XU'].shape==(1,665)


@pytest.mark.parametrize('mode',['original','adapted'])
def test_gpu_prediction_uses_previous_observation_elapsed_interval_and_weighted_control(mode):
    from experiments.force_adapter.backend import GPUPredictor
    calls=[]
    def simulate(state,control,forces,dt):
        calls.append((np.asarray(state).copy(),np.asarray(control).copy(),forces.copy(),dt))
        return np.tile(state,(16,1))
    class Simulation:
        def sim_forward(self,*args):return simulate(*args)
        def workspace_bytes(self):return dict(device_bytes=4804,pinned_host_bytes=0)
    optimizer=SimpleNamespace(owner=SimpleNamespace(lib=SimpleNamespace(Simulation16_float=Simulation)),predict=simulate)
    predictor=GPUPredictor(optimizer,mode)
    previous=np.arange(14,dtype=float);observed=previous+1
    history=(AppliedSegment(0.,.018,np.ones(7)*2),AppliedSegment(.018,.048,np.ones(7)*4))
    tick=TickInputs(0.,.048,previous,observed,history,np.tile(observed,(16,1)),np.zeros((16,192)),np.zeros((16,665)))
    forces=np.zeros((16,6));forces[3]=[1,2,3,.1,.2,.3]
    convention=WrenchConvention(LOCAL_FRAME,LOCAL_POINT,LOCAL_FRAME,LOCAL_POINT)
    predictor(PredictionRequest(tick,forces,convention,transform_local))
    state,control,hypotheses,dt=calls[0]
    np.testing.assert_allclose(state,previous);np.testing.assert_allclose(control,np.ones(7)*3.25)
    np.testing.assert_allclose(hypotheses,forces);assert dt==.048
    assert predictor.last_observation['elapsed_interval_s']==.048


def test_attached_force_initially_points_world_down_and_has_zero_torque(model):
    from experiments.force_adapter.backend import attached_force,force_world_direction
    state=np.r_[(model.lowerPositionLimit+model.upperPositionLimit)/2,np.zeros(7)]
    truth=attached_force(model,state,10.)
    np.testing.assert_allclose(truth[3:],0);assert np.linalg.norm(truth[:3])==pytest.approx(10.)
    result=force_world_direction(model,state[:7],truth)
    np.testing.assert_allclose(result['world_force_N'],[0,0,-10],atol=1e-10)
    assert result['deviation_from_world_down_deg']==pytest.approx(0.)


def test_successor_bank_preserves_starts_hypotheses_and_old_files(tmp_path):
    import hashlib,json
    from experiments.force_adapter.freeze_native import prepare
    old=ROOT/'experiments/force_adapter/native_inputs'
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in old.iterdir() if p.is_file()}
    output=tmp_path/'synthetic-unqualified10N-bank'
    prepare(output,10.,old)
    with np.load(old/'arrays.npz') as original,np.load(output/'arrays.npz') as successor:
        for key in ('previous_states','references','hypotheses','controls'):
            np.testing.assert_array_equal(successor[key],original[key])
        assert successor['truths'].shape==(6,2,6)
        np.testing.assert_allclose(successor['truths'][:,:,3:],0)
    assert before=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in old.iterdir() if p.is_file()}
    protocol=json.loads((output/'protocol.json').read_text())
    assert protocol['protocol_id']=='force-persistent-grid-native-v2'
    assert protocol['E03']['measured_ticks_per_input_method']==20
    assert protocol['params']==json.loads((ROOT/'experiments/force_adapter/task_inputs/t01.json').read_text())['tracking_solver_params']


def test_original_bootstrap_uses_persistent16_lane_optimizer(model):
    from experiments.force_adapter.task_backend import ForceTask
    requests=[]
    class Optimizer:
        construction_wall_s=0.;workspace={};last_stats={}
        def __init__(self):self.measurements=[];self.native_records=[]
        def __call__(self,request):
            requests.append(request);self.measurements.append({'native_solve_us':1.})
            return request.warm_seeds.copy()
    optimizer=Optimizer();task=ForceTask(model,'unused',{},'original',np.zeros(6),lambda *args:None,optimizer)
    state=np.zeros((1,14));reference=np.zeros((1,192));warm=np.zeros((1,665))
    plan,_=task.solve(state,reference,warm)
    assert requests[0].candidate_indices==tuple(range(16))
    assert requests[0].states.shape==(16,14) and plan.shape==(1,665)
    assert task.predictor.simulation is None
