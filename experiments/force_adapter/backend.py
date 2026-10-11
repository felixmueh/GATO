"""TIAGo body-fixed final-joint wrench backend for the newly defined adapter.

Both Pinocchio prediction and native optimization hold the same force/torque
components constant in arm_right_7_joint coordinates about that joint origin.
These are attached/body-fixed disturbances, not constant world-force tasks.
"""
from pathlib import Path
import sys
import time
import numpy as np
import pinocchio as pin

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'python'))
from bsqp.common import rk4

LOCAL_FRAME='arm_right_7_joint'
LOCAL_POINT='arm_right_7_joint origin'


def transform_local(state,wrench,convention):
    if (convention.hypothesis_frame,convention.backend_frame)!=(LOCAL_FRAME,LOCAL_FRAME) or (
        convention.hypothesis_point,convention.backend_point)!=(LOCAL_POINT,LOCAL_POINT):
        raise ValueError('TIAGo backend requires identical explicit final-joint-local frame/point')
    values=np.asarray(wrench,dtype=float)
    if values.shape!=(6,) or not np.isfinite(values).all():raise ValueError('Six finite linear-first components required')
    return values.copy()


class PinocchioPredictor:
    def __init__(self,model,sim_dt=.003):
        if model.nq!=7 or model.nv!=7 or model.names[-1]!=LOCAL_FRAME:
            raise ValueError('Exact TIAGo right-arm model required')
        if not np.isfinite(sim_dt) or sim_dt<=0:raise ValueError('Positive finite integration step required')
        self.model=model;self.sim_dt=float(sim_dt)

    def __call__(self,request):
        predictions=[]
        for force in request.hypothesis_wrenches:
            state=np.asarray(request.inputs.previous_state,dtype=float).copy()
            data=self.model.createData()
            def local(q):
                # Honor the orchestration callback at every dynamics stage.
                # The declared body-fixed frame/point makes this an identity.
                value=request.transform_wrench(np.r_[q,state[7:]],force,request.convention)
                forces=pin.StdVec_Force()
                for _ in range(self.model.njoints):forces.append(pin.Force.Zero())
                forces[-1]=pin.Force(value[:3],value[3:])
                return forces
            for segment in request.inputs.applied_history:
                remaining=segment.end_s-segment.start_s
                while remaining>1e-14:
                    dt=min(remaining,self.sim_dt)
                    q,v=rk4(self.model,data,state[:7],state[7:],segment.control,dt,local)
                    state=np.r_[q,v];remaining-=dt
            predictions.append(state)
        return np.asarray(predictions)


class NativeOptimizer:
    """One persistent public BSQP owner; construction is outside tick timing."""
    def __init__(self,model_path,params,N=32,dt=.03,batch_size=1,memory_query=None):
        from bsqp.interface import BSQP
        self.model_path=str(model_path);self.params=dict(params);self.N=N;self.dt=dt
        self.batch_size=int(batch_size);self.native_records=[];self.measurements=[];self.last_stats={}
        self.memory_query=memory_query
        started=time.perf_counter()
        self.owner=BSQP(self.model_path,self.batch_size,N,dt,plant_type='tiago_right',**self.params)
        native=self.owner.solver
        class Capture:
            def __getattr__(capture,name):return getattr(native,name)
            def solve(capture,xu,dt,state,reference):
                reply=native.solve(xu,dt,state,reference)
                self.native_records.append(dict(candidate_indices=self._candidate_indices,native_reply=reply))
                return reply
        self.owner.solver=Capture()
        self.construction_wall_s=time.perf_counter()-started
        self.workspace=dict(native.workspace_bytes())

    def reset_dual(self):self.owner.reset_dual()
    def reset_rho(self):self.owner.reset_rho()

    def __call__(self,request):
        if len(request.candidate_indices)!=self.batch_size:
            raise ValueError('Persistent optimizer batch differs from requested candidates')
        for state,wrench in zip(request.states,request.backend_wrenches):
            transform_local(state,wrench,request.convention)
        self._candidate_indices=request.candidate_indices
        # The task loop owns reaching/tracking rho reset policy.
        self.owner.set_f_ext_B(request.backend_wrenches)
        started=time.perf_counter()
        plans,native_us=self.owner.solve(request.states,request.references,request.warm_seeds.copy())
        self.measurements.append(dict(candidate_indices=request.candidate_indices,
            public_solve_wall_s=time.perf_counter()-started,native_solve_us=float(native_us)))
        self.last_stats=self.owner.get_stats()
        return plans.copy()

    def predict(self,state,control,hypotheses,dt):
        if self.batch_size!=16:raise ValueError('Only original B16 optimizer supplies prediction')
        self.owner.set_f_ext_B(hypotheses)
        return self.owner.sim_forward(state,control,dt)


class GPUPredictor:
    """One unchanged GRiD step over elapsed time with applied-control average."""
    def __init__(self,optimizer,mode):
        self.optimizer=optimizer;self.mode=mode;started=time.perf_counter()
        self.simulation=None
        if mode=='adapted':self.simulation=optimizer.owner.lib.Simulation16_float()
        self.construction_wall_s=time.perf_counter()-started
        self.workspace={} if self.simulation is None else dict(self.simulation.workspace_bytes())

    def __call__(self,request):
        inputs=request.inputs
        elapsed=inputs.observed_time_s-inputs.previous_time_s
        control=sum((segment.end_s-segment.start_s)*np.asarray(segment.control)
                    for segment in inputs.applied_history)/elapsed
        hypotheses=np.asarray([request.transform_wrench(inputs.previous_state,f,request.convention)
                               for f in request.hypothesis_wrenches],dtype=np.float32)
        self.last_observation=dict(elapsed_interval_s=elapsed,average_applied_control=control.copy(),
            approximation='one existing GRiD TYPE2 step over actual elapsed interval using time-weighted applied control')
        if self.simulation is None:
            return self.optimizer.predict(inputs.previous_state,control,hypotheses,elapsed)
        return self.simulation.sim_forward(np.asarray(inputs.previous_state,dtype=np.float32),
            np.asarray(control,dtype=np.float32),hypotheses,float(elapsed))


def attached_force(model,state,magnitude):
    """Pure last-joint-origin force aligned with world-Z downward at start."""
    if not np.isfinite(magnitude) or magnitude<0:raise ValueError('Finite nonnegative force required')
    data=model.createData();pin.forwardKinematics(model,data,np.asarray(state)[:7])
    local_direction=data.oMi[-1].rotation.T@np.array([0.,0.,-1.])
    return np.r_[float(magnitude)*local_direction,np.zeros(3)]


def force_world_direction(model,q,truth):
    data=model.createData();pin.forwardKinematics(model,data,q)
    force=data.oMi[-1].rotation@np.asarray(truth)[:3]
    norm=np.linalg.norm(force)
    direction=None if norm==0 else force/norm
    angle=None if direction is None else float(np.rad2deg(np.arccos(np.clip(-direction[2],-1.,1.))))
    return dict(world_force_N=force,world_force_direction=direction,deviation_from_world_down_deg=angle)
