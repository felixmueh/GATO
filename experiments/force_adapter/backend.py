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
    """Fresh complete public BSQP owner per tick; actual native replies retained."""
    def __init__(self,model_path,params,N=32,dt=.03,memory_query=None):
        self.model_path=str(model_path);self.params=dict(params);self.N=N;self.dt=dt
        self.native_records=[]
        self.measurements=[];self.memory_query=memory_query
        self.last_stats={}

    def __call__(self,request):
        from bsqp.interface import BSQP
        from .memory import CudaMemory,summarize
        for state,wrench in zip(request.states,request.backend_wrenches):
            transform_local(state,wrench,request.convention)
        metric=dict(candidate_indices=request.candidate_indices,constructor_wall_s=None,
                    public_solve_wall_s=None,native_solve_us=None,memory=None,memory_error=None)
        self.measurements.append(metric)
        snapshots=[];owner=None;native=None
        try:
            query=self.memory_query or CudaMemory()
        except Exception as error:
            metric['memory_error']=str(error);query=None
        def sample(stage):
            if query is not None:
                try:snapshots.append(query(stage))
                except Exception as error:metric['memory_error']=str(error)
        try:
            sample('before_owner');started=time.perf_counter()
            owner=BSQP(self.model_path,len(request.candidate_indices),self.N,self.dt,
                       plant_type='tiago_right',**self.params)
            metric['constructor_wall_s']=time.perf_counter()-started
            sample('after_owner')
            owner.reset_dual();owner.reset_rho();owner.set_f_ext_B(request.backend_wrenches)
            # Public interface performs the explicit linear->angular conversion.
            # Capture the complete native response before wrapper normalization.
            native=owner.solver
            class Capture:
                def __getattr__(self,name):return getattr(native,name)
                def solve(capture,xu,dt,state,reference):
                    reply=native.solve(xu,dt,state,reference)
                    self.native_records.append(dict(candidate_indices=request.candidate_indices,native_reply=reply))
                    return reply
            owner.solver=Capture()
            warm=request.warm_seeds.copy()
            started=time.perf_counter()
            plans,native_us=owner.solve(request.states,request.references,warm)
            metric['public_solve_wall_s']=time.perf_counter()-started
            metric['native_solve_us']=float(native_us)
            self.last_stats=owner.get_stats() if hasattr(owner,'get_stats') else {}
            sample('after_solve')
            return plans.copy()
        finally:
            if owner is not None:owner.solver=None
            native=None
            sample('after_close')
            metric['memory']=summarize(snapshots) if snapshots else None
