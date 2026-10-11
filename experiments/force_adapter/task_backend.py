"""Force orchestration facade for the unchanged accepted mode B plant clock."""
import numpy as np
import pinocchio as pin

from .backend import NativeOptimizer,PinocchioPredictor,transform_local,LOCAL_FRAME,LOCAL_POINT
from .estimator import FrozenBatchForceEstimator
from .orchestration import AppliedSegment,TickInputs,TORequest,WrenchConvention,run_tick
from bsqp.common import rk4


class ForceTask:
    def __init__(self,model,model_path,params,mode,truth,emit,optimizer=None):
        self.model=model;self.mode=mode;self.truth=np.asarray(truth).copy();self.emit=emit
        self.params=dict(params)
        self.N=32;self.dt=.03;self.batch_size=1;self.nq=self.nv=self.nq_robot=self.nv_robot=self.nu=7;self.nx=14
        self.has_pendulum=False;self.final_approach=None;self.track_full_stats=True
        self.estimator=FrozenBatchForceEstimator(16,initial_radius=5.,min_radius=2.,max_radius=20.,smoothing_factor=.5)
        self.predictor=PinocchioPredictor(model)
        self.optimizer=optimizer or NativeOptimizer(model_path,params)
        self.convention=WrenchConvention(LOCAL_FRAME,LOCAL_POINT,LOCAL_FRAME,LOCAL_POINT)
        self.solver=self;self.previous_state=None;self.previous_time=0.;self.time=0.;self.history=[]
        self.pending=[];self.selected_index=0;self.selected_stats_index=0
        self.actual=pin.StdVec_Force()
        for _ in range(model.njoints):self.actual.append(pin.Force.Zero())
        self.actual[-1]=pin.Force(self.truth[:3],self.truth[3:])
        self.actual_f_ext=self.actual

    def reset_dual(self):pass # Fresh native TO owner resets dual/rho each call.
    def reset_rho(self):pass
    def update_force_batch(self,q):pass # run_tick freezes its own hypotheses.
    def evaluate_best_trajectory(self,*args):return 0 # Facade returns one selected plan.

    def integrate(self,q,v,u,dt):
        start=self.time;self.time+=float(dt)
        self.history.append(AppliedSegment(start,self.time,np.asarray(u).copy()))
        return rk4(self.model,self.model.createData(),q,v,u,dt,self.actual)

    def ee_pos(self,q):
        data=self.model.createData();pin.framesForwardKinematics(self.model,data,q)
        return data.oMf[self.model.getFrameId('arm_right_tool_link')].translation.copy()

    def ee_tool_axis_error(self,q,target_rpy):
        data=self.model.createData();pin.framesForwardKinematics(self.model,data,q)
        z=data.oMf[self.model.getFrameId('arm_right_tool_link')].rotation[:,2]
        return float(np.arccos(np.clip(z@pin.rpy.rpyToMatrix(*map(float,target_rpy))[:,2],-1,1)))

    def get_stats(self):
        stats=dict(self.optimizer.last_stats)
        for key in ('sqp_iters','kkt_converged'):
            values=np.asarray(stats.get(key,[])).reshape(-1)
            if values.size:stats[key]=values[[self.selected_stats_index]]
        return stats

    def solve(self,states,references,warm):
        try:
            return self._solve(states,references,warm)
        except BaseException as error:
            self.pending.append(('force_algorithm_failed',dict(error_type=type(error).__name__,error=str(error),
                native_records=list(self.optimizer.native_records),backend_measurements=list(self.optimizer.measurements))))
            raise

    def _solve(self,states,references,warm):
        state=states[0].copy();before=len(self.optimizer.native_records);memory_before=len(self.optimizer.measurements)
        if self.previous_state is None or self.mode=='no_estimator':
            # No observation history exists yet. Use the estimator's prior lane0
            # and one B1 startup solve, independent of the hidden plant truth.
            force=np.zeros(6) if self.mode=='no_estimator' else self.estimator.generate_batch()[0].copy()
            request=TORequest((0,),states,references,warm,force[None,:],force[None,:],self.convention)
            plan=self.optimizer(request)[0]
            self.pending.append(('force_baseline' if self.previous_state is not None else 'force_bootstrap',dict(policy='zero-wrench B1 without estimator' if self.mode=='no_estimator' else 'prior lane0, one B1 TO, no estimator update; first actual control history starts at t=0; bootstrap compute uncharged by accepted modeB',
                state=state.copy(),force=force.copy(),reference=references.copy(),warm_seed=warm.copy(),
                native_records=list(self.optimizer.native_records[before:]),backend_measurements=list(self.optimizer.measurements[memory_before:]))))
            self.selected_stats_index=0
        else:
            tick=TickInputs(self.previous_time,self.time,self.previous_state,state,tuple(self.history),
                           np.tile(state,(16,1)),np.tile(references[0],(16,1)),np.tile(warm[0],(16,1)))
            def observe(record):
                self.pending.append(('force_tick',dict(record,native_records=list(self.optimizer.native_records[before:]),
                    backend_measurements=list(self.optimizer.measurements[memory_before:]))))
            result=run_tick(self.mode,tick,estimator=self.estimator,predictor=self.predictor,optimizer=self.optimizer,
                            convention=self.convention,transform_wrench=transform_local,observer=observe)
            self.selected_index=result['selected_index'];plan=result['selected_plan']
            self.selected_stats_index=self.selected_index if self.mode=='original' else 0
        self.previous_state=state;self.previous_time=self.time;self.history=[]
        native_us=self.optimizer.measurements[-1]['native_solve_us']
        return plan[None,:],native_us

    def record(self,event,record):
        # Called after the modeB solve timer; disk/array serialization is excluded
        # from plant advancement. Full algorithm/owner/memory queries are inside.
        while self.pending:
            name,value=self.pending.pop(0);self.emit(name,value)
        self.optimizer.native_records.clear();self.optimizer.measurements.clear()
        if event=='solve_started':
            self.emit('force_algorithm_inputs',dict(hypotheses=self.estimator.generate_batch(),
                previous_state=self.previous_state,previous_time_s=self.previous_time,observed_time_s=self.time,
                applied_history=tuple(self.history),rng_state=np.random.get_state(),
                estimator_state={key:getattr(self.estimator,key) for key in
                    ('estimate','momentum','smoothed_estimate','radius','current_rotation')}))
        self.emit(event,record)

    def flush(self):
        while self.pending:
            name,value=self.pending.pop(0);self.emit(name,value)
