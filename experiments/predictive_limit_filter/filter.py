"""Small coupled-dynamics torque filter; CPU prediction, no optimizer/native imports.

A passed sampled model forecast is not a robust/continuous-time safety guarantee.
The fixed candidate set can fail even when an untested feasible torque exists.
"""
import time
import numpy as np
import pinocchio as pin

DEFAULTS = dict(q_margin_rad=.02, velocity_margin_rad_s=.10, effort_fraction=.95,
                forecast_s=.03, forecast_step_s=.003, nominal_control_step_s=.003)


class PredictiveLimitFilter:
    def __init__(self, model, rk4, *, settings=None, forward_acceleration=None, inverse_dynamics=None):
        self.model=model;self.data=model.createData();self.rk4=rk4
        self.settings={**DEFAULTS,**(settings or {})}
        if set(self.settings)!=set(DEFAULTS):raise ValueError('Unknown filter setting')
        s=self.settings
        if (any(not np.isfinite(v) or v<=0 for v in s.values()) or s['effort_fraction']>1
                or s['forecast_step_s']>s['forecast_s']):raise ValueError('Invalid filter setting')
        self.qlo=np.asarray(model.lowerPositionLimit,dtype=float)
        self.qhi=np.asarray(model.upperPositionLimit,dtype=float)
        self.vmax=np.asarray(model.velocityLimit,dtype=float)
        self.umax=np.asarray(model.effortLimit,dtype=float)
        self.qlow=self.qlo+s['q_margin_rad'];self.qhigh=self.qhi-s['q_margin_rad']
        self.vguard=self.vmax-s['velocity_margin_rad_s'];self.uguard=self.umax*s['effort_fraction']
        if np.any(self.qlow>=self.qhigh) or np.any(self.vguard<=0):raise ValueError('Empty guard interval')
        self.acceleration=forward_acceleration or (lambda q,v,u:pin.aba(model,self.data,q,v,u).copy())
        self.inverse=inverse_dynamics or (lambda q,v,a:pin.rnea(model,self.data,q,v,a).copy())

    def state_summary(self, states):
        x=np.asarray(states,dtype=float).reshape(-1,2*len(self.qlo))
        if not np.isfinite(x).all():
            return dict(finite=False,physical_state_pass=False,guard_state_pass=False,
                        minimum_physical_q_clearance_rad=None,maximum_speed_ratio=None,guard_excess=None)
        q,v=x[:,:len(self.qlo)],x[:,len(self.qlo):]
        rawq=float(np.min(np.minimum(q-self.qlo,self.qhi-q)))
        excess=np.maximum.reduce([np.max((self.qlow-q)/self.settings['q_margin_rad'],axis=1),
                                  np.max((q-self.qhigh)/self.settings['q_margin_rad'],axis=1),
                                  np.max((np.abs(v)-self.vguard)/self.settings['velocity_margin_rad_s'],axis=1),
                                  np.zeros(len(x))])
        return dict(finite=True,physical_state_pass=bool(rawq>=-1e-9 and np.all(np.abs(v)<=self.vmax+1e-9)),
                    guard_state_pass=bool(np.all(q>=self.qlow-1e-9) and np.all(q<=self.qhigh+1e-9) and np.all(np.abs(v)<=self.vguard+1e-9)),
                    minimum_physical_q_clearance_rad=rawq,maximum_speed_ratio=float(np.max(np.abs(v)/self.vmax)),
                    guard_excess=float(np.max(excess)),endpoint_guard_excess=float(excess[-1]))

    def forecast(self, actual_state, torque):
        """Thirty-ms held-torque sampled RK4 forecast from this measured state."""
        x=np.asarray(actual_state,dtype=float).copy();u=np.asarray(torque,dtype=float)
        states=[x.copy()];elapsed=0.;n=len(self.qlo)
        while elapsed<self.settings['forecast_s']-1e-12:
            h=min(self.settings['forecast_step_s'],self.settings['forecast_s']-elapsed)
            q,v=self.rk4(self.model,self.data,x[:n],x[n:],u,h)
            x=np.r_[q,v];states.append(x.copy());elapsed+=h
            if not np.isfinite(x).all():break
        d=self.state_summary(states)
        effort=float(np.max(np.abs(u)/self.umax)) if np.isfinite(u).all() else None
        d.update(maximum_effort_ratio=effort,
                 physical_effort_pass=bool(effort is not None and effort<=1+1e-9),
                 guard_effort_pass=bool(np.isfinite(u).all() and np.all(np.abs(u)<=self.uguard+1e-9)),
                 duration_s=elapsed,sample_count=len(states))
        return d

    def filter(self, actual_state, proposed_torque):
        start=time.perf_counter();x=np.asarray(actual_state,dtype=float);requested=np.asarray(proposed_torque,dtype=float)
        n=len(self.qlo)
        if x.shape!=(2*n,) or requested.shape!=(n,):raise ValueError('State/control shape differs')
        initial=self.state_summary(x)
        report=dict(settings=dict(self.settings),initial=initial,proposed_torque=requested.tolist(),
                    candidates=[],selected_torque=None,selected_candidate=None,guard_certified=False,
                    acceleration_interval_conflict_joints=[],no_feasible_control_claim=False)
        def finish(status):
            wall=time.perf_counter()-start
            report.update(status=status,filter_wall_s=wall,wall_to_nominal_step_ratio=wall/self.settings['nominal_control_step_s'],
                          nominal_step_overrun=wall>self.settings['nominal_control_step_s'])
            return report
        if not initial['finite'] or not np.isfinite(requested).all():return finish('nonfinite_input_no_command')
        if not initial['physical_state_pass']:return finish('initial_physical_violation_no_certificate')
        proposed_forecast=self.forecast(x,requested)
        if (proposed_forecast['physical_state_pass'] and proposed_forecast['physical_effort_pass'] and
                proposed_forecast['guard_state_pass'] and proposed_forecast['guard_effort_pass']):
            r=dict(name='proposed',torque=requested.tolist(),forecast=proposed_forecast,accepted=True,
                   guard_certified=True,normalized_squared_torque_change=0.)
            report['candidates']=[r]
            report.update(selected_torque=requested.tolist(),selected_candidate='proposed',
                          selected_forecast=proposed_forecast,guard_certified=True,modified=False)
            return finish('guard_forecast_pass')
        q,v=x[:n],x[n:];H=self.settings['forecast_s']
        bounded=np.clip(requested,-self.uguard,self.uguard)
        candidate_list=[('proposed',requested),('effort_bounded',bounded)]
        lower=np.maximum((-self.vguard-v)/H,2*(self.qlow-q-v*H)/(H*H))
        upper=np.minimum((self.vguard-v)/H,2*(self.qhigh-q-v*H)/(H*H))
        conflicts=np.flatnonzero(lower>upper)
        report['acceleration_interval_conflict_joints']=conflicts.tolist()
        if not len(conflicts):
            a=np.clip(np.asarray(self.acceleration(q,v,bounded),dtype=float),lower,upper)
            candidate_list.append(('projected_acceleration_RNEA',np.clip(self.inverse(q,v,a),-self.uguard,self.uguard)))
        # Bias compensation includes gravity and coupling; zero torque is not a safe anchor.
        bias=np.clip(self.inverse(q,v,np.zeros(n)),-self.uguard,self.uguard)
        recovery=np.clip(q,self.qlow,self.qhigh)-q
        brake_a=-v/H+2*recovery/(H*H)
        brake=np.clip(self.inverse(q,v,brake_a),-self.uguard,self.uguard)
        strong=np.clip(self.inverse(q,v,2*brake_a),-self.uguard,self.uguard)
        candidate_list.extend([('bias_compensation_RNEA',bias),('brake_RNEA',brake),('strong_brake_RNEA',strong)])
        for alpha in (.75,.5,.25):
            candidate_list.append((f'blend_{alpha:g}',alpha*bounded+(1-alpha)*brake))
        # Same candidate set/objective as phase1; test nearest candidates first.
        # Strict passes are possible only when the initial state is inside guard;
        # otherwise all accepted candidates are recovery-only. First passing in
        # this order therefore equals the old exhaustive closest-passing result.
        candidate_list=sorted(enumerate(candidate_list),key=lambda z:(float(np.sum(((np.asarray(z[1][1])-requested)/self.umax)**2)),z[0]))
        seen=set();passed=[]
        for candidate_index,(name,u) in candidate_list:
            u=np.asarray(u,dtype=float);key=u.tobytes()
            if key in seen:continue
            seen.add(key)
            if not np.isfinite(u).all():
                report['candidates'].append(dict(name=name,torque=u.tolist(),accepted=False,reason='nonfinite_candidate'))
                continue
            f=proposed_forecast if name=='proposed' else self.forecast(x,u)
            raw=f['physical_state_pass'] and f['physical_effort_pass']
            strict=raw and f['guard_state_pass'] and f['guard_effort_pass']
            # Already-consumed buffers can be recovered without claiming a guard certificate.
            recovery_ok=(not initial['guard_state_pass'] and raw and f['guard_effort_pass'] and
                         f['guard_excess']<=initial['guard_excess']+1e-8 and
                         f['endpoint_guard_excess']<initial['guard_excess']-1e-8)
            accepted=bool(strict or recovery_ok)
            distance=float(np.sum(((u-requested)/self.umax)**2))
            r=dict(name=name,torque=u.tolist(),forecast=f,accepted=accepted,guard_certified=bool(strict),
                   normalized_squared_torque_change=distance)
            report['candidates'].append(r)
            if accepted:
                passed.append((not strict,distance,len(report['candidates'])-1,u.copy()))
                report['candidate_evaluation_policy']='normalized_distance_order_first_pass_same_fixed_set'
                break
        if not passed:return finish('no_candidate_certified')
        _,_,idx,u=min(passed,key=lambda z:z[:3]);chosen=report['candidates'][idx]
        report.update(selected_torque=u.tolist(),selected_candidate=chosen['name'],selected_forecast=chosen['forecast'],
                      guard_certified=chosen['guard_certified'],modified=bool(not np.array_equal(u,requested)))
        return finish('guard_forecast_pass' if chosen['guard_certified'] else 'recovery_only_not_guard_certified')
