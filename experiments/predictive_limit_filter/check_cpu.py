#!/usr/bin/env python3
"""Three deterministic coupled-gravity functional cases; no native solver."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

spec=importlib.util.spec_from_file_location('filter_impl',Path(__file__).with_name('filter.py'))
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)


def main():
    M=np.array([[2.,.4],[.4,1.]])
    gravity=np.array([-20.,10.])
    model=SimpleNamespace(lowerPositionLimit=np.array([-1.,-1.]),upperPositionLimit=np.array([1.,1.]),
                          velocityLimit=np.array([1.,1.]),effortLimit=np.array([100.,100.]),createData=lambda:None)
    aba=lambda q,v,u:np.linalg.solve(M,u-gravity)
    rnea=lambda q,v,a:M@a+gravity
    def rk4(model,data,q,v,u,h):
        a=aba(q,v,u);return q+h*v+.5*h*h*a,v+h*a
    fil=f.PredictiveLimitFilter(model,rk4,forward_acceleration=aba,inverse_dynamics=rnea)
    # Bias-balanced command is retained exactly, including coupled dynamics.
    hold=fil.filter(np.zeros(4),gravity)
    assert hold['guard_certified'] and not hold['modified'] and hold['selected_candidate']=='proposed'
    # Here zero torque creates outward acceleration; returning zero is unsafe.
    moving=np.array([0.,0.,.85,0.])
    proposed=fil.forecast(moving,np.zeros(2))
    assert not proposed['physical_state_pass']
    corrected=fil.filter(moving,np.zeros(2))
    assert corrected['guard_certified'] and corrected['modified'] and np.linalg.norm(corrected['selected_torque'])>0
    # A consumed guard is not confused with an already violated physical bound.
    recovery=fil.filter(np.array([0.,0.,.95,0.]),np.zeros(2))
    assert recovery['initial']['physical_state_pass'] and not recovery['initial']['guard_state_pass']
    assert recovery['status']=='recovery_only_not_guard_certified' and not recovery['guard_certified']
    bad=fil.filter(np.array([1.01,0.,0.,0.]),gravity)
    assert bad['status']=='initial_physical_violation_no_certificate' and bad['selected_torque'] is None
    print(json.dumps(dict(status='PASS',native_calls=0,coupled_mass_matrix=M.tolist(),gravity=gravity.tolist(),
        cases={'bias_hold':hold,'zero_torque_is_unsafe_but_correction_passes':corrected,
               'consumed_buffer_recovery':recovery,'already_physical_violation':bad}),indent=2,allow_nan=False))

if __name__=='__main__':main()
