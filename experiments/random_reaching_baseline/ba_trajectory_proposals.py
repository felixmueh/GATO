"""Exact historical BA trajectory proposal helpers; no native imports or solves.

Source84bad9a: examples/randomized_multimodal/tiago_reaching_batch_pilot.py.
No target conditioning, IK, clipping, residual diagnostics or controller policy.
"""
import numpy as np
import pinocchio as pin

N = 32

def unpack(z):
    z = np.atleast_2d(z)
    body = z[:, :(N-1)*21].reshape(-1, N-1, 21)
    return np.concatenate((body[:,:,:14], z[:,None,-14:]), axis=1), body[:,:,14:]


def pack(x, u):
    return np.concatenate((np.concatenate((x[:,:-1],u), axis=2).reshape(len(x),-1), x[:,-1]), axis=1).astype(np.float32)


def perturbations(model, nominal, batch, dt, rng):
    """Analytic smooth delta q/v/a; RNEA about the nominal ABA acceleration.

    The s^2 envelope has delta q(0)=delta v(0)=0. No terminal condition is
    imposed. All coefficients are fresh, independent of target configuration.
    Nominal dynamic defects are retained; perturbed seeds need not be feasible.
    """
    if batch == 1: return nominal[None].copy(), []
    xs, us = unpack(nominal)
    base_x, base_u = xs[0].astype(float), us[0].astype(float)
    data = model.createData()
    base_a = np.asarray([pin.aba(model,data,x[:7],x[7:],u).copy() for x,u in zip(base_x,base_u)])
    s = np.linspace(0,1,N)[:,None]
    omega = np.pi*np.arange(1,4)[None,:]
    b = s*s*np.cos(s*omega)
    db = 2*s*np.cos(s*omega)-s*s*omega*np.sin(s*omega)
    ddb = 2*np.cos(s*omega)-4*s*omega*np.sin(s*omega)-s*s*omega**2*np.cos(s*omega)
    duration = (N-1)*dt
    result, metadata = [nominal.copy()], []
    for lane in range(1,batch):
        amplitude = (.02,.05,.1)[(lane-1)//5]
        coeff = rng.uniform(-1,1,(3,7))
        normalizer = max(float(np.max(np.abs(b@coeff))),1e-12)
        dq = amplitude*(b@coeff)/normalizer
        dv = amplitude*(db@coeff)/(normalizer*duration)
        da = amplitude*(ddb@coeff)/(normalizer*duration**2)
        candidate_x = base_x + np.concatenate((dq,dv),axis=1)
        candidate_u = np.asarray([pin.rnea(model,data,x[:7],x[7:],a).copy()
                                  for x,a in zip(candidate_x,base_a+da[:-1])])
        result.append(pack(candidate_x[None],candidate_u[None])[0])
        metadata.append(dict(lane=lane, amplitude_rad=amplitude, coefficients=coeff,
                             peak_delta_q_rad=float(np.max(np.abs(dq)))))
    return np.asarray(result,dtype=np.float32), metadata
