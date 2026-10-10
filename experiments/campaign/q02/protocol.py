"""Frozen Q02 mathematical checks; CPU helpers never import CUDA/native code."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
URDF = ROOT / 'gato/dynamics/tiago_right/tiago_right_arm.urdf'
WEIGHTS = dict(position=30., terminal_position=300., orientation=3., terminal_orientation=30.,
               velocity=.1, control=.001, q_barrier=.01, v_barrier=.1, u_barrier=.01)
LAYOUT = {'qdd': (0,7), 'dynamics_jac': (7,154), 'step': (154,168), 'A': (168,364),
          'B': (364,462), 'pose': (462,468), 'pose_jac': (468,510), 'cost': (510,511),
          'gradient': (511,532), 'Q': (532,728), 'R': (728,777), 'terminal_cost': (777,778),
          'terminal_gradient': (778,792), 'terminal_Q': (792,988), 'qdd_gradient_path': (988,995)}
SHAPES = {'dynamics_jac': (7,21), 'A': (14,14), 'B': (14,7), 'pose_jac': (6,7),
          'Q': (14,14), 'R': (7,7), 'terminal_Q': (14,14)}

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_protocol():
    return json.loads((HERE/'protocol.json').read_text())

def central_jacobian(function, point, step):
    point = np.asarray(point, dtype=np.float64)
    columns=[]
    for j in range(point.size):
        plus=point.copy(); minus=point.copy()
        plus[j]+=step; minus[j]-=step
        columns.append((np.asarray(function(plus))-np.asarray(function(minus)))/(2*step))
    return np.stack(columns,axis=-1)

def pose_difference(a,b):
    d=np.asarray(a)-np.asarray(b)
    d=np.array(d,copy=True); d[...,3:]=(d[...,3:]+np.pi)%(2*np.pi)-np.pi
    return d

def one_step(xu, acceleration, dt=.03):
    q,v=np.asarray(xu)[:7],np.asarray(xu)[7:14]
    return np.concatenate((q+dt*v+.5*dt*dt*acceleration,v+dt*acceleration))

def barrier(z, limits):
    lower=np.asarray(limits)[:,0]; upper=np.asarray(limits)[:,1]
    if not np.all((z>lower)&(z<upper)):
        raise ValueError('Q02 cost reference accepts strict interior inputs only')
    return -np.log(z-lower)-np.log(upper-z)

def barrier_hessian(z,limits):
    limits=np.asarray(limits)
    return 1/(z-limits[:,0])**2+1/(limits[:,1]-z)**2

def cost_reference(xu,pose,goal,limits,terminal=False):
    w=WEIGHTS; q,v,u=np.split(np.asarray(xu),3)
    e=pose_difference(pose,goal)
    pw=w['terminal_position'] if terminal else w['position']
    ow=w['terminal_orientation'] if terminal else w['orientation']
    value=.5*(pw*np.dot(e[:3],e[:3])+ow*np.dot(e[3:],e[3:])+w['velocity']*np.dot(v,v))
    value+=w['q_barrier']*barrier(q,limits['q']).sum()+w['v_barrier']*barrier(v,limits['v']).sum()
    if not terminal:
        value+=.5*w['control']*np.dot(u,u)+w['u_barrier']*barrier(u,limits['u']).sum()
    return float(value)

def gn_reference(xu,pose_jac,limits,terminal=False):
    """J' W J + analytic regularizer/barrier curvature; not exact pose Hessian."""
    w=WEIGHTS; q,v,u=np.split(np.asarray(xu),3)
    pw=w['terminal_position'] if terminal else w['position']
    ow=w['terminal_orientation'] if terminal else w['orientation']
    Q=np.zeros((14,14)); Q[:7,:7]=pose_jac.T@np.diag([pw]*3+[ow]*3)@pose_jac
    Q[:7,:7]+=np.diag(w['q_barrier']*barrier_hessian(q,limits['q']))
    Q[7:,7:]=np.diag(w['velocity']+w['v_barrier']*barrier_hessian(v,limits['v']))
    R=np.diag(w['control']+w['u_barrier']*barrier_hessian(u,limits['u']))
    return Q,R

class PinReference:
    def __init__(self,urdf=URDF):
        import pinocchio as pin
        self.pin=pin; self.model=pin.buildModelFromUrdf(str(urdf)); self.data=self.model.createData()
        self.model.gravity.linear=np.array([0.,0.,-9.81])
        expected=['universe']+[f'arm_right_{i}_joint' for i in range(1,8)]
        if self.model.nq!=7 or self.model.nv!=7 or list(self.model.names)!=expected:
            raise ValueError('Unexpected native joint order/model dimensions')
        self.frame=self.model.getFrameId('arm_right_tool_link')
        if self.frame>=len(self.model.frames): raise ValueError('Missing arm_right_tool_link')
    def pose(self,q):
        self.pin.framesForwardKinematics(self.model,self.data,np.asarray(q,dtype=float))
        transform=self.data.oMf[self.frame]
        return np.r_[transform.translation,self.pin.rpy.matrixToRpy(transform.rotation)]
    def acceleration(self,xu):
        return self.pin.aba(self.model,self.data,xu[:7],xu[7:14],xu[14:21]).copy()
    def references(self,xu,goal,limits):
        h=read_protocol()['reference_fd_step']
        pose=self.pose(xu[:7]); acc=self.acceleration(xu)
        # Reference central differences are independently evaluated in float64.
        pj=central_jacobian(lambda q: pose_difference(self.pose(q),pose),xu[:7],h)
        dj=central_jacobian(self.acceleration,xu,h)
        sj=central_jacobian(lambda z: one_step(z,self.acceleration(z)),xu,h)
        g=central_jacobian(lambda z: cost_reference(z,self.pose(z[:7]),goal,limits),xu,h)
        tg=central_jacobian(lambda z: cost_reference(np.r_[z,xu[14:]],self.pose(z[:7]),goal,limits,True),xu[:14],h)
        Q,R=gn_reference(xu,pj,limits); tQ,_=gn_reference(xu,pj,limits,True)
        return dict(qdd=acc,qdd_gradient_path=acc,step=one_step(xu,acc),dynamics_jac=dj,A=sj[:,:14],B=sj[:,14:],
                    pose=pose,pose_jac=pj,cost=np.array([cost_reference(xu,pose,goal,limits)]),gradient=g,
                    Q=Q,R=R,terminal_cost=np.array([cost_reference(xu,pose,goal,limits,True)]),
                    terminal_gradient=tg,terminal_Q=tQ)

def decode(flat):
    flat=np.asarray(flat)
    if flat.ndim!=2 or flat.shape[1]!=995: raise ValueError('Unexpected native output shape')
    result={key:flat[:,a:b] for key,(a,b) in LAYOUT.items()}
    for key,shape in SHAPES.items():
        # native matrices use column-major; Q/R symmetric also follow this contract.
        result[key]=np.stack([row.reshape(shape,order='F') for row in result[key]])
    return result

def compare(actual,expected,atol,rtol,angular=False):
    a=np.asarray(actual); b=np.asarray(expected)
    if a.shape!=b.shape: raise ValueError(f'Comparison shape mismatch {a.shape} != {b.shape}')
    finite=bool(np.isfinite(a).all() and np.isfinite(b).all())
    error=np.abs(pose_difference(a,b) if angular else a-b)
    allowed=atol+rtol*np.abs(b)
    return {'pass':bool(finite and np.all(error<=allowed)), 'finite':finite,
            'max_abs_error':float(np.max(error)) if finite else None,
            'max_allowed_ratio':float(np.max(error/allowed)) if finite else None}
