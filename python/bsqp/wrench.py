"""Explicit wrench components, application points and Pinocchio force frames.

Python uses [Fx,Fy,Fz,Tx,Ty,Tz]; GRiD uses angular-first components. A world
wrench here means world-expressed force and torque ABOUT the named tool origin.
"""
import numpy as np
import pinocchio as pin


def linear_to_angular_first(wrenches):
    values=np.asarray(wrenches)
    if values.shape[-1:]!=(6,) or not np.isfinite(values).all():
        raise ValueError('Wrenches require six finite linear-first components')
    return np.concatenate((values[...,3:],values[...,:3]),axis=-1)


def tool_frame_name(model):
    for name in ('arm_right_tool_link','EE'):
        if model.existFrame(name):return name
    raise ValueError('Model requires an explicit tool frame')


def world_tool_to_joint(model,q,wrench,frame_name=None):
    """Rotate world components and shift tool-origin moment to parent joint."""
    wrench=np.asarray(wrench,dtype=float)
    if wrench.shape!=(6,) or not np.isfinite(wrench).all():
        raise ValueError('World wrench must have six finite linear-first components')
    frame_name=frame_name or tool_frame_name(model)
    if not model.existFrame(frame_name):raise ValueError('Named wrench application frame is absent')
    data=model.createData();pin.framesForwardKinematics(model,data,np.asarray(q,dtype=float))
    frame_id=model.getFrameId(frame_name);joint_id=model.frames[frame_id].parentJoint
    joint=data.oMi[joint_id];point=data.oMf[frame_id].translation
    force=joint.rotation.T@wrench[:3]
    torque=joint.rotation.T@(wrench[3:]+np.cross(point-joint.translation,wrench[:3]))
    return joint_id,np.r_[force,torque]


class WorldToolWrench:
    """Re-express the same world/tool-point wrench at EVERY RK dynamics stage."""
    def __init__(self,model,wrench,frame_name=None):
        self.model=model;self.wrench=np.asarray(wrench,dtype=float).copy()
        self.frame_name=frame_name or tool_frame_name(model)
        if self.wrench.shape!=(6,) or not np.isfinite(self.wrench).all():
            raise ValueError('World wrench must have six finite components')

    def __call__(self,q):
        joint,value=world_tool_to_joint(self.model,q,self.wrench,self.frame_name)
        forces=pin.StdVec_Force()
        for _ in range(self.model.njoints):forces.append(pin.Force.Zero())
        forces[joint]=pin.Force(value[:3],value[3:])
        return forces
