#!/usr/bin/env python3
"""Show screened geometry against the full-body URDF; no executed-motion claim."""
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
import pinocchio as pin
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tiago_src'))
from gato_tiago.safety_monitor import (build_tiago_collision_model, NamedJointState,
    state_to_qv, DEFAULT_COLLISION_URDF_PATH, DEFAULT_COLLISION_BLACKLIST_PATH)


def load_visual_meshes(collision):
    meshes=[]
    urdf=Path('/workspace/GATO/TiagoProURDF/tiago_pro.urdf'); tree=ET.parse(urdf)
    for link in tree.findall('link'):
        fid=collision.model.getFrameId(link.attrib['name'])
        if fid>=len(collision.model.frames):continue
        for visual in link.findall('visual'):
            mesh=visual.find('geometry/mesh')
            if mesh is None:continue
            name=mesh.attrib['filename'].replace('package://','')
            path=urdf.parent/name
            if not path.is_file():continue
            raw=path.read_bytes();count=int.from_bytes(raw[80:84],'little')
            if len(raw)!=84+50*count:raise ValueError('Expected binary STL: '+str(path))
            triangles=np.frombuffer(raw, dtype=np.dtype({'names':['vertices'],
                'formats':[('<f4',(3,3))], 'offsets':[12], 'itemsize':50}), offset=84)['vertices']
            vertices=triangles.reshape(-1,3)
            # Coarsen the mesh for a readable static body rendering while retaining
            # complete surface coverage, rather than dropping isolated triangles.
            _, ids, mapped=np.unique(np.round(vertices/.018),axis=0,return_index=True,return_inverse=True)
            vertices=vertices[ids];faces=mapped.reshape(-1,3)
            keep=(faces[:,0]!=faces[:,1])&(faces[:,1]!=faces[:,2])&(faces[:,0]!=faces[:,2])
            faces=np.unique(faces[keep],axis=0)
            origin=visual.find('origin');xyz=np.zeros(3);rpy=np.zeros(3)
            if origin is not None:
                xyz=np.fromstring(origin.attrib.get('xyz','0 0 0'),sep=' ')
                rpy=np.fromstring(origin.attrib.get('rpy','0 0 0'),sep=' ')
            local=pin.SE3(pin.rpy.rpyToMatrix(rpy),xyz)
            vertices=vertices*np.fromstring(mesh.attrib.get('scale','1 1 1'),sep=' ')
            arm='arm_right' in link.attrib['name'] or 'gripper_right' in link.attrib['name']
            meshes.append((fid,local,vertices,faces,arm))
    return meshes


def draw_robot(ax,collision,data,meshes):
    torso=data.oMf[collision.model.getFrameId('torso_lift_link')].copy()
    for fid,local,vertices,faces,arm in meshes:
        pose=torso.inverse()*data.oMf[fid]*local
        world=vertices@pose.rotation.T+pose.translation
        ax.add_collection3d(Poly3DCollection(world[faces],facecolor='#3d73a8' if arm else '#a6adb2',alpha=.9,edgecolor='none'))


def main():
    folder=Path(sys.argv[1]);s=json.loads((folder/'screen.json').read_text())
    collision=build_tiago_collision_model(urdf_path=DEFAULT_COLLISION_URDF_PATH,
        blacklist_path=DEFAULT_COLLISION_BLACKLIST_PATH,monitor_only_collision_pairs=True)
    body=dict(s['body_positions'],**{f'arm_right_{i+1}_joint':v for i,v in enumerate(s['initial_q_rad'])})
    q,_=state_to_qv(collision.model,NamedJointState(position=body,velocity={k:0. for k in body}))
    data=collision.model.createData();pin.framesForwardKinematics(collision.model,data,q)
    torso=data.oMf[collision.model.getFrameId('torso_lift_link')].copy()
    fig=plt.figure(figsize=(14,7));a=fig.add_subplot(131,projection='3d');b=fig.add_subplot(132);c=fig.add_subplot(133)
    draw_robot(a,collision,data,load_visual_meshes(collision))
    offset=np.array(((0,-.15,-.15),(0,-.15,.15),(0,.15,.15),(0,.15,-.15),(0,0,0)))
    for center,color,label in [((.5,-.4,0),'#999999','old v5 (right)'),((.5,0,-.1),'#c13d39','30 cm centered front'),((.5,-.05,-.1),'#d08a17','30 cm front, 5 cm right')]:
        targets=np.asarray(center)+offset;border=targets[[0,1,2,3,0]]
        a.plot(*border.T,color=color,lw=2,label=label)
        b.plot(-border[:,1],border[:,2],color=color,lw=2,label=label)
        c.plot(-border[:,1],border[:,0],color=color,lw=2,label=label)
    for ax in (b,c):ax.axvline(0,color='.4',ls=':');ax.set_xlabel('Robot right = -torso Y (m)');ax.grid(alpha=.2);ax.set_aspect('equal')
    b.set_ylabel('Torso Z (m)');b.set_title('Front projection (+X plane)');b.set_xlim(-.35,.70);b.set_ylim(-.45,.35)
    b.plot([-.16,.16],[-.092,-.092],'o-',color='#3d73a8',lw=5,label='shoulder origins')
    c.set_ylabel('Forward = torso X (m)');c.set_title('Top view');c.set_xlim(-.35,.70);c.set_ylim(-.25,.75)
    angle=np.linspace(0,2*np.pi,100);c.plot(.25*np.sin(angle),.35*np.cos(angle),color='.55');c.annotate('FORWARD +X',xy=(0,.30),xytext=(0,0),ha='center',arrowprops=dict(arrowstyle='->'))
    a.set(xlim=(-.4,.75),ylim=(-.7,.4),zlim=(-.95,.45),xlabel='Torso X / forward (m)',ylabel='Torso Y / left (m)',zlabel='Torso Z (m)')
    a.set_box_aspect((1.15,1.1,1.4));a.view_init(elev=22,azim=-125);a.set_title('Full-body URDF, initial posture')
    a.legend(loc='lower left',fontsize=7)
    fig.suptitle('Front square setup screen — all three 30 cm candidates lack a complete comfortable witness',fontsize=12)
    fig.text(.5,.02,'Static setup geometry only. Torso lift 0.10 m; left-arm comfortable pose; head zero. CPU IK failures are preserved; no controller run.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.95));fig.savefig(folder/'geometry.png',dpi=150)


if __name__=='__main__':main()
