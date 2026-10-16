#!/usr/bin/env python3
"""CPU-only mesh viewer for recorded executed MPC episodes; never replay controls."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

HERE=Path(__file__).resolve().parent
COLORS=['#137aa1','#d66329','#7d4abb','#289468','#bb3970','#91721b']
MODEL='gato/dynamics/tiago_right/tiago_right_arm.urdf'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def load(path):
    spec=importlib.util.spec_from_file_location('recorded_mesh_pipeline',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def plot(output,payload,gif=False,fps=8):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation,PillowWriter
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    count=len(payload['labels']);fig=plt.figure(figsize=(5.3*(count+1),5.7));panels=[]
    center=np.array(payload['center']);half=payload['size']/2;static=np.array(payload['static']);meshes=[np.array(m) for m in payload['meshes']]
    for panel in range(count+1):
        ax=fig.add_subplot(1,count+1,panel+1,projection='3d');ax.set_proj_type('ortho')
        ax.add_collection3d(Poly3DCollection(static,facecolors='#d9dfe2',alpha=.12,edgecolors='none'))
        actors=[]
        for c in ([panel] if panel<count else range(count)):
            artists=[Poly3DCollection(m,edgecolors='none',facecolors=payload['colors'][c],alpha=.85 if panel<count else .45) for m in meshes]
            for artist in artists:ax.add_collection3d(artist)
            trail,=ax.plot([],[],[],color=payload['colors'][c],lw=2);elbow,=ax.plot([],[],[],color=payload['colors'][c],ls=':',lw=1.5)
            actors.append((c,artists,trail,elbow))
        ax.scatter(*payload['goal'],color='black',marker='*',s=90)
        for k,label in enumerate(['x','y','z']):getattr(ax,'set_'+label+'lim')(center[k]-half,center[k]+half)
        ax.set_box_aspect((1,1,1));ax.view_init(elev=18,azim=-55);ax.set(xlabel='World x (m)',ylabel='World y (m)',zlabel='World z (m)')
        if panel<count:
            e=payload['episodes'][panel];ax.set_title(f"{payload['labels'][panel]}\n{e['outcome']}; task {'PASS' if e['task_success'] else 'FAIL'}; physical {'PASS' if e['physical_pass'] else 'FAIL'}\nrecord ends at {e['duration_s']:.2f} s",fontsize=10)
        else:ax.set_title('Recorded-path overlay',fontsize=10)
        panels.append((ax,actors,panel))
    title=fig.suptitle('')
    def draw(frame):
        for ax,actors,panel in panels:
            for c,artists,trail,elbow in actors:
                for artist,mesh,tf in zip(artists,meshes,payload['transforms'][c][frame]):
                    tf=np.array(tf);artist.set_verts(mesh@tf[:,:3].T+tf[:,3])
                trail.set_data_3d(*np.array(payload['tools'][c][:frame+1]).T);elbow.set_data_3d(*np.array(payload['elbows'][c][:frame+1]).T)
            if panel<count:
                e=payload['episodes'][panel]
                status=f"{e['outcome']}; task {'PASS' if e['task_success'] else 'FAIL'}; physical {'PASS' if e['physical_pass'] else 'FAIL'}"
                ending=f"ENDED at {e['duration_s']:.2f} s; final recorded pose" if payload['times'][frame]>e['duration_s']+1e-9 else f"record ends at {e['duration_s']:.2f} s"
                ax.set_title(f"{payload['labels'][panel]}\n{status}\n{ending}",fontsize=10)
        title.set_text(f"Recorded executed MPC motion | simulated t={payload['times'][frame]:.2f} s\nDisplay interpolation only; star: common goal; solid/dotted: hand/elbow")
    fig.subplots_adjust(left=.01,right=.98,bottom=.05,top=.79,wspace=.02);draw(len(payload['times'])-1);fig.savefig(output/'comparison.png',dpi=125)
    if gif:FuncAnimation(fig,draw,frames=len(payload['times']),interval=1000/fps,blit=False).save(output/'comparison.gif',writer=PillowWriter(fps=fps),dpi=80)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--episodes',type=Path,nargs='+',required=True);p.add_argument('--labels',nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--renderer-source',type=Path,default=Path('/workspace/GATO-reaching-landscape'));p.add_argument('--full-urdf',type=Path,default=Path('/workspace/GATO/TiagoProURDF/tiago_pro.urdf'));p.add_argument('--context',type=Path,default=HERE/'comparison_context.json');p.add_argument('--fps',type=int,default=8);p.add_argument('--gif',action='store_true');a=p.parse_args()
    if len(a.episodes)<2 or len(a.episodes)!=len(a.labels) or a.fps<=0:p.error('Supply at least two episode directories, one label each, and positive fps')
    out=a.output.resolve()
    if out.exists():raise ValueError('Output must be fresh: '+str(out))
    episodes=[d.resolve() for d in a.episodes]
    if any(out.is_relative_to(d) for d in episodes):raise ValueError('Derived output must be outside measured episode directories')
    import numpy as np
    import pinocchio as pin
    renderer_path=a.renderer_source/'experiments/tiago_reaching_landscape/visualize/render.py';renderer=load(renderer_path)
    histories=[];metadata=[];input_hashes={};initial=None;target=None;rotation=None;reduced_path=None
    for directory in episodes:
        files=[directory/'inputs.npz',directory/'executed.npz',directory/'metrics.json'];metrics=read(files[-1]);provenance_path=directory/'provenance.json'
        if not provenance_path.exists():provenance_path=directory.parent/'provenance.json'
        provenance=read(provenance_path);files.append(provenance_path);input_hashes.update({str(f):sha(f) for f in files})
        with np.load(files[0],allow_pickle=False) as z:x=z['initial_state'].copy();goal=z['target_xyz'].copy();R=z['target_rotation'].copy()
        with np.load(files[1],allow_pickle=False) as z:time=z['times'].astype(float);q=z['states'][:,:7].astype(float)
        if time.ndim!=1 or q.shape!=(len(time),7) or len(time)<2 or abs(time[0])>1e-8 or np.any(np.diff(time)<=0) or not np.isfinite(q).all() or not np.isfinite(time).all():raise ValueError('Invalid/nonfinite recorded trace; no sample filtering: '+str(directory))
        if initial is not None and (not np.allclose(x,initial,rtol=0,atol=1e-5) or not np.allclose(goal,target,rtol=0,atol=1e-5) or not np.allclose(R,rotation,rtol=0,atol=1e-5)):raise ValueError('Episode initial state/goal differs; this viewer compares one shared task')
        initial,target,rotation=x,goal,R;histories.append((time,q));model_path=Path(provenance['source'])/MODEL
        if reduced_path is not None and sha(model_path)!=sha(reduced_path):raise ValueError('Episode reduced URDFs differ')
        reduced_path=model_path;metadata.append(dict(episode=str(directory),duration_s=float(time[-1]),outcome=metrics['outcome'],task_success=metrics['task_success'],physical_pass=metrics['physical_pass'],execution_error=metrics['execution_error'],metrics=metrics,provenance=provenance,provenance_path=str(provenance_path),recorded_samples=len(time)))
    duration=max(t[-1] for t,q in histories);times=np.linspace(0,duration,int(np.ceil(duration*a.fps))+1)
    rendered_q=np.array([np.column_stack([np.interp(np.minimum(times,t[-1]),t,q[:,j]) for j in range(7)]) for t,q in histories])
    context=read(a.context);positions=context['positions_by_name'];model=pin.buildModelFromUrdf(str(a.full_urdf));data=model.createData();geom=pin.buildGeomFromUrdf(model,str(a.full_urdf),pin.GeometryType.VISUAL,None,[str(a.full_urdf.parent)]);gdata=pin.GeometryData(geom)
    names=[f'arm_right_{j}_joint' for j in range(1,8)];joint_ids=[model.getJointId(n) for n in names];tool=model.getFrameId('arm_right_tool_link');torso=model.getFrameId('torso_lift_link')
    if max(joint_ids)>=model.njoints or max(tool,torso)>=model.nframes:raise ValueError('TIAGo context model lacks required joints/frames')
    local=[renderer.simplify_visual_mesh(renderer._local_triangles(o)) for o in geom.geometryObjects];moving=[]
    for obj in geom.geometryObjects:
        parent=obj.parentJoint
        while parent and parent not in joint_ids:parent=model.parents[parent]
        moving.append(parent in joint_ids)
    moving_ids=[i for i,m in enumerate(moving) if m];base=renderer._q_from_positions(model,positions|dict(zip(names,initial[:7])));pin.framesForwardKinematics(model,data,base);torso_pose=data.oMf[torso].copy();pin.updateGeometryPlacements(model,data,geom,gdata,base)
    static=np.concatenate([local[i]@gdata.oMg[i].rotation.T+gdata.oMg[i].translation for i,m in enumerate(moving) if not m]);arm=pin.buildModelFromUrdf(str(reduced_path));adata=arm.createData();atool=arm.getFrameId('arm_right_tool_link');atorso=arm.getFrameId('torso_lift_link');parity=[];transforms=[];elbows=[];tools=[];cloud=[torso_pose.act(target)[None]]
    for trajectory in rendered_q:
        tt=[];ee=[];el=[]
        for joints in trajectory:
            full=renderer._q_from_positions(model,positions|dict(zip(names,joints)));pin.framesForwardKinematics(model,data,full);pin.updateGeometryPlacements(model,data,geom,gdata,full)
            tt.append([np.c_[gdata.oMg[i].rotation,gdata.oMg[i].translation].tolist() for i in moving_ids]);el.append(data.oMi[joint_ids[3]].translation.copy());ee.append(data.oMf[tool].translation.copy())
            for i in moving_ids:cloud.append((local[i]@gdata.oMg[i].rotation.T+gdata.oMg[i].translation).reshape(-1,3))
            pin.framesForwardKinematics(arm,adata,joints);f=torso_pose.inverse()*data.oMf[tool];r=adata.oMf[atorso].inverse()*adata.oMf[atool];parity.append([float(np.linalg.norm(f.translation-r.translation)),float(np.linalg.norm(pin.log3(f.rotation.T@r.rotation)))])
        transforms.append(tt);elbows.append(el);tools.append(ee)
    parity=np.max(parity,axis=0)
    if max(parity)>1e-4:raise ValueError('Material full/reduced display FK mismatch: '+str(parity))
    cloud=np.concatenate(cloud);center=(cloud.min(axis=0)+cloud.max(axis=0))/2;size=float(np.max(np.ptp(cloud,axis=0))+.25)
    payload=dict(times=times.tolist(),labels=a.labels,colors=[COLORS[i%len(COLORS)] for i in range(len(episodes))],static=np.round(static,5).tolist(),meshes=[np.round(local[i],5).tolist() for i in moving_ids],transforms=np.round(transforms,7).tolist(),elbows=np.round(elbows,6).tolist(),tools=np.round(tools,6).tolist(),goal=torso_pose.act(target).tolist(),goalRotation=(torso_pose.rotation@rotation).tolist(),center=center.tolist(),size=size,episodes=metadata)
    out.mkdir(parents=True);template=HERE/'comparison_player.html';encoded=json.dumps(payload,separators=(',',':'),allow_nan=False).replace('</','<\\/');(out/'comparison.html').write_text(template.read_text().replace('__PAYLOAD__',encoded));np.savez_compressed(out/'display_samples.npz',times=times,q=rendered_q,episode_end_times=np.array([t[-1] for t,q in histories]))
    plot(out,payload,a.gif,a.fps)
    assets={str(Path(o.meshPath)):sha(Path(o.meshPath)) for o in geom.geometryObjects if o.meshPath and Path(o.meshPath).is_file()};producers=[Path(__file__),template,renderer_path,a.renderer_source/'tiago_tools/replay_full_state_history.py',a.renderer_source/'tiago_tools/review_collision_pairs.py']
    receipt=dict(role='Recorded executed trajectories; display check only, no batching/minimum/collision claim',command=sys.argv,input_sha256=input_hashes,episodes=metadata,full_urdf_sha256=sha(a.full_urdf),reduced_urdf_sha256=sha(reduced_path),context=context,context_sha256=sha(a.context),producer_sha256={str(f):sha(f) for f in producers},mesh_sha256=assets,display_fps=a.fps,display_frames=len(times),maximum_full_reduced_FK_error_m_rad=parity.tolist(),interpolation='Linear in recorded joint positions for visual sampling only; final pose held after end and marked ENDED, no unrecorded state claim',versions=dict(numpy=np.__version__,pinocchio=pin.__version__),artifacts_sha256={f.name:sha(f) for f in out.iterdir() if f.is_file()})
    (out/'render_receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(output=str(out),frames=len(times),FK_parity=parity.tolist())))

if __name__=='__main__':main()
