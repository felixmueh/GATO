#!/usr/bin/env python3
"""Full-robot URDF animation at sampled actual recorded feedback states."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
import pinocchio as pin
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from plot_front_screen import load_visual_meshes, build_tiago_collision_model, NamedJointState, state_to_qv, DEFAULT_COLLISION_URDF_PATH, DEFAULT_COLLISION_BLACKLIST_PATH


def main():
    folder=Path(sys.argv[1]);p=json.loads((folder/'plot_inputs.json').read_text());protocol=p['protocol'];rows=list(csv.DictReader((folder/'motion_frames.csv').open()))
    collision=build_tiago_collision_model(urdf_path=DEFAULT_COLLISION_URDF_PATH,blacklist_path=DEFAULT_COLLISION_BLACKLIST_PATH,monitor_only_collision_pairs=True)
    meshes=load_visual_meshes(collision);data=collision.model.createData();body=protocol['CPU_selection']['body_positions']
    poses=[];tips=[];rotations=[];qs=[]
    for row in rows:
        q=np.asarray([float(row[f'q{i+1}_rad']) for i in range(7)]);positions=dict(body,**{f'arm_right_{j+1}_joint':float(v) for j,v in enumerate(q)})
        cq,_=state_to_qv(collision.model,NamedJointState(position=positions,velocity={k:0. for k in positions}));qs.append(cq)
        pin.framesForwardKinematics(collision.model,data,cq);torso=data.oMf[collision.model.getFrameId('torso_lift_link')].copy()
        transforms=[torso.inverse()*data.oMf[fid]*local for fid,local,vertices,faces,arm in meshes];poses.append(transforms)
        tool=torso.inverse()*data.oMf[collision.model.getFrameId('arm_right_tool_link')];tips.append(tool.translation.copy());rotations.append(tool.rotation.copy())
    tips=np.asarray(tips);rotations=np.asarray(rotations);targets=np.asarray([g['pose_m_rad'][:3] for g in protocol['goals']])
    fig=plt.figure(figsize=(9,8));ax=fig.add_subplot(111,projection='3d');fig.subplots_adjust(left=.015,right=.985,bottom=.12,top=.83)
    artists=[]
    for mesh,pose in zip(meshes,poses[0]):
        fid,local,vertices,faces,arm=mesh;world=vertices@pose.rotation.T+pose.translation
        collection=Poly3DCollection(world[faces],facecolor='#3874a6' if arm else '#a6adb2',alpha=.93,edgecolor='none');ax.add_collection3d(collection);artists.append(collection)
    ax.plot(*targets[[0,1,2,3,0]].T,color='#cf8b18',lw=2,ls='--')
    for i,(goal,target) in enumerate(zip(protocol['goals'],targets)):
        color='#238b45' if p['per_goal'][i]['rough_pass'] else '#b73c36';ax.scatter(*target,color=color,s=26)
        ax.quiver(*target,*(np.asarray(goal['tool_axis'])*.065),color=color,arrow_length_ratio=.3)
    trail,=ax.plot([],[],[],color='#136da1',lw=2);tool_axis,=ax.plot([],[],[],color='#122549',lw=2)
    tool_x,=ax.plot([],[],[],color='#bc3e79',lw=2);tool_y,=ax.plot([],[],[],color='#18a3a7',lw=2)
    ax.set(xlim=(-.4,.85),ylim=(-.55,.45),zlim=(-.97,.52),xlabel='Torso X / forward (m)',ylabel='Torso Y / left (m)',zlabel='Torso Z (m)')
    ax.set_box_aspect((1.25,1.,1.49));ax.view_init(elev=18,azim=-35)
    fig.suptitle('20 cm front square — actual saved internal MPC motion',fontsize=14,y=.98)
    m=p['metrics'];cm=p['collision_metrics'];summary=f"Rough {sum(g['rough_pass'] for g in p['per_goal'])}/5 • Strict {sum(g['strict_outcome']=='reached' for g in p['per_goal'])}/5 • q/v/applied-control {'PASS' if m['physical_pass'] else 'FAIL'}"
    fig.text(.5,.942,summary,ha='center',fontsize=10)
    fig.text(.5,.911,f"Full-body collision {'PASS' if cm['all_sampled_at_least_40mm'] else 'FAIL'}: min {cm['minimum']['minimum_clearance_m']*1000:.1f} mm • Below 40 mm: {cm['below_40mm']}/{cm['collision_samples']}",ha='center',fontsize=9,color='#238b45' if cm['all_sampled_at_least_40mm'] else '#b2322b')
    heading=fig.text(.5,.874,'',ha='center',fontsize=10);error=fig.text(.5,.842,'',ha='center',fontsize=9)
    failure=fig.text(.5,.5,'',ha='center',va='center',fontsize=14,color='#b2322b',visible=False)
    def update(i):
        if i>=len(rows):
            ax.set_visible(False);failure.set_visible(True);heading.set_text('Recorded finite motion ended; failure preserved');error.set_text('No motion reconstructed after invalid/failed execution')
            failure.set_text('Internal diagnostic execution failed\n\n'+('First invalid feedback: '+str(p['animation_failure']['first_invalid_time_s'])+' s' if p['animation_failure'] else str(m['execution_error'])[:200]));return
        for mesh,pose,artist in zip(meshes,poses[i],artists):
            fid,local,vertices,faces,arm=mesh
            if arm:artist.set_verts((vertices@pose.rotation.T+pose.translation)[faces])
        trail.set_data_3d(*tips[:i+1].T)
        for j,artist in enumerate((tool_x,tool_y,tool_axis)):artist.set_data_3d(*np.asarray([tips[i],tips[i]+.10*rotations[i,:,j]]).T)
        goal=int(rows[i]['goal_index']);desired=pin.rpy.rpyToMatrix(*protocol['goals'][goal]['pose_m_rad'][3:])
        distance=float(np.linalg.norm(tips[i]-targets[goal]));axis=float(np.rad2deg(np.arccos(np.clip(rotations[i,:,2]@desired[:,2],-1,1))))
        heading.set_text(f"Recorded t={float(rows[i]['time_s']):.3f} s • Goal {goal+1}: {protocol['goals'][goal]['id']}")
        clearance=p.get('frame_collision_samples',[{}]*len(rows))[i].get('sampled_clearance_m')
        label=f" • Clearance ≈{clearance*1000:.1f} mm" if clearance is not None else ''
        error.set_text(f'Position error {distance*1000:.1f} mm • Tool-axis error {axis:.1f}°'+label)
        error.set_color('#b2322b' if clearance is not None and clearance<.04 else '#222222')
    fig.text(.5,.06,'Full-body URDF at actual feedback samples; torso/left arm/head fixed at the declared setup. No IK animation.',ha='center',fontsize=8)
    fig.text(.5,.035,'Collision sampled at 10 ms; no swept certificate or live qualification. Playback accelerated; raw motion retained.',ha='center',fontsize=8)
    update(0);fig.savefig(folder/'first_frame.png',dpi=120);update(len(rows)-1);fig.savefig(folder/'last_frame.png',dpi=120)
    extra=24 if p['animation_failure'] or m['execution_error'] else 0
    FuncAnimation(fig,update,frames=len(rows)+extra,interval=1000/12,blit=False).save(folder/'motion.gif',writer=PillowWriter(fps=12),dpi=90)
    plt.close(fig);print('saved',len(rows),'actual-state frames,',extra,'failure-card frames',flush=True)


if __name__=='__main__':main()
