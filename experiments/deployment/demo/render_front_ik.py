#!/usr/bin/env python3
"""Four full-body URDF IK poses; label failed endpoint/sequence checks explicitly."""
import json
from pathlib import Path
import sys
import numpy as np
import pinocchio as pin
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_front_screen import (load_visual_meshes, draw_robot, build_tiago_collision_model,
    NamedJointState,state_to_qv,DEFAULT_COLLISION_URDF_PATH,DEFAULT_COLLISION_BLACKLIST_PATH)


def main():
    folder=Path(sys.argv[1]);s=json.loads((folder/'screen.json').read_text())
    goals=s['selected'] or s['center_outcomes'][0]['goals'];bygoal={g['goal']:g for g in goals}
    labels=['bottom-right','top-right','top-left','bottom-left']
    for label in labels:
        if label not in bygoal:
            candidates=[r for r in s['attempts'] if r['goal']==label]
            if not candidates:raise ValueError('Missing screened corner: '+label)
            bygoal[label]=min(candidates,key=lambda r:r['position_error_m']+.15*r['orientation_error_rad'])
    collision=build_tiago_collision_model(urdf_path=DEFAULT_COLLISION_URDF_PATH,
        blacklist_path=DEFAULT_COLLISION_BLACKLIST_PATH,monitor_only_collision_pairs=True)
    meshes=load_visual_meshes(collision);data=collision.model.createData()
    targets=np.array([bygoal[label]['target_m'] for label in labels]);border=targets[[0,1,2,3,0]]
    fig=plt.figure(figsize=(12,15))
    for i,label in enumerate(labels):
        row=bygoal[label];ax=fig.add_subplot(2,2,i+1,projection='3d')
        body=dict(s['body_positions'],**{f'arm_right_{j+1}_joint':v for j,v in enumerate(row['q_rad'])})
        q,_=state_to_qv(collision.model,NamedJointState(position=body,velocity={k:0. for k in body}))
        pin.framesForwardKinematics(collision.model,data,q);draw_robot(ax,collision,data,meshes)
        torso=data.oMf[collision.model.getFrameId('torso_lift_link')].copy()
        tool=torso.inverse()*data.oMf[collision.model.getFrameId('arm_right_tool_link')]
        color='#238b45' if row['accepted_endpoint'] else '#c13d39'
        ax.plot(*border.T,color='#d08a17',lw=2,ls='--');ax.scatter(*targets.T,color='#d08a17',s=16)
        ax.scatter(*targets[i],color=color,s=65)
        ax.quiver(*targets[i],*(np.asarray(row['tool_axis'])*.10),color=color,arrow_length_ratio=.3,lw=2)
        # Expose the complete wrist orientation, with actual tool X/Y/Z triad.
        for j,axiscolor in enumerate(['#ba3979','#17a3a8','#202f6e']):
            ax.quiver(*tool.translation,*(tool.rotation[:,j]*.075),color=axiscolor,arrow_length_ratio=.25,lw=2)
        margin=row.get('wrist_margin_rad',row.get('joint_margin_rad',float('nan')))
        clearance=row.get('endpoint_min_m',float('nan'))
        ax.set_title(f"{i+1}. {label} — {'endpoint PASS' if row['accepted_endpoint'] else 'endpoint FAIL'}\n"
            f"position {1000*row['position_error_m']:.2f} mm / axis {np.rad2deg(row['orientation_error_rad']):.2f}°\n"
            f"clearance {1000*clearance:.1f} mm / wrist margin {margin:.2f} rad",fontsize=10,color=color)
        ax.set(xlim=(-.38,.75),ylim=(-.48,.42),zlim=(-.96,.52),
            xlabel='Torso X / forward (m)',ylabel='Torso Y / left (m)',zlabel='Torso Z (m)')
        ax.set_box_aspect((1.13,.9,1.48));ax.view_init(elev=14,azim=-35)
    center=s['centers_m'][0];valid=s['selected'] is not None
    fig.suptitle(f"{round(s['side_m']*100)} cm front square • torso center {center} m\n"
        +('All five IK/endpoint/path-witness checks pass' if valid else
          'Four corners pass; fifth center held; no motion qualification' if all(bygoal[l]['accepted_endpoint'] for l in labels) else
          'Incomplete IK sequence — failed corner preserved; no motion qualification'),fontsize=14)
    fig.text(.5,.02,'Static full-body URDF poses at CPU IK solutions. Torso lift 0.10 m; left arm/head fixed. Arrows: inward desired Z and actual full tool frame.\n'
        'Collision gates ≥40 mm. Travel/comfort and sampled joint interpolation are separate from endpoint feasibility. No executed robot/Gazebo/MPC motion.',
        ha='center',fontsize=9)
    fig.subplots_adjust(left=.01,right=.99,bottom=.09,top=.88,hspace=.32,wspace=.01)
    fig.savefig(folder/'four_corner_ik.png',dpi=150)


if __name__=='__main__':main()
