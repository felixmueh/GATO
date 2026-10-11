#!/usr/bin/env python3
"""Draw frozen demo positions/axes and IK endpoint clearance labels."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=json.loads(Path(sys.argv[1]).read_text());output=Path(sys.argv[2])
goals=p['goals'];xyz=np.array([g['pose_m_rad'][:3] for g in goals]);reference=np.asarray(p['tracking_tasks'][0]['reference_packed']).reshape(-1,6)[:801]
fig=plt.figure(figsize=(12,6));flat=fig.add_subplot(121);space=fig.add_subplot(122,projection='3d')
flat.plot(-xyz[:,1],xyz[:,2],'o-',label='Square sequence')
flat.plot(-xyz[[3,0],1],xyz[[3,0],2],':',color='gray')
for i,g in enumerate(goals):
 z=np.asarray(g['tool_axis']);flat.annotate(f"{i+1} {g['id']}\n{g['endpoint_minimum_collision_distance_m']*1000:.1f}mm",(-xyz[i,1],xyz[i,2]),fontsize=8,xytext=(6,6),textcoords='offset points')
 if i<4:flat.arrow(-xyz[i,1],xyz[i,2],-z[1]*.065,z[2]*.065,width=.002,color='red',length_includes_head=True)
 else:flat.annotate('Tool points toward robot\nnormal to square plane',(-xyz[i,1],xyz[i,2]),xytext=(-100,-45),textcoords='offset points',fontsize=8)
flat.set(xlabel='Robot-view right = -torso y (m)',ylabel='Torso z (m)',title='30cm square, x=0.50m');flat.axis('equal');flat.grid(alpha=.2)
space.plot(xyz[:,0],xyz[:,1],xyz[:,2],'o-',label='Square goals')
for i,g in enumerate(goals):space.quiver(*xyz[i],*g['tool_axis'],length=.07,color='red')
space.plot(reference[:,0],reference[:,1],reference[:,2],color='green',label='Forward-shifted figure8; +Zup')
for j in (0,200,400,600):space.quiver(*reference[j,:3],0,0,1,length=.065,color='green')
space.set(xlabel='Torso x (m)',ylabel='Torso y (m)',zlabel='Torso z (m)',title='Same frozen geometry in all3backends');space.legend(fontsize=8)
fig.suptitle('Visual demonstration proposal — arrows show tool+Z\nSquare IK endpoint clearance labels; all8 figure8 phase endpoints ≥40mm. Full motions/swept paths await review.',fontsize=10)
fig.tight_layout();fig.savefig(output,dpi=150)
