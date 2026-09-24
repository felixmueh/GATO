"""Offline TIAGo tool-route figures from retained transfer-probe arrays.

No simulation/optimization is run. Example: python plot_tiago.py ARTIFACT_DIRECTORY
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from plotting import STYLE, BLUE, ORANGE, INK, GRAY, _save

PURPLE = '#8060A8'


def route_color(lane, history=None, future=None, center=None):
    if history is not None:
        full=np.concatenate((history[:,:2],future[1:,:2]),axis=0)-np.asarray(center)
        if not np.isfinite(full).all(): return GRAY
        angles=np.unwrap(np.arctan2(full[:,1],full[:,0]))
        return BLUE if angles[-1]-angles[0] >= 0 else ORANGE
    return BLUE if lane.get('winding', 0.) >= 0 else ORANGE


def decorate(ax, trial, margin, bounds):
    center=np.asarray(trial['center_after'])
    ax.add_patch(plt.Circle(center, trial['radius']+margin, color='#E7EBEE', zorder=1))
    ax.add_patch(plt.Circle(center, trial['radius'], color='#697782', zorder=2))
    ax.add_patch(plt.Circle(trial['center_before'], trial['radius'], fill=False,
                            edgecolor='#697782', linestyle=':', linewidth=1, zorder=2))
    ax.scatter(*np.asarray(trial['goal'])[:2], marker='*', s=110, color=INK, zorder=8)
    ax.set(xlim=bounds[0], ylim=bounds[1], xlabel='Tool x [m]', ylabel='Tool y [m]')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(color='#DCE2E7', linewidth=.6)
    ax.set_axisbelow(True)


def plot(directory, output=None):
    directory=Path(directory); output=Path(output) if output else directory/'figures'
    payload=json.loads((directory/'summary.json').read_text())
    trial=sorted(payload['trials'], key=lambda t:(t['scene'],t['seed']))[0]
    prefix=f"scene{trial['scene']}_seed{trial['seed']}"
    batches=sorted(map(int,trial['branches']))
    batches=sorted(set((batches[0],batches[-1])))
    last=min(len(trial['branches'][str(b)]['stages']) for b in batches)-1
    stages=sorted(set((0,min(2,last),last)))
    arrays={(b,k):np.load(directory/f'{prefix}_b{b}_stage{k}.npz') for b in batches for k in stages}
    executed={b:np.load(directory/f'{prefix}_b{b}_executed.npz')['xyz'] for b in batches}
    points=np.concatenate([v['xyz'][...,:2].reshape(-1,2) for v in arrays.values()]+[
        np.asarray(trial['goal'])[None,:2],np.asarray(trial['center_before'])[None],
        np.asarray(trial['center_after'])[None]])
    points=points[np.isfinite(points).all(axis=1)]
    lo,hi=points.min(axis=0),points.max(axis=0)
    pad=max(float(np.max(hi-lo))*.075,.012)
    bounds=((lo[0]-pad,hi[0]+pad),(lo[1]-pad,hi[1]+pad))
    margin=payload['config']['margin']; advance=payload['config']['advance']
    paths=[]
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(len(batches),len(stages),figsize=(4.4*len(stages),3.8*len(batches)),
                              squeeze=False,layout='constrained')
        for i,b in enumerate(batches):
            for j,k in enumerate(stages):
                ax=axes[i,j]; record=trial['branches'][str(b)]['stages'][k]
                decorate(ax,trial,margin,bounds)
                xyz=arrays[b,k]['xyz']
                history=executed[b][:k*advance+1]
                for lane,path in zip(record['lanes'],xyz):
                    ax.plot(*path[:,:2].T,color=route_color(lane,history,path,trial['center_after']) if lane['feasible'] else GRAY,
                            lw=1.1,alpha=.55,ls='-' if lane['feasible'] else '--')
                winner=record['winner']; lane=record['lanes'][winner]
                ax.plot(*xyz[winner,:,:2].T,color=route_color(lane,history,xyz[winner],trial['center_after']),lw=3,
                        ls='-' if lane['feasible'] else '--')
                ax.plot(*history[:,:2].T,color=INK,lw=2.6,zorder=6)
                ax.scatter(*xyz[winner,0,:2],color='white',edgecolor=INK,s=36,zorder=8)
                count=sum(l['feasible'] for l in record['lanes'])
                ax.set_title(f"B={b} · replan {k} · {count}/{b} feasible\n"
                             f"predicted cost {lane['objective']['total']:.3f}",fontsize=11)
        experiment_label='TIAGo common-state first replan' if payload['config']['steps']==1 else 'TIAGo randomized MPC'
        fig.suptitle(f"{experiment_label} · scene {trial['scene']} · "
                     f"{'Held-out' if payload['config'].get('heldout') else 'Development pilot'}\n"
                     'Tool center · sampled coarse dynamics · no whole-arm or real-time claim',fontsize=13)
        handles=[Line2D([],[],color=INK,lw=2.6,label='Executed'),
                 Line2D([],[],color=GRAY,lw=3,label='Selected: thick'),
                 Line2D([],[],color=BLUE,lw=1.3,label='CCW route'),
                 Line2D([],[],color=ORANGE,lw=1.3,label='CW route'),
                 Line2D([],[],color=GRAY,ls='--',label='Infeasible'),
                 Line2D([],[],color='#697782',ls=':',label='Previous obstacle')]
        fig.legend(handles=handles,loc='outside lower center',ncol=6,frameon=False)
        paths.extend(_save(fig,output,'tiago_mpc_stages'))

        # Compare only the identical state at stage 0. Offline CPU references
        # must never be overlaid as if they were the online GATO output.
        fig,axes=plt.subplots(1,len(batches),figsize=(5*len(batches),4.8),squeeze=False,layout='constrained')
        for ax,b in zip(axes[0],batches):
            decorate(ax,trial,margin,bounds)
            record=trial['branches'][str(b)]['stages'][0]; winner=record['winner']
            selected=record['lanes'][winner]; initial=arrays[b,0]['seed_xyz']
            for path in initial:
                ax.plot(*path[:,:2].T,color=GRAY,alpha=.35,lw=.8)
            for lane,path in zip(record['lanes'],arrays[b,0]['xyz']):
                ax.plot(*path[:,:2].T,color=route_color(lane) if lane['feasible'] else GRAY,
                        alpha=.5,lw=1,ls='-' if lane['feasible'] else '--')
            ax.plot(*arrays[b,0]['xyz'][winner,:,:2].T,color=route_color(selected),lw=3,
                    label=f"GATO selected: {selected['objective']['total']:.3f}")
            for reference in sorted(directory.glob(f'{prefix}_b{b}_lane*_cpu.npz')):
                cpu=np.load(reference)['xyz']
                ax.plot(*cpu[:,:2].T,color=PURPLE,lw=2,ls='--',label='Independent CPU polish')
            ax.set_title(f'Identical switch state · B={b}',fontsize=12)
            ax.legend(frameon=False,fontsize=9)
        fig.suptitle('Online GATO routes and separate offline CPU diagnostics\n'
                     f"Scene {trial['scene']} · seed {trial['seed']} · "
                     f"{'Held-out' if payload['config'].get('heldout') else 'Development pilot'}",fontsize=13)
        fig.supxlabel('Gray: initial trajectories. Colored solid: GATO replay. Purple dashed: offline CPU result, never an online initializer.',fontsize=9)
        paths.extend(_save(fig,output,'tiago_common_state_routes'))
    return paths


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    for path in plot(args.directory,args.output): print(path)
