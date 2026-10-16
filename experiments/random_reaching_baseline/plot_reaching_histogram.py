#!/usr/bin/env python3
"""Saved-only arrival histogram with a distinct 8 s/no-arrival display bin."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(output,group_key=None,assigned=100):
    output=Path(output)
    with (output/'cases.csv').open() as f:rows=list(csv.DictReader(f))
    groups=list(dict.fromkeys(r[group_key] for r in rows)) if group_key else ['Fresh 100']
    samples=[]
    for r in rows:
        complete=r['complete']=='1';arrival=complete and r['task_success']=='1'
        status='arrival' if arrival else 'execution error' if complete and r['execution_failure']=='1' else 'timeout' if complete and r['outcome']=='timeout' else 'unrecorded'
        samples.append(dict(group=r[group_key] if group_key else groups[0],pair_id=r['pair_id'],status=status,actual_arrival_s=r['arrival_seconds'] if arrival else '',display_time_s=r['arrival_seconds'] if arrival else 8 if status in ('timeout','execution error') else '',completed_episode_physical_pass=r['completed_episode_physical_pass']))
    with (output/'reaching_histogram_samples.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(samples[0]));writer.writeheader();writer.writerows(samples)
    cols=1 if len(groups)==1 else 2;nr=(len(groups)+cols-1)//cols
    fig,axes=plt.subplots(nr,cols,figsize=(8.5 if cols==1 else 11.6,3.6*nr),squeeze=False)
    for ax,group in zip(axes.ravel(),groups):
        selected=[r for r in samples if r['group']==group];times=[float(r['actual_arrival_s']) for r in selected if r['status']=='arrival']
        timeout=sum(r['status']=='timeout' for r in selected);errors=sum(r['status']=='execution error' for r in selected);missing=sum(r['status']=='unrecorded' for r in selected)
        ax.hist(times,bins=np.arange(0,8.01,.5),color='#277da8',edgecolor='white',label='Recorded arrival')
        ax.bar(8.35,timeout,width=.45,color='#d99535',edgecolor='#795211',label='Timeout: no arrival')
        ax.bar(8.35,errors,bottom=timeout,width=.45,color='#bb4f3f',edgecolor='#632b23',hatch='///',label='Execution error: no arrival')
        ax.axvline(8.05,color='gray',ls=':',lw=1)
        peak=max(int(np.histogram(times,bins=np.arange(0,8.01,.5))[0].max()),timeout+errors,1)
        ax.set_ylim(0,peak*1.15)
        ax.set(xlim=(0,8.7),xticks=[0,2,4,6,8.35],xticklabels=['0','2','4','6','8 s /\nno arrival'],xlabel='Recorded arrival time (s); separate non-arrival display bin',ylabel=f'Task count / assigned {assigned}',title=(f'{group_key} {group}' if group_key else group)+f': {len(times)} arrivals, {timeout} timeouts, {errors} errors')
        if missing:ax.text(.02,.98,f'{missing} unrecorded: no inferred bin',transform=ax.transAxes,va='top',fontsize=8)
        ax.grid(axis='y',alpha=.2);ax.legend(fontsize=7)
    for ax in axes.ravel()[len(groups):]:ax.set_visible(False)
    fig.text(.5,.015,'The 8 s/no-arrival bin is a visualization convention, not an arrival or actual error runtime.\nOriginal outcomes and physical flags are unchanged.',ha='center',va='bottom',fontsize=8)
    fig.tight_layout(rect=(0,.10,1,1));fig.savefig(output/'reaching_histogram.png',dpi=180);fig.savefig(output/'reaching_histogram.pdf');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--group-key');p.add_argument('--assigned',type=int,default=100);a=p.parse_args();plot(a.output,a.group_key,a.assigned)
