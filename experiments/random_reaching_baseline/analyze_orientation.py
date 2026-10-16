#!/usr/bin/env python3
"""CPU-only saved-record comparison of M14, orientation OFF and orientation LOW."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
MODEL = 'gato/dynamics/tiago_right/tiago_right_arm.urdf'
BASELINE_SHA = '786f98c8c4762af29342befe7ed169fdce8e9389a4799d61820849720f4ab7fb'
ARMS=('M14','OFF','LOW','LOW_TORQUE','OFF_TORQUE')
OFF_TORQUE_WAIVER_SHA='b6cf8f70c6f404ca2327453d0e14c4950b7d916f6d4b35457fbb29913d7afc8d'


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def finite(x): return x is not None and np.isfinite(x)
def val(x): return float(x) if finite(x) else None
def num(x): return format(float(x), '.5g') if finite(x) else 'null'
def write(path, data): Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
def csv_read(path):
    with Path(path).open() as f: return list(csv.DictReader(f))
def csv_write(path, rows, fields=None):
    with Path(path).open('w') as f:
        w=csv.DictWriter(f, fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)


def input_hashes(path):
    with np.load(path, allow_pickle=False) as z:
        return {k:hashlib.sha256(np.ascontiguousarray(z[k]).tobytes()).hexdigest()
            for k in ('initial_state','source_goal_q','target_xyz','target_rotation')}


def sensitivity(directory,arm,pair,metric,scope):
    path=directory/'solves.jsonl'
    work=[json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []
    rows=[]
    for threshold in (.05,.075,.1):
        times=[w['feedback_time_s']+w['executed_duration_s'] for w in work if w.get('final') and
            finite(w['final']['position_error_m']) and finite(w['final']['velocity_l1_rad_s']) and
            w['final']['position_error_m']<=threshold and w['final']['velocity_l1_rad_s']<=1.]
        rows.append(dict(arm=arm,scope=scope,pair_id=pair,position_threshold_m=threshold,
            observed_arrival=int(bool(times)),first_observed_feedback_s=min(times) if times else None,
            original_outcome=metric['outcome'],original_task_success=int(metric['task_success']),
            observed_duration_s=metric['simulated_seconds'],execution_error=metric['execution_error']))
    return rows


def diagnostic_waiver(arm,root,protocol,prem,expected):
    path=root/'diagnostic_user_waiver.json'
    if not path.is_file():return None
    if arm!='OFF_TORQUE' or sha(path)!=OFF_TORQUE_WAIVER_SHA:
        raise ValueError('No diagnostic preflight exception is authorized for this arm/receipt')
    record=read(path)
    required=dict(arm='OFF_TORQUE',user_answer='Run the 12 diagnostic reaches (Recommended)',
        source='Explicit async answer from Felix, 2026-10-15',
        protocol_sha256=sha(root/'preflight/protocol.json'),
        canonical_metrics_sha256=sha(root/'preflight/preflight/metrics.json'),
        canonical_preflight_pass=False,canonical_physical_pass=True,task_ids=expected,
        arrival_policy=protocol['arrival_policy'],position_tolerance_m=protocol['position_tolerance_m'],
        velocity_l1_tolerance_rad_s=protocol['velocity_l1_tolerance_rad_s'],
        orientation_required_for_arrival=False,original_failure_preserved=True,extends_to_other_arms=False)
    if any(record.get(k)!=v for k,v in required.items()) or not prem or prem['preflight_pass'] or not prem['physical_pass'] or prem['execution_error']:
        raise ValueError('Diagnostic waiver does not match the preserved failed physical-pass canonical')
    return dict(path=str(path),sha256=sha(path),record=record)


def collect(arm, root, protocol, base_provenance, expected, paired, limits, rescore=None):
    pilot=root/'pilot'; provenance=read(pilot/'provenance.json') if (pilot/'provenance.json').is_file() else None
    pre=root/'preflight'; prem_path=pre/'preflight/metrics.json'
    prem=read(prem_path) if prem_path.is_file() else None
    preprov=read(pre/'provenance.json') if (pre/'provenance.json').is_file() else None
    waiver=diagnostic_waiver(arm,root,protocol,prem,expected)
    hashes={}; rows=[]; traces=[]; thresholds=[]
    if arm!='M14':
        for measured in (provenance,preprov):
            if measured is None: continue
            if measured['protocol']!=protocol: raise ValueError('Measured protocol differs: '+arm)
            for key in ('source_sha256','module_sha256','inputs_sha256'):
                if measured[key]!=base_provenance[key]: raise ValueError('Unchanged identity differs: '+arm+'/'+key)
        if provenance and not (prem and prem['preflight_pass']) and not waiver:
            raise ValueError('Pilot launched without matching passed canonical preflight: '+arm)
        if provenance and waiver and datetime.fromisoformat(waiver['record']['approved_utc'])>=datetime.fromisoformat(provenance['created_utc']):
            raise ValueError('Diagnostic approval must precede pilot execution')
        if provenance and preprov and provenance['wrapper_sha256']!=preprov['wrapper_sha256']:
            raise ValueError('Preflight/pilot worker differs: '+arm)
    if provenance and provenance['pairs']!=expected: raise ValueError('Different ordered development12: '+arm)
    summary=read(pilot/'summary.json') if (pilot/'summary.json').is_file() else []
    if [m['pair_id'] for m in summary]!=expected[:len(summary)]: raise ValueError('Summary order differs: '+arm)
    saved={m['pair_id']:m for m in summary}
    complete=len(summary)==12 and all(m['complete'] for m in summary)
    if complete: state='complete'
    elif not provenance and waiver: state='awaiting_approved_diagnostic_pilot'
    elif not provenance and prem and not prem['preflight_pass']: state='held_preflight_failure'
    elif any(m.get('shared_setup_failure') or m.get('interrupted') for m in summary): state='held_pilot_execution'
    elif provenance: state='partial'
    elif prem and prem['preflight_pass']: state='awaiting_pilot'
    elif preprov: state='preflight_in_progress'
    else: state='not_recorded'
    for order,pair in enumerate(expected,1):
        directory=pilot/f'pair_{pair:05d}'; path=directory/'metrics.json'
        m=read(path) if path.is_file() else None
        if pair in saved and saved[pair]!=m: raise ValueError('Summary/case mismatch: '+arm+'/'+str(pair))
        accepted=pair in saved and bool(m and m['complete'])
        original_success=bool(m['task_success']) if accepted else None
        primary=(rescore[pair]['position_speed_feedback_arrival']=='True') if accepted and rescore else original_success
        arrival=float(rescore[pair]['position_speed_feedback_first_s']) if primary and rescore else val(m['reached_seconds']) if accepted else None
        p=m['physical'] if m else None
        physical=bool(accepted and not m['execution_error'] and p['finite'] and m['physical_pass'])
        row=dict(arm=arm,order=order,pair_id=pair,diagnostic_user_waiver=int(bool(waiver)),status=m['outcome'] if accepted else state if state.startswith('held') else 'in_progress' if directory.exists() else 'not_recorded',
            complete=int(accepted),original_outcome=m['outcome'] if m else None,
            original_task_success=int(original_success) if accepted else None,
            primary_position_speed_arrival=int(primary) if accepted else None,
            primary_arrival_seconds=arrival,raw_sampled_physical_pass=int(m['physical_pass']) if accepted else None,
            completed_episode_physical_pass=int(physical) if accepted else None,
            both_pass=int(primary and physical) if accepted else None,
            execution_error=m['execution_error'] if m else None,execution_failure=int(bool(m['execution_error'])) if accepted else None,
            interrupted=int(bool(m.get('interrupted'))) if m else None,
            observed_seconds=val(m['simulated_seconds']) if m else None,
            final_position_mm=val(m['final']['position_error_m']*1000) if m and finite(m['final']['position_error_m']) else None,
            final_tool_axis_deg=val(m['final']['tool_axis_error_deg']) if m else None,
            final_orientation_deg=val(m['final']['orientation_error_deg']) if m else None,
            final_speed_L1_rad_s=val(m['final']['velocity_l1_rad_s']) if m else None,
            maximum_q_excess_rad=val(p['joint']['maximum_excess']) if accepted and not m['execution_error'] and p['finite'] else None,
            minimum_q_clearance_rad=None,
            maximum_speed_ratio=val(p['maximum_velocity_ratio']) if accepted and not m['execution_error'] and p['finite'] else None,
            maximum_torque_ratio=val(p['maximum_effort_ratio']) if accepted and not m['execution_error'] and p['finite'] else None,
            feedback_solves=m['feedback_solves'] if m else None,actual_sqp_sum=m['actual_sqp_sum'] if m else None,
            zero_pcg_exit_solves=m['zero_pcg_exit_solves'] if m else None,
            episode_wall_s=val(m['episode_wall_s']) if m else None)
        for category in ('joint','velocity','effort'):
            first=p[category]['first_violation'] if p else None
            row[category+'_pass']=int(p[category]['pass_check']) if p else None
            row[category+'_first_joint']=first['joint_name'] if first else None
            row[category+'_first_time_s']=val(first['time_s']) if first else None
        inputs=directory/'inputs.npz'
        if inputs.is_file():
            measured=input_hashes(inputs)
            if arm=='M14': paired[pair]=measured
            elif measured!=paired[pair]: raise ValueError('Paired physical input changed: '+arm+'/'+str(pair))
        history=directory/'executed.npz'
        if accepted and not m['execution_error'] and p['finite'] and history.is_file():
            with np.load(history,allow_pickle=False) as z:
                q=z['states'][:,:7]
                row['minimum_q_clearance_rad']=float(np.min(np.minimum(q-limits['lower'],limits['upper']-q))) if np.isfinite(q).all() else None
        hashes[str(pair)]={f.name:sha(f) for f in (path,inputs,history,directory/'solves.jsonl') if f.is_file()}
        if accepted:thresholds+=sensitivity(directory,arm,pair,m,'development12')
        rows.append(row)
    if arm!='M14' and prem and (pre/'preflight/executed.npz').is_file():
        with np.load(pre/'preflight/executed.npz',allow_pickle=False) as z:
            traces=[dict(arm=arm,time_s=float(t),position_error_mm=val(pos*1000),tool_axis_deg=val(axis),speed_L1_rad_s=val(speed))
                for t,pos,axis,speed in zip(z['times'],z['position_error_m'],z['tool_axis_error_deg'],z['velocity_l1_rad_s'])]
        if arm.startswith('OFF') and not provenance:thresholds+=sensitivity(pre/'preflight',arm,-1,prem,'canonical_only')
    measured=[r for r in rows if r['complete']]
    count=dict(arm=arm,state=state,diagnostic_user_waiver=bool(waiver),assigned=12,recorded=len(measured),
        arrivals=sum(r['primary_position_speed_arrival'] for r in measured) if measured else None,
        physical=sum(r['completed_episode_physical_pass'] for r in measured) if measured else None,
        both=sum(r['both_pass'] for r in measured) if measured else None,
        timeouts=sum(r['original_outcome']=='timeout' and not r['primary_position_speed_arrival'] for r in measured) if measured else None,
        execution_failures=sum(r['execution_failure'] for r in measured) if measured else None,
        episode_wall_s=sum(r['episode_wall_s'] or 0 for r in measured) if measured else None)
    for key in ('maximum_q_excess_rad','maximum_speed_ratio','maximum_torque_ratio','minimum_q_clearance_rad'):
        values=[r[key] for r in measured]
        count[key]=(min(values) if key.startswith('minimum') else max(values)) if values and all(finite(v) for v in values) else None
    info=dict(arm=arm,raw_root=str(root),state=state,protocol=protocol,provenance=provenance,diagnostic_user_waiver=waiver,
        canonical=prem,canonical_provenance=preprov,case_sha256=hashes,
        summary_sha256=sha(pilot/'summary.json') if summary else None,
        canonical_sha256=sha(prem_path) if prem else None)
    return info,count,rows,traces,thresholds


def plots(output):
    rows=csv_read(output/'cases.csv'); counts=read(output/'counts.json')
    colors={'M14':'#626b73','OFF':'#d97706','LOW':'#147fa3','LOW_TORQUE':'#8046a5','OFF_TORQUE':'#359b62'}
    fig,ax=plt.subplots(figsize=(7.2,4.3))
    for count in counts:
        arm=count['arm']
        if not count['recorded']: continue
        times=sorted(float(r['primary_arrival_seconds']) for r in rows if r['arm']==arm and r['primary_position_speed_arrival']=='1')
        ax.step([0,*times,8],[0,*np.arange(1,len(times)+1)/12,len(times)/12],where='post',label=f"{arm}: {count['recorded']}/12 recorded",color=colors[arm])
    held=[r['arm'] for r in counts if not r['recorded']]
    if held:ax.text(.02,.97,', '.join(held)+' pilot not run; no arrival-rate curve',transform=ax.transAxes,va='top',fontsize=9)
    ax.set(xlim=(0,8),ylim=(0,1),xlabel='Simulated time (s)',ylabel='Position+speed arrivals / assigned12')
    ax.legend(loc='upper right',fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/'arrival_ecdf.png',dpi=180);fig.savefig(output/'arrival_ecdf.pdf');plt.close(fig)
    for name,keys,labels,thresholds in (
        ('endpoints',('final_position_mm','final_tool_axis_deg','final_orientation_deg'),('Endpoint position error (mm)','Endpoint tool-axis error (deg)','Endpoint full orientation error (deg)'),(50,None,None)),
        ('physical_maxima',('maximum_q_excess_rad','maximum_speed_ratio','maximum_torque_ratio'),('Maximum q excess (rad)','Maximum speed / physical limit','Maximum torque / physical limit'),(0,1,1))):
        fig,axes=plt.subplots(1,3,figsize=(11.6,4.1))
        for ax,key,label,threshold in zip(axes,keys,labels,thresholds):
            for offset,arm in zip(np.linspace(-.3,.3,len(ARMS)),ARMS):
                subset=[r for r in rows if r['arm']==arm and r[key]!='']
                for passed,marker in ((True,'o'),(False,'x')):
                    group=[r for r in subset if (r['completed_episode_physical_pass']=='1')==passed]
                    if group:ax.scatter([int(r['order'])+offset for r in group],[float(r[key]) for r in group],s=22,color=colors[arm],marker=marker,label=arm if passed else None)
            if threshold is not None:ax.axhline(threshold,color='black',ls='--',lw=.8)
            ax.set(xlim=(.4,12.6),xlabel='Frozen development task order',ylabel=label);ax.grid(alpha=.2)
        axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(output/(name+'.png'),dpi=180);fig.savefig(output/(name+'.pdf'));plt.close(fig)
    trace=csv_read(output/'canonical_trace.csv')
    if trace:
        fig,axes=plt.subplots(1,2,figsize=(10.8,3.8))
        for arm in ARMS[1:]:
            group=[r for r in trace if r['arm']==arm]
            if not group:continue
            for ax,key in zip(axes,('position_error_mm','speed_L1_rad_s')):
                valid=[r for r in group if r[key]!='']
                ax.plot([float(r['time_s']) for r in valid],[float(r[key]) for r in valid],label=arm,color=colors[arm])
        for ax,label,gate in zip(axes,('Canonical position error (mm)','Canonical speed L1 (rad/s)'),(50,1)):
            ax.axhline(gate,color='black',ls='--',lw=.8);ax.set(xlabel='Simulated time (s)',ylabel=label);ax.grid(alpha=.2);ax.legend()
        fig.tight_layout();fig.savefig(output/'canonical_trace.png',dpi=180);fig.savefig(output/'canonical_trace.pdf');plt.close(fig)


def report(output,info,counts,rows,final,pending_waiver=False):
    lines=['# Reaching: orientation and ordinary torque costs','',
        ('Report of completed episodes and explicitly held pilots; no unrecorded task outcomes are inferred.' if final else 'INCOMPLETE PREVIEW; unrecorded cases are not failed episodes.'), '',
        'The original uniform-joint/FK task sampler and the exact twelve development pairs are retained. Primary arrival in this comparison requires position error <=50mm and joint-speed L1 <=1rad/s at a 30ms feedback boundary. Orientation is descriptive, with no orientation arrival condition. Physical compliance and arrival remain separate. No arm is automatically selected and no tuning follows from this report.','',
        '## Scope and outcomes','', '| Arm | Pilot state | Recorded /12 | Arrival | Physical | Both | Timeouts | Errors |','|---|---|---:|---:|---:|---:|---:|---:|']
    low=next(c for c in counts if c['arm']=='LOW');torque=next(c for c in counts if c['arm']=='LOW_TORQUE')
    if low['state']=='complete' and torque['state']=='complete':
        headline=f"LOW recorded {low['arrivals']}/12 position+speed arrivals; the lower ordinary torque regularization contrast recorded {torque['arrivals']}/12. "\
            f"Sampled physical passes were {low['physical']}/12 and {torque['physical']}/12, respectively. "
        off=next(c for c in counts if c['arm']=='OFF_TORQUE')
        if off['state']=='complete':
            headline+=f"At the same lower torque weight, OFF_TORQUE recorded {off['arrivals']}/12 arrivals and {off['physical']}/12 physical passes in its explicitly approved diagnostic pilot. "
        lines[2:2]=[headline+'These are bounded paired development observations, not a reliability guarantee or an automatic configuration choice.','']
    if pending_waiver:
        lines[2:2]=['Review checkpoint: the OFF_TORQUE diagnostic-pilot waiver question is awaiting the user\'s answer. No OFF or OFF_TORQUE pilot has been authorized or launched. Both failed canonical preflights and their original5cm criteria remain preserved. Any later explicitly approved diagnostic pilot would require a new report version retaining those failures; this report makes no claim that the pending decision is resolved.','']
    for c in counts:
        fields=[str(c[k])+'/12' if c[k] is not None else 'not run' for k in ('arrivals','physical','both')]
        state=c['state']+(' / diagnostic waiver' if c['diagnostic_user_waiver'] else '')
        lines.append(f"| {c['arm']} | {state} | {c['recorded']}/12 | {' | '.join(fields)} | {c['timeouts'] if c['timeouts'] is not None else 'not run'} | {c['execution_failures'] if c['execution_failures'] is not None else 'not run'} |")
    far=[r for r in rows if r['arm']=='OFF_TORQUE' and r['complete'] and r['original_outcome']=='timeout' and finite(r['final_position_mm']) and r['final_position_mm']>100]
    if far:
        lines+=['', 'OFF_TORQUE timeouts still ending above10cm: '+', '.join(f"ID{r['pair_id']} ({num(r['final_position_mm'])}mm)" for r in far)+'. These failures remain part of the twelve-task denominator.']
    lines+=['','M14 is saved orientation-on motion rescored for position+speed at the original feedback endpoints. Its original rough-arrival labels and outcomes remain unchanged in cases.csv; the rescore also gives1/12. It is not rerun or treated as motion under OFF/LOW costs. A failed canonical normally holds the full pilot; OFF remains held with no twelve-task failure rate inferred.','']
    waived=next((item['diagnostic_user_waiver'] for item in info if item['diagnostic_user_waiver']),None)
    if waived:
        lines+=['OFF_TORQUE is a user-approved diagnostic pilot despite its failed canonical. The original5cm position+speed gate and all parameters are unchanged. Its canonical remains failed, while OFF remains held. The exact written approval is retained in diagnostic_user_waiver.json and metadata.json; the earlier pending-answer report remains a separate raw checkpoint.','']
    lines+=[
        '## Canonical preflights','', '| Arm | Outcome | Primary arrival | Physical prefix | Preflight gate | Final mm | Axis deg | Speed L1 |','|---|---|---|---|---|---:|---:|---:|']
    for item in info:
        if item['arm']=='M14':continue
        m=item['canonical']
        if m:lines.append(f"| {item['arm']} | {m['outcome']} | {m['task_success']} | {m['physical_pass']} | {m['preflight_pass']} | {num(m['final']['position_error_m']*1000) if finite(m['final']['position_error_m']) else 'null'} | {num(m['final']['tool_axis_error_deg'])} | {num(m['final']['velocity_l1_rad_s'])} |")
        else:lines.append(f"| {item['arm']} | not complete | — | — | — | — | — | — |")
    if (output/'canonical_trace.png').exists():lines+=['','![Canonical episodes only, separate from the twelve-task pilots. Every saved dense sample is retained in canonical_trace.csv. Dashed lines mark position and speed conditions; orientation is descriptive. A finite physically compliant timeout still fails preflight; only the separate explicit OFF_TORQUE user waiver permits its diagnostic pilot.](canonical_trace.png)','']
    lines+=['## Saved-trace arrival-threshold sensitivity','',
        'These are descriptive rescoring results at5/7.5/10cm, always with speedL1<=1rad/s and no orientation gate. They do not change original5cm labels, relax a preflight or deploy a new threshold. Each pilot uses its fixed12-task denominator; OFF canonical is explicitly one different task. LOW and LOW_TORQUE observations end at their original5cm arrival or timeout, while M14 observations end under its original position+axis+speed rule. Unobserved continuation or settling cannot be inferred.','',
        '| Arm / scope | Recorded | 5cm observed | 7.5cm observed | 10cm observed |','|---|---:|---:|---:|---:|']
    threshold_rows=csv_read(output/'arrival_sensitivity.csv')
    for arm in ARMS:
        group=[r for r in threshold_rows if r['arm']==arm]
        if not group:continue
        n=len(group)//3
        values=[sum(r['observed_arrival']=='1' for r in group if float(r['position_threshold_m'])==t) for t in (.05,.075,.1)]
        canonical=group[0]['scope']=='canonical_only';scope='canonical only' if canonical else 'development12';denominator=1 if canonical else 12
        lines.append(f"| {arm} / {scope} | {n} | {values[0]}/{denominator} | {values[1]}/{denominator} | {values[2]}/{denominator} |")
    lines+=['## Paired development observations','',
        '![Every curve uses the fixed twelve-task denominator. A held arm with no pilot observations has no curve. Incomplete curves are observed lower bounds, not completed rates. Timeouts are censored at eight seconds; execution errors remain errors with their observed durations and are not assigned an arrival.](arrival_ecdf.png)','',
        '![Paired task indices follow the original frozen order. Position and both orientation errors are recorded at each episode endpoint, which can be an early arrival or eight-second timeout. These are not fixed-time or post-arrival settling comparisons. Orientation panels are descriptive. Circles show error-free sampled physical passes; crosses retain failed observations.](endpoints.png)','',
        '![Actual physical extrema over each recorded episode, including early transients. The unchanged raw URDF checks use q excess<=1e-6rad and speed/torque ratios<=1+1e-6. Error/nonfinite full histories remain null rather than zero. Physical compliance does not imply arrival or continuous-time safety.](physical_maxima.png)','',
        '## Physical extrema by recorded arm','', '| Arm | Min q clearance rad | Max q excess rad | Max speed ratio | Max torque ratio |','|---|---:|---:|---:|---:|']
    for c in counts:lines.append(f"| {c['arm']} | {num(c['minimum_q_clearance_rad'])} | {num(c['maximum_q_excess_rad'])} | {num(c['maximum_speed_ratio'])} | {num(c['maximum_torque_ratio'])} |")
    lines+=['','Null means the full recorded-arm extremum is unavailable, including an error/nonfinite history or an arm with no pilot. No finite failed episode is omitted. A preview covers only its recorded prefix of assigned tasks.','',
        '## All case outcomes','', 'CSV rows retain every planned task, original labels, rescored/primary arrival times, descriptive errors, work, physical flags and first-violation names/times. Held/unrecorded cases carry null outcome fields.']
    for item in info:
        arm=item['arm'];subset=[r for r in rows if r['arm']==arm and r['complete']]
        lines+=['','### '+arm,'']
        if not subset:
            lines+=['No pilot outcomes recorded; the twelve assigned cases were held or are unrecorded. No0/12 success or physical-failure rate is asserted.'];continue
        lines+=['| ID | Original result | Primary / physical | Arrival s | End mm | Axis deg | Speed L1 | SQP / solves |','|---|---|---|---:|---:|---:|---:|---:|']
        for r in subset:lines.append(f"| {r['pair_id']} | {r['original_outcome']} | {r['primary_position_speed_arrival']}/{r['completed_episode_physical_pass']} | {num(r['primary_arrival_seconds'])} | {num(r['final_position_mm'])} | {num(r['final_tool_axis_deg'])} | {num(r['final_speed_L1_rad_s'])} | {r['actual_sqp_sum']}/{r['feedback_solves']} |")
    base=info[0]['protocol'];cfg=base['solver_params']
    lines+=['','## Fixed settings and orientation scale','',
        'All arms use the same M14 B1/N32 configuration: modeA/controller_dt, .03s knots and feedback, .003s RK4 plant and8s timeout; .93s prediction. Computation advances no plant time. Initial rest-state/zero-control seeds, first recorded t0 solve, production warm shift/reanchor, once-per-episode dual/rho reset and retained adaptive rho are unchanged. Final approach is off.','',
        '| Arm | Running orientation | Terminal orientation | Position running / terminal | u cost |','|---|---:|---:|---:|---:|']
    for item in info:
        c=item['protocol']['solver_params'];lines.append(f"| {item['arm']} | {c['ee_orient_cost']} | {c['ee_orient_N_cost']} | {c['q_cost']}/{c['N_cost']} | {c['u_cost']} |")
    lines+=['','Original orientation/position weight ratios are.025, making a1rad wrapped-RPY error norm equivalent in quadratic cost to about158mm Cartesian error (2rad≈316mm). LOW reduces these equivalents to50/100mm; OFF removes orientation-goal cost and allows positional IK redundancy. Wrapped RPY error norms are not physical rotation angles. These are scale comparisons, not causal proof or success guarantees.','',
        'LOW_TORQUE changes only LOW ordinary torque regularization from.0001 to.00001; OFF_TORQUE applies the same lower ordinary torque coefficient with orientation0/0. Each canonical result and the narrow OFF_TORQUE diagnostic exception remain separate. The original OFF failure is unchanged. The ctrl_lim_cost=.02 barrier remains. At zero torque, barrier curvature is2*ctrl_lim_cost/U_cost^2. Total curvature drops about3.25x for joints1/2 and1.98x for3..7, rather than10x, because the barrier is retained. Nonzero gravity torque has larger barrier curvature; this arithmetic is not a motion prediction.','',
        f"Other solver settings: mu={cfg['mu']}, qd_cost={cfg['qd_cost']}, q/v/u limit weights={cfg['q_lim_cost']}/{cfg['vel_lim_cost']}/{cfg['ctrl_lim_cost']}, rho={cfg['rho']}, SQPcap={cfg['max_sqp_iters']}, PCGcap={cfg['max_pcg_iters']}, pcg_tol={cfg['pcg_tol']}, solve_ratio={cfg['solve_ratio']}. "
        'Natural zero-PCG outer stopping remains active as a practical heuristic, not a stationarity certificate. Existing kkt_tol=.001 is recorded and is not an acceptance gate. Interior cost bounds remain q inset.20rad, speed80%, torque85%; physical limits remain unchanged. Starts outside soft intervals are retained without clipping.','',
        '## Provenance and reproduction','', '| Arm | Protocol SHA256 | Native SHA256 |','|---|---|---|']
    for item in info:lines.append(f"| {item['arm']} | {item['protocol_sha256']} | {base['native_module_sha256']} |")
    lines+=['','Numerical core85ef2ea5cf5b96726be6f4b5441f4a7c0ca041b4 and the immutable q020 native module are reused. metadata.json retains full source/worker/input hashes, raw roots, exact launch commands/receipts, rescore identities and per-case metrics/history/input/work hashes. Original M14 data, failed preflights and interrupted earlier100validation remain preserved; the user-stopped validation is not completed or resumed here. No horizon run, new sampler, solver repair or new parameter selection belongs to this study.','',
        'Regenerate from compact editable data:', '', '```sh','python analyze_orientation.py --plot-only .','python report_to_pdf.py --report REPORT.md','```','',
        'This bounded development comparison is not an independent reliability study, continuous-time/collision certificate, paid-computation result or physical-robot qualification. Receding-horizon soft goals do not guarantee eventual arrival.','']
    if (output.parent/'random-reaching-off-target-20261015/REPORT.md').is_file():
        lines+=['Detailed saved-horizon cost and motion analysis of the far-off-goal examples is provided separately in [the off-target report](../random-reaching-off-target-20261015/REPORT.md). It is not a new rollout or a single-cause claim.','']
    (output/'REPORT.md').write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root',type=Path);p.add_argument('--baseline-root',type=Path,default=Path('/workspace/GATO/_expr_results/random-reaching-margins-20261015/M14-joint-speed-mid02125'))
    p.add_argument('--rescore-root',type=Path,default=Path('/workspace/GATO/_expr_results/random-reaching-orientation-baseline-rescore-20261015'))
    p.add_argument('--source',type=Path,default=HERE.parents[1]);p.add_argument('--output',type=Path);p.add_argument('--previews',type=Path)
    p.add_argument('--allow-partial',action='store_true');p.add_argument('--plot-only',type=Path)
    p.add_argument('--pending-off-torque-waiver',action='store_true',help='Label the current complete/held-data report as a review checkpoint awaiting the user answer; does not permit execution')
    args=p.parse_args()
    if args.plot_only:plots(args.plot_only.resolve());return
    if not args.raw_root or not args.output:p.error('Provide --raw-root and --output')
    raw,base,source,output=(getattr(args,k).resolve() for k in ('raw_root','baseline_root','source','output'))
    if output.is_relative_to(raw) or output.is_relative_to(base):raise ValueError('Compact output must be outside raw data')
    protocol=read(base/'pilot/protocol.json');baseprov=read(base/'pilot/provenance.json')
    if sha(base/'pilot/protocol.json')!=BASELINE_SHA:raise ValueError('Baseline M14 protocol differs')
    expected=protocol['development_task_ids'];rescore=read(args.rescore_root/'M14_position_speed_rescore.json')
    rescored={int(r['pair_id']):r for r in csv_read(args.rescore_root/'M14_position_speed_rescore.csv')}
    if list(rescored)!=expected or rescore['protocol_sha256']!=BASELINE_SHA:raise ValueError('Baseline rescore assignment differs')
    for pair,files in rescore['per_case_input_sha256'].items():
        for name,digest in files.items():
            if sha(base/'pilot'/f'pair_{int(pair):05d}'/name)!=digest:raise ValueError('Baseline rescore source changed')
    if sha(source/MODEL)!=baseprov['source_sha256'][MODEL]:raise ValueError('Physical model differs')
    joints=[j for j in ET.parse(source/MODEL).getroot().findall('joint') if j.get('type')!='fixed']
    limits={k:np.array([float(j.find('limit').get(k)) for j in joints]) for k in ('lower','upper')}
    info=[];counts=[];rows=[];traces=[];thresholds=[];paired={}
    for arm in ARMS:
        path=base/'pilot/protocol.json' if arm=='M14' else source/f'experiments/random_reaching_baseline/protocols/{arm}-orientation.json'
        measured=read(path)
        expected_cfg=dict(protocol['solver_params'])
        if arm!='M14':
            expected_cfg.update(ee_orient_cost=0. if arm.startswith('OFF') else .01,ee_orient_N_cost=0. if arm.startswith('OFF') else .2)
            if arm.endswith('_TORQUE'):expected_cfg['u_cost']=.00001
            if measured.get('orientation_required_for_arrival') is not False:raise ValueError('Orientation became arrival gate')
        if measured['solver_params']!=expected_cfg:raise ValueError('Unapproved cost change: '+arm)
        for key in ('source_base','development_task_ids','task_ids','task_ids_sha256','input_sha256','limit_penalty','initialization','timing_mode','offline_timing','N','knot_dt_s','feedback_s','sim_dt_s','timeout_s','position_tolerance_m','velocity_l1_tolerance_rad_s','final_approach','reset_rho_each_solve','native_module_sha256'):
            if measured[key]!=protocol[key]:raise ValueError('Unchanged protocol field differs: '+arm+'/'+key)
        result=collect(arm,base if arm=='M14' else raw/arm,measured,baseprov,expected,paired,limits,rescored if arm=='M14' else None)
        result[0]['protocol_sha256']=sha(path);info.append(result[0]);counts.append(result[1]);rows+=result[2];traces+=result[3];thresholds+=result[4]
    final=all(item['state'] in ('complete','held_preflight_failure','held_pilot_execution') for item in info)
    if args.pending_off_torque_waiver and any(c['recorded'] for c in counts if c['arm'] in ('OFF','OFF_TORQUE')):
        raise ValueError('Pending-waiver checkpoint must contain no OFF pilot outcomes')
    if args.pending_off_torque_waiver and any(item['diagnostic_user_waiver'] for item in info):
        raise ValueError('A recorded approved waiver cannot be labeled pending')
    if not final and (not args.allow_partial or 'thesis_results' in output.parts):raise ValueError('Incomplete arms: temporary --allow-partial output required')
    output.mkdir(parents=True,exist_ok=True);csv_write(output/'cases.csv',rows);write(output/'counts.json',counts)
    csv_write(output/'canonical_trace.csv',traces,['arm','time_s','position_error_mm','tool_axis_deg','speed_L1_rad_s'])
    csv_write(output/'arrival_sensitivity.csv',thresholds)
    for name in ('M14_position_speed_rescore.json','M14_position_speed_rescore.csv'):shutil.copy2(args.rescore_root/name,output/name)
    for item in info:
        if item['diagnostic_user_waiver']:
            shutil.copy2(item['diagnostic_user_waiver']['path'],output/'diagnostic_user_waiver.json')
    commands={str(path):dict(sha256=sha(path),text=path.read_text()) for path in raw.rglob('*receipt*.json')}
    for path in (raw/'preparation').glob('*launch_commands.*'):commands[str(path)]=dict(sha256=sha(path),text=path.read_text())
    write(output/'metadata.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),final=final,raw_root=str(raw),baseline_root=str(base),
        scope_status='review_checkpoint_pending_user_answer' if args.pending_off_torque_waiver else 'complete_or_held_records' if final else 'incomplete_preview',
        off_torque_waiver_question_pending=args.pending_off_torque_waiver,
        rescore_root=str(args.rescore_root.resolve()),rescore_json_sha256=sha(args.rescore_root/'M14_position_speed_rescore.json'),
        rescore_csv_sha256=sha(args.rescore_root/'M14_position_speed_rescore.csv'),
        analysis_command=shlex.join([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]]),analysis_sha256=sha(__file__),cohorts=info,commands=commands))
    plots(output);report(output,info,counts,rows,final,args.pending_off_torque_waiver)
    shutil.copy2(__file__,output/'analyze_orientation.py');helper=source/'experiments/random_reaching_baseline/report_to_pdf.py';shutil.copy2(helper,output/helper.name)
    spec=importlib.util.spec_from_file_location('report_to_pdf',helper);renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
    pages=renderer.render(output/'REPORT.md',args.previews)
    print(json.dumps(dict(final=final,counts=counts,pdf_pages=pages,output=str(output))))


if __name__=='__main__':main()
