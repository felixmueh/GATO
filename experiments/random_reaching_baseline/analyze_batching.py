#!/usr/bin/env python3
"""CPU-only four-task saved-record batching/SQP-allowance comparison."""
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
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
IDS=[7733, 8054, 2497,9651]
ARMS=('B1','B16','B1_EXTRA')
BATCH={'B1':1,'B16':16,'B1_EXTRA':1}
CAP={'B1':1000,'B16':1000,'B1_EXTRA':16000}
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def val(v):return float(v) if v is not None and np.isfinite(v) else None
def num(v):return format(float(v),'.6g') if val(v) is not None else 'null'
def mm(v):return val(v)*1000 if val(v) is not None else None
def write(p,obj):Path(p).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def csv_read(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def csv_write(p,rows):
    with Path(p).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def input_hashes(p):
    with np.load(p,allow_pickle=False) as z:
        return {k:hashlib.sha256(np.ascontiguousarray(z[k]).tobytes()).hexdigest() for k in ('initial_state','source_goal_q','target_xyz','target_rotation')}
def record(p):return dict(path=str(p),sha256=sha(p),data=read(p))


def collect(arm,root,source,paired):
    run=root/'pilot';protofile=run/'protocol.json'
    if not protofile.exists():protofile=source/'experiments/random_reaching_baseline/protocols'/f'batching-{arm}.json'
    proto=read(protofile);prov=read(run/'provenance.json') if (run/'provenance.json').exists() else None
    if arm!='B1' and (proto['pilot_task_ids']!=IDS or proto['batch']!=BATCH[arm]):raise ValueError('Pilot assignment/batch differs')
    if prov and (prov['protocol']!=proto or (arm!='B1' and prov['pairs']!=IDS)):raise ValueError('Recorded protocol/assignment differs')
    summary=read(run/'summary.json') if (run/'summary.json').exists() else []
    if arm!='B1' and [m['pair_id'] for m in summary]!=IDS[:len(summary)]:raise ValueError('New summary is not an ordered four-task prefix')
    indexed={m['pair_id']:m for m in summary};cases=[];work=[];identities={}
    holds=[record(p) for p in root.glob('*held*.json')]
    holds += [record(p) for p in (root/'setup_failure.json',root/'preflight/setup_failure.json',run/'setup_failure.json') if p.exists()]
    prepath=root/'preflight/preflight/metrics.json'
    if prepath.exists() and read(prepath).get('shared_setup_failure'):holds.append(record(prepath))
    if (root/'preflight/protocol.json').exists() and read(root/'preflight/protocol.json')!=proto:raise ValueError('Canonical/pilot protocol differs')
    if (root/'preflight/provenance.json').exists():
        preprov=read(root/'preflight/provenance.json')
        if preprov['module_sha256']!=proto['native_module_sha256']:raise ValueError('Canonical module differs')
        if prov and any(preprov[k]!=prov[k] for k in ('source_sha256','module_sha256','wrapper_sha256')):raise ValueError('Canonical/pilot source differs')
    for order,pair in enumerate(IDS,1):
        d=run/f'pair_{pair:05d}';mp=d/'metrics.json';m=read(mp) if mp.exists() else None
        if pair in indexed and m!=indexed[pair]:raise ValueError(f'Summary/case metrics differ {arm}/{pair}')
        accepted=pair in indexed and bool(m and m['complete']);err=bool(m and m['execution_error']);p=m['physical'] if m else None
        usable=bool(accepted and not err and p['finite']);physical=bool(usable and m['physical_pass']);arrival=bool(accepted and not err and m['task_success'])
        solves=[json.loads(line) for line in (d/'solves.jsonl').read_text().splitlines()] if (d/'solves.jsonl').exists() else []
        for item in solves:
            w=item['work'];work.append(dict(arm=arm,pair_id=pair,step=item['step'],feedback_s=item['feedback_time_s'],executed_duration_s=item['executed_duration_s'],
                selected_lane=item.get('selected_lane',0),outer_SQP=w['actual_sqp'],all_lane_SQP=w.get('all_lane_sqp_sum',w['actual_sqp']),all_lane_PCG=w['pcg_iteration_sum'],
                selected_lane_SQP=w.get('selected_lane_sqp',w['actual_sqp']),selected_lane_PCG=w.get('selected_lane_pcg_iteration_sum',w['pcg_iteration_sum']),
                zero_PCG_exit=int(w['zero_pcg_outer_exit']),stop_reason=w['stop_reason'],nominal_outer_cap=CAP[arm],batch=BATCH[arm],
                native_solve_s=val(item['native_solve_us']/1e6),solve_wall_s=val(item['solve_wall_s']),prefix_assessment_wall_s=val(item.get('prefix_assessment_wall_s'))))
        own=[r for r in work if r['arm']==arm and r['pair_id']==pair]
        if accepted:
            for key,got in [('actual_sqp_sum',sum(r['outer_SQP'] for r in own)),('actual_pcg_iteration_sum',sum(r['all_lane_PCG'] for r in own)),('feedback_solves',len(own))]:
                if m[key]!=got:raise ValueError(f'Work/metrics differ {arm}/{pair}/{key}')
            if arm!='B1' and m['all_lane_sqp_sum']!=sum(r['all_lane_SQP'] for r in own):raise ValueError('All-lane SQP mismatch')
        row=dict(arm=arm,order=order,pair_id=pair,recorded=int(accepted),status=m['outcome'] if accepted else 'held_shared_setup' if holds else 'in_progress' if d.exists() else 'not_recorded',
            arrival=int(arrival) if accepted else None,arrival_s=val(m['reached_seconds']) if arrival else None,physical_pass=int(physical) if accepted else None,
            raw_sampled_physical_pass=int(m['physical_pass']) if m else None,both=int(arrival and physical) if accepted else None,
            execution_error=m['execution_error'] if m else None,execution_failure=int(err) if accepted else None,interrupted=int(bool(m.get('interrupted'))) if m else None,
            batch=BATCH[arm],outer_cap=CAP[arm],nominal_lane_cap_per_solve=BATCH[arm]*CAP[arm],observed_s=val(m['simulated_seconds']) if m else None,
            initial_position_mm=mm(m['initial']['position_error_m']) if m else None,endpoint_position_mm=mm(m['final']['position_error_m']) if m else None,
            endpoint_axis_deg=val(m['final']['tool_axis_error_deg']) if m else None,endpoint_speed_L1=val(m['final']['velocity_l1_rad_s']) if m else None,
            q_excess_rad=val(p['joint']['maximum_excess']) if usable else None,speed_ratio=val(p['maximum_velocity_ratio']) if usable else None,torque_ratio=val(p['maximum_effort_ratio']) if usable else None,
            feedback_solves=m['feedback_solves'] if m else None,outer_SQP=sum(r['outer_SQP'] for r in own) if m else None,all_lane_SQP=sum(r['all_lane_SQP'] for r in own) if m else None,
            all_lane_PCG=m['actual_pcg_iteration_sum'] if m else None,selected_lane_PCG=sum(r['selected_lane_PCG'] or 0 for r in own) if m else None,
            zero_PCG_exits=m['zero_pcg_exit_solves'] if m else None,selected_nonzero_lane_solves=m.get('selected_nonzero_lane_solves') if m else None,
            episode_wall_s=val(m['episode_wall_s']) if m else None,solve_wall_s=val(m['solve_wall_s']) if m else None,native_solve_s=val(m['native_solve_us']/1e6) if m else None,
            proposal_generation_wall_s=val(m.get('proposal_generation_wall_s')) if m else None,prefix_assessment_wall_s=sum(r['prefix_assessment_wall_s'] for r in own) if own and all(r['prefix_assessment_wall_s'] is not None for r in own) else None)
        cases.append(row)
        ip=d/'inputs.npz'
        if ip.exists():
            measured=input_hashes(ip)
            if arm=='B1':paired[pair]=measured
            elif measured!=paired[pair]:raise ValueError('Paired q/goal/state input changed')
        identities[str(pair)]={f.name:sha(f) for f in (mp,ip,d/'executed.npz',d/'solves.jsonl') if f.exists()}
    done=[r for r in cases if r['recorded']]
    count=dict(arm=arm,assigned=4,recorded=len(done),status='complete' if len(done)==4 else 'held_shared_setup' if holds or any(m.get('shared_setup_failure') for m in summary) else 'partial' if run.exists() else 'not_recorded',
        arrivals=sum(r['arrival'] for r in done),physical=sum(r['physical_pass'] for r in done),both=sum(r['both'] for r in done),timeouts=sum(r['status']=='timeout' for r in done),errors=sum(r['execution_failure'] for r in done))
    pre=root/'preflight/preflight/metrics.json'
    meta=dict(root=str(root),protocol=proto,protocol_sha256=sha(protofile),provenance=prov,canonical=record(pre) if pre.exists() else None,summary_sha256=sha(run/'summary.json') if summary else None,records=identities,held_receipts=holds)
    return cases,work,count,meta


def plots(output):
    rows=csv_read(output/'cases.csv');colors={'B1':'#777777','B16':'#147fa3','B1_EXTRA':'#d66b35'}
    fig,axes=plt.subplots(1,3,figsize=(13,4.1))
    for j,arm in enumerate(ARMS):
        data=[r for r in rows if r['arm']==arm and r['recorded']=='1']
        if not data:continue
        times=sorted(float(r['arrival_s']) for r in data if r['arrival']=='1')
        axes[0].step([0,*times,8],[0,*np.arange(1,len(times)+1)/4,len(times)/4],where='post',label=arm,color=colors[arm])
        for r in data:
            if r['endpoint_position_mm']:
                axes[1].scatter(int(r['order'])+(j-1)*.18,float(r['endpoint_position_mm']),color=colors[arm],marker='o' if r['physical_pass']=='1' else 'x',s=38)
            if r['episode_wall_s'] and r['all_lane_PCG']:
                axes[2].scatter(float(r['episode_wall_s']),float(r['all_lane_PCG']),color=colors[arm],marker='o' if r['physical_pass']=='1' else 'x',s=38)
                axes[2].annotate(r['pair_id'],(float(r['episode_wall_s']),float(r['all_lane_PCG'])),fontsize=7,xytext=(2,2),textcoords='offset points')
        axes[1].scatter([],[],color=colors[arm],label=arm);axes[2].scatter([],[],color=colors[arm],label=arm)
    axes[0].set(xlabel='Simulated time (s)',ylabel='Arrivals / selected 4',xlim=(0,8),ylim=(0,1));axes[1].axhline(50,color='black',ls=':',lw=.8)
    axes[1].set(xticks=range(1,5),xticklabels=[str(p) for p in IDS],xlabel='Selected task ID',ylabel='Endpoint position error (mm)')
    axes[2].set(xlabel='Episode wall time (s)',ylabel='All-lane PCG iterations');axes[2].set_xscale('symlog',linthresh=1);axes[2].set_yscale('symlog',linthresh=1)
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(output/'comparison.png',dpi=180);fig.savefig(output/'comparison.pdf');plt.close(fig)


def report(output,counts,rows,metadata,complete):
    lines=['# Four-task batching and SQP-allowance comparison','', 'Complete selected-case diagnostic.' if complete else 'INCOMPLETE PREVIEW: unrecorded cases have no outcome; arrival curves are observed lower bounds.','',
        'The four tasks were selected before these new measurements: three saved OFF_TORQUE timeouts (7733, 8054, 2497) and one known arrival (9651). This is a selected-case comparison, not a population success-rate estimate. The saved B1 cap 1000 episodes are reused unchanged. No twelve-task or historical 100 denominator is mixed into this study.','',
        '| Arm | Recorded /4 | Status | Arrival /4 | Physical /4 | Both /4 | Timeouts | Errors |','|---|---:|---|---:|---:|---:|---:|---:|']
    for c in counts:lines.append(f"| {c['arm']} | {c['recorded']} | {c['status']} | {c['arrivals'] if c['recorded'] else 'unrecorded'} | {c['physical'] if c['recorded'] else 'unrecorded'} | {c['both'] if c['recorded'] else 'unrecorded'} | {c['timeouts']} | {c['errors']} |")
    lines+=['','Primary arrival: position <= 50 mm and speed L1 <= 1 rad/s at 30 ms feedback boundaries, without orientation. Physical checks use all saved 3 ms execution samples and held controls: raw q excess <= 1e-6 rad and speed/effort ratios <= 1+1e-6. A full-episode physical pass requires completion, finiteness and no execution error; raw prefix flags remain separate. Errors and timeouts stay in the assigned-four denominator. Null extrema are not zero.','',
        '![Arrival curves use the fixed selected-four denominator; timeouts are censored at 8 s. Endpoint and timing panels retain all recorded cases. Circles indicate sampled physical passes, crosses physical failures. The dotted 50 mm line alone does not encode the speed arrival gate. Work and wall time are measured, not equalized.](comparison.png)','',
        '## Paired outcomes','', '| ID | Arm | Outcome | Arrival / physical | Arrival s | Endpoint mm | q rad | Speed / torque ratio |','|---|---|---|---|---:|---:|---:|---|']
    for pair in IDS:
        for r in (r for r in rows if r['pair_id']==pair):lines.append(f"| {pair} | {r['arm']} | {r['status']} | {r['arrival']}/{r['physical_pass']} | {num(r['arrival_s'])} | {num(r['endpoint_position_mm'])} | {num(r['q_excess_rad'])} | {num(r['speed_ratio'])}/{num(r['torque_ratio'])} |")
    lines+=['','Only an observed physically compliant arrival on a previously non-arriving task is a candidate batch rescue. Such a result would support this selected-case behavior, not prove multimodality, a local-minimum escape, a global optimum or a general reliability gain. More SQP allowance is also a new closed-loop rollout; it is distinct from refining one saved horizon after an episode.','',
        '## Nominal allowance and actual work','', 'B1 permits 1000 outer iterations per solve. B16 permits 1000 with 16 coupled native lanes. B1_EXTRA permits 16000 with one lane. B16 and B1_EXTRA therefore share 16000 nominal lane-iteration allowance per solve, but actual PCG/SQP work and wall time need not match. B16 uses unchanged global native stopping: a lane with zero PCG may still update while peers continue. It is not 16 independently early-stopped solvers. Natural zero-PCG exits are descriptive practical stopping indicators, not stationarity certificates; KKT values are not acceptance gates.','',
        '| ID / arm | Feedback solves | Outer / all-lane SQP | All-lane PCG | Episode / solve / native s | Seed / prefix CPU s |','|---|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['pair_id']} / {r['arm']} | {r['feedback_solves']} | {r['outer_SQP']}/{r['all_lane_SQP']} | {r['all_lane_PCG']} | {num(r['episode_wall_s'])}/{num(r['solve_wall_s'])}/{num(r['native_solve_s'])} | {num(r['proposal_generation_wall_s'])}/{num(r['prefix_assessment_wall_s'])} |")
    lines+=['','Outer SQP is the maximum per-slot SQP counter for each solve, summed over feedback. All-lane SQP sums every lane counter; PCG sums all inner iteration counters. cases.csv and solves.csv retain selected-lane work, native/host timing, stop reasons and zero-PCG exits. The original B1 worker did not separately time seed generation or prefix assessment: null means unavailable, not zero. Wall totals include different CPU analysis/proposal overhead and Python recording overhead. They are descriptive rather than isolated parallel-throughput measurements.','',
        '## Canonical outcomes and diagnostic scope','', '| Arm | Canonical outcome | Arrival | Physical | Original preflight | End mm |','|---|---|---|---|---|---:|']
    for arm,info in metadata.items():
        m=info['canonical']['data'] if info['canonical'] else None
        lines.append(f"| {arm} | {m['outcome'] if m else 'not_recorded'} | {m['task_success'] if m else None} | {m['physical_pass'] if m else None} | {m['preflight_pass'] if m else None} | {num(mm(m['final']['position_error_m'])) if m else 'null'} |")
    lines+=['','Original scientific canonical labels remain unchanged. The new explicit diagnostic launch policy records canonical first, then retains scientific task/physical negatives without relabeling; root launch receipts authorize the four-task pilots. Shared setup/import/module/constructor/GPU-access failures hold the affected arm, while independent work continues. Individual case failures are saved and remaining cases continue. There are no retries, physical candidate vetoes, threshold changes or solver repairs.','',
        '## Controller, proposals and source identity','', 'One actual plant is controlled per episode. Cold B16 lanes begin with identical actual-rest-state/zero-control seeds; the first solve is applied and recorded, without hidden warmup. Thereafter lane 0 is the shifted selected winner reanchored at the actual state; 15 fresh historical BA cosine perturbations use amplitudes .02/.05/.10 rad, analytic q/v/a and inverse-dynamics control seeds. No goal/IK injection, clipping or physical filtering is added. Candidate eligibility requires a finite complete horizon, finite native merit and finite RK4 prefix; selection minimizes (final_merit, lane). Physical limits remain diagnostic and never veto the winner. Dual/rho are reset once per episode then retained per native slot, including replaced proposals.','',
        'Mode A/controller_dt, N32, .03 s knots/feedback, .003 s RK4, .93 s prediction and 8 s timeout remain fixed. Solve latency advances no plant time. Frozen OFF_TORQUE costs: position 4/terminal 80, orientation 0/0, qd .05, ordinary torque 1e-5, q/v/u limit weights .02125/1.275/.02, mu 10, rho .01; interior bounds q inset .20 rad, speed .80 and effort .85 of raw limits. PCG 120, tolerances and solve_ratio 1 unchanged. Final approach is off. Every initial state, source goal q, target position and rotation is checked against the saved B1 record. All numerical/model/controller source hashes and the immutable q020 module must match the reference.','',
        '| Arm | Protocol SHA256 | Native SHA256 |','|---|---|---|']
    for arm,info in metadata.items():lines.append(f"| {arm} | {info['protocol_sha256']} | {info['protocol']['native_module_sha256']} |")
    lines+=['','Full protocols, exact source/module/input hashes, root freeze/launch receipts and per-case record hashes remain in metadata.json. All raw histories, candidate horizons and arrays remain outside thesis_results. The analysis command is retained verbatim there. Compact files preserve all 12 planned arm-case rows and every recorded solve, including errors and unavailable-field nulls.','',
        '```sh','python analyze_batching.py --plot-only .','python report_to_pdf.py --report REPORT.md','```','',
        'This comparison does not certify continuous-time safety, collision clearance, hardware deployment, general seed superiority or population success. Early stopping at rough arrival prevents post-arrival settling claims.','']
    (output/'REPORT.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--raw-root',type=Path);parser.add_argument('--baseline-root',type=Path,default=Path('/workspace/GATO/_expr_results/random-reaching-orientation-20261015/OFF_TORQUE'))
    parser.add_argument('--source',type=Path,default=HERE.parents[1]);parser.add_argument('--output',type=Path);parser.add_argument('--previews',type=Path);parser.add_argument('--allow-partial',action='store_true');parser.add_argument('--plot-only',type=Path);args=parser.parse_args()
    if args.plot_only:plots(args.plot_only.resolve());return
    if not args.raw_root or not args.output:parser.error('Provide --raw-root and --output')
    raw,baseline,source,output=(getattr(args,k).resolve() for k in ('raw_root','baseline_root','source','output'))
    paired={};rows=[];solves=[];counts=[];metadata={}
    for arm,root in [('B1',baseline),('B16',raw/'B16'),('B1_EXTRA',raw/'B1_EXTRA')]:
        cases,work,count,info=collect(arm,root,source,paired);rows+=cases;solves+=work;counts.append(count);metadata[arm]=info
    ref=metadata['B1'];fixed=('source_base','input_sha256','task_ids','task_ids_sha256','development_task_ids','initialization','limit_penalty','native_module_sha256','timing_mode','offline_timing','N','knot_dt_s','feedback_s','sim_dt_s','timeout_s','position_tolerance_m','velocity_l1_tolerance_rad_s','goal_dwell_s','final_approach','reset_rho_each_solve','arrival_policy','orientation_required_for_arrival')
    for arm in ARMS[1:]:
        info=metadata[arm]
        for key in fixed:
            if info['protocol'][key]!=ref['protocol'][key]:raise ValueError('Frozen protocol setting differs: '+key)
        expected=dict(ref['protocol']['solver_params']);expected['max_sqp_iters']=CAP[arm]
        if info['protocol']['solver_params']!=expected:raise ValueError('Other solver parameters differ')
        if info['provenance']:
            for key in ('source_sha256','module_sha256','inputs_sha256'):
                if info['provenance'][key]!=ref['provenance'][key]:raise ValueError('Frozen source/module/input differs: '+key)
    complete=all(c['status']=='complete' for c in counts)
    if not complete and (not args.allow_partial or 'thesis_results' in output.parts):raise ValueError('Incomplete study requires temporary --allow-partial output outside thesis_results')
    if output.is_relative_to(raw) or output.is_relative_to(baseline):raise ValueError('Do not export over raw records')
    output.mkdir(parents=True,exist_ok=True);csv_write(output/'cases.csv',rows);csv_write(output/'solves.csv',solves);write(output/'counts.json',counts)
    retained={}
    for p in [*raw.glob('*.json'),*raw.glob('*/*receipt*.json'),*raw.glob('preparation/*receipt*.json'),raw/'preparation/launch_commands.json']:
        if p.exists():retained[str(p)]=record(p)
    write(output/'metadata.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),complete=complete,task_ids=IDS,raw_root=str(raw),baseline_root=str(baseline),cohorts=metadata,paired_input_sha256=paired,retained_receipts=retained,
        analysis_command=shlex.join([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]]),analysis_sha256=sha(__file__)))
    plots(output);report(output,counts,rows,metadata,complete)
    for name in ('analyze_batching.py','report_to_pdf.py'):shutil.copy2(HERE/name,output/name)
    spec=importlib.util.spec_from_file_location('report_to_pdf',output/'report_to_pdf.py');renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer);pages=renderer.render(output/'REPORT.md',args.previews)
    print(json.dumps(dict(counts=counts,pdf_pages=pages,output=str(output))))

if __name__=='__main__':main()
