#!/usr/bin/env python3
"""CPU saved-record comparison of OFF_TORQUE SQP caps1/5/100/1000."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import shlex
import shutil
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import analyze_frozen_validation as shared
import plot_reaching_histogram as histogram

read,sha,write,csv_write,csv_read,num=shared.read,shared.sha,shared.write,shared.csv_write,shared.csv_read,shared.num
CAPS=(1,5,100,1000)

def input_hashes(path):
    import hashlib
    with np.load(path,allow_pickle=False) as z:return {k:hashlib.sha256(np.ascontiguousarray(z[k]).tobytes()).hexdigest() for k in ('initial_state','source_goal_q','target_xyz','target_rotation')}

def plots(output):
    rows=csv_read(output/'cases.csv');colors={1:'#bb4f3f',5:'#d99535',100:'#368e77',1000:'#277da8'}
    fig,axes=plt.subplots(1,2,figsize=(11.6,4.2))
    for cap in CAPS:
        group=[r for r in rows if int(r['SQP_cap'])==cap and r['complete']=='1']
        if not group:continue
        times=sorted(float(r['arrival_seconds']) for r in group if r['task_success']=='1');axes[0].step([0,*times,8],[0,*np.arange(1,len(times)+1)/12,len(times)/12],where='post',label=f'cap{cap}',color=colors[cap])
        axes[1].hist(times,bins=np.arange(0,9,.5),histtype='step',label=f'cap{cap}',color=colors[cap])
    axes[0].set(xlim=(0,8),ylim=(0,1),xlabel='Simulated time (s)',ylabel='Position+speed arrivals / assigned12');axes[1].set(xlim=(0,8),xlabel='Arrival time (s), arrivals only',ylabel='Number of arrivals')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(output/'arrivals.png',dpi=180);fig.savefig(output/'arrivals.pdf');plt.close(fig)
    histogram.plot(output,group_key='SQP_cap',assigned=12)
    fig,axes=plt.subplots(1,2,figsize=(11.6,4.2))
    for j,cap in enumerate(CAPS):
        group=[r for r in rows if int(r['SQP_cap'])==cap and r['complete']=='1']
        for r in group:
            if r['final_position_mm']!='':axes[0].scatter(int(r['order'])+(j-1.5)*.12,float(r['final_position_mm']),s=25,color=colors[cap],marker='o' if r['completed_episode_physical_pass']=='1' else 'x')
            if r['episode_wall_s']!='' and r['actual_PCG_sum']!='':axes[1].scatter(float(r['episode_wall_s']),float(r['actual_PCG_sum']),s=25,color=colors[cap],marker='o' if r['completed_episode_physical_pass']=='1' else 'x')
        axes[0].scatter([],[],color=colors[cap],label=f'cap{cap}');axes[1].scatter([],[],color=colors[cap],label=f'cap{cap}')
    axes[0].axhline(50,color='black',ls=':',lw=.8);axes[0].set(xlabel='Frozen development task order',ylabel='Endpoint position error (mm)');axes[1].set(xlabel='Episode wall time (s)',ylabel='Actual PCG iterations');axes[1].set_xscale('symlog',linthresh=1);axes[1].set_yscale('symlog',linthresh=1)
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(output/'endpoints_work.png',dpi=180);fig.savefig(output/'endpoints_work.pdf');plt.close(fig)


def report(output,rows,counts,metadata,complete):
    lines=['# Frozen development12: SQP-cap comparison','', 'Complete paired development study.' if complete else 'INCOMPLETE PREVIEW; missing cases have no inferred outcome.','',
        'B1 caps1/5/100 are new one-pass closed-loop episodes on the exact frozen development12; cap1000 reuses the completed OFF_TORQUE baseline. The only changed objective/runtime setting is the maximum outer SQP allowance. Natural zero-PCG stopping remains enabled, so a cap is not a required iteration count. The separate fresh100 validation is neither merged into these rates nor used to choose these caps.','',
        '| SQP cap | Recorded /12 | Arrival /12 | Physical /12 | Both /12 | Timeouts | Errors |','|---:|---:|---:|---:|---:|---:|---:|']
    for c in counts:lines.append(f"| {c['SQP_cap']} | {c['recorded']} | {c['rough_arrivals'] if c['recorded'] else 'unrecorded'} | {c['physical_passes'] if c['recorded'] else 'unrecorded'} | {c['both_passes'] if c['recorded'] else 'unrecorded'} | {c['timeouts']} | {c['execution_failures']} |")
    lines+=['','Primary arrival is position <=50mm plus speed L1 <=1rad/s at30ms feedback boundaries; orientation is off and descriptive only. Physical passes are complete error-free finite episodes respecting raw URDF q excess <=1e-6rad and speed/torque ratios <=1+1e-6. Raw prefix flags remain separate; error/nonfinite full extrema are null. Every assigned task remains in each cap denominator.','',
        '![Every curve divides by assigned12. Timeouts are right-censored at8s; errors are failed episodes and remain in the denominator. Histograms are conditional on observed arrivals. Missing cases contribute no inferred event/censoring time; partial curves are observed lower bounds.](arrivals.png)','',
        '![Each panel retains all12 assigned tasks: recorded arrivals use their actual times; completed non-arrivals use the separate8s/no-arrival display bin, with timeouts and execution errors distinguished. Arrivals use the position+speed criterion regardless of physical compliance, which remains separate. The display value is not an arrival or actual error runtime and does not alter original success labels or means. Missing cases receive no inferred bin.](reaching_histogram.png)','',
        '## Actual work and timing','', '| Cap | Feedback solves | Actual SQP | PCG iterations | Zero-PCG exits | Episode / solve / native s |','|---:|---:|---:|---:|---:|---:|']
    for cap in CAPS:
        group=[r for r in rows if r['SQP_cap']==cap and r['complete']]
        total=lambda key:sum(r[key] for r in group if r[key] is not None)
        lines.append(f"| {cap} | {total('feedback_solves')} | {total('actual_sqp_sum')} | {total('actual_PCG_sum')} | {total('zero_pcg_exit_solves')} | {num(total('episode_wall_s'))}/{num(total('solve_wall_s'))}/{num(total('native_solve_s'))} |")
    lines+=['','These are measured whole closed-loop episodes. Arrival timing changes the number of feedback solves; different caps therefore do not imply equal realized work. Episode wall time includes recording/plant/host overhead. ModeA/controller_dt leaves computation unpaid in plant time. Zero-PCG outer exits are practical stopping heuristics, not mathematical stationarity certificates; current KKT values do not gate acceptance or extend runs.','',
        '![Every finite recorded endpoint and measured work/timing point is retained. Circles denote full-episode physical passes, crosses physical failures. The50mm line alone does not include the speed arrival criterion. Selected caps are not a population comparison, and this plot does not isolate hardware throughput.](endpoints_work.png)','',
        '## Paired cases','', '| ID | Cap | Outcome | Arrival / physical | Arrival s | End mm | q excess rad | Speed / torque ratio |','|---|---:|---|---|---:|---:|---:|---|']
    ids=metadata['1000']['protocol']['development_task_ids']
    for pair in ids:
        for r in (r for r in rows if r['pair_id']==pair):lines.append(f"| {pair} | {r['SQP_cap']} | {r['status']} | {r['task_success']}/{r['completed_episode_physical_pass']} | {num(r['arrival_seconds'])} | {num(r['final_position_mm'])} | {num(r['maximum_q_excess_rad'])} | {num(r['maximum_speed_ratio'])}/{num(r['maximum_torque_ratio'])} |")
    lines+=['','cases.csv preserves all48 planned rows; raw analysis/solves.csv retains every recorded per-call SQP/PCG/timing/stop-reason and native merit. The case table retains, each failure ID and first physical-violation joint/time, errors, observed duration and last-second progress. It also retains actual SQP/PCG/native/host timing, goal/start witness joint clearances and displacement/necessary speed-time descriptors. These geometric descriptors are not filters or dynamic-feasibility proofs.','',
        'Endpoint position-only terms are .5*4*error² running and .5*80*error² terminal at the actual final position. They are not total horizon costs. Last saved native merit includes its native conventions and is descriptive; neither apparent stalling nor a low merit proves a local/global optimum. Soft-goal timeout does not prove that more time would reach.','',
        '## Frozen protocol and failed canonical context','',
        'The existing OFF_TORQUE canonical remains an8s timeout at55.7404mm, physical pass, original preflight false. Its unchanged recorded result is reused openly as diagnostic context, not as a passed qualification for any cap. The explicit iteration-study request authorizes the diagnostic comparison. Shared setup/import/module/controller-constructor failures hold the affected run; individual case/execution/nonfinite failures are recorded and independent cases continue. No retries, filtering, physical veto, criterion relaxation, solver repair or forced continuation is added.','',
        'All arms use the same numerical85ef2ea source and immutable q020 native module, B1/N32, .03s knots/feedback, .003s RK4, .93s prediction and8s episodes. Orientation0/0, position4/terminal80, qd.05, u1e-5, q/v/u penalties.02125/1.275/.02, mu10, rho.01; C2 cost bounds q inset.20rad and speed/effort fractions.80/.85. PCG120, all tolerances and solve_ratio1 remain unchanged. Cold rest-state/zero-control initialization, first applied/recorded t0 solve, warm shift/reanchor and once-per-episode dual/rho reset then retained state remain. Final approach is off.','',
        '| Cap | Protocol SHA256 | Module SHA256 |','|---:|---|---|']
    for cap in CAPS:
        item=metadata[str(cap)];lines.append(f"| {cap} | {item['protocol_sha256']} | {item['provenance']['module_sha256']} |")
    lines+=['','metadata.json retains raw roots, reference freeze/failed canonical, preparation/root launch receipts, full protocols/source/module/input and case identities, plus the exact analysis command. Inputs are checked against original pool pairs and paired saved B1 q/goal/target arrays. Large histories, horizons and archives remain outside thesis_results. Compact CSVs and source scripts support adjustments without native calls.','',
        '```sh','python analyze_sqp_caps.py --plot-only .','python report_to_pdf.py --report REPORT.md','```','',
        'This is a paired twelve-task development contrast, not independent population validation, a reliability guarantee, continuous-time safety/collision certificate, paid-computation experiment or robot deployment. Arrival ends recording, so post-arrival settling is unobserved.','']
    if complete:
        headline='; '.join(f"cap{c['SQP_cap']}: {c['rough_arrivals']}/12 arrivals, {c['physical_passes']}/12 physical, {c['both_passes']}/12 both, {c['execution_failures']} errors" for c in counts)+'.'
        lines[4:4]=[headline,'','Cap100 matches the reused cap1000 arrival/physical counts on these development tasks. Cap5 retains nine rough arrivals but two are physically invalid; cap1 has nine individual nonfinite-plant execution failures. Matching aggregate counts do not establish identical trajectories, convergence or population reliability. Fresh100 remains at the previously frozen cap1000; these results do not retune or select its settings.','']
        errors=[str(r['pair_id']) for r in rows if r['SQP_cap']==1 and r['execution_failure']]
        where=lines.index('## Actual work and timing')
        detail=['Cap1 execution-failure IDs: '+', '.join(errors)+'. These outcomes remain execution failures, not timeouts or successful finite-prefix episodes. Their full-history extrema are null; raw prefix/category flags are preserved separately.','','Cap5 finite physical failures are7132 (maximum speed ratio1.04479) and2342 (joint excess.0689909rad, speed ratio1.58705); both achieved rough arrival. Thus cap5 has7/12 arrivals that also pass all physical checks. All12 starting observations are outside the position+speed arrival gate, with no filtering.','']
        lines[where:where]=detail
    text='\n'.join(lines)
    for a,b in {'Cap100':'Cap 100','cap1000':'cap 1000','Cap5':'Cap 5','cap5':'cap 5','Cap1':'Cap 1','cap1':'cap 1','Fresh100':'Fresh 100','are7132':'are 7132','and2342':'and 2342','ratio1.04479':'ratio 1.04479','excess.0689909rad':'excess .0689909 rad','ratio1.58705':'ratio 1.58705','has7/12':'has 7/12','All12':'All 12','development12':'development 12','caps1/5/100':'caps 1/5/100','cap1000':'cap 1000','fresh100':'fresh 100','assigned12':'assigned 12','<=50mm':'<= 50 mm','<=1rad/s':'<= 1 rad/s','at30ms':'at 30 ms','<=1e-6rad':'<= 1e-6 rad','<=1+1e-6':'<= 1+1e-6','at8s':'at 8 s','The50mm':'The 50 mm','all48':'all 48','an8s':'an 8 s','at55.7404mm':'at 55.7404 mm','numerical85ef2ea':'numerical 85ef2ea','.03s':'.03 s','.003s':'.003 s','.93s':'.93 s','and8s':'and 8 s','Orientation0/0':'Orientation 0/0','position4/terminal80':'position 4/terminal 80','qd.05':'qd .05','u1e-5':'u 1e-5','penalties.02125':'penalties .02125','mu10':'mu 10','rho.01':'rho .01','inset.20rad':'inset .20 rad','fractions.80/.85':'fractions .80/.85','PCG120':'PCG 120','solve_ratio1':'solve_ratio 1','ModeA':'Mode A'}.items():text=text.replace(a,b)
    (output/'REPORT.md').write_text(text)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('raw-root','source','output','previews'):p.add_argument('--'+key,type=Path)
    p.add_argument('--baseline-root',type=Path,default=Path('/workspace/GATO/_expr_results/random-reaching-orientation-20261015/OFF_TORQUE'));p.add_argument('--allow-partial',action='store_true');p.add_argument('--plot-only',type=Path);args=p.parse_args()
    if args.plot_only:plots(args.plot_only.resolve());return
    if not args.raw_root or not args.source or not args.output:p.error('Provide --raw-root --source --output')
    raw,source,output,baseline=(getattr(args,k).resolve() for k in ('raw_root','source','output','baseline_root'));prep=raw/'preparation';rows=[];counts=[];metadata={};paired={};solves=[]
    baseline_proto=read(baseline/'pilot/protocol.json');baseline_prov=read(baseline/'pilot/provenance.json');ids=baseline_proto['development_task_ids']
    with np.load(Path(baseline_prov['pool'])/'pairs.npz',allow_pickle=False) as z:assignment=dict(task_ids=ids,records=[dict(order=i+1,pair_id=pair,start_index=int(z['start_index'][pair]),goal_index=int(z['goal_index'][pair])) for i,pair in enumerate(ids)])
    for pair in ids:paired[pair]=input_hashes(baseline/'pilot'/f'pair_{pair:05d}'/'inputs.npz')
    for cap in CAPS:
        if cap==1000:
            root=baseline;protocol=baseline_proto;provenance=baseline_prov
        else:
            root=raw/f'SQP{cap}';protocol,freeze,canonical,provenance,limits=shared.verify_run(root,prep,source,prep/f'SQP{cap}_protocol.json','development_sqp_cap')
            if any(provenance[k]!=baseline_prov[k] for k in ('source_sha256','module_sha256','inputs_sha256')):raise ValueError('Paired source/module/input differs')
        cases,c,hashes=shared.collect_run(root,assignment,protocol,provenance,limits)
        for r in cases:
            r['SQP_cap']=cap;ip=root/'pilot'/f"pair_{r['pair_id']:05d}"/'inputs.npz'
            if ip.exists() and input_hashes(ip)!=paired[r['pair_id']]:raise ValueError('Paired initial state/goal/target arrays differ')
        solves+=shared.solve_rows(root,dict(protocol,run_task_ids=ids));rows+=cases;c['SQP_cap']=cap;counts.append(c);metadata[str(cap)]=dict(raw_root=str(root),protocol=protocol,protocol_sha256=sha(root/'pilot/protocol.json') if (root/'pilot/protocol.json').exists() else sha(prep/f'SQP{cap}_protocol.json'),provenance=provenance,cases_sha256=hashes)
    complete=all(c['complete'] for c in counts)
    if not complete and (not args.allow_partial or 'thesis_results' in output.parts):raise ValueError('Incomplete paired12 study requires temporary --allow-partial output')
    if output.is_relative_to(raw) or output.is_relative_to(baseline):raise ValueError('Do not overwrite raw storage')
    output.mkdir(parents=True,exist_ok=True);(raw/'analysis').mkdir(parents=True,exist_ok=True);csv_write(output/'cases.csv',rows);csv_write(output/'failure_cases.csv',[r for r in rows if r['complete'] and (not r['task_success'] or not r['completed_episode_physical_pass'] or r['execution_failure'])],list(rows[0]));write(output/'counts.json',counts);csv_write(raw/'analysis'/'solves.csv',solves)
    if (raw/'backup_receipt.json').exists():shutil.copy2(raw/'backup_receipt.json',output/'raw_backup_receipt.json')
    receipts={str(path):dict(sha256=sha(path),data=read(path)) for path in [*prep.glob('*.json'),*raw.glob('*.json'),*raw.glob('*/*receipt*.json')]}
    write(output/'metadata.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),source=str(source),cohorts=metadata,reference_freeze=read(baseline/'research_baseline_freeze.json'),reference_canonical=read(baseline/'preflight/preflight/metrics.json'),receipts=receipts,paired_inputs=paired,analysis_command=shlex.join([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]]),analysis_sha256=sha(__file__),per_call_csv_path=str(raw/'analysis'/'solves.csv'),per_call_csv_sha256=sha(raw/'analysis'/'solves.csv')))
    plots(output);report(output,rows,counts,metadata,complete);shutil.copy2(Path(__file__).resolve(),output/'analyze_sqp_caps.py');pages=shared.render(output,source,args.previews)
    print(dict(counts=counts,pdf_pages=pages,output=str(output)))

if __name__=='__main__':main()
