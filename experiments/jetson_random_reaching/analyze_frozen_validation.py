#!/usr/bin/env python3
"""CPU saved-record export for fresh100 frozen OFF_TORQUE validation."""
import argparse
from datetime import datetime,timezone
import importlib.util
from pathlib import Path
import shlex
import shutil
import sys
import xml.etree.ElementTree as ET
import numpy as np
import analyze_validation as prior
import plot_reaching_histogram as histogram

HERE=Path(__file__).resolve().parent
read,sha,write,csv_write,csv_read,num=prior.read,prior.sha,prior.write,prior.csv_write,prior.csv_read,prior.number

def verify_run(raw,prep,source,protocol_path,expected_role):
    protocol=read(protocol_path);prepared=read(prep/'PREPARED.json')
    if protocol['study_role']!=expected_role:raise ValueError('Wrong study role')
    freeze_path=Path(protocol['research_baseline_freeze']['path']);freeze=read(freeze_path)
    baseline=freeze['protocol'];canonical_path=Path(protocol['reused_canonical']['path']);canonical=read(canonical_path)
    if sha(freeze_path)!=protocol['research_baseline_freeze']['sha256'] or sha(canonical_path)!=protocol['reused_canonical']['sha256']:raise ValueError('Frozen baseline/canonical identity differs')
    if canonical['preflight_pass'] or not canonical['physical_pass'] or canonical['execution_error']:raise ValueError('Reused failed canonical labels differ')
    for key in prior.RUNTIME:
        if key=='solver_params':
            expected=dict(baseline[key]);expected['max_sqp_iters']=protocol[key]['max_sqp_iters']
            if protocol[key]!=expected:raise ValueError('Frozen objective/solver parameters differ')
        elif protocol[key]!=baseline[key]:raise ValueError('Frozen runtime differs: '+key)
    cap=protocol['solver_params']['max_sqp_iters']
    if (expected_role=='fresh100_validation_only' and cap!=1000) or (expected_role=='development_sqp_cap' and cap not in (1,5,100)):raise ValueError('Unexpected SQP allowance')
    arm='VALIDATION100' if expected_role=='fresh100_validation_only' else f'SQP{cap}'
    job=next(c for c in prepared['commands'] if c['arm']==arm)
    if sha(protocol_path)!=job['protocol_sha256']:raise ValueError('Prepared protocol differs')
    provenance=read(raw/'pilot/provenance.json') if (raw/'pilot/provenance.json').exists() else read(prep/f'dryrun-{arm}.json')
    if provenance['protocol']!=protocol or provenance['pairs']!=protocol['run_task_ids']:raise ValueError('Measured assignment/protocol differs')
    if provenance['wrapper_sha256']!=prepared['worker']['sha256'] or sha(source/'experiments/random_reaching_baseline/run_frozen_baseline.py')!=prepared['worker']['sha256']:raise ValueError('Frozen worker differs')
    baseline_prov=read(canonical_path.parents[1]/'provenance.json')
    for key in ('source_sha256','module_sha256','inputs_sha256'):
        if provenance[key]!=baseline_prov[key]:raise ValueError('Reference source/module/input differs: '+key)
    for name,digest in provenance['source_sha256'].items():
        if sha(source/name)!=digest:raise ValueError('Numerical source changed: '+name)
    if sha(provenance['module'])!=provenance['module_sha256']:raise ValueError('Immutable module changed')
    for name,digest in protocol['input_sha256'].items():
        if sha(Path(provenance['pool'])/name)!=digest:raise ValueError('Frozen pool changed')
    joints=[j for j in ET.parse(source/prior.MODEL).getroot().findall('joint') if j.get('type')!='fixed']
    limits={key:np.array([float(j.find('limit').get(key)) for j in joints]) for key in ('lower','upper','velocity','effort')};limits['names']=[j.get('name') for j in joints]
    return protocol,freeze,canonical,provenance,limits


def collect_run(raw,assignment,protocol,provenance,limits):
    rows,counts,hashes=prior.collect(raw,assignment,protocol,provenance,limits)
    count=len(assignment['task_ids']);counts.update(assigned=count,complete=counts['recorded']==count and not any(r['interrupted'] for r in rows),unrecorded=count-counts['recorded'])
    with np.load(Path(provenance['pool'])/'pool.npz',allow_pickle=False) as z:configs=z['q']
    for r in rows:
        start,goal=configs[r['start_index']],configs[r['goal_index']]
        r.update(source_start_q_minimum_physical_clearance_rad=float(np.min(np.minimum(start-limits['lower'],limits['upper']-start))),
            source_goal_q_minimum_physical_clearance_rad=float(np.min(np.minimum(goal-limits['lower'],limits['upper']-goal))),
            source_joint_endpoint_Linf_rad=float(np.max(np.abs(goal-start))),source_joint_endpoint_necessary_speed_time_s=float(np.max(np.abs(goal-start)/limits['velocity'])))
        r['initial_position_error_m']=None;r['initial_speed_L1_rad_s']=None;r['initially_inside_position_speed_gate']=None
        r['actual_PCG_sum']=None;r['solve_wall_s']=None;r['native_solve_s']=None;r['last_native_initial_merit']=None;r['last_native_final_merit']=None
        directory=raw/'pilot'/f"pair_{r['pair_id']:05d}";mp=directory/'metrics.json';m=read(mp) if mp.exists() else None
        solves=[read_line(line) for line in (directory/'solves.jsonl').read_text().splitlines()] if (directory/'solves.jsonl').exists() else []
        if m:
            ip=prior.value(m['initial']['position_error_m']);iv=prior.value(m['initial']['velocity_l1_rad_s'])
            r.update(initial_position_error_m=ip,initial_speed_L1_rad_s=iv,initially_inside_position_speed_gate=int(ip<=protocol['position_tolerance_m'] and iv<=protocol['velocity_l1_tolerance_rad_s']) if ip is not None and iv is not None else None)
            r.update(actual_PCG_sum=m['actual_pcg_iteration_sum'],solve_wall_s=prior.value(m['solve_wall_s']),native_solve_s=prior.value(m['native_solve_us']/1e6))
            if r['complete'] and (sum(w['work']['actual_sqp'] or 0 for w in solves)!=r['actual_sqp_sum'] or sum(w['work']['pcg_iteration_sum'] or 0 for w in solves)!=r['actual_PCG_sum'] or len(solves)!=r['feedback_solves']):raise ValueError('Saved work totals differ')
        if solves:r.update(last_native_initial_merit=prior.value(solves[-1]['work']['initial_merit']),last_native_final_merit=prior.value(solves[-1]['work']['final_merit']))
        e=r['final_position_mm']/1000 if r['final_position_mm'] is not None else None
        r['endpoint_running_position_term']=.5*protocol['solver_params']['q_cost']*e*e if e is not None else None
        r['endpoint_terminal_position_term']=.5*protocol['solver_params']['N_cost']*e*e if e is not None else None
    counts['initially_inside_position_speed_gate']=sum(r['initially_inside_position_speed_gate']==1 for r in rows)
    counts['initial_gate_known_cases']=sum(r['initially_inside_position_speed_gate'] is not None for r in rows)
    return rows,counts,hashes


def solve_rows(raw,protocol):
    rows=[]
    for pair in protocol['run_task_ids']:
        path=raw/'pilot'/f'pair_{pair:05d}'/'solves.jsonl'
        if not path.exists():continue
        for line in path.read_text().splitlines():
            r=read_line(line);w=r['work'];rows.append(dict(pair_id=pair,SQP_cap=protocol['solver_params']['max_sqp_iters'],step=r['step'],feedback_time_s=r['feedback_time_s'],executed_duration_s=r['executed_duration_s'],actual_SQP=w['actual_sqp'],actual_PCG=w['pcg_iteration_sum'],zero_PCG_exit=int(w['zero_pcg_outer_exit']),stop_reason=w['stop_reason'],native_solve_s=prior.value(r['native_solve_us']/1e6),solve_wall_s=prior.value(r['solve_wall_s']),initial_merit=prior.value(w['initial_merit']),final_merit=prior.value(w['final_merit'])))
    return rows


VIOLATION_FIELDS=['pair_id','category','first_actual_violation_joint','first_actual_violation_s','feedback_step','feedback_time_s','executed_duration_s','seed_category_pass','returned_category_pass','actual_prefix_category_pass','seed_maximum_speed_ratio','returned_maximum_speed_ratio','returned_first_speed_violation_joint','returned_first_speed_violation_s','returned_first_future_maximum_speed_ratio','actual_prefix_maximum_speed_ratio','actual_prefix_maximum_q_excess_rad','actual_prefix_maximum_torque_ratio','actual_SQP','zero_PCG_outer_exit','SQP_cap','SQP_cap_hit','stop_reason','selected_returned_npz','selected_returned_sha256','status']

def violation_feedback(raw,rows,protocol,limits):
    """Read the saved feedback associated with each first finite physical violation."""
    result=[]
    for r in rows:
        if not r['complete'] or r['execution_failure'] or not r['physical_finite'] or r['completed_episode_physical_pass']:continue
        directory=raw/'pilot'/f"pair_{r['pair_id']:05d}";path=directory/'solves.jsonl'
        solves=[read_line(line) for line in path.read_text().splitlines()] if path.exists() else []
        for category in ('joint','velocity','effort'):
            when=r[category+'_first_time_s']
            if when is None:continue
            item={key:None for key in VIOLATION_FIELDS};item.update(pair_id=r['pair_id'],category=category,first_actual_violation_joint=r[category+'_first_joint'],first_actual_violation_s=when,status='saved_feedback_unavailable')
            # A state exactly on a feedback boundary belongs to the preceding
            # executed prefix; t0 command violations belong to the first solve.
            prior_steps=[w for w in solves if w['feedback_time_s']<when-1e-9]
            chosen=prior_steps[-1] if prior_steps else solves[0] if solves and abs(when)<1e-9 else None
            if chosen and when<=chosen['feedback_time_s']+chosen['executed_duration_s']+1e-8:
                seed=chosen.get('seed_limits',{});pred=chosen.get('predicted_limits',{});prefix=chosen.get('executed_prefix_limits',{});work=chosen['work'];first=pred.get('velocity',{}).get('first_violation')
                returned=directory/f"step_{chosen['step']:04d}_returned.npz";future_ratio=None
                if returned.exists():
                    with np.load(returned,allow_pickle=False) as z:
                        packed=z['optimized'][0];n=protocol['N'];body=packed[:(n-1)*21].reshape(n-1,21);states=np.vstack([body[:,:14],packed[-14:]])
                        if np.isfinite(states[1,7:]).all():future_ratio=float(np.max(np.abs(states[1,7:])/limits['velocity']))
                item.update(feedback_step=chosen['step'],feedback_time_s=chosen['feedback_time_s'],executed_duration_s=chosen['executed_duration_s'],seed_category_pass=int(seed[category]['pass_check']) if category in seed else None,returned_category_pass=int(pred[category]['pass_check']) if category in pred else None,actual_prefix_category_pass=int(prefix[category]['pass_check']) if category in prefix else None,seed_maximum_speed_ratio=prior.value(seed.get('maximum_velocity_ratio')),returned_maximum_speed_ratio=prior.value(pred.get('maximum_velocity_ratio')),
                    returned_first_speed_violation_joint=first['joint_name'] if first else None,returned_first_speed_violation_s=prior.value(first['time_s']) if first else None,returned_first_future_maximum_speed_ratio=future_ratio,
                    actual_prefix_maximum_speed_ratio=prior.value(prefix.get('maximum_velocity_ratio')),actual_prefix_maximum_q_excess_rad=prior.value(prefix.get('joint',{}).get('maximum_excess')),actual_prefix_maximum_torque_ratio=prior.value(prefix.get('maximum_effort_ratio')),
                    actual_SQP=work['actual_sqp'],zero_PCG_outer_exit=int(work['zero_pcg_outer_exit']),SQP_cap=protocol['solver_params']['max_sqp_iters'],SQP_cap_hit=int(work['actual_sqp']>=protocol['solver_params']['max_sqp_iters']),stop_reason=work['stop_reason'],selected_returned_npz=str(returned),selected_returned_sha256=sha(returned) if returned.exists() else None,status='saved_feedback_matched')
            result.append(item)
    return result

def read_line(line):
    import json
    return json.loads(line)

def timeout_context(rows):
    """Summarize existing endpoint/progress fields without extrapolating arrival."""
    timeouts=[r for r in rows if r['complete'] and r['outcome']=='timeout' and not r['execution_failure']]
    bins=[('<=50',0,50),('>50 to100',50,100),('>100 to200',100,200),('>200',200,float('inf'))]
    distribution=[dict(endpoint_position_error_mm=label,count=sum(r['final_position_mm'] is not None and (r['final_position_mm']<=upper if lower==0 else lower<r['final_position_mm']<=upper) for r in timeouts)) for label,lower,upper in bins]
    def stats(key,scale=1):
        values=[r[key]*scale for r in timeouts if r[key] is not None]
        return dict(observed=len(values),minimum=min(values) if values else None,median=float(np.median(values)) if values else None,maximum=max(values) if values else None)
    return distribution,dict(timeouts=len(timeouts),endpoint_over100mm=sum(r['final_position_mm'] is not None and r['final_position_mm']>100 for r in timeouts),endpoint_mm=stats('final_position_mm'),last_second_position_error_reduction_mm=stats('last_second_net_position_error_reduction_m',1000),last_second_joint_total_variation_L1_rad=stats('last_second_joint_total_variation_L1_rad'),positive_last_second_position_error_reduction=sum(r['last_second_net_position_error_reduction_m'] is not None and r['last_second_net_position_error_reduction_m']>0 for r in timeouts))

def render(output,source,previews):
    for name in ('analyze_validation.py','analyze_frozen_validation.py','report_to_pdf.py','plot_reaching_histogram.py'):
        path=source/'experiments/random_reaching_baseline'/name
        if path.resolve()!=(output/name).resolve():shutil.copy2(path,output/name)
    spec=importlib.util.spec_from_file_location('report_to_pdf',output/'report_to_pdf.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.render(output/'REPORT.md',previews)


def report(output,rows,counts,assignment,protocol,freeze,provenance,canonical):
    lines=['# Frozen OFF_TORQUE: fresh100 validation','',('Complete validation.' if counts['complete'] else 'INCOMPLETE PREVIEW; no outcome is inferred for missing cases.'),'',
        f"Position+speed arrivals {counts['rough_arrivals']}/100; complete finite error-free physical passes {counts['physical_passes']}/100; both {counts['both_passes']}/100. Timeouts {counts['timeouts']}, execution failures {counts['execution_failures']}, unrecorded {counts['unrecorded']}. The fixed100 denominator retains every assigned task. Incomplete numerators/curves are observed lower bounds.",'',
        f"The frozen OFF_TORQUE development configuration had9/12 arrivals and12/12 physical passes. This fresh100 is separate, sampled without replacement from {assignment['remaining_population_size']} of10000 original pair IDs using seed{assignment['seed']}. Every historical100 ID and every old stopped-M14 assignment100 ID is excluded, including all development12. No geometry/outcome filtering, IK resampling, clipping, tuning, parameter selection, retries or rescue follows these validation outcomes. The old M14 partial is not resumed or merged; no112-task rate is reported.",'',
        '## Arrival distribution and physical outcomes','', '| Outcome | Count /100 |','|---|---:|']
    for key in ('rough_arrivals','physical_passes','both_passes','timeouts','execution_failures','physical_timeouts','arrivals_with_physical_failure','unrecorded'):lines.append(f"| {key.replace('_',' ')} | {counts[key]} |")
    lines+=['','![The cumulative curves divide by all100; the histogram is conditional on observed arrivals. Timeouts are right-censored at8s; execution failures are separate failed outcomes. Unrecorded tasks have no event/censoring time. No survival extrapolation is used.](arrivals.png)','',
        '![All assigned100 tasks are represented: successes use recorded arrival times; completed non-arrivals are displayed in the separate8s/no-arrival bin. Timeout and execution-error segments remain distinct. Arrivals use the position+speed criterion regardless of physical compliance, which remains separate. This display value is not an arrival time or actual error runtime; means and success labels are unchanged. Missing cases receive no inferred bin.](reaching_histogram.png)','',
        'Primary arrival uses position <=50mm and speed L1 <=1rad/s at 30 ms feedback boundaries, without orientation. Raw physical checks use saved3ms states/applied controls, q excess <=1e-6rad and speed/torque ratios <=1+1e-6; soft interior bounds do not replace these gates. Physical passes require complete error-free finite observations. Raw prefix flags and observed-prefix extrema remain separate; null error/nonfinite full-history maxima are never zero.','',
        f"Initially inside the position+speed gate: {counts['initially_inside_position_speed_gate']} among {counts['initial_gate_known_cases']} recorded initial observations. No such task is filtered or removed. Initial position error/speed and the flag remain per case; arrival is still first checked after the original first feedback step. Already-close tasks must not be described as newly completed large motions.",'',
        f"Failed category checks: joint{counts['joint_failed_checks']}, speed{counts['velocity_failed_checks']}, torque{counts['effort_failed_checks']}. These may overlap. All physical failures and all timeouts/errors remain in failure_cases.csv; all100 assigned rows remain in cases.csv.",'',
        '![Every available finite full episode contributes signed q clearance and speed/torque peaks, including failures. Negative q clearance means a raw joint bound was exceeded. Error/nonfinite full observations remain null.](physical_extrema.png)','',
        '## Representative and worst observations','', '| ID | Selection | Outcome | Arrival / physical | End mm | q clearance rad | Speed / torque ratio |','|---|---|---|---|---:|---:|---|']
    for r,labels in prior.examples(rows):lines.append(f"| {r['pair_id']} | {', '.join(labels)} | {r['outcome']} | {r['task_success']}/{r['completed_episode_physical_pass']} | {num(r['final_position_mm'])} | {num(r['minimum_q_clearance_rad'])} | {num(r['maximum_speed_ratio'])}/{num(r['maximum_torque_ratio'])} |")
    distribution,context=timeout_context(rows);progress=context['last_second_position_error_reduction_mm'];motion=context['last_second_joint_total_variation_L1_rad']
    lines+=['','## Timeout endpoints and saved progress','',f"Among {context['timeouts']} recorded error-free timeouts, {context['endpoint_over100mm']} ended more than100mm from the target. Median endpoint error was {num(context['endpoint_mm']['median'])}mm. These outcomes remain in the full assigned100 denominator.",'','| Final position error mm | Timeout count |','|---|---:|']
    for r in distribution:lines.append(f"| {r['endpoint_position_error_mm']} | {r['count']} |")
    lines+=['',f"Over the saved final approximately1s, position-error reduction had median {num(progress['median'])}mm and range {num(progress['minimum'])} to {num(progress['maximum'])}mm; {context['positive_last_second_position_error_reduction']}/{context['timeouts']} showed positive reduction. Joint total variation L1 had median {num(motion['median'])}rad and range {num(motion['minimum'])} to {num(motion['maximum'])}rad. Negative reduction means the final error increased. These are saved endpoint/motion descriptions, not convergence tests or evidence that every timeout would reach with more time. A soft goal can progress slowly, change direction or stall. Compact distribution/context files and per-case numeric progress fields are retained.",'',
        'Source-start and source-goal-q minimum raw joint clearances, endpoint joint displacement and necessary speed-only joint time bound are descriptive witness geometry, not filters or dynamic-feasibility certificates. The source goal q is a recorded FK witness, not a newly solved IK seed.','',
        '## Work, objective scale and frozen protocol','', '| Quantity | All recorded episodes |','|---|---:|']
    for key in ('feedback_solves','actual_sqp_sum','actual_PCG_sum','zero_pcg_exit_solves','episode_wall_s','solve_wall_s','native_solve_s'):
        values=[r[key] for r in rows if r['complete'] and r[key] is not None];lines.append(f"| {key} | {num(sum(values)) if values else 'null'} |")
    lines+=['','Every case retains native/host/episode timing and work aggregates; the full per-solve CSV remains in raw analysis/solves.csv. The final native merit is descriptive and includes the native merit conventions. Endpoint running/terminal position-only terms are .5*q_cost*error² and .5*N_cost*error² at the actual final position; they are neither the complete optimized horizon objective nor a proof of local optimality.','',
        f"Mode A/controller_dt, B1/N32, .03s knots/feedback, .003s RK4, .93s prediction,8s episode; solve time advances no plant time. Final approach is off. Saved canonical remains failed at{num(canonical['final']['position_error_m']*1000)}mm after8s, physical pass; it is openly reused as diagnostic context under the explicit continuation request, never as passed qualification. Shared setup/import/module/constructor failures hold; individual failures continue without retries.",'',
        'Cold repeated actual rest state and zero controls, recorded/applied first t0 solve, warm shift/reanchor, once-per-episode dual/rho reset then retained adaptive state remain unchanged. Natural zero-PCG stopping remains a practical heuristic, not a stationarity certificate; no KKT gate or new residual instrumentation is used.','', '| Parameter | Frozen value |','|---|---:|']
    for key,v in protocol['solver_params'].items():lines.append(f'| {key} | {v} |')
    lines+=['','C2 interior margins remain.20rad/.80/.85 with1%-interval continuation; original physical gates/model remain. These are soft penalties, not hard guarantees.','',
        '## Data and reproduction','',f"Assignment SHA256: {sha(output/'assignment.json')}. Native SHA256: {provenance['module_sha256']}. Worker SHA256: {provenance['wrapper_sha256']}.",'',
        'metadata.json retains full assignment/protocol/preparation/launch/freeze/canonical/source/module/input identities, all per-case hashes and the exact analysis command. Large raw horizons/history/archives remain outside thesis_results. Summary equals every recorded metric; final export requires exact ordered100 completion. Editable case/failure data, plots and scripts travel with this self-contained PDF.','',
        '```sh','python analyze_frozen_validation.py --plot-only .','python report_to_pdf.py --report REPORT.md','```','',
        'This one-pass simulated validation describes the frozen configuration and original uniform-q/FK pair sampling. It does not establish universal reliability, continuous-time safety, collision clearance, paid-computation performance or physical-robot behavior. Arrival ends recording; no post-arrival settling claim is possible.','']
    failures=csv_read(output/'violation_feedback.csv')
    if failures:
        where=lines.index('## Representative and worst observations')
        detail=['## Saved prediction versus executed violation','','This table links each first violation in a finite error-free physical-failure episode to the preceding feedback solve. Seed/prediction/prefix maxima are the original saved checks. A full predicted horizon and an actual 30 ms RK4 prefix are distinct objects; these observations do not isolate solver, model or integration causes. All 100 assignments/outcomes remain unchanged.','','| ID / first actual category | First joint / time s | Step / feedback s | Seed / predicted / actual-prefix speed ratio | Predicted first-future speed ratio | Actual SQP / cap | Zero-PCG / cap hit |','|---|---|---|---|---:|---:|---|']
        for r in failures:
            f=lambda key:num(float(r[key])) if r[key] else 'null'
            detail.append(f"| {r['pair_id']} / {r['category']} | {r['first_actual_violation_joint']} / {f('first_actual_violation_s')} | {r['feedback_step'] or 'unavailable'} / {f('feedback_time_s')} | {f('seed_maximum_speed_ratio')} / {f('returned_maximum_speed_ratio')} / {f('actual_prefix_maximum_speed_ratio')} | {f('returned_first_future_maximum_speed_ratio')} | {r['actual_SQP'] or 'null'} / {r['SQP_cap'] or 'null'} | {r['zero_PCG_outer_exit'] or 'null'} / {r['SQP_cap_hit'] or 'null'} |")
        detail+=['','violation_feedback.csv retains seed/returned/actual-prefix pass flags for the violated category, exact first predicted-speed violation joint/time when present, first-future knot speed, actual prefix q/torque maxima, native stop reason and the specific returned-file hash. The first-future value is one predicted knot at 30 ms; the actual-prefix maximum spans all saved plant substeps. Neither is substituted for the other. Error/nonfinite episodes remain separate in the full failure table with null full-history extrema. No new simulation, residual evaluation, retry, convergence gate or parameter decision is performed.','']
        lines[where:where]=detail
    text='\n'.join(lines)
    for a,b in {'fresh100':'fresh 100','fixed100':'fixed 100','all100':'all 100','of10000':'of 10000','seed20261015':'seed 20261015','had9/12':'had 9/12','and12/12':'and 12/12','historical100':'historical 100','assignment100':'assignment 100','development12':'development 12','no112':'no 112','at8s':'at 8 s','<=50mm':'<= 50 mm','<=1rad/s':'<= 1 rad/s','at 30 ms':'at 30 ms','saved3ms':'saved 3 ms','<=1e-6rad':'<= 1e-6 rad','<=1+1e-6':'<= 1+1e-6','joint{':'joint {','after8s':'after 8 s','.20rad':'.20 rad','with1%':'with 1%'}.items():text=text.replace(a,b)
    (output/'REPORT.md').write_text(text)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('raw-root','source','output','previews'):p.add_argument('--'+key,type=Path)
    p.add_argument('--allow-partial',action='store_true');p.add_argument('--plot-only',type=Path);args=p.parse_args()
    if args.plot_only:prior.plot(args.plot_only.resolve());histogram.plot(args.plot_only.resolve());return
    if not args.raw_root or not args.source or not args.output:p.error('Provide --raw-root --source --output')
    raw,source,output=(getattr(args,k).resolve() for k in ('raw_root','source','output'));prep=raw/'preparation'
    protocol,freeze,canonical,provenance,limits=verify_run(raw,prep,source,prep/'validation_protocol.json','fresh100_validation_only');assignment=read(prep/'assignment.json')
    if sha(prep/'assignment.json')!=protocol['validation_assignment']['sha256'] or assignment['task_ids']!=protocol['run_task_ids'] or set(assignment['task_ids'])&set(assignment['excluded_task_ids']):raise ValueError('Frozen fresh assignment differs')
    rows,counts,hashes=collect_run(raw,assignment,protocol,provenance,limits)
    if not counts['complete'] and (not args.allow_partial or 'thesis_results' in output.parts):raise ValueError('Incomplete100 requires temporary --allow-partial output')
    if output.is_relative_to(raw):raise ValueError('Do not overwrite raw storage')
    output.mkdir(parents=True,exist_ok=True);(raw/'analysis').mkdir(parents=True,exist_ok=True);csv_write(output/'cases.csv',rows);csv_write(output/'failure_cases.csv',[r for r in rows if r['complete'] and (not r['task_success'] or not r['completed_episode_physical_pass'] or r['execution_failure'])],list(rows[0]))
    csv_write(raw/'analysis'/'solves.csv',solve_rows(raw,protocol),['pair_id','SQP_cap','step','feedback_time_s','executed_duration_s','actual_SQP','actual_PCG','zero_PCG_exit','stop_reason','native_solve_s','solve_wall_s','initial_merit','final_merit']);csv_write(output/'violation_feedback.csv',violation_feedback(raw,rows,protocol,limits),VIOLATION_FIELDS);write(output/'counts.json',counts);write(output/'protocol.json',protocol);shutil.copy2(prep/'assignment.json',output/'assignment.json')
    distribution,context=timeout_context(rows);csv_write(output/'timeout_endpoint_distribution.csv',distribution);write(output/'timeout_context.json',context)
    if (raw/'backup_receipt.json').exists():shutil.copy2(raw/'backup_receipt.json',output/'raw_backup_receipt.json')
    records={str(path):dict(sha256=sha(path),data=read(path)) for path in [*prep.glob('*.json'),*raw.glob('*.json')]}
    write(output/'metadata.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),raw_root=str(raw),source=str(source),analysis_command=shlex.join([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]]),analysis_sha256=sha(__file__),per_call_csv_path=str(raw/'analysis'/'solves.csv'),per_call_csv_sha256=sha(raw/'analysis'/'solves.csv'),cohort_provenance=provenance,freeze=freeze,canonical=canonical,violation_feedback_sha256=sha(output/'violation_feedback.csv'),timeout_context_sha256=sha(output/'timeout_context.json'),timeout_distribution_sha256=sha(output/'timeout_endpoint_distribution.csv'),records=records,case_sha256=hashes))
    prior.plot(output);histogram.plot(output);report(output,rows,counts,assignment,protocol,freeze,provenance,canonical);pages=render(output,source,args.previews)
    print(dict(counts=counts,pdf_pages=pages,output=str(output)))

if __name__=='__main__':main()
