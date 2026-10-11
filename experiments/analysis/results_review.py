#!/usr/bin/env python3
"""Bounded saved BA compatibility and approved historical BB pooling; no solves."""
import argparse
from collections import Counter
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.f01ab import read, csv_write
from recording import write_json, sha


def controller_crash(metrics):
    # Only the explicitly approved failure; generic execution errors stay unavailable.
    return 'nonfinite executed plant state' in str(metrics.get('execution_error') or metrics.get('failure') or '')


def task_success(metrics):
    if 'task_success' in metrics:
        return metrics['task_success']
    return metrics.get('outcome', '').lower() == 'reached' if metrics.get('complete') else None


def outcome(metrics):
    crash = controller_crash(metrics)
    return dict(known_controller_crash=crash,
        reported_task_success=False if crash else task_success(metrics),
        reported_physical_pass=False if crash else metrics.get('physical_pass', metrics.get('physical', {}).get('physical_valid')))


def episode_rows(root, legacy=False):
    root = Path(root)
    for path in sorted(root.rglob('summary.json' if legacy else 'metrics.json')):
        m = read(path)
        params = read(path.parent/'receipt.json').get('case', {}).get('parameters', {})
        pair_id, batch = m.get('pair_id', params.get('pair_id')), m.get('batch', params.get('batch'))
        if pair_id is None or batch is None or (not legacy and m.get('campaign_id') != 'BB'):
            continue
        manifest = read(path.parent/'manifest.json')
        provenance = read(path.parent/'provenance.json')
        yield dict(source_attempt=str(path.parent), metrics_sha256=sha(path),
            pair_id=pair_id, B=batch, cohort=m.get('cohort_id'),
            module_sha256=provenance.get('module_sha256',manifest.get('binary_sha256')),
            source_commit=provenance.get('source_commit',manifest.get('commit')),
            protocol_sha256=provenance.get('frozen_protocol_sha256',manifest.get('protocol_sha256')),
            stopping_policy='historical-before-zero-PCG-restoration', raw_outcome=m.get('outcome'),
            raw_task_success=m.get('task_success'), raw_physical_pass=m.get('physical_pass',m.get('physical',{}).get('physical_valid')),
            execution_error=m.get('execution_error',m.get('failure')), final_metrics=m.get('final'), **outcome(m))


def combined_bb(legacy, continuation, output):
    rows = list(episode_rows(legacy, True)) + list(episode_rows(continuation))
    keys = [(r['pair_id'],r['B']) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate pair/arm attempts: preserve them, do not silently choose a retry')
    if len(rows) != 200 or len(set(r['pair_id'] for r in rows)) != 100:
        raise ValueError('Approved BB report requires exactly 100 pairs / 200 source attempts')
    if len(set(r['module_sha256'] for r in rows)) != 1 or None in {r['module_sha256'] for r in rows}:
        raise ValueError('BB modules unavailable or incompatible; do not pool stopping policies')
    counts=[]
    for batch in (1,16):
        selected=[r for r in rows if r['B']==batch]
        counts.append(dict(B=batch,denominator=len(selected),known_controller_crashes=sum(r['known_controller_crash'] for r in selected),
            task_pass=sum(r['reported_task_success'] is True for r in selected),
            physical_pass=sum(r['reported_physical_pass'] is True for r in selected),
            both_pass=sum(r['reported_task_success'] is True and r['reported_physical_pass'] is True for r in selected),
            task_unavailable=sum(r['reported_task_success'] is None for r in selected),
            physical_unavailable=sum(r['reported_physical_pass'] is None for r in selected)))
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    csv_write(output/'bb100_attempts.csv',rows);csv_write(output/'bb100_counts.csv',counts)
    write_json(output/'bb100_provenance.json',dict(legacy=str(legacy),continuation=str(continuation),analysis_sha256=sha(__file__),
        scope='Approved historical old-solver 12+88 pairs only; known nonfinite plant crashes count as controller failures. Original errors and unavailable final metrics retained. No cause/reliability claim.'))
    return counts


def compatible(left, right):
    """Check actual task/settings; max SQP and batch are the intended arm changes."""
    a,b = read(left/'manifest.json'),read(right/'manifest.json')
    failures=[]
    for key in ('binary_sha256','N','knot_dt','feedback_seconds','simulation_step','simulation_timeout',
                'endpoint','initialization','proposal','dual_policy','rho_policy','physical_limits','success'):
        if a.get(key) is None or a.get(key)!=b.get(key):failures.append(key)
    sa,sb=dict(a.get('solver_params',{})),dict(b.get('solver_params',{}))
    sa.pop('max_sqp_iters',None);sb.pop('max_sqp_iters',None)
    if not sa or sa!=sb:failures.append('solver_params_except_intended_SQP')
    for key in ('feedback','sim_dt','seconds','position_tolerance','tool_axis_tolerance_deg','velocity_l1_tolerance','seed','prefix_guard'):
        if a.get('args',{}).get(key)!=b.get('args',{}).get(key):failures.append('args.'+key)
    with np.load(left/'inputs.npz',allow_pickle=False) as x, np.load(right/'inputs.npz',allow_pickle=False) as y:
        for key in ('initial_state','target_xyz','target_rotation','reference'):
            if key == 'reference':
                # Batch replication is an intended arm difference; check every lane.
                matches = x[key].shape[-1:] == y[key].shape[-1:] and np.allclose(x[key], y[key], atol=1e-5, rtol=1e-5)
            else:
                matches = x[key].shape == y[key].shape and np.allclose(x[key], y[key], atol=1e-5, rtol=1e-5)
            if not matches:failures.append('task.'+key)
    return failures


def paired_ba(baseline, nominal, output):
    old={};new={}
    for root,target in ((Path(baseline),old),(Path(nominal),new)):
        for p in root.rglob('manifest.json'):
            m=read(p)
            if 'batch' in m and 'solver_params' in m and (p.parent/'summary.json').exists():
                key=(m['pair_id'],m['batch'])
                if key in target:raise ValueError('Multiple BA attempts for pairing key')
                target[key]=p.parent
    rows=[]
    for pair in sorted({k[0] for k in new}):
        for batch in (1,16):
            a,b=old.get((pair,batch)),new.get((pair,1))
            failures=compatible(a,b) if a and b else ['missing_arm']
            row=dict(pair_id=pair,baseline_B=batch,compatible=not failures,failed_fields=failures,
                baseline_attempt=str(a),nominal_attempt=str(b))
            for name,path in (('baseline',a),('nominal16x',b)):
                m=read(path/'summary.json') if path else {}
                row[name+'_task_success']=task_success(m);row[name+'_physical_pass']=m.get('physical',{}).get('physical_valid')
            rows.append(row)
    output=Path(output);output.mkdir(parents=True,exist_ok=True);csv_write(output/'ba_pair_compatibility.csv',rows)
    write_json(output/'ba_pair_compatibility.json',dict(comparisons=len(rows),compatible=sum(r['compatible'] for r in rows),
        failures=dict(Counter(field for r in rows for field in r['failed_fields'])),
        intended_changes='batch B1/B16 and max_sqp_iters 5/80; task arrays atol/rtol1e-5',analysis_sha256=sha(__file__)))
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bb-legacy',type=Path,required=True);p.add_argument('--bb-continuation',type=Path,required=True)
    p.add_argument('--ba-baseline',type=Path,required=True);p.add_argument('--ba-nominal',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(combined_bb(a.bb_legacy,a.bb_continuation,a.output));print(len(paired_ba(a.ba_baseline,a.ba_nominal,a.output)),'BA paired compatibility checks')
