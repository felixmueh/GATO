#!/usr/bin/env python3
"""Measured M04 response/motion envelope from saved fault observations only."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recording import write_json,sha


def lines(path,parse_errors=None):
    if not Path(path).is_file():return []
    result=[]
    raw_lines=Path(path).read_text().splitlines()
    for number,line in enumerate(raw_lines,1):
        try:result.append(json.loads(line))
        except json.JSONDecodeError as error:
            if parse_errors is not None:parse_errors.append(dict(file=str(path),line=number,error=str(error),terminal_line=number==len(raw_lines)))
    return result


def analyze_trial(path,parse_errors=None):
    path=Path(path);meta=json.loads((path/'trial.json').read_text())
    events=sorted([r for p in path.glob('events-*.jsonl') for r in lines(p,parse_errors)],key=lambda r:r['wall'])
    observations=lines(path/'observer.jsonl',parse_errors)
    injected=[r['wall'] for r in events if r.get('kind')=='injected']
    active=[r['wall'] for r in events if r.get('kind')=='effort_activated']
    t0=injected[0] if injected else active[0]+meta.get('inject_after_wall_sec',2.) if meta['case']=='baseline' and active else None
    post=[r for r in observations if t0 is not None and r['wall']>=t0]
    prior=[r for r in observations if t0 is not None and r['wall']<t0]
    before=prior[-1] if prior else None
    def latency(kind):
        found=[r['wall']-t0 for r in events if t0 is not None and r['wall']>=t0 and r.get('kind')==kind]
        return found[0] if found else None
    controller_error=[r['wall']-t0 for r in events if t0 is not None and r['wall']>=t0 and r.get('kind')=='controller_status' and str(r.get('mode','')).startswith('ERROR')]
    result=json.loads((path/'result.json').read_text()) if (path/'result.json').is_file() else {}
    displacement=max((float(np.max(np.abs(np.asarray(r['q'])-before['q']))) for r in post),default=None) if before else None
    settled=None
    for i,r in enumerate(post):
        window=[s for s in post[i:] if s['wall']<=r['wall']+.5]
        if (len(window)>1 and window[-1]['wall']-window[0]['wall']>=.45 and
            max(np.diff([s['wall'] for s in window]))<.1 and all(max(np.abs(s['qd']))<.05 for s in window) and
            np.max(np.ptp(np.asarray([s['q'] for s in window]),axis=0))<.01):
            settled=r['wall']-t0;break
    return dict(path=str(path),case=meta['case'],fault_reference_wall=t0,observed_samples=len(post),
        controller_error_latency_s=controller_error[0] if controller_error else None,
        restore_request_latency_s=latency('restore_requested'),restore_confirmed_latency_s=result.get('restore_confirmed_latency_sec'),
        first_observed_stable_window_s=settled,max_observed_joint_displacement_rad=displacement,
        max_observed_speed_rad_s=max((max(np.abs(r['qd'])) for r in post),default=None),
        observation_duration_s=post[-1]['wall']-t0 if post else None,
        maximum_observer_gap_s=float(np.max(np.diff([r['wall'] for r in post]))) if len(post)>1 else None,
        state_age_s=None,state_age_reason='ROS sim timestamps and host monotonic receipt timestamps are different clock domains; no recorded mapping',
        clearance_m=None,clearance_reason='No co-temporal full-body geometry/clearance trace recovered',
        certified_braking_distance_m=None,braking_reason='Observed motion includes intended reference motion; no validated deceleration bound or stopping certificate',
        outcome=result.get('outcome','inconclusive'),observer_error=meta.get('observer_error'),supervisor_intervened=meta.get('supervisor_intervened'),
        simulation_only=True)


def analyze(raw,output):
    raw,output=Path(raw).resolve(),Path(output).resolve()
    if output.is_relative_to(raw):raise ValueError('Derived envelope must be outside retained raw')
    output.mkdir(parents=True,exist_ok=False)
    rows=[];errors=[];parse_errors=[]
    for path in sorted(raw.rglob('trial.json')):
        try:rows.append(analyze_trial(path.parent,parse_errors))
        except (OSError,ValueError,KeyError,TypeError) as error:errors.append(dict(path=str(path),error=str(error)))
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (output/'response_motion.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    write_json(output/'summary.json',dict(trials=len(rows),errors=errors,parse_errors=parse_errors,raw=str(raw),analysis_sha256=sha(__file__),
        status='measured_saved_data_analysis' if rows else 'historical_source_recovery_pending_no_trials',
        scope='Observed response/displacement/stable-window illustration only. Source protocols/outcomes remain separate; unavailable state age/clearance/braking bounds retained as null. No physical safety guarantee.'))
    return len(errors)+len(parse_errors)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();sys.exit(bool(analyze(args.raw,args.output)))
