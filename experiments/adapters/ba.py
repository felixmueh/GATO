#!/usr/bin/env python3
"""Frozen BA task adapter: canonical A or the declared nominal16x SQP arm."""
import argparse
import json
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from adapters.bb import load_worker
from recording import sha, write_json


def load_protocol(path,base_path):
    base=json.loads(Path(base_path).read_text()); protocol=json.loads(Path(path).read_text())
    expected=dict(base)
    if Path(path).resolve()!=Path(base_path).resolve():
        expected.update(id='BA-nominal16x-A-v1',cohort_id='BA-nominal16x-A-v1',
            status='Prepared nominal16xSQP arm; no realized-equal-work claim',
            arms=[dict(batch_size=1,max_sqp_iters=80,max_pcg_iters=120)],
            solver_params=dict(base['solver_params'],max_sqp_iters=80))
    if protocol!=expected or protocol['timing_mode']!='A' or protocol['offline_timing']!='controller_dt':
        raise ValueError('Frozen BA base/nominal16x protocol differs')
    return protocol


def run_case(source,build,inputs,case,output,preflight=False,batch=None,worker=None,preflight_arm='default'):
    source,build,inputs,output=map(Path,(source,build,inputs,output));output.mkdir(parents=True,exist_ok=True)
    metrics=dict(campaign_id='BA-readiness' if preflight else 'BA',research_question='RQ-batching',
        timing_mode='A',offline_timing='controller_dt',execution_backend='internal',
        task_success=None,physical_pass=None,execution_error=None,
        evidence_kind='synthetic_test' if worker else 'measurement')
    try:
        parameters=case['parameters'] if not preflight else dict(pair_id=-1,batch=batch,arm=preflight_arm)
        arm=parameters['arm']; B=parameters['batch']; pair=parameters['pair_id']
        if arm not in ('default','nominal16x') or B not in (1,16) or (arm=='nominal16x' and B!=1):
            raise ValueError('BA supports only frozen B1/B16 default or nominal16x B1')
        protocol_path=inputs/('ba_nominal16x.json' if arm=='nominal16x' else 'ba_protocol.json')
        protocol=load_protocol(protocol_path,inputs/'ba_protocol.json')
        if not preflight and pair not in protocol['task_ids']:raise ValueError('BA pair outside frozen100 roster')
        matches=list((build/'modules').glob('bsqpN32_tiago_right*.so'))
        if len(matches)!=1:raise ValueError('Expected one BA N32 module')
        write_json(output/'provenance.json',dict(module_sha256=sha(matches[0]),source_root=str(source),
            adapter_sha256=sha(__file__),protocol_sha256=sha(protocol_path),arm=arm,
            cohort_id=protocol.get('cohort_id',protocol['id']),
            evidence_kind='synthetic_test' if worker else 'measurement',
            nominal_compute='B1 SQP80 versus B16 SQP5, PCG120 per SQP; early termination/work/timing descriptive'))
        worker=worker or load_worker(source.resolve(),matches[0].resolve())
        worker.load_protocol=lambda path=None: load_protocol(path or protocol_path,inputs/'ba_protocol.json')
        worker.cohort_id=lambda p:p.get('cohort_id',p['id'])
        args=SimpleNamespace(protocol=protocol_path,pool=inputs,output=output/'episode',
            preflight_input=inputs/('ba_nominal16x_preflight.json' if arm=='nominal16x' else 'ba_preflight.json') if preflight else None,evidence_kind=metrics['evidence_kind'],
            feedback=.03,sim_dt=.003,seconds=8.,position_tolerance=.05,tool_axis_tolerance_deg=30.,
            velocity_l1_tolerance=1.,seed=20260925,vel_lim_cost=.1,qd_cost=.1,prefix_guard=False)
        summary=worker.run_episode(args,pair,B,.03)
        if not summary.get('complete'):raise ValueError('BA incomplete summary')
        metrics.update(summary,physical_pass=summary['physical']['physical_valid'],parameters=parameters)
    except BaseException:
        metrics.update(outcome='execution_failed',execution_error=traceback.format_exc())
    write_json(output/'metrics.json',metrics)
    return 1 if metrics['execution_error'] or (preflight and not (metrics['task_success'] and metrics['physical_pass'])) else 0


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('source','build','inputs','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--case',type=Path)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--batch',type=int,choices=(1,16))
    parser.add_argument('--arm',choices=('default','nominal16x'),default='default',help='Preflight parameterization')
    args=parser.parse_args()
    if not args.preflight and not args.case:parser.error('--case or --preflight required')
    if args.preflight and not args.batch:parser.error('--batch required for preflight')
    case=json.loads(args.case.read_text()) if args.case else None
    sys.exit(run_case(args.source,args.build,args.inputs,case,args.output,args.preflight,args.batch,preflight_arm=args.arm))
