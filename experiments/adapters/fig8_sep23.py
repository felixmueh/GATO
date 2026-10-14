#!/usr/bin/env python3
"""One saved-settings modeB tracking counterpart; CSV extraction is CPU-only."""
import argparse
import csv
import importlib
import json
from pathlib import Path
import sys
import traceback
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from adapters import task
from recording import iter_events,load_controller_stats,sha,write_json


def export_feedback(source,output):
    source,output=Path(source),Path(output)
    protocol=json.loads((output/'protocol.json').read_text())
    sys.path[:0]=[str(source/'python'),str(source/'tiago_src')]
    from bsqp.common import sample_reference
    import pinocchio as pin
    model=pin.buildModelFromUrdf(str(source/protocol['urdf']['path']));data=model.createData();frame=model.getFrameId('arm_right_tool_link')
    reference=np.asarray(protocol['reference_packed'],dtype='<f8')
    stats=load_controller_stats(output)
    planned={float(t):u for t,u in zip(stats.get('timestamps',[]),stats.get('planned_controls',[]))}
    if 'initial_planned_control' in stats:planned[0.]=stats['initial_planned_control']
    solves={};steps={}
    for event in iter_events(output):
        record=event['record']
        if event['event']=='solve_completed':solves[record['solve_index']]=record
        elif event['event']=='step_completed':steps[record['time_s']]=record
    rows=[]
    for index,solve in sorted(solves.items()):
        stamp=float(solve['feedback_time_s']);state=np.asarray(solve['state']).reshape(-1,14)[0]
        target=sample_reference(reference,np.asarray([stamp]),protocol['reference_dt_s'])[0]
        pin.framesForwardKinematics(model,data,state[:7].astype(np.float64));xyz=data.oMf[frame].translation.copy()
        step=steps.get(stamp);control=np.asarray(step['control']) if step is not None else np.full(7,np.nan)
        fresh=np.asarray(planned.get(stamp,np.full(7,np.nan)))
        row=dict(solve_index=index,time_s=stamp,category=solve['category'],reference_x_m=target[0],reference_y_m=target[1],reference_z_m=target[2],
                 fk_x_m=xyz[0],fk_y_m=xyz[1],fk_z_m=xyz[2],error_m=float(np.linalg.norm(xyz-target[:3])),
                 solve_wall_ms=1000*solve['wall_solve_time_s'],solve_native_ms=solve['native_solve_time_ms'],
                 applied_plan_solve_index=step['plan_solve_index'] if step is not None else None,
                 applied_control_age_s=step['active_control_age_s'] if step is not None else None,
                 applied_substep_dt_s=step['step_dt_s'] if step is not None else None)
        row.update({f'q{j}_rad':state[j] for j in range(7)});row.update({f'v{j}_rad_s':state[7+j] for j in range(7)})
        row.update({f'applied_u{j}_Nm':control[j] for j in range(7)});row.update({f'new_planned_u0_{j}_Nm':fresh[j] for j in range(7)})
        rows.append(row)
    if rows:
        with (output/'feedback.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(output/'feedback_schema.json',dict(rows=len(rows),includes_startup=True,reference='linear frozen reference at solve-start/feedback timestamp',
        state_and_fk='measured simulated state at solve start; FK arm_right_tool_link in nominal arm model',
        applied_torque='actual control executed during first completed plant substep starting at this feedback timestamp; old active plan during current solve; startup uses bootstrap plan',
        planned_torque='first control of newly solved selected plan; activation occurs at solve arrival and final plan may be discarded',
        full_applied_history='execution.npz, executed_trace.csv and raw step_completed events retain every applied plant-substep control',
        timing='public solver wall and native host chrono; startup retained but bootstrap wall unpaid by modeB'))
    return len(rows)


def run(source,build,inputs,case,output):
    output=Path(output)
    code=task.run(source,build,case,output,mode='B',inputs=inputs)
    try:export_feedback(source,output)
    except Exception:
        metrics=json.loads((output/'metrics.json').read_text());metrics['feedback_export_error']=traceback.format_exc();write_json(output/'metrics.json',metrics)
        raise
    return code


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','build','inputs','case','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--export-only',action='store_true')
    args=parser.parse_args()
    if args.export_only:export_feedback(args.source,args.output)
    else:raise SystemExit(run(args.source,args.build,args.inputs,json.loads(args.case.read_text()),args.output))
