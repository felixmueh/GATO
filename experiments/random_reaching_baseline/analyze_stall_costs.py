#!/usr/bin/env python3
"""CPU accounting of explicitly selected saved horizons; no solves or tuning.

Example: --source CHECKOUT --run-root SAVED_PILOT --pair 7093 1265 --step 266 --output TEMP
Plot-only mode reads the compact exported CSV/JSON and imports no Pinocchio.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

TERMS=['running_position','terminal_position','running_orientation','terminal_orientation',
       'velocity_regularization','torque_regularization','joint_limits','speed_limits','torque_limits']
NATIVE_PI=3.14159
NATIVE_TWO_PI=6.28318
MODEL='gato/dynamics/tiago_right/tiago_right_arm.urdf'
HELPER_SHA='effd36439784b4bd17931833a5e5d6e7085b55f0d7473b30a8f9baf06a693979'


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def csv_write(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def wrap_errors(error):
    for j in range(3,6):
        while error[j]>NATIVE_PI:error[j]-=NATIVE_TWO_PI
        while error[j]<-NATIVE_PI:error[j]+=NATIVE_TWO_PI
    return error


def analyze(source,run,pair,step,protocol,barrier):
    import numpy as np
    import pinocchio as pin
    n=protocol['N'];dt=protocol['knot_dt_s'];w=protocol['solver_params'];bounds=protocol['limit_penalty']['cost_bounds']
    directory=run/f'pair_{pair:05d}';ip=directory/f'step_{step:04d}_input.npz';op=directory/f'step_{step:04d}_returned.npz'
    with np.load(ip,allow_pickle=False) as a:actual=a['actual_state'].astype(float);reference=a['reference'].reshape(n,6).astype(float)
    with np.load(op,allow_pickle=False) as a:
        packed=a['optimized'].reshape(-1).astype(float);native=float(a['final_merit'].reshape(-1)[0])
    if packed.size!=n*21-7:raise ValueError('Saved packed width differs from protocol N')
    body=packed[:(n-1)*21].reshape(n-1,21);states=np.vstack([body[:,:14],packed[-14:]]);controls=body[:,14:]
    if not all(np.isfinite(x).all() for x in [states,controls,actual,reference]):raise ValueError('Selected snapshot contains nonfinite data; retain it and choose no substitute silently')
    model=pin.buildModelFromUrdf(str(source/MODEL));model.gravity.linear=np.array([0.,0.,-9.81]);data=model.createData()
    lower,upper=model.lowerPositionLimit,model.upperPositionLimit;lo=lower+bounds['joint_margin_rad'];hi=upper-bounds['joint_margin_rad']
    vlim=model.velocityLimit*bounds['velocity_fraction_of_physical_limit'];ulim=model.effortLimit*bounds['effort_fraction_of_physical_limit']
    def pose(q):
        pin.forwardKinematics(model,data,q);pin.updateFramePlacements(model,data)
        p=data.oMf[model.getFrameId('torso_lift_link')].inverse()*data.oMf[model.getFrameId('arm_right_tool_link')]
        return np.r_[p.translation,pin.rpy.matrixToRpy(p.rotation)]
    def errors(q,target):return wrap_errors(pose(q)-target)
    def gradient(q,target,pw,ow):
        h=1e-6;gp=np.zeros(7);go=np.zeros(7)
        for j in range(7):
            delta=np.eye(7)[j]*h;ep=errors(q+delta,target);em=errors(q-delta,target)
            gp[j]=.5*pw*(np.sum(ep[:3]**2)-np.sum(em[:3]**2))/(2*h)
            go[j]=.5*ow*(np.sum(ep[3:]**2)-np.sum(em[3:]**2))/(2*h)
        return gp,go,np.array([w['q_lim_cost']*barrier(q[j],lo[j],hi[j])[1] for j in range(7)])
    raw={k:0. for k in TERMS};centered=raw.copy();joints=[];gradients=[];defects=[];costs=np.zeros(7)
    horizon_gp=[];horizon_go=[];horizon_gq=[]
    def point(scope,knot,q,target,pw,ow):
        gp,go,gq=gradient(q,target,pw,ow);goal=gp+go
        for j in range(7):
            value=barrier(q[j],lo[j],hi[j])[0];center=barrier((lo[j]+hi[j])/2,lo[j],hi[j])[0];delta=.01*(hi[j]-lo[j])
            joints.append(dict(pair_id=pair,step=step,scope=scope,knot=knot,joint=j+1,joint_name=model.names[j+1],q_rad=float(q[j]),
                physical_lower_clearance_rad=float(q[j]-lower[j]),physical_upper_clearance_rad=float(upper[j]-q[j]),
                physical_minimum_clearance_rad=float(min(q[j]-lower[j],upper[j]-q[j])),cost_lower_clearance_rad=float(q[j]-lo[j]),
                cost_upper_clearance_rad=float(hi[j]-q[j]),cost_minimum_clearance_rad=float(min(q[j]-lo[j],hi[j]-q[j])),
                continuation_delta_rad=float(delta),uses_quadratic_continuation=bool(q[j]-lo[j]<delta or hi[j]-q[j]<delta),
                q_limit_raw_cost=float(w['q_lim_cost']*value),q_limit_centered_cost=float(w['q_lim_cost']*(value-center)),
                position_gradient_cost_per_rad=float(gp[j]),orientation_gradient_cost_per_rad=float(go[j]),
                goal_gradient_cost_per_rad=float(goal[j]),joint_limit_gradient_cost_per_rad=float(gq[j])))
        return gp,go,gq
    for k,x in enumerate(states):
        terminal=k==n-1;pw=w['N_cost'] if terminal else w['q_cost'];ow=w['ee_orient_N_cost'] if terminal else w['ee_orient_cost'];e=errors(x[:7],reference[k])
        raw['terminal_position' if terminal else 'running_position']+=float(.5*pw*np.sum(e[:3]**2))
        raw['terminal_orientation' if terminal else 'running_orientation']+=float(.5*ow*np.sum(e[3:]**2))
        raw['velocity_regularization']+=float(.5*w['qd_cost']*np.sum(x[7:]**2))
        for j in range(7):
            for term,value,l,u,weight in [('joint_limits',x[j],lo[j],hi[j],w['q_lim_cost']),('speed_limits',x[7+j],-vlim[j],vlim[j],w['vel_lim_cost'])]:
                b=barrier(value,l,u)[0];b0=barrier((l+u)/2,l,u)[0];raw[term]+=weight*b;centered[term]+=weight*(b-b0)
                if term=='joint_limits':costs[j]+=weight*(b-b0)
        gp,go,gq=point('horizon',k,x[:7],reference[k],pw,ow);horizon_gp.append(gp);horizon_go.append(go);horizon_gq.append(gq)
        if k<n-1:
            raw['torque_regularization']+=float(.5*w['u_cost']*np.sum(controls[k]**2))
            for j in range(7):
                b=barrier(controls[k,j],-ulim[j],ulim[j])[0];b0=barrier(0.,-ulim[j],ulim[j])[0]
                raw['torque_limits']+=w['ctrl_lim_cost']*b;centered['torque_limits']+=w['ctrl_lim_cost']*(b-b0)
            acceleration=pin.aba(model,data,x[:7],x[7:],controls[k]).copy()
            predicted=np.r_[x[:7]+dt*x[7:]+.5*dt*dt*acceleration,x[7:]+dt*acceleration]
            difference=states[k+1]-predicted
            defects.append(dict(pair_id=pair,step=step,interval=k,q_defect_L1_rad=float(np.abs(difference[:7]).sum()),
                v_defect_L1_rad_s=float(np.abs(difference[7:]).sum()),mixed_unit_defect_L1=float(np.abs(difference).sum())))
    for key in TERMS:
        if key not in ['joint_limits','speed_limits','torque_limits']:centered[key]=raw[key]
    def summarize(scope,gp,go,gq):
        g=gp+go;gn=float(np.linalg.norm(g));qn=float(np.linalg.norm(gq));cos=float(np.ravel(g)@np.ravel(gq)/(gn*qn)) if gn*qn else None
        gradients.append(dict(pair_id=pair,step=step,scope=scope,dimensions=g.size,position_gradient_norm=float(np.linalg.norm(gp)),
            orientation_gradient_norm=float(np.linalg.norm(go)),goal_gradient_norm=gn,joint_limit_gradient_norm=qn,
            goal_plus_joint_limit_gradient_norm=float(np.linalg.norm(g+gq)),goal_joint_limit_cosine=cos))
    for scope,q in [('measured_actual',actual[:7]),('predicted_x0',states[0,:7])]:
        summarize(scope,*point(scope,0,q,reference[0],w['q_cost'],w['ee_orient_cost']))
    gp,go,gq=[np.asarray(v) for v in [horizon_gp,horizon_go,horizon_gq]]
    summarize('horizon_flat_q_coordinates',gp,go,gq);summarize('horizon_common_joint_shift',gp.sum(axis=0),go.sum(axis=0),gq.sum(axis=0))
    byjoint=[dict(pair_id=pair,step=step,joint=j+1,centered_limit_cost=float(costs[j]),summed_position_gradient=float(gp[:,j].sum()),
        summed_orientation_gradient=float(go[:,j].sum()),summed_goal_gradient=float((gp[:,j]+go[:,j]).sum()),summed_limit_gradient=float(gq[:,j].sum())) for j in range(7)]
    defect=sum(x['mixed_unit_defect_L1'] for x in defects);initial=float(np.abs(states[0]-actual).sum());cpu=sum(raw.values())+w['mu']*(defect+initial)
    line=next(json.loads(x) for x in (directory/'solves.jsonl').read_text().splitlines() if json.loads(x)['step']==step)
    identity_files=[ip,op,directory/'preparation.json',directory/'metrics.json',directory/'solves.jsonl',run/'protocol.json',run/'provenance.json']
    result=dict(pair_id=pair,step=step,feedback_time_s=line['feedback_time_s'],N=n,knot_dt_s=dt,solver_params=w,raw=raw,centered=centered,
        raw_objective_sum=sum(raw.values()),centered_objective_sum=sum(centered.values()),raw_minus_centered_constant=sum(raw.values())-sum(centered.values()),
        cpu_dynamics_defect_L1_mixed_units=defect,cpu_initial_defect_L1_mixed_units=initial,mu=w['mu'],mu_dynamics_defect_cost=w['mu']*defect,
        mu_initial_defect_cost=w['mu']*initial,cpu_approximate_raw_merit=cpu,saved_native_final_merit=native,cpu_minus_native_merit=cpu-native,
        q_clearance_summary={scope:dict(minimum_physical_clearance_rad=min(x['physical_minimum_clearance_rad'] for x in joints if x['scope']==scope),
            minimum_cost_clearance_rad=min(x['cost_minimum_clearance_rad'] for x in joints if x['scope']==scope),
            quadratic_continuation_entries=sum(x['uses_quadratic_continuation'] for x in joints if x['scope']==scope))
            for scope in ['measured_actual','predicted_x0','horizon']},
        saved_native_merit_is_not_pure_objective=True,actual_state=actual.tolist(),predicted_x0=states[0].tolist(),reference_pose=reference[0].tolist(),
        gradients=gradients,work=line['work'],Pinocchio_version=pin.__version__,finite_difference_step_rad=1e-6,
        identities={str(p):sha(p) for p in identity_files})
    termrows=[dict(pair_id=pair,step=step,term=k,raw_cost=raw[k],centered_cost=centered[k],raw_minus_centered=raw[k]-centered[k]) for k in TERMS]
    return result,termrows,joints,gradients,defects,byjoint


def plots(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (output/'cost_terms.csv').open() as f:terms=list(csv.DictReader(f))
    with (output/'q_joint_points.csv').open() as f:joints=list(csv.DictReader(f))
    snapshots=list(dict.fromkeys((r['pair_id'],r['step']) for r in terms))
    labels=['Position, running','Position, terminal','Orientation, running','Orientation, terminal','Velocity quadratic','Effort quadratic','Joint bounds','Speed bounds','Effort bounds']
    fig,axes=plt.subplots(1,len(snapshots),figsize=(max(6,5*len(snapshots)),5),squeeze=False,constrained_layout=True)
    for ax,(pair,step) in zip(axes[0],snapshots):
        rows=[r for r in terms if (r['pair_id'],r['step'])==(pair,step)]
        ax.barh(range(9),[float(r['centered_cost']) for r in rows],color=['#3878a3']*4+['#8b9b9d']*2+['#df8a32']*3)
        ax.set_yticks(range(9),labels);ax.invert_yaxis();ax.set_xlabel('Center-referenced horizon cost');ax.set_title(f'Task {pair}, step {step}')
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    fig.savefig(output/'objective_terms.png',dpi=170);fig.savefig(output/'objective_terms.pdf');plt.close(fig)
    fig,axes=plt.subplots(len(snapshots),2,figsize=(10,3.3*len(snapshots)),squeeze=False,constrained_layout=True)
    for row,(pair,step) in enumerate(snapshots):
        for col,scope in enumerate(['measured_actual','predicted_x0']):
            ax=axes[row,col];items=[r for r in joints if (r['pair_id'],r['step'],r['scope'])==(pair,step,scope)]
            x=list(range(1,8));goal=[float(r['goal_gradient_cost_per_rad']) for r in items];limit=[float(r['joint_limit_gradient_cost_per_rad']) for r in items]
            ax.bar([j-.17 for j in x],goal,.32,label='Position + orientation',color='#3878a3');ax.bar([j+.17 for j in x],limit,.32,label='Joint bounds',color='#df8a32')
            ax.axhline(0,color='grey',lw=.7);ax.set_xticks(x);ax.set_xlabel('Joint');ax.set_ylabel('Local q gradient [cost/rad]');ax.set_title(f'Task {pair}: {scope.replace("_"," ")}')
            ax.spines[['top','right']].set_visible(False)
    axes[0,0].legend(fontsize=8);fig.savefig(output/'local_q_gradients.png',dpi=170);fig.savefig(output/'local_q_gradients.pdf');plt.close(fig)


def report(output,results,command,source,run):
    w=results[0]['solver_params']
    lines=['# Saved near-stall horizon cost accounting','',
        'These are CPU calculations on explicitly selected saved horizons, not new optimization, simulation or validation runs. They describe local objective tradeoffs; they do not identify a solver defect, prove stationarity or explain a stall causally.','',
        '## Objective components and merit','',
        f"The native {results[0]['N']}-state objective includes pose, velocity regularization and q/speed-bound costs at every state node, including terminal. Effort regularization and effort-bound costs apply only at {results[0]['N']-1} control nodes. Terminal pose weights are {w['N_cost']}/{w['ee_orient_N_cost']}; running pose weights are {w['q_cost']}/{w['ee_orient_cost']}. All sums are unscaled by dt. The last KKT helper promotes terminal weights once; there is no doubled terminal multiplier.",'',
        '| Task / step | Feedback seconds | Centered objective | Raw objective | Mu dynamics cost | Mu initial cost | CPU approximate merit | Saved native merit | CPU minus native |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in results:lines.append('| '+f"{r['pair_id']} / {r['step']}"+' | '+' | '.join(f"{r[k]:.5g}" for k in ['feedback_time_s','centered_objective_sum','raw_objective_sum','mu_dynamics_defect_cost','mu_initial_defect_cost','cpu_approximate_raw_merit','saved_native_final_merit','cpu_minus_native_merit'])+' |')
    lines += ['', 'The merit columns are separate from objective components: mu multiplies L1 modeled dynamics and initial-state defects, mixing unnormalized q and velocity units. Rho is damping and is absent from this objective accounting. CPU/native differences reflect a CPU Pinocchio/double reconstruction versus generated float/fast-math code; they are descriptive, not a numerical acceptance or convergence gate.','',
        '![Nine objective components for each explicitly selected horizon. Bound costs subtract their own interval-center value; pose and regularization are unchanged. Cost size alone does not establish which term causes behavior.](objective_terms.png)','',
        f"The three raw logarithmic bound terms include arbitrary negative constants. For these saved records, raw objective minus center-referenced objective is {results[0]['raw_minus_centered_constant']:.10g}. Center subtraction leaves every gradient, curvature and within-configuration objective difference unchanged. A large negative raw speed-cost magnitude is therefore not evidence that speed penalties dominate the decision. Center-referenced values compare excess costs, while dynamically feasible changes and derivatives still determine a tradeoff.",'',
        '| Task / step | Objective component | Raw cost | Center-referenced cost |','| --- | --- | ---: | ---: |']
    for r in results:
        for term in TERMS:lines.append(f"| {r['pair_id']} / {r['step']} | {term.replace('_',' ')} | {r['raw'][term]:.7g} | {r['centered'][term]:.7g} |")
    lines += ['', '## q-space gradients and clearances','',
        'Measured actual q is the feedback state in the saved solve input; predicted x0 is the returned first state. They need not be identical. Their local running pose/q-limit gradients use the same q coordinates and cost/rad units. Velocity/effort gradients in different units are not mixed into this comparison.','',
        '| Task | Scope | Goal gradient norm | Joint-bound gradient norm | Combined norm | Goal/limit cosine |','| ---: | --- | ---: | ---: | ---: | ---: |']
    for r in results:
        for g in r['gradients']:lines.append(f"| {r['pair_id']} | {g['scope'].replace('_',' ')} | {g['goal_gradient_norm']:.5g} | {g['joint_limit_gradient_norm']:.5g} | {g['goal_plus_joint_limit_gradient_norm']:.5g} | {g['goal_joint_limit_cosine']:.5g} |")
    lines += ['', '| Task | Scope | Minimum physical clearance rad | Minimum cost clearance rad | Quadratic continuation entries |',
        '| ---: | --- | ---: | ---: | ---: |']
    for r in results:
        for scope,c in r['q_clearance_summary'].items():
            lines.append(f"| {r['pair_id']} | {scope.replace('_',' ')} | {c['minimum_physical_clearance_rad']:.7g} | {c['minimum_cost_clearance_rad']:.7g} | {c['quadratic_continuation_entries']} |")
    lines += ['', 'A negative cosine indicates opposing local gradients, not a proof that either term caused the timeout. The flattened horizon norm uses all7N independent q coordinates. The seven-component common-shift vector sums each joint component across nodes: it represents moving every q node in the same joint direction, not the full horizon gradient or a dynamically feasible motion. Neither includes the merit-defect derivative or a projected constrained direction.','',
        '![Local weighted goal and joint-limit gradients at measured actual q and returned predicted x0. Their opposition is descriptive objective accounting; it is not a causal stall diagnosis or a stationarity certificate.](local_q_gradients.png)','',
        'Per-joint raw and centered limit costs, signed gradients, physical and cost-bound clearances, continuation widths and branch flags are in q_joint_points.csv. The actual physical interval remains unchanged. A point can have a substantial interior log-barrier gradient while still being inside its cost interval; entering the quadratic continuation is not required. Horizon per-joint cost and gradient sums are in joint_horizon_sums.csv.','',
        '## Source semantics and limitations','',
        'Tool FK is torso_lift_link to arm_right_tool_link. GRiD negates native joint6 to account for its positive-axis generation; the native URDF/Pinocchio model already uses the negative joint6 axis, so Pinocchio q6 is not negated again. Orientation cost uses componentwise roll/pitch/yaw errors, not the tool-axis success angle or a rotation-log cost. Wrapping uses the native constants3.14159 and6.28318 with strict while comparisons.','',
        'The dynamics merit reconstruction uses Pinocchio ABA at each saved start state/control, then qnext=q+v*dt+0.5*a*dt² and vnext=v+a*dt. This is the native frozen-acceleration approximation, not the 3ms RK4 plant. The separately reported L1 initial-state defect is included once. It is not a feasibility certificate, new residual instrument or reason to extend a run.','',
        'N and dt come from the saved protocol; references come from the exact saved float32 solve input. Pose gradients use CPU central differences at1e−6rad; limit gradients use the reviewed C2 formula. Snapshots, source/model/helper/protocol/native identities and all analysis inputs are retained. No additional case, optimizer solve, GPU call, cost change or retry was introduced.','',
        '## Reproduction','',f'Source: {source}. Saved run: {run}. All bulk trajectories remain in the saved run.','',
        '```sh',shlex.join(command),'```','',
        'The compact plot-only path reads the exported CSV/JSON and imports no Pinocchio or solver:','```sh','python analyze_stall_costs.py --plot-only --output .','```','']
    (output/'REPORT.md').write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path);p.add_argument('--run-root',type=Path)
    p.add_argument('--pair',type=int,nargs='+');p.add_argument('--step',type=int,nargs='+',default=[266]);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plot-only',action='store_true');p.add_argument('--final',action='store_true',help='Use only after root reviews the temporary CPU analysis')
    a=p.parse_args();out=a.output.resolve()
    if a.plot_only:plots(out);return
    if a.source is None or a.run_root is None or not a.pair:p.error('--source, --run-root and explicit --pair required')
    if len(a.step) not in (1,len(a.pair)):p.error('Supply one step for all explicit pairs or one step per pair; no Cartesian grid')
    if not a.final and Path('/workspace/GATO/thesis_results') in out.parents:p.error('Use a temporary output before root review')
    source=a.source.resolve();run=a.run_root.resolve();protocol=read(run/'protocol.json');provenance=read(run/'provenance.json')
    scalar=load('saved_cost_scaling',source/'experiments/random_reaching_baseline/cost_scaling.py')
    plant=source/'gato/dynamics/tiago_right/tiago_right_plant.cuh';text=plant.read_text();start=text.index('        template<class T>\n        __host__ __device__ constexpr T jointBarrierRelaxationDistance');end=text.index('        template<typename T>\n        __device__ void forwardDynamics(',start)
    if hashlib.sha256(text[start:end].encode()).hexdigest()!=HELPER_SHA:raise ValueError('Reviewed helper formula changed')
    snapshots=list(zip(a.pair,a.step*len(a.pair) if len(a.step)==1 else a.step))
    if len(set(snapshots))!=len(snapshots) or any(pair<0 or step<0 for pair,step in snapshots):p.error('Select distinct nonnegative pair/step snapshots')
    results=[];terms=[];joints=[];gradients=[];defects=[];jointsum=[]
    for pair,step in snapshots:
        if pair not in protocol['development_task_ids']:raise ValueError('Only explicitly selected development snapshots are analyzed')
        r,t,j,g,d,s=analyze(source,run,pair,step,protocol,scalar.barrier);results.append(r);terms+=t;joints+=j;gradients+=g;defects+=d;jointsum+=s
    out.mkdir(parents=True,exist_ok=True)
    for name,rows in [('cost_terms',terms),('q_joint_points',joints),('gradient_summary',gradients),('dynamics_defects',defects),('joint_horizon_sums',jointsum)]:csv_write(out/(name+'.csv'),rows)
    write(out/'snapshots.json',results)
    files=[MODEL,'gato/dynamics/tiago_right/tiago_right_plant.cuh','gato/dynamics/tiago_right/tiago_right_grid.cuh',
        'gato/dynamics/tiago_right/tiago_right_limits.cuh','gato/dynamics/tiago_right/tiago_right_cost_limits.cuh',
        'gato/bsqp/kernels/merit.cuh','gato/bsqp/kernels/setup_kkt.cuh','gato/dynamics/integrator.cuh','experiments/random_reaching_baseline/cost_scaling.py']
    native=Path(provenance['module']);identities=dict(protocol=protocol,saved_run_provenance=provenance,
        protocol_sha256=sha(run/'protocol.json'),provenance_sha256=sha(run/'provenance.json'),
        source_commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
        source_sha256={name:sha(source/name) for name in files},helper_body_sha256=HELPER_SHA,
        native_module_recorded_sha256=provenance['module_sha256'],native_module_current_sha256=sha(native) if native.is_file() else None,
        native_PI=NATIVE_PI,native_TWO_PI=NATIVE_TWO_PI,native_imported=False,GPU_calls=0,
        source_references=dict(tracking_value='tiago_right_plant.cuh:361–454',terminal_KKT_promotion='tiago_right_plant.cuh:610–653',
            merit='gato/bsqp/kernels/merit.cuh:70–92',pose_RPY='tiago_right_grid.cuh:3285–3287',frozen_map='gato/dynamics/integrator.cuh:34–37'))
    write(out/'identities.json',identities)
    plots(out)
    command=[sys.executable,str(Path(__file__).resolve()),'--source',str(source),'--run-root',str(run),'--pair',*map(str,a.pair),'--step',*map(str,a.step),'--output',str(out),*(['--final'] if a.final else [])]
    report(out,results,command,source,run)
    renderer=source/'experiments/random_reaching_baseline/report_to_pdf.py';pages=load('stall_pdf_renderer',renderer).render(out/'REPORT.md',out/'previews')
    for file in [Path(__file__).resolve(),renderer]:
        target=out/file.name
        if target!=file:shutil.copyfile(file,target)
    write(out/'export_receipt.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),final=a.final,PDF_pages=pages,
        analyzer_sha256=sha(__file__),renderer_sha256=sha(renderer),explicit_snapshots=[dict(pair=pair,step=step) for pair,step in snapshots],
        output_sha256={f.name:sha(f) for f in out.iterdir() if f.is_file() and f.name!='export_receipt.json'}))
    print(json.dumps(dict(output=str(out),PDF_pages=pages,snapshots=[dict(pair=r['pair_id'],centered_objective=r['centered_objective_sum'],cpu_minus_native_merit=r['cpu_minus_native_merit']) for r in results])))


if __name__=='__main__':main()
