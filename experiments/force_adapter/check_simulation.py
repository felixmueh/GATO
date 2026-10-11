"""Bounded float correspondence of lightweight and existing solver simulation."""
import argparse,hashlib,importlib.util,json,sys,subprocess
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--build',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(ROOT/'python'))
    import bsqp
    from bsqp.interface import BSQP
    modules=list((args.build/'modules').glob('bsqpN32_tiago_right*.so'))
    if len(modules)!=1:raise RuntimeError('One exact N32 force module required')
    module=modules[0];name='bsqp.bsqpN32_tiago_right'
    spec=importlib.util.spec_from_file_location(name,module);native=importlib.util.module_from_spec(spec);sys.modules[name]=native;spec.loader.exec_module(native)
    params=json.loads((ROOT/'experiments/force_adapter/task_inputs/t01.json').read_text())['tracking_solver_params']
    owner=BSQP(str(ROOT/'gato/dynamics/tiago_right/tiago_right_arm.urdf'),16,32,.03,plant_type='tiago_right',**params)
    lightweight=native.Simulation16_float();rows=[];payload={};control=np.linspace(-.05,.05,7,dtype=np.float32)
    with np.load(ROOT/'experiments/force_adapter/native_inputs/arrays.npz') as saved:states=np.unique(saved['previous_states'],axis=0)[:2].copy()
    payload.update(states=states,control=control)
    for state_index,state in enumerate(states):
        for force_kind in ('zero','mixed'):
            forces=np.zeros((16,6),dtype=np.float32)
            if force_kind=='mixed':forces[:,0]=np.linspace(-5,5,16);forces[:,2]=2.;forces[:,3:]=[.2,-.1,.05]
            payload['wrenches-'+force_kind]=forces.copy()
            owner.set_f_ext_B(forces)
            for elapsed in (.03,.047):
                expected=owner.sim_forward(state,control,elapsed);actual=lightweight.sim_forward(state,control,forces,elapsed)
                key=f's{state_index}-{force_kind}-{elapsed}'
                payload[key+'-expected']=expected;payload[key+'-actual']=actual
                passed=bool(np.isfinite(expected).all() and np.isfinite(actual).all() and np.allclose(actual,expected,atol=1e-4,rtol=1e-3))
                rows.append(dict(state_index=state_index,wrench=force_kind,elapsed_s=elapsed,passed=passed,max_abs_difference=float(np.max(abs(actual-expected)))))
    np.savez(args.output/'arrays.npz',**payload)
    summary=dict(passed=all(row['passed'] for row in rows),checks=rows,atol=1e-4,rtol=1e-3,precision='production float',
        module_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),original_workspace=dict(owner.solver.workspace_bytes()),
        source_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),module_path=str(module),
        bank_manifest_sha256=hashlib.sha256((ROOT/'experiments/force_adapter/native_inputs/SHA256SUMS.json').read_bytes()).hexdigest(),
        lightweight_workspace=dict(lightweight.workspace_bytes()),scope='eight matched simulator calls with zero/mixed linear-first force+torque, actual-sized .03/.047 intervals; no TO solve or task qualification')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary));return 0 if summary['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
