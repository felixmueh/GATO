"""One six-input native force/gradient capture against independent Pinocchio."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pinocchio as pin

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'python'))
from bsqp.wrench import linear_to_angular_first


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    model=pin.buildModelFromUrdf(str(ROOT/'gato/dynamics/tiago_right/tiago_right_arm.urdf'))
    center=(model.lowerPositionLimit+model.upperPositionLimit)/2
    inputs=[]
    for q in (center,center+np.linspace(-.1,.1,7)):
        for force in (np.zeros(6),np.array([1.2,-.7,.4,.1,-.2,.3]),np.array([0.,0.,0.,.1,-.2,.3])):
            inputs.append(np.r_[q,np.linspace(-.08,.08,7),np.linspace(-1.,1.,7),linear_to_angular_first(force)])
    inputs=np.asarray(inputs,dtype='<f8')
    inputs.tofile(args.output/'inputs.f64')
    command=[str(args.build/'force_probe'),str(args.output/'inputs.f64'),str(args.output/'native.f32'),str(args.output/'native.f64')]
    receipt=dict(scope='Six model inputs, two poses x zero/mixed-force/pure-torque, body-fixed final joint origin; direct FD and force-aware analytic gradient; no optimizer convergence claim',
        command=command,probe_sha256=sha(args.build/'force_probe'),model_sha256=sha(ROOT/'gato/dynamics/tiago_right/tiago_right_arm.urdf'),
        input_sha256=sha(args.output/'inputs.f64'),force_contract='linear-first Python, angular-first native; fixed final-joint-local force/torque',
        tolerances=dict(float=dict(rtol=2e-4,atol=2e-3),double=dict(rtol=1e-7,atol=1e-6)),status='started')
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    with (args.output/'native.log').open('w') as log:
        result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    receipt['returncode']=result.returncode
    comparisons=[];records={}
    if result.returncode==0:
        for label,dtype in [('float','<f4'),('double','<f8')]:
            output=np.fromfile(args.output/('native.f32' if label=='float' else 'native.f64'),dtype=dtype).reshape(6,161)
            expected=[];source=inputs.astype(dtype).astype(float)
            for row in source:
                q,v,u,w=row[:7],row[7:14],row[14:21],row[21:]
                fext=pin.StdVec_Force()
                for _ in range(model.njoints):fext.append(pin.Force.Zero())
                fext[-1]=pin.Force(w[3:],w[:3])
                data=model.createData()
                accel=pin.aba(model,data,q,v,u,fext).copy()
                gradients=[d.copy() for d in pin.computeABADerivatives(model,data,q,v,u,fext)]
                expected.append(np.r_[accel,accel,np.column_stack(gradients).ravel(order='F')])
            expected=np.asarray(expected)
            records[label+'_inputs']=source;records[label+'_native']=output;records[label+'_pinocchio']=expected
            tolerance=receipt['tolerances'][label]
            for i in range(6):
                for section,a,b in [('fd',0,7),('fd_gradient_accel',7,14),('df_dq',14,63),('df_dv',63,112),('df_du',112,161)]:
                    finite=bool(np.isfinite(output[i,a:b]).all())
                    passed=finite and bool(np.allclose(output[i,a:b],expected[i,a:b],**tolerance))
                    comparisons.append(dict(dtype=label,input_index=i,section=section,finite=finite,passed=passed,
                        max_abs_error=float(np.max(abs(output[i,a:b]-expected[i,a:b]))) if finite else None))
        np.savez(args.output/'oracle-arrays.npz',**records)
    receipt.update(status='passed' if result.returncode==0 and all(r['passed'] for r in comparisons) else 'failed',
                   comparisons=comparisons,script_sha256=sha(__file__))
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=receipt['status'],comparisons=len(comparisons),failed=[r for r in comparisons if not r['passed']]),indent=2))
    return 0 if receipt['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())

