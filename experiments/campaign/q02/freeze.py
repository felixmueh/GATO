"""Prepare frozen inputs/manifest on CPU; never compile or import CUDA."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import numpy as np
from protocol import HERE, ROOT, URDF, WEIGHTS, PinReference, sha256
import xml.etree.ElementTree as ET

SOURCE_FILES=['gato/settings.h','gato/constants.h','gato/dynamics/integrator.cuh',
'gato/dynamics/tiago_right/tiago_right_plant.cuh','gato/dynamics/tiago_right/tiago_right_grid.cuh',
'gato/dynamics/tiago_right/tiago_right_limits.cuh','gato/dynamics/tiago_right/tiago_right_arm.urdf',
'gato/utils/linalg.cuh']

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-manifest-only',action='store_true')
    args=parser.parse_args()
    if not args.refresh_manifest_only:
        if (HERE/'protocol.json').exists():
            parser.error('protocol exists; refusing to regenerate frozen inputs or tolerances')
        ref=PinReference(); root=ET.parse(URDF).getroot()
        joints=[root.find(f"joint[@name='arm_right_{i}_joint']") for i in range(1,8)]
        limits={'q':[[float(j.find('limit').attrib[k]) for k in ['lower','upper']] for j in joints],
                'v':[[-float(j.find('limit').attrib['velocity']),float(j.find('limit').attrib['velocity'])] for j in joints],
                'u':[[-float(j.find('limit').attrib['effort']),float(j.find('limit').attrib['effort'])] for j in joints]}
        qs=np.array([[-.8,-.6,.2,-.8,.4,0.,.1],[-.8,-.6,.2,-.8,.4,-.35,.1],
                     [-.8,-.6,.2,-.8,.4,.35,.1],[-1.,-.45,.35,-.95,.55,-.2,-.05],
                     [-.65,-.75,.05,-.65,.25,.2,.25],[-1.1,-.8,-.15,-.6,.1,-.45,.3],
                     [-.55,-.5,.4,-1.,.65,.45,-.2],[-.9,-.7,.3,-.7,.3,.1,0.]])
        samples=[]
        for i,q in enumerate(qs):
            v=np.linspace(-.08,.08,7)*((-1)**i)
            torque=ref.pin.rnea(ref.model,ref.data,q,v,np.zeros(7)).copy()+np.linspace(-.02,.02,7)
            xu=np.r_[q,v,torque].astype(np.float32).astype(float)
            goal=(ref.pose(xu[:7])+np.array([.02,-.01,.015,.03,-.025,.02])).astype(np.float32).astype(float)
            samples.append(dict(id=f'interior-{i}',xu=xu.tolist(),goal=goal.tolist()))
        protocol=dict(id='Q02-current-normal-v1',source_core='16b33719b4f0fd463eec7262c99b16b0b324366f',
            state='frozen implementation; native qualification pending',timing_mode='A',offline_timing='controller_dt',
            scope='Normal tiago_right current-source isolated model functions; no optimizer/closed-loop task',
            knots=32,knot_dt_s=.03,integrator_type=2,angle_wrap=False,gravity_m_s2=[0.,0.,-9.81],
            model_frame='torso_lift_link',tool_frame='arm_right_tool_link',coordinates='native Tiago; joint6 axis -Z; wrapper maps to GRiD +Z',
            dtype='float32',weights=WEIGHTS,limits=limits,samples=samples,
            reference_fd_step=1e-5,native_fd_step=.004,
            batch_groups=[[0,1,2,3],[4,5,6,7],[3,1,0,2],[7,5,4,6],[0,0,1,1],[6,6,7,7]],
            warmup_calls=1,native_evaluate_calls=351,repeats=1,
            tolerances=dict(value_pose=[2e-5,2e-5],value_qdd=[.002,.0005],value_step=[.0001,.0003],
                            value_cost=[.0001,.0002],reference_dynamics_jac=[.01,.003],reference_step_jac=[.0003,.003],
                            reference_pose_jac=[.0003,.003],reference_gradient=[.003,.003],reference_gn=[.005,.005],
                            native_fd_dynamics_jac=[.03,.015],native_fd_step_jac=[.002,.015],native_fd_pose_jac=[.002,.015],
                            native_fd_gradient=[.01,.015],scalar_batch=[2e-5,2e-5]),
            tolerance_rationale='float32 unit roundoff 5.96e-8; central FD balances O(h^2) truncation and O(eps/h) cancellation, h=.004 near eps^(1/3). Float64 independent reference h=1e-5. Tolerances allow conservative accumulated robot algebra/fast-math rounding and FD truncation; fixed before native compilation, practical agreement thresholds, not a formal error bound. Float32 near-zero outputs use absolute floors. Never refit thresholds to observed GPU outputs.',
            failure_policy='Preserve failed/nonfinite declared outputs as negative results; no fixes, retries, input filtering or tolerance changes; hold qualification and report to Felix. Execution errors preserve partial raw and stop.',
            limits_of_claim=['Eight interior samples only, no outside-bound derivative tests','GN pose curvature J^T W J, not exact nonlinear pose-cost Hessian',
                             'No full KKT/NaN source investigation, determinism, solver correctness or task reliability certificate',
                             'Current-source new protocol does not recover or replace missing historical debug pack',
                             'B1/B4 calls current cost helpers; does not qualify batch-wide solver stopping or higher batches',
                             'No physical robot/Gazebo results; URDF damping/friction not modeled by ABA or GRiD'])
        (HERE/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    paths=[ROOT/p for p in SOURCE_FILES]
    paths+=sorted(p for p in HERE.iterdir() if p.is_file() and p.name not in ['source_manifest.json','cpu_validation.json','review.json'])
    tests=ROOT/'tests/python/test_q02_protocol.py'
    if tests.exists(): paths.append(tests)
    manifest=dict(source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  approved_core='16b33719b4f0fd463eec7262c99b16b0b324366f',
                  files={str(p.relative_to(ROOT)):sha256(p) for p in paths})
    (HERE/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'protocol_sha256':sha256(HERE/'protocol.json'),'manifest_sha256':sha256(HERE/'source_manifest.json')}))
if __name__=='__main__': main()
