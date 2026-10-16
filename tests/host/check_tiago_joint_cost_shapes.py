#!/usr/bin/env python3
"""One focused CPU-only scalar cost check; no native CUDA import or GPU use."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT=Path(__file__).resolve().parents[2]
BASE='22e876eab84887d9c39d9e1f6aa51ddec39c7d36'
OUT=ROOT/'_expr_results/joint-cost-shapes-cpu'
OUT.mkdir(parents=True,exist_ok=True)
original=subprocess.check_output(['git','show',f'{BASE}:gato/dynamics/tiago_right/tiago_right_plant.cuh'],cwd=ROOT,text=True)
start=original.index('        // Continue -log(distance)')
stop=original.index('        template<typename T>\n        __device__ void forwardDynamics',start)
expected=original[start:stop].replace('__device__ T ','GATO_TIAGO_BARRIER_HD T ').replace('__host__ __device__ constexpr T ','GATO_TIAGO_BARRIER_HD constexpr T ')
header=(ROOT/'gato/dynamics/tiago_right/tiago_right_barrier.cuh').read_text()
assert expected in header, 'Original scalar helper arithmetic changed during extraction'
command=['c++','-std=c++17','-O2','-I.', 'tests/host/test_tiago_joint_cost_shapes.cpp','-o',str(OUT/'test_tiago_joint_cost_shapes')]
subprocess.run(command,cwd=ROOT,check=True)
run=subprocess.run([str(OUT/'test_tiago_joint_cost_shapes')],cwd=ROOT,check=True,capture_output=True,text=True)
paths=['gato/dynamics/tiago_right/tiago_right_barrier.cuh','gato/dynamics/tiago_right/tiago_right_plant.cuh','python/bindings.cu','tests/host/test_tiago_joint_cost_shapes.cpp','tests/host/check_tiago_joint_cost_shapes.py']
receipt={'status':'PASS','scope':'CPU scalar mathematics only; native compile/launch and MPC still pending','base':BASE,'extracted_original_scalar_helper_arithmetic_unchanged':True,'command':command,'stdout':run.stdout,'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},'test_notes':['same header scalar dispatch as GPU plant','seven raw URDF bound pairs; .20 rad inset','float/double mode0 scalar identity','value/gradient/Hessian finite differences all three modes','flat interior zero value/gradient/Hessian and nonnegative curvature','C2 soft/raw join and historical one-sided raw/outside slope+curvature','recenter zero reference value/slope and unchanged Hessian, including references inside raw but outside soft bounds']}
(OUT/'CPU_CHECK.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(run.stdout,end='')
print(OUT/'CPU_CHECK.json')
