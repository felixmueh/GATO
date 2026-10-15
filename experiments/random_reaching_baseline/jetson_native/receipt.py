#!/usr/bin/env python3
"""Capture an already compiled module; no extension import or GPU calls."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from datetime import datetime,timezone

HERE=Path(__file__).resolve().parent
BASE='85ef2ea5cf5b96726be6f4b5441f4a7c0ca041b4'
PUBLISHED='fd3fa2f2c3572442ad158c32c5fdfa5c7c661ece'
TREE='c1229807725cd2b9c8ced2e6c158e69d23db055f'
BOUNDS=dict(TIAGO_COST_JOINT_MARGIN_RAD=.2,TIAGO_COST_VELOCITY_LIMIT_FRACTION=.8,TIAGO_COST_CONTROL_LIMIT_FRACTION=.85)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ('source','build','output'):p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--cuda-arch',choices=('61','87'),required=True)
 a=p.parse_args();source,build=a.source.resolve(),a.build.resolve()
 if a.output.exists():p.error('Fresh receipt output required')
 expected=json.loads((HERE.parents[1]/'jetson_random_reaching/inputs/native_source_sha256.json').read_text())['native_files']
 for name,digest in expected.items():
  if sha(source/name)!=digest:p.error('Numerical85ef file differs: '+name)
 modules=list((build/'modules').glob('bsqpN32_tiago_right*.so'))
 if len(modules)!=1:p.error('Exactly one isolated N32 module required')
 commands=json.loads((build/'compile_commands.json').read_text())
 command=next(row['command'] for row in commands if row['file'].endswith('python/bindings.cu'))
 for name,value in [('TIAGO_COST_JOINT_MARGIN_RAD','0.20'),('TIAGO_COST_VELOCITY_LIMIT_FRACTION','0.80'),('TIAGO_COST_CONTROL_LIMIT_FRACTION','0.85')]:
  if '-D'+name+'='+value not in command:p.error('Missing frozen cost-bound define: '+name)
 if 'compute_'+a.cuda_arch not in command:p.error('Compile architecture differs')
 receipt=dict(status='compiled_only_no_GPU_qualification',created_utc=datetime.now(timezone.utc).isoformat(),
  numerical_source_base=BASE,published_native_source=PUBLISHED,published_native_tree=TREE,
  build_source_commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
  build_source_tree=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD^{tree}'],text=True).strip(),
  module=str(modules[0]),module_sha256=sha(modules[0]),cost_bound_defines=BOUNDS,
  compile_flags=command,source_sha256=expected,cuda_arch=int(a.cuda_arch),host_machine=platform.machine(),
  cmake_cache_sha256=sha(build/'CMakeCache.txt'),compile_commands_sha256=sha(build/'compile_commands.json'),
  native_build_recipe_sha256=sha(HERE/'CMakeLists.txt'),validation='No native load, GPU call or task execution by receipt creation')
 a.output.parent.mkdir(parents=True,exist_ok=True)
 a.output.write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
