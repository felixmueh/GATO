#!/usr/bin/env python3
"""Run the existing bad-PCG-seed regression against Schur-only and Schur+guard headers."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

out = Path(sys.argv[1]).resolve()
out.mkdir(parents=True, exist_ok=False)
clean = Path('/workspace/GATO-upstream-precond-reference')
guard = Path('/workspace/GATO-upstream-precond-schur-guard')
source = guard / 'test/cuda/pcg_warm_start.cu'
glass = Path('/workspace/GATO-upstream-contributions/.venv/lib/python3.10/site-packages/grid_codegen/_data/GLASS')
results = []
for label, root in [('schur_only', clean), ('schur_guard', guard)]:
    exe = out / label
    command = ['/usr/local/cuda/bin/nvcc', '-std=c++17', '-O2', '-DNDEBUG', '-arch=native',
               '-I', str(glass), '-DKNOT_POINTS=16', '-DGATO_PLANT_HEADER="plant_shim.cuh"',
               '-I', str(root / 'gato'), '-I', str(guard / 'test/cuda'), str(source), '-o', str(exe)]
    row = dict(label=label, command=command,
               commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
               test_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    with (out / (label + '.build.log')).open('w') as log:
        compiled = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    row['build_returncode'] = compiled.returncode
    if compiled.returncode == 0:
        result = subprocess.run([str(exe)], capture_output=True, text=True)
        (out / (label + '.stdout')).write_text(result.stdout)
        (out / (label + '.stderr')).write_text(result.stderr)
        row.update(returncode=result.returncode, failures=result.stdout.count('FAIL'),
                   passes=result.stdout.count('PASS'))
    results.append(row)
    (out / 'results.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(row), flush=True)
ok = results[0].get('returncode') == 1 and results[0].get('failures') == 9 and \
     results[1].get('returncode') == 0 and results[1].get('passes') == 9
raise SystemExit(0 if ok else 1)
