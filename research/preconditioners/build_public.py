#!/usr/bin/env python3
"""Configure and build the one public Indy7 N128 module, under an external quiet_run lock."""
import json
import os
from pathlib import Path
import subprocess
import sys
root = Path(sys.argv[1]).resolve()
cmake = '/workspace/GATO-upstream-contributions/.venv/lib/python3.10/site-packages/cmake/data/bin/cmake'
commands = [[cmake, '-S', str(root), '-B', str(root / 'build_native'),
    '-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc', '-DCMAKE_CUDA_ARCHITECTURES=61',
    '-DCMAKE_BUILD_TYPE=Release', '-DGATO_BUILD_DEMO=OFF', '-DPLANT=indy7', '-DKNOTS=128',
    '-DPython3_EXECUTABLE=/workspace/GATO-upstream-contributions/.venv/bin/python',
    '-Dpybind11_DIR=/opt/gato-venv/lib/python3.10/site-packages/pybind11/share/cmake/pybind11'],
    [cmake, '--build', str(root / 'build_native'), '-j', '2']]
print(json.dumps(commands), flush=True)
for command in commands:
    subprocess.run(command, cwd=root, check=True)
