#!/bin/bash
#SBATCH --job-name=gato-n32-profile
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:15:00
#SBATCH --output=gato-n32-profile-%j.out
#SBATCH --error=gato-n32-profile-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the GATO checkout root}"
module load devel/cuda
source .venv-h100/bin/activate
task_python="$PWD/.venv-h100/bin/python"
task_output="$PWD/.experiment-results/h100-stage-n32"
task_metadata="$task_output/_jobs/${SLURM_JOB_ID:?}"
task_archive="$PWD/.experiment-work/h100-stage-n32-${SLURM_JOB_ID}-results.tar.gz"
mkdir -p "$task_metadata" "$PWD/.experiment-work"

# Collect after success or ordinary failure; never overwrite another job's archive.
finish() {
    task_status=$?
    trap - EXIT
    set +e
    "$task_python" experiments/collect.py --output "$task_output" --export "$task_archive"
    task_collect_status=$?
    printf 'Runner/job exit: %s; collection exit: %s; archive: %s\n' "$task_status" "$task_collect_status" "$task_archive"
    if (( task_status != 0 )); then exit "$task_status"; fi
    exit "$task_collect_status"
}
trap finish EXIT

nvidia-smi -q > "$task_metadata/nvidia-smi-q.txt" 2>&1 || true
nvcc --version > "$task_metadata/nvcc-version.txt" 2>&1 || true
lscpu > "$task_metadata/lscpu.txt" 2>&1 || true
"$task_python" - "$task_metadata" <<'PY'
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

output = Path(sys.argv[1])
plan = Path('experiments/h100_stage_timing_n32.json')
receipt = dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    plan=str(plan), plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),
    device_ordinal=0, device_name=None, multiprocessor_count=None,
    device_query='CUDA driver API only; no kernels or solves',
    environment={key:os.environ.get(key) for key in
        ['SLURM_JOB_ID','SLURM_JOB_NODELIST','SLURM_CPUS_PER_TASK','CUDA_VISIBLE_DEVICES']})
try:
    driver = C.CDLL('libcuda.so.1')
    def checked(code):
        if code: raise RuntimeError(f'CUDA driver query failed: {code}')
    checked(driver.cuInit(0))
    device, sm = C.c_int(), C.c_int()
    checked(driver.cuDeviceGet(C.byref(device), 0))
    checked(driver.cuDeviceGetAttribute(C.byref(sm), 16, device))
    name = C.create_string_buffer(256)
    checked(driver.cuDeviceGetName(name, len(name), device))
    receipt.update(device_name=name.value.decode(), multiprocessor_count=sm.value)
except Exception as error:
    receipt['query_error'] = f'{type(error).__name__}: {error}'
(output/'job.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2),flush=True)
PY

"$task_python" experiments/run.py --machine h100 --plan experiments/h100_stage_timing_n32.json \
    --python "$task_python" --cache .experiment-cache/h100-stage-n32 --output "$task_output" --dry-run
"$task_python" -u experiments/run.py --machine h100 --plan experiments/h100_stage_timing_n32.json \
    --python "$task_python" --cache .experiment-cache/h100-stage-n32 --output "$task_output"
