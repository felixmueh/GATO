#!/bin/bash
#SBATCH --job-name=gato-nb-matrix
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:19:30
#SBATCH --output=gato-nb-matrix-%j.out
#SBATCH --error=gato-nb-matrix-%j.out

set -euo pipefail
# Exact shared matrix source; the job wrapper has a separate commit identity.
NB_SOURCE_COMMIT="59dfb337bd885251b21e23c898e9790e433f7d89"
NB_COMMAND_FILE="examples/solve_time_heatmap/nb_cells.json"
[[ "$NB_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo 'Final NB source pin is not published in this script; do not launch.' >&2; exit 2; }
cd "${SLURM_SUBMIT_DIR:?Submit from the NEW GATO-h100-nb-matrix clone root}"
module load devel/cuda
source .venv-h100/bin/activate
task_python="$PWD/.venv-h100/bin/python"
task_output="$PWD/.experiment-results/h100-nb-matrix/${SLURM_JOB_ID:?}"
task_cache="$PWD/.experiment-cache/h100-nb-matrix/$NB_SOURCE_COMMIT"
task_source="$task_cache/source"
task_build="$task_cache/build"
task_script="$PWD/experiments/h100_nb_matrix.sbatch"
task_archive="$PWD/.experiment-work/h100-nb-matrix-${SLURM_JOB_ID}-results.tar.gz"

if [[ "${1:-}" != --within-budget ]]; then
    [[ ! -e "$task_output" && ! -e "$task_archive" ]] || { echo 'Fresh job output/archive required.' >&2; exit 2; }
    mkdir -p "$task_output/job" "$task_cache" "$PWD/.experiment-work"
    finish() {
        task_status=$?
        trap - EXIT
        set +e
        printf '%s\n' "$task_status" > "$task_output/job/exit-status.txt"
        # This archive contains only measurements/logs, never build/source caches.
        tar -C "$(dirname "$task_output")" -czf "$task_archive" "$(basename "$task_output")"
        task_archive_status=$?
        if (( task_archive_status == 0 )); then sha256sum "$task_archive" > "$task_archive.sha256"; fi
        printf 'Job exit: %s; archive exit: %s; archive: %s\n' "$task_status" "$task_archive_status" "$task_archive"
        if (( task_status != 0 )); then exit "$task_status"; fi
        exit "$task_archive_status"
    }
    trap finish EXIT
    nvidia-smi -q > "$task_output/job/nvidia-smi-q.txt" 2>&1
    nvcc --version > "$task_output/job/nvcc-version.txt" 2>&1
    lscpu > "$task_output/job/lscpu.txt" 2>&1
    git rev-parse HEAD > "$task_output/job/job-checkout-commit.txt"
    cp "$task_script" "$task_output/job/h100_nb_matrix.sbatch"
    "$task_python" - "$task_output/job/device.json" "$NB_SOURCE_COMMIT" <<'PY'
import ctypes as C,json,os,sys,time
lib=C.CDLL('libcuda.so.1')
def check(code):
    if code:raise RuntimeError(f'CUDA driver query failed: {code}')
check(lib.cuInit(0));device=C.c_int();check(lib.cuDeviceGet(C.byref(device),0))
sm=C.c_int();check(lib.cuDeviceGetAttribute(C.byref(sm),16,device))
name=C.create_string_buffer(256);check(lib.cuDeviceGetName(name,256,device))
uuid=(C.c_ubyte*16)();check(lib.cuDeviceGetUuid_v2(C.byref(uuid),device))
u=bytes(uuid).hex();formatted='GPU-'+ '-'.join([u[:8],u[8:12],u[12:16],u[16:20],u[20:]])
json.dump(dict(captured_epoch_s=time.time(),source_commit=sys.argv[2],gpu_name=name.value.decode(),multiprocessor_count=sm.value,gpu_uuid=formatted,query='CUDA driver identity only; no kernels or solves',environment={k:os.environ.get(k) for k in ['SLURM_JOB_ID','SLURM_JOB_NODELIST','CUDA_VISIBLE_DEVICES','SLURM_CPUS_PER_TASK']}),open(sys.argv[1],'w'),indent=2)
PY
    # Budget includes source setup, shared build and all measurements. 120 s remain
    # before the Slurm cap for preserving partial results and making the archive.
    timeout --signal=INT --kill-after=20s 1050s bash "$task_script" --within-budget
    exit $?
fi

git cat-file -e "$NB_SOURCE_COMMIT^{commit}"
if [[ ! -e "$task_source" ]]; then git worktree add --detach "$task_source" "$NB_SOURCE_COMMIT"; fi
[[ "$(git -C "$task_source" rev-parse HEAD)" == "$NB_SOURCE_COMMIT" ]]
[[ -z "$(git -C "$task_source" status --porcelain --untracked-files=no)" ]]
[[ -f "$task_source/$NB_COMMAND_FILE" ]] || { echo 'Final common NB argument file missing.' >&2; exit 2; }
cp "$task_source/$NB_COMMAND_FILE" "$task_output/job/common-arguments.json"
"$task_python" -c 'import numpy,pinocchio,scipy,pybind11,matplotlib,torch'
# The common driver owns configure, zero-launch resource probes, supported-module
# builds, all 30 cells/two repeats, and incremental compact export. Do not build
# or reinterpret its frozen per-cell arguments in this job wrapper.
"$task_python" "$task_source/examples/solve_time_heatmap/nb_matrix.py" \
    --only nb-matrix --gpu h100 --raw "$task_output/raw" \
    --cache "$task_build" --export "$task_output/export" --jobs 6 --dry-run \
    > "$task_output/job/dry-run.log" 2>&1
"$task_python" -u "$task_source/examples/solve_time_heatmap/nb_matrix.py" \
    --only nb-matrix --gpu h100 --raw "$task_output/raw" \
    --cache "$task_build" --export "$task_output/export" --jobs 6 \
    > "$task_output/job/measurement.log" 2>&1
