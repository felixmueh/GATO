# H100 fixed-update N × B matrix

The script pins the shared published measurement commit
`59dfb337bd885251b21e23c898e9790e433f7d89` in an exact detached source worktree.
Its common `examples/solve_time_heatmap/nb_cells.json` SHA256 is
`1dd1c88734f48dc59074a6bc2d1adfa542170060bc0c4fd817ef7b11665f1446`.
The common source freezes
all GPU settings together: N16/32/64/128/256/512, B1/2/4/8/16, two repeats,
300 measured updates per repeat, one excluded initialization solve, SQP1,
PCG120, 30 ms prediction knots, 10 ms control updates, 1 ms RK4 maximum substep,
no slow-solve cutoff. Each N spans (N-1)*0.03 s. Worker timeout is 180 s per repeat, from the common committed
examples/solve_time_heatmap/nb_cells.json argument file. No rescue repeats or KKT acceptance gate.

Use a NEW clone and a separate environment. These commands never change the
old ~/Programs/GATO-h100 clone, its environment, cache, or overnight results.
Run dependency installation on the login node before requesting GPU time:

```sh
cd ~/Programs
git clone --recurse-submodules --branch experiment/h100-nb-matrix \
  https://github.com/felixmueh/GATO.git GATO-h100-nb-matrix
cd GATO-h100-nb-matrix
uv python install 3.10.12
uv venv --python 3.10.12 .venv-h100
source .venv-h100/bin/activate
uv pip install -r experiments/h100_nb_requirements.txt
uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -c 'import numpy,pinocchio,scipy,pybind11,matplotlib,torch; print("NB dependencies OK")'
```

The separate environment preserves the known Pinocchio/cmeel binary-library
pins and adds Matplotlib. uv may reuse its download cache; no packages are
installed in the old environment. There is no build or GPU benchmark on the
login node. The new job branch is based directly on the final published measurement source;
its exact pinned commit is therefore already an ancestor in this fresh clone.
The job creates a detached worktree at that pin, rather than benchmarking job
branch HEAD. No manual source fetch or old experiment runner is required.

Submit from the new checkout root:

```sh
cd ~/Programs/GATO-h100-nb-matrix
bash -n experiments/h100_nb_matrix.sbatch
sbatch experiments/h100_nb_matrix.sbatch
```

The job requests one H100, six CPUs, and 19 min 30 s on dev_gpu_h100. It uses
an exact detached measurement-source worktree and invokes the unchanged shared
nb_matrix.py driver, first with --dry-run. That driver configures sm90 Release,
builds zero-launch resource probes first, builds only supported N modules with
at most six parallel build jobs, and executes the 30 cells/two repeats
sequentially. All six N values are capacity candidates on H100; recorded resource
queries determine actual support. It captures GPU UUID/SM count, nvidia-smi-q, CUDA and CPU details
before measurements. No new solver/model changes or additional probe solves
are introduced by the job.

Build/source cache: .experiment-cache/h100-nb-matrix/<source-commit>/.
Each submission gets fresh raw output .experiment-results/h100-nb-matrix/<job-id>/.
Job metadata stays in job/, leaving raw/ empty for the common driver. It writes
measurements incrementally under raw/ and compact tables under export/.
The 17 min 30 s internal budget includes setup, build and measurements; it sends
INT and permits 20 s before forced termination, leaving collection headroom
under the scheduler cap. Partial/nonfinite/timeout outcomes remain failures.
Completion within this budget is a target, not a measured promise; no automatic
retry or shortened measurement protocol is used.

On success or ordinary failure, the EXIT handler archives measurements/logs to
.experiment-work/h100-nb-matrix-<job-id>-results.tar.gz and writes its SHA256.
Builds and source caches are excluded. Scheduler/node hard failure can still
prevent collection; keep the raw directory and Slurm log. Return that archive,
its .sha256 file and gato-nb-matrix-<job-id>.out for desktop analysis. Full
measurements stay outside thesis sync; the thesis heatmap is prepared separately.
