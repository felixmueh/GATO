# H100 quick checks

Run inside an allocated, uncontended whole H100 GPU, never on the login node.
The two explicit plans are independent. Neither runs BB or BA.
Use Git, CMake >=3.24, a C++ compiler, CUDA supporting sm90, Python headers,
NumPy, SciPy, Pinocchio (`pin`), pybind11 and CPU-only PyTorch. The public
wrapper imports torch; the numerical solve still uses the native CUDA module.
Activate the existing suitable Python environment and load the site CUDA module.
Both recipes compile with two CPU jobs and select `90-real` explicitly.

On a bare cluster, create the minimal environment with Python 3.10.12 and install
`experiments/h100_requirements.txt` using `uv pip install -r` inside the venv.
Pinocchio 3.9 wheels need the urdfdom 4 / tinyxml2 10 binary libraries; unconstrained
transitive dependencies can select incompatible newer major releases. Assimp is
also pinned to the version used by the working desktop environment. This does
not change experiment parameters.

```sh
uv python install 3.10.12
uv venv --python 3.10.12 .venv-h100
source .venv-h100/bin/activate
uv pip install -r experiments/h100_requirements.txt
uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -c "import numpy, pinocchio, scipy, pybind11, torch; print('Dependencies OK')"
```

```sh
git fetch origin experiment/h100-stage-runtime experiment/h100-fig8-runtime
python3 experiments/run.py --machine h100 --plan experiments/h100_stage_timing.json --dry-run
python3 experiments/run.py --machine h100 --plan experiments/h100_stage_timing.json --python "$(command -v python3)" --cache .experiment-cache/h100-stage --output .experiment-results/h100-stage
python3 experiments/run.py --machine h100 --plan experiments/h100_fig8_sep23.json --dry-run
python3 experiments/run.py --machine h100 --plan experiments/h100_fig8_sep23.json --python "$(command -v python3)" --cache .experiment-cache/h100-fig8 --output .experiment-results/h100-fig8
```

Stage timing is one N128/B16/SQP5/PCG100 measured solve after one excluded
warmup, with fresh solver owners and the retained GTX1070 inputs. Figure-eight
is one separate 24s modeB episode, N32/.008s knots/.001s plant/B1, saved
September weights including qd=.01 and exact saved reference at .008s.
The published figure-eight runtime tree equals the qualified desktop frontend;
its unchanged b400 native solver gets a fresh standard N32 build, stage timing off.
No extra qualification episodes or retries are queued. Preserve failures.
Desktop figure-eight took184s elapsed and395MBraw; H100 duration/storage are
unmeasured, and faster solves can increase feedback/recording counts.
Capture `nvidia-smi -q`, `nvcc --version`, CPU model and Slurm job/node IDs in
the result folder. Compile time is additional; do not assume the30min dev
allocation suffices for environment installation and both fresh builds.

```sh
python3 experiments/collect.py --output .experiment-results/h100-stage --export .experiment-work/h100-stage-results.tar.gz
python3 experiments/collect.py --output .experiment-results/h100-fig8 --export .experiment-work/h100-fig8-results.tar.gz
```

Keep raw results and a separate archive backup. Return both archives for saved-data
analysis. Module/source/input hashes and each attempt remain in runner receipts.
`source_inputs.json` records the desktop source/module provenance for comparison;
the H100 module is the fresh build recorded by its own build/attempt manifests.

## Submit the two missing N32 profiles overnight

From the existing checkout root on the login node, after the environment above
is installed:

```sh
git pull --ff-only origin experiment/h100-stage-timing
sbatch experiments/h100_stage_timing_n32.sbatch
squeue -u "$USER"
```

The script requests one GPU and six CPUs for 15 minutes in the development
`dev_gpu_h100` queue. Keep this checkout unchanged until the job ends; the job does
not pull Git or install packages. The job records its actual Git HEAD and plan
hash, device/SM count when available, CUDA version and CPU details, then builds only N32 sm90
and runs B1/B16 SQP5/PCG100: one warmup and one measured fresh-owner call each.
It preserves the original diverse N32 F01c measured candidates and separate
unrelated warmup. Existing N128 results are not rerun.

Read `gato-n32-profile-<jobid>.out` for progress and the final exit/collection
status. Results resume under `.experiment-results/h100-stage-n32`; completed
and failed attempts stay preserved without automatic retry. On normal exit or
an ordinary failure, collection writes the separate archive
`.experiment-work/h100-stage-n32-<jobid>-results.tar.gz`. Return that archive
and the Slurm log. A hard scheduler kill may interrupt collection; raw files
remain under the result root and can be collected afterward with `collect.py`.
