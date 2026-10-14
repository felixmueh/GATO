# H100 quick checks

Run inside an allocated, uncontended whole H100 GPU, never on the login node.
The two explicit plans are independent. Neither runs BB or BA.
Use Git, CMake >=3.24, a C++ compiler, CUDA supporting sm90, Python headers,
NumPy, SciPy, Pinocchio (`pin`) and pybind11; PyTorch is not required.
Activate the existing suitable Python environment and load the site CUDA module.
Both recipes compile with two CPU jobs and select `90-real` explicitly.

```sh
git fetch origin experiment/h100-stage-runtime852 experiment/h100-fig8-runtime
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
