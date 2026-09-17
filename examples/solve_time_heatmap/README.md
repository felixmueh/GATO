# Solve-time heatmap experiment

Reproduce `plots/fig8_benchmark_heatmap.ipynb` using the unchanged Indy7 solver
on `felix-devel` (run revision recorded in the result). Defaults: float32,
N=8/16/32/64/128/256, batches 1/2/4/8/16/32/64/128/256/512, one 10-second
figure-eight simulation per cell. The reference image is not overwritten.

The branch's base Dockerfile already installs its Python dependencies into
`/opt/gato-venv` using `uv.lock`. For a separate experiment environment,
`requirements.txt` pins the versions used for this run. Its CPU Torch package
satisfies the Python controller's import; all solver GPU execution goes through
the native CUDA extension. CUDA 12.2 and architecture 61 were used here.

```sh
cmake -S examples/solve_time_heatmap -B build/solve-time-heatmap-felix-devel \
  -DCMAKE_CUDA_ARCHITECTURES=61 -DPython3_EXECUTABLE="$(command -v python)"
cmake --build build/solve-time-heatmap-felix-devel --parallel 2 --target \
  bsqpN8_indy7 bsqpN16_indy7 bsqpN32_indy7 bsqpN64_indy7 bsqpN128_indy7 \
  resources_N8 resources_N16 resources_N32 resources_N64 resources_N128 resources_N256
python examples/solve_time_heatmap/smoke.py
python examples/solve_time_heatmap/run.py
```

Run from the repository root. The standalone build writes modules into its
own `modules/` directory. The runner verifies those module paths and refuses
modified solver/Python sources. All modules use matching `-O3 --use_fast_math`
flags with CUDA error checks enabled. Existing extensions under `python/bsqp`
are not overwritten. The experiment makes no changes to the solver, binding,
controller or generated GRiD sources.

The N256 resource probe is built even though its solver module is unnecessary
on Pascal. It reports static plus dynamic shared memory and launch thread
limits for every solver kernel. The oversized PCG launch is also tested:
CUDA rejects it before execution. Unsupported horizons remain visible as
hatched `SMEM` rows. If running on hardware where N256 fits, build
`bsqpN256_indy7` as well. Probes use B=1; these kernels' per-block allocations
and thread counts do not depend on batch size.

A short pilot and a plot-only rerun:

```sh
python examples/solve_time_heatmap/run.py --knots 8 128 256 --batches 1 2 512 \
  --sim-time 0.1 --output test-artifacts/heatmap-pilot-felix-devel
python examples/solve_time_heatmap/run.py --plot-only
```

The full run saves `results.json`, per-cell JSON summaries, raw NPZ samples,
worker logs, a source diff and PNG/PDF/SVG figures under
`test-artifacts/solve-time-heatmap-gtx1070-felix-devel/`. It refuses to overwrite
an existing results manifest. Cell order is shuffled with a recorded seed;
each cell gets a fresh process and one unmeasured warm-up solve. Failed cells
never become numeric timings. Optional `--repeats 3` runs three simulations;
means then give equal weight to each successful repeat, matching the reference
notebook. An asterisk means some requested repeats failed.

## Interpretation

- “GPU Solve Time” retains the reference label. It is synchronized **host wall
  time** inside BSQP, including kernel launch and synchronization overhead.
  It excludes Python processing and the binding's input/output transfers.
  Reciprocal contours are solver rates, not end-to-end MPC rates.
- dt=0.01, sim_dt=0.001, ready start configuration, the reference trajectory,
  one SQP iteration, PCG cap 200 and cost weights match the original benchmark.
  `offline_timing='solve_time'` explicitly selects the original benchmark's
  wall-time-driven simulation using this branch's existing public option.
  Its current controller also shifts warm starts and interpolates references;
  this is a reproduction on `felix-devel`, not identical historical code.
- B>=4 retains the controller's force estimator even without an applied
  external disturbance, with a recorded seed. The estimator rejects B=2;
  an experiment-local subclass uses zero force hypotheses at B=2, as at B=1.
- The GTX 1070 allows 49,152 shared bytes per block. N128 PCG uses 35,476 bytes
  and fits. N256 needs 66,196 bytes and fails. This is a per-block constraint,
not exhaustion of the GPU's 8 GB global memory.
- Timing success requires finite state, tracking and solve-time samples. It is
  not a convergence or tracking-quality claim. Raw samples and tracking errors
  are retained for inspection. No fixes from other branches are applied.
- Run without competing GPU workloads or `CUDA_LAUNCH_BLOCKING`. Device/driver,
  utilization and clock snapshots, source/module hashes, build cache, compiler
  version and dependency versions are recorded. Clock/power settings retain
  their existing values.

The runner starts itself again with the active Python environment's Torch
library directory first in `LD_LIBRARY_PATH` when needed. This avoids a base
container's globally configured Torch library directory overriding a different
version installed in the experiment environment. The effective path is recorded.
