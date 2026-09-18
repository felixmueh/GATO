# Solve-time and tracking matrix

`run.py` is the single entry point for the Indy7 experiment. It runs a matrix
of horizon lengths and batch sizes, records native solver timing and tracking,
and produces solve-time and solver-frequency plots. Choose the SQP iteration
budget with `--max-sqp-iters`; both modes use the same simulation and reporting.
The solver implementation and its native stopping rule are unchanged.

Defaults are float32, horizons 8/16/32/64/128/256, batches
1/2/4/8/16/32/64/128/256/512, a 10-second measured wall-time budget per cell,
and a maximum of 1,000 SQP iterations per solve. Each measured update advances
the simulation by exactly 10 ms. A cell stops after a measured solve exceeds
one second. One initial warm-up solve is excluded from timing, the measured
wall-time budget, and the slow-solve cutoff.

## Build

Run commands from the repository root, using an environment with the packages
in `requirements.txt`. The development container provides `/opt/gato-venv`;
the existing local runs used the separate, non-default interpreter
`build/solve-time-heatmap/venv/bin/python`. That local path is not required on
other machines. Select the same Python environment when building and running.

The following builds for the GTX 1070 (CUDA architecture 61). Change the
architecture for the target GPU. CUDA 12.2 was used for the existing runs.

```sh
cmake -S examples/solve_time_heatmap -B build/solve-time-heatmap-felix-devel \
  -DCMAKE_CUDA_ARCHITECTURES=61 -DPython3_EXECUTABLE="$(command -v python)"
cmake --build build/solve-time-heatmap-felix-devel --parallel 2 --target \
  bsqpN8_indy7 bsqpN16_indy7 bsqpN32_indy7 bsqpN64_indy7 bsqpN128_indy7 \
  resources_N8 resources_N16 resources_N32 resources_N64 resources_N128 resources_N256
python examples/solve_time_heatmap/smoke.py
```

The standalone build writes extensions into its own `modules/` directory,
without overwriting extensions under `python/bsqp`. Modules use matching
`-O3 --use_fast_math` flags with CUDA error checks enabled. Select another
standalone build with `--build-dir`.

Resource probes check kernel shared memory and launch thread limits before
running a horizon. On the GTX 1070, N128 PCG needs 35,476 bytes per block and
fits; N256 needs 66,196 bytes against a 49,152-byte limit. N256 therefore remains
visible as `SMEM`, without requiring its solver module. This is a per-block
limit, not exhaustion of the card's global memory. On hardware where N256
fits, also build `bsqpN256_indy7`. Probe allocations are independent of batch
size for these kernels.

## Run either SQP budget

The current experiment, allowing native stopping up to the safety cap:

```sh
python examples/solve_time_heatmap/run.py \
  --gpu-name 'GeForce GTX 1070' \
  --output test-artifacts/heatmap-native-stop
```

The same experiment with at most one SQP iteration per update:

```sh
python examples/solve_time_heatmap/run.py \
  --gpu-name 'GeForce GTX 1070' --max-sqp-iters 1 \
  --output test-artifacts/heatmap-one-sqp
```

A short pilot:

```sh
python examples/solve_time_heatmap/run.py \
  --gpu-name 'GeForce GTX 1070' --knots 8 16 --batches 1 2 \
  --wall-time 1 --output test-artifacts/heatmap-pilot
```

Every new run requires `--gpu-name`; this is the configurable plot title,
recorded separately from the detected device information. The default output
directory is `test-artifacts/solve-time-heatmap`; use a fresh directory for a
new experiment. `--repeats 3` runs three independent trials per
cell; reported matrix means give equal weight to completed repeats. An
asterisk marks a numeric cell with missing or unsuccessful repeats; a dagger
marks measured cap hits in the native-stopping mode.

`--wall-time` controls the measured wall-time budget per repeat.
`--max-solve-ms` controls the single-solve cutoff: the default is 1000 ms,
strictly greater than the threshold triggers it, and `0` disables it. Both
limits are checked between updates. An in-flight solve finishes and its full
sample is retained, so either limit can be exceeded substantially. Warm-up is
always excluded and can itself take longer than these limits. Optional
`--timeout SECONDS` is a separate hard timeout for the entire worker process,
including warm-up. It is disabled by default and may interrupt an in-flight
solve without a completed sample.

Resume an interrupted matrix with `--resume` and the same experiment arguments
and output directory. Resume checks configuration and source provenance before
reusing results. Recorded outcomes are skipped; an unrecorded partial cell is
archived before being restarted. Existing outputs are otherwise protected
from overwrite.
To redraw a saved matrix without GPU execution:

```sh
python examples/solve_time_heatmap/run.py --plot-only \
  --output test-artifacts/heatmap-native-stop
```

Plot-only mode reuses the saved GPU name. An explicit `--gpu-name` replaces the
saved label, or supplies one for older results that lack it.

## Historical reference protocol

The original `plots/gato_solve_time_heatmap.png` matches the image at commit
`4173824`. Its `benchmark_fig8.py` used one SQP iteration and a different clock.
Reproduce that protocol through the same entry point:

```sh
python examples/solve_time_heatmap/run.py \
  --gpu-name 'GeForce GTX 1070' --protocol reference \
  --sim-time 10 --max-sqp-iters 1 --max-solve-ms 0 \
  --output test-artifacts/heatmap-reference
```

Reference mode advances simulation by `min(previous native solve time, 10 ms)`
per update, including the historical 1 ms RK4 residual-time accumulation.
`--sim-time` sets simulated duration instead of a wall-time budget. Disabling
the slow-solve cutoff preserves the historical behavior: slow cells can take
much longer than ten real seconds. The reference image is not overwritten.
This reproduces the historical experimental protocol using the current
branch's solver and bindings, not its historical binary.

## Reading the results

- Solve time is synchronized host timing inside the native solver for one
  **complete batch**. It excludes initial warm-up, Python simulation and
  recording, and binding input/output transfers. Frequency is
  `1000 / mean_native_ms`, not batch-member throughput or achieved end-to-end
  control frequency.
- Fixed-step wall-time mode gives fast cells more simulated time and more
  figure-eight phases within their budget. A periodic extension of the sampled
  reference supports this. Timing and tracking means across cells therefore
  cover different simulated trajectories. The simulation clock does not model
  a real-time controller's delays.
- GATO's native convergence flag means the inner PCG solver used zero
  iterations. It is not a verified nonlinear KKT tolerance. Native flags,
  SQP iteration counts and cap hits are retained for **every batch member**.
  Reaching the wall-time or simulation-time budget means the experiment
  completed, not that every solve converged. A cap hit is retained as an
  outcome; it does not by itself stop a cell.
- Slow solves, nonfinite trajectories and unsupported resource requirements
  are distinct outcomes. Slow/partial samples remain available for inspection;
  they do not become ordinary completed-cell heatmap values.
- Tracking is the world-coordinate distance from the simulated joint-6 origin
  to the next reference knot. It is neither an offset tool-center-point error
  nor an orientation error. Means are per measured update, not time-weighted,
  and exclude warm-up.
- Both protocols retain the original ready pose, figure-eight geometry, zero
  external forces and unshifted warm starts. Batch 0 controls the robot and its
  plan is broadcast into the next warm start. Rho resets before every solve;
  duals reset once. The remaining original parameters include PCG cap 100,
  PCG tolerance 1e-6, KKT tolerance .001, solve ratio 1, mu 10, position cost 2,
  velocity cost .001, control cost `1e-8*N`, terminal cost 20, zero limit
  penalties and rho .1. Changing the SQP cap does not change these parameters.

Outputs include the matrix `results.json`, per-cell summaries, raw NPZ
samples, worker logs, source snapshots and hashes, and PNG/PDF/SVG figures.
`--save-plans` additionally saves complete batch-0 plans; it is off by default
to keep output size manageable.
The solve-time and frequency figures are named `gato_solve_time_heatmap` and
`gato_solver_frequency_heatmap`. Cells run sequentially in fresh processes,
with a recorded randomized order. Provenance includes the checkout revision,
loaded modules, build configuration, dependencies, and device information.
Generated results remain separate from experiment sources.

Run without competing GPU workloads or `CUDA_LAUNCH_BLOCKING`. The runner
verifies module locations and refuses modified solver/Python sources. It
restarts itself with the active environment's Torch library directory first
in `LD_LIBRARY_PATH` when needed; the effective path is recorded. The CPU
Torch package satisfies Python imports, while native CUDA extensions execute
the solver on the GPU.

## Implementation and checks

All maintained experiment code lives in this directory: `run.py` handles the
CLI and matrix, `experiment.py` shares simulation and recording across modes,
`protocol.py` holds model integration and solver parameters, and `plotting.py`
produces plots, CSVs and the report. There are no runtime imports from local
exploratory scripts under `build/`.

CPU regression checks require the experiment Python packages but no CUDA run:

```sh
python -m unittest discover -s examples/solve_time_heatmap -p 'test_*.py'
```

They cover warm-up exclusion, clock behavior, cutoff and partial outcomes,
historical protocol parity, repeat aggregation, and output preservation. The
short pilot above exercises the compiled solver; repeat it with
`--max-sqp-iters 1` and a different output directory to check both SQP modes.
