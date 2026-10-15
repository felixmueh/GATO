# Two orientation-cost figure runs — standalone instructions

The runtime is a clean detached checkout at `/workspace/GATO-orientation-figure-source`, exact revision `b400bdf2bb6eea543b17cefcb0577d5de4d22ce2` (current `felix-devel` when requested). These are new current-revision runs, not a June reproduction. A bounded search found no matched tracking ON/OFF trajectory pair; random orientation reaching and the differently configured D02 tracking are not substitutes.

**Execution status:** the matching N64 build and the corrected, explicitly bound pair completed on 2026-10-14. Current export: `/workspace/GATO/thesis_results/orientation-cost-comparison-20261014/orientation-cost-figure-bound-20261014T134643.034253Z/`. The first pair is retained with a source-binding-invalid marker: inherited editable packages selected the main runtime instead of the intended isolated runtime. See the current main handoff/report for results; no weights, timing or solver code changed for the correction. Main's existing N64 binary has SHA256 `3a97a4682c949eba483ad7be39fc64416032abae0141e6f049af58405f8b8f9f` and is an unchanged older artifact; no matching b400 provenance was found. Do not use it. No source/controller changes, native build or GPU run occurred during this preparation.

Both runs use N64, 8 ms knots/feedback, 1 ms internal plant, 16 s and B1. Prediction duration is 0.504 s. The horizontal reference has 0.32 m x span, 0.08 m y amplitude, 24 s period, 0.02 m start-edge clearance and five stored cycles; the run covers the first 16 s. Start: `comfortable_high_clearance`, zero velocity; constant force zero. RPY (0,0,0) means tool +Z up in the torso frame. OFF weights:0/0; ON:0.2/4. Public flags are `--ee-orient-cost` and `--ee-orient-N-cost`; the target reference stays identical in both runs.

Current tracking defaults are SQP5, PCG120, KKT/PCG tolerance0.001, solve_ratio1, mu1, position costs4/80, velocity damping0.01, effort cost0.0001, joint-limit0.01, velocity-limit0.001, control-limit0.003, rho0.01. Final approach is disabled; initialization is constant current-state knots/zero controls, followed by the public warmup and warm starts. The unchanged public CLI calls `run_mpc_fig8` with **offline_timing="controller_dt" (mode A)**. It has no mode-B CLI flag; this figure study preserves that public default rather than adopting the campaign's elapsed-solve-time mode B. The summary records actual effective costs; KKT convergence fraction is descriptive, not an execution gate.

These are future local commands, not authorization to execute during preparation. Use the existing CUDA/Pinocchio Python and keep the GPU free of other measurements. From the runner checkout:

```bash
FIGURE_PYTHON="$(python3 -c 'import sys; print(sys.executable)')"
FIGURE_SOURCE=/workspace/GATO-orientation-figure-source
FIGURE_BUILD="$FIGURE_SOURCE/build/orientation-cost-figure-N64"
FIGURE_ARCH=61 # GTX1070; use the actual target architecture if moved
mkdir -p "$FIGURE_BUILD"
git -C "$FIGURE_SOURCE" rev-parse HEAD > "$FIGURE_BUILD/source_commit.txt"
cmake -S "$PWD/experiments/native" -B "$FIGURE_BUILD" -DGATO_SOURCE_DIR="$FIGURE_SOURCE" -DKNOTS=64 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$FIGURE_ARCH-real" -DPython3_EXECUTABLE="$FIGURE_PYTHON" > "$FIGURE_BUILD/configure.log" 2>&1
cmake --build "$FIGURE_BUILD" --target bsqpN64_tiago_right --parallel 2 > "$FIGURE_BUILD/build.log" 2>&1
cp "$FIGURE_BUILD"/modules/bsqpN64_tiago_right*.so "$FIGURE_SOURCE/python/bsqp/"
```

Stop if either build command fails; do not install or run a failed build.

Use a fresh pair directory. Each command below invokes the unchanged public CLI once with explicit N/dt/sim_dt/duration/B1/start/RPY/weights and `--no-gif`. The helper refuses a different/modified source or mismatched installed module. It explicitly selects the pinned Python paths, restricts the BSQP namespace and preloads the exact N64 extension before invoking the unchanged public main in that same process. `loaded_runtime.json` records actual Python paths and native path/SHA before simulation; the receipt embeds it and the bootstrap SHA. Export checks actual loaded versus selected module hashes. This prevents inherited editable packages from silently selecting another checkout. The helper also records command/start/end/machine/GPU/build hashes and retains stdout/stderr. It does not add a preflight or extra optimizer runs:

```bash
FIGURE_STAMP="$(date -u +%Y%m%dT%H%M%S.%NZ)"
FIGURE_RAW="/workspace/GATO/_expr_results/orientation-cost-figure-$FIGURE_STAMP"
"$FIGURE_PYTHON" experiments/orientation_cost_figure/collect.py run --root "$FIGURE_RAW" --source "$FIGURE_SOURCE" --build "$FIGURE_BUILD" --condition disabled
"$FIGURE_PYTHON" experiments/orientation_cost_figure/collect.py run --root "$FIGURE_RAW" --source "$FIGURE_SOURCE" --build "$FIGURE_BUILD" --condition enabled
```

These are two independent planned attempts. Preserve failures and partial files, report them, and do not retry/tune until green. No ROS/Gazebo/robot flag is supplied. Collect only the requested CSV/summary/direction PNG and compact provenance into a fresh timestamped thesis subdirectory:

```bash
"$FIGURE_PYTHON" experiments/orientation_cost_figure/collect.py export --root "$FIGURE_RAW" --destination /workspace/GATO/thesis_results/orientation-cost-comparison-20261014
```

The export checks exit status, finite actual/joint/reference/control CSVs, effective protocol, duration within0.05 s of16 s and identical paired reference hashes. It preserves physical violations separately and does not require KKT convergence. Raw summaries remain byte-identical; nonfinite derived JSON scalars become null, with the original summary retained. Extra plots/logs stay in `_expr_results/`; no GIF is requested. In-memory trajectory data may be unavailable if the public script fails before CSV export; preserve its logs/receipt rather than inventing samples.

The existing thesis `figure_generators/orientation_tracking/generate.py` reads the paired exports and computes mean/max tool-axis angle. Report whether ON stays closer to vertical, either way. Final analysis/reporting uses the verified pair; the source-binding-invalid pair and its original results remain historical evidence.
