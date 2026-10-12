# Additional Jetson F01ab and CPU collision timings

Felix authorized these two additional studies on 2026-10-12. Use the separate
[68-case plan](jetson_additional_f01ab_m02.json). The completed Jetson campaign
and its historical cap-only F01ab data remain separate, including every failure.
This launch adds four CPU collision cases and the complete 64-cell F01ab matrix.

F01ab reuses the current GTX/RTX frozen inputs, collector, build and mode-A
preflight at `fa6f6bede286c122cc41ac65f42a583da84290e0`. Its restored
`zero_pcg_or_finite_kkt` stopping policy differs from the old Jetson runtime
`502fa29`, whose complete campaign took 42.96 minutes. Residual KKT flags remain
separate from the heuristic zero-PCG exit. Retain both cohorts; do not pool their
timings as measurements of one stopping policy. N16/32/64/128 × SQP1/2/5/10 ×
PCG100/200/400/800, all frozen short12 episodes and excluded warmups stay intact.

M02 uses the existing monitor source
`2db284ca8e11f42f8f3248403b879652966893ed`, bundled original/wrapped geometry,
eight named states, 100 query and 20 synchronous-worker repeats per state,
exact/margin cases and 4 cm margin. It requires Pinocchio/Coal on the Jetson CPU,
with no native build or GPU solve. Wrapped means the existing simplified
geometry. Wrapped cases run first; their current working duration estimate is
under one minute for the pair, followed by about 20 minutes for the original
pair. These are planning estimates, not measured Jetson results. A 1,800-second
wall guard per M02 case gives execution headroom; the old plans retain their
600-second guard and scientific samples/settings are unchanged. Safety rejection
is recorded evidence, not a claim of executed task or live monitor qualification.

Allow roughly one hour for F01ab measurements as an initial scheduling allowance,
plus fresh CUDA builds and the applicable frozen suite/canonical preflight;
restored stopping can change runtime. Use the existing Jetson CUDA development
environment (`87-real`). Required timing evidence can occupy about 1.53 GiB
uncompressed; allow at least 3 GiB for these results plus their return archive,
and additional space for source/build caches, extra attempts and an old archive
during replacement. This is an allowance, not a storage cap or measured forecast.
Keep CPU timings and GPU measurements sequential with other work on the Jetson.

After pulling the published `experiment/simple-runner` branch, run this paste
batch from its checkout. It explicitly uses the currently selected Python
interpreter, fetches missing exact source refs through the existing launcher
helper, checks the input/build pins, then lists all 68 cases without execution:

```sh
JETSON_EXPERIMENT_PYTHON="$(python3 -c 'import sys; print(sys.executable)')"
"$JETSON_EXPERIMENT_PYTHON" -c 'import json,sys; sys.path.insert(0,"experiments"); import machine_launch; machine_launch.verify_sources(json.load(open("experiments/launch/jetson_additional_f01ab_m02.json")))'
"$JETSON_EXPERIMENT_PYTHON" experiments/run.py --machine jetson \
  --plan experiments/launch/jetson_additional_f01ab_m02.json \
  --python "$JETSON_EXPERIMENT_PYTHON" \
  --cache .experiment-cache/jetson-additional-f01ab-m02-20261012 \
  --output .experiment-results/jetson-additional-f01ab-m02-20261012 --dry-run
```

Run the CPU study, then the GPU study as separate commands. Each uses the same
fresh additional result root and cache; the existing runner owns source
worktrees/builds and preserves all attempts. Completed cases resume unchanged;
failed/blocked cases require an explicit `--retry-failed` to create a new attempt.

```sh
"$JETSON_EXPERIMENT_PYTHON" experiments/run.py --machine jetson \
  --plan experiments/launch/jetson_additional_f01ab_m02.json \
  --python "$JETSON_EXPERIMENT_PYTHON" \
  --cache .experiment-cache/jetson-additional-f01ab-m02-20261012 \
  --output .experiment-results/jetson-additional-f01ab-m02-20261012 --only M02
"$JETSON_EXPERIMENT_PYTHON" experiments/run.py --machine jetson \
  --plan experiments/launch/jetson_additional_f01ab_m02.json \
  --python "$JETSON_EXPERIMENT_PYTHON" \
  --cache .experiment-cache/jetson-additional-f01ab-m02-20261012 \
  --output .experiment-results/jetson-additional-f01ab-m02-20261012 --only F01ab
"$JETSON_EXPERIMENT_PYTHON" experiments/collect.py \
  --output .experiment-results/jetson-additional-f01ab-m02-20261012 \
  --export .experiment-work/jetson-additional-f01ab-m02-20261012-results.tar.gz
```

A M02 problem does not authorize changing its samples or holding the independent
F01ab study. An F01ab shared build/preflight failure holds its dependent cells;
retain and report unexpected failures. Collect after an interruption as well:
the archive includes existing partial data and every attempt, excluding caches.
Copy that archive to the desktop `/workspace/GATO/_expr_results/` and retain a
separate raw backup. The finalized analysis will retain source, stopping-policy,
warmup, timing, task/physical outcome and failure labels.

Publication requires the runner branch containing these two files and both
immutable source refs on the same origin:

| Published ref | Exact commit |
| --- | --- |
| `experiment/deployment-source-fa6f6bede286` | `fa6f6bede286c122cc41ac65f42a583da84290e0` |
| `experiment/deployment-source-2db284ca8e11` | `2db284ca8e11f42f8f3248403b879652966893ed` |

Local preparation does not claim publication, target builds/preflight or any
new measurement. The main 84-case Jetson, 89-case RTX and existing Gazebo plans
remain unchanged; this plan contains only M02 and F01ab.

Preparation validation (2026-10-12): source/input/build hashes passed; the full
68-case dry-run passed. A bounded static comparison confirmed exact F01ab reuse
except machine selection, unchanged M02 sample/settings, fresh output/cache
paths, interpreter/build-cache compatibility and valid shell commands. Plan
SHA256: `de2f83fa5675b71af08be803107535c6b3dd17dd3860b5bd478104f41162b954`.
No build, GPU solve or CPU measurement was executed during preparation.
