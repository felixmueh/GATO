# Jetson frozen OFF_TORQUE fresh100

J1 is charged mode B / SQP5; J2 is charged B / SQP1. Both are required.
J3 is optional mode A / SQP1000 and must wait for the J1/J2 count checkpoint.
No cap1000 mode-B run, automatic optional arm, filtering, tuning or retries.

`inputs/manifest.json` freezes exact portable pool/pairs/assignment bytes plus
unchanged desktop protocol, freeze, failed canonical context and q020 build
receipt. Ordered task SHA256 is438d9ac51ec264ffcd2bb0dfa40b9caaccfc579990cc3b1b502cd40e48a6a96e.
Native85ef numerical bytes are published as tree-equivalent
fd3fa2f2c3572442ad158c32c5fdfa5c7c661ece; the original source identity remains in
the protocol. Fresh sm87/aarch64 module hash/receipt are mandatory at execution,
with q .20rad inset / velocity .80 / effort .85 cost bounds. The old desktop
module d9bee23f90a1d7816b32db23749bd651db2a7537f94c3dfec15f1a448bb34cc8
is separate context and only reused by the explicitly separate GTX smoke.

The implementation owner provides build/receipt and runtime commands. CPU-only
`prepare.py --module PATH --module-receipt PATH --output FRESH_DIR` materializes
J1/J2/J3 protocols; dry runs may record pending module, actual runs reject it.
Original mode-A functions/JSON remain unchanged. New `validate_worker` checks
only declared timing/cap/platform/module exceptions against frozen settings.
Actual ARM runs reject a desktop x86 sm87 compile receipt.

Raw layout: `RAW/J1/pilot` and `RAW/J2/pilot`; optional `RAW/J3/pilot`.
Each fresh canonical is under `RAW/J1/canonical/preflight` etc. Its recorded
arrival/physical/error label is preserved; infrastructure alone gates launch.
The old55.740448mm canonical timeout must never be relabeled a pass.
Host power/clock logs belong under `RAW/_host_conditions`; save nvpmodel and
jetson_clocks state plus GPU telemetry without changing settings. Export reads
these records; absent conditions are unavailable rather than invented.

CPU export checkpoint and final export (no CUDA or ROS):

```sh
python3 experiments/jetson_random_reaching/export.py --raw-root RAW --arms J1 --counts-only
python3 experiments/jetson_random_reaching/export.py --raw-root RAW --arms J1 J2 --counts-only
python3 experiments/jetson_random_reaching/export.py --raw-root RAW --output /workspace/GATO/thesis_results/random-reaching-jetson-fresh100-20261015 --arms J1 J2
```

Counts-only J1 works before a J2 folder exists. Final output requires100 recorded
outcomes for each selected arm. J1/J2 always precede optional J3; it is included
only when explicitly selected. Original desktop analysis helpers are reused,
with no universal export schema. Original cases fields are extended by arm,
clock/cap, charged/native/wrapper timing, actual work and accepted steps.
`solve_samples.csv.gz` preserves individual distribution data; histories and
failed solve arrays remain raw. First-limit records retain ties and actual
active-plan source; mode B must not attribute an old executed plan to the
just-returned solve. Full-history errors/extrema remain unavailable on failure.
Wilson95 A/P/B/E intervals retain all100 cases. Desktop mode A/SQP1000 is
labeled a juxtaposition for J1/J2, while J3 supports same-mode/cap per-task
agreement. Neither is a causal platform/latency or full convergence claim.

The export creates the requested outcome bars, fixed-denominator arrival ECDF,
solve-time distributions, standalone REPORT.pdf and WRITING_HANDOFF.md using the
existing report renderer. Final archive/backup and rendered-page checks are
performed on collection; no such target result is claimed during preparation.

Bounded CPU checks:

```sh
python3 -m pytest tests/test_jetson_fresh100_export.py -q
```
