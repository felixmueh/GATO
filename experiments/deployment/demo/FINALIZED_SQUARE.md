# Accepted vertical square with forward-facing center

Felix accepted the tested v5 square **as is** on2026-10-12. The shared
`proposal.json` and all three standalone scripts now contain its exact five
poses: the four revised corner orientations, then center at `[0.5,-0.4,0]`m
pointing along torso **+X**. The position/axis geometry, current weights,
N32/30ms knots/3ms plant substeps,15s per goal and settling switch are frozen.
Do not tune or rerun to erase the three strict timeouts.

The GTX1070 internal result passes rough reaching5/5 and sampled physical
checks, with all recorded states/controls finite. Strict settling passes2/5;
top-right/top-left/bottom-left retain timeout outcomes. Raw source:
`/workspace/GATO/_expr_results/square-demo-center-forward-v5-20261012/attempt-001/`.
Compact report/GIF:
`/workspace/GATO/thesis_results/square-demo-center-forward-v5/`.
`finalized_square_check.json` records equality against25 saved motion fields.
Existing focused CPU tests pass14/14, and all three standalone preview checks
pass without native/GPU/ROS imports or motion. No new GPU run was performed.

## Standalone and collector reproduction

Inspect `demo_internal.py`, `demo_gazebo.py` or `demo_robot.py`. Their default
selection is square only; the three figure-eight choices remain available as
explicit optional selections and are otherwise recorded as skipped. Keep
`PREVIEW_ONLY=True` while reviewing the printed configuration. Each script
requires an explicit reviewed module path and successful matching canonical
preflight evidence for actual execution. Defaults do not build or move anything.

The collector source is separate from runtime identity. **Internal execution
needs the fa6f6bede286c122cc41ac65f42a583da84290e0 runtime checkout**, explicitly
passed with `--source` or set as the standalone `RUNTIME_SOURCE`. The collector
checkout itself is based on reviewed live runtime8dbb8b8d6dc1d65e902373a9aa58693e33329c14,
so its shared command's default source is suitable for live execution only.
The existing standalone internal path names `/workspace/GATO-eval-ordinary`;
adjust it to the actual fa6 checkout on the machine. Source identity is checked
before execution; there is no installed-module fallback.

CPU-only configuration inspection:

```sh
python3 experiments/deployment/demo/run_demo.py --backend internal --tasks square --source /path/to/fa6-runtime --preview
python3 experiments/deployment/demo/run_demo.py --backend gazebo --tasks square --preview
python3 experiments/deployment/demo/demo_robot.py
```

For the actual backend, first run the canonical rough-reaching preflight using
the explicit module; then supply its output to the single square sequence:

```sh
python3 experiments/deployment/demo/run_demo.py --backend internal --preflight --source /path/to/fa6-runtime --module /path/to/reviewed/N32-module.so --output /fresh/preflight
python3 experiments/deployment/demo/run_demo.py --backend internal --tasks square --source /path/to/fa6-runtime --module /path/to/reviewed/N32-module.so --preflight-output /fresh/preflight/canonical-preflight --output /fresh/square
```

For Gazebo use `--backend gazebo` and its reviewed live checkout/module, with
the sourced simulation ROS profile and already-running advancing simulator.
Gazebo has not run this square yet. Its separately prepared square case is
outside the unchanged89 RTX/84 Jetson internal queues; the runner owner adds
it explicitly to the Gazebo integration plan.

Physical execution remains standalone/manual and **unrun**. Review every
constant and source/module selection; use the required Jetson robot profile,
fresh physical canonical preflight and fresh timestamped output. The existing
literal `RUN square ON ROBOT` confirmation, collision/joint/torque safety,
state observations, controller handback and separate manual index are unchanged.
Internal task passes are not a robot or swept-collision certificate.

Runner execution uses its already-tested compact recorder adapter. Direct
standalone execution retains the collector's original recorder; that difference
is not a claim of timing equivalence. The actual accepted v5 data used the
23780e1 reference recorder and keeps its original provenance. Original square,
v4 center failure, endpoint-screen failures and all raw measurements remain
preserved; no result was overwritten or relabelled.
