# Recorded MPC mesh comparison

`render_comparison.py` reads two or more episode directories containing
`inputs.npz`, `executed.npz`, and `metrics.json`. Provenance is read from the
episode or its parent run. It compares the same initial state and FK goal,
using recorded joint positions and actual simulated timestamps. It makes no
solver calls, integrates no controls, and constructs no predicted horizon.

Example display check using existing records (no batching-benefit claim):

```sh
/opt/gato-venv/bin/python experiments/random_reaching_baseline/render_comparison.py \
  --episodes \
    /workspace/GATO/_expr_results/random-reaching-orientation-20261015/LOW_TORQUE/pilot/pair_07733 \
    /workspace/GATO/_expr_results/random-reaching-orientation-20261015/OFF_TORQUE/pilot/pair_07733 \
  --labels LOW_TORQUE OFF_TORQUE \
  --output .experiment-render/display-check-7733
```

The output must be fresh and outside measured episode directories. It contains
portable `comparison.html` with embedded meshes and joint transforms, a static
`comparison.png`, compact displayed joint samples, and `render_receipt.json`.
The HTML requires no CDN, server or external mesh files. Drag to rotate shared
cameras; scroll to zoom; pause or scrub the common simulated clock. `--gif` adds
an optional mesh GIF; HTML/PNG are the default. Keep bulky display checks under
the experiment worktree or `_expr_results/`; only selected finalized views
belong in `thesis_results/`.

The adapter reuses mesh/FK routines from the existing reaching-landscape
renderer (`--renderer-source`, default `/workspace/GATO-reaching-landscape`).
Full visual assets default to `/workspace/GATO/TiagoProURDF/tiago_pro.urdf`
(`--full-urdf`); `comparison_context.json` fixes all non-right-arm joints for
display. Full/reduced tool FK parity, renderer/helper hashes, all used mesh
hashes, URDF/context hashes and original episode/provenance hashes are recorded.
The complete robot body is display context, with no whole-body dynamics claim.

Joint positions are linearly interpolated for display only. Both views use one
simulated clock. Once an episode ends, the last pose stays visible with an
explicit **ENDED** label; it is not a claim about post-arrival behavior. Physical
and task labels are the actual recorded outcomes, and physical compliance stays
separate from reaching success. Display meshes do not assess collisions.

An eight-second executed MPC history is distinct from each N32, 0.93-second
prediction window. Equal feedback times in different closed-loop episodes do
not imply a common actual state or optimization input. This adapter does not
perform refinement, compare local minima, establish multimodality or certify
any batching improvement.
