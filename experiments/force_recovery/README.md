# Force-estimator source recovery

This inventory is a bounded, read-only CPU discovery pass on
`experiment/eval-force` at `16b33719b4f0fd463eec7262c99b16b0b324366f`.
Evidence, hashes, limitations and reproduction commands are in
[`source_inventory.json`](source_inventory.json).

## Recovery verdict

The estimator algorithm is recovered. Historical Git has
`ImprovedForceEstimator` (`2a5d902`), and its executable Python AST is
identical to current `ForceEstimator` after renaming the class and removing
docstrings. The current TIAGo controller supplies a separate adaptation layer: it configures
the estimator, transforms 6D world-frame wrench hypotheses to the GATO frame,
sets the batch wrench and selects a hypothesis from one-step prediction error.
The generic Indy7 notebook preserves one concrete configuration: batch sizes
1/32/128 with a −60 N world-z wrench. It has no embedded outputs.

The missing piece is provenance and evidence for the intended adapted
experiment: the controller/config revision, model/frame, hypothesis set and
linked raw measurements. The current pickle is a 5-byte empty dictionary, not
results. Git history has a different 84,114-byte pickle object, but its relation
to this experiment is unverified. The current TIAGo Python interface and
compiled binding both reject nonzero wrench, so their presence cannot establish
a successful adapted nonzero-wrench TIAGo solver run. No comparative benefit is
established.

## Next bounded step

Obtain one concrete lead from Felix: the original adaptation checkout/archive,
or the workstation/Jetson result directory name. Search only that identified
location read-only; hash every candidate and confirm it links source revision,
model/frame, force hypothesis inputs and outputs. If no such lead exists,
preserve this gap and ask whether to define a clearly new, explicit baseline.
Do not relabel the generic notebook/pickle as adapted evidence or recreate the
missing implementation from inference.
