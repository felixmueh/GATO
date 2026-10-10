# Saved F01ab timing plots

`plot_f01ab.py` reads already saved short12 modeA/B1 F01ab attempts. It never
imports solver/controller/native code, executes experiments or changes raw
artifacts. It keeps the required64cells across1070/5060/Jetson visible, even
when only a fewattempts are available. No F01c/BA/BB scope is included.

Root prepares a small explicit selection JSON using existing attempt
`SHA256SUMS.json` receipts. This is a selection table, not a new runner index or
provenance framework. It can be placed beside the results or resolved against
`--input-root`. Every path is relative to that root without `..`; symlink escapes
are rejected. Independently verify the selection and each manifest before
supplying their hashes. A self-generated manifest alone is not an independent
source/build/result audit.

Example shape (placeholders, not measured evidence):

```json
{
  "evidence_kind": "measurement",
  "bank": {"path": "frozen-bank/bank.json", "sha256": "<audited bank JSON hash>"},
  "attempts": [{
    "machine": "gtx1070", "N": 16, "SQP": 1, "PCG": 100,
    "source_commit": "<full experiment commit>", "cohort_id": "<frozen cohort>",
    "attempt_id": "001", "path": "gtx1070/<case>/attempt-001",
    "manifest_sha256": "<independently pinned SHA256SUMS.json hash>"
  }]
}
```

An optional `schema: "f01ab-analysis-index-v1"` names this selection layout.
Realmeasurement is the default; synthetic data require explicit
`evidence_kind: "synthetic_test"`, the same provenance marker, and CLI
`--synthetic-test`. Every synthetic figure is watermarked **SYNTHETIC TEST DATA —
NOT MEASUREMENT**. Synthetic fixtures belong only in temporary test directories;
never use their plots as campaign evidence or publish them as measurements.

```sh
python3 experiments/analysis/plot_f01ab.py \
  --index /path/to/audited-selection.json \
  --index-sha256 <independently-verified-selection-hash> \
  --input-root /path/to/results \
  --output /path/to/new-derived-f01-directory
```

Before rendering, the loader checks selectionhash, bankhash, every selected
manifesthash and every manifest-listedfile hash. Each attempt must include
`protocol.json`, `provenance.json`, `metrics.json`, and `events.jsonl` unless
execution failed before recording anyevent. Every event-index/rawNPZhash/path
must agree with the manifest. All files retain hash verification; only solve
payload NPZs are decoded, avoiding plant/outcome decompression. All NPZ reads
disablepickle. Protocol fields
N/SQP/PCG/short12/A/controller_dt/internal/B1 and source/cohort IDs must match
the explicit selection; provenance machine/source/cohort/bankhash must match.
Changed sources/cohorts require separate invocations. Duplicatecase/attempt
identities, reusedattemptpaths, wrongcategory/slot/mode/cap and unsafe paths
are rejected before outputcreation. Existing source/protocol receipts remain
immutable. Rendering refuses existing outputs and descendants of rawattempts.

The loader groups `solve_started`, `solve_returned`, `solve_completed` and
`solve_failed` by episode_id/slot; these stages represent **one call**, not three
or fourtiming samples. Additional native_return/plant/outcome events are hashed
but not counted as extra solves. A returnedcall that later fails retains its
recordedtiming. warmup is a separately tagged excluded call with noepisode/slot,
retained in CSV and its ownsummary/coverage row. Eachattempt produces241rows:
240scheduledslots (12cold/222warm/6goal-switch), plus1warmup. Missing slots get
explicitmissing rows; failures, nonfinite/nonpositive/missingtimings and negative
task/physical episodes remain visible. No retries, slotreplacement, slow-sample
filter or goal-switch/warm pooling occurs. Episode task/physical/numerical fields
are preserved independently; rawtimings are not conditioned on success.

Native milliseconds and wrapperwall seconds converted to milliseconds have
separate tables/figures. Finitepositive timings, including recordedfailedcalls,
produce mean/median/p95 primarysolve-time views in separate per-N figures,
keeping the fullmatrix labels readable. Missing/nonfinite/invalid counts
remain exported and printed infigurecoverage; unavailable values are marked at0
with grayX in the overview (theX is a missingmarker, not a0measurement).
Secondaryviews are exactly1000/mean_ms,1000/median_ms,1000/p95_ms, labeled as
reciprocalaggregates, not meaninstantaneousHz or actuationfrequency. Native and
wrapper scopes are never pooled. warmup is excluded fromthese primaryviews.

Per-capcell/machine/attempt category-colored histograms use common twentybin
edges over allmatchedmachine/attempt finitepositive samples for that capcell and
scope. Their fixedcolors/binedges are saved inprovenance. Histograms display
actualdistributions without assuming bimodality. Timingcategories remain
cold/warm/goal_switch. No full-period tracking,8sT01success, strictdeadline gate
or exactreplay equivalence is implied: evolvingstates differ acrosscapcells.

Outputs: PNG+PDF primarysolve-time and reciprocalviews forbothscopes;
PNG+PDF per-cell/machine/attempt/category histograms; `all_samples.csv`,
`cell_summaries.csv`, `category_coverage.csv`, `matrix_coverage.csv` (all192cells),
and `provenance.json` with selection/bank/script/attemptfile hashes, sources,
cohort/attempt IDs, commonbins/colors and statistic/interpretation definitions.
Separate attempts remain separate rows/markers/histograms; they are not pooled
into a successrate. No raw or derivedscientific data are produced by CPUtests.

CPU validation:
`/opt/gato-venv/bin/python3 -m pytest tests/python/test_plot_f01ab.py -q`.
Sixbounded tests exercise genuinepersistedRawRecorder fixtures (explicitly
synthetic), deduplicatedcalls, failed/nonfinite/missing/negativeoutcome samples,
categorycounts/reciprocaldefinition, matchedbinrendering/outputrefusal,
source/mode/category/path/byteintegrity rejection and missing192-cell coverage.
Existing hppfcl/coal dependency warning is unrelated toplot behavior.
