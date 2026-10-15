# Four-task batching comparison export

CPU-only saved-record analysis; it imports neither worker nor native solver.
The exact selected order is 7733, 8054, 2497, 9651. The saved OFF_TORQUE
B1 cap1000 reference is reused. B16 cap1000 and B1_EXTRA cap16000 are new
closed-loop episodes. Their nominal lane allowances match; measured work
and timings need not match. All-lane native stopping remains coupled.

During collection, write previews outside thesis_results:

```sh
python experiments/random_reaching_baseline/analyze_batching.py \
  --raw-root /workspace/GATO/_expr_results/random-reaching-batching-20261015 \
  --output /tmp/gato-batching-preview --allow-partial \
  --previews /tmp/gato-batching-preview-pages
```

After all four cases in each arm are complete and root reviews the report,
export without `--allow-partial` to
`/workspace/GATO/thesis_results/random-reaching-batching-20261015`.
Incomplete/held data are rejected at that final destination. Canonical scientific
failures retain their original labels and diagnostic policy. Shared setup holds,
execution failures and missing cases remain distinct; prefix physical flags
cannot count as full-episode physical passes.

The exporter checks summary equality against case metrics, actual saved input
arrays against the paired reference, protocol/source/module/input identities,
and saved work totals against the solve log. The compact export contains cases,
per-solve work/timing, counts, metadata, figures and editable scripts. The full
regeneration command is retained in metadata.json. For laptop adjustments:

```sh
python analyze_batching.py --plot-only .
python report_to_pdf.py --report REPORT.md
```

The selected four cases are not a population sample. Do not combine their rates
with the old twelve-task denominator or claim that a rescue proves multimodality.
