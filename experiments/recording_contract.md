# Future-run measurement contract

Felix approved this smaller contract on 2026-10-12. It supersedes blanket full
solver input/output retention for ordinary future tasks. Historical results,
including failed attempts, remain unchanged. The design target is about 1 GiB
of results per machine with a 2 GiB planning allowance; measured validation and
its limitations are recorded separately. This is not a storage cap or permission
to discard samples. System files, builds and temporary export space are additional.

| Study | Retained measurements | Deliberately omitted from future recordings |
| --- | --- | --- |
| Reaching/tracking, square, closed-loop batching | Every executed state/control and plant substep; feedback/goal observations; every wall/native timing, update interval, SQP/PCG work, quality/selection flag; outcomes and failures | Routine warm-start and predicted trajectory arrays; repeated native KKT matrices and nested solver input/output copies |
| Force disturbance tasks | All of the above plus force truth, estimates, hypothesis predictions/errors/selection, force-algorithm intervals and resources | Routine optimizer horizons and repeated prediction/optimizer request arrays |
| Sustained speed, first response and stage profiles | Every timing sample and native work/quality/profile, warmup label, failure, input-bank/candidate identity | Repeated frozen-bank inputs and optimizer matrices; matrix finiteness/size summaries remain |
| Cold/warm and batching timing comparisons; force hypothesis correspondence | Full required matched inputs/outputs, diagnostics, timings and labels | No comparison data removed |
| Resource queries and Gazebo fault trials | Existing resource/status and fault/feedback/handback observations | No new thinning |

Frozen initial conditions, references/goals, seeds, settings and source/build
identities are retained once per applicable protocol/input family. Dynamic
measurements remain recorded at every observation; no downsampling, new gates,
parameter changes or solver changes are part of this work. Missing/nonfinite
and failed observations are not converted into successful outcomes.

Adapters project payloads before passing them to the existing bounded chunk
writer. Finite native diagnostic arrays of 1–32 elements use shape-preserving
numeric lists to avoid tiny NPZ entry headers; their values are unchanged, but
per-call NumPy dtype is no longer part of that representation. Large, empty and
nonfinite arrays remain NPZ. Rectangular numeric final-feedback histories use
arrays rather than duplicated verbose JSON lists. Metric computation still receives the original in-memory data. Final
controller summaries reference already recorded histories and retain unique
feedback arrays, rather than copying complete histories again. The shared
controller-summary loader accepts both historical JSON and current envelopes.
The existing final tar.gz export compresses finished results; build caches stay
outside it. Recording still uses CPU/RAM and can indirectly affect solve timing
and mode-B trajectories; this change does not claim timing equivalence. The
sustained-speed loop also includes recording overhead, so its completed call
count can change and must retain the recorder version.

The reduced task records no longer support exact replay of every optimizer call
or arbitrary future analyses of omitted predicted horizons. Complete executed
motion, timing and study-specific data remain available for the declared task
plots and metrics. Matched-input studies retain their replay/comparison data.
Normal exits and handled failures flush chunks; an abrupt kill can lose the last
bounded unwritten chunk. No asynchronous recording framework was added.

## Saved-data check (GTX 1070 recordings, 2026-10-12)

Implementation `a83bf0c` passed 58 focused CPU checks. Three saved cases were
converted separately without changing historical data or running the GPU.

| Saved example | Historical data | Reduced working data | gzip-level1 export |
| --- | ---: | ---: | ---: |
| 24 s figure-eight tracking | 2.79 GiB | 309 MiB | 48.7 MiB |
| Adapted-force tracking, 10 N | 4.93 GiB | 359 MiB | 66.8 MiB |
| B16 matched-input timing | 9.26 MiB | 9.26 MiB | 6.26 MiB |

Tracking retained all 48,320 calls and 48,329 plant steps; force retained all
20,745 calls and 20,762 plant steps. Readback preserved motion, timing/work and
projected measurements. Recomputed tracking errors and physical-limit checks
passed; the force task's existing failure label was retained. The timing case
keeps its full comparison data. Figures and machine sizing are in the local
`thesis_results/compact-recording-validation-20261012/` report; the extraction
script is `experiments/analysis/compact_recording_validation.py`.

These measure storage from existing desktop calls, not future hardware call
counts or speedups. The complete cold/warm timing dataset alone is 1.53 GiB
uncompressed, so the 1 GiB working-data target cannot apply to every machine
without removing required comparison evidence. Keep that evidence.

Illustrative full-plan estimates using saved desktop call counts (not target
measurements or bounds):

| Machine | Working results | Return archive | Results + archive |
| --- | ---: | ---: | ---: |
| RTX 5060 internal, 89 cases | 1.82 GiB | 0.36 GiB | 2.19 GiB |
| Jetson internal, 84 cases | 4.53 GiB | 0.78 GiB | 5.31 GiB |
| Gazebo, 23 cases | 0.98 GiB | 0.17 GiB | 1.15 GiB |

Allow additional space for system/builds, extra attempts and an old archive
during replacement (about 2.55/6.09/1.32 GiB respectively for results plus two
archives). Target call counts can differ. The 1 GiB raw/2 GiB peak goals are not
achieved for RTX/Jetson; no required samples were removed to force them to fit.
