# Demo recordings for upcoming runs

The adapter retains the frozen scientific collector and selected runtime, but
uses `demo-chunk-refs-v1` recording. The format is named in `recording.json` and
terminal metrics. Historical per-event NPZ/JSON recordings remain unchanged and
readable.

Arrays are stored in bounded, uncompressed NPZ chunks (64 events or 16 MiB;
one oversized event is permitted). Repeated solve and plant-step lifecycle
records use exact backward references and changed fields. The recorder keeps
only the latest solve/step snapshots; references span at most 16 events. No
per-event hashes are written. Published chunks are hashed once for the manifest.
The recorder flushes before starting each solve, including its inputs. Handled
failures flush during finalization or context-manager close. A hard kill may lose
the current buffered chunk; earlier published chunks remain readable.

Use `recording.iter_events(attempt)` for complete reconstructed scientific
records, including inputs, outputs, all native solver statistics, substep states,
controls, lifecycle times, failures and nonfinite values. The decoder also reads
historical layouts. Small lifecycle records refer to earlier events; they are
not standalone full-payload JSON objects.

Use `recording.load_controller_stats(attempt)` for the feedback summary. Its
arrays live once in the final chunk; `controller_stats.json` is a small NPZ
envelope. Unique feedback fields, goals/outcomes, final-approach histories and
additional summary fields are retained. `raw_solves`/`raw_execution` point to
their lifecycle events. Four duplicated optimizer histories are omitted only
after comparing their values and scope against saved non-bootstrap solves:
`sqp_iters`, `pcg_iters`, `pcg_times_us`, `kkt_converged`. Their
`history_references` identify `solve_selected`, excluding `startup/cold`, and
the selected lane for scalar SQP counts. A mismatch retains the original field.
The demo visual analysis uses this loader; ordinary task/control recordings
retain their existing format. Collection/archive tools copy the raw files
without interpreting their layout.

Task metrics still receive the original complete in-memory stats. This patch
removes duplicate serialization and the unbounded recorder event list, but
does not redesign the controller's resident histories. It changes no solver,
task settings, timing boundaries, safety gates or physical motion. Callbacks run
outside the measured solve timer; their resource pressure can still affect
measured wall/native times and mode-B feedback. New recording cohorts must be
labelled; this is not a production timing equivalence certificate.

Bounded CPU fixture (100 synthetic N32-shaped calls, seed 20261012): all 700
non-summary events roundtrip exactly; rough-reaching metric equality and handled
failure coverage also pass. Total stored files decrease from 11,721,823 to
1,614,401 bytes (86.2%); NPZ count 701 to 101; event JSON 1,043,521 to 300,107
bytes; summary JSON 5,054,304 to 2,187 bytes. Single fixture recording wall times
were 1.093 and 0.195 seconds, respectively. These are CPU fixture observations,
not GPU/remote performance estimates or measured archive compression ratios.
Fixture source and receipt are under the local
`.experiment-work/demo-recording-cpu-20261012/` directory. Existing raw data was
not modified, and no campaign or GPU run was performed.
