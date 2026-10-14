# H100 batching and general timing

This explicit queue contains **615 scientific workers**, in this order:

| Study | Workers | Runtime |
|---|---:|---|
| F01ab cold/warm budget/horizon timing | 64 | fa6f6be |
| F01c batching throughput and same16 serial comparison | 15 | fa6f6be |
| F05 first response and 120s sustained timing, stage OFF | 24 | 852b0cb |
| F05 missing N32 stage profiles: SQP1/PCG100, SQP10/PCG800, B1/B16 | 4 | 852b0cb |
| F04 N16/32/64/128 x B1/B16 compiled resource queries | 8 | 4e5301f |
| BB computation-paid closed-loop batching, 100 pairs x two arms | 200 | 84bad9a |
| BA optimizer/equal-cap comparison, 100 pairs x three arms | 300 | 84bad9a |

Frozen scientific cases, inputs, budgets and source pins come from
`successor/merged/gtx1070_desktop.json`, plus F04 from `launch/rtx5060.json`.
`h100_overnight_plan.py` makes only this selection and the documented H100
readiness changes; `h100_overnight.json` is the executable frozen plan.
No task-parameter, random-reaching, control, force, cylinder, Gazebo or robot
study runs. The completed H100 N32/N128 B1/B16 SQP5 stage profiles and saved
September figure-eight remain separate retained evidence and are excluded.

Native numerical sources stay unchanged. All builds use sm90 and two CPU jobs.
F01ab/F01c share one fa6 four-horizon build; BA/BB share one 84bad N32 build.
F05 OFF builds only its used N32 module; native flags remain unchanged. F05 ON
and F04 keep their original profile/query flags and separate source identities.

Felix waived new three-task/canonical task-success gates for this H100 queue.
Timing cells retain their existing warmups/native observations. BA/BB run their
original canonical B1/B16 arms, plus BA nominal16x B1, as execution-only checks.
The helper preserves actual task/physical negatives and continues; a setup,
incomplete, nonfinite or execution failure holds that dependent family.
There is no fabricated task pass or KKT gate. Independent scientific failures
retain all data while other independent cells continue. No automatic retries.

From the existing checkout root on the login node, with `.venv-h100` already
installed as described in `H100_QUICK_START.md`:

```sh
git pull --ff-only origin experiment/h100-stage-timing
sbatch experiments/h100_overnight.sbatch
squeue -u "$USER"
```

The regular `gpu_h100` job requests one GPU and six CPUs for four hours.
Keep the checkout unchanged while queued/running; the script does not pull or
install packages. It records Git/plan/device identities, verifies frozen
inputs/source refs, prints the explicit dry-run, and uses the existing runner.
GNU timeout interrupts the runner after 3h45m, reserving 15 minutes for cleanup
and collection. Exit124 then indicates the scheduling buffer, not a task result.
A hard scheduler kill can still interrupt collection; raw attempts remain.

Raw results resume at `.experiment-results/h100-overnight`, with shared builds
in `.experiment-cache/h100-overnight`. Re-submit the same script to continue
pending work: completed cases and failed/blocked cases are preserved and skipped.
An interrupted active case may get a new attempt; previous partial data remain.
Do not use `--retry-failed` to repeat scientific negatives. Logs are
`gato-overnight-<jobid>.out`; each normal/handled failure exit collects a distinct
`.experiment-work/h100-overnight-<jobid>-results.tar.gz`. Return the archive and
Slurm log. Keep independent raw backup copies before deleting target data.

The measured GTX proxy is about 2.5 h of case-receipt wall time: F01ab 6.3 min,
F01c 1.2 min, F05 OFF 16.4 min, BB 81.6 min and BA 42.8 min, plus profiles/queries,
builds and readiness. H100's actual duration is unmeasured; faster modeB solves
can increase feedback/CPU/recording work. Four hours is an allocation, not a
completion promise. F01ab/F01c retain full comparison data, measured at 1.53/
0.13 GiB on GTX. Historical full-recording BB/F05 OFF occupied 26.0/10.6 GiB;
current recording omits repeated horizons where already specified, but target
call counts and BA recordings still cost space. Allow roughly 40–60 GiB for raw,
return archives, build caches and extra attempts; this is a planning allowance,
not a sample-dropping cap. Check available project space before submitting.

After a hard interruption, collection alone (no GPU work) is available with:

```sh
source .venv-h100/bin/activate
python experiments/collect.py --output .experiment-results/h100-overnight --export .experiment-work/h100-overnight-recovery-results.tar.gz
```
