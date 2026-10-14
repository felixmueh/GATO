# Complete the N32/N128 by B1/B16 profile grid

Run only two new workers: N32/B16 and N128/B16, both SQP5/PCG100,
stage timingON, internal modeA, one excludedwarmup plus one measured call,
fresh solver/dual/rho per call. Reuse the two retained B1/SQP5 rows from
`/workspace/GATO/_expr_results/f05-gtx1070-n128-stage-20261014/`;
its N128/SQP1 result remains a separate supplement. No B1 reruns.

N32/B16 uses the exact existing F01c16candidate pool, with B1 its candidate0.
N128/B16 repeats the retained F01ab cold B1 state/reference/initial guess in
all16lanes, including the unrelatedwarmup. There is no diverse N128 bank.
Candidate0 and every duplicate lane are checked against retained B1 inputs
with CPU byte comparisons. This controls stage hardware workload; it is not
a solver-quality or input-matched N32-versus-N128 comparison.

Reuse the exact existing N32/N128 sm61 stageON modules at runtime
`852b0cbcfe6e56fe5337a3b6e5fe8e777f959b77` and the original build cache.
Both modules register `BSQP_16_float`. PCG dynamicshared memory depends on
N, not B: N128float requests40644bytes plus20staticbytes, within the GTX1070
49152byte block allowance, as in its completed B1 calls. No solver change,
new build, resource probe, equivalence study, or extra task episode is added.

```sh
cd /workspace/GATO-f05-n128-b16-stage-profile
/opt/gato-venv/bin/python experiments/resource_profiles_plan.py --gtx1070-b16-extension
/opt/gato-venv/bin/python experiments/run.py --plan experiments/f05_gtx1070_b16_stage.json --machine gtx1070 --python /opt/gato-venv/bin/python --cache /workspace/GATO-f05-n128-stage-profile/.experiment-cache/f05-n128 --output <fresh-timestamped-raw-root> --dry-run
/opt/gato-venv/bin/python experiments/run.py --plan experiments/f05_gtx1070_b16_stage.json --machine gtx1070 --python /opt/gato-venv/bin/python --cache /workspace/GATO-f05-n128-stage-profile/.experiment-cache/f05-n128 --output <fresh-timestamped-raw-root>
```

Check for occupied GPU immediately before execution and hold if occupied.
Each new worker gets one attempt, with failures retained and no retries.
Record the reused source/module/input hashes and each row's raw attempt identity.
The combined2x2 report keeps raw cohorts identifiable and retains the exact
historical `stage_windows.csv` columns. Main timing groups divide by actual
SQP iterations per batch, without dividing by B. Stream windows include idle
gaps; stageON overhead and single measured samples limit interpretation.
