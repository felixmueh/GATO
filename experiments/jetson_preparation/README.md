# Jetson overnight preparation

On the actual Orin/SM87 Jetson, use the existing working GATO container and Python environment (CUDA/compiler/Pinocchio already installed). Once the branch is available on your usual remote, fetch it and create a separate checkout:

```bash
git fetch origin experiment/jetson-f01-preparation
git worktree add ../GATO-jetson-preparation origin/experiment/jetson-f01-preparation
cd ../GATO-jetson-preparation
python3 -B experiments/jetson_preparation/prepare_jetson.py
```

Activate the existing GATO Python environment first, or use its executable in place of `python3`. The launcher defaults to this repository, that Python executable, and a fresh persistent `~/gato-jetson-f01ab-preparation-20261010` directory. To use another fresh path:

```bash
python3 -B experiments/jetson_preparation/prepare_jetson.py --work /persistent/fresh/preparation
```

`--check` checks tracked package integrity without building or using the GPU. `--help` shows path overrides. The branch must include runtime commit `502fa29c8b0d649dcb347f9534bf29c6935fa6dd` and science checkpoint `9cbf25721056e5b9dae1a5daf9e24982849b0573`; normal branch history contains both. No bundle or native binary is needed or tracked.

The command makes four sequential isolated Release/87-real native builds at N16/32/64/128 with the accepted original observer ON, then runs the unchanged N32 `make test-tiago-tasks` once. Stop other GPU jobs first. Builds may take much of the night; run in a persistent terminal/session. Logs, build receipts, actual module/source identities and task artifacts are saved under the preparation directory.

The suite uses the genuine source502 Git administrative directory against its identical archived502 sources for provenance. It snapshots the N32 CMake cache before and after Make, checks the pre-Make snapshot against the original builder receipt, and keeps module hashes before/after. The unchanged CMake target writes to `builds/N32/campaign-source/test-artifacts/tiago-tasks`, overriding the environment artifact path. The launcher preserves those originals and copies actual artifacts into `suite/raw/`, including failures. It does not rerun Make or overwrite builder receipts. Target review must reconcile any cache change.

**The command stops for actual target review.** Send `preparation.json`, `logs/`, `builds/N*/f01-build-N*.json`, `builds/N*/build_source_receipt.json`, `suite/CMakeCache.*.txt` and `suite/raw/` for review; retain the whole directory. A failed/interrupted preparation remains held. Do not delete evidence, retry, change parameters or skip tests to obtain a pass. The copied global authority records approve source instrumentation; they do not qualify Jetson builds, readiness or execution.

No canonical mode A preflight, target release or matrix run is claimed by this command. After actual target review, the coordinator supplies the existing guarded canonical-A preflight command and exact reviewed Jetson plan/dry-run. Do not use the legacy default runner queue or another machine's receipts.

The already approved F01ab short12 campaign remains N16/32/64/128 × SQP1/2/5/10 × PCG100/200/400/800, all 64 cells. Per cell: 12 cold, 222 warm, 6 goal-switch and one retained/excluded unrelated CUDA warmup (15,424 calls overall). Mode A .6 s timing subsets are separate from full T01 task evidence; public preflight PCG120 is distinct from matrix PCG100. No slow samples, failed cells or budgets are removed and no strict deadline gate applies. Assuming .1–1 s per solve gives 26 minutes–4.3 hours of solve time plus overhead; this is a scenario, not a measured Jetson forecast. Retained Orin cylinder timings at much larger N64/SQP200/PCG1000 cannot forecast this campaign. BA/BB and other studies are outside this preparation.
