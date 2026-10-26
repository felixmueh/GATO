# Issue #19 timing on the Jetson AGX Orin

Measurement scripts for [A2R-Lab/GATO#19](https://github.com/A2R-Lab/GATO/issues/19).
Both run example 15's problem (indy7, dt 0.01, `max_sqp_iters` 8; a cold solve from
the hold-at-x seed, then a warm solve seeded with the cold solution). Every
configuration runs in its own process, and every repetition runs all
configurations in a freshly shuffled order (fixed seed).

| Script | What it measures | Status |
|---|---|---|
| `bench_merit_threads.py` | merit block size 32 / 64 / 256 and the fused launch at 64, N32, B1/B4/B16 | produced the 24 October tables on v0.1.5 (`4d5f8c2`), run from `results/` of that checkout |
| `bench_horizons.py` | (1) merit block size 32 vs 64 at N64–N512; (2) `pcg`, `bdsv` and `bdsv_first` at N32–N512, with the PCG iteration counts of every SQP iteration | follow-up to the 24 October comment, branch based on v0.1.6 |

## Run the follow-up on the Jetson

```bash
git clone --branch experiment/issue19-orin-horizons https://github.com/felixmueh/GATO.git gato-issue19
cd gato-issue19
./tools/install.sh --no-regen
source .venv/bin/activate
python issue19/bench_horizons.py --dry-run          # the queue and the modules to build
mkdir -p issue19/results
python issue19/bench_horizons.py 2>&1 | tee issue19/results/horizons.log
```

The script builds what the queue needs: the indy7 modules at N32, 64, 128, 256
and 512 with the release default block size (`GATO_MERIT_THREADS` unset; 0.1.6
resolves it to 32 on sm_87, and the script checks the compile flags), and at
N64 to N512 with 64 threads. It builds one module target at a time with
`GATO_BUILD_DEMO=OFF`. A module that fails to build, for example at a long
horizon, is reported and its configurations are dropped; a run that fails is
recorded as FAILED and the queue continues. At the end it restores the default
modules and the CMake cache.

Options: `--reps` (10), `--batches` (1,4,16), `--threads` (32,64),
`--merit-horizons` (64,128,256,512), `--horizons` (32,64,128,256,512),
`--linsys` (pcg,bdsv,bdsv_first), `--jobs` (2), `--arch` (87). With the defaults
the queue holds 57 configurations, 570 runs. The script prints build times and
an ETA as it goes. `--report-only` recomputes the tables from
`issue19/results/horizons/rows.jsonl`.

Optional check of the CMake floor fix (#23) with the system CMake 3.22,
configure only:

```bash
/usr/bin/cmake -S . -B build-floor -DCMAKE_CUDA_ARCHITECTURES=87 -DMODULES=indy7:32 \
  -DPython3_EXECUTABLE=$PWD/.venv/bin/python -Dpybind11_DIR=$(.venv/bin/python -m pybind11 --cmakedir) \
  2>&1 | grep -E "Merit block size|Generating done|Error"
rm -rf build-floor
```

## Output

The report is Markdown, ready for a GitHub comment:

- merit block size: merit and solve time per thread count, their ratio to the
  default, and whether the plans are bit-identical across block sizes
- linear-system path: wall, linsys, merit, kkt and schur times, the linsys share,
  the ratio to `pcg`, and the GPU clock after the runs
- the PCG iteration count of every SQP iteration (cold and warm) with the hits of
  the `max_pcg_iters` cap; on `bdsv`, 1 is a direct solve and 2 a skipped update
- whether the paths return the same plan (final merit, the largest plan
  difference to `pcg`, bit-identity across runs)
- variances (ms²) of wall, linsys and merit for every configuration

Raw rows, `stats.csv` (mean, sd, variance, min and max of every metric), the
module hashes and the verified compile flags stay in `issue19/results/horizons/`,
which git ignores.
