#!/usr/bin/env python3
"""Issue #19 follow-up on one GPU, at several horizons.

Two studies on example 15's problem (indy7, dt 0.01, max_sqp_iters 8, a cold solve
from the hold-at-x seed, then a warm solve seeded with the cold solution), run as
one queue in a freshly shuffled order per repetition, every run its own process:

  merit block size  pcg with merit blocks of each --threads, horizons --merit-horizons
  linear system     pcg, bdsv and bdsv_first (direct on the first SQP iteration, pcg
                    after) at the default block size, horizons --horizons

Builds one module per (block size, horizon) that the queue needs, one target at a
time, keeps them under <out>/modules/ and installs the right one before each run. A
module that fails to build is recorded and its configurations are dropped. The
default block size is built with GATO_MERIT_THREADS unset, so its compile flags also
show what the release picks on this GPU. The default modules and the CMake cache
are restored at the end. Stored modules are reused when the script is run again
with a new --out after a crash (copy <out>/modules/ over).

Run from the root of the GATO checkout with its venv active:
    python issue19/bench_horizons.py --dry-run     # print the plan only
    python issue19/bench_horizons.py               # build, run, report
    python issue19/bench_horizons.py --report-only # statistics from <out>/rows.jsonl
"""
import argparse
import glob
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

MODULE_DIR = Path("python/gato")
STAGES = ["initial_merit", "kkt", "schur", "linsys", "merit", "line_search"]
METRICS = ["wall"] + STAGES + ["host"]
MODES = ("pcg", "bdsv", "bdsv_first")

CHILD = r'''
import json, sys
import numpy as np
import gato
from gato.interface import STAGE_NAMES
mode, B, N = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
plant, DT = "indy7", 0.01
if (plant, N) not in gato.available():
    raise SystemExit(f"no {plant} N{N} module: {sorted(gato.available())}")
# example 15's problem, with the horizon and the linear-system path as parameters
solver = gato.BSQP(str(gato.robot_urdf(plant)), batch_size=B, N=N, dt=DT, plant_type=plant,
                   params=gato.SolverParams(max_sqp_iters=8, linsys=mode))
x = np.zeros((B, solver.nx), np.float32)
goals = np.zeros((B, N * 6), np.float32)
goals[:, 0::6], goals[:, 2::6] = 0.35, 0.5

def record(res):
    st = res.stats
    stage = st.stage_times_us.sum(axis=0)
    pcg = np.asarray(st.pcg_iters).reshape(-1, B)
    out = dict(wall=float(res.solve_time_us), initial_merit=float(st.initial_merit_time_us),
               iters=int(np.max(st.sqp_iters)), linsys_path=st.linsys, fused=bool(st.merit_fused),
               pcg_max=pcg.max(axis=1).tolist(), pcg_min=pcg.min(axis=1).tolist(),
               final_merit=float(st.final_merit[0]), start_merit=float(st.initial_merit[0]),
               step_size=np.asarray(st.step_size).reshape(-1, B)[:, 0].tolist(),
               kkt_converged=int(np.sum(st.kkt_converged)), xu0=res.xu_b(0).astype(float).tolist())
    out.update({n: float(stage[k]) for k, n in enumerate(STAGE_NAMES)})
    out["host"] = max(out["wall"] - out["initial_merit"] - float(stage.sum()), 0.0)
    return out

cold = solver.solve(x, goals)
warm = solver.solve(x, goals, xu_warm=cold.xu)
print(json.dumps(dict(cold=record(cold), warm=record(warm))))
'''


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def label(c):
    return f"mt{c['threads']} {c['linsys']} N{c['N']} B{c['B']}"


def key(c):
    return (c["threads"], c["linsys"], c["N"], c["B"])


def gpu_freq():
    """Current GPU clock in Hz from devfreq, or None where /sys does not expose it."""
    for d in sorted(glob.glob("/sys/class/devfreq/*")):
        if "gpu" in d or "ga10b" in d:
            try:
                return int(Path(d, "cur_freq").read_text().strip())
            except (OSError, ValueError):
                return None
    return None


def module_path(N, required=True):
    """The indy7 N<N> default-variant module, whatever its extension suffix."""
    pattern = re.compile(rf"^bsqpN{N}_indy7(\.[\w.-]+)?\.so$")
    found = sorted(p for p in MODULE_DIR.iterdir() if pattern.match(p.name))
    if len(found) > 1:
        raise SystemExit(f"several indy7 N{N} modules in {MODULE_DIR}: {[p.name for p in found]}")
    if not found and required:
        raise SystemExit(f"no indy7 N{N} module in {MODULE_DIR}")
    return found[0] if found else None


def flags_threads(N):
    flags = Path(f"build/CMakeFiles/bsqpN{N}_indy7.dir/flags.make")
    if not flags.exists():
        raise SystemExit(f"{flags} not found: cannot verify the merit block size of the N{N} build")
    return sorted(set(re.findall(r"GATO_MERIT_THREADS=(\d+)", flags.read_text())))


def stored(out, T, N):
    return out / "modules" / f"mt{T}_N{N}.so"


def build_all(out, needs, default, arch, jobs):
    """needs: {threads: [horizons]}. Builds the non-default block sizes first and the
    default (GATO_MERIT_THREADS unset) last, one module target at a time; reuses stored
    modules. Returns the module digests, the verified compile flags and the failed builds."""
    cmake = shutil.which("cmake")
    print(f"cmake: {cmake} ({subprocess.run([cmake, '--version'], capture_output=True, text=True).stdout.split()[2]})")
    pb11 = subprocess.run([sys.executable, "-m", "pybind11", "--cmakedir"], capture_output=True, text=True,
                          check=True).stdout.strip()
    base = [cmake, "-S", ".", "-B", "build", "-DCMAKE_BUILD_TYPE=Release", f"-DPython3_EXECUTABLE={sys.executable}",
            f"-Dpybind11_DIR={pb11}", f"-DCMAKE_CUDA_ARCHITECTURES={arch}", "-DGATO_BUILD_DEMO=OFF",
            "-DGATO_CONTACT_FORCES=OFF", "-DGATO_EXACT_HESSIAN=OFF", "-DGATO_RECEIPT_PROFILE=OFF"]
    (out / "modules").mkdir(parents=True, exist_ok=True)
    resolved, failed = {}, {}
    for T in sorted(needs, key=lambda t: t == default):
        todo = [N for N in needs[T] if not stored(out, T, N).exists()]
        reused = [N for N in needs[T] if N not in todo]
        if reused:
            print(f"mt{T}: reusing the stored modules for N {reused}", flush=True)
        if not todo:
            continue
        value = "" if T == default else str(T)
        log = out / f"build_mt{T}.log"
        with open(log, "a") as f:
            subprocess.run(base + [f"-DMODULES=indy7:{','.join(map(str, todo))}", f"-DGATO_MERIT_THREADS={value}"],
                           stdout=f, stderr=subprocess.STDOUT, check=True)
            for N in todo:
                print(f"building mt{T} N{N} (log: {log}) ...", flush=True)
                t0 = time.time()
                r = subprocess.run([cmake, "--build", "build", "--target", f"bsqpN{N}_indy7", "--parallel", str(jobs)],
                                   stdout=f, stderr=subprocess.STDOUT)
                if r.returncode:
                    failed[f"mt{T}_N{N}"] = f"build exited {r.returncode} after {(time.time() - t0) / 60:.1f} min (see {log})"
                    print(f"  FAILED: {failed[f'mt{T}_N{N}']}", flush=True)
                    continue
                found = flags_threads(N)
                if found != [str(T)] and not (T == 64 and found == []):
                    raise SystemExit(f"mt{T} N{N}: compile flags carry GATO_MERIT_THREADS={found}")
                resolved[f"mt{T}_N{N}"] = found or ["64 (kernel default)"]
                shutil.copy2(module_path(N), stored(out, T, N))
                print(f"  done in {(time.time() - t0) / 60:.1f} min, flags {resolved[f'mt{T}_N{N}']}", flush=True)
    # leave the CMake cache on the release default for the default horizons (no rebuild here)
    built_default = [N for N in needs.get(default, []) if stored(out, default, N).exists()]
    if built_default:
        subprocess.run(base + [f"-DMODULES=indy7:{','.join(map(str, built_default))}", "-DGATO_MERIT_THREADS="],
                       stdout=subprocess.DEVNULL, check=True)
    digests = {f"mt{T}_N{N}": sha256(stored(out, T, N)) for T, Ns in needs.items() for N in Ns
               if stored(out, T, N).exists()}
    for N in {N for Ns in needs.values() for N in Ns}:
        ds = [digests[f"mt{T}_N{N}"] for T in needs if f"mt{T}_N{N}" in digests]
        if len(set(ds)) != len(ds):
            raise SystemExit(f"N{N}: two block sizes produced the same module")
    return digests, resolved, failed


def install(src, target):
    tmp = target.with_name(target.name + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, target)


def run_all(out, configs, reps, seed, default):
    child = out / "child.py"
    child.write_text(CHILD)
    env = dict(os.environ)
    env.setdefault("GATO_CACHE_DIR", str(Path(".cache/gato").resolve()))
    env.pop("GATO_MERIT_FUSED", None)
    horizons = sorted({c["N"] for c in configs})
    targets = {N: module_path(N) for N in horizons}
    installed = {}
    total, done, t_start = reps * len(configs), 0, time.time()
    try:
        for rep in range(reps):
            order = list(configs)
            random.Random(seed * 1000 + rep).shuffle(order)
            for pos, c in enumerate(order):
                if installed.get(c["N"]) != c["threads"]:
                    install(stored(out, c["threads"], c["N"]), targets[c["N"]])
                    installed[c["N"]] = c["threads"]
                row = dict(c, rep=rep, pos=pos, t=time.time(), gpu_hz_before=gpu_freq())
                proc = subprocess.run([sys.executable, str(child), c["linsys"], str(c["B"]), str(c["N"])],
                                      capture_output=True, text=True, env=env, timeout=1800)
                row["gpu_hz_after"] = gpu_freq()
                try:
                    if proc.returncode:
                        row["stderr_tail"] = proc.stderr[-2000:]
                        raise ValueError(f"exit {proc.returncode}: {(proc.stderr.strip().splitlines() or [''])[-1]}")
                    res = json.loads(proc.stdout.strip().splitlines()[-1])
                    for name in ("cold", "warm"):
                        if res[name]["linsys_path"] != c["linsys"]:
                            raise ValueError(f"{name}: ran linsys {res[name]['linsys_path']}")
                    row.update(res, ok=True)
                except (ValueError, IndexError, KeyError) as exc:
                    row.update(ok=False, error=str(exc), stdout_tail=proc.stdout[-800:])
                with open(out / "rows.jsonl", "a") as f:
                    f.write(json.dumps(row) + "\n")
                done += 1
                eta = (time.time() - t_start) / done * (total - done)
                if row["ok"]:
                    status = f"warm {row['warm']['wall'] / 1000:.1f} ms, pcg {row['warm']['pcg_max']}"
                else:
                    status = f"FAILED: {row['error'][:80]}"
                print(f"[{done}/{total}] rep {rep + 1} {label(c):<26} {status}  (eta {eta / 60:.0f} min)", flush=True)
    finally:
        for N, target in targets.items():
            if stored(out, default, N).exists():
                install(stored(out, default, N), target)
        print(f"restored the default modules (mt{default}) for N {horizons}", flush=True)


def stats(values):
    v = np.asarray(values, float)
    return {"n": v.size, "mean": v.mean(), "var": v.var(ddof=1) if v.size > 1 else float("nan"),
            "sd": v.std(ddof=1) if v.size > 1 else float("nan"), "min": v.min(), "max": v.max()}


def report(out):
    rows = [json.loads(l) for l in open(out / "rows.jsonl")]
    ok = [r for r in rows if r.get("ok")]
    meta = json.loads((out / "meta.json").read_text()) if (out / "meta.json").exists() else {}
    cap = meta.get("max_pcg_iters", 200)
    default = meta.get("default_threads", 32)
    keys = sorted({key(r) for r in rows}, key=lambda k: (k[2], k[3], MODES.index(k[1]), k[0]))
    sel = {k: [r for r in ok if key(r) == k] for k in keys}
    threads = sorted({k[0] for k in keys})
    batches = sorted({k[3] for k in keys})
    merit_Ns = sorted({k[2] for k in keys if k[0] != default})
    linsys_Ns = sorted({k[2] for k in keys if k[0] == default and k[1] != "pcg"})

    table = {}
    for k, rs in sel.items():
        for phase in ("cold", "warm"):
            if not rs:
                continue
            for m in METRICS:
                table[k + (phase, m)] = stats([r[phase][m] / 1000 for r in rs])
            table[k + (phase, "linsys_share")] = stats([100 * r[phase]["linsys"] / r[phase]["wall"] for r in rs])
            clocks = [r["gpu_hz_after"] / 1e6 for r in rs if r.get("gpu_hz_after")]
            if clocks:
                table[k + (phase, "clock")] = stats(clocks)

    def cell(k, phase, m, digits=2):
        s = table.get(k + (phase, m))
        return "–" if s is None else f"{s['mean']:.{digits}f} ± {s['sd']:.{digits}f}"

    def ratio(k, ref, phase, m):
        a, b = table.get(k + (phase, m)), table.get(ref + (phase, m))
        return "–" if a is None or b is None else f"{a['mean'] / b['mean']:.2f}"

    def same_plans(ka, kb, phase):
        a, b = sel.get(ka) or [], sel.get(kb) or []
        if not a or not b:
            return "–"
        return "yes" if a[0][phase]["xu0"] == b[0][phase]["xu0"] else "no"

    print(f"\n## Issue #19 follow-up at several horizons: {len(ok)} of {len(rows)} runs ok")
    print(f"GATO {meta.get('gato', '?')} ({meta.get('commit', '?')}), indy7, dt 0.01, 8 SQP iterations, "
          f"max_pcg_iters {cap}, {meta.get('reps', '?')} interleaved reps; the release default build resolves to "
          f"{meta.get('default_resolves_to', '?')} merit threads. Times in ms, mean ± sample sd over runs.\n")

    others = [T for T in threads if T != default]
    if merit_Ns and others:
        print(f"### Merit block size (pcg)\n")
        for phase in ("warm", "cold"):
            print(f"{phase.capitalize()} solve:\n")
            head = ([f"merit mt{T}" for T in threads] + [f"mt{T} / mt{default}" for T in others]
                    + [f"solve mt{T}" for T in threads] + [f"mt{T} / mt{default}" for T in others]
                    + [f"plans mt{T} = mt{default}" for T in others])
            print("| N | B | " + " | ".join(head) + " |")
            print("|---|---|" + "---|" * len(head))
            for N in merit_Ns:
                for B in batches:
                    ref = (default, "pcg", N, B)
                    cells = [cell((T, "pcg", N, B), phase, "merit") for T in threads]
                    cells += [ratio((T, "pcg", N, B), ref, phase, "merit") for T in others]
                    cells += [cell((T, "pcg", N, B), phase, "wall") for T in threads]
                    cells += [ratio((T, "pcg", N, B), ref, phase, "wall") for T in others]
                    cells += [same_plans((T, "pcg", N, B), ref, phase) for T in others]
                    print(f"| {N} | {B} | " + " | ".join(cells) + " |")
            print()

    if linsys_Ns:
        print(f"### Linear-system path (merit threads {default})\n")
        for phase in ("warm", "cold"):
            print(f"{phase.capitalize()} solve:\n")
            print("| N | B | linsys | n | wall | linsys | merit | kkt | schur | linsys % | wall / pcg | linsys / pcg | clock after, MHz |")
            print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            for N in linsys_Ns:
                for B in batches:
                    ref = (default, "pcg", N, B)
                    for m in MODES:
                        k = (default, m, N, B)
                        if k + (phase, "wall") in table:
                            print(f"| {N} | {B} | {m} | {table[k + (phase, 'wall')]['n']} | {cell(k, phase, 'wall')} | "
                                  f"{cell(k, phase, 'linsys')} | {cell(k, phase, 'merit')} | {cell(k, phase, 'kkt')} | "
                                  f"{cell(k, phase, 'schur')} | {cell(k, phase, 'linsys_share', 1)} | "
                                  f"{ratio(k, ref, phase, 'wall')} | {ratio(k, ref, phase, 'linsys')} | "
                                  f"{cell(k, phase, 'clock', 0)} |")
            print()

        print("### Linear-system work per SQP iteration\n")
        print(f"PCG iterations per SQP iteration (max over the batch; cap {cap}). On bdsv, 1 is one direct "
              "solve, 2 a skipped update (non-PD pivot), 0 converged at start.\n")
        print("| N | B | linsys | cold | warm | cap hits cold / warm | identical across runs |")
        print("|---|---|---|---|---|---|---|")
        for N in linsys_Ns:
            for B in batches:
                for m in MODES:
                    rs = sel.get((default, m, N, B)) or []
                    if not rs:
                        continue
                    cold, warm = rs[0]["cold"]["pcg_max"], rs[0]["warm"]["pcg_max"]
                    same = all(r["cold"]["pcg_max"] == cold and r["warm"]["pcg_max"] == warm for r in rs)
                    spread = all(r[p]["pcg_max"] == r[p]["pcg_min"] for r in rs for p in ("cold", "warm"))
                    hits = "–" if m == "bdsv" else f"{sum(v >= cap for v in cold)} / {sum(v >= cap for v in warm)}"
                    note = ("yes" if same else "no") + ("" if spread else " (batch entries differ)")
                    print(f"| {N} | {B} | {m} | {cold} | {warm} | {hits} | {note} |")
        print()

        print("### Do the paths return the same plan?\n")
        print("Final merit of batch entry 0, and the largest plan difference to pcg relative to the largest plan entry.\n")
        print("| N | B | solve | final merit pcg | bdsv | bdsv_first | plan diff bdsv | plan diff bdsv_first | runs bit-identical |")
        print("|---|---|---|---|---|---|---|---|---|")
        for N in linsys_Ns:
            for B in batches:
                for phase in ("cold", "warm"):
                    ref = sel.get((default, "pcg", N, B)) or []
                    merits, diffs, ident = [], [], []
                    for m in MODES:
                        rs = sel.get((default, m, N, B)) or []
                        merits.append(f"{rs[0][phase]['final_merit']:.6g}" if rs else "–")
                        if rs:
                            same = all(r[phase]["xu0"] == rs[0][phase]["xu0"] for r in rs)
                            ident.append(f"{m} {'yes' if same else 'no'}")
                        if m != "pcg":
                            if rs and ref:
                                a, b = np.asarray(ref[0][phase]["xu0"]), np.asarray(rs[0][phase]["xu0"])
                                diffs.append(f"{np.abs(a - b).max() / np.abs(a).max():.1e}")
                            else:
                                diffs.append("–")
                    print(f"| {N} | {B} | {phase} | " + " | ".join(merits) + " | " + " | ".join(diffs)
                          + f" | {', '.join(ident)} |")
        print()

    print("### Variances (ms²)\n")
    print("| config | cold wall | cold linsys | cold merit | warm wall | warm linsys | warm merit |")
    print("|---|---|---|---|---|---|---|")
    for k in keys:
        if k + ("warm", "wall") in table:
            vals = [table[k + (p, m)]["var"] for p in ("cold", "warm") for m in ("wall", "linsys", "merit")]
            c = dict(threads=k[0], linsys=k[1], N=k[2], B=k[3])
            print(f"| {label(c)} | " + " | ".join(f"{v:.4g}" for v in vals) + " |")
    print()

    freqs = [r[f] for r in ok for f in ("gpu_hz_before", "gpu_hz_after") if r.get(f)]
    if freqs:
        print(f"GPU clock samples (devfreq cur_freq): {min(freqs) / 1e6:.0f} to {max(freqs) / 1e6:.0f} MHz\n")
    for name, why in meta.get("build_failures", {}).items():
        print(f"BUILD FAILED {name}: {why}; its configurations were not run")
    for r in rows:
        if not r.get("ok"):
            print(f"FAILED rep {r['rep'] + 1} {label(r)}: {r['error']}")

    with open(out / "stats.csv", "w") as f:
        f.write("threads,linsys,N,B,phase,metric,unit,n,mean,sd,var,min,max\n")
        for (T, m, N, B, phase, metric), s in sorted(table.items()):
            unit = {"linsys_share": "percent", "clock": "MHz"}.get(metric, "ms")
            f.write(f"{T},{m},{N},{B},{phase},{metric},{unit},{s['n']},{s['mean']:.6g},{s['sd']:.6g},"
                    f"{s['var']:.6g},{s['min']:.6g},{s['max']:.6g}\n")
    print(f"stats: {out / 'stats.csv'}, raw rows: {out / 'rows.jsonl'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--threads", default="32,64", help="merit block sizes for the block-size study")
    ap.add_argument("--default-threads", type=int, default=32,
                    help="what the release picks on this GPU (0.1.6: 32 on sm_72/sm_87, 64 elsewhere); verified")
    ap.add_argument("--merit-horizons", default="64,128,256,512")
    ap.add_argument("--horizons", default="32,64,128,256,512", help="horizons of the linear-system study")
    ap.add_argument("--linsys", default="pcg,bdsv,bdsv_first")
    ap.add_argument("--batches", default="1,4,16")
    ap.add_argument("--arch", default="87")
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("JOBS", 2)))
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="issue19/results/horizons")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    if args.report_only:
        return report(out)
    if not Path("examples/15_solver_stats.py").exists() or not MODULE_DIR.exists():
        raise SystemExit("run this from the root of the GATO checkout")
    if ".venv" not in sys.executable:
        print(f"warning: {sys.executable} is not a .venv interpreter; is the checkout's venv active?", file=sys.stderr)
    ints = lambda s: [int(v) for v in s.split(",") if v]
    default, batches = args.default_threads, ints(args.batches)
    configs, seen = [], set()
    for N in ints(args.merit_horizons):
        for T in sorted(set(ints(args.threads)) | {default}):
            for B in batches:
                configs.append(dict(threads=T, linsys="pcg", N=N, B=B))
    for N in ints(args.horizons):
        for m in args.linsys.split(","):
            for B in batches:
                configs.append(dict(threads=default, linsys=m, N=N, B=B))
    configs = [c for c in configs if not (key(c) in seen or seen.add(key(c)))]
    needs = {}
    for c in configs:
        needs.setdefault(c["threads"], set()).add(c["N"])
    needs = {T: sorted(Ns) for T, Ns in needs.items()}
    print(f"{len(configs)} configurations x {args.reps} reps = {len(configs) * args.reps} runs, random order per rep "
          f"(seed {args.seed}); modules to build: " + ", ".join(f"mt{T} N{Ns}" for T, Ns in needs.items()))
    for c in configs:
        print(f"  {label(c)}")
    if args.dry_run:
        return
    if (out / "rows.jsonl").exists():
        raise SystemExit(f"{out / 'rows.jsonl'} exists: pass another --out or move it away")
    out.mkdir(parents=True, exist_ok=True)

    import gato
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    digests, resolved, failed = build_all(out, needs, default, args.arch, args.jobs)
    runnable = [c for c in configs if f"mt{c['threads']}_N{c['N']}" in digests]
    if len(runnable) < len(configs):
        print(f"{len(configs) - len(runnable)} configurations dropped: their module did not build", flush=True)
    configs = runnable
    if not configs:
        raise SystemExit("no module built")
    default_flags = sorted({",".join(v) for k, v in resolved.items() if k.startswith(f"mt{default}_")})
    meta = dict(gato=gato.__version__, commit=commit, reps=args.reps, seed=args.seed, configs=configs,
                default_threads=default, default_resolves_to=",".join(default_flags) or "(reused, see earlier run)",
                module_flags=resolved, module_sha256=digests, build_failures=failed,
                max_pcg_iters=gato.SolverParams().max_pcg_iters,
                arch=args.arch, python=sys.executable, started=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    run_all(out, configs, args.reps, args.seed, default)
    report(out)


if __name__ == "__main__":
    main()
