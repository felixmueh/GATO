#!/usr/bin/env python3
"""Issue #19 timing on one GPU: examples/15_solver_stats.py at several batch sizes,
merit block sizes and merit launches, interleaved in a random order per repetition.

Builds one indy7 N32 module per merit block size (GATO_MERIT_THREADS; 64 is the
default build), keeps them under <out>/modules/, and installs the one a run needs
as python/gato/bsqpN32_indy7.so before starting that run's process. Every run is a
fresh process (GATO_MERIT_FUSED is read once per process) executing example 15
unchanged except for its batch size. The default module is restored at the end.

Run from the root of the GATO checkout with its venv active:
    python results/bench_issue19.py                    # build, run 10 interleaved reps, report
    python results/bench_issue19.py --dry-run          # print the plan only
    python results/bench_issue19.py --skip-build       # reuse <out>/modules/mt*.so
    python results/bench_issue19.py --report-only      # statistics from <out>/rows.jsonl
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
# pybind11 keeps the extension suffix: bsqpN32_indy7.cpython-310-aarch64-linux-gnu.so
MODULE_RE = re.compile(r"^bsqpN32_indy7(\.[\w.-]+)?\.so$")
EXAMPLE = Path("examples/15_solver_stats.py")
BATCH_LINE = "B, DT = 4, 0.01"
FLAGS = Path("build/CMakeFiles/bsqpN32_indy7.dir/flags.make")
DEFAULT_THREADS = 64
STAGES = {"initial merit": "initial_merit", "kkt": "kkt", "schur": "schur", "linsys": "linsys",
          "merit": "merit", "line_search": "line_search", "host, syncs, launch gaps": "host"}
METRICS = ["wall"] + list(STAGES.values())

HEADER_RE = re.compile(r"^(\S+) N(\d+), (cold|warm) solve")
SUMMARY_RE = re.compile(r"^solve (\d+) us wall, (\d+) iteration\(s\), batch (\d+), linsys (\S+), merit launch (.+)$")
STAGE_RE = re.compile(r"^  (" + "|".join(re.escape(s) for s in STAGES) + r")\s+([\d.]+) us\s+([\d.]+) %$")
PCG_RE = re.compile(r"^per-iteration PCG iterations \(cold\): (\[.*\])$")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def label(c):
    return f"mt{c['threads']}{' fused' if c['fused'] else ''} B{c['B']}"


def gpu_freq():
    """Current GPU clock in Hz from devfreq, or None where /sys does not expose it."""
    for d in sorted(glob.glob("/sys/class/devfreq/*")):
        if "gpu" in d or "ga10b" in d:
            try:
                return int(Path(d, "cur_freq").read_text().strip())
            except (OSError, ValueError):
                return None
    return None


def parse(text):
    """Example 15's output -> {"cold": {...}, "warm": {...}, "pcg_iters_cold": [...]}."""
    phases, phase, pcg = {}, None, None
    for line in text.splitlines():
        if m := HEADER_RE.match(line):
            phase = m.group(3)
            phases[phase] = {"plant": m.group(1), "N": int(m.group(2))}
        elif (m := SUMMARY_RE.match(line)) and phase:
            phases[phase].update(wall=float(m.group(1)), iters=int(m.group(2)), B=int(m.group(3)),
                                 linsys_path=m.group(4), fused=m.group(5).startswith("fused"))
        elif (m := STAGE_RE.match(line)) and phase:
            phases[phase][STAGES[m.group(1)]] = float(m.group(2))
        elif m := PCG_RE.match(line):
            pcg = json.loads(m.group(1))
    for name in ("cold", "warm"):
        missing = [k for k in METRICS if k not in phases.get(name, {})]
        if missing:
            raise ValueError(f"{name} solve: missing {missing} in the example output")
    return {"cold": phases["cold"], "warm": phases["warm"], "pcg_iters_cold": pcg}


def batch_scripts(out, batches):
    src = EXAMPLE.read_text()
    if src.count(BATCH_LINE) != 1:
        raise SystemExit(f"{EXAMPLE}: expected exactly one '{BATCH_LINE}' line")
    scripts = {}
    for B in batches:
        path = out / f"ex15_B{B}.py"
        path.write_text(src.replace(BATCH_LINE, f"B, DT = {B}, 0.01"))
        scripts[B] = path
    return scripts


def module_path(required=True):
    """The indy7 N32 default-variant module in python/gato/, whatever its extension suffix."""
    found = sorted(p for p in MODULE_DIR.iterdir() if MODULE_RE.match(p.name))
    if len(found) > 1:
        raise SystemExit(f"several indy7 N32 modules in {MODULE_DIR}: {[p.name for p in found]}")
    if not found and required:
        raise SystemExit(f"no indy7 N32 module in {MODULE_DIR} (MODULES=\"indy7:32\" ./tools/build.sh)")
    return found[0] if found else None


def flags_threads():
    """GATO_MERIT_THREADS values in the module's compile flags ([] = the default build)."""
    if not FLAGS.exists():
        raise SystemExit(f"{FLAGS} not found: cannot verify the merit block size of a build")
    return sorted(set(re.findall(r"GATO_MERIT_THREADS=(\d+)", FLAGS.read_text())))


def build_modules(out, threads, jobs):
    cmake = shutil.which("cmake")
    version = subprocess.run([cmake, "--version"], capture_output=True, text=True, check=True).stdout.split()[2]
    if tuple(int(x) for x in version.split(".")[:2]) < (3, 25):
        raise SystemExit(f"cmake {version} at {cmake}: v0.1.5 needs >= 3.25 (activate the venv with the pip cmake)")
    mods = out / "modules"
    mods.mkdir(parents=True, exist_ok=True)
    current = module_path(required=False)
    if current and not (mods / "before.so").exists():
        shutil.copy2(current, mods / "before.so")
    todo = list(threads)
    # the installed module is the default build if the last configure was the default and the
    # module was linked after it: reuse it instead of rebuilding
    if (DEFAULT_THREADS in todo and current and not flags_threads()
            and current.stat().st_mtime >= FLAGS.stat().st_mtime):
        shutil.copy2(current, mods / f"mt{DEFAULT_THREADS}.so")
        todo.remove(DEFAULT_THREADS)
        print(f"mt{DEFAULT_THREADS}: reusing the current default build {current.name}", flush=True)
    for T in sorted(todo, key=lambda t: t == DEFAULT_THREADS):
        value = "" if T == DEFAULT_THREADS else str(T)
        log = out / f"build_mt{T}.log"
        print(f"building mt{T} (log: {log}) ...", flush=True)
        t0 = time.time()
        with open(log, "w") as f:
            subprocess.run([cmake, "-S", ".", "-B", "build", f"-DGATO_MERIT_THREADS={value}"],
                           stdout=f, stderr=subprocess.STDOUT, check=True)
            subprocess.run([cmake, "--build", "build", "--parallel", str(jobs)],
                           stdout=f, stderr=subprocess.STDOUT, check=True)
        found = flags_threads()
        if found != ([] if T == DEFAULT_THREADS else [str(T)]):
            raise SystemExit(f"mt{T}: compile flags carry GATO_MERIT_THREADS={found}")
        shutil.copy2(module_path(), mods / f"mt{T}.so")
        print(f"  done in {time.time() - t0:.0f} s, flags {found or ['(default)']}", flush=True)
    if flags_threads():
        # leave the CMake cache on the default, so a later tools/build.sh builds the default again
        subprocess.run([cmake, "-S", ".", "-B", "build", "-DGATO_MERIT_THREADS="],
                       stdout=subprocess.DEVNULL, check=True)
    digests = {T: sha256(mods / f"mt{T}.so") for T in threads}
    if len(set(digests.values())) != len(digests):
        raise SystemExit(f"two block sizes produced the same module: {digests}")
    return digests


def install(out, T, target):
    tmp = target.with_name(target.name + ".tmp")
    shutil.copy2(out / "modules" / f"mt{T}.so", tmp)
    os.replace(tmp, target)


def run_all(out, configs, scripts, reps, seed):
    rows_path = out / "rows.jsonl"
    env = dict(os.environ)
    env.setdefault("GATO_CACHE_DIR", str(Path(".cache/gato").resolve()))
    target = module_path()
    installed = None
    total, done = reps * len(configs), 0
    t_start = time.time()
    try:
        for rep in range(reps):
            order = list(configs)
            random.Random(seed * 1000 + rep).shuffle(order)
            for pos, c in enumerate(order):
                if installed != c["threads"]:
                    install(out, c["threads"], target)
                    installed = c["threads"]
                env["GATO_MERIT_FUSED"] = "1" if c["fused"] else "0"
                row = dict(c, rep=rep, pos=pos, t=time.time(), gpu_hz_before=gpu_freq())
                proc = subprocess.run([sys.executable, str(scripts[c["B"]])], capture_output=True, text=True,
                                      env=env, timeout=900)
                row["gpu_hz_after"] = gpu_freq()
                try:
                    if proc.returncode:
                        last = (proc.stderr.strip().splitlines() or [""])[-1]
                        row["stderr_tail"] = proc.stderr[-2000:]
                        raise ValueError(f"exit {proc.returncode}: {last}")
                    res = parse(proc.stdout)
                    for name in ("cold", "warm"):
                        p = res[name]
                        if (p["plant"], p["N"], p["B"], p["fused"]) != ("indy7", 32, c["B"], bool(c["fused"])):
                            raise ValueError(f"{name}: ran {p['plant']} N{p['N']} B{p['B']} fused={p['fused']}")
                    row.update(res, ok=True)
                except ValueError as exc:
                    row.update(ok=False, error=str(exc), stdout_tail=proc.stdout[-800:])
                with open(rows_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
                done += 1
                eta = (time.time() - t_start) / done * (total - done)
                status = f"warm {row['warm']['wall'] / 1000:.1f} ms" if row["ok"] else f"FAILED: {row['error'][:80]}"
                print(f"[{done}/{total}] rep {rep + 1} {label(c):<16} {status}  (eta {eta / 60:.0f} min)", flush=True)
    finally:
        install(out, DEFAULT_THREADS, target)
        print(f"restored the default module (mt{DEFAULT_THREADS}) as {target}", flush=True)


def stats(values):
    v = np.asarray(values, float)
    return {"n": v.size, "mean": v.mean(), "var": v.var(ddof=1) if v.size > 1 else float("nan"),
            "sd": v.std(ddof=1) if v.size > 1 else float("nan"), "min": v.min(), "max": v.max()}


def report(out):
    rows = [json.loads(l) for l in open(out / "rows.jsonl")]
    ok = [r for r in rows if r.get("ok")]
    meta = json.loads((out / "meta.json").read_text()) if (out / "meta.json").exists() else {}
    keys = sorted({(r["threads"], r["fused"], r["B"]) for r in rows}, key=lambda k: (k[2], k[1], k[0]))
    print(f"\n## Issue #19 timing: {len(ok)} of {len(rows)} runs ok"
          f" (GATO {meta.get('gato', '?')}, {meta.get('commit', '?')}, {meta.get('reps', '?')} interleaved reps)")
    print("Times in ms, mean ± sample sd over runs; variances in ms² below and in stats.csv.\n")

    table = {}   # (threads, fused, B, phase, metric) -> stats
    for k in keys:
        sel = [r for r in ok if (r["threads"], r["fused"], r["B"]) == k]
        if not sel:
            continue
        for phase in ("cold", "warm"):
            for m in METRICS:
                table[k + (phase, m)] = stats([r[phase][m] / 1000 for r in sel])
            table[k + (phase, "merit_share")] = stats([100 * r[phase]["merit"] / r[phase]["wall"] for r in sel])
            table[k + (phase, "merit_per_iter")] = stats([r[phase]["merit"] / 1000 / r[phase]["iters"] for r in sel])

    def cell(k, phase, m, digits=2):
        s = table.get(k + (phase, m))
        return "–" if s is None else f"{s['mean']:.{digits}f} ± {s['sd']:.{digits}f}"

    def ratio(k, phase, m):
        ref = table.get((DEFAULT_THREADS, 0, k[2], phase, m))
        s = table.get(k + (phase, m))
        return "–" if ref is None or s is None else f"{s['mean'] / ref['mean']:.2f}"

    for phase in ("warm", "cold"):
        print(f"### {phase.capitalize()} solve (8 SQP iterations)\n")
        print("| config | n | wall | merit | linsys | kkt | schur | merit % | merit / mt64 | wall / mt64 |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for k in keys:
            if k + (phase, "wall") not in table:
                continue
            c = dict(threads=k[0], fused=k[1], B=k[2])
            print(f"| {label(c)} | {table[k + (phase, 'wall')]['n']} | {cell(k, phase, 'wall')} | {cell(k, phase, 'merit')} "
                  f"| {cell(k, phase, 'linsys')} | {cell(k, phase, 'kkt')} | {cell(k, phase, 'schur')} "
                  f"| {cell(k, phase, 'merit_share', 1)} | {ratio(k, phase, 'merit')} | {ratio(k, phase, 'wall')} |")
        print()

    print("### Variances, warm solve (ms²)\n")
    print("| config | " + " | ".join(METRICS) + " |")
    print("|---|" + "---|" * len(METRICS))
    for k in keys:
        if k + ("warm", "wall") in table:
            c = dict(threads=k[0], fused=k[1], B=k[2])
            print(f"| {label(c)} | " + " | ".join(f"{table[k + ('warm', m)]['var']:.4g}" for m in METRICS) + " |")
    print()

    batches = sorted({k[2] for k in keys})
    if len(batches) > 1:
        lo, hi = batches[0], batches[-1]
        print(f"### Growth from B{lo} to B{hi}, warm solve (ratio of means)\n")
        print("| merit launch | merit | linsys | kkt | schur | wall |")
        print("|---|---|---|---|---|---|")
        for T, fused in sorted({(k[0], k[1]) for k in keys}, key=lambda x: (x[1], x[0])):
            a, b = (T, fused, lo, "warm"), (T, fused, hi, "warm")
            if a + ("wall",) in table and b + ("wall",) in table:
                g = [f"{table[b + (m,)]['mean'] / table[a + (m,)]['mean']:.2f}×"
                     for m in ("merit", "linsys", "kkt", "schur", "wall")]
                print(f"| mt{T}{' fused' if fused else ''} | " + " | ".join(g) + " |")
        print()

    freqs = [r[f] for r in ok for f in ("gpu_hz_before", "gpu_hz_after") if r.get(f)]
    if freqs:
        print(f"GPU clock samples (devfreq cur_freq): {min(freqs) / 1e6:.0f} to {max(freqs) / 1e6:.0f} MHz\n")
    failed = [r for r in rows if not r.get("ok")]
    for r in failed:
        print(f"FAILED rep {r['rep'] + 1} {label(r)}: {r['error']}")

    with open(out / "stats.csv", "w") as f:
        f.write("threads,fused,B,phase,metric,unit,n,mean,sd,var,min,max\n")
        for (T, fused, B, phase, m), s in sorted(table.items()):
            unit = "percent" if m == "merit_share" else "ms"
            f.write(f"{T},{fused},{B},{phase},{m},{unit},{s['n']},{s['mean']:.6g},{s['sd']:.6g},"
                    f"{s['var']:.6g},{s['min']:.6g},{s['max']:.6g}\n")
    print(f"stats: {out / 'stats.csv'}, raw rows: {out / 'rows.jsonl'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--threads", default="32,64,256", help="merit block sizes; 64 is the default build")
    ap.add_argument("--fused-threads", default="64", help="block sizes also run with GATO_MERIT_FUSED=1")
    ap.add_argument("--batches", default="1,4,16")
    ap.add_argument("--seed", type=int, default=19)
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("JOBS", 2)))
    ap.add_argument("--out", default="results/bench")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    if args.report_only:
        return report(out)
    if not EXAMPLE.exists() or not MODULE_DIR.exists():
        raise SystemExit("run this from the root of the GATO checkout")
    if ".venv" not in sys.executable:
        print(f"warning: {sys.executable} is not a .venv interpreter; is the checkout's venv active?", file=sys.stderr)
    threads = [int(t) for t in args.threads.split(",")]
    if DEFAULT_THREADS not in threads:
        raise SystemExit(f"--threads must include the default {DEFAULT_THREADS} (it is restored at the end)")
    fused_threads = [int(t) for t in args.fused_threads.split(",") if t]
    batches = [int(b) for b in args.batches.split(",")]
    configs = [dict(threads=T, fused=0, B=B) for T in threads for B in batches]
    configs += [dict(threads=T, fused=1, B=B) for T in fused_threads for B in batches]
    print(f"{len(configs)} configurations x {args.reps} reps = {len(configs) * args.reps} runs, "
          f"random order per rep (seed {args.seed}):")
    for c in configs:
        print(f"  {label(c)}")
    if args.dry_run:
        return
    if (out / "rows.jsonl").exists():
        raise SystemExit(f"{out / 'rows.jsonl'} exists: pass another --out or move it away")
    out.mkdir(parents=True, exist_ok=True)

    import gato
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    if args.skip_build:
        missing = [T for T in threads if not (out / "modules" / f"mt{T}.so").exists()]
        if missing:
            raise SystemExit(f"--skip-build: missing {[str(out / 'modules' / f'mt{T}.so') for T in missing]}")
        digests = {T: sha256(out / "modules" / f"mt{T}.so") for T in threads}
    else:
        digests = build_modules(out, threads, args.jobs)
    meta = dict(gato=gato.__version__, commit=commit, reps=args.reps, seed=args.seed, configs=configs,
                module_sha256={f"mt{T}": d for T, d in digests.items()}, python=sys.executable,
                started=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    scripts = batch_scripts(out, batches)
    run_all(out, configs, scripts, args.reps, args.seed)
    report(out)


if __name__ == "__main__":
    main()
