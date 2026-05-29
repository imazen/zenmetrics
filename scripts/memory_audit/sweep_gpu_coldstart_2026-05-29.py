#!/usr/bin/env python3
"""GPU cold-start wall sweep across the six -gpu metric crates (task #140).

Measures the fixed one-shot overhead a fresh process pays before any
per-pixel work — CUDA context init + cubecl kernel JIT/PTX load + first
host->device upload + first compute + readback — versus the warm
per-call wall of subsequent calls in the same process. This fixed cost
is what determines when a small one-shot image is faster on CPU than
GPU.

The per-crate driver is `examples/coldstart_one`, controlled by
WORKER_W / WORKER_H / WORKER_REPS env vars. It emits a single line:

  READY <score> client_ms=<f> new_ms=<f> first_compute_ms=<f>
        cold_total_ms=<f> warm_median_ms=<f> warm_all_ms=<csv>

  - client_ms        : CUDA context init (cubecl Backend::client)
  - new_ms           : metric setup / GPU buffer alloc
  - first_compute_ms : kernel JIT + first upload + first compute + readback
  - cold_total_ms    : client_ms + new_ms + first_compute_ms (true one-shot)
  - warm_median_ms   : median of WORKER_REPS warm calls

cold_first_call_ms (task term) = new_ms + first_compute_ms
coldstart_overhead_ms          = cold_total_ms - warm_median_ms

EACH cold sample runs in a FRESH process (cold = new CUDA context). We
take SAMPLES fresh processes per (crate, size) and report the MEDIAN of
each phase, plus the warm median (median of warm_median_ms across the
samples — warm itself is intra-process median of WORKER_REPS calls).

Usage:
    python3 scripts/memory_audit/sweep_gpu_coldstart_2026-05-29.py \
        --metrics all --sizes 512,1024,4mp \
        --out crates/../benchmarks/gpu_coldstart_2026-05-29.tsv
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

# (label, w, h). Cold fixed cost dominates at small sizes (the crossover
# regime). The upload component scales with size; the context+JIT part is
# ~size-independent. 16mp optional via --sizes.
SIZES = [
    ("512", 512, 512),
    ("1024", 1024, 1024),
    ("4mp", 2048, 2048),
    ("16mp", 4096, 4096),
]

# All six GPU metric crates have a coldstart_one example.
METRICS = [
    "butteraugli-gpu",
    "cvvdp-gpu",
    "ssim2-gpu",
    "dssim-gpu",
    "iwssim-gpu",
    "zensim-gpu",
]

READY_RE = re.compile(
    r"READY\s+(?P<score>\S+)\s+"
    r"client_ms=(?P<client>[\d.]+)\s+"
    r"new_ms=(?P<new>[\d.]+)\s+"
    r"first_compute_ms=(?P<first>[\d.]+)\s+"
    r"cold_total_ms=(?P<cold>[\d.]+)\s+"
    r"warm_median_ms=(?P<warm>[\d.]+)\s+"
    r"warm_all_ms=(?P<warm_all>\S+)"
)

CELL_TIMEOUT_S = 600.0


def nvidia_smi_info() -> dict:
    info = {}
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader", "--id=0"],
            text=True, timeout=5,
        ).strip()
        name, mem, drv = [s.strip() for s in out.split(",")]
        info["gpu_name"] = name
        info["gpu_mem_total"] = mem
        info["gpu_driver"] = drv
    except Exception as e:
        info["gpu_err"] = str(e)
    return info


def run_one(bin_path: Path, w: int, h: int, reps: int, env: dict,
            repo_root: Path) -> dict | None:
    """Run one fresh-process cold sample. Returns parsed phases or None."""
    e = {**env}
    e["WORKER_W"] = str(w)
    e["WORKER_H"] = str(h)
    e["WORKER_REPS"] = str(reps)
    try:
        proc = subprocess.run(
            [str(bin_path)], cwd=str(repo_root), env=e,
            capture_output=True, text=True, timeout=CELL_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        print("    TIMEOUT", flush=True)
        return None
    if proc.returncode != 0:
        print(f"    EXIT {proc.returncode}: {proc.stderr.strip()[-300:]}", flush=True)
        return None
    m = None
    for line in proc.stdout.splitlines():
        m = READY_RE.search(line)
        if m:
            break
    if not m:
        print(f"    NO READY LINE. stdout tail: {proc.stdout.strip()[-300:]}", flush=True)
        return None
    return {
        "score": m.group("score"),
        "client_ms": float(m.group("client")),
        "new_ms": float(m.group("new")),
        "first_compute_ms": float(m.group("first")),
        "cold_total_ms": float(m.group("cold")),
        "warm_median_ms": float(m.group("warm")),
    }


def build(repo_root: Path, crate: str, backend: str, env: dict) -> Path | None:
    print(f"  [build] {crate} ({backend}) ...", flush=True)
    cmd = ["cargo", "build", "--release", "--quiet", "--example",
           "coldstart_one", "-p", crate, "--no-default-features",
           "--features", backend]
    res = subprocess.run(cmd, cwd=str(repo_root), env=env,
                         capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  [build FAILED] rc={res.returncode}", flush=True)
        for ln in (res.stderr or "")[-1500:].splitlines()[-20:]:
            print(f"    {ln}", flush=True)
        return None
    # cargo overwrites this path per crate — measure-per-crate serially.
    return repo_root / "target" / "release" / "examples" / "coldstart_one"


def refresh_marker(repo_root: Path, agent: str, activity: str) -> None:
    ts = subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
                                 text=True).strip()
    (repo_root / ".workongoing").write_text(f"{ts} {agent} {activity}\n")


def med(xs: list[float]) -> float:
    return statistics.median(xs) if xs else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="all")
    ap.add_argument("--backends", default="cuda")
    ap.add_argument("--sizes", default="512,1024,4mp")
    ap.add_argument("--samples", type=int, default=5,
                    help="fresh-process cold samples per (crate, size)")
    ap.add_argument("--reps", type=int, default=10,
                    help="warm repeats per process (for warm_median)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--marker-agent", default="claude-gpu-coldstart")
    ap.add_argument("--disk-cache-state", default="warm",
                    help="annotation written to the TSV/notes for this run")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    env = {
        **os.environ,
        "PATH": f"/usr/local/cuda/bin:{os.environ.get('PATH', '')}",
        "LD_LIBRARY_PATH": f"/usr/local/cuda/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}",
        "CARGO_TERM_COLOR": "never",
    }

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    metric_names = (METRICS if args.metrics.strip() == "all"
                    else [m.strip() for m in args.metrics.split(",")])
    size_set = set(s.strip() for s in args.sizes.split(","))
    chosen_sizes = [(lbl, w, h) for (lbl, w, h) in SIZES if lbl in size_set]

    smi = nvidia_smi_info()
    host = os.uname().nodename
    started = subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
                                      text=True).strip()
    git_sha = "?"
    try:
        git_sha = subprocess.check_output(
            ["jj", "log", "-r", "@-", "-T", "commit_id.short()", "--no-graph"],
            cwd=str(repo_root), text=True, stderr=subprocess.DEVNULL,
        ).strip() or "?"
    except Exception:
        pass

    print(f"# gpu_coldstart 2026-05-29  git={git_sha} host={host} "
          f"gpu={smi.get('gpu_name','?')} samples={args.samples} reps={args.reps} "
          f"disk_cache={args.disk_cache_state}", flush=True)

    rows: list[dict] = []
    for backend in backends:
        for crate in metric_names:
            refresh_marker(repo_root, args.marker_agent,
                           f"building {crate} ({backend})")
            bin_path = build(repo_root, crate, backend, env)
            if not bin_path or not bin_path.exists():
                for (lbl, w, h) in chosen_sizes:
                    rows.append({
                        "crate": crate, "size_mp": lbl, "size_w": w,
                        "size_h": h, "backend": backend,
                        "cold_first_call_ms": "nan", "warm_per_call_ms": "nan",
                        "coldstart_overhead_ms": "nan",
                        "disk_cache_state": args.disk_cache_state,
                        "notes": "ERR_BUILD",
                    })
                continue

            for (lbl, w, h) in chosen_sizes:
                refresh_marker(repo_root, args.marker_agent,
                               f"{crate} {lbl} {backend} ({args.samples} cold samples)")
                print(f"==> {crate:16s} {lbl:>5s} ({w}x{h}) {backend} "
                      f"x{args.samples} fresh procs ...", flush=True)
                clients, news, firsts, colds, warms = [], [], [], [], []
                score = "nan"
                for s in range(args.samples):
                    r = run_one(bin_path, w, h, args.reps, env, repo_root)
                    if r is None:
                        continue
                    clients.append(r["client_ms"])
                    news.append(r["new_ms"])
                    firsts.append(r["first_compute_ms"])
                    colds.append(r["cold_total_ms"])
                    warms.append(r["warm_median_ms"])
                    score = r["score"]
                    print(f"    [{s}] client={r['client_ms']:.1f} new={r['new_ms']:.1f} "
                          f"first={r['first_compute_ms']:.1f} cold_total={r['cold_total_ms']:.1f} "
                          f"warm={r['warm_median_ms']:.3f}", flush=True)
                    time.sleep(0.2)  # let GPU settle between fresh contexts

                if not colds:
                    rows.append({
                        "crate": crate, "size_mp": lbl, "size_w": w,
                        "size_h": h, "backend": backend,
                        "cold_first_call_ms": "nan", "warm_per_call_ms": "nan",
                        "coldstart_overhead_ms": "nan",
                        "disk_cache_state": args.disk_cache_state,
                        "notes": "ALL_SAMPLES_FAILED",
                    })
                    continue

                client_med = med(clients)
                new_med = med(news)
                first_med = med(firsts)
                cold_total_med = med(colds)
                warm_med = med(warms)
                # cold_first_call (task def) = first score call wall, EXCLUDING
                # context-init (the per-metric work). cold_total includes it.
                cold_first_call = new_med + first_med
                overhead = cold_total_med - warm_med
                print(f"    => MEDIAN client={client_med:.1f} new={new_med:.1f} "
                      f"first={first_med:.1f} cold_total={cold_total_med:.1f} "
                      f"cold_first_call={cold_first_call:.1f} warm={warm_med:.3f} "
                      f"overhead={overhead:.1f} ms", flush=True)

                rows.append({
                    "crate": crate, "size_mp": lbl, "size_w": w, "size_h": h,
                    "backend": backend,
                    "cold_first_call_ms": f"{cold_first_call:.3f}",
                    "warm_per_call_ms": f"{warm_med:.3f}",
                    "coldstart_overhead_ms": f"{overhead:.3f}",
                    "client_init_ms": f"{client_med:.3f}",
                    "metric_new_ms": f"{new_med:.3f}",
                    "first_compute_ms": f"{first_med:.3f}",
                    "cold_total_ms": f"{cold_total_med:.3f}",
                    "n_samples": len(colds),
                    "score": score,
                    "disk_cache_state": args.disk_cache_state,
                    "notes": "",
                })

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "crate", "size_mp", "size_w", "size_h", "backend",
        "cold_first_call_ms", "warm_per_call_ms", "coldstart_overhead_ms",
        "client_init_ms", "metric_new_ms", "first_compute_ms",
        "cold_total_ms", "n_samples", "score", "disk_cache_state", "notes",
    ]
    # Merge with existing rows (drop matching crate/size/backend/disk_cache).
    existing: list[dict] = []
    if out.exists():
        fresh_keys = {(r["crate"], r["size_mp"], r["backend"],
                       r["disk_cache_state"]) for r in rows}
        with out.open() as f:
            for row in csv.DictReader(f, delimiter="\t"):
                key = (row.get("crate"), row.get("size_mp"),
                       row.get("backend"), row.get("disk_cache_state"))
                if key not in fresh_keys:
                    existing.append(row)
    all_rows = existing + rows
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    meta = out.with_suffix(".meta")
    with meta.open("w") as f:
        f.write(f"tool: scripts/memory_audit/sweep_gpu_coldstart_2026-05-29.py\n")
        f.write(f"task: 140\n")
        f.write(f"git_sha: {git_sha}\n")
        f.write(f"host: {host}\n")
        f.write(f"gpu_name: {smi.get('gpu_name','?')}\n")
        f.write(f"gpu_mem_total: {smi.get('gpu_mem_total','?')}\n")
        f.write(f"gpu_driver: {smi.get('gpu_driver','?')}\n")
        f.write(f"cuda_path: /usr/local/cuda\n")
        f.write(f"backends_run: {','.join(backends)}\n")
        f.write(f"sizes_run: {','.join(s[0] for s in chosen_sizes)}\n")
        f.write(f"metrics_run: {','.join(metric_names)}\n")
        f.write(f"samples_per_cell: {args.samples}\n")
        f.write(f"warm_reps_per_process: {args.reps}\n")
        f.write(f"disk_cache_state: {args.disk_cache_state}\n")
        f.write(f"started_utc: {started}\n")
        f.write("method: fresh-process-per-cold-sample (cold = new CUDA "
                "context); MEDIAN over n_samples processes per phase; "
                "warm_per_call = median of intra-process WORKER_REPS calls. "
                "Every timed call ends in a host readback (client.read_one "
                "inside reduce), forcing GPU sync.\n")
        f.write("command: see header\n")
    print(f"\nwrote {out} ({len(all_rows)} rows, +{len(rows)} this run)", flush=True)
    print(f"wrote {meta}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
