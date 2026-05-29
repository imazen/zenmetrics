#!/usr/bin/env python3
"""In-process GPU warmth transition sweep (task #144).

MEASURE the in-process GPU warmth transitions a single long-lived warm
worker pays — the orchestrator runs mixed metrics through ONE
long-lived warm worker (single-warm-instance pool), so these are
real-deployment-relevant. This replaces statements previously INFERRED
from architecture with committed numbers.

Driver: `crates/zenmetrics-api/examples/inprocess_warmth`, controlled by
WARMTH_W / WARMTH_H / WARMTH_REPS env + `--child <scenario> <a> [b]`.
The driver emits one line per measured phase:

    RESULT\t<scenario>\t<metric_a>\t<metric_b>\t<phase>\t<ms>\t<n>\t<notes>

EACH sample runs in a FRESH process (cold = new CUDA context). We take
SAMPLES fresh processes per cell and report the MEDIAN of each phase.

The four scenarios (see the example's module docs):
  - q1q2 : Q1 cross-metric context sharing + Q2 per-metric kernel warmth.
           A then B in ONE process (the A->B sequence is the point).
  - q3   : Q3 new reference in warm_ref mode (setref1 / warm_call /
           setref2-with-different-pixels / newref_call).
  - q4   : Q4 full-mode, different ref every call.

Correctness: every timed *score* phase ends in a host readback inside
the opaque compute (returns a scalar -> GPU sync). Every timed
set_reference is followed by block_on(client.sync()). See the example.

Usage:
    python3 scripts/memory_audit/sweep_gpu_inprocess_warmth_2026-05-29.py \
        --sizes 512,16mp --samples 5 --reps 5 \
        --out benchmarks/gpu_inprocess_warmth_2026-05-29.tsv
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

# (label, w, h). 512 = where the fixed floor dominates and the effect is
# clearest; 16mp = production size.
SIZES = {
    "512": (512, 512),
    "1024": (1024, 1024),
    "4mp": (2048, 2048),
    "16mp": (4096, 4096),
}

# Q1 cross-metric orderings: prove the ~181 ms context init is paid once
# regardless of which metric is first. cvvdp = eager-alloc, ssim2 =
# eager-alloc, zensim = lazy-alloc; the spread is representative.
Q1Q2_ORDERINGS = [
    ("cvvdp", "ssim2"),   # ordering 1
    ("ssim2", "cvvdp"),   # ordering 2 (reverse — confirms once-per-process)
    ("cvvdp", "zensim"),  # eager -> lazy
    ("zensim", "ssim2"),  # lazy -> eager
]

# Q3 warm_ref-supporting metrics (all six support set_reference +
# compute_with_cached_reference via the umbrella).
Q3_METRICS = ["cvvdp", "ssim2", "butter", "dssim", "iwssim", "zensim"]

# Q4 full-mode different-ref-per-call: representative spread.
Q4_METRICS = ["cvvdp", "ssim2", "zensim"]

CELL_TIMEOUT_S = 600.0
BUILD_FEATURES = "cuda,all-metrics,cubecl-types,pixels"


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


def build(repo_root: Path, env: dict) -> Path | None:
    print(f"  [build] zenmetrics-api inprocess_warmth ({BUILD_FEATURES}) ...", flush=True)
    cmd = ["cargo", "build", "--release", "--quiet", "-p", "zenmetrics-api",
           "--no-default-features", "--features", BUILD_FEATURES,
           "--example", "inprocess_warmth"]
    res = subprocess.run(cmd, cwd=str(repo_root), env=env,
                         capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  [build FAILED] rc={res.returncode}", flush=True)
        for ln in (res.stderr or "")[-2000:].splitlines()[-25:]:
            print(f"    {ln}", flush=True)
        return None
    return repo_root / "target" / "release" / "examples" / "inprocess_warmth"


def run_one(bin_path: Path, scenario: str, a: str, b: str | None,
            w: int, h: int, reps: int, env: dict, repo_root: Path) -> list[dict]:
    """Run one fresh-process sample. Returns the parsed RESULT rows (one
    per phase) or [] on failure."""
    e = {**env, "WARMTH_W": str(w), "WARMTH_H": str(h), "WARMTH_REPS": str(reps)}
    cmd = [str(bin_path), "--child", scenario, a]
    if b is not None:
        cmd.append(b)
    try:
        proc = subprocess.run(cmd, cwd=str(repo_root), env=e,
                              capture_output=True, text=True, timeout=CELL_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        print("    TIMEOUT", flush=True)
        return []
    if proc.returncode != 0:
        print(f"    EXIT {proc.returncode}: {proc.stderr.strip()[-300:]}", flush=True)
        return []
    out = []
    for line in proc.stdout.splitlines():
        if not line.startswith("RESULT\t"):
            continue
        parts = line.split("\t")
        # RESULT, scenario, a, b, phase, ms, n, notes
        if len(parts) < 8:
            continue
        out.append({
            "scenario": parts[1],
            "metric_a": parts[2],
            "metric_b": parts[3],
            "phase": parts[4],
            "ms": float(parts[5]),
            "n_samples": int(parts[6]),
            "notes": parts[7],
        })
    if not out:
        print(f"    NO RESULT LINES. stdout tail: {proc.stdout.strip()[-300:]}", flush=True)
    return out


def refresh_marker(repo_root: Path, agent: str, activity: str) -> None:
    ts = subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
                                 text=True).strip()
    (repo_root / ".workongoing").write_text(f"{ts} {agent} {activity}\n")


def med(xs: list[float]) -> float:
    return statistics.median(xs) if xs else float("nan")


def aggregate(samples: list[list[dict]]) -> list[dict]:
    """Median each phase across fresh-process samples. Keyed by
    (scenario, metric_a, metric_b, phase). Keeps the per-process notes
    from the LAST sample (they carry the per-call 'all=' csv + score)."""
    by_key: dict[tuple, list[float]] = {}
    notes_by_key: dict[tuple, str] = {}
    n_by_key: dict[tuple, int] = {}
    order: list[tuple] = []
    for rows in samples:
        for r in rows:
            key = (r["scenario"], r["metric_a"], r["metric_b"], r["phase"])
            if key not in by_key:
                by_key[key] = []
                order.append(key)
            by_key[key].append(r["ms"])
            notes_by_key[key] = r["notes"]
            n_by_key[key] = r["n_samples"]
    agg = []
    for key in order:
        sc, ma, mb, ph = key
        vals = by_key[key]
        agg.append({
            "scenario": sc,
            "metric_a": ma,
            "metric_b": mb,
            "size": None,  # filled by caller
            "phase": ph,
            "ms_median": med(vals),
            "n_samples": n_by_key[key],
            "n_procs": len(vals),
            "notes": notes_by_key[key],
        })
    return agg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="512,16mp")
    ap.add_argument("--samples", type=int, default=5,
                    help="fresh-process samples per cell (median over these)")
    ap.add_argument("--reps", type=int, default=5,
                    help="warm repeats per process (per-call medians)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--marker-agent", default="claude-inprocess-warmth")
    ap.add_argument("--scenarios", default="q1q2,q3,q4",
                    help="comma list subset of q1q2,q3,q4")
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    env = {
        **os.environ,
        "PATH": f"/usr/local/cuda/bin:{os.environ.get('PATH', '')}",
        "LD_LIBRARY_PATH": f"/usr/local/cuda/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}",
        "CARGO_TERM_COLOR": "never",
    }

    chosen_sizes = [(lbl, *SIZES[lbl]) for lbl in
                    (s.strip() for s in args.sizes.split(",")) if lbl in SIZES]
    scenarios = set(s.strip() for s in args.scenarios.split(","))

    smi = nvidia_smi_info()
    host = os.uname().nodename
    started = subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
                                      text=True).strip()
    git_sha = "?"
    try:
        git_sha = subprocess.check_output(
            ["jj", "log", "-r", "@", "-T", "commit_id.short()", "--no-graph"],
            cwd=str(repo_root), text=True, stderr=subprocess.DEVNULL,
        ).strip() or "?"
    except Exception:
        pass

    print(f"# gpu_inprocess_warmth 2026-05-29  git={git_sha} host={host} "
          f"gpu={smi.get('gpu_name','?')} samples={args.samples} reps={args.reps}",
          flush=True)

    refresh_marker(repo_root, args.marker_agent, "building inprocess_warmth")
    if args.skip_build:
        bin_path = repo_root / "target" / "release" / "examples" / "inprocess_warmth"
    else:
        bin_path = build(repo_root, env)
    if not bin_path or not bin_path.exists():
        print("BUILD FAILED — aborting", flush=True)
        return 2

    # Build the cell list: (scenario, a, b, label, w, h).
    cells: list[tuple] = []
    for (lbl, w, h) in chosen_sizes:
        if "q1q2" in scenarios:
            for (a, b) in Q1Q2_ORDERINGS:
                cells.append(("q1q2", a, b, lbl, w, h))
        if "q3" in scenarios:
            for m in Q3_METRICS:
                cells.append(("q3", m, None, lbl, w, h))
        if "q4" in scenarios:
            for m in Q4_METRICS:
                cells.append(("q4", m, None, lbl, w, h))

    all_rows: list[dict] = []
    for (scenario, a, b, lbl, w, h) in cells:
        tag = f"{scenario} {a}{('->' + b) if b else ''} {lbl}"
        refresh_marker(repo_root, args.marker_agent,
                       f"{tag} ({args.samples} fresh procs)")
        print(f"==> {tag:30s} ({w}x{h}) x{args.samples} fresh procs ...", flush=True)
        samples: list[list[dict]] = []
        for s in range(args.samples):
            rows = run_one(bin_path, scenario, a, b, w, h, args.reps, env, repo_root)
            if rows:
                samples.append(rows)
                # Brief progress: show the headline phase per scenario.
                head = {r["phase"]: r["ms"] for r in rows}
                if scenario == "q1q2":
                    print(f"    [{s}] client_init={head.get('client_init', float('nan')):.1f} "
                          f"B_first={head.get('B_first_same_process', float('nan')):.1f} "
                          f"A_warm={head.get('A_warm', float('nan')):.3f} "
                          f"B_warm={head.get('B_warm', float('nan')):.3f}", flush=True)
                elif scenario == "q3":
                    print(f"    [{s}] setref1={head.get('setref1', float('nan')):.2f} "
                          f"warm_call={head.get('warm_call', float('nan')):.3f} "
                          f"setref2={head.get('setref2', float('nan')):.2f} "
                          f"newref_call={head.get('newref_call', float('nan')):.3f}", flush=True)
                else:  # q4
                    print(f"    [{s}] same_ref={head.get('fullmode_same_ref', float('nan')):.3f} "
                          f"diff_ref={head.get('fullmode_diff_ref', float('nan')):.3f}", flush=True)
            time.sleep(0.2)  # let GPU settle between fresh contexts

        if not samples:
            all_rows.append({
                "scenario": scenario, "metric_a": a, "metric_b": b or "-",
                "size": lbl, "phase": "ALL", "ms_median": float("nan"),
                "n_samples": 0, "n_procs": 0, "notes": "ALL_SAMPLES_FAILED",
            })
            continue

        agg = aggregate(samples)
        for row in agg:
            row["size"] = lbl
            all_rows.append(row)

    # Write TSV.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["scenario", "metric_a", "metric_b", "size", "phase",
            "ms_median", "n_samples", "n_procs", "notes"]
    with open(args.out, "w", newline="") as f:
        wri = csv.writer(f, delimiter="\t")
        wri.writerow(cols)
        for r in all_rows:
            wri.writerow([
                r["scenario"], r["metric_a"], r["metric_b"] or "-", r["size"],
                r["phase"],
                f"{r['ms_median']:.4f}" if r["ms_median"] == r["ms_median"] else "nan",
                r["n_samples"], r["n_procs"], r["notes"],
            ])
    print(f"\n# wrote {len(all_rows)} rows -> {args.out}", flush=True)

    # Write .meta provenance.
    finished = subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
                                        text=True).strip()
    meta = args.out.with_suffix(args.out.suffix + ".meta")
    with open(meta, "w") as f:
        f.write(f"# gpu_inprocess_warmth_2026-05-29.tsv provenance\n")
        f.write(f"task: 144 (in-process GPU warmth transitions)\n")
        f.write(f"git_change: {git_sha}\n")
        f.write(f"host: {host}\n")
        f.write(f"gpu: {smi.get('gpu_name','?')} mem_total={smi.get('gpu_mem_total','?')} "
                f"driver={smi.get('gpu_driver','?')}\n")
        f.write(f"started_utc: {started}\n")
        f.write(f"finished_utc: {finished}\n")
        f.write(f"sizes: {args.sizes}\n")
        f.write(f"scenarios: {args.scenarios}\n")
        f.write(f"samples_per_cell: {args.samples} (median over fresh processes)\n")
        f.write(f"reps_per_process: {args.reps} (warm/per-call medians)\n")
        f.write(f"build: cargo build --release -p zenmetrics-api "
                f"--no-default-features --features {BUILD_FEATURES} "
                f"--example inprocess_warmth\n")
        f.write(f"driver: crates/zenmetrics-api/examples/inprocess_warmth.rs\n")
        f.write(f"harness: scripts/memory_audit/sweep_gpu_inprocess_warmth_2026-05-29.py\n")
        f.write("sync: every timed score readback-syncs (scalar return); "
                "every timed set_reference followed by block_on(client.sync())\n")
        f.write("cold = fresh process per sample (new CUDA context). "
                "Q1's A->B sequence is WITHIN one process.\n")
        f.write("compare-against: benchmarks/gpu_coldstart_2026-05-29.tsv (#140) "
                "cold_total_ms column for the fresh-process baseline.\n")
    print(f"# wrote provenance -> {meta}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
