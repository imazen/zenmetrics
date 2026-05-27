#!/usr/bin/env python3
"""GPU memory + timing audit across the six -gpu metric crates.

For each (crate, mode, size) cell:

1. Launch the crate's `mem_one_size` example as a child process. The
   example creates the metric pipeline, runs one warm-up
   compute_with_reference call, prints `READY <score> warm_ms=<ms>`,
   then sleeps for ~400 ms so the parent can sample nvidia-smi
   memory.used during a quiescent post-compute window.
2. Sample `nvidia-smi memory.used` for both the gpu's process row
   AND the system-wide total, before the child starts and during
   the hold window. Report the peak delta as the cell's VRAM cost.
3. Capture the warm-up wall-clock from the child's READY line — a
   single-shot time that includes set_reference + one
   compute_with_reference (i.e., the encoder hot-loop steady-state
   cost plus per-call overhead).

Subprocess-per-cell is mandatory because cubecl's memory pool keeps
GPU buffers cached across `Drop` for reuse by the next allocation in
the same process. The OS reclaims the pool on child exit, so the
next child sees a clean baseline.

Output: TSV with columns
    crate, mode, w, h, mp,
    baseline_mib, peak_mib, delta_mib,
    warm_ms, score,
    htod_per_iter_prior   (filled from manual nsys table; -1 if N/A)

The orchestrator does NOT run nsys itself — that's a manual follow-up
for the worst offenders (per the task brief). HtoD counts from prior
cross-crate nsys traces are baked in for context.

Usage:
    python3 scripts/memory_audit/audit_gpu_metrics.py \
        --out benchmarks/gpu_memory_audit_2026-05-27.csv

Tested on RTX 5070 / CUDA 13.2.1 / driver 596.21.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

# Crate × supported modes. Names match `cargo -p <crate>` and the
# WORKER_MODE env var the driver reads.
CRATES = [
    ("butteraugli-gpu", ["full", "strip"]),
    ("ssim2-gpu", ["full", "strip"]),
    ("dssim-gpu", ["full", "strip"]),
    ("iwssim-gpu", ["full", "strip"]),
    ("zensim-gpu", ["full"]),
    ("cvvdp-gpu", ["full", "strip_pair"]),
]

# Sizes per task brief (squared).
SIZES = [1024, 2048, 4096]

# HtoD/iter from the task #85 cross-crate report (12 MP). Used only
# for context — orchestrator does not measure HtoD count, that needs
# nsys. -1 means "not measured in the prior report".
HTOD_PRIOR_12MP = {
    ("zensim-gpu", "full"): 2,
    ("butteraugli-gpu", "full"): 12,
    ("iwssim-gpu", "full"): 26,
    ("dssim-gpu", "full"): 27,
    ("ssim2-gpu", "full"): 52,
    ("cvvdp-gpu", "full"): 208,  # pre-fix figure; post-fix expected ~12 (manual nsys follow-up)
}


def nvidia_smi_used_mib() -> int | None:
    """Return system-wide GPU memory.used (MiB), or None on failure."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            text=True,
            timeout=5,
        )
        return int(out.strip().splitlines()[0].strip())
    except Exception as e:
        print(f"  warn: nvidia-smi failed: {e}", file=sys.stderr)
        return None


def measure_cell(
    repo_root: Path,
    crate: str,
    mode: str,
    w: int,
    h: int,
    cuda_env: dict[str, str],
    hold_samples_s: float = 0.35,
) -> dict:
    """Run one (crate, mode, w, h) cell as a subprocess.

    Returns a dict with baseline_mib, peak_mib, delta_mib, warm_ms,
    score, exit_code, stderr_tail.
    """
    env = {**os.environ, **cuda_env}
    env["WORKER_MODE"] = mode
    env["WORKER_W"] = str(w)
    env["WORKER_H"] = str(h)
    env["CARGO_TERM_COLOR"] = "never"

    # Let any prior allocation settle.
    time.sleep(0.3)
    baseline = nvidia_smi_used_mib()
    if baseline is None:
        return {"error": "nvidia-smi baseline failed"}

    cmd = [
        "cargo",
        "run",
        "--release",
        "--quiet",
        "--example",
        "mem_one_size",
        "-p",
        crate,
        "--features",
        "cuda",
    ]
    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    # Read READY line, then sample nvidia-smi for hold window.
    ready_line: str | None = None
    deadline = time.time() + 120.0
    assert proc.stdout is not None
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            # child exited prematurely
            break
        line = line.strip()
        if line.startswith("READY "):
            ready_line = line
            break

    if ready_line is None:
        rc = proc.wait(timeout=10)
        stderr_tail = (proc.stderr.read() or "")[-1500:] if proc.stderr else ""
        return {
            "error": f"no READY (exit {rc})",
            "baseline_mib": baseline,
            "stderr_tail": stderr_tail,
        }

    # Sample memory while child holds the GPU.
    peak = baseline
    sample_until = time.time() + hold_samples_s
    while time.time() < sample_until:
        v = nvidia_smi_used_mib()
        if v is not None and v > peak:
            peak = v
        time.sleep(0.02)

    # Drain & wait.
    rc = proc.wait(timeout=10)
    stderr_tail = (proc.stderr.read() or "")[-1500:] if proc.stderr else ""

    # Parse READY line: "READY <score> warm_ms=<ms>"
    parts = ready_line.split()
    score = parts[1] if len(parts) > 1 else "nan"
    warm_ms = "nan"
    for p in parts[2:]:
        if p.startswith("warm_ms="):
            warm_ms = p.split("=", 1)[1]

    elapsed_s = time.time() - t0

    return {
        "baseline_mib": baseline,
        "peak_mib": peak,
        "delta_mib": peak - baseline,
        "warm_ms": warm_ms,
        "score": score,
        "exit_code": rc,
        "stderr_tail": stderr_tail,
        "wall_s": f"{elapsed_s:.2f}",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output CSV path (also writes .meta sidecar)",
    )
    parser.add_argument(
        "--sizes",
        default=",".join(str(s) for s in SIZES),
        help="comma-separated list of sizes (squared)",
    )
    parser.add_argument(
        "--crates",
        default=",".join(c for c, _ in CRATES),
        help="comma-separated list of crate names; default = all six",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    out_csv = args.out
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    cuda_env = {
        "PATH": f"/usr/local/cuda/bin:{os.environ.get('PATH', '')}",
        "LD_LIBRARY_PATH": f"/usr/local/cuda/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}",
    }

    sizes = [int(s) for s in args.sizes.split(",")]
    want_crates = set(args.crates.split(","))
    plan = [
        (crate, mode, sz)
        for (crate, modes) in CRATES
        if crate in want_crates
        for mode in modes
        for sz in sizes
    ]

    git_sha = "?"
    try:
        git_sha = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        pass

    print(
        f"# gpu_memory_audit — {len(plan)} cells — repo {git_sha} — sizes {sizes}",
        flush=True,
    )

    rows = []
    for crate, mode, sz in plan:
        w = h = sz
        mp = (w * h) / 1e6
        htod_prior = HTOD_PRIOR_12MP.get((crate, mode), -1)
        print(
            f"==> {crate:18s} mode={mode:11s}  {w}x{h}  ({mp:.1f} MP)  ...",
            flush=True,
        )
        r = measure_cell(repo_root, crate, mode, w, h, cuda_env)
        if "error" in r:
            print(f"    ERROR: {r.get('error')}", flush=True)
            if r.get("stderr_tail"):
                print("    stderr tail:", flush=True)
                for line in r["stderr_tail"].splitlines()[-20:]:
                    print(f"      {line}", flush=True)
            rows.append(
                {
                    "crate": crate,
                    "mode": mode,
                    "w": w,
                    "h": h,
                    "mp": f"{mp:.3f}",
                    "baseline_mib": r.get("baseline_mib", -1),
                    "peak_mib": -1,
                    "delta_mib": -1,
                    "warm_ms": "nan",
                    "score": "nan",
                    "htod_12mp_prior": htod_prior,
                    "wall_s": r.get("wall_s", "nan"),
                    "error": r["error"],
                }
            )
            continue
        print(
            f"    baseline={r['baseline_mib']} MiB  peak={r['peak_mib']} MiB  delta={r['delta_mib']:+} MiB  warm={r['warm_ms']} ms  score={r['score']}  wall={r['wall_s']} s",
            flush=True,
        )
        rows.append(
            {
                "crate": crate,
                "mode": mode,
                "w": w,
                "h": h,
                "mp": f"{mp:.3f}",
                "baseline_mib": r["baseline_mib"],
                "peak_mib": r["peak_mib"],
                "delta_mib": r["delta_mib"],
                "warm_ms": r["warm_ms"],
                "score": r["score"],
                "htod_12mp_prior": htod_prior,
                "wall_s": r["wall_s"],
                "error": "",
            }
        )

    fieldnames = [
        "crate",
        "mode",
        "w",
        "h",
        "mp",
        "baseline_mib",
        "peak_mib",
        "delta_mib",
        "warm_ms",
        "score",
        "htod_12mp_prior",
        "wall_s",
        "error",
    ]
    with out_csv.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)

    meta = out_csv.with_suffix(".meta")
    with meta.open("w") as f:
        f.write(f"# gpu_memory_audit metadata\n")
        f.write(f"git_sha: {git_sha}\n")
        f.write(f"sizes: {sizes}\n")
        f.write(f"crates: {sorted(want_crates)}\n")
        f.write(f"cells: {len(plan)}\n")
        f.write(f"host: {os.uname().nodename}\n")
        f.write(f"cuda_path: /usr/local/cuda\n")
        f.write(
            "method: subprocess-per-cell, nvidia-smi delta during ~400ms child hold window after READY.\n"
        )

    print(f"\nwrote {out_csv} + {meta}  ({len(rows)} rows)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
