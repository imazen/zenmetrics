#!/usr/bin/env python3
"""GPU VRAM + wall-time sweep across six -gpu metric crates (task #133).

Extends `audit_gpu_metrics.py` with:
  - 4 sizes (1 MP, 4 MP, 16 MP, 40 MP non-square 7680x5184)
  - All per-metric modes from the task brief (warm_ref / warm_ref_strip /
    cvvdp's 4 MemoryModes + warm-ref combinations / iwssim native-RGB)
  - cuda + wgpu backends per metric, capturing failures as data
  - 3-run median for wall_ms (1 warm + reps controlled by WORKER_REPS)
  - Per-metric TSV output at
    `crates/<metric>-gpu/benchmarks/gpu_vram_sweep_2026-05-28.tsv`
  - OOM captured as data (`peak_vram_human` = "OOM"); ERR_DISPATCH_CAP
    on wgpu captured similarly

TSV columns (per task brief):
  size_mp, mode, backend, peak_vram_bytes, peak_vram_human,
  wall_median_ms, score, notes

`.meta` sidecar records: git_sha, host, driver, GPU, command line.

Subprocess-per-cell is mandatory because cubecl's pool caches buffers
across Drop; only OS-level cleanup gives a clean baseline.

Usage:
    python3 scripts/memory_audit/sweep_gpu_metrics_2026-05-28.py \
        --metrics all --out-base benchmarks/gpu_metrics_sweep_2026-05-28
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ----------------------------------------------------------------------
# Configuration: per-metric mode lists per task brief.
# ----------------------------------------------------------------------

# Sizes per task brief.  40 MP = 7680 × 5184 (~39.8 MP) — common high-end
# camera output / 8K-ish near-square aspect.
# (label, w, h)
SIZES = [
    ("1mp", 1024, 1024),
    ("4mp", 2048, 2048),
    ("16mp", 4096, 4096),
    ("40mp", 7680, 5184),
]

# Per-metric mode lists. Each entry is the WORKER_MODE string.
# `wgpu_modes` is a subset known to compile/launch on wgpu (cuda is
# always the primary). Modes not in wgpu_modes will be skipped for the
# wgpu backend with notes="SKIPPED_WGPU".
METRICS = {
    # task #45: Full + Strip + warm_ref + warm_ref_strip
    "butteraugli-gpu": {
        "modes": ["full", "strip", "warm_ref", "warm_ref_strip"],
        "wgpu_modes": ["full", "strip", "warm_ref", "warm_ref_strip"],
    },
    # task #46: same as #45
    "ssim2-gpu": {
        "modes": ["full", "strip", "warm_ref", "warm_ref_strip"],
        "wgpu_modes": ["full", "strip", "warm_ref", "warm_ref_strip"],
    },
    # task #73: Mode E (warm_ref_strip in the new naming)
    "dssim-gpu": {
        "modes": ["full", "strip", "warm_ref", "warm_ref_strip"],
        "wgpu_modes": ["full", "strip", "warm_ref", "warm_ref_strip"],
    },
    # task #57: native RGB strip
    "iwssim-gpu": {
        "modes": [
            "full",
            "strip",
            "warm_ref",
            "warm_ref_strip",
            "rgb_full",
            "rgb_strip",
            "rgb_warm_ref_strip",
        ],
        "wgpu_modes": [
            "full",
            "strip",
            "warm_ref",
            "warm_ref_strip",
            "rgb_full",
            "rgb_strip",
            "rgb_warm_ref_strip",
        ],
    },
    # task #49 / #75: Strip + Mode E refinement
    "zensim-gpu": {
        "modes": ["full", "warm_ref", "strip", "warm_ref_strip"],
        "wgpu_modes": ["full", "warm_ref", "strip", "warm_ref_strip"],
    },
    # task #133: all 4 MemoryModes plus warm_ref combinations + capped + auto.
    # cvvdp `strip` (cold-ref, Mode E without warm) panics at runtime
    # (Mode E requires warm_reference first) — recorded as NOT_SUPPORTED.
    "cvvdp-gpu": {
        "modes": [
            "full",
            "warm_ref",
            "warm_ref_strip",
            "strip_pair",
            "capped",
            "auto",
        ],
        "wgpu_modes": [
            "full",
            "warm_ref",
            "warm_ref_strip",
            "strip_pair",
            "capped",
            "auto",
        ],
    },
}

CRATE_TSV_NAME = "gpu_vram_sweep_2026-05-28.tsv"

# nvidia-smi sampling cadence inside the hold window.
HOLD_SAMPLE_SECS = 0.40
HOLD_SAMPLE_INTERVAL = 0.020

# Per-cell timeout. 40 MP cvvdp Full warm can take 30+ s the first time
# the kernel cache builds — give plenty of headroom.
CELL_TIMEOUT_S = 600.0


def nvidia_smi_used_mib(gpu_id: int = 0) -> int | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                f"--id={gpu_id}",
            ],
            text=True,
            timeout=5,
        )
        return int(out.strip().splitlines()[0].strip())
    except Exception as e:
        print(f"  warn: nvidia-smi failed: {e}", file=sys.stderr)
        return None


def nvidia_smi_info() -> dict:
    info = {}
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
                "--id=0",
            ],
            text=True,
            timeout=5,
        ).strip()
        name, mem, drv = [s.strip() for s in out.split(",")]
        info["gpu_name"] = name
        info["gpu_mem_total"] = mem
        info["gpu_driver"] = drv
    except Exception as e:
        info["gpu_err"] = str(e)
    return info


def human_bytes(n: int) -> str:
    if n < 0:
        return f"-{human_bytes(-n)}"
    units = ["B", "KiB", "MiB", "GiB"]
    f = float(n)
    u = 0
    while f >= 1024.0 and u < len(units) - 1:
        f /= 1024.0
        u += 1
    return f"{f:.2f}{units[u]}"


def classify_stderr(stderr_tail: str, backend: str) -> str | None:
    """Map common failure patterns to canonical notes.  Returns the
    note string if a known pattern is detected, None otherwise."""
    s = stderr_tail.lower()
    if "ref_full_state must be some before band loop" in s:
        return "NOT_SUPPORTED:strip_requires_warm_ref"
    if "out of memory" in s or "cudaerrormemoryallocation" in s or "oom" in s:
        return "OOM"
    if "dispatch" in s and ("limit" in s or "exceeded" in s or "cap" in s):
        return "ERR_DISPATCH_CAP"
    if "buffer" in s and "alignment" in s:
        return "ERR_BUF_ALIGN"
    if "unsupported" in s or "not_supported" in s or "not yet implemented" in s:
        return "ERR_UNSUPPORTED"
    if backend == "wgpu" and (
        "no_adapter" in s or "no adapter" in s or "request_device" in s
    ):
        return "ERR_NO_WGPU_ADAPTER"
    return None


def measure_cell(
    repo_root: Path,
    crate: str,
    mode: str,
    w: int,
    h: int,
    backend: str,
    cuda_env: dict[str, str],
    binary_paths: dict[tuple[str, str], Path],
    reps: int = 2,
) -> dict:
    """One subprocess-per-cell measurement.

    Returns a dict with at minimum:
      baseline_mib, peak_mib, delta_mib,
      delta_bytes (delta_mib * 1<<20),
      wall_median_ms, score, exit_code,
      stderr_tail, error (note), warm_ms
    """
    env = {**os.environ, **cuda_env}
    env["WORKER_MODE"] = mode
    env["WORKER_W"] = str(w)
    env["WORKER_H"] = str(h)
    env["WORKER_REPS"] = str(reps)
    # Make wgpu adapter selection robust: prefer high-perf when running
    # on a Linux box with multiple adapters (integrated + discrete).
    env.setdefault("WGPU_POWER_PREF", "high")
    env["CARGO_TERM_COLOR"] = "never"

    bin_path = binary_paths.get((crate, backend))
    if bin_path is None or not bin_path.exists():
        return {
            "error": f"BIN_NOT_BUILT ({backend})",
            "baseline_mib": -1,
            "peak_mib": -1,
            "delta_mib": -1,
            "delta_bytes": -1,
            "wall_median_ms": float("nan"),
            "warm_ms": float("nan"),
            "score": "nan",
            "exit_code": -1,
            "stderr_tail": "",
        }

    # Let any prior allocation settle.
    time.sleep(0.3)
    baseline = nvidia_smi_used_mib()
    if baseline is None:
        return {"error": "nvidia-smi baseline failed"}

    cmd = [str(bin_path)]
    t0 = time.time()
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError as e:
        return {
            "error": f"BIN_LAUNCH_FAILED: {e}",
            "baseline_mib": baseline,
            "peak_mib": -1,
            "delta_mib": -1,
            "delta_bytes": -1,
            "wall_median_ms": float("nan"),
            "warm_ms": float("nan"),
            "score": "nan",
            "exit_code": -1,
            "stderr_tail": "",
        }

    # Sample DURING compute (in a thread) so we catch the actual peak,
    # not just the post-compute residual.  cubecl's pool releases some
    # transient scratch by the time READY hits stdout; for big images
    # the post-READY sample under-reports peak by GBs.
    import threading
    sample_stop = threading.Event()
    peak_holder = {"peak": baseline}

    def sampler():
        while not sample_stop.is_set():
            v = nvidia_smi_used_mib()
            if v is not None and v > peak_holder["peak"]:
                peak_holder["peak"] = v
            time.sleep(HOLD_SAMPLE_INTERVAL)

    sampler_thread = threading.Thread(target=sampler, daemon=True)
    sampler_thread.start()

    ready_line: str | None = None
    deadline = time.time() + CELL_TIMEOUT_S
    assert proc.stdout is not None
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            break
        line = line.strip()
        if line.startswith("READY "):
            ready_line = line
            break

    if ready_line is None:
        try:
            rc = proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = proc.wait(timeout=5)
        sample_stop.set()
        sampler_thread.join(timeout=2)
        stderr_tail = (proc.stderr.read() or "")[-3000:] if proc.stderr else ""
        note = classify_stderr(stderr_tail, backend) or f"no READY (exit {rc})"
        return {
            "error": note,
            "baseline_mib": baseline,
            "peak_mib": peak_holder["peak"],
            "delta_mib": peak_holder["peak"] - baseline,
            "delta_bytes": (peak_holder["peak"] - baseline) * (1 << 20),
            "wall_median_ms": float("nan"),
            "warm_ms": float("nan"),
            "score": "nan",
            "exit_code": rc,
            "stderr_tail": stderr_tail,
        }

    # Continue sampling during the post-READY hold window in case
    # any retained state pushes the peak higher.
    hold_until = time.time() + HOLD_SAMPLE_SECS
    while time.time() < hold_until:
        time.sleep(HOLD_SAMPLE_INTERVAL)
    sample_stop.set()
    sampler_thread.join(timeout=2)
    peak = peak_holder["peak"]

    rc = proc.wait(timeout=10)
    stderr_tail = (proc.stderr.read() or "")[-3000:] if proc.stderr else ""

    # Parse READY line:
    # "READY <score> warm_ms=<ms> wall_median_ms=<ms> warm_then_reps_ms=<csv>"
    score = "nan"
    warm_ms = "nan"
    wall_median = "nan"
    runs_csv = ""
    parts = ready_line.split()
    if len(parts) > 1:
        score = parts[1]
    for p in parts[2:]:
        if p.startswith("warm_ms="):
            warm_ms = p.split("=", 1)[1]
        elif p.startswith("wall_median_ms="):
            wall_median = p.split("=", 1)[1]
        elif p.startswith("warm_then_reps_ms="):
            runs_csv = p.split("=", 1)[1]

    elapsed_s = time.time() - t0
    delta_mib = peak - baseline
    delta_bytes = delta_mib * (1 << 20)

    return {
        "error": "",
        "baseline_mib": baseline,
        "peak_mib": peak,
        "delta_mib": delta_mib,
        "delta_bytes": delta_bytes,
        "wall_median_ms": wall_median,
        "warm_ms": warm_ms,
        "runs_csv": runs_csv,
        "score": score,
        "exit_code": rc,
        "stderr_tail": stderr_tail,
        "wall_s": f"{elapsed_s:.2f}",
    }


def find_binary(repo_root: Path, crate: str, backend: str) -> Path:
    return repo_root / "target" / "release" / "examples" / "mem_one_size"


def ensure_binary(
    repo_root: Path, crate: str, backend: str, cuda_env: dict[str, str]
) -> Path | None:
    """Build the mem_one_size example for `crate` with the given backend.

    Returns the (cargo-default-located) binary path, or None on build
    failure.  Note: cargo overwrites
    `target/release/examples/mem_one_size` per crate — so we MUST
    rebuild + measure-per-crate-and-backend serially before moving to
    the next.  Helper returns the path for clarity.
    """
    print(f"  [build] {crate} ({backend})  ...", flush=True)
    cmd = [
        "cargo",
        "build",
        "--release",
        "--quiet",
        "--example",
        "mem_one_size",
        "-p",
        crate,
        "--features",
        backend,
    ]
    env = {**os.environ, **cuda_env}
    res = subprocess.run(cmd, cwd=str(repo_root), env=env, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  [build FAILED]  rc={res.returncode}", flush=True)
        tail = (res.stderr or "")[-1500:]
        for ln in tail.splitlines()[-20:]:
            print(f"    {ln}", flush=True)
        return None
    return find_binary(repo_root, crate, backend)


def write_tsv(
    out_path: Path,
    rows: list[dict],
    meta: dict,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "size_mp",
        "size_w",
        "size_h",
        "mode",
        "backend",
        "peak_vram_bytes",
        "peak_vram_human",
        "wall_median_ms",
        "warm_ms",
        "score",
        "notes",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    metap = out_path.with_suffix(".meta")
    with metap.open("w") as f:
        for k, v in meta.items():
            f.write(f"{k}: {v}\n")


def refresh_marker(repo_root: Path, agent_id: str, activity: str) -> None:
    ts = subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True).strip()
    (repo_root / ".workongoing").write_text(f"{ts} {agent_id} {activity}\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-base", type=Path, required=True,
                    help="output base path (per-metric TSV at "
                         "crates/<metric>/benchmarks/gpu_vram_sweep_2026-05-28.tsv; "
                         "global summary at out_base.tsv)")
    ap.add_argument("--metrics", default="all",
                    help="comma-separated list of crate names or 'all'")
    ap.add_argument("--backends", default="cuda",
                    help="comma-separated cuda,wgpu")
    ap.add_argument("--sizes", default="1mp,4mp,16mp,40mp",
                    help="subset of size labels")
    ap.add_argument("--reps", type=int, default=2,
                    help="number of post-warm reps (median = median of these)")
    ap.add_argument("--marker-agent", default="claude-gpu-metrics-sweep-v2")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    cuda_env = {
        "PATH": f"/usr/local/cuda/bin:{os.environ.get('PATH', '')}",
        "LD_LIBRARY_PATH": f"/usr/local/cuda/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}",
    }

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    metrics_arg = (args.metrics or "all").strip()
    if metrics_arg == "all":
        metric_names = list(METRICS.keys())
    else:
        metric_names = [m.strip() for m in metrics_arg.split(",")]

    size_set = set(s.strip() for s in args.sizes.split(","))
    chosen_sizes = [(lbl, w, h) for (lbl, w, h) in SIZES if lbl in size_set]

    git_sha = "?"
    try:
        git_sha = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        pass
    if git_sha == "?":
        # jj workspace — no .git/ visible from here.
        try:
            out = subprocess.check_output(
                ["jj", "log", "-r", "@-", "-T", "commit_id.short()",
                 "--no-graph"],
                cwd=str(repo_root), text=True, stderr=subprocess.DEVNULL,
            ).strip()
            if out:
                git_sha = out
        except Exception:
            pass

    smi_info = nvidia_smi_info()
    host = os.uname().nodename
    started = subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True).strip()

    print(f"# sweep_gpu_metrics 2026-05-28", flush=True)
    print(f"# git_sha={git_sha} host={host} gpu={smi_info.get('gpu_name', '?')}", flush=True)
    print(f"# sizes={[s[0] for s in chosen_sizes]} backends={backends}", flush=True)

    global_rows = []
    for crate in metric_names:
        if crate not in METRICS:
            print(f"!! unknown metric: {crate}", flush=True)
            continue
        cfg = METRICS[crate]
        crate_rows: list[dict] = []
        for backend in backends:
            refresh_marker(
                repo_root,
                args.marker_agent,
                f"building {crate} ({backend})",
            )
            bin_path = ensure_binary(repo_root, crate, backend, cuda_env)
            binary_paths = {(crate, backend): bin_path} if bin_path else {}
            modes_for_backend = (
                cfg["modes"] if backend == "cuda" else cfg.get("wgpu_modes", [])
            )
            if not bin_path:
                # Build failed — emit a single error row per (mode, size).
                for mode in cfg["modes"]:
                    for (lbl, w, h) in chosen_sizes:
                        crate_rows.append(
                            {
                                "size_mp": lbl,
                                "size_w": w,
                                "size_h": h,
                                "mode": mode,
                                "backend": backend,
                                "peak_vram_bytes": -1,
                                "peak_vram_human": "ERR_BUILD",
                                "wall_median_ms": "nan",
                                "warm_ms": "nan",
                                "score": "nan",
                                "notes": "ERR_BUILD",
                            }
                        )
                continue
            for mode in cfg["modes"]:
                if backend == "wgpu" and mode not in modes_for_backend:
                    for (lbl, w, h) in chosen_sizes:
                        crate_rows.append(
                            {
                                "size_mp": lbl,
                                "size_w": w,
                                "size_h": h,
                                "mode": mode,
                                "backend": backend,
                                "peak_vram_bytes": -1,
                                "peak_vram_human": "SKIPPED_WGPU",
                                "wall_median_ms": "nan",
                                "warm_ms": "nan",
                                "score": "nan",
                                "notes": "SKIPPED_WGPU",
                            }
                        )
                    continue
                for (lbl, w, h) in chosen_sizes:
                    refresh_marker(
                        repo_root,
                        args.marker_agent,
                        f"{crate} {mode} {lbl} {backend}",
                    )
                    print(
                        f"==> {crate:16s} mode={mode:18s} {lbl:>4s} ({w}x{h})  {backend}  ...",
                        flush=True,
                    )
                    r = measure_cell(
                        repo_root, crate, mode, w, h, backend, cuda_env, binary_paths,
                        reps=args.reps,
                    )
                    note = r.get("error", "")
                    if note:
                        human = note if note in (
                            "OOM",
                            "ERR_DISPATCH_CAP",
                            "ERR_BUF_ALIGN",
                            "ERR_UNSUPPORTED",
                            "ERR_NO_WGPU_ADAPTER",
                        ) else "ERR"
                        print(f"    NOTE: {note}", flush=True)
                        if note.startswith("NOT_SUPPORTED"):
                            human = "NOT_SUPPORTED"
                        crate_rows.append(
                            {
                                "size_mp": lbl,
                                "size_w": w,
                                "size_h": h,
                                "mode": mode,
                                "backend": backend,
                                "peak_vram_bytes": r.get("delta_bytes", -1),
                                "peak_vram_human": human,
                                "wall_median_ms": r.get("wall_median_ms", "nan"),
                                "warm_ms": r.get("warm_ms", "nan"),
                                "score": r.get("score", "nan"),
                                "notes": note,
                            }
                        )
                        continue
                    print(
                        f"    baseline={r['baseline_mib']} MiB  peak={r['peak_mib']} MiB  "
                        f"delta={r['delta_mib']:+} MiB  warm={r['warm_ms']} ms  "
                        f"wall_median={r['wall_median_ms']} ms  score={r['score']}",
                        flush=True,
                    )
                    crate_rows.append(
                        {
                            "size_mp": lbl,
                            "size_w": w,
                            "size_h": h,
                            "mode": mode,
                            "backend": backend,
                            "peak_vram_bytes": r["delta_bytes"],
                            "peak_vram_human": human_bytes(r["delta_bytes"]),
                            "wall_median_ms": r["wall_median_ms"],
                            "warm_ms": r["warm_ms"],
                            "score": r["score"],
                            "notes": "",
                        }
                    )
        # Per-crate TSV — read existing rows from previous runs, drop
        # rows that match (size_mp, mode, backend) we just measured,
        # and append the fresh measurements.
        crate_tsv = repo_root / "crates" / crate / "benchmarks" / CRATE_TSV_NAME
        existing_crate: list[dict] = []
        if crate_tsv.exists():
            with crate_tsv.open() as f:
                rd = csv.DictReader(f, delimiter="\t")
                for row in rd:
                    key = (row.get("size_mp"), row.get("mode"), row.get("backend"))
                    fresh_keys = {(r["size_mp"], r["mode"], r["backend"])
                                  for r in crate_rows}
                    if key not in fresh_keys:
                        existing_crate.append(row)
        merged_crate = existing_crate + crate_rows
        meta = {
            "tool": "scripts/memory_audit/sweep_gpu_metrics_2026-05-28.py",
            "git_sha": git_sha,
            "host": host,
            "gpu_name": smi_info.get("gpu_name", "?"),
            "gpu_mem_total": smi_info.get("gpu_mem_total", "?"),
            "gpu_driver": smi_info.get("gpu_driver", "?"),
            "cuda_path": "/usr/local/cuda",
            "metric": crate,
            "sizes_run": ",".join(s[0] for s in chosen_sizes),
            "backends_run": ",".join(backends),
            "reps": args.reps,
            "method": ("subprocess-per-cell, nvidia-smi delta during "
                       f"~{HOLD_SAMPLE_SECS*1000:.0f}ms hold window after READY; "
                       "wall_median_ms = median of WORKER_REPS post-warm runs."),
            "started_utc": started,
        }
        write_tsv(crate_tsv, merged_crate, meta)
        print(
            f"\nwrote {crate_tsv} ({len(merged_crate)} total, "
            f"+{len(crate_rows)} this run)\n",
            flush=True,
        )
        global_rows.extend(
            [{**r, "crate": crate} for r in merged_crate]
        )

    # Global summary TSV — read existing rows for crates NOT in this
    # run so multi-run invocations accumulate instead of clobbering.
    global_tsv = repo_root / f"{args.out_base}.tsv"
    global_tsv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "crate",
        "size_mp",
        "size_w",
        "size_h",
        "mode",
        "backend",
        "peak_vram_bytes",
        "peak_vram_human",
        "wall_median_ms",
        "warm_ms",
        "score",
        "notes",
    ]
    existing: list[dict] = []
    if global_tsv.exists():
        with global_tsv.open() as f:
            r = csv.DictReader(f, delimiter="\t")
            for row in r:
                # Drop rows for crates we're re-running; keep the rest.
                if row.get("crate") not in metric_names:
                    existing.append(row)
    all_rows = existing + global_rows
    with global_tsv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(
        f"wrote global summary {global_tsv} "
        f"({len(all_rows)} total, +{len(global_rows)} this run)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
