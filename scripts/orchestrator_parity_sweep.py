#!/usr/bin/env python3
"""Phase 7.7 parity sweep — gate for the orchestrator default flip.

Per CLAUDE.md / Phase 7.7 brief, we MUST prove orchestrator == legacy
across all 6 metrics × multiple sizes × multiple distortion levels
BEFORE flipping the default. This script runs the matrix and emits a
CSV + Markdown summary; if any cell fails parity within the metric's
tolerance, the script exits rc=2 and the flip is blocked.

Matrix:
  metrics = [cvvdp, ssim2-gpu, dssim-gpu, butteraugli-gpu,
             iwssim-gpu, zensim-gpu]
  sizes   = [256, 1024, 4096]
  qs      = [20, 50, 80]
  -> 6 * 3 * 3 = 54 cells

Distortion source: PIL JPEG re-encode at the given quality, decoded
back to PNG for the metric's RGB8 input. Identical method used for
both orchestrator and legacy invocations.

Tolerance:
  - bit-identical (|diff| == 0.0) strongly preferred
  - ~5e-5 atomic-add reorder noise acceptable for metrics using
    atomic<f32> in their reductions
  - Anything larger = FAIL

Output:
  benchmarks/orchestrator_parity_2026-05-27.csv  — one row per cell
  benchmarks/orchestrator_parity_2026-05-27.md   — table summary

Usage:
  python3 scripts/orchestrator_parity_sweep.py [--binary PATH]
                                               [--out-csv PATH]
                                               [--out-md PATH]
                                               [--work-dir PATH]
                                               [--metric M ...]
                                               [--size N ...]
                                               [--q N ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Per-metric tolerance + identity values.
#
# Bit-identical is the goal — Phase 7.5 INTEGRATION_NOTES.md guarantees
# bit-identical parquet output for ssim2 + butter. Allow a small atomic
# epsilon for the metrics that use atomic<f32> reductions (zensim,
# iwssim, dssim).
# ---------------------------------------------------------------------------

METRICS = ["cvvdp", "ssim2-gpu", "dssim-gpu", "butteraugli-gpu",
           "iwssim-gpu", "zensim-gpu"]

# Tolerance per metric. Tight bit-identical preferred; relaxed value
# permitted for known atomic-reduction noise (Phase 7.5 docs).
TOLERANCES = {
    "cvvdp":            1e-3,   # JOD ~10 scale, atomic + display model
    "ssim2-gpu":        5e-4,   # ~100 scale
    "dssim-gpu":        5e-5,   # ~0..1 scale
    "butteraugli-gpu":  5e-4,   # 0..30 scale, atomic reductions
    "iwssim-gpu":       5e-5,   # 0..1 scale
    "zensim-gpu":       5e-3,   # ~100 scale, several atomic reductions
}

# Which JSON path holds the metric's primary scalar score. Butteraugli
# emits two columns; we compare butteraugli_max for parity (pnorm3
# checked separately below).
PRIMARY_COL = {
    "cvvdp":           None,        # computed at runtime from CVVDP_COLUMN_NAME
    "ssim2-gpu":       "ssim2_gpu",
    "dssim-gpu":       "dssim_gpu",
    "butteraugli-gpu": "butteraugli_max_gpu",
    "iwssim-gpu":      "iwssim_gpu",
    "zensim-gpu":      "zensim_gpu",
}


# ---------------------------------------------------------------------------
# Test image generation.
#
# We deliberately use a content-rich synthetic — sinusoidal + radial
# gradient + sharp text-like edges + Perlin-style noise. Pure gradients
# produce metric outputs near identity (low signal), and pure noise hits
# the metric's worst-case which is also not ideal. A mixed synthetic
# image puts the metric in its normal operating range.
# ---------------------------------------------------------------------------

def synth_image(size: int, seed: int = 42) -> np.ndarray:
    """Generate an RGB8 synthetic test image of (size, size, 3)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / float(size)
    r = (np.sin(xx * 13.0) * np.cos(yy * 11.0) * 0.5 + 0.5) * 200.0 + 30.0
    g = (np.sin((xx + yy) * 17.0) * 0.5 + 0.5) * 200.0 + 30.0
    b = ((xx - 0.5) ** 2 + (yy - 0.5) ** 2) * 4.0
    b = np.clip(b * 200.0, 0.0, 200.0) + 30.0
    noise = rng.uniform(-15.0, 15.0, size=(size, size, 3)).astype(np.float32)
    img = np.stack([r, g, b], axis=-1) + noise
    return np.clip(img, 0.0, 255.0).astype(np.uint8)


def encode_jpeg_then_back_to_png(ref_path: Path, q: int, out_dist_png: Path) -> None:
    """JPEG-encode at quality q, decode back, write as PNG.

    Using PIL to keep the script dependency-light. The resulting
    distorted PNG is what both legacy and orchestrator paths consume —
    identical input bytes guarantees the metric sees identical pixels.
    """
    img = Image.open(ref_path).convert("RGB")
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as jf:
        jpg_path = Path(jf.name)
    try:
        img.save(jpg_path, format="JPEG", quality=q, optimize=False, progressive=False)
        roundtrip = Image.open(jpg_path).convert("RGB")
        roundtrip.save(out_dist_png, format="PNG", compress_level=1)
    finally:
        jpg_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# CLI invocation.
# ---------------------------------------------------------------------------

@dataclass
class ScoreResult:
    ok: bool
    primary_value: Optional[float]
    columns: dict[str, float]
    stderr: str


def run_score(binary: Path, metric: str, ref: Path, dist: Path,
              use_orchestrator: bool, cache_dir: Path,
              bench_on_start: str = "no",
              env_extra: Optional[dict[str, str]] = None) -> ScoreResult:
    """Invoke `zenmetrics score` in either legacy or orchestrator mode.

    Returns the JSON-parsed output of `--output json`. Both paths emit
    the same `scores.<col>` JSON shape per the Phase 7.5 contract.
    """
    args = [str(binary)]
    if use_orchestrator:
        args.extend(["--use-orchestrator",
                     "--orchestrator-cache", str(cache_dir),
                     "--bench-on-start", bench_on_start])
    args.extend([
        "score",
        "--metric", metric,
        "--reference", str(ref),
        "--distorted", str(dist),
        "--gpu-runtime", "cuda",
        "--output", "json",
    ])
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(args, capture_output=True, text=True, env=env, timeout=600)
    if proc.returncode != 0:
        return ScoreResult(False, None, {}, proc.stderr[-2000:])
    try:
        out = json.loads(proc.stdout.strip())
    except json.JSONDecodeError as e:
        return ScoreResult(False, None, {}, f"json decode {e}: {proc.stdout[:500]}")

    cols = {}
    scores = out.get("scores", {})
    if isinstance(scores, dict):
        for k, v in scores.items():
            try:
                cols[k] = float(v)
            except (TypeError, ValueError):
                pass

    # Determine primary column. For cvvdp it's whichever key starts
    # with "cvvdp_imazen_v"; for others the table mapping above.
    primary_col = PRIMARY_COL.get(metric)
    if primary_col is None:
        # cvvdp: find versioned key
        for k in cols.keys():
            if k.startswith("cvvdp_imazen_v") or k == "cvvdp":
                primary_col = k
                break
    primary_value = cols.get(primary_col) if primary_col else None
    # Column-name divergence fallback: iwssim emits `iwssim_gpu` from
    # the legacy path and `iwssim_imazen_v<MAJOR>_<MINOR>_<PATCH>` from
    # the orchestrator path. If the configured primary column is
    # missing but exactly ONE other f64 column exists, treat it as the
    # primary value (the divergence is recorded in `notes`).
    if primary_value is None and len(cols) == 1:
        primary_value = next(iter(cols.values()))
    return ScoreResult(True, primary_value, cols, proc.stderr[-1000:])


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def parity_verdict(metric: str, legacy: float, orch: float) -> tuple[str, float]:
    """Return ("PASS"|"FAIL", abs_diff)."""
    if legacy is None or orch is None:
        return ("FAIL", float("inf"))
    diff = abs(legacy - orch)
    tol = TOLERANCES.get(metric, 1e-3)
    if diff == 0.0:
        return ("PASS-EXACT", 0.0)
    if diff <= tol:
        return ("PASS", diff)
    return ("FAIL", diff)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary",
                        default="target/release/zenmetrics",
                        help="Path to zenmetrics binary")
    parser.add_argument("--out-csv",
                        default="benchmarks/orchestrator_parity_2026-05-27.csv")
    parser.add_argument("--out-md",
                        default="benchmarks/orchestrator_parity_2026-05-27.md")
    parser.add_argument("--work-dir",
                        default=None,
                        help="Working directory for synth images. "
                             "Defaults to mkdtemp.")
    parser.add_argument("--metric", action="append", default=None,
                        help="Restrict to these metrics (default: all six)")
    parser.add_argument("--size", action="append", type=int, default=None,
                        help="Restrict to these sizes (default: 256,1024,4096)")
    parser.add_argument("--q", action="append", type=int, default=None,
                        help="Restrict to these JPEG qualities (default: 20,50,80)")
    parser.add_argument("--keep-work", action="store_true",
                        help="Don't delete the work dir on exit")
    args = parser.parse_args()

    binary = Path(args.binary).resolve()
    if not binary.exists():
        print(f"ERROR: binary not found at {binary}", file=sys.stderr)
        return 1

    metrics = args.metric if args.metric else METRICS
    sizes = args.size if args.size else [256, 1024, 4096]
    qs = args.q if args.q else [20, 50, 80]

    work_dir = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="orch_parity_"))
    work_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = work_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    refs_dir = work_dir / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    dists_dir = work_dir / "dists"
    dists_dir.mkdir(parents=True, exist_ok=True)

    print(f"[parity] binary={binary}", flush=True)
    print(f"[parity] work_dir={work_dir}", flush=True)
    print(f"[parity] metrics={metrics}", flush=True)
    print(f"[parity] sizes={sizes}", flush=True)
    print(f"[parity] qs={qs}", flush=True)

    # Generate refs and dists once.
    for size in sizes:
        ref_path = refs_dir / f"synth_{size}.png"
        if not ref_path.exists():
            arr = synth_image(size)
            Image.fromarray(arr, mode="RGB").save(ref_path, format="PNG", compress_level=1)
        for q in qs:
            dist_path = dists_dir / f"synth_{size}_q{q}.png"
            if not dist_path.exists():
                encode_jpeg_then_back_to_png(ref_path, q, dist_path)

    # Pre-warm: prime orchestrator cache once on the smallest size to
    # avoid first-cell warmup confounding the parity numbers. Running
    # the bench at startup populates the cache for ALL six metrics in
    # one go (~30s on this hardware) — subsequent score calls use the
    # cached measurements with --bench-on-start=no.
    print(f"[parity] priming orchestrator cache with bench-on-start=yes (256, ssim2-gpu)...", flush=True)
    prime = run_score(binary, "ssim2-gpu",
                      refs_dir / f"synth_{sizes[0]}.png",
                      dists_dir / f"synth_{sizes[0]}_q{qs[0]}.png",
                      use_orchestrator=True, cache_dir=cache_dir,
                      bench_on_start="yes")
    if not prime.ok:
        print(f"[parity] PRIME FAILED: stderr={prime.stderr}", flush=True)
        return 1
    print(f"[parity] prime ok, primary={prime.primary_value}", flush=True)

    out_csv = Path(args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["metric", "size", "q", "legacy", "orchestrator", "abs_diff",
              "tolerance", "verdict", "legacy_extra_cols", "orch_extra_cols",
              "notes"]
    rows: list[dict] = []

    total = len(metrics) * len(sizes) * len(qs)
    cell = 0
    for metric in metrics:
        for size in sizes:
            for q in qs:
                cell += 1
                ref_path = refs_dir / f"synth_{size}.png"
                dist_path = dists_dir / f"synth_{size}_q{q}.png"
                print(f"[parity] cell {cell}/{total}: metric={metric} size={size} q={q}", flush=True)

                legacy = run_score(binary, metric, ref_path, dist_path,
                                   use_orchestrator=False, cache_dir=cache_dir)
                orch = run_score(binary, metric, ref_path, dist_path,
                                 use_orchestrator=True, cache_dir=cache_dir,
                                 bench_on_start="no")

                notes_parts = []
                if not legacy.ok:
                    notes_parts.append(f"legacy_err={legacy.stderr[-200:]}")
                if not orch.ok:
                    notes_parts.append(f"orch_err={orch.stderr[-200:]}")

                verdict, diff = parity_verdict(metric, legacy.primary_value, orch.primary_value)
                # Detect column-name divergence (values bit-identical
                # but the score lives under a different parquet column).
                # This violates the bit-identical-sidecar contract even
                # though the numbers match.
                legacy_cols = set(legacy.columns.keys())
                orch_cols = set(orch.columns.keys())
                if verdict.startswith("PASS") and legacy_cols != orch_cols:
                    notes_parts.append(
                        f"col-name diverged: legacy={sorted(legacy_cols)} orch={sorted(orch_cols)}"
                    )
                    verdict = "FAIL-COLNAME"
                # Compare any extra columns too.
                extra_legacy = ",".join(f"{k}={v:.6f}" for k, v in legacy.columns.items() if k != PRIMARY_COL.get(metric))
                extra_orch = ",".join(f"{k}={v:.6f}" for k, v in orch.columns.items() if k != PRIMARY_COL.get(metric))

                # For butteraugli, also compare pnorm3 column.
                if metric == "butteraugli-gpu":
                    leg_pn = legacy.columns.get("butteraugli_pnorm3_gpu")
                    orch_pn = orch.columns.get("butteraugli_pnorm3_gpu")
                    if leg_pn is not None and orch_pn is not None:
                        pn_diff = abs(leg_pn - orch_pn)
                        if pn_diff > TOLERANCES.get(metric, 1e-3):
                            verdict = "FAIL"
                            notes_parts.append(f"pnorm3 diverged by {pn_diff:.6e}")
                        else:
                            notes_parts.append(f"pnorm3_diff={pn_diff:.6e}")

                row = {
                    "metric": metric,
                    "size": size,
                    "q": q,
                    "legacy": f"{legacy.primary_value:.6f}" if legacy.primary_value is not None else "",
                    "orchestrator": f"{orch.primary_value:.6f}" if orch.primary_value is not None else "",
                    "abs_diff": f"{diff:.6e}",
                    "tolerance": f"{TOLERANCES.get(metric, 1e-3):.6e}",
                    "verdict": verdict,
                    "legacy_extra_cols": extra_legacy,
                    "orch_extra_cols": extra_orch,
                    "notes": "; ".join(notes_parts),
                }
                rows.append(row)
                print(f"[parity]   legacy={row['legacy']} orch={row['orchestrator']} diff={row['abs_diff']} -> {verdict}", flush=True)

    # Write CSV.
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Write Markdown table summary.
    out_md = Path(args.out_md).resolve()
    with open(out_md, "w") as f:
        f.write(f"# Orchestrator vs legacy parity sweep — 2026-05-27\n\n")
        f.write(f"Binary: `{binary}`  \n")
        f.write(f"Total cells: {len(rows)}  \n")
        pass_n = sum(1 for r in rows if r["verdict"].startswith("PASS"))
        fail_n = sum(1 for r in rows if r["verdict"] == "FAIL")
        fail_col_n = sum(1 for r in rows if r["verdict"] == "FAIL-COLNAME")
        f.write(f"PASS: {pass_n}  FAIL (value): {fail_n}  FAIL (column-name): {fail_col_n}  \n\n")
        f.write("| metric | size | q | legacy | orchestrator | abs_diff | tol | verdict |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['metric']} | {r['size']} | {r['q']} | {r['legacy']} | "
                    f"{r['orchestrator']} | {r['abs_diff']} | {r['tolerance']} | "
                    f"{r['verdict']} |\n")
        if fail_n > 0 or fail_col_n > 0:
            f.write("\n## Failures\n\n")
            for r in rows:
                if r["verdict"] == "FAIL":
                    f.write(f"- **{r['metric']} size={r['size']} q={r['q']}** (value): "
                            f"legacy={r['legacy']} orch={r['orchestrator']} "
                            f"diff={r['abs_diff']} > tol={r['tolerance']}. "
                            f"Notes: {r['notes']}\n")
                elif r["verdict"] == "FAIL-COLNAME":
                    f.write(f"- **{r['metric']} size={r['size']} q={r['q']}** (column-name): "
                            f"values bit-identical ({r['legacy']}) but column keys "
                            f"diverge. Notes: {r['notes']}\n")

    fail_total = fail_n + fail_col_n
    print(f"\n[parity] wrote {out_csv}")
    print(f"[parity] wrote {out_md}")
    print(f"[parity] PASS={pass_n} FAIL_value={fail_n} FAIL_colname={fail_col_n} of {len(rows)}")

    if not args.keep_work:
        try:
            shutil.rmtree(work_dir)
        except Exception:
            pass

    return 0 if fail_total == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
