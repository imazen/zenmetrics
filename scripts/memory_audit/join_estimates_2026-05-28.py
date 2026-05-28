#!/usr/bin/env python3
"""Compute pipeline-estimator output per (crate, mode, w, h) cell and
join it to the GPU sweep TSV for gap analysis.

Estimator formulas are inlined from each crate's `memory_mode.rs`
(constants verified against source 2026-05-28). When the source
formula changes, update this file and re-run.

Output:
- `benchmarks/gpu_metrics_estimates_2026-05-28.tsv` — pure estimates.
- `benchmarks/gpu_metrics_gap_2026-05-28.tsv` — sweep ∪ estimator
  joined on (crate, mode, size_mp), with `gap_bytes` and
  `peak_over_estimate_ratio` columns.

Run:
    python3 scripts/memory_audit/join_estimates_2026-05-28.py
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from typing import Iterable

SIZES = [
    ("1mp", 1024, 1024),
    ("4mp", 2048, 2048),
    ("16mp", 4096, 4096),
    ("40mp", 7680, 5184),
]

BODY = 256  # default strip body in the harness


# ---- butteraugli-gpu ----------------------------------------------
BUTTER_HALO_ROWS = 40
BUTTER_PLANES = 50


def butter_full(w: int, h: int) -> int:
    return BUTTER_PLANES * w * h * 4


def butter_strip(w: int, body: int) -> int:
    strip_h = body + 2 * BUTTER_HALO_ROWS
    return BUTTER_PLANES * w * strip_h * 4


# ---- ssim2-gpu ----------------------------------------------------
SSIM2_NUM_SCALES = 6
SSIM2_PLANES_PER_SCALE = 57
SSIM2_HALO = 256


def ssim2_full(w: int, h: int) -> int:
    total = 0
    cw, ch = w, h
    for _ in range(SSIM2_NUM_SCALES):
        if cw < 8 or ch < 8:
            break
        total += SSIM2_PLANES_PER_SCALE * cw * ch * 4
        cw = (cw + 1) // 2
        ch = (ch + 1) // 2
    total += w * h * 4 * 2
    return total


def ssim2_strip(w: int, body: int) -> int:
    strip_h = body + 2 * SSIM2_HALO
    cw, ch = w, strip_h
    total = 0
    for _ in range(SSIM2_NUM_SCALES):
        if cw < 8 or ch < 8:
            break
        total += SSIM2_PLANES_PER_SCALE * cw * ch * 4
        cw = (cw + 1) // 2
        ch = (ch + 1) // 2
    total += w * strip_h * 4 * 2
    return total


# ---- dssim-gpu (recalibrated, task137 — mirrors src/memory_mode.rs) -
DSSIM_NUM_SCALES = 5
DSSIM_PLANES_PER_SCALE = 31  # was 13; Scale::new = 9·alloc_3 + 4 singles
DSSIM_HALO = 256
DSSIM_CONTEXT_BASE_BYTES = 208 * 1024 * 1024
DSSIM_CONTEXT_PER_PIXEL_BYTES = 18


def dssim_full(w: int, h: int) -> int:
    total = 0
    cw, ch = w, h
    for _ in range(DSSIM_NUM_SCALES):
        w_eff = max(cw, 8)
        h_eff = max(ch, 8)
        total += DSSIM_PLANES_PER_SCALE * w_eff * h_eff * 4
        cw = (cw + 1) // 2
        ch = (ch + 1) // 2
    n0 = w * h
    total += n0 * 4 * 2
    total += DSSIM_CONTEXT_BASE_BYTES + n0 * DSSIM_CONTEXT_PER_PIXEL_BYTES
    return total


def dssim_strip(w: int, body: int) -> int:
    strip_h = body + 2 * DSSIM_HALO
    cw, ch = w, strip_h
    total = 0
    for _ in range(DSSIM_NUM_SCALES):
        w_eff = max(cw, 8)
        h_eff = max(ch, 8)
        total += DSSIM_PLANES_PER_SCALE * w_eff * h_eff * 4
        cw = (cw + 1) // 2
        ch = (ch + 1) // 2
    n = w * strip_h
    total += n * 4 * 2
    total += DSSIM_CONTEXT_BASE_BYTES + n * DSSIM_CONTEXT_PER_PIXEL_BYTES
    return total


# ---- iwssim-gpu (recalibrated, task137 — mirrors src/memory_mode.rs) -
IWSSIM_NUM_SCALES = 5
IWSSIM_PLANES_PER_SCALE = 19  # was 10; Scale::new = 19 f32 planes
IWSSIM_HALO = 256
# Reduction/cov scratch: partials 9·16·256·4 + sums 9·4 + cov 100·64·256·4.
IWSSIM_REDUCTION_SCRATCH_BYTES = (9 * 16 * 256) * 4 + 9 * 4 + (100 * 64 * 256) * 4
IWSSIM_POOL_NUM = 7  # 7/5 = 1.40 pool factor
IWSSIM_POOL_DEN = 5
IWSSIM_FLOOR_BYTES = 256 * 1024 * 1024


def _iwssim_pool_floor(raw: int) -> int:
    return max(raw * IWSSIM_POOL_NUM // IWSSIM_POOL_DEN, IWSSIM_FLOOR_BYTES)


def iwssim_full(w: int, h: int) -> int:
    total = 0
    cw, ch = w, h
    for _ in range(IWSSIM_NUM_SCALES):
        total += IWSSIM_PLANES_PER_SCALE * cw * ch * 4
        cw = (cw + 1) // 2
        ch = (ch + 1) // 2
    total += w * h * 4 * 2
    total += IWSSIM_REDUCTION_SCRATCH_BYTES
    return _iwssim_pool_floor(total)


def iwssim_strip(w: int, body: int) -> int:
    strip_h = body + 2 * IWSSIM_HALO
    cw, ch = w, strip_h
    total = 0
    for _ in range(IWSSIM_NUM_SCALES):
        total += IWSSIM_PLANES_PER_SCALE * cw * ch * 4
        cw = (cw + 1) // 2
        ch = (ch + 1) // 2
    total += w * strip_h * 4 * 2
    total += IWSSIM_REDUCTION_SCRATCH_BYTES
    return _iwssim_pool_floor(total)


# ---- zensim-gpu (Basic regime) ------------------------------------
def zensim_pyramid_pixels(w: int, h: int) -> int:
    # Mirror crates/zensim-gpu/src/memory_mode.rs pyramid_pixels:
    # accumulate level pixels until min(w,h) < 8.
    total = 0
    cw, ch = w, h
    while cw >= 8 and ch >= 8:
        total += cw * ch
        cw = (cw + 1) // 2
        ch = (ch + 1) // 2
    return total


def zensim_full(w: int, h: int) -> int:
    # Basic regime: base_mb=0, beta_b_per_pyr=41
    return zensim_pyramid_pixels(w, h) * 41


def zensim_strip(w: int, body: int) -> int:
    # Same coefficients on the strip-height pyramid.
    return zensim_pyramid_pixels(w, body + 2 * 40) * 41


# ---- cvvdp-gpu (REAL Rust estimator output, task137) --------------
# DURABLE ANTI-DRIFT FIX (task137): the cvvdp-gpu per-mode estimates
# are NO LONGER hand-copied proxy formulas here. The prior version used
# a 218-MiB/MP linear proxy for Full and 2×strip for Mode B — both of
# which drifted hard from the real `pipeline.rs` estimators (the Mode E
# "13× off" finding in docs/AUDIT_2026-05-28.md was a pure proxy
# artifact, not a real estimator bug). Instead we shell out to the Rust
# example `cvvdp-gpu --example mem_estimate_tsv`, which calls the actual
# `estimate_gpu_memory_bytes{,_strip,_strip_pair,_capped}` functions and
# prints their output. When a Rust estimator formula changes, this
# table changes with it — no parallel formula to keep in sync.
#
# The example is built+run once and its TSV cached in this dict.
_CVVDP_CACHE: dict[tuple[str, int, int], int] = {}


def _load_cvvdp_estimates(repo_root: Path) -> dict[tuple[str, int, int], int]:
    """Build + run the cvvdp-gpu `mem_estimate_tsv` example and parse
    its `(mode, w, h) -> bytes` output. Cached process-wide."""
    if _CVVDP_CACHE:
        return _CVVDP_CACHE
    cmd = [
        "cargo", "run", "--release", "--quiet",
        "--example", "mem_estimate_tsv",
        "-p", "cvvdp-gpu",
        "--no-default-features", "--features", "cubecl-types",
    ]
    print(f"  [cvvdp] shelling out to Rust estimator: {' '.join(cmd)}",
          file=sys.stderr)
    out = subprocess.run(
        cmd, cwd=repo_root, capture_output=True, text=True, check=True,
    ).stdout
    for line in out.splitlines():
        line = line.strip()
        if not line or line.startswith("mode\t"):
            continue
        parts = line.split("\t")
        if len(parts) != 4:
            continue
        mode, sw, sh, val = parts
        if not val:
            continue  # estimator returned None for this (mode, size)
        _CVVDP_CACHE[(mode, int(sw), int(sh))] = int(val)
    return _CVVDP_CACHE


def cvvdp_estimate(mode: str, w: int, h: int, repo_root: Path) -> int | None:
    """Look up the real Rust estimator output for a cvvdp mode/size."""
    return _load_cvvdp_estimates(repo_root).get((mode, w, h))


# Repo root, set in main() before ESTIMATORS is iterated so the cvvdp
# closures can locate the crate to build the example from.
_REPO_ROOT: Path | None = None


def _cvvdp(mode: str):
    """Build a `(w, h, body) -> bytes | -1` closure for a cvvdp mode
    that defers to the real Rust estimator output."""
    def fn(w: int, h: int, _b: int) -> int:
        assert _REPO_ROOT is not None, "_REPO_ROOT must be set before estimating"
        v = cvvdp_estimate(mode, w, h, _REPO_ROOT)
        return v if v is not None else -1
    return fn


def human(n: int) -> str:
    if n < 0:
        return "?"
    units = ["B", "KiB", "MiB", "GiB"]
    f = float(n)
    u = 0
    while f >= 1024 and u < len(units) - 1:
        f /= 1024
        u += 1
    return f"{f:.2f}{units[u]}"


# Map (crate, mode) → estimator function (w, h, body) -> bytes
ESTIMATORS: dict[tuple[str, str], callable] = {
    # butteraugli
    ("butteraugli-gpu", "full"): lambda w, h, b: butter_full(w, h),
    ("butteraugli-gpu", "warm_ref"): lambda w, h, b: butter_full(w, h),
    ("butteraugli-gpu", "strip"): lambda w, h, b: butter_strip(w, b),
    ("butteraugli-gpu", "warm_ref_strip"): lambda w, h, b: butter_strip(w, b),
    # ssim2
    ("ssim2-gpu", "full"): lambda w, h, b: ssim2_full(w, h),
    ("ssim2-gpu", "warm_ref"): lambda w, h, b: ssim2_full(w, h),
    ("ssim2-gpu", "strip"): lambda w, h, b: ssim2_strip(w, b),
    ("ssim2-gpu", "warm_ref_strip"): lambda w, h, b: ssim2_strip(w, b),
    # dssim
    ("dssim-gpu", "full"): lambda w, h, b: dssim_full(w, h),
    ("dssim-gpu", "warm_ref"): lambda w, h, b: dssim_full(w, h),
    ("dssim-gpu", "strip"): lambda w, h, b: dssim_strip(w, b),
    ("dssim-gpu", "warm_ref_strip"): lambda w, h, b: dssim_strip(w, b),
    # iwssim — gray + rgb share same estimator (RGB path uses same
    # pyramid pre-conversion to luma).
    ("iwssim-gpu", "full"): lambda w, h, b: iwssim_full(w, h),
    ("iwssim-gpu", "warm_ref"): lambda w, h, b: iwssim_full(w, h),
    ("iwssim-gpu", "strip"): lambda w, h, b: iwssim_strip(w, b),
    ("iwssim-gpu", "warm_ref_strip"): lambda w, h, b: iwssim_strip(w, b),
    ("iwssim-gpu", "rgb_full"): lambda w, h, b: iwssim_full(w, h),
    ("iwssim-gpu", "rgb_strip"): lambda w, h, b: iwssim_strip(w, b),
    ("iwssim-gpu", "rgb_warm_ref_strip"): lambda w, h, b: iwssim_strip(w, b),
    # zensim
    ("zensim-gpu", "full"): lambda w, h, b: zensim_full(w, h),
    ("zensim-gpu", "warm_ref"): lambda w, h, b: zensim_full(w, h),
    ("zensim-gpu", "strip"): lambda w, h, b: zensim_strip(w, b),
    ("zensim-gpu", "warm_ref_strip"): lambda w, h, b: zensim_strip(w, b),
    # cvvdp — REAL Rust estimator output (task137), shelled out via the
    # `mem_estimate_tsv` example. Mode names mirror the sweep TSV's
    # cvvdp-gpu `mode` column.
    ("cvvdp-gpu", "full"): _cvvdp("full"),
    ("cvvdp-gpu", "warm_ref"): _cvvdp("warm_ref"),
    ("cvvdp-gpu", "warm_ref_strip"): _cvvdp("warm_ref_strip"),
    # Mode E's estimator is `estimate_gpu_memory_bytes_strip`; the sweep
    # exercises it under the `warm_ref_strip` mode. There is no separate
    # `strip` sweep mode for cvvdp-gpu (the cvvdp Strip variant IS
    # Mode E / warm_ref_strip), so map a defensive `strip` alias too.
    ("cvvdp-gpu", "strip"): _cvvdp("warm_ref_strip"),
    ("cvvdp-gpu", "strip_pair"): _cvvdp("strip_pair"),
    ("cvvdp-gpu", "capped"): _cvvdp("capped"),
    ("cvvdp-gpu", "auto"): _cvvdp("auto"),
}


def main() -> int:
    global _REPO_ROOT
    repo_root = Path(__file__).resolve().parents[2]
    _REPO_ROOT = repo_root
    out_est = repo_root / "benchmarks" / "gpu_metrics_estimates_2026-05-28.tsv"
    out_est.parent.mkdir(parents=True, exist_ok=True)
    with out_est.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow([
            "crate", "mode", "size_mp", "size_w", "size_h",
            "estimate_bytes", "estimate_human",
        ])
        for (crate, mode), fn in sorted(ESTIMATORS.items()):
            for label, sw, sh in SIZES:
                b = fn(sw, sh, BODY)
                w.writerow([crate, mode, label, sw, sh, b, human(b)])

    print(f"wrote {out_est}")

    # Now join to the sweep TSV.
    sweep_tsv = repo_root / "benchmarks" / "gpu_metrics_sweep_2026-05-28.tsv"
    if not sweep_tsv.exists():
        print(f"sweep TSV not found: {sweep_tsv}")
        return 1
    sweep_rows = list(csv.DictReader(sweep_tsv.open(), delimiter="\t"))

    # Build estimate lookup (crate, mode, size_mp) -> bytes
    est_lookup: dict[tuple[str, str, str], int] = {}
    for (crate, mode), fn in ESTIMATORS.items():
        for label, sw, sh in SIZES:
            est_lookup[(crate, mode, label)] = fn(sw, sh, BODY)

    out_gap = repo_root / "benchmarks" / "gpu_metrics_gap_2026-05-28.tsv"
    with out_gap.open("w", newline="") as f:
        fieldnames = [
            "crate", "size_mp", "size_w", "size_h", "mode", "backend",
            "peak_vram_bytes", "peak_vram_human",
            "estimate_bytes", "estimate_human",
            "gap_bytes", "gap_human",
            "peak_over_estimate_ratio",
            "wall_median_ms", "warm_ms", "score", "notes",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for r in sweep_rows:
            key = (r["crate"], r["mode"], r["size_mp"])
            est = est_lookup.get(key, -1)
            try:
                peak = int(r["peak_vram_bytes"])
            except (ValueError, TypeError):
                peak = -1
            if est > 0 and peak > 0:
                gap = peak - est
                ratio = peak / est
            else:
                gap = -1
                ratio = -1.0
            row = {
                **r,
                "estimate_bytes": est if est > 0 else "",
                "estimate_human": human(est) if est > 0 else "",
                "gap_bytes": gap if gap >= 0 else "",
                "gap_human": human(gap) if gap >= 0 else "",
                "peak_over_estimate_ratio": (
                    f"{ratio:.3f}" if ratio > 0 else ""
                ),
            }
            w.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"wrote {out_gap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
