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


# ---- dssim-gpu ----------------------------------------------------
DSSIM_NUM_SCALES = 5
DSSIM_PLANES_PER_SCALE = 13
DSSIM_HALO = 256


def dssim_full(w: int, h: int) -> int:
    total = 0
    cw, ch = w, h
    for _ in range(DSSIM_NUM_SCALES):
        w_eff = max(cw, 8)
        h_eff = max(ch, 8)
        total += DSSIM_PLANES_PER_SCALE * w_eff * h_eff * 4
        cw = (cw + 1) // 2
        ch = (ch + 1) // 2
    total += w * h * 4 * 2
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
    total += w * strip_h * 4 * 2
    return total


# ---- iwssim-gpu ---------------------------------------------------
IWSSIM_NUM_SCALES = 5
IWSSIM_PLANES_PER_SCALE = 10
IWSSIM_HALO = 256


def iwssim_full(w: int, h: int) -> int:
    total = 0
    cw, ch = w, h
    for _ in range(IWSSIM_NUM_SCALES):
        total += IWSSIM_PLANES_PER_SCALE * cw * ch * 4
        cw = (cw + 1) // 2
        ch = (ch + 1) // 2
    total += w * h * 4 * 2
    return total


def iwssim_strip(w: int, body: int) -> int:
    strip_h = body + 2 * IWSSIM_HALO
    cw, ch = w, strip_h
    total = 0
    for _ in range(IWSSIM_NUM_SCALES):
        total += IWSSIM_PLANES_PER_SCALE * cw * ch * 4
        cw = (cw + 1) // 2
        ch = (ch + 1) // 2
    total += w * strip_h * 4 * 2
    return total


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


# ---- cvvdp-gpu (read off pipeline.rs source 2026-05-28) ------------
# Per docs/CHANGELOG: linear ~218 MB/MP for Full, ~36 MB/MP for Strip
# Mode E.  Use the per-MP slope.  These are approximations — the real
# estimator computes per-level pyramid sums.  For the GAP table we
# report the rule-of-thumb so the table has coverage; production
# callers should query estimate_gpu_memory_bytes(_strip|_strip_pair|
# _capped) directly.
def cvvdp_full(w: int, h: int) -> int:
    n_mp = (w * h) / 1_000_000
    return int(n_mp * 218 * 1024 * 1024)


def cvvdp_strip(w: int, body: int) -> int:
    # Strip working set ~ 36 MB/MP equivalent at typical strip dims;
    # rough: width × strip_height × pyramid-share.
    strip_h = body + 2 * 40
    n_mp = (w * strip_h) / 1_000_000
    return int(n_mp * 218 * 1024 * 1024)  # within-strip working set


def cvvdp_strip_pair(w: int, body: int) -> int:
    # Mode B walks both sides — peak ≈ 2 × strip working set.
    return 2 * cvvdp_strip(w, body)


def cvvdp_capped(w: int, h: int, levels: int = 5) -> int:
    # Truncated pyramid; rough proxy as a fraction of Full based on
    # cumulative pixel share. For natural depth 9 → 5, levels 6/7/8/9
    # at 1/64..1/512 of base are dropped; ~97% of pixels remain in
    # levels 1-5. The truncation saves d_scratch + transient buffers
    # for the dropped levels — typical observed savings: ~5-10%.
    return int(cvvdp_full(w, h) * 0.95)


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
    # cvvdp
    ("cvvdp-gpu", "full"): lambda w, h, b: cvvdp_full(w, h),
    ("cvvdp-gpu", "warm_ref"): lambda w, h, b: cvvdp_full(w, h),
    ("cvvdp-gpu", "warm_ref_strip"): lambda w, h, b: cvvdp_strip(w, b),
    ("cvvdp-gpu", "strip"): lambda w, h, b: cvvdp_strip(w, b),
    ("cvvdp-gpu", "strip_pair"): lambda w, h, b: cvvdp_strip_pair(w, b),
    ("cvvdp-gpu", "capped"): lambda w, h, b: cvvdp_capped(w, h),
    ("cvvdp-gpu", "auto"): lambda w, h, b: cvvdp_full(w, h),
}


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
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
