# HDR-feeding validation against real HDR MOS (UPIQ)

**Date:** 2026-06-03 · **host:** lilith (RTX 5070) · **branch:** `feat/hdr-metrics`
**Data:** UPIQ (Mikhailiuk et al. 2021) — **380 HDR compression pairs** (narwaria + korshunov,
JPEG / JPEG-XT), images in absolute cd/m², per-pair subjective **JOD**.
`/mnt/v/datasets/upiq_extracted/` + `upiq_subjective_scores.csv`.
**Harness:** `zenmetrics batch --hdr --hdr-transfer {pu-clamp|pq|pu-rescale}` over
`/tmp/upiq_hdr_pairs.tsv`; correlation `scripts/hdr/upiq_corr.py`.

## Why this exists

The earlier "faithful HDR" work (chunks 4a/4b) asserted improvements without checking
the literature or correlating against HDR MOS. Two errors surfaced:

1. **`to_pu_rgb8` clamped PU21 to u8 255** — but PU21 (Mantiuk-Azimi PU21, PCS 2021) is
   designed to output full-float values up to ~600 (100 cd/m²→256), fed to the metric at
   full precision with `DynamicRange=256`, **never clamped**. The clamp collapses the
   entire >~100 cd/m² highlight range — the exact signal PU21 exists to preserve.
2. **"cvvdp is THE HDR metric"** — dataset-specific. HDR-VDP-2 is the field's gold standard;
   ssim2/butteraugli/cvvdp trade places across datasets.

## Reference bars (UPIQ HDR, pre-computed scores in the dataset CSV, |SRCC| vs JOD)

| metric | \|SRCC\| | note |
|---|---|---|
| HDR-VDP-2.2 | **0.812** | gold standard |
| PU_SSIM (full-precision PU21 + SSIM) | 0.740 | |
| PU_FSIM (PU21) | 0.719 | vs naive FSIM **0.457** — PU21 lifts +0.26 |
| PU_PSNR (PU21) | 0.549 | vs naive PSNR 0.461 |

This reproduces the PU21 paper's central claim: **full-precision PU21 ≫ naive feeding.**

## Our metrics × feeding (|SRCC| vs JOD, n=380)

| config | \|SRCC\| | \|PLCC\| |
|---|---|---|
| **cvvdp — faithful linear-planes** | **0.758** | 0.722 |
| dssim — **pu-rescale** | **0.660** | — |
| ssim2 — **pu-rescale** | **0.652** | 0.619 |
| dssim — pq | 0.629 | 0.424 |
| butteraugli-gpu — faithful linear-planes | 0.628 | 0.196 |
| ssim2 — pq | 0.617 | 0.570 |
| dssim — pu-clamp | 0.584 | 0.407 |
| butteraugli (CPU) — pu-rescale u8 | 0.564 | 0.172 |
| **ssim2 — pu-clamp (the bug)** | **0.551** | 0.537 |

**pu-rescale is the best SDR feeding for BOTH ssim2 (0.652) and dssim (0.660)**, and the
clamp is the worst for both — an unambiguous result.

**Butteraugli faithful-vs-u8:** the faithful linear-planes butteraugli-gpu (0.628) beats
u8-rescale CPU butteraugli (0.564) by +0.06 — so the faithful linear feeding is justified
for butteraugli (it has a native linear-planes entry). ssim2/dssim have no such entry, so
pu-rescale-u8 is their best available path. (CPU vs GPU are different butteraugli impls, so
this is suggestive, not a same-impl A/B.)

## Conclusions (measured, not asserted)

1. **The u8 PU clamp is empirically the worst feeding** (ssim2 0.551, dssim 0.584). Confirmed bug.
2. **The fix is no-clamp PU**: rescaling PU21 to fit u8 (`pu-rescale`) recovers ssim2 to
   **0.652 (+0.10 SRCC)** — better than PQ (0.617) and far better than the clamp. PQ is a
   close second and is the simplest (fits [0,1] by design). **Default is now `pu-rescale`.**
3. **Faithful cvvdp linear-planes (0.758) is the strongest of our metrics** — beats the
   reference PU_SSIM (0.740), near HDR-VDP-2.2 (0.812). That path is validated; kept as the
   cvvdp `--hdr` default.
4. **Faithful butteraugli-gpu (0.628 SRCC)** is decent but its raw output is highly
   non-linear vs JOD (PLCC 0.196 — needs a logistic rescale before any absolute use).
5. **cvvdp is NOT universally best** — on UPIQ it's our strongest, but on AIC-HDR2025
   (PQ-HDR, different config) it underperformed ssim2/butteraugli. HDR-VDP-2 is the gold
   standard and is absent from our stack (a documented gap, no open Rust port).

## What changed in code

- `hdr.rs`: `to_pu_rgb8` (clamp) → `to_sdr_rgb8(img, HdrTransfer)` with `pu-clamp | pq |
  pu-rescale`; added `pq_oetf` (ST.2084). `--hdr-transfer` flag (default `pu-rescale`).
- cvvdp + butteraugli-gpu keep their faithful linear-planes paths (validated above).
