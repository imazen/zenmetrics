# UPSTREAM_PARITY_AUDIT — cvvdp-cpu vs gfxdisp/ColorVideoVDP

**Snapshot date:** 2026-05-25
**cvvdp-cpu HEAD:** v0.0.1 + this branch
**Upstream HEAD:** [gfxdisp/ColorVideoVDP](https://github.com/gfxdisp/ColorVideoVDP) `main` (clone at `/tmp/ColorVideoVDP`)
**Reference parity:** pycvvdp v0.5.4 (gold-pinned via cvvdp-gpu shadow-JOD goldens). v0.5.4 stays the lock; v0.5.6 (the latest upstream JSON tag) bundles additional CSF parameters but produces a v0.5.4-compatible JOD scalar.

This document is a **finding-only** audit. Each row pairs an upstream
subsystem with the cvvdp-cpu / cvvdp-gpu line numbers that implement
(or stub out) that subsystem, and labels the gap. The follow-up code
chunks reference these rows by number.

## 1. Display geometry — *what produces pixels-per-degree*

Upstream class: `vvdp_display_geometry` in
[`/tmp/ColorVideoVDP/pycvvdp/display_model.py:431-626`](file:///tmp/ColorVideoVDP/pycvvdp/display_model.py).
Ours: `DisplayGeometry` in
[`crates/cvvdp-gpu/src/params.rs:514-582`](../../cvvdp-gpu/src/params.rs#L514).
cvvdp-cpu re-exports it via [`lib.rs:52`](../src/lib.rs#L52).

| # | Field / path | Upstream | cvvdp-cpu (via cvvdp-gpu) | Status |
|---|---|---|---|---|
| G1 | `resolution` (W, H) | tuple in JSON, set at line 605-609 | `resolution_w`, `resolution_h: u32` | **PRESENT** |
| G2 | `viewing_distance_meters` | line 617 | `distance_m: f32` | **PRESENT** |
| G3 | `viewing_distance_inches` (×0.0254) | line 618 | **derived externally** (caller must convert) | **GAP — minor** |
| G4 | `diagonal_size_inches` | line 622 | `diagonal_inches: f32` | **PRESENT** |
| G5 | `diagonal_size_meters` (÷0.0254) | line 621 | **derived externally** | **GAP — minor** |
| G6 | `fov_diagonal` (VR headsets) | line 614, 478-483 | **absent** (panic + `unimplemented` path) | **GAP — major (blocks `htc_vive_pro`, `standard_hmd`)** |
| G7 | `fov_horizontal`, `fov_vertical` | lines 467-475 | **absent** | GAP — minor (no preset uses) |
| G8 | `pixels_per_degree` direct override | line 612, 510-511 | **absent** | GAP — minor |
| G9 | `distance_display_heights` | line 449-457 | **absent** (auxiliary; no preset uses) | GAP — minor |
| G10 | `pixels_per_degree()` centre eccentricity | `get_ppd()` no-arg, line 510-519 | `pixels_per_degree()` matches | **PRESENT** |
| G11 | `get_ppd(eccentricity)` foveated | line 521-526 | **absent** | GAP — out-of-scope (still-image, no foveation) |
| G12 | `pix2eccentricity` | line 536-556 | **absent** | GAP — out-of-scope |
| G13 | `display_size_m` (in metres) | line 447, 469, etc. | **derived internally only**, not exposed | GAP — minor (debug-print convenience) |
| G14 | `display_size_deg` | line 484-485 | **absent** | GAP — minor |

## 2. Display photometry / EOTF

Upstream class: `vvdp_display_photo_eotf` in
[`/tmp/ColorVideoVDP/pycvvdp/display_model.py:278-389`](file:///tmp/ColorVideoVDP/pycvvdp/display_model.py).
Ours: `DisplayModel` + `Eotf` + `Primaries` in
[`crates/cvvdp-gpu/src/params.rs:53-484`](../../cvvdp-gpu/src/params.rs#L53).

| # | Field / path | Upstream | cvvdp-cpu (via cvvdp-gpu) | Status |
|---|---|---|---|---|
| D1 | `Y_peak` (cd/m²) | line 307 | `y_peak: f32` | **PRESENT** |
| D2 | `contrast` (Y_peak / Y_black) | line 308 | derived from `y_peak / contrast` ctor | **PRESENT** |
| D3 | `Y_black` (computed) | line 374 | `y_black: f32` | **PRESENT** |
| D4 | `E_ambient` (lux) | line 309 | `e_ambient_lux: f32` | **PRESENT** |
| D5 | `k_refl` | line 310 | `k_refl: f32` | **PRESENT** |
| D6 | `Y_refl = E_ambient / π * k_refl` | line 373 | `y_refl: f32` + `compute_y_refl()` | **PRESENT** |
| D7 | `EOTF` enum: `sRGB`, `PQ`, `linear`, `HLG`, numeric gamma | lines 341-364 | `Eotf::{Srgb, Pq, Hlg, Linear, Bt1886, Gamma(f32)}` | **PRESENT (defined)** |
| D7a | `Eotf::forward()` per-channel | line 333-365 | `Eotf::forward()` at [`params.rs:133-183`](../../cvvdp-gpu/src/params.rs#L133) | **PRESENT (defined)** |
| D7b | **`Eotf` is actually USED in `srgb_to_dkl_planar`** | applied per-pixel | hardcoded `srgb_eotf` LUT in [`color.rs:40-42`](../src/color.rs#L40) | **GAP — CRITICAL (every non-sRGB EOTF silently ignored)** |
| D8 | `Primaries`: BT.709 / BT.2020 / DisplayP3 / DciP3 | `colorspace` field at line 167-170 + `color_spaces.json` | `Primaries::*` defined + per-variant `linear_rgb_to_dkl()` matrix | **PRESENT (defined)** |
| D8a | **`Primaries` is actually USED in the planar conv** | applied via matrix mul | hardcoded `SRGB_LINEAR_TO_DKL` import in [`color.rs:11,46`](../src/color.rs#L46) | **GAP — CRITICAL (BT.2020-PQ silently uses BT.709 matrix)** |
| D9 | `exposure` multiplier | line 312, 342-359 | **absent** | **GAP — minor** (only `iphone_14_pro_hdr_vert`-class HDR uses non-1.0 exposure in upstream; none in stock display_models.json. Reserved for future opt-in.) |
| D10 | HLG OOTF gamma | line 351-355 + `hlg_system_gamma()` | `hlg_system_gamma()` at [`params.rs:241-252`](../../cvvdp-gpu/src/params.rs#L241) | **PRESENT (defined)** but not threaded through |
| D11 | `is_input_display_encoded()` (gates PU-encode vs no-op) | line 315-317 | **absent** | GAP — minor (cvvdp-cpu doesn't ship PU21) |

## 3. Display presets

Upstream JSON: `/tmp/ColorVideoVDP/pycvvdp/vvdp_data/display_models.json`
(20 presets). Ours: `DisplayGeometry::STANDARD_4K` and
`DisplayModel::STANDARD_4K` constants only (each 1 preset).

| # | Preset name | Resolution | Distance | Peak | Contrast | Ambient | Colorspace | Status |
|---|---|---|---|---|---|---|---|---|
| P1 | `standard_4k` | 3840×2160 | 0.7472 m | 200 | 1000 | 250 | sRGB | **PRESENT** (`STANDARD_4K`) |
| P2 | `standard_hdr_pq` | 3840×2160 | 0.7472 m | 1500 | 1e6 | 10 | BT.2020-PQ | **GAP** |
| P3 | `standard_hdr_hlg` | 3840×2160 | 0.7472 m | 1500 | 1e6 | 10 | BT.2020-HLG | **GAP** |
| P4 | `standard_hdr_linear` | 3840×2160 | 0.7472 m | 1500 | 1e6 | 10 | BT.709-linear | **GAP** |
| P5 | `standard_hdr_linear_dark` | 3840×2160 | 0.7472 m | 1500 | 1e6 | 0 | BT.709-linear | **GAP** |
| P6 | `standard_hdr_linear_zoom` | 3840×2160 | 0.25 m | 10000 | 1e6 | 10 | BT.709-linear | **GAP** |
| P7 | `standard_fhd` | 1920×1080 | 0.6 m | 200 | 1000 | 250 | sRGB | **GAP** |
| P8 | `standard_hmd` | 1440×1600 | 3 m | 100 | 1000 | 0 | sRGB | **GAP** (needs fov_diagonal) |
| P9 | `standard_phone` | 2400×1080 | 0.4 m | 500 | derived | 250 | sRGB | **GAP** |
| P10 | `sdr_4k_30` | 3840×2160 | 0.6 m | 100 | 1000 | 250 | sRGB | **GAP** |
| P11 | `sdr_fhd_24` | 1920×1080 | 0.6 m | 100 | 1000 | 250 | sRGB | **GAP** |
| P12 | `htc_vive_pro` | 1440×1600 | 3 m | 133.3 | 1333 | 0 | sRGB | **GAP** (needs fov_diagonal) |
| P13 | `iphone_12_pro` | 2532×1170 | 0.508 m | 825 | 2062500 | 250 | sRGB | **GAP** (needs `viewing_distance_inches`) |
| P14 | `iphone_14_pro` | 2532×1170 | 0.508 m | 1025 | 2562500 | 250 | sRGB | **GAP** |
| P15 | `iphone_14_pro_vert` | 1170×2532 | 0.508 m | 1025 | 2562500 | 250 | sRGB | **GAP** |
| P16 | `iphone_14_pro_hdr` | 2532×1170 | 0.508 m | 1590 | 3975000 | 10 | BT.2020-HLG | **GAP** (HDR HLG) |
| P17 | `iphone_14_pro_hdr_vert` | 1170×2532 | 0.508 m | 1590 | 3975000 | 10 | BT.2020-HLG | **GAP** |
| P18 | `ipad_pro_12_9` | 2732×2048 | 0.508 m | 600 | 1621 | 250 | sRGB | **GAP** |
| P19 | `macbook_pro_16` | 3072×1920 | 0.635 m | 500 | 1351 | 250 | sRGB | **GAP** |
| P20 | `lg_oled_2017_sdr` | 3840×2160 | 2.5654 m | 272 | 19428 | 100 | sRGB | **GAP** |
| P21 | `lg_oled_2017_hdr` | 3840×2160 | 2.5654 m | 754 | 19842 | 100 | sRGB | **GAP** |
| P22 | `eizo_CG3146` | 4096×2160 | 0.73406 m | 300 | 3000 | 0 | sRGB | **GAP** |
| P23 | `65inch_hdr_pq_4knit` | 3840×2160 | 1.98 m | 4000 | 1e6 | 5 | BT.2020-PQ | **GAP** |
| P24 | `65inch_hdr_pq_2Knit` | 3840×2160 | 1.98 m | 2000 | 1e6 | 5 | BT.2020-PQ | **GAP** |
| P25 | `65inch_hdr_pq_1Knit` | 3840×2160 | 1.98 m | 1000 | 1e6 | 5 | BT.2020-PQ | **GAP** |
| P26 | `lg_oled_2026_hdr_pq` | 3840×2160 | 2.2 m | 3000 | 6e6 | 5 | BT.2020-PQ | **GAP** |

**Findings**: 25 / 26 named upstream presets are not callable via a
single constant. The cvvdp-gpu / cvvdp-cpu API has every primitive
field needed; what's missing is the named-preset surface AND the
EOTF + Primaries plumbing through `color::srgb_to_dkl_planar`.

## 4. CSF parameters

Upstream JSON: `cvvdp_parameters.json`. Ours: vendored LUT in
[`crates/cvvdp-gpu/src/kernels/csf_lut/v0_5_4.rs`](../../cvvdp-gpu/src/kernels/csf_lut/v0_5_4.rs)
+ `CsfChannel` enum (per-channel `precompute_logs_row`).

| # | Field | Upstream `cvvdp_parameters.json` | cvvdp-cpu | Status |
|---|---|---|---|---|
| C1 | `csf` selector ("weber_fixed_size") | string field | hardcoded — only weber_fixed_size LUT vendored | **PRESENT (only weber_fixed_size)** |
| C2 | `csf_sigma` | -1.5 | folded into vendored LUT | **PRESENT (baked)** |
| C3 | `sensitivity_correction` | -0.2797423 dB | imported `SENSITIVITY_CORRECTION_DB` constant | **PRESENT** |
| C4 | `rho` (band frequencies) | derived from PPD | `band_frequencies()` returns Vec<f32> | **PRESENT** |
| C5 | `CSF_BASEBAND_RHO` (baseband freq) | hardcoded | imported via cvvdp-gpu | **PRESENT** |
| C6 | per-channel `CH_GAIN` (A, RG, VY) | derived | `cvvdp_gpu::kernels::masking::CH_GAIN` | **PRESENT** |
| C7 | `LOG_L_BKG_AXIS` (32-entry axis) | computed | imported via cvvdp-gpu | **PRESENT** |
| C8 | Alternative LUTs (weber, log, dkl_cone, none, weber_supra) | 6 alternatives in JSON | **only weber_fixed_size** | **GAP — minor (no production need for alternates)** |
| C9 | Manual / runtime CSF param override | JSON-loaded | **absent** (LUT is `const`) | **GAP — minor (precluded by Strict goldens)** |

## 5. Masking parameters

Upstream JSON section + paper §3.4. Ours: `cvvdp_gpu::kernels::masking`.

| # | Field | Upstream JSON | cvvdp-cpu (via cvvdp-gpu) | Status |
|---|---|---|---|---|
| M1 | `mask_p` | 2.264 | `MASK_P` const | **PRESENT** |
| M2 | `mask_q` (per-channel) | `[1.30, 2.89, 3.68, 3.59]` (4 = A, RG, VY, Y_t) | `MASK_Q: [f32; 3]` (still-image: 3 channels) | **PRESENT** (4th drops temporal) |
| M3 | `mask_c` | -0.795 | `MASK_C` const | **PRESENT** |
| M4 | `d_max` | 2.564 | `D_MAX` const | **PRESENT** |
| M5 | `xcm_weights` (4×4 = 16 floats) | full | `XCM_3X3: [f32; 9]` (still-image 3-ch) | **PRESENT** (4th drops temporal) |
| M6 | `xchannel_masking: "on"` | bool toggle | always on | **PRESENT (matches default)** |
| M7 | `pu_dilate` | 3 | `PU_PADSIZE` (= 3) | **PRESENT** |
| M8 | `pu_blur_kernel_1d` | derived from `pu_dilate` | `PU_BLUR_KERNEL_1D` | **PRESENT** |
| M9 | `masking_model: "mult-mutual"` | string | hardcoded | **PRESENT** (only model implemented) |
| M10 | `local_adapt: "gpyr"` | string | hardcoded (Weber pyramid local adaptation) | **PRESENT** |

## 6. Pooling parameters

Upstream JSON: `beta`, `beta_t`, `beta_tch`, `beta_sch`, `image_int`,
`baseband_weight`. Ours: `cvvdp_gpu::kernels::pool`.

| # | Field | Upstream JSON | cvvdp-cpu | Status |
|---|---|---|---|---|
| O1 | `beta` (spatial) | 2 | `BETA_SPATIAL` (= 2.0) | **PRESENT** |
| O2 | `beta_t` (temporal) | 2 | n/a (still image) | **PRESENT-by-skip** |
| O3 | `beta_tch` (transient channels) | 4 | n/a (still image; T_t dropped) | **PRESENT-by-skip** |
| O4 | `beta_sch` (spatial chunks) | 4 | `BETA_BAND` (= 4) and `BETA_CH` (= 4) | **PRESENT** |
| O5 | `image_int` | 0.578 | `IMAGE_INT` | **PRESENT** |
| O6 | `baseband_weight` | `[0.0036, 1.66, 4.12, 25.26]` | `BASEBAND_W` (3-ch slice: A/RG/VY drop Y_t) | **PRESENT** |
| O7 | `per_ch_w` (channel weights) | `ch_chrom_w=1`, `ch_trans_w=0.808` | `PER_CH_W` | **PRESENT** |

## 7. JOD scaling

Upstream JSON: `jod_a`, `jod_exp`. Ours: `cvvdp_gpu::kernels::pool::met2jod`.

| # | Field | Upstream JSON | cvvdp-cpu | Status |
|---|---|---|---|---|
| J1 | `jod_a` | 0.04396 | `JOD_A` const | **PRESENT** |
| J2 | `jod_exp` | 0.9302 | `JOD_EXP` const | **PRESENT** |
| J3 | piecewise knee at Q=0.1 | per upstream `met2jod` | `met2jod()` | **PRESENT** |

## 8. DKL color conversion (re-stated for completeness)

Upstream: `lms2006_to_dkld65` (line 35-42) composed with `XYZ_to_LMS2006`
(line 17-20) composed with per-primaries `RGB_to_XYZ` (line 27-33,
plus `color_spaces.json` for BT.2020 + Display P3). Ours: three
precomputed const matrices in `params.rs` (`SRGB_LINEAR_TO_DKL`,
`BT2020_LINEAR_TO_DKL`, `DISPLAY_P3_LINEAR_TO_DKL`) +
`Primaries::linear_rgb_to_dkl()` accessor.

Defined ✅ — but not threaded through `srgb_to_dkl_planar`. See D8a.

## 9. Pyramid

Upstream: `pycvvdp/lpyr_dec.py` Gaussian-Laplacian pyramid +
`band_frequencies` band-truncation rule.
Ours: `crates/cvvdp-cpu/src/pyramid.rs` mirrors
`cvvdp_gpu::kernels::pyramid::band_frequencies`.

| # | Component | Upstream | cvvdp-cpu | Status |
|---|---|---|---|---|
| Y1 | Burt-Adelson 5-tap separable filter | `lpyr_dec.py` | `pyramid::PyramidScratch` + `weber_contrast_pyr` | **PRESENT** |
| Y2 | Ceil-halving reduce | `lpyr_dec.py` | `reduce_to_quad_avg` | **PRESENT** |
| Y3 | Band truncation < 0.2 cy/deg | `band_frequencies` | matches | **PRESENT** |
| Y4 | Local adaptation L_bkg | per ref achromatic plane | `log_l_bkg` per band | **PRESENT** |
| Y5 | `pad_replicate` border mode | `lpyr_dec.py` | matches | **PRESENT** |

## 10. Diffmap

Upstream: emit per-pixel D_px via `vdp_metric.compute_diff_map`. Ours:
[`crates/cvvdp-cpu/src/diffmap.rs`](../src/diffmap.rs).

Diffmap is a cvvdp-cpu **extension** (upstream returns scalar JOD +
optional dump of per-channel maps; we emit a fully-pooled D_px so
butteraugli-loop callers can iterate per-pixel). NOT a parity gap.

## 11. Multi-mode (PerfMode) + auto fallback

cvvdp-cpu has `PerfMode::Strict` only. Upstream pycvvdp has no
"Fast" mode. Strict matches upstream. **Not a gap.**

## 12. Foveation / saliency

Upstream: `pycvvdp/cvvdp_ml_metric.py` ships a `cvvdp_ml_saliency`
ML model for foveated weighting. We don't ship this.
**Out-of-scope** for still-image / web-encoder use-case (per
cvvdp-gpu README) — surface only if requested.

## 13. Temporal channels

Upstream: 4 channels (A, RG, VY, Y_t) for video. We ship 3 (A, RG, VY)
for still-image only. The 4th column of `mask_q` / `xcm_weights` /
`baseband_weight` is dropped. **Out-of-scope** — Phase 1c if temporal
ever ships.

## Summary of ACTIONABLE gaps

Ordered by impact-per-effort:

1. **D7b + D8a — EOTF + Primaries plumbing through `color::srgb_to_dkl_planar`.**
   CRITICAL because any caller setting `display.eotf = Eotf::Pq` or
   `display.primaries = Primaries::Bt2020` SILENTLY gets sRGB+BT709
   numerics today. Fix is per-pixel: branch on `display.eotf` for the
   inverse-EOTF step and pick `display.primaries.linear_rgb_to_dkl()`
   for the matrix. Cost: ~150 LOC in `color.rs` + plumbing.
   Impact: unlocks **every HDR preset** (P2/P3/P4/P5/P6 + P16/P17 +
   P23/P24/P25/P26 = 11 presets) and ensures BT.2020/P3 displays score
   correctly.

2. **D9 — `exposure` field on `DisplayModel`.** Cheap once D7b lands.

3. **P2-P26 + G3/G5/G6/G8 — Named-preset surface + auxiliary geometry
   constructors.** Ship `DisplayGeometry::{STANDARD_FHD, STANDARD_HDR,
   SDR_4K_30, IPHONE_14_PRO, ...}` + paired `DisplayModel::*` consts
   that match the upstream JSON line-for-line. Cost: ~400 LOC of
   `const Self = ...` definitions + 1 cross-table preset bundle. Add
   `DisplayGeometry::new_with_fov_diagonal()` for the VR headset
   path (G6).

4. **G14 — `display_size_deg` getter.** Trivial add.

5. **G11/G12 — Eccentricity / foveation.** Skipped (still-image,
   no foveation contract).

## Diff against `cvvdp-cpu` v0.0.1 scope

cvvdp-cpu v0.0.1 explicitly tracked cvvdp-gpu's `host_scalar` —
NOT pycvvdp directly. host_scalar itself hardcodes sRGB+BT709. So
v0.0.1 was "parity with the host_scalar reference, which itself
diverges from upstream on EOTF/Primaries." The v0.1.0 work below
closes the cvvdp-cpu → upstream gap. cvvdp-gpu will need a parallel
chunk (Phase 1c) to honour EOTF/Primaries on the GPU path — flagged
in the chunk plan.

## Test budget

v0.0.1 has 21 tests (9 lib + 6 diffmap-invariants + 4 parity +
2 parity-corpus + 0 pixels-integration). The chunk plan adds:

- ≥ 5 EOTF parity tests (per-EOTF round-trip vs upstream formula).
- ≥ 5 Primaries parity tests (per-primary matrix-mul vs concatenated
  upstream matrices, at f64 then truncated).
- ≥ 8 preset round-trip tests (PPD + display_model fields match
  upstream JSON line-for-line).
- ≥ 4 EOTF + Primaries integration tests (full Cvvdp::score on
  HDR PQ + BT.2020-HLG + linear inputs).

= **22 new tests** → 43 total.

## What this audit does NOT change

- No source file edited.
- No constant moved.
- No public-API name changed.
- Existing 1e-4 JOD parity floor against pycvvdp v0.5.4 / host_scalar
  **stays.**

This is finding-only. Subsequent chunks land code keyed by row number.
