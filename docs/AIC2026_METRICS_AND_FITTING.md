# AIC2026 metric remappings, JND normalization, and naming ledger

Working notes for the AIC2026 (JPEG AIC-4) corpus: how each `JND_*` score is
produced, what our measured errors look like once JND-normalized, what to fix
first, and how to name/qualify our implementations so similarly-named but
algorithmically distinct metrics are not conflated.

This file was missing from the repo; it is now the authoritative record. It was
reconstructed 2026-09-25 from `metrics_fullres.csv` (NFS export at
`/tmp/v_ro/input/datasets/aic2026/`, DaRUS mirror), the AIC-4 paper
(arXiv:2607.22783), and the benchmark fitting-tool mapping table provided by
the user.

## Two different "JND" objects — do not conflate

1. **`JND_*` dataset columns.** The published CSV carries 7 precomputed JND
   columns next to raw metric columns: `JND_CVVDP`, `JND_SSIMULACRA2`,
   `JND_PSNR-Y`, `JND_SSIM`, `JND_MS-SSIM`, `JND_PSNR-HVS`, `JND_VMAF-neg`.
   Each is a deterministic function of its own metric's raw score (verified:
   every unique raw value maps to a unique JND value over all 9,618 rows;
   non-monotonicity is confined to ≤6 knots of ~9,600, a CSV/fit artifact).

2. **Benchmark fitted mappings.** The AIC-4 evaluation UI can fit/display a
   `metric → JND` remap under user-chosen settings. The captured table below
   uses: full-resolution images, ground truth "Curve recon", method "Simple",
   optimized for RMSE, no clamp, JND range 0–7, public train+test = 5,768
   images, 91 metrics. These maps are the benchmark protocol's normalization;
   they do **not** reproduce the `JND_*` CSV columns (median residual ~0.6–0.9
   JND) — they are a second, broader fitted mapping covering 91 metrics.

Both are power-law-vs-score-ceiling curves of the form
`JND = a · max(0, b − x)^c` (score-decreasing metrics) or
`JND = a · max(0, x)^c` (score-increasing / distance metrics).

**Cross-check of the pasted fitted maps against the data (2026-09-25):**
they agree with the `JND_*` columns on *family and ceiling* (`b`: 93≈92.9,
97≈96) but disagree on midrange amplitude — median |pasted − JND_col| is
0.57–0.86 JND (SSIMULACRA2/VMAF-neg; up to 12.8 at the extrapolated tail,
e.g. at SSIMULACRA2 x=62 the paste gives 1.36 vs the column's 2.26).
Against the canonical ladder `JND_CVVDP`, the `JND_*` columns are actually
*closer* than the pasted maps are (med 0.35 vs 0.54) — i.e. the paste is a
weaker/different-target fit, not a better one. Likely causes: fit on the
5,768-image public split under "Simple"/no-clamp settings, against
Curve-recon ground truth rather than the published column target. Treat the
pasted table as a leaderboard-view approximation; the `JND_*` columns are
the dataset's pinned record and the safer normalization to gate on. Our
deltas are reported under both anyway — since |f(ours) − f(published)| uses
the same f on both sides, the remap uncertainty mostly cancels and the
conclusions are identical either way.

## The JND anchor: CVVDP

Distortion levels were selected by CVVDP, so `JND_CVVDP` is the dataset's
canonical JND scale. The paper documents it as a fit against AIC-3 subjective
data:

```
JND_CVVDP = 3.1889 · (10 − CVVDP)^1.0129        # verified vs CSV to 5e-11
```

Display preset used by the dataset: `standard_fhd` — 37.84 px/deg, 200 cd/m²
peak, 0.2 cd/m² black, 0.3979 cd/m² reflected light.

Dataset design targets ≈0.2–4.0 JND; the subjective core is ≈0–2.5 JND.
Higher JNDs (up to ~16 in `JND_SSIMULACRA2`, ~12 in `JND_VMAF-neg`) are
extrapolated remaps at the extreme-distortion tail, not subjectively
validated.

## Fitted JND mappings — benchmark table (authoritative)

Captured from the fitting tool 2026-09-25 (partial listing; more rows exist in
the full 91-metric table — extend as they are captured):

| Metric | Fitted JND mapping |
|---|---|
| Butteraugli | `0.7624 · max(0, x)^0.7387` |
| Butteraugli-3norm | `0.9074 · max(0, x)^0.8905` |
| HaarPSI | `26.64 · max(0, 1 − x)^1.139` |
| SSIMULACRA2 | `0.01165 · max(0, 93 − x)^1.389` |
| VSI-k4 | `67.34 · max(0, 1 − x)^0.5689` |
| FSIMc-k4 | `75.26 · max(0, 1 − x)^0.7668` |
| VSI-k3 | `81.52 · max(0, 1 − x)^0.6448` |
| GMSD | `51.99 · max(0, x)^0.8995` |
| PSNR-HVS-Y | `5.43e-4 · max(0, 58 − x)^2.706` |
| VMAF-neg v0 | `0.174 · max(0, 97 − x)^0.9738` |

Notes:

- `k3`/`k4` are implementation-parameter variants of the same metric (VSI's
  pooling weights). The CSV publishes a single `VSI`/`FSIMc` column; which `k`
  it corresponds to is unresolved — our deltas are similar under either map.
- `PSNR-HVS-Y`'s map expects the Daala/Xiph `dump_psnrhvs` Y-plane score, not
  our Ponomarenko `psnrhvs` — see the naming ledger below.
- `Butteraugli` / `Butteraugli-3norm` maps exist in the 91-metric fit set even
  though `metrics_fullres.csv` only publishes `proposal-Butteraugli` (an AIC-4
  submission, not libjxl butteraugli).

## `JND_*` CSV columns — reconstructed forms

The dataset's own per-metric display remaps, least-squares recovered over
9,618 rows (same `a·(b−x)^c` family; the `b` ceiling pins at the score's max).
Approximate — the exact published fit parameters are not in the paper text:

| Column | Recovered `JND = a·(b−x)^c` | log-space RMS |
|---|---|---|
| `JND_CVVDP` | `3.1889·(10−x)^1.0129` (documented, exact) | <1e-10 |
| `JND_PSNR-Y` | `0.062·(50−x)^1.41` | 0.007 |
| `JND_PSNR-HVS` | `0.061·(50−x)^1.42` | 0.011 |
| `JND_VMAF-neg` | `0.372·(96−x)^0.83` | 0.017 |
| `JND_SSIMULACRA2` | `0.091·(93−x)^0.94` | 0.033 |
| `JND_SSIM` | `38.9·(1−x)^1.00` | 0.063 |
| `JND_MS-SSIM` | `78.7·(1−x)^0.81` | 0.093 |

The JND_* columns are not interchangeable JND scales: vs `JND_CVVDP`,
Spearman ρ ranges 0.66 (`JND_SSIM`) to 0.93 (`JND_MS-SSIM`/`JND_SSIMULACRA2`),
with median |JND_x − JND_CVVDP| ≈0.3–1.8. Each remap is calibrated to its own
metric's score distribution — applying it to a different implementation
variant conflates implementation error with remap-fit error.

## Our errors, JND-normalized (53-pair stratified subset)

Subset: S01/S03/S19/S38/S57 × AVIF/JXL/JPG/J2K/WEBP × levels {02,09,16}
= 53 pairs, run 2026-09-25 (`~/tmp/aic2026/full/`). "A" = delta under the
authoritative fitted map; "B" = delta under the `JND_*` CSV column;
"C" = JND-equivalent via local regression of metric→`JND_CVVDP` (rough, for
non-anchor metrics with no official map).

| Ours | Column | Space | med \|ΔJND\| | max \|ΔJND\| | Verdict |
|---|---|---|---:|---:|---|
| `gmsd` | `GMSD` | A | 0.0000 | 0.0001 | exact |
| `haarpsi` | `HaarPSI` | A | 0.0002 | 0.0012 | reproduction-grade |
| `ssim2` | `SSIMULACRA2` | A | 0.0003 | 0.0022 | drift immaterial in JND |
| `ssim2` | `SSIMULACRA2` | B | 0.0005 | 0.0041 | same |
| `vmaf` probe | `VMAF-neg` | A | 0.0004 | 0.0027 | reproduction-grade |
| `vmaf` probe | `VMAF-neg` | B | 0.0006 | 0.0038 | same |
| studio-Y PSNR | `PSNR-Y` | B | 0.0001 | 0.0009 | convention identified, ~exact |
| `cvvdp` `--display-model standard_fhd` | `CVVDP` | B | 0.0004 | 0.0043 | reproduction-grade |
| `vsi` | `VSI` | A (k3) | 0.0076 | 0.063 | good |
| `vsi` | `VSI` | A (k4) | 0.0093 | 0.078 | good |
| `vsi` | `VSI` | C | 0.013 | 0.21 | good |
| `fsimc` | `FSIMc` | A (k4) | 0.0065 | 0.31 | acceptable; one JPG-16 tail |
| `fsim` | `FSIM` | C | 0.017 | 0.27 | acceptable |
| `dssim` | `DSSIM` | C | 0.0001 | 0.0004 | exact |
| `iwssim` (studio-Y) | `IW-SSIM` | C | 0.077 | 0.68 | pyiqa-variant gap, visible in JND |
| `msssim` (studio-Y) | `MS-SSIM` | B | 0.067 | **2.64** | MATLAB-variant gap — superseded by `msssim-libvmaf` |
| `ssim-libvmaf` | `SSIM` | B | 0.00004 | 0.0007 | **implemented** — FFI-verified libvmaf port (raw med 1.0e-6 / max 1.8e-5) |
| `msssim-libvmaf` | `MS-SSIM` | B | 0.0001 | 0.0022 | **implemented** — FFI-verified libvmaf port (raw med 1.0e-6 / max 1.6e-5) |
| `vif` (studio-Y) | `VIF` | C | 0.26 | **5.85** | wavelet `vifvec` needed |
| `psnrhvs` (Ponomarenko-Y) | `PSNR-HVS-Y` | A | 1.90 | **6.38** | wrong algorithm family — superseded by `psnrhvs-daala` |
| `psnrhvs` | `PSNR-HVS` | B | 1.82 | 3.63 | wrong algorithm family |
| `psnrhvsm` | `PSNR-HVS` | B | 0.83 | 2.60 | closer but still wrong family |
| `psnrhvs-daala` | `PSNR-HVS` | B | 0.0002 | 0.0032 | **implemented** — FFI-verified Daala port on studio-601 YUV444 (raw med 1.1e-3 / max 4.5e-2 dB) |
| `SSIM` | — | — | — | — | superseded — see `ssim-libvmaf` row |

The `ssim-libvmaf`/`msssim-libvmaf`/`psnrhvs-daala` rows are measured as
`|f(ours) − f(theirs)|` under the reconstructed `JND_*` remap — the remap
is applied to both scores so its own fit error cancels (the published
`JND_MS-SSIM` cell itself deviates from the power-law fit by ~0.13 JND at
x≈0.97). Measuring those rows against `JND_pub` instead would report
med ~0.004 / max ~0.13 — remap misfit, not implementation error.

Reading: raw-score deltas that looked alarming are often immaterial in JND
(SSIMULACRA2's 6.3e-3 drift → ≤0.004 JND), while modest raw deltas on steep
curve regions blow up (MS-SSIM 0.02 raw → 2.64 JND at J2K-16). JND space is
the right prioritization axis.

## Tolerance policy (JND)

| Tier | med \|ΔJND\| | max \|ΔJND\| | Meaning |
|---|---|---|---|
| Reproduction-grade | ≤0.01 | ≤0.05 | convention + implementation identified |
| Acceptable | ≤0.1 | ≤0.5 | tracks the published column; residual impl-variant drift |
| Gap | >0.1 or >0.5 | — | wrong variant/family; new implementation needed |

## Priority queue (JND-ranked)

~~1. **Daala/Xiph `dump_psnrhvs`** → `PSNR-HVS`(+`-Y/-Cb/-Cr`).~~ **DONE** —
   `psnrhvs-daala` (`crates/psnrhvs/src/daala.rs`): integer bin-DCT
   FFI-verified vs vendored libvmaf (≤2e-4 dB). Dataset reproduction on
   studio-601 **YUV444** chroma (the AIC-4 convention — identified via the
   `-Cb`/`-Cr` columns): combined med 1.1e-3 dB / max 4.5e-2 ≈ ≤0.003 JND.
   Parity gotcha: `od_bin_fdct8x8` order is column-DCT→transpose→column-DCT;
   row-first gives ~1.3% higher MSE (0.06 dB).
~~2. **libvmaf `float_ms_ssim` + `float_ssim`** → `MS-SSIM`, `SSIM` anchors.~~
   **DONE** — `ssim-libvmaf` / `msssim-libvmaf`
   (`crates/msssim/src/libvmaf.rs`, FFI-verified incl. 10-bit hbd scaling
   and `round(min_dim/256)` auto-decimate). Studio-601 Y ingress: raw med
   1.0e-6 / max 1.8e-5 ≈ ≤0.002 JND on both columns.
3. **Wavelet `vifvec`** → `VIF` column (pyiqa SP5 steerable-pyramid VIF).
   5.9 JND-equivalent worst case; our `vif` is pixel-domain `vifp`.
4. **pyiqa `iwssim` variant** → 0.68 JND-eq worst; systematic +1.2e-3 raw.
~~5. **`mDCT-PSNR`** → new coverage (Richter QoMEX 2009; official C++ ref
   `thorfdbg/mDCTpsnr`).~~ **DONE** — `crates/mdctpsnr`, CLI `mdctpsnr`.
   Reproduces the *compiled* reference (GCC `-O3 -ffast-math` + AVX2 +
   glibc libmvec) to med 4.0e-7 / max 1.13e-5 dB vs the published column
   over the 53-pair subset — essentially exact; see `DIVERGENCES.md` for
   the codegen-order details (reassociated reductions, `rcpps`+NR mask
   division, libmvec `powf`/`expf`).
6. HDR-VDP-2 ingress config + HDR-VDP-3 → `HDR_VDP_2`/`HDR_VDP_3` columns.

Explicitly deprioritized by JND evidence: SSIMULACRA2 version drift
(≤0.004 JND), and GMSD/HaarPSI/CVVDP/PSNR-Y/VMAF-neg (already
reproduction-grade).

## Naming / qualification ledger

AIC-4 columns that share a metric *name* often measure different
implementations. Qualify ours by reference + variant, not just metric name:

| AIC-4 column | Actual implementation (identified) | Ours | Qualification |
|---|---|---|---|
| `PSNR` | RGB mean-MSE PSNR | — | computable; no dedicated CLI metric |
| `PSNR-Y` | PSNR on `round(16+(65.481R+128.553G+24.966B)/255)` (JPEG studio-601) | `psnr-y-studio601` (or `--luma-ingress yuv601-studio` + `psnr-y`) | exact convention identified |
| `PSNR-YCbCr611` | 6:1:1 YCbCr composite PSNR | — | identified |
| `PSNR-HVS`, `-Y`, `-Cb`, `-Cr` | **Daala/Xiph `dump_psnrhvs`** (7×7 CSF-weighted, per-plane + combined) | `psnrhvs-daala` | ✅ FFI-verified port — studio-601 YUV444 chroma; `psnrhvs`/`psnrhvsm` remain the Ponomarenko MATLAB variant (**different algorithm family**) |
| `SSIM` | libvmaf `float_ssim` (auto-scale `round(min_dim/256)` box decimate, 11×11 Gaussian, zli l·c·s, L=255) | `ssim-libvmaf` | ✅ FFI-verified port |
| `SSIM-pyiqa` | pyiqa SSIM | — | third variant |
| `SSIMc` | MATLAB `ssim()` per-RGB-channel mean | — | computable |
| `MS-SSIM` | libvmaf `float_ms_ssim` | `msssim-libvmaf` | ✅ FFI-verified port; `msssim` remains the Wang-MATLAB variant |
| `MS-SSIM-pyiqa` | pyiqa ms_ssim (matched by ours to 3e-7 on one point) | `msssim` via `yuv601-studio` | variants differ at low quality |
| `VIF` | **pyiqa `vif` = wavelet `vifvec`** (SP5 steerable pyramid) | `vif` | ours = pixel-domain `vifp_mscale` — **different algorithm family** |
| `VIF_vmaf` | libvmaf `vif` feature, Σ4 scales | `vmaf` crate `vif_scales` | matches |
| `VMAF` / `VMAF-neg` / `ADM2` | libvmaf 3.2.1 `vmaf_v0.6.1[neg]` + `adm2` | `vmaf` crate | reproduction-grade |
| `CVVDP` | pycvvdp `standard_fhd` | `cvvdp` + `--display-model` | reproduction-grade |
| `SSIMULACRA2` | "SSIMULACRA v2.1" | `ssim2` → sibling `fast-ssim2` | current lineage; JND delta ≤0.004 |
| `GMSD` | standard GMSD | `gmsd` | exact |
| `HaarPSI` | HaarPSI | `haarpsi` | reproduction-grade |
| `VSI` | VSI, `k3`/`k4` weight variants in fit set | `vsi` | our port = paper weights; which `k` the CSV used is unresolved |
| `FSIM`/`FSIMc` | FSIM/FSIMc, `k4` variant in fit | `fsim`, `fsimc` | acceptable; JPG-16 tail 0.3 JND |
| `DSSIM` | dssim-core ^3.4 | `dssim` | exact |
| `IW-SSIM` | pyiqa `iwssim` on studio-Y | `iwssim` | ours = Python-IW-SSIM oracle variant; +1.2e-3 systematic vs pyiqa |
| `proposal-Butteraugli` | AIC-4 **submission** (not libjxl) | `butteraugli` | cannot reproduce — different metric entirely |
| `proposal-mDCTPSNR`, `proposal-DVIFM*` | AIC-4 submissions | — | cannot reproduce |
| `HDR_VDP_2` | HDR-VDP-2.x, display config TBD | `hdrvdp` | needs absolute-nits ingress config |
| `HDR_VDP_3` | HDR-VDP-3 | — | different metric version, unimplemented |
| `mDCT-PSNR` | Richter QoMEX-2009 mDCT-PSNR | `mdctpsnr` | ✅ port of official C++ ref (`thorfdbg/mDCTpsnr`); med 4.0e-7 / max 1.13e-5 dB vs column |
| `CW-SSIM`, `NLPD`, `MSSWD`, `FLIP`, `CIEDE2000` | conventional metrics | — | unimplemented |
| `DISTS`, `LPIPS×2`, `PieAPP`, `WaDIQaM`, `DeepDC`, `DreamSim`, `TOPIQ×2`, `AHIQ`, `STLPIPS×2` | deep metrics (torch) | — | out of scope for pure-Rust |

## Method / reproduction

- Corpus: `/tmp/v_ro/input/datasets/aic2026/` (NFS, read-only). PNGs
  stream-extracted via `unzip -p`; per-image S3 endpoints also exist.
- Harness: `~/tmp/aic2026/full/` — `pairs.tsv`, `run_batches.sh`,
  `compare.py`, `results/*.tsv`, `raw/*.rgb` + `vmaf` probe example
  (`crates/vmaf/examples/aic_probe.rs`).
- Fitted-map deltas: `JND(x) =` the table above, applied to ours vs published.
- `JND_*`-column deltas: piecewise-linear evaluation of the published
  x→JND graph (deterministic; denser than any parametric refit).
- Caveats: (a) `JND_*` columns inherit the published implementation's
  conventions — a JND delta measures "disagreement with *their* pipeline",
  not intrinsic metric error; (b) non-anchor JND-equivalents regress against
  `JND_CVVDP` and carry the metric's own validity scatter; (c) everything is
  measured on 53 stratified pairs, not the full 9,618 — tails are samples,
  not bounds.
