# avif + jpeg knob-ablation — DE-RISKING first-cut (2026-06-28)

De-risking pre-flight for the avif/jpeg arm of the JXL knob-space ablation program
(`docs/JXL_LOSSY_KNOBSPACE_ABLATION_PROGRAM.md`). **Question:** do avif/jpeg's
UNSWEPT broad axes carry **content-dependent RD value worth picking** — enough to let
a picker hug the pareto materially tighter than the best fixed config + q?

**Target metric = SSIMULACRA2 (`ssim2`)** (the production/trustworthy target); `zensim`
reported side-by-side. Verdicts below are the **ssim2** verdict.

## Setup (cheap, local, persist-everything)

- **Corpus:** 16 content-diverse imazen-26 origins (k-means K=16 on 94 zenanalyze
  `feat_*`, centroid-nearest per cluster; train-origin only via `origin_split`, even
  last-digit), each downscaled (Lanczos, downscale-only) to {256,512,768} long-edge →
  **47 sources** across 10 content classes (photo/people/nature/interiors, web-screens,
  line-art plots ×4, NPS/NOAA/patent docs, AI products/illustrations). picks +
  provenance in the data dir.
- **Sweep:** `zenmetrics sweep --plan modes_full --max-deviations 1` (isolated
  main-effects: all-defaults baseline + every single-axis deviation) at zq
  {30,45,60,75,85,92}. **avif 30 distinct cells / jpeg 40 cells.** Both metrics scored
  CPU; encoded bytes + features persisted. avif 7,614 rows / jpeg 10,904 rows, 0 fail.
- Ran LOCALLY under `run-heavy` in ~12 min (avif) + ~2 min (jpeg). **No fleet launched.**
- **NOTE — `filter_sharpness` is NOT an avif knob** (it is a *webp* knob,
  PLAN_SWEEPS §zenwebp); it does not exist in `zenavif/src/encode_plan.rs`. The real
  avif broad axes in `modes_full` are vaq / vaq_strength / seg_boost / cdef /
  tune_still_image / fast_deblock / rdotx / sgr / seg_complex / encode_bottomup / lrf /
  partition_range / complex_prediction / trellis.

## Method

Per axis, per image, an **iso-q RD** metric: at each matched zq convert the value's
(Δlog-bytes, Δscore) vs baseline to a quality-equivalent byte delta using the baseline's
byte-per-quality slope (negative = saves bytes = better). Robust to the
quality-scrambling some knobs cause (a naive global BD-rate mis-signed those — e.g. it
called `complex_prediction` a "50% win" when the raw curves show it *craters* quality at
equal bytes). Per axis:
- `best_fixed%` = bytes a fixed non-default default saves vs baseline (universal win).
- `picker%` = extra bytes a per-image oracle saves over that best fixed value (content value).
- `flip` = max over values of min(frac images materially-better, frac materially-worse) —
  a noise-robust two-sided content-dependence score. **Validation:** `subsampling` (the
  known content-dependent axis already in the picker) scores the highest flip (0.43-0.45),
  confirming the score detects real content-dependence.

Verdict: **flip ≥ 0.15 → PICKER**; else `best_fixed ≥ 0.5%` (not-worse-anywhere) → **CODE=value**; else **CODE/DROP**.

## AVIF — per-axis verdict (ssim2 | zensim)

| axis | broad | flip s2/zs | picker% s2/zs | best_fixed% s2 | ssim2 verdict |
|---|---|---|---|---|---|
| partition_range | ✓ | 0.17 / 0.17 | **2.59** / 2.29 | 0 | PICKER (only notable broad axis) |
| vaq_strength | ✓ | 0.36 / 0.19 | 0.90 / 0.34 | 0 | marginal (low-q veto: vaqs3) |
| seg_boost | ✓ | 0.26 / 0.36 | 0.99 / 0.71 | 0 | CODE/DROP (dominated: +bytes, −quality) |
| vaq | ✓ | 0.17 / 0.23 | 0.28 / 0.28 | 0 | CODE/DROP (marginal) |
| tune_still_image | ✓ | 0.11 / 0.02 | 0.17 / 0.10 | 0 | CODE/DROP (inert) |
| fast_deblock | ✓ | 0.11 / 0.13 | 0.24 / 0.16 | 0 | CODE/DROP (inert; low-q veto) |
| **cdef** | ✓ | 0.02 / 0.02 | 0.05 / 0.05 | 0.29 | **CODE/DROP (inert)** |
| complex_prediction | ✓ | 0 / 0 | 0 / 0 | 0 | CODE=off (cratering — veto) |
| encode_bottomup | ✓ | 0 / 0 | 0 / 0 | 0 | CODE=off (worst low-q veto) |
| rdotx / sgr / lrf / trellis / seg_complex | ✓ | ≤0.15 | ≤0.4 | ≤0.4 | CODE/DROP |
| subsampling *(already picked)* | – | 0.43 / 0.40 | 1.34 / 1.67 | 0 | PICKER (validation) |
| speed *(compute axis)* | – | – | – | best=s2 | the real avif RD lever |

## JPEG — per-axis verdict (ssim2 | zensim)

| axis | flip s2/zs | picker% s2/zs | best_fixed% s2 | ssim2 verdict |
|---|---|---|---|---|
| aq_coupling | 0.47 / 0.47 | 0.40 / 0.34 | **5.31** | PICKER (+ universal component) |
| subsampling | 0.45 / 0.38 | **2.47** / 1.98 | 2.33 | PICKER (strong) |
| trellis_lambda | 0.38 / 0.38 | 0.85 / 0.59 | **5.11** | PICKER |
| xyb_quant | 0.36 / 0.43 | **5.22** / 4.15 | 0 | PICKER but **low-q veto** (bloats low-q) |
| chroma_dist_scales | 0.26 / 0.28 | 0.83 / 0.67 | 0 | PICKER (modest) |
| sharpening | 0.26 / 0.13 | 0.31 / 0.17 | 0 | PICKER (ssim2) / marginal (zensim) |
| quant_family | 0.15 / 0.23 | 0.64 / 0.72 | 0 | borderline |
| aq (on/off) | 0.09 | 0.53 | 3.96 | CODE=off (universal) |
| delta_dc / trellis_lambda2 | ≤0.15 | ≤0.4 | ~4.2-4.4 | CODE (universal) |
| **scans (strategy)** | 0 / 0 | 0 / 0 | 1.52 | **CODE=psrch** (confirms: strategy axis low value) |

## Oracle gap — how much can picking buy? (per-image, iso-quality % over best-fixed)

| codec / scope | metric | best-fixed | mean | p90 | p99 | max | >100% | >200% |
|---|---|---|---|---|---|---|---|---|
| avif full md1 | ssim2 | s2 | 2.8% | 7.2 | 15.2 | 19.3 | **0** | 0 |
| avif broad@fixed-speed | ssim2 | s4-bd10 | 3.4% | 10.5 | 31.0 | 39.0 | **0** | 0 |
| avif full md1 | zensim | s2 | 1.9% | 4.8 | 14.2 | 15.9 | 0 | 0 |
| jpeg full md1 | ssim2 | jp3_tr16_420 | 4.1% | 12.4 | 21.2 | 22.9 | **0** | 0 |
| jpeg full md1 | zensim | jp3_t0_444 | 5.6% | 11.0 | 14.8 | 15.2 | 0 | 0 |

**No catastrophic (>100%) per-image oracle gaps under either metric.** avif's best-fixed
is `s2` (slowest speed) — the dominant avif RD lever is **compute, not the broad knobs**.

## Low-zq per-row tail + VETO scan (ssim2 — where the steeper metric is least settled)

Best-fixed config's own low-q tail is **clean** (avif s2 p99 5.3%, jpeg p99 20%). The
danger is in specific broad-axis cells that produce catastrophic **low-q** rows under
ssim2 — these are **veto candidates** (exclude from the picker / never pick at low q),
not pickable value:

| codec | worst low-zq cells (ssim2 p99 overhead) |
|---|---|
| avif | encode_bottomup **258%**, color_model=rgb 183%, complex_prediction 174%, no-qm 95%, fast_deblock 69%, vaq_strength=3 67% |
| jpeg | xyb_quant 58%, delta_dc(ddc) 48%, pre_blur(pw4) 46%, trellis_lambda2(l216) 44%, delta_dc(+dc) 43%, aq-off 39% |

ssim2 tails run a bit fatter than zensim (expected — ssim2 is steeper/more-discriminating).
zensim is noisier at *high*-q (jpeg baseline high-q zensim p99 141% w/ 4 rows >100% vs
ssim2 66.8%) — corroborating ssim2 as the more trustworthy target.

## HEADLINE — is a full fleet sweep justified, and on which axes?

- **AVIF — NO (for the broad axes). Code the defaults.** The named broad axes
  (vaq/cdef/tune_still_image/fast_deblock) carry **~no content-dependent RD value worth
  picking** (flip ≤ 0.17, picker ≤ 0.28%); several (encode_bottomup, complex_prediction,
  color_model=rgb, vaq_strength=3, fast_deblock) are *actively dangerous at low-q* →
  **veto, don't pick.** Even a perfect ssim2 picker over the whole broad+picker space at
  fixed speed buys only **~3.4% mean** over a good fixed cell with **no catastrophic
  tail**. `partition_range` is the lone marginal candidate (~2.5% picker ceiling). The
  avif pareto is essentially captured by **best-fixed-cell + speed + q + the existing
  sub/bd/qm picker** → a full avif broad-axis fleet sweep is **not** justified by this
  data. (Small universal CODE wins available: bit_depth=10 / seg_complex=on.)

- **JPEG — PARTIALLY YES, on a focused grid (NOT the strategy axis).** `subsampling`,
  `trellis_lambda`, `aq_coupling`, and `xyb_quant` carry real content-dependent RD value
  (flip 0.36-0.47; full-space oracle gap ~4-6%). The **strategy/scans axis confirms low
  value** (CODE=psrch) — consistent with the prior "jpeg picker barely beats best-fixed
  on strategy." → a fuller jpeg sweep is justified on a **focused 4-axis grid
  {subsampling × trellis_lambda × aq_coupling × xyb_quant}** with **low-q veto rules**
  (xyb/pre_blur/delta_dc bloat at low-q). Code the universal axes (aq=off, delta_dc,
  trellis_lambda2).

## Caveats

- Broad axes evaluated at the baseline speed (avif s4) — max-dev-1 probes them as single
  deviations; an axis whose payoff is only at another speed isn't crossed here (per the
  program's locked LEAN P0 decision).
- 47 sources (16 origins × ~3 sizes), ≤768px, CPU metrics. First-cut directional signal,
  not the full ML-discipline corpus. The jpeg "yes" axes warrant the fuller sweep to
  confirm magnitude + fit the picker; the avif "no" is strong enough to code the defaults.

## Data + provenance

`/mnt/v/output/avif-jpeg-knobablation-firstcut-2026-06-28/` (+ Tower mirror): `avif.tsv`
/ `jpeg.tsv` (per-cell rows, both metrics), `{avif,jpeg}_enc/` (7,614 + 10,904 persisted
encoded variants), `{avif,jpeg}_features.tsv`, `picks16.json`, `sources.json`, and the
`*.rdanalysis_*.json` / `*.tail.json` analysis outputs. Scripts: `scripts/sweep/knobablation_firstcut_*.py`.
Binary: `zenmetrics-cli` built `--features sweep,png,jpeg,webp,avif,cpu-metrics,gpu-ssim2`
(gpu-ssim2 supplies the ssim2 params type; the CPU fast-ssim2 scorer does the work — no GPU used).
