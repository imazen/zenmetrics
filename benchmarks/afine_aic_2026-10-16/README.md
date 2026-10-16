# A-FINE AIC-4 convention search — 2026-10-16

Black-box search for the convention behind the published JPEG AIC-4
`A-FINE` column (61 full-res pairs, `~/tmp/aic-sub/` images +
`~/tmp/aic-site/site_fullres_subset61.csv` published values + JND map
`~/tmp/aic-site/config.json`). Extends `../afine_aic_probe/` (which
drove the *upstream* torch pipeline, mostly on the PTC crops) — this
round drives **our** port's outputs on the full-res inputs.

## Inputs

- Dumped sub-scores: `dump_*.tsv` — written by
  `crates/zenmetrics-nn/examples/afine_aic_dump.rs` (originally a
  temporary env-gated test, moved to the example in the follow-up):
  per (pair, variant) the raw `qhead`/`dhead` outputs, the NLM-scaled
  `s_r`/`s_d`/`s_f`, the adaptive `lambda`, and the assembled
  `afine_all`.
  Variants: `native` (CLIP-norm + pad32 bicubic up — the shipped
  convention), `r224`/`r384`/`r512` (bicubic pre-resize, then pad32),
  `nonorm` (no CLIP normalization), `bgr` (channel swap).
- Runner: `afine_aic_search.py` — evaluates every sub-score, structured
  combination, fitted affine, 2-term fit and λ-grid against the
  published column, reporting raw |Δ| and site-JND residuals
  (`jnd(x) = a·sign(x−b)·|x−b|^c`, A-FINE a=3.9897 b=2 c=0.6167).

## Data

- `dump_native.tsv` — rten trunk, `native` variant: 60/61 pairs with
  full sub-scores + S67 `afine_all` from the ort-cuda fused graph
  (rten needs ~28 GB of activations for the 3355×2516 pair — beyond
  this box's headroom; ort-cuda agrees with rten to ~2e-7 on every
  shared pair, e.g. S01 −2.84463930 vs −2.84463948).
- `dump_ort_partial.tsv` — ort-cuda fused score only (no sub-scores),
  39 pairs × all 6 preprocessing variants before the ORT CUDA arena
  OOM'd at pair 45 (shape-varying arena accumulation — since fixed,
  see "ort-cuda arena fix" below).
- `dump_ort_full.tsv` — post-fix ort-cuda re-run: all 61 pairs × 6
  variants, zero failures, bit-identical to `dump_ort_partial.tsv` on
  the 234 shared rows.
- `candidates.tsv` — all 703 evaluated formulas over the native dump,
  sorted by site-JND median.
- `candidates_ort_preproc.tsv` — same runner over the ort dump's
  `afine_all` under the 6 preprocessing variants (39 pairs).

## Result: NOT reproduced — no accepted convention

Gate for adding an `afine-aic` CLI variant: ≤0.01 JND median, ≤0.05 JND
max vs the published column (≈0.007/0.034 raw units at x≈−1.9). No
candidate comes close.

Best **raw** (unfitted) conventions:

| formula | JND med | JND max | raw med | raw max | SRCC |
|---|---|---|---|---|---|
| s_f + (s_d − s_r) | 0.106 | 0.594 | 0.073 | 0.36 | 0.907 |
| s_f (fidelity alone) | 0.129 | 0.884 | 0.088 | 0.54 | 0.920 |
| s_f + λ·(s_r − s_d) | 0.134 | 0.920 | 0.092 | 0.56 | 0.882 |
| **afine_all (shipped)** | **1.058** | **1.963** | **0.747** | **1.42** | 0.804 |

Best *fitted* candidate (3 free params): `a·s_f + b·clip(afine_all) + c`
at JND med 0.038 / max 0.46 — still >3× over the median gate with a
fit we cannot justify as a convention. Fitted SRCC ceilings at ≈0.92.

Preprocessing (ort, 39 pairs, raw |Δ| median of `afine_all`):
native 0.689 < bgr 0.685 ≈ nonorm 0.794 << r384 1.23 < r512 1.26 <<
r224 2.18 (plus r224 max 152). Resolution is not the gap — native
full-res ≈ the earlier crop-res run (−2.845 vs −2.84 on S01), and
forcing training-size resizes only worsens it.

## Interpretation

`s_f` alone is the single best term, and the residual `pub − s_f`
(median +0.088) correlates 0.61 with (s_d − s_r) — the published column
behaves like fidelity plus a *damped* naturalness term. But every
structured recombination of the dumped sub-scores leaves median JND
>0.07, and even 3-parameter fits cap SRCC ≈0.92 — a rank disagreement
no monotone convention can repair. The likeliest explanation is that
the AIC-4 site ran a different checkpoint or pipeline step we cannot
observe (upstream ships one `afine.pth`; the site may have retrained
or used a variant). The shipped metric stays upstream-verified
`afine_all` (parity ≤1.5e-5 vs upstream goldens).

S67 (3355×2516, 8.4 MP) is also a resource finding: the rten trunk
needs ~28 GB on it — the only pair we could not score with the
sub-score path on this box. Its `afine_all` came from ort-cuda.

## ort-cuda arena fix (follow-up)

The original ort-cuda sweep OOM'd at pair 45 — ORT's device-side
memory-pattern arena retains a per-shape block pool, so 366
shape-varying runs exhaust VRAM (the last alloc attempt was a 2.4 GB
Softmax). Fix, ported from WD's `ae582a3b` into a shared
`engine::ort_engine::cuda_session`: `with_memory_pattern(false)`,
CPU provider `.with_arena_allocator(false)`, outputs consumed by
move-iteration. Applies to every ort-cuda backend — `OrtCudaPair`
(afine, ckdn, deepdc, dists, dreamsim, lpips/stlpips), `OrtCudaImage`
(arniqa, hyperiqa, maniqa, liqe, musiq, paq2piq), `OrtCudaTrunk` (wd).

Measured, same 61-pair × 6-variant sequence, `/usr/bin/time -v`:

| run | rows scored | Softmax arena failures | peak RSS | wall |
|---|---|---|---|---|
| pre-fix  | 357/366 | 9 (S46/S54/S67 × native,nonorm,bgr) | 1.95 GiB | 105 s |
| post-fix | 366/366 | 0 — S67 `afine_all` now comes through ort-cuda too | 2.00 GiB | 129 s |

All 357 shared rows bit-identical (0.0 abs diff). Host RSS is a wash —
the retention was device-side; the failure signature is gone. afine
ladder parity unchanged: rten 1.42e-5 JND, ort-cuda 1.94e-5 JND.

## Reproduce

```sh
# sub-score dump — needs model dir + pairs list, writes dump tsv.
# variants: native r224 r384 r512 nonorm bgr (comma list, default all)
cargo run --release -p zenmetrics-nn --features ort \
    --example afine_aic_dump -- \
    <model_dir> <pairs.tsv> dump_native.tsv rten native
# ort-cuda additionally needs ORT_DYLIB_PATH + cuDNN 9.19 libs on
# LD_LIBRARY_PATH (cuDNN 9.24 has a ~10 GB conv-workspace regression);
# `just nn-afine-aic-dump <models> <pairs> <out> <engine> [variants]`

# candidate evaluation
python3 afine_aic_search.py --dump dump_native.tsv \
    --site ~/tmp/aic-site/site_fullres_subset61.csv \
    --config ~/tmp/aic-site/config.json > candidates.tsv
```
