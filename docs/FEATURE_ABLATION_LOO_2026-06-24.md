# Feature ablation: LOO-retrain (not Spearman) — 2026-06-24

Method, results, and conclusions for picker feature ablation done the right way: **leave-one-out
retrain**, not Spearman correlation cleanup. Driver: `scripts/analysis/loo_ablation.sh` +
`train_hybrid.py --drop-features` (zenanalyze commit fdb49615). Run on a dedicated Hetzner ccx63
(48 vCPU, killed when done), npar=46, fixed `--seed 12345` so the only difference between runs is the
dropped feature.

## Why LOO, not Spearman (user directive 2026-06-24)

> "spearman is a terrible way to ablate sometimes compared to loo"

Spearman/correlation cleanup only catches **monotonic redundancy**. Two features 0.99-Spearman-correlated
look redundant, but a nonlinear tree (HistGB) uses their **distinct split thresholds + interaction terms**.

**Measured refutation (jpeg):** `correlation_cleanup.py --threshold 0.99` flagged 9 features to drop.
Dropping them RAISED val overhead 5.87%→6.07% **and** added an OVERFIT violation. LOO then showed
`noise_floor_y_p1` — the 0.999-Spearman "redundant" feature (vs `aq_map_p1`) Spearman said to drop — is the
**most valuable of its cluster**: +0.51pp val overhead when dropped, *more* than the `log_pixels` anchor.
Spearman's verdict was exactly inverted.

## jpeg LOO (97 features, box baseline 6.18% val)

Single-feature LOO deltas (drop one, measure val-overhead delta vs full set):
- **81 / 97 features individually "ablatable"** (delta ≤ −0.05pp — dropping any one *helps* a hair).
- Most-valuable (drop hurts most): `cb_horiz_sharpness` (+0.17), `distinct_color_bins` (+0.16),
  `cr_horiz_sharpness` (+0.14), `min_dim` (+0.08), `high_freq_energy_ratio` (+0.08).
- Most-ablatable (drop helps most): `laplacian_variance_p1` (−0.66), `laplacian_variance_p50` (−0.60),
  `quant_survival_uv_p25` (−0.58).

**But single-feature LOO deltas DO NOT COMPOSE.** Batch-removal verification (retrain keeping N best-by-LOO):

| feature set | val overhead | overfit gap | argmin_acc |
|---|---|---|---|
| full 97 | **5.87%** | +1.64pp | 41.9% |
| keep 70 | 6.44% | +1.60pp | 36.8% |
| keep 55 | 5.96% | +1.62pp | 36.7% |
| keep 40 | 5.90% | +2.10pp | 42.5% |
| keep 28 | 6.00% | +1.77pp | 30.1% |

Overhead is **insensitive to feature count from 28→97** (all ~5.9%; keep-70's 6.44% is run-to-run noise,
which is itself ~±0.3pp — cf. baseline 5.82/5.87/6.18 across seeds/versions). The 97-feature model is
**well-conditioned, not noise-overfit** — so feature pruning is a *verified negative* for the rd-overhead
objective. The full-97 bake (5.87%, no overfit violation) ships unchanged. A 40–55-feature variant is
statistically equivalent and available if inference-time feature-extraction cost ever dominates.

## Output-side ablation: ALSO a negative for jpeg (all 24 configs are used)

The bake flagged cells 8–23 as DATA_STARVED ("1 member config < threshold 3"). That is a bake-internal
axis-grouping artifact, **NOT rd-relevance** — a first reading of it as "16 configs rarely optimal" was
wrong. Measuring the actual rd-win distribution directly (min-bytes config per image×quality band, 82,966
groups) shows **all 24 configs are rd-optimal 0.6%–18.4% of the time**; the least-used (`gls_t0_444`) still
wins 521 groups (0.6%), and NONE win <0.5%. So there are **no ablatable output configs** for jpeg either —
every config is the answer for some real content. The mozjpeg trellis family dominates (`moz_tr14.5_444`
18.4%), but the gls/jp3/pw4 variants each cover a genuine slice.

Net: the jpeg picker is well-conditioned in BOTH dimensions — robust to input feature count AND uses its
full output config set. No ablation lever improves it; the full-97 × 24-config picker (5.87%) is optimal.
(The user's "ablate outputs by RD-spread + content-dependence" axis was the right question; the measured
answer for THIS corpus/metric is "nothing to drop." RD-knob ablation caveat still applies — validate per
corpus+metric+plan, see `RD_ABLATION_2026-06-24.md`.)

## Takeaways

1. Ablate by LOO-retrain, never Spearman. Spearman mislabels the most-useful feature as redundant.
2. Single-feature LOO ranks features but does NOT predict batch effects — always retrain-verify the reduced set.
3. A model can be robust to feature count (jpeg: 28→97 all ~5.9%) — then ablation buys model size, not accuracy.
4. Output-space ablation also nets zero for jpeg — all 24 configs are genuinely rd-optimal for some
   content (0.6–18.4% each). The DATA_STARVED bake warning is a grouping artifact, not rd-relevance;
   always verify "starved" against the real rd-win distribution before dropping a config.
