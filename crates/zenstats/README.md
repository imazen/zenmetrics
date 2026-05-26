# zenstats

Paper-correct IQA statistical panel — Mohammadi 2025 / Wu 2017 / ITU-T
P.1401 definitions, in one place, with the full 6-stat panel plus the
A-vs-B comparison machinery (MRR z-test, bootstrap CI delta, decisive
rule per § A.9 of `PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM.md`).

## What's in the box

| Function | Returns | Source |
|---|---|---|
| `spearman`, `pearson`, `kendall_tau` | rank/Pearson correlations | Textbook |
| `outlier_ratio` | corpus-σ Outlier Ratio | Mohammadi Eq 2-4 / ITU-T P.1401 § C.4 |
| `outlier_ratio_per_sample` | per-stimulus σᵢ Outlier Ratio | Mohammadi § IV-B |
| `pwrc_sa_st_auc` | normalised SA-ST AUC (in [0, 1]) | Mohammadi 2025 § VII Figure 4 |
| `sa_st_curve` | (ST, SA(ST)) curve points for plotting | Mohammadi § VII |
| `pwrc_proxy_weighted_rank` | weighted-rank-Pearson proxy | pre-2026-05-26 zensim panel.rs body |
| `z_rmse`, `z_rmse_per_sample` | corpus-σ / per-stimulus-σ Z-RMSE | Mohammadi Eq 6 |
| `rescale_logistic` | 4-parameter logistic rescale | Mohammadi § IV-A |
| `compute_panel` / `compute_light_panel` | full 6-stat aggregator | this crate |
| `mrr_h`, `phi`, `two_sided_p` | Meng-Rosenthal-Rubin paired z-test | Meng, Rosenthal & Rubin 1992 |
| `bootstrap_ci_delta`, `decisive` | A-vs-B decision gate | § A.9 |

## Polarity

Bake / metric output can be either **distance-shaped** (low = high
quality) or **score-shaped** (high = high quality). `compute_panel` is
polarity-tolerant out of the box: SROCC / KROCC are `.abs()`, and PLCC
/ OR / PWRC / Z-RMSE are computed on the 4-param-logistic-rescaled
prediction (which absorbs polarity AND saturation). Passing raw
distance-shaped values **directly** into `outlier_ratio` or
`pwrc_sa_st_auc` will give a spurious 100 % OR and PWRC ≈ 0.

## Feature flags

* `parallel` (default) — enable rayon parallelism in
  `bootstrap_ci_delta`. Strip when embedding in a `no_std + alloc`
  target or when the caller controls its own thread pool.

## History

Extracted on 2026-05-26 from zensim's `zensim-validate/src/panel.rs`
— the prior pre-2026-05-26 panel shipped a two-level-z-score-residual
OR and a weighted-rank-Pearson PWRC proxy that did **not** match the
paper definitions; both were replaced in the same session in which the
crate was carved out. See `imazen/zensim`'s `CHANGELOG.md` for the
BREAKING semantics migration note.

The crate exists because the same statistical math was reimplemented
across `zensim`, `zenanalyze`, `coefficient`, `zenmetrics`, and
`jxl-encoder` — sometimes correctly, often subtly wrong. The original
audit lives at `imazen/zensim`'s
`benchmarks/dedup_VERIFIED_synthesis_2026-05-26.md` (Tier-2 #7).

## License

MIT OR Apache-2.0
