//! # zenstats — paper-correct IQA statistical panel
//!
//! Canonical home for the statistical panel used across imazen's zen
//! image-quality stack. Implements the **exact** definitions from
//! Mohammadi 2025 *"Evaluation of Objective Image Quality Metrics for
//! High-Fidelity Image Compression"* (IEEE Access, DOI
//! `10.1109/ACCESS.2026.3669417`), including its source paper for
//! PWRC (Wu et al. 2017 *"A Perceptually Weighted Rank Correlation
//! Indicator for Objective Image Quality Assessment,"*
//! `arXiv:1705.05126`).
//!
//! ## What's here
//!
//! * **Rank-based correlation primitives** — `ranks`, `spearman`,
//!   `pearson`, `kendall_tau`. Average-tie ranks (panel-canonical
//!   convention, not numpy's `argsort(argsort)`).
//! * **Outlier Ratio (Mohammadi Eq 2-4 / ITU-T P.1401 § C.4)** —
//!   `outlier_ratio` (corpus σ on target) and `outlier_ratio_per_sample`
//!   (per-stimulus σᵢ, the form used when bootstrap σ is available
//!   per AIC-3 / CID22 / KonJND). Both expect `predicted` already on
//!   `target`'s scale (use after `rescale_logistic`).
//! * **PWRC (Mohammadi § VII SA-ST AUC)** — `pwrc_sa_st_auc` (the
//!   canonical [0, 1] AUC), `sa_st_curve` (the curve points for
//!   plotting), `pwrc_proxy_weighted_rank` (the pre-2026-05-26
//!   panel.rs body, preserved for forensic comparison).
//! * **Z-RMSE (Mohammadi Eq 6)** — `z_rmse` (corpus σ) and
//!   `z_rmse_per_sample` (per-stimulus σᵢ).
//! * **4-parameter logistic rescale (Mohammadi § IV-A)** —
//!   `rescale_logistic` with a 13-start Levenberg-Marquardt under the
//!   hood; `rescale_affine` as the fallback when the LM fails to
//!   converge.
//! * **Panel aggregator** — `compute_panel` (full 6-stat,
//!   release-grade) and `compute_light_panel` (3-stat,
//!   `O(n log n)` for per-epoch checkpoint selection).
//! * **MRR (Meng-Rosenthal-Rubin) paired-correlation z-test** —
//!   `mrr_h`, `phi`, `two_sided_p`, plus `polarity_factor`,
//!   `bootstrap_ci_delta`, and `decisive` for ship-grade A-vs-B
//!   comparison gates per *PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM.md* § A.9.
//!
//! ## Polarity convention
//!
//! Bake / metric outputs may be either **distance-shaped** (low =
//! high quality) or **score-shaped** (high = high quality). All
//! callers should treat `compute_panel` as polarity-tolerant:
//! SROCC / KROCC are taken `.abs()`, and PLCC / OR / PWRC / Z-RMSE
//! are computed on the 4-param-logistic-rescaled prediction which
//! absorbs both polarity AND saturation. Passing raw distance-shaped
//! values directly into `outlier_ratio` or `pwrc_sa_st_auc` will
//! give a spurious 100 % OR and PWRC ≈ 0.
//!
//! ## History
//!
//! Extracted on 2026-05-26 from zensim's
//! `zensim-validate/src/panel.rs` — the prior pre-2026-05-26 panel
//! shipped a two-level z-score-residual OR and a weighted-rank-Pearson
//! PWRC proxy that did **not** match the paper definitions; both were
//! replaced in the same session in which the crate was carved out.
//! See `imazen/zensim`'s CHANGELOG.md for the BREAKING semantics
//! migration note.

pub mod panel;

pub use panel::{
    Decision, DecisiveOutcome, LightPanel, PanelStats, ValAggregate, bootstrap_ci_delta,
    compute_light_panel, compute_panel, decisive, kendall_tau, mrr_h, outlier_ratio,
    outlier_ratio_per_sample, pearson, phi, polarity_factor, pwrc_proxy_weighted_rank,
    pwrc_sa_st_auc, ranks, rescale_logistic, sa_st_curve, spearman, two_sided_p, z_rmse,
    z_rmse_per_sample,
};
