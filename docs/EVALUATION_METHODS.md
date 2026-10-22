# Evaluating a metric against human data

How to evaluate any metric in this repository against a human-labelled image
quality dataset, using one implementation of every statistic (`zenmetrics-stats`, named `zenstats` until 2026-10-11).
The same procedure applies to zensim, SSIMULACRA2, Butteraugli, CVVDP and the
rest; nothing here is specific to one metric.

## What to resample: sources and codecs, not rows

An IQA dataset samples source images and codecs. Every distorted image of one
source shares its content, and every image from one codec shares its
artefacts. A row bootstrap treats them as independent and gives intervals
that are too narrow for any claim about new images or codecs.

- Use `zenmetrics_stats::resample::cluster_bootstrap` with `Design::OneWay(source)`,
  or `Design::TwoWay(source, codec)` (Owen's pigeonhole bootstrap) when the
  codecs are crossed with the sources.
- Give a codec applied to only some sources a label per (source, codec), so it
  is nested in its source rather than crossed.
- Put the interval on every criterion and on every paired difference between
  two metrics. Compute both metrics inside one statistic closure so they see
  the same resample.
- **A source drawn twice is two sources.** The statistic receives a `Draw`:
  - sums over rows use `Draw::row_weights`;
  - statistics over pairs of rows pass the draw to `zenmetrics_stats::pairs` (or use
    a `PairTable`), which weights a pair with `Draw::pair_weight`. Within-source
    pairs then count once per copy, not once per pair of copies;
  - statistics that group rows (`per_group_srocc`) or refit something
    (`jnd::out_of_fold`) run on `Draw::expand`, whose copy-aware labels make
    the two copies different groups. Expanding without those labels merges
    the copies into one group and under-covers.
- `Design::TwoWay` is mildly conservative (Owen 2007): with few levels it
  inflates the source and codec variance terms and triples the interaction
  term, so a cross-codec comparison gets the widest intervals.
- With few sources every simple interval under-covers. Measured on a
  synthetic 12-source population (`crates/zenmetrics-stats/tests/coverage.rs`): 95 %
  percentile 0.884, Student `estimate ± t(G−1)·se·√(G/(G−1))` 0.911, exact t
  over per-source values 0.910. Report the Student form beside the percentile
  one with fewer than about 30 sources; with five or so the interval is
  descriptive, and say so.
- Print the row-bootstrap interval beside it only as a labelled comparison.

## Split pairwise agreement into three classes

`zenmetrics_stats::pairs` splits every pair of stimuli by what it compares:

| class | compares | tests |
|---|---|---|
| same source, same codec | two rungs of one ladder | monotonicity (nearly every metric passes) |
| same source, different codec | one image through two codecs | the codec-comparison question |
| different source | different images | calibration across content |

With S sources only about 1/S of all pairs share a source, so a pooled
correlation on a many-source dataset mostly measures the last row. Report
pooled and per-source SROCC side by side (`per_group_srocc`), and report the
cross-codec class on its own.

Two scores per class:

- `class_agreement`: sign agreement, metric ties scored 0.5, target ties
  excluded and counted.
- `expected_agreement`: the share of observers expected to agree with the
  metric's ordering under a Thurstone Case V observer, normalised between
  chance and the oracle ceiling. It needs only the metric's ordering, so no
  mapping to target units is involved. For a JND unit defined at 75 % correct
  choice, use `HumanNoise::JND_75`. Prefer it over `HumanNoise::Measurement`,
  whose σ shrinks as a study adds observers.

## Absolute accuracy: declare the mapping, fit it out of fold

Most metrics do not output the target's units, so absolute accuracy is the
accuracy of the metric plus a mapping. The mapping family can change an error
criterion more than the metric does.

1. **Declare the family before looking at the data** (`zenmetrics_stats::jnd::MapFamily`):
   `Linear`, `Logistic4`, `KneePower` (a·max(0, b − x)^c, the form used to put
   metrics on a JND scale) or `Isotonic` (the best monotone shape).
2. **Fit leave-one-source-out** (`jnd::out_of_fold`), so no stimulus is
   predicted by a map that saw its source. A metric that outputs the target
   unit natively is also scored as delivered, so the mapping's share is
   visible.
3. **Compute statistics on the pooled out-of-fold predictions.** For the
   interval, refit the maps inside the cluster bootstrap (`Draw::expand`, then
   `out_of_fold` with the copy-aware source labels as folds). A bootstrap of
   fixed out-of-fold predictions leaves out the map-fitting variance and is
   too narrow; Bates, Hastie & Tibshirani (2024) show the analogous failure of
   the standard cross-validation interval and propose nested cross-validation.
4. **Headline: RMSE in target units and τ̂** (`jnd::excess_error_sd`), the
   error standard deviation left after removing the target's measurement
   noise. τ̂ does not grow as a study adds observers, and it does not let the
   few stimuli with the smallest σ dominate.
5. **Diagnostics, not rankings:** the normalised error `(pred − target)/σ`
   (`z_rmse_per_sample`) is weighted by 1/σ²; on data whose σ shrinks toward
   zero distortion it mostly scores the near-threshold stimuli and changes
   rank with the mapping family. Report it as a goodness-of-fit check, beside
   the outlier ratio and P.1401 rmse* (`jnd::rmse_star`, which takes each
   stimulus' own interval bounds, so asymmetric JND intervals are handled;
   use d = 0 for out-of-fold predictions).
6. **Charge the map:** report the declared family, a first-order map and an
   out-of-fold isotonic map side by side.
7. **Transfer:** for a map fitted on one dataset and applied to another, report
   zero-shot error, and separately the error after a re-anchor of at most two
   parameters (P.1401 §7.3.3: a mapping compensates offsets and gradient
   without changing the rank order). Zero-shot measures absolute calibration; re-anchored measures shape.

## Raw forced choices

When a dataset releases its raw 2AFC or triplet responses, score them
directly (`zenmetrics_stats::forced_choice`):

- **Prepare the responses like the reconstruction did.** Screen workers and
  drop trap questions and questions that show the same image on both sides.
  Keep "not sure" answers as `Choice::NotSure`; they count half for each side,
  the JPEG AIC convention. On unscreened crowd responses the human scale
  itself can score near chance.
- **Check the unit.** `fit_observer_slope` on the human scale values should
  come out near 0.6745 for a JND defined at 75 %. Report it beside every
  log-likelihood.
- `agreement`: response-weighted agreement with three ceilings, each named:
  the in-sample majority oracle `ceiling_response` (an upper bound), its
  leave-one-response-out counterpart `ceiling_loo`, and
  `observer_agreement`, the expected agreement of two different observers
  (p² + (1 − p)², the human ceiling LPIPS reports). Compare datasets with
  different numbers of responses per question on the cross-fitted ones.
- `log_likelihood`: once a metric's map is frozen, the held-out
  log-likelihood of the choices under P = Φ(ΔJND / s), normalised between
  chance and the per-group ceiling, in sample (`normalized`) and bias
  corrected (`normalized_corrected`). It is a proper scoring rule, keeps
  magnitudes, and weights comparisons by real observer disagreement.

Score unboosted comparisons on native pixels first, and declare which pixels
(native or as displayed) each score used.

## Viewing geometry

A metric with a display model (CVVDP, HDR-VDP) changes its score with the
assumed display. State the geometry on every report, and use the geometry the
dataset's study used. zensim, SSIMULACRA2 and Butteraugli use native pixels
with no display model.

## Selection effects in the dataset

If a metric chose the dataset's stimuli (placed the quality levels, or picked
the sources by disagreement), that metric's and its peers' correlations are
biased by construction. Flag the selecting metrics on every report, report
strata selected by different rules separately, and keep the level-placing
metric out of comparisons within a level.

## Many metrics

Use paired cluster-bootstrap differences with Holm adjustment
(`zenmetrics_stats::multiple::holm`). P.1401 §7.6.5 requires correcting for multiple
comparisons and names Bonferroni; Holm controls the same family-wise error
rate and is never less powerful. For an overall order, use pairwise tests
over sources (Wilcoxon signed-rank or sign tests) with Holm, and draw any
critical-difference style diagram from those. Do not use the Nemenyi
mean-rank post-hoc test behind the usual Friedman critical-difference diagram:
its outcome depends on which other metrics are in the pool (Benavoli et al.
2016). With five sources an exact two-sided Wilcoxon test cannot go below
p = 0.0625, so no pair can be declared different at 0.05 by it.

For a single descriptive order across several criteria (SROCC, PLCC after a
declared map, RMSE, PWRC, outlier ratio, …), use
`zenmetrics_stats::consensus`: each criterion ranks the metrics best-first, a
metric's consensus rank is the median of its per-criterion ranks, and ties on
the median go to the lower mean rank. The median keeps one outlying criterion
from moving a metric far. It is a summary, not a test: report the criterion
list with it, and use the paired tests above to say whether neighbours differ.

## Implementation status

| method | `zenmetrics-stats` | wired into a report |
|---|---|---|
| source / source × codec bootstrap with copy-aware draws | `resample` | not yet |
| pair classes, expected observer agreement, `PairTable` | `pairs` | not yet |
| declared maps, out-of-fold, RMSE, τ̂, rmse* | `jnd` | not yet |
| raw forced-choice agreement | `forced_choice::agreement` | zensim `panel --pairwise` (its own copy until it moves to this one) |
| forced-choice log-likelihood, observer slope, cross-fitted ceilings | `forced_choice` | not yet |
| Holm | `multiple` | not yet |
| consensus ranking across criteria (median rank, mean-rank tie-break) | `consensus` | not yet |

The evaluation CLI that applies all of this to every metric is the next step:
today zensim's `panel evaluate` (`zensim/docs/METRIC_EVALUATION_CLI.md`) is
the generic evaluator, and it calls these statistics only where noted.

## Consolidating zensim's statistics here

zensim's rules make one owner per statistic, and zenmetrics-stats is that owner for
IQA statistics. This is where the zensim implementations stand (surveyed on
zensim main `fc72d499`):

| zensim implementation | zenmetrics-stats | status |
|---|---|---|
| `zensim-validate/src/pairwise.rs` (forced-choice agreement, majority ceiling, group re-keying) | `forced_choice` | moved; zensim keeps its copy until it switches |
| `zensim-validate/src/difference_auc.rs` (DS-AUC) | `pairs::ds_auc` | moved with the same pair selection and arithmetic |
| `bake_verdict` `bootstrap_srocc_ci` (pair bootstrap of SROCC) | `resample::spearman_interval` | replacement; resample references, not pairs |
| Python reference-clustered bootstraps feeding `panel --batch` (`band_reliability.py`, `e2b_stats.py`, `rev4_e1*/analyze.py`, `paired_perref_boot.py`, `rev4_featpot/*_ci.py`) | `resample::cluster_bootstrap` | same design; recorded evidence stays as computed |
| `rev4_e1b/crosscodec.py` pair classes (same configuration / same format / different format, inside each reference) | `pairs` with `PairInput::family` | reproduced exactly (`family_classes_reproduce_the_crosscodec_pseudo_choice_construction`) |
| `v2_confirm_read.py::holm` (rejections; incomplete entries count as p = 1) | `multiple::holm_reject` | same decisions on its registered test cases |
| `zenmetrics_stats::rescale_logistic` | `jnd::MapFamily::Logistic4` | same fit, reused |

Conventions that differ, and which one zenmetrics-stats uses:

- **A cluster drawn twice.** The Python scripts concatenate a drawn
  reference's rows, so within-reference pairs of a twice-drawn reference count
  four times. For pooled SROCC over many references that is a small effect;
  for pair statistics restricted to within-reference pairs it widens the
  interval (see the PR #67 review). zenmetrics-stats keeps copies distinct
  (`Draw::pair_weight`, `Draw::expand`). `spearman_interval` computes pooled
  Spearman on the expanded rows, matching the Python method.
- **Percentiles.** Decided 2026-10-11: one rule across the crate, Hyndman–Fan
  type 7 (numpy's default, `bake_verdict`'s), so cross-checks agree exactly.
  `bootstrap_ci_delta` used order statistics (⌊0.025·n⌋, ⌈0.975·n⌉ − 1) before;
  the switch moves a bound by far less than re-running with another seed.
  Record the rule, B and seed with every result. Results computed before the
  switch are labelled with the old rule.
- **Bootstrap p-values.** zensim scripts report one-sided `#{δ > 0}/B`;
  `Interval::p_zero` is two-sided with +1/(B + 1).
- **"Not sure" answers.** Decided 2026-10-11: count them half for each side
  (`Choice::NotSure`), as the JPEG AIC scale reconstruction, Men et al. 2021
  and zensim's `konfig_fscale.py` do. zensim's earlier forced-choice reads
  dropped them. Their `acc_norm` and per-group majority agreement are
  unchanged by the split and keep their recorded values; raw agreement, the
  ceilings, tie rate, response counts and any log-likelihood are labelled
  "not-sure dropped". The SDR25 reconstruction and the MLP training loss
  model not-sure as a third ordered-probit outcome; that is a different
  model, not this convention. New reads also apply the dataset's own worker
  screening (AIC-3: batches with at least 70 % correct on their 10 trap
  questions).
- **Holm.** `holm` returns adjusted p-values and leaves NaN out of the family;
  pass 1.0 for an entry that must count but has no result.

Still in zensim, to switch once this crate's revision is pinned there:
`panel --pairwise` and `panel evaluate` to `forced_choice`; `bake_verdict`
DS-AUC and `srocc_ci` to `pairs::ds_auc` and `spearman_interval`;
`panel evaluate` comparisons to a clustered design. The training-time
ordered-probit loss (`mlp_train/strategy.rs`) and the serving calibration
splines (`dial_spline.rs`) are not evaluation statistics and stay.

## References

- Owen 2007, *The pigeonhole bootstrap*, doi:10.1214/07-AOAS122
- ITU-T P.1401 (01/2020), https://www.itu.int/rec/T-REC-P.1401-202001-I/en
- Testolina et al. 2024 (JND unit, 75 % correct), arXiv:2410.09501
- Krasula et al., QoMEX 2016, doi:10.1109/QoMEX.2016.7498936; Hanhart et al.,
  QoMEX 2016, doi:10.1109/QoMEX.2016.7498960
- Veroniki et al. 2016 (between-study variance estimators),
  doi:10.1002/jrsm.1164; DerSimonian & Laird 1986,
  doi:10.1016/0197-2456(86)90046-2
- Men et al. 2021 (boosted triplet comparisons), doi:10.1109/ACCESS.2021.3118295
- Varma & Simon 2006, doi:10.1186/1471-2105-7-91; Roberts et al. 2017,
  doi:10.1111/ecog.02881; Bates, Hastie & Tibshirani 2024,
  doi:10.1080/01621459.2023.2197686
- Holm 1979, *Scand. J. Statist.* 6(2); Demšar 2006, JMLR 7; Benavoli et al.
  2016, arXiv:1505.02288
- Zhang et al. 2018 (LPIPS; 2AFC scored against p² + (1 − p)²), arXiv:1801.03924
