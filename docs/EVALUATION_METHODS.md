# Evaluating a metric against human data

How to evaluate any metric in this repository against a human-labelled image
quality dataset, using one implementation of every statistic (`zenstats`).
The same procedure applies to zensim, SSIMULACRA2, Butteraugli, CVVDP and the
rest; nothing here is specific to one metric.

## What to resample: sources and codecs, not rows

An IQA dataset samples source images and codecs. Every distorted image of one
source shares its content, and every image from one codec shares its
artefacts. A row bootstrap treats them as independent and gives intervals
that are too narrow for any claim about new images or codecs.

- Use `zenstats::resample::cluster_bootstrap` with `Design::OneWay(source)`,
  or `Design::TwoWay(source, codec)` (Owen's pigeonhole bootstrap) when the
  codecs are crossed with the sources.
- Give a codec applied to only some sources a label per (source, codec), so it
  is nested in its source rather than crossed.
- Put the interval on every criterion and on every paired difference between
  two metrics. Compute both metrics inside one statistic closure so they see
  the same resample.
- With five or so sources the interval is descriptive; say so.
- Print the row-bootstrap interval beside it only as a labelled comparison.

## Split pairwise agreement into three classes

`zenstats::pairs` splits every pair of stimuli by what it compares:

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

1. **Declare the family before looking at the data** (`zenstats::jnd::MapFamily`):
   `Linear`, `Logistic4`, `KneePower` (a·max(0, b − x)^c, the form used to put
   metrics on a JND scale) or `Isotonic` (the best monotone shape).
2. **Fit leave-one-source-out** (`jnd::out_of_fold`), so no stimulus is
   predicted by a map that saw its source. A metric that outputs the target
   unit natively is also scored as delivered, so the mapping's share is
   visible.
3. **Compute statistics on the pooled out-of-fold predictions**, with the
   cluster interval. Per-fold intervals under-cover.
4. **Headline: RMSE in target units and τ̂** (`jnd::excess_error_sd`), the
   error standard deviation left after removing the target's measurement
   noise. τ̂ does not grow as a study adds observers, and it does not let the
   few stimuli with the smallest σ dominate.
5. **Diagnostics, not rankings:** the normalised error `(pred − target)/σ`
   (`z_rmse_per_sample`) is weighted by 1/σ²; on data whose σ shrinks toward
   zero distortion it mostly scores the near-threshold stimuli and changes
   rank with the mapping family. Report it as a goodness-of-fit check, beside
   the outlier ratio and P.1401 rmse* (`jnd::rmse_star`).
6. **Charge the map:** report the declared family, a first-order map and an
   out-of-fold isotonic map side by side.
7. **Transfer:** for a map fitted on one dataset and applied to another, report
   zero-shot error, and separately the error after a re-anchor of at most two
   parameters (P.1401 §7.3.3 allows offsets that do not change the rank
   order). Zero-shot measures absolute calibration; re-anchored measures shape.

## Raw forced choices

When a dataset releases its raw 2AFC or triplet responses, score them
directly (`zenstats::forced_choice`):

- `agreement`: response-weighted agreement and the majority-oracle ceiling.
  Always report the ceiling; the same accuracy means opposite things against a
  0.75 and a 0.95 ceiling.
- `log_likelihood`: once a metric's map is frozen, the held-out
  log-likelihood of the choices under P = Φ(ΔJND / s), normalised between
  chance and the per-group ceiling. It is a proper scoring rule, keeps
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
(`zenstats::multiple::holm`, P.1401 §7.6.5). A Friedman test over sources with
a critical-difference diagram can summarise an overall order. Avoid
mean-rank post-hoc tests, whose outcome depends on which other metrics are in
the pool.

## Implementation status

| method | `zenstats` | wired into a report |
|---|---|---|
| source / source × codec bootstrap | `resample` | not yet |
| pair classes, expected observer agreement | `pairs` | not yet |
| declared maps, out-of-fold, RMSE, τ̂, rmse* | `jnd` | not yet |
| raw forced-choice agreement | `forced_choice::agreement` | zensim `panel --pairwise` (its own copy until it moves to this one) |
| forced-choice log-likelihood | `forced_choice::log_likelihood` | not yet |
| Holm | `multiple` | not yet |

The evaluation CLI that applies all of this to every metric is the next step:
today zensim's `panel evaluate` (`zensim/docs/METRIC_EVALUATION_CLI.md`) is
the generic evaluator, and it calls these statistics only where noted.

## References

- Owen 2007, *The pigeonhole bootstrap*, doi:10.1214/07-AOAS122
- ITU-T P.1401 (01/2020), https://www.itu.int/rec/T-REC-P.1401-202001-I/en
- Testolina et al. 2024 (JND unit, 75 % correct), arXiv:2410.09501
- Krasula et al., QoMEX 2016, doi:10.1109/QoMEX.2016.7498936; Hanhart et al.,
  QoMEX 2016, doi:10.1109/QoMEX.2016.7498960
- DerSimonian & Laird 1986, doi:10.1016/0197-2456(86)90046-2
- Varma & Simon 2006, doi:10.1186/1471-2105-7-91; Roberts et al. 2017,
  doi:10.1111/ecog.02881; Bates, Hastie & Tibshirani 2024,
  doi:10.1080/01621459.2023.2197686
- Holm 1979, *Scand. J. Statist.* 6(2); Demšar 2006, JMLR 7; Benavoli et al.
  2016, arXiv:1505.02288
- Zhang et al. 2018 (2AFC with a human ceiling), arXiv:1801.03924
