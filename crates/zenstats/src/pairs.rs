//! Pairwise agreement split by what a pair compares.
//!
//! A pooled rank statistic mixes three different questions, and on most IQA
//! datasets the mix is dominated by whichever class has the most pairs:
//!
//! * [`PairClass::SameSourceSameCodec`]: two rungs of one codec's ladder on
//!   one image. Nearly every metric orders these; it tests monotonicity.
//! * [`PairClass::SameSourceCrossCodec`]: the same image through two
//!   codecs. This is the codec-comparison question.
//! * [`PairClass::CrossSource`]: different images. This asks whether the
//!   metric puts different content on one absolute scale (calibration
//!   across content). With S equally sized sources only about 1/S of all
//!   pairs share a source, so a pooled statistic on a many-source dataset is
//!   almost entirely this class.
//!
//! Two scores are reported per class:
//!
//! * [`ClassCounts`]: sign agreement between the metric and the target, with
//!   metric ties scored 0.5 and target ties (|Δ target| ≤ `target_tie`)
//!   excluded and counted.
//! * [`ExpectedAgreement`]: the share of *observers* expected to agree with
//!   the metric's ordering, under a Thurstone Case V observer model. For a
//!   target difference Δ the probability that an observer picks the side
//!   with the larger target is Φ(Δ / s), where s is either a fixed observer
//!   scale or the pair's measurement uncertainty ([`HumanNoise`]). A pair the
//!   metric orders correctly earns that probability, a reversed pair earns its
//!   complement, a metric tie earns 0.5. The oracle that always orders
//!   correctly earns the **ceiling**; `normalized = (agreement − 0.5) /
//!   (ceiling − 0.5)` is the share of achievable agreement captured. Pairs the
//!   observers cannot tell apart count near 0.5 for every metric, so they
//!   neither reward nor penalise. This needs no mapping from metric units to
//!   target units: only the metric's ordering is used. For the JPEG AIC JND
//!   scale, one JND is defined as a 75 % correct choice, so the observer scale
//!   is s = 1 / Φ⁻¹(0.75) and Φ(Δ/s) = Φ(0.6745 Δ) (Testolina et al. 2024,
//!   arXiv:2410.09501).
//!
//! Related published criteria: Krasula et al., QoMEX 2016
//! (doi:10.1109/QoMEX.2016.7498936); Hanhart et al., QoMEX 2016
//! (doi:10.1109/QoMEX.2016.7498960).
//!
//! Both walks visit every pair, O(n²). That is fine for a few thousand rows;
//! inside a bootstrap over tens of thousands of rows it is not.

use crate::panel::{Orientation, spearman};
use crate::special::norm_cdf;

/// Which question a pair asks. See the module docs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PairClass {
    /// Same source image, same codec.
    SameSourceSameCodec,
    /// Same source image, different codec.
    SameSourceCrossCodec,
    /// Different source images.
    CrossSource,
}

impl PairClass {
    /// All classes, in report order.
    pub const ALL: [PairClass; 3] = [
        PairClass::SameSourceSameCodec,
        PairClass::SameSourceCrossCodec,
        PairClass::CrossSource,
    ];

    fn of(src_a: u32, src_b: u32, codec_a: u32, codec_b: u32) -> Self {
        if src_a != src_b {
            PairClass::CrossSource
        } else if codec_a != codec_b {
            PairClass::SameSourceCrossCodec
        } else {
            PairClass::SameSourceSameCodec
        }
    }

    fn slot(self) -> usize {
        match self {
            PairClass::SameSourceSameCodec => 0,
            PairClass::SameSourceCrossCodec => 1,
            PairClass::CrossSource => 2,
        }
    }
}

/// Weighted sign-agreement counts for one pair class.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ClassCounts {
    /// Pairs whose target difference exceeds the tie threshold (weighted).
    pub pairs: f64,
    /// Pairs the metric orders like the target.
    pub concordant: f64,
    /// Pairs the metric orders opposite to the target.
    pub discordant: f64,
    /// Pairs where the metric ties (scored 0.5).
    pub metric_ties: f64,
    /// Pairs excluded because the target tied.
    pub target_ties: f64,
}

impl ClassCounts {
    /// `(concordant + 0.5 · metric_ties) / pairs`; NaN with no pairs.
    pub fn agreement(&self) -> f64 {
        if self.pairs > 0.0 {
            (self.concordant + 0.5 * self.metric_ties) / self.pairs
        } else {
            f64::NAN
        }
    }
}

/// Expected observer agreement for one pair class.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ExpectedAgreement {
    /// Pairs walked (weighted).
    pub pairs: f64,
    /// Expected share of observers agreeing with the metric's ordering.
    pub agreement: f64,
    /// The same for an oracle that orders every pair like the target.
    pub ceiling: f64,
    /// `(agreement − 0.5) / (ceiling − 0.5)`; NaN when the ceiling is 0.5.
    pub normalized: f64,
}

/// How observers' choices scatter around the target difference.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum HumanNoise<'a> {
    /// A fixed observer scale `s` in target units: P = Φ(Δ / s). For a JND
    /// target defined at 75 % correct, use [`HumanNoise::JND_75`].
    Observer { scale: f64 },
    /// Per-stimulus measurement standard deviation of the target:
    /// P = Φ(Δ / √(σᵢ² + σⱼ²)). This version drifts toward plain sign
    /// agreement as a study adds observers (σ shrinks), so it describes the
    /// study more than the metric; prefer `Observer` when the unit defines one.
    Measurement { sigma: &'a [f64] },
}

impl HumanNoise<'_> {
    /// Observer scale for a JND unit defined as 75 % correct choice:
    /// s = 1 / Φ⁻¹(0.75).
    pub const JND_75: HumanNoise<'static> = HumanNoise::Observer {
        scale: 1.0 / 0.674_489_750_196_081_7,
    };
}

/// Per-class results plus the pooled total.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PerClass<T> {
    /// [`PairClass::SameSourceSameCodec`].
    pub same_source_same_codec: T,
    /// [`PairClass::SameSourceCrossCodec`].
    pub same_source_cross_codec: T,
    /// [`PairClass::CrossSource`].
    pub cross_source: T,
    /// Every pair.
    pub pooled: T,
}

impl<T: Copy> PerClass<T> {
    /// The value for one class.
    pub fn get(&self, class: PairClass) -> T {
        match class {
            PairClass::SameSourceSameCodec => self.same_source_same_codec,
            PairClass::SameSourceCrossCodec => self.same_source_cross_codec,
            PairClass::CrossSource => self.cross_source,
        }
    }
}

/// Why a pair statistic could not run.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PairError {
    /// Input slices differ in length; the payload names the first offender.
    Length(&'static str),
    /// A score, target, weight or σ is not finite (or a weight/σ is negative).
    NonFinite(&'static str),
    /// A noise scale is not positive.
    BadScale,
}

impl core::fmt::Display for PairError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Length(what) => write!(f, "`{what}` has a different length from `predicted`"),
            Self::NonFinite(what) => write!(f, "`{what}` holds a non-finite or negative value"),
            Self::BadScale => write!(f, "observer scale must be positive and finite"),
        }
    }
}

impl std::error::Error for PairError {}

/// Rows for a pair walk: scores, targets, keys and optional row weights.
#[derive(Clone, Copy, Debug)]
pub struct PairInput<'a> {
    /// Metric scores.
    pub predicted: &'a [f64],
    /// Target values (human scores or reconstructed JND).
    pub target: &'a [f64],
    /// Source-image label per row.
    pub source: &'a [u32],
    /// Codec label per row. For a codec nested in its source, any labelling
    /// works: only equality within a source matters here.
    pub codec: &'a [u32],
    /// Optional non-negative row weights (for example from
    /// [`crate::resample::cluster_bootstrap`]); a pair weighs `wᵢ·wⱼ`.
    pub weights: Option<&'a [f64]>,
    /// Whether the metric rises or falls with the target. `Auto` resolves the
    /// sign once from the pooled Spearman correlation, never per class.
    pub orientation: Orientation,
}

impl PairInput<'_> {
    fn validate(&self) -> Result<f64, PairError> {
        let n = self.predicted.len();
        for (name, len) in [
            ("target", self.target.len()),
            ("source", self.source.len()),
            ("codec", self.codec.len()),
        ] {
            if len != n {
                return Err(PairError::Length(name));
            }
        }
        if !self.predicted.iter().all(|x| x.is_finite()) {
            return Err(PairError::NonFinite("predicted"));
        }
        if !self.target.iter().all(|x| x.is_finite()) {
            return Err(PairError::NonFinite("target"));
        }
        if let Some(w) = self.weights {
            if w.len() != n {
                return Err(PairError::Length("weights"));
            }
            if !w.iter().all(|x| x.is_finite() && *x >= 0.0) {
                return Err(PairError::NonFinite("weights"));
            }
        }
        Ok(match self.orientation {
            Orientation::HigherIsBetter => 1.0,
            Orientation::LowerIsBetter => -1.0,
            Orientation::Auto => {
                if spearman(self.predicted, self.target) < 0.0 {
                    -1.0
                } else {
                    1.0
                }
            }
        })
    }

    fn weight(&self, i: usize) -> f64 {
        self.weights.map_or(1.0, |w| w[i])
    }
}

/// Sign agreement per pair class. Target differences with magnitude at most
/// `target_tie` are excluded (and counted in `target_ties`).
pub fn class_agreement(
    input: &PairInput<'_>,
    target_tie: f64,
) -> Result<PerClass<ClassCounts>, PairError> {
    let sign = input.validate()?;
    let n = input.predicted.len();
    let mut acc = [ClassCounts::default(); 3];
    for i in 0..n {
        let wi = input.weight(i);
        if wi == 0.0 {
            continue;
        }
        for j in (i + 1)..n {
            let w = wi * input.weight(j);
            if w == 0.0 {
                continue;
            }
            let c = &mut acc[PairClass::of(
                input.source[i],
                input.source[j],
                input.codec[i],
                input.codec[j],
            )
            .slot()];
            let dt = input.target[i] - input.target[j];
            if dt.abs() <= target_tie {
                c.target_ties += w;
                continue;
            }
            c.pairs += w;
            let dp = sign * (input.predicted[i] - input.predicted[j]);
            if dp == 0.0 {
                c.metric_ties += w;
            } else if (dp > 0.0) == (dt > 0.0) {
                c.concordant += w;
            } else {
                c.discordant += w;
            }
        }
    }
    let mut pooled = ClassCounts::default();
    for c in &acc {
        pooled.pairs += c.pairs;
        pooled.concordant += c.concordant;
        pooled.discordant += c.discordant;
        pooled.metric_ties += c.metric_ties;
        pooled.target_ties += c.target_ties;
    }
    Ok(PerClass {
        same_source_same_codec: acc[0],
        same_source_cross_codec: acc[1],
        cross_source: acc[2],
        pooled,
    })
}

/// Expected observer agreement per pair class. See the module docs.
pub fn expected_agreement(
    input: &PairInput<'_>,
    noise: HumanNoise<'_>,
) -> Result<PerClass<ExpectedAgreement>, PairError> {
    let sign = input.validate()?;
    let n = input.predicted.len();
    match noise {
        HumanNoise::Observer { scale } => {
            if !(scale.is_finite() && scale > 0.0) {
                return Err(PairError::BadScale);
            }
        }
        HumanNoise::Measurement { sigma } => {
            if sigma.len() != n {
                return Err(PairError::Length("sigma"));
            }
            if !sigma.iter().all(|s| s.is_finite() && *s >= 0.0) {
                return Err(PairError::NonFinite("sigma"));
            }
        }
    }
    // per class: (weight, metric agreement sum, oracle sum)
    let mut acc = [[0.0f64; 3]; 3];
    for i in 0..n {
        let wi = input.weight(i);
        if wi == 0.0 {
            continue;
        }
        for j in (i + 1)..n {
            let w = wi * input.weight(j);
            if w == 0.0 {
                continue;
            }
            let dt = input.target[i] - input.target[j];
            let z = match noise {
                HumanNoise::Observer { scale } => dt.abs() / scale,
                HumanNoise::Measurement { sigma } => {
                    let s = (sigma[i] * sigma[i] + sigma[j] * sigma[j]).sqrt();
                    if s > 0.0 {
                        dt.abs() / s
                    } else if dt == 0.0 {
                        0.0
                    } else {
                        f64::INFINITY
                    }
                }
            };
            // probability an observer orders the pair like the target, and its complement
            let p_right = norm_cdf(z);
            let p_wrong = norm_cdf(-z);
            let dp = sign * (input.predicted[i] - input.predicted[j]);
            let metric = if dp == 0.0 || dt == 0.0 {
                0.5
            } else if (dp > 0.0) == (dt > 0.0) {
                p_right
            } else {
                p_wrong
            };
            let a = &mut acc[PairClass::of(
                input.source[i],
                input.source[j],
                input.codec[i],
                input.codec[j],
            )
            .slot()];
            a[0] += w;
            a[1] += w * metric;
            a[2] += w * p_right;
        }
    }
    let finish = |a: [f64; 3]| -> ExpectedAgreement {
        if a[0] <= 0.0 {
            return ExpectedAgreement {
                pairs: 0.0,
                agreement: f64::NAN,
                ceiling: f64::NAN,
                normalized: f64::NAN,
            };
        }
        let agreement = a[1] / a[0];
        let ceiling = a[2] / a[0];
        ExpectedAgreement {
            pairs: a[0],
            agreement,
            ceiling,
            normalized: if ceiling > 0.5 {
                (agreement - 0.5) / (ceiling - 0.5)
            } else {
                f64::NAN
            },
        }
    };
    let pooled = [
        acc[0][0] + acc[1][0] + acc[2][0],
        acc[0][1] + acc[1][1] + acc[2][1],
        acc[0][2] + acc[1][2] + acc[2][2],
    ];
    Ok(PerClass {
        same_source_same_codec: finish(acc[0]),
        same_source_cross_codec: finish(acc[1]),
        cross_source: finish(acc[2]),
        pooled: finish(pooled),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input<'a>(p: &'a [f64], t: &'a [f64], s: &'a [u32], c: &'a [u32]) -> PairInput<'a> {
        PairInput {
            predicted: p,
            target: t,
            source: s,
            codec: c,
            weights: None,
            orientation: Orientation::HigherIsBetter,
        }
    }

    /// Two sources × two codecs × two rungs. The metric orders every ladder
    /// and every codec pair correctly but puts source 1 on the wrong absolute
    /// level, so only cross-source pairs disagree.
    fn calibration_failure() -> (Vec<f64>, Vec<f64>, Vec<u32>, Vec<u32>) {
        let mut p = Vec::new();
        let mut t = Vec::new();
        let mut s = Vec::new();
        let mut c = Vec::new();
        for src in 0..2u32 {
            for codec in 0..2u32 {
                for rung in 0..2u32 {
                    let truth = 10.0 * src as f64 + 2.0 * codec as f64 + rung as f64;
                    let shift = if src == 1 { -100.0 } else { 0.0 };
                    p.push(truth + shift);
                    t.push(truth);
                    s.push(src);
                    c.push(codec);
                }
            }
        }
        (p, t, s, c)
    }

    #[test]
    fn calibration_failure_shows_only_in_cross_source_pairs() {
        let (p, t, s, c) = calibration_failure();
        let r = class_agreement(&input(&p, &t, &s, &c), 0.0).unwrap();
        assert_eq!(r.same_source_same_codec.agreement(), 1.0);
        assert_eq!(r.same_source_cross_codec.agreement(), 1.0);
        assert_eq!(r.cross_source.agreement(), 0.0);
        assert_eq!(r.same_source_same_codec.pairs, 4.0);
        assert_eq!(r.same_source_cross_codec.pairs, 8.0);
        assert_eq!(r.cross_source.pairs, 16.0);
        assert_eq!(r.pooled.pairs, 28.0);
    }

    #[test]
    fn orientation_flips_and_auto_resolves_once() {
        let (p, t, s, c) = calibration_failure();
        let neg: Vec<f64> = p.iter().map(|x| -x).collect();
        let mut inp = input(&neg, &t, &s, &c);
        inp.orientation = Orientation::LowerIsBetter;
        let r = class_agreement(&inp, 0.0).unwrap();
        assert_eq!(r.same_source_same_codec.agreement(), 1.0);
        assert_eq!(r.cross_source.agreement(), 0.0);
    }

    #[test]
    fn weights_match_row_duplication() {
        let p = [1.0, 2.0, 3.0, 0.5];
        let t = [1.0, 3.0, 2.0, 0.0];
        let s = [0u32, 0, 1, 1];
        let c = [0u32, 1, 0, 0];
        let w = [2.0, 1.0, 1.0, 3.0];
        let mut inp = input(&p, &t, &s, &c);
        inp.weights = Some(&w);
        let weighted = class_agreement(&inp, 0.0).unwrap();
        let (mut dp, mut dt, mut ds, mut dc) = (vec![], vec![], vec![], vec![]);
        for i in 0..4 {
            for _ in 0..(w[i] as usize) {
                dp.push(p[i]);
                dt.push(t[i]);
                ds.push(s[i]);
                dc.push(c[i]);
            }
        }
        let dup = class_agreement(&input(&dp, &dt, &ds, &dc), 0.0).unwrap();
        // duplicated copies of one row form target-tied pairs; everything else matches
        for class in PairClass::ALL {
            let (a, b) = (weighted.get(class), dup.get(class));
            assert_eq!(a.concordant, b.concordant, "{class:?}");
            assert_eq!(a.discordant, b.discordant, "{class:?}");
            assert_eq!(a.pairs, b.pairs, "{class:?}");
        }
    }

    #[test]
    fn expected_agreement_rewards_magnitude_and_ignores_indistinguishable_pairs() {
        // Pair (0,1): 2 JND apart, metric right. Pair (0,2): 0.01 JND, metric wrong.
        let p = [0.0, 1.0, -1.0];
        let t = [0.0, 2.0, 0.01];
        let s = [0u32, 0, 0];
        let c = [0u32, 0, 0];
        let r = expected_agreement(&input(&p, &t, &s, &c), HumanNoise::JND_75).unwrap();
        let pooled = r.pooled;
        let p2 = norm_cdf(0.6744897501960817 * 2.0);
        let p001 = norm_cdf(0.6744897501960817 * 0.01);
        let p199 = norm_cdf(0.6744897501960817 * 1.99);
        // pairs: (0,1) right → p2; (0,2) wrong → 1−p001; (1,2) metric 1 > −1 and target 2 > 0.01 → right → p199
        let want = (p2 + (1.0 - p001) + p199) / 3.0;
        assert!((pooled.agreement - want).abs() < 1e-12);
        assert!((pooled.ceiling - (p2 + p001 + p199) / 3.0).abs() < 1e-12);
        // 1 JND is exactly 75 % by definition of the unit
        assert!((norm_cdf(1.0 / (1.0 / 0.6744897501960817)) - 0.75).abs() < 1e-7);
    }

    #[test]
    fn oracle_normalizes_to_one_and_reversal_to_minus_one() {
        let t = [0.0, 0.5, 1.2, 3.0];
        let s = [0u32; 4];
        let c = [0u32; 4];
        let r = expected_agreement(&input(&t, &t, &s, &c), HumanNoise::JND_75).unwrap();
        assert!((r.pooled.normalized - 1.0).abs() < 1e-12);
        let rev: Vec<f64> = t.iter().map(|x| -x).collect();
        let r = expected_agreement(&input(&rev, &t, &s, &c), HumanNoise::JND_75).unwrap();
        assert!((r.pooled.normalized + 1.0).abs() < 1e-12);
    }

    #[test]
    fn refuses_bad_input() {
        let p = [1.0, f64::NAN];
        let t = [1.0, 2.0];
        let k = [0u32, 0];
        assert_eq!(
            class_agreement(&input(&p, &t, &k, &k), 0.0),
            Err(PairError::NonFinite("predicted"))
        );
        let p = [1.0, 2.0];
        assert_eq!(
            class_agreement(&input(&p, &t, &k[..1], &k), 0.0),
            Err(PairError::Length("source"))
        );
        assert_eq!(
            expected_agreement(&input(&p, &t, &k, &k), HumanNoise::Observer { scale: 0.0 }),
            Err(PairError::BadScale)
        );
    }
}
