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
//! Both functions visit every pair, O(n²). Inside a bootstrap, build a
//! [`PairTable`] once instead: it sums pairs per pair of (source, codec) cells,
//! so each draw costs O(K²) for K cells and gives the same result.
//!
//! Under a cluster bootstrap, pass the [`Draw`] (not row weights): a source
//! drawn twice is two sources, so its within-source pairs count twice, not
//! four times ([`Draw::pair_weight`]).

use crate::panel::{Orientation, spearman};
use crate::resample::{Design, Draw};
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
    /// A noise scale is not positive (or a table was built without noise).
    BadScale,
    /// A bootstrap draw or design does not match the source/codec labels.
    DesignMismatch,
}

impl core::fmt::Display for PairError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Length(what) => write!(f, "`{what}` has a different length from `predicted`"),
            Self::NonFinite(what) => write!(f, "`{what}` holds a non-finite or negative value"),
            Self::BadScale => write!(f, "observer scale must be positive and finite"),
            Self::DesignMismatch => {
                write!(
                    f,
                    "the bootstrap design does not match the source/codec labels"
                )
            }
        }
    }
}

impl std::error::Error for PairError {}

/// Rows for a pair walk: scores, targets, keys and an optional bootstrap draw.
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
    /// Optional bootstrap resample. Pairs are weighted by
    /// [`Draw::pair_weight`], which keeps copies of a drawn source distinct.
    /// The draw's first factor must partition rows exactly like `source`
    /// (`Design::OneWay(source)` or `Design::TwoWay(source, …)`), and a second
    /// factor must agree with `codec` inside every source; `Design::Rows`
    /// draws are accepted as they are.
    pub draw: Option<&'a Draw<'a>>,
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
        if let Some(d) = self.draw {
            if d.n_rows() != n {
                return Err(PairError::Length("draw"));
            }
            check_design(self.source, self.codec, d)?;
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
}

/// The draw's factors must match the pair classes: factor 0 is the source
/// partition, factor 1 (if any) agrees with the codec labels inside a source.
fn check_design(source: &[u32], codec: &[u32], d: &Draw<'_>) -> Result<(), PairError> {
    use std::collections::HashMap;
    if d.is_row_design() {
        return Ok(());
    }
    let mut s2a: HashMap<u32, usize> = HashMap::new();
    let mut a2s: HashMap<usize, u32> = HashMap::new();
    for (i, &s) in source.iter().enumerate() {
        let a = d.level(0, i);
        if *s2a.entry(s).or_insert(a) != a || *a2s.entry(a).or_insert(s) != s {
            return Err(PairError::DesignMismatch);
        }
    }
    if d.n_factors() > 1 {
        let mut sc2b: HashMap<(u32, u32), usize> = HashMap::new();
        let mut sb2c: HashMap<(u32, usize), u32> = HashMap::new();
        for i in 0..source.len() {
            let b = d.level(1, i);
            if *sc2b.entry((source[i], codec[i])).or_insert(b) != b
                || *sb2c.entry((source[i], b)).or_insert(codec[i]) != codec[i]
            {
                return Err(PairError::DesignMismatch);
            }
        }
    }
    Ok(())
}

fn check_noise(noise: &HumanNoise<'_>, n: usize) -> Result<(), PairError> {
    match noise {
        HumanNoise::Observer { scale } => {
            if !(scale.is_finite() && *scale > 0.0) {
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
    Ok(())
}

/// Unweighted contributions of pairs, per class.
#[derive(Clone, Copy, Debug, Default)]
struct Acc {
    counts: [ClassCounts; 3],
    /// per class: (pairs, metric agreement sum, oracle sum)
    expected: [[f64; 3]; 3],
}

impl Acc {
    fn add_scaled(&mut self, other: &Acc, w: f64) {
        for k in 0..3 {
            let (a, b) = (&mut self.counts[k], &other.counts[k]);
            a.pairs += w * b.pairs;
            a.concordant += w * b.concordant;
            a.discordant += w * b.discordant;
            a.metric_ties += w * b.metric_ties;
            a.target_ties += w * b.target_ties;
            for f in 0..3 {
                self.expected[k][f] += w * other.expected[k][f];
            }
        }
    }

    fn counts(&self) -> PerClass<ClassCounts> {
        let mut pooled = ClassCounts::default();
        for c in &self.counts {
            pooled.pairs += c.pairs;
            pooled.concordant += c.concordant;
            pooled.discordant += c.discordant;
            pooled.metric_ties += c.metric_ties;
            pooled.target_ties += c.target_ties;
        }
        PerClass {
            same_source_same_codec: self.counts[0],
            same_source_cross_codec: self.counts[1],
            cross_source: self.counts[2],
            pooled,
        }
    }

    fn expected(&self) -> PerClass<ExpectedAgreement> {
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
        let e = &self.expected;
        let pooled = [
            e[0][0] + e[1][0] + e[2][0],
            e[0][1] + e[1][1] + e[2][1],
            e[0][2] + e[1][2] + e[2][2],
        ];
        PerClass {
            same_source_same_codec: finish(e[0]),
            same_source_cross_codec: finish(e[1]),
            cross_source: finish(e[2]),
            pooled: finish(pooled),
        }
    }
}

/// Add the pair (i, j) with weight `w` to `acc`. `noise` None skips the
/// expected-agreement sums.
#[allow(clippy::too_many_arguments)]
fn add_pair(
    acc: &mut Acc,
    input: &PairInput<'_>,
    sign: f64,
    target_tie: f64,
    noise: Option<&HumanNoise<'_>>,
    i: usize,
    j: usize,
    w: f64,
) {
    let slot = PairClass::of(
        input.source[i],
        input.source[j],
        input.codec[i],
        input.codec[j],
    )
    .slot();
    let dt = input.target[i] - input.target[j];
    let dp = sign * (input.predicted[i] - input.predicted[j]);
    let c = &mut acc.counts[slot];
    if dt.abs() <= target_tie {
        c.target_ties += w;
    } else {
        c.pairs += w;
        if dp == 0.0 {
            c.metric_ties += w;
        } else if (dp > 0.0) == (dt > 0.0) {
            c.concordant += w;
        } else {
            c.discordant += w;
        }
    }
    if let Some(noise) = noise {
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
        let metric = if dp == 0.0 || dt == 0.0 {
            0.5
        } else if (dp > 0.0) == (dt > 0.0) {
            p_right
        } else {
            p_wrong
        };
        let a = &mut acc.expected[slot];
        a[0] += w;
        a[1] += w * metric;
        a[2] += w * p_right;
    }
}

fn walk(input: &PairInput<'_>, sign: f64, target_tie: f64, noise: Option<&HumanNoise<'_>>) -> Acc {
    let n = input.predicted.len();
    let mut acc = Acc::default();
    for i in 0..n {
        for j in (i + 1)..n {
            let w = input.draw.map_or(1.0, |d| d.pair_weight(i, j));
            if w != 0.0 {
                add_pair(&mut acc, input, sign, target_tie, noise, i, j, w);
            }
        }
    }
    acc
}

/// Sign agreement per pair class. Target differences with magnitude at most
/// `target_tie` are excluded (and counted in `target_ties`).
pub fn class_agreement(
    input: &PairInput<'_>,
    target_tie: f64,
) -> Result<PerClass<ClassCounts>, PairError> {
    let sign = input.validate()?;
    Ok(walk(input, sign, target_tie, None).counts())
}

/// Expected observer agreement per pair class. See the module docs.
pub fn expected_agreement(
    input: &PairInput<'_>,
    noise: HumanNoise<'_>,
) -> Result<PerClass<ExpectedAgreement>, PairError> {
    let sign = input.validate()?;
    check_noise(&noise, input.predicted.len())?;
    Ok(walk(input, sign, 0.0, Some(&noise)).expected())
}

/// Pair statistics precomputed per pair of cells, for fast bootstrapping.
///
/// A cell is one (source, codec) combination. Every pair of rows is visited
/// once at construction (O(n²)); each bootstrap draw then costs O(K²) for K
/// cells, because a draw's pair weight depends only on the two rows' design
/// levels, which are constant within a cell. Results equal
/// [`class_agreement`] and [`expected_agreement`] with the same draw.
#[derive(Clone, Debug)]
pub struct PairTable {
    n_rows: usize,
    /// Design levels per cell, per factor.
    cell_levels: Vec<Vec<usize>>,
    /// Upper-triangular (p ≤ q) cell-pair accumulators, row-major.
    acc: Vec<Acc>,
    n_cells: usize,
    has_noise: bool,
}

impl PairTable {
    /// Precompute over `input` (its `draw` must be `None`) for draws from
    /// `design`, which must be `Design::OneWay(source)` or
    /// `Design::TwoWay(source, codec-or-nested-codec)`. `noise` enables
    /// [`PairTable::expected_agreement`].
    pub fn new(
        input: &PairInput<'_>,
        design: &Design<'_>,
        target_tie: f64,
        noise: Option<HumanNoise<'_>>,
    ) -> Result<Self, PairError> {
        if input.draw.is_some() {
            return Err(PairError::DesignMismatch);
        }
        let sign = input.validate()?;
        let n = input.predicted.len();
        if let Some(nz) = &noise {
            check_noise(nz, n)?;
        }
        if matches!(design, Design::Rows) {
            return Err(PairError::DesignMismatch);
        }
        // a unit draw gives access to the design's dense levels
        let dense =
            crate::resample::DesignLevels::new(n, design).map_err(|_| PairError::DesignMismatch)?;
        let draw = dense.unit_draw();
        check_design(input.source, input.codec, &draw)?;
        let nf = draw.n_factors();
        // cells keyed by (source, codec)
        let mut cell_of = std::collections::HashMap::new();
        let mut cells = Vec::with_capacity(n);
        let mut cell_levels: Vec<Vec<usize>> = Vec::new();
        for i in 0..n {
            let next = cell_levels.len();
            let c = *cell_of
                .entry((input.source[i], input.codec[i]))
                .or_insert(next);
            if c == next {
                cell_levels.push((0..nf).map(|f| draw.level(f, i)).collect());
            } else if (0..nf).any(|f| cell_levels[c][f] != draw.level(f, i)) {
                return Err(PairError::DesignMismatch);
            }
            cells.push(c);
        }
        let k = cell_levels.len();
        let mut acc = vec![Acc::default(); k * k];
        for i in 0..n {
            for j in (i + 1)..n {
                let (p, q) = (cells[i].min(cells[j]), cells[i].max(cells[j]));
                add_pair(
                    &mut acc[p * k + q],
                    input,
                    sign,
                    target_tie,
                    noise.as_ref(),
                    i,
                    j,
                    1.0,
                );
            }
        }
        Ok(Self {
            n_rows: n,
            cell_levels,
            acc,
            n_cells: k,
            has_noise: noise.is_some(),
        })
    }

    fn total(&self, draw: &Draw<'_>) -> Result<Acc, PairError> {
        if draw.n_rows() != self.n_rows
            || draw.is_row_design()
            || draw.n_factors() != self.cell_levels.first().map_or(0, Vec::len)
        {
            return Err(PairError::DesignMismatch);
        }
        let k = self.n_cells;
        let mut out = Acc::default();
        for p in 0..k {
            for q in p..k {
                let a = &self.acc[p * k + q];
                let lv = |c: usize| -> [usize; 2] {
                    let l = &self.cell_levels[c];
                    [l[0], l.get(1).copied().unwrap_or(0)]
                };
                let w = crate::resample::hierarchical_pair_weight(
                    false,
                    draw.n_factors(),
                    lv(p),
                    lv(q),
                    |f, l| draw.multiplicity(f, l),
                );
                if w != 0.0 {
                    out.add_scaled(a, w);
                }
            }
        }
        Ok(out)
    }

    /// [`class_agreement`] under `draw`.
    pub fn class_agreement(&self, draw: &Draw<'_>) -> Result<PerClass<ClassCounts>, PairError> {
        Ok(self.total(draw)?.counts())
    }

    /// [`expected_agreement`] under `draw`; requires `noise` at construction.
    pub fn expected_agreement(
        &self,
        draw: &Draw<'_>,
    ) -> Result<PerClass<ExpectedAgreement>, PairError> {
        if !self.has_noise {
            return Err(PairError::BadScale);
        }
        Ok(self.total(draw)?.expected())
    }
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
            draw: None,
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

    /// A small crossed dataset: 4 sources × 3 codecs × 3 rungs, with a
    /// metric that is mostly right, sometimes reversed, sometimes tied.
    fn crossed() -> (Vec<f64>, Vec<f64>, Vec<u32>, Vec<u32>) {
        let (mut p, mut t, mut s, mut c) = (vec![], vec![], vec![], vec![]);
        for src in 0..4u32 {
            for codec in 0..3u32 {
                for rung in 0..3u32 {
                    let truth = 0.4 * rung as f64 + 0.15 * codec as f64 + 0.3 * src as f64;
                    let noise = (((src * 7 + codec * 3 + rung * 5) % 5) as f64 - 2.0) * 0.12;
                    p.push(((truth + noise) * 10.0).round() / 10.0);
                    t.push(truth);
                    s.push(src);
                    c.push(codec);
                }
            }
        }
        (p, t, s, c)
    }

    /// Independent reference: materialise the resample with explicit copy
    /// identities and count pairs the plain way. Pairs between different
    /// copies of one source, or (two-way) different copies of one codec inside
    /// one source copy, are dropped; every other pair counts once.
    fn materialised(inp: &PairInput<'_>, d: &Draw<'_>, two_way: bool) -> PerClass<ClassCounts> {
        let n = inp.predicted.len();
        // (row, source copy, codec copy)
        let mut rows = Vec::new();
        for i in 0..n {
            let ms = d.multiplicity(0, d.level(0, i)) as usize;
            let mc = if two_way {
                d.multiplicity(1, d.level(1, i)) as usize
            } else {
                1
            };
            for k in 0..ms {
                for l in 0..mc {
                    rows.push((i, k, l));
                }
            }
        }
        let mut acc = [ClassCounts::default(); 3];
        for x in 0..rows.len() {
            for y in (x + 1)..rows.len() {
                let ((i, ki, li), (j, kj, lj)) = (rows[x], rows[y]);
                let same_src = inp.source[i] == inp.source[j];
                if same_src && ki != kj {
                    continue;
                }
                if two_way && same_src && inp.codec[i] == inp.codec[j] && li != lj {
                    continue;
                }
                let cls = PairClass::of(inp.source[i], inp.source[j], inp.codec[i], inp.codec[j]);
                let c = &mut acc[cls.slot()];
                let dt = inp.target[i] - inp.target[j];
                if dt == 0.0 {
                    c.target_ties += 1.0;
                    continue;
                }
                c.pairs += 1.0;
                let dp = inp.predicted[i] - inp.predicted[j];
                if dp == 0.0 {
                    c.metric_ties += 1.0;
                } else if (dp > 0.0) == (dt > 0.0) {
                    c.concordant += 1.0;
                } else {
                    c.discordant += 1.0;
                }
            }
        }
        PerClass {
            same_source_same_codec: acc[0],
            same_source_cross_codec: acc[1],
            cross_source: acc[2],
            pooled: ClassCounts::default(),
        }
    }

    #[test]
    fn draw_weights_match_a_materialised_resample_with_copy_identity() {
        use crate::resample::{BootstrapConfig, cluster_bootstrap};
        use std::sync::Mutex;
        let (p, t, s, c) = crossed();
        let failures = Mutex::new(Vec::new());
        for two_way in [false, true] {
            let design = if two_way {
                Design::TwoWay(&s, &c)
            } else {
                Design::OneWay(&s)
            };
            let cfg = BootstrapConfig {
                resamples: 60,
                seed: 11,
                level: 0.95,
            };
            let table = PairTable::new(
                &input(&p, &t, &s, &c),
                &design,
                0.0,
                Some(HumanNoise::JND_75),
            )
            .unwrap();
            cluster_bootstrap(p.len(), &design, &cfg, |d| {
                let mut inp = input(&p, &t, &s, &c);
                inp.draw = Some(d);
                let direct = class_agreement(&inp, 0.0).unwrap();
                let want = materialised(&inp, d, two_way);
                let fast = table.class_agreement(d).unwrap();
                let direct_e = expected_agreement(&inp, HumanNoise::JND_75).unwrap();
                let fast_e = table.expected_agreement(d).unwrap();
                for class in PairClass::ALL {
                    let (a, b, f) = (direct.get(class), want.get(class), fast.get(class));
                    let same = |x: f64, y: f64| (x - y).abs() <= 1e-9 * (1.0 + x.abs());
                    let ok = [
                        (a.pairs, b.pairs),
                        (a.concordant, b.concordant),
                        (a.discordant, b.discordant),
                        (a.metric_ties, b.metric_ties),
                        (a.target_ties, b.target_ties),
                        (a.pairs, f.pairs),
                        (a.concordant, f.concordant),
                        (a.discordant, f.discordant),
                    ]
                    .iter()
                    .all(|&(x, y)| same(x, y));
                    let (de, fe) = (direct_e.get(class), fast_e.get(class));
                    let ok_e = (de.pairs - fe.pairs).abs() < 1e-9
                        && (de.agreement.is_nan() && fe.agreement.is_nan()
                            || (de.agreement - fe.agreement).abs() < 1e-12);
                    if !ok || !ok_e {
                        failures
                            .lock()
                            .unwrap()
                            .push(format!("two_way={two_way} {class:?}: {a:?} {b:?} {f:?}"));
                    }
                }
                Vec::new()
            })
            .unwrap();
        }
        let f = failures.into_inner().unwrap();
        assert!(f.is_empty(), "{} mismatches, first: {}", f.len(), f[0]);
    }

    #[test]
    fn a_draw_that_does_not_match_the_source_labels_is_refused() {
        use crate::resample::{BootstrapConfig, cluster_bootstrap};
        let (p, t, s, c) = crossed();
        // cluster on codec while pairs are classed by source
        let cfg = BootstrapConfig {
            resamples: 1,
            seed: 1,
            level: 0.95,
        };
        cluster_bootstrap(p.len(), &Design::OneWay(&c), &cfg, |d| {
            let mut inp = input(&p, &t, &s, &c);
            inp.draw = Some(d);
            assert_eq!(class_agreement(&inp, 0.0), Err(PairError::DesignMismatch));
            Vec::new()
        })
        .unwrap();
        assert!(matches!(
            PairTable::new(&input(&p, &t, &s, &c), &Design::Rows, 0.0, None),
            Err(PairError::DesignMismatch)
        ));
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
