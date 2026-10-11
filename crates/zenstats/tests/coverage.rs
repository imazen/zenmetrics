//! Coverage of the source-clustered bootstrap for a pair statistic.
//!
//! Synthetic population: every source has its own metric-noise level, so
//! ladder agreement varies by source and the source is the sampling unit.
//! For many datasets of 12 sources × 3 codecs × 6 levels, the 95 % interval
//! for same-source same-codec agreement (resampling sources, pairs weighted
//! with copy identity through `PairTable`) should cover the population value
//! close to 95 % of the time. A naive interval that treats pairs as
//! independent is the negative control: it under-covers badly.

use zenstats::pairs::{PairInput, PairTable};
use zenstats::panel::Orientation;
use zenstats::resample::{BootstrapConfig, Design, cluster_bootstrap};

const SOURCES: u32 = 12;
const CODECS: u32 = 3;
const LEVELS: u32 = 6;
/// Student t, 11 degrees of freedom, 97.5th percentile.
const T_975_11: f64 = 2.200985160082949;

/// SplitMix64 + Box–Muller; deterministic, no dependency.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

fn truth(codec: u32, level: u32) -> f64 {
    0.4 * level as f64 + 0.1 * codec as f64
}

/// One source's rows: (metric, target, codec).
fn source_rows(rng: &mut Rng) -> Vec<(f64, f64, u32)> {
    // heterogeneous per-source noise: some sources are easy, some hard
    let sd = 0.05 + 0.6 * rng.uniform().powi(2);
    let mut rows = Vec::new();
    for c in 0..CODECS {
        for l in 0..LEVELS {
            let t = truth(c, l);
            rows.push((t + sd * rng.normal(), t, c));
        }
    }
    rows
}

/// Ladder agreement of one source (same codec pairs), and its pair count.
fn ladder(rows: &[(f64, f64, u32)]) -> (f64, f64) {
    let (mut agree, mut pairs) = (0.0, 0.0);
    for i in 0..rows.len() {
        for j in (i + 1)..rows.len() {
            if rows[i].2 != rows[j].2 {
                continue;
            }
            let dt = rows[i].1 - rows[j].1;
            let dp = rows[i].0 - rows[j].0;
            pairs += 1.0;
            agree += if dp == 0.0 {
                0.5
            } else if (dp > 0.0) == (dt > 0.0) {
                1.0
            } else {
                0.0
            };
        }
    }
    (agree, pairs)
}

#[test]
fn source_clustered_pair_interval_covers_and_a_naive_one_does_not() {
    // population value: every source has the same pair count, so it is the
    // mean of per-source ladder agreement over many sources
    let mut rng = Rng(20261011);
    let n_pop = 40_000;
    let pop: f64 = (0..n_pop)
        .map(|_| {
            let (a, p) = ladder(&source_rows(&mut rng));
            a / p
        })
        .sum::<f64>()
        / n_pop as f64;

    let datasets = 400;
    let cfg = BootstrapConfig {
        resamples: 400,
        seed: 5,
        level: 0.95,
    };
    let (mut covered, mut naive_covered, mut t_covered, mut exact_covered) =
        (0usize, 0usize, 0usize, 0usize);
    for d in 0..datasets {
        let (mut pred, mut target, mut source, mut codec) = (vec![], vec![], vec![], vec![]);
        let mut per_source = Vec::new();
        for s in 0..SOURCES {
            let rows = source_rows(&mut rng);
            let (a, p) = ladder(&rows);
            per_source.push(a / p);
            for (p, t, c) in rows {
                pred.push(p);
                target.push(t);
                source.push(s);
                codec.push(c);
            }
        }
        let input = PairInput {
            predicted: &pred,
            target: &target,
            source: &source,
            codec: &codec,
            draw: None,
            orientation: Orientation::HigherIsBetter,
        };
        let design = Design::OneWay(&source);
        let table = PairTable::new(&input, &design, 0.0, None).unwrap();
        let ci = cluster_bootstrap(
            pred.len(),
            &design,
            &BootstrapConfig { seed: d, ..cfg },
            |draw| {
                vec![
                    table
                        .class_agreement(draw)
                        .unwrap()
                        .same_source_same_codec
                        .agreement(),
                ]
            },
        )
        .unwrap()[0];
        if ci.lo <= pop && pop <= ci.hi {
            covered += 1;
        }
        // small-G Student interval: t(G−1) quantile and the G/(G−1) variance correction
        let g = SOURCES as f64;
        let half_t = T_975_11 * ci.se * (g / (g - 1.0)).sqrt();
        if ci.estimate - half_t <= pop && pop <= ci.estimate + half_t {
            t_covered += 1;
        }
        // no bootstrap at all: t interval over the 12 per-source agreements
        let g = SOURCES as f64;
        let m = per_source.iter().sum::<f64>() / g;
        let sd = (per_source.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (g - 1.0)).sqrt();
        if (m - pop).abs() <= T_975_11 * sd / g.sqrt() {
            exact_covered += 1;
        }
        // naive: binomial standard error over pairs, as if pairs were independent
        let n_pairs = (SOURCES * CODECS) as f64 * (LEVELS * (LEVELS - 1) / 2) as f64;
        let p = ci.estimate;
        let half = 1.96 * (p * (1.0 - p) / n_pairs).sqrt();
        if p - half <= pop && pop <= p + half {
            naive_covered += 1;
        }
    }
    let cov = covered as f64 / datasets as f64;
    let naive = naive_covered as f64 / datasets as f64;
    let tcov = t_covered as f64 / datasets as f64;
    let ecov = exact_covered as f64 / datasets as f64;
    eprintln!(
        "population {pop:.4}: percentile {cov:.3}, student {tcov:.3}, per-source t {ecov:.3}, naive {naive:.3} over {datasets} datasets"
    );
    // Measured with 1000 datasets and B = 1000: percentile 0.884, Student
    // 0.911, per-source t 0.910, naive 0.661. With 12 sources and a skewed
    // per-source statistic no simple interval reaches 95 %; the bootstrap with
    // copy-aware pair weights matches the exact per-source t interval, which
    // is the check that the weighting is right.
    assert!(
        (tcov - ecov).abs() <= 0.03,
        "student {tcov} vs per-source t {ecov}"
    );
    assert!(cov >= 0.85, "percentile coverage {cov}");
    assert!(tcov >= 0.87, "student coverage {tcov}");
    assert!(naive <= 0.75, "naive coverage {naive} should under-cover");
}
