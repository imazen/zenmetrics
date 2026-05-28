//! Scalar pooling + JOD helpers for still-image cvvdp.
//!
//! Phase 8c.1-B moved these out of `cvvdp-gpu::kernels::pool` so the
//! CPU crate owns the canonical scalar implementation; cvvdp-gpu
//! continues to re-export the same paths via shim. The GPU-side
//! `#[cube(launch)]` kernels remain in `cvvdp-gpu::kernels::pool`.
//!
//! cvvdp v0.5.4's pipeline collapses per-pixel masked differences `D`
//! into a scalar quality-in-JOD via a 3-stage Minkowski pool plus a
//! piecewise transform:
//!
//! 1. **Spatial pooling per band per channel** (beta = 2 = RMS).
//! 2. **Band pooling per channel** (beta_sch = 4).
//! 3. **Channel pooling** (beta_tch = 4).
//! 4. **Image integration**: `Q = Q_tc * image_int`.
//! 5. **JOD mapping**: piecewise (smooth at Q = 0.1).
//!
//! Still-image 3-channel only.

use alloc::vec::Vec;

/// Spatial-pooling exponent (cvvdp `beta`). RMS-equivalent for p=2.
pub const BETA_SPATIAL: f32 = 2.0;

/// Spatial-channels (= spatial bands) pooling exponent (`beta_sch`).
pub const BETA_BAND: f32 = 4.0;

/// Temporal/chromatic-channel pooling exponent (`beta_tch`). For
/// still-image 3-channel use this is the across-channel exponent.
///
/// # Examples
///
/// ```
/// use cvvdp::kernels::pool::{BETA_BAND, BETA_CH, BETA_SPATIAL};
///
/// // Spatial pooling is RMS (p = 2); band + channel pooling are
/// // higher-order (p = 4) for steeper peak-emphasis.
/// assert_eq!(BETA_SPATIAL, 2.0);
/// assert_eq!(BETA_BAND, 4.0);
/// assert_eq!(BETA_CH, 4.0);
///
/// // BETA_BAND == BETA_CH; spatial is the gentler exponent.
/// assert!(BETA_SPATIAL < BETA_BAND);
/// ```
pub const BETA_CH: f32 = 4.0;

/// Image integration correction (`image_int`).
pub const IMAGE_INT: f32 = 0.577_918_3;

/// JOD mapping scale (`jod_a`).
pub const JOD_A: f32 = 0.043_956_94;

/// JOD mapping exponent (`jod_exp`).
///
/// # Examples
///
/// ```
/// use cvvdp::kernels::pool::{JOD_A, JOD_EXP, IMAGE_INT, met2jod};
///
/// // The piecewise met2jod uses both: above Q = 0.1, output is
/// // `10 - JOD_A * Q^JOD_EXP`. Quick sanity check at Q = 1:
/// let q = 1.0_f32;
/// let expected = 10.0 - JOD_A * q.powf(JOD_EXP);
/// assert!((met2jod(q) - expected).abs() < 1e-5);
///
/// // IMAGE_INT is a positive correction factor in (0, 1) used
/// // when scaling cross-band aggregates.
/// assert!(IMAGE_INT > 0.0 && IMAGE_INT < 1.0);
/// ```
pub const JOD_EXP: f32 = 0.930_204_27;

/// Per-channel weights for still-image 3-channel.
///
/// # Examples
///
/// ```
/// use cvvdp::kernels::pool::PER_CH_W;
///
/// assert_eq!(PER_CH_W, [1.0, 1.0, 1.0]);
/// ```
pub const PER_CH_W: [f32; 3] = [1.0, 1.0, 1.0];

/// Baseband weight per channel.
///
/// # Examples
///
/// ```
/// use cvvdp::kernels::pool::BASEBAND_W;
///
/// assert_eq!(BASEBAND_W.len(), 3);
/// for &w in &BASEBAND_W {
///     assert!(w > 0.0 && w.is_finite());
/// }
/// assert!(BASEBAND_W[2] > BASEBAND_W[0]);
/// assert!(BASEBAND_W[1] > BASEBAND_W[0]);
/// ```
pub const BASEBAND_W: [f32; 3] = [0.003_633_448_6, 1.662_772_4, 4.118_745_3];

/// Epsilon used by cvvdp's `safe_pow` throughout `lp_norm`.
const LP_SAFE_EPS: f32 = 1e-5;

#[inline]
fn safe_pow_lp(x: f32, p: f32) -> f32 {
    (x.abs() + LP_SAFE_EPS).powf(p) - LP_SAFE_EPS.powf(p)
}

/// cvvdp's `lp_norm` with `normalize=True`.
///
/// # Examples
///
/// ```
/// use cvvdp::kernels::pool::lp_norm_mean;
///
/// assert_eq!(lp_norm_mean(&[], 2.0), 0.0);
///
/// let v = lp_norm_mean(&[3.0_f32; 4], 2.0);
/// assert!((v - 3.0).abs() < 0.01, "got {v}, expected ≈ 3");
///
/// let pos = lp_norm_mean(&[3.0_f32, 4.0], 2.0);
/// let mixed = lp_norm_mean(&[-3.0_f32, 4.0], 2.0);
/// assert!((pos - mixed).abs() < 1e-5);
/// ```
#[must_use]
pub fn lp_norm_mean(values: &[f32], p: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let n = values.len() as f32;
    let acc: f32 = values.iter().map(|v| safe_pow_lp(*v, p)).sum();
    safe_pow_lp(acc / n, 1.0 / p)
}

/// cvvdp's `lp_norm` with `normalize=False`.
///
/// # Examples
///
/// ```
/// use cvvdp::kernels::pool::lp_norm_sum;
///
/// let v = lp_norm_sum(&[3.0_f32, 4.0], 2.0);
/// assert!((v - 5.0).abs() < 0.01, "got {v}, expected ≈ 5");
///
/// assert_eq!(lp_norm_sum(&[], 2.0), 0.0);
///
/// let pos = lp_norm_sum(&[3.0_f32, 4.0], 2.0);
/// let neg = lp_norm_sum(&[-3.0_f32, -4.0], 2.0);
/// assert!((pos - neg).abs() < 1e-5);
/// ```
#[must_use]
pub fn lp_norm_sum(values: &[f32], p: f32) -> f32 {
    let acc: f32 = values.iter().map(|v| safe_pow_lp(*v, p)).sum();
    safe_pow_lp(acc, 1.0 / p)
}

/// cvvdp's smooth piecewise JOD mapping (`met2jod`).
///
/// # Examples
///
/// ```
/// use cvvdp::kernels::pool::met2jod;
///
/// assert_eq!(met2jod(0.0), 10.0);
///
/// assert!(met2jod(0.5) < met2jod(0.0));
/// assert!(met2jod(1.0) < met2jod(0.5));
/// assert!(met2jod(5.0) < met2jod(1.0));
///
/// assert!(met2jod(1e6).is_finite());
/// assert!(met2jod(1e6) < 0.0);
/// ```
#[must_use]
pub fn met2jod(q: f32) -> f32 {
    let q_t = 0.1_f32;
    if q <= q_t {
        let jod_a_p = JOD_A * q_t.powf(JOD_EXP - 1.0);
        10.0 - jod_a_p * q
    } else {
        10.0 - JOD_A * q.powf(JOD_EXP)
    }
}

/// Apply the full cvvdp pool + JOD pipeline for a still image with
/// 3 channels.
///
/// # Panics
///
/// Panics if `q_per_ch` is empty.
///
/// # Examples
///
/// ```
/// use cvvdp::kernels::pool::do_pooling_and_jod_still_3ch;
///
/// let zero = vec![[0.0_f32; 3]; 5];
/// let jod_max = do_pooling_and_jod_still_3ch(&zero);
/// assert!((jod_max - 10.0).abs() < 1e-3);
///
/// let some_distortion = vec![[0.3_f32, 0.2, 0.15]; 5];
/// let jod = do_pooling_and_jod_still_3ch(&some_distortion);
/// assert!(jod < jod_max);
/// assert!(jod.is_finite());
/// ```
#[must_use]
pub fn do_pooling_and_jod_still_3ch(q_per_ch: &[[f32; 3]]) -> f32 {
    let n_levels = q_per_ch.len();
    assert!(n_levels >= 1, "need at least one pyramid level");

    let mut q_sc = [0.0_f32; 3];
    for c in 0..3 {
        let mut weighted = Vec::with_capacity(n_levels);
        for (k, level) in q_per_ch.iter().enumerate() {
            let per_sband_w = if k == n_levels - 1 {
                BASEBAND_W[c]
            } else {
                1.0
            };
            weighted.push(level[c] * PER_CH_W[c] * per_sband_w);
        }
        q_sc[c] = lp_norm_sum(&weighted, BETA_BAND);
    }

    let q_tc = lp_norm_sum(&q_sc, BETA_CH);
    let q = q_tc * IMAGE_INT;
    met2jod(q)
}
