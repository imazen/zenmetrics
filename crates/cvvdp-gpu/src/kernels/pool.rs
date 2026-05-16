//! Pooling + JOD for still-image cvvdp.
//!
//! cvvdp v0.5.4's pipeline collapses per-pixel masked differences `D`
//! into a scalar quality-in-JOD via a 3-stage Minkowski pool plus a
//! piecewise transform:
//!
//! 1. **Spatial pooling per band per channel** (beta = 2 = RMS):
//!    `Q_per_ch[c, k] = (mean over pixels of D[c, :, :]^2)^(1/2)`.
//! 2. **Band pooling per channel** (beta_sch = 4):
//!    `Q_sc[c] = (sum_k (Q_per_ch[c, k] * per_ch_w[c] * per_sband_w[c, k])^4)^(1/4)`
//!    where `per_sband_w[c, k] = 1` for `k < n_levels - 1` and
//!    `per_sband_w[c, last] = baseband_weight[c]`.
//! 3. **Channel pooling** (beta_tch = 4):
//!    `Q_tc = (sum_c Q_sc[c]^4)^(1/4)`.
//! 4. **Image integration**: `Q = Q_tc * image_int`.
//! 5. **JOD mapping**: piecewise (smooth at Q = 0.1).
//!
//! Constants are baked from `cvvdp_parameters.json`. Still-image
//! 3-channel only — temporal channel (no_frames > 1) lives outside
//! this module.

// Tick 514: silence missing_docs warnings on items emitted by the
// #[cube(launch)] macro (see kernels/color.rs for full rationale).
#![allow(missing_docs)]

use cubecl::prelude::*;

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
/// use cvvdp_gpu::kernels::pool::{BETA_BAND, BETA_CH, BETA_SPATIAL};
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
/// use cvvdp_gpu::kernels::pool::{JOD_A, JOD_EXP, IMAGE_INT, met2jod};
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

/// Per-channel weights for still-image 3-channel. Derived from
/// `per_ch_w_all = [1, ch_chrom_w, ch_chrom_w, ch_trans_w]` sliced
/// to first 3 channels. `ch_chrom_w = 1.0` in cvvdp v0.5.4.
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::kernels::pool::PER_CH_W;
///
/// // 3 channels, all weights 1.0 (no per-channel attenuation
/// // at the pool stage — chroma weighting happens earlier via
/// // masking::CH_GAIN).
/// assert_eq!(PER_CH_W, [1.0, 1.0, 1.0]);
/// ```
pub const PER_CH_W: [f32; 3] = [1.0, 1.0, 1.0];

/// Baseband (= last spatial band) weight per channel. cvvdp uses
/// the first 3 entries of `baseband_weight` for still-image
/// 3-channel.
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::kernels::pool::BASEBAND_W;
///
/// // 3 channels, all positive — the achromatic channel (index 0)
/// // is heavily attenuated at the baseband (≈ 0.004) because
/// // low-spatial-frequency luminance is below the CSF threshold;
/// // chroma channels (RG/VY) are weighted more (≈ 1.66 / 4.12).
/// assert_eq!(BASEBAND_W.len(), 3);
/// for &w in &BASEBAND_W {
///     assert!(w > 0.0 && w.is_finite());
/// }
/// // Chroma dominance at baseband — Vy is the largest.
/// assert!(BASEBAND_W[2] > BASEBAND_W[0]);
/// assert!(BASEBAND_W[1] > BASEBAND_W[0]);
/// ```
pub const BASEBAND_W: [f32; 3] = [0.003_633_448_6, 1.662_772_4, 4.118_745_3];

/// Epsilon used by cvvdp's `safe_pow` throughout `lp_norm`. The
/// epsilon-shifted form is what makes `lp_norm` differentiable at
/// x=0; matching it bit-exactly matters because near-zero pooling
/// inputs (high-quality images) produce wildly different results
/// from the naive `x^p` form.
const LP_SAFE_EPS: f32 = 1e-5;

#[inline]
fn safe_pow_lp(x: f32, p: f32) -> f32 {
    (x.abs() + LP_SAFE_EPS).powf(p) - LP_SAFE_EPS.powf(p)
}

/// cvvdp's `lp_norm` with `normalize=True`. Matches:
/// `safe_pow(sum_i(safe_pow(x_i, p)) / N, 1/p)` where N = `values.len()`.
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::kernels::pool::lp_norm_mean;
///
/// // Empty input → 0 (documented contract).
/// assert_eq!(lp_norm_mean(&[], 2.0), 0.0);
///
/// // For a uniform constant input, lp_norm_mean ≈ that constant
/// // (mean-of-N-copies normalization cancels). The small eps-tail
/// // bias from safe_pow is < 0.01.
/// let v = lp_norm_mean(&[3.0_f32; 4], 2.0);
/// assert!((v - 3.0).abs() < 0.01, "got {v}, expected ≈ 3");
///
/// // Sign-insensitive via |x| — negatives don't break the result.
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

/// cvvdp's `lp_norm` with `normalize=False`. Matches:
/// `safe_pow(sum_i(safe_pow(x_i, p)), 1/p)`.
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::kernels::pool::lp_norm_sum;
///
/// // Pythagorean: lp_norm_sum([3, 4], 2) ≈ 5 (L2 norm minus a tiny
/// // eps-tail bias from safe_pow's regularization).
/// let v = lp_norm_sum(&[3.0_f32, 4.0], 2.0);
/// assert!((v - 5.0).abs() < 0.01, "got {v}, expected ≈ 5");
///
/// // Empty input — sum is 0, so the result is `safe_pow(0, 1/p) = 0`.
/// assert_eq!(lp_norm_sum(&[], 2.0), 0.0);
///
/// // Sign-insensitive via |x| inside safe_pow.
/// let pos = lp_norm_sum(&[3.0_f32, 4.0], 2.0);
/// let neg = lp_norm_sum(&[-3.0_f32, -4.0], 2.0);
/// assert!((pos - neg).abs() < 1e-5);
/// ```
#[must_use]
pub fn lp_norm_sum(values: &[f32], p: f32) -> f32 {
    let acc: f32 = values.iter().map(|v| safe_pow_lp(*v, p)).sum();
    safe_pow_lp(acc, 1.0 / p)
}

/// cvvdp's smooth piecewise JOD mapping (`met2jod`):
///
/// - For `Q ≤ 0.1`: `Q_JOD = 10 - jod_a * 0.1^(jod_exp - 1) * Q`
///   (linear extension that matches the slope of the power curve
///   at Q = 0.1, avoiding the zero-derivative singularity).
/// - For `Q > 0.1`: `Q_JOD = 10 - jod_a * Q^jod_exp`.
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::kernels::pool::met2jod;
///
/// // Perfect-quality limit.
/// assert_eq!(met2jod(0.0), 10.0);
///
/// // JOD declines monotonically with distortion magnitude.
/// assert!(met2jod(0.5) < met2jod(0.0));
/// assert!(met2jod(1.0) < met2jod(0.5));
/// assert!(met2jod(5.0) < met2jod(1.0));
///
/// // Output stays finite even at extreme distortion (goes
/// // deeply negative, but never NaN/Inf).
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
/// 3 channels. Input shape: `q_per_ch[c][k]` = quality per channel
/// per pyramid level, where level 0 is the finest band and the
/// last level is the coarse residual.
///
/// Returns JOD ∈ [0, 10] where 10 = imperceptible difference.
///
/// # Panics
///
/// Panics if `q_per_ch` is empty (`n_levels == 0`). At least one
/// pyramid level (the baseband) must be present — cvvdp's pool
/// stage is undefined on a zero-band input.
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::kernels::pool::do_pooling_and_jod_still_3ch;
///
/// // All-zero contrasts → no distortion → JOD ≈ 10.
/// let zero = vec![[0.0_f32; 3]; 5];
/// let jod_max = do_pooling_and_jod_still_3ch(&zero);
/// assert!((jod_max - 10.0).abs() < 1e-3);
///
/// // Some non-zero contrasts → JOD < 10.
/// let some_distortion = vec![[0.3_f32, 0.2, 0.15]; 5];
/// let jod = do_pooling_and_jod_still_3ch(&some_distortion);
/// assert!(jod < jod_max);
/// assert!(jod.is_finite());
/// ```
#[must_use]
pub fn do_pooling_and_jod_still_3ch(q_per_ch: &[[f32; 3]]) -> f32 {
    let n_levels = q_per_ch.len();
    assert!(n_levels >= 1, "need at least one pyramid level");

    // Step 1: per-channel band pooling. For each channel c, build the
    // vector of n_levels weighted values then lp_norm_sum at beta_sch.
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

    // Step 2: across-channel pooling.
    let q_tc = lp_norm_sum(&q_sc, BETA_CH);

    // Step 3: image integration correction.
    let q = q_tc * IMAGE_INT;

    // Step 4: JOD mapping.
    met2jod(q)
}

/// One thread per pixel computes cvvdp's `safe_pow(|x|, β) =
/// (|x| + 1e-5)^β - 1e-5^β` for the pixel and atomically adds it
/// into the f32 accumulator at `partials[partial_idx]`. Host folds
/// the partial to the final lp_norm via:
///
/// ```text
/// Q = safe_pow(partial / n_pixels, 1/β)
///   = ((partial / n_pixels) + 1e-5)^(1/β) - 1e-5^(1/β)
/// ```
///
/// `partial_idx` lets the caller pack multiple (band, channel)
/// partials into the same buffer. Works on cubecl backends with
/// `Atomic<f32>::fetch_add` support — CUDA, DX12, HIP (per
/// butteraugli-gpu's notes; Metal silently no-ops on the f32 add).
///
/// **Not dispatched by `Cvvdp::compute_dkl_jod`** — the production
/// path uses the 3-channel fused [`pool_band_3ch_kernel()`] (one
/// launch per band instead of three). `pool_band_kernel` is kept
/// as a test-only entry point for the scalar parity test
/// `tests/pool_scalar.rs::pool_band_kernel_matches_host_lp_norm_mean`.
#[cube(launch)]
pub fn pool_band_kernel(
    band_diff: &Array<f32>,
    partials: &mut Array<Atomic<f32>>,
    beta: f32,
    partial_idx: u32,
    n: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= n as usize {
        terminate!();
    }
    let v = band_diff[idx];
    let abs_v = if v < f32::new(0.0) { -v } else { v };
    let eps = f32::new(1e-5);
    // safe_pow_lp(|v|, beta) — accumulator gets the raw safe-pow
    // contribution; the - eps^beta and 1/beta exponentiation
    // happen host-side once per (band, channel).
    let contribution = f32::powf(abs_v + eps, beta) - f32::powf(eps, beta);
    partials[partial_idx as usize].fetch_add(contribution);
}

/// 3-channel fused version of `pool_band_kernel`. Same per-pixel
/// safe_pow math, but takes 3 input arrays and 3 partial slot
/// indices, doing 3 atomic-adds per thread (each into a different
/// slot of `partials`). Eliminates 2/3 of the launch overhead for
/// the per-band pool dispatch in `compute_dkl_jod`.
///
/// Each thread reads `band_diff_{a,rg,vy}[idx]`, computes the
/// `safe_pow` contribution for each channel, and atomically adds
/// to `partials[partial_idx_{a,rg,vy}]`. The host-side fold and
/// `pool_band_finalize` semantics are unchanged.
///
/// Pool atomics into distinct slots don't contend across channels,
/// so the atomic-throughput characteristic is the same as 3 separate
/// launches — the win is purely launch-overhead reduction (which
/// matters more at small image sizes per the tick 164 size sweep).
#[cube(launch)]
pub fn pool_band_3ch_kernel(
    band_diff_a: &Array<f32>,
    band_diff_rg: &Array<f32>,
    band_diff_vy: &Array<f32>,
    partials: &mut Array<Atomic<f32>>,
    beta: f32,
    partial_idx_a: u32,
    partial_idx_rg: u32,
    partial_idx_vy: u32,
    n: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= n as usize {
        terminate!();
    }
    let eps = f32::new(1e-5);
    let eps_pow_beta = f32::powf(eps, beta);

    let v_a = band_diff_a[idx];
    let abs_a = if v_a < f32::new(0.0) { -v_a } else { v_a };
    let c_a = f32::powf(abs_a + eps, beta) - eps_pow_beta;
    partials[partial_idx_a as usize].fetch_add(c_a);

    let v_rg = band_diff_rg[idx];
    let abs_rg = if v_rg < f32::new(0.0) { -v_rg } else { v_rg };
    let c_rg = f32::powf(abs_rg + eps, beta) - eps_pow_beta;
    partials[partial_idx_rg as usize].fetch_add(c_rg);

    let v_vy = band_diff_vy[idx];
    let abs_vy = if v_vy < f32::new(0.0) { -v_vy } else { v_vy };
    let c_vy = f32::powf(abs_vy + eps, beta) - eps_pow_beta;
    partials[partial_idx_vy as usize].fetch_add(c_vy);
}

/// Write the same `value` to every slot of `dest`. Used by the
/// baseband CSF path in `_dispatch_d_bands_into_scratch` to fill
/// `baseband_log_l_bkg` from the host-computed scalar
/// `log_l_bkg_baseband` — replaces a host `vec![value; n]` alloc
/// + GPU upload with a single GPU launch and zero host bytes.
#[cube(launch)]
pub fn fill_f32_kernel(dest: &mut Array<f32>, value: f32, n: u32) {
    let idx = ABSOLUTE_POS;
    if idx >= n as usize {
        terminate!();
    }
    dest[idx] = value;
}

/// Finish the host-side fold for the per-band atomic-pool
/// kernels ([`pool_band_kernel()`] and the fused
/// [`pool_band_3ch_kernel()`] used in production): given the
/// atomic partial sum and pixel count for one (band, channel)
/// slot, return the lp_norm_mean(β) value matching
/// `kernels::pool::lp_norm_mean`. Same algebra regardless of
/// which kernel produced the partial — both write the raw
/// `safe_pow(|x|, β)` contribution into `partials[partial_idx]`.
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::kernels::pool::pool_band_finalize;
///
/// // Zero partial → zero output (apart from the eps-tail bias,
/// // which the function explicitly cancels via `- eps.powf(1/β)`).
/// assert_eq!(pool_band_finalize(0.0, 100, 2.0), 0.0);
///
/// // Negative partial clamps to zero (the kernel sometimes produces
/// // a tiny negative from f32 atomic rounding).
/// assert_eq!(pool_band_finalize(-1e-7, 100, 2.0), 0.0);
///
/// // For a uniform |x| = c contribution, partial = N * c^β,
/// // and the finalized output is ≈ c minus the constant eps-tail
/// // `eps^(1/β)` (~ 0.056 at β=4 for eps=1e-5; ~ 0.003 at β=2).
/// // Use β=2 here so the tolerance is meaningful.
/// let c = 2.0_f32;
/// let n = 100_usize;
/// let beta = 2.0_f32;
/// let partial = (n as f32) * c.powf(beta);
/// let v = pool_band_finalize(partial, n, beta);
/// assert!((v - c).abs() < 0.01, "got {v}, expected ≈ {c}");
/// ```
#[must_use]
pub fn pool_band_finalize(partial: f32, n_pixels: usize, beta: f32) -> f32 {
    let n = n_pixels as f32;
    let eps = 1e-5_f32;
    ((partial / n).max(0.0) + eps).powf(1.0 / beta) - eps.powf(1.0 / beta)
}
