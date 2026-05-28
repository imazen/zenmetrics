//! Contrast-sensitivity weighting per pyramid band.
//!
//! cvvdp v0.5.4 uses a LUT-based castleCSF (the
//! `csf_lut_weber_fixed_size.json` variant referenced in
//! `cvvdp_parameters.json` as `csf = "weber_fixed_size"`):
//!
//! - 32 log-spaced background-luminance points (`L_bkg` in cd/m²)
//! - 32 log-spaced spatial-frequency points (`rho` in cy/deg)
//! - 3 channels at temporal frequency `omega = 0`:
//!   - `o0_c1` = achromatic (A)
//!   - `o0_c2` = red-green   (RG)
//!   - `o0_c3` = violet-yellow (VY)
//! - 1 channel at `omega = 5` for the achromatic temporal pathway
//!   (out of scope for still-image cvvdp; not ported here).
//!
//! Sensitivity is bilinear-interpolated in log space:
//!
//! ```text
//! logS_at_rho(L_bkg)   = interp1(log_rho, logS[L_bkg, *], log10(rho))
//! S(rho, L_bkg, cc)    = 10 ** interp1(log_L_bkg, logS_at_rho, log10(L_bkg))
//! ```
//!
//! Then multiplied by `10 ** (sensitivity_correction / 20)` from
//! `cvvdp_parameters.json` — captured as a runtime scalar
//! [`SENSITIVITY_CORRECTION_DB`] so per-pin tweaks don't require a
//! recompile.

// Tick 514: silence missing_docs warnings on items emitted by the
// #[cube(launch)] macro (see kernels/color.rs for full rationale).
#![allow(missing_docs)]

use cubecl::prelude::*;

// Phase 8c.1-C: scalar items (the csf_lut_v0_5_4 LUT module, the
// `SENSITIVITY_CORRECTION_DB` / `CSF_BASEBAND_RHO` / `N_L_BKG` /
// `N_RHO` constants, the `CsfChannel` enum, and the
// `sensitivity_scalar` / `sensitivity_corrected_scalar` /
// `precompute_logs_row` host-scalar helpers) live in
// `cvvdp::kernels::csf` so the CPU crate owns the canonical scalar
// implementation. Re-export the full surface so existing
// `cvvdp_gpu::kernels::csf::*` callsites resolve unchanged.
//
// The cube-macro `#[cube(launch)]` kernels below (`csf_apply_*_kernel`,
// `weight_band_kernel`) do not reference any of these moved items by
// name inside their cube bodies — they use literal `f32::new(...)`
// values for the LUT-axis math and consume runtime `Array<f32>` LUT
// rows via kernel arguments. No cube-macro name-resolution
// interaction.
pub use cvvdp::kernels::csf::{
    CSF_BASEBAND_RHO, CsfChannel, GE_SIGMA, LOG_L_BKG_AXIS, LOG_RHO_AXIS, LOG_S_O0_C1,
    LOG_S_O0_C2, LOG_S_O0_C3, N_L_BKG, N_RHO, SENSITIVITY_CORRECTION_DB,
    csf_lut_v0_5_4, precompute_logs_row, sensitivity_corrected_scalar, sensitivity_scalar,
};
#[cube(launch)]
pub fn csf_apply_per_pixel_kernel(
    weber: &Array<f32>,
    log_l_bkg: &Array<f32>,
    logs_row: &Array<f32>,
    t_p: &mut Array<f32>,
    ch_gain: f32,
    n: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= n as usize {
        terminate!();
    }

    // LUT axis constants (uniform log spacing).
    let axis_min = f32::new(-2.301_03);
    // 31 / (4.0 - (-2.3010299957)) = 4.919830570740858 at f64.
    // Tick 203 fix: the previous literal 4.920_640_4 was 31/6.3
    // (denominator rounded to 1 decimal) instead of 31/6.30103
    // — a 1.6e-4 relative error that compounded through the linear
    // interp to produce the 0.9% rel T_p divergence chased through
    // ticks 196-202. See docs/CHROMA_DRIFT_INVESTIGATION.md.
    let inv_step = f32::new(4.919_830_6);
    // Upper clamp just below 31.0 so floor(off_lo) ∈ [0, 30] and
    // hi_idx = lo_idx + 1 stays in [1, 31] — last valid bracket of
    // the 32-point axis. Setting it to 30.0 would mishandle queries
    // at log_l_bkg = 4.0 (axis max) by collapsing frac to 0.
    let max_idx_f = f32::new(30.999_999);

    // sensitivity_correction in log10 space: log10(10^(-0.279742/20))
    //                                    = -0.279742 / 20 = -0.013987.
    let log_correction = f32::new(-0.013_987_1);

    let log_l = log_l_bkg[idx];
    let off_raw = (log_l - axis_min) * inv_step;

    let off_lo = if off_raw < f32::new(0.0) {
        f32::new(0.0)
    } else if off_raw > max_idx_f {
        max_idx_f
    } else {
        off_raw
    };

    let lo_idx_f = f32::floor(off_lo);
    let frac = off_lo - lo_idx_f;
    let lo_idx = lo_idx_f as u32 as usize;
    let hi_idx = lo_idx + 1;

    let lo = logs_row[lo_idx];
    let hi = logs_row[hi_idx];
    let log_s_raw = lo + frac * (hi - lo);
    let log_s_corr = log_s_raw + log_correction;
    // 10^x = exp(x * ln(10)). `exp` typically maps to a single
    // hardware instruction on cubecl backends while `powf(10, x)`
    // takes the general powf path (uses logs/exps internally).
    let ln_10 = f32::new(core::f32::consts::LN_10);
    let s = f32::exp(log_s_corr * ln_10);

    t_p[idx] = weber[idx] * s * ch_gain;
}

/// 3-channel fused CSF apply — same math as `csf_apply_per_pixel_kernel`
/// but processes all three opponent channels (A, RG, VY) in one launch.
/// Replaces 3 separate launches with 1, saving the per-launch sync
/// overhead that dominates 12 MP CSF time (~70 ms per launch on cuda).
///
/// The LUT bracket computation (`off_lo`, `lo_idx`, `frac`) is
/// computed once per pixel and shared across all three channels —
/// avoids redundant `floor` + axis arithmetic in addition to the
/// launch-count win.
#[cube(launch)]
pub fn csf_apply_3ch_kernel(
    weber_a: &Array<f32>,
    weber_rg: &Array<f32>,
    weber_vy: &Array<f32>,
    log_l_bkg: &Array<f32>,
    logs_row_a: &Array<f32>,
    logs_row_rg: &Array<f32>,
    logs_row_vy: &Array<f32>,
    t_p_a: &mut Array<f32>,
    t_p_rg: &mut Array<f32>,
    t_p_vy: &mut Array<f32>,
    ch_gain_a: f32,
    ch_gain_rg: f32,
    ch_gain_vy: f32,
    n: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= n as usize {
        terminate!();
    }

    // LUT axis constants (uniform log spacing) — same as the
    // per-channel kernel above.
    let axis_min = f32::new(-2.301_03);
    let inv_step = f32::new(4.919_830_6); // tick 203 fix — see csf_apply_per_pixel_kernel
    let max_idx_f = f32::new(30.999_999);
    let log_correction = f32::new(-0.013_987_1);

    // Bracket math: shared across all 3 channels.
    let log_l = log_l_bkg[idx];
    let off_raw = (log_l - axis_min) * inv_step;
    let off_lo = if off_raw < f32::new(0.0) {
        f32::new(0.0)
    } else if off_raw > max_idx_f {
        max_idx_f
    } else {
        off_raw
    };
    let lo_idx_f = f32::floor(off_lo);
    let frac = off_lo - lo_idx_f;
    let lo_idx = lo_idx_f as u32 as usize;
    let hi_idx = lo_idx + 1;

    // 10^x = exp(x * ln(10)) — single-instruction `exp` is faster
    // than the general-purpose `powf`. Shared across channels.
    let ln_10 = f32::new(core::f32::consts::LN_10);

    // A channel.
    let lo_a = logs_row_a[lo_idx];
    let hi_a = logs_row_a[hi_idx];
    let log_s_a = lo_a + frac * (hi_a - lo_a) + log_correction;
    let s_a = f32::exp(log_s_a * ln_10);
    t_p_a[idx] = weber_a[idx] * s_a * ch_gain_a;

    // RG channel.
    let lo_rg = logs_row_rg[lo_idx];
    let hi_rg = logs_row_rg[hi_idx];
    let log_s_rg = lo_rg + frac * (hi_rg - lo_rg) + log_correction;
    let s_rg = f32::exp(log_s_rg * ln_10);
    t_p_rg[idx] = weber_rg[idx] * s_rg * ch_gain_rg;

    // VY channel.
    let lo_vy = logs_row_vy[lo_idx];
    let hi_vy = logs_row_vy[hi_idx];
    let log_s_vy = lo_vy + frac * (hi_vy - lo_vy) + log_correction;
    let s_vy = f32::exp(log_s_vy * ln_10);
    t_p_vy[idx] = weber_vy[idx] * s_vy * ch_gain_vy;
}

/// 6-channel fused CSF apply — runs both sides (REF + DIST) of one
/// pyramid level in a single launch. Replaces two
/// `csf_apply_3ch_kernel` launches per non-baseband level.
///
/// The per-pixel LUT bracket math (`log_l_bkg → (lo_idx, frac)`) is
/// computed once and shared across all 6 outputs. The 3 logs_rows
/// (one per channel) are shared between the REF and DIST sides since
/// cvvdp's `weber_g1` contract uses the REF's `log_l_bkg` + the
/// per-channel sensitivity row for both sides. The 3 `ch_gain`
/// values are also shared — REF and DIST go through the same CSF
/// weighting.
///
/// Inputs:
/// - `weber_ref_*` — REF Weber-contrast band, 3 channels.
/// - `weber_dis_*` — DIST Weber-contrast band, 3 channels.
/// - `log_l_bkg`   — per-pixel `log10(L_bkg)` from REF achromatic.
/// - `logs_row_*`  — 32-entry sensitivity LUT row per channel.
/// - `ch_gain_*`   — per-channel CSF gain (CH_GAIN × band_mul or
///                   1.0 at baseband + edge bands).
/// - `n`           — pixel count.
///
/// Outputs:
/// - `t_p_ref_*`   — REF post-CSF per channel.
/// - `t_p_dis_*`   — DIST post-CSF per channel.
#[cube(launch)]
pub fn csf_apply_6ch_kernel(
    weber_ref_a: &Array<f32>,
    weber_ref_rg: &Array<f32>,
    weber_ref_vy: &Array<f32>,
    weber_dis_a: &Array<f32>,
    weber_dis_rg: &Array<f32>,
    weber_dis_vy: &Array<f32>,
    log_l_bkg: &Array<f32>,
    logs_row_a: &Array<f32>,
    logs_row_rg: &Array<f32>,
    logs_row_vy: &Array<f32>,
    t_p_ref_a: &mut Array<f32>,
    t_p_ref_rg: &mut Array<f32>,
    t_p_ref_vy: &mut Array<f32>,
    t_p_dis_a: &mut Array<f32>,
    t_p_dis_rg: &mut Array<f32>,
    t_p_dis_vy: &mut Array<f32>,
    ch_gain_a: f32,
    ch_gain_rg: f32,
    ch_gain_vy: f32,
    n: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= n as usize {
        terminate!();
    }

    let axis_min = f32::new(-2.301_03);
    let inv_step = f32::new(4.919_830_6); // tick 203 fix — see csf_apply_per_pixel_kernel
    let max_idx_f = f32::new(30.999_999);
    let log_correction = f32::new(-0.013_987_1);

    // Bracket math: shared across all 6 outputs.
    let log_l = log_l_bkg[idx];
    let off_raw = (log_l - axis_min) * inv_step;
    let off_lo = if off_raw < f32::new(0.0) {
        f32::new(0.0)
    } else if off_raw > max_idx_f {
        max_idx_f
    } else {
        off_raw
    };
    let lo_idx_f = f32::floor(off_lo);
    let frac = off_lo - lo_idx_f;
    let lo_idx = lo_idx_f as u32 as usize;
    let hi_idx = lo_idx + 1;

    let ln_10 = f32::new(core::f32::consts::LN_10);

    // Per-channel: load logs_row pair, interpolate, scale once.
    // Apply to both REF and DIST.
    let lo_a = logs_row_a[lo_idx];
    let hi_a = logs_row_a[hi_idx];
    let log_s_a = lo_a + frac * (hi_a - lo_a) + log_correction;
    let s_a = f32::exp(log_s_a * ln_10);
    let scale_a = s_a * ch_gain_a;
    t_p_ref_a[idx] = weber_ref_a[idx] * scale_a;
    t_p_dis_a[idx] = weber_dis_a[idx] * scale_a;

    let lo_rg = logs_row_rg[lo_idx];
    let hi_rg = logs_row_rg[hi_idx];
    let log_s_rg = lo_rg + frac * (hi_rg - lo_rg) + log_correction;
    let s_rg = f32::exp(log_s_rg * ln_10);
    let scale_rg = s_rg * ch_gain_rg;
    t_p_ref_rg[idx] = weber_ref_rg[idx] * scale_rg;
    t_p_dis_rg[idx] = weber_dis_rg[idx] * scale_rg;

    let lo_vy = logs_row_vy[lo_idx];
    let hi_vy = logs_row_vy[hi_idx];
    let log_s_vy = lo_vy + frac * (hi_vy - lo_vy) + log_correction;
    let s_vy = f32::exp(log_s_vy * ln_10);
    let scale_vy = s_vy * ch_gain_vy;
    t_p_ref_vy[idx] = weber_ref_vy[idx] * scale_vy;
    t_p_dis_vy[idx] = weber_dis_vy[idx] * scale_vy;
}

/// Multiply `band` in-place by `weights[weight_idx]`. `weights` is a
/// flat per-(level, channel) CSF sensitivity table uploaded once per
/// pipeline init. `weight_idx` is the host-resolved slot for the
/// (level, channel) pair this launch handles.
///
/// Per-pixel L_bkg variation is NOT modeled here; this kernel
/// applies the same scalar weight to every pixel of the band, which
/// matches the "global L_bkg approximation" code path (used by
/// `compute_dkl_csf_weighted_bands` — test-only). The full
/// per-pixel form, which takes `gauss_a\[1\]` as L_bkg per pixel
/// and is what production uses, lives in `csf_apply_per_pixel_kernel`
/// + its fused 3-channel and 6-channel variants.
#[cube(launch)]
pub fn weight_band_kernel(band: &mut Array<f32>, weights: &Array<f32>, weight_idx: u32, n: u32) {
    let idx = ABSOLUTE_POS;
    if idx >= n as usize {
        terminate!();
    }
    band[idx] = band[idx] * weights[weight_idx as usize];
}

/// Per-band, per-channel CSF weight table for the "global L_bkg
/// approximation" path. Combines [`crate::params::DisplayGeometry`]
/// and [`crate::kernels::pyramid::band_frequencies`] with this
/// module's `sensitivity_corrected_scalar` to produce a
/// `Vec<[f32; 3]>` indexed as `[level][channel]`.
///
/// Use case: the application path that approximates background
/// luminance as a single scalar (e.g. display peak / 2, or a
/// per-image mean). The kernel above (`weight_band_kernel`) consumes
/// the flat-layout version of this table.
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::kernels::csf::precomputed_band_weights;
/// use cvvdp_gpu::kernels::pyramid::band_frequencies;
/// use cvvdp_gpu::params::DisplayGeometry;
///
/// let ppd = DisplayGeometry::STANDARD_4K.pixels_per_degree();
/// let l_bkg = 100.0_f32.log10(); // photopic 100 cd/m²
/// let weights = precomputed_band_weights(ppd, 1024, 1024, l_bkg);
///
/// // One [A, Rg, Vy] triple per pyramid band.
/// assert_eq!(weights.len(), band_frequencies(ppd, 1024, 1024).len());
///
/// // All weights are positive sensitivities, all finite.
/// for [a, rg, vy] in &weights {
///     assert!(*a > 0.0 && a.is_finite());
///     assert!(*rg > 0.0 && rg.is_finite());
///     assert!(*vy > 0.0 && vy.is_finite());
/// }
/// ```
#[must_use]
pub fn precomputed_band_weights(
    ppd: f32,
    width: usize,
    height: usize,
    l_bkg: f32,
) -> Vec<[f32; 3]> {
    let freqs = crate::kernels::pyramid::band_frequencies(ppd, width, height);
    freqs
        .iter()
        .map(|&rho| {
            [
                sensitivity_corrected_scalar(rho, l_bkg, CsfChannel::A),
                sensitivity_corrected_scalar(rho, l_bkg, CsfChannel::Rg),
                sensitivity_corrected_scalar(rho, l_bkg, CsfChannel::Vy),
            ]
        })
        .collect()
}

/// Flatten a `precomputed_band_weights` result to the layout
/// expected by `weight_band_kernel`:
/// `[lvl0_chA, lvl0_chRG, lvl0_chVY, lvl1_chA, …]`.
///
/// `weight_idx = level * 3 + channel`.
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::kernels::csf::flatten_band_weights;
///
/// // Empty input → empty output.
/// assert!(flatten_band_weights(&[]).is_empty());
///
/// // Two-level input: `[lvl0_chA, lvl0_chRG, lvl0_chVY, lvl1_chA, …]`.
/// let flat = flatten_band_weights(&[[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
/// assert_eq!(flat, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
///
/// // `weight_idx = level * 3 + channel`.
/// assert_eq!(flat[0 * 3 + 1], 2.0); // lvl0, ch_RG
/// assert_eq!(flat[1 * 3 + 2], 6.0); // lvl1, ch_VY
/// ```
#[must_use]
pub fn flatten_band_weights(weights: &[[f32; 3]]) -> Vec<f32> {
    let mut out = Vec::with_capacity(weights.len() * 3);
    for w in weights {
        out.extend_from_slice(w);
    }
    out
}

