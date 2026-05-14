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

use cubecl::prelude::*;

mod csf_lut_v0_5_4 {
    include!("csf_lut/v0_5_4.rs");
}

pub use csf_lut_v0_5_4::{
    GE_SIGMA, LOG_L_BKG_AXIS, LOG_RHO_AXIS, LOG_S_O0_C1, LOG_S_O0_C2, LOG_S_O0_C3,
};

/// cvvdp v0.5.4's `sensitivity_correction` parameter (dB). The
/// effective scale on every CSF sensitivity is
/// `10 ** (SENSITIVITY_CORRECTION_DB / 20)`. Negative values reduce
/// metric sensitivity.
pub const SENSITIVITY_CORRECTION_DB: f32 = -0.279_742_33;

/// Channel index for the per-channel `o0_c*` tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CsfChannel {
    A = 0,
    Rg = 1,
    Vy = 2,
}

/// Number of grid points along each LUT axis.
pub const N_L_BKG: usize = 32;
pub const N_RHO: usize = 32;

/// 1-D linear interpolation in log-space along a monotonically
/// increasing axis. Returns the y-value at `x`. Clamps to the axis
/// endpoints — matches torch's `interp1q`'s behavior for queries
/// outside the table range.
fn interp1_clamped(xs: &[f32], ys: &[f32], x: f32) -> f32 {
    debug_assert_eq!(xs.len(), ys.len());
    let n = xs.len();
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[n - 1] {
        return ys[n - 1];
    }
    // Binary search for the bracket.
    let mut lo = 0usize;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if xs[mid] <= x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let t = (x - xs[lo]) / (xs[hi] - xs[lo]);
    ys[lo] + t * (ys[hi] - ys[lo])
}

fn channel_lut(cc: CsfChannel) -> &'static [f32; N_L_BKG * N_RHO] {
    match cc {
        CsfChannel::A => &LOG_S_O0_C1,
        CsfChannel::Rg => &LOG_S_O0_C2,
        CsfChannel::Vy => &LOG_S_O0_C3,
    }
}

/// Host-scalar CSF sensitivity for still-image cvvdp (omega = 0).
///
/// - `rho` — spatial frequency in cy/deg.
/// - `log_l_bkg` — log10 of background luminance in cd/m². cvvdp's
///   `csf.sensitivity()` expects this argument already in log10 space
///   (matches the LUT's L_bkg axis). The pycvvdp pipeline applies the
///   log10 before the call via `weber_contrast_pyr` — host_scalar
///   does it explicitly.
/// - `cc` — opponent channel.
pub fn sensitivity_scalar(rho: f32, log_l_bkg: f32, cc: CsfChannel) -> f32 {
    let log_rho_q = rho.max(1e-6).log10();
    let lut = channel_lut(cc);

    let mut logs_row = [0.0_f32; N_L_BKG];
    for l_idx in 0..N_L_BKG {
        let row = &lut[l_idx * N_RHO..(l_idx + 1) * N_RHO];
        logs_row[l_idx] = interp1_clamped(&LOG_RHO_AXIS, row, log_rho_q);
    }

    let log_s = interp1_clamped(&LOG_L_BKG_AXIS, &logs_row, log_l_bkg);

    10.0_f32.powf(log_s)
}

/// Sensitivity with cvvdp's published correction applied. Same
/// log10-space `log_l_bkg` convention as `sensitivity_scalar`.
pub fn sensitivity_corrected_scalar(rho: f32, log_l_bkg: f32, cc: CsfChannel) -> f32 {
    let s = sensitivity_scalar(rho, log_l_bkg, cc);
    let correction = 10.0_f32.powf(SENSITIVITY_CORRECTION_DB / 20.0);
    s * correction
}

/// Host helper: precompute the `logs_row` 32-entry array that
/// `csf_apply_per_pixel_kernel` consumes for a given `(rho, cc)`
/// pair. Interpolates the per-channel `o0_cN` LUT along the
/// `log_rho` axis at the band's spatial frequency, yielding a
/// length-`N_L_BKG` vector of `log10(S)` values parameterized by
/// `log_L_bkg`. cvvdp's `castleCSF.sensitivity` does this row
/// pull-out before its per-pixel L_bkg interp.
pub fn precompute_logs_row(rho: f32, cc: CsfChannel) -> [f32; N_L_BKG] {
    let log_rho_q = rho.max(1e-6).log10();
    let lut = channel_lut(cc);
    let mut row = [0.0_f32; N_L_BKG];
    for l_idx in 0..N_L_BKG {
        let r = &lut[l_idx * N_RHO..(l_idx + 1) * N_RHO];
        row[l_idx] = interp1_clamped(&LOG_RHO_AXIS, r, log_rho_q);
    }
    row
}

/// Per-pixel CSF apply: interpolates `logs_row` along the
/// `log_L_bkg` axis at each pixel's `log_l_bkg`, raises to 10^,
/// applies cvvdp's sensitivity correction in log space, and
/// multiplies the weber-contrast band by `S * ch_gain` to produce
/// `T_p`.
///
/// Host pre-collapses the `rho` axis via [`precompute_logs_row`];
/// the kernel only needs the resulting 32-entry row plus per-pixel
/// `log_l_bkg`.
///
/// The `log_L_bkg` axis is uniform in log space
/// (`step = (4.0 − (−2.301)) / 31`), so the kernel computes the
/// bracket index by direct arithmetic instead of binary search.
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
    let inv_step = f32::new(4.920_640_4); // 31 / (4.0 - (-2.30103))
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
    let ln_10 = f32::new(2.302_585_1);
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
    let inv_step = f32::new(4.920_640_4);
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
    let ln_10 = f32::new(2.302_585_1);

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

/// Multiply `band` in-place by `weights[weight_idx]`. `weights` is a
/// flat per-(level, channel) CSF sensitivity table uploaded once per
/// pipeline init. `weight_idx` is the host-resolved slot for the
/// (level, channel) pair this launch handles.
///
/// Per-pixel L_bkg variation is NOT modeled here; this kernel
/// applies the same scalar weight to every pixel of the band, which
/// matches the "global L_bkg approximation" code path. The full
/// per-pixel form (using gauss_a[1] as L_bkg per pixel) is the next
/// chunk after this.
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
/// expected by [`weight_band_kernel`]:
/// `[lvl0_chA, lvl0_chRG, lvl0_chVY, lvl1_chA, …]`.
///
/// `weight_idx = level * 3 + channel`.
pub fn flatten_band_weights(weights: &[[f32; 3]]) -> Vec<f32> {
    let mut out = Vec::with_capacity(weights.len() * 3);
    for w in weights {
        out.extend_from_slice(w);
    }
    out
}
