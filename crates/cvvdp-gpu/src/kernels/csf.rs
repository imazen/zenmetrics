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
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::kernels::csf::SENSITIVITY_CORRECTION_DB;
///
/// // Small negative dB (~ -0.28). The linear factor is just below 1.
/// assert!(SENSITIVITY_CORRECTION_DB < 0.0);
/// let factor = 10.0_f32.powf(SENSITIVITY_CORRECTION_DB / 20.0);
/// assert!((0.9..1.0).contains(&factor)); // ≈ 0.9684
/// ```
pub const SENSITIVITY_CORRECTION_DB: f32 = -0.279_742_33;

/// Rho (cy/deg) used for the BASEBAND CSF sensitivity lookup.
///
/// pycvvdp `process_block_of_frames` overrides the geometric
/// rho at the last pyramid band:
///
/// ```python
/// rho_band = lpyr.get_freqs()
/// rho_band[lpyr.get_band_count()-1] = 0.1 # Baseband
/// ```
/// (cvvdp_metric.py:628).
///
/// Our `band_frequencies(ppd, w, h)` returns the geometric value
/// (e.g. 0.190 at 256² standard_4k), not 0.1. Using the geometric
/// rho for the baseband CSF lookup gave a 0.117 JOD drift on
/// chroma_shift — surfaced in tick 204 by dumping pycvvdp's
/// Q_per_ch values and tracing back to baseband.
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::kernels::csf::CSF_BASEBAND_RHO;
///
/// // Hard-coded 0.1 cy/deg per pycvvdp's override.
/// assert_eq!(CSF_BASEBAND_RHO, 0.1);
///
/// // Below the geometric baseband rho at typical viewing — e.g.
/// // standard_4K + 256² produces a geometric ≈ 0.19 cy/deg.
/// assert!(CSF_BASEBAND_RHO < 0.19);
/// ```
pub const CSF_BASEBAND_RHO: f32 = 0.1;

/// Channel index for the per-channel `o0_c*` tables.
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::kernels::csf::CsfChannel;
///
/// // Discriminants pin the [A, Rg, Vy] index ordering. Many call
/// // sites do `channel as usize` to index `[_; 3]` arrays, so a
/// // reorder would silently corrupt every per-channel buffer.
/// // Pinned by tests/csf_channel_invariants.rs at compile time.
/// assert_eq!(CsfChannel::A as u32, 0);
/// assert_eq!(CsfChannel::Rg as u32, 1);
/// assert_eq!(CsfChannel::Vy as u32, 2);
///
/// // Copy + PartialEq + Debug derives ensure ergonomic use in
/// // tests and error messages.
/// let cc = CsfChannel::Rg;
/// let copy = cc;
/// assert_eq!(cc, copy);
/// assert!(format!("{cc:?}").contains("Rg"));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CsfChannel {
    /// Achromatic channel — luminance / DC.
    A = 0,
    /// Red-green chromatic opponent channel.
    Rg = 1,
    /// Violet-yellow chromatic opponent channel.
    Vy = 2,
}

/// Number of grid points along the `log_L_bkg` LUT axis. The GPU
/// kernels assume this specific size via array sizing and stride
/// arithmetic; pinned by `csf_constants_match_pycvvdp_v0_5_4` (tick
/// 394) — bumping it requires resizing the per-band `logs_row`
/// buffer in `Cvvdp::new` plus the kernel's index arithmetic.
pub const N_L_BKG: usize = 32;

/// Number of grid points along the `log_rho` LUT axis. Same kernel-
/// sizing constraints as [`N_L_BKG`] above. The two are coincidentally
/// equal at 32 in cvvdp v0.5.4's `weber_fixed_size` variant, but
/// they're conceptually independent — a future LUT revision could
/// rebalance the axes.
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::kernels::csf::{N_L_BKG, N_RHO, LOG_L_BKG_AXIS, LOG_RHO_AXIS};
///
/// // Both axes are 32 entries at cvvdp v0.5.4 — the LUTs are
/// // N_L_BKG × N_RHO = 1024 entries.
/// assert_eq!(N_L_BKG, 32);
/// assert_eq!(N_RHO, 32);
/// assert_eq!(LOG_L_BKG_AXIS.len(), N_L_BKG);
/// assert_eq!(LOG_RHO_AXIS.len(), N_RHO);
/// ```
pub const N_RHO: usize = 32;

/// 1-D linear interpolation on a UNIFORMLY-spaced axis via
/// global-stride rescale. Returns the y-value at `x`, clamping
/// queries to the axis endpoints.
///
/// Equivalent to pycvvdp's `interp1q` + `get_interpolants_quick` —
/// matching it bit-exactly closes a ~0.9% relative drift in CSF S
/// values at chrominance frequencies (tick 199 finding,
/// `docs/CHROMA_DRIFT_INVESTIGATION.md`). For uniform axes,
/// binary-search and uniform-rescale forms are mathematically
/// equivalent, but f32 storage makes the stored axis values
/// slightly non-uniform at ULP boundaries, and binary-search uses
/// a LOCAL diff `(xs[hi] − xs[lo])` while uniform-rescale uses
/// the GLOBAL diff `(xs[N−1] − xs[0]) / (N − 1)` — these can
/// disagree at the last bit, producing accumulated drift.
///
/// Used for the outer `log_L_bkg` interp in `sensitivity_scalar`
/// and `precompute_logs_row`; the inner `log_rho` interp uses
/// [`interp1_clamped`] (binary-search) instead, mirroring
/// pycvvdp's per-axis choice.
fn interp1_uniform(xs: &[f32], ys: &[f32], x: f32) -> f32 {
    debug_assert_eq!(xs.len(), ys.len());
    let n = xs.len();
    debug_assert!(n >= 2);
    // ind = (x - xs[0]) / (xs[N-1] - xs[0]) * (N - 1), clamped to [0, N-1].
    let raw_ind = (x - xs[0]) / (xs[n - 1] - xs[0]) * ((n - 1) as f32);
    let ind = raw_ind.clamp(0.0, (n - 1) as f32);
    let imin = ind as usize; // truncation = floor for non-negative
    let imax = (imin + 1).min(n - 1);
    let ifrc = ind - (imin as f32);
    ys[imin] * (1.0 - ifrc) + ys[imax] * ifrc
}

/// 1-D linear interpolation via binary-search bracket lookup.
/// Returns the y-value at `x`, clamping queries outside the
/// axis range to the endpoint y-values.
///
/// Works on any monotonically-increasing `xs`, uniformly-spaced
/// or not. Slower than [`interp1_uniform`] but doesn't depend on
/// the axis being grid-uniform at f32 precision. Used for the
/// inner `log_rho` axis interp in `sensitivity_scalar` and
/// `precompute_logs_row` — matching pycvvdp's choice of
/// `torch.searchsorted` + linear interp for the rho axis (vs.
/// `interp1q` for L_bkg). See [`interp1_uniform`] for the
/// uniform-axis fast path used on the outer L_bkg interp.
fn interp1_clamped(xs: &[f32], ys: &[f32], x: f32) -> f32 {
    debug_assert_eq!(xs.len(), ys.len());
    let n = xs.len();
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[n - 1] {
        return ys[n - 1];
    }
    // Binary search for the bracket. `usize::midpoint` is the
    // overflow-safe form — the `(lo + hi) / 2` shorthand would
    // wrap on platforms where `lo + hi > usize::MAX`, which
    // can't happen at our 32-entry LUT sizes but is the
    // canonical idiom under MSRV 1.85+.
    let mut lo = 0usize;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = usize::midpoint(lo, hi);
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
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::kernels::csf::{sensitivity_scalar, CsfChannel};
///
/// // Standard photopic background (100 cd/m² → log10 = 2.0) at
/// // 4 cy/deg (near peak CSF). Achromatic sensitivity peaks here.
/// let s_a = sensitivity_scalar(4.0, 2.0, CsfChannel::A);
/// assert!(s_a > 0.0 && s_a.is_finite());
///
/// // CSF falls off at very high frequencies: 30 cy/deg << 4 cy/deg
/// // for the achromatic channel (steep high-pass roll-off).
/// let s_high = sensitivity_scalar(30.0, 2.0, CsfChannel::A);
/// assert!(s_high < s_a);
///
/// // Channels are independent — Rg and Vy have their own LUT
/// // tables. All three positive at typical photopic L_bkg.
/// for cc in [CsfChannel::A, CsfChannel::Rg, CsfChannel::Vy] {
///     let s = sensitivity_scalar(4.0, 2.0, cc);
///     assert!(s > 0.0, "{cc:?} sensitivity should be positive");
/// }
/// ```
#[must_use]
pub fn sensitivity_scalar(rho: f32, log_l_bkg: f32, cc: CsfChannel) -> f32 {
    let log_rho_q = rho.max(1e-6).log10();
    let lut = channel_lut(cc);

    let mut logs_row = [0.0_f32; N_L_BKG];
    for l_idx in 0..N_L_BKG {
        let row = &lut[l_idx * N_RHO..(l_idx + 1) * N_RHO];
        logs_row[l_idx] = interp1_clamped(&LOG_RHO_AXIS, row, log_rho_q);
    }

    // pycvvdp uses interp1q (uniform-axis rescale) for the L_bkg
    // axis — matching this closed a ~0.9% relative drift surfaced
    // in tick 199's T_p REF probe. The rho axis ABOVE keeps
    // binary search because rho's first interval has a different
    // ratio (0.3228 vs 0.5 for the rest) — not uniform in log10.
    let log_s = interp1_uniform(&LOG_L_BKG_AXIS, &logs_row, log_l_bkg);

    10.0_f32.powf(log_s)
}

/// Sensitivity with cvvdp's published correction applied. Same
/// log10-space `log_l_bkg` convention as `sensitivity_scalar`.
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::kernels::csf::{
///     CsfChannel, sensitivity_corrected_scalar, sensitivity_scalar,
///     SENSITIVITY_CORRECTION_DB,
/// };
///
/// // Standard photopic L_bkg = 100 cd/m² (log10 = 2.0).
/// let s = sensitivity_corrected_scalar(4.0, 2.0_f32, CsfChannel::A);
/// assert!(s > 0.0, "sensitivity must be positive");
///
/// // The correction is a constant multiplicative factor —
/// // corrected = uncorrected × 10^(DB/20).
/// let factor = 10.0_f32.powf(SENSITIVITY_CORRECTION_DB / 20.0);
/// let s_unc = sensitivity_scalar(4.0, 2.0_f32, CsfChannel::A);
/// assert!((s / s_unc - factor).abs() < 1e-5);
/// ```
#[must_use]
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
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::kernels::csf::{precompute_logs_row, sensitivity_scalar, CsfChannel, N_L_BKG, LOG_L_BKG_AXIS};
///
/// // Same (rho, channel) precompute returns N_L_BKG entries.
/// let row = precompute_logs_row(4.0, CsfChannel::A);
/// assert_eq!(row.len(), N_L_BKG);
///
/// // Each entry is `log10(S)` at the L_bkg-axis grid point. Bit-
/// // identical to `sensitivity_scalar(rho, LOG_L_BKG_AXIS[i], cc).log10()`
/// // at every grid point (no interpolation needed when log_L_bkg
/// // hits an axis sample exactly).
/// for (i, &log_s) in row.iter().enumerate() {
///     let s_direct = sensitivity_scalar(4.0, LOG_L_BKG_AXIS[i], CsfChannel::A);
///     let log_s_direct = s_direct.log10();
///     assert!(
///         (log_s - log_s_direct).abs() < 1e-5,
///         "row[{i}] = {log_s} vs direct {log_s_direct}",
///     );
/// }
/// ```
#[must_use]
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
