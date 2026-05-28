//! Contrast-sensitivity weighting per pyramid band (scalar reference).
//!
//! Phase 8c.1-B moved these out of `cvvdp-gpu::kernels::csf` so the
//! CPU crate owns the canonical scalar implementation; cvvdp-gpu
//! continues to re-export the same paths. GPU-side `#[cube(launch)]`
//! kernels remain in `cvvdp-gpu::kernels::csf`.

#[allow(dead_code, missing_docs)]
pub mod csf_lut_v0_5_4 {
    include!("csf_lut/v0_5_4.rs");
}

pub use csf_lut_v0_5_4::{
    GE_SIGMA, LOG_L_BKG_AXIS, LOG_RHO_AXIS, LOG_S_O0_C1, LOG_S_O0_C2, LOG_S_O0_C3,
};

/// cvvdp v0.5.4's `sensitivity_correction` parameter (dB).
///
/// # Examples
///
/// ```
/// use cvvdp::kernels::csf::SENSITIVITY_CORRECTION_DB;
///
/// assert!(SENSITIVITY_CORRECTION_DB < 0.0);
/// let factor = 10.0_f32.powf(SENSITIVITY_CORRECTION_DB / 20.0);
/// assert!((0.9..1.0).contains(&factor));
/// ```
pub const SENSITIVITY_CORRECTION_DB: f32 = -0.279_742_33;

/// Rho (cy/deg) used for the BASEBAND CSF sensitivity lookup.
///
/// # Examples
///
/// ```
/// use cvvdp::kernels::csf::CSF_BASEBAND_RHO;
///
/// assert_eq!(CSF_BASEBAND_RHO, 0.1);
/// assert!(CSF_BASEBAND_RHO < 0.19);
/// ```
pub const CSF_BASEBAND_RHO: f32 = 0.1;

/// Channel index for the per-channel `o0_c*` tables.
///
/// # Examples
///
/// ```
/// use cvvdp::kernels::csf::CsfChannel;
///
/// assert_eq!(CsfChannel::A as u32, 0);
/// assert_eq!(CsfChannel::Rg as u32, 1);
/// assert_eq!(CsfChannel::Vy as u32, 2);
///
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

/// Number of grid points along the `log_L_bkg` LUT axis.
pub const N_L_BKG: usize = 32;

/// Number of grid points along the `log_rho` LUT axis.
///
/// # Examples
///
/// ```
/// use cvvdp::kernels::csf::{N_L_BKG, N_RHO, LOG_L_BKG_AXIS, LOG_RHO_AXIS};
///
/// assert_eq!(N_L_BKG, 32);
/// assert_eq!(N_RHO, 32);
/// assert_eq!(LOG_L_BKG_AXIS.len(), N_L_BKG);
/// assert_eq!(LOG_RHO_AXIS.len(), N_RHO);
/// ```
pub const N_RHO: usize = 32;

/// 1-D linear interpolation on a uniformly-spaced axis.
fn interp1_uniform(xs: &[f32], ys: &[f32], x: f32) -> f32 {
    debug_assert_eq!(xs.len(), ys.len());
    let n = xs.len();
    debug_assert!(n >= 2);
    let raw_ind = (x - xs[0]) / (xs[n - 1] - xs[0]) * ((n - 1) as f32);
    let ind = raw_ind.clamp(0.0, (n - 1) as f32);
    let imin = ind as usize;
    let imax = (imin + 1).min(n - 1);
    let ifrc = ind - (imin as f32);
    ys[imin] * (1.0 - ifrc) + ys[imax] * ifrc
}

/// Returns the y-value at `x` on a monotonically-increasing `xs` axis,
/// matching pycvvdp's `interp.get_interpolants_v1` + `interp1` for the
/// `log_rho` axis EXACTLY.
fn interp1_rho_extrap(xs: &[f32], ys: &[f32], x: f32) -> f32 {
    debug_assert_eq!(xs.len(), ys.len());
    let n = xs.len();
    if x <= xs[0] {
        return ys[0];
    }
    let lo = if x >= xs[n - 1] {
        n - 2
    } else {
        let mut l = 0usize;
        let mut h = n - 1;
        while h - l > 1 {
            let mid = usize::midpoint(l, h);
            if xs[mid] <= x {
                l = mid;
            } else {
                h = mid;
            }
        }
        l
    };
    let t = (x - xs[lo]) / (xs[lo + 1] - xs[lo]);
    ys[lo] + t * (ys[lo + 1] - ys[lo])
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
/// # Examples
///
/// ```
/// use cvvdp::kernels::csf::{sensitivity_scalar, CsfChannel};
///
/// let s_a = sensitivity_scalar(4.0, 2.0, CsfChannel::A);
/// assert!(s_a > 0.0 && s_a.is_finite());
///
/// let s_high = sensitivity_scalar(30.0, 2.0, CsfChannel::A);
/// assert!(s_high < s_a);
///
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
        logs_row[l_idx] = interp1_rho_extrap(&LOG_RHO_AXIS, row, log_rho_q);
    }

    let log_s = interp1_uniform(&LOG_L_BKG_AXIS, &logs_row, log_l_bkg);

    10.0_f32.powf(log_s)
}

/// Sensitivity with cvvdp's published correction applied.
///
/// # Examples
///
/// ```
/// use cvvdp::kernels::csf::{
///     CsfChannel, sensitivity_corrected_scalar, sensitivity_scalar,
///     SENSITIVITY_CORRECTION_DB,
/// };
///
/// let s = sensitivity_corrected_scalar(4.0, 2.0_f32, CsfChannel::A);
/// assert!(s > 0.0, "sensitivity must be positive");
///
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

/// Precompute the `logs_row` 32-entry array for a given `(rho, cc)` pair.
///
/// # Examples
///
/// ```
/// use cvvdp::kernels::csf::{precompute_logs_row, sensitivity_scalar, CsfChannel, N_L_BKG, LOG_L_BKG_AXIS};
///
/// let row = precompute_logs_row(4.0, CsfChannel::A);
/// assert_eq!(row.len(), N_L_BKG);
///
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
        row[l_idx] = interp1_rho_extrap(&LOG_RHO_AXIS, r, log_rho_q);
    }
    row
}
