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

#[cfg(test)]
mod tests {
    use super::*;

    /// Pre-fix behaviour: flat-clamp at BOTH axis ends. Kept inline as
    /// the regression reference so the interior bit-identity test below
    /// can prove the high-PPD fix (`interp1_rho_extrap`) did not move
    /// any in-axis query. This is exactly the old `interp1_clamped`
    /// that the conformance Finding-A fix replaced.
    fn interp1_flat_clamp_ref(xs: &[f32], ys: &[f32], x: f32) -> f32 {
        let n = xs.len();
        if x <= xs[0] {
            return ys[0];
        }
        if x >= xs[n - 1] {
            return ys[n - 1]; // the clamp the fix replaced with extrapolation
        }
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
        let t = (x - xs[l]) / (xs[l + 1] - xs[l]);
        ys[l] + t * (ys[l + 1] - ys[l])
    }

    /// Pins the three regimes of `interp1_rho_extrap` on a synthetic
    /// axis with an exact (representable) slope: flat-clamp below,
    /// linear interp inside, LINEAR EXTRAPOLATION above (matching
    /// pycvvdp's `get_interpolants_v1`, which does NOT clamp `ifrc` at
    /// the high end).
    #[test]
    fn interp1_rho_extrap_contract_synthetic() {
        let xs = [0.0_f32, 1.0, 2.0, 3.0];
        let ys = [10.0_f32, 20.0, 30.0, 40.0]; // slope +10 / interval

        // Below the axis: flat clamp to ys[0].
        assert_eq!(interp1_rho_extrap(&xs, &ys, -5.0), 10.0);
        assert_eq!(interp1_rho_extrap(&xs, &ys, 0.0), 10.0);

        // Interior: exact knots + linear midpoints.
        assert_eq!(interp1_rho_extrap(&xs, &ys, 2.0), 30.0);
        assert_eq!(interp1_rho_extrap(&xs, &ys, 1.5), 25.0);

        // At the top knot: exact.
        assert_eq!(interp1_rho_extrap(&xs, &ys, 3.0), 40.0);

        // ABOVE the axis: extrapolate with the last interval's slope.
        // lo = n-2 = 2, t = (x - xs[2]) / (xs[3] - xs[2]).
        //   x = 4.0 -> t = 2 -> 30 + 2*(40-30) = 50.
        //   x = 5.0 -> t = 3 -> 30 + 3*(40-30) = 60.
        assert_eq!(interp1_rho_extrap(&xs, &ys, 4.0), 50.0);
        assert_eq!(interp1_rho_extrap(&xs, &ys, 5.0), 60.0);

        // The OLD flat-clamp would have returned 40.0 above the axis —
        // the whole point of the fix is that these now differ.
        assert_ne!(
            interp1_rho_extrap(&xs, &ys, 5.0),
            interp1_flat_clamp_ref(&xs, &ys, 5.0)
        );
        assert_eq!(interp1_flat_clamp_ref(&xs, &ys, 5.0), 40.0);
    }

    /// Conformance regression: the Finding-A fix changes ONLY
    /// above-axis queries. Every in-axis query must be BIT-IDENTICAL
    /// to the pre-fix flat-clamp behaviour. This is the unit-level
    /// proof of "zero regression on the 248 non-iphone conformance
    /// cells" (whose finest band sits inside the 64 cy/deg axis).
    #[test]
    fn interp1_rho_extrap_interior_bit_identical_to_flat_clamp() {
        let xs = LOG_RHO_AXIS;
        let ys = &LOG_S_O0_C1[..N_RHO]; // first L_bkg row of channel A
        let lo = xs[0];
        let hi = xs[N_RHO - 1];

        // Sweep strictly-interior points (k/1000 < 1 keeps x < hi).
        for k in 0..1000 {
            let x = lo + (hi - lo) * (k as f32) / 1000.0;
            let a = interp1_rho_extrap(&xs, ys, x);
            let b = interp1_flat_clamp_ref(&xs, ys, x);
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "interior query x={x} moved: extrap={a} clamp={b}"
            );
        }

        // Every interior knot (exclude the top knot, where the extrap
        // path computes ys[n-2] + 1.0*(ys[n-1]-ys[n-2]) and may differ
        // from ys[n-1] by up to 1 ULP via fp re-association — not an
        // in-axis query).
        for (i, &x) in xs.iter().enumerate().take(N_RHO - 1) {
            assert_eq!(
                interp1_rho_extrap(&xs, ys, x).to_bits(),
                interp1_flat_clamp_ref(&xs, ys, x).to_bits(),
                "knot {i} (x={x}) moved",
            );
        }
    }

    /// Conformance Finding-A guard. `iphone_14_pro` (ppd ≈ 159.6) puts
    /// the finest pyramid band at rho ≈ ppd/2 ≈ 79.8 cy/deg — BEYOND
    /// the CSF `log_rho` axis top of 64 cy/deg. The pre-fix flat-clamp
    /// held the 64-cy/deg sensitivity (~2× too high); pycvvdp keeps the
    /// CSF falling off (linear extrapolation in log-S). This test fails
    /// if anyone reverts the rho axis to flat-clamp.
    #[test]
    fn interp1_rho_extrap_high_ppd_falls_off_below_clamp() {
        let rho_band0 = 79.8_f32; // ≈ iphone_14_pro ppd / 2
        let log_rho = rho_band0.log10();

        // The query is genuinely above the axis (else this guard is moot).
        assert!(
            log_rho > LOG_RHO_AXIS[N_RHO - 1],
            "expected above-axis query: log_rho={log_rho} axis_top={}",
            LOG_RHO_AXIS[N_RHO - 1]
        );

        let ys = &LOG_S_O0_C1[..N_RHO]; // channel-A bottom L_bkg row
        // Precondition: CSF is rolling off at the axis top.
        assert!(
            ys[N_RHO - 1] < ys[N_RHO - 2],
            "precondition: CSF log-S must be falling at the axis top"
        );

        let extrap = interp1_rho_extrap(&LOG_RHO_AXIS, ys, log_rho);
        let clamp_top = ys[N_RHO - 1];
        assert!(
            extrap < clamp_top,
            "extrapolated log-S {extrap} should fall below the clamp {clamp_top}"
        );

        // Exact contract on the real axis + LUT: last-interval extrapolation.
        let t = (log_rho - LOG_RHO_AXIS[N_RHO - 2])
            / (LOG_RHO_AXIS[N_RHO - 1] - LOG_RHO_AXIS[N_RHO - 2]);
        let manual = ys[N_RHO - 2] + t * (ys[N_RHO - 1] - ys[N_RHO - 2]);
        assert_eq!(extrap.to_bits(), manual.to_bits());

        // End-to-end via the public sensitivity API (the path
        // cvvdp's pipeline drives through `precompute_logs_row`):
        // sensitivity at the above-axis band must be below the
        // (clamped) axis-top sensitivity, and finite/positive.
        let log_l = 2.0_f32; // photopic 100 cd/m²
        let s_band0 = sensitivity_scalar(rho_band0, log_l, CsfChannel::A);
        let rho_axis_top = 10.0_f32.powf(LOG_RHO_AXIS[N_RHO - 1]); // ≈ 64 cy/deg
        let s_axis_top = sensitivity_scalar(rho_axis_top, log_l, CsfChannel::A);
        assert!(
            s_band0 < s_axis_top,
            "high-PPD sensitivity {s_band0} should be below axis-top {s_axis_top}"
        );
        assert!(s_band0 > 0.0 && s_band0.is_finite());
    }
}
