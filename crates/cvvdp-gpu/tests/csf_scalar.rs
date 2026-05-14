//! Parity test for the host-scalar CSF against pycvvdp v0.5.4's
//! `castleCSF(csf_version='weber_fixed_size')`.
//!
//! Sweeps 5 spatial frequencies × 4 background luminances × 3
//! channels (= 60 points) and asserts the Rust output is within
//! 1e-3 relative of the Python reference. Tolerance reflects:
//!
//! - Bilinear LUT interp at the same axis points produces tight
//!   parity (the 32×32 grid is identical), but
//! - The `10^x` at the end amplifies relative error in the
//!   interpolation by ~ln(10) ≈ 2.3×, so a few ULPs in `log_s` can
//!   show up as ~1e-4 relative at the sensitivity output.
//!
//! 1e-3 is a safe ceiling that catches axis-mismatches but absorbs
//! float-precision tails.
//!
//! Generator:
//!
//! ```python
//! from pycvvdp.csf import castleCSF
//! csf = castleCSF(csf_version='weber_fixed_size', device='cpu')
//! # iterate rho × l_bkg × cc, call csf.sensitivity(rho, omega=0, ...)
//! ```

use cvvdp_gpu::kernels::csf::{
    CsfChannel, flatten_band_weights, precomputed_band_weights, sensitivity_scalar,
};
use cvvdp_gpu::params::DisplayGeometry;

// Goldens generated with cvvdp's actual production semantics:
// `logL_bkg` parameter passed RAW (not log10) — matches
// cvvdp_metric.py:651. Saturation visible at L_bkg = 30 and 200
// (same S because the LUT's log-axis clamps for query > axis max).
#[rustfmt::skip]
const CSF_GOLDENS: &[(f32, f32, u32, f32)] = &[
    (0.5, 0.1,   0, 4.5400723e+01), (0.5, 0.1,   1, 7.3239441e+01), (0.5, 0.1,   2, 1.9086563e+01),
    (0.5, 1.0,   0, 5.8508675e+01), (0.5, 1.0,   1, 1.8132967e+02), (0.5, 1.0,   2, 4.0926369e+01),
    (0.5, 30.0,  0, 3.5690781e+01), (0.5, 30.0,  1, 3.8426025e+02), (0.5, 30.0,  2, 9.2338120e+01),
    (0.5, 200.0, 0, 3.5690781e+01), (0.5, 200.0, 1, 3.8426025e+02), (0.5, 200.0, 2, 9.2338120e+01),
    (1.0, 0.1,   0, 8.6913551e+01), (1.0, 0.1,   1, 7.7920288e+01), (1.0, 0.1,   2, 2.0157164e+01),
    (1.0, 1.0,   0, 1.2711158e+02), (1.0, 1.0,   1, 1.9291873e+02), (1.0, 1.0,   2, 4.3221989e+01),
    (1.0, 30.0,  0, 7.0371452e+01), (1.0, 30.0,  1, 4.0881894e+02), (1.0, 30.0,  2, 9.7517532e+01),
    (1.0, 200.0, 0, 7.0371452e+01), (1.0, 200.0, 1, 4.0881894e+02), (1.0, 200.0, 2, 9.7517532e+01),
    (3.0, 0.1,   0, 8.7911255e+01), (3.0, 0.1,   1, 5.2003807e+01), (3.0, 0.1,   2, 1.2698052e+01),
    (3.0, 1.0,   0, 2.0253583e+02), (3.0, 1.0,   1, 1.2875351e+02), (3.0, 1.0,   2, 2.7227802e+01),
    (3.0, 30.0,  0, 1.6329150e+02), (3.0, 30.0,  1, 2.7284497e+02), (3.0, 30.0,  2, 6.1431396e+01),
    (3.0, 200.0, 0, 1.6329150e+02), (3.0, 200.0, 1, 2.7284497e+02), (3.0, 200.0, 2, 6.1431396e+01),
    (8.0, 0.1,   0, 2.4652798e+01), (8.0, 0.1,   1, 2.3304274e+01), (8.0, 0.1,   2, 5.6526446e+00),
    (8.0, 1.0,   0, 8.5216652e+01), (8.0, 1.0,   1, 5.7697838e+01), (8.0, 1.0,   2, 1.2120680e+01),
    (8.0, 30.0,  0, 9.6102974e+01), (8.0, 30.0,  1, 1.2226897e+02), (8.0, 30.0,  2, 2.7346699e+01),
    (8.0, 200.0, 0, 9.6102974e+01), (8.0, 200.0, 1, 1.2226897e+02), (8.0, 200.0, 2, 2.7346699e+01),
    (20.0, 0.1,   0, 2.6307533e+00), (20.0, 0.1,   1, 9.3154564e+00), (20.0, 0.1,   2, 2.2712996e+00),
    (20.0, 1.0,   0, 1.3284497e+01), (20.0, 1.0,   1, 2.3063650e+01), (20.0, 1.0,   2, 4.8702335e+00),
    (20.0, 30.0,  0, 2.0498335e+01), (20.0, 30.0,  1, 4.8874760e+01), (20.0, 30.0,  2, 1.0988228e+01),
    (20.0, 200.0, 0, 2.0498335e+01), (20.0, 200.0, 1, 4.8874760e+01), (20.0, 200.0, 2, 1.0988228e+01),
];

fn channel_from_idx(cc: u32) -> CsfChannel {
    match cc {
        0 => CsfChannel::A,
        1 => CsfChannel::Rg,
        2 => CsfChannel::Vy,
        _ => unreachable!("only 3 still-image cvvdp channels"),
    }
}

#[test]
fn sensitivity_matches_pycvvdp_v0_5_4() {
    let mut worst_rel = 0.0_f32;
    let mut worst_pt = (0.0_f32, 0.0_f32, 0u32, 0.0_f32, 0.0_f32);
    for &(rho, l_bkg, cc, expected) in CSF_GOLDENS {
        let got = sensitivity_scalar(rho, l_bkg, channel_from_idx(cc));
        let rel = (got - expected).abs() / expected;
        if rel > worst_rel {
            worst_rel = rel;
            worst_pt = (rho, l_bkg, cc, got, expected);
        }
    }
    assert!(
        worst_rel < 1e-3,
        "CSF max relative error vs pycvvdp = {worst_rel}; \
         worst point (rho, l_bkg, cc) = ({}, {}, {}), got {}, expected {}",
        worst_pt.0,
        worst_pt.1,
        worst_pt.2,
        worst_pt.3,
        worst_pt.4
    );
}

#[test]
fn precomputed_band_weights_match_pointwise() {
    // Compose the helper from its underlying primitives; this is a
    // sanity check that the helper does what its docstring claims.
    let ppd = DisplayGeometry::STANDARD_4K.pixels_per_degree();
    let l_bkg = 100.0;
    let weights = precomputed_band_weights(ppd, 256, 256, l_bkg);
    let freqs = cvvdp_gpu::kernels::pyramid::band_frequencies(ppd, 256, 256);
    let correction = 10.0_f32.powf(cvvdp_gpu::kernels::csf::SENSITIVITY_CORRECTION_DB / 20.0);

    assert_eq!(weights.len(), freqs.len());
    for (i, &rho) in freqs.iter().enumerate() {
        let exp_a = sensitivity_scalar(rho, l_bkg, CsfChannel::A) * correction;
        let exp_rg = sensitivity_scalar(rho, l_bkg, CsfChannel::Rg) * correction;
        let exp_vy = sensitivity_scalar(rho, l_bkg, CsfChannel::Vy) * correction;
        let [a, rg, vy] = weights[i];
        for (got, exp, tag) in [(a, exp_a, "A"), (rg, exp_rg, "Rg"), (vy, exp_vy, "Vy")] {
            let rel = ((got - exp) / exp).abs();
            assert!(rel < 1e-6, "level {i} {tag}: got {got}, expected {exp}");
        }
    }
}

#[test]
fn flatten_band_weights_layout() {
    let weights = vec![[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let flat = flatten_band_weights(&weights);
    assert_eq!(flat, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn sensitivity_is_finite_at_extremes() {
    // Out-of-table queries get clamped to the endpoints rather than
    // extrapolating to NaN/Inf. Catches a regression where someone
    // swaps the clamp branch for an unbounded extrapolation.
    let s_low = sensitivity_scalar(0.001, 0.0001, CsfChannel::A);
    let s_high = sensitivity_scalar(1000.0, 1.0e6, CsfChannel::A);
    assert!(s_low.is_finite() && s_low > 0.0);
    assert!(s_high.is_finite() && s_high > 0.0);
}
