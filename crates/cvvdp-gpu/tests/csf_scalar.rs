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
#![allow(clippy::excessive_precision)]

use cvvdp_gpu::kernels::csf::{
    CsfChannel, LOG_L_BKG_AXIS, N_L_BKG, SENSITIVITY_CORRECTION_DB, flatten_band_weights,
    precompute_logs_row, precomputed_band_weights, sensitivity_corrected_scalar,
    sensitivity_scalar,
};
use cvvdp_gpu::params::DisplayGeometry;

// Tick 557: compile-time bit-pins for the 2 scalar CSF constants
// outside the LUT-axis arrays. Same pattern as ticks 522-524,
// 548-556.
const _: () = {
    use cvvdp_gpu::kernels::csf::CSF_BASEBAND_RHO;
    assert!(
        SENSITIVITY_CORRECTION_DB.to_bits() == (-0.279_742_33_f32).to_bits(),
        "SENSITIVITY_CORRECTION_DB drifted from cvvdp v0.5.4 = -0.279_742_33 dB",
    );
    assert!(
        CSF_BASEBAND_RHO.to_bits() == 0.1_f32.to_bits(),
        "CSF_BASEBAND_RHO drifted from cvvdp v0.5.4 baseband override = 0.1 cy/deg",
    );
    // Tick 568: sign-bit invariants on the CSF scalars. Both are
    // semantic contracts the per-value bit-pins already capture
    // but are worth pinning directly:
    //   - CSF_BASEBAND_RHO > 0: it's a spatial frequency (cy/deg);
    //     negative would be physically meaningless.
    //   - SENSITIVITY_CORRECTION_DB < 0: cvvdp's calibrated peak-
    //     match correction is slightly attenuating; a sign flip
    //     would amplify the CSF instead.
    assert!(
        CSF_BASEBAND_RHO.is_sign_positive(),
        "CSF_BASEBAND_RHO must be positive (spatial frequency in cy/deg)",
    );
    assert!(
        SENSITIVITY_CORRECTION_DB.is_sign_negative(),
        "SENSITIVITY_CORRECTION_DB must be negative (calibrated attenuation, not amplification)",
    );
};

// Goldens: each tuple's L_bkg is documented in linear cd/m²; the
// query is log10(L_bkg) (matching cvvdp's `csf.sensitivity` contract,
// which expects logL_bkg in log10 space, not raw linear). Pre-tick-22
// generation accidentally passed raw L_bkg; tick 22's notes called
// this a 'cvvdp quirk' but the actual pycvvdp pipeline applies the
// log10 inside its weber_contrast_pyr path.
#[rustfmt::skip]
const CSF_GOLDENS: &[(f32, f32, u32, f32)] = &[
    (0.5, 0.1,   0, 2.9781271e+01), (0.5, 0.1,   1, 2.1673922e+01), (0.5, 0.1,   2, 7.1538310e+00),
    (0.5, 1.0,   0, 4.4333603e+01), (0.5, 1.0,   1, 6.5686882e+01), (0.5, 1.0,   2, 1.7471285e+01),
    (0.5, 30.0,  0, 6.7193130e+01), (0.5, 30.0,  1, 2.5991888e+02), (0.5, 30.0,  2, 5.7562096e+01),
    (0.5, 200.0, 0, 7.7058289e+01), (0.5, 200.0, 1, 3.5398764e+02), (0.5, 200.0, 2, 8.2094475e+01),
    (1.0, 0.1,   0, 3.8786003e+01), (1.0, 0.1,   1, 2.3059135e+01), (1.0, 0.1,   2, 7.5551014e+00),
    (1.0, 1.0,   0, 8.1998466e+01), (1.0, 1.0,   1, 6.9885033e+01), (1.0, 1.0,   2, 1.8451275e+01),
    (1.0, 30.0,  0, 1.4168808e+02), (1.0, 30.0,  1, 2.7653070e+02), (1.0, 30.0,  2, 6.0790859e+01),
    (1.0, 200.0, 0, 1.5456792e+02), (1.0, 200.0, 1, 3.7661154e+02), (1.0, 200.0, 2, 8.6699295e+01),
    (3.0, 0.1,   0, 2.1304867e+01), (3.0, 0.1,   1, 1.5389618e+01), (3.0, 0.1,   2, 4.7593536e+00),
    (3.0, 1.0,   0, 7.8535110e+01), (3.0, 1.0,   1, 4.6641109e+01), (3.0, 1.0,   2, 1.1623424e+01),
    (3.0, 30.0,  0, 2.7037811e+02), (3.0, 30.0,  1, 1.8455606e+02), (3.0, 30.0,  2, 3.8295353e+01),
    (3.0, 200.0, 0, 3.4425330e+02), (3.0, 200.0, 1, 2.5134953e+02), (3.0, 200.0, 2, 5.4616428e+01),
    (8.0, 0.1,   0, 3.4639730e+00), (8.0, 0.1,   1, 6.8964891e+00), (8.0, 0.1,   2, 2.1186664e+00),
    (8.0, 1.0,   0, 2.0976240e+01), (8.0, 1.0,   1, 2.0901104e+01), (8.0, 1.0,   2, 5.1742654e+00),
    (8.0, 30.0,  0, 1.3363443e+02), (8.0, 30.0,  1, 8.2704361e+01), (8.0, 30.0,  2, 1.7047497e+01),
    (8.0, 200.0, 0, 1.9532117e+02), (8.0, 200.0, 1, 1.1263638e+02), (8.0, 200.0, 2, 2.4312958e+01),
    (20.0, 0.1,   0, 2.2214642e-01), (20.0, 0.1,   1, 2.7567453e+00), (20.0, 0.1,   2, 8.5130525e-01),
    (20.0, 1.0,   0, 2.1388271e+00), (20.0, 1.0,   1, 8.3548326e+00), (20.0, 1.0,   2, 2.0790808e+00),
    (20.0, 30.0,  0, 2.4213770e+01), (20.0, 30.0,  1, 3.3059555e+01), (20.0, 30.0,  2, 6.8498859e+00),
    (20.0, 200.0, 0, 4.0260078e+01), (20.0, 200.0, 1, 4.5024319e+01), (20.0, 200.0, 2, 9.7692356e+00),
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
        let log_l = l_bkg.log10();
        let got = sensitivity_scalar(rho, log_l, channel_from_idx(cc));
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
    let l_bkg = 100.0_f32;
    let log_l = l_bkg.log10();
    let weights = precomputed_band_weights(ppd, 256, 256, log_l);
    let freqs = cvvdp_gpu::kernels::pyramid::band_frequencies(ppd, 256, 256);
    let correction = 10.0_f32.powf(cvvdp_gpu::kernels::csf::SENSITIVITY_CORRECTION_DB / 20.0);

    assert_eq!(weights.len(), freqs.len());
    for (i, &rho) in freqs.iter().enumerate() {
        let exp_a = sensitivity_scalar(rho, log_l, CsfChannel::A) * correction;
        let exp_rg = sensitivity_scalar(rho, log_l, CsfChannel::Rg) * correction;
        let exp_vy = sensitivity_scalar(rho, log_l, CsfChannel::Vy) * correction;
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

// `sensitivity_corrected_scalar` applies cvvdp v0.5.4's
// `sensitivity_correction` dB-scale tweak on top of the
// uncorrected sensitivity. The production path
// (`precomputed_band_weights`, `csf_apply_*_kernel` host-side
// row-precompute) reads through `sensitivity_corrected_scalar`
// — but the only direct csf_scalar.rs coverage was
// `sensitivity_matches_pycvvdp_v0_5_4` against UNCORRECTED
// values. Pin the correction factor's algebra + invariants so a
// refactor that uses the wrong sign convention (corrections in
// audio dB are typically negative; cvvdp's is -0.28 dB) or
// applies it to log-space when it should be linear (or vice
// versa) trips here directly.

#[test]
fn sensitivity_corrected_applies_constant_multiplicative_factor() {
    // The correction is `10^(DB / 20)` in linear space (voltage-
    // ratio convention). It must be independent of (rho, log_L_bkg,
    // channel) — a constant scalar multiplier. So the ratio
    // corrected/uncorrected must be bit-identical across every
    // input. Sweeps 3 channels × 3 rho × 3 log_l_bkg = 27 points.
    let expected_correction = 10.0_f32.powf(SENSITIVITY_CORRECTION_DB / 20.0);
    let mut max_rel_drift = 0.0_f32;
    for cc in [CsfChannel::A, CsfChannel::Rg, CsfChannel::Vy] {
        for &rho in &[0.5_f32, 4.0, 32.0] {
            for &log_l in &[-2.0_f32, 0.0, 2.0] {
                let s = sensitivity_scalar(rho, log_l, cc);
                let sc = sensitivity_corrected_scalar(rho, log_l, cc);
                let ratio = sc / s.max(1e-9);
                let rel = ((ratio - expected_correction) / expected_correction).abs();
                if rel > max_rel_drift {
                    max_rel_drift = rel;
                }
                assert!(
                    rel < 1e-5,
                    "corrected/uncorrected drift at cc={cc:?} rho={rho} log_l={log_l}: \
                     s={s}, sc={sc}, ratio={ratio}, expected={expected_correction} (rel={rel:.4e})",
                );
            }
        }
    }
    assert!(
        max_rel_drift < 1e-5,
        "max ratio drift across 27 points = {max_rel_drift:.4e}",
    );
}

#[test]
fn sensitivity_correction_is_a_small_attenuation() {
    // cvvdp's `sensitivity_correction` value is a small negative dB
    // (≈ -0.28 dB per the LUT source), so the linear factor is
    // slightly less than 1 (~0.968). Pin both:
    //   - Constant magnitude in [0.9, 1.0) — catches a refactor
    //     that swaps the sign (would give factor > 1) or uses an
    //     order-of-magnitude wrong DB (e.g. -28 dB = 0.04, which
    //     would suppress every band by 25×).
    //   - Specific value: 10^(-0.279_742_33 / 20) ≈ 0.9684.
    let factor = 10.0_f32.powf(SENSITIVITY_CORRECTION_DB / 20.0);
    assert!(
        (0.9..1.0).contains(&factor),
        "correction factor {factor} should be in [0.9, 1.0) for a small attenuation; \
         SENSITIVITY_CORRECTION_DB = {SENSITIVITY_CORRECTION_DB}",
    );
    let expected_factor = 10.0_f32.powf(-0.279_742_33 / 20.0);
    let rel = ((factor - expected_factor) / expected_factor).abs();
    assert!(
        rel < 1e-5,
        "correction factor {factor} ≠ expected {expected_factor} (rel = {rel:.4e}); \
         did SENSITIVITY_CORRECTION_DB change?",
    );
}

#[test]
fn sensitivity_corrected_is_finite_at_extremes() {
    // Same out-of-table clamping contract as `sensitivity_scalar`:
    // extreme inputs must clamp at the LUT endpoints rather than
    // extrapolate to NaN/Inf. Pin separately because a refactor
    // could break either the uncorrected path or the multiplicative
    // correction step independently. The corrected output is
    // still strictly positive (correction factor > 0).
    let s_low = sensitivity_corrected_scalar(0.001, (0.0001_f32).log10(), CsfChannel::A);
    let s_high = sensitivity_corrected_scalar(1000.0, (1.0e6_f32).log10(), CsfChannel::A);
    assert!(s_low.is_finite() && s_low > 0.0, "low extreme: {s_low}");
    assert!(s_high.is_finite() && s_high > 0.0, "high extreme: {s_high}");
}

#[test]
fn sensitivity_is_finite_at_extremes() {
    // Out-of-table queries get clamped to the endpoints rather than
    // extrapolating to NaN/Inf. Catches a regression where someone
    // swaps the clamp branch for an unbounded extrapolation.
    let s_low = sensitivity_scalar(0.001, (0.0001_f32).log10(), CsfChannel::A);
    let s_high = sensitivity_scalar(1000.0, (1.0e6_f32).log10(), CsfChannel::A);
    assert!(s_low.is_finite() && s_low > 0.0);
    assert!(s_high.is_finite() && s_high > 0.0);
}

// `precompute_logs_row` is a public helper that the GPU CSF apply
// kernel consumes per (rho, channel) — it pulls the `log_rho` axis
// out of the 32×32 LUT, leaving a length-N_L_BKG row parameterised
// by `log_L_bkg`. Until this tick it was exercised only by the
// GPU-gated `csf_kernel.rs` tests, which means CPU-only test runs
// (no atomic-f32 GPU, no display, no CUDA toolkit) had zero
// coverage. Add direct unit tests that pin: shape, the identity
// against `sensitivity_scalar` at axis points, frequency
// dependence, channel dependence, and the `rho.max(1e-6)` clamp.
// Same gap-shape as ticks 351/383 closure on lp_norm_sum /
// pool_band_finalize.

#[test]
fn precompute_logs_row_returns_n_l_bkg_entries() {
    // Length is part of the public contract — `csf_apply_per_pixel_kernel`
    // indexes `logs_row[0..N_L_BKG]` directly. A refactor that
    // changes the row size (or accidentally drops the last bin)
    // would corrupt every per-pixel CSF lookup.
    for cc in [CsfChannel::A, CsfChannel::Rg, CsfChannel::Vy] {
        for &rho in &[0.01_f32, 0.5, 4.0, 64.0] {
            let row = precompute_logs_row(rho, cc);
            assert_eq!(
                row.len(),
                N_L_BKG,
                "row length mismatch at rho={rho}, cc={cc:?}: got {}, expected {N_L_BKG}",
                row.len(),
            );
        }
    }
}

#[test]
fn precompute_logs_row_at_axis_points_matches_sensitivity_log10() {
    // Closed-form identity: precompute_logs_row(rho, cc)[k] is the
    // log10 of the un-corrected sensitivity at LOG_L_BKG_AXIS[k].
    // sensitivity_scalar applies interp1_uniform over LOG_L_BKG_AXIS,
    // which returns exactly logs_row[k] at every axis point (interp
    // weight = 1.0). So sensitivity_scalar(rho, axis[k], cc) =
    // 10^precompute_logs_row(rho, cc)[k]. Pin so a refactor that
    // changes either the precompute or the axis-indexed lookup
    // diverges immediately. Sweeps all three channels × four rho
    // values × all 32 axis points = 384 points total.
    for cc in [CsfChannel::A, CsfChannel::Rg, CsfChannel::Vy] {
        for &rho in &[0.1_f32, 1.0, 8.0, 32.0] {
            let row = precompute_logs_row(rho, cc);
            for k in 0..N_L_BKG {
                let log_l_bkg = LOG_L_BKG_AXIS[k];
                let s = sensitivity_scalar(rho, log_l_bkg, cc);
                let expected = 10.0_f32.powf(row[k]);
                let rel = ((s - expected) / expected.abs().max(1e-9)).abs();
                assert!(
                    rel < 1e-4,
                    "precompute/sensitivity divergence at cc={cc:?} rho={rho} k={k} \
                     log_l_bkg={log_l_bkg}: precompute row={} → 10^={}; sensitivity={s} \
                     (rel = {rel:.4e})",
                    row[k],
                    expected,
                );
            }
        }
    }
}

#[test]
fn precompute_logs_row_varies_with_rho() {
    // Different spatial frequencies must yield different rows
    // (otherwise the LUT's rho axis is being collapsed, breaking
    // every per-band CSF lookup). Compare row at rho=0.5 cy/deg
    // vs rho=16 cy/deg — the achromatic CSF peaks near 4 cy/deg
    // and falls off at both extremes, so we expect substantial
    // divergence between these two queries.
    let row_low = precompute_logs_row(0.5, CsfChannel::A);
    let row_high = precompute_logs_row(16.0, CsfChannel::A);
    // Use the max-abs difference across the 32 entries.
    let mut max_diff = 0.0_f32;
    for k in 0..N_L_BKG {
        let d = (row_low[k] - row_high[k]).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    assert!(
        max_diff > 0.1,
        "precompute_logs_row collapses across rho: max |diff| = {max_diff} between 0.5 and 16 cy/deg",
    );
}

#[test]
fn precompute_logs_row_varies_with_channel() {
    // Different opponent channels must yield different rows. The
    // achromatic vs chromatic CSF shapes differ substantially —
    // achromatic peaks much higher than chrominance — so the rows
    // should diverge. Pin so a refactor that points all channels
    // at the same LUT (e.g. a typo in channel_lut dispatch) trips
    // immediately.
    let row_a = precompute_logs_row(4.0, CsfChannel::A);
    let row_rg = precompute_logs_row(4.0, CsfChannel::Rg);
    let row_vy = precompute_logs_row(4.0, CsfChannel::Vy);
    // No two channels should produce identical rows.
    for (label_a, ra, label_b, rb) in [
        ("A", &row_a, "Rg", &row_rg),
        ("A", &row_a, "Vy", &row_vy),
        ("Rg", &row_rg, "Vy", &row_vy),
    ] {
        let mut max_diff = 0.0_f32;
        for k in 0..N_L_BKG {
            let d = (ra[k] - rb[k]).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
        assert!(
            max_diff > 1e-3,
            "{label_a} vs {label_b} channels collapse to same row: max |diff| = {max_diff}",
        );
    }
}

#[test]
fn precompute_logs_row_clamps_rho_at_zero() {
    // `precompute_logs_row` applies `rho.max(1e-6).log10()` — a
    // 0 / negative rho input must not produce -inf or NaN in the
    // log_rho_q computation. Pin so a refactor that drops the
    // clamp (e.g. directly calling `rho.log10()` on an unsafe
    // path) surfaces here before NaN-poisons a downstream band.
    let row0 = precompute_logs_row(0.0, CsfChannel::A);
    let row_neg = precompute_logs_row(-1.0, CsfChannel::A);
    let row_eps = precompute_logs_row(1e-6, CsfChannel::A);
    for k in 0..N_L_BKG {
        assert!(
            row0[k].is_finite(),
            "rho=0 yielded non-finite at k={k}: {}",
            row0[k]
        );
        assert!(
            row_neg[k].is_finite(),
            "rho=-1 yielded non-finite at k={k}: {}",
            row_neg[k]
        );
        // Per the .max(1e-6) clamp, rho=0 and rho=-1 must produce
        // identical output to rho=1e-6.
        assert_eq!(
            row0[k].to_bits(),
            row_eps[k].to_bits(),
            "rho=0 should match rho=1e-6 exactly at k={k} (clamp invariant)",
        );
        assert_eq!(
            row_neg[k].to_bits(),
            row_eps[k].to_bits(),
            "rho=-1 should match rho=1e-6 exactly at k={k} (clamp invariant)",
        );
    }
}

// Pin the exact f32 bit patterns of the cvvdp v0.5.4 CSF
// constants in `kernels::csf`. Same shape as tick 393's pool
// constant pin: a silent edit (typo, sign flip, decimal-point
// shift) cascades into JOD drift across every parity gate. Pin
// each constant by `.to_bits()` so the failure message names the
// specific constant + expected value.
//
// `N_L_BKG` (32) is the LUT axis size; pinning it is the
// length-contract pin from tick 386's
// `precompute_logs_row_returns_n_l_bkg_entries` test. The
// numeric f32 constants here are pinned separately because they
// drift independently of the LUT size.

#[test]
fn csf_constants_match_pycvvdp_v0_5_4() {
    use cvvdp_gpu::kernels::csf::{CSF_BASEBAND_RHO, N_L_BKG, N_RHO, SENSITIVITY_CORRECTION_DB};

    // SENSITIVITY_CORRECTION_DB: cvvdp's published
    // `sensitivity_correction` parameter in dB. The linear-space
    // multiplier `10^(DB / 20)` is applied to every CSF lookup
    // via `sensitivity_corrected_scalar`. Negative dB → linear
    // factor < 1 (small attenuation).
    assert_eq!(
        SENSITIVITY_CORRECTION_DB.to_bits(),
        (-0.279_742_33_f32).to_bits(),
        "SENSITIVITY_CORRECTION_DB = {SENSITIVITY_CORRECTION_DB}, expected -0.279_742_33 (cvvdp v0.5.4)",
    );

    // CSF_BASEBAND_RHO: pycvvdp's `process_block_of_frames`
    // overrides `rho_band[last]` to 0.1 cy/deg for the baseband
    // CSF lookup, separately from the geometric band frequencies.
    // See tick 204 for the chroma_shift parity history.
    assert_eq!(
        CSF_BASEBAND_RHO.to_bits(),
        0.1_f32.to_bits(),
        "CSF_BASEBAND_RHO = {CSF_BASEBAND_RHO}, expected 0.1 (cvvdp v0.5.4 baseband override)",
    );

    // N_L_BKG and N_RHO are the LUT grid sizes (32 × 32). The
    // GPU kernels assume these specific values via array sizing
    // and stride arithmetic; a refactor that bumps either without
    // resizing the kernel buffers would corrupt every per-pixel
    // CSF lookup.
    assert_eq!(
        N_L_BKG, 32,
        "N_L_BKG = {N_L_BKG}, expected 32 (cvvdp v0.5.4 LUT axis)"
    );
    assert_eq!(
        N_RHO, 32,
        "N_RHO = {N_RHO}, expected 32 (cvvdp v0.5.4 LUT axis)"
    );
}
