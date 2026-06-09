//! Reference-value tests for [`Eotf`] and [`Primaries`].
//!
//! Locks the EOTF scalar entry points against published spec
//! values (SMPTE ST 2084 PQ, BT.2100 HLG, sRGB / BT.709) and
//! checks that the new per-primaries DKL matrices distinguish
//! BT.2020 / Display P3 from the BT.709 baseline.
//!
//! These tests intentionally exercise the public scalar API only
//! ([`cvvdp_gpu::params`] free functions + the `display_*`
//! scalars in `kernels::color`) so they run without a working
//! GPU and don't pull in cubecl. They DO use the
//! `cubecl-types` feature gate because that's how every other
//! cvvdp-gpu test file is wired in [`Cargo.toml`] — the gate is
//! a build-time switch, not a runtime requirement.

use cvvdp_gpu::kernels::color::{display_byte_to_dkl_scalar, srgb_byte_to_dkl_scalar};
use cvvdp_gpu::params::{
    DisplayModel, Eotf, Primaries, hlg_inverse_oetf_scalar, hlg_system_gamma, pq_eotf_scalar,
    srgb_eotf_scalar,
};

// =================== EOTF reference values ====================

#[test]
fn pq_eotf_matches_smpte_st2084_reference_values() {
    // SMPTE ST 2084 reference points (computed at f64 from the
    // canonical formula). pycvvdp's pq2lin uses the same
    // constants (verified against the upstream source).
    // V=0.5 → 92.25 cd/m² (a frequently-cited spot on the curve).
    // V=0.508... → ~100 cd/m² (the "SDR white" reference).
    // V=1.0 → 10000 cd/m² (PQ-encoded "diffuse white" maximum).
    let pq_05 = pq_eotf_scalar(0.5);
    assert!(
        (pq_05 - 92.246_6).abs() < 0.05,
        "PQ(0.5) = {pq_05}, expected ≈ 92.25 cd/m^2"
    );

    let pq_1 = pq_eotf_scalar(1.0);
    assert!(
        (pq_1 - 10_000.0).abs() < 1.0,
        "PQ(1.0) = {pq_1}, expected ≈ 10000 cd/m^2"
    );

    // V=0 should yield 0 cd/m² (the formula has (im_t - c1) clipped
    // to >=0 in the numerator).
    let pq_0 = pq_eotf_scalar(0.0);
    assert!(pq_0.abs() < 1e-3, "PQ(0.0) = {pq_0}, expected 0");
}

#[test]
fn pq_eotf_is_monotonic_on_0_to_1() {
    let mut prev = -1.0_f32;
    for i in 0..=100 {
        let v = i as f32 / 100.0;
        let l = pq_eotf_scalar(v);
        assert!(
            l > prev,
            "PQ not monotone: PQ({}) = {}, prev = {}",
            v,
            l,
            prev
        );
        prev = l;
    }
}

#[test]
fn hlg_inverse_oetf_matches_bt2100_reference_values() {
    // BT.2100-1 Table 5 reference points:
    // V=0.5 → 1/12 = 0.083333... (lower-branch formula V^2 / 3
    //         at the seam evaluates to 0.0833...; identical because
    //         (0.5)^2/3 = 0.5/6 = 0.0833...)
    // V=1.0 → 1.0 (upper branch upper bound).
    let v05 = hlg_inverse_oetf_scalar(0.5);
    assert!(
        (v05 - 1.0 / 12.0).abs() < 1e-6,
        "HLG(0.5) = {v05}, expected 1/12"
    );

    let v1 = hlg_inverse_oetf_scalar(1.0);
    assert!((v1 - 1.0).abs() < 1e-4, "HLG(1.0) = {v1}, expected 1.0");

    let v0 = hlg_inverse_oetf_scalar(0.0);
    assert!(v0.abs() < 1e-6, "HLG(0.0) = {v0}, expected 0");
}

#[test]
fn hlg_inverse_oetf_is_monotonic_on_0_to_1() {
    let mut prev = -1.0_f32;
    for i in 0..=200 {
        let v = i as f32 / 200.0;
        let l = hlg_inverse_oetf_scalar(v);
        assert!(l >= prev, "HLG not monotone at V={v}");
        prev = l;
    }
}

#[test]
fn hlg_system_gamma_matches_bt2100_for_sub_1000_nit_peak() {
    // For Y_peak <= 1000 cd/m^2 the spec fixes gamma = 1.2.
    assert!((hlg_system_gamma(100.0, 250.0) - 1.2).abs() < 1e-6);
    assert!((hlg_system_gamma(500.0, 250.0) - 1.2).abs() < 1e-6);
    assert!((hlg_system_gamma(1000.0, 250.0) - 1.2).abs() < 1e-6);
}

#[test]
fn hlg_system_gamma_lifts_for_hdr_peak() {
    // 4000 cd/m^2 peak at 5 lux ambient: gamma > 1.2 per WHP369.
    let g = hlg_system_gamma(4000.0, 5.0);
    assert!(
        g > 1.2,
        "HLG system gamma at 4000 nit should exceed 1.2; got {g}"
    );
    assert!(g < 2.0, "HLG system gamma should be bounded; got {g}");
}

#[test]
fn srgb_eotf_matches_iec_61966_2_1_seam() {
    // The branch is at V = 0.04045. Above: ((V+0.055)/1.055)^2.4;
    // below: V / 12.92. Both branches must agree to machine
    // precision at the seam.
    let lhs = 0.040_45_f32 / 12.92;
    let rhs = ((0.040_45_f32 + 0.055) / 1.055).powf(2.4);
    assert!((lhs - rhs).abs() < 1e-6, "sRGB EOTF seam discontinuity");
    // Endpoint sanity.
    assert_eq!(srgb_eotf_scalar(0.0), 0.0);
    assert!((srgb_eotf_scalar(1.0) - 1.0).abs() < 1e-6);
}

// =================== Eotf::forward dispatcher =================

#[test]
fn eotf_forward_srgb_white_at_standard_4k_emits_y_peak_plus_refl() {
    let d = DisplayModel::STANDARD_4K;
    let l = Eotf::Srgb.forward(1.0, d.y_peak, d.y_black, d.y_refl);
    // (y_peak - y_black) * 1.0 + y_black + y_refl = y_peak + y_refl
    let expected = d.y_peak + d.y_refl;
    assert!(
        (l - expected).abs() < 1e-3,
        "sRGB(1.0) under STANDARD_4K = {l}, expected {expected}"
    );
}

#[test]
fn eotf_forward_pq_at_v_05_emits_92_nit_plus_ambient() {
    // 1500 cd/m² peak; PQ is absolute so the (y_peak - y_black)
    // multiply does NOT apply. V=0.5 -> ~92.25 cd/m^2 + bias.
    let d = DisplayModel::by_name("standard_hdr_pq").unwrap();
    let l = Eotf::Pq.forward(0.5, d.y_peak, d.y_black, d.y_refl);
    let expected = 92.246_6 + d.y_black + d.y_refl;
    assert!(
        (l - expected).abs() < 0.1,
        "PQ(0.5) under standard_hdr_pq = {l}, expected ≈ {expected}"
    );
}

#[test]
fn eotf_forward_linear_clips_and_offsets() {
    // Linear EOTF: V already in cd/m². Pass 100 → 100 + y_refl
    // (no y_peak multiply since input is absolute).
    let d = DisplayModel::by_name("standard_hdr_linear").unwrap();
    let l = Eotf::Linear.forward(100.0, d.y_peak, d.y_black, d.y_refl);
    let expected = 100.0 + d.y_refl;
    assert!(
        (l - expected).abs() < 1e-3,
        "Linear(100.0) under standard_hdr_linear = {l}, expected {expected}"
    );

    // V above y_peak clips to y_peak.
    let l_clip = Eotf::Linear.forward(20000.0, d.y_peak, d.y_black, d.y_refl);
    assert!(
        l_clip <= d.y_peak + d.y_refl + 1e-3,
        "Linear(20000) clipping failed: {l_clip}"
    );
}

#[test]
fn eotf_forward_bt1886_endpoints_match_y_peak_and_y_black() {
    // BT.1886 is a lifted power-law: L(0) = y_black, L(1) = y_peak.
    let y_peak = 100.0_f32;
    let y_black = 0.1_f32;
    let y_refl = 0.4_f32; // arbitrary ambient
    let l0 = Eotf::Bt1886.forward(0.0, y_peak, y_black, y_refl);
    let l1 = Eotf::Bt1886.forward(1.0, y_peak, y_black, y_refl);
    assert!(
        (l0 - (y_black + y_refl)).abs() < 1e-3,
        "BT.1886(0) = {l0}, expected y_black + y_refl"
    );
    assert!(
        (l1 - (y_peak + y_refl)).abs() < 1e-2,
        "BT.1886(1) = {l1}, expected y_peak + y_refl"
    );
}

#[test]
fn eotf_forward_gamma_matches_simple_power_law() {
    let y_peak = 200.0_f32;
    let y_black = 0.2_f32;
    let y_refl = 0.4_f32;
    let g = 2.2_f32;
    let v = 0.6_f32;
    let l = Eotf::Gamma(g).forward(v, y_peak, y_black, y_refl);
    let expected = (y_peak - y_black) * v.powf(g) + y_black + y_refl;
    assert!(
        (l - expected).abs() < 1e-3,
        "Gamma(2.2)({v}) = {l}, expected {expected}"
    );
}

// =================== Primaries → DKL matrices =================

#[test]
fn primaries_bt709_matches_bit_pinned_srgb_matrix() {
    let m = Primaries::Bt709.linear_rgb_to_dkl();
    let s = cvvdp_gpu::params::SRGB_LINEAR_TO_DKL;
    for row in 0..3 {
        for col in 0..3 {
            assert_eq!(
                m[row][col].to_bits(),
                s[row][col].to_bits(),
                "BT.709 dispatch must equal the pinned SRGB_LINEAR_TO_DKL row {row} col {col}"
            );
        }
    }
}

#[test]
fn primaries_bt2020_differs_from_bt709_on_chroma_rows() {
    // A saturated BT.2020 red interpreted through the BT.2020
    // matrix should produce a measurably different DKL coordinate
    // than the same byte values interpreted through BT.709 — the
    // BT.2020 red lobe is more spectrally pure, so it shifts more
    // luminance into the achromatic channel and produces a
    // larger RG response.
    let m709 = Primaries::Bt709.linear_rgb_to_dkl();
    let m2020 = Primaries::Bt2020.linear_rgb_to_dkl();

    // Saturated red, linear values, ignoring display scaling.
    let r = 1.0_f32;
    let g = 0.0_f32;
    let b = 0.0_f32;

    let rg_709 = m709[1][0] * r + m709[1][1] * g + m709[1][2] * b;
    let rg_2020 = m2020[1][0] * r + m2020[1][1] * g + m2020[1][2] * b;
    assert!(
        (rg_709 - rg_2020).abs() > 0.05,
        "BT.709 vs BT.2020 RG row should diverge on saturated red; got 709={rg_709}, 2020={rg_2020}"
    );

    // VY (yellow-violet) row also distinguishable.
    let vy_709 = m709[2][0] * r + m709[2][1] * g + m709[2][2] * b;
    let vy_2020 = m2020[2][0] * r + m2020[2][1] * g + m2020[2][2] * b;
    assert!(
        (vy_709 - vy_2020).abs() > 0.01,
        "BT.709 vs BT.2020 VY row should diverge on saturated red; got 709={vy_709}, 2020={vy_2020}"
    );
}

#[test]
fn primaries_display_p3_differs_from_bt709_on_saturated_red() {
    // Both BT.709 and Display P3 are D65; their A-row sums on
    // equal-energy white are identical to ~6 sig figs because
    // the white point pins the gain. The gamut difference shows
    // up on off-axis colours — a saturated R=1 input maps to a
    // distinguishably different luminance contribution because
    // Display P3's red primary is more spectrally pure than
    // BT.709's. Check the A row entry for the R column directly.
    let m709 = Primaries::Bt709.linear_rgb_to_dkl();
    let mp3 = Primaries::DisplayP3.linear_rgb_to_dkl();
    assert!(
        (m709[0][0] - mp3[0][0]).abs() > 0.01,
        "Display P3 vs BT.709 A[0] (red->luminance) should differ; got 709={}, p3={}",
        m709[0][0],
        mp3[0][0]
    );
    // Chroma rows should also distinguish.
    assert!(
        (m709[1][0] - mp3[1][0]).abs() > 0.01,
        "Display P3 vs BT.709 RG[0] should differ; got 709={}, p3={}",
        m709[1][0],
        mp3[1][0]
    );
}

#[test]
fn primaries_dci_p3_aliases_display_p3() {
    // Today DciP3 returns the same matrix as DisplayP3 (D65 white).
    // Pinned by params.rs docs; surfacing a test makes a future
    // theatrical-DCI variant explicit (it would need its own matrix
    // and the pin here would have to be updated).
    let mp3 = Primaries::DisplayP3.linear_rgb_to_dkl();
    let mdci = Primaries::DciP3.linear_rgb_to_dkl();
    for row in 0..3 {
        for col in 0..3 {
            assert_eq!(
                mp3[row][col].to_bits(),
                mdci[row][col].to_bits(),
                "DciP3 and DisplayP3 must alias today (row {row} col {col})"
            );
        }
    }
}

// ============ display_byte_to_dkl_scalar parity ===============

#[test]
fn display_byte_dkl_under_standard_4k_matches_srgb_path_bit_for_bit() {
    let d = DisplayModel::STANDARD_4K;
    // 8 spot checks covering corners + mid-greys + saturated colours.
    let pixels: [(u8, u8, u8); 8] = [
        (0, 0, 0),
        (255, 255, 255),
        (128, 128, 128),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (200, 50, 100),
        (32, 200, 128),
    ];
    for (r, g, b) in pixels {
        let (a1, rg1, vy1) = srgb_byte_to_dkl_scalar(r, g, b, d.y_peak, d.y_black, d.y_refl);
        let (a2, rg2, vy2) = display_byte_to_dkl_scalar(r, g, b, d);
        assert_eq!(a1.to_bits(), a2.to_bits(), "A drift at ({r}, {g}, {b})");
        assert_eq!(rg1.to_bits(), rg2.to_bits(), "RG drift at ({r}, {g}, {b})");
        assert_eq!(vy1.to_bits(), vy2.to_bits(), "VY drift at ({r}, {g}, {b})");
    }
}

#[test]
fn display_byte_dkl_under_pq_display_does_not_match_srgb_path() {
    // A non-sRGB EOTF (PQ) yields a measurably different DKL
    // coordinate for the same byte triple. (Sanity that we
    // actually dispatch on display.eotf.)
    let d = DisplayModel::by_name("standard_hdr_pq").unwrap();
    let (a_pq, _, _) = display_byte_to_dkl_scalar(128, 128, 128, d);
    let (a_srgb, _, _) = srgb_byte_to_dkl_scalar(128, 128, 128, d.y_peak, d.y_black, d.y_refl);
    // PQ at V≈0.5 → ~92 cd/m² absolute, ignoring (y_peak - y_black)
    // scale. The sRGB path multiplies by (y_peak - y_black) = ~1500
    // for the same display, so the achromatic gets pushed way
    // higher. Difference should be on the order of hundreds of
    // cd/m^2 in the A coordinate.
    assert!(
        (a_pq - a_srgb).abs() > 10.0,
        "PQ vs sRGB dispatch must produce different A; got pq={a_pq}, srgb={a_srgb}"
    );
}

#[test]
fn display_byte_dkl_finite_for_every_eotf_at_v_05() {
    // No EOTF should produce NaN/Inf on a mid-grey byte at any of
    // the public preset displays.
    for name in [
        "standard_4k",
        "standard_hdr_pq",
        "standard_hdr_hlg",
        "standard_hdr_linear",
        "standard_fhd",
        "macbook_pro_16",
    ] {
        let d = DisplayModel::by_name(name).unwrap();
        let (a, rg, vy) = display_byte_to_dkl_scalar(128, 128, 128, d);
        assert!(a.is_finite(), "preset {name} A={a} not finite");
        assert!(rg.is_finite(), "preset {name} RG={rg} not finite");
        assert!(vy.is_finite(), "preset {name} VY={vy} not finite");
    }
}

// ============ DisplayModel::new constructor parity ============

#[test]
fn display_model_new_matches_standard_4k_bit_for_bit() {
    let d = DisplayModel::new(200.0, 1000.0, 250.0, 0.005, Eotf::Srgb, Primaries::Bt709);
    let s = DisplayModel::STANDARD_4K;
    assert_eq!(d.y_peak.to_bits(), s.y_peak.to_bits());
    assert_eq!(d.y_black.to_bits(), s.y_black.to_bits());
    // y_refl comes from compute_y_refl — match within f32 noise of
    // the pinned const value.
    assert!((d.y_refl - s.y_refl).abs() < 1e-6);
    assert_eq!(d.eotf, s.eotf);
    assert_eq!(d.primaries, s.primaries);
    assert_eq!(d.e_ambient_lux, s.e_ambient_lux);
    assert_eq!(d.k_refl, s.k_refl);
}

#[test]
fn compute_y_refl_matches_pi_division() {
    // 250 lux × 0.005 / π = 0.397_887_36 (matches the STANDARD_4K
    // pin).
    let r = DisplayModel::compute_y_refl(250.0, 0.005);
    assert!((r - DisplayModel::STANDARD_4K.y_refl).abs() < 1e-6);
}
