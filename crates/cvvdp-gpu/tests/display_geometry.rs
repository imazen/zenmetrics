//! Parity test for `DisplayGeometry::pixels_per_degree` against
//! pycvvdp's `vvdp_display_geometry.get_ppd()`.
//!
//! 5 configurations spanning realistic viewing-condition variety.
//! Tolerance is 1e-4 absolute on the PPD itself — for a typical
//! standard_4k (PPD ≈ 75.4) that's 1.3 ppm relative, well within
//! f32 noise.
#![allow(clippy::excessive_precision)]

use cvvdp_gpu::params::DisplayGeometry;

#[test]
fn ppd_matches_pycvvdp_standard_4k() {
    let g = DisplayGeometry::STANDARD_4K;
    let ppd = g.pixels_per_degree();
    let expected = 75.402_449_f32;
    assert!(
        (ppd - expected).abs() < 1e-4,
        "STANDARD_4K PPD: got {ppd}, expected {expected}"
    );
}

#[test]
fn ppd_matches_pycvvdp_for_varied_configs() {
    let cases: &[(&str, DisplayGeometry, f32)] = &[
        (
            "1080p_24in_0.6m",
            DisplayGeometry {
                resolution_w: 1920,
                resolution_h: 1080,
                distance_m: 0.60,
                diagonal_inches: 24.0,
            },
            37.842_504,
        ),
        (
            "1080p_27in_1.0m",
            DisplayGeometry {
                resolution_w: 1920,
                resolution_h: 1080,
                distance_m: 1.00,
                diagonal_inches: 27.0,
            },
            56.062_968,
        ),
        (
            "4k_27in_0.6m",
            DisplayGeometry {
                resolution_w: 3840,
                resolution_h: 2160,
                distance_m: 0.60,
                diagonal_inches: 27.0,
            },
            67.275_562,
        ),
        (
            "phone_5.5in_0.4m",
            DisplayGeometry {
                resolution_w: 1920,
                resolution_h: 1080,
                distance_m: 0.40,
                diagonal_inches: 5.5,
            },
            110.087_282,
        ),
    ];
    for (name, geom, expected) in cases {
        let got = geom.pixels_per_degree();
        assert!(
            (got - expected).abs() < 1e-4,
            "{name}: got {got}, expected {expected}"
        );
    }
}

// Pin the exact f32 bit patterns of the cvvdp v0.5.4 standard_4k
// display constants (`DisplayModel::STANDARD_4K` and
// `DisplayGeometry::STANDARD_4K`). The v1 R2 manifest goldens
// were captured under this display configuration — a silent edit
// to any field would invalidate every parity test that loads the
// goldens. Same shape as ticks 393/394/395 constant pins.

#[test]
fn display_model_standard_4k_matches_pycvvdp_v0_5_4() {
    use cvvdp_gpu::params::DisplayModel;
    let d = DisplayModel::STANDARD_4K;

    // y_peak: 200 cd/m² (peak luminance — cvvdp v0.5.4 default).
    assert_eq!(
        d.y_peak.to_bits(),
        200.0_f32.to_bits(),
        "y_peak = {}, expected 200.0 (cvvdp v0.5.4 standard_4k)",
        d.y_peak,
    );

    // y_black: 0.2 cd/m² (= y_peak / contrast = 200 / 1000).
    assert_eq!(
        d.y_black.to_bits(),
        0.2_f32.to_bits(),
        "y_black = {}, expected 0.2 (= 200 / 1000 contrast)",
        d.y_black,
    );

    // y_refl: 250 lux ambient × 0.005 k_refl / π ≈ 0.397_887_36.
    // Per the docstring: precomputed host-side from
    // `E_ambient / π * k_refl`. The reference value is f32-rounded.
    assert_eq!(
        d.y_refl.to_bits(),
        0.397_887_36_f32.to_bits(),
        "y_refl = {}, expected 0.397_887_36 (250 lux * 0.005 / π)",
        d.y_refl,
    );
}

#[test]
fn display_geometry_standard_4k_matches_pycvvdp_v0_5_4() {
    use cvvdp_gpu::params::DisplayGeometry;
    let g = DisplayGeometry::STANDARD_4K;

    // Resolution: 3840×2160 (UHD). Used to derive PPD via the
    // physical-size pipeline.
    assert_eq!(
        g.resolution_w, 3840,
        "resolution_w = {}, expected 3840",
        g.resolution_w,
    );
    assert_eq!(
        g.resolution_h, 2160,
        "resolution_h = {}, expected 2160",
        g.resolution_h,
    );

    // Viewing distance: 0.7472 m (cvvdp's standard_4k default).
    assert_eq!(
        g.distance_m.to_bits(),
        0.7472_f32.to_bits(),
        "distance_m = {}, expected 0.7472",
        g.distance_m,
    );

    // Diagonal: 30 inches.
    assert_eq!(
        g.diagonal_inches.to_bits(),
        30.0_f32.to_bits(),
        "diagonal_inches = {}, expected 30.0",
        g.diagonal_inches,
    );

    // Derived PPD: ≈ 75.402 (companion to the existing
    // `ppd_matches_pycvvdp_standard_4k` test). If any field above
    // drifts, the derived PPD shifts and every CSF band frequency
    // moves; pin both the inputs and the output.
    let ppd = g.pixels_per_degree();
    assert!(
        (ppd - 75.402_449_f32).abs() < 1e-4,
        "STANDARD_4K PPD = {ppd}, expected ≈ 75.402_449",
    );
}
