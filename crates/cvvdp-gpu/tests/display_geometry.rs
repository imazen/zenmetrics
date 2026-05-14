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
