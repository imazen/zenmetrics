//! Parity test for `DisplayGeometry::pixels_per_degree` against
//! pycvvdp's `vvdp_display_geometry.get_ppd()`.
//!
//! 5 configurations spanning realistic viewing-condition variety.
//! Tolerance is 1e-4 absolute on the PPD itself — for a typical
//! standard_4k (PPD ≈ 75.4) that's 1.3 ppm relative, well within
//! f32 noise.
#![allow(clippy::excessive_precision)]

use cvvdp_gpu::params::DisplayGeometry;

// Tick 551: compile-time pin of `DisplayGeometry::STANDARD_4K` and
// `DisplayModel::STANDARD_4K` field values. The v1 R2 manifest
// goldens were captured under this configuration — a silent edit
// to any field would invalidate every parity test that loads the
// goldens, but the change would only surface at test time. Promote
// the field-value runtime asserts (in the existing fn-tests below)
// to compile-time `const _: () = assert!(...)`. Same pattern as
// ticks 522-524, 548-550. Use `to_bits()` for f32 fields because
// `f32::PartialEq::eq` isn't `const fn` in stable Rust yet, but
// `f32::to_bits` is (since 1.83).
const _: () = assert!(
    DisplayGeometry::STANDARD_4K.resolution_w == 3840,
    "STANDARD_4K resolution_w drifted from 3840 (UHD)",
);
const _: () = assert!(
    DisplayGeometry::STANDARD_4K.resolution_h == 2160,
    "STANDARD_4K resolution_h drifted from 2160 (UHD)",
);
const _: () = assert!(
    DisplayGeometry::STANDARD_4K.distance_m.to_bits() == 0.7472_f32.to_bits(),
    "STANDARD_4K distance_m drifted from pycvvdp 0.7472 m",
);
const _: () = assert!(
    DisplayGeometry::STANDARD_4K.diagonal_inches.to_bits() == 30.0_f32.to_bits(),
    "STANDARD_4K diagonal_inches drifted from 30.0",
);

const _: () = {
    use cvvdp_gpu::params::DisplayModel;
    assert!(
        DisplayModel::STANDARD_4K.y_peak.to_bits() == 200.0_f32.to_bits(),
        "DisplayModel::STANDARD_4K.y_peak drifted from cvvdp v0.5.4 200 cd/m²",
    );
    assert!(
        DisplayModel::STANDARD_4K.y_black.to_bits() == 0.2_f32.to_bits(),
        "DisplayModel::STANDARD_4K.y_black drifted from 0.2 cd/m² (= y_peak / 1000 contrast)",
    );
    assert!(
        DisplayModel::STANDARD_4K.y_refl.to_bits() == 0.397_887_36_f32.to_bits(),
        "DisplayModel::STANDARD_4K.y_refl drifted from precomputed 0.397_887_36 (= 250 lux × 0.005 / π)",
    );
};

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

#[test]
fn ppd_is_positive_for_realistic_geometries() {
    // The CSF stage's per-band rho query depends on PPD; a negative
    // or zero PPD would silently zero out every band's spatial-
    // frequency contribution. Pin positivity across realistic-
    // viewing-range geometries.
    use cvvdp_gpu::params::DisplayGeometry;
    let cases = [
        ("phone-handheld", 1080u32, 1920u32, 0.30_f32, 5.5_f32),
        ("tablet-arm-length", 2048, 1536, 0.45, 9.7),
        ("desktop-monitor", 1920, 1080, 0.60, 27.0),
        ("cinema-far", 3840, 2160, 3.50, 60.0),
        ("UHD-living-room", 3840, 2160, 3.00, 65.0),
    ];
    for &(label, w, h, dist, diag) in &cases {
        let g = DisplayGeometry {
            resolution_w: w,
            resolution_h: h,
            distance_m: dist,
            diagonal_inches: diag,
        };
        let ppd = g.pixels_per_degree();
        assert!(
            ppd > 0.0 && ppd.is_finite(),
            "{label}: PPD = {ppd} must be finite + positive",
        );
        // Sanity: realistic PPDs are in [5, 500].
        assert!(
            (5.0..=500.0).contains(&ppd),
            "{label}: PPD = {ppd} out of realistic [5, 500] range",
        );
    }
}

#[test]
fn ppd_is_monotonically_increasing_in_distance() {
    // More viewing distance → smaller angle subtended per pixel →
    // higher PPD. Pin this monotonicity. A refactor that swaps a
    // sign or division order would invert the relationship and
    // silently mis-calibrate the CSF for every viewer position.
    use cvvdp_gpu::params::DisplayGeometry;
    let base = DisplayGeometry::STANDARD_4K;
    let mut prev = 0.0_f32;
    for dist_m in [0.30_f32, 0.50, 0.75, 1.0, 1.5, 2.5, 4.0] {
        let g = DisplayGeometry {
            distance_m: dist_m,
            ..base
        };
        let ppd = g.pixels_per_degree();
        assert!(
            ppd > prev,
            "monotonicity broken at distance_m={dist_m}: ppd={ppd} <= prev={prev}",
        );
        prev = ppd;
    }
}

#[test]
fn ppd_is_monotonically_decreasing_in_diagonal_inches() {
    // Larger physical screen at the same distance → larger angle
    // per pixel → lower PPD. Pin so a refactor that uses height/m
    // instead of width/m in the denominator would surface here.
    use cvvdp_gpu::params::DisplayGeometry;
    let base = DisplayGeometry::STANDARD_4K;
    let mut prev = f32::INFINITY;
    for diag in [10.0_f32, 15.0, 24.0, 30.0, 55.0, 80.0] {
        let g = DisplayGeometry {
            diagonal_inches: diag,
            ..base
        };
        let ppd = g.pixels_per_degree();
        assert!(
            ppd < prev,
            "monotonicity broken at diagonal_inches={diag}: ppd={ppd} >= prev={prev}",
        );
        prev = ppd;
    }
}

#[test]
fn ppd_is_monotonically_increasing_in_resolution_width() {
    // More horizontal pixels at the same physical size + distance
    // → more pixels per degree. Pin so a refactor that ignores
    // resolution and only uses physical size would surface here.
    use cvvdp_gpu::params::DisplayGeometry;
    let base = DisplayGeometry::STANDARD_4K;
    let mut prev = 0.0_f32;
    for w in [1280u32, 1920, 2560, 3840, 5120, 7680] {
        let h = (w as f32 * 9.0 / 16.0) as u32;
        let g = DisplayGeometry {
            resolution_w: w,
            resolution_h: h,
            ..base
        };
        let ppd = g.pixels_per_degree();
        assert!(
            ppd > prev,
            "monotonicity broken at resolution_w={w}: ppd={ppd} <= prev={prev}",
        );
        prev = ppd;
    }
}

#[test]
fn ppd_does_not_panic_on_degenerate_inputs() {
    // Tick 495: pin the stability contract that
    // `DisplayGeometry::pixels_per_degree` is a total function — it
    // doesn't panic on degenerate inputs (zero distance, zero
    // diagonal, zero resolution). It MAY return ±∞ or NaN
    // (mathematically appropriate for division by zero or 0/0), but
    // it must not abort.
    //
    // Callers like `Cvvdp::compute_dkl_jod(ref, dist, ppd)` accept
    // arbitrary ppd inputs and will either succeed with a degraded
    // pyramid level count or surface an error via the dim-check
    // path. A future refactor that adds an upfront
    // `assert!(distance_m > 0)` (or equivalent) to ppd computation
    // would change the contract from "total + degraded output" to
    // "panicking" — surface that change here.
    //
    // Pins (no panic; output is finite-OR-Inf-OR-NaN, never aborts):
    //   - distance_m = 0
    //   - diagonal_inches = 0
    //   - resolution_w = 0
    //   - resolution_h = 0
    //   - all-zero degenerate config
    use cvvdp_gpu::params::DisplayGeometry;
    let base = DisplayGeometry::STANDARD_4K;
    let cases: &[(&str, DisplayGeometry)] = &[
        (
            "zero distance",
            DisplayGeometry {
                distance_m: 0.0,
                ..base
            },
        ),
        (
            "zero diagonal",
            DisplayGeometry {
                diagonal_inches: 0.0,
                ..base
            },
        ),
        (
            "zero resolution_w",
            DisplayGeometry {
                resolution_w: 0,
                ..base
            },
        ),
        (
            "zero resolution_h",
            DisplayGeometry {
                resolution_h: 0,
                ..base
            },
        ),
        (
            "all-zero",
            DisplayGeometry {
                resolution_w: 0,
                resolution_h: 0,
                distance_m: 0.0,
                diagonal_inches: 0.0,
            },
        ),
    ];
    for (label, g) in cases {
        // The contract is: doesn't panic. We don't pin a specific
        // finite/Inf/NaN value because that's implementation-defined
        // and a refactor of the formula could legitimately shift
        // ±0 → ±Inf or vice versa without breaking the
        // "doesn't-panic" guarantee. Calling .is_finite() exercises
        // the result without asserting on it.
        let ppd = g.pixels_per_degree();
        let _ = ppd.is_finite();
        eprintln!("{label}: ppd = {ppd}");
    }
}
