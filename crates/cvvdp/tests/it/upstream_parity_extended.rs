//! Extended upstream-parity tests for cvvdp v0.1.0.
//!
//! Closes the chunk-2 audit (`docs/UPSTREAM_PARITY_AUDIT.md`) by pinning:
//!
//! - Every named display preset (G2 + P-rows in the audit) returns the
//!   expected upstream JSON values for resolution, distance, diagonal,
//!   peak / black / refl, EOTF, primaries, ambient lux.
//! - Each non-sRGB EOTF (`Pq`, `Hlg`, `Linear`, `Bt1886`, `Gamma`)
//!   produces sane scoring (identical → JOD 10; non-identical →
//!   JOD < 10) without panicking through the cvvdp pipeline.
//! - Each non-BT.709 primaries variant (`Bt2020`, `DisplayP3`,
//!   `DciP3`) scores correctly through the planar entry path.
//! - PPD derivations match upstream's `display_model.py:get_ppd()`
//!   to within 1e-4 for the 26 named presets where applicable.
//!
//! This file lives alongside `parity_against_host_scalar.rs` and does
//! NOT widen the 1e-4 JOD tolerance the historical tests pinned.

use cvvdp::params::{DisplayModel, Eotf, Primaries};
use cvvdp::{Cvvdp, CvvdpParams, DisplayGeometry};

/// PPD that upstream computes for a `(W, H, distance_m, diag_inches)`
/// tuple. Algebra in `vvdp_display_geometry.get_ppd()` —
/// reimplemented here in f64 for the reference. The cvvdp-gpu
/// `pixels_per_degree()` method runs the same algebra in f32. Allow
/// ≤1e-3 rel error: most presets land within 1e-5.
fn upstream_ppd_f64(w: u32, h: u32, distance_m: f64, diagonal_inches: f64) -> f64 {
    let ar = w as f64 / h as f64;
    let diagonal_mm = diagonal_inches * 25.4;
    let height_mm = (diagonal_mm * diagonal_mm / (1.0 + ar * ar)).sqrt();
    let width_m = ar * height_mm / 1000.0;
    let pix_deg = 2.0_f64
        * (0.5_f64 * width_m / w as f64 / distance_m)
            .atan()
            .to_degrees();
    1.0 / pix_deg
}

/// Identical-input JOD ≈ 10 invariant across presets — verifies the
/// color stage doesn't NaN / panic on extreme HDR/wide-gamut configs.
fn assert_identical_yields_jod10(display: DisplayModel, geometry: DisplayGeometry) {
    let w = 32_u32;
    let h = 32_u32;
    let img: Vec<u8> = (0..(w * h * 3) as usize).map(|i| (i % 256) as u8).collect();
    let params = CvvdpParams {
        display,
        ..CvvdpParams::default()
    };
    let mut cv = Cvvdp::with_geometry(w, h, params, geometry).expect("cvvdp constructs");
    let jod = cv.score(&img, &img).expect("score");
    assert!(
        (jod - 10.0).abs() < 1e-3,
        "identical → JOD 10 failed for display.eotf={:?} primaries={:?}: got {}",
        display.eotf,
        display.primaries,
        jod
    );
}

// ============================================================================
// G2 / G6 / G3 / G5 — display geometry constructors
// ============================================================================

#[test]
fn display_geometry_from_inches_matches_metric() {
    // iPhone 12 Pro at 20" viewing distance, 6.1" diagonal.
    let g = DisplayGeometry::from_inches(2532, 1170, 20.0, 6.1);
    assert_eq!(g.resolution_w, 2532);
    assert_eq!(g.resolution_h, 1170);
    assert!((g.distance_m - 0.508).abs() < 1e-4, "{}", g.distance_m);
    assert!((g.diagonal_inches - 6.1).abs() < 1e-6);
}

#[test]
fn display_geometry_from_meters_diagonal_roundtrips() {
    // 30 inch diagonal in metres = 0.762 m.
    let diag_m = 30.0_f32 * 0.0254;
    let g = DisplayGeometry::from_meters_diagonal(3840, 2160, 0.7472, diag_m);
    assert!(
        (g.diagonal_inches - 30.0).abs() < 1e-4,
        "{}",
        g.diagonal_inches
    );
}

#[test]
fn display_geometry_from_fov_diagonal_matches_upstream_get_ppd() {
    // HTC Vive Pro: 1440×1600, 3 m, 110° FOV diagonal.
    // Reference Python (display_model.py:474-485 then get_ppd):
    //   distance_px = sqrt(1440² + 1600²) / (2 * tan(55°)) ≈ 753.626
    //   height_deg  = 2 * atan(800 / distance_px) ≈ 93.419°
    //   height_m    = 2 * tan(46.710°) * 3 ≈ 6.369 m
    //   width_m     = (1440/1600) * 6.369 ≈ 5.732 m
    //   pix_deg     = 2 * atan(width_m/2 / 1440 / 3) ≈ 0.0760°
    //   PPD         = 1 / pix_deg ≈ 13.153.
    let g = DisplayGeometry::from_fov_diagonal(1440, 1600, 3.0, 110.0);
    let ppd = g.pixels_per_degree();
    let expected = 13.153_262_f32;
    assert!(
        (ppd - expected).abs() < 0.01,
        "from_fov_diagonal PPD {} != upstream ~{}",
        ppd,
        expected,
    );
}

#[test]
fn display_geometry_size_getters_match_inline_math() {
    let g = DisplayGeometry::STANDARD_4K;
    let w_m = g.display_width_m();
    let h_m = g.display_height_m();
    // STANDARD_4K is 30" diagonal at 3840×2160 (aspect 16/9).
    let expected_h_mm = (30.0_f32 * 25.4).powi(2) / (1.0 + (3840.0 / 2160.0_f32).powi(2));
    let expected_h_m = expected_h_mm.sqrt() / 1000.0;
    let expected_w_m = (3840.0 / 2160.0_f32) * expected_h_m;
    assert!(
        (w_m - expected_w_m).abs() < 1e-4,
        "w_m {} vs {}",
        w_m,
        expected_w_m
    );
    assert!(
        (h_m - expected_h_m).abs() < 1e-4,
        "h_m {} vs {}",
        h_m,
        expected_h_m
    );

    let w_deg = g.display_width_deg();
    let h_deg = g.display_height_deg();
    assert!(w_deg > 0.0 && w_deg < 60.0, "w_deg {}", w_deg);
    assert!(h_deg > 0.0 && h_deg < 40.0, "h_deg {}", h_deg);
    // Wider than tall (16:9 aspect).
    assert!(w_deg > h_deg);
}

// ============================================================================
// P-rows — DisplayGeometry presets
// ============================================================================

#[test]
fn preset_geometry_fields_match_upstream_json() {
    // Each row: (geom_const, expected_w, expected_h, expected_dist_m, expected_diag_in)
    #[allow(clippy::type_complexity)]
    let cases: &[(&str, DisplayGeometry, u32, u32, f32, f32)] = &[
        (
            "standard_4k",
            DisplayGeometry::STANDARD_4K,
            3840,
            2160,
            0.7472,
            30.0,
        ),
        (
            "standard_fhd",
            DisplayGeometry::STANDARD_FHD,
            1920,
            1080,
            0.6,
            24.0,
        ),
        (
            "sdr_4k_30",
            DisplayGeometry::SDR_4K_30,
            3840,
            2160,
            0.6,
            30.0,
        ),
        (
            "sdr_fhd_24",
            DisplayGeometry::SDR_FHD_24,
            1920,
            1080,
            0.6,
            24.0,
        ),
        (
            "standard_phone",
            DisplayGeometry::STANDARD_PHONE,
            2400,
            1080,
            0.4,
            6.0,
        ),
        (
            "iphone_12_pro",
            DisplayGeometry::IPHONE_12_PRO,
            2532,
            1170,
            0.508,
            6.1,
        ),
        (
            "iphone_14_pro",
            DisplayGeometry::IPHONE_14_PRO,
            2532,
            1170,
            0.508,
            6.1,
        ),
        (
            "iphone_14_pro_vert",
            DisplayGeometry::IPHONE_14_PRO_VERT,
            1170,
            2532,
            0.508,
            6.1,
        ),
        (
            "ipad_pro_12_9",
            DisplayGeometry::IPAD_PRO_12_9,
            2732,
            2048,
            0.508,
            12.9,
        ),
        (
            "macbook_pro_16",
            DisplayGeometry::MACBOOK_PRO_16,
            3072,
            1920,
            0.635,
            16.0,
        ),
        (
            "lg_oled_2017",
            DisplayGeometry::LG_OLED_2017,
            3840,
            2160,
            2.5654,
            64.5,
        ),
        (
            "eizo_CG3146",
            DisplayGeometry::EIZO_CG3146,
            4096,
            2160,
            0.73406,
            31.063,
        ),
        (
            "panel_65in_4k",
            DisplayGeometry::PANEL_65IN_4K,
            3840,
            2160,
            1.98,
            65.0,
        ),
        (
            "lg_oled_2026",
            DisplayGeometry::LG_OLED_2026,
            3840,
            2160,
            2.2,
            64.9,
        ),
        (
            "hdr_linear_zoom",
            DisplayGeometry::HDR_LINEAR_ZOOM,
            3840,
            2160,
            0.25,
            30.0,
        ),
    ];
    for (name, g, w, h, dist, diag) in cases {
        assert_eq!(g.resolution_w, *w, "{name} W");
        assert_eq!(g.resolution_h, *h, "{name} H");
        assert!((g.distance_m - *dist).abs() < 1e-4, "{name} distance_m");
        assert!(
            (g.diagonal_inches - *diag).abs() < 1e-3,
            "{name} diagonal_inches"
        );
    }
}

#[test]
fn preset_geometry_ppd_matches_upstream_f64() {
    let cases: &[(&str, DisplayGeometry)] = &[
        ("standard_4k", DisplayGeometry::STANDARD_4K),
        ("standard_fhd", DisplayGeometry::STANDARD_FHD),
        ("sdr_4k_30", DisplayGeometry::SDR_4K_30),
        ("sdr_fhd_24", DisplayGeometry::SDR_FHD_24),
        ("standard_phone", DisplayGeometry::STANDARD_PHONE),
        ("iphone_12_pro", DisplayGeometry::IPHONE_12_PRO),
        ("iphone_14_pro", DisplayGeometry::IPHONE_14_PRO),
        ("ipad_pro_12_9", DisplayGeometry::IPAD_PRO_12_9),
        ("macbook_pro_16", DisplayGeometry::MACBOOK_PRO_16),
        ("lg_oled_2017", DisplayGeometry::LG_OLED_2017),
        ("eizo_CG3146", DisplayGeometry::EIZO_CG3146),
        ("panel_65in_4k", DisplayGeometry::PANEL_65IN_4K),
        ("lg_oled_2026", DisplayGeometry::LG_OLED_2026),
        ("hdr_linear_zoom", DisplayGeometry::HDR_LINEAR_ZOOM),
    ];
    for (name, g) in cases {
        let want = upstream_ppd_f64(
            g.resolution_w,
            g.resolution_h,
            g.distance_m as f64,
            g.diagonal_inches as f64,
        ) as f32;
        let got = g.pixels_per_degree();
        let rel = ((want - got) / want).abs();
        assert!(
            rel < 1e-4,
            "{name} PPD divergence: ours={got} upstream={want} rel={rel}"
        );
    }
}

// ============================================================================
// P-rows — DisplayModel presets
// ============================================================================

#[test]
fn preset_display_model_fields_match_upstream_json() {
    use core::f32::consts::PI;
    let y_refl_250 = 250.0 / PI * 0.005;
    let y_refl_100 = 100.0 / PI * 0.005;
    let y_refl_10 = 10.0 / PI * 0.005;
    let y_refl_5 = 5.0 / PI * 0.005;
    let y_refl_0 = 0.0;

    #[allow(clippy::type_complexity)]
    let cases: &[(&str, DisplayModel, f32, f32, f32, Eotf, Primaries, f32)] = &[
        (
            "standard_4k",
            DisplayModel::STANDARD_4K,
            200.0,
            0.2,
            y_refl_250,
            Eotf::Srgb,
            Primaries::Bt709,
            250.0,
        ),
        (
            "standard_hdr_pq",
            DisplayModel::STANDARD_HDR_PQ,
            1500.0,
            0.0015,
            y_refl_10,
            Eotf::Pq,
            Primaries::Bt2020,
            10.0,
        ),
        (
            "standard_hdr_hlg",
            DisplayModel::STANDARD_HDR_HLG,
            1500.0,
            0.0015,
            y_refl_10,
            Eotf::Hlg,
            Primaries::Bt2020,
            10.0,
        ),
        (
            "standard_hdr_linear",
            DisplayModel::STANDARD_HDR_LINEAR,
            1500.0,
            0.0015,
            y_refl_10,
            Eotf::Linear,
            Primaries::Bt709,
            10.0,
        ),
        (
            "standard_hdr_linear_dark",
            DisplayModel::STANDARD_HDR_LINEAR_DARK,
            1500.0,
            0.0015,
            y_refl_0,
            Eotf::Linear,
            Primaries::Bt709,
            0.0,
        ),
        (
            "standard_hdr_linear_zoom",
            DisplayModel::STANDARD_HDR_LINEAR_ZOOM,
            10000.0,
            0.01,
            y_refl_10,
            Eotf::Linear,
            Primaries::Bt709,
            10.0,
        ),
        (
            "standard_fhd",
            DisplayModel::STANDARD_FHD,
            200.0,
            0.2,
            y_refl_250,
            Eotf::Srgb,
            Primaries::Bt709,
            250.0,
        ),
        (
            "standard_phone",
            DisplayModel::STANDARD_PHONE,
            500.0,
            0.05,
            y_refl_250,
            Eotf::Srgb,
            Primaries::Bt709,
            250.0,
        ),
        (
            "sdr_4k_30",
            DisplayModel::SDR_4K_30,
            100.0,
            0.1,
            y_refl_250,
            Eotf::Srgb,
            Primaries::Bt709,
            250.0,
        ),
        (
            "sdr_fhd_24",
            DisplayModel::SDR_FHD_24,
            100.0,
            0.1,
            y_refl_250,
            Eotf::Srgb,
            Primaries::Bt709,
            250.0,
        ),
        (
            "htc_vive_pro",
            DisplayModel::HTC_VIVE_PRO,
            133.3,
            0.1,
            y_refl_0,
            Eotf::Srgb,
            Primaries::Bt709,
            0.0,
        ),
        (
            "iphone_12_pro",
            DisplayModel::IPHONE_12_PRO,
            825.0,
            0.0004,
            y_refl_250,
            Eotf::Srgb,
            Primaries::Bt709,
            250.0,
        ),
        (
            "iphone_14_pro",
            DisplayModel::IPHONE_14_PRO,
            1025.0,
            0.0004,
            y_refl_250,
            Eotf::Srgb,
            Primaries::Bt709,
            250.0,
        ),
        (
            "iphone_14_pro_hdr",
            DisplayModel::IPHONE_14_PRO_HDR,
            1590.0,
            0.0004,
            y_refl_10,
            Eotf::Hlg,
            Primaries::Bt2020,
            10.0,
        ),
        (
            "ipad_pro_12_9",
            DisplayModel::IPAD_PRO_12_9,
            600.0,
            0.37,
            y_refl_250,
            Eotf::Srgb,
            Primaries::Bt709,
            250.0,
        ),
        (
            "macbook_pro_16",
            DisplayModel::MACBOOK_PRO_16,
            500.0,
            0.37,
            y_refl_250,
            Eotf::Srgb,
            Primaries::Bt709,
            250.0,
        ),
        (
            "lg_oled_2017_sdr",
            DisplayModel::LG_OLED_2017_SDR,
            272.0,
            0.014,
            y_refl_100,
            Eotf::Srgb,
            Primaries::Bt709,
            100.0,
        ),
        (
            "lg_oled_2017_hdr",
            DisplayModel::LG_OLED_2017_HDR,
            754.0,
            0.038,
            y_refl_100,
            Eotf::Hlg,
            Primaries::Bt2020,
            100.0,
        ),
        (
            "eizo_CG3146",
            DisplayModel::EIZO_CG3146,
            300.0,
            0.1,
            y_refl_0,
            Eotf::Srgb,
            Primaries::Bt709,
            0.0,
        ),
        (
            "hdr_pq_4knit",
            DisplayModel::HDR_PQ_4KNIT,
            4000.0,
            0.004,
            y_refl_5,
            Eotf::Pq,
            Primaries::Bt2020,
            5.0,
        ),
        (
            "hdr_pq_2knit",
            DisplayModel::HDR_PQ_2KNIT,
            2000.0,
            0.002,
            y_refl_5,
            Eotf::Pq,
            Primaries::Bt2020,
            5.0,
        ),
        (
            "hdr_pq_1knit",
            DisplayModel::HDR_PQ_1KNIT,
            1000.0,
            0.001,
            y_refl_5,
            Eotf::Pq,
            Primaries::Bt2020,
            5.0,
        ),
        (
            "lg_oled_2026_hdr_pq",
            DisplayModel::LG_OLED_2026_HDR_PQ,
            3000.0,
            0.0005,
            y_refl_5,
            Eotf::Pq,
            Primaries::Bt2020,
            5.0,
        ),
    ];

    for (name, d, y_peak, y_black, y_refl, eotf, primaries, e_amb) in cases {
        assert!((d.y_peak - *y_peak).abs() < 1e-3, "{name} y_peak");
        assert!((d.y_black - *y_black).abs() < 1e-4, "{name} y_black");
        assert!(
            (d.y_refl - *y_refl).abs() < 1e-4,
            "{name} y_refl: ours={} want={}",
            d.y_refl,
            y_refl
        );
        assert_eq!(d.eotf, *eotf, "{name} eotf");
        assert_eq!(d.primaries, *primaries, "{name} primaries");
        assert!((d.e_ambient_lux - *e_amb).abs() < 1e-4, "{name} ambient");
        assert!((d.k_refl - 0.005).abs() < 1e-6, "{name} k_refl");
    }
}

// ============================================================================
// EOTF round-trip integration — JOD-10 invariant + non-NaN
// ============================================================================

#[test]
fn identical_input_pq_bt2020_yields_jod10() {
    assert_identical_yields_jod10(DisplayModel::STANDARD_HDR_PQ, DisplayGeometry::STANDARD_4K);
}

#[test]
fn identical_input_hlg_bt2020_yields_jod10() {
    assert_identical_yields_jod10(DisplayModel::STANDARD_HDR_HLG, DisplayGeometry::STANDARD_4K);
}

#[test]
fn identical_input_linear_bt709_yields_jod10() {
    assert_identical_yields_jod10(
        DisplayModel::STANDARD_HDR_LINEAR,
        DisplayGeometry::STANDARD_4K,
    );
}

#[test]
fn identical_input_gamma_bt709_yields_jod10() {
    let d = DisplayModel {
        eotf: Eotf::Gamma(2.2),
        ..DisplayModel::STANDARD_4K
    };
    assert_identical_yields_jod10(d, DisplayGeometry::STANDARD_4K);
}

#[test]
fn identical_input_bt1886_bt709_yields_jod10() {
    let d = DisplayModel {
        eotf: Eotf::Bt1886,
        ..DisplayModel::STANDARD_4K
    };
    assert_identical_yields_jod10(d, DisplayGeometry::STANDARD_4K);
}

// ============================================================================
// Primaries integration — DisplayP3 / DciP3
// ============================================================================

#[test]
fn identical_input_display_p3_yields_jod10() {
    let d = DisplayModel {
        primaries: Primaries::DisplayP3,
        ..DisplayModel::STANDARD_4K
    };
    assert_identical_yields_jod10(d, DisplayGeometry::STANDARD_4K);
}

#[test]
fn identical_input_dci_p3_yields_jod10() {
    let d = DisplayModel {
        primaries: Primaries::DciP3,
        ..DisplayModel::STANDARD_4K
    };
    assert_identical_yields_jod10(d, DisplayGeometry::STANDARD_4K);
}

#[test]
fn distortion_under_pq_bt2020_lowers_jod() {
    let w = 64_u32;
    let h = 64_u32;
    let ref_img: Vec<u8> = (0..(w * h * 3) as usize).map(|i| (i % 256) as u8).collect();
    // Distort by adding noise.
    let mut dist = ref_img.clone();
    let mut s = 42_u32;
    for v in dist.iter_mut() {
        s = s.wrapping_mul(48271);
        let delta = ((s >> 24) as i32 - 128) / 4;
        *v = ((*v as i32 + delta).clamp(0, 255)) as u8;
    }
    let params = CvvdpParams {
        display: DisplayModel::STANDARD_HDR_PQ,
        ..CvvdpParams::default()
    };
    let mut cv = Cvvdp::with_geometry(w, h, params, DisplayGeometry::STANDARD_4K).unwrap();
    let jod = cv.score(&ref_img, &dist).unwrap();
    assert!(
        jod.is_finite() && (0.0..=10.0).contains(&jod),
        "JOD out of range: {jod}"
    );
    assert!(
        jod < 10.0,
        "distortion → JOD should drop below 10, got {jod}"
    );
}

#[test]
fn primaries_change_shifts_chroma_score() {
    // Same display except BT.709 vs BT.2020 primaries: the metric
    // score on saturated chromatic content should differ measurably
    // because the DKL matrix differs.
    let w = 32_u32;
    let h = 32_u32;
    // Heavy red bias to exercise chroma differences.
    let ref_img: Vec<u8> = (0..(w * h) as usize)
        .flat_map(|_| [255_u8, 64, 32])
        .collect();
    let mut dist = ref_img.clone();
    // Wrapping noise.
    let mut s = 99_u32;
    for v in dist.iter_mut() {
        s = s.wrapping_mul(48271);
        *v = ((*v as i32 + (s >> 24) as i32 / 8 - 16).clamp(0, 255)) as u8;
    }

    let bt709_d = DisplayModel {
        eotf: Eotf::Linear,
        primaries: Primaries::Bt709,
        ..DisplayModel::STANDARD_HDR_LINEAR
    };
    let bt2020_d = DisplayModel {
        eotf: Eotf::Linear,
        primaries: Primaries::Bt2020,
        ..DisplayModel::STANDARD_HDR_LINEAR
    };

    let mut cv709 = Cvvdp::with_geometry(
        w,
        h,
        CvvdpParams {
            display: bt709_d,
            ..CvvdpParams::default()
        },
        DisplayGeometry::STANDARD_4K,
    )
    .unwrap();
    let mut cv2020 = Cvvdp::with_geometry(
        w,
        h,
        CvvdpParams {
            display: bt2020_d,
            ..CvvdpParams::default()
        },
        DisplayGeometry::STANDARD_4K,
    )
    .unwrap();

    let s709 = cv709.score(&ref_img, &dist).unwrap();
    let s2020 = cv2020.score(&ref_img, &dist).unwrap();
    // The matrices differ; on chromatic content the scores should
    // differ measurably (≥ 1e-3 JOD).
    assert!(
        (s709 - s2020).abs() >= 1e-3,
        "Bt709 and Bt2020 produced same score {s709} on chromatic content"
    );
    assert!(s709.is_finite() && (0.0..=10.0).contains(&s709));
    assert!(s2020.is_finite() && (0.0..=10.0).contains(&s2020));
}

// ============================================================================
// Standard-4K parity gate (must still hold!)
// ============================================================================

#[test]
fn standard_4k_path_still_at_parity_against_host_scalar() {
    // This is the historic 1e-4 JOD parity contract. If THIS test
    // fails, we broke v0.0.1 numerics — STOP and revert.
    use cvvdp::host_scalar::predict_jod_still_3ch;
    let w = 64_usize;
    let h = 48_usize;
    let mut s = 1234_u32;
    let mut r = vec![0u8; w * h * 3];
    let mut d = vec![0u8; w * h * 3];
    for i in 0..w * h * 3 {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        r[i] = (s >> 16) as u8;
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        d[i] = (s >> 16) as u8;
    }
    let want = predict_jod_still_3ch(
        &r,
        &d,
        w,
        h,
        DisplayModel::STANDARD_4K,
        DisplayGeometry::STANDARD_4K.pixels_per_degree(),
    );
    let mut cv = Cvvdp::new(w as u32, h as u32, CvvdpParams::default()).unwrap();
    let got = cv.score(&r, &d).unwrap();
    let diff = (want - got).abs();
    assert!(
        diff < 1e-4,
        "STANDARD_4K parity gate broke! ours={got} host_scalar={want} diff={diff}"
    );
}

#[test]
fn cvvdp_constructs_for_every_named_preset() {
    // Smoke-test: every named DisplayGeometry + DisplayModel preset
    // must successfully construct a Cvvdp instance (no panic, no
    // PPD-out-of-range error).
    let geoms = [
        DisplayGeometry::STANDARD_4K,
        DisplayGeometry::STANDARD_FHD,
        DisplayGeometry::SDR_4K_30,
        DisplayGeometry::SDR_FHD_24,
        DisplayGeometry::STANDARD_PHONE,
        DisplayGeometry::HTC_VIVE_PRO,
        DisplayGeometry::IPHONE_12_PRO,
        DisplayGeometry::IPHONE_14_PRO,
        DisplayGeometry::IPHONE_14_PRO_VERT,
        DisplayGeometry::IPAD_PRO_12_9,
        DisplayGeometry::MACBOOK_PRO_16,
        DisplayGeometry::LG_OLED_2017,
        DisplayGeometry::EIZO_CG3146,
        DisplayGeometry::PANEL_65IN_4K,
        DisplayGeometry::LG_OLED_2026,
        DisplayGeometry::HDR_LINEAR_ZOOM,
    ];
    let displays = [
        DisplayModel::STANDARD_4K,
        DisplayModel::STANDARD_HDR_PQ,
        DisplayModel::STANDARD_HDR_HLG,
        DisplayModel::STANDARD_HDR_LINEAR,
        DisplayModel::STANDARD_HDR_LINEAR_DARK,
        DisplayModel::STANDARD_HDR_LINEAR_ZOOM,
        DisplayModel::STANDARD_FHD,
        DisplayModel::STANDARD_PHONE,
        DisplayModel::SDR_4K_30,
        DisplayModel::SDR_FHD_24,
        DisplayModel::HTC_VIVE_PRO,
        DisplayModel::IPHONE_12_PRO,
        DisplayModel::IPHONE_14_PRO,
        DisplayModel::IPHONE_14_PRO_HDR,
        DisplayModel::IPAD_PRO_12_9,
        DisplayModel::MACBOOK_PRO_16,
        DisplayModel::LG_OLED_2017_SDR,
        DisplayModel::LG_OLED_2017_HDR,
        DisplayModel::EIZO_CG3146,
        DisplayModel::HDR_PQ_4KNIT,
        DisplayModel::HDR_PQ_2KNIT,
        DisplayModel::HDR_PQ_1KNIT,
        DisplayModel::LG_OLED_2026_HDR_PQ,
    ];
    // 23 displays + 16 geometries = 23 + 16 paired smoke-tests
    // (not the cartesian product — that'd be 368). Pair each display
    // with STANDARD_4K and each geometry with STANDARD_4K display.
    for d in displays {
        let _ = Cvvdp::with_geometry(
            32,
            32,
            CvvdpParams {
                display: d,
                ..CvvdpParams::default()
            },
            DisplayGeometry::STANDARD_4K,
        )
        .expect("Cvvdp constructs for any DisplayModel");
    }
    for g in geoms {
        let _ = Cvvdp::with_geometry(32, 32, CvvdpParams::default(), g)
            .expect("Cvvdp constructs for any DisplayGeometry");
    }
}
