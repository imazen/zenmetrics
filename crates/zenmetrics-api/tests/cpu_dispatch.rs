//! `Backend::Cpu` (optimized native-CPU) dispatch — task #159 phase 2.
//!
//! Proves the umbrella routes `Backend::Cpu` to the optimized native crate
//! (`fast-ssim2` for ssim2, in-tree `cvvdp` for cvvdp), produces a finite,
//! in-range, *discriminating* score (not a constant), and reports kind/dims
//! correctly. Gated on the wired `cpu-*` features via the Cargo.toml
//! `[[test]] required-features` entry (grows as metrics are wired) — the
//! skip decision lives in the CI→justfile→test chain, not in the test body
//! (NO graceful skips).

use zenmetrics_api::{Backend, Metric, MetricKind, MetricParams};

/// Deterministic `w×h` packed sRGB (`R, G, B, …`) image.
fn img(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 3]) -> Vec<u8> {
    let mut v = Vec::with_capacity((w as usize) * (h as usize) * 3);
    for y in 0..h {
        for x in 0..w {
            v.extend_from_slice(&f(x, y));
        }
    }
    v
}

/// An ssim2 scorer on the optimized native-CPU backend.
fn ssim2_cpu(w: u32, h: u32) -> Metric {
    Metric::new(
        MetricKind::Ssim2,
        Backend::Cpu,
        w,
        h,
        MetricParams::default_for(MetricKind::Ssim2),
    )
    .expect("Backend::Cpu ssim2 must construct when cpu-ssim2 is built")
}

/// A cvvdp scorer on the optimized native-CPU backend.
fn cvvdp_cpu(w: u32, h: u32) -> Metric {
    Metric::new(
        MetricKind::Cvvdp,
        Backend::Cpu,
        w,
        h,
        MetricParams::default_for(MetricKind::Cvvdp),
    )
    .expect("Backend::Cpu cvvdp must construct when cpu-cvvdp is built")
}

/// An iwssim scorer on the optimized native-CPU backend (min side 176).
fn iwssim_cpu(w: u32, h: u32) -> Metric {
    Metric::new(
        MetricKind::Iwssim,
        Backend::Cpu,
        w,
        h,
        MetricParams::default_for(MetricKind::Iwssim),
    )
    .expect("Backend::Cpu iwssim must construct when cpu-iwssim is built")
}

/// A zensim scorer on the optimized native-CPU backend.
fn zensim_cpu(w: u32, h: u32) -> Metric {
    Metric::new(
        MetricKind::Zensim,
        Backend::Cpu,
        w,
        h,
        MetricParams::default_for(MetricKind::Zensim),
    )
    .expect("Backend::Cpu zensim must construct when cpu-zensim is built")
}

/// A dssim scorer on the optimized native-CPU backend.
fn dssim_cpu(w: u32, h: u32) -> Metric {
    Metric::new(
        MetricKind::Dssim,
        Backend::Cpu,
        w,
        h,
        MetricParams::default_for(MetricKind::Dssim),
    )
    .expect("Backend::Cpu dssim must construct when cpu-dssim is built")
}

/// A butteraugli scorer on the optimized native-CPU backend.
fn butter_cpu(w: u32, h: u32) -> Metric {
    Metric::new(
        MetricKind::Butter,
        Backend::Cpu,
        w,
        h,
        MetricParams::default_for(MetricKind::Butter),
    )
    .expect("Backend::Cpu butter must construct when cpu-butter is built")
}

/// `compute_pixels` (zenpixels input, task #159 phase 3) must route the
/// `Backend::Cpu` path correctly. For an `RGB8_SRGB` slice the conversion is
/// a no-op fast path, so `compute_pixels` feeds byte-identical input to
/// `compute_srgb_u8` — the scores must match exactly (no pixels-path drift).
#[cfg(feature = "pixels")]
#[test]
fn cpu_compute_pixels_matches_srgb_u8_rgb8() {
    use zenpixels::{PixelDescriptor, PixelSlice};
    let (w, h) = (64u32, 64u32);
    let ref_bytes = img(w, h, |x, y| {
        [
            x.wrapping_mul(4) as u8,
            y.wrapping_mul(4) as u8,
            (x ^ y).wrapping_mul(3) as u8,
        ]
    });
    let dist_bytes = img(w, h, |x, y| {
        [x.wrapping_mul(4) as u8, 255 - y.wrapping_mul(4) as u8, 64]
    });
    let row = (w as usize) * 3;
    let desc = PixelDescriptor::RGB8_SRGB;
    for kind in [MetricKind::Ssim2, MetricKind::Cvvdp] {
        let r = PixelSlice::new(&ref_bytes, w, h, row, desc).expect("ref slice");
        let d = PixelSlice::new(&dist_bytes, w, h, row, desc).expect("dist slice");
        let mut mp = Metric::new(kind, Backend::Cpu, w, h, MetricParams::default_for(kind))
            .expect("Backend::Cpu metric");
        let via_pixels = mp.compute_pixels(r, d).expect("compute_pixels");
        let mut mb = Metric::new(kind, Backend::Cpu, w, h, MetricParams::default_for(kind))
            .expect("Backend::Cpu metric");
        let via_bytes = mb
            .compute_srgb_u8(&ref_bytes, &dist_bytes)
            .expect("compute_srgb_u8");
        assert_eq!(
            via_pixels.value, via_bytes.value,
            "{kind:?}: compute_pixels must match compute_srgb_u8 on RGB8_SRGB input"
        );
        assert_eq!(via_pixels.metric_name, via_bytes.metric_name);
    }
}

#[test]
fn dssim_cpu_is_finite_and_discriminates() {
    let (w, h) = (64u32, 64u32);
    let reference = img(w, h, |x, y| {
        [
            x.wrapping_mul(4) as u8,
            y.wrapping_mul(4) as u8,
            (x ^ y).wrapping_mul(3) as u8,
        ]
    });
    let mut m = dssim_cpu(w, h);
    let identical = m
        .compute_srgb_u8(&reference, &reference)
        .expect("identical-pair dssim score");
    assert!(
        identical.value.is_finite(),
        "identical dssim not finite: {}",
        identical.value
    );
    assert_eq!(identical.metric_name, "dssim");
    // DSSIM: 0 = identical, higher = worse.
    assert!(
        identical.value < 0.05,
        "identical pair should score ~0, got {}",
        identical.value
    );
    let distorted = img(w, h, |x, y| {
        [
            255 - x.wrapping_mul(4) as u8,
            255 - y.wrapping_mul(4) as u8,
            128,
        ]
    });
    let mut m2 = dssim_cpu(w, h);
    let bad = m2
        .compute_srgb_u8(&reference, &distorted)
        .expect("distorted-pair dssim score");
    assert!(
        bad.value.is_finite(),
        "distorted dssim not finite: {}",
        bad.value
    );
    assert!(
        bad.value - identical.value > 0.01,
        "expected distorted ({}) materially worse than identical ({})",
        bad.value,
        identical.value
    );
}

#[test]
fn butter_cpu_is_finite_and_discriminates() {
    let (w, h) = (64u32, 64u32);
    let reference = img(w, h, |x, y| {
        [
            x.wrapping_mul(4) as u8,
            y.wrapping_mul(4) as u8,
            (x ^ y).wrapping_mul(3) as u8,
        ]
    });
    let mut m = butter_cpu(w, h);
    let identical = m
        .compute_srgb_u8(&reference, &reference)
        .expect("identical-pair butter score");
    assert!(
        identical.value.is_finite(),
        "identical butter not finite: {}",
        identical.value
    );
    assert_eq!(identical.metric_name, "butter");
    // Butteraugli: 0 = identical, higher = worse.
    assert!(
        identical.value < 0.5,
        "identical pair should score ~0, got {}",
        identical.value
    );
    let distorted = img(w, h, |x, y| {
        [
            255 - x.wrapping_mul(4) as u8,
            255 - y.wrapping_mul(4) as u8,
            128,
        ]
    });
    let mut m2 = butter_cpu(w, h);
    let bad = m2
        .compute_srgb_u8(&reference, &distorted)
        .expect("distorted-pair butter score");
    assert!(
        bad.value.is_finite(),
        "distorted butter not finite: {}",
        bad.value
    );
    assert!(
        bad.value - identical.value > 0.1,
        "expected distorted ({}) materially worse than identical ({})",
        bad.value,
        identical.value
    );
}

#[test]
fn dssim_butter_cpu_report_kind_and_dims() {
    let (w, h) = (64u32, 96u32);
    assert_eq!(dssim_cpu(w, h).kind(), MetricKind::Dssim);
    assert_eq!(dssim_cpu(w, h).dims(), (w, h));
    assert_eq!(butter_cpu(w, h).kind(), MetricKind::Butter);
    assert_eq!(butter_cpu(w, h).dims(), (w, h));
}

#[test]
fn iwssim_cpu_is_finite_and_discriminates() {
    // IW-SSIM requires side >= 176; use 256.
    let (w, h) = (256u32, 256u32);
    let reference = img(w, h, |x, y| {
        [
            x.wrapping_mul(4) as u8,
            y.wrapping_mul(4) as u8,
            (x ^ y).wrapping_mul(3) as u8,
        ]
    });
    let mut m = iwssim_cpu(w, h);
    let identical = m
        .compute_srgb_u8(&reference, &reference)
        .expect("identical-pair iwssim score");
    assert!(
        identical.value.is_finite(),
        "identical iwssim not finite: {}",
        identical.value
    );
    assert_eq!(identical.metric_name, "iwssim");
    // IW-SSIM in [0, 1], 1.0 = identical.
    assert!(
        identical.value > 0.9,
        "identical pair should score ~1.0, got {}",
        identical.value
    );
    let distorted = img(w, h, |x, y| {
        [
            255 - x.wrapping_mul(4) as u8,
            255 - y.wrapping_mul(4) as u8,
            128,
        ]
    });
    let mut m2 = iwssim_cpu(w, h);
    let bad = m2
        .compute_srgb_u8(&reference, &distorted)
        .expect("distorted-pair iwssim score");
    assert!(
        bad.value.is_finite(),
        "distorted iwssim not finite: {}",
        bad.value
    );
    assert!(
        identical.value - bad.value > 0.1,
        "expected distorted ({}) materially below identical ({})",
        bad.value,
        identical.value
    );
}

#[test]
fn zensim_cpu_is_finite_and_discriminates() {
    let (w, h) = (256u32, 256u32);
    let reference = img(w, h, |x, y| {
        [
            x.wrapping_mul(4) as u8,
            y.wrapping_mul(4) as u8,
            (x ^ y).wrapping_mul(3) as u8,
        ]
    });
    let mut m = zensim_cpu(w, h);
    let identical = m
        .compute_srgb_u8(&reference, &reference)
        .expect("identical-pair zensim score");
    assert!(
        identical.value.is_finite(),
        "identical zensim not finite: {}",
        identical.value
    );
    assert_eq!(identical.metric_name, "zensim");
    let distorted = img(w, h, |x, y| {
        [
            255 - x.wrapping_mul(4) as u8,
            255 - y.wrapping_mul(4) as u8,
            128,
        ]
    });
    let mut m2 = zensim_cpu(w, h);
    let bad = m2
        .compute_srgb_u8(&reference, &distorted)
        .expect("distorted-pair zensim score");
    assert!(
        bad.value.is_finite(),
        "distorted zensim not finite: {}",
        bad.value
    );
    // zensim's scale/direction isn't asserted here — only that it
    // discriminates (a real distortion changes the score), proving the CPU
    // path runs the metric rather than returning a constant.
    assert!(
        (identical.value - bad.value).abs() > 1e-4,
        "zensim should discriminate identical ({}) vs distorted ({})",
        identical.value,
        bad.value
    );
}

#[test]
fn iwssim_zensim_cpu_report_kind_and_dims() {
    let (w, h) = (256u32, 192u32);
    assert_eq!(iwssim_cpu(w, h).kind(), MetricKind::Iwssim);
    assert_eq!(iwssim_cpu(w, h).dims(), (w, h));
    assert_eq!(zensim_cpu(w, h).kind(), MetricKind::Zensim);
    assert_eq!(zensim_cpu(w, h).dims(), (w, h));
}

#[test]
fn cvvdp_cpu_is_finite_and_discriminates() {
    let (w, h) = (64u32, 64u32);
    let reference = img(w, h, |x, y| {
        [
            x.wrapping_mul(4) as u8,
            y.wrapping_mul(4) as u8,
            (x ^ y).wrapping_mul(3) as u8,
        ]
    });
    let mut m = cvvdp_cpu(w, h);
    let identical = m
        .compute_srgb_u8(&reference, &reference)
        .expect("identical-pair cvvdp score");
    assert!(
        identical.value.is_finite(),
        "identical cvvdp score not finite: {}",
        identical.value
    );
    assert_eq!(identical.metric_name, "cvvdp");
    // CVVDP JOD: 10 = identical (no perceived difference).
    assert!(
        identical.value > 9.0,
        "identical pair should score ~10 JOD, got {}",
        identical.value
    );

    let distorted = img(w, h, |x, y| {
        [
            255 - x.wrapping_mul(4) as u8,
            255 - y.wrapping_mul(4) as u8,
            128,
        ]
    });
    let mut m2 = cvvdp_cpu(w, h);
    let bad = m2
        .compute_srgb_u8(&reference, &distorted)
        .expect("distorted-pair cvvdp score");
    assert!(
        bad.value.is_finite(),
        "distorted cvvdp score not finite: {}",
        bad.value
    );
    // A real distortion must drop the JOD materially below the identical max.
    assert!(
        identical.value - bad.value > 0.5,
        "expected distorted ({}) materially below identical ({})",
        bad.value,
        identical.value
    );
}

#[test]
fn cvvdp_cpu_reports_kind_and_dims() {
    let (w, h) = (80u32, 48u32);
    let m = cvvdp_cpu(w, h);
    assert_eq!(m.kind(), MetricKind::Cvvdp);
    assert_eq!(m.dims(), (w, h));
}

#[test]
fn ssim2_cpu_is_finite_and_discriminates() {
    let (w, h) = (64u32, 64u32);
    let reference = img(w, h, |x, y| {
        [
            x.wrapping_mul(4) as u8,
            y.wrapping_mul(4) as u8,
            (x ^ y).wrapping_mul(3) as u8,
        ]
    });

    // Identical pair → SSIMULACRA2 max (100).
    let mut m = ssim2_cpu(w, h);
    let identical = m
        .compute_srgb_u8(&reference, &reference)
        .expect("identical-pair score");
    assert!(
        identical.value.is_finite(),
        "identical score not finite: {}",
        identical.value
    );
    assert_eq!(identical.metric_name, "ssim2");
    assert!(
        identical.value > 90.0,
        "identical pair should score ~100, got {}",
        identical.value
    );

    // Heavily distorted (channel-inverted) → materially lower. A real
    // distortion scoring far below the max proves the CPU path actually
    // ran the metric rather than returning a constant.
    let distorted = img(w, h, |x, y| {
        [
            255 - x.wrapping_mul(4) as u8,
            255 - y.wrapping_mul(4) as u8,
            128,
        ]
    });
    let mut m2 = ssim2_cpu(w, h);
    let bad = m2
        .compute_srgb_u8(&reference, &distorted)
        .expect("distorted-pair score");
    assert!(
        bad.value.is_finite(),
        "distorted score not finite: {}",
        bad.value
    );
    assert!(
        identical.value - bad.value > 10.0,
        "expected distorted ({}) materially below identical ({})",
        bad.value,
        identical.value
    );
}

#[test]
fn ssim2_cpu_reports_kind_and_dims() {
    let (w, h) = (96u32, 48u32);
    let m = ssim2_cpu(w, h);
    assert_eq!(m.kind(), MetricKind::Ssim2);
    assert_eq!(m.dims(), (w, h));
}

#[test]
fn ssim2_cpu_rejects_wrong_input_size() {
    let mut m = ssim2_cpu(64, 64);
    // Buffers far too short for 64×64×3 → clean Err, never a panic.
    let short = vec![0u8; 12];
    assert!(
        m.compute_srgb_u8(&short, &short).is_err(),
        "wrong-size input must return Err, not panic"
    );
}
