//! `hdr::HdrScorer` on `Backend::Cpu` -- the native-CPU counterpart to
//! `hdr_scorer.rs` (CUDA). Added 2026-07-03: HDR had zero CPU test coverage,
//! which is exactly the gap that let `MetricParams::try_default_for` panic
//! (via `build_hdr_metric`'s `MetricParams::default_for` fallback) on a
//! cpu-only build for ssim2/dssim go undetected. NO GRACEFUL SKIPS.
#![cfg(all(
    feature = "hdr",
    feature = "cpu-butter",
    feature = "cpu-ssim2",
    feature = "cpu-zensim"
))]

use zenmetrics_api::hdr::{HDR_PEAK_NITS, HdrScorer};
use zenmetrics_api::{Backend, MetricKind};

/// Same synthetic HDR pair as `hdr_scorer.rs::hdr_pair` -- kept identical so
/// CPU and GPU results are directly comparable if ever cross-checked.
fn hdr_pair(w: u32, h: u32) -> (Vec<f32>, Vec<f32>) {
    let n = (w * h) as usize;
    let mut r = vec![0.0f32; n * 3];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let v = 50.0 + 600.0 * (x + y) as f32 / (w + h) as f32;
            let i = (y * w as usize + x) * 3;
            r[i] = v;
            r[i + 1] = v;
            r[i + 2] = v;
        }
    }
    let d: Vec<f32> = r.iter().map(|&v| v * 0.9).collect();
    (r, d)
}

#[test]
fn butter_hdr_scorer_cpu_linear_max_and_pnorm3() {
    let (w, h) = (256u32, 256u32);
    let (r, d) = hdr_pair(w, h);
    let mut s = HdrScorer::new(MetricKind::Butter, Backend::Cpu, w, h, HDR_PEAK_NITS)
        .expect("HdrScorer::new butter (cpu)");
    assert_eq!(s.kind(), MetricKind::Butter);
    assert_eq!(s.peak_nits(), HDR_PEAK_NITS);

    let sc = s.compute_multi(&r, &d).expect("compute_multi");
    assert_eq!(sc.metric_name, "butter");
    assert_eq!(sc.scores.len(), 2);
    let max = sc.get("max").unwrap();
    let pnorm3 = sc.get("pnorm_3").unwrap();
    assert!(max.is_finite() && pnorm3.is_finite(), "{max} {pnorm3}");
    assert!(max > 0.0 && pnorm3 > 0.0, "darkened HDR pair must differ");

    let id = s.compute_multi(&r, &r).expect("identity");
    assert!(id.primary() <= 1e-3, "identity max {}", id.primary());
}

#[test]
fn ssim2_hdr_scorer_cpu_integrated_pu() {
    let (w, h) = (256u32, 256u32);
    let (r, d) = hdr_pair(w, h);
    let mut s = HdrScorer::new(MetricKind::Ssim2, Backend::Cpu, w, h, HDR_PEAK_NITS)
        .expect("HdrScorer::new ssim2 (cpu)");

    let identity = s.compute_multi(&r, &r).expect("identity");
    let distorted = s.compute_multi(&r, &d).expect("distorted");
    assert_eq!(identity.metric_name, "ssim2");
    assert!(identity.scores.len() == 1 && identity.features.is_empty());
    assert!(
        identity.primary() >= 99.9,
        "identity {}",
        identity.primary()
    );
    assert!(
        distorted.primary() < identity.primary(),
        "distorted {} should be < identity {}",
        distorted.primary(),
        identity.primary()
    );
}

#[test]
fn zensim_hdr_scorer_cpu_exposes_features() {
    let (w, h) = (256u32, 256u32);
    let (r, d) = hdr_pair(w, h);
    let mut s = HdrScorer::new(MetricKind::Zensim, Backend::Cpu, w, h, HDR_PEAK_NITS)
        .expect("HdrScorer::new zensim (cpu)");

    let sc = s.compute_multi(&r, &d).expect("compute_multi");
    assert_eq!(sc.metric_name, "zensim");
    assert!(
        matches!(sc.features.len(), 228 | 300 | 372),
        "expected a regime-length feature vector, got {}",
        sc.features.len()
    );
    assert!(sc.features.iter().all(|f| f.is_finite()));
    assert!(sc.primary().is_finite());
}

/// dssim has NO HDR path by design (hdr.rs:453 -- external dssim-core
/// transform measured ~0.6 SRCC on UPIQ, deemed not worth an HDR feeding).
/// This is backend-agnostic: the rejection fires in `hdr.rs` before any
/// CPU/GPU dispatch. Asserts the documented rejection, not success.
#[cfg(feature = "cpu-dssim")]
#[test]
fn dssim_hdr_scorer_cpu_rejected_by_design() {
    let (w, h) = (256u32, 256u32);
    let (r, d) = hdr_pair(w, h);
    let mut s = HdrScorer::new(MetricKind::Dssim, Backend::Cpu, w, h, HDR_PEAK_NITS).expect(
        "HdrScorer::new dssim (cpu) -- construction succeeds, rejection is in compute_multi",
    );
    let err = match s.compute_multi(&r, &d) {
        Ok(_) => panic!("dssim HDR must be rejected by design"),
        Err(e) => e,
    };
    assert!(
        err.to_string().contains("no HDR path by design"),
        "unexpected error: {err}"
    );
}

/// APPENDIX AA (measured-nits): `new_with_display_peak` splits the two peak
/// roles.
///
/// 1. The DISPLAY peak drives the luminance-aware linear-planes path: a
///    ~3900-nit highlight pair scored at a *configured* 1000-nit display
///    clips both sides to 1000 before the metric sees them; at the MEASURED
///    content peak nothing clips — the scores must differ. (This is the
///    "actual peak != configured peak" failure the measured parameterization
///    exists to fix.)
/// 2. The u8/PU-shell feedings ignore the display peak: ssim2's
///    integrated-PU score takes absolute nits, so it is bit-identical across
///    display peaks (the shell/regime peak is the constructor's other arg,
///    held fixed).
#[test]
fn display_peak_split_measured_vs_config() {
    let (w, h) = (64u32, 64u32);
    // Bright pair: hdr_pair spans 50..650; ×6 → 300..3900 cd/m², well above
    // the 1000 cd/m² shell constant.
    let (r, d) = {
        let (r, d) = hdr_pair(w, h);
        let r6: Vec<f32> = r.iter().map(|&v| v * 6.0).collect();
        let d6: Vec<f32> = d.iter().map(|&v| v * 6.0).collect();
        (r6, d6)
    };
    let measured_peak = r.iter().fold(0.0f32, |a, &v| a.max(v));
    assert!(measured_peak > 3800.0 && measured_peak < 4000.0);

    // (1) butter linear planes: configured-1000 clips, measured does not.
    let mut cfg =
        HdrScorer::new(MetricKind::Butter, Backend::Cpu, w, h, HDR_PEAK_NITS).expect("cfg butter");
    let mut meas = HdrScorer::new_with_display_peak(
        MetricKind::Butter,
        Backend::Cpu,
        w,
        h,
        HDR_PEAK_NITS,
        measured_peak,
    )
    .expect("measured butter");
    assert_eq!(meas.peak_nits(), HDR_PEAK_NITS);
    assert_eq!(meas.display_peak_nits(), measured_peak);
    let s_cfg = cfg.compute(&r, &d).expect("cfg score").value;
    let s_meas = meas.compute(&r, &d).expect("measured score").value;
    assert!(s_cfg.is_finite() && s_meas.is_finite());
    assert!(
        (s_cfg - s_meas).abs() > 1e-6,
        "configured-1000 (clipping) vs measured-peak butter must differ: {s_cfg} vs {s_meas}"
    );

    // (2) ssim2 integrated PU (absolute nits in): display peak irrelevant —
    // bit-identical across display peaks at a fixed shell peak.
    let mut a =
        HdrScorer::new(MetricKind::Ssim2, Backend::Cpu, w, h, HDR_PEAK_NITS).expect("ssim2 a");
    let mut b = HdrScorer::new_with_display_peak(
        MetricKind::Ssim2,
        Backend::Cpu,
        w,
        h,
        HDR_PEAK_NITS,
        measured_peak,
    )
    .expect("ssim2 b");
    let va = a.compute(&r, &d).expect("ssim2 a score").value;
    let vb = b.compute(&r, &d).expect("ssim2 b score").value;
    assert_eq!(
        va.to_bits(),
        vb.to_bits(),
        "integrated-PU ssim2 must ignore the display peak: {va} vs {vb}"
    );
}
