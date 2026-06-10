//! `hdr::HdrScorer` — HDR-aware multi-score over the umbrella. The caller hands
//! over absolute-luminance linear-RGB (cd/m²); the scorer auto-applies the
//! validated per-metric feeding (pu-rescale u8 for the SSIM-family; display-
//! relative linear planes for butteraugli) and returns lossless `Scores`.
//! CUDA-gated; NO GRACEFUL SKIPS.
#![cfg(all(
    feature = "cuda",
    feature = "hdr",
    feature = "butter",
    feature = "ssim2",
    feature = "zensim"
))]

use zenmetrics_api::hdr::{HDR_PEAK_NITS, HdrScorer};
use zenmetrics_api::{Backend, MetricKind};

/// Interleaved absolute-luminance linear-RGB (cd/m²): a smooth HDR gradient
/// (50..650 cd/m²) and a uniformly 10%-darker distorted copy.
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
fn butter_hdr_scorer_linear_max_and_pnorm3() {
    let (w, h) = (256u32, 256u32);
    let (r, d) = hdr_pair(w, h);
    let mut s = HdrScorer::new(MetricKind::Butter, Backend::Cuda, w, h, HDR_PEAK_NITS)
        .expect("HdrScorer::new butter");
    assert_eq!(s.kind(), MetricKind::Butter);
    assert_eq!(s.peak_nits(), HDR_PEAK_NITS);

    let sc = s.compute_multi(&r, &d).expect("compute_multi");
    assert_eq!(sc.metric_name, "butter");
    // butteraugli came back through the umbrella with BOTH norms.
    assert_eq!(sc.scores.len(), 2);
    let max = sc.get("max").unwrap();
    let pnorm3 = sc.get("pnorm_3").unwrap();
    assert!(max.is_finite() && pnorm3.is_finite(), "{max} {pnorm3}");
    assert!(max > 0.0 && pnorm3 > 0.0, "darkened HDR pair must differ");

    // identity → 0 (and the scorer is reusable across pairs).
    let id = s.compute_multi(&r, &r).expect("identity");
    assert!(id.primary() <= 1e-3, "identity max {}", id.primary());
}

#[test]
fn ssim2_hdr_scorer_integrated_pu() {
    let (w, h) = (256u32, 256u32);
    let (r, d) = hdr_pair(w, h);
    let mut s = HdrScorer::new(MetricKind::Ssim2, Backend::Cuda, w, h, HDR_PEAK_NITS)
        .expect("HdrScorer::new ssim2");

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
fn zensim_hdr_scorer_exposes_features() {
    let (w, h) = (256u32, 256u32);
    let (r, d) = hdr_pair(w, h);
    let mut s = HdrScorer::new(MetricKind::Zensim, Backend::Cuda, w, h, HDR_PEAK_NITS)
        .expect("HdrScorer::new zensim");

    let sc = s.compute_multi(&r, &d).expect("compute_multi");
    assert_eq!(sc.metric_name, "zensim");
    // zensim's regime-length feature vector survives the HDR-aware path.
    assert!(
        matches!(sc.features.len(), 228 | 300 | 372),
        "expected a regime-length feature vector, got {}",
        sc.features.len()
    );
    assert!(sc.features.iter().all(|f| f.is_finite()));
    assert!(sc.primary().is_finite());
}
