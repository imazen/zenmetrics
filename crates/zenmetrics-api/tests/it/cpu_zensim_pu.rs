//! CPU-zensim integrated-PU HDR routing (`hdr::HdrFeeding::IntegratedPuNits`):
//! `HdrScorer` on `Backend::Cpu` routes zensim through
//! `cpu_dispatch::compute_pu_nits_interleaved` → `zensim::Zensim::
//! compute_pu_linear` (zensim PR #44 — PU21 banding_glare in place of the SDR
//! cube-root, absolute-nits f32 in, no u8 round-trip). Scoring itself is
//! pure-CPU; the GPU `zensim` feature is required only because
//! `MetricParams::Zensim` (which `HdrScorer::new` default-constructs) lives
//! behind it — same constraint as every `Backend::Cpu` construction path.
//! NO graceful skips.
#![cfg(all(feature = "hdr", feature = "cpu-zensim", feature = "zensim"))]

use zenmetrics_api::hdr::{HDR_PEAK_NITS, HdrScorer};
use zenmetrics_api::{Backend, MetricKind};

/// Interleaved absolute-luminance linear-RGB (cd/m²): a smooth HDR gradient
/// (50..650 cd/m²) and a uniformly 10%-darker distorted copy — the same pair
/// shape `hdr_scorer.rs` uses for the GPU-side scorers.
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

/// The umbrella's CPU-zensim HDR score equals calling
/// `zensim::Zensim::compute_pu_linear` directly with the same profile the
/// dispatch constructs (`ZensimProfile::latest_preview()`) — proving the
/// routing reaches the integrated PU entry and adds nothing on top.
#[test]
fn cpu_zensim_pu_matches_direct_compute_pu_linear() {
    let (w, h) = (128u32, 96u32);
    let (r, d) = hdr_pair(w, h);

    let mut s = HdrScorer::new(MetricKind::Zensim, Backend::Cpu, w, h, HDR_PEAK_NITS)
        .expect("HdrScorer::new zensim on Backend::Cpu");
    let umbrella = s.compute_multi(&r, &d).expect("compute_multi");
    assert_eq!(umbrella.metric_name, "zensim");

    let z = zensim::Zensim::new(zensim::ZensimProfile::latest_preview());
    let direct = z
        .compute_pu_linear(
            &r,
            &d,
            w as usize,
            h as usize,
            3 * w as usize,
            3 * w as usize,
        )
        .expect("direct compute_pu_linear");

    assert_eq!(
        umbrella.primary(),
        direct.score(),
        "umbrella IntegratedPuNits routing must be the direct compute_pu_linear score"
    );
}

/// Identity at the integrated-PU entry scores exactly 100 (the PR #44
/// `mark_identical` contract), and a darkened copy scores strictly below it.
#[test]
fn cpu_zensim_pu_identity_100_and_discriminates() {
    let (w, h) = (128u32, 96u32);
    let (r, d) = hdr_pair(w, h);

    let mut s = HdrScorer::new(MetricKind::Zensim, Backend::Cpu, w, h, HDR_PEAK_NITS)
        .expect("HdrScorer::new zensim on Backend::Cpu");

    let identity = s.compute_multi(&r, &r).expect("identity");
    assert_eq!(
        identity.primary(),
        100.0,
        "identity must short-circuit to 100"
    );

    let distorted = s.compute_multi(&r, &d).expect("distorted");
    assert!(
        distorted.primary() < identity.primary(),
        "distorted {} must score below identity {}",
        distorted.primary(),
        identity.primary()
    );
    assert!(distorted.primary().is_finite());
}
