//! CPU-ssim2 integrated-PU HDR routing (`hdr::HdrFeeding::IntegratedPuNits`):
//! `HdrScorer` on `Backend::Cpu` routes ssim2 through
//! `cpu_dispatch::compute_pu_nits_interleaved` →
//! `fast_ssim2::compute_ssimulacra2_pu_nits` (`hdr-pu` feature — PU21
//! banding_glare in place of the cube-root opsin nonlinearity, absolute-nits
//! f32 in, no u8 round-trip; consumed via the workspace `[patch.crates-io]`
//! pin until a fast-ssim2 release ships the feature). Scoring itself is
//! pure-CPU; the GPU `ssim2` feature is required only because
//! `MetricParams::Ssim2` (which `HdrScorer::new` default-constructs) lives
//! behind it — same constraint as every `Backend::Cpu` construction path.
//! NO graceful skips.
#![cfg(all(feature = "hdr", feature = "cpu-ssim2", feature = "ssim2"))]

use zenmetrics_api::hdr::{HDR_PEAK_NITS, HdrScorer};
use zenmetrics_api::{Backend, MetricKind};

/// Interleaved absolute-luminance linear-RGB (cd/m²): a smooth HDR gradient
/// (50..650 cd/m²) and a uniformly 10%-darker distorted copy — the same pair
/// shape `cpu_zensim_pu.rs` / `hdr_scorer.rs` use.
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

/// Repack interleaved nits as the `LinearRgbImage` the direct entry takes.
fn nits_image(nits: &[f32], w: u32, h: u32) -> fast_ssim2::LinearRgbImage {
    let data: Vec<[f32; 3]> = nits.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
    fast_ssim2::LinearRgbImage::new(data, w as usize, h as usize)
}

/// The umbrella's CPU-ssim2 HDR score equals calling
/// `fast_ssim2::compute_ssimulacra2_pu_nits` directly on the same nits
/// buffers — proving the routing reaches the integrated PU entry and adds
/// nothing on top. Bit-equal: both sides run the identical deterministic
/// pipeline in the same process.
#[test]
fn cpu_ssim2_pu_matches_direct_compute_ssimulacra2_pu_nits() {
    let (w, h) = (128u32, 96u32);
    let (r, d) = hdr_pair(w, h);

    let mut s = HdrScorer::new(MetricKind::Ssim2, Backend::Cpu, w, h, HDR_PEAK_NITS)
        .expect("HdrScorer::new ssim2 on Backend::Cpu");
    let umbrella = s.compute_multi(&r, &d).expect("compute_multi");
    assert_eq!(umbrella.metric_name, "ssim2");

    let direct =
        fast_ssim2::compute_ssimulacra2_pu_nits(nits_image(&r, w, h), nits_image(&d, w, h))
            .expect("direct compute_ssimulacra2_pu_nits");

    assert_eq!(
        umbrella.primary(),
        direct,
        "umbrella IntegratedPuNits routing must be the direct compute_ssimulacra2_pu_nits score"
    );
}

/// Identity at the integrated-PU entry scores exactly 100 (fast-ssim2's own
/// identical-images contract), and a darkened copy scores strictly below it.
#[test]
fn cpu_ssim2_pu_identity_100_and_discriminates() {
    let (w, h) = (128u32, 96u32);
    let (r, d) = hdr_pair(w, h);

    let mut s = HdrScorer::new(MetricKind::Ssim2, Backend::Cpu, w, h, HDR_PEAK_NITS)
        .expect("HdrScorer::new ssim2 on Backend::Cpu");

    let identity = s.compute_multi(&r, &r).expect("identity");
    assert_eq!(
        identity.primary(),
        100.0,
        "identical inputs must score exactly 100"
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
