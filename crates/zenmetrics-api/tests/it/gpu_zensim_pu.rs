//! GPU-zensim integrated-PU HDR routing (`hdr::HdrFeeding::IntegratedPuNits`,
//! zenmetrics#25 — the last u8-shell row): `HdrScorer` on a GPU-class backend
//! routes zensim through `Metric::compute_pu_nits_interleaved_multi` →
//! `zensim_gpu::ZensimOpaque::compute_pu_linear_nits_interleaved` (PU21 in
//! place of the SDR cube-root on-device, absolute-nits f32 in, scored with
//! the profile's PU-linear bake B → BHdr, no u8 round-trip). Runs on CUDA or
//! Metal/wgpu. NO graceful skips.
#![cfg(all(
    any(feature = "cuda", feature = "wgpu"),
    feature = "hdr",
    feature = "zensim"
))]

use zenmetrics_api::hdr::{HDR_PEAK_NITS, HdrFeeding, HdrScorer, hdr_feeding};
use zenmetrics_api::{Backend, MetricKind, MetricParams};

#[cfg(feature = "cuda")]
const BACKEND: Backend = Backend::Cuda;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
const BACKEND: Backend = Backend::Wgpu;
#[cfg(feature = "cuda")]
const GPU_BACKEND: zensim_gpu::Backend = zensim_gpu::Backend::Cuda;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
const GPU_BACKEND: zensim_gpu::Backend = zensim_gpu::Backend::Wgpu;

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

/// The umbrella's GPU-zensim HDR score is **bit-equal** to calling the
/// opaque's `compute_pu_linear_nits_interleaved` directly with the umbrella's
/// default params (`MetricParams::default_for(Zensim)`), and the features
/// come through — proving the routing reaches the integrated PU entry and
/// adds nothing on top. Before #25 closed this row the umbrella scored a
/// PU-u8 shell through the SDR bake instead.
#[test]
fn gpu_zensim_pu_matches_direct_opaque_call_and_exposes_features() {
    let (w, h) = (128u32, 96u32);
    let (r, d) = hdr_pair(w, h);
    assert_eq!(
        hdr_feeding(MetricKind::Zensim, BACKEND),
        HdrFeeding::IntegratedPuNits
    );

    let mut s = HdrScorer::new(MetricKind::Zensim, BACKEND, w, h, HDR_PEAK_NITS)
        .expect("HdrScorer::new zensim on a GPU backend");
    let umbrella = s.compute_multi(&r, &d).expect("compute_multi");
    assert_eq!(umbrella.metric_name, "zensim");
    assert_eq!(
        umbrella.features.len(),
        372,
        "canonical profile → WithIw → 372 PU features"
    );
    assert!(umbrella.features.iter().all(|f| f.is_finite()));

    let MetricParams::Zensim(params) = MetricParams::default_for(MetricKind::Zensim) else {
        panic!("default_for(Zensim) must be MetricParams::Zensim");
    };
    let mut direct =
        zensim_gpu::ZensimOpaque::new(GPU_BACKEND, w, h, params).expect("direct opaque");
    let (direct_score, direct_features) = direct
        .compute_pu_linear_nits_interleaved(&r, &d)
        .expect("direct compute_pu_linear_nits_interleaved");
    assert_eq!(
        umbrella.primary(),
        direct_score.value,
        "umbrella IntegratedPuNits routing must be the direct opaque PU score"
    );
    assert_eq!(
        umbrella.features, direct_features,
        "features pass through untouched"
    );
}

/// Identity at the integrated-PU entry scores exactly 100 (the same
/// short-circuit contract as the CPU `mark_identical` path), a darkened
/// copy scores strictly below it, and the scorer is reusable across pairs.
#[test]
fn gpu_zensim_pu_identity_100_and_discriminates() {
    let (w, h) = (128u32, 96u32);
    let (r, d) = hdr_pair(w, h);
    let mut s = HdrScorer::new(MetricKind::Zensim, BACKEND, w, h, HDR_PEAK_NITS)
        .expect("HdrScorer::new zensim on a GPU backend");

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

/// Textured absolute-nits pair for the cross-backend check: log-spaced
/// 0.005..4000 cd/m² with a per-pixel chroma tilt + LCG noise (the
/// zensim-gpu `pu_xyb_parity` generator), distorted = ×0.9. The smooth
/// `hdr_pair` ramp is deliberately NOT used here: on a pure linear gradient
/// zensim's `hf_mag_loss` feature is ill-conditioned (a box mean reproduces
/// a ramp exactly, so Σ|x−μ| is rounding noise on both backends and the
/// GPU/CPU values are arbitrary — 4.3 points apart through the BHdr bake,
/// measured on Metal; the same feature disagrees on the SDR path too, where
/// the B bake just weights it lightly). See zensim-gpu
/// `tests/it/pu_xyb_parity.rs::PU_SCORE_ABS_TOL`.
#[cfg(feature = "cpu-zensim")]
fn textured_nits_pair(w: u32, h: u32) -> (Vec<f32>, Vec<f32>) {
    let (w, h) = (w as usize, h as usize);
    let mut r = Vec::with_capacity(w * h * 3);
    let mut s = 8u32;
    for y in 0..h {
        for x in 0..w {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            let noise = ((s >> 16) & 0xff) as f32 / 255.0;
            let t = ((x as f32 / w.max(1) as f32 + y as f32 / h.max(1) as f32) * 0.5 + noise * 0.1)
                .clamp(0.0, 1.0);
            let yb = 0.005f32 * (4000.0f32 / 0.005).powf(t);
            r.extend_from_slice(&[yb * 1.15, yb, yb * 0.8]);
        }
    }
    let d: Vec<f32> = r.iter().map(|&v| v * 0.9).collect();
    (r, d)
}

/// GPU ↔ CPU: the umbrella's GPU-zensim HDR score tracks the CPU-canonical
/// `zensim::Zensim::compute_pu_linear` on the same textured pair — the two
/// backends now score the SAME metric (PU-XYB features → BHdr bake), where
/// the old u8 shell scored a different regime entirely. The envelope is the
/// zensim-gpu `pu_xyb_parity` gate's (peak-feature f32 drift at HDR
/// magnitudes, ≤ 0.47 measured on Metal; this 128×96 pair: see the
/// eprintln).
#[cfg(feature = "cpu-zensim")]
#[test]
fn gpu_zensim_pu_tracks_cpu_compute_pu_linear() {
    let (w, h) = (128u32, 96u32);
    let (r, d) = textured_nits_pair(w, h);
    let mut s = HdrScorer::new(MetricKind::Zensim, BACKEND, w, h, HDR_PEAK_NITS)
        .expect("HdrScorer::new zensim on a GPU backend");
    let gpu = s.compute_multi(&r, &d).expect("compute_multi").primary();
    let cpu = zensim::Zensim::new(zensim::ZensimProfile::latest_preview())
        .compute_pu_linear(
            &r,
            &d,
            w as usize,
            h as usize,
            3 * w as usize,
            3 * w as usize,
        )
        .expect("cpu compute_pu_linear")
        .score();
    eprintln!(
        "gpu_zensim_pu 128x96: gpu {gpu} cpu {cpu} |Δ| {:.3e}",
        (gpu - cpu).abs()
    );
    assert!(
        (gpu - cpu).abs() < 1.0,
        "GPU integrated-PU {gpu} vs CPU compute_pu_linear {cpu}"
    );
}
