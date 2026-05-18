//! Smoke tests for the umbrella crate. Each test instantiates a
//! `Metric` for one of the six variants, runs a single identical-image
//! score, and verifies the returned `Score` is finite and reports the
//! expected `metric_name`.
//!
//! These tests are GPU-dependent — they require the `cuda` feature
//! (default) and a working CUDA runtime. On CI runners without a GPU
//! the test will fail at construction with a `BackendNotEnabled` or
//! `Metric` error and surface that diagnostic rather than silently
//! skipping (per CLAUDE.md "NO GRACEFUL SKIPS").

#![cfg(feature = "cuda")]

use zenmetrics_api::{Backend, Metric, MetricKind, MetricParams};

/// Test image dims chosen to clear every metric's minimum:
/// - cvvdp / dssim accept >= 8×8
/// - ssim2 accepts >= 8×8
/// - iwssim REQUIRES min(w,h) >= 176 (5-level pyramid + 11×11 valid
///   blur)
/// - butteraugli & zensim accept tiny images
const W: u32 = 256;
const H: u32 = 256;

fn identity_inputs() -> (Vec<u8>, Vec<u8>) {
    let n = (W as usize) * (H as usize) * 3;
    // Use a non-uniform pattern so metrics that special-case constant
    // images don't trivially return their "perfect" sentinel.
    let mut r = vec![0u8; n];
    for (i, b) in r.iter_mut().enumerate() {
        *b = ((i * 7919) & 0xFF) as u8;
    }
    let d = r.clone();
    (r, d)
}

fn score_identity(kind: MetricKind) -> zenmetrics_api::Score {
    let params = MetricParams::default_for(kind);
    let mut m = Metric::new(kind, Backend::Cuda, W, H, params)
        .unwrap_or_else(|e| panic!("Metric::new({kind:?}, Cuda, {W}x{H}) failed: {e}"));
    assert_eq!(m.kind(), kind, "kind() must roundtrip");
    assert_eq!(m.dims(), (W, H), "dims() must echo constructor args");
    let (r, d) = identity_inputs();
    m.compute_srgb_u8(&r, &d)
        .unwrap_or_else(|e| panic!("compute_srgb_u8 for {kind:?} failed: {e}"))
}

#[cfg(feature = "cvvdp")]
#[test]
fn dispatch_cvvdp() {
    let s = score_identity(MetricKind::Cvvdp);
    assert_eq!(s.metric_name, "cvvdp");
    assert!(s.value.is_finite(), "cvvdp identity score should be finite, got {}", s.value);
    // Identical inputs → JOD == 10 (cvvdp's "no distortion").
    assert!(
        (s.value - 10.0).abs() < 1e-3,
        "cvvdp identity must be ~10, got {}",
        s.value
    );
}

#[cfg(feature = "butter")]
#[test]
fn dispatch_butter() {
    let s = score_identity(MetricKind::Butter);
    assert_eq!(s.metric_name, "butter");
    assert!(s.value.is_finite(), "butteraugli identity score should be finite, got {}", s.value);
    // Identical inputs → max-norm score ~0 (small non-zero due to f32
    // arithmetic on the bundled multiplier).
    assert!(s.value < 1e-2, "butter identity must be ~0, got {}", s.value);
}

#[cfg(feature = "ssim2")]
#[test]
fn dispatch_ssim2() {
    let s = score_identity(MetricKind::Ssim2);
    assert_eq!(s.metric_name, "ssim2");
    assert!(s.value.is_finite(), "ssim2 identity score should be finite, got {}", s.value);
    // SSIMULACRA2 returns ~100 for identical inputs.
    assert!(
        (s.value - 100.0).abs() < 1e-1,
        "ssim2 identity must be ~100, got {}",
        s.value
    );
}

#[cfg(feature = "dssim")]
#[test]
fn dispatch_dssim() {
    let s = score_identity(MetricKind::Dssim);
    assert_eq!(s.metric_name, "dssim");
    assert!(s.value.is_finite(), "dssim identity score should be finite, got {}", s.value);
    // DSSIM is 0 for identical inputs.
    assert!(s.value < 1e-4, "dssim identity must be ~0, got {}", s.value);
}

#[cfg(feature = "iwssim")]
#[test]
fn dispatch_iwssim() {
    // IW-SSIM requires min(w,h) >= 176 — bump the test image to 256
    // (already W,H so it passes).
    //
    // The per-scale information-weighting Σ(cs·iw)/Σ(iw) is 0/0 on
    // truly identical pairs; iwssim-gpu's pipeline detects the
    // degenerate slot and collapses it to the perfect-score value
    // (1.0) so the final score is well-defined. See
    // `iwssim_gpu::pipeline::run_pipeline_post_pyramid` and the
    // `compute_on_identical_returns_1` test in iwssim-gpu's opaque
    // suite.
    let s = score_identity(MetricKind::Iwssim);
    assert_eq!(s.metric_name, "iwssim");
    assert!(s.value.is_finite(), "iwssim identity score must be finite, got {}", s.value);
    // The per-scale ratio collapses to 1.0 exactly only when the CS
    // map is identically 1 — that requires a smooth input where the
    // pyramid stages preserve the σ₁² == σ₁σ₂ == σ₂² invariant. The
    // pseudo-random input above triggers the same degenerate slot
    // but accumulates f32 noise in the IW-weighted ratio
    // (Σ(cs·iw)/Σ(iw) with cs ≈ 1 ± f32-eps). 1e-6 covers that
    // noise band while still catching real regressions (the prior
    // bug returned 0 or NaN, both >>1e-6 from 1.0).
    assert!(
        (s.value - 1.0).abs() < 1e-6,
        "iwssim identity must be ~1.0 within f32 noise (degenerate weighting collapsed by pipeline), got {}",
        s.value
    );

    // Also verify a spatially-structured non-identical pair returns a
    // finite score in [0, 1]. Random byte patterns can degenerate
    // IW-SSIM's information-weighting (the weighted log-sums explode
    // / underflow on near-noise inputs); use a smooth ramp + per-row
    // perturbation matching iwssim-gpu's own opaque test pattern.
    let n = (W as usize) * (H as usize) * 3;
    let mut r = Vec::with_capacity(n);
    let mut d = Vec::with_capacity(n);
    for y in 0..H {
        for x in 0..W {
            let rr = (x & 0xff) as u8;
            let rg = (y & 0xff) as u8;
            let rb = ((x ^ y) & 0xff) as u8;
            r.extend_from_slice(&[rr, rg, rb]);
            let dr = ((x.wrapping_add(7)) & 0xff) as u8;
            let dg = ((y.wrapping_add(21)) & 0xff) as u8;
            let db = ((x ^ y ^ 7) & 0xff) as u8;
            d.extend_from_slice(&[dr, dg, db]);
        }
    }
    let mut m = Metric::new(
        MetricKind::Iwssim,
        Backend::Cuda,
        W,
        H,
        MetricParams::default_for(MetricKind::Iwssim),
    )
    .expect("Metric::new(Iwssim) failed");
    let s2 = m.compute_srgb_u8(&r, &d).expect("iwssim non-identity compute_srgb_u8");
    assert!(
        s2.value.is_finite() && (0.0..=1.0).contains(&s2.value),
        "iwssim non-identical score must be finite in [0,1], got {}",
        s2.value
    );
}

#[cfg(feature = "zensim")]
#[test]
fn dispatch_zensim() {
    // The umbrella's MetricParams::default_for(Zensim) bakes in the
    // canonical WEIGHTS_PREVIEW_V0_2 weights, so the score must be
    // finite. identity_inputs() returns the SAME bytes for ref and
    // dist (let d = r.clone()) — all difference features collapse to
    // ~0, so the per-scale raw distance is ~0 and score_from_features
    // returns ~100 (the perfect-similarity sentinel for the basic-
    // regime linear score). Allow a small f32-noise band around 100.
    //
    // Tolerance note: the CPU `zensim` crate short-circuits identical
    // inputs to all-zero features (and thus score == 100.0 exactly),
    // but zensim-gpu has no such short-circuit and runs the full f32
    // SSIM / blur / max-pool kernel on byte-equal inputs. That picks
    // up sub-ULP rounding at the coarsest pyramid scales (peak-pooled
    // SSIM `sd`, `artifact`, `detail_lost` and `hf_mag_loss` powf(0.125)
    // accumulators), producing ~0.2 score drift on identity. This is
    // f32-precision, not algorithmic divergence; see
    // `crates/zensim-gpu/tests/cpu_parity.rs::identical_input_all_zeros`
    // which already documents the same behaviour with `max_abs < 5e-2`
    // per-feature (which weights up to ~1.0 in score-domain). Using
    // `< 1.0` here keeps regression coverage (NaN / channel-swap / zeroed
    // weights would all far exceed it) while not re-investigating an
    // already-documented f32-noise band. See investigation memory
    // `zensim_gpu_identity_drift_investigation_2026-05-19.md`.
    let s = score_identity(MetricKind::Zensim);
    assert_eq!(s.metric_name, "zensim");
    assert!(
        s.value.is_finite(),
        "zensim default-weights identity score must be finite, got {}",
        s.value
    );
    assert!(
        (s.value - 100.0).abs() < 1.0,
        "zensim default-weights identity score must be ~100 within f32 noise (no distortion), got {}",
        s.value
    );
}

/// MetricKind roundtrip: constructed metric reports the same kind back.
#[test]
fn kind_roundtrip() {
    for kind in enabled_metrics() {
        let params = MetricParams::default_for(kind);
        let m = Metric::new(kind, Backend::Cuda, W, H, params)
            .unwrap_or_else(|e| panic!("Metric::new({kind:?}) failed: {e}"));
        assert_eq!(m.kind(), kind);
        assert_eq!(m.dims(), (W, H));
    }
}

fn enabled_metrics() -> Vec<MetricKind> {
    let mut v = Vec::new();
    #[cfg(feature = "cvvdp")]
    v.push(MetricKind::Cvvdp);
    #[cfg(feature = "butter")]
    v.push(MetricKind::Butter);
    #[cfg(feature = "ssim2")]
    v.push(MetricKind::Ssim2);
    #[cfg(feature = "dssim")]
    v.push(MetricKind::Dssim);
    #[cfg(feature = "iwssim")]
    v.push(MetricKind::Iwssim);
    #[cfg(feature = "zensim")]
    v.push(MetricKind::Zensim);
    v
}
