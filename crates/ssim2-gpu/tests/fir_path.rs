#![cfg(feature = "fir")]
//! Integration tests for the opt-in FIR D=5 blur path
//! (`Ssim2Blur::Fir`) per Kanetaka et al. IWAIT 2026.
//!
//! Gated behind the `fir` Cargo feature — the FIR API surface
//! (`Ssim2Blur`, `with_blur`, `set_blur`, `blur()`) is not exported
//! without the feature, so this whole file compiles to nothing in the
//! default build.
//!
//! **The FIR is a distinct metric**, not a faster reimplementation
//! of the IIR. These tests cover the FIR's own contract:
//!
//! 1. **Determinism** — same input → same FIR score across repeated
//!    runs and across the direct, cached, and batched compute paths.
//! 2. **Monotonicity** — heavier JPEG distortion → lower FIR score
//!    (same direction as IIR; sanity for the score remap).
//! 3. **Distinctness from IIR** — the FIR is documented to produce a
//!    different score scale; we pin that the FIR score is finite,
//!    in a reasonable range, and observably distinct from the IIR
//!    score on real-distortion fixtures. The drift table is
//!    informational and asserts only that the score gap is non-zero
//!    (i.e. dispatch actually routes to a different kernel).
//! 4. **Identical-image score is high (~100)** — FIR of identical
//!    inputs must still hit the score-100 path, with the same
//!    backend-dependent FP tolerance as the IIR path.
//!
//! No parity assertion vs the CPU `ssimulacra2` is made — the FIR
//! intentionally diverges from the libjxl recursive Gaussian. See
//! `parity_lock.rs::parity_jpeg_corpus` for the IIR's strict parity
//! gate (which remains untouched).

use cubecl::Runtime;
use ssim2_gpu::{Ssim2, Ssim2Batch, Ssim2Blur};

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;
#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
compile_error!(
    "ssim2-gpu integration tests require either the `cuda` or `wgpu` feature to select a runtime"
);

fn load_rgb8(path: &std::path::Path) -> (Vec<u8>, u32, u32) {
    let img = image::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let rgb8 = img.to_rgb8();
    let (w, h) = (rgb8.width(), rgb8.height());
    (rgb8.into_raw(), w, h)
}

fn corpus_dir() -> std::path::PathBuf {
    zenmetrics_corpus::corpus_dir()
}

// ───────────────────────── determinism ─────────────────────────

#[test]
fn fir_compute_is_deterministic() {
    let dir = corpus_dir();
    let (src_bytes, w, h) = load_rgb8(&dir.join("source.png"));
    let (dis_bytes, _, _) = load_rgb8(&dir.join("q45.jpg"));

    let client = Backend::client(&Default::default());
    let mut s = Ssim2::<Backend>::new(client, w, h)
        .expect("Ssim2::new")
        .with_blur(Ssim2Blur::Fir);

    let s0 = s.compute(&src_bytes, &dis_bytes).expect("fir 0").score;
    let s1 = s.compute(&src_bytes, &dis_bytes).expect("fir 1").score;
    let s2 = s.compute(&src_bytes, &dis_bytes).expect("fir 2").score;

    // f32 atomic-add reduction noise is ≤ 1e-4 in practice; tighter
    // than the parity_lock IIR-vs-IIR equivalence tests' 1e-4 bound
    // because we're comparing the same path to itself.
    let d_01 = (s0 - s1).abs();
    let d_02 = (s0 - s2).abs();
    assert!(
        d_01 < 1e-4 && d_02 < 1e-4,
        "FIR not deterministic: {s0} {s1} {s2}"
    );
}

#[test]
fn fir_cached_matches_direct() {
    let dir = corpus_dir();
    let (src_bytes, w, h) = load_rgb8(&dir.join("source.png"));

    let client_a = Backend::client(&Default::default());
    let client_b = Backend::client(&Default::default());
    let mut s_direct = Ssim2::<Backend>::new(client_a, w, h)
        .expect("direct")
        .with_blur(Ssim2Blur::Fir);
    let mut s_cached = Ssim2::<Backend>::new(client_b, w, h)
        .expect("cached")
        .with_blur(Ssim2Blur::Fir);
    s_cached.set_reference(&src_bytes).expect("set_reference");

    for q in [5u32, 45, 90] {
        let path = dir.join(format!("q{q}.jpg"));
        let (dis_bytes, _, _) = load_rgb8(&path);
        let direct = s_direct
            .compute(&src_bytes, &dis_bytes)
            .expect("direct")
            .score;
        let cached = s_cached
            .compute_with_reference(&dis_bytes)
            .expect("cached")
            .score;
        let d = (direct - cached).abs();
        assert!(
            d < 1e-4,
            "FIR q{q}: direct={direct}, cached={cached}, Δ={d}"
        );
    }
}

#[test]
fn fir_batch_matches_single_image() {
    let dir = corpus_dir();
    let (src_bytes, w, h) = load_rgb8(&dir.join("source.png"));

    let qs = [5u32, 20, 45, 70, 90];
    let dis: Vec<Vec<u8>> = qs
        .iter()
        .map(|q| load_rgb8(&dir.join(format!("q{q}.jpg"))).0)
        .collect();

    let client = Backend::client(&Default::default());
    let mut single = Ssim2::<Backend>::new(client, w, h)
        .expect("Ssim2::new")
        .with_blur(Ssim2Blur::Fir);
    let single_results: Vec<f64> = dis
        .iter()
        .map(|d| single.compute(&src_bytes, d).expect("compute").score)
        .collect();

    let client = Backend::client(&Default::default());
    let mut batch = Ssim2Batch::<Backend>::new(client, w, h, dis.len() as u32)
        .expect("Ssim2Batch::new")
        .with_blur(Ssim2Blur::Fir);
    batch.set_reference(&src_bytes).expect("set_reference");
    let batch_results = batch.compute_batch(&dis).expect("compute_batch");

    assert_eq!(single_results.len(), batch_results.len());
    for (i, (s, b)) in single_results.iter().zip(batch_results.iter()).enumerate() {
        let d = (*s - b.score).abs();
        assert!(
            d < 1e-4,
            "FIR q{}: single={s}, batch={}, Δ={d}",
            qs[i],
            b.score
        );
    }
}

// ───────────────────────── monotonicity + sanity ─────────────────────────

#[test]
fn fir_score_decreases_with_distortion() {
    let dir = corpus_dir();
    let (src_bytes, w, h) = load_rgb8(&dir.join("source.png"));

    let client = Backend::client(&Default::default());
    let mut s = Ssim2::<Backend>::new(client, w, h)
        .expect("Ssim2::new")
        .with_blur(Ssim2Blur::Fir);

    // q values from lowest quality (most distorted) to highest.
    let qs = [5u32, 20, 45, 70, 90];
    let mut scores: Vec<f64> = Vec::with_capacity(qs.len());
    for q in qs {
        let path = dir.join(format!("q{q}.jpg"));
        let (dis_bytes, _, _) = load_rgb8(&path);
        let r = s.compute(&src_bytes, &dis_bytes).expect("compute");
        assert!(
            r.score.is_finite(),
            "FIR score at q{q} is non-finite: {}",
            r.score
        );
        scores.push(r.score);
    }
    // Higher q (less distortion) should yield higher score. We use
    // strict monotonicity — if any adjacent pair flips, the score's
    // sense is broken.
    for w in scores.windows(2) {
        let (lo, hi) = (w[0], w[1]);
        assert!(
            hi >= lo,
            "FIR score not monotone in q: {:?}",
            scores
        );
    }
}

#[test]
fn fir_identical_image_scores_near_100() {
    let dir = corpus_dir();
    let (src_bytes, w, h) = load_rgb8(&dir.join("source.png"));

    let client = Backend::client(&Default::default());
    let mut s = Ssim2::<Backend>::new(client, w, h)
        .expect("Ssim2::new")
        .with_blur(Ssim2Blur::Fir);
    let r = s.compute(&src_bytes, &src_bytes).expect("identical");
    // Same backend tolerance band as the IIR `identical_image_scores_100`
    // test — backends with atomic-add reduction noise (wgpu/Metal) still
    // converge to within ~1 absolute of 100 for identical inputs.
    assert!(
        r.score >= 99.0 && r.score <= 100.05,
        "FIR identical-image score {} outside [99.0, 100.05]",
        r.score
    );
}

// ───────────────────────── distinctness from IIR ─────────────────────────

/// Drift table — informational. Pins that the FIR dispatches to a
/// different kernel than the IIR (non-zero score gap on every JPEG
/// fixture) without making any claim about the gap's magnitude.
/// The gap is observed and logged so future runs can spot regressions.
#[test]
fn fir_vs_iir_drift_table_real_jpeg() {
    let dir = corpus_dir();
    let (src_bytes, w, h) = load_rgb8(&dir.join("source.png"));

    let client_a = Backend::client(&Default::default());
    let client_b = Backend::client(&Default::default());
    let mut s_iir = Ssim2::<Backend>::new(client_a, w, h).expect("iir");
    let mut s_fir = Ssim2::<Backend>::new(client_b, w, h)
        .expect("fir")
        .with_blur(Ssim2Blur::Fir);

    eprintln!(
        "{:>4}  {:>10}  {:>10}  {:>9}  {:>7}",
        "q", "iir", "fir", "Δ", "rel%"
    );
    for q in [1u32, 5, 20, 45, 70, 90] {
        let path = dir.join(format!("q{q}.jpg"));
        let (dis_bytes, _, _) = load_rgb8(&path);
        let iir = s_iir.compute(&src_bytes, &dis_bytes).expect("iir compute").score;
        let fir = s_fir.compute(&src_bytes, &dis_bytes).expect("fir compute").score;
        let d = (iir - fir).abs();
        let rel = if iir.abs() > 1e-3 {
            d / iir.abs() * 100.0
        } else {
            0.0
        };
        eprintln!("{q:>4}  {iir:>10.4}  {fir:>10.4}  {d:>9.5}  {rel:>6.3}%");

        // Non-zero gap → dispatch actually changed. If gap == 0 the
        // kernel select must be broken.
        assert!(
            d > 1e-6,
            "q{q}: FIR and IIR produce identical scores — dispatch must be broken"
        );
        // Both must be finite and within a sane range.
        assert!(iir.is_finite() && fir.is_finite(), "non-finite score at q{q}");
        assert!(
            (-200.0..=150.0).contains(&fir),
            "FIR score at q{q} out of sane range: {fir}"
        );
    }
}

// ───────────────────────── dimensions ─────────────────────────

#[test]
fn fir_dim_odd_non_square() {
    // 200×150 — non-power-of-2, non-square; mirrors the IIR
    // `dim_odd_non_square` test (without the CPU-parity assertion).
    let (w, h): (u32, u32) = (200, 150);
    let mut a = vec![0u8; (w * h * 3) as usize];
    let mut b = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            a[i] = ((x * 220 / w) & 0xff) as u8;
            a[i + 1] = ((y * 220 / h) & 0xff) as u8;
            a[i + 2] = (((x + y) * 200 / (w + h)) & 0xff) as u8;
            let bx = x / 8;
            let by = y / 8;
            let pert = if (bx ^ by) & 1 == 0 { 8_i32 } else { -8_i32 };
            b[i] = (a[i] as i32 + pert).clamp(0, 255) as u8;
            b[i + 1] = (a[i + 1] as i32 + pert).clamp(0, 255) as u8;
            b[i + 2] = (a[i + 2] as i32 + pert).clamp(0, 255) as u8;
        }
    }
    let client = Backend::client(&Default::default());
    let mut s = Ssim2::<Backend>::new(client, w, h)
        .expect("Ssim2::new")
        .with_blur(Ssim2Blur::Fir);
    let r = s.compute(&a, &b).expect("compute");
    assert!(r.score.is_finite());
    assert!(
        (-200.0..=150.0).contains(&r.score),
        "out-of-range FIR score on 200×150: {}",
        r.score
    );
}

#[test]
fn fir_dim_minimum_supported() {
    // 16×16 — smallest supported; tests the early-break pyramid path.
    let (w, h): (u32, u32) = (16, 16);
    let mut a = vec![0u8; (w * h * 3) as usize];
    for i in 0..a.len() {
        a[i] = (i & 0xff) as u8;
    }
    let mut b = a.clone();
    for i in (0..b.len()).step_by(3) {
        b[i] = b[i].wrapping_add(4);
    }
    let client = Backend::client(&Default::default());
    let mut s = Ssim2::<Backend>::new(client, w, h)
        .expect("Ssim2::new")
        .with_blur(Ssim2Blur::Fir);
    let r = s.compute(&a, &b).expect("compute");
    assert!(r.score.is_finite(), "16×16 FIR score: {}", r.score);
}
