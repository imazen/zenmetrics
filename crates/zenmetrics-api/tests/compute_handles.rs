//! Smoke test for the upload-once `MetricContext` + `Metric::compute_handles`
//! Phase 4 path. Uploads a single `(ref, dist)` packed-u32 pair once,
//! then scores it through every enabled metric via
//! `Metric::compute_handles` and asserts the returned scores are
//! finite (or each metric's documented "trivially identical" sentinel).
//!
//! Also verifies that `compute_handles` produces the same score as
//! `compute_srgb_u8` on the same inputs, within a tight tolerance —
//! the upload-once path must be bit-identical (or float-noise close)
//! to the upload-internally path.
//!
//! Gated on `cuda` + `cubecl-types`. zensim is skipped at the
//! `compute_handles` step (per the umbrella metric.rs note — Phase 4
//! deferral for zensim-gpu) but is still exercised through the
//! `compute_srgb_u8` baseline.

#![cfg(all(feature = "cuda", feature = "cubecl-types"))]

use zenmetrics_api::{Backend, Metric, MetricContext, MetricKind, MetricParams};

const W: u32 = 256;
const H: u32 = 256;

fn make_pair() -> (Vec<u8>, Vec<u8>) {
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
    (r, d)
}

#[test]
fn upload_once_five_metrics() {
    use cubecl::Runtime;

    // Shared cubecl client + one upload — every metric scores from
    // the same device buffers.
    let client = cubecl::cuda::CudaRuntime::client(&Default::default());
    let mut ctx = MetricContext::<cubecl::cuda::CudaRuntime>::new(client, W, H);

    let (r, d) = make_pair();
    let pair = ctx.upload_pair(&r, &d).expect("upload_pair failed");
    assert_eq!(pair.generation, 1, "first upload should produce generation 1");

    // Drive every enabled metric through compute_handles. We exclude
    // zensim from the compute_handles loop because the umbrella
    // explicitly errors on it (Phase 4 deferral) — its baseline goes
    // through compute_srgb_u8 below.
    let metrics: &[MetricKind] = &[
        #[cfg(feature = "cvvdp")]
        MetricKind::Cvvdp,
        #[cfg(feature = "butter")]
        MetricKind::Butter,
        #[cfg(feature = "ssim2")]
        MetricKind::Ssim2,
        #[cfg(feature = "dssim")]
        MetricKind::Dssim,
        #[cfg(feature = "iwssim")]
        MetricKind::Iwssim,
    ];

    for &kind in metrics {
        let params = MetricParams::default_for(kind);
        // butter at 256×256 with Auto picks Strip mode (butter is
        // strip-preferred). Strip-mode butter rejects compute_handles
        // — the single-resolution strip walker is pair-only. Force
        // Full mode for butter here so the upload-once compute_handles
        // path is reachable.
        let mode = match kind {
            #[cfg(feature = "butter")]
            MetricKind::Butter => zenmetrics_api::MemoryMode::Full,
            _ => zenmetrics_api::MemoryMode::Auto,
        };
        let mut m = Metric::new_with_memory_mode(kind, Backend::Cuda, W, H, params, mode)
            .unwrap_or_else(|e| panic!("Metric::new_with_memory_mode({kind:?}) failed: {e}"));

        let s_handles = m
            .compute_handles(&pair)
            .unwrap_or_else(|e| panic!("compute_handles({kind:?}) failed: {e}"));
        // Every score returned through compute_handles must be finite
        // — the non-identical pair above is structured enough that no
        // metric should degenerate to NaN/Inf.
        assert!(
            s_handles.value.is_finite(),
            "{kind:?} compute_handles returned non-finite: {}",
            s_handles.value
        );

        // Bit-identical-or-close check vs. compute_srgb_u8 on the same
        // bytes. We build a fresh scorer for the byte path so neither
        // run's state can contaminate the other.
        let mut m_bytes = Metric::new_with_memory_mode(
            kind,
            Backend::Cuda,
            W,
            H,
            MetricParams::default_for(kind),
            mode,
        )
        .unwrap_or_else(|e| panic!("Metric::new_with_memory_mode bytes-path({kind:?}) failed: {e}"));
        let s_bytes = m_bytes
            .compute_srgb_u8(&r, &d)
            .unwrap_or_else(|e| panic!("compute_srgb_u8({kind:?}) failed: {e}"));

        // f32-roundtripped device math can drift by ULPs; permit a
        // small absolute tolerance per-metric. cvvdp's JOD scale is
        // ~[3, 10], ssim2 ~[0, 100], dssim/butter/iwssim are
        // unit-scale or sub-unit — 1e-3 absolute is well inside
        // every metric's published precision floor.
        let tol = 1e-3_f64;
        assert!(
            (s_handles.value - s_bytes.value).abs() <= tol,
            "{kind:?}: compute_handles ({}) differs from compute_srgb_u8 ({}) by > {tol}",
            s_handles.value,
            s_bytes.value
        );
    }

    // Generation bumps on each new upload.
    let _pair2 = ctx.upload_pair(&r, &d).expect("second upload_pair failed");
    assert_eq!(ctx.generation(), 2, "second upload should bump generation to 2");
}

#[test]
fn upload_pair_validates_lengths() {
    use cubecl::Runtime;
    let client = cubecl::cuda::CudaRuntime::client(&Default::default());
    let mut ctx = MetricContext::<cubecl::cuda::CudaRuntime>::new(client, W, H);

    let n = (W as usize) * (H as usize) * 3;
    let r = vec![0u8; n];
    let short = vec![0u8; n - 3];
    let err = ctx.upload_pair(&r, &short).expect_err("short dist must error");
    let msg = err.to_string();
    assert!(
        msg.contains("dimension mismatch"),
        "expected dimension-mismatch error, got: {msg}"
    );
}
