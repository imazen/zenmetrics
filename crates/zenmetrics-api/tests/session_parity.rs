//! `SessionMetric` (isolated-stream) must produce the **same** cvvdp
//! JOD as the owned `Metric` (shared default stream) for the same
//! `(ref, dist)` pair. A `MetricSession` changes only WHERE the metric
//! allocates (its private cubecl stream), not the math — so the score
//! must match to within the metric's own reduction noise.
//!
//! ## Why these compare with a tolerance, not bit-exactly
//!
//! cvvdp's per-band pooling sums via `Atomic<f32>`, whose accumulation
//! ORDER is not deterministic — it shifts with kernel launch scheduling
//! and with whatever else is dispatching on the GPU concurrently. This
//! is a property of the cvvdp kernels, NOT of `MetricSession`. Measured
//! on this RTX 5070 box: repeated owned-vs-owned scores of the same
//! pair span `~1e-6` JOD on BOTH the one-shot and warm-ref paths
//! (e.g. 6.5223283.. / 6.5223288.. / 6.5223293..) — and the spread
//! only manifests once other GPU work interleaves (a single isolated
//! run can land bit-identical by luck). So the session must agree with
//! the owned metric to within that reduction-noise band — landing
//! OUTSIDE it would mean the session changed the computation (a real
//! bug). We bound at `1e-5` JOD: well above the ~1e-6 reduction noise,
//! far below any difference a genuine miscomputation would produce.
//!
//! Gated on `cuda` (+ default `cvvdp`). Requires a working CUDA runtime
//! + a physical GPU. On a GPU-less runner this fails loudly at the
//! first GPU dispatch (per CLAUDE.md "NO GRACEFUL SKIPS").

#![cfg(all(feature = "cuda", feature = "cvvdp"))]

use zenmetrics_api::{Backend, Metric, MetricKind, MetricParams, MetricSession};

/// cvvdp `Atomic<f32>` reduction-order noise band (measured ~1e-6 on
/// this box, bounded at 1e-5 with margin). Applies to BOTH the one-shot
/// and warm-ref paths — see the module docs.
const CVVDP_REDUCTION_NOISE_JOD: f64 = 1e-5;

const W: u32 = 256;
const H: u32 = 256;

fn make_pair() -> (Vec<u8>, Vec<u8>) {
    let n = (W as usize) * (H as usize) * 3;
    let mut r = Vec::with_capacity(n);
    let mut d = Vec::with_capacity(n);
    for y in 0..H {
        for x in 0..W {
            r.push((x & 0xff) as u8);
            r.push((y & 0xff) as u8);
            r.push(((x ^ y) & 0xff) as u8);
            d.push(((x.wrapping_add(9)) & 0xff) as u8);
            d.push(((y.wrapping_add(17)) & 0xff) as u8);
            d.push(((x ^ y ^ 11) & 0xff) as u8);
        }
    }
    (r, d)
}

/// One-shot score: owned `Metric` vs session `SessionMetric`, same
/// inputs → identical JOD.
#[test]
fn session_score_matches_owned_cvvdp() {
    let (r, d) = make_pair();

    // Owned metric on the shared default stream.
    let owned_score = {
        let mut m = Metric::new(
            MetricKind::Cvvdp,
            Backend::Cuda,
            W,
            H,
            MetricParams::default_for(MetricKind::Cvvdp),
        )
        .expect("owned Metric::new(Cvvdp) failed");
        m.compute_srgb_u8(&r, &d).expect("owned score failed")
    };

    // Session metric on a private isolated stream.
    let session_score = {
        let ctx = MetricSession::acquire(Backend::Cuda).expect("acquire session");
        let mut sm = ctx
            .metric(
                MetricKind::Cvvdp,
                W,
                H,
                MetricParams::default_for(MetricKind::Cvvdp),
            )
            .expect("ctx.metric(Cvvdp) failed");
        sm.score(&r, &d).expect("session score failed")
        // ctx drops here → cleanup on the private stream.
    };

    assert_eq!(owned_score.metric_name, "cvvdp");
    assert_eq!(session_score.metric_name, "cvvdp");
    assert!(
        owned_score.value.is_finite() && session_score.value.is_finite(),
        "both scores must be finite (owned={}, session={})",
        owned_score.value,
        session_score.value
    );
    // The session changes WHERE buffers allocate, not the kernel math.
    // The score must agree to within cvvdp's atomic-reduction noise.
    let delta = (owned_score.value - session_score.value).abs();
    assert!(
        delta <= CVVDP_REDUCTION_NOISE_JOD,
        "session score ({}) and owned score ({}) differ by {delta:.3e} JOD, which EXCEEDS \
         the cvvdp Atomic<f32> reduction-noise band ({CVVDP_REDUCTION_NOISE_JOD:.0e}) — the \
         private stream changed the JOD (a real bug), not just reduction order",
        session_score.value,
        owned_score.value
    );
}

/// Warm-reference path: owned vs session must agree to within the
/// metric's own `Atomic<f32>` reduction-order noise band (see the
/// module docs — same rationale as the one-shot path).
#[test]
fn session_warm_ref_matches_owned_cvvdp() {
    let (r, d) = make_pair();

    let owned_warm = {
        let mut m = Metric::new(
            MetricKind::Cvvdp,
            Backend::Cuda,
            W,
            H,
            MetricParams::default_for(MetricKind::Cvvdp),
        )
        .expect("owned Metric::new(Cvvdp) failed");
        m.set_reference_srgb_u8(&r).expect("owned set_reference");
        m.compute_with_cached_reference_srgb_u8(&d)
            .expect("owned warm score")
    };

    let session_warm = {
        let ctx = MetricSession::acquire(Backend::Cuda).expect("acquire session");
        let mut sm = ctx
            .metric(
                MetricKind::Cvvdp,
                W,
                H,
                MetricParams::default_for(MetricKind::Cvvdp),
            )
            .expect("ctx.metric(Cvvdp) failed");
        sm.set_reference_srgb_u8(&r).expect("session set_reference");
        sm.score_with_warm_ref(&d).expect("session warm score")
    };

    assert!(owned_warm.value.is_finite() && session_warm.value.is_finite());
    let delta = (owned_warm.value - session_warm.value).abs();
    assert!(
        delta <= CVVDP_REDUCTION_NOISE_JOD,
        "session warm-ref score ({}) and owned warm-ref score ({}) differ by {delta:.3e} JOD, \
         which EXCEEDS the cvvdp Atomic<f32> reduction-noise band ({CVVDP_REDUCTION_NOISE_JOD:.0e}) — \
         the session changed the computation (a real bug), not just reduction order",
        session_warm.value,
        owned_warm.value
    );
}

/// ssim2 is the second metric wired to `MetricSession`. Same property:
/// the session (private stream) must match the owned metric (default
/// stream) to within ssim2's own `Atomic<f32>` reduction noise. ssim2's
/// score range is ~0..100, so the band is scaled up accordingly
/// (1e-3 on a 0..100 scale ≈ the same relative tolerance as cvvdp's
/// 1e-5 on a ~10 scale).
#[test]
fn session_score_matches_owned_ssim2() {
    let (r, d) = make_pair();

    let owned = {
        let mut m = Metric::new(
            MetricKind::Ssim2,
            Backend::Cuda,
            W,
            H,
            MetricParams::default_for(MetricKind::Ssim2),
        )
        .expect("owned Metric::new(Ssim2) failed");
        m.compute_srgb_u8(&r, &d).expect("owned ssim2 score")
    };

    let session = {
        let ctx = MetricSession::acquire(Backend::Cuda).expect("acquire session");
        let mut sm = ctx
            .metric(
                MetricKind::Ssim2,
                W,
                H,
                MetricParams::default_for(MetricKind::Ssim2),
            )
            .expect("ctx.metric(Ssim2) failed");
        sm.score(&r, &d).expect("session ssim2 score")
    };

    assert_eq!(owned.metric_name, "ssim2");
    assert_eq!(session.metric_name, "ssim2");
    assert!(owned.value.is_finite() && session.value.is_finite());
    const SSIM2_REDUCTION_NOISE: f64 = 1e-3; // ~0..100 scale
    let delta = (owned.value - session.value).abs();
    assert!(
        delta <= SSIM2_REDUCTION_NOISE,
        "session ssim2 score ({}) and owned ({}) differ by {delta:.3e}, exceeding the \
         Atomic<f32> reduction-noise band ({SSIM2_REDUCTION_NOISE:.0e}) — the session changed \
         the computation, not just reduction order",
        session.value,
        owned.value
    );
}

/// All-wired-metrics one-shot parity: for every metric that is BOTH
/// feature-enabled AND wired into `MetricSession`, the session score
/// must match the owned-metric score to within an `abs + rel` tolerance
/// covering that metric's `Atomic<f32>` reduction-order noise (the noise
/// is a property of the kernels, not the session — see module docs).
/// Metrics not yet wired are skipped at the *caller* level here: the
/// list is built from the enabled features, and a not-wired metric would
/// surface `ctx.metric(...)` → Err which we treat as an explicit
/// "not wired" (asserted distinct from a compute error).
#[test]
fn session_parity_all_wired_metrics() {
    let (r, d) = make_pair();
    // (kind, abs_tol, rel_tol) — bands sized to each metric's value
    // scale; all comfortably above ~1e-6 relative reduction noise and
    // far below any genuine miscomputation.
    let cases: &[(MetricKind, f64, f64)] = &[
        #[cfg(feature = "cvvdp")]
        (MetricKind::Cvvdp, 1e-5, 1e-5),
        #[cfg(feature = "ssim2")]
        (MetricKind::Ssim2, 1e-3, 1e-5),
        #[cfg(feature = "butter")]
        (MetricKind::Butter, 1e-3, 1e-4),
        #[cfg(feature = "dssim")]
        (MetricKind::Dssim, 1e-4, 1e-4),
        #[cfg(feature = "iwssim")]
        (MetricKind::Iwssim, 1e-4, 1e-4),
        #[cfg(feature = "zensim")]
        (MetricKind::Zensim, 1e-3, 1e-4),
    ];

    for &(kind, abs_tol, rel_tol) in cases {
        let owned = {
            let mut m = Metric::new(kind, Backend::Cuda, W, H, MetricParams::default_for(kind))
                .unwrap_or_else(|e| panic!("owned Metric::new({kind:?}) failed: {e}"));
            m.compute_srgb_u8(&r, &d)
                .unwrap_or_else(|e| panic!("owned {kind:?} score failed: {e}"))
        };
        let session = {
            let ctx = MetricSession::acquire(Backend::Cuda).expect("acquire session");
            let mut sm = ctx
                .metric(kind, W, H, MetricParams::default_for(kind))
                .unwrap_or_else(|e| {
                    panic!("ctx.metric({kind:?}) failed — metric is enabled but not wired into MetricSession: {e}")
                });
            sm.score(&r, &d)
                .unwrap_or_else(|e| panic!("session {kind:?} score failed: {e}"))
        };
        assert!(
            owned.value.is_finite() && session.value.is_finite(),
            "{kind:?}: non-finite (owned={}, session={})",
            owned.value,
            session.value
        );
        let delta = (owned.value - session.value).abs();
        let tol = abs_tol + rel_tol * owned.value.abs();
        assert!(
            delta <= tol,
            "{kind:?}: session score ({}) and owned ({}) differ by {delta:.3e} > tol {tol:.3e} \
             — the private stream changed the computation (a real bug), not just reduction order",
            session.value,
            owned.value
        );
    }
}
