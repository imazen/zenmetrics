//! Non-cvvdp session VRAM-reclaim test (CUDA-gated, ssim2).
//!
//! Regression test for the cvvdp-only cleanup-routing bug: before the
//! 6-arm fix, `cleanup_session_stream` / `stream_reserved_bytes` routed
//! ONLY to cvvdp, so in a build with cvvdp compiled OUT but another metric
//! IN, dropping a non-cvvdp `MetricSession` reclaimed NOTHING — silently
//! violating the crate's "drop frees exactly this session's VRAM" contract.
//!
//! This test builds an **ssim2** session (not cvvdp), scores it (forcing a
//! real device working set on the session's private stream), then drops it
//! and asserts the stream pool's `bytes_reserved` → 0. It is gated on
//! `ssim2 + cuda` so it runs in BOTH the default build and, crucially, the
//! `--no-default-features --features ssim2,cuda` build that exercises the
//! ssim2 cleanup arm (cvvdp's arm is compiled out there, so a pass proves
//! the per-metric routing — not cvvdp's metric-agnostic fallback). Pre-fix,
//! the no-cvvdp run leaves a nonzero pool and FAILS the reclaim assertion.
//!
//! NO GRACEFUL SKIPS — requires a real CUDA GPU; fails loudly without one.
#![cfg(all(feature = "cuda", feature = "ssim2"))]

use zenmetrics_api::{Backend, MetricKind, MetricParams, MetricSession};

const W: u32 = 1024;
const H: u32 = 1024;

fn pair() -> (Vec<u8>, Vec<u8>) {
    let n = (W as usize) * (H as usize) * 3;
    let mut r = vec![0u8; n];
    for (i, b) in r.iter_mut().enumerate() {
        *b = ((i * 2654435761usize) >> 13) as u8;
    }
    let mut d = r.clone();
    for (i, b) in d.iter_mut().enumerate() {
        *b = b.wrapping_add(((i * 40503) & 0x1f) as u8);
    }
    (r, d)
}

fn reserved(backend: Backend, stream_value: u64) -> Option<u64> {
    zenmetrics_api::__stream_reserved_bytes(backend, stream_value)
}

#[test]
fn dropping_ssim2_session_reclaims_its_pool() {
    let backend = Backend::Cuda;
    let (r, d) = pair();

    let ctx = MetricSession::acquire(backend).expect("acquire ssim2 session");
    let stream = ctx.__stream_value();
    let mut m = ctx
        .metric(
            MetricKind::Ssim2,
            W,
            H,
            MetricParams::default_for(MetricKind::Ssim2),
        )
        .expect("metric(Ssim2)");
    let s = m.score(&r, &d).expect("ssim2 score");
    assert!(s.value.is_finite(), "ssim2 score must be finite");

    let live = reserved(backend, stream)
        .expect("stream_reserved_bytes must route to the ssim2 arm (Some), not the no-op None");
    assert!(
        live > 0,
        "ssim2 session's stream pool must hold a nonzero working set after scoring (got {live})"
    );

    // Drop the metric handle then the session. The session's Drop runs
    // memory_cleanup()+sync() on its private stream — which must route to
    // ssim2's cleanup_stream (the fix), not the cvvdp-only no-op.
    drop(m);
    drop(ctx);

    let after = reserved(backend, stream).expect("stream_reserved_bytes Some after drop");
    assert_eq!(
        after, 0,
        "after dropping the ssim2 session its stream pool must be reclaimed to the driver \
         (bytes_reserved → 0); got {after}. Pre-fix this was the silent cvvdp-only-routing leak."
    );
}
