//! VRAM isolation test (CUDA-gated) — mirrors the task #152 spike
//! through the real `MetricSession` API.
//!
//! Two sessions each build a cvvdp metric (real device working set on
//! the session's PRIVATE cubecl stream). We read each stream's pool
//! `memory_usage().bytes_reserved` (the in-API truth of device bytes
//! that stream's pool holds, sampled after `sync()`), then drop one
//! session — its `Drop` runs `memory_cleanup()` + `sync()` on its own
//! stream. We assert:
//!
//! - the dropped session's stream pool `bytes_reserved` → 0, AND
//! - the surviving session's stream pool stays fully resident.
//!
//! This is the load-bearing isolation property: dropping one session
//! frees EXACTLY its own VRAM, independent of the other. Measured via
//! the cubecl per-stream `memory_usage()` accounting (not extrapolated,
//! not nvidia-smi which can't attribute per-stream).
//!
//! Gated on `cuda` (+ default `cvvdp`). Requires a working CUDA runtime
//! + a physical GPU. Fails loudly without one (NO GRACEFUL SKIPS).

#![cfg(all(feature = "cuda", feature = "cvvdp"))]

use zenmetrics_api::{Backend, MetricKind, MetricParams, MetricSession};

// Large enough that the cvvdp working set reserves a clearly-nonzero,
// multi-page pool — so the "freed to ~0" assertion is unambiguous and
// not swamped by per-page rounding. 1024×1024 cvvdp Full mode is well
// into the hundreds-of-MiB range.
const W: u32 = 1024;
const H: u32 = 1024;

fn pair() -> (Vec<u8>, Vec<u8>) {
    let n = (W as usize) * (H as usize) * 3;
    let mut r = vec![0u8; n];
    for (i, b) in r.iter_mut().enumerate() {
        *b = ((i * 2654435761usize) >> 13) as u8;
    }
    let mut d = r.clone();
    // Perturb d so the score isn't the trivial-identical sentinel and
    // the dist-side pipeline actually allocates its working set.
    for (i, b) in d.iter_mut().enumerate() {
        *b = b.wrapping_add(((i * 40503) & 0x1f) as u8);
    }
    (r, d)
}

fn reserved(backend: Backend, stream_value: u64) -> u64 {
    zenmetrics_api::__stream_reserved_bytes(backend, stream_value)
        .expect("stream_reserved_bytes returned None (cvvdp/cuda must be enabled)")
}

#[test]
fn two_sessions_drop_one_frees_only_its_pool() {
    let backend = Backend::Cuda;
    let (r, d) = pair();

    // --- Session A: acquire, build metric, score (forces device alloc)
    let ctx_a = MetricSession::acquire(backend).expect("acquire A");
    let stream_a = ctx_a.__stream_value();
    let mut m_a = ctx_a
        .metric(
            MetricKind::Cvvdp,
            W,
            H,
            MetricParams::default_for(MetricKind::Cvvdp),
        )
        .expect("A.metric(Cvvdp)");
    let score_a = m_a.score(&r, &d).expect("A score");
    assert!(score_a.value.is_finite(), "A score finite");

    let a_reserved_live = reserved(backend, stream_a);
    assert!(
        a_reserved_live > 0,
        "session A's stream pool must hold a nonzero working set after scoring (got {a_reserved_live} bytes)"
    );

    // --- Session B: acquire, build metric, score -------------------
    let ctx_b = MetricSession::acquire(backend).expect("acquire B");
    let stream_b = ctx_b.__stream_value();
    assert_ne!(
        stream_a, stream_b,
        "two live sessions must own DISTINCT streams (no aliasing): a={stream_a} b={stream_b}"
    );
    let mut m_b = ctx_b
        .metric(
            MetricKind::Cvvdp,
            W,
            H,
            MetricParams::default_for(MetricKind::Cvvdp),
        )
        .expect("B.metric(Cvvdp)");
    let score_b = m_b.score(&r, &d).expect("B score");
    assert!(score_b.value.is_finite(), "B score finite");

    let b_reserved_live = reserved(backend, stream_b);
    assert!(
        b_reserved_live > 0,
        "session B's stream pool must hold a nonzero working set (got {b_reserved_live} bytes)"
    );

    // Per-stream isolation while both alive: A's pool accounting must
    // reflect ONLY A's footprint, not A+B.
    let a_reserved_both_alive = reserved(backend, stream_a);
    assert_eq!(
        a_reserved_both_alive, a_reserved_live,
        "A's pool must be unchanged by B's allocations on B's stream \
         (A={a_reserved_both_alive}, was {a_reserved_live}) — pools are per-stream"
    );

    // --- Drop A's metric handles + the session (Drop = cleanup+sync)
    drop(m_a);
    drop(ctx_a); // Drop runs memory_cleanup() + sync() on stream_a.

    // A's pool must be reclaimed to ~0; B's must stay fully resident.
    let a_reserved_after_drop = reserved(backend, stream_a);
    let b_reserved_after_a_drop = reserved(backend, stream_b);

    eprintln!(
        "[isolation] A live={a_reserved_live} B live={b_reserved_live} \
         | after drop A: A={a_reserved_after_drop} B={b_reserved_after_a_drop}"
    );

    assert_eq!(
        a_reserved_after_drop, 0,
        "dropping session A must reclaim its ENTIRE pool to the driver \
         (bytes_reserved should be 0, got {a_reserved_after_drop}) — nothing else \
         allocated on A's private stream so every page is fully-free"
    );
    assert_eq!(
        b_reserved_after_a_drop, b_reserved_live,
        "session B's pool must be UNTOUCHED by A's drop+cleanup \
         (B={b_reserved_after_a_drop}, was {b_reserved_live}) — ISOLATION"
    );

    // --- Drop B too → its pool also frees --------------------------
    drop(m_b);
    drop(ctx_b);
    let b_reserved_after_drop = reserved(backend, stream_b);
    assert_eq!(
        b_reserved_after_drop, 0,
        "dropping session B must reclaim its pool too (got {b_reserved_after_drop})"
    );
}
