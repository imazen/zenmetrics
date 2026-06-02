//! `OwnedSessionMetric` (issue #17 / task #155 Phase A) tests, CUDA-gated.
//!
//! `OwnedSessionMetric` is the borrow-leash-free sibling of
//! `SessionMetric<'ctx>`: it owns both the scorer and the
//! `MetricSession` whose private stream it allocates on, so it can be
//! stored past the scope that built it (the warm-pool entry shape). The
//! soundness lever is field order — `scorer` drops before the session's
//! `Drop` cleans the stream — so the owned shape must:
//!
//! 1. **parity** — produce the SAME score as a borrowed `SessionMetric`
//!    AND a plain owned `Metric`, within the metric's `Atomic<f32>`
//!    reduction-noise band (the noise is a property of the kernels, not
//!    the session — see `session_parity.rs` module docs).
//! 2. **isolation** — dropping ONE `OwnedSessionMetric` reclaims exactly
//!    its own pool (`bytes_reserved → 0`) while a sibling's stays
//!    resident; this proves precise per-entry reclaim (the property the
//!    multi-warm pool relies on for VRAM-bounded eviction).
//! 3. **cap** — `into_metric` still respects the 128-slot allocator cap;
//!    a dropped `OwnedSessionMetric` recycles its slot, `leak()` does not.
//!
//! Gated on `cuda` (+ default `cvvdp`/`ssim2`). Requires a working CUDA
//! runtime + a physical GPU. Fails loudly without one (per CLAUDE.md
//! "NO GRACEFUL SKIPS").

#![cfg(all(feature = "cuda", feature = "cvvdp"))]

use zenmetrics_api::{
    Backend, Metric, MetricKind, MetricParams, MetricSession, OwnedSessionMetric,
};

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

/// Score the same `(ref, dist)` pair three ways: plain owned `Metric`,
/// borrowed `SessionMetric`, and `OwnedSessionMetric`. All three must
/// agree within the metric's `Atomic<f32>` reduction-noise band — an
/// `OwnedSessionMetric` changes only WHERE buffers allocate, never the
/// kernel math.
fn parity_for(kind: MetricKind, abs_tol: f64, rel_tol: f64) {
    let (r, d) = make_pair();

    // (1) plain owned Metric on the shared default stream.
    let plain = {
        let mut m = Metric::new(kind, Backend::Cuda, W, H, MetricParams::default_for(kind))
            .unwrap_or_else(|e| panic!("plain Metric::new({kind:?}) failed: {e}"));
        m.compute_srgb_u8(&r, &d)
            .unwrap_or_else(|e| panic!("plain {kind:?} score failed: {e}"))
    };

    // (2) borrowed SessionMetric on a private stream.
    let borrowed = {
        let ctx = MetricSession::acquire(Backend::Cuda).expect("acquire (borrowed)");
        let mut sm = ctx
            .metric(kind, W, H, MetricParams::default_for(kind))
            .unwrap_or_else(|e| panic!("ctx.metric({kind:?}) failed: {e}"));
        sm.score(&r, &d)
            .unwrap_or_else(|e| panic!("borrowed {kind:?} score failed: {e}"))
    };

    // (3) OwnedSessionMetric on a private stream — the welded shape.
    let owned = {
        let ctx = MetricSession::acquire(Backend::Cuda).expect("acquire (owned)");
        let mut om: OwnedSessionMetric = ctx
            .into_metric(kind, W, H, MetricParams::default_for(kind))
            .unwrap_or_else(|e| panic!("ctx.into_metric({kind:?}) failed: {e}"));
        assert_eq!(om.kind(), kind, "owned metric kind mismatch");
        assert_eq!(om.dims(), (W, H), "owned metric dims mismatch");
        assert_eq!(om.backend(), Backend::Cuda, "owned metric backend mismatch");
        om.score(&r, &d)
            .unwrap_or_else(|e| panic!("owned {kind:?} score failed: {e}"))
        // om drops here → scorer drops first, then session Drop cleans.
    };

    for v in [plain.value, borrowed.value, owned.value] {
        assert!(v.is_finite(), "{kind:?}: non-finite score {v}");
    }
    let tol = abs_tol + rel_tol * plain.value.abs();
    let d_bo = (borrowed.value - owned.value).abs();
    let d_po = (plain.value - owned.value).abs();
    let d_pb = (plain.value - borrowed.value).abs();
    assert!(
        d_bo <= tol && d_po <= tol && d_pb <= tol,
        "{kind:?}: scores diverged beyond tol {tol:.3e} — plain={}, borrowed={}, owned={} \
         (|b-o|={d_bo:.3e}, |p-o|={d_po:.3e}, |p-b|={d_pb:.3e}). An OwnedSessionMetric changing \
         the JOD/score is a real bug, not just reduction order.",
        plain.value,
        borrowed.value,
        owned.value
    );
}

#[test]
fn owned_parity_cvvdp() {
    parity_for(MetricKind::Cvvdp, 1e-5, 1e-5);
}

#[test]
#[cfg(feature = "ssim2")]
fn owned_parity_ssim2() {
    parity_for(MetricKind::Ssim2, 1e-3, 1e-5);
}

/// Warm-ref parity through the owned bundle: `set_reference` +
/// `score_with_warm_ref` must match the plain owned warm path.
#[test]
fn owned_warm_ref_matches_plain_cvvdp() {
    let (r, d) = make_pair();
    let kind = MetricKind::Cvvdp;

    let plain = {
        let mut m = Metric::new(kind, Backend::Cuda, W, H, MetricParams::default_for(kind))
            .expect("plain Metric::new(Cvvdp)");
        m.set_reference_srgb_u8(&r).expect("plain set_reference");
        m.compute_with_reference_srgb_u8(&d)
            .expect("plain warm score")
    };
    let owned = {
        let ctx = MetricSession::acquire(Backend::Cuda).expect("acquire owned");
        let mut om = ctx
            .into_metric(kind, W, H, MetricParams::default_for(kind))
            .expect("into_metric(Cvvdp)");
        om.set_reference_srgb_u8(&r).expect("owned set_reference");
        assert!(om.has_reference(), "owned must report cached ref");
        let s = om.score_with_warm_ref(&d).expect("owned warm score");
        om.clear_reference();
        s
    };
    assert!(plain.value.is_finite() && owned.value.is_finite());
    let delta = (plain.value - owned.value).abs();
    assert!(
        delta <= 1e-5,
        "owned warm-ref ({}) vs plain warm-ref ({}) differ by {delta:.3e} > 1e-5 JOD",
        owned.value,
        plain.value
    );
}

fn reserved(backend: Backend, stream_value: u64) -> u64 {
    zenmetrics_api::__stream_reserved_bytes(backend, stream_value)
        .expect("stream_reserved_bytes returned None (cvvdp/cuda must be enabled)")
}

/// Two `OwnedSessionMetric`s, each on its own private stream. Drop ONE
/// and assert its pool `bytes_reserved → 0` (precise per-entry reclaim
/// via the field-order drop) while the OTHER stays fully resident; then
/// drop the second → 0. This is the load-bearing property the multi-warm
/// pool relies on: evicting one entry frees exactly its VRAM.
#[test]
fn owned_drop_one_frees_only_its_pool() {
    let backend = Backend::Cuda;
    // 1024² so the cvvdp working set is clearly multi-page (hundreds of
    // MiB) and the "freed to 0" assertion isn't swamped by rounding.
    const WW: u32 = 1024;
    const HH: u32 = 1024;
    let n = (WW as usize) * (HH as usize) * 3;
    let mut r = vec![0u8; n];
    for (i, b) in r.iter_mut().enumerate() {
        *b = ((i * 2654435761usize) >> 13) as u8;
    }
    let mut d = r.clone();
    for (i, b) in d.iter_mut().enumerate() {
        *b = b.wrapping_add(((i * 40503) & 0x1f) as u8);
    }

    let ctx_a = MetricSession::acquire(backend).expect("acquire A");
    let mut a = ctx_a
        .into_metric(
            MetricKind::Cvvdp,
            WW,
            HH,
            MetricParams::default_for(MetricKind::Cvvdp),
        )
        .expect("A.into_metric");
    let stream_a = a.__stream_value();
    assert!(a.score(&r, &d).expect("A score").value.is_finite());
    let a_live = reserved(backend, stream_a);
    assert!(
        a_live > 0,
        "A pool must hold a nonzero working set (got {a_live})"
    );

    let ctx_b = MetricSession::acquire(backend).expect("acquire B");
    let mut b = ctx_b
        .into_metric(
            MetricKind::Cvvdp,
            WW,
            HH,
            MetricParams::default_for(MetricKind::Cvvdp),
        )
        .expect("B.into_metric");
    let stream_b = b.__stream_value();
    assert_ne!(
        stream_a, stream_b,
        "owned metrics must own DISTINCT streams"
    );
    assert!(b.score(&r, &d).expect("B score").value.is_finite());
    let b_live = reserved(backend, stream_b);
    assert!(
        b_live > 0,
        "B pool must hold a nonzero working set (got {b_live})"
    );

    // Per-stream isolation while both alive.
    assert_eq!(
        reserved(backend, stream_a),
        a_live,
        "A's pool must be unchanged by B's allocations — pools are per-stream"
    );

    // Drop A (scorer drops first → session Drop cleans its stream).
    drop(a);
    let a_after = reserved(backend, stream_a);
    let b_after_a = reserved(backend, stream_b);
    eprintln!(
        "[owned-isolation] A live={a_live} B live={b_live} | after drop A: A={a_after} B={b_after_a}"
    );
    assert_eq!(
        a_after, 0,
        "dropping OwnedSessionMetric A must reclaim its ENTIRE pool (got {a_after}) — \
         scorer dropped first, then session Drop ran cleanup on the now-empty stream"
    );
    assert_eq!(
        b_after_a, b_live,
        "B's pool must be UNTOUCHED by A's drop (B={b_after_a}, was {b_live}) — ISOLATION"
    );

    // Drop B too → its pool frees as well.
    drop(b);
    assert_eq!(
        reserved(backend, stream_b),
        0,
        "dropping OwnedSessionMetric B must reclaim its pool too"
    );
}

// NOTE: the `into_metric` cap/recycle/leak test lives in its OWN test
// file (`tests/session_owned_cap.rs`) so it's the sole test in that
// binary. The process-global 128-slot allocator must NOT be raced by a
// parallel sibling `#[test]` that also acquires slots (cargo runs a
// binary's `#[test]`s on parallel threads sharing statics) — the same
// reason `tests/session_cap.rs` is its own file.
