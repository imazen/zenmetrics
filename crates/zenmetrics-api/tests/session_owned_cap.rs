//! `OwnedSessionMetric` allocator cap + recycle + leak test (task #155
//! Phase A), CUDA-gated.
//!
//! Sole test in this binary: the process-global 128-slot-per-backend
//! allocator must NOT be raced by a parallel sibling `#[test]` that also
//! acquires slots (cargo runs a binary's `#[test]`s on parallel threads
//! sharing statics). Same isolation reason as `tests/session_cap.rs`.
//!
//! Verifies: `into_metric` claims a slot and respects the cap (129th
//! acquire → `TooManyContexts`, never aliases); dropping an
//! `OwnedSessionMetric` recycles its slot; `leak()` does not.
//!
//! Gated on `cuda` (+ default `cvvdp`). Promoting a session to an owned
//! metric via `into_metric` touches the GPU, so this needs a working
//! CUDA runtime + a physical GPU. Fails loudly without one (per
//! CLAUDE.md "NO GRACEFUL SKIPS").

#![cfg(all(feature = "cuda", feature = "cvvdp"))]

use zenmetrics_api::{
    Backend, Error, MAX_SESSIONS_PER_BACKEND, MetricKind, MetricParams, MetricSession,
};

const W: u32 = 256;
const H: u32 = 256;

#[test]
fn owned_into_metric_respects_cap_and_recycles() {
    let backend = Backend::Cuda;
    let baseline = MetricSession::live_count(backend);
    let room = MAX_SESSIONS_PER_BACKEND - baseline;
    assert!(room >= 3, "need >=3 free slots; baseline={baseline}");

    // Fill all-but-two free slots with bare sessions (no GPU work —
    // `acquire` claims a slot without building a client).
    let mut sessions = Vec::new();
    for _ in 0..(room - 2) {
        sessions.push(MetricSession::acquire(backend).expect("acquire bare"));
    }
    // Promote one slot to an owned metric (touches GPU).
    let owned = MetricSession::acquire(backend)
        .expect("acquire for owned")
        .into_metric(
            MetricKind::Cvvdp,
            W,
            H,
            MetricParams::default_for(MetricKind::Cvvdp),
        )
        .expect("into_metric");
    // One slot left. Take it with a bare session so we're exactly at cap.
    sessions.push(MetricSession::acquire(backend).expect("acquire last"));
    assert_eq!(
        MetricSession::live_count(backend),
        MAX_SESSIONS_PER_BACKEND,
        "all slots claimed (room-2 bare + 1 owned + 1 last)"
    );

    // 129th acquire must error, not alias.
    match MetricSession::acquire(backend) {
        Err(Error::TooManyContexts { backend: b }) => assert_eq!(b, "cuda"),
        Err(other) => panic!("expected TooManyContexts at cap, got: {other}"),
        Ok(_) => panic!("acquire succeeded past the cap — silent stream aliasing!"),
    }

    // Drop the owned metric → recycles exactly one slot (scorer drops
    // first, then session Drop cleans + releases the slot).
    drop(owned);
    assert_eq!(
        MetricSession::live_count(backend),
        MAX_SESSIONS_PER_BACKEND - 1,
        "dropping an OwnedSessionMetric recycles its slot"
    );

    // Re-acquire-and-promote fills the freed slot.
    let owned2 = MetricSession::acquire(backend)
        .expect("re-acquire after drop")
        .into_metric(
            MetricKind::Cvvdp,
            W,
            H,
            MetricParams::default_for(MetricKind::Cvvdp),
        )
        .expect("into_metric 2");
    assert_eq!(
        MetricSession::live_count(backend),
        MAX_SESSIONS_PER_BACKEND,
        "re-acquire fills the freed slot"
    );

    // leak() does NOT recycle. Free one slot first so we have room.
    let one = sessions.pop().expect("had sessions");
    drop(one);
    assert_eq!(
        MetricSession::live_count(backend),
        MAX_SESSIONS_PER_BACKEND - 1
    );
    let to_leak = MetricSession::acquire(backend)
        .expect("acquire for leak")
        .into_metric(
            MetricKind::Cvvdp,
            W,
            H,
            MetricParams::default_for(MetricKind::Cvvdp),
        )
        .expect("into_metric leak");
    assert_eq!(MetricSession::live_count(backend), MAX_SESSIONS_PER_BACKEND);
    to_leak.leak();
    assert_eq!(
        MetricSession::live_count(backend),
        MAX_SESSIONS_PER_BACKEND,
        "OwnedSessionMetric::leak() must NOT recycle the slot"
    );

    // Cleanup: drop the rest. owned2 + all bare sessions recycle; the
    // one leaked slot stays held permanently (by design).
    drop(owned2);
    drop(sessions);
    assert_eq!(
        MetricSession::live_count(backend),
        baseline + 1,
        "after dropping all but the leaked owned metric, exactly one (leaked) slot remains"
    );
}
