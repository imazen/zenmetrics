//! `MetricSession` allocator cap + recycle tests.
//!
//! These exercise the process-global 128-slot-per-backend allocator
//! and `Drop`-based slot recycling **without touching the GPU**:
//! `MetricSession::acquire` claims a slot but builds no cubecl client
//! until `.metric(...)` is called, so the cap/recycle logic is testable
//! on any build with the `cvvdp` feature + a backend feature (default).
//!
//! The allocator is process-global, so the cap/recycle/leak assertions
//! that fill all 128 slots must NOT race a sibling `#[test]` (cargo runs
//! `#[test]`s on parallel threads sharing statics). They are therefore
//! collapsed into one sequential `#[test]` (`allocator_cap_recycle_leak`).
//!
//! Gated on `cuda` only because `acquire(Backend::Cuda)` must see cvvdp
//! report the cuda backend as usable (the default build). On a CI
//! runner without a physical GPU these still pass — no device work is
//! performed (per CLAUDE.md "NO GRACEFUL SKIPS": the test runs real
//! assertions, it doesn't skip).

#![cfg(all(feature = "cuda", feature = "cvvdp"))]

use zenmetrics_api::{Backend, Error, MetricSession, MAX_SESSIONS_PER_BACKEND};

#[test]
fn cap_is_128() {
    assert_eq!(MAX_SESSIONS_PER_BACKEND, 128);
}

/// One sequential test covering: acquire to the cap, error past it,
/// recycle on drop, and `leak()` not recycling. Single `#[test]` so the
/// process-global allocator isn't raced by a parallel sibling test.
#[test]
fn allocator_cap_recycle_leak() {
    let backend = Backend::Cuda;

    // --- baseline ---------------------------------------------------
    // Capture how many slots are already held (should be 0 in a fresh
    // binary, but be robust if another test in this binary leaked).
    let baseline = MetricSession::live_count(backend);
    let room = MAX_SESSIONS_PER_BACKEND - baseline;
    assert!(room >= 2, "need >=2 free slots; baseline={baseline}");

    // --- acquire every free slot ------------------------------------
    let mut sessions = Vec::new();
    for i in 0..room {
        let s = MetricSession::acquire(backend)
            .unwrap_or_else(|e| panic!("acquire #{i} of {room} free failed: {e}"));
        sessions.push(s);
    }
    assert_eq!(
        MetricSession::live_count(backend),
        MAX_SESSIONS_PER_BACKEND,
        "claiming all free slots fills the allocator"
    );

    // --- past the cap must ERROR, not alias -------------------------
    match MetricSession::acquire(backend) {
        Err(Error::TooManyContexts { backend: b }) => assert_eq!(b, "cuda"),
        Err(other) => panic!("expected TooManyContexts at the cap, got: {other}"),
        Ok(_) => panic!(
            "acquire succeeded past the {MAX_SESSIONS_PER_BACKEND}-slot cap — silent stream aliasing!"
        ),
    }

    // --- drop one → frees exactly one slot --------------------------
    let dropped = sessions.pop().expect("had sessions");
    drop(dropped);
    assert_eq!(
        MetricSession::live_count(backend),
        MAX_SESSIONS_PER_BACKEND - 1,
        "dropping a session frees exactly one slot"
    );

    // --- re-acquire now succeeds (slot recycled) --------------------
    let reacquired =
        MetricSession::acquire(backend).expect("acquire after drop must succeed (recycled)");
    assert_eq!(
        MetricSession::live_count(backend),
        MAX_SESSIONS_PER_BACKEND,
        "re-acquire fills the freed slot"
    );
    sessions.push(reacquired);

    // --- leak() consumes WITHOUT recycling --------------------------
    // Free one real slot first so we have room to acquire-then-leak.
    let s_for_room = sessions.pop().expect("had sessions");
    drop(s_for_room);
    let live_after_one_drop = MetricSession::live_count(backend);
    assert_eq!(live_after_one_drop, MAX_SESSIONS_PER_BACKEND - 1);

    let to_leak = MetricSession::acquire(backend).expect("acquire for leak ok");
    assert_eq!(MetricSession::live_count(backend), MAX_SESSIONS_PER_BACKEND);
    to_leak.leak();
    assert_eq!(
        MetricSession::live_count(backend),
        MAX_SESSIONS_PER_BACKEND,
        "leak() must NOT recycle the slot (pool stays resident, counts against cap)"
    );

    // --- cleanup: drop everything we still hold ---------------------
    // The leaked slot stays held permanently (by design). Everything in
    // `sessions` drops here, returning the allocator to baseline + 1
    // (the one leaked slot).
    drop(sessions);
    assert_eq!(
        MetricSession::live_count(backend),
        baseline + 1,
        "after dropping all but the leaked session, exactly one (leaked) slot remains held"
    );
}
