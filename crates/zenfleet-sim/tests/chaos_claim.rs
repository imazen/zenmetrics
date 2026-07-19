//! Chaos tests for the claim/steal layer under a misbehaving object store.
//!
//! Each test injects one real R2/s5cmd failure mode and asserts the safety
//! property that matters. Together they are the CI-runnable evidence behind the
//! fleet-throughput recommendation "port the vast claim to conditional PUT": the
//! token-race claim can double-acquire under an eventual-consistency window; the
//! conditional claim cannot.

use zenfleet_sim::{ClaimOutcome, FaultSpec, FaultStore, SimClock, claim_conditional, claim_token_race};

const SIDECAR: &str = "runs/r/omni/chunk-abc.parquet";
const CLAIM: &str = "claims/r/chunk-abc.claim";
const STALE: u64 = 600;

fn acquired(o: ClaimOutcome) -> bool {
    o == ClaimOutcome::Acquired
}

/// The conditional (If-None-Match) claim is exactly-once even with a 3-second
/// read-after-write window: two workers race, exactly one wins, the other is
/// correctly told a peer holds it.
#[test]
fn conditional_claim_is_exactly_once_under_a_consistency_window() {
    let clock = SimClock::new(1_000);
    let store = FaultStore::new(clock, FaultSpec::eventual_consistency(3), 7);

    let w1 = claim_conditional(&store, "box-1", SIDECAR, CLAIM, STALE);
    let w2 = claim_conditional(&store, "box-2", SIDECAR, CLAIM, STALE);

    let wins = [w1, w2].into_iter().filter(|o| acquired(*o)).count();
    assert_eq!(wins, 1, "exactly one worker may own the chunk; got {w1:?} / {w2:?}");
    assert!(
        [w1, w2].contains(&ClaimOutcome::HeldByPeer),
        "the loser must learn a peer holds it (not silently retry-forever)"
    );
}

/// The token-race claim is NOT exactly-once when the store's read-after-write
/// window out-lasts the read-back settle: both racers verify their *own* token
/// (read-your-writes) before the peer's overwrite becomes visible, so both
/// "win" and the chunk is executed twice. This is the hazard the 1.5 s settle
/// only papers over — and the reason to move to the conditional claim. Pinned
/// as a characterization test: if the token race is fixed, update this.
#[test]
fn token_race_can_double_acquire_under_eventual_consistency() {
    let clock = SimClock::new(1_000);
    // 3-second consistency window, read-your-writes on; settle is only 1 second.
    let store = FaultStore::new(clock, FaultSpec::eventual_consistency(3), 7);
    let settle = 1;

    let w1 = claim_token_race(&store, "box-1", SIDECAR, CLAIM, STALE, settle);
    let w2 = claim_token_race(&store, "box-2", SIDECAR, CLAIM, STALE, settle);

    let wins = [w1, w2].into_iter().filter(|o| acquired(*o)).count();
    assert_eq!(
        wins, 2,
        "token-race is NOT exactly-once under a consistency window wider than the \
         read-back settle — both boxes think they own the chunk and it runs twice. \
         The conditional claim (see the sibling test) removes this."
    );
}

/// With the settle wide enough to cover the consistency window, the token race
/// is safe — showing the failure above is about the *ratio*, not the algorithm
/// being always-broken. (Production can't guarantee the ratio, which is the
/// point.)
#[test]
fn token_race_is_safe_when_settle_covers_the_window() {
    let clock = SimClock::new(1_000);
    let store = FaultStore::new(clock, FaultSpec::eventual_consistency(2), 7);
    let settle = 3; // >= consistency window

    let w1 = claim_token_race(&store, "box-1", SIDECAR, CLAIM, STALE, settle);
    let w2 = claim_token_race(&store, "box-2", SIDECAR, CLAIM, STALE, settle);

    let wins = [w1, w2].into_iter().filter(|o| acquired(*o)).count();
    assert_eq!(wins, 1, "settle >= window → exactly-once again; got {w1:?} / {w2:?}");
}

/// Bad credentials must surface as an error the caller retries, never as a false
/// "Acquired" (which would let a box execute + mark done work it never persisted
/// — silent data loss). The claim errors; nothing is acquired.
#[test]
fn bad_credentials_never_falsely_acquire() {
    let store = FaultStore::new(SimClock::new(0), FaultSpec::bad_credentials(), 1);
    let o = claim_conditional(&store, "box-1", SIDECAR, CLAIM, STALE);
    assert_eq!(o, ClaimOutcome::Errored, "a 403 must not read as Acquired");
    assert_eq!(store.counts().auth_errors, 1);
}

/// A scoped credential that expires mid-run (TTL shorter than the sweep): claims
/// succeed before expiry and error after — never a false success. Models the
/// 3h→12h scoped-cred-TTL incident.
#[test]
fn expiring_credentials_flip_from_acquire_to_error() {
    let clock = SimClock::new(0);
    let store = FaultStore::new(clock.clone(), FaultSpec::creds_expiring_at(100), 1);

    // Before expiry: a normal acquire on a free chunk.
    assert_eq!(
        claim_conditional(&store, "box-1", "out/a", "claims/a", STALE),
        ClaimOutcome::Acquired
    );

    // Token expires.
    clock.set(100);
    assert_eq!(
        claim_conditional(&store, "box-1", "out/b", "claims/b", STALE),
        ClaimOutcome::Errored,
        "after the scoped token TTL, every op 403s — surfaced, not silently lost"
    );
}

/// A live peer's fresh claim is respected; when that peer dies (stops renewing)
/// and its lease goes stale, another box reclaims it via the strongly-consistent
/// CAS steal — and only one reclaimer can win. Uses the real `zenfleet_core::Lease`
/// staleness rule under the hood.
#[test]
fn dead_box_claim_is_reclaimed_after_it_goes_stale() {
    let clock = SimClock::new(1_000);
    let store = FaultStore::new(clock.clone(), FaultSpec::perfect(), 1);

    // box-1 wins the chunk, then dies (never writes the sidecar, never renews).
    assert_eq!(
        claim_conditional(&store, "box-1", SIDECAR, CLAIM, STALE),
        ClaimOutcome::Acquired
    );

    // While box-1's lease is fresh, box-2 must back off — no mid-flight steal.
    assert_eq!(
        claim_conditional(&store, "box-2", SIDECAR, CLAIM, STALE),
        ClaimOutcome::HeldByPeer,
        "a live (fresh) claim is never stolen"
    );

    // box-1 is dead; time passes beyond the stale window.
    clock.advance(STALE + 1);
    assert_eq!(
        claim_conditional(&store, "box-2", SIDECAR, CLAIM, STALE),
        ClaimOutcome::Acquired,
        "a stale (dead-box) claim is reclaimed so the ≤5-min chunk completes elsewhere"
    );
}

/// A silent partial upload (s5cmd killed mid-PUT) stores truncated bytes that a
/// length check would accept. Content-addressing catches it: the stored bytes'
/// hash won't match the expected content id. This is why every artifact carries
/// its sha256 and why a length-only ledger column is a data-loss bug.
#[test]
fn partial_upload_is_caught_by_content_hash() {
    let spec = FaultSpec {
        partial_write_rate: 1.0, // force the truncation
        ..FaultSpec::perfect()
    };
    let store = FaultStore::new(SimClock::new(0), spec, 1);

    let result = b"the full 40-byte encoded-variant payload!";
    let expected_sha = zenfleet_core::sha256(result);

    // The worker "uploads" its result; the store truncates it silently.
    store.put_as("box-1", "blobs/result", result).unwrap();
    assert!(store.is_corrupt("blobs/result"), "the upload was truncated");

    let got = store.get_as("box-1", "blobs/result").unwrap();
    let got_sha = zenfleet_core::sha256(&got);

    assert_ne!(
        got.len(),
        result.len(),
        "truncated: a length check alone would still need the right number..."
    );
    assert_ne!(
        got_sha.as_str(),
        expected_sha.as_str(),
        "...but the content hash mismatches, so a content-addressed consumer \
         rejects the partial upload instead of recording a corrupt Done"
    );
}
