//! The two claim algorithms, side by side, over the [`FaultStore`].
//!
//! Production runs two different claims today:
//!
//! - [`claim_token_race`] mirrors `zenfleet-vastai::worker::claim::try_claim` —
//!   the vast/Hetzner sweep path: probe the sidecar, read the claim, write our
//!   own, **sleep a read-back delay**, re-read and hope our token survived. It
//!   uses only plain `put`/`get`, so its correctness depends entirely on the
//!   read-back settle out-lasting R2's read-after-write window.
//! - [`claim_conditional`] models the job-system path
//!   (`zenfleet-worker::claim_or_steal_r2`): one atomic `If-None-Match: *`
//!   create, and an `If-Match` compare-and-swap to steal a stale claim. No
//!   read-back, because the winner is decided strongly-consistently.
//!
//! Both decide staleness with the **real** [`zenfleet_core::Lease`], so the sim
//! exercises production lease logic rather than a copy. The chaos tests
//! (`tests/chaos_claim.rs`) run both under an eventual-consistency window and
//! show the token-race can double-acquire while the conditional claim stays
//! exactly-once — the evidence behind "port the claim to conditional PUT".

use zenfleet_core::Lease;

use crate::store::FaultStore;

/// The outcome of one claim attempt — the same shape the production claim
/// returns (`zenfleet-vastai`'s `ClaimOutcome`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClaimOutcome {
    /// We own the chunk; safe to execute.
    Acquired,
    /// The output sidecar already exists — nothing to do.
    AlreadyDone,
    /// A peer holds a fresh (non-stale) claim.
    HeldByPeer,
    /// We wrote a claim but a peer's write won — we do not own it.
    LostRace,
    /// A store op errored (transient / creds). The caller retries next pass.
    Errored,
}

/// Parse a claim body `"{token}\t{ts}\t{owner}"`.
fn parse_claim(s: &[u8]) -> Option<(String, u64, String)> {
    let text = std::str::from_utf8(s).ok()?;
    let mut parts = text.splitn(3, '\t');
    let token = parts.next()?.to_string();
    let ts: u64 = parts.next()?.parse().ok()?;
    let owner = parts.next()?.to_string();
    if token.is_empty() || owner.is_empty() {
        return None;
    }
    Some((token, ts, owner))
}

fn claim_body(worker: &str, now: u64) -> (String, Vec<u8>) {
    let token = format!("{worker}-{now}");
    let body = format!("{token}\t{now}\t{worker}").into_bytes();
    (token, body)
}

/// Token-race claim — the current vast/Hetzner path. `settle_secs` is the
/// read-back delay (production default 1.5 s, here in whole seconds). Safe only
/// while `settle_secs` covers the store's read-after-write window.
pub fn claim_token_race(
    store: &FaultStore,
    worker: &str,
    sidecar_key: &str,
    claim_key: &str,
    stale_secs: u64,
    settle_secs: u64,
) -> ClaimOutcome {
    // 1. Idempotency — did someone already produce the output?
    match store.head_as(worker, sidecar_key) {
        Ok(Some(_)) => return ClaimOutcome::AlreadyDone,
        Ok(None) => {}
        Err(_) => return ClaimOutcome::Errored,
    }

    // 2. Probe an existing claim; back off only for a fresh peer.
    let now = store.clock().now();
    match store.get_as(worker, claim_key) {
        Ok(bytes) => {
            if let Some((_, ts, owner)) = parse_claim(&bytes) {
                let fresh = !Lease::new(owner.clone(), ts, stale_secs).can_steal(now);
                if fresh && owner != worker {
                    return ClaimOutcome::HeldByPeer;
                }
                // stale or self-owned → fall through and (over)write.
            }
        }
        // Absent-or-not-visible reads as "no claim" — the token-race can't tell
        // the difference, which is the whole hazard.
        Err(_) => {}
    }

    // 3. Write our claim.
    let (token, body) = claim_body(worker, now);
    if store.put_as(worker, claim_key, &body).is_err() {
        return ClaimOutcome::Errored;
    }

    // 4. Read-back settle — the delay that's supposed to let a peer's write land.
    store.clock().advance(settle_secs);

    // 5. Verify our token survived.
    match store.get_as(worker, claim_key) {
        Ok(bytes) => match parse_claim(&bytes) {
            Some((t, _, _)) if t == token => ClaimOutcome::Acquired,
            _ => ClaimOutcome::LostRace,
        },
        Err(_) => ClaimOutcome::LostRace,
    }
}

/// Conditional-PUT claim — the job-system path. One atomic create; an
/// `If-Match` CAS to steal a stale claim. No read-back; the winner is decided
/// strongly-consistently, so it is exactly-once regardless of the consistency
/// window.
pub fn claim_conditional(
    store: &FaultStore,
    worker: &str,
    sidecar_key: &str,
    claim_key: &str,
    stale_secs: u64,
) -> ClaimOutcome {
    // 1. Idempotency.
    match store.head_as(worker, sidecar_key) {
        Ok(Some(_)) => return ClaimOutcome::AlreadyDone,
        Ok(None) => {}
        Err(_) => return ClaimOutcome::Errored,
    }

    // 2. Atomic create-if-absent (If-None-Match: *).
    let now = store.clock().now();
    let (_, body) = claim_body(worker, now);
    match store.put_if_absent(worker, claim_key, &body) {
        Ok(true) => return ClaimOutcome::Acquired,
        Ok(false) => {}
        Err(_) => return ClaimOutcome::Errored,
    }

    // 3. It exists. Read it (with its ETag) to decide fresh vs stale.
    let (existing, etag) = match store.get_with_etag(claim_key) {
        Ok(Some(v)) => v,
        // Created by a peer but not yet visible to us — treat as held; we can't
        // safely steal a claim we can't read. (Strong consistency of the *create*
        // guaranteed exactly-one already won; that's enough.)
        Ok(None) => return ClaimOutcome::HeldByPeer,
        Err(_) => return ClaimOutcome::Errored,
    };
    let Some((_, ts, owner)) = parse_claim(&existing) else {
        return ClaimOutcome::Errored;
    };
    if owner == worker {
        return ClaimOutcome::Acquired;
    }
    if !Lease::new(owner, ts, stale_secs).can_steal(now) {
        return ClaimOutcome::HeldByPeer;
    }

    // 4. Stale → steal via CAS on the observed ETag. Two reclaimers racing on the
    //    same ETag can't both win.
    let (_, steal_body) = claim_body(worker, now);
    match store.cas(worker, claim_key, &etag, &steal_body) {
        Ok(true) => ClaimOutcome::Acquired,
        Ok(false) => ClaimOutcome::LostRace,
        Err(_) => ClaimOutcome::Errored,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FaultSpec, SimClock};
    use zenfleet_cloud::BlobStorage;

    #[test]
    fn conditional_acquires_a_free_chunk() {
        let s = FaultStore::new(SimClock::new(0), FaultSpec::perfect(), 1);
        assert_eq!(
            claim_conditional(&s, "w1", "out/c.parquet", "claims/c", 600),
            ClaimOutcome::Acquired
        );
    }

    #[test]
    fn already_done_when_sidecar_present() {
        let s = FaultStore::new(SimClock::new(0), FaultSpec::perfect(), 1);
        s.put(&"out/c.parquet".into(), b"result").unwrap();
        assert_eq!(
            claim_conditional(&s, "w1", "out/c.parquet", "claims/c", 600),
            ClaimOutcome::AlreadyDone
        );
        assert_eq!(
            claim_token_race(&s, "w1", "out/c.parquet", "claims/c", 600, 2),
            ClaimOutcome::AlreadyDone
        );
    }
}
