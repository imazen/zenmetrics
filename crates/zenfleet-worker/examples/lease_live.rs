//! Live verification of the R2 dead-worker reclaim (`claim_or_steal_r2`). Drives the claim with
//! controlled timestamps against a real R2 prefix — proves a stale claim is stolen and a fresh one
//! is not. Run with AWS_* creds + endpoint in env:
//!
//!   ZEN_R2_ENDPOINT=https://<acct>.r2.cloudflarestorage.com ZEN_LEASE_PREFIX=lease-test/$$ \
//!     cargo run -p zenfleet-worker --example lease_live
//!
//! Not a #[test] — it needs creds + network, so the *caller* decides when to run it.

use zenfleet_core::{JobId, JobKind, sha256};
use zenfleet_worker::claim_or_steal_r2;

fn main() {
    let ep = std::env::var("ZEN_R2_ENDPOINT").expect("set ZEN_R2_ENDPOINT");
    let bucket =
        std::env::var("ZEN_LEASE_BUCKET").unwrap_or_else(|_| "zen-tuning-ephemeral".into());
    let prefix = std::env::var("ZEN_LEASE_PREFIX").expect("set ZEN_LEASE_PREFIX");
    let kind = JobKind::Metric {
        metric: "cvvdp".into(),
    };

    // (1) stale-steal: a "ghost" worker claims at ts=0, then a live worker at ts=1000 (ttl=10) steals.
    let j1 = JobId::of(&kind, &[sha256(b"stale-job")]);
    assert!(
        claim_or_steal_r2(&ep, &bucket, &prefix, &j1, 0, 10, None, "ghost"),
        "ghost makes the fresh claim"
    );
    let stolen = claim_or_steal_r2(&ep, &bucket, &prefix, &j1, 1000, 10, None, "live-worker");
    println!("stale-steal      = {stolen}  (expect true: 1000-0 >= ttl 10 → reclaimed)");

    // (2) fresh-skip: ghost claims at ts=995, live worker at ts=1000 (ttl=10) must NOT steal (5 < 10).
    let j2 = JobId::of(&kind, &[sha256(b"fresh-job")]);
    assert!(
        claim_or_steal_r2(&ep, &bucket, &prefix, &j2, 995, 10, None, "ghost"),
        "ghost makes the fresh claim"
    );
    let not_stolen = !claim_or_steal_r2(&ep, &bucket, &prefix, &j2, 1000, 10, None, "live-worker");
    println!("fresh-not-stolen = {not_stolen}  (expect true: 1000-995 < ttl 10 → left alone)");

    // (3) brand-new job is claimed outright.
    let j3 = JobId::of(&kind, &[sha256(b"new-job")]);
    let new_claim = claim_or_steal_r2(&ep, &bucket, &prefix, &j3, 1000, 10, None, "live-worker");
    println!("new-claim        = {new_claim}  (expect true)");

    assert!(stolen, "stale claim MUST be reclaimable");
    assert!(not_stolen, "fresh claim MUST NOT be stolen");
    assert!(new_claim, "unclaimed job MUST be claimable");
    println!("LEASE RECLAIM OK");
}
