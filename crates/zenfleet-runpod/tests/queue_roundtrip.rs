//! Off-node roundtrip test of the RunPod pull `JobQueue`.
//!
//! The RunPod Pods path is *pull*: the worker drains an R2
//! `chunks.jsonl` manifest, racing for each chunk's claim. The atomic
//! R2 token-race itself (`try_claim`) shells out to `s5cmd` against a
//! real bucket, so it is exercised by the operator's real-pod smoke run
//! (the user's gate), not here. What IS testable off a RunPod node — and
//! what this file covers — is the [`zenfleet_cloud::JobQueue`] contract
//! the generic `run_worker` loop depends on:
//!
//! - `next_chunk` surfaces each manifest line as a [`Chunk`] (id +
//!   payload), in manifest order, then returns `None` when drained;
//! - `ack_chunk` is a no-op that always succeeds (the claim + sidecar
//!   are the durable per-chunk state — there is no ack channel).
//!
//! We drive this through the `skip_claims` path, which is the real,
//! shipped code path single-instance smoke runs use to bypass claim
//! contention — so the test exercises production code, not a mock.

use zenfleet_cloud::{ChunkId, ChunkOutcome, JobQueue};
use zenfleet_runpod::queue::{RunpodChunkQueue, RunpodQueueConfig};
use zenfleet_s3::S3Client;

fn client() -> S3Client {
    // No network is touched on the skip_claims path; the endpoint is
    // never dialed.
    S3Client::new("s5cmd", "https://acct.r2.cloudflarestorage.com", "r2")
}

/// A two-chunk manifest drains in order through `next_chunk`, each chunk
/// carries its raw manifest line as `payload`, each is `ack`'d, and the
/// queue then yields `None`.
#[test]
fn pull_queue_drains_manifest_in_order_and_acks() {
    let mut cfg = RunpodQueueConfig::for_run("smoke-run");
    cfg.skip_claims = true; // single-instance smoke: bypass the R2 race.

    let lines = vec![
        r#"{"chunk_id":"chunk-0","codec":"jpeg"}"#.to_string(),
        r#"{"chunk_id":"chunk-1","codec":"webp"}"#.to_string(),
    ];
    let mut queue = RunpodChunkQueue::from_lines(client(), "pod-test", cfg, lines);

    // First chunk.
    let c0 = queue
        .next_chunk()
        .expect("next_chunk ok")
        .expect("first chunk present");
    assert_eq!(c0.id.as_str(), "chunk-0");
    assert!(c0.payload.contains("\"codec\":\"jpeg\""));
    queue
        .ack_chunk(&c0.id, ChunkOutcome::Done)
        .expect("ack of a Done chunk is a no-op that succeeds");

    // Second chunk.
    let c1 = queue
        .next_chunk()
        .expect("next_chunk ok")
        .expect("second chunk present");
    assert_eq!(c1.id.as_str(), "chunk-1");
    assert!(c1.payload.contains("\"codec\":\"webp\""));
    // A Failed ack is also a no-op on the pull path (the claim is left
    // for a peer to steal once stale).
    queue
        .ack_chunk(
            &c1.id,
            ChunkOutcome::Failed {
                error: "simulated".into(),
            },
        )
        .expect("ack of a Failed chunk is a no-op that succeeds");

    // Drained.
    assert!(
        queue.next_chunk().expect("next_chunk ok").is_none(),
        "queue should be drained after both chunks"
    );
}

/// An empty manifest yields `None` immediately.
#[test]
fn empty_manifest_yields_none() {
    let cfg = RunpodQueueConfig::for_run("empty-run");
    let mut queue = RunpodChunkQueue::from_lines(client(), "pod-test", cfg, Vec::new());
    assert!(queue.next_chunk().expect("next_chunk ok").is_none());
}

/// `ack_chunk` succeeds for every outcome variant and never requires an
/// in-flight chunk (no per-chunk state to manage on the pull path).
#[test]
fn ack_succeeds_for_all_outcomes() {
    let cfg = RunpodQueueConfig::for_run("ack-run");
    let mut queue = RunpodChunkQueue::from_lines(client(), "pod-test", cfg, Vec::new());
    let id = ChunkId("any".into());
    for outcome in [
        ChunkOutcome::Done,
        ChunkOutcome::Skipped {
            reason: "exists".into(),
        },
        ChunkOutcome::Retryable {
            error: "net".into(),
        },
        ChunkOutcome::Failed {
            error: "boom".into(),
        },
    ] {
        assert!(queue.ack_chunk(&id, outcome).is_ok());
    }
}
