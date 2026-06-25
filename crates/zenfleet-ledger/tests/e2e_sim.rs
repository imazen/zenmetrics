//! End-to-end, in-process simulation of one sweep through the whole engine — the closest thing to
//! "operational" achievable without the real fleet / R2 / Railway. Exercises the composed behavior:
//!
//! declare desired → reconcile (gap) → lease-claim → execute (some succeed, some fail) →
//! persist per-chunk ledger sidecars → compact (latest-wins) → re-reconcile to CONVERGENCE →
//! GC plan over produced blobs.
//!
//! Proves goals A (declare→gap, idempotent), E (lease + convergence/self-heal), F (retry transient,
//! poison deterministic), G (GC keeps referenced, refuses irreplaceable orphans), I (no dup work).

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};

use zenfleet_core::{
    BlobIndexEntry, CellId, DesiredJob, ErrorClass, JobId, JobKind, JobStatus, Lease, LedgerRow,
    Regenerability, RetryPolicy, Sha256Hex, gc_plan, reconcile, sha256,
};
use zenfleet_ledger::{compact_ledger, read_ledger, write_ledger};

static N: AtomicU64 = AtomicU64::new(0);
fn tmp(tag: &str) -> std::path::PathBuf {
    let n = N.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "zenledger_e2e_{}_{}_{tag}.parquet",
        std::process::id(),
        n
    ))
}

/// Desired: score 4 distinct encodes with cvvdp.
fn desired_set() -> Vec<DesiredJob> {
    (0..4)
        .map(|i| {
            DesiredJob::new(
                JobKind::Metric {
                    metric: "cvvdp".into(),
                },
                vec![sha256(format!("encode-{i}").as_bytes())],
                CellId {
                    image_path: format!("img/{i}.png"),
                    codec: "zenjpeg".into(),
                    q: 80,
                    knob_tuple_json: "{}".into(),
                },
            )
        })
        .collect()
}

fn row(
    d: &DesiredJob,
    status: JobStatus,
    err: Option<ErrorClass>,
    attempts: u32,
    ts: u64,
    out: Option<Sha256Hex>,
) -> LedgerRow {
    LedgerRow {
        job_id: d.job_id(),
        kind: d.kind.clone(),
        cell: d.cell.clone(),
        output_sha: out,
        status,
        error_class: err,
        attempts,
        ts,
        worker: "w1".into(),
        provider: "oracle".into(),
    }
}

#[test]
fn full_run_converges() {
    let desired = desired_set();
    let policy = RetryPolicy { max_attempts: 3 };

    // ---- Round 1: empty ledger → the whole set is the gap (goal A: declare→enqueue). ----
    let view0 = zenfleet_core::LedgerView::new();
    let plan0 = reconcile(&desired, &view0, policy);
    assert_eq!(plan0.enqueue.len(), 4);
    assert_eq!(plan0.done, 0);

    // ---- Simulate execution. job0,job1 succeed; job2 transient-fails; job3 deterministic-fails. ----
    let sidecar1 = tmp("round1");
    let exec1 = vec![
        row(
            &desired[0],
            JobStatus::Done,
            None,
            1,
            100,
            Some(sha256(b"score0")),
        ),
        row(
            &desired[1],
            JobStatus::Done,
            None,
            1,
            100,
            Some(sha256(b"score1")),
        ),
        row(
            &desired[2],
            JobStatus::Failed,
            Some(ErrorClass::Timeout),
            1,
            100,
            None,
        ),
        row(
            &desired[3],
            JobStatus::Failed,
            Some(ErrorClass::MetricNan),
            1,
            100,
            None,
        ),
    ];
    write_ledger(&sidecar1, &exec1).unwrap();

    // ---- Round 2: read back, reconcile. done=2; transient retried; deterministic poisoned (goal F). ----
    let view1 = zenfleet_core::LedgerView::from_rows(read_ledger(&sidecar1).unwrap());
    let plan1 = reconcile(&desired, &view1, policy);
    assert_eq!(plan1.done, 2);
    assert_eq!(
        plan1.enqueue,
        vec![desired[2].job_id()],
        "transient failure retried"
    );
    assert_eq!(
        plan1.poison,
        vec![desired[3].job_id()],
        "deterministic failure poisoned, not retried"
    );

    // ---- Simulate: retry job2 → success; record job3 as POISON (the caller writes poison rows). ----
    let sidecar2 = tmp("round2");
    let exec2 = vec![
        row(
            &desired[2],
            JobStatus::Done,
            None,
            2,
            200,
            Some(sha256(b"score2")),
        ),
        row(
            &desired[3],
            JobStatus::Poison,
            Some(ErrorClass::MetricNan),
            1,
            200,
            None,
        ),
    ];
    write_ledger(&sidecar2, &exec2).unwrap();

    // ---- Compact the two per-chunk sidecars → one consolidated file (latest-wins). ----
    let compacted = tmp("compacted");
    let n = compact_ledger(&[&sidecar1, &sidecar2], &compacted).unwrap();
    assert_eq!(
        n, 4,
        "4 distinct jobs after collapsing the job2 Failed→Done history"
    );

    // ---- Round 3: reconcile over the compacted ledger → CONVERGENCE (goal E). ----
    let view2 = zenfleet_core::LedgerView::from_rows(read_ledger(&compacted).unwrap());
    let plan2 = reconcile(&desired, &view2, policy);
    assert!(
        plan2.enqueue.is_empty(),
        "nothing left to do — the run converged"
    );
    assert!(
        plan2.poison.is_empty(),
        "job3 already poisoned, not re-poisoned"
    );
    assert_eq!(plan2.done, 3, "3 succeeded; the 4th is terminally poisoned");

    // ---- Idempotency (goal A/I): re-declaring the same set is still a no-op gap. ----
    let plan2b = reconcile(&desired, &view2, policy);
    assert_eq!(
        plan2b, plan2,
        "re-running the reconciler is deterministic and idempotent"
    );

    // ---- GC over produced blobs (goal G). 3 score blobs referenced; 2 orphans. ----
    let referenced: HashSet<Sha256Hex> = [sha256(b"score0"), sha256(b"score1"), sha256(b"score2")]
        .into_iter()
        .collect();
    let index = vec![
        BlobIndexEntry {
            sha: sha256(b"score0"),
            size: 10,
            regenerability: Regenerability::CheapRegenerable,
            last_ref_secs: 1,
        },
        BlobIndexEntry {
            sha: sha256(b"score1"),
            size: 10,
            regenerability: Regenerability::CheapRegenerable,
            last_ref_secs: 1,
        },
        BlobIndexEntry {
            sha: sha256(b"score2"),
            size: 10,
            regenerability: Regenerability::CheapRegenerable,
            last_ref_secs: 1,
        },
        BlobIndexEntry {
            sha: sha256(b"orphan-jpeg"),
            size: 500,
            regenerability: Regenerability::CheapRegenerable,
            last_ref_secs: 1,
        },
        BlobIndexEntry {
            sha: sha256(b"orphan-source"),
            size: 9_000_000,
            regenerability: Regenerability::NotRegenerable,
            last_ref_secs: 1,
        },
    ];
    let gc = gc_plan(&index, &referenced, &HashSet::new());
    assert_eq!(
        gc.keep.len(),
        3,
        "referenced score blobs are kept (can't over-delete)"
    );
    assert_eq!(
        gc.evict_cheap.len(),
        1,
        "the orphaned jpeg is LRU-evictable"
    );
    assert_eq!(
        gc.refuse_surface.len(),
        1,
        "the orphaned irreplaceable source is NEVER auto-deleted"
    );
    assert!(gc.evict_under_pressure.is_empty());

    for f in [&sidecar1, &sidecar2, &compacted] {
        std::fs::remove_file(f).ok();
    }
}

#[test]
fn dead_worker_lease_is_reclaimed_but_live_one_is_not() {
    // A claimed job's lease expires if the worker stops heartbeating → reclaimable in minutes (goal E),
    // but a live worker that renews keeps its claim across a long job (no mid-flight steal).
    let mut lease = Lease::new("w1", 1_000, 120);
    assert!(
        !lease.can_steal(1_050),
        "live holder within ttl — not stealable"
    );

    // w1 dies (no renewal); after ttl, w2 can reclaim.
    assert!(lease.can_steal(1_120));

    // Counterfactual: had w1 heartbeated at 1_100, the claim would survive past the original window.
    lease.renew(1_100);
    assert!(!lease.can_steal(1_200), "renewed lease is held");
    assert!(
        lease.can_steal(1_221),
        "stale again relative to the last heartbeat"
    );

    // The reclaimed job is just a normal gap on the next reconcile — convergence, not a special case.
    let d = DesiredJob::new(
        JobKind::Metric {
            metric: "ssim2".into(),
        },
        vec![sha256(b"enc")],
        CellId {
            image_path: "x".into(),
            codec: "zenjpeg".into(),
            q: 1,
            knob_tuple_json: "{}".into(),
        },
    );
    let plan = reconcile(
        std::slice::from_ref(&d),
        &zenfleet_core::LedgerView::new(),
        RetryPolicy::default(),
    );
    assert_eq!(plan.enqueue, vec![d.job_id()]);
    // sanity: JobId stable across calls
    assert_eq!(JobId::of(&d.kind, &d.inputs), d.job_id());
}
