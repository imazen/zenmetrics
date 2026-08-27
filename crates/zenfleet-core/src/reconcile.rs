//! The reconciler: desired-vs-actual over the ledger. It is the engine's beating heart —
//!
//! - **Enqueue only the gap** (goals A & I): jobs with no terminal row. Declaring already-done work
//!   is a no-op because identity is content-addressed.
//! - **Self-heal** (goal E): the ledger is truth; a dropped queue message is re-found next pass.
//! - **Retry transient, poison deterministic** (goal F): a transient failure under the attempt cap
//!   is re-enqueued; a deterministic failure (bad bytes, panic, NaN) or one over the cap becomes
//!   POISON so doomed work stops burning money.
//!
//! Pure function — no I/O. The Parquet/queue wiring calls this and acts on the returned plan.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::ids::JobId;
use crate::ledger::{DesiredJob, LedgerView};
use crate::status::JobStatus;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RetryPolicy {
    /// Max delivery attempts before a transient failure is poisoned.
    pub max_attempts: u32,
    /// When set, a `Claimed`/`Pending` row whose age (`now - ts`) has reached this many seconds is
    /// treated as abandoned and re-enqueued by [`reconcile_at`] instead of counting `in_flight`
    /// forever (the 2026-08-26 stuck-29 deadlock: a dead worker's claim row pins its cell because
    /// the claim-time TTL steal only fires for cells that get enqueued). Callers wire the claim
    /// TTL here; `None` keeps the old behavior bit-for-bit (and [`reconcile`] always behaves as
    /// `None` — it passes no clock).
    pub stale_claim_after: Option<u64>,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            stale_claim_after: None,
        }
    }
}

/// What the reconciler decided. The caller dispatches `enqueue` to the queue and writes `poison`
/// rows to the ledger.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReconcilePlan {
    /// Jobs to (re)dispatch — the gap.
    pub enqueue: Vec<JobId>,
    /// Jobs to mark POISON (stop retrying).
    pub poison: Vec<JobId>,
    /// Already-complete, skipped (idempotent).
    pub done: usize,
    /// Claimed/pending, not yet terminal.
    pub in_flight: usize,
}

/// Compute the gap between what's desired and what the ledger shows. Desired entries that resolve to
/// the same [`JobId`] are de-duplicated, so declaring overlapping work never double-enqueues.
pub fn reconcile(desired: &[DesiredJob], view: &LedgerView, policy: RetryPolicy) -> ReconcilePlan {
    reconcile_at(desired, view, policy, 0, None)
}

/// [`reconcile`] with claim-staleness awareness (the 2026-08-26 stuck-29 fix).
///
/// A `Claimed` row is normally `in_flight` — never enqueued, so no claim attempt happens, so the
/// claim-time TTL steal ([`claim_is_stale`]'s consumer) can never fire for it: a dead worker's
/// claim row pins its cell in_flight FOREVER. This deadlocked 29 sf2-cpu cells on 2026-08-26
/// (broken-image worker claimed chunks, died mid-era; its cells were unreachable until the claim
/// objects were deleted by hand). With `stale_claim_after = Some(ttl)` a Claimed/Pending row whose
/// age (`now - ts`) has reached the TTL is treated as abandoned and re-enqueued — the same
/// staleness rule the claim-object steal uses, applied where the decision actually gets made.
/// `None` (or `reconcile`) keeps the old behavior bit-for-bit.
pub fn reconcile_at(
    desired: &[DesiredJob],
    view: &LedgerView,
    policy: RetryPolicy,
    now: u64,
    stale_claim_after: Option<u64>,
) -> ReconcilePlan {
    let mut plan = ReconcilePlan::default();
    let mut seen: HashSet<JobId> = HashSet::new();

    for d in desired {
        let id = d.job_id();
        if !seen.insert(id.clone()) {
            continue; // duplicate desired entry — count once
        }
        match view.get(&id) {
            None => plan.enqueue.push(id), // never seen → the gap
            Some(r) => match r.status {
                JobStatus::Done => plan.done += 1,
                JobStatus::Poison => {} // already given up; recorded
                JobStatus::Pending | JobStatus::Claimed => {
                    match stale_claim_after {
                        // Abandoned in-flight row (owner presumed dead) → back into the gap.
                        Some(ttl) if now.saturating_sub(r.ts) >= ttl => plan.enqueue.push(id),
                        _ => plan.in_flight += 1,
                    }
                }
                JobStatus::Failed => {
                    let transient = r.error_class.map(|e| e.is_transient()).unwrap_or(false);
                    if transient && r.attempts < policy.max_attempts {
                        plan.enqueue.push(id); // retry
                    } else {
                        plan.poison.push(id); // deterministic, or over the cap → stop the burn
                    }
                }
            },
        }
    }
    plan
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::content::sha256;
    use crate::ids::CellId;
    use crate::job::JobKind;
    use crate::ledger::{LedgerRow, LedgerView};
    use crate::status::ErrorClass;

    fn desired(metric: &str, enc: &[u8]) -> DesiredJob {
        DesiredJob {
            requires: vec![],
            kind: JobKind::Metric {
                metric: metric.into(),
            },
            inputs: vec![sha256(enc)],
            cell: CellId {
                image_path: "x.png".into(),
                codec: "zenjpeg".into(),
                q: 80,
                knob_tuple_json: "{}".into(),
            },
            hint: None,
        }
    }

    fn row(id: JobId, status: JobStatus, err: Option<ErrorClass>, attempts: u32) -> LedgerRow {
        LedgerRow {
            job_id: id,
            kind: JobKind::Metric {
                metric: "cvvdp".into(),
            },
            cell: CellId {
                image_path: "x.png".into(),
                codec: "zenjpeg".into(),
                q: 80,
                knob_tuple_json: "{}".into(),
            },
            output_sha: None,
            status,
            error_class: err,
            attempts,
            ts: 1,
            worker: "w".into(),
            provider: "local".into(),
        }
    }

    #[test]
    fn empty_ledger_enqueues_everything() {
        let d = vec![desired("cvvdp", b"a"), desired("ssim2", b"a")];
        let plan = reconcile(&d, &LedgerView::new(), RetryPolicy::default());
        assert_eq!(plan.enqueue.len(), 2);
        assert_eq!(plan.done, 0);
    }

    #[test]
    fn done_is_skipped_not_reenqueued() {
        let d = desired("cvvdp", b"a");
        let view = LedgerView::from_rows([row(d.job_id(), JobStatus::Done, None, 1)]);
        let plan = reconcile(&[d], &view, RetryPolicy::default());
        assert!(plan.enqueue.is_empty());
        assert_eq!(plan.done, 1);
    }

    #[test]
    fn transient_failure_under_cap_retries() {
        let d = desired("cvvdp", b"a");
        let view = LedgerView::from_rows([row(
            d.job_id(),
            JobStatus::Failed,
            Some(ErrorClass::Timeout),
            1,
        )]);
        let plan = reconcile(
            std::slice::from_ref(&d),
            &view,
            RetryPolicy {
                max_attempts: 3,
                stale_claim_after: None,
            },
        );
        assert_eq!(plan.enqueue, vec![d.job_id()]);
        assert!(plan.poison.is_empty());
    }

    #[test]
    fn transient_failure_at_cap_poisons() {
        let d = desired("cvvdp", b"a");
        let view = LedgerView::from_rows([row(
            d.job_id(),
            JobStatus::Failed,
            Some(ErrorClass::Timeout),
            3,
        )]);
        let plan = reconcile(
            std::slice::from_ref(&d),
            &view,
            RetryPolicy {
                max_attempts: 3,
                stale_claim_after: None,
            },
        );
        assert_eq!(plan.poison, vec![d.job_id()]);
        assert!(plan.enqueue.is_empty());
    }

    #[test]
    fn deterministic_failure_poisons_immediately() {
        let d = desired("cvvdp", b"a");
        let view = LedgerView::from_rows([row(
            d.job_id(),
            JobStatus::Failed,
            Some(ErrorClass::DecodeError),
            1, // well under cap, but deterministic → no point retrying
        )]);
        let plan = reconcile(
            std::slice::from_ref(&d),
            &view,
            RetryPolicy {
                max_attempts: 3,
                stale_claim_after: None,
            },
        );
        assert_eq!(plan.poison, vec![d.job_id()]);
        assert!(plan.enqueue.is_empty());
    }

    #[test]
    fn claimed_counts_as_in_flight() {
        let d = desired("cvvdp", b"a");
        let view = LedgerView::from_rows([row(d.job_id(), JobStatus::Claimed, None, 1)]);
        let plan = reconcile(&[d], &view, RetryPolicy::default());
        assert_eq!(plan.in_flight, 1);
        assert!(plan.enqueue.is_empty());
    }

    #[test]
    fn duplicate_desired_enqueues_once() {
        let d = desired("cvvdp", b"a");
        let plan = reconcile(
            &[d.clone(), d.clone(), d],
            &LedgerView::new(),
            RetryPolicy::default(),
        );
        assert_eq!(
            plan.enqueue.len(),
            1,
            "content-addressed dedup: same work declared thrice = one enqueue"
        );
    }

    /// The stuck-29 class (2026-08-26): a Claimed row from a dead worker must become enqueueable
    /// once its age reaches the TTL — and must stay in_flight while fresh.
    #[test]
    fn stale_claimed_row_reenqueues_at_ttl() {
        let d = desired("cvvdp", b"enc");
        let mut r = row(d.job_id(), JobStatus::Claimed, None, 1);
        r.ts = 1_000;
        let view = LedgerView::from_rows([r]);
        let ds = [d.clone()];

        // Fresh (age < ttl): in_flight, nothing enqueued.
        let p = reconcile_at(&ds, &view, RetryPolicy::default(), 1_500, Some(600));
        assert_eq!(p.in_flight, 1);
        assert!(p.enqueue.is_empty());

        // Stale (age >= ttl): back into the gap.
        let p = reconcile_at(&ds, &view, RetryPolicy::default(), 1_600, Some(600));
        assert_eq!(p.in_flight, 0);
        assert_eq!(p.enqueue, vec![d.job_id()]);

        // Staleness disabled: old behavior exactly (in_flight forever).
        let p = reconcile_at(&ds, &view, RetryPolicy::default(), 9_999_999, None);
        assert_eq!(p.in_flight, 1);
        assert!(p.enqueue.is_empty());

        // The plain reconcile() wrapper keeps the old behavior bit-for-bit.
        let p = reconcile(&ds, &view, RetryPolicy::default());
        assert_eq!(p.in_flight, 1);
        assert!(p.enqueue.is_empty());
    }
}
