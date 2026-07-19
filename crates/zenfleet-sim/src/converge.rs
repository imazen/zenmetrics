//! Multi-pass declare → reconcile → execute convergence, over the fault store —
//! the "jobs go wrong, but the run still finishes correctly" half.
//!
//! This drives the REAL [`zenfleet_core::reconcile`] + [`zenfleet_core::RetryPolicy`]
//! over a ledger that lives in the [`FaultStore`], with per-job outcome scripts
//! that can succeed, fail transiently (retryable), or fail deterministically
//! (poison). Workers claim each gap job via the conditional claim, "execute" it
//! per its script, record a ledger row, and release the claim; the loop repeats
//! until the reconciler reports nothing left to do.
//!
//! It proves the engine's core guarantees hold even when the substrate misbehaves
//! (transient store errors just cause a claim to be retried next pass — self-heal):
//! every job reaches a terminal state, transient failures are retried and recover,
//! deterministic failures poison (stop the burn) instead of retrying forever, and
//! the run converges.

use std::collections::HashMap;

use zenfleet_cloud::{ArtifactKey, BlobStorage};
use zenfleet_core::{
    DesiredJob, ErrorClass, JobId, JobStatus, LedgerRow, LedgerView, RetryPolicy, Sha256Hex,
    reconcile, sha256,
};

use crate::claim::{ClaimOutcome, claim_conditional};
use crate::clock::SimClock;
use crate::store::FaultStore;

/// What executing a job produces on a given attempt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecResult {
    /// Completed; an output blob was produced.
    Done,
    /// A transient failure (network blip, R2 503, OOM-killed) — retryable under
    /// the attempt cap.
    Transient(ErrorClass),
    /// A deterministic failure (bad bytes, NaN score) — no point retrying.
    Deterministic(ErrorClass),
}

/// A job plus its outcome script: `attempt -> result`. `attempt` starts at 1.
pub struct JobSpec {
    pub desired: DesiredJob,
    pub script: fn(u32) -> ExecResult,
}

/// The result of running a scenario to convergence (or the pass cap).
#[derive(Clone, Debug, Default)]
pub struct ConvergeReport {
    /// Passes executed before convergence.
    pub passes: u32,
    /// Jobs terminally Done.
    pub done: usize,
    /// Jobs terminally Poison.
    pub poison: usize,
    /// Claim/store ops that failed transiently and were retried a later pass —
    /// the substrate-induced self-heal count.
    pub claim_errors: u64,
    /// `true` if the reconciler reported nothing left before the pass cap.
    pub converged: bool,
    /// Final terminal status per job id.
    pub final_status: HashMap<JobId, JobStatus>,
}

fn make_row(
    d: &DesiredJob,
    status: JobStatus,
    error_class: Option<ErrorClass>,
    output_sha: Option<Sha256Hex>,
    attempts: u32,
    ts: u64,
    worker: &str,
) -> LedgerRow {
    LedgerRow {
        job_id: d.job_id(),
        kind: d.kind.clone(),
        cell: d.cell.clone(),
        output_sha,
        status,
        error_class,
        attempts,
        ts,
        worker: worker.to_string(),
        provider: "sim".into(),
    }
}

/// Run the declare→reconcile→execute loop until convergence or `max_passes`.
///
/// `workers` rotate across passes (each releases its claim after executing, so a
/// retry can be picked up by any worker). The ledger is kept latest-wins in a map
/// and rebuilt into a [`LedgerView`] each pass — the same shape the real
/// per-chunk-sidecar + compaction produces.
pub fn run_to_convergence(
    store: &FaultStore,
    clock: &SimClock,
    jobs: &[JobSpec],
    policy: RetryPolicy,
    workers: &[&str],
    max_passes: u32,
) -> ConvergeReport {
    let desired: Vec<DesiredJob> = jobs.iter().map(|j| j.desired.clone()).collect();
    let by_id: HashMap<JobId, &JobSpec> = jobs.iter().map(|j| (j.desired.job_id(), j)).collect();

    let mut ledger: HashMap<JobId, LedgerRow> = HashMap::new();
    let mut report = ConvergeReport::default();

    loop {
        let view = LedgerView::from_rows(ledger.values().cloned());
        let plan = reconcile(&desired, &view, policy);

        if plan.enqueue.is_empty() && plan.poison.is_empty() {
            report.converged = true;
            break;
        }
        if report.passes >= max_passes {
            break;
        }
        let worker = workers[report.passes as usize % workers.len()];
        report.passes += 1;

        // Execute the gap.
        for id in &plan.enqueue {
            let spec = by_id[id];
            let sidecar = format!("out/{}", id.as_str());
            let claim_key = format!("claims/{}", id.as_str());

            match claim_conditional(store, worker, &sidecar, &claim_key, 600) {
                ClaimOutcome::Acquired => {}
                ClaimOutcome::Errored => {
                    report.claim_errors += 1; // transient store error → retry next pass
                    continue;
                }
                // AlreadyDone / HeldByPeer / LostRace: someone else is on it.
                _ => continue,
            }

            let attempts = view.get(id).map(|r| r.attempts + 1).unwrap_or(1);
            let row = match (spec.script)(attempts) {
                ExecResult::Done => {
                    let out = sha256(format!("{}-{attempts}", id.as_str()).as_bytes());
                    make_row(&spec.desired, JobStatus::Done, None, Some(out), attempts, clock.now(), worker)
                }
                ExecResult::Transient(e) => {
                    make_row(&spec.desired, JobStatus::Failed, Some(e), None, attempts, clock.now(), worker)
                }
                ExecResult::Deterministic(e) => {
                    make_row(&spec.desired, JobStatus::Failed, Some(e), None, attempts, clock.now(), worker)
                }
            };
            ledger.insert(id.clone(), row);
            // Release the claim so a retry (if this was a failure) is free to any
            // worker next pass. A real claim would age out; here we release it.
            let _ = store.delete(&ArtifactKey(claim_key));
        }

        // Record the reconciler's poison decisions (the caller writes these).
        for id in &plan.poison {
            let spec = by_id[id];
            let prev = ledger.get(id);
            let attempts = prev.map(|r| r.attempts).unwrap_or(1);
            let err = prev.and_then(|r| r.error_class);
            let row = make_row(&spec.desired, JobStatus::Poison, err, None, attempts, clock.now(), worker);
            ledger.insert(id.clone(), row);
        }

        clock.advance(10); // time between reconcile passes
    }

    for (id, row) in &ledger {
        match row.status {
            JobStatus::Done => report.done += 1,
            JobStatus::Poison => report.poison += 1,
            _ => {}
        }
        report.final_status.insert(id.clone(), row.status);
    }
    report
}
