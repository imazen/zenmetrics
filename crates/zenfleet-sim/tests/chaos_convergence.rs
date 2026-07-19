//! Convergence chaos: declare → reconcile → execute to a terminal state for
//! every job, driven through the REAL `zenfleet_core::reconcile` + `RetryPolicy`,
//! with jobs that fail in different ways and a substrate that is sometimes flaky.
//!
//! The point: the engine's guarantees survive both job faults and store faults —
//! transient failures retry and recover, deterministic failures poison (stop the
//! burn), and the run converges even when the store throws transient errors mid-run.

use std::collections::HashMap;

use zenfleet_sim::{ExecResult, FaultSpec, FaultStore, JobSpec, SimClock, run_to_convergence};
use zenfleet_core::{CellId, DesiredJob, ErrorClass, JobKind, JobStatus, RetryPolicy, sha256};

// --- outcome scripts: attempt -> result ---
fn always_done(_a: u32) -> ExecResult {
    ExecResult::Done
}
fn flaky_recovers(a: u32) -> ExecResult {
    // OOM-killed on the first try, succeeds on the retry.
    if a >= 2 { ExecResult::Done } else { ExecResult::Transient(ErrorClass::Timeout) }
}
fn poison_now(_a: u32) -> ExecResult {
    // Bad bytes / NaN score — deterministic, no point retrying.
    ExecResult::Deterministic(ErrorClass::DecodeError)
}
fn always_times_out(_a: u32) -> ExecResult {
    ExecResult::Transient(ErrorClass::Timeout)
}

fn job(i: usize, script: fn(u32) -> ExecResult) -> JobSpec {
    JobSpec {
        desired: DesiredJob::new(
            JobKind::Metric { metric: "cvvdp".into() },
            vec![sha256(format!("encode-{i}").as_bytes())],
            CellId {
                image_path: format!("img/{i}.png"),
                codec: "zenjpeg".into(),
                q: 80,
                knob_tuple_json: "{}".into(),
            },
        ),
        script,
    }
}

const POLICY: RetryPolicy = RetryPolicy { max_attempts: 3 };

fn status_of<'a>(r: &'a HashMap<zenfleet_core::JobId, JobStatus>, j: &JobSpec) -> &'a JobStatus {
    r.get(&j.desired.job_id()).expect("job has a terminal status")
}

#[test]
fn clean_run_converges_every_job_done() {
    let clock = SimClock::new(0);
    let store = FaultStore::new(clock.clone(), FaultSpec::perfect(), 1);
    let jobs = [job(0, always_done), job(1, always_done), job(2, always_done)];

    let r = run_to_convergence(&store, &clock, &jobs, POLICY, &["w1"], 20);
    assert!(r.converged, "clean run converges");
    assert_eq!(r.done, 3);
    assert_eq!(r.poison, 0);
}

#[test]
fn transient_failures_retry_and_recover() {
    let clock = SimClock::new(0);
    let store = FaultStore::new(clock.clone(), FaultSpec::perfect(), 1);
    let jobs = [job(0, flaky_recovers), job(1, flaky_recovers)];

    let r = run_to_convergence(&store, &clock, &jobs, POLICY, &["w1"], 20);
    assert!(r.converged);
    assert_eq!(r.done, 2, "both recover on the retry");
    assert_eq!(r.poison, 0);
    for j in &jobs {
        assert_eq!(status_of(&r.final_status, j), &JobStatus::Done);
    }
}

#[test]
fn deterministic_failure_poisons_without_endless_retry() {
    let clock = SimClock::new(0);
    let store = FaultStore::new(clock.clone(), FaultSpec::perfect(), 1);
    let jobs = [job(0, poison_now)];

    let r = run_to_convergence(&store, &clock, &jobs, POLICY, &["w1"], 20);
    assert!(r.converged);
    assert_eq!(r.poison, 1);
    assert_eq!(status_of(&r.final_status, &jobs[0]), &JobStatus::Poison);
    // One execute + one poison-record + one converge-check = a handful of passes,
    // NOT retried to the cap.
    assert!(r.passes <= 3, "poisoned promptly, not retried forever (passes={})", r.passes);
}

#[test]
fn transient_failure_poisons_at_the_attempt_cap() {
    let clock = SimClock::new(0);
    let store = FaultStore::new(clock.clone(), FaultSpec::perfect(), 1);
    let jobs = [job(0, always_times_out)];

    let r = run_to_convergence(&store, &clock, &jobs, POLICY, &["w1"], 20);
    assert!(r.converged);
    assert_eq!(r.poison, 1, "a job that never stops timing out is poisoned at the cap");
    assert_eq!(status_of(&r.final_status, &jobs[0]), &JobStatus::Poison);
}

/// The headline: a mixed workload converges to the right terminal states even
/// when the object store throws transient errors on ~20% of ops (each just
/// defers a claim to a later pass — self-heal).
#[test]
fn converges_despite_a_flaky_store() {
    let clock = SimClock::new(0);
    let store = FaultStore::new(clock.clone(), FaultSpec::flaky(0.2), 12345);
    let jobs = [
        job(0, always_done),
        job(1, always_done),
        job(2, flaky_recovers),
        job(3, flaky_recovers),
        job(4, poison_now),
        job(5, always_times_out),
    ];

    let r = run_to_convergence(&store, &clock, &jobs, POLICY, &["solo"], 200);

    assert!(r.converged, "converged despite substrate chaos (passes={})", r.passes);
    assert_eq!(r.done, 4, "2 clean + 2 recovered");
    assert_eq!(r.poison, 2, "1 deterministic + 1 timed-out-to-cap");
    // Each job ended in exactly the right terminal state.
    for j in [&jobs[0], &jobs[1], &jobs[2], &jobs[3]] {
        assert_eq!(status_of(&r.final_status, j), &JobStatus::Done);
    }
    for j in [&jobs[4], &jobs[5]] {
        assert_eq!(status_of(&r.final_status, j), &JobStatus::Poison);
    }
    // The flaky store did force retries — proving self-heal, not a no-op.
    assert!(r.claim_errors > 0, "the flaky store induced retries that still converged");
}
