//! Phase 5 integration tests — streaming + batch API + worker pool.
//!
//! Each test that touches the GPU is `#[ignore]`d so it doesn't run
//! during `cargo test --workspace` (which would consume real VRAM and
//! invoke nvidia-smi). Run them manually with:
//!
//! ```bash
//! cargo test --features cuda -p zenmetrics-orchestrator \
//!     --test streaming -- --ignored --nocapture
//! ```
//!
//! ## Sibling-workspace caveat
//!
//! When this is built from a `zenmetrics--*` jj sibling workspace,
//! cargo discovers two `butteraugli-gpu` crates via jxl-encoder's
//! hardcoded `../zenmetrics/crates/butteraugli-gpu` path and emits a
//! lockfile collision. Build from the primary `zenmetrics/` checkout
//! instead — same workaround as the Phase 4 `executor` tests.

#![cfg(all(feature = "bench", feature = "cuda"))]

use std::time::{Duration, Instant};

use zenmetrics_api::MetricKind;
use zenmetrics_orchestrator::{
    synth_pair_offset_dist, Orchestrator, OrchestratorConfig, Task, TaskData,
};

/// Build an orchestrator, warm the cache if it's empty, and return it.
/// Used by every test that needs a populated capability profile.
fn make_warm_orchestrator() -> Orchestrator {
    let mut orch =
        Orchestrator::new(OrchestratorConfig::default()).expect("Orchestrator::new");
    // Warm if no metrics cached yet. If a prior bench populated the
    // cache the test runs against that snapshot.
    if orch.capability().metrics.is_empty() {
        orch.warm().expect("warm bench");
    }
    orch
}

/// Build N synthetic cvvdp tasks at `size×size` with distinct task_ids.
fn build_cvvdp_tasks(n: usize, size: u32) -> Vec<Task> {
    let (r, d) = synth_pair_offset_dist(size, size);
    (0..n)
        .map(|i| Task {
            task_id: 1000 + i as u64,
            ref_data: TaskData::Srgb8(r.clone()),
            dist_data: TaskData::Srgb8(d.clone()),
            width: size,
            height: size,
            metric: MetricKind::Cvvdp,
            params: None,
        })
        .collect()
}

#[test]
#[ignore = "requires CUDA + populated capability cache"]
fn streaming_submit_poll_completes_all() {
    let mut orch = make_warm_orchestrator();
    let tasks = build_cvvdp_tasks(10, 1024);
    let total = tasks.len();

    // Submit all, capture handles.
    let mut handles = Vec::with_capacity(total);
    let mut expected_task_ids: Vec<u64> = Vec::with_capacity(total);
    for t in tasks {
        expected_task_ids.push(t.task_id);
        let h = orch.submit(t).expect("submit");
        handles.push(h);
    }

    // Drain via poll_any_blocking until every handle returns a result.
    let mut got: Vec<u64> = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(180);
    while got.len() < total {
        if Instant::now() > deadline {
            panic!("streaming drain timed out at {}/{}", got.len(), total);
        }
        if let Some(result) = orch.poll_any_blocking() {
            assert!(
                expected_task_ids.contains(&result.task_id),
                "unexpected task_id {}",
                result.task_id
            );
            match result.outcome {
                Ok(score) => {
                    assert!(score.value.is_finite(), "non-finite score");
                }
                Err(e) => panic!("task {} failed: {e}", result.task_id),
            }
            got.push(result.task_id);
        }
    }

    // Every expected task_id arrived.
    got.sort_unstable();
    let mut expected_sorted = expected_task_ids.clone();
    expected_sorted.sort_unstable();
    assert_eq!(got, expected_sorted);
}

#[test]
#[ignore = "requires CUDA + populated capability cache"]
fn batch_run_all_yields_all_tasks() {
    let mut orch = make_warm_orchestrator();
    let tasks = build_cvvdp_tasks(20, 1024);
    let total = tasks.len();

    let results: Vec<_> = orch.run_all(tasks).collect();
    assert_eq!(results.len(), total);
    // Every result should reference a task_id in 1000..1020.
    for r in &results {
        assert!((1000..1020).contains(&r.task_id));
    }
    let ok_count = results.iter().filter(|r| r.outcome.is_ok()).count();
    assert_eq!(ok_count, total, "every task should produce a score");
}

#[test]
#[ignore = "requires CUDA + populated capability cache"]
fn worker_pool_parallel_speedup() {
    // Measure one task's time, then ten tasks via run_all. The brief
    // expects ≥1.5× speedup; with a single GPU the bound is the
    // overlap between PTX-warmup and dist upload. Skip this assertion
    // if the single-task time is implausibly large (>5s) — that's
    // PTX-compile dominated and parallel isn't the right benchmark.
    let mut orch = make_warm_orchestrator();

    // Single-task baseline.
    let mut singles = build_cvvdp_tasks(1, 1024);
    let t0 = Instant::now();
    let r = orch.run_all(singles.drain(..)).collect::<Vec<_>>();
    let single_us = t0.elapsed().as_micros() as u64;
    assert_eq!(r.len(), 1);
    assert!(r[0].outcome.is_ok(), "single task failed: {:?}", r[0].outcome);

    // Ten tasks (same ref/dist so PTX is warm, signature reused on
    // worker side — measures pure dispatch latency + GPU work overlap).
    let many = build_cvvdp_tasks(10, 1024);
    let t1 = Instant::now();
    let rs = orch.run_all(many).collect::<Vec<_>>();
    let ten_us = t1.elapsed().as_micros() as u64;
    assert_eq!(rs.len(), 10);
    let ok = rs.iter().filter(|r| r.outcome.is_ok()).count();
    assert_eq!(ok, 10);

    // Effective speedup. 10 tasks / wall_time vs 1 task / wall_time.
    // A perfectly serial worker would give 1.0×; the brief expects
    // ≥1.5× from PTX/buffer reuse alone. Print numbers regardless.
    let speedup = (10.0 * single_us as f64) / ten_us as f64;
    eprintln!(
        "single_us={single_us}  ten_us={ten_us}  speedup={speedup:.2}× (target ≥1.5×)"
    );
    // Don't hard-fail when single_us > 5s — that's PTX-compile noise.
    if single_us < 5_000_000 {
        assert!(
            speedup >= 1.5,
            "parallel speedup {speedup:.2}× < 1.5× (single={single_us}us, ten={ten_us}us)"
        );
    }
}

#[test]
#[ignore = "requires CUDA + populated capability cache"]
fn live_vram_watcher_reports_a_value() {
    // We can't easily inject a low-VRAM scenario without a mock
    // backend. What we *can* test is that the watcher thread runs and
    // populates the snapshot after one sample interval.
    let mut orch = make_warm_orchestrator();
    // Submitting one task forces the pool to initialize, which spawns
    // the watcher.
    let task = build_cvvdp_tasks(1, 256)
        .into_iter()
        .next()
        .expect("one task");
    let h = orch.submit(task).expect("submit");
    // Drain the result so the worker is idle.
    let _ = orch.poll_any_blocking().expect("one result");
    let _ = h;

    // Sleep one watcher-interval-plus-a-bit to ensure the snapshot
    // got at least one real probe (default 250 ms).
    std::thread::sleep(Duration::from_millis(400));
    let mib = orch.vram_watcher_mib().expect("watcher running");
    // First probe should produce something less than usize::MAX (the
    // initial sentinel). 0 is allowed (means the probe ran but
    // returned 0 free) but usize::MAX means the probe never fired.
    assert_ne!(mib, usize::MAX, "watcher never produced a sample");
    eprintln!("vram_watcher_mib = {mib}");
}
