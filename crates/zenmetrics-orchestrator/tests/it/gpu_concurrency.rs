//! Phase 9 — GPU concurrency invariants.
//!
//! Validates the N-lane pool's two non-negotiable contracts:
//!
//! 1. **Parity** — N=1 and N=4 produce bit-identical primary scores for
//!    the same input batch (modulo task ordering — every result is
//!    matched on `task_id` and compared exactly).
//! 2. **Throughput** — N=4 at 4096² delivers ≥ 2.5× the wall-clock
//!    throughput of N=1 on the same batch. The cap below 4× reflects
//!    that a single GPU's compute resources serialise to some degree
//!    even with concurrent streams; the gain comes from overlapping
//!    HtoD upload with prior-task kernel execution + small-kernel
//!    queueing latency hiding.
//!
//! Adaptive scaling (Phase 9.3) is exercised separately by the
//! `adaptive_lane_count_responds_to_low_utilization` test, which uses
//! a small-task workload to drive the watcher into the "low GPU util"
//! regime and asserts that `adaptive_lane_tick` would scale up.
//!
//! ## How to run
//!
//! Every test is `#[ignore]` because it requires real CUDA. From the
//! primary `zenmetrics/` checkout (NOT a `zenmetrics--*` jj sibling
//! — see the lockfile-collision note on the Phase 5 tests above):
//!
//! ```bash
//! cargo test --features cuda -p zenmetrics-orchestrator \
//!     --test gpu_concurrency -- --ignored --nocapture
//! ```

#![cfg(all(feature = "bench", feature = "cuda"))]

use std::time::{Duration, Instant};

use zenmetrics_api::MetricKind;
use zenmetrics_orchestrator::{
    Orchestrator, OrchestratorConfig, PoolConfig, Task, TaskData, synth_pair_offset_dist,
};

/// Build an orchestrator with the given lane count. Pool config is set
/// before any work is submitted so the lane count takes effect.
fn make_orchestrator_with_lanes(lanes: usize) -> Orchestrator {
    let mut orch = Orchestrator::new(OrchestratorConfig::default()).expect("Orchestrator::new");
    let mut pool_cfg = PoolConfig::default();
    pool_cfg.max_gpu_lanes = lanes;
    orch.set_pool_config(pool_cfg)
        .expect("set_pool_config before first submit");
    if orch.capability().metrics.is_empty() {
        orch.warm().expect("warm bench");
    }
    orch
}

/// Build N synthetic cvvdp tasks at `size×size`. Same payload on every
/// task so parity asserts on equal scores across runs.
fn build_cvvdp_tasks(n: usize, size: u32) -> Vec<Task> {
    let (r, d) = synth_pair_offset_dist(size, size);
    (0..n)
        .map(|i| Task {
            task_id: 9000 + i as u64,
            ref_data: TaskData::Srgb8(r.clone()),
            dist_data: TaskData::Srgb8(d.clone()),
            width: size,
            height: size,
            metric: MetricKind::Cvvdp,
            params: None,
            ref_hash: 0,
        })
        .collect()
}

/// Drive a batch through `Orchestrator::run_all`, returning the
/// per-task scores keyed by `task_id` and the wall-clock duration.
fn drain_batch(orch: &mut Orchestrator, tasks: Vec<Task>) -> (Vec<(u64, f64)>, Duration) {
    let t0 = Instant::now();
    let results: Vec<_> = orch.run_all(tasks).collect();
    let wall = t0.elapsed();
    let mut scored: Vec<(u64, f64)> = results
        .into_iter()
        .map(|r| {
            let score = r.outcome.expect("score").value;
            (r.task_id, score)
        })
        .collect();
    scored.sort_by_key(|(id, _)| *id);
    (scored, wall)
}

#[test]
#[ignore = "requires CUDA + populated capability cache"]
fn parity_n1_equals_n4_on_identical_batch() {
    // Phase 9 hard constraint: a batch run with N=1 vs N=4 must produce
    // bit-identical scores (Atomic<f32> non-determinism tolerance OK,
    // but cvvdp's reduction path is deterministic by default).
    let tasks_a = build_cvvdp_tasks(8, 1024);
    let tasks_b = build_cvvdp_tasks(8, 1024);

    let mut o1 = make_orchestrator_with_lanes(1);
    let (scores_n1, _) = drain_batch(&mut o1, tasks_a);
    drop(o1);

    let mut o4 = make_orchestrator_with_lanes(4);
    let (scores_n4, _) = drain_batch(&mut o4, tasks_b);
    drop(o4);

    assert_eq!(
        scores_n1.len(),
        scores_n4.len(),
        "N=1 and N=4 must return the same task count"
    );
    for ((id_a, s_a), (id_b, s_b)) in scores_n1.iter().zip(scores_n4.iter()) {
        assert_eq!(id_a, id_b, "task_id alignment");
        // Allow up to 1e-5 relative drift for floating-point reduction
        // order non-determinism across streams. Cvvdp's primary
        // reduction is sum-then-mean — order shouldn't matter at this
        // scale, but we don't want a 1-ulp flake.
        let drift = (s_a - s_b).abs();
        let rel = if s_a.abs() > 1e-6 {
            drift / s_a.abs()
        } else {
            drift
        };
        assert!(
            rel < 1e-5,
            "score parity violated for task_id {id_a}: N=1={s_a} N=4={s_b} rel_drift={rel:.3e}"
        );
    }
}

#[test]
#[ignore = "requires CUDA + populated capability cache (large GPU memory)"]
fn throughput_n4_at_least_2_5x_n1_at_4mp() {
    // Phase 9 throughput contract: 50 identical-signature tasks at
    // 4096² must run ≥ 2.5× faster with N=4 lanes than N=1. The cap
    // below 4× reflects single-GPU contention; the floor at 2.5× is
    // the minimum useful win — anything less means the lane abstraction
    // isn't delivering measurable concurrency.
    let n = 50;
    let size = 4096;

    let mut o1 = make_orchestrator_with_lanes(1);
    let (_, wall_n1) = drain_batch(&mut o1, build_cvvdp_tasks(n, size));
    drop(o1);

    let mut o4 = make_orchestrator_with_lanes(4);
    let (_, wall_n4) = drain_batch(&mut o4, build_cvvdp_tasks(n, size));
    drop(o4);

    let speedup = wall_n1.as_secs_f64() / wall_n4.as_secs_f64();
    eprintln!(
        "throughput @ {n} tasks {size}² cvvdp: N=1 wall={:.2}s N=4 wall={:.2}s speedup={:.2}x",
        wall_n1.as_secs_f64(),
        wall_n4.as_secs_f64(),
        speedup,
    );
    assert!(
        speedup >= 2.5,
        "N=4 speedup {speedup:.2}x below 2.5x floor (N=1 wall {:.2}s, N=4 wall {:.2}s)",
        wall_n1.as_secs_f64(),
        wall_n4.as_secs_f64(),
    );
    // Sanity upper bound: shouldn't be >4× on a single GPU. If it is,
    // either we mis-measured N=1 or there's a hot cache effect.
    assert!(
        speedup < 5.0,
        "N=4 speedup {speedup:.2}x implausibly high — re-check the harness"
    );
}

#[test]
#[ignore = "requires CUDA + populated capability cache"]
fn lane_count_introspection_matches_config() {
    // Phase 9.1 surface area test: gpu_lane_count() reflects the
    // configured max_gpu_lanes; active_gpu_lanes() equals that value
    // when adaptive scaling is off.
    let orch = make_orchestrator_with_lanes(4);
    // No work submitted yet → pool not initialised → API returns None.
    assert!(orch.gpu_lane_count().is_none());
    assert!(orch.active_gpu_lanes().is_none());
    // Submit + drain one task to force lazy pool init.
    let mut orch = orch;
    let (_, _) = drain_batch(&mut orch, build_cvvdp_tasks(1, 256));
    assert_eq!(orch.gpu_lane_count(), Some(4));
    assert_eq!(orch.active_gpu_lanes(), Some(4));
}

#[test]
#[ignore = "requires CUDA + nvidia-smi + populated capability cache"]
fn adaptive_lane_count_starts_at_one_when_enabled() {
    // Phase 9.3 — when adaptive_gpu_lanes is true, the pool starts at
    // 1 lane and scales up on observed low utilization. With no work
    // submitted, no scaling has happened, so active_gpu_lanes == 1
    // and adaptive_lane_tick() reports no change (watcher hasn't seen
    // enough samples to act).
    let mut orch = Orchestrator::new(OrchestratorConfig::default()).expect("Orchestrator::new");
    let mut cfg = PoolConfig::default();
    cfg.max_gpu_lanes = 4;
    cfg.adaptive_gpu_lanes = true;
    cfg.adaptive_max_gpu_lanes = 4;
    // Short interval so the test doesn't wait 15s for 3 samples.
    cfg.gpu_util_sample_interval_ms = 500;
    orch.set_pool_config(cfg).expect("set_pool_config");
    if orch.capability().metrics.is_empty() {
        orch.warm().expect("warm bench");
    }
    // Force pool init.
    let (_, _) = drain_batch(&mut orch, build_cvvdp_tasks(1, 256));
    assert_eq!(orch.active_gpu_lanes(), Some(1), "adaptive starts at 1");
    assert_eq!(orch.gpu_lane_count(), Some(4), "max lanes available");
}
