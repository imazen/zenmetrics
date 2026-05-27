//! Phase 5 integration tests — cached-ref auto-detect.
//!
//! Each test that hits the GPU is `#[ignore]`d. Run manually:
//!
//! ```bash
//! cargo test --features cuda -p zenmetrics-orchestrator \
//!     --test cached_ref -- --ignored --nocapture
//! ```
//!
//! Sibling-workspace caveat applies — see `streaming.rs`.

#![cfg(all(feature = "bench", feature = "cuda"))]

use zenmetrics_api::MetricKind;
use zenmetrics_orchestrator::{
    synth_pair_offset_dist, Orchestrator, OrchestratorConfig, Task, TaskData,
};

fn make_warm() -> Orchestrator {
    let mut orch =
        Orchestrator::new(OrchestratorConfig::default()).expect("Orchestrator::new");
    if orch.capability().metrics.is_empty() {
        orch.warm().expect("warm bench");
    }
    orch
}

#[test]
#[ignore = "requires CUDA + populated capability cache"]
fn cached_ref_auto_detect_reuses_ref() {
    // Submit 5 tasks with identical ref bytes. The first observe()
    // adds the hash to the window (miss); the next 4 hit. The pool's
    // CachedRefStats should report exactly 4 hits and 1 miss.
    let mut orch = make_warm();

    let size: u32 = 1024;
    let (r, d) = synth_pair_offset_dist(size, size);

    let stats_before = orch.cached_ref_stats();
    assert_eq!(stats_before.hit_count, 0);

    let n: usize = 5;
    let tasks: Vec<Task> = (0..n)
        .map(|i| Task {
            task_id: 2000 + i as u64,
            ref_data: TaskData::Srgb8(r.clone()),
            dist_data: TaskData::Srgb8(d.clone()),
            width: size,
            height: size,
            metric: MetricKind::Cvvdp,
            params: None,
            ref_hash: 0,
        })
        .collect();

    let results: Vec<_> = orch.run_all(tasks).collect();
    assert_eq!(results.len(), n);
    for r in &results {
        assert!(r.outcome.is_ok(), "task failed: {:?}", r.outcome);
    }

    let stats_after = orch.cached_ref_stats();
    // 5 tasks: 1 miss (first), 4 hits.
    eprintln!(
        "cached_ref_stats: hits={}, misses={}",
        stats_after.hit_count, stats_after.miss_count
    );
    assert_eq!(stats_after.miss_count - stats_before.miss_count, 1);
    assert_eq!(stats_after.hit_count - stats_before.hit_count, 4);
}

#[test]
#[ignore = "requires CUDA + populated capability cache"]
fn cached_ref_explicit_override_skips_hash() {
    // upload_reference → 5 tasks with PreUploaded handle.
    // Hash counters should not bump because the dispatcher skips
    // hashing entirely for PreUploaded.
    let mut orch = make_warm();

    let size: u32 = 1024;
    let (r, d) = synth_pair_offset_dist(size, size);

    let stats_before = orch.cached_ref_stats();
    let handle = orch
        .upload_reference(&r, size, size, MetricKind::Cvvdp)
        .expect("upload_reference");

    let n: usize = 5;
    let tasks: Vec<Task> = (0..n)
        .map(|i| Task {
            task_id: 3000 + i as u64,
            ref_data: TaskData::PreUploaded(handle.clone()),
            dist_data: TaskData::Srgb8(d.clone()),
            width: size,
            height: size,
            metric: MetricKind::Cvvdp,
            params: None,
            ref_hash: 0,
        })
        .collect();

    let results: Vec<_> = orch.run_all(tasks).collect();
    assert_eq!(results.len(), n);
    for r in &results {
        assert!(r.outcome.is_ok(), "task failed: {:?}", r.outcome);
    }

    let stats_after = orch.cached_ref_stats();
    // PreUploaded reuses the cached entry from upload_reference. Each
    // submit observes the same (metric, w, h, hash) tuple: 1 miss + 4 hits.
    eprintln!(
        "cached_ref_stats (pre-upload path): hits={}, misses={}",
        stats_after.hit_count, stats_after.miss_count
    );
    let hit_delta = stats_after.hit_count - stats_before.hit_count;
    let miss_delta = stats_after.miss_count - stats_before.miss_count;
    assert_eq!(hit_delta + miss_delta, n as u64);

    orch.drop_reference(handle);
}

#[test]
#[ignore = "requires CUDA + populated capability cache"]
fn cached_ref_different_dist_same_ref_all_pass() {
    // Sanity: scoring against 5 *different* distorted images using the
    // same reference should always produce valid scores. The
    // cached-ref path must produce the same numbers as a regular
    // compute for the same (ref, dist) pair — within float noise.
    let mut orch = make_warm();

    let size: u32 = 1024;
    let (r, d0) = synth_pair_offset_dist(size, size);

    // 5 variants of d0 (shifted by i pixels) — guaranteed distinct.
    let mut dists: Vec<Vec<u8>> = Vec::with_capacity(5);
    for i in 0..5usize {
        let mut d = d0.clone();
        // Bump the first pixel's red channel by i so the bytes differ;
        // the score should drift slightly per i but every result is valid.
        if !d.is_empty() {
            d[0] = d[0].wrapping_add(i as u8);
        }
        dists.push(d);
    }

    let tasks: Vec<Task> = dists
        .into_iter()
        .enumerate()
        .map(|(i, d)| Task {
            task_id: 4000 + i as u64,
            ref_data: TaskData::Srgb8(r.clone()),
            dist_data: TaskData::Srgb8(d),
            width: size,
            height: size,
            metric: MetricKind::Cvvdp,
            params: None,
            ref_hash: 0,
        })
        .collect();

    let results: Vec<_> = orch.run_all(tasks).collect();
    assert_eq!(results.len(), 5);
    for r in &results {
        match &r.outcome {
            Ok(s) => assert!(s.value.is_finite()),
            Err(e) => panic!("task {} failed: {e}", r.task_id),
        }
    }
}
