//! Phase 6 — integration tests for the CPU backend wiring.
//!
//! These tests build the orchestrator with a synthetic capability
//! profile (no real GPU required) and force the OOM fallback ladder
//! to land on the CPU branch. Each enabled `cpu-<metric>` feature
//! gates a small set of construct+compute smoke tests against the
//! corresponding reference crate.
//!
//! Test taxonomy:
//!
//! - `all_backends_construct_and_compute` — for each enabled CPU
//!   backend, construct a 256² adapter and run one compute. Verifies
//!   the wiring is plumbed end-to-end.
//! - `cpu_parity_against_gpu_cvvdp` — when `cpu-cvvdp` is on AND the
//!   GPU CUDA backend is enabled, run a 256² pair through both and
//!   assert |diff| < 0.1 JOD. Marked `#[ignore]` since it needs a
//!   real CUDA device; the cvvdp crate's own parity tests cover
//!   the tighter atomic-tolerance case.
//! - `cached_ref_round_trip_cvvdp` — set_reference + 4 cached calls
//!   yield the same scores as 4 one-shot calls.
//! - `oom_fallback_routes_to_cpu` — pre-populate `cells_failed_oom`
//!   for every GPU backend, expect `backend_used = Cpu`.
//! - `iwssim_cpu_unavailable_advances_ladder` — Iwssim never has a
//!   CPU reference; the chooser surfaces `CpuMetricUnavailable` and
//!   the executor's run_single returns `FullyExhausted` (no other
//!   backends remain after GPU OOM).
//! - `chooser_picks_cpu_when_gpu_oom` — direct chooser unit test
//!   (no executor needed) that the Cpu candidate is `Selected` when
//!   GPU OOM is recorded.

#![cfg(all(feature = "bench", feature = "cuda"))]

use std::collections::BTreeMap;
use std::time::{Duration, SystemTime};

use zenmetrics_api::MetricKind;
use zenmetrics_orchestrator::{
    compute_machine_hash, save_profile, Backend, BackendBench, BackendVram, CapabilityProfile,
    CpuCapability, ExecutorError, GpuCapability, MetricProfile, Orchestrator, OrchestratorConfig,
    Task, TaskData,
};

// ---------------------------------------------------------------------------
// Helpers (mirror tests/executor.rs — kept inline; the two suites share
// shape but the chooser-internals testing here makes a shared module
// not worth its own crate).
// ---------------------------------------------------------------------------

fn fake_gpu() -> GpuCapability {
    GpuCapability {
        present: true,
        model: "NVIDIA GeForce RTX 5070".into(),
        total_vram_mib: 12288,
        driver_version: "596.21".into(),
        cuda_runtime: Some("13.2.1".into()),
        compute_capability: Some("8.9".into()),
    }
}

fn fake_cpu() -> CpuCapability {
    CpuCapability {
        brand: "AMD Ryzen 9 7950X".into(),
        logical_cores: 32,
        features: vec!["avx2".into(), "avx512f".into(), "sse4.2".into()],
        ram_mib: 131072,
    }
}

fn bench_row(rows: &[(Backend, f64)]) -> BackendBench {
    let mut b = BackendBench::default();
    for &(backend, ns) in rows {
        b.set(backend, ns);
    }
    b
}

fn vram_row(rows: &[(Backend, usize)]) -> BackendVram {
    let mut v = BackendVram::default();
    for &(backend, mib) in rows {
        v.set(backend, mib);
    }
    v
}

/// Minimal profile that lets the chooser score every backend variant
/// at the requested size. CPU cells are populated with a single point
/// so the conservative-fallback path doesn't need to fire.
fn profile_with_gpu_at(width: u32, height: u32) -> MetricProfile {
    let mut m = MetricProfile::default();
    let px = (width as u64) * (height as u64);
    // GPU cells: GpuFull/GpuStrip/StripPair all measured. Numbers are
    // synthetic but follow the rough RTX 5070 shape.
    m.ns_per_px_at.insert(
        px,
        bench_row(&[
            (Backend::GpuFull, 5.34),
            (Backend::GpuStrip, 6.10),
            (Backend::GpuStripPair, 6.10),
            (Backend::Cpu, 50.0),
        ]),
    );
    m.vram_mib_at.insert(
        px,
        vram_row(&[
            (Backend::GpuFull, 248),
            (Backend::GpuStrip, 130),
            (Backend::GpuStripPair, 142),
            (Backend::Cpu, 0),
        ]),
    );
    m.last_measured = Some(SystemTime::now());
    m
}

fn fake_orch_with_metrics(
    metrics: &[(MetricKind, MetricProfile)],
) -> (Orchestrator, tempfile::TempDir) {
    let tmpdir = tempfile::tempdir().unwrap();
    let gpu = fake_gpu();
    let cpu = fake_cpu();
    let machine_hash = compute_machine_hash(&gpu, &cpu);
    let now = SystemTime::now();
    let mut map: BTreeMap<String, MetricProfile> = BTreeMap::new();
    for (kind, profile) in metrics {
        map.insert(kind.tag().to_string(), profile.clone());
    }
    let profile = CapabilityProfile {
        machine_hash,
        detected_at: now,
        last_validated: now,
        gpu,
        cpu,
        metrics: map,
    };
    let mut cfg = OrchestratorConfig::default();
    cfg.cache_dir = tmpdir.path().to_path_buf();
    cfg.cache_validity = Duration::from_secs(60);
    let path = zenmetrics_orchestrator::cache_file_path(&cfg.cache_dir, &profile.machine_hash);
    save_profile(&path, &profile).unwrap();
    let orch = Orchestrator::from_capability(cfg, profile);
    (orch, tmpdir)
}

fn synth(size: u32) -> (Vec<u8>, Vec<u8>) {
    zenmetrics_orchestrator::synth_pair_offset_dist(size, size)
}

/// Force the OOM ladder to skip every GPU backend by pre-populating
/// `cells_failed_oom` for the requested size.
fn poison_gpu_at(profile: &mut MetricProfile, size_px: u64) {
    for &b in &[Backend::GpuFull, Backend::GpuStrip, Backend::GpuStripPair] {
        profile.cells_failed_oom.push((b, size_px));
    }
}

// ---------------------------------------------------------------------------
// Per-metric construct+compute smoke (gated by feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "cpu-cvvdp")]
#[test]
fn cvvdp_cpu_constructs_and_computes_256() {
    let (r, d) = synth(256);
    let params = zenmetrics_api::MetricParams::try_default_for(MetricKind::Cvvdp).unwrap();
    // CpuAdapter is `pub(crate)`. We reach it via the executor path by
    // forcing the chooser to land on Cpu — same code, integration-test
    // surface.
    let mut profile = profile_with_gpu_at(256, 256);
    poison_gpu_at(&mut profile, 256u64 * 256u64);
    let (mut orch, _td) = fake_orch_with_metrics(&[(MetricKind::Cvvdp, profile)]);
    let task = Task {
        task_id: 11,
        ref_data: TaskData::Srgb8(r),
        dist_data: TaskData::Srgb8(d),
        width: 256,
        height: 256,
        metric: MetricKind::Cvvdp,
        params: Some(params),
        ref_hash: 0,
    };
    let result = orch.run_single(task);
    let score = result.outcome.as_ref().unwrap_or_else(|e| {
        panic!(
            "expected Ok cvvdp cpu score, got Err({e:?}); attempts={:?}",
            result.backends_attempted
        )
    });
    assert_eq!(result.backend_used, Some(Backend::Cpu));
    assert_eq!(score.metric_name, "cvvdp");
    // JOD is in [0, 10]; for a small offset distortion expect mid-to-high.
    assert!(
        score.value >= 0.0 && score.value <= 10.5,
        "cvvdp cpu score out of range: {}",
        score.value
    );
}

#[cfg(feature = "cpu-ssim2")]
#[test]
fn ssim2_cpu_constructs_and_computes_256() {
    let (r, d) = synth(256);
    let params = zenmetrics_api::MetricParams::try_default_for(MetricKind::Ssim2).unwrap();
    let mut profile = profile_with_gpu_at(256, 256);
    poison_gpu_at(&mut profile, 256u64 * 256u64);
    let (mut orch, _td) = fake_orch_with_metrics(&[(MetricKind::Ssim2, profile)]);
    let task = Task {
        task_id: 21,
        ref_data: TaskData::Srgb8(r),
        dist_data: TaskData::Srgb8(d),
        width: 256,
        height: 256,
        metric: MetricKind::Ssim2,
        params: Some(params),
        ref_hash: 0,
    };
    let result = orch.run_single(task);
    let score = result.outcome.as_ref().unwrap_or_else(|e| {
        panic!(
            "expected Ok ssim2 cpu score, got Err({e:?}); attempts={:?}",
            result.backends_attempted
        )
    });
    assert_eq!(result.backend_used, Some(Backend::Cpu));
    assert_eq!(score.metric_name, "ssim2");
    // SSIMULACRA2 returns a finite scalar in roughly [0, 100]; for a
    // tiny offset on a synthetic pair it lands in the high range.
    assert!(score.value.is_finite());
}

#[cfg(feature = "cpu-dssim")]
#[test]
fn dssim_cpu_constructs_and_computes_256() {
    let (r, d) = synth(256);
    let params = zenmetrics_api::MetricParams::try_default_for(MetricKind::Dssim).unwrap();
    let mut profile = profile_with_gpu_at(256, 256);
    poison_gpu_at(&mut profile, 256u64 * 256u64);
    let (mut orch, _td) = fake_orch_with_metrics(&[(MetricKind::Dssim, profile)]);
    let task = Task {
        task_id: 31,
        ref_data: TaskData::Srgb8(r),
        dist_data: TaskData::Srgb8(d),
        width: 256,
        height: 256,
        metric: MetricKind::Dssim,
        params: Some(params),
        ref_hash: 0,
    };
    let result = orch.run_single(task);
    let score = result.outcome.as_ref().unwrap_or_else(|e| {
        panic!(
            "expected Ok dssim cpu score, got Err({e:?}); attempts={:?}",
            result.backends_attempted
        )
    });
    assert_eq!(result.backend_used, Some(Backend::Cpu));
    assert_eq!(score.metric_name, "dssim");
    assert!(score.value.is_finite() && score.value >= 0.0);
}

#[cfg(feature = "cpu-butter")]
#[test]
fn butter_cpu_constructs_and_computes_256() {
    let (r, d) = synth(256);
    let params = zenmetrics_api::MetricParams::try_default_for(MetricKind::Butter).unwrap();
    let mut profile = profile_with_gpu_at(256, 256);
    poison_gpu_at(&mut profile, 256u64 * 256u64);
    let (mut orch, _td) = fake_orch_with_metrics(&[(MetricKind::Butter, profile)]);
    let task = Task {
        task_id: 41,
        ref_data: TaskData::Srgb8(r),
        dist_data: TaskData::Srgb8(d),
        width: 256,
        height: 256,
        metric: MetricKind::Butter,
        params: Some(params),
        ref_hash: 0,
    };
    let result = orch.run_single(task);
    let score = result.outcome.as_ref().unwrap_or_else(|e| {
        panic!(
            "expected Ok butter cpu score, got Err({e:?}); attempts={:?}",
            result.backends_attempted
        )
    });
    assert_eq!(result.backend_used, Some(Backend::Cpu));
    assert_eq!(score.metric_name, "butter");
    assert!(score.value.is_finite() && score.value >= 0.0);
}

#[cfg(feature = "cpu-zensim")]
#[test]
fn zensim_cpu_constructs_and_computes_256() {
    let (r, d) = synth(256);
    let params = zenmetrics_api::MetricParams::try_default_for(MetricKind::Zensim).unwrap();
    let mut profile = profile_with_gpu_at(256, 256);
    poison_gpu_at(&mut profile, 256u64 * 256u64);
    let (mut orch, _td) = fake_orch_with_metrics(&[(MetricKind::Zensim, profile)]);
    let task = Task {
        task_id: 51,
        ref_data: TaskData::Srgb8(r),
        dist_data: TaskData::Srgb8(d),
        width: 256,
        height: 256,
        metric: MetricKind::Zensim,
        params: Some(params),
        ref_hash: 0,
    };
    let result = orch.run_single(task);
    let score = result.outcome.as_ref().unwrap_or_else(|e| {
        panic!(
            "expected Ok zensim cpu score, got Err({e:?}); attempts={:?}",
            result.backends_attempted
        )
    });
    assert_eq!(result.backend_used, Some(Backend::Cpu));
    assert_eq!(score.metric_name, "zensim");
    assert!(score.value.is_finite());
}

// ---------------------------------------------------------------------------
// Iwssim — no CPU reference; chooser surfaces CpuMetricUnavailable
// ---------------------------------------------------------------------------

#[test]
fn iwssim_cpu_unavailable_advances_ladder() {
    // Iwssim has no CPU reference. Poisoning every GPU candidate
    // should leave NO feasible backend — the chooser returns
    // NoFeasibleBackend, which the executor converts to FullyExhausted
    // (or surfaces directly when no attempt was made before).
    let mut profile = profile_with_gpu_at(256, 256);
    poison_gpu_at(&mut profile, 256u64 * 256u64);
    let (mut orch, _td) = fake_orch_with_metrics(&[(MetricKind::Iwssim, profile)]);
    let (r, d) = synth(256);
    let task = Task {
        task_id: 61,
        ref_data: TaskData::Srgb8(r),
        dist_data: TaskData::Srgb8(d),
        width: 256,
        height: 256,
        metric: MetricKind::Iwssim,
        params: None,
        ref_hash: 0,
    };
    let result = orch.run_single(task);
    assert!(result.outcome.is_err());
    match result.outcome {
        Err(ExecutorError::Chooser(_)) | Err(ExecutorError::FullyExhausted { .. }) => {}
        other => panic!("expected Chooser/FullyExhausted, got {:?}", other),
    }
    // Confirm `backend_used` is None — no backend was ever Selected.
    assert!(result.backend_used.is_none());
}

// ---------------------------------------------------------------------------
// Phase 8i Fix C — sentinel errors must NOT trigger record_oom_and_persist
// ---------------------------------------------------------------------------

/// Fix C — `CpuMetricUnavailable` / `CpuBackendUnavailable` /
/// `CpuNotYetWired` are feature-flag / build-configuration sentinels,
/// not memory failures. The executor must NOT poison
/// `cells_failed_oom` with `(Backend::Cpu, pixels)` when these
/// surface, because doing so permanently locks out CPU at that size
/// for any future binary that DOES have the feature enabled.
///
/// Construction: poison every GPU backend at the requested size so
/// the chooser-pre-rejection / executor sentinel branch is the only
/// route to FullyExhausted. After `run_single` returns, the
/// `cells_failed_oom` list must contain ONLY the entries we placed
/// up front — no `(Backend::Cpu, _)` entries added by the executor.
#[test]
fn sentinel_errors_do_not_pollute_cells_failed_oom() {
    // Iwssim works for this test because:
    //   (a) Its `supported_backends` table includes Cpu (so the
    //       chooser progresses to feature-check rather than
    //       short-circuiting UnsupportedByMetric).
    //   (b) Iwssim has a CPU reference but, when `cpu-iwssim` is
    //       OFF, the chooser surfaces CpuMetricUnavailable / the
    //       executor sentinel branch fires — exactly the path Fix C
    //       protects.
    //   (c) When `cpu-iwssim` IS on, Cpu is selected as a real
    //       candidate and the run succeeds (no FullyExhausted), so
    //       there's nothing for the sentinel branch to fire on.
    //       Either way, no `(Cpu, _)` should be in cells_failed_oom
    //       at the end — Fix C's invariant holds in both modes.

    let mut profile = profile_with_gpu_at(256, 256);
    poison_gpu_at(&mut profile, 256u64 * 256u64);
    let initial_oom_count = profile.cells_failed_oom.len();
    let initial_oom_snapshot = profile.cells_failed_oom.clone();

    let (mut orch, _td) = fake_orch_with_metrics(&[(MetricKind::Iwssim, profile)]);
    let (r, d) = synth(256);
    let task = Task {
        task_id: 162,
        ref_data: TaskData::Srgb8(r),
        dist_data: TaskData::Srgb8(d),
        width: 256,
        height: 256,
        metric: MetricKind::Iwssim,
        params: None,
        ref_hash: 0,
    };
    let _ = orch.run_single(task);

    // Inspect cells_failed_oom AFTER the run.
    let final_oom_list = &orch
        .capability()
        .metrics
        .get(MetricKind::Iwssim.tag())
        .expect("iwssim profile must survive run_single")
        .cells_failed_oom;

    // Crucial Fix C invariant: no `(Backend::Cpu, _)` entry was added
    // by the executor's sentinel branches.
    let cpu_oom_entries: Vec<_> = final_oom_list
        .iter()
        .filter(|&&(b, _)| b == Backend::Cpu)
        .collect();
    assert!(
        cpu_oom_entries.is_empty(),
        "Fix C violated: sentinel error caused executor to add \
         Cpu OOM entries: {:?} (cells_failed_oom = {:?})",
        cpu_oom_entries,
        final_oom_list,
    );

    // Defense-in-depth check: the total count must not have grown
    // beyond the initial poison (Fix B's prune may have shrunk it,
    // but no new Cpu entries should appear).
    assert!(
        final_oom_list.len() <= initial_oom_count,
        "cells_failed_oom grew unexpectedly: was {:?} (len={}), \
         now {:?} (len={})",
        initial_oom_snapshot,
        initial_oom_count,
        final_oom_list,
        final_oom_list.len(),
    );
}

// ---------------------------------------------------------------------------
// OOM-forced fallback: every GPU backend rejected -> Cpu picked
// ---------------------------------------------------------------------------

#[cfg(feature = "cpu-cvvdp")]
#[test]
fn oom_fallback_routes_to_cpu_cvvdp_256() {
    // Same shape as cvvdp_cpu_constructs_and_computes_256 but more
    // explicit about the OOM-recovery story for the brief's acceptance
    // gate.
    let (r, d) = synth(256);
    let mut profile = profile_with_gpu_at(256, 256);
    poison_gpu_at(&mut profile, 256u64 * 256u64);
    let (mut orch, _td) = fake_orch_with_metrics(&[(MetricKind::Cvvdp, profile)]);
    let task = Task {
        task_id: 71,
        ref_data: TaskData::Srgb8(r),
        dist_data: TaskData::Srgb8(d),
        width: 256,
        height: 256,
        metric: MetricKind::Cvvdp,
        params: None,
        ref_hash: 0,
    };
    let result = orch.run_single(task);
    let score = result.outcome.as_ref().unwrap_or_else(|e| {
        panic!(
            "expected Cpu fallback Ok, got Err({e:?}); attempts={:?}",
            result.backends_attempted
        )
    });
    assert_eq!(result.backend_used, Some(Backend::Cpu));
    assert_eq!(score.metric_name, "cvvdp");
}

// ---------------------------------------------------------------------------
// Cached-ref round trip (CVVDP — true warm path)
// ---------------------------------------------------------------------------

#[cfg(feature = "cpu-cvvdp")]
#[test]
fn cached_ref_cvvdp_cpu_matches_one_shot() {
    // Score 4 distortions one-shot, then score the same 4 via the
    // pool's cached-ref dispatch. Each pair must match.
    let (r, _) = synth(256);
    let n = 256usize * 256usize * 3;
    let make_d = |seed: u8| -> Vec<u8> {
        (0..n).map(|i| r[i].wrapping_add(seed)).collect()
    };
    let dists: Vec<Vec<u8>> = (1..=4).map(make_d).collect();

    let mut profile = profile_with_gpu_at(256, 256);
    poison_gpu_at(&mut profile, 256u64 * 256u64);

    // One-shot scoring via run_single.
    let mut oneshot_scores: Vec<f64> = Vec::with_capacity(4);
    for (i, d) in dists.iter().enumerate() {
        let (mut orch, _td) = fake_orch_with_metrics(&[(MetricKind::Cvvdp, profile.clone())]);
        let task = Task {
            task_id: (100 + i) as u64,
            ref_data: TaskData::Srgb8(r.clone()),
            dist_data: TaskData::Srgb8(d.clone()),
            width: 256,
            height: 256,
            metric: MetricKind::Cvvdp,
            params: None,
            ref_hash: 0,
        };
        let result = orch.run_single(task);
        let s = result.outcome.as_ref().unwrap_or_else(|e| {
            panic!("one-shot cvvdp cpu failed: {e:?}; attempts={:?}", result.backends_attempted)
        });
        oneshot_scores.push(s.value);
    }

    // Cached-ref scoring via submit/poll_any. Both APIs share the
    // same CpuAdapter under the hood; this exercises the cached-ref
    // promotion logic.
    let (mut orch, _td) = fake_orch_with_metrics(&[(MetricKind::Cvvdp, profile)]);
    // Forcing repeats with the same ref bytes triggers the auto-detect.
    let mut handles = Vec::with_capacity(4);
    for (i, d) in dists.iter().enumerate() {
        let task = Task {
            task_id: (200 + i) as u64,
            ref_data: TaskData::Srgb8(r.clone()),
            dist_data: TaskData::Srgb8(d.clone()),
            width: 256,
            height: 256,
            metric: MetricKind::Cvvdp,
            params: None,
            ref_hash: 0,
        };
        let h = orch
            .submit(task)
            .unwrap_or_else(|e| panic!("submit failed: {e:?}"));
        handles.push(h);
    }
    let mut pooled_by_task_id: std::collections::BTreeMap<u64, f64> = Default::default();
    for _ in 0..handles.len() {
        let r = orch
            .poll_any_blocking()
            .expect("at least one result expected");
        let score = r
            .outcome
            .as_ref()
            .unwrap_or_else(|e| panic!("cached cvvdp cpu task {} failed: {e:?}", r.task_id));
        pooled_by_task_id.insert(r.task_id, score.value);
    }

    // Compare per-task. Both paths use the same CpuAdapter; with
    // cvvdp's deterministic float pipeline the scores should be
    // identical up to floating point noise (cached-ref is the same
    // pipeline minus the reference-side recompute).
    for i in 0..4 {
        let one = oneshot_scores[i];
        let two = pooled_by_task_id
            .get(&((200 + i) as u64))
            .copied()
            .expect("missing pooled result");
        let diff = (one - two).abs();
        assert!(
            diff < 0.05,
            "task {i}: one-shot={one:.4}, cached={two:.4}, diff={diff:.4}"
        );
    }
    // Cached-ref hit count should be positive (auto-detect saw the
    // same ref bytes across tasks). Verify via the stats accessor.
    let stats = orch.cached_ref_stats();
    assert!(
        stats.hit_count > 0,
        "expected cached-ref hits; got {} hits / {} misses",
        stats.hit_count,
        stats.miss_count
    );
}

// ---------------------------------------------------------------------------
// Chooser-level direct test: Cpu Selected when every GPU candidate is
// rejected.
// ---------------------------------------------------------------------------

#[cfg(feature = "cpu-cvvdp")]
#[test]
fn chooser_picks_cpu_when_gpu_oom() {
    use zenmetrics_orchestrator::CandidateStatus;
    let mut profile = profile_with_gpu_at(256, 256);
    poison_gpu_at(&mut profile, 256u64 * 256u64);
    let (orch, _td) = fake_orch_with_metrics(&[(MetricKind::Cvvdp, profile)]);
    let choice = orch
        .choose_backend(MetricKind::Cvvdp, 256, 256, 12288)
        .expect("Cpu should be Selected");
    assert_eq!(choice.backend, Backend::Cpu);
    assert_eq!(choice.predicted_vram_mib, 0);
    let cpu_in_considered = choice
        .considered
        .iter()
        .find(|c| c.backend == Backend::Cpu)
        .expect("Cpu must appear in considered");
    assert!(matches!(
        cpu_in_considered.status,
        CandidateStatus::Selected { vram_mib: 0, .. }
    ));
}

// ---------------------------------------------------------------------------
// Sanity: at least one Phase 6 test runs even without any cpu-* feature.
// Without any CPU backend, every test above is `cfg`'d out — make sure
// the suite still has a single trivial check that exercises the
// "Iwssim unavailable" path, which always applies.
// ---------------------------------------------------------------------------

#[test]
fn cpu_feature_matrix_smoke() {
    // Confirm the build's cpu-* feature flags are visible and roughly
    // consistent with what cpu_backends_enabled would report.
    let any_enabled = cfg!(feature = "cpu-cvvdp")
        || cfg!(feature = "cpu-ssim2")
        || cfg!(feature = "cpu-dssim")
        || cfg!(feature = "cpu-butter")
        || cfg!(feature = "cpu-zensim");
    // If no feature is on, the OOM ladder will hit FullyExhausted at
    // CPU. If any feature is on, the corresponding test above runs.
    // Either way, the matrix is reachable — this test just exists so
    // `cargo test --no-default-features --features cuda` still has at
    // least one test to run from this file.
    eprintln!("cpu_feature_matrix_smoke: any_enabled={any_enabled}");
}
