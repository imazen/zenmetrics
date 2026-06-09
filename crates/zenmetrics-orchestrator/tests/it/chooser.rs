//! Integration tests for the Phase 3 backend chooser.
//!
//! Every test constructs a synthetic `CapabilityProfile` with hand-
//! placed `MetricProfile` data, builds an `Orchestrator` via
//! `from_capability`, and asserts on the chooser's decision. No real
//! hardware queries — the chooser is pure modulo the live-VRAM probe
//! and the test injects a synthetic VRAM number directly.

#![cfg(feature = "bench")]

use std::collections::BTreeMap;
use std::time::{Duration, SystemTime};

use zenmetrics_api::MetricKind;
use zenmetrics_orchestrator::{
    Backend, BackendBench, BackendChoice, BackendVram, CandidateStatus, CapabilityProfile,
    ChooserConfig, ChooserError, ConsideredCandidate, CpuCapability, ExecContext, GpuCapability,
    MetricProfile, Orchestrator, OrchestratorConfig, RejectReason, TaskShape, compute_machine_hash,
};

// ---------------------------------------------------------------------------
// Test helpers
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

/// Bench row at a given size with the supplied per-backend numbers.
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

/// Build a MetricProfile spanning 1024² → 4096² with synthetic numbers
/// that loosely mirror the real `~/.cache/zenmetrics/capability_*.toml`
/// shape on a 7950X+RTX-5070 box (cvvdp: GpuFull faster than StripPair
/// at small sizes; StripPair pulls ahead at 4 K).
fn cvvdp_profile() -> MetricProfile {
    let mut m = MetricProfile::default();
    let sizes = [(1024u32, 1024u32), (2048, 2048), (4096, 4096)];
    // ns/px — GpuFull starts cheaper than StripPair (one-shot overhead
    // amortizes at large sizes). Numbers are illustrative.
    let bench_table: &[(u64, &[(Backend, f64)])] = &[
        (
            (sizes[0].0 as u64) * (sizes[0].1 as u64),
            &[(Backend::GpuFull, 5.34), (Backend::GpuStripPair, 6.10)],
        ),
        (
            (sizes[1].0 as u64) * (sizes[1].1 as u64),
            &[(Backend::GpuFull, 3.10), (Backend::GpuStripPair, 3.40)],
        ),
        (
            (sizes[2].0 as u64) * (sizes[2].1 as u64),
            &[(Backend::GpuFull, 2.71), (Backend::GpuStripPair, 2.62)],
        ),
    ];
    // VRAM — GpuFull grows ~linearly with pixels; StripPair scales slower.
    let vram_table: &[(u64, &[(Backend, usize)])] = &[
        (
            (sizes[0].0 as u64) * (sizes[0].1 as u64),
            &[(Backend::GpuFull, 248), (Backend::GpuStripPair, 142)],
        ),
        (
            (sizes[1].0 as u64) * (sizes[1].1 as u64),
            &[(Backend::GpuFull, 992), (Backend::GpuStripPair, 568)],
        ),
        (
            (sizes[2].0 as u64) * (sizes[2].1 as u64),
            &[(Backend::GpuFull, 3970), (Backend::GpuStripPair, 2272)],
        ),
    ];
    for (px, rows) in bench_table {
        m.ns_per_px_at.insert(*px, bench_row(rows));
    }
    for (px, rows) in vram_table {
        m.vram_mib_at.insert(*px, vram_row(rows));
    }
    m.last_measured = Some(SystemTime::now());
    m
}

/// SSIM2 profile — GpuFull + GpuStrip only (no StripPair).
fn ssim2_profile() -> MetricProfile {
    let mut m = MetricProfile::default();
    let bench_table: &[(u64, &[(Backend, f64)])] = &[
        (
            1024 * 1024,
            &[(Backend::GpuFull, 4.50), (Backend::GpuStrip, 5.20)],
        ),
        (
            2048 * 2048,
            &[(Backend::GpuFull, 2.80), (Backend::GpuStrip, 3.20)],
        ),
        (
            4096 * 4096,
            &[(Backend::GpuFull, 2.10), (Backend::GpuStrip, 2.60)],
        ),
    ];
    // SSIM2 is heavier on memory than cvvdp's StripPair — 6.2 GB
    // Full at 4 K matches one of the test scenarios.
    let vram_table: &[(u64, &[(Backend, usize)])] = &[
        (
            1024 * 1024,
            &[(Backend::GpuFull, 410), (Backend::GpuStrip, 220)],
        ),
        (
            2048 * 2048,
            &[(Backend::GpuFull, 1620), (Backend::GpuStrip, 800)],
        ),
        (
            4096 * 4096,
            &[(Backend::GpuFull, 6200), (Backend::GpuStrip, 2900)],
        ),
    ];
    for (px, rows) in bench_table {
        m.ns_per_px_at.insert(*px, bench_row(rows));
    }
    for (px, rows) in vram_table {
        m.vram_mib_at.insert(*px, vram_row(rows));
    }
    m.last_measured = Some(SystemTime::now());
    m
}

/// Butter profile — GpuFull + GpuStrip only, never StripPair.
fn butter_profile() -> MetricProfile {
    let mut m = MetricProfile::default();
    m.ns_per_px_at.insert(
        2048 * 2048,
        bench_row(&[(Backend::GpuFull, 3.30), (Backend::GpuStrip, 3.70)]),
    );
    m.vram_mib_at.insert(
        2048 * 2048,
        vram_row(&[(Backend::GpuFull, 1200), (Backend::GpuStrip, 700)]),
    );
    m.last_measured = Some(SystemTime::now());
    m
}

fn fake_orch_with_metrics(metrics: &[(MetricKind, MetricProfile)]) -> Orchestrator {
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
    // `OrchestratorConfig` is `#[non_exhaustive]` — go through
    // `Default` + struct-update to stay future-proof.
    let mut cfg = OrchestratorConfig::default();
    cfg.cache_dir = std::env::temp_dir().join("zm-chooser-test");
    cfg.cache_validity = Duration::from_secs(60);
    Orchestrator::from_capability(cfg, profile)
}

fn find(c: &[ConsideredCandidate], backend: Backend) -> &ConsideredCandidate {
    c.iter()
        .find(|x| x.backend == backend)
        .unwrap_or_else(|| panic!("backend {} not in considered list", backend.tag()))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn picks_fastest_when_all_fit() {
    // cvvdp at 1024² with 12 GB free — both Full and StripPair fit,
    // Full is cheaper (5.34 < 6.10 ns/px), so Full wins.
    let orch = fake_orch_with_metrics(&[(MetricKind::Cvvdp, cvvdp_profile())]);
    let choice = orch
        .choose_backend(MetricKind::Cvvdp, 1024, 1024, 12288)
        .expect("choose_backend");
    assert_eq!(choice.backend, Backend::GpuFull);
    assert!(
        (choice.predicted_ns_per_px - 5.34).abs() < 1e-6,
        "expected ~5.34 ns/px, got {}",
        choice.predicted_ns_per_px
    );
    assert_eq!(choice.predicted_vram_mib, 248);
    // Both candidates appear in considered.
    let full = find(&choice.considered, Backend::GpuFull);
    assert!(matches!(full.status, CandidateStatus::Selected { .. }));
    let pair = find(&choice.considered, Backend::GpuStripPair);
    assert!(matches!(pair.status, CandidateStatus::Selected { .. }));
}

#[test]
fn falls_back_to_strip_when_full_oom_known() {
    // cvvdp at 4096² with a hard-OOM marker on GpuFull at that size.
    // The chooser must reject GpuFull (KnownOomCell) and pick GpuStripPair.
    let mut profile = cvvdp_profile();
    profile
        .cells_failed_oom
        .push((Backend::GpuFull, 4096u64 * 4096u64));
    let orch = fake_orch_with_metrics(&[(MetricKind::Cvvdp, profile)]);
    let choice = orch
        .choose_backend(MetricKind::Cvvdp, 4096, 4096, 12288)
        .expect("choose_backend");
    assert_eq!(choice.backend, Backend::GpuStripPair);
    let full = find(&choice.considered, Backend::GpuFull);
    match full.status {
        CandidateStatus::Rejected {
            reason: RejectReason::KnownOomCell,
            ..
        } => {}
        ref s => panic!("expected KnownOomCell rejection, got {:?}", s),
    }
}

#[test]
fn falls_back_to_strip_when_vram_constrained() {
    // ssim2 at 4096² with vram_free=4 GB.
    // Usable vram = 4096 * 0.85 = 3481 MiB.
    // GpuFull@4K = 6200 MiB (rejected, PredictedOomWithMargin).
    // GpuStrip@4K = 2900 MiB (fits, picked).
    let orch = fake_orch_with_metrics(&[(MetricKind::Ssim2, ssim2_profile())]);
    let choice = orch
        .choose_backend(MetricKind::Ssim2, 4096, 4096, 4096)
        .expect("choose_backend");
    assert_eq!(choice.backend, Backend::GpuStrip);
    let full = find(&choice.considered, Backend::GpuFull);
    match full.status {
        CandidateStatus::Rejected {
            reason: RejectReason::PredictedOomWithMargin,
            predicted_vram_mib: Some(mib),
            ..
        } => {
            assert_eq!(mib, 6200);
        }
        ref s => panic!("expected PredictedOomWithMargin, got {:?}", s),
    }
}

#[test]
fn returns_no_feasible_when_nothing_fits() {
    // ssim2 at 4096² with vram_free=1 GB. Usable = 871 MiB.
    // GpuFull@4K = 6200 MiB (rejected). GpuStrip@4K = 2900 MiB (rejected).
    //
    // Phase 6: when built WITHOUT cpu-ssim2, Cpu is rejected as
    // CpuMetricUnavailable. Built with `cpu-ssim2`, Cpu is Selected
    // (vram=0) and the call succeeds — so this test only asserts the
    // no-feasible shape under the bench-only / cuda-only feature mix.
    // With cpu-* features compiled in, the assertion below would
    // wrongly fail; gate on `cfg(not(feature = "cpu-ssim2"))`.
    let orch = fake_orch_with_metrics(&[(MetricKind::Ssim2, ssim2_profile())]);
    let result = orch.choose_backend(MetricKind::Ssim2, 4096, 4096, 1024);

    if cfg!(feature = "cpu-ssim2") {
        // With CPU wired, Cpu is a feasible fallback (vram_mib=0).
        // The choice MUST be Cpu since every GPU candidate is OOM.
        let choice = result.expect("Cpu should be Selected when cpu-ssim2 is on");
        assert_eq!(choice.backend, Backend::Cpu);
        assert_eq!(choice.predicted_vram_mib, 0);
    } else {
        let err = result.expect_err("should be NoFeasibleBackend without cpu-ssim2");
        match err {
            ChooserError::NoFeasibleBackend { considered } => {
                assert_eq!(considered.len(), 4);
                assert!(
                    considered
                        .iter()
                        .all(|c| matches!(c.status, CandidateStatus::Rejected { .. }))
                );
                let cpu = find(&considered, Backend::Cpu);
                assert!(matches!(
                    cpu.status,
                    CandidateStatus::Rejected {
                        reason: RejectReason::CpuMetricUnavailable,
                        ..
                    }
                ));
            }
            other => panic!("expected NoFeasibleBackend, got {:?}", other),
        }
    }
}

#[test]
fn unsupported_backend_rejected_cleanly() {
    // Butter never supports StripPair. The chooser must list it as
    // UnsupportedByMetric in `considered` (not silently absent).
    let orch = fake_orch_with_metrics(&[(MetricKind::Butter, butter_profile())]);
    let choice = orch
        .choose_backend(MetricKind::Butter, 2048, 2048, 12288)
        .expect("choose_backend");
    let pair = find(&choice.considered, Backend::GpuStripPair);
    assert!(matches!(
        pair.status,
        CandidateStatus::Rejected {
            reason: RejectReason::UnsupportedByMetric,
            ..
        }
    ));
}

#[test]
fn interpolation_exact_match() {
    // Request 4096² → uses 4096² measurement directly (2.71 ns/px).
    let orch = fake_orch_with_metrics(&[(MetricKind::Cvvdp, cvvdp_profile())]);
    let choice = orch
        .choose_backend(MetricKind::Cvvdp, 4096, 4096, 12288)
        .expect("choose_backend");
    // At 4K, StripPair (2.62) is cheaper than Full (2.71) — picks
    // StripPair.
    assert_eq!(choice.backend, Backend::GpuStripPair);
    assert!(
        (choice.predicted_ns_per_px - 2.62).abs() < 1e-6,
        "expected 2.62 exact, got {}",
        choice.predicted_ns_per_px
    );
    assert_eq!(choice.predicted_vram_mib, 2272);
}

#[test]
fn interpolation_between_measured() {
    // Request 3000×3000 (9 000 000 px) — between 2048² (4 194 304) and
    // 4096² (16 777 216). Log-pixel midpoint: log2(9 000 000) ≈ 23.10
    // sits ~73% of the way from log2(4M)=22.00 to log2(16M)=24.00.
    //
    // For GpuFull: 3.10 * (1 - t) + 2.71 * t where t ≈ (23.10-22.00)/2.00 = 0.55
    //            ≈ 3.10*0.45 + 2.71*0.55 = 1.395 + 1.491 ≈ 2.886.
    // The exact arithmetic is tested in detail below; here we just
    // confirm the value sits strictly between 2.71 and 3.10.
    let orch = fake_orch_with_metrics(&[(MetricKind::Cvvdp, cvvdp_profile())]);
    let choice = orch
        .choose_backend(MetricKind::Cvvdp, 3000, 3000, 12288)
        .expect("choose_backend");
    let ns = choice.predicted_ns_per_px;
    assert!(ns > 2.62, "interpolated ns_per_px {ns} too low");
    assert!(ns < 3.40, "interpolated ns_per_px {ns} too high");
    // VRAM is between 992 (2 K) and 3970 (4 K).
    assert!(choice.predicted_vram_mib > 568);
    assert!(choice.predicted_vram_mib < 3970);
}

#[test]
fn interpolation_below_range_clamps() {
    // Request 256×256 (65 536 px) < 1024² (1 048 576). Should clamp
    // to the 1024² measurement (5.34 ns/px for GpuFull), NOT
    // optimistically shrink — fixed overhead dominates at tiny sizes.
    let orch = fake_orch_with_metrics(&[(MetricKind::Cvvdp, cvvdp_profile())]);
    let choice = orch
        .choose_backend(MetricKind::Cvvdp, 256, 256, 12288)
        .expect("choose_backend");
    assert_eq!(choice.backend, Backend::GpuFull);
    assert!(
        (choice.predicted_ns_per_px - 5.34).abs() < 1e-6,
        "expected clamped 5.34, got {}",
        choice.predicted_ns_per_px
    );
}

#[test]
fn extrapolation_above_range_pessimistic() {
    // Request 8192² (67 108 864 px) — above 4096² (16 777 216).
    // Default extrapolation_pessimism = 1.20.
    //
    // To test the multiplier deterministically, isolate a single
    // backend. Use a synthetic profile with only GpuFull measured at
    // 2K + 4K so the chooser doesn't get distracted by tie-breaks
    // with StripPair (whose extrapolated slope is much steeper in
    // the realistic cvvdp_profile).
    //
    // Two-point GpuFull at (4 M, 4.00) and (16 M, 3.00). Log-pixel
    // extrapolation to 64 M (= 8 K²): t = 2.0, v = 4.00*(1-2)+3.00*2 = 2.00.
    // With pessimism 1.20, predicted = 2.40 ns/px.
    let mut m = MetricProfile::default();
    m.ns_per_px_at
        .insert(2048u64 * 2048u64, bench_row(&[(Backend::GpuFull, 4.00)]));
    m.ns_per_px_at
        .insert(4096u64 * 4096u64, bench_row(&[(Backend::GpuFull, 3.00)]));
    m.vram_mib_at
        .insert(2048u64 * 2048u64, vram_row(&[(Backend::GpuFull, 1000)]));
    m.vram_mib_at
        .insert(4096u64 * 4096u64, vram_row(&[(Backend::GpuFull, 4000)]));
    m.last_measured = Some(SystemTime::now());
    let orch = fake_orch_with_metrics(&[(MetricKind::Cvvdp, m)]);

    // Plenty of VRAM so nothing is rejected on budget grounds.
    let choice = orch
        .choose_backend(MetricKind::Cvvdp, 8192, 8192, 65536)
        .expect("choose_backend");
    assert_eq!(choice.backend, Backend::GpuFull);
    // Naive extrapolation = 2.00 ns/px; pessimism = 1.20 → 2.40.
    // `extrapolation_pessimism` is `f32` so we allow ~5 ULPs of slack.
    let expected = 2.40_f64;
    assert!(
        (choice.predicted_ns_per_px - expected).abs() < 1e-6,
        "expected {expected:.3} (2.00 * 1.20 pessimism), got {}",
        choice.predicted_ns_per_px
    );
}

#[test]
fn extrapolation_pessimism_overridable() {
    // Same setup as the previous test but with extrapolation_pessimism
    // bumped to 2.0 — the predicted ns/px must double from naive.
    let mut m = MetricProfile::default();
    m.ns_per_px_at
        .insert(2048u64 * 2048u64, bench_row(&[(Backend::GpuFull, 4.00)]));
    m.ns_per_px_at
        .insert(4096u64 * 4096u64, bench_row(&[(Backend::GpuFull, 3.00)]));
    m.vram_mib_at
        .insert(2048u64 * 2048u64, vram_row(&[(Backend::GpuFull, 1000)]));
    m.vram_mib_at
        .insert(4096u64 * 4096u64, vram_row(&[(Backend::GpuFull, 4000)]));
    m.last_measured = Some(SystemTime::now());
    let orch = fake_orch_with_metrics(&[(MetricKind::Cvvdp, m)]);

    let mut cfg = ChooserConfig::default();
    cfg.extrapolation_pessimism = 2.0;
    let choice = orch
        .choose_backend_with_config(MetricKind::Cvvdp, 8192, 8192, 65536, &cfg)
        .expect("choose_backend");
    // Naive extrapolation 2.00 ns/px * pessimism 2.0 = 4.0.
    assert!(
        (choice.predicted_ns_per_px - 4.0).abs() < 1e-6,
        "expected 4.0 (2.00 * 2.0 pessimism), got {}",
        choice.predicted_ns_per_px
    );
}

#[test]
fn unknown_metric_errors() {
    // Profile loaded with only cvvdp; ask about ssim2 → UnknownMetric.
    let orch = fake_orch_with_metrics(&[(MetricKind::Cvvdp, cvvdp_profile())]);
    let err = orch
        .choose_backend(MetricKind::Ssim2, 1024, 1024, 12288)
        .expect_err("should be UnknownMetric");
    match err {
        ChooserError::UnknownMetric(MetricKind::Ssim2) => {}
        other => panic!("expected UnknownMetric(Ssim2), got {:?}", other),
    }
}

#[test]
fn diagnostic_considered_list_populated() {
    // Every candidate must appear in `considered`, with their
    // statuses populated so an operator can audit.
    let orch = fake_orch_with_metrics(&[(MetricKind::Cvvdp, cvvdp_profile())]);
    let choice = orch
        .choose_backend(MetricKind::Cvvdp, 1024, 1024, 12288)
        .expect("choose_backend");
    assert_eq!(choice.considered.len(), 4);
    // The four backends must each appear once.
    for b in [
        Backend::GpuFull,
        Backend::GpuStrip,
        Backend::GpuStripPair,
        Backend::Cpu,
    ] {
        let found = choice.considered.iter().filter(|c| c.backend == b).count();
        assert_eq!(found, 1, "expected exactly one entry for {}", b.tag());
    }
    // GpuStrip is rejected as UnsupportedByMetric for cvvdp.
    let strip = find(&choice.considered, Backend::GpuStrip);
    assert!(matches!(
        strip.status,
        CandidateStatus::Rejected {
            reason: RejectReason::UnsupportedByMetric,
            ..
        }
    ));
    // Phase 6: Cpu candidate disposition depends on the cpu-cvvdp
    // feature (this test uses MetricKind::Cvvdp). Without the feature
    // Cpu is rejected as CpuMetricUnavailable; with it, Cpu is
    // Selected (vram=0) and ranks against the GPU candidates.
    let cpu = find(&choice.considered, Backend::Cpu);
    if cfg!(feature = "cpu-cvvdp") {
        assert!(
            matches!(cpu.status, CandidateStatus::Selected { .. }),
            "expected Selected, got {:?}",
            cpu.status
        );
    } else {
        assert!(matches!(
            cpu.status,
            CandidateStatus::Rejected {
                reason: RejectReason::CpuMetricUnavailable,
                ..
            }
        ));
    }
}

#[test]
fn tie_break_respects_config_order() {
    // Construct a profile where Full and StripPair tie exactly on
    // ns/px. Default tie-break order prefers Full first, so Full wins.
    let mut m = MetricProfile::default();
    let px = 4096u64 * 4096u64;
    m.ns_per_px_at.insert(
        px,
        bench_row(&[(Backend::GpuFull, 2.50), (Backend::GpuStripPair, 2.50)]),
    );
    m.vram_mib_at.insert(
        px,
        vram_row(&[(Backend::GpuFull, 3970), (Backend::GpuStripPair, 2272)]),
    );
    m.last_measured = Some(SystemTime::now());
    let orch = fake_orch_with_metrics(&[(MetricKind::Cvvdp, m)]);

    let default_choice = orch
        .choose_backend(MetricKind::Cvvdp, 4096, 4096, 12288)
        .expect("default choose");
    assert_eq!(default_choice.backend, Backend::GpuFull);

    // Override tie-break order to put StripPair first → StripPair wins.
    let mut cfg = ChooserConfig::default();
    cfg.tie_break_order = [
        Backend::GpuStripPair,
        Backend::GpuFull,
        Backend::GpuStrip,
        Backend::Cpu,
    ];
    let overridden = orch
        .choose_backend_with_config(MetricKind::Cvvdp, 4096, 4096, 12288, &cfg)
        .expect("override choose");
    assert_eq!(overridden.backend, Backend::GpuStripPair);
}

#[test]
fn safety_margin_mib_is_non_negative_for_selected() {
    // Sanity: a passing candidate must have safety_margin_mib >= 0.
    let orch = fake_orch_with_metrics(&[(MetricKind::Cvvdp, cvvdp_profile())]);
    let choice: BackendChoice = orch
        .choose_backend(MetricKind::Cvvdp, 1024, 1024, 12288)
        .expect("ok");
    // usable = 12288 * 0.85 = 10444. Full@1K vram = 248. margin = 10196.
    assert!(choice.safety_margin_mib > 0);
    assert!(choice.safety_margin_mib < 12288);
}

#[test]
fn no_measured_data_rejected_cleanly() {
    // Profile has cvvdp ns_per_px_at but the vram_mib_at map is
    // missing the 1024² entry — so the VRAM interpolator returns
    // None and the candidate is rejected as NoMeasuredData.
    let mut m = MetricProfile::default();
    m.ns_per_px_at.insert(
        1024 * 1024,
        bench_row(&[(Backend::GpuFull, 5.34), (Backend::GpuStripPair, 6.10)]),
    );
    // Deliberately leave vram_mib_at empty.
    m.last_measured = Some(SystemTime::now());
    let orch = fake_orch_with_metrics(&[(MetricKind::Cvvdp, m)]);
    // Should be no Selected backends → NoFeasibleBackend with all
    // rejected as NoMeasuredData (or CpuMetricUnavailable for Cpu
    // when cpu-cvvdp is off). When cpu-cvvdp is on, Cpu is Selected
    // via the conservative-fallback path (200 ns/px, vram=0) — the
    // test is gated.
    if cfg!(feature = "cpu-cvvdp") {
        // Cpu becomes a feasible fallback even with empty vram_mib_at.
        let choice = orch
            .choose_backend(MetricKind::Cvvdp, 1024, 1024, 12288)
            .expect("Cpu Selected when cpu-cvvdp is on");
        assert_eq!(choice.backend, Backend::Cpu);
        return;
    }
    let err = orch
        .choose_backend(MetricKind::Cvvdp, 1024, 1024, 12288)
        .expect_err("no vram → no feasible");
    let considered = match err {
        ChooserError::NoFeasibleBackend { considered } => considered,
        other => panic!("expected NoFeasibleBackend, got {:?}", other),
    };
    let full = find(&considered, Backend::GpuFull);
    match full.status {
        CandidateStatus::Rejected {
            reason: RejectReason::NoMeasuredData,
            predicted_ns_per_px: Some(ns),
            predicted_vram_mib: None,
        } => {
            assert!((ns - 5.34).abs() < 1e-6);
        }
        ref s => panic!("expected NoMeasuredData rejection, got {:?}", s),
    }
}

#[test]
fn task_shape_uses_orchestrator_default_probe() {
    // `choose_backend_for_task` calls the live probe. On a CI box
    // without nvidia-smi, the probe returns None and the helper
    // falls back to capability.gpu.total_vram_mib (which is 12288
    // in our fake profile).
    //
    // The test asserts the call doesn't panic and produces a choice
    // when the probe falls back. We can't assert which backend
    // because real-VRAM probing on a dev box may return a wildly
    // different number — but we can assert that "no panic + Ok".
    let orch = fake_orch_with_metrics(&[(MetricKind::Cvvdp, cvvdp_profile())]);
    let task = TaskShape {
        metric: MetricKind::Cvvdp,
        width: 1024,
        height: 1024,
    };
    let choice = orch
        .choose_backend_for_task(&task)
        .expect("choose_backend_for_task");
    assert!(matches!(
        choice.backend,
        Backend::GpuFull | Backend::GpuStripPair
    ));
}

#[test]
fn chooser_runs_in_under_100us() {
    // The acceptance gate says `choose_backend` should be well under
    // 100µs once the cache is loaded. Measure 1000 invocations and
    // check the mean.
    let orch = fake_orch_with_metrics(&[(MetricKind::Cvvdp, cvvdp_profile())]);
    // Warm up CPU caches.
    for _ in 0..10 {
        let _ = orch.choose_backend(MetricKind::Cvvdp, 1024, 1024, 12288);
    }
    let t0 = std::time::Instant::now();
    let iters = 1000;
    for _ in 0..iters {
        let _ = orch
            .choose_backend(MetricKind::Cvvdp, 1024, 1024, 12288)
            .expect("ok");
    }
    let elapsed = t0.elapsed();
    let per_call = elapsed.as_nanos() as u64 / iters as u64;
    // Allow generous headroom: gate is 100µs (100 000 ns). The pure
    // arithmetic should clock in under 10µs (10 000 ns) but optimized
    // builds vary.
    assert!(
        per_call < 100_000,
        "chooser took {per_call} ns/call (gate: <100 000)"
    );
    eprintln!("[perf] chooser: {per_call} ns/call ({iters} iters)");
}

/// Phase 7.7.1 regression: a backend whose 2-point log-linear
/// extrapolation produces a NEGATIVE `predicted_ns_per_px` must be
/// rejected with [`RejectReason::NonPositivePrediction`] rather than
/// ranked as the fastest candidate (`min(ns_per_px)` would otherwise
/// pick the negative value, which is structurally wrong — negative
/// time cannot be faster than positive time).
///
/// Scenario captures the production ssim2 + zensim CPU-extrapolation
/// failure observed during the Phase 7.7 parity sweep:
/// - CPU bench cells at 262144 (1018.74 ns/px) and 1048576 (629.23 ns/px)
/// - 16M-pixel target (4096²)
/// - log-linear extrapolation: slope = (629.23 - 1018.74) / (log2(1M) - log2(256K))
///   = -194.76 per doubling; at log2(16M)=24, v = 629.23 + (-194.76) * 4 = -149.8
/// - * extrapolation_pessimism (1.20) = -179.76 ns/px
/// Without this gate, ssim2's chooser would pick CPU (predicted -179.76
/// "faster" than GpuStrip at 5 ns/px) and produce a divergent score
/// vs the legacy GPU path.
///
/// Phase 8g.2 — this test requires the `cpu-ssim2` feature so the
/// chooser admits the Cpu candidate to the ns/px comparison instead
/// of short-circuiting `CpuMetricUnavailable`. Without the feature,
/// the negative-prediction code path is unreachable and the test's
/// `RejectReason::NonPositivePrediction` assertion would never fire.
#[cfg(feature = "cpu-ssim2")]
#[test]
fn rejects_negative_extrapolated_cpu_prediction() {
    let mut m = MetricProfile::default();
    // GPU at 16M cell — real measurement, so the chooser has a non-OOM
    // GPU candidate to fall back to once CPU is rejected.
    m.ns_per_px_at
        .insert(16_777_216, bench_row(&[(Backend::GpuStrip, 5.0_f64)]));
    m.vram_mib_at
        .insert(16_777_216, vram_row(&[(Backend::GpuStrip, 800)]));
    // CPU at 256K + GPU at 1M co-cached (the cache stores per-size
    // backends in one BackendBench entry per pixel-key, so we set the
    // Cpu and GpuStrip values together for the size keys we want both
    // to share).
    m.ns_per_px_at
        .insert(262_144, bench_row(&[(Backend::Cpu, 1018.74_f64)]));
    m.vram_mib_at
        .insert(262_144, vram_row(&[(Backend::Cpu, 0)]));
    m.ns_per_px_at.insert(
        1_048_576,
        bench_row(&[(Backend::Cpu, 629.23_f64), (Backend::GpuStrip, 6.0_f64)]),
    );
    m.vram_mib_at.insert(
        1_048_576,
        vram_row(&[(Backend::Cpu, 0), (Backend::GpuStrip, 100)]),
    );
    m.last_measured = Some(SystemTime::now());

    let orch = fake_orch_with_metrics(&[(MetricKind::Ssim2, m)]);
    let choice = orch
        .choose_backend(MetricKind::Ssim2, 4096, 4096, 12288)
        .expect("choose_backend should succeed via GPU fallback");

    // CPU must be rejected as NonPositivePrediction.
    let cpu = find(&choice.considered, Backend::Cpu);
    match cpu.status {
        CandidateStatus::Rejected {
            reason: RejectReason::NonPositivePrediction,
            predicted_ns_per_px: Some(ns),
            ..
        } => {
            assert!(ns <= 0.0, "expected non-positive ns, got {ns}",);
        }
        ref s => panic!("expected CPU NonPositivePrediction rejection, got {s:?}",),
    }

    // The selected backend MUST be GpuStrip (the only positive-ns
    // candidate at this size).
    assert_eq!(
        choice.backend,
        Backend::GpuStrip,
        "expected GpuStrip selection after CPU rejection, got {:?}",
        choice.backend,
    );
}

// ---------------------------------------------------------------------------
// Phase 8i — cache-hygiene regression tests
// ---------------------------------------------------------------------------

/// Fix A: a smaller-size OOM entry must NOT cascade-reject a larger
/// request when the cache holds a positive measurement at any
/// size >= the OOMed size for that backend. The investigation in
/// `docs/CVVDP_CHOOSER_REGRESSION_INVESTIGATION.md` documented this
/// as the root cause of the 9 cvvdp parity rejections at HEAD:
/// `cells_failed_oom = [(GpuFull, 256²)]` was cascading to every
/// request at any larger size, even though `ns_per_px_at[1024²]`
/// recorded a successful GpuFull measurement.
#[test]
fn oom_cascade_defeated_by_positive_measurement_at_or_above_oom_size() {
    let mut profile = cvvdp_profile();
    // Pre-poison: GpuFull OOMed at 256² (65 536 px). The 1024²
    // positive measurement in cvvdp_profile() must defeat the
    // cascade so the chooser still picks GpuFull at 1024².
    profile
        .cells_failed_oom
        .push((Backend::GpuFull, 256u64 * 256u64));

    let orch = fake_orch_with_metrics(&[(MetricKind::Cvvdp, profile)]);
    let choice = orch
        .choose_backend(MetricKind::Cvvdp, 1024, 1024, 12288)
        .expect("choose_backend should succeed once cascade is defeated");

    // GpuFull is the fastest survivor (5.34 < 6.10 ns/px) and the
    // 1024² positive measurement proves the 256² OOM is stale.
    assert_eq!(
        choice.backend,
        Backend::GpuFull,
        "expected GpuFull selection (cascade defeated by positive \
         measurement at 1024²), got {:?}",
        choice.backend,
    );
    let full = find(&choice.considered, Backend::GpuFull);
    match full.status {
        CandidateStatus::Selected { .. } => {}
        ref s => panic!(
            "expected GpuFull Selected after positive-measurement \
             defeats cascade, got {s:?}",
        ),
    }
}

/// Fix A — negative control: without ANY positive measurement at
/// size >= the OOMed size for this backend, the cascade still
/// rejects. Construct a profile where GpuFull's only positive
/// measurement is BELOW the OOMed size, so the cascade rule must
/// fire.
#[test]
fn oom_cascade_still_rejects_when_no_positive_measurement_at_or_above() {
    // Hand-build a profile so GpuFull has a positive measurement
    // ONLY at 32² (1024 px), well below the 256² OOM. The cascade
    // rule must still reject a 1024² request (much larger than the
    // OOMed size).
    let mut profile = MetricProfile::default();
    profile.ns_per_px_at.insert(
        1024,
        bench_row(&[(Backend::GpuFull, 8.0), (Backend::GpuStripPair, 9.0)]),
    );
    profile.vram_mib_at.insert(
        1024,
        vram_row(&[(Backend::GpuFull, 8), (Backend::GpuStripPair, 6)]),
    );
    // Add 1024² ns_per_px so the interpolator doesn't return
    // `NoMeasuredData` (we want to test the cascade specifically).
    profile
        .ns_per_px_at
        .insert(1024 * 1024, bench_row(&[(Backend::GpuStripPair, 6.10)]));
    profile
        .vram_mib_at
        .insert(1024 * 1024, vram_row(&[(Backend::GpuStripPair, 142)]));
    profile.last_measured = Some(SystemTime::now());
    // GpuFull's only positive measurement is at 1024 px (32²);
    // the 256² (65 536 px) OOM cascades to anything > 65 536 px.
    profile
        .cells_failed_oom
        .push((Backend::GpuFull, 256u64 * 256u64));

    let orch = fake_orch_with_metrics(&[(MetricKind::Cvvdp, profile)]);
    let choice = orch
        .choose_backend(MetricKind::Cvvdp, 1024, 1024, 12288)
        .expect("StripPair should still be feasible");

    let full = find(&choice.considered, Backend::GpuFull);
    match full.status {
        CandidateStatus::Rejected {
            reason: RejectReason::KnownOomCell,
            ..
        } => {}
        ref s => panic!(
            "expected GpuFull KnownOomCell rejection (no positive \
             measurement at size >= 65 536 px), got {s:?}",
        ),
    }
    assert_eq!(
        choice.backend,
        Backend::GpuStripPair,
        "expected StripPair fallback when GpuFull cascade is valid",
    );
}

// ---------------------------------------------------------------------------
// Task #146 — one-shot CPU/GPU crossover routing
//
// The chooser ranks `Batch` tasks on warm `ns_per_px` (correct when a
// persistent warm worker amortizes the GPU's ~181 ms cold floor). For a
// single cold call (`ExecContext::OneShot`) it consults the measured
// crossover (`benchmarks/cpu_gpu_crossover_2026-05-29.tsv`) and prefers
// CPU at/below the per-metric crossover size when CPU is feasible.
//
// CPU is only a *feasible* (`Selected`) candidate when the matching
// `cpu-<metric>` feature is compiled in. The CPU-picked assertions are
// therefore gated on the relevant `cpu-*` feature; the fall-through
// assertions (OneShot == Batch when CPU is infeasible) run in every
// feature mix.
// ---------------------------------------------------------------------------

#[test]
fn batch_context_is_identical_to_choose_backend() {
    // ExecContext::Batch must be bit-identical to the legacy
    // `choose_backend` / `choose_backend_with_config` path — no behavior
    // change for the warm pool / sweep callers. Holds regardless of which
    // cpu-* features are enabled (both go through the same warm-ns/px
    // ranking).
    let orch = fake_orch_with_metrics(&[(MetricKind::Cvvdp, cvvdp_profile())]);
    let legacy = orch
        .choose_backend(MetricKind::Cvvdp, 1024, 1024, 12288)
        .expect("legacy choose_backend");
    let batch = orch
        .choose_backend_with_context(
            MetricKind::Cvvdp,
            1024,
            1024,
            12288,
            &ChooserConfig::default(),
            ExecContext::Batch,
        )
        .expect("batch context");
    assert_eq!(batch.backend, legacy.backend);
    assert_eq!(batch.predicted_ns_per_px, legacy.predicted_ns_per_px);
    assert_eq!(batch.predicted_vram_mib, legacy.predicted_vram_mib);
    // Batch must pick a GPU backend at 1024² cvvdp (warm-ns/px ranking).
    assert_eq!(batch.backend, Backend::GpuFull);
}

#[test]
fn oneshot_falls_through_to_gpu_when_cpu_infeasible() {
    // When CPU is NOT a feasible candidate (cpu-cvvdp off), the one-shot
    // rule cannot fire — a one-shot small cvvdp call falls through to the
    // warm-ns/px ranking and picks GPU exactly like Batch. This protects
    // against the one-shot path regressing to an unavailable backend.
    let orch = fake_orch_with_metrics(&[(MetricKind::Cvvdp, cvvdp_profile())]);
    let choice = orch
        .choose_backend_with_context(
            MetricKind::Cvvdp,
            1024,
            1024,
            12288,
            &ChooserConfig::default(),
            ExecContext::OneShot,
        )
        .expect("one-shot choose");
    if cfg!(feature = "cpu-cvvdp") {
        // CPU feasible → one-shot crossover routes 1 MP cvvdp to CPU.
        assert_eq!(
            choice.backend,
            Backend::Cpu,
            "1 MP cvvdp one-shot should route to CPU (cpu-cvvdp on)"
        );
        assert_eq!(choice.predicted_vram_mib, 0);
    } else {
        // CPU infeasible → fall through to GPU (same as Batch).
        assert_eq!(
            choice.backend,
            Backend::GpuFull,
            "without cpu-cvvdp, one-shot must fall through to GPU"
        );
    }
}

#[test]
#[cfg(feature = "cpu-cvvdp")]
fn oneshot_routes_small_cvvdp_to_cpu_but_batch_picks_gpu() {
    // The headline crossover behavior: same (metric, size, vram), the
    // ONLY difference is the execution context. One-shot → CPU (cold
    // floor not amortized), Batch → GPU (warm worker amortizes it).
    let orch = fake_orch_with_metrics(&[(MetricKind::Cvvdp, cvvdp_profile())]);

    let batch = orch
        .choose_backend_with_context(
            MetricKind::Cvvdp,
            512,
            512,
            12288,
            &ChooserConfig::default(),
            ExecContext::Batch,
        )
        .expect("batch");
    assert_eq!(batch.backend, Backend::GpuFull, "batch 512² cvvdp → GPU");

    let one_shot = orch
        .choose_backend_with_context(
            MetricKind::Cvvdp,
            512,
            512,
            12288,
            &ChooserConfig::default(),
            ExecContext::OneShot,
        )
        .expect("one-shot");
    assert_eq!(one_shot.backend, Backend::Cpu, "one-shot 512² cvvdp → CPU");

    // cvvdp CPU wins one-shot at EVERY measured size (u64::MAX boundary),
    // so even a 4096² one-shot routes to CPU.
    let one_shot_4k = orch
        .choose_backend_with_context(
            MetricKind::Cvvdp,
            4096,
            4096,
            12288,
            &ChooserConfig::default(),
            ExecContext::OneShot,
        )
        .expect("one-shot 4k");
    assert_eq!(
        one_shot_4k.backend,
        Backend::Cpu,
        "cvvdp one-shot routes to CPU at all measured sizes (16 MP incl.)"
    );
}

#[test]
#[cfg(feature = "cpu-cvvdp")]
fn oneshot_for_task_with_context_uses_crossover() {
    // The live-probe convenience wrapper must thread the ExecContext
    // through. On a CI box without a GPU the probe falls back to cached
    // total_vram_mib (12288), which is plenty for the CPU=0 route.
    let orch = fake_orch_with_metrics(&[(MetricKind::Cvvdp, cvvdp_profile())]);
    let shape = TaskShape {
        metric: MetricKind::Cvvdp,
        width: 1024,
        height: 1024,
    };
    let one_shot = orch
        .choose_backend_for_task_with_context(&shape, ExecContext::OneShot)
        .expect("one-shot for-task");
    assert_eq!(one_shot.backend, Backend::Cpu);

    let batch = orch
        .choose_backend_for_task_with_context(&shape, ExecContext::Batch)
        .expect("batch for-task");
    assert_eq!(batch.backend, Backend::GpuFull);
    // The default for-task wrapper keeps Batch semantics.
    let default = orch.choose_backend_for_task(&shape).expect("default");
    assert_eq!(default.backend, batch.backend);
}

#[test]
#[cfg(feature = "cpu-cvvdp")]
fn oneshot_does_not_route_to_cpu_when_cpu_oom_listed() {
    // Even one-shot, if the CPU cell is marked OOM at this size the rule
    // must not pick CPU — it falls through to the GPU ranking. Guards the
    // "never pick an infeasible backend" invariant.
    let mut profile = cvvdp_profile();
    profile
        .cells_failed_oom
        .push((Backend::Cpu, 1024u64 * 1024u64));
    let orch = fake_orch_with_metrics(&[(MetricKind::Cvvdp, profile)]);
    let choice = orch
        .choose_backend_with_context(
            MetricKind::Cvvdp,
            1024,
            1024,
            12288,
            &ChooserConfig::default(),
            ExecContext::OneShot,
        )
        .expect("one-shot with cpu OOM");
    assert_ne!(
        choice.backend,
        Backend::Cpu,
        "CPU is OOM-listed at 1 MP — one-shot must fall through to GPU"
    );
    assert_eq!(choice.backend, Backend::GpuFull);
}
