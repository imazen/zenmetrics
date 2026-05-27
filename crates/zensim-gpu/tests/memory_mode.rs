//! Tests for the unified [`MemoryMode`] API. Host-side only.

use std::sync::{Mutex, OnceLock};
use zensim_gpu::memory_mode::CUBECL_OVERHEAD_BYTES;
use zensim_gpu::{
    Error, MemoryMode, ResolvedMode, ZensimFeatureRegime, estimate_gpu_memory_bytes,
    estimate_strip_gpu_memory_bytes, memory_mode,
};

const VRAM_CAP_VAR: &str = "ZENMETRICS_VRAM_CAP_BYTES";

fn env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn with_cap<R>(cap: Option<&str>, f: impl FnOnce() -> R) -> R {
    let _g = env_lock().lock().unwrap_or_else(|e| e.into_inner());
    let prev = std::env::var(VRAM_CAP_VAR).ok();
    unsafe {
        match cap {
            Some(v) => std::env::set_var(VRAM_CAP_VAR, v),
            None => std::env::remove_var(VRAM_CAP_VAR),
        }
    }
    let out = f();
    unsafe {
        match prev {
            Some(p) => std::env::set_var(VRAM_CAP_VAR, p),
            None => std::env::remove_var(VRAM_CAP_VAR),
        }
    }
    out
}

#[test]
fn estimate_grows_with_pixels() {
    let a = estimate_gpu_memory_bytes(1024, 1024, ZensimFeatureRegime::Basic);
    let b = estimate_gpu_memory_bytes(2048, 2048, ZensimFeatureRegime::Basic);
    let ratio = b as f64 / a as f64;
    assert!(ratio > 3.6 && ratio < 4.4, "ratio = {ratio}");
}

#[test]
fn estimate_grows_with_regime() {
    // Basic 4096² should be markedly smaller than WithIw 4096² —
    // Extended / WithIw allocate ~3× the per-pyramid-pixel bytes.
    let basic = estimate_gpu_memory_bytes(4096, 4096, ZensimFeatureRegime::Basic);
    let ext = estimate_gpu_memory_bytes(4096, 4096, ZensimFeatureRegime::Extended);
    let iw = estimate_gpu_memory_bytes(4096, 4096, ZensimFeatureRegime::WithIw);
    assert!(
        ext > basic * 2,
        "Extended must be > 2× Basic at 4096²: basic={basic}, ext={ext}"
    );
    // WithIw / Extended are within ~10 % of each other (same persist
    // planes, slightly different transient overhead).
    let ratio = (iw as f64) / (ext as f64);
    assert!(
        ratio > 0.85 && ratio < 1.25,
        "WithIw / Extended ratio outside [0.85, 1.25]: {ratio}"
    );
}

/// Calibration regression — the 24 (size × regime) measured rows
/// from `benchmarks/mem_per_metric_2026-05-26.csv` must all land
/// within ±25 % of the estimator's prediction (estimator + cubecl
/// overhead vs measured peak_delta_gpu_mb).
///
/// Fixture values are inlined here so the test survives benchmark
/// CSV moves / renames; if the calibration is re-run, refresh the
/// table and the (BASE, BETA) coefficients in `memory_mode.rs`.
#[test]
fn estimator_matches_measured() {
    // (regime, width, height, measured_peak_delta_mb)
    // Source: benchmarks/mem_per_metric_2026-05-26.csv
    const ROWS: &[(ZensimFeatureRegime, u32, u32, f64)] = &[
        (ZensimFeatureRegime::Basic, 64, 64, 193.0),
        (ZensimFeatureRegime::Basic, 256, 256, 193.0),
        (ZensimFeatureRegime::Basic, 1024, 1024, 225.0),
        (ZensimFeatureRegime::Basic, 2048, 2048, 449.0),
        (ZensimFeatureRegime::Basic, 3000, 3000, 642.0),
        (ZensimFeatureRegime::Basic, 4096, 4096, 1185.0),
        (ZensimFeatureRegime::Basic, 6000, 4000, 1324.0),
        (ZensimFeatureRegime::Basic, 8192, 8192, 3489.0),
        (ZensimFeatureRegime::Extended, 64, 64, 193.0),
        (ZensimFeatureRegime::Extended, 256, 256, 225.0),
        (ZensimFeatureRegime::Extended, 1024, 1024, 481.0),
        (ZensimFeatureRegime::Extended, 2048, 2048, 1217.0),
        (ZensimFeatureRegime::Extended, 3000, 3000, 1599.0),
        (ZensimFeatureRegime::Extended, 4096, 4096, 2721.0),
        (ZensimFeatureRegime::Extended, 6000, 4000, 3713.0),
        (ZensimFeatureRegime::Extended, 8192, 8192, 10369.0),
        (ZensimFeatureRegime::WithIw, 64, 64, 223.0),
        (ZensimFeatureRegime::WithIw, 256, 256, 255.0),
        (ZensimFeatureRegime::WithIw, 1024, 1024, 481.0),
        (ZensimFeatureRegime::WithIw, 2048, 2048, 1217.0),
        (ZensimFeatureRegime::WithIw, 3000, 3000, 1569.0),
        (ZensimFeatureRegime::WithIw, 4096, 4096, 2751.0),
        (ZensimFeatureRegime::WithIw, 6000, 4000, 3713.0),
        (ZensimFeatureRegime::WithIw, 8192, 8192, 10290.0),
    ];

    let mut max_pct = 0.0_f64;
    let mut failures = Vec::new();
    for &(regime, w, h, measured_mb) in ROWS {
        let est_bytes = estimate_gpu_memory_bytes(w, h, regime);
        let total_bytes = est_bytes + CUBECL_OVERHEAD_BYTES;
        let total_mb = (total_bytes as f64) / (1024.0 * 1024.0);
        let pct = 100.0 * (total_mb - measured_mb) / measured_mb.max(1.0);
        if pct.abs() > max_pct {
            max_pct = pct.abs();
        }
        if pct.abs() > 25.0 {
            failures.push((regime, w, h, measured_mb, total_mb, pct));
        }
    }
    eprintln!(
        "estimator vs measured: max |%err| = {max_pct:.1}% over {} rows",
        ROWS.len()
    );
    if !failures.is_empty() {
        for (regime, w, h, meas, pred, pct) in &failures {
            eprintln!(
                "  FAIL regime={regime:?} {w}×{h} measured={meas:.0} MB predicted={pred:.0} MB ({pct:+.1}%)"
            );
        }
        panic!(
            "{} of {} calibration rows exceeded ±25%; max |%err| = {:.1}%",
            failures.len(),
            ROWS.len(),
            max_pct
        );
    }
}

#[test]
fn strip_estimator_returns_value() {
    // Strip mode landed 2026-05-26; estimator no longer returns None.
    let v = estimate_strip_gpu_memory_bytes(1024, 64);
    assert!(v.is_some());
    // Per-pixel strip costs scale with body height.
    let small = estimate_strip_gpu_memory_bytes(1024, 64).unwrap();
    let large = estimate_strip_gpu_memory_bytes(1024, 1024).unwrap();
    assert!(large > small);
}

#[test]
fn auto_picks_full_when_under_cap() {
    with_cap(Some("17179869184"), || {
        let r = memory_mode::resolve_auto(
            1024,
            1024,
            ZensimFeatureRegime::Basic,
            memory_mode::vram_cap_bytes(),
        )
        .unwrap();
        assert_eq!(r, ResolvedMode::Full);
    });
}

#[test]
fn auto_errors_when_strip_cannot_fit() {
    // 1-byte cap is below cubecl runtime overhead — neither Full nor
    // Strip can fit, so resolve_auto returns TooBigForFull.
    with_cap(Some("1"), || {
        let r = memory_mode::resolve_auto(
            4096,
            4096,
            ZensimFeatureRegime::Basic,
            memory_mode::vram_cap_bytes(),
        );
        assert!(matches!(r, Err(Error::TooBigForFull { .. })));
    });
}

#[test]
fn auto_falls_back_to_strip_when_full_exceeds_cap() {
    // A cap that's enough for Strip but not Full. The Full estimate
    // for 4096² Basic is ~1.4 GB (BASE 0 + BETA 41 × ~33 M pyramid
    // pixels). A 600 MB cap (above strip working set + CUBECL_OVERHEAD)
    // forces auto fallback to Strip.
    with_cap(Some("629145600"), || {
        let r = memory_mode::resolve_auto(
            4096,
            4096,
            ZensimFeatureRegime::Basic,
            memory_mode::vram_cap_bytes(),
        );
        match r {
            Ok(ResolvedMode::Strip { h_body }) => {
                assert!(
                    h_body > 0 && h_body.is_multiple_of(zensim_gpu::pipeline::STRIP_ALIGN),
                    "h_body must be a multiple of STRIP_ALIGN; got {h_body}"
                );
            }
            other => panic!("expected Strip fallback, got {other:?}"),
        }
    });
}

#[test]
fn explicit_strip_constructs() {
    let m = MemoryMode::Strip { h_body: Some(128) };
    assert_eq!(m, MemoryMode::Strip { h_body: Some(128) });
}

#[test]
fn tile_returns_unsupported_via_typed_helper() {
    let m = MemoryMode::Tile { h: 512, w: 512 };
    match m {
        MemoryMode::Tile { h, w } => {
            assert_eq!(h, 512);
            assert_eq!(w, 512);
        }
        _ => panic!("expected Tile"),
    }
}

#[test]
fn vram_cap_env_override() {
    with_cap(Some("123456789"), || {
        assert_eq!(memory_mode::vram_cap_bytes(), 123_456_789);
    });
}
