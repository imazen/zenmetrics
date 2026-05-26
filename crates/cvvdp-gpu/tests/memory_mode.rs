//! Tests for the unified [`MemoryMode`] API. Host-side only — real
//! GPU integration is covered by `tests/pipeline_score.rs` and
//! friends.

use cvvdp_gpu::{
    Error, MemoryMode, ResolvedMode, estimate_gpu_memory_bytes_usize, memory_mode,
};
use std::sync::{Mutex, OnceLock};

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
    let a = estimate_gpu_memory_bytes_usize(1024, 1024);
    let b = estimate_gpu_memory_bytes_usize(2048, 2048);
    let ratio = b as f64 / a as f64;
    assert!(ratio > 3.6 && ratio < 4.4, "ratio = {ratio}");
}

#[test]
fn auto_picks_full_when_under_cap() {
    with_cap(Some("17179869184"), || {
        let r = memory_mode::resolve_auto(1024, 1024, memory_mode::vram_cap_bytes()).unwrap();
        assert_eq!(r, ResolvedMode::Full);
    });
}

#[test]
fn auto_errors_when_too_big_for_full() {
    with_cap(Some("1"), || {
        let r = memory_mode::resolve_auto(4096, 4096, memory_mode::vram_cap_bytes());
        assert!(matches!(r, Err(Error::TooBigForFull { .. })));
    });
}

#[test]
fn enum_variants_match_task79_contract() {
    // Task #79 + Round 2 contract: cvvdp's MemoryMode enum is
    // `{ Auto, Full, Strip { h_body }, StripPair { h_body },
    // CappedPyramid { levels } }`.
    // The capped-pyramid Strip variant rolled back in task #77 was
    // re-introduced as the standalone `CappedPyramid` variant (Option
    // B safety net, 2026-05-26) — it is **JOD-shifting** and is never
    // picked by `Auto`. The mode-E `Strip` variant landed in task
    // #79 (ref-full + per-strip dist cached-ref, JOD-preserving).
    // The mode-B `StripPair` variant landed in Round 2 (one-shot pair
    // stripwise, no ref cache).
    let _auto = MemoryMode::Auto;
    let _full = MemoryMode::Full;
    let _strip = MemoryMode::Strip { h_body: None };
    let _strip_explicit = MemoryMode::Strip {
        h_body: Some(memory_mode::STRIP_H_BODY_DEFAULT),
    };
    let _strip_pair = MemoryMode::StripPair { h_body: None };
    let _strip_pair_explicit = MemoryMode::StripPair {
        h_body: Some(memory_mode::STRIP_H_BODY_DEFAULT),
    };
    let _capped = MemoryMode::CappedPyramid { levels: 5 };
    // Exhaustive match — would fail to compile if a variant were
    // added without updating this test.
    let m = MemoryMode::Auto;
    let _ = match m {
        MemoryMode::Auto => 0u32,
        MemoryMode::Full => 1u32,
        MemoryMode::Strip { .. } => 2u32,
        MemoryMode::StripPair { .. } => 3u32,
        MemoryMode::CappedPyramid { .. } => 4u32,
    };
}

#[test]
fn strip_pair_estimator_aligned_validation() {
    use cvvdp_gpu::memory_mode::estimate_gpu_memory_bytes_for_mode;
    // Unaligned h_body yields usize::MAX (estimator-level "invalid").
    let bad = estimate_gpu_memory_bytes_for_mode(
        1024,
        1024,
        MemoryMode::StripPair { h_body: Some(100) },
    );
    assert_eq!(bad, usize::MAX);
    // Aligned h_body yields a real estimate.
    let ok = estimate_gpu_memory_bytes_for_mode(
        1024,
        1024,
        MemoryMode::StripPair {
            h_body: Some(memory_mode::STRIP_H_BODY_DEFAULT),
        },
    );
    assert!(ok < usize::MAX);
    assert!(ok > 0);
}

#[test]
fn strip_pair_estimator_smaller_than_strip_mode() {
    use cvvdp_gpu::memory_mode::estimate_gpu_memory_bytes_for_mode;
    // Mode B (StripPair) does NOT allocate the dedicated
    // `RefFullState`; Mode E (Strip) does. So at equal `h_body`,
    // Mode B's conservative bound is strictly less than Mode E's.
    let body = Some(memory_mode::STRIP_H_BODY_DEFAULT);
    let pair = estimate_gpu_memory_bytes_for_mode(
        2048,
        2048,
        MemoryMode::StripPair { h_body: body },
    );
    let cached_ref = estimate_gpu_memory_bytes_for_mode(
        2048,
        2048,
        MemoryMode::Strip { h_body: body },
    );
    assert!(
        pair < cached_ref,
        "Mode B ({pair}) should be smaller than Mode E ({cached_ref}) — \
         Mode B skips RefFullState",
    );
}

#[test]
fn umbrella_strip_maps_to_cvvdp_strip() {
    // Task #79: the umbrella `MemoryMode::Strip { h_body }` maps to
    // cvvdp's own `Strip { h_body }` (no longer falls back to Auto
    // since the JOD-preserving Mode E variant ships).
    //
    // Tile { h, w } still falls back to Auto — no tile-walker is
    // implemented in any per-crate pipeline.
    //
    // The actual zenmetrics_api::MemoryMode → cvvdp_gpu::MemoryMode
    // From impl is covered by zenmetrics-api's own tests; here we
    // pin that cvvdp's Auto path resolves to Full at small sizes
    // (the standard happy path).
    let cap = memory_mode::vram_cap_bytes();
    let r = memory_mode::resolve_auto(256, 256, cap).expect("Auto should resolve");
    assert_eq!(r, ResolvedMode::Full);
}

#[test]
fn strip_align_matches_max_levels() {
    // STRIP_ALIGN must equal 2^(MAX_LEVELS - 1) so the per-level
    // ceil-div halving in the Weber pyramid doesn't drift through
    // the strip body boundary. MAX_LEVELS = 9 → STRIP_ALIGN = 256.
    assert_eq!(
        memory_mode::STRIP_ALIGN,
        1 << (cvvdp_gpu::MAX_LEVELS as u32 - 1)
    );
    assert_eq!(memory_mode::STRIP_ALIGN, 256);
}

#[test]
fn strip_h_body_default_is_aligned_and_positive() {
    assert!(memory_mode::STRIP_H_BODY_DEFAULT > 0);
    assert_eq!(
        memory_mode::STRIP_H_BODY_DEFAULT % memory_mode::STRIP_ALIGN,
        0
    );
}

#[test]
fn vram_cap_env_override() {
    with_cap(Some("123456789"), || {
        assert_eq!(memory_mode::vram_cap_bytes(), 123_456_789);
    });
}

#[test]
fn vram_cap_default_is_8gb() {
    // Task #51 (commit e6660cc1) added a live nvidia-smi VRAM probe that
    // takes precedence over the 8 GiB fallback when the env var is
    // unset. The fallback only kicks in when the probe fails (no
    // nvidia-smi, AMD/Intel GPU, CI runner, etc.). Both are valid
    // contracts — the test verifies whichever applies on this host.
    with_cap(None, || {
        let cap = memory_mode::vram_cap_bytes();
        let probe = memory_mode::live_vram_probe_bytes();
        match probe {
            None => assert_eq!(cap, 8 * 1024 * 1024 * 1024),
            Some(p) => assert_eq!(cap, p, "must equal live probe when available"),
        }
    });
}

#[test]
fn vram_cap_env_override_wins_over_probe() {
    with_cap(Some("17179869184"), || {
        assert_eq!(memory_mode::vram_cap_bytes(), 17_179_869_184);
    });
}

#[test]
fn live_probe_returns_sensible_value_when_available() {
    let probe = memory_mode::live_vram_probe_bytes();
    if let Some(bytes) = probe {
        assert!(bytes > 0, "live probe returned zero bytes");
        assert!(
            bytes <= 1024 * 1024 * 1024 * 1024,
            "live probe absurdly large: {bytes}"
        );
    }
}
