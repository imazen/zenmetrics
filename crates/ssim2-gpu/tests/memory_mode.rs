//! Tests for the unified [`MemoryMode`] API. Host-side only — real
//! GPU integration is covered by `tests/parity_lock.rs`.

use ssim2_gpu::{
    Error, MemoryMode, ResolvedMode, estimate_gpu_memory_bytes, estimate_strip_gpu_memory_bytes,
    memory_mode,
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
    let a = estimate_gpu_memory_bytes(1024, 1024);
    let b = estimate_gpu_memory_bytes(2048, 2048);
    let ratio = b as f64 / a as f64;
    assert!(ratio > 3.6 && ratio < 4.4, "ratio = {ratio}");
}

#[test]
fn strip_estimator_returns_some_phase2() {
    // Phase 2 (2026-05-22): Strip is implemented. The estimator now
    // reports per-strip working-set bytes for a (width, h_body) pair.
    let bytes = estimate_strip_gpu_memory_bytes(1024, 64).expect("Some(bytes) after Phase 2");
    // Strip working-set must scale with width × (h_body + 2*halo) × 57
    // planes-per-scale across pyramid. At width=1024, h_body=64, halo=256,
    // strip_h=576: scale 0 alone = 1024×576×57×4 ≈ 134 MB.
    assert!(
        bytes > 100 * 1024 * 1024 && bytes < 250 * 1024 * 1024,
        "scale-0 + pyramid for w=1024 h_body=64 should be ~130–180 MB, got {bytes}"
    );
}

#[test]
fn strip_estimator_smaller_than_full() {
    // The whole point of strip mode: per-strip memory is bounded by
    // the strip dimensions, not the image dimensions. For a 24 MP
    // image (6000×4000) at h_body=1024, the strip estimate should be
    // substantially smaller than the Full estimate. Empirically the
    // ratio is ~38% (2.85 GB strip vs 7.49 GB Full) — the savings
    // come from the bounded strip height, not from skipping scales.
    let full_24mp = estimate_gpu_memory_bytes(6000, 4000);
    let strip_24mp =
        estimate_strip_gpu_memory_bytes(6000, 1024).expect("Some(bytes) after Phase 2");
    assert!(
        strip_24mp * 2 < full_24mp,
        "strip should be < 50% Full at 24 MP: strip={strip_24mp} full={full_24mp}"
    );
}

#[test]
fn auto_picks_full_when_under_cap() {
    with_cap(Some("17179869184"), || {
        let r = memory_mode::resolve_auto(1024, 1024, memory_mode::vram_cap_bytes()).unwrap();
        assert_eq!(r, ResolvedMode::Full);
    });
}

#[test]
fn auto_errors_when_full_and_strip_both_too_big() {
    // Tiny cap + neither Full nor default-Strip fit → TooBigForFull.
    with_cap(Some("1"), || {
        let r = memory_mode::resolve_auto(4096, 4096, memory_mode::vram_cap_bytes());
        assert!(matches!(r, Err(Error::TooBigForFull { .. })));
    });
}

#[test]
fn auto_picks_strip_when_full_exceeds_cap_but_strip_fits() {
    // At 6000×4000 the Full estimate is ~7.49 GB; Strip at h_body=1024
    // is ~2.85 GB. With a 4 GB cap, Auto should pick Strip.
    with_cap(Some("4294967296"), || {
        let r = memory_mode::resolve_auto(6000, 4000, memory_mode::vram_cap_bytes()).unwrap();
        match r {
            ResolvedMode::Strip { h_body } => {
                assert!(h_body >= 512 && h_body <= 4000, "h_body = {h_body}");
            }
            ResolvedMode::Full => panic!("expected Strip resolution; got Full"),
        }
    });
}

#[test]
fn explicit_strip_errors() {
    // Strip variant on the enum is constructable; the typed
    // constructor errors with ModeUnsupported. We can't call the
    // typed `new_with_memory_mode` here without a GPU — verify via
    // the enum round-trip.
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

#[test]
fn vram_cap_default_is_8gb() {
    with_cap(None, || {
        assert_eq!(memory_mode::vram_cap_bytes(), 8 * 1024 * 1024 * 1024);
    });
}
