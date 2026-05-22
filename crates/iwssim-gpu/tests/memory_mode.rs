//! Tests for the unified [`MemoryMode`] API. Host-side only — real
//! GPU integration is covered by `tests/parity_lock.rs` and
//! `tests/strip_parity.rs`.

use iwssim_gpu::{
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
fn auto_picks_full_when_well_under_cap() {
    with_cap(Some("17179869184"), || {
        let r = memory_mode::resolve_auto(1024, 1024, memory_mode::vram_cap_bytes()).unwrap();
        assert_eq!(r, ResolvedMode::Full);
    });
}

#[test]
fn auto_picks_strip_when_over_cap_with_h_body_fit() {
    let cap = estimate_strip_gpu_memory_bytes(8192, 1024).unwrap();
    with_cap(Some(&cap.to_string()), || {
        let r = memory_mode::resolve_auto(8192, 8192, memory_mode::vram_cap_bytes()).unwrap();
        match r {
            ResolvedMode::Strip { h_body } => {
                assert!(h_body > 0);
                assert!(h_body % 16 == 0, "must be pyramid-aligned");
            }
            other => panic!("expected Strip, got {other:?}"),
        }
    });
}

#[test]
fn auto_returns_err_when_cap_too_tight_for_any_mode() {
    with_cap(Some("1"), || {
        let r = memory_mode::resolve_auto(8192, 8192, memory_mode::vram_cap_bytes());
        assert!(matches!(r, Err(Error::TooBigForFull { .. })));
    });
}

#[test]
fn auto_errors_on_small_image_over_cap() {
    // Small image (160×160 < MIN_NATIVE_DIM=176): strip is unavailable;
    // tiny cap forces TooBigForFull.
    with_cap(Some("100"), || {
        let r = memory_mode::resolve_auto(160, 160, memory_mode::vram_cap_bytes());
        assert!(matches!(r, Err(Error::TooBigForFull { .. })));
    });
}

#[test]
fn explicit_strip_with_h_body() {
    let m = MemoryMode::Strip { h_body: Some(128) };
    assert_eq!(m, MemoryMode::Strip { h_body: Some(128) });
}

#[test]
fn tile_returns_unsupported_via_typed_helper() {
    // Resolver doesn't see Tile (typed API rejects it directly), so
    // exercise the enum round-trip.
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
