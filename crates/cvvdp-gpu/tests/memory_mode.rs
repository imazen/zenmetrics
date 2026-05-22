//! Tests for the unified [`MemoryMode`] API. Host-side only — real
//! GPU integration is covered by `tests/pipeline_score.rs` and
//! friends.

use cvvdp_gpu::{
    Error, MemoryMode, ResolvedMode, estimate_gpu_memory_bytes_usize,
    estimate_strip_gpu_memory_bytes, memory_mode,
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
fn strip_estimator_is_none() {
    assert!(estimate_strip_gpu_memory_bytes(1024, 64).is_none());
}

#[test]
fn auto_picks_full_when_under_cap() {
    with_cap(Some("17179869184"), || {
        let r = memory_mode::resolve_auto(1024, 1024, memory_mode::vram_cap_bytes()).unwrap();
        assert_eq!(r, ResolvedMode::Full);
    });
}

#[test]
fn auto_errors_when_strip_unsupported_and_too_big() {
    with_cap(Some("1"), || {
        let r = memory_mode::resolve_auto(4096, 4096, memory_mode::vram_cap_bytes());
        assert!(matches!(r, Err(Error::TooBigForFull { .. })));
    });
}

#[test]
fn explicit_strip_constructs() {
    let m = MemoryMode::Strip { h_body: Some(128), capped_levels: None };
    assert_eq!(
        m,
        MemoryMode::Strip { h_body: Some(128), capped_levels: None }
    );
}

#[test]
fn explicit_strip_with_cap_constructs() {
    // The capped-pyramid variant — `Strip { capped_levels: Some(k) }`
    // is what unblocks 24 MP square via reduced σ=3 PU-blur halo (see
    // `docs/STRIP_PROCESSING.md`).
    let m = MemoryMode::Strip { h_body: None, capped_levels: Some(8) };
    match m {
        MemoryMode::Strip { capped_levels, h_body } => {
            assert_eq!(capped_levels, Some(8));
            assert_eq!(h_body, None);
        }
        _ => panic!("expected Strip"),
    }
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
