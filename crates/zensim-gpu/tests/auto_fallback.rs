//! Auto-fallback contract tests for zensim-gpu.
//!
//! zensim-gpu has no Strip implementation (the 4-channel + 4-scale +
//! Extended-regime allocator is interlocked enough that strip would
//! need a dedicated design pass — see the module-level docs in
//! `src/memory_mode.rs`). Auto resolves to Full whenever Full fits,
//! and to TooBigForFull otherwise. There is no auto-fallback path,
//! by design.
//!
//! Pins:
//! - Auto picks Full at generous cap.
//! - Auto returns TooBigForFull when Full exceeds cap.
//! - Strip estimator stays `None`.

use std::sync::{Mutex, OnceLock};
use zensim_gpu::{Error, MemoryMode, ResolvedMode, estimate_strip_gpu_memory_bytes, memory_mode};

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
fn auto_picks_full_when_cap_generous() {
    with_cap(Some("17179869184"), || {
        let cap = memory_mode::vram_cap_bytes();
        let r = memory_mode::resolve_auto(1024, 1024, cap).expect("resolve");
        assert_eq!(r, ResolvedMode::Full);
    });
}

#[test]
fn auto_returns_too_big_when_full_exceeds_cap() {
    // No strip path: when Full doesn't fit, Auto can only error.
    with_cap(Some("1"), || {
        let cap = memory_mode::vram_cap_bytes();
        let r = memory_mode::resolve_auto(4096, 4096, cap);
        match r {
            Err(Error::TooBigForFull { needed, cap: c }) => {
                assert!(needed > 0);
                assert_eq!(c, 1);
            }
            other => panic!("expected TooBigForFull, got {other:?}"),
        }
    });
}

#[test]
fn auto_returns_too_big_at_explicit_tiny_cap() {
    with_cap(Some("1000000"), || {
        let cap = memory_mode::vram_cap_bytes();
        let r = memory_mode::resolve_auto(4096, 4096, cap);
        assert!(
            matches!(r, Err(Error::TooBigForFull { .. })),
            "1 MB cap + 4096² must error (no Strip path), got {r:?}"
        );
    });
}

#[test]
fn strip_estimator_is_none_by_design() {
    // zensim deliberately has no per-strip estimator — the Extended-
    // regime allocator interlocks per-scale persist planes with the
    // working set, so a strip-mode design has to come first.
    assert!(estimate_strip_gpu_memory_bytes(1024, 64).is_none());
    assert!(estimate_strip_gpu_memory_bytes(8192, 256).is_none());
}

#[test]
fn explicit_strip_variant_carries_through_enum() {
    let m = MemoryMode::Strip { h_body: Some(128) };
    match m {
        MemoryMode::Strip { h_body } => assert_eq!(h_body, Some(128)),
        _ => panic!("expected Strip variant"),
    }
}

#[test]
fn auto_huge_cap_picks_full() {
    with_cap(Some("1099511627776"), || {
        let cap = memory_mode::vram_cap_bytes();
        let r = memory_mode::resolve_auto(6000, 4000, cap).expect("resolve");
        assert_eq!(r, ResolvedMode::Full);
    });
}
