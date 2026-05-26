//! Auto-fallback contract tests for cvvdp-gpu.
//!
//! cvvdp-gpu only supports Full + Auto as of task #77 — the
//! capped-pyramid Strip variant was rolled back because it changed
//! the JOD value at any k < 9 (see `docs/STRIP_PROCESSING.md`).
//! Auto always resolves to Full when it fits the cap; otherwise it
//! surfaces TooBigForFull so the caller can pick a different metric
//! or split the image at the application layer.
//!
//! Pins:
//! - Auto picks Full at generous cap.
//! - Auto returns TooBigForFull when Full exceeds cap (no silent
//!   metric-altering fallback).

use cvvdp_gpu::{Error, ResolvedMode, memory_mode};
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
fn auto_picks_full_when_cap_generous() {
    with_cap(Some("17179869184"), || {
        let cap = memory_mode::vram_cap_bytes();
        let r = memory_mode::resolve_auto(1024, 1024, cap).expect("resolve");
        assert_eq!(r, ResolvedMode::Full);
    });
}

#[test]
fn auto_returns_too_big_no_silent_fallback() {
    // cvvdp deliberately surfaces TooBigForFull when Full exceeds
    // the cap — there is no Strip / Tile path to fall back to.
    // (Pre-task-#77, this test asserted the absence of capped-Strip
    // auto-selection. The variant no longer exists, but the
    // contract — Auto must not silently change the metric value —
    // still holds, and is documented in `docs/STRIP_PROCESSING.md`.)
    with_cap(Some("1"), || {
        let cap = memory_mode::vram_cap_bytes();
        let r = memory_mode::resolve_auto(4096, 4096, cap);
        match r {
            Err(Error::TooBigForFull { needed, cap: c }) => {
                assert!(needed > 0);
                assert_eq!(c, 1);
            }
            other => panic!("expected TooBigForFull (no fallback), got {other:?}"),
        }
    });
}

#[test]
fn auto_returns_too_big_at_tight_cap() {
    // Pins the "no silent metric-altering fallback" contract: tight
    // caps surface TooBigForFull rather than picking a smaller-memory
    // path that would change the JOD value.
    with_cap(Some("1000000"), || {
        let cap = memory_mode::vram_cap_bytes();
        let r = memory_mode::resolve_auto(4096, 4096, cap);
        assert!(
            matches!(r, Err(Error::TooBigForFull { .. })),
            "Auto must surface TooBigForFull, got {r:?}"
        );
    });
}

#[test]
fn auto_huge_cap_picks_full() {
    with_cap(Some("1099511627776"), || {
        let cap = memory_mode::vram_cap_bytes();
        let r = memory_mode::resolve_auto(6000, 4000, cap).expect("resolve");
        assert_eq!(r, ResolvedMode::Full);
    });
}
