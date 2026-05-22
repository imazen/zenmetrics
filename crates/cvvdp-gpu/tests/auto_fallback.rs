//! Auto-fallback contract tests for cvvdp-gpu.
//!
//! cvvdp-gpu does NOT auto-fall back to Strip: capping the pyramid
//! depth changes the JOD value (see `docs/STRIP_PROCESSING.md` and
//! `benchmarks/cvvdp_capped_levels_2026-05-22.csv`), so Auto cannot
//! silently pick the capped path. Callers who want capped-Strip must
//! opt in explicitly via `MemoryMode::Strip { capped_levels: Some(k) }`.
//!
//! Pins:
//! - Auto picks Full at generous cap.
//! - Auto returns TooBigForFull when Full exceeds cap (no auto-cap).
//! - `Strip { capped_levels: Some(_) }` constructs correctly (typed).
//! - Strip estimator stays `None` (no per-strip estimate path).

use cvvdp_gpu::{Error, MemoryMode, ResolvedMode, estimate_strip_gpu_memory_bytes, memory_mode};
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
fn auto_returns_too_big_no_silent_capped_fallback() {
    // cvvdp deliberately does NOT auto-fall back to `Strip { capped_levels }`
    // even though that path WOULD fit a tighter cap — capping changes
    // the JOD value, so picking it silently would mislead callers.
    // Tighter-cap scenarios surface TooBigForFull; callers can then
    // opt in to capped-Strip explicitly.
    with_cap(Some("1"), || {
        let cap = memory_mode::vram_cap_bytes();
        let r = memory_mode::resolve_auto(4096, 4096, cap);
        match r {
            Err(Error::TooBigForFull { needed, cap: c }) => {
                assert!(needed > 0);
                assert_eq!(c, 1);
            }
            other => panic!("expected TooBigForFull (no auto-cap), got {other:?}"),
        }
    });
}

#[test]
fn auto_returns_too_big_at_tight_cap_even_when_capped_strip_would_fit() {
    // Cvvdp's capped-Strip variant *would* fit a 1 GB cap on most 4 MP
    // images, but Auto must refuse to pick it. Pins the
    // "no silent metric-altering fallback" contract.
    with_cap(Some("1000000"), || {
        let cap = memory_mode::vram_cap_bytes();
        let r = memory_mode::resolve_auto(4096, 4096, cap);
        assert!(
            matches!(r, Err(Error::TooBigForFull { .. })),
            "Auto must NOT silently switch to capped-Strip, got {r:?}"
        );
    });
}

#[test]
fn explicit_strip_with_cap_is_addressable() {
    // Confirms the capped variant is constructable via the typed enum
    // path — callers can opt in to the depth-cap fallback explicitly.
    let m = MemoryMode::Strip {
        h_body: None,
        capped_levels: Some(8),
    };
    match m {
        MemoryMode::Strip {
            h_body,
            capped_levels,
        } => {
            assert_eq!(capped_levels, Some(8));
            assert_eq!(h_body, None);
        }
        _ => panic!("expected Strip"),
    }
}

#[test]
fn strip_estimator_remains_none_per_design() {
    // Per the design note in memory_mode.rs: cvvdp has no separate
    // strip-mode estimator (the spatial-frequency decomposition is
    // full-image by construction), so the strip estimator returns
    // None and the resolver doesn't try to fall back.
    assert!(estimate_strip_gpu_memory_bytes(1024, 64).is_none());
    assert!(estimate_strip_gpu_memory_bytes(8192, 256).is_none());
}

#[test]
fn auto_huge_cap_picks_full() {
    with_cap(Some("1099511627776"), || {
        let cap = memory_mode::vram_cap_bytes();
        let r = memory_mode::resolve_auto(6000, 4000, cap).expect("resolve");
        assert_eq!(r, ResolvedMode::Full);
    });
}
