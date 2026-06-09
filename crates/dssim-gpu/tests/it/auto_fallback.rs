//! Cross-crate auto-fallback contract tests for dssim-gpu.
//!
//! dssim is **NOT strip-preferred** (strip walker is 2-5× slower than
//! whole-image per `benchmarks/dssim_strip_vs_whole_2026-05-22.md`),
//! so Auto picks Full whenever it fits the cap and Strip only when
//! Full would exceed the cap. Pins:
//!
//! - Auto picks Full at generous cap.
//! - Auto picks Strip when Full exceeds the cap (last-ditch).
//! - Auto's Strip body is pyramid-aligned and fits the cap.
//! - Auto returns TooBigForFull only when neither fits.
//!
//! Host-side only.

use dssim_gpu::{
    Error, ResolvedMode, estimate_gpu_memory_bytes, estimate_strip_gpu_memory_bytes, memory_mode,
};
use std::sync::{Mutex, OnceLock};

const VRAM_CAP_VAR: &str = "ZENMETRICS_VRAM_CAP_BYTES";
/// `1 << (NUM_SCALES - 1)` with `NUM_SCALES = 5`.
const PYRAMID_ALIGN: u32 = 16;

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
        assert_eq!(r, ResolvedMode::Full, "Full-preferred at generous cap");
    });
}

#[test]
fn auto_picks_strip_when_full_exceeds_cap() {
    let full_est = estimate_gpu_memory_bytes(8192, 8192);
    let strip_min = estimate_strip_gpu_memory_bytes(8192, PYRAMID_ALIGN).unwrap();
    assert!(strip_min < full_est, "premise: strip-min < full");
    let cap_bytes = strip_min.saturating_mul(2).min(full_est - 1);
    with_cap(Some(&cap_bytes.to_string()), || {
        let cap = memory_mode::vram_cap_bytes();
        let r = memory_mode::resolve_auto(8192, 8192, cap).expect("resolve");
        match r {
            ResolvedMode::Strip { h_body } => {
                assert!(h_body > 0);
                assert_eq!(
                    h_body % PYRAMID_ALIGN,
                    0,
                    "h_body must be pyramid-aligned, got {h_body}"
                );
            }
            other => panic!("expected Strip when Full > cap, got {other:?}"),
        }
    });
}

#[test]
fn auto_strip_h_body_fits_inside_cap() {
    let cap_bytes = estimate_strip_gpu_memory_bytes(8192, 256).unwrap();
    with_cap(Some(&cap_bytes.to_string()), || {
        let cap = memory_mode::vram_cap_bytes();
        let r = memory_mode::resolve_auto(8192, 8192, cap).expect("resolve");
        match r {
            ResolvedMode::Strip { h_body } => {
                let est = estimate_strip_gpu_memory_bytes(8192, h_body).unwrap();
                assert!(
                    est <= cap_bytes,
                    "auto-sized strip ({est}) exceeds cap ({cap_bytes})"
                );
            }
            other => panic!("expected Strip, got {other:?}"),
        }
    });
}

#[test]
fn auto_returns_too_big_when_neither_fits() {
    with_cap(Some("1"), || {
        let cap = memory_mode::vram_cap_bytes();
        let r = memory_mode::resolve_auto(8192, 8192, cap);
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
fn auto_huge_cap_picks_full_at_24mp() {
    // 24 MP image under a 1 TiB cap: Full fits comfortably and dssim
    // is Full-preferred — Auto stays on Full.
    with_cap(Some("1099511627776"), || {
        let cap = memory_mode::vram_cap_bytes();
        let r = memory_mode::resolve_auto(6000, 4000, cap).expect("resolve");
        assert_eq!(r, ResolvedMode::Full);
    });
}

#[test]
fn auto_fallback_at_explicit_tiny_cap() {
    // 1 MB cap is too tight for any plausible strip body at 4096²
    // (strip baseline alone exceeds 1 MB at 4096-wide).
    with_cap(Some("1000000"), || {
        let cap = memory_mode::vram_cap_bytes();
        let r = memory_mode::resolve_auto(4096, 4096, cap);
        assert!(
            matches!(r, Err(Error::TooBigForFull { .. })),
            "1 MB cap + 4096² must error, got {r:?}"
        );
    });
}
