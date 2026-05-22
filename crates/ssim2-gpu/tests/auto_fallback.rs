//! Cross-crate auto-fallback contract tests for ssim2-gpu.
//!
//! ssim2's strip processing shipped Phase 2 at 4a17f5db; this file
//! pins the auto-fallback contract that the unified MemoryMode API
//! requires across all strip-capable metric crates.
//!
//! Pins:
//! - Auto picks Full at generous cap.
//! - Auto picks Strip with an auto-sized body when Full exceeds the cap.
//! - Auto's Strip body fits the cap.
//! - Auto returns TooBigForFull only when neither path fits.
//! - At a tight 4 GB cap on 24 MP, Auto picks Strip (per the
//!   STRIP_H_BODY_DEFAULT/halo budget in `docs/STRIP_PROCESSING.md`).
//!
//! Host-side only.

use ssim2_gpu::{
    Error, ResolvedMode, estimate_gpu_memory_bytes, estimate_strip_gpu_memory_bytes, memory_mode,
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
fn auto_picks_full_when_cap_generous() {
    with_cap(Some("17179869184"), || {
        let cap = memory_mode::vram_cap_bytes();
        let r = memory_mode::resolve_auto(1024, 1024, cap).expect("resolve");
        assert_eq!(r, ResolvedMode::Full, "Full-preferred at generous cap");
    });
}

#[test]
fn auto_picks_strip_when_full_exceeds_cap() {
    // 24 MP image at 4 GB cap. Per docs/STRIP_PROCESSING.md, Full is
    // ~7.5 GB and Strip with default body is ~2.87 GB — Auto picks
    // Strip.
    with_cap(Some("4294967296"), || {
        let cap = memory_mode::vram_cap_bytes();
        let r = memory_mode::resolve_auto(6000, 4000, cap).expect("resolve");
        match r {
            ResolvedMode::Strip { h_body } => {
                assert!(h_body > 0, "h_body must be positive");
                assert!(
                    h_body <= 4000,
                    "h_body must not exceed image height: {h_body}"
                );
            }
            other => panic!("expected Strip when Full > cap, got {other:?}"),
        }
    });
}

#[test]
fn auto_strip_h_body_fits_inside_cap() {
    // Cap = strip estimate at h_body=512. Auto-resolver must pick a
    // body whose estimated bytes ≤ this cap.
    let cap_bytes = estimate_strip_gpu_memory_bytes(6000, 512).unwrap();
    let full_est = estimate_gpu_memory_bytes(6000, 4000);
    assert!(cap_bytes < full_est, "premise: strip-cap < full");
    with_cap(Some(&cap_bytes.to_string()), || {
        let cap = memory_mode::vram_cap_bytes();
        let r = memory_mode::resolve_auto(6000, 4000, cap).expect("resolve");
        match r {
            ResolvedMode::Strip { h_body } => {
                let est = estimate_strip_gpu_memory_bytes(6000, h_body).unwrap();
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
    with_cap(Some("1099511627776"), || {
        let cap = memory_mode::vram_cap_bytes();
        let r = memory_mode::resolve_auto(6000, 4000, cap).expect("resolve");
        assert_eq!(r, ResolvedMode::Full);
    });
}

#[test]
fn auto_fallback_at_explicit_tiny_cap() {
    // 1 MB cap on a 4096² image: Full and even small strip body both
    // exceed the cap. Auto MUST surface TooBigForFull.
    with_cap(Some("1000000"), || {
        let cap = memory_mode::vram_cap_bytes();
        let r = memory_mode::resolve_auto(4096, 4096, cap);
        assert!(
            matches!(r, Err(Error::TooBigForFull { .. })),
            "1 MB cap + 4096² must error, got {r:?}"
        );
    });
}

#[test]
fn auto_searches_smaller_body_when_default_does_not_fit() {
    // The auto-fallback contract: when STRIP_H_BODY_DEFAULT (1024)
    // doesn't fit but a smaller body would, the resolver must pick
    // the smaller body rather than giving up. This was a regression
    // hazard before the auto-search landed.
    let small_body_bytes = estimate_strip_gpu_memory_bytes(6000, 64).unwrap();
    let default_body_bytes = estimate_strip_gpu_memory_bytes(6000, 1024).unwrap();
    assert!(
        small_body_bytes < default_body_bytes,
        "premise: small body is cheaper than default"
    );
    // Cap is small enough that the default body won't fit, but at
    // least the 64-row body must.
    let cap_bytes = small_body_bytes.saturating_mul(2).min(default_body_bytes - 1);
    with_cap(Some(&cap_bytes.to_string()), || {
        let cap = memory_mode::vram_cap_bytes();
        let r = memory_mode::resolve_auto(6000, 4000, cap).expect("resolve");
        match r {
            ResolvedMode::Strip { h_body } => {
                let est = estimate_strip_gpu_memory_bytes(6000, h_body).unwrap();
                assert!(
                    est <= cap_bytes,
                    "auto-search picked body ({h_body}) whose estimate ({est}) exceeds cap ({cap_bytes})"
                );
            }
            other => panic!("expected Strip with auto-searched body, got {other:?}"),
        }
    });
}
