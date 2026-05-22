//! Cross-crate auto-fallback contract tests for butteraugli-gpu.
//!
//! butter is **strip-preferred**: when both Full and Strip fit the cap,
//! Auto picks Strip (the strip walker is 1.9-4.9× faster on this
//! crate — see `benchmarks/butter_strip_vs_whole_2026-05-21.md`).
//! When Strip is too small to be worthwhile (image_h ≤ MIN_STRIP_BODY
//! + 2×HALO_ROWS) Auto falls through to Full instead.
//!
//! Pins:
//! - Auto picks Strip when both fit (strip-preferred).
//! - Auto picks Full at small image sizes (degenerate strip case).
//! - Auto picks Strip when Full exceeds the cap (last-ditch).
//! - Auto's Strip body fits inside the cap.
//! - Auto returns TooBigForFull only when neither fits.
//!
//! Host-side only. No GPU integration here.

use butteraugli_gpu::{
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
fn auto_picks_strip_when_generous_cap_and_large_image() {
    // Strip-preferred: with a generous 16 GiB cap and a 1024² image
    // (well above the small-image floor), Auto MUST pick Strip even
    // though Full would also fit.
    with_cap(Some("17179869184"), || {
        let cap = memory_mode::vram_cap_bytes();
        let r = memory_mode::resolve_auto(1024, 1024, cap).expect("resolve");
        assert!(
            matches!(r, ResolvedMode::Strip { .. }),
            "butter is strip-preferred when both fit, got {r:?}"
        );
    });
}

#[test]
fn auto_picks_full_at_small_image() {
    // Small image (64×64) is below the degenerate-strip threshold —
    // butter's resolver falls back to Full even though Strip would
    // technically allocate. The small-image fix at 99330592 pins this.
    with_cap(Some("17179869184"), || {
        let cap = memory_mode::vram_cap_bytes();
        let r = memory_mode::resolve_auto(64, 64, cap).expect("resolve");
        assert_eq!(
            r,
            ResolvedMode::Full,
            "small images must fall through to Full (degenerate strip)"
        );
    });
}

#[test]
fn auto_picks_strip_when_full_exceeds_cap() {
    // 4096² image, cap somewhere between strip-min and full estimate.
    // Auto MUST pick Strip — this is the "Strip last-ditch when Full
    // doesn't fit" branch of the resolver.
    let full_est = estimate_gpu_memory_bytes(4096, 4096);
    let strip_min = estimate_strip_gpu_memory_bytes(4096, 64).unwrap();
    assert!(
        strip_min < full_est,
        "test premise: strip-min < full at 4096²"
    );
    let cap_bytes = strip_min.saturating_mul(3).min(full_est - 1);
    with_cap(Some(&cap_bytes.to_string()), || {
        let cap = memory_mode::vram_cap_bytes();
        let r = memory_mode::resolve_auto(4096, 4096, cap).expect("resolve");
        match r {
            ResolvedMode::Strip { h_body } => {
                assert!(h_body > 0, "h_body must be positive");
            }
            other => panic!("expected Strip when Full > cap, got {other:?}"),
        }
    });
}

#[test]
fn auto_strip_h_body_fits_inside_cap() {
    let cap_bytes = estimate_strip_gpu_memory_bytes(4096, 256).unwrap();
    with_cap(Some(&cap_bytes.to_string()), || {
        let cap = memory_mode::vram_cap_bytes();
        let r = memory_mode::resolve_auto(4096, 4096, cap).expect("resolve");
        match r {
            ResolvedMode::Strip { h_body } => {
                let est = estimate_strip_gpu_memory_bytes(4096, h_body).unwrap();
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
        let r = memory_mode::resolve_auto(4096, 4096, cap);
        assert!(
            matches!(r, Err(Error::TooBigForFull { .. })),
            "cap=1 must error, got {r:?}"
        );
    });
}

#[test]
fn auto_fallback_at_explicit_tiny_cap() {
    // Task recipe: `ZENMETRICS_VRAM_CAP_BYTES=1000000` forces Strip
    // selection (small body) when Full doesn't fit on a large image.
    // At 4096² Full is ~3 GB, well above 1 MB. The smallest aligned
    // strip body must fit or we return TooBigForFull. Empirically at
    // butter's planes count the smallest strip body also exceeds 1 MB,
    // so this is a TooBigForFull at this size.
    with_cap(Some("1000000"), || {
        let cap = memory_mode::vram_cap_bytes();
        let r = memory_mode::resolve_auto(4096, 4096, cap);
        // Pin behaviour: with 1 MB cap and 4096² image neither path fits.
        assert!(
            matches!(r, Err(Error::TooBigForFull { .. })),
            "1 MB cap + 4096² = TooBigForFull, got {r:?}"
        );
    });
}

#[test]
fn auto_huge_cap_picks_strip_at_24mp() {
    // 1 TiB cap, 24 MP image — Full fits easily but butter is
    // strip-preferred, so Auto still picks Strip.
    with_cap(Some("1099511627776"), || {
        let cap = memory_mode::vram_cap_bytes();
        let r = memory_mode::resolve_auto(6000, 4000, cap).expect("resolve");
        assert!(
            matches!(r, ResolvedMode::Strip { .. }),
            "strip-preferred at huge cap, got {r:?}"
        );
    });
}
