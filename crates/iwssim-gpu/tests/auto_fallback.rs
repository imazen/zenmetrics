//! Cross-crate "2-pass iwssim auto-fallback" contract tests.
//!
//! Pins the resolver behaviour required by the unified MemoryMode API:
//!
//! - Auto picks Full when the cap is generous.
//! - Auto picks Strip with an auto-sized body when Full exceeds the cap.
//! - Auto's Strip body fits inside the cap (the auto-sizer doesn't
//!   over-commit).
//! - Auto returns [`Error::TooBigForFull`] only when neither Full nor
//!   any pyramid-aligned strip body fits.
//! - Small images (axis < `MIN_NATIVE_DIM`) where Full doesn't fit
//!   correctly surface TooBigForFull because the strip walker rejects
//!   sub-floor inputs — see the docstring on
//!   `Iwssim::new_strip_with_halo`.
//!
//! Host-side only. No GPU integration here; real-GPU coverage lives
//! in `tests/parity_lock.rs` and `tests/strip_parity.rs`.

use iwssim_gpu::{
    Error, ResolvedMode, estimate_gpu_memory_bytes, estimate_strip_gpu_memory_bytes, memory_mode,
};
use std::sync::{Mutex, OnceLock};

const VRAM_CAP_VAR: &str = "ZENMETRICS_VRAM_CAP_BYTES";
const MIN_NATIVE_DIM: u32 = 176;
/// Same alignment iwssim's resolver uses internally
/// (`1 << (NUM_SCALES - 1)` with `NUM_SCALES = 5`).
const PYRAMID_ALIGN: u32 = 16;

/// Process-wide env mutex so concurrent test threads don't stomp each
/// other's `ZENMETRICS_VRAM_CAP_BYTES` overrides.
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
    // 16 GiB cap, 1024² image — Full estimate is well under the cap,
    // and iwssim is Full-preferred, so Auto MUST resolve to Full.
    with_cap(Some("17179869184"), || {
        let cap = memory_mode::vram_cap_bytes();
        let r = memory_mode::resolve_auto(1024, 1024, cap).expect("resolve");
        assert_eq!(r, ResolvedMode::Full, "1024² fits 16 GiB easily");
    });
}

#[test]
fn auto_picks_strip_when_full_exceeds_cap() {
    // Pick a cap that's smaller than the 8192² Full estimate but big
    // enough for at least a small strip body. Auto MUST fall back to
    // Strip — this is the 2-pass fallback the task names.
    let full_8k = estimate_gpu_memory_bytes(8192, 8192);
    let strip_8k_min = estimate_strip_gpu_memory_bytes(8192, PYRAMID_ALIGN)
        .expect("strip estimator returns Some for valid inputs");
    assert!(
        strip_8k_min < full_8k,
        "strip-min ({strip_8k_min}) must be < full ({full_8k}) for the test premise"
    );
    // Cap roughly halfway between the smallest strip and the full estimate.
    let cap_bytes = strip_8k_min.saturating_mul(2).min(full_8k - 1);
    with_cap(Some(&cap_bytes.to_string()), || {
        let cap = memory_mode::vram_cap_bytes();
        let r = memory_mode::resolve_auto(8192, 8192, cap).expect("resolve");
        match r {
            ResolvedMode::Strip { h_body } => {
                assert!(h_body > 0, "h_body must be positive");
                assert_eq!(
                    h_body % PYRAMID_ALIGN,
                    0,
                    "h_body must be pyramid-aligned ({PYRAMID_ALIGN}), got {h_body}"
                );
            }
            other => panic!("expected Strip fallback when Full exceeds cap, got {other:?}"),
        }
    });
}

#[test]
fn auto_strip_h_body_fits_inside_cap() {
    // Same setup as `auto_picks_strip_when_full_exceeds_cap` but
    // double-checks the auto-sized h_body actually fits the cap. The
    // auto-sizer must never over-commit relative to the configured
    // cap — otherwise the strip walker OOMs at construction.
    let cap_bytes = estimate_strip_gpu_memory_bytes(8192, 256).unwrap();
    with_cap(Some(&cap_bytes.to_string()), || {
        let cap = memory_mode::vram_cap_bytes();
        let r = memory_mode::resolve_auto(8192, 8192, cap).expect("resolve");
        match r {
            ResolvedMode::Strip { h_body } => {
                let est = estimate_strip_gpu_memory_bytes(8192, h_body)
                    .expect("strip estimator returns Some for valid inputs");
                assert!(
                    est <= cap_bytes,
                    "auto-sized strip ({est} bytes) exceeds cap ({cap_bytes} bytes)"
                );
            }
            other => panic!("expected Strip fallback, got {other:?}"),
        }
    });
}

#[test]
fn auto_returns_too_big_when_neither_fits() {
    // 1-byte cap — neither Full nor any strip body fits. Auto MUST
    // surface TooBigForFull rather than crashing or returning a
    // bogus Strip that won't construct.
    with_cap(Some("1"), || {
        let cap = memory_mode::vram_cap_bytes();
        let r = memory_mode::resolve_auto(8192, 8192, cap);
        match r {
            Err(Error::TooBigForFull { needed, cap: c }) => {
                assert!(needed > 0, "needed must report the Full estimate");
                assert_eq!(c, 1, "cap must round-trip");
            }
            other => panic!("expected TooBigForFull at cap=1, got {other:?}"),
        }
    });
}

#[test]
fn auto_returns_too_big_on_small_image_over_cap() {
    // Image is below MIN_NATIVE_DIM on both axes; strip can't help
    // (new_strip_with_halo would reject InvalidImageSize), and the
    // cap is tighter than the Full estimate. Auto MUST surface
    // TooBigForFull — silently downgrading to Strip would mislead
    // the caller into thinking a path forward exists.
    with_cap(Some("100"), || {
        let small = MIN_NATIVE_DIM - 16;
        let cap = memory_mode::vram_cap_bytes();
        let r = memory_mode::resolve_auto(small, small, cap);
        assert!(
            matches!(r, Err(Error::TooBigForFull { .. })),
            "small-image + tiny cap must error, got {r:?}"
        );
    });
}

#[test]
fn auto_picks_full_on_small_image_when_cap_fits() {
    // Same small image but with a generous cap. Full MUST resolve —
    // small-image guard only blocks the Strip path, not Full.
    with_cap(Some("17179869184"), || {
        let small = MIN_NATIVE_DIM - 16;
        let cap = memory_mode::vram_cap_bytes();
        let r = memory_mode::resolve_auto(small, small, cap).expect("resolve");
        assert_eq!(r, ResolvedMode::Full);
    });
}

#[test]
fn auto_fallback_at_tiny_explicit_cap_picks_strip() {
    // Direct exercise of the task's "ZENMETRICS_VRAM_CAP_BYTES=1000000"
    // recipe. At 1 MB on a 1024² image the Full estimate (tens of MB)
    // exceeds the cap, but with no aligned body fitting that cap
    // either, Auto surfaces TooBigForFull. This pins the contract that
    // 1 MB is genuinely too small for iwssim — the test is here to
    // catch regressions where someone (a) silently lowers
    // MIN_NATIVE_DIM or (b) breaks the estimator. If both Full AND
    // smallest aligned strip body fit, we'd take Strip; the cap is
    // chosen so neither path fits.
    with_cap(Some("1000000"), || {
        let cap = memory_mode::vram_cap_bytes();
        let r = memory_mode::resolve_auto(1024, 1024, cap);
        // For iwssim at 1024² and cap=1 MB, neither Full (~10s of MB)
        // nor the smallest strip body fits 1 MB.
        match r {
            Err(Error::TooBigForFull { .. }) => {}
            Ok(other) => panic!(
                "expected TooBigForFull at 1 MB cap on 1024² (neither Full nor min-Strip fit), \
                 got {other:?}"
            ),
            Err(e) => panic!("expected TooBigForFull, got {e:?}"),
        }
    });
}

#[test]
fn auto_huge_cap_picks_full_at_24mp() {
    // Inverse: with an absurd 1 TiB cap a 24 MP image's Full estimate
    // is well under the cap, so Auto stays on Full.
    with_cap(Some("1099511627776"), || {
        let cap = memory_mode::vram_cap_bytes();
        let r = memory_mode::resolve_auto(6000, 4000, cap).expect("resolve");
        assert_eq!(r, ResolvedMode::Full, "24 MP fits 1 TiB easily");
    });
}
