#![cfg(feature = "fir")]
//! API-level tests for the `Ssim2Blur` opt-in mode selector.
//!
//! Gated behind the `fir` Cargo feature — without it the `Ssim2Blur`
//! enum and its surfaces (`with_blur` / `set_blur` / `blur()` /
//! `SSIM2_FIR_COLUMN_NAME` / `column_name_for_blur`) are not exported,
//! so this whole file compiles to nothing in the default build.
//!
//! These tests pin the contract for commit 1 of T_y.B.2:
//!
//! - `Ssim2Blur::default() == Ssim2Blur::Iir` (existing behaviour is
//!   the unchanged default).
//! - `Ssim2::new` instances start in `Iir` mode.
//! - `with_blur` / `set_blur` round-trip.
//! - Switching to `Ssim2Blur::Fir` and calling any compute method
//!   returns `Error::FirNotYetImplemented` (skeleton; commit 3 lands
//!   the kernel).
//! - Switching blur modes invalidates the cached reference.
//!
//! `parity_lock.rs` exercises the **default (IIR) path** end-to-end and
//! is the load-bearing parity gate. This file is only about the new
//! opt-in API surface — no score values are asserted here.

use cubecl::Runtime;
use ssim2_gpu::{
    Error, SSIM2_FIR_COLUMN_NAME, SSIM2_IIR_COLUMN_NAME, Ssim2, Ssim2Batch, Ssim2Blur,
    column_name_for_blur,
};

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;
#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
compile_error!(
    "ssim2-gpu integration tests require either the `cuda` or `wgpu` feature to select a runtime"
);

#[test]
fn default_blur_is_iir() {
    assert_eq!(Ssim2Blur::default(), Ssim2Blur::Iir);
}

#[test]
fn ssim2_new_starts_in_iir_mode() {
    let client = Backend::client(&Default::default());
    let s = Ssim2::<Backend>::new(client, 64, 64).expect("Ssim2::new");
    assert_eq!(s.blur(), Ssim2Blur::Iir);
}

#[test]
fn ssim2_with_blur_round_trips() {
    let client = Backend::client(&Default::default());
    let s = Ssim2::<Backend>::new(client, 64, 64)
        .expect("Ssim2::new")
        .with_blur(Ssim2Blur::Fir);
    assert_eq!(s.blur(), Ssim2Blur::Fir);
}

#[test]
fn ssim2_set_blur_round_trips() {
    let client = Backend::client(&Default::default());
    let mut s = Ssim2::<Backend>::new(client, 64, 64).expect("Ssim2::new");
    assert_eq!(s.blur(), Ssim2Blur::Iir);
    s.set_blur(Ssim2Blur::Fir);
    assert_eq!(s.blur(), Ssim2Blur::Fir);
    s.set_blur(Ssim2Blur::Iir);
    assert_eq!(s.blur(), Ssim2Blur::Iir);
}

/// Pre-commit-3 behaviour was `Err(Error::FirNotYetImplemented)`;
/// from commit 3 onward the FIR kernel exists and `compute` returns
/// `Ok`. The score value is asserted by the dedicated FIR test file
/// (`fir_path.rs`); here we just pin that the API call succeeds.
#[test]
fn ssim2_fir_compute_returns_ok() {
    let client = Backend::client(&Default::default());
    let mut s = Ssim2::<Backend>::new(client, 64, 64)
        .expect("Ssim2::new")
        .with_blur(Ssim2Blur::Fir);
    let buf = vec![0_u8; 64 * 64 * 3];
    let r = s.compute(&buf, &buf).expect("FIR compute should succeed");
    assert!(r.score.is_finite(), "FIR score must be finite, got {}", r.score);
}

#[test]
fn ssim2_fir_set_reference_returns_ok() {
    let client = Backend::client(&Default::default());
    let mut s = Ssim2::<Backend>::new(client, 64, 64)
        .expect("Ssim2::new")
        .with_blur(Ssim2Blur::Fir);
    let buf = vec![0_u8; 64 * 64 * 3];
    s.set_reference(&buf).expect("FIR set_reference should succeed");
    assert!(s.has_cached_reference());
}

#[test]
fn ssim2_fir_compute_with_reference_returns_ok() {
    let client = Backend::client(&Default::default());
    let mut s = Ssim2::<Backend>::new(client, 64, 64).expect("Ssim2::new");
    let buf = vec![0_u8; 64 * 64 * 3];
    // Cache a reference in IIR mode first so we observe the
    // mode-switch cache invalidation.
    s.set_reference(&buf).expect("iir set_reference");
    assert!(s.has_cached_reference());
    s.set_blur(Ssim2Blur::Fir);
    assert!(!s.has_cached_reference(), "switching modes must invalidate");
    // Re-arm under FIR.
    s.set_reference(&buf).expect("fir set_reference");
    let r = s
        .compute_with_reference(&buf)
        .expect("FIR compute_with_reference should succeed");
    assert!(r.score.is_finite());
}

#[test]
fn ssim2_switching_blur_invalidates_cache() {
    let client = Backend::client(&Default::default());
    let mut s = Ssim2::<Backend>::new(client, 64, 64).expect("Ssim2::new");
    let buf = vec![0_u8; 64 * 64 * 3];
    s.set_reference(&buf).expect("set_reference iir");
    assert!(s.has_cached_reference());
    // Same-mode set is a no-op for the cache.
    s.set_blur(Ssim2Blur::Iir);
    assert!(s.has_cached_reference());
    // Switching to a different mode invalidates.
    s.set_blur(Ssim2Blur::Fir);
    assert!(!s.has_cached_reference());
}

#[test]
fn ssim2batch_new_starts_in_iir_mode() {
    let client = Backend::client(&Default::default());
    let b = Ssim2Batch::<Backend>::new(client, 64, 64, 2).expect("Ssim2Batch::new");
    assert_eq!(b.blur(), Ssim2Blur::Iir);
}

#[test]
fn ssim2batch_with_blur_round_trips() {
    let client = Backend::client(&Default::default());
    let b = Ssim2Batch::<Backend>::new(client, 64, 64, 2)
        .expect("Ssim2Batch::new")
        .with_blur(Ssim2Blur::Fir);
    assert_eq!(b.blur(), Ssim2Blur::Fir);
}

// ───────────────── impl-tag column names ─────────────────

#[test]
fn column_names_are_distinct() {
    assert_ne!(SSIM2_IIR_COLUMN_NAME, SSIM2_FIR_COLUMN_NAME);
}

#[test]
fn column_names_match_blur_helper() {
    assert_eq!(column_name_for_blur(Ssim2Blur::Iir), SSIM2_IIR_COLUMN_NAME);
    assert_eq!(column_name_for_blur(Ssim2Blur::Fir), SSIM2_FIR_COLUMN_NAME);
}

#[test]
fn column_names_have_correct_prefixes() {
    // Default form is `ssim2_imazen_<blur>_v<MAJOR>_<MINOR>_<PATCH>`.
    // The env-var overrides can produce arbitrary strings, but in the
    // default build (which is what every consumer sees) the prefix is
    // load-bearing for parquet column-name discovery and for the
    // distinction between IIR and FIR.
    assert!(
        SSIM2_IIR_COLUMN_NAME.starts_with("ssim2_imazen_iir_v")
            || std::env::var("SSIM2_IIR_IMPL_TAG").is_ok(),
        "unexpected IIR column name: {SSIM2_IIR_COLUMN_NAME}"
    );
    assert!(
        SSIM2_FIR_COLUMN_NAME.starts_with("ssim2_imazen_fir_v")
            || std::env::var("SSIM2_FIR_IMPL_TAG").is_ok(),
        "unexpected FIR column name: {SSIM2_FIR_COLUMN_NAME}"
    );
}

#[test]
fn ssim2batch_fir_compute_batch_returns_ok() {
    let client = Backend::client(&Default::default());
    let mut b = Ssim2Batch::<Backend>::new(client, 64, 64, 2).expect("Ssim2Batch::new");
    let buf = vec![0_u8; 64 * 64 * 3];
    b.set_reference(&buf).expect("set_reference iir");
    b.set_blur(Ssim2Blur::Fir);
    // Mode switch invalidated cache → expect NoCachedReference.
    let r = b.compute_batch(&[buf.clone()]);
    assert!(
        matches!(r, Err(Error::NoCachedReference)),
        "expected NoCachedReference after mode switch, got {r:?}"
    );
    // Re-arm under FIR.
    b.set_reference(&buf).expect("set_reference fir");
    let r = b
        .compute_batch(&[buf.clone()])
        .expect("FIR compute_batch should succeed");
    assert_eq!(r.len(), 1);
    assert!(r[0].score.is_finite());
}
