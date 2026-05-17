//! API-level tests for the `Ssim2Blur` opt-in mode selector.
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
use ssim2_gpu::{Error, Ssim2, Ssim2Batch, Ssim2Blur};

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

#[test]
fn ssim2_fir_compute_returns_not_implemented() {
    let client = Backend::client(&Default::default());
    let mut s = Ssim2::<Backend>::new(client, 64, 64)
        .expect("Ssim2::new")
        .with_blur(Ssim2Blur::Fir);
    let buf = vec![0_u8; 64 * 64 * 3];
    let r = s.compute(&buf, &buf);
    assert!(
        matches!(r, Err(Error::FirNotYetImplemented)),
        "expected FirNotYetImplemented, got {r:?}"
    );
}

#[test]
fn ssim2_fir_set_reference_returns_not_implemented() {
    let client = Backend::client(&Default::default());
    let mut s = Ssim2::<Backend>::new(client, 64, 64)
        .expect("Ssim2::new")
        .with_blur(Ssim2Blur::Fir);
    let buf = vec![0_u8; 64 * 64 * 3];
    let r = s.set_reference(&buf);
    assert!(
        matches!(r, Err(Error::FirNotYetImplemented)),
        "expected FirNotYetImplemented, got {r:?}"
    );
}

#[test]
fn ssim2_fir_compute_with_reference_returns_not_implemented() {
    let client = Backend::client(&Default::default());
    let mut s = Ssim2::<Backend>::new(client, 64, 64).expect("Ssim2::new");
    let buf = vec![0_u8; 64 * 64 * 3];
    // Cache a reference in IIR mode first so we'd otherwise pass the
    // NoCachedReference check.
    s.set_reference(&buf).expect("iir set_reference");
    assert!(s.has_cached_reference());
    // Switching modes invalidates the cache.
    s.set_blur(Ssim2Blur::Fir);
    assert!(!s.has_cached_reference());
    let r = s.compute_with_reference(&buf);
    assert!(
        matches!(r, Err(Error::FirNotYetImplemented)),
        "expected FirNotYetImplemented, got {r:?}"
    );
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

#[test]
fn ssim2batch_fir_compute_batch_returns_not_implemented() {
    let client = Backend::client(&Default::default());
    let mut b = Ssim2Batch::<Backend>::new(client, 64, 64, 2).expect("Ssim2Batch::new");
    let buf = vec![0_u8; 64 * 64 * 3];
    b.set_reference(&buf).expect("set_reference iir");
    b.set_blur(Ssim2Blur::Fir);
    let r = b.compute_batch(&[buf.clone()]);
    assert!(
        matches!(r, Err(Error::FirNotYetImplemented)),
        "expected FirNotYetImplemented, got {r:?}"
    );
}
