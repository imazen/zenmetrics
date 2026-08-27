//! zenmetrics#30 — untrusted-dimension hardening: `Ssim2::new` and
//! `Ssim2Batch::new` return [`Error::InvalidDimensions`] when the packed-u32
//! upload byte count (`width × height × 4`) or its batch multiple overflows
//! `usize`, before any device allocation. `u32::MAX × u32::MAX` pixels fits
//! a 64-bit usize but its `× 4` byte count does not, so the check is
//! exercised on every target (on 32-bit the pixel product itself wraps —
//! previously a silent under-allocation).

use cubecl::Runtime;
use ssim2_gpu::{Error, Ssim2, Ssim2Batch};

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;
#[cfg(all(feature = "cpu", not(feature = "cuda"), not(feature = "wgpu")))]
type Backend = cubecl::cpu::CpuRuntime;

fn client() -> cubecl::client::ComputeClient<Backend> {
    Backend::client(&Default::default())
}

#[test]
fn new_rejects_dimension_overflow() {
    let err = Ssim2::<Backend>::new(client(), u32::MAX, u32::MAX)
        .err()
        .expect("u32::MAX² × 4 bytes overflows usize; must be Err, not a panic or an alloc");
    assert!(
        matches!(
            err,
            Error::InvalidDimensions {
                width: u32::MAX,
                height: u32::MAX
            }
        ),
        "expected InvalidDimensions, got {err:?}"
    );
    assert!(err.to_string().contains("overflows usize"), "{err}");
}

#[test]
fn batch_new_rejects_dimension_overflow() {
    // Per-image overflow — caught before the inner pipeline allocates.
    let err = Ssim2Batch::<Backend>::new(client(), u32::MAX, u32::MAX, 1)
        .err()
        .expect("must be Err");
    assert!(matches!(err, Error::InvalidDimensions { .. }), "{err:?}");
    // Per-image fits (2^32 px, 2^34 bytes); 2^32 × (2^32-1) images fits usize
    // but its × 4 byte count does not — the batch multiple is what overflows.
    let err = Ssim2Batch::<Backend>::new(client(), 65536, 65536, u32::MAX)
        .err()
        .expect("must be Err");
    assert!(matches!(err, Error::InvalidDimensions { .. }), "{err:?}");
}

#[test]
fn new_accepts_ordinary_dimensions() {
    // Sanity: the check is not over-eager.
    let s = Ssim2::<Backend>::new(client(), 64, 48).expect("ordinary dims build");
    assert_eq!(s.dimensions(), (64, 48));
}
