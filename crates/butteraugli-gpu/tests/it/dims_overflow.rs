//! zenmetrics#30 — untrusted-dimension hardening: the fallible constructors
//! return [`Error::InvalidDimensions`] for a `width × height × 3` (or batch
//! multiple) that overflows `usize`, instead of the documented panic of the
//! infallible `new`. `u32::MAX × u32::MAX` pixels fits a 64-bit usize but
//! its `× 3` byte count does not, so the check is exercised on every target.
//! The check runs before any device allocation, so the client is untouched.

use butteraugli_gpu::{Butteraugli, ButteraugliBatch, Error, MemoryMode};
use cubecl::Runtime;

#[cfg(feature = "cuda")]
type BackendT = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type BackendT = cubecl::wgpu::WgpuRuntime;
#[cfg(all(feature = "cpu", not(feature = "cuda"), not(feature = "wgpu")))]
type BackendT = cubecl::cpu::CpuRuntime;

fn client() -> cubecl::client::ComputeClient<BackendT> {
    BackendT::client(&Default::default())
}

#[test]
fn try_new_rejects_dimension_overflow() {
    let err = Butteraugli::<BackendT>::try_new(client(), u32::MAX, u32::MAX)
        .err()
        .expect("u32::MAX² × 3 bytes overflows usize; must be Err, not a panic or an alloc");
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
fn try_new_multires_and_memory_mode_full_reject_dimension_overflow() {
    let err = Butteraugli::<BackendT>::try_new_multires(client(), u32::MAX, u32::MAX)
        .err()
        .expect("must be Err");
    assert!(matches!(err, Error::InvalidDimensions { .. }), "{err:?}");
    // The MemoryMode::Full arm routes through try_new, so the umbrella
    // constructor surfaces the same error instead of panicking.
    let err = Butteraugli::<BackendT>::new_with_memory_mode(
        client(),
        u32::MAX,
        u32::MAX,
        MemoryMode::Full,
    )
    .err()
    .expect("must be Err");
    assert!(matches!(err, Error::InvalidDimensions { .. }), "{err:?}");
}

#[test]
fn batch_try_new_rejects_dimension_overflow() {
    // Per-image overflow.
    let err = ButteraugliBatch::<BackendT>::try_new(client(), u32::MAX, u32::MAX, 1)
        .err()
        .expect("must be Err");
    assert!(matches!(err, Error::InvalidDimensions { .. }), "{err:?}");
    // Per-image fits comfortably; the batch multiple is what overflows.
    let err = ButteraugliBatch::<BackendT>::try_new(client(), 4096, 4096, usize::MAX)
        .err()
        .expect("must be Err");
    assert!(matches!(err, Error::InvalidDimensions { .. }), "{err:?}");
}

#[test]
fn try_new_accepts_ordinary_dimensions() {
    // Sanity: the check is not over-eager — a normal instance still builds
    // and reports the same dimensions the infallible path would.
    let b = Butteraugli::<BackendT>::try_new(client(), 64, 48).expect("ordinary dims build");
    assert_eq!(b.dimensions(), (64, 48));
}
