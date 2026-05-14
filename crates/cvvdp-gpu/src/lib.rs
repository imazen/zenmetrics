//! Multi-vendor GPU implementation of ColorVideoVDP (still-image mode),
//! built on [CubeCL](https://github.com/tracel-ai/cubecl) so the same
//! `#[cube]` kernel source dispatches across:
//!
//! - **CUDA** (NVIDIA) via the cubecl CUDA runtime
//! - **WGPU** (cross-platform) via Vulkan/Metal/DX12/WebGPU
//! - **HIP** (AMD ROCm) when the `hip` feature is enabled
//! - **CPU** via the cubecl CPU runtime (build-only until parity work
//!   finalizes which atomics we depend on)
//!
//! ## Scope: still images, JOD score
//!
//! Targets bit-stable parity with the published `ColorVideoVDP` Python
//! reference (gfxdisp/ColorVideoVDP **v0.5.4**) for the **still-image**
//! code path. Video / temporal channels (sustained + transient) are
//! intentionally out of scope for v0; defer until still-mode parity is
//! locked.
//!
//! ## Algorithm shape
//!
//! Per (reference, distorted) sRGB-u8 pair:
//!
//! 1. **Display model**: sRGB byte → linear → display-emitted luminance
//!    (gamma + peak luminance + ambient).
//! 2. **Color transform**: linear RGB → DKL opponent space
//!    `(A, RG, VY)`.
//! 3. **Pyramid**: per-channel Laplacian (still mode uses Laplacian,
//!    not steerable). ~7 levels for a 1024-wide image.
//! 4. **CSF**: per-band contrast sensitivity weighting (castleCSF for
//!    achromatic, separate chrom CSF for RG/VY).
//! 5. **Masking**: within-channel + cross-channel contrast masking with
//!    cvvdp's power-law model.
//! 6. **Pooling**: Minkowski-norm per band → per channel → overall
//!    distortion `D`.
//! 7. **JOD**: `JOD = jod_a - jod_b * D^jod_c`.
//!
//! ## Status
//!
//! Scaffolding — kernel bodies are stubs. Goldens against the Python
//! reference land per stage. See `docs/PORT_STATUS.md` once written.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

pub mod host_scalar;
pub mod kernels;
pub mod params;
pub mod pipeline;

pub use params::CvvdpParams;
pub use pipeline::Cvvdp;

/// Number of color channels in DKL opponent space (achromatic +
/// red-green + violet-yellow).
pub const N_CHANNELS: usize = 3;

/// Maximum pyramid depth supported by the kernel allocations. Image
/// sizes larger than `2^MAX_LEVELS × base_min` use only the lower
/// `MAX_LEVELS` bands.
pub const MAX_LEVELS: usize = 8;

/// Smallest logical width/height at which the pyramid keeps building
/// further coarse levels. Below this, the band is the coarse residual.
pub const PYRAMID_MIN_DIM: u32 = 4;

#[derive(Debug, Clone)]
pub enum Error {
    /// Buffer length doesn't match `width × height × 3`.
    DimensionMismatch { expected: usize, got: usize },
    /// `compute_with_reference` was called without a prior `set_reference`.
    NoCachedReference,
    /// Image is too small for the configured pyramid.
    InvalidImageSize,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::DimensionMismatch { expected, got } => write!(
                f,
                "dimension mismatch: expected {expected} bytes, got {got}"
            ),
            Error::NoCachedReference => write!(f, "no cached reference; call set_reference first"),
            Error::InvalidImageSize => write!(f, "image is too small for the configured pyramid"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
