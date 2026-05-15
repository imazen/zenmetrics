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
//! 3. **Pyramid**: per-channel **Weber-contrast** pyramid
//!    (`contrast="weber_g1"`) — non-baseband bands are
//!    `clip(layer / max(L_bkg, 0.01), ±1000)` with `L_bkg` taken from
//!    the per-pixel expanded achromatic gauss. Baseband bypasses
//!    Weber and feeds directly into pooling. ~7 levels for a
//!    1024-wide image.
//! 4. **CSF**: per-pixel LUT lookup of castleCSF
//!    `weber_fixed_size` — bilinear interp over
//!    `(log_rho, log_L_bkg)` for the three `omega = 0` channels
//!    (achromatic A, red-green RG, violet-yellow VY), then `T_p =
//!    weber × S × CH_GAIN`.
//! 5. **Masking**: cvvdp's `mult-mutual` with cross-channel pooling
//!    (`MASK_P / MASK_Q / MASK_C / D_MAX` + the `XCM_3X3` matrix).
//!    Bands smaller than `pu_padsize = 6` skip the σ = 3 PU blur;
//!    larger bands run separable 13-tap Gaussian blur first.
//! 6. **Pooling**: 3-stage Minkowski fold per `(band, channel)` →
//!    per-channel → overall `D`.
//! 7. **JOD**: piecewise [`kernels::pool::met2jod`] — two
//!    `jod_a/b/c` regimes joined continuously at `Q = 0.1`.
//!
//! ## Status
//!
//! Still-image score matches pycvvdp v0.5.4 within ~0.006 JOD across
//! q1–q90 fixtures on the v1 R2 manifest.
//!
//! The full GPU composition path is wired through
//! [`Cvvdp::compute_dkl_jod`]: color, Weber pyramid, CSF, masking,
//! and spatial pool all run on GPU; only the 3-stage Minkowski fold
//! and the `met2jod` mapping happen host-side, on a ~144-byte
//! partials Vec. The parity tests
//! `compute_dkl_jod_matches_host_scalar`,
//! `compute_dkl_jod_on_v1_manifest_corpus`, and
//! `compute_dkl_jod_vs_host_scalar_on_corpus` all lock the GPU path
//! within f32-precision tolerance of the host scalar reference.
//!
//! The public [`Cvvdp::score`] API still routes through
//! [`host_scalar::predict_jod_still_3ch`] (kept stable while the GPU
//! path's manifest-level parity is held by `shadow_jod`). Switching
//! `score` over to the GPU path is the remaining chunk of pipeline
//! work.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
// cvvdp parameters + the per-(rho, L_bkg, channel) CSF LUT are imported
// verbatim from pycvvdp v0.5.4 source. The literals carry more digits
// than f32 can represent so the values document the source even though
// LLVM rounds at compile time.
#![allow(clippy::excessive_precision)]

pub mod host_scalar;
pub mod kernels;
pub mod params;
pub mod pipeline;

pub use params::CvvdpParams;
pub use pipeline::Cvvdp;

/// Number of color channels in DKL opponent space (achromatic +
/// red-green + violet-yellow).
pub const N_CHANNELS: usize = 3;

/// Maximum pyramid depth supported by the kernel allocations.
/// `pipeline::pyramid_levels` caps the per-image pyramid depth at
/// this value, so images with `min(w, h) > PYRAMID_MIN_DIM ×
/// 2^MAX_LEVELS` (≈ 1024 with the defaults) get only `MAX_LEVELS`
/// bands — coarser frequency content above the cap is folded into
/// the baseband.
pub const MAX_LEVELS: usize = 8;

/// Smallest logical width/height at which the pyramid keeps
/// building further coarse levels. Once `min(w, h) < 2 ×
/// PYRAMID_MIN_DIM`, the current level becomes the baseband.
pub const PYRAMID_MIN_DIM: u32 = 4;

#[derive(Debug, Clone)]
pub enum Error {
    /// Buffer length doesn't match `width × height × 3`.
    DimensionMismatch { expected: usize, got: usize },
    /// `Cvvdp::score_with_reference` was called without a prior
    /// `Cvvdp::set_reference`.
    NoCachedReference,
    /// Image is too small for the configured pyramid, **or** a GPU
    /// read-back / dispatch failed. The two get the same variant
    /// because cubecl's read errors aren't easily separable yet —
    /// callers in tests / production should treat this as "GPU
    /// pipeline failed, retry or surface to user".
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
