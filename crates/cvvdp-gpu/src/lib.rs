//! Multi-vendor GPU implementation of ColorVideoVDP (still-image mode),
//! built on [CubeCL](https://github.com/tracel-ai/cubecl) so the same
//! `#[cube]` kernel source dispatches across:
//!
//! - **CUDA** (NVIDIA) via the cubecl CUDA runtime
//! - **WGPU** (cross-platform) via Vulkan/Metal/DX12/WebGPU
//! - **HIP** (AMD ROCm) when the `hip` feature is enabled
//! - **CPU** via the cubecl CPU runtime — library compiles, but
//!   `Cvvdp::compute_dkl_jod` won't run there because
//!   `pool_band_kernel` uses `Atomic<f32>::fetch_add`, which
//!   cubecl-cpu lacks. A per-block partial-tree reduction port
//!   would unblock the cpu backend; see
//!   `docs/PORT_STATUS.md` for context.
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
//! For batch scoring (one reference vs many distorted candidates),
//! [`Cvvdp::warm_reference`] + [`Cvvdp::compute_dkl_jod_with_warm_ref`]
//! caches the REF GPU state and skips that half of the pipeline per
//! candidate — 1.75× faster per DIST at 12 MP. Parity vs the cold
//! path is locked at ≤ 1e-5 JOD by
//! `compute_dkl_jod_with_warm_ref_matches_unwarm_path`.
//!
//! ## How we compare to the canonical reference
//!
//! On an RTX 5070 at 12 MP the **canonical** pycvvdp v0.5.4 CUDA path
//! lands at ~14 ns/px steady-state (after a 1–13 s PyTorch graph
//! compile). With ceil-div correctness in place (tick 175), our cold
//! path runs ~62 ns/px and our warm-ref path ~34 ns/px — **4.4× /
//! 2.4× slower than pycvvdp**. The pre-tick-175 numbers (36 / 21
//! ns/px) reflected a broken pyramid that drifted 0.586 JOD vs
//! pycvvdp; the current numbers come from correct output (≤ 0.0003
//! JOD drift, gated by `compute_dkl_jod_matches_pycvvdp_at_12mp_synth`).
//!
//! pycvvdp benefits from cuDNN-optimised separable convolutions on
//! the downscale/upscale pyramid; our hand-written cubecl kernels
//! don't reach that level of optimisation yet. The ~25% post-fix
//! slowdown vs pre-fix is open investigation — total pixel work
//! barely changed between floor-div and ceil-div pyramids.
//!
//! Where we win: multi-vendor backends (WGPU + HIP work; pycvvdp is
//! CUDA-only via PyTorch), static-binary deployment (~50 MB vs ~3 GB
//! PyTorch runtime), and ~1 s warm-up. See
//! `benchmarks/pycvvdp_12mp_cuda_2026-05-14.md` for the original
//! head-to-head and `benchmarks/pycvvdp_parity_tick175_2026-05-15.md`
//! for the post-ceil-div correctness + perf numbers.
//!
//! The public [`Cvvdp::score`] API still routes through
//! [`host_scalar::predict_jod_still_3ch`] (kept stable while the GPU
//! path's manifest-level parity is held by `shadow_jod`). Switching
//! `score` over to the GPU path is the remaining chunk of pipeline
//! work.

#![allow(clippy::needless_range_loop)]
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
pub const MAX_LEVELS: usize = 9;

/// Smallest logical width/height at which the pyramid keeps
/// building further coarse levels. Once `min(w, h) < 2 ×
/// PYRAMID_MIN_DIM`, the current level becomes the baseband.
pub const PYRAMID_MIN_DIM: u32 = 4;

/// Failure modes for `Cvvdp::*` methods. Implements
/// `std::error::Error` so callers can use `?` against
/// `Box<dyn Error>` or `anyhow::Error` as usual.
#[derive(Debug, Clone)]
pub enum Error {
    /// Buffer length doesn't match `width × height × 3`.
    DimensionMismatch { expected: usize, got: usize },
    /// `Cvvdp::score_with_reference` was called without a prior
    /// `Cvvdp::set_reference`.
    NoCachedReference,
    /// `Cvvdp::compute_dkl_jod_with_warm_ref` was called without a
    /// prior `Cvvdp::warm_reference`, **or** the warm state was
    /// invalidated by an intervening call to a method that
    /// dispatches the REF weber pyramid (e.g. `compute_dkl_jod`,
    /// `compute_dkl_d_bands`).
    NoWarmReference,
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
            Error::NoWarmReference => write!(
                f,
                "no warm GPU reference; call warm_reference first (or warm state was invalidated by an intervening REF dispatch)"
            ),
            Error::InvalidImageSize => write!(f, "image is too small for the configured pyramid"),
        }
    }
}

impl std::error::Error for Error {}

/// `Result<T, cvvdp_gpu::Error>` — the crate's standard fallible
/// return type. Every `Cvvdp::*` constructor and dispatch method
/// returns this.
pub type Result<T> = std::result::Result<T, Error>;
