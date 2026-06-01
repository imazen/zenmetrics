//! Multi-vendor GPU implementation of the butteraugli perceptual image quality metric.
//!
//! Built on [CubeCL](https://github.com/tracel-ai/cubecl). Single Rust kernel
//! source, dispatchable across:
//! - **CUDA** (NVIDIA) — native PTX/SASS via CubeCL CUDA runtime
//! - **WGPU** (cross-platform) — Vulkan/Metal/DX12/WebGPU via wgpu
//! - **HIP** (AMD ROCm) — when the `hip` feature is enabled
//! - **CPU** (SIMD) — when the `cpu` feature is enabled
//!
//! The CPU backend is intended only as a correctness reference; it's not
//! competitive with the dedicated [`butteraugli`](https://crates.io/crates/butteraugli)
//! crate's autoversioned SIMD path.
//!
//! ## Algorithmic parity with `butteraugli` v0.9.2
//!
//! Aggregations match the CPU crate exactly: `score` is the max-norm
//! distance, `pnorm_3` is the libjxl 3-norm aggregation
//! (`butteraugli_main --pnorm` default). Both are produced in a single
//! fused on-device reduction pass over the diffmap.
//!
//! ## Status
//!
//! Early port from `butteraugli-cuda`. The reduction is the first kernel
//! ported end-to-end; full pipeline (opsin / blur / Malta / masking / diffmap
//! combination) is in progress. See `PORT_STATUS.md`.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

// `kernels` is reached by-path from this crate's own GPU parity
// examples (blur/colors harnesses; they need a GPU runtime and are
// built via `--all-targets`, not run as CI unit tests). `#[doc(hidden)]`:
// reachable for them, not a supported per-crate API.
#[doc(hidden)]
pub mod kernels;
pub mod memory_mode;
pub(crate) mod opaque;
// `pipeline` is reached by-path cross-crate (zen-metrics-cli's
// orchestrator_runner) — `#[doc(hidden)]`, like `session`.
#[doc(hidden)]
pub mod pipeline;
pub(crate) mod pipeline_batch;
// Stream-bound session plumbing for `zenmetrics_api::MetricSession`
// (issue #17). `#[doc(hidden)]`, gated `cubecl-types`. Not a supported
// per-crate API.
#[cfg(feature = "cubecl-types")]
#[doc(hidden)]
pub mod session;
#[cfg(feature = "cubecl-types")]
pub(crate) mod strip;

// Uniform opaque API (Phase 2 of API uniformity refactor). See
// `opaque.rs` and the matching shim in `dssim-gpu`.
pub use memory_mode::{
    MemoryMode, ResolvedMode, estimate_gpu_memory_bytes, estimate_strip_gpu_memory_bytes,
    vram_cap_bytes,
};
pub use opaque::{Backend, ButteraugliOpaque, Score};

// Typed-generic API (gated behind `cubecl-types`). Internal callers
// can still reach the types via `butteraugli_gpu::pipeline::Butteraugli`.
#[cfg(feature = "cubecl-types")]
pub use pipeline::Butteraugli;
#[cfg(feature = "cubecl-types")]
pub use pipeline_batch::ButteraugliBatch;

#[cfg(feature = "cubecl-types")]
use cubecl::prelude::*;

/// Result of a butteraugli comparison.
///
/// Mirrors `butteraugli::ButteraugliResult` from the CPU crate. `score` is
/// the max-norm; `pnorm_3` is the libjxl 3-norm aggregation, available
/// "for free" because the fused reduction kernel produces both in one pass.
#[derive(Debug, Clone, Copy)]
pub struct GpuButteraugliResult {
    /// Max-norm difference score. < 1.0 is "good", > 2.0 is "bad".
    pub score: f32,
    /// libjxl 3-norm aggregation — average of three p-norms at exponents
    /// 3, 6, 12. Matches `butteraugli_main --pnorm` and the CPU crate's
    /// `ButteraugliResult.pnorm_3`.
    pub pnorm_3: f32,
}

/// Tunable parameters for a butteraugli comparison. Mirrors
/// `butteraugli::ButteraugliParams` and `butteraugli_cuda`'s
/// `compute_with_options` arguments. Use [`ButteraugliParams::default`]
/// for the standard 80-nit display, symmetric, full-color comparison.
#[derive(Debug, Clone, Copy)]
pub struct ButteraugliParams {
    /// Asymmetry between the two error directions. 1.0 = symmetric;
    /// > 1.0 penalises distorted < reference (artifact penalty
    /// > stronger than blur penalty); < 1.0 penalises distorted >
    /// > reference more.
    pub hf_asymmetry: f32,
    /// Display intensity multiplier in nits. Default 80.0 for an
    /// 80-nit SDR display; HDR encoders typically set this to 250+
    /// to match their target display.
    pub intensity_target: f32,
    /// Per-channel weight on the X (chroma) component. Default 1.0;
    /// 0.5 halves chroma penalty (useful for chroma subsampling
    /// rate-distortion).
    pub xmul: f32,
}

impl Default for ButteraugliParams {
    fn default() -> Self {
        Self {
            hf_asymmetry: 1.0,
            intensity_target: 80.0,
            xmul: 1.0,
        }
    }
}

impl ButteraugliParams {
    pub fn with_intensity_target(mut self, intensity_target: f32) -> Self {
        self.intensity_target = intensity_target;
        self
    }
    pub fn with_hf_asymmetry(mut self, hf_asymmetry: f32) -> Self {
        self.hf_asymmetry = hf_asymmetry;
        self
    }
    pub fn with_xmul(mut self, xmul: f32) -> Self {
        self.xmul = xmul;
        self
    }
}

/// Errors that the GPU butteraugli pipeline can return. All other
/// failures (kernel-launch errors, OOM) currently panic — they
/// indicate runtime/driver problems rather than user input issues.
#[derive(Debug, Clone)]
pub enum Error {
    /// `compute*` was called with a buffer length that doesn't match
    /// the configured `width × height × 3` of the instance.
    DimensionMismatch { expected: usize, got: usize },
    /// `compute_with_reference*` was called without a prior
    /// `set_reference`.
    NoCachedReference,
    /// `ButteraugliParams` had a non-finite or non-positive value.
    InvalidParams(&'static str),
    /// A whole-image-only API (`set_reference`, `new_multires` sibling,
    /// etc.) was called on a strip-mode instance constructed via
    /// `new_strip`. Strip mode currently runs single-resolution
    /// pair-only — re-allocate via `new` if you need the cached-
    /// reference / multi-resolution paths.
    StripModeUnsupported(&'static str),
    /// The requested [`MemoryMode`](crate::MemoryMode) variant isn't
    /// implemented yet in this crate (e.g. `Tile {...}`). The string
    /// names the unsupported variant for diagnostics.
    ModeUnsupported(&'static str),
    /// [`MemoryMode::Auto`](crate::MemoryMode) couldn't satisfy the
    /// caller's image size — the Full allocation exceeds the VRAM cap
    /// AND the Strip walker can't fit either. Surface the gap so the
    /// caller can either raise `ZENMETRICS_VRAM_CAP_BYTES`, drop to a
    /// smaller image, or pick [`MemoryMode::Strip`](crate::MemoryMode)
    /// with an explicit `h_body`.
    TooBigForFull { needed: usize, cap: usize },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::DimensionMismatch { expected, got } => write!(
                f,
                "dimension mismatch: expected {expected} bytes, got {got}"
            ),
            Error::NoCachedReference => write!(f, "no cached reference; call set_reference first"),
            Error::InvalidParams(msg) => write!(f, "invalid params: {msg}"),
            Error::StripModeUnsupported(api) => write!(
                f,
                "strip-mode instance does not support `{api}` (single-resolution pair-only); \
                use `Butteraugli::new` for whole-image / cached-reference / multi-resolution paths"
            ),
            Error::ModeUnsupported(variant) => write!(
                f,
                "MemoryMode::{variant} is not yet implemented in butteraugli-gpu"
            ),
            Error::TooBigForFull { needed, cap } => write!(
                f,
                "Auto could not place image in {cap} byte cap; \
                 estimated whole-image working set is {needed} bytes \
                 (set ZENMETRICS_VRAM_CAP_BYTES or pass MemoryMode::Strip explicitly)"
            ),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// Aggregate a diffmap into (score, pnorm_3) on the GPU using a single
/// fused reduction pass — runs on whatever CubeCL runtime `R` you pick.
///
/// This is the smallest end-to-end CubeCL kernel in the crate; it serves
/// as both the score-extraction step of the full butteraugli pipeline
/// (when the rest is ported) and as a self-contained validation target.
///
/// Diffmap values must be non-negative finite f32 (the butteraugli pipeline
/// guarantees this — diffmap is `sqrt` of sums of squares).
///
/// Gated behind the `cubecl-types` feature — the cubecl `Runtime` /
/// `ComputeClient` / `Handle` types in this signature pin the caller
/// to a specific cubecl version, which the opaque API hides.
#[cfg(feature = "cubecl-types")]
pub fn reduce_diffmap_to_score<R: Runtime>(
    client: &ComputeClient<R>,
    diffmap_handle: cubecl::server::Handle,
    n_pixels: usize,
) -> GpuButteraugliResult {
    kernels::reduction::reduce(client, diffmap_handle, n_pixels)
}
