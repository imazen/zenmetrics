//! Multi-vendor GPU implementation of the SSIMULACRA2 perceptual image
//! quality metric.
//!
//! Built on [CubeCL](https://github.com/tracel-ai/cubecl) — single Rust
//! kernel source, dispatchable across:
//! - **CUDA** (NVIDIA) via the cubecl CUDA runtime
//! - **WGPU** (cross-platform) via Vulkan/Metal/DX12/WebGPU
//! - **HIP** (AMD ROCm) when the `hip` feature is enabled
//! - **CPU** (SIMD) reference path when `cpu` is enabled
//!
//! Algorithmic parity target is the published `ssimulacra2` v0.5.1 crate
//! (the canonical Rust port of `cloudinary/ssimulacra2`). At the
//! resolution-pyramid level the implementation also matches
//! `crates/ssimulacra2-cuda/`, which uses the same Charalampidis
//! recursive Gaussian and the same 6-octave reduction.
//!
//! ## Single-image usage
//!
//! ```no_run
//! use cubecl::Runtime;
//! use cubecl::wgpu::WgpuRuntime;
//! use ssim2_gpu::Ssim2;
//!
//! let client = WgpuRuntime::client(&Default::default());
//! let mut s = Ssim2::<WgpuRuntime>::new(client, 256, 256)?;
//!
//! let ref_srgb: Vec<u8> = vec![0; 256 * 256 * 3];
//! let dist_srgb: Vec<u8> = vec![0; 256 * 256 * 3];
//!
//! let result = s.compute(&ref_srgb, &dist_srgb)?;
//! println!("score = {:.3}", result.score);
//! # Ok::<(), ssim2_gpu::Error>(())
//! ```
//!
//! ## Cached-reference usage (encoder rate-distortion)
//!
//! ```no_run
//! use cubecl::Runtime;
//! use cubecl::wgpu::WgpuRuntime;
//! use ssim2_gpu::Ssim2;
//!
//! # fn candidates() -> Vec<Vec<u8>> { vec![] }
//! let client = WgpuRuntime::client(&Default::default());
//! let mut s = Ssim2::<WgpuRuntime>::new(client, 256, 256)?;
//!
//! let ref_srgb: Vec<u8> = vec![0; 256 * 256 * 3];
//! s.set_reference(&ref_srgb)?;
//!
//! for distorted_candidate in candidates() {
//!     let r = s.compute_with_reference(&distorted_candidate)?;
//!     // ... use r.score in the rate-distortion search ...
//! }
//! # Ok::<(), ssim2_gpu::Error>(())
//! ```
//!
//! ## Batched usage (N images vs one cached reference)
//!
//! ```no_run
//! use cubecl::Runtime;
//! use cubecl::wgpu::WgpuRuntime;
//! use ssim2_gpu::Ssim2Batch;
//!
//! # fn collect_distorted() -> Vec<Vec<u8>> { vec![] }
//! let client = WgpuRuntime::client(&Default::default());
//! let mut batch = Ssim2Batch::<WgpuRuntime>::new(client, 256, 256, 8)?;
//!
//! let ref_srgb: Vec<u8> = vec![0; 256 * 256 * 3];
//! batch.set_reference(&ref_srgb)?;
//!
//! let dis_images: Vec<Vec<u8>> = collect_distorted();
//! let results = batch.compute_batch(&dis_images)?;
//! for r in &results {
//!     println!("score = {:.3}", r.score);
//! }
//! # Ok::<(), ssim2_gpu::Error>(())
//! ```
//!
//! ## Score interpretation
//!
//! Output is in roughly the 0–100 range:
//! - **100** = identical (or near-identical)
//! - **90+** = visually indistinguishable for most observers
//! - **70+** = high quality
//! - **30–60** = noticeable distortion
//! - **<0** = the SSIMULACRA2 polynomial overshoot region for severely
//!   distorted images; the CPU `ssimulacra2` produces the same
//!   negative values there — not a GPU-side bug.
//!
//! ## Status
//!
//! Initial port from `ssimulacra2-cuda`. See `PORT_STATUS.md` and
//! `HANDOFF.md`. Validated against CPU `ssimulacra2` v0.5.1 to
//! ≤ 0.06 % relative on JPEG q5..q90; cached and batched paths agree
//! with the direct path to ≤ 1.3e-5 absolute.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

pub mod kernels;
pub mod memory_mode;
pub mod opaque;
pub mod pipeline;
pub mod pipeline_batch;
// Stream-bound session plumbing for `zenmetrics_api::MetricSession`
// (issue #17). `#[doc(hidden)]`, gated `cubecl-types`. Not a supported
// per-crate API.
#[cfg(feature = "cubecl-types")]
#[doc(hidden)]
pub mod session;
pub mod skipmap;

pub use memory_mode::{
    MemoryMode, ResolvedMode, STRIP_H_BODY_DEFAULT, STRIP_HALO_ROWS, estimate_gpu_memory_bytes,
    estimate_strip_gpu_memory_bytes, vram_cap_bytes,
};
// Uniform opaque API (Phase 2). See `opaque.rs`.
pub use opaque::{Backend, Score, Ssim2Opaque, Ssim2Params};
// Ssim2Mode is part of the opaque public params, so re-export it
// unconditionally.
pub use skipmap::Ssim2Mode;

// Typed-generic API (gated behind `cubecl-types`).
#[cfg(feature = "cubecl-types")]
pub use pipeline::Ssim2;
#[cfg(feature = "cubecl-types")]
pub use pipeline_batch::Ssim2Batch;

#[cfg(feature = "fir")]
pub use blur_mode::Ssim2Blur;

/// Number of pyramid scales — matches both the CPU and CUDA references.
pub const NUM_SCALES: usize = 6;

#[cfg(feature = "fir")]
mod blur_mode {

/// Blur-kernel implementation selector — **gated behind the `fir`
/// Cargo feature**.
///
/// SSIMULACRA2's per-scale Gaussian blur (σ = 1.5) admits multiple
/// algorithmic realisations that produce different per-pixel responses.
/// This enum picks which one a given `Ssim2` / `Ssim2Batch` instance
/// uses. The choice is INVISIBLE to the score's interpretation only if
/// the chosen kernel matches the canonical libjxl recursive Gaussian
/// — i.e. `Iir`. Other modes (currently `Fir`) produce scores on a
/// **different scale** and should be tagged distinctly downstream (see
/// `column_name_for_blur`).
///
/// Default is `Iir`, which matches the published CPU `ssimulacra2`
/// crate's behaviour bit-identically modulo f32 rounding (the
/// pre-T_y.B opt-in baseline). Without the `fir` feature, the IIR
/// blur is the only available path and `Ssim2` / `Ssim2Batch` have no
/// `with_blur` / `set_blur` / `blur()` knobs at all.
///
/// ```no_run
/// use cubecl::Runtime;
/// use cubecl::wgpu::WgpuRuntime;
/// use ssim2_gpu::{Ssim2, Ssim2Blur};
///
/// let client = WgpuRuntime::client(&Default::default());
/// let s = Ssim2::<WgpuRuntime>::new(client, 256, 256)?
///     .with_blur(Ssim2Blur::default()); // == Iir
/// # Ok::<(), ssim2_gpu::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub enum Ssim2Blur {
    /// Charalampidis 2016 truncated-cosine recursive IIR Gaussian.
    ///
    /// The canonical libjxl SSIMULACRA2 blur, ported to GPU verbatim
    /// from `ssimulacra2-cuda`. Produces scores that match the
    /// published CPU `ssimulacra2` crate to f32-reduction noise
    /// (≤ 0.06 % relative on JPEG q5..q90; see
    /// `tests/parity_lock.rs::parity_jpeg_corpus`).
    ///
    /// Default. Use this unless you have a specific reason to opt in
    /// to a different blur metric.
    #[default]
    Iir,
    /// Separable 5-tap (D = 5) truncated Gaussian FIR at σ = 1.5,
    /// per Kanetaka et al. "Fast Implementation of SSIMULACRA2 for
    /// Image Quality Assessment", IWAIT 2026 (DOI 10.1117/12.3100969).
    ///
    /// **This is a distinct metric.** Per-image scores diverge from
    /// `Iir` because the FIR's effective impulse-response support is
    /// narrower than the IIR's (~2 vs ~5 effective radius). The paper
    /// reports SROCC vs MOS on CID22 of 0.890387 for D=5 — slightly
    /// higher than the libjxl IIR baseline's 0.889297 — but the
    /// per-image score values are NOT the same scale. Downstream
    /// pipelines must tag this implementation distinctly (see
    /// `column_name_for_blur(Ssim2Blur::Fir)`).
    ///
    /// Implementation: a single horizontal 5-tap FIR kernel is used
    /// for both passes — the second pass runs on a transposed
    /// intermediate, so its horizontal walk is a vertical walk in the
    /// original frame. Same `pass → transpose → pass` structure as the
    /// IIR path; same `*_full` output orientation; just a different
    /// per-pass kernel. See `kernels::blur::blur_h_fir5_kernel`.
    Fir,
}

} // end mod blur_mode (cfg fir)

/// Stable column-name identifier for the IIR (default) blur path.
///
/// Used by sweep tooling (`zen-metrics-cli` and downstream pipelines)
/// to land ssim2 scores in parquet sidecars without colliding with
/// other ssim2 variants — when the `fir` feature is enabled the FIR
/// opt-in path uses [`SSIM2_FIR_COLUMN_NAME`] to keep its (distinct)
/// scores in a separate column. Mirrors the `CVVDP_COLUMN_NAME`
/// pattern in `crates/cvvdp-gpu/src/lib.rs`.
///
/// Default form: `ssim2_imazen_iir_v<MAJOR>_<MINOR>_<PATCH>` derived
/// from `CARGO_PKG_VERSION` with `.` rewritten to `_`. The
/// `SSIM2_IIR_IMPL_TAG` build-time env var overrides the entire
/// string when set (e.g. CI bakes in a git short hash to
/// distinguish iterations within the same crate version).
pub const SSIM2_IIR_COLUMN_NAME: &str = match option_env!("SSIM2_IIR_IMPL_TAG") {
    Some(t) => t,
    None => concat!(
        "ssim2_imazen_iir_v",
        env!("CARGO_PKG_VERSION_MAJOR"),
        "_",
        env!("CARGO_PKG_VERSION_MINOR"),
        "_",
        env!("CARGO_PKG_VERSION_PATCH"),
    ),
};

/// Stable column-name identifier for the FIR (opt-in) blur path —
/// **gated behind the `fir` Cargo feature**.
///
/// **Distinct from [`SSIM2_IIR_COLUMN_NAME`]** — the FIR is a
/// different metric (different score scale) per Kanetaka et al. IWAIT
/// 2026. Downstream pipelines must land FIR scores in a different
/// parquet column than IIR scores so they aren't mixed.
///
/// Default form: `ssim2_imazen_fir_v<MAJOR>_<MINOR>_<PATCH>`. Override
/// via the `SSIM2_FIR_IMPL_TAG` build-time env var.
#[cfg(feature = "fir")]
pub const SSIM2_FIR_COLUMN_NAME: &str = match option_env!("SSIM2_FIR_IMPL_TAG") {
    Some(t) => t,
    None => concat!(
        "ssim2_imazen_fir_v",
        env!("CARGO_PKG_VERSION_MAJOR"),
        "_",
        env!("CARGO_PKG_VERSION_MINOR"),
        "_",
        env!("CARGO_PKG_VERSION_PATCH"),
    ),
};

/// Versioned column name for the score produced by a given blur mode
/// — **gated behind the `fir` Cargo feature**.
///
/// Equivalent to:
/// - `Ssim2Blur::Iir` → [`SSIM2_IIR_COLUMN_NAME`]
/// - `Ssim2Blur::Fir` → [`SSIM2_FIR_COLUMN_NAME`]
///
/// Use this to derive the right parquet column name at runtime when
/// the blur mode is data-driven (e.g. CLI flag, config file).
///
/// ```
/// use ssim2_gpu::{Ssim2Blur, column_name_for_blur};
///
/// assert!(column_name_for_blur(Ssim2Blur::Iir).starts_with("ssim2_imazen_iir_v"));
/// assert!(column_name_for_blur(Ssim2Blur::Fir).starts_with("ssim2_imazen_fir_v"));
/// assert_ne!(
///     column_name_for_blur(Ssim2Blur::Iir),
///     column_name_for_blur(Ssim2Blur::Fir),
/// );
/// ```
#[cfg(feature = "fir")]
pub const fn column_name_for_blur(blur: Ssim2Blur) -> &'static str {
    match blur {
        Ssim2Blur::Iir => SSIM2_IIR_COLUMN_NAME,
        Ssim2Blur::Fir => SSIM2_FIR_COLUMN_NAME,
    }
}

/// Result of an SSIMULACRA2 comparison.
///
/// `score` is in roughly the 0–100 range — higher = better quality, 100 =
/// identical, 0 = visually broken. Mirrors the scalar returned by
/// `ssimulacra2::compute_frame_ssimulacra2`.
#[derive(Debug, Clone, Copy)]
pub struct GpuSsim2Result {
    pub score: f64,
}

/// Errors that the GPU SSIMULACRA2 pipeline can return.
#[derive(Debug, Clone)]
pub enum Error {
    /// Buffer length doesn't match the configured `width × height × 3`.
    DimensionMismatch { expected: usize, got: usize },
    /// `compute_with_reference*` was called without a prior `set_reference`.
    NoCachedReference,
    /// Image is smaller than 8×8 — SSIMULACRA2 is undefined there.
    InvalidImageSize,
    /// `Ssim2Batch::new` was called with `batch_size == 0`, or
    /// `compute_batch` got more inputs than the instance's batch_size.
    InvalidBatchSize { got: usize, max: usize },
    /// The requested [`MemoryMode`](crate::MemoryMode) variant isn't
    /// implemented yet (`Tile` in ssim2-gpu's current revision; Strip
    /// shipped 2026-05-22).
    ModeUnsupported(&'static str),
    /// [`MemoryMode::Auto`](crate::MemoryMode) couldn't fit the image
    /// into the VRAM cap with any supported mode. `needed` is the
    /// Full estimate; Strip was tried at the default body height
    /// and also exceeded the cap.
    TooBigForFull { needed: usize, cap: usize },
    /// `set_reference` was called on a strip-mode instance. The
    /// strip pipeline doesn't currently cache per-strip reference
    /// state (would require persisting halo IIR state and the
    /// per-strip ref_xyb_t / mu1_full / sigma11_full buffers — out
    /// of scope for the first strip release). Fall back to the
    /// whole-image path for RD-search if you need a cached reference,
    /// or wait for the v2 strip-aware set_reference.
    CachedRefNotSupportedInStripMode,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::DimensionMismatch { expected, got } => write!(
                f,
                "dimension mismatch: expected {expected} bytes, got {got}"
            ),
            Error::NoCachedReference => write!(f, "no cached reference; call set_reference first"),
            Error::InvalidImageSize => write!(f, "image must be at least 8×8 pixels"),
            Error::InvalidBatchSize { got, max } => write!(
                f,
                "invalid batch size: got {got} images for batch_size = {max}"
            ),
            Error::ModeUnsupported(variant) => write!(
                f,
                "MemoryMode::{variant} is not yet implemented in ssim2-gpu \
                 (Phase 2 strip work is a planned follow-up)"
            ),
            Error::TooBigForFull { needed, cap } => write!(
                f,
                "Auto could not place image in {cap} byte cap; needs at least {needed} bytes \
                 (Strip mode tried at the default body height also exceeded the cap — \
                 raise ZENMETRICS_VRAM_CAP_BYTES or use a smaller image / smaller h_body)"
            ),
            Error::CachedRefNotSupportedInStripMode => write!(
                f,
                "set_reference is not supported on strip-mode instances — \
                 fall back to the whole-image path (Ssim2::new) for cached-reference RD-search"
            ),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
