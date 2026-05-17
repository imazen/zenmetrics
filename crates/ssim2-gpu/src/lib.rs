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
pub mod pipeline;
pub mod pipeline_batch;

pub use pipeline::Ssim2;
pub use pipeline_batch::Ssim2Batch;

/// Number of pyramid scales — matches both the CPU and CUDA references.
pub const NUM_SCALES: usize = 6;

/// Blur-kernel implementation selector.
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
/// pre-T_y.B opt-in baseline).
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
    ///
    /// Available as of T_y.B.2 commit 3.
    Fir,
}

/// Stable column-name identifier for the IIR (default) blur path.
///
/// Used by sweep tooling (`zen-metrics-cli` and downstream pipelines)
/// to land ssim2 scores in parquet sidecars without colliding with
/// other ssim2 variants — currently the FIR opt-in path
/// ([`SSIM2_FIR_COLUMN_NAME`]). Mirrors the `CVVDP_COLUMN_NAME`
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

/// Stable column-name identifier for the FIR (opt-in) blur path.
///
/// **Distinct from [`SSIM2_IIR_COLUMN_NAME`]** — the FIR is a
/// different metric (different score scale) per Kanetaka et al. IWAIT
/// 2026. Downstream pipelines must land FIR scores in a different
/// parquet column than IIR scores so they aren't mixed.
///
/// Default form: `ssim2_imazen_fir_v<MAJOR>_<MINOR>_<PATCH>`. Override
/// via the `SSIM2_FIR_IMPL_TAG` build-time env var.
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

/// Versioned column name for the score produced by a given blur mode.
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
    /// `compute*` was called on an instance whose blur mode is
    /// `Ssim2Blur::Fir`, but the FIR kernel hasn't been wired yet
    /// (T_y.B.2 commit 1 ships the skeleton; commit 3 lands the
    /// kernel). Set the blur back to `Ssim2Blur::Iir` (the default)
    /// to compute scores, or upgrade to a later crate version.
    FirNotYetImplemented,
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
            Error::FirNotYetImplemented => write!(
                f,
                "Ssim2Blur::Fir is reserved but not yet implemented in this build; \
                 use Ssim2Blur::Iir (default) or upgrade"
            ),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
