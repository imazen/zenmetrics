#![forbid(unsafe_code)]

//! GPU IW-SSIM (Information-Content Weighted SSIM) score via the
//! `iwssim-gpu` crate. Score range `[0, 1]` where 1 = identical.
//!
//! Image size constraint: `min(W, H) >= 176` per the paper's 5-level
//! pyramid + 11×11 valid-mode SSIM stats requirement. Smaller images
//! return `Iwssim::Error::InvalidImageSize`; this surfaces to the
//! caller as a failed pair (NaN in score-pairs output).
//!
//! Mirrors `cvvdp_gpu`'s [`CvvdpBatchScorer`] cache pattern: a single
//! `Iwssim::new` instance allocates ~50 MB of GPU buffers at 1024² +
//! triggers per-runtime kernel JIT (NVRTC PTX cache, ~5 s cold).
//! Recreating per-pair would dominate sweep wall time and OOM the
//! 1080-Ti class boxes; the cached-by-(W, H) scorer below survives
//! across thousands of same-dim pairs.

use cubecl::Runtime;

use crate::decode::Rgb8Image;
use crate::metrics::GpuRuntime;
use crate::metrics::gpu_runtime_dispatch::{auto_order, runtime_label};

/// Batched IW-SSIM scorer that caches the `iwssim_gpu::Iwssim`
/// instance by `(width, height)` across many score-pairs calls.
///
/// Same rationale as [`crate::metrics::cvvdp_gpu::CvvdpBatchScorer`]
/// — `Iwssim::new` is expensive (GPU buffer allocation + JIT) and
/// the per-pair `run::<R>()` path would multiply that cost by chunk
/// size. Sort sweep inputs by `(W, H)` to maximise cache hit rate.
pub enum IwssimBatchScorer {
    #[cfg(feature = "gpu-cuda")]
    Cuda(IwssimBatchScorerState<cubecl::cuda::CudaRuntime>),
    #[cfg(feature = "gpu-wgpu")]
    Wgpu(IwssimBatchScorerState<cubecl::wgpu::WgpuRuntime>),
    #[cfg(feature = "gpu-hip")]
    Hip(IwssimBatchScorerState<cubecl::hip::HipRuntime>),
    #[cfg(feature = "gpu-cpu")]
    Cpu(IwssimBatchScorerState<cubecl::cpu::CpuRuntime>),
}

/// Per-runtime cache slot. Generic on `R: Runtime` so the cached
/// `Iwssim` retains its concrete type for the `score` call.
#[doc(hidden)]
pub struct IwssimBatchScorerState<R: Runtime> {
    client: cubecl::client::ComputeClient<R>,
    cached: Option<(u32, u32, iwssim_gpu::Iwssim<R>)>,
}

impl IwssimBatchScorer {
    /// Construct the scorer for the given runtime. `Auto` walks the
    /// compiled-in order (CUDA → wgpu → HIP → CPU), returning the
    /// first runtime that initialises a `ComputeClient`. Once locked,
    /// it stays — no silent per-call fallback.
    pub fn new(runtime: GpuRuntime) -> Result<Self, Box<dyn std::error::Error>> {
        let candidates: Vec<GpuRuntime> = match runtime {
            GpuRuntime::Auto => auto_order().to_vec(),
            other => vec![other],
        };
        let mut last_error: Option<String> = None;
        for rt in candidates {
            match Self::try_new_with_runtime(rt) {
                Ok(s) => return Ok(s),
                Err(e) => last_error = Some(format!("{}: {e}", runtime_label(rt))),
            }
        }
        Err(format!(
            "IwssimBatchScorer::new: no runtime succeeded; last error: {}",
            last_error.unwrap_or_else(|| "none".into())
        )
        .into())
    }

    fn try_new_with_runtime(runtime: GpuRuntime) -> Result<Self, Box<dyn std::error::Error>> {
        match runtime {
            #[cfg(feature = "gpu-cuda")]
            GpuRuntime::Cuda => Ok(Self::Cuda(IwssimBatchScorerState {
                client: <cubecl::cuda::CudaRuntime as Runtime>::client(&Default::default()),
                cached: None,
            })),
            #[cfg(not(feature = "gpu-cuda"))]
            GpuRuntime::Cuda => {
                Err("cuda runtime not compiled in (rebuild with `--features gpu-cuda`)".into())
            }
            #[cfg(feature = "gpu-wgpu")]
            GpuRuntime::Wgpu => Ok(Self::Wgpu(IwssimBatchScorerState {
                client: <cubecl::wgpu::WgpuRuntime as Runtime>::client(&Default::default()),
                cached: None,
            })),
            #[cfg(not(feature = "gpu-wgpu"))]
            GpuRuntime::Wgpu => {
                Err("wgpu runtime not compiled in (rebuild with `--features gpu-wgpu`)".into())
            }
            #[cfg(feature = "gpu-hip")]
            GpuRuntime::Hip => Ok(Self::Hip(IwssimBatchScorerState {
                client: <cubecl::hip::HipRuntime as Runtime>::client(&Default::default()),
                cached: None,
            })),
            #[cfg(not(feature = "gpu-hip"))]
            GpuRuntime::Hip => {
                Err("hip runtime not compiled in (rebuild with `--features gpu-hip`)".into())
            }
            #[cfg(feature = "gpu-cpu")]
            GpuRuntime::Cpu => Ok(Self::Cpu(IwssimBatchScorerState {
                client: <cubecl::cpu::CpuRuntime as Runtime>::client(&Default::default()),
                cached: None,
            })),
            #[cfg(not(feature = "gpu-cpu"))]
            GpuRuntime::Cpu => {
                Err("cpu runtime not compiled in (rebuild with `--features gpu-cpu`)".into())
            }
            GpuRuntime::Auto => unreachable!("Auto is expanded by new()"),
        }
    }

    /// Score a (reference, distorted) pair. Reuses the cached
    /// `Iwssim` instance when dims match; rebuilds otherwise.
    pub fn score(
        &mut self,
        reference: &Rgb8Image,
        distorted: &Rgb8Image,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        match self {
            #[cfg(feature = "gpu-cuda")]
            Self::Cuda(state) => score_pair_cached(state, reference, distorted),
            #[cfg(feature = "gpu-wgpu")]
            Self::Wgpu(state) => score_pair_cached(state, reference, distorted),
            #[cfg(feature = "gpu-hip")]
            Self::Hip(state) => score_pair_cached(state, reference, distorted),
            #[cfg(feature = "gpu-cpu")]
            Self::Cpu(state) => score_pair_cached(state, reference, distorted),
            #[cfg(not(any(
                feature = "gpu-cuda",
                feature = "gpu-wgpu",
                feature = "gpu-hip",
                feature = "gpu-cpu",
            )))]
            _ => {
                let _ = (reference, distorted);
                Err("no CubeCL runtime feature enabled at build time \
                     (rebuild with at least one of `gpu-cuda`, `gpu-wgpu`, \
                     `gpu-hip`, `gpu-cpu`)"
                    .into())
            }
        }
    }
}

fn score_pair_cached<R: Runtime>(
    state: &mut IwssimBatchScorerState<R>,
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
) -> Result<f64, Box<dyn std::error::Error>> {
    let (w, h) = (reference.width, reference.height);
    if distorted.width != w || distorted.height != h {
        return Err(format!(
            "dimension mismatch: ref {}x{} vs dist {}x{}",
            w, h, distorted.width, distorted.height
        )
        .into());
    }
    let needs_rebuild = !matches!(state.cached, Some((cw, ch, _)) if cw == w && ch == h);
    if needs_rebuild {
        // Drop the prior cache slot before allocating the new one
        // so peak GPU memory stays at one instance's worth. `Iwssim`
        // releases all its buffers on drop.
        state.cached = None;
        let i = iwssim_gpu::Iwssim::<R>::new(state.client.clone(), w, h)
            .map_err(|e| format!("Iwssim::new ({w}x{h}): {e}"))?;
        state.cached = Some((w, h, i));
    }
    let i = &mut state.cached.as_mut().expect("just populated").2;
    let result = i
        .compute_rgb(&reference.pixels, &distorted.pixels)
        .map_err(|e| format!("Iwssim::compute_rgb: {e}"))?;
    if !result.score.is_finite() {
        return Err(format!("iwssim produced non-finite score: {}", result.score).into());
    }
    Ok(result.score)
}

/// Single-pair score path — mirrors `dssim_gpu::score`. Used by the
/// general `run_metric` dispatch when the caller doesn't go through
/// score-pairs. score-pairs uses [`IwssimBatchScorer`] instead.
pub fn score(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
    runtime: GpuRuntime,
) -> Result<f64, Box<dyn std::error::Error>> {
    let candidates: Vec<GpuRuntime> = match runtime {
        GpuRuntime::Auto => auto_order().to_vec(),
        other => vec![other],
    };

    let mut last_error: Option<String> = None;
    for rt in candidates {
        match score_with_runtime(reference, distorted, rt) {
            Ok(value) => return Ok(value),
            Err(e) => {
                last_error = Some(format!("{}: {e}", runtime_label(rt)));
            }
        }
    }
    Err(format!(
        "iwssim-gpu: no runtime succeeded; last error: {}",
        last_error.unwrap_or_else(|| "none".into())
    )
    .into())
}

fn score_with_runtime(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
    runtime: GpuRuntime,
) -> Result<f64, Box<dyn std::error::Error>> {
    match runtime {
        GpuRuntime::Cuda => {
            #[cfg(feature = "gpu-cuda")]
            {
                run::<cubecl::cuda::CudaRuntime>(reference, distorted)
            }
            #[cfg(not(feature = "gpu-cuda"))]
            {
                Err("cuda runtime not compiled in (rebuild with `--features gpu-cuda`)".into())
            }
        }
        GpuRuntime::Wgpu => {
            #[cfg(feature = "gpu-wgpu")]
            {
                run::<cubecl::wgpu::WgpuRuntime>(reference, distorted)
            }
            #[cfg(not(feature = "gpu-wgpu"))]
            {
                Err("wgpu runtime not compiled in (rebuild with `--features gpu-wgpu`)".into())
            }
        }
        GpuRuntime::Hip => {
            #[cfg(feature = "gpu-hip")]
            {
                run::<cubecl::hip::HipRuntime>(reference, distorted)
            }
            #[cfg(not(feature = "gpu-hip"))]
            {
                Err("hip runtime not compiled in (rebuild with `--features gpu-hip`)".into())
            }
        }
        GpuRuntime::Cpu => {
            #[cfg(feature = "gpu-cpu")]
            {
                run::<cubecl::cpu::CpuRuntime>(reference, distorted)
            }
            #[cfg(not(feature = "gpu-cpu"))]
            {
                Err("cpu runtime not compiled in (rebuild with `--features gpu-cpu`)".into())
            }
        }
        GpuRuntime::Auto => unreachable!("Auto is expanded by score()"),
    }
}

fn run<R: Runtime>(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
) -> Result<f64, Box<dyn std::error::Error>> {
    let client = R::client(&Default::default());
    let mut i = iwssim_gpu::Iwssim::<R>::new(client, reference.width, reference.height)
        .map_err(|e| format!("Iwssim::new: {e}"))?;
    let result = i
        .compute_rgb(&reference.pixels, &distorted.pixels)
        .map_err(|e| format!("Iwssim::compute_rgb: {e}"))?;
    if !result.score.is_finite() {
        return Err(format!("iwssim-gpu produced non-finite score: {}", result.score).into());
    }
    Ok(result.score)
}
