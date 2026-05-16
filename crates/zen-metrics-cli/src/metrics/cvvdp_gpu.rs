#![forbid(unsafe_code)]

//! ColorVideoVDP (still-image) score via the `cvvdp-gpu` crate.
//!
//! cvvdp's JOD is on a 0–10 scale where 10 = imperceptible. Routes
//! through `cvvdp_gpu::Cvvdp::score`, which since cvvdp-gpu tick 213
//! runs the **full GPU composition path** (`compute_dkl_jod`: color
//! → Weber pyramid → CSF → masking → GPU atomic pool → host
//! Minkowski fold). Output matches pycvvdp v0.5.4 on the v1 R2
//! manifest within **0.005 JOD** (measured 0.0000–0.0031 across
//! q=1,5,20,45,70,90) under the default `PerfMode::Strict`;
//! `shadow_jod_gpu` pins this. Earlier doc revisions described
//! the CLI as still routing through the host-scalar path with a
//! pending retarget — that retarget landed in cvvdp-gpu tick 213
//! (`compute_dkl_jod_matches_pycvvdp_at_*` parity tests close
//! q=1 drift to 0.0000 after ticks 204/206 fixed the
//! chroma_shift + 73×91 odd-dim drifts).
//!
//! Backend dispatch: `--gpu-runtime auto` tries Cuda → Wgpu → Hip
//! → Cpu in order, returning the first successful score. The Cpu
//! arm routes through `Cvvdp::compute_dkl_jod_host_pool` rather
//! than the GPU pool kernel, since the latter uses
//! `Atomic<f32>::fetch_add` which `cubecl-cpu` does not implement
//! (would panic "atomic<f32> not implemented"). Same JOD output
//! either way — parity-locked at f32 noise by
//! `compute_dkl_jod_host_pool_matches_compute_dkl_jod` in
//! cvvdp-gpu.

use cubecl::Runtime;

use crate::decode::Rgb8Image;
use crate::metrics::GpuRuntime;
use crate::metrics::gpu_runtime_dispatch::{auto_order, runtime_label};

/// Batched cvvdp scorer that caches the `cvvdp_gpu::Cvvdp` instance
/// by `(width, height)` across many score-pairs calls.
///
/// **Why this exists**: `cvvdp_gpu::Cvvdp::new()` allocates ~200 MB
/// of GPU buffers at 1024² and triggers NVRTC kernel compilation
/// (one-time per-process cost). Creating a fresh `Cvvdp` per pair
/// — as the simple `run::<R>()` path does — multiplies that cost
/// by the pair count and produces real-world OOM on workers
/// running 100-pair chunks even with PARALLEL=1 + 16 GB RAM (the
/// driver-side allocator fragments + the host-pinned NVRTC PTX
/// cache balloons). Caching the instance flips the per-call
/// allocation cost back to zero for the common sweep pattern
/// (homogeneous-dim batches).
///
/// Runtime is locked at construction time — once a scorer picks
/// CUDA, it stays on CUDA for every pair. This is a deliberate
/// simplification vs `score_with_runtime`'s per-call auto
/// fall-through: a sweep wants consistent backend selection, not
/// surprise CPU fallbacks half-way through a chunk.
///
/// Dimension cache shape: a single slot. When a pair arrives with
/// dims differing from the cached slot, the old `Cvvdp` is dropped
/// + a fresh one allocated. Sweep producers should sort pairs by
/// (width, height) to maximise hit rate; the cvvdp-backfill
/// pipeline does this implicitly by grouping by source image.
pub enum CvvdpBatchScorer {
    #[cfg(feature = "gpu-cuda")]
    Cuda(CvvdpBatchScorerState<cubecl::cuda::CudaRuntime>),
    #[cfg(feature = "gpu-wgpu")]
    Wgpu(CvvdpBatchScorerState<cubecl::wgpu::WgpuRuntime>),
    #[cfg(feature = "gpu-hip")]
    Hip(CvvdpBatchScorerState<cubecl::hip::HipRuntime>),
    #[cfg(feature = "gpu-cpu")]
    Cpu(CvvdpBatchScorerCpuState<cubecl::cpu::CpuRuntime>),
}

/// Per-runtime cache slot for [`CvvdpBatchScorer`]. Generic on the
/// concrete `R: Runtime` so the cached `Cvvdp` can be stored
/// without type erasure.
#[doc(hidden)]
pub struct CvvdpBatchScorerState<R: Runtime> {
    client: cubecl::client::ComputeClient<R>,
    cached: Option<(u32, u32, cvvdp_gpu::Cvvdp<R>)>,
}

/// CPU variant — routes through `compute_dkl_jod_host_pool` since
/// `Cvvdp::score`'s atomic-f32 pool kernel doesn't run on
/// `cubecl-cpu`. Stored separately so the `score` dispatch matches
/// the right runtime path.
#[doc(hidden)]
#[cfg(feature = "gpu-cpu")]
pub struct CvvdpBatchScorerCpuState<R: Runtime> {
    client: cubecl::client::ComputeClient<R>,
    cached: Option<(u32, u32, cvvdp_gpu::Cvvdp<R>)>,
}

impl CvvdpBatchScorer {
    /// Construct a scorer for the given runtime. Auto resolves to
    /// the first available runtime (Cuda → Wgpu → Hip → Cpu) at
    /// construction time, NOT per-call. If the chosen runtime
    /// fails on the first `score` call, the caller should rebuild
    /// the scorer with an explicit fallback runtime — silent
    /// per-call fall-through is the bug the cache fixes.
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
            "CvvdpBatchScorer::new: no runtime succeeded; last error: {}",
            last_error.unwrap_or_else(|| "none".into())
        )
        .into())
    }

    fn try_new_with_runtime(runtime: GpuRuntime) -> Result<Self, Box<dyn std::error::Error>> {
        match runtime {
            #[cfg(feature = "gpu-cuda")]
            GpuRuntime::Cuda => Ok(Self::Cuda(CvvdpBatchScorerState {
                client: <cubecl::cuda::CudaRuntime as Runtime>::client(&Default::default()),
                cached: None,
            })),
            #[cfg(not(feature = "gpu-cuda"))]
            GpuRuntime::Cuda => {
                Err("cuda runtime not compiled in (rebuild with `--features gpu-cuda`)".into())
            }
            #[cfg(feature = "gpu-wgpu")]
            GpuRuntime::Wgpu => Ok(Self::Wgpu(CvvdpBatchScorerState {
                client: <cubecl::wgpu::WgpuRuntime as Runtime>::client(&Default::default()),
                cached: None,
            })),
            #[cfg(not(feature = "gpu-wgpu"))]
            GpuRuntime::Wgpu => {
                Err("wgpu runtime not compiled in (rebuild with `--features gpu-wgpu`)".into())
            }
            #[cfg(feature = "gpu-hip")]
            GpuRuntime::Hip => Ok(Self::Hip(CvvdpBatchScorerState {
                client: <cubecl::hip::HipRuntime as Runtime>::client(&Default::default()),
                cached: None,
            })),
            #[cfg(not(feature = "gpu-hip"))]
            GpuRuntime::Hip => {
                Err("hip runtime not compiled in (rebuild with `--features gpu-hip`)".into())
            }
            #[cfg(feature = "gpu-cpu")]
            GpuRuntime::Cpu => Ok(Self::Cpu(CvvdpBatchScorerCpuState {
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
    /// `Cvvdp` instance when the dims match the previous call;
    /// rebuilds otherwise.
    pub fn score(
        &mut self,
        reference: &Rgb8Image,
        distorted: &Rgb8Image,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        match self {
            #[cfg(feature = "gpu-cuda")]
            Self::Cuda(state) => score_pair_cached_gpu(state, reference, distorted),
            #[cfg(feature = "gpu-wgpu")]
            Self::Wgpu(state) => score_pair_cached_gpu(state, reference, distorted),
            #[cfg(feature = "gpu-hip")]
            Self::Hip(state) => score_pair_cached_gpu(state, reference, distorted),
            #[cfg(feature = "gpu-cpu")]
            Self::Cpu(state) => score_pair_cached_cpu(state, reference, distorted),
        }
    }
}

/// Cached-instance scoring on a GPU runtime — routes through
/// `Cvvdp::score` which uses the GPU atomic-f32 pool kernel.
fn score_pair_cached_gpu<R: Runtime>(
    state: &mut CvvdpBatchScorerState<R>,
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
        // so peak GPU memory stays at one instance's worth rather
        // than two. `Cvvdp` releases all its buffers on drop.
        state.cached = None;
        let c = cvvdp_gpu::Cvvdp::<R>::new(
            state.client.clone(),
            w,
            h,
            cvvdp_gpu::CvvdpParams::PLACEHOLDER,
        )
        .map_err(|e| format!("Cvvdp::new ({w}x{h}): {e}"))?;
        state.cached = Some((w, h, c));
    }
    let c = &mut state.cached.as_mut().expect("just populated").2;
    let jod = c
        .score(&reference.pixels, &distorted.pixels)
        .map_err(|e| format!("Cvvdp::score: {e}"))?;
    if !jod.is_finite() {
        return Err(format!("cvvdp produced non-finite JOD: {jod}").into());
    }
    Ok(jod)
}

/// Cached-instance scoring on `cubecl-cpu` — routes through
/// `compute_dkl_jod_host_pool` since the GPU atomic pool kernel
/// doesn't run on the CPU backend.
#[cfg(feature = "gpu-cpu")]
fn score_pair_cached_cpu<R: Runtime>(
    state: &mut CvvdpBatchScorerCpuState<R>,
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
        state.cached = None;
        let c = cvvdp_gpu::Cvvdp::<R>::new(
            state.client.clone(),
            w,
            h,
            cvvdp_gpu::CvvdpParams::PLACEHOLDER,
        )
        .map_err(|e| format!("Cvvdp::new (cpu host_pool, {w}x{h}): {e}"))?;
        state.cached = Some((w, h, c));
    }
    let c = &mut state.cached.as_mut().expect("just populated").2;
    let ppd = cvvdp_gpu::params::DisplayGeometry::STANDARD_4K.pixels_per_degree();
    let jod = c
        .compute_dkl_jod_host_pool(&reference.pixels, &distorted.pixels, ppd)
        .map_err(|e| format!("Cvvdp::compute_dkl_jod_host_pool: {e}"))?;
    if !jod.is_finite() {
        return Err(format!("cvvdp (cpu host_pool) produced non-finite JOD: {jod}").into());
    }
    Ok(f64::from(jod))
}

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
        "cvvdp: no runtime succeeded; last error: {}",
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
                // cvvdp-gpu's pool kernel uses Atomic<f32>::fetch_add,
                // which cubecl-cpu does not implement (panic at
                // cubecl-cpu/compiler/visitor/elem.rs: "atomic<f32>
                // not implemented"). The crate ships
                // Cvvdp::compute_dkl_jod_host_pool for exactly this
                // case — same JOD output, but pools the per-band
                // D values on the host instead of via the atomic
                // kernel. Without this routing the auto-dispatch
                // fall-through (Cuda → Wgpu → Hip → Cpu) bombs the
                // whole sweep on boxes where CUDA fails to init.
                run_cpu_host_pool::<cubecl::cpu::CpuRuntime>(reference, distorted)
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
    let mut c = cvvdp_gpu::Cvvdp::<R>::new(
        client,
        reference.width,
        reference.height,
        cvvdp_gpu::CvvdpParams::PLACEHOLDER,
    )
    .map_err(|e| format!("Cvvdp::new: {e}"))?;
    let jod = c
        .score(&reference.pixels, &distorted.pixels)
        .map_err(|e| format!("Cvvdp::score: {e}"))?;
    if !jod.is_finite() {
        return Err(format!("cvvdp produced non-finite JOD: {jod}").into());
    }
    Ok(jod)
}

/// Sibling of `run` that routes through
/// [`cvvdp_gpu::Cvvdp::compute_dkl_jod_host_pool`] instead of
/// `Cvvdp::score`. Same JOD output (parity-locked at f32 noise
/// by `compute_dkl_jod_host_pool_matches_compute_dkl_jod`), but
/// pools per-band D values host-side so it runs on runtimes
/// without `Atomic<f32>::fetch_add` (`cubecl-cpu`, Metal via
/// `cubecl-wgpu` — see [`cvvdp_gpu::Cvvdp::compute_dkl_jod`]'s
/// Backend support section).
#[cfg(feature = "gpu-cpu")]
fn run_cpu_host_pool<R: Runtime>(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
) -> Result<f64, Box<dyn std::error::Error>> {
    let client = R::client(&Default::default());
    let mut c = cvvdp_gpu::Cvvdp::<R>::new(
        client,
        reference.width,
        reference.height,
        cvvdp_gpu::CvvdpParams::PLACEHOLDER,
    )
    .map_err(|e| format!("Cvvdp::new (cpu host_pool): {e}"))?;
    // Cvvdp::new uses DisplayGeometry::STANDARD_4K (per pipeline.rs:478);
    // mirror that here since `geometry` is a private field and there's
    // no public accessor. Cvvdp::score does the same internally.
    let ppd = cvvdp_gpu::params::DisplayGeometry::STANDARD_4K.pixels_per_degree();
    let jod = c
        .compute_dkl_jod_host_pool(&reference.pixels, &distorted.pixels, ppd)
        .map_err(|e| format!("Cvvdp::compute_dkl_jod_host_pool: {e}"))?;
    if !jod.is_finite() {
        return Err(format!("cvvdp (cpu host_pool) produced non-finite JOD: {jod}").into());
    }
    Ok(f64::from(jod))
}
