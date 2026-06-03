#![forbid(unsafe_code)]

//! ColorVideoVDP (still-image) score via the `cvvdp-gpu` crate.
//!
//! cvvdp's JOD is on a 0–10 scale where 10 = imperceptible. Routes
//! through `cvvdp::Cvvdp::score`, which since cvvdp-gpu tick 213
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
//!
//! The typed `Cvvdp<R>` / `CvvdpParams` types are reached through the
//! umbrella's `zenmetrics_api::cvvdp` re-export so this CLI keeps a
//! single GPU-metric dep line. The umbrella's `cubecl-types` feature
//! is what makes the typed surface visible — see
//! `Cargo.toml`'s `gpu-cvvdp` feature, which propagates that flag.

use cubecl::Runtime;
use zenmetrics_api::cvvdp;

use crate::decode::Rgb8Image;
use crate::metrics::{GpuRuntime, auto_order, runtime_label};

/// Display viewing conditions for a cvvdp scorer: photometry
/// (`DisplayModel` — peak/black luminance, ambient reflection, EOTF)
/// + geometry (`DisplayGeometry` — resolution / distance / diagonal,
/// which derives pixels-per-degree).
///
/// Both halves matter and both must be set together: the photometry
/// drives the sRGB→linear-cd/m² conversion in the color stage, and
/// the geometry drives the CSF LUT that is pre-uploaded at
/// `Cvvdp::new_with_geometry` time. Scoring the same pair under two
/// different `DisplayTarget`s yields different JOD scores — that is
/// the whole point of display-aware quality assessment (a phone at
/// arm's length has higher PPD + higher peak luminance than a 4K
/// desktop monitor, so artifacts are differently visible).
///
/// The default is `STANDARD_4K` (200 cd/m², 75.4 PPD), matching every
/// historical CLI score and the v1 R2 parity goldens.
#[derive(Debug, Clone, Copy)]
pub struct DisplayTarget {
    /// Photometric model — peak/black luminance, ambient, EOTF.
    pub display: cvvdp::params::DisplayModel,
    /// Viewing geometry — resolution + distance + diagonal → PPD.
    pub geometry: cvvdp::params::DisplayGeometry,
}

impl Default for DisplayTarget {
    fn default() -> Self {
        Self {
            display: cvvdp::params::DisplayModel::STANDARD_4K,
            geometry: cvvdp::params::DisplayGeometry::STANDARD_4K,
        }
    }
}

impl DisplayTarget {
    /// Resolve a `DisplayTarget` from a preset name (as found in the
    /// vendored `display_models.json`, e.g. `"standard_4k"`,
    /// `"iphone_14_pro"`, `"standard_phone"`). Both the photometry
    /// and the geometry are loaded for the same name via
    /// [`cvvdp::params::DisplayModel::by_name`] /
    /// [`cvvdp::params::DisplayGeometry::by_name`].
    ///
    /// Returns `Err` listing the available preset names when `name`
    /// is unknown, or when the named preset has photometry but no
    /// geometry (FOV-only presets without resolution).
    pub fn by_name(name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let display = cvvdp::params::DisplayModel::by_name(name).ok_or_else(|| {
            format!(
                "unknown --display-model preset {name:?}; \
                 see `display_models.json` for valid names \
                 (e.g. standard_4k, iphone_14_pro, standard_phone)"
            )
        })?;
        let geometry = cvvdp::params::DisplayGeometry::by_name(name).ok_or_else(|| {
            format!(
                "--display-model preset {name:?} has photometry but no \
                 geometry (resolution); cvvdp scoring needs both"
            )
        })?;
        Ok(Self { display, geometry })
    }

    /// An HDR display target for the faithful linear-planes HDR path: a
    /// `peak_nits` cd/m² display with a linear EOTF (the linear-planes entry
    /// skips the EOTF — input is already linear-light) and BT.709 primaries
    /// (the same color treatment as the sRGB SDR path, so the HDR win is
    /// isolated to luminance range; wide-gamut primaries are a follow-up).
    /// Geometry is `STANDARD_4K`, matching the default viewing conditions.
    /// Pair with `crate::hdr::to_cvvdp_linear_planes` (which normalizes nits to
    /// `peak_nits`). `peak_nits` should equal `crate::hdr::HDR_DISPLAY_PEAK_NITS`.
    pub fn hdr(peak_nits: f32) -> Self {
        Self {
            display: cvvdp::params::DisplayModel {
                y_peak: peak_nits,
                ..cvvdp::params::DisplayModel::STANDARD_HDR_LINEAR
            },
            geometry: cvvdp::params::DisplayGeometry::STANDARD_4K,
        }
    }

    /// The `CvvdpParams` for this target: `PLACEHOLDER` with the
    /// photometric `display` field swapped in. The other scaffolding
    /// sub-bundles are unused by the production kernels (see
    /// `CvvdpParams::PLACEHOLDER` docs).
    fn params(&self) -> cvvdp::CvvdpParams {
        cvvdp::CvvdpParams {
            display: self.display,
            ..cvvdp::CvvdpParams::PLACEHOLDER
        }
    }
}

/// Batched cvvdp scorer that caches the `cvvdp::Cvvdp` instance
/// by `(width, height)` across many score-pairs calls.
///
/// **Why this exists**: `cvvdp::Cvvdp::new()` allocates ~200 MB
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
    cached: Option<(u32, u32, cvvdp::Cvvdp<R>)>,
    target: DisplayTarget,
}

/// CPU variant — routes through `compute_dkl_jod_host_pool` since
/// `Cvvdp::score`'s atomic-f32 pool kernel doesn't run on
/// `cubecl-cpu`. Stored separately so the `score` dispatch matches
/// the right runtime path.
#[doc(hidden)]
#[cfg(feature = "gpu-cpu")]
pub struct CvvdpBatchScorerCpuState<R: Runtime> {
    client: cubecl::client::ComputeClient<R>,
    cached: Option<(u32, u32, cvvdp::Cvvdp<R>)>,
    target: DisplayTarget,
}

impl CvvdpBatchScorer {
    /// Construct a scorer for the given runtime. Auto resolves to
    /// the first available runtime (Cuda → Wgpu → Hip → Cpu) at
    /// construction time, NOT per-call. If the chosen runtime
    /// fails on the first `score` call, the caller should rebuild
    /// the scorer with an explicit fallback runtime — silent
    /// per-call fall-through is the bug the cache fixes.
    pub fn new(runtime: GpuRuntime) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_with_target(runtime, DisplayTarget::default())
    }

    /// Construct a scorer for the given runtime AND a specific
    /// [`DisplayTarget`] (photometry + geometry). The target is baked
    /// into every `Cvvdp` instance this scorer caches — both the
    /// CSF-LUT PPD (from `target.geometry`) and the color-stage
    /// photometry (from `target.display`). `new` is the back-compat
    /// `STANDARD_4K` shorthand.
    pub fn new_with_target(
        runtime: GpuRuntime,
        target: DisplayTarget,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let candidates: Vec<GpuRuntime> = match runtime {
            GpuRuntime::Auto => auto_order().to_vec(),
            other => vec![other],
        };
        let mut last_error: Option<String> = None;
        for rt in candidates {
            match Self::try_new_with_runtime(rt, target) {
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

    fn try_new_with_runtime(
        runtime: GpuRuntime,
        target: DisplayTarget,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        match runtime {
            #[cfg(feature = "gpu-cuda")]
            GpuRuntime::Cuda => Ok(Self::Cuda(CvvdpBatchScorerState {
                client: <cubecl::cuda::CudaRuntime as Runtime>::client(&Default::default()),
                cached: None,
                target,
            })),
            #[cfg(not(feature = "gpu-cuda"))]
            GpuRuntime::Cuda => {
                Err("cuda runtime not compiled in (rebuild with `--features gpu-cuda`)".into())
            }
            #[cfg(feature = "gpu-wgpu")]
            GpuRuntime::Wgpu => Ok(Self::Wgpu(CvvdpBatchScorerState {
                client: <cubecl::wgpu::WgpuRuntime as Runtime>::client(&Default::default()),
                cached: None,
                target,
            })),
            #[cfg(not(feature = "gpu-wgpu"))]
            GpuRuntime::Wgpu => {
                Err("wgpu runtime not compiled in (rebuild with `--features gpu-wgpu`)".into())
            }
            #[cfg(feature = "gpu-hip")]
            GpuRuntime::Hip => Ok(Self::Hip(CvvdpBatchScorerState {
                client: <cubecl::hip::HipRuntime as Runtime>::client(&Default::default()),
                cached: None,
                target,
            })),
            #[cfg(not(feature = "gpu-hip"))]
            GpuRuntime::Hip => {
                Err("hip runtime not compiled in (rebuild with `--features gpu-hip`)".into())
            }
            #[cfg(feature = "gpu-cpu")]
            GpuRuntime::Cpu => Ok(Self::Cpu(CvvdpBatchScorerCpuState {
                client: <cubecl::cpu::CpuRuntime as Runtime>::client(&Default::default()),
                cached: None,
                target,
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
            // When none of the gpu-cuda/wgpu/hip/cpu features are enabled the
            // enum has zero constructable variants, but `&mut Self` is still
            // an inhabited type so the match would be non-exhaustive without
            // a wildcard. `Self::new()` always errors in that configuration,
            // so this arm is unreachable in practice — keep it as a runtime
            // guard rather than a `todo!()` panic.
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

    /// Faithful HDR scoring from display-relative `[0,1]` linear planes
    /// (see `crate::hdr::to_cvvdp_linear_planes`). Routes through cvvdp's
    /// native `score_from_linear_planes` — no sRGB8 round-trip, full HDR
    /// highlight precision. The scorer's `DisplayTarget` (typically
    /// `DisplayTarget::hdr(..)`) supplies the HDR peak + primaries. Tightly
    /// packed planes (`padded_width == width`). Requires a GPU runtime — the
    /// cubecl-cpu backend lacks the atomic-f32 pool kernel this path uses.
    #[allow(clippy::too_many_arguments)]
    pub fn score_from_linear_planes(
        &mut self,
        ref_r: &[f32],
        ref_g: &[f32],
        ref_b: &[f32],
        dis_r: &[f32],
        dis_g: &[f32],
        dis_b: &[f32],
        width: u32,
        height: u32,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        match self {
            #[cfg(feature = "gpu-cuda")]
            Self::Cuda(state) => score_pair_cached_linear_planes_gpu(
                state, ref_r, ref_g, ref_b, dis_r, dis_g, dis_b, width, height,
            ),
            #[cfg(feature = "gpu-wgpu")]
            Self::Wgpu(state) => score_pair_cached_linear_planes_gpu(
                state, ref_r, ref_g, ref_b, dis_r, dis_g, dis_b, width, height,
            ),
            #[cfg(feature = "gpu-hip")]
            Self::Hip(state) => score_pair_cached_linear_planes_gpu(
                state, ref_r, ref_g, ref_b, dis_r, dis_g, dis_b, width, height,
            ),
            #[cfg(feature = "gpu-cpu")]
            Self::Cpu(_) => Err("faithful HDR cvvdp (linear planes) requires a GPU \
                 runtime; the cubecl-cpu backend lacks the atomic-f32 pool kernel"
                .into()),
            #[cfg(not(any(
                feature = "gpu-cuda",
                feature = "gpu-wgpu",
                feature = "gpu-hip",
                feature = "gpu-cpu",
            )))]
            _ => {
                let _ = (ref_r, ref_g, ref_b, dis_r, dis_g, dis_b, width, height);
                Err("no CubeCL runtime feature enabled at build time \
                     (rebuild with at least one of `gpu-cuda`, `gpu-wgpu`, \
                     `gpu-hip`, `gpu-cpu`)"
                    .into())
            }
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
        // Bake BOTH the photometry (params.display) AND the geometry
        // (PPD → CSF LUT) of the requested display target into this
        // instance. STANDARD_4K is the default; iphone_14_pro etc.
        // flow through `--display-model`.
        let c = cvvdp::Cvvdp::<R>::new_with_geometry(
            state.client.clone(),
            w,
            h,
            state.target.params(),
            state.target.geometry,
        )
        .map_err(|e| format!("Cvvdp::new_with_geometry ({w}x{h}): {e}"))?;
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

/// Faithful HDR linear-planes scoring on a GPU runtime — mirrors
/// [`score_pair_cached_gpu`] but feeds cvvdp's `score_from_linear_planes`
/// (display-relative `[0,1]` f32 planes) instead of sRGB8 bytes. Same
/// dim-keyed instance cache.
#[allow(clippy::too_many_arguments)]
fn score_pair_cached_linear_planes_gpu<R: Runtime>(
    state: &mut CvvdpBatchScorerState<R>,
    ref_r: &[f32],
    ref_g: &[f32],
    ref_b: &[f32],
    dis_r: &[f32],
    dis_g: &[f32],
    dis_b: &[f32],
    w: u32,
    h: u32,
) -> Result<f64, Box<dyn std::error::Error>> {
    let n = (w as usize) * (h as usize);
    for (name, p) in [
        ("ref_r", ref_r),
        ("ref_g", ref_g),
        ("ref_b", ref_b),
        ("dis_r", dis_r),
        ("dis_g", dis_g),
        ("dis_b", dis_b),
    ] {
        if p.len() != n {
            return Err(format!("plane {name} len {} != {w}x{h}={n}", p.len()).into());
        }
    }
    let needs_rebuild = !matches!(state.cached, Some((cw, ch, _)) if cw == w && ch == h);
    if needs_rebuild {
        state.cached = None;
        let c = cvvdp::Cvvdp::<R>::new_with_geometry(
            state.client.clone(),
            w,
            h,
            state.target.params(),
            state.target.geometry,
        )
        .map_err(|e| format!("Cvvdp::new_with_geometry ({w}x{h}): {e}"))?;
        state.cached = Some((w, h, c));
    }
    let c = &mut state.cached.as_mut().expect("just populated").2;
    // cvvdp-gpu's typed scorer assumes tight W×H planes (no padded_width arg);
    // the explicit length check above guarantees that.
    let jod = c
        .score_from_linear_planes(ref_r, ref_g, ref_b, dis_r, dis_g, dis_b)
        .map_err(|e| format!("Cvvdp::score_from_linear_planes: {e}"))? as f64;
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
        // Bake the display target's photometry + geometry into the
        // instance, mirroring the GPU path. The per-call `ppd` below
        // must match the construction-time geometry (debug builds
        // assert this in compute_dkl_*).
        let c = cvvdp::Cvvdp::<R>::new_with_geometry(
            state.client.clone(),
            w,
            h,
            state.target.params(),
            state.target.geometry,
        )
        .map_err(|e| format!("Cvvdp::new_with_geometry (cpu host_pool, {w}x{h}): {e}"))?;
        state.cached = Some((w, h, c));
    }
    let c = &mut state.cached.as_mut().expect("just populated").2;
    // PPD derived from the SAME geometry the instance was built with —
    // no longer hardcoded to STANDARD_4K (that bug pinned every CVVDP
    // score to 4K viewing geometry regardless of --display-model).
    let ppd = state.target.geometry.pixels_per_degree();
    let jod = c
        .compute_dkl_jod_host_pool(&reference.pixels, &distorted.pixels, ppd)
        .map_err(|e| format!("Cvvdp::compute_dkl_jod_host_pool: {e}"))?;
    if !jod.is_finite() {
        return Err(format!("cvvdp (cpu host_pool) produced non-finite JOD: {jod}").into());
    }
    Ok(f64::from(jod))
}

// The previous file shipped `score()` / `score_with_runtime()` /
// `run::<R>()` / `run_cpu_host_pool::<R>()` cascade for the
// single-shot cvvdp path. Those are now handled by the umbrella's
// `Metric::compute_srgb_u8` (see `mod.rs::run_gpu_via_umbrella`) —
// only the batch-scoring instance cache remains, since it depends
// on holding a typed `cvvdp::Cvvdp<R>` across pairs.
