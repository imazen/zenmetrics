#![forbid(unsafe_code)]

//! Sweep-scoped metric instance cache.
//!
//! ## Why this exists
//!
//! Before this cache, every cell in a sweep called `Metric::new(...)`
//! → `compute_*` → drop for each enabled GPU metric. Each construction
//! allocated the per-metric persist planes (zensim WithIw regime
//! reserves ~200 MB per instance at 1080p; cvvdp similar; iwssim
//! similar). cubecl-cuda's pool does not promptly release the freed
//! buffers — the underlying CUDA driver retains them in its async
//! free list and they accumulate. The Rust object is dropped but the
//! GPU memory footprint keeps climbing until the next chunk OOMs on
//! a 12 GB card after ~80 cell-allocations.
//!
//! The fix is the same pattern [`crate::metrics::cvvdp_gpu::CvvdpBatchScorer`]
//! already uses: construct each [`zenmetrics_api::Metric`] **once per
//! (kind, dims, regime)** and reuse across cells. The cache also
//! holds a typed `butteraugli_gpu::Butteraugli<R>` for the two-column
//! emit path (pnorm3), so butter no longer pays per-cell construction.
//!
//! ## Scope and threading
//!
//! The cache is meant to live for the duration of one
//! [`crate::sweep::run_sweep`] call (one knob-group on the fleet).
//! Sharing across rayon worker threads is done through a `Mutex`
//! around the whole cache — GPU work is serialized by the driver
//! anyway, and the encode + decode-back phases happen *before* the
//! lock is taken, so CPU parallelism on the per-cell prep is
//! preserved. Lock contention is therefore bounded by GPU compute
//! time, which is the dominant cost per cell.
//!
//! ## Eviction on dim change
//!
//! When a source image with different `(width, height)` arrives, the
//! cache rebuilds the slot for that metric: drops the old `Metric`
//! (releasing its persist planes back to cubecl), then allocates a
//! new one. Sweeps over a homogeneous corpus (typical case) pay one
//! construction per metric per sweep call; mixed-dim corpora pay one
//! per dim transition.

#[cfg(any(
    feature = "gpu-butteraugli",
    feature = "gpu-ssim2",
    feature = "gpu-dssim",
    feature = "gpu-iwssim",
    feature = "gpu-zensim",
    feature = "gpu-cvvdp"
))]
use std::collections::HashMap;
#[cfg(any(
    feature = "gpu-butteraugli",
    feature = "gpu-ssim2",
    feature = "gpu-dssim",
    feature = "gpu-iwssim",
    feature = "gpu-zensim",
    feature = "gpu-cvvdp"
))]
use std::error::Error;

#[cfg(any(
    feature = "gpu-butteraugli",
    feature = "gpu-ssim2",
    feature = "gpu-dssim",
    feature = "gpu-iwssim",
    feature = "gpu-zensim",
    feature = "gpu-cvvdp"
))]
use crate::decode::Rgb8Image;
#[cfg(any(
    feature = "gpu-butteraugli",
    feature = "gpu-ssim2",
    feature = "gpu-dssim",
    feature = "gpu-iwssim",
    feature = "gpu-zensim",
    feature = "gpu-cvvdp"
))]
use crate::metrics::{
    GpuRuntime, MetricKind, ZensimFeatureRegime, auto_order, gpu_runtime_to_backend,
    resolve_default_params, runtime_label,
};

/// Cached GPU metric instances keyed by ([`MetricKind`], dims).
///
/// One slot per metric kind. The slot's `(width, height)` is recorded
/// on first construction; a subsequent score request for a different
/// `(width, height)` evicts and rebuilds it. The cache stores the
/// resolved [`zenmetrics_api::Backend`] alongside the [`Metric`] so
/// auto-runtime fall-through is computed once and not re-evaluated
/// per cell.
#[cfg(any(
    feature = "gpu-butteraugli",
    feature = "gpu-ssim2",
    feature = "gpu-dssim",
    feature = "gpu-iwssim",
    feature = "gpu-zensim",
    feature = "gpu-cvvdp"
))]
pub(crate) struct MetricCache {
    /// Selected GPU runtime. `Auto` is expanded on first use against
    /// [`auto_order`] and the resolved backend is then locked in for
    /// the rest of the cache's life.
    gpu_runtime: GpuRuntime,
    /// Cached zenmetrics-api umbrella `Metric` instances. One slot
    /// per metric kind; rebuilt on dim change.
    umbrella: HashMap<zenmetrics_api::MetricKind, UmbrellaSlot>,
    /// Butteraugli GPU two-column emit (`butteraugli_max_gpu` +
    /// `butteraugli_pnorm3_gpu`) is reached via the typed surface
    /// because the opaque `Score` only carries the max-norm. The
    /// cache holds a typed scorer per runtime variant, parallel to
    /// [`crate::metrics::cvvdp_gpu::CvvdpBatchScorer`].
    #[cfg(feature = "gpu-butteraugli")]
    butter: Option<butter::ButterBatchScorer>,
}

#[cfg(any(
    feature = "gpu-butteraugli",
    feature = "gpu-ssim2",
    feature = "gpu-dssim",
    feature = "gpu-iwssim",
    feature = "gpu-zensim",
    feature = "gpu-cvvdp"
))]
struct UmbrellaSlot {
    width: u32,
    height: u32,
    /// For zensim, the params used at construction time encode the
    /// regime; if a later request needs a different regime we treat
    /// it as a cache miss the same way a dim change would.
    regime: Option<ZensimFeatureRegime>,
    metric: zenmetrics_api::Metric,
    backend: zenmetrics_api::Backend,
}

#[cfg(any(
    feature = "gpu-butteraugli",
    feature = "gpu-ssim2",
    feature = "gpu-dssim",
    feature = "gpu-iwssim",
    feature = "gpu-zensim",
    feature = "gpu-cvvdp"
))]
impl MetricCache {
    pub(crate) fn new(gpu_runtime: GpuRuntime) -> Self {
        Self {
            gpu_runtime,
            umbrella: HashMap::new(),
            #[cfg(feature = "gpu-butteraugli")]
            butter: None,
        }
    }

    /// Score one GPU metric on `(reference, distorted)`. Returns the
    /// per-metric columns in the same shape [`crate::metrics::run_metric`]
    /// produces, so callers can splice the result directly into the
    /// sweep TSV row.
    ///
    /// CPU metrics are NOT routed through this cache — they have no
    /// device-pool pressure and the umbrella does not handle them.
    pub(crate) fn run_metric_cached(
        &mut self,
        kind: MetricKind,
        reference: &Rgb8Image,
        distorted: &Rgb8Image,
    ) -> Result<Vec<(&'static str, f64)>, Box<dyn Error>> {
        match kind {
            #[cfg(feature = "gpu-ssim2")]
            MetricKind::Ssim2Gpu => self
                .compute_umbrella(
                    zenmetrics_api::MetricKind::Ssim2,
                    reference,
                    distorted,
                    None,
                )
                .map(|v| vec![("ssim2_gpu", v)]),
            #[cfg(feature = "gpu-dssim")]
            MetricKind::DssimGpu => self
                .compute_umbrella(
                    zenmetrics_api::MetricKind::Dssim,
                    reference,
                    distorted,
                    None,
                )
                .map(|v| vec![("dssim_gpu", v)]),
            #[cfg(feature = "gpu-iwssim")]
            MetricKind::IwssimGpu => self
                .compute_umbrella(
                    zenmetrics_api::MetricKind::Iwssim,
                    reference,
                    distorted,
                    None,
                )
                .map(|v| vec![("iwssim_gpu", v)]),
            #[cfg(feature = "gpu-iwssim")]
            MetricKind::Iwssim => self
                .compute_umbrella(
                    zenmetrics_api::MetricKind::Iwssim,
                    reference,
                    distorted,
                    None,
                )
                .map(|v| vec![(zenmetrics_api::iwssim::IWSSIM_COLUMN_NAME, v)]),
            #[cfg(feature = "gpu-cvvdp")]
            MetricKind::Cvvdp => self
                .compute_umbrella(
                    zenmetrics_api::MetricKind::Cvvdp,
                    reference,
                    distorted,
                    None,
                )
                .map(|v| vec![(zenmetrics_api::cvvdp::CVVDP_COLUMN_NAME, v)]),
            #[cfg(feature = "gpu-zensim")]
            MetricKind::ZensimGpu => self
                .compute_umbrella(
                    zenmetrics_api::MetricKind::Zensim,
                    reference,
                    distorted,
                    // Score-only path — the runner will request features
                    // separately via `compute_zensim_features` when it
                    // wants the regime-appropriate vector. Use Basic
                    // here for the cheapest persist planes that still
                    // produce the correct umbrella score (zensim's
                    // basic-block score is regime-independent — the
                    // extended / iw features feed picker training, not
                    // the scalar score).
                    Some(ZensimFeatureRegime::Basic),
                )
                .map(|v| vec![("zensim_gpu", v)]),
            #[cfg(feature = "gpu-butteraugli")]
            MetricKind::ButteraugliGpu => {
                let (max, pnorm3) = self.compute_butter(reference, distorted)?;
                Ok(vec![
                    ("butteraugli_max_gpu", max),
                    ("butteraugli_pnorm3_gpu", pnorm3),
                ])
            }
            // Unknown / disabled GPU metrics fall back to the
            // (uncached) per-call path so behaviour matches the
            // existing semantics. CPU metrics also land here.
            _ => crate::metrics::run_metric(kind, reference, distorted, self.gpu_runtime),
        }
    }

    /// Run **GPU** zensim with the regime-appropriate feature vector.
    /// Mirrors [`crate::metrics::run_zensim_gpu_with_features`] but
    /// reuses the cached `Metric` instance.
    #[cfg(feature = "gpu-zensim")]
    pub(crate) fn compute_zensim_features(
        &mut self,
        reference: &Rgb8Image,
        distorted: &Rgb8Image,
        regime: ZensimFeatureRegime,
    ) -> Result<(f64, Vec<f64>), Box<dyn Error>> {
        if reference.width != distorted.width || reference.height != distorted.height {
            return Err(format!(
                "zensim-gpu: reference ({}×{}) and distorted ({}×{}) differ in size",
                reference.width, reference.height, distorted.width, distorted.height
            )
            .into());
        }
        let slot = self.get_or_build_umbrella(
            zenmetrics_api::MetricKind::Zensim,
            reference.width,
            reference.height,
            Some(regime),
        )?;
        match slot.metric.compute_features_srgb_u8(
            &reference.pixels,
            &distorted.pixels,
        ) {
            Ok((score, features)) => {
                if !score.value.is_finite() {
                    return Err(format!(
                        "zensim-gpu ({}): non-finite score {}",
                        backend_label(slot.backend),
                        score.value
                    )
                    .into());
                }
                if features.len() != regime.total_features() {
                    return Err(format!(
                        "zensim-gpu ({}): expected {} features, got {}",
                        backend_label(slot.backend),
                        regime.total_features(),
                        features.len()
                    )
                    .into());
                }
                Ok((score.value, features))
            }
            Err(e) => Err(format!(
                "zensim-gpu ({}): {e}",
                backend_label(slot.backend)
            )
            .into()),
        }
    }

    fn compute_umbrella(
        &mut self,
        umbrella_kind: zenmetrics_api::MetricKind,
        reference: &Rgb8Image,
        distorted: &Rgb8Image,
        regime: Option<ZensimFeatureRegime>,
    ) -> Result<f64, Box<dyn Error>> {
        if reference.width != distorted.width || reference.height != distorted.height {
            return Err(format!(
                "{}: reference ({}×{}) and distorted ({}×{}) differ in size",
                umbrella_kind.tag(),
                reference.width,
                reference.height,
                distorted.width,
                distorted.height
            )
            .into());
        }
        let slot = self.get_or_build_umbrella(
            umbrella_kind,
            reference.width,
            reference.height,
            regime,
        )?;
        match slot
            .metric
            .compute_srgb_u8(&reference.pixels, &distorted.pixels)
        {
            Ok(score) => {
                if !score.value.is_finite() {
                    return Err(format!(
                        "{} ({}): non-finite score {}",
                        umbrella_kind.tag(),
                        backend_label(slot.backend),
                        score.value
                    )
                    .into());
                }
                Ok(score.value)
            }
            Err(e) => Err(format!(
                "{} ({}): {e}",
                umbrella_kind.tag(),
                backend_label(slot.backend)
            )
            .into()),
        }
    }

    fn get_or_build_umbrella(
        &mut self,
        kind: zenmetrics_api::MetricKind,
        width: u32,
        height: u32,
        regime: Option<ZensimFeatureRegime>,
    ) -> Result<&mut UmbrellaSlot, Box<dyn Error>> {
        // Cache key matches on (kind, dims, regime). Different regime
        // for the same kind needs a different `Metric` because the
        // persist-plane footprint differs.
        let need_rebuild = match self.umbrella.get(&kind) {
            Some(s) => s.width != width || s.height != height || s.regime != regime,
            None => true,
        };
        if need_rebuild {
            // Drop the prior slot before allocating the new one so
            // peak GPU memory stays at one instance's worth rather
            // than two during the transition.
            self.umbrella.remove(&kind);
            let (metric, backend) = construct_umbrella(
                kind,
                width,
                height,
                regime,
                self.gpu_runtime,
            )?;
            self.umbrella.insert(
                kind,
                UmbrellaSlot {
                    width,
                    height,
                    regime,
                    metric,
                    backend,
                },
            );
        }
        Ok(self.umbrella.get_mut(&kind).expect("just populated"))
    }

    #[cfg(feature = "gpu-butteraugli")]
    fn compute_butter(
        &mut self,
        reference: &Rgb8Image,
        distorted: &Rgb8Image,
    ) -> Result<(f64, f64), Box<dyn Error>> {
        if reference.width != distorted.width || reference.height != distorted.height {
            return Err(format!(
                "butteraugli-gpu: reference ({}×{}) and distorted ({}×{}) differ in size",
                reference.width, reference.height, distorted.width, distorted.height
            )
            .into());
        }
        if self.butter.is_none() {
            self.butter = Some(butter::ButterBatchScorer::new(self.gpu_runtime)?);
        }
        let b = self.butter.as_mut().expect("just populated");
        b.score(reference, distorted)
    }
}

/// Construct one umbrella `Metric` instance for the given
/// `(kind, dims, regime)`. Walks the [`auto_order`] runtime cascade
/// when `gpu_runtime` is `Auto`; returns on the first runtime that
/// successfully allocates.
#[cfg(any(
    feature = "gpu-butteraugli",
    feature = "gpu-ssim2",
    feature = "gpu-dssim",
    feature = "gpu-iwssim",
    feature = "gpu-zensim",
    feature = "gpu-cvvdp"
))]
fn construct_umbrella(
    kind: zenmetrics_api::MetricKind,
    width: u32,
    height: u32,
    regime: Option<ZensimFeatureRegime>,
    gpu_runtime: GpuRuntime,
) -> Result<(zenmetrics_api::Metric, zenmetrics_api::Backend), Box<dyn Error>> {
    let candidates: Vec<GpuRuntime> = match gpu_runtime {
        GpuRuntime::Auto => auto_order().to_vec(),
        other => vec![other],
    };
    let mut errors: Vec<String> = Vec::with_capacity(candidates.len());
    for rt in candidates {
        let backend = match gpu_runtime_to_backend(rt) {
            Ok(b) => b,
            Err(e) => {
                errors.push(format!("{}: {e}", runtime_label(rt)));
                continue;
            }
        };
        let params = match build_params(kind, regime) {
            Ok(p) => p,
            Err(e) => {
                errors.push(format!("{}: {e}", runtime_label(rt)));
                continue;
            }
        };
        match zenmetrics_api::Metric::new(kind, backend, width, height, params) {
            Ok(metric) => return Ok((metric, backend)),
            Err(e) => errors.push(format!("{}: {e}", runtime_label(rt))),
        }
    }
    Err(format!(
        "{}: no runtime succeeded; tried [{}]",
        kind.tag(),
        if errors.is_empty() {
            "none".to_string()
        } else {
            errors.join("; ")
        }
    )
    .into())
}

/// Build the per-metric [`zenmetrics_api::MetricParams`] for a kind,
/// honouring the optional zensim regime override and the
/// `--allow-small-images` iwssim flag (the latter via
/// [`resolve_default_params`]).
#[cfg(any(
    feature = "gpu-butteraugli",
    feature = "gpu-ssim2",
    feature = "gpu-dssim",
    feature = "gpu-iwssim",
    feature = "gpu-zensim",
    feature = "gpu-cvvdp"
))]
fn build_params(
    kind: zenmetrics_api::MetricKind,
    regime: Option<ZensimFeatureRegime>,
) -> Result<zenmetrics_api::MetricParams, zenmetrics_api::Error> {
    #[cfg(feature = "gpu-zensim")]
    {
        if matches!(kind, zenmetrics_api::MetricKind::Zensim) {
            if let Some(r) = regime {
                let zp = zenmetrics_api::zensim::ZensimParams::default_weights()
                    .with_regime(r.into());
                return Ok(zenmetrics_api::MetricParams::Zensim(zp));
            }
        }
    }
    let _ = regime; // unused when zensim feature off
    resolve_default_params(kind)
}

#[cfg(any(
    feature = "gpu-butteraugli",
    feature = "gpu-ssim2",
    feature = "gpu-dssim",
    feature = "gpu-iwssim",
    feature = "gpu-zensim",
    feature = "gpu-cvvdp"
))]
fn backend_label(b: zenmetrics_api::Backend) -> &'static str {
    match b {
        zenmetrics_api::Backend::Cuda => "cuda",
        zenmetrics_api::Backend::Wgpu => "wgpu",
        zenmetrics_api::Backend::Hip => "hip",
        zenmetrics_api::Backend::Cpu => "cpu",
    }
}

// ----------------------------------------------------------------
// Butteraugli typed batch scorer — parallel to
// `cvvdp_gpu::CvvdpBatchScorer` for the two-column emit path.
// ----------------------------------------------------------------

#[cfg(feature = "gpu-butteraugli")]
mod butter {
    use std::error::Error;

    use cubecl::Runtime;
    use zenmetrics_api::butter;

    use crate::decode::Rgb8Image;
    use crate::metrics::{GpuRuntime, auto_order, runtime_label};

    /// Caches one `butteraugli_gpu::Butteraugli<R>` instance per
    /// dim slot. Mirrors [`super::super::cvvdp_gpu::CvvdpBatchScorer`].
    pub(super) enum ButterBatchScorer {
        #[cfg(feature = "gpu-cuda")]
        Cuda(State<cubecl::cuda::CudaRuntime>),
        #[cfg(feature = "gpu-wgpu")]
        Wgpu(State<cubecl::wgpu::WgpuRuntime>),
        #[cfg(feature = "gpu-hip")]
        Hip(State<cubecl::hip::HipRuntime>),
        #[cfg(feature = "gpu-cpu")]
        Cpu(State<cubecl::cpu::CpuRuntime>),
    }

    pub(super) struct State<R: Runtime> {
        client: cubecl::client::ComputeClient<R>,
        cached: Option<(u32, u32, butter::Butteraugli<R>)>,
    }

    impl ButterBatchScorer {
        pub(super) fn new(runtime: GpuRuntime) -> Result<Self, Box<dyn Error>> {
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
                "ButterBatchScorer::new: no runtime succeeded; last error: {}",
                last_error.unwrap_or_else(|| "none".into())
            )
            .into())
        }

        fn try_new_with_runtime(runtime: GpuRuntime) -> Result<Self, Box<dyn Error>> {
            match runtime {
                #[cfg(feature = "gpu-cuda")]
                GpuRuntime::Cuda => Ok(Self::Cuda(State {
                    client: <cubecl::cuda::CudaRuntime as Runtime>::client(&Default::default()),
                    cached: None,
                })),
                #[cfg(not(feature = "gpu-cuda"))]
                GpuRuntime::Cuda => {
                    Err("cuda runtime not compiled in (rebuild with `--features gpu-cuda`)".into())
                }
                #[cfg(feature = "gpu-wgpu")]
                GpuRuntime::Wgpu => Ok(Self::Wgpu(State {
                    client: <cubecl::wgpu::WgpuRuntime as Runtime>::client(&Default::default()),
                    cached: None,
                })),
                #[cfg(not(feature = "gpu-wgpu"))]
                GpuRuntime::Wgpu => {
                    Err("wgpu runtime not compiled in (rebuild with `--features gpu-wgpu`)".into())
                }
                #[cfg(feature = "gpu-hip")]
                GpuRuntime::Hip => Ok(Self::Hip(State {
                    client: <cubecl::hip::HipRuntime as Runtime>::client(&Default::default()),
                    cached: None,
                })),
                #[cfg(not(feature = "gpu-hip"))]
                GpuRuntime::Hip => {
                    Err("hip runtime not compiled in (rebuild with `--features gpu-hip`)".into())
                }
                #[cfg(feature = "gpu-cpu")]
                GpuRuntime::Cpu => Ok(Self::Cpu(State {
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

        pub(super) fn score(
            &mut self,
            reference: &Rgb8Image,
            distorted: &Rgb8Image,
        ) -> Result<(f64, f64), Box<dyn Error>> {
            match self {
                #[cfg(feature = "gpu-cuda")]
                Self::Cuda(state) => score_cached(state, reference, distorted),
                #[cfg(feature = "gpu-wgpu")]
                Self::Wgpu(state) => score_cached(state, reference, distorted),
                #[cfg(feature = "gpu-hip")]
                Self::Hip(state) => score_cached(state, reference, distorted),
                #[cfg(feature = "gpu-cpu")]
                Self::Cpu(state) => score_cached(state, reference, distorted),
                #[cfg(not(any(
                    feature = "gpu-cuda",
                    feature = "gpu-wgpu",
                    feature = "gpu-hip",
                    feature = "gpu-cpu",
                )))]
                _ => {
                    let _ = (reference, distorted);
                    Err("no CubeCL runtime feature enabled at build time".into())
                }
            }
        }
    }

    fn score_cached<R: Runtime>(
        state: &mut State<R>,
        reference: &Rgb8Image,
        distorted: &Rgb8Image,
    ) -> Result<(f64, f64), Box<dyn Error>> {
        let (w, h) = (reference.width, reference.height);
        let needs_rebuild = !matches!(state.cached, Some((cw, ch, _)) if cw == w && ch == h);
        if needs_rebuild {
            // Drop before allocating to keep peak GPU memory at one
            // instance's worth across the transition.
            state.cached = None;
            let b = butter::Butteraugli::<R>::new_multires(state.client.clone(), w, h);
            state.cached = Some((w, h, b));
        }
        let b = &mut state.cached.as_mut().expect("just populated").2;
        let result = b
            .compute(&reference.pixels, &distorted.pixels)
            .map_err(|e| format!("butteraugli-gpu: {e}"))?;
        let max = result.score as f64;
        let pnorm3 = result.pnorm_3 as f64;
        if !max.is_finite() {
            return Err(
                format!("butteraugli-gpu produced non-finite max-norm: {max}").into(),
            );
        }
        if !pnorm3.is_finite() {
            return Err(
                format!("butteraugli-gpu produced non-finite pnorm_3: {pnorm3}").into(),
            );
        }
        Ok((max, pnorm3))
    }
}
