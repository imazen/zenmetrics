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
use std::sync::{Mutex, OnceLock};

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
    /// `(pointer, length)` fingerprint of the reference image whose
    /// per-metric ref-side state is currently warm on device.
    /// `None` after construction; populated on the first successful
    /// `set_reference_srgb_u8` call.
    ///
    /// **Why pointer+length, not a content hash:** the sweep
    /// per-source loop borrows the decoded source for every cell in
    /// the same iteration; the `Vec<u8>` lives at the same address
    /// across all cells. Pointer identity is a 99.99%-precise cache
    /// key that costs ~10 ns vs ~7 ms for a 36 MB content hash. The
    /// 0.01% false-positive risk (two different sources happening to
    /// land at the same address after a drop+realloc) is bounded by
    /// the cache being invalidated on dim transitions and rebuilt
    /// on every dim change; the worst-case false positive is one
    /// metric's score being computed against the wrong ref, which
    /// the next dim transition flushes.
    ref_fingerprint: Option<(usize, usize)>,
    /// `true` when an earlier `set_reference_srgb_u8` returned an
    /// error (most commonly: butteraugli-gpu in strip mode rejects
    /// cached-ref). Once set, this slot stays on the one-shot
    /// `compute_srgb_u8` path for its lifetime. Re-evaluated on
    /// every dim transition (the slot is rebuilt fresh).
    set_reference_unsupported: bool,
}

#[cfg(any(
    feature = "gpu-butteraugli",
    feature = "gpu-ssim2",
    feature = "gpu-dssim",
    feature = "gpu-iwssim",
    feature = "gpu-zensim",
    feature = "gpu-cvvdp"
))]
static GLOBAL_CACHE: OnceLock<Mutex<MetricCache>> = OnceLock::new();

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

    /// Process-static metric cache. Initializes on first call with
    /// the supplied `gpu_runtime`; subsequent calls ignore the
    /// argument (returns the existing global). The cache outlives
    /// individual `run_sweep` calls so cached `Metric` instances are
    /// reused across:
    ///
    /// - Groups within a chunk (per the original cubecl pool OOM
    ///   diagnosis — repeated allocation of zensim WithIw persist
    ///   planes saturates the pool faster than the driver can free
    ///   them).
    /// - Chunks within a worker process (`zenfleet-sweep worker`'s
    ///   chunk loop calls `process_chunk_inline` per chunk; the
    ///   global cache means cross-chunk dim transitions are the
    ///   only allocation events).
    ///
    /// Safety: returns a `&'static Mutex<MetricCache>`; callers
    /// should use [`lock_global`] which handles mutex poisoning
    /// gracefully.
    pub(crate) fn global(gpu_runtime: GpuRuntime) -> &'static Mutex<MetricCache> {
        GLOBAL_CACHE.get_or_init(|| Mutex::new(MetricCache::new(gpu_runtime)))
    }

    /// Drop all cached `Metric` instances. Called between source
    /// images in the sweep outer loop (when explicitly requested via
    /// `SWEEP_CLEANUP_BETWEEN_SOURCES=1`) so the next source's
    /// per-metric persist planes are reallocated from a freshly
    /// dropped pool slot rather than constructed alongside the prior
    /// ones.
    ///
    /// 2026-05-22 fix: this used to also call cubecl's runtime-wide
    /// `memory_cleanup()` hint to flush dropped pages back to the
    /// underlying CUDA driver. That call panicked at
    /// `cubecl-cuda/src/compute/stream.rs:101` with "Memory page 0
    /// doesn't exist" — `memory_cleanup` invalidates pool pages that
    /// any other Binding (in this same or a concurrent worker) still
    /// references, and the next kernel call dereferences a now-stale
    /// handle. Dropping the cached `Metric` instances is sufficient:
    /// their handles return to the pool's free list and the next
    /// allocation reuses those exact pages, so footprint stays at
    /// one-instance-per-metric without the destructive global hint.
    ///
    /// Returns the number of slots evicted.
    pub(crate) fn cleanup_all(&mut self) -> usize {
        let n = self.umbrella.len();
        self.umbrella.clear();
        #[cfg(feature = "gpu-butteraugli")]
        {
            self.butter = None;
        }
        n
    }

    /// Acquire the global cache, recovering from a poisoned lock.
    ///
    /// A panic inside a cell while holding the cache lock — e.g.
    /// the cubecl-cuda OOM panic propagating up through
    /// `compute_srgb_u8` — poisons the Mutex. Without recovery,
    /// every subsequent cell on the same source image cascades
    /// through the same poisoned-lock panic and the whole chunk is
    /// lost. Recovery is sound because the cached `Metric` state
    /// either survived the panic (no mutation in flight) or is
    /// stale-but-safe (we rebuild on dim mismatch). Worst case the
    /// recovered cache holds a Metric whose internal device state
    /// is broken; the next compute call will fail loudly, which the
    /// caller already handles as a metric failure.
    pub(crate) fn lock_global(
        gpu_runtime: GpuRuntime,
    ) -> std::sync::MutexGuard<'static, MetricCache> {
        let mtx = Self::global(gpu_runtime);
        match mtx.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
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
            // `cvvdp-gpu` = GPU cvvdp (the prior bare-`Cvvdp` cache arm).
            // The unsuffixed `Cvvdp`/`Iwssim` are now the native-CPU ports
            // and have no device-pool pressure — they fall through to the
            // `_` arm below (uncached `run_metric` → `Backend::Cpu`).
            #[cfg(feature = "gpu-cvvdp")]
            MetricKind::CvvdpGpu => self
                .compute_umbrella(
                    zenmetrics_api::MetricKind::Cvvdp,
                    reference,
                    distorted,
                    None,
                )
                .map(|v| vec![(zenmetrics_api::cvvdp::CVVDP_COLUMN_NAME, v)]),
            #[cfg(feature = "gpu-zensim")]
            MetricKind::ZensimGpu => {
                // Sub-64px images are reflect(mirror)-padded to the 64px
                // 4-scale-pyramid floor on BOTH paths now — the GPU crate
                // (`ZensimOpaque`) and the CPU `zensim` use the same reflect-101
                // rule, so they agree to within f32-kernel drift (≤0.03 score).
                // We still route sub-64px to the CPU here: it's bit-exact with
                // the GPU at those sizes AND skips a wasteful 64×64 GPU pipeline
                // build + upload + 7 kernel launches for a handful of pixels.
                // The GPU is a throughput optimization for large images; tiny
                // images are cheaper on CPU. Column stays "zensim_gpu" so the
                // output schema is unchanged.
                if reference.width.min(reference.height) < 64 {
                    crate::metrics::zensim::score(reference, distorted)
                        .map(|v| vec![("zensim_gpu", v)])
                } else {
                    self.compute_umbrella(
                        zenmetrics_api::MetricKind::Zensim,
                        reference,
                        distorted,
                        // Score-only path. `None` => `build_params` builds the
                        // metric's DEFAULT params (`ZensimParams::default_weights()`):
                        // profile = `ZensimProfile::latest()` (= `A`, the v47-strict
                        // QAT bake) with its NATURAL regime, `WithIw` (372 features).
                        //
                        // This regime MUST match the profile's MLP input width.
                        // The shipped default bake is a 372-input MLP, so a
                        // narrower regime (Basic=228 / Extended=300) makes the
                        // forward pass fail with `ModelForwardFailed`, which
                        // `score_from_profile_vec` maps to `NaN` — every ≥64px
                        // cell then fails with "non-finite score NaN". An earlier
                        // version forced `Some(Basic)` here on the (false) premise
                        // that "the scalar score is regime-independent"; that is
                        // only true for the legacy linear V0_1/V0_2 weights path,
                        // not for the MLP profiles that have shipped since V0_3.
                        // The one-shot `score` subcommand never hit this because it
                        // builds params via `resolve_default_params` (= the matched
                        // 372/372 default) rather than overriding the regime.
                        //
                        // The wider regime's persist planes (~600 MB at 12 MP) are
                        // the cost of a correct score; the feature-emit path
                        // (`compute_zensim_features`) builds its own slot keyed on
                        // its requested regime, so this does not double-allocate
                        // when both score and features are produced for a sweep.
                        None,
                    )
                    .map(|v| vec![("zensim_gpu", v)])
                }
            }
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
        match slot
            .metric
            .compute_features_srgb_u8(&reference.pixels, &distorted.pixels)
        {
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
            Err(e) => Err(format!("zensim-gpu ({}): {e}", backend_label(slot.backend)).into()),
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
        let ref_fingerprint = (reference.pixels.as_ptr() as usize, reference.pixels.len());
        let slot =
            self.get_or_build_umbrella(umbrella_kind, reference.width, reference.height, regime)?;

        // Cached-ref fast path (Phase 2C). When the reference's
        // (pointer, len) fingerprint matches the slot's last
        // set_reference call, skip the ref upload and ref-side
        // pre-processing by calling compute_with_cached_reference.
        //
        // First call against a new source (or after a dim transition
        // rebuilt the slot): set_reference, then compute against
        // cache. set_reference failure (butter strip-mode rejects)
        // marks the slot as set_reference_unsupported and falls
        // through to one-shot compute_srgb_u8 for the slot's
        // lifetime.
        let use_cached_ref = !slot.set_reference_unsupported;
        let score_result = if use_cached_ref {
            if slot.ref_fingerprint != Some(ref_fingerprint) {
                match slot.metric.set_reference_srgb_u8(&reference.pixels) {
                    Ok(()) => slot.ref_fingerprint = Some(ref_fingerprint),
                    Err(_) => {
                        // Most likely: butter in strip mode rejects
                        // set_reference. Mark and fall through to
                        // one-shot for this slot's lifetime.
                        slot.set_reference_unsupported = true;
                        slot.ref_fingerprint = None;
                    }
                }
            }
            if slot.ref_fingerprint == Some(ref_fingerprint) {
                slot.metric
                    .compute_with_reference_srgb_u8(&distorted.pixels)
            } else {
                slot.metric
                    .compute_srgb_u8(&reference.pixels, &distorted.pixels)
            }
        } else {
            slot.metric
                .compute_srgb_u8(&reference.pixels, &distorted.pixels)
        };

        match score_result {
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
        // ENTRYPOINT PANIC — GPU zensim kernel DISABLED (2026-07-19). Both cache
        // paths (the ZensimGpu score arm + compute_zensim_features) construct the
        // GPU zensim metric HERE; zensim is CPU features-only now, so a GPU zensim
        // slot must never be built. Fail LOUD rather than run the stale v1 kernel.
        assert!(
            !matches!(kind, zenmetrics_api::MetricKind::Zensim),
            "GPU zensim is DISABLED (2026-07-19): cache tried to build a GPU zensim \
             slot — zensim is CPU features-only (use run_zensim_features), no GPU"
        );
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
            // than two during the transition. Dropping releases the
            // persist-plane Handles back to cubecl's pool.
            //
            // Capture the dropped slot's backend (if any) so we can
            // reclaim its pool BEFORE the new pipeline allocates. See
            // the correctness note below — leaving freed pages in the
            // pool for blind reuse miscomputes the next score.
            let prev_backend = self.umbrella.get(&kind).map(|s| s.backend);
            self.umbrella.remove(&kind);

            // CORRECTNESS (2026-06-22): reclaim the dropped pipeline's
            // pooled VRAM before building the new one. Without this, a
            // freed page from the previous-size pipeline can be handed
            // back to the new pipeline's `client.empty()` allocation
            // still holding the previous pipeline's data, and read
            // before the new pipeline overwrites it — a stale-page read
            // that silently corrupts the score. It reproduced as a
            // ~7-9 JOD divergence on `zenmetrics sweep --metric
            // zensim-gpu` at 1448×1448 ONLY when a differently-sized
            // pipeline (e.g. 1024²) had been scored and dropped first
            // (regression test:
            // `zensim-gpu/tests/it/cached_ref_slot_rebuild.rs`). The
            // same image scored correctly in isolation and via the
            // standalone `score` subcommand; only the in-process
            // rebuild path was affected. `reclaim_pooled_vram` returns
            // the freed pages to the driver so the next allocation gets
            // clean (zeroed) memory, which cures the divergence (probe
            // measured 9.10 → 0.04 JOD).
            //
            // This supersedes the 2026-05-22 "omit the hint entirely"
            // note: that finding was that `memory_cleanup` panics on a
            // FRESHLY-INITIALIZED runtime (`get_cursor` → None at
            // cubecl-cuda stream.rs). We only reclaim when a slot
            // actually existed (`prev_backend.is_some()`), i.e. the
            // runtime is warm and has live pool pages — never on a
            // fresh client. `reclaim_pooled_vram` (memory_cleanup +
            // sync) is the supported way to release dropped-instance
            // pages and is safe in this warm-rebuild position.
            if let Some(b) = prev_backend {
                zenmetrics_api::reclaim_pooled_vram(b);
            }

            let (metric, backend) =
                construct_umbrella(kind, width, height, regime, self.gpu_runtime)?;
            self.umbrella.insert(
                kind,
                UmbrellaSlot {
                    width,
                    height,
                    regime,
                    metric,
                    backend,
                    ref_fingerprint: None,
                    set_reference_unsupported: false,
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
                let zp =
                    zenmetrics_api::zensim::ZensimParams::default_weights().with_regime(r.into());
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
        // `Auto` is resolved by the umbrella before a metric is built;
        // it should not reach here, but the match must stay exhaustive.
        zenmetrics_api::Backend::Auto => "auto",
        zenmetrics_api::Backend::Cuda => "cuda",
        zenmetrics_api::Backend::Wgpu => "wgpu",
        zenmetrics_api::Backend::Hip => "hip",
        // The umbrella renamed its old `Cpu` (cubecl-cpu reference path)
        // to `CubeclCpu`; the CLI's `GpuRuntime::Cpu` still maps onto it
        // and keeps the historical "cpu" label.
        zenmetrics_api::Backend::CubeclCpu => "cpu",
        // The fast native-CPU backend (task #159 phase 2: fast-ssim2 / zensim /
        // butteraugli / dssim / in-tree cvvdp+iwssim). Distinct label so error
        // and status lines don't conflate it with the slow cubecl-cpu path.
        zenmetrics_api::Backend::Cpu => "cpu-native",
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
            return Err(format!("butteraugli-gpu produced non-finite max-norm: {max}").into());
        }
        if !pnorm3.is_finite() {
            return Err(format!("butteraugli-gpu produced non-finite pnorm_3: {pnorm3}").into());
        }
        Ok((max, pnorm3))
    }
}
