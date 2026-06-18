//! Phase 6 — CPU backend adapter.
//!
//! Sits one layer above the per-crate CPU reference implementations,
//! exposing the same `compute_srgb_u8` shape the GPU `ExecMetric`
//! already uses. The orchestrator's OOM-fallback ladder swaps a GPU
//! backend for a CPU one without changing the call site.
//!
//! ## Per-metric mapping (see `docs/CPU_BACKENDS.md`)
//!
//! | Metric  | CPU reference crate     | Feature flag   |
//! |---------|-------------------------|----------------|
//! | Cvvdp   | `cvvdp` (in-tree)       | `cpu-cvvdp`    |
//! | Ssim2   | `fast-ssim2` (Imazen)   | `cpu-ssim2`    |
//! | Dssim   | `dssim-core`            | `cpu-dssim`    |
//! | Butter  | `butteraugli`           | `cpu-butter`   |
//! | Zensim  | `zensim`                | `cpu-zensim`   |
//! | Iwssim  | `iwssim` (in-tree, 8g)  | `cpu-iwssim`   |
//!
//! Phase 8h (2026-05-27): the ssim2 row was switched from upstream
//! `ssimulacra2 0.5` to Imazen's SIMD-accelerated `fast-ssim2 0.8`.
//! Per-call scores may shift by atomic-add tolerance vs. the prior
//! implementation; the input shape is unchanged (sRGB u8 × 3-channel)
//! and the call surface (`compute` / `set_reference` / `compute_with_cached_reference`)
//! is untouched. See `docs/CPU_BACKENDS.md` for the rationale.
//!
//! Phase 8g (2026-05-27): added `iwssim` — a pure-Rust CPU port of the
//! canonical Python-IW-SSIM reference (Wang & Li 2011) with magetypes
//! SIMD on the SSIM-stats hot loops. The historical `Unavailable`
//! arm for iwssim is retained for build configurations that omit the
//! `cpu-iwssim` feature.
//!
//! ## Cached-reference semantics
//!
//! Each CPU backend has a different relationship with reference reuse:
//!
//! - **cvvdp** has a true cached-reference path (`warm_reference` +
//!   `score_with_warm_ref`). Skips ~50% of the pipeline.
//! - **butteraugli** (Phase 9.Y, 2026-05-27): now wired to the
//!   `ButteraugliReference::new(&[u8], …) + .compare(&[u8])` precompute
//!   API. The ref-side sRGB→linear→XYB→frequency-separated→mask path
//!   is built once and reused across compare calls. Replaces the prior
//!   byte-stash wiring that recomputed `full` on every warm-ref call.
//! - **dssim-core** lets you `create_image(reference)` once and reuse;
//!   the adapter caches the prepared `DssimImage`.
//! - **fast-ssim2** has a true cached-ref path (`Ssimulacra2Reference::new`
//!   + `compare`) that skips ~50 % of the pipeline. The adapter wires it
//!   up so `set_reference` + `compute_with_cached_reference` are now
//!   amortised, not recompute. **Change vs. Phase 6's `ssimulacra2` 0.5
//!   wiring**: that crate had no precompute API and the adapter just
//!   stashed bytes for shape parity. fast-ssim2's `Ssimulacra2Reference`
//!   replaces that with a true warm path.
//! - **zensim** (task #134, 2026-05-28): the `Zensim::precompute_reference`
//!   + `compute_with_ref` pair is wired through `set_reference` +
//!   `compute_with_cached_reference`. The reference-side sRGB → linear → XYB
//!   conversion + multi-scale downscale pyramid runs once per source;
//!   subsequent compare calls reuse the cached `PrecomputedReference`.
//!   Measured speedup on the 7950X at 16 MP is ≈+12 % per amortized
//!   warm call (3-trial median, 10-distorteds-per-ref sweep — see
//!   `benchmarks/zensim_cached_ref_cpu_2026-05-28.meta`). The previous
//!   wiring stashed raw bytes and re-converted them on every call —
//!   `supports_cached_ref` returned `false`, so the orchestrator never
//!   picked the warm dispatch. The strip warm-ref variant now
//!   dispatches through `compute_with_ref_streaming_strips` so the
//!   warm dispatch carries into the memory-bounded strip mode too.
//!   (Note: the GPU cached_ref sweep at
//!   `benchmarks/zensim_cached_ref_2026-05-22.csv` measures 38–40 % on
//!   CUDA / wgpu — the GPU win is larger because it skips device
//!   uploads and ref-side kernel launches across the sweep, neither of
//!   which apply on the CPU path.)
//!
//! The pool's worker decides whether to dispatch through
//! `compute_with_cached_reference` based on a static feature query
//! ([`CpuAdapter::supports_cached_ref`]); backends without acceleration
//! still produce a correct score.
//!
//! ## Memory characteristics
//!
//! CPU backends use RAM, not VRAM. Resident-set growth depends on
//! image size:
//!
//! - cvvdp: ~5-7 bytes/pixel scratch (Weber pyramid + DKL planes
//!   + diffmap). 4096² = ~120 MiB.
//! - butteraugli: ~30-40 bytes/pixel internal (XYB working set + blur
//!   buffers). 4096² = ~600 MiB.
//! - dssim-core: ~40 bytes/pixel (multi-scale LAB pyramid).
//!   4096² = ~700 MiB.
//! - fast-ssim2: ~50 bytes/pixel (XYB + sub-band buffers; ~24 image-sized
//!   f32 planes plus the downscale pyramid per `fast_ssim2::MAX_IMAGE_PIXELS`
//!   docs). 4096² = ~850 MiB. fast-ssim2 caps inputs at 16384² to bound
//!   the working set.
//! - zensim: ~10-15 bytes/pixel (XYB working set + per-scale features).
//!   4096² = ~250 MiB.
//!
//! Phase 6 records these as `ram_mib` cells in the capability cache via
//! the bench runner's CPU extension (see `bench::run_impl_cpu`).

#![cfg(feature = "bench")]

use zenmetrics_api::{MetricKind, MetricParams, Score};

// ---------------------------------------------------------------------------
// Public adapter type
// ---------------------------------------------------------------------------

/// CPU adapter — one per metric, per (w, h) signature.
///
/// Internal state is feature-gated per metric. Construction selects the
/// concrete CPU implementation based on `metric`; if the matching feature
/// is disabled, [`Self::new`] returns
/// [`CpuAdapterError::FeatureNotEnabled`] so the caller can advance the
/// fallback ladder.
///
/// `pub(crate)` — only the orchestrator's executor + pool create these.
pub(crate) struct CpuAdapter {
    metric: MetricKind,
    width: u32,
    height: u32,
    state: CpuAdapterState,
}

/// Per-metric internal state. Each arm holds the heap-allocated CPU
/// scorer + any cached reference state.
#[allow(clippy::large_enum_variant)]
enum CpuAdapterState {
    #[cfg(feature = "cpu-cvvdp")]
    Cvvdp(Box<cvvdp::Cvvdp>),
    #[cfg(feature = "cpu-ssim2")]
    Ssim2(Ssim2State),
    #[cfg(feature = "cpu-dssim")]
    Dssim(DssimState),
    #[cfg(feature = "cpu-butter")]
    Butter(ButterState),
    #[cfg(feature = "cpu-zensim")]
    Zensim(ZensimState),
    #[cfg(feature = "cpu-iwssim")]
    Iwssim(Box<iwssim::Iwssim>),
    /// Built without ANY CPU backend feature, or built without the
    /// specific feature for `metric`. The compute path returns
    /// [`CpuAdapterError::FeatureNotEnabled`].
    #[allow(dead_code)]
    FeatureDisabled(MetricKind),
    /// Reserved for any future metric whose CPU backend isn't yet
    /// implemented. Phase 8g landed iwssim; this arm is currently
    /// unreachable for ordinary callers but kept for symmetry.
    #[allow(dead_code)]
    Unavailable(MetricKind),
}

// ---------------------------------------------------------------------------
// Per-metric state structs (only compiled when the feature is on)
// ---------------------------------------------------------------------------

#[cfg(feature = "cpu-ssim2")]
struct Ssim2State {
    width: usize,
    height: usize,
    /// Cached fast-ssim2 reference — `set_reference` builds the precomputed
    /// reference data (~50 % of the SSIMULACRA2 pipeline) once; subsequent
    /// `compute_with_cached_reference` calls reuse it. Phase 8h replaced
    /// the prior `Option<Vec<u8>>` byte-stash (used by the ssimulacra2 0.5
    /// fallback) with this true warm-state cache.
    cached_ref: Option<fast_ssim2::Ssimulacra2Reference>,
}

#[cfg(feature = "cpu-dssim")]
struct DssimState {
    width: usize,
    height: usize,
    dssim: dssim_core::Dssim,
    /// dssim-core builds a multi-scale internal representation once via
    /// `create_image` and reuses it across compares. Cache when
    /// `set_reference` fires.
    cached_ref: Option<dssim_core::DssimImage<f32>>,
}

#[cfg(feature = "cpu-butter")]
struct ButterState {
    width: usize,
    height: usize,
    params: butteraugli::ButteraugliParams,
    /// Phase 9.Y (2026-05-27): butteraugli 0.9.2 exposes a true
    /// `ButteraugliReference::new(&[u8], w, h, params)` precompute API
    /// with `.compare(&[u8]) -> ButteraugliResult`. The precompute path
    /// runs sRGB → linear → XYB → mask + frequency-separation on the
    /// reference once and reuses the result across compare calls — the
    /// dist-side pipeline still runs per call, but the ref-side half
    /// (≈30-50% of the per-pair work) is hoisted. Replaces the prior
    /// `Option<Vec<u8>>` byte-stash that just recomputed `full` on the
    /// warm path. The corresponding `supports_cached_ref()` arm is
    /// flipped from `false` to `true`.
    cached_ref: Option<butteraugli::ButteraugliReference>,
}

#[cfg(feature = "cpu-zensim")]
struct ZensimState {
    width: usize,
    height: usize,
    zensim: zensim::Zensim,
    /// Task #134 (2026-05-28): zensim's `Zensim::precompute_reference`
    /// returns a `PrecomputedReference` that owns the multi-scale XYB
    /// pyramid for the source image. We cache it here so subsequent
    /// `compute_with_ref` calls can skip the sRGB → linear → XYB
    /// conversion + downscale (≈+12 % per amortized warm call on the
    /// 7950X at 16 MP — see `benchmarks/zensim_cached_ref_cpu_2026-05-28.meta`
    /// for the measured 3-trial median). Replaces the prior
    /// `Option<Vec<u8>>` byte stash that recomputed the full cold path
    /// on every warm-ref call.
    cached_ref: Option<zensim::PrecomputedReference>,
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Adapter-level errors. Translated into the executor's
/// [`crate::executor::OrchestratorError`] at the boundary.
#[derive(Debug, Clone)]
#[allow(dead_code)] // some variants only fire when specific cpu-* features are on
pub(crate) enum CpuAdapterError {
    /// Build does not include the feature for this metric's CPU
    /// reference. e.g. `--features bench` without `cpu-cvvdp`.
    /// Ladder advances to the next backend.
    FeatureNotEnabled(MetricKind),
    /// Metric has no clean CPU reference upstream (Iwssim). Documented
    /// in `docs/CPU_BACKENDS.md`. Ladder advances.
    Unavailable(MetricKind),
    /// Construction or compute failed inside the CPU reference crate.
    /// `String` carries the rendered error from the crate's own
    /// `Display`. Not retryable — ladder treats this as a hard fault
    /// for this backend at this size.
    Failed(String),
    /// Input byte length doesn't match `width × height × 3`. Validation
    /// guard before passing the slice to the underlying crate (some
    /// of which panic on mismatch rather than returning an error).
    InvalidInputSize { expected: usize, got: usize },
}

impl std::fmt::Display for CpuAdapterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CpuAdapterError::FeatureNotEnabled(k) => write!(
                f,
                "cpu adapter: feature 'cpu-{}' not enabled in this build",
                k.tag()
            ),
            CpuAdapterError::Unavailable(k) => {
                write!(f, "cpu adapter: metric '{}' has no CPU reference", k.tag())
            }
            CpuAdapterError::Failed(msg) => write!(f, "cpu adapter: {msg}"),
            CpuAdapterError::InvalidInputSize { expected, got } => write!(
                f,
                "cpu adapter: invalid input slice (expected {expected} bytes, got {got})"
            ),
        }
    }
}

impl std::error::Error for CpuAdapterError {}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

impl CpuAdapter {
    /// Build a CPU adapter for `metric` at `width × height` with
    /// `params`. Returns `Err(FeatureNotEnabled)` when the matching
    /// `cpu-<metric>` feature is off in the current build.
    pub fn new(
        metric: MetricKind,
        width: u32,
        height: u32,
        params: &MetricParams,
    ) -> Result<Self, CpuAdapterError> {
        let state = match metric {
            MetricKind::Cvvdp => construct_cvvdp(width, height, params),
            MetricKind::Ssim2 => construct_ssim2(width, height, params),
            MetricKind::Dssim => construct_dssim(width, height, params),
            MetricKind::Butter => construct_butter(width, height, params),
            MetricKind::Zensim => construct_zensim(width, height, params),
            MetricKind::Iwssim => construct_iwssim(width, height, params),
        }?;
        Ok(Self {
            metric,
            width,
            height,
            state,
        })
    }

    /// Which metric this adapter scores.
    #[allow(dead_code)]
    pub fn metric(&self) -> MetricKind {
        self.metric
    }

    /// Width in pixels.
    #[allow(dead_code)]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height in pixels.
    #[allow(dead_code)]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Whether this backend has a true cached-reference fast path. When
    /// false, the worker pool's cached-ref dispatch still produces a
    /// correct score but pays the full per-call cost.
    #[allow(dead_code)] // only called when feature = "cuda" is on (via pool.rs)
    pub fn supports_cached_ref(&self) -> bool {
        match self.state {
            #[cfg(feature = "cpu-cvvdp")]
            CpuAdapterState::Cvvdp(_) => true,
            #[cfg(feature = "cpu-dssim")]
            CpuAdapterState::Dssim(_) => true,
            // Phase 8h: fast-ssim2's `Ssimulacra2Reference` precomputes
            // the source's XYB / sub-bands once and reuses them across
            // distorted candidates — true warm path. The prior
            // `ssimulacra2` 0.5 wiring returned false here because that
            // crate had no precompute API.
            #[cfg(feature = "cpu-ssim2")]
            CpuAdapterState::Ssim2(_) => true,
            // Phase 9.Y (2026-05-27): butteraugli 0.9.2's
            // `ButteraugliReference` precomputes the reference XYB +
            // masks once and reuses across `compare(dist_bytes)` calls.
            // Prior wiring stored raw bytes and recomputed `full` per
            // warm-ref call — equivalent but with no speedup. The
            // upgrade keeps the public API identical and turns
            // `set_reference` + `compute_with_cached_reference` into a
            // true warm path.
            #[cfg(feature = "cpu-butter")]
            CpuAdapterState::Butter(_) => true,
            // Task #134 (2026-05-28): zensim has a true cached-reference
            // path via `Zensim::precompute_reference` + `compute_with_ref`.
            // The reference XYB conversion + downscale pyramid is hoisted
            // out of the per-pair compare — measured +12 % per amortized
            // warm call on the 7950X at 16 MP. The prior wiring returned
            // `false` here even though `set_reference` stashed bytes, so
            // the orchestrator never selected the warm dispatch.
            #[cfg(feature = "cpu-zensim")]
            CpuAdapterState::Zensim(_) => true,
            // iwssim has a true warm path: `warm_reference` caches the
            // 5-level Laplacian pyramid + per-scale Gaussian bands; the
            // distorted-side pyramid still has to build on every call,
            // but the ref-side eigendecomposition (10×10 covariance,
            // 5 scales) is hoisted out of the inner loop. Task #136
            // (2026-05-28): `compute_with_cached_reference` routes
            // through `score_with_warm_ref_strip` so the cached-ref
            // entry carries the strip walker's -48 % peak heap win at
            // 16 / 40 MP; score parity is within the documented 1e-4
            // strip tolerance. See the dispatch arm in
            // `compute_with_cached_reference` for measurements.
            #[cfg(feature = "cpu-iwssim")]
            CpuAdapterState::Iwssim(_) => true,
            CpuAdapterState::FeatureDisabled(_) | CpuAdapterState::Unavailable(_) => false,
        }
    }

    /// One-shot compute: hand both buffers, get a score back.
    pub fn compute(
        &mut self,
        ref_bytes: &[u8],
        dist_bytes: &[u8],
    ) -> Result<Score, CpuAdapterError> {
        let expected = (self.width as usize) * (self.height as usize) * 3;
        if ref_bytes.len() != expected {
            return Err(CpuAdapterError::InvalidInputSize {
                expected,
                got: ref_bytes.len(),
            });
        }
        if dist_bytes.len() != expected {
            return Err(CpuAdapterError::InvalidInputSize {
                expected,
                got: dist_bytes.len(),
            });
        }
        match &mut self.state {
            #[cfg(feature = "cpu-cvvdp")]
            CpuAdapterState::Cvvdp(c) => compute_cvvdp(c, ref_bytes, dist_bytes),
            #[cfg(feature = "cpu-ssim2")]
            CpuAdapterState::Ssim2(s) => compute_ssim2(s, ref_bytes, dist_bytes),
            #[cfg(feature = "cpu-dssim")]
            CpuAdapterState::Dssim(s) => compute_dssim(s, ref_bytes, dist_bytes),
            #[cfg(feature = "cpu-butter")]
            CpuAdapterState::Butter(s) => compute_butter(s, ref_bytes, dist_bytes),
            #[cfg(feature = "cpu-zensim")]
            CpuAdapterState::Zensim(s) => compute_zensim(s, ref_bytes, dist_bytes),
            #[cfg(feature = "cpu-iwssim")]
            CpuAdapterState::Iwssim(c) => compute_iwssim(c, ref_bytes, dist_bytes),
            CpuAdapterState::FeatureDisabled(k) => Err(CpuAdapterError::FeatureNotEnabled(*k)),
            CpuAdapterState::Unavailable(k) => Err(CpuAdapterError::Unavailable(*k)),
        }
    }

    /// Install reference bytes for subsequent cached-ref calls. On
    /// backends without a true cached-ref API, this caches the bytes
    /// internally and `compute_with_cached_reference` recomputes from
    /// the cached buffer.
    #[allow(dead_code)] // only called when feature = "cuda" is on (via pool.rs)
    pub fn set_reference(&mut self, ref_bytes: &[u8]) -> Result<(), CpuAdapterError> {
        let expected = (self.width as usize) * (self.height as usize) * 3;
        if ref_bytes.len() != expected {
            return Err(CpuAdapterError::InvalidInputSize {
                expected,
                got: ref_bytes.len(),
            });
        }
        match &mut self.state {
            #[cfg(feature = "cpu-cvvdp")]
            CpuAdapterState::Cvvdp(c) => c
                .warm_reference(ref_bytes)
                .map_err(|e| CpuAdapterError::Failed(e.to_string())),
            #[cfg(feature = "cpu-ssim2")]
            CpuAdapterState::Ssim2(s) => {
                let img = ssim2_image_ref(ref_bytes, s.width, s.height);
                let precomputed = fast_ssim2::Ssimulacra2Reference::new(img).map_err(|e| {
                    CpuAdapterError::Failed(format!("fast-ssim2 Ssimulacra2Reference::new: {e}"))
                })?;
                s.cached_ref = Some(precomputed);
                Ok(())
            }
            #[cfg(feature = "cpu-dssim")]
            CpuAdapterState::Dssim(s) => {
                let img = make_dssim_image(&s.dssim, ref_bytes, s.width, s.height)?;
                s.cached_ref = Some(img);
                Ok(())
            }
            #[cfg(feature = "cpu-butter")]
            CpuAdapterState::Butter(s) => {
                // Phase 9.Y: build a `ButteraugliReference` from the
                // sRGB-u8 reference bytes once. The precompute runs:
                //   sRGB → linear → XYB → frequency-separated (LF/MF/HF/UHF)
                //   → reference mask. Roughly 30-50% of the per-pair
                // pipeline cost — the half/sub-res mirror builds in
                // parallel via rayon. Holds onto a BufferPool that the
                // compare path reuses.
                let pre = butteraugli::ButteraugliReference::new(
                    ref_bytes,
                    s.width,
                    s.height,
                    s.params.clone(),
                )
                .map_err(|e| {
                    CpuAdapterError::Failed(format!("butteraugli ButteraugliReference::new: {e:?}"))
                })?;
                s.cached_ref = Some(pre);
                Ok(())
            }
            #[cfg(feature = "cpu-zensim")]
            CpuAdapterState::Zensim(s) => {
                // Task #134 (2026-05-28): build the cached
                // `PrecomputedReference` once. The crate runs the source
                // sRGB → linear → XYB conversion + multi-scale pyramid
                // downscale internally; subsequent
                // `compute_with_cached_reference` calls dispatch through
                // `compute_with_ref` and skip both stages. Measured
                // +12 % per amortized warm call on the 7950X at 16 MP
                // (see `benchmarks/zensim_cached_ref_cpu_2026-05-28.meta`
                // for the 3-trial median wall data).
                let src: &[[u8; 3]] = bytemuck::cast_slice(ref_bytes);
                let ref_slice = zensim::RgbSlice::new(src, s.width, s.height);
                let precomputed = s.zensim.precompute_reference(&ref_slice).map_err(|e| {
                    CpuAdapterError::Failed(format!("zensim precompute_reference: {e:?}"))
                })?;
                s.cached_ref = Some(precomputed);
                Ok(())
            }
            #[cfg(feature = "cpu-iwssim")]
            CpuAdapterState::Iwssim(c) => c
                .warm_reference(ref_bytes)
                .map_err(|e| CpuAdapterError::Failed(e.to_string())),
            CpuAdapterState::FeatureDisabled(k) => Err(CpuAdapterError::FeatureNotEnabled(*k)),
            CpuAdapterState::Unavailable(k) => Err(CpuAdapterError::Unavailable(*k)),
        }
    }

    /// Strip-mode compute: walk image in horizontal slabs of
    /// `strip_height` rows + halo. Reduces peak heap for 40 MP+
    /// inputs on metrics that support strip dispatch.
    ///
    /// Phase 9.Z.B follow-on status (2026-05-28):
    /// - **iwssim**: real strip walker (see `iwssim::Iwssim::score_strip`).
    ///   Bit-identical-tolerance parity vs full at < 1e-4 abs JOD.
    /// - **ssim2**: real strip walker via fast-ssim2 0.8.1
    ///   `compute_ssimulacra2_strip` (96-row halo, ~24 × strip_h ×
    ///   width × 4 B peak).
    /// - **butter**: real strip walker via butteraugli 0.9.3
    ///   `butteraugli_strip` (64-row halo, 3.8x heap reduction at 40 MP).
    /// - **zensim**: real strip walker via zensim
    ///   `compute_streaming_strips` (per-strip ref + dist, ~125 MB at
    ///   80 MP).
    /// - **cvvdp**: API stub — delegates to `score()` (no memory win
    ///   yet; multi-day walker queued). Returns the same score.
    /// - **dssim**: not yet wired (no upstream strip API on dssim-core 3.4).
    ///
    /// Pass `strip_height = 0` to get the metric's default body size
    /// (256 rows for ssim2/butter/zensim; iwssim/cvvdp keep their own
    /// per-crate defaults).
    #[allow(dead_code)] // only called via the strip-aware executor path
    #[allow(unused_variables)] // strip_height is unused only when no cpu-* backend feature is enabled
    pub fn compute_strip(
        &mut self,
        ref_bytes: &[u8],
        dist_bytes: &[u8],
        strip_height: u32,
    ) -> Result<Score, CpuAdapterError> {
        let expected = (self.width as usize) * (self.height as usize) * 3;
        if ref_bytes.len() != expected {
            return Err(CpuAdapterError::InvalidInputSize {
                expected,
                got: ref_bytes.len(),
            });
        }
        if dist_bytes.len() != expected {
            return Err(CpuAdapterError::InvalidInputSize {
                expected,
                got: dist_bytes.len(),
            });
        }
        match &mut self.state {
            #[cfg(feature = "cpu-cvvdp")]
            CpuAdapterState::Cvvdp(c) => {
                let h = if strip_height == 0 { 512 } else { strip_height };
                let v = c
                    .score_strip(ref_bytes, dist_bytes, h)
                    .map_err(|e| CpuAdapterError::Failed(e.to_string()))?;
                Ok(make_score("cvvdp", cvvdp_cpu_version(), v as f64))
            }
            #[cfg(feature = "cpu-iwssim")]
            CpuAdapterState::Iwssim(c) => {
                let h = if strip_height == 0 {
                    iwssim::STRIP_BODY_DEFAULT
                } else {
                    strip_height
                };
                let result = c
                    .score_strip(ref_bytes, dist_bytes, h)
                    .map_err(|e| CpuAdapterError::Failed(e.to_string()))?;
                Ok(make_score("iwssim", iwssim_cpu_version(), result.score))
            }
            #[cfg(feature = "cpu-ssim2")]
            CpuAdapterState::Ssim2(s) => {
                // fast-ssim2 0.8.1 `compute_ssimulacra2_strip` walks the
                // image in horizontal strips with a 96-row halo. Bounds
                // peak heap to ~24 × strip_h × width × 4 B. Default
                // strip_height of 256 matches the documented sweet spot
                // (220 MiB at 7700x5200).
                let h = if strip_height == 0 { 256 } else { strip_height };
                let ref_img = ssim2_image_ref(ref_bytes, s.width, s.height);
                let dist_img = ssim2_image_ref(dist_bytes, s.width, s.height);
                let v =
                    fast_ssim2::compute_ssimulacra2_strip(ref_img, dist_img, h).map_err(|e| {
                        CpuAdapterError::Failed(format!(
                            "fast-ssim2 compute_ssimulacra2_strip: {e}"
                        ))
                    })?;
                Ok(make_score("ssim2", env!("CARGO_PKG_VERSION"), v))
            }
            #[cfg(feature = "cpu-dssim")]
            CpuAdapterState::Dssim(_) => Err(CpuAdapterError::Failed(
                "dssim strip-mode not yet wired in cpu_adapter".into(),
            )),
            #[cfg(feature = "cpu-butter")]
            CpuAdapterState::Butter(s) => {
                // butteraugli 0.9.3 `butteraugli_strip` walks the dist
                // side in horizontal strips with a 64-row halo (FIR
                // chained blur stack). Peak heap drops 3.8x at 40 MP
                // (7.43 GB -> 1.94 GB) at equivalent wall time. Default
                // strip_height of 256.
                use imgref::ImgRef;
                let h = if strip_height == 0 { 256 } else { strip_height };
                let ref_rgb: &[rgb::RGB<u8>] = bytemuck::cast_slice(ref_bytes);
                let dist_rgb: &[rgb::RGB<u8>] = bytemuck::cast_slice(dist_bytes);
                let ref_img = ImgRef::new(ref_rgb, s.width, s.height);
                let dist_img = ImgRef::new(dist_rgb, s.width, s.height);
                let result = butteraugli::butteraugli_strip(ref_img, dist_img, &s.params, h)
                    .map_err(|e| CpuAdapterError::Failed(format!("butteraugli_strip: {e:?}")))?;
                Ok(make_score(
                    "butter",
                    env!("CARGO_PKG_VERSION"),
                    result.score,
                ))
            }
            #[cfg(feature = "cpu-zensim")]
            CpuAdapterState::Zensim(s) => {
                // zensim `compute_streaming_strips` decomposes the
                // multiscale pyramid into horizontal strips with
                // `strip_inner` body + 2*`strip_margin` halo per side.
                // Best for one-off pairs (the warm-ref variant lives
                // on `compute_with_cached_reference_strip`). The crate
                // default geometry is (256, 128).
                let inner = if strip_height == 0 {
                    256
                } else {
                    strip_height as usize
                };
                let margin = inner / 2; // matches the documented default ratio
                let src: &[[u8; 3]] = bytemuck::cast_slice(ref_bytes);
                let dst: &[[u8; 3]] = bytemuck::cast_slice(dist_bytes);
                let ref_slice = zensim::RgbSlice::new(src, s.width, s.height);
                let dist_slice = zensim::RgbSlice::new(dst, s.width, s.height);
                let result = s
                    .zensim
                    .compute_streaming_strips(&ref_slice, &dist_slice, inner, margin)
                    .map_err(|e| {
                        CpuAdapterError::Failed(format!("zensim compute_streaming_strips: {e:?}"))
                    })?;
                Ok(make_score(
                    "zensim",
                    env!("CARGO_PKG_VERSION"),
                    result.score(),
                ))
            }
            CpuAdapterState::FeatureDisabled(k) => Err(CpuAdapterError::FeatureNotEnabled(*k)),
            CpuAdapterState::Unavailable(k) => Err(CpuAdapterError::Unavailable(*k)),
        }
    }

    /// Strip-mode compute against the cached reference. See
    /// [`Self::compute_strip`] for per-metric implementation status.
    ///
    /// Phase 9.Z.B follow-on status (2026-05-28):
    /// - **iwssim**: real warm_ref_strip walker (eigendecomposition
    ///   cached in `WarmState`; ref state full + one strip dist
    ///   working set).
    /// - **ssim2**: real warm_ref_strip via fast-ssim2 0.8.1
    ///   `Ssimulacra2Reference::compare_strip` (~220 MiB at 40 MP).
    /// - **butter**: real warm_ref_strip via butteraugli 0.9.3
    ///   `ButteraugliReference::compare_strip` (requires reference
    ///   built via `new`/`new_linear`; planar refs return
    ///   `InvalidParameter`).
    /// - **zensim**: real warm_ref_strip via zensim
    ///   `compute_with_ref_streaming_strips` — reuses cached
    ///   `PrecomputedReference` across strips for batch encoder loops.
    /// - **cvvdp**: API stub — delegates to `score_with_warm_ref()`.
    /// - **dssim**: not yet wired (no upstream strip API).
    #[allow(dead_code)]
    #[allow(unused_variables)] // strip_height is unused only when no cpu-* backend feature is enabled
    pub fn compute_with_cached_reference_strip(
        &mut self,
        dist_bytes: &[u8],
        strip_height: u32,
    ) -> Result<Score, CpuAdapterError> {
        let expected = (self.width as usize) * (self.height as usize) * 3;
        if dist_bytes.len() != expected {
            return Err(CpuAdapterError::InvalidInputSize {
                expected,
                got: dist_bytes.len(),
            });
        }
        match &mut self.state {
            #[cfg(feature = "cpu-cvvdp")]
            CpuAdapterState::Cvvdp(c) => {
                let h = if strip_height == 0 { 512 } else { strip_height };
                let v = c
                    .score_with_warm_ref_strip(dist_bytes, h)
                    .map_err(|e| CpuAdapterError::Failed(e.to_string()))?;
                Ok(make_score("cvvdp", cvvdp_cpu_version(), v as f64))
            }
            #[cfg(feature = "cpu-iwssim")]
            CpuAdapterState::Iwssim(c) => {
                if !c.has_reference() {
                    return Err(CpuAdapterError::Failed(
                        "iwssim: no cached reference; call set_reference first".into(),
                    ));
                }
                let h = if strip_height == 0 {
                    iwssim::STRIP_BODY_DEFAULT
                } else {
                    strip_height
                };
                let result = c
                    .score_with_warm_ref_strip(dist_bytes, h)
                    .map_err(|e| CpuAdapterError::Failed(e.to_string()))?;
                Ok(make_score("iwssim", iwssim_cpu_version(), result.score))
            }
            #[cfg(feature = "cpu-ssim2")]
            CpuAdapterState::Ssim2(s) => {
                // fast-ssim2 0.8.1 `Ssimulacra2Reference::compare_strip`
                // strip-walks the dist side; cached ref-side data stays
                // resident (so peak heap is bounded by dist-side only,
                // ~220 MiB at 40 MP).
                let precomputed = s.cached_ref.as_ref().ok_or_else(|| {
                    CpuAdapterError::Failed(
                        "ssim2: no cached reference; call set_reference first".into(),
                    )
                })?;
                let h = if strip_height == 0 { 256 } else { strip_height };
                let dist_img = ssim2_image_ref(dist_bytes, s.width, s.height);
                let v = precomputed.compare_strip(dist_img, h).map_err(|e| {
                    CpuAdapterError::Failed(format!("fast-ssim2 compare_strip: {e}"))
                })?;
                Ok(make_score("ssim2", env!("CARGO_PKG_VERSION"), v))
            }
            #[cfg(feature = "cpu-dssim")]
            CpuAdapterState::Dssim(_) => Err(CpuAdapterError::Failed(
                "dssim warm_ref strip-mode not yet wired in cpu_adapter".into(),
            )),
            #[cfg(feature = "cpu-butter")]
            CpuAdapterState::Butter(s) => {
                // butteraugli 0.9.3 `ButteraugliReference::compare_strip`
                // requires the reference to have been built via
                // `new` or `new_linear` (planar constructor doesn't
                // retain interleaved source — `compare_strip` returns
                // `InvalidParameter` in that case).
                let pre = s.cached_ref.as_ref().ok_or_else(|| {
                    CpuAdapterError::Failed(
                        "butter: no cached reference; call set_reference first".into(),
                    )
                })?;
                let h = if strip_height == 0 { 256 } else { strip_height };
                let result = pre.compare_strip(dist_bytes, h).map_err(|e| {
                    CpuAdapterError::Failed(format!(
                        "butteraugli ButteraugliReference::compare_strip: {e:?}"
                    ))
                })?;
                Ok(make_score(
                    "butter",
                    env!("CARGO_PKG_VERSION"),
                    result.score,
                ))
            }
            #[cfg(feature = "cpu-zensim")]
            CpuAdapterState::Zensim(s) => {
                // Task #134 (2026-05-28): dispatch through the cached
                // `PrecomputedReference` and the strip-aggregating
                // `compute_with_ref_streaming_strips` entrypoint. The
                // distorted side still walks horizontal strips (peak
                // dist memory ~O(strip_h × width)), but the source-side
                // pyramid is reused from the cached precompute — the
                // best of both worlds for batch quantization loops.
                //
                // Replaces the prior wiring which fell back to the cold
                // `compute_streaming_strips` path (rebuilds the per-strip
                // reference precompute on every call), losing the warm
                // amortization for any caller that combined memory-bounded
                // strip dispatch with cached_ref hints.
                let precomputed = s.cached_ref.as_ref().ok_or_else(|| {
                    CpuAdapterError::Failed(
                        "zensim: no cached reference; call set_reference first".into(),
                    )
                })?;
                let inner = if strip_height == 0 {
                    256
                } else {
                    strip_height as usize
                };
                let margin = inner / 2;
                let dst: &[[u8; 3]] = bytemuck::cast_slice(dist_bytes);
                let dist_slice = zensim::RgbSlice::new(dst, s.width, s.height);
                let result = s
                    .zensim
                    .compute_with_ref_streaming_strips(precomputed, &dist_slice, inner, margin)
                    .map_err(|e| {
                        CpuAdapterError::Failed(format!(
                            "zensim compute_with_ref_streaming_strips: {e:?}"
                        ))
                    })?;
                Ok(make_score(
                    "zensim",
                    env!("CARGO_PKG_VERSION"),
                    result.score(),
                ))
            }
            CpuAdapterState::FeatureDisabled(k) => Err(CpuAdapterError::FeatureNotEnabled(*k)),
            CpuAdapterState::Unavailable(k) => Err(CpuAdapterError::Unavailable(*k)),
        }
    }

    /// Whether this backend supports memory-bounded strip dispatch.
    /// Used by the orchestrator's chooser to decide whether
    /// `MemoryMode::Strip` is a candidate for CPU dispatch on this
    /// metric.
    ///
    /// Phase 9.Z.B follow-on (2026-05-28): ssim2, butter, zensim, and
    /// iwssim all expose real strip walkers via the sibling crates.
    /// cvvdp returns `false` (API stub delegates to full path; multi-day
    /// walker queued).
    #[allow(dead_code)]
    pub fn supports_strip(&self) -> bool {
        match self.state {
            #[cfg(feature = "cpu-cvvdp")]
            CpuAdapterState::Cvvdp(_) => false,
            #[cfg(feature = "cpu-iwssim")]
            CpuAdapterState::Iwssim(_) => true,
            #[cfg(feature = "cpu-ssim2")]
            CpuAdapterState::Ssim2(_) => true,
            #[cfg(feature = "cpu-butter")]
            CpuAdapterState::Butter(_) => true,
            #[cfg(feature = "cpu-zensim")]
            CpuAdapterState::Zensim(_) => true,
            _ => false,
        }
    }

    /// Compute against the previously-set reference. Returns
    /// `Err(Failed)` if [`Self::set_reference`] hasn't been called yet
    /// (or was reset by a prior `compute` on backends without cached
    /// state).
    #[allow(dead_code)] // only called when feature = "cuda" is on (via pool.rs)
    pub fn compute_with_cached_reference(
        &mut self,
        dist_bytes: &[u8],
    ) -> Result<Score, CpuAdapterError> {
        let expected = (self.width as usize) * (self.height as usize) * 3;
        if dist_bytes.len() != expected {
            return Err(CpuAdapterError::InvalidInputSize {
                expected,
                got: dist_bytes.len(),
            });
        }
        match &mut self.state {
            #[cfg(feature = "cpu-cvvdp")]
            CpuAdapterState::Cvvdp(c) => {
                let v = c
                    .score_with_warm_ref(dist_bytes)
                    .map_err(|e| CpuAdapterError::Failed(e.to_string()))?;
                Ok(make_score("cvvdp", cvvdp_cpu_version(), v as f64))
            }
            #[cfg(feature = "cpu-ssim2")]
            CpuAdapterState::Ssim2(s) => {
                let precomputed = s.cached_ref.as_ref().ok_or_else(|| {
                    CpuAdapterError::Failed(
                        "ssim2: no cached reference; call set_reference first".into(),
                    )
                })?;
                let dist_img = ssim2_image_ref(dist_bytes, s.width, s.height);
                let v = precomputed
                    .compare(dist_img)
                    .map_err(|e| CpuAdapterError::Failed(format!("fast-ssim2 compare: {e}")))?;
                Ok(make_score("ssim2", env!("CARGO_PKG_VERSION"), v))
            }
            #[cfg(feature = "cpu-dssim")]
            CpuAdapterState::Dssim(s) => {
                let r = s.cached_ref.as_ref().ok_or_else(|| {
                    CpuAdapterError::Failed(
                        "dssim: no cached reference; call set_reference first".into(),
                    )
                })?;
                let dist_img = make_dssim_image(&s.dssim, dist_bytes, s.width, s.height)?;
                let (score, _maps) = s.dssim.compare(r, dist_img);
                Ok(make_score("dssim", dssim_cpu_version(), f64::from(score)))
            }
            #[cfg(feature = "cpu-butter")]
            CpuAdapterState::Butter(s) => {
                // Phase 9.Y: dispatch against the cached `ButteraugliReference`.
                // The compare path runs the dist-side sRGB → linear → XYB
                // → frequency-separation, then diffs against the precomputed
                // ref-side data. ~30-50 % fewer ops per call than `full`.
                let pre = s.cached_ref.as_ref().ok_or_else(|| {
                    CpuAdapterError::Failed(
                        "butter: no cached reference; call set_reference first".into(),
                    )
                })?;
                let result = pre
                    .compare(dist_bytes)
                    .map_err(|e| CpuAdapterError::Failed(format!("butteraugli compare: {e:?}")))?;
                Ok(make_score(
                    "butter",
                    env!("CARGO_PKG_VERSION"),
                    result.score,
                ))
            }
            #[cfg(feature = "cpu-zensim")]
            CpuAdapterState::Zensim(s) => {
                // Task #134 (2026-05-28): dispatch through the cached
                // `PrecomputedReference`. The compare path runs the
                // distorted-side sRGB → linear → XYB + pyramid downscale,
                // then diffs against the precomputed ref pyramid — the
                // ref-side conversion + downscale (≈50 % of the per-pair
                // pipeline by CPU sweep #132 measurement) is hoisted out.
                // Replaces the prior wiring which cloned the raw byte
                // stash and re-ran the full cold path on every call.
                let precomputed = s.cached_ref.as_ref().ok_or_else(|| {
                    CpuAdapterError::Failed(
                        "zensim: no cached reference; call set_reference first".into(),
                    )
                })?;
                let dst: &[[u8; 3]] = bytemuck::cast_slice(dist_bytes);
                let dist_slice = zensim::RgbSlice::new(dst, s.width, s.height);
                let result = s
                    .zensim
                    .compute_with_ref(precomputed, &dist_slice)
                    .map_err(|e| {
                        CpuAdapterError::Failed(format!("zensim compute_with_ref: {e:?}"))
                    })?;
                Ok(make_score(
                    "zensim",
                    env!("CARGO_PKG_VERSION"),
                    result.score(),
                ))
            }
            #[cfg(feature = "cpu-iwssim")]
            CpuAdapterState::Iwssim(c) => {
                if !c.has_reference() {
                    return Err(CpuAdapterError::Failed(
                        "iwssim: no cached reference; call set_reference first".into(),
                    ));
                }
                // Task #136 (2026-05-28): route through the strip walker
                // even on the "cached_ref" entry point. The non-strip
                // `score_with_warm_ref` retains `lp_ref + g_ref` but
                // STILL builds a full-image `lp_dis` pyramid and a
                // full-image `compute_iw_maps` working set inside
                // `score_with_split` (~11 × `h*w` f32 just in
                // `box_stats_3x3`; +`nexp × big_n × 4` in
                // `build_y_matrix`). Measured (heaptrack process peak,
                // 7950X, synth pair):
                //
                //   size  | full     | warm_ref  | warm_ref_strip
                //   ------+----------+-----------+----------------
                //   1 MP  | 153.8 MB | 153.8 MB  | 103.6 MB   (-33 %)
                //   16 MP | 2.47 GB  | 2.47 GB   | 1.29 GB    (-48 %)
                //   40 MP | 5.90 GB  | 5.90 GB   | 3.07 GB    (-48 %)
                //
                // Per-pair score diff at all measured sizes ≤ 2e-6
                // absolute — well inside iwssim's documented strip
                // parity tolerance of 1e-4 (see
                // `crates/iwssim/tests/strip_parity.rs` and
                // `iwssim::Iwssim::score_with_warm_ref_strip` docs).
                // Wall time regresses 9.5 % (1 MP) → 23.9 % (16 MP) →
                // 34.7 % (40 MP) — accepted because the
                // cached-ref entry's value proposition for the
                // orchestrator is amortizing the ref-side pyramid,
                // which the strip variant ALSO does (warm state
                // identical: `lp_ref`, `g_ref`, per-scale `eigs`).
                // The explicit strip-mode entry
                // `compute_with_cached_reference_strip` remains for
                // callers that want to pin a body height.
                let result = c
                    .score_with_warm_ref_strip(dist_bytes, iwssim::STRIP_BODY_DEFAULT)
                    .map_err(|e| CpuAdapterError::Failed(e.to_string()))?;
                Ok(make_score("iwssim", iwssim_cpu_version(), result.score))
            }
            CpuAdapterState::FeatureDisabled(k) => Err(CpuAdapterError::FeatureNotEnabled(*k)),
            CpuAdapterState::Unavailable(k) => Err(CpuAdapterError::Unavailable(*k)),
        }
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

#[allow(dead_code)] // only used when at least one cpu-* feature is on
fn make_score(name: &'static str, version: &'static str, value: f64) -> Score {
    Score {
        value,
        metric_name: name,
        metric_version: version,
    }
}

// ---------------------------------------------------------------------------
// cvvdp wiring
// ---------------------------------------------------------------------------

#[cfg(feature = "cpu-cvvdp")]
fn construct_cvvdp(
    width: u32,
    height: u32,
    params: &MetricParams,
) -> Result<CpuAdapterState, CpuAdapterError> {
    // cvvdp re-exports CvvdpParams from cvvdp-gpu, and the umbrella
    // wraps the *same* struct in MetricParams::Cvvdp. So we can lift
    // the params without an extra translation table.
    let p = match params {
        MetricParams::Cvvdp(p) => p.clone(),
        _ => {
            return Err(CpuAdapterError::Failed(format!(
                "expected MetricParams::Cvvdp, got {params:?}"
            )));
        }
    };
    let c =
        cvvdp::Cvvdp::new(width, height, p).map_err(|e| CpuAdapterError::Failed(e.to_string()))?;
    Ok(CpuAdapterState::Cvvdp(Box::new(c)))
}

#[cfg(not(feature = "cpu-cvvdp"))]
#[allow(unused_variables)]
fn construct_cvvdp(
    _width: u32,
    _height: u32,
    _params: &MetricParams,
) -> Result<CpuAdapterState, CpuAdapterError> {
    Err(CpuAdapterError::FeatureNotEnabled(MetricKind::Cvvdp))
}

#[cfg(feature = "cpu-cvvdp")]
fn compute_cvvdp(c: &mut cvvdp::Cvvdp, r: &[u8], d: &[u8]) -> Result<Score, CpuAdapterError> {
    let v = c
        .score(r, d)
        .map_err(|e| CpuAdapterError::Failed(e.to_string()))?;
    Ok(make_score("cvvdp", cvvdp_cpu_version(), v as f64))
}

#[cfg(feature = "cpu-cvvdp")]
fn cvvdp_cpu_version() -> &'static str {
    // Crate's package version is the canonical identifier — cvvdp
    // re-uses cvvdp-gpu's PYCVVDP_REFERENCE_VERSION constant, but that's
    // the upstream pycvvdp version, not the adapter version. Keep them
    // distinct: this string identifies which Rust impl produced the score.
    env!("CARGO_PKG_VERSION")
}

// ---------------------------------------------------------------------------
// fast-ssim2 wiring (Phase 8h — replaces upstream ssimulacra2 0.5)
// ---------------------------------------------------------------------------
//
// fast-ssim2 implements `ToLinearRgb` for `ImgRef<'_, [u8; 3]>` (with the
// `imgref` feature, enabled by our `cpu-ssim2`). The crate takes care of
// sRGB → linear RGB → XYB conversion internally, including a SIMD-accelerated
// LUT for the u8 sRGB → linear step (`srgb_u8_to_linear`). We hand it an
// `ImgRef` backed by an interleaved `[u8; 3]` buffer constructed from the
// raw sRGB-u8 input. The prior ssimulacra2 0.5 path did this conversion
// manually via `Xyb::try_from(Rgb::new(...))`, which forced the adapter to
// build a `Vec<[f32; 3]>` of normalised RGB before every call. fast-ssim2's
// `ImgRef` path skips that intermediate allocation when the caller already
// has u8 bytes on hand (our case).

#[cfg(feature = "cpu-ssim2")]
fn construct_ssim2(
    width: u32,
    height: u32,
    _params: &MetricParams,
) -> Result<CpuAdapterState, CpuAdapterError> {
    Ok(CpuAdapterState::Ssim2(Ssim2State {
        width: width as usize,
        height: height as usize,
        cached_ref: None,
    }))
}

#[cfg(not(feature = "cpu-ssim2"))]
#[allow(unused_variables)]
fn construct_ssim2(
    _width: u32,
    _height: u32,
    _params: &MetricParams,
) -> Result<CpuAdapterState, CpuAdapterError> {
    Err(CpuAdapterError::FeatureNotEnabled(MetricKind::Ssim2))
}

/// Borrow an interleaved sRGB-u8 byte buffer as an `ImgRef<'_, [u8; 3]>`.
///
/// fast-ssim2's `ToLinearRgb` impl for `ImgRef<'_, [u8; 3]>` reads `[r, g, b]`
/// triplets directly. `[u8; 3]` is `bytemuck::Pod`, so we can reinterpret the
/// raw bytes in place via `bytemuck::cast_slice` — no allocation, no copy.
///
/// Phase 9.Y (2026-05-27): replaces the prior `chunks_exact(3).collect()`
/// path that built a 120 MB `Vec<[u8; 3]>` per side at 40 MP. Heaptrack
/// confirmed 240 MB / pair adapter overhead on the ssim2 row before this
/// swap.
#[cfg(feature = "cpu-ssim2")]
fn ssim2_image_ref<'a>(bytes: &'a [u8], w: usize, h: usize) -> imgref::ImgRef<'a, [u8; 3]> {
    let pixels: &[[u8; 3]] = bytemuck::cast_slice(bytes);
    imgref::ImgRef::new(pixels, w, h)
}

#[cfg(feature = "cpu-ssim2")]
fn compute_ssim2(
    s: &mut Ssim2State,
    ref_bytes: &[u8],
    dist_bytes: &[u8],
) -> Result<Score, CpuAdapterError> {
    let ref_img = ssim2_image_ref(ref_bytes, s.width, s.height);
    let dist_img = ssim2_image_ref(dist_bytes, s.width, s.height);
    let v = fast_ssim2::compute_ssimulacra2(ref_img, dist_img)
        .map_err(|e| CpuAdapterError::Failed(format!("fast-ssim2 compute_ssimulacra2: {e}")))?;
    Ok(make_score("ssim2", env!("CARGO_PKG_VERSION"), v))
}

// ---------------------------------------------------------------------------
// dssim-core wiring
// ---------------------------------------------------------------------------

#[cfg(feature = "cpu-dssim")]
fn construct_dssim(
    width: u32,
    height: u32,
    _params: &MetricParams,
) -> Result<CpuAdapterState, CpuAdapterError> {
    let dssim = dssim_core::Dssim::new();
    Ok(CpuAdapterState::Dssim(DssimState {
        width: width as usize,
        height: height as usize,
        dssim,
        cached_ref: None,
    }))
}

#[cfg(not(feature = "cpu-dssim"))]
#[allow(unused_variables)]
fn construct_dssim(
    _width: u32,
    _height: u32,
    _params: &MetricParams,
) -> Result<CpuAdapterState, CpuAdapterError> {
    Err(CpuAdapterError::FeatureNotEnabled(MetricKind::Dssim))
}

#[cfg(feature = "cpu-dssim")]
fn make_dssim_image(
    dssim: &dssim_core::Dssim,
    bytes: &[u8],
    w: usize,
    h: usize,
) -> Result<dssim_core::DssimImage<f32>, CpuAdapterError> {
    // dssim-core 3.4 exposes `create_image_rgb(&[RGB<u8>], w, h)` — a thin
    // wrapper that runs `to_rgblu()` internally and then dispatches to the
    // generic `create_image`. `rgb::RGB<u8>` is `bytemuck::Pod` (via the
    // `as-bytes` default-on rgb feature), so we can reinterpret the raw
    // interleaved byte buffer as `&[RGB<u8>]` in place — no allocation.
    //
    // Phase 9.Y (2026-05-27): replaces the prior `chunks_exact(3).collect()`
    // path that built a 120 MB `Vec<RGB<u8>>` per side at 40 MP. The
    // upstream multi-scale LAB pyramid still allocates per call (~9 GB at
    // 40 MP) — that's a dssim-core internal we don't touch.
    let rgb: &[rgb::RGB<u8>] = bytemuck::cast_slice(bytes);
    dssim
        .create_image_rgb(rgb, w, h)
        .ok_or_else(|| CpuAdapterError::Failed("dssim_core create_image_rgb returned None".into()))
}

#[cfg(feature = "cpu-dssim")]
fn compute_dssim(
    s: &mut DssimState,
    ref_bytes: &[u8],
    dist_bytes: &[u8],
) -> Result<Score, CpuAdapterError> {
    let ref_img = make_dssim_image(&s.dssim, ref_bytes, s.width, s.height)?;
    let dist_img = make_dssim_image(&s.dssim, dist_bytes, s.width, s.height)?;
    let (score, _maps) = s.dssim.compare(&ref_img, dist_img);
    Ok(make_score("dssim", dssim_cpu_version(), f64::from(score)))
}

#[cfg(feature = "cpu-dssim")]
fn dssim_cpu_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

// ---------------------------------------------------------------------------
// butteraugli wiring
// ---------------------------------------------------------------------------

#[cfg(feature = "cpu-butter")]
fn construct_butter(
    width: u32,
    height: u32,
    _params: &MetricParams,
) -> Result<CpuAdapterState, CpuAdapterError> {
    // The umbrella's MetricParams::Butter wraps butteraugli_gpu's params,
    // not the CPU crate's. We can't lift them across the boundary
    // (different types). The adapter just uses CPU defaults — production
    // callers tuning butteraugli should configure via the GPU path; the
    // CPU adapter is the OOM-fallback safety net, not a perf primary.
    let params = butteraugli::ButteraugliParams::new();
    Ok(CpuAdapterState::Butter(ButterState {
        width: width as usize,
        height: height as usize,
        params,
        cached_ref: None,
    }))
}

#[cfg(not(feature = "cpu-butter"))]
#[allow(unused_variables)]
fn construct_butter(
    _width: u32,
    _height: u32,
    _params: &MetricParams,
) -> Result<CpuAdapterState, CpuAdapterError> {
    Err(CpuAdapterError::FeatureNotEnabled(MetricKind::Butter))
}

#[cfg(feature = "cpu-butter")]
fn compute_butter(
    s: &mut ButterState,
    ref_bytes: &[u8],
    dist_bytes: &[u8],
) -> Result<Score, CpuAdapterError> {
    use imgref::ImgRef;
    // `rgb::RGB<u8>` is `bytemuck::Pod` (via rgb's default `as-bytes`
    // feature), so we can reinterpret the raw interleaved byte buffer as
    // `&[RGB<u8>]` in place — no allocation, no copy.
    //
    // Phase 9.Y (2026-05-27): replaces the prior `chunks_exact(3).collect()`
    // pair that built two 120 MB `Vec<RGB<u8>>` allocations at 40 MP.
    let ref_rgb: &[rgb::RGB<u8>] = bytemuck::cast_slice(ref_bytes);
    let dist_rgb: &[rgb::RGB<u8>] = bytemuck::cast_slice(dist_bytes);
    let ref_img = ImgRef::new(ref_rgb, s.width, s.height);
    let dist_img = ImgRef::new(dist_rgb, s.width, s.height);
    let result = butteraugli::butteraugli(ref_img, dist_img, &s.params)
        .map_err(|e| CpuAdapterError::Failed(format!("butteraugli: {e:?}")))?;
    Ok(make_score(
        "butter",
        env!("CARGO_PKG_VERSION"),
        result.score,
    ))
}

// ---------------------------------------------------------------------------
// zensim wiring
// ---------------------------------------------------------------------------
//
// Task #134 (2026-05-28): the cold-call helper `compute_zensim` below
// dispatches the one-shot `Zensim::compute` path (no cached reference).
// The warm-ref path uses `Zensim::precompute_reference` +
// `Zensim::compute_with_ref`, plumbed through `set_reference` +
// `compute_with_cached_reference` above. Wired so the orchestrator's
// cached-ref dispatch (which queries `supports_cached_ref`) actually
// selects the +12 %-faster warm path on the 7950X at 16 MP (measured
// 3-trial median, 10-distorteds-per-ref sweep; see
// `benchmarks/zensim_cached_ref_cpu_2026-05-28.meta`). The strip variant
// (`compute_with_cached_reference_strip`) dispatches through
// `Zensim::compute_with_ref_streaming_strips` so the warm amortization
// carries into memory-bounded strip mode.

#[cfg(feature = "cpu-zensim")]
fn construct_zensim(
    width: u32,
    height: u32,
    _params: &MetricParams,
) -> Result<CpuAdapterState, CpuAdapterError> {
    // zensim crate exposes the same default profile that the GPU crate
    // wraps. Use `latest_preview()` (replacement for the deprecated
    // `latest()`) to match production sweep workers.
    let zensim = zensim::Zensim::new(zensim::ZensimProfile::latest_preview());
    Ok(CpuAdapterState::Zensim(ZensimState {
        width: width as usize,
        height: height as usize,
        zensim,
        cached_ref: None,
    }))
}

#[cfg(not(feature = "cpu-zensim"))]
#[allow(unused_variables)]
fn construct_zensim(
    _width: u32,
    _height: u32,
    _params: &MetricParams,
) -> Result<CpuAdapterState, CpuAdapterError> {
    Err(CpuAdapterError::FeatureNotEnabled(MetricKind::Zensim))
}

#[cfg(feature = "cpu-zensim")]
fn compute_zensim(
    s: &mut ZensimState,
    ref_bytes: &[u8],
    dist_bytes: &[u8],
) -> Result<Score, CpuAdapterError> {
    // RgbSlice expects `&[[u8; 3]]`. `[u8; 3]` is `bytemuck::Pod`, so we
    // can reinterpret the raw interleaved byte buffer as `&[[u8; 3]]`
    // in place via `bytemuck::cast_slice` — no allocation, no copy, and
    // (importantly) no `unsafe` code in our adapter (bytemuck's `cast_slice`
    // is the safe wrapper around the underlying transmute).
    //
    // Phase 9.Y (2026-05-27): replaces the prior `chunks_exact(3).collect()`
    // pair that built two 120 MB `Vec<[u8; 3]>` allocations at 40 MP.
    let src: &[[u8; 3]] = bytemuck::cast_slice(ref_bytes);
    let dst: &[[u8; 3]] = bytemuck::cast_slice(dist_bytes);
    let ref_slice = zensim::RgbSlice::new(src, s.width, s.height);
    let dist_slice = zensim::RgbSlice::new(dst, s.width, s.height);
    let result = s
        .zensim
        .compute(&ref_slice, &dist_slice)
        .map_err(|e| CpuAdapterError::Failed(format!("zensim: {e:?}")))?;
    Ok(make_score(
        "zensim",
        env!("CARGO_PKG_VERSION"),
        result.score(),
    ))
}

// ---------------------------------------------------------------------------
// iwssim wiring (Phase 8g)
// ---------------------------------------------------------------------------

#[cfg(feature = "cpu-iwssim")]
fn construct_iwssim(
    width: u32,
    height: u32,
    _params: &MetricParams,
) -> Result<CpuAdapterState, CpuAdapterError> {
    // The umbrella has no per-metric IwssimParams variant (the GPU
    // path uses iwssim_gpu::IwssimParams which is feature-gated).
    // The CPU adapter uses crate defaults — production callers tuning
    // IW-SSIM should configure via the typed API in `iwssim::Iwssim`
    // directly. `allow_small = false` mirrors the GPU port's default
    // (sub-176 inputs are rejected); the OOM ladder downgrades to a
    // smaller backend rather than tiling.
    let scorer = iwssim::Iwssim::new(width, height)
        .map_err(|e| CpuAdapterError::Failed(format!("iwssim::Iwssim::new: {e}")))?;
    Ok(CpuAdapterState::Iwssim(Box::new(scorer)))
}

#[cfg(not(feature = "cpu-iwssim"))]
#[allow(unused_variables)]
fn construct_iwssim(
    _width: u32,
    _height: u32,
    _params: &MetricParams,
) -> Result<CpuAdapterState, CpuAdapterError> {
    Err(CpuAdapterError::FeatureNotEnabled(MetricKind::Iwssim))
}

#[cfg(feature = "cpu-iwssim")]
fn compute_iwssim(
    c: &mut iwssim::Iwssim,
    ref_bytes: &[u8],
    dist_bytes: &[u8],
) -> Result<Score, CpuAdapterError> {
    let result = c
        .score(ref_bytes, dist_bytes)
        .map_err(|e| CpuAdapterError::Failed(e.to_string()))?;
    Ok(make_score("iwssim", iwssim_cpu_version(), result.score))
}

#[cfg(feature = "cpu-iwssim")]
fn iwssim_cpu_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

// ---------------------------------------------------------------------------
// Compile-time fallthrough: when ALL cpu-* features are off, the
// constructors above route to FeatureNotEnabled. To keep the dispatch
// uniform we declare versions for cvvdp_cpu_version / dssim_cpu_version
// only under their feature; the call sites are also gated.
// ---------------------------------------------------------------------------

// Compile-time guarantee: if a single cpu-* feature is on without the
// supporting `bench` feature, the `#![cfg(feature = "bench")]` at the
// crate root makes the module disappear and the executor reverts to
// CpuMetricUnavailable. This mirrors how the executor itself only exists
// behind `bench + cuda`.

#[cfg(test)]
mod tests {
    use super::*;

    // The tests below only construct adapters — they don't exercise
    // the underlying CPU crates, which would otherwise pull large
    // tables / cubecl / linker symbols into the test binary.

    #[test]
    fn iwssim_constructs_or_feature_not_enabled() {
        let params = MetricParams::try_default_for(MetricKind::Iwssim).unwrap();
        // Default test config uses 256×256 (above the 176-px floor
        // required by Iwssim's algorithm).
        let r = CpuAdapter::new(MetricKind::Iwssim, 256, 256, &params);
        if cfg!(feature = "cpu-iwssim") {
            match r {
                Ok(adapter) => {
                    assert_eq!(adapter.metric(), MetricKind::Iwssim);
                    // iwssim does have a true warm-reference path
                    // (refs's pyramid + per-scale Cu eig hoisted).
                    assert!(
                        adapter.supports_cached_ref(),
                        "iwssim should advertise cached-ref support"
                    );
                }
                Err(e) => panic!("expected Ok(adapter) with cpu-iwssim enabled, got {e:?}"),
            }
        } else {
            // Build without cpu-iwssim feature → adapter must surface
            // FeatureNotEnabled so the ladder can advance.
            match r {
                Err(CpuAdapterError::FeatureNotEnabled(MetricKind::Iwssim)) => {}
                Err(other) => panic!(
                    "expected FeatureNotEnabled(Iwssim) without cpu-iwssim, got error {other:?}"
                ),
                Ok(_) => panic!("expected FeatureNotEnabled(Iwssim) without cpu-iwssim, got Ok"),
            }
        }
    }

    /// Generate a blended distortion pair similar to the iwssim
    /// parity tests' fixtures so the IW eigendecomposition stays
    /// in a numerically-similar regime between strip and full
    /// modes. Pure noise inputs cause strip-vs-full lambda drift
    /// of O(1e-2) which is uninformative for adapter wiring tests.
    fn synth_iwssim_pair(w: u32, h: u32, seed: u64) -> (Vec<u8>, Vec<u8>) {
        let n = (w as usize) * (h as usize) * 3;
        let mut ref_buf = vec![0u8; n];
        let mut dist_buf = vec![0u8; n];
        let mut s_ref = seed;
        let mut s_dis = seed.wrapping_mul(2_654_435_769);
        for i in 0..n {
            s_ref ^= s_ref << 13;
            s_ref ^= s_ref >> 7;
            s_ref ^= s_ref << 17;
            ref_buf[i] = (s_ref & 0xFF) as u8;
            s_dis ^= s_dis << 13;
            s_dis ^= s_dis >> 7;
            s_dis ^= s_dis << 17;
            let mixed = (ref_buf[i] as u16) * 230 + ((s_dis as u8) as u16) * 25;
            dist_buf[i] = ((mixed / 256) as u8).min(255);
        }
        (ref_buf, dist_buf)
    }

    #[test]
    #[cfg(feature = "cpu-iwssim")]
    fn iwssim_strip_dispatch_works() {
        let params = MetricParams::try_default_for(MetricKind::Iwssim).unwrap();
        let mut adapter = CpuAdapter::new(MetricKind::Iwssim, 256, 256, &params)
            .expect("cpu-iwssim adapter constructs");
        assert!(
            adapter.supports_strip(),
            "iwssim should report strip support"
        );
        let (ref_bytes, dist_bytes) = synth_iwssim_pair(256, 256, 0xc0_ffee_12_34);
        let strip_score = adapter
            .compute_strip(&ref_bytes, &dist_bytes, 128)
            .expect("strip compute");
        let full_score = adapter
            .compute(&ref_bytes, &dist_bytes)
            .expect("full compute");
        let diff = (strip_score.value - full_score.value).abs();
        assert!(
            diff < 1e-3,
            "strip-mode iwssim should match full within 1e-3; strip={}, full={}, diff={}",
            strip_score.value,
            full_score.value,
            diff
        );
    }

    #[test]
    #[cfg(feature = "cpu-iwssim")]
    fn iwssim_warm_ref_strip_dispatch_works() {
        let params = MetricParams::try_default_for(MetricKind::Iwssim).unwrap();
        let mut adapter = CpuAdapter::new(MetricKind::Iwssim, 256, 256, &params)
            .expect("cpu-iwssim adapter constructs");
        let (ref_bytes, dist_bytes) = synth_iwssim_pair(256, 256, 0xa1_b2_c3_d4);
        adapter.set_reference(&ref_bytes).expect("set_reference");
        let strip_score = adapter
            .compute_with_cached_reference_strip(&dist_bytes, 128)
            .expect("warm strip compute");
        let warm_score = adapter
            .compute_with_cached_reference(&dist_bytes)
            .expect("warm full compute");
        let diff = (strip_score.value - warm_score.value).abs();
        assert!(
            diff < 1e-3,
            "warm strip iwssim should match warm full within 1e-3; strip={}, warm={}, diff={}",
            strip_score.value,
            warm_score.value,
            diff
        );
    }

    /// Phase 9.Z.B follow-on (2026-05-28): verify ssim2 strip walker
    /// produces a finite score and roughly tracks the full-image path.
    /// The strip walker's score may differ by atomic-tolerance vs full
    /// (~1e-2 on the 0..100 scale at the test size) — guard the
    /// magnitude of the divergence rather than expecting bit-exact.
    #[test]
    #[cfg(feature = "cpu-ssim2")]
    fn ssim2_strip_dispatch_works() {
        let params = MetricParams::try_default_for(MetricKind::Ssim2).unwrap();
        let mut adapter = CpuAdapter::new(MetricKind::Ssim2, 256, 256, &params)
            .expect("cpu-ssim2 adapter constructs");
        assert!(
            adapter.supports_strip(),
            "ssim2 should report strip support after Phase 9.Z.B"
        );
        let (ref_bytes, dist_bytes) = synth_iwssim_pair(256, 256, 0xe5_55_1a_22_u64);
        let strip_score = adapter
            .compute_strip(&ref_bytes, &dist_bytes, 128)
            .expect("ssim2 strip compute");
        let full_score = adapter
            .compute(&ref_bytes, &dist_bytes)
            .expect("ssim2 full compute");
        assert!(
            strip_score.value.is_finite() && full_score.value.is_finite(),
            "both scores must be finite"
        );
        let diff = (strip_score.value - full_score.value).abs();
        assert!(
            diff < 0.5,
            "ssim2 strip vs full diff < 0.5 (0..100 scale); strip={}, full={}, diff={}",
            strip_score.value,
            full_score.value,
            diff
        );
    }

    /// Phase 9.Z.B follow-on (2026-05-28): verify ssim2 warm-ref strip
    /// path produces a finite score consistent with the warm-ref full
    /// path.
    #[test]
    #[cfg(feature = "cpu-ssim2")]
    fn ssim2_warm_ref_strip_dispatch_works() {
        let params = MetricParams::try_default_for(MetricKind::Ssim2).unwrap();
        let mut adapter = CpuAdapter::new(MetricKind::Ssim2, 256, 256, &params)
            .expect("cpu-ssim2 adapter constructs");
        let (ref_bytes, dist_bytes) = synth_iwssim_pair(256, 256, 0xe5_55_1a_44_u64);
        adapter.set_reference(&ref_bytes).expect("set_reference");
        let strip_score = adapter
            .compute_with_cached_reference_strip(&dist_bytes, 128)
            .expect("ssim2 warm strip compute");
        let warm_score = adapter
            .compute_with_cached_reference(&dist_bytes)
            .expect("ssim2 warm full compute");
        assert!(
            strip_score.value.is_finite() && warm_score.value.is_finite(),
            "both scores must be finite"
        );
        let diff = (strip_score.value - warm_score.value).abs();
        assert!(
            diff < 0.5,
            "ssim2 warm strip vs warm full diff < 0.5; strip={}, warm={}, diff={}",
            strip_score.value,
            warm_score.value,
            diff
        );
    }

    /// Phase 9.Z.B follow-on (2026-05-28): butter strip walker
    /// (butteraugli 0.9.3).
    #[test]
    #[cfg(feature = "cpu-butter")]
    fn butter_strip_dispatch_works() {
        let params = MetricParams::try_default_for(MetricKind::Butter).unwrap();
        let mut adapter = CpuAdapter::new(MetricKind::Butter, 256, 256, &params)
            .expect("cpu-butter adapter constructs");
        assert!(
            adapter.supports_strip(),
            "butter should report strip support after Phase 9.Z.B"
        );
        let (ref_bytes, dist_bytes) = synth_iwssim_pair(256, 256, 0xb0_77_e1_2a);
        let strip_score = adapter
            .compute_strip(&ref_bytes, &dist_bytes, 128)
            .expect("butter strip compute");
        let full_score = adapter
            .compute(&ref_bytes, &dist_bytes)
            .expect("butter full compute");
        assert!(
            strip_score.value.is_finite() && full_score.value.is_finite(),
            "both scores must be finite"
        );
        // butteraugli FIR strip parity is documented at ~1e-2 at 1024.
        // Tighter at 256.
        let diff = (strip_score.value - full_score.value).abs();
        assert!(
            diff < 0.05,
            "butter strip vs full diff < 0.05; strip={}, full={}, diff={}",
            strip_score.value,
            full_score.value,
            diff
        );
    }

    /// Phase 9.Z.B follow-on (2026-05-28): butter warm-ref strip.
    #[test]
    #[cfg(feature = "cpu-butter")]
    fn butter_warm_ref_strip_dispatch_works() {
        let params = MetricParams::try_default_for(MetricKind::Butter).unwrap();
        let mut adapter = CpuAdapter::new(MetricKind::Butter, 256, 256, &params)
            .expect("cpu-butter adapter constructs");
        let (ref_bytes, dist_bytes) = synth_iwssim_pair(256, 256, 0xb0_77_e1_2b);
        adapter.set_reference(&ref_bytes).expect("set_reference");
        let strip_score = adapter
            .compute_with_cached_reference_strip(&dist_bytes, 128)
            .expect("butter warm strip compute");
        let warm_score = adapter
            .compute_with_cached_reference(&dist_bytes)
            .expect("butter warm full compute");
        assert!(
            strip_score.value.is_finite() && warm_score.value.is_finite(),
            "both scores must be finite"
        );
        let diff = (strip_score.value - warm_score.value).abs();
        assert!(
            diff < 0.05,
            "butter warm strip vs warm full diff < 0.05; strip={}, warm={}, diff={}",
            strip_score.value,
            warm_score.value,
            diff
        );
    }

    /// Task #134 (2026-05-28): verify the zensim cached-reference
    /// dispatch is wired. `supports_cached_ref` must be `true` (the
    /// prior wiring returned `false` even though `set_reference`
    /// stashed bytes, so the orchestrator never selected the warm
    /// dispatch). The warm-ref score must match the cold compute
    /// score within zensim's documented byte-exact tolerance vs
    /// `compute_with_ref` (< 1e-13 rel; check < 1e-6 here against
    /// the synth pair which scores in the high 90s).
    #[test]
    #[cfg(feature = "cpu-zensim")]
    fn zensim_cached_ref_dispatch_matches_cold() {
        let params = MetricParams::try_default_for(MetricKind::Zensim).unwrap();
        let mut adapter = CpuAdapter::new(MetricKind::Zensim, 256, 256, &params)
            .expect("cpu-zensim adapter constructs");
        assert!(
            adapter.supports_cached_ref(),
            "zensim must advertise cached-ref support after task #134"
        );
        let (ref_bytes, dist_bytes) = synth_iwssim_pair(256, 256, 0x7a_57_4e_05);
        let cold = adapter
            .compute(&ref_bytes, &dist_bytes)
            .expect("cold compute");
        adapter.set_reference(&ref_bytes).expect("set_reference");
        let warm = adapter
            .compute_with_cached_reference(&dist_bytes)
            .expect("warm compute");
        let diff = (warm.value - cold.value).abs();
        assert!(
            diff < 1e-6,
            "zensim warm vs cold must be byte-exact; warm={}, cold={}, diff={}",
            warm.value,
            cold.value,
            diff
        );
        // The second warm call should reuse the cached precompute.
        let warm2 = adapter
            .compute_with_cached_reference(&dist_bytes)
            .expect("second warm compute");
        assert_eq!(warm.value, warm2.value);
    }

    /// Task #134 (2026-05-28): the strip warm-ref path now dispatches
    /// through `compute_with_ref_streaming_strips` (re-uses the
    /// PrecomputedReference across strips). Score must match the
    /// non-strip warm path within zensim's documented strip-aggregation
    /// tolerance (~1 unit on the 0..100 scale at 256×256).
    #[test]
    #[cfg(feature = "cpu-zensim")]
    fn zensim_warm_ref_strip_matches_warm_full() {
        let params = MetricParams::try_default_for(MetricKind::Zensim).unwrap();
        let mut adapter = CpuAdapter::new(MetricKind::Zensim, 256, 256, &params)
            .expect("cpu-zensim adapter constructs");
        let (ref_bytes, dist_bytes) = synth_iwssim_pair(256, 256, 0x7a_57_4e_06);
        adapter.set_reference(&ref_bytes).expect("set_reference");
        let warm_full = adapter
            .compute_with_cached_reference(&dist_bytes)
            .expect("warm full");
        let warm_strip = adapter
            .compute_with_cached_reference_strip(&dist_bytes, 128)
            .expect("warm strip");
        assert!(
            warm_full.value.is_finite() && warm_strip.value.is_finite(),
            "both scores must be finite"
        );
        let diff = (warm_strip.value - warm_full.value).abs();
        assert!(
            diff < 1.0,
            "zensim warm strip vs warm full diff < 1.0; strip={}, warm={}, diff={}",
            warm_strip.value,
            warm_full.value,
            diff
        );
    }

    /// Phase 9.Z.B follow-on (2026-05-28): zensim
    /// compute_streaming_strips dispatch.
    #[test]
    #[cfg(feature = "cpu-zensim")]
    fn zensim_strip_dispatch_works() {
        let params = MetricParams::try_default_for(MetricKind::Zensim).unwrap();
        let mut adapter = CpuAdapter::new(MetricKind::Zensim, 256, 256, &params)
            .expect("cpu-zensim adapter constructs");
        assert!(
            adapter.supports_strip(),
            "zensim should report strip support after Phase 9.Z.B"
        );
        let (ref_bytes, dist_bytes) = synth_iwssim_pair(256, 256, 0x2e_15_1d_77);
        let strip_score = adapter
            .compute_strip(&ref_bytes, &dist_bytes, 128)
            .expect("zensim strip compute");
        let full_score = adapter
            .compute(&ref_bytes, &dist_bytes)
            .expect("zensim full compute");
        assert!(
            strip_score.value.is_finite() && full_score.value.is_finite(),
            "both scores must be finite"
        );
        // zensim is documented byte-exact equivalent to compute_with_ref
        // within f64 epsilon — but compute() vs compute_streaming_strips
        // is a slightly different reduction order. Allow ~1 unit on the
        // 0..100 scale.
        let diff = (strip_score.value - full_score.value).abs();
        assert!(
            diff < 1.0,
            "zensim strip vs full diff < 1.0; strip={}, full={}, diff={}",
            strip_score.value,
            full_score.value,
            diff
        );
    }

    #[test]
    #[cfg(feature = "cpu-cvvdp")]
    fn cvvdp_strip_stub_returns_same_as_full() {
        let params = MetricParams::try_default_for(MetricKind::Cvvdp).unwrap();
        let mut adapter = CpuAdapter::new(MetricKind::Cvvdp, 256, 256, &params)
            .expect("cpu-cvvdp adapter constructs");
        assert!(
            !adapter.supports_strip(),
            "cvvdp strip stub does not yet deliver memory savings"
        );
        let n = 256 * 256 * 3;
        let mut ref_bytes = vec![0u8; n];
        let mut dist_bytes = vec![0u8; n];
        let mut s = 0xdeadbeefu64;
        for i in 0..n {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ref_bytes[i] = (s & 0xFF) as u8;
            dist_bytes[i] = ((s >> 8) & 0xFF) as u8;
        }
        let strip = adapter
            .compute_strip(&ref_bytes, &dist_bytes, 128)
            .expect("strip compute");
        let full = adapter
            .compute(&ref_bytes, &dist_bytes)
            .expect("full compute");
        assert_eq!(
            strip.value, full.value,
            "cvvdp strip stub must equal full (no walker yet)"
        );
    }

    #[test]
    fn error_display_renders() {
        // Display impl must produce something non-empty for each variant
        let e = CpuAdapterError::FeatureNotEnabled(MetricKind::Cvvdp);
        assert!(!format!("{e}").is_empty());
        let e = CpuAdapterError::Unavailable(MetricKind::Iwssim);
        assert!(!format!("{e}").is_empty());
        let e = CpuAdapterError::Failed("oops".into());
        assert!(format!("{e}").contains("oops"));
        let e = CpuAdapterError::InvalidInputSize {
            expected: 12_288,
            got: 0,
        };
        assert!(format!("{e}").contains("12288"));
    }
}
