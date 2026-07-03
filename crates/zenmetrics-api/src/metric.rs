//! `Metric` enum + per-metric variant dispatch.

use crate::Result;
use crate::error::Error;
use crate::memory_mode::MemoryMode;

#[cfg(feature = "pixels")]
use zenpixels::PixelSlice;

// ---------------------------------------------------------------
// Backend
// ---------------------------------------------------------------

/// Selects the compute backend the underlying metric crate dispatches
/// against. Each concrete variant corresponds to a Cargo feature on the
/// umbrella; variants for disabled features still surface a
/// [`Error::BackendNotEnabled`] at construction time so a single
/// `Backend::Cuda` constant in caller code keeps compiling regardless
/// of which backends are enabled in a given build.
///
/// This enum is **always exhaustive** (every backend has a variant
/// regardless of feature flags). The cfg-gating happens inside
/// `Metric::new` — disabled backends return `Err(BackendNotEnabled)`
/// at runtime. This keeps the consumer's match arms stable across
/// builds with different backend feature sets.
///
/// ## Phase 1 of the ideal-API redesign (task #159)
///
/// This redesign reserves `Backend::Cpu` for the **optimized native
/// CPU crates** (fast-ssim2, butteraugli, dssim-core, zensim, in-tree
/// cvvdp/iwssim — the fast path). That dispatch lands in **phase 2**, so
/// the `Cpu` variant does not exist yet. Until then the cubecl-cpu
/// reference path (GPU kernels executed on CPU) is reachable as
/// [`Backend::CubeclCpu`], and the new [`Backend::Auto`] resolves only
/// over the backends available today (`Cuda` / `Wgpu` / `Hip` else
/// `CubeclCpu`). After phase 2, `Auto` will prefer the optimized `Cpu`
/// path over `CubeclCpu` on GPU-less machines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Pick the best **available** backend, resolved by
    /// [`Backend::resolve_auto`]. Today: a working GPU device
    /// (`Cuda` / `Wgpu` / `Hip`, in that preference order, gated on the
    /// backends compiled in) if one is detected, otherwise
    /// [`Backend::CubeclCpu`]. `Auto` never selects a score-shifting
    /// mode — only a backend. Resolution is observable: call
    /// [`Backend::resolve_auto`] to see what it would pick. Phase 2 will
    /// repoint the GPU-less fallback to the optimized `Cpu` path.
    Auto,
    /// CUDA backend (NVIDIA, requires the `cuda` umbrella feature).
    Cuda,
    /// WGPU backend (cross-vendor, requires the `wgpu` umbrella feature).
    Wgpu,
    /// HIP backend (AMD ROCm, requires the `hip` umbrella feature).
    Hip,
    /// **Optimized native-CPU** backend (task #159 phase 2): the fast
    /// hand-written / SIMD CPU crates — `fast-ssim2`, `dssim-core`,
    /// `butteraugli`, `zensim`, and the in-tree `cvvdp` / `iwssim` —
    /// **not** the cubecl-cpu runtime. This is the CPU path users want:
    /// it is the fast one. Requires the matching `cpu-<metric>` feature;
    /// a metric built without its `cpu-*` feature returns
    /// [`Error::BackendNotEnabled`] for this backend. Contrast
    /// [`Backend::CubeclCpu`] (slow, parity/debug only).
    Cpu,
    /// cubecl-cpu reference backend: the GPU metric kernels executed on
    /// CPU via the cubecl-cpu runtime (requires the `cpu` umbrella
    /// feature). This is slow and exists for parity/debug only — it is
    /// **not** the optimized native-CPU path (that becomes `Backend::Cpu`
    /// in phase 2 of the redesign; see the type-level docs). Note that
    /// several metric crates rely on `Atomic<f32>` operations that
    /// cubecl-cpu does not support — kernels may panic at first dispatch
    /// even when this variant is accepted by the constructor. See each
    /// metric crate's `Backend::Cpu` doc (the per-crate enums keep the
    /// historical `Cpu` name for their cubecl-cpu variant).
    CubeclCpu,
}

impl Backend {
    pub(crate) fn tag(self) -> &'static str {
        match self {
            Backend::Auto => "auto",
            Backend::Cuda => "cuda",
            Backend::Wgpu => "wgpu",
            Backend::Hip => "hip",
            Backend::Cpu => "cpu",
            Backend::CubeclCpu => "cubecl_cpu",
        }
    }

    /// Resolve [`Backend::Auto`] to a concrete backend by probing for an
    /// available GPU device. **Observable** (public): callers can see
    /// exactly what `Auto` would pick rather than have it chosen behind
    /// their back.
    ///
    /// Resolution order (phase 1):
    /// 1. If the `cuda` feature is compiled in **and** a CUDA device is
    ///    detected → [`Backend::Cuda`].
    /// 2. else if the `wgpu` feature is compiled in **and** a GPU adapter
    ///    is detected → [`Backend::Wgpu`].
    /// 3. else if the `hip` feature is compiled in **and** a ROCm device
    ///    is detected → [`Backend::Hip`].
    /// 4. else → [`Backend::CubeclCpu`] (the only CPU-side backend that
    ///    exists pre-phase-2; phase 2 repoints this to the optimized
    ///    `Cpu` path).
    ///
    /// Calling this on a non-`Auto` variant returns that variant
    /// unchanged. It **never** returns [`Backend::Auto`] and never
    /// panics.
    ///
    /// The GPU presence check is intentionally lightweight (an
    /// `nvidia-smi` query for CUDA), matching
    /// `zenmetrics-orchestrator`'s detection and honoring the same
    /// `ZENMETRICS_FORCE_NO_GPU=1` override so tests/CI can force the
    /// no-GPU path. It detects that a device is *present*, not that the
    /// full cubecl client initializes — construction still surfaces
    /// [`Error::BackendNotEnabled`] for a backend whose feature is off.
    pub fn resolve_auto() -> Backend {
        Backend::Auto.resolve()
    }

    /// Resolve `self` to a concrete backend. `Auto` probes for hardware
    /// (see [`Backend::resolve_auto`]); every other variant is returned
    /// unchanged. Guaranteed never to return [`Backend::Auto`].
    pub fn resolve(self) -> Backend {
        match self {
            Backend::Auto => crate::capability::resolve_auto_backend(),
            other => other,
        }
    }
}

// ---------------------------------------------------------------
// MetricKind
// ---------------------------------------------------------------

/// Tag enum identifying which of the six metric crates the umbrella
/// dispatches to. Always exhaustive — disabled metrics return
/// [`Error::MetricNotEnabled`] from [`Metric::new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetricKind {
    /// ColorVideoVDP — `cvvdp-gpu`. Range: ~3..10 (10 = identical).
    Cvvdp,
    /// Butteraugli max-norm — `butteraugli-gpu`. Lower = better.
    Butter,
    /// SSIMULACRA2 — `ssim2-gpu`. Range: ~0..100 (100 = identical).
    Ssim2,
    /// DSSIM (D-SSIM) — `dssim-gpu`. Lower = better.
    Dssim,
    /// IW-SSIM — `iwssim-gpu`. Range: `[0, 1]` (1.0 = identical).
    Iwssim,
    /// Zensim (228-feature perceptual extractor + trained weights) —
    /// `zensim-gpu`.
    Zensim,
}

impl MetricKind {
    /// Short stable tag, used in error messages and column headers.
    pub fn tag(self) -> &'static str {
        match self {
            MetricKind::Cvvdp => "cvvdp",
            MetricKind::Butter => "butter",
            MetricKind::Ssim2 => "ssim2",
            MetricKind::Dssim => "dssim",
            MetricKind::Iwssim => "iwssim",
            MetricKind::Zensim => "zensim",
        }
    }
}

// ---------------------------------------------------------------
// Score
// ---------------------------------------------------------------

/// Score returned by every variant of [`Metric`]. The per-crate
/// `Score` types are converted to this umbrella type at the boundary
/// so callers can store scores from multiple metrics in the same
/// `Vec<Score>` / table column.
///
/// Interpretation of [`Self::value`] is per-metric — read
/// [`Self::metric_name`] before comparing values across metrics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Score {
    /// The numeric score (interpretation is per-metric).
    pub value: f64,
    /// Short metric identifier (`"cvvdp"`, `"butter"`, `"ssim2"`,
    /// `"dssim"`, `"iwssim"`, `"zensim"`).
    pub metric_name: &'static str,
    /// Implementation version tag (each metric crate's
    /// `CARGO_PKG_VERSION` or build-time override).
    pub metric_version: &'static str,
}

/// One named scalar output of a metric. A metric's primary score plus any
/// secondary scalars (butteraugli's `pnorm_3`, future cvvdp subscores) are
/// each a `NamedScore`; `name` identifies which one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NamedScore {
    /// Label for this scalar (e.g. `"max"`, `"pnorm_3"`, the metric name).
    pub name: &'static str,
    /// The value (interpretation per `name`).
    pub value: f64,
}

/// The **full** result of scoring one pair: the primary scalar, any
/// secondary scalars, and — for feature-extractor metrics like zensim —
/// the per-pair feature vector. This is the umbrella's lossless return:
/// where [`Metric::compute_srgb_u8`] collapses to a single [`Score`],
/// [`Metric::compute_srgb_u8_multi`] returns everything the metric
/// produced, so callers don't have to bypass the umbrella to reach a
/// metric crate's typed API (butter's `pnorm_3`, zensim's features).
///
/// `scores` always has ≥ 1 entry; `scores[0]` is the primary (the value
/// [`Score::value`] carries). `features` is empty for pure scalar metrics
/// and carries the regime-length vector (228 / 300 / 372) for zensim.
#[derive(Debug, Clone, PartialEq)]
pub struct Scores {
    /// Short metric identifier (matches [`Score::metric_name`]).
    pub metric_name: &'static str,
    /// Implementation version tag (matches [`Score::metric_version`]).
    pub metric_version: &'static str,
    /// Named scalar outputs, ≥ 1. `scores[0]` is the primary score.
    pub scores: Vec<NamedScore>,
    /// Per-pair feature vector for feature-extractor metrics (zensim);
    /// empty for metrics that emit only scalar scores.
    pub features: Vec<f64>,
}

impl Scores {
    /// The primary scalar (`scores[0].value`), or `NaN` if somehow empty.
    pub fn primary(&self) -> f64 {
        self.scores.first().map_or(f64::NAN, |s| s.value)
    }

    /// Look up a named scalar by label (e.g. `"pnorm_3"`).
    pub fn get(&self, name: &str) -> Option<f64> {
        self.scores.iter().find(|s| s.name == name).map(|s| s.value)
    }

    /// Collapse to the single-value [`Score`] (the primary), for callers
    /// that only need the scalar.
    pub fn primary_score(&self) -> Score {
        Score {
            value: self.primary(),
            metric_name: self.metric_name,
            metric_version: self.metric_version,
        }
    }

    /// Build a single-score (no features) `Scores` from a [`Score`] —
    /// the shape every scalar-only metric returns. `allow(dead_code)`
    /// because it's used by the scalar-metric match arms
    /// (cvvdp/ssim2/dssim/iwssim/CPU), which are all cfg'd out in a
    /// `butter`+`zensim`-only build.
    #[allow(dead_code)]
    fn single(s: Score) -> Self {
        Scores {
            metric_name: s.metric_name,
            metric_version: s.metric_version,
            scores: vec![NamedScore {
                name: s.metric_name,
                value: s.value,
            }],
            features: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------
// MetricParams
// ---------------------------------------------------------------

/// Per-metric parameter bundle. Wraps each metric crate's own
/// `<Metric>Params` so the umbrella keeps a single parameter type.
///
/// Use [`MetricParams::default_for`] for a quick-start; reach into
/// the per-crate `params` field for tuning.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum MetricParams {
    /// [`cvvdp_gpu::CvvdpParams`] passthrough.
    #[cfg(feature = "cvvdp")]
    Cvvdp(cvvdp_gpu::CvvdpParams),
    /// [`butteraugli_gpu::ButteraugliParams`] passthrough.
    #[cfg(feature = "butter")]
    Butter(butteraugli_gpu::ButteraugliParams),
    /// CPU-only placeholder for a build with `cpu-butter` but not the GPU
    /// `butter` feature. `cpu_dispatch::CpuMetricState::new`'s Butter arm
    /// only reads the GPU-typed payload behind its own `#[cfg(feature =
    /// "butter")]` lift (mutually exclusive with this arm), so there is
    /// nothing to carry here -- this exists purely so a CPU-only caller can
    /// construct SOME `MetricParams::Butter` value at all (found 2026-07-03:
    /// without it, `MetricParams::try_default_for` had no arm for
    /// `MetricKind::Butter` on a cpu-butter-only build, so `sweep --metric
    /// butteraugli` failed with "not enabled in this build" even though the
    /// CPU path is fully compiled in).
    #[cfg(all(feature = "cpu-butter", not(feature = "butter")))]
    Butter(()),
    /// [`ssim2_gpu::Ssim2Params`] passthrough.
    #[cfg(feature = "ssim2")]
    Ssim2(ssim2_gpu::Ssim2Params),
    /// CPU-only placeholder, see the `cpu-butter`/`Butter(())` doc above --
    /// same issue, same fix, for `sweep --metric ssim2` on a cpu-ssim2-only
    /// build. `cpu_dispatch::CpuMetricState::new`'s Ssim2 arm ignores
    /// `params` entirely.
    #[cfg(all(feature = "cpu-ssim2", not(feature = "ssim2")))]
    Ssim2(()),
    /// [`dssim_gpu::DssimParams`] passthrough.
    #[cfg(feature = "dssim")]
    Dssim(dssim_gpu::DssimParams),
    /// CPU-only placeholder, see the `cpu-butter`/`Butter(())` doc above --
    /// same issue, same fix, for `sweep --metric dssim` on a cpu-dssim-only
    /// build. `cpu_dispatch::CpuMetricState::new`'s Dssim arm ignores
    /// `params` entirely (dssim-core has no configurable params).
    #[cfg(all(feature = "cpu-dssim", not(feature = "dssim")))]
    Dssim(()),
    /// [`iwssim_gpu::IwssimParams`] passthrough.
    #[cfg(feature = "iwssim")]
    Iwssim(iwssim_gpu::IwssimParams),
    /// [`zensim_gpu::ZensimParams`] passthrough.
    #[cfg(feature = "zensim")]
    Zensim(zensim_gpu::ZensimParams),
    /// CPU-only placeholder, see the `cpu-butter`/`Butter(())` doc above --
    /// same issue, same fix, for `Metric::new(MetricKind::Zensim,
    /// Backend::Cpu, ...)` on a cpu-zensim-only build (the sweep CLI's
    /// production path is unaffected -- it calls `zensim::score()` directly,
    /// bypassing this umbrella entirely -- but zenmetrics-api's OWN
    /// `cpu_dispatch` tests hit this construction path directly and caught
    /// it). `cpu_dispatch::CpuMetricState::new`'s Zensim arm ignores
    /// `params` entirely (constructs from `ZensimProfile::latest_preview()`).
    #[cfg(all(feature = "cpu-zensim", not(feature = "zensim")))]
    Zensim(()),
}

impl MetricParams {
    /// Default-construct the params variant matching `kind`. Panics if
    /// the requested metric's Cargo feature is disabled in this build
    /// — callers should match the build's enabled metrics or use
    /// `try_default_for` instead.
    pub fn default_for(kind: MetricKind) -> Self {
        Self::try_default_for(kind).unwrap_or_else(|e| panic!("{e}"))
    }

    /// cvvdp params with a custom photometric display
    /// ([`cvvdp::params::DisplayModel`](crate::cvvdp::params::DisplayModel))
    /// — e.g. an HDR display peak. cvvdp is **display-aware** (a different
    /// peak luminance yields a different JOD), so this is how you target an
    /// HDR display *through the umbrella* without dropping to a per-crate
    /// scorer:
    ///
    /// ```ignore
    /// use zenmetrics_api::{Backend, Metric, MetricKind, MetricParams};
    /// use zenmetrics_api::cvvdp::params::DisplayModel;
    /// let hdr = DisplayModel { y_peak: 1000.0, ..DisplayModel::STANDARD_HDR_LINEAR };
    /// let m = Metric::new(MetricKind::Cvvdp, Backend::Cuda, w, h,
    ///                     MetricParams::cvvdp_with_display(hdr))?;
    /// ```
    ///
    /// (Only the photometry is set here; the viewing geometry stays the
    /// `STANDARD_4K` default — use [`Metric::new`] with a geometry-bearing
    /// path if you also need a non-standard PPD.)
    #[cfg(feature = "cvvdp")]
    pub fn cvvdp_with_display(display: cvvdp_gpu::params::DisplayModel) -> Self {
        Self::Cvvdp(cvvdp_gpu::CvvdpParams {
            display,
            ..Default::default()
        })
    }

    /// Fallible counterpart of [`Self::default_for`] — returns
    /// `Err(MetricNotEnabled)` if the requested metric's feature is
    /// disabled.
    #[allow(unused_variables)]
    pub fn try_default_for(kind: MetricKind) -> Result<Self> {
        match kind {
            #[cfg(feature = "cvvdp")]
            MetricKind::Cvvdp => Ok(Self::Cvvdp(cvvdp_gpu::CvvdpParams::default())),
            #[cfg(feature = "butter")]
            MetricKind::Butter => Ok(Self::Butter(butteraugli_gpu::ButteraugliParams::default())),
            #[cfg(all(feature = "cpu-butter", not(feature = "butter")))]
            MetricKind::Butter => Ok(Self::Butter(())),
            #[cfg(feature = "ssim2")]
            MetricKind::Ssim2 => Ok(Self::Ssim2(ssim2_gpu::Ssim2Params::default())),
            #[cfg(all(feature = "cpu-ssim2", not(feature = "ssim2")))]
            MetricKind::Ssim2 => Ok(Self::Ssim2(())),
            #[cfg(feature = "dssim")]
            MetricKind::Dssim => Ok(Self::Dssim(dssim_gpu::DssimParams::DEFAULT)),
            #[cfg(all(feature = "cpu-dssim", not(feature = "dssim")))]
            MetricKind::Dssim => Ok(Self::Dssim(())),
            #[cfg(feature = "iwssim")]
            MetricKind::Iwssim => Ok(Self::Iwssim(iwssim_gpu::IwssimParams::DEFAULT)),
            #[cfg(feature = "zensim")]
            MetricKind::Zensim => Ok(Self::Zensim(zensim_gpu::ZensimParams::default_weights())),
            #[cfg(all(feature = "cpu-zensim", not(feature = "zensim")))]
            MetricKind::Zensim => Ok(Self::Zensim(())),
            #[allow(unreachable_patterns)]
            other => Err(Error::MetricNotEnabled { kind: other.tag() }),
        }
    }
}

// ---------------------------------------------------------------
// Metric
// ---------------------------------------------------------------

/// Enum-dispatched scorer. One variant per metric crate; every variant
/// wraps the corresponding `<Metric>Opaque` value so the cubecl
/// generic stays inside the metric crate.
///
/// Construct with [`Self::new`]; the per-crate type lives behind the
/// variant so you can pattern-match on `metric.kind()` without
/// re-importing each crate. Most consumers should never need to
/// destructure — the [`Self::compute_srgb_u8`] / [`Self::compute_pixels`]
/// methods forward to the right variant.
#[non_exhaustive]
// `MetricInner::Cpu` wraps the crate-internal `cpu_dispatch::CpuMetricState`,
// exposed only through this `#[non_exhaustive]` variant (external code can
// neither construct nor destructure it) — an intentional opaque, not a
// leaked public type, so the private-interface lint is expected here.
#[allow(private_interfaces)]
#[doc(hidden)]
pub enum MetricInner {
    /// [`cvvdp_gpu::CvvdpOpaque`] variant.
    #[cfg(feature = "cvvdp")]
    Cvvdp(cvvdp_gpu::CvvdpOpaque),
    /// [`butteraugli_gpu::ButteraugliOpaque`] variant.
    #[cfg(feature = "butter")]
    Butter(butteraugli_gpu::ButteraugliOpaque),
    /// [`ssim2_gpu::Ssim2Opaque`] variant.
    #[cfg(feature = "ssim2")]
    Ssim2(ssim2_gpu::Ssim2Opaque),
    /// [`dssim_gpu::DssimOpaque`] variant.
    #[cfg(feature = "dssim")]
    Dssim(dssim_gpu::DssimOpaque),
    /// [`iwssim_gpu::IwssimOpaque`] variant.
    #[cfg(feature = "iwssim")]
    Iwssim(iwssim_gpu::IwssimOpaque),
    /// [`zensim_gpu::ZensimOpaque`] variant.
    #[cfg(feature = "zensim")]
    Zensim(zensim_gpu::ZensimOpaque),
    /// Optimized native-CPU scorer (`Backend::Cpu`, task #159 phase 2):
    /// dispatched to the fast native crates via
    /// [`crate::cpu_dispatch::CpuMetricState`] rather than a `-gpu` opaque
    /// shim. Holds no GPU device handles.
    #[cfg(any(
        feature = "cpu-ssim2",
        feature = "cpu-cvvdp",
        feature = "cpu-dssim",
        feature = "cpu-butter",
        feature = "cpu-zensim",
        feature = "cpu-iwssim"
    ))]
    // Second field is the cached reference (packed sRGB8) for the warm
    // path: `set_reference` stores it, `compute_with_cached_reference`
    // replays the one-shot compute on it — score-identical to one-off
    // (task #159 phase 4b buffer-replay; uniform with the GPU variants'
    // `set_reference`/`compute_with_cached_reference`).
    Cpu(Box<crate::cpu_dispatch::CpuMetricState>, Option<Vec<u8>>),
}

/// Default [`Metric::display_peak`] — the SDR reference white (cd/m²). SDR
/// scoring never consults it (the native sRGB8 path is taken); it only matters
/// if an HDR slice is fed to a metric that wasn't configured for an HDR display,
/// where it bounds the pu-rescale / display-relative mapping. Set an explicit
/// peak for HDR content via [`Metric::with_display_peak`] or [`crate::hdr::HdrScorer`].
pub const SDR_REFERENCE_NITS: f32 = 203.0;

/// A constructed metric scorer plus the HDR **display peak** (cd/m²) it feeds
/// HDR inputs at. Wraps the enum-dispatched [`MetricInner`]; every non-HDR
/// method (`compute_srgb_u8`, `dims`, `set_reference`, …) forwards to it through
/// `Deref`/`DerefMut`, so existing call sites are unchanged.
///
/// [`compute_pixels`](Self::compute_pixels) /
/// [`compute_pixels_multi`](Self::compute_pixels_multi) are **descriptor-driven**:
/// an `RGB8_SRGB` slice scores on the native SDR path (bit-identical to
/// [`MetricInner::compute_srgb_u8`], so validated SDR scores are preserved); any
/// HDR descriptor (PQ / HLG / linear) is fed through the validated per-metric
/// recipe ([`crate::hdr::hdr_feeding`]) at [`display_peak`](Self::display_peak) —
/// pu-rescale u8 for the SSIM-family, display-relative linear planes for
/// cvvdp/butteraugli — so one call scores SDR and HDR alike on every metric,
/// with no silent SDR collapse.
///
/// The peak defaults to [`SDR_REFERENCE_NITS`]; the HDR constructors
/// ([`crate::hdr::HdrScorer`]) set it (and, for cvvdp/butteraugli, the matching
/// display model) together. Set it directly with
/// [`with_display_peak`](Self::with_display_peak).
pub struct Metric {
    inner: MetricInner,
    display_peak: f32,
}

impl core::ops::Deref for Metric {
    type Target = MetricInner;
    #[inline]
    fn deref(&self) -> &MetricInner {
        &self.inner
    }
}
impl core::ops::DerefMut for Metric {
    #[inline]
    fn deref_mut(&mut self) -> &mut MetricInner {
        &mut self.inner
    }
}

impl Metric {
    /// Construct a scorer (same dispatch and args as before). The display peak
    /// defaults to [`SDR_REFERENCE_NITS`]; for HDR use [`crate::hdr::HdrScorer`]
    /// or [`with_display_peak`](Self::with_display_peak).
    pub fn new(
        kind: MetricKind,
        backend: Backend,
        width: u32,
        height: u32,
        params: MetricParams,
    ) -> Result<Self> {
        MetricInner::new(kind, backend, width, height, params).map(Self::with_sdr_peak)
    }

    /// Construct with an explicit [`MemoryMode`]; display peak as in
    /// [`new`](Self::new).
    pub fn new_with_memory_mode(
        kind: MetricKind,
        backend: Backend,
        width: u32,
        height: u32,
        params: MetricParams,
        mode: MemoryMode,
    ) -> Result<Self> {
        MetricInner::new_with_memory_mode(kind, backend, width, height, params, mode)
            .map(Self::with_sdr_peak)
    }

    /// Construct a **native-CPU** HDR scorer (`Backend::Cpu` → `cpu_dispatch`,
    /// the optimized native crates — NEVER cubecl-cpu) at an explicit HDR
    /// display `peak_nits`, baked into the display-aware metrics (butteraugli
    /// `intensity_target`, cvvdp display peak). Unlike [`new`](Self::new) this
    /// needs **no GPU metric feature**: it builds from the `cpu-*` crates
    /// alone, so a pure-CPU build (no cubecl compiled) can score HDR pairs.
    /// Used by [`crate::hdr::HdrScorer`] when the requested backend resolves to
    /// `Backend::Cpu`. The CPU dispatch implements every HDR feeding entry
    /// (`compute_from_linear_interleaved` / `compute_pu_nits_interleaved` /
    /// `compute_pu_luma_gray`), so scoring is fully wired, not a stub.
    #[cfg(any(
        feature = "cpu-ssim2",
        feature = "cpu-cvvdp",
        feature = "cpu-dssim",
        feature = "cpu-butter",
        feature = "cpu-zensim",
        feature = "cpu-iwssim"
    ))]
    pub fn new_cpu_hdr(kind: MetricKind, width: u32, height: u32, peak_nits: f32) -> Result<Self> {
        crate::cpu_dispatch::CpuMetricState::new_hdr(kind, width, height, peak_nits)
            .map(|s| Self::from_inner_with_peak(MetricInner::Cpu(Box::new(s), None), peak_nits))
    }

    /// Wrap an inner scorer at the SDR reference peak.
    #[inline]
    fn with_sdr_peak(inner: MetricInner) -> Self {
        Self {
            inner,
            display_peak: SDR_REFERENCE_NITS,
        }
    }

    /// Wrap an inner scorer at an explicit display peak (crate-internal — the
    /// HDR constructors use this to keep the peak and the display model in sync).
    #[inline]
    pub(crate) fn from_inner_with_peak(inner: MetricInner, peak_nits: f32) -> Self {
        Self {
            inner,
            display_peak: peak_nits,
        }
    }

    /// Set the HDR **display peak** (cd/m²) used to feed HDR slices in
    /// [`compute_pixels`](Self::compute_pixels). For cvvdp/butteraugli this must
    /// match the peak the metric's display model was built with — the
    /// [`crate::hdr::HdrScorer`] constructors keep the two in sync.
    #[must_use]
    pub fn with_display_peak(mut self, peak_nits: f32) -> Self {
        self.display_peak = peak_nits;
        self
    }

    /// The display peak (cd/m²) this scorer feeds HDR inputs at.
    pub fn display_peak(&self) -> f32 {
        self.display_peak
    }

    /// Release this metric's pooled GPU resources back to the device (consumes
    /// `self`). Forwards to [`MetricInner::release`] — needed as an inherent
    /// method because by-value `self` can't move through `Deref`.
    pub fn release(self, backend: Backend) {
        self.inner.release(backend)
    }

    /// Mutable access to the inner enum, for crate-internal and orchestrator
    /// variant-specific paths (e.g. butteraugli warm-up). Prefer the forwarded
    /// methods; reach for this only to pattern-match a specific variant.
    #[doc(hidden)]
    pub fn inner_mut(&mut self) -> &mut MetricInner {
        &mut self.inner
    }
}

#[cfg(feature = "pixels")]
impl Metric {
    /// **Descriptor-driven** scoring → lossless [`Scores`]. The pixel descriptor
    /// decides the path, so SDR and HDR are one call on any metric:
    ///
    /// - `RGB8_SRGB` slice → native SDR path ([`MetricInner::compute_srgb_u8_multi`]
    ///   on the converted bytes — bit-identical to the SDR score).
    /// - any HDR descriptor (PQ / HLG / linear) → the validated per-metric
    ///   feeding ([`crate::hdr::hdr_feeding`]) at [`display_peak`](Self::display_peak):
    ///   SSIM-family → pu-rescale u8; cvvdp/butteraugli → display-relative linear
    ///   planes. No silent SDR collapse.
    pub fn compute_pixels_multi(&mut self, r: PixelSlice<'_>, d: PixelSlice<'_>) -> Result<Scores> {
        let (w, h) = self.inner.dims();
        if r.width() != w || r.rows() != h {
            return Err(Error::DimensionMismatch {
                expected: (w, h),
                got: (r.width(), r.rows()),
            });
        }
        if d.width() != w || d.rows() != h {
            return Err(Error::DimensionMismatch {
                expected: (w, h),
                got: (d.width(), d.rows()),
            });
        }
        let both_srgb8 = r.descriptor() == zenpixels::PixelDescriptor::RGB8_SRGB
            && d.descriptor() == zenpixels::PixelDescriptor::RGB8_SRGB;
        if both_srgb8 {
            let rb = crate::hdr::slice_to_srgb8(&r, w, h)?;
            let db = crate::hdr::slice_to_srgb8(&d, w, h)?;
            return self.inner.compute_srgb_u8_multi(&rb, &db);
        }
        let peak = self.display_peak;
        // Feeding routes on (metric, backend-class). The concrete GPU flavor
        // never changes the recipe, so any GPU-class `Backend` stands in for
        // the non-`Cpu` case; only the native-CPU dispatch differs (its
        // ssim2 is fast-ssim2, which has no integrated-PU entry until a
        // release ships `hdr-pu` — see `hdr::hdr_feeding` docs).
        let backend_class = {
            #[cfg(any(
                feature = "cpu-ssim2",
                feature = "cpu-cvvdp",
                feature = "cpu-dssim",
                feature = "cpu-butter",
                feature = "cpu-zensim",
                feature = "cpu-iwssim"
            ))]
            {
                if matches!(self.inner, MetricInner::Cpu(..)) {
                    Backend::Cpu
                } else {
                    Backend::Cuda
                }
            }
            #[cfg(not(any(
                feature = "cpu-ssim2",
                feature = "cpu-cvvdp",
                feature = "cpu-dssim",
                feature = "cpu-butter",
                feature = "cpu-zensim",
                feature = "cpu-iwssim"
            )))]
            {
                Backend::Cuda
            }
        };
        match crate::hdr::hdr_feeding(self.inner.kind(), backend_class) {
            crate::hdr::HdrFeeding::Unsupported => Err(Error::Metric {
                kind: "dssim",
                message: "no HDR path by design — see hdr::hdr_feeding docs".into(),
            }),
            crate::hdr::HdrFeeding::LinearPlanes => {
                let rr = crate::hdr::slice_to_display_relative_linear_interleaved(&r, peak)?;
                let dd = crate::hdr::slice_to_display_relative_linear_interleaved(&d, peak)?;
                self.inner.compute_from_linear_interleaved_multi(&rr, &dd)
            }
            crate::hdr::HdrFeeding::SdrU8(transfer) => {
                let rb = crate::hdr::slice_to_pu_rescaled_u8(&r, transfer, peak)?;
                let db = crate::hdr::slice_to_pu_rescaled_u8(&d, transfer, peak)?;
                self.inner.compute_srgb_u8_multi(&rb, &db)
            }
            crate::hdr::HdrFeeding::IntegratedPuNits => {
                let rn = crate::hdr::slice_to_absolute_nits_interleaved(&r, peak)?;
                let dn = crate::hdr::slice_to_absolute_nits_interleaved(&d, peak)?;
                self.inner.compute_pu_nits_interleaved_multi(&rn, &dn)
            }
            // Float PU(luma) gray (iwssim): absolute nits → PU21(bt709-luma)
            // at full f32 precision, no u8 round-trip.
            crate::hdr::HdrFeeding::PuLumaGrayF32 => {
                let rn = crate::hdr::slice_to_absolute_nits_interleaved(&r, peak)?;
                let dn = crate::hdr::slice_to_absolute_nits_interleaved(&d, peak)?;
                let rg = crate::hdr::nits_interleaved_to_pu_luma_gray(&rn, peak);
                let dg = crate::hdr::nits_interleaved_to_pu_luma_gray(&dn, peak);
                self.inner.compute_pu_luma_gray_multi(&rg, &dg)
            }
        }
    }

    /// Single-score [`compute_pixels_multi`](Self::compute_pixels_multi). The SDR
    /// path delegates to the inner native single-score `compute_pixels`, so a
    /// validated SDR score is byte-for-byte preserved; HDR returns the primary
    /// of the multi-score.
    pub fn compute_pixels(&mut self, r: PixelSlice<'_>, d: PixelSlice<'_>) -> Result<Score> {
        let both_srgb8 = r.descriptor() == zenpixels::PixelDescriptor::RGB8_SRGB
            && d.descriptor() == zenpixels::PixelDescriptor::RGB8_SRGB;
        if both_srgb8 {
            return self.inner.compute_pixels(r, d);
        }
        self.compute_pixels_multi(r, d).map(|s| s.primary_score())
    }
}

impl MetricInner {
    /// Construct a scorer for `width × height` images on the given
    /// `backend` and per-metric `params`.
    ///
    /// # Errors
    ///
    /// - [`Error::MetricNotEnabled`] if `kind`'s Cargo feature is
    ///   disabled in this build.
    /// - [`Error::BackendNotEnabled`] if `backend`'s Cargo feature is
    ///   disabled in this build.
    /// - [`Error::Metric`] if the underlying metric crate's
    ///   constructor fails (e.g. invalid image size).
    ///
    /// # Panics
    ///
    /// Panics if `params` does not match `kind` (e.g. asking for
    /// `MetricKind::Cvvdp` with `MetricParams::Dssim(...)`). Use
    /// [`MetricParams::default_for`] when in doubt.
    #[allow(unused_variables)]
    pub fn new(
        kind: MetricKind,
        backend: Backend,
        width: u32,
        height: u32,
        params: MetricParams,
    ) -> Result<Self> {
        // `Backend::Cpu` (optimized native, task #159 phase 2) routes to the
        // fast native crates, not the per-crate `-gpu` opaque shims — so
        // intercept it before the GPU backend conversion below. `resolve()`
        // keeps this correct once `Auto` learns to pick `Cpu` (phase 4).
        #[cfg(any(
            feature = "cpu-ssim2",
            feature = "cpu-cvvdp",
            feature = "cpu-dssim",
            feature = "cpu-butter",
            feature = "cpu-zensim",
            feature = "cpu-iwssim"
        ))]
        if backend.resolve() == Backend::Cpu {
            return crate::cpu_dispatch::CpuMetricState::new(kind, width, height, &params)
                .map(|s| MetricInner::Cpu(Box::new(s), None));
        }
        match kind {
            #[cfg(feature = "cvvdp")]
            MetricKind::Cvvdp => {
                let p = match params {
                    MetricParams::Cvvdp(p) => p,
                    _ => panic!("MetricParams variant mismatch (expected Cvvdp)"),
                };
                let b = cvvdp_backend(backend)?;
                cvvdp_gpu::CvvdpOpaque::new(b, width, height, p)
                    .map(MetricInner::Cvvdp)
                    .map_err(|e| Error::Metric {
                        kind: "cvvdp",
                        message: e.to_string(),
                    })
            }
            #[cfg(feature = "butter")]
            MetricKind::Butter => {
                let p = match params {
                    MetricParams::Butter(p) => p,
                    _ => panic!("MetricParams variant mismatch (expected Butter)"),
                };
                let b = butter_backend(backend)?;
                butteraugli_gpu::ButteraugliOpaque::new(b, width, height, p)
                    .map(MetricInner::Butter)
                    .map_err(|e| Error::Metric {
                        kind: "butter",
                        message: e.to_string(),
                    })
            }
            #[cfg(feature = "ssim2")]
            MetricKind::Ssim2 => {
                let p = match params {
                    MetricParams::Ssim2(p) => p,
                    _ => panic!("MetricParams variant mismatch (expected Ssim2)"),
                };
                let b = ssim2_backend(backend)?;
                ssim2_gpu::Ssim2Opaque::new(b, width, height, p)
                    .map(MetricInner::Ssim2)
                    .map_err(|e| Error::Metric {
                        kind: "ssim2",
                        message: e.to_string(),
                    })
            }
            #[cfg(feature = "dssim")]
            MetricKind::Dssim => {
                let p = match params {
                    MetricParams::Dssim(p) => p,
                    _ => panic!("MetricParams variant mismatch (expected Dssim)"),
                };
                let b = dssim_backend(backend)?;
                dssim_gpu::DssimOpaque::new(b, width, height, p)
                    .map(MetricInner::Dssim)
                    .map_err(|e| Error::Metric {
                        kind: "dssim",
                        message: e.to_string(),
                    })
            }
            #[cfg(feature = "iwssim")]
            MetricKind::Iwssim => {
                let p = match params {
                    MetricParams::Iwssim(p) => p,
                    _ => panic!("MetricParams variant mismatch (expected Iwssim)"),
                };
                let b = iwssim_backend(backend)?;
                iwssim_gpu::IwssimOpaque::new(b, width, height, p)
                    .map(MetricInner::Iwssim)
                    .map_err(|e| Error::Metric {
                        kind: "iwssim",
                        message: e.to_string(),
                    })
            }
            #[cfg(feature = "zensim")]
            MetricKind::Zensim => {
                let p = match params {
                    MetricParams::Zensim(p) => p,
                    _ => panic!("MetricParams variant mismatch (expected Zensim)"),
                };
                let b = zensim_backend(backend)?;
                zensim_gpu::ZensimOpaque::new(b, width, height, p)
                    .map(MetricInner::Zensim)
                    .map_err(|e| Error::Metric {
                        kind: "zensim",
                        message: e.to_string(),
                    })
            }
            #[allow(unreachable_patterns)]
            other => Err(Error::MetricNotEnabled { kind: other.tag() }),
        }
    }

    /// Construct a scorer with an explicit [`MemoryMode`] policy.
    ///
    /// Identical to [`Self::new`] but routes through each per-crate
    /// `new_with_memory_mode` so callers can request Full / Strip /
    /// Tile / Auto resolution at the umbrella API. [`MemoryMode::Auto`]
    /// is the implicit default for [`Self::new`].
    ///
    /// Strip/Tile semantics are per-crate; see each metric crate's
    /// `MemoryMode` docs for what `resolve_auto` picks. cvvdp + zensim
    /// fall back to `Full` for `Strip`/`Tile` umbrella inputs at the
    /// boundary `From` conversion — their per-crate constructors then
    /// surface a clear `ModeUnsupported` if the resolved mode isn't
    /// supported.
    ///
    /// # Errors
    ///
    /// Same as [`Self::new`] plus per-crate
    /// [`crate::Error::Metric`] when the requested mode isn't
    /// supported for that metric (e.g. cvvdp/zensim with `Strip`).
    ///
    /// # Panics
    ///
    /// Same as [`Self::new`] — panics on `MetricParams` ↔ `kind`
    /// mismatch.
    #[allow(unused_variables)]
    pub fn new_with_memory_mode(
        kind: MetricKind,
        backend: Backend,
        width: u32,
        height: u32,
        params: MetricParams,
        mode: MemoryMode,
    ) -> Result<Self> {
        // Backend::Cpu (optimized native, task #159 phase 2): MemoryMode is a
        // GPU concern, so `mode` is ignored and we route to the native CPU
        // dispatch. `resolve()` keeps this correct once Auto can pick Cpu.
        #[cfg(any(
            feature = "cpu-ssim2",
            feature = "cpu-cvvdp",
            feature = "cpu-dssim",
            feature = "cpu-butter",
            feature = "cpu-zensim",
            feature = "cpu-iwssim"
        ))]
        if backend.resolve() == Backend::Cpu {
            return crate::cpu_dispatch::CpuMetricState::new(kind, width, height, &params)
                .map(|s| MetricInner::Cpu(Box::new(s), None));
        }
        match kind {
            #[cfg(feature = "cvvdp")]
            MetricKind::Cvvdp => {
                let p = match params {
                    MetricParams::Cvvdp(p) => p,
                    _ => panic!("MetricParams variant mismatch (expected Cvvdp)"),
                };
                let b = cvvdp_backend(backend)?;
                cvvdp_gpu::CvvdpOpaque::new_with_memory_mode(b, width, height, p, mode.into())
                    .map(MetricInner::Cvvdp)
                    .map_err(|e| Error::Metric {
                        kind: "cvvdp",
                        message: e.to_string(),
                    })
            }
            #[cfg(feature = "butter")]
            MetricKind::Butter => {
                let p = match params {
                    MetricParams::Butter(p) => p,
                    _ => panic!("MetricParams variant mismatch (expected Butter)"),
                };
                let b = butter_backend(backend)?;
                butteraugli_gpu::ButteraugliOpaque::new_with_memory_mode(
                    b,
                    width,
                    height,
                    p,
                    mode.into(),
                )
                .map(MetricInner::Butter)
                .map_err(|e| Error::Metric {
                    kind: "butter",
                    message: e.to_string(),
                })
            }
            #[cfg(feature = "ssim2")]
            MetricKind::Ssim2 => {
                let p = match params {
                    MetricParams::Ssim2(p) => p,
                    _ => panic!("MetricParams variant mismatch (expected Ssim2)"),
                };
                let b = ssim2_backend(backend)?;
                ssim2_gpu::Ssim2Opaque::new_with_memory_mode(b, width, height, p, mode.into())
                    .map(MetricInner::Ssim2)
                    .map_err(|e| Error::Metric {
                        kind: "ssim2",
                        message: e.to_string(),
                    })
            }
            #[cfg(feature = "dssim")]
            MetricKind::Dssim => {
                let p = match params {
                    MetricParams::Dssim(p) => p,
                    _ => panic!("MetricParams variant mismatch (expected Dssim)"),
                };
                let b = dssim_backend(backend)?;
                dssim_gpu::DssimOpaque::new_with_memory_mode(b, width, height, p, mode.into())
                    .map(MetricInner::Dssim)
                    .map_err(|e| Error::Metric {
                        kind: "dssim",
                        message: e.to_string(),
                    })
            }
            #[cfg(feature = "iwssim")]
            MetricKind::Iwssim => {
                let p = match params {
                    MetricParams::Iwssim(p) => p,
                    _ => panic!("MetricParams variant mismatch (expected Iwssim)"),
                };
                let b = iwssim_backend(backend)?;
                iwssim_gpu::IwssimOpaque::new_with_memory_mode(b, width, height, p, mode.into())
                    .map(MetricInner::Iwssim)
                    .map_err(|e| Error::Metric {
                        kind: "iwssim",
                        message: e.to_string(),
                    })
            }
            #[cfg(feature = "zensim")]
            MetricKind::Zensim => {
                let p = match params {
                    MetricParams::Zensim(p) => p,
                    _ => panic!("MetricParams variant mismatch (expected Zensim)"),
                };
                let b = zensim_backend(backend)?;
                zensim_gpu::ZensimOpaque::new_with_memory_mode(b, width, height, p, mode.into())
                    .map(MetricInner::Zensim)
                    .map_err(|e| Error::Metric {
                        kind: "zensim",
                        message: e.to_string(),
                    })
            }
            #[allow(unreachable_patterns)]
            other => Err(Error::MetricNotEnabled { kind: other.tag() }),
        }
    }

    /// The [`MetricKind`] this scorer dispatches.
    pub fn kind(&self) -> MetricKind {
        match self {
            #[cfg(any(
                feature = "cpu-ssim2",
                feature = "cpu-cvvdp",
                feature = "cpu-dssim",
                feature = "cpu-butter",
                feature = "cpu-zensim",
                feature = "cpu-iwssim"
            ))]
            MetricInner::Cpu(s, _) => s.kind(),
            #[cfg(feature = "cvvdp")]
            MetricInner::Cvvdp(_) => MetricKind::Cvvdp,
            #[cfg(feature = "butter")]
            MetricInner::Butter(_) => MetricKind::Butter,
            #[cfg(feature = "ssim2")]
            MetricInner::Ssim2(_) => MetricKind::Ssim2,
            #[cfg(feature = "dssim")]
            MetricInner::Dssim(_) => MetricKind::Dssim,
            #[cfg(feature = "iwssim")]
            MetricInner::Iwssim(_) => MetricKind::Iwssim,
            #[cfg(feature = "zensim")]
            MetricInner::Zensim(_) => MetricKind::Zensim,
        }
    }

    /// The configured `(width, height)`.
    pub fn dims(&self) -> (u32, u32) {
        match self {
            #[cfg(any(
                feature = "cpu-ssim2",
                feature = "cpu-cvvdp",
                feature = "cpu-dssim",
                feature = "cpu-butter",
                feature = "cpu-zensim",
                feature = "cpu-iwssim"
            ))]
            MetricInner::Cpu(s, _) => s.dims(),
            #[cfg(feature = "cvvdp")]
            MetricInner::Cvvdp(m) => m.dims(),
            #[cfg(feature = "butter")]
            MetricInner::Butter(m) => m.dims(),
            #[cfg(feature = "ssim2")]
            MetricInner::Ssim2(m) => m.dims(),
            #[cfg(feature = "dssim")]
            MetricInner::Dssim(m) => m.dims(),
            #[cfg(feature = "iwssim")]
            MetricInner::Iwssim(m) => m.dims(),
            #[cfg(feature = "zensim")]
            MetricInner::Zensim(m) => m.dims(),
        }
    }

    /// Score one reference / distorted pair of packed sRGB `R, G, B,
    /// R, G, B, …` buffers (length `width × height × 3`).
    pub fn compute_srgb_u8(&mut self, r: &[u8], d: &[u8]) -> Result<Score> {
        match self {
            #[cfg(any(
                feature = "cpu-ssim2",
                feature = "cpu-cvvdp",
                feature = "cpu-dssim",
                feature = "cpu-butter",
                feature = "cpu-zensim",
                feature = "cpu-iwssim"
            ))]
            MetricInner::Cpu(s, _) => s.compute_srgb_u8(r, d),
            #[cfg(feature = "cvvdp")]
            MetricInner::Cvvdp(m) => {
                m.compute_srgb_u8(r, d)
                    .map(convert_score)
                    .map_err(|e| Error::Metric {
                        kind: "cvvdp",
                        message: e.to_string(),
                    })
            }
            #[cfg(feature = "butter")]
            MetricInner::Butter(m) => {
                m.compute_srgb_u8(r, d)
                    .map(convert_score_butter)
                    .map_err(|e| Error::Metric {
                        kind: "butter",
                        message: e.to_string(),
                    })
            }
            #[cfg(feature = "ssim2")]
            MetricInner::Ssim2(m) => {
                m.compute_srgb_u8(r, d)
                    .map(convert_score_ssim2)
                    .map_err(|e| Error::Metric {
                        kind: "ssim2",
                        message: e.to_string(),
                    })
            }
            #[cfg(feature = "dssim")]
            MetricInner::Dssim(m) => {
                m.compute_srgb_u8(r, d)
                    .map(convert_score_dssim)
                    .map_err(|e| Error::Metric {
                        kind: "dssim",
                        message: e.to_string(),
                    })
            }
            #[cfg(feature = "iwssim")]
            MetricInner::Iwssim(m) => {
                m.compute_srgb_u8(r, d)
                    .map(convert_score_iwssim)
                    .map_err(|e| Error::Metric {
                        kind: "iwssim",
                        message: e.to_string(),
                    })
            }
            #[cfg(feature = "zensim")]
            MetricInner::Zensim(m) => {
                m.compute_srgb_u8(r, d)
                    .map(convert_score_zensim)
                    .map_err(|e| Error::Metric {
                        kind: "zensim",
                        message: e.to_string(),
                    })
            }
        }
    }

    /// Score one sRGB pair and return **everything the metric produced** —
    /// the primary scalar plus any secondary scalars and feature vector —
    /// as [`Scores`]. This is the lossless counterpart to
    /// [`Self::compute_srgb_u8`] (which keeps only the primary):
    ///
    /// - **butter** (GPU or CPU) → `scores = [max, pnorm_3]` (the libjxl
    ///   3-norm the single-value path drops); GPU from one fused reduction
    ///   kernel, CPU via `CpuMetricState::compute_srgb_u8_multi`.
    /// - **zensim** → `scores = [zensim]` + `features` = the regime-length
    ///   feature vector (228 / 300 / 372), extracted in the same pass.
    /// - **cvvdp / ssim2 / dssim / iwssim (and CPU non-butter)** → a single
    ///   scalar.
    ///
    /// Callers that need a metric's extra outputs can stay on the umbrella
    /// instead of constructing a parallel per-crate instance.
    pub fn compute_srgb_u8_multi(&mut self, r: &[u8], d: &[u8]) -> Result<Scores> {
        match self {
            #[cfg(any(
                feature = "cpu-ssim2",
                feature = "cpu-cvvdp",
                feature = "cpu-dssim",
                feature = "cpu-butter",
                feature = "cpu-zensim",
                feature = "cpu-iwssim"
            ))]
            MetricInner::Cpu(s, _) => {
                let (score, pnorm3) = s.compute_srgb_u8_multi(r, d)?;
                Ok(match pnorm3 {
                    // Only butter yields a pnorm_3 — same `[max, pnorm_3]` shape
                    // as the GPU butter arm below and the CPU fast-path in
                    // `compute_from_linear_interleaved_multi`.
                    Some(p) => Scores {
                        metric_name: score.metric_name,
                        metric_version: score.metric_version,
                        scores: vec![
                            NamedScore {
                                name: "max",
                                value: score.value,
                            },
                            NamedScore {
                                name: "pnorm_3",
                                value: p,
                            },
                        ],
                        features: Vec::new(),
                    },
                    None => Scores::single(score),
                })
            }
            #[cfg(feature = "cvvdp")]
            MetricInner::Cvvdp(_) => self.compute_srgb_u8(r, d).map(Scores::single),
            #[cfg(feature = "butter")]
            MetricInner::Butter(m) => m
                .compute_srgb_u8_with_pnorm3(r, d)
                .map(|(s, pnorm3)| Scores {
                    metric_name: "butter",
                    metric_version: s.metric_version,
                    scores: vec![
                        NamedScore {
                            name: "max",
                            value: s.value,
                        },
                        NamedScore {
                            name: "pnorm_3",
                            value: pnorm3,
                        },
                    ],
                    features: Vec::new(),
                })
                .map_err(|e| Error::Metric {
                    kind: "butter",
                    message: e.to_string(),
                }),
            #[cfg(feature = "ssim2")]
            MetricInner::Ssim2(_) => self.compute_srgb_u8(r, d).map(Scores::single),
            #[cfg(feature = "dssim")]
            MetricInner::Dssim(_) => self.compute_srgb_u8(r, d).map(Scores::single),
            #[cfg(feature = "iwssim")]
            MetricInner::Iwssim(_) => self.compute_srgb_u8(r, d).map(Scores::single),
            #[cfg(feature = "zensim")]
            MetricInner::Zensim(m) => m
                .compute_srgb_u8_with_features(r, d)
                .map(|(s, features)| Scores {
                    metric_name: "zensim",
                    metric_version: s.metric_version,
                    scores: vec![NamedScore {
                        name: "zensim",
                        value: s.value,
                    }],
                    features,
                })
                .map_err(|e| Error::Metric {
                    kind: "zensim",
                    message: e.to_string(),
                }),
        }
    }

    /// Faithful HDR scoring from display-relative `[0,1]` linear-RGB f32
    /// planes — six tight `width × height` planes (ref R/G/B, dist R/G/B,
    /// each `nits / intensity_target`). This is the
    /// `hdr::HdrFeeding::LinearPlanes` path: the luminance-aware metrics
    /// (`cvvdp`, `butter`) score the HDR signal natively — no sRGB→linear
    /// LUT, no u8 quantization — so the full highlight range survives.
    /// `zensim` also has a linear path. (The `hdr` module is feature-gated,
    /// so those names are written here as plain text, not doc links.)
    ///
    /// The SSIM-family metrics (`ssim2`, `dssim`, `iwssim`) and the CPU
    /// backend have **no** absolute-luminance linear path. They return
    /// [`Error::Metric`] here (rather than silently mis-scoring) so a
    /// caller that ignored `hdr::hdr_feeding` fails loudly; feed those via
    /// [`Self::compute_srgb_u8`] over `hdr::to_sdr_u8` (the
    /// `hdr::HdrFeeding::SdrU8` path).
    ///
    /// For `butter` the libjxl `pnorm_3` aggregation is dropped here (same
    /// as [`Self::compute_srgb_u8`]); use [`Self::compute_from_linear_planes_multi`]
    /// if you need both the max-norm and 3-norm.
    ///
    /// **`butter` requires whole-image construction**: butteraugli's
    /// [`MemoryMode::Auto`] is strip-preferred, but the linear-planes path is
    /// whole-image only and rejects a strip instance with
    /// [`Error`]`::Metric` (`StripModeUnsupported`). Build the `Metric` with
    /// [`Self::new_with_memory_mode`] + [`MemoryMode::Full`] for the faithful
    /// `butter` HDR path. (`cvvdp`/`zensim` are unaffected — cvvdp is
    /// Full-only and zensim's linear path is whole-image.)
    #[allow(clippy::too_many_arguments)]
    pub fn compute_from_linear_planes(
        &mut self,
        ref_r: &[f32],
        ref_g: &[f32],
        ref_b: &[f32],
        dis_r: &[f32],
        dis_g: &[f32],
        dis_b: &[f32],
    ) -> Result<Score> {
        match self {
            #[cfg(any(
                feature = "cpu-ssim2",
                feature = "cpu-cvvdp",
                feature = "cpu-dssim",
                feature = "cpu-butter",
                feature = "cpu-zensim",
                feature = "cpu-iwssim"
            ))]
            MetricInner::Cpu(..) => Err(Error::Metric {
                kind: "cpu",
                message: "CPU backend has no linear-planes HDR path; feed via \
                          compute_srgb_u8(to_sdr_u8(..)) per hdr_feeding()"
                    .to_string(),
            }),
            #[cfg(feature = "cvvdp")]
            MetricInner::Cvvdp(m) => m
                .compute_from_linear_planes(ref_r, ref_g, ref_b, dis_r, dis_g, dis_b, None)
                .map(convert_score)
                .map_err(|e| Error::Metric {
                    kind: "cvvdp",
                    message: e.to_string(),
                }),
            #[cfg(feature = "butter")]
            MetricInner::Butter(m) => m
                .compute_from_linear_planes(ref_r, ref_g, ref_b, dis_r, dis_g, dis_b)
                .map(|(s, _pnorm3)| convert_score_butter(s))
                .map_err(|e| Error::Metric {
                    kind: "butter",
                    message: e.to_string(),
                }),
            #[cfg(feature = "ssim2")]
            MetricInner::Ssim2(_) => Err(no_linear_planes_path("ssim2")),
            #[cfg(feature = "dssim")]
            MetricInner::Dssim(_) => Err(no_linear_planes_path("dssim")),
            #[cfg(feature = "iwssim")]
            MetricInner::Iwssim(_) => Err(no_linear_planes_path("iwssim")),
            #[cfg(feature = "zensim")]
            MetricInner::Zensim(m) => m
                .score_from_linear_planes(ref_r, ref_g, ref_b, dis_r, dis_g, dis_b)
                .map(|v| Score {
                    value: v as f64,
                    metric_name: "zensim",
                    metric_version: env!("CARGO_PKG_VERSION"),
                })
                .map_err(|e| Error::Metric {
                    kind: "zensim",
                    message: e.to_string(),
                }),
        }
    }

    /// Lossless multi-output counterpart to
    /// [`Self::compute_from_linear_planes`] — the faithful HDR path that
    /// also returns butter's `pnorm_3`. Same per-metric feeding rules:
    /// `cvvdp`/`butter`/`zensim` score natively from the linear planes;
    /// the SSIM-family + CPU return [`Error::Metric`] (no linear path).
    /// `butter` → `scores = [max, pnorm_3]`; the others → one scalar.
    ///
    /// Like [`Self::compute_from_linear_planes`], the `butter` linear path
    /// requires a whole-image instance — construct via
    /// [`Self::new_with_memory_mode`] + [`MemoryMode::Full`] (butter's `Auto`
    /// is strip-preferred and the linear path rejects strip).
    #[allow(clippy::too_many_arguments)]
    pub fn compute_from_linear_planes_multi(
        &mut self,
        ref_r: &[f32],
        ref_g: &[f32],
        ref_b: &[f32],
        dis_r: &[f32],
        dis_g: &[f32],
        dis_b: &[f32],
    ) -> Result<Scores> {
        match self {
            #[cfg(any(
                feature = "cpu-ssim2",
                feature = "cpu-cvvdp",
                feature = "cpu-dssim",
                feature = "cpu-butter",
                feature = "cpu-zensim",
                feature = "cpu-iwssim"
            ))]
            MetricInner::Cpu(..) => Err(Error::Metric {
                kind: "cpu",
                message: "CPU backend has no linear-planes HDR path; feed via \
                          compute_srgb_u8(to_sdr_u8(..)) per hdr_feeding()"
                    .to_string(),
            }),
            #[cfg(feature = "cvvdp")]
            MetricInner::Cvvdp(m) => m
                .compute_from_linear_planes(ref_r, ref_g, ref_b, dis_r, dis_g, dis_b, None)
                .map(convert_score)
                .map(Scores::single)
                .map_err(|e| Error::Metric {
                    kind: "cvvdp",
                    message: e.to_string(),
                }),
            #[cfg(feature = "butter")]
            MetricInner::Butter(m) => m
                .compute_from_linear_planes(ref_r, ref_g, ref_b, dis_r, dis_g, dis_b)
                .map(|(s, pnorm3)| Scores {
                    metric_name: "butter",
                    metric_version: s.metric_version,
                    scores: vec![
                        NamedScore {
                            name: "max",
                            value: s.value,
                        },
                        NamedScore {
                            name: "pnorm_3",
                            value: pnorm3,
                        },
                    ],
                    features: Vec::new(),
                })
                .map_err(|e| Error::Metric {
                    kind: "butter",
                    message: e.to_string(),
                }),
            #[cfg(feature = "ssim2")]
            MetricInner::Ssim2(_) => Err(no_linear_planes_path("ssim2")),
            #[cfg(feature = "dssim")]
            MetricInner::Dssim(_) => Err(no_linear_planes_path("dssim")),
            #[cfg(feature = "iwssim")]
            MetricInner::Iwssim(_) => Err(no_linear_planes_path("iwssim")),
            #[cfg(feature = "zensim")]
            MetricInner::Zensim(m) => m
                .score_from_linear_planes(ref_r, ref_g, ref_b, dis_r, dis_g, dis_b)
                .map(|v| {
                    Scores::single(Score {
                        value: v as f64,
                        metric_name: "zensim",
                        metric_version: env!("CARGO_PKG_VERSION"),
                    })
                })
                .map_err(|e| Error::Metric {
                    kind: "zensim",
                    message: e.to_string(),
                }),
        }
    }

    /// Non-planar convenience for [`Self::compute_from_linear_planes`]: takes
    /// two **interleaved** linear-RGB f32 buffers (`[R,G,B, R,G,B, …]`, each
    /// length `width·height·3`) instead of six planar slices, deinterleaving
    /// on the host before dispatch. Same per-metric feeding rules — and the
    /// same Full-mode requirement for `butter` — as the planar method.
    pub fn compute_from_linear_interleaved(
        &mut self,
        ref_rgb: &[f32],
        dis_rgb: &[f32],
    ) -> Result<Score> {
        // Native CPU butter/cvvdp take interleaved linear directly (no
        // deinterleave→re-interleave round-trip; their native crates want
        // interleaved `RGB<f32>`). GPU metrics deinterleave to their planar
        // kernels below. Never `Backend::CubeclCpu`.
        #[cfg(any(
            feature = "cpu-ssim2",
            feature = "cpu-cvvdp",
            feature = "cpu-dssim",
            feature = "cpu-butter",
            feature = "cpu-zensim",
            feature = "cpu-iwssim"
        ))]
        if let MetricInner::Cpu(s, _) = self {
            return s
                .compute_from_linear_interleaved(ref_rgb, dis_rgb)
                .map(|(s, _)| s);
        }
        let (rr, rg, rb) = deinterleave_rgb(ref_rgb, "reference")?;
        let (dr, dg, db) = deinterleave_rgb(dis_rgb, "distorted")?;
        self.compute_from_linear_planes(&rr, &rg, &rb, &dr, &dg, &db)
    }

    /// Multi-output non-planar counterpart of
    /// [`Self::compute_from_linear_interleaved`] — see
    /// [`Self::compute_from_linear_planes_multi`] (butter → `[max, pnorm_3]`).
    pub fn compute_from_linear_interleaved_multi(
        &mut self,
        ref_rgb: &[f32],
        dis_rgb: &[f32],
    ) -> Result<Scores> {
        // Native CPU fast-path (interleaved, no deinterleave). butter keeps its
        // `[max, pnorm_3]` pair — `butteraugli_linear` returns pnorm_3 too — so
        // the CPU multi-output matches the GPU butter arm.
        #[cfg(any(
            feature = "cpu-ssim2",
            feature = "cpu-cvvdp",
            feature = "cpu-dssim",
            feature = "cpu-butter",
            feature = "cpu-zensim",
            feature = "cpu-iwssim"
        ))]
        if let MetricInner::Cpu(s, _) = self {
            let (score, pnorm3) = s.compute_from_linear_interleaved(ref_rgb, dis_rgb)?;
            return Ok(match pnorm3 {
                // Only butter yields a pnorm_3 — same `[max, pnorm_3]` shape as
                // the GPU butter arm in `compute_from_linear_planes_multi`.
                Some(p) => Scores {
                    metric_name: score.metric_name,
                    metric_version: score.metric_version,
                    scores: vec![
                        NamedScore {
                            name: "max",
                            value: score.value,
                        },
                        NamedScore {
                            name: "pnorm_3",
                            value: p,
                        },
                    ],
                    features: Vec::new(),
                },
                None => Scores::single(score),
            });
        }
        let (rr, rg, rb) = deinterleave_rgb(ref_rgb, "reference")?;
        let (dr, dg, db) = deinterleave_rgb(dis_rgb, "distorted")?;
        self.compute_from_linear_planes_multi(&rr, &rg, &rb, &dr, &dg, &db)
    }

    /// **Integrated PU21 HDR entry** (`hdr::HdrFeeding::IntegratedPuNits`):
    /// score a pair of **absolute-luminance** interleaved linear-RGB f32
    /// buffers (cd/m², `[R,G,B, …]`, each length `width·height·3`), with the
    /// PU21 perceptual encode applied **inside** the metric pipeline.
    ///
    /// Implemented by the GPU ssim2 opaque
    /// (`ssim2_gpu::Ssim2Opaque::compute_linear_nits`, which swaps the
    /// cube-root XYB stage for PU21 — UPIQ SRCC 0.7040, imazen/zenmetrics#25),
    /// by **ssim2 on the native-CPU dispatch**
    /// (`fast_ssim2::compute_ssimulacra2_pu_nits`, the same
    /// PU21-for-cube-root swap in the CPU pipeline — UPIQ SRCC 0.7044 at
    /// fast-ssim2 git 35f198af; `hdr-pu` feature, workspace `[patch]` pin
    /// until a fast-ssim2 release ships it), and by **zensim on the
    /// native-CPU dispatch** (`zensim::Zensim::compute_pu_linear`, the PU21
    /// banding_glare front-end from zensim PR #44 replacing the SDR
    /// cube-root — no u8 round-trip). Every other variant returns
    /// [`Error::Metric`] so a caller that ignored `hdr::hdr_feeding` fails
    /// loudly instead of silently mis-scoring; feed those metrics per their
    /// own `hdr_feeding` recipe.
    pub fn compute_pu_nits_interleaved_multi(
        &mut self,
        ref_nits: &[f32],
        dis_nits: &[f32],
    ) -> Result<Scores> {
        // Only the GPU-opaque arms below reference this; gate it to match so
        // cpu-only feature configurations don't warn it dead.
        #[cfg(any(
            feature = "cvvdp",
            feature = "butter",
            feature = "dssim",
            feature = "iwssim",
            feature = "zensim"
        ))]
        fn no_pu_nits_path(kind: &'static str) -> Error {
            Error::Metric {
                kind,
                message: format!(
                    "metric '{kind}' has no integrated-PU21 nits path; feed it per \
                     hdr::hdr_feeding() (SdrU8 shell or LinearPlanes)"
                ),
            }
        }
        match self {
            #[cfg(feature = "ssim2")]
            MetricInner::Ssim2(m) => m
                .compute_linear_nits(ref_nits, dis_nits)
                .map(convert_score_ssim2)
                .map(Scores::single)
                .map_err(|e| Error::Metric {
                    kind: "ssim2",
                    message: e.to_string(),
                }),
            #[cfg(any(
                feature = "cpu-ssim2",
                feature = "cpu-cvvdp",
                feature = "cpu-dssim",
                feature = "cpu-butter",
                feature = "cpu-zensim",
                feature = "cpu-iwssim"
            ))]
            MetricInner::Cpu(state, _) => state
                .compute_pu_nits_interleaved(ref_nits, dis_nits)
                .map(|(score, features)| {
                    let mut s = Scores::single(score);
                    s.features = features;
                    s
                }),
            #[cfg(feature = "cvvdp")]
            MetricInner::Cvvdp(_) => Err(no_pu_nits_path("cvvdp")),
            #[cfg(feature = "butter")]
            MetricInner::Butter(_) => Err(no_pu_nits_path("butter")),
            #[cfg(feature = "dssim")]
            MetricInner::Dssim(_) => Err(no_pu_nits_path("dssim")),
            #[cfg(feature = "iwssim")]
            MetricInner::Iwssim(_) => Err(no_pu_nits_path("iwssim")),
            #[cfg(feature = "zensim")]
            MetricInner::Zensim(_) => Err(no_pu_nits_path("zensim")),
        }
    }

    /// Float-PU(luma) gray feeding ([`crate::hdr::HdrFeeding::PuLumaGrayF32`]):
    /// score a pair of PU21-encoded BT.709-luma gray planes (f32, 0..255
    /// scale, `width × height` samples each). Supported by **iwssim** on both
    /// the GPU opaque (`compute_gray_f32`) and the native-CPU dispatch
    /// (`Iwssim::score_gray`) — every other variant returns [`Error::Metric`]
    /// so a caller that ignored `hdr::hdr_feeding` fails loudly.
    pub fn compute_pu_luma_gray_multi(
        &mut self,
        ref_gray: &[f32],
        dis_gray: &[f32],
    ) -> Result<Scores> {
        // Only the GPU-opaque arms below reference this; gate it to match so
        // cpu-only feature configurations don't warn it dead.
        #[cfg(any(
            feature = "ssim2",
            feature = "cvvdp",
            feature = "butter",
            feature = "dssim",
            feature = "zensim"
        ))]
        fn no_pu_gray_path(kind: &'static str) -> Error {
            Error::Metric {
                kind,
                message: format!(
                    "metric '{kind}' has no float-PU(luma) gray path; feed it per                      hdr::hdr_feeding()"
                ),
            }
        }
        match self {
            #[cfg(feature = "iwssim")]
            MetricInner::Iwssim(m) => m
                .compute_gray_f32(ref_gray, dis_gray)
                .map(convert_score_iwssim)
                .map(Scores::single)
                .map_err(|e| Error::Metric {
                    kind: "iwssim",
                    message: e.to_string(),
                }),
            #[cfg(any(
                feature = "cpu-ssim2",
                feature = "cpu-cvvdp",
                feature = "cpu-dssim",
                feature = "cpu-butter",
                feature = "cpu-zensim",
                feature = "cpu-iwssim"
            ))]
            MetricInner::Cpu(state, _) => state
                .compute_pu_luma_gray(ref_gray, dis_gray)
                .map(Scores::single),
            #[cfg(feature = "ssim2")]
            MetricInner::Ssim2(_) => Err(no_pu_gray_path("ssim2")),
            #[cfg(feature = "cvvdp")]
            MetricInner::Cvvdp(_) => Err(no_pu_gray_path("cvvdp")),
            #[cfg(feature = "butter")]
            MetricInner::Butter(_) => Err(no_pu_gray_path("butter")),
            #[cfg(feature = "dssim")]
            MetricInner::Dssim(_) => Err(no_pu_gray_path("dssim")),
            #[cfg(feature = "zensim")]
            MetricInner::Zensim(_) => Err(no_pu_gray_path("zensim")),
        }
    }

    /// Score against pre-uploaded packed-u32 device handles —
    /// upload-once Phase 4 entry point. Both handles MUST come from
    /// a [`crate::MetricContext::upload_pair`] call on the same
    /// cubecl client this scorer was constructed against (sharing
    /// handles across clients is undefined behaviour at the cubecl
    /// layer and is not validated here).
    ///
    /// Equivalent to [`Self::compute_srgb_u8`] but skips the
    /// host-to-device upload, so one packed `(ref, dist)` upload
    /// can feed several metrics on the same GPU.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Metric`] if the underlying metric crate's
    /// dispatch fails.
    #[cfg(feature = "cubecl-types")]
    pub fn compute_handles(&mut self, pair: &crate::context::PairHandles) -> Result<Score> {
        let ref_h = &pair.ref_handle;
        let dis_h = &pair.dist_handle;
        match self {
            #[cfg(feature = "cvvdp")]
            MetricInner::Cvvdp(m) => {
                m.compute_handles(ref_h, dis_h)
                    .map(convert_score)
                    .map_err(|e| Error::Metric {
                        kind: "cvvdp",
                        message: e.to_string(),
                    })
            }
            #[cfg(feature = "butter")]
            MetricInner::Butter(m) => m
                .compute_handles(ref_h, dis_h)
                .map(convert_score_butter)
                .map_err(|e| Error::Metric {
                    kind: "butter",
                    message: e.to_string(),
                }),
            #[cfg(feature = "ssim2")]
            MetricInner::Ssim2(m) => m
                .compute_handles(ref_h, dis_h)
                .map(convert_score_ssim2)
                .map_err(|e| Error::Metric {
                    kind: "ssim2",
                    message: e.to_string(),
                }),
            #[cfg(feature = "dssim")]
            MetricInner::Dssim(m) => m
                .compute_handles(ref_h, dis_h)
                .map(convert_score_dssim)
                .map_err(|e| Error::Metric {
                    kind: "dssim",
                    message: e.to_string(),
                }),
            #[cfg(feature = "iwssim")]
            MetricInner::Iwssim(m) => m
                .compute_handles(ref_h, dis_h)
                .map(convert_score_iwssim)
                .map_err(|e| Error::Metric {
                    kind: "iwssim",
                    message: e.to_string(),
                }),
            #[cfg(feature = "zensim")]
            MetricInner::Zensim(_) => {
                // zensim-gpu compute_handles is parked while the parallel
                // rework lands — fall back to compute_srgb_u8 isn't
                // available without the original bytes, so surface a
                // clear "not yet wired" error rather than silently
                // running a different code path.
                Err(Error::Metric {
                    kind: "zensim",
                    message: "compute_handles not wired for zensim-gpu (Phase 4 deferred — see umbrella commit)".into(),
                })
            }
            // The optimized native-CPU backend has no GPU device handles —
            // upload-once is a GPU-only concept. Surface a clear error rather
            // than silently routing elsewhere.
            #[cfg(any(
                feature = "cpu-ssim2",
                feature = "cpu-cvvdp",
                feature = "cpu-dssim",
                feature = "cpu-butter",
                feature = "cpu-zensim",
                feature = "cpu-iwssim"
            ))]
            MetricInner::Cpu(..) => Err(Error::Metric {
                kind: "cpu",
                message:
                    "compute_handles is a GPU upload-once path; Backend::Cpu has no device handles \
                          (use compute_srgb_u8 / compute_pixels)"
                        .into(),
            }),
        }
    }

    /// Score one reference / distorted pair of packed sRGB buffers
    /// AND return the regime-appropriate feature vector when the
    /// configured metric is [`MetricKind::Zensim`]. Other metrics
    /// return `(Score, Vec::new())` so a single call-site can collect
    /// features when present without branching on `kind()`.
    ///
    /// Zensim feature vector length matches the configured
    /// `ZensimFeatureRegime`:
    ///
    /// - 228 floats on `Basic` (default)
    /// - 300 floats on `Extended`
    /// - 372 floats on `WithIw`
    ///
    /// Pass the regime via `MetricParams::Zensim(ZensimParams::new().with_regime(...))`
    /// or the [`zensim_params_with_regime`] helper when constructing
    /// the metric.
    ///
    /// # Errors
    ///
    /// - [`Error::Metric`] if the underlying metric crate's
    ///   dispatch fails.
    #[cfg(feature = "zensim")]
    pub fn compute_features_srgb_u8(&mut self, r: &[u8], d: &[u8]) -> Result<(Score, Vec<f64>)> {
        match self {
            #[cfg(any(
                feature = "cpu-ssim2",
                feature = "cpu-cvvdp",
                feature = "cpu-dssim",
                feature = "cpu-butter",
                feature = "cpu-zensim",
                feature = "cpu-iwssim"
            ))]
            MetricInner::Cpu(..) => Err(Error::Metric {
                kind: "cpu",
                message: "compute_features_srgb_u8 (feature export) is not implemented for \
                          Backend::Cpu"
                    .into(),
            }),
            #[cfg(feature = "zensim")]
            MetricInner::Zensim(m) => {
                // One pipeline pass: compute the regime-appropriate
                // feature vector, then derive the basic-block score
                // from the same data (matches what `compute_srgb_u8`
                // does internally on the basic block).
                let features = m
                    .compute_features_vec_srgb_u8(r, d)
                    .map_err(|e| Error::Metric {
                        kind: "zensim",
                        message: e.to_string(),
                    })?;
                // Re-derive the umbrella Score from the basic block so
                // the value is identical to `compute_srgb_u8` for the
                // same pair. We must score externally because the
                // opaque shim holds the weights privately.
                let score = m
                    .compute_srgb_u8(r, d)
                    .map(convert_score_zensim)
                    .map_err(|e| Error::Metric {
                        kind: "zensim",
                        message: e.to_string(),
                    })?;
                // NB: the second call above is cheap-ish — `Zensim`
                // re-runs the pyramid + features pass. If this becomes
                // a hot path, refactor `ZensimOpaque` to expose a
                // combined "score + features" entry point. For sweep
                // workloads (one call per cell, GPU dominates) the
                // double-dispatch overhead is negligible.
                Ok((score, features))
            }
            // All other metric variants: score normally, return an
            // empty feature vector. Pattern lets one call-site collect
            // features-when-present without branching on `kind()`.
            #[cfg(feature = "cvvdp")]
            MetricInner::Cvvdp(_) => self.compute_srgb_u8(r, d).map(|s| (s, Vec::new())),
            #[cfg(feature = "butter")]
            MetricInner::Butter(_) => self.compute_srgb_u8(r, d).map(|s| (s, Vec::new())),
            #[cfg(feature = "ssim2")]
            MetricInner::Ssim2(_) => self.compute_srgb_u8(r, d).map(|s| (s, Vec::new())),
            #[cfg(feature = "dssim")]
            MetricInner::Dssim(_) => self.compute_srgb_u8(r, d).map(|s| (s, Vec::new())),
            #[cfg(feature = "iwssim")]
            MetricInner::Iwssim(_) => self.compute_srgb_u8(r, d).map(|s| (s, Vec::new())),
        }
    }

    /// Score one reference / distorted pair from [`PixelSlice`]
    /// inputs. Per-crate conversion semantics apply — see each
    /// metric crate's `compute_pixels` docs.
    #[cfg(feature = "pixels")]
    pub fn compute_pixels(&mut self, r: PixelSlice<'_>, d: PixelSlice<'_>) -> Result<Score> {
        // Validate dims before dispatch — every per-crate impl does
        // this internally, but doing it here lets us surface a uniform
        // `DimensionMismatch` rather than a crate-specific error.
        let (w, h) = self.dims();
        if r.width() != w || r.rows() != h {
            return Err(Error::DimensionMismatch {
                expected: (w, h),
                got: (r.width(), r.rows()),
            });
        }
        if d.width() != w || d.rows() != h {
            return Err(Error::DimensionMismatch {
                expected: (w, h),
                got: (d.width(), d.rows()),
            });
        }
        match self {
            #[cfg(any(
                feature = "cpu-ssim2",
                feature = "cpu-cvvdp",
                feature = "cpu-dssim",
                feature = "cpu-butter",
                feature = "cpu-zensim",
                feature = "cpu-iwssim"
            ))]
            MetricInner::Cpu(s, _) => {
                // Optimized-CPU path (task #159 phase 3): convert both
                // PixelSlices to packed sRGB8 (strided-correct, one-line via
                // zenpixels-convert) then score on the native crate. HDR is
                // handled later via the cvvdp approach.
                let (w, h) = s.dims();
                let ref_buf = to_srgb_rgb8(&r, w, h)?;
                let dis_buf = to_srgb_rgb8(&d, w, h)?;
                s.compute_srgb_u8(&ref_buf, &dis_buf)
            }
            #[cfg(feature = "cvvdp")]
            MetricInner::Cvvdp(m) => {
                m.compute_pixels(r, d)
                    .map(convert_score)
                    .map_err(|e| Error::Metric {
                        kind: "cvvdp",
                        message: e.to_string(),
                    })
            }
            #[cfg(feature = "butter")]
            MetricInner::Butter(m) => {
                m.compute_pixels(r, d)
                    .map(convert_score_butter)
                    .map_err(|e| Error::Metric {
                        kind: "butter",
                        message: e.to_string(),
                    })
            }
            #[cfg(feature = "ssim2")]
            MetricInner::Ssim2(m) => m
                .compute_pixels(r, d)
                .map(convert_score_ssim2)
                .map_err(|e| Error::Metric {
                    kind: "ssim2",
                    message: e.to_string(),
                }),
            #[cfg(feature = "dssim")]
            MetricInner::Dssim(m) => m
                .compute_pixels(r, d)
                .map(convert_score_dssim)
                .map_err(|e| Error::Metric {
                    kind: "dssim",
                    message: e.to_string(),
                }),
            #[cfg(feature = "iwssim")]
            MetricInner::Iwssim(m) => {
                m.compute_pixels(r, d)
                    .map(convert_score_iwssim)
                    .map_err(|e| Error::Metric {
                        kind: "iwssim",
                        message: e.to_string(),
                    })
            }
            #[cfg(feature = "zensim")]
            MetricInner::Zensim(m) => {
                m.compute_pixels(r, d)
                    .map(convert_score_zensim)
                    .map_err(|e| Error::Metric {
                        kind: "zensim",
                        message: e.to_string(),
                    })
            }
        }
    }

    // -----------------------------------------------------------
    // Cached-reference API (Phase 2A — cvvdp + zensim + iwssim).
    //
    // For RD-search workloads where the same reference is scored
    // against many distortions, set_reference_srgb_u8 uploads the
    // ref once and pre-computes ref-side state. Subsequent
    // compute_with_reference_srgb_u8 calls skip the
    // ref-side pyramid build / blur cascade / IW weight maps.
    //
    // butter / ssim2 / dssim opaque shims don't yet expose the
    // cached-ref methods — Phase 2B adds them (tasks #45/#46 and
    // a sibling for dssim). Until then the umbrella returns
    // [`Error::Metric`] with a "not yet wired" message for those
    // three metrics, so callers can detect-and-fallback to
    // `compute_srgb_u8` without changing call shape.
    // -----------------------------------------------------------

    /// Cache the reference image's metric-side state on device.
    /// Subsequent [`Self::compute_with_reference_srgb_u8`]
    /// calls skip the reference's per-call upload + ref-side
    /// pre-processing.
    ///
    /// # Errors
    ///
    /// - [`Error::Metric`] when the underlying metric crate
    ///   doesn't yet wire cached-ref (butter / ssim2 / dssim
    ///   pending Phase 2B), or when the per-crate dispatch fails.
    pub fn set_reference_srgb_u8(&mut self, r: &[u8]) -> Result<()> {
        match self {
            #[cfg(any(
                feature = "cpu-ssim2",
                feature = "cpu-cvvdp",
                feature = "cpu-dssim",
                feature = "cpu-butter",
                feature = "cpu-zensim",
                feature = "cpu-iwssim"
            ))]
            MetricInner::Cpu(s, _) => {
                // True precompute warm path (2026-06-27): build the reference
                // XYB / pyramid / masks once via the native crate's warm API
                // (`cpu_dispatch::set_reference`), folded down from the
                // orchestrator's cpu_adapter — replaces the prior buffer-replay
                // stash. Length is validated inside `set_reference`.
                s.set_reference(r)
            }
            #[cfg(feature = "cvvdp")]
            MetricInner::Cvvdp(m) => m.set_reference_srgb_u8(r).map_err(|e| Error::Metric {
                kind: "cvvdp",
                message: e.to_string(),
            }),
            #[cfg(feature = "zensim")]
            MetricInner::Zensim(m) => m.set_reference_srgb_u8(r).map_err(|e| Error::Metric {
                kind: "zensim",
                message: e.to_string(),
            }),
            #[cfg(feature = "iwssim")]
            MetricInner::Iwssim(m) => m.set_reference_srgb_u8(r).map_err(|e| Error::Metric {
                kind: "iwssim",
                message: e.to_string(),
            }),
            #[cfg(feature = "butter")]
            MetricInner::Butter(m) => m.set_reference_srgb_u8(r).map_err(|e| Error::Metric {
                kind: "butter",
                message: e.to_string(),
            }),
            #[cfg(feature = "ssim2")]
            MetricInner::Ssim2(m) => m.set_reference_srgb_u8(r).map_err(|e| Error::Metric {
                kind: "ssim2",
                message: e.to_string(),
            }),
            #[cfg(feature = "dssim")]
            MetricInner::Dssim(m) => m.set_reference_srgb_u8(r).map_err(|e| Error::Metric {
                kind: "dssim",
                message: e.to_string(),
            }),
        }
    }

    /// Score a distorted candidate against the cached reference.
    /// Pre-requisite: [`Self::set_reference_srgb_u8`] must have
    /// been called (or [`Self::has_reference`] returns true).
    ///
    /// # Errors
    ///
    /// - Per-crate `NoCachedReference` when no reference is cached.
    pub fn compute_with_reference_srgb_u8(&mut self, d: &[u8]) -> Result<Score> {
        match self {
            #[cfg(any(
                feature = "cpu-ssim2",
                feature = "cpu-cvvdp",
                feature = "cpu-dssim",
                feature = "cpu-butter",
                feature = "cpu-zensim",
                feature = "cpu-iwssim"
            ))]
            MetricInner::Cpu(s, _) => {
                // True precompute warm path (2026-06-27): score against the
                // reference installed by `set_reference`, reusing its
                // precomputed XYB / pyramid
                // (`cpu_dispatch::compute_with_cached_reference`).
                s.compute_with_cached_reference(d)
            }
            #[cfg(feature = "cvvdp")]
            MetricInner::Cvvdp(m) => m
                .compute_with_reference_srgb_u8(d)
                .map(convert_score)
                .map_err(|e| Error::Metric {
                    kind: "cvvdp",
                    message: e.to_string(),
                }),
            #[cfg(feature = "zensim")]
            MetricInner::Zensim(m) => m
                .compute_with_reference_srgb_u8(d)
                .map(convert_score_zensim)
                .map_err(|e| Error::Metric {
                    kind: "zensim",
                    message: e.to_string(),
                }),
            #[cfg(feature = "iwssim")]
            MetricInner::Iwssim(m) => m
                .compute_with_reference_srgb_u8(d)
                .map(convert_score_iwssim)
                .map_err(|e| Error::Metric {
                    kind: "iwssim",
                    message: e.to_string(),
                }),
            #[cfg(feature = "butter")]
            MetricInner::Butter(m) => m
                .compute_with_reference_srgb_u8(d)
                .map(convert_score_butter)
                .map_err(|e| Error::Metric {
                    kind: "butter",
                    message: e.to_string(),
                }),
            #[cfg(feature = "ssim2")]
            MetricInner::Ssim2(m) => m
                .compute_with_reference_srgb_u8(d)
                .map(convert_score_ssim2)
                .map_err(|e| Error::Metric {
                    kind: "ssim2",
                    message: e.to_string(),
                }),
            #[cfg(feature = "dssim")]
            MetricInner::Dssim(m) => m
                .compute_with_reference_srgb_u8(d)
                .map(convert_score_dssim)
                .map_err(|e| Error::Metric {
                    kind: "dssim",
                    message: e.to_string(),
                }),
        }
    }

    /// Drop cached reference state. No-op for cvvdp/zensim whose
    /// opaque shims don't expose an explicit clear accessor —
    /// they implicitly overwrite on the next `set_reference_srgb_u8`.
    pub fn clear_reference(&mut self) {
        match self {
            #[cfg(any(
                feature = "cpu-ssim2",
                feature = "cpu-cvvdp",
                feature = "cpu-dssim",
                feature = "cpu-butter",
                feature = "cpu-zensim",
                feature = "cpu-iwssim"
            ))]
            MetricInner::Cpu(s, _) => s.clear_reference(),
            // cvvdp's set_reference_srgb_u8 overwrites prior state — no
            // explicit clear API on opaque (see pipeline.rs:4234).
            #[cfg(feature = "cvvdp")]
            MetricInner::Cvvdp(_) => {}
            // zensim has clear_reference on the typed pipeline but no
            // opaque accessor yet — Phase 2C can add it if needed.
            #[cfg(feature = "zensim")]
            MetricInner::Zensim(_) => {}
            #[cfg(feature = "iwssim")]
            MetricInner::Iwssim(m) => m.clear_reference(),
            #[cfg(feature = "butter")]
            MetricInner::Butter(m) => m.clear_reference(),
            #[cfg(feature = "ssim2")]
            MetricInner::Ssim2(m) => m.clear_reference(),
            #[cfg(feature = "dssim")]
            MetricInner::Dssim(m) => m.clear_reference(),
        }
    }

    /// Returns `true` if [`Self::set_reference_srgb_u8`] has been
    /// called and the cached reference state is still valid.
    ///
    /// cvvdp/zensim return `false` until they expose a `has_*`
    /// accessor (Phase 2C). The umbrella treats `false`
    /// conservatively: callers that branch on this should also
    /// handle the `NoCachedReference` error from
    /// [`Self::compute_with_reference_srgb_u8`].
    pub fn has_reference(&self) -> bool {
        match self {
            #[cfg(any(
                feature = "cpu-ssim2",
                feature = "cpu-cvvdp",
                feature = "cpu-dssim",
                feature = "cpu-butter",
                feature = "cpu-zensim",
                feature = "cpu-iwssim"
            ))]
            MetricInner::Cpu(s, _) => s.has_reference(),
            #[cfg(feature = "iwssim")]
            MetricInner::Iwssim(m) => m.has_reference(),
            #[cfg(feature = "butter")]
            MetricInner::Butter(m) => m.has_reference(),
            #[cfg(feature = "ssim2")]
            MetricInner::Ssim2(m) => m.has_reference(),
            #[cfg(feature = "dssim")]
            MetricInner::Dssim(m) => m.has_reference(),
            // cvvdp gained `has_reference` in task #79 (Mode E).
            // The strip-mode cache survives intervening dispatches
            // because it lives in dedicated buffers; the Full-mode
            // cache invalidates per `Cvvdp::warm_reference`'s
            // contract. Either way the accessor reflects the
            // currently-cached state.
            #[cfg(feature = "cvvdp")]
            MetricInner::Cvvdp(m) => m.has_reference(),
            #[cfg(feature = "zensim")]
            MetricInner::Zensim(_) => false,
        }
    }

    /// Drop this scorer **and** reclaim its pooled device memory back to
    /// the driver.
    ///
    /// Dropping a [`Metric`] alone returns its buffers to cubecl's pool
    /// (still resident); this convenience consumes `self`, runs the
    /// drop, then calls [`reclaim_pooled_vram`] for `backend` so the
    /// freed pages go back to the driver. Pass the same [`Backend`] the
    /// scorer was constructed with.
    ///
    /// Call this from the thread that owns the scorer (the cubecl pool
    /// is per-thread). Equivalent to `drop(metric); reclaim_pooled_vram(backend);`.
    pub fn release(self, backend: Backend) {
        drop(self);
        reclaim_pooled_vram(backend);
    }
}

// ---------------------------------------------------------------
// PixelSlice -> packed sRGB8 conversion for the optimized-CPU
// `compute_pixels` path (task #159 phase 3). Mirrors the per-crate
// `to_srgb_rgb8` helpers (e.g. `ssim2_gpu::opaque`): validate dims,
// fast-path an already-RGB8_SRGB slice, else convert per-row
// (strided-correct) via zenpixels-convert. The GPU arms convert inside
// their `-gpu` crate instead, so this only compiles for the CPU path.
// ---------------------------------------------------------------

#[cfg(feature = "pixels")]
fn to_srgb_rgb8(s: &PixelSlice<'_>, expected_w: u32, expected_h: u32) -> Result<Vec<u8>> {
    if s.width() != expected_w || s.rows() != expected_h {
        return Err(Error::DimensionMismatch {
            expected: (expected_w, expected_h),
            got: (s.width(), s.rows()),
        });
    }
    let target = zenpixels::PixelDescriptor::RGB8_SRGB;
    if s.descriptor() == target {
        return Ok(s.contiguous_bytes().into_owned());
    }
    convert_to_srgb_rgb8(s, target).map_err(|_| Error::DimensionMismatch {
        expected: (expected_w, expected_h),
        got: (s.width(), s.rows()),
    })
}

#[cfg(feature = "pixels")]
fn convert_to_srgb_rgb8(
    s: &PixelSlice<'_>,
    target: zenpixels::PixelDescriptor,
) -> core::result::Result<Vec<u8>, zenpixels_convert::ConvertError> {
    use zenpixels_convert::{ConvertPlan, convert_row};
    let plan = ConvertPlan::new(s.descriptor(), target).map_err(|e| e.decompose().0)?;
    let w = s.width();
    let h = s.rows();
    let row_bytes = (w as usize) * target.bytes_per_pixel();
    let mut out = vec![0u8; row_bytes * (h as usize)];
    for y in 0..h {
        let src_row = s.row(y);
        let start = (y as usize) * row_bytes;
        let dst_row = &mut out[start..start + row_bytes];
        convert_row(&plan, src_row, dst_row, w);
    }
    Ok(out)
}

// ---------------------------------------------------------------
// Per-crate Score conversion. Each metric crate has its own
// non_exhaustive Score struct (same field names, but distinct
// types). We rebuild the umbrella's Score from the public fields.
// ---------------------------------------------------------------

#[cfg(feature = "cvvdp")]
fn convert_score(s: cvvdp_gpu::Score) -> Score {
    Score {
        value: s.value,
        metric_name: s.metric_name,
        metric_version: s.metric_version,
    }
}

#[cfg(feature = "butter")]
fn convert_score_butter(s: butteraugli_gpu::Score) -> Score {
    Score {
        value: s.value,
        metric_name: s.metric_name,
        metric_version: s.metric_version,
    }
}

#[cfg(feature = "ssim2")]
fn convert_score_ssim2(s: ssim2_gpu::Score) -> Score {
    Score {
        value: s.value,
        metric_name: s.metric_name,
        metric_version: s.metric_version,
    }
}

#[cfg(feature = "dssim")]
fn convert_score_dssim(s: dssim_gpu::Score) -> Score {
    Score {
        value: s.value,
        metric_name: s.metric_name,
        metric_version: s.metric_version,
    }
}

#[cfg(feature = "iwssim")]
fn convert_score_iwssim(s: iwssim_gpu::Score) -> Score {
    Score {
        value: s.value,
        metric_name: s.metric_name,
        metric_version: s.metric_version,
    }
}

#[cfg(feature = "zensim")]
fn convert_score_zensim(s: zensim_gpu::Score) -> Score {
    Score {
        value: s.value,
        metric_name: s.metric_name,
        metric_version: s.metric_version,
    }
}

/// Split an interleaved linear-RGB f32 buffer (`[R,G,B, …]`, length `3·n`)
/// into three planar buffers for the planar linear-planes dispatch. Inlined
/// (rather than via `zenmetrics_gpu_core::deinterleave_rgb_f32`) so the
/// non-planar entry points compile in a CPU-only build with no GPU-metric
/// dependency on gpu-core. `which` labels the buffer in the error.
fn deinterleave_rgb(rgb: &[f32], which: &'static str) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    if !rgb.len().is_multiple_of(3) {
        return Err(Error::Metric {
            kind: "interleaved",
            message: format!(
                "{which} interleaved RGB length {} is not a multiple of 3",
                rgb.len()
            ),
        });
    }
    let n = rgb.len() / 3;
    let (mut r, mut g, mut b) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for px in rgb.chunks_exact(3) {
        r.push(px[0]);
        g.push(px[1]);
        b.push(px[2]);
    }
    Ok((r, g, b))
}

/// Error for the SSIM-family metrics' missing linear-planes HDR path —
/// see [`Metric::compute_from_linear_planes`].
#[cfg(any(feature = "ssim2", feature = "dssim", feature = "iwssim"))]
fn no_linear_planes_path(kind: &'static str) -> Error {
    Error::Metric {
        kind,
        message: "SSIM-family metric has no absolute-luminance linear-planes path; \
                  feed HDR via compute_srgb_u8(to_sdr_u8(..)) per hdr_feeding()"
            .to_string(),
    }
}

// ---------------------------------------------------------------
// Per-crate Backend conversion. Each crate has its own non_exhaustive
// Backend enum cfg-gated on the same `cuda`/`wgpu`/`cpu` features.
// Wgpu/cpu/hip variants are gated on metric features that may or may
// not exist; we surface a `BackendNotEnabled` error when the caller
// requests a backend that this build of the metric crate doesn't
// support.
//
// `Backend::Auto` resolves first (`b.resolve()`, which probes hardware)
// and re-enters the conversion with the concrete backend, so callers can
// hand `Auto` straight through. The per-crate enums keep the historical
// `Cpu` name for their cubecl-cpu variant, so the umbrella's renamed
// `Backend::CubeclCpu` maps onto `<crate>::Backend::Cpu`.
// ---------------------------------------------------------------

/// Map the umbrella [`Backend`] (which carries `Auto` + `CubeclCpu`) to the
/// shared [`zenmetrics_gpu_core::Backend`] every `*-gpu` opaque shim accepts.
/// Single source of truth; the six per-metric `*_backend` fns below are thin
/// re-typing delegators kept so their 40+ call sites stay put.
#[cfg(any(
    feature = "cvvdp",
    feature = "butter",
    feature = "ssim2",
    feature = "dssim",
    feature = "iwssim",
    feature = "zensim"
))]
pub(crate) fn gpu_backend(b: Backend) -> Result<zenmetrics_gpu_core::Backend> {
    use zenmetrics_gpu_core::Backend as Gpu;
    match b {
        Backend::Auto => gpu_backend(b.resolve()),
        #[cfg(feature = "cuda")]
        Backend::Cuda => Ok(Gpu::Cuda),
        #[cfg(feature = "wgpu")]
        Backend::Wgpu => Ok(Gpu::Wgpu),
        #[cfg(feature = "cpu")]
        Backend::CubeclCpu => Ok(Gpu::Cpu),
        // hip isn't surfaced as a per-crate Backend variant even when
        // `cubecl/hip` is enabled — the opaque shims only expose
        // cuda/wgpu/cpu. Surface as BackendNotEnabled.
        _ => Err(Error::BackendNotEnabled { backend: b.tag() }),
    }
}

#[cfg(feature = "cvvdp")]
pub(crate) fn cvvdp_backend(b: Backend) -> Result<cvvdp_gpu::Backend> {
    gpu_backend(b)
}

#[cfg(feature = "butter")]
pub(crate) fn butter_backend(b: Backend) -> Result<butteraugli_gpu::Backend> {
    gpu_backend(b)
}

#[cfg(feature = "ssim2")]
pub(crate) fn ssim2_backend(b: Backend) -> Result<ssim2_gpu::Backend> {
    gpu_backend(b)
}

#[cfg(feature = "dssim")]
pub(crate) fn dssim_backend(b: Backend) -> Result<dssim_gpu::Backend> {
    gpu_backend(b)
}

#[cfg(feature = "iwssim")]
pub(crate) fn iwssim_backend(b: Backend) -> Result<iwssim_gpu::Backend> {
    gpu_backend(b)
}

#[cfg(feature = "zensim")]
pub(crate) fn zensim_backend(b: Backend) -> Result<zensim_gpu::Backend> {
    gpu_backend(b)
}

// ---------------------------------------------------------------
// VRAM pool reclaim
// ---------------------------------------------------------------

/// Return pooled-but-unreferenced GPU memory for `backend` to the
/// driver.
///
/// **Why this exists.** Every metric variant holds its device buffers
/// as cubecl `Handle`s. Dropping a [`Metric`] drops those handles, but
/// cubecl *pools* the underlying device pages for reuse rather than
/// freeing them immediately — so the dropped metric's VRAM stays
/// resident (the plateau a leak-check sees as steady-state). To
/// actually hand the memory back to the driver you must call this
/// function **after** dropping the [`Metric`] (or any other live
/// instance on `backend`):
///
/// ```no_run
/// use zenmetrics_api::{Backend, Metric, MetricKind, MetricParams, reclaim_pooled_vram};
/// let m = Metric::new(MetricKind::Dssim, Backend::Cuda, 256, 256,
///     MetricParams::default_for(MetricKind::Dssim))?;
/// // ... score with `m` ...
/// drop(m);                       // handles → cubecl pool (still resident)
/// reclaim_pooled_vram(Backend::Cuda); // pool free pages → driver
/// # Ok::<(), zenmetrics_api::Error>(())
/// ```
///
/// [`Metric::release`] bundles the drop + reclaim into one call.
///
/// **Thread/stream scoped.** cubecl's CUDA pool is per-stream and the
/// stream is keyed on the *calling thread's* id, so this only reclaims
/// the pool owned by the thread that calls it — call it from the same
/// thread that dropped the metric.
///
/// **Do NOT call while a metric on `backend` is still alive on this
/// thread, and never between scores of a warm metric** — reclaiming
/// pages a live binding still references panics the cubecl allocator on
/// the next dispatch, and it discards the warm working set the next
/// score would have reused. Intended call sites: after a metric drops,
/// and at orchestrator metric-swap / idle.
///
/// No-op when `backend`'s Cargo feature is disabled in this build.
/// Best-effort: cubecl frees only the pages its allocator deems
/// reclaimable.
#[allow(unused_variables)]
// Intentional early-return cascade (reclaim once on the first enabled crate —
// they share the client). Which block is "last" varies by metric feature combo,
// so the needless_return false-positives under some combos (e.g. cvvdp+butter).
#[allow(clippy::needless_return)]
pub fn reclaim_pooled_vram(backend: Backend) {
    // All enabled metric crates obtain the same per-(device, thread)
    // cubecl client for a given backend, so calling any one crate's
    // reclaim cleans this thread's stream pool for that backend. Prefer
    // the first enabled crate in a fixed order; the conversion error
    // (backend disabled in that crate) is swallowed — `reclaim` is a
    // best-effort hint.
    #[cfg(feature = "cvvdp")]
    if let Ok(b) = cvvdp_backend(backend) {
        cvvdp_gpu::memory_mode::reclaim_pooled_vram(b);
        return;
    }
    #[cfg(feature = "butter")]
    if let Ok(b) = butter_backend(backend) {
        butteraugli_gpu::memory_mode::reclaim_pooled_vram(b);
        return;
    }
    #[cfg(feature = "ssim2")]
    if let Ok(b) = ssim2_backend(backend) {
        ssim2_gpu::memory_mode::reclaim_pooled_vram(b);
        return;
    }
    #[cfg(feature = "dssim")]
    if let Ok(b) = dssim_backend(backend) {
        dssim_gpu::memory_mode::reclaim_pooled_vram(b);
        return;
    }
    #[cfg(feature = "iwssim")]
    if let Ok(b) = iwssim_backend(backend) {
        iwssim_gpu::memory_mode::reclaim_pooled_vram(b);
        return;
    }
    #[cfg(feature = "zensim")]
    if let Ok(b) = zensim_backend(backend) {
        zensim_gpu::memory_mode::reclaim_pooled_vram(b);
    }
}

/// One-shot convenience: construct `kind` on `backend` at `width`×`height`
/// with that metric's default [`MetricParams`], score a single
/// (reference, distorted) sRGB-u8 pair, and drop the metric.
///
/// `reference_srgb_u8` / `distorted_srgb_u8` are tightly-packed RGB8,
/// `width * height * 3` bytes each.
///
/// This re-pays metric construction on **every** call — and on a GPU backend
/// that includes the per-process context floor (~181 ms) plus the first-kernel
/// JIT. For more than one pair, hold a [`Metric`] and reuse it
/// ([`Metric::compute_srgb_u8`], or [`Metric::set_reference_srgb_u8`] +
/// [`Metric::compute_with_reference_srgb_u8`] to amortize the
/// reference-side precompute across many distorted images), or a reusable
/// [`MetricSession`](crate::MetricSession). See the crate-level perf notes.
///
/// # Errors
///
/// Propagates construction errors from [`Metric::new`] and scoring errors
/// from [`Metric::compute_srgb_u8`].
///
/// ```no_run
/// use zenmetrics_api::{score_pair, Backend, MetricKind};
/// let (w, h) = (64u32, 64u32);
/// let n = (w * h * 3) as usize;
/// let reference = vec![128u8; n];
/// let distorted = vec![120u8; n];
/// let s = score_pair(MetricKind::Cvvdp, Backend::Cuda, w, h, &reference, &distorted)?;
/// println!("{} = {:.4}", s.metric_name, s.value);
/// # Ok::<(), zenmetrics_api::Error>(())
/// ```
pub fn score_pair(
    kind: MetricKind,
    backend: Backend,
    width: u32,
    height: u32,
    reference_srgb_u8: &[u8],
    distorted_srgb_u8: &[u8],
) -> Result<Score> {
    let mut metric = Metric::new(
        kind,
        backend,
        width,
        height,
        MetricParams::default_for(kind),
    )?;
    metric.compute_srgb_u8(reference_srgb_u8, distorted_srgb_u8)
}

/// Caller optimization priority for the score front doors / [`resolve_memory_mode`]
/// (task #159 phase 4). Default [`Priority::Speed`]. Only ever selects between
/// **score-safe** modes — priority never changes the score, only perf/memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Priority {
    /// Fastest score-safe mode that fits (default).
    #[default]
    Speed,
    /// Lowest-peak-memory score-safe mode (prefers strip on large images).
    Memory,
}

/// Whether a reference is scored once or reused across many distorted images
/// — the dominant axis in `benchmarks/mode_wall_2026-05-31.csv`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reuse {
    /// Single `(reference, distorted)` pair ([`score`] / [`score_encoded`]).
    OneOff,
    /// One reference, many distorted ([`warm_reference`]).
    Warm,
}

/// Resolve the Auto-safe optimal [`MemoryMode`] for `(kind, width, height,
/// reuse, priority)` (task #159 phase 4d) — **observable**, like
/// [`Backend::resolve_auto`], so callers can see what Auto picks.
///
/// Coarse rules seeded from `benchmarks/mode_wall_2026-05-31.csv` (#157): warm
/// reuse makes [`MemoryMode::Full`] fastest; one-off **large** images favor
/// [`MemoryMode::Strip`] (fastest one-off AND lowest peak); one-off **small**
/// images stay [`MemoryMode::Full`] (strip overhead isn't worth it);
/// [`Priority::Memory`] prefers strip on large warm images too. Only ever
/// returns **score-safe** modes (never the JOD-shifting cvvdp capped-pyramid);
/// metrics that can't strip a given input fall back to Full at their own
/// boundary, so the score is unchanged. The dense per-(metric, size) data fit
/// is #157 Phase B; this is the rule-based resolver.
pub fn resolve_memory_mode(
    kind: MetricKind,
    width: u32,
    height: u32,
    reuse: Reuse,
    priority: Priority,
) -> MemoryMode {
    // Per-metric refinement (some metrics strip-win at smaller sizes than
    // others) is #157 Phase B; the coarse size threshold is metric-agnostic.
    let _ = kind;
    let pixels = (width as u64) * (height as u64);
    /// One-off images at or below this stay Full (strip overhead > benefit).
    const SMALL_PX: u64 = 512 * 512;
    match reuse {
        Reuse::Warm => match priority {
            Priority::Speed => MemoryMode::Full,
            Priority::Memory if pixels > SMALL_PX => MemoryMode::Strip { h_body: None },
            Priority::Memory => MemoryMode::Full,
        },
        Reuse::OneOff if pixels <= SMALL_PX => MemoryMode::Full,
        Reuse::OneOff => MemoryMode::Strip { h_body: None },
    }
}

/// One-shot score of a decoded `(reference, distorted)` pair on `backend`
/// (task #159 phase 4) — the 90%-case front door: construct + score in a
/// single call, using the metric's Auto-safe optimal one-off mode.
///
/// Inputs are zenpixels [`PixelSlice`]s carrying their own format + dims, so
/// there is no `(w, h, &[u8])` mismatch footgun; strided and non-sRGB inputs
/// are converted per-call (HDR is handled later via the cvvdp approach). The
/// dims come from `reference`. For **encoded file bytes** use `score_encoded`
/// (phase 4); for **one reference, many distorted** use `warm_reference`
/// (phase 4); for a memory-priority or explicit [`MemoryMode`], construct via
/// [`Metric::new_with_memory_mode`].
///
/// `backend` accepts [`Backend::Auto`] (resolves to a GPU device if present,
/// else the optimized [`Backend::Cpu`] path); resolution never changes the
/// score.
#[cfg(feature = "pixels")]
pub fn score(
    kind: MetricKind,
    backend: Backend,
    reference: PixelSlice<'_>,
    distorted: PixelSlice<'_>,
) -> Result<Score> {
    // `PixelSlice` is a move-only borrow-wrapper (the pixel bytes live in the
    // caller's buffer, not here), so the front door takes it by value and
    // hands it straight to `compute_pixels` — no pixel copy.
    let (w, h) = (reference.width(), reference.rows());
    let mode = resolve_memory_mode(kind, w, h, Reuse::OneOff, Priority::Speed);
    let mut metric =
        Metric::new_with_memory_mode(kind, backend, w, h, MetricParams::default_for(kind), mode)?;
    metric.compute_pixels(reference, distorted)
}

/// A warmed scorer — one reference, many distorted (task #159 phase 4). The
/// reference is installed once; each [`Warm::score`] reuses it. Built by
/// [`warm_reference`]. **Backend-agnostic:** GPU backends use device-side
/// cached-reference, `Backend::Cpu` uses score-safe buffer-replay — either
/// way `score` returns the same value a fresh one-off would. Dims are fixed
/// at the reference.
#[cfg(feature = "pixels")]
pub struct Warm {
    metric: Metric,
    width: u32,
    height: u32,
}

#[cfg(feature = "pixels")]
impl Warm {
    /// Score `distorted` against the warmed reference (must match the
    /// reference's dimensions).
    pub fn score(&mut self, distorted: PixelSlice<'_>) -> Result<Score> {
        let dist = to_srgb_rgb8(&distorted, self.width, self.height)?;
        self.metric.compute_with_reference_srgb_u8(&dist)
    }

    /// The metric and dims this warm scorer was built for.
    pub fn kind(&self) -> MetricKind {
        self.metric.kind()
    }
}

/// Warm a `(kind, backend)` scorer with `reference` for repeated scoring
/// against many distorted images (task #159 phase 4) — the reuse-implied
/// front door, amortizing the reference work across calls. Identical API
/// for every backend; `Backend::Auto` resolves (GPU else optimized `Cpu`)
/// without changing the score.
#[cfg(feature = "pixels")]
pub fn warm_reference(
    kind: MetricKind,
    backend: Backend,
    reference: PixelSlice<'_>,
) -> Result<Warm> {
    let (w, h) = (reference.width(), reference.rows());
    let mode = resolve_memory_mode(kind, w, h, Reuse::Warm, Priority::Speed);
    let mut metric =
        Metric::new_with_memory_mode(kind, backend, w, h, MetricParams::default_for(kind), mode)?;
    let ref_bytes = to_srgb_rgb8(&reference, w, h)?;
    metric.set_reference_srgb_u8(&ref_bytes)?;
    Ok(Warm {
        metric,
        width: w,
        height: h,
    })
}

/// One-shot score of an **encoded** `(reference, distorted)` pair — PNG/JPEG
/// file bytes decoded internally to RGB8, then scored (task #159 phase 4c).
/// Both images must decode to the same dimensions. `backend` accepts
/// [`Backend::Auto`]. Requires the `encoded` feature (default-on); for
/// already-decoded pixels use [`score`].
#[cfg(feature = "encoded")]
pub fn score_encoded(
    kind: MetricKind,
    backend: Backend,
    reference: &[u8],
    distorted: &[u8],
) -> Result<Score> {
    let (rw, rh, ref_bytes) = decode_rgb8(reference, "reference")?;
    let (dw, dh, dist_bytes) = decode_rgb8(distorted, "distorted")?;
    if (rw, rh) != (dw, dh) {
        return Err(Error::DimensionMismatch {
            expected: (rw, rh),
            got: (dw, dh),
        });
    }
    let mode = resolve_memory_mode(kind, rw, rh, Reuse::OneOff, Priority::Speed);
    let mut metric =
        Metric::new_with_memory_mode(kind, backend, rw, rh, MetricParams::default_for(kind), mode)?;
    metric.compute_srgb_u8(&ref_bytes, &dist_bytes)
}

/// Decode encoded image `bytes` (PNG / JPEG) to packed sRGB8 `(w, h, bytes)`
/// via the `image` crate. `which` labels the side in error messages.
#[cfg(feature = "encoded")]
fn decode_rgb8(bytes: &[u8], which: &'static str) -> Result<(u32, u32, Vec<u8>)> {
    let img = image::load_from_memory(bytes).map_err(|e| Error::Metric {
        kind: "encoded",
        message: format!("decode {which} image: {e}"),
    })?;
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    Ok((w, h, rgb.into_raw()))
}

#[cfg(test)]
mod tests {
    //! Pure-logic coverage for [`resolve_memory_mode`] (task #159 phase 4d).
    //! No backend/feature needed — exercises the coarse rule table directly.
    use super::*;

    #[test]
    fn scores_single_and_accessors() {
        // `single` wraps a scalar Score as a one-entry Scores, no features.
        let s = Score {
            value: 87.5,
            metric_name: "ssim2",
            metric_version: "9.9.9",
        };
        let one = Scores::single(s);
        assert_eq!(one.scores.len(), 1);
        assert_eq!(one.scores[0].name, "ssim2");
        assert_eq!(one.primary(), 87.5);
        assert!(one.features.is_empty());
        assert_eq!(one.primary_score(), s);

        // Multi-scalar (butter-shaped): primary is scores[0]; get() by name.
        let multi = Scores {
            metric_name: "butter",
            metric_version: "1.2.3",
            scores: vec![
                NamedScore {
                    name: "max",
                    value: 4.0,
                },
                NamedScore {
                    name: "pnorm_3",
                    value: 1.5,
                },
            ],
            features: vec![],
        };
        assert_eq!(multi.primary(), 4.0);
        assert_eq!(multi.get("pnorm_3"), Some(1.5));
        assert_eq!(multi.get("max"), Some(4.0));
        assert_eq!(multi.get("nope"), None);
        assert_eq!(multi.primary_score().value, 4.0);
        assert_eq!(multi.primary_score().metric_name, "butter");
    }

    // 512×512 is the one-off Full/Strip threshold (inclusive Full).
    const SMALL: (u32, u32) = (512, 512);
    const BIG: (u32, u32) = (4096, 4096);

    #[test]
    fn warm_speed_is_always_full() {
        for &(w, h) in &[(64, 64), SMALL, BIG] {
            assert_eq!(
                resolve_memory_mode(MetricKind::Dssim, w, h, Reuse::Warm, Priority::Speed),
                MemoryMode::Full,
                "warm+Speed must stay Full at {w}x{h}"
            );
        }
    }

    #[test]
    fn warm_memory_strips_only_large() {
        // Small warm: Full already fits, strip overhead unwarranted.
        assert_eq!(
            resolve_memory_mode(
                MetricKind::Ssim2,
                SMALL.0,
                SMALL.1,
                Reuse::Warm,
                Priority::Memory
            ),
            MemoryMode::Full,
        );
        // Large warm + Memory: prefer the lower-peak strip.
        assert_eq!(
            resolve_memory_mode(
                MetricKind::Ssim2,
                BIG.0,
                BIG.1,
                Reuse::Warm,
                Priority::Memory
            ),
            MemoryMode::Strip { h_body: None },
        );
    }

    #[test]
    fn oneoff_small_is_full_either_priority() {
        for p in [Priority::Speed, Priority::Memory] {
            assert_eq!(
                resolve_memory_mode(MetricKind::Cvvdp, SMALL.0, SMALL.1, Reuse::OneOff, p),
                MemoryMode::Full,
                "one-off ≤512² must be Full ({p:?})"
            );
        }
    }

    #[test]
    fn oneoff_large_strips_either_priority() {
        for p in [Priority::Speed, Priority::Memory] {
            assert_eq!(
                resolve_memory_mode(MetricKind::Cvvdp, BIG.0, BIG.1, Reuse::OneOff, p),
                MemoryMode::Strip { h_body: None },
                "one-off large must strip ({p:?})"
            );
        }
    }

    #[test]
    fn oneoff_threshold_is_inclusive_full() {
        // Exactly 512×512 (== SMALL_PX) stays Full; one pixel over strips.
        assert_eq!(
            resolve_memory_mode(MetricKind::Iwssim, 512, 512, Reuse::OneOff, Priority::Speed),
            MemoryMode::Full,
        );
        assert_eq!(
            resolve_memory_mode(MetricKind::Iwssim, 512, 513, Reuse::OneOff, Priority::Speed),
            MemoryMode::Strip { h_body: None },
        );
    }

    #[test]
    fn resolver_never_returns_auto_or_tile() {
        // Every cell of the (reuse × priority × size) grid must land on a
        // concrete, score-safe mode — never Auto (unresolved) or Tile.
        for &(w, h) in &[(1, 1), (64, 64), SMALL, (513, 512), BIG, (8192, 8192)] {
            for reuse in [Reuse::OneOff, Reuse::Warm] {
                for prio in [Priority::Speed, Priority::Memory] {
                    let m = resolve_memory_mode(MetricKind::Zensim, w, h, reuse, prio);
                    assert!(
                        matches!(m, MemoryMode::Full | MemoryMode::Strip { .. }),
                        "{w}x{h} {reuse:?} {prio:?} resolved to {m:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn priority_default_is_speed() {
        assert_eq!(Priority::default(), Priority::Speed);
    }
}
