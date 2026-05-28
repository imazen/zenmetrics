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
//! | Iwssim  | *(no clean reference)*  | —              |
//!
//! Phase 8h (2026-05-27): the ssim2 row was switched from upstream
//! `ssimulacra2 0.5` to Imazen's SIMD-accelerated `fast-ssim2 0.8`.
//! Per-call scores may shift by atomic-add tolerance vs. the prior
//! implementation; the input shape is unchanged (sRGB u8 × 3-channel)
//! and the call surface (`compute` / `set_reference` / `compute_with_cached_reference`)
//! is untouched. See `docs/CPU_BACKENDS.md` for the rationale.
//!
//! Iwssim returns [`CpuAdapterError::Unavailable`] at construction time —
//! the chooser advances the OOM ladder to the next backend. See
//! `docs/CPU_BACKENDS.md` for the upstream-research notes.
//!
//! ## Cached-reference semantics
//!
//! Each CPU backend has a different relationship with reference reuse:
//!
//! - **cvvdp** has a true cached-reference path (`warm_reference` +
//!   `score_with_warm_ref`). Skips ~50% of the pipeline.
//! - **butteraugli** has no public cached-ref API today. The adapter
//!   falls back to the regular `compute` on cached-ref calls — still
//!   correct, just no speedup.
//! - **dssim-core** lets you `create_image(reference)` once and reuse;
//!   the adapter caches the prepared `DssimImage`.
//! - **fast-ssim2** has a true cached-ref path (`Ssimulacra2Reference::new`
//!   + `compare`) that skips ~50 % of the pipeline. The adapter wires it
//!   up so `set_reference` + `compute_with_cached_reference` are now
//!   amortised, not recompute. **Change vs. Phase 6's `ssimulacra2` 0.5
//!   wiring**: that crate had no precompute API and the adapter just
//!   stashed bytes for shape parity. fast-ssim2's `Ssimulacra2Reference`
//!   replaces that with a true warm path.
//! - **zensim** has no cached-ref API; falls back to recompute.
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
    /// Built without ANY CPU backend feature, or built without the
    /// specific feature for `metric`. The compute path returns
    /// [`CpuAdapterError::FeatureNotEnabled`].
    #[allow(dead_code)]
    FeatureDisabled(MetricKind),
    /// Metric has no CPU reference in this initial release (Iwssim).
    /// Calls return [`CpuAdapterError::Unavailable`].
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
    /// butteraugli has no public cached-ref path; we stash bytes for
    /// API-shape parity.
    cached_ref: Option<Vec<u8>>,
}

#[cfg(feature = "cpu-zensim")]
struct ZensimState {
    width: usize,
    height: usize,
    zensim: zensim::Zensim,
    /// zensim has no cached-ref API; stash bytes for parity.
    cached_ref: Option<Vec<u8>>,
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
    InvalidInputSize {
        expected: usize,
        got: usize,
    },
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
    /// `cpu-<metric>` feature is off, or `Err(Unavailable)` when the
    /// metric has no CPU reference in this release (Iwssim).
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
            MetricKind::Iwssim => {
                // No upstream CPU reference for IW-SSIM that's clean to
                // wire (see docs/CPU_BACKENDS.md). Surface as
                // Unavailable so the ladder advances rather than failing.
                Err(CpuAdapterError::Unavailable(metric))
            }
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
            #[cfg(feature = "cpu-butter")]
            CpuAdapterState::Butter(_) => false,
            #[cfg(feature = "cpu-zensim")]
            CpuAdapterState::Zensim(_) => false,
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
            CpuAdapterState::FeatureDisabled(k) => {
                Err(CpuAdapterError::FeatureNotEnabled(*k))
            }
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
                let precomputed = fast_ssim2::Ssimulacra2Reference::new(img.as_ref())
                    .map_err(|e| {
                        CpuAdapterError::Failed(format!(
                            "fast-ssim2 Ssimulacra2Reference::new: {e}"
                        ))
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
                s.cached_ref = Some(ref_bytes.to_vec());
                Ok(())
            }
            #[cfg(feature = "cpu-zensim")]
            CpuAdapterState::Zensim(s) => {
                s.cached_ref = Some(ref_bytes.to_vec());
                Ok(())
            }
            CpuAdapterState::FeatureDisabled(k) => {
                Err(CpuAdapterError::FeatureNotEnabled(*k))
            }
            CpuAdapterState::Unavailable(k) => Err(CpuAdapterError::Unavailable(*k)),
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
                    .compare(dist_img.as_ref())
                    .map_err(|e| CpuAdapterError::Failed(format!("fast-ssim2 compare: {e}")))?;
                Ok(make_score("ssim2", env!("CARGO_PKG_VERSION"), v))
            }
            #[cfg(feature = "cpu-dssim")]
            CpuAdapterState::Dssim(s) => {
                let r = s.cached_ref.as_ref().ok_or_else(|| {
                    CpuAdapterError::Failed("dssim: no cached reference; call set_reference first".into())
                })?;
                let dist_img = make_dssim_image(&s.dssim, dist_bytes, s.width, s.height)?;
                let (score, _maps) = s.dssim.compare(r, dist_img);
                Ok(make_score("dssim", dssim_cpu_version(), f64::from(score)))
            }
            #[cfg(feature = "cpu-butter")]
            CpuAdapterState::Butter(s) => {
                let r = s
                    .cached_ref
                    .as_ref()
                    .ok_or_else(|| {
                        CpuAdapterError::Failed("butter: no cached reference; call set_reference first".into())
                    })?
                    .clone();
                compute_butter(s, &r, dist_bytes)
            }
            #[cfg(feature = "cpu-zensim")]
            CpuAdapterState::Zensim(s) => {
                let r = s
                    .cached_ref
                    .as_ref()
                    .ok_or_else(|| {
                        CpuAdapterError::Failed("zensim: no cached reference; call set_reference first".into())
                    })?
                    .clone();
                compute_zensim(s, &r, dist_bytes)
            }
            CpuAdapterState::FeatureDisabled(k) => {
                Err(CpuAdapterError::FeatureNotEnabled(*k))
            }
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
    let c = cvvdp::Cvvdp::new(width, height, p)
        .map_err(|e| CpuAdapterError::Failed(e.to_string()))?;
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
fn compute_cvvdp(
    c: &mut cvvdp::Cvvdp,
    r: &[u8],
    d: &[u8],
) -> Result<Score, CpuAdapterError> {
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

/// Build an `imgref::ImgVec<[u8; 3]>` from interleaved sRGB-u8 bytes.
///
/// fast-ssim2's `ToLinearRgb` impl for `ImgRef<'_, [u8; 3]>` reads `[r, g, b]`
/// triplets directly — chunking and collecting once here is cheaper than
/// the per-pixel work the prior ssimulacra2 0.5 path did. The returned
/// `ImgVec` owns the pixel buffer; the caller turns it into a borrowing
/// `ImgRef` via `.as_ref()` at the call site.
#[cfg(feature = "cpu-ssim2")]
fn ssim2_image_ref(bytes: &[u8], w: usize, h: usize) -> imgref::ImgVec<[u8; 3]> {
    let pixels: Vec<[u8; 3]> = bytes
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    imgref::ImgVec::new(pixels, w, h)
}

#[cfg(feature = "cpu-ssim2")]
fn compute_ssim2(
    s: &mut Ssim2State,
    ref_bytes: &[u8],
    dist_bytes: &[u8],
) -> Result<Score, CpuAdapterError> {
    let ref_img = ssim2_image_ref(ref_bytes, s.width, s.height);
    let dist_img = ssim2_image_ref(dist_bytes, s.width, s.height);
    let v = fast_ssim2::compute_ssimulacra2(ref_img.as_ref(), dist_img.as_ref())
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
    use dssim_core::ToRGBAPLU;
    use imgref::ImgVec;
    let rgb: Vec<rgb::RGB<u8>> = bytes
        .chunks_exact(3)
        .map(|c| rgb::RGB::new(c[0], c[1], c[2]))
        .collect();
    let rgbplu = rgb.to_rgblu();
    let img = ImgVec::new(rgbplu, w, h);
    dssim
        .create_image(&img)
        .ok_or_else(|| CpuAdapterError::Failed("dssim_core create_image returned None".into()))
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
    let ref_rgb: Vec<rgb::RGB<u8>> = ref_bytes
        .chunks_exact(3)
        .map(|c| rgb::RGB::new(c[0], c[1], c[2]))
        .collect();
    let dist_rgb: Vec<rgb::RGB<u8>> = dist_bytes
        .chunks_exact(3)
        .map(|c| rgb::RGB::new(c[0], c[1], c[2]))
        .collect();
    let ref_img = ImgRef::new(&ref_rgb, s.width, s.height);
    let dist_img = ImgRef::new(&dist_rgb, s.width, s.height);
    let result = butteraugli::butteraugli(ref_img, dist_img, &s.params)
        .map_err(|e| CpuAdapterError::Failed(format!("butteraugli: {e:?}")))?;
    Ok(make_score("butter", env!("CARGO_PKG_VERSION"), result.score))
}

// ---------------------------------------------------------------------------
// zensim wiring
// ---------------------------------------------------------------------------

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
    // RgbSlice expects `&[[u8; 3]]`. Chunk + collect into owned vectors
    // — the alternative is an unsafe pointer cast that this codebase
    // forbids.
    let to_pix = |buf: &[u8]| -> Vec<[u8; 3]> {
        buf.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
    };
    let src = to_pix(ref_bytes);
    let dst = to_pix(dist_bytes);
    let ref_slice = zensim::RgbSlice::new(&src, s.width, s.height);
    let dist_slice = zensim::RgbSlice::new(&dst, s.width, s.height);
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
    fn iwssim_is_unavailable() {
        let params = MetricParams::try_default_for(MetricKind::Iwssim).unwrap();
        let r = CpuAdapter::new(MetricKind::Iwssim, 64, 64, &params);
        match r {
            Err(CpuAdapterError::Unavailable(MetricKind::Iwssim)) => {}
            Err(other) => panic!("expected Unavailable(Iwssim), got error {other:?}"),
            Ok(_) => panic!("expected Unavailable(Iwssim), got Ok"),
        }
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
