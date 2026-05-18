//! `Metric` enum + per-metric variant dispatch.

use crate::error::Error;
use crate::Result;

#[cfg(feature = "pixels")]
use zenpixels::PixelSlice;

// ---------------------------------------------------------------
// Backend
// ---------------------------------------------------------------

/// Selects the cubecl runtime that the underlying metric crate
/// dispatches against. Each variant corresponds to a Cargo feature on
/// the umbrella; variants for disabled features still surface a
/// [`Error::BackendNotEnabled`] at construction time so a single
/// `Backend::Cuda` constant in caller code keeps compiling regardless
/// of which backends are enabled in a given build.
///
/// This enum is **always exhaustive** (every cubecl backend has a
/// variant regardless of feature flags). The cfg-gating happens inside
/// `Metric::new` — disabled backends return `Err(BackendNotEnabled)`
/// at runtime. This keeps the consumer's match arms stable across
/// builds with different backend feature sets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// CUDA backend (NVIDIA, requires the `cuda` umbrella feature).
    Cuda,
    /// WGPU backend (cross-vendor, requires the `wgpu` umbrella feature).
    Wgpu,
    /// HIP backend (AMD ROCm, requires the `hip` umbrella feature).
    Hip,
    /// CPU reference backend via cubecl-cpu (requires the `cpu`
    /// umbrella feature). Note that several metric crates rely on
    /// `Atomic<f32>` operations that cubecl-cpu does not support —
    /// kernels may panic at first dispatch even when this variant is
    /// accepted by the constructor. See each metric crate's
    /// `Backend::Cpu` doc.
    Cpu,
}

impl Backend {
    fn tag(self) -> &'static str {
        match self {
            Backend::Cuda => "cuda",
            Backend::Wgpu => "wgpu",
            Backend::Hip => "hip",
            Backend::Cpu => "cpu",
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
    /// [`ssim2_gpu::Ssim2Params`] passthrough.
    #[cfg(feature = "ssim2")]
    Ssim2(ssim2_gpu::Ssim2Params),
    /// [`dssim_gpu::DssimParams`] passthrough.
    #[cfg(feature = "dssim")]
    Dssim(dssim_gpu::DssimParams),
    /// [`iwssim_gpu::IwssimParams`] passthrough.
    #[cfg(feature = "iwssim")]
    Iwssim(iwssim_gpu::IwssimParams),
    /// [`zensim_gpu::ZensimParams`] passthrough.
    #[cfg(feature = "zensim")]
    Zensim(zensim_gpu::ZensimParams),
}

impl MetricParams {
    /// Default-construct the params variant matching `kind`. Panics if
    /// the requested metric's Cargo feature is disabled in this build
    /// — callers should match the build's enabled metrics or use
    /// `try_default_for` instead.
    pub fn default_for(kind: MetricKind) -> Self {
        Self::try_default_for(kind).unwrap_or_else(|e| panic!("{e}"))
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
            #[cfg(feature = "ssim2")]
            MetricKind::Ssim2 => Ok(Self::Ssim2(ssim2_gpu::Ssim2Params::default())),
            #[cfg(feature = "dssim")]
            MetricKind::Dssim => Ok(Self::Dssim(dssim_gpu::DssimParams::DEFAULT)),
            #[cfg(feature = "iwssim")]
            MetricKind::Iwssim => Ok(Self::Iwssim(iwssim_gpu::IwssimParams::DEFAULT)),
            #[cfg(feature = "zensim")]
            MetricKind::Zensim => Ok(Self::Zensim(zensim_gpu::ZensimParams::new())),
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
pub enum Metric {
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
}

impl Metric {
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
        match kind {
            #[cfg(feature = "cvvdp")]
            MetricKind::Cvvdp => {
                let p = match params {
                    MetricParams::Cvvdp(p) => p,
                    _ => panic!("MetricParams variant mismatch (expected Cvvdp)"),
                };
                let b = cvvdp_backend(backend)?;
                cvvdp_gpu::CvvdpOpaque::new(b, width, height, p)
                    .map(Metric::Cvvdp)
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
                    .map(Metric::Butter)
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
                    .map(Metric::Ssim2)
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
                    .map(Metric::Dssim)
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
                    .map(Metric::Iwssim)
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
                    .map(Metric::Zensim)
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
            #[cfg(feature = "cvvdp")]
            Metric::Cvvdp(_) => MetricKind::Cvvdp,
            #[cfg(feature = "butter")]
            Metric::Butter(_) => MetricKind::Butter,
            #[cfg(feature = "ssim2")]
            Metric::Ssim2(_) => MetricKind::Ssim2,
            #[cfg(feature = "dssim")]
            Metric::Dssim(_) => MetricKind::Dssim,
            #[cfg(feature = "iwssim")]
            Metric::Iwssim(_) => MetricKind::Iwssim,
            #[cfg(feature = "zensim")]
            Metric::Zensim(_) => MetricKind::Zensim,
        }
    }

    /// The configured `(width, height)`.
    pub fn dims(&self) -> (u32, u32) {
        match self {
            #[cfg(feature = "cvvdp")]
            Metric::Cvvdp(m) => m.dims(),
            #[cfg(feature = "butter")]
            Metric::Butter(m) => m.dims(),
            #[cfg(feature = "ssim2")]
            Metric::Ssim2(m) => m.dims(),
            #[cfg(feature = "dssim")]
            Metric::Dssim(m) => m.dims(),
            #[cfg(feature = "iwssim")]
            Metric::Iwssim(m) => m.dims(),
            #[cfg(feature = "zensim")]
            Metric::Zensim(m) => m.dims(),
        }
    }

    /// Score one reference / distorted pair of packed sRGB `R, G, B,
    /// R, G, B, …` buffers (length `width × height × 3`).
    pub fn compute_srgb_u8(&mut self, r: &[u8], d: &[u8]) -> Result<Score> {
        match self {
            #[cfg(feature = "cvvdp")]
            Metric::Cvvdp(m) => m
                .compute_srgb_u8(r, d)
                .map(convert_score)
                .map_err(|e| Error::Metric {
                    kind: "cvvdp",
                    message: e.to_string(),
                }),
            #[cfg(feature = "butter")]
            Metric::Butter(m) => m
                .compute_srgb_u8(r, d)
                .map(convert_score_butter)
                .map_err(|e| Error::Metric {
                    kind: "butter",
                    message: e.to_string(),
                }),
            #[cfg(feature = "ssim2")]
            Metric::Ssim2(m) => m
                .compute_srgb_u8(r, d)
                .map(convert_score_ssim2)
                .map_err(|e| Error::Metric {
                    kind: "ssim2",
                    message: e.to_string(),
                }),
            #[cfg(feature = "dssim")]
            Metric::Dssim(m) => m
                .compute_srgb_u8(r, d)
                .map(convert_score_dssim)
                .map_err(|e| Error::Metric {
                    kind: "dssim",
                    message: e.to_string(),
                }),
            #[cfg(feature = "iwssim")]
            Metric::Iwssim(m) => m
                .compute_srgb_u8(r, d)
                .map(convert_score_iwssim)
                .map_err(|e| Error::Metric {
                    kind: "iwssim",
                    message: e.to_string(),
                }),
            #[cfg(feature = "zensim")]
            Metric::Zensim(m) => m
                .compute_srgb_u8(r, d)
                .map(convert_score_zensim)
                .map_err(|e| Error::Metric {
                    kind: "zensim",
                    message: e.to_string(),
                }),
        }
    }

    /// Score one reference / distorted pair from [`PixelSlice`]
    /// inputs. Per-crate conversion semantics apply — see each
    /// metric crate's `compute_pixels` docs.
    #[cfg(feature = "pixels")]
    pub fn compute_pixels(
        &mut self,
        r: PixelSlice<'_>,
        d: PixelSlice<'_>,
    ) -> Result<Score> {
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
            #[cfg(feature = "cvvdp")]
            Metric::Cvvdp(m) => m
                .compute_pixels(r, d)
                .map(convert_score)
                .map_err(|e| Error::Metric {
                    kind: "cvvdp",
                    message: e.to_string(),
                }),
            #[cfg(feature = "butter")]
            Metric::Butter(m) => m
                .compute_pixels(r, d)
                .map(convert_score_butter)
                .map_err(|e| Error::Metric {
                    kind: "butter",
                    message: e.to_string(),
                }),
            #[cfg(feature = "ssim2")]
            Metric::Ssim2(m) => m
                .compute_pixels(r, d)
                .map(convert_score_ssim2)
                .map_err(|e| Error::Metric {
                    kind: "ssim2",
                    message: e.to_string(),
                }),
            #[cfg(feature = "dssim")]
            Metric::Dssim(m) => m
                .compute_pixels(r, d)
                .map(convert_score_dssim)
                .map_err(|e| Error::Metric {
                    kind: "dssim",
                    message: e.to_string(),
                }),
            #[cfg(feature = "iwssim")]
            Metric::Iwssim(m) => m
                .compute_pixels(r, d)
                .map(convert_score_iwssim)
                .map_err(|e| Error::Metric {
                    kind: "iwssim",
                    message: e.to_string(),
                }),
            #[cfg(feature = "zensim")]
            Metric::Zensim(m) => m
                .compute_pixels(r, d)
                .map(convert_score_zensim)
                .map_err(|e| Error::Metric {
                    kind: "zensim",
                    message: e.to_string(),
                }),
        }
    }
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

// ---------------------------------------------------------------
// Per-crate Backend conversion. Each crate has its own non_exhaustive
// Backend enum cfg-gated on the same `cuda`/`wgpu`/`cpu` features.
// Wgpu/cpu/hip variants are gated on metric features that may or may
// not exist; we surface a `BackendNotEnabled` error when the caller
// requests a backend that this build of the metric crate doesn't
// support.
// ---------------------------------------------------------------

#[cfg(feature = "cvvdp")]
fn cvvdp_backend(b: Backend) -> Result<cvvdp_gpu::Backend> {
    match b {
        #[cfg(feature = "cuda")]
        Backend::Cuda => Ok(cvvdp_gpu::Backend::Cuda),
        #[cfg(feature = "wgpu")]
        Backend::Wgpu => Ok(cvvdp_gpu::Backend::Wgpu),
        #[cfg(feature = "cpu")]
        Backend::Cpu => Ok(cvvdp_gpu::Backend::Cpu),
        // hip isn't surfaced as a variant on the per-crate Backend
        // even when `cubecl/hip` is enabled — cvvdp-gpu's opaque
        // shim only exposes cuda/wgpu/cpu. Surface as BackendNotEnabled.
        _ => Err(Error::BackendNotEnabled { backend: b.tag() }),
    }
}

#[cfg(feature = "butter")]
fn butter_backend(b: Backend) -> Result<butteraugli_gpu::Backend> {
    match b {
        #[cfg(feature = "cuda")]
        Backend::Cuda => Ok(butteraugli_gpu::Backend::Cuda),
        #[cfg(feature = "wgpu")]
        Backend::Wgpu => Ok(butteraugli_gpu::Backend::Wgpu),
        #[cfg(feature = "cpu")]
        Backend::Cpu => Ok(butteraugli_gpu::Backend::Cpu),
        _ => Err(Error::BackendNotEnabled { backend: b.tag() }),
    }
}

#[cfg(feature = "ssim2")]
fn ssim2_backend(b: Backend) -> Result<ssim2_gpu::Backend> {
    match b {
        #[cfg(feature = "cuda")]
        Backend::Cuda => Ok(ssim2_gpu::Backend::Cuda),
        #[cfg(feature = "wgpu")]
        Backend::Wgpu => Ok(ssim2_gpu::Backend::Wgpu),
        #[cfg(feature = "cpu")]
        Backend::Cpu => Ok(ssim2_gpu::Backend::Cpu),
        _ => Err(Error::BackendNotEnabled { backend: b.tag() }),
    }
}

#[cfg(feature = "dssim")]
fn dssim_backend(b: Backend) -> Result<dssim_gpu::Backend> {
    match b {
        #[cfg(feature = "cuda")]
        Backend::Cuda => Ok(dssim_gpu::Backend::Cuda),
        #[cfg(feature = "wgpu")]
        Backend::Wgpu => Ok(dssim_gpu::Backend::Wgpu),
        #[cfg(feature = "cpu")]
        Backend::Cpu => Ok(dssim_gpu::Backend::Cpu),
        _ => Err(Error::BackendNotEnabled { backend: b.tag() }),
    }
}

#[cfg(feature = "iwssim")]
fn iwssim_backend(b: Backend) -> Result<iwssim_gpu::Backend> {
    match b {
        #[cfg(feature = "cuda")]
        Backend::Cuda => Ok(iwssim_gpu::Backend::Cuda),
        #[cfg(feature = "wgpu")]
        Backend::Wgpu => Ok(iwssim_gpu::Backend::Wgpu),
        #[cfg(feature = "cpu")]
        Backend::Cpu => Ok(iwssim_gpu::Backend::Cpu),
        _ => Err(Error::BackendNotEnabled { backend: b.tag() }),
    }
}

#[cfg(feature = "zensim")]
fn zensim_backend(b: Backend) -> Result<zensim_gpu::Backend> {
    match b {
        #[cfg(feature = "cuda")]
        Backend::Cuda => Ok(zensim_gpu::Backend::Cuda),
        #[cfg(feature = "wgpu")]
        Backend::Wgpu => Ok(zensim_gpu::Backend::Wgpu),
        #[cfg(feature = "cpu")]
        Backend::Cpu => Ok(zensim_gpu::Backend::Cpu),
        _ => Err(Error::BackendNotEnabled { backend: b.tag() }),
    }
}
