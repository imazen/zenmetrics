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
            MetricKind::Zensim => Ok(Self::Zensim(zensim_gpu::ZensimParams::default_weights())),
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
        match kind {
            #[cfg(feature = "cvvdp")]
            MetricKind::Cvvdp => {
                let p = match params {
                    MetricParams::Cvvdp(p) => p,
                    _ => panic!("MetricParams variant mismatch (expected Cvvdp)"),
                };
                let b = cvvdp_backend(backend)?;
                cvvdp_gpu::CvvdpOpaque::new_with_memory_mode(b, width, height, p, mode.into())
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
                butteraugli_gpu::ButteraugliOpaque::new_with_memory_mode(
                    b,
                    width,
                    height,
                    p,
                    mode.into(),
                )
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
                ssim2_gpu::Ssim2Opaque::new_with_memory_mode(b, width, height, p, mode.into())
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
                dssim_gpu::DssimOpaque::new_with_memory_mode(b, width, height, p, mode.into())
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
                iwssim_gpu::IwssimOpaque::new_with_memory_mode(b, width, height, p, mode.into())
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
                zensim_gpu::ZensimOpaque::new_with_memory_mode(b, width, height, p, mode.into())
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
            Metric::Cvvdp(m) => {
                m.compute_srgb_u8(r, d)
                    .map(convert_score)
                    .map_err(|e| Error::Metric {
                        kind: "cvvdp",
                        message: e.to_string(),
                    })
            }
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
            Metric::Cvvdp(m) => m
                .compute_handles(ref_h, dis_h)
                .map(convert_score)
                .map_err(|e| Error::Metric {
                    kind: "cvvdp",
                    message: e.to_string(),
                }),
            #[cfg(feature = "butter")]
            Metric::Butter(m) => m
                .compute_handles(ref_h, dis_h)
                .map(convert_score_butter)
                .map_err(|e| Error::Metric {
                    kind: "butter",
                    message: e.to_string(),
                }),
            #[cfg(feature = "ssim2")]
            Metric::Ssim2(m) => m
                .compute_handles(ref_h, dis_h)
                .map(convert_score_ssim2)
                .map_err(|e| Error::Metric {
                    kind: "ssim2",
                    message: e.to_string(),
                }),
            #[cfg(feature = "dssim")]
            Metric::Dssim(m) => m
                .compute_handles(ref_h, dis_h)
                .map(convert_score_dssim)
                .map_err(|e| Error::Metric {
                    kind: "dssim",
                    message: e.to_string(),
                }),
            #[cfg(feature = "iwssim")]
            Metric::Iwssim(m) => m
                .compute_handles(ref_h, dis_h)
                .map(convert_score_iwssim)
                .map_err(|e| Error::Metric {
                    kind: "iwssim",
                    message: e.to_string(),
                }),
            #[cfg(feature = "zensim")]
            Metric::Zensim(_) => {
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
            #[cfg(feature = "zensim")]
            Metric::Zensim(m) => {
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
            Metric::Cvvdp(_) => self.compute_srgb_u8(r, d).map(|s| (s, Vec::new())),
            #[cfg(feature = "butter")]
            Metric::Butter(_) => self.compute_srgb_u8(r, d).map(|s| (s, Vec::new())),
            #[cfg(feature = "ssim2")]
            Metric::Ssim2(_) => self.compute_srgb_u8(r, d).map(|s| (s, Vec::new())),
            #[cfg(feature = "dssim")]
            Metric::Dssim(_) => self.compute_srgb_u8(r, d).map(|s| (s, Vec::new())),
            #[cfg(feature = "iwssim")]
            Metric::Iwssim(_) => self.compute_srgb_u8(r, d).map(|s| (s, Vec::new())),
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
            #[cfg(feature = "cvvdp")]
            Metric::Cvvdp(m) => {
                m.compute_pixels(r, d)
                    .map(convert_score)
                    .map_err(|e| Error::Metric {
                        kind: "cvvdp",
                        message: e.to_string(),
                    })
            }
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

    // -----------------------------------------------------------
    // Cached-reference API (Phase 2A — cvvdp + zensim + iwssim).
    //
    // For RD-search workloads where the same reference is scored
    // against many distortions, set_reference_srgb_u8 uploads the
    // ref once and pre-computes ref-side state. Subsequent
    // compute_with_cached_reference_srgb_u8 calls skip the
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
    /// Subsequent [`Self::compute_with_cached_reference_srgb_u8`]
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
            #[cfg(feature = "cvvdp")]
            Metric::Cvvdp(m) => m.warm_reference_srgb(r).map_err(|e| Error::Metric {
                kind: "cvvdp",
                message: e.to_string(),
            }),
            #[cfg(feature = "zensim")]
            Metric::Zensim(m) => m.set_reference_srgb_u8(r).map_err(|e| Error::Metric {
                kind: "zensim",
                message: e.to_string(),
            }),
            #[cfg(feature = "iwssim")]
            Metric::Iwssim(m) => m.set_reference_srgb_u8(r).map_err(|e| Error::Metric {
                kind: "iwssim",
                message: e.to_string(),
            }),
            #[cfg(feature = "butter")]
            Metric::Butter(m) => m.set_reference_srgb_u8(r).map_err(|e| Error::Metric {
                kind: "butter",
                message: e.to_string(),
            }),
            #[cfg(feature = "ssim2")]
            Metric::Ssim2(m) => m.set_reference_srgb_u8(r).map_err(|e| Error::Metric {
                kind: "ssim2",
                message: e.to_string(),
            }),
            #[cfg(feature = "dssim")]
            Metric::Dssim(m) => m.set_reference_srgb_u8(r).map_err(|e| Error::Metric {
                kind: "dssim",
                message: e.to_string(),
            }),
        }
    }

    /// Score a distorted candidate against the cached reference.
    /// Pre-requisite: [`Self::set_reference_srgb_u8`] must have
    /// been called (or [`Self::has_cached_reference`] returns true).
    ///
    /// # Errors
    ///
    /// - Per-crate `NoCachedReference` when no reference is cached.
    pub fn compute_with_cached_reference_srgb_u8(&mut self, d: &[u8]) -> Result<Score> {
        match self {
            #[cfg(feature = "cvvdp")]
            Metric::Cvvdp(m) => m
                .compute_with_warm_ref_srgb(d, None)
                .map(convert_score)
                .map_err(|e| Error::Metric {
                    kind: "cvvdp",
                    message: e.to_string(),
                }),
            #[cfg(feature = "zensim")]
            Metric::Zensim(m) => m
                .compute_with_cached_reference_score_srgb_u8(d)
                .map(convert_score_zensim)
                .map_err(|e| Error::Metric {
                    kind: "zensim",
                    message: e.to_string(),
                }),
            #[cfg(feature = "iwssim")]
            Metric::Iwssim(m) => m
                .compute_with_cached_reference_srgb_u8(d)
                .map(convert_score_iwssim)
                .map_err(|e| Error::Metric {
                    kind: "iwssim",
                    message: e.to_string(),
                }),
            #[cfg(feature = "butter")]
            Metric::Butter(m) => m
                .compute_with_cached_reference_srgb_u8(d)
                .map(convert_score_butter)
                .map_err(|e| Error::Metric {
                    kind: "butter",
                    message: e.to_string(),
                }),
            #[cfg(feature = "ssim2")]
            Metric::Ssim2(m) => m
                .compute_with_cached_reference_srgb_u8(d)
                .map(convert_score_ssim2)
                .map_err(|e| Error::Metric {
                    kind: "ssim2",
                    message: e.to_string(),
                }),
            #[cfg(feature = "dssim")]
            Metric::Dssim(m) => m
                .compute_with_cached_reference_srgb_u8(d)
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
            // cvvdp's warm_reference_srgb overwrites prior state — no
            // explicit clear API on opaque (see pipeline.rs:4234).
            #[cfg(feature = "cvvdp")]
            Metric::Cvvdp(_) => {}
            // zensim has clear_reference on the typed pipeline but no
            // opaque accessor yet — Phase 2C can add it if needed.
            #[cfg(feature = "zensim")]
            Metric::Zensim(_) => {}
            #[cfg(feature = "iwssim")]
            Metric::Iwssim(m) => m.clear_reference(),
            #[cfg(feature = "butter")]
            Metric::Butter(m) => m.clear_reference(),
            #[cfg(feature = "ssim2")]
            Metric::Ssim2(m) => m.clear_reference(),
            #[cfg(feature = "dssim")]
            Metric::Dssim(m) => m.clear_reference(),
        }
    }

    /// Returns `true` if [`Self::set_reference_srgb_u8`] has been
    /// called and the cached reference state is still valid.
    ///
    /// cvvdp/zensim return `false` until they expose a `has_*`
    /// accessor (Phase 2C). The umbrella treats `false`
    /// conservatively: callers that branch on this should also
    /// handle the `NoCachedReference` error from
    /// [`Self::compute_with_cached_reference_srgb_u8`].
    pub fn has_cached_reference(&self) -> bool {
        match self {
            #[cfg(feature = "iwssim")]
            Metric::Iwssim(m) => m.has_cached_reference(),
            #[cfg(feature = "butter")]
            Metric::Butter(m) => m.has_cached_reference(),
            #[cfg(feature = "ssim2")]
            Metric::Ssim2(m) => m.has_cached_reference(),
            #[cfg(feature = "dssim")]
            Metric::Dssim(m) => m.has_cached_reference(),
            // cvvdp gained `has_warm_reference` in task #79 (Mode E).
            // The strip-mode cache survives intervening dispatches
            // because it lives in dedicated buffers; the Full-mode
            // cache invalidates per `Cvvdp::warm_reference`'s
            // contract. Either way the accessor reflects the
            // currently-cached state.
            #[cfg(feature = "cvvdp")]
            Metric::Cvvdp(m) => m.has_warm_reference(),
            #[cfg(feature = "zensim")]
            Metric::Zensim(_) => false,
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
//
// `Backend::Auto` resolves first (`b.resolve()`, which probes hardware)
// and re-enters the conversion with the concrete backend, so callers can
// hand `Auto` straight through. The per-crate enums keep the historical
// `Cpu` name for their cubecl-cpu variant, so the umbrella's renamed
// `Backend::CubeclCpu` maps onto `<crate>::Backend::Cpu`.
// ---------------------------------------------------------------

#[cfg(feature = "cvvdp")]
pub(crate) fn cvvdp_backend(b: Backend) -> Result<cvvdp_gpu::Backend> {
    match b {
        Backend::Auto => cvvdp_backend(b.resolve()),
        #[cfg(feature = "cuda")]
        Backend::Cuda => Ok(cvvdp_gpu::Backend::Cuda),
        #[cfg(feature = "wgpu")]
        Backend::Wgpu => Ok(cvvdp_gpu::Backend::Wgpu),
        #[cfg(feature = "cpu")]
        Backend::CubeclCpu => Ok(cvvdp_gpu::Backend::Cpu),
        // hip isn't surfaced as a variant on the per-crate Backend
        // even when `cubecl/hip` is enabled — cvvdp-gpu's opaque
        // shim only exposes cuda/wgpu/cpu. Surface as BackendNotEnabled.
        _ => Err(Error::BackendNotEnabled { backend: b.tag() }),
    }
}

#[cfg(feature = "butter")]
pub(crate) fn butter_backend(b: Backend) -> Result<butteraugli_gpu::Backend> {
    match b {
        Backend::Auto => butter_backend(b.resolve()),
        #[cfg(feature = "cuda")]
        Backend::Cuda => Ok(butteraugli_gpu::Backend::Cuda),
        #[cfg(feature = "wgpu")]
        Backend::Wgpu => Ok(butteraugli_gpu::Backend::Wgpu),
        #[cfg(feature = "cpu")]
        Backend::CubeclCpu => Ok(butteraugli_gpu::Backend::Cpu),
        _ => Err(Error::BackendNotEnabled { backend: b.tag() }),
    }
}

#[cfg(feature = "ssim2")]
pub(crate) fn ssim2_backend(b: Backend) -> Result<ssim2_gpu::Backend> {
    match b {
        Backend::Auto => ssim2_backend(b.resolve()),
        #[cfg(feature = "cuda")]
        Backend::Cuda => Ok(ssim2_gpu::Backend::Cuda),
        #[cfg(feature = "wgpu")]
        Backend::Wgpu => Ok(ssim2_gpu::Backend::Wgpu),
        #[cfg(feature = "cpu")]
        Backend::CubeclCpu => Ok(ssim2_gpu::Backend::Cpu),
        _ => Err(Error::BackendNotEnabled { backend: b.tag() }),
    }
}

#[cfg(feature = "dssim")]
pub(crate) fn dssim_backend(b: Backend) -> Result<dssim_gpu::Backend> {
    match b {
        Backend::Auto => dssim_backend(b.resolve()),
        #[cfg(feature = "cuda")]
        Backend::Cuda => Ok(dssim_gpu::Backend::Cuda),
        #[cfg(feature = "wgpu")]
        Backend::Wgpu => Ok(dssim_gpu::Backend::Wgpu),
        #[cfg(feature = "cpu")]
        Backend::CubeclCpu => Ok(dssim_gpu::Backend::Cpu),
        _ => Err(Error::BackendNotEnabled { backend: b.tag() }),
    }
}

#[cfg(feature = "iwssim")]
pub(crate) fn iwssim_backend(b: Backend) -> Result<iwssim_gpu::Backend> {
    match b {
        Backend::Auto => iwssim_backend(b.resolve()),
        #[cfg(feature = "cuda")]
        Backend::Cuda => Ok(iwssim_gpu::Backend::Cuda),
        #[cfg(feature = "wgpu")]
        Backend::Wgpu => Ok(iwssim_gpu::Backend::Wgpu),
        #[cfg(feature = "cpu")]
        Backend::CubeclCpu => Ok(iwssim_gpu::Backend::Cpu),
        _ => Err(Error::BackendNotEnabled { backend: b.tag() }),
    }
}

#[cfg(feature = "zensim")]
pub(crate) fn zensim_backend(b: Backend) -> Result<zensim_gpu::Backend> {
    match b {
        Backend::Auto => zensim_backend(b.resolve()),
        #[cfg(feature = "cuda")]
        Backend::Cuda => Ok(zensim_gpu::Backend::Cuda),
        #[cfg(feature = "wgpu")]
        Backend::Wgpu => Ok(zensim_gpu::Backend::Wgpu),
        #[cfg(feature = "cpu")]
        Backend::CubeclCpu => Ok(zensim_gpu::Backend::Cpu),
        _ => Err(Error::BackendNotEnabled { backend: b.tag() }),
    }
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
        return;
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
/// [`Metric::compute_with_cached_reference_srgb_u8`] to amortize the
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
