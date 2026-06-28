//! Optimized native-CPU metric dispatch backing [`crate::Backend::Cpu`]
//! (task #159 phase 2).
//!
//! Mirrors `zenmetrics-orchestrator`'s `cpu_adapter`: each metric routes
//! to its fast native crate (`fast-ssim2`, `dssim-core`, `butteraugli`,
//! `zensim`, and the in-tree `cvvdp` / `iwssim`) behind a `cpu-<metric>`
//! feature, instead of the cubecl `-gpu` opaque shim. This is the
//! one-shot **score** path; warm / cached-reference dispatch for
//! `Backend::Cpu` is phase 4.
//!
//! A metric whose `cpu-*` feature is not built in this configuration
//! resolves to [`CpuMetricState::FeatureDisabled`] and returns
//! [`crate::Error::BackendNotEnabled`] (`backend: "cpu"`) when scored —
//! the honest "this build can't run that metric on the optimized CPU
//! path" signal, symmetric with the per-crate GPU backend converters.

use crate::{Error, MetricKind, MetricParams, Result, Score};

/// Per-metric optimized-CPU scorer state. Feature-gated exactly like
/// `cpu_adapter::CpuAdapterState`; one wired variant per metric whose
/// `cpu-*` feature is on, plus [`CpuMetricState::FeatureDisabled`] for
/// the rest. Each variant carries the `(width, height)` it was built for
/// so [`CpuMetricState::dims`] is uniform.
pub(crate) enum CpuMetricState {
    /// `fast-ssim2` (Imazen, SIMD) — sRGB→linear→XYB is internal; the
    /// scorer is stateless so we only stash the dims. `cached_ref` holds
    /// the warm-path precompute (`Ssimulacra2Reference`) when set.
    #[cfg(feature = "cpu-ssim2")]
    Ssim2 {
        width: u32,
        height: u32,
        cached_ref: Option<fast_ssim2::Ssimulacra2Reference>,
    },
    /// in-tree `cvvdp` — stateful (internal scratch), so hold the instance.
    #[cfg(feature = "cpu-cvvdp")]
    Cvvdp {
        inner: Box<cvvdp::Cvvdp>,
        width: u32,
        height: u32,
    },
    /// in-tree `iwssim` — stateful (dims baked in at `new`, min side 176).
    #[cfg(feature = "cpu-iwssim")]
    Iwssim {
        inner: Box<iwssim::Iwssim>,
        width: u32,
        height: u32,
    },
    /// `zensim` — stateful scorer built from a profile; dims live on the
    /// per-call `RgbSlice`, so stash them here. `cached_ref` holds the
    /// warm-path `PrecomputedReference` when set.
    #[cfg(feature = "cpu-zensim")]
    Zensim {
        inner: Box<zensim::Zensim>,
        width: u32,
        height: u32,
        cached_ref: Option<zensim::PrecomputedReference>,
    },
    /// `dssim-core` — holds the `Dssim` config object; images are built
    /// per call. Lower is better (0 = identical). `cached_ref` holds the
    /// warm-path multi-scale `DssimImage` when set.
    #[cfg(feature = "cpu-dssim")]
    Dssim {
        inner: Box<dssim_core::Dssim>,
        width: u32,
        height: u32,
        cached_ref: Option<dssim_core::DssimImage<f32>>,
    },
    /// `butteraugli` — free-function scorer; hold the params. Lower is
    /// better (0 = identical). `cached_ref` holds the warm-path
    /// `ButteraugliReference` (reference XYB + masks) when set.
    #[cfg(feature = "cpu-butter")]
    Butter {
        params: butteraugli::ButteraugliParams,
        width: u32,
        height: u32,
        cached_ref: Option<butteraugli::ButteraugliReference>,
    },
    /// `kind`'s `cpu-*` feature is not built — optimized-CPU scoring for
    /// it is unavailable in this configuration.
    FeatureDisabled(MetricKind),
}

impl CpuMetricState {
    /// Build the optimized-CPU scorer for `kind` at `width × height`.
    /// Cheap for stateless metrics; for stateful ones this builds the
    /// native instance. Returns [`CpuMetricState::FeatureDisabled`] for
    /// metrics whose `cpu-*` feature is off rather than failing, so the
    /// error surfaces at score time with a clear backend message.
    pub(crate) fn new(
        kind: MetricKind,
        width: u32,
        height: u32,
        params: &MetricParams,
    ) -> Result<Self> {
        // Borrow everything so unused-arg lints stay quiet in any single
        // `cpu-*` feature configuration (different arms use different args).
        let _ = (width, height, &params);
        match kind {
            #[cfg(feature = "cpu-ssim2")]
            MetricKind::Ssim2 => Ok(CpuMetricState::Ssim2 {
                width,
                height,
                cached_ref: None,
            }),
            #[cfg(feature = "cpu-cvvdp")]
            MetricKind::Cvvdp => {
                // cvvdp re-exports cvvdp-gpu's `CvvdpParams`, the same
                // struct the umbrella wraps in `MetricParams::Cvvdp` — no
                // translation needed (see cpu_adapter::construct_cvvdp).
                let p = match params {
                    MetricParams::Cvvdp(p) => *p,
                    _ => {
                        return Err(Error::Metric {
                            kind: "cvvdp",
                            message: "expected MetricParams::Cvvdp for Backend::Cpu cvvdp".into(),
                        });
                    }
                };
                let inner = cvvdp::Cvvdp::new(width, height, p).map_err(|e| Error::Metric {
                    kind: "cvvdp",
                    message: format!("cvvdp::Cvvdp::new: {e}"),
                })?;
                Ok(CpuMetricState::Cvvdp {
                    inner: Box::new(inner),
                    width,
                    height,
                })
            }
            #[cfg(feature = "cpu-iwssim")]
            MetricKind::Iwssim => {
                // No umbrella IwssimParams variant — the CPU port uses crate
                // defaults (mirrors cpu_adapter::construct_iwssim). `new`
                // rejects sub-176 sides (allow_small = false).
                let inner = iwssim::Iwssim::new(width, height).map_err(|e| Error::Metric {
                    kind: "iwssim",
                    message: format!("iwssim::Iwssim::new: {e}"),
                })?;
                Ok(CpuMetricState::Iwssim {
                    inner: Box::new(inner),
                    width,
                    height,
                })
            }
            #[cfg(feature = "cpu-zensim")]
            MetricKind::Zensim => {
                // zensim exposes the same default profile the GPU crate
                // wraps; `latest_preview()` matches production sweep workers
                // (mirrors cpu_adapter::construct_zensim).
                let inner = zensim::Zensim::new(zensim::ZensimProfile::latest_preview());
                Ok(CpuMetricState::Zensim {
                    inner: Box::new(inner),
                    width,
                    height,
                    cached_ref: None,
                })
            }
            #[cfg(feature = "cpu-dssim")]
            MetricKind::Dssim => {
                // dssim-core uses crate defaults; the umbrella's
                // MetricParams::Dssim wraps dssim-gpu params we don't lift
                // (mirrors cpu_adapter::construct_dssim).
                Ok(CpuMetricState::Dssim {
                    inner: Box::new(dssim_core::Dssim::new()),
                    width,
                    height,
                    cached_ref: None,
                })
            }
            #[cfg(feature = "cpu-butter")]
            MetricKind::Butter => {
                // butteraugli CPU defaults, but lift `intensity_target` from the
                // umbrella's `MetricParams::Butter` (butteraugli-gpu params) so the
                // native linear HDR path scores at the same display peak as the GPU
                // path (HdrScorer sets it via `with_intensity_target(peak_nits)`).
                // The lift is `butter`-gated: `MetricParams::Butter` only exists when
                // the GPU `butter` feature is on, and an HDR butter scorer can only be
                // constructed in that case anyway (build_hdr_metric needs it). In a
                // pure `cpu-butter` build the native `new()` default (80 cd/m²) stands.
                let mut native = butteraugli::ButteraugliParams::new();
                #[cfg(feature = "butter")]
                if let MetricParams::Butter(gpu) = params {
                    native = native.with_intensity_target(gpu.intensity_target);
                }
                Ok(CpuMetricState::Butter {
                    params: native,
                    width,
                    height,
                    cached_ref: None,
                })
            }
            // Load-bearing for partial-feature builds (a metric whose
            // `cpu-*` feature is off lands here); unreachable only when all
            // six arms compile in, hence the localized allow.
            #[allow(unreachable_patterns)]
            other => Ok(CpuMetricState::FeatureDisabled(other)),
        }
    }

    /// Build the optimized-CPU scorer for **HDR** scoring at `width × height`,
    /// baking the HDR display `peak_nits` into the display-aware metrics
    /// (butteraugli `intensity_target`, cvvdp display `y_peak`). Unlike
    /// [`Self::new`] this takes **no** [`MetricParams`] — so an HDR CPU scorer
    /// builds from the native `cpu-*` crates alone, with **no GPU metric
    /// feature** (and therefore no cubecl) compiled. The SSIM-family
    /// (ssim2/zensim/iwssim/dssim) ignore the peak here: their HDR feeding
    /// (integrated PU21 / float-PU luma) is applied at score time by the
    /// `Metric::compute_*` HDR entries.
    #[cfg(any(
        feature = "cpu-ssim2",
        feature = "cpu-cvvdp",
        feature = "cpu-dssim",
        feature = "cpu-butter",
        feature = "cpu-zensim",
        feature = "cpu-iwssim"
    ))]
    pub(crate) fn new_hdr(
        kind: MetricKind,
        width: u32,
        height: u32,
        peak_nits: f32,
    ) -> Result<Self> {
        // Keep unused-arg lints quiet in any single-`cpu-*` configuration.
        let _ = (width, height, peak_nits);
        match kind {
            #[cfg(feature = "cpu-ssim2")]
            MetricKind::Ssim2 => Ok(CpuMetricState::Ssim2 {
                width,
                height,
                cached_ref: None,
            }),
            #[cfg(feature = "cpu-cvvdp")]
            MetricKind::Cvvdp => {
                // Display-aware: bake the HDR peak into the cvvdp display model
                // (mirrors the GPU path's `MetricParams::cvvdp_with_display`).
                // No `MetricParams`, so no `cvvdp` GPU feature needed — the
                // native `cvvdp` crate owns the same `CvvdpParams`/`DisplayModel`.
                let params = cvvdp::CvvdpParams {
                    display: cvvdp::DisplayModel {
                        y_peak: peak_nits,
                        ..cvvdp::DisplayModel::STANDARD_HDR_LINEAR
                    },
                    ..cvvdp::CvvdpParams::default()
                };
                let inner =
                    cvvdp::Cvvdp::new(width, height, params).map_err(|e| Error::Metric {
                        kind: "cvvdp",
                        message: format!("cvvdp::Cvvdp::new: {e}"),
                    })?;
                Ok(CpuMetricState::Cvvdp {
                    inner: Box::new(inner),
                    width,
                    height,
                })
            }
            #[cfg(feature = "cpu-iwssim")]
            MetricKind::Iwssim => {
                let inner = iwssim::Iwssim::new(width, height).map_err(|e| Error::Metric {
                    kind: "iwssim",
                    message: format!("iwssim::Iwssim::new: {e}"),
                })?;
                Ok(CpuMetricState::Iwssim {
                    inner: Box::new(inner),
                    width,
                    height,
                })
            }
            #[cfg(feature = "cpu-zensim")]
            MetricKind::Zensim => {
                let inner = zensim::Zensim::new(zensim::ZensimProfile::latest_preview());
                Ok(CpuMetricState::Zensim {
                    inner: Box::new(inner),
                    width,
                    height,
                    cached_ref: None,
                })
            }
            #[cfg(feature = "cpu-dssim")]
            MetricKind::Dssim => Ok(CpuMetricState::Dssim {
                inner: Box::new(dssim_core::Dssim::new()),
                width,
                height,
                cached_ref: None,
            }),
            #[cfg(feature = "cpu-butter")]
            MetricKind::Butter => {
                // Display-aware: native butteraugli `intensity_target` = HDR peak
                // (the GPU path sets the same via `with_intensity_target`).
                let params = butteraugli::ButteraugliParams::new().with_intensity_target(peak_nits);
                Ok(CpuMetricState::Butter {
                    params,
                    width,
                    height,
                    cached_ref: None,
                })
            }
            #[allow(unreachable_patterns)]
            other => Ok(CpuMetricState::FeatureDisabled(other)),
        }
    }

    /// The metric this state scores.
    pub(crate) fn kind(&self) -> MetricKind {
        match self {
            #[cfg(feature = "cpu-ssim2")]
            CpuMetricState::Ssim2 { .. } => MetricKind::Ssim2,
            #[cfg(feature = "cpu-cvvdp")]
            CpuMetricState::Cvvdp { .. } => MetricKind::Cvvdp,
            #[cfg(feature = "cpu-iwssim")]
            CpuMetricState::Iwssim { .. } => MetricKind::Iwssim,
            #[cfg(feature = "cpu-zensim")]
            CpuMetricState::Zensim { .. } => MetricKind::Zensim,
            #[cfg(feature = "cpu-dssim")]
            CpuMetricState::Dssim { .. } => MetricKind::Dssim,
            #[cfg(feature = "cpu-butter")]
            CpuMetricState::Butter { .. } => MetricKind::Butter,
            CpuMetricState::FeatureDisabled(k) => *k,
        }
    }

    /// Image dimensions this scorer was constructed for. `(0, 0)` for a
    /// feature-disabled state (it was never given real dims).
    pub(crate) fn dims(&self) -> (u32, u32) {
        match self {
            #[cfg(feature = "cpu-ssim2")]
            CpuMetricState::Ssim2 { width, height, .. } => (*width, *height),
            #[cfg(feature = "cpu-cvvdp")]
            CpuMetricState::Cvvdp { width, height, .. } => (*width, *height),
            #[cfg(feature = "cpu-iwssim")]
            CpuMetricState::Iwssim { width, height, .. } => (*width, *height),
            #[cfg(feature = "cpu-zensim")]
            CpuMetricState::Zensim { width, height, .. } => (*width, *height),
            #[cfg(feature = "cpu-dssim")]
            CpuMetricState::Dssim { width, height, .. } => (*width, *height),
            #[cfg(feature = "cpu-butter")]
            CpuMetricState::Butter { width, height, .. } => (*width, *height),
            CpuMetricState::FeatureDisabled(_) => (0, 0),
        }
    }

    /// One-shot score of a packed sRGB `R, G, B, R, G, B, …` pair
    /// (`width × height × 3` bytes per side).
    /// Float-PU(luma) gray feeding for iwssim: PU21-encoded BT.709-luma gray
    /// planes (f32, 0..255 scale) straight into `Iwssim::score_gray` — the u8
    /// quantization this skips costs ~0.18 SROCC on UPIQ HDR (#25). Every
    /// other metric returns a loud error; feed it per `hdr::hdr_feeding()`.
    pub(crate) fn compute_pu_luma_gray(
        &mut self,
        ref_gray: &[f32],
        dis_gray: &[f32],
    ) -> Result<Score> {
        // Quiet unused-arg lints in single-`cpu-*` feature configurations
        // (same convention as `CpuMetricState::new`).
        let _ = (ref_gray, dis_gray);
        match self {
            #[cfg(feature = "cpu-iwssim")]
            CpuMetricState::Iwssim {
                inner,
                width,
                height,
            } => {
                let expected = (*width as usize) * (*height as usize);
                if ref_gray.len() != expected || dis_gray.len() != expected {
                    return Err(Error::Metric {
                        kind: "iwssim",
                        message: format!(
                            "gray plane length mismatch: expected {expected}, got {} / {}",
                            ref_gray.len(),
                            dis_gray.len()
                        ),
                    });
                }
                let result = inner.score_gray(ref_gray, dis_gray).map_err(|e| Error::Metric {
                    kind: "iwssim",
                    message: e.to_string(),
                })?;
                Ok(Score {
                    value: result.score,
                    metric_name: "iwssim",
                    metric_version: env!("CARGO_PKG_VERSION"),
                })
            }
            _ => Err(Error::Metric {
                kind: "cpu",
                message: "no float-PU(luma) gray path for this metric on the CPU dispatch;                           feed it per hdr::hdr_feeding()"
                    .into(),
            }),
        }
    }

    /// Integrated-PU HDR feeding for ssim2 + zensim: **absolute-luminance
    /// interleaved linear-RGB f32** (cd/m², `[R,G,B, …]`) straight into the
    /// metric's PU21 entry — `fast_ssim2::compute_ssimulacra2_pu_nits`
    /// (`hdr-pu` feature, git-pinned via the workspace `[patch.crates-io]`
    /// until a fast-ssim2 release ships it) / `Zensim::compute_pu_linear`
    /// (zensim PR #44, squashed as 3f0334de). The PU21 (banding_glare)
    /// front-end replaces the SDR cube-root *inside* each metric, so there
    /// is no u8 round-trip and no input-side PU shell (the shell capped at
    /// ~0.61 UPIQ, #25). Every other metric returns a loud error; feed it
    /// per `hdr::hdr_feeding()`.
    pub(crate) fn compute_pu_nits_interleaved(
        &mut self,
        ref_nits: &[f32],
        dis_nits: &[f32],
    ) -> Result<Score> {
        // Quiet unused-arg lints in single-`cpu-*` feature configurations
        // (same convention as `CpuMetricState::new`).
        let _ = (ref_nits, dis_nits);
        match self {
            #[cfg(feature = "cpu-ssim2")]
            CpuMetricState::Ssim2 { width, height, .. } => {
                let (w, h) = (*width as usize, *height as usize);
                let expected = w * h * 3;
                if ref_nits.len() != expected || dis_nits.len() != expected {
                    return Err(Error::Metric {
                        kind: "ssim2",
                        message: format!(
                            "nits buffer length mismatch: expected {expected}, got {} / {}",
                            ref_nits.len(),
                            dis_nits.len()
                        ),
                    });
                }
                // `[f32; 3]` is `bytemuck::Pod`, so the interleaved buffer
                // reinterprets per-pixel without a copy; `LinearRgbImage`
                // takes owned data, so one `to_vec` per side is the cost.
                let to_img = |nits: &[f32]| {
                    fast_ssim2::LinearRgbImage::try_new(
                        bytemuck::cast_slice(nits).to_vec(),
                        w,
                        h,
                    )
                    .map_err(|e| Error::Metric {
                        kind: "ssim2",
                        message: format!("fast-ssim2 LinearRgbImage: {e}"),
                    })
                };
                let v =
                    fast_ssim2::compute_ssimulacra2_pu_nits(to_img(ref_nits)?, to_img(dis_nits)?)
                        .map_err(|e| Error::Metric {
                            kind: "ssim2",
                            message: format!("fast-ssim2 compute_ssimulacra2_pu_nits: {e}"),
                        })?;
                Ok(Score {
                    value: v,
                    metric_name: "ssim2",
                    metric_version: env!("CARGO_PKG_VERSION"),
                })
            }
            #[cfg(feature = "cpu-zensim")]
            CpuMetricState::Zensim {
                inner,
                width,
                height,
                ..
            } => {
                let (w, h) = (*width as usize, *height as usize);
                let expected = w * h * 3;
                if ref_nits.len() != expected || dis_nits.len() != expected {
                    return Err(Error::Metric {
                        kind: "zensim",
                        message: format!(
                            "nits buffer length mismatch: expected {expected}, got {} / {}",
                            ref_nits.len(),
                            dis_nits.len()
                        ),
                    });
                }
                let result = inner
                    .compute_pu_linear(ref_nits, dis_nits, w, h, 3 * w, 3 * w)
                    .map_err(|e| Error::Metric {
                        kind: "zensim",
                        message: format!("zensim compute_pu_linear: {e:?}"),
                    })?;
                Ok(Score {
                    value: result.score(),
                    metric_name: "zensim",
                    metric_version: env!("CARGO_PKG_VERSION"),
                })
            }
            _ => Err(Error::Metric {
                kind: "cpu",
                message: "no integrated-PU nits path for this metric on the CPU dispatch;                           feed it per hdr::hdr_feeding()"
                    .into(),
            }),
        }
    }

    pub(crate) fn compute_srgb_u8(&mut self, r: &[u8], d: &[u8]) -> Result<Score> {
        match self {
            #[cfg(feature = "cpu-ssim2")]
            CpuMetricState::Ssim2 { width, height, .. } => compute_ssim2(*width, *height, r, d),
            #[cfg(feature = "cpu-cvvdp")]
            CpuMetricState::Cvvdp {
                inner,
                width,
                height,
            } => compute_cvvdp(inner, *width, *height, r, d),
            #[cfg(feature = "cpu-iwssim")]
            CpuMetricState::Iwssim {
                inner,
                width,
                height,
            } => compute_iwssim(inner, *width, *height, r, d),
            #[cfg(feature = "cpu-zensim")]
            CpuMetricState::Zensim {
                inner,
                width,
                height,
                ..
            } => compute_zensim(inner, *width, *height, r, d),
            #[cfg(feature = "cpu-dssim")]
            CpuMetricState::Dssim {
                inner,
                width,
                height,
                ..
            } => compute_dssim(inner, *width, *height, r, d),
            #[cfg(feature = "cpu-butter")]
            CpuMetricState::Butter {
                params,
                width,
                height,
                ..
            } => compute_butter(params, *width, *height, r, d),
            CpuMetricState::FeatureDisabled(_) => Err(Error::BackendNotEnabled { backend: "cpu" }),
        }
    }

    // ---- warm / cached-reference path (folded in from the orchestrator's
    // `cpu_adapter`, 2026-06-27) — true precompute reused across distorted
    // candidates, replacing the umbrella's prior buffer-replay warm-ref. ----

    /// Whether this metric has a true cached-reference fast path.
    // Consumed by the orchestrator executor's umbrella CPU leaf in the
    // cpu_adapter-merge step 2 (it replaces `CpuAdapter::supports_cached_ref`);
    // the other warm methods are already wired through `Metric`'s warm arms.
    #[allow(dead_code)]
    pub(crate) fn supports_cached_ref(&self) -> bool {
        match self {
            #[cfg(feature = "cpu-cvvdp")]
            CpuMetricState::Cvvdp { .. } => true,
            #[cfg(feature = "cpu-ssim2")]
            CpuMetricState::Ssim2 { .. } => true,
            #[cfg(feature = "cpu-dssim")]
            CpuMetricState::Dssim { .. } => true,
            #[cfg(feature = "cpu-butter")]
            CpuMetricState::Butter { .. } => true,
            #[cfg(feature = "cpu-zensim")]
            CpuMetricState::Zensim { .. } => true,
            #[cfg(feature = "cpu-iwssim")]
            CpuMetricState::Iwssim { .. } => true,
            CpuMetricState::FeatureDisabled(_) => false,
        }
    }

    /// Install the reference for subsequent [`Self::compute_with_cached_reference`]
    /// calls (true precompute: builds the reference XYB / pyramid / masks once).
    pub(crate) fn set_reference(&mut self, r: &[u8]) -> Result<()> {
        let (w, h) = self.dims();
        let expected = (w as usize) * (h as usize) * 3;
        if r.len() != expected {
            return Err(Error::Metric {
                kind: "cpu",
                message: format!(
                    "cpu set_reference: expected {expected} sRGB bytes, got {}",
                    r.len()
                ),
            });
        }
        match self {
            #[cfg(feature = "cpu-cvvdp")]
            CpuMetricState::Cvvdp { inner, .. } => {
                inner.warm_reference(r).map_err(|e| Error::Metric {
                    kind: "cvvdp",
                    message: format!("cvvdp warm_reference: {e}"),
                })
            }
            #[cfg(feature = "cpu-ssim2")]
            CpuMetricState::Ssim2 {
                width,
                height,
                cached_ref,
            } => {
                let img = ssim2_image_ref(r, *width as usize, *height as usize);
                let pre =
                    fast_ssim2::Ssimulacra2Reference::new(img).map_err(|e| Error::Metric {
                        kind: "ssim2",
                        message: format!("fast-ssim2 Ssimulacra2Reference::new: {e}"),
                    })?;
                *cached_ref = Some(pre);
                Ok(())
            }
            #[cfg(feature = "cpu-dssim")]
            CpuMetricState::Dssim {
                inner,
                width,
                height,
                cached_ref,
            } => {
                let img = make_dssim_image(inner, r, *width as usize, *height as usize)?;
                *cached_ref = Some(img);
                Ok(())
            }
            #[cfg(feature = "cpu-butter")]
            CpuMetricState::Butter {
                params,
                width,
                height,
                cached_ref,
            } => {
                let pre = butteraugli::ButteraugliReference::new(
                    r,
                    *width as usize,
                    *height as usize,
                    params.clone(),
                )
                .map_err(|e| Error::Metric {
                    kind: "butter",
                    message: format!("butteraugli ButteraugliReference::new: {e:?}"),
                })?;
                *cached_ref = Some(pre);
                Ok(())
            }
            #[cfg(feature = "cpu-zensim")]
            CpuMetricState::Zensim {
                inner,
                width,
                height,
                cached_ref,
            } => {
                let src: &[[u8; 3]] = bytemuck::cast_slice(r);
                let ref_slice = zensim::RgbSlice::new(src, *width as usize, *height as usize);
                let pre = inner
                    .precompute_reference(&ref_slice)
                    .map_err(|e| Error::Metric {
                        kind: "zensim",
                        message: format!("zensim precompute_reference: {e:?}"),
                    })?;
                *cached_ref = Some(pre);
                Ok(())
            }
            #[cfg(feature = "cpu-iwssim")]
            CpuMetricState::Iwssim { inner, .. } => {
                inner.warm_reference(r).map_err(|e| Error::Metric {
                    kind: "iwssim",
                    message: format!("iwssim warm_reference: {e}"),
                })
            }
            CpuMetricState::FeatureDisabled(_) => Err(Error::BackendNotEnabled { backend: "cpu" }),
        }
    }

    /// Score `dist` against the reference installed by [`Self::set_reference`].
    pub(crate) fn compute_with_cached_reference(&mut self, d: &[u8]) -> Result<Score> {
        let (w, h) = self.dims();
        let expected = (w as usize) * (h as usize) * 3;
        if d.len() != expected {
            return Err(Error::Metric {
                kind: "cpu",
                message: format!(
                    "cpu compute_with_cached_reference: expected {expected} sRGB bytes, got {}",
                    d.len()
                ),
            });
        }
        match self {
            #[cfg(feature = "cpu-cvvdp")]
            CpuMetricState::Cvvdp { inner, .. } => {
                let v = inner.score_with_warm_ref(d).map_err(|e| Error::Metric {
                    kind: "cvvdp",
                    message: format!("cvvdp score_with_warm_ref: {e}"),
                })?;
                Ok(Score {
                    value: v as f64,
                    metric_name: "cvvdp",
                    metric_version: env!("CARGO_PKG_VERSION"),
                })
            }
            #[cfg(feature = "cpu-ssim2")]
            CpuMetricState::Ssim2 {
                width,
                height,
                cached_ref,
            } => {
                let pre = cached_ref.as_ref().ok_or_else(|| Error::Metric {
                    kind: "ssim2",
                    message: "ssim2: no cached reference; call set_reference first".into(),
                })?;
                let dist_img = ssim2_image_ref(d, *width as usize, *height as usize);
                let v = pre.compare(dist_img).map_err(|e| Error::Metric {
                    kind: "ssim2",
                    message: format!("fast-ssim2 compare: {e}"),
                })?;
                Ok(Score {
                    value: v,
                    metric_name: "ssim2",
                    metric_version: env!("CARGO_PKG_VERSION"),
                })
            }
            #[cfg(feature = "cpu-dssim")]
            CpuMetricState::Dssim {
                inner,
                width,
                height,
                cached_ref,
            } => {
                let cached = cached_ref.as_ref().ok_or_else(|| Error::Metric {
                    kind: "dssim",
                    message: "dssim: no cached reference; call set_reference first".into(),
                })?;
                let dist_img = make_dssim_image(inner, d, *width as usize, *height as usize)?;
                let (score, _maps) = inner.compare(cached, dist_img);
                Ok(Score {
                    value: f64::from(score),
                    metric_name: "dssim",
                    metric_version: env!("CARGO_PKG_VERSION"),
                })
            }
            #[cfg(feature = "cpu-butter")]
            CpuMetricState::Butter { cached_ref, .. } => {
                let pre = cached_ref.as_ref().ok_or_else(|| Error::Metric {
                    kind: "butter",
                    message: "butter: no cached reference; call set_reference first".into(),
                })?;
                let result = pre.compare(d).map_err(|e| Error::Metric {
                    kind: "butter",
                    message: format!("butteraugli compare: {e:?}"),
                })?;
                Ok(Score {
                    value: result.score,
                    metric_name: "butter",
                    metric_version: env!("CARGO_PKG_VERSION"),
                })
            }
            #[cfg(feature = "cpu-zensim")]
            CpuMetricState::Zensim {
                inner,
                width,
                height,
                cached_ref,
            } => {
                let pre = cached_ref.as_ref().ok_or_else(|| Error::Metric {
                    kind: "zensim",
                    message: "zensim: no cached reference; call set_reference first".into(),
                })?;
                let dst: &[[u8; 3]] = bytemuck::cast_slice(d);
                let dist_slice = zensim::RgbSlice::new(dst, *width as usize, *height as usize);
                let result =
                    inner
                        .compute_with_ref(pre, &dist_slice)
                        .map_err(|e| Error::Metric {
                            kind: "zensim",
                            message: format!("zensim compute_with_ref: {e:?}"),
                        })?;
                Ok(Score {
                    value: result.score(),
                    metric_name: "zensim",
                    metric_version: env!("CARGO_PKG_VERSION"),
                })
            }
            #[cfg(feature = "cpu-iwssim")]
            CpuMetricState::Iwssim { inner, .. } => {
                if !inner.has_reference() {
                    return Err(Error::Metric {
                        kind: "iwssim",
                        message: "iwssim: no cached reference; call set_reference first".into(),
                    });
                }
                // Matches cpu_adapter: the warm cached-ref entry routes through
                // the strip walker (ref-side pyramid amortized; ≤1e-4 vs full).
                let result = inner
                    .score_with_warm_ref_strip(d, iwssim::STRIP_BODY_DEFAULT)
                    .map_err(|e| Error::Metric {
                        kind: "iwssim",
                        message: format!("iwssim score_with_warm_ref_strip: {e}"),
                    })?;
                Ok(Score {
                    value: result.score,
                    metric_name: "iwssim",
                    metric_version: env!("CARGO_PKG_VERSION"),
                })
            }
            CpuMetricState::FeatureDisabled(_) => Err(Error::BackendNotEnabled { backend: "cpu" }),
        }
    }

    /// Whether a reference is currently installed.
    pub(crate) fn has_reference(&self) -> bool {
        match self {
            #[cfg(feature = "cpu-cvvdp")]
            CpuMetricState::Cvvdp { inner, .. } => inner.has_reference(),
            #[cfg(feature = "cpu-iwssim")]
            CpuMetricState::Iwssim { inner, .. } => inner.has_reference(),
            #[cfg(feature = "cpu-ssim2")]
            CpuMetricState::Ssim2 { cached_ref, .. } => cached_ref.is_some(),
            #[cfg(feature = "cpu-dssim")]
            CpuMetricState::Dssim { cached_ref, .. } => cached_ref.is_some(),
            #[cfg(feature = "cpu-butter")]
            CpuMetricState::Butter { cached_ref, .. } => cached_ref.is_some(),
            #[cfg(feature = "cpu-zensim")]
            CpuMetricState::Zensim { cached_ref, .. } => cached_ref.is_some(),
            CpuMetricState::FeatureDisabled(_) => false,
        }
    }

    /// Drop any installed reference. cvvdp/iwssim hold warm state internally
    /// and expose no clear API; the next `set_reference` overwrites it, so
    /// clear is a documented no-op for them (the four cache-field metrics
    /// drop their `cached_ref`).
    pub(crate) fn clear_reference(&mut self) {
        match self {
            #[cfg(feature = "cpu-ssim2")]
            CpuMetricState::Ssim2 { cached_ref, .. } => *cached_ref = None,
            #[cfg(feature = "cpu-dssim")]
            CpuMetricState::Dssim { cached_ref, .. } => *cached_ref = None,
            #[cfg(feature = "cpu-butter")]
            CpuMetricState::Butter { cached_ref, .. } => *cached_ref = None,
            #[cfg(feature = "cpu-zensim")]
            CpuMetricState::Zensim { cached_ref, .. } => *cached_ref = None,
            // cvvdp / iwssim: no crate clear API; next set_reference overwrites.
            #[allow(unreachable_patterns)]
            _ => {}
        }
    }

    /// One-shot score of an **interleaved linear-light** `R, G, B, …` f32 pair
    /// (`width × height × 3` f32 per side) — the native HDR feeding for the
    /// luminance-aware metrics (butter/cvvdp), no u8 round-trip and no
    /// `Backend::CubeclCpu`. Values are display-relative `[0,1]` where `1.0`
    /// is the metric's `intensity_target` (the display peak baked in at
    /// construction). Metrics with no native linear model (the SSIM-family,
    /// fed via `compute_srgb_u8` after pu-rescale) return a clear error.
    pub(crate) fn compute_from_linear_interleaved(
        &mut self,
        r: &[f32],
        d: &[f32],
    ) -> Result<(Score, Option<f64>)> {
        // Quiet unused-arg lints in single-`cpu-*` feature configurations
        // (same convention as `CpuMetricState::new`).
        let _ = (r, d);
        match self {
            #[cfg(feature = "cpu-butter")]
            CpuMetricState::Butter {
                params,
                width,
                height,
                ..
            } => compute_butter_linear(params, *width, *height, r, d).map(|(s, p)| (s, Some(p))),
            #[cfg(feature = "cpu-cvvdp")]
            CpuMetricState::Cvvdp {
                inner,
                width,
                height,
            } => compute_cvvdp_linear(inner, *width, *height, r, d).map(|s| (s, None)),
            CpuMetricState::FeatureDisabled(_) => Err(Error::BackendNotEnabled { backend: "cpu" }),
            // SSIM-family: no native absolute-luminance model — fed via
            // compute_srgb_u8 after pu-rescale, never the linear path.
            #[allow(unreachable_patterns)]
            other => Err(Error::Metric {
                kind: "cpu",
                message: format!(
                    "CPU {:?} has no linear-light feeding; feed via \
                     compute_srgb_u8(to_sdr_u8(..)) per hdr_feeding()",
                    other.kind()
                ),
            }),
        }
    }
}

/// Validate that both sides are exactly `width × height × 3` packed bytes.
#[cfg(any(
    feature = "cpu-ssim2",
    feature = "cpu-cvvdp",
    feature = "cpu-iwssim",
    feature = "cpu-zensim",
    feature = "cpu-dssim",
    feature = "cpu-butter"
))]
fn check_srgb_len(kind: &'static str, width: u32, height: u32, r: &[u8], d: &[u8]) -> Result<()> {
    let expected = (width as usize) * (height as usize) * 3;
    if r.len() != expected || d.len() != expected {
        return Err(Error::Metric {
            kind,
            message: format!(
                "cpu {kind}: expected {expected} packed sRGB bytes per side ({width}×{height}×3), \
                 got ref={} dist={}",
                r.len(),
                d.len()
            ),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// fast-ssim2 wiring — mirrors cpu_adapter::{ssim2_image_ref, compute_ssim2}.
// ---------------------------------------------------------------------------

/// Borrow an interleaved sRGB-u8 buffer as `ImgRef<'_, [u8; 3]>`.
/// `[u8; 3]` is `bytemuck::Pod`, so this reinterprets in place — no copy.
#[cfg(feature = "cpu-ssim2")]
fn ssim2_image_ref<'a>(bytes: &'a [u8], w: usize, h: usize) -> imgref::ImgRef<'a, [u8; 3]> {
    let pixels: &[[u8; 3]] = bytemuck::cast_slice(bytes);
    imgref::ImgRef::new(pixels, w, h)
}

#[cfg(feature = "cpu-ssim2")]
fn compute_ssim2(width: u32, height: u32, r: &[u8], d: &[u8]) -> Result<Score> {
    check_srgb_len("ssim2", width, height, r, d)?;
    let ref_img = ssim2_image_ref(r, width as usize, height as usize);
    let dist_img = ssim2_image_ref(d, width as usize, height as usize);
    let v = fast_ssim2::compute_ssimulacra2(ref_img, dist_img).map_err(|e| Error::Metric {
        kind: "ssim2",
        message: format!("fast-ssim2 compute_ssimulacra2: {e}"),
    })?;
    Ok(Score {
        value: v,
        metric_name: "ssim2",
        metric_version: env!("CARGO_PKG_VERSION"),
    })
}

// ---------------------------------------------------------------------------
// cvvdp wiring — mirrors cpu_adapter::compute_cvvdp.
// ---------------------------------------------------------------------------

#[cfg(feature = "cpu-cvvdp")]
fn compute_cvvdp(
    c: &mut cvvdp::Cvvdp,
    width: u32,
    height: u32,
    r: &[u8],
    d: &[u8],
) -> Result<Score> {
    check_srgb_len("cvvdp", width, height, r, d)?;
    let v = c.score(r, d).map_err(|e| Error::Metric {
        kind: "cvvdp",
        message: format!("cvvdp score: {e}"),
    })?;
    Ok(Score {
        value: v as f64,
        metric_name: "cvvdp",
        metric_version: env!("CARGO_PKG_VERSION"),
    })
}

/// Split interleaved `[R,G,B, …]` f32 into tight planar `(R…, G…, B…)`.
/// cvvdp's native scorer is planar; butter's is interleaved, so this is
/// cvvdp-only (butter takes the interleaved buffer zero-copy).
#[cfg(feature = "cpu-cvvdp")]
fn deinterleave_f32(rgb: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = rgb.len() / 3;
    let mut r = Vec::with_capacity(n);
    let mut g = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    for px in rgb.chunks_exact(3) {
        r.push(px[0]);
        g.push(px[1]);
        b.push(px[2]);
    }
    (r, g, b)
}

/// Faithful native cvvdp on interleaved display-relative `[0,1]` linear via
/// [`cvvdp::Cvvdp::score_from_linear_planes`] — pure-Rust SIMD (archmage), no
/// cubecl. The DisplayModel (peak/black/refl) comes from the params the scorer
/// was built with (`MetricParams::Cvvdp`, threaded by `cpu_dispatch::new`), so
/// `1.0` maps to the same display peak as the GPU cvvdp linear path.
#[cfg(feature = "cpu-cvvdp")]
fn compute_cvvdp_linear(
    c: &mut cvvdp::Cvvdp,
    width: u32,
    height: u32,
    r: &[f32],
    d: &[f32],
) -> Result<Score> {
    check_linear_len("cvvdp", width, height, r, d)?;
    let (rr, rg, rb) = deinterleave_f32(r);
    let (dr, dg, db) = deinterleave_f32(d);
    let v = c
        .score_from_linear_planes(&rr, &rg, &rb, &dr, &dg, &db, width as usize)
        .map_err(|e| Error::Metric {
            kind: "cvvdp",
            message: format!("cvvdp score_from_linear_planes: {e}"),
        })?;
    Ok(Score {
        value: v as f64,
        metric_name: "cvvdp",
        metric_version: env!("CARGO_PKG_VERSION"),
    })
}

// ---------------------------------------------------------------------------
// iwssim wiring — mirrors cpu_adapter::compute_iwssim.
// ---------------------------------------------------------------------------

#[cfg(feature = "cpu-iwssim")]
fn compute_iwssim(
    c: &mut iwssim::Iwssim,
    width: u32,
    height: u32,
    r: &[u8],
    d: &[u8],
) -> Result<Score> {
    check_srgb_len("iwssim", width, height, r, d)?;
    let result = c.score(r, d).map_err(|e| Error::Metric {
        kind: "iwssim",
        message: format!("iwssim score: {e}"),
    })?;
    Ok(Score {
        value: result.score,
        metric_name: "iwssim",
        metric_version: env!("CARGO_PKG_VERSION"),
    })
}

// ---------------------------------------------------------------------------
// zensim wiring — mirrors cpu_adapter::compute_zensim.
// ---------------------------------------------------------------------------

#[cfg(feature = "cpu-zensim")]
fn compute_zensim(
    z: &mut zensim::Zensim,
    width: u32,
    height: u32,
    r: &[u8],
    d: &[u8],
) -> Result<Score> {
    check_srgb_len("zensim", width, height, r, d)?;
    // RgbSlice expects `&[[u8; 3]]`; `[u8; 3]` is `bytemuck::Pod`, so we
    // reinterpret the interleaved bytes in place (no copy, no `unsafe`).
    let src: &[[u8; 3]] = bytemuck::cast_slice(r);
    let dst: &[[u8; 3]] = bytemuck::cast_slice(d);
    let ref_slice = zensim::RgbSlice::new(src, width as usize, height as usize);
    let dist_slice = zensim::RgbSlice::new(dst, width as usize, height as usize);
    let result = z
        .compute(&ref_slice, &dist_slice)
        .map_err(|e| Error::Metric {
            kind: "zensim",
            message: format!("zensim compute: {e:?}"),
        })?;
    Ok(Score {
        value: result.score(),
        metric_name: "zensim",
        metric_version: env!("CARGO_PKG_VERSION"),
    })
}

// ---------------------------------------------------------------------------
// dssim wiring — mirrors cpu_adapter::{make_dssim_image, compute_dssim}.
// ---------------------------------------------------------------------------

/// Reinterpret interleaved sRGB-u8 bytes as `&[rgb::RGB<u8>]` (Pod, no copy)
/// and build the dssim-core multi-scale image.
#[cfg(feature = "cpu-dssim")]
fn make_dssim_image(
    dssim: &dssim_core::Dssim,
    bytes: &[u8],
    w: usize,
    h: usize,
) -> Result<dssim_core::DssimImage<f32>> {
    let rgb: &[rgb::RGB<u8>] = bytemuck::cast_slice(bytes);
    dssim
        .create_image_rgb(rgb, w, h)
        .ok_or_else(|| Error::Metric {
            kind: "dssim",
            message: "dssim_core create_image_rgb returned None".into(),
        })
}

#[cfg(feature = "cpu-dssim")]
fn compute_dssim(
    dssim: &dssim_core::Dssim,
    width: u32,
    height: u32,
    r: &[u8],
    d: &[u8],
) -> Result<Score> {
    check_srgb_len("dssim", width, height, r, d)?;
    let ref_img = make_dssim_image(dssim, r, width as usize, height as usize)?;
    let dist_img = make_dssim_image(dssim, d, width as usize, height as usize)?;
    let (score, _maps) = dssim.compare(&ref_img, dist_img);
    Ok(Score {
        value: f64::from(score),
        metric_name: "dssim",
        metric_version: env!("CARGO_PKG_VERSION"),
    })
}

// ---------------------------------------------------------------------------
// butteraugli wiring — mirrors cpu_adapter::compute_butter.
// ---------------------------------------------------------------------------

#[cfg(feature = "cpu-butter")]
fn compute_butter(
    params: &butteraugli::ButteraugliParams,
    width: u32,
    height: u32,
    r: &[u8],
    d: &[u8],
) -> Result<Score> {
    check_srgb_len("butter", width, height, r, d)?;
    // `rgb::RGB<u8>` is `bytemuck::Pod` (rgb's default `as-bytes` feature),
    // so reinterpret the interleaved bytes in place — no copy.
    let ref_rgb: &[rgb::RGB<u8>] = bytemuck::cast_slice(r);
    let dist_rgb: &[rgb::RGB<u8>] = bytemuck::cast_slice(d);
    let ref_img = imgref::ImgRef::new(ref_rgb, width as usize, height as usize);
    let dist_img = imgref::ImgRef::new(dist_rgb, width as usize, height as usize);
    let result =
        butteraugli::butteraugli(ref_img, dist_img, params).map_err(|e| Error::Metric {
            kind: "butter",
            message: format!("butteraugli: {e:?}"),
        })?;
    Ok(Score {
        value: result.score,
        metric_name: "butter",
        metric_version: env!("CARGO_PKG_VERSION"),
    })
}

/// Validate that both sides are exactly `width × height × 3` interleaved f32.
#[cfg(any(feature = "cpu-butter", feature = "cpu-cvvdp"))]
fn check_linear_len(
    kind: &'static str,
    width: u32,
    height: u32,
    r: &[f32],
    d: &[f32],
) -> Result<()> {
    let expected = (width as usize) * (height as usize) * 3;
    if r.len() != expected || d.len() != expected {
        return Err(Error::Metric {
            kind,
            message: format!(
                "cpu {kind}: expected {expected} interleaved linear f32 per side \
                 ({width}×{height}×3), got ref={} dist={}",
                r.len(),
                d.len()
            ),
        });
    }
    Ok(())
}

/// Faithful native butteraugli on interleaved linear-light f32 via
/// [`butteraugli::butteraugli_linear`]. `params.intensity_target` (set at
/// construction from the display peak) defines the absolute scale that
/// linear `1.0` maps to — matching the GPU `compute_from_linear_planes`
/// convention (plane-value `1.0` == `intensity_target`).
#[cfg(feature = "cpu-butter")]
fn compute_butter_linear(
    params: &butteraugli::ButteraugliParams,
    width: u32,
    height: u32,
    r: &[f32],
    d: &[f32],
) -> Result<(Score, f64)> {
    check_linear_len("butter", width, height, r, d)?;
    // `rgb::RGB<f32>` is `bytemuck::Pod`, so reinterpret interleaved f32 in
    // place — no copy (same pattern as the sRGB-u8 path's `RGB<u8>`).
    let ref_rgb: &[rgb::RGB<f32>] = bytemuck::cast_slice(r);
    let dist_rgb: &[rgb::RGB<f32>] = bytemuck::cast_slice(d);
    let ref_img = imgref::ImgRef::new(ref_rgb, width as usize, height as usize);
    let dist_img = imgref::ImgRef::new(dist_rgb, width as usize, height as usize);
    let result =
        butteraugli::butteraugli_linear(ref_img, dist_img, params).map_err(|e| Error::Metric {
            kind: "butter",
            message: format!("butteraugli_linear: {e:?}"),
        })?;
    Ok((
        Score {
            value: result.score,
            metric_name: "butter",
            metric_version: env!("CARGO_PKG_VERSION"),
        },
        result.pnorm_3,
    ))
}
