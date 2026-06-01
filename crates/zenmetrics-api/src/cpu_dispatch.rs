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
    /// scorer is stateless so we only stash the dims.
    #[cfg(feature = "cpu-ssim2")]
    Ssim2 { width: u32, height: u32 },
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
    /// per-call `RgbSlice`, so stash them here.
    #[cfg(feature = "cpu-zensim")]
    Zensim {
        inner: Box<zensim::Zensim>,
        width: u32,
        height: u32,
    },
    /// `dssim-core` — holds the `Dssim` config object; images are built
    /// per call. Lower is better (0 = identical).
    #[cfg(feature = "cpu-dssim")]
    Dssim {
        inner: Box<dssim_core::Dssim>,
        width: u32,
        height: u32,
    },
    /// `butteraugli` — free-function scorer; hold the params. Lower is
    /// better (0 = identical).
    #[cfg(feature = "cpu-butter")]
    Butter {
        params: butteraugli::ButteraugliParams,
        width: u32,
        height: u32,
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
            MetricKind::Ssim2 => Ok(CpuMetricState::Ssim2 { width, height }),
            #[cfg(feature = "cpu-cvvdp")]
            MetricKind::Cvvdp => {
                // cvvdp re-exports cvvdp-gpu's `CvvdpParams`, the same
                // struct the umbrella wraps in `MetricParams::Cvvdp` — no
                // translation needed (see cpu_adapter::construct_cvvdp).
                let p = match params {
                    MetricParams::Cvvdp(p) => p.clone(),
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
                })
            }
            #[cfg(feature = "cpu-butter")]
            MetricKind::Butter => {
                // butteraugli CPU defaults; the umbrella's MetricParams::Butter
                // wraps butteraugli-gpu's params (different type — not lifted).
                // Mirrors cpu_adapter::construct_butter.
                Ok(CpuMetricState::Butter {
                    params: butteraugli::ButteraugliParams::new(),
                    width,
                    height,
                })
            }
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
            CpuMetricState::Ssim2 { width, height } => (*width, *height),
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
    pub(crate) fn compute_srgb_u8(&mut self, r: &[u8], d: &[u8]) -> Result<Score> {
        match self {
            #[cfg(feature = "cpu-ssim2")]
            CpuMetricState::Ssim2 { width, height } => compute_ssim2(*width, *height, r, d),
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
            } => compute_zensim(inner, *width, *height, r, d),
            #[cfg(feature = "cpu-dssim")]
            CpuMetricState::Dssim {
                inner,
                width,
                height,
            } => compute_dssim(inner, *width, *height, r, d),
            #[cfg(feature = "cpu-butter")]
            CpuMetricState::Butter {
                params,
                width,
                height,
            } => compute_butter(params, *width, *height, r, d),
            CpuMetricState::FeatureDisabled(_) => Err(Error::BackendNotEnabled { backend: "cpu" }),
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
