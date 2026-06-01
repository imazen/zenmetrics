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
/// the rest.
pub(crate) enum CpuMetricState {
    /// `fast-ssim2` (Imazen, SIMD) — sRGB→linear→XYB is internal.
    #[cfg(feature = "cpu-ssim2")]
    Ssim2 { width: usize, height: usize },
    /// `kind`'s `cpu-*` feature is not built — optimized-CPU scoring for
    /// it is unavailable in this configuration.
    FeatureDisabled(MetricKind),
}

impl CpuMetricState {
    /// Build the optimized-CPU scorer for `kind` at `width × height`.
    /// Cheap (no device init); returns [`CpuMetricState::FeatureDisabled`]
    /// for metrics whose `cpu-*` feature is off rather than failing, so
    /// the error surfaces at score time with a clear backend message.
    pub(crate) fn new(
        kind: MetricKind,
        width: u32,
        height: u32,
        _params: &MetricParams,
    ) -> Result<Self> {
        let _ = (width, height);
        match kind {
            #[cfg(feature = "cpu-ssim2")]
            MetricKind::Ssim2 => Ok(CpuMetricState::Ssim2 {
                width: width as usize,
                height: height as usize,
            }),
            other => Ok(CpuMetricState::FeatureDisabled(other)),
        }
    }

    /// The metric this state scores.
    pub(crate) fn kind(&self) -> MetricKind {
        match self {
            #[cfg(feature = "cpu-ssim2")]
            CpuMetricState::Ssim2 { .. } => MetricKind::Ssim2,
            CpuMetricState::FeatureDisabled(k) => *k,
        }
    }

    /// Image dimensions this scorer was constructed for. `(0, 0)` for a
    /// feature-disabled state (it was never given real dims).
    pub(crate) fn dims(&self) -> (u32, u32) {
        match self {
            #[cfg(feature = "cpu-ssim2")]
            CpuMetricState::Ssim2 { width, height } => (*width as u32, *height as u32),
            CpuMetricState::FeatureDisabled(_) => (0, 0),
        }
    }

    /// One-shot score of a packed sRGB `R, G, B, R, G, B, …` pair
    /// (`width × height × 3` bytes per side).
    pub(crate) fn compute_srgb_u8(&mut self, r: &[u8], d: &[u8]) -> Result<Score> {
        match self {
            #[cfg(feature = "cpu-ssim2")]
            CpuMetricState::Ssim2 { width, height } => compute_ssim2(*width, *height, r, d),
            CpuMetricState::FeatureDisabled(_) => Err(Error::BackendNotEnabled { backend: "cpu" }),
        }
    }
}

// ---------------------------------------------------------------------------
// fast-ssim2 wiring — mirrors cpu_adapter::{ssim2_image_ref, compute_ssim2}.
// ---------------------------------------------------------------------------

/// Borrow an interleaved sRGB-u8 buffer as `ImgRef<'_, [u8; 3]>`.
/// `[u8; 3]` is `bytemuck::Pod`, so this reinterprets in place — no copy.
/// fast-ssim2's `ToLinearRgb for ImgRef<'_, [u8; 3]>` reads the triplets
/// directly and handles sRGB→linear→XYB internally.
#[cfg(feature = "cpu-ssim2")]
fn ssim2_image_ref<'a>(bytes: &'a [u8], w: usize, h: usize) -> imgref::ImgRef<'a, [u8; 3]> {
    let pixels: &[[u8; 3]] = bytemuck::cast_slice(bytes);
    imgref::ImgRef::new(pixels, w, h)
}

#[cfg(feature = "cpu-ssim2")]
fn compute_ssim2(width: usize, height: usize, r: &[u8], d: &[u8]) -> Result<Score> {
    let expected = width * height * 3;
    if r.len() != expected || d.len() != expected {
        return Err(Error::Metric {
            kind: "ssim2",
            message: format!(
                "cpu ssim2: expected {expected} packed sRGB bytes per side ({width}×{height}×3), \
                 got ref={} dist={}",
                r.len(),
                d.len()
            ),
        });
    }
    let ref_img = ssim2_image_ref(r, width, height);
    let dist_img = ssim2_image_ref(d, width, height);
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
