//! HDR scoring front-end (gated by the `hdr` feature).
//!
//! Decodes HDR sources — EXR (absolute-luminance), Ultra HDR JPEG, gain-map
//! HEIC — to absolute-luminance RGB (cd/m²), then preps per metric:
//!   - **SDR metrics** (ssim2/dssim/CPU-butteraugli): HDR→u8 via [`HdrTransfer`]
//!     (default `pu-rescale` — PU21 rescaled to fit u8 with NO highlight clamp;
//!     validated best vs HDR MOS on UPIQ, ssim2 0.55→0.65 SRCC). `pu-clamp` is
//!     the legacy degraded path. See `benchmarks/hdr_feeding_validation_2026-06-03.md`.
//!   - **cvvdp** (GPU): the **faithful** path — split into display-relative
//!     `[0,1]` f32 planes (`to_cvvdp_linear_planes`) fed to cvvdp's native
//!     `score_from_linear_planes` with an HDR `DisplayModel`. No u8 round-trip;
//!     cvvdp reconstructs `≈nits` and runs its CSF at the true HDR peak. (The
//!     `to_cvvdp_rgb8` sRGB8 path is the fallback when cvvdp can't take planes.)
//!
//! Decode mirrors `zenhdr-corpus`; PU21/transfer mirror `zensim::{pu21,transfer}`
//! (the canonical copy is `zenmetrics_api::hdr`). See zensim `docs/HDR_PLAN.md`.

use std::path::Path;

use crate::decode::Rgb8Image;

type Err = Box<dyn std::error::Error>;

/// `ultrahdr-rs` LinearFloat output: 1.0 = SDR white = 203 cd/m² (BT.2408).
const SDR_WHITE_NITS: f32 = 203.0;
/// HDR display headroom to reconstruct from gain-map sources (4× ≈ 812 nits).
const DISPLAY_BOOST: f32 = 4.0;

/// Absolute-luminance image: interleaved RGB f32 in cd/m².
pub struct NitsImage {
    pub rgb: Vec<f32>,
    pub width: u32,
    pub height: u32,
}

impl NitsImage {
    fn max_luma(&self) -> f32 {
        self.rgb
            .chunks_exact(3)
            .map(|p| 0.2627 * p[0] + 0.6780 * p[1] + 0.0593 * p[2])
            .fold(0.0f32, f32::max)
    }
}

// ── PU21 (banding_glare) — canonical coeffs, see zenmetrics_api::hdr ──────────
#[inline]
fn pu21_encode(y: f32) -> f32 {
    const P: [f32; 7] = [
        0.353_487_9,
        0.373_465_86,
        8.277_049e-5,
        0.906_256_26,
        0.091_503_03,
        0.909_951_7,
        596.314_8,
    ];
    let y = y.clamp(0.005, 10000.0);
    let yp = y.powf(P[3]);
    let inner = (P[0] + P[1] * yp) / (1.0 + P[2] * yp);
    (P[6] * (inner.powf(P[4]) - P[5])).max(0.0)
}

#[inline]
fn srgb_oetf(lin: f32) -> f32 {
    if lin <= 0.003_130_8 {
        12.92 * lin
    } else {
        1.055 * lin.powf(1.0 / 2.4) - 0.055
    }
}

/// PQ (SMPTE ST.2084) inverse-EOTF: absolute luminance (cd/m²) → coded `[0,1]`.
/// PQ is designed so the full 0..10000 cd/m² HDR range fits `[0,1]` — so a u8
/// quantization of PQ has NO highlight-clamp (unlike PU21, whose range
/// overshoots 256). This is the validated way to feed HDR to an SDR metric's
/// 8-bit path (AIC-HDR2025 fed SSIMULACRA2 the PQ signal → SROCC ≈ 0.91).
#[inline]
fn pq_oetf(nits: f32) -> f32 {
    const M1: f32 = 0.159_301_76;
    const M2: f32 = 78.84375;
    const C1: f32 = 0.835_937_5;
    const C2: f32 = 18.851_562;
    const C3: f32 = 18.6875;
    let y = (nits / 10000.0).clamp(0.0, 1.0);
    let yp = y.powf(M1);
    ((C1 + C2 * yp) / (1.0 + C3 * yp)).powf(M2)
}

// ── Decode → absolute-luminance nits ─────────────────────────────────────────

/// Decode an HDR source (EXR / Ultra HDR JPEG / gain-map HEIC) to
/// absolute-luminance interleaved RGB f32 (cd/m²).
pub fn decode_to_nits(path: &Path) -> Result<NitsImage, Err> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "exr" => decode_exr(path),
        "heic" | "heif" => decode_heic(path),
        "jpg" | "jpeg" => decode_ultrahdr_jpeg(path),
        other => Err(format!("unsupported HDR input extension: .{other}").into()),
    }
}

fn decode_exr(path: &Path) -> Result<NitsImage, Err> {
    // The corpus EXRs store absolute photometric units (cd/m²) directly.
    let rgb = image::open(path)?.to_rgb32f();
    let (w, h) = (rgb.width(), rgb.height());
    Ok(NitsImage {
        rgb: rgb.into_raw(),
        width: w,
        height: h,
    })
}

fn decode_ultrahdr_jpeg(path: &Path) -> Result<NitsImage, Err> {
    let bytes = std::fs::read(path)?;
    let dec = ultrahdr_rs::Decoder::new(&bytes).map_err(|e| format!("{e:?}"))?;
    if !dec.is_ultrahdr() {
        return Err("JPEG has no gain map (SDR)".into());
    }
    let hdr = dec
        .decode_hdr(DISPLAY_BOOST)
        .map_err(|e| format!("{e:?}"))?;
    Ok(pixelbuffer_to_nits(&hdr))
}

fn decode_heic(path: &Path) -> Result<NitsImage, Err> {
    use ultrahdr_core::gainmap::apply::apply_gainmap;
    use ultrahdr_core::{
        ColorPrimaries, GainMap, HdrOutputFormat, Iso21496Format, PixelFormat, TransferFunction,
        parse_iso21496_fmt, pixel_buffer_from_vec,
    };

    let bytes = std::fs::read(path)?;
    let cfg = heic::DecoderConfig::new();
    if !cfg.has_gain_map(&bytes).map_err(|e| format!("{e:?}"))? {
        return Err("HEIC has no gain map (SDR / PQ)".into());
    }
    let base = cfg
        .decode(&bytes, heic::PixelLayout::Rgb8)
        .map_err(|e| format!("base decode: {e:?}"))?;
    let gm = cfg
        .decode_gain_map(&bytes)
        .map_err(|e| format!("gainmap decode: {e:?}"))?;
    let iso = gm
        .iso21496
        .ok_or("HEIC gain map has no ISO 21496-1 tmap metadata (legacy Apple aux only)")?;
    let params = parse_iso21496_fmt(&iso, Iso21496Format::AvifTmap)
        .map_err(|e| format!("iso parse: {e:?}"))?;
    let sdr = pixel_buffer_from_vec(
        base.data,
        base.width,
        base.height,
        PixelFormat::Rgb8,
        ColorPrimaries::DisplayP3,
        TransferFunction::Srgb,
    )
    .map_err(|e| format!("base buffer: {e:?}"))?;
    let gainmap = GainMap {
        width: gm.width,
        height: gm.height,
        channels: 1,
        data: gm.data,
    };
    let hdr = apply_gainmap(
        &sdr,
        &gainmap,
        &params,
        DISPLAY_BOOST,
        HdrOutputFormat::LinearFloat,
        ultrahdr_core::Unstoppable,
    )
    .map_err(|e| format!("apply: {e:?}"))?;
    Ok(pixelbuffer_to_nits(&hdr))
}

/// ultrahdr PixelBuffer (RGBA f32, 1.0 = SDR white) → tight RGB in nits.
fn pixelbuffer_to_nits(hdr: &ultrahdr_core::PixelBuffer) -> NitsImage {
    let w = hdr.width() as usize;
    let h = hdr.height() as usize;
    let raw = hdr.as_slice().as_strided_bytes();
    let f: &[f32] = bytemuck::cast_slice(raw);
    let row_px = f.len() / 4 / h;
    let mut rgb = vec![0.0f32; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let si = (y * row_px + x) * 4;
            let di = (y * w + x) * 3;
            rgb[di] = f[si] * SDR_WHITE_NITS;
            rgb[di + 1] = f[si + 1] * SDR_WHITE_NITS;
            rgb[di + 2] = f[si + 2] * SDR_WHITE_NITS;
        }
    }
    NitsImage {
        rgb,
        width: w as u32,
        height: h as u32,
    }
}

// ── Per-metric prep ──────────────────────────────────────────────────────────

/// HDR→u8 transfer for the SDR-metric path. The metric receives an 8-bit
/// signal; which transfer is used decides whether the HDR highlight range
/// survives. (Literature recipe: Mantiuk-Azimi PU21 full-precision, or PQ —
/// see `benchmarks/hdr_feeding_validation_*`.)
#[derive(Copy, Clone, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum HdrTransfer {
    /// PU21 (banding_glare) **clamped** to u8 — collapses everything above
    /// ~100 cd/m² to 255 (PU21 ranges to ~600). DEGRADES highlights; legacy.
    PuClamp,
    /// PQ (SMPTE ST.2084) → u8. The full 0..10000 cd/m² range fits `[0,1]` by
    /// design, so there is **no highlight clamp**. Validated (AIC-HDR2025).
    Pq,
    /// PU21 rescaled so the display peak (`HDR_DISPLAY_PEAK_NITS`) maps to 255
    /// — keeps the full PU range (no collapse), u8-quantized.
    PuRescale,
}

/// Encode absolute-luminance RGB to sRGB8 for the SDR-metric kernels via the
/// chosen [`HdrTransfer`].
pub fn to_sdr_rgb8(img: &NitsImage, transfer: HdrTransfer) -> Rgb8Image {
    let pu_max = pu21_encode(HDR_DISPLAY_PEAK_NITS).max(1.0);
    let enc = |y: f32| -> u8 {
        let v = match transfer {
            HdrTransfer::PuClamp => pu21_encode(y),
            HdrTransfer::Pq => pq_oetf(y) * 255.0,
            HdrTransfer::PuRescale => pu21_encode(y) * (255.0 / pu_max),
        };
        v.round().clamp(0.0, 255.0) as u8
    };
    Rgb8Image {
        pixels: img.rgb.iter().map(|&y| enc(y)).collect(),
        width: img.width,
        height: img.height,
    }
}

/// Back-compat: PU21-clamp prep (the legacy degraded path). Prefer
/// [`to_sdr_rgb8`] with an explicit [`HdrTransfer`].
pub fn to_pu_rgb8(img: &NitsImage) -> Rgb8Image {
    to_sdr_rgb8(img, HdrTransfer::PuClamp)
}

/// Normalize by the display peak → sRGB-encode → sRGB8, for cvvdp (which scales
/// sRGB back to its display peak internally). Returns `(image, peak_nits)` so
/// the caller can build the matching display model.
pub fn to_cvvdp_rgb8(img: &NitsImage) -> (Rgb8Image, f32) {
    // Cap the peak at a sane HDR display (1000 cd/m²) so a stray super-bright
    // pixel doesn't crush everything else into the bottom of the u8 range.
    let peak = img.max_luma().clamp(SDR_WHITE_NITS, 1000.0);
    let pixels = img
        .rgb
        .iter()
        .map(|&y| (srgb_oetf((y / peak).clamp(0.0, 1.0)) * 255.0).round() as u8)
        .collect();
    (
        Rgb8Image {
            pixels,
            width: img.width,
            height: img.height,
        },
        peak,
    )
}

/// Reference HDR display peak (cd/m²) for the faithful cvvdp path. Content is
/// normalized display-relative to this peak; cvvdp's CSF adapts its sensitivity
/// to it, and content brighter than this clips (as a real display would). 1000
/// cd/m² is the common HDR mastering target. Must match the `y_peak` the caller
/// gives `DisplayTarget::hdr`.
pub const HDR_DISPLAY_PEAK_NITS: f32 = 1000.0;

/// Split absolute-luminance RGB (cd/m²) into the three display-relative `[0,1]`
/// planes the **faithful** GPU HDR paths expect — no u8 round-trip, full
/// highlight precision up to the display peak. `v = nits / HDR_DISPLAY_PEAK_NITS`
/// clamped to `[0,1]`. Tightly packed (`padded_width == width`). Shared by
/// cvvdp's `score_from_linear_planes` (paired with `DisplayTarget::hdr(peak)`,
/// reconstructs `≈nits`) and butteraugli's linear-planes path (paired with
/// `intensity_target = peak`, its opsin scales `v` back to `≈nits`).
pub fn to_cvvdp_linear_planes(img: &NitsImage) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = (img.width as usize) * (img.height as usize);
    let inv = 1.0 / HDR_DISPLAY_PEAK_NITS;
    let (mut r, mut g, mut b) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for px in img.rgb.chunks_exact(3) {
        r.push((px[0] * inv).clamp(0.0, 1.0));
        g.push((px[1] * inv).clamp(0.0, 1.0));
        b.push((px[2] * inv).clamp(0.0, 1.0));
    }
    (r, g, b)
}

// ─── Umbrella HDR-aware scoring (HdrScorer) ───────────────────────────────────

/// Map a CLI metric to its umbrella [`zenmetrics_api::MetricKind`] when the
/// corresponding GPU backend feature is enabled. `None` means the metric has no
/// umbrella HDR path (CPU metrics, or the `gpu-*` feature is off) — the caller
/// falls back to the u8 path.
fn to_umbrella_kind(m: crate::metrics::MetricKind) -> Option<zenmetrics_api::MetricKind> {
    use crate::metrics::MetricKind as C;
    use zenmetrics_api::MetricKind as U;
    #[allow(unreachable_patterns)]
    match m {
        // CPU metric variants → the umbrella's native CPU path (Backend::Cpu,
        // selected in `score_via_hdr_scorer`). The `hdr` feature forwards
        // `zenmetrics-api/cpu-metrics`, so the umbrella can always run these on
        // CPU — and they get the SAME validated feeding (butter/cvvdp linear,
        // SSIM-family pu-rescale) as the GPU path, retiring the hand-rolled
        // CPU u8 conversion. Never `Backend::CubeclCpu`.
        C::Ssim2 => Some(U::Ssim2),
        C::Butteraugli => Some(U::Butter),
        C::Dssim => Some(U::Dssim),
        C::Zensim => Some(U::Zensim),
        // GPU metric variants → the umbrella GPU path (gated on the gpu-* feature).
        #[cfg(feature = "gpu-cvvdp")]
        C::Cvvdp => Some(U::Cvvdp),
        #[cfg(feature = "gpu-butteraugli")]
        C::ButteraugliGpu => Some(U::Butter),
        #[cfg(feature = "gpu-ssim2")]
        C::Ssim2Gpu => Some(U::Ssim2),
        #[cfg(feature = "gpu-dssim")]
        C::DssimGpu => Some(U::Dssim),
        #[cfg(feature = "gpu-iwssim")]
        C::IwssimGpu => Some(U::Iwssim),
        #[cfg(feature = "gpu-zensim")]
        C::ZensimGpu => Some(U::Zensim),
        _ => None,
    }
}

fn to_umbrella_transfer(t: HdrTransfer) -> zenmetrics_api::hdr::HdrTransfer {
    match t {
        HdrTransfer::PuClamp => zenmetrics_api::hdr::HdrTransfer::PuClamp,
        HdrTransfer::Pq => zenmetrics_api::hdr::HdrTransfer::Pq,
        HdrTransfer::PuRescale => zenmetrics_api::hdr::HdrTransfer::PuRescale,
    }
}

/// Score an HDR pair through the umbrella's HDR-aware [`zenmetrics_api::hdr::HdrScorer`],
/// mapping the lossless `Scores` back to the CLI's `(column, value)` output rows
/// using the canonical [`column_names`](crate::metrics::MetricKind::column_names)
/// (so the parquet/TSV schema is unchanged). Returns `None` when the metric has
/// no umbrella HDR path (caller falls back to the u8 / cvvdp-rgb8 path); the
/// umbrella opaque supports cuda/wgpu/cpu, not hip — hip also falls back.
pub fn score_via_hdr_scorer(
    metric: crate::metrics::MetricKind,
    r: &NitsImage,
    d: &NitsImage,
    transfer: HdrTransfer,
    runtime: crate::metrics::GpuRuntime,
) -> Option<Result<Vec<(&'static str, f64)>, Box<dyn std::error::Error>>> {
    let kind = to_umbrella_kind(metric)?;
    let backend = if metric.requires_gpu() {
        // GPU metric variant → the runtime's backend, but only cuda/wgpu: the
        // umbrella opaque can't express hip or cubecl-cpu (and we never want
        // CubeclCpu), so those have no umbrella HDR path and fall back.
        #[cfg(any(
            feature = "gpu-butteraugli",
            feature = "gpu-ssim2",
            feature = "gpu-dssim",
            feature = "gpu-iwssim",
            feature = "gpu-zensim",
            feature = "gpu-cvvdp"
        ))]
        {
            match crate::metrics::gpu_runtime_to_backend(runtime) {
                Ok(b @ (zenmetrics_api::Backend::Cuda | zenmetrics_api::Backend::Wgpu)) => b,
                _ => return None,
            }
        }
        #[cfg(not(any(
            feature = "gpu-butteraugli",
            feature = "gpu-ssim2",
            feature = "gpu-dssim",
            feature = "gpu-iwssim",
            feature = "gpu-zensim",
            feature = "gpu-cvvdp"
        )))]
        {
            let _ = runtime;
            return None;
        }
    } else {
        // CPU metric variant → the umbrella's native CPU backend (the `hdr`
        // feature forwards `zenmetrics-api/cpu-metrics`). Never CubeclCpu.
        let _ = runtime;
        zenmetrics_api::Backend::Cpu
    };
    Some(score_via_hdr_scorer_inner(
        metric, kind, backend, r, d, transfer,
    ))
}

fn score_via_hdr_scorer_inner(
    metric: crate::metrics::MetricKind,
    kind: zenmetrics_api::MetricKind,
    backend: zenmetrics_api::Backend,
    r: &NitsImage,
    d: &NitsImage,
    transfer: HdrTransfer,
) -> Result<Vec<(&'static str, f64)>, Box<dyn std::error::Error>> {
    let mut scorer = zenmetrics_api::hdr::HdrScorer::new(
        kind,
        backend,
        r.width,
        r.height,
        HDR_DISPLAY_PEAK_NITS,
    )?
    .with_transfer(to_umbrella_transfer(transfer));
    let scores = scorer.compute_multi(&r.rgb, &d.rgb)?;
    let cols = metric.column_names();
    let rows: Vec<(&'static str, f64)> = if cols.len() >= 2 {
        // butteraugli: max + libjxl pnorm_3.
        vec![
            (
                cols[0],
                scores.get("max").unwrap_or_else(|| scores.primary()),
            ),
            (cols[1], scores.get("pnorm_3").unwrap_or(f64::NAN)),
        ]
    } else {
        vec![(cols[0], scores.primary())]
    };
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pu21_100_nits_near_256() {
        assert!((pu21_encode(100.0) - 256.0).abs() < 1.5);
    }

    #[test]
    fn pu_clamp_collapses_highlights_pq_and_rescale_do_not() {
        // Two distinct HDR highlights: 600 vs 4000 cd/m². The legacy PU-clamp
        // maps BOTH to 255 (the bug — they become indistinguishable). PQ and
        // PU-rescale keep them distinct, preserving the highlight signal that
        // makes PU21 correlate with HDR MOS (UPIQ: 0.55 clamp → 0.65 rescale).
        let lo = NitsImage {
            rgb: vec![600.0, 600.0, 600.0],
            width: 1,
            height: 1,
        };
        let hi = NitsImage {
            rgb: vec![4000.0, 4000.0, 4000.0],
            width: 1,
            height: 1,
        };
        // PU-clamp: both pin at 255 (collapsed).
        assert_eq!(to_sdr_rgb8(&lo, HdrTransfer::PuClamp).pixels[0], 255);
        assert_eq!(to_sdr_rgb8(&hi, HdrTransfer::PuClamp).pixels[0], 255);
        // PQ + PU-rescale: 600 < 4000 stays distinct (no collapse).
        for t in [HdrTransfer::Pq, HdrTransfer::PuRescale] {
            let l = to_sdr_rgb8(&lo, t).pixels[0];
            let h = to_sdr_rgb8(&hi, t).pixels[0];
            assert!(l < h, "{t:?}: 600cd/m² ({l}) should be < 4000cd/m² ({h})");
            assert!(l < 255, "{t:?}: 600cd/m² should not pin at the ceiling");
        }
    }

    #[test]
    fn metric_prep_shapes() {
        // px0 = 600 cd/m² white highlight, px1 = 5 cd/m² shadow.
        let img = NitsImage {
            rgb: vec![600.0, 600.0, 600.0, 5.0, 5.0, 5.0],
            width: 2,
            height: 1,
        };
        let pu = to_pu_rgb8(&img);
        assert_eq!(pu.pixels.len(), 6);
        assert_eq!(pu.pixels[0], 255); // 600 cd/m² highlight clamps at the u8 ceiling
        assert!(pu.pixels[3] < 200); // 5 cd/m² shadow is well below it
        let (cv, peak) = to_cvvdp_rgb8(&img);
        assert_eq!(cv.pixels.len(), 6);
        assert!((peak - 600.0).abs() < 1.0); // luma-weighted max drives the peak
        assert_eq!(cv.pixels[0], 255); // highlight normalizes to peak → sRGB 255
    }

    #[test]
    fn cvvdp_linear_planes_are_display_relative() {
        // px0 = peak-bright, px1 = 100 cd/m² (0.1 of a 1000-nit display).
        let img = NitsImage {
            rgb: vec![
                HDR_DISPLAY_PEAK_NITS,
                HDR_DISPLAY_PEAK_NITS,
                HDR_DISPLAY_PEAK_NITS,
                100.0,
                100.0,
                100.0,
            ],
            width: 2,
            height: 1,
        };
        let (r, g, b) = to_cvvdp_linear_planes(&img);
        assert_eq!((r.len(), g.len(), b.len()), (2, 2, 2));
        assert!((r[0] - 1.0).abs() < 1e-6); // peak → 1.0 (no u8 clamp)
        assert!((r[1] - 0.1).abs() < 1e-6); // 100/1000 → 0.1, full precision
        // A 4000-nit highlight clamps to 1.0 (display can't exceed its peak).
        let bright = NitsImage {
            rgb: vec![4000.0, 4000.0, 4000.0],
            width: 1,
            height: 1,
        };
        assert!((to_cvvdp_linear_planes(&bright).0[0] - 1.0).abs() < 1e-6);
    }
}
