//! HDR scoring front-end (gated by the `hdr` feature).
//!
//! Decodes HDR sources — EXR (absolute-luminance), Ultra HDR JPEG, gain-map
//! HEIC — to absolute-luminance RGB (cd/m²), then preps per metric:
//!   - **Primary path**: [`score_via_hdr_scorer`] hands absolute nits to the
//!     umbrella's `HdrScorer`, which applies the per-metric feeding from
//!     `zenmetrics_api::hdr::hdr_feeding` — cvvdp/butter linear planes, ssim2
//!     integrated PU21 on every backend, CPU zensim integrated PU, iwssim
//!     float PU(luma) on every backend, the remaining SSIM-family the u8
//!     shell. `--hdr-transfer` only affects the u8-shell metrics (it cannot
//!     override the integrated/float feedings).
//!   - **Fallback** (kinds with no umbrella mapping / hip runtime): HDR→u8 via
//!     [`HdrTransfer`] (default `pu-rescale`; `pu-clamp` is the legacy degraded
//!     path). See `benchmarks/hdr_feeding_validation_2026-06-03.md`.
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
/// Also the cvvdp-rgb8 peak floor in [`to_cvvdp_rgb8`] (core path).
const SDR_WHITE_NITS: f32 = 203.0;
/// HDR display headroom to reconstruct from gain-map sources (4× ≈ 812 nits).
#[cfg(feature = "hdr-gainmap")]
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

#[inline]
fn srgb_oetf(lin: f32) -> f32 {
    if lin <= 0.003_130_8 {
        12.92 * lin
    } else {
        1.055 * lin.powf(1.0 / 2.4) - 0.055
    }
}

// ── Decode → absolute-luminance nits ─────────────────────────────────────────

/// Decode an HDR source (EXR / Ultra HDR JPEG / gain-map HEIC / 16-bit PQ
/// PNG with cICP) to absolute-luminance interleaved RGB f32 (cd/m²).
pub fn decode_to_nits(path: &Path) -> Result<NitsImage, Err> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "exr" => decode_exr(path),
        // Gain-map sources (Ultra HDR JPEG / HEIC) need the opt-in
        // `hdr-gainmap` feature — their decoders (ultrahdr-rs / heic) are not
        // in the default build. Core HDR covers PQ-PNG / PQ-JXL / EXR.
        #[cfg(feature = "hdr-gainmap")]
        "heic" | "heif" => decode_heic(path),
        #[cfg(feature = "hdr-gainmap")]
        "jpg" | "jpeg" => decode_ultrahdr_jpeg(path),
        #[cfg(not(feature = "hdr-gainmap"))]
        "heic" | "heif" | "jpg" | "jpeg" => Err(format!(
            "HDR gain-map source .{ext} needs the `hdr-gainmap` build feature \
             (Ultra HDR JPEG / gain-map HEIC decoders); this build has core HDR \
             only, which covers PQ-PNG / PQ-JXL / EXR references"
        )
        .into()),
        "png" => decode_pq_png(path),
        "jxl" => decode_pq_jxl(path),
        other => Err(format!("unsupported HDR input extension: .{other}").into()),
    }
}

/// Decode a JPEG XL HDR variant (16-bit PQ + CICP) to absolute nits. This is
/// the decode-back path the HDR datagen split needs: `score-pairs --hdr` takes
/// the encoded `.jxl` directly as the distorted image (the SDR split feeds the
/// encoded file as `dist_path`; the HDR split does the same, but the variant is
/// JXL and carries PQ signaling). Mirrors the sweep's
/// `sweep::hdr::decode_jxl_to_nits` exactly: decode → require PQ cICP (or a PQ
/// descriptor transfer) → PQ EOTF to cd/m². A variant that lost its HDR color
/// encoding is refused, never approximated.
#[cfg(feature = "jxl")]
fn decode_pq_jxl(path: &Path) -> Result<NitsImage, Err> {
    let data = std::fs::read(path)?;
    let output = zenjxl::decode(&data, None, &[]).map_err(|e| format!("zenjxl: {e}"))?;
    let cicp_is_pq = matches!(output.info.cicp, Some((_, 16, _, _)));
    let desc_is_pq =
        output.pixels.as_slice().descriptor().transfer() == zenpixels::TransferFunction::Pq;
    if !cicp_is_pq && !desc_is_pq {
        return Err(format!(
            "HDR JXL decode: variant carries no PQ signaling (info.cicp={:?}, \
             descriptor transfer={:?}) — the codec did not round-trip the HDR \
             color encoding, so this is not an HDR variant (refusing to guess a \
             nits scale). Score it through the SDR path instead (drop --hdr).",
            output.info.cicp,
            output.pixels.as_slice().descriptor().transfer(),
        )
        .into());
    }
    pq_slice_to_nits(&output.pixels.as_slice())
}

#[cfg(not(feature = "jxl"))]
fn decode_pq_jxl(_path: &Path) -> Result<NitsImage, Err> {
    Err("JPEG XL HDR decode requires the `jxl` build feature".into())
}

/// Strided **PQ-coded** `PixelSlice` (validated by the caller via the
/// codestream CICP / descriptor) → absolute nits via the PQ EOTF. u8 / u16
/// samples normalise to `[0,1]` code values first; f32 samples ARE the code
/// values (the JXL decoder's f32 output is PQ-coded, not linear, when the
/// codestream carries CICP PQ). Alpha drops, gray broadcasts. Mirrors
/// `sweep::hdr::pq_slice_to_nits` so the inline-sweep and score-pairs HDR
/// decode-back paths agree bit-for-bit.
#[cfg(feature = "jxl")]
fn pq_slice_to_nits(s: &zenpixels::PixelSlice<'_>) -> Result<NitsImage, Err> {
    use zenmetrics_api::hdr::pq_eotf;
    use zenpixels::{ChannelLayout, ChannelType};

    let desc = s.descriptor();
    let (w, h) = (s.width() as usize, s.rows() as usize);
    let channels: usize = match desc.layout() {
        ChannelLayout::Rgb => 3,
        ChannelLayout::Rgba => 4,
        ChannelLayout::Gray => 1,
        ChannelLayout::GrayAlpha => 2,
        other => {
            return Err(format!("HDR JXL decode: unsupported channel layout {other:?}").into());
        }
    };
    let color_channels = if channels >= 3 { 3 } else { 1 };
    let bytes = s.as_strided_bytes();
    let stride = s.stride();
    let mut rgb = Vec::with_capacity(w * h * 3);
    let push_px = |rgb: &mut Vec<f32>, px: &[f32]| {
        if color_channels == 1 {
            let v = pq_eotf(px[0]);
            rgb.extend_from_slice(&[v, v, v]);
        } else {
            rgb.extend_from_slice(&[pq_eotf(px[0]), pq_eotf(px[1]), pq_eotf(px[2])]);
        }
    };
    match desc.channel_type() {
        ChannelType::U16 => {
            let row_bytes = w * channels * 2;
            for y in 0..h {
                let row = &bytes[y * stride..y * stride + row_bytes];
                let mut px = [0f32; 4];
                for (x, sample) in row.chunks_exact(2).enumerate() {
                    px[x % channels] =
                        f32::from(u16::from_ne_bytes([sample[0], sample[1]])) / 65535.0;
                    if x % channels == channels - 1 {
                        push_px(&mut rgb, &px);
                    }
                }
            }
        }
        ChannelType::U8 => {
            let row_bytes = w * channels;
            for y in 0..h {
                let row = &bytes[y * stride..y * stride + row_bytes];
                let mut px = [0f32; 4];
                for (x, &sample) in row.iter().enumerate() {
                    px[x % channels] = f32::from(sample) / 255.0;
                    if x % channels == channels - 1 {
                        push_px(&mut rgb, &px);
                    }
                }
            }
        }
        ChannelType::F32 => {
            let row_bytes = w * channels * 4;
            for y in 0..h {
                let row = &bytes[y * stride..y * stride + row_bytes];
                let mut px = [0f32; 4];
                for (x, sample) in row.chunks_exact(4).enumerate() {
                    px[x % channels] =
                        f32::from_ne_bytes([sample[0], sample[1], sample[2], sample[3]])
                            .clamp(0.0, 1.0);
                    if x % channels == channels - 1 {
                        push_px(&mut rgb, &px);
                    }
                }
            }
        }
        other => {
            return Err(format!("HDR JXL decode: unsupported channel type {other:?}").into());
        }
    }
    Ok(NitsImage {
        rgb,
        width: w as u32,
        height: h as u32,
    })
}

/// Decode a PQ-PNG (PNG 3.0 cICP, transfer 16) to absolute nits.
///
/// Corpus contract (imazen-26-png-v2 `.hdr.png`): 16-bit samples are PQ
/// code values produced as `pq_oetf(linear · 203 / 10000)` with linear
/// `1.0` = SDR white — i.e. the PQ encoding of absolute light at the
/// BT.2408 203 cd/m² SDR-white anchor. The PQ EOTF alone therefore
/// recovers absolute cd/m² (SDR white lands at 203). Primaries (1 or 12)
/// pass through: metrics score ref and dist in the same primaries.
///
/// HLG (transfer 18) is rejected loudly — faithful HLG display light
/// needs the peak-dependent OOTF, which no zen scoring path implements.
/// Everything else (no cICP / SDR transfer) is rejected too: silently
/// treating SDR code values as PQ would produce garbage nits.
#[cfg(feature = "png")]
fn decode_pq_png(path: &Path) -> Result<NitsImage, Err> {
    let data = std::fs::read(path)?;
    let (rgb16, width, height, cicp) = png_to_rgb16_pq(&data)?;
    Ok(match cicp.transfer_characteristics {
        18 => rgb16_hlg_to_nits(&rgb16, width, height),
        _ => rgb16_pq_to_nits(&rgb16, width, height),
    })
}

#[cfg(not(feature = "png"))]
fn decode_pq_png(_path: &Path) -> Result<NitsImage, Err> {
    Err("PQ-PNG HDR decode requires the `png` build feature (zenpng)".into())
}

/// Decode PNG bytes and normalise to tight interleaved **RGB u16 PQ code
/// values**, validating the cICP HDR contract (transfer must be 16 = PQ).
/// Returns `(rgb16, width, height, cicp)` — the u16s are the raw PQ code
/// values (alpha dropped, gray broadcast, 8-bit scaled by 257), so the
/// buffer is both nits-convertible ([`rgb16_pq_to_nits`]) and directly
/// re-encodable as HDR input for codecs that take 16-bit + CICP.
#[cfg(feature = "png")]
pub(crate) fn png_to_rgb16_pq(data: &[u8]) -> Result<(Vec<u16>, u32, u32, zenpixels::Cicp), Err> {
    use zenpng::{PngDecodeConfig, decode};
    let cancel: Box<dyn enough::Stop + Send + Sync> = Box::new(enough::Unstoppable);
    let output = decode(data, &PngDecodeConfig::default(), &*cancel)?;
    let cicp = output.info.cicp.ok_or(
        "PNG carries no cICP chunk — not an HDR PQ PNG. \
         Score it through the SDR path instead (drop --hdr)",
    )?;
    match cicp.transfer_characteristics {
        16 => {}
        18 => {} // HLG — decoded via the BT.2100 OOTF (rgb16_hlg_to_nits)
        t => {
            return Err(format!(
                "PNG cICP transfer {t} is not an HDR transfer (PQ=16) — \
                 score it through the SDR path instead (drop --hdr)"
            )
            .into());
        }
    }
    let (rgb16, width, height) = slice_to_rgb16(&output.pixels.as_slice())?;
    Ok((rgb16, width, height, cicp))
}

/// Normalise any u8/u16 slice layout to tight interleaved RGB u16 **code
/// values** (no transfer math): alpha dropped, gray broadcast, u8 scaled
/// by 257 (0..255 → 0..65535). Strided input handled per row.
#[cfg(feature = "png")]
fn slice_to_rgb16(s: &zenpixels::PixelSlice<'_>) -> Result<(Vec<u16>, u32, u32), Err> {
    use zenpixels::{ChannelLayout, ChannelType};
    let (w, h) = (s.width() as usize, s.rows() as usize);
    let desc = s.descriptor();
    let channels: usize = match desc.layout() {
        ChannelLayout::Rgb => 3,
        ChannelLayout::Rgba => 4,
        ChannelLayout::Gray => 1,
        ChannelLayout::GrayAlpha => 2,
        other => return Err(format!("HDR PNG: unsupported channel layout {other:?}").into()),
    };
    let color_channels = if channels >= 3 { 3 } else { 1 };
    let bytes = s.as_strided_bytes();
    let stride = s.stride();
    let mut rgb16 = Vec::with_capacity(w * h * 3);
    let push_px = |rgb16: &mut Vec<u16>, px: &[u16]| {
        if color_channels == 1 {
            rgb16.extend_from_slice(&[px[0], px[0], px[0]]);
        } else {
            rgb16.extend_from_slice(&px[..3]);
        }
    };
    match desc.channel_type() {
        ChannelType::U16 => {
            let row_bytes = w * channels * 2;
            for y in 0..h {
                let row = &bytes[y * stride..y * stride + row_bytes];
                let mut px = [0u16; 4];
                for (x, sample) in row.chunks_exact(2).enumerate() {
                    px[x % channels] = u16::from_ne_bytes([sample[0], sample[1]]);
                    if x % channels == channels - 1 {
                        push_px(&mut rgb16, &px);
                    }
                }
            }
        }
        ChannelType::U8 => {
            let row_bytes = w * channels;
            for y in 0..h {
                let row = &bytes[y * stride..y * stride + row_bytes];
                let mut px = [0u16; 4];
                for (x, &sample) in row.iter().enumerate() {
                    px[x % channels] = u16::from(sample) * 257;
                    if x % channels == channels - 1 {
                        push_px(&mut rgb16, &px);
                    }
                }
            }
        }
        other => {
            return Err(format!("HDR PNG: unsupported channel type {other:?}").into());
        }
    }
    Ok((rgb16, w as u32, h as u32))
}

/// Tight interleaved RGB u16 PQ code values → absolute-luminance nits via
/// the SMPTE ST 2084 EOTF (`pq_eotf(v / 65535)`, output in cd/m²).
#[cfg(feature = "png")]

/// Decode interleaved **RGB u16 HLG code values** to absolute display nits
/// per BT.2100 (ARIB STD-B67): per-channel inverse OETF to scene-linear,
/// then the peak-dependent OOTF `RGB_d = (peak − black) · Y_s^(γ−1) · RGB_s`
/// with `Y_s = 0.2627 R + 0.6780 G + 0.0593 B` and
/// γ = [`cvvdp::params::hlg_system_gamma`] (1.2 at a 1000 cd/m² peak; the
/// in-repo reference used by cvvdp's own color kernel). Display: 1000 cd/m²
/// peak, 0.005 black, 100 lux ambient — the same nominal HDR display the PQ
/// path targets.
#[allow(dead_code)] // wired behind the png/jxl features
pub(crate) fn rgb16_hlg_to_nits(rgb16: &[u16], width: u32, height: u32) -> NitsImage {
    use cvvdp::params::{hlg_inverse_oetf_scalar, hlg_system_gamma};
    let (y_peak, y_black) = (1000.0f32, 0.005f32);
    let gamma = hlg_system_gamma(y_peak, 100.0);
    let mut rgb = Vec::with_capacity(rgb16.len());
    for px in rgb16.chunks_exact(3) {
        let r = hlg_inverse_oetf_scalar(f32::from(px[0]) / 65535.0);
        let g = hlg_inverse_oetf_scalar(f32::from(px[1]) / 65535.0);
        let b = hlg_inverse_oetf_scalar(f32::from(px[2]) / 65535.0);
        let ys = (0.2627 * r + 0.6780 * g + 0.0593 * b).max(1e-9);
        let scale = (y_peak - y_black) * ys.powf(gamma - 1.0);
        rgb.push(r * scale + y_black);
        rgb.push(g * scale + y_black);
        rgb.push(b * scale + y_black);
    }
    NitsImage { rgb, width, height }
}

pub(crate) fn rgb16_pq_to_nits(rgb16: &[u16], width: u32, height: u32) -> NitsImage {
    let rgb = rgb16
        .iter()
        .map(|&v| zenmetrics_api::hdr::pq_eotf(f32::from(v) / 65535.0))
        .collect();
    NitsImage { rgb, width, height }
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

#[cfg(feature = "hdr-gainmap")]
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

#[cfg(feature = "hdr-gainmap")]
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
#[cfg(feature = "hdr-gainmap")]
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
    // Delegate to the canonical encoder in zenmetrics-api (single PU21 copy).
    let pixels = zenmetrics_api::hdr::to_sdr_u8(
        &img.rgb,
        to_umbrella_transfer(transfer),
        HDR_DISPLAY_PEAK_NITS,
    );
    Rgb8Image {
        pixels,
        width: img.width,
        height: img.height,
    }
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
pub(crate) fn to_umbrella_kind(
    m: crate::metrics::MetricKind,
) -> Option<zenmetrics_api::MetricKind> {
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
        // The unsuffixed `cvvdp` is the native CPU port: it maps to the
        // umbrella's `Backend::Cpu` HDR path when `cpu-cvvdp` is compiled,
        // running the native `cvvdp` crate via `cpu_dispatch` (NEVER
        // cubecl-cpu). The backend (always Cpu for this variant) is chosen in
        // the sweep's `umbrella_kind_and_backend`.
        #[cfg(feature = "cpu-cvvdp")]
        C::Cvvdp => Some(U::Cvvdp),
        // The unsuffixed `iwssim` is likewise the native CPU port.
        #[cfg(feature = "cpu-iwssim")]
        C::Iwssim => Some(U::Iwssim),
        // GPU metric variants → the umbrella GPU path (gated on the gpu-* feature).
        #[cfg(feature = "gpu-cvvdp")]
        C::CvvdpGpu => Some(U::Cvvdp),
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
        assert!((zenmetrics_api::hdr::pu21_encode(100.0) - 256.0).abs() < 1.5);
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
        let pu = to_sdr_rgb8(&img, HdrTransfer::PuRescale);
        assert_eq!(pu.pixels.len(), 6);
        // PuRescale keeps the 600-nit highlight BELOW the u8 ceiling (no
        // collapse) while staying far above the 5-nit shadow.
        assert!(pu.pixels[0] > 200 && pu.pixels[0] < 255);
        assert!(pu.pixels[3] < pu.pixels[0] - 50);
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

#[cfg(all(test, feature = "png"))]
mod hlg_decode_tests {
    use super::*;

    #[test]
    fn hlg_to_nits_matches_bt2100_reference_points() {
        // Reference values computed from the same BT.2100 formulas the
        // conversion uses (inverse OETF + OOTF, γ=1.2 @ 1000 nit peak):
        // code 0.5 (u16 32768) achromatic → scene 1/12 ≈ 0.08333,
        // Y_s=0.08333, display = 999.995·0.08333^0.2·0.08333 + 0.005.
        let half = (0.5f32 * 65535.0) as u16;
        let img = rgb16_hlg_to_nits(&[half, half, half], 1, 1);
        let ys = 1.0f32 / 12.0;
        let expected = (1000.0 - 0.005) * ys.powf(0.2) * ys + 0.005;
        assert!(
            (img.rgb[0] - expected).abs() < 0.05,
            "HLG mid-code: got {} want {expected}",
            img.rgb[0]
        );
        // Peak white (code 1.0): scene 1.0 → display = peak.
        let img = rgb16_hlg_to_nits(&[65535, 65535, 65535], 1, 1);
        assert!(
            (img.rgb[0] - 1000.0).abs() < 0.5,
            "HLG peak: got {} want ~1000",
            img.rgb[0]
        );
        // Monotone in code value.
        let a = rgb16_hlg_to_nits(&[10000, 10000, 10000], 1, 1).rgb[0];
        let b = rgb16_hlg_to_nits(&[40000, 40000, 40000], 1, 1).rgb[0];
        assert!(a < b);
    }
}
