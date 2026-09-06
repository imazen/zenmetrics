#![forbid(unsafe_code)]

//! Format-detected decode of a path on disk into a flat 8-bit sRGB RGB
//! buffer. Each codec dependency is gated by a cargo feature; an
//! unsupported format returns an explanatory error rather than panicking.
//!
//! The output is always 3 bytes per pixel (`width * height * 3` bytes).
//! The metric layer assumes sRGB-encoded data — codec-side colour
//! management (ICC, CICP) is intentionally not applied here. Image-quality
//! metrics by convention compare the encoded sRGB pixel values directly.

use std::fs;
use std::path::Path;

/// Owned decoded image in flat sRGB RGB8 layout (`width * height * 3` bytes).
pub struct Rgb8Image {
    // `pixels` is consumed by metric backends (cpu-metrics + every
    // gpu-* feature). When the CLI is built with no metrics enabled
    // the field looks unused — annotate so clippy doesn't fail CI.
    #[allow(dead_code)]
    pub pixels: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// Decode `path` into 8-bit sRGB RGB. Format is sniffed from the file's
/// magic bytes first, with extension as a fall-back tiebreaker.
pub fn decode_image_to_rgb8(path: &Path) -> Result<Rgb8Image, Box<dyn std::error::Error>> {
    let data = fs::read(path)?;
    let format = sniff_format(&data, path);
    decode_bytes_to_rgb8(&data, format)
}

/// Decode RGB8 straight from in-memory bytes — no temp file. `name_hint` (the object
/// key / filename, may be empty) supplies the extension fallback for format sniffing
/// when the magic bytes are ambiguous. Used by the in-process jobexec fetch path.
#[allow(dead_code)] // only the jobexec/sweep executor calls this
pub fn decode_rgb8_from_bytes(
    data: &[u8],
    name_hint: &str,
) -> Result<Rgb8Image, Box<dyn std::error::Error>> {
    let format = sniff_format(data, Path::new(name_hint));
    decode_bytes_to_rgb8(data, format)
}

/// File-format identifier. Variants present here are independent of which
/// crate features are enabled — the dispatch layer rejects formats whose
/// decoder feature was not compiled in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    Png,
    Jpeg,
    Webp,
    Avif,
    Jxl,
    Gif,
    Tiff,
    Bmp,
}

fn sniff_format(data: &[u8], path: &Path) -> Option<ImageFormat> {
    if data.starts_with(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]) {
        return Some(ImageFormat::Png);
    }
    if data.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return Some(ImageFormat::Jpeg);
    }
    if data.len() >= 12 && &data[0..4] == b"RIFF" && &data[8..12] == b"WEBP" {
        return Some(ImageFormat::Webp);
    }
    // AVIF: ISOBMFF ftyp box with brand "avif" / "avis" / "heic"-as-AV1.
    if data.len() >= 12 && &data[4..8] == b"ftyp" {
        let brand = &data[8..12];
        if brand == b"avif" || brand == b"avis" || brand == b"mif1" {
            return Some(ImageFormat::Avif);
        }
    }
    // JPEG XL: bare codestream (FF 0A) or container (00 00 00 0C 4A 58 4C 20 0D 0A 87 0A).
    if data.starts_with(&[0xFF, 0x0A]) {
        return Some(ImageFormat::Jxl);
    }
    if data.len() >= 12
        && data[0..4] == [0x00, 0x00, 0x00, 0x0C]
        && &data[4..12] == b"JXL \r\n\x87\n"
    {
        return Some(ImageFormat::Jxl);
    }
    // GIF: "GIF87a" / "GIF89a".
    if data.starts_with(b"GIF87a") || data.starts_with(b"GIF89a") {
        return Some(ImageFormat::Gif);
    }
    // TIFF: little-endian "II*\0" or big-endian "MM\0*" (classic + BigTIFF
    // both start with these byte-order marks; the version word differs).
    if data.len() >= 4
        && (data[0..4] == [0x49, 0x49, 0x2A, 0x00] || data[0..4] == [0x4D, 0x4D, 0x00, 0x2A])
    {
        return Some(ImageFormat::Tiff);
    }
    // BMP: "BM" file-header signature (Windows/OS2 bitmap).
    if data.starts_with(b"BM") {
        return Some(ImageFormat::Bmp);
    }

    // Fall back to extension.
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
    {
        Some(ext) => match ext.as_str() {
            "png" => Some(ImageFormat::Png),
            "jpg" | "jpeg" => Some(ImageFormat::Jpeg),
            "webp" => Some(ImageFormat::Webp),
            "avif" | "avis" | "heic" | "heif" => Some(ImageFormat::Avif),
            "jxl" => Some(ImageFormat::Jxl),
            "gif" => Some(ImageFormat::Gif),
            "tif" | "tiff" => Some(ImageFormat::Tiff),
            "bmp" => Some(ImageFormat::Bmp),
            _ => None,
        },
        None => None,
    }
}

fn decode_bytes_to_rgb8(
    data: &[u8],
    format: Option<ImageFormat>,
) -> Result<Rgb8Image, Box<dyn std::error::Error>> {
    let format = format.ok_or("could not detect image format from magic bytes or extension")?;
    match format {
        ImageFormat::Png => decode_png(data),
        ImageFormat::Jpeg => decode_jpeg(data),
        ImageFormat::Webp => decode_webp(data),
        ImageFormat::Avif => decode_avif(data),
        ImageFormat::Jxl => decode_jxl(data),
        ImageFormat::Gif => decode_gif(data),
        ImageFormat::Tiff => decode_tiff(data),
        ImageFormat::Bmp => decode_bmp(data),
    }
}

#[cfg(feature = "png")]
fn decode_png(data: &[u8]) -> Result<Rgb8Image, Box<dyn std::error::Error>> {
    use zenpng::{PngDecodeConfig, decode};
    // zenpng decode returns a PixelBuffer in whatever native format the PNG
    // happens to be in (RGB8, RGBA8, gray, 16-bit, ...). We funnel through
    // the unified pixel-buffer-to-RGB8 helper so the metric sees the same
    // layout regardless of source.
    let cancel = enough_unstoppable();
    let output = decode(data, &PngDecodeConfig::default(), &*cancel)?;
    // HDR tripwire: a PNG whose cICP says PQ (16) or HLG (18) carries
    // absolute-luminance HDR code values. The RGB8 funnel below would
    // silently quantise 16-bit PQ to "8-bit sRGB" — scores computed on
    // that are garbage with no error anywhere (the imazen/zenmetrics#25
    // failure class). Error loudly instead; HDR-aware callers route
    // through `hdr::decode_to_nits` (`--hdr` on score / score-pairs /
    // sweep).
    if let Some(cicp) = &output.info.cicp
        && matches!(cicp.transfer_characteristics, 16 | 18)
    {
        return Err(format!(
            "PNG signals an HDR transfer via cICP (transfer={}, {}): refusing \
             to crush it through the 8-bit SDR decode path — score it with \
             --hdr instead",
            cicp.transfer_characteristics,
            if cicp.transfer_characteristics == 16 {
                "PQ"
            } else {
                "HLG"
            },
        )
        .into());
    }
    pixel_buffer_to_rgb8(&output.pixels)
}

#[cfg(not(feature = "png"))]
fn decode_png(_data: &[u8]) -> Result<Rgb8Image, Box<dyn std::error::Error>> {
    Err("PNG decoding is disabled (compile with `--features png`)".into())
}

#[cfg(feature = "jpeg")]
fn decode_jpeg(data: &[u8]) -> Result<Rgb8Image, Box<dyn std::error::Error>> {
    use zenjpeg::JpegDecoderConfig;
    // zenjpeg's decode() returns a DecodeOutput; the underlying PixelBuffer
    // is in one of zenjpeg's native output formats (RGB8 or RGBA8 are the
    // common cases for typical JPEGs).
    let output = JpegDecoderConfig::new().decode(data)?;
    pixel_slice_to_rgb8(&output.pixels())
}

#[cfg(not(feature = "jpeg"))]
fn decode_jpeg(_data: &[u8]) -> Result<Rgb8Image, Box<dyn std::error::Error>> {
    Err("JPEG decoding is disabled (compile with `--features jpeg`)".into())
}

#[cfg(feature = "webp")]
fn decode_webp(data: &[u8]) -> Result<Rgb8Image, Box<dyn std::error::Error>> {
    // zenwebp exposes a tight one-shot RGB8 decode that handles both lossy
    // and lossless WebP and returns flat (Vec<u8>, w, h). No PixelBuffer
    // conversion needed.
    let (pixels, width, height) = zenwebp::decoder::decode_rgb(data)?;
    Ok(Rgb8Image {
        pixels,
        width,
        height,
    })
}

#[cfg(not(feature = "webp"))]
fn decode_webp(_data: &[u8]) -> Result<Rgb8Image, Box<dyn std::error::Error>> {
    Err("WebP decoding is disabled (compile with `--features webp`)".into())
}

/// Name the ITU-T H.273 transfer characteristics that carry absolute-luminance
/// HDR, or `None` for every display-referred (SDR) code.
///
/// The same policy the PNG cICP tripwire above applies, so one rule covers both
/// formats: **only** PQ (16) and HLG (18) are refused. The 8-vs-10-bit SDR AVIF
/// track encodes 10-bit files that must keep decoding through this path, and
/// BT.2020's SDR transfers (14, 15) sit adjacent to PQ/HLG in the CICP table —
/// an over-broad guard would break both. Narrowing a 10-bit *SDR* AVIF to 8 bits
/// is this module's documented contract; relabelling an HDR transfer as sRGB is
/// not.
#[cfg(feature = "avif")]
fn hdr_transfer_name(transfer_characteristics: u8) -> Option<&'static str> {
    match transfer_characteristics {
        16 => Some("PQ"),
        18 => Some("HLG"),
        _ => None,
    }
}

#[cfg(feature = "avif")]
fn decode_avif(data: &[u8]) -> Result<Rgb8Image, Box<dyn std::error::Error>> {
    // Single-threaded decode (threads(1)): rav1d-safe's default multi-threaded
    // frame decode (n_threads=0=auto) races on the frame buffer under the sweep's
    // rayon parallelism — DisjointMut overlap panic / deadlock (imazen/rav1d-safe#15).
    // One thread per decode removes the internal race; cross-decode parallelism is
    // safe because each decode owns its decoder/frame. ~no throughput loss in the
    // sweep (the rayon walk already parallelizes across cells).
    let cfg = zenavif::DecoderConfig::new().threads(1);
    // `ManagedAvifDecoder::decode_full` rather than `zenavif::decode_with`,
    // because the HDR tripwire below needs the `ImageInfo` that only
    // `decode_full` hands back. It is the same decoder `decode_with` selects
    // for this build (the `AvifDecoder` sibling is `unsafe-asm`-gated and we
    // never enable it), and the same one `sweep::hdr::decode_avif_to_nits`
    // already drives — so the decoded pixels are unchanged.
    //
    // It is also the single entry serving BOTH the grid (tiled) and non-grid
    // shapes, so the tripwire covers both from one site: `decode_full` branches
    // on `grid_config()` internally and returns an `ImageInfo` either way.
    let mut decoder =
        zenavif::ManagedAvifDecoder::new(data, &cfg).map_err(|e| format!("zenavif: {e}"))?;
    let (pixels, info) = decoder
        .decode_full(&enough::Unstoppable)
        .map_err(|e| format!("zenavif: {e}"))?;
    // HDR tripwire — the AVIF twin of the PNG cICP refusal above. An AVIF whose
    // `colr`/`nclx` box (or, absent one, whose AV1 sequence header) signals PQ or
    // HLG carries absolute-luminance code values. The RGB8 funnel below would
    // narrow them to 8 bits and relabel them sRGB, producing scores that look
    // plausible and mean nothing, with no error anywhere — the
    // imazen/zenmetrics#25 failure class.
    //
    // The second-line zenpixels `HdrSourceRequiresPeak` guard cannot catch this:
    // the buffered decode never calls `descriptor_with_cicp` (only the row-sink
    // paths do), so the buffer reaches `RowConverter` tagged
    // `TransferFunction::Unknown` and the conversion is a byte passthrough.
    // Measured pre-fix: a PQ-signalled copy of `tests/fixtures/ref_64.avif`
    // scored bit-identically to the sRGB original, on both the plain and the
    // grid-tiled shapes.
    //
    // HDR-aware callers route through `hdr::decode_to_nits` (`--hdr` on score /
    // score-pairs / sweep), which preserves all 10 bits into f32 cd/m².
    if let Some(name) = hdr_transfer_name(info.transfer_characteristics.0) {
        return Err(format!(
            "AVIF signals an HDR transfer via CICP (transfer={}, {}) at {}-bit: \
             refusing to crush it through the 8-bit SDR decode path — score it \
             with --hdr instead",
            info.transfer_characteristics.0, name, info.bit_depth,
        )
        .into());
    }
    pixel_buffer_to_rgb8(&pixels)
}

#[cfg(not(feature = "avif"))]
fn decode_avif(_data: &[u8]) -> Result<Rgb8Image, Box<dyn std::error::Error>> {
    Err("AVIF decoding is disabled (compile with `--features avif`)".into())
}

#[cfg(feature = "jxl")]
fn decode_jxl(data: &[u8]) -> Result<Rgb8Image, Box<dyn std::error::Error>> {
    use zenjxl::decode;
    // Pass an empty preferred-format list so zenjxl returns its native
    // pixel format; we normalise downstream.
    let output = decode(data, None, &[]).map_err(|e| format!("zenjxl: {e}"))?;
    pixel_buffer_to_rgb8(&output.pixels)
}

#[cfg(not(feature = "jxl"))]
fn decode_jxl(_data: &[u8]) -> Result<Rgb8Image, Box<dyn std::error::Error>> {
    Err("JPEG XL decoding is disabled (compile with `--features jxl`)".into())
}

#[cfg(feature = "gif")]
fn decode_gif(data: &[u8]) -> Result<Rgb8Image, Box<dyn std::error::Error>> {
    // GIF stills carry one frame; the sweep encodes single-frame stills, so
    // we score the first composed frame. `decode_gif` returns RGBA (disposal
    // + transparency already applied); drop alpha for the SDR RGB8 metric.
    let (_meta, frames, _stats) =
        zengif::decode_gif(data, zengif::Limits::none(), &enough::Unstoppable)
            .map_err(|e| format!("zengif: {e}"))?;
    let frame = frames
        .into_iter()
        .next()
        .ok_or("zengif: decoded GIF has no frames")?;
    let mut pixels = Vec::with_capacity(frame.pixels.len() * 3);
    for px in &frame.pixels {
        pixels.push(px.r);
        pixels.push(px.g);
        pixels.push(px.b);
    }
    Ok(Rgb8Image {
        pixels,
        width: u32::from(frame.width),
        height: u32::from(frame.height),
    })
}

#[cfg(not(feature = "gif"))]
fn decode_gif(_data: &[u8]) -> Result<Rgb8Image, Box<dyn std::error::Error>> {
    Err("GIF decoding is disabled (compile with `--features gif`)".into())
}

#[cfg(feature = "tiff")]
fn decode_tiff(data: &[u8]) -> Result<Rgb8Image, Box<dyn std::error::Error>> {
    // zentiff::decode returns a PixelBuffer in the TIFF's native layout
    // (Gray/RGB/RGBA × u8/u16/f32); funnel through the unified RGB8
    // normaliser like the other PixelBuffer-returning decoders.
    let output = zentiff::decode(
        data,
        &zentiff::TiffDecodeConfig::default(),
        &enough::Unstoppable,
    )
    .map_err(|e| format!("zentiff: {e}"))?;
    pixel_buffer_to_rgb8(&output.pixels)
}

#[cfg(not(feature = "tiff"))]
fn decode_tiff(_data: &[u8]) -> Result<Rgb8Image, Box<dyn std::error::Error>> {
    Err("TIFF decoding is disabled (compile with `--features tiff`)".into())
}

#[cfg(feature = "bmp")]
fn decode_bmp(data: &[u8]) -> Result<Rgb8Image, Box<dyn std::error::Error>> {
    // `zenbitmaps::decode_bmp` returns a `DecodeOutput` in whatever native
    // `PixelLayout` the file's bit depth/bitfields decode to (1/2/4/8/16/24/32-bit,
    // RLE and BITFIELDS all normalise to one of the palette-free layouts below —
    // zenbitmaps' own job, not this module's). Funnel every layout to flat RGB8
    // the same way the other arms funnel their native `PixelBuffer`.
    let output = zenbitmaps::decode_bmp(data, enough::Unstoppable)
        .map_err(|e| format!("zenbitmaps: {e}"))?;
    bmp_layout_to_rgb8(output.pixels(), output.width, output.height, output.layout)
}

#[cfg(feature = "bmp")]
fn bmp_layout_to_rgb8(
    pixels: &[u8],
    width: u32,
    height: u32,
    layout: zenbitmaps::PixelLayout,
) -> Result<Rgb8Image, Box<dyn std::error::Error>> {
    use zenbitmaps::PixelLayout;
    let n = (width as usize) * (height as usize);
    let mut out = vec![0u8; n * 3];
    macro_rules! fill {
        ($chunk_len:expr, $body:expr) => {{
            let body: fn(&[u8], &mut [u8]) = $body;
            for (src, dst) in pixels.chunks_exact($chunk_len).zip(out.chunks_exact_mut(3)) {
                body(src, dst);
            }
        }};
    }
    match layout {
        PixelLayout::Rgb8 => out.copy_from_slice(&pixels[..n * 3]),
        PixelLayout::Gray8 => {
            fill!(1, |s, d: &mut [u8]| {
                d[0] = s[0];
                d[1] = s[0];
                d[2] = s[0];
            });
        }
        PixelLayout::Bgr8 => {
            fill!(3, |s, d: &mut [u8]| {
                d[0] = s[2];
                d[1] = s[1];
                d[2] = s[0];
            });
        }
        PixelLayout::Rgba8 => {
            fill!(4, |s, d: &mut [u8]| {
                d[0] = s[0];
                d[1] = s[1];
                d[2] = s[2];
            });
        }
        PixelLayout::Bgra8 | PixelLayout::Bgrx8 => {
            fill!(4, |s, d: &mut [u8]| {
                d[0] = s[2];
                d[1] = s[1];
                d[2] = s[0];
            });
        }
        // Not a reachable BMP output today (the format's max depth is 32bpp
        // integer per zenbitmaps' README — no 16-bit-per-channel BMP variant),
        // but `PixelLayout` is `#[non_exhaustive]` and shared across every
        // format the crate decodes, so this stays a real arm rather than an
        // `unreachable!()`: narrow native-endian 16-bit to the top byte.
        PixelLayout::Gray16 => {
            fill!(2, |s, d: &mut [u8]| {
                let v8 = (u16::from_ne_bytes([s[0], s[1]]) >> 8) as u8;
                d[0] = v8;
                d[1] = v8;
                d[2] = v8;
            });
        }
        PixelLayout::Rgba16 => {
            fill!(8, |s, d: &mut [u8]| {
                d[0] = (u16::from_ne_bytes([s[0], s[1]]) >> 8) as u8;
                d[1] = (u16::from_ne_bytes([s[2], s[3]]) >> 8) as u8;
                d[2] = (u16::from_ne_bytes([s[4], s[5]]) >> 8) as u8;
            });
        }
        PixelLayout::GrayF32 => {
            fill!(4, |s, d: &mut [u8]| {
                let v = f32::from_ne_bytes([s[0], s[1], s[2], s[3]]).clamp(0.0, 1.0);
                let v8 = (v * 255.0 + 0.5) as u8;
                d[0] = v8;
                d[1] = v8;
                d[2] = v8;
            });
        }
        PixelLayout::RgbF32 => {
            fill!(12, |s, d: &mut [u8]| {
                for c in 0..3 {
                    let v =
                        f32::from_ne_bytes([s[c * 4], s[c * 4 + 1], s[c * 4 + 2], s[c * 4 + 3]])
                            .clamp(0.0, 1.0);
                    d[c] = (v * 255.0 + 0.5) as u8;
                }
            });
        }
        other => {
            return Err(format!(
                "zenbitmaps: BMP decoded to pixel layout {other:?}, which this module's \
                 RGB8 funnel has no arm for (zenbitmaps' PixelLayout is #[non_exhaustive] \
                 -- add the arm here, never silently reinterpret the bytes)"
            )
            .into());
        }
    }
    Ok(Rgb8Image {
        pixels: out,
        width,
        height,
    })
}

#[cfg(not(feature = "bmp"))]
fn decode_bmp(_data: &[u8]) -> Result<Rgb8Image, Box<dyn std::error::Error>> {
    Err("BMP decoding is disabled (compile with `--features bmp`)".into())
}

// ── PixelBuffer → RGB8 normalisation ─────────────────────────────────────
//
// Several zen decoders return a `zenpixels::PixelBuffer` whose underlying
// pixel layout depends on the source image (RGB/RGBA/Gray × u8/u16/f32,
// straight or premultiplied alpha, sRGB or linear). We collapse all of
// those down to flat sRGB RGB8 — quality metrics expect tightly packed
// 3-byte pixels in sRGB-encoded space.
//
// The conversion is delegated to `zenpixels_convert::RowConverter`, which
// already knows how to do every supported source → RGB8_SRGB pair via
// kernels covering u16 → u8 (`v >> 8`), f32 → u8 (clamp/quantise),
// gray → RGB broadcast, channel-reorder (BGRA/RGBX/BGRX), and alpha drop.
// Earlier revisions of this file shipped per-format helpers that recreated
// (more naively) what the converter already does; those have been removed
// to keep all pixel-format logic in the canonical crate.

#[cfg(any(
    feature = "png",
    feature = "jpeg",
    feature = "avif",
    feature = "jxl",
    feature = "tiff"
))]
fn pixel_buffer_to_rgb8(
    buf: &zenpixels::PixelBuffer,
) -> Result<Rgb8Image, Box<dyn std::error::Error>> {
    pixel_slice_to_rgb8(&buf.as_slice())
}

#[cfg(any(
    feature = "png",
    feature = "jpeg",
    feature = "avif",
    feature = "jxl",
    feature = "tiff"
))]
fn pixel_slice_to_rgb8(
    pixels: &zenpixels::PixelSlice<'_>,
) -> Result<Rgb8Image, Box<dyn std::error::Error>> {
    use zenpixels::PixelDescriptor;
    use zenpixels_convert::converter::RowConverter;

    let width = pixels.width();
    let height = pixels.rows();
    let src_stride = pixels.stride();
    let src_desc = pixels.descriptor();
    let src_bytes = pixels.as_strided_bytes();

    let dst_desc = PixelDescriptor::RGB8_SRGB;
    let dst_stride = width as usize * 3;
    let mut dst = vec![0u8; dst_stride * height as usize];

    let mut conv = RowConverter::new(src_desc, dst_desc)
        .map_err(|e| format!("decode: cannot plan {src_desc:?} → RGB8_SRGB: {e}"))?;
    conv.convert_rows(src_bytes, src_stride, &mut dst, dst_stride, width, height)
        .map_err(|e| format!("decode: row conversion failed: {e}"))?;

    Ok(Rgb8Image {
        pixels: dst,
        width,
        height,
    })
}

// zenpng wants `&dyn enough::Stop`. The crate exports `Unstoppable` but we
// only pull it in via zenpng's transitive dep, so we re-spell it here in
// the cheapest possible way.
#[cfg(feature = "png")]
fn enough_unstoppable() -> Box<dyn enough::Stop + Send + Sync> {
    Box::new(enough::Unstoppable)
}

#[cfg(all(test, feature = "avif"))]
mod tests {
    use super::hdr_transfer_name;

    /// The AVIF HDR tripwire's policy, pinned over the whole ITU-T H.273
    /// transfer code space. Exhaustive rather than sampled because the guard
    /// sits in front of every SDR AVIF this repo scores: an over-broad rule
    /// would refuse live sweep cells, and a narrow one would let the
    /// zenmetrics#25 corruption back through. Both directions are asserted.
    ///
    /// This also covers the tiled-vs-plain question by construction: there is
    /// exactly ONE policy function and exactly one call site, reached from
    /// `ManagedAvifDecoder::decode_full`, which serves the grid and non-grid
    /// shapes alike. Tiling cannot select a different answer.
    #[test]
    fn only_pq_and_hlg_are_refused() {
        for tc in 0u8..=255 {
            let got = hdr_transfer_name(tc);
            let want = match tc {
                16 => Some("PQ"),
                18 => Some("HLG"),
                _ => None,
            };
            assert_eq!(got, want, "transfer code {tc} classified wrongly");
        }
    }

    /// The two SDR codes most at risk from an over-broad guard: BT.2020's
    /// 10-bit and 12-bit transfers, which neighbour PQ/HLG in the CICP table
    /// and appear on wide-gamut SDR content.
    #[test]
    fn bt2020_sdr_transfers_are_not_hdr() {
        assert_eq!(hdr_transfer_name(14), None);
        assert_eq!(hdr_transfer_name(15), None);
    }
}

/// The REV2 RECALCULATION LAN wave's blocker (docs/PLAN_REV2_RECALC_2026-09-06.md
/// §7.2): the stored LIVE 372-eval table was built from `.bmp`, and this CLI
/// had no BMP arm at all. These tests gate the new arm with REAL BMP bytes —
/// produced by `zenbitmaps`' own encoder (never hand-rolled binary literals,
/// per the no-duplicate-implementations rule) — round-tripped through the
/// SAME public entry point (`decode_rgb8_from_bytes`) a fleet Feature job
/// calls, so a magic-byte sniff regression or a channel-order bug in the new
/// funnel fails loud here rather than as a silent wrong-pixel score.
///
/// This does NOT replace the cross-repo check: `scripts/jobsys/
/// rev2_bitexact_gate.py --pairs live_r2_pairs.tsv` (bmp paths) is the
/// authority that this arm's pixels lead to bit-identical zensim features
/// against the stored postC root — see that gate's run in the commit this
/// module change ships with.
#[cfg(all(test, feature = "bmp"))]
mod bmp_tests {
    use super::*;
    use zenbitmaps::PixelLayout;

    /// Distinct, non-grayscale per-pixel values so a channel swap (R<->B) or
    /// a row-order bug cannot hide behind symmetric test data. 3x2, not a
    /// power of two, so a stride miscalculation would show up as garbage
    /// rather than an accidental pass.
    const W: u32 = 3;
    const H: u32 = 2;
    const RGB8_SRC: [u8; (W * H * 3) as usize] = [
        10, 20, 30, // (0,0)
        200, 150, 50, // (1,0)
        1, 254, 128, // (2,0)
        90, 5, 200, // (0,1)
        255, 0, 0, // (1,1)
        0, 255, 255, // (2,1)
    ];

    #[test]
    fn detects_bm_magic_and_round_trips_24bit_rgb() {
        let bmp_bytes =
            zenbitmaps::encode_bmp(&RGB8_SRC, W, H, PixelLayout::Rgb8, enough::Unstoppable)
                .expect("zenbitmaps encode_bmp");
        assert!(bmp_bytes.starts_with(b"BM"), "not a BMP file header");
        assert_eq!(
            sniff_format(&bmp_bytes, Path::new("ignored.bmp")),
            Some(ImageFormat::Bmp)
        );

        let decoded =
            decode_rgb8_from_bytes(&bmp_bytes, "fixture.bmp").expect("decode_rgb8_from_bytes");
        assert_eq!((decoded.width, decoded.height), (W, H));
        assert_eq!(
            decoded.pixels, RGB8_SRC,
            "24-bit BMP round-trip must be lossless RGB8"
        );
    }

    #[test]
    fn magic_bytes_win_over_a_mismatched_extension() {
        // Format detection is magic-byte first, extension only as a
        // tiebreaker (matching every other arm in `sniff_format`) — a BMP
        // saved with the wrong extension must still decode.
        let bmp_bytes =
            zenbitmaps::encode_bmp(&RGB8_SRC, W, H, PixelLayout::Rgb8, enough::Unstoppable)
                .expect("zenbitmaps encode_bmp");
        let decoded =
            decode_rgb8_from_bytes(&bmp_bytes, "actually_a.png").expect("decode_rgb8_from_bytes");
        assert_eq!(decoded.pixels, RGB8_SRC);
    }

    #[test]
    fn thirty_two_bit_bgra_drops_alpha_consistently() {
        // encode_bmp_rgba's `layout` parameter names the SOURCE buffer's
        // layout, not zenbitmaps' internal on-disk order; feed it via Rgba8
        // and confirm the RGB channels survive a partial-alpha round trip
        // (alpha itself is intentionally dropped, matching decode_gif's
        // alpha-drop policy elsewhere in this file).
        let mut rgba = Vec::with_capacity(RGB8_SRC.len() / 3 * 4);
        for (i, px) in RGB8_SRC.as_chunks::<3>().0.iter().enumerate() {
            rgba.extend_from_slice(px);
            rgba.push((i as u8) * 40); // varying, non-255 alpha
        }
        let bmp_bytes =
            zenbitmaps::encode_bmp_rgba(&rgba, W, H, PixelLayout::Rgba8, enough::Unstoppable)
                .expect("zenbitmaps encode_bmp_rgba");
        assert_eq!(
            sniff_format(&bmp_bytes, Path::new("x.bmp")),
            Some(ImageFormat::Bmp)
        );
        let decoded =
            decode_rgb8_from_bytes(&bmp_bytes, "fixture32.bmp").expect("decode_rgb8_from_bytes");
        assert_eq!((decoded.width, decoded.height), (W, H));
        assert_eq!(
            decoded.pixels, RGB8_SRC,
            "RGB channels must survive alpha drop"
        );
    }

    #[test]
    fn every_named_pixel_layout_maps_to_rgb8_without_panicking() {
        // `PixelLayout` is `#[non_exhaustive]`; this pins that every variant
        // the crate names TODAY has a real (non-wildcard-error) arm in
        // `bmp_layout_to_rgb8`, so a future zenbitmaps upgrade that adds a
        // new variant is the only way to hit the "add the arm here" error.
        let cases: &[(PixelLayout, usize)] = &[
            (PixelLayout::Gray8, 1),
            (PixelLayout::Gray16, 2),
            (PixelLayout::Rgb8, 3),
            (PixelLayout::Rgba8, 4),
            (PixelLayout::Bgr8, 3),
            (PixelLayout::Bgra8, 4),
            (PixelLayout::Bgrx8, 4),
            (PixelLayout::GrayF32, 4),
            (PixelLayout::RgbF32, 12),
            (PixelLayout::Rgba16, 8),
        ];
        for &(layout, bpp) in cases {
            let px = vec![0u8; bpp * (W * H) as usize];
            let out = bmp_layout_to_rgb8(&px, W, H, layout)
                .unwrap_or_else(|e| panic!("{layout:?} should have a real arm: {e}"));
            assert_eq!(out.pixels.len(), (W * H * 3) as usize);
        }
    }
}
