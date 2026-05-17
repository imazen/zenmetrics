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

#[cfg(feature = "avif")]
fn decode_avif(data: &[u8]) -> Result<Rgb8Image, Box<dyn std::error::Error>> {
    let pixels = zenavif::decode(data).map_err(|e| format!("zenavif: {e}"))?;
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

// ── PixelBuffer → RGB8 normalisation ─────────────────────────────────────
//
// Several zen decoders return a `zenpixels::PixelBuffer` whose underlying
// format depends on the source image. For the CLI we reduce that down to
// flat sRGB RGB8 — quality metrics expect tightly packed 3-byte pixels in
// sRGB-encoded space. We support the formats that actually come out of
// our decoders in practice:
//   * 8-bit packed: RGB8, RGBA8, RGBX8, BGRA8, BGRX8, Gray8, GrayA8
//   * 16-bit packed: RGB16, RGBA16, Gray16, GrayA16 — downconverted to
//     8-bit by `(v >> 8)` (i.e. divide-by-256 truncation; matches the
//     `image` crate's `into_rgb8()` semantics).
//   * F32 packed: RgbF32, RgbaF32, GrayF32, GrayAF32 — converted to 8-bit
//     by `(f.clamp(0.0, 1.0) * 255.0 + 0.5) as u8`. F32 buffers from zen
//     decoders are sRGB-encoded floats in [0, 1].
// f16, Oklab, and CMYK variants are not yet wired and surface as a clear
// error rather than being silently routed through the wrong transform.

#[cfg(any(feature = "png", feature = "jpeg", feature = "avif", feature = "jxl"))]
fn pixel_buffer_to_rgb8(
    buf: &zenpixels::PixelBuffer,
) -> Result<Rgb8Image, Box<dyn std::error::Error>> {
    pixel_slice_to_rgb8(&buf.as_slice())
}

#[cfg(any(feature = "png", feature = "jpeg", feature = "avif", feature = "jxl"))]
fn pixel_slice_to_rgb8(
    pixels: &zenpixels::PixelSlice<'_>,
) -> Result<Rgb8Image, Box<dyn std::error::Error>> {
    use zenpixels::PixelFormat;
    let width = pixels.width();
    let height = pixels.rows();
    let stride = pixels.stride();
    let format = pixels.descriptor().pixel_format();
    let bytes = pixels.as_strided_bytes();

    match format {
        PixelFormat::Rgb8 => Ok(Rgb8Image {
            pixels: copy_packed(
                bytes,
                width as usize,
                height as usize,
                stride,
                3,
                &[0, 1, 2],
            ),
            width,
            height,
        }),
        PixelFormat::Rgba8 => Ok(Rgb8Image {
            pixels: copy_packed(
                bytes,
                width as usize,
                height as usize,
                stride,
                4,
                &[0, 1, 2],
            ),
            width,
            height,
        }),
        PixelFormat::Rgbx8 => Ok(Rgb8Image {
            pixels: copy_packed(
                bytes,
                width as usize,
                height as usize,
                stride,
                4,
                &[0, 1, 2],
            ),
            width,
            height,
        }),
        PixelFormat::Bgra8 => Ok(Rgb8Image {
            pixels: copy_packed(
                bytes,
                width as usize,
                height as usize,
                stride,
                4,
                &[2, 1, 0],
            ),
            width,
            height,
        }),
        PixelFormat::Bgrx8 => Ok(Rgb8Image {
            pixels: copy_packed(
                bytes,
                width as usize,
                height as usize,
                stride,
                4,
                &[2, 1, 0],
            ),
            width,
            height,
        }),
        PixelFormat::Gray8 => Ok(Rgb8Image {
            pixels: gray_to_rgb8(bytes, width as usize, height as usize, stride, 1),
            width,
            height,
        }),
        PixelFormat::GrayA8 => Ok(Rgb8Image {
            pixels: gray_to_rgb8(bytes, width as usize, height as usize, stride, 2),
            width,
            height,
        }),

        // ── 16-bit packed formats ───────────────────────────────────────
        // Downconvert by truncating to the high byte (`v >> 8`). This
        // matches the image crate's into_rgb8() semantics and is what
        // every consumer of an 8-bit-only pipeline expects.
        PixelFormat::Rgb16 => Ok(Rgb8Image {
            pixels: copy_packed_u16(
                bytes,
                width as usize,
                height as usize,
                stride,
                3,
                &[0, 1, 2],
            ),
            width,
            height,
        }),
        PixelFormat::Rgba16 => Ok(Rgb8Image {
            pixels: copy_packed_u16(
                bytes,
                width as usize,
                height as usize,
                stride,
                4,
                &[0, 1, 2],
            ),
            width,
            height,
        }),
        PixelFormat::Gray16 => Ok(Rgb8Image {
            pixels: gray_u16_to_rgb8(bytes, width as usize, height as usize, stride, 1),
            width,
            height,
        }),
        PixelFormat::GrayA16 => Ok(Rgb8Image {
            pixels: gray_u16_to_rgb8(bytes, width as usize, height as usize, stride, 2),
            width,
            height,
        }),

        // ── F32 packed formats ──────────────────────────────────────────
        // Float buffers are sRGB-encoded values in [0, 1]; clamp + scale
        // by 255 with round-half-up.
        PixelFormat::RgbF32 => Ok(Rgb8Image {
            pixels: copy_packed_f32(
                bytes,
                width as usize,
                height as usize,
                stride,
                3,
                &[0, 1, 2],
            ),
            width,
            height,
        }),
        PixelFormat::RgbaF32 => Ok(Rgb8Image {
            pixels: copy_packed_f32(
                bytes,
                width as usize,
                height as usize,
                stride,
                4,
                &[0, 1, 2],
            ),
            width,
            height,
        }),
        PixelFormat::GrayF32 => Ok(Rgb8Image {
            pixels: gray_f32_to_rgb8(bytes, width as usize, height as usize, stride, 1),
            width,
            height,
        }),
        PixelFormat::GrayAF32 => Ok(Rgb8Image {
            pixels: gray_f32_to_rgb8(bytes, width as usize, height as usize, stride, 2),
            width,
            height,
        }),

        other => Err(format!(
            "decoder returned unsupported pixel format {other:?}; the CLI handles \
             RGB/RGBA/Gray in u8/u16/f32 and the BGRA/RGBX/BGRX 8-bit aliases, but \
             f16, Oklab, and CMYK source images are not yet wired through."
        )
        .into()),
    }
}

#[cfg(any(feature = "png", feature = "jpeg", feature = "avif", feature = "jxl"))]
fn copy_packed(
    bytes: &[u8],
    width: usize,
    height: usize,
    stride: usize,
    src_bpp: usize,
    src_indices: &[u8; 3],
) -> Vec<u8> {
    let mut out = vec![0u8; width * height * 3];
    for y in 0..height {
        let row = &bytes[y * stride..y * stride + width * src_bpp];
        let dst_row = &mut out[y * width * 3..(y + 1) * width * 3];
        for x in 0..width {
            let src = &row[x * src_bpp..x * src_bpp + src_bpp];
            let dst = &mut dst_row[x * 3..x * 3 + 3];
            dst[0] = src[src_indices[0] as usize];
            dst[1] = src[src_indices[1] as usize];
            dst[2] = src[src_indices[2] as usize];
        }
    }
    out
}

#[cfg(any(feature = "png", feature = "jpeg", feature = "avif", feature = "jxl"))]
fn gray_to_rgb8(
    bytes: &[u8],
    width: usize,
    height: usize,
    stride: usize,
    src_bpp: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; width * height * 3];
    for y in 0..height {
        let row = &bytes[y * stride..y * stride + width * src_bpp];
        let dst_row = &mut out[y * width * 3..(y + 1) * width * 3];
        for x in 0..width {
            let g = row[x * src_bpp];
            dst_row[x * 3] = g;
            dst_row[x * 3 + 1] = g;
            dst_row[x * 3 + 2] = g;
        }
    }
    out
}

// ── 16-bit / F32 → RGB8 helpers ──────────────────────────────────────────
//
// zenpixels lays multi-byte channels out in native byte order. The decoders
// we use (zenpng, zenjpeg, zenavif, zenjxl) all produce little-endian on the
// platforms we ship to, but we read via `u16::from_ne_bytes` / `f32::from_ne_bytes`
// to keep that platform-neutral within the running binary.

#[cfg(any(feature = "png", feature = "jpeg", feature = "avif", feature = "jxl"))]
fn copy_packed_u16(
    bytes: &[u8],
    width: usize,
    height: usize,
    stride: usize,
    src_channels: usize,
    src_indices: &[u8; 3],
) -> Vec<u8> {
    let src_bpp = src_channels * 2;
    let mut out = vec![0u8; width * height * 3];
    for y in 0..height {
        let row = &bytes[y * stride..y * stride + width * src_bpp];
        let dst_row = &mut out[y * width * 3..(y + 1) * width * 3];
        for x in 0..width {
            let src = &row[x * src_bpp..x * src_bpp + src_bpp];
            let dst = &mut dst_row[x * 3..x * 3 + 3];
            for c in 0..3 {
                let i = src_indices[c] as usize * 2;
                let v = u16::from_ne_bytes([src[i], src[i + 1]]);
                dst[c] = (v >> 8) as u8;
            }
        }
    }
    out
}

#[cfg(any(feature = "png", feature = "jpeg", feature = "avif", feature = "jxl"))]
fn gray_u16_to_rgb8(
    bytes: &[u8],
    width: usize,
    height: usize,
    stride: usize,
    src_channels: usize,
) -> Vec<u8> {
    let src_bpp = src_channels * 2;
    let mut out = vec![0u8; width * height * 3];
    for y in 0..height {
        let row = &bytes[y * stride..y * stride + width * src_bpp];
        let dst_row = &mut out[y * width * 3..(y + 1) * width * 3];
        for x in 0..width {
            let src = &row[x * src_bpp..x * src_bpp + 2];
            let g = (u16::from_ne_bytes([src[0], src[1]]) >> 8) as u8;
            dst_row[x * 3] = g;
            dst_row[x * 3 + 1] = g;
            dst_row[x * 3 + 2] = g;
        }
    }
    out
}

#[cfg(any(feature = "png", feature = "jpeg", feature = "avif", feature = "jxl"))]
#[inline]
fn f32_to_u8(f: f32) -> u8 {
    // NaN-safe: NaN comparisons always false → falls through to 0.
    let scaled = (f.clamp(0.0, 1.0) * 255.0) + 0.5;
    scaled as u8
}

#[cfg(any(feature = "png", feature = "jpeg", feature = "avif", feature = "jxl"))]
fn copy_packed_f32(
    bytes: &[u8],
    width: usize,
    height: usize,
    stride: usize,
    src_channels: usize,
    src_indices: &[u8; 3],
) -> Vec<u8> {
    let src_bpp = src_channels * 4;
    let mut out = vec![0u8; width * height * 3];
    for y in 0..height {
        let row = &bytes[y * stride..y * stride + width * src_bpp];
        let dst_row = &mut out[y * width * 3..(y + 1) * width * 3];
        for x in 0..width {
            let src = &row[x * src_bpp..x * src_bpp + src_bpp];
            let dst = &mut dst_row[x * 3..x * 3 + 3];
            for c in 0..3 {
                let i = src_indices[c] as usize * 4;
                let v = f32::from_ne_bytes([src[i], src[i + 1], src[i + 2], src[i + 3]]);
                dst[c] = f32_to_u8(v);
            }
        }
    }
    out
}

#[cfg(any(feature = "png", feature = "jpeg", feature = "avif", feature = "jxl"))]
fn gray_f32_to_rgb8(
    bytes: &[u8],
    width: usize,
    height: usize,
    stride: usize,
    src_channels: usize,
) -> Vec<u8> {
    let src_bpp = src_channels * 4;
    let mut out = vec![0u8; width * height * 3];
    for y in 0..height {
        let row = &bytes[y * stride..y * stride + width * src_bpp];
        let dst_row = &mut out[y * width * 3..(y + 1) * width * 3];
        for x in 0..width {
            let src = &row[x * src_bpp..x * src_bpp + 4];
            let g = f32_to_u8(f32::from_ne_bytes([src[0], src[1], src[2], src[3]]));
            dst_row[x * 3] = g;
            dst_row[x * 3 + 1] = g;
            dst_row[x * 3 + 2] = g;
        }
    }
    out
}

// zenpng wants `&dyn enough::Stop`. The crate exports `Unstoppable` but we
// only pull it in via zenpng's transitive dep, so we re-spell it here in
// the cheapest possible way.
#[cfg(feature = "png")]
fn enough_unstoppable() -> Box<dyn enough::Stop + Send + Sync> {
    Box::new(enough::Unstoppable)
}
