//! Stage-1 decode: HDR source → absolute-luminance EXR.
//!
//! Reconstructs HDR from gain-map sources via existing crates:
//!   - **Ultra HDR JPEG** (Android / Adobe): `ultrahdr-rs` one-call decode.
//!   - **HDR gain-map HEIC** (Apple / Samsung): `heic` decodes base P3 + gain
//!     map; `zencodec::gainmap::parse_iso21496` parses the ISO 21496-1 XMP
//!     (which IS `ultrahdr_core::GainMapMetadata` — a type alias), and
//!     `ultrahdr_core::apply_gainmap` reconstructs HDR.
//!
//! Output is linear RGBA f32 (1.0 = SDR white = 203 cd/m²); we scale ×203 to
//! absolute luminance and write an EXR (the UPIQ convention) so the same
//! `compute_pu_linear_planar` / pycvvdp scoring applies.
//!
//! Usage: hdr-decode [SRC_DIR] [OUT_DIR]   (walks SRC_DIR one level deep)

use std::path::{Path, PathBuf};

use enough::Unstoppable;
use ultrahdr_core::gainmap::apply::apply_gainmap;
// Use ultrahdr-core's re-exports (incl. its zencodec parser) so the
// `GainMapParams` type matches `apply_gainmap` — depending on zencodec directly
// pulls a second instance and the types won't unify.
use ultrahdr_core::{
    ColorPrimaries, GainMap, HdrOutputFormat, Iso21496Format, PixelBuffer, PixelFormat,
    TransferFunction, parse_iso21496_fmt, pixel_buffer_from_vec,
};
use ultrahdr_rs::Decoder;

/// `ultrahdr-core` LinearFloat output: 1.0 = SDR white = 203 cd/m² (BT.2408).
const SDR_WHITE_NITS: f32 = 203.0;
/// HDR display headroom to reconstruct (4× ≈ 812 nits peak).
const DISPLAY_BOOST: f32 = 4.0;

/// Ultra HDR / gain-map JPEG → HDR PixelBuffer (linear RGBA f32, 1.0 = SDR white).
fn decode_jpeg(bytes: &[u8]) -> Result<PixelBuffer, String> {
    let dec = Decoder::new(bytes).map_err(|e| format!("parse: {e:?}"))?;
    if !dec.is_ultrahdr() {
        return Err("no gain map (SDR JPEG)".into());
    }
    dec.decode_hdr(DISPLAY_BOOST)
        .map_err(|e| format!("decode_hdr: {e:?}"))
}

/// HDR gain-map HEIC → HDR PixelBuffer. heic decodes the base (P3) + gain map;
/// zencodec parses the ISO 21496-1 XMP; ultrahdr-core applies it.
fn decode_heic(bytes: &[u8]) -> Result<PixelBuffer, String> {
    let cfg = heic::DecoderConfig::new();
    if !cfg
        .has_gain_map(bytes)
        .map_err(|e| format!("probe: {e:?}"))?
    {
        return Err("no gain map (SDR / PQ HEIC)".into());
    }
    let base = cfg
        .decode(bytes, heic::PixelLayout::Rgb8)
        .map_err(|e| format!("base decode: {e:?}"))?;
    let gm = cfg
        .decode_gain_map(bytes)
        .map_err(|e| format!("gainmap decode: {e:?}"))?;
    // Prefer the ISO 21496-1 binary from the `tmap` item (AVIF tmap byte
    // layout: version byte + ISO payload). The legacy Apple `aux:hdrgainmap`
    // XMP-RDF is a proprietary fallback our ISO parser can't read.
    let params = if let Some(iso) = &gm.iso21496 {
        parse_iso21496_fmt(iso, Iso21496Format::AvifTmap)
            .map_err(|e| format!("tmap ISO 21496-1 parse ({} bytes): {e:?}", iso.len()))?
    } else {
        return Err(format!(
            "no ISO 21496-1 tmap metadata (origin {:?}; only legacy Apple aux XMP {} bytes)",
            gm.origin,
            gm.xmp.as_ref().map(|x| x.len()).unwrap_or(0)
        ));
    };
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
    apply_gainmap(
        &sdr,
        &gainmap,
        &params,
        DISPLAY_BOOST,
        HdrOutputFormat::LinearFloat,
        Unstoppable,
    )
    .map_err(|e| format!("apply: {e:?}"))
}

/// PixelBuffer (RGBA f32, 1.0 = SDR white) → (w, h, tight RGB in nits, mean luma, max luma).
fn to_rgb_nits(hdr: &PixelBuffer) -> (usize, usize, Vec<f32>, f64, f32) {
    let w = hdr.width() as usize;
    let h = hdr.height() as usize;
    let raw = hdr.as_slice().as_strided_bytes();
    let f: &[f32] = bytemuck::cast_slice(raw);
    let row_px = f.len() / 4 / h;
    let mut rgb = vec![0.0f32; w * h * 3];
    let (mut max_y, mut sum_y) = (0.0f32, 0.0f64);
    for y in 0..h {
        for x in 0..w {
            let si = (y * row_px + x) * 4;
            let di = (y * w + x) * 3;
            let (r, g, b) = (
                f[si] * SDR_WHITE_NITS,
                f[si + 1] * SDR_WHITE_NITS,
                f[si + 2] * SDR_WHITE_NITS,
            );
            rgb[di] = r;
            rgb[di + 1] = g;
            rgb[di + 2] = b;
            let lum = 0.2627 * r + 0.6780 * g + 0.0593 * b;
            max_y = max_y.max(lum);
            sum_y += lum as f64;
        }
    }
    (w, h, rgb, sum_y / (w * h) as f64, max_y)
}

fn collect_files(src_dir: &str) -> Vec<PathBuf> {
    let mut files = vec![];
    for e in std::fs::read_dir(src_dir).into_iter().flatten().flatten() {
        let p = e.path();
        if p.is_dir() {
            for e2 in std::fs::read_dir(&p).into_iter().flatten().flatten() {
                files.push(e2.path());
            }
        } else {
            files.push(p);
        }
    }
    files
}

fn main() {
    let src_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/mnt/v/heic".to_string());
    let out_dir = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "/mnt/v/hdr-corpus/refs".to_string());
    std::fs::create_dir_all(&out_dir).expect("create out dir");

    let (mut ok, mut skip) = (0usize, 0usize);
    for path in collect_files(&src_dir) {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let decoded = match ext.as_str() {
            "jpg" | "jpeg" => decode_jpeg(&bytes),
            "heic" => decode_heic(&bytes),
            _ => continue,
        };
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let hdr = match decoded {
            Ok(h) => h,
            Err(e) => {
                eprintln!("skip {name}: {e}");
                skip += 1;
                continue;
            }
        };
        let (w, h, rgb, mean_y, max_y) = to_rgb_nits(&hdr);
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("img");
        let exr_path = Path::new(&out_dir).join(format!("{stem}.exr"));
        let img: image::Rgb32FImage = match image::ImageBuffer::from_raw(w as u32, h as u32, rgb) {
            Some(i) => i,
            None => {
                eprintln!("skip {name}: buffer build failed");
                skip += 1;
                continue;
            }
        };
        if let Err(e) = img.save(&exr_path) {
            eprintln!("skip {name}: exr save {e}");
            skip += 1;
            continue;
        }
        println!(
            "{:<34} [{ext:>4}] {w}x{h}  luma mean {mean_y:>6.1}  max {max_y:>7.1} cd/m²",
            name
        );
        ok += 1;
    }
    eprintln!("done: {ok} HDR EXR written, {skip} skipped → {out_dir}");
}
