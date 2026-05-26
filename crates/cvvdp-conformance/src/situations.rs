//! Deterministic situation corpus for the cvvdp conformance matrix.
//!
//! A *situation* is one `(ref_rgb, dist_rgb)` pair plus a label and
//! the content/distortion class it exercises. Every situation is
//! produced by pure, PRNG-free modular arithmetic (synthetic) or by
//! loading a fixed corpus image + applying a deterministic distortion
//! (real). Determinism matters because the SAME bytes must be scored
//! by both the Rust impls AND the pycvvdp reference: the golden
//! builder (`scripts/cvvdp_goldens/build_conformance_goldens.py`)
//! reads the exact PNGs `emit_situations` writes, so there is zero
//! risk of a synth-vs-Python divergence.
//!
//! Situations are grouped into:
//!
//! - **Common content** — CID22 photo + GB82-SC screenshot crops.
//! - **Common distortions** — JPEG q5/q30/q60/q90, gaussian blur,
//!   white noise at several strengths spanning JOD ~10 → ~6.
//! - **Niche content** — tiny / large / odd-prime dims, flat color,
//!   high-frequency checkerboard, smooth gradient, 1px spike,
//!   near-black / near-white extremes.
//! - **Niche distortions** — near-lossless, heavily distorted, pure
//!   chroma shift, pure luma shift, single-block perturbation,
//!   banding.
//! - **HDR-specific** — highlight-clipping content + wide-gamut
//!   out-of-sRGB colors (scored on PQ / HLG / linear display models).

use std::path::PathBuf;

/// One conformance situation: a labelled `(ref, dist)` image pair.
#[derive(Clone)]
pub struct Situation {
    /// Stable identifier — used as the manifest key and TSV column.
    pub name: &'static str,
    /// Content/distortion class for grouping in reports.
    pub class: SituationClass,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Reference RGB8 bytes, row-major, `width*height*3` long.
    pub reference: Vec<u8>,
    /// Distorted RGB8 bytes, same layout/length.
    pub distorted: Vec<u8>,
}

/// Grouping of situations for reporting + tolerance policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SituationClass {
    CommonPhoto,
    CommonScreenshot,
    CommonDistortion,
    NicheContent,
    NicheDistortion,
    Hdr,
}

impl SituationClass {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            SituationClass::CommonPhoto => "common_photo",
            SituationClass::CommonScreenshot => "common_screenshot",
            SituationClass::CommonDistortion => "common_distortion",
            SituationClass::NicheContent => "niche_content",
            SituationClass::NicheDistortion => "niche_distortion",
            SituationClass::Hdr => "hdr",
        }
    }
}

// ---------------------------------------------------------------------------
// Deterministic synthetic patterns (PRNG-free).
// ---------------------------------------------------------------------------

/// Mixed-coefficient deterministic photo-ish reference (matches the
/// `synth_pair_ref` family used by the existing cvvdp-gpu fixtures so
/// the conformance corpus is consistent with prior goldens).
fn synth_photo(w: usize, h: usize) -> Vec<u8> {
    let mut b = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let r = (((x * 17 + y * 5) % 251) as u8).wrapping_add(40);
            let g = (((x * 11 + y * 13) % 247) as u8).wrapping_add(40);
            let bb = (((x * 7 + y * 19) % 241) as u8).wrapping_add(40);
            let i = (y * w + x) * 3;
            b[i] = r;
            b[i + 1] = g;
            b[i + 2] = bb;
        }
    }
    b
}

/// Smooth diagonal luma gradient (0..255 across the diagonal), gray.
fn gradient(w: usize, h: usize) -> Vec<u8> {
    let mut b = vec![0u8; w * h * 3];
    let denom = (w + h).max(1) as f32;
    for y in 0..h {
        for x in 0..w {
            let v = (((x + y) as f32 / denom) * 255.0).round().clamp(0.0, 255.0) as u8;
            let i = (y * w + x) * 3;
            b[i] = v;
            b[i + 1] = v;
            b[i + 2] = v;
        }
    }
    b
}

/// 1px-period checkerboard — maximum spatial frequency. Alternates
/// near-black / near-white per pixel.
fn checkerboard(w: usize, h: usize) -> Vec<u8> {
    let mut b = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let v = if (x + y) % 2 == 0 { 16u8 } else { 239u8 };
            let i = (y * w + x) * 3;
            b[i] = v;
            b[i + 1] = v;
            b[i + 2] = v;
        }
    }
    b
}

/// Flat single color fill.
fn flat(w: usize, h: usize, rgb: [u8; 3]) -> Vec<u8> {
    let mut b = vec![0u8; w * h * 3];
    for px in b.chunks_exact_mut(3) {
        px.copy_from_slice(&rgb);
    }
    b
}

/// HDR-ish content with a bright highlight region (clips to white on
/// SDR, but informative on a PQ/HLG/linear display model) over a
/// mid-gray field with a colored wedge.
fn highlight_content(w: usize, h: usize) -> Vec<u8> {
    let mut b = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            // mid-gray base
            let mut r = 96u8;
            let mut g = 96u8;
            let mut bb = 96u8;
            // bright clipping disk near the center
            let cx = w as f32 / 2.0;
            let cy = h as f32 / 2.0;
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let rad = (dx * dx + dy * dy).sqrt();
            if rad < (w.min(h) as f32) * 0.18 {
                r = 255;
                g = 255;
                bb = 255;
            }
            // colored wedge on the left third
            if (x as f32) < (w as f32) * 0.33 {
                r = r.saturating_add(80);
                bb = bb.saturating_sub(40);
            }
            let i = (y * w + x) * 3;
            b[i] = r;
            b[i + 1] = g;
            b[i + 2] = bb;
        }
    }
    b
}

/// Wide-gamut-ish content: heavily saturated primaries that sit at
/// or beyond the sRGB gamut edge (pure R/G/B bars). On BT.2020/P3
/// display models these exercise the primaries matrix path.
fn wide_gamut_bars(w: usize, h: usize) -> Vec<u8> {
    let mut b = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let band = (x * 4) / w; // 0..3
            let rgb = match band {
                0 => [255u8, 0, 0],
                1 => [0, 255, 0],
                2 => [0, 0, 255],
                _ => [255, 255, 0],
            };
            let i = (y * w + x) * 3;
            b[i] = rgb[0];
            b[i + 1] = rgb[1];
            b[i + 2] = rgb[2];
        }
    }
    b
}

// ---------------------------------------------------------------------------
// Deterministic distortions (PRNG-free where possible; the "noise"
// distortion uses a fixed LCG seeded per-image so it's reproducible).
// ---------------------------------------------------------------------------

/// Per-channel saturating offset — the canonical cvvdp fixture
/// distortion (small directional perturbation).
fn offset_dist(reference: &[u8], dr: i16, dg: i16, db: i16) -> Vec<u8> {
    reference
        .chunks_exact(3)
        .flat_map(|p| {
            [
                (i16::from(p[0]) + dr).clamp(0, 255) as u8,
                (i16::from(p[1]) + dg).clamp(0, 255) as u8,
                (i16::from(p[2]) + db).clamp(0, 255) as u8,
            ]
        })
        .collect()
}

/// Deterministic additive noise via a fixed-seed LCG. `amp` is the
/// peak ± perturbation magnitude in code values.
fn noise_dist(reference: &[u8], amp: i32, seed: u64) -> Vec<u8> {
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut next = || {
        // SplitMix64-ish step — fully deterministic.
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    reference
        .iter()
        .map(|&v| {
            let r = (next() % ((2 * amp + 1) as u64)) as i32 - amp;
            (i32::from(v) + r).clamp(0, 255) as u8
        })
        .collect()
}

/// Separable box-blur approximation of a gaussian — `radius` taps
/// each axis. Deterministic. Edges clamp to the border pixel.
fn blur_dist(reference: &[u8], w: usize, h: usize, radius: usize) -> Vec<u8> {
    let blur_axis = |src: &[u8], horizontal: bool| -> Vec<u8> {
        let mut out = vec![0u8; src.len()];
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    let mut acc = 0u32;
                    let mut cnt = 0u32;
                    for d in -(radius as isize)..=(radius as isize) {
                        let (sx, sy) = if horizontal {
                            ((x as isize + d).clamp(0, w as isize - 1) as usize, y)
                        } else {
                            (x, (y as isize + d).clamp(0, h as isize - 1) as usize)
                        };
                        acc += u32::from(src[(sy * w + sx) * 3 + c]);
                        cnt += 1;
                    }
                    out[(y * w + x) * 3 + c] = (acc / cnt) as u8;
                }
            }
        }
        out
    };
    let h_pass = blur_axis(reference, true);
    blur_axis(&h_pass, false)
}

/// Posterize each channel to `levels` bins — models banding.
fn banding_dist(reference: &[u8], levels: u32) -> Vec<u8> {
    let step = 256 / levels.max(2);
    reference
        .iter()
        .map(|&v| {
            let q = (u32::from(v) / step) * step + step / 2;
            q.min(255) as u8
        })
        .collect()
}

/// Perturb a single 8×8 block in the top-left quadrant — tests
/// whether the spatial pool localizes a small error correctly
/// (the kind of error a global JOD can mask).
fn single_block_dist(reference: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = reference.to_vec();
    let bx = (w / 4).min(w.saturating_sub(8));
    let by = (h / 4).min(h.saturating_sub(8));
    for y in by..(by + 8).min(h) {
        for x in bx..(bx + 8).min(w) {
            let i = (y * w + x) * 3;
            out[i] = out[i].saturating_add(64);
            out[i + 1] = out[i + 1].saturating_sub(40);
            out[i + 2] = out[i + 2].saturating_add(20);
        }
    }
    out
}

/// JPEG round-trip at quality `q` using the `image` crate's encoder.
/// Deterministic for a given input + q. Returns RGB8 of the decoded
/// JPEG (which is what the metric sees).
fn jpeg_roundtrip(reference: &[u8], w: u32, h: u32, q: u8) -> Vec<u8> {
    use image::codecs::jpeg::JpegEncoder;
    use image::{ImageBuffer, Rgb};
    let img: ImageBuffer<Rgb<u8>, _> =
        ImageBuffer::from_raw(w, h, reference.to_vec()).expect("rgb buffer");
    let mut buf = Vec::new();
    {
        let mut enc = JpegEncoder::new_with_quality(&mut buf, q);
        enc.encode(reference, w, h, image::ExtendedColorType::Rgb8)
            .expect("jpeg encode");
        let _ = &img; // keep ImageBuffer alive for documentation parity
    }
    let decoded = image::load_from_memory_with_format(&buf, image::ImageFormat::Jpeg)
        .expect("jpeg decode")
        .to_rgb8();
    assert_eq!(decoded.width(), w);
    assert_eq!(decoded.height(), h);
    decoded.into_raw()
}

// ---------------------------------------------------------------------------
// Real corpus loading.
// ---------------------------------------------------------------------------

/// Load a center crop of a corpus PNG at `(w, h)`. Returns `None` if
/// the corpus image is not present on this host (so the harness
/// degrades to synthetic-only without a hard failure).
fn load_corpus_crop(path: &PathBuf, w: u32, h: u32) -> Option<Vec<u8>> {
    let img = image::ImageReader::open(path)
        .ok()?
        .decode()
        .ok()?
        .to_rgb8();
    let (iw, ih) = (img.width(), img.height());
    if iw < w || ih < h {
        return None;
    }
    let ox = (iw - w) / 2;
    let oy = (ih - h) / 2;
    let mut out = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let p = img.get_pixel(ox + x, oy + y);
            let i = ((y * w + x) * 3) as usize;
            out[i] = p[0];
            out[i + 1] = p[1];
            out[i + 2] = p[2];
        }
    }
    Some(out)
}

/// Default CID22-512 photo + GB82-SC screenshot used for the
/// "common content" rows. Both are well-known fixtures in
/// `~/work/codec-corpus`.
fn cid22_photo_path() -> PathBuf {
    corpus_root().join("CID22/CID22-512/training/1001682.png")
}

fn gb82_screenshot_path() -> PathBuf {
    corpus_root().join("gb82-sc/codec_wiki.png")
}

fn corpus_root() -> PathBuf {
    if let Ok(p) = std::env::var("CODEC_CORPUS_ROOT") {
        PathBuf::from(p)
    } else if let Some(home) = std::env::var_os("HOME") {
        PathBuf::from(home).join("work/codec-corpus")
    } else {
        PathBuf::from("/home/lilith/work/codec-corpus")
    }
}

// ---------------------------------------------------------------------------
// The situation registry.
// ---------------------------------------------------------------------------

/// Build the full conformance situation corpus. Real-corpus
/// situations are included only when the corpus file is present;
/// the synthetic situations (the majority) are always present, so the
/// matrix exceeds the ≥15-situation acceptance gate even on a host
/// without `~/work/codec-corpus`.
#[must_use]
pub fn all_situations() -> Vec<Situation> {
    let mut out = Vec::new();

    let mk = |name: &'static str,
              class: SituationClass,
              w: u32,
              h: u32,
              reference: Vec<u8>,
              distorted: Vec<u8>|
     -> Situation {
        assert_eq!(reference.len(), (w * h * 3) as usize, "{name} ref len");
        assert_eq!(distorted.len(), (w * h * 3) as usize, "{name} dist len");
        Situation {
            name,
            class,
            width: w,
            height: h,
            reference,
            distorted,
        }
    };

    // --- Common content: a 256×256 synthetic photo + a real CID22
    //     photo crop + a real GB82-SC screenshot crop, each with a
    //     moderate JPEG distortion. ---
    {
        let r = synth_photo(256, 256);
        let d = jpeg_roundtrip(&r, 256, 256, 60);
        out.push(mk(
            "synth_photo_256_jpeg60",
            SituationClass::CommonPhoto,
            256,
            256,
            r,
            d,
        ));
    }
    if let Some(r) = load_corpus_crop(&cid22_photo_path(), 512, 512) {
        let d = jpeg_roundtrip(&r, 512, 512, 60);
        out.push(mk(
            "cid22_photo_512_jpeg60",
            SituationClass::CommonPhoto,
            512,
            512,
            r,
            d,
        ));
    }
    if let Some(r) = load_corpus_crop(&gb82_screenshot_path(), 512, 512) {
        let d = jpeg_roundtrip(&r, 512, 512, 90);
        out.push(mk(
            "gb82_screenshot_512_jpeg90",
            SituationClass::CommonScreenshot,
            512,
            512,
            r,
            d,
        ));
    }

    // --- Common distortions on a 256×256 synth photo, spanning the
    //     JOD range from near-lossless (q90) down to heavy (q5),
    //     plus blur + noise at a couple of strengths. ---
    {
        let r = synth_photo(256, 256);
        for q in [90u8, 60, 30, 5] {
            let d = jpeg_roundtrip(&r, 256, 256, q);
            let name: &'static str = match q {
                90 => "synth_jpeg_q90",
                60 => "synth_jpeg_q60",
                30 => "synth_jpeg_q30",
                5 => "synth_jpeg_q5",
                _ => unreachable!(),
            };
            out.push(mk(
                name,
                SituationClass::CommonDistortion,
                256,
                256,
                r.clone(),
                d,
            ));
        }
        out.push(mk(
            "synth_blur_r2",
            SituationClass::CommonDistortion,
            256,
            256,
            r.clone(),
            blur_dist(&r, 256, 256, 2),
        ));
        out.push(mk(
            "synth_blur_r5",
            SituationClass::CommonDistortion,
            256,
            256,
            r.clone(),
            blur_dist(&r, 256, 256, 5),
        ));
        out.push(mk(
            "synth_noise_amp12",
            SituationClass::CommonDistortion,
            256,
            256,
            r.clone(),
            noise_dist(&r, 12, 0xC0FFEE),
        ));
        out.push(mk(
            "synth_noise_amp40",
            SituationClass::CommonDistortion,
            256,
            256,
            r.clone(),
            noise_dist(&r, 40, 0xC0FFEE),
        ));
    }

    // --- Niche content: extreme dims + degenerate patterns. ---
    {
        // tiny
        let r = synth_photo(16, 16);
        out.push(mk(
            "tiny_16x16_offset",
            SituationClass::NicheContent,
            16,
            16,
            r.clone(),
            offset_dist(&r, -8, -4, 12),
        ));
        let r = synth_photo(32, 32);
        out.push(mk(
            "tiny_32x32_offset",
            SituationClass::NicheContent,
            32,
            32,
            r.clone(),
            offset_dist(&r, -8, -4, 12),
        ));
        // large
        let r = synth_photo(1024, 1024);
        out.push(mk(
            "large_1024_jpeg60",
            SituationClass::NicheContent,
            1024,
            1024,
            r.clone(),
            jpeg_roundtrip(&r, 1024, 1024, 60),
        ));
        // odd / prime dims
        let r = synth_photo(97, 101);
        out.push(mk(
            "odd_97x101_offset",
            SituationClass::NicheContent,
            97,
            101,
            r.clone(),
            offset_dist(&r, -8, -4, 12),
        ));
        let r = synth_photo(255, 255);
        out.push(mk(
            "odd_255x255_offset",
            SituationClass::NicheContent,
            255,
            255,
            r.clone(),
            offset_dist(&r, -8, -4, 12),
        ));
        // flat single color
        let r = flat(128, 128, [128, 128, 128]);
        out.push(mk(
            "flat_gray_offset",
            SituationClass::NicheContent,
            128,
            128,
            r.clone(),
            offset_dist(&r, 4, 4, 4),
        ));
        // high-frequency checkerboard
        let r = checkerboard(128, 128);
        out.push(mk(
            "checkerboard_blur_r2",
            SituationClass::NicheContent,
            128,
            128,
            r.clone(),
            blur_dist(&r, 128, 128, 2),
        ));
        // smooth gradient + banding
        let r = gradient(256, 256);
        out.push(mk(
            "gradient_banding_16",
            SituationClass::NicheContent,
            256,
            256,
            r.clone(),
            banding_dist(&r, 16),
        ));
        // 1px spike on a flat field
        let mut r = flat(128, 128, [100, 100, 100]);
        let mut d = r.clone();
        let spike = (64 * 128 + 64) * 3;
        d[spike] = 255;
        d[spike + 1] = 255;
        d[spike + 2] = 255;
        out.push(mk(
            "single_pixel_spike",
            SituationClass::NicheContent,
            128,
            128,
            std::mem::take(&mut r),
            std::mem::take(&mut d),
        ));
        // near-black extreme
        let r = flat(128, 128, [2, 2, 2]);
        out.push(mk(
            "near_black_offset",
            SituationClass::NicheContent,
            128,
            128,
            r.clone(),
            offset_dist(&r, 6, 6, 6),
        ));
        // near-white extreme
        let r = flat(128, 128, [252, 252, 252]);
        out.push(mk(
            "near_white_offset",
            SituationClass::NicheContent,
            128,
            128,
            r.clone(),
            offset_dist(&r, -6, -6, -6),
        ));
    }

    // --- Niche distortions on a 256×256 synth photo. ---
    {
        let r = synth_photo(256, 256);
        // near-lossless: 1-code offset only on blue
        out.push(mk(
            "near_lossless_b1",
            SituationClass::NicheDistortion,
            256,
            256,
            r.clone(),
            offset_dist(&r, 0, 0, 1),
        ));
        // heavily distorted: JPEG q2
        out.push(mk(
            "heavy_jpeg_q2",
            SituationClass::NicheDistortion,
            256,
            256,
            r.clone(),
            jpeg_roundtrip(&r, 256, 256, 2),
        ));
        // pure chroma shift: rotate R↔B (luma roughly preserved)
        let chroma = r
            .chunks_exact(3)
            .flat_map(|p| [p[2], p[1], p[0]])
            .collect::<Vec<u8>>();
        out.push(mk(
            "pure_chroma_swap",
            SituationClass::NicheDistortion,
            256,
            256,
            r.clone(),
            chroma,
        ));
        // pure luma shift: uniform +18 on all channels
        out.push(mk(
            "pure_luma_shift_18",
            SituationClass::NicheDistortion,
            256,
            256,
            r.clone(),
            offset_dist(&r, 18, 18, 18),
        ));
        // single-block perturbation
        out.push(mk(
            "single_block_8x8",
            SituationClass::NicheDistortion,
            256,
            256,
            r.clone(),
            single_block_dist(&r, 256, 256),
        ));
        // banding (8 levels — aggressive)
        out.push(mk(
            "banding_8",
            SituationClass::NicheDistortion,
            256,
            256,
            r.clone(),
            banding_dist(&r, 8),
        ));
    }

    // --- HDR-specific content (scored on HDR display models). ---
    {
        let r = highlight_content(256, 256);
        out.push(mk(
            "hdr_highlight_clip_offset",
            SituationClass::Hdr,
            256,
            256,
            r.clone(),
            offset_dist(&r, -10, -6, 14),
        ));
        let r = wide_gamut_bars(256, 256);
        out.push(mk(
            "hdr_wide_gamut_offset",
            SituationClass::Hdr,
            256,
            256,
            r.clone(),
            offset_dist(&r, -12, 8, -12),
        ));
        let r = highlight_content(256, 256);
        out.push(mk(
            "hdr_highlight_blur_r3",
            SituationClass::Hdr,
            256,
            256,
            r.clone(),
            blur_dist(&r, 256, 256, 3),
        ));
    }

    out
}
