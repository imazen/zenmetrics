//! VMAF scoring. The model scores (v0.6.1, v0.6.1neg, 4k v0.6.1,
//! v1.0.16 3d0h) run in-process through the in-tree `vmaf` crate, a
//! pure-Rust match of libvmaf v3.2.1. The former libvmaf aux-feature
//! extractors are replaced by in-tree ports (`msssim`'s libvmaf module
//! covers `float_ssim`/`float_ms_ssim`; `psnr`/`psnr-y` are the
//! `classical` metrics) — nothing here shells out to a `vmaf` binary.
//! Input is converted from encoded sRGB to 8-bit limited-range YUV420: BT.709
//! for `vmaf`/`vmaf-neg`/`vmaf-4k`/`vmaf-v1`, studio-swing BT.601 for the
//! `*-studio601` variants (the convention behind the JPEG AIC-4 `VMAF` and
//! `VMAF-neg` columns). Odd widths or heights are cropped to the largest even
//! size (one column/row dropped): 4:2:0 needs even planes, and on the AIC-4
//! odd-size pairs cropping and edge-replicate padding agree within 0.003.

use crate::decode::Rgb8Image;
use std::{fs::File, io::Write};

/// RGB → YUV matrix for the VMAF input planes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ingress {
    /// BT.709 limited range (the historical `vmaf*` columns).
    Bt709,
    /// Studio-swing BT.601: `Y = round(16 + (65.481R + 128.553G + 24.966B)/255)`,
    /// the `psnr-y-studio601` plane; chroma averaged over each 2×2 block.
    Studio601,
}

#[derive(Clone, Copy)]
pub enum Model {
    V061,
    V061Neg,
    V061FourK,
    V1,
}

/// Score one pair with a built-in VMAF model via the in-tree `vmaf` crate.
#[cfg(feature = "cpu-vmaf")]
pub fn score(
    r: &Rgb8Image,
    d: &Rgb8Image,
    model: Model,
    ingress: Ingress,
) -> Result<f64, Box<dyn std::error::Error>> {
    check_dims(r, d)?;
    let (w, h) = even_size(r);
    let widen = |p: Vec<u8>| p.into_iter().map(u16::from).collect::<Vec<u16>>();
    let (ry, ru, rv) = to_yuv420_even(r, ingress)?;
    let (dy, du, dv) = to_yuv420_even(d, ingress)?;
    let (ry, ru, rv) = (widen(ry), widen(ru), widen(rv));
    let (dy, du, dv) = (widen(dy), widen(du), widen(dv));
    let reference = [vmaf::Yuv420Frame {
        y: &ry,
        u: &ru,
        v: &rv,
    }];
    let distorted = [vmaf::Yuv420Frame {
        y: &dy,
        u: &du,
        v: &dv,
    }];
    let v0 = |variant| -> Result<f64, Box<dyn std::error::Error>> {
        let frames = vmaf::score_v0_420(&reference, &distorted, w, h, 8, variant)?;
        Ok(frames.first().ok_or("vmaf returned no frame")?.score)
    };
    match model {
        Model::V061 => v0(vmaf::VmafV0Variant::Standard),
        Model::V061Neg => v0(vmaf::VmafV0Variant::StandardNeg),
        Model::V061FourK => v0(vmaf::VmafV0Variant::FourK),
        Model::V1 => {
            let frames = vmaf::score_v1_420(
                &reference,
                &distorted,
                w,
                h,
                8,
                vmaf::ModelVariant::Standard1080p,
            )?;
            Ok(frames.first().ok_or("vmaf returned no frame")?.score)
        }
    }
}

pub(crate) fn check_dims(r: &Rgb8Image, d: &Rgb8Image) -> Result<(), Box<dyn std::error::Error>> {
    let (w, h) = even_size(r);
    if r.width != d.width || r.height != d.height || w < 32 || h < 32 {
        return Err(
            "VMAF requires matching dimensions of at least 32 (after cropping odd sizes to even)"
                .into(),
        );
    }
    Ok(())
}

/// The largest even size inside `image` (VMAF scores that region).
fn even_size(image: &Rgb8Image) -> (usize, usize) {
    (image.width as usize & !1, image.height as usize & !1)
}

/// [`to_yuv420`] over the even-sized top-left region, in either matrix.
#[allow(clippy::type_complexity)]
fn to_yuv420_even(
    image: &Rgb8Image,
    ingress: Ingress,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>), Box<dyn std::error::Error>> {
    let (stride, (w, h)) = (image.width as usize, even_size(image));
    if image.pixels.len() != stride * image.height as usize * 3 {
        return Err("VMAF expected RGB8 pixel buffer".into());
    }
    if ingress == Ingress::Bt709 && (w, h) == (stride, image.height as usize) {
        return to_yuv420(image);
    }
    let mut y = vec![0u8; w * h];
    let mut cb = vec![0u8; w * h / 4];
    let mut cr = vec![0u8; w * h / 4];
    for by in 0..h / 2 {
        for bx in 0..w / 2 {
            let (mut u_sum, mut v_sum) = (0.0f64, 0.0f64);
            for dy in 0..2 {
                for dx in 0..2 {
                    let (row, col) = (2 * by + dy, 2 * bx + dx);
                    let p = row * stride + col;
                    let rgb = &image.pixels[3 * p..3 * p + 3];
                    let (r, g, b) = (f64::from(rgb[0]), f64::from(rgb[1]), f64::from(rgb[2]));
                    match ingress {
                        Ingress::Bt709 => {
                            let (r, g, b) = (r / 255.0, g / 255.0, b / 255.0);
                            let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                            y[row * w + col] =
                                (16.0_f64 + 219.0 * lum).round().clamp(16.0, 235.0) as u8;
                            u_sum += -0.114572 * r - 0.385428 * g + 0.5 * b;
                            v_sum += 0.5 * r - 0.454153 * g - 0.045847 * b;
                        }
                        Ingress::Studio601 => {
                            let lum = 16.0 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0;
                            y[row * w + col] = lum.round().clamp(0.0, 255.0) as u8;
                            u_sum += (-37.797 * r - 74.203 * g + 112.0 * b) / 255.0;
                            v_sum += (112.0 * r - 93.786 * g - 18.214 * b) / 255.0;
                        }
                    }
                }
            }
            let (u, v) = match ingress {
                Ingress::Bt709 => (224.0 * u_sum / 4.0, 224.0 * v_sum / 4.0),
                Ingress::Studio601 => (u_sum / 4.0, v_sum / 4.0),
            };
            cb[by * (w / 2) + bx] = (128.0_f64 + u).round().clamp(16.0, 240.0) as u8;
            cr[by * (w / 2) + bx] = (128.0_f64 + v).round().clamp(16.0, 240.0) as u8;
        }
    }
    Ok((y, cb, cr))
}

/// Write `image` as a single-frame 8-bit BT.709 limited-range YUV420
/// plane file (the libvmaf `--pixel_format 420` input layout: Y plane,
/// then Cb, then Cr). `pub` so benchmarks/heaptrack drivers can time the
/// RGB→YUV420 conversion + serialization side of the adapter.
#[allow(dead_code)] // used by the lib target's callers, not by the bin target
pub fn write_yuv420(
    path: &std::path::Path,
    image: &Rgb8Image,
) -> Result<(), Box<dyn std::error::Error>> {
    let (y, cb, cr) = to_yuv420(image)?;
    let mut f = File::create(path)?;
    f.write_all(&y)?;
    f.write_all(&cb)?;
    f.write_all(&cr)?;
    Ok(())
}

/// Convert `image` to 8-bit BT.709 limited-range YUV420 planes (Y, Cb, Cr).
#[allow(clippy::type_complexity)]
pub(crate) fn to_yuv420(
    image: &Rgb8Image,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>), Box<dyn std::error::Error>> {
    let w = image.width as usize;
    let h = image.height as usize;
    if image.pixels.len() != w * h * 3 {
        return Err("VMAF expected RGB8 pixel buffer".into());
    }
    let mut y = vec![0u8; w * h];
    let mut cb = vec![0u8; w * h / 4];
    let mut cr = vec![0u8; w * h / 4];
    for by in 0..h / 2 {
        for bx in 0..w / 2 {
            let mut u_sum = 0.0;
            let mut v_sum = 0.0;
            for dy in 0..2 {
                for dx in 0..2 {
                    let p = (2 * by + dy) * w + 2 * bx + dx;
                    let rgb = &image.pixels[3 * p..3 * p + 3];
                    let r = f64::from(rgb[0]) / 255.0;
                    let g = f64::from(rgb[1]) / 255.0;
                    let b = f64::from(rgb[2]) / 255.0;
                    let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                    y[p] = (16.0_f64 + 219.0 * lum).round().clamp(16.0, 235.0) as u8;
                    u_sum += -0.114572 * r - 0.385428 * g + 0.5 * b;
                    v_sum += 0.5 * r - 0.454153 * g - 0.045847 * b;
                }
            }
            cb[by * (w / 2) + bx] =
                (128.0_f64 + 224.0 * u_sum / 4.0).round().clamp(16.0, 240.0) as u8;
            cr[by * (w / 2) + bx] =
                (128.0_f64 + 224.0 * v_sum / 4.0).round().clamp(16.0, 240.0) as u8;
        }
    }
    Ok((y, cb, cr))
}

/// Standalone CAMBI defaults with the AIC VMAF studio-601/even-crop ingress.
#[cfg(feature = "cpu-vmaf")]
pub(crate) fn cambi(r: &Rgb8Image, d: &Rgb8Image) -> Result<f64, Box<dyn std::error::Error>> {
    check_dims(r, d)?;
    let (w, h) = even_size(d);
    let (y, _, _) = to_yuv420_even(d, Ingress::Studio601)?;
    let y: Vec<_> = y.into_iter().map(u16::from).collect();
    Ok(vmaf::cambi_from_luma(&y, w, h, w, 8)?)
}

/// Integer luma conversion observed for swscale's RGB24 to limited BT.601.
#[cfg(feature = "cpu-vmaf")]
pub(crate) fn cambi_aic2026(
    r: &Rgb8Image,
    d: &Rgb8Image,
) -> Result<f64, Box<dyn std::error::Error>> {
    check_dims(r, d)?;
    let (w, h) = even_size(d);
    let stride = d.width as usize * 3;
    let mut y = Vec::with_capacity(w * h);
    for row in d.pixels.chunks_exact(stride).take(h) {
        for rgb in row.chunks_exact(3).take(w) {
            let t = (8414 * u32::from(rgb[0])
                + 16519 * u32::from(rgb[1])
                + 3208 * u32::from(rgb[2])
                + (32 << 14)
                + (1 << 8))
                >> 9;
            y.push(((t + 32) >> 6) as u16);
        }
    }
    Ok(vmaf::cambi_from_luma(&y, w, h, w, 8)?)
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "cpu-vmaf")]
    #[test]
    fn cambi_uses_distorted_even_crop_and_not_reference_pixels() {
        let make = |fill| Rgb8Image {
            pixels: vec![fill; 217 * 219 * 3],
            width: 217,
            height: 219,
        };
        let r = make(128);
        let mut d = make(0);
        for y in 0..219 {
            for x in 0..217 {
                let v = (32 + (x / 8) * 5) as u8;
                d.pixels[(y * 217 + x) * 3..(y * 217 + x) * 3 + 3].fill(v);
            }
        }
        let actual = cambi(&r, &d).unwrap();
        assert!(actual.is_finite());
        assert_eq!(actual, cambi(&make(11), &d).unwrap());
        let pixels: Vec<_> = (0..218)
            .flat_map(|y| d.pixels[y * 217 * 3..y * 217 * 3 + 216 * 3].iter().copied())
            .collect();
        let crop = Rgb8Image {
            pixels,
            width: 216,
            height: 218,
        };
        assert_eq!(actual, cambi(&crop, &crop).unwrap());
        assert!(cambi(&r, &crop).is_err());
    }

    use super::*;

    fn pair() -> (Rgb8Image, Rgb8Image) {
        let pixels: Vec<u8> = (0..256 * 256)
            .flat_map(|i| {
                let v = ((i * 73) % 256) as u8;
                [v, v.wrapping_add(17), v.wrapping_add(39)]
            })
            .collect();
        let r = Rgb8Image {
            pixels: pixels.clone(),
            width: 256,
            height: 256,
        };
        let mut d = Rgb8Image {
            pixels,
            width: 256,
            height: 256,
        };
        for (i, value) in d.pixels.iter_mut().enumerate() {
            if i % 7 == 0 {
                *value = value.saturating_add(12);
            }
        }
        (r, d)
    }

    #[cfg(feature = "cpu-vmaf")]
    #[test]
    fn in_tree_models_score_distortion_below_identity() {
        let (r, d) = pair();
        for (model, ingress) in [Model::V061, Model::V061Neg, Model::V061FourK, Model::V1]
            .into_iter()
            .flat_map(|m| [(m, Ingress::Bt709), (m, Ingress::Studio601)])
        {
            let same = score(&r, &r, model, ingress).unwrap();
            let dist = score(&r, &d, model, ingress).unwrap();
            assert!(same.is_finite() && dist.is_finite());
            assert!(
                dist < same,
                "distorted {dist} should score below identical {same}"
            );
        }
    }

    /// The generic converter reproduces `to_yuv420` bit for bit on even
    /// input, and on odd input scores exactly the even crop.
    #[test]
    fn even_crop_and_bt709_path_agree() {
        let (r, _) = pair();
        assert_eq!(
            to_yuv420_even(&r, Ingress::Bt709).unwrap(),
            to_yuv420(&r).unwrap()
        );
        let odd = Rgb8Image {
            pixels: (0..257 * 255)
                .flat_map(|i| [(i % 251) as u8, (i % 13) as u8, (i % 97) as u8])
                .collect(),
            width: 257,
            height: 255,
        };
        let crop = Rgb8Image {
            pixels: odd
                .pixels
                .as_chunks::<{ 257 * 3 }>()
                .0
                .iter()
                .take(254)
                .flat_map(|row| row[..256 * 3].iter().copied())
                .collect(),
            width: 256,
            height: 254,
        };
        for ingress in [Ingress::Bt709, Ingress::Studio601] {
            assert_eq!(
                to_yuv420_even(&odd, ingress).unwrap(),
                to_yuv420_even(&crop, ingress).unwrap()
            );
        }
    }

    /// Studio-601 Y is exactly the `psnr-y-studio601` plane formula.
    #[test]
    fn studio601_luma_matches_the_psnr_y_studio601_formula() {
        let (r, _) = pair();
        let (y, _, _) = to_yuv420_even(&r, Ingress::Studio601).unwrap();
        for (i, px) in r.pixels.as_chunks::<3>().0.iter().enumerate() {
            let (a, b, c) = (f64::from(px[0]), f64::from(px[1]), f64::from(px[2]));
            let want = (16.0 + (65.481 * a + 128.553 * b + 24.966 * c) / 255.0).round() as u8;
            assert_eq!(y[i], want);
        }
    }

    /// `psnr-y-libvmaf` must equal PSNR on the Y plane `to_yuv420`
    /// writes — that plane is exactly what the removed exec adapter fed
    /// libvmaf's `psnr` aux feature.
    #[test]
    fn psnr_y_libvmaf_matches_adapter_y_plane() {
        let (r, d) = pair();
        let (ry, _, _) = to_yuv420(&r).unwrap();
        let (dy, _, _) = to_yuv420(&d).unwrap();
        let mse = ry
            .iter()
            .zip(&dy)
            .map(|(&a, &b)| {
                let dl = f64::from(a) - f64::from(b);
                dl * dl
            })
            .sum::<f64>()
            / ry.len() as f64;
        let expect = 10.0 * (255.0f64 * 255.0 / mse).log10();
        let got =
            crate::metrics::classical::score(crate::metrics::classical::Kind::PsnrYLibvmaf, &r, &d)
                .unwrap();
        assert_eq!(got, expect);
    }
}
