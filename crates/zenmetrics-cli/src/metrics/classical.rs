//! Basic full-reference metrics on encoded RGB8 pixels.
//!
//! SSIM follows Wang et al.'s 11x11 Gaussian window (sigma 1.5) with
//! reflected borders and constants for unit-range signals. MS-SSIM is the
//! upstream `msssim` crate (`--metric msssim`), not this module.

use crate::decode::Rgb8Image;

#[derive(Clone, Copy)]
pub enum Kind {
    Psnr,
    PsnrY,
    PsnrY601,
    PsnrYStudio601,
    PsnrYLibvmaf,
    Ssim,
}

/// MSE between the per-pixel luma of `r` and `d` under luma function
/// `y` — the shared shape of every luma-PSNR variant (they differ only
/// in the luma convention `y`).
fn luma_mse(r: &Rgb8Image, d: &Rgb8Image, y: impl Fn(&[u8]) -> f64) -> f64 {
    r.pixels
        .as_chunks::<3>()
        .0
        .iter()
        .zip(d.pixels.as_chunks::<3>().0)
        .map(|(a, b)| {
            let delta = y(a) - y(b);
            delta * delta
        })
        .sum::<f64>()
        / (r.pixels.len() / 3) as f64
}

pub fn score(kind: Kind, r: &Rgb8Image, d: &Rgb8Image) -> Result<f64, String> {
    if r.width != d.width || r.height != d.height || r.pixels.len() != d.pixels.len() {
        return Err("metric images must have matching dimensions".into());
    }
    let n = (r.width as usize)
        .checked_mul(r.height as usize)
        .ok_or("image dimensions overflow")?;
    if r.pixels.len() != n * 3 || n == 0 {
        return Err("expected nonempty RGB8 images".into());
    }
    match kind {
        Kind::Psnr => {
            let mse = r
                .pixels
                .iter()
                .zip(&d.pixels)
                .map(|(&a, &b)| {
                    let delta = f64::from(a) - f64::from(b);
                    delta * delta
                })
                .sum::<f64>()
                / (3 * n) as f64;
            Ok(psnr(mse))
        }
        Kind::PsnrY => {
            // BT.709 full-range luma on encoded RGB values.
            Ok(psnr(luma_mse(r, d, |x| {
                0.2126 * f64::from(x[0]) + 0.7152 * f64::from(x[1]) + 0.0722 * f64::from(x[2])
            })))
        }
        Kind::PsnrY601 => {
            // BT.601 full-range luma — the MATLAB `rgb2gray` weights
            // (0.299/0.587/0.114), the most common "PSNR-Y" convention
            // in the literature. NOT the same numbers as `PsnrY`.
            Ok(psnr(luma_mse(r, d, |x| {
                0.299 * f64::from(x[0]) + 0.587 * f64::from(x[1]) + 0.114 * f64::from(x[2])
            })))
        }
        Kind::PsnrYStudio601 => {
            // Studio-swing BT.601 Y, u8-rounded: `round(16 +
            // (65.481R + 128.553G + 24.966B)/255)` — the JPEG/JFIF
            // YUV420 luma convention, the ffmpeg/libvmaf Y plane, and
            // the JPEG AIC-4 `PSNR-Y` column. Same per-pixel formula
            // (and f32 arithmetic) as `studio601_y_plane`.
            Ok(psnr(luma_mse(r, d, |x| {
                f64::from(
                    (16.0
                        + (65.481 * x[0] as f32 + 128.553 * x[1] as f32 + 24.966 * x[2] as f32)
                            / 255.0)
                        .round()
                        .clamp(0.0, 255.0) as u8,
                )
            })))
        }
        Kind::PsnrYLibvmaf => {
            // Studio-swing BT.709 Y, u8-rounded — the `to_yuv420` Y
            // plane in `metrics/vmaf.rs` (`round(16 + 219·L709)` on
            // normalized RGB), i.e. what libvmaf's `psnr` aux feature
            // reported under the removed exec adapter. f64 arithmetic
            // and operation order match `to_yuv420` exactly.
            Ok(psnr(luma_mse(r, d, |x| {
                let lum = 0.2126 * (f64::from(x[0]) / 255.0)
                    + 0.7152 * (f64::from(x[1]) / 255.0)
                    + 0.0722 * (f64::from(x[2]) / 255.0);
                f64::from((16.0 + 219.0 * lum).round().clamp(16.0, 235.0) as u8)
            })))
        }
        Kind::Ssim => {
            let mut sum = 0.0;
            for channel in 0..3 {
                let a: Vec<f64> = r
                    .pixels
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .map(|p| f64::from(p[channel]) / 255.0)
                    .collect();
                let b: Vec<f64> = d
                    .pixels
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .map(|p| f64::from(p[channel]) / 255.0)
                    .collect();
                sum += window_score(&a, &b, r.width as usize, r.height as usize);
            }
            Ok(sum / 3.0)
        }
    }
}

fn psnr(mse: f64) -> f64 {
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0 * 255.0 / mse).log10()
    }
}

fn reflect(x: isize, n: usize) -> usize {
    if n == 1 {
        return 0;
    }
    let period = 2 * (n as isize - 1);
    let v = x.rem_euclid(period);
    if v >= n as isize {
        (period - v) as usize
    } else {
        v as usize
    }
}

fn kernel11() -> [f64; 11] {
    let mut k = [0.0; 11];
    for (i, v) in k.iter_mut().enumerate() {
        *v = (-((i as f64 - 5.0).powi(2)) / 4.5).exp();
    }
    let norm: f64 = k.iter().sum();
    for v in &mut k {
        *v /= norm;
    }
    k
}

/// Separable 11-tap Gaussian blur. `src2`/`sq` fuse the input transform into
/// the row read (product with a second image, or square), so no transformed
/// plane is ever materialized. Border outputs use reflected taps; interior
/// outputs accumulate over equal-length slices — the same i-ordered sum the
/// per-pixel loop computed, but bounds-check-free and vectorizable.
fn gaussian_into(
    src: &[f64],
    src2: Option<&[f64]>,
    sq: bool,
    w: usize,
    h: usize,
    k: &[f64; 11],
    dst: &mut [f64],
    temp: &mut [f64],
) {
    let mut row = vec![0.0_f64; w];
    let xlo = 5.min(w);
    let xhi = w.saturating_sub(5).max(xlo);
    for y in 0..h {
        let srow = &src[y * w..y * w + w];
        let out = &mut temp[y * w..y * w + w];
        match src2 {
            None if !sq => row.copy_from_slice(srow),
            None => {
                for (r, &v) in row.iter_mut().zip(srow) {
                    *r = v * v;
                }
            }
            Some(b2) => {
                let brow = &b2[y * w..y * w + w];
                for ((r, &v), &u) in row.iter_mut().zip(srow).zip(brow) {
                    *r = v * u;
                }
            }
        }
        for (x, o) in out.iter_mut().enumerate().take(xlo) {
            *o = htap(&row, w, x, k);
        }
        for (x, o) in out.iter_mut().enumerate().take(w).skip(xhi) {
            *o = htap(&row, w, x, k);
        }
        let span = xhi - xlo;
        if span > 0 {
            let s0 = &row[0..span];
            for (o, &v) in out[xlo..xhi].iter_mut().zip(s0) {
                *o = v * k[0];
            }
            for (i, &ki) in k.iter().enumerate().skip(1) {
                let s = &row[i..i + span];
                for (o, &v) in out[xlo..xhi].iter_mut().zip(s) {
                    *o += v * ki;
                }
            }
        }
    }
    let ylo = 5.min(h);
    let yhi = h.saturating_sub(5).max(ylo);
    for y in 0..ylo {
        for x in 0..w {
            dst[y * w + x] = vtap(temp, w, h, x, y, k);
        }
    }
    for y in yhi..h {
        for x in 0..w {
            dst[y * w + x] = vtap(temp, w, h, x, y, k);
        }
    }
    let vspan = (yhi - ylo) * w;
    if vspan > 0 {
        let s0 = &temp[(ylo - 5) * w..(ylo - 5) * w + vspan];
        for (o, &v) in dst[ylo * w..yhi * w].iter_mut().zip(s0) {
            *o = v * k[0];
        }
        for (i, &ki) in k.iter().enumerate().skip(1) {
            let s = &temp[(ylo - 5 + i) * w..(ylo - 5 + i) * w + vspan];
            for (o, &v) in dst[ylo * w..yhi * w].iter_mut().zip(s) {
                *o += v * ki;
            }
        }
    }
}

fn htap(row: &[f64], w: usize, x: usize, k: &[f64; 11]) -> f64 {
    let mut sum = 0.0;
    for (i, &ki) in k.iter().enumerate() {
        sum += ki * row[reflect(x as isize + i as isize - 5, w)];
    }
    sum
}

fn vtap(t: &[f64], w: usize, h: usize, x: usize, y: usize, k: &[f64; 11]) -> f64 {
    let mut sum = 0.0;
    for (i, &ki) in k.iter().enumerate() {
        sum += ki * t[reflect(y as isize + i as isize - 5, h) * w + x];
    }
    sum
}

// Returns means of SSIM and contrast-structure terms.
fn window_score(a: &[f64], b: &[f64], w: usize, h: usize) -> f64 {
    let k = kernel11();
    let n = a.len();
    let mut temp = vec![0.0; n];
    let mut ma = vec![0.0; n];
    let mut mb = vec![0.0; n];
    let mut aa = vec![0.0; n];
    let mut bb = vec![0.0; n];
    let mut ab = vec![0.0; n];
    gaussian_into(a, None, false, w, h, &k, &mut ma, &mut temp);
    gaussian_into(b, None, false, w, h, &k, &mut mb, &mut temp);
    gaussian_into(a, None, true, w, h, &k, &mut aa, &mut temp);
    gaussian_into(b, None, true, w, h, &k, &mut bb, &mut temp);
    gaussian_into(a, Some(b), false, w, h, &k, &mut ab, &mut temp);
    let mut ssim = 0.0;
    for ((((&aa_i, &bb_i), &ab_i), &ma_i), &mb_i) in aa.iter().zip(&bb).zip(&ab).zip(&ma).zip(&mb) {
        let va = (aa_i - ma_i * ma_i).max(0.0);
        let vb = (bb_i - mb_i * mb_i).max(0.0);
        let cov = ab_i - ma_i * mb_i;
        let luminance = (2.0 * ma_i * mb_i + 0.0001) / (ma_i * ma_i + mb_i * mb_i + 0.0001);
        let contrast = (2.0 * cov + 0.0009) / (va + vb + 0.0009);
        ssim += luminance * contrast;
    }
    ssim / n as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn identical_and_impulse() {
        let pixels = vec![128; 192 * 192 * 3];
        let r = Rgb8Image {
            pixels: pixels.clone(),
            width: 192,
            height: 192,
        };
        let mut d = Rgb8Image {
            pixels,
            width: 192,
            height: 192,
        };
        assert!(score(Kind::Psnr, &r, &d).unwrap().is_infinite());
        assert!((score(Kind::Ssim, &r, &d).unwrap() - 1.0).abs() < 1e-10);
        d.pixels[0] = 0;
        assert!(score(Kind::Psnr, &r, &d).unwrap().is_finite());
        assert!(score(Kind::PsnrY, &r, &d).unwrap().is_finite());
        assert!(score(Kind::Ssim, &r, &d).unwrap() < 1.0);
    }

    /// Every luma-PSNR convention scores identically-shaped sane values:
    /// infinite on identical input, finite on distorted, and the
    /// distinct conventions generally disagree (they are different
    /// metrics, not aliases).
    #[test]
    fn luma_psnr_variants_are_distinct_and_sane() {
        let n = 64 * 64;
        let r = Rgb8Image {
            pixels: (0..n)
                .flat_map(|i| [i as u8, (i * 3) as u8, (i * 7) as u8])
                .collect(),
            width: 64,
            height: 64,
        };
        let mut d = r.pixels.clone();
        for (i, p) in d.iter_mut().enumerate() {
            if i % 5 == 0 {
                *p = p.wrapping_add(30);
            }
        }
        let d = Rgb8Image {
            pixels: d,
            width: 64,
            height: 64,
        };
        let kinds = [
            Kind::PsnrY,
            Kind::PsnrY601,
            Kind::PsnrYStudio601,
            Kind::PsnrYLibvmaf,
        ];
        for k in kinds {
            assert!(score(k, &r, &r).unwrap().is_infinite());
            assert!(score(k, &r, &d).unwrap().is_finite());
        }
        // The four conventions should not all collapse to one number.
        let scores: Vec<f64> = kinds.iter().map(|&k| score(k, &r, &d).unwrap()).collect();
        assert!(scores.iter().any(|&s| s != scores[0]));
    }

    /// `psnr-y-studio601` must equal `psnr-y` scored on the
    /// `yuv601-studio` luma ingress — the ingress builds the same
    /// rounded studio-swing plane and the house BT.709 luma of a gray
    /// (y, y, y) pixel is exactly `y` (0.2126+0.7152+0.0722 = 1).
    #[test]
    fn studio601_variant_matches_luma_ingress_path() {
        let r = Rgb8Image {
            pixels: (0..64 * 64)
                .flat_map(|i| [(i * 5) as u8, (i * 3) as u8, (i * 11) as u8])
                .collect(),
            width: 64,
            height: 64,
        };
        let mut px = r.pixels.clone();
        for (i, p) in px.iter_mut().enumerate() {
            if i % 4 == 0 {
                *p = p.wrapping_add(19);
            }
        }
        let d = Rgb8Image {
            pixels: px,
            width: 64,
            height: 64,
        };
        let via_variant = score(Kind::PsnrYStudio601, &r, &d).unwrap();
        let via_ingress = score(
            Kind::PsnrY,
            &crate::metrics::studio601_gray(&r),
            &crate::metrics::studio601_gray(&d),
        )
        .unwrap();
        // The house luma of (y,y,y) is (0.2126+0.7152+0.0722)·y — 1.0
        // only up to f64 rounding, so compare within a tight epsilon.
        assert!((via_variant - via_ingress).abs() < 1e-9);
    }
}
