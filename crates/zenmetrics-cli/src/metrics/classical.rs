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
    Ssim,
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
            let mse = r
                .pixels
                .chunks_exact(3)
                .zip(d.pixels.chunks_exact(3))
                .map(|(a, b)| {
                    // BT.709 full-range luma on encoded RGB values.
                    let y = |x: &[u8]| {
                        0.2126 * f64::from(x[0])
                            + 0.7152 * f64::from(x[1])
                            + 0.0722 * f64::from(x[2])
                    };
                    let delta = y(a) - y(b);
                    delta * delta
                })
                .sum::<f64>()
                / n as f64;
            Ok(psnr(mse))
        }
        Kind::Ssim => {
            let mut sum = 0.0;
            for channel in 0..3 {
                let a: Vec<f64> = r
                    .pixels
                    .chunks_exact(3)
                    .map(|p| f64::from(p[channel]) / 255.0)
                    .collect();
                let b: Vec<f64> = d
                    .pixels
                    .chunks_exact(3)
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
        for x in 0..xlo {
            out[x] = htap(&row, w, x, k);
        }
        for x in xhi..w {
            out[x] = htap(&row, w, x, k);
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
}
