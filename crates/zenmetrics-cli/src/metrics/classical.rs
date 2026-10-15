//! Basic full-reference metrics on encoded RGB8 pixels.
//!
//! SSIM follows Wang et al.'s 11x11 Gaussian window (sigma 1.5) with
//! reflected borders and constants for unit-range signals. MS-SSIM is the
//! upstream `msssim` crate (`--metric msssim`), not this module.

use crate::decode::Rgb8Image;

#[derive(Clone, Copy)]
pub(crate) enum Kind {
    Psnr,
    PsnrY,
    PsnrY601,
    PsnrYStudio601,
    PsnrYLibvmaf,
    PsnrYCbCr611,
    Ssim,
    Ciede2000,
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

pub(crate) fn score(kind: Kind, r: &Rgb8Image, d: &Rgb8Image) -> Result<f64, String> {
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
        Kind::PsnrYCbCr611 => {
            // 6:1:1 weighted PSNR on studio-swing BT.601 YCbCr —
            // `(6·PSNR_Y + PSNR_Cb + PSNR_Cr) / 8` computed in the MSE
            // domain (weighted MSE average, then converted to dB).
            // Matches JPEG AIC-4 `PSNR-YCbCr611` column.
            let (mse_y, mse_cb, mse_cr) = yuv444_studio601_mse(r, d);
            let mse = (6.0 * mse_y + mse_cb + mse_cr) / 8.0;
            Ok(psnr(mse))
        }
        Kind::Ssim => {
            Ok(ssim_streamed(&r.pixels, &d.pixels, r.width as usize, r.height as usize) / 3.0)
        }
        Kind::Ciede2000 => {
            // Mean CIEDE2000 colour difference (ΔE00).
            // Pipeline: sRGB → linear → XYZ D65 → CIELAB → ΔE00 per pixel → mean.
            // Matches JPEG AIC-4 `CIEDE2000` column.
            Ok(ciede2000_mean(r, d))
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

/// Compute MSE for studio-swing BT.601 Y, Cb, Cr planes between two RGB8 images.
/// Uses the same per-pixel conversion as `studio601_yuv444` in mod.rs:
/// Y  = round(16 + (65.481R + 128.553G + 24.966B)/255)
/// Cb = round(128 + (-37.797R - 74.203G + 112B)/255)
/// Cr = round(128 + (112R - 93.786G - 18.214B)/255)
fn yuv444_studio601_mse(r: &Rgb8Image, d: &Rgb8Image) -> (f64, f64, f64) {
    let n = (r.width as usize)
        .checked_mul(r.height as usize)
        .expect("image dimensions overflow");
    let mut sum_y = 0.0_f64;
    let mut sum_cb = 0.0_f64;
    let mut sum_cr = 0.0_f64;

    for (a, b) in r
        .pixels
        .as_chunks::<3>()
        .0
        .iter()
        .zip(d.pixels.as_chunks::<3>().0)
    {
        let (r1, g1, b1) = (a[0] as f32, a[1] as f32, a[2] as f32);
        let (r2, g2, b2) = (b[0] as f32, b[1] as f32, b[2] as f32);

        // Studio-swing BT.601 YCbCr (same as studio601_yuv444 in mod.rs)
        let y1 = (16.0 + (65.481 * r1 + 128.553 * g1 + 24.966 * b1) / 255.0)
            .round()
            .clamp(0.0, 255.0);
        let y2 = (16.0 + (65.481 * r2 + 128.553 * g2 + 24.966 * b2) / 255.0)
            .round()
            .clamp(0.0, 255.0);
        let cb1 = (128.0 + (-37.797 * r1 - 74.203 * g1 + 112.0 * b1) / 255.0)
            .round()
            .clamp(0.0, 255.0);
        let cb2 = (128.0 + (-37.797 * r2 - 74.203 * g2 + 112.0 * b2) / 255.0)
            .round()
            .clamp(0.0, 255.0);
        let cr1 = (128.0 + (112.0 * r1 - 93.786 * g1 - 18.214 * b1) / 255.0)
            .round()
            .clamp(0.0, 255.0);
        let cr2 = (128.0 + (112.0 * r2 - 93.786 * g2 - 18.214 * b2) / 255.0)
            .round()
            .clamp(0.0, 255.0);

        let dy = y1 - y2;
        let dcb = cb1 - cb2;
        let dcr = cr1 - cr2;
        sum_y += (dy * dy) as f64;
        sum_cb += (dcb * dcb) as f64;
        sum_cr += (dcr * dcr) as f64;
    }

    let n_f = n as f64;
    (sum_y / n_f, sum_cb / n_f, sum_cr / n_f)
}

/// Mean CIEDE2000 (ΔE00) between two RGB8 images.
/// Pipeline: sRGB → linear → XYZ D65 → CIELAB → ΔE00 per pixel → mean.
fn ciede2000_mean(r: &Rgb8Image, d: &Rgb8Image) -> f64 {
    let n = (r.width as usize)
        .checked_mul(r.height as usize)
        .expect("image dimensions overflow");
    let mut sum_de = 0.0_f64;

    // sRGB to XYZ D65 matrix
    const M: [[f64; 3]; 3] = [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ];

    for (a, b) in r
        .pixels
        .as_chunks::<3>()
        .0
        .iter()
        .zip(d.pixels.as_chunks::<3>().0)
    {
        // sRGB to linear
        let r1_lin = srgb_to_linear(a[0]);
        let g1_lin = srgb_to_linear(a[1]);
        let b1_lin = srgb_to_linear(a[2]);
        let r2_lin = srgb_to_linear(b[0]);
        let g2_lin = srgb_to_linear(b[1]);
        let b2_lin = srgb_to_linear(b[2]);

        // Linear RGB to XYZ
        let x1 = M[0][0] * r1_lin + M[0][1] * g1_lin + M[0][2] * b1_lin;
        let y1 = M[1][0] * r1_lin + M[1][1] * g1_lin + M[1][2] * b1_lin;
        let z1 = M[2][0] * r1_lin + M[2][1] * g1_lin + M[2][2] * b1_lin;

        let x2 = M[0][0] * r2_lin + M[0][1] * g2_lin + M[0][2] * b2_lin;
        let y2 = M[1][0] * r2_lin + M[1][1] * g2_lin + M[1][2] * b2_lin;
        let z2 = M[2][0] * r2_lin + M[2][1] * g2_lin + M[2][2] * b2_lin;

        // XYZ to CIELAB
        let (l1, a1, b1) = xyz_to_lab(x1, y1, z1);
        let (l2, a2, b2) = xyz_to_lab(x2, y2, z2);

        // CIEDE2000
        let de = ciede2000(l1, a1, b1, l2, a2, b2);
        sum_de += de;
    }

    sum_de / n as f64
}

/// Convert sRGB component [0, 255] to linear [0, 1].
fn srgb_to_linear(c: u8) -> f64 {
    let srgb = c as f64 / 255.0;
    if srgb <= 0.04045 {
        srgb / 12.92
    } else {
        ((srgb + 0.055) / 1.055).powf(2.4)
    }
}

/// Convert XYZ to CIELAB (D65 white point).
fn xyz_to_lab(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    const XN: f64 = 0.95047;
    const YN: f64 = 1.00000;
    const ZN: f64 = 1.08883;

    let fx = f_lab(x / XN);
    let fy = f_lab(y / YN);
    let fz = f_lab(z / ZN);

    let l = 116.0 * fy - 16.0;
    let a = 500.0 * (fx - fy);
    let b = 200.0 * (fy - fz);
    (l, a, b)
}

fn f_lab(t: f64) -> f64 {
    const DELTA: f64 = 6.0 / 29.0;
    if t > DELTA * DELTA * DELTA {
        t.cbrt()
    } else {
        t / (3.0 * DELTA * DELTA) + 4.0 / 29.0
    }
}

/// CIEDE2000 colour difference between two CIELAB points.
fn ciede2000(l1: f64, a1: f64, b1: f64, l2: f64, a2: f64, b2: f64) -> f64 {
    // Parametric factors (standard: kL = kC = kH = 1)
    const KL: f64 = 1.0;
    const KC: f64 = 1.0;
    const KH: f64 = 1.0;

    // Step 1: C1, C2
    let c1 = (a1 * a1 + b1 * b1).sqrt();
    let c2 = (a2 * a2 + b2 * b2).sqrt();

    // Step 2: C_bar
    let c_bar = (c1 + c2) * 0.5;

    // Step 3: G
    let c_bar7 = c_bar.powi(7);
    let g = 0.5 * (1.0 - (c_bar7 / (c_bar7 + 25.0_f64.powi(7))).sqrt());

    // Step 4: a_prime
    let a1p = a1 * (1.0 + g);
    let a2p = a2 * (1.0 + g);

    // Step 5: C_prime
    let c1p = (a1p * a1p + b1 * b1).sqrt();
    let c2p = (a2p * a2p + b2 * b2).sqrt();

    // Step 6: h_prime
    let h1p = b1.atan2(a1p);
    let h2p = b2.atan2(a2p);

    // Normalize to [0, 2π)
    let h1p = if h1p < 0.0 {
        h1p + 2.0 * std::f64::consts::PI
    } else {
        h1p
    };
    let h2p = if h2p < 0.0 {
        h2p + 2.0 * std::f64::consts::PI
    } else {
        h2p
    };

    // Step 7: delta L', C', H'
    let dl_p = l2 - l1;
    let dc_p = c2p - c1p;

    let dh_p = h2p - h1p;
    let dh_p = if c1p * c2p == 0.0 {
        0.0
    } else if dh_p > std::f64::consts::PI {
        dh_p - 2.0 * std::f64::consts::PI
    } else if dh_p < -std::f64::consts::PI {
        dh_p + 2.0 * std::f64::consts::PI
    } else {
        dh_p
    };
    let dh_p = 2.0 * (c1p * c2p).sqrt() * (dh_p * 0.5).sin();

    // Step 8: L_bar_prime, C_bar_prime, h_bar_prime
    let l_bar_p = (l1 + l2) * 0.5;
    let c_bar_p = (c1p + c2p) * 0.5;

    let h_bar_p = if c1p * c2p == 0.0 {
        h1p + h2p
    } else if (h1p - h2p).abs() <= std::f64::consts::PI {
        (h1p + h2p) * 0.5
    } else if h1p + h2p < 2.0 * std::f64::consts::PI {
        (h1p + h2p + 2.0 * std::f64::consts::PI) * 0.5
    } else {
        (h1p + h2p - 2.0 * std::f64::consts::PI) * 0.5
    };

    // Step 9: T
    let h_bar_p_deg = h_bar_p * 180.0 / std::f64::consts::PI;
    let t = 1.0 - 0.17 * (h_bar_p - std::f64::consts::PI / 6.0).cos()
        + 0.24 * (2.0 * h_bar_p).cos()
        + 0.32 * (3.0 * h_bar_p + std::f64::consts::PI / 30.0).cos()
        - 0.20 * (4.0 * h_bar_p - 63.0 * std::f64::consts::PI / 180.0).cos();

    // Step 10: SL, SC, SH
    let dl_bar_p = l_bar_p - 50.0;
    let sl = 1.0 + (0.015 * dl_bar_p * dl_bar_p) / (20.0 + dl_bar_p * dl_bar_p).sqrt();
    let sc = 1.0 + 0.045 * c_bar_p;
    let sh = 1.0 + 0.015 * c_bar_p * t;

    // Step 11: RT
    let delta_theta = std::f64::consts::PI / 6.0 * (-((h_bar_p_deg - 275.0) / 25.0).powi(2)).exp();
    let rc = 2.0 * (c_bar_p.powi(7) / (c_bar_p.powi(7) + 25.0_f64.powi(7))).sqrt();
    let rt = -rc * (2.0 * delta_theta).sin();

    // Step 12: dE00
    let term1 = (dl_p / (KL * sl)).powi(2);
    let term2 = (dc_p / (KC * sc)).powi(2);
    let term3 = (dh_p / (KH * sh)).powi(2);
    let term4 = rt * (dc_p / (KC * sc)) * (dh_p / (KH * sh));

    (term1 + term2 + term3 + term4).sqrt()
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
    let xlo = 5.min(w);
    let xhi = w.saturating_sub(5).max(xlo);
    #[cfg(not(target_family = "wasm"))]
    let mut ran_par = false;
    #[cfg(target_family = "wasm")]
    let ran_par = false;
    #[cfg(not(target_family = "wasm"))]
    if rayon::current_num_threads() > 1 && h >= 32 {
        use rayon::prelude::*;
        // Horizontal pass — rows write disjoint `temp` slices; each task
        // keeps its own tap-row scratch.
        temp.par_chunks_mut(w).enumerate().for_each_init(
            || vec![0.0_f64; w],
            |row, (y, out)| {
                let srow = &src[y * w..y * w + w];
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
                    *o = htap(row, w, x, k);
                }
                for (x, o) in out.iter_mut().enumerate().take(w).skip(xhi) {
                    *o = htap(row, w, x, k);
                }
                let span = xhi - xlo;
                if span > 0 {
                    gaussian_interior(row, 1, k, &mut out[xlo..xhi]);
                }
            },
        );
        ran_par = true;
    }
    if !ran_par {
        let mut row = vec![0.0_f64; w];
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
                gaussian_interior(&row, 1, k, &mut out[xlo..xhi]);
            }
        }
    }
    let ylo = 5.min(h);
    let yhi = h.saturating_sub(5).max(ylo);
    #[cfg(not(target_family = "wasm"))]
    let mut ran_par2 = false;
    #[cfg(target_family = "wasm")]
    let ran_par2 = false;
    #[cfg(not(target_family = "wasm"))]
    if rayon::current_num_threads() > 1 && h >= 32 {
        use rayon::prelude::*;
        dst[..ylo * w]
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, o)| {
                *o = vtap(temp, w, h, i % w, i / w, k);
            });
        dst[yhi * w..]
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, o)| {
                let i = i + yhi * w;
                *o = vtap(temp, w, h, i % w, i / w, k);
            });
        let vspan = (yhi - ylo) * w;
        if vspan > 0 {
            // Row-chunked interior: each chunk applies the same ordered
            // 11-tap accumulate to its own span — per-element order kept.
            dst[ylo * w..yhi * w]
                .par_chunks_mut(w * 16)
                .enumerate()
                .for_each(|(ci, chunk)| {
                    let base = (ylo - 5) * w + ci * w * 16;
                    gaussian_interior(&temp[base..], w, k, chunk);
                });
        }
        ran_par2 = true;
    }
    if !ran_par2 {
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
            gaussian_interior(&temp[(ylo - 5) * w..], w, k, &mut dst[ylo * w..yhi * w]);
        }
    }
}

/// Accumulate each interior block locally before writing it once. `stride`
/// is one for horizontal taps and the row width for vertical taps. The
/// eleven products and additions retain the previous f64 order.
fn gaussian_interior(src: &[f64], stride: usize, k: &[f64; 11], dst: &mut [f64]) {
    let (blocks, tail) = dst.as_chunks_mut::<16>();
    for (block, out) in blocks.iter_mut().enumerate() {
        let base = block * 16;
        let first: &[f64; 16] = src[base..base + 16].try_into().unwrap();
        let mut acc: [f64; 16] = core::array::from_fn(|j| first[j] * k[0]);
        for (tap, &weight) in k.iter().enumerate().skip(1) {
            let off = base + tap * stride;
            let values: &[f64; 16] = src[off..off + 16].try_into().unwrap();
            for j in 0..16 {
                acc[j] += values[j] * weight;
            }
        }
        *out = acc;
    }
    let base = blocks.len() * 16;
    for (j, out) in tail.iter_mut().enumerate() {
        let mut acc = src[base + j] * k[0];
        for (tap, &weight) in k.iter().enumerate().skip(1) {
            acc += src[base + j + tap * stride] * weight;
        }
        *out = acc;
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

/// Row span covered by one streaming band.
const SSIM_BAND: usize = 64;
/// Ring depth: the 11-tap window plus no more — a slot per window row.
const RING: usize = 11;

/// Streaming `window_score`: the five blurred planes (a, b, a², b², ab)
/// never materialize. Each band of [`SSIM_BAND`] output rows keeps a
/// private 11-row-per-target ring of horizontally blurred rows and folds
/// the SSIM map row as each vertical window completes. Per-element
/// arithmetic is identical to the staged pipeline (same tap order, same
/// l·c expressions); only the per-band partial sums reassociate the
/// global fold, so the result is deterministic at any thread count.
fn ssim_streamed(r: &[u8], d: &[u8], w: usize, h: usize) -> f64 {
    let k = kernel11();
    let nb = h.div_ceil(SSIM_BAND);
    let mut partials = vec![0.0; 3 * nb];
    #[cfg(not(target_family = "wasm"))]
    {
        use rayon::prelude::*;
        partials
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, out)| *out = band_ssim(r, d, w, h, i / nb, i % nb, &k));
    }
    #[cfg(target_family = "wasm")]
    for (i, out) in partials.iter_mut().enumerate() {
        *out = band_ssim(r, d, w, h, i / nb, i % nb, &k);
    }
    let n = (w * h) as f64;
    let mut sum = 0.0;
    for channel in partials.as_chunks::<3>().0 {
        sum += channel.iter().sum::<f64>() / n;
    }
    sum
}

/// One band of the streamed SSIM fold: fills the ring with the
/// H-blurred transform rows its output window needs, then emits each
/// output row's five vertical convolutions and the l·c terms in order.
fn band_ssim(r: &[u8], d: &[u8], w: usize, h: usize, ch: usize, band: usize, k: &[f64; 11]) -> f64 {
    let m0 = band * SSIM_BAND;
    let m1 = (m0 + SSIM_BAND).min(h);
    let rlo = m0.saturating_sub(5);
    let rhi = (m1 + 4).min(h - 1);
    let xlo = 5.min(w);
    let xhi = w.saturating_sub(5).max(xlo);
    let mut ring = vec![0.0; 5 * RING * w];
    // Transform rows: source channel values plus the squared/product
    // planes the five convolutions read.
    let mut trows = vec![0.0; 5 * w];
    let mut acc = vec![0.0; 5 * w];
    let mut partial = 0.0;
    let mut y_emit = m0;
    for rr in rlo..=rhi {
        let srow = &r[rr * w * 3..(rr * w + w) * 3];
        let drow = &d[rr * w * 3..(rr * w + w) * 3];
        let (ra, rest) = trows.split_at_mut(w);
        let (rb, rest) = rest.split_at_mut(w);
        let (a2, rest) = rest.split_at_mut(w);
        let (b2, ab) = rest.split_at_mut(w);
        for x in 0..w {
            let av = f64::from(srow[x * 3 + ch]) / 255.0;
            let bv = f64::from(drow[x * 3 + ch]) / 255.0;
            ra[x] = av;
            rb[x] = bv;
            a2[x] = av * av;
            b2[x] = bv * bv;
            ab[x] = av * bv;
        }
        let slot = rr % RING;
        for t in 0..5 {
            let trow = &trows[t * w..(t + 1) * w];
            let orow = &mut ring[(t * RING + slot) * w..(t * RING + slot + 1) * w];
            for (x, o) in orow.iter_mut().enumerate().take(xlo) {
                *o = htap(trow, w, x, k);
            }
            for (x, o) in orow.iter_mut().enumerate().take(w).skip(xhi) {
                *o = htap(trow, w, x, k);
            }
            if xhi > xlo {
                gaussian_interior(trow, 1, k, &mut orow[xlo..xhi]);
            }
        }
        // Emit output rows whose 11-row window is now resident.
        while y_emit < m1 && (y_emit + 5 <= rr || rr == h - 1) {
            let mut s = [0usize; RING];
            for (i, slot) in s.iter_mut().enumerate() {
                *slot = reflect(y_emit as isize + i as isize - 5, h) % RING;
            }
            for t in 0..5 {
                let base = (t * RING + s[0]) * w;
                let acc_t = &mut acc[t * w..(t + 1) * w];
                for (a, &v) in acc_t.iter_mut().zip(&ring[base..base + w]) {
                    *a = k[0] * v;
                }
                for (i, &slot_i) in s.iter().enumerate().skip(1) {
                    let ki = k[i];
                    let row_i = &ring[(t * RING + slot_i) * w..(t * RING + slot_i) * w + w];
                    for (a, &v) in acc_t.iter_mut().zip(row_i) {
                        *a += ki * v;
                    }
                }
            }
            for x in 0..w {
                let ma = acc[x];
                let mb = acc[w + x];
                let aa = acc[2 * w + x];
                let bb = acc[3 * w + x];
                let ab = acc[4 * w + x];
                let va = (aa - ma * ma).max(0.0);
                let vb = (bb - mb * mb).max(0.0);
                let cov = ab - ma * mb;
                let luminance = (2.0 * ma * mb + 0.0001) / (ma * ma + mb * mb + 0.0001);
                let contrast = (2.0 * cov + 0.0009) / (va + vb + 0.0009);
                partial += luminance * contrast;
            }
            y_emit += 1;
        }
    }
    partial
}

// Returns means of SSIM and contrast-structure terms.
#[allow(dead_code)]
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
    fn blocked_gaussian_matches_reflected_scalar_bits() {
        for (w, h) in [
            (1, 37),
            (37, 1),
            (4, 6),
            (11, 11),
            (19, 17),
            (33, 35),
            (64, 63),
        ] {
            let a: Vec<f64> = (0..w * h).map(|i| (i * 37 % 251) as f64 / 251.0).collect();
            let b: Vec<f64> = (0..w * h).map(|i| (i * 91 % 241) as f64 / 241.0).collect();
            let k = kernel11();
            for (src2, sq) in [(None, false), (None, true), (Some(b.as_slice()), false)] {
                let transformed: Vec<f64> = a
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| {
                        if let Some(other) = src2 {
                            v * other[i]
                        } else if sq {
                            v * v
                        } else {
                            v
                        }
                    })
                    .collect();
                let mut expected_h = vec![0.0; w * h];
                for y in 0..h {
                    for x in 0..w {
                        expected_h[y * w + x] = htap(&transformed[y * w..(y + 1) * w], w, x, &k);
                    }
                }
                let mut actual_h = vec![0.0; w * h];
                let mut actual = vec![0.0; w * h];
                gaussian_into(&a, src2, sq, w, h, &k, &mut actual, &mut actual_h);
                for y in 0..h {
                    for x in 0..w {
                        let i = y * w + x;
                        assert_eq!(
                            actual_h[i].to_bits(),
                            expected_h[i].to_bits(),
                            "horizontal {w}x{h}, ({x},{y}), square={sq}"
                        );
                        let expected = vtap(&expected_h, w, h, x, y, &k);
                        assert_eq!(
                            actual[i].to_bits(),
                            expected.to_bits(),
                            "vertical {w}x{h}, ({x},{y}), square={sq}"
                        );
                    }
                }
            }
        }
    }

    /// The streamed pipeline produces the same per-element arithmetic as
    /// the staged five-plane pipeline; only the global sum's association
    /// differs, so the scores agree far inside any sane tolerance.
    #[test]
    fn streamed_ssim_matches_staged() {
        let mut s = 88172645463325252u64;
        for (w, h) in [
            (1, 1),
            (2, 2),
            (3, 5),
            (7, 4),
            (16, 9),
            (64, 48),
            (129, 65),
            (200, 150),
            (257, 131),
        ] {
            let mut r = vec![0u8; w * h * 3];
            let mut d = vec![0u8; w * h * 3];
            for p in r.iter_mut().chain(d.iter_mut()) {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                *p = s as u8;
            }
            let streamed = ssim_streamed(&r, &d, w, h) / 3.0;
            let mut staged = 0.0;
            for channel in 0..3 {
                let a: Vec<f64> = r
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .map(|p| f64::from(p[channel]) / 255.0)
                    .collect();
                let b: Vec<f64> = d
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .map(|p| f64::from(p[channel]) / 255.0)
                    .collect();
                staged += window_score(&a, &b, w, h);
            }
            let staged = staged / 3.0;
            assert!(
                (streamed - staged).abs() < 1e-12,
                "{w}x{h}: streamed {streamed} vs staged {staged}"
            );
        }
    }

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
