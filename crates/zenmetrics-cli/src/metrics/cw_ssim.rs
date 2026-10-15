//! Complex-wavelet SSIM (Wang & Simoncelli 2005; Sampat et al. 2009).
//! Frequency-domain complex steerable pyramid: four levels, eight orientations,
//! order seven, octave raised-cosine radial transition, circular boundaries.
//! Pyramid masks follow MIT-licensed pyrtools 1.0.11 (LICENSE-pyrtools),
//! using its interpolated lookup tables rather than analytic approximations.
//! Odd dimensions use true centred DFT bin frequencies; this grid was selected
//! against the published 61-pair panel (see the benchmark receipt).
//! The coarsest oriented level is scored with a valid 7x7 window, K=0,
//! and a centred Gaussian spatial weight, sigma = uncropped band height / 4.
use super::cw_fft::{Complex, transform};
use std::f64::consts::PI;

fn interpolate(x: f64, values: &[f64], origin: f64, step: f64) -> f64 {
    let pos = ((x - origin) / step).clamp(0.0, (values.len() - 1) as f64);
    let i = (pos.floor() as usize).min(values.len() - 2);
    values[i] + (pos - i as f64) * (values[i + 1] - values[i])
}
fn radial_tables() -> (Vec<f64>, Vec<f64>) {
    let mut hi: Vec<_> = (-257..=1)
        .map(|k| ((PI * k as f64 / 512.0).cos().powi(2)).sqrt())
        .collect();
    hi[0] = hi[1];
    hi[258] = hi[257];
    let lo = hi.iter().map(|&x| (1.0 - x * x).max(0.0).sqrt()).collect();
    (hi, lo)
}
fn angular_table() -> Vec<f64> {
    let factorial = |n: u32| (1..=n).map(f64::from).product::<f64>();
    let c = (2.0f64.powi(14) * factorial(7).powi(2) / (8.0 * factorial(14))).sqrt();
    (-2049..=1025)
        .map(|k| {
            let x = PI * k as f64 / 1024.0;
            let alpha = (PI + x).rem_euclid(2.0 * PI) - PI;
            if alpha.abs() < PI / 2.0 {
                2.0 * c * x.cos().powi(7)
            } else {
                0.0
            }
        })
        .collect()
}
fn luma(rgb: &[u8], w: usize, h: usize, stride: usize) -> Result<Vec<Complex>, &'static str> {
    let need = h
        .checked_sub(1)
        .and_then(|n| n.checked_mul(stride))
        .and_then(|n| n.checked_add(w.checked_mul(3)?));
    if w < 64 || h < 64 || stride < w.saturating_mul(3) || need.is_none_or(|n| rgb.len() < n) {
        return Err("CW-SSIM: invalid RGB geometry; four levels require at least 64x64");
    }
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        for p in rgb[y * stride..y * stride + w * 3].as_chunks::<3>().0 {
            out.push(Complex::new(
                (0.299 * f64::from(p[0]) + 0.587 * f64::from(p[1]) + 0.114 * f64::from(p[2]))
                    .round_ties_even(),
                0.0,
            ));
        }
    }
    Ok(out)
}
fn shift(a: &[Complex], w: usize, h: usize, inverse: bool) -> Vec<Complex> {
    let dx = if inverse { w / 2 } else { w.div_ceil(2) };
    let dy = if inverse { h / 2 } else { h.div_ceil(2) };
    (0..w * h)
        .map(|i| a[((i / w + dy) % h) * w + (i % w + dx) % w])
        .collect()
}
fn crop<T: Copy>(a: &[T], w: usize, h: usize) -> Vec<T> {
    let nw = w.div_ceil(2);
    let nh = h.div_ceil(2);
    let sx = (w + 2) / 2 - (nw + 2) / 2;
    let sy = (h + 2) / 2 - (nh + 2) / 2;
    (0..nw * nh)
        .map(|i| a[(i / nw + sy) * w + i % nw + sx])
        .collect()
}
fn frequency(i: usize, n: usize) -> f64 {
    2.0 * (i as f64 - (n / 2) as f64) / n as f64
}
// Discard unused fine orientation bands: they cannot affect the coarse level.
fn coarse(
    mut a: Vec<Complex>,
    mut w: usize,
    mut h: usize,
) -> (Vec<Complex>, Vec<f64>, Vec<f64>, usize, usize) {
    let ow = w;
    let oh = h;
    transform(&mut a, w, h, false);
    a = shift(&a, w, h, false);
    let mut angle = Vec::with_capacity(w * h);
    let mut radius = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let xx = frequency(x, w);
            let yy = frequency(y, h);
            angle.push(yy.atan2(xx));
            radius.push((xx * xx + yy * yy).sqrt());
        }
    }
    let cy = (h + 2) / 2 - 1;
    let cx = (w + 2) / 2 - 1;
    radius[cy * w + cx] = radius[cy * w + cx - 1];
    for r in &mut radius {
        *r = r.log2();
    }
    let (hi, lo) = radial_tables();
    for (v, &r) in a.iter_mut().zip(&radius) {
        let m = interpolate(r, &lo, -257.0 / 256.0, 1.0 / 256.0);
        v.re *= m;
        v.im *= m;
    }
    for level in 0..3 {
        a = crop(&a, w, h);
        radius = crop(&radius, w, h);
        angle = crop(&angle, w, h);
        w = w.div_ceil(2);
        h = h.div_ceil(2);
        let origin = -257.0 / 256.0 - (level + 1) as f64;
        for (v, &r) in a.iter_mut().zip(&radius) {
            let m = interpolate(r, &lo, origin, 1.0 / 256.0);
            v.re *= m;
            v.im *= m;
        }
    }
    debug_assert_eq!((w, h), (ow.div_ceil(8), oh.div_ceil(8)));
    let mask: Vec<_> = radius
        .iter()
        .map(|&r| interpolate(r, &hi, -257.0 / 256.0 - 4.0, 1.0 / 256.0))
        .collect();
    (a, angle, mask, w, h)
}
fn band(
    a: &[Complex],
    angles: &[f64],
    radial: &[f64],
    w: usize,
    h: usize,
    orientation: usize,
    table: &[f64],
) -> Vec<Complex> {
    let mut a: Vec<_> = a
        .iter()
        .zip(angles)
        .zip(radial)
        .map(|((&v, &angle), &r)| {
            let m = r * interpolate(
                angle,
                table,
                -2049.0 * PI / 1024.0 + PI * orientation as f64 / 8.0,
                PI / 1024.0,
            );
            // (-i)^7 = i.
            Complex::new(-v.im * m, v.re * m)
        })
        .collect();
    a = shift(&a, w, h, true);
    transform(&mut a, w, h, true);
    a
}
// Horizontal then vertical running box sums, retaining the valid region.
fn box7(a: &[f64], w: usize, h: usize) -> Vec<f64> {
    let nw = w - 6;
    let nh = h - 6;
    let mut rows = vec![0.0; nw * h];
    for y in 0..h {
        let mut sum = a[y * w..y * w + 7].iter().sum::<f64>();
        rows[y * nw] = sum;
        for x in 1..nw {
            sum += a[y * w + x + 6] - a[y * w + x - 1];
            rows[y * nw + x] = sum;
        }
    }
    let mut out = vec![0.0; nw * nh];
    for x in 0..nw {
        let mut sum = (0..7).map(|y| rows[y * nw + x]).sum::<f64>();
        out[x] = sum;
        for y in 1..nh {
            sum += rows[(y + 6) * nw + x] - rows[(y - 1) * nw + x];
            out[y * nw + x] = sum;
        }
    }
    out
}
fn similarity(a: &[Complex], b: &[Complex], w: usize, h: usize) -> f64 {
    let real: Vec<_> = a
        .iter()
        .zip(b)
        .map(|(a, b)| a.re * b.re + a.im * b.im)
        .collect();
    let imag: Vec<_> = a
        .iter()
        .zip(b)
        .map(|(a, b)| a.im * b.re - a.re * b.im)
        .collect();
    let energy: Vec<_> = a
        .iter()
        .zip(b)
        .map(|(a, b)| a.re * a.re + a.im * a.im + b.re * b.re + b.im * b.im)
        .collect();
    let re = box7(&real, w, h);
    let im = box7(&imag, w, h);
    let en = box7(&energy, w, h);
    let nw = w - 6;
    let nh = h - 6;
    let sigma = h as f64 / 4.0;
    let mut sum = 0.0;
    let mut weights = 0.0;
    for y in 0..nh {
        for x in 0..nw {
            let xx = x as f64 - (nw - 1) as f64 / 2.0;
            let yy = y as f64 - (nh - 1) as f64 / 2.0;
            let weight = (-(xx * xx + yy * yy) / (2.0 * sigma * sigma)).exp();
            let i = y * nw + x;
            // K=0: define two exactly zero-energy windows as identical.
            let q = if en[i] == 0.0 {
                1.0
            } else {
                2.0 * (re[i] * re[i] + im[i] * im[i]).sqrt() / en[i]
            };
            sum += q * weight;
            weights += weight;
        }
    }
    sum / weights
}
pub(super) fn score(
    r: &[u8],
    d: &[u8],
    w: usize,
    h: usize,
    rs: usize,
    ds: usize,
) -> Result<f64, &'static str> {
    let a = luma(r, w, h, rs)?;
    let b = luma(d, w, h, ds)?;
    let (a, angles, radial, bw, bh) = coarse(a, w, h);
    let (b, _, _, _, _) = coarse(b, w, h);
    let table = angular_table();
    let mut total = 0.0;
    for ori in 0..8 {
        let x = band(&a, &angles, &radial, bw, bh, ori, &table);
        let y = band(&b, &angles, &radial, bw, bh, ori, &table);
        total += similarity(&x, &y, bw, bh);
    }
    Ok(total / 8.0)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn odd_grid_contains_the_actual_dft_bins() {
        assert_eq!(
            (0..5).map(|i| frequency(i, 5)).collect::<Vec<_>>(),
            vec![-0.8, -0.4, 0.0, 0.4, 0.8]
        );
        assert_eq!(
            (0..4).map(|i| frequency(i, 4)).collect::<Vec<_>>(),
            vec![-1.0, -0.5, 0.0, 0.5]
        );
    }
    #[test]
    fn coefficient_phase_invariance_and_amplitude_ratio() {
        let (w, h) = (17, 19);
        let a: Vec<_> = (0..w * h)
            .map(|i| Complex::new((i % 13 + 1) as f64, (i % 7) as f64))
            .collect();
        // A global quarter turn changes every complex coefficient's phase,
        // while preserving the local absolute correlation exactly.
        let rotated: Vec<_> = a.iter().map(|v| Complex::new(-v.im, v.re)).collect();
        assert!((similarity(&a, &rotated, w, h) - 1.0).abs() < 1e-12);
        let half: Vec<_> = a
            .iter()
            .map(|v| Complex::new(v.re * 0.5, v.im * 0.5))
            .collect();
        // 2a/(1+a²) = 0.8 for a = 0.5, for every nonzero-energy window.
        assert!((similarity(&a, &half, w, h) - 0.8).abs() < 1e-12);
    }

    #[test]
    fn box_sum_matches_independent_windows() {
        let (w, h) = (17, 19);
        let a: Vec<_> = (0..w * h).map(|i| (i % 31) as f64).collect();
        let out = box7(&a, w, h);
        for y in 0..h - 6 {
            for x in 0..w - 6 {
                let expected = (y..y + 7)
                    .flat_map(|yy| (x..x + 7).map(move |xx| yy * w + xx))
                    .map(|i| a[i])
                    .sum::<f64>();
                assert_eq!(out[y * (w - 6) + x], expected);
            }
        }
    }
    #[test]
    fn identity_and_strided_odd_geometry() {
        let (w, h) = (73, 65);
        let r: Vec<_> = (0..w * h * 3)
            .map(|i| ((i * 37 + i / 11) % 256) as u8)
            .collect();
        let mut d = r.clone();
        for i in (0..d.len()).step_by(17) {
            d[i] ^= 63;
        }
        assert!((score(&r, &r, w, h, w * 3, w * 3).unwrap() - 1.0).abs() < 1e-12);
        let expected = score(&r, &d, w, h, w * 3, w * 3).unwrap();
        assert!(expected < 1.0);
        let mut sr = vec![222; (w * 3 + 7) * h];
        let mut sd = sr.clone();
        for y in 0..h {
            sr[y * (w * 3 + 7)..y * (w * 3 + 7) + w * 3]
                .copy_from_slice(&r[y * w * 3..(y + 1) * w * 3]);
            sd[y * (w * 3 + 7)..y * (w * 3 + 7) + w * 3]
                .copy_from_slice(&d[y * w * 3..(y + 1) * w * 3]);
        }
        assert_eq!(
            expected,
            score(&sr, &sd, w, h, w * 3 + 7, w * 3 + 7).unwrap()
        );
        assert!(score(&r, &d, 63, h, w * 3, w * 3).is_err());
        let zero = vec![0; w * h * 3];
        assert_eq!(score(&zero, &zero, w, h, w * 3, w * 3).unwrap(), 1.0);
    }
}
