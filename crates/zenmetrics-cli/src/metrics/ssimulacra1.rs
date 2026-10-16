//! Original Cloudinary SSIMULACRA (2017), ported from Apache-2.0 C++
//! at 1fa99a464b571b31311938c7fb6a773407454312. See NOTICE-ssimulacra1.
//! RGB8 sRGB, linear RGB -> normalized Lab, up to six area-resized scales.
//! Preserves asymmetric edge, worst row/column and worst 4x4-block pooling.
#![allow(clippy::excessive_precision)]
const SCALE: [[f32; 6]; 3] = [
    [0.0448, 0.2856, 0.3001, 0.2363, 0.1333, 0.1],
    [0.015, 0.0448, 0.2856, 0.3001, 0.3363, 0.25],
    [0.015, 0.0448, 0.2856, 0.3001, 0.3363, 0.25],
];
const WORST: [[f32; 6]; 3] = [
    [0.2, 0.3, 0.25, 0.2, 0.12, 0.05],
    [0.01, 0.05, 0.2, 0.3, 0.35, 0.35],
    [0.01, 0.05, 0.2, 0.3, 0.35, 0.35],
];
fn ingress(rgb: &[u8], w: usize, h: usize, stride: usize) -> Result<[Vec<f32>; 3], &'static str> {
    let needed = h
        .checked_sub(1)
        .and_then(|n| n.checked_mul(stride))
        .and_then(|n| n.checked_add(w.checked_mul(3)?));
    if w < 8 || h < 8 || stride < w.saturating_mul(3) || needed.is_none_or(|n| rgb.len() < n) {
        return Err("SSIMULACRA1: invalid RGB geometry; requires at least 8x8");
    }
    let lut: [f32; 256] = std::array::from_fn(|i| {
        let c = f64::from(i as f32 / 255.0);
        if c <= 0.04045 {
            (c / 12.92) as f32
        } else {
            ((c + 0.055) / 1.055).powf(2.4) as f32
        }
    });
    let mut out = std::array::from_fn(|_| Vec::with_capacity(w * h));
    for y in 0..h {
        for p in rgb[y * stride..y * stride + w * 3].as_chunks::<3>().0 {
            let [r, g, b] = p.map(|v| lut[usize::from(v)]);
            let f = |v: f32| {
                if v > 0.00885645167903563081 {
                    v.powf(1.0 / 3.0) - 0.13793103448275862068
                } else {
                    7.78703703703703703703 * v
                }
            };
            let x = f(r * 0.43393624408206207259
                + g * 0.37619779063650710152
                + b * 0.18983429773803261441);
            let y = f(r * 0.2126729 + g * 0.7151522 + b * 0.0721750);
            let z = f(r * 0.01775381083562901744
                + g * 0.10945087235996326905
                + b * 0.87263921028466483011);
            for (c, v) in [
                y * 1.16,
                0.39181818181818181818 + 2.27272727272727272727 * (x - y),
                0.49045454545454545454 + 0.90909090909090909090 * (y - z),
            ]
            .into_iter()
            .enumerate()
            {
                out[c].push(v);
            }
        }
    }
    Ok(out)
}
fn reflect(mut i: isize, n: usize) -> usize {
    // BORDER_REFLECT_101 is only defined for n >= 2; with a single row or
    // column every tap maps to index 0. Without this, n = 1 makes the loop
    // oscillate between +/-i forever. `blur` is only reached with n >= 8 in
    // `score`, so this guard changes no production result.
    if n < 2 {
        return 0;
    }
    while i < 0 || i >= n as isize {
        i = if i < 0 { -i } else { 2 * n as isize - 2 - i };
    }
    i as usize
}
fn blur(a: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut kernel = [0.0f64; 11];
    let mut sum = 0.0;
    for (i, k) in kernel.iter_mut().enumerate() {
        let x = i as f64 - 5.0;
        *k = (-x * x / (2.0 * 1.5 * 1.5)).exp();
        sum += *k;
    }
    let k = kernel.map(|v| (v / sum) as f32);
    let mut tmp = vec![0.0; w * h];
    let mut out = vec![0.0; w * h];
    // OpenCV's 11-tap row filter, symmetric column filter, BORDER_REFLECT_101.
    // Interior pixels touch no reflected border, so their tap indices are
    // contiguous; only the 5-pixel edge uses `reflect`. Both paths emit the
    // same ordered `mul_add` chain.
    for y in 0..h {
        let row = &a[y * w..y * w + w];
        let trow = &mut tmp[y * w..y * w + w];
        for x in 0..w {
            let mut value;
            if x >= 5 && x + 5 < w {
                value = row[x - 5] * k[0];
                for (j, &weight) in k.iter().enumerate().skip(1) {
                    value = row[x + j - 5].mul_add(weight, value);
                }
            } else {
                value = a[y * w + reflect(x as isize - 5, w)] * k[0];
                for (j, &weight) in k.iter().enumerate().skip(1) {
                    value =
                        a[y * w + reflect(x as isize + j as isize - 5, w)].mul_add(weight, value);
                }
            }
            trow[x] = value;
        }
    }
    for y in 0..h {
        let ybase = y * w;
        for x in 0..w {
            let mut value = tmp[ybase + x] * k[5];
            if y >= 5 && y + 5 < h {
                for j in 1..=5 {
                    value = (tmp[(y - j) * w + x] + tmp[(y + j) * w + x]).mul_add(k[5 + j], value);
                }
            } else {
                for j in 1..=5 {
                    value = (tmp[reflect(y as isize - j as isize, h) * w + x]
                        + tmp[reflect(y as isize + j as isize, h) * w + x])
                        .mul_add(k[5 + j], value);
                }
            }
            out[ybase + x] = value;
        }
    }
    out
}
fn down(a: &[f32], w: usize, h: usize, f: usize) -> (Vec<f32>, usize, usize) {
    let nw = (w as f64 / f as f64).round_ties_even() as usize;
    let nh = (h as f64 / f as f64).round_ties_even() as usize;
    let mut out = Vec::with_capacity(nw * nh);
    for y in 0..nh {
        for x in 0..nw {
            let mut sum = 0.0f32;
            // OpenCV's complete 4x4 area blocks: four row partial sums.
            if f == 4 && (x + 1) * f <= w && (y + 1) * f <= h {
                for yy in y * f..(y + 1) * f {
                    let row = &a[yy * w + x * f..][..4];
                    sum += row[0] + row[1] + row[2] + row[3];
                }
                out.push(sum * (1.0 / 16.0));
                continue;
            }
            let mut n = 0;
            for yy in y * f..((y + 1) * f).min(h) {
                for xx in x * f..((x + 1) * f).min(w) {
                    sum += a[yy * w + xx];
                    n += 1;
                }
            }
            out.push(sum / n as f32);
        }
    }
    (out, nw, nh)
}
fn average(a: &[f32]) -> f64 {
    a.iter().map(|&x| f64::from(x)).sum::<f64>() / a.len() as f64
}
fn grid(a: &[f32], w: usize, h: usize) -> f64 {
    let mut rows: Vec<_> = a.chunks_exact(w).map(average).collect();
    let mut cols: Vec<_> = (0..w)
        .map(|x| (0..h).map(|y| f64::from(a[y * w + x])).sum::<f64>() / h as f64)
        .collect();
    rows.sort_by(f64::total_cmp);
    cols.sort_by(f64::total_cmp);
    rows[h / 50] + cols[w / 50]
}
pub(super) fn score(
    r: &[u8],
    d: &[u8],
    mut w: usize,
    mut h: usize,
    rs: usize,
    ds: usize,
) -> Result<f64, &'static str> {
    let mut a = ingress(r, w, h, rs)?;
    let mut b = ingress(d, w, h, ds)?;
    let mut total = 0.0;
    let mut norm = 0.0;
    for scale in 0..6 {
        if w < 8 || h < 8 {
            break;
        }
        let (mut na, mut nb): (Vec<Vec<f32>>, Vec<Vec<f32>>) = (Vec::new(), Vec::new());
        let mut nw = 0;
        let mut nh = 0;
        for c in 0..3 {
            let x = &a[c];
            let y = &b[c];
            let mx = blur(x, w, h);
            let my = blur(y, w, h);
            let xy: Vec<_> = x.iter().zip(y).map(|(&a, &b)| a * b).collect();
            let xx: Vec<_> = x.iter().map(|&a| a * a).collect();
            let yy: Vec<_> = y.iter().map(|&a| a * a).collect();
            let exy = blur(&xy, w, h);
            let exx = blur(&xx, w, h);
            let eyy = blur(&yy, w, h);
            let mut map = Vec::with_capacity(w * h);
            for i in 0..w * h {
                let mu2 = 2.0 * mx[i] * my[i];
                let sigma12 = 2.0 * exy[i] - mu2 + 0.0004;
                let numerator = (mu2 + 0.0001) * sigma12;
                let musq = mx[i] * mx[i] + my[i] * my[i];
                let denom = (musq + 0.0001) * (exx[i] + eyy[i] - musq + 0.0004);
                map.push(numerator / denom);
            }
            if scale == 0 {
                let edge: Vec<_> = (0..w * h)
                    .map(|i| 1.0 - ((y[i] - my[i]).abs() - (x[i] - mx[i]).abs()).max(0.0))
                    .collect();
                let ew = if c == 0 { 1.5 } else { 0.1 };
                let gw = if c == 0 { 1.0 } else { 0.1 };
                total += ew * average(&edge) + gw * grid(&edge, w, h) + gw * grid(&map, w, h);
                norm += ew + 4.0 * gw;
            }
            let weight = if c == 0 { 1.0 } else { 0.2 } * f64::from(SCALE[c][scale]);
            total += weight * average(&map);
            norm += weight;
            let (blocks, _, _) = down(&map, w, h, 4);
            let worst = blocks.into_iter().fold(f32::INFINITY, f32::min);
            let weight = if c == 0 { 0.1 } else { 0.005 } * f64::from(WORST[c][scale]);
            total += weight * f64::from(worst);
            norm += weight;
            let (p, xw, xh) = down(x, w, h, 2);
            na.push(p);
            nb.push(down(y, w, h, 2).0);
            nw = xw;
            nh = xh;
        }
        a = na.try_into().map_err(|_| "SSIMULACRA1 plane count")?;
        b = nb.try_into().map_err(|_| "SSIMULACRA1 plane count")?;
        w = nw;
        h = nh;
    }
    Ok((norm / total - 1.0).clamp(0.0, 1.0))
}
#[cfg(test)]
mod tests {
    use super::*;

    /// The pre-optimization `blur`, kept as an independent bit oracle.
    fn blur_ordered_ref(a: &[f32], w: usize, h: usize) -> Vec<f32> {
        let mut kernel = [0.0f64; 11];
        let mut sum = 0.0;
        for (i, k) in kernel.iter_mut().enumerate() {
            let x = i as f64 - 5.0;
            *k = (-x * x / (2.0 * 1.5 * 1.5)).exp();
            sum += *k;
        }
        let k = kernel.map(|v| (v / sum) as f32);
        let mut tmp = vec![0.0; w * h];
        let mut out = vec![0.0; w * h];
        for y in 0..h {
            for x in 0..w {
                let mut value = a[y * w + reflect(x as isize - 5, w)] * k[0];
                for (j, &weight) in k.iter().enumerate().skip(1) {
                    value =
                        a[y * w + reflect(x as isize + j as isize - 5, w)].mul_add(weight, value);
                }
                tmp[y * w + x] = value;
            }
        }
        for y in 0..h {
            for x in 0..w {
                let mut value = tmp[y * w + x] * k[5];
                for j in 1..=5 {
                    value = (tmp[reflect(y as isize - j as isize, h) * w + x]
                        + tmp[reflect(y as isize + j as isize, h) * w + x])
                        .mul_add(k[5 + j], value);
                }
                out[y * w + x] = value;
            }
        }
        out
    }

    #[test]
    fn blur_interior_fast_path_matches_ordered_reference() {
        let geoms = [
            (8, 8),
            (9, 8),
            (8, 9),
            (10, 10),
            (11, 11),
            (13, 10),
            (31, 29),
            (37, 41),
        ];
        for &(w, h) in &geoms {
            let n = w * h;
            let pat: Vec<f32> = (0..n)
                .map(|i| ((i * 37 + i / 5) % 101) as f32 * 0.01 - 0.5)
                .collect();
            assert_eq!(
                blur(&pat, w, h),
                blur_ordered_ref(&pat, w, h),
                "pattern {w}x{h}"
            );
            let ramp: Vec<f32> = (0..n).map(|i| i as f32 * 0.001).collect();
            assert_eq!(
                blur(&ramp, w, h),
                blur_ordered_ref(&ramp, w, h),
                "ramp {w}x{h}"
            );
            let consts = vec![-1.0f32; n];
            assert_eq!(
                blur(&consts, w, h),
                blur_ordered_ref(&consts, w, h),
                "const {w}x{h}"
            );
        }
    }

    #[test]
    fn stride_and_dimensions() {
        let (w, h) = (37, 41);
        let r: Vec<_> = (0..w * h * 3)
            .map(|i| ((i * 37 + i / 5) % 256) as u8)
            .collect();
        let mut d = r.clone();
        for p in d.iter_mut().step_by(7) {
            *p ^= 127;
        }
        let expected = score(&r, &d, w, h, w * 3, w * 3).unwrap();
        assert!(expected > 0.0);
        let mut sr = vec![222; (w * 3 + 7) * h];
        let mut sd = sr.clone();
        for y in 0..h {
            sr[y * (w * 3 + 7)..y * (w * 3 + 7) + w * 3]
                .copy_from_slice(&r[y * w * 3..(y + 1) * w * 3]);
            sd[y * (w * 3 + 7)..y * (w * 3 + 7) + w * 3]
                .copy_from_slice(&d[y * w * 3..(y + 1) * w * 3]);
        }
        assert_eq!(
            score(&sr, &sd, w, h, w * 3 + 7, w * 3 + 7).unwrap(),
            expected
        );
        assert!(score(&r, &d, 7, h, w * 3, w * 3).is_err());
    }
}

#[cfg(test)]
mod oracle_tests {
    use super::*;
    // Fixture and original-author values: benchmarks/aic_review_2026-10-15/pin_oracles.py.
    #[test]
    fn original_author_synthetic_pin() {
        let (w, h) = (64, 64);
        let r: Vec<u8> = (0..h)
            .flat_map(|y| {
                (0..w).flat_map(move |x| {
                    (0..3).map(move |c| ((x * 13 + y * 17 + c * 71 + (x * y) % 29) % 256) as u8)
                })
            })
            .collect();
        let mut d = r.clone();
        for p in d.iter_mut().step_by(17) {
            *p ^= 63;
        }
        // Cloudinary 1fa99a46 + OpenCV 4.10, full-precision printf.
        // This complete-block fixture has no SIMD/area tails. The 1e-12
        // allowance covers final f64 rounding, not f32 reassociation.
        assert!((score(&r, &d, w, h, w * 3, w * 3).unwrap() - 0.01399088125957948).abs() < 1e-12);
    }
}
