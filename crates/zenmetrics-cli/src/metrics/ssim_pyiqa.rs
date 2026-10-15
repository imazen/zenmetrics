//! Wang et al. SSIM (2004) and MS-SSIM (2003), with AIC2026 conventions.
//! Written from the papers and the measured column description, without pyiqa code.
//! The fixed-factor variants average complete N×N blocks starting at (0,0),
//! dropping incomplete bottom/right blocks; the site's values do not
//! establish a unique downscale kernel. Input and all accumulation are f64;
//! `f32` in the CLI name means unrounded input, not f32 arithmetic.
//! Rounded luma uses ties-to-even: on the 61-pair AIC gate it has lower mean
//! absolute error for both SSIM and MS-SSIM than half-away rounding. Both
//! measurements are retained in `benchmarks/aic_variants_2026-10-15/round2_rounding.tsv`.

use crate::decode::Rgb8Image;

pub(super) fn score(
    r: &Rgb8Image,
    d: &Rgb8Image,
    color: bool,
    multiscale: bool,
    rounded: bool,
    factor: usize,
) -> Result<f64, Box<dyn std::error::Error>> {
    if (r.width, r.height) != (d.width, d.height) {
        return Err("SSIM-pyiqa: dimension mismatch".into());
    }
    let (w, h) = (r.width as usize, r.height as usize);
    let (mut sw, mut sh) = (w / factor, h / factor);
    for scale in 0..if multiscale { 5 } else { 1 } {
        if sw < 11 || sh < 11 {
            return Err(format!("SSIM-pyiqa: scale {} is {sw}x{sh}; valid 11x11 window requires at least 11x11 (fixed factor {factor})",scale+1).into());
        }
        sw = sw.div_ceil(2);
        sh = sh.div_ceil(2);
    }
    let mut total = 0.0;
    for channel in 0..if color { 3 } else { 1 } {
        let a = plane(&r.pixels, w, h, w * 3, color, channel, rounded)?;
        let b = plane(&d.pixels, w, h, w * 3, color, channel, rounded)?;
        let (mut a, mut w, mut h) = boxdown(&a, w, h, factor);
        let (mut b, _, _) = boxdown(&b, r.width as usize, r.height as usize, factor);
        let weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333];
        let mut product = 1.0;
        for (scale, &weight) in weights
            .iter()
            .enumerate()
            .take(if multiscale { 5 } else { 1 })
        {
            let (ssim, cs) = statistics(&a, &b, w, h)?;
            if !multiscale {
                product = ssim;
                break;
            }
            product *= if scale == 4 { ssim } else { cs }.powf(weight);
            if scale < 4 {
                let (next, nw, nh) = pool2(&a, w, h);
                a = next;
                b = pool2(&b, w, h).0;
                w = nw;
                h = nh;
            }
        }
        total += product;
    }
    Ok(total / if color { 3.0 } else { 1.0 })
}

fn plane(
    data: &[u8],
    w: usize,
    h: usize,
    stride: usize,
    color: bool,
    channel: usize,
    rounded: bool,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let needed = h
        .checked_sub(1)
        .and_then(|n| n.checked_mul(stride))
        .and_then(|n| n.checked_add(w.checked_mul(3)?));
    if w == 0 || h == 0 || stride < w * 3 || needed.is_none_or(|n| data.len() < n) {
        return Err("SSIM-pyiqa: invalid RGB geometry".into());
    }
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        for px in data[y * stride..y * stride + w * 3].as_chunks::<3>().0 {
            let v = if color {
                f64::from(px[channel])
            } else {
                0.299 * f64::from(px[0]) + 0.587 * f64::from(px[1]) + 0.114 * f64::from(px[2])
            };
            out.push(if rounded { v.round_ties_even() } else { v });
        }
    }
    Ok(out)
}

fn boxdown(a: &[f64], w: usize, h: usize, f: usize) -> (Vec<f64>, usize, usize) {
    if f == 1 {
        return (a.to_vec(), w, h);
    }
    let (nw, nh) = (w / f, h / f);
    let mut out = Vec::with_capacity(nw * nh);
    for y in 0..nh {
        for x in 0..nw {
            let mut sum = 0.0;
            for ky in 0..f {
                for kx in 0..f {
                    sum += a[(y * f + ky) * w + x * f + kx];
                }
            }
            out.push(sum / (f * f) as f64);
        }
    }
    (out, nw, nh)
}

// PyTorch-style symmetric one-cell zero padding on BOTH ends of an odd
// dimension, with the first 2×2 window starting at the padded top/left.
fn pool2(a: &[f64], w: usize, h: usize) -> (Vec<f64>, usize, usize) {
    let (nw, nh) = (w.div_ceil(2), h.div_ceil(2));
    let mut out = Vec::with_capacity(nw * nh);
    for y in 0..nh {
        for x in 0..nw {
            let mut sum = 0.0;
            for dy in 0..2 {
                for dx in 0..2 {
                    let yy = (2 * y + dy) as isize - (h % 2) as isize;
                    let xx = (2 * x + dx) as isize - (w % 2) as isize;
                    if yy >= 0 && xx >= 0 {
                        sum += a[yy as usize * w + xx as usize];
                    }
                }
            }
            out.push(sum * 0.25);
        }
    }
    (out, nw, nh)
}

fn statistics(
    a: &[f64],
    b: &[f64],
    w: usize,
    h: usize,
) -> Result<(f64, f64), Box<dyn std::error::Error>> {
    if w < 11 || h < 11 {
        return Err(format!(
            "SSIM-pyiqa: valid 11x11 window requires at least 11x11 at each scale, got {w}x{h}"
        )
        .into());
    }
    let mut g = [0.0; 11];
    for (i, v) in g.iter_mut().enumerate() {
        *v = (-((i as f64 - 5.0).powi(2)) / (2.0 * 1.5 * 1.5)).exp();
    }
    let norm: f64 = g.iter().sum();
    for v in &mut g {
        *v /= norm;
    }
    let ow = w - 10;
    // Horizontal moments in an eleven-row ring; vertical pass then pooling.
    let mut ring = vec![[0.0; 5]; 11 * ow];
    let mut ssim = 0.0;
    let mut cs_sum = 0.0;
    for y in 0..h {
        for x in 0..ow {
            let mut v = [0.0; 5];
            for (k, &weight) in g.iter().enumerate() {
                let ar = a[y * w + x + k];
                let br = b[y * w + x + k];
                for (acc, m) in v.iter_mut().zip([ar, br, ar * ar, br * br, ar * br]) {
                    *acc += weight * m;
                }
            }
            ring[(y % 11) * ow + x] = v;
        }
        if y < 10 {
            continue;
        }
        for x in 0..ow {
            let mut v = [0.0; 5];
            for (k, &weight) in g.iter().enumerate() {
                for (acc, &moment) in v.iter_mut().zip(&ring[((y + 1 + k) % 11) * ow + x]) {
                    *acc += weight * moment;
                }
            }
            let [ma, mb, aa, bb, ab] = v;
            let cs = ((2.0 * (ab - ma * mb) + 58.5225) / (aa - ma * ma + bb - mb * mb + 58.5225))
                .max(0.0);
            cs_sum += cs;
            ssim += cs * (2.0 * ma * mb + 6.5025) / (ma * ma + mb * mb + 6.5025);
        }
    }
    let n = ((w - 10) * (h - 10)) as f64;
    Ok((ssim / n, cs_sum / n))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_filter_matches_direct_two_dimensional_oracle() {
        let (w, h) = (17, 15);
        let a: Vec<f64> = (0..w * h)
            .map(|i| ((i * 37 + i / 7) % 256) as f64)
            .collect();
        let b: Vec<f64> = a
            .iter()
            .enumerate()
            .map(|(i, &v)| if i % 5 == 0 { (v + 41.0) % 256.0 } else { v })
            .collect();
        let (expected_ssim, expected_cs) = direct_statistics(&a, &b, w, h);
        let (ssim, cs) = statistics(&a, &b, w, h).unwrap();
        assert!((ssim - expected_ssim).abs() < 1e-12);
        assert!((cs - expected_cs).abs() < 1e-12);
    }

    fn direct_statistics(a: &[f64], b: &[f64], w: usize, h: usize) -> (f64, f64) {
        let mut g = [0.0; 11];
        for (i, v) in g.iter_mut().enumerate() {
            *v = (-((i as f64 - 5.0).powi(2)) / 4.5).exp();
        }
        let norm: f64 = g.iter().sum();
        let (mut sum, mut csum) = (0.0, 0.0);
        for y in 0..h - 10 {
            for x in 0..w - 10 {
                let mut v = [0.0; 5];
                for (ky, &gy) in g.iter().enumerate() {
                    for (kx, &gx) in g.iter().enumerate() {
                        let weight = gy * gx / (norm * norm);
                        let ar = a[(y + ky) * w + x + kx];
                        let br = b[(y + ky) * w + x + kx];
                        for (acc, m) in v.iter_mut().zip([ar, br, ar * ar, br * br, ar * br]) {
                            *acc += weight * m;
                        }
                    }
                }
                let [ma, mb, aa, bb, ab] = v;
                let cs = ((2.0 * (ab - ma * mb) + 58.5225)
                    / (aa - ma * ma + bb - mb * mb + 58.5225))
                    .max(0.0);
                csum += cs;
                sum += cs * (2.0 * ma * mb + 6.5025) / (ma * ma + mb * mb + 6.5025);
            }
        }
        let n = ((w - 10) * (h - 10)) as f64;
        (sum / n, csum / n)
    }

    #[test]
    fn multiscale_score_matches_direct_scale_product() {
        for (w, h) in [(161, 177), (162, 176)] {
            let r = Rgb8Image {
                width: w,
                height: h,
                pixels: (0..w * h * 3)
                    .map(|i| ((i * 37 + i / 11) % 256) as u8)
                    .collect(),
            };
            let d = Rgb8Image {
                width: w,
                height: h,
                pixels: r
                    .pixels
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| if i % 7 == 0 { v.wrapping_add(43) } else { v })
                    .collect(),
            };
            let mut a = plane(
                &r.pixels,
                w as usize,
                h as usize,
                w as usize * 3,
                false,
                0,
                false,
            )
            .unwrap();
            let mut b = plane(
                &d.pixels,
                w as usize,
                h as usize,
                w as usize * 3,
                false,
                0,
                false,
            )
            .unwrap();
            let (mut sw, mut sh) = (w as usize, h as usize);
            let mut expected = 1.0;
            for (level, weight) in [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
                .into_iter()
                .enumerate()
            {
                let (ssim, cs) = direct_statistics(&a, &b, sw, sh);
                expected *= if level == 4 { ssim } else { cs }.powf(weight);
                if level < 4 {
                    // Materialize the padded image, independently of the ring/scoring path.
                    let down = |src: &[f64]| {
                        let (pw, ph) = (sw + 2 * (sw % 2), sh + 2 * (sh % 2));
                        let mut padded = vec![0.0; pw * ph];
                        for y in 0..sh {
                            for x in 0..sw {
                                padded[(y + sh % 2) * pw + x + sw % 2] = src[y * sw + x];
                            }
                        }
                        let mut dst = Vec::new();
                        for y in (0..ph - 1).step_by(2) {
                            for x in (0..pw - 1).step_by(2) {
                                dst.push(
                                    (padded[y * pw + x]
                                        + padded[y * pw + x + 1]
                                        + padded[(y + 1) * pw + x]
                                        + padded[(y + 1) * pw + x + 1])
                                        / 4.0,
                                );
                            }
                        }
                        dst
                    };
                    a = down(&a);
                    b = down(&b);
                    sw = sw.div_ceil(2);
                    sh = sh.div_ceil(2);
                }
            }
            let actual = score(&r, &d, false, true, false, 1).unwrap();
            assert!(actual > 0.0 && actual < 1.0);
            assert!(
                (actual - expected).abs() < 1e-12,
                "{w}x{h}: {actual} vs {expected}"
            );
        }
    }

    #[test]
    fn rounded_luma_uses_the_measured_ties_even_rule() {
        let rgb = [0, 0, 250];
        assert_eq!(plane(&rgb, 1, 1, 3, false, 0, false).unwrap(), vec![28.5]);
        assert_eq!(plane(&rgb, 1, 1, 3, false, 0, true).unwrap(), vec![28.0]);
    }

    #[test]
    fn odd_pool_has_top_left_zero_padding_and_fixed_divisor() {
        let a: Vec<f64> = (1..=9).map(f64::from).collect();
        assert_eq!(pool2(&a, 3, 3), (vec![0.25, 1.25, 2.75, 7.0], 2, 2));
        assert_eq!(pool2(&[1.0, 2.0, 3.0, 4.0], 2, 2), (vec![2.5], 1, 1));
    }

    #[test]
    fn stride_rounding_and_decimation() {
        let (w, h) = (23, 19);
        let packed: Vec<u8> = (0..w * h * 3).map(|i| (i * 17 % 256) as u8).collect();
        let stride = w * 3 + 13;
        let mut strided = vec![253; stride * h];
        for y in 0..h {
            strided[y * stride..y * stride + w * 3]
                .copy_from_slice(&packed[y * w * 3..(y + 1) * w * 3]);
        }
        for color in [false, true] {
            for rounded in [false, true] {
                for channel in 0..if color { 3 } else { 1 } {
                    let a = plane(&packed, w, h, w * 3, color, channel, rounded).unwrap();
                    let b = plane(&strided, w, h, stride, color, channel, rounded).unwrap();
                    assert_eq!(a, b);
                    let (ssim, cs) = statistics(&a, &b, w, h).unwrap();
                    assert_eq!(ssim, 1.0);
                    assert_eq!(cs, 1.0);
                }
            }
        }
        assert!(plane(&strided[..3], w, h, stride, false, 0, false).is_err());
        assert!(statistics(&[0.0; 100], &[0.0; 100], 10, 10).is_err());
    }

    #[test]
    fn anticorrelation_is_clipped_before_pooling() {
        let a: Vec<f64> = (0..121)
            .map(|i| if i % 2 == 0 { 0.0 } else { 255.0 })
            .collect();
        let b: Vec<f64> = a.iter().map(|v| 255.0 - v).collect();
        assert_eq!(statistics(&a, &b, 11, 11).unwrap(), (0.0, 0.0));
    }
}

#[cfg(test)]
mod boundary_tests {
    use super::*;

    #[test]
    fn fixed_box_uses_complete_blocks() {
        let a: Vec<f64> = (1..=15).map(f64::from).collect();
        assert_eq!(boxdown(&a, 5, 3, 2), (vec![4.0, 6.0], 2, 1));
    }

    #[test]
    fn multiscale_rejects_unsupported_geometry_before_scoring() {
        let image = Rgb8Image {
            pixels: vec![],
            width: 840,
            height: 944,
        };
        let error = score(&image, &image, false, true, false, 6)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("scale 5") && error.contains("11x11"),
            "{error}"
        );
    }
}
