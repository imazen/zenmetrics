//! WS-HV: Cheraaqee et al., IEEE TCSVT 32(2), 566–578 (2022),
//! DOI 10.1109/TCSVT.2021.3067627. Independent implementation of three-scale
//! horizontal/vertical Haar similarities, diagonal-detail weighted pooling.
//! The author notebook at IVP-Lab/WS_HV is an external validation oracle only.
//! Uses OpenCV-compatible 15-bit RGB-to-gray and cubic (A=-0.75) interpolation.
#[derive(Clone)]
struct Image {
    v: Vec<f64>,
    w: usize,
    h: usize,
}
struct Bands {
    ll: Image,
    horizontal: Image,
    vertical: Image,
    diagonal: Image,
}
fn haar(a: &Image) -> Bands {
    let w = a.w.div_ceil(2);
    let h = a.h.div_ceil(2);
    let mut values: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::with_capacity(w * h));
    let c = std::f64::consts::FRAC_1_SQRT_2;
    for y in 0..h {
        for x in 0..w {
            let x0 = 2 * x;
            let x1 = (x0 + 1).min(a.w - 1);
            let y0 = 2 * y;
            let y1 = (y0 + 1).min(a.h - 1);
            let lo0 = (a.v[y0 * a.w + x0] + a.v[y0 * a.w + x1]) * c;
            let lo1 = (a.v[y1 * a.w + x0] + a.v[y1 * a.w + x1]) * c;
            let hi0 = (a.v[y0 * a.w + x0] - a.v[y0 * a.w + x1]) * c;
            let hi1 = (a.v[y1 * a.w + x0] - a.v[y1 * a.w + x1]) * c;
            for (out, v) in values.iter_mut().zip([
                (lo0 + lo1) * c,
                (lo0 - lo1) * c,
                (hi0 + hi1) * c,
                (hi0 - hi1) * c,
            ]) {
                out.push(v);
            }
        }
    }
    let [ll, horizontal, vertical, diagonal] = values.map(|v| Image { v, w, h });
    Bands {
        ll,
        horizontal,
        vertical,
        diagonal,
    }
}
fn cubic(t: f32) -> [f32; 4] {
    let a = -0.75;
    let p = ((a * (t + 1.0) - 5.0 * a) * (t + 1.0) + 8.0 * a) * (t + 1.0) - 4.0 * a;
    let q = ((a + 2.0) * t - (a + 3.0)) * t * t + 1.0;
    let u = 1.0 - t;
    let r = ((a + 2.0) * u - (a + 3.0)) * u * u + 1.0;
    [p, q, r, 1.0 - p - q - r]
}
fn resize(a: &Image, w: usize, h: usize, is_cubic: bool) -> Image {
    let mut out = vec![0.0; w * h];
    // The source coordinate and interpolation weights depend on one axis
    // alone, so compute them once per column and once per row instead of
    // per pixel. Values and their order are unchanged.
    let columns: Vec<(isize, [f32; 4])> = (0..w)
        .map(|x| {
            let xx = ((x as f64 + 0.5) * a.w as f64 / w as f64 - 0.5) as f32;
            let sx = xx.floor() as isize;
            let tx = xx - sx as f32;
            let wx = if is_cubic {
                cubic(tx)
            } else {
                [0.0, 1.0 - tx, tx, 0.0]
            };
            (sx, wx)
        })
        .collect();
    let rows: Vec<(isize, [f32; 4])> = (0..h)
        .map(|y| {
            let yy = ((y as f64 + 0.5) * a.h as f64 / h as f64 - 0.5) as f32;
            let sy = yy.floor() as isize;
            let ty = yy - sy as f32;
            let wy = if is_cubic {
                cubic(ty)
            } else {
                [0.0, 1.0 - ty, ty, 0.0]
            };
            (sy, wy)
        })
        .collect();
    for y in 0..h {
        let (sy, wy) = rows[y];
        for x in 0..w {
            let (sx, wx) = columns[x];
            let mut value = 0.0;
            for (j, &ky) in wy.iter().enumerate() {
                let iy = (sy + j as isize - 1).clamp(0, a.h as isize - 1) as usize;
                let mut row = 0.0;
                for (i, &kx) in wx.iter().enumerate() {
                    let ix = (sx + i as isize - 1).clamp(0, a.w as isize - 1) as usize;
                    row += a.v[iy * a.w + ix] * f64::from(kx);
                }
                value += row * f64::from(ky);
            }
            out[y * w + x] = value;
        }
    }
    Image { v: out, w, h }
}
fn up(a: &Image) -> Image {
    resize(a, a.w * 2, a.h * 2, true)
}
fn ingress(a: &[u8], w: usize, h: usize, stride: usize, swap: bool) -> Result<Image, &'static str> {
    let need = h
        .checked_sub(1)
        .and_then(|n| n.checked_mul(stride))
        .and_then(|n| n.checked_add(w.checked_mul(3)?));
    if w == 0 || h == 0 || stride < w.saturating_mul(3) || need.is_none_or(|n| a.len() < n) {
        return Err("WS-HV: invalid RGB geometry");
    }
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        for p in a[y * stride..y * stride + w * 3].as_chunks::<3>().0 {
            let (r, b) = if swap { (p[2], p[0]) } else { (p[0], p[2]) };
            out.push(f64::from(
                (u32::from(r) * 9798 + u32::from(p[1]) * 19235 + u32::from(b) * 3735 + 16384) >> 15,
            ));
        }
    }
    Ok(Image { v: out, w, h })
}
fn details(a: &Image) -> ([Image; 6], Image) {
    let first = haar(a);
    let second = haar(&first.ll);
    let third = haar(&second.ll);
    let (w, h) = (first.diagonal.w, first.diagonal.h);
    let h3 = up(&up(&third.horizontal));
    let v3 = up(&up(&third.vertical));
    let h3 = resize(&h3, w, h, false);
    let v3 = resize(&v3, w, h, false);
    (
        [
            first.horizontal,
            first.vertical,
            up(&second.horizontal),
            up(&second.vertical),
            h3,
            v3,
        ],
        up(&second.diagonal),
    )
}
pub(super) fn score(
    r: &[u8],
    d: &[u8],
    w: usize,
    h: usize,
    rs: usize,
    ds: usize,
) -> Result<f64, &'static str> {
    score_order(r, d, w, h, rs, ds, false)
}
// The alternate order exists only to measure the original notebook's BGR/RGB
// boundary. The CLI always accepts RGB and uses the selected convention above.
pub(super) fn score_order(
    r: &[u8],
    d: &[u8],
    w: usize,
    h: usize,
    rs: usize,
    ds: usize,
    swap: bool,
) -> Result<f64, &'static str> {
    let (a, sa) = details(&ingress(r, w, h, rs, swap)?);
    let (b, sb) = details(&ingress(d, w, h, ds, swap)?);
    let mw = a
        .iter()
        .chain(&b)
        .chain([&sa, &sb])
        .map(|x| x.w)
        .min()
        .unwrap();
    let mh = a
        .iter()
        .chain(&b)
        .chain([&sa, &sb])
        .map(|x| x.h)
        .min()
        .unwrap();
    let f = ((w.max(h) as f64 / 512.0 + 5e-16).round_ties_even() as i32).min(3);
    let t = 10.0 * 2.0f64.powi(f);
    let constants = [3.0 * t, 3.0 * t, t, t, 2.0 * t, 2.0 * t];
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for x in 0..mw {
        let mut column = 0.0;
        let mut weight_col = 0.0;
        for y in 0..mh {
            let mut similarity = 1.0;
            for c in 0..6 {
                let u = a[c].v[y * a[c].w + x].abs();
                let v = b[c].v[y * b[c].w + x].abs();
                similarity *= (2.0 * u * v + constants[c]) / (u * u + v * v + constants[c]);
            }
            let weight = sa.v[y * sa.w + x].abs().max(sb.v[y * sb.w + x].abs());
            column += similarity * weight;
            weight_col += weight;
        }
        numerator += column;
        denominator += weight_col;
    }
    // The original metric assigns zero when the diagonal saliency is zero,
    // including an identical perfectly flat pair.
    Ok(if numerator == 0.0 {
        0.0
    } else {
        numerator / denominator
    })
}
#[cfg(test)]
mod tests {
    use super::*;

    /// The pre-change resize: the weights are recomputed for every pixel.
    fn resize_ordered_ref(a: &Image, w: usize, h: usize, is_cubic: bool) -> Image {
        let mut out = vec![0.0; w * h];
        for y in 0..h {
            let yy = ((y as f64 + 0.5) * a.h as f64 / h as f64 - 0.5) as f32;
            let sy = yy.floor() as isize;
            let ty = yy - sy as f32;
            let wy = if is_cubic {
                cubic(ty)
            } else {
                [0.0, 1.0 - ty, ty, 0.0]
            };
            for x in 0..w {
                let xx = ((x as f64 + 0.5) * a.w as f64 / w as f64 - 0.5) as f32;
                let sx = xx.floor() as isize;
                let tx = xx - sx as f32;
                let wx = if is_cubic {
                    cubic(tx)
                } else {
                    [0.0, 1.0 - tx, tx, 0.0]
                };
                let mut value = 0.0;
                for (j, &ky) in wy.iter().enumerate() {
                    let iy = (sy + j as isize - 1).clamp(0, a.h as isize - 1) as usize;
                    let mut row = 0.0;
                    for (i, &kx) in wx.iter().enumerate() {
                        let ix = (sx + i as isize - 1).clamp(0, a.w as isize - 1) as usize;
                        row += a.v[iy * a.w + ix] * f64::from(kx);
                    }
                    value += row * f64::from(ky);
                }
                out[y * w + x] = value;
            }
        }
        Image { v: out, w, h }
    }

    #[test]
    fn resize_precompute_matches_per_pixel_reference() {
        for &(aw, ah) in &[(3usize, 5usize), (8, 8), (37, 41)] {
            let src = Image {
                v: (0..aw * ah)
                    .map(|i| ((i * 13) % 97) as f64 * 0.25)
                    .collect(),
                w: aw,
                h: ah,
            };
            for &(w, h) in &[(aw * 2, ah * 2), (aw / 2 + 1, ah / 2 + 1), (23, 19)] {
                for is_cubic in [false, true] {
                    assert_eq!(
                        resize(&src, w, h, is_cubic).v,
                        resize_ordered_ref(&src, w, h, is_cubic).v,
                        "{aw}x{ah} -> {w}x{h} cubic={is_cubic}"
                    );
                }
            }
        }
    }
    #[test]
    fn known_haar_coefficients_and_constant_convention() {
        let a = Image {
            v: vec![1.0, 2.0, 3.0, 4.0],
            w: 2,
            h: 2,
        };
        let b = haar(&a);
        for (got, want) in [
            b.ll.v[0],
            b.horizontal.v[0],
            b.vertical.v[0],
            b.diagonal.v[0],
        ]
        .into_iter()
        .zip([5.0, -2.0, -1.0, 0.0])
        {
            assert!((got - want).abs() < 1e-12);
        }
        let zero = vec![0; 17 * 19 * 3];
        assert_eq!(score(&zero, &zero, 17, 19, 51, 51).unwrap(), 0.0);
    }
    #[test]
    fn strided_odd_sizes_match() {
        let (w, h) = (37, 41);
        let r: Vec<_> = (0..w * h * 3)
            .map(|i| ((i * 37 + i / 11) % 256) as u8)
            .collect();
        let mut d = r.clone();
        for p in d.iter_mut().step_by(17) {
            *p ^= 127;
        }
        let value = score(&r, &d, w, h, w * 3, w * 3).unwrap();
        assert!(value > 0.0 && value < 1.0);
        let mut sr = vec![222; (w * 3 + 7) * h];
        let mut sd = sr.clone();
        for y in 0..h {
            sr[y * (w * 3 + 7)..y * (w * 3 + 7) + w * 3]
                .copy_from_slice(&r[y * w * 3..(y + 1) * w * 3]);
            sd[y * (w * 3 + 7)..y * (w * 3 + 7) + w * 3]
                .copy_from_slice(&d[y * w * 3..(y + 1) * w * 3]);
        }
        assert_eq!(value, score(&sr, &sd, w, h, w * 3 + 7, w * 3 + 7).unwrap());
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
        // IVP-Lab notebook e31cb50f, RGB ingress, cv2 5.0 / pywt 1.10.
        // 1e-15 allows a few f64 Haar/pool rounding ULPs; f32 pooling
        // or a changed similarity constant exceeds this allowance.
        assert!((score(&r, &d, w, h, w * 3, w * 3).unwrap() - 0.7486863524749465).abs() < 1e-15);
    }
}
