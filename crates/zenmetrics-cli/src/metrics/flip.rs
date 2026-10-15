//! LDR-FLIP, ported from NVlabs/flip FLIP.h at
//! d3ab02a4425eb167d8a4021b382a2bafc6e6c838 (BSD-3-Clause; LICENSE-flip).
//! Uses the reference default 0.7 m / 3840 px / 0.7 m viewing conditions.
// These are the reference f32 coefficients; shortening them changes rounding.
#![allow(clippy::excessive_precision)]
use std::f32::consts::PI;
type Pixel = [f32; 3];
const WHITE: Pixel = [0.950428545, 1.0, 1.088900371];
const INV_WHITE: Pixel = [1.052156925, 1.0, 0.918357670];

fn xyz(p: Pixel) -> Pixel {
    [
        10135552.0 / 24577794.0 * p[0]
            + 8788810.0 / 24577794.0 * p[1]
            + 4435075.0 / 24577794.0 * p[2],
        2613072.0 / 12288897.0 * p[0]
            + 8788810.0 / 12288897.0 * p[1]
            + 887015.0 / 12288897.0 * p[2],
        1425312.0 / 73733382.0 * p[0]
            + 8788810.0 / 73733382.0 * p[1]
            + 70074185.0 / 73733382.0 * p[2],
    ]
}
fn hunt_lab(p: Pixel) -> Pixel {
    let q = xyz(p);
    let delta = 6.0f32 / 29.0;
    let f: Pixel = std::array::from_fn(|i| {
        let t = q[i] * INV_WHITE[i];
        if t > delta * delta * delta {
            t.powf(1.0 / 3.0)
        } else {
            1.0 / (3.0 * delta * delta) * t + 4.0 / 29.0
        }
    });
    let l = 116.0 * f[1] - 16.0;
    [
        l,
        0.01 * l * (500.0 * (f[0] - f[1])),
        0.01 * l * (200.0 * (f[1] - f[2])),
    ]
}
fn hyab(a: Pixel, b: Pixel) -> f32 {
    (a[0] - b[0]).abs() + ((a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}
fn filtered_lab(p: [f32; 4]) -> Pixel {
    let y = (p[0] + 16.0) / 116.0;
    let q = [
        (y + p[1] / 500.0) * WHITE[0],
        y,
        (y - (p[2] + p[3]) / 200.0) * WHITE[2],
    ];
    hunt_lab([
        (3.241003275 * q[0] - 1.537398934 * q[1] - 0.498615861 * q[2]).clamp(0.0, 1.0),
        (-0.969224334 * q[0] + 1.875930071 * q[1] + 0.041554224 * q[2]).clamp(0.0, 1.0),
        (0.055639423 * q[0] - 0.204011202 * q[1] + 1.057148933 * q[2]).clamp(0.0, 1.0),
    ])
}
fn ingress(data: &[u8], w: usize, h: usize, stride: usize) -> Result<Vec<Pixel>, &'static str> {
    let need = h
        .checked_sub(1)
        .and_then(|n| n.checked_mul(stride))
        .and_then(|n| n.checked_add(w.checked_mul(3)?));
    if w == 0 || h == 0 || stride < w.saturating_mul(3) || need.is_none_or(|n| data.len() < n) {
        return Err("FLIP: invalid RGB geometry");
    }
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        for p in data[y * stride..y * stride + w * 3].as_chunks::<3>().0 {
            let q = xyz(p.map(|v| {
                let s = f32::from(v) / 255.0;
                if s <= 0.04045 {
                    s / 12.92
                } else {
                    ((s + 0.055) / 1.055).powf(2.4)
                }
            }));
            let q: Pixel = std::array::from_fn(|i| q[i] * INV_WHITE[i]);
            out.push([
                116.0 * q[1] - 16.0,
                500.0 * (q[0] - q[1]),
                200.0 * (q[1] - q[2]),
            ]);
        }
    }
    Ok(out)
}
fn spatial_filter() -> Vec<[f32; 4]> {
    let ppd = 0.7 * (3840.0 / 0.7) * (PI / 180.0);
    let radius = (3.0 * (0.04 / (2.0 * PI * PI)).sqrt() * ppd).ceil() as i32;
    let mut f = Vec::new();
    let mut sum = [0.0; 4];
    for x in -radius..=radius {
        let ix = x as f32 * (1.0 / ppd);
        let x2 = ix * ix;
        let mut v = [0.0; 4];
        for (c, (a, b)) in [(1.0, 0.0047), (1.0, 0.0053), (34.1, 0.04), (13.5, 0.025)]
            .into_iter()
            .enumerate()
        {
            let t: f32 = a * (PI / b).sqrt();
            v[c] = if c < 2 { t } else { t.sqrt() } * (-PI * PI * x2 / b).exp();
            sum[c] += v[c];
        }
        f.push(v);
    }
    let norm = [
        1.0 / sum[0],
        1.0 / sum[1],
        1.0 / (sum[2] * sum[2] + sum[3] * sum[3]).sqrt(),
        1.0 / (sum[2] * sum[2] + sum[3] * sum[3]).sqrt(),
    ];
    for v in &mut f {
        for i in 0..4 {
            v[i] *= norm[i];
        }
    }
    f
}
fn feature_filter() -> Vec<Pixel> {
    let sd = 0.5 * 0.082 * (0.7 * (3840.0 / 0.7) * (PI / 180.0));
    let radius = (3.0 * sd).ceil() as i32;
    let mut f = Vec::new();
    let mut pos = [0.0; 3];
    let mut neg = [0.0; 3];
    for x in -radius..=radius {
        let x = x as f32;
        let g = (-(x * x) / (2.0 * sd * sd)).exp();
        let v = [g, -x * g, (x * x / (sd * sd) - 1.0) * g];
        for i in 0..3 {
            if v[i] > 0.0 {
                pos[i] += v[i];
            } else {
                neg[i] -= v[i];
            }
        }
        f.push(v);
    }
    for v in &mut f {
        for i in 0..3 {
            v[i] /= if v[i] > 0.0 { pos[i] } else { neg[i] };
        }
    }
    f
}
// Every convolution extends the nearest border sample, as in the C++ reference.
fn convolve<const N: usize>(
    a: &[[f32; N]],
    w: usize,
    h: usize,
    f: &[[f32; N]],
    vertical: bool,
) -> Vec<[f32; N]> {
    let mut out = vec![[0.0; N]; w * h];
    let r = (f.len() / 2) as isize;
    for y in 0..h {
        for x in 0..w {
            let v = &mut out[y * w + x];
            for (k, t) in f.iter().enumerate() {
                let xx = if vertical {
                    x
                } else {
                    (x as isize + k as isize - r).clamp(0, w as isize - 1) as usize
                };
                let yy = if vertical {
                    (y as isize + k as isize - r).clamp(0, h as isize - 1) as usize
                } else {
                    y
                };
                for c in 0..N {
                    v[c] += t[c] * a[yy * w + xx][c];
                }
            }
        }
    }
    out
}
fn colors(a: &[Pixel], w: usize, h: usize, f: &[[f32; 4]]) -> Vec<Pixel> {
    let a: Vec<_> = a.iter().map(|p| [p[0], p[1], p[2], p[2]]).collect();
    let a = convolve(&a, w, h, f, false);
    convolve(&a, w, h, f, true)
        .into_iter()
        .map(filtered_lab)
        .collect()
}
fn features(a: &[Pixel], w: usize, h: usize, f: &[Pixel]) -> Vec<[f32; 2]> {
    let a: Vec<_> = a
        .iter()
        .map(|p| {
            let y = p[0] * (1.0 / 116.0) + 16.0 / 116.0;
            [y; 3]
        })
        .collect();
    // Horizontal channels are dx, ddx, Gaussian. Vertical channels yield
    // dx, ddx, dy and ddy independently (no derivatives are applied twice).
    let hf: Vec<_> = f.iter().map(|p| [p[1], p[2], p[0]]).collect();
    let a = convolve(&a, w, h, &hf, false);
    let a: Vec<_> = a.iter().map(|p| [p[0], p[1], p[2], p[2]]).collect();
    let vf: Vec<_> = f.iter().map(|p| [p[0], p[0], p[1], p[2]]).collect();
    convolve(&a, w, h, &vf, true)
        .into_iter()
        .map(|p| {
            [
                (p[0] * p[0] + p[2] * p[2]).sqrt(),
                (p[1] * p[1] + p[3] * p[3]).sqrt(),
            ]
        })
        .collect()
}
/// Raw error map; arithmetic deliberately follows the f32 C++ reference.
pub(super) fn error_map(
    r: &[u8],
    d: &[u8],
    w: usize,
    h: usize,
    rs: usize,
    ds: usize,
) -> Result<Vec<f32>, &'static str> {
    let r = ingress(r, w, h, rs)?;
    let d = ingress(d, w, h, ds)?;
    let sf = spatial_filter();
    let ff = feature_filter();
    let cr = colors(&r, w, h, &sf);
    let cd = colors(&d, w, h, &sf);
    let fr = features(&r, w, h, &ff);
    let fd = features(&d, w, h, &ff);
    let cmax = hyab(hunt_lab([0.0, 1.0, 0.0]), hunt_lab([0.0, 0.0, 1.0])).powf(0.7);
    let pc = 0.4 * cmax;
    Ok((0..w * h)
        .map(|i| {
            let c = hyab(cr[i], cd[i]).powf(0.7);
            let c = if c < pc {
                c * (0.95 / pc)
            } else {
                0.95 + ((c - pc) / (cmax - pc)) * (1.0 - 0.95)
            };
            let f = ((1.0 / 2.0f32.sqrt())
                * (fr[i][0] - fd[i][0]).abs().max((fr[i][1] - fd[i][1]).abs()))
            .powf(0.5);
            c.powf(1.0 - f)
        })
        .collect())
}

// flip-evaluator 1.7 and the AIC column accumulate row-major in f32.
// Later NVlabs revisions use f64; changing the reduction changes the column.
pub(super) fn mean(map: &[f32]) -> f64 {
    f64::from(map.iter().fold(0.0f32, |s, &v| s + v) / map.len() as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn identity_stride_and_border_distortion() {
        let (w, h) = (19, 17);
        let r: Vec<_> = (0..w * h * 3)
            .map(|i| ((i * 37 + i / 19) % 256) as u8)
            .collect();
        let mut d = r.clone();
        d[0] ^= 127;
        let map = error_map(&r, &d, w, h, w * 3, w * 3).unwrap();
        assert!(mean(&map) > 0.0);
        let mut rs = vec![211; (w * 3 + 5) * h];
        let mut ds = rs.clone();
        for y in 0..h {
            rs[y * (w * 3 + 5)..y * (w * 3 + 5) + w * 3]
                .copy_from_slice(&r[y * w * 3..(y + 1) * w * 3]);
            ds[y * (w * 3 + 5)..y * (w * 3 + 5) + w * 3]
                .copy_from_slice(&d[y * w * 3..(y + 1) * w * 3]);
        }
        assert_eq!(
            map,
            error_map(&rs, &ds, w, h, w * 3 + 5, w * 3 + 5).unwrap()
        );
        assert!(
            error_map(&r, &r, w, h, w * 3, w * 3)
                .unwrap()
                .iter()
                .all(|&v| v == 0.0)
        );
        assert!(error_map(&r, &d, w, h, w * 3 - 1, w * 3).is_err());
    }
}
