//! Separable image resampling matching `torchvision.transforms.functional.resize`
//! (and `torch.nn.functional.interpolate`) exactly, for the model conventions that
//! bake a loader transform into the score (CKDN's AIC-4 column).
//!
//! - [`resize_nearest`]: torch `InterpolationMode.NEAREST` — `src = floor(i·in/out)`
//!   per axis (the legacy `nearest`, not `nearest_exact`).
//! - [`resize_bicubic_aa`]: torch `BICUBIC` + `antialias=True` — Catmull-Rom cubic
//!   (a = −0.5), centre `(i + 0.5)·in/out − 0.5`, support `2·max(1, in/out)`,
//!   weights renormalised over the clamped window. Verified bit-exact against
//!   `F.interpolate` impulse responses (a = −0.5 reproduces torch's weights to
//!   0.0; a = −0.75 — torch's own non-antialiased bicubic — does not.
//!   Pillow's `bicubic_filter` in `libImaging/Resample.c` is also a = −0.5).
//! - [`resize_bicubic_cv2_u8`]: OpenCV `cv2.resize` INTER_CUBIC on CV_8U.
//!
//! Licence notes: the PIL/Pillow-derived fixed-point resampling below
//! (`pil_1d`, `resample_u8_axis`) reproduces `libImaging/Resample.c`
//! conventions — Pillow's MIT-CMU licence is in ../LICENSE-pillow.
//! `resize_bicubic_cv2_u8` reproduces OpenCV `resize.cpp` conventions —
//! OpenCV's Apache-2.0 licence is in ../LICENSE-opencv.

use crate::par;

// A scoped launch is worthwhile only for sufficiently large passes. The
// estimate counts actual scalar MACs (or copied elements for crop/nearest).
fn resize_threads(work: usize) -> usize {
    if work < 2 * 1024 * 1024 {
        1
    } else {
        par::threads_for(work)
    }
}

/// One 1-D resampling entry: the source index a kernel window starts at and its
/// normalised weights.
struct Tap1D {
    first: usize,
    weights: Vec<f32>,
}

/// Bicubic-antialias weights for one axis: `in_size` → `out_size`.
/// `weight(j) = cubic((j − centre) / max(1, in/out))`, renormalised over
/// `j ∈ [⌈c−sup⌉, ⌊c+sup⌋] ∩ [0, in−1]` with `sup = 2·max(1, in/out)`.
fn bicubic_aa_1d(in_size: usize, out_size: usize) -> Vec<Tap1D> {
    let scale = in_size as f64 / out_size as f64;
    let down = scale.max(1.0);
    let support = 2.0 * down;
    // Catmull-Rom cubic, a = −0.5: (a+2)x³−(a+3)x²+1 / a(x³−5x²+8x−4).
    let cubic = |x: f64| {
        let x = x.abs();
        if x <= 1.0 {
            1.5 * x * x * x - 2.5 * x * x + 1.0
        } else if x < 2.0 {
            -0.5 * (x * x * x - 5.0 * x * x + 8.0 * x - 4.0)
        } else {
            0.0
        }
    };
    (0..out_size)
        .map(|i| {
            let centre = (i as f64 + 0.5) * scale - 0.5;
            let lo = (centre - support).ceil().max(0.0) as usize;
            let hi = (centre + support).floor().min(in_size as f64 - 1.0) as usize;
            let mut weights: Vec<f32> = (lo..=hi)
                .map(|j| cubic((j as f64 - centre) / down) as f32)
                .collect();
            let sum: f64 = weights.iter().map(|&w| f64::from(w)).sum();
            for w in &mut weights {
                *w = (*w as f64 / sum) as f32;
            }
            Tap1D { first: lo, weights }
        })
        .collect()
}

/// Fixed-point 1-D resampling entry for the PIL u8 path: start index and
/// `PRECISION_BITS`-scaled weights.
struct Tap1DFix {
    first: usize,
    /// `w·2²²` as `i32` (Pillow's `(int)(±0.5 + w·(1<<22))`).
    weights: Vec<i32>,
}

/// PIL/Pillow `Image.resize` coefficients for one axis
/// (`libImaging/Resample.c` `precompute_coeffs`; Pillow MIT-CMU licence
/// in ../LICENSE-pillow): window bounds are
/// `trunc(center ± support + 0.5)` — C `(int)` rounding — not torch's
/// ceil/floor, and the weights are quantised to `i32` fixed point at `2⁻²²`
/// with round-half-away. `support` is the filter's base support and
/// `kernel` its function of the scaled distance.
fn pil_1d(
    in_size: usize,
    out_size: usize,
    support: f64,
    kernel: impl Fn(f64) -> f64,
) -> Vec<Tap1DFix> {
    const PRECISION_BITS: u32 = 22;
    let scale = in_size as f64 / out_size as f64;
    let down = scale.max(1.0);
    let support = support * down;
    (0..out_size)
        .map(|i| {
            let centre = (i as f64 + 0.5) * scale;
            // C `(int)(center - support + 0.5)` truncates toward zero.
            let lo = ((centre - support + 0.5) as i64).max(0) as usize;
            let hi = (((centre + support + 0.5) as i64) as usize).min(in_size);
            let mut ws: Vec<f64> = (lo..hi)
                .map(|j| kernel((j as f64 + 0.5 - centre) / down))
                .collect();
            let sum: f64 = ws.iter().sum();
            if sum != 0.0 {
                for w in &mut ws {
                    *w /= sum;
                }
            }
            let q = (1i64 << PRECISION_BITS) as f64;
            Tap1DFix {
                first: lo,
                weights: ws
                    .iter()
                    .map(|&w| {
                        if w > 0.0 {
                            (0.5 + w * q) as i32
                        } else {
                            (-0.5 + w * q) as i32
                        }
                    })
                    .collect(),
            }
        })
        .collect()
}

/// Catmull-Rom cubic (a = −0.5), support 2 — `bicubic_filter` in Resample.c.
fn bicubic_pil_1d(in_size: usize, out_size: usize) -> Vec<Tap1DFix> {
    pil_1d(in_size, out_size, 2.0, |x| {
        let x = x.abs();
        if x <= 1.0 {
            (1.5 * x - 2.5) * x * x + 1.0
        } else if x < 2.0 {
            -0.5 * ((x - 5.0) * x + 8.0) * x + 2.0
        } else {
            0.0
        }
    })
}

/// The tent kernel `1 − |x|`, support 1 — `bilinear_filter` in Resample.c.
fn bilinear_pil_1d(in_size: usize, out_size: usize) -> Vec<Tap1DFix> {
    pil_1d(in_size, out_size, 1.0, |x| (1.0 - x.abs()).max(0.0))
}

/// One fixed-point pass over a u8 plane: `clip8((2²¹ + Σ px·k) >> 22)`.
/// `out[y,i]`, `tmp[y,i]` are planar `[c, n, m]`; `axis` 0 resamples rows,
/// 1 resamples columns.
fn resample_u8_axis(
    img: &[u8],
    c: usize,
    h: usize,
    w: usize,
    wts: &[Tap1DFix],
    axis: u8,
) -> Vec<u8> {
    if c == 0 || h == 0 || w == 0 || wts.is_empty() {
        return Vec::new();
    }
    let (oh, ow) = if axis == 0 {
        (wts.len(), w)
    } else {
        (h, wts.len())
    };
    let mut out = vec![0u8; c * oh * ow];
    let macs =
        c * if axis == 0 { ow } else { oh } * wts.iter().map(|t| t.weights.len()).sum::<usize>();
    par::for_each_row(&mut out, ow, resize_threads(macs), |i, drow| {
        let (ch, r) = (i / oh, i % oh);
        let inp = &img[ch * h * w..(ch + 1) * h * w];
        if axis == 1 {
            let row = &inp[r * w..];
            for (x, t) in wts.iter().enumerate() {
                let mut ss = 1i64 << 21;
                for (j, &k) in t.weights.iter().enumerate() {
                    ss += row[t.first + j] as i64 * k as i64;
                }
                drow[x] = (ss >> 22).clamp(0, 255) as u8;
            }
        } else {
            let t = &wts[r];
            for x in 0..w {
                let mut ss = 1i64 << 21;
                for (j, &k) in t.weights.iter().enumerate() {
                    ss += inp[(t.first + j) * w + x] as i64 * k as i64;
                }
                drow[x] = (ss >> 22).clamp(0, 255) as u8;
            }
        }
    });
    out
}

/// Planar u8 `[c, h, w]` → `[c, oh, ow]`, bit-exact with Pillow's
/// `Image.resize((ow, oh), BICUBIC)` on an RGB image: separable
/// fixed-point resample with a u8 intermediate between the horizontal and
/// vertical passes (`Resample.c`: `PRECISION_BITS = 22`, `2²¹` rounding
/// bias, `clip8` per pass).
pub(crate) fn resize_bicubic_pil_u8(
    img: &[u8],
    c: usize,
    h: usize,
    w: usize,
    oh: usize,
    ow: usize,
) -> Vec<u8> {
    if c == 0 || h == 0 || w == 0 || oh == 0 || ow == 0 {
        return Vec::new();
    }
    let wx = bicubic_pil_1d(w, ow);
    let tmp = resample_u8_axis(img, c, h, w, &wx, 1);
    let wy = bicubic_pil_1d(h, oh);
    resample_u8_axis(&tmp, c, h, ow, &wy, 0)
}

/// Planar u8 `[c, h, w]` → `[c, oh, ow]`, bit-exact with Pillow's
/// `Image.resize((ow, oh), BILINEAR)` — same separable fixed-point pass as
/// [`resize_bicubic_pil_u8`] with the tent kernel. torchvision's
/// `Resize((oh, ow))` on a PIL image is this call.
pub(crate) fn resize_bilinear_pil_u8(
    img: &[u8],
    c: usize,
    h: usize,
    w: usize,
    oh: usize,
    ow: usize,
) -> Vec<u8> {
    if c == 0 || h == 0 || w == 0 || oh == 0 || ow == 0 {
        return Vec::new();
    }
    let wx = bilinear_pil_1d(w, ow);
    let tmp = resample_u8_axis(img, c, h, w, &wx, 1);
    let wy = bilinear_pil_1d(h, oh);
    resample_u8_axis(&tmp, c, h, ow, &wy, 0)
}

/// OpenCV `cv2.resize((ow, oh), INTER_CUBIC)` on a u8 image, planar
/// `[c, h, w]` → `[c, oh, ow]`.
///
/// OpenCV `cv2.resize(..., INTER_CUBIC)` on CV_8U — the `resize.cpp`
/// `resizeGeneric_<HResizeCubic<uchar,int,short>,
/// VResizeCubic<uchar,int,short,FixedPtCast<int,uchar,22>,
/// VResizeCubicVec_32s8u>>` scalar path (OpenCV licence in
/// LICENSE-opencv):
///
/// - per axis `fx = f32((dx + 0.5)·(in/out as f64) − 0.5)`,
///   `sx = floor(fx)`, frac `fx − sx` (f32);
/// - `interpolateCubic` f32 Horner coefficients (`A = −0.75`,
///   `c3 = 1 − c0 − c1 − c2`), quantised `rint(c·2048)` → i16;
/// - taps `sx−1 .. sx+2`, border-replicate clamped;
/// - horizontal pass `mid[y][x] = Σ_k u8·alpha_k` in i32 (frac 11);
/// - vertical pass is the x86 SIMD `VResizeCubicVec_32s8u` f32 chain,
///   unfused at the pinned build's SSE baseline:
///   `rint_even(m0·b0 + (m1·b1 + (m2·b2 + m3·b3)))` with
///   `b_k = f32(beta_k)·2⁻²²`, packed with u8 saturation. (At
///   `ow = 224` every lane width divides it, so no integer
///   `FixedPtCast` tail ever runs.)
///
/// Measured: bitwise-identical to `cv2.resize` u8 with
/// `OPENCV_IPP=disabled` on 25+ random and structured images, and
/// cv2 4.12 ≡ cv2 5.0 bitwise — the wheels' IPP build intercepts u8
/// resize instead and differs on ~4% of pixels; the scalar path is
/// the documented, version-stable convention.
pub(crate) fn resize_bicubic_cv2_u8(
    img: &[u8],
    c: usize,
    h: usize,
    w: usize,
    oh: usize,
    ow: usize,
) -> Vec<u8> {
    if c == 0 || h == 0 || w == 0 || oh == 0 || ow == 0 {
        return Vec::new();
    }
    /// Per-axis taps: border-clamped indices + quantised i16 coeffs.
    fn cv2_cubic_1d(n_in: usize, n_out: usize) -> Vec<([usize; 4], [i16; 4])> {
        let scale = n_in as f64 / n_out as f64;
        (0..n_out)
            .map(|dx| {
                let fx = ((dx as f64 + 0.5) * scale - 0.5) as f32;
                let sx = fx.floor() as i64;
                let x = fx - sx as f32;
                let a = -0.75f32;
                let c0 = ((a * (x + 1.0) - 5.0 * a) * (x + 1.0) + 8.0 * a) * (x + 1.0) - 4.0 * a;
                let c1 = ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
                let c2 = ((a + 2.0) * (1.0 - x) - (a + 3.0)) * (1.0 - x) * (1.0 - x) + 1.0;
                let c3 = 1.0 - c0 - c1 - c2;
                let coef = [c0, c1, c2, c3].map(|v| (v * 2048.0).round_ties_even() as i16);
                let idx = [-1i64, 0, 1, 2].map(|k| (sx + k).clamp(0, n_in as i64 - 1) as usize);
                (idx, coef)
            })
            .collect()
    }

    let wx = cv2_cubic_1d(w, ow);
    // Horizontal pass: integer accumulation, mid = value × 2¹¹.
    let mut tmp = vec![0i32; c * h * ow];
    par::for_each_row(&mut tmp, ow, resize_threads(c * h * ow * 4), |i, drow| {
        let (ch, y) = (i / h, i % h);
        let row = &img[ch * h * w + y * w..];
        for (dx, (idx, coef)) in wx.iter().enumerate() {
            let mut v = 0i32;
            for (&i, &k) in idx.iter().zip(coef) {
                v += i32::from(row[i]) * i32::from(k);
            }
            drow[dx] = v;
        }
    });
    let wy = cv2_cubic_1d(h, oh);
    let inv = 1.0f32 / (2048.0 * 2048.0);
    let mut out = vec![0u8; c * oh * ow];
    par::for_each_row(&mut out, ow, resize_threads(c * oh * ow * 4), |i, drow| {
        let (ch, dy) = (i / oh, i % oh);
        let inp = &tmp[ch * h * ow..];
        let (idx, coef) = &wy[dy];
        let beta = coef.map(|k| f32::from(k) * inv);
        for x in 0..ow {
            let m = |k: usize| inp[idx[k] * ow + x] as f32;
            // Unfused: the pinned cv2 5.0 build has baseline SSE only (no
            // FMA3), so `v_muladd` rounds the product and the sum apart.
            let s = m(0) * beta[0] + (m(1) * beta[1] + (m(2) * beta[2] + m(3) * beta[3]));
            drow[x] = (s.round_ties_even() as i32).clamp(0, 255) as u8;
        }
    });
    out
}

/// Planar `[c, h, w]` → `[c, oh, ow]`, torch `antialias=True` bicubic.
pub(crate) fn resize_bicubic_aa(
    img: &[f32],
    c: usize,
    h: usize,
    w: usize,
    oh: usize,
    ow: usize,
) -> Vec<f32> {
    if c == 0 || h == 0 || w == 0 || oh == 0 || ow == 0 {
        return Vec::new();
    }
    let wx = bicubic_aa_1d(w, ow);
    let tmp = resample_w(img, c, h, w, &wx);
    let wy = bicubic_aa_1d(h, oh);
    resample_h(&tmp, c, h, ow, &wy)
}

/// Planar `[c, h, w]` → `[c, oh, ow]`, torch `InterpolationMode.NEAREST`:
/// `src = floor(dst · in/out)` per axis (legacy `nearest`, not `nearest_exact`).
pub(crate) fn resize_nearest(
    img: &[f32],
    c: usize,
    h: usize,
    w: usize,
    oh: usize,
    ow: usize,
) -> Vec<f32> {
    if c == 0 || h == 0 || w == 0 || oh == 0 || ow == 0 {
        return Vec::new();
    }
    let idx = |n_in: usize, n_out: usize| {
        (0..n_out)
            .map(|i| (i * n_in / n_out).min(n_in - 1))
            .collect::<Vec<_>>()
    };
    let ys = idx(h, oh);
    let xs = idx(w, ow);
    let mut out = vec![0.0f32; c * oh * ow];
    par::for_each_row(&mut out, ow, resize_threads(c * oh * ow), |i, drow| {
        let (ch, y) = (i / oh, i % oh);
        let srow = &img[ch * h * w + ys[y] * w..ch * h * w + ys[y] * w + w];
        for (x, &sx) in xs.iter().enumerate() {
            drow[x] = srow[sx];
        }
    });
    out
}

/// One 1-D resampling entry with individually clamped source indices — for
/// torch's non-antialiased bicubic, whose four taps replicate edge pixels
/// instead of renormalising the window.
struct Tap4 {
    idx: [usize; 4],
    w: [f32; 4],
}

/// Non-antialiased bicubic weights for one axis: `in_size` → `out_size`.
/// `F.interpolate(mode='bicubic', align_corners=False)`: centre
/// `(i + 0.5)·in/out − 0.5`, four taps `⌊c⌋−1 ..= ⌊c⌋+2`, cubic a = −0.75,
/// source indices clamped to `[0, in−1]`, weights NOT renormalised.
fn bicubic_1d(in_size: usize, out_size: usize) -> Vec<Tap4> {
    let scale = in_size as f64 / out_size as f64;
    let cubic = |x: f64| {
        let x = x.abs();
        if x <= 1.0 {
            1.25 * x * x * x - 2.25 * x * x + 1.0
        } else if x < 2.0 {
            -0.75 * (x * x * x - 5.0 * x * x + 8.0 * x - 4.0)
        } else {
            0.0
        }
    };
    (0..out_size)
        .map(|i| {
            let centre = (i as f64 + 0.5) * scale - 0.5;
            let base = centre.floor() as i64;
            let mut tap = Tap4 {
                idx: [0; 4],
                w: [0.0; 4],
            };
            for k in 0..4 {
                let j = base - 1 + k as i64;
                tap.idx[k] = j.clamp(0, in_size as i64 - 1) as usize;
                tap.w[k] = cubic(j as f64 - centre) as f32;
            }
            tap
        })
        .collect()
}

/// Planar `[c, h, w]` → `[c, oh, ow]`, `F.interpolate` bicubic (a = −0.75,
/// no antialias — the resize A-FINE applies to its normalised input when
/// padding up to a multiple of 32).
pub(crate) fn resize_bicubic(
    img: &[f32],
    c: usize,
    h: usize,
    w: usize,
    oh: usize,
    ow: usize,
) -> Vec<f32> {
    if c == 0 || h == 0 || w == 0 || oh == 0 || ow == 0 {
        return Vec::new();
    }
    if h == oh && w == ow {
        return img.to_vec();
    }
    let wx = bicubic_1d(w, ow);
    let mut tmp = vec![0.0f32; c * h * ow];
    par::for_each_row(&mut tmp, ow, resize_threads(c * h * ow * 4), |i, drow| {
        let (ch, y) = (i / h, i % h);
        let srow = &img[ch * h * w + y * w..ch * h * w + (y + 1) * w];
        for (x, t) in wx.iter().enumerate() {
            drow[x] = (0..4).map(|k| t.w[k] * srow[t.idx[k]]).sum();
        }
    });
    let wy = bicubic_1d(h, oh);
    let mut out = vec![0.0f32; c * oh * ow];
    par::for_each_row(&mut out, ow, resize_threads(c * oh * ow * 4), |i, drow| {
        let (ch, y) = (i / oh, i % oh);
        let src = &tmp[ch * h * ow..(ch + 1) * h * ow];
        let t = &wy[y];
        for x in 0..ow {
            drow[x] = (0..4).map(|k| t.w[k] * src[t.idx[k] * ow + x]).sum();
        }
    });
    out
}

/// Centre-crop planar `[c, h, w]` → `[c, k, k]` (torch `CenterCrop` semantics:
/// `(in − k) / 2` floor, so an odd margin's extra pixel goes bottom/right).
pub(crate) fn center_crop(img: &[f32], c: usize, h: usize, w: usize, k: usize) -> Vec<f32> {
    if c == 0 || h == 0 || w == 0 || k == 0 {
        return Vec::new();
    }
    if (h, w) == (k, k) {
        return img.to_vec();
    }
    let y0 = (h - k) / 2;
    let x0 = (w - k) / 2;
    let mut out = vec![0.0f32; c * k * k];
    par::for_each_row(&mut out, k, resize_threads(c * k * k), |i, drow| {
        let (ch, y) = (i / k, i % k);
        drow.copy_from_slice(
            &img[ch * h * w + (y0 + y) * w + x0..ch * h * w + (y0 + y) * w + x0 + k],
        );
    });
    out
}

/// Horizontal pass of a 1-D tap table: `[c, h, w]` → `[c, h, taps.len()]`.
fn resample_w(img: &[f32], c: usize, h: usize, w: usize, taps: &[Tap1D]) -> Vec<f32> {
    if c == 0 || h == 0 || w == 0 || taps.is_empty() {
        return Vec::new();
    }
    let ow = taps.len();
    let mut out = vec![0.0f32; c * h * ow];
    let macs = c * h * taps.iter().map(|t| t.weights.len()).sum::<usize>();
    par::for_each_row(&mut out, ow, resize_threads(macs), |i, drow| {
        let (ch, y) = (i / h, i % h);
        let srow = &img[ch * h * w + y * w..ch * h * w + (y + 1) * w];
        for (x, tap) in taps.iter().enumerate() {
            drow[x] = tap
                .weights
                .iter()
                .zip(&srow[tap.first..tap.first + tap.weights.len()])
                .map(|(&k, &v)| k * v)
                .sum();
        }
    });
    out
}

/// Vertical pass of a 1-D tap table: `[c, h, w]` → `[c, taps.len(), w]`.
fn resample_h(img: &[f32], c: usize, h: usize, w: usize, taps: &[Tap1D]) -> Vec<f32> {
    if c == 0 || h == 0 || w == 0 || taps.is_empty() {
        return Vec::new();
    }
    let oh = taps.len();
    let mut out = vec![0.0f32; c * oh * w];
    let macs = c * w * taps.iter().map(|t| t.weights.len()).sum::<usize>();
    par::for_each_row(&mut out, w, resize_threads(macs), |i, drow| {
        let (ch, y) = (i / oh, i % oh);
        let src = &img[ch * h * w..(ch + 1) * h * w];
        let tap = &taps[y];
        for (k, &wt) in tap.weights.iter().enumerate() {
            let srow = &src[(tap.first + k) * w..(tap.first + k + 1) * w];
            for (d, &s) in drow.iter_mut().zip(srow) {
                *d += wt * s;
            }
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_resize_paths_are_thread_bit_identical_and_empty_safe() {
        type FloatResize = fn(&[f32], usize, usize, usize, usize, usize) -> Vec<f32>;
        type ByteResize = fn(&[u8], usize, usize, usize, usize, usize) -> Vec<u8>;
        let (h, w, oh, ow) = (385, 513, 769, 1027);
        let bytes: Vec<u8> = (0..3 * h * w)
            .map(|i| ((i * 73 + i / 7 * 19) % 256) as u8)
            .collect();
        let floats: Vec<f32> = bytes
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                if i % 13 == 0 {
                    -0.0
                } else {
                    f32::from(v) / 113.0 - 1.0
                }
            })
            .collect();
        for resize in [resize_nearest, resize_bicubic, resize_bicubic_aa] as [FloatResize; 3] {
            let (a, spawned) = par::observe_workers(Some(1), || resize(&floats, 3, h, w, oh, ow));
            assert_eq!(spawned, 0);
            let (b, parallel_spawns) =
                par::observe_workers(Some(16), || resize(&floats, 3, h, w, oh, ow));
            assert!(parallel_spawns > 0);
            assert_eq!(
                a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            for (c, h, w, oh, ow) in [
                (0, 2, 2, 3, 3),
                (3, 0, 2, 3, 3),
                (3, 2, 0, 3, 3),
                (3, 2, 2, 0, 3),
                (3, 2, 2, 3, 0),
            ] {
                assert!(resize(&[], c, h, w, oh, ow).is_empty());
            }
        }
        for resize in [
            resize_bicubic_pil_u8,
            resize_bilinear_pil_u8,
            resize_bicubic_cv2_u8,
        ] as [ByteResize; 3]
        {
            let (a, spawned) = par::observe_workers(Some(1), || resize(&bytes, 3, h, w, oh, ow));
            assert_eq!(spawned, 0);
            let (b, parallel_spawns) =
                par::observe_workers(Some(16), || resize(&bytes, 3, h, w, oh, ow));
            assert!(parallel_spawns > 0);
            assert_eq!(a, b);
            for (c, h, w, oh, ow) in [
                (0, 2, 2, 3, 3),
                (3, 0, 2, 3, 3),
                (3, 2, 0, 3, 3),
                (3, 2, 2, 0, 3),
                (3, 2, 2, 3, 0),
            ] {
                assert!(resize(&[], c, h, w, oh, ow).is_empty());
            }
        }
        let crop_input: Vec<f32> = (0..3 * 1025 * 1027)
            .map(|i| (i % 257) as f32 / 113.0)
            .collect();
        let (a, spawns) =
            par::observe_workers(Some(1), || center_crop(&crop_input, 3, 1025, 1027, 1001));
        assert_eq!(spawns, 0);
        let (b, spawns) =
            par::observe_workers(Some(16), || center_crop(&crop_input, 3, 1025, 1027, 1001));
        assert!(spawns > 0);
        assert_eq!(
            a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert!(center_crop(&[], 3, 0, 2, 0).is_empty());
    }

    #[test]
    fn small_resize_passes_stay_on_the_calling_thread() {
        for n in [64, 256] {
            let x = vec![0.125; 3 * n * n];
            let bytes = vec![31; 3 * n * n];
            let (_, spawns) = par::observe_workers(Some(16), || {
                let _ = resize_bicubic_aa(&x, 3, n, n, 329, 329);
                let _ = resize_bicubic(&x, 3, n, n, 329, 329);
                let _ = resize_nearest(&x, 3, n, n, 329, 329);
                let _ = resize_bicubic_pil_u8(&bytes, 3, n, n, 224, 224);
                let _ = resize_bilinear_pil_u8(&bytes, 3, n, n, 224, 224);
                let _ = resize_bicubic_cv2_u8(&bytes, 3, n, n, 224, 224);
            });
            assert_eq!(spawns, 0, "{n} pixel input launched helper threads");
        }
    }

    /// torch `NEAREST` floor(i·in/out), verified against cv2/torch output.
    #[test]
    fn nearest_downsamples_by_floor() {
        let img: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let out = resize_nearest(&img, 1, 1, 16, 1, 6);
        // floor(i * 16/6) = 0,2,5,8,10,13
        assert_eq!(out, vec![0.0, 2.0, 5.0, 8.0, 10.0, 13.0]);
    }

    /// Antialiased bicubic weights: impulse responses measured against
    /// `F.interpolate(mode='bicubic', antialias=True)` at in=8 → out=3.
    #[test]
    fn bicubic_aa_matches_torch_impulse_weights() {
        const TORCH_W: [[f32; 8]; 3] = [
            [
                0.32377, 0.40009, 0.26135, 0.06227, -0.025, -0.02174, -0.00074, 0.0,
            ],
            [
                -0.0271, 0.0142, 0.17462, 0.33827, 0.33827, 0.17462, 0.0142, -0.0271,
            ],
            [
                0.0, -0.00074, -0.02174, -0.025, 0.06227, 0.26135, 0.40009, 0.32377,
            ],
        ];
        let taps = bicubic_aa_1d(8, 3);
        for (o, expect) in TORCH_W.iter().enumerate() {
            let mut got = [0.0f32; 8];
            for (k, &wt) in taps[o].weights.iter().enumerate() {
                got[taps[o].first + k] = wt;
            }
            for (g, e) in got.iter().zip(expect) {
                assert!((g - e).abs() < 2e-5, "out {o}: {got:?} vs {expect:?}");
            }
        }
    }

    /// Non-antialiased bicubic upsample: outputs measured against
    /// `F.interpolate(mode='bicubic', align_corners=False)`, in=8 → out=12
    /// and in=10 → out=14 (edge taps replicate, weights unnormalised).
    #[test]
    fn bicubic_interp_matches_torch() {
        const TORCH_8_12: [f32; 12] = [
            -8.680556e-2,
            4.062499e-1,
            1.2129624,
            1.7870368,
            2.4999993,
            3.2129622,
            3.7870367,
            4.499999,
            5.2129617,
            5.7870374,
            6.5937486,
            7.086803,
        ];
        let img: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let got = resize_bicubic(&img, 1, 1, 8, 1, 12);
        // torch accumulates in f32 in its own tap order; ~ulp differences
        // (≤ 3e-6 seen) are expected — far below the metric parity budget.
        for (g, e) in got.iter().zip(TORCH_8_12) {
            assert!((g - e).abs() < 1e-5, "{got:?} vs {TORCH_8_12:?}");
        }
        const TORCH_10_14: [f32; 14] = [
            -2.0551026,
            -1.667346,
            -1.0693876,
            -0.59999996,
            -0.13061233,
            0.41224435,
            0.93061334,
            1.369387,
            1.8877549,
            2.4306109,
            2.8999996,
            3.3693874,
            3.9673464,
            4.3551006,
        ];
        let img: Vec<f32> = (0..10).map(|i| i as f32 * 0.7 - 2.0).collect();
        let got = resize_bicubic(&img, 1, 1, 10, 1, 14);
        for (g, e) in got.iter().zip(TORCH_10_14) {
            assert!((g - e).abs() < 1e-5, "{got:?} vs {TORCH_10_14:?}");
        }
    }

    /// `resize_bicubic_pil_u8` reproduces Pillow 12.3's
    /// `Image.resize(BICUBIC)` output byte-for-byte: a fixed 9x7 RGB
    /// pattern down to 5x4 (interleaved goldens generated with Pillow).
    #[test]
    fn pil_u8_matches_pillow() {
        const IN: [u8; 9 * 7 * 3] = [
            102, 179, 92, 14, 106, 71, 188, 20, 102, 121, 210, 214, 74, 202, 87, 116, 99, 103, 151,
            130, 149, 52, 1, 87, 235, 157, 37, 129, 191, 187, 20, 160, 203, 57, 21, 252, 235, 88,
            48, 218, 58, 254, 169, 255, 219, 187, 207, 14, 189, 189, 174, 189, 50, 107, 54, 243,
            63, 248, 130, 228, 50, 134, 20, 72, 166, 17, 131, 88, 59, 13, 241, 249, 8, 89, 52, 129,
            83, 91, 110, 187, 198, 171, 252, 7, 174, 34, 205, 80, 163, 49, 103, 131, 1, 253, 133,
            53, 105, 3, 53, 220, 190, 145, 217, 43, 161, 201, 189, 227, 13, 94, 47, 14, 199, 205,
            214, 251, 248, 189, 39, 212, 207, 236, 81, 110, 52, 23, 153, 216, 251, 187, 123, 236,
            40, 156, 14, 44, 64, 88, 70, 8, 87, 128, 235, 135, 215, 62, 138, 242, 80, 135, 162,
            162, 32, 122, 4, 233, 230, 249, 40, 27, 134, 200, 71, 11, 161, 32, 47, 246, 150, 61,
            215, 36, 98, 171, 103, 213, 218, 34, 192, 226, 100, 174, 205, 130, 0, 4, 217, 246, 254,
        ];
        const WANT: [u8; 4 * 5 * 3] = [
            105, 128, 87, 161, 135, 105, 89, 152, 192, 116, 119, 130, 176, 96, 148, 132, 146, 90,
            113, 181, 112, 108, 109, 92, 140, 131, 132, 163, 166, 98, 158, 143, 186, 56, 143, 95,
            145, 138, 134, 152, 108, 130, 170, 109, 102, 171, 109, 165, 107, 120, 161, 102, 176,
            148, 75, 89, 140, 127, 99, 188,
        ];
        // The golden is interleaved RGB; the kernel takes planar input.
        let mut planar = vec![0u8; IN.len()];
        for (i, px) in IN.as_chunks::<3>().0.iter().enumerate() {
            for c in 0..3 {
                planar[c * 9 * 7 + i] = px[c];
            }
        }
        let out = resize_bicubic_pil_u8(&planar, 3, 9, 7, 4, 5);
        let mut got = vec![0u8; WANT.len()];
        for i in 0..4 * 5 {
            for c in 0..3 {
                got[i * 3 + c] = out[c * 4 * 5 + i];
            }
        }
        assert_eq!(got, WANT);
    }

    /// `resize_bilinear_pil_u8` reproduces Pillow 12.x's
    /// `Image.resize(BILINEAR)` output byte-for-byte on the same 9x7 → 4x5
    /// pattern as the bicubic golden.
    #[test]
    fn pil_u8_bilinear_matches_pillow() {
        const IN: [u8; 9 * 7 * 3] = [
            153, 39, 38, 46, 191, 142, 124, 17, 80, 84, 252, 4, 91, 28, 43, 170, 81, 125, 149, 118,
            183, 143, 177, 49, 13, 69, 149, 185, 76, 225, 32, 237, 161, 118, 146, 107, 71, 194,
            236, 26, 82, 243, 68, 124, 24, 79, 200, 57, 109, 154, 69, 3, 92, 150, 118, 229, 176,
            98, 196, 124, 253, 158, 128, 211, 160, 61, 26, 241, 45, 143, 194, 23, 236, 180, 205,
            217, 125, 234, 195, 227, 132, 243, 222, 186, 34, 26, 185, 182, 140, 32, 214, 27, 129,
            255, 148, 132, 74, 231, 214, 228, 37, 60, 15, 201, 245, 117, 110, 219, 231, 68, 113,
            150, 212, 29, 67, 42, 81, 76, 161, 188, 91, 164, 224, 161, 152, 161, 41, 172, 248, 105,
            62, 5, 145, 146, 86, 248, 67, 126, 226, 29, 70, 133, 19, 253, 171, 114, 253, 80, 114,
            197, 49, 199, 242, 240, 108, 40, 192, 46, 112, 6, 125, 124, 76, 194, 79, 236, 216, 235,
            96, 126, 122, 209, 23, 140, 49, 105, 155, 79, 57, 88, 70, 91, 144, 48, 89, 204, 156,
            81, 107,
        ];
        const WANT: [u8; 4 * 5 * 3] = [
            101, 124, 69, 99, 101, 135, 75, 170, 107, 110, 134, 130, 116, 125, 190, 110, 144, 73,
            120, 162, 65, 156, 153, 157, 155, 176, 168, 173, 183, 166, 108, 105, 167, 169, 124, 70,
            160, 100, 111, 123, 121, 169, 137, 139, 201, 92, 122, 172, 136, 111, 121, 143, 72, 106,
            75, 105, 138, 155, 137, 181,
        ];
        let mut planar = vec![0u8; IN.len()];
        for (i, px) in IN.as_chunks::<3>().0.iter().enumerate() {
            for c in 0..3 {
                planar[c * 9 * 7 + i] = px[c];
            }
        }
        let out = resize_bilinear_pil_u8(&planar, 3, 9, 7, 4, 5);
        let mut got = vec![0u8; WANT.len()];
        for i in 0..4 * 5 {
            for c in 0..3 {
                got[i * 3 + c] = out[c * 4 * 5 + i];
            }
        }
        assert_eq!(got, WANT);
    }

    /// A constant image stays constant through both kernels (renormalised
    /// weights), and the crop takes the geometric centre.
    #[test]
    fn const_and_crop() {
        let img = vec![3.0f32; 3 * 10 * 12];
        let r = resize_bicubic_aa(&img, 3, 10, 12, 6, 7);
        assert!(r.iter().all(|&v| (v - 3.0).abs() < 1e-6));
        let mut t = vec![0.0f32; 5 * 5];
        t[2 * 5 + 2] = 9.0;
        let c = center_crop(&t, 1, 5, 5, 3);
        assert_eq!(c[4], 9.0);
    }
}
