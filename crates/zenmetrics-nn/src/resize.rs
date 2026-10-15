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

/// PIL/Pillow `Image.resize(BICUBIC)` coefficients for one axis
/// (`libImaging/Resample.c`): same Catmull-Rom kernel as
/// [`bicubic_aa_1d`], but the window bounds are `trunc(center ± support +
/// 0.5)` — C `(int)` rounding — not torch's ceil/floor, and the weights are
/// quantised to `i32` fixed point at `2⁻²²` with round-half-away.
fn bicubic_pil_1d(in_size: usize, out_size: usize) -> Vec<Tap1DFix> {
    const PRECISION_BITS: u32 = 22;
    let scale = in_size as f64 / out_size as f64;
    let down = scale.max(1.0);
    let support = 2.0 * down;
    let cubic = |x: f64| {
        let x = x.abs();
        if x <= 1.0 {
            (1.5 * x - 2.5) * x * x + 1.0
        } else if x < 2.0 {
            -0.5 * ((x - 5.0) * x + 8.0) * x + 2.0
        } else {
            0.0
        }
    };
    (0..out_size)
        .map(|i| {
            let centre = (i as f64 + 0.5) * scale;
            // C `(int)(center - support + 0.5)` truncates toward zero.
            let lo = ((centre - support + 0.5) as i64).max(0) as usize;
            let hi = (((centre + support + 0.5) as i64) as usize).min(in_size);
            let mut ws: Vec<f64> = (lo..hi)
                .map(|j| cubic((j as f64 + 0.5 - centre) / down))
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

/// PIL/Pillow `Image.resize(BILINEAR)` coefficients for one axis
/// (`libImaging/Resample.c` `bilinear_filter`): triangle kernel
/// `1 − |x|`, support `1·max(1, in/out)`; same fixed-point window bounds
/// and 2⁻²² weight quantisation as [`bicubic_pil_1d`].
fn bilinear_pil_1d(in_size: usize, out_size: usize) -> Vec<Tap1DFix> {
    const PRECISION_BITS: u32 = 22;
    let scale = in_size as f64 / out_size as f64;
    let down = scale.max(1.0);
    let support = down;
    let tri = |x: f64| (1.0 - x.abs()).max(0.0);
    (0..out_size)
        .map(|i| {
            let centre = (i as f64 + 0.5) * scale;
            let lo = ((centre - support + 0.5) as i64).max(0) as usize;
            let hi = (((centre + support + 0.5) as i64) as usize).min(in_size);
            let mut ws: Vec<f64> = (lo..hi)
                .map(|j| tri((j as f64 + 0.5 - centre) / down))
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

/// Planar u8 `[c, h, w]` → `[c, oh, ow]`, bit-exact with Pillow's
/// `Image.resize((ow, oh), BILINEAR)` — the same fixed-point separable
/// recipe as [`resize_bicubic_pil_u8`] with the triangle kernel (DMM's
/// `prepare_image_adt` calls `F.resize(img, tar_size)`, default BILINEAR).
pub(crate) fn resize_bilinear_pil_u8(
    img: &[u8],
    c: usize,
    h: usize,
    w: usize,
    oh: usize,
    ow: usize,
) -> Vec<u8> {
    let wx = bilinear_pil_1d(w, ow);
    let tmp = resample_u8_axis(img, c, h, w, &wx, 1);
    let wy = bilinear_pil_1d(h, oh);
    resample_u8_axis(&tmp, c, h, ow, &wy, 0)
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
    let (oh, ow) = if axis == 0 {
        (wts.len(), w)
    } else {
        (h, wts.len())
    };
    let mut out = vec![0u8; c * oh * ow];
    for ch in 0..c {
        let (inp, outp) = (&img[ch * h * w..(ch + 1) * h * w], &mut out[ch * oh * ow..]);
        if axis == 1 {
            for (i, t) in wts.iter().enumerate() {
                for y in 0..h {
                    let row = &inp[y * w..];
                    let mut ss = 1i64 << 21;
                    for (j, &k) in t.weights.iter().enumerate() {
                        ss += row[t.first + j] as i64 * k as i64;
                    }
                    outp[y * ow + i] = (ss >> 22).clamp(0, 255) as u8;
                }
            }
        } else {
            for (i, t) in wts.iter().enumerate() {
                for x in 0..w {
                    let mut ss = 1i64 << 21;
                    for (j, &k) in t.weights.iter().enumerate() {
                        ss += inp[(t.first + j) * w + x] as i64 * k as i64;
                    }
                    outp[i * ow + x] = (ss >> 22).clamp(0, 255) as u8;
                }
            }
        }
    }
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
    let wx = bicubic_pil_1d(w, ow);
    let tmp = resample_u8_axis(img, c, h, w, &wx, 1);
    let wy = bicubic_pil_1d(h, oh);
    resample_u8_axis(&tmp, c, h, ow, &wy, 0)
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
    let idx = |n_in: usize, n_out: usize| {
        (0..n_out)
            .map(|i| (i * n_in / n_out).min(n_in - 1))
            .collect::<Vec<_>>()
    };
    let ys = idx(h, oh);
    let xs = idx(w, ow);
    let mut out = vec![0.0f32; c * oh * ow];
    for ch in 0..c {
        let src = &img[ch * h * w..(ch + 1) * h * w];
        let dst = &mut out[ch * oh * ow..(ch + 1) * oh * ow];
        for (y, &sy) in ys.iter().enumerate() {
            let srow = &src[sy * w..sy * w + w];
            let drow = &mut dst[y * ow..(y + 1) * ow];
            for (x, &sx) in xs.iter().enumerate() {
                drow[x] = srow[sx];
            }
        }
    }
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
    if h == oh && w == ow {
        return img.to_vec();
    }
    let wx = bicubic_1d(w, ow);
    let mut tmp = vec![0.0f32; c * h * ow];
    for ch in 0..c {
        let src = &img[ch * h * w..(ch + 1) * h * w];
        let dst = &mut tmp[ch * h * ow..(ch + 1) * h * ow];
        for y in 0..h {
            let srow = &src[y * w..(y + 1) * w];
            for (x, t) in wx.iter().enumerate() {
                dst[y * ow + x] = (0..4).map(|k| t.w[k] * srow[t.idx[k]]).sum();
            }
        }
    }
    let wy = bicubic_1d(h, oh);
    let mut out = vec![0.0f32; c * oh * ow];
    for ch in 0..c {
        let src = &tmp[ch * h * ow..(ch + 1) * h * ow];
        let dst = &mut out[ch * oh * ow..(ch + 1) * oh * ow];
        for (y, t) in wy.iter().enumerate() {
            for x in 0..ow {
                dst[y * ow + x] = (0..4).map(|k| t.w[k] * src[t.idx[k] * ow + x]).sum();
            }
        }
    }
    out
}

/// Centre-crop planar `[c, h, w]` → `[c, k, k]` (torch `CenterCrop` semantics:
/// `(in − k) / 2` floor, so an odd margin's extra pixel goes bottom/right).
pub(crate) fn center_crop(img: &[f32], c: usize, h: usize, w: usize, k: usize) -> Vec<f32> {
    if (h, w) == (k, k) {
        return img.to_vec();
    }
    let y0 = (h - k) / 2;
    let x0 = (w - k) / 2;
    let mut out = vec![0.0f32; c * k * k];
    for ch in 0..c {
        let src = &img[ch * h * w..(ch + 1) * h * w];
        let dst = &mut out[ch * k * k..(ch + 1) * k * k];
        for y in 0..k {
            dst[y * k..(y + 1) * k].copy_from_slice(&src[(y0 + y) * w + x0..(y0 + y) * w + x0 + k]);
        }
    }
    out
}

/// Horizontal pass of a 1-D tap table: `[c, h, w]` → `[c, h, taps.len()]`.
fn resample_w(img: &[f32], c: usize, h: usize, w: usize, taps: &[Tap1D]) -> Vec<f32> {
    let ow = taps.len();
    let mut out = vec![0.0f32; c * h * ow];
    for ch in 0..c {
        let src = &img[ch * h * w..(ch + 1) * h * w];
        let dst = &mut out[ch * h * ow..(ch + 1) * h * ow];
        for y in 0..h {
            let srow = &src[y * w..(y + 1) * w];
            let drow = &mut dst[y * ow..(y + 1) * ow];
            for (x, tap) in taps.iter().enumerate() {
                drow[x] = tap
                    .weights
                    .iter()
                    .zip(&srow[tap.first..tap.first + tap.weights.len()])
                    .map(|(&k, &v)| k * v)
                    .sum();
            }
        }
    }
    out
}

/// Vertical pass of a 1-D tap table: `[c, h, w]` → `[c, taps.len(), w]`.
fn resample_h(img: &[f32], c: usize, h: usize, w: usize, taps: &[Tap1D]) -> Vec<f32> {
    let oh = taps.len();
    let mut out = vec![0.0f32; c * oh * w];
    for ch in 0..c {
        let src = &img[ch * h * w..(ch + 1) * h * w];
        let dst = &mut out[ch * oh * w..(ch + 1) * oh * w];
        for (y, tap) in taps.iter().enumerate() {
            let drow = &mut dst[y * w..(y + 1) * w];
            for (k, &wt) in tap.weights.iter().enumerate() {
                let srow = &src[(tap.first + k) * w..(tap.first + k + 1) * w];
                for (d, &s) in drow.iter_mut().zip(srow) {
                    *d += wt * s;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

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
