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
//!   0.0; a = −0.75, the OpenCV/PIL convention, does not).

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
