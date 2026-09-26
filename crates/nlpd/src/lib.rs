//! Full-reference Normalized Laplacian Pyramid Distance (NLPD).
//!
//! This is a native implementation of the six-level RGB configuration
//! published by Laparra et al. and implemented in the authors' PyTorch
//! reference. Input RGB values are sRGB-encoded floats in `[0, 1]`;
//! the metric does not linearize them. Lower scores indicate closer images.

#![forbid(unsafe_code)]

mod simd;

use std::fmt;

const LOWPASS: [f32; 5] = [0.05, 0.25, 0.4, 0.25, 0.05];
const SIGMAS: [f32; 6] = [0.0248, 0.0185, 0.0179, 0.0191, 0.0220, 0.2782];
// Neighbour weights are top, left, right, bottom. The original filters
// have a zero centre and zero corners.
const DN: [[f32; 4]; 6] = [
    [0.1011, 0.1493, 0.1460, 0.1015],
    [0.0757, 0.1986, 0.1846, 0.0837],
    [0.0477, 0.2138, 0.2243, 0.0467],
    [0.0, 0.2503, 0.2616, 0.0],
    [0.0, 0.2598, 0.2552, 0.0],
    [0.0, 0.2215, 0.0717, 0.0],
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    InvalidDimensions,
    InvalidLength { expected: usize, actual: usize },
    NonFiniteInput,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimensions => write!(f, "NLPD requires width and height >= 96"),
            Self::InvalidLength { expected, actual } => {
                write!(f, "expected {expected} RGB values, got {actual}")
            }
            Self::NonFiniteInput => write!(f, "NLPD input contains a non-finite value"),
        }
    }
}

impl std::error::Error for Error {}

/// Score interleaved 8-bit RGB images using six pyramid levels.
pub fn score_rgb_u8(
    reference: &[u8],
    distorted: &[u8],
    width: usize,
    height: usize,
) -> Result<f64, Error> {
    let expected = validate_dimensions(width, height)?;
    if reference.len() != expected {
        return Err(Error::InvalidLength {
            expected,
            actual: reference.len(),
        });
    }
    if distorted.len() != expected {
        return Err(Error::InvalidLength {
            expected,
            actual: distorted.len(),
        });
    }
    // u8 -> [0,1] f32 folded into the planar deinterleave: one pass, no
    // intermediate full-size f32 copies of the interleaved inputs.
    let ref_planes = deinterleave_u8(reference, width * height);
    let dis_planes = deinterleave_u8(distorted, width * height);
    Ok(score_planes(ref_planes, dis_planes, width, height))
}

/// Score interleaved RGB images in `[0, 1]`, matching the PyTorch reference.
pub fn score_rgb_f32(
    reference: &[f32],
    distorted: &[f32],
    width: usize,
    height: usize,
) -> Result<f64, Error> {
    let expected = validate_dimensions(width, height)?;
    if reference.len() != expected {
        return Err(Error::InvalidLength {
            expected,
            actual: reference.len(),
        });
    }
    if distorted.len() != expected {
        return Err(Error::InvalidLength {
            expected,
            actual: distorted.len(),
        });
    }
    if reference.iter().chain(distorted).any(|v| !v.is_finite()) {
        return Err(Error::NonFiniteInput);
    }

    let ref_planes = deinterleave_f32(reference, width * height);
    let dis_planes = deinterleave_f32(distorted, width * height);
    Ok(score_planes(ref_planes, dis_planes, width, height))
}

fn score_planes(
    mut ref_planes: Vec<Vec<f32>>,
    mut dis_planes: Vec<Vec<f32>>,
    width: usize,
    height: usize,
) -> f64 {
    let mut w = width;
    let mut h = height;
    let mut weighted_sum = 0.0_f64;
    // Per-side scratch slabs, sized once at the level-0 extent and sliced
    // down for deeper levels. Replaces the per-(channel × level) Vec churn —
    // every slab region is fully written by its producing pass before read.
    let mut ref_scratch = Scratch::new(w * h);
    let mut dis_scratch = Scratch::new(w * h);

    for level in 0..6 {
        let mut squared_sum = 0.0_f64;
        let mut next_ref = Vec::with_capacity(3);
        let mut next_dis = Vec::with_capacity(3);
        for channel in 0..3 {
            let (ref_low, low_w, low_h) =
                level_transform(&ref_planes[channel], w, h, level, &mut ref_scratch);
            let (dis_low, _, _) =
                level_transform(&dis_planes[channel], w, h, level, &mut dis_scratch);
            squared_sum += simd::squared_difference(
                &ref_scratch.normalized[..w * h],
                &dis_scratch.normalized[..w * h],
            );
            next_ref.push(ref_low);
            next_dis.push(dis_low);
            debug_assert_eq!(low_w, w.div_ceil(2));
            debug_assert_eq!(low_h, h.div_ceil(2));
        }
        let rms = (squared_sum / (w * h * 3) as f64).sqrt();
        weighted_sum += rms.powf(0.6);
        ref_planes = next_ref;
        dis_planes = next_dis;
        w = w.div_ceil(2);
        h = h.div_ceil(2);
    }
    weighted_sum.powf(1.0 / 0.6)
}

fn validate_dimensions(width: usize, height: usize) -> Result<usize, Error> {
    if width < 96 || height < 96 {
        return Err(Error::InvalidDimensions);
    }
    width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(3))
        .ok_or(Error::InvalidDimensions)
}

fn deinterleave_u8(src: &[u8], n: usize) -> Vec<Vec<f32>> {
    let mut planes = vec![vec![0.0; n]; 3];
    for (i, px) in src.as_chunks::<3>().0.iter().enumerate() {
        for c in 0..3 {
            planes[c][i] = px[c] as f32 / 255.0;
        }
    }
    planes
}

fn deinterleave_f32(src: &[f32], n: usize) -> Vec<Vec<f32>> {
    let mut planes = vec![vec![0.0; n]; 3];
    for (i, px) in src.as_chunks::<3>().0.iter().enumerate() {
        for c in 0..3 {
            planes[c][i] = px[c];
        }
    }
    planes
}

fn reflect(mut x: isize, len: usize) -> usize {
    let bound = len as isize;
    while x < 0 || x >= bound {
        if x < 0 {
            x = -x;
        }
        if x >= bound {
            x = 2 * bound - 2 - x;
        }
    }
    x as usize
}

/// Reusable per-side work buffers for `level_transform`. Slabs are allocated
/// once at the level-0 extent and sliced narrower for deeper levels — each
/// region is fully written by its producing pass before any read, so no
/// stale contents are ever observed.
struct Scratch {
    /// Horizontal low-pass output (downsample: `out_w*h`; blur: `w*h`).
    tmp_h: Vec<f32>,
    /// Bilinear-upsampled low band.
    up: Vec<f32>,
    /// Blur horizontal-pass output.
    tmp_b: Vec<f32>,
    /// Smoothed band, then reused in place as the Laplacian (`src - smoothed`).
    lap: Vec<f32>,
    /// Normalized Laplacian band (per-channel result plane).
    normalized: Vec<f32>,
}

impl Scratch {
    fn new(n: usize) -> Self {
        Self {
            tmp_h: vec![0.0; n],
            up: vec![0.0; n],
            tmp_b: vec![0.0; n],
            lap: vec![0.0; n],
            normalized: vec![0.0; n],
        }
    }
}

fn downsample(src: &[f32], w: usize, h: usize, tmp_h: &mut [f32]) -> (Vec<f32>, usize, usize) {
    let out_w = w.div_ceil(2);
    let out_h = h.div_ceil(2);
    let pad_x = ((out_w - 1) * 2 + 5 - w) / 2;
    let pad_y = ((out_h - 1) * 2 + 5 - h) / 2;
    filter5_horizontal(src, w, h, &mut tmp_h[..out_w * h], out_w, 2, pad_x);
    let mut out = vec![0.0; out_w * out_h];
    simd::filter5_vertical(&tmp_h[..out_w * h], out_w, h, &mut out, out_h, 2, pad_y);
    (out, out_w, out_h)
}

/// Horizontal 5-tap low-pass. Interior outputs (all taps in-bounds) are
/// computed reflect-free in accumulate form — `out[ox] += w_k*src[ox*stride+k-pad]`
/// iterates k outer, ox inner, over equal-length slice pairs so bounds checks
/// hoist and LLVM vectorizes each pass into a contiguous SAXPY; per-output
/// additions stay in k order, bit-identical to the naive formulation. Only
/// the outputs whose taps fall off the row edge take the scalar `reflect`
/// path. For `stride == 2` (the pyramid decimator) the taps are a stride-2
/// subset of a contiguous conv, so we compute that full conv — vectorized —
/// and decimate; identical values, ~2x the taps at ~4x the lane throughput.
fn filter5_horizontal(
    src: &[f32],
    w: usize,
    h: usize,
    dst: &mut [f32],
    out_w: usize,
    stride: usize,
    pad: usize,
) {
    debug_assert_eq!(dst.len(), out_w * h);
    // Interior ox range: every tap index ox*stride+k-pad ∈ [0, w) for k∈0..5.
    let lo = pad.div_ceil(stride).min(out_w);
    // Last ox with ox*stride + 4 - pad <= w-1 → ox <= (w+pad-5)/stride.
    let hi = if w + pad >= 5 {
        ((w + pad - 5) / stride + 1).min(out_w)
    } else {
        0
    };
    let hi = hi.max(lo);
    if stride == 2 && w >= 5 {
        filter5_horizontal_decimate(src, w, h, dst, out_w, pad, lo, hi);
        return;
    }
    if stride != 1 {
        // Unseen stride: conservative direct-index path (interior only).
        for y in 0..h {
            let row = &src[y * w..(y + 1) * w];
            let out = &mut dst[y * out_w..(y + 1) * out_w];
            for (ox, o) in out.iter_mut().enumerate().take(lo) {
                *o = tap5(row, w, ox, stride, pad);
            }
            for (ox, o) in out.iter_mut().enumerate().take(out_w).skip(hi) {
                *o = tap5(row, w, ox, stride, pad);
            }
            for ox in lo..hi {
                let mut sum = 0.0_f32;
                for (kx, weight) in LOWPASS.into_iter().enumerate() {
                    sum += row[ox * stride + kx - pad] * weight;
                }
                out[ox] = sum;
            }
        }
        return;
    }
    for y in 0..h {
        let row = &src[y * w..(y + 1) * w];
        let out = &mut dst[y * out_w..(y + 1) * out_w];
        for (ox, o) in out.iter_mut().enumerate().take(lo) {
            *o = tap5(row, w, ox, stride, pad);
        }
        for (ox, o) in out.iter_mut().enumerate().take(out_w).skip(hi) {
            *o = tap5(row, w, ox, stride, pad);
        }
        if lo < hi {
            let span = hi - lo;
            // k = 0 initializes (dst may be reused scratch); k = 1..5 accumulate.
            let s0 = &row[lo * stride - pad..lo * stride - pad + span];
            for (o, &v) in out[lo..hi].iter_mut().zip(s0) {
                *o = v * LOWPASS[0];
            }
            for (k, weight) in LOWPASS.iter().enumerate().skip(1) {
                // stride == 1: contiguous source window for contiguous out.
                let s = &row[lo * stride + k - pad..lo * stride + k - pad + span];
                for (o, &v) in out[lo..hi].iter_mut().zip(s) {
                    *o += v * weight;
                }
            }
        }
    }
}

/// Stride-2 path of [`filter5_horizontal`]: G[j] = Σ_k w_k·row[j+k] for
/// j∈[0, w-4] (every tap position), then `out[ox] = G[2ox - pad]` — the same
/// k-ordered sum `tap5` computes, produced by contiguous vectorizable
/// accumulates instead of a strided gather.
fn filter5_horizontal_decimate(
    src: &[f32],
    w: usize,
    h: usize,
    dst: &mut [f32],
    out_w: usize,
    pad: usize,
    lo: usize,
    hi: usize,
) {
    let g_len = w - 4;
    let mut g = vec![0.0_f32; g_len];
    for y in 0..h {
        let row = &src[y * w..(y + 1) * w];
        let out = &mut dst[y * out_w..(y + 1) * out_w];
        for (ox, o) in out.iter_mut().enumerate().take(lo) {
            *o = tap5(row, w, ox, 2, pad);
        }
        for (ox, o) in out.iter_mut().enumerate().take(out_w).skip(hi) {
            *o = tap5(row, w, ox, 2, pad);
        }
        if lo < hi {
            g.iter_mut().for_each(|v| *v = 0.0);
            for (k, weight) in LOWPASS.iter().enumerate() {
                for (j, &v) in g.iter_mut().zip(&row[k..k + g_len]) {
                    *j += v * weight;
                }
            }
            for ox in lo..hi {
                out[ox] = g[2 * ox - pad];
            }
        }
    }
}

fn tap5(row: &[f32], w: usize, ox: usize, stride: usize, pad: usize) -> f32 {
    let mut sum = 0.0_f32;
    for (kx, weight) in LOWPASS.into_iter().enumerate() {
        let sx = reflect((ox * stride + kx) as isize - pad as isize, w);
        sum += row[sx] * weight;
    }
    sum
}

fn upsample_bilinear(src: &[f32], sw: usize, sh: usize, w: usize, h: usize, dst: &mut [f32]) {
    debug_assert_eq!(dst.len(), w * h);
    // The x mapping is row-invariant — hoist the (x0, x1, tx) tables out of
    // the y loop; identical arithmetic to the per-pixel recomputation.
    let mut xtab = vec![(0usize, 0usize, 0.0_f32); w];
    for (x, t) in xtab.iter_mut().enumerate() {
        let fx = if w > 1 {
            x as f32 * (sw - 1) as f32 / (w - 1) as f32
        } else {
            0.0
        };
        let x0 = fx.floor() as usize;
        let x1 = (x0 + 1).min(sw - 1);
        *t = (x0, x1, fx - x0 as f32);
    }
    for y in 0..h {
        let fy = if h > 1 {
            y as f32 * (sh - 1) as f32 / (h - 1) as f32
        } else {
            0.0
        };
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(sh - 1);
        let ty = fy - y0 as f32;
        let (r0, r1) = (&src[y0 * sw..(y0 + 1) * sw], &src[y1 * sw..(y1 + 1) * sw]);
        let out = &mut dst[y * w..(y + 1) * w];
        for (x, &(x0, x1, tx)) in xtab.iter().enumerate() {
            let a = r0[x0] * (1.0 - tx) + r0[x1] * tx;
            let b = r1[x0] * (1.0 - tx) + r1[x1] * tx;
            out[x] = a * (1.0 - ty) + b * ty;
        }
    }
}

fn blur5(src: &[f32], w: usize, h: usize, tmp_b: &mut [f32], out: &mut [f32]) {
    filter5_horizontal(src, w, h, &mut tmp_b[..w * h], w, 1, 2);
    simd::filter5_vertical(&tmp_b[..w * h], w, h, out, h, 1, 2);
}

/// Normalized Laplacian band for one channel. Interior pixels
/// (x∈[1,w-1), y∈[1,h-1)) take a reflect-free elementwise path that
/// auto-vectorizes; the border ring keeps `reflect` indexing. Identical
/// values either way.
fn normalize_band(lap: &[f32], w: usize, h: usize, level: usize, normalized: &mut [f32]) {
    let [top, left, right, bottom] = DN[level];
    let sigma = SIGMAS[level];
    for y in 0..h {
        if y == 0 || y + 1 == h {
            for x in 0..w {
                normalized[y * w + x] = norm_px(lap, w, h, x, y, top, left, right, bottom, sigma);
            }
        } else {
            normalized[y * w] = norm_px(lap, w, h, 0, y, top, left, right, bottom, sigma);
            normalized[y * w + w - 1] =
                norm_px(lap, w, h, w - 1, y, top, left, right, bottom, sigma);
            // Interior row as five equal-length zipped slices — no per-element
            // bounds checks, auto-vectorizes (abs + fma + div).
            let out = &mut normalized[y * w + 1..y * w + w - 1];
            let mid = &lap[y * w + 1..y * w + w - 1];
            let top_r = &lap[(y - 1) * w + 1..(y - 1) * w + w - 1];
            let left_r = &lap[y * w..y * w + w - 2];
            let right_r = &lap[y * w + 2..y * w + w];
            let bot_r = &lap[(y + 1) * w + 1..(y + 1) * w + w - 1];
            for i in 0..w - 2 {
                let denom = sigma
                    + top * top_r[i].abs()
                    + left * left_r[i].abs()
                    + right * right_r[i].abs()
                    + bottom * bot_r[i].abs();
                out[i] = mid[i] / denom;
            }
        }
    }
}

fn norm_px(
    lap: &[f32],
    w: usize,
    h: usize,
    x: usize,
    y: usize,
    top: f32,
    left: f32,
    right: f32,
    bottom: f32,
    sigma: f32,
) -> f32 {
    let ym = reflect(y as isize - 1, h);
    let yp = reflect(y as isize + 1, h);
    let xm = reflect(x as isize - 1, w);
    let xp = reflect(x as isize + 1, w);
    let denom = sigma
        + top * lap[ym * w + x].abs()
        + left * lap[y * w + xm].abs()
        + right * lap[y * w + xp].abs()
        + bottom * lap[yp * w + x].abs();
    lap[y * w + x] / denom
}

/// One (level, channel) transform: downsample → bilinear upsample → smooth →
/// subtract → energy-normalize into `scratch.normalized[..w*h]`. Returns the
/// low band that feeds the next pyramid level.
fn level_transform(
    src: &[f32],
    w: usize,
    h: usize,
    level: usize,
    scratch: &mut Scratch,
) -> (Vec<f32>, usize, usize) {
    let n = w * h;
    let (low, lw, lh) = downsample(src, w, h, &mut scratch.tmp_h);
    upsample_bilinear(&low, lw, lh, w, h, &mut scratch.up[..n]);
    blur5(&scratch.up[..n], w, h, &mut scratch.tmp_b, &mut scratch.lap);
    // lap = src - smoothed, folded elementwise over the smoothed slab.
    for (s, &a) in scratch.lap[..n].iter_mut().zip(src.iter()) {
        *s = a - *s;
    }
    normalize_band(&scratch.lap[..n], w, h, level, &mut scratch.normalized);
    (low, lw, lh)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_and_asymmetry() {
        let w = 97;
        let h = 99;
        let reference = (0..w * h * 3)
            .map(|i| ((i * 37 + i / 97) % 256) as u8)
            .collect::<Vec<_>>();
        let mut changed = reference.clone();
        for p in changed.chunks_exact_mut(3).step_by(19) {
            p[0] = p[0].saturating_add(20);
        }
        assert!(score_rgb_u8(&reference, &reference, w, h).unwrap() < 1e-9);
        let a = score_rgb_u8(&reference, &changed, w, h).unwrap();
        // PyTorch 2.5.1+cpu, Valerolaparra/NLPD_Pytorch, six RGB levels.
        assert!((a - 0.477_978_587_150_573_73).abs() < 1e-5, "{a}");
        let b = score_rgb_u8(&changed, &reference, w, h).unwrap();
        assert!(a > 0.0 && (a - b).abs() < 1e-9);
    }
}
