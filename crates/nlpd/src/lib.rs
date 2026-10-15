//! Full-reference Normalized Laplacian Pyramid Distance (NLPD).
//!
//! This is a native implementation of the six-level RGB configuration
//! published by Laparra et al. and implemented in the authors' PyTorch
//! reference. Input RGB values are sRGB-encoded floats in `[0, 1]`;
//! the metric does not linearize them. Lower scores indicate closer images.

#![forbid(unsafe_code)]

pub mod iqa;
mod simd;

use std::fmt;

use zenmetrics_abstractions as za;

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
    Ok(score_planes(ref_planes, dis_planes, width, height, None))
}

/// [`score_rgb_u8`] plus each level's squared normalized-Laplacian
/// difference summed over R, G and B (`levels[l]` is `⌈w/2^l⌉ × ⌈h/2^l⌉`).
/// The score is computed exactly as [`score_rgb_u8`] computes it.
fn score_rgb_u8_maps(
    reference: &[u8],
    distorted: &[u8],
    width: usize,
    height: usize,
) -> Result<(f64, Vec<Vec<f32>>), Error> {
    let expected = validate_dimensions(width, height)?;
    for buf in [reference, distorted] {
        if buf.len() != expected {
            return Err(Error::InvalidLength {
                expected,
                actual: buf.len(),
            });
        }
    }
    let ref_planes = deinterleave_u8(reference, width * height);
    let dis_planes = deinterleave_u8(distorted, width * height);
    let mut levels = Vec::with_capacity(LEVELS);
    let score = score_planes(ref_planes, dis_planes, width, height, Some(&mut levels));
    Ok((score, levels))
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
    Ok(score_planes(ref_planes, dis_planes, width, height, None))
}

/// Pyramid levels of the authors' configuration.
const LEVELS: usize = 6;

/// The six-level score. With `maps`, each level's per-pixel
/// `Σ_c (N_ref − N_dist)²` is pushed as well — computed from the same
/// normalized planes after the score's own reduction has read them.
fn score_planes(
    mut ref_planes: Vec<Vec<f32>>,
    mut dis_planes: Vec<Vec<f32>>,
    width: usize,
    height: usize,
    mut maps: Option<&mut Vec<Vec<f32>>>,
) -> f64 {
    let mut w = width;
    let mut h = height;
    let mut weighted_sum = 0.0_f64;
    // Per-side scratch slabs, sized once at the level-0 extent and sliced
    // down for deeper levels. Replaces the per-(channel × level) Vec churn —
    // every slab region is fully written by its producing pass before read.
    let mut ref_scratch = Scratch::new(w * h);
    let mut dis_scratch = Scratch::new(w * h);

    for level in 0..LEVELS {
        let mut squared_sum = 0.0_f64;
        let mut next_ref = Vec::with_capacity(3);
        let mut next_dis = Vec::with_capacity(3);
        let mut level_map = maps.is_some().then(|| vec![0.0_f32; w * h]);
        for channel in 0..3 {
            let (ref_low, low_w, low_h) =
                level_transform(&ref_planes[channel], w, h, level, &mut ref_scratch);
            let (dis_low, _, _) =
                level_transform(&dis_planes[channel], w, h, level, &mut dis_scratch);
            squared_sum += simd::squared_difference(
                &ref_scratch.normalized[..w * h],
                &dis_scratch.normalized[..w * h],
            );
            if let Some(m) = level_map.as_mut() {
                let pairs = ref_scratch.normalized[..w * h]
                    .iter()
                    .zip(&dis_scratch.normalized[..w * h]);
                for (acc, (&a, &b)) in m.iter_mut().zip(pairs) {
                    let d = a - b;
                    *acc += d * d;
                }
            }
            next_ref.push(ref_low);
            next_dis.push(dis_low);
            debug_assert_eq!(low_w, w.div_ceil(2));
            debug_assert_eq!(low_h, h.div_ceil(2));
        }
        let rms = (squared_sum / (w * h * 3) as f64).sqrt();
        weighted_sum += rms.powf(0.6);
        if let (Some(sink), Some(m)) = (maps.as_deref_mut(), level_map) {
            sink.push(m);
        }
        for v in core::mem::take(&mut ref_planes)
            .into_iter()
            .chain(core::mem::take(&mut dis_planes))
        {
            buf_give(v);
        }
        ref_planes = next_ref;
        dis_planes = next_dis;
        w = w.div_ceil(2);
        h = h.div_ceil(2);
    }
    for v in ref_planes
        .into_iter()
        .chain(dis_planes)
        .chain(ref_scratch.into_bufs())
        .chain(dis_scratch.into_bufs())
    {
        buf_give(v);
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
    let mut planes: Vec<Vec<f32>> = (0..3).map(|_| buf_take(n)).collect();
    for (c, plane) in planes.iter_mut().enumerate() {
        simd::deinterleave_u8_ch(src, c, plane);
    }
    planes
}

fn deinterleave_f32(src: &[f32], n: usize) -> Vec<Vec<f32>> {
    let mut planes: Vec<Vec<f32>> = (0..3).map(|_| buf_take(n)).collect();
    for (c, plane) in planes.iter_mut().enumerate() {
        simd::deinterleave_f32_ch(src, c, plane);
    }
    planes
}

thread_local! {
    /// Cross-call plane-buffer pool. Every slab and plane region is
    /// fully written by its producing pass before it is read, so stale
    /// pooled contents are never observed — the same invariant the
    /// `Scratch` docs assert. Reusing allocations skips the zero-fill
    /// and page-fault cost of ~n·60 bytes of `vec![0.0]` per call.
    static BUF_POOL: std::cell::RefCell<Vec<Vec<f32>>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Pooled buffers retained per thread — enough for one call's working
/// set (6 planes + 8 scratch slabs + ~18 live level bands).
const POOL_CAP: usize = 40;

/// Take a length-`n` plane buffer, reusing a pooled allocation when one
/// fits. Contents are unspecified — every element is written by the
/// caller's producing pass before it is read.
fn buf_take(n: usize) -> Vec<f32> {
    let mut v = BUF_POOL.with(|p| {
        let mut p = p.borrow_mut();
        let idx = p
            .iter()
            .enumerate()
            .filter(|(_, v)| v.capacity() >= n)
            // Prefer `len >= n` so `resize` truncates rather than
            // zero-filling an extension; then smallest capacity.
            .min_by_key(|(_, v)| (v.len() < n, v.capacity()))
            .map(|(i, _)| i);
        idx.map(|i| p.swap_remove(i)).unwrap_or_default()
    });
    v.resize(n, 0.0);
    v
}

/// Return a buffer to the pool for reuse by a later call.
fn buf_give(v: Vec<f32>) {
    if v.capacity() > 0 {
        BUF_POOL.with(|p| {
            let mut p = p.borrow_mut();
            if p.len() < POOL_CAP {
                p.push(v);
            }
        });
    }
}

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Row-chunked iteration over `dst`'s `w`-element rows. Serial without the
/// `parallel` feature; rayon `par_chunks_mut` with it. `f(y, dst_row)` must
/// write only within `dst_row` (reads may alias arbitrarily) — the disjoint
/// per-row writes keep results bit-identical either way. Chunk sizing
/// targets ≥ ~64 KiB of output per task.
fn for_each_row(dst: &mut [f32], w: usize, f: impl Fn(usize, &mut [f32]) + Sync + Send) {
    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && dst.len() >= (1 << 15) {
        let rows_per = (32768usize / w.max(1)).clamp(4, 512);
        dst.par_chunks_mut(w * rows_per)
            .enumerate()
            .for_each(|(ci, chunk)| {
                for (r, out) in chunk.chunks_mut(w).enumerate() {
                    f(ci * rows_per + r, out);
                }
            });
        return;
    }
    for (y, out) in dst.chunks_mut(w).enumerate() {
        f(y, out);
    }
}

/// `for_each_row` carrying per-chunk scratch `T` built by `init` — for row
/// bodies that need a reused workspace.
fn for_each_row_init<T: Send>(
    dst: &mut [f32],
    w: usize,
    init: impl Fn() -> T + Sync + Send,
    f: impl Fn(usize, &mut [f32], &mut T) + Sync + Send,
) {
    for_each_row_init_min(dst, w, 4, init, f);
}

fn for_each_row_init_min<T: Send>(
    dst: &mut [f32],
    w: usize,
    _min_rows: usize,
    init: impl Fn() -> T + Sync + Send,
    f: impl Fn(usize, &mut [f32], &mut T) + Sync + Send,
) {
    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && dst.len() >= (1 << 15) {
        let rows_per = (32768usize / w.max(1)).clamp(_min_rows, 512);
        dst.par_chunks_mut(w * rows_per)
            .enumerate()
            .for_each(|(ci, chunk)| {
                let mut t = init();
                for (r, out) in chunk.chunks_mut(w).enumerate() {
                    f(ci * rows_per + r, out, &mut t);
                }
            });
        return;
    }
    let mut t = init();
    for (y, out) in dst.chunks_mut(w).enumerate() {
        f(y, out, &mut t);
    }
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
    /// Normalized Laplacian band (per-channel result plane).
    normalized: Vec<f32>,
}

impl Scratch {
    fn new(n: usize) -> Self {
        Self {
            tmp_h: buf_take(n),
            normalized: buf_take(n),
        }
    }

    /// Drain the slabs for return to [`buf_give`].
    fn into_bufs(self) -> impl Iterator<Item = Vec<f32>> {
        [self.tmp_h, self.normalized].into_iter()
    }
}

fn downsample(src: &[f32], w: usize, h: usize, tmp_h: &mut [f32]) -> (Vec<f32>, usize, usize) {
    let out_w = w.div_ceil(2);
    let out_h = h.div_ceil(2);
    let pad_x = ((out_w - 1) * 2 + 5 - w) / 2;
    let pad_y = ((out_h - 1) * 2 + 5 - h) / 2;
    filter5_horizontal(src, w, h, &mut tmp_h[..out_w * h], out_w, 2, pad_x);
    let mut out = buf_take(out_w * out_h);
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
        for_each_row(dst, out_w, |y, out| {
            let row = &src[y * w..(y + 1) * w];
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
        });
        return;
    }
    for_each_row(dst, out_w, |y, out| {
        filter5_row(&src[y * w..(y + 1) * w], w, out, lo, hi);
    });
}

/// One row of the stride-1/pad-2 horizontal 5-tap: `tap5` borders plus
/// the fused interior pass — the per-row body of [`filter5_horizontal`].
fn filter5_row(row: &[f32], w: usize, out: &mut [f32], lo: usize, hi: usize) {
    for (ox, o) in out.iter_mut().enumerate().take(lo) {
        *o = tap5(row, w, ox, 1, 2);
    }
    let out_w = out.len();
    for (ox, o) in out.iter_mut().enumerate().take(out_w).skip(hi) {
        *o = tap5(row, w, ox, 1, 2);
    }
    if lo < hi {
        // Fused single-pass conv — identical per-output product/add
        // order to the former 5-pass accumulate.
        simd::h5_fused::<false>(&row[lo - 2..], &mut out[lo..hi]);
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
    let _ = h; // rows are driven off `dst`'s length now
    let g_len = w - 4;
    // pad ∈ {1,2} here: out_w = ceil(w/2) forces it. Splitting each row
    // into even/odd halves turns the stride-2 taps into contiguous
    // loads at half length — only the kept outputs are computed.
    let eo = pad == 1 || pad == 2;
    let ne = w.div_ceil(2);
    let no = w / 2;
    for_each_row_init(
        dst,
        out_w,
        || {
            (
                vec![0.0_f32; ne.max(1)],
                vec![0.0_f32; no.max(g_len.max(1))],
            )
        },
        |y, out, eo_g| {
            let row = &src[y * w..(y + 1) * w];
            for (ox, o) in out.iter_mut().enumerate().take(lo) {
                *o = tap5(row, w, ox, 2, pad);
            }
            for (ox, o) in out.iter_mut().enumerate().take(out_w).skip(hi) {
                *o = tap5(row, w, ox, 2, pad);
            }
            if lo < hi {
                if eo {
                    // Deinterleaved conv — same k-ordered sum as tap5,
                    // evaluated only at kept positions.
                    let (e, og) = eo_g;
                    let (e, o) = (&mut e[..ne], &mut og[..no]);
                    simd::deinterleave2(row, e, o);
                    simd::h5_eo(e, o, pad, lo, hi, out);
                } else {
                    // Fused single-pass G — same k-ordered sum as the
                    // former 5-pass accumulate, ZINIT replicates the
                    // cleared-slot `+=`.
                    simd::h5_fused::<true>(row, &mut eo_g.1[..g_len]);
                    for ox in lo..hi {
                        out[ox] = eo_g.1[2 * ox - pad];
                    }
                }
            }
        },
    );
}

fn tap5(row: &[f32], w: usize, ox: usize, stride: usize, pad: usize) -> f32 {
    let mut sum = 0.0_f32;
    for (kx, weight) in LOWPASS.into_iter().enumerate() {
        let sx = reflect((ox * stride + kx) as isize - pad as isize, w);
        sum += row[sx] * weight;
    }
    sum
}

/// Row-invariant x-mapping tables for the align-corners bilinear
/// resample `(sw → w)`: per output column, the two source indices and
/// the fractional weight. Split arrays so `tx` loads contiguously in the
/// vector row kernel.
fn upsample_xtabs(sw: usize, w: usize) -> (Vec<usize>, Vec<usize>, Vec<f32>) {
    let mut x0v = vec![0usize; w];
    let mut x1v = vec![0usize; w];
    let mut txv = vec![0.0_f32; w];
    for x in 0..w {
        let fx = if w > 1 {
            x as f32 * (sw - 1) as f32 / (w - 1) as f32
        } else {
            0.0
        };
        let x0 = fx.floor() as usize;
        x0v[x] = x0;
        x1v[x] = (x0 + 1).min(sw - 1);
        txv[x] = fx - x0 as f32;
    }
    (x0v, x1v, txv)
}

/// A 2-slot ring of horizontally-lerped source rows keyed by source-row
/// index. Output-row y advances y0 by ≤1, so each source row is lerped
/// once instead of once per output row — halving the gather work.
/// Bit-identical: the same two lerps feed the same blend formula.
struct LerpRing {
    bufs: [Vec<f32>; 2],
    idx: [usize; 2],
}

impl LerpRing {
    fn new(w: usize) -> Self {
        Self {
            bufs: [vec![0.0; w], vec![0.0; w]],
            idx: [usize::MAX; 2],
        }
    }

    /// Write output row `y` of the `(sw,sh) → (·,h)` bilinear resample of
    /// `src` into `out` (w floats, with `xtabs` from [`upsample_xtabs`]).
    #[allow(clippy::too_many_arguments)]
    fn produce(
        &mut self,
        src: &[f32],
        sw: usize,
        sh: usize,
        h: usize,
        y: usize,
        x0v: &[usize],
        x1v: &[usize],
        txv: &[f32],
        out: &mut [f32],
    ) {
        let fy = if h > 1 {
            y as f32 * (sh - 1) as f32 / (h - 1) as f32
        } else {
            0.0
        };
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(sh - 1);
        let ty = fy - y0 as f32;
        // Slot 0 tracks y0: an advance by one rotates the old
        // y1 lerp into place; anything else recomputes.
        if self.idx[0] != y0 {
            if self.idx[1] == y0 {
                self.bufs.swap(0, 1);
                self.idx.swap(0, 1);
            } else {
                simd::hlerp_row(
                    &src[y0 * sw..(y0 + 1) * sw],
                    x0v,
                    x1v,
                    txv,
                    &mut self.bufs[0],
                );
                self.idx[0] = y0;
            }
        }
        if y1 == y0 {
            // Clamped last source row: a == b.
            simd::blend_rows(&self.bufs[0], &self.bufs[0], ty, out);
        } else {
            if self.idx[1] != y1 {
                simd::hlerp_row(
                    &src[y1 * sw..(y1 + 1) * sw],
                    x0v,
                    x1v,
                    txv,
                    &mut self.bufs[1],
                );
                self.idx[1] = y1;
            }
            simd::blend_rows(&self.bufs[0], &self.bufs[1], ty, out);
        }
    }
}

#[cfg(test)]
fn upsample_bilinear(src: &[f32], sw: usize, sh: usize, w: usize, h: usize, dst: &mut [f32]) {
    debug_assert_eq!(dst.len(), w * h);
    let (x0v, x1v, txv) = upsample_xtabs(sw, w);
    for_each_row_init(
        dst,
        w,
        || LerpRing::new(w),
        |y, out, ring| ring.produce(src, sw, sh, h, y, &x0v, &x1v, &txv, out),
    );
}

#[cfg(test)]
fn blur5(src: &[f32], w: usize, h: usize, tmp_b: &mut [f32], out: &mut [f32]) {
    filter5_horizontal(src, w, h, &mut tmp_b[..w * h], w, 1, 2);
    simd::filter5_vertical(&tmp_b[..w * h], w, h, out, h, 1, 2);
}

/// Normalized Laplacian band for one channel. Interior pixels
/// (x∈[1,w-1), y∈[1,h-1)) take a reflect-free elementwise path that
/// auto-vectorizes; the border ring keeps `reflect` indexing. Identical
/// values either way.
#[cfg(test)]
fn normalize_band(lap: &[f32], w: usize, h: usize, level: usize, normalized: &mut [f32]) {
    let [top, left, right, bottom] = DN[level];
    let sigma = SIGMAS[level];
    // `normalized` is the level-0-sized scratch slab — iterate only this
    // level's w×h subregion.
    for_each_row(&mut normalized[..w * h], w, |y, out_row| {
        if y == 0 || y + 1 == h {
            for x in 0..w {
                out_row[x] = norm_px(lap, w, h, x, y, top, left, right, bottom, sigma);
            }
        } else {
            out_row[0] = norm_px(lap, w, h, 0, y, top, left, right, bottom, sigma);
            out_row[w - 1] = norm_px(lap, w, h, w - 1, y, top, left, right, bottom, sigma);
            // Interior row as five equal-length zipped slices — no per-element
            // bounds checks, auto-vectorizes (abs + fma + div).
            let out = &mut out_row[1..w - 1];
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
    });
}

#[cfg(test)]
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
    let (low, lw, lh) = downsample(src, w, h, &mut scratch.tmp_h);
    // The upsample → H conv → lap → normalize chain runs as one fused
    // row-ring pass: up rows are produced into a per-chunk scratch only
    // as the H conv consumes them — neither `up` nor a horizontal-blur
    // plane is ever materialized.
    simd::normalize_filtered(
        src,
        &low,
        lw,
        lh,
        w,
        h,
        w,
        h,
        level,
        &mut scratch.normalized[..w * h],
    );

    (low, lw, lh)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fused_normalization_matches_materialized_bits() {
        for (w, h) in [(3, 3), (5, 7), (11, 13), (97, 99), (257, 193), (1024, 65)] {
            let n = w * h;
            let original: Vec<f32> = (0..n).map(|i| (i * 37 % 251) as f32 / 251.0).collect();
            // Real low band — the fused path derives its own `up` rows.
            let mut tmp = vec![0.0; n];
            let (low, lw, lh) = downsample(&original, w, h, &mut tmp);
            let mut up = vec![0.0; n];
            upsample_bilinear(&low, lw, lh, w, h, &mut up);
            let mut horizontal = vec![0.0; n];
            let mut lap = vec![0.0; n];
            blur5(&up, w, h, &mut horizontal, &mut lap);
            for (value, &src) in lap.iter_mut().zip(&original) {
                *value = src - *value;
            }
            for level in 0..6 {
                let mut expected = vec![0.0; n];
                normalize_band(&lap, w, h, level, &mut expected);
                let mut actual = vec![0.0; n];
                simd::normalize_filtered(&original, &low, lw, lh, w, h, w, h, level, &mut actual);
                for (i, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
                    assert_eq!(a.to_bits(), e.to_bits(), "{w}x{h} level={level} pixel={i}");
                }
            }
        }
    }

    /// Every level map, rebuilt from the materialized reference path
    /// (`downsample` → `upsample_bilinear` → `blur5` → subtract →
    /// `normalize_band`, which the fused kernel is pinned to above), equals
    /// the map the scorer returns bit for bit.
    #[test]
    fn level_maps_match_the_materialized_bands() {
        for (w, h) in [(97usize, 99usize), (112, 96)] {
            let a: Vec<u8> = (0..w * h * 3)
                .map(|i| ((i * 37 + i / 97) % 256) as u8)
                .collect();
            let b: Vec<u8> = a
                .iter()
                .enumerate()
                .map(|(i, &v)| if i % 23 == 0 { v.wrapping_add(31) } else { v })
                .collect();
            let (_, maps) = score_rgb_u8_maps(&a, &b, w, h).unwrap();
            let mut planes = [deinterleave_u8(&a, w * h), deinterleave_u8(&b, w * h)];
            let (mut lw, mut lh) = (w, h);
            for (level, got) in maps.iter().enumerate() {
                let mut want = vec![0.0f32; lw * lh];
                let mut next: [Vec<Vec<f32>>; 2] = [Vec::new(), Vec::new()];
                let mut bands: [Vec<Vec<f32>>; 2] = [Vec::new(), Vec::new()];
                for side in 0..2 {
                    for c in 0..3 {
                        let src = &planes[side][c];
                        let mut tmp = vec![0.0; lw * lh];
                        let (low, sw, sh) = downsample(src, lw, lh, &mut tmp);
                        let mut up = vec![0.0; lw * lh];
                        upsample_bilinear(&low, sw, sh, lw, lh, &mut up);
                        let (mut hb, mut lap) = (vec![0.0; lw * lh], vec![0.0; lw * lh]);
                        blur5(&up, lw, lh, &mut hb, &mut lap);
                        for (v, &x) in lap.iter_mut().zip(src) {
                            *v = x - *v;
                        }
                        let mut norm = vec![0.0; lw * lh];
                        normalize_band(&lap, lw, lh, level, &mut norm);
                        bands[side].push(norm);
                        next[side].push(low);
                    }
                }
                for c in 0..3 {
                    for (acc, (&x, &y)) in want.iter_mut().zip(bands[0][c].iter().zip(&bands[1][c]))
                    {
                        let d = x - y;
                        *acc += d * d;
                    }
                }
                assert_eq!(got.len(), want.len());
                for (i, (g, e)) in got.iter().zip(&want).enumerate() {
                    assert_eq!(g.to_bits(), e.to_bits(), "{w}x{h} level {level} pixel {i}");
                }
                planes = next;
                (lw, lh) = (lw.div_ceil(2), lh.div_ceil(2));
            }
        }
    }

    /// The downsample's symmetric unit-sum taps reproduce a linear ramp at
    /// each output's centre, so a ramp pyramid reads back every level's
    /// sample positions: they must be the geometry's cell centres.
    #[test]
    fn level_geometry_matches_the_downsample_phase() {
        for (w, h) in [(96usize, 97usize), (97, 96), (130, 101), (100, 128)] {
            let geometry = level_geometry(w, h);
            let ramp_x: Vec<f32> = (0..w * h).map(|i| (i % w) as f32).collect();
            let ramp_y: Vec<f32> = (0..w * h).map(|i| (i / w) as f32).collect();
            let (mut px, mut py, mut lw, mut lh) = (ramp_x, ramp_y, w, h);
            for (l, g) in geometry.iter().enumerate() {
                assert_eq!(
                    (g.cells_x as usize, g.cells_y as usize),
                    (lw, lh),
                    "level {l}"
                );
                let centre = |origin: i64, cell: u32, k: usize| {
                    origin as f32 + (cell / 2) as f32 + (k * cell as usize) as f32
                };
                // Interior samples only: the reflected border taps bend the ramp.
                let margin = 3;
                for y in margin..lh.saturating_sub(margin) {
                    for x in margin..lw.saturating_sub(margin) {
                        let (vx, vy) = (px[y * lw + x], py[y * lw + x]);
                        if l > 0 {
                            assert!(
                                (vx - centre(g.origin_x, g.cell_width, x)).abs() < 1e-3,
                                "{w}x{h} l{l} x{x}: {vx}"
                            );
                            assert!(
                                (vy - centre(g.origin_y, g.cell_height, y)).abs() < 1e-3,
                                "{w}x{h} l{l} y{y}: {vy}"
                            );
                        } else {
                            assert_eq!((vx, vy), (x as f32, y as f32));
                        }
                    }
                }
                let mut tmp = vec![0.0; lw * lh];
                let (nx, nw, nh) = downsample(&px, lw, lh, &mut tmp);
                let (ny, _, _) = downsample(&py, lw, lh, &mut tmp);
                (px, py, lw, lh) = (nx, ny, nw, nh);
            }
        }
    }

    #[test]
    fn identity_and_asymmetry() {
        let w = 97;
        let h = 99;
        let reference = (0..w * h * 3)
            .map(|i| ((i * 37 + i / 97) % 256) as u8)
            .collect::<Vec<_>>();
        let mut changed = reference.clone();
        for p in changed.as_chunks_mut::<3>().0.iter_mut().step_by(19) {
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

/// The NLPD metric as a generic scorer — accepts `RGB8_SRGB`
/// `PixelSlice` inputs (packed or strided — strided is repacked).
///
/// Maps, on request, all [`za::MapPolarity::Difference`] (`0` = identical):
///
/// - `diffmap` (primary): NLPD's own spatial breakdown at input resolution.
///   NLPD is homogeneous of degree ½ in the level maps, so twice each
///   sample's `x·∂NLPD/∂x` is an exact share,
///   `NLPD^0.4 · rms_l^-1.4 · x/(3·N_l)`; each level's shares spread evenly
///   over the input pixels their cells cover, scaled by the pixel count, so
///   **the map's mean is NLPD** (computed from the stored `f32` levels).
///   The `rms_l` normalization makes its support global.
/// - One map per pyramid level, `level_0` (full resolution) … `level_5`.
///   Each value is the squared difference of the two images' normalized
///   Laplacian responses at that level, summed over R, G and B — the
///   per-pixel term whose level mean feeds the score
///   (`rms_l = sqrt(mean/3)`, `NLPD = (Σ_l rms_l^0.6)^(1/0.6)`). These are
///   per-level components, not a per-pixel NLPD.
#[derive(Debug, Default, Clone, Copy)]
pub struct Nlpd;

/// Map names, finest level first.
const LEVEL_NAMES: [&str; LEVELS] = [
    "level_0", "level_1", "level_2", "level_3", "level_4", "level_5",
];

/// Level-`l` geometry for a `w × h` input. Sample `k` of a level sits at
/// input coordinate `a_l + 2^l·k`: the stride-2 5-tap downsample centres
/// its output `k` on parent sample `2k + 1` when the parent extent is
/// even (pad 1) and `2k` when odd (pad 2), so `a_l = a_{l-1} + 2^{l-1}`
/// exactly when that parent extent is even. Each cell is the `2^l`-wide
/// block centred on its sample. A value depends on the level image within
/// eight level samples (blur 2, the align-corners upsample and the 5-tap
/// downsample behind it, the divisive normalization's neighbour), and a
/// level sample on its input within `2^{l+1} − 2` pixels: the halo
/// `10·2^l − 2` bounds both.
fn level_geometry(width: usize, height: usize) -> [za::MapGeometry; LEVELS] {
    let (mut w, mut h) = (width, height);
    let (mut ax, mut ay) = (0i64, 0i64);
    core::array::from_fn(|l| {
        if l > 0 {
            let half = 1i64 << (l - 1);
            if w % 2 == 0 {
                ax += half;
            }
            if h % 2 == 0 {
                ay += half;
            }
            (w, h) = (w.div_ceil(2), h.div_ceil(2));
        }
        let cell = 1u32 << l;
        let halo = 10 * cell - 2;
        za::MapGeometry::grid(
            width as u32,
            height as u32,
            ax - i64::from(cell / 2),
            ay - i64::from(cell / 2),
            cell,
            cell,
            w as u32,
            h as u32,
        )
        .with_halo(halo, halo)
    })
}

/// Fuses per-cell shares from several grids into one map: each share
/// spreads evenly over the input pixels its cell covers, the map is cropped
/// to the pixels some cell covers, and scaled so its mean is the sum of all
/// shares.
fn fuse(parts: &[(Vec<f64>, za::MapGeometry)], w: u32, h: u32) -> (Vec<f32>, za::MapGeometry) {
    let n = (w * h) as usize;
    let (mut acc, mut covered) = (vec![0f64; n], vec![false; n]);
    for (shares, g) in parts {
        let cell = |x, y| g.cell_at(x, y).map(|(i, j)| (j * g.cells_x + i) as usize);
        let mut count = vec![0u32; shares.len()];
        for y in 0..h {
            for x in 0..w {
                if let Some(k) = cell(x, y) {
                    count[k] += 1;
                }
            }
        }
        for y in 0..h {
            for x in 0..w {
                if let Some(k) = cell(x, y) {
                    acc[(y * w + x) as usize] += shares[k] / f64::from(count[k]);
                    covered[(y * w + x) as usize] = true;
                }
            }
        }
    }
    let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
    for y in 0..h {
        for x in 0..w {
            if covered[(y * w + x) as usize] {
                (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x + 1), y1.max(y + 1));
            }
        }
    }
    if x0 >= x1 {
        (x0, y0, x1, y1) = (0, 0, w, h);
    }
    let (bw, bh) = (x1 - x0, y1 - y0);
    let scale = f64::from(bw) * f64::from(bh);
    let mut out = Vec::with_capacity((bw * bh) as usize);
    for y in y0..y1 {
        out.extend((x0..x1).map(|x| (acc[(y * w + x) as usize] * scale) as f32));
    }
    let g = za::MapGeometry::grid(w, h, i64::from(x0), i64::from(y0), 1, 1, bw, bh)
        .with_support(za::Support::Global);
    (out, g)
}

/// The `diffmap` described on [`Nlpd`]: per-sample shares of NLPD at input
/// resolution, mean = NLPD of the stored levels.
fn nlpd_diffmap(
    levels: &[Vec<f32>],
    geometry: &[za::MapGeometry; LEVELS],
) -> (Vec<f32>, za::MapGeometry) {
    let rms: Vec<f64> = levels
        .iter()
        .map(|l| (l.iter().map(|&v| f64::from(v)).sum::<f64>() / (l.len() * 3) as f64).sqrt())
        .collect();
    let nlpd = rms.iter().map(|r| r.powf(0.6)).sum::<f64>().powf(1.0 / 0.6);
    let parts: Vec<_> = levels
        .iter()
        .enumerate()
        .map(|(l, level)| {
            let unit = if rms[l] > 0.0 {
                nlpd.powf(0.4) * rms[l].powf(-1.4) / (3 * level.len()) as f64
            } else {
                0.0
            };
            (
                level.iter().map(|&v| unit * f64::from(v)).collect(),
                geometry[l],
            )
        })
        .collect();
    let g0 = geometry[0];
    fuse(&parts, g0.input_width, g0.input_height)
}

/// A scored pair with its fused map and per-level maps.
struct NlpdMaps {
    nlpd: f64,
    diffmap: (Vec<f32>, za::MapGeometry),
    levels: Vec<Vec<f32>>,
    geometry: [za::MapGeometry; LEVELS],
}

impl NlpdMaps {
    /// `0` is the fused map, `1 + l` level `l`.
    fn view(&self, i: usize) -> Option<za::MapView<'_>> {
        let (name, data, g) = match i {
            0 => ("diffmap", &self.diffmap.0, self.diffmap.1),
            l => (
                LEVEL_NAMES[l - 1],
                &self.levels[l - 1],
                self.geometry[l - 1],
            ),
        };
        let px = za::gray_f32_slice(data, g.cells_x, g.cells_y).ok()?;
        za::MapView::new(name, px, za::MapPolarity::Difference, g).ok()
    }
}

impl za::Evaluation for NlpdMaps {
    fn score(&self) -> f64 {
        self.nlpd
    }

    fn scores(&self) -> za::Scores<'_> {
        Box::new(core::iter::once((
            std::borrow::Cow::Borrowed("nlpd"),
            self.nlpd,
        )))
    }

    fn map(&self) -> Option<za::MapView<'_>> {
        self.view(0)
    }

    fn maps(&self) -> Vec<za::MapView<'_>> {
        (0..=self.levels.len())
            .filter_map(|i| self.view(i))
            .collect()
    }

    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

impl Error {
    fn kind(&self) -> za::ErrorKind {
        match self {
            Error::InvalidDimensions => za::ErrorKind::TooSmall,
            Error::InvalidLength { .. } => za::ErrorKind::InvalidInput,
            Error::NonFiniteInput => za::ErrorKind::InvalidInput,
        }
    }
}

/// Output revision of the generic-surface scorer(s) in this crate.
const SCORE_REVISION: u32 = 1;

impl za::Metric for Nlpd {
    fn id(&self) -> za::MetricId {
        za::MetricId::new("nlpd", za::implementation!(SCORE_REVISION))
            .with_reference("Laparra et al. NLPD, authors' PyTorch six-level RGB configuration")
    }

    fn outputs(&self) -> std::borrow::Cow<'static, [za::ScoreSpec]> {
        const OUTPUTS: &[za::ScoreSpec] = &[za::ScoreSpec::new("nlpd", za::ScoreKind::Difference)];
        std::borrow::Cow::Borrowed(OUTPUTS)
    }

    fn accepts(&self, format: &za::InputFormat) -> bool {
        za::is_rgb8_srgb(format)
    }

    fn evaluate(
        &mut self,
        r: za::Image<'_>,
        d: za::Image<'_>,
        opts: za::EvalOpts<'_>,
    ) -> za::MetricResult<std::boxed::Box<dyn za::Evaluation>> {
        use za::ResultMetricExt as _;
        opts.checkpoint()?;
        let (r, d) = za::rgb8_pair(&r, &d)?;
        let (rp, dp) = (za::contiguous_rgb8(&r), za::contiguous_rgb8(&d));
        if let Some(req) = opts.map {
            let (nlpd, levels) = score_rgb_u8_maps(&rp, &dp, r.width, r.height)
                .metric_err_with("nlpd", Error::kind)?;
            opts.checkpoint()?;
            let geometry = level_geometry(r.width, r.height);
            let eval = NlpdMaps {
                nlpd,
                diffmap: nlpd_diffmap(&levels, &geometry),
                levels,
                geometry,
            };
            return za::apply_map_request(Box::new(eval), req);
        }
        let s = score_rgb_u8(&rp, &dp, r.width, r.height).metric_err_with("nlpd", Error::kind)?;
        Ok(std::boxed::Box::new(za::ScalarEval::single("nlpd", s)))
    }
}
