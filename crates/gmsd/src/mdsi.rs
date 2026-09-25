//! MDSI, the Mean Deviation Similarity Index (Nafchi, Shahkolaei, Hedjam and
//! Cheriet, "Mean Deviation Similarity Index: Efficient and Reliable
//! Full-Reference Image Quality Evaluator", IEEE Access 4, 2016).
//!
//! Written from the paper alone. Where the paper is silent or ambiguous the
//! reading taken is recorded in `crates/gmsd/docs/MDSI_CHOICES.md`, with the
//! paper sentence each choice rests on. The implementation was validated
//! against scores produced by the authors' reference software (not included
//! here); see the crate README.
//!
//! # Algorithm (paper section numbers)
//!
//! 1. **III-H**: average-filter each RGB channel with an `M×M` box and keep
//!    every `M`-th sample, `M = max(1, round(min(h, w) / 256))`.
//! 2. **II**: luminance `L = 0.2989 R + 0.5870 G + 0.1140 B` and the two
//!    chromaticity channels `H`, `M` of the Gaussian colour model (eq. 1).
//! 3. **II-A/B**: Prewitt gradient magnitude of `L` for the reference, the
//!    distorted image and their fusion `F = (L_R + L_D) / 2`; the proposed
//!    gradient similarity `GS^c = GS_RD + (GS_DF - GS_RF)` (eqs. 2-5).
//! 4. **II-C**: joint chromaticity similarity (eq. 7).
//! 5. `GCS = α·GS^c + (1 - α)·CS` (eq. 9), `α = 0.6`, `C1 = 140`, `C2 = 55`,
//!    `C3 = 550` (III-F).
//! 6. **II-D**: deviation pooling with `ρ = 1`, `q = o = 1/4` (eq. 13):
//!    `MDSI = ( mean |GCS^¼ - mean(GCS^¼)| )^¼`. 0 = identical, larger = worse.
//!
//! # Structure and exactness
//!
//! The averaged planes are at most about `256` samples on the short side, so
//! the full-resolution work is the box average alone. It is done in exact
//! integer arithmetic (vertical `u16`/`u32` window sums, then horizontal
//! window sums) and the single division by `M²` is the only rounding, exactly
//! as in [`reference`]. The similarity maps run eight f64 pixels at a time in
//! every tier with the operation order of [`reference`] and no FMA; the
//! fourth roots and the two pooling sums are evaluated in pixel order. The
//! score is therefore bit-identical to [`reference`] (the straight-line
//! transcription of the paper's equations that the tests pin it to), at
//! every tier and every thread count.

use alloc::vec;
use alloc::vec::Vec;
#[cfg(target_arch = "aarch64")]
use archmage::NeonToken;
#[cfg(target_arch = "wasm32")]
use archmage::Wasm128Token;
#[cfg(target_arch = "x86_64")]
use archmage::X64V3Token;
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
use archmage::X64V4Token;
use archmage::magetypes;

use crate::{BAND_ROWS, Error, Result, check_rgb8};

/// The paper's constants (section III-F).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Params {
    /// Gradient-similarity stability constant `C1` (eq. 2).
    pub(crate) c1: f64,
    /// Fused-image gradient-similarity constant `C2` (eqs. 3, 4).
    pub(crate) c2: f64,
    /// Chromaticity-similarity constant `C3` (eq. 7).
    pub(crate) c3: f64,
    /// Weight `α` of the gradient similarity in eq. 9.
    pub(crate) alpha: f64,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            c1: 140.0,
            c2: 55.0,
            c3: 550.0,
            alpha: 0.6,
        }
    }
}

/// `M = max(1, round(min(h, w) / 256))`, rounding half away from zero
/// (exact for positive integers). Images below 128 px on the short side are
/// not averaged.
pub(crate) fn downsample_factor(width: usize, height: usize) -> usize {
    let s = width.min(height);
    (s / 256 + usize::from(s % 256 >= 128)).max(1)
}

/// `⌊(M-1)/2⌋`: the box window of output sample `i` starts at `i·M - this`.
fn window_lead(m: usize) -> usize {
    (m - 1) / 2
}

/// Window `[start, end)` of output sample `i` along an axis of length `n`,
/// clipped to the image (samples outside count as zero).
#[inline(always)]
fn window(i: usize, m: usize, n: usize) -> (usize, usize) {
    let a = i * m;
    let lo = window_lead(m);
    let start = a.saturating_sub(lo);
    let end = (a + m).saturating_sub(lo).min(n);
    (start.min(end), end)
}

/// Correctly rounded square root in every build (the crate-level `no_std`
/// helper is a Newton iteration that can be one ulp off).
fn sqrt(v: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        v.sqrt()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::sqrt(v)
    }
}

fn quarter_pow(v: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        // Defensive pin: a runtime exponent keeps LLVM from rewriting the libm call by
        // context (it does rewrite 0.5 -> sqrt). No 0.25 rewrite has been observed here.
        v.powf(core::hint::black_box(0.25))
    }
    #[cfg(not(feature = "std"))]
    {
        libm::pow(v, 0.25)
    }
}

fn sin_cos_pi_4() -> (f64, f64) {
    #[cfg(feature = "std")]
    {
        core::hint::black_box(core::f64::consts::FRAC_PI_4).sin_cos()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::sincos(core::f64::consts::FRAC_PI_4)
    }
}

/// Principal fourth root as `(re, im)`: real for `v ≥ 0`, `|v|^¼·e^{iπ/4}`
/// for `v < 0`. `sc` is `sin_cos_pi_4()`.
#[inline]
fn fourth_root(v: f64, sc: (f64, f64)) -> (f64, f64) {
    if v >= 0.0 {
        (quarter_pow(v), 0.0)
    } else {
        let r = quarter_pow(-v);
        (r * sc.1, r * sc.0)
    }
}

// ---------------------------------------------------------------------------
// The straight-line implementation of the paper's equations.
// ---------------------------------------------------------------------------

/// Straight-line scalar evaluation of the paper's equations. The optimized
/// path is pinned to this bit for bit; it is also the negative-control
/// vehicle (constants are parameters).
#[allow(dead_code)]
pub(crate) mod reference {
    use super::*;

    /// One channel of a strided RGB8 image: `M×M` box average (zero padded),
    /// sampled at `i·M`; output is `⌈w/M⌉ × ⌈h/M⌉`. Window sums are integers,
    /// exact in f64; the one division by `M²` is the only rounding.
    pub(crate) fn box_downsample(
        rgb: &[u8],
        channel: usize,
        width: usize,
        height: usize,
        stride: usize,
        m: usize,
    ) -> (Vec<f64>, usize, usize) {
        let (ow, oh) = (width.div_ceil(m), height.div_ceil(m));
        let lo = ((m - 1) / 2) as isize;
        let range = |i: usize, n: usize| -> (usize, usize) {
            let a = (i * m) as isize - lo;
            let b = a + m as isize;
            (a.max(0) as usize, (b.min(n as isize)).max(0) as usize)
        };
        let mut hsum = vec![0.0f64; height * ow];
        for y in 0..height {
            let row = &rgb[y * stride..y * stride + 3 * width];
            for x in 0..ow {
                let (a, b) = range(x, width);
                let mut s = 0u32;
                for xx in a..b {
                    s += row[3 * xx + channel] as u32;
                }
                hsum[y * ow + x] = s as f64;
            }
        }
        let norm = (m * m) as f64;
        let mut out = vec![0.0f64; ow * oh];
        for y in 0..oh {
            let (a, b) = range(y, height);
            for x in 0..ow {
                let mut s = 0.0;
                for yy in a..b {
                    s += hsum[yy * ow + x];
                }
                out[y * ow + x] = s / norm;
            }
        }
        (out, ow, oh)
    }

    /// Prewitt gradient magnitude: `[1 0 -1]/3` family, zero padded.
    pub(crate) fn prewitt_magnitude(l: &[f64], w: usize, h: usize) -> Vec<f64> {
        let t = 1.0 / 3.0;
        let at = |y: isize, x: isize| -> f64 {
            if y < 0 || x < 0 || y >= h as isize || x >= w as isize {
                0.0
            } else {
                l[y as usize * w + x as usize]
            }
        };
        let mut g = vec![0.0f64; w * h];
        for y in 0..h as isize {
            for x in 0..w as isize {
                let mut gx = 0.0;
                let mut gy = 0.0;
                for k in -1..=1isize {
                    gx += at(y + k, x + 1) - at(y + k, x - 1);
                    gy += at(y + 1, x + k) - at(y - 1, x + k);
                }
                gx *= t;
                gy *= t;
                g[y as usize * w + x as usize] = sqrt(gx * gx + gy * gy);
            }
        }
        g
    }

    /// The whole index with explicit constants. Returns `(score, gcs)`.
    pub(crate) fn score_and_map(
        reference: &[u8],
        distorted: &[u8],
        width: usize,
        height: usize,
        stride_bytes: usize,
        p: &Params,
    ) -> Result<(f64, Vec<f64>)> {
        if width == 0 || height == 0 {
            return Err(Error::TooSmall { width, height });
        }
        check_rgb8(reference, width, height, stride_bytes)?;
        check_rgb8(distorted, width, height, stride_bytes)?;

        let m = downsample_factor(width, height);
        let mut r_ch: Vec<Vec<f64>> = Vec::with_capacity(3);
        let mut d_ch: Vec<Vec<f64>> = Vec::with_capacity(3);
        let (mut w, mut h) = (0, 0);
        for c in 0..3 {
            let (a, ow, oh) = box_downsample(reference, c, width, height, stride_bytes, m);
            let (b, _, _) = box_downsample(distorted, c, width, height, stride_bytes, m);
            (w, h) = (ow, oh);
            r_ch.push(a);
            d_ch.push(b);
        }
        let n = w * h;
        let map = |ch: &[Vec<f64>], k: [f64; 3]| -> Vec<f64> {
            (0..n)
                .map(|i| k[0] * ch[0][i] + k[1] * ch[1][i] + k[2] * ch[2][i])
                .collect()
        };
        // Eq. (1) and the luminance formula.
        let lum = [0.2989, 0.5870, 0.1140];
        let hk = [0.30, 0.04, -0.35];
        let mk = [0.34, -0.60, 0.17];
        let (l_r, l_d) = (map(&r_ch, lum), map(&d_ch, lum));
        let (h_r, h_d) = (map(&r_ch, hk), map(&d_ch, hk));
        let (m_r, m_d) = (map(&r_ch, mk), map(&d_ch, mk));
        let l_f: Vec<f64> = (0..n).map(|i| 0.5 * (l_r[i] + l_d[i])).collect();

        let g_r = prewitt_magnitude(&l_r, w, h);
        let g_d = prewitt_magnitude(&l_d, w, h);
        let g_f = prewitt_magnitude(&l_f, w, h);

        let sim = |a: f64, b: f64, c: f64| (2.0 * a * b + c) / (a * a + b * b + c);
        let sc = sin_cos_pi_4();
        let mut gcs_map = Vec::with_capacity(n);
        let mut vals: Vec<(f64, f64)> = Vec::with_capacity(n);
        for i in 0..n {
            let gs = sim(g_r[i], g_d[i], p.c1); // eq. 2
            let gs_rf = sim(g_r[i], g_f[i], p.c2); // eq. 3
            let gs_df = sim(g_d[i], g_f[i], p.c2); // eq. 4
            let gs_c = gs + (gs_df - gs_rf); // eq. 5
            let cs = (2.0 * (h_r[i] * h_d[i] + m_r[i] * m_d[i]) + p.c3)
                / (h_r[i] * h_r[i] + h_d[i] * h_d[i] + m_r[i] * m_r[i] + m_d[i] * m_d[i] + p.c3); // eq. 7
            let gcs = p.alpha * gs_c + (1.0 - p.alpha) * cs; // eq. 9
            gcs_map.push(gcs);
            vals.push(fourth_root(gcs, sc));
        }

        // Eq. (13): (mean |x_i - mean(x)|)^(1/4), complex-valued x_i.
        let nn = n as f64;
        let mean_re = vals.iter().map(|v| v.0).sum::<f64>() / nn;
        let mean_im = vals.iter().map(|v| v.1).sum::<f64>() / nn;
        let dev = vals
            .iter()
            .map(|v| {
                let (dr, di) = (v.0 - mean_re, v.1 - mean_im);
                sqrt(dr * dr + di * di)
            })
            .sum::<f64>()
            / nn;
        Ok((quarter_pow(dev), gcs_map))
    }
}

// ---------------------------------------------------------------------------
// The optimized path.
// ---------------------------------------------------------------------------

const PLANES: usize = 7; // Lr, Ld, fused L, Hr, Hd, Mr, Md

/// The seven averaged planes, one zero halo row above and below, one zero
/// column left, and slack on the right for the last eight-wide vector. Layout
/// `[halo row][plane][pitch]`, so a band of rows is one contiguous chunk.
pub(crate) struct Planes {
    data: Vec<f64>,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pitch: usize,
}

impl Planes {
    #[cfg(test)]
    fn row_len(&self) -> usize {
        PLANES * self.pitch
    }

    /// Wrap planes filled by a caller-chosen tier (tier-parity checks).
    #[allow(dead_code)]
    pub(crate) fn from_parts(data: Vec<f64>, s: &Source<'_>) -> Self {
        Planes {
            data,
            width: s.ow,
            height: s.height(),
            pitch: s.pitch,
        }
    }
}

/// The two images and the averaging geometry.
#[derive(Clone, Copy)]
pub(crate) struct Source<'a> {
    r: &'a [u8],
    d: &'a [u8],
    w: usize,
    h: usize,
    stride: usize,
    m: usize,
    ow: usize,
    pitch: usize,
}

impl Source<'_> {
    /// Rows of the averaged planes.
    #[allow(dead_code)]
    pub(crate) fn height(&self) -> usize {
        self.h.div_ceil(self.m)
    }

    /// Elements per halo row of the plane storage.
    #[allow(dead_code)]
    pub(crate) fn row_len(&self) -> usize {
        PLANES * self.pitch
    }
}

#[inline(always)]
fn vertical_sums<T>(rgb: &[u8], s: &Source<'_>, oy: usize, acc: &mut [T])
where
    T: Copy + core::ops::AddAssign + From<u8> + Default,
{
    acc.fill(T::default());
    let (ya, yb) = window(oy, s.m, s.h);
    for y in ya..yb {
        let row = &rgb[y * s.stride..y * s.stride + 3 * s.w];
        for (a, &p) in acc.iter_mut().zip(row) {
            *a += T::from(p);
        }
    }
}

/// Exact integer window sums of one output row, three channels interleaved.
#[inline(always)]
fn box_row_sums(
    rgb: &[u8],
    s: &Source<'_>,
    oy: usize,
    acc16: &mut [u16],
    acc32: &mut [u32],
    out: &mut [u64],
) {
    // `u16` holds up to 257 rows of 255; beyond that fall back to `u32`.
    if s.m <= 257 {
        vertical_sums(rgb, s, oy, acc16);
        for x in 0..s.ow {
            let (xa, xb) = window(x, s.m, s.w);
            let mut t = [0u64; 3];
            for xx in xa..xb {
                t[0] += u64::from(acc16[3 * xx]);
                t[1] += u64::from(acc16[3 * xx + 1]);
                t[2] += u64::from(acc16[3 * xx + 2]);
            }
            out[3 * x..3 * x + 3].copy_from_slice(&t);
        }
    } else {
        vertical_sums(rgb, s, oy, acc32);
        for x in 0..s.ow {
            let (xa, xb) = window(x, s.m, s.w);
            let mut t = [0u64; 3];
            for xx in xa..xb {
                t[0] += u64::from(acc32[3 * xx]);
                t[1] += u64::from(acc32[3 * xx + 1]);
                t[2] += u64::from(acc32[3 * xx + 2]);
            }
            out[3 * x..3 * x + 3].copy_from_slice(&t);
        }
    }
}

/// Averaging and colour transform for output rows `y0..`, written into
/// `chunk` (halo rows `y0 + 1..`). Shared body of every tier's entry.
#[inline(always)]
fn prepare_rows(s: &Source<'_>, y0: usize, chunk: &mut [f64]) {
    let row_len = PLANES * s.pitch;
    let mut acc16 = vec![0u16; if s.m <= 257 { 3 * s.w } else { 0 }];
    let mut acc32 = vec![0u32; if s.m > 257 { 3 * s.w } else { 0 }];
    let mut sr = vec![0u64; 3 * s.ow];
    let mut sd = vec![0u64; 3 * s.ow];
    let norm = (s.m * s.m) as f64;
    let lum = [0.2989, 0.5870, 0.1140];
    let hk = [0.30, 0.04, -0.35];
    let mk = [0.34, -0.60, 0.17];
    for (j, row) in chunk.chunks_exact_mut(row_len).enumerate() {
        let oy = y0 + j;
        box_row_sums(s.r, s, oy, &mut acc16, &mut acc32, &mut sr);
        box_row_sums(s.d, s, oy, &mut acc16, &mut acc32, &mut sd);
        for x in 0..s.ow {
            let a = [
                sr[3 * x] as f64 / norm,
                sr[3 * x + 1] as f64 / norm,
                sr[3 * x + 2] as f64 / norm,
            ];
            let b = [
                sd[3 * x] as f64 / norm,
                sd[3 * x + 1] as f64 / norm,
                sd[3 * x + 2] as f64 / norm,
            ];
            let dot = |c: &[f64; 3], k: [f64; 3]| k[0] * c[0] + k[1] * c[1] + k[2] * c[2];
            let (lr, ld) = (dot(&a, lum), dot(&b, lum));
            let values = [
                lr,
                ld,
                0.5 * (lr + ld),
                dot(&a, hk),
                dot(&b, hk),
                dot(&a, mk),
                dot(&b, mk),
            ];
            for (p, v) in values.into_iter().enumerate() {
                row[p * s.pitch + x + 1] = v;
            }
        }
    }
}

macro_rules! prepare_entry {
    ($name:ident, $token:ident) => {
        #[archmage::arcane]
        pub(crate) fn $name(token: $token, s: &Source<'_>, y0: usize, chunk: &mut [f64]) {
            let _ = token;
            prepare_rows(s, y0, chunk);
        }
    };
}
#[cfg(target_arch = "x86_64")]
prepare_entry!(prepare_band_v3, X64V3Token);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
prepare_entry!(prepare_band_v4, X64V4Token);
#[cfg(target_arch = "aarch64")]
prepare_entry!(prepare_band_neon, NeonToken);
#[cfg(target_arch = "wasm32")]
prepare_entry!(prepare_band_wasm128, Wasm128Token);
pub(crate) fn prepare_band_scalar(
    token: archmage::ScalarToken,
    s: &Source<'_>,
    y0: usize,
    chunk: &mut [f64],
) {
    let _ = token;
    prepare_rows(s, y0, chunk);
}

pub(crate) fn source<'a>(
    r: &'a [u8],
    d: &'a [u8],
    w: usize,
    h: usize,
    stride: usize,
) -> Source<'a> {
    source_with_factor(r, d, w, h, stride, downsample_factor(w, h))
}

/// As [`source`] with an explicit averaging factor (tests exercise factors
/// far above what real sizes produce).
pub(crate) fn source_with_factor<'a>(
    r: &'a [u8],
    d: &'a [u8],
    w: usize,
    h: usize,
    stride: usize,
    m: usize,
) -> Source<'a> {
    let ow = w.div_ceil(m);
    Source {
        r,
        d,
        w,
        h,
        stride,
        m,
        ow,
        pitch: ow + 10,
    }
}

/// Averaged planes with the production dispatch.
pub(crate) fn prepare(s: &Source<'_>) -> Planes {
    let oh = s.h.div_ceil(s.m);
    let row_len = PLANES * s.pitch;
    let mut data = vec![0.0f64; (oh + 2) * row_len];
    let run = |i: usize, chunk: &mut [f64]| {
        archmage::incant!(
            prepare_band(s, i * BAND_ROWS, chunk),
            [v4, v3, neon, wasm128, scalar]
        );
    };
    let body = &mut data[row_len..(oh + 1) * row_len];
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        body.par_chunks_mut(BAND_ROWS * row_len)
            .enumerate()
            .for_each(|(i, chunk)| run(i, chunk));
    }
    #[cfg(not(feature = "parallel"))]
    for (i, chunk) in body.chunks_mut(BAND_ROWS * row_len).enumerate() {
        run(i, chunk);
    }
    Planes {
        data,
        width: s.ow,
        height: oh,
        pitch: s.pitch,
    }
}

/// Eight independent f64 pixels in every tier; the operation order is that
/// of [`reference`], with no FMA and no lane reduction.
#[magetypes(rite, define(f64x8), v4, v3, neon, wasm128, scalar)]
fn similarity_row(
    token: Token,
    data: &[f64],
    pitch: usize,
    width: usize,
    y: usize,
    p: &Params,
    gcs: &mut [f64],
    cs: &mut [f64],
) {
    let zero = f64x8::splat(token, 0.0);
    let t = f64x8::splat(token, 1.0 / 3.0);
    let two = f64x8::splat(token, 2.0);
    let c1 = f64x8::splat(token, p.c1);
    let c2 = f64x8::splat(token, p.c2);
    let c3 = f64x8::splat(token, p.c3);
    let alpha = f64x8::splat(token, p.alpha);
    let beta = f64x8::splat(token, 1.0 - p.alpha);
    for x in (0..width).step_by(8) {
        let count = (width - x).min(8);
        // Sample of `plane` at halo row `y + r`, column `x + c`.
        macro_rules! sample {
            ($plane:expr, $r:expr, $c:expr) => {{
                let i = ((y + $r) * PLANES + $plane) * pitch + x + $c;
                let a: [f64; 8] = data[i..i + 8].try_into().unwrap();
                f64x8::from_array(token, a)
            }};
        }
        macro_rules! gradient {
            ($plane:expr) => {{
                let mut gx = zero + (sample!($plane, 0, 2) - sample!($plane, 0, 0));
                gx += sample!($plane, 1, 2) - sample!($plane, 1, 0);
                gx += sample!($plane, 2, 2) - sample!($plane, 2, 0);
                let mut gy = zero + (sample!($plane, 2, 0) - sample!($plane, 0, 0));
                gy += sample!($plane, 2, 1) - sample!($plane, 0, 1);
                gy += sample!($plane, 2, 2) - sample!($plane, 0, 2);
                gx *= t;
                gy *= t;
                (gx * gx + gy * gy).sqrt()
            }};
        }
        let g_r = gradient!(0);
        let g_d = gradient!(1);
        let g_f = gradient!(2);
        let gs = ((two * g_r) * g_d + c1) / ((g_r * g_r + g_d * g_d) + c1);
        let gs_rf = ((two * g_r) * g_f + c2) / ((g_r * g_r + g_f * g_f) + c2);
        let gs_df = ((two * g_d) * g_f + c2) / ((g_d * g_d + g_f * g_f) + c2);
        let gs_c = gs + (gs_df - gs_rf);
        let (h_r, h_d, m_r, m_d) = (
            sample!(3, 1, 1),
            sample!(4, 1, 1),
            sample!(5, 1, 1),
            sample!(6, 1, 1),
        );
        let cv = (two * (h_r * h_d + m_r * m_d) + c3)
            / ((((h_r * h_r + h_d * h_d) + m_r * m_r) + m_d * m_d) + c3);
        let q = alpha * gs_c + beta * cv;
        gcs[x..x + count].copy_from_slice(&q.to_array()[..count]);
        cs[x..x + count].copy_from_slice(&cv.to_array()[..count]);
    }
}

macro_rules! maps_entry {
    ($name:ident, $token:ident, $row:ident) => {
        #[archmage::arcane]
        pub(crate) fn $name(
            token: $token,
            pl: &Planes,
            y0: usize,
            p: &Params,
            gcs: &mut [f64],
            cs: &mut [f64],
        ) {
            for (row, (q, c)) in gcs
                .chunks_mut(pl.width)
                .zip(cs.chunks_mut(pl.width))
                .enumerate()
            {
                $row(token, &pl.data, pl.pitch, pl.width, y0 + row, p, q, c);
            }
        }
    };
}
#[cfg(target_arch = "x86_64")]
maps_entry!(maps_band_v3, X64V3Token, similarity_row_v3);
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
maps_entry!(maps_band_v4, X64V4Token, similarity_row_v4);
#[cfg(target_arch = "aarch64")]
maps_entry!(maps_band_neon, NeonToken, similarity_row_neon);
#[cfg(target_arch = "wasm32")]
maps_entry!(maps_band_wasm128, Wasm128Token, similarity_row_wasm128);
pub(crate) fn maps_band_scalar(
    token: archmage::ScalarToken,
    pl: &Planes,
    y0: usize,
    p: &Params,
    gcs: &mut [f64],
    cs: &mut [f64],
) {
    for (row, (q, c)) in gcs
        .chunks_mut(pl.width)
        .zip(cs.chunks_mut(pl.width))
        .enumerate()
    {
        similarity_row_scalar(token, &pl.data, pl.pitch, pl.width, y0 + row, p, q, c);
    }
}

/// `(GCS, CS)` maps, row-major, with the production dispatch.
pub(crate) fn maps(pl: &Planes, p: &Params) -> (Vec<f64>, Vec<f64>) {
    let n = pl.width * pl.height;
    let mut gcs = vec![0.0; n];
    let mut cs = vec![0.0; n];
    let run = |i: usize, q: &mut [f64], c: &mut [f64]| {
        archmage::incant!(
            maps_band(pl, i * BAND_ROWS, p, q, c),
            [v4, v3, neon, wasm128, scalar]
        );
    };
    let band = BAND_ROWS * pl.width;
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        gcs.par_chunks_mut(band)
            .zip(cs.par_chunks_mut(band))
            .enumerate()
            .for_each(|(i, (q, c))| run(i, q, c));
    }
    #[cfg(not(feature = "parallel"))]
    for (i, (q, c)) in gcs.chunks_mut(band).zip(cs.chunks_mut(band)).enumerate() {
        run(i, q, c);
    }
    (gcs, cs)
}

/// Eq. (13) over a GCS map: fourth roots per pixel (independent, so banded),
/// then the two sums in pixel order.
pub(crate) fn pool(gcs: &[f64]) -> f64 {
    let n = gcs.len();
    let sc = sin_cos_pi_4();
    let mut re = vec![0.0f64; n];
    let mut im = vec![0.0f64; n];
    let band = BAND_ROWS * 64;
    let roots = |g: &[f64], r: &mut [f64], i: &mut [f64]| {
        for ((g, r), i) in g.iter().zip(r.iter_mut()).zip(i.iter_mut()) {
            (*r, *i) = fourth_root(*g, sc);
        }
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        gcs.par_chunks(band)
            .zip(re.par_chunks_mut(band))
            .zip(im.par_chunks_mut(band))
            .for_each(|((g, r), i)| roots(g, r, i));
    }
    #[cfg(not(feature = "parallel"))]
    for ((g, r), i) in gcs
        .chunks(band)
        .zip(re.chunks_mut(band))
        .zip(im.chunks_mut(band))
    {
        roots(g, r, i);
    }
    let nn = n as f64;
    let mean_re = re.iter().sum::<f64>() / nn;
    let mean_im = im.iter().sum::<f64>() / nn;
    let mut dev = vec![0.0f64; n];
    let deviations = |r: &[f64], i: &[f64], d: &mut [f64]| {
        for ((r, i), d) in r.iter().zip(i).zip(d.iter_mut()) {
            let (dr, di) = (*r - mean_re, *i - mean_im);
            *d = sqrt(dr * dr + di * di);
        }
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        re.par_chunks(band)
            .zip(im.par_chunks(band))
            .zip(dev.par_chunks_mut(band))
            .for_each(|((r, i), d)| deviations(r, i, d));
    }
    #[cfg(not(feature = "parallel"))]
    for ((r, i), d) in re
        .chunks(band)
        .zip(im.chunks(band))
        .zip(dev.chunks_mut(band))
    {
        deviations(r, i, d);
    }
    quarter_pow(dev.iter().sum::<f64>() / nn)
}

/// Score with explicit constants (negative controls); `(score, gcs, cs)`.
pub(crate) fn score_with(
    r: &[u8],
    d: &[u8],
    w: usize,
    h: usize,
    stride: usize,
    p: &Params,
) -> Result<(f64, Vec<f64>, Vec<f64>)> {
    if w == 0 || h == 0 {
        return Err(Error::TooSmall {
            width: w,
            height: h,
        });
    }
    check_rgb8(r, w, h, stride)?;
    check_rgb8(d, w, h, stride)?;
    let planes = prepare(&source(r, d, w, h, stride));
    let (gcs, cs) = maps(&planes, p);
    Ok((pool(&gcs), gcs, cs))
}

pub(crate) fn run(r: &[u8], d: &[u8], w: usize, h: usize, stride: usize) -> Result<f64> {
    score_with(r, d, w, h, stride, &Params::default()).map(|s| s.0)
}

#[cfg(test)]
#[path = "mdsi_tests.rs"]
mod tests;
