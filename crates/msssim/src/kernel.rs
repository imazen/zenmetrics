//! MS-SSIM pipeline internals — separable `'valid'` Gaussian
//! statistics per level, `imfilter`-compatible 2×2 downsample, and the
//! weighted-product combine. Scalar helpers (`decimate`,
//! `gaussian_taps`) are shared across tiers; `score_body!` vectorizes
//! the horizontal/vertical stencils and the map+mean stage through
//! `magetypes`, so every tier produces bit-identical output.
//!
//! Provenance: `msssim.m` + `ssim_index_new.m` (Wang, Simoncelli &
//! Bovik). See `validation/README.md`.

use alloc::vec;
use alloc::vec::Vec;

/// `fspecial('gaussian', 11, 1.5)` rank-1 taps, normalised to sum 1.
/// The 2-D window is `outer(k,k)` — `win/sum(win)` in the reference is
/// exactly this since `exp(−(i²+j²)/2σ²)` is separable.
const GAUSS_K: usize = 11;
const GAUSS_HALF: usize = 5;
/// `C1 = (0.01·255)²`, `C2 = (0.03·255)²` — the reference's defaults,
/// on the 0–255 signal scale.
const C1: f32 = 6.5025;
const C2: f32 = 58.5225;
/// Canonical 5-level weights (`msssim.m`); the first `level` entries
/// are used.
const WEIGHTS: [f64; 5] = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333];

/// 1-D Gaussian taps `exp(−i²/(2·1.5²))` for `i ∈ −5..5`, normalised
/// to sum 1. Computed in f64 like `fspecial`, then cast.
fn gaussian_taps() -> [f32; GAUSS_K] {
    let mut g = [0.0f64; GAUSS_K];
    let mut sum = 0.0f64;
    for (i, v) in g.iter_mut().enumerate() {
        let x = i as f64 - GAUSS_HALF as f64;
        *v = libm::exp(-(x * x) / (2.0 * 1.5 * 1.5));
        sum += *v;
    }
    core::array::from_fn(|i| (g[i] / sum) as f32)
}

/// `imfilter(im, ones(2)/4, 'symmetric', 'same')` then `im(1:2:end,
/// 1:2:end)` — forward 2×2 box with whole-point symmetric border
/// (`x(n+1) → x(n)`), verified against Octave.
fn decimate(x: &[f32], w: usize, h: usize) -> (usize, usize, Vec<f32>) {
    let (w2, h2) = (w.div_ceil(2), h.div_ceil(2));
    let mut out = vec![0.0f32; w2 * h2];
    for_each_row(&mut out, w2, |oi, orow| {
        let i0 = oi * 2;
        let i1 = (i0 + 1).min(h - 1);
        for (oj, o) in orow.iter_mut().enumerate() {
            let j0 = oj * 2;
            let j1 = (j0 + 1).min(w - 1);
            *o = ((x[i0 * w + j0] + x[i0 * w + j1]) + (x[i1 * w + j0] + x[i1 * w + j1])) * 0.25;
        }
    });
    (w2, h2, out)
}

/// Row-map driver — `f(y, out_row)` for every row. Under `parallel`
/// rows are `par_chunks_mut`-mapped when the pool has >1 thread;
/// otherwise the serial loop runs. Writes are disjoint per row either
/// way, so results are bit-identical at any thread count.
pub(crate) fn for_each_row(out: &mut [f32], row_w: usize, f: impl Fn(usize, &mut [f32]) + Sync) {
    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 {
        use rayon::prelude::*;
        out.par_chunks_mut(row_w)
            .enumerate()
            .for_each(|(y, drow)| f(y, drow));
        return;
    }
    for (y, drow) in out.chunks_mut(row_w).enumerate() {
        f(y, drow);
    }
}

/// `rayon::join` under `parallel`, sequential otherwise — each side
/// writes its own `Vec`, so results are identical at any thread count.
pub(crate) fn maybe_join<A, B>(
    fa: impl FnOnce() -> A + Send,
    fb: impl FnOnce() -> B + Send,
) -> (A, B)
where
    A: Send,
    B: Send,
{
    #[cfg(feature = "parallel")]
    {
        if rayon::current_num_threads() <= 1 {
            return (fa(), fb());
        }
        rayon::join(fa, fb)
    }
    #[cfg(not(feature = "parallel"))]
    {
        (fa(), fb())
    }
}

/// Arguments to one `msssim` run, bundled so `attuned!` dispatches a
/// single parameter.
#[doc(hidden)]
pub struct Args<'a> {
    /// Level-0 reference plane, packed `w·h`.
    pub im1: &'a [f32],
    /// Level-0 distorted plane, packed `w·h`.
    pub im2: &'a [f32],
    /// Width × height of the level-0 planes.
    pub w: usize,
    /// Height.
    pub h: usize,
    /// Levels (1..=5); weights `WEIGHTS[..level]` are applied.
    pub level: usize,
}

// ============================ tiered core ================================

/// The full pipeline for one `(im1, im2, w, h, level)`. Vector lanes
/// hold consecutive output elements; every per-element expression is a
/// fixed-order `f32` chain, and the two map means accumulate `f64` in
/// raster order — bit-identical on every tier and lane count.
macro_rules! score_body {
    ($token:ident, $F32:ident, $LANES:literal, $a:ident) => {{
        let (mut w, mut h, level) = ($a.w, $a.h, $a.level);
        let taps = gaussian_taps();
        let (mut im1, mut im2) = (
            alloc::borrow::Cow::Borrowed($a.im1),
            alloc::borrow::Cow::Borrowed($a.im2),
        );
        let mut mcs = [0.0f64; 5];
        let mut mssim_last = 0.0f64;
        let (c1v, c2v) = (<$F32>::splat_t($token, C1), <$F32>::splat_t($token, C2));
        let twov = <$F32>::splat_t($token, 2.0);

        for l in 0..level {
            let out_w = w - GAUSS_K + 1;
            let out_h = h - GAUSS_K + 1;
            let n = out_w * out_h;
            let mut smap = vec![0.0f32; n];
            let mut cmap = vec![0.0f32; n];
            // Keep only eleven horizontal rows per moment. Products are
            // formed in the horizontal stencil, with the same f32 rounding
            // and tap order as the materialized product planes. Each band
            // owns its ring and output rows, including under Rayon.
            let band = |first_y: usize, sout: &mut [f32], cout: &mut [f32]| {
                let mut ring: [Vec<f32>; 5] = core::array::from_fn(|_| vec![0.0; GAUSS_K * out_w]);
                let rows = sout.len() / out_w;
                for ry in 0..rows + GAUSS_K - 1 {
                    let slot = ry % GAUSS_K;
                    let base = (first_y + ry) * w;
                    let rrow = &im1[base..base + w];
                    let drow = &im2[base..base + w];
                    let mut x = 0;
                    while x + $LANES <= out_w {
                        let mut acc = [<$F32>::splat_t($token, 0.0); 5];
                        for u in 0..GAUSS_K {
                            let r = <$F32>::from_array_t(
                                $token,
                                core::array::from_fn(|k| rrow[x + u + k]),
                            );
                            let d = <$F32>::from_array_t(
                                $token,
                                core::array::from_fn(|k| drow[x + u + k]),
                            );
                            let tap = <$F32>::splat_t($token, taps[u]);
                            acc[0] += tap * r;
                            acc[1] += tap * d;
                            acc[2] += tap * (r * r);
                            acc[3] += tap * (d * d);
                            acc[4] += tap * (r * d);
                        }
                        for j in 0..5 {
                            ring[j][slot * out_w + x..slot * out_w + x + $LANES]
                                .copy_from_slice(&acc[j].to_array());
                        }
                        x += $LANES;
                    }
                    while x < out_w {
                        let mut acc = [0.0f32; 5];
                        for u in 0..GAUSS_K {
                            let (r, d) = (rrow[x + u], drow[x + u]);
                            let v = [r, d, r * r, d * d, r * d];
                            for j in 0..5 {
                                acc[j] += taps[u] * v[j];
                            }
                        }
                        for j in 0..5 {
                            ring[j][slot * out_w + x] = acc[j];
                        }
                        x += 1;
                    }
                    if ry < GAUSS_K - 1 {
                        continue;
                    }
                    let y = ry - (GAUSS_K - 1);
                    let mut x = 0;
                    while x + $LANES <= out_w {
                        let mut acc = [<$F32>::splat_t($token, 0.0); 5];
                        for u in 0..GAUSS_K {
                            let off = ((y + u) % GAUSS_K) * out_w + x;
                            let tap = <$F32>::splat_t($token, taps[u]);
                            for j in 0..5 {
                                let v = <$F32>::from_array_t(
                                    $token,
                                    core::array::from_fn(|k| ring[j][off + k]),
                                );
                                acc[j] += tap * v;
                            }
                        }
                        let [m1, m2, v1, v2, vc] = acc;
                        let s1sq = v1 - m1 * m1;
                        let s2sq = v2 - m2 * m2;
                        let s12 = vc - m1 * m2;
                        let n1 = twov * m1 * m2 + c1v;
                        let n2 = twov * s12 + c2v;
                        let d1 = m1 * m1 + m2 * m2 + c1v;
                        let d2 = s1sq + s2sq + c2v;
                        let off = y * out_w + x;
                        sout[off..off + $LANES]
                            .copy_from_slice(&((n1 * n2) / (d1 * d2)).to_array());
                        cout[off..off + $LANES].copy_from_slice(&(n2 / d2).to_array());
                        x += $LANES;
                    }
                    while x < out_w {
                        let mut acc = [0.0f32; 5];
                        for u in 0..GAUSS_K {
                            let off = ((y + u) % GAUSS_K) * out_w + x;
                            for j in 0..5 {
                                acc[j] += taps[u] * ring[j][off];
                            }
                        }
                        let [m1, m2, v1, v2, vc] = acc;
                        let s1sq = v1 - m1 * m1;
                        let s2sq = v2 - m2 * m2;
                        let s12 = vc - m1 * m2;
                        let n1 = 2.0 * m1 * m2 + C1;
                        let n2 = 2.0 * s12 + C2;
                        let d1 = m1 * m1 + m2 * m2 + C1;
                        let d2 = s1sq + s2sq + C2;
                        sout[y * out_w + x] = (n1 * n2) / (d1 * d2);
                        cout[y * out_w + x] = n2 / d2;
                        x += 1;
                    }
                }
            };
            #[cfg(feature = "parallel")]
            if rayon::current_num_threads() > 1 {
                use rayon::prelude::*;
                // Each worker computes its own halo; the final fold remains raster-ordered.
                let band_rows = out_h.div_ceil(rayon::current_num_threads());
                smap.par_chunks_mut(band_rows * out_w)
                    .zip(cmap.par_chunks_mut(band_rows * out_w))
                    .enumerate()
                    .for_each(|(b, (s, c))| band(b * band_rows, s, c));
            } else {
                band(0, &mut smap, &mut cmap);
            }
            #[cfg(not(feature = "parallel"))]
            band(0, &mut smap, &mut cmap);
            // Preserve the original raster-order f64 fold exactly.
            let (mut sum_s, mut sum_c) = (0.0f64, 0.0f64);
            for (&s, &c) in smap.iter().zip(&cmap) {
                sum_s += s as f64;
                sum_c += c as f64;
            }
            let npix = n as f64;
            mcs[l] = sum_c / npix;
            mssim_last = sum_s / npix;

            if l + 1 < level {
                let ((w2, h2, d1), (_, _, d2)) =
                    maybe_join(|| decimate(&im1, w, h), || decimate(&im2, w, h));
                im1 = alloc::borrow::Cow::Owned(d1);
                im2 = alloc::borrow::Cow::Owned(d2);
                w = w2;
                h = h2;
            }
        }

        // `prod(mcs(1:L−1).^w(1:L−1)) · mssim(L)^w(L)` — f64, fixed
        // weight order. Means can go negative; MATLAB's `x.^w` is then
        // complex (`|x|^w·e^{iπw}`), the product rotates by `π·Σw_neg`,
        // and the returned score is its real part — same `real(z^λ)`
        // semantics as `crates/vsi`'s chroma term.
        let (mut mag, mut neg_w) = (1.0f64, 0.0f64);
        for (l, &wt) in WEIGHTS.iter().enumerate().take(level - 1) {
            mag *= libm::pow(mcs[l].abs(), wt);
            if mcs[l] < 0.0 {
                neg_w += wt;
            }
        }
        mag *= libm::pow(mssim_last.abs(), WEIGHTS[level - 1]);
        if mssim_last < 0.0 {
            neg_w += WEIGHTS[level - 1];
        }
        mag * libm::cos(neg_w * core::f64::consts::PI)
    }};
}

/// The `v3` tier (AVX2/FMA, 8-wide f32).
#[cfg(target_arch = "x86_64")]
#[archmage::attune]
pub fn msssim_core_tier_v3_t(token: archmage::X64V3Token, a: Args<'_>) -> f64 {
    type V = magetypes::simd::generic::f32x8<archmage::X64V3Token>;
    score_body!(token, V, 8, a)
}

/// The `v4x` tier (AVX-512, 16-wide f32 — crate feature `avx512`).
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::attune]
pub fn msssim_core_tier_v4x_t(token: archmage::X64V4xToken, a: Args<'_>) -> f64 {
    type V = magetypes::simd::generic::f32x16<archmage::X64V4xToken>;
    score_body!(token, V, 16, a)
}

/// The `neon` tier (8-wide f32).
#[cfg(target_arch = "aarch64")]
#[archmage::attune]
pub fn msssim_core_tier_neon_t(token: archmage::NeonToken, a: Args<'_>) -> f64 {
    type V = magetypes::simd::generic::f32x8<archmage::NeonToken>;
    score_body!(token, V, 8, a)
}

/// The `wasm128` tier (8-wide f32).
#[cfg(target_arch = "wasm32")]
#[archmage::attune]
pub fn msssim_core_tier_wasm128_t(token: archmage::Wasm128Token, a: Args<'_>) -> f64 {
    type V = magetypes::simd::generic::f32x8<archmage::Wasm128Token>;
    score_body!(token, V, 8, a)
}

/// The `scalar` tier (8-wide f32, scalar-emulated lanes).
pub fn msssim_core_tier_scalar_t(token: archmage::ScalarToken, a: Args<'_>) -> f64 {
    type V = magetypes::simd::generic::f32x8<archmage::ScalarToken>;
    score_body!(token, V, 8, a)
}

/// Runtime-dispatched score: the best tier this CPU supports.
pub(crate) fn msssim_core(inputs: &crate::Inputs, w: usize, h: usize, level: usize) -> f64 {
    let a = Args {
        im1: &inputs.r,
        im2: &inputs.d,
        w,
        h,
        level,
    };
    archmage::attuned!(
        msssim_core_tier(a),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    )
}
