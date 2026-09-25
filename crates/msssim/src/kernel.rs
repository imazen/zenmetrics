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
    for oi in 0..h2 {
        let i0 = oi * 2;
        let i1 = (i0 + 1).min(h - 1);
        for oj in 0..w2 {
            let j0 = oj * 2;
            let j1 = (j0 + 1).min(w - 1);
            out[oi * w2 + oj] =
                ((x[i0 * w + j0] + x[i0 * w + j1]) + (x[i1 * w + j0] + x[i1 * w + j1])) * 0.25;
        }
    }
    (w2, h2, out)
}

/// Shared per-level elementwise products feeding the five `'valid'`
/// filters: `im1²`, `im2²`, `im1·im2`.
fn product_planes(im1: &[f32], im2: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut p1 = Vec::with_capacity(im1.len());
    let mut p2 = Vec::with_capacity(im1.len());
    let mut pp = Vec::with_capacity(im1.len());
    for (&a, &b) in im1.iter().zip(im2) {
        p1.push(a * a);
        p2.push(b * b);
        pp.push(a * b);
    }
    (p1, p2, pp)
}

/// `rayon::join` under `parallel`, sequential otherwise — each side
/// writes its own `Vec`, so results are identical at any thread count.
fn maybe_join<A, B>(fa: impl FnOnce() -> A + Send, fb: impl FnOnce() -> B + Send) -> (A, B)
where
    A: Send,
    B: Send,
{
    #[cfg(feature = "parallel")]
    {
        rayon::join(fa, fb)
    }
    #[cfg(not(feature = "parallel"))]
    {
        (fa(), fb())
    }
}

/// Arguments to one `msssim` run, bundled so `incant!` dispatches a
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
        let (mut im1, mut im2) = ($a.im1.to_vec(), $a.im2.to_vec());
        let mut mcs = [0.0f64; 5];
        let mut mssim_last = 0.0f64;
        let (c1v, c2v) = (<$F32>::splat($token, C1), <$F32>::splat($token, C2));
        let twov = <$F32>::splat($token, 2.0);

        for l in 0..level {
            let (p1, p2, pp) = product_planes(&im1, &im2);
            // Five separable 'valid' filters. H pass: lanes over
            // output columns, fixed tap order. V pass: lanes over
            // output columns of each output row, fixed tap order.
            let out_w = w - GAUSS_K + 1;
            let out_h = h - GAUSS_K + 1;
            let filt = |src: &[f32]| -> Vec<f32> {
                // horizontal into a `h × out_w` scratch
                let mut tmp = vec![0.0f32; h * out_w];
                for y in 0..h {
                    let srow = &src[y * w..y * w + w];
                    let drow = &mut tmp[y * out_w..y * out_w + out_w];
                    let mut x = 0usize;
                    while x + $LANES <= out_w {
                        let mut v = <$F32>::splat($token, 0.0);
                        for u in 0..GAUSS_K {
                            let xv = <$F32>::from_array(
                                $token,
                                core::array::from_fn(|k| srow[x + u + k]),
                            );
                            v = v + <$F32>::splat($token, taps[u]) * xv;
                        }
                        drow[x..x + $LANES].copy_from_slice(&v.to_array());
                        x += $LANES;
                    }
                    while x < out_w {
                        let mut acc = 0.0f32;
                        for u in 0..GAUSS_K {
                            acc += taps[u] * srow[x + u];
                        }
                        drow[x] = acc;
                        x += 1;
                    }
                }
                // vertical into `out_h × out_w`
                let mut out = vec![0.0f32; out_h * out_w];
                for y in 0..out_h {
                    let drow = &mut out[y * out_w..y * out_w + out_w];
                    let mut x = 0usize;
                    while x + $LANES <= out_w {
                        let mut v = <$F32>::splat($token, 0.0);
                        for u in 0..GAUSS_K {
                            let xv = <$F32>::from_array(
                                $token,
                                core::array::from_fn(|k| tmp[(y + u) * out_w + x + k]),
                            );
                            v = v + <$F32>::splat($token, taps[u]) * xv;
                        }
                        drow[x..x + $LANES].copy_from_slice(&v.to_array());
                        x += $LANES;
                    }
                    while x < out_w {
                        let mut acc = 0.0f32;
                        for u in 0..GAUSS_K {
                            acc += taps[u] * tmp[(y + u) * out_w + x];
                        }
                        drow[x] = acc;
                        x += 1;
                    }
                }
                out
            };
            let ((mu1, s1p), (mu2, s2p)) =
                maybe_join(|| (filt(&im1), filt(&p1)), || (filt(&im2), filt(&p2)));
            let s12p = filt(&pp);

            // Map stage: ssim = (n1·n2)/(d1·d2), cs = n2/d2, means in
            // raster order over the `out_h × out_w` map.
            let n = out_w * out_h;
            let (mut sum_s, mut sum_c) = (0.0f64, 0.0f64);
            let mut i = 0usize;
            while i + $LANES <= n {
                let m1 = <$F32>::from_array($token, core::array::from_fn(|k| mu1[i + k]));
                let m2 = <$F32>::from_array($token, core::array::from_fn(|k| mu2[i + k]));
                let v1 = <$F32>::from_array($token, core::array::from_fn(|k| s1p[i + k]));
                let v2 = <$F32>::from_array($token, core::array::from_fn(|k| s2p[i + k]));
                let vc = <$F32>::from_array($token, core::array::from_fn(|k| s12p[i + k]));
                let s1sq = v1 - m1 * m1;
                let s2sq = v2 - m2 * m2;
                let s12 = vc - m1 * m2;
                let n1 = twov * m1 * m2 + c1v;
                let n2 = twov * s12 + c2v;
                let d1 = m1 * m1 + m2 * m2 + c1v;
                let d2 = s1sq + s2sq + c2v;
                let sm = (n1 * n2) / (d1 * d2);
                let cm = n2 / d2;
                let (sa, ca) = (sm.to_array(), cm.to_array());
                for k in 0..$LANES {
                    sum_s += sa[k] as f64;
                    sum_c += ca[k] as f64;
                }
                i += $LANES;
            }
            while i < n {
                let m1 = mu1[i];
                let m2 = mu2[i];
                let s1sq = s1p[i] - m1 * m1;
                let s2sq = s2p[i] - m2 * m2;
                let s12 = s12p[i] - m1 * m2;
                let n1 = 2.0 * m1 * m2 + C1;
                let n2 = 2.0 * s12 + C2;
                let d1 = m1 * m1 + m2 * m2 + C1;
                let d2 = s1sq + s2sq + C2;
                sum_s += ((n1 * n2) / (d1 * d2)) as f64;
                sum_c += (n2 / d2) as f64;
                i += 1;
            }
            let npix = n as f64;
            mcs[l] = sum_c / npix;
            mssim_last = sum_s / npix;

            if l + 1 < level {
                let ((w2, h2, d1), (_, _, d2)) =
                    maybe_join(|| decimate(&im1, w, h), || decimate(&im2, w, h));
                im1 = d1;
                im2 = d2;
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
#[archmage::arcane]
pub fn msssim_core_tier_v3(token: archmage::X64V3Token, a: Args<'_>) -> f64 {
    type V = magetypes::simd::generic::f32x8<archmage::X64V3Token>;
    score_body!(token, V, 8, a)
}

/// The `v4` tier (AVX-512, 16-wide f32 — crate feature `avx512`).
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::arcane]
pub fn msssim_core_tier_v4(token: archmage::X64V4Token, a: Args<'_>) -> f64 {
    type V = magetypes::simd::generic::f32x16<archmage::X64V4Token>;
    score_body!(token, V, 16, a)
}

/// The `neon` tier (8-wide f32).
#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
pub fn msssim_core_tier_neon(token: archmage::NeonToken, a: Args<'_>) -> f64 {
    type V = magetypes::simd::generic::f32x8<archmage::NeonToken>;
    score_body!(token, V, 8, a)
}

/// The `wasm128` tier (8-wide f32).
#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
pub fn msssim_core_tier_wasm128(token: archmage::Wasm128Token, a: Args<'_>) -> f64 {
    type V = magetypes::simd::generic::f32x8<archmage::Wasm128Token>;
    score_body!(token, V, 8, a)
}

/// The `scalar` tier (8-wide f32, scalar-emulated lanes).
pub fn msssim_core_tier_scalar(token: archmage::ScalarToken, a: Args<'_>) -> f64 {
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
    archmage::incant!(msssim_core_tier(a), [v4, v3, neon, wasm128, scalar])
}
