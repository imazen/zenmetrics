//! VIFp pipeline internals — separable `'valid'` Gaussian statistics
//! per scale, the reference's exact GSM masking chain, `filter2` +
//! `1:2:end` downsampling, and `log10` information sums.
//!
//! The whole pipeline is **f64**, not the f32 used by sibling crates:
//! the reference's `1e-10` degenerate masks sit four orders above
//! f64's noise floor on a 0–255 signal (`E[x²]−μ²` cancellation at
//! `c=255` is ~`1e-13` in f64 but ~`1e-3` in f32 — the f32 residual
//! would defeat every mask). f64 reproduces the reference's mask
//! decisions exactly, including `NaN` on flat references.
//!
//! `score_body!` vectorizes the stencils and the per-element GSM map
//! through `f64x4` (`f64x8` on the `avx512` tier); lane count never
//! enters the per-element expressions, so all tiers produce
//! bit-identical output. Scalar helpers (`gaussian_taps`, `decimate`)
//! are shared.
//!
//! Provenance: `vifp_mscale.m` (Sheikh & Bovik pixel-domain release).
//! See `validation/README.md`.

use alloc::vec;
use alloc::vec::Vec;

/// Scales in the reference release.
const NSCALES: usize = 4;
/// `sigma_nsq = 2` — HVS noise variance, fixed by the reference.
const SIGMA_NSQ: f64 = 2.0;
/// `1e-10` — the reference's degenerate-signal thresholds.
const EPS: f64 = 1e-10;

/// `fspecial('gaussian', N, N/5)` rank-1 taps, normalised to sum 1.
/// `N = 2^(5−scale)+1` → 17, 9, 5, 3 per scale. The 2-D window is
/// `outer(k,k)`; the normalised 1-D outer product equals fspecial's
/// `win/sum(win)` factorisation.
fn gaussian_taps(n: usize) -> Vec<f64> {
    let half = (n / 2) as f64;
    let sigma = n as f64 / 5.0;
    let mut g = alloc::vec![0.0f64; n];
    let mut sum = 0.0f64;
    for (i, v) in g.iter_mut().enumerate() {
        let x = i as f64 - half;
        *v = libm::exp(-(x * x) / (2.0 * sigma * sigma));
        sum += *v;
    }
    g.iter().map(|&v| v / sum).collect()
}

/// `ref(1:2:end, 1:2:end)` on a `'valid'` map — plain stride-2
/// subsample (0-based even indices).
fn decimate(x: &[f64], w: usize, h: usize) -> (usize, usize, Vec<f64>) {
    let (w2, h2) = (w.div_ceil(2), h.div_ceil(2));
    let mut out = vec![0.0f64; w2 * h2];
    for_each_row(&mut out, w2, |oi, orow| {
        for (oj, o) in orow.iter_mut().enumerate() {
            *o = x[(2 * oi) * w + 2 * oj];
        }
    });
    (w2, h2, out)
}

/// Row-map driver — `f(y, out_row)` for every row. Under `parallel`
/// rows are `par_chunks_mut`-mapped when the pool has >1 thread;
/// otherwise the serial loop runs. Writes are disjoint per row either
/// way, so results are bit-identical at any thread count.
pub(crate) fn for_each_row(out: &mut [f64], row_w: usize, f: impl Fn(usize, &mut [f64]) + Sync) {
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
        rayon::join(fa, fb)
    }
    #[cfg(not(feature = "parallel"))]
    {
        (fa(), fb())
    }
}

/// Arguments to one `vifp` run, bundled so `attuned!` dispatches a
/// single parameter.
#[doc(hidden)]
pub struct Args<'a> {
    /// Level-0 reference plane, packed `w·h`.
    pub im1: &'a [f64],
    /// Level-0 distorted plane, packed `w·h`.
    pub im2: &'a [f64],
    /// Width × height of the level-0 planes.
    pub w: usize,
    /// Height.
    pub h: usize,
}

// ============================ tiered core ================================

/// The full pipeline for one `(im1, im2, w, h)`. Vector lanes hold
/// consecutive output elements (`f64x4`, or `f64x8` on `avx512`);
/// every per-element expression is a fixed-order `f64` chain and the
/// two information sums accumulate `f64` in raster order —
/// bit-identical on every tier.
macro_rules! score_body {
    ($token:ident, $V:ident, $LANES:literal, $a:ident) => {{
        let (mut w, mut h) = ($a.w, $a.h);
        let (mut im1, mut im2) = ($a.im1.to_vec(), $a.im2.to_vec());
        let (mut num, mut den) = (0.0f64, 0.0f64);
        let (zero, epsv, snv) = (
            <$V>::splat_t($token, 0.0),
            <$V>::splat_t($token, EPS),
            <$V>::splat_t($token, SIGMA_NSQ),
        );
        let one = <$V>::splat_t($token, 1.0);

        for s in 0..NSCALES {
            let nk = (1 << (NSCALES - s)) + 1; // 17, 9, 5, 3
            let taps = gaussian_taps(nk);
            // Separable 'valid' correlation of `src` (packed `w·h`):
            // horizontal `tmp[y][x] = Σ_u k[u]·src[y][x+u]` then
            // vertical. Fixed tap order per element — bit-identical
            // across tiers.
            let filt = |src: &[f64], w: usize, h: usize| -> Vec<f64> {
                let out_w = w - nk + 1;
                // Horizontal pass — per-row independent (same per-element
                // tap order under either driver).
                let hrow = |drow: &mut [f64], srow: &[f64]| {
                    let mut x = 0usize;
                    while x + $LANES <= out_w {
                        let mut v = <$V>::splat_t($token, 0.0);
                        for u in 0..nk {
                            let xv = <$V>::from_array_t(
                                $token,
                                core::array::from_fn(|k| srow[x + u + k]),
                            );
                            v = v + <$V>::splat_t($token, taps[u]) * xv;
                        }
                        drow[x..x + $LANES].copy_from_slice(&v.to_array());
                        x += $LANES;
                    }
                    while x < out_w {
                        let mut acc = 0.0f64;
                        for u in 0..nk {
                            acc += taps[u] * srow[x + u];
                        }
                        drow[x] = acc;
                        x += 1;
                    }
                };
                let out_h = h - nk + 1;
                let mut out = vec![0.0f64; out_h * out_w];
                // Keep only nk horizontally filtered rows. Each worker owns
                // a stripe and its ring; the tap order remains unchanged.
                let stripe = |first_y: usize, dst: &mut [f64]| {
                    let mut ring = vec![0.0f64; nk * out_w];
                    for u in 0..nk - 1 {
                        hrow(
                            &mut ring[u * out_w..(u + 1) * out_w],
                            &src[(first_y + u) * w..(first_y + u + 1) * w],
                        );
                    }
                    for (dy, drow) in dst.chunks_mut(out_w).enumerate() {
                        let y = first_y + dy;
                        let slot = (dy + nk - 1) % nk;
                        hrow(
                            &mut ring[slot * out_w..(slot + 1) * out_w],
                            &src[(y + nk - 1) * w..(y + nk) * w],
                        );
                        let rows: [&[f64]; 17] = core::array::from_fn(|u| {
                            if u < nk {
                                let slot = (dy + u) % nk;
                                &ring[slot * out_w..(slot + 1) * out_w]
                            } else {
                                &[]
                            }
                        });
                        let mut x = 0usize;
                        while x + $LANES <= out_w {
                            let mut v = <$V>::splat_t($token, 0.0);
                            for u in 0..nk {
                                let xv = <$V>::from_array_t(
                                    $token,
                                    core::array::from_fn(|k| rows[u][x + k]),
                                );
                                v = v + <$V>::splat_t($token, taps[u]) * xv;
                            }
                            drow[x..x + $LANES].copy_from_slice(&v.to_array());
                            x += $LANES;
                        }
                        while x < out_w {
                            let mut acc = 0.0;
                            for u in 0..nk {
                                acc += taps[u] * rows[u][x];
                            }
                            drow[x] = acc;
                            x += 1;
                        }
                    }
                };
                let mut ran_par = false;
                #[cfg(feature = "parallel")]
                if rayon::current_num_threads() > 1 {
                    use rayon::prelude::*;
                    out.par_chunks_mut(out_w * 64)
                        .enumerate()
                        .for_each(|(s, dst)| stripe(s * 64, dst));
                    ran_par = true;
                }
                let _ = ran_par;
                if !ran_par {
                    stripe(0, &mut out);
                }
                out
            };
            if s > 0 {
                // `filter2(win,im,'valid')` then `1:2:end`. When the
                // image is smaller than the kernel the reference's
                // 'valid' is empty — contribute nothing (and stay
                // empty for the rest).
                if w < nk || h < nk {
                    continue;
                }
                let (vw, vh) = (w - nk + 1, h - nk + 1);
                let (f1, f2) = maybe_join(|| filt(&im1, w, h), || filt(&im2, w, h));
                let ((w2, h2, d1), (_, _, d2)) =
                    maybe_join(|| decimate(&f1, vw, vh), || decimate(&f2, vw, vh));
                im1 = d1;
                im2 = d2;
                w = w2;
                h = h2;
            }
            if w < nk || h < nk {
                // 'valid' stats map is empty — the scale contributes
                // 0 to both sums (the reference's actual behaviour;
                // it has no size guard).
                continue;
            }
            let out_w = w - nk + 1;
            let out_h = h - nk + 1;
            let n = out_w * out_h;
            let mut terms_n = vec![0.0; n];
            let mut terms_d = vec![0.0; n];
            // Fuse the five horizontal statistics, then consume their
            // row rings directly in the vertical/GSM pass. Only the two
            // information terms survive for fixed raster-order pooling.
            let stripe = |first_y: usize, dst_n: &mut [f64], dst_d: &mut [f64]| {
                let mut ring: [Vec<f64>; 5] = core::array::from_fn(|_| vec![0.0; nk * out_w]);
                let horizontal = |y: usize, slot: usize, ring: &mut [Vec<f64>; 5]| {
                    let r = &im1[y * w..(y + 1) * w];
                    let d = &im2[y * w..(y + 1) * w];
                    let mut x = 0;
                    while x + $LANES <= out_w {
                        let mut sums = [zero; 5];
                        for u in 0..nk {
                            let rv =
                                <$V>::from_array_t($token, core::array::from_fn(|k| r[x + u + k]));
                            let dv =
                                <$V>::from_array_t($token, core::array::from_fn(|k| d[x + u + k]));
                            let tap = <$V>::splat_t($token, taps[u]);
                            let products = [rv, dv, rv * rv, dv * dv, rv * dv];
                            for k in 0..5 {
                                sums[k] = sums[k] + tap * products[k];
                            }
                        }
                        for k in 0..5 {
                            ring[k][slot * out_w + x..slot * out_w + x + $LANES]
                                .copy_from_slice(&sums[k].to_array());
                        }
                        x += $LANES;
                    }
                    while x < out_w {
                        let mut sums = [0.0; 5];
                        for u in 0..nk {
                            let (rv, dv) = (r[x + u], d[x + u]);
                            let products = [rv, dv, rv * rv, dv * dv, rv * dv];
                            for k in 0..5 {
                                sums[k] += taps[u] * products[k];
                            }
                        }
                        for k in 0..5 {
                            ring[k][slot * out_w + x] = sums[k];
                        }
                        x += 1;
                    }
                };
                for u in 0..nk - 1 {
                    horizontal(first_y + u, u, &mut ring);
                }
                for (dy, (nr, dr)) in dst_n
                    .chunks_mut(out_w)
                    .zip(dst_d.chunks_mut(out_w))
                    .enumerate()
                {
                    horizontal(first_y + dy + nk - 1, (dy + nk - 1) % nk, &mut ring);
                    let rows: [[&[f64]; 17]; 5] = core::array::from_fn(|k| {
                        core::array::from_fn(|u| {
                            if u < nk {
                                let slot = (dy + u) % nk;
                                &ring[k][slot * out_w..(slot + 1) * out_w]
                            } else {
                                &[]
                            }
                        })
                    });
                    let mut x = 0;
                    while x + $LANES <= out_w {
                        let mut sums = [zero; 5];
                        for u in 0..nk {
                            let tap = <$V>::splat_t($token, taps[u]);
                            for k in 0..5 {
                                let v = <$V>::from_array_t(
                                    $token,
                                    core::array::from_fn(|lane| rows[k][u][x + lane]),
                                );
                                sums[k] = sums[k] + tap * v;
                            }
                        }
                        let [m1, m2, v1, v2, vc] = sums;
                        let mut s1sq = (v1 - m1 * m1).max(zero);
                        let s2sq = (v2 - m2 * m2).max(zero);
                        let s12 = vc - m1 * m2;
                        let mut g = s12 / (s1sq + epsv);
                        let mut sv = s2sq - g * s12;
                        let mx = s1sq.simd_lt(epsv);
                        g = <$V>::blend(mx, zero, g);
                        sv = <$V>::blend(mx, s2sq, sv);
                        s1sq = <$V>::blend(mx, zero, s1sq);
                        let my = s2sq.simd_lt(epsv);
                        g = <$V>::blend(my, zero, g);
                        sv = <$V>::blend(my, zero, sv);
                        let mg = g.simd_lt(zero);
                        sv = <$V>::blend(mg, s2sq, sv);
                        g = <$V>::blend(mg, zero, g);
                        sv = <$V>::blend(sv.simd_le(epsv), epsv, sv);
                        let tn = one + (g * g * s1sq) / (sv + snv);
                        let td = one + s1sq / snv;
                        nr[x..x + $LANES].copy_from_slice(&tn.to_array());
                        dr[x..x + $LANES].copy_from_slice(&td.to_array());
                        x += $LANES;
                    }
                    while x < out_w {
                        let mut values = [0.0; 5];
                        for u in 0..nk {
                            for k in 0..5 {
                                values[k] += taps[u] * rows[k][u][x];
                            }
                        }
                        let m1 = values[0];
                        let m2 = values[1];
                        let mut s1sq = (values[2] - m1 * m1).max(0.0);
                        let s2sq = (values[3] - m2 * m2).max(0.0);
                        let s12 = values[4] - m1 * m2;
                        let mut g = s12 / (s1sq + EPS);
                        let mut sv = s2sq - g * s12;
                        if s1sq < EPS {
                            g = 0.0;
                            sv = s2sq;
                        }
                        if s1sq < EPS {
                            s1sq = 0.0;
                        }
                        if s2sq < EPS {
                            g = 0.0;
                            sv = 0.0;
                        }
                        if g < 0.0 {
                            sv = s2sq;
                            g = 0.0;
                        }
                        if sv <= EPS {
                            sv = EPS;
                        }
                        nr[x] = 1.0 + (g * g * s1sq) / (sv + SIGMA_NSQ);
                        dr[x] = 1.0 + s1sq / SIGMA_NSQ;
                        x += 1;
                    }
                }
            };
            let mut ran_par = false;
            #[cfg(feature = "parallel")]
            if rayon::current_num_threads() > 1 {
                use rayon::prelude::*;
                terms_n
                    .par_chunks_mut(out_w * 64)
                    .zip(terms_d.par_chunks_mut(out_w * 64))
                    .enumerate()
                    .for_each(|(s, (nr, dr))| stripe(s * 64, nr, dr));
                ran_par = true;
            }
            let _ = ran_par;
            if !ran_par {
                stripe(0, &mut terms_n, &mut terms_d);
            }
            for (&tn, &td) in terms_n.iter().zip(&terms_d) {
                num += libm::log10(tn);
                den += libm::log10(td);
            }
        }

        num / den
    }};
}

/// The `v3` tier (AVX2, 4-wide f64).
#[cfg(target_arch = "x86_64")]
#[archmage::attune]
pub fn vifp_core_tier_v3_t(token: archmage::X64V3Token, a: Args<'_>) -> f64 {
    type V = magetypes::simd::generic::f64x4<archmage::X64V3Token>;
    score_body!(token, V, 4, a)
}

/// The `v4x` tier (AVX-512, 8-wide f64 — crate feature `avx512`).
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::attune]
pub fn vifp_core_tier_v4x_t(token: archmage::X64V4xToken, a: Args<'_>) -> f64 {
    type V = magetypes::simd::generic::f64x8<archmage::X64V4xToken>;
    score_body!(token, V, 8, a)
}

/// The `neon` tier (4-wide f64).
#[cfg(target_arch = "aarch64")]
#[archmage::attune]
pub fn vifp_core_tier_neon_t(token: archmage::NeonToken, a: Args<'_>) -> f64 {
    type V = magetypes::simd::generic::f64x4<archmage::NeonToken>;
    score_body!(token, V, 4, a)
}

/// The `wasm128` tier (4-wide f64).
#[cfg(target_arch = "wasm32")]
#[archmage::attune]
pub fn vifp_core_tier_wasm128_t(token: archmage::Wasm128Token, a: Args<'_>) -> f64 {
    type V = magetypes::simd::generic::f64x4<archmage::Wasm128Token>;
    score_body!(token, V, 4, a)
}

/// The `scalar` tier (4-wide f64, scalar-emulated lanes).
pub fn vifp_core_tier_scalar_t(token: archmage::ScalarToken, a: Args<'_>) -> f64 {
    type V = magetypes::simd::generic::f64x4<archmage::ScalarToken>;
    score_body!(token, V, 4, a)
}

/// Runtime-dispatched score: the best tier this CPU supports.
pub(crate) fn vifp_core(inputs: &crate::Inputs, w: usize, h: usize) -> f64 {
    let a = Args {
        im1: &inputs.r,
        im2: &inputs.d,
        w,
        h,
    };
    archmage::attuned!(
        vifp_core_tier(a),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    )
}
