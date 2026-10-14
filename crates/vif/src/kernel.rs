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

// `libm::log10` polynomial constants (fdlibm `e_log10.c`) — the lane
// transcription below keeps the same constants, operation order and bit
// tricks, so every lane is bit-identical to `libm::log10` on x >= 1
// (the pooling domain: `1 + nonneg`).
const IVLN10HI: f64 = 4.34294481878168880939e-01;
const IVLN10LO: f64 = 2.50829467116452752298e-11;
const LOG10_2HI: f64 = 3.01029995663611771306e-01;
const LOG10_2LO: f64 = 3.69423907715893078616e-13;
const LG1: f64 = 6.666666666666735130e-01;
const LG2: f64 = 3.999999999940941908e-01;
const LG3: f64 = 2.857142874366239149e-01;
const LG4: f64 = 2.222219843214978396e-01;
const LG5: f64 = 1.818357216161805012e-01;
const LG6: f64 = 1.531383769920937332e-01;
const LG7: f64 = 1.479819860511658591e-01;

/// Lane-vectorised `libm::log10` for `x >= 1`. The frexp-style reduce
/// and `hi` truncation run as per-lane scalar bit ops; the polynomial,
/// divide and hi/lo reconstruction run lane-parallel with *separate*
/// `mul`/`add` (never fused), so each lane reproduces `libm::log10`
/// bitwise. Same shape as `vifvec`'s `log2_lanes!`.
macro_rules! log10_lanes {
    ($token:ident, $V:ty, $lanes:literal, $x:expr) => {{
        let xa: &[f64; $lanes] = $x;
        let mut xr = [0.0f64; $lanes];
        let mut kk = [0.0f64; $lanes];
        for i in 0..$lanes {
            let ui = xa[i].to_bits();
            let mut hx = (ui >> 32) as u32;
            hx = hx.wrapping_add(0x3ff0_0000u32.wrapping_sub(0x3fe6_a09e));
            kk[i] = ((hx >> 20) as i32 - 0x3ff) as f64;
            hx = (hx & 0x000f_ffff).wrapping_add(0x3fe6_a09e);
            xr[i] = f64::from_bits(((hx as u64) << 32) | (ui & 0xffff_ffff));
        }
        let xv = <$V>::from_array_t($token, xr);
        let kv = <$V>::from_array_t($token, kk);
        let f = xv - <$V>::splat_t($token, 1.0);
        let hfsq = <$V>::splat_t($token, 0.5) * f * f;
        let s = f / (<$V>::splat_t($token, 2.0) + f);
        let z = s * s;
        let w = z * z;
        let t1 = w
            * (<$V>::splat_t($token, LG2)
                + w * (<$V>::splat_t($token, LG4) + w * <$V>::splat_t($token, LG6)));
        let t2 = z
            * (<$V>::splat_t($token, LG1)
                + w * (<$V>::splat_t($token, LG3)
                    + w * (<$V>::splat_t($token, LG5) + w * <$V>::splat_t($token, LG7))));
        let r = t2 + t1;
        // hi = f - hfsq with the low 32 mantissa bits cleared.
        let mut ha = (f - hfsq).to_array();
        for v in ha.iter_mut() {
            *v = f64::from_bits(v.to_bits() & !0xffff_ffffu64);
        }
        let hi = <$V>::from_array_t($token, ha);
        let lo = f - hi - hfsq + s * (hfsq + r);
        // val_hi + val_lo ~ log10(1+f) + k·log10(2) — the e_log10
        // reconstruction (y = k·log10_2hi enters like e_log2's k·ivln2hi).
        let val_hi = hi * <$V>::splat_t($token, IVLN10HI);
        let y = kv * <$V>::splat_t($token, LOG10_2HI);
        let val_lo = kv * <$V>::splat_t($token, LOG10_2LO)
            + (lo + hi) * <$V>::splat_t($token, IVLN10LO)
            + lo * <$V>::splat_t($token, IVLN10HI);
        let wv = y + val_hi;
        let val_lo = val_lo + ((y - wv) + val_hi);
        let mut ra = (val_lo + wv).to_array();
        for i in 0..$lanes {
            // libm early-outs for the x >= 1 domain: `x == 1 → 0`;
            // `exp == 0x7ff → x` (inf → inf, NaN → NaN anyway).
            if xa[i] == 1.0 {
                ra[i] = 0.0;
            } else if (xa[i].to_bits() >> 52) & 0x7ff == 0x7ff {
                ra[i] = xa[i];
            }
        }
        ra
    }};
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
                #[cfg(feature = "parallel")]
                let mut ran_par = false;
                #[cfg(not(feature = "parallel"))]
                let ran_par = false;
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
            #[cfg(feature = "parallel")]
            let mut ran_par = false;
            #[cfg(not(feature = "parallel"))]
            let ran_par = false;
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
            // Lane-evaluate `libm::log10` (bit-identical per element via
            // `log10_lanes!`), keep the sequential `+=` per accumulator.
            let n_terms = terms_n.len();
            let mut i = 0usize;
            while i + $LANES <= n_terms {
                let na = log10_lanes!(
                    $token,
                    $V,
                    $LANES,
                    terms_n[i..i + $LANES].try_into().unwrap()
                );
                let da = log10_lanes!(
                    $token,
                    $V,
                    $LANES,
                    terms_d[i..i + $LANES].try_into().unwrap()
                );
                for l in 0..$LANES {
                    num += na[l];
                    den += da[l];
                }
                i += $LANES;
            }
            while i < n_terms {
                num += libm::log10(terms_n[i]);
                den += libm::log10(terms_d[i]);
                i += 1;
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
