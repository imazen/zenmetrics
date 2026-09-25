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
    for oi in 0..h2 {
        for oj in 0..w2 {
            out[oi * w2 + oj] = x[(2 * oi) * w + 2 * oj];
        }
    }
    (w2, h2, out)
}

/// Shared per-scale elementwise products feeding the five `'valid'`
/// filters: `im1²`, `im2²`, `im1·im2`.
fn product_planes(im1: &[f64], im2: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
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

/// Arguments to one `vifp` run, bundled so `incant!` dispatches a
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
            <$V>::splat($token, 0.0),
            <$V>::splat($token, EPS),
            <$V>::splat($token, SIGMA_NSQ),
        );
        let one = <$V>::splat($token, 1.0);

        for s in 0..NSCALES {
            let nk = (1 << (NSCALES - s)) + 1; // 17, 9, 5, 3
            let taps = gaussian_taps(nk);
            // Separable 'valid' correlation of `src` (packed `w·h`):
            // horizontal `tmp[y][x] = Σ_u k[u]·src[y][x+u]` then
            // vertical. Fixed tap order per element — bit-identical
            // across tiers.
            let filt = |src: &[f64], w: usize, h: usize| -> Vec<f64> {
                let out_w = w - nk + 1;
                let mut tmp = vec![0.0f64; h * out_w];
                for y in 0..h {
                    let srow = &src[y * w..y * w + w];
                    let drow = &mut tmp[y * out_w..y * out_w + out_w];
                    let mut x = 0usize;
                    while x + $LANES <= out_w {
                        let mut v = <$V>::splat($token, 0.0);
                        for u in 0..nk {
                            let xv =
                                <$V>::from_array($token, core::array::from_fn(|k| srow[x + u + k]));
                            v = v + <$V>::splat($token, taps[u]) * xv;
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
                }
                let out_h = h - nk + 1;
                let mut out = vec![0.0f64; out_h * out_w];
                for y in 0..out_h {
                    let drow = &mut out[y * out_w..y * out_w + out_w];
                    let mut x = 0usize;
                    while x + $LANES <= out_w {
                        let mut v = <$V>::splat($token, 0.0);
                        for u in 0..nk {
                            let xv = <$V>::from_array(
                                $token,
                                core::array::from_fn(|k| tmp[(y + u) * out_w + x + k]),
                            );
                            v = v + <$V>::splat($token, taps[u]) * xv;
                        }
                        drow[x..x + $LANES].copy_from_slice(&v.to_array());
                        x += $LANES;
                    }
                    while x < out_w {
                        let mut acc = 0.0f64;
                        for u in 0..nk {
                            acc += taps[u] * tmp[(y + u) * out_w + x];
                        }
                        drow[x] = acc;
                        x += 1;
                    }
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
            let (p1, p2, pp) = product_planes(&im1, &im2);
            let out_w = w - nk + 1;
            let out_h = h - nk + 1;
            let ((mu1, s1p), (mu2, s2p)) = maybe_join(
                || (filt(&im1, w, h), filt(&p1, w, h)),
                || (filt(&im2, w, h), filt(&p2, w, h)),
            );
            let s12p = filt(&pp, w, h);

            // GSM map: the reference's exact masking order —
            //   sigma<0 → 0;  g = s12/(s1+eps);  sv = s2 − g·s12;
            //   s1<eps → g=0, sv=s2, s1=0;  s2<eps → g=0, sv=0;
            //   g<0   → sv=s2, g=0;         sv≤eps → sv=eps.
            let n = out_w * out_h;
            let mut i = 0usize;
            while i + $LANES <= n {
                let m1 = <$V>::from_array($token, core::array::from_fn(|k| mu1[i + k]));
                let m2 = <$V>::from_array($token, core::array::from_fn(|k| mu2[i + k]));
                let v1 = <$V>::from_array($token, core::array::from_fn(|k| s1p[i + k]));
                let v2 = <$V>::from_array($token, core::array::from_fn(|k| s2p[i + k]));
                let vc = <$V>::from_array($token, core::array::from_fn(|k| s12p[i + k]));
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
                let (ta, da) = (tn.to_array(), td.to_array());
                for k in 0..$LANES {
                    num += libm::log10(ta[k]);
                    den += libm::log10(da[k]);
                }
                i += $LANES;
            }
            while i < n {
                let m1 = mu1[i];
                let m2 = mu2[i];
                let mut s1sq = (s1p[i] - m1 * m1).max(0.0);
                let s2sq = (s2p[i] - m2 * m2).max(0.0);
                let s12 = s12p[i] - m1 * m2;
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
                num += libm::log10(1.0 + (g * g * s1sq) / (sv + SIGMA_NSQ));
                den += libm::log10(1.0 + s1sq / SIGMA_NSQ);
                i += 1;
            }
        }

        num / den
    }};
}

/// The `v3` tier (AVX2, 4-wide f64).
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
pub fn vifp_core_tier_v3(token: archmage::X64V3Token, a: Args<'_>) -> f64 {
    type V = magetypes::simd::generic::f64x4<archmage::X64V3Token>;
    score_body!(token, V, 4, a)
}

/// The `v4` tier (AVX-512, 8-wide f64 — crate feature `avx512`).
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::arcane]
pub fn vifp_core_tier_v4(token: archmage::X64V4Token, a: Args<'_>) -> f64 {
    type V = magetypes::simd::generic::f64x8<archmage::X64V4Token>;
    score_body!(token, V, 8, a)
}

/// The `neon` tier (4-wide f64).
#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
pub fn vifp_core_tier_neon(token: archmage::NeonToken, a: Args<'_>) -> f64 {
    type V = magetypes::simd::generic::f64x4<archmage::NeonToken>;
    score_body!(token, V, 4, a)
}

/// The `wasm128` tier (4-wide f64).
#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
pub fn vifp_core_tier_wasm128(token: archmage::Wasm128Token, a: Args<'_>) -> f64 {
    type V = magetypes::simd::generic::f64x4<archmage::Wasm128Token>;
    score_body!(token, V, 4, a)
}

/// The `scalar` tier (4-wide f64, scalar-emulated lanes).
pub fn vifp_core_tier_scalar(token: archmage::ScalarToken, a: Args<'_>) -> f64 {
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
    archmage::incant!(vifp_core_tier(a), [v4, v3, neon, wasm128, scalar])
}
