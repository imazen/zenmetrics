//! Caller-owned slice math for CPU image-quality metrics.
//! Midp transcendental inputs must be finite: logarithms and powers require
//! positive normal inputs, and exponentials must stay in the normal output range.
//! NaNs, infinities and subnormal log/pow inputs are outside this unchecked contract.
//! Arithmetic kernels preserve separate multiply/add operations. Reductions use
//! lane order and may differ across native widths. Call row primitives per row
//! for strided planes; no kernel assumes multiple rows are tightly packed.
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#[cfg_attr(test, macro_use)]
extern crate alloc;
#[cfg(test)]
extern crate std;
use alloc::vec::Vec;
use archmage::ScalarToken;
mod policy;
pub use policy::{MathPolicy, Midp};
// Explicit slice names share the same caller-owned implementations.
pub use {
    vexp_into as vexp_into_slice, vlog_into as vlog_into_slice, vpow_into as vpow_into_slice,
};
pub mod par;
#[macro_use]
mod kernels;
/// Explicit-token eight-lane kernels, generic over a math policy.
pub mod width8 {
    kernels!(f32x8, F32x8Convert, 8, exp8, ln8, pow8);
}
/// Explicit-token native AVX-512 kernels, generic over a math policy.
#[cfg(feature = "avx512")]
pub mod width16 {
    kernels!(f32x16, F32x16Convert, 16, exp16, ln16, pow16);
}
/// Finish the epsilon-offset L2 mean from a partial sum and sample count.
#[inline]
pub fn lp2_finish(sum: f32, n: usize) -> f32 {
    const LP_SAFE_EPS: f32 = 1e-5;
    let mean = sum / n as f32;
    Midp::scalar_pow(mean + LP_SAFE_EPS, 0.5) - Midp::scalar_pow(LP_SAFE_EPS, 0.5)
}
/// Finish four epsilon-offset L2 means from ordered channel partial sums.
#[inline]
pub fn xcm4_finish(sums: [f32; 4], n: usize) -> [f32; 4] {
    const LP_SAFE_EPS: f32 = 1e-5;
    let nf = n as f32;
    [
        Midp::scalar_pow(sums[0] / nf + LP_SAFE_EPS, 0.5) - Midp::scalar_pow(LP_SAFE_EPS, 0.5),
        Midp::scalar_pow(sums[1] / nf + LP_SAFE_EPS, 0.5) - Midp::scalar_pow(LP_SAFE_EPS, 0.5),
        Midp::scalar_pow(sums[2] / nf + LP_SAFE_EPS, 0.5) - Midp::scalar_pow(LP_SAFE_EPS, 0.5),
        Midp::scalar_pow(sums[3] / nf + LP_SAFE_EPS, 0.5) - Midp::scalar_pow(LP_SAFE_EPS, 0.5),
    ]
}
pub(crate) fn safe_pow_with_offset_into_scalar(
    token: ScalarToken,
    xs: &[f32],
    out: &mut [f32],
    offset: f32,
    p: f32,
    offset_pow_p: f32,
) {
    width8::safe_pow_with_offset_kernel::<_, Midp>(token, xs, out, offset, p, offset_pow_p);
}

pub(crate) fn vexp_into_scalar(token: ScalarToken, xs: &[f32], out: &mut [f32]) {
    width8::vexp_kernel::<_, Midp>(token, xs, out);
}

pub(crate) fn vlog_into_scalar(token: ScalarToken, xs: &[f32], out: &mut [f32]) {
    width8::vlog_kernel::<_, Midp>(token, xs, out);
}

pub(crate) fn vpow_into_scalar(token: ScalarToken, xs: &[f32], out: &mut [f32], p: f32) {
    width8::vpow_kernel::<_, Midp>(token, xs, out, p);
}

pub(crate) fn vaxpy_into_scalar(token: ScalarToken, dst: &mut [f32], src: &[f32], a: f32) {
    width8::vaxpy_kernel::<_, Midp>(token, dst, src, a);
}

pub(crate) fn vscale_into_scalar(token: ScalarToken, dst: &mut [f32], src: &[f32], a: f32) {
    width8::vscale_kernel::<_, Midp>(token, dst, src, a);
}

pub(crate) fn vabs_diff_into_scalar(token: ScalarToken, out: &mut [f32], x: &[f32], y: &[f32]) {
    width8::vabs_diff_kernel::<_, Midp>(token, out, x, y);
}

pub(crate) fn vmin_abs_into_scalar(token: ScalarToken, out: &mut [f32], x: &[f32], y: &[f32]) {
    width8::vmin_abs_kernel::<_, Midp>(token, out, x, y);
}

pub(crate) fn vweber_band_into_scalar(
    token: ScalarToken,
    band: &mut [f32],
    log: &mut [f32],
    fine: &[f32],
    img_exp: &[f32],
    exp_l: &[f32],
) {
    width8::vweber_band_kernel::<_, Midp>(token, band, log, fine, img_exp, exp_l);
}

pub(crate) fn vweber_band_nolog_into_scalar(
    token: ScalarToken,
    band: &mut [f32],
    fine: &[f32],
    img_exp: &[f32],
    exp_l: &[f32],
) {
    width8::vweber_band_nolog_kernel::<_, Midp>(token, band, fine, img_exp, exp_l);
}

pub(crate) fn vscale2_into_scalar(
    token: ScalarToken,
    d1: &mut [f32],
    d2: &mut [f32],
    src: &[f32],
    a1: f32,
    a2: f32,
) {
    width8::vscale2_kernel::<_, Midp>(token, d1, d2, src, a1, a2);
}

pub(crate) fn vaxpy2_into_scalar(
    token: ScalarToken,
    d1: &mut [f32],
    d2: &mut [f32],
    src: &[f32],
    a1: f32,
    a2: f32,
) {
    width8::vaxpy2_kernel::<_, Midp>(token, d1, d2, src, a1, a2);
}

pub(crate) fn vmul2_scale2_pair_into_scalar(
    token: ScalarToken,
    o1: &mut [f32],
    o2: &mut [f32],
    x1: &[f32],
    x2: &[f32],
    w: &[f32],
    a: f32,
    b: f32,
) {
    width8::vmul2_scale2_pair_kernel::<_, Midp>(token, o1, o2, x1, x2, w, a, b);
}

pub(crate) fn vfir_into_scalar(
    token: ScalarToken,
    dst: &mut [f32],
    srcs: &[&[f32]],
    coeffs: &[f32],
    off: usize,
) {
    width8::vfir_into_kernel::<_, Midp>(token, dst, srcs, coeffs, off)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn vfir2_into_scalar(
    token: ScalarToken,
    d0: &mut [f32],
    d1: &mut [f32],
    srcs: &[&[f32]],
    coeffs0: &[f32],
    coeffs1: &[f32],
    off: usize,
) {
    width8::vfir2_into_kernel::<_, Midp>(token, d0, d1, srcs, coeffs0, coeffs1, off)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn vabs_diff_pow_scalar(
    token: ScalarToken,
    out: &mut [f32],
    x: &[f32],
    y: &[f32],
    offset: f32,
    p: f32,
    offset_pow_p: f32,
) {
    width8::vabs_diff_pow_kernel::<_, Midp>(token, out, x, y, offset, p, offset_pow_p)
}

pub(crate) fn vabs_diff_mul_lp2_sum_scalar(
    token: ScalarToken,
    t: &[f32],
    r: &[f32],
    s: &[f32],
) -> f32 {
    width8::vabs_diff_mul_lp2_sum_kernel::<_, Midp>(token, t, r, s)
}

pub(crate) fn vxcm_pool_clamp_4ch_sqsum_partial_scalar(
    token: ScalarToken,
    d: &[&[f32]; 4],
    t: &[&[f32]; 4],
    w: &[[f32; 4]; 4],
    d_max: f32,
) -> [f32; 4] {
    width8::vxcm_pool_clamp_4ch_sqsum_partial_kernel::<_, Midp>(token, d, t, w, d_max)
}

include!("tier_v3.rs");
include!("tier_v4.rs");
include!("tier_v4x.rs");
include!("tier_neon.rs");
include!("tier_wasm128.rs");
/// pipeline always satisfies this because the inputs are
/// `|magnitude| + SAFE_EPS` with `SAFE_EPS = 1e-5`.
///
/// `offset_pow_p` is `offset.powf(p)` — pass it once (hoisted by
/// the caller); the kernel reuses it on every lane.
#[inline]
pub fn safe_pow_with_offset_into(
    xs: &[f32],
    out: &mut [f32],
    offset: f32,
    p: f32,
    offset_pow_p: f32,
) {
    debug_assert_eq!(xs.len(), out.len());
    archmage::incant!(
        safe_pow_with_offset_into(xs, out, offset, p, offset_pow_p),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

/// Convenience: resize `out` to `xs.len()` then dispatch.
///
/// Use this when the caller owns a reusable `Vec<f32>` scratch and
/// wants the size handled in one call (mirrors `Vec::clear` +
/// `Vec::resize` shape used elsewhere in the masking module).
#[inline]
#[allow(dead_code)]
pub fn safe_pow_with_offset_into_vec(
    xs: &[f32],
    out: &mut Vec<f32>,
    offset: f32,
    p: f32,
    offset_pow_p: f32,
) {
    out.clear();
    out.resize(xs.len(), 0.0);
    safe_pow_with_offset_into(xs, out.as_mut_slice(), offset, p, offset_pow_p)
}

/// `out[i] = exp(xs[i])`. Reusable from Chunk 5 (CSF SIMD).
#[inline]
#[allow(dead_code)]
pub fn vexp_into(xs: &[f32], out: &mut [f32]) {
    debug_assert_eq!(xs.len(), out.len());
    archmage::incant!(vexp_into(xs, out), [v4x, v4, v3, neon, wasm128, scalar])
}

/// `out[i] = ln(xs[i])`. Positive inputs only. Reusable from Chunk 5.
#[inline]
#[allow(dead_code)]
pub fn vlog_into(xs: &[f32], out: &mut [f32]) {
    debug_assert_eq!(xs.len(), out.len());
    archmage::incant!(vlog_into(xs, out), [v4x, v4, v3, neon, wasm128, scalar])
}

/// `out[i] = xs[i].powf(p)`. Positive inputs only. Reusable from Chunk 5.
#[inline]
#[allow(dead_code)]
pub fn vpow_into(xs: &[f32], out: &mut [f32], p: f32) {
    debug_assert_eq!(xs.len(), out.len());
    archmage::incant!(vpow_into(xs, out, p), [v4x, v4, v3, neon, wasm128, scalar])
}

// ---------------------------------------------------------------------------
// Elementwise dispatchers — the video pipeline's glue loops.
// ---------------------------------------------------------------------------

/// `dst[i] += a * src[i]`. Temporal FIR accumulate.
#[inline]
pub fn vaxpy_into(dst: &mut [f32], src: &[f32], a: f32) {
    debug_assert_eq!(dst.len(), src.len());
    archmage::incant!(
        vaxpy_into(dst, src, a),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

/// `dst[i] = a * src[i]` into a distinct buffer.
#[inline]
pub fn vscale_into(dst: &mut [f32], src: &[f32], a: f32) {
    debug_assert_eq!(dst.len(), src.len());
    archmage::incant!(
        vscale_into(dst, src, a),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

/// `out[i] = |x[i] - y[i]|`.
#[inline]
pub fn vabs_diff_into(out: &mut [f32], x: &[f32], y: &[f32]) {
    debug_assert_eq!(out.len(), x.len());
    debug_assert_eq!(out.len(), y.len());
    archmage::incant!(
        vabs_diff_into(out, x, y),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

/// `out[i] = min(|x[i]|, |y[i]|)` — mutual-mask raw term.
#[inline]
pub fn vmin_abs_into(out: &mut [f32], x: &[f32], y: &[f32]) {
    debug_assert_eq!(out.len(), x.len());
    debug_assert_eq!(out.len(), y.len());
    archmage::incant!(
        vmin_abs_into(out, x, y),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

/// `band[i] = clamp((fine[i] − img_exp[i]) / max(exp_l[i], 0.01), ±1000)`
/// and `log[i] = log10(max(exp_l[i], 0.01))` — the Weber-contrast
/// non-baseband fill, fused so `expanded_l` is loaded once.
pub fn vweber_band_into(
    band: &mut [f32],
    log: &mut [f32],
    fine: &[f32],
    img_exp: &[f32],
    exp_l: &[f32],
) {
    assert_eq!(band.len(), log.len());
    assert_eq!(band.len(), fine.len());
    assert_eq!(band.len(), img_exp.len());
    assert_eq!(band.len(), exp_l.len());
    archmage::incant!(
        vweber_band_into(band, log, fine, img_exp, exp_l),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

/// `band[i] = clamp((fine[i] − img_exp[i]) / max(exp_l[i], 0.01), ±1000)`
/// — [`vweber_band_into`] without the `log_l_bkg` output, for the
/// pyramids whose `log_l_bkg` is never consumed.
pub fn vweber_band_nolog_into(band: &mut [f32], fine: &[f32], img_exp: &[f32], exp_l: &[f32]) {
    assert_eq!(band.len(), fine.len());
    assert_eq!(band.len(), img_exp.len());
    assert_eq!(band.len(), exp_l.len());
    archmage::incant!(
        vweber_band_nolog_into(band, fine, img_exp, exp_l),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

/// `d1[i] = a1·src[i]`, `d2[i] = a2·src[i]` — shared-source dual
/// scale (transient + sustained FIR over the same plane).
#[inline]
pub fn vscale2_into(d1: &mut [f32], d2: &mut [f32], src: &[f32], a1: f32, a2: f32) {
    debug_assert_eq!(d1.len(), src.len());
    debug_assert_eq!(d2.len(), src.len());
    archmage::incant!(
        vscale2_into(d1, d2, src, a1, a2),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

/// `d1[i] += a1·src[i]`, `d2[i] += a2·src[i]` — shared-source dual
/// accumulate.
#[inline]
pub fn vaxpy2_into(d1: &mut [f32], d2: &mut [f32], src: &[f32], a1: f32, a2: f32) {
    debug_assert_eq!(d1.len(), src.len());
    debug_assert_eq!(d2.len(), src.len());
    archmage::incant!(
        vaxpy2_into(d1, d2, src, a1, a2),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

/// `dst[i] = Σ_k coeffs[k]·srcs[k][off+i]` — fused N-tap temporal
/// FIR, one pass over all source planes. Band callers pass the
/// band's start index in `off`.
#[inline]
pub fn vfir_into(dst: &mut [f32], srcs: &[&[f32]], coeffs: &[f32], off: usize) {
    debug_assert_eq!(srcs.len(), coeffs.len());
    archmage::incant!(
        vfir_into(dst, srcs, coeffs, off),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

/// `d0[i] = Σ_k coeffs0[k]·srcs[k][off+i]`,
/// `d1[i] = Σ_k coeffs1[k]·srcs[k][off+i]` — paired fused FIR.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn vfir2_into(
    d0: &mut [f32],
    d1: &mut [f32],
    srcs: &[&[f32]],
    coeffs0: &[f32],
    coeffs1: &[f32],
    off: usize,
) {
    debug_assert_eq!(d0.len(), d1.len());
    debug_assert_eq!(srcs.len(), coeffs0.len());
    debug_assert_eq!(srcs.len(), coeffs1.len());
    archmage::incant!(
        vfir2_into(d0, d1, srcs, coeffs0, coeffs1, off),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

/// `o1[i] = ((x1[i]·a)·w[i])·b`, `o2[i] = ((x2[i]·a)·w[i])·b` — the
/// test/reference CSF-weight pair in one pass.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn vmul2_scale2_pair_into(
    o1: &mut [f32],
    o2: &mut [f32],
    x1: &[f32],
    x2: &[f32],
    w: &[f32],
    a: f32,
    b: f32,
) {
    debug_assert_eq!(o1.len(), x1.len());
    debug_assert_eq!(o2.len(), x2.len());
    debug_assert_eq!(o1.len(), w.len());
    archmage::incant!(
        vmul2_scale2_pair_into(o1, o2, x1, x2, w, a, b),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

/// Raw `Σ u²` for `lp_norm_mean(|t−r|·s, 2.0)` — the video baseband
/// pooled difference, fused (no intermediate plane). Apply
/// [`lp2_finish`] to the folded total for the final norm.
#[inline]
pub fn vabs_diff_mul_lp2_sum(t: &[f32], r: &[f32], s: &[f32]) -> f32 {
    debug_assert_eq!(t.len(), r.len());
    debug_assert_eq!(t.len(), s.len());
    archmage::incant!(
        vabs_diff_mul_lp2_sum(t, r, s),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

/// `out[i] = (|x[i] − y[i]| + offset)^p − offset_pow_p` — the
/// `vabs_diff` + `safe_pow_with_offset` pair fused into one pass.
#[allow(clippy::too_many_arguments)]
pub fn vabs_diff_pow_into(
    out: &mut [f32],
    x: &[f32],
    y: &[f32],
    offset: f32,
    p: f32,
    offset_pow_p: f32,
) {
    archmage::incant!(
        vabs_diff_pow(out, x, y, offset, p, offset_pow_p),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

/// Per-channel raw `Σ u²` for the 4-channel cross-channel pool +
/// soft clamp + `lp_norm_mean_p2`, fully fused: `d` holds the
/// `safe_pow(|T−R|, p)` planes, `t` the `safe_pow(|M_mm|, q)` terms.
/// Apply [`xcm4_finish`] to the folded per-channel totals.
#[inline]
pub fn vxcm_pool_clamp_4ch_sqsum_partial(
    d: &[&[f32]; 4],
    t: &[&[f32]; 4],
    w: &[[f32; 4]; 4],
    d_max: f32,
) -> [f32; 4] {
    archmage::incant!(
        vxcm_pool_clamp_4ch_sqsum_partial(d, t, w, d_max),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Accuracy budget: the masking parity test downstream uses a
    /// 1e-3 relative tolerance on `mult_mutual_band` output; magetypes'
    /// `pow_midp_unchecked` is documented at ~1e-5 relative error /
    /// ≤128 ULP. We assert 5e-5 here — that's an order of magnitude
    /// tighter than the consumer needs, so any future regression
    /// (e.g. dropping to `pow_lowp`) is caught here, not silently in
    /// the parity gate.
    const REL_TOL: f32 = 5e-5;

    fn assert_close(got: f32, want: f32, idx: usize, ctx: &str) {
        let denom = want.abs().max(1e-6);
        let rel = (got - want).abs() / denom;
        assert!(
            rel <= REL_TOL,
            "{ctx} idx={idx}: got={got} want={want} rel={rel} > {REL_TOL}"
        );
    }

    #[test]
    fn safe_pow_matches_scalar_powf() {
        // Sweep typical masking-stage magnitudes (0 .. ~200) and
        // typical exponents (MASK_Q ∈ [1.3, 3.7], MASK_P ≈ 2.26).
        // Include sizes that cross the 8-lane boundary so the scalar
        // tail is exercised.
        let offset = 1e-5_f32;
        let exponents: &[f32] = &[1.3, 1.8, 2.26, 3.0, 3.7];
        let sizes: &[usize] = &[0, 1, 7, 8, 9, 15, 16, 17, 100, 1024];

        let mut s = 0x12345678_u32;
        let mut prng = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 8) as f32 / (1u32 << 24) as f32 * 200.0
        };

        for &n in sizes {
            let xs: Vec<f32> = (0..n).map(|_| prng()).collect();
            for &p in exponents {
                let offset_pow_p = offset.powf(p);
                let mut got = vec![0.0_f32; n];
                safe_pow_with_offset_into(&xs, &mut got, offset, p, offset_pow_p);
                for (i, &x) in xs.iter().enumerate() {
                    let want = (x + offset).powf(p) - offset_pow_p;
                    assert_close(got[i], want, i, &format!("n={n} p={p}"));
                }
            }
        }
    }

    #[test]
    fn vpow_matches_powf() {
        let exponents: &[f32] = &[0.5, 1.0, 1.3, 2.26, 3.0];
        let sizes: &[usize] = &[0, 7, 8, 17, 256];

        let mut s = 0x9abcdef0_u32;
        let mut prng = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            // Strictly positive — ln/log are undefined on 0.
            0.01 + (s >> 8) as f32 / (1u32 << 24) as f32 * 100.0
        };
        for &n in sizes {
            let xs: Vec<f32> = (0..n).map(|_| prng()).collect();
            for &p in exponents {
                let mut got = vec![0.0_f32; n];
                vpow_into(&xs, &mut got, p);
                for (i, &x) in xs.iter().enumerate() {
                    assert_close(got[i], x.powf(p), i, &format!("n={n} p={p}"));
                }
            }
        }
    }

    #[test]
    fn vexp_matches_exp() {
        // exp domain in the masking / CSF chain is roughly [-20, 20]
        // (log10 luminance ~5 max → x * ln(10) ~12). Cover that.
        let mut s = 0xdeadbeef_u32;
        let mut prng = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 40.0
        };
        for &n in &[0, 7, 8, 17, 256] {
            let xs: Vec<f32> = (0..n).map(|_| prng()).collect();
            let mut got = vec![0.0_f32; n];
            vexp_into(&xs, &mut got);
            for (i, &x) in xs.iter().enumerate() {
                let want = x.exp();
                let denom = want.abs().max(1e-12);
                let rel = (got[i] - want).abs() / denom;
                assert!(
                    rel <= 5e-5,
                    "vexp idx={i} x={x} got={} want={want} rel={rel}",
                    got[i]
                );
            }
        }
    }

    #[test]
    fn vlog_matches_ln() {
        let mut s = 0xcafebabe_u32;
        let mut prng = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            // Spread inputs across many orders of magnitude — the
            // CSF / contrast stage sees `log10(L_bkg)` over a wide
            // range. log2 has ~3 ULP error; relative error is largest
            // near 1 (ln(1)=0). We give the test the same epsilon
            // floor the consumer uses.
            (1e-4 + (s >> 8) as f32 / (1u32 << 24) as f32) * 10.0
        };
        for &n in &[0, 7, 8, 17, 256] {
            let xs: Vec<f32> = (0..n).map(|_| prng()).collect();
            let mut got = vec![0.0_f32; n];
            vlog_into(&xs, &mut got);
            for (i, &x) in xs.iter().enumerate() {
                let want = x.ln();
                let denom = want.abs().max(1e-4);
                let rel = (got[i] - want).abs() / denom;
                assert!(
                    rel <= 5e-4,
                    "vlog idx={i} x={x} got={} want={want} rel={rel}",
                    got[i]
                );
            }
        }
    }

    #[test]
    fn safe_pow_vec_resizes() {
        let xs: Vec<f32> = (0..23).map(|i| i as f32 * 0.5).collect();
        let mut out = Vec::new();
        let p = 2.26;
        let offset = 1e-5_f32;
        let offset_pow_p = offset.powf(p);
        safe_pow_with_offset_into_vec(&xs, &mut out, offset, p, offset_pow_p);
        assert_eq!(out.len(), xs.len());
        for (i, &x) in xs.iter().enumerate() {
            assert_close(out[i], (x + offset).powf(p) - offset_pow_p, i, "vec resize");
        }
    }
}

/// `out[i] = lerp(lut[floor(idx)], lut[floor(idx)+1], fract(idx))`,
/// `idx = clamp((xs[i]-min)*inv_step, 0, max_index)`.
/// Finite inputs only; `0 <= max_index < lut.len()-1`. No allocations.
pub fn gather_lerp_into(
    xs: &[f32],
    lut: &[f32],
    min: f32,
    inv_step: f32,
    max_index: f32,
    out: &mut [f32],
) {
    archmage::incant!(
        gather_lerp(xs, lut, min, inv_step, max_index, 0.0, 1.0, out),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}
/// Fused gather/interpolate followed by `exp((value+offset)*scale)`.
/// Same bracket contract as [`gather_lerp_into`], plus [`Midp`]'s exp domain.
pub fn gather_lerp_exp_into(
    xs: &[f32],
    lut: &[f32],
    min: f32,
    inv_step: f32,
    max_index: f32,
    offset: f32,
    scale: f32,
    out: &mut [f32],
) {
    archmage::incant!(
        gather_lerp_exp(xs, lut, min, inv_step, max_index, offset, scale, out),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}
pub(crate) fn gather_lerp_scalar(
    token: ScalarToken,
    xs: &[f32],
    lut: &[f32],
    min: f32,
    inv_step: f32,
    max_index: f32,
    offset: f32,
    scale: f32,
    out: &mut [f32],
) {
    width8::gather_lerp_kernel::<_, Midp, false>(
        token, xs, lut, min, inv_step, max_index, offset, scale, out,
    )
}
pub(crate) fn gather_lerp_exp_scalar(
    token: ScalarToken,
    xs: &[f32],
    lut: &[f32],
    min: f32,
    inv_step: f32,
    max_index: f32,
    offset: f32,
    scale: f32,
    out: &mut [f32],
) {
    width8::gather_lerp_kernel::<_, Midp, true>(
        token, xs, lut, min, inv_step, max_index, offset, scale, out,
    )
}

#[cfg(test)]
mod tier_tests;

#[inline]
fn scalar_floor(x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        x.floor()
    }
    #[cfg(not(feature = "std"))]
    {
        magetypes::simd::generic::f32x8::splat(ScalarToken, x)
            .floor()
            .to_array()[0]
    }
}
#[inline]
fn scalar_log10(x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        x.log10()
    }
    #[cfg(not(feature = "std"))]
    {
        Midp::scalar_ln(x) * core::f32::consts::LOG10_E
    }
}
