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

    /// x86-64-v4 specialization of `gather_lerp_kernel`: replaces the
    /// 2×16 scalar loads + `vinsertps` assembly a lane gather emits
    /// with two `VGATHERDPS` instructions. Bit-identical: same lanes,
    /// same lerp, same exp input — only the table reads differ in
    /// mechanism. `lut[lanes+1]` is `lut[1..][lanes]`, so the second
    /// gather reuses the same index vector on a shifted table.
    ///
    /// `idx` is clamped to `[0, max_index]` before `floor`, so
    /// `floor ≤ lut.len() - 2` and every lane read is in range; the
    /// `or` fallback lanes never fire.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    pub(crate) fn gather_lerp_kernel_v4<P: super::MathPolicy, const EXP: bool>(
        token: archmage::X64V4xToken,
        xs: &[f32],
        lut: &[f32],
        min: f32,
        inv_step: f32,
        max_index: f32,
        offset: f32,
        scale: f32,
        out: &mut [f32],
    ) {
        use magetypes::simd::generic::{f32x16, u32x16};
        type V = f32x16<archmage::X64V4Token>;
        type U = u32x16<archmage::X64V4Token>;
        assert_eq!(xs.len(), out.len());
        assert!(lut.len() >= 2 && max_index >= 0.0 && max_index < (lut.len() - 1) as f32);
        let token = token.v4();
        let min_v = V::splat_t(token, min);
        let step_v = V::splat_t(token, inv_step);
        let zero = V::zero_t(token);
        let max_v = V::splat_t(token, max_index);
        let offset_v = V::splat_t(token, offset);
        let scale_v = V::splat_t(token, scale);
        let lut_hi = &lut[1..];
        let or_v = V::zero_t(token);
        let (xc, xt) = V::partition_slice_t(token, xs);
        let (oc, ot) = V::partition_slice_mut_t(token, out);
        for (x, o) in xc.iter().zip(oc) {
            let idx = ((V::load_t(token, x) - min_v) * step_v)
                .max(zero)
                .min(max_v);
            let floor = idx.floor();
            let lanes = U::from_raw_t(token, idx.to_i32().raw());
            let lo = V::gather_or(lut, lanes, or_v);
            let hi = V::gather_or(lut_hi, lanes, or_v);
            let value = lo + (idx - floor) * (hi - lo);
            if EXP {
                P::exp16::<archmage::X64V4Token>((value + offset_v) * scale_v).store(o);
            } else {
                value.store(o);
            }
        }
        for (x, o) in xt.iter().zip(ot) {
            let idx = ((*x - min) * inv_step).clamp(0.0, max_index);
            let floor = super::scalar_floor(idx);
            let lo = (floor as usize).min(lut.len() - 2);
            let value = lut[lo] + (idx - floor) * (lut[lo + 1] - lut[lo]);
            *o = if EXP {
                P::scalar_exp((value + offset) * scale)
            } else {
                value
            };
        }
    }

    /// x86-64-v4 specialization of `gather_lerp_exp3_apply_kernel`:
    /// shares one bracket computation across the three channels and
    /// hardware-gathers the lo/hi table entries per channel.
    /// Bit-identical to the lane-gathered generic body.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gather_lerp_exp3_apply_kernel_v4<P: super::MathPolicy>(
        token: archmage::X64V4xToken,
        xs: &[f32],
        luts: [&[f32]; 3],
        min: f32,
        inv_step: f32,
        max_index: f32,
        offset: f32,
        scale: f32,
        band_mul: f32,
        gains: [f32; 3],
        ins_a: [&[f32]; 3],
        ins_b: [&[f32]; 3],
        outs_a: [&mut [f32]; 3],
        outs_b: [&mut [f32]; 3],
    ) {
        use magetypes::simd::generic::{f32x16, u32x16};
        type V = f32x16<archmage::X64V4Token>;
        type U = u32x16<archmage::X64V4Token>;
        assert_eq!(xs.len(), ins_a[0].len());
        for c in 1..3 {
            assert_eq!(xs.len(), ins_a[c].len());
        }
        for c in 0..3 {
            assert_eq!(xs.len(), ins_b[c].len());
            assert_eq!(xs.len(), outs_a[c].len());
            assert_eq!(xs.len(), outs_b[c].len());
            assert!(luts[c].len() >= 2 && max_index < (luts[c].len() - 1) as f32);
        }
        assert!(max_index >= 0.0);
        let token = token.v4();
        let min_v = V::splat_t(token, min);
        let step_v = V::splat_t(token, inv_step);
        let zero = V::zero_t(token);
        let max_v = V::splat_t(token, max_index);
        let offset_v = V::splat_t(token, offset);
        let scale_v = V::splat_t(token, scale);
        let bm_v = V::splat_t(token, band_mul);
        let gain_v = gains.map(|g| V::splat_t(token, g));
        let or_v = V::zero_t(token);
        let (xc, xt) = V::partition_slice_t(token, xs);
        let pa = ins_a.map(|s| V::partition_slice_t(token, s));
        let pb = ins_b.map(|s| V::partition_slice_t(token, s));
        let oa = outs_a.map(|s| V::partition_slice_mut_t(token, s));
        let ob = outs_b.map(|s| V::partition_slice_mut_t(token, s));
        // Same check-free indexing contract as the generic kernel:
        // cut every stream's chunk view to the shared block count.
        let n = xc.len();
        let pa_c = [&pa[0].0[..n], &pa[1].0[..n], &pa[2].0[..n]];
        let pb_c = [&pb[0].0[..n], &pb[1].0[..n], &pb[2].0[..n]];
        let pa_t = [pa[0].1, pa[1].1, pa[2].1];
        let pb_t = [pb[0].1, pb[1].1, pb[2].1];
        let [(oa0c, oa0t), (oa1c, oa1t), (oa2c, oa2t)] = oa;
        let [(ob0c, ob0t), (ob1c, ob1t), (ob2c, ob2t)] = ob;
        let oa_c = [&mut oa0c[..n], &mut oa1c[..n], &mut oa2c[..n]];
        let ob_c = [&mut ob0c[..n], &mut ob1c[..n], &mut ob2c[..n]];
        let oa_t = [oa0t, oa1t, oa2t];
        let ob_t = [ob0t, ob1t, ob2t];
        // Two-phase: shared bracket per stack tile, then per-channel
        // gather/apply — same shape as the generic tile kernel.
        const TB: usize = 8;
        let mut lanes_t = [[0u32; 16]; TB];
        let mut frac_t = [[0f32; 16]; TB];
        let mut blk = 0usize;
        while blk < n {
            let tb = (n - blk).min(TB);
            for t in 0..tb {
                let idx = ((V::load_t(token, &xc[blk + t]) - min_v) * step_v)
                    .max(zero)
                    .min(max_v);
                let frac = idx - idx.floor();
                lanes_t[t] = U::from_raw_t(token, idx.to_i32().raw()).to_array();
                frac_t[t] = frac.to_array();
            }
            for c in 0..3 {
                let pa_c = pa_c[c];
                let pb_c = pb_c[c];
                let oa_c = &mut *oa_c[c];
                let ob_c = &mut *ob_c[c];
                for t in 0..tb {
                    let lanes = U::from_array_t(token, lanes_t[t]);
                    let lo = V::gather_or(luts[c], lanes, or_v);
                    let hi = V::gather_or(&luts[c][1..], lanes, or_v);
                    let frac = V::from_array_t(token, frac_t[t]);
                    let value = lo + frac * (hi - lo);
                    let sens = P::exp16::<archmage::X64V4Token>((value + offset_v) * scale_v);
                    let bg = bm_v * sens;
                    let va = V::load_t(token, &pa_c[blk + t]);
                    let vb = V::load_t(token, &pb_c[blk + t]);
                    ((va * bg) * gain_v[c]).store(&mut oa_c[blk + t]);
                    ((vb * bg) * gain_v[c]).store(&mut ob_c[blk + t]);
                }
            }
            blk += tb;
        }
        for t in 0..xt.len() {
            let idx = ((xt[t] - min) * inv_step).clamp(0.0, max_index);
            let floor = super::scalar_floor(idx);
            for c in 0..3 {
                let lut = luts[c];
                let lo = (floor as usize).min(lut.len() - 2);
                let value = lut[lo] + (idx - floor) * (lut[lo + 1] - lut[lo]);
                let sens = P::scalar_exp((value + offset) * scale);
                let bg = band_mul * sens;
                oa_t[c][t] = (pa_t[c][t] * bg) * gains[c];
                ob_t[c][t] = (pb_t[c][t] * bg) * gains[c];
            }
        }
    }
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
pub(crate) fn safe_pow_with_offset_into_scalar_t(
    token: ScalarToken,
    xs: &[f32],
    out: &mut [f32],
    offset: f32,
    p: f32,
    offset_pow_p: f32,
) {
    width8::safe_pow_with_offset_kernel::<_, Midp>(token, xs, out, offset, p, offset_pow_p);
}

pub(crate) fn vexp_into_scalar_t(token: ScalarToken, xs: &[f32], out: &mut [f32]) {
    width8::vexp_kernel::<_, Midp>(token, xs, out);
}

pub(crate) fn vlog_into_scalar_t(token: ScalarToken, xs: &[f32], out: &mut [f32]) {
    width8::vlog_kernel::<_, Midp>(token, xs, out);
}

pub(crate) fn vpow_into_scalar_t(token: ScalarToken, xs: &[f32], out: &mut [f32], p: f32) {
    width8::vpow_kernel::<_, Midp>(token, xs, out, p);
}

pub(crate) fn vaxpy_into_scalar_t(token: ScalarToken, dst: &mut [f32], src: &[f32], a: f32) {
    width8::vaxpy_kernel::<_, Midp>(token, dst, src, a);
}

pub(crate) fn vscale_into_scalar_t(token: ScalarToken, dst: &mut [f32], src: &[f32], a: f32) {
    width8::vscale_kernel::<_, Midp>(token, dst, src, a);
}

pub(crate) fn vabs_diff_into_scalar_t(token: ScalarToken, out: &mut [f32], x: &[f32], y: &[f32]) {
    width8::vabs_diff_kernel::<_, Midp>(token, out, x, y);
}

pub(crate) fn vmin_abs_into_scalar_t(token: ScalarToken, out: &mut [f32], x: &[f32], y: &[f32]) {
    width8::vmin_abs_kernel::<_, Midp>(token, out, x, y);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn mask_fold_pool_scalar_t(
    token: ScalarToken,
    d: [&mut [f32]; 3],
    t: [&[f32]; 3],
    r: [&[f32]; 3],
    mm: [&[f32]; 3],
    offset: f32,
    p: f32,
    offset_pow_p: f32,
    qs: [f32; 3],
    offset_pow_qs: [f32; 3],
    w: &[[f32; 3]; 3],
    d_max: f32,
) {
    width8::mask_fold_pool_kernel::<_, Midp>(
        token,
        d,
        t,
        r,
        mm,
        offset,
        p,
        offset_pow_p,
        qs,
        offset_pow_qs,
        w,
        d_max,
    );
}

pub(crate) fn vweber_band_into_scalar_t(
    token: ScalarToken,
    band: &mut [f32],
    log: &mut [f32],
    fine: &[f32],
    img_exp: &[f32],
    exp_l: &[f32],
) {
    width8::vweber_band_kernel::<_, Midp>(token, band, log, fine, img_exp, exp_l);
}

pub(crate) fn vweber_band_nolog_into_scalar_t(
    token: ScalarToken,
    band: &mut [f32],
    fine: &[f32],
    img_exp: &[f32],
    exp_l: &[f32],
) {
    width8::vweber_band_nolog_kernel::<_, Midp>(token, band, fine, img_exp, exp_l);
}

pub(crate) fn vscale2_into_scalar_t(
    token: ScalarToken,
    d1: &mut [f32],
    d2: &mut [f32],
    src: &[f32],
    a1: f32,
    a2: f32,
) {
    width8::vscale2_kernel::<_, Midp>(token, d1, d2, src, a1, a2);
}

pub(crate) fn vaxpy2_into_scalar_t(
    token: ScalarToken,
    d1: &mut [f32],
    d2: &mut [f32],
    src: &[f32],
    a1: f32,
    a2: f32,
) {
    width8::vaxpy2_kernel::<_, Midp>(token, d1, d2, src, a1, a2);
}

pub(crate) fn vmul2_scale2_pair_into_scalar_t(
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

pub(crate) fn vfir_into_scalar_t(
    token: ScalarToken,
    dst: &mut [f32],
    srcs: &[&[f32]],
    coeffs: &[f32],
    off: usize,
) {
    width8::vfir_into_kernel::<_, Midp>(token, dst, srcs, coeffs, off)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn vfir2_into_scalar_t(
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
pub(crate) fn vabs_diff_pow_scalar_t(
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

pub(crate) fn vabs_diff_mul_lp2_sum_scalar_t(
    token: ScalarToken,
    t: &[f32],
    r: &[f32],
    s: &[f32],
) -> f32 {
    width8::vabs_diff_mul_lp2_sum_kernel::<_, Midp>(token, t, r, s)
}

pub(crate) fn vxcm_pool_clamp_4ch_sqsum_partial_scalar_t(
    token: ScalarToken,
    d: &[&[f32]; 4],
    t: &[&[f32]; 4],
    w: &[[f32; 4]; 4],
    d_max: f32,
) -> [f32; 4] {
    width8::vxcm_pool_clamp_4ch_sqsum_partial_kernel::<_, Midp>(token, d, t, w, d_max)
}

include!("tier_v3.rs");
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
    archmage::attuned!(
        safe_pow_with_offset_into(xs, out, offset, p, offset_pow_p),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
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
    archmage::attuned!(
        vexp_into(xs, out),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    )
}

/// `out[i] = ln(xs[i])`. Positive inputs only. Reusable from Chunk 5.
#[inline]
#[allow(dead_code)]
pub fn vlog_into(xs: &[f32], out: &mut [f32]) {
    debug_assert_eq!(xs.len(), out.len());
    archmage::attuned!(
        vlog_into(xs, out),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    )
}

/// `out[i] = xs[i].powf(p)`. Positive inputs only. Reusable from Chunk 5.
#[inline]
#[allow(dead_code)]
pub fn vpow_into(xs: &[f32], out: &mut [f32], p: f32) {
    debug_assert_eq!(xs.len(), out.len());
    archmage::attuned!(
        vpow_into(xs, out, p),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    )
}

// ---------------------------------------------------------------------------
// Elementwise dispatchers — the video pipeline's glue loops.
// ---------------------------------------------------------------------------

/// `dst[i] += a * src[i]`. Temporal FIR accumulate.
#[inline]
pub fn vaxpy_into(dst: &mut [f32], src: &[f32], a: f32) {
    debug_assert_eq!(dst.len(), src.len());
    archmage::attuned!(
        vaxpy_into(dst, src, a),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    )
}

/// `dst[i] = a * src[i]` into a distinct buffer.
#[inline]
pub fn vscale_into(dst: &mut [f32], src: &[f32], a: f32) {
    debug_assert_eq!(dst.len(), src.len());
    archmage::attuned!(
        vscale_into(dst, src, a),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    )
}

/// `out[i] = |x[i] - y[i]|`.
#[inline]
pub fn vabs_diff_into(out: &mut [f32], x: &[f32], y: &[f32]) {
    debug_assert_eq!(out.len(), x.len());
    debug_assert_eq!(out.len(), y.len());
    archmage::attuned!(
        vabs_diff_into(out, x, y),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    )
}

/// `out[i] = min(|x[i]|, |y[i]|)` — mutual-mask raw term.
#[inline]
pub fn vmin_abs_into(out: &mut [f32], x: &[f32], y: &[f32]) {
    debug_assert_eq!(out.len(), x.len());
    debug_assert_eq!(out.len(), y.len());
    archmage::attuned!(
        vmin_abs_into(out, x, y),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
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
    archmage::attuned!(
        vweber_band_into(band, log, fine, img_exp, exp_l),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    )
}

/// `band[i] = clamp((fine[i] − img_exp[i]) / max(exp_l[i], 0.01), ±1000)`
/// — [`vweber_band_into`] without the `log_l_bkg` output, for the
/// pyramids whose `log_l_bkg` is never consumed.
pub fn vweber_band_nolog_into(band: &mut [f32], fine: &[f32], img_exp: &[f32], exp_l: &[f32]) {
    assert_eq!(band.len(), fine.len());
    assert_eq!(band.len(), img_exp.len());
    assert_eq!(band.len(), exp_l.len());
    archmage::attuned!(
        vweber_band_nolog_into(band, fine, img_exp, exp_l),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    )
}

/// `d1[i] = a1·src[i]`, `d2[i] = a2·src[i]` — shared-source dual
/// scale (transient + sustained FIR over the same plane).
#[inline]
pub fn vscale2_into(d1: &mut [f32], d2: &mut [f32], src: &[f32], a1: f32, a2: f32) {
    debug_assert_eq!(d1.len(), src.len());
    debug_assert_eq!(d2.len(), src.len());
    archmage::attuned!(
        vscale2_into(d1, d2, src, a1, a2),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    )
}

/// `d1[i] += a1·src[i]`, `d2[i] += a2·src[i]` — shared-source dual
/// accumulate.
#[inline]
pub fn vaxpy2_into(d1: &mut [f32], d2: &mut [f32], src: &[f32], a1: f32, a2: f32) {
    debug_assert_eq!(d1.len(), src.len());
    debug_assert_eq!(d2.len(), src.len());
    archmage::attuned!(
        vaxpy2_into(d1, d2, src, a1, a2),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    )
}

/// `dst[i] = Σ_k coeffs[k]·srcs[k][off+i]` — fused N-tap temporal
/// FIR, one pass over all source planes. Band callers pass the
/// band's start index in `off`.
#[inline]
pub fn vfir_into(dst: &mut [f32], srcs: &[&[f32]], coeffs: &[f32], off: usize) {
    debug_assert_eq!(srcs.len(), coeffs.len());
    archmage::attuned!(
        vfir_into(dst, srcs, coeffs, off),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
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
    archmage::attuned!(
        vfir2_into(d0, d1, srcs, coeffs0, coeffs1, off),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
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
    archmage::attuned!(
        vmul2_scale2_pair_into(o1, o2, x1, x2, w, a, b),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    )
}

/// Raw `Σ u²` for `lp_norm_mean(|t−r|·s, 2.0)` — the video baseband
/// pooled difference, fused (no intermediate plane). Apply
/// [`lp2_finish`] to the folded total for the final norm.
#[inline]
pub fn vabs_diff_mul_lp2_sum(t: &[f32], r: &[f32], s: &[f32]) -> f32 {
    debug_assert_eq!(t.len(), r.len());
    debug_assert_eq!(t.len(), s.len());
    archmage::attuned!(
        vabs_diff_mul_lp2_sum(t, r, s),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
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
    archmage::attuned!(
        vabs_diff_pow(out, x, y, offset, p, offset_pow_p),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
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
    archmage::attuned!(
        vxcm_pool_clamp_4ch_sqsum_partial(d, t, w, d_max),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    )
}

/// Three-channel masking fold, fully fused into one pass: for each
/// element and channel c,
/// `term_c = (mm_c + offset)^qs_c − offset_pow_qs_c`,
/// `pw_c = (|t_c − r_c| + offset)^p − offset_pow_p`,
/// `m_c = w[0][c]·term_0 + w[1][c]·term_1 + w[2][c]·term_2`,
/// `d_c = d_max·(pw_c/(1+m_c)) / (d_max + pw_c/(1+m_c))`.
/// Bit-identical to chaining `safe_pow_with_offset` on `mm`,
/// `vabs_diff_pow` on `(t, r)`, then the scalar XCM pool + soft clamp.
/// `w` is `XCM_3X3` indexed `[in][c]`.
#[allow(clippy::too_many_arguments)]
pub fn mask_fold_pool_into(
    d: [&mut [f32]; 3],
    t: [&[f32]; 3],
    r: [&[f32]; 3],
    mm: [&[f32]; 3],
    offset: f32,
    p: f32,
    offset_pow_p: f32,
    qs: [f32; 3],
    offset_pow_qs: [f32; 3],
    w: &[[f32; 3]; 3],
    d_max: f32,
) {
    debug_assert!(d[1].len() == d[0].len() && d[2].len() == d[0].len());
    archmage::attuned!(
        mask_fold_pool(
            d,
            t,
            r,
            mm,
            offset,
            p,
            offset_pow_p,
            qs,
            offset_pow_qs,
            w,
            d_max
        ),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
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
    archmage::attuned!(
        gather_lerp(xs, lut, min, inv_step, max_index, 0.0, 1.0, out),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
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
    archmage::attuned!(
        gather_lerp_exp(xs, lut, min, inv_step, max_index, offset, scale, out),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    )
}
/// Three-channel fused variant of [`gather_lerp_exp_into`]: the three
/// `luts` share `xs`'s uniform axis, so the bracket/floor/lane-index
/// chain runs once per block rather than once per channel, and each
/// channel's sensitivity applies to two planes in the same pass:
/// `out[c][i] = (in[c][i] * (band_mul * sens_c[i])) * gains[c]`.
/// Bit-identical to three `gather_lerp_exp` calls plus a per-channel
/// `(v * (band_mul*s)) * gain` multiply pass.
#[allow(clippy::too_many_arguments)]
pub fn gather_lerp_exp3_apply_into(
    xs: &[f32],
    luts: [&[f32]; 3],
    min: f32,
    inv_step: f32,
    max_index: f32,
    offset: f32,
    scale: f32,
    band_mul: f32,
    gains: [f32; 3],
    ins_a: [&[f32]; 3],
    ins_b: [&[f32]; 3],
    outs_a: [&mut [f32]; 3],
    outs_b: [&mut [f32]; 3],
) {
    archmage::attuned!(
        gather_lerp_exp3_apply(
            xs, luts, min, inv_step, max_index, offset, scale, band_mul, gains, ins_a, ins_b,
            outs_a, outs_b
        ),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    )
}
pub(crate) fn gather_lerp_scalar_t(
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
pub(crate) fn gather_lerp_exp_scalar_t(
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
#[allow(clippy::too_many_arguments)]
pub(crate) fn gather_lerp_exp3_apply_scalar_t(
    token: ScalarToken,
    xs: &[f32],
    luts: [&[f32]; 3],
    min: f32,
    inv_step: f32,
    max_index: f32,
    offset: f32,
    scale: f32,
    band_mul: f32,
    gains: [f32; 3],
    ins_a: [&[f32]; 3],
    ins_b: [&[f32]; 3],
    outs_a: [&mut [f32]; 3],
    outs_b: [&mut [f32]; 3],
) {
    width8::gather_lerp_exp3_apply_kernel::<_, Midp>(
        token, xs, luts, min, inv_step, max_index, offset, scale, band_mul, gains, ins_a, ins_b,
        outs_a, outs_b,
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
        magetypes::simd::generic::f32x8::splat_t(ScalarToken, x)
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
