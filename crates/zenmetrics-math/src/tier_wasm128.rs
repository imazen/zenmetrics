#[cfg(target_arch = "wasm32")]
mod wasm128 {
    use super::*;
    use archmage::Wasm128Token;
    #[archmage::attune]
    pub(crate) fn safe_pow_with_offset_into_wasm128_t(
        token: Wasm128Token,
        xs: &[f32],
        out: &mut [f32],
        offset: f32,
        p: f32,
        offset_pow_p: f32,
    ) {
        width8::safe_pow_with_offset_kernel::<_, Midp>(token, xs, out, offset, p, offset_pow_p);
    }

    #[archmage::attune]
    pub(crate) fn vexp_into_wasm128_t(token: Wasm128Token, xs: &[f32], out: &mut [f32]) {
        width8::vexp_kernel::<_, Midp>(token, xs, out);
    }

    #[archmage::attune]
    pub(crate) fn vlog_into_wasm128_t(token: Wasm128Token, xs: &[f32], out: &mut [f32]) {
        width8::vlog_kernel::<_, Midp>(token, xs, out);
    }

    #[archmage::attune]
    pub(crate) fn vpow_into_wasm128_t(token: Wasm128Token, xs: &[f32], out: &mut [f32], p: f32) {
        width8::vpow_kernel::<_, Midp>(token, xs, out, p);
    }

    #[archmage::attune]
    pub(crate) fn vpow_libm_into_wasm128_t(token: Wasm128Token, xs: &[f32], out: &mut [f32], p: f32) {
        fdlibm::vpow_libm_kernel_8(token, xs, out, p);
    }

    #[archmage::attune]
    pub(crate) fn vpowf_libm_pairs_into_wasm128_t(
        token: Wasm128Token,
        xs: &[f32],
        ys: &[f32],
        out: &mut [f32],
    ) {
        fdlibm::vpowf_libm_pairs_kernel_8(token, xs, ys, out);
    }

    #[archmage::attune]
    pub(crate) fn vpow64_libm_into_wasm128_t(token: Wasm128Token, xs: &[f64], out: &mut [f64], p: f64) {
        fdlibm64::vpow64_libm_kernel_4(token, xs, out, p);
    }

    #[archmage::attune]
    pub(crate) fn vpow64_libm_pairs_into_wasm128_t(
        token: Wasm128Token,
        xs: &[f64],
        ys: &[f64],
        out: &mut [f64],
    ) {
        fdlibm64::vpow64_libm_pairs_kernel_4(token, xs, ys, out);
    }

    #[archmage::attune]
    pub(crate) fn vexpf_libm_into_wasm128_t(token: Wasm128Token, xs: &[f32], out: &mut [f32]) {
        fdlibm::vexpf_libm_kernel_8(token, xs, out);
    }

    #[archmage::attune]
    pub(crate) fn vaxpy_into_wasm128_t(token: Wasm128Token, dst: &mut [f32], src: &[f32], a: f32) {
        width8::vaxpy_kernel::<_, Midp>(token, dst, src, a);
    }

    #[archmage::attune]
    pub(crate) fn vscale_into_wasm128_t(token: Wasm128Token, dst: &mut [f32], src: &[f32], a: f32) {
        width8::vscale_kernel::<_, Midp>(token, dst, src, a);
    }

    #[archmage::attune]
    pub(crate) fn vabs_diff_into_wasm128_t(
        token: Wasm128Token,
        out: &mut [f32],
        x: &[f32],
        y: &[f32],
    ) {
        width8::vabs_diff_kernel::<_, Midp>(token, out, x, y);
    }

    #[archmage::attune]
    pub(crate) fn vmin_abs_into_wasm128_t(
        token: Wasm128Token,
        out: &mut [f32],
        x: &[f32],
        y: &[f32],
    ) {
        width8::vmin_abs_kernel::<_, Midp>(token, out, x, y);
    }

    #[archmage::attune]
    pub(crate) fn vweber_band_into_wasm128_t(
        token: Wasm128Token,
        band: &mut [f32],
        log: &mut [f32],
        fine: &[f32],
        img_exp: &[f32],
        exp_l: &[f32],
    ) {
        width8::vweber_band_kernel::<_, Midp>(token, band, log, fine, img_exp, exp_l);
    }

    #[archmage::attune]
    pub(crate) fn vweber_band_nolog_into_wasm128_t(
        token: Wasm128Token,
        band: &mut [f32],
        fine: &[f32],
        img_exp: &[f32],
        exp_l: &[f32],
    ) {
        width8::vweber_band_nolog_kernel::<_, Midp>(token, band, fine, img_exp, exp_l);
    }

    #[archmage::attune]
    pub(crate) fn vscale2_into_wasm128_t(
        token: Wasm128Token,
        d1: &mut [f32],
        d2: &mut [f32],
        src: &[f32],
        a1: f32,
        a2: f32,
    ) {
        width8::vscale2_kernel::<_, Midp>(token, d1, d2, src, a1, a2);
    }

    #[archmage::attune]
    pub(crate) fn vaxpy2_into_wasm128_t(
        token: Wasm128Token,
        d1: &mut [f32],
        d2: &mut [f32],
        src: &[f32],
        a1: f32,
        a2: f32,
    ) {
        width8::vaxpy2_kernel::<_, Midp>(token, d1, d2, src, a1, a2);
    }

    #[archmage::attune]
    pub(crate) fn vmul2_scale2_pair_into_wasm128_t(
        token: Wasm128Token,
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

    #[archmage::attune]
    pub(crate) fn vfir_into_wasm128_t(
        token: Wasm128Token,
        dst: &mut [f32],
        srcs: &[&[f32]],
        coeffs: &[f32],
        off: usize,
    ) {
        width8::vfir_into_kernel::<_, Midp>(token, dst, srcs, coeffs, off)
    }

    #[allow(clippy::too_many_arguments)]
    #[archmage::attune]
    pub(crate) fn vfir2_into_wasm128_t(
        token: Wasm128Token,
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
    #[archmage::attune]
    pub(crate) fn vabs_diff_pow_wasm128_t(
        token: Wasm128Token,
        out: &mut [f32],
        x: &[f32],
        y: &[f32],
        offset: f32,
        p: f32,
        offset_pow_p: f32,
    ) {
        width8::vabs_diff_pow_kernel::<_, Midp>(token, out, x, y, offset, p, offset_pow_p)
    }

    #[archmage::attune]
    pub(crate) fn vabs_diff_mul_lp2_sum_wasm128_t(
        token: Wasm128Token,
        t: &[f32],
        r: &[f32],
        s: &[f32],
    ) -> f32 {
        width8::vabs_diff_mul_lp2_sum_kernel::<_, Midp>(token, t, r, s)
    }

    #[archmage::attune]
    pub(crate) fn vxcm_pool_clamp_4ch_sqsum_partial_wasm128_t(
        token: Wasm128Token,
        d: &[&[f32]; 4],
        t: &[&[f32]; 4],
        w: &[[f32; 4]; 4],
        d_max: f32,
    ) -> [f32; 4] {
        width8::vxcm_pool_clamp_4ch_sqsum_partial_kernel::<_, Midp>(token, d, t, w, d_max)
    }

    #[archmage::attune]
    pub(crate) fn gather_lerp_wasm128_t(
        token: Wasm128Token,
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
    #[archmage::attune]
    pub(crate) fn gather_lerp_exp_wasm128_t(
        token: Wasm128Token,
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
    #[archmage::attune]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gather_lerp_exp3_apply_wasm128_t(
        token: Wasm128Token,
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
            token, xs, luts, min, inv_step, max_index, offset, scale, band_mul, gains, ins_a,
            ins_b, outs_a, outs_b,
        )
    }
    #[archmage::attune]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn mask_fold_pool_wasm128_t(
        token: Wasm128Token,
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
            token, d, t, r, mm, offset, p, offset_pow_p, qs, offset_pow_qs, w, d_max,
        )
    }
}
#[cfg(target_arch = "wasm32")]
use wasm128::*;
