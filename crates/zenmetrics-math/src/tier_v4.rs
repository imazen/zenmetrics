#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
mod v4 {
    use super::*;
    use archmage::X64V4Token;
    #[archmage::arcane]
    pub(crate) fn safe_pow_with_offset_into_v4(
        token: X64V4Token,
        xs: &[f32],
        out: &mut [f32],
        offset: f32,
        p: f32,
        offset_pow_p: f32,
    ) {
        width16::safe_pow_with_offset_kernel::<_, Midp>(token, xs, out, offset, p, offset_pow_p);
    }

    #[archmage::arcane]
    pub(crate) fn vexp_into_v4(token: X64V4Token, xs: &[f32], out: &mut [f32]) {
        width16::vexp_kernel::<_, Midp>(token, xs, out);
    }

    #[archmage::arcane]
    pub(crate) fn vlog_into_v4(token: X64V4Token, xs: &[f32], out: &mut [f32]) {
        width16::vlog_kernel::<_, Midp>(token, xs, out);
    }

    #[archmage::arcane]
    pub(crate) fn vpow_into_v4(token: X64V4Token, xs: &[f32], out: &mut [f32], p: f32) {
        width16::vpow_kernel::<_, Midp>(token, xs, out, p);
    }

    #[archmage::arcane]
    pub(crate) fn vaxpy_into_v4(token: X64V4Token, dst: &mut [f32], src: &[f32], a: f32) {
        width16::vaxpy_kernel::<_, Midp>(token, dst, src, a);
    }

    #[archmage::arcane]
    pub(crate) fn vscale_into_v4(token: X64V4Token, dst: &mut [f32], src: &[f32], a: f32) {
        width16::vscale_kernel::<_, Midp>(token, dst, src, a);
    }

    #[archmage::arcane]
    pub(crate) fn vabs_diff_into_v4(token: X64V4Token, out: &mut [f32], x: &[f32], y: &[f32]) {
        width16::vabs_diff_kernel::<_, Midp>(token, out, x, y);
    }

    #[archmage::arcane]
    pub(crate) fn vmin_abs_into_v4(token: X64V4Token, out: &mut [f32], x: &[f32], y: &[f32]) {
        width16::vmin_abs_kernel::<_, Midp>(token, out, x, y);
    }

    #[archmage::arcane]
    pub(crate) fn vweber_band_into_v4(
        token: X64V4Token,
        band: &mut [f32],
        log: &mut [f32],
        fine: &[f32],
        img_exp: &[f32],
        exp_l: &[f32],
    ) {
        width16::vweber_band_kernel::<_, Midp>(token, band, log, fine, img_exp, exp_l);
    }

    #[archmage::arcane]
    pub(crate) fn vweber_band_nolog_into_v4(
        token: X64V4Token,
        band: &mut [f32],
        fine: &[f32],
        img_exp: &[f32],
        exp_l: &[f32],
    ) {
        width16::vweber_band_nolog_kernel::<_, Midp>(token, band, fine, img_exp, exp_l);
    }

    #[archmage::arcane]
    pub(crate) fn vscale2_into_v4(
        token: X64V4Token,
        d1: &mut [f32],
        d2: &mut [f32],
        src: &[f32],
        a1: f32,
        a2: f32,
    ) {
        width16::vscale2_kernel::<_, Midp>(token, d1, d2, src, a1, a2);
    }

    #[archmage::arcane]
    pub(crate) fn vaxpy2_into_v4(
        token: X64V4Token,
        d1: &mut [f32],
        d2: &mut [f32],
        src: &[f32],
        a1: f32,
        a2: f32,
    ) {
        width16::vaxpy2_kernel::<_, Midp>(token, d1, d2, src, a1, a2);
    }

    #[archmage::arcane]
    pub(crate) fn vmul2_scale2_pair_into_v4(
        token: X64V4Token,
        o1: &mut [f32],
        o2: &mut [f32],
        x1: &[f32],
        x2: &[f32],
        w: &[f32],
        a: f32,
        b: f32,
    ) {
        width16::vmul2_scale2_pair_kernel::<_, Midp>(token, o1, o2, x1, x2, w, a, b);
    }

    #[archmage::arcane]
    pub(crate) fn vfir_into_v4(
        token: X64V4Token,
        dst: &mut [f32],
        srcs: &[&[f32]],
        coeffs: &[f32],
        off: usize,
    ) {
        width16::vfir_into_kernel::<_, Midp>(token, dst, srcs, coeffs, off)
    }

    #[allow(clippy::too_many_arguments)]
    #[archmage::arcane]
    pub(crate) fn vfir2_into_v4(
        token: X64V4Token,
        d0: &mut [f32],
        d1: &mut [f32],
        srcs: &[&[f32]],
        coeffs0: &[f32],
        coeffs1: &[f32],
        off: usize,
    ) {
        width16::vfir2_into_kernel::<_, Midp>(token, d0, d1, srcs, coeffs0, coeffs1, off)
    }

    #[allow(clippy::too_many_arguments)]
    #[archmage::arcane]
    pub(crate) fn vabs_diff_pow_v4(
        token: X64V4Token,
        out: &mut [f32],
        x: &[f32],
        y: &[f32],
        offset: f32,
        p: f32,
        offset_pow_p: f32,
    ) {
        width16::vabs_diff_pow_kernel::<_, Midp>(token, out, x, y, offset, p, offset_pow_p)
    }

    #[archmage::arcane]
    pub(crate) fn vabs_diff_mul_lp2_sum_v4(
        token: X64V4Token,
        t: &[f32],
        r: &[f32],
        s: &[f32],
    ) -> f32 {
        width16::vabs_diff_mul_lp2_sum_kernel::<_, Midp>(token, t, r, s)
    }

    #[archmage::arcane]
    pub(crate) fn vxcm_pool_clamp_4ch_sqsum_partial_v4(
        token: X64V4Token,
        d: &[&[f32]; 4],
        t: &[&[f32]; 4],
        w: &[[f32; 4]; 4],
        d_max: f32,
    ) -> [f32; 4] {
        width16::vxcm_pool_clamp_4ch_sqsum_partial_kernel::<_, Midp>(token, d, t, w, d_max)
    }

    #[archmage::arcane]
    pub(crate) fn gather_lerp_v4(
        token: X64V4Token,
        xs: &[f32],
        lut: &[f32],
        min: f32,
        inv_step: f32,
        max_index: f32,
        offset: f32,
        scale: f32,
        out: &mut [f32],
    ) {
        width16::gather_lerp_kernel::<_, Midp, false>(
            token, xs, lut, min, inv_step, max_index, offset, scale, out,
        )
    }
    #[archmage::arcane]
    pub(crate) fn gather_lerp_exp_v4(
        token: X64V4Token,
        xs: &[f32],
        lut: &[f32],
        min: f32,
        inv_step: f32,
        max_index: f32,
        offset: f32,
        scale: f32,
        out: &mut [f32],
    ) {
        width16::gather_lerp_kernel::<_, Midp, true>(
            token, xs, lut, min, inv_step, max_index, offset, scale, out,
        )
    }
}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
use v4::*;
