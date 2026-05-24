//! sRGB byte / linear-f32 → DKL opponent (A, RG, VY) planar.
//!
//! Re-uses the canonical 256-entry sRGB→linear LUT and the combined
//! 3×3 sRGB-linear→DKL matrix from `cvvdp_gpu::kernels::color` +
//! `cvvdp_gpu::params`. Same numerics as host_scalar, but converts
//! to planar f32 in one allocation per channel instead of three.

use alloc::vec::Vec;

use cvvdp_gpu::kernels::color::SRGB8_TO_LINEAR_LUT;
use cvvdp_gpu::params::{DisplayModel, SRGB_LINEAR_TO_DKL as M};

/// sRGB packed-u8 (RGBRGB…) → DKL planar (A, RG, VY).
///
/// Strided-friendly: source slice is contiguous `RGB` triples,
/// destination planes are row-major contiguous f32 with stride ==
/// width. Caller-owned destination Vecs are sized + filled.
pub(crate) fn srgb_to_dkl_planar(
    src: &[u8],
    width: usize,
    height: usize,
    display: DisplayModel,
    out_a: &mut Vec<f32>,
    out_rg: &mut Vec<f32>,
    out_vy: &mut Vec<f32>,
) {
    let n = width * height;
    debug_assert_eq!(src.len(), n * 3);
    out_a.resize(n, 0.0);
    out_rg.resize(n, 0.0);
    out_vy.resize(n, 0.0);

    let s = display.y_peak - display.y_black;
    let bias = display.y_black + display.y_refl;

    for i in 0..n {
        let r = src[i * 3] as usize;
        let g = src[i * 3 + 1] as usize;
        let b = src[i * 3 + 2] as usize;
        let lin_r = SRGB8_TO_LINEAR_LUT[r];
        let lin_g = SRGB8_TO_LINEAR_LUT[g];
        let lin_b = SRGB8_TO_LINEAR_LUT[b];
        let lr = s * lin_r + bias;
        let lg = s * lin_g + bias;
        let lb = s * lin_b + bias;
        out_a[i] = M[0][0] * lr + M[0][1] * lg + M[0][2] * lb;
        out_rg[i] = M[1][0] * lr + M[1][1] * lg + M[1][2] * lb;
        out_vy[i] = M[2][0] * lr + M[2][1] * lg + M[2][2] * lb;
    }
}

/// Linear-f32 RGB planes (display-relative `[0, 1]`) → DKL planar
/// (A, RG, VY).
///
/// Each plane has the form `[row0_pixel0, row0_pixel1, ..., row1_pixel0, ...]`
/// where each row is `width` floats and the stride between rows is
/// `padded_width` floats (`padded_width >= width`). When `padded_width
/// == width`, the plane is row-tight; the function handles both.
///
/// Output planes are row-tight (`width × height`).
pub(crate) fn linear_planes_to_dkl_planar(
    r: &[f32],
    g: &[f32],
    b: &[f32],
    width: usize,
    height: usize,
    padded_width: usize,
    display: DisplayModel,
    out_a: &mut Vec<f32>,
    out_rg: &mut Vec<f32>,
    out_vy: &mut Vec<f32>,
) {
    debug_assert!(padded_width >= width);
    debug_assert_eq!(r.len(), padded_width * height);
    debug_assert_eq!(g.len(), padded_width * height);
    debug_assert_eq!(b.len(), padded_width * height);
    let n = width * height;
    out_a.resize(n, 0.0);
    out_rg.resize(n, 0.0);
    out_vy.resize(n, 0.0);

    let s = display.y_peak - display.y_black;
    let bias = display.y_black + display.y_refl;

    for y in 0..height {
        let src_row = y * padded_width;
        let dst_row = y * width;
        for x in 0..width {
            let lin_r = r[src_row + x];
            let lin_g = g[src_row + x];
            let lin_b = b[src_row + x];
            let lr = s * lin_r + bias;
            let lg = s * lin_g + bias;
            let lb = s * lin_b + bias;
            let i = dst_row + x;
            out_a[i] = M[0][0] * lr + M[0][1] * lg + M[0][2] * lb;
            out_rg[i] = M[1][0] * lr + M[1][1] * lg + M[1][2] * lb;
            out_vy[i] = M[2][0] * lr + M[2][1] * lg + M[2][2] * lb;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_byte_matches_scalar_reference() {
        // Identical pixel data feeding our planar fn should match
        // cvvdp-gpu's host scalar (which is the goldens contract).
        use cvvdp_gpu::kernels::color::srgb_byte_to_dkl_scalar;
        let display = DisplayModel::STANDARD_4K;
        let w = 4;
        let h = 4;
        let mut src = vec![0u8; w * h * 3];
        for i in 0..w * h {
            src[i * 3] = (i * 7) as u8;
            src[i * 3 + 1] = (i * 11) as u8;
            src[i * 3 + 2] = (i * 13) as u8;
        }
        let mut a = vec![];
        let mut rg = vec![];
        let mut vy = vec![];
        srgb_to_dkl_planar(&src, w, h, display, &mut a, &mut rg, &mut vy);
        for i in 0..w * h {
            let (ea, erg, evy) = srgb_byte_to_dkl_scalar(
                src[i * 3],
                src[i * 3 + 1],
                src[i * 3 + 2],
                display.y_peak,
                display.y_black,
                display.y_refl,
            );
            assert!((a[i] - ea).abs() < 1e-6);
            assert!((rg[i] - erg).abs() < 1e-6);
            assert!((vy[i] - evy).abs() < 1e-6);
        }
    }

    #[test]
    fn linear_planes_strided_equals_tight() {
        // padded_width > width path must produce same output as
        // tight planes when only the in-bounds pixels are the same.
        let display = DisplayModel::STANDARD_4K;
        let w = 5;
        let h = 3;
        let stride = 8;
        let mut r_tight = vec![0.0f32; w * h];
        let mut g_tight = vec![0.0f32; w * h];
        let mut b_tight = vec![0.0f32; w * h];
        for i in 0..w * h {
            r_tight[i] = (i as f32) * 0.01;
            g_tight[i] = (i as f32) * 0.02;
            b_tight[i] = (i as f32) * 0.005;
        }
        let mut r_pad = vec![999.0f32; stride * h];
        let mut g_pad = vec![999.0f32; stride * h];
        let mut b_pad = vec![999.0f32; stride * h];
        for y in 0..h {
            for x in 0..w {
                r_pad[y * stride + x] = r_tight[y * w + x];
                g_pad[y * stride + x] = g_tight[y * w + x];
                b_pad[y * stride + x] = b_tight[y * w + x];
            }
        }
        let mut a_t = vec![];
        let mut rg_t = vec![];
        let mut vy_t = vec![];
        linear_planes_to_dkl_planar(
            &r_tight, &g_tight, &b_tight, w, h, w, display, &mut a_t, &mut rg_t, &mut vy_t,
        );
        let mut a_p = vec![];
        let mut rg_p = vec![];
        let mut vy_p = vec![];
        linear_planes_to_dkl_planar(
            &r_pad, &g_pad, &b_pad, w, h, stride, display, &mut a_p, &mut rg_p, &mut vy_p,
        );
        for i in 0..w * h {
            assert!((a_t[i] - a_p[i]).abs() < 1e-6);
            assert!((rg_t[i] - rg_p[i]).abs() < 1e-6);
            assert!((vy_t[i] - vy_p[i]).abs() < 1e-6);
        }
    }
}
