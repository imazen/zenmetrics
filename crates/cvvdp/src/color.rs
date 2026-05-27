//! sRGB byte / linear-f32 → DKL opponent (A, RG, VY) planar.
//!
//! Honors `display.eotf` (Srgb / Pq / Hlg / Linear / Bt1886 /
//! Gamma(g)) and `display.primaries` (Bt709 / Bt2020 / DisplayP3 /
//! DciP3) via the host-scalar helpers in
//! [`cvvdp_gpu::kernels::color`]. For `Eotf::Srgb + Primaries::Bt709`
//! the result is bit-identical to the historical hardcoded path (and
//! to `cvvdp_gpu::host_scalar::predict_jod_still_3ch`). For any
//! other configuration the EOTF + matrix dispatch fires per-pixel.
//!
//! The 256-entry sRGB→linear LUT path is preserved for the common
//! sRGB case so the per-pixel work stays at "one LUT load + matrix
//! mul," matching v0.0.1 perf.

use alloc::vec::Vec;

use cvvdp_gpu::kernels::color::{
    SRGB8_TO_LINEAR_LUT, display_byte_to_dkl_scalar, display_linear_rgb_to_dkl_scalar,
};
use cvvdp_gpu::params::{DisplayModel, Eotf, Primaries};

/// sRGB packed-u8 (RGBRGB…) → DKL planar (A, RG, VY).
///
/// Strided-friendly: source slice is contiguous `RGB` triples,
/// destination planes are row-major contiguous f32 with stride ==
/// width. Caller-owned destination Vecs are sized + filled.
///
/// Branches on `(display.eotf, display.primaries)`:
///
/// - `(Srgb, Bt709)` → fast LUT path (bit-identical to v0.0.1).
/// - Any other combination → per-pixel
///   [`display_byte_to_dkl_scalar`] dispatch.
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

    if matches!(display.eotf, Eotf::Srgb) && matches!(display.primaries, Primaries::Bt709) {
        // Fast path — preserve v0.0.1 numerics bit-for-bit. The
        // hard-coded SRGB_LINEAR_TO_DKL matrix is also what
        // `Primaries::Bt709.linear_rgb_to_dkl()` returns, so this
        // path is equivalent to the dispatch path at the f32 level.
        let m = display.primaries.linear_rgb_to_dkl();
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
            out_a[i] = m[0][0] * lr + m[0][1] * lg + m[0][2] * lb;
            out_rg[i] = m[1][0] * lr + m[1][1] * lg + m[1][2] * lb;
            out_vy[i] = m[2][0] * lr + m[2][1] * lg + m[2][2] * lb;
        }
    } else {
        // Dispatch path — honors arbitrary EOTF + primaries. HLG
        // needs per-RGB-triple OOTF, hence the per-pixel call.
        for i in 0..n {
            let (a, rg, vy) =
                display_byte_to_dkl_scalar(src[i * 3], src[i * 3 + 1], src[i * 3 + 2], display);
            out_a[i] = a;
            out_rg[i] = rg;
            out_vy[i] = vy;
        }
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
#[allow(clippy::too_many_arguments)]
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

    // Linear-planes entry does NOT run an inverse-EOTF (the input is
    // already linear-light). It only applies the display-scaling step
    // and the per-primaries DKL matrix — equivalent to the
    // [`Eotf::Linear`] branch of `display_byte_to_dkl_scalar`. Per
    // upstream `display_model.py:348-349` the `linear` EOTF clips to
    // `[max(0.005, y_black), y_peak]` and then adds `y_refl`. The cvvdp
    // historical contract is "input in display-relative [0, 1] OR in
    // cd/m² when EOTF=Linear" — we preserve both shapes by routing
    // through `display_linear_rgb_to_dkl_scalar`, which itself
    // assumes [0, 1] display-relative (multiplies by `y_peak - y_black`
    // and adds `y_black + y_refl`). Callers passing cd/m² directly are
    // already running in the `Eotf::Linear + y_peak = 1.0` regime and
    // get the same numerics.
    //
    // BT.709 + Srgb-EOTF-implied call stays bit-identical to v0.0.1
    // via the fast-path branch — same matrix, same scalar arithmetic.
    if matches!(display.primaries, Primaries::Bt709) {
        let m = display.primaries.linear_rgb_to_dkl();
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
                out_a[i] = m[0][0] * lr + m[0][1] * lg + m[0][2] * lb;
                out_rg[i] = m[1][0] * lr + m[1][1] * lg + m[1][2] * lb;
                out_vy[i] = m[2][0] * lr + m[2][1] * lg + m[2][2] * lb;
            }
        }
    } else {
        for y in 0..height {
            let src_row = y * padded_width;
            let dst_row = y * width;
            for x in 0..width {
                let lin_r = r[src_row + x];
                let lin_g = g[src_row + x];
                let lin_b = b[src_row + x];
                let (a, rg, vy) = display_linear_rgb_to_dkl_scalar(lin_r, lin_g, lin_b, display);
                let i = dst_row + x;
                out_a[i] = a;
                out_rg[i] = rg;
                out_vy[i] = vy;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cvvdp_gpu::params::{DisplayModel, Eotf, Primaries};

    /// W44 test naming: each test pins ONE upstream parity claim.

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

    /// Fast-path BT.709+Srgb must produce BIT-IDENTICAL output to
    /// the per-pixel `display_byte_to_dkl_scalar` dispatch path on
    /// the STANDARD_4K display. This pins our v0.0.1 numerical
    /// contract against any future re-routing through the dispatch
    /// helper.
    #[test]
    fn fast_path_matches_dispatch_on_standard_4k() {
        use cvvdp_gpu::kernels::color::display_byte_to_dkl_scalar;
        let display = DisplayModel::STANDARD_4K;
        let w = 8;
        let h = 8;
        let mut src = vec![0u8; w * h * 3];
        for i in 0..w * h {
            src[i * 3] = (i * 31).wrapping_mul(7) as u8;
            src[i * 3 + 1] = (i * 37).wrapping_mul(11) as u8;
            src[i * 3 + 2] = (i * 41).wrapping_mul(13) as u8;
        }
        let mut a = vec![];
        let mut rg = vec![];
        let mut vy = vec![];
        srgb_to_dkl_planar(&src, w, h, display, &mut a, &mut rg, &mut vy);
        for i in 0..w * h {
            let (ea, erg, evy) =
                display_byte_to_dkl_scalar(src[i * 3], src[i * 3 + 1], src[i * 3 + 2], display);
            // Bit-identical: the fast path uses the exact same
            // matrix Primaries::Bt709 returns.
            assert_eq!(a[i].to_bits(), ea.to_bits(), "A mismatch at i={i}");
            assert_eq!(rg[i].to_bits(), erg.to_bits(), "RG mismatch at i={i}");
            assert_eq!(vy[i].to_bits(), evy.to_bits(), "VY mismatch at i={i}");
        }
    }

    /// PQ EOTF + BT.2020 primaries (e.g. `standard_hdr_pq` preset)
    /// must use the dispatch path. Verifies the dispatch produces
    /// per-pixel output that matches `display_byte_to_dkl_scalar`.
    #[test]
    fn dispatch_path_matches_scalar_for_pq_bt2020() {
        use cvvdp_gpu::kernels::color::display_byte_to_dkl_scalar;
        let display = DisplayModel {
            y_peak: 1500.0,
            y_black: 0.0015,
            y_refl: 10.0_f32 / core::f32::consts::PI * 0.005,
            eotf: Eotf::Pq,
            primaries: Primaries::Bt2020,
            e_ambient_lux: 10.0,
            k_refl: 0.005,
        };
        let w = 4;
        let h = 4;
        let mut src = vec![0u8; w * h * 3];
        for i in 0..w * h {
            src[i * 3] = (i * 17) as u8;
            src[i * 3 + 1] = (i * 23) as u8;
            src[i * 3 + 2] = (i * 29) as u8;
        }
        let mut a = vec![];
        let mut rg = vec![];
        let mut vy = vec![];
        srgb_to_dkl_planar(&src, w, h, display, &mut a, &mut rg, &mut vy);
        for i in 0..w * h {
            let (ea, erg, evy) =
                display_byte_to_dkl_scalar(src[i * 3], src[i * 3 + 1], src[i * 3 + 2], display);
            assert!(
                (a[i] - ea).abs() < 1e-4,
                "A mismatch at i={i}: {} vs {}",
                a[i],
                ea
            );
            assert!((rg[i] - erg).abs() < 1e-4, "RG mismatch at i={i}");
            assert!((vy[i] - evy).abs() < 1e-4, "VY mismatch at i={i}");
        }
    }

    /// HLG EOTF + BT.2020 primaries (e.g. `standard_hdr_hlg` preset).
    /// HLG needs the per-RGB-triple OOTF so this exercises that
    /// branch end-to-end through the planar entry point.
    #[test]
    fn dispatch_path_matches_scalar_for_hlg_bt2020() {
        use cvvdp_gpu::kernels::color::display_byte_to_dkl_scalar;
        let display = DisplayModel {
            y_peak: 1500.0,
            y_black: 0.0015,
            y_refl: 10.0_f32 / core::f32::consts::PI * 0.005,
            eotf: Eotf::Hlg,
            primaries: Primaries::Bt2020,
            e_ambient_lux: 10.0,
            k_refl: 0.005,
        };
        let w = 4;
        let h = 4;
        let mut src = vec![0u8; w * h * 3];
        for i in 0..w * h {
            src[i * 3] = (i * 17) as u8;
            src[i * 3 + 1] = (i * 23) as u8;
            src[i * 3 + 2] = (i * 29) as u8;
        }
        let mut a = vec![];
        let mut rg = vec![];
        let mut vy = vec![];
        srgb_to_dkl_planar(&src, w, h, display, &mut a, &mut rg, &mut vy);
        for i in 0..w * h {
            let (ea, erg, evy) =
                display_byte_to_dkl_scalar(src[i * 3], src[i * 3 + 1], src[i * 3 + 2], display);
            assert!((a[i] - ea).abs() < 1e-3, "A mismatch at i={i}");
            assert!((rg[i] - erg).abs() < 1e-3, "RG mismatch at i={i}");
            assert!((vy[i] - evy).abs() < 1e-3, "VY mismatch at i={i}");
        }
    }

    /// BT.709 + non-Srgb EOTF (e.g. Gamma(2.2) for Adobe RGB) must
    /// take the dispatch path because the fast path is sRGB-only.
    /// Verifies dispatch correctness for the non-LUT EOTF.
    #[test]
    fn dispatch_path_matches_scalar_for_gamma_bt709() {
        use cvvdp_gpu::kernels::color::display_byte_to_dkl_scalar;
        let display = DisplayModel {
            y_peak: 200.0,
            y_black: 0.2,
            y_refl: 0.397_887_36,
            eotf: Eotf::Gamma(2.2),
            primaries: Primaries::Bt709,
            e_ambient_lux: 250.0,
            k_refl: 0.005,
        };
        let w = 4;
        let h = 4;
        let mut src = vec![0u8; w * h * 3];
        for i in 0..w * h {
            src[i * 3] = (i * 17) as u8;
            src[i * 3 + 1] = (i * 23) as u8;
            src[i * 3 + 2] = (i * 29) as u8;
        }
        let mut a = vec![];
        let mut rg = vec![];
        let mut vy = vec![];
        srgb_to_dkl_planar(&src, w, h, display, &mut a, &mut rg, &mut vy);
        for i in 0..w * h {
            let (ea, erg, evy) =
                display_byte_to_dkl_scalar(src[i * 3], src[i * 3 + 1], src[i * 3 + 2], display);
            assert!((a[i] - ea).abs() < 1e-5, "A mismatch at i={i}");
            assert!((rg[i] - erg).abs() < 1e-5, "RG mismatch at i={i}");
            assert!((vy[i] - evy).abs() < 1e-5, "VY mismatch at i={i}");
        }
    }

    /// Linear planes with BT.2020 primaries (e.g. HDR EXR input).
    /// `linear_planes_to_dkl_planar` must route through the
    /// non-Bt709 branch and apply the BT.2020 matrix.
    #[test]
    fn linear_planes_dispatch_for_bt2020() {
        let display = DisplayModel {
            y_peak: 1500.0,
            y_black: 0.0015,
            y_refl: 10.0_f32 / core::f32::consts::PI * 0.005,
            eotf: Eotf::Linear,
            primaries: Primaries::Bt2020,
            e_ambient_lux: 10.0,
            k_refl: 0.005,
        };
        let w = 4;
        let h = 4;
        let mut r = vec![0.0f32; w * h];
        let mut g = vec![0.0f32; w * h];
        let mut b = vec![0.0f32; w * h];
        for i in 0..w * h {
            r[i] = ((i as f32) * 0.03).min(0.95);
            g[i] = ((i as f32) * 0.05).min(0.95);
            b[i] = ((i as f32) * 0.07).min(0.95);
        }
        let mut a = vec![];
        let mut rg = vec![];
        let mut vy = vec![];
        linear_planes_to_dkl_planar(&r, &g, &b, w, h, w, display, &mut a, &mut rg, &mut vy);
        // BT.2020 matrix produces NOTICEABLY different chroma values
        // than BT.709 for the same primary-saturated input.
        use cvvdp_gpu::kernels::color::display_linear_rgb_to_dkl_scalar;
        for i in 0..w * h {
            let (ea, erg, evy) = display_linear_rgb_to_dkl_scalar(r[i], g[i], b[i], display);
            assert!((a[i] - ea).abs() < 1e-4, "A mismatch at i={i}");
            assert!((rg[i] - erg).abs() < 1e-4, "RG mismatch at i={i}");
            assert!((vy[i] - evy).abs() < 1e-4, "VY mismatch at i={i}");
        }
    }

    /// BT.709 STANDARD_4K linear-planes path: bit-identical to the
    /// historical fast-path numerics. Pinned via to_bits() so any
    /// future refactor of the BT.709 branch must preserve f32.
    #[test]
    fn linear_planes_fast_path_bit_identical_for_standard_4k() {
        let display = DisplayModel::STANDARD_4K;
        let w = 6;
        let h = 4;
        let mut r = vec![0.0f32; w * h];
        let mut g = vec![0.0f32; w * h];
        let mut b = vec![0.0f32; w * h];
        for i in 0..w * h {
            r[i] = (i as f32) * 0.013;
            g[i] = (i as f32) * 0.017;
            b[i] = (i as f32) * 0.019;
        }
        let mut a = vec![];
        let mut rg = vec![];
        let mut vy = vec![];
        linear_planes_to_dkl_planar(&r, &g, &b, w, h, w, display, &mut a, &mut rg, &mut vy);
        // Recompute via the historical inline arithmetic.
        use cvvdp_gpu::params::SRGB_LINEAR_TO_DKL as M;
        let s = display.y_peak - display.y_black;
        let bias = display.y_black + display.y_refl;
        for i in 0..w * h {
            let lr = s * r[i] + bias;
            let lg = s * g[i] + bias;
            let lb = s * b[i] + bias;
            let ea = M[0][0] * lr + M[0][1] * lg + M[0][2] * lb;
            let erg = M[1][0] * lr + M[1][1] * lg + M[1][2] * lb;
            let evy = M[2][0] * lr + M[2][1] * lg + M[2][2] * lb;
            assert_eq!(a[i].to_bits(), ea.to_bits(), "A at {i}");
            assert_eq!(rg[i].to_bits(), erg.to_bits(), "RG at {i}");
            assert_eq!(vy[i].to_bits(), evy.to_bits(), "VY at {i}");
        }
    }
}
