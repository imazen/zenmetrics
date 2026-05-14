//! Composed host-scalar still-image cvvdp pipeline.
//!
//! Chains all the per-stage host scalars into a single
//! `predict_jod_still_3ch` entry point. Each stage's parity vs
//! pycvvdp v0.5.4 is verified by a dedicated test; the composed
//! result is exercised by `tests/shadow_jod.rs` against the v1 R2
//! manifest.

use crate::kernels::color::srgb_byte_to_dkl_scalar;
use crate::kernels::csf::{CsfChannel, sensitivity_corrected_scalar};
use crate::kernels::masking::{CH_GAIN, mult_mutual_band};
use crate::kernels::pool::{
    BETA_SPATIAL, do_pooling_and_jod_still_3ch, lp_norm_mean,
};
use crate::kernels::pyramid::{
    band_frequencies, gausspyr_reduce_scalar, laplacian_pyramid_dec_scalar,
};
use crate::params::DisplayModel;

/// Predict cvvdp JOD for a still-image (reference, distorted) pair.
///
/// Inputs:
/// - `ref_srgb`, `dist_srgb` — packed RGBRGB… bytes of the two
///   images, each of length `width * height * 3`.
/// - `width`, `height` — image dimensions in pixels.
/// - `display` — photometric display model (luminance + EOTF).
/// - `ppd` — pixels-per-degree (from `DisplayGeometry::pixels_per_degree`).
///
/// Returns predicted JOD in `[0, 10]` (10 = imperceptible).
///
/// Uses per-pixel L_bkg from the reference's achromatic Gaussian
/// pyramid, matching cvvdp v0.5.4's `process_block_of_frames` —
/// each band's CSF lookup queries `gauss_a[bb][i]` where `bb` is
/// the band index and `i` is the per-pixel index into the
/// band-sized buffer.
pub fn predict_jod_still_3ch(
    ref_srgb: &[u8],
    dist_srgb: &[u8],
    width: usize,
    height: usize,
    display: DisplayModel,
    ppd: f32,
) -> f32 {
    assert_eq!(ref_srgb.len(), width * height * 3);
    assert_eq!(dist_srgb.len(), width * height * 3);

    let n = width * height;
    let mut ref_planes: [Vec<f32>; 3] = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
    let mut dis_planes: [Vec<f32>; 3] = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
    for i in 0..n {
        let (a, rg, vy) = srgb_byte_to_dkl_scalar(
            ref_srgb[i * 3],
            ref_srgb[i * 3 + 1],
            ref_srgb[i * 3 + 2],
            display.y_peak,
            display.y_black,
            display.y_refl,
        );
        ref_planes[0][i] = a;
        ref_planes[1][i] = rg;
        ref_planes[2][i] = vy;
        let (a, rg, vy) = srgb_byte_to_dkl_scalar(
            dist_srgb[i * 3],
            dist_srgb[i * 3 + 1],
            dist_srgb[i * 3 + 2],
            display.y_peak,
            display.y_black,
            display.y_refl,
        );
        dis_planes[0][i] = a;
        dis_planes[1][i] = rg;
        dis_planes[2][i] = vy;
    }

    // Build Laplacian pyramid per channel for both sides.
    let n_levels_query = 0; // default: floor(log2(min(w, h))) - 1 + 1
    let ref_bands: [Vec<crate::kernels::pyramid::Band>; 3] = [
        laplacian_pyramid_dec_scalar(&ref_planes[0], width, height, n_levels_query),
        laplacian_pyramid_dec_scalar(&ref_planes[1], width, height, n_levels_query),
        laplacian_pyramid_dec_scalar(&ref_planes[2], width, height, n_levels_query),
    ];
    let dis_bands: [Vec<crate::kernels::pyramid::Band>; 3] = [
        laplacian_pyramid_dec_scalar(&dis_planes[0], width, height, n_levels_query),
        laplacian_pyramid_dec_scalar(&dis_planes[1], width, height, n_levels_query),
        laplacian_pyramid_dec_scalar(&dis_planes[2], width, height, n_levels_query),
    ];
    let n_levels = ref_bands[0].len();

    // Build the Gaussian pyramid of the reference's achromatic plane
    // — this is the per-pixel L_bkg cvvdp uses for the CSF lookup at
    // each band.
    let freqs = band_frequencies(ppd, width, height);
    let channels = [CsfChannel::A, CsfChannel::Rg, CsfChannel::Vy];
    let mut gauss_ref_a: Vec<Vec<f32>> = Vec::with_capacity(n_levels);
    gauss_ref_a.push(ref_planes[0].clone());
    let mut prev_w = width;
    let mut prev_h = height;
    for _ in 1..n_levels {
        let mut next = Vec::new();
        let (nw, nh) = gausspyr_reduce_scalar(
            gauss_ref_a.last().unwrap(),
            prev_w,
            prev_h,
            &mut next,
        );
        gauss_ref_a.push(next);
        prev_w = nw;
        prev_h = nh;
    }

    // For each band: apply CSF weighting → masking → spatial pool.
    // Baseband-bypass and rho-clamp behaviour mirror cvvdp's
    // weber_contrast_pyr path which we have NOT yet ported (vanilla
    // Laplacian + linear DKL bands here vs. cvvdp's Weber-contrast
    // Laplacian + log10(gauss) for L_bkg). Documented in PORT_STATUS.
    let mut q_per_ch: Vec<[f32; 3]> = Vec::with_capacity(n_levels);
    for k in 0..n_levels {
        let bw = ref_bands[0][k].w;
        let bh = ref_bands[0][k].h;
        let n_px = bw * bh;
        let rho = freqs[k];
        let l_bkg_band = &gauss_ref_a[k];
        debug_assert_eq!(l_bkg_band.len(), n_px);

        // Build T_p, R_p per channel: T * S(rho, log10(L_bkg[i]), cc) * CH_GAIN.
        // cvvdp's `csf.sensitivity` expects the L_bkg argument in
        // log10 space (the LUT's L_bkg axis is log10); the cvvdp
        // pipeline log10s its gauss-pyramid output before the call.
        let mut t_p_per_ch: [Vec<f32>; 3] =
            [vec![0.0; n_px], vec![0.0; n_px], vec![0.0; n_px]];
        let mut r_p_per_ch: [Vec<f32>; 3] =
            [vec![0.0; n_px], vec![0.0; n_px], vec![0.0; n_px]];
        for i in 0..n_px {
            let log_l = l_bkg_band[i].max(1e-6).log10();
            let s_a = sensitivity_corrected_scalar(rho, log_l, channels[0]);
            let s_rg = sensitivity_corrected_scalar(rho, log_l, channels[1]);
            let s_vy = sensitivity_corrected_scalar(rho, log_l, channels[2]);
            t_p_per_ch[0][i] = dis_bands[0][k].data[i] * s_a * CH_GAIN[0];
            t_p_per_ch[1][i] = dis_bands[1][k].data[i] * s_rg * CH_GAIN[1];
            t_p_per_ch[2][i] = dis_bands[2][k].data[i] * s_vy * CH_GAIN[2];
            r_p_per_ch[0][i] = ref_bands[0][k].data[i] * s_a * CH_GAIN[0];
            r_p_per_ch[1][i] = ref_bands[1][k].data[i] * s_rg * CH_GAIN[1];
            r_p_per_ch[2][i] = ref_bands[2][k].data[i] * s_vy * CH_GAIN[2];
        }

        let d_per_ch = mult_mutual_band(&t_p_per_ch, &r_p_per_ch, bw, bh);

        // Spatial pool per channel (RMS = beta_spatial).
        let mut q_band = [0.0_f32; 3];
        for c in 0..3 {
            q_band[c] = lp_norm_mean(&d_per_ch[c], BETA_SPATIAL);
        }
        q_per_ch.push(q_band);
    }

    do_pooling_and_jod_still_3ch(&q_per_ch)
}
