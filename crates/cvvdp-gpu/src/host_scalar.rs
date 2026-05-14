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
use crate::kernels::pyramid::{band_frequencies, weber_contrast_pyr_dec_scalar};
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

    // Build Weber-contrast pyramids per side per channel. L_bkg
    // always comes from the side's own achromatic plane (cvvdp's
    // weber_g1 contract): for ref-side bands use ref_planes[0]; for
    // dist-side bands use dis_planes[0].
    let n_levels_query = 0;
    let ref_weber = [
        weber_contrast_pyr_dec_scalar(&ref_planes[0], &ref_planes[0], width, height, n_levels_query),
        weber_contrast_pyr_dec_scalar(&ref_planes[1], &ref_planes[0], width, height, n_levels_query),
        weber_contrast_pyr_dec_scalar(&ref_planes[2], &ref_planes[0], width, height, n_levels_query),
    ];
    let dis_weber = [
        weber_contrast_pyr_dec_scalar(&dis_planes[0], &dis_planes[0], width, height, n_levels_query),
        weber_contrast_pyr_dec_scalar(&dis_planes[1], &dis_planes[0], width, height, n_levels_query),
        weber_contrast_pyr_dec_scalar(&dis_planes[2], &dis_planes[0], width, height, n_levels_query),
    ];
    let n_levels = ref_weber[0].bands.len();

    // For the CSF lookup cvvdp uses the reference's achromatic
    // log_L_bkg per band — already produced by `weber_contrast_pyr`'s
    // pass on channel 0.
    let freqs = band_frequencies(ppd, width, height);
    let channels = [CsfChannel::A, CsfChannel::Rg, CsfChannel::Vy];

    // For each band: apply CSF weighting → masking → spatial pool.
    // Baseband-bypass and rho-clamp behaviour mirror cvvdp's
    // weber_contrast_pyr path which we have NOT yet ported (vanilla
    // Laplacian + linear DKL bands here vs. cvvdp's Weber-contrast
    // Laplacian + log10(gauss) for L_bkg). Documented in PORT_STATUS.
    let mut q_per_ch: Vec<[f32; 3]> = Vec::with_capacity(n_levels);
    for k in 0..n_levels {
        let is_first = k == 0;
        let is_baseband = k == n_levels - 1;
        // cvvdp's `lpyr.get_band` applies a band-readout gain of
        // 2.0 to all non-edge bands (first = 0, last = n-1 keep
        // 1.0). Mirror that here at the band-consumption site so
        // the underlying Weber pyramid data stays canonical.
        let band_mul: f32 = if is_first || is_baseband { 1.0 } else { 2.0 };

        let bw = ref_weber[0].bands[k].w;
        let bh = ref_weber[0].bands[k].h;
        let n_px = bw * bh;
        let rho = freqs[k];
        let log_l_bkg_band = &ref_weber[0].log_l_bkg[k];
        debug_assert_eq!(log_l_bkg_band.len(), n_px);

        // T_p, R_p: band_mul * Weber-contrast * S(rho, log_L_bkg[i], cc) * CH_GAIN.
        let mut t_p_per_ch: [Vec<f32>; 3] =
            [vec![0.0; n_px], vec![0.0; n_px], vec![0.0; n_px]];
        let mut r_p_per_ch: [Vec<f32>; 3] =
            [vec![0.0; n_px], vec![0.0; n_px], vec![0.0; n_px]];
        for i in 0..n_px {
            let log_l = log_l_bkg_band[i];
            let s_a = sensitivity_corrected_scalar(rho, log_l, channels[0]);
            let s_rg = sensitivity_corrected_scalar(rho, log_l, channels[1]);
            let s_vy = sensitivity_corrected_scalar(rho, log_l, channels[2]);
            t_p_per_ch[0][i] = band_mul * dis_weber[0].bands[k].data[i] * s_a * CH_GAIN[0];
            t_p_per_ch[1][i] = band_mul * dis_weber[1].bands[k].data[i] * s_rg * CH_GAIN[1];
            t_p_per_ch[2][i] = band_mul * dis_weber[2].bands[k].data[i] * s_vy * CH_GAIN[2];
            r_p_per_ch[0][i] = band_mul * ref_weber[0].bands[k].data[i] * s_a * CH_GAIN[0];
            r_p_per_ch[1][i] = band_mul * ref_weber[1].bands[k].data[i] * s_rg * CH_GAIN[1];
            r_p_per_ch[2][i] = band_mul * ref_weber[2].bands[k].data[i] * s_vy * CH_GAIN[2];
        }

        // Baseband bypass: cvvdp uses D = |T_f - R_f| * S for the
        // coarsest band (no masking model, no clamp). Other bands
        // run the full mult-mutual + xchannel + PU-blur pipeline.
        // With Weber-contrast band magnitudes the |T - R| * S form
        // produces values in the same range as cvvdp's pre-pool D.
        let d_per_ch = if is_baseband {
            // T_f - R_f here means the *unweighted* Weber bands times
            // the CSF S (no CH_GAIN for baseband — cvvdp's apply_masking_model
            // is what applies CH_GAIN; baseband bypasses it entirely).
            let mut out: [Vec<f32>; 3] = [vec![0.0; n_px], vec![0.0; n_px], vec![0.0; n_px]];
            for i in 0..n_px {
                let log_l = log_l_bkg_band[i];
                let s_a = sensitivity_corrected_scalar(rho, log_l, channels[0]);
                let s_rg = sensitivity_corrected_scalar(rho, log_l, channels[1]);
                let s_vy = sensitivity_corrected_scalar(rho, log_l, channels[2]);
                let diff_a = dis_weber[0].bands[k].data[i] - ref_weber[0].bands[k].data[i];
                let diff_rg = dis_weber[1].bands[k].data[i] - ref_weber[1].bands[k].data[i];
                let diff_vy = dis_weber[2].bands[k].data[i] - ref_weber[2].bands[k].data[i];
                out[0][i] = diff_a.abs() * s_a;
                out[1][i] = diff_rg.abs() * s_rg;
                out[2][i] = diff_vy.abs() * s_vy;
            }
            out
        } else {
            mult_mutual_band(&t_p_per_ch, &r_p_per_ch, bw, bh)
        };

        // Spatial pool per channel (RMS = beta_spatial).
        let mut q_band = [0.0_f32; 3];
        for c in 0..3 {
            q_band[c] = lp_norm_mean(&d_per_ch[c], BETA_SPATIAL);
        }
        q_per_ch.push(q_band);
    }

    do_pooling_and_jod_still_3ch(&q_per_ch)
}
