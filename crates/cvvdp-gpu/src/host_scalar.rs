//! Composed host-scalar still-image cvvdp pipeline.
//!
//! Chains all the per-stage host scalars into a single
//! `predict_jod_still_3ch` entry point. The intermediate values
//! match cvvdp v0.5.4 at each stage (verified by per-stage parity
//! tests); the composed result diverges from pycvvdp's manifest
//! JOD wherever the port has documented simplifications:
//!
//! - **Global L_bkg approximation**: cvvdp uses per-pixel L_bkg
//!   from the achromatic channel's Gaussian pyramid at level 1.
//!   This module accepts a scalar `l_bkg` and applies it to all
//!   pixels — accurate for uniform images, biased for high-contrast
//!   ones.
//! - **No phase-uncertainty Gaussian blur**: cvvdp applies a σ=3
//!   separable Gaussian to the M_mm tensor for bands > 6×6 px.
//!   The Rust port currently uses the no-blur path everywhere,
//!   which produces a sharper masker and biases the JOD lower.
//!
//! Use this for sanity-checking the math composition; whole-image
//! pycvvdp parity needs the gaps closed.

use crate::kernels::color::srgb_byte_to_dkl_scalar;
use crate::kernels::csf::{CsfChannel, sensitivity_corrected_scalar};
use crate::kernels::masking::{CH_GAIN, mult_mutual_pixel};
use crate::kernels::pool::{
    BETA_SPATIAL, do_pooling_and_jod_still_3ch, lp_norm_mean,
};
use crate::kernels::pyramid::{band_frequencies, laplacian_pyramid_dec_scalar};
use crate::params::DisplayModel;

/// Predict cvvdp JOD for a still-image (reference, distorted) pair.
///
/// Inputs:
/// - `ref_srgb`, `dist_srgb` — packed RGBRGB… bytes of the two
///   images, each of length `width * height * 3`.
/// - `width`, `height` — image dimensions in pixels.
/// - `display` — photometric display model (luminance + EOTF).
/// - `ppd` — pixels-per-degree (from `DisplayGeometry::pixels_per_degree`).
/// - `l_bkg` — global background-luminance approximation (cd/m²).
///
/// Returns predicted JOD in `[0, 10]` (10 = imperceptible).
pub fn predict_jod_still_3ch(
    ref_srgb: &[u8],
    dist_srgb: &[u8],
    width: usize,
    height: usize,
    display: DisplayModel,
    ppd: f32,
    l_bkg: f32,
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

    // Per-band per-channel CSF sensitivity (one scalar per band per
    // channel using the global L_bkg approximation).
    let freqs = band_frequencies(ppd, width, height);
    let channels = [CsfChannel::A, CsfChannel::Rg, CsfChannel::Vy];
    let mut s_per_band: Vec<[f32; 3]> = Vec::with_capacity(n_levels);
    for k in 0..n_levels {
        let rho = freqs[k];
        s_per_band.push([
            sensitivity_corrected_scalar(rho, l_bkg, channels[0]),
            sensitivity_corrected_scalar(rho, l_bkg, channels[1]),
            sensitivity_corrected_scalar(rho, l_bkg, channels[2]),
        ]);
    }

    // For each band: apply CSF weighting → masking → spatial pool.
    // Build Q_per_ch[level][channel] for the final pooling stage.
    let mut q_per_ch: Vec<[f32; 3]> = Vec::with_capacity(n_levels);
    for k in 0..n_levels {
        let bw = ref_bands[0][k].w;
        let bh = ref_bands[0][k].h;
        let n_px = bw * bh;
        let s = s_per_band[k];

        // Per-pixel masking with CSF-weighted contrasts. cvvdp's
        // `mult-mutual` path multiplies by S and CH_GAIN before
        // masking — same form is replicated here.
        let mut d_per_ch: [Vec<f32>; 3] =
            [vec![0.0; n_px], vec![0.0; n_px], vec![0.0; n_px]];
        for i in 0..n_px {
            let t_p = [
                dis_bands[0][k].data[i] * s[0] * CH_GAIN[0],
                dis_bands[1][k].data[i] * s[1] * CH_GAIN[1],
                dis_bands[2][k].data[i] * s[2] * CH_GAIN[2],
            ];
            let r_p = [
                ref_bands[0][k].data[i] * s[0] * CH_GAIN[0],
                ref_bands[1][k].data[i] * s[1] * CH_GAIN[1],
                ref_bands[2][k].data[i] * s[2] * CH_GAIN[2],
            ];
            let d = mult_mutual_pixel(t_p, r_p);
            d_per_ch[0][i] = d[0];
            d_per_ch[1][i] = d[1];
            d_per_ch[2][i] = d[2];
        }

        // Spatial pool per channel (RMS = beta_spatial).
        let mut q_band = [0.0_f32; 3];
        for c in 0..3 {
            q_band[c] = lp_norm_mean(&d_per_ch[c], BETA_SPATIAL);
        }
        q_per_ch.push(q_band);
    }

    do_pooling_and_jod_still_3ch(&q_per_ch)
}
