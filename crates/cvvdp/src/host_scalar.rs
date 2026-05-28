//! Composed host-scalar still-image cvvdp pipeline.
//!
//! Phase 8c.1-B moved this out of `cvvdp-gpu::host_scalar` so the CPU
//! crate owns the canonical "no GPU" reference algorithm. cvvdp-gpu
//! continues to re-export the same paths.
//!
//! Chains all the per-stage host scalars into a single
//! `predict_jod_still_3ch` entry point. Each stage's parity vs
//! pycvvdp v0.5.4 is verified by dedicated tests.

use alloc::vec;
use alloc::vec::Vec;

use crate::kernels::color::display_byte_to_dkl_scalar;
use crate::kernels::csf::{CSF_BASEBAND_RHO, CsfChannel, sensitivity_corrected_scalar};
use crate::kernels::masking::{CH_GAIN, mult_mutual_band};
use crate::kernels::pool::{BETA_SPATIAL, do_pooling_and_jod_still_3ch, lp_norm_mean};
use crate::kernels::pyramid::{band_frequencies, weber_contrast_pyr_dec_scalar};
use crate::params::DisplayModel;

/// Predict cvvdp JOD for a still-image (reference, distorted) pair.
///
/// # Examples
///
/// ```
/// use cvvdp::host_scalar::predict_jod_still_3ch;
/// use cvvdp::params::{DisplayGeometry, DisplayModel};
///
/// let (w, h) = (64usize, 64usize);
/// let bytes = vec![128u8; w * h * 3];
/// let ppd = DisplayGeometry::STANDARD_4K.pixels_per_degree();
/// let jod = predict_jod_still_3ch(
///     &bytes,
///     &bytes,
///     w,
///     h,
///     DisplayModel::STANDARD_4K,
///     ppd,
/// );
/// assert!((jod - 10.0).abs() < 1e-3, "expected JOD ≈ 10, got {jod}");
/// ```
///
/// # Panics
///
/// Panics if `ref_srgb.len() != width * height * 3` or
/// `dist_srgb.len() != width * height * 3`.
#[must_use]
pub fn predict_jod_still_3ch(
    ref_srgb: &[u8],
    dist_srgb: &[u8],
    width: usize,
    height: usize,
    display: DisplayModel,
    ppd: f32,
) -> f32 {
    predict_jod_still_3ch_capped(ref_srgb, dist_srgb, width, height, display, ppd, None)
}

/// Variant of [`predict_jod_still_3ch`] that truncates the pyramid at
/// `cap_levels` bands.
#[must_use]
pub fn predict_jod_still_3ch_capped(
    ref_srgb: &[u8],
    dist_srgb: &[u8],
    width: usize,
    height: usize,
    display: DisplayModel,
    ppd: f32,
    cap_levels: Option<usize>,
) -> f32 {
    assert_eq!(ref_srgb.len(), width * height * 3);
    assert_eq!(dist_srgb.len(), width * height * 3);

    let n = width * height;
    let mut ref_planes: [Vec<f32>; 3] = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
    let mut dis_planes: [Vec<f32>; 3] = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
    for i in 0..n {
        let (a, rg, vy) = display_byte_to_dkl_scalar(
            ref_srgb[i * 3],
            ref_srgb[i * 3 + 1],
            ref_srgb[i * 3 + 2],
            display,
        );
        ref_planes[0][i] = a;
        ref_planes[1][i] = rg;
        ref_planes[2][i] = vy;
        let (a, rg, vy) = display_byte_to_dkl_scalar(
            dist_srgb[i * 3],
            dist_srgb[i * 3 + 1],
            dist_srgb[i * 3 + 2],
            display,
        );
        dis_planes[0][i] = a;
        dis_planes[1][i] = rg;
        dis_planes[2][i] = vy;
    }

    let natural_n_levels = band_frequencies(ppd, width, height).len();
    let n_levels_query = match cap_levels {
        Some(cap) if cap >= 1 => cap.min(natural_n_levels),
        _ => natural_n_levels,
    };
    let ref_weber = [
        weber_contrast_pyr_dec_scalar(
            &ref_planes[0],
            &ref_planes[0],
            width,
            height,
            n_levels_query,
        ),
        weber_contrast_pyr_dec_scalar(
            &ref_planes[1],
            &ref_planes[0],
            width,
            height,
            n_levels_query,
        ),
        weber_contrast_pyr_dec_scalar(
            &ref_planes[2],
            &ref_planes[0],
            width,
            height,
            n_levels_query,
        ),
    ];
    let dis_weber = [
        weber_contrast_pyr_dec_scalar(
            &dis_planes[0],
            &dis_planes[0],
            width,
            height,
            n_levels_query,
        ),
        weber_contrast_pyr_dec_scalar(
            &dis_planes[1],
            &dis_planes[0],
            width,
            height,
            n_levels_query,
        ),
        weber_contrast_pyr_dec_scalar(
            &dis_planes[2],
            &dis_planes[0],
            width,
            height,
            n_levels_query,
        ),
    ];
    let n_levels = ref_weber[0].bands.len();

    let freqs = band_frequencies(ppd, width, height);
    let channels = [CsfChannel::A, CsfChannel::Rg, CsfChannel::Vy];

    let mut q_per_ch: Vec<[f32; 3]> = Vec::with_capacity(n_levels);
    for k in 0..n_levels {
        let is_first = k == 0;
        let is_baseband = k == n_levels - 1;
        let band_mul: f32 = if is_first || is_baseband { 1.0 } else { 2.0 };

        let bw = ref_weber[0].bands[k].w;
        let bh = ref_weber[0].bands[k].h;
        let n_px = bw * bh;
        let rho = if is_baseband {
            CSF_BASEBAND_RHO
        } else {
            freqs[k]
        };
        let log_l_bkg_band = &ref_weber[0].log_l_bkg[k];
        debug_assert_eq!(log_l_bkg_band.len(), n_px);

        let mut t_p_per_ch: [Vec<f32>; 3] = [vec![0.0; n_px], vec![0.0; n_px], vec![0.0; n_px]];
        let mut r_p_per_ch: [Vec<f32>; 3] = [vec![0.0; n_px], vec![0.0; n_px], vec![0.0; n_px]];
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

        let d_per_ch = if is_baseband {
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

        let mut q_band = [0.0_f32; 3];
        for c in 0..3 {
            q_band[c] = lp_norm_mean(&d_per_ch[c], BETA_SPATIAL);
        }
        q_per_ch.push(q_band);
    }

    do_pooling_and_jod_still_3ch(&q_per_ch)
}
