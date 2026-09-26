//! `hdrvdp_visual_pathway` — display-native luminance → JND-space
//! photoreceptor response → sp0 pyramid.
//!
//! Order of operations (the reference's, `hdrvdp_visual_pathway.py`):
//!
//! 1. NaN → 1e-5.
//! 2. Per-channel convolution with the optical MTF on a 2× lattice, padded
//!    per `surround`, clamped to `[1e-5, 1e10]`.
//! 3. Aging optical density (`do_aod`): multiply the *emission table* by
//!    `10^{−PCHIP(OD−OD_y)}` — spectrally, before the matrix multiply.
//! 4. Receptor mixing `M_img_lmsr[l,k] = trapz(LMSR_k·IMG_E_l·683.002, λ)`,
//!    normalised so `Σ_L+M` over all channels is 1, applied per pixel,
//!    clamped to `[1e-8, 1e10]`.
//! 5. Local adaptation `L_adapt = exp(gauss(log(max(L+M,1e-6)), 0.165°σ))`.
//! 6. Senile miosis (`do_slum`): scale `R_LMSR` by `(pupil(age)/pupil(28))²`
//!    over the geometric-mean adapting luminance and the image's angular
//!    area.
//! 7. Photoreceptor non-linearity: L, M and rod channels each mapped to
//!    JND space through the `pn` lookup tables; `P = JND_L + JND_M +
//!    JND_rod`.
//! 8. `sp0` decomposition; the base band's DC is removed *after*
//!    decomposition (`BB − mean(BB)`).

use super::fft64::Pad;
use super::ingress::{joint_rod_cone_sens, local_adapt, pn_jnd_luts, pupil_d_unified, rod_sens};
use super::params::TaskPar;
use super::spectral;
use super::spyr0::Spyr0;
use crate::Result;
use crate::interp::{interp1_linear, trapz};

/// What a single image's trip through the pathway produces.
pub(crate) struct PathwayOut {
    /// The sp0 pyramid of the JND response, baseband DC-removed.
    pub bands: Spyr0,
    /// The local-adaptation map (photopic luminance, full resolution).
    pub l_adapt: Vec<f64>,
    /// The achromatic JND response `P` (pre-decomposition). Not read by the
    /// metric itself — kept for diagnostics/tests.
    #[allow(dead_code)]
    pub p: Vec<f64>,
    /// Whether the input contained NaNs (upstream: warn + replace 1e-5).
    pub had_nan: bool,
}

/// `create_cycdeg_image` — per-coefficient spatial frequency in cpd for a
/// `w×h` lattice sampled at `ppd` pixels/degree.
pub(crate) fn cycdeg_image(width: usize, height: usize, ppd: f64) -> Vec<f64> {
    let mut out = vec![0.0; width * height];
    for y in 0..height {
        let ky = ((0.5 + y as f64 / height as f64) % 1.0 - 0.5) * ppd;
        for x in 0..width {
            let kx = ((0.5 + x as f64 / width as f64) % 1.0 - 0.5) * ppd;
            out[y * width + x] = (kx * kx + ky * ky).sqrt();
        }
    }
    out
}

/// `hdrvdp_mtf` — `Σ a_k·exp(−b_k·ρ)` evaluated on the frequency grid.
pub(crate) fn mtf_filter(rho: &[f64], a: &[f64; 4], b: &[f64; 4]) -> Vec<f64> {
    rho.iter()
        .map(|&r| (0..4).map(|k| a[k] * (-b[k] * r).exp()).sum::<f64>())
        .collect()
}

/// The geomean of one channel's pixels.
pub(crate) fn geomean(v: &[f64]) -> f64 {
    (v.iter().map(|x| x.max(0.0).ln()).sum::<f64>() / v.len() as f64).exp()
}

/// `hdrvdp_visual_pathway(img_native, name, metric_par, …)`.
///
/// `img` is interleaved `width·height·channels`, already in *native*
/// display units (post-EOTF, post-colour-transform, gamut-fixed).
/// `img_e` is the display's emission columns on the 420-point grid.
/// `pad` is the resolved surround padding (per [`Surround`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn visual_pathway(
    img: &[f64],
    width: usize,
    height: usize,
    channels: usize,
    par: &TaskPar,
    viewing: &super::params::ViewingConditions,
    img_e: &[Vec<f64>],
    pad: &dyn Fn(usize) -> Pad,
) -> Result<PathwayOut> {
    let npix = width * height;
    let had_nan = img.iter().any(|v| v.is_nan());
    let mut img = img.to_vec();
    for v in img.iter_mut() {
        if v.is_nan() {
            *v = 1e-5;
        }
    }

    let ppd = viewing.pixels_per_degree;

    // ---- optical MTF (Fourier domain on the 2× lattice) ----
    let rho2 = cycdeg_image(width * 2, height * 2, ppd);
    let mtf = mtf_filter(&rho2, &par.mtf_params_a, &par.mtf_params_b);
    let mut l_o = vec![0.0; npix * channels];
    for k in 0..channels {
        let ch: Vec<f64> = img.iter().skip(k).step_by(channels).copied().collect();
        let filtered =
            super::fft64::conv_fft_pad(&ch, width, height, &mtf, width * 2, height * 2, pad(k));
        for (i, &v) in filtered.iter().enumerate() {
            l_o[i * channels + k] = v.clamp(1e-5, 1e10);
        }
    }

    // ---- aging optical density (spectral, on the emission table) ----
    let mut img_e = img_e.to_vec();
    if par.do_aod {
        let trans = spectral::aging_optical_density_filter(viewing.observer_age);
        for col in img_e.iter_mut() {
            for (v, &t) in col.iter_mut().zip(&trans) {
                *v *= t;
            }
        }
    }

    // ---- receptor mixing matrix ----
    let lamb = spectral::lamb();
    let lmsr = spectral::receptor_sensitivities()?;
    let mut m_img_lmsr = vec![[0.0f64; 4]; channels];
    for (k, sens) in lmsr.iter().enumerate() {
        for (l, e) in img_e.iter().enumerate().take(channels) {
            let prod: Vec<f64> = (0..spectral::LAMB_N)
                .map(|i| sens[i] * e[i] * 683.002)
                .collect();
            m_img_lmsr[l][k] = trapz(&lamb, &prod);
        }
    }
    let norm: f64 = m_img_lmsr.iter().map(|row| row[0] + row[1]).sum();
    for row in m_img_lmsr.iter_mut() {
        for v in row.iter_mut() {
            *v /= norm;
        }
    }
    let mut r_lmsr = vec![0.0f64; npix * 4];
    for px in 0..npix {
        for k in 0..4 {
            let mut acc = 0.0;
            for l in 0..channels {
                acc += l_o[px * channels + l] * m_img_lmsr[l][k];
            }
            r_lmsr[px * 4 + k] = acc.clamp(1e-8, 1e10);
        }
    }

    // ---- local adaptation ----
    let l_adapt = {
        let lum: Vec<f64> = (0..npix)
            .map(|i| r_lmsr[i * 4] + r_lmsr[i * 4 + 1])
            .collect();
        local_adapt(&lum, width, height, ppd)
    };

    // ---- senile miosis ----
    if par.do_slum {
        let l_a = geomean(&l_adapt);
        let area = npix as f64 / (ppd * ppd);
        let d_ref = pupil_d_unified(l_a, area, 28.0);
        let d_age = pupil_d_unified(l_a, area, f64::from(viewing.observer_age));
        let lum_reduction = (d_age * d_age) / (d_ref * d_ref);
        for v in r_lmsr.iter_mut() {
            *v *= lum_reduction;
        }
    }

    // ---- photoreceptor non-linearity (JND space) ----
    let c_l: Vec<f64> = (0..2048)
        .map(|i| 10f64.powf(-5.0 + 10.0 * i as f64 / 2047.0))
        .collect();
    let s_a: Vec<f64> = c_l
        .iter()
        .map(|&l| joint_rod_cone_sens(l, &par.csf_sa))
        .collect();
    let s_r: Vec<f64> = c_l
        .iter()
        .map(|&l| rod_sens(l, &par.csf_sr) * 10f64.powf(par.rod_sensitivity))
        .collect();
    let (y_lut, jnd_cone, jnd_rod) = pn_jnd_luts(&s_a, &s_r, &c_l, par.base_sensitivity_correction);
    let y0 = 10f64.powf(y_lut[0]);
    let y1 = 10f64.powf(y_lut[y_lut.len() - 1]);

    let mut p = vec![0.0; npix];
    for i in 0..npix {
        for (k, lut) in [(0usize, &jnd_cone), (1, &jnd_cone), (3, &jnd_rod)] {
            let v = r_lmsr[i * 4 + k].clamp(y0, y1).log10();
            p[i] += interp1_linear(&y_lut, lut, v);
        }
    }

    // ---- sp0 decomposition + baseband DC removal ----
    let mut bands =
        Spyr0::decompose(&p, width, height, ppd).ok_or(crate::Error::ImageTooSmall {
            size: (width, height),
            min: 13,
        })?;
    let last = bands.band_count() - 1;
    let bb = bands.band(last);
    let mean = bb.iter().sum::<f64>() / bb.len() as f64;
    let centred: Vec<f64> = bb.iter().map(|&v| v - mean).collect();
    bands.set_band(last, &centred);

    Ok(PathwayOut {
        bands,
        l_adapt,
        p,
        had_nan,
    })
}
