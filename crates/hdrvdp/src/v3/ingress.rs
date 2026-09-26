//! Input conditioning for HDR-VDP-3 — the display models and EOTF decoders
//! that turn code values into absolute display-native values.
//!
//! Two layers exist upstream and both live here:
//!
//! 1. The **`color_encoding` models** inside `hdrvdp3` itself:
//!    `luma-display` (`L = 99·V^2.2 + 1`), `sRGB-display`
//!    (`99·srgb_lin + 1`), and the pass-through absolute encodings.
//! 2. The **caller-side EOTF decoders** the JPEG-AI quality-assessment
//!    framework applies before handing absolute RGB to the metric —
//!    `pq2lin` (BT.2100 PQ → 0–10000 cd/m²) and `TF_HLG.decode` — plus the
//!    ambient-reflection addition `L_refl = ρ·E_ambient/π` it applies for
//!    its `HDR_VDP_3` column.
//!
//! After decoding, absolute-unit inputs get the standard→native transform
//! (`hdrvdp_colorspace_transform` + `fix_out_of_gamut`) for the encodings
//! that carry one.

use crate::interp::cumtrapz;

/// `display_model`: `peak·V^gamma + black_level` — the `luma-display`
/// encoding's `99·V^2.2 + 1`.
#[must_use]
pub fn display_model(v: &[f64], gamma: f64, peak: f64, black_level: f64) -> Vec<f64> {
    v.iter()
        .map(|&x| peak * x.powf(gamma) + black_level)
        .collect()
}

/// `display_model_srgb`: sRGB EOTF then `99·linear + 1` per channel.
#[must_use]
pub fn display_model_srgb(v: &[f64]) -> Vec<f64> {
    const A: f64 = 0.055;
    const THR: f64 = 0.04045;
    v.iter()
        .map(|&x| {
            let lin = if x <= THR {
                x / 12.92
            } else {
                ((x + A) / (1.0 + A)).powf(2.4)
            };
            99.0 * lin + 1.0
        })
        .collect()
}

/// `pq2lin` — BT.2100 PQ code values in `[0,1]` → absolute luminance in
/// cd/m² (0–10000 range). Used by the AIC harness for its `EOTF=pq` runs.
#[must_use]
pub fn pq_to_linear(v: &[f64]) -> Vec<f64> {
    const L_MAX: f64 = 10000.0;
    const N: f64 = 0.1593017578125;
    const M: f64 = 78.84375;
    const C1: f64 = 0.8359375;
    const C2: f64 = 18.8515625;
    const C3: f64 = 18.6875;
    v.iter()
        .map(|&x| {
            let t = x.max(0.0).powf(1.0 / M);
            L_MAX * ((t - C1).max(0.0) / (C2 - C3 * t)).powf(1.0 / N)
        })
        .collect()
}

/// `TF_HLG.decode_float` — BT.2100 HLG *code values* `0..2^bits−1` →
/// scene-referred linear signal `0..2^bits−1` (the harness's own
/// `decode` additionally rounds to integers; use this and add `0.5`-round
/// yourself if bit-exact harness parity is needed).
///
/// Note: as upstream writes it, the decoded range is **code values**, not
/// cd/m² — the harness feeds them to the metric unscaled. Keep that
/// contract in mind: this helper reproduces the reference behaviour, which
/// is probably only meaningful for the harness's own conventions.
#[must_use]
pub fn hlg_decode_float(v: &[f64], bits: u32) -> Vec<f64> {
    const A: f64 = 0.17883277;
    const B: f64 = 0.28466892;
    const C: f64 = 0.55991073;
    const CUT: f64 = 1.0 / 12.0;
    let max_code = f64::from((1u32 << bits) - 1);
    // enc_cut = encode(cut·max_code), computed the same way.
    let enc_cut = A * (12.0 * CUT - B).ln() + C;
    v.iter()
        .map(|&x| {
            let l = if x > enc_cut {
                (((x - C) / A).exp() + B) / 12.0
            } else {
                (x / 3.0f64.sqrt()).powi(2)
            };
            l.clamp(0.0, 1.0) * max_code
        })
        .collect()
}

/// The AIC harness's ambient-light reflection term:
/// `L_refl = reflectivity·E_ambient_lux/π`, then `min(L + L_refl, L_peak +
/// L_refl)` per sample. `hdrvdp_pix_per_deg`-adjacent: pass the values the
/// harness publishes (`reflectivity 0.005`, `E_ambient 100 lux`,
/// `L_peak 10000`).
#[must_use]
pub fn add_ambient_reflection(
    v: &[f64],
    ambient_lux: f64,
    reflectivity: f64,
    peak: f64,
) -> Vec<f64> {
    let l_refl = reflectivity * ambient_lux / core::f64::consts::PI;
    v.iter().map(|&x| (x + l_refl).min(peak + l_refl)).collect()
}

/// `fix_out_of_gamut` — clamp native-channel values below `1e-6` up to it
/// and report the fraction of *pixels* (not samples) that were clamped.
#[must_use]
pub(crate) fn fix_out_of_gamut(img: &mut [f64], channels: usize) -> f64 {
    const MIN_V: f64 = 1e-6;
    let mut clamped_pixels = 0usize;
    for px in img.chunks_exact_mut(channels) {
        if px.iter().any(|&v| v < MIN_V) {
            clamped_pixels += 1;
            for v in px.iter_mut() {
                if *v < MIN_V {
                    *v = MIN_V;
                }
            }
        }
    }
    clamped_pixels as f64 / (img.len() / channels).max(1) as f64
}

/// `hdrvdp_colorspace_transform` — per-pixel `out = M·v` on interleaved
/// `channels` data (must be 3).
pub(crate) fn colorspace_transform(img: &mut [f64], m: &[[f64; 3]; 3]) {
    for px in img.as_chunks_mut::<3>().0 {
        let (a, b, c) = (px[0], px[1], px[2]);
        px[0] = m[0][0] * a + m[0][1] * b + m[0][2] * c;
        px[1] = m[1][0] * a + m[1][1] * b + m[1][2] * c;
        px[2] = m[2][0] * a + m[2][1] * b + m[2][2] * c;
    }
}

/// The `check_if_values_plausible` warning, returned as a bool: `true`
/// when the maximum of the given channel is ≤ 1 (upstream reads the *green*
/// channel for RGB encodings, the whole image for `luminance`).
#[must_use]
pub(crate) fn looks_relative(img: &[f64], channels: usize, channel: usize) -> bool {
    img.chunks_exact(channels)
        .map(|px| px[channel])
        .fold(0.0f64, f64::max)
        <= 1.0
}

/// `hdrvdp_local_adapt`'s active model (`if 0:` disables the fancy one):
/// `exp(fast_gauss(log(max(L, 1e-6)), 10^-0.781367·ppd))`.
pub(crate) fn local_adapt(l_otf: &[f64], width: usize, height: usize, ppd: f64) -> Vec<f64> {
    let sigma = 10f64.powf(-0.781367) * ppd;
    let log_l: Vec<f64> = l_otf.iter().map(|&v| v.max(1e-6).ln()).collect();
    super::gauss::fast_gauss(
        &log_l,
        width,
        height,
        sigma,
        true,
        super::fft64::Pad::Replicate,
    )
    .iter()
    .map(|&v| v.exp())
    .collect()
}

/// `pupil_d_unified` — Watson & Yellott unified pupil model.
pub(crate) fn pupil_d_unified(l: f64, area_deg2: f64, age: f64) -> f64 {
    const Y0: f64 = 28.58;
    let y = age.clamp(20.0, 83.0);
    let d_sd = pupil_d_stanley_davies(l, area_deg2);
    d_sd + (y - Y0) * (0.02132 - 0.009562 * d_sd)
}

fn pupil_d_stanley_davies(l: f64, area_deg2: f64) -> f64 {
    let la = l * area_deg2;
    7.75 - 5.75 * ((la / 846.0).powf(0.41) / ((la / 846.0).powf(0.41) + 2.0))
}

/// `hdrvdp_joint_rod_cone_sens` — combined receptor sensitivity vs adapting
/// luminance, driven by `csf_sa`.
pub(crate) fn joint_rod_cone_sens(la: f64, csf_sa: &[f64; 4]) -> f64 {
    let (s0, drop, trans, low) = (csf_sa[0], csf_sa[1], csf_sa[2], csf_sa[3]);
    s0 * ((drop / la).powf(trans) + 1.0).powf(-low)
}

/// `hdrvdp_rod_sens` — rod sensitivity vs luminance, driven by `csf_sr`.
pub(crate) fn rod_sens(la: f64, csf_sr: &[f64; 6]) -> f64 {
    let (peak_l, low_s, low_e, high_s, high_e, sens) = (
        csf_sr[0], csf_sr[1], csf_sr[2], csf_sr[3], csf_sr[4], csf_sr[5],
    );
    let s = if la > peak_l {
        (-(la / peak_l).log10().abs().powf(high_e) / high_s).exp()
    } else {
        (-(la / peak_l).log10().abs().powf(low_e) / low_s).exp()
    };
    s * 10f64.powf(sens)
}

/// `create_pn_jnd` + `build_jndspace_from_S` — the intensity→JND lookup
/// tables, `base_sensitivity_correction` applied. Returns
/// `(log10_lum, jnd_cone, jnd_rod)` where the `jnd` arrays are aligned to
/// `log10_lum` (upstream's `insert(jnd, 0, 0)` after the `n−1` cumtrapz).
pub(crate) fn pn_jnd_luts(
    joint: &[f64],
    rod: &[f64],
    c_l: &[f64],
    base_sensitivity_correction: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // s_C = 0.5·interp1d(c_l, max(s_A−s_R, 1e-3))(min(2·c_l, c_l[-1])).
    let cone_s: Vec<f64> = (0..c_l.len())
        .map(|i| (joint[i] - rod[i]).max(1e-3))
        .collect();
    // Linear interp of `cone_s` at 2·c_l (clamped at the last abscissa).
    let s_c: Vec<f64> = c_l
        .iter()
        .map(|&cl| {
            let q = (2.0 * cl).min(c_l[c_l.len() - 1]);
            crate::interp::interp1_linear(c_l, &cone_s, q) * 0.5
        })
        .collect();

    let scale = 10f64.powf(base_sensitivity_correction);
    let log_l: Vec<f64> = c_l.iter().map(|v| v.log10()).collect();
    // `build_jndspace_from_S`: thr = L/S, dL = (1/thr)·L·ln10 = S·ln10 —
    // the JND increment accumulates with *sensitivity*, not threshold.
    let dl_c: Vec<f64> = s_c.iter().map(|&s| s * std::f64::consts::LN_10).collect();
    let dl_r: Vec<f64> = rod.iter().map(|&s| s * std::f64::consts::LN_10).collect();
    // `cumtrapz` (our left-aligned, length-n version) already equals
    // scipy's `cumtrapz` + `insert(0, 0)`.
    let mut jnd_cone = cumtrapz(&log_l, &dl_c);
    let mut jnd_rod = cumtrapz(&log_l, &dl_r);
    for v in jnd_cone.iter_mut().chain(jnd_rod.iter_mut()) {
        *v *= scale;
    }
    (log_l, jnd_cone, jnd_rod)
}
