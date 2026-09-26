//! `hdrvdp3` — the metric driver: encode → display-native → pathway →
//! per-band masking → `P_map`, `Q`, `Q_JOD`.
//!
//! Faithful to `VDP3/hdrvdp3.py` in the `jpeg-ai-qaf` HDR-VDP-3 port,
//! including its quirks (`band_freq[b−1]` in the `ignore_freqs` check,
//! masking skipping the base-band's +1 neighbour, the quality transform).

use super::fft64::Pad;
use super::gauss::fast_gauss;
use super::ingress::{
    colorspace_transform, display_model, display_model_srgb, fix_out_of_gamut, looks_relative,
};
use super::params::{Emission, InputEncoding, Params, Surround, TaskPar, ViewingConditions};
use super::pathway::{cycdeg_image, mtf_filter, visual_pathway};
use super::spectral;
use super::spyr0::Spyr0;
use crate::interp::interp1_linear;
use crate::resize::imresize;
use crate::{Error, Result};

/// Everything HDR-VDP-3 predicts for one image pair.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct HdrVdp3Result {
    /// `res['Q_JOD']` — the quality correlate in JOD units (10 = perfect).
    /// This is the value the JPEG-AI quality-assessment framework publishes
    /// as its `HDR_VDP_3` column.
    pub q_jod: f64,
    /// `res['Q']` — the raw quality correlate (`10 − 4 − ln(e^{−4} + Q_err)`).
    pub q: f64,
    /// `res['P_map']` — per-pixel detection probability (after spatial
    /// probability summation when enabled), row-major.
    pub p_map: Vec<f64>,
    /// `|reconSpyr(D)|` before the probability transform — the
    /// normalised-difference map (1 ≈ threshold).
    pub s_map: Vec<f64>,
    /// Image width.
    pub width: usize,
    /// Image height.
    pub height: usize,
    /// Fraction of pixels whose native-channel values were clamped to
    /// `1e-6` by the out-of-gamut fix.
    pub gamut_clamped_fraction: f64,
    /// The `hdrvdp:lowvals` condition — the input looked like relative
    /// (0–1) values for an absolute-unit encoding.
    pub input_looks_relative: bool,
    /// Either image contained NaN (replaced by 1e-5, matching upstream).
    pub had_nan: bool,
}

/// `hdrvdp3(task, test, reference, color_encoding, pixels_per_degree, {})`
/// — except everything that was ambient upstream is required in `par`:
/// viewing conditions, display emission, the lot.
///
/// * `test`, `reference` — interleaved `width·height·C` in the units
///   `par.encoding` declares (`f64`).
///
/// # Errors
/// `Error::SizeMismatch` when the two buffers disagree, `ChannelMismatch`
/// on buffer-length or encoding/channel-count mismatches,
/// `InvalidResolution` for `ppd < 4` (raised by [`Params::new`]),
/// `MissingEmission` for generic encoding without a table,
/// `SpectralData` when the emission table's channel count disagrees with
/// the image. `ImageTooSmall` is a defensive floor for degenerate
/// geometry (`ceil(log2(ppd)) − 2 < 0`, i.e. `ppd ≤ 2`) — unreachable
/// through the validated `Params` path; small *images* are legal upstream
/// (`maxPyrHt` clamps to 0 levels, yielding the 2-band hi/lo pyramid).
pub fn hdrvdp3(
    test: &[f64],
    reference: &[f64],
    width: usize,
    height: usize,
    par: &Params,
) -> Result<HdrVdp3Result> {
    if test.len() != reference.len() {
        return Err(Error::SizeMismatch {
            reference: (width, height),
            distorted: (width, height),
        });
    }
    let mp = par.task_par();
    let channels = match par.encoding.channels() {
        Some(c) => c,
        None => match &par.emission {
            Emission::Custom { columns, .. } => columns.len(),
            _ => return Err(Error::MissingEmission),
        },
    };
    if channels == 0 || test.len() != width * height * channels {
        return Err(Error::ChannelMismatch {
            expected: width * height * channels,
            got: test.len(),
        });
    }

    // ---- encoding → absolute units ----
    let mut test_n = decode_encoding(test, par.encoding);
    let mut ref_n = decode_encoding(reference, par.encoding);

    // ---- emission spectra (IMG_E) ----
    let mut img_e = spectral::emission_columns(&par.emission)?;
    if channels == 1 && img_e.len() > 1 {
        // sum the spectral responses into one luminance channel.
        let n = img_e[0].len();
        let mut sum = vec![0.0; n];
        for c in &img_e {
            for (s, &v) in sum.iter_mut().zip(c) {
                *s += v;
            }
        }
        img_e = vec![sum];
    }
    if img_e.len() != channels {
        return Err(Error::SpectralData(format!(
            "emission has {} channels, image has {channels}",
            img_e.len()
        )));
    }

    // ---- standard → native colour transform + gamut fix ----
    let mut gamut_fraction = 0.0;
    if let Some(m) = spectral::itu2native(par.encoding, &img_e)? {
        colorspace_transform(&mut test_n, &m);
        colorspace_transform(&mut ref_n, &m);
        gamut_fraction =
            fix_out_of_gamut(&mut test_n, channels).max(fix_out_of_gamut(&mut ref_n, channels));
    }

    // ---- plausible-input signal (the `lowvals` warning) ----
    let looks_rel = !mp.disable_lowvals_warning
        && match par.encoding {
            InputEncoding::Luminance | InputEncoding::LumaDisplay => {
                looks_relative(&test_n, channels, 0)
            }
            InputEncoding::SrgbDisplay
            | InputEncoding::RgbBt709
            | InputEncoding::RgbBt2020
            | InputEncoding::RgbNative
            | InputEncoding::Xyz => looks_relative(&ref_n, channels, 1),
            InputEncoding::Generic => false,
        };

    // ---- surround → padding mode ----
    let pad = determine_pad(&par.viewing.surround, &ref_n, channels);
    let pad_at = |k: usize| match &pad {
        PadKind::Symmetric => Pad::Symmetric,
        PadKind::PerChannel(v) => Pad::Constant(v[k]),
    };

    // ---- both images through the visual pathway ----
    let pr = visual_pathway(
        &ref_n,
        width,
        height,
        channels,
        &mp,
        &par.viewing,
        &img_e,
        &pad_at,
    )?;
    let pt = visual_pathway(
        &test_n,
        width,
        height,
        channels,
        &mp,
        &par.viewing,
        &img_e,
        &pad_at,
    )?;
    let had_nan = pr.had_nan || pt.had_nan;
    let b_t = &pt.bands;
    let b_r = &pr.bands;
    let band_freq = b_t.freqs().to_vec();
    let b_count = b_t.band_count();

    // ---- per-adaptation-luminance CSF tables ----
    let csf_la: Vec<f64> = (0..256)
        .map(|i| 10f64.powf(-5.0 + 10.0 * i as f64 / 255.0))
        .collect();
    let csf_log_la: Vec<f64> = csf_la.iter().map(|v| v.log10()).collect();
    let csf: Vec<Vec<f64>> = (0..b_count)
        .map(|b| {
            csf_la
                .iter()
                .map(|&la| ncsf(band_freq[b], la, &mp))
                .collect()
        })
        .collect();

    let l_mean_adapt: Vec<f64> = pr
        .l_adapt
        .iter()
        .zip(&pt.l_adapt)
        .map(|(&r, &t)| (r + t) / 2.0)
        .collect();
    let log_la: Vec<f64> = l_mean_adapt
        .iter()
        .map(|&v| v.clamp(csf_la[0], csf_la[255]).log10())
        .collect();

    // ---- per-pixel difference mask (native units, >0.1% rel. change) ----
    let npix = width * height;
    let mut diff_mask = vec![0.0f64; npix];
    for i in 0..npix {
        let mut any = false;
        for c in 0..channels {
            let t = test_n[i * channels + c];
            let r = ref_n[i * channels + c];
            if ((t - r) / r).abs() > 0.001 {
                any = true;
            }
        }
        if any {
            diff_mask[i] = 1.0;
        }
    }

    let mut d_bands = b_t.clone();
    let mut q_err = 0.0f64;

    for b in 0..b_count {
        let p = 10f64.powf(mp.mask_p);
        let q = 10f64.powf(mp.mask_q);
        let pf = 10f64.powf(mp.psych_func_slope) / p;

        let (bw, bh) = b_t.band_dims(b);
        // CSF_b: nCSF at this band's frequency, per-pixel on the band's
        // lattice via the resized log adaptation map.
        let log_la_rs: Vec<f64> = imresize(&log_la, width, height, bw, bh)
            .iter()
            .map(|&v| v.clamp(csf_log_la[0], csf_log_la[255]))
            .collect();
        let csf_b: Vec<f64> = log_la_rs
            .iter()
            .map(|&l| interp1_linear(&csf_log_la, &csf[b], l))
            .collect();

        let band_diff: Vec<f64> = b_t
            .band(b)
            .iter()
            .zip(b_r.band(b).iter())
            .map(|(&t, &r)| t - r)
            .collect();

        let n_ncsf: Vec<f64> = if b == b_count - 1 {
            // Base band: dominant-frequency sensitivity, scalar.
            let rho_bb = cycdeg_image(bw, bh, band_freq[b] * 4.0);
            let mut f = super::fft64::fft2_of(&band_diff, bw, bh);
            f[0] = super::fft64::C64::new(0.0, 0.0);
            let (mut best, mut idx) = (0.0f64, 0usize);
            for (i, c) in f.iter().enumerate() {
                let m = (c.re * c.re + c.im * c.im).sqrt();
                if m > best {
                    best = m;
                    idx = i;
                }
            }
            let bb_freq = rho_bb[idx];
            let l_mean = 10f64.powf(log_la_rs.iter().sum::<f64>() / log_la_rs.len() as f64);
            vec![1.0 / ncsf(bb_freq, l_mean, &mp); bw * bh]
        } else {
            csf_b.iter().map(|&c| 1.0 / c).collect()
        };

        let ex_diff: Vec<f64> = band_diff.iter().map(|&v| sign_pow(v, p)).collect();

        if let Some(thr) = mp.ignore_freqs_lower_than {
            // Upstream quirk: reads the PREVIOUS band's frequency
            // (b−1 wraps to the last row at b=0).
            let fb = band_freq[b.wrapping_sub(1).min(b_count - 1)];
            if fb < thr {
                d_bands.set_band(b, &vec![0.0; bw * bh]);
            } else {
                apply_band(
                    &mut d_bands,
                    b,
                    &ex_diff,
                    &n_ncsf,
                    b_t,
                    b_r,
                    &diff_mask,
                    width,
                    height,
                    bw,
                    bh,
                    p,
                    q,
                    pf,
                    &mp,
                    &mut q_err,
                    b_count,
                );
            }
        } else {
            apply_band(
                &mut d_bands,
                b,
                &ex_diff,
                &n_ncsf,
                b_t,
                b_r,
                &diff_mask,
                width,
                height,
                bw,
                bh,
                p,
                q,
                pf,
                &mp,
                &mut q_err,
                b_count,
            );
        }
    }

    let levs: Vec<usize> = (0..b_count).collect();
    let s_map: Vec<f64> = d_bands.reconstruct(&levs).iter().map(|v| v.abs()).collect();
    let mut p_map: Vec<f64> = s_map
        .iter()
        .map(|&s| 1.0 - (0.5f64.ln() * s).exp())
        .collect();
    if mp.do_sprob_sum {
        let si_sigma = 10f64.powf(mp.si_sigma) * ppd_of(&par.viewing);
        let log1m: Vec<f64> = p_map.iter().map(|&v| (1.0 - v + 1e-4).ln()).collect();
        let g = fast_gauss(&log1m, width, height, si_sigma, true, Pad::Constant(0.0));
        for (pm, &gv) in p_map.iter_mut().zip(&g) {
            *pm = 1.0 - gv.exp();
        }
    }
    if mp.do_pixel_threshold {
        let m_thr = 0.01;
        for (i, pm) in p_map.iter_mut().enumerate() {
            let is_diff = 1.0
                - (0.5f64.ln()
                    * (((pr.l_adapt[i] - pt.l_adapt[i]).abs() / pr.l_adapt[i] / m_thr).powf(3.5)))
                .exp();
            *pm *= is_diff;
        }
    }

    let quality_floor = -4.0f64;
    let max_q = 10.0f64;
    let q_score = max_q + quality_floor - (quality_floor.exp() + q_err).ln();
    let q_jod = 10.0 - 0.52 * (10.0 - q_score).max(0.0).powf(1.2812);

    Ok(HdrVdp3Result {
        q_jod,
        q: q_score,
        p_map,
        s_map,
        width,
        height,
        gamut_clamped_fraction: gamut_fraction,
        input_looks_relative: looks_rel,
        had_nan,
    })
}

fn ppd_of(v: &ViewingConditions) -> f64 {
    v.pixels_per_degree
}

/// The per-band masking + difference block from `hdrvdp3.py` — extracted
/// for readability; identical arithmetic, `o` fixed at 0 (sp0 has a single
/// orientation per level).
#[allow(clippy::too_many_arguments)]
fn apply_band(
    d_bands: &mut Spyr0,
    b: usize,
    ex_diff: &[f64],
    n_ncsf: &[f64],
    b_t: &Spyr0,
    b_r: &Spyr0,
    diff_mask: &[f64],
    img_w: usize,
    img_h: usize,
    bw: usize,
    bh: usize,
    p: f64,
    q: f64,
    pf: f64,
    mp: &TaskPar,
    q_err: &mut f64,
    b_count: usize,
) {
    let d: Vec<f64> = if mp.do_masking {
        let k_self = 10f64.powf(mp.mask_self);
        let k_xo = 10f64.powf(mp.mask_xo); // multiplies 0 for sp0
        let k_xn = 10f64.powf(mp.mask_xn);
        let do_mask_xn = mp.mask_xn > -10.0;

        let self_mask = mutual_masking(b_t, b_r, b, mp);
        let mut mask_xn = vec![0.0; bw * bh];
        if b > 0 && do_mask_xn {
            let prev = mutual_masking(b_t, b_r, b - 1, mp);
            let resized = imresize(
                &prev,
                b_t.band_dims(b - 1).0,
                b_t.band_dims(b - 1).1,
                bw,
                bh,
            );
            for (m, &v) in mask_xn.iter_mut().zip(&resized) {
                *m = v.max(0.0);
            }
        }
        if b < b_count - 2 && do_mask_xn {
            let next = mutual_masking(b_t, b_r, b + 1, mp);
            let resized = imresize(
                &next,
                b_t.band_dims(b + 1).0,
                b_t.band_dims(b + 1).1,
                bw,
                bh,
            );
            for (m, &v) in mask_xn.iter_mut().zip(&resized) {
                *m += v.max(0.0);
            }
        }
        ex_diff
            .iter()
            .enumerate()
            .map(|(i, &e)| {
                let n_mask = k_self * self_mask[i].abs().powf(q)
                    + k_xo * 0.0f64.powf(q)
                    + k_xn * mask_xn[i].abs().powf(q);
                e / (n_ncsf[i].powf(2.0 * p) + n_mask.powi(2)).sqrt()
            })
            .collect()
    } else {
        ex_diff
            .iter()
            .enumerate()
            .map(|(i, &e)| e / n_ncsf[i].powf(p))
            .collect()
    };
    let d_norm: Vec<f64> = d.iter().map(|&v| sign_pow(v, pf)).collect();
    d_bands.set_band(b, &d_norm);

    // Per-band error → Q accumulator.
    let diff_mask_b = if (bw, bh) == (img_w, img_h) {
        diff_mask.to_vec()
    } else {
        imresize(diff_mask, img_w, img_h, bw, bh)
    };
    let mut acc = 0.0;
    for (i, &dv) in d.iter().enumerate() {
        acc += (dv * diff_mask_b[i]).abs().powf(0.8);
    }
    *q_err += (acc / (bw * bh) as f64).powf(1.0 / 0.8) / b_count as f64;
}

/// `mutual_masking(b, 0)` — 3×3 mean pool of `min(|T|, |R|)` with
/// zero-filled borders (scipy `convolve2d 'same'`).
fn mutual_masking(b_t: &Spyr0, b_r: &Spyr0, b: usize, mp: &TaskPar) -> Vec<f64> {
    let (bw, bh) = b_t.band_dims(b);
    let tb = b_t.band(b);
    let rb = b_r.band(b);
    let mut m: Vec<f64> = tb
        .iter()
        .zip(&rb)
        .map(|(&t, &r)| t.abs().min(r.abs()))
        .collect();
    if mp.do_si_gauss {
        let sigma = 10f64.powf(mp.si_size);
        m = fast_gauss(&m, bw, bh, sigma, false, Pad::Constant(0.0));
        return m;
    }
    let fs = mp.masking_pool_size;
    let norm = 10f64.powf(mp.masking_norm);
    let c = (fs / 2) as isize;
    let inv = 1.0 / (fs * fs) as f64;
    let mut out = vec![0.0; bw * bh];
    for y in 0..bh {
        for x in 0..bw {
            let mut acc = 0.0;
            for ky in 0..fs {
                for kx in 0..fs {
                    let sy = y as isize + ky as isize - c;
                    let sx = x as isize + kx as isize - c;
                    if (0..bh as isize).contains(&sy) && (0..bw as isize).contains(&sx) {
                        acc += m[sy as usize * bw + sx as usize];
                    }
                }
            }
            out[y * bw + x] = (acc * inv).powf(1.0 / norm);
        }
    }
    out
}

/// The neural CSF — `hdrvdp_ncsf` for one (ρ, La) pair.
pub(crate) fn ncsf(rho: f64, lum: f64, mp: &TaskPar) -> f64 {
    if rho <= 1e-4 {
        return 0.0;
    }
    let lum_lut: Vec<f64> = mp.csf_lums.iter().map(|v| v.log10()).collect();
    let log_lum = lum.log10().clamp(lum_lut[0], lum_lut[6]);
    let mut par = [0.0f64; 4];
    for (k, p) in par.iter_mut().enumerate() {
        let col: Vec<f64> = mp.csf_params.iter().map(|row| row[k + 1]).collect();
        *p = interp1_linear(&lum_lut, &col, log_lum);
    }
    let s = par[3]
        / ((1.0 + (par[0] * rho).powf(par[1]))
            * (1.0 / (1.0 - (-(rho / 7.0).powi(2)).exp())).powf(par[2]))
        .sqrt();
    let mtf = mtf_filter(&[rho], &mp.mtf_params_a, &mp.mtf_params_b)[0];
    let mut s = s / mtf;
    if mp.do_aesl {
        // `hdrvdp_aesl` — 10^(−(10^slope_freq·log2(ρ+γ))·max(0, age−24)).
        let gamma = 10f64.powf(mp.aesl_base);
        s *= 10f64.powf(
            -(10f64.powf(mp.aesl_slope_freq) * (rho + gamma).log2()) * (mp.age - 24.0).max(0.0),
        );
    }
    s
}

fn sign_pow(x: f64, e: f64) -> f64 {
    x.signum() * x.abs().powf(e)
}

/// `decode_encoding` — map `encoding`'s code values to absolute
/// display-relative units (per upstream's per-encoding display models).
fn decode_encoding(img: &[f64], enc: InputEncoding) -> Vec<f64> {
    match enc {
        InputEncoding::Luminance
        | InputEncoding::RgbBt709
        | InputEncoding::RgbBt2020
        | InputEncoding::RgbNative
        | InputEncoding::Xyz
        | InputEncoding::Generic => img.to_vec(),
        InputEncoding::LumaDisplay => display_model(img, 2.2, 99.0, 1.0),
        InputEncoding::SrgbDisplay => display_model_srgb(img),
    }
}

/// The resolved padding — `hdrvdp_determine_padding`.
enum PadKind {
    Symmetric,
    PerChannel(Vec<f64>),
}

fn determine_pad(surround: &Surround, reference: &[f64], channels: usize) -> PadKind {
    match surround {
        Surround::None => PadKind::Symmetric,
        Surround::Mean => {
            let mut g = vec![0.0; channels];
            for (c, slot) in g.iter_mut().enumerate() {
                let col: Vec<f64> = reference
                    .iter()
                    .skip(c)
                    .step_by(channels)
                    .copied()
                    .collect();
                *slot = super::pathway::geomean(&col);
            }
            PadKind::PerChannel(g)
        }
        Surround::Uniform(v) => PadKind::PerChannel(vec![*v; channels]),
        Surround::PerChannel(v) => PadKind::PerChannel(v.clone()),
    }
}
