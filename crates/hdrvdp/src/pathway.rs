//! The visual pathway: absolute luminance → achromatic photoreceptor response.
//!
//! Ported from `hdrvdp_visual_pathway.m` of the MATLAB `hdrvdp-2.2.x` release,
//! up to (but not including) the multi-scale decomposition, which lands in the
//! next chunk.
//!
//! Copyright (c) 2011, Rafal Mantiuk <mantiuk@gmail.com>
//!
//! Permission to use, copy, modify, and/or distribute this software for any
//! purpose with or without fee is hereby granted, provided that the above
//! copyright notice and this permission notice appear in all copies.
//!
//! THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
//! WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
//! MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
//! ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
//! WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
//! ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
//! OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
//!
//! Stages, in order:
//!
//! 1. **Optical MTF** — each channel is convolved with the eye's MTF in the
//!    Fourier domain, padded to 2× with the surround luminance, then clamped to
//!    `[1e-5, 1e10]` cd/m².
//! 2. **Photoreceptor spectral sensitivity** — the channels are mixed into
//!    `(L, M, S, R)` responses through the display's emission spectra.
//! 3. **Adapting luminance** — `L_adapt = R_L + R_M`.
//! 4. **Photoreceptor non-linearity** — L, M and R go through their JND-space
//!    lookup tables. S is deliberately dropped: HDR-VDP-2 is achromatic.
//! 5. **DC removal, separately per pathway** — the cone sum and the rod
//!    response each have their own mean subtracted before being added, so a
//!    scotopic and a photopic region of the same image do not cross-contaminate
//!    each other's mean.

use crate::csf::mtf;
use crate::fft::{conv_fft_real, cycles_per_degree_grid};
use crate::params::Params;
use crate::photoreceptor::Photoreceptor;

/// Lower clamp applied after the optical transfer function, in cd/m².
pub const MIN_NITS: f64 = 1e-5;
/// Upper clamp applied after the optical transfer function, in cd/m².
pub const MAX_NITS: f64 = 1e10;

/// What the visual pathway produces for one image.
#[derive(Debug, Clone)]
pub struct Pathway {
    /// Achromatic photoreceptor response, DC-removed, row-major `h × w`.
    pub achromatic: Vec<f64>,
    /// Per-pixel adapting luminance in cd/m² (`L + M`), row-major `h × w`.
    pub l_adapt: Vec<f64>,
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
}

/// Run stages 1–5 for one image.
///
/// * `nits` — interleaved absolute luminance, `h · w · channels` values.
/// * `lmsr` — the `channels × 4` mixing matrix from
///   [`crate::spectral::lmsr_matrix`].
/// * `surround` — surround luminance per channel, used as the FFT pad value.
///
/// # Panics
/// If the buffer lengths are inconsistent with `width`, `height` and the
/// matrix's channel count.
#[must_use]
pub fn visual_pathway(
    nits: &[f64],
    width: usize,
    height: usize,
    par: &Params,
    pn: &Photoreceptor,
    lmsr: &[[f64; 4]],
    surround: &[f64],
) -> Pathway {
    let mtf_filter = mtf_filter_for(width, height, par);
    visual_pathway_with_filter(
        nits,
        width,
        height,
        par,
        pn,
        lmsr,
        surround,
        mtf_filter.as_deref(),
    )
}

/// The Fourier-domain optical MTF filter [`visual_pathway`] applies, on the
/// 2×-padded lattice — `None` when `par.do_mtf` is off.
///
/// The filter depends only on the geometry and `par`, never on the pixels, so
/// the metric computes it once and shares it across the reference and test
/// images instead of rebuilding it (a full grid of `exp` calls) per image.
pub(crate) fn mtf_filter_for(width: usize, height: usize, par: &Params) -> Option<Vec<f64>> {
    let (pad_w, pad_h) = (width * 2, height * 2);
    par.do_mtf.then(|| {
        cycles_per_degree_grid(pad_w, pad_h, par.pix_per_deg)
            .into_iter()
            .map(|rho| mtf(rho, par))
            .collect()
    })
}

/// [`visual_pathway`] with the MTF filter (from [`mtf_filter_for`]) supplied
/// by the caller so an image pair shares one.
#[allow(clippy::too_many_arguments)]
pub(crate) fn visual_pathway_with_filter(
    nits: &[f64],
    width: usize,
    height: usize,
    par: &Params,
    pn: &Photoreceptor,
    lmsr: &[[f64; 4]],
    surround: &[f64],
    mtf_filter: Option<&[f64]>,
) -> Pathway {
    let channels = lmsr.len();
    assert_eq!(
        nits.len(),
        width * height * channels,
        "visual_pathway: pixel buffer does not match {width}×{height}×{channels}"
    );
    assert_eq!(
        surround.len(),
        channels,
        "visual_pathway: need one surround value per channel"
    );
    let n = width * height;

    // ── 1. Optical transfer function ─────────────────────────────────────
    let (pad_w, pad_h) = (width * 2, height * 2);

    let mut optical: Vec<Vec<f64>> = Vec::with_capacity(channels);
    for c in 0..channels {
        let plane: Vec<f64> = (0..n).map(|i| nits[i * channels + c]).collect();
        let filtered = match mtf_filter {
            Some(f) => conv_fft_real(&plane, width, height, f, pad_w, pad_h, surround[c])
                .into_iter()
                .map(|v| crate::interp::clamp(v, MIN_NITS, MAX_NITS))
                .collect(),
            None => plane,
        };
        optical.push(filtered);
    }

    // ── 2. Photoreceptor spectral sensitivity ────────────────────────────
    // R[k][i] for k in {L, M, S, R}; S is computed for completeness and then
    // dropped, exactly as upstream does.
    let mut r_lmsr = [vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]];
    for (c, plane) in optical.iter().enumerate() {
        for (k, out) in r_lmsr.iter_mut().enumerate() {
            let m = lmsr[c][k];
            for (o, v) in out.iter_mut().zip(plane) {
                *o += m * v;
            }
        }
    }

    // ── 3. Adapting luminance ────────────────────────────────────────────
    let l_adapt: Vec<f64> = r_lmsr[0]
        .iter()
        .zip(&r_lmsr[1])
        .map(|(l, m)| l + m)
        .collect();

    // ── 4. Photoreceptor non-linearity ───────────────────────────────────
    let p_l: Vec<f64> = r_lmsr[0].iter().map(|&v| pn.cone(v)).collect();
    let p_m: Vec<f64> = r_lmsr[1].iter().map(|&v| pn.cone(v)).collect();
    let p_r: Vec<f64> = r_lmsr[3].iter().map(|&v| pn.rod(v)).collect();

    // ── 5. DC removal, cone and rod pathways separately ──────────────────
    let mut cones: Vec<f64> = p_l.iter().zip(&p_m).map(|(a, b)| a + b).collect();
    remove_mean(&mut cones);
    let mut rods = p_r;
    remove_mean(&mut rods);

    let achromatic: Vec<f64> = cones.iter().zip(&rods).map(|(a, b)| a + b).collect();

    Pathway {
        achromatic,
        l_adapt,
        width,
        height,
    }
}

/// Per-channel surround luminance: either the configured constant, or the
/// geometric mean of each channel (upstream's `surround_l = -1`).
///
/// The geometric mean is computed in the log domain over positive values;
/// non-positive samples are skipped (they cannot contribute a finite log), and
/// a channel with no positive sample falls back to [`MIN_NITS`].
#[must_use]
pub fn surround_per_channel(nits: &[f64], channels: usize, configured: Option<f64>) -> Vec<f64> {
    if let Some(v) = configured {
        return vec![v; channels];
    }
    (0..channels)
        .map(|c| {
            let mut sum = 0.0;
            let mut count = 0usize;
            for p in nits.chunks_exact(channels) {
                if p[c] > 0.0 {
                    sum += p[c].ln();
                    count += 1;
                }
            }
            if count == 0 {
                MIN_NITS
            } else {
                (sum / count as f64).exp()
            }
        })
        .collect()
}

fn remove_mean(v: &mut [f64]) {
    if v.is_empty() {
        return;
    }
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    for x in v.iter_mut() {
        *x -= mean;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spectral::{DisplaySpectra, emission_spectra, lmsr_matrix};

    fn fixture(channels: usize) -> (Params, Photoreceptor, Vec<[f64; 4]>) {
        let par = Params::new(30.0);
        let pn = Photoreceptor::new(&par);
        let spectra = if channels == 1 {
            DisplaySpectra::D65
        } else {
            DisplaySpectra::CcflLcd
        };
        let lmsr = lmsr_matrix(&emission_spectra(spectra, channels));
        (par, pn, lmsr)
    }

    #[test]
    fn uniform_field_matching_its_surround_yields_a_zero_response() {
        // A flat field surrounded by the same luminance has no contrast
        // anywhere, so after DC removal the achromatic response is identically
        // zero — at every adaptation level. This also pins that the MTF has
        // unit DC gain: any gain error would show up as a non-zero constant
        // before the mean is subtracted, and as an `l_adapt` shift after.
        let (par, pn, lmsr) = fixture(3);
        let (w, h) = (16usize, 12usize);
        for &y in &[0.05f64, 5.0, 500.0] {
            let nits = vec![y; w * h * 3];
            let out = visual_pathway(&nits, w, h, &par, &pn, &lmsr, &[y; 3]);
            for v in &out.achromatic {
                assert!(v.abs() < 1e-9, "flat field at {y} cd/m² gave {v}");
            }
            let a0 = out.l_adapt[0];
            assert!(a0 > 0.0);
            for v in &out.l_adapt {
                assert!((v - a0).abs() < 1e-9 * a0.max(1.0));
            }
        }
    }

    #[test]
    fn a_dark_surround_bleeds_into_a_flat_field() {
        // The complement of the test above, kept because it is a real property
        // of the model and a trap for anyone reading a "flat field" result: the
        // optical MTF is applied by circular convolution over a domain padded
        // with `surround_l`, whose default is 1e-5 cd/m². A bright flat field
        // against that dark surround is therefore NOT flat afterwards — it is
        // pulled down, most at the border. Upstream 2.1.3 changed `surround_l`
        // handling for exactly this reason ("avoids false detection at the
        // image border"), so the effect is expected, not a bug — but the
        // reference and test images must share one surround value or the
        // difference between them is pure border artefact.
        let (par, pn, lmsr) = fixture(1);
        let (w, h) = (16usize, 16usize);
        let nits = vec![100.0; w * h];
        let out = visual_pathway(&nits, w, h, &par, &pn, &lmsr, &[1e-5]);
        let corner = out.achromatic[0];
        let centre = out.achromatic[(h / 2) * w + w / 2];
        assert!(
            corner < centre,
            "the dark surround should depress the corner ({corner}) below the centre ({centre})"
        );
        assert!(
            (corner - centre).abs() > 1e-3,
            "the surround effect should be measurable, got {corner} vs {centre}"
        );
    }

    #[test]
    fn achromatic_response_is_mean_free() {
        let (par, pn, lmsr) = fixture(3);
        let (w, h) = (13usize, 9usize);
        let nits: Vec<f64> = (0..w * h * 3)
            .map(|i| 1.0 + 200.0 * ((i as f64) * 0.11).sin().abs())
            .collect();
        let out = visual_pathway(&nits, w, h, &par, &pn, &lmsr, &[1e-5; 3]);
        let mean = out.achromatic.iter().sum::<f64>() / out.achromatic.len() as f64;
        assert!(mean.abs() < 1e-9, "mean = {mean}");
    }

    #[test]
    fn contrast_grows_the_response_magnitude() {
        // A higher-amplitude pattern at the same mean luminance must produce a
        // larger achromatic response. This exercises the whole chain: MTF,
        // spectral mixing, JND space, DC removal.
        let (par, pn, lmsr) = fixture(1);
        let (w, h) = (32usize, 32usize);
        // Surround = mean luminance, so the measured energy is the grating's
        // and not the border falloff's (see `a_dark_surround_bleeds_into_a_
        // flat_field`).
        let energy = |amp: f64| -> f64 {
            let nits: Vec<f64> = (0..w * h)
                .map(|i| {
                    let x = (i % w) as f64;
                    100.0 * (1.0 + amp * (2.0 * core::f64::consts::PI * x / 8.0).sin())
                })
                .collect();
            let out = visual_pathway(&nits, w, h, &par, &pn, &lmsr, &[100.0]);
            out.achromatic.iter().map(|v| v * v).sum::<f64>().sqrt()
        };
        let e_small = energy(0.01);
        let e_big = energy(0.20);
        assert!(e_small > 0.0);
        assert!(
            e_big > 5.0 * e_small,
            "20% contrast ({e_big}) should dwarf 1% ({e_small})"
        );
    }

    #[test]
    fn mtf_attenuates_high_frequencies_more_than_low() {
        // The optics are a low-pass: at a fixed contrast, a fine grating must
        // lose more amplitude than a coarse one once do_mtf is on.
        let (mut par, pn, lmsr) = fixture(1);
        let (w, h) = (64usize, 8usize);
        // Surround = mean luminance so the border falloff does not swamp the
        // grating. A 4 px period is the finest usable one here: a 2 px period
        // samples sin(πx) = 0 at every integer x, i.e. it is invisible to the
        // sampler before the optics ever see it.
        let energy = |par: &Params, period_px: f64| -> f64 {
            let nits: Vec<f64> = (0..w * h)
                .map(|i| {
                    let x = (i % w) as f64;
                    100.0 * (1.0 + 0.1 * (2.0 * core::f64::consts::PI * x / period_px).sin())
                })
                .collect();
            let out = visual_pathway(&nits, w, h, par, &pn, &lmsr, &[100.0]);
            out.achromatic.iter().map(|v| v * v).sum::<f64>().sqrt()
        };
        let coarse_on = energy(&par, 32.0);
        let fine_on = energy(&par, 4.0);
        par.do_mtf = false;
        let coarse_off = energy(&par, 32.0);
        let fine_off = energy(&par, 4.0);
        assert!(coarse_off > 0.0 && fine_off > 0.0);
        // Relative loss from enabling the MTF.
        let loss_coarse = 1.0 - coarse_on / coarse_off;
        let loss_fine = 1.0 - fine_on / fine_off;
        assert!(
            loss_fine > loss_coarse,
            "MTF should hurt the 4 px grating ({loss_fine}) more than the 32 px one ({loss_coarse})"
        );
        assert!(loss_fine > 0.0);
    }

    #[test]
    fn surround_per_channel_configured_and_geometric() {
        let nits = [1.0, 10.0, 100.0, 1000.0]; // 2 pixels × 2 channels
        assert_eq!(surround_per_channel(&nits, 2, Some(0.25)), vec![0.25, 0.25]);
        let g = surround_per_channel(&nits, 2, None);
        // channel 0: geomean(1, 100) = 10; channel 1: geomean(10, 1000) = 100.
        assert!((g[0] - 10.0).abs() < 1e-9, "{g:?}");
        assert!((g[1] - 100.0).abs() < 1e-9, "{g:?}");
        // No positive samples → the floor, not NaN.
        assert_eq!(surround_per_channel(&[0.0, 0.0], 1, None), vec![MIN_NITS]);
    }

    #[test]
    fn optical_stage_clamps_into_the_representable_range() {
        // Absurd inputs must not escape the [1e-5, 1e10] clamp into the JND
        // lookup, which would otherwise index off the end of the table.
        let (par, pn, lmsr) = fixture(1);
        let (w, h) = (8usize, 8usize);
        let mut nits = vec![1e-30; w * h];
        nits[0] = 1e20;
        let out = visual_pathway(&nits, w, h, &par, &pn, &lmsr, &[1e-5]);
        assert!(out.achromatic.iter().all(|v| v.is_finite()));
        assert!(out.l_adapt.iter().all(|v| v.is_finite() && *v >= 0.0));
    }
}
