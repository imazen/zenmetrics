//! Contrast masking and the per-band difference signal `D`.
//!
//! Ported from the band loop of `hdrvdp.m` (MATLAB `hdrvdp-2.2.x`).
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
//! ## What masking means here
//!
//! A difference is harder to see on top of busy content than on a flat field.
//! HDR-VDP-2 models that with a divisive transducer: the excitation difference
//! between the two images is divided by a denominator built from the *masking
//! activity* present in **both** images at that band, orientation and pixel.
//!
//! "Present in both" is the key: the masker is `min(|B_test|, |B_ref|)`
//! (upstream's `mutual_masking`), blurred by a 3×3 box as a crude
//! phase-uncertainty mechanism. Using either image alone would let a
//! distortion mask *itself*.
//!
//! Three masking terms are summed, each with its own fitted weight:
//! **self** (same band, same orientation), **cross-orientation** (same band,
//! the other orientations), and **cross-neighbouring-band** (the level above
//! and below, resampled onto this band's grid).

use crate::bands::BandPyramid;
use crate::csf::ncsf;
use crate::interp::{clamp, clamp32, interp1_linear, interp1_linear32};
use crate::params::Params;
use crate::resize::imresize32;
use crate::spyr::Band;

/// Number of samples in the adapting-luminance CSF lookup
/// (upstream: `logspace(-5, 5, 256)`).
const CSF_LUT_N: usize = 256;

/// Relative difference above which a pixel counts as "actually different" for
/// the quality pooling (upstream's `diff_mask` threshold, 0.1 %).
pub const DIFF_MASK_THRESHOLD: f32 = 0.001;

/// `sign(x) · |x|^e` — upstream's `sign_pow`, the odd-symmetric power that
/// keeps the transducer's polarity.
#[must_use]
#[inline]
pub fn sign_pow(x: f32, e: f32) -> f32 {
    x.signum() * x.abs().powf(e)
}

/// Masking activity common to both images at one band and orientation:
/// `min(|B_test|, |B_ref|)` blurred by a 3×3 box.
///
/// The blur is upstream's "simplistic phase-uncertainty mechanism": without it
/// a masker whose zero crossings line up with the difference would fail to
/// mask it at all.
#[must_use]
pub fn mutual_masking(test: &Band, reference: &Band) -> Band {
    debug_assert_eq!(
        (test.width, test.height),
        (reference.width, reference.height)
    );
    let (w, h) = (test.width, test.height);
    let m: Vec<f32> = test
        .data
        .iter()
        .zip(&reference.data)
        .map(|(t, r)| t.abs().min(r.abs()))
        .collect();
    Band {
        width: w,
        height: h,
        data: box3x3(&m, w, h),
    }
}

/// 3×3 box blur with `conv2(..., 'same')` semantics — zero outside the image,
/// so border pixels see a *smaller* sum divided by the same 9.
///
/// Interior pixels take a branch-free path that performs the identical nine
/// additions in the identical `(dy, dx)` row-major order (starting from the
/// same `0.0` accumulator).
fn box3x3(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0.0; w * h];

    let border = |out: &mut Vec<f32>, y: usize, xs: core::ops::Range<usize>| {
        for x in xs {
            let mut acc = 0.0;
            for dy in -1isize..=1 {
                let sy = y as isize + dy;
                if sy < 0 || sy >= h as isize {
                    continue;
                }
                for dx in -1isize..=1 {
                    let sx = x as isize + dx;
                    if sx < 0 || sx >= w as isize {
                        continue;
                    }
                    acc += src[sy as usize * w + sx as usize];
                }
            }
            out[y * w + x] = acc / 9.0;
        }
    };

    if w >= 3 && h >= 3 {
        border(&mut out, 0, 0..w);
        for y in 1..h - 1 {
            let up = &src[(y - 1) * w..y * w];
            let mid = &src[y * w..(y + 1) * w];
            let dn = &src[(y + 1) * w..(y + 2) * w];
            border(&mut out, y, 0..1);
            let orow = &mut out[y * w..(y + 1) * w];
            for x in 1..w - 1 {
                let acc = 0.0
                    + up[x - 1]
                    + up[x]
                    + up[x + 1]
                    + mid[x - 1]
                    + mid[x]
                    + mid[x + 1]
                    + dn[x - 1]
                    + dn[x]
                    + dn[x + 1];
                orow[x] = acc / 9.0;
            }
            border(&mut out, y, w - 1..w);
        }
        border(&mut out, h - 1, 0..w);
    } else {
        for y in 0..h {
            border(&mut out, y, 0..w);
        }
    }
    out
}

/// What the band loop produces.
#[derive(Debug, Clone)]
pub struct Masking {
    /// The per-band, per-orientation difference signal, reshaped by the
    /// psychometric slope — the input to visibility pooling.
    pub d_bands: BandPyramid,
    /// One quality term per `(band, orientation)`, already weighted and
    /// divided by the total plane count. Summing them gives upstream's `Q`.
    pub quality_terms: Vec<f64>,
}

/// Run the masking / transducer band loop over a decomposed pair.
///
/// * `test`, `reference` — decomposed band pyramids of the two images.
/// * `l_adapt` — the pair's mean adapting luminance map (cd/m²), full size:
///   upstream averages the two images' `L_adapt` before this loop.
/// * `diff_mask` — per-pixel flag, full size, for pixels whose relative
///   difference exceeds [`DIFF_MASK_THRESHOLD`]; used only by the quality
///   accumulation, where it stops a tiny localised difference from being
///   diluted across the whole image.
///
/// # Panics
/// If the two pyramids differ in shape, or the maps do not match the image
/// size.
#[must_use]
pub fn run(
    test: &BandPyramid,
    reference: &BandPyramid,
    l_adapt: &[f32],
    diff_mask: &[f32],
    par: &Params,
) -> Masking {
    let (d_bands, quality_terms) = run_impl(test, reference, l_adapt, diff_mask, par, true);
    Masking {
        d_bands: d_bands.unwrap(),
        quality_terms,
    }
}

/// Score-only variant: the per-plane quality terms without materialising the
/// reshaped `d_bands` (which only feed [`crate::pool::visibility`]).
#[must_use]
pub fn run_terms(
    test: &BandPyramid,
    reference: &BandPyramid,
    l_adapt: &[f32],
    diff_mask: &[f32],
    par: &Params,
) -> Vec<f64> {
    run_impl(test, reference, l_adapt, diff_mask, par, false).1
}

fn run_impl(
    test: &BandPyramid,
    reference: &BandPyramid,
    l_adapt: &[f32],
    diff_mask: &[f32],
    par: &Params,
    want_d_bands: bool,
) -> (Option<BandPyramid>, Vec<f64>) {
    assert_eq!(test.count(), reference.count(), "band count mismatch");
    let (w, h) = (test.width, test.height);
    assert_eq!(l_adapt.len(), w * h, "l_adapt does not match the image");
    assert_eq!(diff_mask.len(), w * h, "diff_mask does not match the image");

    let b_count = test.count();
    let total_planes = test.total_planes();
    let band_freq = test.frequencies(par.pix_per_deg);

    // Adapting-luminance axis for the per-band CSF lookup. The axis is built
    // in f64 (scalar table) and cast down once; the per-pixel lookups run f32.
    let csf_la: Vec<f64> = (0..CSF_LUT_N)
        .map(|i| 10f64.powf(-5.0 + 10.0 * i as f64 / (CSF_LUT_N - 1) as f64))
        .collect();
    let csf_log_la: Vec<f32> = csf_la.iter().map(|v| v.log10() as f32).collect();
    // CSF[b][i] = nCSF(band_freq[b], csf_la[i]).
    let csf: Vec<Vec<f32>> = band_freq
        .iter()
        .map(|f| csf_la.iter().map(|la| ncsf(*f, *la, par) as f32).collect())
        .collect();
    let la_lo = csf_la[0] as f32;
    let la_hi = csf_la[CSF_LUT_N - 1] as f32;

    let log_la: Vec<f32> = l_adapt
        .iter()
        .map(|v| clamp32(*v, la_lo, la_hi).log10())
        .collect();

    // Quality weights are tabulated on a DECREASING frequency axis; reverse
    // both so the shared linear interpolator can be used.
    let qf: Vec<f64> = par.quality_band_freq.iter().rev().copied().collect();
    let qw: Vec<f64> = par.quality_band_w.iter().rev().copied().collect();

    let p = par.transducer_p() as f32;
    let q = par.transducer_q() as f32;
    let pf = par.transducer_pf() as f32;
    let k_self = 10f64.powf(par.mask_self) as f32;
    let k_xo = 10f64.powf(par.mask_xo) as f32;
    let k_xn = 10f64.powf(par.mask_xn) as f32;

    let mut d_bands = want_d_bands.then(|| test.zeros_like());
    let mut quality_terms = Vec::with_capacity(total_planes);

    // Mutual masking per (band, orientation), computed once: the loop reads
    // each band's own, its neighbours', and the sum across orientations.
    let mm: Vec<Vec<Band>> = (0..b_count)
        .map(|b| {
            (0..test.orientations(b))
                .map(|o| mutual_masking(test.band(b, o), reference.band(b, o)))
                .collect()
        })
        .collect();

    for b in 0..b_count {
        let (bw, bh) = (test.band(b, 0).width, test.band(b, 0).height);
        let band_norm = 2f32.powi(b as i32);

        // Cross-orientation masking: total activity in this band.
        let mut mask_xo_total = vec![0.0f32; bw * bh];
        for plane in &mm[b] {
            for (a, v) in mask_xo_total.iter_mut().zip(&plane.data) {
                *a += v;
            }
        }

        // Per-pixel contrast sensitivity at this band's frequency, from the
        // local adapting luminance resampled onto the band grid.
        let log_la_rs = imresize32(&log_la, w, h, bw, bh);
        let csf_b: Vec<f32> = log_la_rs
            .iter()
            .map(|l| {
                let l = clamp32(*l, csf_log_la[0], csf_log_la[CSF_LUT_N - 1]);
                interp1_linear32(&csf_log_la, &csf[b], l)
            })
            .collect();

        // Quality weight for this band's frequency (scalar — stays f64).
        let f_b = clamp(band_freq[b], qf[0], qf[qf.len() - 1]);
        let w_f = interp1_linear(&qf, &qw, f_b);
        let diff_mask_b = imresize32(diff_mask, w, h, bw, bh);

        for o in 0..test.orientations(b) {
            let t = test.band(b, o);
            let r = reference.band(b, o);
            let self_mask = &mm[b][o];

            // Cross-neighbouring-band masking, resampled onto this grid.
            let mut mask_xn = vec![0.0f32; bw * bh];
            if b > 0 {
                let src = &mm[b - 1][o.min(mm[b - 1].len() - 1)];
                let rs = imresize32(&src.data, src.width, src.height, bw, bh);
                for (a, v) in mask_xn.iter_mut().zip(&rs) {
                    *a += v.max(0.0) / (band_norm / 2.0);
                }
            }
            if b + 2 < b_count {
                let src = &mm[b + 1][o.min(mm[b + 1].len() - 1)];
                let rs = imresize32(&src.data, src.width, src.height, bw, bh);
                for (a, v) in mask_xn.iter_mut().zip(&rs) {
                    *a += v.max(0.0) / (band_norm * 2.0);
                }
            }

            let mut d = vec![0.0f32; bw * bh];
            for i in 0..bw * bh {
                let band_diff = t.data[i] - r.data[i];
                let ex_diff = sign_pow(band_diff / band_norm, p) * band_norm;

                // The base band carries no CSF weighting; it was already
                // CSF-filtered during decomposition.
                let n_ncsf = if b == b_count - 1 {
                    1.0
                } else {
                    1.0 / csf_b[i]
                };

                d[i] = if par.do_masking {
                    let sm = self_mask.data[i];
                    let xo = (mask_xo_total[i] - sm).max(0.0);
                    let n_mask = band_norm
                        * (k_self * (sm / n_ncsf / band_norm).powf(q)
                            + k_xo * (xo / n_ncsf / band_norm).powf(q)
                            + k_xn * (mask_xn[i] / n_ncsf).powf(q));
                    ex_diff / (n_ncsf.powf(2.0 * p) + n_mask * n_mask).sqrt()
                } else {
                    ex_diff / n_ncsf.powf(p)
                };
            }

            // Quality term, from `D` *before* the psychometric reshaping —
            // upstream's `(log(msre+eps) − log(eps)) · w_f`, summed into `Q`
            // (res.Q = 100 − Q). No plane-count normalisation. The sum itself
            // is a reduction, so it accumulates in f64 over the f32 plane.
            let msre = {
                let s: f64 = d
                    .iter()
                    .zip(&diff_mask_b)
                    .map(|(v, m)| {
                        let vm = *v * *m;
                        f64::from(vm) * f64::from(vm)
                    })
                    .sum();
                s.sqrt() / (bw * bh) as f64
            };
            quality_terms.push(((msre + 1e-12).ln() - 1e-12f64.ln()) * w_f);

            // Reshape by the psychometric slope for the visibility pooling.
            if let Some(db) = d_bands.as_mut() {
                let out = db.band_mut(b, o);
                for (dst, v) in out.data.iter_mut().zip(&d) {
                    *dst = sign_pow(v / band_norm, pf) * band_norm;
                }
            }
        }
    }

    (d_bands, quality_terms)
}

/// Per-pixel "actually different" flag over an interleaved pair, as `1.0` /
/// `0.0` — upstream's
/// `any((abs(test − reference) ./ reference) > 0.001, 3)`.
///
/// Note the denominator is the **reference**, so this is a relative test, and
/// a reference pixel of 0 makes any difference infinite (i.e. flagged), which
/// is the intended behaviour for absolute-luminance input where 0 cd/m² is
/// already clamped away.
#[must_use]
pub fn diff_mask(test: &[f32], reference: &[f32], channels: usize) -> Vec<f32> {
    assert_eq!(test.len(), reference.len());
    assert!(channels >= 1);
    test.chunks(channels)
        .zip(reference.chunks(channels))
        .map(|(t, r)| {
            let any = t
                .iter()
                .zip(r)
                .any(|(a, b)| ((a - b) / b).abs() > DIFF_MASK_THRESHOLD);
            f32::from(any)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bands::decompose;
    use crate::csf::{ncsf_coeffs, ncsf_from_coeffs};
    use crate::pathway::{Pathway, visual_pathway};
    use crate::photoreceptor::Photoreceptor;
    use crate::spectral::{DisplaySpectra, emission_spectra, lmsr_matrix};

    fn pathway_of(im: &[f32], w: usize, h: usize, par: &Params) -> Pathway {
        let pn = Photoreceptor::new(par);
        let lmsr = lmsr_matrix(&emission_spectra(DisplaySpectra::D65, 1));
        let mean = im.iter().map(|v| *v as f64).sum::<f64>() / im.len() as f64;
        visual_pathway(im, w, h, par, &pn, &lmsr, &[mean])
    }

    /// A textured reference plus an optional additive perturbation.
    fn pair(w: usize, h: usize, amp: f32, busy: f32) -> (Vec<f32>, Vec<f32>) {
        let pi = core::f32::consts::PI;
        let reference: Vec<f32> = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32, (i / w) as f32);
                100.0 * (1.0 + busy * (2.0 * pi * x / 6.0).sin() * (2.0 * pi * y / 6.0).cos())
            })
            .collect();
        let test: Vec<f32> = reference
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let (x, y) = ((i % w) as f32, (i / w) as f32);
                v + 100.0 * amp * (2.0 * pi * x / 5.0).sin() * (2.0 * pi * y / 7.0).sin()
            })
            .collect();
        (reference, test)
    }

    fn run_pair(reference: &[f32], test: &[f32], w: usize, h: usize, par: &Params) -> Masking {
        let pr = pathway_of(reference, w, h, par);
        let pt = pathway_of(test, w, h, par);
        let (br, pad) = decompose(&pr, par, None);
        let (bt, _) = decompose(&pt, par, Some(pad));
        let l_adapt: Vec<f32> = pr
            .l_adapt
            .iter()
            .zip(&pt.l_adapt)
            .map(|(a, b)| 0.5 * (a + b))
            .collect();
        let dm = diff_mask(test, reference, 1);
        run(&bt, &br, &l_adapt, &dm, par)
    }

    #[test]
    fn sign_pow_is_odd_and_preserves_polarity() {
        assert!((sign_pow(8.0, 1.0 / 3.0) - 2.0).abs() < 1e-6);
        assert!((sign_pow(-8.0, 1.0 / 3.0) + 2.0).abs() < 1e-6);
        assert_eq!(sign_pow(0.0, 3.0), 0.0);
        for x in [-3.0, -0.4, 0.4, 3.0] {
            assert!((sign_pow(x, 2.0) + sign_pow(-x, 2.0)).abs() < 1e-6);
        }
    }

    #[test]
    fn mutual_masking_takes_the_smaller_of_the_two() {
        // A masker present in only ONE image must not mask: min() kills it.
        let a = Band {
            width: 3,
            height: 3,
            data: vec![10.0f32; 9],
        };
        let b = Band {
            width: 3,
            height: 3,
            data: vec![0.0f32; 9],
        };
        let m = mutual_masking(&a, &b);
        assert!(m.data.iter().all(|v| *v == 0.0));
        // Present in both → survives, and the 3×3 box preserves a flat field
        // in the interior while attenuating it at the border (conv2 'same').
        let m = mutual_masking(&a, &a);
        assert!((m.data[4] - 10.0).abs() < 1e-6, "centre {}", m.data[4]);
        assert!(m.data[0] < 10.0, "corner should see zero-padding");
        // Sign is discarded: masking is about magnitude.
        let neg = Band {
            width: 3,
            height: 3,
            data: vec![-10.0f32; 9],
        };
        assert_eq!(mutual_masking(&a, &neg).data, m.data);
    }

    #[test]
    fn diff_mask_flags_only_relative_changes_past_the_threshold() {
        let r = [100.0f32, 100.0, 100.0];
        let t = [100.0f32, 100.05, 101.0]; // 0 %, 0.05 %, 1 %
        let m = diff_mask(&t, &r, 1);
        assert_eq!(m, vec![0.0f32, 0.0, 1.0]);
        // Multi-channel: any channel over the threshold flags the pixel.
        let r3 = [100.0f32, 100.0, 100.0];
        let t3 = [100.0f32, 100.0, 101.0];
        assert_eq!(diff_mask(&t3, &r3, 3), vec![1.0f32]);
    }

    #[test]
    fn an_identical_pair_produces_no_difference_signal() {
        let par = Params::new(30.0);
        let (w, h) = (64usize, 64usize);
        let (reference, _) = pair(w, h, 0.0, 0.2);
        let m = run_pair(&reference, &reference, w, h, &par);
        for b in 0..m.d_bands.count() {
            for o in 0..m.d_bands.orientations(b) {
                for v in &m.d_bands.band(b, o).data {
                    assert!(v.abs() < 1e-6, "identical pair gave D = {v}");
                }
            }
        }
        // Every quality term is then log(eps) — a large negative number, the
        // most-similar end of the scale.
        assert!(m.quality_terms.iter().all(|t| t.is_finite()));
    }

    #[test]
    fn a_bigger_distortion_produces_a_bigger_difference_signal() {
        let par = Params::new(30.0);
        let (w, h) = (64usize, 64usize);
        let energy = |amp: f32| -> f64 {
            let (r, t) = pair(w, h, amp, 0.2);
            let m = run_pair(&r, &t, w, h, &par);
            (0..m.d_bands.count())
                .flat_map(|b| (0..m.d_bands.orientations(b)).map(move |o| (b, o)))
                .map(|(b, o)| {
                    m.d_bands
                        .band(b, o)
                        .data
                        .iter()
                        .map(|v| (*v as f64) * (*v as f64))
                        .sum::<f64>()
                })
                .sum::<f64>()
                .sqrt()
        };
        let (e_small, e_big) = (energy(0.01), energy(0.10));
        assert!(e_small > 0.0);
        assert!(
            e_big > 3.0 * e_small,
            "10 % distortion ({e_big}) should dwarf 1 % ({e_small})"
        );
    }

    #[test]
    fn masking_reduces_the_difference_signal_on_busy_content() {
        // The whole point of the transducer: the SAME distortion is less
        // visible on a busy background than on a smooth one.
        let par = Params::new(30.0);
        let (w, h) = (64usize, 64usize);
        let energy = |busy: f32| -> f64 {
            let (r, t) = pair(w, h, 0.05, busy);
            let m = run_pair(&r, &t, w, h, &par);
            (0..m.d_bands.count())
                .flat_map(|b| (0..m.d_bands.orientations(b)).map(move |o| (b, o)))
                .map(|(b, o)| {
                    m.d_bands
                        .band(b, o)
                        .data
                        .iter()
                        .map(|v| (*v as f64) * (*v as f64))
                        .sum::<f64>()
                })
                .sum::<f64>()
                .sqrt()
        };
        let smooth = energy(0.0);
        let busy = energy(0.5);
        assert!(
            busy < smooth,
            "masking should suppress the difference on busy content: {busy} vs {smooth}"
        );
    }

    #[test]
    fn disabling_masking_raises_the_difference_signal() {
        // Sanity that `do_masking` actually reaches the denominator.
        let (w, h) = (64usize, 64usize);
        let (r, t) = pair(w, h, 0.05, 0.4);
        let energy = |par: &Params| -> f64 {
            let m = run_pair(&r, &t, w, h, par);
            (0..m.d_bands.count())
                .flat_map(|b| (0..m.d_bands.orientations(b)).map(move |o| (b, o)))
                .map(|(b, o)| {
                    m.d_bands
                        .band(b, o)
                        .data
                        .iter()
                        .map(|v| (*v as f64) * (*v as f64))
                        .sum::<f64>()
                })
                .sum::<f64>()
                .sqrt()
        };
        let on = energy(&Params::new(30.0));
        let mut off_par = Params::new(30.0);
        off_par.do_masking = false;
        let off = energy(&off_par);
        assert!(
            off > on,
            "masking-off ({off}) should exceed masking-on ({on})"
        );
    }

    #[test]
    fn quality_terms_are_one_per_plane_and_rise_with_distortion() {
        let par = Params::new(30.0);
        let (w, h) = (64usize, 64usize);
        let sum_q = |amp: f32| -> (f64, usize) {
            let (r, t) = pair(w, h, amp, 0.2);
            let m = run_pair(&r, &t, w, h, &par);
            (
                m.quality_terms.iter().sum::<f64>(),
                m.d_bands.total_planes(),
            )
        };
        let (q_small, planes) = sum_q(0.01);
        let (q_big, planes2) = sum_q(0.10);
        assert_eq!(planes, planes2);
        // 64×64 → 2 oriented levels → 1 + 2·4 + 1 = 10 planes.
        assert_eq!(planes, 10);
        // Each term is (log(msre+ε) − log ε)·w_f ≥ 0 — zero for an identical
        // pair — so the accumulated `Q` is non-negative and RISES as the
        // distortion grows; `res.Q = 100 − Q` falls.
        assert!(q_small >= 0.0 && q_big >= 0.0, "{q_small} / {q_big}");
        assert!(
            q_big > q_small,
            "Q should rise with distortion: {q_small} → {q_big}"
        );
    }

    #[test]
    fn csf_lookup_helpers_agree_with_the_direct_call() {
        // The band loop resolves nCSF coefficients once per adapting
        // luminance; make sure that fast path is the same function.
        let par = Params::new(30.0);
        for &la in &[0.01f64, 1.0, 90.0] {
            let c = ncsf_coeffs(la, &par);
            for &rho in &[0.5, 4.0, 15.0] {
                assert!((ncsf(rho, la, &par) - ncsf_from_coeffs(rho, &c)).abs() < 1e-15);
            }
        }
    }
}
