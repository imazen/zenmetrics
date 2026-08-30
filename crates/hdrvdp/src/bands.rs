//! The band list HDR-VDP-2's masking loop walks, and the base-band CSF
//! filtering that produces it.
//!
//! Ported from the tail of `hdrvdp_visual_pathway.m` (MATLAB `hdrvdp-2.2.x`).
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
//! ## Band indexing
//!
//! The masking loop does **not** walk a [`SteerablePyramid`] directly; it walks
//! a flat list whose per-band orientation counts are `[1, 4, 4, …, 4, 1]`:
//! band 0 is the high-pass residual, bands `1..=H` are the oriented levels
//! (fine → coarse), band `H+1` is the low-pass residual. Upstream's
//! `get_band(bands, b, o)` clamps with `oc = min(o, sz(b))`, so asking the
//! high-pass or low-pass band for orientation 3 returns its single plane —
//! [`BandPyramid::band`] reproduces that clamp rather than panicking.
//!
//! ## Only the base band is CSF-filtered
//!
//! The oriented bands are *not* CSF-filtered here; their sensitivity enters
//! later, through `N_nCSF = 1/CSF_b` in the masking step. Only the low-pass
//! residual is filtered, at the mean adapting luminance, with the frequency
//! grid built at `band_freq(end) · 2 · √2` — upstream carries a comment
//! wondering why `√2` works better than the 2 the derivation implies. It is
//! reproduced as-is: it is part of the calibration.

use crate::csf::ncsf;
use crate::fft::{conv_fft_real, cycles_per_degree_grid};
use crate::params::Params;
use crate::pathway::Pathway;
use crate::spyr::{Band, ORIENTATIONS, SteerablePyramid, build};

/// The flat band list the masking loop walks.
#[derive(Debug, Clone)]
pub struct BandPyramid {
    /// `bands[b]` holds 1 plane for `b = 0` (high-pass) and for the last `b`
    /// (low-pass), and [`ORIENTATIONS`] planes for every band in between.
    pub bands: Vec<Vec<Band>>,
    /// Source image width.
    pub width: usize,
    /// Source image height.
    pub height: usize,
}

impl BandPyramid {
    /// Number of bands — upstream's `b_count`, i.e. `spyrHt + 2`.
    #[must_use]
    pub fn count(&self) -> usize {
        self.bands.len()
    }

    /// Orientations carried by band `b` — upstream's `bands.sz(b)`.
    #[must_use]
    pub fn orientations(&self, b: usize) -> usize {
        self.bands[b].len()
    }

    /// Total plane count across all bands — upstream's `sum(bands.sz)`, the
    /// divisor the quality accumulation uses.
    #[must_use]
    pub fn total_planes(&self) -> usize {
        self.bands.iter().map(Vec::len).sum()
    }

    /// Band `b`, orientation `o`, with upstream's `oc = min(o, sz(b))` clamp
    /// so the single-plane high-pass and low-pass bands answer any `o`.
    #[must_use]
    pub fn band(&self, b: usize, o: usize) -> &Band {
        let planes = &self.bands[b];
        &planes[o.min(planes.len() - 1)]
    }

    /// Mutable access with the same clamp.
    pub fn band_mut(&mut self, b: usize, o: usize) -> &mut Band {
        let planes = &mut self.bands[b];
        let i = o.min(planes.len() - 1);
        &mut planes[i]
    }

    /// Band centre frequencies in cycles/degree, one per band.
    #[must_use]
    pub fn frequencies(&self, pix_per_deg: f64) -> Vec<f64> {
        (0..self.count())
            .map(|b| 2f64.powi(-(b as i32)) * pix_per_deg / 2.0)
            .collect()
    }

    /// An all-zero pyramid with the same shape — the accumulator the masking
    /// loop fills with `D` bands.
    #[must_use]
    pub fn zeros_like(&self) -> Self {
        Self {
            bands: self
                .bands
                .iter()
                .map(|planes| {
                    planes
                        .iter()
                        .map(|b| Band::zeros(b.width, b.height))
                        .collect()
                })
                .collect(),
            width: self.width,
            height: self.height,
        }
    }
}

/// Decompose an achromatic response into the band list, CSF-filtering the base
/// band.
///
/// `bb_padvalue` is the FFT pad value for the base-band filtering. Upstream
/// processes the **reference first with `None`** (which uses the base band's
/// own mean) and then passes the *same* value when processing the test image —
/// otherwise a difference in the two padding constants shows up as a false
/// detection all along the image border. Returns the value used, to be handed
/// back in for the second image.
#[must_use]
pub fn decompose(pathway: &Pathway, par: &Params, bb_padvalue: Option<f64>) -> (BandPyramid, f64) {
    let pyr: SteerablePyramid = build(&pathway.achromatic, pathway.width, pathway.height, None);
    let levels = pyr.height_levels();

    // Move the pyramid's planes into the band list — it is consumed here, so
    // cloning ~1.3× the image in f64 planes would be pure memcpy waste.
    let mut bands: Vec<Vec<Band>> = Vec::with_capacity(levels + 2);
    bands.push(vec![pyr.high_pass]);
    for level in pyr.levels {
        bands.push(level.into());
    }
    bands.push(vec![pyr.low_pass]);
    debug_assert_eq!(bands.len(), levels + 2);
    debug_assert!(
        bands[1..bands.len() - 1]
            .iter()
            .all(|p| p.len() == ORIENTATIONS)
    );

    // Base-band CSF filter, at the mean adapting luminance.
    let l_mean = pathway.l_adapt.iter().sum::<f64>() / pathway.l_adapt.len() as f64;
    let last = bands.len() - 1;
    let bb = &bands[last][0];
    let pad =
        bb_padvalue.unwrap_or_else(|| bb.data.iter().sum::<f64>() / bb.data.len().max(1) as f64);

    let band_freq_last = 2f64.powi(-((bands.len() - 1) as i32)) * par.pix_per_deg / 2.0;
    let (pw, ph) = (bb.width * 2, bb.height * 2);
    // `band_freq(end)·2·√2` — upstream's own comment: "I wish to know why
    // sqrt(2) works better, as it should be 2".
    let ppd_bb = band_freq_last * 2.0 * core::f64::consts::SQRT_2;
    let csf_bb: Vec<f64> = cycles_per_degree_grid(pw, ph, ppd_bb)
        .into_iter()
        .map(|rho| ncsf(rho, l_mean, par))
        .collect();
    let filtered = conv_fft_real(&bb.data, bb.width, bb.height, &csf_bb, pw, ph, pad);
    bands[last][0].data = filtered;

    (
        BandPyramid {
            bands,
            width: pathway.width,
            height: pathway.height,
        },
        pad,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::photoreceptor::Photoreceptor;
    use crate::spectral::{DisplaySpectra, emission_spectra, lmsr_matrix};

    fn pathway_of(im: &[f64], w: usize, h: usize, par: &Params) -> Pathway {
        let pn = Photoreceptor::new(par);
        let lmsr = lmsr_matrix(&emission_spectra(DisplaySpectra::D65, 1));
        let mean = im.iter().sum::<f64>() / im.len() as f64;
        crate::pathway::visual_pathway(im, w, h, par, &pn, &lmsr, &[mean])
    }

    fn textured(w: usize, h: usize) -> Vec<f64> {
        let pi = core::f64::consts::PI;
        (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f64, (i / w) as f64);
                100.0 * (1.0 + 0.3 * (2.0 * pi * x / 9.0).sin() + 0.2 * (2.0 * pi * y / 13.0).cos())
            })
            .collect()
    }

    #[test]
    fn band_list_has_the_reference_shape() {
        let par = Params::new(30.0);
        let (w, h) = (128usize, 96usize);
        let p = pathway_of(&textured(w, h), w, h, &par);
        let (bp, _) = decompose(&p, &par, None);
        // 128×96 with a 17-tap lofilt: 96→48→24 < 17? no, 24 ≥ 17 → 12 < 17.
        // So 3 oriented levels, 5 bands total.
        assert_eq!(bp.count(), 5);
        assert_eq!(bp.orientations(0), 1, "band 0 is the high-pass residual");
        assert_eq!(bp.orientations(4), 1, "the last band is the low-pass");
        for b in 1..4 {
            assert_eq!(bp.orientations(b), ORIENTATIONS);
        }
        // sum(bands.sz) = 1 + 3·4 + 1
        assert_eq!(bp.total_planes(), 14);
        // Frequencies halve per band and there is one per band.
        let f = bp.frequencies(30.0);
        assert_eq!(f.len(), 5);
        assert!((f[0] - 15.0).abs() < 1e-12);
        assert!((f[4] - 15.0 / 16.0).abs() < 1e-12);
    }

    #[test]
    fn orientation_index_clamps_on_the_single_plane_bands() {
        let par = Params::new(30.0);
        let (w, h) = (64usize, 64usize);
        let p = pathway_of(&textured(w, h), w, h, &par);
        let (bp, _) = decompose(&p, &par, None);
        let last = bp.count() - 1;
        for o in 0..ORIENTATIONS {
            assert!(
                std::ptr::eq(bp.band(0, o), bp.band(0, 0)),
                "high-pass must answer every orientation with its one plane"
            );
            assert!(std::ptr::eq(bp.band(last, o), bp.band(last, 0)));
        }
        // Oriented bands are genuinely distinct planes.
        assert!(!std::ptr::eq(bp.band(1, 0), bp.band(1, 2)));
    }

    #[test]
    fn base_band_padvalue_is_reused_across_the_pair() {
        // The reference's base-band pad value must be reused for the test
        // image; upstream changed this in 2.1.3 precisely because letting the
        // two differ produces a false detection along the border.
        let par = Params::new(30.0);
        let (w, h) = (64usize, 64usize);
        let a = pathway_of(&textured(w, h), w, h, &par);
        let mut dist = textured(w, h);
        for v in dist.iter_mut().take(w * h / 2) {
            *v *= 1.4;
        }
        let b = pathway_of(&dist, w, h, &par);

        let (_, pad_ref) = decompose(&a, &par, None);
        let (with_ref_pad, pad_used) = decompose(&b, &par, Some(pad_ref));
        assert_eq!(pad_used, pad_ref, "the supplied pad value must be honoured");

        let (with_own_pad, own) = decompose(&b, &par, None);
        assert!(
            (own - pad_ref).abs() > 0.0,
            "this fixture should give the two images different own-pad values"
        );
        let last = with_ref_pad.count() - 1;
        let diff = with_ref_pad.bands[last][0]
            .data
            .iter()
            .zip(&with_own_pad.bands[last][0].data)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f64, f64::max);
        assert!(
            diff > 0.0,
            "the pad value must actually reach the base-band filter"
        );
    }

    #[test]
    fn zeros_like_matches_the_shape_but_not_the_content() {
        let par = Params::new(30.0);
        let (w, h) = (64usize, 48usize);
        let p = pathway_of(&textured(w, h), w, h, &par);
        let (bp, _) = decompose(&p, &par, None);
        let z = bp.zeros_like();
        assert_eq!(z.count(), bp.count());
        for b in 0..bp.count() {
            assert_eq!(z.orientations(b), bp.orientations(b));
            for o in 0..bp.orientations(b) {
                assert_eq!(
                    (z.band(b, o).width, z.band(b, o).height),
                    (bp.band(b, o).width, bp.band(b, o).height)
                );
                assert!(z.band(b, o).data.iter().all(|v| *v == 0.0));
            }
        }
        assert!(bp.band(1, 0).data.iter().any(|v| v.abs() > 1e-9));
    }
}
