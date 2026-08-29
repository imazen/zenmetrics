//! HDR-VDP-2.2 calibrated parameters.
//!
//! Ported from `hdrvdp_parse_options.m` of the MATLAB `hdrvdp-2.2.x` release.
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
//! Every value here is a *calibration constant*: it was fitted to psychophysical
//! data (ModelFest, csfla, complexfest for the visibility model; LDR + HDR
//! quality datasets for the 2.2 pooling). Changing one silently invalidates the
//! published correlations, so they are `pub` for inspection but the struct is
//! constructed through [`Params::default`] and adjusted through named methods.

/// Peak contrast sensitivity from Daly's CSF at `L_adapt = 30 cd/m²`.
///
/// Only used to derive [`Params::sensitivity_correction`]; upstream exposes it
/// through the `peak_sensitivity` option as
/// `daly_peak_contrast_sens / 10^-peak_sensitivity`.
pub const DALY_PEAK_CONTRAST_SENS: f64 = 0.006_894_596;

/// The `peak_sensitivity` exponent baked into the 2.2 defaults (upstream:
/// `sensitivity_correction = daly_peak_contrast_sens / 10.^-2.4`).
pub const DEFAULT_PEAK_SENSITIVITY: f64 = 2.4;

/// Number of orientations in the steerable-pyramid decomposition.
pub const ORIENT_COUNT: usize = 4;

/// HDR-VDP-2.2 model parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct Params {
    /// Visual resolution in pixels per degree of visual angle. There is no
    /// meaningful default — it depends on display size, resolution and viewing
    /// distance — so callers must supply it (see [`crate::pix_per_deg`]).
    pub pix_per_deg: f64,

    /// `daly_peak_contrast_sens / 10^-peak_sensitivity`; scales the
    /// photoreceptor JND space.
    pub sensitivity_correction: f64,

    /// Luminance of the surround, in cd/m², per colour channel. Upstream
    /// default is a single very low value (`1e-5`) replicated across channels;
    /// `None` means "use the geometric mean of the reference image", upstream's
    /// `surround_l = -1`.
    pub surround_l: Option<f64>,

    // ── Optical MTF: MTF(ρ) = Σ_k a_k · exp(−b_k · ρ) ──────────────────────
    /// MTF amplitudes, derived upstream from the two-parameter
    /// `par = [0.061466549455263, 0.99727370023777070]` parametrization.
    pub mtf_params_a: [f64; 4],
    /// MTF decay rates in 1/(cycles per degree).
    pub mtf_params_b: [f64; 4],

    // ── Neural CSF ────────────────────────────────────────────────────────
    /// Six rows of `[unused, p1, p2, p3, p4]`, one per entry of
    /// [`Self::csf_lums`]. Only columns 1..=4 are read; column 0 is carried
    /// verbatim from upstream and unused by the nCSF.
    pub csf_params: [[f64; 5]; 6],
    /// The adapting luminances (cd/m²) the nCSF was calibrated at.
    pub csf_lums: [f64; 6],
    /// Joint rod+cone sensitivity parameters `[peak, p6, p7, p8]` (paper §p6-p8).
    pub csf_sa: [f64; 4],
    /// Rod sensitivity parameters
    /// `[peak_l, low_s, low_exp, high_s, high_exp, rod_sens]`.
    pub csf_sr_par: [f64; 6],
    /// `log10` scale applied to the rod pathway (upstream default 0 → ×1).
    pub rod_sensitivity: f64,

    // ── Masking / contrast transducer (all log10 in the reference) ─────────
    /// Transducer exponent, as `log10(p)`.
    pub mask_p: f64,
    /// Self-masking weight, as `log10(k)`.
    pub mask_self: f64,
    /// Cross-orientation masking weight, as `log10(k)`.
    pub mask_xo: f64,
    /// Cross-neighbouring-band masking weight, as `log10(k)`.
    pub mask_xn: f64,
    /// Masking exponent, as `log10(q)`.
    pub mask_q: f64,
    /// Slope of the psychometric function, as `log10(slope)`.
    pub psych_func_slope: f64,

    // ── Quality pooling (the 2.2 recalibration) ───────────────────────────
    /// Band centre frequencies (cycles/degree) the quality weights are given at.
    pub quality_band_freq: [f64; 7],
    /// Per-band quality weights (LDR + HDR fit, 2.2).
    pub quality_band_w: [f64; 7],
    /// Logistic mapping `Q → Q_MOS`, slope.
    pub quality_logistic_q1: f64,
    /// Logistic mapping `Q → Q_MOS`, offset.
    pub quality_logistic_q2: f64,

    // ── Optional stages ───────────────────────────────────────────────────
    /// Apply the optical MTF (upstream `do_mtf`).
    pub do_mtf: bool,
    /// Apply contrast masking (upstream `do_masking`).
    pub do_masking: bool,
    /// Apply spatial pooling to the visibility map (upstream
    /// `do_spatial_pooling`).
    pub do_spatial_pooling: bool,
}

impl Default for Params {
    fn default() -> Self {
        // Upstream: par = [0.061466549455263 0.99727370023777070], then
        //   a = [par2*0.426, par2*0.574, (1-par2)*par1, (1-par2)*(1-par1)]
        let p1 = 0.061_466_549_455_263_f64;
        let p2 = 0.997_273_700_237_770_7_f64;
        Self {
            // No sane default exists; `Params::default()` is only useful once
            // the caller sets this (or builds through `Params::new`).
            pix_per_deg: f64::NAN,
            sensitivity_correction: DALY_PEAK_CONTRAST_SENS / 10f64.powf(-DEFAULT_PEAK_SENSITIVITY),
            surround_l: Some(1e-5),

            mtf_params_a: [
                p2 * 0.426,
                p2 * 0.574,
                (1.0 - p2) * p1,
                (1.0 - p2) * (1.0 - p1),
            ],
            mtf_params_b: [0.028, 0.37, 37.0, 360.0],

            csf_params: [
                [0.0160737, 0.991265, 3.74038, 0.50722, 4.46044],
                [0.383873, 0.800889, 3.54104, 0.682505, 4.94958],
                [0.929301, 0.476505, 4.37453, 0.750315, 5.28678],
                [1.29776, 0.405782, 4.40602, 0.935314, 5.61425],
                [1.49222, 0.334278, 3.79542, 1.07327, 6.4635],
                [1.46213, 0.394533, 2.7755, 1.16577, 7.45665],
            ],
            csf_lums: [0.002, 0.02, 0.2, 2.0, 20.0, 150.0],
            csf_sa: [30.162, 4.0627, 1.6596, 0.2712],
            csf_sr_par: [1.1732, 1.1478, 1.2167, 0.5547, 2.9899, 1.1414],
            rod_sensitivity: 0.0,

            mask_p: 0.544068,
            mask_self: 0.189065,
            mask_xo: 0.449199,
            mask_xn: 1.52512,
            mask_q: 0.49576,
            psych_func_slope: 3.5f64.log10(),

            quality_band_freq: [15.0, 7.5, 3.75, 1.875, 0.9375, 0.4688, 0.2344],
            quality_band_w: [0.2832, 0.2142, 0.2690, 0.0398, 0.0003, 0.0003, 0.0002],
            quality_logistic_q1: 3.455,
            quality_logistic_q2: 0.8886,

            do_mtf: true,
            do_masking: true,
            do_spatial_pooling: true,
        }
    }
}

impl Params {
    /// Defaults at a given visual resolution (pixels per degree).
    #[must_use]
    pub fn new(pix_per_deg: f64) -> Self {
        Self {
            pix_per_deg,
            ..Self::default()
        }
    }

    /// Override the peak-sensitivity exponent (upstream's `peak_sensitivity`
    /// option): `sensitivity_correction = daly_peak / 10^-peak_sensitivity`.
    #[must_use]
    pub fn with_peak_sensitivity(mut self, peak_sensitivity: f64) -> Self {
        self.sensitivity_correction = DALY_PEAK_CONTRAST_SENS / 10f64.powf(-peak_sensitivity);
        self
    }

    /// The transducer exponent `p = 10^mask_p`.
    #[must_use]
    pub fn transducer_p(&self) -> f64 {
        10f64.powf(self.mask_p)
    }

    /// The masking exponent `q = 10^mask_q`.
    #[must_use]
    pub fn transducer_q(&self) -> f64 {
        10f64.powf(self.mask_q)
    }

    /// `pf = 10^psych_func_slope / p` — the exponent the transducer output is
    /// re-shaped by before pooling.
    #[must_use]
    pub fn transducer_pf(&self) -> f64 {
        10f64.powf(self.psych_func_slope) / self.transducer_p()
    }
}

/// Visual resolution in pixels per degree, from display geometry.
///
/// Ported from `hdrvdp_pix_per_deg.m` (ISC, see the module header).
///
/// * `display_diagonal_in` — display diagonal in inches.
/// * `resolution` — `[horizontal, vertical]` pixel count.
/// * `viewing_distance_m` — viewing distance in metres.
///
/// Assumes square pixels and that the display's physical aspect ratio equals
/// `resolution[0] : resolution[1]`.
#[must_use]
pub fn pix_per_deg(display_diagonal_in: f64, resolution: [f64; 2], viewing_distance_m: f64) -> f64 {
    let ar = resolution[0] / resolution[1];
    let height_mm = ((display_diagonal_in * 25.4).powi(2) / (1.0 + ar * ar)).sqrt();
    // `atand` in the reference — degrees.
    let height_deg = 2.0
        * (0.5 * height_mm / (viewing_distance_m * 1000.0))
            .atan()
            .to_degrees();
    resolution[1] / height_deg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mtf_amplitudes_sum_to_one() {
        // Σ a_k = par2*(0.426+0.574) + (1-par2)*(par1 + 1 - par1) = 1, so the
        // MTF is unity at ρ = 0 (no attenuation at DC).
        let p = Params::default();
        let sum: f64 = p.mtf_params_a.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "Σ a_k = {sum}");
    }

    #[test]
    fn sensitivity_correction_matches_reference() {
        // 0.006894596 / 10^-2.4
        let p = Params::default();
        let want = 0.006_894_596 / 10f64.powf(-2.4);
        assert!((p.sensitivity_correction - want).abs() < 1e-15);
        // ... and the documented default is reproducible through the setter.
        let q = Params::default().with_peak_sensitivity(DEFAULT_PEAK_SENSITIVITY);
        assert_eq!(p.sensitivity_correction, q.sensitivity_correction);
    }

    #[test]
    fn pix_per_deg_matches_reference_example() {
        // The worked example in hdrvdp_pix_per_deg.m's docstring:
        //   ppd = hdrvdp_pix_per_deg( 24, [1920 1200], 0.5 )
        // ar = 1.6
        // height_mm  = sqrt((24·25.4)² / (1 + 1.6²)) = 609.6/1.88680 = 323.09
        // height_deg = 2·atand(0.5·323.09/500) = 2·17.9146 = 35.829
        // ppd        = 1200/35.829 = 33.49
        let ppd = pix_per_deg(24.0, [1920.0, 1200.0], 0.5);
        assert!(
            (ppd - 33.49).abs() < 0.03,
            "ppd = {ppd}, expected ≈ 33.49 for a 24\" 1920×1200 at 0.5 m"
        );
        // Moving twice as close roughly halves the pixels-per-degree, but only
        // roughly: `atan` compresses, so the true factor is a little above ½.
        let near = pix_per_deg(24.0, [1920.0, 1200.0], 0.25);
        assert!(
            near > ppd / 2.0 && near < ppd,
            "ppd at 0.25 m ({near}) should sit between half and all of the 0.5 m value ({ppd})"
        );
    }

    #[test]
    fn transducer_exponents() {
        let p = Params::default();
        // The fit landed `mask_p` almost exactly on `psych_func_slope`:
        // 10^0.544068 = 3.499_999_6 ≈ 3.5 = 10^log10(3.5). That makes the
        // post-transducer reshaping exponent `pf = 10^slope / p` essentially
        // 1 — i.e. the 2.2 calibration effectively reshapes by nothing, and a
        // future refit that moves `mask_p` will make `pf` bite.
        assert!(
            (p.transducer_p() - 3.5).abs() < 1e-6,
            "{}",
            p.transducer_p()
        );
        assert!(
            (p.transducer_q() - 3.131_55).abs() < 1e-4,
            "{}",
            p.transducer_q()
        );
        assert!(
            (p.transducer_pf() - 1.0).abs() < 1e-6,
            "{}",
            p.transducer_pf()
        );
    }
}
