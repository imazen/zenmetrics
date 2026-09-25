#![forbid(unsafe_code)]

//! Display-aware native-CPU cvvdp scoring — the ONLY SDR route for CPU
//! `--metric cvvdp`. cvvdp has no default display in zenmetrics: every SDR
//! score names one (`--display-model <name>`, jobs `cvvdp@<name>`). This
//! builds the in-tree CPU port with BOTH halves of an upstream display
//! preset — photometry (`DisplayModel::by_name`) and viewing geometry
//! (`DisplayGeometry::by_name`, which sets pixels per degree) — and caches
//! one instance per image size.
//!
//! Every SDR cvvdp column names its display: `<cpu column>_<display name>`
//! (e.g. `cvvdp_cpu_imazen_v0_1_0_standard_4k`). Columns without a display
//! suffix predate 2026-09-25 and hold `standard_4k` scores; only the HDR
//! routes (their own reference-peak display, qualified by `hdr_mode`) still
//! write the display-less form.
//!
//! `standard_4k` (75.40 pixels per degree) is the preset upstream
//! ColorVideoVDP defaults to and every historical zenmetrics score used.
//! `standard_fhd` is what the JPEG AIC evaluation passes for its CVVDP anchor
//! on SDR images (AIC-4 Common Test Conditions v2.0, wg1n101246 §4:
//! `cvvdp -d standard_fhd`, 37.84 pixels per degree, 200 cd/m² peak, 0.2
//! cd/m² black, 0.3979 cd/m² reflected). Measured effect on the 300 AIC-4
//! sample pairs: `benchmarks/cvvdp_aic_discrepancy_2026-09-22.md`.

use zenmetrics_api::cvvdp_cpu::{Cvvdp, CvvdpParams, DisplayGeometry, DisplayModel};

use crate::metrics::display::CvvdpDisplay;

use crate::decode::Rgb8Image;

/// A CPU cvvdp scorer bound to one display ([`CvvdpDisplay`]: an official
/// preset or a custom photometry + geometry).
pub(crate) struct CpuCvvdpDisplayScorer {
    display: CvvdpDisplay,
    model: DisplayModel,
    geometry: DisplayGeometry,
    /// Scorer for the most recent image size; rebuilt when the size changes.
    cached: Option<(u32, u32, Box<Cvvdp>)>,
}

impl CpuCvvdpDisplayScorer {
    /// Bind the CPU port to `display` — its photometry AND geometry.
    pub(crate) fn new(display: CvvdpDisplay) -> Self {
        Self {
            model: display.model(),
            geometry: display.geometry(),
            display,
            cached: None,
        }
    }

    /// Output column for this display: `<cpu cvvdp column>_<display name>`
    /// ([`crate::metrics::MetricKind::columns_for`]).
    pub(crate) fn column(&self) -> &'static str {
        crate::metrics::MetricKind::Cvvdp.columns_for(Some(&self.display))[0]
    }

    /// Pixels per degree this scorer's geometry derives.
    pub(crate) fn pixels_per_degree(&self) -> f32 {
        self.geometry.pixels_per_degree()
    }

    /// Score one sRGB8 pair (JOD, 10 = identical).
    pub(crate) fn score(
        &mut self,
        reference: &Rgb8Image,
        distorted: &Rgb8Image,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        if reference.width != distorted.width || reference.height != distorted.height {
            return Err(format!(
                "cvvdp: reference ({}×{}) and distorted ({}×{}) differ in size",
                reference.width, reference.height, distorted.width, distorted.height
            )
            .into());
        }
        let (w, h) = (reference.width, reference.height);
        if !matches!(&self.cached, Some((cw, ch, _)) if *cw == w && *ch == h) {
            self.cached = None;
            let params = CvvdpParams {
                display: self.model,
                ..CvvdpParams::default()
            };
            let inner = Cvvdp::with_geometry(w, h, params, self.geometry)
                .map_err(|e| format!("cvvdp::Cvvdp::with_geometry: {e}"))?;
            self.cached = Some((w, h, Box::new(inner)));
        }
        let (_, _, scorer) = self.cached.as_mut().expect("built above");
        let v = scorer
            .score(&reference.pixels, &distorted.pixels)
            .map_err(|e| format!("cvvdp score ({}): {e}", self.display))?;
        let v = f64::from(v);
        if !v.is_finite() {
            return Err(format!("cvvdp (cpu, {}): non-finite score {v}", self.display).into());
        }
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::display::{DisplayPreset, parse_display};

    fn pair(w: u32, h: u32) -> (Rgb8Image, Rgb8Image) {
        let n = (w * h * 3) as usize;
        let reference: Vec<u8> = (0..n).map(|i| ((i * 37 + i / 7) % 251) as u8).collect();
        let distorted: Vec<u8> = reference
            .iter()
            .enumerate()
            .map(|(i, &v)| if i % 11 == 0 { v.saturating_add(9) } else { v })
            .collect();
        (
            Rgb8Image {
                pixels: reference,
                width: w,
                height: h,
            },
            Rgb8Image {
                pixels: distorted,
                width: w,
                height: h,
            },
        )
    }

    fn scorer(p: DisplayPreset) -> CpuCvvdpDisplayScorer {
        CpuCvvdpDisplayScorer::new(p.into())
    }

    /// `standard_4k` must reproduce the cvvdp crate's own default
    /// (`Cvvdp::new` with default params — 4K photometry and geometry)
    /// bit-for-bit: that is every historical zenmetrics cvvdp score.
    #[test]
    fn standard_4k_matches_the_cvvdp_crate_default_exactly() {
        let (r, d) = pair(96, 80);
        let named = scorer(DisplayPreset::Standard4k).score(&r, &d).unwrap();
        let historical = Cvvdp::new(96, 80, CvvdpParams::default())
            .unwrap()
            .score(&r.pixels, &d.pixels)
            .unwrap();
        let historical = f64::from(historical);
        assert_eq!(
            named.to_bits(),
            historical.to_bits(),
            "{named} vs {historical}"
        );
    }

    /// The AIC display is 37.84 pixels per degree (AIC CTC v2.0), and
    /// `standard_4k` (75.40) scores differently.
    #[test]
    fn standard_fhd_is_the_aic_ctc_geometry_and_moves_the_score() {
        let mut fhd = scorer(DisplayPreset::StandardFhd);
        assert!(
            (fhd.pixels_per_degree() - 37.8425).abs() < 1e-3,
            "{}",
            fhd.pixels_per_degree()
        );
        assert_eq!(fhd.model.y_peak, 200.0);
        let (r, d) = pair(96, 80);
        let s_fhd = fhd.score(&r, &d).unwrap();
        let s_4k = scorer(DisplayPreset::Standard4k).score(&r, &d).unwrap();
        assert!(
            (s_fhd - s_4k).abs() > 1e-3,
            "display geometry must move the score: fhd {s_fhd} vs 4k {s_4k}"
        );
    }

    #[test]
    fn columns_always_name_the_display() {
        let base = crate::metrics::MetricKind::Cvvdp.column_names()[0];
        assert_eq!(
            scorer(DisplayPreset::Standard4k).column(),
            format!("{base}_standard_4k")
        );
        assert_eq!(
            scorer(DisplayPreset::StandardFhd).column(),
            format!("{base}_standard_fhd")
        );
        let study = CpuCvvdpDisplayScorer::new(parse_display("squintly_n1").unwrap());
        assert_eq!(study.column(), format!("{base}_squintly_n1"));
    }
}
