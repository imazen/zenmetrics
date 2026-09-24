#![forbid(unsafe_code)]

//! Display-aware native-CPU cvvdp scoring for `batch` / `score-pairs
//! --metric cvvdp --display-model <name>`.
//!
//! The default `--metric cvvdp` path goes through the umbrella
//! (`run_cpu_native_via_umbrella`), whose `cvvdp::Cvvdp::new` is fixed to
//! the `standard_4k` display (200 cd/m², 75.40 pixels per degree). That
//! default is unchanged. This module is the opt-in alternative: it builds
//! the in-tree CPU port with BOTH halves of a named upstream display preset
//! — photometry (`DisplayModel::by_name`) and viewing geometry
//! (`DisplayGeometry::by_name`, which sets pixels per degree) — and caches
//! one instance per image size.
//!
//! Scores for a non-default display land in their own column,
//! `<cpu column>_<display name>` (e.g.
//! `cvvdp_cpu_imazen_v0_1_0_standard_fhd`), so they can never be joined or
//! averaged with default-display scores by accident. `standard_4k` resolves
//! to the same parameters as the default and keeps the plain column.
//!
//! `standard_fhd` is the configuration the JPEG AIC evaluation uses for its
//! CVVDP anchor on SDR images (AIC-4 Common Test Conditions v2.0, wg1n101246
//! §4: `cvvdp -d standard_fhd`, 37.84 pixels per degree, 200 cd/m² peak,
//! 0.2 cd/m² black, 0.3979 cd/m² reflected). Measured effect on the 300 AIC-4
//! sample pairs: `benchmarks/cvvdp_aic_discrepancy_2026-09-22.md`.

use zenmetrics_api::cvvdp_cpu::{Cvvdp, CvvdpParams, DisplayGeometry, DisplayModel};

use crate::decode::Rgb8Image;

/// Display preset that reproduces the default (umbrella) CPU cvvdp scores.
pub(crate) const DEFAULT_DISPLAY: &str = "standard_4k";

/// Viewing geometries for the Squintly chroma study that upstream's
/// `display_models.json` does not carry: an iPhone-class 6.1" 1170x2532 panel
/// (~457 ppi) held at 30 cm (`squintly_n1`, the fixed-1x "normal" block,
/// ~94 ppd), and the same panel under 2x integer zoom (`squintly_m2`, ~47
/// ppd). Zoom makes each source pixel cover 2x2 display pixels, which halves
/// source pixels per degree; that is modelled as half the viewing distance.
fn study_geometry(name: &str) -> Option<DisplayGeometry> {
    let at = |distance_m: f32| DisplayGeometry {
        resolution_w: 1170,
        resolution_h: 2532,
        distance_m,
        diagonal_inches: 6.1,
    };
    match name {
        "squintly_n1" => Some(at(0.30)),
        "squintly_m2" => Some(at(0.15)),
        _ => None,
    }
}

/// A CPU cvvdp scorer bound to one named display preset.
pub(crate) struct CpuCvvdpDisplayScorer {
    name: String,
    display: DisplayModel,
    geometry: DisplayGeometry,
    /// Scorer for the most recent image size; rebuilt when the size changes.
    cached: Option<(u32, u32, Box<Cvvdp>)>,
}

impl CpuCvvdpDisplayScorer {
    /// Resolve `name` from the vendored upstream `display_models.json`.
    /// Both photometry and geometry are required: a preset without a
    /// resolution (FOV-only) cannot produce pixels per degree here.
    pub(crate) fn by_name(name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        if let Some(geometry) = study_geometry(name) {
            // Study presets are geometry-only additions; photometry is the
            // upstream `standard_phone` (500 cd/m², 250 lux ambient).
            let display = DisplayModel::by_name("standard_phone")
                .ok_or("vendored display_models.json lost its standard_phone preset")?;
            return Ok(Self {
                name: name.to_string(),
                display,
                geometry,
                cached: None,
            });
        }
        let display = DisplayModel::by_name(name).ok_or_else(|| {
            format!(
                "unknown --display-model preset {name:?}; see cvvdp's vendored \
                 `display_models.json` for valid names (e.g. standard_4k, standard_fhd, \
                 standard_phone)"
            )
        })?;
        let geometry = DisplayGeometry::by_name(name).ok_or_else(|| {
            format!(
                "--display-model preset {name:?} has photometry but no geometry \
                 (resolution); cvvdp scoring needs both"
            )
        })?;
        Ok(Self {
            name: name.to_string(),
            display,
            geometry,
            cached: None,
        })
    }

    /// Output column for this display: the plain CPU column for the default
    /// display, otherwise `<cpu column>_<display name>`.
    pub(crate) fn column_name(&self, base: &str) -> String {
        if self.name == DEFAULT_DISPLAY {
            base.to_string()
        } else {
            format!("{base}_{}", self.name)
        }
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
                display: self.display,
                ..CvvdpParams::default()
            };
            let inner = Cvvdp::with_geometry(w, h, params, self.geometry)
                .map_err(|e| format!("cvvdp::Cvvdp::with_geometry: {e}"))?;
            self.cached = Some((w, h, Box::new(inner)));
        }
        let (_, _, scorer) = self.cached.as_mut().expect("built above");
        let v = scorer
            .score(&reference.pixels, &distorted.pixels)
            .map_err(|e| format!("cvvdp score ({}): {e}", self.name))?;
        let v = f64::from(v);
        if !v.is_finite() {
            return Err(format!("cvvdp (cpu, {}): non-finite score {v}", self.name).into());
        }
        Ok(v)
    }
}

#[cfg(test)]
mod tests {

    /// The study geometries resolve, keep their own column suffix, and derive
    /// the pixels-per-degree the design registers (N1 ~94, M2 half of it).
    #[test]
    fn squintly_study_geometries_resolve_with_the_registered_ppd() {
        let n1 = super::study_geometry("squintly_n1")
            .unwrap()
            .pixels_per_degree();
        let m2 = super::study_geometry("squintly_m2")
            .unwrap()
            .pixels_per_degree();
        assert!((90.0..98.0).contains(&n1), "N1 ppd {n1}");
        assert!((45.0..49.0).contains(&m2), "M2 ppd {m2}");
        assert!(
            (n1 / m2 - 2.0).abs() < 0.05,
            "M2 must be ~half of N1: {n1} vs {m2}"
        );
        assert!(super::CpuCvvdpDisplayScorer::by_name("squintly_n1").is_ok());
        assert!(super::CpuCvvdpDisplayScorer::by_name("squintly_m2").is_ok());
    }

    use super::*;

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

    /// `--display-model standard_4k` must reproduce the default umbrella
    /// path bit-for-bit, so naming the default can never move a score.
    #[test]
    fn standard_4k_matches_default_umbrella_path_exactly() {
        let (r, d) = pair(96, 80);
        let named = CpuCvvdpDisplayScorer::by_name(DEFAULT_DISPLAY)
            .unwrap()
            .score(&r, &d)
            .unwrap();
        let default =
            crate::metrics::run_cpu_native_via_umbrella(zenmetrics_api::MetricKind::Cvvdp, &r, &d)
                .unwrap();
        assert_eq!(named.to_bits(), default.to_bits(), "{named} vs {default}");
    }

    /// The AIC display must actually change the viewing geometry: 37.84
    /// pixels per degree (AIC CTC v2.0), not the default 75.40.
    #[test]
    fn standard_fhd_is_the_aic_ctc_geometry_and_moves_the_score() {
        let fhd = CpuCvvdpDisplayScorer::by_name("standard_fhd").unwrap();
        assert!(
            (fhd.pixels_per_degree() - 37.8425).abs() < 1e-3,
            "{}",
            fhd.pixels_per_degree()
        );
        assert_eq!(fhd.display.y_peak, 200.0);
        let (r, d) = pair(96, 80);
        let mut fhd = fhd;
        let s_fhd = fhd.score(&r, &d).unwrap();
        let s_4k = CpuCvvdpDisplayScorer::by_name(DEFAULT_DISPLAY)
            .unwrap()
            .score(&r, &d)
            .unwrap();
        assert!(
            (s_fhd - s_4k).abs() > 1e-3,
            "display geometry must move the score: fhd {s_fhd} vs 4k {s_4k}"
        );
    }

    #[test]
    fn column_names_keep_default_plain_and_suffix_others() {
        let base = "cvvdp_cpu_imazen_v0_1_0";
        let s4k = CpuCvvdpDisplayScorer::by_name("standard_4k").unwrap();
        let fhd = CpuCvvdpDisplayScorer::by_name("standard_fhd").unwrap();
        assert_eq!(s4k.column_name(base), base);
        assert_eq!(
            fhd.column_name(base),
            "cvvdp_cpu_imazen_v0_1_0_standard_fhd"
        );
        assert!(CpuCvvdpDisplayScorer::by_name("no_such_display").is_err());
    }
}
