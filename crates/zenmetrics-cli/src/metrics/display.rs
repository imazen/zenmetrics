//! cvvdp display selection for the CLI.
//!
//! cvvdp has no default display: every SDR cvvdp score names one. The CLI
//! accepts an official preset name ([`DisplayPreset`], the vendored
//! `display_models.json` registry) or one of the CLI-defined study displays
//! below, parses it ONCE at the edge (`--display-model`, a job's
//! `cvvdp@<display>`) into a typed [`CvvdpDisplay`], and passes that type
//! inward. The display's [`CvvdpDisplay::slug`] names the output column.

#[cfg(any(feature = "cpu-cvvdp", feature = "gpu-cvvdp"))]
pub use zenmetrics_api::cvvdp::params::{CustomDisplay, CvvdpDisplay, DisplayPreset};

/// Study displays the CLI defines on top of the official presets. Not in
/// upstream's `display_models.json`, so they are [`CustomDisplay`]s.
#[cfg(any(feature = "cpu-cvvdp", feature = "gpu-cvvdp"))]
pub const STUDY_DISPLAYS: &[&str] = &["squintly_n1", "squintly_m2"];

/// The Squintly chroma study's viewing geometries: an iPhone-class 6.1″
/// 1170×2532 panel (~457 ppi) held at 30 cm (`squintly_n1`, the fixed-1×
/// "normal" block, ~94 ppd), and the same panel under 2× integer zoom
/// (`squintly_m2`, ~47 ppd). Zoom makes each source pixel cover 2×2 display
/// pixels, which halves source pixels per degree; that is modelled as half
/// the viewing distance. Photometry is upstream's `standard_phone`
/// (500 cd/m², 250 lux ambient).
#[cfg(any(feature = "cpu-cvvdp", feature = "gpu-cvvdp"))]
fn study_display(name: &str) -> Option<CvvdpDisplay> {
    use zenmetrics_api::cvvdp::params::DisplayGeometry;
    let distance_m = match name {
        "squintly_n1" => 0.30,
        "squintly_m2" => 0.15,
        _ => return None,
    };
    let geometry = DisplayGeometry {
        resolution_w: 1170,
        resolution_h: 2532,
        distance_m,
        diagonal_inches: 6.1,
    };
    let custom = CustomDisplay::new(name, DisplayPreset::StandardPhone.model(), geometry)
        .expect("study slugs are valid and not preset names");
    Some(custom.into())
}

/// Parse a display name: an official preset or a CLI study display.
#[cfg(any(feature = "cpu-cvvdp", feature = "gpu-cvvdp"))]
pub fn parse_display(name: &str) -> Result<CvvdpDisplay, String> {
    if let Some(d) = study_display(name) {
        return Ok(d);
    }
    name.parse::<CvvdpDisplay>()
        .map_err(|e| format!("{e}; CLI study displays: {}", STUDY_DISPLAYS.join(", ")))
}

/// clap value parser for `--display-model`: `--help` lists every preset and
/// study display, and an unknown name is rejected with suggestions.
#[cfg(any(feature = "cpu-cvvdp", feature = "gpu-cvvdp"))]
pub fn display_value_parser() -> impl clap::builder::TypedValueParser<Value = CvvdpDisplay> {
    use clap::builder::TypedValueParser;
    let names = DisplayPreset::ALL
        .iter()
        .map(|p| p.name())
        .chain(STUDY_DISPLAYS.iter().copied());
    clap::builder::PossibleValuesParser::new(names)
        .map(|s| parse_display(&s).expect("clap only passes listed display names"))
}

/// Stand-in for builds without any cvvdp backend: no display value exists,
/// so every display-typed path is statically unreachable.
#[cfg(not(any(feature = "cpu-cvvdp", feature = "gpu-cvvdp")))]
#[derive(Debug, Clone, PartialEq)]
pub enum CvvdpDisplay {}

#[cfg(not(any(feature = "cpu-cvvdp", feature = "gpu-cvvdp")))]
impl CvvdpDisplay {
    /// Unreachable: no value of this type exists.
    pub fn slug(&self) -> &str {
        match *self {}
    }
}

#[cfg(not(any(feature = "cpu-cvvdp", feature = "gpu-cvvdp")))]
pub fn parse_display(_name: &str) -> Result<CvvdpDisplay, String> {
    Err("cvvdp is not built into this binary (enable cpu-cvvdp or gpu-cvvdp)".into())
}

#[cfg(not(any(feature = "cpu-cvvdp", feature = "gpu-cvvdp")))]
pub fn display_value_parser() -> impl clap::builder::TypedValueParser<Value = CvvdpDisplay> {
    use clap::builder::TypedValueParser;
    clap::builder::NonEmptyStringValueParser::new().try_map(|s| parse_display(&s))
}

#[cfg(all(test, any(feature = "cpu-cvvdp", feature = "gpu-cvvdp")))]
mod tests {
    use super::*;

    #[test]
    fn presets_and_study_displays_parse() {
        assert_eq!(
            parse_display("standard_4k").unwrap(),
            CvvdpDisplay::Preset(DisplayPreset::Standard4k)
        );
        for name in STUDY_DISPLAYS {
            let d = parse_display(name).unwrap();
            assert_eq!(d.slug(), *name);
            assert!(matches!(d, CvvdpDisplay::Custom(_)));
        }
        let n1 = parse_display("squintly_n1")
            .unwrap()
            .geometry()
            .pixels_per_degree();
        let m2 = parse_display("squintly_m2")
            .unwrap()
            .geometry()
            .pixels_per_degree();
        assert!((90.0..98.0).contains(&n1), "N1 ppd {n1}");
        assert!(
            (n1 / m2 - 2.0).abs() < 0.05,
            "M2 must be ~half of N1: {n1} vs {m2}"
        );
        let err = parse_display("no_such_display").unwrap_err();
        assert!(
            err.contains("standard_4k") && err.contains("squintly_n1"),
            "{err}"
        );
    }
}
