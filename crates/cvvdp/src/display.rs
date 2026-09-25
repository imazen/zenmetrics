//! Typed cvvdp viewing conditions.
//!
//! cvvdp is display-aware: peak luminance, ambient light and pixels per
//! degree all move the JOD. [`CvvdpDisplay`] names the viewing condition a
//! score was computed under, either as one of the official
//! [`DisplayPreset`]s (every entry of the vendored `display_models.json`
//! plus the Imazen-added `display_models_imazen.json`) or as a
//! [`CustomDisplay`] carrying its own photometry, geometry and slug.
//!
//! The preset enum and the vendored JSON are held in sync by tests in
//! both directions (every variant resolves in the registry, every registry
//! entry has a variant), so a vendored-JSON bump that adds a preset fails
//! CI rather than drifting silently.
//!
//! The [`CvvdpDisplay::slug`] is the stable string form: the preset name
//! for presets, the caller's validated slug for custom displays. Callers
//! that persist scores name their output columns with it.
//!
//! # Examples
//!
//! ```
//! use cvvdp::display::{CustomDisplay, CvvdpDisplay, DisplayPreset};
//! use cvvdp::params::{DisplayGeometry, DisplayModel};
//!
//! let fhd: CvvdpDisplay = "standard_fhd".parse().unwrap();
//! assert_eq!(fhd, CvvdpDisplay::Preset(DisplayPreset::StandardFhd));
//! assert_eq!(fhd.slug(), "standard_fhd");
//! assert!((fhd.geometry().pixels_per_degree() - 37.84).abs() < 0.01);
//!
//! // A custom display carries its own values; its slug may not shadow a preset.
//! let custom = CustomDisplay::new(
//!     "lab_monitor_a",
//!     DisplayModel::STANDARD_4K,
//!     DisplayGeometry::STANDARD_4K,
//! )
//! .unwrap();
//! assert_eq!(CvvdpDisplay::from(custom).slug(), "lab_monitor_a");
//! assert!(CustomDisplay::new("standard_4k", DisplayModel::STANDARD_4K, DisplayGeometry::STANDARD_4K).is_err());
//! ```

use alloc::string::{String, ToString};
use core::fmt;
use core::str::FromStr;

use crate::params::{DisplayGeometry, DisplayModel};

macro_rules! display_presets {
    ($($(#[$doc:meta])* $variant:ident => $name:literal,)*) => {
        /// An official cvvdp display preset: every entry of the vendored
        /// upstream `display_models.json` plus the Imazen-added
        /// `display_models_imazen.json`. Photometry and geometry come from
        /// that registry ([`DisplayModel::by_name`] /
        /// [`DisplayGeometry::by_name`]).
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        #[non_exhaustive]
        pub enum DisplayPreset {
            $($(#[$doc])* $variant,)*
        }

        impl DisplayPreset {
            /// Every preset, in registry-name order.
            pub const ALL: &'static [DisplayPreset] = &[$(DisplayPreset::$variant,)*];

            /// The registry name (`"standard_4k"`, …) — the preset's slug.
            #[must_use]
            pub const fn name(self) -> &'static str {
                match self {
                    $(DisplayPreset::$variant => $name,)*
                }
            }
        }
    };
}

display_presets! {
    /// 65″ HDR PQ panel, 1000 cd/m² peak.
    Inch65HdrPq1Knit => "65inch_hdr_pq_1Knit",
    /// 65″ HDR PQ panel, 2000 cd/m² peak.
    Inch65HdrPq2Knit => "65inch_hdr_pq_2Knit",
    /// 65″ HDR PQ panel, 4000 cd/m² peak.
    Inch65HdrPq4Knit => "65inch_hdr_pq_4knit",
    /// EIZO ColorEdge CG3146 reference monitor.
    EizoCg3146 => "eizo_CG3146",
    /// HTC Vive Pro head-mounted display.
    HtcVivePro => "htc_vive_pro",
    /// iPad Pro 12.9″.
    IpadPro12_9 => "ipad_pro_12_9",
    /// iPhone 12 Pro.
    Iphone12Pro => "iphone_12_pro",
    /// iPhone 14 Pro, landscape SDR.
    Iphone14Pro => "iphone_14_pro",
    /// iPhone 14 Pro, landscape HDR.
    Iphone14ProHdr => "iphone_14_pro_hdr",
    /// iPhone 14 Pro, portrait HDR.
    Iphone14ProHdrVert => "iphone_14_pro_hdr_vert",
    /// iPhone 14 Pro, portrait SDR.
    Iphone14ProVert => "iphone_14_pro_vert",
    /// LG OLED (2017), HDR.
    LgOled2017Hdr => "lg_oled_2017_hdr",
    /// LG OLED (2017), SDR.
    LgOled2017Sdr => "lg_oled_2017_sdr",
    /// LG OLED (2026), HDR PQ.
    LgOled2026HdrPq => "lg_oled_2026_hdr_pq",
    /// MacBook Pro 16″.
    MacbookPro16 => "macbook_pro_16",
    /// Imazen: modern OLED phone, indoor SDR viewing (not upstream).
    ModernOledPhoneIndoor => "modern_oled_phone_indoor",
    /// 30″ SDR 4K monitor.
    Sdr4k30 => "sdr_4k_30",
    /// 24″ SDR Full HD monitor.
    SdrFhd24 => "sdr_fhd_24",
    /// Upstream ColorVideoVDP's default: 4K desktop monitor, 200 cd/m²,
    /// 75.4 pixels per degree. Every historical zenmetrics cvvdp score.
    Standard4k => "standard_4k",
    /// Full HD desktop monitor, 200 cd/m², 37.84 pixels per degree — the
    /// display the JPEG AIC evaluation uses for its CVVDP anchor.
    StandardFhd => "standard_fhd",
    /// Standard HDR display, HLG.
    StandardHdrHlg => "standard_hdr_hlg",
    /// Standard HDR display, linear-light input.
    StandardHdrLinear => "standard_hdr_linear",
    /// Standard HDR display, linear-light input, dark room.
    StandardHdrLinearDark => "standard_hdr_linear_dark",
    /// Standard HDR display, linear-light input, zoomed.
    StandardHdrLinearZoom => "standard_hdr_linear_zoom",
    /// Standard HDR display, PQ.
    StandardHdrPq => "standard_hdr_pq",
    /// Standard head-mounted display.
    StandardHmd => "standard_hmd",
    /// Standard phone.
    StandardPhone => "standard_phone",
}

impl DisplayPreset {
    /// The preset's photometric model.
    #[must_use]
    pub fn model(self) -> DisplayModel {
        DisplayModel::by_name(self.name())
            .expect("DisplayPreset variants are sync-tested against the vendored registry")
    }

    /// The preset's viewing geometry.
    #[must_use]
    pub fn geometry(self) -> DisplayGeometry {
        DisplayGeometry::by_name(self.name())
            .expect("DisplayPreset variants are sync-tested against the vendored registry")
    }
}

impl fmt::Display for DisplayPreset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for DisplayPreset {
    type Err = DisplayError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .iter()
            .copied()
            .find(|p| p.name() == s)
            .ok_or_else(|| DisplayError::UnknownPreset(s.to_string()))
    }
}

/// A viewing condition that is not an official preset: explicit photometry
/// and geometry under a caller-chosen slug.
#[derive(Debug, Clone, PartialEq)]
pub struct CustomDisplay {
    slug: String,
    model: DisplayModel,
    geometry: DisplayGeometry,
}

impl CustomDisplay {
    /// Build a custom display. The slug is 1–64 ASCII letters, digits or
    /// `_`, and must not equal a [`DisplayPreset`] name — a custom display
    /// named like a preset would label its scores as that preset's.
    pub fn new(
        slug: &str,
        model: DisplayModel,
        geometry: DisplayGeometry,
    ) -> Result<Self, DisplayError> {
        let valid = !slug.is_empty()
            && slug.len() <= 64
            && slug.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_');
        if !valid {
            return Err(DisplayError::InvalidSlug(slug.to_string()));
        }
        if DisplayPreset::from_str(slug).is_ok() {
            return Err(DisplayError::SlugIsPreset(slug.to_string()));
        }
        Ok(Self {
            slug: slug.to_string(),
            model,
            geometry,
        })
    }

    /// The custom display's slug.
    #[must_use]
    pub fn slug(&self) -> &str {
        &self.slug
    }
}

/// A cvvdp viewing condition: an official preset or a custom display.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum CvvdpDisplay {
    /// An official preset from the vendored registry.
    Preset(DisplayPreset),
    /// Explicit photometry and geometry under a caller-chosen slug.
    Custom(CustomDisplay),
}

impl CvvdpDisplay {
    /// Stable string form: the preset name, or the custom slug.
    #[must_use]
    pub fn slug(&self) -> &str {
        match self {
            CvvdpDisplay::Preset(p) => p.name(),
            CvvdpDisplay::Custom(c) => c.slug(),
        }
    }

    /// Photometric model.
    #[must_use]
    pub fn model(&self) -> DisplayModel {
        match self {
            CvvdpDisplay::Preset(p) => p.model(),
            CvvdpDisplay::Custom(c) => c.model,
        }
    }

    /// Viewing geometry.
    #[must_use]
    pub fn geometry(&self) -> DisplayGeometry {
        match self {
            CvvdpDisplay::Preset(p) => p.geometry(),
            CvvdpDisplay::Custom(c) => c.geometry,
        }
    }
}

impl From<DisplayPreset> for CvvdpDisplay {
    fn from(p: DisplayPreset) -> Self {
        CvvdpDisplay::Preset(p)
    }
}

impl From<CustomDisplay> for CvvdpDisplay {
    fn from(c: CustomDisplay) -> Self {
        CvvdpDisplay::Custom(c)
    }
}

impl fmt::Display for CvvdpDisplay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.slug())
    }
}

/// Parses preset names only. A custom display has no string form: it
/// carries values a name cannot.
impl FromStr for CvvdpDisplay {
    type Err = DisplayError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        DisplayPreset::from_str(s).map(CvvdpDisplay::Preset)
    }
}

/// Display-selection errors.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DisplayError {
    /// Not a [`DisplayPreset`] name.
    UnknownPreset(String),
    /// A custom slug outside `[A-Za-z0-9_]{1,64}`.
    InvalidSlug(String),
    /// A custom slug equal to a preset name.
    SlugIsPreset(String),
}

impl fmt::Display for DisplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DisplayError::UnknownPreset(s) => {
                write!(f, "unknown cvvdp display {s:?}; presets: ")?;
                for (i, p) in DisplayPreset::ALL.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    f.write_str(p.name())?;
                }
                Ok(())
            }
            DisplayError::InvalidSlug(s) => write!(
                f,
                "invalid custom display slug {s:?} (1-64 ASCII letters, digits or '_')"
            ),
            DisplayError::SlugIsPreset(s) => write!(
                f,
                "custom display slug {s:?} is a preset name; pick a distinct slug"
            ),
        }
    }
}

impl std::error::Error for DisplayError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::presets::list_presets;

    /// Every variant resolves to BOTH halves in the vendored registry.
    #[test]
    fn every_variant_resolves_in_the_registry() {
        for p in DisplayPreset::ALL {
            assert!(
                DisplayModel::by_name(p.name()).is_some(),
                "{p}: no photometry"
            );
            assert!(
                DisplayGeometry::by_name(p.name()).is_some(),
                "{p}: no geometry"
            );
        }
    }

    /// Every registry entry has a variant — a vendored-JSON bump that adds
    /// a preset must add the variant too.
    #[test]
    fn every_registry_entry_has_a_variant() {
        for name in list_presets() {
            assert!(
                DisplayPreset::from_str(name).is_ok(),
                "registry preset {name:?} has no DisplayPreset variant"
            );
        }
        assert_eq!(DisplayPreset::ALL.len(), list_presets().len());
    }

    #[test]
    fn names_round_trip_and_are_unique() {
        let mut seen = std::collections::BTreeSet::new();
        for p in DisplayPreset::ALL {
            assert!(seen.insert(p.name()), "duplicate name {}", p.name());
            assert_eq!(DisplayPreset::from_str(p.name()).unwrap(), *p);
            assert_eq!(p.to_string(), p.name());
        }
    }

    /// `Standard4k` is bit-identical to the crate's `STANDARD_4K` consts —
    /// the historical default.
    #[test]
    fn standard_4k_is_the_historical_default() {
        let d = CvvdpDisplay::from(DisplayPreset::Standard4k);
        assert_eq!(d.model(), DisplayModel::STANDARD_4K);
        assert_eq!(d.geometry(), DisplayGeometry::STANDARD_4K);
    }

    #[test]
    fn custom_slugs_are_validated() {
        let (m, g) = (DisplayModel::STANDARD_4K, DisplayGeometry::STANDARD_4K);
        assert!(CustomDisplay::new("squintly_n1", m, g).is_ok());
        for bad in ["", "has space", "dash-ed", "é", &"x".repeat(65)] {
            assert!(
                matches!(
                    CustomDisplay::new(bad, m, g),
                    Err(DisplayError::InvalidSlug(_))
                ),
                "{bad:?}"
            );
        }
        assert!(matches!(
            CustomDisplay::new("standard_fhd", m, g),
            Err(DisplayError::SlugIsPreset(_))
        ));
        assert!(matches!(
            "squintly_n1".parse::<CvvdpDisplay>(),
            Err(DisplayError::UnknownPreset(_))
        ));
    }
}
