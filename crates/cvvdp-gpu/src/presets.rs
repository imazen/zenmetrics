//! Preset display registry — every named entry from upstream's
//! `display_models.json` is loadable here via
//! [`DisplayModel::by_name`] / [`DisplayGeometry::by_name`].
//!
//! Two JSON files are vendored under `crates/cvvdp-gpu/data/`:
//!
//! - `display_models.json` — named device profiles (resolution,
//!   viewing distance, peak luminance, ambient, EOTF + primaries
//!   selector via the optional `colorspace` field).
//! - `color_spaces.json` — primaries + EOTF lookup keyed by the
//!   string a preset's `colorspace` field references
//!   (`BT.2020-PQ`, `Display P3 Apple`, …).
//!
//! Both files are sourced verbatim from ColorVideoVDP's
//! `pycvvdp/vvdp_data/` directory (commit fetched 2026-05-25,
//! upstream is MIT-licensed; full license text vendored as
//! `data/UPSTREAM_LICENSE_MIT.txt`). The registry mirrors the
//! lookup `pycvvdp` performs in
//! `vvdp_display_photometry.load(display_name, config_paths)`.
//!
//! Presets with a viewing-mode that's not yet ported to
//! [`crate::params::DisplayGeometry`] (FOV-diagonal only, no
//! `diagonal_size_inches` + `viewing_distance_meters` pair) are
//! still loadable for their [`DisplayModel`] fields, but
//! `geometry_by_name` returns `None` for them — see [`PRESETS`]
//! for the full enumeration.
//!
//! # Examples
//!
//! ```
//! use cvvdp_gpu::params::{DisplayModel, Eotf, Primaries};
//!
//! // STANDARD_4K loaded from the registry matches the const.
//! let s = DisplayModel::by_name("standard_4k").unwrap();
//! assert_eq!(s.y_peak, DisplayModel::STANDARD_4K.y_peak);
//! assert_eq!(s.eotf, Eotf::Srgb);
//! assert_eq!(s.primaries, Primaries::Bt709);
//!
//! // The HDR PQ display picks up BT.2020 + PQ from color_spaces.json.
//! let h = DisplayModel::by_name("standard_hdr_pq").unwrap();
//! assert_eq!(h.y_peak, 1500.0);
//! assert_eq!(h.eotf, Eotf::Pq);
//! assert_eq!(h.primaries, Primaries::Bt2020);
//! ```

use crate::params::{DisplayGeometry, DisplayModel, Eotf, Primaries};
use std::collections::BTreeMap;
use std::sync::OnceLock;

const DISPLAY_MODELS_JSON: &str = include_str!("../data/display_models.json");
const COLOR_SPACES_JSON: &str = include_str!("../data/color_spaces.json");

/// Sorted list of every preset name in the registry. Stable
/// across releases — drift here signals upstream added or
/// renamed a preset, which we should pick up explicitly.
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::presets::list_presets;
/// let all = list_presets();
/// assert!(all.contains(&"standard_4k"));
/// assert!(all.contains(&"standard_hdr_pq"));
/// // No duplicates.
/// let mut sorted: Vec<&str> = all.to_vec();
/// sorted.sort_unstable();
/// sorted.dedup();
/// assert_eq!(sorted.len(), all.len());
/// ```
#[must_use]
pub fn list_presets() -> &'static [&'static str] {
    PRESET_NAMES.get_or_init(|| {
        let mut names: Vec<&'static str> =
            registry().display.keys().map(|s| s.as_str()).collect();
        // Leak the strings to get 'static — the JSON itself is
        // 'static so the keys outlive any normal program lifetime,
        // but the BTreeMap's owned String keys aren't &'static.
        // Allocate one Vec<&'static str> by leaking the names so
        // we hand back stable slices.
        let leaked: Vec<&'static str> =
            names.drain(..).map(|s| Box::leak(s.to_string().into_boxed_str()) as &'static str).collect();
        leaked.into_boxed_slice()
    })
}

static PRESET_NAMES: OnceLock<Box<[&'static str]>> = OnceLock::new();

impl DisplayModel {
    /// Load a named preset from the vendored upstream
    /// `display_models.json`. Returns `None` if the name doesn't
    /// match any preset.
    ///
    /// For the canonical sRGB display (`"standard_4k"`) this is
    /// bit-identical to [`DisplayModel::STANDARD_4K`].
    ///
    /// # Examples
    ///
    /// ```
    /// use cvvdp_gpu::params::DisplayModel;
    ///
    /// let s = DisplayModel::by_name("standard_4k").unwrap();
    /// // bit-identical to STANDARD_4K
    /// let k = DisplayModel::STANDARD_4K;
    /// assert_eq!(s.y_peak.to_bits(), k.y_peak.to_bits());
    /// assert_eq!(s.y_black.to_bits(), k.y_black.to_bits());
    ///
    /// // Unknown preset returns None.
    /// assert!(DisplayModel::by_name("this_does_not_exist").is_none());
    /// ```
    #[must_use]
    pub fn by_name(name: &str) -> Option<Self> {
        registry().display.get(name).map(|p| p.display)
    }
}

impl DisplayGeometry {
    /// Load a named preset's display geometry from the vendored
    /// upstream `display_models.json`. Returns `None` if the
    /// preset doesn't exist OR doesn't expose
    /// `diagonal_size_inches + viewing_distance_meters` (e.g.,
    /// the FOV-only `standard_hmd` entry).
    ///
    /// # Examples
    ///
    /// ```
    /// use cvvdp_gpu::params::DisplayGeometry;
    ///
    /// let g = DisplayGeometry::by_name("standard_4k").unwrap();
    /// assert_eq!(g.resolution_w, 3840);
    /// assert_eq!(g.resolution_h, 2160);
    ///
    /// // FOV-only presets return None for geometry today.
    /// assert!(DisplayGeometry::by_name("standard_hmd").is_none());
    /// // Unknown preset is also None.
    /// assert!(DisplayGeometry::by_name("nope").is_none());
    /// ```
    #[must_use]
    pub fn by_name(name: &str) -> Option<Self> {
        registry().display.get(name).and_then(|p| p.geometry)
    }
}

#[derive(Debug, Clone, Copy)]
struct Preset {
    display: DisplayModel,
    geometry: Option<DisplayGeometry>,
}

struct Registry {
    display: BTreeMap<String, Preset>,
}

fn registry() -> &'static Registry {
    static REG: OnceLock<Registry> = OnceLock::new();
    REG.get_or_init(load_registry)
}

fn load_registry() -> Registry {
    let colors: serde_json::Value =
        serde_json::from_str(COLOR_SPACES_JSON).expect("vendored color_spaces.json must parse");
    let displays: serde_json::Value =
        serde_json::from_str(DISPLAY_MODELS_JSON).expect("vendored display_models.json must parse");

    let mut out = BTreeMap::new();
    let displays_obj = displays
        .as_object()
        .expect("display_models.json root must be an object");
    for (name, value) in displays_obj {
        let preset = parse_preset(name, value, &colors)
            .unwrap_or_else(|err| panic!("preset {name:?} failed to load: {err}"));
        out.insert(name.clone(), preset);
    }
    Registry { display: out }
}

fn parse_preset(name: &str, value: &serde_json::Value, colors: &serde_json::Value) -> Result<Preset, String> {
    let obj = value
        .as_object()
        .ok_or_else(|| format!("preset {name} is not an object"))?;

    let y_peak = num_field(obj, "max_luminance")?;
    let (eotf, primaries) = resolve_colorspace(obj.get("colorspace"), colors)?;

    let contrast = if let Some(min_lum) = obj.get("min_luminance").and_then(serde_json::Value::as_f64) {
        if min_lum > 0.0 {
            (y_peak as f64 / min_lum) as f32
        } else {
            500.0
        }
    } else if let Some(c) = obj.get("contrast").and_then(serde_json::Value::as_f64) {
        c as f32
    } else {
        // Matches pycvvdp default for displays that omit both
        // min_luminance and contrast (e.g. some legacy entries).
        500.0
    };

    let e_ambient_lux = obj
        .get("E_ambient")
        .and_then(serde_json::Value::as_f64)
        .map(|v| v as f32)
        .unwrap_or(0.0);

    // upstream default per vvdp_display_photo_eotf.__init__
    let k_refl = obj
        .get("k_refl")
        .and_then(serde_json::Value::as_f64)
        .map(|v| v as f32)
        .unwrap_or(0.005);

    let display = DisplayModel::new(y_peak, contrast, e_ambient_lux, k_refl, eotf, primaries);
    let geometry = parse_geometry(obj);

    Ok(Preset { display, geometry })
}

fn parse_geometry(obj: &serde_json::Map<String, serde_json::Value>) -> Option<DisplayGeometry> {
    let resolution = obj.get("resolution")?;
    let arr = resolution.as_array()?;
    if arr.len() != 2 {
        return None;
    }
    let w = arr[0].as_u64()? as u32;
    let h = arr[1].as_u64()? as u32;

    let diagonal_inches = obj
        .get("diagonal_size_inches")
        .and_then(serde_json::Value::as_f64)
        .map(|v| v as f32)?;

    let distance_m = if let Some(m) = obj
        .get("viewing_distance_meters")
        .and_then(serde_json::Value::as_f64)
    {
        m as f32
    } else if let Some(inches) = obj
        .get("viewing_distance_inches")
        .and_then(serde_json::Value::as_f64)
    {
        (inches * 0.0254) as f32
    } else {
        return None;
    };

    Some(DisplayGeometry {
        resolution_w: w,
        resolution_h: h,
        distance_m,
        diagonal_inches,
    })
}

fn num_field(obj: &serde_json::Map<String, serde_json::Value>, key: &str) -> Result<f32, String> {
    obj.get(key)
        .and_then(serde_json::Value::as_f64)
        .map(|v| v as f32)
        .ok_or_else(|| format!("missing or non-numeric field {key:?}"))
}

fn resolve_colorspace(
    cs: Option<&serde_json::Value>,
    colors: &serde_json::Value,
) -> Result<(Eotf, Primaries), String> {
    // Default: sRGB / BT.709 — used by every preset that omits
    // the `colorspace` field.
    let Some(name_val) = cs else {
        return Ok((Eotf::Srgb, Primaries::Bt709));
    };
    let name = name_val
        .as_str()
        .ok_or_else(|| "colorspace must be a string".to_string())?;

    let colors_obj = colors
        .as_object()
        .ok_or_else(|| "color_spaces.json root must be an object".to_string())?;
    let entry = colors_obj
        .get(name)
        .ok_or_else(|| format!("color space {name:?} not in color_spaces.json"))?;
    let entry_obj = entry
        .as_object()
        .ok_or_else(|| format!("color space {name:?} entry is not an object"))?;

    let eotf_str = entry_obj
        .get("EOTF")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| format!("color space {name:?} missing EOTF"))?;
    let eotf = match eotf_str {
        "sRGB" => Eotf::Srgb,
        "PQ" => Eotf::Pq,
        "HLG" => Eotf::Hlg,
        "linear" => Eotf::Linear,
        // upstream encodes numeric gammas as e.g. "2.2", "1.8"
        other => {
            let g: f32 = other
                .parse()
                .map_err(|_| format!("unknown EOTF {other:?} for color space {name:?}"))?;
            Eotf::Gamma(g)
        }
    };

    let primaries = match name {
        "sRGB" | "BT.709" | "BT.709-linear" => Primaries::Bt709,
        "BT.2020-PQ" | "BT.2020-HLG" | "BT.2020-linear" => Primaries::Bt2020,
        "Display P3 Apple" => Primaries::DisplayP3,
        // No primaries info (just luminance) → fall back to BT.709.
        // The metric will still produce a sensible output if the
        // EOTF is linear; chroma will be approximate.
        "luminance" => Primaries::Bt709,
        // Other named entries in color_spaces.json (Adobe RGB,
        // Apple RGB, Best RGB, ...) ship with their own RGB→XYZ
        // matrices but none of the bundled display presets use
        // them. Fall back to BT.709 with a warning-via-eotf-only
        // fidelity. A future tick can wire per-primaries
        // matrices for these by reading the RGB2X/Y/Z rows.
        _ => Primaries::Bt709,
    };

    Ok((eotf, primaries))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_4k_round_trips() {
        let s = DisplayModel::by_name("standard_4k").unwrap();
        let k = DisplayModel::STANDARD_4K;
        assert_eq!(s.y_peak.to_bits(), k.y_peak.to_bits());
        assert_eq!(s.y_black.to_bits(), k.y_black.to_bits());
        assert!((s.y_refl - k.y_refl).abs() < 1e-6);
        assert_eq!(s.eotf, k.eotf);
        assert_eq!(s.primaries, k.primaries);
    }

    #[test]
    fn standard_hdr_pq_routes_bt2020_pq() {
        let h = DisplayModel::by_name("standard_hdr_pq").unwrap();
        assert_eq!(h.eotf, Eotf::Pq);
        assert_eq!(h.primaries, Primaries::Bt2020);
        assert_eq!(h.y_peak, 1500.0);
    }

    #[test]
    fn standard_hdr_hlg_routes_bt2020_hlg() {
        let h = DisplayModel::by_name("standard_hdr_hlg").unwrap();
        assert_eq!(h.eotf, Eotf::Hlg);
        assert_eq!(h.primaries, Primaries::Bt2020);
    }

    #[test]
    fn standard_hdr_linear_routes_bt709_linear() {
        let h = DisplayModel::by_name("standard_hdr_linear").unwrap();
        assert_eq!(h.eotf, Eotf::Linear);
        assert_eq!(h.primaries, Primaries::Bt709);
        assert_eq!(h.y_peak, 1500.0);
    }

    #[test]
    fn unknown_returns_none() {
        assert!(DisplayModel::by_name("not_a_preset").is_none());
        assert!(DisplayGeometry::by_name("not_a_preset").is_none());
    }

    #[test]
    fn all_documented_presets_load() {
        for name in [
            "standard_4k",
            "standard_hdr_pq",
            "standard_hdr_hlg",
            "standard_hdr_linear",
            "standard_hdr_linear_dark",
            "standard_hdr_linear_zoom",
            "standard_fhd",
            "standard_hmd",
            "standard_phone",
            "sdr_4k_30",
            "sdr_fhd_24",
            "htc_vive_pro",
            "iphone_12_pro",
            "iphone_14_pro",
            "iphone_14_pro_vert",
            "iphone_14_pro_hdr",
            "iphone_14_pro_hdr_vert",
            "ipad_pro_12_9",
            "macbook_pro_16",
            "lg_oled_2017_sdr",
            "lg_oled_2017_hdr",
            "eizo_CG3146",
            "65inch_hdr_pq_4knit",
            "65inch_hdr_pq_2Knit",
            "65inch_hdr_pq_1Knit",
            "lg_oled_2026_hdr_pq",
        ] {
            let d = DisplayModel::by_name(name)
                .unwrap_or_else(|| panic!("preset {name} should load"));
            assert!(d.y_peak > 0.0, "{name} y_peak");
            assert!(d.y_black >= 0.0, "{name} y_black");
            assert!(d.k_refl > 0.0, "{name} k_refl");
        }
    }

    #[test]
    fn iphone_14_pro_hdr_routes_bt2020_hlg() {
        let h = DisplayModel::by_name("iphone_14_pro_hdr").unwrap();
        assert_eq!(h.eotf, Eotf::Hlg);
        assert_eq!(h.primaries, Primaries::Bt2020);
        // Geometry uses viewing_distance_inches — verify the
        // inches→meters conversion lands.
        let g = DisplayGeometry::by_name("iphone_14_pro_hdr").unwrap();
        assert!((g.distance_m - 0.508).abs() < 1e-3); // 20 inches
    }

    #[test]
    fn list_presets_contains_canonical() {
        let names = list_presets();
        assert!(names.contains(&"standard_4k"));
        assert!(names.contains(&"standard_hdr_pq"));
        assert!(!names.is_empty());
    }
}
