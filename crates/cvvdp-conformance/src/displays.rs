//! Display-model selection for the conformance matrix.
//!
//! Every entry is an UPSTREAM pycvvdp display name (present in
//! `pycvvdp/vvdp_data/display_models.json`) that ALSO resolves in our
//! `DisplayModel::by_name` / `DisplayGeometry::by_name` registry. This
//! is the apples-to-apples contract: pycvvdp is invoked with
//! `display_name=<upstream_name>` and our impls are configured via
//! `by_name(<upstream_name>)`, so all three scorers see the same
//! photometric + geometric display model.
//!
//! Imazen-only presets (e.g. `modern_oled_phone_indoor`,
//! `65inch_hdr_pq_*`, `lg_oled_2026_hdr_pq`) are deliberately EXCLUDED
//! from the conformance matrix because pycvvdp can't generate a
//! reference golden for a display name it doesn't know. They are
//! pinned elsewhere (cvvdp-gpu `presets.rs` tests) as self-consistency
//! checks, not against pycvvdp.

/// One display model in the conformance matrix.
#[derive(Clone, Copy, Debug)]
pub struct ConformanceDisplay {
    /// pycvvdp display name (also the `by_name` registry key).
    pub upstream_name: &'static str,
    /// Human-readable role for the report.
    pub role: &'static str,
}

/// The conformance display selection. 9 models spanning common
/// (sRGB desktop / 1080p / phone), HDR (PQ + BT.2020, HLG + BT.2020,
/// dim-ambient linear), and niche (VR HMD with fov-diagonal geometry,
/// bright auto-brightness phone) configurations.
///
/// Acceptance gate (b) requires >= 8 display models; we ship 9.
#[must_use]
pub fn conformance_displays() -> &'static [ConformanceDisplay] {
    &[
        ConformanceDisplay {
            upstream_name: "standard_4k",
            role: "sRGB/BT.709 desktop 4K (canonical reference)",
        },
        ConformanceDisplay {
            upstream_name: "sdr_4k_30",
            role: "standard SDR desktop 100-nit",
        },
        ConformanceDisplay {
            upstream_name: "standard_fhd",
            role: "1080p SDR desktop 200-nit",
        },
        ConformanceDisplay {
            upstream_name: "standard_phone",
            role: "SDR phone 500-nit",
        },
        ConformanceDisplay {
            upstream_name: "iphone_14_pro",
            role: "bright phone, 1025-nit auto-brightness (sRGB)",
        },
        ConformanceDisplay {
            upstream_name: "standard_hdr_pq",
            role: "HDR PQ + BT.2020 wide-gamut, 1500-nit",
        },
        ConformanceDisplay {
            upstream_name: "standard_hdr_hlg",
            role: "HDR HLG + BT.2020 wide-gamut",
        },
        ConformanceDisplay {
            upstream_name: "standard_hdr_linear_dark",
            role: "HDR linear EOTF, dim ambient (dark-adapted)",
        },
        ConformanceDisplay {
            upstream_name: "htc_vive_pro",
            role: "VR HMD, fov-diagonal geometry, 133-nit",
        },
    ]
}
