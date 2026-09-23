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
//! Imazen-only presets (`modern_oled_phone_indoor`, in
//! `display_models_imazen.json`) are deliberately EXCLUDED from the
//! conformance matrix because pycvvdp can't generate a reference golden
//! for a display name it doesn't know. They are pinned elsewhere
//! (cvvdp-gpu `presets.rs` tests) as self-consistency checks, not
//! against pycvvdp.
//!
//! The `65inch_hdr_pq_{1Knit,2Knit,4knit}` and `lg_oled_2026_hdr_pq`
//! presets used to be in that imazen-only group. pycvvdp v0.5.7 ships
//! them upstream with values identical to our vendored
//! `display_models.json`, so since 2026-09-22 they are in the matrix.
//! Their goldens need pycvvdp >= 0.5.7; the still-image model is
//! otherwise unchanged from v0.5.4
//! (benchmarks/cvvdp_aic_discrepancy_2026-09-22.md §4b).

/// One display model in the conformance matrix.
#[derive(Clone, Copy, Debug)]
pub struct ConformanceDisplay {
    /// pycvvdp display name (also the `by_name` registry key).
    pub upstream_name: &'static str,
    /// Human-readable role for the report.
    pub role: &'static str,
}

/// The conformance display selection. 13 models spanning common
/// (sRGB desktop / 1080p / phone), HDR (PQ + BT.2020, HLG + BT.2020,
/// dim-ambient linear), and niche (VR HMD with fov-diagonal geometry,
/// bright auto-brightness phone) configurations.
///
/// Acceptance gate (b) requires >= 8 display models; we ship 13.
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
        ConformanceDisplay {
            upstream_name: "65inch_hdr_pq_1Knit",
            role: "65\" HDR PQ + BT.2020 OLED, 1000-nit, 5 lux (pycvvdp >= 0.5.7)",
        },
        ConformanceDisplay {
            upstream_name: "65inch_hdr_pq_2Knit",
            role: "65\" HDR PQ + BT.2020 OLED, 2000-nit, 5 lux (pycvvdp >= 0.5.7)",
        },
        ConformanceDisplay {
            upstream_name: "65inch_hdr_pq_4knit",
            role: "65\" HDR PQ + BT.2020 OLED, 4000-nit, 5 lux (pycvvdp >= 0.5.7)",
        },
        ConformanceDisplay {
            upstream_name: "lg_oled_2026_hdr_pq",
            role: "LG G6 2026 HDR PQ, 3000-nit, min_luminance black (pycvvdp >= 0.5.7)",
        },
    ]
}
