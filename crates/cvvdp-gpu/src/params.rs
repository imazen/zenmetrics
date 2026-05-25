//! ColorVideoVDP model parameters.
//!
//! Mirrors the JSON parameter set the Python reference loads on init
//! (display model, CSF coefficients, masking exponents, pooling
//! exponents, JOD mapping). Concrete numbers are filled in once the
//! reference version is pinned and the JSON is vendored under
//! `crates/cvvdp-gpu/data/`.
//!
//! Keeping this in a single struct so:
//! - tests can construct a `CvvdpParams` from the reference JSON and
//!   compare each Rust kernel's output against Python numbers using
//!   the same coefficients;
//! - alternate parameter sets (e.g. HDR display models) can be plugged
//!   in without changing the pipeline shape.
//!
//! All fields are stored as f32 — cvvdp's published parameters are
//! single-precision; matching the Python reference's `.float()` calls.

/// Electro-optical transfer function — how display-encoded pixel
/// values map to relative-linear or absolute luminance before the
/// peak/black/ambient scaling step.
///
/// Mirrors the EOTF branches in pycvvdp's
/// `vvdp_display_photo_eotf.forward`:
///
/// - [`Eotf::Srgb`] — the historical default and the one every v1
///   parity golden was captured against. Inputs in 0..1 normalized;
///   output normalized in 0..1 (multiplied by `y_peak - y_black`
///   downstream).
/// - [`Eotf::Pq`] — SMPTE ST 2084 PQ. Absolute: `pq2lin(V)` returns
///   cd/m² directly; the display scaling step does not multiply by
///   `(y_peak - y_black)`. Output is clipped to `[0.005, y_peak]`
///   then offset by `y_black + y_refl`. Used for `BT.2020-PQ`
///   presets (`standard_hdr_pq`, the 65" HDR OLED variants).
/// - [`Eotf::Hlg`] — Rec. BT.2100 Hybrid Log-Gamma. Inverse OETF
///   then the system OOTF (gamma = 1.2 boosted slightly for displays
///   above 1000 cd/m²). Normalized in 0..1, scaled by
///   `(y_peak - y_black)` downstream.
/// - [`Eotf::Linear`] — input is already linear-light. Clipped to
///   `[max(0.005, y_black), y_peak]` then offset by `y_refl`. Used
///   for `BT.709-linear` and `luminance` color spaces.
/// - [`Eotf::Bt1886`] — Rec. BT.1886 display gamma (fixed 2.4 with
///   black-level lift). Not directly used by any upstream preset
///   today (those use numeric `gamma` like `"2.2"` or `"1.8"`) but
///   wired in for completeness; calling code that needs an arbitrary
///   power-law gamma should reach for [`Eotf::Gamma`] instead.
/// - [`Eotf::Gamma`] — generic power-law gamma. Used for upstream
///   color spaces whose EOTF field is a numeric string ("1.8" for
///   Apple RGB, "2.2" for Adobe RGB / NTSC / Wide Gamut RGB, etc.).
///
/// All variants produce cd/m² output via [`Eotf::forward`]; that's
/// the only entry point callers should use.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Eotf {
    /// sRGB / BT.709 EOTF. Default — what every v1 parity golden
    /// was captured against.
    #[default]
    Srgb,
    /// SMPTE ST 2084 PQ (absolute, up to 10000 cd/m²).
    Pq,
    /// Rec. BT.2100 Hybrid Log-Gamma (relative + system OOTF).
    Hlg,
    /// Linear-light (input already in 0..1 normalized, or already
    /// in cd/m² for HDR linear sources — same code path either way
    /// since the multiply-by-y_peak step is gated on the EOTF being
    /// relative).
    Linear,
    /// Rec. BT.1886 display gamma 2.4 with black-level lift. The
    /// reference EOTF for studio video grading; rarely used for
    /// authored content because the lift term depends on the
    /// display's `y_black`.
    Bt1886,
    /// Generic power-law gamma `L = (Y_peak - Y_black) * V^gamma +
    /// Y_black + Y_refl`. Used for Adobe RGB (2.2), Apple RGB
    /// (1.8), and the other numeric-gamma color spaces in
    /// upstream's `color_spaces.json`. Inner f32 is the gamma
    /// exponent; defaults to 2.2 when constructed via
    /// [`Eotf::default_gamma`].
    Gamma(f32),
}

impl Eotf {
    /// Convenience: `Eotf::Gamma(2.2)`. Saves callers from writing
    /// the literal in the common Adobe-RGB / Wide-Gamut case.
    ///
    /// # Examples
    /// ```
    /// use cvvdp_gpu::params::Eotf;
    /// assert!(matches!(Eotf::default_gamma(), Eotf::Gamma(g) if (g - 2.2).abs() < 1e-6));
    /// ```
    #[must_use]
    pub const fn default_gamma() -> Self {
        Self::Gamma(2.2)
    }

    /// Inverse-EOTF entry point: convert a single display-encoded
    /// channel value `v` (0..1 normalized for relative EOTFs;
    /// 0..1 PQ-encoded for [`Eotf::Pq`]; 0..1 or cd/m² for
    /// [`Eotf::Linear`]) to absolute cd/m² emitted from the
    /// display, including ambient reflection.
    ///
    /// Inputs:
    /// - `v` — display-encoded value. Clamped to 0..1 for the
    ///   relative EOTFs (sRGB / HLG / Gamma / Bt1886). Linear and
    ///   PQ accept inputs as-is.
    /// - `y_peak` — display peak luminance in cd/m².
    /// - `y_black` — display black level in cd/m² (computed
    ///   host-side as `y_peak / contrast`).
    /// - `y_refl` — light reflected off the display in cd/m²
    ///   (`E_ambient / π * k_refl`).
    ///
    /// Output: cd/m² emitted luminance.
    ///
    /// Mirrors pycvvdp v0.5.4's
    /// `vvdp_display_photo_eotf.forward` for the corresponding
    /// EOTF branch.
    ///
    /// # Examples
    ///
    /// ```
    /// use cvvdp_gpu::params::Eotf;
    ///
    /// // sRGB at the white point: V=1 → y_peak + y_refl.
    /// let l_white = Eotf::Srgb.forward(1.0, 200.0, 0.2, 0.4);
    /// assert!((l_white - 200.4).abs() < 1e-3);
    ///
    /// // Linear EOTF: V already in cd/m². V=100 → 100 + y_refl,
    /// // clipped to [max(0.005, y_black), y_peak].
    /// let l_lin = Eotf::Linear.forward(100.0, 200.0, 0.2, 0.4);
    /// assert!((l_lin - 100.4).abs() < 1e-3);
    /// ```
    #[must_use]
    pub fn forward(self, v: f32, y_peak: f32, y_black: f32, y_refl: f32) -> f32 {
        let bias = y_black + y_refl;
        match self {
            Self::Srgb => {
                let v = v.clamp(0.0, 1.0);
                let lin = srgb_eotf_scalar(v);
                (y_peak - y_black) * lin + bias
            }
            Self::Pq => {
                let lin = pq_eotf_scalar(v);
                let clipped = lin.clamp(0.005, y_peak);
                clipped + bias
            }
            Self::Hlg => {
                // The OOTF (Y_s^(gamma-1)) needs the full RGB
                // triple to compute Y_s. The single-channel
                // forward applies inverse OETF only; the
                // per-pixel OOTF is applied by the color stage
                // because it depends on all three channels.
                // Callers wanting the full HLG forward at a
                // single channel get the relative-linear value
                // before OOTF.
                let v = v.clamp(0.0, 1.0);
                let lin = hlg_inverse_oetf_scalar(v);
                (y_peak - y_black) * lin + bias
            }
            Self::Linear => {
                let floor = if y_black > 0.005 { y_black } else { 0.005 };
                let clipped = v.clamp(floor, y_peak);
                clipped + y_refl
            }
            Self::Bt1886 => {
                // BT.1886: L = a * (max(V + b, 0))^gamma with
                // gamma = 2.4 and a/b chosen so L(0) = y_black,
                // L(1) = y_peak. Simplifies to the lifted
                // power-law form below. We then add y_refl per
                // the cvvdp ambient convention (BT.1886 itself
                // doesn't model ambient).
                let gamma = 2.4_f32;
                let v = v.clamp(0.0, 1.0);
                let lift_a = (y_peak.powf(1.0 / gamma) - y_black.powf(1.0 / gamma)).powf(gamma);
                let lift_b = y_black.powf(1.0 / gamma) / (y_peak.powf(1.0 / gamma) - y_black.powf(1.0 / gamma));
                let l = lift_a * (v + lift_b).max(0.0).powf(gamma);
                l + y_refl
            }
            Self::Gamma(g) => {
                let v = v.clamp(0.0, 1.0);
                (y_peak - y_black) * v.powf(g) + bias
            }
        }
    }
}

/// Scalar sRGB EOTF (display-encoded 0..1 → relative-linear 0..1).
/// Pulled out so unit tests can compare the LUT against the formula.
#[must_use]
#[inline]
pub fn srgb_eotf_scalar(v: f32) -> f32 {
    if v > 0.040_45 {
        ((v + 0.055) / 1.055).powf(2.4)
    } else {
        v / 12.92
    }
}

/// Scalar SMPTE ST 2084 PQ EOTF (PQ-encoded 0..1 → cd/m² in
/// `[0, 10000]`). Reference: pycvvdp `pq2lin`.
#[must_use]
#[inline]
pub fn pq_eotf_scalar(v: f32) -> f32 {
    const L_MAX: f32 = 10000.0;
    const N: f32 = 0.159_301_75; // m1
    const M: f32 = 78.843_75; // m2
    const C1: f32 = 0.835_937_5;
    const C2: f32 = 18.851_562;
    const C3: f32 = 18.687_5;

    let im_t = v.powf(1.0 / M);
    let num = (im_t - C1).max(0.0);
    let den = C2 - C3 * im_t;
    L_MAX * (num / den).powf(1.0 / N)
}

/// Scalar HLG inverse-OETF (display-encoded 0..1 → scene-relative
/// linear in 0..12). Per-channel; the OOTF (system gamma) is a
/// per-pixel multiplier applied by the color stage because it
/// depends on the RGB triple's luminance.
///
/// Reference: pycvvdp `hlg2lin`; BT.2100-1 Table 5.
#[must_use]
#[inline]
pub fn hlg_inverse_oetf_scalar(v: f32) -> f32 {
    const A: f32 = 0.178_832_77;
    const B: f32 = 1.0 - 4.0 * A;
    let c = 0.5 - A * (4.0 * A).ln();
    if v <= 0.5 {
        (v * v) / 3.0
    } else {
        (((v - c) / A).exp() + B) / 12.0
    }
}

/// HLG system gamma per BT.2100-1 / BBC WHP369. For a 1000 cd/m²
/// peak the gamma is 1.2; brighter peaks add a luminance term and
/// an ambient-light correction. Mirrors pycvvdp's `forward`
/// branch for `EOTF=='HLG'`.
#[must_use]
#[inline]
pub fn hlg_system_gamma(y_peak: f32, e_ambient_lux: f32) -> f32 {
    if y_peak <= 1000.0 {
        1.2
    } else {
        // The "log10(e_ambient / 5)" correction is undefined at
        // E=0; pycvvdp guards by branching only when peak > 1000,
        // implicitly assuming a non-zero ambient when an HDR peak
        // is set. Match its arithmetic.
        let amb = if e_ambient_lux > 0.0 { e_ambient_lux } else { 5.0 };
        1.2 + 0.42 * (y_peak / 1000.0).log10() - 0.076_23 * (amb / 5.0).log10()
    }
}

/// RGB color primaries (chromaticities of the R/G/B emitters and
/// white point) for the input pixel-encoding color space.
///
/// Each variant determines the per-stage RGB→XYZ matrix that gets
/// chained into the combined RGB→DKL transform. Selecting the wrong
/// primaries shifts every chroma decision the metric makes; e.g. a
/// saturated BT.2020 red interpreted as BT.709 ends up with a much
/// larger achromatic component (because BT.2020's red lobe is more
/// spectrally pure).
///
/// Matches the `RGB2X/Y/Z` rows in upstream's `color_spaces.json`.
/// Variants below cover every primaries set that any preset in
/// `display_models.json` references. Bt709 is the default — it's
/// what sRGB / BT.709 / BT.709-linear all use, plus every preset
/// that omits the `colorspace` field.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Primaries {
    /// BT.709 / sRGB primaries with a D65 white point. Default —
    /// every SDR preset and every v1 R2 golden uses this.
    #[default]
    Bt709,
    /// BT.2020 primaries (wide-gamut UHDTV) with a D65 white
    /// point. Used by every `BT.2020-*` color space:
    /// `BT.2020-PQ`, `BT.2020-HLG`, `BT.2020-linear`.
    Bt2020,
    /// Display P3 primaries (Apple's wide-gamut consumer space)
    /// with a D65 white point. Sometimes labelled "P3-D65" to
    /// distinguish from theatrical DCI-P3.
    DisplayP3,
    /// DCI-P3 primaries with a D65 white point. Currently an
    /// alias for [`Primaries::DisplayP3`] — upstream doesn't ship
    /// a theatrical (DCI white, 48 nit) preset either. Kept as a
    /// separate variant for callers that prefer the DCI label.
    DciP3,
}

impl Primaries {
    /// 3×3 row-major matrix mapping linear RGB in these primaries
    /// to DKLd65 opponent space, computed as
    /// `LMS2006_to_DKLd65 @ XYZ_to_LMS2006 @ RGB_to_XYZ` at f64
    /// precision then truncated to f32. The BT.709 row is
    /// bit-identical to the existing [`SRGB_LINEAR_TO_DKL`]
    /// constant.
    #[must_use]
    pub const fn linear_rgb_to_dkl(self) -> [[f32; 3]; 3] {
        match self {
            Self::Bt709 => SRGB_LINEAR_TO_DKL,
            Self::Bt2020 => BT2020_LINEAR_TO_DKL,
            // DCI-P3 and DisplayP3 share primaries here — see the
            // variant docs for why theatrical DCI-P3 (DCI white)
            // isn't a separate matrix yet.
            Self::DisplayP3 | Self::DciP3 => DISPLAY_P3_LINEAR_TO_DKL,
        }
    }
}

/// Display model: how display-encoded pixels are mapped to physical
/// luminance (cd/m²) before perceptual processing.
///
/// Matches cvvdp's `vvdp_display_photo_eotf.forward` for the
/// configured EOTF branch:
///
/// ```text
/// L_rgb = EOTF(byte / 255, y_peak, y_black, y_refl)  // sRGB / Gamma / HLG / BT.1886
///       = pq2lin(V).clamp(0.005, y_peak) + y_black + y_refl  // PQ
///       = V.clamp(max(0.005, y_black), y_peak) + y_refl       // Linear
/// ```
///
/// The historical 3-field shape (`y_peak`, `y_black`, `y_refl`) is
/// preserved — every v1 caller continues to compile and produce
/// bit-identical scores. Added fields default to the sRGB /
/// BT.709 / 250 lux ambient configuration of [`Self::STANDARD_4K`].
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::params::DisplayModel;
///
/// // The `STANDARD_4K` preset is what every v1 R2 golden was
/// // captured against.
/// let d = DisplayModel::STANDARD_4K;
/// assert_eq!(d.y_peak, 200.0); // 200 cd/m² peak luminance
/// assert_eq!(d.y_black, 0.2);  // 0.2 cd/m² black level (contrast 1000)
///
/// // Construct a custom display (e.g. HDR400-ish) by aggregate-
/// // updating from STANDARD_4K — added fields inherit:
/// let hdr400 = DisplayModel { y_peak: 400.0, ..d };
/// assert!(hdr400.y_peak > d.y_peak);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct DisplayModel {
    /// Peak display luminance in cd/m².
    pub y_peak: f32,
    /// Black level in cd/m² (`y_peak / contrast`).
    pub y_black: f32,
    /// Light reflected from the screen, precomputed host-side from
    /// `E_ambient / π * k_refl`. Kept as a public field for
    /// back-compat with v1 callers that wrote the precomputed value
    /// directly — host code may also derive it via
    /// [`Self::compute_y_refl`] from the new `e_ambient_lux` and
    /// `k_refl` fields.
    pub y_refl: f32,
    /// Display EOTF — how display-encoded pixels are linearised.
    /// Defaults to [`Eotf::Srgb`] for back-compat with v1 callers.
    pub eotf: Eotf,
    /// RGB primaries / chromaticities of the input pixel-encoding
    /// space. Defaults to [`Primaries::Bt709`].
    pub primaries: Primaries,
    /// Ambient illuminance in lux at the display surface. Used
    /// host-side together with [`Self::k_refl`] to derive
    /// [`Self::y_refl`] via [`Self::compute_y_refl`].
    /// 250 lux matches cvvdp's `standard_4k`. Stored alongside
    /// the precomputed `y_refl` for inspection / round-tripping;
    /// the GPU kernels only consume `y_refl`.
    pub e_ambient_lux: f32,
    /// Display reflectivity coefficient (fraction of incident
    /// illuminance reflected back to the viewer). Default 0.005
    /// matches cvvdp's `vvdp_display_photo_eotf`. Stored for
    /// inspection / round-tripping.
    pub k_refl: f32,
}

impl DisplayModel {
    /// cvvdp's `standard_4k` defaults — peak 200 cd/m², contrast 1000,
    /// 250 lux ambient, k_refl 0.005, sRGB EOTF, BT.709 primaries.
    /// The v1 R2 goldens were captured under this display, so every
    /// existing parity test pins these three values bit-exactly.
    ///
    /// # Examples
    ///
    /// ```
    /// use cvvdp_gpu::params::{DisplayModel, Eotf, Primaries};
    /// // Pinned by tests/display_geometry.rs::display_model_standard_4k_matches_pycvvdp_v0_5_4
    /// // — y_peak, y_black, y_refl bit-pinned via `.to_bits()`.
    /// let s = DisplayModel::STANDARD_4K;
    /// assert_eq!(s.y_peak, 200.0);
    /// assert_eq!(s.y_black, 0.2);
    /// // y_refl is precomputed host-side from 250 lux × 0.005 / π.
    /// assert!((s.y_refl - 0.397_887_36).abs() < 1e-6);
    /// // EOTF / primaries are sRGB / BT.709 by default.
    /// assert_eq!(s.eotf, Eotf::Srgb);
    /// assert_eq!(s.primaries, Primaries::Bt709);
    /// assert_eq!(s.e_ambient_lux, 250.0);
    /// assert!((s.k_refl - 0.005).abs() < 1e-6);
    /// ```
    pub const STANDARD_4K: Self = Self {
        y_peak: 200.0,
        y_black: 0.2,
        y_refl: 0.397_887_36,
        eotf: Eotf::Srgb,
        primaries: Primaries::Bt709,
        e_ambient_lux: 250.0,
        k_refl: 0.005,
    };

    /// `standard_hdr_pq` — 1500 cd/m² HDR PQ with BT.2020 primaries
    /// at 10 lux ambient. Matches upstream `display_models.json`.
    pub const STANDARD_HDR_PQ: Self = Self {
        y_peak: 1500.0,
        y_black: 0.001_5,
        y_refl: 0.015_915_494,
        eotf: Eotf::Pq,
        primaries: Primaries::Bt2020,
        e_ambient_lux: 10.0,
        k_refl: 0.005,
    };

    /// `standard_hdr_hlg` — 1500 cd/m² HDR HLG with BT.2020 primaries.
    pub const STANDARD_HDR_HLG: Self = Self {
        y_peak: 1500.0,
        y_black: 0.001_5,
        y_refl: 0.015_915_494,
        eotf: Eotf::Hlg,
        primaries: Primaries::Bt2020,
        e_ambient_lux: 10.0,
        k_refl: 0.005,
    };

    /// `standard_hdr_linear` — 1500 cd/m² HDR linear with BT.709 primaries.
    pub const STANDARD_HDR_LINEAR: Self = Self {
        y_peak: 1500.0,
        y_black: 0.001_5,
        y_refl: 0.015_915_494,
        eotf: Eotf::Linear,
        primaries: Primaries::Bt709,
        e_ambient_lux: 10.0,
        k_refl: 0.005,
    };

    /// `standard_hdr_linear_dark` — 1500 cd/m² HDR linear, dark room
    /// (0 lux ambient).
    pub const STANDARD_HDR_LINEAR_DARK: Self = Self {
        y_peak: 1500.0,
        y_black: 0.001_5,
        y_refl: 0.0,
        eotf: Eotf::Linear,
        primaries: Primaries::Bt709,
        e_ambient_lux: 0.0,
        k_refl: 0.005,
    };

    /// `standard_hdr_linear_zoom` — 10000 cd/m² zoomed-in HDR linear.
    /// Y_peak per upstream JSON (10000, NOT 4000 as the comment claims).
    pub const STANDARD_HDR_LINEAR_ZOOM: Self = Self {
        y_peak: 10000.0,
        y_black: 0.01,
        y_refl: 0.015_915_494,
        eotf: Eotf::Linear,
        primaries: Primaries::Bt709,
        e_ambient_lux: 10.0,
        k_refl: 0.005,
    };

    /// `standard_fhd` — 200 cd/m² FHD with sRGB at office ambient.
    pub const STANDARD_FHD: Self = Self {
        y_peak: 200.0,
        y_black: 0.2,
        y_refl: 0.397_887_36,
        eotf: Eotf::Srgb,
        primaries: Primaries::Bt709,
        e_ambient_lux: 250.0,
        k_refl: 0.005,
    };

    /// `standard_phone` — 500 cd/m² mobile phone display, 250 lux ambient.
    /// `min_luminance = 0.05` → contrast = 10000.
    pub const STANDARD_PHONE: Self = Self {
        y_peak: 500.0,
        y_black: 0.05,
        y_refl: 0.397_887_36,
        eotf: Eotf::Srgb,
        primaries: Primaries::Bt709,
        e_ambient_lux: 250.0,
        k_refl: 0.005,
    };

    /// `sdr_4k_30` — 100 cd/m² SDR 4K monitor, 250 lux ambient.
    pub const SDR_4K_30: Self = Self {
        y_peak: 100.0,
        y_black: 0.1,
        y_refl: 0.397_887_36,
        eotf: Eotf::Srgb,
        primaries: Primaries::Bt709,
        e_ambient_lux: 250.0,
        k_refl: 0.005,
    };

    /// `sdr_fhd_24` — 100 cd/m² SDR FHD monitor, 250 lux ambient.
    pub const SDR_FHD_24: Self = Self {
        y_peak: 100.0,
        y_black: 0.1,
        y_refl: 0.397_887_36,
        eotf: Eotf::Srgb,
        primaries: Primaries::Bt709,
        e_ambient_lux: 250.0,
        k_refl: 0.005,
    };

    /// `htc_vive_pro` — VR HMD at 133.3 cd/m², 0 lux ambient. Contrast
    /// = 1333.3.
    pub const HTC_VIVE_PRO: Self = Self {
        y_peak: 133.3,
        y_black: 0.1,
        y_refl: 0.0,
        eotf: Eotf::Srgb,
        primaries: Primaries::Bt709,
        e_ambient_lux: 0.0,
        k_refl: 0.005,
    };

    /// `iphone_12_pro` — 825 cd/m², min_luminance 0.0004 → contrast
    /// = 2062500. 250 lux ambient.
    pub const IPHONE_12_PRO: Self = Self {
        y_peak: 825.0,
        y_black: 0.000_4,
        y_refl: 0.397_887_36,
        eotf: Eotf::Srgb,
        primaries: Primaries::Bt709,
        e_ambient_lux: 250.0,
        k_refl: 0.005,
    };

    /// `iphone_14_pro` — 1025 cd/m², min_luminance 0.0004. 250 lux
    /// ambient.
    pub const IPHONE_14_PRO: Self = Self {
        y_peak: 1025.0,
        y_black: 0.000_4,
        y_refl: 0.397_887_36,
        eotf: Eotf::Srgb,
        primaries: Primaries::Bt709,
        e_ambient_lux: 250.0,
        k_refl: 0.005,
    };

    /// `iphone_14_pro_hdr` — 1590 cd/m² HDR HLG with BT.2020 primaries.
    /// 10 lux ambient.
    pub const IPHONE_14_PRO_HDR: Self = Self {
        y_peak: 1590.0,
        y_black: 0.000_4,
        y_refl: 0.015_915_494,
        eotf: Eotf::Hlg,
        primaries: Primaries::Bt2020,
        e_ambient_lux: 10.0,
        k_refl: 0.005,
    };

    /// `ipad_pro_12_9` — 600 cd/m², min_luminance 0.37 → contrast
    /// ≈ 1621. 250 lux ambient.
    pub const IPAD_PRO_12_9: Self = Self {
        y_peak: 600.0,
        y_black: 0.37,
        y_refl: 0.397_887_36,
        eotf: Eotf::Srgb,
        primaries: Primaries::Bt709,
        e_ambient_lux: 250.0,
        k_refl: 0.005,
    };

    /// `macbook_pro_16` — 500 cd/m², min_luminance 0.37 → contrast
    /// ≈ 1351. 250 lux ambient.
    pub const MACBOOK_PRO_16: Self = Self {
        y_peak: 500.0,
        y_black: 0.37,
        y_refl: 0.397_887_36,
        eotf: Eotf::Srgb,
        primaries: Primaries::Bt709,
        e_ambient_lux: 250.0,
        k_refl: 0.005,
    };

    /// `lg_oled_2017_sdr` — 272 cd/m² OLED SDR. min_luminance 0.014.
    /// 100 lux ambient.
    pub const LG_OLED_2017_SDR: Self = Self {
        y_peak: 272.0,
        y_black: 0.014,
        y_refl: 0.159_154_94,
        eotf: Eotf::Srgb,
        primaries: Primaries::Bt709,
        e_ambient_lux: 100.0,
        k_refl: 0.005,
    };

    /// `lg_oled_2017_hdr` — 754 cd/m² OLED HDR (HLG-encoded). 100 lux
    /// ambient.
    pub const LG_OLED_2017_HDR: Self = Self {
        y_peak: 754.0,
        y_black: 0.038,
        y_refl: 0.159_154_94,
        eotf: Eotf::Hlg,
        primaries: Primaries::Bt2020,
        e_ambient_lux: 100.0,
        k_refl: 0.005,
    };

    /// `eizo_CG3146` — 300 cd/m², contrast 3000, 0 lux ambient.
    pub const EIZO_CG3146: Self = Self {
        y_peak: 300.0,
        y_black: 0.1,
        y_refl: 0.0,
        eotf: Eotf::Srgb,
        primaries: Primaries::Bt709,
        e_ambient_lux: 0.0,
        k_refl: 0.005,
    };

    /// `65inch_hdr_pq_4knit` — 4000 cd/m² HDR PQ OLED, 5 lux ambient.
    pub const HDR_PQ_4KNIT: Self = Self {
        y_peak: 4000.0,
        y_black: 0.004,
        y_refl: 0.007_957_747,
        eotf: Eotf::Pq,
        primaries: Primaries::Bt2020,
        e_ambient_lux: 5.0,
        k_refl: 0.005,
    };

    /// `65inch_hdr_pq_2Knit` — 2000 cd/m² HDR PQ OLED, 5 lux ambient.
    pub const HDR_PQ_2KNIT: Self = Self {
        y_peak: 2000.0,
        y_black: 0.002,
        y_refl: 0.007_957_747,
        eotf: Eotf::Pq,
        primaries: Primaries::Bt2020,
        e_ambient_lux: 5.0,
        k_refl: 0.005,
    };

    /// `65inch_hdr_pq_1Knit` — 1000 cd/m² HDR PQ OLED, 5 lux ambient.
    pub const HDR_PQ_1KNIT: Self = Self {
        y_peak: 1000.0,
        y_black: 0.001,
        y_refl: 0.007_957_747,
        eotf: Eotf::Pq,
        primaries: Primaries::Bt2020,
        e_ambient_lux: 5.0,
        k_refl: 0.005,
    };

    /// `lg_oled_2026_hdr_pq` — 3000 cd/m² HDR PQ OLED. min_luminance
    /// 0.0005 → contrast 6e6. 5 lux ambient.
    pub const LG_OLED_2026_HDR_PQ: Self = Self {
        y_peak: 3000.0,
        y_black: 0.000_5,
        y_refl: 0.007_957_747,
        eotf: Eotf::Pq,
        primaries: Primaries::Bt2020,
        e_ambient_lux: 5.0,
        k_refl: 0.005,
    };

    /// Derive `y_refl` from ambient illuminance and screen
    /// reflectivity per cvvdp's
    /// `vvdp_display_photo_eotf.get_black_level`:
    /// `y_refl = E_ambient / pi * k_refl`.
    ///
    /// # Examples
    /// ```
    /// use cvvdp_gpu::params::DisplayModel;
    /// // 250 lux × 0.005 / π = 0.397_887_36 (matches STANDARD_4K).
    /// let r = DisplayModel::compute_y_refl(250.0, 0.005);
    /// assert!((r - DisplayModel::STANDARD_4K.y_refl).abs() < 1e-6);
    /// ```
    #[must_use]
    pub fn compute_y_refl(e_ambient_lux: f32, k_refl: f32) -> f32 {
        e_ambient_lux / std::f32::consts::PI * k_refl
    }

    /// Constructor matching upstream's
    /// `vvdp_display_photo_eotf.__init__(Y_peak, contrast, EOTF,
    /// E_ambient, k_refl, ...)`. Derives `y_black` and `y_refl`
    /// host-side per cvvdp's `get_black_level`.
    ///
    /// Inputs:
    /// - `y_peak` — peak luminance in cd/m².
    /// - `contrast` — display contrast ratio; `y_black = y_peak /
    ///   contrast`. Use the on-paper number (1000 for office LCD,
    ///   1e6 for an OLED HDR panel).
    /// - `e_ambient_lux` — ambient illuminance at the screen, lux.
    /// - `k_refl` — screen reflectivity. Default in cvvdp is
    ///   0.005; pass that when the preset doesn't override.
    /// - `eotf` — display EOTF.
    /// - `primaries` — RGB primaries / chromaticities.
    ///
    /// # Examples
    /// ```
    /// use cvvdp_gpu::params::{DisplayModel, Eotf, Primaries};
    /// let d = DisplayModel::new(
    ///     200.0, 1000.0, 250.0, 0.005,
    ///     Eotf::Srgb, Primaries::Bt709,
    /// );
    /// // Matches STANDARD_4K bit-for-bit.
    /// assert_eq!(d.y_peak, DisplayModel::STANDARD_4K.y_peak);
    /// assert_eq!(d.y_black, DisplayModel::STANDARD_4K.y_black);
    /// assert!((d.y_refl - DisplayModel::STANDARD_4K.y_refl).abs() < 1e-6);
    /// ```
    #[must_use]
    pub fn new(
        y_peak: f32,
        contrast: f32,
        e_ambient_lux: f32,
        k_refl: f32,
        eotf: Eotf,
        primaries: Primaries,
    ) -> Self {
        let y_black = y_peak / contrast;
        let y_refl = Self::compute_y_refl(e_ambient_lux, k_refl);
        Self {
            y_peak,
            y_black,
            y_refl,
            eotf,
            primaries,
            e_ambient_lux,
            k_refl,
        }
    }
}

impl Default for DisplayModel {
    /// `STANDARD_4K` — every v1 caller pinned this implicitly via
    /// `CvvdpParams::PLACEHOLDER.display`, so the new explicit
    /// `Default` impl matches.
    fn default() -> Self {
        Self::STANDARD_4K
    }
}

/// Display geometry — resolution + viewing distance + physical size.
/// Used to derive pixels-per-degree, which the CSF stage consumes via
/// each pyramid band's spatial frequency.
///
/// Matches cvvdp's `vvdp_display_geometry` for the `diagonal_inches +
/// distance_m` path (the one the `standard_4k` JSON uses). Other
/// cvvdp paths (`fov_diagonal`, `fov_horizontal`, etc.) are not
/// ported until a use case appears.
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::params::DisplayGeometry;
///
/// let g = DisplayGeometry::STANDARD_4K;
/// assert_eq!(g.resolution_w, 3840);
/// assert_eq!(g.resolution_h, 2160);
///
/// // Custom geometry — e.g. phone at arm's length:
/// let phone = DisplayGeometry {
///     resolution_w: 1920,
///     resolution_h: 1080,
///     distance_m: 0.40,
///     diagonal_inches: 5.5,
/// };
/// // Smaller display at closer distance → higher PPD than 4K.
/// assert!(phone.pixels_per_degree() > g.pixels_per_degree());
/// ```
#[derive(Debug, Clone, Copy)]
pub struct DisplayGeometry {
    /// Display width in pixels.
    pub resolution_w: u32,
    /// Display height in pixels.
    pub resolution_h: u32,
    /// Viewing distance in meters.
    pub distance_m: f32,
    /// Display diagonal in inches.
    pub diagonal_inches: f32,
}

impl DisplayGeometry {
    /// cvvdp's `standard_4k`: 3840×2160, 30" diagonal, 0.7472 m. The
    /// PPD this derives to (~75.40) is what the v1 R2 manifest's
    /// `rho_band[0]` (= ppd/2 ≈ 37.70) is computed against.
    ///
    /// # Examples
    ///
    /// ```
    /// use cvvdp_gpu::params::DisplayGeometry;
    /// // Pinned by tests/display_geometry.rs — all four fields
    /// // bit-pinned, plus the derived PPD pinned to 75.402_449
    /// // (within 1e-4) by `display_geometry_standard_4k_matches_pycvvdp_v0_5_4`.
    /// let g = DisplayGeometry::STANDARD_4K;
    /// assert_eq!(g.resolution_w, 3840);
    /// assert_eq!(g.resolution_h, 2160);
    /// assert!((g.distance_m - 0.7472).abs() < 1e-6);
    /// assert!((g.diagonal_inches - 30.0).abs() < 1e-6);
    /// ```
    pub const STANDARD_4K: Self = Self {
        resolution_w: 3840,
        resolution_h: 2160,
        distance_m: 0.7472,
        diagonal_inches: 30.0,
    };

    /// `standard_fhd` — 24-inch FHD monitor at 0.6 m, 200 cd/m².
    /// Matches the `standard_fhd` entry in upstream `display_models.json`.
    pub const STANDARD_FHD: Self = Self {
        resolution_w: 1920,
        resolution_h: 1080,
        distance_m: 0.6,
        diagonal_inches: 24.0,
    };

    /// `sdr_4k_30` — 30-inch 4K SDR monitor at 0.6 m, 100 cd/m².
    pub const SDR_4K_30: Self = Self {
        resolution_w: 3840,
        resolution_h: 2160,
        distance_m: 0.6,
        diagonal_inches: 30.0,
    };

    /// `sdr_fhd_24` — 24-inch FHD SDR monitor at 0.6 m, 100 cd/m².
    pub const SDR_FHD_24: Self = Self {
        resolution_w: 1920,
        resolution_h: 1080,
        distance_m: 0.6,
        diagonal_inches: 24.0,
    };

    /// `standard_phone` — 6-inch 2400×1080 phone at 0.4 m, 500 cd/m².
    pub const STANDARD_PHONE: Self = Self {
        resolution_w: 2400,
        resolution_h: 1080,
        distance_m: 0.4,
        diagonal_inches: 6.0,
    };

    /// `iphone_12_pro` — 6.1" 2532×1170 at 20" viewing distance.
    /// 20" = 0.508 m.
    pub const IPHONE_12_PRO: Self = Self {
        resolution_w: 2532,
        resolution_h: 1170,
        distance_m: 0.508,
        diagonal_inches: 6.1,
    };

    /// `iphone_14_pro` — 6.1" 2532×1170 at 20" viewing distance.
    pub const IPHONE_14_PRO: Self = Self {
        resolution_w: 2532,
        resolution_h: 1170,
        distance_m: 0.508,
        diagonal_inches: 6.1,
    };

    /// `iphone_14_pro_vert` — iPhone 14 Pro held vertically (W/H swapped).
    pub const IPHONE_14_PRO_VERT: Self = Self {
        resolution_w: 1170,
        resolution_h: 2532,
        distance_m: 0.508,
        diagonal_inches: 6.1,
    };

    /// `ipad_pro_12_9` — 12.9" 2732×2048 at 20" viewing distance.
    pub const IPAD_PRO_12_9: Self = Self {
        resolution_w: 2732,
        resolution_h: 2048,
        distance_m: 0.508,
        diagonal_inches: 12.9,
    };

    /// `macbook_pro_16` — 16" 3072×1920 at 25" viewing distance.
    /// 25" = 0.635 m.
    pub const MACBOOK_PRO_16: Self = Self {
        resolution_w: 3072,
        resolution_h: 1920,
        distance_m: 0.635,
        diagonal_inches: 16.0,
    };

    /// `lg_oled_2017` — 64.5" 3840×2160 at 101" viewing distance.
    /// Shared geometry between SDR + HDR LG OLED 2017 presets.
    /// 101" = 2.5654 m.
    pub const LG_OLED_2017: Self = Self {
        resolution_w: 3840,
        resolution_h: 2160,
        distance_m: 2.5654,
        diagonal_inches: 64.5,
    };

    /// `eizo_CG3146` — 31.063" 4096×2160 at 0.73406 m.
    pub const EIZO_CG3146: Self = Self {
        resolution_w: 4096,
        resolution_h: 2160,
        distance_m: 0.73406,
        diagonal_inches: 31.063,
    };

    /// `65inch_hdr_pq_*` family geometry — 65" 3840×2160 at 1.98 m.
    pub const PANEL_65IN_4K: Self = Self {
        resolution_w: 3840,
        resolution_h: 2160,
        distance_m: 1.98,
        diagonal_inches: 65.0,
    };

    /// `lg_oled_2026_hdr_pq` — 64.9" 3840×2160 at 86.62" (2.2 m).
    pub const LG_OLED_2026: Self = Self {
        resolution_w: 3840,
        resolution_h: 2160,
        distance_m: 2.2,
        diagonal_inches: 64.9,
    };

    /// `standard_hdr_linear_zoom` — 4K at very close distance (0.25 m)
    /// to spot super-resolution artifacts.
    pub const HDR_LINEAR_ZOOM: Self = Self {
        resolution_w: 3840,
        resolution_h: 2160,
        distance_m: 0.25,
        diagonal_inches: 30.0,
    };

    /// `htc_vive_pro` / `standard_hmd` — VR HMD with 1440×1600 per eye,
    /// 110° diagonal FOV at 3 m. Computed from upstream's `fov_diagonal`
    /// path (display_model.py:474-485): an equivalent diagonal_inches
    /// chosen so `pixels_per_degree()` returns the same value as the
    /// FOV-based path would.
    ///
    /// Derivation (per upstream):
    /// distance_px = sqrt(W² + H²) / (2 * tan(fov_diag/2))
    ///             = sqrt(1440² + 1600²) / (2 * tan(55°))
    ///             ≈ 753.16
    /// height_deg  = degrees(atan(H/2 / distance_px)) * 2 ≈ 93.575°
    /// height_m    = 2 * tan(height_deg/2) * 3 m ≈ 6.4115 m
    /// width_m     = aspect * height_m ≈ 5.770 m
    /// diag_m      = sqrt(width_m² + height_m²) ≈ 8.628 m
    /// diag_inches = diag_m / 0.0254 ≈ 339.7 inches
    ///
    /// The resulting PPD ≈ 11.84 matches upstream's get_ppd() output.
    pub const HTC_VIVE_PRO: Self = Self {
        resolution_w: 1440,
        resolution_h: 1600,
        distance_m: 3.0,
        diagonal_inches: 339.7,
    };

    /// Build a [`DisplayGeometry`] from physical fields. Identical to
    /// struct-update syntax — provided for symmetry with the
    /// `with_*` builder family.
    #[must_use]
    pub fn new(resolution_w: u32, resolution_h: u32, distance_m: f32, diagonal_inches: f32) -> Self {
        Self {
            resolution_w,
            resolution_h,
            distance_m,
            diagonal_inches,
        }
    }

    /// Build a [`DisplayGeometry`] from inch-denominated fields.
    /// Mirrors upstream's `viewing_distance_inches` +
    /// `diagonal_size_inches` JSON path. Inputs converted to metres
    /// at `inches × 0.0254`.
    ///
    /// # Examples
    /// ```
    /// use cvvdp_gpu::params::DisplayGeometry;
    /// // iPhone 12 Pro: 20" viewing distance, 6.1" diagonal.
    /// let g = DisplayGeometry::from_inches(2532, 1170, 20.0, 6.1);
    /// assert_eq!(g.resolution_w, 2532);
    /// assert!((g.distance_m - 0.508).abs() < 1e-4);
    /// assert!((g.diagonal_inches - 6.1).abs() < 1e-6);
    /// ```
    #[must_use]
    pub fn from_inches(
        resolution_w: u32,
        resolution_h: u32,
        viewing_distance_inches: f32,
        diagonal_inches: f32,
    ) -> Self {
        Self {
            resolution_w,
            resolution_h,
            distance_m: viewing_distance_inches * 0.0254,
            diagonal_inches,
        }
    }

    /// Build a [`DisplayGeometry`] from a metre-denominated diagonal.
    /// Mirrors upstream's `diagonal_size_meters` JSON path; inches
    /// derived at `metres / 0.0254`.
    #[must_use]
    pub fn from_meters_diagonal(
        resolution_w: u32,
        resolution_h: u32,
        distance_m: f32,
        diagonal_meters: f32,
    ) -> Self {
        Self {
            resolution_w,
            resolution_h,
            distance_m,
            diagonal_inches: diagonal_meters / 0.0254,
        }
    }

    /// Build a [`DisplayGeometry`] from a diagonal field-of-view
    /// angle (in degrees). Mirrors upstream's `fov_diagonal` JSON
    /// path used by VR headset presets (`htc_vive_pro`,
    /// `standard_hmd`). Computes an equivalent `diagonal_inches` so
    /// [`Self::pixels_per_degree`] returns the same PPD as upstream's
    /// FOV-based getter.
    ///
    /// Derivation matches `display_model.py:474-485`:
    /// 1. `distance_px = sqrt(W² + H²) / (2 · tan(fov_diag/2))`
    /// 2. `height_deg = 2 · atan((H/2) / distance_px)`
    /// 3. `height_m = 2 · tan(height_deg/2) · distance_m`
    /// 4. `width_m = aspect · height_m`
    /// 5. `diagonal_m = sqrt(width_m² + height_m²)`
    ///
    /// # Examples
    /// ```
    /// use cvvdp_gpu::params::DisplayGeometry;
    /// let g = DisplayGeometry::from_fov_diagonal(1440, 1600, 3.0, 110.0);
    /// // 110° FOV at 3 m on 1440×1600 → ~11.84 ppd, matches upstream
    /// // get_ppd() to within 0.05.
    /// let ppd = g.pixels_per_degree();
    /// assert!((ppd - 11.84).abs() < 0.05, "got {ppd}");
    /// ```
    #[must_use]
    pub fn from_fov_diagonal(
        resolution_w: u32,
        resolution_h: u32,
        distance_m: f32,
        fov_diagonal_deg: f32,
    ) -> Self {
        let w = resolution_w as f32;
        let h = resolution_h as f32;
        let half_fov_rad = (fov_diagonal_deg * 0.5_f32).to_radians();
        let distance_px = (w * w + h * h).sqrt() / (2.0 * half_fov_rad.tan());
        let height_deg = ((h * 0.5) / distance_px).atan().to_degrees() * 2.0;
        let height_m = 2.0 * (height_deg * 0.5).to_radians().tan() * distance_m;
        let aspect = w / h;
        let width_m = aspect * height_m;
        let diagonal_m = (width_m * width_m + height_m * height_m).sqrt();
        Self {
            resolution_w,
            resolution_h,
            distance_m,
            diagonal_inches: diagonal_m / 0.0254,
        }
    }

    /// Display width in metres (derived from `diagonal_inches` and
    /// aspect ratio). Mirrors `display_size_m[0]` from upstream's
    /// `vvdp_display_geometry`.
    #[must_use]
    pub fn display_width_m(&self) -> f32 {
        let ar = self.resolution_w as f32 / self.resolution_h as f32;
        let diagonal_mm = self.diagonal_inches * 25.4;
        let height_mm = (diagonal_mm * diagonal_mm / (1.0 + ar * ar)).sqrt();
        ar * height_mm / 1000.0
    }

    /// Display height in metres. Mirrors `display_size_m[1]`.
    #[must_use]
    pub fn display_height_m(&self) -> f32 {
        let ar = self.resolution_w as f32 / self.resolution_h as f32;
        let diagonal_mm = self.diagonal_inches * 25.4;
        let height_mm = (diagonal_mm * diagonal_mm / (1.0 + ar * ar)).sqrt();
        height_mm / 1000.0
    }

    /// Display width in visual degrees. Mirrors `display_size_deg[0]`.
    #[must_use]
    pub fn display_width_deg(&self) -> f32 {
        let w_m = self.display_width_m();
        2.0 * (w_m / (2.0 * self.distance_m)).atan().to_degrees()
    }

    /// Display height in visual degrees. Mirrors `display_size_deg[1]`.
    #[must_use]
    pub fn display_height_deg(&self) -> f32 {
        let h_m = self.display_height_m();
        2.0 * (h_m / (2.0 * self.distance_m)).atan().to_degrees()
    }

    /// Pixels-per-degree at the display centre (eccentricity = 0).
    /// Matches cvvdp's `vvdp_display_geometry.get_ppd()` for the
    /// no-eccentricity path.
    ///
    /// # Examples
    ///
    /// ```
    /// use cvvdp_gpu::params::DisplayGeometry;
    ///
    /// // Standard 4K (3840×2160, 30 inch, 0.7472 m) → 75.402_449 ppd.
    /// // The 1e-4 tolerance matches the runtime parity test
    /// // `tests/display_geometry.rs::ppd_matches_pycvvdp_standard_4k`
    /// // — well within f32 noise for a value ~75.
    /// let ppd = DisplayGeometry::STANDARD_4K.pixels_per_degree();
    /// assert!((ppd - 75.402_449_f32).abs() < 1e-4);
    ///
    /// // PPD is positive and in a sane range for realistic viewing.
    /// assert!(ppd.is_finite() && (5.0..=500.0).contains(&ppd));
    /// ```
    #[must_use]
    pub fn pixels_per_degree(&self) -> f32 {
        let ar = self.resolution_w as f32 / self.resolution_h as f32;
        let diagonal_mm = self.diagonal_inches * 25.4;
        let height_mm = (diagonal_mm * diagonal_mm / (1.0 + ar * ar)).sqrt();
        let width_m = ar * height_mm / 1000.0;
        let pix_deg = 2.0
            * (0.5 * width_m / self.resolution_w as f32 / self.distance_m)
                .atan()
                .to_degrees();
        1.0 / pix_deg
    }
}

/// Combined `sRGB linear-RGB → DKLd65` matrix (row-major), computed at
/// f64 precision from cvvdp's published per-stage matrices:
/// `LMS2006_to_DKLd65 @ XYZ_to_LMS2006 @ sRGB_to_XYZ`.
///
/// Apply per-pixel as:
/// `dkl[c] = M[c][0]*r_lin + M[c][1]*g_lin + M[c][2]*b_lin`.
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::params::SRGB_LINEAR_TO_DKL as M;
///
/// // Equal-energy linear input (1.0, 1.0, 1.0) → A row sums to the
/// // luminance gain (≈ 1.05); chroma rows are mean-zero by DKL
/// // construction so are small relative to A.
/// let a_sum = M[0][0] + M[0][1] + M[0][2];
/// let rg_sum = M[1][0] + M[1][1] + M[1][2];
/// let vy_sum = M[2][0] + M[2][1] + M[2][2];
/// assert!(a_sum > 0.5 && a_sum < 2.0, "A row sum {a_sum}");
/// assert!(rg_sum.abs() < a_sum.abs());
/// assert!(vy_sum.abs() < a_sum.abs());
/// ```
pub const SRGB_LINEAR_TO_DKL: [[f32; 3]; 3] = [
    [0.233_201_21, 0.728_830_8, 0.088_995_87],
    [0.127_620_77, -0.087_068_09, -0.036_777_39],
    [-0.214_822_5, -0.626_253_7, 0.851_403_3],
];

/// Combined `BT.2020 linear-RGB → DKLd65` matrix (row-major),
/// computed at f64 precision from upstream's per-stage matrices:
/// `LMS2006_to_DKLd65 @ XYZ_to_LMS2006 @ BT2020_RGB2XYZ`, with the
/// BT.2020 primaries from `color_spaces.json`. Used for every
/// `BT.2020-PQ`, `BT.2020-HLG`, and `BT.2020-linear` preset.
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::params::BT2020_LINEAR_TO_DKL as M;
/// // A row sum positive (achromatic gain on equal-energy white).
/// let a_sum = M[0][0] + M[0][1] + M[0][2];
/// assert!(a_sum > 0.5 && a_sum < 2.0, "A row sum {a_sum}");
/// // Distinguishably different from the BT.709 matrix on at
/// // least one chroma row entry (BT.2020 red lobe is more
/// // saturated).
/// use cvvdp_gpu::params::SRGB_LINEAR_TO_DKL as S;
/// assert!((M[1][0] - S[1][0]).abs() > 0.05);
/// ```
pub const BT2020_LINEAR_TO_DKL: [[f32; 3]; 3] = [
    [0.294_774_83, 0.679_742_5, 0.076_514_23],
    [0.223_412_68, -0.169_937_05, -0.049_714_08],
    [-0.294_110_88, -0.668_908_85, 0.973_610_63],
];

/// Combined `Display P3 (D65) linear-RGB → DKLd65` matrix
/// (row-major). Mirrors upstream's `Display P3 Apple` color space
/// (`color_spaces.json`). Also returned for [`Primaries::DciP3`]
/// today — see that variant for why.
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::params::DISPLAY_P3_LINEAR_TO_DKL as M;
/// let a_sum = M[0][0] + M[0][1] + M[0][2];
/// assert!(a_sum > 0.5 && a_sum < 2.0);
/// ```
pub const DISPLAY_P3_LINEAR_TO_DKL: [[f32; 3]; 3] = [
    [0.253_237_6, 0.700_016_25, 0.097_775_31],
    [0.160_692_72, -0.116_510_76, -0.040_388_58],
    [-0.253_514_47, -0.671_234_8, 0.935_045_8],
];

/// castleCSF achromatic + chrom params. Scaffolding for a planned
/// "load from vendored cvvdp JSON" path that hasn't landed; the
/// production code reads CSF sensitivity straight from the
/// `csf_lut_weber_fixed_size` LUT vendored in `kernels/csf_lut/`,
/// not from this struct. See `CvvdpParams::PLACEHOLDER` for the
/// full unused-scaffolding picture.
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::CvvdpParams;
/// // PLACEHOLDER fills these with zeroed scaffolding values.
/// // Production CSF runs from the vendored LUT — see the struct
/// // docs.
/// let p = CvvdpParams::PLACEHOLDER;
/// assert_eq!(p.csf.a_peak, 0.0);
/// assert_eq!(p.csf.rg_peak, 0.0);
/// assert_eq!(p.csf.vy_peak, 0.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct CsfParams {
    /// Sensitivity peak for the achromatic channel.
    pub a_peak: f32,
    /// Sensitivity peak for the RG channel.
    pub rg_peak: f32,
    /// Sensitivity peak for the VY channel.
    pub vy_peak: f32,
    // ...remaining castleCSF coefficients added when the JSON is vendored.
}

/// Contrast masking model (within-channel + cross-channel).
///
/// Currently unused scaffolding — the GPU + host-scalar masking
/// kernels read `MASK_P`, `MASK_Q`, `MASK_C`, `D_MAX`, and
/// `XCM_3X3` as inlined `const`s in `kernels::masking`. See
/// `CvvdpParams::PLACEHOLDER` for the full picture.
///
/// The field set below doesn't 1:1 mirror the production
/// constants — `p` does (`MASK_P`, scalar), but `q` is a single
/// `f32` here vs. cvvdp's per-channel `MASK_Q: [f32; 3]`
/// `[A, RG, VY]`, and `k` doesn't correspond to any single named
/// masking constant (the production model uses `MASK_C` for the
/// phase-uncertainty `10^c` post-scale and `D_MAX` for the
/// clamp ceiling, both log10-encoded; there is no "saturation
/// epsilon" constant). A future "load from vendored cvvdp JSON"
/// path will need to widen `q` to `[f32; 3]` and split `k` into
/// the corresponding `MASK_C` / `D_MAX` fields — a breaking
/// change tracked separately.
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::CvvdpParams;
/// // PLACEHOLDER scaffolding values; production reads from
/// // `kernels::masking::{MASK_P, MASK_Q, MASK_C, D_MAX, XCM_3X3}`.
/// // Pinned by tests/params_placeholder_non_display.rs.
/// let m = CvvdpParams::PLACEHOLDER.masking;
/// assert_eq!(m.p, 2.4);
/// assert_eq!(m.q, 2.2);
/// assert!((m.k - 0.04).abs() < 1e-6);
/// // All three scaffolding fields are positive — required because
/// // they're future exponents on non-negative quantities.
/// assert!(m.p > 0.0 && m.q > 0.0 && m.k > 0.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct MaskingParams {
    /// Excitation exponent (cvvdp `p`) — matches `MASK_P`.
    pub p: f32,
    /// Inhibition exponent (cvvdp `q`). Shape mismatch: production
    /// is `MASK_Q: [f32; 3]` per-channel.
    pub q: f32,
    /// Reserved scaffolding for a future saturation constant; no
    /// production code path reads this. See struct-level docs.
    pub k: f32,
}

/// Minkowski pooling exponents.
///
/// Currently unused scaffolding — the pool kernels read
/// `BETA_SPATIAL` / `BETA_BAND` / `BETA_CH` as inlined
/// `const`s in `kernels::pool`. See `CvvdpParams::PLACEHOLDER`.
/// `beta_channel` here corresponds to the kernel's `BETA_CH`
/// (cvvdp `beta_tch`).
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::CvvdpParams;
/// // PLACEHOLDER fills with the scaffolding triple 4.0/4.0/4.0
/// // (uniform Minkowski exponents). Production reads
/// // BETA_SPATIAL=2.0, BETA_BAND=4.0, BETA_CH=4.0 from the
/// // kernel-level consts.
/// let p = CvvdpParams::PLACEHOLDER.pooling;
/// assert_eq!(p.beta_spatial, 4.0);
/// assert_eq!(p.beta_band, 4.0);
/// assert_eq!(p.beta_channel, 4.0);
/// // All three must be positive (negative exponents invert Minkowski
/// // pool semantics). Pinned by tests/params_placeholder_non_display.rs.
/// assert!(p.beta_spatial > 0.0 && p.beta_band > 0.0 && p.beta_channel > 0.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct PoolingParams {
    /// Per-band spatial pooling exponent (Minkowski `beta`).
    pub beta_spatial: f32,
    /// Across-band pooling exponent.
    pub beta_band: f32,
    /// Across-channel pooling exponent.
    pub beta_channel: f32,
}

/// JOD mapping coefficients.
///
/// Currently unused scaffolding — the actual JOD mapping is
/// `kernels::pool::met2jod`, a piecewise function with two
/// regimes joined continuously at `Q = 0.1`:
///
/// - `Q ≤ 0.1`: `JOD = 10 − JOD_A · 0.1^(JOD_EXP − 1) · Q`
///   (linear extension that matches the power curve's slope at the
///   knee, avoiding the zero-derivative singularity at `Q = 0`).
/// - `Q  > 0.1`: `JOD = 10 − JOD_A · Q^JOD_EXP`.
///
/// `JOD_A` (`= 0.0439…`) and `JOD_EXP` (`= 0.9302…`) are inlined
/// as `const`s in `kernels::pool`. The struct fields below map to
/// `jod_a → JOD_A`, `jod_c → JOD_EXP`; `jod_b` is unused (the
/// formula has no separate `b` coefficient). See
/// `CvvdpParams::PLACEHOLDER`.
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::CvvdpParams;
/// // PLACEHOLDER scaffolding values; production reads
/// // JOD_A=0.0439… and JOD_EXP=0.9302… from kernels::pool.
/// // Pinned by tests/params_placeholder_non_display.rs.
/// let j = CvvdpParams::PLACEHOLDER.jod;
/// assert_eq!(j.jod_a, 10.0);
/// assert_eq!(j.jod_b, 1.0);
/// assert!((j.jod_c - 0.30).abs() < 1e-6);
/// // All three positive — required for the future met2jod algebra.
/// assert!(j.jod_a > 0.0 && j.jod_b > 0.0 && j.jod_c > 0.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct JodParams {
    /// JOD mapping scale parameter `a` from cvvdp's
    /// `Q_JOD = 10 - a * Q^b`.
    pub jod_a: f32,
    /// JOD mapping exponent parameter `b`. Production code reads
    /// `kernels::pool::JOD_EXP` instead of this struct field.
    pub jod_b: f32,
    /// Reserved scaffolding parameter `c`. cvvdp v0.5.4's met2jod
    /// formula has no separate `c` coefficient — kept for future
    /// JSON-driven parameter loading.
    pub jod_c: f32,
}

/// Parity-vs-perf trade-off knob for the cvvdp pipeline.
///
/// Most callers want [`PerfMode::Strict`] — that's what every parity
/// test and the v1 R2 manifest are calibrated against. [`PerfMode::Fast`]
/// is the opt-in entry point for future stage-level relaxations that
/// trade measurable per-call cost for a bounded JOD drift versus the
/// strict path (the canonical pycvvdp v0.5.4 reference).
///
/// The variant exists as the public API surface even when no
/// optimization has landed yet — better to design the opt-in once
/// than to add it later and force a breaking change. As individual
/// stages add Fast-mode fast paths they document their drift budget
/// (e.g. nearest-neighbor CSF LUT lookup vs. bilinear) and gate on
/// `self.perf_mode == PerfMode::Fast`. The running list of Fast-mode
/// optimizations lives in `CHANGELOG.md`.
///
/// # Examples
///
/// Default (Strict) builds match all existing parity tests:
///
/// ```
/// use cvvdp_gpu::params::{CvvdpParams, PerfMode};
/// let params = CvvdpParams::PLACEHOLDER;
/// assert_eq!(params.perf_mode, PerfMode::Strict);
/// ```
///
/// Opt into Fast mode by overriding the field:
///
/// ```
/// use cvvdp_gpu::params::{CvvdpParams, PerfMode};
/// let params = CvvdpParams {
///     perf_mode: PerfMode::Fast,
///     ..CvvdpParams::PLACEHOLDER
/// };
/// assert_eq!(params.perf_mode, PerfMode::Fast);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PerfMode {
    /// Match pycvvdp v0.5.4 bit-for-bit within f32 noise. Every
    /// parity test in `tests/` is calibrated against this mode.
    /// Pyramid downscale/upscale, CSF apply, masking, and pool all
    /// run the canonical (slower, higher-precision) path.
    #[default]
    Strict,
    /// Opt in to stage-level relaxations that trade measurable
    /// per-call cost for a bounded JOD drift vs. Strict. Currently
    /// a no-op — no Fast-mode fast paths have landed yet. The
    /// variant exists so callers can wire the opt-in once; future
    /// per-stage optimizations gate on `perf_mode == Fast` and
    /// document their individual drift budget in `CHANGELOG.md`.
    Fast,
}

/// Top-level cvvdp parameter bundle.
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::{CvvdpParams, PerfMode};
/// use cvvdp_gpu::params::DisplayModel;
///
/// // Most callers want PLACEHOLDER — STANDARD_4K display + Strict
/// // perf mode + scaffolding scalars for the not-yet-wired fields.
/// let p = CvvdpParams::PLACEHOLDER;
/// assert_eq!(p.perf_mode, PerfMode::Strict);
///
/// // Override a single field via struct-update syntax — e.g.
/// // opt into a custom display while keeping all other defaults.
/// let p_hdr = CvvdpParams {
///     display: DisplayModel { y_peak: 1000.0, ..p.display },
///     ..p
/// };
/// assert_eq!(p_hdr.display.y_peak, 1000.0);
/// assert_eq!(p_hdr.perf_mode, PerfMode::Strict); // unchanged
/// ```
#[derive(Debug, Clone, Copy)]
pub struct CvvdpParams {
    /// Photometric display model: peak / black luminance, ambient
    /// reflection. Consumed by the color stage's sRGB→linear-cd/m²
    /// conversion.
    pub display: DisplayModel,
    /// castleCSF scaffolding parameters. **Unused** by the
    /// production code — CSF runs from the vendored
    /// `kernels::csf::csf_lut/v0_5_4.rs` LUT. See `PLACEHOLDER`.
    pub csf: CsfParams,
    /// Contrast-masking scaffolding parameters. **Unused** by the
    /// production code — masking runs from the inline `const`s in
    /// `kernels::masking`. See `PLACEHOLDER`.
    pub masking: MaskingParams,
    /// Spatial / band / channel pooling scaffolding parameters.
    /// **Unused** by the production code — pooling reads the
    /// `BETA_SPATIAL` / `BETA_BAND` / `BETA_CH` const triple in
    /// `kernels::pool`. See `PLACEHOLDER`.
    pub pooling: PoolingParams,
    /// JOD-mapping scaffolding parameters. **Unused** by the
    /// production code — `met2jod` reads `JOD_A` and `JOD_EXP` from
    /// `kernels::pool`. See `PLACEHOLDER`.
    pub jod: JodParams,
    /// Parity-vs-perf trade-off. Defaults to [`PerfMode::Strict`]
    /// via [`CvvdpParams::PLACEHOLDER`]. See [`PerfMode`] for the
    /// opt-in mechanics.
    pub perf_mode: PerfMode,
}

impl CvvdpParams {
    /// Default parameter bundle. The `display` field is read by the
    /// host scalar (`predict_jod_still_3ch` uses
    /// `display.y_peak/y_black/y_refl`) and the GPU color kernel; the
    /// `csf`/`masking`/`pooling`/`jod` sub-bundles are currently
    /// **unused** because the per-stage cvvdp v0.5.4 numbers are
    /// inlined as `const`s in `kernels::pool` (`BETA_SPATIAL` /
    /// `BETA_BAND` / `BETA_CH` / `IMAGE_INT` / `PER_CH_W` /
    /// `BASEBAND_W`), `kernels::masking` (`MASK_P` / `MASK_Q` /
    /// `MASK_C` / `D_MAX` / `XCM_3X3` / `CH_GAIN` /
    /// `PU_BLUR_KERNEL_1D` / `PU_PADSIZE`), and
    /// `kernels::pool::met2jod` (`JOD_A` / `JOD_EXP`). The fields
    /// exist as scaffolding for the planned "load parameters from
    /// the vendored cvvdp JSON" path which hasn't landed; the
    /// placeholder numbers below are approximate but won't affect
    /// any test because no code path reads them.
    ///
    /// # Examples
    ///
    /// ```
    /// use cvvdp_gpu::{CvvdpParams, PerfMode};
    /// use cvvdp_gpu::params::DisplayModel;
    ///
    /// // PLACEHOLDER pins the parity-test baseline. Pinned by
    /// // tests/params_placeholder.rs (display + perf_mode) and
    /// // tests/params_placeholder_non_display.rs (scaffolding subs).
    /// let p = CvvdpParams::PLACEHOLDER;
    /// assert_eq!(p.display.y_peak, DisplayModel::STANDARD_4K.y_peak);
    /// assert_eq!(p.perf_mode, PerfMode::Strict);
    /// ```
    ///
    /// Prefer `CvvdpParams::default()` in new code — it returns the
    /// same values and reads more idiomatically. `PLACEHOLDER` is
    /// kept as a `const` for callers that need a const context
    /// (`const`-initialized statics, match-arm constants, etc.) where
    /// `Default::default()` isn't usable.
    pub const PLACEHOLDER: Self = Self {
        display: DisplayModel::STANDARD_4K,
        csf: CsfParams {
            a_peak: 0.0,
            rg_peak: 0.0,
            vy_peak: 0.0,
        },
        masking: MaskingParams {
            p: 2.4,
            q: 2.2,
            k: 0.04,
        },
        pooling: PoolingParams {
            beta_spatial: 4.0,
            beta_band: 4.0,
            beta_channel: 4.0,
        },
        jod: JodParams {
            jod_a: 10.0,
            jod_b: 1.0,
            jod_c: 0.30,
        },
        perf_mode: PerfMode::Strict,
    };
}

/// Default returns the canonical `standard_4k` display geometry +
/// `PerfMode::Strict` — what every v1 R2 parity golden was captured
/// against. Bit-identical to [`CvvdpParams::PLACEHOLDER`].
///
/// Most callers should configure for their target display before
/// computing — the perceptual model is sensitive to peak luminance,
/// black level, ambient, and viewing distance. Use the default for
/// quick-start work and parity tests that match the standard_4k
/// reference; replace the [`DisplayModel`] field for any production
/// scoring where the actual viewing conditions are known.
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::{CvvdpParams, PerfMode};
///
/// // Default and PLACEHOLDER are bit-identical.
/// let d = CvvdpParams::default();
/// let p = CvvdpParams::PLACEHOLDER;
/// assert_eq!(d.display.y_peak, p.display.y_peak);
/// assert_eq!(d.display.y_black, p.display.y_black);
/// assert_eq!(d.display.y_refl, p.display.y_refl);
/// assert_eq!(d.perf_mode, p.perf_mode);
/// assert_eq!(d.perf_mode, PerfMode::Strict);
/// ```
impl Default for CvvdpParams {
    fn default() -> Self {
        Self::PLACEHOLDER
    }
}
