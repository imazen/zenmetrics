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

/// Display model: how sRGB bytes are mapped to physical luminance
/// (cd/m²) before perceptual processing.
///
/// Matches cvvdp's `vvdp_display_photo_eotf.forward` for the sRGB EOTF
/// branch:
///
/// ```text
/// L_rgb = (y_peak - y_black) * srgb2lin(byte / 255) + y_black + y_refl
/// ```
#[derive(Debug, Clone, Copy)]
pub struct DisplayModel {
    /// Peak display luminance in cd/m².
    pub y_peak: f32,
    /// Black level in cd/m² (`y_peak / contrast`).
    pub y_black: f32,
    /// Light reflected from the screen, precomputed host-side from
    /// `E_ambient / π * k_refl`.
    pub y_refl: f32,
}

impl DisplayModel {
    /// cvvdp's `standard_4k` defaults — peak 200 cd/m², contrast 1000,
    /// 250 lux ambient, k_refl 0.005, sRGB EOTF. The v1 R2 goldens
    /// were captured under this display.
    pub const STANDARD_4K: Self = Self {
        y_peak: 200.0,
        y_black: 0.2,
        y_refl: 0.397_887_36,
    };
}

/// Display geometry — resolution + viewing distance + physical size.
/// Used to derive pixels-per-degree, which the CSF stage consumes via
/// each pyramid band's spatial frequency.
///
/// Matches cvvdp's `vvdp_display_geometry` for the `diagonal_inches +
/// distance_m` path (the one the `standard_4k` JSON uses). Other
/// cvvdp paths (`fov_diagonal`, `fov_horizontal`, etc.) are not
/// ported until a use case appears.
#[derive(Debug, Clone, Copy)]
pub struct DisplayGeometry {
    pub resolution_w: u32,
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
    pub const STANDARD_4K: Self = Self {
        resolution_w: 3840,
        resolution_h: 2160,
        distance_m: 0.7472,
        diagonal_inches: 30.0,
    };

    /// Pixels-per-degree at the display centre (eccentricity = 0).
    /// Matches cvvdp's `vvdp_display_geometry.get_ppd()` for the
    /// no-eccentricity path.
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
pub const SRGB_LINEAR_TO_DKL: [[f32; 3]; 3] = [
    [0.233_201_21, 0.728_830_8, 0.088_995_87],
    [0.127_620_77, -0.087_068_09, -0.036_777_39],
    [-0.214_822_5, -0.626_253_7, 0.851_403_3],
];

/// castleCSF achromatic + chrom params. Scaffolding for a planned
/// "load from vendored cvvdp JSON" path that hasn't landed; the
/// production code reads CSF sensitivity straight from the
/// `csf_lut_weber_fixed_size` LUT vendored in `kernels/csf_lut/`,
/// not from this struct. See `CvvdpParams::PLACEHOLDER` for the
/// full unused-scaffolding picture.
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
/// kernels read `MASK_P`, `MASK_Q`, `MASK_C`, and `XCM_3X3` as
/// inlined `const`s in `kernels::masking`. See
/// `CvvdpParams::PLACEHOLDER` for the full picture.
#[derive(Debug, Clone, Copy)]
pub struct MaskingParams {
    /// Excitation exponent (cvvdp `p`).
    pub p: f32,
    /// Inhibition exponent (cvvdp `q`).
    pub q: f32,
    /// Saturation constant (cvvdp `epsilon` / `k`).
    pub k: f32,
}

/// Minkowski pooling exponents.
///
/// Currently unused scaffolding — the pool kernels read
/// `BETA_SPATIAL` / `BETA_BAND` / `BETA_CH` as inlined
/// `const`s in `kernels::pool`. See `CvvdpParams::PLACEHOLDER`.
/// `beta_channel` here corresponds to the kernel's `BETA_CH`
/// (cvvdp `beta_tch`).
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
#[derive(Debug, Clone, Copy)]
pub struct JodParams {
    pub jod_a: f32,
    pub jod_b: f32,
    pub jod_c: f32,
}

/// Top-level cvvdp parameter bundle.
#[derive(Debug, Clone, Copy)]
pub struct CvvdpParams {
    pub display: DisplayModel,
    pub csf: CsfParams,
    pub masking: MaskingParams,
    pub pooling: PoolingParams,
    pub jod: JodParams,
}

impl CvvdpParams {
    /// Default parameter bundle. The `display` field is read by the
    /// host scalar (`predict_jod_still_3ch` uses
    /// `display.y_peak/y_black/y_refl`) and the GPU color kernel; the
    /// `csf`/`masking`/`pooling`/`jod` sub-bundles are currently
    /// **unused** because the per-stage cvvdp v0.5.4 numbers are
    /// inlined as `const`s in `kernels::pool` (BETA_*),
    /// `kernels::masking` (MASK_P / MASK_Q / MASK_C / XCM_3X3), and
    /// `kernels::pool::met2jod` (JOD_A / JOD_EXP). The fields exist
    /// as scaffolding for the planned "load parameters from the
    /// vendored cvvdp JSON" path which hasn't landed; the placeholder
    /// numbers below are approximate but won't affect any test
    /// because no code path reads them.
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
    };
}
