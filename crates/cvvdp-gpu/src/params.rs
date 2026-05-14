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

/// castleCSF achromatic + chrom params. Concrete numbers come from the
/// vendored cvvdp JSON; placeholders until pin.
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
#[derive(Debug, Clone, Copy)]
pub struct PoolingParams {
    /// Per-band spatial pooling exponent (Minkowski `beta`).
    pub beta_spatial: f32,
    /// Across-band pooling exponent.
    pub beta_band: f32,
    /// Across-channel pooling exponent.
    pub beta_channel: f32,
}

/// JOD mapping: `JOD = jod_a - jod_b * D^jod_c`.
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
    /// Placeholder defaults. Replace with values loaded from the
    /// vendored cvvdp JSON once the reference is pinned. Tests that
    /// require parity must not use these defaults — they should
    /// construct `CvvdpParams` from the vendored numbers explicitly.
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
