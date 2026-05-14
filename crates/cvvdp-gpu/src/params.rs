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
#[derive(Debug, Clone, Copy)]
pub struct DisplayModel {
    /// Peak display luminance in cd/m². cvvdp default: 100.0 (sRGB).
    pub y_peak: f32,
    /// Black level in cd/m² (display contrast = `y_peak / y_black`).
    pub y_black: f32,
    /// Ambient illumination reflected off the screen (cd/m²).
    pub y_refl: f32,
    /// Display EOTF gamma. cvvdp default for sRGB-byte input: the sRGB
    /// piecewise EOTF (signaled by `is_srgb = true`).
    pub gamma: f32,
    /// Flag selecting the sRGB piecewise EOTF vs pure power-law gamma.
    pub is_srgb: bool,
}

impl DisplayModel {
    /// cvvdp's published sRGB / standard-display defaults.
    pub const SRGB_STD: Self = Self {
        y_peak: 100.0,
        y_black: 0.5,
        y_refl: 0.0,
        gamma: 2.2,
        is_srgb: true,
    };
}

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
        display: DisplayModel::SRGB_STD,
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
