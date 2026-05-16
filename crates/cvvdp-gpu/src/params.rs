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
/// // updating from STANDARD_4K:
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
    /// `E_ambient / π * k_refl`.
    pub y_refl: f32,
}

impl DisplayModel {
    /// cvvdp's `standard_4k` defaults — peak 200 cd/m², contrast 1000,
    /// 250 lux ambient, k_refl 0.005, sRGB EOTF. The v1 R2 goldens
    /// were captured under this display.
    ///
    /// # Examples
    ///
    /// ```
    /// use cvvdp_gpu::params::DisplayModel;
    /// // Pinned by tests/display_geometry.rs::display_model_standard_4k_matches_pycvvdp_v0_5_4
    /// // — all three fields bit-pinned via `.to_bits()`.
    /// let s = DisplayModel::STANDARD_4K;
    /// assert_eq!(s.y_peak, 200.0);
    /// assert_eq!(s.y_black, 0.2);
    /// // y_refl is precomputed host-side from 250 lux × 0.005 / π.
    /// assert!((s.y_refl - 0.397_887_36).abs() < 1e-6);
    /// ```
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
