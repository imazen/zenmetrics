//! Configuration knobs for [`crate::Iwssim`].
//!
//! Mirrors the Python reference's `config.py` defaults so a vanilla
//! `Iwssim::new` reproduces the upstream's score-for-score behavior.

/// Luma extraction convention for [`Iwssim::score`] — how packed
/// sRGB-u8 input becomes the gray plane the pyramid runs on.
///
/// The two conventions exist because the two reference
/// implementations differ upstream:
///
/// - Python-IW-SSIM's `utils.rgb2gray` **rounds** to integers:
///   `np.round(0.2989·R + 0.5870·G + 0.1140·B)`.
/// - piq's `information_weighted_ssim` feeds `rgb2yiq(x)[:, :1]` —
///   **unrounded** `0.299·R + 0.587·G + 0.114·B` on the 0–255 scale
///   (the convention the JPEG AIC-4 `IW-SSIM` column was published
///   under — verified 2026-09-25 vs `metrics_fullres.csv`,
///   med|Δ| = 3.9e-6 over the 54-pair CLI subset).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LumaConvention {
    /// `round(0.2989·R + 0.5870·G + 0.1140·B)` — Python-IW-SSIM
    /// `utils.rgb2gray`. The default, matching the crate's oracle.
    #[default]
    Bt601Rounded,
    /// `0.299·R + 0.587·G + 0.114·B` unrounded f32 — piq `rgb2yiq`
    /// channel 0 on the 0–255 scale.
    YiqUnrounded,
}

/// Knobs surfaced from the Python reference's `config.py`.
///
/// Defaults match the upstream `cfg`:
/// ```text
/// iw_flag   = True
/// Nsc       = 5
/// blSzX     = 3
/// blSzY     = 3
/// parent    = True
/// sigma_nsq = 0.4
/// ```
///
/// Plus a small-image escape hatch (`allow_small`) borrowed from
/// `iwssim-gpu`: when true, sub-176-px inputs are tiled up to
/// `MIN_NATIVE_DIM` on the short axis instead of rejected.
#[derive(Debug, Clone, Copy)]
pub struct IwssimParams {
    /// Include IW pooling (`true`, default) — the metric's whole point.
    /// Setting `false` reduces it to plain MS-SSIM (each scale pooled
    /// by unweighted mean instead of `Σ(cs · iw) / Σ(iw)`).
    pub iw_flag: bool,
    /// Neighborhood block size in X (paper `blSzX`, default `3`).
    /// Must equal `blSzY` for the current implementation.
    pub bl_sz_x: u32,
    /// Neighborhood block size in Y (paper `blSzY`, default `3`).
    /// Must equal `bl_sz_x` for the current implementation.
    pub bl_sz_y: u32,
    /// Include the coarser-scale parent band in the neighborhood
    /// covariance (`true`, default). Disabling this shrinks `N` from
    /// 10 → 9 at all scales except the second-coarsest.
    pub parent: bool,
    /// HVS noise variance σ²_nsq (paper §II-C, default `0.4`).
    pub sigma_nsq: f32,
    /// Accept sub-176-px inputs by tiling up to the minimum dim. Default
    /// is `false` — return [`crate::Error::InvalidImageSize`].
    pub allow_small: bool,
    /// Luma extraction for [`crate::Iwssim::score`] — see
    /// [`LumaConvention`]. Default [`LumaConvention::Bt601Rounded`]
    /// (the Python reference's `rgb2gray`). [`LumaConvention::YiqUnrounded`]
    /// reproduces the piq / MATLAB-on-float-luma convention the AIC-4
    /// `IW-SSIM` column used.
    pub luma: LumaConvention,
}

impl Default for IwssimParams {
    fn default() -> Self {
        Self {
            iw_flag: true,
            bl_sz_x: 3,
            bl_sz_y: 3,
            parent: true,
            sigma_nsq: 0.4,
            allow_small: false,
            luma: LumaConvention::default(),
        }
    }
}

impl IwssimParams {
    /// Construct with all upstream defaults — matches Python's
    /// `IW_SSIM(iw_flag=True, Nsc=5, blSzX=3, blSzY=3, parent=True, sigma_nsq=0.4)`.
    pub const fn new() -> Self {
        Self {
            iw_flag: true,
            bl_sz_x: 3,
            bl_sz_y: 3,
            parent: true,
            sigma_nsq: 0.4,
            allow_small: false,
            luma: LumaConvention::Bt601Rounded,
        }
    }

    /// Construct with [`allow_small`](Self::allow_small) set explicitly.
    pub const fn allow_small(allow: bool) -> Self {
        let mut p = Self::new();
        p.allow_small = allow;
        p
    }

    /// The piq / MATLAB-column convention: same algorithm knobs, but
    /// the RGB→luma ingress is unrounded `0.299/0.587/0.114` (piq's
    /// `rgb2yiq` channel 0) instead of the Python reference's rounded
    /// BT.601. This is the variant the JPEG AIC-4 `IW-SSIM` column
    /// was computed under (verified 2026-09-25: med|Δ| = 3.9e-6 vs
    /// `metrics_fullres.csv` on the 54-pair CLI subset).
    pub const fn piq_luma() -> Self {
        let mut p = Self::new();
        p.luma = LumaConvention::YiqUnrounded;
        p
    }

    /// Derived: `bound = ceil((winsize-1) / 2)` = `5` for 11-tap window.
    #[inline]
    pub(crate) const fn bound(&self) -> u32 {
        // (11 - 1) / 2 = 5 (winsize hard-coded to 11 in the paper).
        5
    }

    /// Derived: `bound1 = bound - floor((blSzX-1)/2)` = `5 - 1 = 4` for
    /// the default `blSzX=3`. Used to crop the IW weight map down to
    /// the SSIM cs-map's spatial extent.
    #[inline]
    pub(crate) const fn bound1(&self) -> u32 {
        // bound - ((blSzX-1)/2) — Python uses floor division.
        self.bound() - ((self.bl_sz_x - 1) / 2)
    }
}
