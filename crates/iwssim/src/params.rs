//! Configuration knobs for [`crate::Iwssim`].
//!
//! Mirrors the Python reference's `config.py` defaults so a vanilla
//! `Iwssim::new` reproduces the upstream's score-for-score behavior.

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
        }
    }

    /// Construct with [`allow_small`](Self::allow_small) set explicitly.
    pub const fn allow_small(allow: bool) -> Self {
        let mut p = Self::new();
        p.allow_small = allow;
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
