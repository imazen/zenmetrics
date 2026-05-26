//! `MemoryMode` + `CachedRefStripPolicy` — umbrella-level policy enums
//! that callers pass through [`crate::Metric::new_with_memory_mode`].
//!
//! ## Why these live here (not in `zenmetrics-core`)
//!
//! Each per-crate `MemoryMode` enum has the same *shape* (`Auto`,
//! `Full`, `Strip { h_body }`, `Tile { h, w }`) but the per-crate
//! `resolve_auto` policies, error-message advice strings, and a few
//! variant fields (cvvdp adds `capped_levels` to its `Strip` variant)
//! differ in meaningful ways. Hoisting the enum to a shared crate would
//! either force every callsite through `From` conversions or force a
//! single canonical shape that lies about cvvdp's capped-pyramid mode.
//!
//! Instead the umbrella owns the *user-facing* policy enum and converts
//! at the per-crate boundary inside [`crate::Metric::new_with_memory_mode`].
//! Per-crate code keeps its own `MemoryMode` enum and its own
//! `resolve_auto` — the umbrella never sees those.

/// Memory-budget policy passed to [`crate::Metric::new_with_memory_mode`].
///
/// Per-crate implementations interpret these variants according to their
/// own working-set math:
///
/// - [`Self::Auto`]: each crate's `resolve_auto` picks the largest
///   variant that fits the VRAM cap (env var `ZENMETRICS_VRAM_CAP_BYTES`
///   → cubecl free-VRAM query → 8 GB default).
/// - [`Self::Full`]: whole-image working set on device.
/// - [`Self::Strip { h_body }`]: process vertical strips of `h_body`
///   rows + halo; `h_body = None` lets the crate pick a default.
/// - [`Self::Tile { h, w }`]: not yet implemented in any per-crate
///   pipeline — reserved for future work.
///
/// Callers default to [`Self::Auto`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryMode {
    /// Per-crate `resolve_auto` picks the variant that fits the cap.
    Auto,
    /// Whole-image working set on device.
    Full,
    /// Vertical strips of `h_body` rows + halo. `None` → crate-default.
    Strip {
        /// Strip body height in rows (not counting halo). `None` lets
        /// the per-crate `resolve_auto` pick.
        h_body: Option<u32>,
    },
    /// Reserved — not yet implemented in any per-crate pipeline.
    Tile {
        /// Tile height in rows.
        h: u32,
        /// Tile width in columns.
        w: u32,
    },
}

impl Default for MemoryMode {
    fn default() -> Self {
        Self::Auto
    }
}

/// Cached-reference strip-mode policy passed through [`crate::MetricParams`].
///
/// When a metric is constructed in [`MemoryMode::Strip`] AND
/// [`crate::Metric::set_reference_srgb_u8`] is called, two valid
/// implementations exist:
///
/// - [`Self::RefFull`]: keep whole-image ref-side state alive on device;
///   each dist call allocates only one strip's working set. Peak =
///   `ref_full + one_strip_dist`. Simpler, more memory.
/// - [`Self::BothStripped`]: walk ref strip-by-strip and cache per-strip
///   ref state; dist walks the same strip layout. Peak per-call = one
///   strip; persistent cache ≈ full ref pyramid sliced. Best at large
///   sizes (24 MP+).
///
/// [`Self::Auto`] picks based on the per-crate VRAM-cap policy.
///
/// In [`MemoryMode::Full`] this enum is ignored (cached-ref always uses
/// the whole-image state). Today only iwssim ships the
/// [`Self::BothStripped`] cached-ref-strip path; per-crate
/// implementations may surface [`crate::Error::Metric`] indicating
/// fall-back to one-shot when they don't yet support
/// [`Self::BothStripped`] under [`MemoryMode::Strip`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CachedRefStripPolicy {
    /// VRAM-aware default — picks `RefFull` when `ref_full + one_strip
    /// fits`, else `BothStripped`.
    Auto,
    /// Hold whole-image ref-side state on device; dist strips per-call.
    RefFull,
    /// Per-strip ref cache + per-strip dist (iwssim's pattern).
    BothStripped,
}

impl Default for CachedRefStripPolicy {
    fn default() -> Self {
        Self::Auto
    }
}

// ---------------------------------------------------------------
// Per-crate `From` conversions.
//
// These keep the umbrella API stable: callers pass the umbrella's
// `MemoryMode`; per-crate constructors receive their own crate's
// `MemoryMode` after conversion at the call site inside
// `Metric::new_with_memory_mode`. The umbrella never sees the per-crate
// enums; conversions live here so the per-crate code stays unchanged.
// ---------------------------------------------------------------

#[cfg(feature = "butter")]
impl From<MemoryMode> for butteraugli_gpu::MemoryMode {
    fn from(m: MemoryMode) -> Self {
        match m {
            MemoryMode::Auto => butteraugli_gpu::MemoryMode::Auto,
            MemoryMode::Full => butteraugli_gpu::MemoryMode::Full,
            MemoryMode::Strip { h_body } => butteraugli_gpu::MemoryMode::Strip { h_body },
            MemoryMode::Tile { h: _, w: _ } => butteraugli_gpu::MemoryMode::Auto,
        }
    }
}

#[cfg(feature = "ssim2")]
impl From<MemoryMode> for ssim2_gpu::MemoryMode {
    fn from(m: MemoryMode) -> Self {
        match m {
            MemoryMode::Auto => ssim2_gpu::MemoryMode::Auto,
            MemoryMode::Full => ssim2_gpu::MemoryMode::Full,
            MemoryMode::Strip { h_body } => ssim2_gpu::MemoryMode::Strip { h_body },
            MemoryMode::Tile { h: _, w: _ } => ssim2_gpu::MemoryMode::Auto,
        }
    }
}

#[cfg(feature = "dssim")]
impl From<MemoryMode> for dssim_gpu::MemoryMode {
    fn from(m: MemoryMode) -> Self {
        match m {
            MemoryMode::Auto => dssim_gpu::MemoryMode::Auto,
            MemoryMode::Full => dssim_gpu::MemoryMode::Full,
            MemoryMode::Strip { h_body } => dssim_gpu::MemoryMode::Strip { h_body },
            MemoryMode::Tile { h: _, w: _ } => dssim_gpu::MemoryMode::Auto,
        }
    }
}

#[cfg(feature = "iwssim")]
impl From<MemoryMode> for iwssim_gpu::MemoryMode {
    fn from(m: MemoryMode) -> Self {
        match m {
            MemoryMode::Auto => iwssim_gpu::MemoryMode::Auto,
            MemoryMode::Full => iwssim_gpu::MemoryMode::Full,
            MemoryMode::Strip { h_body } => iwssim_gpu::MemoryMode::Strip { h_body },
            MemoryMode::Tile { h: _, w: _ } => iwssim_gpu::MemoryMode::Auto,
        }
    }
}

// cvvdp + zensim only support Full + Auto today; Strip/Tile fall back
// to Auto so callers get the closest-meaning policy without an error
// at the umbrella boundary. Per-crate `new_with_memory_mode` already
// surfaces a clear error if the resolved mode isn't supported.

#[cfg(feature = "cvvdp")]
impl From<MemoryMode> for cvvdp_gpu::MemoryMode {
    fn from(m: MemoryMode) -> Self {
        match m {
            MemoryMode::Auto => cvvdp_gpu::MemoryMode::Auto,
            MemoryMode::Full => cvvdp_gpu::MemoryMode::Full,
            // cvvdp's Strip variant is `Strip { h_body, capped_levels }`;
            // umbrella callers only pass `h_body` so capped_levels stays
            // at its per-crate default.
            MemoryMode::Strip { h_body } => cvvdp_gpu::MemoryMode::Strip {
                h_body,
                capped_levels: None,
            },
            MemoryMode::Tile { h: _, w: _ } => cvvdp_gpu::MemoryMode::Auto,
        }
    }
}

#[cfg(feature = "zensim")]
impl From<MemoryMode> for zensim_gpu::MemoryMode {
    fn from(m: MemoryMode) -> Self {
        match m {
            MemoryMode::Auto => zensim_gpu::MemoryMode::Auto,
            MemoryMode::Full => zensim_gpu::MemoryMode::Full,
            MemoryMode::Strip { h_body } => zensim_gpu::MemoryMode::Strip { h_body },
            MemoryMode::Tile { h: _, w: _ } => zensim_gpu::MemoryMode::Auto,
        }
    }
}
