//! Public `Cvvdp` scorer + the end-to-end pipeline orchestration.
//!
//! Mirrors `crate::host_scalar::predict_jod_still_3ch_capped`
//! algorithmically — the host-scalar path IS the f32-precision
//! reference the GPU pipeline is validated against. We re-use cvvdp-gpu's
//! constants and per-pixel masking/CSF helpers verbatim. The CPU-port
//! contribution is structural: persistent scratch + diffmap output +
//! optional rayon outer parallelism.

use alloc::vec::Vec;

use crate::color::{linear_planes_to_dkl_planar, srgb_to_dkl_planar};
use crate::diffmap::{DiffmapAccum, accumulate_band_diffmap, finalize_diffmap};
use crate::pool::{
    BASEBAND_W, BETA_BAND, BETA_CH, BETA_SPATIAL, IMAGE_INT, PER_CH_W,
    do_pooling_and_jod_still_3ch, lp_norm_mean,
};
use crate::pyramid::{WeberPyramid, WeberPyramidCache, band_frequencies, weber_contrast_pyr_into};
use crate::scratch::{BandWorkspace, Scratch, StripBandWorkspace};
use crate::strip::{LpNormAccumulator, mode_b_halo_at_level};
use crate::{CvvdpParams, DisplayGeometry, Error, Result};

use crate::kernels::csf::{
    CSF_BASEBAND_RHO, CsfChannel, LOG_L_BKG_AXIS, N_L_BKG, SENSITIVITY_CORRECTION_DB,
    precompute_logs_row,
};
use crate::kernels::masking::{CH_GAIN, D_MAX, MASK_C, MASK_P, MASK_Q, PU_PADSIZE, XCM_3X3};
use crate::simd_math::safe_pow_with_offset_into;
use crate::simd_pyramid::gaussian_blur_sigma3_simd;

use crate::masking::mult_mutual_band_into;

/// Sensitivity correction in log10 space, premultiplied so we can
/// add to `log_s` before the `10^x` step (matches the GPU 3ch fused
/// kernel's `log_correction` constant).
const LOG_SENSITIVITY_CORRECTION: f32 = SENSITIVITY_CORRECTION_DB / 20.0;
/// `1.0 / (LOG_L_BKG_AXIS[N-1] - LOG_L_BKG_AXIS[0]) * (N - 1)` = 31 /
/// 6.30103 ≈ 4.919830570... The CSF L_bkg axis is uniform in log10
/// space.
const CSF_L_BKG_INV_STEP: f32 = 4.919_830_6;
const CSF_L_BKG_AXIS_MIN: f32 = -2.301_03;
const CSF_L_BKG_MAX_IDX: f32 = 30.999_999;

/// Per-pixel CSF apply via a precomputed `logs_row[N_L_BKG]`.
///
/// Replicates `csf_apply_3ch_kernel`'s per-pixel arithmetic
/// (uniform-axis bracket index from `(log_l - min) / step`, linear
/// interp into the row, add the constant correction in log space,
/// then `exp(log_s · ln 10)`). The rho-axis interp has already been
/// folded into `logs_row` by `precompute_logs_row`, so the inner
/// loop here has NO binary searches and the whole CSF evaluation
/// reduces to 2 indexed reads + a linear combine + a single
/// `f32::exp` per pixel per channel.
#[inline]
fn apply_csf_row_per_pixel(log_l: f32, logs_row: &[f32; N_L_BKG]) -> f32 {
    let off_raw = (log_l - CSF_L_BKG_AXIS_MIN) * CSF_L_BKG_INV_STEP;
    let off_lo = off_raw.clamp(0.0, CSF_L_BKG_MAX_IDX);
    let lo_idx_f = off_lo.floor();
    let frac = off_lo - lo_idx_f;
    let lo_idx = lo_idx_f as usize;
    let hi_idx = lo_idx + 1;
    let lo = logs_row[lo_idx];
    let hi = logs_row[hi_idx];
    let log_s_raw = lo + frac * (hi - lo);
    let log_s = log_s_raw + LOG_SENSITIVITY_CORRECTION;
    // exp(log_s * ln 10) == 10^log_s
    (log_s * core::f32::consts::LN_10).exp()
}

/// CPU scorer for cvvdp still-image JOD.
///
/// Construct once for a given `(width, height)`, then call `score`
/// (or `warm_reference` + `score_with_warm_ref` for buttloop reuse).
///
/// Internally holds:
///
/// - Persistent scratch buffers sized to the image so per-call
///   allocations are minimal.
/// - The cached warm-reference state when `warm_reference()` was
///   invoked.
///
/// All API functions are `&mut self` because the scratch is mutated;
/// concurrent scoring of different images requires one `Cvvdp` per
/// thread. Rayon-style outer parallelism wraps `Cvvdp::new` per
/// thread, not per call.
pub struct Cvvdp {
    width: usize,
    height: usize,
    params: CvvdpParams,
    ppd: f32,
    scratch: Scratch,
    /// `true` after a successful `warm_reference` call until the next
    /// cold `score*` call clears the cache. The cached DKL planes
    /// live in `scratch.ref_a/ref_rg/ref_vy` and the cached weber
    /// pyramid in `scratch.weber_ref` — the boolean just gates
    /// `score_with_warm_ref*` against using stale or never-populated
    /// scratch slots. Phase 9.YA: replaces the prior
    /// `warm: Option<ReferenceState>` which double-allocated 480 MB
    /// of DKL planes at 40 MP per `warm_reference` call.
    warm_active: bool,
    /// Phase 9.Z.B (#124): when `Some(h_body)`, the band-fold's spatial
    /// Minkowski pool partitions each band's `d` array into row-strips
    /// of `(h_body >> k).max(1)` rows and accumulates `Σ safe_pow_lp`
    /// across strips. Bit-identical to single-pass `lp_norm_mean`
    /// because spatial Minkowski is associative under row-order
    /// dispatch (see `crate::strip::LpNormAccumulator` doc). Mirrors
    /// the GPU's shipped pool-stage K_SPLIT walker (Phase 3 Approach B
    /// incremental landing, see `cvvdp-gpu/docs/STRIP_PROCESSING.md`).
    ///
    /// Memory impact in this incremental landing: ZERO. Like the GPU
    /// today, only the pool stage iterates in strips; the full
    /// weber pyramid + d_scratch + bands_ref + bands_dis remain
    /// full-image-sized. The pool stage is a tiny fraction of the
    /// working set. This landing proves the walker is correct
    /// end-to-end (strip accumulator associativity + per-strip
    /// iteration + counter visibility). The per-strip pyramid kernels
    /// are gated on a future chunk — same status as the GPU.
    strip_h_body: core::cell::Cell<Option<u32>>,
    /// Counts the number of strip iterations dispatched in the most
    /// recent strip-mode `score*` call. Visible via
    /// [`Self::strip_dispatch_counter`]. Tests assert N >= 2 strip
    /// iterations at sizes large enough to actually partition. Reset
    /// to 0 at the start of each strip-mode call.
    strip_dispatch_counter: core::sync::atomic::AtomicU32,
}

impl Cvvdp {
    /// Create a new scorer for `width × height` images using the
    /// given parameter bundle. PPD is derived from the parameter
    /// bundle's display geometry (defaults to `STANDARD_4K`).
    pub fn new(width: u32, height: u32, params: CvvdpParams) -> Result<Self> {
        Self::with_geometry(width, height, params, DisplayGeometry::STANDARD_4K)
    }

    /// Same as `new` but with an explicit display geometry. Use this
    /// to score images that have a known on-screen size + viewing
    /// distance different from the cvvdp default.
    pub fn with_geometry(
        width: u32,
        height: u32,
        params: CvvdpParams,
        geometry: DisplayGeometry,
    ) -> Result<Self> {
        if width < 8 || height < 8 {
            return Err(Error::InvalidImageSize { width, height });
        }
        let w = width as usize;
        let h = height as usize;
        let ppd = geometry.pixels_per_degree();
        // Pre-compute n_levels so Scratch can pre-allocate every
        // per-level weber pyramid Vec<f32>. Phase 9.YA Part 2.
        let n_levels = band_frequencies(ppd, w, h).len();
        Ok(Self {
            width: w,
            height: h,
            params,
            ppd,
            scratch: Scratch::new(w, h, n_levels),
            warm_active: false,
            strip_h_body: core::cell::Cell::new(None),
            strip_dispatch_counter: core::sync::atomic::AtomicU32::new(0),
        })
    }

    /// Image width.
    pub fn width(&self) -> u32 {
        self.width as u32
    }

    /// Image height.
    pub fn height(&self) -> u32 {
        self.height as u32
    }

    /// Configured pixels-per-degree.
    pub fn pixels_per_degree(&self) -> f32 {
        self.ppd
    }

    /// One-shot score: sRGB-byte REF + DIST, returns JOD ∈ `[0, 10]`.
    pub fn score(&mut self, ref_srgb: &[u8], dist_srgb: &[u8]) -> Result<f32> {
        self.check_srgb(ref_srgb)?;
        self.check_srgb(dist_srgb)?;
        self.warm_active = false;
        // Convert both sides to DKL planar.
        let display = self.params.display;
        // SAFETY: split-borrow Scratch fields via separate &mut. Done
        // by passing field refs through helpers so we don't alias.
        let (ra, rrg, rvy, da, drg, dvy) = scratch_dkl_planes(&mut self.scratch);
        srgb_to_dkl_planar(ref_srgb, self.width, self.height, display, ra, rrg, rvy);
        srgb_to_dkl_planar(dist_srgb, self.width, self.height, display, da, drg, dvy);
        let (jod, _) = self.score_internal(false)?;
        Ok(jod)
    }

    /// One-shot score with per-pixel diffmap. `diffmap_out` is
    /// resized to `width × height` (row-major). See [`crate::diffmap`].
    pub fn score_with_diffmap(
        &mut self,
        ref_srgb: &[u8],
        dist_srgb: &[u8],
        diffmap_out: &mut Vec<f32>,
    ) -> Result<f32> {
        self.check_srgb(ref_srgb)?;
        self.check_srgb(dist_srgb)?;
        self.warm_active = false;
        let display = self.params.display;
        let (ra, rrg, rvy, da, drg, dvy) = scratch_dkl_planes(&mut self.scratch);
        srgb_to_dkl_planar(ref_srgb, self.width, self.height, display, ra, rrg, rvy);
        srgb_to_dkl_planar(dist_srgb, self.width, self.height, display, da, drg, dvy);
        let (jod, diff) = self.score_internal(true)?;
        let dmap = diff.expect("with_diffmap=true returns Some");
        *diffmap_out = dmap;
        Ok(jod)
    }

    /// Cache the reference's DKL planes + Weber pyramid so subsequent
    /// `score_with_warm_ref` calls skip the half of the pipeline that
    /// only depends on the reference.
    ///
    /// Phase 9.YA: reuses persistent `scratch.ref_*` planes (allocated
    /// once in `Cvvdp::new`) and `scratch.weber_ref` slots (capacity
    /// persists across calls) so this is now allocation-free after the
    /// first warm_reference call. Saves 480 MB / call at 40 MP.
    pub fn warm_reference(&mut self, ref_srgb: &[u8]) -> Result<()> {
        self.check_srgb(ref_srgb)?;
        let display = self.params.display;
        let w = self.width;
        let h = self.height;
        // 1. Fill scratch.ref_a/ref_rg/ref_vy directly — these are
        //    pre-allocated W*H f32 buffers from Scratch::new and reused
        //    across calls.
        let (ra, rrg, rvy, _, _, _) = scratch_dkl_planes(&mut self.scratch);
        srgb_to_dkl_planar(ref_srgb, w, h, display, ra, rrg, rvy);
        // 2. Build the per-channel weber pyramid into scratch.weber_ref.
        //    Uses local pyramid caches that are freed at function exit
        //    so the warm cache footprint after warm_reference returns
        //    is just the DKL planes (480 MB) + weber output (~240 MB),
        //    NOT the gauss_img/gauss_l intermediates (~700 MB).
        let n_levels = band_frequencies(self.ppd, w, h).len();
        let Scratch {
            ref_a,
            ref_rg,
            ref_vy,
            weber_ref,
            ..
        } = &mut self.scratch;
        build_one_side_warm_ref_into(
            ref_a, ref_rg, ref_vy, w, h, n_levels, weber_ref,
        );
        // 3. Mark warm cached. The cache lives entirely in scratch:
        //    DKL planes in scratch.ref_*, weber pyramid in scratch.weber_ref.
        self.warm_active = true;
        Ok(())
    }

    /// Score against the warm reference.
    pub fn score_with_warm_ref(&mut self, dist_srgb: &[u8]) -> Result<f32> {
        if !self.warm_active {
            return Err(Error::NoWarmReference);
        }
        self.check_srgb(dist_srgb)?;
        let display = self.params.display;
        let (_, _, _, da, drg, dvy) = scratch_dkl_planes(&mut self.scratch);
        srgb_to_dkl_planar(dist_srgb, self.width, self.height, display, da, drg, dvy);
        let (jod, _) = self.score_internal_with_warm(false)?;
        Ok(jod)
    }

    /// Score against the warm reference + emit a diffmap.
    pub fn score_with_warm_ref_diffmap(
        &mut self,
        dist_srgb: &[u8],
        diffmap_out: &mut Vec<f32>,
    ) -> Result<f32> {
        if !self.warm_active {
            return Err(Error::NoWarmReference);
        }
        self.check_srgb(dist_srgb)?;
        let display = self.params.display;
        let (_, _, _, da, drg, dvy) = scratch_dkl_planes(&mut self.scratch);
        srgb_to_dkl_planar(dist_srgb, self.width, self.height, display, da, drg, dvy);
        let (jod, diff) = self.score_internal_with_warm(true)?;
        *diffmap_out = diff.expect("with_diffmap=true returns Some");
        Ok(jod)
    }

    /// One-shot score from linear-f32 RGB planes (display-relative
    /// `[0, 1]`). Bypasses the sRGB byte → linear LUT — useful for
    /// JPEG XL encoder paths that already hold linear-f32 planes.
    #[allow(clippy::too_many_arguments)]
    pub fn score_from_linear_planes(
        &mut self,
        ref_r: &[f32],
        ref_g: &[f32],
        ref_b: &[f32],
        dist_r: &[f32],
        dist_g: &[f32],
        dist_b: &[f32],
        padded_width: usize,
    ) -> Result<f32> {
        self.check_linear_planes(ref_r, padded_width)?;
        self.check_linear_planes(ref_g, padded_width)?;
        self.check_linear_planes(ref_b, padded_width)?;
        self.check_linear_planes(dist_r, padded_width)?;
        self.check_linear_planes(dist_g, padded_width)?;
        self.check_linear_planes(dist_b, padded_width)?;
        self.warm_active = false;
        let display = self.params.display;
        let w = self.width;
        let h = self.height;
        let (ra, rrg, rvy, da, drg, dvy) = scratch_dkl_planes(&mut self.scratch);
        linear_planes_to_dkl_planar(
            ref_r,
            ref_g,
            ref_b,
            w,
            h,
            padded_width,
            display,
            ra,
            rrg,
            rvy,
        );
        linear_planes_to_dkl_planar(
            dist_r,
            dist_g,
            dist_b,
            w,
            h,
            padded_width,
            display,
            da,
            drg,
            dvy,
        );
        let (jod, _) = self.score_internal(false)?;
        Ok(jod)
    }

    /// As `score_from_linear_planes` but also returns the per-pixel
    /// diffmap.
    #[allow(clippy::too_many_arguments)]
    pub fn score_from_linear_planes_with_diffmap(
        &mut self,
        ref_r: &[f32],
        ref_g: &[f32],
        ref_b: &[f32],
        dist_r: &[f32],
        dist_g: &[f32],
        dist_b: &[f32],
        padded_width: usize,
        diffmap_out: &mut Vec<f32>,
    ) -> Result<f32> {
        self.check_linear_planes(ref_r, padded_width)?;
        self.check_linear_planes(ref_g, padded_width)?;
        self.check_linear_planes(ref_b, padded_width)?;
        self.check_linear_planes(dist_r, padded_width)?;
        self.check_linear_planes(dist_g, padded_width)?;
        self.check_linear_planes(dist_b, padded_width)?;
        self.warm_active = false;
        let display = self.params.display;
        let w = self.width;
        let h = self.height;
        let (ra, rrg, rvy, da, drg, dvy) = scratch_dkl_planes(&mut self.scratch);
        linear_planes_to_dkl_planar(
            ref_r,
            ref_g,
            ref_b,
            w,
            h,
            padded_width,
            display,
            ra,
            rrg,
            rvy,
        );
        linear_planes_to_dkl_planar(
            dist_r,
            dist_g,
            dist_b,
            w,
            h,
            padded_width,
            display,
            da,
            drg,
            dvy,
        );
        let (jod, diff) = self.score_internal(true)?;
        *diffmap_out = diff.expect("with_diffmap=true returns Some");
        Ok(jod)
    }

    // --- internal helpers ---

    fn check_srgb(&self, buf: &[u8]) -> Result<()> {
        let need = self.width * self.height * 3;
        if buf.len() == need {
            Ok(())
        } else {
            Err(Error::DimensionMismatch {
                expected: need,
                got: buf.len(),
            })
        }
    }

    fn check_linear_planes(&self, buf: &[f32], padded_width: usize) -> Result<()> {
        let need = padded_width * self.height;
        if buf.len() == need {
            Ok(())
        } else {
            Err(Error::PlaneShapeMismatch {
                expected: need,
                got: buf.len(),
            })
        }
    }

    /// Core: REF planes in `scratch.ref_*`, DIST planes in
    /// `scratch.dist_*`. Builds per-channel Weber pyramids, runs
    /// CSF/masking/pool. Returns `(jod, diffmap_or_none)`.
    fn score_internal(&mut self, want_diffmap: bool) -> Result<(f32, Option<Vec<f32>>)> {
        let w = self.width;
        let h = self.height;
        let n_levels = band_frequencies(self.ppd, w, h).len();

        // The 6 weber pyramid builds are fully independent. With rayon
        // each runs in parallel. Each pyramid build writes into a
        // persistent WeberPyramid slot owned by `self.scratch`, so the
        // band Vec<f32> capacity is reused across calls — no fresh
        // per-band allocation.
        build_both_sides_into(&mut self.scratch, w, h, n_levels);

        // Phase 9.Z.F chunk 6 step 6: release weber cache memory after
        // build. The `weber_cache_dist` (and `weber_cache_ref` if it
        // grew) holds ~537 MB at 16 MP of gauss_img + gauss_l data that
        // was only needed during the weber pyramid build. The fold
        // stage reads from `weber_dist` / `weber_ref`, never the
        // caches. Releasing the cache capacity here drops peak heap
        // during fold by the cache size.
        //
        // **Tradeoff:** the cache reallocates on the next call. At 16
        // MP that's ~537 MB of fresh allocation per call, but the
        // dominant cost is the actual gauss build (decompose +
        // convolve), not the allocation itself. Per-call wall-time
        // impact is small.
        //
        // **Bit-identical safety:** clearing capacity does not affect
        // computed values — only the band-loop reads weber_*, which
        // are unaffected.
        for c in 0..3 {
            self.scratch.weber_cache_dist[c].gauss_img.clear();
            self.scratch.weber_cache_dist[c].gauss_l.clear();
            self.scratch.weber_cache_dist[c].scratch.vscratch.clear();
            self.scratch.weber_cache_dist[c].scratch.vscratch.shrink_to_fit();
            self.scratch.weber_cache_dist[c].scratch.expanded.clear();
            self.scratch.weber_cache_dist[c].scratch.expanded.shrink_to_fit();
            self.scratch.weber_cache_dist[c].scratch.gauss_tmp.clear();
            self.scratch.weber_cache_dist[c].scratch.gauss_tmp.shrink_to_fit();
            self.scratch.weber_cache_ref[c].gauss_img.clear();
            self.scratch.weber_cache_ref[c].gauss_l.clear();
            self.scratch.weber_cache_ref[c].scratch.vscratch.clear();
            self.scratch.weber_cache_ref[c].scratch.vscratch.shrink_to_fit();
            self.scratch.weber_cache_ref[c].scratch.expanded.clear();
            self.scratch.weber_cache_ref[c].scratch.expanded.shrink_to_fit();
            self.scratch.weber_cache_ref[c].scratch.gauss_tmp.clear();
            self.scratch.weber_cache_ref[c].scratch.gauss_tmp.shrink_to_fit();
        }

        // Now consume the scratch slots immutably to fold bands. We
        // temporarily move the WeberPyramids out so we can pass them
        // by &[WeberPyramid; 3] reference while holding &mut self.
        let ref_weber = core::mem::replace(
            &mut self.scratch.weber_ref,
            [
                WeberPyramid::empty(),
                WeberPyramid::empty(),
                WeberPyramid::empty(),
            ],
        );
        let dist_weber = core::mem::replace(
            &mut self.scratch.weber_dist,
            [
                WeberPyramid::empty(),
                WeberPyramid::empty(),
                WeberPyramid::empty(),
            ],
        );
        let (jod, diffmap) = self.fold_bands(&ref_weber, &dist_weber, n_levels, w, h, want_diffmap);
        // Stash back so the band buffer memory persists.
        self.scratch.weber_ref = ref_weber;
        self.scratch.weber_dist = dist_weber;
        Ok((jod, diffmap))
    }

    fn score_internal_with_warm(&mut self, want_diffmap: bool) -> Result<(f32, Option<Vec<f32>>)> {
        // Build dist pyramids in parallel; REF pyramids come from
        // warm cache (scratch.weber_ref, populated by warm_reference).
        let w = self.width;
        let h = self.height;
        let n_levels = band_frequencies(self.ppd, w, h).len();
        build_one_side_dist_into(&mut self.scratch, w, h, n_levels);

        // Phase 9.Z.F chunk 6 step 6: release weber cache memory after
        // build. See [`Self::score_internal`] doc for tradeoff details.
        // Only `weber_cache_dist` was used in this code path (REF is warm).
        for c in 0..3 {
            self.scratch.weber_cache_dist[c].gauss_img.clear();
            self.scratch.weber_cache_dist[c].gauss_l.clear();
            self.scratch.weber_cache_dist[c].scratch.vscratch.clear();
            self.scratch.weber_cache_dist[c].scratch.vscratch.shrink_to_fit();
            self.scratch.weber_cache_dist[c].scratch.expanded.clear();
            self.scratch.weber_cache_dist[c].scratch.expanded.shrink_to_fit();
            self.scratch.weber_cache_dist[c].scratch.gauss_tmp.clear();
            self.scratch.weber_cache_dist[c].scratch.gauss_tmp.shrink_to_fit();
        }

        // Pull weber slots out of scratch via mem::replace so we can
        // pass them as `&[WeberPyramid; 3]` reference while holding
        // `&mut self` for the band loop. Same pattern as score_internal
        // for the cold path. Stashed back before return so the warm
        // cache persists.
        let ref_weber = core::mem::replace(
            &mut self.scratch.weber_ref,
            [
                WeberPyramid::empty(),
                WeberPyramid::empty(),
                WeberPyramid::empty(),
            ],
        );
        let dist_weber = core::mem::replace(
            &mut self.scratch.weber_dist,
            [
                WeberPyramid::empty(),
                WeberPyramid::empty(),
                WeberPyramid::empty(),
            ],
        );
        let (jod, diffmap) =
            self.fold_bands(&ref_weber, &dist_weber, n_levels, w, h, want_diffmap);
        self.scratch.weber_ref = ref_weber;
        self.scratch.weber_dist = dist_weber;
        Ok((jod, diffmap))
    }

    /// The band-loop: applies CSF → masking → spatial pool per band
    /// per channel, accumulates the per-pixel diffmap if requested,
    /// then folds to scalar JOD. Mirrors host_scalar exactly.
    fn fold_bands(
        &mut self,
        ref_weber: &[WeberPyramid; 3],
        dist_weber: &[WeberPyramid; 3],
        n_levels: usize,
        w: usize,
        h: usize,
        want_diffmap: bool,
    ) -> (f32, Option<Vec<f32>>) {
        #[cfg(feature = "parallel")]
        {
            self.fold_bands_parallel(ref_weber, dist_weber, n_levels, w, h, want_diffmap)
        }
        #[cfg(not(feature = "parallel"))]
        {
            self.fold_bands_sequential(ref_weber, dist_weber, n_levels, w, h, want_diffmap)
        }
    }

    /// Sequential band loop. Used when `parallel` feature is off and
    /// as the inner-loop body for the parallel path. Reuses
    /// `self.scratch.band_ws[0]` for all bands (sequential — no need
    /// for per-band slots).
    #[cfg_attr(feature = "parallel", allow(dead_code))]
    fn fold_bands_sequential(
        &mut self,
        ref_weber: &[WeberPyramid; 3],
        dist_weber: &[WeberPyramid; 3],
        n_levels: usize,
        w: usize,
        h: usize,
        want_diffmap: bool,
    ) -> (f32, Option<Vec<f32>>) {
        let freqs = band_frequencies(self.ppd, w, h);
        let channels = [CsfChannel::A, CsfChannel::Rg, CsfChannel::Vy];
        // Phase 9.Z.B: strip pool config (None outside strip mode →
        // single-pass lp_norm_mean; Some(h_body) → row-strip walk).
        let strip_h_body = self.strip_h_body.get();

        // Phase 9.Z.F chunk 4 wiring: shallow levels (k < k_split)
        // dispatch through the strip-major helper using
        // [`StripBandWorkspace`].
        let k_split = if let Some(h_body) = strip_h_body {
            crate::strip::mode_b_k_split(h_body, n_levels as u32) as usize
        } else {
            0
        };

        let mut q_per_ch: Vec<[f32; 3]> = Vec::with_capacity(n_levels);
        let mut accum = if want_diffmap {
            Some(DiffmapAccum::new(w, h))
        } else {
            None
        };

        // Reuse band_ws[0] as the single sequential scratch slot for
        // deep + baseband bands. For shallow bands (k < k_split), the
        // strip-major helper writes into strip_band_ws[0] (we use a
        // single shared slot since we're sequential).
        self.scratch.ensure_band_ws(1);
        if k_split > 0 {
            self.scratch.ensure_strip_band_ws(1);
        }

        for k in 0..n_levels {
            let is_first = k == 0;
            let is_baseband = k == n_levels - 1;
            let band_mul: f32 = if is_first || is_baseband { 1.0 } else { 2.0 };

            let bw = ref_weber[0].bands[k].w;
            let bh = ref_weber[0].bands[k].h;
            let n_px = bw * bh;
            let rho = if is_baseband {
                CSF_BASEBAND_RHO
            } else {
                freqs[k]
            };
            let log_l_bkg_band = &ref_weber[0].log_l_bkg[k];
            debug_assert_eq!(log_l_bkg_band.len(), n_px);

            let logs_row_a = precompute_logs_row(rho, channels[0]);
            let logs_row_rg = precompute_logs_row(rho, channels[1]);
            let logs_row_vy = precompute_logs_row(rho, channels[2]);
            debug_assert_eq!(logs_row_a.len(), N_L_BKG);
            debug_assert_eq!(LOG_L_BKG_AXIS.len(), N_L_BKG);

            let ref_a_band = &ref_weber[0].bands[k].data;
            let ref_rg_band = &ref_weber[1].bands[k].data;
            let ref_vy_band = &ref_weber[2].bands[k].data;
            let dis_a_band = &dist_weber[0].bands[k].data;
            let dis_rg_band = &dist_weber[1].bands[k].data;
            let dis_vy_band = &dist_weber[2].bands[k].data;
            let ch_gain_a = CH_GAIN[0];
            let ch_gain_rg = CH_GAIN[1];
            let ch_gain_vy = CH_GAIN[2];

            // Shallow strip-major branch: dispatch through the helper.
            if k < k_split && !is_baseband {
                let h_body = strip_h_body.expect("strip mode active");
                let ws = &mut self.scratch.band_ws[0];
                let diffmap_d_out: Option<(&mut Vec<f32>, &mut Vec<f32>, &mut Vec<f32>)> =
                    if want_diffmap {
                        Some((&mut ws.d_a, &mut ws.d_rg, &mut ws.d_vy))
                    } else {
                        None
                    };
                let sws = &mut self.scratch.strip_band_ws.as_mut().unwrap()[0];
                let q_band = process_shallow_strip_band(
                    sws,
                    ref_a_band,
                    ref_rg_band,
                    ref_vy_band,
                    dis_a_band,
                    dis_rg_band,
                    dis_vy_band,
                    log_l_bkg_band,
                    bw,
                    bh,
                    k,
                    is_first,
                    rho,
                    h_body,
                    k_split as u32,
                    &self.strip_dispatch_counter,
                    diffmap_d_out,
                );
                q_per_ch.push(q_band);
                if let Some(acc) = accum.as_mut() {
                    let ws = &mut self.scratch.band_ws[0];
                    let d_per_ch: [Vec<f32>; 3] =
                        [ws.d_a.clone(), ws.d_rg.clone(), ws.d_vy.clone()];
                    accumulate_band_diffmap(acc, &d_per_ch, bw, bh, false, n_levels);
                }
                continue;
            }

            // Full-image deep / baseband path (also the full-mode path
            // when k_split == 0).
            let ws = &mut self.scratch.band_ws[0];

            // Compute T_p + R_p per channel into recycled workspace.
            ws.t_p_a.clear();
            ws.t_p_a.resize(n_px, 0.0);
            ws.t_p_rg.clear();
            ws.t_p_rg.resize(n_px, 0.0);
            ws.t_p_vy.clear();
            ws.t_p_vy.resize(n_px, 0.0);
            ws.r_p_a.clear();
            ws.r_p_a.resize(n_px, 0.0);
            ws.r_p_rg.clear();
            ws.r_p_rg.resize(n_px, 0.0);
            ws.r_p_vy.clear();
            ws.r_p_vy.resize(n_px, 0.0);

            for i in 0..n_px {
                let log_l = log_l_bkg_band[i];
                let s_a = apply_csf_row_per_pixel(log_l, &logs_row_a);
                let s_rg = apply_csf_row_per_pixel(log_l, &logs_row_rg);
                let s_vy = apply_csf_row_per_pixel(log_l, &logs_row_vy);
                let bm_sa = band_mul * s_a;
                let bm_srg = band_mul * s_rg;
                let bm_svy = band_mul * s_vy;
                ws.t_p_a[i] = dis_a_band[i] * bm_sa * ch_gain_a;
                ws.t_p_rg[i] = dis_rg_band[i] * bm_srg * ch_gain_rg;
                ws.t_p_vy[i] = dis_vy_band[i] * bm_svy * ch_gain_vy;
                ws.r_p_a[i] = ref_a_band[i] * bm_sa * ch_gain_a;
                ws.r_p_rg[i] = ref_rg_band[i] * bm_srg * ch_gain_rg;
                ws.r_p_vy[i] = ref_vy_band[i] * bm_svy * ch_gain_vy;
            }

            if is_baseband {
                ws.d_a.clear();
                ws.d_a.resize(n_px, 0.0);
                ws.d_rg.clear();
                ws.d_rg.resize(n_px, 0.0);
                ws.d_vy.clear();
                ws.d_vy.resize(n_px, 0.0);
                for i in 0..n_px {
                    let log_l = log_l_bkg_band[i];
                    let s_a = apply_csf_row_per_pixel(log_l, &logs_row_a);
                    let s_rg = apply_csf_row_per_pixel(log_l, &logs_row_rg);
                    let s_vy = apply_csf_row_per_pixel(log_l, &logs_row_vy);
                    let diff_a = dis_a_band[i] - ref_a_band[i];
                    let diff_rg = dis_rg_band[i] - ref_rg_band[i];
                    let diff_vy = dis_vy_band[i] - ref_vy_band[i];
                    ws.d_a[i] = diff_a.abs() * s_a;
                    ws.d_rg[i] = diff_rg.abs() * s_rg;
                    ws.d_vy[i] = diff_vy.abs() * s_vy;
                }
            } else {
                // mult_mutual_band_into wants `&[Vec<f32>; 3]`. Move
                // t_p / r_p slots out, call, move back.
                let t_p_taken: [Vec<f32>; 3] = [
                    core::mem::take(&mut ws.t_p_a),
                    core::mem::take(&mut ws.t_p_rg),
                    core::mem::take(&mut ws.t_p_vy),
                ];
                let r_p_taken: [Vec<f32>; 3] = [
                    core::mem::take(&mut ws.r_p_a),
                    core::mem::take(&mut ws.r_p_rg),
                    core::mem::take(&mut ws.r_p_vy),
                ];
                mult_mutual_band_into(
                    &t_p_taken,
                    &r_p_taken,
                    bw,
                    bh,
                    &mut ws.d_a,
                    &mut ws.d_rg,
                    &mut ws.d_vy,
                    &mut ws.m_mm_a,
                    &mut ws.m_mm_rg,
                    &mut ws.m_mm_vy,
                    &mut ws.term_a,
                    &mut ws.term_rg,
                    &mut ws.term_vy,
                    &mut ws.pu_h,
                );
                let [t_a, t_rg, t_vy] = t_p_taken;
                let [r_a, r_rg, r_vy] = r_p_taken;
                ws.t_p_a = t_a;
                ws.t_p_rg = t_rg;
                ws.t_p_vy = t_vy;
                ws.r_p_a = r_a;
                ws.r_p_rg = r_rg;
                ws.r_p_vy = r_vy;
            };

            // Spatial pool per channel using the workspace d_*.
            // In strip mode the per-band d arrays are partitioned into
            // row-strips and Σ safe_pow_lp accumulates across strips;
            // bit-identical to lp_norm_mean(..) because the spatial
            // pool is associative under row-order dispatch.
            let q_band = pool_band_3ch(
                &ws.d_a,
                &ws.d_rg,
                &ws.d_vy,
                bw,
                bh,
                k,
                strip_h_body,
                &self.strip_dispatch_counter,
            );
            q_per_ch.push(q_band);

            // Accumulate diffmap. accumulate_band_diffmap takes
            // `&[Vec<f32>; 3]`. Clone (sequential path is rare; this
            // doesn't show in any benchmark since `parallel` is the
            // default feature).
            if let Some(acc) = accum.as_mut() {
                let d_per_ch: [Vec<f32>; 3] = [ws.d_a.clone(), ws.d_rg.clone(), ws.d_vy.clone()];
                accumulate_band_diffmap(acc, &d_per_ch, bw, bh, is_baseband, n_levels);
            }
        }

        let jod = do_pooling_and_jod_still_3ch(&q_per_ch);
        let diffmap = accum.map(finalize_diffmap);
        (jod, diffmap)
    }

    /// Parallel band loop. Each band runs independently on a rayon
    /// thread; results merge via reduce at the end. Each band gets
    /// its own `BandWorkspace` slot from `self.scratch.band_ws`,
    /// indexed by band id. The Vec<f32> capacities in each slot
    /// persist across calls — no fresh per-band allocation.
    ///
    /// **Chunk 4 wiring (Phase 9.Z.F)**: in strip mode, shallow levels
    /// (`k < k_split`) dispatch to [`process_shallow_strip_band`]
    /// which routes work through strip-shaped
    /// [`StripBandWorkspace`] slots (sized `R_k × bw`, not `bh ×
    /// bw`). Deep levels (`k >= k_split`) and Full mode use the
    /// existing full-image [`BandWorkspace`] path unchanged.
    #[cfg(feature = "parallel")]
    fn fold_bands_parallel(
        &mut self,
        ref_weber: &[WeberPyramid; 3],
        dist_weber: &[WeberPyramid; 3],
        n_levels: usize,
        w: usize,
        h: usize,
        want_diffmap: bool,
    ) -> (f32, Option<Vec<f32>>) {
        use rayon::prelude::*;

        let freqs = band_frequencies(self.ppd, w, h);
        let ppd_freqs: Vec<f32> = freqs.to_vec();
        // Phase 9.Z.B: strip pool config snapshotted before the
        // par_iter_mut borrow takes effect. Each band closure captures
        // these by value (`Option<u32>` is `Copy`) / by reference
        // (`&AtomicU32` is `Sync`).
        let strip_h_body = self.strip_h_body.get();
        let strip_counter = &self.strip_dispatch_counter;

        // Phase 9.Z.F chunk 4 wiring: pick K_SPLIT once for the whole
        // band loop. Shallow levels (k < k_split) dispatch through the
        // strip-major helper; deep levels use the legacy full-image
        // path. In full mode k_split == 0 (no shallow levels).
        let k_split = if let Some(h_body) = strip_h_body {
            crate::strip::mode_b_k_split(h_body, n_levels as u32) as usize
        } else {
            0
        };

        // Grow the per-band workspace vec to at least n_levels and
        // borrow it mutably so each band closure gets its own slot.
        self.scratch.ensure_band_ws(n_levels);
        if k_split > 0 {
            self.scratch.ensure_strip_band_ws(k_split);
        }
        // Split-borrow: shallow indices [0, k_split) use strip_band_ws
        // slots; deep indices [k_split, n_levels) use band_ws slots.
        // Diffmap (if requested) at shallow levels also writes into
        // band_ws[k].d_* for the full-band per-pixel storage that
        // accumulate_band_diffmap expects.
        let (shallow_band_ws, deep_band_ws) =
            self.scratch.band_ws[..n_levels].split_at_mut(k_split);
        let shallow_strip_ws: Option<&mut [StripBandWorkspace]> = if k_split > 0 {
            Some(
                &mut self
                    .scratch
                    .strip_band_ws
                    .as_mut()
                    .expect("ensure_strip_band_ws was called above")[..k_split],
            )
        } else {
            None
        };

        // Each band's result: (q_per_ch, optional accumulated diffmap).
        // Run shallow + deep in parallel via rayon::join so both pools
        // make progress; within each, par_iter_mut over its band slots.
        let n_deep = n_levels - k_split;

        let process_deep_band = |k_deep_local: usize,
                                 ws: &mut BandWorkspace|
         -> ([f32; 3], Option<DiffmapAccum>) {
            let k = k_split + k_deep_local;
            let is_first = k == 0;
            let is_baseband = k == n_levels - 1;
            let band_mul: f32 = if is_first || is_baseband { 1.0 } else { 2.0 };
            let rho = if is_baseband {
                CSF_BASEBAND_RHO
            } else {
                ppd_freqs[k]
            };
            let bw = ref_weber[0].bands[k].w;
            let bh = ref_weber[0].bands[k].h;
            let n_px = bw * bh;
            let log_l_bkg_band = &ref_weber[0].log_l_bkg[k];
            debug_assert_eq!(log_l_bkg_band.len(), n_px);

            let logs_row_a = precompute_logs_row(rho, CsfChannel::A);
            let logs_row_rg = precompute_logs_row(rho, CsfChannel::Rg);
            let logs_row_vy = precompute_logs_row(rho, CsfChannel::Vy);
            debug_assert_eq!(logs_row_a.len(), N_L_BKG);

            let ref_a_band = &ref_weber[0].bands[k].data;
            let ref_rg_band = &ref_weber[1].bands[k].data;
            let ref_vy_band = &ref_weber[2].bands[k].data;
            let dis_a_band = &dist_weber[0].bands[k].data;
            let dis_rg_band = &dist_weber[1].bands[k].data;
            let dis_vy_band = &dist_weber[2].bands[k].data;
            let ch_gain_a = CH_GAIN[0];
            let ch_gain_rg = CH_GAIN[1];
            let ch_gain_vy = CH_GAIN[2];

            if is_baseband {
                ws.d_a.clear();
                ws.d_a.resize(n_px, 0.0);
                ws.d_rg.clear();
                ws.d_rg.resize(n_px, 0.0);
                ws.d_vy.clear();
                ws.d_vy.resize(n_px, 0.0);
                for i in 0..n_px {
                    let log_l = log_l_bkg_band[i];
                    let s_a = apply_csf_row_per_pixel(log_l, &logs_row_a);
                    let s_rg = apply_csf_row_per_pixel(log_l, &logs_row_rg);
                    let s_vy = apply_csf_row_per_pixel(log_l, &logs_row_vy);
                    let diff_a = dis_a_band[i] - ref_a_band[i];
                    let diff_rg = dis_rg_band[i] - ref_rg_band[i];
                    let diff_vy = dis_vy_band[i] - ref_vy_band[i];
                    ws.d_a[i] = diff_a.abs() * s_a;
                    ws.d_rg[i] = diff_rg.abs() * s_rg;
                    ws.d_vy[i] = diff_vy.abs() * s_vy;
                }
            } else {
                ws.t_p_a.clear();
                ws.t_p_a.resize(n_px, 0.0);
                ws.t_p_rg.clear();
                ws.t_p_rg.resize(n_px, 0.0);
                ws.t_p_vy.clear();
                ws.t_p_vy.resize(n_px, 0.0);
                ws.r_p_a.clear();
                ws.r_p_a.resize(n_px, 0.0);
                ws.r_p_rg.clear();
                ws.r_p_rg.resize(n_px, 0.0);
                ws.r_p_vy.clear();
                ws.r_p_vy.resize(n_px, 0.0);
                for i in 0..n_px {
                    let log_l = log_l_bkg_band[i];
                    let s_a = apply_csf_row_per_pixel(log_l, &logs_row_a);
                    let s_rg = apply_csf_row_per_pixel(log_l, &logs_row_rg);
                    let s_vy = apply_csf_row_per_pixel(log_l, &logs_row_vy);
                    let bm_sa = band_mul * s_a;
                    let bm_srg = band_mul * s_rg;
                    let bm_svy = band_mul * s_vy;
                    ws.t_p_a[i] = dis_a_band[i] * bm_sa * ch_gain_a;
                    ws.t_p_rg[i] = dis_rg_band[i] * bm_srg * ch_gain_rg;
                    ws.t_p_vy[i] = dis_vy_band[i] * bm_svy * ch_gain_vy;
                    ws.r_p_a[i] = ref_a_band[i] * bm_sa * ch_gain_a;
                    ws.r_p_rg[i] = ref_rg_band[i] * bm_srg * ch_gain_rg;
                    ws.r_p_vy[i] = ref_vy_band[i] * bm_svy * ch_gain_vy;
                }
                let t_p_taken: [Vec<f32>; 3] = [
                    core::mem::take(&mut ws.t_p_a),
                    core::mem::take(&mut ws.t_p_rg),
                    core::mem::take(&mut ws.t_p_vy),
                ];
                let r_p_taken: [Vec<f32>; 3] = [
                    core::mem::take(&mut ws.r_p_a),
                    core::mem::take(&mut ws.r_p_rg),
                    core::mem::take(&mut ws.r_p_vy),
                ];
                mult_mutual_band_into(
                    &t_p_taken,
                    &r_p_taken,
                    bw,
                    bh,
                    &mut ws.d_a,
                    &mut ws.d_rg,
                    &mut ws.d_vy,
                    &mut ws.m_mm_a,
                    &mut ws.m_mm_rg,
                    &mut ws.m_mm_vy,
                    &mut ws.term_a,
                    &mut ws.term_rg,
                    &mut ws.term_vy,
                    &mut ws.pu_h,
                );
                let [t_a, t_rg, t_vy] = t_p_taken;
                let [r_a, r_rg, r_vy] = r_p_taken;
                ws.t_p_a = t_a;
                ws.t_p_rg = t_rg;
                ws.t_p_vy = t_vy;
                ws.r_p_a = r_a;
                ws.r_p_rg = r_rg;
                ws.r_p_vy = r_vy;
            }

            let q_band = pool_band_3ch(
                &ws.d_a,
                &ws.d_rg,
                &ws.d_vy,
                bw,
                bh,
                k,
                strip_h_body,
                strip_counter,
            );

            let band_accum = if want_diffmap {
                let mut acc = DiffmapAccum::new(w, h);
                let d_per_ch: [Vec<f32>; 3] =
                    [ws.d_a.clone(), ws.d_rg.clone(), ws.d_vy.clone()];
                accumulate_band_diffmap(&mut acc, &d_per_ch, bw, bh, is_baseband, n_levels);
                Some(acc)
            } else {
                None
            };

            (q_band, band_accum)
        };

        let process_shallow_band = |k: usize,
                                    sws: &mut StripBandWorkspace,
                                    bws_for_diff: &mut BandWorkspace|
         -> ([f32; 3], Option<DiffmapAccum>) {
            debug_assert!(k < k_split);
            // Shallow strip bands are never the baseband (k_split's
            // design table caps it at log2(h_body / 12) + 1, well
            // below n_levels - 1 for realistic h_body).
            debug_assert!(k != n_levels - 1, "k_split should never include the baseband");
            let is_first = k == 0;
            let rho = ppd_freqs[k];
            let bw = ref_weber[0].bands[k].w;
            let bh = ref_weber[0].bands[k].h;

            let ref_a_band = &ref_weber[0].bands[k].data;
            let ref_rg_band = &ref_weber[1].bands[k].data;
            let ref_vy_band = &ref_weber[2].bands[k].data;
            let dis_a_band = &dist_weber[0].bands[k].data;
            let dis_rg_band = &dist_weber[1].bands[k].data;
            let dis_vy_band = &dist_weber[2].bands[k].data;
            let log_l_bkg_band = &ref_weber[0].log_l_bkg[k];

            let h_body =
                strip_h_body.expect("strip mode active when processing shallow band");

            // For diffmap, allocate the full-band d_* in the BandWorkspace
            // so accumulate_band_diffmap can read them. Pass references
            // to process_shallow_strip_band, which writes per-strip body
            // rows into them.
            let diffmap_d_out: Option<(&mut Vec<f32>, &mut Vec<f32>, &mut Vec<f32>)> =
                if want_diffmap {
                    Some((
                        &mut bws_for_diff.d_a,
                        &mut bws_for_diff.d_rg,
                        &mut bws_for_diff.d_vy,
                    ))
                } else {
                    None
                };

            let q_band = process_shallow_strip_band(
                sws,
                ref_a_band,
                ref_rg_band,
                ref_vy_band,
                dis_a_band,
                dis_rg_band,
                dis_vy_band,
                log_l_bkg_band,
                bw,
                bh,
                k,
                is_first,
                rho,
                h_body,
                k_split as u32,
                strip_counter,
                diffmap_d_out,
            );

            let band_accum = if want_diffmap {
                let mut acc = DiffmapAccum::new(w, h);
                let d_per_ch: [Vec<f32>; 3] = [
                    bws_for_diff.d_a.clone(),
                    bws_for_diff.d_rg.clone(),
                    bws_for_diff.d_vy.clone(),
                ];
                accumulate_band_diffmap(&mut acc, &d_per_ch, bw, bh, false, n_levels);
                Some(acc)
            } else {
                None
            };

            (q_band, band_accum)
        };

        // Run shallow + deep band-fold halves in parallel via rayon::join.
        // Each half does its own par_iter_mut over its band slot vector.
        let (shallow_results, deep_results): (
            Vec<([f32; 3], Option<DiffmapAccum>)>,
            Vec<([f32; 3], Option<DiffmapAccum>)>,
        ) = rayon::join(
            || {
                if k_split == 0 {
                    return Vec::new();
                }
                // shallow_strip_ws is Some when k_split > 0
                let sws_slice = shallow_strip_ws.expect("shallow_strip_ws Some when k_split > 0");
                // Zip strip_band_ws[k] with shallow_band_ws[k] (the per-band
                // BandWorkspace slot, used only for full-band diffmap d_*
                // storage when diffmap is requested).
                sws_slice
                    .par_iter_mut()
                    .zip(shallow_band_ws.par_iter_mut())
                    .enumerate()
                    .map(|(k, (sws, bws))| process_shallow_band(k, sws, bws))
                    .collect()
            },
            || {
                if n_deep == 0 {
                    return Vec::new();
                }
                deep_band_ws
                    .par_iter_mut()
                    .enumerate()
                    .map(|(k_deep_local, ws)| process_deep_band(k_deep_local, ws))
                    .collect()
            },
        );

        let band_results: Vec<([f32; 3], Option<DiffmapAccum>)> = shallow_results
            .into_iter()
            .chain(deep_results.into_iter())
            .collect();

        let q_per_ch: Vec<[f32; 3]> = band_results.iter().map(|(q, _)| *q).collect();

        let merged_accum = if want_diffmap {
            // Reduce the per-band per-pixel-per-channel accumulators by
            // summing each channel plane.
            let mut merged = DiffmapAccum::new(w, h);
            for (_, opt_acc) in &band_results {
                if let Some(acc) = opt_acc {
                    for c in 0..3 {
                        for i in 0..w * h {
                            merged.channels[c][i] += acc.channels[c][i];
                        }
                    }
                }
            }
            Some(merged)
        } else {
            None
        };

        let jod = do_pooling_and_jod_still_3ch(&q_per_ch);
        let diffmap = merged_accum.map(finalize_diffmap);
        (jod, diffmap)
    }

    /// Drop the warm reference cache. The scratch buffers themselves
    /// are NOT freed — only the `warm_active` flag is cleared so the
    /// next `score_with_warm_ref` call returns `NoWarmReference`.
    pub fn drop_warm_reference(&mut self) {
        self.warm_active = false;
    }

    /// Whether a warm reference is currently cached.
    pub fn has_warm_reference(&self) -> bool {
        self.warm_active
    }

    /// Strip-mode score: walks the band-fold's spatial Minkowski pool
    /// in horizontal slabs of `(strip_h_body >> k).max(1)` rows per
    /// band. Returns the same JOD as [`Self::score`] **bit-identically
    /// in row-order f32 add sequence** — the spatial pool is
    /// associative across strips, and the walker dispatches strips
    /// top-down so the accumulator sees the same `acc += x_i`
    /// sequence as the single-pass `lp_norm_mean` call.
    ///
    /// # Status (Phase 9.Z.B / task #124)
    ///
    /// This method ports the GPU's shipped strip pool walker — the
    /// "Phase 3 Approach B incremental landing" documented at
    /// `crates/cvvdp-gpu/docs/STRIP_PROCESSING.md:281-320`. **Like the
    /// GPU today, only the pool stage iterates in strips; the rest of
    /// the pipeline (weber pyramid build, CSF, masking, mult-mutual)
    /// stays full-image-sized.** Memory impact: zero. The walker is
    /// load-bearing for the parity invariant — once the per-strip
    /// pyramid kernels ship (multi-day kernel work, see GPU notes),
    /// the same outer-walker geometry holds.
    ///
    /// The [`crate::strip::mode_b_k_split`] / `mode_b_strip_h_at_level`
    /// helpers are wired but unused at the pool stage (the pool walker
    /// uses the simpler `body_at_k = (h_body >> k).max(1)` rule because
    /// the pool stage doesn't reflect-across-rows; it's a pure
    /// elementwise sum). They become load-bearing when the per-strip
    /// pyramid kernels land — which is when the K_SPLIT decision
    /// gates allocation of strip-vs-full-image buffers per level.
    ///
    /// # Parameters
    ///
    /// * `strip_h_body` — strip body height at scale 0. Must be a
    ///   positive power of two; see [`crate::strip::is_valid_strip_h_body`].
    ///   Use [`crate::strip::STRIP_H_BODY_DEFAULT`] (512) when unsure.
    pub fn score_strip(
        &mut self,
        ref_srgb: &[u8],
        dist_srgb: &[u8],
        strip_h_body: u32,
    ) -> Result<f32> {
        if !crate::strip::is_valid_strip_h_body(strip_h_body) {
            return Err(Error::InvalidImageSize {
                width: strip_h_body,
                height: 0,
            });
        }
        self.strip_h_body.set(Some(strip_h_body));
        let result = self.score(ref_srgb, dist_srgb);
        self.strip_h_body.set(None);
        result
    }

    /// Strip-mode score against the warm reference. See
    /// [`Self::score_strip`] for the walker contract.
    ///
    /// Phase 9.Z.B: walks the spatial pool in row-strips. Bit-identical
    /// to [`Self::score_with_warm_ref`] under row-order dispatch.
    pub fn score_with_warm_ref_strip(
        &mut self,
        dist_srgb: &[u8],
        strip_h_body: u32,
    ) -> Result<f32> {
        if !crate::strip::is_valid_strip_h_body(strip_h_body) {
            return Err(Error::InvalidImageSize {
                width: strip_h_body,
                height: 0,
            });
        }
        self.strip_h_body.set(Some(strip_h_body));
        let result = self.score_with_warm_ref(dist_srgb);
        self.strip_h_body.set(None);
        result
    }

    /// Cumulative number of strip-iteration dispatches since this
    /// `Cvvdp` was constructed (or [`Self::reset_strip_dispatch_counter`]
    /// was last called). Each strip dispatched in `pool_band_3ch`
    /// increments by 3 (mirrors the GPU's single-launch-covers-3-
    /// channels dispatch).
    ///
    /// Not part of the stable public API — exposed via `#[doc(hidden)]`
    /// for tests asserting the walker partitions at large sizes.
    /// Mirrors `cvvdp_gpu::Cvvdp::strip_dispatch_counter`.
    #[doc(hidden)]
    pub fn strip_dispatch_counter(&self) -> u32 {
        self.strip_dispatch_counter
            .load(core::sync::atomic::Ordering::Relaxed)
    }

    /// Reset the strip-dispatch counter to 0. Use between sub-tests
    /// that want to assert partition counts independently. Mirrors
    /// `cvvdp_gpu::Cvvdp::reset_strip_dispatch_counter`.
    #[doc(hidden)]
    pub fn reset_strip_dispatch_counter(&self) {
        self.strip_dispatch_counter
            .store(0, core::sync::atomic::Ordering::Relaxed);
    }

    /// Score from `zenpixels::PixelSlice` references — converts the
    /// input to the `RGB8_SRGB` descriptor first via
    /// `zenpixels_convert`, then dispatches to [`Cvvdp::score`].
    ///
    /// Use this when the caller already holds a `PixelSlice` (e.g.
    /// from a decoder / processor that emits zenpixels-typed
    /// pixel data). For raw `&[u8]` callers, prefer `score` to
    /// skip the trivial copy.
    #[cfg(feature = "pixels")]
    pub fn score_pixels(
        &mut self,
        reference: zenpixels::PixelSlice<'_>,
        distorted: zenpixels::PixelSlice<'_>,
    ) -> Result<f32> {
        let ref_buf = to_srgb_rgb8(&reference, self.width as u32, self.height as u32)?;
        let dis_buf = to_srgb_rgb8(&distorted, self.width as u32, self.height as u32)?;
        self.score(&ref_buf, &dis_buf)
    }
}

#[cfg(feature = "pixels")]
fn to_srgb_rgb8(
    s: &zenpixels::PixelSlice<'_>,
    expected_w: u32,
    expected_h: u32,
) -> Result<Vec<u8>> {
    if s.width() != expected_w || s.rows() != expected_h {
        let expected = (expected_w as usize) * (expected_h as usize) * 3;
        let got = (s.width() as usize) * (s.rows() as usize) * 3;
        return Err(Error::DimensionMismatch { expected, got });
    }
    let target = zenpixels::PixelDescriptor::RGB8_SRGB;
    if s.descriptor() == target {
        return Ok(s.contiguous_bytes().into_owned());
    }
    convert_to_srgb_rgb8(s, target).map_err(|_| Error::DimensionMismatch {
        expected: (expected_w as usize) * (expected_h as usize) * 3,
        got: (s.width() as usize) * (s.rows() as usize) * 3,
    })
}

#[cfg(feature = "pixels")]
fn convert_to_srgb_rgb8(
    s: &zenpixels::PixelSlice<'_>,
    target: zenpixels::PixelDescriptor,
) -> core::result::Result<Vec<u8>, zenpixels_convert::ConvertError> {
    use zenpixels_convert::{ConvertPlan, convert_row};
    let plan = ConvertPlan::new(s.descriptor(), target).map_err(|e| e.decompose().0)?;
    let w = s.width();
    let h = s.rows();
    let row_bytes = (w as usize) * target.bytes_per_pixel();
    let mut out = vec![0u8; row_bytes * (h as usize)];
    for y in 0..h {
        let src_row = s.row(y);
        let start = (y as usize) * row_bytes;
        let dst_row = &mut out[start..start + row_bytes];
        convert_row(&plan, src_row, dst_row, w);
    }
    Ok(out)
}

/// 6-tuple of `&mut Vec<f32>` over the per-side DKL plane scratch
/// slots in `Scratch` — returned by `scratch_dkl_planes` and only
/// used at the entry point of each scoring path.
type ScratchDklPlanesMut<'a> = (
    &'a mut Vec<f32>,
    &'a mut Vec<f32>,
    &'a mut Vec<f32>,
    &'a mut Vec<f32>,
    &'a mut Vec<f32>,
    &'a mut Vec<f32>,
);

/// Helper for `score()` to obtain the 6 plane scratches from
/// `Scratch` as separate `&mut Vec<f32>`. Required because Rust's
/// borrow checker rejects 6 simultaneous `&mut` of `self.scratch.*`
/// directly through field projection on a single line in some
/// configurations; this helper makes the split-borrow explicit.
fn scratch_dkl_planes(scratch: &mut Scratch) -> ScratchDklPlanesMut<'_> {
    (
        &mut scratch.ref_a,
        &mut scratch.ref_rg,
        &mut scratch.ref_vy,
        &mut scratch.dist_a,
        &mut scratch.dist_rg,
        &mut scratch.dist_vy,
    )
}

// Tiny shim — keep clippy quiet about unused.
#[allow(dead_code)]
const _: () = {
    let _ = BETA_BAND;
    let _ = BETA_CH;
    let _ = IMAGE_INT;
    let _ = PER_CH_W;
    let _ = BASEBAND_W;
};

/// Spatial Minkowski pool for one band, 3 channels.
///
/// In Full mode (`strip_h_body == None`) this is a direct three-call
/// `lp_norm_mean(&d, BETA_SPATIAL)` per channel — bit-identical to the
/// pre-Phase-9.Z.B behavior. In Strip mode (`strip_h_body == Some(hb)`)
/// the band's `d_*` arrays are partitioned into row-strips of
/// `(hb >> k).max(1)` rows where `k` is the band's pyramid level;
/// each strip's `Σ safe_pow_lp` accumulates into a per-channel
/// [`crate::strip::LpNormAccumulator`] which is finalized after the
/// last strip.
///
/// **Bit-identical to single-pass `lp_norm_mean`** because the
/// spatial pool is associative and the strips are dispatched in
/// row-order (the same `acc += x_i` sequence as the single-pass call).
/// See `crate::strip::LpNormAccumulator` doc for the proof.
///
/// `bw` is the band width (so a strip of `s` rows is `s * bw` pixels).
/// `bh` is the band height (so total pixels per channel = `bw * bh`).
/// `k` is the band's pyramid level (0 = finest); used as the right-
/// shift exponent for the scale-0 `h_body` so the band-local strip
/// body matches the GPU's `(strip_h_body >> k).max(1)` rule.
/// `strip_counter` is incremented by 3 per strip dispatched (one count
/// per channel, mirroring the GPU's `pool_band_3ch_offset_kernel`
/// dispatch where each launch covers all 3 channels of one slab).
fn pool_band_3ch(
    d_a: &[f32],
    d_rg: &[f32],
    d_vy: &[f32],
    bw: usize,
    bh: usize,
    k: usize,
    strip_h_body: Option<u32>,
    strip_counter: &core::sync::atomic::AtomicU32,
) -> [f32; 3] {
    let Some(h_body) = strip_h_body else {
        // Full mode: identical to the pre-strip code path.
        return [
            lp_norm_mean(d_a, BETA_SPATIAL),
            lp_norm_mean(d_rg, BETA_SPATIAL),
            lp_norm_mean(d_vy, BETA_SPATIAL),
        ];
    };

    // Strip mode: per-band strip body at this level.
    // Mirrors GPU pipeline.rs:7840:
    //   `let strip_h_at_band = (strip_h_body >> k).max(1);`
    let strip_h_at_band = crate::strip::strip_h_at_band(h_body, k as u32) as usize;
    let n_strips = if bh <= strip_h_at_band {
        1
    } else {
        bh.div_ceil(strip_h_at_band)
    };

    let mut acc_a = crate::strip::LpNormAccumulator::default();
    let mut acc_rg = crate::strip::LpNormAccumulator::default();
    let mut acc_vy = crate::strip::LpNormAccumulator::default();
    let mut strips_dispatched: u32 = 0;
    for s in 0..n_strips {
        let row_start = s * strip_h_at_band;
        let row_count = (bh - row_start).min(strip_h_at_band);
        let start = row_start * bw;
        let end = start + row_count * bw;
        acc_a.accumulate_slab(&d_a[start..end], BETA_SPATIAL);
        acc_rg.accumulate_slab(&d_rg[start..end], BETA_SPATIAL);
        acc_vy.accumulate_slab(&d_vy[start..end], BETA_SPATIAL);
        // Mirror the GPU's "one launch per (level, strip) covers all
        // 3 channels in a single kernel". Counter increment is 3 per
        // strip so tests can sanity-check both partitioning AND
        // channel-fold count.
        strips_dispatched = strips_dispatched.saturating_add(3);
    }
    strip_counter.fetch_add(
        strips_dispatched,
        core::sync::atomic::Ordering::Relaxed,
    );
    [
        acc_a.finalize(BETA_SPATIAL),
        acc_rg.finalize(BETA_SPATIAL),
        acc_vy.finalize(BETA_SPATIAL),
    ]
}

/// Chunk 4 wiring (Phase 9.Z.F): process one shallow non-baseband band
/// in strip-major mode using a strip-shaped [`StripBandWorkspace`]
/// instead of the full-image-sized [`BandWorkspace`].
///
/// **Memory contract.** All per-band CSF + masking transients
/// (`t_p_*`, `r_p_*`, `m_mm_*`, `term_*`, `pu_h`, `d_*`) live in the
/// caller-owned [`StripBandWorkspace`], sized at `n_strip = strip_window_h
/// × bw` instead of `bh × bw`. For a 4096² source at level 0 with
/// h_body = 512, strip_window_h is `≤ 512 + 2·8 = 528` rows, ~7.7×
/// smaller than the full band's 4096 rows.
///
/// **Bit-identical to full-mode for body output.** The strip walker
/// dispatches strips in row-order (`s = 0, 1, 2, ...`), each strip
/// holding `[top_global, bot_global)` rows of the full-image
/// `ref_*_band` / `dis_*_band` / `log_l_bkg_band` (just sliced). The
/// per-pixel CSF + masking chain is run on the strip buffer, with the
/// key invariant that **`gaussian_blur_sigma3_simd` on the strip
/// buffer produces bit-identical body row output to the same SIMD
/// call on the full band** — because:
///
/// 1. All body rows in a strip with `halo_band ≥ 6` halo on both sides
///    land in the SIMD interior region `[6, strip_h - 6)`, where the
///    SIMD path uses direct reads (no reflection). The 13-tap window
///    around each body row reads strip-buffer rows that correspond
///    exactly to the same logical rows as the full-image SIMD would
///    read at the matching logical y.
/// 2. For edge strips (top or bottom of the band), the strip's clamped
///    halo (0 on the edge side) means some body rows fall in the
///    SIMD scalar boundary region (rows `[0, 6) ∪ [strip_h-6,
///    strip_h)`). The scalar boundary code uses `reflect_pu_idx(y +
///    t - 6, strip_h)`. For the top edge (`body_off = 0`,
///    `top_halo = 0`), negative indices reflect into positive indices
///    `0..6` — exactly the strip's top body rows = logical rows
///    `0..6` — same result as full-image reflection. For the bottom
///    edge (`bot_global = bh`, `bot_halo = 0`), out-of-range indices
///    reflect into rows `strip_h - 2..strip_h - 7` — corresponding to
///    logical rows `bh - 2..bh - 7` — same as full-image reflection.
///    The strip and full reflections commute with the
///    `top_global` offset translation.
///
/// **Per-strip pool accumulation.** Each strip's body rows feed a
/// per-channel [`LpNormAccumulator`]; after the last strip we finalize
/// to per-channel `q`. Bit-identical to single-pass `lp_norm_mean`
/// over the full-band d array because the spatial Minkowski pool is
/// associative under row-order dispatch (see [`LpNormAccumulator`]).
///
/// **Diffmap support.** When `diffmap_d_out` is `Some`, the function
/// also writes the per-strip body's d values back into the full-band
/// d_a/rg/vy buffers at the body row range, so the caller can run
/// [`accumulate_band_diffmap`] afterwards on the full-band data.
/// When `None`, no full-band d storage is needed — only the
/// strip-sized [`StripBandWorkspace::d_*`] is written.
///
/// **Caller invariants.**
/// - `sws.t_p_*` / `r_p_*` / `m_mm_*` / `term_*` / `d_*` are resized
///   internally to fit `n_strip`.
/// - `ref_*_band`, `dis_*_band`, `log_l_bkg_band` have length `bw * bh`.
/// - `k < k_split` (this function is for SHALLOW levels only).
/// - `h_body` is a positive power of 2.
#[allow(clippy::too_many_arguments)]
fn process_shallow_strip_band(
    sws: &mut StripBandWorkspace,
    ref_a_band: &[f32],
    ref_rg_band: &[f32],
    ref_vy_band: &[f32],
    dis_a_band: &[f32],
    dis_rg_band: &[f32],
    dis_vy_band: &[f32],
    log_l_bkg_band: &[f32],
    bw: usize,
    bh: usize,
    k: usize,
    is_first: bool,
    rho: f32,
    h_body: u32,
    k_split: u32,
    strip_counter: &core::sync::atomic::AtomicU32,
    diffmap_d_out: Option<(&mut Vec<f32>, &mut Vec<f32>, &mut Vec<f32>)>,
) -> [f32; 3] {
    debug_assert_eq!(ref_a_band.len(), bw * bh);
    debug_assert_eq!(log_l_bkg_band.len(), bw * bh);

    let halo_band = mode_b_halo_at_level(k as u32, k_split) as usize;
    let strip_h_at_band = crate::strip::strip_h_at_band(h_body, k as u32) as usize;
    let n_strips = if bh <= strip_h_at_band {
        1
    } else {
        bh.div_ceil(strip_h_at_band)
    };

    // Size the workspace to the max strip window any iteration will need.
    // For interior strips: strip_h_at_band + 2*halo_band rows.
    // For edge strips: less (clamped halo). Sizing to max keeps the
    // allocation stable across iterations.
    let max_window_h = strip_h_at_band + 2 * halo_band;
    let max_window_h = max_window_h.min(bh);
    let n_strip = bw * max_window_h;
    sws.ensure_strip_sized(n_strip);

    // Optionally clear the full-band diffmap d output. Caller provides
    // pre-sized full-band Vecs.
    let mut diffmap_d_out = diffmap_d_out;
    if let Some((d_a_full, d_rg_full, d_vy_full)) = diffmap_d_out.as_mut() {
        let n_full = bw * bh;
        d_a_full.clear();
        d_a_full.resize(n_full, 0.0);
        d_rg_full.clear();
        d_rg_full.resize(n_full, 0.0);
        d_vy_full.clear();
        d_vy_full.resize(n_full, 0.0);
    }

    // Per-band CSF row constants.
    let channels = [CsfChannel::A, CsfChannel::Rg, CsfChannel::Vy];
    let logs_row_a = precompute_logs_row(rho, channels[0]);
    let logs_row_rg = precompute_logs_row(rho, channels[1]);
    let logs_row_vy = precompute_logs_row(rho, channels[2]);
    let band_mul: f32 = if is_first { 1.0 } else { 2.0 };
    let ch_gain_a = CH_GAIN[0];
    let ch_gain_rg = CH_GAIN[1];
    let ch_gain_vy = CH_GAIN[2];

    // Masking constants (hoisted from mult_mutual_band_into).
    const SAFE_EPS: f32 = 1e-5;
    let mask_c_lin: f32 = 10.0_f32.powf(MASK_C);
    let q_a = MASK_Q[0];
    let q_rg = MASK_Q[1];
    let q_vy = MASK_Q[2];
    let eps_qa = SAFE_EPS.powf(q_a);
    let eps_qrg = SAFE_EPS.powf(q_rg);
    let eps_qvy = SAFE_EPS.powf(q_vy);
    let xcm00 = XCM_3X3[0][0];
    let xcm10 = XCM_3X3[1][0];
    let xcm20 = XCM_3X3[2][0];
    let xcm01 = XCM_3X3[0][1];
    let xcm11 = XCM_3X3[1][1];
    let xcm21 = XCM_3X3[2][1];
    let xcm02 = XCM_3X3[0][2];
    let xcm12 = XCM_3X3[1][2];
    let xcm22 = XCM_3X3[2][2];
    let p = MASK_P;
    let eps_p = SAFE_EPS.powf(p);
    let d_max_lin: f32 = 10.0_f32.powf(D_MAX);

    let mut acc_a = LpNormAccumulator::default();
    let mut acc_rg = LpNormAccumulator::default();
    let mut acc_vy = LpNormAccumulator::default();
    let mut strips_dispatched: u32 = 0;

    for s in 0..n_strips {
        let body_off = s * strip_h_at_band;
        let body_h = (bh - body_off).min(strip_h_at_band);
        let top_global = body_off.saturating_sub(halo_band);
        let bot_global = (body_off + body_h + halo_band).min(bh);
        let strip_window_h = bot_global - top_global;
        let body_row_in_strip = body_off - top_global; // 0 at top edge, halo_band interior

        let n_strip_window = bw * strip_window_h;

        // Step 1: slice ref_*/dis_*/log_l_bkg into strip-shaped buffers.
        // Use sws.t_p_a as the temp slot for ref_a, sws.t_p_rg for ref_rg,
        // sws.t_p_vy for ref_vy, sws.r_p_a/rg/vy for dis_*. (Later we'll
        // overwrite these with CSF outputs.)
        //
        // Actually simpler: just copy. We need to compute T_p / R_p in
        // sws.t_p_*/r_p_* anyway, and we read from full-band slices for
        // CSF input. So no need to copy band data first — CSF reads from
        // ref_*_band[top_global*bw .. bot_global*bw] directly.

        // Step 2: CSF apply per-pixel over the strip window. Writes into
        // sws.t_p_*/r_p_* at indices [0, n_strip_window).
        for sy in 0..strip_window_h {
            let strip_row_off = sy * bw;
            let full_row = top_global + sy;
            let full_row_off = full_row * bw;
            for x in 0..bw {
                let i = strip_row_off + x;
                let j = full_row_off + x;
                let log_l = log_l_bkg_band[j];
                let s_a = apply_csf_row_per_pixel(log_l, &logs_row_a);
                let s_rg = apply_csf_row_per_pixel(log_l, &logs_row_rg);
                let s_vy = apply_csf_row_per_pixel(log_l, &logs_row_vy);
                let bm_sa = band_mul * s_a;
                let bm_srg = band_mul * s_rg;
                let bm_svy = band_mul * s_vy;
                sws.t_p_a[i] = dis_a_band[j] * bm_sa * ch_gain_a;
                sws.t_p_rg[i] = dis_rg_band[j] * bm_srg * ch_gain_rg;
                sws.t_p_vy[i] = dis_vy_band[j] * bm_svy * ch_gain_vy;
                sws.r_p_a[i] = ref_a_band[j] * bm_sa * ch_gain_a;
                sws.r_p_rg[i] = ref_rg_band[j] * bm_srg * ch_gain_rg;
                sws.r_p_vy[i] = ref_vy_band[j] * bm_svy * ch_gain_vy;
            }
        }

        // Step 3: mult_mutual chain on strip buffers. The PU blur uses
        // gaussian_blur_sigma3_simd on the strip dim. For interior body
        // rows in SIMD-interior region [6, strip_window_h - 6), this is
        // bit-identical to the same SIMD on the full-band. For edge
        // strips with clamped halo, the SIMD scalar boundary's reflection
        // against strip_window_h commutes with the top_global offset
        // translation, producing the same per-row values as full-band
        // SIMD.

        // Step 3.1: M_mm_raw = min(|T|, |R|).
        for i in 0..n_strip_window {
            let ta = sws.t_p_a[i].abs();
            let ra = sws.r_p_a[i].abs();
            sws.m_mm_a[i] = ta.min(ra);
            let trg = sws.t_p_rg[i].abs();
            let rrg = sws.r_p_rg[i].abs();
            sws.m_mm_rg[i] = trg.min(rrg);
            let tvy = sws.t_p_vy[i].abs();
            let rvy = sws.r_p_vy[i].abs();
            sws.m_mm_vy[i] = tvy.min(rvy);
        }

        // Step 3.2: PU blur per channel using SIMD on the strip buffer.
        // gaussian_blur_sigma3_simd expects bw > PU_PADSIZE && bh >
        // PU_PADSIZE. For shallow levels we should always satisfy that
        // by the k_split design table. Use the no-blur fallback in
        // edge cases.
        if bw > PU_PADSIZE && strip_window_h > PU_PADSIZE {
            // SIMD H+V blur on the strip. h_pass = sws.pu_h.
            // term_a/rg/vy gets the blurred output.
            gaussian_blur_sigma3_simd(
                &sws.m_mm_a[..n_strip_window],
                bw,
                strip_window_h,
                &mut sws.pu_h,
                &mut sws.term_a,
            );
            for i in 0..n_strip_window {
                sws.m_mm_a[i] = sws.term_a[i] * mask_c_lin;
            }
            gaussian_blur_sigma3_simd(
                &sws.m_mm_rg[..n_strip_window],
                bw,
                strip_window_h,
                &mut sws.pu_h,
                &mut sws.term_rg,
            );
            for i in 0..n_strip_window {
                sws.m_mm_rg[i] = sws.term_rg[i] * mask_c_lin;
            }
            gaussian_blur_sigma3_simd(
                &sws.m_mm_vy[..n_strip_window],
                bw,
                strip_window_h,
                &mut sws.pu_h,
                &mut sws.term_vy,
            );
            for i in 0..n_strip_window {
                sws.m_mm_vy[i] = sws.term_vy[i] * mask_c_lin;
            }
        } else {
            for i in 0..n_strip_window {
                sws.m_mm_a[i] *= mask_c_lin;
                sws.m_mm_rg[i] *= mask_c_lin;
                sws.m_mm_vy[i] *= mask_c_lin;
            }
        }

        // Step 3.3: term[ch] = safe_pow(|M_mm[ch]|, q[ch]).
        safe_pow_with_offset_into(
            &sws.m_mm_a[..n_strip_window],
            &mut sws.term_a[..n_strip_window],
            SAFE_EPS,
            q_a,
            eps_qa,
        );
        safe_pow_with_offset_into(
            &sws.m_mm_rg[..n_strip_window],
            &mut sws.term_rg[..n_strip_window],
            SAFE_EPS,
            q_rg,
            eps_qrg,
        );
        safe_pow_with_offset_into(
            &sws.m_mm_vy[..n_strip_window],
            &mut sws.term_vy[..n_strip_window],
            SAFE_EPS,
            q_vy,
            eps_qvy,
        );

        // Step 3.4: Pass 1: diff[c] = |T - R| → m_mm_* (free scratch).
        for i in 0..n_strip_window {
            sws.m_mm_a[i] = (sws.t_p_a[i] - sws.r_p_a[i]).abs();
            sws.m_mm_rg[i] = (sws.t_p_rg[i] - sws.r_p_rg[i]).abs();
            sws.m_mm_vy[i] = (sws.t_p_vy[i] - sws.r_p_vy[i]).abs();
        }

        // Step 3.5: Pass 2: pow[c] = (diff[c] + eps)^p - eps^p → d_*.
        safe_pow_with_offset_into(
            &sws.m_mm_a[..n_strip_window],
            &mut sws.d_a[..n_strip_window],
            SAFE_EPS,
            p,
            eps_p,
        );
        safe_pow_with_offset_into(
            &sws.m_mm_rg[..n_strip_window],
            &mut sws.d_rg[..n_strip_window],
            SAFE_EPS,
            p,
            eps_p,
        );
        safe_pow_with_offset_into(
            &sws.m_mm_vy[..n_strip_window],
            &mut sws.d_vy[..n_strip_window],
            SAFE_EPS,
            p,
            eps_p,
        );

        // Step 3.6: Pass 3: cross-channel pool + soft clamp, scalar.
        for i in 0..n_strip_window {
            let t0 = sws.term_a[i];
            let t1 = sws.term_rg[i];
            let t2 = sws.term_vy[i];
            let m0 = xcm00 * t0 + xcm10 * t1 + xcm20 * t2;
            let m1 = xcm01 * t0 + xcm11 * t1 + xcm21 * t2;
            let m2 = xcm02 * t0 + xcm12 * t1 + xcm22 * t2;
            let pow0 = sws.d_a[i];
            let pow1 = sws.d_rg[i];
            let pow2 = sws.d_vy[i];
            let du0 = pow0 / (1.0 + m0);
            let du1 = pow1 / (1.0 + m1);
            let du2 = pow2 / (1.0 + m2);
            sws.d_a[i] = d_max_lin * du0 / (d_max_lin + du0);
            sws.d_rg[i] = d_max_lin * du1 / (d_max_lin + du1);
            sws.d_vy[i] = d_max_lin * du2 / (d_max_lin + du2);
        }

        // Step 4: pool body rows of d_* into per-band accumulator.
        // Body rows in strip are [body_row_in_strip, body_row_in_strip
        // + body_h). The pool sees them in row order — bit-identical to
        // lp_norm_mean over the full band's d (since strips dispatch
        // in row order).
        let body_start = body_row_in_strip * bw;
        let body_end = body_start + body_h * bw;
        acc_a.accumulate_slab(&sws.d_a[body_start..body_end], BETA_SPATIAL);
        acc_rg.accumulate_slab(&sws.d_rg[body_start..body_end], BETA_SPATIAL);
        acc_vy.accumulate_slab(&sws.d_vy[body_start..body_end], BETA_SPATIAL);
        strips_dispatched = strips_dispatched.saturating_add(3);

        // Step 5: optional: copy strip's body d_* back to full-band
        // d_* for diffmap accumulation later.
        if let Some((d_a_full, d_rg_full, d_vy_full)) = diffmap_d_out.as_mut() {
            let full_off = body_off * bw;
            d_a_full[full_off..full_off + body_h * bw]
                .copy_from_slice(&sws.d_a[body_start..body_end]);
            d_rg_full[full_off..full_off + body_h * bw]
                .copy_from_slice(&sws.d_rg[body_start..body_end]);
            d_vy_full[full_off..full_off + body_h * bw]
                .copy_from_slice(&sws.d_vy[body_start..body_end]);
        }
    }

    strip_counter.fetch_add(
        strips_dispatched,
        core::sync::atomic::Ordering::Relaxed,
    );

    [
        acc_a.finalize(BETA_SPATIAL),
        acc_rg.finalize(BETA_SPATIAL),
        acc_vy.finalize(BETA_SPATIAL),
    ]
}

/// Build the 3-channel weber pyramid for one side, writing into
/// caller-supplied `out` slots + reusing caller-supplied `caches`.
/// Used by the hot path (score_internal / score_internal_with_warm)
/// so band Vec<f32> capacity persists across calls.
#[allow(clippy::too_many_arguments)]
fn build_one_side_recycle(
    plane_a: &[f32],
    plane_rg: &[f32],
    plane_vy: &[f32],
    w: usize,
    h: usize,
    n_levels: usize,
    caches: &mut [WeberPyramidCache; 3],
    out: &mut [WeberPyramid; 3],
) {
    #[cfg(feature = "parallel")]
    {
        // Split-borrow the 3 cache + 3 out slots.
        let (ca0, ca_rest) = caches.split_at_mut(1);
        let (ca1, ca2) = ca_rest.split_at_mut(1);
        let (o0, o_rest) = out.split_at_mut(1);
        let (o1, o2) = o_rest.split_at_mut(1);
        rayon::join(
            || {
                weber_contrast_pyr_into(plane_a, plane_a, w, h, n_levels, &mut ca0[0], &mut o0[0]);
            },
            || {
                rayon::join(
                    || {
                        weber_contrast_pyr_into(
                            plane_rg,
                            plane_a,
                            w,
                            h,
                            n_levels,
                            &mut ca1[0],
                            &mut o1[0],
                        );
                    },
                    || {
                        weber_contrast_pyr_into(
                            plane_vy,
                            plane_a,
                            w,
                            h,
                            n_levels,
                            &mut ca2[0],
                            &mut o2[0],
                        );
                    },
                );
            },
        );
    }
    #[cfg(not(feature = "parallel"))]
    {
        weber_contrast_pyr_into(
            plane_a,
            plane_a,
            w,
            h,
            n_levels,
            &mut caches[0],
            &mut out[0],
        );
        weber_contrast_pyr_into(
            plane_rg,
            plane_a,
            w,
            h,
            n_levels,
            &mut caches[1],
            &mut out[1],
        );
        weber_contrast_pyr_into(
            plane_vy,
            plane_a,
            w,
            h,
            n_levels,
            &mut caches[2],
            &mut out[2],
        );
    }
}

/// Build REF + DIST weber pyramids into the persistent slots on
/// `Scratch`. Slot Vec<f32> capacities persist across calls.
fn build_both_sides_into(scratch: &mut Scratch, w: usize, h: usize, n_levels: usize) {
    // Need to split-borrow ref planes / dist planes (immutable) vs
    // weber_ref / weber_dist / caches (mutable). All live on
    // `scratch`, so do it by destructuring.
    let Scratch {
        dist_a,
        dist_rg,
        dist_vy,
        ref_a,
        ref_rg,
        ref_vy,
        weber_ref,
        weber_dist,
        weber_cache_ref,
        weber_cache_dist,
        ..
    } = scratch;

    #[cfg(feature = "parallel")]
    {
        rayon::join(
            || {
                build_one_side_recycle(
                    ref_a,
                    ref_rg,
                    ref_vy,
                    w,
                    h,
                    n_levels,
                    weber_cache_ref,
                    weber_ref,
                );
            },
            || {
                build_one_side_recycle(
                    dist_a,
                    dist_rg,
                    dist_vy,
                    w,
                    h,
                    n_levels,
                    weber_cache_dist,
                    weber_dist,
                );
            },
        );
    }
    #[cfg(not(feature = "parallel"))]
    {
        build_one_side_recycle(
            ref_a,
            ref_rg,
            ref_vy,
            w,
            h,
            n_levels,
            weber_cache_ref,
            weber_ref,
        );
        build_one_side_recycle(
            dist_a,
            dist_rg,
            dist_vy,
            w,
            h,
            n_levels,
            weber_cache_dist,
            weber_dist,
        );
    }
}

/// Same as `build_both_sides_into` but only the DIST side (used by
/// `score_internal_with_warm`).
fn build_one_side_dist_into(scratch: &mut Scratch, w: usize, h: usize, n_levels: usize) {
    let Scratch {
        dist_a,
        dist_rg,
        dist_vy,
        weber_dist,
        weber_cache_dist,
        ..
    } = scratch;
    build_one_side_recycle(
        dist_a,
        dist_rg,
        dist_vy,
        w,
        h,
        n_levels,
        weber_cache_dist,
        weber_dist,
    );
}

/// Build the warm REF side using *local* `WeberPyramidCache` slots
/// that are dropped at function exit. Writes the resulting per-channel
/// `WeberPyramid`s into the persistent `scratch.weber_ref` slot so
/// `score_internal_with_warm` can use them without re-building.
///
/// Why local caches: the cache buffers (`gauss_img`/`gauss_l`) are
/// pyramid INTERMEDIATES only needed during the build itself, not
/// during subsequent `score_with_warm_ref` calls. Persisting them in
/// `scratch.weber_cache_ref` would hold ~700 MB of dead memory at
/// 40 MP between warm_reference and the next reuse — Phase 9.YA
/// measured this as a 2 GB peak-heap regression on the
/// cpu-profile-driver-shaped one-shot benchmark. The local-cache
/// variant frees those intermediates before `score_with_warm_ref`
/// runs while still saving the 480 MB DKL-plane allocation that was
/// the original Phase 9.YA Part 1 target.
///
/// On the cold path (`score()`), the persistent caches in
/// `scratch.weber_cache_*` ARE the right choice because the cold loop
/// re-builds both sides each call and benefits from reuse.
fn build_one_side_warm_ref_into(
    plane_a: &[f32],
    plane_rg: &[f32],
    plane_vy: &[f32],
    w: usize,
    h: usize,
    n_levels: usize,
    weber_ref_out: &mut [WeberPyramid; 3],
) {
    let mut caches = [
        WeberPyramidCache::default(),
        WeberPyramidCache::default(),
        WeberPyramidCache::default(),
    ];
    build_one_side_recycle(
        plane_a,
        plane_rg,
        plane_vy,
        w,
        h,
        n_levels,
        &mut caches,
        weber_ref_out,
    );
    // `caches` drop here — gauss_img / gauss_l intermediates are freed
    // (~700 MB at 40 MP).
}
