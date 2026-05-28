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
use crate::scratch::{BandWorkspace, Scratch};
use crate::{CvvdpParams, DisplayGeometry, Error, Result};

use crate::kernels::csf::{
    CSF_BASEBAND_RHO, CsfChannel, LOG_L_BKG_AXIS, N_L_BKG, SENSITIVITY_CORRECTION_DB,
    precompute_logs_row,
};
use crate::kernels::masking::CH_GAIN;

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

        let mut q_per_ch: Vec<[f32; 3]> = Vec::with_capacity(n_levels);
        let mut accum = if want_diffmap {
            Some(DiffmapAccum::new(w, h))
        } else {
            None
        };

        // Reuse band_ws[0] as the single sequential scratch slot.
        self.scratch.ensure_band_ws(1);
        let ws = &mut self.scratch.band_ws[0];

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

            let ref_a_band = &ref_weber[0].bands[k].data;
            let ref_rg_band = &ref_weber[1].bands[k].data;
            let ref_vy_band = &ref_weber[2].bands[k].data;
            let dis_a_band = &dist_weber[0].bands[k].data;
            let dis_rg_band = &dist_weber[1].bands[k].data;
            let dis_vy_band = &dist_weber[2].bands[k].data;
            let ch_gain_a = CH_GAIN[0];
            let ch_gain_rg = CH_GAIN[1];
            let ch_gain_vy = CH_GAIN[2];
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
            let mut q_band = [0.0_f32; 3];
            q_band[0] = lp_norm_mean(&ws.d_a, BETA_SPATIAL);
            q_band[1] = lp_norm_mean(&ws.d_rg, BETA_SPATIAL);
            q_band[2] = lp_norm_mean(&ws.d_vy, BETA_SPATIAL);
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

        // Grow the per-band workspace vec to at least n_levels and
        // borrow it mutably so each band closure gets its own slot.
        self.scratch.ensure_band_ws(n_levels);
        let ws_slice: &mut [BandWorkspace] = &mut self.scratch.band_ws[..n_levels];

        // Each band's result: (q_per_ch, optional accumulated diffmap).
        let band_results: Vec<([f32; 3], Option<DiffmapAccum>)> = ws_slice
            .par_iter_mut()
            .enumerate()
            .map(|(k, ws)| {
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
                    // Reuse d_a/d_rg/d_vy slots in the workspace.
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
                    // Reuse t_p_*/r_p_* slots in the workspace.
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
                    // `mult_mutual_band_into` requires `&[Vec<f32>; 3]`
                    // for t_p and r_p. Move our workspace slots into
                    // local arrays, call, then move them back.
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
                    // Restore the t_p / r_p slots so next call reuses
                    // the Vec capacity.
                    let [t_a, t_rg, t_vy] = t_p_taken;
                    let [r_a, r_rg, r_vy] = r_p_taken;
                    ws.t_p_a = t_a;
                    ws.t_p_rg = t_rg;
                    ws.t_p_vy = t_vy;
                    ws.r_p_a = r_a;
                    ws.r_p_rg = r_rg;
                    ws.r_p_vy = r_vy;
                }

                let mut q_band = [0.0_f32; 3];
                q_band[0] = lp_norm_mean(&ws.d_a, BETA_SPATIAL);
                q_band[1] = lp_norm_mean(&ws.d_rg, BETA_SPATIAL);
                q_band[2] = lp_norm_mean(&ws.d_vy, BETA_SPATIAL);

                let band_accum = if want_diffmap {
                    let mut acc = DiffmapAccum::new(w, h);
                    // accumulate_band_diffmap takes `&[Vec<f32>; 3]`.
                    // Build a transient array view by cloning out the
                    // d Vecs. We could avoid this by changing
                    // accumulate_band_diffmap to take three &[f32]
                    // refs, which is a follow-on chunk.
                    let d_per_ch: [Vec<f32>; 3] =
                        [ws.d_a.clone(), ws.d_rg.clone(), ws.d_vy.clone()];
                    accumulate_band_diffmap(&mut acc, &d_per_ch, bw, bh, is_baseband, n_levels);
                    Some(acc)
                } else {
                    None
                };

                (q_band, band_accum)
            })
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

    /// Strip-mode score: walks image in horizontal slabs of
    /// `strip_height` rows + halo. Designed to reduce peak heap on
    /// 40 MP+ inputs where the full pipeline's 11.3 GB peak crowds
    /// the budget.
    ///
    /// # Status (Phase 9.Z.A)
    ///
    /// **This method is API-only — it delegates to [`Self::score`]
    /// and does NOT reduce peak heap.** The memory-bounded walker is
    /// queued; reasons:
    /// 1. cvvdp's 9-level Weber pyramid + per-band σ=3 PU blur
    ///    produces cumulative halo at scale 0 of `~8 × 2^k` rows for
    ///    level k. At level 8 of 4096² this is `~2048 rows ≈ 50%
    ///    of image height` — strip pattern requires hybrid K_SPLIT
    ///    dispatch (shallow bands per-strip, deep bands full-image)
    ///    as documented in the GPU cvvdp Mode E strip design at
    ///    `crates/cvvdp-gpu/docs/STRIP_PROCESSING.md`.
    /// 2. The band-fold's spatial Minkowski pool IS strip-associative
    ///    (`Σ|d|^β` + `n` accumulate cleanly across strips); the
    ///    refactor is plumbing, not algorithmic.
    /// 3. The cvvdp-gpu Phase 1 + Phase 2 investigation explicitly
    ///    documented this as multi-day work; the CPU port faces the
    ///    same structural blocker.
    ///
    /// The API is exposed NOW so the orchestrator's `cpu_adapter` can
    /// wire a `MemoryMode::Strip` dispatch slot for cvvdp without API
    /// churn when the walker ships. **Until then, this method delivers
    /// correct scores at no memory benefit.**
    pub fn score_strip(
        &mut self,
        ref_srgb: &[u8],
        dist_srgb: &[u8],
        _strip_height: u32,
    ) -> Result<f32> {
        self.score(ref_srgb, dist_srgb)
    }

    /// Strip-mode score against the warm reference. See
    /// [`Self::score_strip`] for implementation status.
    ///
    /// Phase 9.Z.A: delegates to [`Self::score_with_warm_ref`]; the
    /// memory-bounded walker is queued. API exists for orchestrator
    /// `MemoryMode::CachedStrip` integration without churning the
    /// public API when the walker ships.
    pub fn score_with_warm_ref_strip(
        &mut self,
        dist_srgb: &[u8],
        _strip_height: u32,
    ) -> Result<f32> {
        self.score_with_warm_ref(dist_srgb)
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
