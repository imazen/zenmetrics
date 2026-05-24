//! Public `Cvvdp` scorer + the end-to-end pipeline orchestration.
//!
//! Mirrors `cvvdp_gpu::host_scalar::predict_jod_still_3ch_capped`
//! algorithmically — the host-scalar path IS the f32-precision
//! reference the GPU pipeline is validated against. We re-use cvvdp-gpu's
//! constants and per-pixel masking/CSF helpers verbatim. The CPU-port
//! contribution is structural: persistent scratch + diffmap output +
//! optional rayon outer parallelism.

use alloc::vec;
use alloc::vec::Vec;

use crate::ReferenceState;
use crate::color::{linear_planes_to_dkl_planar, srgb_to_dkl_planar};
use crate::diffmap::{DiffmapAccum, accumulate_band_diffmap, finalize_diffmap};
use crate::pool::{
    BASEBAND_W, BETA_BAND, BETA_CH, BETA_SPATIAL, IMAGE_INT, PER_CH_W,
    do_pooling_and_jod_still_3ch, lp_norm_mean,
};
use crate::pyramid::{WeberPyramid, band_frequencies, weber_contrast_pyr};
use crate::scratch::Scratch;
use crate::{CvvdpParams, DisplayGeometry, Error, Result};

use cvvdp_gpu::kernels::csf::{
    CSF_BASEBAND_RHO, CsfChannel, LOG_L_BKG_AXIS, N_L_BKG, SENSITIVITY_CORRECTION_DB,
    precompute_logs_row,
};
use cvvdp_gpu::kernels::masking::CH_GAIN;

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
    warm: Option<ReferenceState>,
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
        Ok(Self {
            width: w,
            height: h,
            params,
            ppd,
            scratch: Scratch::new(w, h),
            warm: None,
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
        self.warm = None;
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
        self.warm = None;
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
    pub fn warm_reference(&mut self, ref_srgb: &[u8]) -> Result<()> {
        self.check_srgb(ref_srgb)?;
        let display = self.params.display;
        let w = self.width;
        let h = self.height;
        let mut ref_a = vec![0.0; w * h];
        let mut ref_rg = vec![0.0; w * h];
        let mut ref_vy = vec![0.0; w * h];
        srgb_to_dkl_planar(
            ref_srgb,
            w,
            h,
            display,
            &mut ref_a,
            &mut ref_rg,
            &mut ref_vy,
        );
        let n_levels = band_frequencies(self.ppd, w, h).len();
        let [weber_a, weber_rg, weber_vy] =
            build_one_side(&ref_a, &ref_rg, &ref_vy, w, h, n_levels);
        self.warm = Some(ReferenceState {
            w,
            h,
            planes: [ref_a, ref_rg, ref_vy],
            weber: [weber_a, weber_rg, weber_vy],
            display,
        });
        Ok(())
    }

    /// Score against the warm reference.
    pub fn score_with_warm_ref(&mut self, dist_srgb: &[u8]) -> Result<f32> {
        if self.warm.is_none() {
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
        if self.warm.is_none() {
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
        self.warm = None;
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
        self.warm = None;
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

        // The 6 weber pyramid builds are fully independent. Each owns
        // its own PyramidScratch. With rayon, run them in parallel; on
        // a 7950X with 8 channels of work that's ~5× wall-time
        // reduction over sequential. Each side's PyramidScratch is
        // allocated fresh inside the closure — we drop the persistent
        // self.scratch.pyr slot for these (it's still used by
        // warm_reference).
        let (ref_weber, dist_weber) = build_both_sides(
            &self.scratch.ref_a,
            &self.scratch.ref_rg,
            &self.scratch.ref_vy,
            &self.scratch.dist_a,
            &self.scratch.dist_rg,
            &self.scratch.dist_vy,
            w,
            h,
            n_levels,
        );

        let (jod, diffmap) = self.fold_bands(&ref_weber, &dist_weber, n_levels, w, h, want_diffmap);
        Ok((jod, diffmap))
    }

    fn score_internal_with_warm(&mut self, want_diffmap: bool) -> Result<(f32, Option<Vec<f32>>)> {
        // Build dist pyramids in parallel; REF pyramids come from
        // warm cache.
        let w = self.width;
        let h = self.height;
        let n_levels = band_frequencies(self.ppd, w, h).len();
        let dist_weber = build_one_side(
            &self.scratch.dist_a,
            &self.scratch.dist_rg,
            &self.scratch.dist_vy,
            w,
            h,
            n_levels,
        );

        // Pull the warm reference out so we can call fold_bands with
        // a clean &mut self. Restored before returning so subsequent
        // score_with_warm_ref calls still find it cached.
        let warm = self.warm.take().expect("checked by caller");
        let (jod, diffmap) =
            self.fold_bands(&warm.weber, &dist_weber, n_levels, w, h, want_diffmap);
        self.warm = Some(warm);
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
    /// as the inner-loop body for the parallel path.
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

            // Hoist the rho-axis interp out of the per-pixel loop.
            // `precompute_logs_row` returns a 32-entry row of
            // `log10(S)` parameterized by `log_l_bkg`, one per
            // channel.
            let logs_row_a = precompute_logs_row(rho, channels[0]);
            let logs_row_rg = precompute_logs_row(rho, channels[1]);
            let logs_row_vy = precompute_logs_row(rho, channels[2]);
            // Sanity — pinned by N_L_BKG below.
            debug_assert_eq!(logs_row_a.len(), N_L_BKG);
            debug_assert_eq!(LOG_L_BKG_AXIS.len(), N_L_BKG);

            // Compute T_p + R_p per channel via the fast CSF path.
            let mut t_p_per_ch: [Vec<f32>; 3] = [vec![0.0; n_px], vec![0.0; n_px], vec![0.0; n_px]];
            let mut r_p_per_ch: [Vec<f32>; 3] = [vec![0.0; n_px], vec![0.0; n_px], vec![0.0; n_px]];
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
                t_p_per_ch[0][i] = dis_a_band[i] * bm_sa * ch_gain_a;
                t_p_per_ch[1][i] = dis_rg_band[i] * bm_srg * ch_gain_rg;
                t_p_per_ch[2][i] = dis_vy_band[i] * bm_svy * ch_gain_vy;
                r_p_per_ch[0][i] = ref_a_band[i] * bm_sa * ch_gain_a;
                r_p_per_ch[1][i] = ref_rg_band[i] * bm_srg * ch_gain_rg;
                r_p_per_ch[2][i] = ref_vy_band[i] * bm_svy * ch_gain_vy;
            }

            // Baseband bypass vs full mult-mutual.
            let d_per_ch: [Vec<f32>; 3] = if is_baseband {
                let mut out: [Vec<f32>; 3] = [vec![0.0; n_px], vec![0.0; n_px], vec![0.0; n_px]];
                for i in 0..n_px {
                    let log_l = log_l_bkg_band[i];
                    let s_a = apply_csf_row_per_pixel(log_l, &logs_row_a);
                    let s_rg = apply_csf_row_per_pixel(log_l, &logs_row_rg);
                    let s_vy = apply_csf_row_per_pixel(log_l, &logs_row_vy);
                    let diff_a = dis_a_band[i] - ref_a_band[i];
                    let diff_rg = dis_rg_band[i] - ref_rg_band[i];
                    let diff_vy = dis_vy_band[i] - ref_vy_band[i];
                    out[0][i] = diff_a.abs() * s_a;
                    out[1][i] = diff_rg.abs() * s_rg;
                    out[2][i] = diff_vy.abs() * s_vy;
                }
                out
            } else {
                // Fast in-scratch masking. Borrow each scratch slot via
                // a fresh tuple to avoid &mut self alias.
                let s = &mut self.scratch;
                let mut d_a = core::mem::take(&mut s.d_a);
                let mut d_rg = core::mem::take(&mut s.d_rg);
                let mut d_vy = core::mem::take(&mut s.d_vy);
                let mut m_mm_a = core::mem::take(&mut s.m_mm_a);
                let mut m_mm_rg = core::mem::take(&mut s.m_mm_rg);
                let mut m_mm_vy = core::mem::take(&mut s.m_mm_vy);
                let mut term_a = core::mem::take(&mut s.t_p_a);
                let mut term_rg = core::mem::take(&mut s.t_p_rg);
                let mut term_vy = core::mem::take(&mut s.t_p_vy);
                let mut pu_scratch = core::mem::take(&mut s.pu_h);
                mult_mutual_band_into(
                    &t_p_per_ch,
                    &r_p_per_ch,
                    bw,
                    bh,
                    &mut d_a,
                    &mut d_rg,
                    &mut d_vy,
                    &mut m_mm_a,
                    &mut m_mm_rg,
                    &mut m_mm_vy,
                    &mut term_a,
                    &mut term_rg,
                    &mut term_vy,
                    &mut pu_scratch,
                );
                let out = [d_a, d_rg, d_vy];
                // Stash scratch back.
                s.m_mm_a = m_mm_a;
                s.m_mm_rg = m_mm_rg;
                s.m_mm_vy = m_mm_vy;
                s.t_p_a = term_a;
                s.t_p_rg = term_rg;
                s.t_p_vy = term_vy;
                s.pu_h = pu_scratch;
                // d_a/d_rg/d_vy live on as `out`; stash empty Vecs as
                // placeholders so the field is always Vec (cleared on
                // next call).
                s.d_a = Vec::new();
                s.d_rg = Vec::new();
                s.d_vy = Vec::new();
                out
            };

            // Spatial pool per channel.
            let mut q_band = [0.0_f32; 3];
            for c in 0..3 {
                q_band[c] = lp_norm_mean(&d_per_ch[c], BETA_SPATIAL);
            }
            q_per_ch.push(q_band);

            // Accumulate diffmap.
            if let Some(acc) = accum.as_mut() {
                accumulate_band_diffmap(acc, &d_per_ch, bw, bh, is_baseband, n_levels);
            }
        }

        let jod = do_pooling_and_jod_still_3ch(&q_per_ch);
        let diffmap = accum.map(finalize_diffmap);
        (jod, diffmap)
    }

    /// Parallel band loop. Each band runs independently on a rayon
    /// thread; results merge via reduce at the end. Diffmap path
    /// allocates per-band accumulators since the bilinear upsample
    /// already operates on its own band-sized buffer.
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

        // Each band's result: (q_per_ch, optional accumulated diffmap).
        let band_results: Vec<([f32; 3], Option<DiffmapAccum>)> = (0..n_levels)
            .into_par_iter()
            .map(|k| {
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

                let d_per_ch: [Vec<f32>; 3] = if is_baseband {
                    let mut out: [Vec<f32>; 3] =
                        [vec![0.0; n_px], vec![0.0; n_px], vec![0.0; n_px]];
                    for i in 0..n_px {
                        let log_l = log_l_bkg_band[i];
                        let s_a = apply_csf_row_per_pixel(log_l, &logs_row_a);
                        let s_rg = apply_csf_row_per_pixel(log_l, &logs_row_rg);
                        let s_vy = apply_csf_row_per_pixel(log_l, &logs_row_vy);
                        let diff_a = dis_a_band[i] - ref_a_band[i];
                        let diff_rg = dis_rg_band[i] - ref_rg_band[i];
                        let diff_vy = dis_vy_band[i] - ref_vy_band[i];
                        out[0][i] = diff_a.abs() * s_a;
                        out[1][i] = diff_rg.abs() * s_rg;
                        out[2][i] = diff_vy.abs() * s_vy;
                    }
                    out
                } else {
                    let mut t_p: [Vec<f32>; 3] =
                        [vec![0.0; n_px], vec![0.0; n_px], vec![0.0; n_px]];
                    let mut r_p: [Vec<f32>; 3] =
                        [vec![0.0; n_px], vec![0.0; n_px], vec![0.0; n_px]];
                    for i in 0..n_px {
                        let log_l = log_l_bkg_band[i];
                        let s_a = apply_csf_row_per_pixel(log_l, &logs_row_a);
                        let s_rg = apply_csf_row_per_pixel(log_l, &logs_row_rg);
                        let s_vy = apply_csf_row_per_pixel(log_l, &logs_row_vy);
                        let bm_sa = band_mul * s_a;
                        let bm_srg = band_mul * s_rg;
                        let bm_svy = band_mul * s_vy;
                        t_p[0][i] = dis_a_band[i] * bm_sa * ch_gain_a;
                        t_p[1][i] = dis_rg_band[i] * bm_srg * ch_gain_rg;
                        t_p[2][i] = dis_vy_band[i] * bm_svy * ch_gain_vy;
                        r_p[0][i] = ref_a_band[i] * bm_sa * ch_gain_a;
                        r_p[1][i] = ref_rg_band[i] * bm_srg * ch_gain_rg;
                        r_p[2][i] = ref_vy_band[i] * bm_svy * ch_gain_vy;
                    }
                    // Per-thread scratch (no shared self.scratch).
                    let mut d_a = Vec::new();
                    let mut d_rg = Vec::new();
                    let mut d_vy = Vec::new();
                    let mut m_mm_a = Vec::new();
                    let mut m_mm_rg = Vec::new();
                    let mut m_mm_vy = Vec::new();
                    let mut term_a = Vec::new();
                    let mut term_rg = Vec::new();
                    let mut term_vy = Vec::new();
                    let mut pu_scratch = Vec::new();
                    mult_mutual_band_into(
                        &t_p,
                        &r_p,
                        bw,
                        bh,
                        &mut d_a,
                        &mut d_rg,
                        &mut d_vy,
                        &mut m_mm_a,
                        &mut m_mm_rg,
                        &mut m_mm_vy,
                        &mut term_a,
                        &mut term_rg,
                        &mut term_vy,
                        &mut pu_scratch,
                    );
                    [d_a, d_rg, d_vy]
                };

                let mut q_band = [0.0_f32; 3];
                for c in 0..3 {
                    q_band[c] = lp_norm_mean(&d_per_ch[c], BETA_SPATIAL);
                }

                let band_accum = if want_diffmap {
                    let mut acc = DiffmapAccum::new(w, h);
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

    /// Drop the warm reference cache.
    pub fn drop_warm_reference(&mut self) {
        self.warm = None;
    }

    /// Whether a warm reference is currently cached.
    pub fn has_warm_reference(&self) -> bool {
        self.warm.is_some()
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

/// Build the 3-channel weber pyramid for one side (REF or DIST) using
/// rayon parallelism per channel when the `parallel` feature is on,
/// falling back to sequential otherwise.
fn build_one_side(
    plane_a: &[f32],
    plane_rg: &[f32],
    plane_vy: &[f32],
    w: usize,
    h: usize,
    n_levels: usize,
) -> [WeberPyramid; 3] {
    #[cfg(feature = "parallel")]
    {
        // Each closure allocates its own PyramidScratch. The
        // overhead of the 3 fresh scratch buffers is dwarfed by the
        // 3× speedup over sequential.
        let mut pyramids: [Option<WeberPyramid>; 3] = [None, None, None];
        let (a, rg_vy) = rayon::join(
            || {
                let mut s = crate::pyramid::PyramidScratch::default();
                weber_contrast_pyr(plane_a, plane_a, w, h, n_levels, &mut s)
            },
            || {
                rayon::join(
                    || {
                        let mut s = crate::pyramid::PyramidScratch::default();
                        weber_contrast_pyr(plane_rg, plane_a, w, h, n_levels, &mut s)
                    },
                    || {
                        let mut s = crate::pyramid::PyramidScratch::default();
                        weber_contrast_pyr(plane_vy, plane_a, w, h, n_levels, &mut s)
                    },
                )
            },
        );
        pyramids[0] = Some(a);
        pyramids[1] = Some(rg_vy.0);
        pyramids[2] = Some(rg_vy.1);
        [
            pyramids[0].take().unwrap(),
            pyramids[1].take().unwrap(),
            pyramids[2].take().unwrap(),
        ]
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut s = crate::pyramid::PyramidScratch::default();
        let a = weber_contrast_pyr(plane_a, plane_a, w, h, n_levels, &mut s);
        let rg = weber_contrast_pyr(plane_rg, plane_a, w, h, n_levels, &mut s);
        let vy = weber_contrast_pyr(plane_vy, plane_a, w, h, n_levels, &mut s);
        [a, rg, vy]
    }
}

/// Build REF + DIST weber pyramids in parallel (or sequentially if
/// `parallel` feature is off).
#[allow(clippy::too_many_arguments)]
fn build_both_sides(
    ref_a: &[f32],
    ref_rg: &[f32],
    ref_vy: &[f32],
    dist_a: &[f32],
    dist_rg: &[f32],
    dist_vy: &[f32],
    w: usize,
    h: usize,
    n_levels: usize,
) -> ([WeberPyramid; 3], [WeberPyramid; 3]) {
    #[cfg(feature = "parallel")]
    {
        rayon::join(
            || build_one_side(ref_a, ref_rg, ref_vy, w, h, n_levels),
            || build_one_side(dist_a, dist_rg, dist_vy, w, h, n_levels),
        )
    }
    #[cfg(not(feature = "parallel"))]
    {
        (
            build_one_side(ref_a, ref_rg, ref_vy, w, h, n_levels),
            build_one_side(dist_a, dist_rg, dist_vy, w, h, n_levels),
        )
    }
}
