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

use crate::color::{linear_planes_to_dkl_planar, srgb_to_dkl_planar};
use crate::diffmap::{accumulate_band_diffmap, finalize_diffmap, DiffmapAccum};
use crate::pool::{
    do_pooling_and_jod_still_3ch, lp_norm_mean, BASEBAND_W, BETA_BAND, BETA_CH, BETA_SPATIAL,
    IMAGE_INT, PER_CH_W,
};
use crate::pyramid::{band_frequencies, weber_contrast_pyr, WeberPyramid};
use crate::scratch::Scratch;
use crate::ReferenceState;
use crate::{CvvdpParams, DisplayGeometry, Error, Result};

use cvvdp_gpu::kernels::csf::{CsfChannel, CSF_BASEBAND_RHO};
use cvvdp_gpu::kernels::masking::{mult_mutual_band, CH_GAIN};

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
        srgb_to_dkl_planar(ref_srgb, w, h, display, &mut ref_a, &mut ref_rg, &mut ref_vy);
        let n_levels = band_frequencies(self.ppd, w, h).len();
        let weber_a = weber_contrast_pyr(&ref_a, &ref_a, w, h, n_levels, &mut self.scratch.pyr);
        let weber_rg = weber_contrast_pyr(&ref_rg, &ref_a, w, h, n_levels, &mut self.scratch.pyr);
        let weber_vy = weber_contrast_pyr(&ref_vy, &ref_a, w, h, n_levels, &mut self.scratch.pyr);
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

        // Build per-channel Weber pyramids for REF using REF's A as L_bkg.
        let ref_a_clone = self.scratch.ref_a.clone();
        let ref_weber_a = weber_contrast_pyr(
            &self.scratch.ref_a,
            &ref_a_clone,
            w,
            h,
            n_levels,
            &mut self.scratch.pyr,
        );
        let ref_weber_rg = weber_contrast_pyr(
            &self.scratch.ref_rg,
            &ref_a_clone,
            w,
            h,
            n_levels,
            &mut self.scratch.pyr,
        );
        let ref_weber_vy = weber_contrast_pyr(
            &self.scratch.ref_vy,
            &ref_a_clone,
            w,
            h,
            n_levels,
            &mut self.scratch.pyr,
        );
        let dist_a_clone = self.scratch.dist_a.clone();
        let dist_weber_a = weber_contrast_pyr(
            &self.scratch.dist_a,
            &dist_a_clone,
            w,
            h,
            n_levels,
            &mut self.scratch.pyr,
        );
        let dist_weber_rg = weber_contrast_pyr(
            &self.scratch.dist_rg,
            &dist_a_clone,
            w,
            h,
            n_levels,
            &mut self.scratch.pyr,
        );
        let dist_weber_vy = weber_contrast_pyr(
            &self.scratch.dist_vy,
            &dist_a_clone,
            w,
            h,
            n_levels,
            &mut self.scratch.pyr,
        );

        let ref_weber = [ref_weber_a, ref_weber_rg, ref_weber_vy];
        let dist_weber = [dist_weber_a, dist_weber_rg, dist_weber_vy];

        let (jod, diffmap) = self.fold_bands(&ref_weber, &dist_weber, n_levels, w, h, want_diffmap);
        Ok((jod, diffmap))
    }

    fn score_internal_with_warm(&mut self, want_diffmap: bool) -> Result<(f32, Option<Vec<f32>>)> {
        // Build dist pyramids; REF pyramids come from warm cache.
        let w = self.width;
        let h = self.height;
        let n_levels = band_frequencies(self.ppd, w, h).len();
        let dist_a_clone = self.scratch.dist_a.clone();
        let dist_weber_a = weber_contrast_pyr(
            &self.scratch.dist_a,
            &dist_a_clone,
            w,
            h,
            n_levels,
            &mut self.scratch.pyr,
        );
        let dist_weber_rg = weber_contrast_pyr(
            &self.scratch.dist_rg,
            &dist_a_clone,
            w,
            h,
            n_levels,
            &mut self.scratch.pyr,
        );
        let dist_weber_vy = weber_contrast_pyr(
            &self.scratch.dist_vy,
            &dist_a_clone,
            w,
            h,
            n_levels,
            &mut self.scratch.pyr,
        );
        let dist_weber = [dist_weber_a, dist_weber_rg, dist_weber_vy];

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

            // Compute T_p + R_p per channel.
            let mut t_p_per_ch: [Vec<f32>; 3] =
                [vec![0.0; n_px], vec![0.0; n_px], vec![0.0; n_px]];
            let mut r_p_per_ch: [Vec<f32>; 3] =
                [vec![0.0; n_px], vec![0.0; n_px], vec![0.0; n_px]];
            for i in 0..n_px {
                let log_l = log_l_bkg_band[i];
                let s_a = cvvdp_gpu::kernels::csf::sensitivity_corrected_scalar(
                    rho,
                    log_l,
                    channels[0],
                );
                let s_rg = cvvdp_gpu::kernels::csf::sensitivity_corrected_scalar(
                    rho,
                    log_l,
                    channels[1],
                );
                let s_vy = cvvdp_gpu::kernels::csf::sensitivity_corrected_scalar(
                    rho,
                    log_l,
                    channels[2],
                );
                t_p_per_ch[0][i] = band_mul * dist_weber[0].bands[k].data[i] * s_a * CH_GAIN[0];
                t_p_per_ch[1][i] = band_mul * dist_weber[1].bands[k].data[i] * s_rg * CH_GAIN[1];
                t_p_per_ch[2][i] = band_mul * dist_weber[2].bands[k].data[i] * s_vy * CH_GAIN[2];
                r_p_per_ch[0][i] = band_mul * ref_weber[0].bands[k].data[i] * s_a * CH_GAIN[0];
                r_p_per_ch[1][i] = band_mul * ref_weber[1].bands[k].data[i] * s_rg * CH_GAIN[1];
                r_p_per_ch[2][i] = band_mul * ref_weber[2].bands[k].data[i] * s_vy * CH_GAIN[2];
            }

            // Baseband bypass vs full mult-mutual.
            let d_per_ch: [Vec<f32>; 3] = if is_baseband {
                let mut out: [Vec<f32>; 3] =
                    [vec![0.0; n_px], vec![0.0; n_px], vec![0.0; n_px]];
                for i in 0..n_px {
                    let log_l = log_l_bkg_band[i];
                    let s_a = cvvdp_gpu::kernels::csf::sensitivity_corrected_scalar(
                        rho,
                        log_l,
                        channels[0],
                    );
                    let s_rg = cvvdp_gpu::kernels::csf::sensitivity_corrected_scalar(
                        rho,
                        log_l,
                        channels[1],
                    );
                    let s_vy = cvvdp_gpu::kernels::csf::sensitivity_corrected_scalar(
                        rho,
                        log_l,
                        channels[2],
                    );
                    let diff_a = dist_weber[0].bands[k].data[i] - ref_weber[0].bands[k].data[i];
                    let diff_rg =
                        dist_weber[1].bands[k].data[i] - ref_weber[1].bands[k].data[i];
                    let diff_vy =
                        dist_weber[2].bands[k].data[i] - ref_weber[2].bands[k].data[i];
                    out[0][i] = diff_a.abs() * s_a;
                    out[1][i] = diff_rg.abs() * s_rg;
                    out[2][i] = diff_vy.abs() * s_vy;
                }
                out
            } else {
                mult_mutual_band(&t_p_per_ch, &r_p_per_ch, bw, bh)
            };

            // Spatial pool per channel.
            let mut q_band = [0.0_f32; 3];
            for c in 0..3 {
                q_band[c] = lp_norm_mean(&d_per_ch[c], BETA_SPATIAL);
            }
            q_per_ch.push(q_band);

            // Accumulate diffmap.
            if let Some(acc) = accum.as_mut() {
                accumulate_band_diffmap(
                    acc,
                    &d_per_ch,
                    bw,
                    bh,
                    is_baseband,
                    n_levels,
                );
            }
        }

        let jod = do_pooling_and_jod_still_3ch(&q_per_ch);
        let diffmap = accum.map(finalize_diffmap);
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
}

/// Helper for `score()` to obtain the 6 plane scratches from
/// `Scratch` as separate `&mut Vec<f32>`. Required because Rust's
/// borrow checker rejects 6 simultaneous `&mut` of `self.scratch.*`
/// directly through field projection on a single line in some
/// configurations; this helper makes the split-borrow explicit.
fn scratch_dkl_planes(
    scratch: &mut Scratch,
) -> (
    &mut Vec<f32>,
    &mut Vec<f32>,
    &mut Vec<f32>,
    &mut Vec<f32>,
    &mut Vec<f32>,
    &mut Vec<f32>,
) {
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
