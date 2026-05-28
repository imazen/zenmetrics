//! Top-level IW-SSIM scorer — composes pyramid + SSIM stats + IW maps.

use alloc::vec::Vec;

use crate::filters::{SCALE_WEIGHTS, SSIM_WIN_LEN};
use crate::params::IwssimParams;
use crate::pyramid::{PyrLevel, build_laplacian_pyramid, pyramid_dims};
use crate::ssim::compute_cs;
use crate::weights::compute_iw_maps;
use crate::{Error, IwssimScore, MIN_NATIVE_DIM, NUM_SCALES, Result, rgb_u8_to_gray_bt601};

/// Scratch-owning IW-SSIM scorer.
///
/// Lifecycle:
///
/// - [`Iwssim::new`] / [`Iwssim::with_params`] build a scorer sized for
///   `width × height` images.
/// - [`Iwssim::score`] / [`Iwssim::score_gray`] one-shot scoring.
/// - [`Iwssim::warm_reference`] caches the reference's pyramid + IW
///   weight maps. [`Iwssim::score_with_warm_ref`] then re-uses them.
pub struct Iwssim {
    width: u32,
    height: u32,
    params: IwssimParams,
    /// Working dims after small-image padding (might equal original
    /// when `width >= MIN_NATIVE_DIM && height >= MIN_NATIVE_DIM`).
    work_w: usize,
    work_h: usize,
    /// Cached state from `warm_reference`, if any.
    warm: Option<WarmState>,
}

/// State cached by `warm_reference`. Holds the per-scale Laplacian
/// bands + per-scale Gaussian bands (for parent-band lookup) so the
/// scoring path only has to build the distorted side.
struct WarmState {
    /// Reference Laplacian bands (one per scale, finest first).
    lp_ref: Vec<Vec<f32>>,
    /// Reference Gaussian bands (one per scale).
    g_ref: Vec<Vec<f32>>,
}

impl Iwssim {
    /// Construct an IW-SSIM scorer with default params for
    /// `width × height` images.
    pub fn new(width: u32, height: u32) -> Result<Self> {
        Self::with_params(width, height, IwssimParams::default())
    }

    /// Construct with custom params.
    pub fn with_params(width: u32, height: u32, params: IwssimParams) -> Result<Self> {
        if width.min(height) < MIN_NATIVE_DIM && !params.allow_small {
            return Err(Error::InvalidImageSize { width, height });
        }
        let work_w = width.max(MIN_NATIVE_DIM) as usize;
        let work_h = height.max(MIN_NATIVE_DIM) as usize;
        Ok(Self {
            width,
            height,
            params,
            work_w,
            work_h,
            warm: None,
        })
    }

    /// Configured `(width, height)`.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// `true` if a warm reference is cached.
    pub fn has_warm_reference(&self) -> bool {
        self.warm.is_some()
    }

    /// Drop the cached reference state.
    pub fn clear_warm_reference(&mut self) {
        self.warm = None;
    }

    /// Score `ref_rgb` vs `dis_rgb` (both `width × height × 3` sRGB-u8).
    pub fn score(&mut self, ref_rgb: &[u8], dis_rgb: &[u8]) -> Result<IwssimScore> {
        let expected = (self.width as usize) * (self.height as usize) * 3;
        if ref_rgb.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: ref_rgb.len(),
            });
        }
        if dis_rgb.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: dis_rgb.len(),
            });
        }
        let mut ref_gray = alloc::vec![0.0_f32; (self.width as usize) * (self.height as usize)];
        let mut dis_gray = alloc::vec![0.0_f32; (self.width as usize) * (self.height as usize)];
        rgb_u8_to_gray_bt601(ref_rgb, &mut ref_gray);
        rgb_u8_to_gray_bt601(dis_rgb, &mut dis_gray);
        self.score_gray(&ref_gray, &dis_gray)
    }

    /// Score from grayscale-f32 inputs (`width * height` samples each).
    ///
    /// This is the canonical entry point — `score()` simply calls this
    /// after a BT.601 RGB → gray conversion.
    pub fn score_gray(&mut self, ref_gray: &[f32], dis_gray: &[f32]) -> Result<IwssimScore> {
        let expected = (self.width as usize) * (self.height as usize);
        if ref_gray.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: ref_gray.len(),
            });
        }
        if dis_gray.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: dis_gray.len(),
            });
        }
        let (ref_work, dis_work) = (self.pad_gray(ref_gray), self.pad_gray(dis_gray));
        score_from_gray(&ref_work, &dis_work, self.work_w, self.work_h, &self.params)
    }

    /// Cache `ref_rgb`'s pyramid state — subsequent
    /// [`Self::score_with_warm_ref`] calls skip re-building the
    /// reference pyramid.
    pub fn warm_reference(&mut self, ref_rgb: &[u8]) -> Result<()> {
        let expected = (self.width as usize) * (self.height as usize) * 3;
        if ref_rgb.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: ref_rgb.len(),
            });
        }
        let mut ref_gray = alloc::vec![0.0_f32; (self.width as usize) * (self.height as usize)];
        rgb_u8_to_gray_bt601(ref_rgb, &mut ref_gray);
        self.warm_reference_gray(&ref_gray)
    }

    /// Cache from gray-f32 reference (mirrors [`Self::warm_reference`]).
    pub fn warm_reference_gray(&mut self, ref_gray: &[f32]) -> Result<()> {
        let expected = (self.width as usize) * (self.height as usize);
        if ref_gray.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: ref_gray.len(),
            });
        }
        let work = self.pad_gray(ref_gray);
        let levels = build_laplacian_pyramid(&work, self.work_w, self.work_h, NUM_SCALES);
        let (lp_ref, g_ref) = split_levels(levels);
        self.warm = Some(WarmState { lp_ref, g_ref });
        Ok(())
    }

    /// Score a candidate against the previously-warmed reference.
    pub fn score_with_warm_ref(&mut self, dis_rgb: &[u8]) -> Result<IwssimScore> {
        let expected = (self.width as usize) * (self.height as usize) * 3;
        if dis_rgb.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: dis_rgb.len(),
            });
        }
        let mut dis_gray = alloc::vec![0.0_f32; (self.width as usize) * (self.height as usize)];
        rgb_u8_to_gray_bt601(dis_rgb, &mut dis_gray);
        self.score_with_warm_ref_gray(&dis_gray)
    }

    /// Score from gray-f32 candidate against the warmed reference.
    pub fn score_with_warm_ref_gray(&mut self, dis_gray: &[f32]) -> Result<IwssimScore> {
        let expected = (self.width as usize) * (self.height as usize);
        if dis_gray.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: dis_gray.len(),
            });
        }
        let work = self.pad_gray(dis_gray);
        let warm = self.warm.as_ref().ok_or(Error::NoWarmReference)?;
        let dis_levels = build_laplacian_pyramid(&work, self.work_w, self.work_h, NUM_SCALES);
        let (lp_dis, _g_dis) = split_levels(dis_levels);
        score_with_split(
            &warm.lp_ref,
            &lp_dis,
            &warm.g_ref,
            self.work_w,
            self.work_h,
            &self.params,
        )
    }

    /// Tile / replicate `src` to `(work_w, work_h)` when `allow_small`
    /// is enabled and the source is smaller; otherwise the source is
    /// copied 1:1.
    fn pad_gray(&self, src: &[f32]) -> Vec<f32> {
        let sw = self.width as usize;
        let sh = self.height as usize;
        if sw == self.work_w && sh == self.work_h {
            return src.to_vec();
        }
        let mut out = alloc::vec![0.0_f32; self.work_w * self.work_h];
        for y in 0..self.work_h {
            let sy = y % sh;
            for x in 0..self.work_w {
                let sx = x % sw;
                out[y * self.work_w + x] = src[sy * sw + sx];
            }
        }
        out
    }
}

fn split_levels(levels: Vec<PyrLevel>) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut lp = Vec::with_capacity(levels.len());
    let mut g = Vec::with_capacity(levels.len());
    for level in levels {
        lp.push(level.lp);
        g.push(level.g);
    }
    (lp, g)
}

/// Core one-shot scoring path — used when the reference state isn't
/// pre-cached. Builds both pyramids from scratch.
fn score_from_gray(
    ref_gray: &[f32],
    dis_gray: &[f32],
    work_w: usize,
    work_h: usize,
    params: &IwssimParams,
) -> Result<IwssimScore> {
    let ref_levels = build_laplacian_pyramid(ref_gray, work_w, work_h, NUM_SCALES);
    let dis_levels = build_laplacian_pyramid(dis_gray, work_w, work_h, NUM_SCALES);
    let (lp_ref, g_ref) = split_levels(ref_levels);
    let (lp_dis, _g_dis) = split_levels(dis_levels);
    score_with_split(&lp_ref, &lp_dis, &g_ref, work_w, work_h, params)
}

/// Shared back-end that consumes already-split pyramid bands and
/// produces the final score. Used by both the one-shot and warm-ref
/// paths.
fn score_with_split(
    lp_ref: &[Vec<f32>],
    lp_dis: &[Vec<f32>],
    g_ref: &[Vec<f32>],
    work_w: usize,
    work_h: usize,
    params: &IwssimParams,
) -> Result<IwssimScore> {
    let dims = pyramid_dims(work_w, work_h, NUM_SCALES);

    // Compute cs maps at all scales and the luminance at the coarsest.
    // We compute cs at each scale (cs at top is `cs * l`).
    let mut cs_per_scale: Vec<crate::ssim::CsStats> = Vec::with_capacity(NUM_SCALES);
    for s in 0..NUM_SCALES {
        let (w, h) = dims[s];
        let cs = compute_cs(&lp_ref[s], &lp_dis[s], h, w, s == NUM_SCALES - 1);
        cs_per_scale.push(cs);
    }

    // Compute IW weight maps for finer scales (1..Nsc-1).
    let iw_maps = if params.iw_flag {
        compute_iw_maps(lp_ref, lp_dis, g_ref, &dims, params)
    } else {
        Vec::new()
    };

    // Pool each scale.
    let bound1 = params.bound1() as usize;
    let mut wmcs: [f64; NUM_SCALES] = [0.0; NUM_SCALES];
    for s in 0..NUM_SCALES {
        let cs = &cs_per_scale[s];
        let scale_w = cs.cs_w;
        let scale_h = cs.cs_h;
        if params.iw_flag && s < NUM_SCALES - 1 {
            // IW weighted average.
            let iw = &iw_maps[s];
            // Crop iw by bound1 on every side (matches the Python
            // `iw[:, :, bound1: -bound1, bound1: -bound1]`).
            let (crop_h, crop_w, iw_crop) = crop_2d(&iw.infow, iw.h, iw.w, bound1);
            // The cropped iw shape MUST equal the cs map shape.
            // Sanity: cs_h = h - 10; iw.h = nblv = h - block_h + 1.
            // For block_h=3, nblv = h-2; iw_crop = nblv - 2*bound1 =
            // h - 2 - 8 = h - 10 = cs_h. ✓
            debug_assert_eq!(
                (crop_h, crop_w),
                (scale_h, scale_w),
                "iw crop dims must match cs dims"
            );
            let (sum_csiw, sum_iw) = weighted_sum(&cs.cs, &iw_crop, scale_h * scale_w);
            let denom = if sum_iw == 0.0 { 1.0 } else { sum_iw };
            wmcs[s] = (sum_csiw / denom) as f64;
        } else {
            // Plain mean (when iw_flag = false, or at the top scale
            // where the Python sets `iw = torch.ones(cs.shape)`).
            let n = scale_h * scale_w;
            let mut sum = 0.0_f64;
            for &v in &cs.cs {
                sum += v as f64;
            }
            wmcs[s] = sum / (n as f64);
        }
    }

    // Final score: Π |wmcs_j|^β_j.
    let mut score = 1.0_f64;
    for s in 0..NUM_SCALES {
        score *= wmcs[s].abs().powf(SCALE_WEIGHTS[s] as f64);
    }

    Ok(IwssimScore {
        score,
        per_scale: wmcs,
    })
}

/// Crop a 2D `(h, w)` slab by `bound` on every side. Returns
/// `(crop_h, crop_w, cropped_data)`.
///
/// When `2*bound >= h` or `2*bound >= w`, returns the central pixel
/// (or single column/row) — matches the Python's `[bound:-bound]`
/// behavior for tiny slabs.
fn crop_2d(src: &[f32], h: usize, w: usize, bound: usize) -> (usize, usize, Vec<f32>) {
    // The Python's `iw[:, :, bound1:-bound1, ...]` is `iw[bound:h-bound, bound:w-bound]`.
    // For h = nblv at scale 1, with block=3: nblv = h_work - 2 ≥ 174,
    // bound1 = 4. crop_h = nblv - 8 ≥ 166 ≥ cs_h = h_work - 10 ≥ 166. ✓
    assert!(bound * 2 < h, "crop bound too large for height {h}");
    assert!(bound * 2 < w, "crop bound too large for width {w}");
    let crop_h = h - 2 * bound;
    let crop_w = w - 2 * bound;
    let mut out = alloc::vec![0.0_f32; crop_h * crop_w];
    for r in 0..crop_h {
        let src_row = &src[(r + bound) * w + bound..(r + bound) * w + bound + crop_w];
        let dst_row = &mut out[r * crop_w..(r + 1) * crop_w];
        dst_row.copy_from_slice(src_row);
    }
    (crop_h, crop_w, out)
}

/// Σ cs·iw and Σ iw (both as `f64` for precision). SIMD-routed via
/// `simd_kernels::weighted_sum_pair`.
fn weighted_sum(cs: &[f32], iw: &[f32], n: usize) -> (f32, f32) {
    debug_assert_eq!(cs.len(), n);
    debug_assert_eq!(iw.len(), n);
    let (sum_csiw, sum_iw) = crate::simd_kernels::weighted_sum_pair(cs, iw);
    (sum_csiw as f32, sum_iw as f32)
}

// Compile-time sanity: NUM_SCALES and SCALE_WEIGHTS lengths must agree.
const _: () = {
    assert!(SCALE_WEIGHTS.len() == NUM_SCALES);
    // Ensure 11-tap window radius arithmetic is consistent.
    assert!(SSIM_WIN_LEN == 11);
};
