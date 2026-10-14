//! Top-level IW-SSIM scorer — composes pyramid + SSIM stats + IW maps.

use alloc::vec::Vec;

use crate::filters::{SCALE_WEIGHTS, SSIM_WIN_LEN};
use crate::params::IwssimParams;
use crate::pyramid::{PyrLevel, build_laplacian_pyramid, pyramid_dims};
use crate::ssim::compute_cs;
use crate::weights::compute_iw_maps;
use crate::{Error, IwssimScore, MIN_NATIVE_DIM, NUM_SCALES, Result, rgb_u8_to_gray};

/// Persistent per-call scratch buffers, allocated once at
/// [`Iwssim::new`] and reused across calls.
///
/// Phase 9.YA Part 1: eliminates the per-call
/// `alloc::vec![0.0_f32; w*h]` for the input-side gray planes and the
/// `pad_gray` working buffers. At 40 MP that's 4 × 160 MB = 640 MB
/// per `score()` call of pure allocation churn that's now replaced by
/// in-place writes into these scratch buffers.
struct Scratch {
    /// RGB → BT.601 grayscale destination for the REF side. Sized
    /// `width * height` f32 at `Iwssim::new`. Reused across `score` /
    /// `warm_reference` calls.
    ref_gray: Vec<f32>,
    /// Same for the DIST side.
    dis_gray: Vec<f32>,
    /// Padded REF working buffer used by [`Iwssim::pad_gray_into`].
    /// Sized `work_w * work_h` f32 at `Iwssim::new`. When no padding
    /// is required (the common case where `width >= MIN_NATIVE_DIM`
    /// etc.), `pad_gray_into` does a single `copy_from_slice` into
    /// this buffer.
    ref_work: Vec<f32>,
    /// Same for the DIST side.
    dis_work: Vec<f32>,
}

impl Scratch {
    fn new(width: usize, height: usize, work_w: usize, work_h: usize) -> Self {
        Self {
            ref_gray: alloc::vec![0.0_f32; width * height],
            dis_gray: alloc::vec![0.0_f32; width * height],
            ref_work: alloc::vec![0.0_f32; work_w * work_h],
            dis_work: alloc::vec![0.0_f32; work_w * work_h],
        }
    }
}

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
    /// Per-call scratch buffers — see [`Scratch`].
    scratch: Scratch,
    /// Cached state from `warm_reference`, if any.
    pub(crate) warm: Option<WarmState>,
    /// When `Some`, the next full-image score also collects its per-scale
    /// contrast-structure maps here (the generic surface's map requests).
    map_sink: Option<Vec<CsMap>>,
}

/// State cached by `warm_reference`. Holds the per-scale Laplacian
/// bands + per-scale Gaussian bands (for parent-band lookup) so the
/// scoring path only has to build the distorted side.
///
/// Phase 9.Z.A also caches the per-scale eigendecomposition results
/// (lambdas + Cu_inv) so `score_with_warm_ref_strip` can use the
/// global IW covariance without recomputing it across strips.
pub(crate) struct WarmState {
    /// Reference Laplacian bands (one per scale, finest first).
    pub(crate) lp_ref: Vec<Vec<f32>>,
    /// Per-scale eigendecomposition (one per IW scale; index s holds
    /// the result for scale s ∈ 0..NUM_SCALES-1). `None` indicates
    /// either an empty scale or a build before the strip path was
    /// exercised — in the latter case the strip path lazily fills it
    /// from `lp_ref` / `g_ref`.
    pub(crate) eigs: Vec<Option<crate::eig::EigResult>>,
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
        let scratch = Scratch::new(width as usize, height as usize, work_w, work_h);
        Ok(Self {
            width,
            height,
            params,
            work_w,
            work_h,
            scratch,
            warm: None,
            map_sink: None,
        })
    }

    /// Configured `(width, height)`.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// `true` if a warm reference is cached.
    pub fn has_reference(&self) -> bool {
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
        // Phase 9.YA Part 1: reuse scratch.ref_gray / scratch.dis_gray
        // instead of allocating 2 × W*H*4 fresh per call. Then
        // pad_gray_into routes into scratch.ref_work / scratch.dis_work,
        // also reused.  At 40 MP these two changes save 4 × 160 MB =
        // 640 MB of per-call allocator churn.
        rgb_u8_to_gray(self.params.luma, ref_rgb, &mut self.scratch.ref_gray);
        rgb_u8_to_gray(self.params.luma, dis_rgb, &mut self.scratch.dis_gray);
        self.score_gray_internal()
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
        // Caller-supplied gray planes — copy into the persistent
        // scratch slots so downstream paths can rely on `self.scratch`
        // exclusively.
        self.scratch.ref_gray.copy_from_slice(ref_gray);
        self.scratch.dis_gray.copy_from_slice(dis_gray);
        self.score_gray_internal()
    }

    /// Inner score-from-gray path. Assumes `self.scratch.ref_gray` and
    /// `self.scratch.dis_gray` have been populated by the caller.
    /// Routes through `pad_gray_into` for the work-size buffers.
    fn score_gray_internal(&mut self) -> Result<IwssimScore> {
        // pad_gray_into fills scratch.ref_work / scratch.dis_work in
        // place, reusing capacity across calls.
        Self::pad_gray_into(
            &self.scratch.ref_gray,
            &mut self.scratch.ref_work,
            self.width as usize,
            self.height as usize,
            self.work_w,
            self.work_h,
        );
        Self::pad_gray_into(
            &self.scratch.dis_gray,
            &mut self.scratch.dis_work,
            self.width as usize,
            self.height as usize,
            self.work_w,
            self.work_h,
        );
        score_from_gray(
            &self.scratch.ref_work,
            &self.scratch.dis_work,
            self.work_w,
            self.work_h,
            &self.params,
            self.map_sink.as_mut(),
        )
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
        // Phase 9.YA Part 1: write into scratch.ref_gray instead of
        // allocating a fresh `vec![0.0; w*h]` (saves 160 MB at 40 MP).
        rgb_u8_to_gray(self.params.luma, ref_rgb, &mut self.scratch.ref_gray);
        self.warm_reference_gray_internal()
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
        // Copy into scratch.ref_gray so the internal path can rely on
        // scratch buffers exclusively.
        self.scratch.ref_gray.copy_from_slice(ref_gray);
        self.warm_reference_gray_internal()
    }

    /// Inner warm path — assumes `self.scratch.ref_gray` has been
    /// populated.
    fn warm_reference_gray_internal(&mut self) -> Result<()> {
        Self::pad_gray_into(
            &self.scratch.ref_gray,
            &mut self.scratch.ref_work,
            self.width as usize,
            self.height as usize,
            self.work_w,
            self.work_h,
        );
        let levels =
            build_laplacian_pyramid(&self.scratch.ref_work, self.work_w, self.work_h, NUM_SCALES);
        let lp_ref = split_levels(levels);
        let eigs = (0..NUM_SCALES - 1).map(|_| None).collect();
        self.warm = Some(WarmState { lp_ref, eigs });
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
        // Phase 9.YA Part 1: write into scratch.dis_gray instead of
        // allocating a fresh `vec![0.0; w*h]` (saves 160 MB at 40 MP).
        rgb_u8_to_gray(self.params.luma, dis_rgb, &mut self.scratch.dis_gray);
        self.score_with_warm_ref_gray_internal()
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
        self.scratch.dis_gray.copy_from_slice(dis_gray);
        self.score_with_warm_ref_gray_internal()
    }

    /// Strip-mode score: walks image in horizontal slabs of `strip_height`
    /// rows + halo. Peak working-set is one strip's working set, not
    /// the full image. Score is bit-identical (within atomic float-add
    /// reduction order tolerance ≤ 1e-5 relative) to [`Self::score`]
    /// when `strip_height >= STRIP_BODY_MIN`.
    ///
    /// Phase 9.Z.A: enables 40 MP+ images to score on hosts where Full
    /// mode's `~5.9 GB` peak heap would crowd the budget. Strip target:
    /// ≤ 3 GB peak at 40 MP.
    ///
    /// `strip_height` is the strip body height in scale-0 rows. Halo
    /// rows beyond body extend `STRIP_HALO_ROWS = 320` per side, clamped
    /// at image edges. Pass `STRIP_BODY_DEFAULT` (512) for the
    /// production-recommended default.
    ///
    /// # Memory profile
    ///
    /// Per-strip peak: `(body + 2·halo) × work_w × 4 × ~3` (lp_ref +
    /// lp_dis staged across 5 pyramid scales). At 40 MP
    /// (6500×6500) with `strip_height=512` and 320-row halo: roughly
    /// `(512+640) × 6500 × 4 × 5 ≈ 150 MB` per strip vs 5.9 GB Full.
    ///
    /// # Wall-time
    ///
    /// Strip mode runs two passes through the image (eigendecomposition
    /// is global) — wall-time penalty ~1.4× vs Full. Acceptable for
    /// memory-bounded production sweeps.
    pub fn score_strip(
        &mut self,
        ref_rgb: &[u8],
        dis_rgb: &[u8],
        strip_height: u32,
    ) -> Result<IwssimScore> {
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
        rgb_u8_to_gray(self.params.luma, ref_rgb, &mut self.scratch.ref_gray);
        rgb_u8_to_gray(self.params.luma, dis_rgb, &mut self.scratch.dis_gray);
        self.score_strip_gray_internal(strip_height)
    }

    /// Strip-mode score from gray-f32 inputs. See [`Self::score_strip`].
    pub fn score_strip_gray(
        &mut self,
        ref_gray: &[f32],
        dis_gray: &[f32],
        strip_height: u32,
    ) -> Result<IwssimScore> {
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
        self.scratch.ref_gray.copy_from_slice(ref_gray);
        self.scratch.dis_gray.copy_from_slice(dis_gray);
        self.score_strip_gray_internal(strip_height)
    }

    fn score_strip_gray_internal(&mut self, strip_height: u32) -> Result<IwssimScore> {
        // Pad ref/dis into work buffers.
        Self::pad_gray_into(
            &self.scratch.ref_gray,
            &mut self.scratch.ref_work,
            self.width as usize,
            self.height as usize,
            self.work_w,
            self.work_h,
        );
        Self::pad_gray_into(
            &self.scratch.dis_gray,
            &mut self.scratch.dis_work,
            self.width as usize,
            self.height as usize,
            self.work_w,
            self.work_h,
        );
        crate::strip::score_strip_internal(
            &self.scratch.ref_work,
            &self.scratch.dis_work,
            self.work_w,
            self.work_h,
            strip_height as usize,
            &self.params,
        )
    }

    /// Strip-mode score against the warm reference. The cached ref
    /// state (full-image Laplacian pyramid + per-scale `Cu` eigendecomp)
    /// lives in `WarmState`; the dist side is walked in strips of
    /// `strip_height` rows + `STRIP_HALO_ROWS` halo.
    ///
    /// Phase 9.Z.A: best-of-both-worlds memory profile for "many
    /// dist per ref" batch sweeps:
    /// - Ref state stays full-image cached (~150 MB at 4096²).
    /// - Per-strip dist working set bounded at one strip's pyramid.
    /// - No two-pass walk — the eigendecomp is in `WarmState` from
    ///   `warm_reference`.
    ///
    /// Peak memory at 40 MP: roughly RAM(`lp_ref + g_ref` full image)
    /// + RAM(one strip's dist pyramid) ≈ `image_h × work_w × 4 × 5 ×
    /// 2 (lp + g) + (body + 2*halo) × work_w × 4 × 5 ≈ 5 × 130 + 150
    /// ≈ 800 MB` vs 5.58 GB warm-ref Full.
    pub fn score_with_warm_ref_strip(
        &mut self,
        dis_rgb: &[u8],
        strip_height: u32,
    ) -> Result<IwssimScore> {
        let expected = (self.width as usize) * (self.height as usize) * 3;
        if dis_rgb.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: dis_rgb.len(),
            });
        }
        rgb_u8_to_gray(self.params.luma, dis_rgb, &mut self.scratch.dis_gray);
        self.score_with_warm_ref_strip_gray_internal(strip_height)
    }

    /// Strip-mode warm-ref score from gray-f32 dist. See
    /// [`Self::score_with_warm_ref_strip`].
    pub fn score_with_warm_ref_strip_gray(
        &mut self,
        dis_gray: &[f32],
        strip_height: u32,
    ) -> Result<IwssimScore> {
        let expected = (self.width as usize) * (self.height as usize);
        if dis_gray.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: dis_gray.len(),
            });
        }
        self.scratch.dis_gray.copy_from_slice(dis_gray);
        self.score_with_warm_ref_strip_gray_internal(strip_height)
    }

    fn score_with_warm_ref_strip_gray_internal(
        &mut self,
        strip_height: u32,
    ) -> Result<IwssimScore> {
        // Pad dist into work buffer.
        Self::pad_gray_into(
            &self.scratch.dis_gray,
            &mut self.scratch.dis_work,
            self.width as usize,
            self.height as usize,
            self.work_w,
            self.work_h,
        );
        let warm = self.warm.as_mut().ok_or(Error::NoWarmReference)?;
        crate::strip::score_with_warm_ref_strip_internal(
            warm,
            &self.scratch.dis_work,
            self.work_w,
            self.work_h,
            strip_height as usize,
            &self.params,
        )
    }

    /// Inner warm-score path — assumes `self.scratch.dis_gray` has
    /// been populated.
    fn score_with_warm_ref_gray_internal(&mut self) -> Result<IwssimScore> {
        Self::pad_gray_into(
            &self.scratch.dis_gray,
            &mut self.scratch.dis_work,
            self.width as usize,
            self.height as usize,
            self.work_w,
            self.work_h,
        );
        let warm = self.warm.as_ref().ok_or(Error::NoWarmReference)?;
        let dis_levels =
            build_laplacian_pyramid(&self.scratch.dis_work, self.work_w, self.work_h, NUM_SCALES);
        let lp_dis = split_levels(dis_levels);
        let r = score_with_split(
            &warm.lp_ref,
            &lp_dis,
            self.work_w,
            self.work_h,
            &self.params,
            self.map_sink.as_mut(),
        );
        for v in lp_dis {
            crate::pool::buf_give(v);
        }
        r
    }

    /// Tile / replicate `src` to `(work_w, work_h)` when `allow_small`
    /// is enabled and the source is smaller; otherwise the source is
    /// copied 1:1 into `dst`.
    ///
    /// Associated function (no `&self`) so it can be called via
    /// `Self::pad_gray_into` while holding split borrows on
    /// `self.scratch.ref_gray` and `self.scratch.ref_work` at the same
    /// time.
    ///
    /// Phase 9.YA Part 1: replaces the prior `pad_gray(&self, src) ->
    /// Vec<f32>` which allocated 160 MB per call at 40 MP. The
    /// destination buffer is now pre-allocated in `Scratch::new` and
    /// reused across calls.
    fn pad_gray_into(
        src: &[f32],
        dst: &mut [f32],
        sw: usize,
        sh: usize,
        work_w: usize,
        work_h: usize,
    ) {
        debug_assert_eq!(src.len(), sw * sh);
        debug_assert_eq!(dst.len(), work_w * work_h);
        if sw == work_w && sh == work_h {
            dst.copy_from_slice(src);
            return;
        }
        for y in 0..work_h {
            let sy = y % sh;
            for x in 0..work_w {
                let sx = x % sw;
                dst[y * work_w + x] = src[sy * sw + sx];
            }
        }
    }
}

fn split_levels(levels: Vec<PyrLevel>) -> Vec<Vec<f32>> {
    levels.into_iter().map(|level| level.lp).collect()
}

/// Core one-shot scoring path — used when the reference state isn't
/// pre-cached. Builds both pyramids from scratch.
fn score_from_gray(
    ref_gray: &[f32],
    dis_gray: &[f32],
    work_w: usize,
    work_h: usize,
    params: &IwssimParams,
    maps: Option<&mut Vec<CsMap>>,
) -> Result<IwssimScore> {
    let ref_levels = build_laplacian_pyramid(ref_gray, work_w, work_h, NUM_SCALES);
    let dis_levels = build_laplacian_pyramid(dis_gray, work_w, work_h, NUM_SCALES);
    let lp_ref = split_levels(ref_levels);
    let lp_dis = split_levels(dis_levels);
    let r = score_with_split(&lp_ref, &lp_dis, work_w, work_h, params, maps);
    for v in lp_ref.into_iter().chain(lp_dis) {
        crate::pool::buf_give(v);
    }
    r
}

/// Shared back-end that consumes already-split pyramid bands and
/// produces the final score. Used by both the one-shot and warm-ref
/// paths.
/// One scale's contrast-structure map (`cs·l` at the coarsest scale), the
/// valid-mode `(w_s − 10) × (h_s − 10)` plane the scale's pooling reads.
pub(crate) struct CsMap {
    pub(crate) w: usize,
    pub(crate) h: usize,
    pub(crate) cs: Vec<f32>,
}

fn score_with_split(
    lp_ref: &[Vec<f32>],
    lp_dis: &[Vec<f32>],
    work_w: usize,
    work_h: usize,
    params: &IwssimParams,
    mut maps: Option<&mut Vec<CsMap>>,
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
        compute_iw_maps(lp_ref, lp_dis, &dims, params)
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
            // Read iw through the bound1 crop window strided (matches
            // the Python `iw[:, :, bound1: -bound1, bound1: -bound1]`)
            // — no materialized crop.
            let crop_w = iw.w - 2 * bound1;
            // The cropped iw shape MUST equal the cs map shape.
            // Sanity: cs_h = h - 10; iw.h = nblv = h - block_h + 1.
            // For block_h=3, nblv = h-2; iw_crop = nblv - 2*bound1 =
            // h - 2 - 8 = h - 10 = cs_h. ✓
            debug_assert_eq!(
                (iw.h - 2 * bound1, crop_w),
                (scale_h, scale_w),
                "iw crop dims must match cs dims"
            );
            let (sum_csiw64, sum_iw64) = crate::simd_kernels::weighted_sum_pair_iw_strided(
                &cs.cs, &iw.infow, iw.w, bound1, bound1, crop_w,
            );
            let (sum_csiw, sum_iw) = (sum_csiw64 as f32, sum_iw64 as f32);
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

    for s in cs_per_scale {
        if let Some(sink) = maps.as_deref_mut() {
            sink.push(CsMap {
                w: s.cs_w,
                h: s.cs_h,
                cs: s.cs.clone(),
            });
        }
        s.release();
    }
    for m in iw_maps {
        m.release();
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

// Compile-time sanity: NUM_SCALES and SCALE_WEIGHTS lengths must agree.
const _: () = {
    assert!(SCALE_WEIGHTS.len() == NUM_SCALES);
    // Ensure 11-tap window radius arithmetic is consistent.
    assert!(SSIM_WIN_LEN == 11);
};

// ---------------------------------------------------------------------------
// Generic Metric surface.
// ---------------------------------------------------------------------------

use zenmetrics_abstractions as za;

impl Error {
    fn kind(&self) -> za::ErrorKind {
        match self {
            Error::DimensionMismatch { .. } => za::ErrorKind::DimensionMismatch,
            Error::NoWarmReference => za::ErrorKind::Internal,
            Error::InvalidImageSize { .. } => za::ErrorKind::TooSmall,
        }
    }
}

/// Output revision of [`Iwssim`] on the generic surface.
const SCORE_REVISION: u32 = 1;

const OUTPUTS: &[za::ScoreSpec] = &[
    za::ScoreSpec::new("iwssim", za::ScoreKind::Similarity),
    za::ScoreSpec::new("wmcs_0", za::ScoreKind::Similarity),
    za::ScoreSpec::new("wmcs_1", za::ScoreKind::Similarity),
    za::ScoreSpec::new("wmcs_2", za::ScoreKind::Similarity),
    za::ScoreSpec::new("wmcs_3", za::ScoreKind::Similarity),
    za::ScoreSpec::new("wmcs_4", za::ScoreKind::Similarity),
];
const _: () = assert!(OUTPUTS.len() == NUM_SCALES + 1);

fn za_scalars(s: &IwssimScore) -> za::ScalarEval {
    let mut out = za::ScalarEval::single("iwssim", s.score);
    for (spec, &v) in OUTPUTS[1..].iter().zip(&s.per_scale) {
        out.push(spec.key.clone(), v);
    }
    out
}

fn za_evaluation(s: IwssimScore) -> alloc::boxed::Box<dyn za::Evaluation> {
    alloc::boxed::Box::new(za_scalars(&s))
}

/// Map names, finest scale first; the coarsest map includes luminance.
const CS_NAMES: [&str; NUM_SCALES] = ["cs_0", "cs_1", "cs_2", "cs_3", "csl_4"];

/// A scored pair with its per-scale contrast-structure maps.
struct IwssimMaps {
    score: IwssimScore,
    maps: Vec<CsMap>,
    geometry: Vec<za::MapGeometry>,
}

impl IwssimMaps {
    /// Scale `s` samples the input at `2^s·k` (the binom5 `corr_dn` keeps
    /// even positions), and cell `i` of its valid 11×11 window map is
    /// centred on scale sample `i + 5`; cells are `2^s`-wide blocks around
    /// that centre. The value reads the scale's Laplacian band within 5
    /// samples, and a band sample reads the input within `6·2^s − 2`
    /// pixels (`2^{s+1} − 2` for the residual lowpass at the top scale):
    /// halo `11·2^s − 2`, `7·2^s − 2` at the top. When `allow_small`
    /// tiled a small input up to the working size, values also read the
    /// wrapped copies, so support is global.
    fn new(score: IwssimScore, maps: Vec<CsMap>, input: (u32, u32), tiled: bool) -> Self {
        let geometry = maps
            .iter()
            .enumerate()
            .map(|(s, m)| {
                let cell = 1u32 << s;
                let origin = 5 * i64::from(cell) - i64::from(cell / 2);
                let g = za::MapGeometry::grid(
                    input.0, input.1, origin, origin, cell, cell, m.w as u32, m.h as u32,
                );
                if tiled {
                    g.with_support(za::Support::Global)
                } else {
                    let halo = if s + 1 == NUM_SCALES {
                        7 * cell - 2
                    } else {
                        11 * cell - 2
                    };
                    g.with_halo(halo, halo)
                }
            })
            .collect();
        Self {
            score,
            maps,
            geometry,
        }
    }

    fn view(&self, s: usize) -> Option<za::MapView<'_>> {
        let g = self.geometry[s];
        let px = za::gray_f32_slice(&self.maps[s].cs, g.cells_x, g.cells_y).ok()?;
        za::MapView::new(CS_NAMES[s], px, za::MapPolarity::Similarity, g).ok()
    }
}

impl za::Evaluation for IwssimMaps {
    fn score(&self) -> f64 {
        self.score.score
    }

    fn scores(&self) -> za::Scores<'_> {
        alloc::boxed::Box::new(za_scalars(&self.score).named.into_iter())
    }

    fn map(&self) -> Option<za::MapView<'_>> {
        self.view(0)
    }

    fn maps(&self) -> Vec<za::MapView<'_>> {
        (0..self.maps.len()).filter_map(|s| self.view(s)).collect()
    }

    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

impl Iwssim {
    /// Run `score` with the map sink armed; the sink is always disarmed
    /// again, whatever `score` returns.
    fn za_with_maps(
        &mut self,
        req: za::MapRequest,
        score: impl FnOnce(&mut Self) -> Result<IwssimScore>,
    ) -> za::MetricResult<alloc::boxed::Box<dyn za::Evaluation>> {
        use za::ResultMetricExt as _;
        self.map_sink = Some(Vec::new());
        let scored = score(self);
        let maps = self.map_sink.take().unwrap_or_default();
        let s = scored.metric_err_with("iwssim", Error::kind)?;
        let tiled = (self.work_w, self.work_h) != (self.width as usize, self.height as usize);
        let eval = IwssimMaps::new(s, maps, (self.width, self.height), tiled);
        za::apply_map_request(alloc::boxed::Box::new(eval), req)
    }

    /// Re-plan for a `w × h` input when it differs from the configured
    /// size (keeping the params), so the generic surface takes any size.
    fn za_fit(&mut self, w: usize, h: usize) -> za::MetricResult<()> {
        use za::ResultMetricExt as _;
        if (self.width as usize, self.height as usize) != (w, h) {
            *self = Iwssim::with_params(w as u32, h as u32, self.params)
                .metric_err_with("iwssim", Error::kind)?;
        }
        Ok(())
    }
}

/// `Iwssim` implements the generic surface on packed `RGB8_SRGB`, re-planning
/// when the input size differs from the constructed one. `per_scale` values
/// ride `scores()` as `wmcs_0..4`. [`za::Metric::warm_reference`] reuses the
/// reference pyramid ([`Iwssim::warm_reference`]); warm candidates serve maps
/// too, bit-identical to cold ones.
///
/// Maps, on request: the per-scale contrast-structure maps the scales pool,
/// `cs_0` (finest, primary) … `cs_3` and `csl_4` (coarsest, `cs·l`), each the
/// valid 11×11-window `(w_s − 10) × (h_s − 10)` plane at scale `s`
/// ([`za::MapPolarity::Similarity`], `1` = identical). Scales 0–3 pool their
/// map with the information-content weights, the top scale by its plain
/// mean, and the score is `Π wmcs_s^β_s`: these are per-scale components,
/// not a per-pixel IW-SSIM.
impl za::Metric for Iwssim {
    fn id(&self) -> za::MetricId {
        let p = &self.params;
        za::MetricId::new("iwssim", za::implementation!(SCORE_REVISION))
            .with_reference("Python-IW-SSIM f9de37cd (Wang & Li 2011)")
            .with_config("iw_flag", p.iw_flag)
            .with_config("bl_sz", alloc::format!("{}x{}", p.bl_sz_x, p.bl_sz_y))
            .with_config("parent", p.parent)
            .with_config("sigma_nsq", p.sigma_nsq)
            .with_config("allow_small", p.allow_small)
            .with_config("luma", alloc::format!("{:?}", p.luma))
    }

    fn outputs(&self) -> alloc::borrow::Cow<'static, [za::ScoreSpec]> {
        alloc::borrow::Cow::Borrowed(OUTPUTS)
    }

    fn accepts(&self, format: &za::InputFormat) -> bool {
        za::is_rgb8_srgb(format)
    }

    fn evaluate(
        &mut self,
        r: za::Image<'_>,
        d: za::Image<'_>,
        opts: za::EvalOpts<'_>,
    ) -> za::MetricResult<alloc::boxed::Box<dyn za::Evaluation>> {
        use za::ResultMetricExt as _;
        opts.checkpoint()?;
        let (r, d) = za::rgb8_pair(&r, &d)?;
        self.za_fit(r.width, r.height)?;
        let (rp, dp) = (za::contiguous_rgb8(&r), za::contiguous_rgb8(&d));
        if let Some(req) = opts.map {
            return self.za_with_maps(req, |m| m.score(&rp, &dp));
        }
        let s = self
            .score(&rp, &dp)
            .metric_err_with("iwssim", Error::kind)?;
        Ok(za_evaluation(s))
    }

    fn warm_reference<'m>(
        &'m mut self,
        r: za::Image<'_>,
        opts: za::EvalOpts<'_>,
    ) -> za::MetricResult<alloc::boxed::Box<dyn za::WarmReference + 'm>> {
        use za::ResultMetricExt as _;
        opts.checkpoint()?;
        let r = za::rgb8_view(&r)?;
        self.za_fit(r.width, r.height)?;
        self.warm_reference(&za::contiguous_rgb8(&r))
            .metric_err_with("iwssim", Error::kind)?;
        Ok(alloc::boxed::Box::new(IwssimWarm { metric: self }))
    }
}

/// A warmed [`Iwssim`]: candidates reuse the reference pyramid.
struct IwssimWarm<'m> {
    metric: &'m mut Iwssim,
}

impl za::WarmReference for IwssimWarm<'_> {
    fn evaluate(
        &mut self,
        d: za::Image<'_>,
        opts: za::EvalOpts<'_>,
    ) -> za::MetricResult<alloc::boxed::Box<dyn za::Evaluation>> {
        use za::ResultMetricExt as _;
        opts.checkpoint()?;
        let d = za::rgb8_view(&d)?;
        if (d.width, d.height) != (self.metric.width as usize, self.metric.height as usize) {
            return za::fail(
                za::MetricError::new(za::ErrorKind::DimensionMismatch)
                    .with_metric("iwssim")
                    .with_message(alloc::format!(
                        "warm reference {}x{}, candidate {}x{}",
                        self.metric.width,
                        self.metric.height,
                        d.width,
                        d.height
                    )),
            );
        }
        let dp = za::contiguous_rgb8(&d);
        if let Some(req) = opts.map {
            return self
                .metric
                .za_with_maps(req, |m| m.score_with_warm_ref(&dp));
        }
        let s = self
            .metric
            .score_with_warm_ref(&dp)
            .metric_err_with("iwssim", Error::kind)?;
        Ok(za_evaluation(s))
    }

    fn reuses_reference_work(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod za_tests {
    use super::*;

    #[test]
    fn generic_surface_conforms_and_warm_reuses() {
        // allow_small lets the conformance pair stay cheap; the warm path
        // must still be bit-identical to cold scoring.
        let params = IwssimParams {
            allow_small: true,
            ..IwssimParams::default()
        };
        let mut m = Iwssim::with_params(8, 8, params).unwrap();
        za::conformance::assert_conforms_rgb8(&mut m, 64, 48);
        let a = za::conformance::textured_rgb8(64, 48, 9);
        let warm = za::Metric::warm_reference(
            &mut m,
            za::conformance::rgb8_image(&a, 64, 48),
            za::EvalOpts::SCORE_ONLY,
        )
        .unwrap();
        assert!(warm.reuses_reference_work());
    }
}
