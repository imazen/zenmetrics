//! IW-SSIM pipeline orchestration.
//!
//! Wires the kernels together into the full IW-SSIM algorithm:
//!
//! ```text
//!   gray u8/f32 ──► LP pyramid (5 levels, pyrtools binom5 + reflect1)
//!     ├─► [per scale]  11×11 Gaussian (valid) → µ, σ stats → cs / l
//!     └─► [scales 0..3]  3×3 box stats → g, vv
//!                        imenlarge2(LP[s+1])  → parent band
//!                        cov accumulate (atomic) → C_u
//!                        CPU eigendecomp + invert → C_u_inv, λ_k
//!                        per-pixel quadratic form + infow → iw_map
//!   Σ(cs·iw)/Σ(iw) and mean(cs·l)  →  wmcs[s]
//!   score = Π |wmcs[s]|^β[s]
//! ```
//!
//! Per `Iwssim::new` we pre-allocate every device buffer the pipeline
//! needs at the configured `(width, height)`. Subsequent
//! `compute_gray` / `compute_rgb` calls upload the inputs and re-use
//! the pre-allocated buffers — no per-call allocation on the GPU.
//!
//! Buffer naming follows the IW-SSIM reference:
//! - `lp_ref[s]` / `lp_dis[s]` — Laplacian band at scale `s`.
//! - `cs[s]` — contrast-structure SSIM map at scale `s` (shape
//!   `(h_s − 10, w_s − 10)`).
//! - `iw[s]` — info-content weight map at scale `s` (shape
//!   `(h_s − 2, w_s − 2)`).

use cubecl::prelude::*;

use crate::eig;
use crate::filters;
use crate::kernels::{
    box3, cov, gauss11, imenlarge2, infow, lap_pyramid, reduction, rgb2gray, ssim_combine, util,
};
use crate::{Error, GpuIwssimResult, NUM_SCALES, Result};

/// MS-SSIM Gaussian window radius — used to compute `bound1` (the
/// crop applied to `iw_j` so it aligns with `cs_j`).
const BOUND: u32 = 5;
/// blSzX = 3 in the reference → floor((3 − 1) / 2) = 1.
const BLK_HALF: u32 = 1;
/// Cropped offset applied to `iw_j` before pooling against `cs_j`.
const BOUND1: u32 = BOUND - BLK_HALF; // = 4

/// Per-scale device buffer set.
struct Scale {
    /// LP shape.
    h: u32,
    w: u32,
    /// `(h − 10, w − 10)` — SSIM cs/l shape at this scale.
    cs_h: u32,
    cs_w: u32,
    /// `(h − 2, w − 2)` — IW shape at this scale.
    iw_h: u32,
    iw_w: u32,

    // LP coefficients at this scale (both sides).
    lp_ref: cubecl::server::Handle,
    lp_dis: cubecl::server::Handle,

    // Gaussian pyramid (working buffers used during LP build).
    g_ref: cubecl::server::Handle,
    g_dis: cubecl::server::Handle,

    // SSIM 11×11 valid-mode intermediates.
    // gh_* = post horizontal pass (h × (w − 10)); g_* = post vertical
    // pass ((h − 10) × (w − 10)).
    gh_ref: cubecl::server::Handle,
    gh_dis: cubecl::server::Handle,
    gh_ref2: cubecl::server::Handle,
    gh_dis2: cubecl::server::Handle,
    gh_refdis: cubecl::server::Handle,
    mu1: cubecl::server::Handle,
    mu2: cubecl::server::Handle,
    m11: cubecl::server::Handle,
    m22: cubecl::server::Handle,
    m12: cubecl::server::Handle,

    /// cs_map (j < 4) or `cs · l` (j = 4).
    cs: cubecl::server::Handle,

    // IW path (allocated for j < 4; allocated at full LP shape for
    // simplicity — j=4 buffers are unused).
    g_buf: cubecl::server::Handle,
    vv_buf: cubecl::server::Handle,
    parent_band: cubecl::server::Handle,
    iw: cubecl::server::Handle,

    // C_u atomic accumulator (always 10*10 = 100 f32 — wastes 19
    // entries when N=9 but cheap and avoids two-variant allocation).
    cu_atomic: cubecl::server::Handle,
    cu_inv_dev: cubecl::server::Handle,
    lambda_dev: cubecl::server::Handle,
}

fn alloc<R: Runtime>(client: &ComputeClient<R>, n: usize) -> cubecl::server::Handle {
    client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]))
}

impl Scale {
    fn new<R: Runtime>(client: &ComputeClient<R>, h: u32, w: u32) -> Self {
        let n_lp = (h as usize) * (w as usize);
        let cs_h = h - 10;
        let cs_w = w - 10;
        let n_cs = (cs_h as usize) * (cs_w as usize);
        let iw_h = h - 2;
        let iw_w = w - 2;
        let n_iw = (iw_h as usize) * (iw_w as usize);
        // Horizontal-pass intermediates have width (w − 10) and full
        // height h. Vertical-pass output has shape ((h − 10), (w − 10)).
        let n_h = (h as usize) * ((w - 10) as usize);
        Self {
            h,
            w,
            cs_h,
            cs_w,
            iw_h,
            iw_w,
            lp_ref: alloc(client, n_lp),
            lp_dis: alloc(client, n_lp),
            g_ref: alloc(client, n_lp),
            g_dis: alloc(client, n_lp),
            gh_ref: alloc(client, n_h),
            gh_dis: alloc(client, n_h),
            gh_ref2: alloc(client, n_h),
            gh_dis2: alloc(client, n_h),
            gh_refdis: alloc(client, n_h),
            mu1: alloc(client, n_cs),
            mu2: alloc(client, n_cs),
            m11: alloc(client, n_cs),
            m22: alloc(client, n_cs),
            m12: alloc(client, n_cs),
            cs: alloc(client, n_cs),
            g_buf: alloc(client, n_lp),
            vv_buf: alloc(client, n_lp),
            parent_band: alloc(client, n_lp),
            iw: alloc(client, n_iw),
            cu_atomic: alloc(client, 100),
            cu_inv_dev: alloc(client, 100),
            lambda_dev: alloc(client, 10),
        }
    }
}

/// Per-instance allocations + per-call orchestration. Construct once
/// for a given `(width, height)`, reuse across many image pairs.
pub struct Iwssim<R: Runtime> {
    client: ComputeClient<R>,
    width: u32,
    height: u32,

    /// sRGB u8 staging — re-uploaded per RGB call. None when the
    /// instance has only ever consumed grayscale planes.
    src_u32_a: cubecl::server::Handle,
    src_u32_b: cubecl::server::Handle,
    // T_x.O (2026-05-17): `pack_scratch: Vec<u32>` removed. The
    // upload path now packs u8×3 → u32 directly into the pinned
    // staging buffer reserved per call (`client.reserve_staging`),
    // collapsing two host-side passes (pack to pageable + memcpy to
    // pinned) into one. Mirrors butter T_x.O (10a5b996).

    /// One Scale per pyramid level. `scales.len() == NUM_SCALES` for
    /// validly-sized inputs.
    scales: Vec<Scale>,

    /// 11 reduction slots — 4 × (Σ(cs·iw), Σ(iw)) for j ∈ 0..3, plus
    /// 1 × Σ(cs·l) for j = 4 plus 2 unused. Plain `f32` partials with
    /// the portable two-stage reduction pattern.
    partials: cubecl::server::Handle,
    sums: cubecl::server::Handle,

    /// `set_reference` populates `scales[s].lp_ref` for every scale
    /// and flips this flag. Subsequent `compute_with_reference` calls
    /// skip the ref-side LP pyramid build.
    has_cached_reference: bool,
}

/// Slot layout in the partials / sums buffer. Indices match the order
/// in which the host reads sums back.
const SLOT_CSIW_BASE: u32 = 0; // 4 slots: j ∈ 0..3
const SLOT_IW_BASE: u32 = 4; // 4 slots: j ∈ 0..3
const SLOT_CSL: u32 = 8; // 1 slot: j = 4
const NUM_SLOTS: u32 = 9;

impl<R: Runtime> Iwssim<R> {
    /// Allocate the pipeline for the given image dimensions. Returns
    /// `Err(InvalidImageSize)` if either dimension is too small for a
    /// 5-level pyramid with 11×11 valid-mode SSIM stats at the
    /// coarsest scale.
    pub fn new(client: ComputeClient<R>, width: u32, height: u32) -> Result<Self> {
        // Coarsest scale needs at least 11 pixels per axis for a
        // valid-mode 11×11 conv. With 5 pyramid levels that's
        // 11 · 2^(NUM_SCALES − 1) = 11 · 16 = 176 at the input.
        if width < 176 || height < 176 {
            return Err(Error::InvalidImageSize);
        }
        let mut dims = Vec::with_capacity(NUM_SCALES);
        let mut h = height;
        let mut w = width;
        for _ in 0..NUM_SCALES {
            dims.push((h, w));
            h = h.div_ceil(2);
            w = w.div_ceil(2);
        }
        let scales: Vec<Scale> = dims
            .iter()
            .map(|&(h, w)| Scale::new(&client, h, w))
            .collect();

        // T4.L (2026-05-16): pack 3 sRGB bytes per pixel into ONE u32
        // (R | G<<8 | B<<16). Length = n_pixels, not n_pixels × 3.
        // Cuts per-call host→device upload from 12 B/pixel to 4 B/pixel.
        let n_pixels_usize = (width * height) as usize;
        let src_u32_a =
            client.create_from_slice(u32::as_bytes(&vec![0_u32; n_pixels_usize]));
        let src_u32_b =
            client.create_from_slice(u32::as_bytes(&vec![0_u32; n_pixels_usize]));

        let partials_len = (NUM_SLOTS * reduction::NUM_BLOCKS * reduction::BLOCK_SIZE) as usize;
        let partials = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; partials_len]));
        let sums = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; NUM_SLOTS as usize]));

        Ok(Self {
            client,
            width,
            height,
            src_u32_a,
            src_u32_b,
            scales,
            partials,
            sums,
            has_cached_reference: false,
        })
    }

    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
    pub fn n_scales(&self) -> usize {
        self.scales.len()
    }
    pub fn has_cached_reference(&self) -> bool {
        self.has_cached_reference
    }
    /// Drop any cached reference state. `compute_with_reference` will
    /// fail with `NoCachedReference` until a fresh `set_reference` is
    /// run.
    pub fn clear_reference(&mut self) {
        self.has_cached_reference = false;
    }

    /// Upload `ref_gray` and pre-compute the reference-side Laplacian
    /// pyramid. Subsequent `compute_with_reference` calls reuse the
    /// cached `lp_ref[s]` at every scale, skipping the ref-side
    /// downsample + upConv work.
    ///
    /// Saves roughly half the LP-pyramid build time per call (and at
    /// 4096² the much larger reference upload), with no parity impact:
    /// the rest of the pipeline reads `lp_ref` exactly as before.
    pub fn set_reference(&mut self, ref_gray: &[f32]) -> Result<()> {
        let expected = (self.width * self.height) as usize;
        if ref_gray.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: ref_gray.len(),
            });
        }
        let h_ref = self.client.create_from_slice(f32::as_bytes(ref_gray));
        self.scales[0].g_ref = h_ref;
        // Build only the ref-side pyramid; the dis-side will be built
        // in `compute_with_reference`.
        self.build_laplacian_pyramid(true);
        self.has_cached_reference = true;
        Ok(())
    }

    /// Score one distortion against the cached reference. Returns
    /// `Err(NoCachedReference)` if `set_reference` hasn't been called.
    pub fn compute_with_reference(&mut self, dis_gray: &[f32]) -> Result<GpuIwssimResult> {
        if !self.has_cached_reference {
            return Err(Error::NoCachedReference);
        }
        let expected = (self.width * self.height) as usize;
        if dis_gray.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: dis_gray.len(),
            });
        }
        let h_dis = self.client.create_from_slice(f32::as_bytes(dis_gray));
        self.scales[0].g_dis = h_dis;
        // Skip the ref-side pyramid; only build dis-side.
        self.build_laplacian_pyramid(false);
        // Then the rest of the pipeline reads both `lp_ref[s]` (cached)
        // and `lp_dis[s]` (just built) — same as `run_pipeline`'s
        // post-pyramid stages.
        self.run_pipeline_post_pyramid()
    }

    /// Score one RGB-u8 pair. Both buffers must be `width × height × 3`
    /// in RGB byte order. The pipeline performs the BT.601 rgb→gray
    /// + half-up rounding step on the GPU.
    pub fn compute_rgb(&mut self, ref_rgb: &[u8], dis_rgb: &[u8]) -> Result<GpuIwssimResult> {
        let expected = (self.width * self.height * 3) as usize;
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
        // T_x.O (2026-05-17): pack u8×3 → u32 directly into the
        // pinned staging buffer (one host-side pass instead of two).
        // Previously we packed into `self.pack_scratch` and then
        // `create_from_slice_pinned` copied that scratch into a
        // pinned buffer — two ~48 MB host writes per upload. The
        // reserve_staging path lets us produce the packed bytes
        // straight into the pinned buffer. T4.M's pinned-DMA fast
        // path is preserved (handle from `client.create`).
        //
        // Layout (unchanged from T4.L): 4 bytes per pixel — R | G<<8
        // | B<<16 (alpha unused). Reader (`rgb_u32_to_gray_kernel`)
        // sees the same `[u32]` packing.
        self.src_u32_a = Self::pack_into_pinned(&self.client, ref_rgb);
        self.src_u32_b = Self::pack_into_pinned(&self.client, dis_rgb);

        self.rgb_u32_to_gray_from_packed();
        self.run_pipeline()
    }

    /// Pack a `width × height × 3` sRGB-u8 buffer into the packed-u32
    /// device handle layout that [`Self::compute_handles`] expects.
    /// Uses the same pinned-staging fast path as the internal upload.
    ///
    /// Returns `Err(DimensionMismatch)` if `srgb.len() != width *
    /// height * 3`.
    pub fn pack_srgb_into_packed_u32_handle(
        &self,
        srgb: &[u8],
    ) -> Result<cubecl::server::Handle> {
        let expected = (self.width * self.height * 3) as usize;
        if srgb.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: srgb.len(),
            });
        }
        Ok(Self::pack_into_pinned(&self.client, srgb))
    }

    /// Compute against pre-uploaded packed-u32 device handles —
    /// upload-once Phase 4 entry point. Equivalent to
    /// [`Self::compute_rgb`] but skips the internal byte pack/upload.
    ///
    /// Handle layout MUST be the packed-u32 form produced by
    /// [`Self::pack_srgb_into_packed_u32_handle`] (one `u32` per
    /// pixel, `R | G<<8 | B<<16`, length `width × height`). The
    /// handle is expected to live on the same cubecl client that
    /// constructed this `Iwssim<R>`.
    pub fn compute_handles(
        &mut self,
        ref_handle: &cubecl::server::Handle,
        dis_handle: &cubecl::server::Handle,
    ) -> Result<GpuIwssimResult> {
        self.src_u32_a = ref_handle.clone();
        self.src_u32_b = dis_handle.clone();
        self.rgb_u32_to_gray_from_packed();
        self.run_pipeline()
    }

    /// Run the packed-u32 → grayscale kernel on whichever handles
    /// currently sit in `src_u32_a` / `src_u32_b`. Split out of
    /// [`Self::compute_rgb`] so [`Self::compute_handles`] can reuse
    /// the dispatch step without re-packing bytes.
    fn rgb_u32_to_gray_from_packed(&self) {
        let n_pixels = (self.width * self.height) as usize;
        let s0 = &self.scales[0];
        unsafe {
            rgb2gray::rgb_u32_to_gray_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(n_pixels),
                Self::cube_dim_1d(),
                // T4.L: one u32 per pixel.
                ArrayArg::from_raw_parts(self.src_u32_a.clone(), n_pixels),
                ArrayArg::from_raw_parts(s0.g_ref.clone(), n_pixels),
            );
            rgb2gray::rgb_u32_to_gray_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(n_pixels),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(self.src_u32_b.clone(), n_pixels),
                ArrayArg::from_raw_parts(s0.g_dis.clone(), n_pixels),
            );
        }
    }

    /// Score one grayscale-f32 pair. Both buffers must be `width × height`
    /// floats in the 0..=255 range (matches the reference convention
    /// of `L = 255` for the SSIM constants).
    pub fn compute_gray(&mut self, ref_gray: &[f32], dis_gray: &[f32]) -> Result<GpuIwssimResult> {
        let profile = std::env::var("IWSSIM_PROFILE").is_ok();
        let expected = (self.width * self.height) as usize;
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
        // Direct upload into g_ref / g_dis (scale-0 working Gaussian).
        // Replace the handle so the new contents are visible.
        let t = std::time::Instant::now();
        let h_ref = self.client.create_from_slice(f32::as_bytes(ref_gray));
        let h_dis = self.client.create_from_slice(f32::as_bytes(dis_gray));
        // Swap handles into scale-0. Earlier g_ref/g_dis is dropped.
        self.scales[0].g_ref = h_ref;
        self.scales[0].g_dis = h_dis;
        if profile {
            self.client.sync();
            eprintln!(
                "    stage 'upload': {:.3} ms",
                t.elapsed().as_secs_f64() * 1e3
            );
        }
        self.run_pipeline()
    }

    // ───────────────────────── helpers ─────────────────────────

    fn cube_count_1d(n: usize) -> CubeCount {
        const TPB: u32 = 256;
        let cubes = ((n as u32) + TPB - 1) / TPB;
        CubeCount::Static(cubes.max(1), 1, 1)
    }
    fn cube_dim_1d() -> CubeDim {
        CubeDim::new_1d(256)
    }
    /// T_x.O (2026-05-17): pack 3 sRGB bytes per pixel into ONE u32
    /// (R | G<<8 | B<<16), writing the packed bytes directly into a
    /// freshly-reserved pinned staging buffer and handing the buffer
    /// to `client.create` for DMA. One host-side pass for the data,
    /// no intermediate `Vec<u32>` allocation.
    ///
    /// Layout is little-endian by construction: the kernel reader
    /// `rgb_u32_to_gray_kernel` sees one u32 per pixel.
    fn pack_into_pinned(client: &ComputeClient<R>, src: &[u8]) -> cubecl::server::Handle {
        debug_assert!(src.len().is_multiple_of(3));
        let n_pixels = src.len() / 3;
        let pinned_len = n_pixels * 4;
        let mut staging = client.reserve_staging(&[pinned_len]);
        let mut bytes = staging
            .pop()
            .expect("reserve_staging returned no buffers");
        {
            let dst: &mut [u8] = &mut bytes;
            debug_assert_eq!(dst.len(), pinned_len);
            for (chunk_out, triple) in dst.chunks_exact_mut(4).zip(src.chunks_exact(3)) {
                chunk_out[0] = triple[0];
                chunk_out[1] = triple[1];
                chunk_out[2] = triple[2];
                chunk_out[3] = 0;
            }
        }
        client.create(bytes)
    }

    fn run_pipeline(&mut self) -> Result<GpuIwssimResult> {
        let profile = std::env::var("IWSSIM_PROFILE").is_ok();
        let t0 = std::time::Instant::now();
        // 1. Build Gaussian pyramid + extract LP bands.
        self.build_laplacian_pyramid(true);
        self.build_laplacian_pyramid(false);
        if profile {
            self.client.sync();
            eprintln!(
                "    stage 'lp_pyramid': {:.3} ms",
                t0.elapsed().as_secs_f64() * 1e3
            );
        }
        self.run_pipeline_post_pyramid()
    }

    /// Stages 2-6: SSIM stats → IW path → reductions → score. Called
    /// after `lp_ref[s]` and `lp_dis[s]` are populated at every scale
    /// (either by the full `run_pipeline` flow or by the cached
    /// `compute_with_reference` flow).
    fn run_pipeline_post_pyramid(&mut self) -> Result<GpuIwssimResult> {
        let profile = std::env::var("IWSSIM_PROFILE").is_ok();
        let total_t = std::time::Instant::now();

        // Optional per-scale pyramid stats (set `IWSSIM_DEBUG=1`). Kept
        // because the upConv DC scaling is the most failure-prone piece
        // of the port and these prints are the fastest way to catch a
        // regression.
        if std::env::var("IWSSIM_DEBUG").is_ok() {
            self.debug_pyramid_stats();
        }

        // 2. Per-scale SSIM stats + combine.
        let t = std::time::Instant::now();
        for s in 0..self.scales.len() {
            self.run_ssim_stats(s);
        }
        if profile {
            self.client.sync();
            eprintln!(
                "    stage 'ssim_stats': {:.3} ms",
                t.elapsed().as_secs_f64() * 1e3
            );
        }

        // 3. Per-scale IW path (j = 0..3).
        let t = std::time::Instant::now();
        for s in 0..(self.scales.len() - 1) {
            self.run_iw_scale(s);
            if std::env::var("IWSSIM_DEBUG").is_ok() {
                self.debug_iw_stats(s);
            }
        }
        if profile {
            self.client.sync();
            eprintln!(
                "    stage 'iw_scales': {:.3} ms",
                t.elapsed().as_secs_f64() * 1e3
            );
        }

        // Debug-only: bypass the IW weighting (treat all weights as 1).
        // Reproduces the reference's `iw_flag=False` mode for sanity-
        // checking the CS path independent of the IW path.
        if std::env::var("IWSSIM_NO_IW").is_ok() {
            for s in 0..(self.scales.len() - 1) {
                let sc = &self.scales[s];
                let n_iw = (sc.iw_h as usize) * (sc.iw_w as usize);
                let ones = vec![1.0_f32; n_iw];
                self.scales[s].iw = self.client.create_from_slice(f32::as_bytes(&ones));
            }
        }

        // 4. Reductions per scale.
        // partials and sums are pre-allocated in `new()`. Each
        // weighted_sum / iw_sum / plain_sum thread writes to its own
        // slot (no accumulation), and the finalizer overwrites sums.
        // So nothing needs clearing between calls.
        let partials_len = (NUM_SLOTS * reduction::NUM_BLOCKS * reduction::BLOCK_SIZE) as usize;

        let t = std::time::Instant::now();
        for s in 0..(self.scales.len() - 1) {
            let sc = &self.scales[s];
            let cs_n = (sc.cs_h as usize) * (sc.cs_w as usize);
            let iw_n = (sc.iw_h as usize) * (sc.iw_w as usize);
            reduction::launch_weighted_sum::<R>(
                &self.client,
                sc.cs.clone(),
                cs_n,
                sc.iw.clone(),
                iw_n,
                self.partials.clone(),
                partials_len,
                sc.cs_h,
                sc.cs_w,
                sc.iw_h,
                sc.iw_w,
                BOUND1,
                SLOT_CSIW_BASE + s as u32,
            );
            reduction::launch_iw_sum::<R>(
                &self.client,
                sc.iw.clone(),
                iw_n,
                self.partials.clone(),
                partials_len,
                sc.cs_h,
                sc.cs_w,
                sc.iw_h,
                sc.iw_w,
                BOUND1,
                SLOT_IW_BASE + s as u32,
            );
        }
        // Top scale: Σ(cs · l) over its native shape.
        let top = self.scales.len() - 1;
        let sc_top = &self.scales[top];
        let cs_top_n = (sc_top.cs_h as usize) * (sc_top.cs_w as usize);
        reduction::launch_plain_sum::<R>(
            &self.client,
            sc_top.cs.clone(),
            cs_top_n,
            self.partials.clone(),
            partials_len,
            SLOT_CSL,
        );

        reduction::launch_finalize::<R>(
            &self.client,
            self.partials.clone(),
            partials_len,
            self.sums.clone(),
            NUM_SLOTS as usize,
            NUM_SLOTS,
        );
        if profile {
            self.client.sync();
            eprintln!(
                "    stage 'reductions': {:.3} ms",
                t.elapsed().as_secs_f64() * 1e3
            );
        }

        // 5. Read back and finish on host.
        let t = std::time::Instant::now();
        let bytes = self.client.read_one(self.sums.clone()).expect("read sums");
        let sums = f32::from_bytes(&bytes);
        debug_assert_eq!(sums.len(), NUM_SLOTS as usize);

        let mut per_scale = [0.0_f64; NUM_SCALES];
        for s in 0..(self.scales.len() - 1) {
            let num = sums[(SLOT_CSIW_BASE + s as u32) as usize] as f64;
            let den = sums[(SLOT_IW_BASE + s as u32) as usize] as f64;
            // Reference Python:  wmcs[s] = Σ(cs·iw) / Σ(iw)
            per_scale[s] = if den != 0.0 { num / den } else { 0.0 };
        }
        let top_sum = sums[SLOT_CSL as usize] as f64;
        let top_n = (sc_top.cs_h as usize * sc_top.cs_w as usize) as f64;
        per_scale[top] = top_sum / top_n;

        // 6. Final product: score = Π |wmcs[s]|^β[s]
        let mut score = 1.0_f64;
        for s in 0..self.scales.len() {
            let b = filters::SCALE_WEIGHTS[s] as f64;
            let v = per_scale[s].abs();
            score *= v.powf(b);
        }
        if profile {
            eprintln!(
                "    stage 'readback+score': {:.3} ms",
                t.elapsed().as_secs_f64() * 1e3
            );
            eprintln!(
                "    >>> TOTAL pipeline: {:.3} ms",
                total_t.elapsed().as_secs_f64() * 1e3
            );
        }
        Ok(GpuIwssimResult { score, per_scale })
    }

    /// Build the 5-level Laplacian pyramid for one side (ref or dis).
    /// On entry, `scales[0].g_{ref|dis}` already holds the grayscale
    /// input; on exit, `scales[s].lp_{ref|dis}` holds the LP band at
    /// each scale, and `scales[s].g_{ref|dis}` holds the matching
    /// Gaussian band.
    fn build_laplacian_pyramid(&mut self, is_ref: bool) {
        let n_levels = self.scales.len();

        // 1. Downsample chain: g[s+1] = corrDn_v(corrDn_h(g[s]), binom5).
        for s in 0..(n_levels - 1) {
            let (h_cur, w_cur) = (self.scales[s].h, self.scales[s].w);
            let (h_nxt, w_nxt) = (self.scales[s + 1].h, self.scales[s + 1].w);
            // Reuse the per-scale gh_ref / gh_dis buffer (sized for
            // an `h_cur × (w_cur − 10)` SSIM intermediate, which is
            // strictly larger than `h_cur × w_nxt` we need here — fits).
            let scratch = if is_ref {
                self.scales[s].gh_ref.clone()
            } else {
                self.scales[s].gh_dis.clone()
            };
            let g_cur = if is_ref {
                self.scales[s].g_ref.clone()
            } else {
                self.scales[s].g_dis.clone()
            };
            let g_nxt = if is_ref {
                self.scales[s + 1].g_ref.clone()
            } else {
                self.scales[s + 1].g_dis.clone()
            };
            let n_scratch = (h_cur as usize) * (w_nxt as usize);
            let n_nxt = (h_nxt as usize) * (w_nxt as usize);
            unsafe {
                lap_pyramid::corr_dn_horizontal_kernel::launch_unchecked::<R>(
                    &self.client,
                    Self::cube_count_1d(n_scratch),
                    Self::cube_dim_1d(),
                    ArrayArg::from_raw_parts(g_cur, (h_cur * w_cur) as usize),
                    ArrayArg::from_raw_parts(scratch.clone(), n_scratch),
                    h_cur,
                    w_cur,
                    w_nxt,
                );
                lap_pyramid::corr_dn_vertical_kernel::launch_unchecked::<R>(
                    &self.client,
                    Self::cube_count_1d(n_nxt),
                    Self::cube_dim_1d(),
                    ArrayArg::from_raw_parts(scratch, n_scratch),
                    ArrayArg::from_raw_parts(g_nxt, n_nxt),
                    h_nxt,
                    h_cur,
                    w_nxt,
                );
            }
        }

        // 2. LP residual at top scale.
        let top = n_levels - 1;
        {
            let sc = &self.scales[top];
            let n_top = (sc.h as usize) * (sc.w as usize);
            // LP_top = g_top.  Single copy via pointwise_sub against zero
            // doesn't exist; just clone the handle.
            if is_ref {
                self.scales[top].lp_ref = sc.g_ref.clone();
            } else {
                self.scales[top].lp_dis = sc.g_dis.clone();
            }
            let _ = n_top;
        }

        // 3. Upsample chain: LP[s] = g[s] − upConv_v(upConv_h(g[s+1])).
        for s in (0..(n_levels - 1)).rev() {
            let (h_cur, w_cur) = (self.scales[s].h, self.scales[s].w);
            let (_, w_nxt) = (self.scales[s + 1].h, self.scales[s + 1].w);
            let scratch = if is_ref {
                self.scales[s].gh_ref2.clone()
            } else {
                self.scales[s].gh_dis2.clone()
            };
            let g_nxt = if is_ref {
                self.scales[s + 1].g_ref.clone()
            } else {
                self.scales[s + 1].g_dis.clone()
            };
            let g_cur = if is_ref {
                self.scales[s].g_ref.clone()
            } else {
                self.scales[s].g_dis.clone()
            };
            let lp_cur = if is_ref {
                self.scales[s].lp_ref.clone()
            } else {
                self.scales[s].lp_dis.clone()
            };
            let h_nxt = self.scales[s + 1].h;
            let n_h_scratch = (h_nxt as usize) * (w_cur as usize);
            let n_cur = (h_cur as usize) * (w_cur as usize);
            // expanded: insert zeros + binom5 along width, output (h_nxt, w_cur).
            unsafe {
                lap_pyramid::up_conv_horizontal_kernel::launch_unchecked::<R>(
                    &self.client,
                    Self::cube_count_1d(n_h_scratch),
                    Self::cube_dim_1d(),
                    ArrayArg::from_raw_parts(g_nxt, (h_nxt * w_nxt) as usize),
                    ArrayArg::from_raw_parts(scratch.clone(), n_h_scratch),
                    h_nxt,
                    w_nxt,
                    w_cur,
                );
            }
            // Borrow the per-scale parent_band buffer as scratch for
            // the second pass: it's sized for (h, w) at this scale,
            // which is exactly what we need.
            let scratch2 = self.scales[s].parent_band.clone();
            unsafe {
                lap_pyramid::up_conv_vertical_kernel::launch_unchecked::<R>(
                    &self.client,
                    Self::cube_count_1d(n_cur),
                    Self::cube_dim_1d(),
                    ArrayArg::from_raw_parts(scratch, n_h_scratch),
                    ArrayArg::from_raw_parts(scratch2.clone(), n_cur),
                    h_cur,
                    h_nxt,
                    w_cur,
                );
                lap_pyramid::pointwise_sub_kernel::launch_unchecked::<R>(
                    &self.client,
                    Self::cube_count_1d(n_cur),
                    Self::cube_dim_1d(),
                    ArrayArg::from_raw_parts(g_cur, n_cur),
                    ArrayArg::from_raw_parts(scratch2, n_cur),
                    ArrayArg::from_raw_parts(lp_cur, n_cur),
                );
            }
        }
    }

    /// SSIM stats at one scale: 11×11 separable Gaussian on
    /// `LP_ref`, `LP_dis`, `LP_ref²`, `LP_dis²`, `LP_ref · LP_dis`,
    /// then combine into `cs` (or `cs · l` at the top scale).
    fn run_ssim_stats(&mut self, s: usize) {
        let sc = &self.scales[s];
        let h = sc.h;
        let w = sc.w;
        let cs_w = sc.cs_w;
        let cs_h = sc.cs_h;
        let n_lp = (h as usize) * (w as usize);
        let n_h = (h as usize) * (cs_w as usize);
        let n_cs = (cs_h as usize) * (cs_w as usize);

        unsafe {
            // Horizontal passes: 5 inputs → 5 hstrip outputs.
            // mu1, mu2 (identity)
            gauss11::gauss11_h_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(n_h),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(sc.lp_ref.clone(), n_lp),
                ArrayArg::from_raw_parts(sc.gh_ref.clone(), n_h),
                h,
                w,
                cs_w,
            );
            gauss11::gauss11_h_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(n_h),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(sc.lp_dis.clone(), n_lp),
                ArrayArg::from_raw_parts(sc.gh_dis.clone(), n_h),
                h,
                w,
                cs_w,
            );
            // m11, m22 (squared)
            gauss11::gauss11_h_sq_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(n_h),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(sc.lp_ref.clone(), n_lp),
                ArrayArg::from_raw_parts(sc.gh_ref2.clone(), n_h),
                h,
                w,
                cs_w,
            );
            gauss11::gauss11_h_sq_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(n_h),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(sc.lp_dis.clone(), n_lp),
                ArrayArg::from_raw_parts(sc.gh_dis2.clone(), n_h),
                h,
                w,
                cs_w,
            );
            // m12 (product)
            gauss11::gauss11_h_prod_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(n_h),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(sc.lp_ref.clone(), n_lp),
                ArrayArg::from_raw_parts(sc.lp_dis.clone(), n_lp),
                ArrayArg::from_raw_parts(sc.gh_refdis.clone(), n_h),
                h,
                w,
                cs_w,
            );

            // Vertical passes: 5 hstrip inputs → 5 cs-shaped outputs.
            gauss11::gauss11_v_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(n_cs),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(sc.gh_ref.clone(), n_h),
                ArrayArg::from_raw_parts(sc.mu1.clone(), n_cs),
                cs_h,
                h,
                cs_w,
            );
            gauss11::gauss11_v_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(n_cs),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(sc.gh_dis.clone(), n_h),
                ArrayArg::from_raw_parts(sc.mu2.clone(), n_cs),
                cs_h,
                h,
                cs_w,
            );
            gauss11::gauss11_v_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(n_cs),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(sc.gh_ref2.clone(), n_h),
                ArrayArg::from_raw_parts(sc.m11.clone(), n_cs),
                cs_h,
                h,
                cs_w,
            );
            gauss11::gauss11_v_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(n_cs),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(sc.gh_dis2.clone(), n_h),
                ArrayArg::from_raw_parts(sc.m22.clone(), n_cs),
                cs_h,
                h,
                cs_w,
            );
            gauss11::gauss11_v_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(n_cs),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(sc.gh_refdis.clone(), n_h),
                ArrayArg::from_raw_parts(sc.m12.clone(), n_cs),
                cs_h,
                h,
                cs_w,
            );

            // Combine. Top scale: `cs · l`. Others: `cs` only.
            let is_top = s == self.scales.len() - 1;
            if is_top {
                ssim_combine::ssim_cs_l_kernel::launch_unchecked::<R>(
                    &self.client,
                    Self::cube_count_1d(n_cs),
                    Self::cube_dim_1d(),
                    ArrayArg::from_raw_parts(sc.mu1.clone(), n_cs),
                    ArrayArg::from_raw_parts(sc.mu2.clone(), n_cs),
                    ArrayArg::from_raw_parts(sc.m11.clone(), n_cs),
                    ArrayArg::from_raw_parts(sc.m22.clone(), n_cs),
                    ArrayArg::from_raw_parts(sc.m12.clone(), n_cs),
                    ArrayArg::from_raw_parts(sc.cs.clone(), n_cs),
                );
            } else {
                ssim_combine::ssim_cs_kernel::launch_unchecked::<R>(
                    &self.client,
                    Self::cube_count_1d(n_cs),
                    Self::cube_dim_1d(),
                    ArrayArg::from_raw_parts(sc.mu1.clone(), n_cs),
                    ArrayArg::from_raw_parts(sc.mu2.clone(), n_cs),
                    ArrayArg::from_raw_parts(sc.m11.clone(), n_cs),
                    ArrayArg::from_raw_parts(sc.m22.clone(), n_cs),
                    ArrayArg::from_raw_parts(sc.m12.clone(), n_cs),
                    ArrayArg::from_raw_parts(sc.cs.clone(), n_cs),
                );
            }
        }
    }

    /// Probe pyramid coefficients at every scale — called when
    /// `IWSSIM_DEBUG=1`. Prints G[s] and LP[s] mean/RMS so the upConv
    /// DC scaling can be sanity-checked against the pyrtools reference
    /// (`LP mean` should be ~0 for `s < top`; `LP[top] = G[top]`).
    fn debug_pyramid_stats(&self) {
        for s in 0..self.scales.len() {
            let sc = &self.scales[s];
            let n_lp = (sc.h as usize) * (sc.w as usize);
            let lp_bytes = self.client.read_one(sc.lp_ref.clone()).expect("lp read");
            let lp = f32::from_bytes(&lp_bytes);
            let lp_active = &lp[..n_lp];
            let lp_mean: f64 = lp_active.iter().map(|&v| v as f64).sum::<f64>() / (n_lp as f64);
            let lp_rms = (lp_active
                .iter()
                .map(|&v| (v as f64) * (v as f64))
                .sum::<f64>()
                / n_lp as f64)
                .sqrt();
            let g_bytes = self.client.read_one(sc.g_ref.clone()).expect("g read");
            let g = f32::from_bytes(&g_bytes);
            let g_active = &g[..n_lp];
            let g_mean: f64 = g_active.iter().map(|&v| v as f64).sum::<f64>() / (n_lp as f64);
            let g_rms = (g_active
                .iter()
                .map(|&v| (v as f64) * (v as f64))
                .sum::<f64>()
                / n_lp as f64)
                .sqrt();
            eprintln!(
                "PYR | s={} (h={},w={}) | G mean={:.4} rms={:.4} | LP mean={:.4} rms={:.4}",
                s, sc.h, sc.w, g_mean, g_rms, lp_mean, lp_rms,
            );
        }
    }

    /// Probe iw values for the most recent IW scale — called at the
    /// end of `run_iw_scale` when IWSSIM_DEBUG is set.
    fn debug_iw_stats(&self, s: usize) {
        let sc = &self.scales[s];
        let n_iw = (sc.iw_h as usize) * (sc.iw_w as usize);
        let iw_bytes = self.client.read_one(sc.iw.clone()).expect("iw read");
        let iw_arr = f32::from_bytes(&iw_bytes);
        let active = &iw_arr[..n_iw];
        let any_inf = active.iter().any(|v| v.is_infinite());
        let any_nan = active.iter().any(|v| v.is_nan());
        let iw_max = active.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v));
        let iw_min = active.iter().fold(f32::INFINITY, |a, &v| a.min(v));
        let iw_mean: f64 = active.iter().map(|&v| v as f64).sum::<f64>() / (n_iw as f64);
        eprintln!(
            "scale {} | iw min={:.3e} max={:.3e} mean={:.3e} any_inf={} any_nan={}",
            s, iw_min, iw_max, iw_mean, any_inf, any_nan
        );
    }

    /// Per-scale IW path for scale `s ∈ 0..NUM_SCALES − 1`.
    fn run_iw_scale(&mut self, s: usize) {
        let sc = &self.scales[s];
        let h = sc.h;
        let w = sc.w;
        let n_lp = (h as usize) * (w as usize);
        let n_iw = (sc.iw_h as usize) * (sc.iw_w as usize);

        // 1. 3×3 box stats → g, vv at LP shape.
        unsafe {
            box3::box3_gv_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(n_lp),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(sc.lp_ref.clone(), n_lp),
                ArrayArg::from_raw_parts(sc.lp_dis.clone(), n_lp),
                ArrayArg::from_raw_parts(sc.g_buf.clone(), n_lp),
                ArrayArg::from_raw_parts(sc.vv_buf.clone(), n_lp),
                w,
                h,
            );
        }

        // 2. Parent band? Present when s < Nsc − 2 (matches Python's
        //    `parent and scale < Nsc − 1` with 1-indexed scale).
        let has_parent = s < self.scales.len() - 2;

        if has_parent {
            // imenlarge2(LP[s+1]) cropped to (h, w).
            let nxt = &self.scales[s + 1];
            unsafe {
                imenlarge2::imenlarge2_kernel::launch_unchecked::<R>(
                    &self.client,
                    Self::cube_count_1d(n_lp),
                    Self::cube_dim_1d(),
                    ArrayArg::from_raw_parts(nxt.lp_ref.clone(), (nxt.h * nxt.w) as usize),
                    ArrayArg::from_raw_parts(sc.parent_band.clone(), n_lp),
                    nxt.h,
                    nxt.w,
                    h,
                    w,
                );
            }
        }

        // 3. Reset cu_atomic via a tiny zero kernel — cheaper than
        // re-allocating the 100-float buffer via create_from_slice
        // each call.
        unsafe {
            util::zero_kernel::launch_unchecked::<R>(
                &self.client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(128),
                ArrayArg::from_raw_parts(sc.cu_atomic.clone(), 100),
            );
        }
        let sc = &self.scales[s]; // re-borrow
        let n_blk = ((sc.iw_h as u32) * (sc.iw_w as u32)) as usize;
        unsafe {
            if has_parent {
                cov::cov_accum_with_parent_kernel::launch_unchecked::<R>(
                    &self.client,
                    CubeCount::Static(64, 1, 1),
                    CubeDim::new_1d(256),
                    ArrayArg::from_raw_parts(sc.lp_ref.clone(), n_lp),
                    ArrayArg::from_raw_parts(sc.parent_band.clone(), n_lp),
                    ArrayArg::from_raw_parts(sc.cu_atomic.clone(), 100),
                    h,
                    w,
                );
            } else {
                cov::cov_accum_no_parent_kernel::launch_unchecked::<R>(
                    &self.client,
                    CubeCount::Static(64, 1, 1),
                    CubeDim::new_1d(256),
                    ArrayArg::from_raw_parts(sc.lp_ref.clone(), n_lp),
                    ArrayArg::from_raw_parts(sc.cu_atomic.clone(), 100),
                    h,
                    w,
                );
            }
        }
        let _ = n_blk;

        // 4. Read C_u back to host, eigendecompose + invert.
        let cu_bytes = self
            .client
            .read_one(sc.cu_atomic.clone())
            .expect("read C_u");
        let cu_f32 = f32::from_bytes(&cu_bytes);
        let n_dim = if has_parent { 10 } else { 9 };
        // C_u = (1/nexp) · accumulated Yᵀ Y
        let nexp = (sc.iw_h as f64) * (sc.iw_w as f64);
        let mut cu_f64 = vec![0.0_f64; n_dim * n_dim];
        // The atomic buffer is laid out as 10×10; for n_dim=9 we read
        // only the top-left 9×9 block (the rest is unused / zero).
        for i in 0..n_dim {
            for j in 0..n_dim {
                let src_idx = i * if has_parent { 10 } else { 9 } + j;
                cu_f64[i * n_dim + j] = (cu_f32[src_idx] as f64) / nexp;
            }
        }
        if std::env::var("IWSSIM_DEBUG").is_ok() {
            let trace: f64 = (0..n_dim).map(|i| cu_f64[i * n_dim + i]).sum();
            let max_abs = cu_f64.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
            let any_nan = cu_f64.iter().any(|v| v.is_nan());
            // Also probe the LP and parent buffers — RMS + first
            // few values.
            let lp_bytes = self.client.read_one(sc.lp_ref.clone()).expect("lp read");
            let lp = f32::from_bytes(&lp_bytes);
            let lp_active = &lp[..(h as usize) * (w as usize)];
            let lp_rms = (lp_active
                .iter()
                .map(|&v| (v as f64) * (v as f64))
                .sum::<f64>()
                / lp_active.len() as f64)
                .sqrt();
            let lp_nan = lp_active.iter().any(|v| v.is_nan());
            eprintln!(
                "scale {} | n_dim {} | nexp {} | C_u trace={:.6e} max|·|={:.6e} any_nan={} | LP rms={:.4} nan={} first5={:.3?} ",
                s,
                n_dim,
                nexp,
                trace,
                max_abs,
                any_nan,
                lp_rms,
                lp_nan,
                &lp_active[..5]
            );
        }
        let eig_result = eig::decompose_and_invert(&cu_f64, n_dim);
        if std::env::var("IWSSIM_DEBUG").is_ok() {
            let lam: Vec<f32> = eig_result.lambda[..n_dim].to_vec();
            eprintln!("scale {} | eigvals: {:?}", s, lam);
            let cu_inv_min = eig_result.c_u_inv[..n_dim * n_dim]
                .iter()
                .fold(f32::INFINITY, |a, &v| a.min(v));
            let cu_inv_max = eig_result.c_u_inv[..n_dim * n_dim]
                .iter()
                .fold(f32::NEG_INFINITY, |a, &v| a.max(v));
            eprintln!(
                "scale {} | C_u_inv range [{:.3e}, {:.3e}]",
                s, cu_inv_min, cu_inv_max
            );
        }

        // 5. Upload eigenvalues and C_u_inv back to device.
        let lambda_slice = &eig_result.lambda[..n_dim];
        let cu_inv_slice = &eig_result.c_u_inv[..n_dim * n_dim];
        self.scales[s].lambda_dev = self.client.create_from_slice(f32::as_bytes(lambda_slice));
        self.scales[s].cu_inv_dev = self.client.create_from_slice(f32::as_bytes(cu_inv_slice));
        let sc = &self.scales[s];

        if std::env::var("IWSSIM_DEBUG").is_ok() {
            let g_bytes = self.client.read_one(sc.g_buf.clone()).expect("g read");
            let g_arr = f32::from_bytes(&g_bytes);
            let vv_bytes = self.client.read_one(sc.vv_buf.clone()).expect("vv read");
            let vv_arr = f32::from_bytes(&vv_bytes);
            let np = (h as usize) * (w as usize);
            let g_min = g_arr[..np].iter().cloned().fold(f32::INFINITY, f32::min);
            let g_max = g_arr[..np]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let vv_min = vv_arr[..np].iter().cloned().fold(f32::INFINITY, f32::min);
            let vv_max = vv_arr[..np]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            eprintln!(
                "scale {} | g range [{:.3e}, {:.3e}] vv range [{:.3e}, {:.3e}]",
                s, g_min, g_max, vv_min, vv_max
            );
        }

        // 6. infow kernel.
        unsafe {
            if has_parent {
                infow::infow_with_parent_kernel::launch_unchecked::<R>(
                    &self.client,
                    Self::cube_count_1d(n_iw),
                    Self::cube_dim_1d(),
                    ArrayArg::from_raw_parts(sc.lp_ref.clone(), n_lp),
                    ArrayArg::from_raw_parts(sc.parent_band.clone(), n_lp),
                    ArrayArg::from_raw_parts(sc.g_buf.clone(), n_lp),
                    ArrayArg::from_raw_parts(sc.vv_buf.clone(), n_lp),
                    ArrayArg::from_raw_parts(sc.cu_inv_dev.clone(), 100),
                    ArrayArg::from_raw_parts(sc.lambda_dev.clone(), 10),
                    ArrayArg::from_raw_parts(sc.iw.clone(), n_iw),
                    h,
                    w,
                    0.4_f32, // sigma_nsq — paper default
                );
            } else {
                infow::infow_no_parent_kernel::launch_unchecked::<R>(
                    &self.client,
                    Self::cube_count_1d(n_iw),
                    Self::cube_dim_1d(),
                    ArrayArg::from_raw_parts(sc.lp_ref.clone(), n_lp),
                    ArrayArg::from_raw_parts(sc.g_buf.clone(), n_lp),
                    ArrayArg::from_raw_parts(sc.vv_buf.clone(), n_lp),
                    ArrayArg::from_raw_parts(sc.cu_inv_dev.clone(), 81),
                    ArrayArg::from_raw_parts(sc.lambda_dev.clone(), 9),
                    ArrayArg::from_raw_parts(sc.iw.clone(), n_iw),
                    h,
                    w,
                    0.4_f32,
                );
            }
        }
    }
}
