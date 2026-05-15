//! cvvdp pipeline orchestration.
//!
//! Wires the kernels in [`crate::kernels`] into a still-image
//! ColorVideoVDP scorer.
//!
//! Public entry points:
//! - [`Cvvdp::new`] + [`Cvvdp::score`] — one-shot scoring of a
//!   (reference, distorted) pair.
//! - [`Cvvdp::set_reference`] + [`Cvvdp::score_with_reference`] —
//!   reference-side cache for encoder loops that compare many
//!   candidates to the same source.
//!
//! ## Algorithm overview (per call)
//!
//! 1. Upload sRGB-u8 bytes for both sides (or skip reference side
//!    when cached).
//! 2. Run `color::srgb_to_dkl_kernel` once per side → 3 planar DKL
//!    buffers each (achromatic + RG + VY).
//! 3. Build per-channel Weber-contrast pyramids
//!    (`pyramid::weber_contrast_compute_kernel` over each band of a
//!    decimating Gaussian pyramid built via `downscale_kernel` +
//!    `upscale_{v,h}_kernel` + `subtract_kernel`). Yields
//!    `n_levels` Weber-contrast bands per channel per side plus a
//!    per-pixel `log10(L_bkg)` map from the achromatic gauss for
//!    step 4. The coarsest band (the gaussian base) bypasses Weber
//!    contrast and feeds directly into pooling.
//! 4. Per-pixel CSF apply via `csf::csf_apply_per_pixel_kernel`
//!    (per-band `rho` resolved via `csf::precompute_logs_row`).
//!    Output `T_p` = Weber × S(rho, L_bkg, channel) × CH_GAIN.
//! 5. Multi-channel mult-mutual masking via
//!    `masking::mult_mutual_3ch_no_blur_kernel` (small bands) or the
//!    `min_abs_3ch_kernel` → `pu_blur_h_kernel` → `pu_blur_v_kernel`
//!    → `mult_mutual_3ch_with_blurred_kernel` chain (bands larger
//!    than `PU_PADSIZE`).
//! 6. Per-band Minkowski accumulation (`pool::pool_band_kernel`) →
//!    per-band f32 partials.
//! 7. Host-side fold: per-band → per-channel → overall `D` via the
//!    3-stage Minkowski pool, then `pool::met2jod` piecewise.
//!
//! ## Buffer layout
//!
//! Per side, per channel: one `width × height` plane at level 0, then
//! `width/2 × height/2`, … geometrically decimating. No SIMD-pad columns
//! (cvvdp's reference doesn't pad). One `Handle` per (side, channel,
//! level) — for a 1024² image with 3 channels and 7 levels that's 42
//! plane handles, allocated once in `new()` and reused.

use cubecl::prelude::*;

use crate::kernels::color::{SRGB8_TO_LINEAR_LUT, srgb_to_dkl_kernel};
use crate::kernels::csf::{
    CsfChannel, csf_apply_3ch_kernel, csf_apply_6ch_kernel, flatten_band_weights,
    precompute_logs_row, precomputed_band_weights, weight_band_kernel,
};
use crate::kernels::masking::{
    CH_GAIN, MASK_C, PU_PADSIZE, diff_abs_3ch_kernel, min_abs_3ch_kernel,
    mult_mutual_3ch_no_blur_kernel, mult_mutual_3ch_with_blurred_kernel, pu_blur_h_3ch_kernel,
    pu_blur_v_3ch_scaled_kernel,
};
use crate::kernels::pool::{
    BETA_SPATIAL, do_pooling_and_jod_still_3ch, pool_band_finalize, pool_band_kernel,
};
use crate::kernels::pyramid::{
    band_frequencies, downscale_kernel, subtract_kernel, subtract_weber_3ch_kernel,
    upscale_h_kernel, upscale_v_kernel,
};
use crate::params::CvvdpParams;
use crate::{Error, MAX_LEVELS, N_CHANNELS, PYRAMID_MIN_DIM, Result};

/// Return shape of [`Cvvdp::compute_dkl_weber_pyramid`].
///
/// - `.0` — `levels[k] = [a, rg, vy]` Weber-contrast bands. Same
///   layout as `compute_dkl_laplacian_pyramid`'s output.
/// - `.1` — `levels[k]` per-pixel `log10(L_bkg)` plane for non-
///   baseband levels, replicated scalar for the baseband. Matches
///   `host_scalar::WeberPyramid::log_l_bkg`.
pub type WeberPyramidGpu = (Vec<[Vec<f32>; 3]>, Vec<Vec<f32>>);

/// One pyramid level: a `width × height` planar f32 buffer per channel.
struct Level {
    w: u32,
    h: u32,
    /// One f32 plane per DKL channel.
    planes: [cubecl::server::Handle; N_CHANNELS],
}

/// Per-level scratch buffers reused by `compute_dkl_d_bands` so the
/// hot loop doesn't allocate per band. At 12 MP the function would
/// otherwise allocate ~1.5 GB of transient GPU buffers per call (3
/// channels × 2 sides × 6 buffer kinds × per-level size). Pre-
/// allocating once on `Cvvdp::new` keeps the steady-state cost off
/// the per-frame budget.
struct DBandsScratch {
    /// CSF-applied bands per channel for ref and dist sides.
    /// `compute_dkl_d_bands` runs `csf_apply_per_pixel_kernel` into
    /// these (one launch per side per channel).
    t_p_ref: [cubecl::server::Handle; N_CHANNELS],
    t_p_dis: [cubecl::server::Handle; N_CHANNELS],
    /// Masking-chain scratch (non-baseband levels only).
    m_raw: [cubecl::server::Handle; N_CHANNELS],
    m_mid: [cubecl::server::Handle; N_CHANNELS],
    m_blur: [cubecl::server::Handle; N_CHANNELS],
    /// Per-band masked-difference output (consumed by host
    /// `lp_norm_mean` after read-back).
    d: [cubecl::server::Handle; N_CHANNELS],
}

fn alloc_zeros_f32<R: Runtime>(client: &ComputeClient<R>, n: usize) -> cubecl::server::Handle {
    client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]))
}

/// Per-level scratch buffers reused by `compute_dkl_weber_pyramid`.
/// At 12 MP the function would otherwise allocate ~140 MB of
/// transient GPU buffers per call (l_bkg_fine, vscratch_a, log_l_bkg
/// per level + vscratch_c/upscaled_c per (level, channel)).
/// Called twice per `compute_dkl_d_bands` so doubled per d_bands call.
///
/// `coarse_w * fine_h` shape (n_v) is the vertical-pass scratch for
/// upscale_v_kernel; `fine_w * fine_h` shape (n_fine) is everything
/// else.
struct WeberScratch {
    /// Expanded achromatic L_bkg, shared across channels (n_fine).
    l_bkg_fine: cubecl::server::Handle,
    /// Vertical-pass scratch for achromatic L_bkg expand (n_v).
    vscratch_a: cubecl::server::Handle,
    /// Per-pixel log10(L_bkg) plane (n_fine). Re-used for both sides.
    log_l_bkg: cubecl::server::Handle,
    /// Per-channel vertical/horizontal expand scratch (n_v, n_fine).
    /// The previous `layer_c` intermediate is gone — tick 91 fuses
    /// `subtract + weber` into a single 3-channel kernel that reads
    /// `fine` + `upscaled_c` directly.
    vscratch_c: [cubecl::server::Handle; N_CHANNELS],
    upscaled_c: [cubecl::server::Handle; N_CHANNELS],
}

fn build_weber_scratch<R: Runtime>(
    client: &ComputeClient<R>,
    n_levels: usize,
    width: u32,
    height: u32,
) -> Vec<WeberScratch> {
    let mut out = Vec::with_capacity(n_levels.saturating_sub(1));
    let mut fine_w = width;
    let mut fine_h = height;
    // Only non-baseband levels need scratch (baseband bypasses the
    // expand/subtract/weber chain).
    for _ in 0..n_levels.saturating_sub(1) {
        let coarse_w = fine_w / 2;
        let n_fine = (fine_w as usize) * (fine_h as usize);
        let n_v = (coarse_w as usize) * (fine_h as usize);
        out.push(WeberScratch {
            l_bkg_fine: alloc_zeros_f32(client, n_fine),
            vscratch_a: alloc_zeros_f32(client, n_v),
            log_l_bkg: alloc_zeros_f32(client, n_fine),
            vscratch_c: [
                alloc_zeros_f32(client, n_v),
                alloc_zeros_f32(client, n_v),
                alloc_zeros_f32(client, n_v),
            ],
            upscaled_c: [
                alloc_zeros_f32(client, n_fine),
                alloc_zeros_f32(client, n_fine),
                alloc_zeros_f32(client, n_fine),
            ],
        });
        fine_w = coarse_w;
        fine_h /= 2;
    }
    out
}

fn build_d_bands_scratch<R: Runtime>(
    client: &ComputeClient<R>,
    n_levels: usize,
    width: u32,
    height: u32,
) -> Vec<DBandsScratch> {
    let mut out = Vec::with_capacity(n_levels);
    let mut w = width;
    let mut h = height;
    for _ in 0..n_levels {
        let n = (w as usize) * (h as usize);
        out.push(DBandsScratch {
            t_p_ref: [
                alloc_zeros_f32(client, n),
                alloc_zeros_f32(client, n),
                alloc_zeros_f32(client, n),
            ],
            t_p_dis: [
                alloc_zeros_f32(client, n),
                alloc_zeros_f32(client, n),
                alloc_zeros_f32(client, n),
            ],
            m_raw: [
                alloc_zeros_f32(client, n),
                alloc_zeros_f32(client, n),
                alloc_zeros_f32(client, n),
            ],
            m_mid: [
                alloc_zeros_f32(client, n),
                alloc_zeros_f32(client, n),
                alloc_zeros_f32(client, n),
            ],
            m_blur: [
                alloc_zeros_f32(client, n),
                alloc_zeros_f32(client, n),
                alloc_zeros_f32(client, n),
            ],
            d: [
                alloc_zeros_f32(client, n),
                alloc_zeros_f32(client, n),
                alloc_zeros_f32(client, n),
            ],
        });
        w /= 2;
        h /= 2;
    }
    out
}

/// Reference-side state kept across `score_with_reference` calls.
///
/// Stashes the raw sRGB bytes so the host-scalar pipeline can re-run
/// end-to-end per distorted candidate — same bytes that `score()`
/// would have re-uploaded, just kept around. Exact-parity with
/// `score(ref, dist)`.
///
/// The fast path that materializes the reference's CSF-weighted
/// pyramid bands once (`Vec<Vec<Handle>>`, indexed `[level][channel]`)
/// is the obvious next optimization but isn't wired yet — every
/// `score_with_reference` call still re-runs the full host pipeline.
/// The GPU helpers (`compute_dkl_weber_pyramid` and friends) exist
/// and could be retargeted here once `Cvvdp::score` itself routes
/// through the GPU composition.
struct CachedReference {
    /// Cached reference sRGB bytes (length `width * height * 3`).
    ref_srgb: Vec<u8>,
}

/// ColorVideoVDP scorer.
///
/// Allocates GPU buffers up front for a fixed image size and reuses
/// them across calls. To score images of a different size, construct
/// a new `Cvvdp`.
pub struct Cvvdp<R: Runtime> {
    client: ComputeClient<R>,
    params: CvvdpParams,
    /// Viewing geometry — drives PPD (= cy/deg) for the CSF lookup.
    /// Independent of `width`/`height` (the image dimensions) since
    /// cvvdp's PPD is a display property, not an image one.
    geometry: crate::params::DisplayGeometry,
    width: u32,
    height: u32,
    n_levels: u32,

    /// sRGB byte upload scratch. The GPU helpers reuse this slot
    /// for both ref and dist (writing one side, running the
    /// pipeline, reading back, then overwriting for the other
    /// side). A second `_dis` slot was originally allocated but
    /// went unused — kept on one buffer to save ~3 MB at 256×256.
    src_ref: cubecl::server::Handle,

    /// 256-entry sRGB→linear LUT, uploaded once.
    srgb_lut: cubecl::server::Handle,

    /// Gaussian pyramid buffers (per channel, per level). Reused
    /// for both sides — each `compute_dkl_*` call overwrites these
    /// for the side it's currently processing then reads back.
    gauss_ref: Vec<Level>,

    /// Pyramid-band buffers (per channel, per level). Reused for
    /// both sides like `gauss_ref`. Coarsest level shares storage
    /// with the coarsest gaussian for the Weber baseband path.
    bands_ref: Vec<Level>,

    /// Per-level scratch for `compute_dkl_d_bands`'s CSF, masking,
    /// and D output buffers. Pre-allocated so the hot loop doesn't
    /// churn GPU allocations per band (~1.5 GB worth at 12 MP).
    d_scratch: Vec<DBandsScratch>,

    /// Per-non-baseband-level scratch for `compute_dkl_weber_pyramid`'s
    /// expand/subtract/weber chain. Pre-allocated; reused per side
    /// per call. ~176 MB worth at 12 MP per call.
    weber_scratch: Vec<WeberScratch>,

    /// Stable per-pixel `log10(L_bkg)` storage for the REFERENCE
    /// side, intended for future use by `compute_dkl_d_bands` to
    /// preserve REF log_l_bkg across the dist weber call (cvvdp's
    /// `weber_g1` rule). Currently unused — tick 85's attempt to
    /// wire this into a GPU-resident d_bands caused a 5× perf
    /// regression in standalone weber for reasons not yet diagnosed
    /// (tick 86 bisect confirmed the allocation + parameter passing
    /// are NOT the cause; the regression is in the d_bands rewrite
    /// itself). One handle per non-baseband level, n_pixels f32 each.
    #[allow(dead_code)]
    ref_log_l_bkg: Vec<cubecl::server::Handle>,

    /// Pre-uploaded logs_row buffers for the CSF per-pixel apply.
    /// Indexed `[level][channel]`. Each holds the 32-entry
    /// `precompute_logs_row(rho_k, channel)` result. rho_k depends
    /// on `geometry.pixels_per_degree()` which is fixed per Cvvdp
    /// — so these are stable across calls and reuploading per band
    /// is pure waste (was 24 uploads of 128 B per call).
    logs_row: Vec<[cubecl::server::Handle; N_CHANNELS]>,

    /// Reference-side cache (used by `score_with_reference`).
    cached: Option<CachedReference>,
}

fn pyramid_levels(width: u32, height: u32) -> u32 {
    let min = width.min(height);
    let mut levels = 1u32;
    let mut cur = min;
    while cur >= 2 * PYRAMID_MIN_DIM && (levels as usize) < MAX_LEVELS {
        cur /= 2;
        levels += 1;
    }
    levels
}

impl<R: Runtime> Cvvdp<R> {
    /// Allocate GPU buffers for a fixed `width × height` image and the
    /// given parameter bundle. Uses
    /// [`crate::params::DisplayGeometry::STANDARD_4K`] as the viewing
    /// geometry — equivalent to `new_with_geometry(..., STANDARD_4K)`.
    /// Override via `new_with_geometry` for non-4K displays.
    ///
    /// Returns [`Error::InvalidImageSize`] if either dimension is
    /// smaller than [`PYRAMID_MIN_DIM`] × 2 (no usable pyramid).
    pub fn new(
        client: ComputeClient<R>,
        width: u32,
        height: u32,
        params: CvvdpParams,
    ) -> Result<Self> {
        Self::new_with_geometry(
            client,
            width,
            height,
            params,
            crate::params::DisplayGeometry::STANDARD_4K,
        )
    }

    /// Allocate GPU buffers + record a custom viewing geometry. The
    /// geometry is used by `score` to derive PPD (and thus the
    /// per-band spatial frequencies the CSF table is queried with).
    pub fn new_with_geometry(
        client: ComputeClient<R>,
        width: u32,
        height: u32,
        params: CvvdpParams,
        geometry: crate::params::DisplayGeometry,
    ) -> Result<Self> {
        if width < PYRAMID_MIN_DIM * 2 || height < PYRAMID_MIN_DIM * 2 {
            return Err(Error::InvalidImageSize);
        }
        let n_levels = pyramid_levels(width, height);

        let n0 = (width as usize) * (height as usize);
        // Source-byte buffers are u32-slot arrays of length `n0 * 3`
        // — one byte per slot, RGBRGB row-major. Matches what
        // `srgb_to_dkl_kernel` expects.
        let src_ref = client.create_from_slice(u32::as_bytes(&vec![0u32; n0 * 3]));
        let srgb_lut = client.create_from_slice(f32::as_bytes(&SRGB8_TO_LINEAR_LUT));

        let build_pyramid = |client: &ComputeClient<R>| -> Vec<Level> {
            let mut out = Vec::with_capacity(n_levels as usize);
            let mut w = width;
            let mut h = height;
            for _ in 0..n_levels {
                let n = (w as usize) * (h as usize);
                out.push(Level {
                    w,
                    h,
                    planes: [
                        alloc_zeros_f32(client, n),
                        alloc_zeros_f32(client, n),
                        alloc_zeros_f32(client, n),
                    ],
                });
                w /= 2;
                h /= 2;
            }
            out
        };

        let gauss_ref = build_pyramid(&client);
        let bands_ref = build_pyramid(&client);
        let d_scratch = build_d_bands_scratch(&client, n_levels as usize, width, height);
        let weber_scratch = build_weber_scratch(&client, n_levels as usize, width, height);

        // Stable REF log_l_bkg buffers (one per non-baseband level).
        // Allocated for future GPU-resident d_bands use.
        let mut ref_log_l_bkg: Vec<cubecl::server::Handle> =
            Vec::with_capacity(n_levels as usize - 1);
        let mut log_w = width;
        let mut log_h = height;
        for _ in 0..(n_levels as usize).saturating_sub(1) {
            ref_log_l_bkg.push(alloc_zeros_f32(
                &client,
                (log_w as usize) * (log_h as usize),
            ));
            log_w /= 2;
            log_h /= 2;
        }

        // Pre-upload logs_row per (level, channel) — depends only on
        // (rho_k, channel) which are fixed for this Cvvdp.
        let ppd = geometry.pixels_per_degree();
        let freqs = band_frequencies(ppd, width as usize, height as usize);
        let channels = [CsfChannel::A, CsfChannel::Rg, CsfChannel::Vy];
        let mut logs_row: Vec<[cubecl::server::Handle; N_CHANNELS]> =
            Vec::with_capacity(n_levels as usize);
        for k in 0..n_levels as usize {
            let rho_k = freqs[k];
            logs_row.push([
                client.create_from_slice(f32::as_bytes(&precompute_logs_row(
                    rho_k,
                    channels[0],
                ))),
                client.create_from_slice(f32::as_bytes(&precompute_logs_row(
                    rho_k,
                    channels[1],
                ))),
                client.create_from_slice(f32::as_bytes(&precompute_logs_row(
                    rho_k,
                    channels[2],
                ))),
            ]);
        }

        Ok(Self {
            client,
            params,
            geometry,
            width,
            height,
            n_levels,
            src_ref,
            srgb_lut,
            gauss_ref,
            bands_ref,
            d_scratch,
            weber_scratch,
            ref_log_l_bkg,
            logs_row,
            cached: None,
        })
    }

    /// Pyramid level `k`'s spatial dimensions as `(bw, bh, n_px)`.
    /// `bw = width >> k`, `bh = height >> k`. At `k = 0` the shift
    /// is a no-op so this matches the source-image dims; deeper
    /// levels are floor-halved. Mirrors what `build_weber_scratch`
    /// and `build_d_bands_scratch` use when allocating per-level
    /// scratch.
    fn level_dims(&self, k: usize) -> (usize, usize, usize) {
        let bw = (self.width as usize) >> k;
        let bh = (self.height as usize) >> k;
        (bw, bh, bw * bh)
    }

    /// Run only the color stage: upload sRGB bytes, launch the
    /// `srgb_to_dkl_kernel`, and read back three planar `f32` buffers
    /// (A, RG, VY) in row-major order.
    ///
    /// Used by integration tests + downstream stages that consume DKL
    /// planes. Equivalent to:
    ///
    /// ```text
    /// for pixel in srgb { srgb_byte_to_dkl_scalar(pixel, display) }
    /// ```
    ///
    /// but executed on the GPU.
    pub fn compute_dkl_planes(&mut self, srgb: &[u8]) -> Result<[Vec<f32>; 3]> {
        let expected = (self.width as usize) * (self.height as usize) * 3;
        if srgb.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: srgb.len(),
            });
        }
        let n0 = (self.width as usize) * (self.height as usize);

        // Widen bytes to u32 slots and re-upload.
        let src_u32: Vec<u32> = srgb.iter().map(|&b| b as u32).collect();
        self.src_ref = self.client.create_from_slice(u32::as_bytes(&src_u32));

        // The level-0 gauss planes double as the color stage's output
        // — that's where the pyramid expects DKL to land at scale 0.
        let a_handle = self.gauss_ref[0].planes[0].clone();
        let rg_handle = self.gauss_ref[0].planes[1].clone();
        let vy_handle = self.gauss_ref[0].planes[2].clone();

        let cube_dim = CubeDim::new_1d(64);
        let cube_count = CubeCount::Static((n0 as u32).div_ceil(64), 1, 1);

        let display = self.params.display;
        unsafe {
            srgb_to_dkl_kernel::launch::<R>(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(self.src_ref.clone(), n0 * 3),
                ArrayArg::from_raw_parts(self.srgb_lut.clone(), SRGB8_TO_LINEAR_LUT.len()),
                ArrayArg::from_raw_parts(a_handle.clone(), n0),
                ArrayArg::from_raw_parts(rg_handle.clone(), n0),
                ArrayArg::from_raw_parts(vy_handle.clone(), n0),
                self.width,
                self.height,
                display.y_peak,
                display.y_black,
                display.y_refl,
            );
        }

        let a_bytes = self
            .client
            .read_one(a_handle)
            .map_err(|_| Error::InvalidImageSize)?;
        let rg_bytes = self
            .client
            .read_one(rg_handle)
            .map_err(|_| Error::InvalidImageSize)?;
        let vy_bytes = self
            .client
            .read_one(vy_handle)
            .map_err(|_| Error::InvalidImageSize)?;

        Ok([
            f32::from_bytes(&a_bytes).to_vec(),
            f32::from_bytes(&rg_bytes).to_vec(),
            f32::from_bytes(&vy_bytes).to_vec(),
        ])
    }

    /// Run color stage + Gaussian-pyramid reduce loop. Returns the
    /// pyramid as `levels[k] = [a, rg, vy]` planar f32 vecs, with
    /// `levels[0]` at base resolution and each subsequent level
    /// halved (cvvdp's `div_ceil(2)` convention).
    pub fn compute_dkl_gauss_pyramid(&mut self, srgb: &[u8]) -> Result<Vec<[Vec<f32>; 3]>> {
        let _ = self.compute_dkl_planes(srgb)?;

        // The color stage left level-0 planes filled in gauss_ref.
        // Now chain downscale_kernel: gauss[k-1].channel[c] → gauss[k].channel[c]
        // for k in 1..n_levels.
        let cube_dim = CubeDim::new_1d(64);
        for k in 1..(self.n_levels as usize) {
            let prev_w = self.gauss_ref[k - 1].w;
            let prev_h = self.gauss_ref[k - 1].h;
            let curr_w = self.gauss_ref[k].w;
            let curr_h = self.gauss_ref[k].h;
            let n_curr = (curr_w * curr_h) as usize;
            let n_prev = (prev_w * prev_h) as usize;
            let cube_count = CubeCount::Static((n_curr as u32).div_ceil(64), 1, 1);

            for c in 0..N_CHANNELS {
                let src = self.gauss_ref[k - 1].planes[c].clone();
                let dst = self.gauss_ref[k].planes[c].clone();
                unsafe {
                    downscale_kernel::launch::<R>(
                        &self.client,
                        cube_count.clone(),
                        cube_dim,
                        ArrayArg::from_raw_parts(src, n_prev),
                        ArrayArg::from_raw_parts(dst, n_curr),
                        prev_w,
                        prev_h,
                        curr_w,
                        curr_h,
                    );
                }
            }
        }

        // Read back every level × every channel.
        let mut out: Vec<[Vec<f32>; 3]> = Vec::with_capacity(self.n_levels as usize);
        for k in 0..(self.n_levels as usize) {
            let mut planes = [Vec::new(), Vec::new(), Vec::new()];
            for c in 0..N_CHANNELS {
                let h = self.gauss_ref[k].planes[c].clone();
                let bytes = self
                    .client
                    .read_one(h)
                    .map_err(|_| Error::InvalidImageSize)?;
                planes[c] = f32::from_bytes(&bytes).to_vec();
            }
            out.push(planes);
        }
        Ok(out)
    }

    /// Run color + full Laplacian-pyramid decomposition. Returns
    /// `levels[k] = [a, rg, vy]` planar f32 bands matching cvvdp's
    /// `lpyr_dec.laplacian_pyramid_dec`:
    ///
    /// - `levels[k]` for `k < n_levels - 1` = `gauss[k] - expand(gauss[k+1])`
    /// - `levels[n_levels - 1]` = `gauss[n_levels - 1]` (coarse residual)
    ///
    /// Per-level temp buffers are allocated per call (no scratch
    /// pool yet). Future ticks can extend `Cvvdp::new` to allocate
    /// these once.
    pub fn compute_dkl_laplacian_pyramid(&mut self, srgb: &[u8]) -> Result<Vec<[Vec<f32>; 3]>> {
        // Builds the Gaussian pyramid first (color → reduce chain).
        let _ = self.compute_dkl_gauss_pyramid(srgb)?;

        let cube_dim = CubeDim::new_1d(64);

        // Now produce Laplacian bands top-down. For each level k <
        // n_levels - 1: expand gauss[k+1] → temp, then subtract
        // (gauss[k] - temp) → bands_ref[k].
        for k in 0..(self.n_levels as usize - 1) {
            let coarse_w = self.gauss_ref[k + 1].w;
            let coarse_h = self.gauss_ref[k + 1].h;
            let fine_w = self.gauss_ref[k].w;
            let fine_h = self.gauss_ref[k].h;
            let n_v = (coarse_w * fine_h) as usize;
            let n_fine = (fine_w * fine_h) as usize;

            // Per-channel: upscale_v(coarse → vscratch), upscale_h(vscratch →
            // upscaled), subtract(fine, upscaled → band).
            for c in 0..N_CHANNELS {
                let coarse = self.gauss_ref[k + 1].planes[c].clone();
                let fine = self.gauss_ref[k].planes[c].clone();
                let band = self.bands_ref[k].planes[c].clone();

                let vscratch = alloc_zeros_f32(&self.client, n_v);
                let upscaled = alloc_zeros_f32(&self.client, n_fine);

                let count_v = CubeCount::Static((n_v as u32).div_ceil(64), 1, 1);
                let count_h = CubeCount::Static((n_fine as u32).div_ceil(64), 1, 1);
                let count_sub = CubeCount::Static((n_fine as u32).div_ceil(64), 1, 1);
                let n_coarse = (coarse_w * coarse_h) as usize;

                unsafe {
                    upscale_v_kernel::launch::<R>(
                        &self.client,
                        count_v,
                        cube_dim,
                        ArrayArg::from_raw_parts(coarse, n_coarse),
                        ArrayArg::from_raw_parts(vscratch.clone(), n_v),
                        coarse_w,
                        coarse_h,
                        fine_h,
                    );
                    upscale_h_kernel::launch::<R>(
                        &self.client,
                        count_h,
                        cube_dim,
                        ArrayArg::from_raw_parts(vscratch, n_v),
                        ArrayArg::from_raw_parts(upscaled.clone(), n_fine),
                        coarse_w,
                        fine_w,
                        fine_h,
                    );
                    subtract_kernel::launch::<R>(
                        &self.client,
                        count_sub,
                        cube_dim,
                        ArrayArg::from_raw_parts(fine, n_fine),
                        ArrayArg::from_raw_parts(upscaled, n_fine),
                        ArrayArg::from_raw_parts(band, n_fine),
                        n_fine as u32,
                    );
                }
            }
        }

        // Coarsest band = coarsest gauss (no subtraction). Read it
        // directly from gauss_ref. For symmetry with the rest, copy
        // into bands_ref[last] via a host trip — small buffer.
        let n_levels = self.n_levels as usize;
        let last = n_levels - 1;
        for c in 0..N_CHANNELS {
            let g = self.gauss_ref[last].planes[c].clone();
            let bytes = self
                .client
                .read_one(g)
                .map_err(|_| Error::InvalidImageSize)?;
            // Re-upload as bands_ref[last] so the read-back loop is
            // uniform across levels.
            self.bands_ref[last].planes[c] = self.client.create_from_slice(&bytes);
        }

        // Read back every band × every channel.
        let mut out: Vec<[Vec<f32>; 3]> = Vec::with_capacity(n_levels);
        for k in 0..n_levels {
            let mut planes = [Vec::new(), Vec::new(), Vec::new()];
            for c in 0..N_CHANNELS {
                let h = self.bands_ref[k].planes[c].clone();
                let bytes = self
                    .client
                    .read_one(h)
                    .map_err(|_| Error::InvalidImageSize)?;
                planes[c] = f32::from_bytes(&bytes).to_vec();
            }
            out.push(planes);
        }
        Ok(out)
    }

    /// Run color + Weber-contrast pyramid on GPU. Matches what
    /// `host_scalar::weber_contrast_pyr_dec_scalar` builds for each
    /// of the 3 DKL channels, using each image's own achromatic
    /// channel as the `L_bkg` source (cvvdp's `weber_g1` rule).
    ///
    /// For non-baseband levels `k < N-1`:
    /// 1. `layer_c = gauss_c[k] - expand(gauss_c[k+1])` per channel
    ///    (built via `upscale_v_kernel` + `upscale_h_kernel` +
    ///    `subtract_kernel`, sharing the expand of `gauss_A[k+1]`
    ///    across channels for the L_bkg pathway).
    /// 2. `L_bkg = expand(gauss_A[k+1])`, clamped to ≥ 0.01 inside
    ///    `weber_contrast_compute_kernel`.
    /// 3. `contrast_c = clamp(layer_c / L_bkg, ±1000)` and
    ///    `log_l_bkg = log10(L_bkg)` via
    ///    `weber_contrast_compute_kernel`.
    ///
    /// For the baseband level `k = N-1`, cvvdp uses a SCALAR mean
    /// of `max(gauss_A[N-1], 0.01)`. The mean is computed host-side
    /// from a read-back of the achromatic baseband (≤16 pixels at
    /// 1024² × 7 levels), then each channel's baseband is divided
    /// by that scalar host-side. Avoids a GPU reduction for tiny
    /// data; the per-pixel divide is also tiny.
    ///
    /// Returns `(bands, log_l_bkg)`:
    /// - `bands[k] = [a, rg, vy]` Weber-contrast planar f32 vecs,
    ///   matching the shape of `compute_dkl_laplacian_pyramid`.
    /// - `log_l_bkg[k]` is a per-pixel `log10(L_bkg)` plane for
    ///   non-baseband levels and a scalar (replicated 1×1) for the
    ///   baseband. Same shape convention as
    ///   `WeberPyramid::log_l_bkg` in host_scalar.
    ///
    /// GPU-only Weber pyramid dispatch. Writes:
    /// - `self.bands_ref[k].planes[c]` — Weber-contrast bands per
    ///   level per channel (non-baseband levels). Baseband level
    ///   gets the per-channel gauss[last] divided by the achromatic
    ///   baseband's scalar mean (host-side).
    /// - `self.weber_scratch[k].log_l_bkg` — per-pixel
    ///   `log10(L_bkg)` plane per non-baseband level.
    ///
    /// Returns the baseband's scalar `log10(L_bkg)` since that's
    /// computed host-side (small reduction over the achromatic
    /// baseband). Callers handle the per-pixel readback themselves
    /// — this function does NO readback of the per-level band /
    /// log_l_bkg data.
    ///
    /// Used by `compute_dkl_weber_pyramid` (which wraps with
    /// readback to host Vecs). A future GPU-resident
    /// `compute_dkl_d_bands` will call this with the stable
    /// `self.ref_log_l_bkg` destination so REF log_l_bkg survives
    /// the dist weber call — currently blocked on a separate
    /// regression in the d_bands rewrite (tick 85/86).
    ///
    /// `log_l_bkg_dest` must have length `n_levels - 1` (one handle
    /// per non-baseband level).
    fn _dispatch_weber_pyramid_gpu(
        &mut self,
        srgb: &[u8],
        log_l_bkg_dest: &[cubecl::server::Handle],
    ) -> Result<f32> {
        // Build Gaussian pyramids on GPU. The function leaves
        // self.gauss_ref[k].planes[c] populated for k = 0..n_levels.
        let _ = self.compute_dkl_gauss_pyramid(srgb)?;

        let cube_dim = CubeDim::new_1d(64);
        let n_levels = self.n_levels as usize;

        // Non-baseband levels: build layers + expanded L_bkg, then
        // launch weber_contrast_compute_kernel per channel.
        for k in 0..n_levels.saturating_sub(1) {
            let coarse_w = self.gauss_ref[k + 1].w;
            let coarse_h = self.gauss_ref[k + 1].h;
            let fine_w = self.gauss_ref[k].w;
            let fine_h = self.gauss_ref[k].h;
            let n_v = (coarse_w * fine_h) as usize;
            let n_fine = (fine_w * fine_h) as usize;
            let n_coarse = (coarse_w * coarse_h) as usize;

            let count_v = CubeCount::Static((n_v as u32).div_ceil(64), 1, 1);
            let count_fine = CubeCount::Static((n_fine as u32).div_ceil(64), 1, 1);

            // Pre-allocated per-level scratch (Cvvdp.weber_scratch).
            // Reuses the same handles across calls + across both sides
            // of compute_dkl_d_bands. Each call writes-then-reads-back
            // before the next call overwrites, so the read-back captures
            // the data correctly.
            let scratch = &self.weber_scratch[k];
            let l_bkg_fine = scratch.l_bkg_fine.clone();
            let vscratch_a = scratch.vscratch_a.clone();
            let coarse_a = self.gauss_ref[k + 1].planes[0].clone();
            unsafe {
                upscale_v_kernel::launch::<R>(
                    &self.client,
                    count_v.clone(),
                    cube_dim,
                    ArrayArg::from_raw_parts(coarse_a, n_coarse),
                    ArrayArg::from_raw_parts(vscratch_a.clone(), n_v),
                    coarse_w,
                    coarse_h,
                    fine_h,
                );
                upscale_h_kernel::launch::<R>(
                    &self.client,
                    count_fine.clone(),
                    cube_dim,
                    ArrayArg::from_raw_parts(vscratch_a, n_v),
                    ArrayArg::from_raw_parts(l_bkg_fine.clone(), n_fine),
                    coarse_w,
                    fine_w,
                    fine_h,
                );
            }

            // Per channel: upscale coarse → fine (separable v + h).
            // Subtract + Weber-contrast + log_l_bkg are fused into a
            // single 3-channel launch (tick 91) below — eliminates 3
            // subtract_kernel launches per level + the `layer_c`
            // intermediate Vec materialization step.
            let log_l_bkg = log_l_bkg_dest[k].clone();
            for c in 0..N_CHANNELS {
                let coarse = self.gauss_ref[k + 1].planes[c].clone();
                let vscratch_c = scratch.vscratch_c[c].clone();
                let upscaled_c = scratch.upscaled_c[c].clone();

                unsafe {
                    upscale_v_kernel::launch::<R>(
                        &self.client,
                        count_v.clone(),
                        cube_dim,
                        ArrayArg::from_raw_parts(coarse, n_coarse),
                        ArrayArg::from_raw_parts(vscratch_c.clone(), n_v),
                        coarse_w,
                        coarse_h,
                        fine_h,
                    );
                    upscale_h_kernel::launch::<R>(
                        &self.client,
                        count_fine.clone(),
                        cube_dim,
                        ArrayArg::from_raw_parts(vscratch_c, n_v),
                        ArrayArg::from_raw_parts(upscaled_c, n_fine),
                        coarse_w,
                        fine_w,
                        fine_h,
                    );
                }
            }
            // Fused subtract + 3-channel weber-contrast. One launch
            // does `band[c] = clamp((fine[c] - upscaled[c]) / L_bkg)`
            // for all three channels plus the shared log_l_bkg.
            let fine_a = self.gauss_ref[k].planes[0].clone();
            let fine_rg = self.gauss_ref[k].planes[1].clone();
            let fine_vy = self.gauss_ref[k].planes[2].clone();
            let upsc_a = scratch.upscaled_c[0].clone();
            let upsc_rg = scratch.upscaled_c[1].clone();
            let upsc_vy = scratch.upscaled_c[2].clone();
            let band_a = self.bands_ref[k].planes[0].clone();
            let band_rg = self.bands_ref[k].planes[1].clone();
            let band_vy = self.bands_ref[k].planes[2].clone();
            unsafe {
                subtract_weber_3ch_kernel::launch::<R>(
                    &self.client,
                    count_fine.clone(),
                    cube_dim,
                    ArrayArg::from_raw_parts(fine_a, n_fine),
                    ArrayArg::from_raw_parts(fine_rg, n_fine),
                    ArrayArg::from_raw_parts(fine_vy, n_fine),
                    ArrayArg::from_raw_parts(upsc_a, n_fine),
                    ArrayArg::from_raw_parts(upsc_rg, n_fine),
                    ArrayArg::from_raw_parts(upsc_vy, n_fine),
                    ArrayArg::from_raw_parts(l_bkg_fine, n_fine),
                    ArrayArg::from_raw_parts(band_a, n_fine),
                    ArrayArg::from_raw_parts(band_rg, n_fine),
                    ArrayArg::from_raw_parts(band_vy, n_fine),
                    ArrayArg::from_raw_parts(log_l_bkg, n_fine),
                    n_fine as u32,
                );
            }
        }

        // Baseband: scalar L_bkg = mean of max(gauss_A[N-1], 0.01).
        let last = n_levels - 1;
        let baseband_w = self.gauss_ref[last].w as usize;
        let baseband_h = self.gauss_ref[last].h as usize;
        let baseband_n = baseband_w * baseband_h;

        let gauss_a_last = self.gauss_ref[last].planes[0].clone();
        let bytes_a = self
            .client
            .read_one(gauss_a_last)
            .map_err(|_| Error::InvalidImageSize)?;
        let gauss_a_data: &[f32] = f32::from_bytes(&bytes_a);
        let l_bkg_sum: f32 = gauss_a_data.iter().map(|v| v.max(0.01)).sum();
        let l_bkg_mean = l_bkg_sum / baseband_n as f32;
        let log_l_bkg_baseband = l_bkg_mean.log10();

        // Per channel: copy gauss[last][c] into bands_ref[last] divided by mean.
        for c in 0..N_CHANNELS {
            let g = self.gauss_ref[last].planes[c].clone();
            let bytes = self
                .client
                .read_one(g)
                .map_err(|_| Error::InvalidImageSize)?;
            let data: &[f32] = f32::from_bytes(&bytes);
            let divided: Vec<f32> = data.iter().map(|v| v / l_bkg_mean).collect();
            self.bands_ref[last].planes[c] = self.client.create_from_slice(f32::as_bytes(&divided));
        }

        Ok(log_l_bkg_baseband)
    }

    pub fn compute_dkl_weber_pyramid(&mut self, srgb: &[u8]) -> Result<WeberPyramidGpu> {
        let trace_weber = std::env::var_os("CVVDP_TRACE_WEBER").is_some();
        let t_dispatch = std::time::Instant::now();

        // Build dests Vec (cloned from self.weber_scratch[*].log_l_bkg).
        let dests: Vec<cubecl::server::Handle> = self
            .weber_scratch
            .iter()
            .map(|s| s.log_l_bkg.clone())
            .collect();
        let log_l_bkg_baseband = self._dispatch_weber_pyramid_gpu(srgb, &dests)?;

        let n_levels = self.n_levels as usize;
        let last = n_levels - 1;
        let baseband_n = (self.gauss_ref[last].w as usize) * (self.gauss_ref[last].h as usize);

        if trace_weber {
            eprintln!(
                "[weber-trace] GPU dispatch + baseband host (before readback): {:?}",
                t_dispatch.elapsed()
            );
        }
        let t_readback = std::time::Instant::now();

        // Read back every band × every channel for return.
        let mut bands_out: Vec<[Vec<f32>; 3]> = Vec::with_capacity(n_levels);
        for k in 0..n_levels {
            let mut planes = [Vec::new(), Vec::new(), Vec::new()];
            for c in 0..N_CHANNELS {
                let h = self.bands_ref[k].planes[c].clone();
                let bytes = self
                    .client
                    .read_one(h)
                    .map_err(|_| Error::InvalidImageSize)?;
                planes[c] = f32::from_bytes(&bytes).to_vec();
            }
            bands_out.push(planes);
        }

        if trace_weber {
            eprintln!(
                "[weber-trace] bands readback ({n_levels} levels): {:?}",
                t_readback.elapsed()
            );
        }
        let t_log_readback = std::time::Instant::now();

        // Read back log_l_bkg per band: non-baseband from GPU
        // (reconstruct handle from self.weber_scratch[k].log_l_bkg
        // since _dispatch_weber_pyramid_gpu left the data there),
        // baseband as replicated scalar matching host_scalar's
        // WeberPyramid shape.
        let mut log_l_bkg_out: Vec<Vec<f32>> = Vec::with_capacity(n_levels);
        for k in 0..n_levels.saturating_sub(1) {
            let log_h = self.weber_scratch[k].log_l_bkg.clone();
            let bytes = self
                .client
                .read_one(log_h)
                .map_err(|_| Error::InvalidImageSize)?;
            log_l_bkg_out.push(f32::from_bytes(&bytes).to_vec());
        }
        log_l_bkg_out.push(vec![log_l_bkg_baseband; baseband_n]);

        if trace_weber {
            eprintln!(
                "[weber-trace] log_l_bkg readback: {:?}",
                t_log_readback.elapsed()
            );
        }

        Ok((bands_out, log_l_bkg_out))
    }

    /// Run color + Weber-contrast pyramid + per-pixel CSF apply on
    /// GPU. Returns the `T_p` bands that the masking stage consumes:
    ///
    /// ```text
    /// T_p[k][c][i] = band_mul[k] * weber[k][c][i] * S(rho_k, log_l_bkg[k][i], c) * CH_GAIN_eff
    /// ```
    ///
    /// where:
    /// - `band_mul = 1.0` for the first level (`k == 0`) and baseband
    ///   (`k == N-1`), `2.0` otherwise. Matches cvvdp's
    ///   `lpyr.get_band` ×2 band-readout gain on non-edge levels.
    /// - `S` is the per-pixel CSF sensitivity (with the
    ///   `sensitivity_correction` log offset baked in) from the
    ///   `csf_lut_weber_fixed_size` LUT. The kernel interpolates
    ///   `logs_row[rho_k, c]` along the per-pixel `log10(L_bkg)` axis.
    /// - `CH_GAIN_eff = CH_GAIN[c] = [1, 1.45, 1]` for non-baseband
    ///   levels. For the baseband, cvvdp's `apply_masking_model`
    ///   bypasses `CH_GAIN`, so this helper sets `CH_GAIN_eff = 1.0`
    ///   on the baseband — the caller can still subtract sides
    ///   directly to obtain the per-channel `D` (cvvdp's baseband
    ///   formula is `|T_p - R_p|` with `CH_GAIN` absorbed only in
    ///   `T_p` / `R_p` of non-baseband bands).
    ///
    /// `ppd` is pixels-per-degree (from `DisplayGeometry::pixels_per_degree()`).
    /// Each level's `rho_k` is resolved via
    /// [`crate::kernels::pyramid::band_frequencies`].
    ///
    /// Returns `levels[k] = [a, rg, vy]` planar f32 vecs, same shape
    /// as `compute_dkl_weber_pyramid`'s `.0`.
    pub fn compute_dkl_t_p_bands(&mut self, srgb: &[u8], ppd: f32) -> Result<Vec<[Vec<f32>; 3]>> {
        // Build Weber bands + log_l_bkg on GPU. Side effect leaves
        // weber bands resident in self.bands_ref and log_l_bkg as
        // host-side data.
        //
        // Tick 101: fused 3-channel CSF apply (was 3 per-channel
        // launches per level) AND read weber from `self.bands_ref`
        // handles directly (was re-uploading from the host Vec
        // returned by compute_dkl_weber_pyramid). Per non-baseband
        // level: 3 host uploads + 3 kernel launches → 0 uploads +
        // 1 launch.
        let (_weber_bands_unused, log_l_bkg) = self.compute_dkl_weber_pyramid(srgb)?;
        let n_levels = self.n_levels as usize;
        // ppd unused — logs_row is pre-uploaded against the geometry
        // baked into Cvvdp::new. compute_dkl_t_p_bands still takes
        // ppd in the signature for source-compatibility.
        let _ = ppd;

        let cube_dim = CubeDim::new_1d(64);

        let mut t_p_bands: Vec<[Vec<f32>; 3]> = Vec::with_capacity(n_levels);
        for k in 0..n_levels {
            let is_first = k == 0;
            let is_baseband = k == n_levels - 1;
            let band_mul: f32 = if is_first || is_baseband { 1.0 } else { 2.0 };
            let (_bw, _bh, n_px) = self.level_dims(k);
            debug_assert_eq!(log_l_bkg[k].len(), n_px);

            let log_l_bkg_h = self.client.create_from_slice(f32::as_bytes(&log_l_bkg[k]));
            let count = CubeCount::Static((n_px as u32).div_ceil(64), 1, 1);

            let ch_gain_a = if is_baseband { 1.0 } else { band_mul * CH_GAIN[0] };
            let ch_gain_rg = if is_baseband { 1.0 } else { band_mul * CH_GAIN[1] };
            let ch_gain_vy = if is_baseband { 1.0 } else { band_mul * CH_GAIN[2] };

            let t_p_a_h = alloc_zeros_f32(&self.client, n_px);
            let t_p_rg_h = alloc_zeros_f32(&self.client, n_px);
            let t_p_vy_h = alloc_zeros_f32(&self.client, n_px);

            unsafe {
                csf_apply_3ch_kernel::launch::<R>(
                    &self.client,
                    count.clone(),
                    cube_dim,
                    ArrayArg::from_raw_parts(self.bands_ref[k].planes[0].clone(), n_px),
                    ArrayArg::from_raw_parts(self.bands_ref[k].planes[1].clone(), n_px),
                    ArrayArg::from_raw_parts(self.bands_ref[k].planes[2].clone(), n_px),
                    ArrayArg::from_raw_parts(log_l_bkg_h, n_px),
                    ArrayArg::from_raw_parts(self.logs_row[k][0].clone(), 32),
                    ArrayArg::from_raw_parts(self.logs_row[k][1].clone(), 32),
                    ArrayArg::from_raw_parts(self.logs_row[k][2].clone(), 32),
                    ArrayArg::from_raw_parts(t_p_a_h.clone(), n_px),
                    ArrayArg::from_raw_parts(t_p_rg_h.clone(), n_px),
                    ArrayArg::from_raw_parts(t_p_vy_h.clone(), n_px),
                    ch_gain_a,
                    ch_gain_rg,
                    ch_gain_vy,
                    n_px as u32,
                );
            }

            let mut planes = [Vec::new(), Vec::new(), Vec::new()];
            for (c, h) in [t_p_a_h, t_p_rg_h, t_p_vy_h].into_iter().enumerate() {
                let bytes = self
                    .client
                    .read_one(h)
                    .map_err(|_| Error::InvalidImageSize)?;
                planes[c] = f32::from_bytes(&bytes).to_vec();
            }
            t_p_bands.push(planes);
        }

        Ok(t_p_bands)
    }

    /// Run the full per-band D dispatch and leave each level's per-
    /// channel D plane in `self.d_scratch[k].d[c]`.
    ///
    /// Pipeline (all GPU after tick 96):
    /// - Color: sRGB → DKL (cached source bytes).
    /// - Weber pyramid: per-level upscale + fused
    ///   `subtract_weber_3ch_kernel` writing all three channels +
    ///   shared `log_l_bkg` in one launch.
    /// - Per-pixel CSF: `csf_apply_6ch_kernel` runs REF and DIST in
    ///   a single launch per non-baseband level (the LUT bracket
    ///   math is shared across all 6 outputs). Per cvvdp's
    ///   `weber_g1` contract, REF's `log10(L_bkg)` is used for
    ///   both sides.
    /// - Masking:
    ///   - Non-baseband bands: `min_abs_3ch_kernel →
    ///     pu_blur_h_3ch_kernel → pu_blur_v_3ch_scaled_kernel
    ///     (folds `* 10^MASK_C`) → mult_mutual_3ch_with_blurred_kernel`
    ///     (or `mult_mutual_3ch_no_blur_kernel` when `bw ≤ PU_PADSIZE`
    ///     or `bh ≤ PU_PADSIZE`).
    ///   - Baseband: `diff_abs_3ch_kernel` writes `|T_p_dis - T_p_ref|`
    ///     for all three channels in one launch (since tick 94 every
    ///     level's D plane lives in the same `d_scratch.d[k][c]` slot).
    ///
    /// No GPU→host readback inside this helper. Callers that want
    /// the host-side `Vec<[Vec<f32>; 3]>` snapshot use
    /// [`Cvvdp::compute_dkl_d_bands`]; callers that pool on GPU
    /// (`Cvvdp::compute_dkl_jod`) read straight from the resident
    /// handles via `pool_band_kernel`.
    fn _dispatch_d_bands_into_scratch(
        &mut self,
        ref_srgb: &[u8],
        dist_srgb: &[u8],
        ppd: f32,
    ) -> Result<()> {
        // CVVDP_TRACE=1 enables per-phase eprintln timings so we can
        // see where the dispatch spends its time without committing
        // instrumentation. Zero cost when unset.
        let trace = std::env::var_os("CVVDP_TRACE").is_some();

        let t_weber_ref = std::time::Instant::now();
        let (ref_weber, ref_log_l_bkg) = self.compute_dkl_weber_pyramid(ref_srgb)?;
        if trace {
            eprintln!("[trace] weber(ref):  {:?}", t_weber_ref.elapsed());
        }
        // Discard dist log_l_bkg — cvvdp's weber_g1 uses ref's log_l_bkg.
        let t_weber_dis = std::time::Instant::now();
        let (dist_weber, _) = self.compute_dkl_weber_pyramid(dist_srgb)?;
        if trace {
            eprintln!("[trace] weber(dist): {:?}", t_weber_dis.elapsed());
        }

        let n_levels = self.n_levels as usize;
        // ppd unused — logs_row is pre-uploaded against the geometry
        // baked into Cvvdp::new. compute_dkl_d_bands keeps ppd in the
        // signature for source-compatibility.
        let _ = ppd;
        let cube_dim = CubeDim::new_1d(64);
        let _csf_channels = [CsfChannel::A, CsfChannel::Rg, CsfChannel::Vy];

        let t_band_loop = std::time::Instant::now();
        for k in 0..n_levels {
            let is_first = k == 0;
            let is_baseband = k == n_levels - 1;
            let band_mul: f32 = if is_first || is_baseband { 1.0 } else { 2.0 };
            let (bw, bh, n_px) = self.level_dims(k);
            debug_assert_eq!(ref_weber[k][0].len(), n_px);
            debug_assert_eq!(ref_log_l_bkg[k].len(), n_px);

            let t_band = std::time::Instant::now();

            // Upload ref log_l_bkg once per band; both sides reuse it.
            let t_log_upload = std::time::Instant::now();
            let log_l_bkg_h = self
                .client
                .create_from_slice(f32::as_bytes(&ref_log_l_bkg[k]));
            if trace {
                eprintln!(
                    "[trace] L{k} log_l_bkg upload ({bw}×{bh}): {:?}",
                    t_log_upload.elapsed()
                );
            }
            let count = CubeCount::Static((n_px as u32).div_ceil(64), 1, 1);

            // Reuse the pre-allocated per-level scratch (Cvvdp.d_scratch).
            // T_p / m_* / d handles are kept resident so the masking kernels
            // can consume them without a round-trip to host AND without
            // per-band alloc_zeros_f32 churn (~1.5 GB worth at 12 MP).
            let scratch = &self.d_scratch[k];
            let t_p_ref_h: [cubecl::server::Handle; 3] = [
                scratch.t_p_ref[0].clone(),
                scratch.t_p_ref[1].clone(),
                scratch.t_p_ref[2].clone(),
            ];
            let t_p_dis_h: [cubecl::server::Handle; 3] = [
                scratch.t_p_dis[0].clone(),
                scratch.t_p_dis[1].clone(),
                scratch.t_p_dis[2].clone(),
            ];

            let t_csf = std::time::Instant::now();

            // Fused 3-channel CSF apply — one launch per side instead
            // of three. The per-pixel LUT bracket math is shared across
            // the A/RG/VY channels.
            let ch_gain_a: f32 = if is_baseband { 1.0 } else { band_mul * CH_GAIN[0] };
            let ch_gain_rg: f32 = if is_baseband { 1.0 } else { band_mul * CH_GAIN[1] };
            let ch_gain_vy: f32 = if is_baseband { 1.0 } else { band_mul * CH_GAIN[2] };

            // Fused 6-channel CSF apply: one launch runs both sides
            // (REF + DIST) and shares the per-pixel LUT bracket math.
            // REF weber comes from the host upload (bands_ref was
            // overwritten with DIST by the weber(dist) call earlier
            // in the band loop). DIST weber reads self.bands_ref
            // handles directly. Tick 93 — was 2 separate
            // csf_apply_3ch launches.
            {
                let weber_ref_a_h = self
                    .client
                    .create_from_slice(f32::as_bytes(&ref_weber[k][0]));
                let weber_ref_rg_h = self
                    .client
                    .create_from_slice(f32::as_bytes(&ref_weber[k][1]));
                let weber_ref_vy_h = self
                    .client
                    .create_from_slice(f32::as_bytes(&ref_weber[k][2]));
                let _ = &dist_weber; // dist host upload path retired
                unsafe {
                    csf_apply_6ch_kernel::launch::<R>(
                        &self.client,
                        count.clone(),
                        cube_dim,
                        ArrayArg::from_raw_parts(weber_ref_a_h, n_px),
                        ArrayArg::from_raw_parts(weber_ref_rg_h, n_px),
                        ArrayArg::from_raw_parts(weber_ref_vy_h, n_px),
                        ArrayArg::from_raw_parts(self.bands_ref[k].planes[0].clone(), n_px),
                        ArrayArg::from_raw_parts(self.bands_ref[k].planes[1].clone(), n_px),
                        ArrayArg::from_raw_parts(self.bands_ref[k].planes[2].clone(), n_px),
                        ArrayArg::from_raw_parts(log_l_bkg_h.clone(), n_px),
                        ArrayArg::from_raw_parts(self.logs_row[k][0].clone(), 32),
                        ArrayArg::from_raw_parts(self.logs_row[k][1].clone(), 32),
                        ArrayArg::from_raw_parts(self.logs_row[k][2].clone(), 32),
                        ArrayArg::from_raw_parts(t_p_ref_h[0].clone(), n_px),
                        ArrayArg::from_raw_parts(t_p_ref_h[1].clone(), n_px),
                        ArrayArg::from_raw_parts(t_p_ref_h[2].clone(), n_px),
                        ArrayArg::from_raw_parts(t_p_dis_h[0].clone(), n_px),
                        ArrayArg::from_raw_parts(t_p_dis_h[1].clone(), n_px),
                        ArrayArg::from_raw_parts(t_p_dis_h[2].clone(), n_px),
                        ch_gain_a,
                        ch_gain_rg,
                        ch_gain_vy,
                        n_px as u32,
                    );
                }
            }
            if trace {
                eprintln!("[trace] L{k} csf 2 launches:        {:?}", t_csf.elapsed());
            }

            let t_mask = std::time::Instant::now();
            if is_baseband {
                // Baseband: cvvdp's `|T_p_dis - T_p_ref|` bypass. Tick
                // 94 — GPU fused 3-channel diff into scratch.d so the
                // baseband output lives in d_scratch.d[k][c] like every
                // other level (prep for GPU pool in tick 95).
                let d_h: [cubecl::server::Handle; 3] = [
                    scratch.d[0].clone(),
                    scratch.d[1].clone(),
                    scratch.d[2].clone(),
                ];
                unsafe {
                    diff_abs_3ch_kernel::launch::<R>(
                        &self.client,
                        count.clone(),
                        cube_dim,
                        ArrayArg::from_raw_parts(t_p_dis_h[0].clone(), n_px),
                        ArrayArg::from_raw_parts(t_p_dis_h[1].clone(), n_px),
                        ArrayArg::from_raw_parts(t_p_dis_h[2].clone(), n_px),
                        ArrayArg::from_raw_parts(t_p_ref_h[0].clone(), n_px),
                        ArrayArg::from_raw_parts(t_p_ref_h[1].clone(), n_px),
                        ArrayArg::from_raw_parts(t_p_ref_h[2].clone(), n_px),
                        ArrayArg::from_raw_parts(d_h[0].clone(), n_px),
                        ArrayArg::from_raw_parts(d_h[1].clone(), n_px),
                        ArrayArg::from_raw_parts(d_h[2].clone(), n_px),
                        n_px as u32,
                    );
                }
                let _ = d_h; // baseband result lives in d_scratch[k].d[c]
            } else {
                // GPU masking. D output + masking-chain scratch all
                // come from the pre-allocated d_scratch[k] (no
                // per-band alloc_zeros_f32 churn).
                let d_h: [cubecl::server::Handle; 3] = [
                    scratch.d[0].clone(),
                    scratch.d[1].clone(),
                    scratch.d[2].clone(),
                ];
                let use_blur = bw > PU_PADSIZE && bh > PU_PADSIZE;
                unsafe {
                    if use_blur {
                        // min_abs → pu_blur_h → pu_blur_v → mult_mutual_3ch_with_blurred.
                        let m_raw_h: [cubecl::server::Handle; 3] = [
                            scratch.m_raw[0].clone(),
                            scratch.m_raw[1].clone(),
                            scratch.m_raw[2].clone(),
                        ];
                        min_abs_3ch_kernel::launch::<R>(
                            &self.client,
                            count.clone(),
                            cube_dim,
                            ArrayArg::from_raw_parts(t_p_dis_h[0].clone(), n_px),
                            ArrayArg::from_raw_parts(t_p_dis_h[1].clone(), n_px),
                            ArrayArg::from_raw_parts(t_p_dis_h[2].clone(), n_px),
                            ArrayArg::from_raw_parts(t_p_ref_h[0].clone(), n_px),
                            ArrayArg::from_raw_parts(t_p_ref_h[1].clone(), n_px),
                            ArrayArg::from_raw_parts(t_p_ref_h[2].clone(), n_px),
                            ArrayArg::from_raw_parts(m_raw_h[0].clone(), n_px),
                            ArrayArg::from_raw_parts(m_raw_h[1].clone(), n_px),
                            ArrayArg::from_raw_parts(m_raw_h[2].clone(), n_px),
                            n_px as u32,
                        );
                        // PU blur: 3-channel h pass → 3-channel v pass
                        // with `* 10^MASK_C` post-scale folded in (tick
                        // 92). One launch each = 2 launches total
                        // instead of 9 (3× pu_blur_h + 3× pu_blur_v +
                        // 3× weight_band).
                        let m_mid_h: [cubecl::server::Handle; 3] = [
                            scratch.m_mid[0].clone(),
                            scratch.m_mid[1].clone(),
                            scratch.m_mid[2].clone(),
                        ];
                        let m_blur_h: [cubecl::server::Handle; 3] = [
                            scratch.m_blur[0].clone(),
                            scratch.m_blur[1].clone(),
                            scratch.m_blur[2].clone(),
                        ];
                        let pu_scale = 10.0_f32.powf(MASK_C);
                        pu_blur_h_3ch_kernel::launch::<R>(
                            &self.client,
                            count.clone(),
                            cube_dim,
                            ArrayArg::from_raw_parts(m_raw_h[0].clone(), n_px),
                            ArrayArg::from_raw_parts(m_raw_h[1].clone(), n_px),
                            ArrayArg::from_raw_parts(m_raw_h[2].clone(), n_px),
                            ArrayArg::from_raw_parts(m_mid_h[0].clone(), n_px),
                            ArrayArg::from_raw_parts(m_mid_h[1].clone(), n_px),
                            ArrayArg::from_raw_parts(m_mid_h[2].clone(), n_px),
                            bw as u32,
                            bh as u32,
                        );
                        pu_blur_v_3ch_scaled_kernel::launch::<R>(
                            &self.client,
                            count.clone(),
                            cube_dim,
                            ArrayArg::from_raw_parts(m_mid_h[0].clone(), n_px),
                            ArrayArg::from_raw_parts(m_mid_h[1].clone(), n_px),
                            ArrayArg::from_raw_parts(m_mid_h[2].clone(), n_px),
                            ArrayArg::from_raw_parts(m_blur_h[0].clone(), n_px),
                            ArrayArg::from_raw_parts(m_blur_h[1].clone(), n_px),
                            ArrayArg::from_raw_parts(m_blur_h[2].clone(), n_px),
                            pu_scale,
                            bw as u32,
                            bh as u32,
                        );
                        mult_mutual_3ch_with_blurred_kernel::launch::<R>(
                            &self.client,
                            count.clone(),
                            cube_dim,
                            ArrayArg::from_raw_parts(t_p_dis_h[0].clone(), n_px),
                            ArrayArg::from_raw_parts(t_p_dis_h[1].clone(), n_px),
                            ArrayArg::from_raw_parts(t_p_dis_h[2].clone(), n_px),
                            ArrayArg::from_raw_parts(t_p_ref_h[0].clone(), n_px),
                            ArrayArg::from_raw_parts(t_p_ref_h[1].clone(), n_px),
                            ArrayArg::from_raw_parts(t_p_ref_h[2].clone(), n_px),
                            ArrayArg::from_raw_parts(m_blur_h[0].clone(), n_px),
                            ArrayArg::from_raw_parts(m_blur_h[1].clone(), n_px),
                            ArrayArg::from_raw_parts(m_blur_h[2].clone(), n_px),
                            ArrayArg::from_raw_parts(d_h[0].clone(), n_px),
                            ArrayArg::from_raw_parts(d_h[1].clone(), n_px),
                            ArrayArg::from_raw_parts(d_h[2].clone(), n_px),
                            n_px as u32,
                        );
                    } else {
                        // Small band: inline no-blur masker (band ≤ PU_PADSIZE).
                        mult_mutual_3ch_no_blur_kernel::launch::<R>(
                            &self.client,
                            count.clone(),
                            cube_dim,
                            ArrayArg::from_raw_parts(t_p_dis_h[0].clone(), n_px),
                            ArrayArg::from_raw_parts(t_p_dis_h[1].clone(), n_px),
                            ArrayArg::from_raw_parts(t_p_dis_h[2].clone(), n_px),
                            ArrayArg::from_raw_parts(t_p_ref_h[0].clone(), n_px),
                            ArrayArg::from_raw_parts(t_p_ref_h[1].clone(), n_px),
                            ArrayArg::from_raw_parts(t_p_ref_h[2].clone(), n_px),
                            ArrayArg::from_raw_parts(d_h[0].clone(), n_px),
                            ArrayArg::from_raw_parts(d_h[1].clone(), n_px),
                            ArrayArg::from_raw_parts(d_h[2].clone(), n_px),
                            n_px as u32,
                        );
                    }
                }
                let _ = d_h; // non-baseband result lives in d_scratch[k].d[c]
            }
            if trace {
                eprintln!(
                    "[trace] L{k} mask:                  {:?}   (band total: {:?})",
                    t_mask.elapsed(),
                    t_band.elapsed()
                );
            }
        }
        if trace {
            eprintln!(
                "[trace] band loop total ({n_levels} levels): {:?}",
                t_band_loop.elapsed()
            );
        }

        Ok(())
    }

    /// Host-side readback wrapper around the GPU D-bands dispatch.
    /// Runs the full GPU dispatch (color → weber → CSF → masking)
    /// into `self.d_scratch[k].d[c]` then copies each band's D plane
    /// out into a `Vec<[Vec<f32>; 3]>`. Use this when you need the
    /// raw band values (parity checks, debugging, downstream host
    /// scalar processing); use [`Cvvdp::compute_dkl_jod`] directly
    /// when you want the JOD scalar — that path pools on GPU and
    /// avoids the full ~432 MB per-band readback at 12 MP.
    pub fn compute_dkl_d_bands(
        &mut self,
        ref_srgb: &[u8],
        dist_srgb: &[u8],
        ppd: f32,
    ) -> Result<Vec<[Vec<f32>; 3]>> {
        self._dispatch_d_bands_into_scratch(ref_srgb, dist_srgb, ppd)?;

        let n_levels = self.n_levels as usize;
        let mut d_bands: Vec<[Vec<f32>; 3]> = Vec::with_capacity(n_levels);
        for k in 0..n_levels {
            let (_bw, _bh, n_px) = self.level_dims(k);
            let mut planes: [Vec<f32>; 3] =
                [vec![0.0; n_px], vec![0.0; n_px], vec![0.0; n_px]];
            for c in 0..N_CHANNELS {
                let bytes = self
                    .client
                    .read_one(self.d_scratch[k].d[c].clone())
                    .map_err(|_| Error::InvalidImageSize)?;
                planes[c] = f32::from_bytes(&bytes).to_vec();
            }
            d_bands.push(planes);
        }
        Ok(d_bands)
    }

    /// Final JOD for a (reference, distorted) sRGB pair, computed
    /// through the full GPU composition:
    ///
    /// ```text
    /// sRGB → DKL (GPU)
    ///      → Weber pyramid (GPU, fused subtract+weber 3ch per level)
    ///      → per-pixel CSF apply (GPU, fused REF+DIST 6ch per level)
    ///      → mult-mutual masking (GPU, fused min_abs + pu_blur 3ch +
    ///        mult_mutual_3ch_with_blurred per level — baseband uses
    ///        diff_abs_3ch)
    ///      → spatial pool (GPU, pool_band_kernel per (band, channel),
    ///        atomic-f32 accumulation into a partials Vec)
    ///      → 3-stage Minkowski fold + met2jod (host scalar — operates
    ///        on the `n_levels × N_CHANNELS` partials Vec, ~144 bytes
    ///        total, sub-microsecond regardless of image size).
    /// ```
    ///
    /// Only the GPU→host readback of the partials Vec touches host
    /// memory in proportion to anything other than the pyramid depth
    /// — and that readback is tiny (≤ 36 floats for typical 4K
    /// imagery). The full per-band D Vec readback was removed in
    /// tick 96; callers that still want the host-side band Vecs use
    /// [`Cvvdp::compute_dkl_d_bands`] directly.
    ///
    /// Returns JOD on cvvdp's 0–10 scale (10 = imperceptible).
    ///
    /// The shadow_jod test still pins the public `Cvvdp::score`
    /// path through `host_scalar::predict_jod_still_3ch` against
    /// the v1 R2 manifest (≤ 0.006 JOD). This helper exposes the
    /// GPU-composed path so its parity vs the host scalar can be
    /// measured independently — see
    /// `tests/pipeline_color.rs::compute_dkl_jod_matches_host_scalar`,
    /// `tests/pipeline_score.rs::compute_dkl_jod_on_v1_manifest_corpus`,
    /// and the drift sweep `compute_dkl_jod_vs_host_scalar_on_corpus`.
    /// Once the GPU JOD parity vs the host scalar is locked at
    /// f32-precision tolerance, `Cvvdp::score` will switch to this
    /// helper and the manifest-parity test will retarget.
    pub fn compute_dkl_jod(&mut self, ref_srgb: &[u8], dist_srgb: &[u8], ppd: f32) -> Result<f32> {
        // Run the full D-bands dispatch. compute_dkl_d_bands leaves the
        // per-band D planes resident in self.d_scratch[k].d[c] — those
        // handles persist across the call (the returned Vec is just a
        // host-readback snapshot of those handles). After tick 94 the
        // baseband path also writes through diff_abs_3ch into
        // d_scratch.d, so every level's D plane lives in the same slot.
        //
        // Tick 95 plumbed `pool_band_kernel` into this function; tick
        // 96 dropped the per-band host readback by routing through
        // the new `_dispatch_d_bands_into_scratch` helper. The full
        // ~432 MB per-band readback at 12 MP is gone — only the
        // `n_levels * N_CHANNELS` partials Vec comes back to host.
        self._dispatch_d_bands_into_scratch(ref_srgb, dist_srgb, ppd)?;
        let n_levels = self.n_levels as usize;

        let partials_init = vec![0.0_f32; n_levels * N_CHANNELS];
        let partials_h = self
            .client
            .create_from_slice(f32::as_bytes(&partials_init));
        let cube_dim = CubeDim::new_1d(64);
        for k in 0..n_levels {
            let (_bw, _bh, n_px) = self.level_dims(k);
            let count = CubeCount::Static((n_px as u32).div_ceil(64), 1, 1);
            for c in 0..N_CHANNELS {
                let partial_idx = (k * N_CHANNELS + c) as u32;
                let d_handle = self.d_scratch[k].d[c].clone();
                unsafe {
                    pool_band_kernel::launch::<R>(
                        &self.client,
                        count.clone(),
                        cube_dim,
                        ArrayArg::from_raw_parts(d_handle, n_px),
                        ArrayArg::from_raw_parts(partials_h.clone(), n_levels * N_CHANNELS),
                        BETA_SPATIAL,
                        partial_idx,
                        n_px as u32,
                    );
                }
            }
        }

        let bytes = self
            .client
            .read_one(partials_h)
            .map_err(|_| Error::InvalidImageSize)?;
        let partials_data: &[f32] = f32::from_bytes(&bytes);

        let mut q_per_ch: Vec<[f32; 3]> = Vec::with_capacity(n_levels);
        for k in 0..n_levels {
            let (_bw, _bh, n_px_k) = self.level_dims(k);
            let mut q = [0.0_f32; 3];
            for c in 0..N_CHANNELS {
                q[c] = pool_band_finalize(
                    partials_data[k * N_CHANNELS + c],
                    n_px_k,
                    BETA_SPATIAL,
                );
            }
            q_per_ch.push(q);
        }

        Ok(do_pooling_and_jod_still_3ch(&q_per_ch))
    }

    /// Run color + Laplacian-pyramid + per-band CSF weighting.
    ///
    /// `ppd` is pixels-per-degree (from `DisplayGeometry::pixels_per_degree()`).
    /// `l_bkg` is the scalar background-luminance approximation used
    /// for every pyramid band — typically a per-image mean or
    /// display-peak / 2. The per-pixel L_bkg form (cvvdp's exact
    /// behaviour) lands once we wire the achromatic `gauss\[1\]`
    /// read path into the kernel.
    ///
    /// Returns the same shape as `compute_dkl_laplacian_pyramid`:
    /// `levels[k] = [a, rg, vy]` planar f32 vecs, with each pixel
    /// already multiplied by `sensitivity_corrected_scalar(rho_k,
    /// l_bkg, channel)`.
    pub fn compute_dkl_csf_weighted_bands(
        &mut self,
        srgb: &[u8],
        ppd: f32,
        l_bkg: f32,
    ) -> Result<Vec<[Vec<f32>; 3]>> {
        // Side effect: leaves the un-weighted Laplacian bands in
        // self.bands_ref[k].planes[c].
        let _ = self.compute_dkl_laplacian_pyramid(srgb)?;

        let weights_per_level =
            precomputed_band_weights(ppd, self.width as usize, self.height as usize, l_bkg);
        let flat_weights = flatten_band_weights(&weights_per_level);
        let weights_handle = self.client.create_from_slice(f32::as_bytes(&flat_weights));

        let cube_dim = CubeDim::new_1d(64);
        let n_levels = self.n_levels as usize;
        for k in 0..n_levels {
            let n_px = (self.bands_ref[k].w * self.bands_ref[k].h) as usize;
            let cube_count = CubeCount::Static((n_px as u32).div_ceil(64), 1, 1);
            for c in 0..N_CHANNELS {
                let weight_idx = (k * N_CHANNELS + c) as u32;
                let band = self.bands_ref[k].planes[c].clone();
                unsafe {
                    weight_band_kernel::launch::<R>(
                        &self.client,
                        cube_count.clone(),
                        cube_dim,
                        ArrayArg::from_raw_parts(band, n_px),
                        ArrayArg::from_raw_parts(weights_handle.clone(), flat_weights.len()),
                        weight_idx,
                        n_px as u32,
                    );
                }
            }
        }

        // Read back every band × every channel.
        let mut out: Vec<[Vec<f32>; 3]> = Vec::with_capacity(n_levels);
        for k in 0..n_levels {
            let mut planes = [Vec::new(), Vec::new(), Vec::new()];
            for c in 0..N_CHANNELS {
                let h = self.bands_ref[k].planes[c].clone();
                let bytes = self
                    .client
                    .read_one(h)
                    .map_err(|_| Error::InvalidImageSize)?;
                planes[c] = f32::from_bytes(&bytes).to_vec();
            }
            out.push(planes);
        }
        Ok(out)
    }

    /// Score a (reference, distorted) sRGB pair, returning JOD on
    /// the cvvdp scale (0–10; 10 = imperceptible).
    ///
    /// Currently routes through the parity-locked host scalar
    /// (`host_scalar::predict_jod_still_3ch`). The full GPU
    /// composition path is implemented and parity-tested as
    /// [`Cvvdp::compute_dkl_jod`] (color → pyramid → CSF → masking →
    /// `pool_band_kernel` → host fold); `score` will retarget once
    /// the v1 R2 manifest parity is held by the GPU path through a
    /// `shadow_jod`-style anchor.
    ///
    /// Score matches pycvvdp v0.5.4 on the v1 R2 manifest within
    /// 0.006 JOD across q1–q90.
    ///
    /// The viewing geometry comes from `self.geometry` — set via
    /// `Cvvdp::new_with_geometry` or defaulted to STANDARD_4K by
    /// `Cvvdp::new`.
    pub fn score(&mut self, reference_srgb: &[u8], distorted_srgb: &[u8]) -> Result<f64> {
        let expected = (self.width as usize) * (self.height as usize) * 3;
        if reference_srgb.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: reference_srgb.len(),
            });
        }
        if distorted_srgb.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: distorted_srgb.len(),
            });
        }
        let ppd = self.geometry.pixels_per_degree();
        let jod = crate::host_scalar::predict_jod_still_3ch(
            reference_srgb,
            distorted_srgb,
            self.width as usize,
            self.height as usize,
            self.params.display,
            ppd,
        );
        Ok(jod as f64)
    }

    /// Cache the reference side for repeated scoring against many
    /// distorted candidates.
    ///
    /// Today this just stashes the sRGB bytes (the host-scalar path
    /// re-runs the reference side per call); the planned GPU
    /// composition will materialise the CSF-weighted pyramid here so
    /// the reference work happens once per `set_reference`.
    pub fn set_reference(&mut self, reference_srgb: &[u8]) -> Result<()> {
        let expected = (self.width as usize) * (self.height as usize) * 3;
        if reference_srgb.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: reference_srgb.len(),
            });
        }
        self.cached = Some(CachedReference {
            ref_srgb: reference_srgb.to_vec(),
        });
        Ok(())
    }

    /// Score a distorted candidate against the cached reference.
    /// Matches `score(ref, dist)` exactly — the fast path lands when
    /// GPU composition stops re-running the reference side.
    pub fn score_with_reference(&mut self, distorted_srgb: &[u8]) -> Result<f64> {
        let cached = self.cached.as_ref().ok_or(Error::NoCachedReference)?;
        let expected = (self.width as usize) * (self.height as usize) * 3;
        if distorted_srgb.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: distorted_srgb.len(),
            });
        }
        let ppd = self.geometry.pixels_per_degree();
        let jod = crate::host_scalar::predict_jod_still_3ch(
            &cached.ref_srgb,
            distorted_srgb,
            self.width as usize,
            self.height as usize,
            self.params.display,
            ppd,
        );
        Ok(jod as f64)
    }
}
