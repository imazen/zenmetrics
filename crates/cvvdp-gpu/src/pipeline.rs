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
    CsfChannel, csf_apply_per_pixel_kernel, flatten_band_weights, precompute_logs_row,
    precomputed_band_weights, weight_band_kernel,
};
use crate::kernels::masking::{CH_GAIN, mult_mutual_band};
use crate::kernels::pyramid::{
    band_frequencies, downscale_kernel, subtract_kernel, upscale_h_kernel, upscale_v_kernel,
    weber_contrast_compute_kernel,
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
#[allow(dead_code)]
struct Level {
    w: u32,
    h: u32,
    /// One f32 plane per DKL channel.
    planes: [cubecl::server::Handle; N_CHANNELS],
}

fn alloc_zeros_f32<R: Runtime>(client: &ComputeClient<R>, n: usize) -> cubecl::server::Handle {
    client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]))
}

/// Reference-side state kept across `score_with_reference` calls.
///
/// Today this just stashes the raw sRGB bytes so the host-scalar
/// pipeline can re-run end-to-end per distorted candidate — matches
/// what `score()` does internally, just without re-uploading the
/// reference on every call. Once the GPU composition path lands, this
/// will hold the CSF-weighted pyramid bands (`Vec<Vec<Handle>>`,
/// indexed `[level][channel]`) so the reference side of the pipeline
/// runs once and is reused for every distorted candidate.
struct CachedReference {
    /// Cached reference sRGB bytes (length `width * height * 3`).
    ref_srgb: Vec<u8>,
}

/// ColorVideoVDP scorer.
///
/// Allocates GPU buffers up front for a fixed image size and reuses
/// them across calls. To score images of a different size, construct
/// a new `Cvvdp`.
#[allow(dead_code)]
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

    /// sRGB byte upload scratch (one per side).
    src_ref: cubecl::server::Handle,
    src_dis: cubecl::server::Handle,

    /// 256-entry sRGB→linear LUT, uploaded once.
    srgb_lut: cubecl::server::Handle,

    /// Gaussian pyramids per side. Indexed `[level].planes[channel]`.
    gauss_ref: Vec<Level>,
    gauss_dis: Vec<Level>,

    /// Laplacian-band buffers per side. Indexed `[level].planes[channel]`.
    /// The coarsest level shares storage with the coarsest gaussian.
    bands_ref: Vec<Level>,
    bands_dis: Vec<Level>,

    /// Per-band f32 Minkowski partials, length `n_levels × N_CHANNELS`.
    pool_partials: cubecl::server::Handle,

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
        let src_dis = client.create_from_slice(u32::as_bytes(&vec![0u32; n0 * 3]));
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
        let gauss_dis = build_pyramid(&client);
        let bands_ref = build_pyramid(&client);
        let bands_dis = build_pyramid(&client);

        let pool_n = (n_levels as usize) * N_CHANNELS;
        let pool_partials = client.create_from_slice(f32::as_bytes(&vec![0.0f32; pool_n]));

        Ok(Self {
            client,
            params,
            geometry,
            width,
            height,
            n_levels,
            src_ref,
            src_dis,
            srgb_lut,
            gauss_ref,
            gauss_dis,
            bands_ref,
            bands_dis,
            pool_partials,
            cached: None,
        })
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
    pub fn compute_dkl_weber_pyramid(&mut self, srgb: &[u8]) -> Result<WeberPyramidGpu> {
        // Build Gaussian pyramids on GPU. The function leaves
        // self.gauss_ref[k].planes[c] populated for k = 0..n_levels.
        let _ = self.compute_dkl_gauss_pyramid(srgb)?;

        let cube_dim = CubeDim::new_1d(64);
        let n_levels = self.n_levels as usize;

        // Non-baseband levels: build layers + expanded L_bkg, then
        // launch weber_contrast_compute_kernel per channel.
        let mut log_l_bkg_handles: Vec<cubecl::server::Handle> = Vec::with_capacity(n_levels);
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

            // Expand achromatic gauss[k+1] once → l_bkg_fine.
            let l_bkg_fine = alloc_zeros_f32(&self.client, n_fine);
            let vscratch_a = alloc_zeros_f32(&self.client, n_v);
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

            // Per channel: build the Laplacian-style layer + run weber.
            let log_l_bkg = alloc_zeros_f32(&self.client, n_fine);
            for c in 0..N_CHANNELS {
                let coarse = self.gauss_ref[k + 1].planes[c].clone();
                let fine = self.gauss_ref[k].planes[c].clone();
                let band = self.bands_ref[k].planes[c].clone();

                let vscratch_c = alloc_zeros_f32(&self.client, n_v);
                let upscaled_c = alloc_zeros_f32(&self.client, n_fine);
                let layer_c = alloc_zeros_f32(&self.client, n_fine);

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
                        ArrayArg::from_raw_parts(upscaled_c.clone(), n_fine),
                        coarse_w,
                        fine_w,
                        fine_h,
                    );
                    subtract_kernel::launch::<R>(
                        &self.client,
                        count_fine.clone(),
                        cube_dim,
                        ArrayArg::from_raw_parts(fine, n_fine),
                        ArrayArg::from_raw_parts(upscaled_c, n_fine),
                        ArrayArg::from_raw_parts(layer_c.clone(), n_fine),
                        n_fine as u32,
                    );
                    weber_contrast_compute_kernel::launch::<R>(
                        &self.client,
                        count_fine.clone(),
                        cube_dim,
                        ArrayArg::from_raw_parts(layer_c, n_fine),
                        ArrayArg::from_raw_parts(l_bkg_fine.clone(), n_fine),
                        ArrayArg::from_raw_parts(band, n_fine),
                        ArrayArg::from_raw_parts(log_l_bkg.clone(), n_fine),
                        n_fine as u32,
                    );
                }
            }
            log_l_bkg_handles.push(log_l_bkg);
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

        // Read back log_l_bkg per band: non-baseband from GPU,
        // baseband as replicated scalar matching host_scalar's
        // WeberPyramid shape.
        let mut log_l_bkg_out: Vec<Vec<f32>> = Vec::with_capacity(n_levels);
        for log_h in log_l_bkg_handles.drain(..) {
            let bytes = self
                .client
                .read_one(log_h)
                .map_err(|_| Error::InvalidImageSize)?;
            log_l_bkg_out.push(f32::from_bytes(&bytes).to_vec());
        }
        log_l_bkg_out.push(vec![log_l_bkg_baseband; baseband_n]);

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
        let (weber_bands, log_l_bkg) = self.compute_dkl_weber_pyramid(srgb)?;
        let n_levels = self.n_levels as usize;
        let freqs = band_frequencies(ppd, self.width as usize, self.height as usize);

        let cube_dim = CubeDim::new_1d(64);
        let csf_channels = [CsfChannel::A, CsfChannel::Rg, CsfChannel::Vy];

        let mut t_p_bands: Vec<[Vec<f32>; 3]> = Vec::with_capacity(n_levels);
        for k in 0..n_levels {
            let rho_k = freqs[k];
            let is_first = k == 0;
            let is_baseband = k == n_levels - 1;
            let band_mul: f32 = if is_first || is_baseband { 1.0 } else { 2.0 };
            let bw = if k == 0 {
                self.width as usize
            } else {
                (self.width as usize) >> k
            };
            let bh = if k == 0 {
                self.height as usize
            } else {
                (self.height as usize) >> k
            };
            let n_px = bw * bh;
            debug_assert_eq!(weber_bands[k][0].len(), n_px);
            debug_assert_eq!(log_l_bkg[k].len(), n_px);

            let log_l_bkg_h = self.client.create_from_slice(f32::as_bytes(&log_l_bkg[k]));

            let count = CubeCount::Static((n_px as u32).div_ceil(64), 1, 1);
            let mut planes = [Vec::new(), Vec::new(), Vec::new()];

            for (c, cc) in csf_channels.iter().enumerate() {
                let logs_row = precompute_logs_row(rho_k, *cc);
                let logs_row_h = self.client.create_from_slice(f32::as_bytes(&logs_row));
                let weber_h = self
                    .client
                    .create_from_slice(f32::as_bytes(&weber_bands[k][c]));
                let t_p_h = alloc_zeros_f32(&self.client, n_px);

                let ch_gain_eff = if is_baseband {
                    1.0
                } else {
                    band_mul * CH_GAIN[c]
                };

                unsafe {
                    csf_apply_per_pixel_kernel::launch::<R>(
                        &self.client,
                        count.clone(),
                        cube_dim,
                        ArrayArg::from_raw_parts(weber_h, n_px),
                        ArrayArg::from_raw_parts(log_l_bkg_h.clone(), n_px),
                        ArrayArg::from_raw_parts(logs_row_h, 32),
                        ArrayArg::from_raw_parts(t_p_h.clone(), n_px),
                        ch_gain_eff,
                        n_px as u32,
                    );
                }

                let bytes = self
                    .client
                    .read_one(t_p_h)
                    .map_err(|_| Error::InvalidImageSize)?;
                planes[c] = f32::from_bytes(&bytes).to_vec();
            }
            t_p_bands.push(planes);
        }

        Ok(t_p_bands)
    }

    /// Per-band per-channel masked-difference (`D`) bands.
    ///
    /// Composes the GPU Weber-pyramid + per-pixel CSF apply for
    /// both reference and distorted sides, then runs the
    /// host-scalar [`mult_mutual_band`] masker. Output per cvvdp's
    /// `apply_masking_model`:
    /// - Non-baseband levels: `D = mult_mutual_band(T_p_dis, T_p_ref)`
    ///   — cvvdp's per-channel masker (min-abs + phase-uncertainty +
    ///   xchannel pool + soft clamp).
    /// - Baseband level: `D[c] = |T_p_dis[c] - T_p_ref[c]|` —
    ///   matches cvvdp's baseband bypass because both `T_p` are
    ///   built with `CH_GAIN_eff = 1.0` and `band_mul = 1.0` at the
    ///   baseband, so the subtraction equals
    ///   `(dis_weber - ref_weber) * S`.
    ///
    /// Per cvvdp's `weber_g1` contract, the CSF lookup uses the
    /// **reference's** achromatic `log10(L_bkg)` for both `T_p`
    /// (test) and `R_p` (reference) — see `host_scalar` line ~124.
    /// `compute_dkl_t_p_bands` uses each side's own log_l_bkg, so
    /// this helper inlines the CSF apply and shares the ref-side
    /// log_l_bkg + logs_row uploads across both sides.
    ///
    /// Moving the masking step to a GPU composition is the next
    /// chunk after this — the GPU masking kernels already exist
    /// and are parity-tested in `tests/masking_kernel.rs`.
    pub fn compute_dkl_d_bands(
        &mut self,
        ref_srgb: &[u8],
        dist_srgb: &[u8],
        ppd: f32,
    ) -> Result<Vec<[Vec<f32>; 3]>> {
        let (ref_weber, ref_log_l_bkg) = self.compute_dkl_weber_pyramid(ref_srgb)?;
        // Discard dist log_l_bkg — cvvdp's weber_g1 uses ref's log_l_bkg.
        let (dist_weber, _) = self.compute_dkl_weber_pyramid(dist_srgb)?;

        let n_levels = self.n_levels as usize;
        let freqs = band_frequencies(ppd, self.width as usize, self.height as usize);
        let cube_dim = CubeDim::new_1d(64);
        let csf_channels = [CsfChannel::A, CsfChannel::Rg, CsfChannel::Vy];

        let mut d_bands: Vec<[Vec<f32>; 3]> = Vec::with_capacity(n_levels);
        for k in 0..n_levels {
            let rho_k = freqs[k];
            let is_first = k == 0;
            let is_baseband = k == n_levels - 1;
            let band_mul: f32 = if is_first || is_baseband { 1.0 } else { 2.0 };
            let bw = if k == 0 {
                self.width as usize
            } else {
                (self.width as usize) >> k
            };
            let bh = if k == 0 {
                self.height as usize
            } else {
                (self.height as usize) >> k
            };
            let n_px = bw * bh;
            debug_assert_eq!(ref_weber[k][0].len(), n_px);
            debug_assert_eq!(ref_log_l_bkg[k].len(), n_px);

            // Upload ref log_l_bkg once per band; both sides reuse it.
            let log_l_bkg_h = self
                .client
                .create_from_slice(f32::as_bytes(&ref_log_l_bkg[k]));
            let count = CubeCount::Static((n_px as u32).div_ceil(64), 1, 1);

            let mut t_p_ref: [Vec<f32>; 3] =
                [vec![0.0; n_px], vec![0.0; n_px], vec![0.0; n_px]];
            let mut t_p_dis: [Vec<f32>; 3] =
                [vec![0.0; n_px], vec![0.0; n_px], vec![0.0; n_px]];

            for (c, cc) in csf_channels.iter().enumerate() {
                let logs_row = precompute_logs_row(rho_k, *cc);
                let logs_row_h = self.client.create_from_slice(f32::as_bytes(&logs_row));
                let ch_gain_eff = if is_baseband {
                    1.0
                } else {
                    band_mul * CH_GAIN[c]
                };

                for (side_weber, dst) in
                    [(&ref_weber, &mut t_p_ref), (&dist_weber, &mut t_p_dis)]
                {
                    let weber_h = self
                        .client
                        .create_from_slice(f32::as_bytes(&side_weber[k][c]));
                    let t_p_h = alloc_zeros_f32(&self.client, n_px);
                    unsafe {
                        csf_apply_per_pixel_kernel::launch::<R>(
                            &self.client,
                            count.clone(),
                            cube_dim,
                            ArrayArg::from_raw_parts(weber_h, n_px),
                            ArrayArg::from_raw_parts(log_l_bkg_h.clone(), n_px),
                            ArrayArg::from_raw_parts(logs_row_h.clone(), 32),
                            ArrayArg::from_raw_parts(t_p_h.clone(), n_px),
                            ch_gain_eff,
                            n_px as u32,
                        );
                    }
                    let bytes = self
                        .client
                        .read_one(t_p_h)
                        .map_err(|_| Error::InvalidImageSize)?;
                    dst[c] = f32::from_bytes(&bytes).to_vec();
                }
            }

            let d = if is_baseband {
                let mut planes: [Vec<f32>; 3] =
                    [vec![0.0; n_px], vec![0.0; n_px], vec![0.0; n_px]];
                for c in 0..N_CHANNELS {
                    for i in 0..n_px {
                        planes[c][i] = (t_p_dis[c][i] - t_p_ref[c][i]).abs();
                    }
                }
                planes
            } else {
                mult_mutual_band(&t_p_dis, &t_p_ref, bw, bh)
            };
            d_bands.push(d);
        }

        Ok(d_bands)
    }

    /// Run color + Laplacian-pyramid + per-band CSF weighting.
    ///
    /// `ppd` is pixels-per-degree (from `DisplayGeometry::pixels_per_degree()`).
    /// `l_bkg` is the scalar background-luminance approximation used
    /// for every pyramid band — typically a per-image mean or
    /// display-peak / 2. The per-pixel L_bkg form (cvvdp's exact
    /// behaviour) lands once we wire the achromatic gauss[1] read
    /// path into the kernel.
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
    /// (`host_scalar::predict_jod_still_3ch`). The kernels for every
    /// stage exist and are individually parity-tested; replacing the
    /// host scalar with a fully-GPU composition is the next chunk of
    /// pipeline work. Score matches pycvvdp v0.5.4 on the v1 R2
    /// manifest within 0.006 JOD across q1–q90.
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
