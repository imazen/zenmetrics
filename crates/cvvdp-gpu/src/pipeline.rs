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
//!    buffers each.
//! 3. Build per-channel Laplacian pyramids (downscale + upscale +
//!    subtract) → `n_levels` bands per channel per side.
//! 4. Apply CSF weights per band (`weight_band_kernel`).
//! 5. Compute masked differences per band (`masked_diff_kernel`).
//! 6. Per-band Minkowski accumulation (`pool_band_kernel`) → per-band
//!    f32 partials.
//! 7. Host-side fold: per-band → per-channel → overall `D`, then JOD.
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
use crate::params::CvvdpParams;
use crate::{Error, MAX_LEVELS, N_CHANNELS, PYRAMID_MIN_DIM, Result};

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

/// Reference-side pyramid kept across `score_with_reference` calls.
#[allow(dead_code)]
struct CachedReference {
    /// CSF-weighted pyramid bands. Indexed `[channel][level]`.
    bands: Vec<Vec<cubecl::server::Handle>>,
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
    /// given parameter bundle.
    ///
    /// Returns [`Error::InvalidImageSize`] if either dimension is
    /// smaller than [`PYRAMID_MIN_DIM`] × 2 (no usable pyramid).
    pub fn new(
        client: ComputeClient<R>,
        width: u32,
        height: u32,
        params: CvvdpParams,
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

    /// Score a (reference, distorted) pair, returning JOD.
    ///
    /// **Stub** — wiring complete enough for the type to exist; the
    /// kernel bodies don't yet produce parity numbers. Returns `0.0`
    /// until per-stage goldens land.
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
        // TODO: launch kernels, fold partials, apply JOD mapping.
        let _ = self.params; // keep field "used" until pipeline lands
        Ok(0.0)
    }

    /// Cache the reference-side CSF-weighted pyramid for repeated
    /// scoring against many distorted candidates.
    pub fn set_reference(&mut self, reference_srgb: &[u8]) -> Result<()> {
        let expected = (self.width as usize) * (self.height as usize) * 3;
        if reference_srgb.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: reference_srgb.len(),
            });
        }
        // TODO: run color → pyramid → CSF stages and stash bands.
        self.cached = Some(CachedReference { bands: Vec::new() });
        Ok(())
    }

    /// Score a distorted candidate against the cached reference.
    pub fn score_with_reference(&mut self, distorted_srgb: &[u8]) -> Result<f64> {
        if self.cached.is_none() {
            return Err(Error::NoCachedReference);
        }
        let expected = (self.width as usize) * (self.height as usize) * 3;
        if distorted_srgb.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: distorted_srgb.len(),
            });
        }
        // TODO: reuse cached reference bands.
        Ok(0.0)
    }
}
