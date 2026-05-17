//! Butteraugli pipeline orchestration.
//!
//! The full butteraugli algorithm wired together as kernel launches over
//! pre-allocated CubeCL buffers. Three entry points:
//!
//! - [`Butteraugli::new`] + [`Butteraugli::compute`] — single-resolution.
//! - [`Butteraugli::new_multires`] + [`Butteraugli::compute`] —
//!   single-resolution + half-resolution sibling supersample-added in,
//!   matching CPU butteraugli's default mode.
//! - [`Butteraugli::set_reference`] + [`Butteraugli::compute_with_reference`]
//!   — cache reference-side intermediates and reuse them across many
//!   distorted-image comparisons (encoder rate-distortion search).
//!
//! Constants and orchestration follow the CPU `butteraugli` v0.9.2
//! crate's `compute_psycho_diff_malta` and `compute_mask_from_hf_uhf`
//! stages — see comments next to each step for the CPU function it
//! mirrors.

use cubecl::prelude::*;

use crate::kernels::{
    blur, blur_lut, colors, diffmap, downscale, frequency, malta, masking, reduction,
};
use crate::{ButteraugliParams, Error, GpuButteraugliResult, Result};

/// Default intensity multiplier — value of one display nit relative to
/// linear-light input scale. CPU butteraugli passes `80.0` for the
/// standard 80-nit display directly to `opsin_dynamics_image`. Linear
/// inputs already live on [0, 1] (after sRGB transfer); they get scaled
/// to [0, 80] inside opsin, *not* divided by 255 again. Public so a
/// caller can build a [`ButteraugliParams`] referring to the SDR
/// default.
pub const DEFAULT_INTENSITY_MULTIPLIER: f32 = 80.0;
#[allow(dead_code)]
const _: f32 = DEFAULT_INTENSITY_MULTIPLIER; // silence unused-const warning post-refactor

// ═══ frequency separation ═══
const SIGMA_LF: f32 = 7.155_933_4;
/// Sigma for opsin's sensitivity-input blur. CPU butteraugli's
/// `opsin_dynamics_image` always uses this 5-tap blur (with mirrored
/// boundaries, kernel radius 2) — *not* SIGMA_LF. Mismatching this
/// causes the per-pixel sensitivity to be smoothed over a 16-pixel
/// radius instead of a 2-pixel radius, which drops the perturbed
/// pixel's apparent contrast by ~12 % on tiny perturbations and
/// roughly doubles the diffmap when one bright pixel is involved.
const SIGMA_OPSIN: f32 = 1.2;
const SIGMA_HF: f32 = 3.224_899;
const SIGMA_UHF: f32 = 1.564_163_3;
const REMOVE_MF_RANGE: f32 = 0.29;
const ADD_MF_RANGE: f32 = 0.1;
const REMOVE_HF_RANGE: f32 = 1.5;
const REMOVE_UHF_RANGE: f32 = 0.04;
const SUPPRESS_XY: f32 = 46.0;

// (Default HF asymmetry = 1.0 — runtime-overridable via ButteraugliParams.)

// ═══ Malta band parameters (libjxl/butteraugli) ═══
const W_MF_MALTA: f64 = 37.081_987_039_9;
const NORM1_MF: f64 = 130_262_059.556;
const W_MF_MALTA_X: f64 = 8_246.753_213_53;
const NORM1_MF_X: f64 = 1_009_002.705_82;
const W_HF_MALTA: f64 = 18.723_741_438_7;
const NORM1_HF: f64 = 4_498_534.452_32;
const W_HF_MALTA_X: f64 = 6_923.994_761_09;
const NORM1_HF_X: f64 = 8_051.158_332_47;
const W_UHF_MALTA: f64 = 1.100_390_325_55;
const NORM1_UHF: f64 = 71.780_027_516_9;
const W_UHF_MALTA_X: f64 = 173.5;
const NORM1_UHF_X: f64 = 5.0;

// ═══ frequency-band weights (l2_diff and DC contribution) ═══
const WMUL: [f64; 9] = [
    400.0,
    1.508_157_031_18,
    0.0,
    2_150.0,
    10.619_543_323_9,
    16.217_604_315_2,
    29.235_379_799_4,
    0.844_626_970_982,
    0.703_646_627_719,
];

// ═══ mask blur radius ═══
const MASK_RADIUS: f32 = 2.7;

/// Compute Malta `(norm2_0gt1, norm2_0lt1, norm1_f32)` host-side at
/// f64 precision. Mirrors the f64 prelude in CPU `malta_diff_map_impl`.
fn malta_norm(w_0gt1: f64, w_0lt1: f64, norm1: f64, use_lf: bool) -> (f32, f32, f32) {
    const K_WEIGHT0: f64 = 0.5;
    const K_WEIGHT1: f64 = 0.33;
    const LEN: f64 = 3.75;
    let mulli = if use_lf {
        0.611_612_573_796
    } else {
        0.399_058_176_37
    };
    let w_pre0gt1 = mulli * (K_WEIGHT0 * w_0gt1).sqrt() / (LEN * 2.0 + 1.0);
    let w_pre0lt1 = mulli * (K_WEIGHT1 * w_0lt1).sqrt() / (LEN * 2.0 + 1.0);
    let norm2_0gt1 = (w_pre0gt1 * norm1) as f32;
    let norm2_0lt1 = (w_pre0lt1 * norm1) as f32;
    (norm2_0gt1, norm2_0lt1, norm1 as f32)
}

/// Per-instance allocations + per-call orchestration of the full
/// butteraugli kernel pipeline. Construct once, reuse for many
/// comparisons at the same resolution.
pub struct Butteraugli<R: Runtime> {
    client: ComputeClient<R>,
    width: u32,
    height: u32,
    n: usize,

    // sRGB u8 staging
    src_u8_a: cubecl::server::Handle,
    src_u8_b: cubecl::server::Handle,

    // Planar linear RGB / XYB after opsin (×2 images × 3 channels = 6 buffers)
    lin_a: [cubecl::server::Handle; 3],
    lin_b: [cubecl::server::Handle; 3],

    // Blurred linear RGB (for opsin dynamics adaptation)
    blur_a: [cubecl::server::Handle; 3],
    blur_b: [cubecl::server::Handle; 3],

    // Frequency bands per channel: [UHF, HF, MF, LF] × [X, Y, B] × 2 images
    freq_a: [[cubecl::server::Handle; 3]; 4],
    freq_b: [[cubecl::server::Handle; 3]; 4],

    // Per-pixel block diff accumulators [X, Y, B]
    block_diff_dc: [cubecl::server::Handle; 3],
    block_diff_ac: [cubecl::server::Handle; 3],

    // Mask plane + scratch for the blurred mask of image B
    mask: cubecl::server::Handle,
    mask_scratch: cubecl::server::Handle,
    /// Cached `blur(combine+precompute(image A))` for the mask pipeline
    /// — needed by both fuzzy-erosion (→ self.mask) and mask_to_error
    /// (against image B's blurred). Permanent so a `set_reference` call
    /// can keep it across many `compute_with_reference` calls.
    cached_blurred_a: cubecl::server::Handle,

    // Final diffmap
    diffmap_buf: cubecl::server::Handle,

    // Generic temp planes
    temp1: cubecl::server::Handle,
    temp2: cubecl::server::Handle,

    /// Half-resolution sibling for the multi-resolution pass. `None`
    /// for half-res instances themselves and for `Butteraugli::new` (which
    /// is single-resolution only). Populated by [`Butteraugli::new_multires`].
    half_res: Option<Box<Butteraugli<R>>>,

    /// Set by [`set_reference`]. While true, the reference-side
    /// intermediates (lin_a XYB, freq_a[*][*], cached_blurred_a, mask)
    /// are valid and `compute_with_reference` may skip recomputing them.
    has_cached_reference: bool,

    /// Persistent host-side packing scratch for sRGB → u32 widening
    /// (one u32 per byte; WGSL has no `Array<u8>` storage type). Sized
    /// to `n × 3` at construction; reused across uploads instead of
    /// allocating a fresh `Vec<u32>` per `populate_linear_from_srgb`.
    /// Same shape as zensim-gpu's `pack_scratch`.
    pack_scratch: Vec<u32>,

    /// Active comparison parameters. Overwritten by
    /// `compute_with_options` and `set_reference_with_options`; the
    /// non-`_with_options` entry points use [`ButteraugliParams::default`].
    /// Stored on the struct so internal helpers can read it without
    /// threading the value through every call.
    params: ButteraugliParams,

    /// Pre-computed Gaussian weight + integral tables, one per fixed
    /// sigma the pipeline uses. Uploaded once at construction; the LUT
    /// blur kernels read them per tap instead of calling `powf`. See
    /// [`crate::kernels::blur_lut`] for the layout. Indices match
    /// [`BLUR_SIGMAS`] below.
    blur_tables: [cubecl::server::Handle; 5],
    blur_radii: [u32; 5],
    blur_table_lens: [usize; 5],
}

/// Fixed sigmas referenced by the LUT blur tables, indexed via
/// [`BlurKind`]. Stored as `f32` so the tables match the kernels'
/// `f32::exp(-0.5*(d/s)^2)` exactly.
const BLUR_SIGMAS: [f32; 5] = [
    SIGMA_OPSIN,
    SIGMA_LF,
    SIGMA_HF,
    SIGMA_UHF,
    MASK_RADIUS,
];

/// Index into [`BLUR_SIGMAS`] / [`Butteraugli::blur_tables`].
#[derive(Clone, Copy)]
#[repr(usize)]
enum BlurKind {
    Opsin = 0,
    Lf = 1,
    Hf = 2,
    Uhf = 3,
    Mask = 4,
}

/// Exact-bit match against [`BLUR_SIGMAS`]. Returns `None` for an
/// unrecognised sigma; the caller then falls back to the powf-per-tap
/// blur kernel (preserves correctness for any future caller passing a
/// novel sigma).
fn blur_kind_for_sigma(sigma: f32) -> Option<BlurKind> {
    if sigma == SIGMA_OPSIN {
        Some(BlurKind::Opsin)
    } else if sigma == SIGMA_LF {
        Some(BlurKind::Lf)
    } else if sigma == SIGMA_HF {
        Some(BlurKind::Hf)
    } else if sigma == SIGMA_UHF {
        Some(BlurKind::Uhf)
    } else if sigma == MASK_RADIUS {
        Some(BlurKind::Mask)
    } else {
        None
    }
}

fn alloc_plane<R: Runtime>(client: &ComputeClient<R>, n: usize) -> cubecl::server::Handle {
    client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]))
}

/// Reject NaN, +Inf, and non-positive intensity_target.
pub(crate) fn validate_params(params: &ButteraugliParams) -> Result<()> {
    if !params.intensity_target.is_finite() || params.intensity_target <= 0.0 {
        return Err(Error::InvalidParams("intensity_target must be > 0"));
    }
    if !params.hf_asymmetry.is_finite() || params.hf_asymmetry <= 0.0 {
        return Err(Error::InvalidParams("hf_asymmetry must be > 0"));
    }
    if !params.xmul.is_finite() || params.xmul < 0.0 {
        return Err(Error::InvalidParams("xmul must be >= 0"));
    }
    Ok(())
}

/// Downsample 2× the linear-RGB planes from `full` into the matching
/// `half` instance. Free-standing so callers can keep `&mut` borrows of
/// both instances. Mirrors CPU butteraugli's `subsample_linear_rgb_2x`,
/// but operates plane-by-plane on already-deinterleaved buffers.
fn populate_half_res_linear<R: Runtime>(full: &Butteraugli<R>, half: &Butteraugli<R>, is_a: bool) {
    let (full_lin, half_lin) = if is_a {
        (&full.lin_a, &half.lin_a)
    } else {
        (&full.lin_b, &half.lin_b)
    };
    const TPB: u32 = 256;
    let cubes = (half.n as u32).div_ceil(TPB);
    let dim = CubeCount::Static(cubes, 1, 1);
    let block = CubeDim::new_1d(TPB);
    // Fused 3-channel downsample (1 launch instead of 3). Bit-exact
    // with the single-channel kernel (sum/count, not sum*(1/count)).
    unsafe {
        downscale::downsample_2x_3ch_kernel::launch_unchecked::<R>(
            &full.client,
            dim,
            block,
            ArrayArg::from_raw_parts(full_lin[0].clone(), full.n),
            ArrayArg::from_raw_parts(full_lin[1].clone(), full.n),
            ArrayArg::from_raw_parts(full_lin[2].clone(), full.n),
            ArrayArg::from_raw_parts(half_lin[0].clone(), half.n),
            ArrayArg::from_raw_parts(half_lin[1].clone(), half.n),
            ArrayArg::from_raw_parts(half_lin[2].clone(), half.n),
            full.width,
            full.height,
            half.width,
            half.height,
        );
    }
}

fn alloc_3<R: Runtime>(client: &ComputeClient<R>, n: usize) -> [cubecl::server::Handle; 3] {
    [
        alloc_plane(client, n),
        alloc_plane(client, n),
        alloc_plane(client, n),
    ]
}

impl<R: Runtime> Butteraugli<R> {
    /// Allocate all per-instance buffers for `width × height` images.
    ///
    /// # Panics
    ///
    /// Panics if `width × height × 3` overflows `usize`. Callers passing
    /// untrusted dimensions should pre-validate with the same upper bound
    /// the sibling pipelines use (e.g. reject anything where
    /// `width.checked_mul(height).is_none()`).
    pub fn new(client: ComputeClient<R>, width: u32, height: u32) -> Self {
        // Widen to usize before the multiply: a single `(width * height) as usize`
        // wraps the u32-typed product silently on huge dimensions in release,
        // producing under-allocated GPU buffers and garbage scores. Sibling
        // pipelines (ssim2-gpu, dssim-gpu, zensim-gpu) already widen first.
        let n = (width as usize)
            .checked_mul(height as usize)
            .expect("width × height overflows usize");
        let n_bytes = n
            .checked_mul(3)
            .expect("width × height × 3 overflows usize");
        // T4.L (2026-05-16): pack 3 sRGB bytes per pixel into ONE u32
        // (R | G<<8 | B<<16; alpha unused). Length = n, not n*3. Cuts
        // per-call host→device upload from `n × 12 B` to `n × 4 B`.
        let src_u8_a = client.create_from_slice(u32::as_bytes(&vec![0_u32; n]));
        let src_u8_b = client.create_from_slice(u32::as_bytes(&vec![0_u32; n]));

        let lin_a = alloc_3(&client, n);
        let lin_b = alloc_3(&client, n);
        let blur_a = alloc_3(&client, n);
        let blur_b = alloc_3(&client, n);

        let freq_a = [
            alloc_3(&client, n),
            alloc_3(&client, n),
            alloc_3(&client, n),
            alloc_3(&client, n),
        ];
        let freq_b = [
            alloc_3(&client, n),
            alloc_3(&client, n),
            alloc_3(&client, n),
            alloc_3(&client, n),
        ];

        let block_diff_dc = alloc_3(&client, n);
        let block_diff_ac = alloc_3(&client, n);

        let mask = alloc_plane(&client, n);
        let mask_scratch = alloc_plane(&client, n);
        let cached_blurred_a = alloc_plane(&client, n);
        let diffmap_buf = alloc_plane(&client, n);
        let temp1 = alloc_plane(&client, n);
        let temp2 = alloc_plane(&client, n);

        // Pre-compute and upload one Gaussian LUT per fixed sigma. Each
        // table is small (≤ 67 floats for σ=7.16, the largest), so the
        // five allocs are negligible. Reused across every blur call.
        let mut blur_tables: [Option<cubecl::server::Handle>; 5] =
            [None, None, None, None, None];
        let mut blur_radii = [0_u32; 5];
        let mut blur_table_lens = [0_usize; 5];
        for (i, &sigma) in BLUR_SIGMAS.iter().enumerate() {
            let (table, r) = blur_lut::make_table(sigma);
            blur_table_lens[i] = table.len();
            blur_radii[i] = r as u32;
            blur_tables[i] = Some(client.create_from_slice(f32::as_bytes(&table)));
        }
        let blur_tables = blur_tables.map(|h| h.unwrap());

        Self {
            client,
            width,
            height,
            n,
            src_u8_a,
            src_u8_b,
            lin_a,
            lin_b,
            blur_a,
            blur_b,
            freq_a,
            freq_b,
            block_diff_dc,
            block_diff_ac,
            mask,
            mask_scratch,
            cached_blurred_a,
            diffmap_buf,
            temp1,
            temp2,
            half_res: None,
            has_cached_reference: false,
            // T4.L: one u32 per pixel (packed RGBA), not 3 u32 per pixel.
            pack_scratch: vec![0_u32; n],
            params: ButteraugliParams::default(),
            blur_tables,
            blur_radii,
            blur_table_lens,
        }
    }

    /// Construct a multi-resolution `Butteraugli` instance — same as
    /// [`Butteraugli::new`] plus a `(w/2)×(h/2)` sibling whose diffmap
    /// is supersample-added into the full-res diffmap before reduction.
    /// Matches CPU butteraugli's default (non-`single_resolution`) mode.
    ///
    /// For very small images (`w < 16` or `h < 16`) the sibling is
    /// skipped — same threshold CPU butteraugli uses.
    pub fn new_multires(client: ComputeClient<R>, width: u32, height: u32) -> Self {
        const MIN_SIZE_FOR_SUBSAMPLE: u32 = 16;
        let mut full = Self::new(client.clone(), width, height);
        if width >= MIN_SIZE_FOR_SUBSAMPLE && height >= MIN_SIZE_FOR_SUBSAMPLE {
            let half_w = width.div_ceil(2);
            let half_h = height.div_ceil(2);
            full.half_res = Some(Box::new(Self::new(client, half_w, half_h)));
        }
        full
    }

    /// Compute the butteraugli `(score, pnorm_3)` for one image pair.
    /// Both images are sRGB u8 packed RGB (`width × height × 3` bytes).
    ///
    /// If this instance was created with [`Butteraugli::new_multires`],
    /// the half-resolution sibling's diffmap is supersample-added into
    /// the full-res diffmap before reduction (matches CPU butteraugli's
    /// default mode). With [`Butteraugli::new`] the call is single-
    /// resolution only.
    ///
    /// Returns [`Error::DimensionMismatch`] if either input length
    /// doesn't match `width × height × 3`.
    pub fn compute(&mut self, ref_srgb: &[u8], dist_srgb: &[u8]) -> Result<GpuButteraugliResult> {
        self.compute_with_options(ref_srgb, dist_srgb, &ButteraugliParams::default())
    }

    /// `compute` with runtime-tunable [`ButteraugliParams`] (HDR
    /// intensity target, asymmetric weights, chroma multiplier).
    /// Returns `Err` on dimension mismatch or invalid params.
    pub fn compute_with_options(
        &mut self,
        ref_srgb: &[u8],
        dist_srgb: &[u8],
        params: &ButteraugliParams,
    ) -> Result<GpuButteraugliResult> {
        validate_params(params)?;
        self.check_dims(ref_srgb)?;
        self.check_dims(dist_srgb)?;
        self.set_params_recursive(params);
        self.populate_linear_from_srgb(true, ref_srgb);
        self.populate_linear_from_srgb(false, dist_srgb);
        self.run_pipeline_from_linear(true, true);
        Ok(reduction::reduce::<R>(
            &self.client,
            self.diffmap_buf.clone(),
            self.n,
        ))
    }

    /// Cache the reference image's intermediate state. After this call,
    /// [`Butteraugli::compute_with_reference`] can be called any number
    /// of times with different distorted images; each one skips the
    /// reference-side ~half of the pipeline (sRGB→linear→opsin→
    /// frequency separation → reference mask blur).
    ///
    /// Returns [`Error::DimensionMismatch`] if `ref_srgb.len()` doesn't
    /// match `width × height × 3`.
    pub fn set_reference(&mut self, ref_srgb: &[u8]) -> Result<()> {
        self.set_reference_with_options(ref_srgb, &ButteraugliParams::default())
    }

    /// Cache the reference image with a specific [`ButteraugliParams`].
    /// All subsequent [`compute_with_reference`] (or
    /// [`compute_with_reference_with_options`]) calls reuse those params
    /// — call again to change them.
    pub fn set_reference_with_options(
        &mut self,
        ref_srgb: &[u8],
        params: &ButteraugliParams,
    ) -> Result<()> {
        validate_params(params)?;
        self.check_dims(ref_srgb)?;
        self.set_params_recursive(params);
        self.populate_linear_from_srgb(true, ref_srgb);
        if let Some(half) = self.half_res.as_deref() {
            populate_half_res_linear(self, half, true);
        }
        self.apply_opsin(true);
        self.separate_frequencies(true);
        self.compute_mask_pipeline_reference_only();
        self.has_cached_reference = true;
        if let Some(half) = self.half_res.as_mut() {
            half.apply_opsin(true);
            half.separate_frequencies(true);
            half.compute_mask_pipeline_reference_only();
            half.has_cached_reference = true;
        }
        Ok(())
    }

    /// Drop the cached reference state. The next call must be
    /// `set_reference`/`set_reference_with_options` again.
    pub fn clear_reference(&mut self) {
        self.has_cached_reference = false;
        if let Some(half) = self.half_res.as_mut() {
            half.has_cached_reference = false;
        }
    }

    /// Compute butteraugli against the cached reference (must follow a
    /// [`set_reference`] on this instance). Roughly halves per-call cost
    /// compared to [`compute`] when iterating many distorted images
    /// against a fixed reference (encoder rate-distortion search).
    ///
    /// Returns [`Error::NoCachedReference`] if [`set_reference`] hasn't
    /// been called, or [`Error::DimensionMismatch`] if `dist_srgb.len()`
    /// doesn't match `width × height × 3`.
    pub fn compute_with_reference(&mut self, dist_srgb: &[u8]) -> Result<GpuButteraugliResult> {
        self.compute_with_reference_inner(dist_srgb)
    }

    /// Deprecated alias for [`compute_with_reference`]. Kept for
    /// source-compat with callers that imported the old `try_*` name.
    #[deprecated(
        since = "0.0.2",
        note = "use compute_with_reference (now Result-typed)"
    )]
    pub fn try_compute_with_reference(&mut self, dist_srgb: &[u8]) -> Result<GpuButteraugliResult> {
        self.compute_with_reference_inner(dist_srgb)
    }

    /// Compute butteraugli against the cached reference, taking the
    /// distorted side as **3 already-uploaded planar f32 GPU handles**
    /// in linear-RGB space (each `width × height × 4` bytes, in `[0, 1]`
    /// pre-opsin). Skips the sRGB-bytes upload + sRGB→linear GPU
    /// conversion that [`compute_with_reference`] does internally.
    ///
    /// The provided handles' contents are **mutated in place** by the
    /// opsin / frequency separation kernels (which write back into
    /// `lin_b` per the existing pipeline). If the caller wants to keep
    /// them intact, clone the handles before calling — cubecl `Handle`
    /// is reference-counted so a clone is cheap, but the underlying
    /// GPU buffer will then be allocated separately by the next
    /// downstream consumer.
    ///
    /// **Use case**: encoder rate-distortion search where the encoder
    /// already produces the reconstructed image as planar linear-RGB
    /// GPU planes (e.g. jxl-encoder-gpu's recon planes). Eliminates the
    /// recon-download → host sRGB-convert → re-upload boundary work
    /// (~30-60 ms per iter at 1 MP, scales linearly with size — at
    /// 16 MP this can save several hundred ms per refinement iter).
    ///
    /// Each caller-supplied plane MUST hold exactly `width × height`
    /// f32 values in row-major order, contiguous, no padding.
    ///
    /// Gated behind the `internals` cargo feature (mirrors the existing
    /// CPU `butteraugli` crate's `internals` escape hatch). Not part of
    /// the stable API; field layout / kernel order may shift with
    /// internal pipeline refactors.
    #[cfg(feature = "internals")]
    pub fn compute_with_reference_from_linear_planes(
        &mut self,
        dist_r: cubecl::server::Handle,
        dist_g: cubecl::server::Handle,
        dist_b: cubecl::server::Handle,
    ) -> Result<GpuButteraugliResult> {
        if !self.has_cached_reference {
            return Err(Error::NoCachedReference);
        }
        // Replace the distorted-side linear-RGB plane handles. The
        // existing apply_opsin / separate_frequencies / mask kernels
        // overwrite these in-place — see this struct's pipeline
        // documentation for the chain.
        self.lin_b[0] = dist_r;
        self.lin_b[1] = dist_g;
        self.lin_b[2] = dist_b;
        // do_a=false: reference side cached; do_b=true: distorted side
        // needs full opsin / frequency / mask / diff pipeline run.
        self.run_pipeline_from_linear(false, true);
        Ok(reduction::reduce::<R>(
            &self.client,
            self.diffmap_buf.clone(),
            self.n,
        ))
    }

    fn compute_with_reference_inner(&mut self, dist_srgb: &[u8]) -> Result<GpuButteraugliResult> {
        if !self.has_cached_reference {
            return Err(Error::NoCachedReference);
        }
        self.check_dims(dist_srgb)?;
        self.populate_linear_from_srgb(false, dist_srgb);
        // do_a=false: reference side is cached; do_b=true: distorted side needs computing.
        self.run_pipeline_from_linear(false, true);
        Ok(reduction::reduce::<R>(
            &self.client,
            self.diffmap_buf.clone(),
            self.n,
        ))
    }

    fn check_dims(&self, srgb: &[u8]) -> Result<()> {
        let expected = self.n * 3;
        if srgb.len() != expected {
            Err(Error::DimensionMismatch {
                expected,
                got: srgb.len(),
            })
        } else {
            Ok(())
        }
    }

    fn set_params_recursive(&mut self, params: &ButteraugliParams) {
        self.params = *params;
        if let Some(half) = self.half_res.as_mut() {
            half.params = *params;
        }
    }

    /// Internal: run the pipeline assuming `lin_a` and/or `lin_b` are
    /// populated with linear RGB. `do_a` / `do_b` select which sides to
    /// (re)compute; the other side is assumed cached.
    fn run_pipeline_from_linear(&mut self, do_a: bool, do_b: bool) {
        // Downsample full-res linear into the half-res sibling before
        // opsin overwrites lin in place.
        if let Some(half) = self.half_res.as_deref() {
            if do_a {
                populate_half_res_linear(self, half, true);
            }
            if do_b {
                populate_half_res_linear(self, half, false);
            }
        }
        if do_a {
            self.apply_opsin(true);
            self.separate_frequencies(true);
        }
        if do_b {
            self.apply_opsin(false);
            self.separate_frequencies(false);
        }
        self.compute_psycho_diff();
        self.compute_dc_diff();
        if do_a && do_b {
            self.compute_mask_pipeline_full();
        } else if do_a {
            self.compute_mask_pipeline_reference_only();
            // No distorted side yet — the caller is `set_reference`,
            // which doesn't run the diffmap.
            return;
        } else {
            self.compute_mask_pipeline_distorted_only();
        }
        unsafe {
            self.launch_compute_diffmap();
        }
        // Take the half-res sibling out so we can call methods on both
        // it and `self` (each `&mut self`) without splitting the borrow.
        if let Some(mut half) = self.half_res.take() {
            half.run_pipeline_from_linear(do_a, do_b);
            // Recursion stops because `half.half_res` is None.
            let src = half.diffmap_buf.clone();
            let (sw, sh) = (half.width, half.height);
            self.launch_add_supersampled_2x_from(&src, sw, sh);
            self.half_res = Some(half);
        }
    }

    // ───────────────────────────── helpers ─────────────────────────────

    fn cube_count_1d(&self) -> CubeCount {
        const TPB: u32 = 256;
        let cubes = (self.n as u32).div_ceil(TPB);
        CubeCount::Static(cubes, 1, 1)
    }

    fn cube_dim_1d(&self) -> CubeDim {
        CubeDim::new_1d(256)
    }

    fn cube_count_2d(&self) -> CubeCount {
        let bx = self.width.div_ceil(16);
        let by = self.height.div_ceil(16);
        CubeCount::Static(bx, by, 1)
    }

    fn cube_dim_2d(&self) -> CubeDim {
        CubeDim::new_2d(16, 16)
    }

    /// Upload sRGB u8 input and convert to planar linear RGB into
    /// `lin_a` / `lin_b`. Linear values stay in [0, 1] until opsin
    /// scales by `intensity_multiplier=80`.
    fn populate_linear_from_srgb(&mut self, is_a: bool, srgb: &[u8]) {
        let n_bytes = self.n * 3;
        // Defense-in-depth check: every public caller goes through
        // `check_dims` first, so a release-mode panic here would only
        // fire on a buggy internal caller. Demoted to debug_assert.
        debug_assert_eq!(srgb.len(), n_bytes, "input length mismatch");
        // T4.L (2026-05-16): pack 3 sRGB bytes into ONE u32
        // (R | G<<8 | B<<16). Cuts host→device upload 3× vs the prior
        // one-byte-per-u32 layout — was the dominant warm-loop cost
        // per nsys (see docs/CUBECL_GOTCHAS.md G6.6).
        debug_assert_eq!(self.pack_scratch.len(), self.n);
        for (dst, triple) in self.pack_scratch.iter_mut().zip(srgb.chunks_exact(3)) {
            *dst =
                u32::from(triple[0]) | (u32::from(triple[1]) << 8) | (u32::from(triple[2]) << 16);
        }
        // T4.M (2026-05-16): pinned-host fast path — copies straight
        // into a pinned staging buffer (12-25 GB/s DMA on PCIe 4.0 vs
        // 5-6 GB/s from pageable). See CUBECL_GOTCHAS.md G6.5.
        let bytes = u32::as_bytes(&self.pack_scratch);
        if is_a {
            self.src_u8_a = self.client.create_from_slice_pinned(bytes);
        } else {
            self.src_u8_b = self.client.create_from_slice_pinned(bytes);
        }
        unsafe {
            self.launch_srgb_to_linear(is_a);
        }
    }

    /// Apply opsin: blur(σ=1.2) for sensitivity input, then opsin
    /// dynamics → planar XYB (overwrites `lin_a` / `lin_b` in place).
    fn apply_opsin(&self, is_a: bool) {
        let (lin, bl) = if is_a {
            (&self.lin_a, &self.blur_a)
        } else {
            (&self.lin_b, &self.blur_b)
        };
        // Fused 3-channel blur (2 launches instead of 6). Uses temp1,
        // temp2, mask_scratch as the H→V scratches; all 3 are written
        // here in full and not relied on across this call.
        self.blur_3ch_via(
            &lin[0].clone(),
            &lin[1].clone(),
            &lin[2].clone(),
            &bl[0].clone(),
            &bl[1].clone(),
            &bl[2].clone(),
            &self.temp1.clone(),
            &self.temp2.clone(),
            &self.mask_scratch.clone(),
            SIGMA_OPSIN,
        );
        unsafe {
            self.launch_opsin(is_a);
        }
    }

    unsafe fn launch_srgb_to_linear(&self, is_a: bool) {
        let (src, lin) = if is_a {
            (&self.src_u8_a, &self.lin_a)
        } else {
            (&self.src_u8_b, &self.lin_b)
        };
        unsafe {
            colors::srgb_u8_to_linear_planar_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                // T4.L: one u32 per pixel, not n_bytes.
                ArrayArg::from_raw_parts(src.clone(), self.n),
                ArrayArg::from_raw_parts(lin[0].clone(), self.n),
                ArrayArg::from_raw_parts(lin[1].clone(), self.n),
                ArrayArg::from_raw_parts(lin[2].clone(), self.n),
            );
        }
    }

    unsafe fn launch_opsin(&self, is_a: bool) {
        let (lin, bl) = if is_a {
            (&self.lin_a, &self.blur_a)
        } else {
            (&self.lin_b, &self.blur_b)
        };
        unsafe {
            colors::opsin_dynamics_planar_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(lin[0].clone(), self.n),
                ArrayArg::from_raw_parts(lin[1].clone(), self.n),
                ArrayArg::from_raw_parts(lin[2].clone(), self.n),
                ArrayArg::from_raw_parts(bl[0].clone(), self.n),
                ArrayArg::from_raw_parts(bl[1].clone(), self.n),
                ArrayArg::from_raw_parts(bl[2].clone(), self.n),
                self.params.intensity_target,
            );
        }
    }

    /// Helper: H+V Gaussian blur with given sigma. Two kernel launches.
    /// `temp1` is reused as the H→V intermediate.
    fn blur_plane(&self, src: &cubecl::server::Handle, dst: &cubecl::server::Handle, sigma: f32) {
        unsafe {
            blur::horizontal_blur_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(src.clone(), self.n),
                ArrayArg::from_raw_parts(self.temp1.clone(), self.n),
                self.width,
                self.height,
                sigma,
            );
            blur::vertical_blur_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(self.temp1.clone(), self.n),
                ArrayArg::from_raw_parts(dst.clone(), self.n),
                self.width,
                self.height,
                sigma,
            );
        }
    }

    /// H+V blur with a caller-supplied scratch (so we can blur into
    /// `temp1` without overwriting it mid-pass).
    ///
    /// Uses the LUT-based kernels with the precomputed weight table
    /// for `sigma`. `sigma` must be one of [`BLUR_SIGMAS`]; the dispatch
    /// is exact-equality so a typo silently falls back to the old
    /// powf-per-tap path (no panic — preserves correctness).
    fn blur_plane_via(
        &self,
        src: &cubecl::server::Handle,
        dst: &cubecl::server::Handle,
        scratch: &cubecl::server::Handle,
        sigma: f32,
    ) {
        if let Some(kind) = blur_kind_for_sigma(sigma) {
            let table = &self.blur_tables[kind as usize];
            let table_len = self.blur_table_lens[kind as usize];
            let radius = self.blur_radii[kind as usize];
            unsafe {
                blur_lut::horizontal_blur_lut_kernel::launch_unchecked::<R>(
                    &self.client,
                    self.cube_count_1d(),
                    self.cube_dim_1d(),
                    ArrayArg::from_raw_parts(src.clone(), self.n),
                    ArrayArg::from_raw_parts(scratch.clone(), self.n),
                    ArrayArg::from_raw_parts(table.clone(), table_len),
                    self.width,
                    self.height,
                    radius,
                );
                blur_lut::vertical_blur_lut_kernel::launch_unchecked::<R>(
                    &self.client,
                    self.cube_count_1d(),
                    self.cube_dim_1d(),
                    ArrayArg::from_raw_parts(scratch.clone(), self.n),
                    ArrayArg::from_raw_parts(dst.clone(), self.n),
                    ArrayArg::from_raw_parts(table.clone(), table_len),
                    self.width,
                    self.height,
                    radius,
                );
            }
            return;
        }
        unsafe {
            blur::horizontal_blur_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(src.clone(), self.n),
                ArrayArg::from_raw_parts(scratch.clone(), self.n),
                self.width,
                self.height,
                sigma,
            );
            blur::vertical_blur_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(scratch.clone(), self.n),
                ArrayArg::from_raw_parts(dst.clone(), self.n),
                self.width,
                self.height,
                sigma,
            );
        }
    }

    /// 3-channel fused variant of [`Self::blur_plane_via`]. Uses the
    /// same scratch buffer pair (temp1/temp2 typed; here we need 3
    /// distinct H→V scratches per channel — caller must supply).
    /// Two launches total instead of six.
    ///
    /// LUT-based fast path: if `sigma` is one of [`BLUR_SIGMAS`], uses
    /// the precomputed weight table; otherwise falls back to the
    /// powf-per-tap path. The two paths produce bit-identical output
    /// on every backend we test (CUDA, DX12, HIP, WGPU Vulkan).
    #[allow(clippy::too_many_arguments)]
    fn blur_3ch_via(
        &self,
        src_x: &cubecl::server::Handle,
        src_y: &cubecl::server::Handle,
        src_b: &cubecl::server::Handle,
        dst_x: &cubecl::server::Handle,
        dst_y: &cubecl::server::Handle,
        dst_b: &cubecl::server::Handle,
        scratch_x: &cubecl::server::Handle,
        scratch_y: &cubecl::server::Handle,
        scratch_b: &cubecl::server::Handle,
        sigma: f32,
    ) {
        if let Some(kind) = blur_kind_for_sigma(sigma) {
            let table = &self.blur_tables[kind as usize];
            let table_len = self.blur_table_lens[kind as usize];
            let radius = self.blur_radii[kind as usize];
            unsafe {
                blur_lut::horizontal_blur_3ch_lut_kernel::launch_unchecked::<R>(
                    &self.client,
                    self.cube_count_1d(),
                    self.cube_dim_1d(),
                    ArrayArg::from_raw_parts(src_x.clone(), self.n),
                    ArrayArg::from_raw_parts(src_y.clone(), self.n),
                    ArrayArg::from_raw_parts(src_b.clone(), self.n),
                    ArrayArg::from_raw_parts(scratch_x.clone(), self.n),
                    ArrayArg::from_raw_parts(scratch_y.clone(), self.n),
                    ArrayArg::from_raw_parts(scratch_b.clone(), self.n),
                    ArrayArg::from_raw_parts(table.clone(), table_len),
                    self.width,
                    self.height,
                    radius,
                );
                blur_lut::vertical_blur_3ch_lut_kernel::launch_unchecked::<R>(
                    &self.client,
                    self.cube_count_1d(),
                    self.cube_dim_1d(),
                    ArrayArg::from_raw_parts(scratch_x.clone(), self.n),
                    ArrayArg::from_raw_parts(scratch_y.clone(), self.n),
                    ArrayArg::from_raw_parts(scratch_b.clone(), self.n),
                    ArrayArg::from_raw_parts(dst_x.clone(), self.n),
                    ArrayArg::from_raw_parts(dst_y.clone(), self.n),
                    ArrayArg::from_raw_parts(dst_b.clone(), self.n),
                    ArrayArg::from_raw_parts(table.clone(), table_len),
                    self.width,
                    self.height,
                    radius,
                );
            }
            return;
        }
        unsafe {
            crate::kernels::blur_3ch::horizontal_blur_3ch_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(src_x.clone(), self.n),
                ArrayArg::from_raw_parts(src_y.clone(), self.n),
                ArrayArg::from_raw_parts(src_b.clone(), self.n),
                ArrayArg::from_raw_parts(scratch_x.clone(), self.n),
                ArrayArg::from_raw_parts(scratch_y.clone(), self.n),
                ArrayArg::from_raw_parts(scratch_b.clone(), self.n),
                self.width,
                self.height,
                sigma,
            );
            crate::kernels::blur_3ch::vertical_blur_3ch_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(scratch_x.clone(), self.n),
                ArrayArg::from_raw_parts(scratch_y.clone(), self.n),
                ArrayArg::from_raw_parts(scratch_b.clone(), self.n),
                ArrayArg::from_raw_parts(dst_x.clone(), self.n),
                ArrayArg::from_raw_parts(dst_y.clone(), self.n),
                ArrayArg::from_raw_parts(dst_b.clone(), self.n),
                self.width,
                self.height,
                sigma,
            );
        }
    }

    fn copy_plane(&self, src: &cubecl::server::Handle, dst: &cubecl::server::Handle) {
        unsafe {
            frequency::copy_plane_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(src.clone(), self.n),
                ArrayArg::from_raw_parts(dst.clone(), self.n),
            );
        }
    }

    fn zero_plane(&self, dst: &cubecl::server::Handle) {
        unsafe {
            frequency::zero_plane_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(dst.clone(), self.n),
            );
        }
    }

    fn subtract_arrays(
        &self,
        src1: &cubecl::server::Handle,
        src2: &cubecl::server::Handle,
        dst: &cubecl::server::Handle,
    ) {
        unsafe {
            frequency::subtract_arrays_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(src1.clone(), self.n),
                ArrayArg::from_raw_parts(src2.clone(), self.n),
                ArrayArg::from_raw_parts(dst.clone(), self.n),
            );
        }
    }

    /// Frequency separation (LF/MF, MF/HF, HF/UHF) for one of the two
    /// image sides. Mirrors CPU `psycho::separate_frequencies` — see that
    /// function for the algorithm.
    fn separate_frequencies(&mut self, is_a: bool) {
        // Borrow split: take cloned handles up front so we can mutate
        // freq_*[1][0]/[1][1] in-place via copy at the UHF step.
        let lin = if is_a {
            self.lin_a.clone()
        } else {
            self.lin_b.clone()
        };
        let freq = if is_a { &self.freq_a } else { &self.freq_b };

        // ── Step 1: LF (low-pass) and MF = XYB − LF ──
        // Fused 3-channel LF blur (2 launches instead of 6). Uses
        // temp1/temp2/mask_scratch as H→V scratches.
        self.blur_3ch_via(
            &lin[0],
            &lin[1],
            &lin[2],
            &freq[3][0],
            &freq[3][1],
            &freq[3][2],
            &self.temp1.clone(),
            &self.temp2.clone(),
            &self.mask_scratch.clone(),
            SIGMA_LF,
        );
        // MF = XYB − LF (still per-channel; small kernel).
        for ch in 0..3 {
            self.subtract_arrays(&lin[ch], &freq[3][ch], &freq[2][ch]);
        }
        // xyb_low_freq_to_vals on LF — CPU `xyb_low_freq_to_vals`.
        unsafe {
            frequency::xyb_low_freq_to_vals_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(freq[3][0].clone(), self.n),
                ArrayArg::from_raw_parts(freq[3][1].clone(), self.n),
                ArrayArg::from_raw_parts(freq[3][2].clone(), self.n),
            );
        }

        // ── Step 2: MF/HF separation ──
        // T_x.B (2026-05-17): fuse the 3 SIGMA_HF blurs into ONE
        // 3-channel call (2 launches vs 6 — H+V each handle all
        // channels at once). Outputs land in scratch planes
        // freq[0][0..3], which UHF will overwrite in Step 3 so they're
        // free here. Scratches for the H-pass are temp1 / temp2 /
        // mask_scratch.
        self.blur_3ch_via(
            &freq[2][0],
            &freq[2][1],
            &freq[2][2],
            &freq[0][0],
            &freq[0][1],
            &freq[0][2],
            &self.temp1.clone(),
            &self.temp2.clone(),
            &self.mask_scratch.clone(),
            SIGMA_HF,
        );
        // X (ch=0): HF_X = orig - blur, MF_X = remove_range(blur, REMOVE_MF_RANGE)
        unsafe {
            frequency::split_band_remove_inplace_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(freq[2][0].clone(), self.n),
                ArrayArg::from_raw_parts(freq[0][0].clone(), self.n),
                ArrayArg::from_raw_parts(freq[1][0].clone(), self.n),
                REMOVE_MF_RANGE,
            );
        }
        // Y (ch=1): HF_Y = orig - blur, MF_Y = amplify_range(blur, ADD_MF_RANGE)
        unsafe {
            frequency::split_band_amplify_inplace_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(freq[2][1].clone(), self.n),
                ArrayArg::from_raw_parts(freq[0][1].clone(), self.n),
                ArrayArg::from_raw_parts(freq[1][1].clone(), self.n),
                ADD_MF_RANGE,
            );
        }
        // B (ch=2): copy blurred → MF_B (no HF for B).
        self.copy_plane(&freq[0][2].clone(), &freq[2][2]);

        // suppress_x_by_y(HF_y → HF_x)
        unsafe {
            frequency::suppress_x_by_y_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(freq[1][0].clone(), self.n),
                ArrayArg::from_raw_parts(freq[1][1].clone(), self.n),
                SUPPRESS_XY,
            );
        }

        // ── Step 3: HF/UHF separation ──
        // X (ch=0): blur(HF_X) → temp1; split → UHF_X (freq[0][0]),
        //           final HF_X (temp2); copy temp2 → freq[1][0].
        self.blur_plane_via(
            &freq[1][0],
            &self.temp1.clone(),
            &self.mask_scratch.clone(),
            SIGMA_UHF,
        );
        unsafe {
            frequency::split_uhf_hf_x_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(freq[1][0].clone(), self.n),
                ArrayArg::from_raw_parts(self.temp1.clone(), self.n),
                ArrayArg::from_raw_parts(freq[0][0].clone(), self.n),
                ArrayArg::from_raw_parts(self.temp2.clone(), self.n),
                REMOVE_UHF_RANGE,
                REMOVE_HF_RANGE,
            );
        }
        self.copy_plane(&self.temp2.clone(), &freq[1][0]);

        // Y (ch=1): same shape, Y kernel with maximum_clamp + amplify_range.
        self.blur_plane_via(
            &freq[1][1],
            &self.temp1.clone(),
            &self.mask_scratch.clone(),
            SIGMA_UHF,
        );
        unsafe {
            frequency::split_uhf_hf_y_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(freq[1][1].clone(), self.n),
                ArrayArg::from_raw_parts(self.temp1.clone(), self.n),
                ArrayArg::from_raw_parts(freq[0][1].clone(), self.n),
                ArrayArg::from_raw_parts(self.temp2.clone(), self.n),
            );
        }
        self.copy_plane(&self.temp2.clone(), &freq[1][1]);
    }

    /// Compute the AC half of the diffmap: 6 Malta diffs + 2 L2-asym
    /// (HF X/Y) + 3 L2 (MF X/Y/B). Mirrors CPU `compute_psycho_diff_malta`.
    fn compute_psycho_diff(&self) {
        let asym = self.params.hf_asymmetry as f64;
        let sqrt_asym = asym.sqrt();

        // Index conventions: freq[k] where k ∈ {0=UHF, 1=HF, 2=MF, 3=LF};
        //                    freq[k][0]=X, freq[k][1]=Y, freq[k][2]=B.

        // UHF Y (use_lf=false, 9-tap = "_hf" kernel)
        let (g, l, n1) = malta_norm(W_UHF_MALTA * asym, W_UHF_MALTA / asym, NORM1_UHF, false);
        self.zero_plane(&self.block_diff_ac[1]);
        self.malta_hf(
            &self.freq_a[0][1],
            &self.freq_b[0][1],
            &self.block_diff_ac[1],
            g,
            l,
            n1,
        );

        // UHF X
        let (g, l, n1) = malta_norm(
            W_UHF_MALTA_X * asym,
            W_UHF_MALTA_X / asym,
            NORM1_UHF_X,
            false,
        );
        self.zero_plane(&self.block_diff_ac[0]);
        self.malta_hf(
            &self.freq_a[0][0],
            &self.freq_b[0][0],
            &self.block_diff_ac[0],
            g,
            l,
            n1,
        );

        // HF Y (use_lf=true, 5-tap = "_lf" kernel)
        let (g, l, n1) = malta_norm(
            W_HF_MALTA * sqrt_asym,
            W_HF_MALTA / sqrt_asym,
            NORM1_HF,
            true,
        );
        self.malta_lf(
            &self.freq_a[1][1],
            &self.freq_b[1][1],
            &self.block_diff_ac[1],
            g,
            l,
            n1,
        );

        // HF X
        let (g, l, n1) = malta_norm(
            W_HF_MALTA_X * sqrt_asym,
            W_HF_MALTA_X / sqrt_asym,
            NORM1_HF_X,
            true,
        );
        self.malta_lf(
            &self.freq_a[1][0],
            &self.freq_b[1][0],
            &self.block_diff_ac[0],
            g,
            l,
            n1,
        );

        // MF Y (symmetric, use_lf=true)
        let (g, l, n1) = malta_norm(W_MF_MALTA, W_MF_MALTA, NORM1_MF, true);
        self.malta_lf(
            &self.freq_a[2][1],
            &self.freq_b[2][1],
            &self.block_diff_ac[1],
            g,
            l,
            n1,
        );

        // MF X
        let (g, l, n1) = malta_norm(W_MF_MALTA_X, W_MF_MALTA_X, NORM1_MF_X, true);
        self.malta_lf(
            &self.freq_a[2][0],
            &self.freq_b[2][0],
            &self.block_diff_ac[0],
            g,
            l,
            n1,
        );

        // L2_asym on HF X (WMUL[0]) and HF Y (WMUL[1])
        self.l2_diff_asym(
            &self.freq_a[1][0],
            &self.freq_b[1][0],
            &self.block_diff_ac[0],
            (WMUL[0] as f32) * self.params.hf_asymmetry,
            (WMUL[0] as f32) / self.params.hf_asymmetry,
        );
        self.l2_diff_asym(
            &self.freq_a[1][1],
            &self.freq_b[1][1],
            &self.block_diff_ac[1],
            (WMUL[1] as f32) * self.params.hf_asymmetry,
            (WMUL[1] as f32) / self.params.hf_asymmetry,
        );
        // WMUL[2] = 0.0, skip HF B.

        // L2 on MF X (WMUL[3]) and MF Y (WMUL[4]) — accumulate.
        self.l2_diff(
            &self.freq_a[2][0],
            &self.freq_b[2][0],
            &self.block_diff_ac[0],
            WMUL[3] as f32,
        );
        self.l2_diff(
            &self.freq_a[2][1],
            &self.freq_b[2][1],
            &self.block_diff_ac[1],
            WMUL[4] as f32,
        );

        // L2 on MF B (WMUL[5]) — write-only (block_diff_ac[2] hasn't been touched yet).
        self.l2_diff_write(
            &self.freq_a[2][2],
            &self.freq_b[2][2],
            &self.block_diff_ac[2],
            WMUL[5] as f32,
        );
    }

    /// DC contributions: per-channel `WMUL[6+ch] · (LF_a[ch] − LF_b[ch])²`
    /// written into `block_diff_dc[ch]`. CPU folds this into
    /// `combine_channels_to_diffmap_fused`; we do it as a separate pass.
    fn compute_dc_diff(&self) {
        for ch in 0..3 {
            self.l2_diff_write(
                &self.freq_a[3][ch],
                &self.freq_b[3][ch],
                &self.block_diff_dc[ch],
                WMUL[6 + ch] as f32,
            );
        }
    }

    /// CPU `compute_mask_from_hf_uhf`: combine UHF+HF → diff_precompute →
    /// blur σ=2.7 → fuzzy_erosion. Also accumulates mask-to-error for Y.
    /// Reference-side mask pipeline: combine(HF_a, UHF_a) →
    /// diff_precompute → blur(σ=2.7) → `cached_blurred_a`, then
    /// fuzzy_erosion(cached_blurred_a) → `self.mask`. Both buffers are
    /// reusable across many `compute_with_reference` calls.
    fn compute_mask_pipeline_reference_only(&self) {
        unsafe {
            masking::combine_channels_for_masking_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(self.freq_a[1][0].clone(), self.n),
                ArrayArg::from_raw_parts(self.freq_a[0][0].clone(), self.n),
                ArrayArg::from_raw_parts(self.freq_a[1][1].clone(), self.n),
                ArrayArg::from_raw_parts(self.freq_a[0][1].clone(), self.n),
                ArrayArg::from_raw_parts(self.temp1.clone(), self.n),
            );
            masking::diff_precompute_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(self.temp1.clone(), self.n),
                ArrayArg::from_raw_parts(self.mask_scratch.clone(), self.n),
            );
        }
        // Blur mask_scratch (image-A combined+precompute) → cached_blurred_a;
        // use temp1 as H-pass scratch (the diff_precompute output above no
        // longer needs it).
        self.blur_plane_via(
            &self.mask_scratch.clone(),
            &self.cached_blurred_a.clone(),
            &self.temp1.clone(),
            MASK_RADIUS,
        );
        unsafe {
            masking::fuzzy_erosion_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(self.cached_blurred_a.clone(), self.n),
                ArrayArg::from_raw_parts(self.mask.clone(), self.n),
                self.width,
                self.height,
            );
        }
    }

    /// Distorted-side mask pipeline: combine(HF_b, UHF_b) →
    /// diff_precompute → blur(σ=2.7) → `mask_scratch`, then
    /// `mask_to_error_mul(cached_blurred_a, mask_scratch, block_diff_ac[1])`.
    /// Assumes `cached_blurred_a` is populated by an earlier
    /// [`compute_mask_pipeline_reference_only`].
    fn compute_mask_pipeline_distorted_only(&self) {
        unsafe {
            masking::combine_channels_for_masking_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(self.freq_b[1][0].clone(), self.n),
                ArrayArg::from_raw_parts(self.freq_b[0][0].clone(), self.n),
                ArrayArg::from_raw_parts(self.freq_b[1][1].clone(), self.n),
                ArrayArg::from_raw_parts(self.freq_b[0][1].clone(), self.n),
                ArrayArg::from_raw_parts(self.mask_scratch.clone(), self.n),
            );
            masking::diff_precompute_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(self.mask_scratch.clone(), self.n),
                ArrayArg::from_raw_parts(self.temp2.clone(), self.n),
            );
        }
        // Blur temp2 → mask_scratch (using diffmap_buf as H-pass scratch
        // — not written until `launch_compute_diffmap` runs later).
        self.blur_plane_via(
            &self.temp2.clone(),
            &self.mask_scratch.clone(),
            &self.diffmap_buf.clone(),
            MASK_RADIUS,
        );
        // block_diff_ac[1] += MASK_TO_ERROR_MUL · (cached_blurred_a − mask_scratch)²
        unsafe {
            masking::mask_to_error_mul_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(self.cached_blurred_a.clone(), self.n),
                ArrayArg::from_raw_parts(self.mask_scratch.clone(), self.n),
                ArrayArg::from_raw_parts(self.block_diff_ac[1].clone(), self.n),
            );
        }
    }

    fn compute_mask_pipeline_full(&self) {
        self.compute_mask_pipeline_reference_only();
        self.compute_mask_pipeline_distorted_only();
    }

    unsafe fn launch_compute_diffmap(&self) {
        unsafe {
            diffmap::compute_diffmap_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(self.mask.clone(), self.n),
                ArrayArg::from_raw_parts(self.block_diff_dc[0].clone(), self.n),
                ArrayArg::from_raw_parts(self.block_diff_dc[1].clone(), self.n),
                ArrayArg::from_raw_parts(self.block_diff_dc[2].clone(), self.n),
                ArrayArg::from_raw_parts(self.block_diff_ac[0].clone(), self.n),
                ArrayArg::from_raw_parts(self.block_diff_ac[1].clone(), self.n),
                ArrayArg::from_raw_parts(self.block_diff_ac[2].clone(), self.n),
                ArrayArg::from_raw_parts(self.diffmap_buf.clone(), self.n),
                self.params.xmul,
            );
        }
    }

    fn malta_hf(
        &self,
        a: &cubecl::server::Handle,
        b: &cubecl::server::Handle,
        acc: &cubecl::server::Handle,
        norm2_0gt1: f32,
        norm2_0lt1: f32,
        norm1: f32,
    ) {
        unsafe {
            malta::malta_diff_map_hf_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_2d(),
                self.cube_dim_2d(),
                ArrayArg::from_raw_parts(a.clone(), self.n),
                ArrayArg::from_raw_parts(b.clone(), self.n),
                ArrayArg::from_raw_parts(acc.clone(), self.n),
                self.width,
                self.height,
                norm2_0gt1,
                norm2_0lt1,
                norm1,
            );
        }
    }

    fn malta_lf(
        &self,
        a: &cubecl::server::Handle,
        b: &cubecl::server::Handle,
        acc: &cubecl::server::Handle,
        norm2_0gt1: f32,
        norm2_0lt1: f32,
        norm1: f32,
    ) {
        unsafe {
            malta::malta_diff_map_lf_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_2d(),
                self.cube_dim_2d(),
                ArrayArg::from_raw_parts(a.clone(), self.n),
                ArrayArg::from_raw_parts(b.clone(), self.n),
                ArrayArg::from_raw_parts(acc.clone(), self.n),
                self.width,
                self.height,
                norm2_0gt1,
                norm2_0lt1,
                norm1,
            );
        }
    }

    fn l2_diff(
        &self,
        a: &cubecl::server::Handle,
        b: &cubecl::server::Handle,
        acc: &cubecl::server::Handle,
        weight: f32,
    ) {
        unsafe {
            diffmap::l2_diff_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(a.clone(), self.n),
                ArrayArg::from_raw_parts(b.clone(), self.n),
                ArrayArg::from_raw_parts(acc.clone(), self.n),
                weight,
            );
        }
    }

    fn l2_diff_write(
        &self,
        a: &cubecl::server::Handle,
        b: &cubecl::server::Handle,
        acc: &cubecl::server::Handle,
        weight: f32,
    ) {
        unsafe {
            diffmap::l2_diff_write_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(a.clone(), self.n),
                ArrayArg::from_raw_parts(b.clone(), self.n),
                ArrayArg::from_raw_parts(acc.clone(), self.n),
                weight,
            );
        }
    }

    fn l2_diff_asym(
        &self,
        a: &cubecl::server::Handle,
        b: &cubecl::server::Handle,
        acc: &cubecl::server::Handle,
        w_gt: f32,
        w_lt: f32,
    ) {
        unsafe {
            diffmap::l2_asym_diff_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(a.clone(), self.n),
                ArrayArg::from_raw_parts(b.clone(), self.n),
                ArrayArg::from_raw_parts(acc.clone(), self.n),
                w_gt,
                w_lt,
            );
        }
    }

    /// Multi-res supersample-add: add `src` (a half-resolution diffmap of
    /// dims `src_w × src_h`) into `self.diffmap_buf` with weight=0.5 and
    /// the libjxl K_HEURISTIC_MIXING_VALUE=0.3 attenuation:
    ///   `dst[i] = dst[i] · (1 − 0.3·0.5) + 0.5 · src[upsampled_i]`
    fn launch_add_supersampled_2x_from(
        &self,
        src: &cubecl::server::Handle,
        src_w: u32,
        src_h: u32,
    ) {
        let src_n = (src_w as usize) * (src_h as usize);
        unsafe {
            downscale::add_upsample_2x_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(self.diffmap_buf.clone(), self.n),
                ArrayArg::from_raw_parts(src.clone(), src_n),
                self.width,
                self.height,
                src_w,
                0.5_f32,
            );
        }
    }

    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Read the current diffmap back to host memory.
    pub fn copy_diffmap(&self) -> Vec<f32> {
        self.read_plane(&self.diffmap_buf)
    }

    /// True iff [`set_reference`] has been called and the reference-side
    /// state is valid.
    pub fn has_cached_reference(&self) -> bool {
        self.has_cached_reference
    }

    /// The active comparison parameters (last set via
    /// `compute_with_options` / `set_reference_with_options`). Defaults
    /// after construction.
    pub fn params(&self) -> &ButteraugliParams {
        &self.params
    }

    /// Read the diffmap into a caller-supplied buffer (no allocation).
    /// `dst.len()` must be ≥ `width × height`. Use this when scoring in
    /// a hot loop to skip the per-call `Vec` allocation that
    /// [`copy_diffmap`] does.
    pub fn copy_diffmap_to(&self, dst: &mut [f32]) -> Result<()> {
        let bytes = self
            .client
            .read_one(self.diffmap_buf.clone())
            .expect("read_one diffmap");
        let src = f32::from_bytes(&bytes);
        if dst.len() < src.len() {
            return Err(Error::DimensionMismatch {
                expected: src.len(),
                got: dst.len(),
            });
        }
        dst[..src.len()].copy_from_slice(src);
        Ok(())
    }

    /// Half-resolution sibling for the multi-resolution pass. Public so
    /// [`crate::ButteraugliBatch`] can reach into the cached reference
    /// state on both resolutions.
    pub fn half_res(&self) -> Option<&Self> {
        self.half_res.as_deref()
    }

    /// Cached reference frequency band `freq_a[band][channel]`. Returns
    /// the underlying CubeCL handle so a batched scorer can broadcast it
    /// against many distorted-side planes.
    pub fn cached_freq(&self, band: usize, channel: usize) -> &cubecl::server::Handle {
        &self.freq_a[band][channel]
    }

    /// Cached reference XYB plane.
    pub fn cached_xyb(&self, channel: usize) -> &cubecl::server::Handle {
        &self.lin_a[channel]
    }

    /// Cached blurred image-A mask plane (input to fuzzy_erosion AND
    /// to mask_to_error_mul).
    pub fn cached_blurred_a(&self) -> &cubecl::server::Handle {
        &self.cached_blurred_a
    }

    /// Cached fuzzy-erosion mask plane.
    pub fn cached_mask(&self) -> &cubecl::server::Handle {
        &self.mask
    }

    fn read_plane(&self, h: &cubecl::server::Handle) -> Vec<f32> {
        let bytes = self.client.read_one(h.clone()).expect("read_one plane");
        f32::from_bytes(&bytes).to_vec()
    }

    /// Debug: read the AC accumulator for one channel. Available after
    /// [`compute`].
    pub fn debug_block_diff_ac(&self, ch: usize) -> Vec<f32> {
        self.read_plane(&self.block_diff_ac[ch])
    }

    /// Debug: read the DC accumulator for one channel.
    pub fn debug_block_diff_dc(&self, ch: usize) -> Vec<f32> {
        self.read_plane(&self.block_diff_dc[ch])
    }

    /// Debug: read the fuzzy-erosion mask plane.
    pub fn debug_mask(&self) -> Vec<f32> {
        self.read_plane(&self.mask)
    }

    /// Debug: read one of the LF (low-frequency, vals-space) planes for
    /// one of the two image sides.
    pub fn debug_lf(&self, is_a: bool, ch: usize) -> Vec<f32> {
        let f = if is_a { &self.freq_a } else { &self.freq_b };
        self.read_plane(&f[3][ch])
    }

    /// Debug: read the per-channel HF / UHF / MF / LF plane (k ∈ 0..=3).
    pub fn debug_freq(&self, is_a: bool, k: usize, ch: usize) -> Vec<f32> {
        let f = if is_a { &self.freq_a } else { &self.freq_b };
        self.read_plane(&f[k][ch])
    }
}
