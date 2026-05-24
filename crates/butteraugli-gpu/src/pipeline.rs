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
    /// Width of the buffers and (for strip mode) of the image. For
    /// whole-image mode this equals the image width.
    width: u32,
    /// Height of the allocated buffers. For whole-image mode this is the
    /// image height; for strip mode it is `body_h_max + 2 * halo_h`,
    /// i.e. the per-strip working slab.
    height: u32,
    /// `width × height` (allocation size in f32 pixels).
    n: usize,

    /// Logical image height for strip mode. Equals `self.height` for
    /// whole-image mode. Used by `compute_strip` to bound the walk.
    image_h: u32,
    /// Per-strip body row count for strip mode (the inner band whose
    /// per-pixel diffmap is folded into the running aggregate). Equal
    /// to `self.height` for whole-image mode.
    body_h: u32,
    /// Halo rows above and below the body inside each strip. Zero in
    /// whole-image mode.
    halo_h: u32,

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
        // Defensive overflow check: `n * 3` is the upper bound on
        // sRGB input length (3 bytes per pixel). Pre-validating here
        // means downstream `vec![0_u32; n]` / `vec![0.0_f32; n]`
        // allocations can't surface a confusing alloc-failure for a
        // caller's mistake; they'll trip this expect first. Result
        // unused — kept for the side-effect panic.
        let _n_bytes = n
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
            // Whole-image construction: image_h == height, body_h ==
            // height, halo == 0. compute_strip detects whole-image mode
            // by halo_h == 0 and short-circuits to compute_with_options.
            image_h: height,
            body_h: height,
            halo_h: 0,
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
            // T_x.O (2026-05-17): pack_scratch is gone — packing now
            // writes directly into the pinned staging buffer reserved
            // each upload, saving one host-side R/W roundtrip per call.
            params: ButteraugliParams::default(),
            blur_tables,
            blur_radii,
            blur_table_lens,
        }
    }

    /// Allocate buffers sized for a `body_h + 2 × HALO_ROWS` strip,
    /// for processing a logical `image_w × image_h` image strip by
    /// strip.
    ///
    /// Each call to [`Butteraugli::compute_strip`] walks the image in
    /// `body_h` row-tall bands, populating halo rows from real image
    /// content (edge-clamped at image top/bottom). Body rows of the
    /// resulting diffmap are folded into running `(max, p3, p6, p12)`
    /// partials and the final score is identical to the whole-image
    /// path up to f64 reduction order.
    ///
    /// Use [`Butteraugli::new`] for whole-image mode; [`Butteraugli::new_strip`]
    /// trades a small per-strip launch-latency overhead for ~`image_h /
    /// (body_h + 2 × HALO_ROWS)` peak memory savings — well over 10× at
    /// 24 MP with sensible body sizes.
    ///
    /// **Constraints (MVP):**
    /// - Single-resolution only — the multi-resolution sibling that
    ///   `new_multires` allocates is not strip-stitched in this
    ///   revision. Use whole-image `new_multires` for the half-res
    ///   contribution if needed.
    /// - `set_reference` / `compute_with_reference` not yet supported.
    ///   Strip-mode `compute_strip` re-runs the full pipeline both
    ///   sides every call.
    ///
    /// Panics if `body_h == 0` or `image_w × (body_h + 2 × HALO_ROWS) ×
    /// 3` overflows `usize`.
    pub fn new_strip(client: ComputeClient<R>, image_w: u32, image_h: u32, body_h: u32) -> Self {
        assert!(body_h > 0, "body_h must be > 0");
        assert!(image_w > 0 && image_h > 0, "image dims must be > 0");
        let halo_h = crate::strip::HALO_ROWS;
        // If the image fits in a single strip (body_h + 2*halo >=
        // image_h) we still allocate the strip slab; the walker just
        // runs one strip whose body covers the whole image.
        let body_h_eff = body_h.min(image_h);
        let strip_h_total = body_h_eff
            .saturating_add(halo_h.saturating_mul(2))
            .min(image_h.saturating_add(halo_h.saturating_mul(2)));
        // Build a whole-image-style instance sized to (image_w,
        // strip_h_total), then patch the strip metadata in.
        let mut inst = Self::new(client, image_w, strip_h_total);
        inst.image_h = image_h;
        inst.body_h = body_h_eff;
        inst.halo_h = halo_h;
        inst
    }

    /// Unified [`MemoryMode`](crate::MemoryMode) constructor for the
    /// single-resolution path. butteraugli-gpu is **strip-preferred**:
    /// when Strip fits the VRAM cap, Auto picks Strip even if Full
    /// would also fit (Strip is 1.9-4.9× faster than whole-image on
    /// this crate per the bench at
    /// `benchmarks/butter_strip_vs_whole_2026-05-21.md`).
    ///
    /// - `MemoryMode::Auto` picks between Full and Strip via
    ///   [`crate::memory_mode::resolve_auto`].
    /// - `MemoryMode::Full` constructs via [`Self::new`].
    /// - `MemoryMode::Strip { h_body }` constructs via [`Self::new_strip`].
    ///   `h_body == None` auto-sizes within the cap.
    /// - `MemoryMode::Tile {..}` returns
    ///   [`Error::ModeUnsupported`](crate::Error::ModeUnsupported) —
    ///   the variant is reserved for a future implementation.
    ///
    /// Note: this constructor does NOT engage the half-resolution
    /// sibling. Use [`Self::new_multires_with_memory_mode`] for the
    /// CPU-butteraugli-default multi-resolution path (currently only
    /// supported with `MemoryMode::Full` or `Auto` since the
    /// half-res strip walker isn't implemented yet — Auto falls
    /// through to Full for the multi-res variant).
    pub fn new_with_memory_mode(
        client: ComputeClient<R>,
        width: u32,
        height: u32,
        mode: crate::MemoryMode,
    ) -> crate::Result<Self> {
        use crate::MemoryMode;
        use crate::memory_mode::{ResolvedMode, resolve_auto, vram_cap_bytes};
        match mode {
            MemoryMode::Full => Ok(Self::new(client, width, height)),
            MemoryMode::Strip { h_body } => {
                let body = h_body.unwrap_or_else(|| {
                    let cap = vram_cap_bytes();
                    crate::memory_mode::auto_strip_body_for(width, height, cap)
                });
                Ok(Self::new_strip(client, width, height, body))
            }
            MemoryMode::Tile { .. } => Err(crate::Error::ModeUnsupported("Tile")),
            MemoryMode::Auto => {
                let cap = vram_cap_bytes();
                match resolve_auto(width, height, cap)? {
                    ResolvedMode::Full => Ok(Self::new(client, width, height)),
                    ResolvedMode::Strip { h_body } => {
                        Ok(Self::new_strip(client, width, height, h_body))
                    }
                }
            }
        }
    }

    /// Multi-resolution [`MemoryMode`](crate::MemoryMode) constructor.
    /// `MemoryMode::Full` allocates whole-image planes for both the
    /// full-res and half-res passes; `MemoryMode::Strip { h_body }`
    /// allocates strip-sized planes for BOTH the full-res and half-res
    /// passes (the strip-multires walker runs both in tandem with a
    /// halved halo on the half-res side).
    /// `MemoryMode::Auto` resolves to Strip when it fits the VRAM cap
    /// (butter is strip-preferred), Full otherwise.
    pub fn new_multires_with_memory_mode(
        client: ComputeClient<R>,
        width: u32,
        height: u32,
        mode: crate::MemoryMode,
    ) -> crate::Result<Self> {
        use crate::MemoryMode;
        use crate::memory_mode::{ResolvedMode, resolve_auto, vram_cap_bytes};
        match mode {
            MemoryMode::Full => Ok(Self::new_multires(client, width, height)),
            MemoryMode::Strip { h_body } => {
                let body = h_body.unwrap_or_else(|| {
                    let cap = vram_cap_bytes();
                    crate::memory_mode::auto_strip_body_for(width, height, cap)
                });
                Ok(Self::new_multires_strip(client, width, height, body))
            }
            MemoryMode::Tile { .. } => Err(crate::Error::ModeUnsupported("Tile")),
            MemoryMode::Auto => {
                let cap = vram_cap_bytes();
                match resolve_auto(width, height, cap)? {
                    ResolvedMode::Full => Ok(Self::new_multires(client, width, height)),
                    ResolvedMode::Strip { h_body } => {
                        Ok(Self::new_multires_strip(client, width, height, h_body))
                    }
                }
            }
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

    /// Construct a multi-resolution strip-mode `Butteraugli` instance.
    /// Builds a strip-mode full-res instance via [`Self::new_strip`]
    /// plus a strip-mode half-res sibling allocated at
    /// `(image_w.div_ceil(2), image_h.div_ceil(2))` with body
    /// `body_h.div_ceil(2)` and the same `HALO_ROWS` halo.
    ///
    /// The strip walker drives both instances in tandem: each full-res
    /// strip's body has a matching half-res strip body produced by
    /// 2× downsampling the full-res strip's linear-RGB slab. After the
    /// half-res strip's diffmap is computed, it's supersample-added
    /// into the full-res strip's diffmap BEFORE the body-band
    /// reduction.
    ///
    /// **Constraints**:
    /// - `body_h` is internally rounded DOWN to the nearest even value
    ///   so the half-res body alignment is exact. The minimum body_h
    ///   after rounding is 2 (caller passing `body_h = 1` will see a
    ///   panic from the inner `new_strip` call). Pass even values to
    ///   avoid surprise.
    /// - For very small images (`w < 16` or `h < 16`) the half-res
    ///   sibling is skipped — same threshold as [`Self::new_multires`]
    ///   — and the constructor degenerates to a single-resolution
    ///   strip instance.
    pub fn new_multires_strip(
        client: ComputeClient<R>,
        image_w: u32,
        image_h: u32,
        body_h: u32,
    ) -> Self {
        const MIN_SIZE_FOR_SUBSAMPLE: u32 = 16;
        // Round body_h DOWN to even so half-res body alignment is
        // exact. body_h_full = 2k → body_h_half = k; every body_top
        // is a multiple of 2k, also even.
        let body_h_even = (body_h / 2).max(1) * 2;
        let body_h_full = body_h_even.min(image_h);
        let mut full = Self::new_strip(client.clone(), image_w, image_h, body_h_full);
        if image_w >= MIN_SIZE_FOR_SUBSAMPLE && image_h >= MIN_SIZE_FOR_SUBSAMPLE {
            let half_w = image_w.div_ceil(2);
            let half_h = image_h.div_ceil(2);
            // Half-res body is body_h_full / 2 — exact because
            // body_h_full is even. The half-res HALO is still
            // HALO_ROWS (the LF/HF/UHF blurs operate in pixel-space
            // and have the same radii on the half-res image).
            let body_h_half = (body_h_full / 2).max(1);
            full.half_res = Some(Box::new(Self::new_strip(
                client, half_w, half_h, body_h_half,
            )));
        }
        full
    }

    /// Compute strip-by-strip butteraugli over the logical image
    /// configured at [`Butteraugli::new_strip`].
    ///
    /// `ref_srgb.len()` and `dist_srgb.len()` must equal `image_w ×
    /// image_h × 3`. Returns the same `(score, pnorm_3)` shape as
    /// [`Butteraugli::compute`], identical to the whole-image path up
    /// to f64 reduction order (verified < 1e-4 rel at 1024² in the
    /// parity tests).
    ///
    /// Panics if called on a whole-image instance (one constructed via
    /// [`Butteraugli::new`] / [`Butteraugli::new_multires`]); use
    /// [`Butteraugli::compute`] there.
    pub fn compute_strip(
        &mut self,
        ref_srgb: &[u8],
        dist_srgb: &[u8],
    ) -> Result<GpuButteraugliResult> {
        self.compute_strip_with_options(ref_srgb, dist_srgb, &ButteraugliParams::default())
    }

    /// `compute_strip` with runtime-tunable params. Same constraints
    /// and validation rules as [`Self::compute_with_options`].
    pub fn compute_strip_with_options(
        &mut self,
        ref_srgb: &[u8],
        dist_srgb: &[u8],
        params: &ButteraugliParams,
    ) -> Result<GpuButteraugliResult> {
        if self.halo_h == 0 {
            // Whole-image instance — caller probably meant `compute`.
            return Err(Error::StripModeUnsupported(
                "compute_strip-on-whole-image-instance (use Butteraugli::new_strip)",
            ));
        }
        if self.half_res.is_some() {
            crate::strip::run_strip_pipeline_multires(
                self,
                ref_srgb,
                dist_srgb,
                self.width,
                self.image_h,
                self.body_h,
                self.halo_h,
                params,
            )
        } else {
            crate::strip::run_strip_pipeline(
                self,
                ref_srgb,
                dist_srgb,
                self.width,
                self.image_h,
                self.body_h,
                self.halo_h,
                params,
            )
        }
    }

    /// True if this instance was constructed via
    /// [`Butteraugli::new_strip`] (and therefore expects
    /// [`Butteraugli::compute_strip`] rather than
    /// [`Butteraugli::compute`]).
    pub fn is_strip_mode(&self) -> bool {
        self.halo_h > 0
    }

    /// Logical image height (returned even in strip mode where
    /// `self.height` is the per-strip slab height, not the image
    /// height). For whole-image mode this matches `self.height`.
    pub fn image_height(&self) -> u32 {
        self.image_h
    }

    /// Logical strip body row count (one per call to the inner
    /// pipeline, NOT including halo).
    pub fn strip_body_h(&self) -> u32 {
        self.body_h
    }

    /// Per-strip halo row count (HALO_ROWS for strip mode, 0 for
    /// whole-image).
    pub fn strip_halo_h(&self) -> u32 {
        self.halo_h
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
        if self.halo_h > 0 {
            // Strip-mode instance — caller probably meant compute_strip.
            // Without this guard the check_dims rejection that follows
            // would surface a misleading "expected N×slab_h×3 bytes"
            // message (slab geometry, not image geometry).
            return Err(Error::StripModeUnsupported("compute"));
        }
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
        let expected = self.n * 3;
        if srgb.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: srgb.len(),
            });
        }
        let pinned_len = self.n * 4;
        let mut staging = self.client.reserve_staging(&[pinned_len]);
        let mut bytes = staging
            .pop()
            .expect("reserve_staging returned no buffers");
        {
            let dst: &mut [u8] = &mut bytes;
            debug_assert_eq!(dst.len(), pinned_len);
            for (chunk_out, triple) in dst.chunks_exact_mut(4).zip(srgb.chunks_exact(3)) {
                chunk_out[0] = triple[0];
                chunk_out[1] = triple[1];
                chunk_out[2] = triple[2];
                chunk_out[3] = 0;
            }
        }
        Ok(self.client.create(bytes))
    }

    /// Compute against pre-uploaded packed-u32 device handles —
    /// upload-once Phase 4 entry point. Skips the
    /// `client.reserve_staging` + byte-pack work that
    /// [`Self::compute`] does internally, letting one
    /// `(ref, dist)` upload feed several metrics on the same client.
    ///
    /// Handle layout MUST be the packed-u32 form produced by
    /// [`Self::pack_srgb_into_packed_u32_handle`] (one `u32` per
    /// pixel, `R | G<<8 | B<<16`, length `width × height`). The
    /// handle is expected to live on the same cubecl client that
    /// constructed this `Butteraugli<R>`.
    pub fn compute_handles(
        &mut self,
        ref_handle: &cubecl::server::Handle,
        dis_handle: &cubecl::server::Handle,
    ) -> Result<GpuButteraugliResult> {
        self.compute_handles_with_options(ref_handle, dis_handle, &ButteraugliParams::default())
    }

    /// Mode-explicit counterpart of [`Self::compute_handles`] — same
    /// param-validation semantics as [`Self::compute_with_options`].
    pub fn compute_handles_with_options(
        &mut self,
        ref_handle: &cubecl::server::Handle,
        dis_handle: &cubecl::server::Handle,
        params: &ButteraugliParams,
    ) -> Result<GpuButteraugliResult> {
        if self.halo_h > 0 {
            return Err(Error::StripModeUnsupported("compute_handles"));
        }
        validate_params(params)?;
        self.set_params_recursive(params);
        self.install_packed_handle_and_srgb_to_linear(true, ref_handle);
        self.install_packed_handle_and_srgb_to_linear(false, dis_handle);
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
        if self.halo_h > 0 {
            // Strip-mode instances can't cache reference-side
            // intermediates: the strip walker rewrites lin_a/freq_a
            // per strip, so reusing them across compute_strip calls
            // would mix strips. Surface this as a clear error instead
            // of silently corrupting state.
            return Err(Error::StripModeUnsupported("set_reference"));
        }
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

    /// Cache the reference image, taking the reference side as **3
    /// already-uploaded planar f32 GPU handles** in linear-RGB space
    /// (each `width × height` f32 values, in `[0, 1]` pre-opsin). Skips
    /// the sRGB-bytes upload + sRGB→linear GPU conversion that
    /// [`set_reference`] does internally.
    ///
    /// The provided handles' contents are **mutated in place** by the
    /// opsin / frequency separation kernels (which overwrite the
    /// `lin_a` planes per the existing pipeline). If the caller wants
    /// to keep the original linear-RGB values intact, clone the handles
    /// before calling — cubecl `Handle` is reference-counted so a clone
    /// is cheap, but the underlying GPU buffer will then be allocated
    /// separately by the next downstream consumer.
    ///
    /// **Use case**: encoder rate-distortion search where the encoder
    /// already produces the source image as planar linear-RGB GPU
    /// planes. Combined with
    /// [`compute_with_reference_from_linear_planes`], the whole
    /// reference + distorted pipeline can run without ever touching
    /// sRGB-u8 — eliminates the host-side linear→sRGB pack
    /// (~5-15 ms / iter at 1 MP with the LUT path; ~150-300 ms / iter
    /// with the scalar `powf` path).
    ///
    /// Each caller-supplied plane MUST hold exactly `width × height`
    /// f32 values in row-major order, contiguous, no padding. The
    /// kernels assume a tight stride; pass tight planes only.
    ///
    /// Gated behind the `internals` cargo feature (mirrors
    /// [`compute_with_reference_from_linear_planes`]). Not part of the
    /// stable API; field layout / kernel order may shift with
    /// internal pipeline refactors.
    #[cfg(feature = "internals")]
    pub fn set_reference_from_linear_planes(
        &mut self,
        ref_r: cubecl::server::Handle,
        ref_g: cubecl::server::Handle,
        ref_b: cubecl::server::Handle,
    ) -> Result<()> {
        self.set_reference_from_linear_planes_with_options(
            ref_r,
            ref_g,
            ref_b,
            &ButteraugliParams::default(),
        )
    }

    /// Variant of [`set_reference_from_linear_planes`] that takes
    /// explicit [`ButteraugliParams`]. All subsequent
    /// `compute_with_reference*` calls reuse those params — call again
    /// to change them.
    #[cfg(feature = "internals")]
    pub fn set_reference_from_linear_planes_with_options(
        &mut self,
        ref_r: cubecl::server::Handle,
        ref_g: cubecl::server::Handle,
        ref_b: cubecl::server::Handle,
        params: &ButteraugliParams,
    ) -> Result<()> {
        validate_params(params)?;
        self.set_params_recursive(params);
        // Install caller-supplied linear-RGB plane handles into lin_a.
        // The opsin / frequency / mask kernels will overwrite these
        // in-place — see this struct's pipeline documentation for the
        // chain. cubecl `Handle` is reference-counted so the swap is a
        // pointer-level operation; the underlying GPU buffers are
        // adopted by `self`.
        self.lin_a[0] = ref_r;
        self.lin_a[1] = ref_g;
        self.lin_a[2] = ref_b;
        // Downsample full-res linear into the half-res sibling BEFORE
        // opsin overwrites lin_a in place. Mirrors run_pipeline_from_linear
        // for the reference-only side.
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
        if self.halo_h > 0 {
            return Err(Error::StripModeUnsupported("compute_with_reference"));
        }
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
    /// Install a caller-supplied packed-u32 device handle as the
    /// ref/dist input AND run the sRGB→linear kernel. Handle layout
    /// MUST match what [`Self::populate_linear_from_srgb`] produces:
    /// `width × height` `u32`s, each `R | G<<8 | B<<16` (alpha
    /// unused). After return `lin_a` (or `lin_b`) holds the linear
    /// RGB planes and the rest of the pipeline can run.
    fn install_packed_handle_and_srgb_to_linear(
        &mut self,
        is_a: bool,
        handle: &cubecl::server::Handle,
    ) {
        if is_a {
            self.src_u8_a = handle.clone();
        } else {
            self.src_u8_b = handle.clone();
        }
        unsafe {
            self.launch_srgb_to_linear(is_a);
        }
    }

    /// Strip-mode helper used by [`crate::strip::run_strip_pipeline`].
    /// Sets the active comparison parameters on `self` (no per-strip
    /// state separate from the existing whole-image storage).
    pub(crate) fn set_params(&mut self, params: ButteraugliParams) {
        self.params = params;
    }

    /// Strip-mode helper: returns a reference to the underlying CubeCL
    /// client so the strip walker can drive `read_one` for per-strip
    /// diffmap reduction.
    pub(crate) fn client_ref(&self) -> &ComputeClient<R> {
        &self.client
    }

    /// Strip-mode helper: a clone of the diffmap-buf handle for the
    /// just-completed strip pipeline run.
    pub(crate) fn diffmap_buf_handle(&self) -> cubecl::server::Handle {
        self.diffmap_buf.clone()
    }

    /// Strip-mode helper: pack an `image_w × image_h × 3` sRGB-u8
    /// source into the (already-allocated) `src_u8_a` / `src_u8_b`
    /// handle for one strip. Halo rows are populated with edge-clamped
    /// image rows (mirrors the blur kernels' `saturating_sub` /
    /// `min(h - 1)` edge handling, so body-row outputs match the
    /// whole-image path exactly).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn upload_strip_srgb(
        &mut self,
        is_a: bool,
        srgb: &[u8],
        image_w: u32,
        image_h: u32,
        body_top_img: u32,
        strip_h_total: u32,
        halo_top: u32,
    ) {
        debug_assert_eq!(srgb.len(), (image_w as usize) * (image_h as usize) * 3);
        debug_assert_eq!(image_w, self.width);
        debug_assert!(strip_h_total <= self.height);

        let pinned_len = (image_w as usize) * (strip_h_total as usize) * 4;
        let mut staging = self.client.reserve_staging(&[pinned_len]);
        let mut bytes = staging
            .pop()
            .expect("reserve_staging returned no buffers");
        {
            let dst: &mut [u8] = &mut bytes;
            debug_assert_eq!(dst.len(), pinned_len);
            crate::strip::pack_strip_srgb_into(
                dst,
                srgb,
                image_w,
                image_h,
                body_top_img,
                strip_h_total,
                halo_top,
            );
        }
        let handle = self.client.create(bytes);
        if is_a {
            self.src_u8_a = handle;
        } else {
            self.src_u8_b = handle;
        }
        unsafe {
            self.launch_srgb_to_linear(is_a);
        }
    }

    /// Strip-mode helper: drive the same kernel chain as `compute`
    /// (both sides; no cached-reference fast path) on the currently
    /// loaded strip planes. `strip_h_total` is the populated height of
    /// the strip; this method temporarily clamps `self.height` to that
    /// value for the duration of the pipeline so kernels launched with
    /// `cube_count_1d / 2d` cover the strip but not the unused
    /// trailing rows of the slab.
    ///
    /// This method does NOT recurse into `self.half_res` — the
    /// multires-strip walker drives the full-res and half-res
    /// instances side-by-side instead (see
    /// [`crate::strip::run_strip_pipeline_multires`]). For a
    /// single-resolution strip instance, `self.half_res` is `None`.
    pub(crate) fn run_strip_pipeline_compute(&mut self, strip_h_total: u32) {
        let saved_height = self.height;
        let saved_n = self.n;
        // Clamp height + n to the strip's actually-populated slice.
        // self.height drives every cube_count_1d / cube_count_2d call;
        // self.n drives every ArrayArg::from_raw_parts length. After
        // the pipeline pass we restore the slab-sized values so that
        // the next strip's upload sees the full pinned-len.
        self.height = strip_h_total;
        self.n = (self.width as usize) * (strip_h_total as usize);

        self.apply_opsin(true);
        self.apply_opsin(false);
        self.separate_frequencies(true);
        self.separate_frequencies(false);
        self.compute_psycho_diff();
        self.compute_dc_diff();
        self.compute_mask_pipeline_full();
        unsafe {
            self.launch_compute_diffmap();
        }

        self.height = saved_height;
        self.n = saved_n;
    }

    /// Strip-multires helper: downsample the full-res strip's
    /// `lin_a` / `lin_b` planes into the half-res sibling's `lin_a`
    /// / `lin_b` planes. Operates slab-to-slab — both instances must
    /// have their `height` clamped to their respective strip-totals
    /// before this call so that the fused 3-channel downsample kernel
    /// covers exactly the populated rows.
    ///
    /// `full_strip_h_total` and `half_strip_h_total` are the populated
    /// strip slab heights (full = 2 × half + parity overhang at the
    /// bottom edge). The constructor guarantees these line up.
    pub(crate) fn downsample_full_strip_into_half(
        &self,
        half: &Butteraugli<R>,
        full_strip_h_total: u32,
        half_strip_h_total: u32,
    ) {
        // Reuse the existing `downsample_2x_3ch_kernel` — it operates
        // on plain planar buffers; the caller swears the half-res slab
        // is half the size of the full-res slab to within rounding.
        let half_n = (half.width as usize) * (half_strip_h_total as usize);
        const TPB: u32 = 256;
        let cubes = (half_n as u32).div_ceil(TPB);
        let block = CubeDim::new_1d(TPB);
        let full_n = (self.width as usize) * (full_strip_h_total as usize);
        for &is_a in &[true, false] {
            let (full_lin, half_lin) = if is_a {
                (&self.lin_a, &half.lin_a)
            } else {
                (&self.lin_b, &half.lin_b)
            };
            // CubeCount/Dim aren't Copy — recreate per launch.
            let dim = CubeCount::Static(cubes, 1, 1);
            unsafe {
                downscale::downsample_2x_3ch_kernel::launch_unchecked::<R>(
                    &self.client,
                    dim,
                    block,
                    ArrayArg::from_raw_parts(full_lin[0].clone(), full_n),
                    ArrayArg::from_raw_parts(full_lin[1].clone(), full_n),
                    ArrayArg::from_raw_parts(full_lin[2].clone(), full_n),
                    ArrayArg::from_raw_parts(half_lin[0].clone(), half_n),
                    ArrayArg::from_raw_parts(half_lin[1].clone(), half_n),
                    ArrayArg::from_raw_parts(half_lin[2].clone(), half_n),
                    self.width,
                    full_strip_h_total,
                    half.width,
                    half_strip_h_total,
                );
            }
        }
    }

    /// Strip-multires helper: supersample-add the half-res strip
    /// diffmap into the full-res strip diffmap. The 2× upsample-add
    /// kernel reads `src[y/2 * src_w + x/2]`, mapping each full-res
    /// pixel (x, y) (slab coords) to a half-res pixel (x/2, y/2)
    /// (slab coords). For the body rows of the strip to land on the
    /// correct half-res rows, the constructor enforces `body_h_full
    /// = 2 × body_h_half` and `halo_top_full = 2 × halo_top_half`;
    /// the strip walker maintains this invariant per-strip.
    ///
    /// Both instances must have their slab-clamped `height` set
    /// (full = `full_strip_h_total`, half = `half_strip_h_total`)
    /// before this call.
    pub(crate) fn add_supersampled_from_half_strip(
        &self,
        half: &Butteraugli<R>,
        full_strip_h_total: u32,
        half_strip_h_total: u32,
    ) {
        let full_n = (self.width as usize) * (full_strip_h_total as usize);
        let half_n = (half.width as usize) * (half_strip_h_total as usize);
        const TPB: u32 = 256;
        let cubes = (full_n as u32).div_ceil(TPB);
        let dim = CubeCount::Static(cubes, 1, 1);
        let block = CubeDim::new_1d(TPB);
        unsafe {
            downscale::add_upsample_2x_kernel::launch_unchecked::<R>(
                &self.client,
                dim,
                block,
                ArrayArg::from_raw_parts(self.diffmap_buf.clone(), full_n),
                ArrayArg::from_raw_parts(half.diffmap_buf.clone(), half_n),
                self.width,
                full_strip_h_total,
                half.width,
                0.5_f32,
            );
        }
    }

    /// Strip-multires helper: drive the same one-strip kernel chain
    /// as [`Self::run_strip_pipeline_compute`] but skip the
    /// sRGB→linear stage on the full-res side. The half-res
    /// instance's `lin_a/b` are populated by an earlier
    /// [`Self::downsample_full_strip_into_half`] from the full-res
    /// strip's linear planes; we skip the sRGB upload + sRGB→linear
    /// kernel on the half-res side because there's no half-res
    /// sRGB buffer to read.
    pub(crate) fn run_strip_pipeline_compute_lin_only(&mut self, strip_h_total: u32) {
        let saved_height = self.height;
        let saved_n = self.n;
        self.height = strip_h_total;
        self.n = (self.width as usize) * (strip_h_total as usize);

        self.apply_opsin(true);
        self.apply_opsin(false);
        self.separate_frequencies(true);
        self.separate_frequencies(false);
        self.compute_psycho_diff();
        self.compute_dc_diff();
        self.compute_mask_pipeline_full();
        unsafe {
            self.launch_compute_diffmap();
        }

        self.height = saved_height;
        self.n = saved_n;
    }

    /// Strip-multires helper: borrow the half-res sibling mutably so
    /// the strip walker can clamp its height during a strip pass.
    pub(crate) fn half_res_mut(&mut self) -> Option<&mut Box<Butteraugli<R>>> {
        self.half_res.as_mut()
    }

    /// Strip-multires helper: take the half-res sibling out so the
    /// strip walker can pass `&mut self` and `&mut half` to
    /// kernel-launch helpers without splitting the borrow. The walker
    /// MUST restore it via [`Self::restore_half_res`] before
    /// returning.
    pub(crate) fn take_half_res(&mut self) -> Option<Box<Butteraugli<R>>> {
        self.half_res.take()
    }

    /// Strip-multires helper: put the half-res sibling back after a
    /// [`Self::take_half_res`].
    pub(crate) fn restore_half_res(&mut self, half: Box<Butteraugli<R>>) {
        self.half_res = Some(half);
    }

    fn populate_linear_from_srgb(&mut self, is_a: bool, srgb: &[u8]) {
        let n_bytes = self.n * 3;
        // Defense-in-depth check: every public caller goes through
        // `check_dims` first, so a release-mode panic here would only
        // fire on a buggy internal caller. Demoted to debug_assert.
        debug_assert_eq!(srgb.len(), n_bytes, "input length mismatch");

        // T_x.O (2026-05-17): pack u8×3 → u32 directly into the pinned
        // staging buffer (one host-side pass instead of two). Previously
        // we packed into `self.pack_scratch` and then
        // `create_from_slice_pinned` copied that scratch into a pinned
        // buffer — two full 48 MB host writes for the same data. The
        // reserve_staging path lets us produce the packed bytes
        // straight into the pinned buffer.
        //
        // Layout: 4 bytes per pixel — R | G<<8 | B<<16 (alpha unused).
        // Reader (srgb_u8_to_linear_planar_kernel) sees the same `[u32]`
        // packing T4.L put in place.
        let pinned_len = self.n * 4;
        let mut staging = self.client.reserve_staging(&[pinned_len]);
        let mut bytes = staging
            .pop()
            .expect("reserve_staging returned no buffers");
        {
            let dst: &mut [u8] = &mut bytes;
            debug_assert_eq!(dst.len(), pinned_len);
            // Write four bytes per pixel (R, G, B, 0) directly into the
            // pinned buffer. Endianness: u32 packing `R | G<<8 | B<<16`
            // is little-endian, which matches every supported runtime.
            for (chunk_out, triple) in dst.chunks_exact_mut(4).zip(srgb.chunks_exact(3)) {
                chunk_out[0] = triple[0];
                chunk_out[1] = triple[1];
                chunk_out[2] = triple[2];
                chunk_out[3] = 0;
            }
        }
        // T4.M (2026-05-16): pinned-host fast path — direct DMA (12-25
        // GB/s on PCIe 4.0 vs 5-6 GB/s from pageable).
        // T_x.O: skipping the pack_scratch intermediate saves one
        // ~48 MB host write per upload.
        let handle = self.client.create(bytes);
        if is_a {
            self.src_u8_a = handle;
        } else {
            self.src_u8_b = handle;
        }
        unsafe {
            self.launch_srgb_to_linear(is_a);
        }
    }

    /// Apply opsin: blur(σ=1.2) for sensitivity input, then opsin
    /// dynamics → planar XYB (overwrites `lin_a` / `lin_b` in place).
    ///
    /// T_x.C (2026-05-17): vertical blur and opsin dynamics fused into
    /// one kernel. The intermediate fully-blurred plane is no longer
    /// materialised; opsin reads the per-output-pixel blur sum
    /// directly. Saves a ~144 MB write+read pair at 12 MP.
    fn apply_opsin(&self, is_a: bool) {
        let (lin, bl) = if is_a {
            (&self.lin_a, &self.blur_a)
        } else {
            (&self.lin_b, &self.blur_b)
        };
        // H-pass: fused 3-channel horizontal blur into temp planes.
        let table = &self.blur_tables[BlurKind::Opsin as usize];
        let table_len = self.blur_table_lens[BlurKind::Opsin as usize];
        let radius = self.blur_radii[BlurKind::Opsin as usize];
        unsafe {
            blur_lut::horizontal_blur_3ch_lut_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(lin[0].clone(), self.n),
                ArrayArg::from_raw_parts(lin[1].clone(), self.n),
                ArrayArg::from_raw_parts(lin[2].clone(), self.n),
                ArrayArg::from_raw_parts(self.temp1.clone(), self.n),
                ArrayArg::from_raw_parts(self.temp2.clone(), self.n),
                ArrayArg::from_raw_parts(self.mask_scratch.clone(), self.n),
                ArrayArg::from_raw_parts(table.clone(), table_len),
                self.width,
                self.height,
                radius,
            );
            // V-pass + opsin fused: reads h-blurred from temp planes
            // AND original linear-RGB from lin (per-thread, idx-local
            // only — NOT a window). Writes XYB back into lin
            // in-place; each thread only writes its own idx so the
            // overlapping V-blur window reads (which only touch the
            // h-blurred temp planes, not lin) are safe.
            // T_x.N attempted a 2D 32×8 launch for the opsin V-blur
            // (σ=1.2, radius=2-3); profiling showed unchanged GPU time
            // (669 µs both layouts). The small window already fits L1
            // with the 1D layout. Keeping 1D since a 2D variant adds
            // kernel binary size without measurable benefit. Per G6.8.
            blur_lut::vertical_blur_3ch_opsin_lut_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(self.temp1.clone(), self.n),
                ArrayArg::from_raw_parts(self.temp2.clone(), self.n),
                ArrayArg::from_raw_parts(self.mask_scratch.clone(), self.n),
                ArrayArg::from_raw_parts(lin[0].clone(), self.n),
                ArrayArg::from_raw_parts(lin[1].clone(), self.n),
                ArrayArg::from_raw_parts(lin[2].clone(), self.n),
                ArrayArg::from_raw_parts(lin[0].clone(), self.n),
                ArrayArg::from_raw_parts(lin[1].clone(), self.n),
                ArrayArg::from_raw_parts(lin[2].clone(), self.n),
                ArrayArg::from_raw_parts(table.clone(), table_len),
                self.width,
                self.height,
                radius,
                self.params.intensity_target,
            );
        }
        // The `bl` planes (blur_a/blur_b) are no longer used for the
        // opsin sensitivity — they're free to be reused as scratch by
        // later steps. (Kept allocated so future pipeline refactors
        // can re-purpose them without alloc churn.)
        let _ = bl;
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
        // T_x.F (2026-05-17): the SIGMA_LF blur, the MF = XYB − LF
        // subtracts (×3), and the xyb_low_freq_to_vals in-place mul
        // are all fused into the V-pass of the LF blur. The H pass
        // still runs separately (it's a separable reduction);
        // V-pass + post-blur math runs in one kernel.
        let table = &self.blur_tables[BlurKind::Lf as usize];
        let table_len = self.blur_table_lens[BlurKind::Lf as usize];
        let radius = self.blur_radii[BlurKind::Lf as usize];
        unsafe {
            blur_lut::horizontal_blur_3ch_lut_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(lin[0].clone(), self.n),
                ArrayArg::from_raw_parts(lin[1].clone(), self.n),
                ArrayArg::from_raw_parts(lin[2].clone(), self.n),
                ArrayArg::from_raw_parts(self.temp1.clone(), self.n),
                ArrayArg::from_raw_parts(self.temp2.clone(), self.n),
                ArrayArg::from_raw_parts(self.mask_scratch.clone(), self.n),
                ArrayArg::from_raw_parts(table.clone(), table_len),
                self.width,
                self.height,
                radius,
            );
            // T_x.M: 2D launch (32×8 = 256 threads) for better L1 cache
            // sharing on the σ=7.16 V-blur (33-tap window). With the
            // 1D 256-wide cube layout, each thread of the cube reads a
            // unique column's 33-row strip (~99 KB total per cube,
            // evicting L1). With the 2D 32×8 layout each column is
            // shared by 8 threads; working set drops to ~12 KB/cube.
            let bx = self.width.div_ceil(32);
            let by = self.height.div_ceil(8);
            blur_lut::vertical_blur_3ch_lf_split_lut_kernel_2d::launch_unchecked::<R>(
                &self.client,
                CubeCount::Static(bx, by, 1),
                CubeDim::new_2d(32, 8),
                ArrayArg::from_raw_parts(self.temp1.clone(), self.n),
                ArrayArg::from_raw_parts(self.temp2.clone(), self.n),
                ArrayArg::from_raw_parts(self.mask_scratch.clone(), self.n),
                ArrayArg::from_raw_parts(lin[0].clone(), self.n),
                ArrayArg::from_raw_parts(lin[1].clone(), self.n),
                ArrayArg::from_raw_parts(lin[2].clone(), self.n),
                ArrayArg::from_raw_parts(freq[3][0].clone(), self.n),
                ArrayArg::from_raw_parts(freq[3][1].clone(), self.n),
                ArrayArg::from_raw_parts(freq[3][2].clone(), self.n),
                ArrayArg::from_raw_parts(freq[2][0].clone(), self.n),
                ArrayArg::from_raw_parts(freq[2][1].clone(), self.n),
                ArrayArg::from_raw_parts(freq[2][2].clone(), self.n),
                ArrayArg::from_raw_parts(table.clone(), table_len),
                self.width,
                self.height,
                radius,
            );
        }

        // ── Step 2: MF/HF separation ──
        // T_x.G (2026-05-17): the SIGMA_HF V-blur and the 3 downstream
        // split kernels (split_band_remove for X, split_band_amplify
        // for Y, copy-equivalent for B) are all fused into a single
        // V-pass kernel. The H-pass stays separate (it's a separable
        // reduction). 4 launches saved per side per call.
        let table_hf = &self.blur_tables[BlurKind::Hf as usize];
        let table_hf_len = self.blur_table_lens[BlurKind::Hf as usize];
        let radius_hf = self.blur_radii[BlurKind::Hf as usize];
        unsafe {
            blur_lut::horizontal_blur_3ch_lut_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(freq[2][0].clone(), self.n),
                ArrayArg::from_raw_parts(freq[2][1].clone(), self.n),
                ArrayArg::from_raw_parts(freq[2][2].clone(), self.n),
                ArrayArg::from_raw_parts(self.temp1.clone(), self.n),
                ArrayArg::from_raw_parts(self.temp2.clone(), self.n),
                ArrayArg::from_raw_parts(self.mask_scratch.clone(), self.n),
                ArrayArg::from_raw_parts(table_hf.clone(), table_hf_len),
                self.width,
                self.height,
                radius_hf,
            );
            // V-pass + 3 splits fused: reads h-blurred temp planes
            // (window) + orig freq[2][X,Y] (idx-local); writes 5
            // outputs (HF_X, HF_Y, MF_X, MF_Y, MF_B).
            // T_x.N attempted a 2D 32×8 launch for this kernel; profiling
            // showed a ~13 µs regression (782 vs 768 µs), so kept the 1D
            // path. The smaller-radius HF window (σ=1.564, radius=3-4)
            // already fits L1 with the 1D layout, and 2D apparently adds
            // address-arithmetic overhead. Per docs/CUBECL_GOTCHAS.md G6.8.
            blur_lut::vertical_blur_3ch_hf_split_lut_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(self.temp1.clone(), self.n),
                ArrayArg::from_raw_parts(self.temp2.clone(), self.n),
                ArrayArg::from_raw_parts(self.mask_scratch.clone(), self.n),
                ArrayArg::from_raw_parts(freq[2][0].clone(), self.n),
                ArrayArg::from_raw_parts(freq[2][1].clone(), self.n),
                ArrayArg::from_raw_parts(freq[1][0].clone(), self.n),
                ArrayArg::from_raw_parts(freq[1][1].clone(), self.n),
                ArrayArg::from_raw_parts(freq[2][0].clone(), self.n),
                ArrayArg::from_raw_parts(freq[2][1].clone(), self.n),
                ArrayArg::from_raw_parts(freq[2][2].clone(), self.n),
                ArrayArg::from_raw_parts(table_hf.clone(), table_hf_len),
                self.width,
                self.height,
                radius_hf,
                REMOVE_MF_RANGE,
                ADD_MF_RANGE,
            );
        }

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
        // T_x.H (2026-05-17): fuse the two single-channel UHF blurs
        // (X and Y) into one 2-channel blur, and fuse the V-pass with
        // the X and Y split kernels into one launch. 4 launches saved
        // per side per call (2 V-blurs + 2 splits → 1 fused; 1 H-blur
        // for both channels vs 2 separate). Plus the 2 copy_planes
        // already removed by T_x.D.
        let table_uhf = &self.blur_tables[BlurKind::Uhf as usize];
        let table_uhf_len = self.blur_table_lens[BlurKind::Uhf as usize];
        let radius_uhf = self.blur_radii[BlurKind::Uhf as usize];
        unsafe {
            blur_lut::horizontal_blur_2ch_lut_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(freq[1][0].clone(), self.n),
                ArrayArg::from_raw_parts(freq[1][1].clone(), self.n),
                ArrayArg::from_raw_parts(self.temp1.clone(), self.n),
                ArrayArg::from_raw_parts(self.temp2.clone(), self.n),
                ArrayArg::from_raw_parts(table_uhf.clone(), table_uhf_len),
                self.width,
                self.height,
                radius_uhf,
            );
            // T_x.N attempted 2D 32×8 launch for UHF V-blur + split;
            // unchanged GPU time (615 µs both layouts) since the σ=1.564
            // window (radius 3-4) already fits L1. Kept 1D per G6.8.
            blur_lut::vertical_blur_2ch_uhf_split_lut_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(self.temp1.clone(), self.n),
                ArrayArg::from_raw_parts(self.temp2.clone(), self.n),
                ArrayArg::from_raw_parts(freq[1][0].clone(), self.n),
                ArrayArg::from_raw_parts(freq[1][1].clone(), self.n),
                ArrayArg::from_raw_parts(freq[0][0].clone(), self.n),
                ArrayArg::from_raw_parts(freq[0][1].clone(), self.n),
                ArrayArg::from_raw_parts(freq[1][0].clone(), self.n),
                ArrayArg::from_raw_parts(freq[1][1].clone(), self.n),
                ArrayArg::from_raw_parts(table_uhf.clone(), table_uhf_len),
                self.width,
                self.height,
                radius_uhf,
                REMOVE_UHF_RANGE,
                REMOVE_HF_RANGE,
            );
        }
    }

    /// Compute the AC half of the diffmap: 6 Malta diffs + 2 L2-asym
    /// (HF X/Y) + 3 L2 (MF X/Y/B). Mirrors CPU `compute_psycho_diff_malta`.
    fn compute_psycho_diff(&self) {
        let asym = self.params.hf_asymmetry as f64;
        let sqrt_asym = asym.sqrt();

        // Index conventions: freq[k] where k ∈ {0=UHF, 1=HF, 2=MF, 3=LF};
        //                    freq[k][0]=X, freq[k][1]=Y, freq[k][2]=B.
        //
        // T_x.K (2026-05-17): the 3 maltas per channel (UHF/HF/MF) used
        // to be 3 launches + 1 zero_plane each = 8 launches. Now collapsed
        // to 1 launch per channel via malta_diff_map_triple_kernel, which
        // loads all 3 diff tiles into LDS and overwrites the accumulator
        // (eliminating the zero_plane prelude).

        // Y channel triple-malta (UHF hf-pattern + HF lf-pattern + MF lf-pattern)
        let (uhf_g, uhf_l, uhf_n) =
            malta_norm(W_UHF_MALTA * asym, W_UHF_MALTA / asym, NORM1_UHF, false);
        let (hf_g, hf_l, hf_n) = malta_norm(
            W_HF_MALTA * sqrt_asym,
            W_HF_MALTA / sqrt_asym,
            NORM1_HF,
            true,
        );
        let (mf_g, mf_l, mf_n) = malta_norm(W_MF_MALTA, W_MF_MALTA, NORM1_MF, true);
        self.malta_triple(
            &self.freq_a[0][1],
            &self.freq_b[0][1],
            &self.freq_a[1][1],
            &self.freq_b[1][1],
            &self.freq_a[2][1],
            &self.freq_b[2][1],
            &self.block_diff_ac[1],
            uhf_g,
            uhf_l,
            uhf_n,
            hf_g,
            hf_l,
            hf_n,
            mf_g,
            mf_l,
            mf_n,
        );

        // X channel triple-malta
        let (uhf_g, uhf_l, uhf_n) = malta_norm(
            W_UHF_MALTA_X * asym,
            W_UHF_MALTA_X / asym,
            NORM1_UHF_X,
            false,
        );
        let (hf_g, hf_l, hf_n) = malta_norm(
            W_HF_MALTA_X * sqrt_asym,
            W_HF_MALTA_X / sqrt_asym,
            NORM1_HF_X,
            true,
        );
        let (mf_g, mf_l, mf_n) = malta_norm(W_MF_MALTA_X, W_MF_MALTA_X, NORM1_MF_X, true);
        self.malta_triple(
            &self.freq_a[0][0],
            &self.freq_b[0][0],
            &self.freq_a[1][0],
            &self.freq_b[1][0],
            &self.freq_a[2][0],
            &self.freq_b[2][0],
            &self.block_diff_ac[0],
            uhf_g,
            uhf_l,
            uhf_n,
            hf_g,
            hf_l,
            hf_n,
            mf_g,
            mf_l,
            mf_n,
        );

        // T_x.L (2026-05-17): fuse l2_asym (HF) + l2 (MF) per channel.
        // WMUL[2] = 0.0 (HF B is skipped) so only X and Y get the
        // fused asym+l2; B-channel MF still uses write-only l2_diff
        // since block_diff_ac[2] hasn't been touched yet.

        // X channel: l2_asym(HF X, WMUL[0]) + l2(MF X, WMUL[3])
        self.l2_asym_plus_l2(
            &self.freq_a[1][0],
            &self.freq_b[1][0],
            &self.freq_a[2][0],
            &self.freq_b[2][0],
            &self.block_diff_ac[0],
            (WMUL[0] as f32) * self.params.hf_asymmetry,
            (WMUL[0] as f32) / self.params.hf_asymmetry,
            WMUL[3] as f32,
        );

        // Y channel: l2_asym(HF Y, WMUL[1]) + l2(MF Y, WMUL[4])
        self.l2_asym_plus_l2(
            &self.freq_a[1][1],
            &self.freq_b[1][1],
            &self.freq_a[2][1],
            &self.freq_b[2][1],
            &self.block_diff_ac[1],
            (WMUL[1] as f32) * self.params.hf_asymmetry,
            (WMUL[1] as f32) / self.params.hf_asymmetry,
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
    ///
    /// T_x.J (2026-05-17): all three channels fused into a single launch
    /// (was 3 × l2_diff_write_kernel). Saves 2 kernel launches + 2
    /// launch-latency roundtrips per iter; the per-pixel work itself is
    /// trivial (one FMA per channel) so the launch overhead dominated.
    fn compute_dc_diff(&self) {
        unsafe {
            diffmap::l2_diff_write_3ch_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(self.freq_a[3][0].clone(), self.n),
                ArrayArg::from_raw_parts(self.freq_b[3][0].clone(), self.n),
                ArrayArg::from_raw_parts(self.block_diff_dc[0].clone(), self.n),
                ArrayArg::from_raw_parts(self.freq_a[3][1].clone(), self.n),
                ArrayArg::from_raw_parts(self.freq_b[3][1].clone(), self.n),
                ArrayArg::from_raw_parts(self.block_diff_dc[1].clone(), self.n),
                ArrayArg::from_raw_parts(self.freq_a[3][2].clone(), self.n),
                ArrayArg::from_raw_parts(self.freq_b[3][2].clone(), self.n),
                ArrayArg::from_raw_parts(self.block_diff_dc[2].clone(), self.n),
                WMUL[6] as f32,
                WMUL[7] as f32,
                WMUL[8] as f32,
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
            // T_x.I: combine + diff_precompute fused (both pointwise,
            // saves one launch + one full-plane R/W roundtrip).
            masking::combine_channels_and_diff_precompute_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(self.freq_a[1][0].clone(), self.n),
                ArrayArg::from_raw_parts(self.freq_a[0][0].clone(), self.n),
                ArrayArg::from_raw_parts(self.freq_a[1][1].clone(), self.n),
                ArrayArg::from_raw_parts(self.freq_a[0][1].clone(), self.n),
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
            // T_x.I: combine + diff_precompute fused.
            masking::combine_channels_and_diff_precompute_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(self.freq_b[1][0].clone(), self.n),
                ArrayArg::from_raw_parts(self.freq_b[0][0].clone(), self.n),
                ArrayArg::from_raw_parts(self.freq_b[1][1].clone(), self.n),
                ArrayArg::from_raw_parts(self.freq_b[0][1].clone(), self.n),
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

    /// T_x.K (2026-05-17): single-launch UHF (hf-pattern) + HF (lf-pattern)
    /// + MF (lf-pattern) per channel. Overwrites the accumulator.
    #[allow(clippy::too_many_arguments)]
    fn malta_triple(
        &self,
        uhf_a: &cubecl::server::Handle,
        uhf_b: &cubecl::server::Handle,
        hf_a: &cubecl::server::Handle,
        hf_b: &cubecl::server::Handle,
        mf_a: &cubecl::server::Handle,
        mf_b: &cubecl::server::Handle,
        acc: &cubecl::server::Handle,
        uhf_norm2_0gt1: f32,
        uhf_norm2_0lt1: f32,
        uhf_norm1: f32,
        hf_norm2_0gt1: f32,
        hf_norm2_0lt1: f32,
        hf_norm1: f32,
        mf_norm2_0gt1: f32,
        mf_norm2_0lt1: f32,
        mf_norm1: f32,
    ) {
        unsafe {
            malta::malta_diff_map_triple_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_2d(),
                self.cube_dim_2d(),
                ArrayArg::from_raw_parts(uhf_a.clone(), self.n),
                ArrayArg::from_raw_parts(uhf_b.clone(), self.n),
                ArrayArg::from_raw_parts(hf_a.clone(), self.n),
                ArrayArg::from_raw_parts(hf_b.clone(), self.n),
                ArrayArg::from_raw_parts(mf_a.clone(), self.n),
                ArrayArg::from_raw_parts(mf_b.clone(), self.n),
                ArrayArg::from_raw_parts(acc.clone(), self.n),
                self.width,
                self.height,
                uhf_norm2_0gt1,
                uhf_norm2_0lt1,
                uhf_norm1,
                hf_norm2_0gt1,
                hf_norm2_0lt1,
                hf_norm1,
                mf_norm2_0gt1,
                mf_norm2_0lt1,
                mf_norm1,
            );
        }
    }

    /// T_x.L: fused asym(HF) + l2(MF) accumulator per channel.
    #[allow(clippy::too_many_arguments)]
    fn l2_asym_plus_l2(
        &self,
        asym_a: &cubecl::server::Handle,
        asym_b: &cubecl::server::Handle,
        l2_a: &cubecl::server::Handle,
        l2_b: &cubecl::server::Handle,
        acc: &cubecl::server::Handle,
        asym_weight_gt: f32,
        asym_weight_lt: f32,
        l2_weight: f32,
    ) {
        unsafe {
            diffmap::l2_asym_plus_l2_kernel::launch_unchecked::<R>(
                &self.client,
                self.cube_count_1d(),
                self.cube_dim_1d(),
                ArrayArg::from_raw_parts(asym_a.clone(), self.n),
                ArrayArg::from_raw_parts(asym_b.clone(), self.n),
                ArrayArg::from_raw_parts(l2_a.clone(), self.n),
                ArrayArg::from_raw_parts(l2_b.clone(), self.n),
                ArrayArg::from_raw_parts(acc.clone(), self.n),
                asym_weight_gt,
                asym_weight_lt,
                l2_weight,
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
