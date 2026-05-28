//! Per-pixel diffmap construction for the cvvdp pipeline.
//!
//! The cvvdp scalar JOD is produced by reducing the per-pixel
//! per-(band, DKL channel) masked-difference planes (`d_scratch[k].d[c]`)
//! through a 3-stage Minkowski pool: spatial → bands → channels (see
//! `kernels::pool::do_pooling_and_jod_still_3ch`). The buttloop
//! consumer in jxl-encoder needs a *per-pixel* signal whose Minkowski
//! pool approximates the JOD, so it can drive per-block 8×8 median /
//! MAD heuristics. This module implements that signal.
//!
//! ## Recipe (binding for both the cvvdp-gpu extension and the
//! cvvdp port; matches `docs/RFC_CVVDP_FORK.md` §3)
//!
//! 1. After masking we have per-pixel masked error `D[k][c][i]` for
//!    each pyramid level `k`, DKL channel `c ∈ {A, RG, VY}`, and
//!    pixel `i` of size `level_w(k) × level_h(k)`.
//! 2. **Bilinear-upsample each `D[k][c]` to base resolution**
//!    (`W × H`). Coordinates use the OpenCV-INTER_LINEAR /
//!    PyTorch align_corners=False convention: a base pixel
//!    `(x, y)` samples the coarse plane at
//!    `((x + 0.5) * w_k / W − 0.5, (y + 0.5) * h_k / H − 0.5)`,
//!    clamped to the coarse range.
//! 3. **Sum across bands** with the same per-band weights the scalar
//!    pool uses: `per_ch[c][i] = Σ_k per_sband_w[k][c] *
//!    PER_CH_W[c] * D_up[k][c][i]`. `per_sband_w[k][c] = 1` for
//!    `k < n_levels − 1` and `BASEBAND_W[c]` at the baseband, the
//!    same scheme [`crate::kernels::pool::do_pooling_and_jod_still_3ch`]
//!    applies during the scalar fold.
//! 4. **Pool across DKL channels with Minkowski-p**:
//!    `diffmap[i] = lp_norm_sum({per_ch[c][i] for c}, p = BETA_CH)`,
//!    where the `lp_norm_sum` definition is the same `safe_pow`-
//!    regularised form used by
//!    [`crate::kernels::pool::lp_norm_sum`].
//!
//! Output is a `W × H` row-major f32 plane, non-negative,
//! suitable for direct consumption by the jxl-encoder buttloop's
//! per-block median+MAD reducer.
//!
//! ## Relationship to the scalar JOD (divergence note)
//!
//! RFC §3 sketches an invariant
//! `JOD == 10 - minkowski_p_norm(diffmap_flat, p = beta_sch)`.
//! That invariant cannot hold *exactly* for the canonical cvvdp v0.5.4
//! pool because the scalar JOD uses **three different Minkowski
//! exponents at three reduction stages** (`BETA_SPATIAL = 2` over
//! pixels, `BETA_BAND = 4` across bands, `BETA_CH = 4` across
//! channels), and there is no single per-pixel signal whose
//! Minkowski-p norm collapses all three stages identically. The
//! divergence is documented in
//! `crates/cvvdp-gpu/docs/DIFFMAP_DIVERGENCES.md`.
//!
//! The diffmap we produce *does* satisfy weaker properties that
//! cover the buttloop use case (see `tests/diffmap_invariants.rs`):
//!
//! - Identity inputs (`ref ≡ dist`) → all zeros to 1e-7 absolute
//!   (no false floor).
//! - Non-negative everywhere (Minkowski-p is non-negative).
//! - Monotonic with distortion magnitude: scaling the per-band
//!   D planes by α scales the diffmap by α.
//! - The lp_norm_mean of the diffmap (β = 2) correlates strongly
//!   with the scalar JOD across synthetic fixtures.

// Tick 514: silence missing_docs warnings on items emitted by the
// #[cube(launch)] macro (see kernels/color.rs for full rationale).
#![allow(missing_docs)]

use cubecl::prelude::*;

// Phase 8c.1-C: scalar helpers (`bilinear_sample_scalar`,
// `channel_pool_scalar`) live in `cvvdp::kernels::diffmap` so the
// CPU crate owns the canonical scalar implementation; re-export
// keeps the existing `cvvdp_gpu::kernels::diffmap::*` import paths
// working unchanged. No cube-macro interaction here — the diffmap
// `#[cube(launch)]` kernels below do not reference the scalar
// helpers.
pub use cvvdp::kernels::diffmap::{bilinear_sample_scalar, channel_pool_scalar};

/// Convert one linear-RGB pixel triple (already in linear-light, no
/// EOTF needed) to the cvvdp DKL opponent triple. The math mirrors
/// [`crate::kernels::color::srgb_byte_to_dkl_scalar`] verbatim from
/// the display-model step onward — the only difference is that this
/// kernel skips the sRGB→linear LUT lookup because the caller has
/// pre-linearised the input.
///
/// Inputs:
/// - `lin_r`, `lin_g`, `lin_b` — three planar `W × H` linear-light
///   f32 buffers in the unit `[0, 1]` range (or HDR cd/m²-scaled when
///   paired with a Linear-EOTF display model). Values outside
///   `[0, 1]` are accepted.
///
/// Outputs:
/// - `out_a`, `out_rg`, `out_vy` — `W × H` planar f32 in DKL
///   opponent space (cd/m²-scaled).
///
/// Display constants (`y_peak`, `y_black`, `y_refl`) are pushed as
/// runtime scalars. The DKL matrix is also pushed as 9 runtime
/// scalars so a single kernel binary serves every
/// [`crate::params::Primaries`] variant — LLVM still folds the
/// linear combination at codegen time when the values are constant
/// across the launch.
#[cube(launch)]
pub fn linear_rgb_planes_to_dkl_kernel(
    lin_r: &Array<f32>,
    lin_g: &Array<f32>,
    lin_b: &Array<f32>,
    out_a: &mut Array<f32>,
    out_rg: &mut Array<f32>,
    out_vy: &mut Array<f32>,
    width: u32,
    height: u32,
    y_peak: f32,
    y_black: f32,
    y_refl: f32,
    m00: f32,
    m01: f32,
    m02: f32,
    m10: f32,
    m11: f32,
    m12: f32,
    m20: f32,
    m21: f32,
    m22: f32,
) {
    let idx = ABSOLUTE_POS;
    let total = (width * height) as usize;
    if idx >= total {
        terminate!();
    }

    let r = lin_r[idx];
    let g = lin_g[idx];
    let b = lin_b[idx];

    let s = y_peak - y_black;
    let bias = y_black + y_refl;
    let lr = s * r + bias;
    let lg = s * g + bias;
    let lb = s * b + bias;

    out_a[idx] = m00 * lr + m01 * lg + m02 * lb;
    out_rg[idx] = m10 * lr + m11 * lg + m12 * lb;
    out_vy[idx] = m20 * lr + m21 * lg + m22 * lb;
}

/// Bilinear-upsample one (band, channel)'s masked D plane to the
/// base resolution `W × H` and add the weighted sample into the
/// running accumulator. One thread per **base-resolution** pixel.
///
/// Sampling convention (OpenCV INTER_LINEAR / PyTorch
/// align_corners=False):
///
/// ```text
/// fx = (x_dst + 0.5) * src_w / dst_w - 0.5
/// fy = (y_dst + 0.5) * src_h / dst_h - 0.5
/// ```
///
/// Coordinates are clamped to `[0, src_w − 1]` / `[0, src_h − 1]`
/// before the four-tap bilinear blend. When `src_w == dst_w` AND
/// `src_h == dst_h` (level 0 — base resolution band), the kernel
/// degenerates to a per-pixel copy: integer sample coords, no
/// interpolation, no off-by-one risk.
///
/// Weights:
/// - `weight` is the precomputed `per_sband_w[k][c] * PER_CH_W[c]`
///   scalar the caller would otherwise apply in the host fold. The
///   kernel writes `acc[i] += weight * D_upsampled[i]` (no sign
///   handling — the cvvdp scalar fold takes `|x|` inside
///   `safe_pow` later via the channel pool, so we keep the signed
///   value here to preserve cross-channel interactions).
///
/// Numerical stability:
/// - The kernel does not check for `weight == 0`. Callers that want
///   to skip a band must skip the launch.
/// - Sources passed in must contain finite f32 values; the cvvdp
///   masking stage that produces `d_scratch[k].d[c]` guarantees
///   this within the cvvdp pipeline.
#[cube(launch)]
pub fn diffmap_band_accumulate_kernel(
    src_a: &Array<f32>,
    src_rg: &Array<f32>,
    src_vy: &Array<f32>,
    acc_a: &mut Array<f32>,
    acc_rg: &mut Array<f32>,
    acc_vy: &mut Array<f32>,
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    w_a: f32,
    w_rg: f32,
    w_vy: f32,
) {
    let idx = ABSOLUTE_POS;
    let total = (dst_w * dst_h) as usize;
    if idx >= total {
        terminate!();
    }

    let x_dst = (idx as u32) % dst_w;
    let y_dst = (idx as u32) / dst_w;

    // Half-pixel-aligned bilinear: equivalent to OpenCV INTER_LINEAR
    // with align_corners=False. We compute in f32 throughout.
    let dst_w_f = dst_w as f32;
    let dst_h_f = dst_h as f32;
    let src_w_f = src_w as f32;
    let src_h_f = src_h as f32;

    let fx = (x_dst as f32 + f32::new(0.5)) * (src_w_f / dst_w_f) - f32::new(0.5);
    let fy = (y_dst as f32 + f32::new(0.5)) * (src_h_f / dst_h_f) - f32::new(0.5);

    // Clamp fx/fy to [0, src_w-1] / [0, src_h-1] so out-of-range
    // sample positions snap to the boundary (extend-by-edge), then
    // compute the integer/fractional split.
    let zero = f32::new(0.0);
    let fx_c = if fx < zero {
        zero
    } else if fx > src_w_f - f32::new(1.0) {
        src_w_f - f32::new(1.0)
    } else {
        fx
    };
    let fy_c = if fy < zero {
        zero
    } else if fy > src_h_f - f32::new(1.0) {
        src_h_f - f32::new(1.0)
    } else {
        fy
    };

    let x0_f = f32::floor(fx_c);
    let y0_f = f32::floor(fy_c);
    let dx = fx_c - x0_f;
    let dy = fy_c - y0_f;

    let x0 = x0_f as u32;
    let y0 = y0_f as u32;
    // x1/y1 saturate at src_w-1 / src_h-1 so the +1 step never
    // walks off the source plane (covers fx == src_w-1 exactly).
    let src_w_m1 = src_w - 1u32;
    let src_h_m1 = src_h - 1u32;
    let x1 = if x0 < src_w_m1 { x0 + 1u32 } else { src_w_m1 };
    let y1 = if y0 < src_h_m1 { y0 + 1u32 } else { src_h_m1 };

    let i00 = (y0 * src_w + x0) as usize;
    let i01 = (y0 * src_w + x1) as usize;
    let i10 = (y1 * src_w + x0) as usize;
    let i11 = (y1 * src_w + x1) as usize;

    let one = f32::new(1.0);
    let w00 = (one - dx) * (one - dy);
    let w01 = dx * (one - dy);
    let w10 = (one - dx) * dy;
    let w11 = dx * dy;

    let v_a = w00 * src_a[i00] + w01 * src_a[i01] + w10 * src_a[i10] + w11 * src_a[i11];
    let v_rg = w00 * src_rg[i00] + w01 * src_rg[i01] + w10 * src_rg[i10] + w11 * src_rg[i11];
    let v_vy = w00 * src_vy[i00] + w01 * src_vy[i01] + w10 * src_vy[i10] + w11 * src_vy[i11];

    acc_a[idx] = acc_a[idx] + w_a * v_a;
    acc_rg[idx] = acc_rg[idx] + w_rg * v_rg;
    acc_vy[idx] = acc_vy[idx] + w_vy * v_vy;
}

/// Pool the per-channel accumulator planes into a single per-pixel
/// diffmap via Minkowski sum across the 3 DKL channels with
/// non-negative inputs:
///
/// ```text
/// diffmap[i] = (max(acc_a[i], 0)^β
///             + max(acc_rg[i], 0)^β
///             + max(acc_vy[i], 0)^β)^(1/β)
/// ```
///
/// This matches `cvvdp::diffmap::finalize_diffmap` byte-for-byte
/// at f32 precision so the CPU and GPU diffmap impls produce the
/// same per-pixel values on identical inputs (verified by the
/// `cvvdp::diffmap` recipe at master `da816947`).
///
/// **Why max(., 0) and not |.|**: the cvvdp masking stage produces
/// `D_per_ch[c]` planes whose values are already non-negative
/// magnitudes (`|T_p_dist − T_p_ref|` for the baseband, the post-
/// masking absolute response for non-baseband). The weighted band
/// sum in [`diffmap_band_accumulate_kernel`] preserves the
/// non-negative-input property in the analytical limit; only f32
/// rounding can drive a tiny negative. `max(., 0)` is the cheapest
/// way to mask that rounding noise without introducing an
/// asymmetric bias on legitimate negatives (which don't occur here
/// by construction).
///
/// **Why no safe_pow epsilon**: differentiability-at-zero is a
/// pycvvdp pool-stage concern (it pools `safe_pow(|D|, β)`
/// reductions through several stages so the encoder gets a smooth
/// gradient on the JOD scalar). The diffmap is consumed by the
/// jxl-encoder buttloop's per-block 8×8 median + MAD reducer, which
/// is non-differentiable anyway — the epsilon would only add a
/// constant bias to every pixel without changing the relative
/// shape. `cvvdp::diffmap::finalize_diffmap` skips the epsilon
/// for the same reason; the GPU kernel matches.
///
/// `beta` is parametrised so callers can probe pool sensitivity
/// without recompiling. Production passes `BETA_CH = 4`.
#[cube(launch)]
pub fn diffmap_channel_pool_kernel(
    acc_a: &Array<f32>,
    acc_rg: &Array<f32>,
    acc_vy: &Array<f32>,
    out: &mut Array<f32>,
    beta: f32,
    n: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= n as usize {
        terminate!();
    }

    let zero = f32::new(0.0);

    let v_a = acc_a[idx];
    let v_a_pos = if v_a < zero { zero } else { v_a };
    let c_a = f32::powf(v_a_pos, beta);

    let v_rg = acc_rg[idx];
    let v_rg_pos = if v_rg < zero { zero } else { v_rg };
    let c_rg = f32::powf(v_rg_pos, beta);

    let v_vy = acc_vy[idx];
    let v_vy_pos = if v_vy < zero { zero } else { v_vy };
    let c_vy = f32::powf(v_vy_pos, beta);

    let acc = c_a + c_rg + c_vy;
    let one = f32::new(1.0);
    out[idx] = f32::powf(acc, one / beta);
}

/// Write zero to every slot of `dest`. Used to zero the diffmap
/// accumulator planes before the per-band accumulate pass.
///
/// Duplicates [`crate::kernels::pool::fill_f32_kernel`] (with the
/// scalar fixed to 0.0) to avoid forwarding the per-call value
/// scalar across the diffmap pipeline. Identical launch semantics:
/// one thread per slot, `n` is the total slot count.
#[cube(launch)]
pub fn diffmap_zero_kernel(dest: &mut Array<f32>, n: u32) {
    let idx = ABSOLUTE_POS;
    if idx >= n as usize {
        terminate!();
    }
    dest[idx] = f32::new(0.0);
}

// Scalar helpers `bilinear_sample_scalar` and `channel_pool_scalar`
// — plus their unit tests — moved to `cvvdp::kernels::diffmap` in
// Phase 8c.1-C. Both functions are still reachable at the old
// `cvvdp_gpu::kernels::diffmap::*` paths through the `pub use`
// re-export at the top of this file; tests live in cvvdp now.
