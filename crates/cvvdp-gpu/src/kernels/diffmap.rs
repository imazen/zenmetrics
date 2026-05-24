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
//! cvvdp-cpu port; matches `docs/RFC_CVVDP_FORK.md` §3)
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

/// Convert one linear-RGB pixel triple (already in linear-light, no
/// EOTF needed) to the cvvdp DKL opponent triple. The math mirrors
/// [`crate::kernels::color::srgb_byte_to_dkl_scalar`] verbatim from
/// the display-model step onward — the only difference is that this
/// kernel skips the sRGB→linear LUT lookup because the caller has
/// pre-linearised the input.
///
/// Inputs:
/// - `lin_r`, `lin_g`, `lin_b` — three planar `W × H` linear-light
///   sRGB f32 buffers in the unit `[0, 1]` range. Values outside
///   `[0, 1]` are accepted (cvvdp's DKL is a linear transform; HDR
///   highlights and below-black blacks pass through unmodified).
///
/// Outputs:
/// - `out_a`, `out_rg`, `out_vy` — `W × H` planar f32 in DKL
///   opponent space (cd/m²-scaled).
///
/// Display constants (`y_peak`, `y_black`, `y_refl`) are pushed as
/// runtime scalars; the DKL matrix is captured as kernel-local f32
/// constants so LLVM folds the linear combination at codegen time
/// (same bit-pinned values as `srgb_to_dkl_kernel`).
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

    // Same row-bit-pinned f32 constants as srgb_to_dkl_kernel.
    let m00 = f32::new(0.233_201_21);
    let m01 = f32::new(0.728_830_8);
    let m02 = f32::new(0.088_995_87);
    let m10 = f32::new(0.127_620_77);
    let m11 = f32::new(-0.087_068_09);
    let m12 = f32::new(-0.036_777_39);
    let m20 = f32::new(-0.214_822_5);
    let m21 = f32::new(-0.626_253_7);
    let m22 = f32::new(0.851_403_3);

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
/// This matches `cvvdp_cpu::diffmap::finalize_diffmap` byte-for-byte
/// at f32 precision so the CPU and GPU diffmap impls produce the
/// same per-pixel values on identical inputs (verified by the
/// `cvvdp_cpu::diffmap` recipe at master `da816947`).
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
/// shape. `cvvdp_cpu::diffmap::finalize_diffmap` skips the epsilon
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

/// Host-side scalar reference for [`diffmap_band_accumulate_kernel`]'s
/// bilinear sample. Used by `tests/diffmap_invariants.rs` to pin
/// boundary semantics without a GPU.
///
/// Returns the bilinear sample of `src[src_w × src_h]` at the dst
/// coordinate `(x_dst, y_dst)` of a `dst_w × dst_h` output.
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::kernels::diffmap::bilinear_sample_scalar;
///
/// // 2×2 → 4×4 upsample. Top-left dst pixel samples the top-left
/// // src pixel at half-pixel offset = (0,0), interpolating between
/// // (0,0) and (1,0)/(0,1)/(1,1) of the 2×2 source.
/// let src = [1.0_f32, 2.0, 3.0, 4.0]; // row-major 2×2
/// let v = bilinear_sample_scalar(&src, 2, 2, 0, 0, 4, 4);
/// // Edge-clamped: dst (0, 0) samples src at (-0.25, -0.25) → clamps
/// // to (0, 0) → value = 1.0.
/// assert!((v - 1.0).abs() < 1e-6);
///
/// // dst (3, 3) → src (1.75, 1.75) → clamps to (1.0, 1.0) → 4.0
/// let v = bilinear_sample_scalar(&src, 2, 2, 3, 3, 4, 4);
/// assert!((v - 4.0).abs() < 1e-6);
///
/// // Identity (src_dims == dst_dims) → exact copy.
/// for (i, &expected) in src.iter().enumerate() {
///     let x = (i as u32) % 2;
///     let y = (i as u32) / 2;
///     let v = bilinear_sample_scalar(&src, 2, 2, x, y, 2, 2);
///     assert!((v - expected).abs() < 1e-6);
/// }
/// ```
#[must_use]
pub fn bilinear_sample_scalar(
    src: &[f32],
    src_w: u32,
    src_h: u32,
    x_dst: u32,
    y_dst: u32,
    dst_w: u32,
    dst_h: u32,
) -> f32 {
    let src_w_f = src_w as f32;
    let src_h_f = src_h as f32;
    let dst_w_f = dst_w as f32;
    let dst_h_f = dst_h as f32;

    let fx = (x_dst as f32 + 0.5) * (src_w_f / dst_w_f) - 0.5;
    let fy = (y_dst as f32 + 0.5) * (src_h_f / dst_h_f) - 0.5;

    let fx_c = fx.clamp(0.0, src_w_f - 1.0);
    let fy_c = fy.clamp(0.0, src_h_f - 1.0);

    let x0_f = fx_c.floor();
    let y0_f = fy_c.floor();
    let dx = fx_c - x0_f;
    let dy = fy_c - y0_f;

    let x0 = x0_f as u32;
    let y0 = y0_f as u32;
    let src_w_m1 = src_w.saturating_sub(1);
    let src_h_m1 = src_h.saturating_sub(1);
    let x1 = if x0 < src_w_m1 { x0 + 1 } else { src_w_m1 };
    let y1 = if y0 < src_h_m1 { y0 + 1 } else { src_h_m1 };

    let i00 = (y0 * src_w + x0) as usize;
    let i01 = (y0 * src_w + x1) as usize;
    let i10 = (y1 * src_w + x0) as usize;
    let i11 = (y1 * src_w + x1) as usize;

    let w00 = (1.0 - dx) * (1.0 - dy);
    let w01 = dx * (1.0 - dy);
    let w10 = (1.0 - dx) * dy;
    let w11 = dx * dy;

    w00 * src[i00] + w01 * src[i01] + w10 * src[i10] + w11 * src[i11]
}

/// Host-side scalar reference for [`diffmap_channel_pool_kernel`].
/// Computes the per-pixel Minkowski-p pool across the 3 DKL
/// channels for one pixel. Returns the same value the GPU kernel
/// would write to `out[idx]`.
///
/// Matches `cvvdp_cpu::diffmap::finalize_diffmap`'s per-pixel
/// pool exactly: `max(., 0)` clamp per channel, plain `pow(p)` (no
/// safe_pow epsilon), sum, `pow(1/p)`. See the kernel docstring
/// for the rationale on the clamp + no-epsilon choices.
///
/// Used by `tests/diffmap_invariants.rs` to pin GPU↔CPU parity
/// without firing GPU dispatch.
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::kernels::diffmap::channel_pool_scalar;
///
/// // All-zero channels → output 0 (no epsilon-tail bias).
/// assert_eq!(channel_pool_scalar(0.0, 0.0, 0.0, 4.0), 0.0);
///
/// // Single-channel signal at +v exactly recovers v (the other two
/// // channels contribute 0 after the max-clamp).
/// let v = channel_pool_scalar(2.0, 0.0, 0.0, 4.0);
/// assert!((v - 2.0).abs() < 1e-6, "got {v}, expected = 2.0");
///
/// // max(., 0) clamps negatives to 0 — asymmetric by design, see
/// // the kernel docstring for rationale (rounding-noise mask).
/// let pos = channel_pool_scalar(2.0, 0.0, 0.0, 4.0);
/// let neg = channel_pool_scalar(-2.0, 0.0, 0.0, 4.0);
/// assert_eq!(neg, 0.0);
/// assert!(pos > 0.0);
/// ```
#[must_use]
pub fn channel_pool_scalar(a: f32, rg: f32, vy: f32, beta: f32) -> f32 {
    let a_pos = a.max(0.0);
    let rg_pos = rg.max(0.0);
    let vy_pos = vy.max(0.0);

    let acc = a_pos.powf(beta) + rg_pos.powf(beta) + vy_pos.powf(beta);
    acc.powf(1.0 / beta)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bilinear_identity_copies_exactly() {
        // src_dims == dst_dims → output is the source.
        let src: Vec<f32> = (0..16).map(|i| i as f32).collect();
        for y in 0..4 {
            for x in 0..4 {
                let v = bilinear_sample_scalar(&src, 4, 4, x, y, 4, 4);
                let expected = src[(y * 4 + x) as usize];
                assert!(
                    (v - expected).abs() < 1e-5,
                    "identity sample at ({x},{y}) got {v}, expected {expected}",
                );
            }
        }
    }

    #[test]
    fn bilinear_2x_upsample_4_corners_match_src() {
        // A 2×2 src upsampled to 4×4. The 4 corner DST pixels should
        // collapse to the matching corner SRC pixel (after clamp).
        let src = [1.0_f32, 2.0, 3.0, 4.0]; // row-major 2×2
        assert!((bilinear_sample_scalar(&src, 2, 2, 0, 0, 4, 4) - 1.0).abs() < 1e-6);
        assert!((bilinear_sample_scalar(&src, 2, 2, 3, 0, 4, 4) - 2.0).abs() < 1e-6);
        assert!((bilinear_sample_scalar(&src, 2, 2, 0, 3, 4, 4) - 3.0).abs() < 1e-6);
        assert!((bilinear_sample_scalar(&src, 2, 2, 3, 3, 4, 4) - 4.0).abs() < 1e-6);
    }

    #[test]
    fn bilinear_constant_input_propagates() {
        let src = vec![7.5_f32; 16]; // 4×4 constant
        for y in 0..8 {
            for x in 0..8 {
                let v = bilinear_sample_scalar(&src, 4, 4, x, y, 8, 8);
                assert!((v - 7.5).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn channel_pool_identity_input_returns_zero() {
        for &v in &[0.0_f32, 1.0, -2.5, 1e-6, -1e-6] {
            assert!(channel_pool_scalar(0.0, 0.0, 0.0, 4.0) < 1e-6, "{v}");
        }
    }

    #[test]
    fn channel_pool_clamps_negatives_to_zero() {
        // max(., 0) clamp: any negative channel contributes 0 to the
        // pool. This is asymmetric vs `lp_norm_sum` (which abs-pools
        // both signs), but it's the contract we share with
        // `cvvdp_cpu::diffmap::finalize_diffmap`. The masking stage
        // guarantees non-negative inputs in the analytical limit, so
        // this only masks f32 rounding noise.
        let beta = 4.0;
        assert_eq!(channel_pool_scalar(-1.0, 0.0, 0.0, beta), 0.0);
        assert_eq!(channel_pool_scalar(-1.0, -1.0, -1.0, beta), 0.0);
        assert_eq!(channel_pool_scalar(0.0, 0.0, 0.0, beta), 0.0);

        // Positive channels are recovered cleanly via Minkowski sum.
        let v = channel_pool_scalar(2.0_f32, 0.0, 0.0, beta);
        assert!((v - 2.0).abs() < 1e-6);

        // Three equal positives at β=4: (3 * v^4)^(1/4) = v * 3^(1/4) ≈ v * 1.3161
        let v = channel_pool_scalar(1.0, 1.0, 1.0, beta);
        let expected = (3.0_f32).powf(0.25);
        assert!((v - expected).abs() < 1e-6, "got {v}, expected {expected}");
    }

    #[test]
    fn channel_pool_matches_cvvdp_cpu_recipe() {
        // Mirror cvvdp_cpu::diffmap::finalize_diffmap exactly. Both
        // GPU and CPU diffmap impls share the same recipe; the
        // per-pixel pool fold must produce identical values on
        // identical inputs (to f32 precision).
        let cases = [
            (1.0_f32, 1.0, 1.0),
            (-1.0, 2.0, -3.0),
            (0.5, 0.0, 0.5),
            (10.0, -10.0, 0.1),
            (0.001, 0.001, 0.001),
            (0.0, 0.0, 0.0),
        ];
        for &(a, rg, vy) in &cases {
            let beta = 4.0;
            let direct = channel_pool_scalar(a, rg, vy, beta);
            // Mirror cvvdp_cpu::diffmap::finalize_diffmap byte-for-byte.
            let a_pos = a.max(0.0);
            let rg_pos = rg.max(0.0);
            let vy_pos = vy.max(0.0);
            let cpu = (a_pos.powf(beta) + rg_pos.powf(beta) + vy_pos.powf(beta)).powf(1.0 / beta);
            assert!(
                (direct - cpu).abs() < 1e-6,
                "channel_pool({a}, {rg}, {vy}) = {direct} vs cvvdp-cpu = {cpu}",
            );
        }
    }

    #[test]
    fn channel_pool_monotone_in_magnitude() {
        let a = channel_pool_scalar(0.5, 0.5, 0.5, 4.0);
        let b = channel_pool_scalar(1.0, 1.0, 1.0, 4.0);
        let c = channel_pool_scalar(2.0, 2.0, 2.0, 4.0);
        assert!(a < b, "channel_pool not monotone: {a} >= {b}");
        assert!(b < c, "channel_pool not monotone: {b} >= {c}");
    }
}
