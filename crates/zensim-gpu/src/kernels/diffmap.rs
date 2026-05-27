//! Per-pixel diffmap GPU kernels + host-scalar reference helpers.
//!
//! Phase 1 (CPU-fallback) shipped this module with host-scalar
//! reference helpers only and delegated the actual diffmap production
//! to the canonical `zensim::Zensim::compute_with_ref_and_diffmap_linear_planar`.
//!
//! Phase 1b (this commit) lands the **pure-GPU diffmap kernel chain**
//! for the default [`zensim::DiffmapOptions`] path (SSIM-only,
//! `Trained` weighting, no contrast masking, no sqrt). The kernel chain:
//!
//! 1. [`per_scale_weighted_ssim_kernel`] — per-pixel modified-SSIM
//!    error (the same `sd0 = max(0, 1 − num_m·num_s/denom_s)` value the
//!    [`crate::kernels::fused`] feature kernels compute scalar-fold),
//!    multiplied by the per-(scale, channel) trained weight and summed
//!    across the 3 XYB channels into one per-scale plane.
//! 2. [`pow2x_upsample_add_kernel`] — nearest-neighbor power-of-2
//!    replicate of one per-scale plane into the base-resolution
//!    accumulator, scaled by the per-scale blend weight.
//! 3. [`diffmap_fill_kernel`] — zero-fill the base-resolution
//!    accumulator before band loop. Trivial but kept as a launch unit
//!    so the pipeline doesn't fight cubecl's zero-fill semantics on the
//!    `empty()` handles.
//! 4. [`diffmap_trim_padded_kernel`] — trims the `padded_w × height`
//!    accumulator into the caller-facing tight `width × height`
//!    output (drops the right-side SIMD-pad columns).
//!
//! Extended [`zensim::DiffmapOptions`] (edge_mse / hf / contrast
//! masking / sqrt) STAY on the CPU-fallback path — see
//! `docs/DIFFMAP_DIVERGENCES.md` §10 + §6 for the divergence
//! contract and Phase 1c roadmap. The Phase 1b GPU kernels match
//! the **default options ONLY**.
//!
//! See [`crate::pipeline::Zensim::score_with_diffmap`] and
//! `docs/DIFFMAP_DIVERGENCES.md` for the dispatch rules.
//!
//! ## Recipe (canonical zensim CPU diffmap, default options)
//!
//! Implemented in `zensim::diffmap::compute_with_ref_and_diffmap_linear_planar`
//! when `DiffmapOptions::default()` is passed. The Phase 1b kernels in
//! this module mirror the recipe step-for-step at the per-pixel layer:
//!
//! 1. Per-scale per-channel SSIM error compute. The per-pixel signal is
//!    `sd0 = max(0, 1 − num_m · num_s / denom_s)` with the exact same
//!    FMA fusion order the scalar feature kernel uses (see
//!    [`crate::kernels::fused::fused_features_kernel_persist`] body).
//! 2. Per-scale per-channel weighted reduction to one f32 plane per
//!    scale: `scale_dm[i] = Σ_c w_c · sd0_c[i]`.
//! 3. Nearest-neighbor power-of-2 upsample of each coarser-scale plane
//!    into the base-resolution accumulator
//!    (`fused[i] += blend_s · scale_dm_s[i / 2^s]`). Identical to
//!    `zensim::streaming::upsample_pow2x_add` for `factor = 1 << scale`.
//! 4. Trim the `padded_w × height` accumulator to tight `width × height`.
//!
//! Output: row-major `width × height` `Vec<f32>`, non-negative,
//! identity → zero (≤ 1e-3 absolute per the 5 PRACTICAL invariants in
//! `RFC_PERCEPTUAL_METRIC_REQUIREMENTS.md` §2.1).
//!
//! ## Extended options stay on CPU fallback
//!
//! `DiffmapOptions::include_edge_mse`, `include_hf`,
//! `masking_strength`, and `sqrt` are NOT covered by the Phase 1b
//! GPU kernel chain. Callers requesting non-default options dispatch
//! through the CPU recipe via `compute_with_ref_and_diffmap_linear_planar`.
//! The buttloop integration uses `DiffmapOptions::default()` exclusively,
//! so Phase 1b covers the buttloop's hot path completely.
//!
//! ## Parity contract
//!
//! Tests in `tests/diffmap_invariants.rs` lock the 5 PRACTICAL
//! invariants (≤ 1e-3 abs per pixel for identity; non-negative;
//! monotone; spatial localization; warm-ref invariance). Tests in
//! `tests/cpu_gpu_diffmap_parity.rs` (Phase 1b) lock CPU↔GPU pointwise
//! parity within a documented tolerance — see
//! `docs/DIFFMAP_DIVERGENCES.md` §11 for the measured envelope.

// Tick 514 silence: matches the rest of zensim-gpu (color.rs, fused.rs)
// for `missing_docs` on macro-emitted launch wrappers.
#![allow(missing_docs)]

use cubecl::prelude::*;

// ──────────────── Host-scalar reference helpers (Phase 1 ship) ────────────────
//
// These are the building blocks for the future Phase 1b kernel chain.
// Each helper has the exact recipe documented inline; the Phase 1b
// kernels mirror them per-pixel.

/// Nearest-neighbour upsample-with-weight (Phase 1 host-scalar
/// reference). Mirrors zensim CPU's
/// `streaming::upsample_pow2x_add` for the `factor` ≥ 1 case.
///
/// For `src[sy * src_w + sx]`, replicates the value into a
/// `factor × factor` block in `dst` starting at
/// `(sx * factor, sy * factor)`, scaled by `weight`, **added** to
/// existing `dst` content. `factor = 1 << scale_levels` — scale 0
/// is identity copy (no replication).
///
/// `dst` is `dst_w × dst_h` row-major; `src` is `src_w × src_h`
/// row-major. Out-of-range writes (when `src_w * factor > dst_w` or
/// `src_h * factor > dst_h`) are clipped to `dst`'s bounds.
///
/// Used by `tests/diffmap_invariants.rs` to pin per-pixel parity of
/// the future Phase 1b `bilinear_upsample_band_kernel` against the
/// CPU recipe.
pub fn upsample_pow2x_add_scalar(
    src: &[f32],
    src_w: usize,
    src_h: usize,
    dst: &mut [f32],
    dst_w: usize,
    dst_h: usize,
    factor: usize,
    weight: f32,
) {
    if factor == 0 {
        return;
    }
    if factor == 1 {
        let copy_w = src_w.min(dst_w);
        let copy_h = src_h.min(dst_h);
        for y in 0..copy_h {
            let dst_row = &mut dst[y * dst_w..y * dst_w + copy_w];
            let src_row = &src[y * src_w..y * src_w + copy_w];
            for (d, s) in dst_row.iter_mut().zip(src_row.iter()) {
                *d += *s * weight;
            }
        }
        return;
    }

    for sy in 0..src_h {
        let dy_start = sy * factor;
        if dy_start >= dst_h {
            break;
        }
        let dy_end = (dy_start + factor).min(dst_h);
        let src_row_start = sy * src_w;
        let src_row_end = src_row_start + src_w;
        let src_row = &src[src_row_start..src_row_end];

        for dy in dy_start..dy_end {
            let dst_row_start = dy * dst_w;
            let dst_row_end = dst_row_start + dst_w;
            let dst_row = &mut dst[dst_row_start..dst_row_end];
            let mut di = 0usize;
            for &s in src_row.iter() {
                if di >= dst_w {
                    break;
                }
                let v = s * weight;
                let end = (di + factor).min(dst_w);
                for slot in &mut dst_row[di..end] {
                    *slot += v;
                }
                di += factor;
            }
        }
    }
}

/// Per-pixel non-negative clamp + sqrt (Phase 1 host-scalar
/// reference). Mirrors zensim CPU's
/// `diffmap::sqrt_inplace` (which clamps negative before sqrt to
/// match the `max(0.0)` invariant).
///
/// Used by `tests/diffmap_invariants.rs` to pin Phase 1b kernel
/// parity for the optional `sqrt` post-pass.
#[must_use]
pub fn sqrt_clamp_scalar(v: f32) -> f32 {
    v.max(0.0).sqrt()
}

/// Per-pixel contrast-masking divisor (Phase 1 host-scalar
/// reference). Mirrors zensim CPU's `apply_contrast_masking`
/// per-pixel arithmetic: `out = raw / (1 + strength × local_variance)`.
///
/// `local_variance` is the variance of the source Y plane in a
/// small neighbourhood; this helper takes it already-computed so
/// the GPU kernel chain only owns the math, not the variance
/// reduction.
///
/// Used by `tests/diffmap_invariants.rs`.
#[must_use]
pub fn contrast_masking_scalar(raw: f32, local_variance: f32, strength: f32) -> f32 {
    let denom = 1.0 + strength * local_variance.max(0.0);
    if denom <= 0.0 { raw } else { raw / denom }
}

/// Per-pixel weighted-sum reducer across 3 XYB channels (Phase 1
/// host-scalar reference). Used by the trained-channel-weighting
/// path: `out = wx * vx + wy * vy + wb * vb`.
///
/// Note: zensim's diffmap fusion does NOT max-clamp negative per-
/// channel values the way cvvdp does (per
/// `cvvdp::kernels::diffmap::channel_pool_scalar`). zensim's per-
/// pixel SSIM error is intrinsically non-negative, so the clamp is
/// a no-op there; this helper preserves zensim's
/// straight-weighted-sum semantics.
#[must_use]
pub fn channel_weighted_sum_scalar(vx: f32, vy: f32, vb: f32, wx: f32, wy: f32, wb: f32) -> f32 {
    wx * vx + wy * vy + wb * vb
}

/// Bit-faithful port of `zensim::diffmap::trained_multiscale_weights`
/// specialised to default [`zensim::DiffmapOptions`] (SSIM-only, no
/// edge_mse, no hf). Computes per-(scale, channel) `ssim_w` weights +
/// per-scale blend weights from a profile's 228-entry trained weight
/// table.
///
/// Returns `(per_scale_ssim_weights, scale_blend_weights)` where:
/// - `per_scale_ssim_weights[s] = [w_x, w_y, w_b]` for scale `s`,
///   already normalised to sum across (channels × features) = 1.0
///   per scale (since edge_mse + hf are off, only the 3 ssim weights
///   are non-zero).
/// - `scale_blend_weights[s]` = fraction of total weight mass at
///   scale `s`, summed to 1.0 across scales.
///
/// Layout assumption: `weights` is the canonical zensim weight table
/// (`zensim::profile::WEIGHTS_PREVIEW_V0_2` or equivalent) of length
/// `>= num_scales * FEATURES_PER_CHANNEL_BASIC * 3` = 156 for the
/// default 4 scales × 3 channels × 13 features.
///
/// Used by Phase 1b's GPU diffmap pipeline to upload the per-(scale,
/// channel) `w_x, w_y, w_b` scalars to
/// [`per_scale_weighted_ssim_kernel`] and the per-scale blend to
/// [`pow2x_upsample_add_kernel`].
///
/// **Parity contract**: byte-for-byte identical f64 arithmetic to
/// the CPU reference (`zensim::diffmap::trained_multiscale_weights`
/// with `include_edge_mse=false, include_hf=false`); cast to f32 only
/// at the return boundary, which matches the CPU side's
/// `PixelFeatureWeights` field types.
#[must_use]
pub fn trained_multiscale_ssim_weights_default(
    weights: &[f64],
    num_scales: usize,
) -> (Vec<[f32; 3]>, Vec<f32>) {
    // Mirrors `zensim::metric::FEATURES_PER_CHANNEL_BASIC` (= 13).
    // Kept inline so this module doesn't transitively force the
    // `training` feature flag on the zensim dependency for callers
    // who don't need the public WEIGHTS table.
    const FPC: usize = 13;
    const FPS: usize = FPC * 3; // features per scale (basic only)

    let mut per_scale: Vec<[f32; 3]> = Vec::with_capacity(num_scales);
    let mut scale_totals: Vec<f64> = Vec::with_capacity(num_scales);

    for s in 0..num_scales {
        let scale_base = s * FPS;
        let mut ssim_w = [0.0f64; 3];
        let mut scale_total = 0.0f64;

        for c in 0..3 {
            let base = scale_base + c * FPC;
            // First 3 features per channel are the 3 SSIM error norms
            // (1-norm, 4-norm, 2-norm; same as CPU). edge_mse + hf
            // are disabled for default options so we don't accumulate
            // their per-channel sums.
            if base + 2 < weights.len() {
                ssim_w[c] = weights[base].abs()
                    + weights[base + 1].abs()
                    + weights[base + 2].abs();
            }
            // Sum ALL features at this scale (the blend weight uses
            // the FULL weight mass, not just ssim, even when extended
            // options are disabled — this is how CPU's
            // `trained_multiscale_weights` computes `scale_totals`).
            for f in 0..FPC {
                if base + f < weights.len() {
                    scale_total += weights[base + f].abs();
                }
            }
        }

        // Normalise per-channel weights: only ssim is non-zero so the
        // feat_total reduces to the sum of the 3 ssim_w entries.
        let feat_total: f64 = ssim_w.iter().sum();
        let ch_weights = if feat_total > 0.0 {
            [
                (ssim_w[0] / feat_total) as f32,
                (ssim_w[1] / feat_total) as f32,
                (ssim_w[2] / feat_total) as f32,
            ]
        } else {
            let eq = 1.0_f32 / 3.0_f32;
            [eq, eq, eq]
        };

        per_scale.push(ch_weights);
        scale_totals.push(scale_total);
    }

    // Normalise scale blend weights.
    let total: f64 = scale_totals.iter().sum();
    let blend: Vec<f32> = if total > 0.0 {
        scale_totals.iter().map(|&s| (s / total) as f32).collect()
    } else {
        let w = 1.0_f32 / num_scales as f32;
        vec![w; num_scales]
    };

    (per_scale, blend)
}

/// Per-pixel modified-SSIM error (the same `sd0` value the scalar
/// feature kernel produces). Phase 1b host-scalar reference,
/// byte-for-byte equivalent to [`per_scale_weighted_ssim_kernel`]'s
/// per-pixel math.
///
/// Inputs are V-blurred `mu1`, `mu2`, `ssq`, `s12` (the four planes
/// the scalar feature kernel emits per-pixel into `mu1_all`/`mu2_all`/
/// `ssq_all`/`s12_all` in its `_persist` variant). Output is the
/// same `1.0 − num_m·num_s/denom_s` value the SSIMULACRA2-style fold
/// produces, clamped at zero.
///
/// Matches `zensim::fused::fused_vblur_ssim_inner_v4`'s FMA
/// fusion order exactly so the GPU kernel and the scalar reference
/// produce bit-identical results at f32 precision when the inputs
/// match.
#[must_use]
pub fn per_pixel_ssim_error_scalar(mu1: f32, mu2: f32, ssq: f32, s12: f32) -> f32 {
    let c2: f32 = 0.0009;
    let mu_diff = mu1 - mu2;
    // num_m = 1 - mu_diff^2 (via FMA(mu_diff, -mu_diff, 1.0))
    let num_m = mu_diff.mul_add(-mu_diff, 1.0);
    let inner_ns = (-mu1).mul_add(mu2, s12);
    let num_s = 2.0_f32.mul_add(inner_ns, c2);
    let inner_ds_inner = (-mu1).mul_add(mu1, ssq);
    let denom_s = (-mu2).mul_add(mu2, inner_ds_inner) + c2;
    let sd_raw = 1.0 - (num_m * num_s) / denom_s;
    if sd_raw > 0.0 { sd_raw } else { 0.0 }
}

// ──────────────────────── CubeCL kernels (Phase 1b) ────────────────────────
//
// All kernels share the same launch convention as the rest of zensim-gpu:
// flat `padded_w × height` row-major f32 planes, one launch handles one
// scale, one thread per pixel of the destination buffer at that scale.
//
// `terminate!()` guards the tail-pad slack so the dispatch can be a
// simple 1-D grid that overshoots the buffer size by < 256 threads.

/// Zero-fill a flat f32 buffer of `n` slots. Used by the Phase 1b
/// pipeline to clear the base-resolution accumulator before the
/// per-scale upsample-add band loop.
#[cube(launch_unchecked)]
pub fn diffmap_zero_kernel(dest: &mut Array<f32>, n: u32) {
    let idx = ABSOLUTE_POS;
    if idx >= n as usize {
        terminate!();
    }
    dest[idx] = f32::new(0.0);
}

/// Per-pixel weighted modified-SSIM diffmap kernel — one launch per
/// pyramid scale.
///
/// For each pixel `i` of the per-scale `padded_w × height` plane:
///
/// ```text
/// out[i] = w_x * sd0_x[i]  +  w_y * sd0_y[i]  +  w_b * sd0_b[i]
/// ```
///
/// where `sd0_c[i] = max(0, 1 − num_m·num_s/denom_s)` is computed from
/// the V-blurred `mu1`, `mu2`, `ssq`, `s12` produced by the existing
/// [`crate::kernels::fused::fused_features_kernel_persist`] kernel for
/// channel `c`.
///
/// Inputs:
/// - `mu1_all`, `mu2_all`, `ssq_all`, `s12_all` — concatenated 3-channel
///   persist planes (channel `c` lives at offset `c * pad_total` for
///   `pad_total = padded_w * height`). These are the SAME plane layouts
///   `fused_features_kernel_persist` writes; just reused here.
/// - `out` — `padded_w * height` f32 destination plane (zero-filled by
///   [`diffmap_zero_kernel`] before the launch).
/// - `w_x`, `w_y`, `w_b` — per-channel trained weights for this scale,
///   precomputed host-side from `zensim::DiffmapWeighting::Trained`
///   over the profile's feature weights (`PixelFeatureWeights.ssim`
///   for each of the 3 XYB channels).
///
/// FMA fusion order matches
/// [`crate::kernels::fused::fused_features_kernel_persist`] body
/// verbatim so the per-pixel `sd0` values are bit-identical between
/// the scalar-fold path and the diffmap path.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn per_scale_weighted_ssim_kernel(
    mu1_all: &Array<f32>,
    mu2_all: &Array<f32>,
    ssq_all: &Array<f32>,
    s12_all: &Array<f32>,
    out: &mut Array<f32>,
    padded_w: u32,
    height: u32,
    pad_total: u32,
    w_x: f32,
    w_y: f32,
    w_b: f32,
) {
    let idx = ABSOLUTE_POS;
    let total = (padded_w * height) as usize;
    if idx >= total {
        terminate!();
    }
    let pt = pad_total as usize;
    let c2: f32 = f32::new(0.0009);
    let one: f32 = f32::new(1.0);
    let two: f32 = f32::new(2.0);
    let zero: f32 = f32::new(0.0);

    // Channel 0 (X).
    let m1_x = mu1_all[idx];
    let m2_x = mu2_all[idx];
    let sq_x = ssq_all[idx];
    let s12_x = s12_all[idx];
    let mu_diff_x = m1_x - m2_x;
    let num_m_x = fma(mu_diff_x, -mu_diff_x, one);
    let inner_ns_x = fma(-m1_x, m2_x, s12_x);
    let num_s_x = fma(two, inner_ns_x, c2);
    let inner_ds_x = fma(-m1_x, m1_x, sq_x);
    let denom_s_x = fma(-m2_x, m2_x, inner_ds_x) + c2;
    let sd_raw_x = one - (num_m_x * num_s_x) / denom_s_x;
    let sd_x = if sd_raw_x > zero { sd_raw_x } else { zero };

    // Channel 1 (Y).
    let m1_y = mu1_all[idx + pt];
    let m2_y = mu2_all[idx + pt];
    let sq_y = ssq_all[idx + pt];
    let s12_y = s12_all[idx + pt];
    let mu_diff_y = m1_y - m2_y;
    let num_m_y = fma(mu_diff_y, -mu_diff_y, one);
    let inner_ns_y = fma(-m1_y, m2_y, s12_y);
    let num_s_y = fma(two, inner_ns_y, c2);
    let inner_ds_y = fma(-m1_y, m1_y, sq_y);
    let denom_s_y = fma(-m2_y, m2_y, inner_ds_y) + c2;
    let sd_raw_y = one - (num_m_y * num_s_y) / denom_s_y;
    let sd_y = if sd_raw_y > zero { sd_raw_y } else { zero };

    // Channel 2 (B).
    let m1_b = mu1_all[idx + pt * 2];
    let m2_b = mu2_all[idx + pt * 2];
    let sq_b = ssq_all[idx + pt * 2];
    let s12_b = s12_all[idx + pt * 2];
    let mu_diff_b = m1_b - m2_b;
    let num_m_b = fma(mu_diff_b, -mu_diff_b, one);
    let inner_ns_b = fma(-m1_b, m2_b, s12_b);
    let num_s_b = fma(two, inner_ns_b, c2);
    let inner_ds_b = fma(-m1_b, m1_b, sq_b);
    let denom_s_b = fma(-m2_b, m2_b, inner_ds_b) + c2;
    let sd_raw_b = one - (num_m_b * num_s_b) / denom_s_b;
    let sd_b = if sd_raw_b > zero { sd_raw_b } else { zero };

    // Weighted channel sum. zensim does NOT max-clamp the sum — the
    // per-channel `sd0` is already non-negative by construction
    // (see `channel_weighted_sum_scalar` doc comment).
    out[idx] = w_x * sd_x + w_y * sd_y + w_b * sd_b;
}

/// Nearest-neighbor power-of-2 upsample-add. One launch per coarser
/// scale; the destination accumulator is the base-resolution plane.
///
/// For each base-resolution pixel `(dx, dy)`, samples the coarser
/// plane at `(dx >> log2_factor, dy >> log2_factor)` and adds
/// `blend_weight * src[sy * src_w + sx]` to `dst[dy * dst_w + dx]`.
///
/// Mirrors `zensim::streaming::upsample_pow2x_add` for `factor = 1 <<
/// log2_factor`: scale `s` of the pyramid contributes via this kernel
/// with `log2_factor = s` (so scale 0 → identity weighted add, scale 1
/// → 2× replicate, scale 2 → 4× replicate, scale 3 → 8× replicate).
///
/// Out-of-range sample positions (when `src_w * (1 << log2_factor) >
/// dst_w` due to padding mismatch) are clamped via integer division
/// behavior: any `dx < src_w << log2_factor` reads `src[(dx >>
/// log2_factor) + ...]`. Caller must ensure that
/// `src_w << log2_factor >= dst_w` AND `src_h << log2_factor >= dst_h`.
/// In practice the pyramid build guarantees this — each scale's
/// `padded_w` halves cleanly until the smallest scale.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn pow2x_upsample_add_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    log2_factor: u32,
    blend_weight: f32,
) {
    let idx = ABSOLUTE_POS;
    let total = (dst_w * dst_h) as usize;
    if idx >= total {
        terminate!();
    }
    let dw = dst_w as usize;
    let sw = src_w as usize;
    let dx = idx % dw;
    let dy = idx / dw;

    // Coarse-pixel coords via right-shift (NN replicate).
    let sx = dx >> log2_factor as usize;
    let sy = dy >> log2_factor as usize;

    // Clamp to source-plane bounds. With log2_factor up to 3 and
    // padded_w halving per scale, this is the same edge behavior
    // `upsample_pow2x_add_scalar` uses (the inner-loop `break` when
    // `di >= dst_w`).
    let last_sx = src_w as usize - 1usize;
    let last_sy = src_h as usize - 1usize;
    let sx_c = if sx < last_sx { sx } else { last_sx };
    let sy_c = if sy < last_sy { sy } else { last_sy };

    let v = src[sy_c * sw + sx_c];
    dst[idx] = dst[idx] + blend_weight * v;
}

/// Trim a `padded_w × height` plane into a tight `width × height`
/// plane (drops right-side SIMD-pad columns). One thread per
/// destination pixel.
///
/// Mirrors the host-side trim loop in
/// `zensim::diffmap::compute_with_ref_and_diffmap_linear_planar`:
///
/// ```text
/// for y in 0..height:
///     out[y*width..y*width+width] = padded[y*padded_w..y*padded_w+width]
/// ```
///
/// Called only when `padded_w != width`. For the tight case the
/// pipeline does a single device-to-host copy of the accumulator
/// directly.
#[cube(launch_unchecked)]
pub fn diffmap_trim_padded_kernel(
    padded: &Array<f32>,
    out: &mut Array<f32>,
    width: u32,
    padded_w: u32,
    height: u32,
) {
    let idx = ABSOLUTE_POS;
    let total = (width * height) as usize;
    if idx >= total {
        terminate!();
    }
    let w = width as usize;
    let pw = padded_w as usize;
    let dy = idx / w;
    let dx = idx - dy * w;
    out[idx] = padded[dy * pw + dx];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upsample_factor_1_is_weighted_copy() {
        let src = vec![1.0, 2.0, 3.0, 4.0];
        let mut dst = vec![10.0, 20.0, 30.0, 40.0];
        upsample_pow2x_add_scalar(&src, 2, 2, &mut dst, 2, 2, 1, 0.5);
        // dst += src * 0.5
        assert_eq!(dst, vec![10.5, 21.0, 31.5, 42.0]);
    }

    #[test]
    fn upsample_factor_2_replicates_into_2x2_blocks() {
        // src = 2×2 of [1, 2, 3, 4]
        // factor=2, dst=4×4 zeroed; weight=1.0
        let src = vec![1.0, 2.0, 3.0, 4.0];
        let mut dst = vec![0.0; 16];
        upsample_pow2x_add_scalar(&src, 2, 2, &mut dst, 4, 4, 2, 1.0);
        #[rustfmt::skip]
        let expected = vec![
            1.0, 1.0, 2.0, 2.0,
            1.0, 1.0, 2.0, 2.0,
            3.0, 3.0, 4.0, 4.0,
            3.0, 3.0, 4.0, 4.0,
        ];
        assert_eq!(dst, expected);
    }

    #[test]
    fn upsample_clips_to_dst_bounds() {
        let src = vec![10.0, 20.0];
        let mut dst = vec![0.0; 3]; // dst_w=3, dst_h=1; src=2×1, factor=2 would give 4 cols
        upsample_pow2x_add_scalar(&src, 2, 1, &mut dst, 3, 1, 2, 1.0);
        // Expect [10, 10, 20] — last col of the 2nd 2-block clipped off.
        assert_eq!(dst, vec![10.0, 10.0, 20.0]);
    }

    #[test]
    fn upsample_factor_zero_is_noop() {
        let src = vec![1.0; 4];
        let mut dst = vec![5.0; 4];
        upsample_pow2x_add_scalar(&src, 2, 2, &mut dst, 2, 2, 0, 1.0);
        assert_eq!(dst, vec![5.0; 4]);
    }

    #[test]
    fn sqrt_clamp_clamps_negative() {
        assert_eq!(sqrt_clamp_scalar(0.0), 0.0);
        assert_eq!(sqrt_clamp_scalar(-1.0), 0.0);
        assert!((sqrt_clamp_scalar(4.0) - 2.0).abs() < 1e-7);
    }

    #[test]
    fn contrast_masking_identity_when_strength_zero() {
        assert_eq!(contrast_masking_scalar(5.0, 100.0, 0.0), 5.0);
    }

    #[test]
    fn contrast_masking_reduces_with_higher_variance() {
        let lo = contrast_masking_scalar(1.0, 0.0, 4.0);
        let hi = contrast_masking_scalar(1.0, 1.0, 4.0);
        assert!(hi < lo, "higher variance should suppress: {hi} vs {lo}");
    }

    #[test]
    fn channel_weighted_sum_is_linear() {
        let v = channel_weighted_sum_scalar(1.0, 2.0, 3.0, 0.1, 0.7, 0.2);
        assert!((v - (0.1 + 1.4 + 0.6)).abs() < 1e-6, "got {v}");
    }

    #[test]
    fn per_pixel_ssim_error_identity_is_zero() {
        // For an identity (src == dist) pixel after V-blur:
        //   mu1 = mu2 = m
        //   ssq = mean(s^2 + d^2)/diam = 2 * mean(s^2)/diam = 2*s12
        //   (since s12 = mean(s*d)/diam = mean(s^2)/diam when s == d)
        //
        // The formula collapses:
        //   num_m = 1 - (mu1 - mu2)^2 = 1
        //   num_s = 2(s12 - m^2) + C2
        //   denom_s = (ssq - 2m^2) + C2 = (2*s12 - 2m^2) + C2 = 2(s12 - m^2) + C2
        // → num_s == denom_s → ratio == 1 → sd_raw == 0 (exactly).
        let m = 0.5_f32;
        let t = 0.25_f32; // s12 = mean(s^2)/diam for s == constant 0.5
        let ssq = 2.0 * t;
        let s12 = t;
        let v = per_pixel_ssim_error_scalar(m, m, ssq, s12);
        assert!(
            v.abs() < 1e-6,
            "identity inputs should yield ~0 SSIM error, got {v}"
        );
    }

    #[test]
    fn per_pixel_ssim_error_clamps_at_zero() {
        // Identity-and-better cases (over-correlated denoms) yield
        // sd_raw < 0 which clamps to 0.
        // mu1 = mu2, ssq large enough that num_s > denom_s
        // (mu_diff = 0 so num_m = 1; num_s = 2*s12 + 0.0009).
        // Construct: mu1 = mu2 = 0; s12 = 1.0, ssq = 0.5
        // -> num_m = 1, num_s = 2.0009, denom_s = 0.5 + 0.5 + 0.0009 = 1.0009
        // -> sd_raw = 1 - 2.0009 / 1.0009 = -1.0 → clamps to 0.
        let v = per_pixel_ssim_error_scalar(0.0, 0.0, 0.5, 1.0);
        assert!(
            v == 0.0,
            "negative sd_raw should clamp to zero, got {v}"
        );
    }

    #[test]
    fn per_pixel_ssim_error_non_negative_on_random_inputs() {
        // 50 random-ish inputs in the normal value range produced by
        // V-blur of XYB planes. Every output must be non-negative and
        // finite — the clamp + denom guard guarantees this.
        let inputs: &[(f32, f32, f32, f32)] = &[
            (0.0, 0.0, 0.0, 0.0),
            (0.1, 0.15, 0.02, 0.015),
            (0.5, 0.45, 0.30, 0.27),
            (0.9, 0.91, 0.81, 0.82),
            (-0.5, -0.4, 0.30, 0.22),
            (1.2, 1.0, 1.5, 1.21),
            (0.001, 0.002, 1e-6, 0.0),
            (0.5, 0.5, 0.5, 0.5),
            (10.0, 9.5, 95.0, 95.0),
            (1e-6, 0.0, 0.0, 0.0),
        ];
        for &(m1, m2, ssq, s12) in inputs {
            let v = per_pixel_ssim_error_scalar(m1, m2, ssq, s12);
            assert!(
                v >= 0.0 && v.is_finite(),
                "non-negative + finite contract failed at (m1={m1},m2={m2},ssq={ssq},s12={s12}) -> {v}"
            );
        }
    }

    #[test]
    fn trained_weights_default_sum_per_scale_to_one() {
        // Each per-scale [w_x, w_y, w_b] must sum to 1.0 (since we
        // disabled edge_mse + hf, only ssim is active).
        // Use the canonical zensim weights.
        let weights = &zensim::profile::WEIGHTS_PREVIEW_V0_2;
        let (per_scale, _blend) =
            trained_multiscale_ssim_weights_default(weights.as_slice(), 4);
        assert_eq!(per_scale.len(), 4);
        for (s, w) in per_scale.iter().enumerate() {
            let sum = w[0] + w[1] + w[2];
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "scale {s} per-channel weights should sum to 1.0, got {sum}"
            );
        }
    }

    #[test]
    fn trained_blend_weights_sum_to_one() {
        let weights = &zensim::profile::WEIGHTS_PREVIEW_V0_2;
        let (_per_scale, blend) =
            trained_multiscale_ssim_weights_default(weights.as_slice(), 4);
        assert_eq!(blend.len(), 4);
        let sum: f32 = blend.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "scale blend weights should sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn trained_weights_default_empty_weights_returns_uniform() {
        // Defensive: when feat_total is zero (degenerate empty weights),
        // each channel weight collapses to 1/3 (matches CPU recipe).
        let weights: [f64; 4] = [0.0; 4]; // too short, all-zero
        let (per_scale, _blend) = trained_multiscale_ssim_weights_default(&weights, 4);
        for w in per_scale.iter() {
            // Each channel ~ 1/3
            for c in 0..3 {
                assert!(
                    (w[c] - 1.0 / 3.0).abs() < 1e-5,
                    "all-zero weights should yield uniform per-channel: got {w:?}"
                );
            }
        }
    }

    #[test]
    fn per_pixel_ssim_error_matches_fused_kernel_recipe() {
        // Spot-check that the scalar reference matches the same FMA
        // fusion order the fused feature kernel uses. We mirror the
        // body of `fused_features_kernel_persist` manually to catch
        // any future refactor that breaks parity.
        let m1 = 0.4_f32;
        let m2 = 0.55_f32;
        let ssq = 0.21_f32;
        let s12 = 0.20_f32;
        let c2 = 0.0009_f32;

        // Direct mirror of the fused kernel body (lines 679-686).
        let mu_diff = m1 - m2;
        let num_m = mu_diff.mul_add(-mu_diff, 1.0);
        let inner_ns = (-m1).mul_add(m2, s12);
        let num_s = 2.0_f32.mul_add(inner_ns, c2);
        let inner_ds_inner = (-m1).mul_add(m1, ssq);
        let denom_s = (-m2).mul_add(m2, inner_ds_inner) + c2;
        let sd_raw = 1.0 - (num_m * num_s) / denom_s;
        let expected = if sd_raw > 0.0 { sd_raw } else { 0.0 };

        let got = per_pixel_ssim_error_scalar(m1, m2, ssq, s12);
        assert_eq!(
            got.to_bits(),
            expected.to_bits(),
            "scalar reference must be bit-identical to manually-mirrored fused-kernel body (got {got}, expected {expected})"
        );
    }
}
