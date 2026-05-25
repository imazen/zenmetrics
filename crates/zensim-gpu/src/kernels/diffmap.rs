//! Per-pixel diffmap helpers for zensim-gpu Phase 1.
//!
//! This module ships **host-scalar reference helpers** for the per-pixel
//! diffmap output. The actual diffmap *production* in Phase 1 is
//! delegated to the canonical CPU `zensim::Zensim::compute_with_ref_and_diffmap_linear_planar`
//! path via host-side fallback — see
//! [`crate::pipeline::Zensim::score_with_diffmap`] and
//! `docs/DIFFMAP_DIVERGENCES.md` for the rationale.
//!
//! ## Why CPU-fallback in Phase 1
//!
//! Porting zensim's per-band SSIM error → multi-scale bilinear-upsample
//! → trained-blend → optional contrast-masking pipeline as CubeCL
//! kernels is a multi-week chunk on top of the existing 4-scale
//! feature pyramid. The Phase 1 brief (RFC #4 §3) accepts a partial
//! ship that:
//!
//! 1. Exposes the **complete public API surface** (7 methods on
//!    [`Zensim<R>`](crate::pipeline::Zensim) + 7 on
//!    [`ZensimOpaque`](crate::opaque::ZensimOpaque)) so the
//!    jxl-encoder buttloop integration can wire diffmap-aware backends
//!    immediately.
//! 2. Produces **bit-exact correct diffmaps** that match the CPU
//!    reference recipe pointwise (because we ARE the CPU reference
//!    for that frame).
//! 3. Captures **wall-overhead measurements** so Phase 1b (true GPU
//!    diffmap kernels) can be scoped against a known baseline.
//!
//! The host-scalar helpers below are the building blocks the future
//! GPU kernel chain will need, and the parity gate Phase 1b tests
//! must clear pointwise.
//!
//! ## Recipe (canonical zensim CPU diffmap)
//!
//! Implemented in `zensim::diffmap::compute_with_ref_and_diffmap_linear_planar`:
//!
//! 1. Per-scale per-channel SSIM error compute (via the same
//!    pipeline that produces zensim's scalar features).
//! 2. Optional edge-MSE + HF feature accumulation (per [`DiffmapOptions`]).
//! 3. Per-scale per-channel weighted reduction to one f32 plane per
//!    scale.
//! 4. Bilinear upsample of each coarser-scale plane to base resolution.
//!    (`upsample_pow2x_add` — nearest-neighbor by `factor = 1 << scale`).
//! 5. Sum across scales with the profile's trained `scale_blend` weights.
//! 6. Optional `contrast_masking` post-pass.
//! 7. Optional `sqrt` post-pass.
//!
//! Output: row-major `W × H` `Vec<f32>`, non-negative, identity → zero
//! (1e-7 absolute on f32 SIMD round-off path).
//!
//! ## Future Phase 1b kernel chain
//!
//! When the pure-GPU port lands, the new kernels will be:
//!
//! - `ssim_error_per_pixel_kernel` — per-pixel SSIM error from
//!   already-computed mu1/mu2/ssq/s12 persist planes. Already partially
//!   present in `fused::fused_features_kernel_persist` as a by-product;
//!   needs a separate output buffer rather than accumulation.
//! - `bilinear_upsample_band_kernel` — per-scale 2^s × replicate
//!   (mirror cvvdp-gpu's `kernels::diffmap::bilinear_sample_scalar`).
//! - `multi_scale_blend_kernel` — sum-with-weights across scales at
//!   base resolution.
//! - `contrast_masking_kernel` — optional post-pass.
//!
//! Tests in `tests/diffmap_invariants.rs` lock the host-scalar
//! reference values; Phase 1b kernel impls must match within 1e-6
//! absolute per-pixel.

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
}
