//! Per-pixel diffmap construction.
//!
//! ## Why a diffmap
//!
//! cvvdp natively produces a scalar JOD. JPEG XL's iterative
//! quantization loop needs a per-pixel "where is the error" map for
//! the same reason butteraugli produces a diffmap — to decide which
//! blocks to refine. This module defines the recipe used by both the
//! cvvdp port AND the eventual cvvdp-gpu diffmap extension (so
//! the two impls stay byte-comparable at the per-pixel level).
//!
//! ## Recipe
//!
//! Per band `k`, the masking stage produces `D_per_ch[c][i]` —
//! per-pixel per-channel masked error in that band. We:
//!
//! 1. Upsample each band's `D_per_ch[c]` to base resolution
//!    via **bilinear interpolation** (one upsample per (band,
//!    channel)).
//! 2. Sum across bands per channel with the same `BAND_W` shape
//!    cvvdp's scalar pool uses (`1.0` for non-baseband; the
//!    `BASEBAND_W[c]` triple for the coarsest band).
//! 3. Combine across channels via the same `BETA_CH` Minkowski
//!    exponent the global pool uses (with `PER_CH_W` channel
//!    weights, both currently `[1, 1, 1]`).
//!
//! Result: one `WxH` `Vec<f32>`, row-major, contiguous (stride =
//! width). The scalar JOD and the diffmap remain consistent
//! because both are folded with identical Minkowski exponents and
//! channel weights; the spatial dimension is reduced by averaging
//! the same `D` values, just *with the spatial-average step
//! deferred until after the cross-channel fold instead of done
//! per-band*. This produces a meaningful per-pixel error signal
//! (large where the masked error is large in any channel/band)
//! without re-running the pipeline.
//!
//! ## Invariants
//!
//! - Identical inputs → diffmap is **identically zero** (asserted in
//!   `tests/diffmap_invariants.rs`).
//! - Diffmap values are non-negative (masked error is non-negative
//!   before any spatial fold).
//! - For any non-identical input, `diffmap.iter().sum() > 0`.
//! - `diffmap.len() == width * height`.

use alloc::vec;
use alloc::vec::Vec;

use crate::pool::{BASEBAND_W, BETA_CH, PER_CH_W};

/// Accumulator: per-channel summed-band per-pixel error.
pub(crate) struct DiffmapAccum {
    pub w: usize,
    pub h: usize,
    /// `[ch][i]` at base resolution.
    pub channels: [Vec<f32>; 3],
}

impl DiffmapAccum {
    pub(crate) fn new(w: usize, h: usize) -> Self {
        Self {
            w,
            h,
            channels: [vec![0.0; w * h], vec![0.0; w * h], vec![0.0; w * h]],
        }
    }
}

/// Bilinear-upsample `src` (size `bw × bh`) into `dst` (size `w × h`),
/// then add into `dst_accum[ch]` scaled by `w_ch`.
///
/// We compute the upsample on-the-fly into the accumulator without
/// allocating a temporary upsample buffer per band — saves
/// `n_bands × 3 × w × h × 4` bytes of intermediate storage on a
/// 1024² image with 8 bands × 3 channels ≈ 96 MB of intermediates.
fn upsample_add(
    src: &[f32],
    bw: usize,
    bh: usize,
    dst: &mut [f32],
    dw: usize,
    dh: usize,
    weight: f32,
) {
    if bw == dw && bh == dh {
        // Tightest path — same resolution, just add.
        for i in 0..dw * dh {
            dst[i] += src[i] * weight;
        }
        return;
    }
    let sx = bw as f32 / dw as f32;
    let sy = bh as f32 / dh as f32;
    for y in 0..dh {
        // Sample center of dest pixel → src.
        let fy = (y as f32 + 0.5) * sy - 0.5;
        let y0 = fy.floor();
        let y1 = y0 + 1.0;
        let wy1 = fy - y0;
        let wy0 = 1.0 - wy1;
        let iy0 = (y0.max(0.0) as usize).min(bh - 1);
        let iy1 = (y1.max(0.0) as usize).min(bh - 1);
        for x in 0..dw {
            let fx = (x as f32 + 0.5) * sx - 0.5;
            let x0 = fx.floor();
            let x1 = x0 + 1.0;
            let wx1 = fx - x0;
            let wx0 = 1.0 - wx1;
            let ix0 = (x0.max(0.0) as usize).min(bw - 1);
            let ix1 = (x1.max(0.0) as usize).min(bw - 1);
            let v00 = src[iy0 * bw + ix0];
            let v01 = src[iy0 * bw + ix1];
            let v10 = src[iy1 * bw + ix0];
            let v11 = src[iy1 * bw + ix1];
            let v = v00 * (wx0 * wy0) + v01 * (wx1 * wy0) + v10 * (wx0 * wy1) + v11 * (wx1 * wy1);
            dst[y * dw + x] += v * weight;
        }
    }
}

/// Add band `k`'s per-channel masked-error map into the diffmap
/// accumulator. Caller specifies whether this is the baseband (for
/// per-channel weight selection).
pub(crate) fn accumulate_band_diffmap(
    accum: &mut DiffmapAccum,
    d_per_ch: &[Vec<f32>; 3],
    bw: usize,
    bh: usize,
    is_baseband: bool,
    _n_levels: usize,
) {
    let dw = accum.w;
    let dh = accum.h;
    for c in 0..3 {
        let band_w = if is_baseband { BASEBAND_W[c] } else { 1.0 };
        let w_ch = band_w * PER_CH_W[c];
        upsample_add(&d_per_ch[c], bw, bh, &mut accum.channels[c], dw, dh, w_ch);
    }
}

/// Fold the per-channel accumulators into a single per-pixel
/// non-negative diffmap via the `BETA_CH`-norm Minkowski combine.
pub(crate) fn finalize_diffmap(accum: DiffmapAccum) -> Vec<f32> {
    let n = accum.w * accum.h;
    let mut out = vec![0.0_f32; n];
    let p = BETA_CH;
    for i in 0..n {
        let a = accum.channels[0][i].max(0.0);
        let rg = accum.channels[1][i].max(0.0);
        let vy = accum.channels[2][i].max(0.0);
        // p-norm without safe_pow epsilon — diffmap values are already
        // non-negative band sums; we don't need the differentiability
        // hack here.
        let sum = a.powf(p) + rg.powf(p) + vy.powf(p);
        out[i] = sum.powf(1.0 / p);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upsample_add_constant_doubles() {
        let src = vec![2.0_f32; 4 * 4];
        let mut dst = vec![1.0_f32; 8 * 8];
        upsample_add(&src, 4, 4, &mut dst, 8, 8, 1.0);
        // Each dst pixel was 1.0, accumulator added 2.0 → 3.0.
        for &v in &dst {
            assert!((v - 3.0).abs() < 1e-5);
        }
    }

    #[test]
    fn upsample_add_same_size_is_passthrough() {
        let src: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut dst = vec![0.0_f32; 16];
        upsample_add(&src, 4, 4, &mut dst, 4, 4, 1.0);
        for i in 0..16 {
            assert_eq!(dst[i], src[i]);
        }
    }

    #[test]
    fn finalize_zero_accum_yields_zero() {
        let accum = DiffmapAccum::new(8, 8);
        let dm = finalize_diffmap(accum);
        assert_eq!(dm.len(), 64);
        for v in &dm {
            assert_eq!(*v, 0.0);
        }
    }
}
