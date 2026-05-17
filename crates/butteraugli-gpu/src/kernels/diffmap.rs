//! Diffmap-combination + L2 difference accumulators.
//!
//! Translated from `butteraugli-cuda-kernel/src/diffmap.rs`. The fused
//! reduction kernel that produces the score lives in `reduction.rs`.

use cubecl::prelude::*;

const GLOBAL_SCALE: f32 = 0.070_936_545;

/// MaskY for the AC term. Mirrors the CPU butteraugli `mask_y` function.
#[cube]
fn mask_y(delta: f32) -> f32 {
    const OFFSET: f32 = 0.829_591_75;
    const SCALER: f32 = 0.451_936_92;
    const MUL: f32 = 2.548_594_4;
    let c = MUL / (SCALER * delta + OFFSET);
    let retval = GLOBAL_SCALE * (1.0 + c);
    retval * retval
}

/// MaskDcY for the DC term.
#[cube]
fn mask_dc_y(delta: f32) -> f32 {
    const OFFSET: f32 = 0.200_255_78;
    const SCALER: f32 = 3.874_494;
    const MUL: f32 = 0.505_054_53;
    let c = MUL / (SCALER * delta + OFFSET);
    let retval = GLOBAL_SCALE * (1.0 + c);
    retval * retval
}

/// `dst = sqrt(maskY · ΣAC + maskDcY · ΣDC)` — the per-pixel butteraugli diffmap.
/// The X (chroma) channel contributions are pre-scaled by `xmul`
/// (default 1.0; set to 0.5 to halve chroma penalty etc.) — matches
/// CPU butteraugli's `combine_channels_to_diffmap_fused`.
#[cube(launch_unchecked)]
pub fn compute_diffmap_kernel(
    mask: &Array<f32>,
    block_diff_dc0: &Array<f32>,
    block_diff_dc1: &Array<f32>,
    block_diff_dc2: &Array<f32>,
    block_diff_ac0: &Array<f32>,
    block_diff_ac1: &Array<f32>,
    block_diff_ac2: &Array<f32>,
    dst: &mut Array<f32>,
    xmul: f32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= dst.len() {
        terminate!();
    }
    let m = mask[idx];
    let mac = mask_y(m);
    let mdc = mask_dc_y(m);
    let ac = block_diff_ac0[idx] * xmul + block_diff_ac1[idx] + block_diff_ac2[idx];
    let dc = block_diff_dc0[idx] * xmul + block_diff_dc1[idx] + block_diff_dc2[idx];
    dst[idx] = f32::sqrt(mac * ac + mdc * dc);
}

/// Accumulate a weighted squared-difference into `dst`.
#[cube(launch_unchecked)]
pub fn l2_diff_kernel(src1: &Array<f32>, src2: &Array<f32>, dst: &mut Array<f32>, weight: f32) {
    let idx = ABSOLUTE_POS;
    if idx >= dst.len() {
        terminate!();
    }
    let diff = src1[idx] - src2[idx];
    dst[idx] = dst[idx] + weight * diff * diff;
}

/// Write-only L2 diff — overwrites `dst` (no accumulation). Use for the
/// first contribution to a per-channel accumulator so the buffer doesn't
/// need a separate zero pass.
#[cube(launch_unchecked)]
pub fn l2_diff_write_kernel(
    src1: &Array<f32>,
    src2: &Array<f32>,
    dst: &mut Array<f32>,
    weight: f32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= dst.len() {
        terminate!();
    }
    let diff = src1[idx] - src2[idx];
    dst[idx] = weight * diff * diff;
}

/// Write-only L2 diff for THREE channels in a single launch — overwrites
/// all three `dst` planes. Used by `compute_dc_diff` (which always runs
/// all three LF channels back-to-back). Saves 2 kernel launches + 2
/// launch-latency round-trips per iter vs three separate l2_diff_write
/// launches. Each channel can have a different weight.
#[cube(launch_unchecked)]
pub fn l2_diff_write_3ch_kernel(
    src1_0: &Array<f32>,
    src2_0: &Array<f32>,
    dst_0: &mut Array<f32>,
    src1_1: &Array<f32>,
    src2_1: &Array<f32>,
    dst_1: &mut Array<f32>,
    src1_2: &Array<f32>,
    src2_2: &Array<f32>,
    dst_2: &mut Array<f32>,
    weight_0: f32,
    weight_1: f32,
    weight_2: f32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= dst_0.len() {
        terminate!();
    }
    let d0 = src1_0[idx] - src2_0[idx];
    let d1 = src1_1[idx] - src2_1[idx];
    let d2 = src1_2[idx] - src2_2[idx];
    dst_0[idx] = weight_0 * d0 * d0;
    dst_1[idx] = weight_1 * d1 * d1;
    dst_2[idx] = weight_2 * d2 * d2;
}

/// Batched broadcast l2_diff: `src1` is one plane (cached reference),
/// `src2` and `dst` are `N` planes packed contiguously. Accumulates.
#[cube(launch_unchecked)]
pub fn l2_diff_broadcast_batched_kernel(
    src1: &Array<f32>,
    src2: &Array<f32>,
    dst: &mut Array<f32>,
    plane_stride: u32,
    weight: f32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= dst.len() {
        terminate!();
    }
    let local = idx - (idx / (plane_stride as usize)) * (plane_stride as usize);
    let diff = src1[local] - src2[idx];
    dst[idx] = dst[idx] + weight * diff * diff;
}

/// Batched broadcast write-only l2_diff. Overwrites.
#[cube(launch_unchecked)]
pub fn l2_diff_write_broadcast_batched_kernel(
    src1: &Array<f32>,
    src2: &Array<f32>,
    dst: &mut Array<f32>,
    plane_stride: u32,
    weight: f32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= dst.len() {
        terminate!();
    }
    let local = idx - (idx / (plane_stride as usize)) * (plane_stride as usize);
    let diff = src1[local] - src2[idx];
    dst[idx] = weight * diff * diff;
}

/// Batched broadcast asymmetric L2 diff.
#[cube(launch_unchecked)]
pub fn l2_asym_diff_broadcast_batched_kernel(
    src1: &Array<f32>,
    src2: &Array<f32>,
    dst: &mut Array<f32>,
    plane_stride: u32,
    weight_gt: f32,
    weight_lt: f32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= dst.len() {
        terminate!();
    }
    let local = idx - (idx / (plane_stride as usize)) * (plane_stride as usize);
    let v0 = src1[local];
    let v1 = src2[idx];
    let vw_gt = weight_gt * 0.8;
    let vw_lt = weight_lt * 0.8;
    let diff = v0 - v1;
    let mut total = dst[idx] + diff * diff * vw_gt;

    let fabs0 = f32::abs(v0);
    let too_small = 0.4 * fabs0;
    let too_big = fabs0;

    let v = if v0 < 0.0 {
        if v1 > -too_small {
            v1 + too_small
        } else if v1 < -too_big {
            -v1 - too_big
        } else {
            f32::new(0.0)
        }
    } else if v1 < too_small {
        too_small - v1
    } else if v1 > too_big {
        v1 - too_big
    } else {
        f32::new(0.0)
    };

    total += vw_lt * v * v;
    dst[idx] = total;
}

/// Asymmetric L2 — primary squared diff plus a half-open penalty for
/// distorted values that drop too far below or rise too far above the
/// reference's magnitude band. Matches the CPU `L2DiffAsymmetric`.
#[cube(launch_unchecked)]
pub fn l2_asym_diff_kernel(
    src1: &Array<f32>,
    src2: &Array<f32>,
    dst: &mut Array<f32>,
    weight_gt: f32,
    weight_lt: f32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= dst.len() {
        terminate!();
    }
    let v0 = src1[idx];
    let v1 = src2[idx];
    let vw_gt = weight_gt * 0.8;
    let vw_lt = weight_lt * 0.8;
    let diff = v0 - v1;
    let mut total = dst[idx] + diff * diff * vw_gt;

    let fabs0 = f32::abs(v0);
    let too_small = 0.4 * fabs0;
    let too_big = fabs0;

    let v = if v0 < 0.0 {
        if v1 > -too_small {
            v1 + too_small
        } else if v1 < -too_big {
            -v1 - too_big
        } else {
            f32::new(0.0)
        }
    } else if v1 < too_small {
        too_small - v1
    } else if v1 > too_big {
        v1 - too_big
    } else {
        f32::new(0.0)
    };

    total += vw_lt * v * v;
    dst[idx] = total;
}
