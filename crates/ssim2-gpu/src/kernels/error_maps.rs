//! SSIMULACRA2 per-pixel error maps.
//!
//! Pointwise kernel; 7 inputs in, 3 outputs out (all single-plane f32):
//!
//! - **ssim** =  `max(0, 1 − num_m · num_s / denom_s)`
//!   - `num_m  = 1 − (mu1 − mu2)²`
//!   - `num_s  = 2·(sigma12 − mu1·mu2) + C2`
//!   - `denom_s = (sigma11 − mu1²) + (sigma22 − mu2²) + C2`
//! - **artifact**     = `max(0,  d1)`
//! - **detail_loss**  = `max(0, −d1)`, where
//!   `d1 = (1 + |dist − mu2|) / (1 + |src − mu1|) − 1`
//!
//! Matches `ssimulacra2::ssim_map` and `edge_diff_map` (both pointwise,
//! same constants, no boundary handling needed). Verbatim from
//! `ssimulacra2-cuda-kernel/src/error_maps.rs`.
//!
//! C2 = 0.0009.

use cubecl::prelude::*;

const C2: f32 = 0.0009;

/// Pointwise broadcast multiply: `out = a_broadcast · b_batched`.
///
/// `a` is a single plane (length = `plane_stride`) shared across the
/// batch; `b` and `out` are batched buffers (length =
/// `plane_stride · batch_size`). Used by `Ssim2Batch::compute_batch`
/// to compute `sigma12 = ref_xyb · dis_xyb_batched`.
#[cube(launch_unchecked)]
pub fn pointwise_mul_broadcast_batched_kernel(
    a: &Array<f32>,
    b: &Array<f32>,
    out: &mut Array<f32>,
    plane_stride: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= out.len() {
        terminate!();
    }
    let pl = plane_stride as usize;
    let local = idx - (idx / pl) * pl;
    out[idx] = a[local] * b[idx];
}

/// Compute the three SSIMULACRA2 per-pixel error maps for one channel.
/// All input/output buffers are single-plane f32 of length `n_pixels`.
#[cube(launch_unchecked)]
pub fn error_maps_kernel(
    source: &Array<f32>,
    distorted: &Array<f32>,
    mu1: &Array<f32>,
    mu2: &Array<f32>,
    sigma11: &Array<f32>,
    sigma22: &Array<f32>,
    sigma12: &Array<f32>,
    out_ssim: &mut Array<f32>,
    out_artifact: &mut Array<f32>,
    out_detail: &mut Array<f32>,
) {
    let idx = ABSOLUTE_POS;
    let n = out_ssim.len();
    if idx >= n {
        terminate!();
    }
    let m1 = mu1[idx];
    let m2 = mu2[idx];
    let s11 = sigma11[idx];
    let s22 = sigma22[idx];
    let s12 = sigma12[idx];
    let src = source[idx];
    let dis = distorted[idx];

    let mu11 = m1 * m1;
    let mu22 = m2 * m2;
    let mu12 = m1 * m2;
    let mu_diff = m1 - m2;
    let num_m = 1.0 - mu_diff * mu_diff;
    let num_s = 2.0 * (s12 - mu12) + C2;
    let denom_s = (s11 - mu11) + (s22 - mu22) + C2;
    let mut d_ssim = 1.0 - (num_m * num_s) / denom_s;
    if d_ssim < 0.0 {
        d_ssim = 0.0;
    }
    out_ssim[idx] = d_ssim;

    let denom = 1.0 / (1.0 + f32::abs(src - m1));
    let numer = 1.0 + f32::abs(dis - m2);
    let d1 = numer * denom - 1.0;

    let art = if d1 > 0.0 { d1 } else { f32::new(0.0) };
    let dl = if d1 < 0.0 { -d1 } else { f32::new(0.0) };
    out_artifact[idx] = art;
    out_detail[idx] = dl;
}

/// Pointwise multiply with an offset into the `a` buffer:
/// `out[i] = a[i + a_offset] * b[i]`.
///
/// Used by strip-mode mode E to compute `sigma12_strip = ref_xyb_full_slice
/// · dis_xyb_strip` without materialising the slice in a separate
/// buffer. `a` is the full-image pre-transpose ref-XYB plane (length
/// `full_n`); `a_offset` is the starting flat index of the strip's
/// body+halo in that plane (= `strip_top_at_scale * image_w_at_scale`).
/// `b` and `out` are the strip-shaped dist and sigma12 planes.
#[cube(launch_unchecked)]
pub fn pointwise_mul_offset_kernel(
    a: &Array<f32>,
    b: &Array<f32>,
    out: &mut Array<f32>,
    a_offset: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= out.len() {
        terminate!();
    }
    let off = a_offset as usize;
    out[idx] = a[off + idx] * b[idx];
}

/// Strip-mode error maps with cached **full-image** reference buffers.
///
/// Mode E from task #46 — "ref-full + dist-strip cached-ref". The
/// reference-side state (`source` = raw transposed XYB, `mu1`,
/// `sigma11`) lives in full-image-sized transposed buffers cached on
/// device by an earlier `set_reference` call. The distorted side and
/// the cross-terms (`distorted`, `mu2`, `sigma22`, `sigma12`) live in
/// the strip-sized transposed buffers that the strip pipeline already
/// allocates. The three error maps are written into strip-sized output
/// buffers, ready for the existing strip-aware reduction launcher.
///
/// Buffer orientation: all inputs and outputs are in the SSIMULACRA2
/// "transposed" orientation produced by `kernels::transpose` — `dst[r, c]`
/// at flat index `r * inner_stride + c`, where the inner dimension is
/// the **original frame's Y axis** and the outer dimension is the
/// original frame's X axis (= the kernel-internal "width" / "outer
/// stride" parameter). Strip buffers have `inner_stride = strip_h`;
/// full buffers have `inner_stride = image_h` at the matching scale.
///
/// `inner_offset` is where the strip's first row lands in the full
/// transposed buffer — i.e. the row offset within original-frame Y
/// where the strip's body+halo starts. The kernel reads ref pixels at
/// `outer * full_inner_stride + inner_offset + inner`.
///
/// Pointwise per strip pixel; processes `strip_n` = `strip_h × outer`
/// pixels total.
#[cube(launch_unchecked)]
pub fn error_maps_strip_from_full_ref_kernel(
    source_full: &Array<f32>,
    distorted_strip: &Array<f32>,
    mu1_full: &Array<f32>,
    mu2_strip: &Array<f32>,
    sigma11_full: &Array<f32>,
    sigma22_strip: &Array<f32>,
    sigma12_strip: &Array<f32>,
    out_ssim: &mut Array<f32>,
    out_artifact: &mut Array<f32>,
    out_detail: &mut Array<f32>,
    strip_inner_stride: u32,
    full_inner_stride: u32,
    inner_offset: u32,
) {
    let idx = ABSOLUTE_POS;
    let n = out_ssim.len();
    if idx >= n {
        terminate!();
    }
    // Decompose strip-buffer flat idx → (outer, inner) in transposed
    // orientation. Inner index = original frame Y within the strip;
    // outer index = original frame X (column).
    let strip_in = strip_inner_stride as usize;
    let outer = idx / strip_in;
    let inner = idx - outer * strip_in;
    // Corresponding flat idx in the full ref buffers: same outer, but
    // inner = inner_offset + inner-within-strip.
    let full_in = full_inner_stride as usize;
    let ref_idx = outer * full_in + (inner_offset as usize) + inner;

    let m1 = mu1_full[ref_idx];
    let m2 = mu2_strip[idx];
    let s11 = sigma11_full[ref_idx];
    let s22 = sigma22_strip[idx];
    let s12 = sigma12_strip[idx];
    let src = source_full[ref_idx];
    let dis = distorted_strip[idx];

    let mu11 = m1 * m1;
    let mu22 = m2 * m2;
    let mu12 = m1 * m2;
    let mu_diff = m1 - m2;
    let num_m = 1.0 - mu_diff * mu_diff;
    let num_s = 2.0 * (s12 - mu12) + C2;
    let denom_s = (s11 - mu11) + (s22 - mu22) + C2;
    let mut d_ssim = 1.0 - (num_m * num_s) / denom_s;
    if d_ssim < 0.0 {
        d_ssim = 0.0;
    }
    out_ssim[idx] = d_ssim;

    let denom = 1.0 / (1.0 + f32::abs(src - m1));
    let numer = 1.0 + f32::abs(dis - m2);
    let d1 = numer * denom - 1.0;

    let art = if d1 > 0.0 { d1 } else { f32::new(0.0) };
    let dl = if d1 < 0.0 { -d1 } else { f32::new(0.0) };
    out_artifact[idx] = art;
    out_detail[idx] = dl;
}

/// Broadcast-batched error_maps for `Ssim2Batch`.
///
/// Reference-side inputs (`source`, `mu1`, `sigma11`) are single
/// planes shared across the batch and indexed at `idx % plane_stride`.
/// Distorted-side inputs (`distorted`, `mu2`, `sigma22`, `sigma12`)
/// and the three outputs are per-image batched buffers indexed at
/// `idx`. Each plane is `plane_stride` floats; the kernel processes
/// `plane_stride · batch_size` total pixels.
#[cube(launch_unchecked)]
pub fn error_maps_broadcast_batched_kernel(
    source: &Array<f32>,
    distorted: &Array<f32>,
    mu1: &Array<f32>,
    mu2: &Array<f32>,
    sigma11: &Array<f32>,
    sigma22: &Array<f32>,
    sigma12: &Array<f32>,
    out_ssim: &mut Array<f32>,
    out_artifact: &mut Array<f32>,
    out_detail: &mut Array<f32>,
    plane_stride: u32,
) {
    let idx = ABSOLUTE_POS;
    let n = out_ssim.len();
    if idx >= n {
        terminate!();
    }
    let pl = plane_stride as usize;
    let local = idx - (idx / pl) * pl;
    let m1 = mu1[local];
    let m2 = mu2[idx];
    let s11 = sigma11[local];
    let s22 = sigma22[idx];
    let s12 = sigma12[idx];
    let src = source[local];
    let dis = distorted[idx];

    let mu11 = m1 * m1;
    let mu22 = m2 * m2;
    let mu12 = m1 * m2;
    let mu_diff = m1 - m2;
    let num_m = 1.0 - mu_diff * mu_diff;
    let num_s = 2.0 * (s12 - mu12) + C2;
    let denom_s = (s11 - mu11) + (s22 - mu22) + C2;
    let mut d_ssim = 1.0 - (num_m * num_s) / denom_s;
    if d_ssim < 0.0 {
        d_ssim = 0.0;
    }
    out_ssim[idx] = d_ssim;

    let denom = 1.0 / (1.0 + f32::abs(src - m1));
    let numer = 1.0 + f32::abs(dis - m2);
    let d1 = numer * denom - 1.0;

    let art = if d1 > 0.0 { d1 } else { f32::new(0.0) };
    let dl = if d1 < 0.0 { -d1 } else { f32::new(0.0) };
    out_artifact[idx] = art;
    out_detail[idx] = dl;
}
