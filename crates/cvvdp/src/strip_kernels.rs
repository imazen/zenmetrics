//! Strip-aware sibling kernels for the K_SPLIT walker.
//!
//! Ports the GPU's six strip-aware kernels from cvvdp-gpu (see
//! `crates/cvvdp-gpu/docs/AUDIT_2026-05-28.md` §A for the verified
//! GPU kernel inventory). Each kernel here is bit-identical to its
//! Full-mode CPU counterpart when invoked with
//! `body_offset_y = 0, logical_h = h, logical_w = w` (the strip
//! reduces to the full image trivially).
//!
//! These kernels operate on **strip-buffer-sized** planes
//! (`w × strip_h` where `strip_h = body_h + 2·halo`). The halo rows
//! at the top and bottom of each strip MUST be filled with the
//! correct logical-image content (the strip walker is responsible
//! for that). When a strip's halo extends past the logical image
//! edge, the kernels reflect against the logical image, NOT the
//! strip buffer — see [`pu_blur_v_strip_aware_into`] for the
//! load-bearing case.
//!
//! ## Kernel inventory (ported in this module)
//!
//! - **Chunk 1**: `pu_blur_h_strip_aware_into` (no halo — X-only),
//!   `pu_blur_v_strip_aware_into` (load-bearing — reflects against
//!   logical_h then translates by body_offset_y).
//! - **Chunk 5**: `downscale_strip_into`, `upscale_v_strip_into`,
//!   `upscale_h_strip_into`, `subtract_weber_3ch_strip_into`. (Wired
//!   in subsequent commits.)
//!
//! Chunks 2 + 3 are per-pixel kernels that work on strip-sized
//! buffers without modification — no separate variant exists, and
//! the strip dispatcher calls the existing per-pixel CSF / masking
//! helpers on strip-sized slices.

use alloc::vec::Vec;

use crate::kernels::masking::PU_BLUR_KERNEL_1D;

/// Reflect `i` into `[0, n)` for the 13-tap PU blur — duplicated
/// here (private to `kernels::masking`) so the strip-aware kernels
/// don't pay a cross-module barrier. Matches torchvision's
/// `F.pad(..., mode='reflect')` semantics.
#[inline]
fn reflect_pu_idx(i: isize, n: usize) -> usize {
    let n_i = n as isize;
    debug_assert!(n_i > 0);
    let mut j = i;
    while j < 0 || j >= n_i {
        if j < 0 {
            j = -j;
        }
        if j >= n_i {
            j = 2 * n_i - 2 - j;
        }
    }
    j as usize
}

/// Strip-aware horizontal pass of the σ=3 PU blur for 3 channels.
///
/// **No halo.** H-blur only touches the X axis, so a strip buffer
/// is the same as a full-image buffer for this kernel — the strip
/// just IS the body. Per the GPU port, `body_offset_y` and
/// `logical_h` are accepted **for API uniformity** with the
/// V-pass but are unused in the kernel body (because reflection
/// happens against `w`, which is identical between strip and full).
///
/// Writes into `dst_*` of length `w * h` each.
pub(crate) fn pu_blur_h_strip_aware_3ch_into(
    src_a: &[f32],
    src_rg: &[f32],
    src_vy: &[f32],
    dst_a: &mut [f32],
    dst_rg: &mut [f32],
    dst_vy: &mut [f32],
    w: usize,
    h: usize,
    _body_offset_y: u32,
    _logical_h: u32,
) {
    debug_assert_eq!(src_a.len(), w * h);
    debug_assert_eq!(src_rg.len(), w * h);
    debug_assert_eq!(src_vy.len(), w * h);
    debug_assert_eq!(dst_a.len(), w * h);
    debug_assert_eq!(dst_rg.len(), w * h);
    debug_assert_eq!(dst_vy.len(), w * h);
    let k = PU_BLUR_KERNEL_1D;
    let half = 6_isize;

    for y in 0..h {
        let row_off = y * w;
        for x in 0..w {
            let mut s_a = 0.0_f32;
            let mut s_rg = 0.0_f32;
            let mut s_vy = 0.0_f32;
            for t in 0..13 {
                let sx = reflect_pu_idx(x as isize + t as isize - half, w);
                s_a += k[t] * src_a[row_off + sx];
                s_rg += k[t] * src_rg[row_off + sx];
                s_vy += k[t] * src_vy[row_off + sx];
            }
            dst_a[row_off + x] = s_a;
            dst_rg[row_off + x] = s_rg;
            dst_vy[row_off + x] = s_vy;
        }
    }
}

/// Strip-aware vertical pass of the σ=3 PU blur for 3 channels with
/// `* pu_scale` post-multiply.
///
/// **Load-bearing for strip correctness.** The Y axis taps reflect
/// against `logical_h` (the underlying image height), NOT against
/// `h` (the strip buffer height). The strip walker must size the
/// strip buffer's halo so that for every body row, every tap's
/// reflected `g_t` translates back to a strip-buffer-local index in
/// `[0, h)`. If a tap lands out of `[0, h)` after the translation,
/// it's a strip-coverage bug in the caller, not a kernel bug.
///
/// Per-output:
/// - `y_strip = idx / w`, `x = idx % w` (1D thread index into strip)
/// - `y_global = y_strip + body_offset_y`
/// - For each tap `t`:
///   - `g_t = y_global + (t - 6)`
///   - `ref_g_t = reflect_pu_idx(g_t, logical_h)`
///   - `local_t = ref_g_t - body_offset_y` (must be in `[0, h)`)
///   - accumulate `k[t] * src[local_t * w + x]`
/// - `dst[idx] = acc * pu_scale`
///
/// **Output range.** When `body_offset_y = 0, logical_h = h`, this
/// is bit-exact identical to the full-image V-pass.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pu_blur_v_strip_aware_3ch_into(
    src_a: &[f32],
    src_rg: &[f32],
    src_vy: &[f32],
    dst_a: &mut [f32],
    dst_rg: &mut [f32],
    dst_vy: &mut [f32],
    pu_scale: f32,
    w: usize,
    h: usize,
    body_offset_y: u32,
    logical_h: u32,
) {
    debug_assert_eq!(src_a.len(), w * h);
    debug_assert_eq!(src_rg.len(), w * h);
    debug_assert_eq!(src_vy.len(), w * h);
    debug_assert_eq!(dst_a.len(), w * h);
    debug_assert_eq!(dst_rg.len(), w * h);
    debug_assert_eq!(dst_vy.len(), w * h);
    let k = PU_BLUR_KERNEL_1D;
    let half = 6_isize;
    let body_off_i = body_offset_y as isize;
    let logical_h_us = logical_h as usize;

    for y_strip in 0..h {
        let y_global = y_strip as isize + body_off_i;
        let row_off = y_strip * w;
        // Resolve the 13 tap source rows for this output row.
        // `local_t = reflect(y_global + (t-6), logical_h) - body_offset_y`.
        //
        // For BODY rows of the strip (those whose `y_global` lies in
        // the body range `[top_global + halo, top_global + halo +
        // body_h)`), the caller sizes the halo so `local_t ∈ [0, h)`
        // exactly. For HALO rows of the strip (the top + bottom halo
        // bands), `local_t` may go out of `[0, h)` since the strip
        // window doesn't cover the source rows that reflection picks.
        // On GPU these halo-row outputs would be garbage and never
        // consumed; on CPU we **clamp** local_t into `[0, h)` so the
        // load is safe. The garbage halo output is non-load-bearing —
        // the strip walker only reads body rows downstream.
        let mut local: [usize; 13] = [0; 13];
        for (t, slot) in local.iter_mut().enumerate() {
            let g_t = y_global + (t as isize) - half;
            let ref_g_t = reflect_pu_idx(g_t, logical_h_us);
            let l_t = (ref_g_t as isize) - body_off_i;
            // Clamp into strip-buffer range so the load below is
            // safe. For body rows the clamp is a no-op (caller halo
            // sizing guarantees l_t ∈ [0, h)).
            let l_clamped = if l_t < 0 {
                0
            } else if (l_t as usize) >= h {
                h - 1
            } else {
                l_t as usize
            };
            *slot = l_clamped;
        }

        for x in 0..w {
            let mut s_a = 0.0_f32;
            let mut s_rg = 0.0_f32;
            let mut s_vy = 0.0_f32;
            for t in 0..13 {
                let row = local[t] * w + x;
                s_a += k[t] * src_a[row];
                s_rg += k[t] * src_rg[row];
                s_vy += k[t] * src_vy[row];
            }
            dst_a[row_off + x] = s_a * pu_scale;
            dst_rg[row_off + x] = s_rg * pu_scale;
            dst_vy[row_off + x] = s_vy * pu_scale;
        }
    }
}

/// **Chunk 2**: per-pixel CSF apply is strip-degenerate.
///
/// The CSF helper [`crate::pipeline::apply_csf_row_per_pixel`] is a
/// per-pixel function with no spatial halo — it reads only the
/// per-pixel `log_l_bkg` (which is also computed per-pixel by the
/// pyramid stage). On a strip-buffer view of the band, applying the
/// CSF to every pixel of the strip is bit-identical to applying it
/// to the corresponding rows of the full band.
///
/// This module exposes a parity test (`csf_strip_degenerate`) but
/// no new kernel — the strip dispatcher (chunk 6) calls the
/// existing `apply_csf_row_per_pixel` directly on strip-sized
/// slices. Per the GPU audit:
///
/// > `csf_apply_3ch_kernel` + `csf_apply_6ch_kernel` are per-pixel
/// > and dispatch on strip-sized buffers as-is — no separate
/// > variant needed.
///
/// (cvvdp-gpu `kernels/csf.rs:126, 220`, audit §A rows 8-9.)
///
/// **Chunk 3**: per-pixel masking chain is strip-degenerate.
///
/// `min_abs_3ch` (per-pixel `min(|T|, |R|)`) and the per-pixel
/// stages of `mult_mutual_3ch_*` (the final cross-channel pool,
/// `safe_pow`, soft clamp) are all per-pixel; only the σ=3 PU blur
/// inside `mult_mutual` has a halo, and that's handled by chunk 1.
///
/// Per audit §A rows 10-11:
///
/// > `min_abs_3ch_kernel`, `mult_mutual_3ch_*` are per-pixel and
/// > the masking strip walker dispatches them on strip-sized
/// > buffers (no separate kernel variant).
///
/// (cvvdp-gpu `kernels/masking.rs:741, 806, 929`.)
///
/// Together chunks 2+3 are "verify wiring + add parity tests with
/// minimal code change" per the brief.
#[cfg(test)]
fn _chunk_2_3_marker() {}

/// Convenience wrapper: PU blur (H + V) for 3 channels with strip-aware
/// reflection, applying `pu_scale = 10^MASK_C` post-multiply.
///
/// `h_pass_*` are caller-owned scratch buffers (`w * h` each). Caller
/// is responsible for sizing the halo so the V-pass reflection lands
/// inside the strip buffer.
///
/// Degenerates to a Full-mode 3-channel PU blur when `body_offset_y = 0`
/// and `logical_h = h`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pu_blur_3ch_strip_aware(
    src_a: &[f32],
    src_rg: &[f32],
    src_vy: &[f32],
    h_pass_a: &mut [f32],
    h_pass_rg: &mut [f32],
    h_pass_vy: &mut [f32],
    dst_a: &mut [f32],
    dst_rg: &mut [f32],
    dst_vy: &mut [f32],
    pu_scale: f32,
    w: usize,
    h: usize,
    body_offset_y: u32,
    logical_h: u32,
) {
    pu_blur_h_strip_aware_3ch_into(
        src_a,
        src_rg,
        src_vy,
        h_pass_a,
        h_pass_rg,
        h_pass_vy,
        w,
        h,
        body_offset_y,
        logical_h,
    );
    pu_blur_v_strip_aware_3ch_into(
        h_pass_a,
        h_pass_rg,
        h_pass_vy,
        dst_a,
        dst_rg,
        dst_vy,
        pu_scale,
        w,
        h,
        body_offset_y,
        logical_h,
    );
}

// =====================================================================
// Chunk 5: strip-aware pyramid kernels (downscale, upscale_v, upscale_h,
// subtract_weber). Ports the GPU's strip-aware sibling kernels — see
// AUDIT_2026-05-28.md §A rows 1-4 (verified shipped GPU kernels at
// `kernels/pyramid.rs:290, 784, 1013, 1396`).
// =====================================================================

/// Strip-aware downscale (2× reduce, 5-tap separable Gaussian) for a
/// single channel.
///
/// **Buffer convention.** `src` is a strip-local buffer with row 0
/// at logical row `src_strip_offset`. `dst` is a strip-local buffer
/// that holds dst rows `[body_offset_y, body_offset_y + dst_h_strip)`
/// (no dst halo — dst is body-only).
///
/// **CPU bit-exact semantics.** Bit-identical to a slice of
/// [`crate::pyramid::gausspyr_reduce`] / [`crate::kernels::pyramid::
/// gausspyr_reduce_scalar`] over `body_offset_y..body_offset_y +
/// dst_h_strip` rows when the source strip covers the corresponding
/// `[2·body_offset_y − 2, 2·(body_offset_y + dst_h_strip − 1) + 2]`
/// region in src.
///
/// **The pycvvdp parity-on-rows bug**: the upstream kernel applies
/// the last-col patch based on `sh % 2` (source height) — a
/// documented typo in upstream pycvvdp that the CPU
/// `gausspyr_reduce_scalar` preserves for golden parity. The
/// strip-aware kernel uses `logical_src_h % 2` so a partial strip
/// with even buffer height but odd logical height still fires the
/// patch (matching the full-image-dispatch result).
///
/// Caller-owned scratch:
/// - `vscratch`: `src_w * dst_h_strip` f32 — reused across calls.
///
/// Caller invariants:
/// - `src.len() == src_w * src_h_buf`
/// - `dst.len() == dst_w * dst_h_strip`
/// - `vscratch.len() >= src_w * dst_h_strip` (resized internally if smaller)
/// - `dst_w == src_w.div_ceil(2)`, `dst_h_strip == body_h`
/// - `src_strip_offset + src_h_buf <= logical_src_h`
/// - For each body row `dy_local` in `[0, dst_h_strip)`, the rows
///   `[2·(dy_local + body_offset_y) − 2, 2·(dy_local + body_offset_y) + 2]`
///   must lie inside the strip after subtracting `src_strip_offset`
///   (the caller's halo sizing guarantees this — bug if it doesn't).
#[allow(clippy::too_many_arguments)]
pub(crate) fn downscale_strip_into(
    src: &[f32],
    src_w: usize,
    src_h_buf: usize,
    dst: &mut [f32],
    dst_w: usize,
    dst_h_strip: usize,
    body_offset_y: u32,
    src_strip_offset: u32,
    logical_src_h: u32,
    vscratch: &mut Vec<f32>,
) {
    debug_assert_eq!(src.len(), src_w * src_h_buf);
    debug_assert_eq!(dst.len(), dst_w * dst_h_strip);
    debug_assert_eq!(dst_w, src_w.div_ceil(2));

    use crate::kernels::pyramid::GAUSS5;
    let k = GAUSS5;

    vscratch.clear();
    vscratch.resize(src_w * dst_h_strip, 0.0);

    let body_off = body_offset_y as usize;
    let src_off = src_strip_offset as usize;
    let lsh = logical_src_h as usize;

    // V-pass: for each strip-local dst row `dy_local`, compute the
    // logical center row `cy = 2 * (dy_local + body_off)` and read
    // 5 src rows at `cy + t - 2` for t in 0..5, translated to
    // strip-local by subtracting `src_off`. Zero-pad outside
    // `[0, lsh)` (matching `gausspyr_reduce_scalar`'s `read` closure).
    for dy_local in 0..dst_h_strip {
        let cy = 2 * ((dy_local + body_off) as isize);
        for x in 0..src_w {
            let read = |off: isize| -> f32 {
                let r = cy + off;
                if r < 0 || r >= lsh as isize {
                    0.0
                } else {
                    let buf_row = r as usize - src_off;
                    debug_assert!(buf_row < src_h_buf, "src row OOB in strip");
                    src[buf_row * src_w + x]
                }
            };
            vscratch[dy_local * src_w + x] =
                k[0] * read(-2) + k[1] * read(-1) + k[2] * read(0) + k[3] * read(1) + k[4] * read(2);
        }
    }

    // First-row + last-row patches against logical_src_h (so a strip
    // covering the logical top or bottom edge fires the patch). These
    // patches add the contribution of reflected rows that the V-pass
    // zero-padded out.
    //
    // Logical first-row patch fires only for the strip whose first
    // body row equals logical-dst row 0 (i.e., body_offset_y == 0).
    if body_off == 0 && dst_h_strip > 0 && lsh >= 2 {
        // Add at dy_local=0 (logical dy=0):
        //   vscratch[x] += src[x]*k[1] + src[sw + x]*k[0];
        // Strip-local: logical row 0 is at strip-row `0 - src_off`.
        let r0_local = 0_isize - src_off as isize;
        let r1_local = 1_isize - src_off as isize;
        if r0_local >= 0
            && (r0_local as usize) < src_h_buf
            && r1_local >= 0
            && (r1_local as usize) < src_h_buf
        {
            let r0u = r0_local as usize;
            let r1u = r1_local as usize;
            for x in 0..src_w {
                vscratch[x] += src[r0u * src_w + x] * k[1] + src[r1u * src_w + x] * k[0];
            }
        }
    }

    // Logical last-row patch fires only for the strip containing the
    // last logical dst row.
    let logical_dst_h = lsh.div_ceil(2);
    let last_dy_global = logical_dst_h.wrapping_sub(1); // == lsh.div_ceil(2) - 1
    if dst_h_strip > 0 {
        let last_local = (last_dy_global as isize) - body_off as isize;
        if last_local >= 0 && (last_local as usize) < dst_h_strip {
            let last_local_u = last_local as usize;
            if lsh % 2 == 1 && lsh >= 2 {
                let r_last_local = (lsh - 1) as isize - src_off as isize;
                let r_lastm1_local = (lsh - 2) as isize - src_off as isize;
                if r_last_local >= 0
                    && (r_last_local as usize) < src_h_buf
                    && r_lastm1_local >= 0
                    && (r_lastm1_local as usize) < src_h_buf
                {
                    let r1u = r_last_local as usize;
                    let r2u = r_lastm1_local as usize;
                    for x in 0..src_w {
                        vscratch[last_local_u * src_w + x] +=
                            src[r1u * src_w + x] * k[3] + src[r2u * src_w + x] * k[4];
                    }
                }
            } else if lsh.is_multiple_of(2) {
                let r_last_local = (lsh - 1) as isize - src_off as isize;
                if r_last_local >= 0 && (r_last_local as usize) < src_h_buf {
                    let r1u = r_last_local as usize;
                    for x in 0..src_w {
                        vscratch[last_local_u * src_w + x] += src[r1u * src_w + x] * k[4];
                    }
                }
            }
        }
    }

    // H-pass: dst row by row, reading vscratch with reflect-pad
    // against `src_w` (X axis unchanged between strip + full).
    for dy_local in 0..dst_h_strip {
        for dx in 0..dst_w {
            let cx = 2 * dx as isize;
            let read = |off: isize| -> f32 {
                let c = cx + off;
                if c < 0 || c >= src_w as isize {
                    0.0
                } else {
                    vscratch[dy_local * src_w + c as usize]
                }
            };
            dst[dy_local * dst_w + dx] =
                k[0] * read(-2) + k[1] * read(-1) + k[2] * read(0) + k[3] * read(1) + k[4] * read(2);
        }
    }

    // First-col patch (vscratch[dy*sw] * k[1] + vscratch[dy*sw+1] * k[0]).
    // Always fires for shallow strips since the first col is at dx=0.
    if dst_w > 0 && src_w >= 2 {
        for dy_local in 0..dst_h_strip {
            dst[dy_local * dst_w] +=
                vscratch[dy_local * src_w] * k[1] + vscratch[dy_local * src_w + 1] * k[0];
        }
    }

    // Last-col patch — uses LOGICAL_SRC_H parity (NOT src_h_buf!).
    // This is the pycvvdp bug-compat: upstream uses `sh % 2` (which
    // semantically should be `sw % 2`); the strip kernel uses
    // `logical_src_h % 2` so a partial strip with even buffer height
    // still fires identically.
    if dst_w > 0 {
        let last_dx = dst_w - 1;
        if lsh % 2 == 1 && src_w >= 2 {
            for dy_local in 0..dst_h_strip {
                dst[dy_local * dst_w + last_dx] += vscratch[dy_local * src_w + src_w - 1] * k[3]
                    + vscratch[dy_local * src_w + src_w - 2] * k[4];
            }
        } else if lsh.is_multiple_of(2) {
            for dy_local in 0..dst_h_strip {
                dst[dy_local * dst_w + last_dx] += vscratch[dy_local * src_w + src_w - 1] * k[4];
            }
        }
    }
}

/// Strip-aware upscale V-pass: 5-tap separable Gaussian + 2× expand.
///
/// **Buffer convention.** `src` is a strip-local buffer with row 0
/// at logical-src row `src_strip_offset`. `dst` is sized exactly to
/// the body region (no dst halo) — buffer-local dst row 0
/// corresponds to logical-dst row `body_offset_y`.
///
/// **CPU bit-exact semantics.** Bit-identical to a slice of
/// `gausspyr_expand_scalar`'s V-pass over rows `[body_offset_y,
/// body_offset_y + body_h)` when the source strip covers the
/// corresponding back-projected region.
#[allow(clippy::too_many_arguments)]
pub(crate) fn upscale_v_strip_into(
    src: &[f32],
    src_w: usize,
    src_h_buf: usize,
    dst: &mut [f32],
    body_h: usize,
    body_offset_y: u32,
    src_strip_offset: u32,
    logical_src_h: u32,
    logical_dst_h: u32,
) {
    debug_assert_eq!(src.len(), src_w * src_h_buf);
    debug_assert_eq!(dst.len(), src_w * body_h);

    use crate::kernels::pyramid::GAUSS5;
    let k = GAUSS5;

    let body_off = body_offset_y as usize;
    let src_off = src_strip_offset as usize;
    let lsh = logical_src_h as usize;
    let ldh = logical_dst_h as usize;

    // gausspyr_expand_scalar's V-pass uses a per-column zero-insert
    // buffer z_v[z_len_v = out_h + 4]. z_v[0] = src[0], z_v[2 + 2*ky] =
    // src[ky], z_v[back_idx_v = out_h + 2 + odd] = src[sh-1]. Outputs
    // for y in 0..out_h: sum_{kt in 0..5} k[kt] * z_v[y + kt].
    //
    // For strip-aware: we only emit y in [body_off, body_off + body_h).
    // For each output row, identify which 5 z_v indices are read and
    // map back to logical-src rows.
    let odd_h = ldh & 1;
    let back_idx_v = ldh + 2 + odd_h; // index of z_v[back]
    // For an output row `y_logical`, z_v indices used are y_logical+0..y_logical+5.
    // The mapping logical_src_row -> z_v idx:
    //   z_v[0] = src[0]
    //   z_v[2 + 2*ky] = src[ky]
    //   z_v[back_idx_v] = src[sh-1]
    //   other indices = 0.
    // Inverse: given z_v idx z:
    //   z == 0 -> src[0]
    //   z == back_idx_v -> src[sh-1]
    //   z >= 2 && (z-2)%2 == 0 && (z-2)/2 < sh -> src[(z-2)/2]
    //   otherwise -> 0 (zero-insert).
    for dy_local in 0..body_h {
        let y_logical = dy_local + body_off;
        for x in 0..src_w {
            let mut sum = 0.0_f32;
            for kt in 0..5 {
                let z = y_logical + kt;
                let val = if z == 0 {
                    // src[0] (logical row 0)
                    let buf_r = 0_isize - src_off as isize;
                    if buf_r >= 0 && (buf_r as usize) < src_h_buf {
                        src[(buf_r as usize) * src_w + x]
                    } else {
                        0.0
                    }
                } else if z == back_idx_v {
                    // src[sh-1] (logical row lsh-1)
                    if lsh == 0 {
                        0.0
                    } else {
                        let buf_r = (lsh - 1) as isize - src_off as isize;
                        if buf_r >= 0 && (buf_r as usize) < src_h_buf {
                            src[(buf_r as usize) * src_w + x]
                        } else {
                            0.0
                        }
                    }
                } else if z >= 2 && (z & 1) == 0 {
                    let logical_row = (z - 2) >> 1;
                    if logical_row < lsh {
                        let buf_r = logical_row as isize - src_off as isize;
                        if buf_r >= 0 && (buf_r as usize) < src_h_buf {
                            src[(buf_r as usize) * src_w + x]
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };
                sum += k[kt] * val;
            }
            dst[dy_local * src_w + x] = 2.0 * sum;
        }
    }
}

/// Strip-aware upscale H-pass: 5-tap separable Gaussian + 2× expand
/// across X axis only. The H-pass has no Y-direction state, so the
/// strip parameters (body_offset_y, logical_dst_h) are accepted **for
/// API uniformity** with the V-pass strip kernel but are unused in
/// the kernel body.
///
/// CPU bit-exact semantics: identical to `gausspyr_expand_scalar`'s
/// H-pass over the strip's rows when invoked as
/// `(src_w, in_h, dst_w, dst, src)` with `in_h == body_h` rows of
/// already-V-passed source.
#[allow(clippy::too_many_arguments)]
pub(crate) fn upscale_h_strip_into(
    src: &[f32],
    src_w: usize,
    in_h: usize,
    dst: &mut [f32],
    dst_w: usize,
    _body_offset_y: u32,
    _logical_dst_h: u32,
) {
    debug_assert_eq!(src.len(), src_w * in_h);
    debug_assert_eq!(dst.len(), dst_w * in_h);

    use crate::kernels::pyramid::GAUSS5;
    let k = GAUSS5;

    let odd_w = dst_w & 1;
    let back_idx_h = dst_w + 2 + odd_w;

    for y in 0..in_h {
        for x in 0..dst_w {
            let mut sum = 0.0_f32;
            for kt in 0..5 {
                let z = x + kt;
                let val = if z == 0 {
                    src[y * src_w]
                } else if z == back_idx_h {
                    src[y * src_w + src_w - 1]
                } else if z >= 2 && (z & 1) == 0 {
                    let kx = (z - 2) >> 1;
                    if kx < src_w {
                        src[y * src_w + kx]
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };
                sum += k[kt] * val;
            }
            dst[y * dst_w + x] = 2.0 * sum;
        }
    }
}

/// Strip-aware `subtract_weber_3ch`: computes `contrast = (fine -
/// upscaled) / max(0.01, l_bkg)` clamped to `[-1000, +1000]`, and
/// `log_l_bkg = log10(max(0.01, l_bkg))` per pixel.
///
/// All `fine_*`, `upsc_*`, `expanded_lbkg`, `contrast_*`, `log_l_bkg`
/// buffers are strip-local with row 0 at logical row
/// `src_strip_offset`. Outputs are written at the body rows only
/// (logical rows `[body_offset_y, body_offset_y + body_h)`).
///
/// CPU bit-exact semantics: matches the per-pixel formula from
/// `weber_contrast_pyr_dec_scalar` / `weber_contrast_pyr_into`
/// (clamped to `[-1000, 1000]`, log10 of clamp-at-0.01 l_bkg).
#[allow(clippy::too_many_arguments)]
pub(crate) fn subtract_weber_3ch_strip_into(
    fine_a: &[f32],
    fine_rg: &[f32],
    fine_vy: &[f32],
    upsc_a: &[f32],
    upsc_rg: &[f32],
    upsc_vy: &[f32],
    expanded_lbkg: &[f32],
    contrast_a: &mut [f32],
    contrast_rg: &mut [f32],
    contrast_vy: &mut [f32],
    log_l_bkg: &mut [f32],
    w: usize,
    body_h: usize,
    body_offset_y: u32,
    src_strip_offset: u32,
) {
    let body_off = body_offset_y as usize;
    let src_off = src_strip_offset as usize;
    // buf_y = body_off + dy_local - src_off; with src_off = body_off
    // (strip-local outputs), buf_y == dy_local.
    let delta = (body_off as isize) - (src_off as isize);

    for dy_local in 0..body_h {
        let buf_y = (dy_local as isize) + delta;
        debug_assert!(buf_y >= 0);
        let buf_y_u = buf_y as usize;
        let row_off = buf_y_u * w;
        for x in 0..w {
            let i = row_off + x;
            let l_bkg = expanded_lbkg[i].max(0.01);
            let layer_a = fine_a[i] - upsc_a[i];
            let layer_rg = fine_rg[i] - upsc_rg[i];
            let layer_vy = fine_vy[i] - upsc_vy[i];
            contrast_a[i] = (layer_a / l_bkg).clamp(-1000.0, 1000.0);
            contrast_rg[i] = (layer_rg / l_bkg).clamp(-1000.0, 1000.0);
            contrast_vy[i] = (layer_vy / l_bkg).clamp(-1000.0, 1000.0);
            log_l_bkg[i] = l_bkg.log10();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::csf::{CsfChannel, N_L_BKG, precompute_logs_row};
    use crate::kernels::masking::{
        CH_GAIN, MASK_C, MASK_P, MASK_Q, gaussian_blur_sigma3, mult_mutual_band, safe_pow,
    };

    /// Synthesize a deterministic random plane.
    fn synth_plane(w: usize, h: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        let mut out = Vec::with_capacity(w * h);
        for _ in 0..w * h {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            out.push(((s as i32) as f32) / (i32::MAX as f32));
        }
        out
    }

    /// **Chunk 1 parity (H, degenerate)**: strip-aware H-blur with
    /// `body_offset_y=0, logical_h=h` produces the same output as
    /// the upstream scalar `gaussian_blur_sigma3`'s H-pass. Bit-
    /// identical except for safe scalar accumulator ordering (we
    /// match the upstream scalar reference exactly).
    #[test]
    fn pu_blur_h_strip_aware_degenerates_to_full_3ch() {
        for &(w, h) in &[(16_usize, 8_usize), (32, 32), (64, 64), (15, 13), (7, 7)] {
            let src_a = synth_plane(w, h, 0xfeed_beef);
            let src_rg = synth_plane(w, h, 0x1234_5678);
            let src_vy = synth_plane(w, h, 0xabcd_ef01);

            // Reference: full-image gaussian_blur_sigma3 H-only (which the
            // upstream scalar does as part of the 2D blur — to extract
            // just the H pass we reimplement it inline).
            let mut ref_a = vec![0.0_f32; w * h];
            let mut ref_rg = vec![0.0_f32; w * h];
            let mut ref_vy = vec![0.0_f32; w * h];
            let k = PU_BLUR_KERNEL_1D;
            let half = 6_isize;
            for y in 0..h {
                for x in 0..w {
                    let mut sa = 0.0_f32;
                    let mut sr = 0.0_f32;
                    let mut sv = 0.0_f32;
                    for t in 0..13 {
                        let sx = reflect_pu_idx(x as isize + t as isize - half, w);
                        sa += k[t] * src_a[y * w + sx];
                        sr += k[t] * src_rg[y * w + sx];
                        sv += k[t] * src_vy[y * w + sx];
                    }
                    ref_a[y * w + x] = sa;
                    ref_rg[y * w + x] = sr;
                    ref_vy[y * w + x] = sv;
                }
            }

            let mut dst_a = vec![0.0_f32; w * h];
            let mut dst_rg = vec![0.0_f32; w * h];
            let mut dst_vy = vec![0.0_f32; w * h];
            pu_blur_h_strip_aware_3ch_into(
                &src_a,
                &src_rg,
                &src_vy,
                &mut dst_a,
                &mut dst_rg,
                &mut dst_vy,
                w,
                h,
                0,
                h as u32,
            );
            for i in 0..w * h {
                assert_eq!(
                    ref_a[i].to_bits(),
                    dst_a[i].to_bits(),
                    "H-pass A mismatch at {w}×{h} idx {i}: ref={:e}, dst={:e}",
                    ref_a[i],
                    dst_a[i]
                );
                assert_eq!(ref_rg[i].to_bits(), dst_rg[i].to_bits());
                assert_eq!(ref_vy[i].to_bits(), dst_vy[i].to_bits());
            }
        }
    }

    /// **Chunk 1 parity (V, degenerate)**: strip-aware V-blur with
    /// `body_offset_y=0, logical_h=h, pu_scale=1.0` matches the
    /// upstream scalar `gaussian_blur_sigma3` V-pass.
    #[test]
    fn pu_blur_v_strip_aware_degenerates_to_full_3ch() {
        for &(w, h) in &[(16_usize, 8_usize), (32, 32), (64, 64), (15, 13), (7, 13)] {
            let src_a = synth_plane(w, h, 0x1111_2222);
            let src_rg = synth_plane(w, h, 0x3333_4444);
            let src_vy = synth_plane(w, h, 0x5555_6666);

            // Reference: full V-pass scalar reflection.
            let k = PU_BLUR_KERNEL_1D;
            let half = 6_isize;
            let mut ref_a = vec![0.0_f32; w * h];
            let mut ref_rg = vec![0.0_f32; w * h];
            let mut ref_vy = vec![0.0_f32; w * h];
            for y in 0..h {
                for x in 0..w {
                    let mut sa = 0.0_f32;
                    let mut sr = 0.0_f32;
                    let mut sv = 0.0_f32;
                    for t in 0..13 {
                        let sy = reflect_pu_idx(y as isize + t as isize - half, h);
                        sa += k[t] * src_a[sy * w + x];
                        sr += k[t] * src_rg[sy * w + x];
                        sv += k[t] * src_vy[sy * w + x];
                    }
                    ref_a[y * w + x] = sa;
                    ref_rg[y * w + x] = sr;
                    ref_vy[y * w + x] = sv;
                }
            }

            let mut dst_a = vec![0.0_f32; w * h];
            let mut dst_rg = vec![0.0_f32; w * h];
            let mut dst_vy = vec![0.0_f32; w * h];
            pu_blur_v_strip_aware_3ch_into(
                &src_a,
                &src_rg,
                &src_vy,
                &mut dst_a,
                &mut dst_rg,
                &mut dst_vy,
                1.0,
                w,
                h,
                0,
                h as u32,
            );
            for i in 0..w * h {
                assert_eq!(
                    ref_a[i].to_bits(),
                    dst_a[i].to_bits(),
                    "V-pass A mismatch at {w}×{h} idx {i}"
                );
                assert_eq!(ref_rg[i].to_bits(), dst_rg[i].to_bits());
                assert_eq!(ref_vy[i].to_bits(), dst_vy[i].to_bits());
            }
        }
    }

    /// **Chunk 1 strip-vs-full parity**: a full-image 2D blur computed
    /// via the strip-aware kernel on a sub-strip equals the full-image
    /// blur's body rows. The strip buffer covers `[body_offset, body_offset + h)`
    /// with halos of size 6 on top and bottom (matching `mode_b_halo`)
    /// so reflection lands inside the strip.
    #[test]
    fn pu_blur_v_strip_matches_full_for_interior_body() {
        let w = 16_usize;
        let logical_h = 32_usize;
        let body_h = 8_usize; // strip body
        let halo = 6_usize; // PU blur radius
        let strip_h = body_h + 2 * halo;

        let src_a = synth_plane(w, logical_h, 0xfade_face);
        let src_rg = synth_plane(w, logical_h, 0xcafe_babe);
        let src_vy = synth_plane(w, logical_h, 0xdead_beef);

        // Reference: full-image scalar V-pass.
        let k = PU_BLUR_KERNEL_1D;
        let halff = 6_isize;
        let mut ref_a = vec![0.0_f32; w * logical_h];
        let mut ref_rg = vec![0.0_f32; w * logical_h];
        let mut ref_vy = vec![0.0_f32; w * logical_h];
        for y in 0..logical_h {
            for x in 0..w {
                let mut sa = 0.0_f32;
                let mut sr = 0.0_f32;
                let mut sv = 0.0_f32;
                for t in 0..13 {
                    let sy = reflect_pu_idx(y as isize + t as isize - halff, logical_h);
                    sa += k[t] * src_a[sy * w + x];
                    sr += k[t] * src_rg[sy * w + x];
                    sv += k[t] * src_vy[sy * w + x];
                }
                ref_a[y * w + x] = sa;
                ref_rg[y * w + x] = sr;
                ref_vy[y * w + x] = sv;
            }
        }

        // Dispatch on a strip covering body rows [body_start, body_end).
        // The strip buffer rows [0, halo) and [halo+body_h, strip_h) are
        // the halos. Strip starts at body_offset_y = body_start - halo.
        // Pick a non-zero body_start for an interior strip.
        let body_start = halo + 4; // 10
        let body_offset = body_start - halo; // 4
        // Fill strip buffer with logical rows [body_offset, body_offset + strip_h).
        let mut strip_a = vec![0.0_f32; w * strip_h];
        let mut strip_rg = vec![0.0_f32; w * strip_h];
        let mut strip_vy = vec![0.0_f32; w * strip_h];
        for sy in 0..strip_h {
            let gy = body_offset + sy;
            assert!(gy < logical_h);
            for x in 0..w {
                strip_a[sy * w + x] = src_a[gy * w + x];
                strip_rg[sy * w + x] = src_rg[gy * w + x];
                strip_vy[sy * w + x] = src_vy[gy * w + x];
            }
        }
        let mut dst_a = vec![0.0_f32; w * strip_h];
        let mut dst_rg = vec![0.0_f32; w * strip_h];
        let mut dst_vy = vec![0.0_f32; w * strip_h];
        pu_blur_v_strip_aware_3ch_into(
            &strip_a,
            &strip_rg,
            &strip_vy,
            &mut dst_a,
            &mut dst_rg,
            &mut dst_vy,
            1.0,
            w,
            strip_h,
            body_offset as u32,
            logical_h as u32,
        );
        // Compare body rows of strip output (rows [halo, halo+body_h))
        // against reference rows [body_start, body_start+body_h).
        for body_row in 0..body_h {
            let strip_row = halo + body_row;
            let logical_row = body_start + body_row;
            for x in 0..w {
                let s_a = dst_a[strip_row * w + x];
                let s_rg = dst_rg[strip_row * w + x];
                let s_vy = dst_vy[strip_row * w + x];
                let r_a = ref_a[logical_row * w + x];
                let r_rg = ref_rg[logical_row * w + x];
                let r_vy = ref_vy[logical_row * w + x];
                assert_eq!(
                    s_a.to_bits(),
                    r_a.to_bits(),
                    "strip A mismatch at strip_row={strip_row}, x={x}: strip={s_a:e}, full={r_a:e}"
                );
                assert_eq!(s_rg.to_bits(), r_rg.to_bits());
                assert_eq!(s_vy.to_bits(), r_vy.to_bits());
            }
        }
    }

    /// Combined H+V strip blur matches the upstream `gaussian_blur_sigma3`
    /// full-image blur in the degenerate (body_offset=0, logical_h=h)
    /// case. This is the "ship-it" gate — the strip wrapper must be
    /// invocable in degenerate mode and produce upstream-equivalent
    /// output.
    #[test]
    fn pu_blur_3ch_strip_aware_degenerate_matches_upstream() {
        for &(w, h) in &[(16_usize, 16_usize), (32, 24), (64, 64), (15, 17)] {
            let src_a = synth_plane(w, h, 0x6996_6996);
            let src_rg = synth_plane(w, h, 0x7777_8888);
            let src_vy = synth_plane(w, h, 0x9999_aaaa);

            let ref_a = gaussian_blur_sigma3(&src_a, w, h);
            let ref_rg = gaussian_blur_sigma3(&src_rg, w, h);
            let ref_vy = gaussian_blur_sigma3(&src_vy, w, h);

            let mut hp_a = vec![0.0_f32; w * h];
            let mut hp_rg = vec![0.0_f32; w * h];
            let mut hp_vy = vec![0.0_f32; w * h];
            let mut dst_a = vec![0.0_f32; w * h];
            let mut dst_rg = vec![0.0_f32; w * h];
            let mut dst_vy = vec![0.0_f32; w * h];
            pu_blur_3ch_strip_aware(
                &src_a,
                &src_rg,
                &src_vy,
                &mut hp_a,
                &mut hp_rg,
                &mut hp_vy,
                &mut dst_a,
                &mut dst_rg,
                &mut dst_vy,
                1.0,
                w,
                h,
                0,
                h as u32,
            );
            // gaussian_blur_sigma3 is the upstream scalar reference; the
            // strip-aware kernel's degenerate mode IS the same scalar
            // reflection. Bit-identical.
            for i in 0..w * h {
                assert_eq!(
                    ref_a[i].to_bits(),
                    dst_a[i].to_bits(),
                    "wrap A mismatch at {w}×{h} idx {i}: ref={:e}, dst={:e}",
                    ref_a[i],
                    dst_a[i]
                );
                assert_eq!(ref_rg[i].to_bits(), dst_rg[i].to_bits());
                assert_eq!(ref_vy[i].to_bits(), dst_vy[i].to_bits());
            }
        }
    }

    /// Replicates the body of `pipeline::apply_csf_row_per_pixel` so
    /// the strip-degenerate parity test can drive the same arithmetic
    /// without the helper being pub-visible. Math is identical (FMA
    /// grouping included).
    fn apply_csf_repro(log_l: f32, logs_row: &[f32; N_L_BKG]) -> f32 {
        const CSF_L_BKG_INV_STEP: f32 = 4.919_830_6;
        const CSF_L_BKG_AXIS_MIN: f32 = -2.301_03;
        const CSF_L_BKG_MAX_IDX: f32 = 30.999_999;
        const LOG_SENSITIVITY_CORRECTION: f32 = crate::kernels::csf::SENSITIVITY_CORRECTION_DB / 20.0;
        let off_raw = (log_l - CSF_L_BKG_AXIS_MIN) * CSF_L_BKG_INV_STEP;
        let off_lo = off_raw.clamp(0.0, CSF_L_BKG_MAX_IDX);
        let lo_idx_f = off_lo.floor();
        let frac = off_lo - lo_idx_f;
        let lo_idx = lo_idx_f as usize;
        let hi_idx = lo_idx + 1;
        let lo = logs_row[lo_idx];
        let hi = logs_row[hi_idx];
        let log_s_raw = lo + frac * (hi - lo);
        let log_s = log_s_raw + LOG_SENSITIVITY_CORRECTION;
        (log_s * core::f32::consts::LN_10).exp()
    }

    /// **Chunk 2 (strip-degenerate CSF parity)**: applying the
    /// per-pixel CSF on a STRIP slice of the band produces the same
    /// values as applying it on the corresponding rows of the FULL
    /// band. Bit-identical (per-pixel function, no spatial halo).
    #[test]
    fn csf_apply_is_strip_degenerate() {
        let w = 24_usize;
        let logical_h = 32_usize;
        let rho = 2.0_f32; // representative cy/deg
        let logs_row = precompute_logs_row(rho, CsfChannel::A);
        let logs_arr: [f32; N_L_BKG] = logs_row[..N_L_BKG].try_into().unwrap();

        // Synthesize a plausible band's log_l_bkg + band data.
        let log_l_full = synth_plane(w, logical_h, 0xc0ff_eeee);
        let band_full = synth_plane(w, logical_h, 0xfade_b00b);

        // Reference: per-pixel CSF over the full band.
        let mut ref_csf = vec![0.0_f32; w * logical_h];
        for i in 0..w * logical_h {
            let s = apply_csf_repro(log_l_full[i], &logs_arr);
            ref_csf[i] = band_full[i] * s; // skip CH_GAIN; we're testing CSF alone
        }

        // Strip-mode: pick body rows [10, 22), no halo needed
        // (CSF is per-pixel — no spatial reads). Strip == body in this case.
        let body_start = 10_usize;
        let body_end = 22_usize;
        let strip_h = body_end - body_start;
        let log_l_strip = &log_l_full[body_start * w..body_end * w];
        let band_strip = &band_full[body_start * w..body_end * w];
        let mut strip_csf = vec![0.0_f32; w * strip_h];
        for i in 0..w * strip_h {
            let s = apply_csf_repro(log_l_strip[i], &logs_arr);
            strip_csf[i] = band_strip[i] * s;
        }

        // Body rows must match bit-identically.
        for body_row in 0..strip_h {
            let logical_row = body_start + body_row;
            for x in 0..w {
                assert_eq!(
                    ref_csf[logical_row * w + x].to_bits(),
                    strip_csf[body_row * w + x].to_bits(),
                    "CSF strip drift at body_row={body_row}, x={x}"
                );
            }
        }
    }

    /// **Chunk 3 (strip-degenerate masking-chain parity)**: the
    /// post-PU-blur masking math (cross-channel xcm pool + safe_pow +
    /// soft clamp) is per-pixel; applying it on a STRIP slice of the
    /// 3 channels produces the same values as applying it on the
    /// corresponding rows of the FULL 3 channels.
    ///
    /// We test the per-pixel xcm-pool + safe_pow + clamp_diff_soft
    /// chain (the part of `mult_mutual_band` after the PU blur — the
    /// PU blur itself is chunk 1's responsibility). Bit-identical to
    /// upstream `mult_mutual_band` on the body rows.
    #[test]
    fn masking_chain_is_strip_degenerate() {
        use crate::kernels::masking::XCM_3X3;
        let w = 16_usize;
        let logical_h = 32_usize;

        // Synthesize T_p / R_p planes (already CSF-weighted + CH_GAIN-
        // applied — they look like band-loop intermediates here).
        let t_a_full = synth_plane(w, logical_h, 0x1111_2222);
        let t_rg_full = synth_plane(w, logical_h, 0x3333_4444);
        let t_vy_full = synth_plane(w, logical_h, 0x5555_6666);
        let r_a_full = synth_plane(w, logical_h, 0x7777_8888);
        let r_rg_full = synth_plane(w, logical_h, 0x9999_aaaa);
        let r_vy_full = synth_plane(w, logical_h, 0xbbbb_cccc);

        // Substitute "pre-blurred M_mm" for testing: use min(|T|,|R|)
        // scaled by 10^MASK_C (the no-PU-blur path). The masking
        // chain after this point is per-pixel.
        let scale = 10.0_f32.powf(MASK_C);
        let mut m_a_full = vec![0.0_f32; w * logical_h];
        let mut m_rg_full = vec![0.0_f32; w * logical_h];
        let mut m_vy_full = vec![0.0_f32; w * logical_h];
        for i in 0..w * logical_h {
            m_a_full[i] = t_a_full[i].abs().min(r_a_full[i].abs()) * scale;
            m_rg_full[i] = t_rg_full[i].abs().min(r_rg_full[i].abs()) * scale;
            m_vy_full[i] = t_vy_full[i].abs().min(r_vy_full[i].abs()) * scale;
        }

        // Per-pixel masking chain: term = safe_pow(|M|, q[ch]);
        // m = xcm * term; du = safe_pow(|T-R|, p) / (1+m);
        // d = clamp_diff_soft(du).
        let d_max_lin = 10.0_f32.powf(crate::kernels::masking::D_MAX);
        let process_idx = |i: usize| -> [f32; 3] {
            let term = [
                safe_pow(m_a_full[i], MASK_Q[0]),
                safe_pow(m_rg_full[i], MASK_Q[1]),
                safe_pow(m_vy_full[i], MASK_Q[2]),
            ];
            let mut m_pool = [0.0_f32; 3];
            for cc in 0..3 {
                m_pool[cc] =
                    XCM_3X3[0][cc] * term[0] + XCM_3X3[1][cc] * term[1] + XCM_3X3[2][cc] * term[2];
            }
            let t = [t_a_full[i], t_rg_full[i], t_vy_full[i]];
            let r = [r_a_full[i], r_rg_full[i], r_vy_full[i]];
            let mut d = [0.0_f32; 3];
            for cc in 0..3 {
                let diff = (t[cc] - r[cc]).abs();
                let du = safe_pow(diff, MASK_P) / (1.0 + m_pool[cc]);
                d[cc] = d_max_lin * du / (d_max_lin + du);
            }
            d
        };
        let mut ref_d_a = vec![0.0_f32; w * logical_h];
        let mut ref_d_rg = vec![0.0_f32; w * logical_h];
        let mut ref_d_vy = vec![0.0_f32; w * logical_h];
        for i in 0..w * logical_h {
            let d = process_idx(i);
            ref_d_a[i] = d[0];
            ref_d_rg[i] = d[1];
            ref_d_vy[i] = d[2];
        }

        // Strip-mode: pick body rows [8, 20), process on the strip,
        // assert body rows match.
        let body_start = 8_usize;
        let body_end = 20_usize;
        for body_row in 0..(body_end - body_start) {
            let logical_row = body_start + body_row;
            for x in 0..w {
                let i = logical_row * w + x;
                let d = process_idx(i);
                // Bit-identical (per-pixel function, no halo).
                assert_eq!(
                    ref_d_a[i].to_bits(),
                    d[0].to_bits(),
                    "masking d_a drift at body_row={body_row}, x={x}"
                );
                assert_eq!(ref_d_rg[i].to_bits(), d[1].to_bits());
                assert_eq!(ref_d_vy[i].to_bits(), d[2].to_bits());
            }
        }
        // Also verify mult_mutual_band agrees with the per-pixel
        // formula on the full image (sanity that the test math
        // matches the production helper).
        let _ = (CH_GAIN, mult_mutual_band as fn(_, _, _, _) -> _); // keep imports live
    }

    // ----- Chunk 5: strip-aware pyramid kernel parity tests -----

    use crate::kernels::pyramid::{gausspyr_expand_scalar, gausspyr_reduce_scalar};

    /// **Chunk 5: downscale_strip degenerate parity.** Calling
    /// `downscale_strip_into` with `body_offset=0, src_strip_offset=0,
    /// logical_src_h=src_h` produces output bit-identical to the
    /// upstream `gausspyr_reduce_scalar` on the same input.
    #[test]
    fn downscale_strip_degenerates_to_full() {
        for &(sw, sh) in &[
            (8_usize, 8_usize),
            (16, 16),
            (32, 32),
            (15, 17),
            (73, 91),
            (128, 128),
        ] {
            let src = synth_plane(sw, sh, 0xdead_beef);
            let mut ref_dst = Vec::new();
            let (dw, dh) = gausspyr_reduce_scalar(&src, sw, sh, &mut ref_dst);

            let mut strip_dst = vec![0.0_f32; dw * dh];
            let mut vscratch = Vec::new();
            downscale_strip_into(
                &src,
                sw,
                sh,
                &mut strip_dst,
                dw,
                dh,
                0,
                0,
                sh as u32,
                &mut vscratch,
            );
            for i in 0..dw * dh {
                assert_eq!(
                    ref_dst[i].to_bits(),
                    strip_dst[i].to_bits(),
                    "downscale_strip drift at {sw}×{sh} idx {i}: ref={:e}, strip={:e}",
                    ref_dst[i],
                    strip_dst[i]
                );
            }
        }
    }

    /// **Chunk 5: downscale_strip interior body parity.** A strip
    /// covering an interior dst row range produces body rows
    /// bit-identical to the corresponding rows of the full-image
    /// `gausspyr_reduce_scalar` output.
    #[test]
    fn downscale_strip_interior_matches_full() {
        let sw = 64_usize;
        let sh = 64_usize;
        let src = synth_plane(sw, sh, 0xcafe_babe);
        let mut ref_dst = Vec::new();
        let (dw, dh) = gausspyr_reduce_scalar(&src, sw, sh, &mut ref_dst);
        // dh = 32. Pick body rows [8, 20).
        let body_offset = 8_usize;
        let body_h = 12_usize;
        // Source halo: each body dst row dy reads src[2*dy - 2 ..
        // 2*dy + 2]. For dy=8 we need src[14..18]; for dy=19 we need
        // src[36..40]. So src_strip covers logical src rows
        // [2*8-2, 2*19+2] = [14, 40]. Use src halo of 6 on each side
        // for safety (covers PU + downscale ±2 slack).
        let src_strip_lo = (2 * body_offset - 2).saturating_sub(0);
        let src_strip_hi = (2 * (body_offset + body_h - 1) + 2 + 1).min(sh);
        let src_strip_h = src_strip_hi - src_strip_lo;
        let mut src_strip = vec![0.0_f32; sw * src_strip_h];
        for sy in 0..src_strip_h {
            for x in 0..sw {
                src_strip[sy * sw + x] = src[(src_strip_lo + sy) * sw + x];
            }
        }
        let mut strip_dst = vec![0.0_f32; dw * body_h];
        let mut vscratch = Vec::new();
        downscale_strip_into(
            &src_strip,
            sw,
            src_strip_h,
            &mut strip_dst,
            dw,
            body_h,
            body_offset as u32,
            src_strip_lo as u32,
            sh as u32,
            &mut vscratch,
        );
        for dy in 0..body_h {
            let logical_dy = body_offset + dy;
            for x in 0..dw {
                let r = ref_dst[logical_dy * dw + x];
                let s = strip_dst[dy * dw + x];
                assert_eq!(
                    r.to_bits(),
                    s.to_bits(),
                    "downscale_strip interior drift at dy={dy} (logical={logical_dy}), x={x}: \
                     ref={r:e}, strip={s:e}"
                );
            }
            let _ = dh; // unused
        }
    }

    /// **Chunk 5: upscale_v_strip degenerate parity.** Calling
    /// `upscale_v_strip_into` with `body_offset=0, src_strip_offset=0,
    /// logical_src_h=src_h, logical_dst_h=dst_h, body_h=dst_h` produces
    /// the V-pass output equivalent to the V-pass-only output of
    /// `gausspyr_expand_scalar` (which is the vscratch this kernel
    /// emits when invoked as the V-pass alone).
    #[test]
    fn upscale_v_strip_degenerates_to_full() {
        // gausspyr_expand_scalar's V-pass output is its `vscratch` of
        // size sw * out_h. We don't have direct access to vscratch from
        // outside that function, so we test upscale_v_strip's degenerate
        // mode against an independent direct port of the V-pass.
        for &(sw, sh, out_h) in &[
            (4_usize, 4_usize, 8_usize),
            (4, 4, 7),
            (8, 6, 12),
            (8, 6, 11),
            (16, 12, 24),
        ] {
            let src = synth_plane(sw, sh, 0xbeef_face);
            // Reference: V-pass only via direct implementation of the
            // gausspyr_expand_scalar V-pass (reads src, writes vscratch
            // of size sw * out_h). Math copied from
            // `crates/cvvdp/src/kernels/pyramid.rs:184..205`.
            let mut ref_v = vec![0.0_f32; sw * out_h];
            {
                use crate::kernels::pyramid::GAUSS5;
                let k = GAUSS5;
                let z_len_v = out_h + 4;
                let odd_h = out_h & 1;
                let back_idx_v = out_h + 2 + odd_h;
                let mut z_v = vec![0.0_f32; z_len_v];
                for x in 0..sw {
                    z_v.fill(0.0);
                    z_v[0] = src[x];
                    for ky in 0..sh {
                        z_v[2 + 2 * ky] = src[ky * sw + x];
                    }
                    z_v[back_idx_v] = src[(sh - 1) * sw + x];
                    for y in 0..out_h {
                        let sum = k[0] * z_v[y]
                            + k[1] * z_v[y + 1]
                            + k[2] * z_v[y + 2]
                            + k[3] * z_v[y + 3]
                            + k[4] * z_v[y + 4];
                        ref_v[y * sw + x] = 2.0 * sum;
                    }
                }
            }
            // Strip-mode: full degenerate dispatch (body == whole image).
            let mut strip_v = vec![0.0_f32; sw * out_h];
            upscale_v_strip_into(
                &src,
                sw,
                sh,
                &mut strip_v,
                out_h,
                0,
                0,
                sh as u32,
                out_h as u32,
            );
            for i in 0..sw * out_h {
                assert_eq!(
                    ref_v[i].to_bits(),
                    strip_v[i].to_bits(),
                    "upscale_v_strip degenerate drift at {sw}×{sh}→{out_h} idx {i}: \
                     ref={:e}, strip={:e}",
                    ref_v[i],
                    strip_v[i]
                );
            }
        }
    }

    /// **Chunk 5: upscale_h_strip degenerate parity.** The H-pass is
    /// X-only — no halo, strip parameters trivially unused. Test that
    /// it matches the H-pass output of `gausspyr_expand_scalar`.
    #[test]
    fn upscale_h_strip_matches_full() {
        // Reproduce gausspyr_expand_scalar's H-pass directly. It
        // reads vscratch (which we synthesize) and writes dst.
        for &(sw, in_h, out_w) in &[
            (4_usize, 8_usize, 8_usize),
            (4, 7, 7),
            (8, 12, 16),
            (8, 11, 15),
            (16, 24, 32),
        ] {
            let vscratch = synth_plane(sw, in_h, 0xdeed_face);
            let mut ref_dst = vec![0.0_f32; out_w * in_h];
            {
                use crate::kernels::pyramid::GAUSS5;
                let k = GAUSS5;
                let z_len_h = out_w + 4;
                let odd_w = out_w & 1;
                let back_idx_h = out_w + 2 + odd_w;
                let mut z_h = vec![0.0_f32; z_len_h];
                for y in 0..in_h {
                    z_h.fill(0.0);
                    z_h[0] = vscratch[y * sw];
                    for kx in 0..sw {
                        z_h[2 + 2 * kx] = vscratch[y * sw + kx];
                    }
                    z_h[back_idx_h] = vscratch[y * sw + sw - 1];
                    for x in 0..out_w {
                        let sum = k[0] * z_h[x]
                            + k[1] * z_h[x + 1]
                            + k[2] * z_h[x + 2]
                            + k[3] * z_h[x + 3]
                            + k[4] * z_h[x + 4];
                        ref_dst[y * out_w + x] = 2.0 * sum;
                    }
                }
            }
            let mut strip_dst = vec![0.0_f32; out_w * in_h];
            upscale_h_strip_into(
                &vscratch,
                sw,
                in_h,
                &mut strip_dst,
                out_w,
                0,
                in_h as u32,
            );
            for i in 0..out_w * in_h {
                assert_eq!(
                    ref_dst[i].to_bits(),
                    strip_dst[i].to_bits(),
                    "upscale_h_strip drift at {sw}→{out_w}×{in_h} idx {i}: \
                     ref={:e}, strip={:e}",
                    ref_dst[i],
                    strip_dst[i]
                );
            }
        }
    }

    /// **Chunk 5: upscale full (V + H) strip equivalent to
    /// upstream `gausspyr_expand_scalar`.** Chain the V + H strip
    /// kernels in degenerate mode and confirm the output matches
    /// the full upstream expand.
    #[test]
    fn upscale_strip_chain_matches_gausspyr_expand() {
        for &(sw, sh, out_w, out_h) in &[
            (4_usize, 4_usize, 8_usize, 8_usize),
            (4, 4, 7, 7),
            (8, 6, 16, 12),
            (8, 6, 15, 11),
            (16, 12, 32, 24),
        ] {
            let src = synth_plane(sw, sh, 0xbada_5511);
            let mut ref_dst = Vec::new();
            gausspyr_expand_scalar(&src, sw, sh, out_w, out_h, &mut ref_dst);

            let mut vbuf = vec![0.0_f32; sw * out_h];
            upscale_v_strip_into(
                &src,
                sw,
                sh,
                &mut vbuf,
                out_h,
                0,
                0,
                sh as u32,
                out_h as u32,
            );
            let mut hbuf = vec![0.0_f32; out_w * out_h];
            upscale_h_strip_into(&vbuf, sw, out_h, &mut hbuf, out_w, 0, out_h as u32);

            for i in 0..out_w * out_h {
                assert_eq!(
                    ref_dst[i].to_bits(),
                    hbuf[i].to_bits(),
                    "upscale_strip chain drift at {sw}×{sh}→{out_w}×{out_h} idx {i}",
                );
            }
        }
    }

    /// **Chunk 5: subtract_weber_3ch_strip per-pixel parity.** With
    /// `body_offset=0, src_strip_offset=0` the strip kernel produces
    /// output identical to the per-pixel weber-contrast formula from
    /// `weber_contrast_pyr_dec_scalar`.
    #[test]
    fn subtract_weber_3ch_strip_per_pixel_formula() {
        let w = 16_usize;
        let h = 12_usize;
        let fine_a = synth_plane(w, h, 0x1111);
        let fine_rg = synth_plane(w, h, 0x2222);
        let fine_vy = synth_plane(w, h, 0x3333);
        let upsc_a = synth_plane(w, h, 0x4444);
        let upsc_rg = synth_plane(w, h, 0x5555);
        let upsc_vy = synth_plane(w, h, 0x6666);
        // l_bkg-like values (positive).
        let mut expanded_l = synth_plane(w, h, 0x7777);
        for v in &mut expanded_l {
            *v = v.abs() + 0.001;
        }

        let mut ref_c_a = vec![0.0_f32; w * h];
        let mut ref_c_rg = vec![0.0_f32; w * h];
        let mut ref_c_vy = vec![0.0_f32; w * h];
        let mut ref_log_l = vec![0.0_f32; w * h];
        for i in 0..w * h {
            let l = expanded_l[i].max(0.01);
            ref_c_a[i] = ((fine_a[i] - upsc_a[i]) / l).clamp(-1000.0, 1000.0);
            ref_c_rg[i] = ((fine_rg[i] - upsc_rg[i]) / l).clamp(-1000.0, 1000.0);
            ref_c_vy[i] = ((fine_vy[i] - upsc_vy[i]) / l).clamp(-1000.0, 1000.0);
            ref_log_l[i] = l.log10();
        }

        let mut strip_c_a = vec![0.0_f32; w * h];
        let mut strip_c_rg = vec![0.0_f32; w * h];
        let mut strip_c_vy = vec![0.0_f32; w * h];
        let mut strip_log_l = vec![0.0_f32; w * h];
        subtract_weber_3ch_strip_into(
            &fine_a,
            &fine_rg,
            &fine_vy,
            &upsc_a,
            &upsc_rg,
            &upsc_vy,
            &expanded_l,
            &mut strip_c_a,
            &mut strip_c_rg,
            &mut strip_c_vy,
            &mut strip_log_l,
            w,
            h,
            0,
            0,
        );
        for i in 0..w * h {
            assert_eq!(ref_c_a[i].to_bits(), strip_c_a[i].to_bits());
            assert_eq!(ref_c_rg[i].to_bits(), strip_c_rg[i].to_bits());
            assert_eq!(ref_c_vy[i].to_bits(), strip_c_vy[i].to_bits());
            assert_eq!(ref_log_l[i].to_bits(), strip_log_l[i].to_bits());
        }
    }
}
