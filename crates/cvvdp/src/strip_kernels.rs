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

use alloc::vec;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::masking::gaussian_blur_sigma3;

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
}
