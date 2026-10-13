//! SIMD-vectorized 5-tap pyramid reduce / expand kernels (Chunk 2 of
//! the SIMD optimization plan).
//!
//! Targets the inner column/row sweeps of [`gausspyr_reduce`] and
//! [`gausspyr_expand`] (in [`super::pyramid`]) which together account
//! for ~24 % of cvvdp wall time at 1024² (per the
//! `cvvdp_cpu_flamegraph_2026-05-25.svg` attribution).
//!
//! Approach: column-major SIMD over 8/16 contiguous f32 columns using
//! [`archmage`]'s safe capability-token dispatch with [`magetypes`]
//! generic SIMD primitives. Boundary rows/columns + odd-width tails
//! stay scalar — the inner SIMD path covers the bulk pixels.
//!
//! All SIMD entry points produce output matched within `<= 1e-5 abs`
//! to the scalar paths in [`super::pyramid`] (FMA grouping differs at
//! the lane boundaries — we accumulate in a 5-tap dot pattern that
//! LLVM may fuse differently than the scalar `+` chain). The actual
//! delta is far below the 1e-4 JOD parity floor.
//!
//! Boundary handling matches the pycvvdp v0.5.4 quirk exactly. The
//! SIMD inner passes do NOT apply the first/last row/column patches —
//! those stay scalar in the caller (in `pyramid::gausspyr_reduce`)
//! so the FMA grouping of the patches is preserved.

#![allow(clippy::needless_range_loop)]
// The per-tier #[attune] family expansion makes some loops appear as
// "manual memcpy" candidates to clippy, but the loop body inside the
// expansion may include the SIMD-load pattern's bookkeeping. Suppress
// at module scope rather than per-site (cleaner).
#![allow(clippy::manual_memcpy)]

use alloc::vec::Vec;

use crate::kernels::masking::PU_BLUR_KERNEL_1D;
use crate::kernels::pyramid::GAUSS5;

/// Reflect `i` into `[0, n)` for the 13-tap PU blur. Matches
/// torchvision's `F.pad(..., mode='reflect')` behaviour exactly —
/// bit-equivalent to `crate::kernels::masking::reflect_idx_for_blur`
/// (private upstream, re-implemented locally for SIMD boundary use).
#[inline]
fn reflect_idx_for_blur(i: isize, n: usize) -> usize {
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

// Two magetypes blocks: AVX-512 family uses 16-wide; everyone else
// uses 8-wide (native on AVX2, polyfilled to 2× on NEON/wasm/scalar).
// This mirrors the zensim `downscale_2x_into_inner` pattern (zensim
// 0.3.0 src/blur.rs:3318-3391).
//
// The `define(...)` clause injects a local `f32x{N} = generic::f32x{N}<Token>`
// alias at the top of every variant body, so the same code compiles
// against every tier with the right SIMD width.

// ============================================================================
// reduce_vertical_pass — sw × dh output buffer
// ============================================================================
//
// Per scalar reference (crate::kernels::pyramid::gausspyr_reduce_scalar):
//   for dy in 0..dh:
//     cy = 2 * dy
//     for x in 0..sw:
//       vscratch[dy*sw + x] = Σᵢ GAUSS5[i] · R(cy + i - 2, x)
//   R(r, x) = src[r*sw + x] if 0 ≤ r < sh else 0.
//
// SIMD across x: per dy, the 5 source-row offsets are uniform; we sweep
// f32x{N} chunks across columns, broadcast-multiplying by the loop-
// invariant scalar GAUSS5 taps.

#[archmage::attune(_v4x_t(cfg(avx512)), define(f32x16))]
fn reduce_v_inner(
    token: Token,
    src: &[f32],
    sw: usize,
    sh: usize,
    dh: usize,
    vscratch: &mut [f32],
) {
    let k = GAUSS5;
    let k0 = f32x16::splat_t(token, k[0]);
    let k1 = f32x16::splat_t(token, k[1]);
    let k2 = f32x16::splat_t(token, k[2]);
    let k3 = f32x16::splat_t(token, k[3]);
    let k4 = f32x16::splat_t(token, k[4]);

    debug_assert_eq!(vscratch.len(), sw * dh);
    let n_groups = sw / 16;
    crate::par::map_rows(vscratch, sw, |y0, vscratch| {
        let band_h = vscratch.len() / sw;
        for dy_l in 0..band_h {
            let cy = 2 * (y0 + dy_l) as isize;
            // Bounds check on row offsets is uniform across the row → one branch
            // per dy, not per column.
            let row_at = |off: isize| -> Option<usize> {
                let r = cy + off;
                if r < 0 || r >= sh as isize {
                    None
                } else {
                    Some(r as usize * sw)
                }
            };
            let r_m2 = row_at(-2);
            let r_m1 = row_at(-1);
            let r_0 = row_at(0);
            let r_p1 = row_at(1);
            let r_p2 = row_at(2);
            let out_base = dy_l * sw;

            for cg in 0..n_groups {
                let col = cg * 16;
                let ld = |base: usize| -> f32x16 {
                    let arr: [f32; 16] = src[base + col..base + col + 16].try_into().unwrap();
                    f32x16::from_array_t(token, arr)
                };
                // Interior rows have all five taps in bounds — one match
                // per column-group instead of per-tap Option handling.
                let (v0, v1, v2, v3v, v4x) = match (r_m2, r_m1, r_0, r_p1, r_p2) {
                    (Some(b0), Some(b1), Some(b2), Some(b3), Some(b4)) => {
                        (ld(b0), ld(b1), ld(b2), ld(b3), ld(b4))
                    }
                    _ => {
                        let z = f32x16::zero_t(token);
                        let or = |row: Option<usize>| row.map_or(z, ld);
                        (or(r_m2), or(r_m1), or(r_0), or(r_p1), or(r_p2))
                    }
                };
                let acc = v0 * k0 + v1 * k1 + v2 * k2 + v3v * k3 + v4x * k4;
                let arr = acc.to_array();
                vscratch[out_base + col..out_base + col + 16].copy_from_slice(&arr);
            }

            // Scalar tail (< 16 columns).
            for x in n_groups * 16..sw {
                let read = |row: Option<usize>| -> f32 {
                    match row {
                        Some(base) => src[base + x],
                        None => 0.0,
                    }
                };
                vscratch[out_base + x] = k[0] * read(r_m2)
                    + k[1] * read(r_m1)
                    + k[2] * read(r_0)
                    + k[3] * read(r_p1)
                    + k[4] * read(r_p2);
            }
        }
    });
}

#[archmage::attune(_*_t, define(f32x8))]
fn reduce_v_inner(
    token: Token,
    src: &[f32],
    sw: usize,
    sh: usize,
    dh: usize,
    vscratch: &mut [f32],
) {
    let k = GAUSS5;
    let k0 = f32x8::splat_t(token, k[0]);
    let k1 = f32x8::splat_t(token, k[1]);
    let k2 = f32x8::splat_t(token, k[2]);
    let k3 = f32x8::splat_t(token, k[3]);
    let k4 = f32x8::splat_t(token, k[4]);

    debug_assert_eq!(vscratch.len(), sw * dh);
    let n_groups = sw / 8;
    crate::par::map_rows(vscratch, sw, |y0, vscratch| {
        let band_h = vscratch.len() / sw;
        for dy_l in 0..band_h {
            let cy = 2 * (y0 + dy_l) as isize;
            let row_at = |off: isize| -> Option<usize> {
                let r = cy + off;
                if r < 0 || r >= sh as isize {
                    None
                } else {
                    Some(r as usize * sw)
                }
            };
            let r_m2 = row_at(-2);
            let r_m1 = row_at(-1);
            let r_0 = row_at(0);
            let r_p1 = row_at(1);
            let r_p2 = row_at(2);
            let out_base = dy_l * sw;

            for cg in 0..n_groups {
                let col = cg * 8;
                let ld = |base: usize| -> f32x8 {
                    let arr: [f32; 8] = src[base + col..base + col + 8].try_into().unwrap();
                    f32x8::from_array_t(token, arr)
                };
                let (v0, v1, v2, v3v, v4x) = match (r_m2, r_m1, r_0, r_p1, r_p2) {
                    (Some(b0), Some(b1), Some(b2), Some(b3), Some(b4)) => {
                        (ld(b0), ld(b1), ld(b2), ld(b3), ld(b4))
                    }
                    _ => {
                        let z = f32x8::zero_t(token);
                        let or = |row: Option<usize>| row.map_or(z, ld);
                        (or(r_m2), or(r_m1), or(r_0), or(r_p1), or(r_p2))
                    }
                };
                let acc = v0 * k0 + v1 * k1 + v2 * k2 + v3v * k3 + v4x * k4;
                let arr = acc.to_array();
                vscratch[out_base + col..out_base + col + 8].copy_from_slice(&arr);
            }

            for x in n_groups * 8..sw {
                let read = |row: Option<usize>| -> f32 {
                    match row {
                        Some(base) => src[base + x],
                        None => 0.0,
                    }
                };
                vscratch[out_base + x] = k[0] * read(r_m2)
                    + k[1] * read(r_m1)
                    + k[2] * read(r_0)
                    + k[3] * read(r_p1)
                    + k[4] * read(r_p2);
            }
        }
    });
}

/// Vertical pass of `gausspyr_reduce`: writes the `sw × dh` scratch
/// buffer.
pub(crate) fn reduce_vertical_pass(
    src: &[f32],
    sw: usize,
    sh: usize,
    dh: usize,
    vscratch: &mut [f32],
) {
    debug_assert_eq!(vscratch.len(), sw * dh);
    archmage::attuned!(
        reduce_v_inner(src, sw, sh, dh, vscratch),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
}

// ============================================================================
// reduce_horizontal_pass — dw × dh output buffer (stride-2 read)
// ============================================================================
//
// Per scalar reference:
//   for dy in 0..dh:
//     for dx in 0..dw:
//       cx = 2 * dx
//       dst[dy*dw + dx] = Σᵢ GAUSS5[i] · C(cx + i - 2, dy)
//   C(c, dy) = vscratch[dy*sw + c] if 0 ≤ c < sw else 0.
//
// Stride-2 reads make column-vectorisation gather-heavy. We instead
// vectorise across rows (dy): for each dx, process N output rows in
// parallel. Each row reads vscratch[dy*sw + (cx-2..=cx+2)]. The 5 taps
// stay loop-invariant scalars splatted across the N lanes.

#[archmage::attune(_v4x_t(cfg(avx512)), define(f32x16))]
fn reduce_h_inner(
    token: Token,
    vscratch: &[f32],
    sw: usize,
    dw: usize,
    dh: usize,
    dst: &mut [f32],
) {
    let k = GAUSS5;
    let k0 = f32x16::splat_t(token, k[0]);
    let k1 = f32x16::splat_t(token, k[1]);
    let k2 = f32x16::splat_t(token, k[2]);
    let k3 = f32x16::splat_t(token, k[3]);
    let k4 = f32x16::splat_t(token, k[4]);

    debug_assert_eq!(dst.len(), dw * dh);
    crate::par::map_rows(dst, dw, |y0, dst| {
        let band_dh = dst.len() / dw;
        // Per-row stride-2 split: pe[j+1] = v[2j], po[j+1] = v[2j+1]
        // (zero outside — the C(c)=0 edge clamp baked in). dst[dx] =
        // k0·pe[dx] + k1·po[dx] + k2·pe[dx+1] + k3·po[dx+1] + k4·pe[dx+2],
        // so every tap is a contiguous unaligned vector load.
        let mut pe = vec![0.0f32; dw + 24];
        let mut po = vec![0.0f32; dw + 24];
        for dy in 0..band_dh {
            let row = &vscratch[(y0 + dy) * sw..(y0 + dy) * sw + sw];
            pe[..dw + 4].fill(0.0);
            po[..dw + 4].fill(0.0);
            for (j, pair) in row.chunks(2).enumerate().take(dw + 6) {
                pe[j + 1] = pair[0];
                po[j + 1] = *pair.get(1).unwrap_or(&0.0);
            }

            let n_groups = dw / 16;
            for cg in 0..n_groups {
                let x = cg * 16;
                let v0 = f32x16::from_slice_t(token, &pe[x..x + 16]);
                let v1 = f32x16::from_slice_t(token, &po[x..x + 16]);
                let v2 = f32x16::from_slice_t(token, &pe[x + 1..x + 17]);
                let v3v = f32x16::from_slice_t(token, &po[x + 1..x + 17]);
                let v4x = f32x16::from_slice_t(token, &pe[x + 2..x + 18]);
                let acc = v0 * k0 + v1 * k1 + v2 * k2 + v3v * k3 + v4x * k4;
                let arr = acc.to_array();
                dst[dy * dw + x..dy * dw + x + 16].copy_from_slice(&arr);
            }
            for dx in n_groups * 16..dw {
                dst[dy * dw + dx] = k[0] * pe[dx]
                    + k[1] * po[dx]
                    + k[2] * pe[dx + 1]
                    + k[3] * po[dx + 1]
                    + k[4] * pe[dx + 2];
            }
        }
    });
}

#[archmage::attune(_*_t, define(f32x8))]
fn reduce_h_inner(
    token: Token,
    vscratch: &[f32],
    sw: usize,
    dw: usize,
    dh: usize,
    dst: &mut [f32],
) {
    let k = GAUSS5;
    let k0 = f32x8::splat_t(token, k[0]);
    let k1 = f32x8::splat_t(token, k[1]);
    let k2 = f32x8::splat_t(token, k[2]);
    let k3 = f32x8::splat_t(token, k[3]);
    let k4 = f32x8::splat_t(token, k[4]);

    debug_assert_eq!(dst.len(), dw * dh);
    crate::par::map_rows(dst, dw, |y0, dst| {
        let band_dh = dst.len() / dw;
        // Per-row stride-2 split — see the f32x16 twin above.
        let mut pe = vec![0.0f32; dw + 16];
        let mut po = vec![0.0f32; dw + 16];
        for dy in 0..band_dh {
            let row = &vscratch[(y0 + dy) * sw..(y0 + dy) * sw + sw];
            pe[..dw + 4].fill(0.0);
            po[..dw + 4].fill(0.0);
            for (j, pair) in row.chunks(2).enumerate().take(dw + 6) {
                pe[j + 1] = pair[0];
                po[j + 1] = *pair.get(1).unwrap_or(&0.0);
            }

            let n_groups = dw / 8;
            for cg in 0..n_groups {
                let x = cg * 8;
                let v0 = f32x8::from_slice_t(token, &pe[x..x + 8]);
                let v1 = f32x8::from_slice_t(token, &po[x..x + 8]);
                let v2 = f32x8::from_slice_t(token, &pe[x + 1..x + 9]);
                let v3v = f32x8::from_slice_t(token, &po[x + 1..x + 9]);
                let v4x = f32x8::from_slice_t(token, &pe[x + 2..x + 10]);
                let acc = v0 * k0 + v1 * k1 + v2 * k2 + v3v * k3 + v4x * k4;
                let arr = acc.to_array();
                dst[dy * dw + x..dy * dw + x + 8].copy_from_slice(&arr);
            }
            for dx in n_groups * 8..dw {
                dst[dy * dw + dx] = k[0] * pe[dx]
                    + k[1] * po[dx]
                    + k[2] * pe[dx + 1]
                    + k[3] * po[dx + 1]
                    + k[4] * pe[dx + 2];
            }
        }
    });
}

/// Horizontal pass of `gausspyr_reduce`: reads `sw × dh` scratch,
/// writes `dw × dh` output.
pub(crate) fn reduce_horizontal_pass(
    vscratch: &[f32],
    sw: usize,
    dw: usize,
    dh: usize,
    dst: &mut [f32],
) {
    debug_assert_eq!(vscratch.len(), sw * dh);
    debug_assert_eq!(dst.len(), dw * dh);
    archmage::attuned!(
        reduce_h_inner(vscratch, sw, dw, dh, dst),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
}

// ============================================================================
// expand_vertical_pass — sw × out_h output buffer (zero-insert 5-tap)
// ============================================================================
//
// Per scalar reference (crate::kernels::pyramid::gausspyr_expand_scalar):
//   For each column x:
//     Build z_v of length (out_h + 4):
//       z_v[0]              = src[0,x]                   (left mirror)
//       z_v[2 + 2*ky]       = src[ky,x]   for ky in 0..sh
//       z_v[back_idx_v]     = src[sh-1,x]                (right mirror)
//     (entries not written remain 0 — the zero-insertion).
//     For each y in 0..out_h:
//       vscratch[y*sw + x] = 2 * Σᵢ GAUSS5[i] · z_v[y + i]
//
// SIMD: build the z buffer per-column-group (group-wide z scratch),
// then sweep y with f32x{N} mul-adds. The 2× scaling is folded into
// the kernel taps (DOUBLED_GAUSS5).

#[archmage::attune(_v4x_t(cfg(avx512)), define(f32x16))]
fn expand_v_inner(
    token: Token,
    src: &[f32],
    sw: usize,
    sh: usize,
    out_h: usize,
    vscratch: &mut [f32],
) {
    let k = GAUSS5;
    let dk0 = f32x16::splat_t(token, 2.0 * k[0]);
    let dk1 = f32x16::splat_t(token, 2.0 * k[1]);
    let dk2 = f32x16::splat_t(token, 2.0 * k[2]);
    let dk3 = f32x16::splat_t(token, 2.0 * k[3]);
    let dk4 = f32x16::splat_t(token, 2.0 * k[4]);

    let odd_h = out_h & 1;
    let back_idx_v = out_h + 2 + odd_h;
    // `z[p]` is nonzero only at even `p <= 2*sh` (p/2−1, saturating at
    // the edges) and at `back_idx_v` — read src rows directly instead
    // of materializing the zero-inserted column.
    let z_row = |p: usize| -> Option<usize> {
        if p == back_idx_v {
            Some(sh - 1)
        } else if p & 1 == 0 && p <= 2 * sh {
            Some((p / 2).saturating_sub(1).min(sh - 1))
        } else {
            None
        }
    };
    let dk = [dk0, dk1, dk2, dk3, dk4];
    let n_groups = sw / 16;

    debug_assert_eq!(vscratch.len(), sw * out_h);
    crate::par::map_rows(vscratch, sw, |y0, vscratch| {
        let band_h = vscratch.len() / sw;
        for cg in 0..n_groups {
            let col_base = cg * 16;
            // Adjacent even/odd output rows share their source rows:
            // y = 2m reads {m-1, m, m+1} (taps 0,2,4) and y = 2m+1
            // reads {m, m+1} (taps 1,3) — 3 loads serve both rows.
            // `min(m+1, sh-1)` covers both the clamp and the
            // `back_idx_v` edge row. Per-row tap order is preserved.
            let mut y_l = 0usize;
            while y_l < band_h {
                let y = y0 + y_l;
                if y & 1 == 0 && y_l + 1 < band_h && y + 1 < out_h {
                    let m = y / 2;
                    let ra = m.saturating_sub(1).min(sh - 1) * sw + col_base;
                    let rb = m.min(sh - 1) * sw + col_base;
                    let rc = (m + 1).min(sh - 1) * sw + col_base;
                    let va = f32x16::from_array_t(token, src[ra..ra + 16].try_into().unwrap());
                    let vb = f32x16::from_array_t(token, src[rb..rb + 16].try_into().unwrap());
                    let vc = f32x16::from_array_t(token, src[rc..rc + 16].try_into().unwrap());
                    let mut acc_e = va * dk0;
                    acc_e = vb * dk2 + acc_e;
                    acc_e = vc * dk4 + acc_e;
                    let mut acc_o = vb * dk1;
                    acc_o = vc * dk3 + acc_o;
                    let arre = acc_e.to_array();
                    let arro = acc_o.to_array();
                    vscratch[y_l * sw + col_base..y_l * sw + col_base + 16].copy_from_slice(&arre);
                    vscratch[(y_l + 1) * sw + col_base..(y_l + 1) * sw + col_base + 16]
                        .copy_from_slice(&arro);
                    y_l += 2;
                } else {
                    let mut acc = f32x16::splat_t(token, 0.0);
                    for (j, dkj) in dk.iter().enumerate() {
                        if let Some(r) = z_row(y + j) {
                            let off = r * sw + col_base;
                            let v =
                                f32x16::from_array_t(token, src[off..off + 16].try_into().unwrap());
                            acc = v * *dkj + acc;
                        }
                    }
                    let arr = acc.to_array();
                    vscratch[y_l * sw + col_base..y_l * sw + col_base + 16].copy_from_slice(&arr);
                    y_l += 1;
                }
            }
        }

        // Scalar tail columns.
        for x in n_groups * 16..sw {
            for y_l in 0..band_h {
                let y = y0 + y_l;
                let mut sum = 0.0f32;
                for (j, kj) in k.iter().enumerate() {
                    if let Some(r) = z_row(y + j) {
                        sum += kj * src[r * sw + x];
                    }
                }
                vscratch[y_l * sw + x] = 2.0 * sum;
            }
        }
    });
}

#[archmage::attune(_*_t, define(f32x8))]
fn expand_v_inner(
    token: Token,
    src: &[f32],
    sw: usize,
    sh: usize,
    out_h: usize,
    vscratch: &mut [f32],
) {
    let k = GAUSS5;
    let dk0 = f32x8::splat_t(token, 2.0 * k[0]);
    let dk1 = f32x8::splat_t(token, 2.0 * k[1]);
    let dk2 = f32x8::splat_t(token, 2.0 * k[2]);
    let dk3 = f32x8::splat_t(token, 2.0 * k[3]);
    let dk4 = f32x8::splat_t(token, 2.0 * k[4]);

    let odd_h = out_h & 1;
    let back_idx_v = out_h + 2 + odd_h;
    let z_row = |p: usize| -> Option<usize> {
        if p == back_idx_v {
            Some(sh - 1)
        } else if p & 1 == 0 && p <= 2 * sh {
            Some((p / 2).saturating_sub(1).min(sh - 1))
        } else {
            None
        }
    };
    let dk = [dk0, dk1, dk2, dk3, dk4];
    let n_groups = sw / 8;

    debug_assert_eq!(vscratch.len(), sw * out_h);
    crate::par::map_rows(vscratch, sw, |y0, vscratch| {
        let band_h = vscratch.len() / sw;
        for cg in 0..n_groups {
            let col_base = cg * 8;
            // Adjacent even/odd output rows share their source rows:
            // y = 2m reads {m-1, m, m+1} (taps 0,2,4) and y = 2m+1
            // reads {m, m+1} (taps 1,3) — 3 loads serve both rows.
            // `min(m+1, sh-1)` covers both the clamp and the
            // `back_idx_v` edge row. Per-row tap order is preserved.
            let mut y_l = 0usize;
            while y_l < band_h {
                let y = y0 + y_l;
                if y & 1 == 0 && y_l + 1 < band_h && y + 1 < out_h {
                    let m = y / 2;
                    let ra = m.saturating_sub(1).min(sh - 1) * sw + col_base;
                    let rb = m.min(sh - 1) * sw + col_base;
                    let rc = (m + 1).min(sh - 1) * sw + col_base;
                    let va = f32x8::from_array_t(token, src[ra..ra + 8].try_into().unwrap());
                    let vb = f32x8::from_array_t(token, src[rb..rb + 8].try_into().unwrap());
                    let vc = f32x8::from_array_t(token, src[rc..rc + 8].try_into().unwrap());
                    let mut acc_e = va * dk0;
                    acc_e = vb * dk2 + acc_e;
                    acc_e = vc * dk4 + acc_e;
                    let mut acc_o = vb * dk1;
                    acc_o = vc * dk3 + acc_o;
                    let arre = acc_e.to_array();
                    let arro = acc_o.to_array();
                    vscratch[y_l * sw + col_base..y_l * sw + col_base + 8].copy_from_slice(&arre);
                    vscratch[(y_l + 1) * sw + col_base..(y_l + 1) * sw + col_base + 8]
                        .copy_from_slice(&arro);
                    y_l += 2;
                } else {
                    let mut acc = f32x8::splat_t(token, 0.0);
                    for (j, dkj) in dk.iter().enumerate() {
                        if let Some(r) = z_row(y + j) {
                            let off = r * sw + col_base;
                            let v =
                                f32x8::from_array_t(token, src[off..off + 8].try_into().unwrap());
                            acc = v * *dkj + acc;
                        }
                    }
                    let arr = acc.to_array();
                    vscratch[y_l * sw + col_base..y_l * sw + col_base + 8].copy_from_slice(&arr);
                    y_l += 1;
                }
            }
        }

        for x in n_groups * 8..sw {
            for y_l in 0..band_h {
                let y = y0 + y_l;
                let mut sum = 0.0f32;
                for (j, kj) in k.iter().enumerate() {
                    if let Some(r) = z_row(y + j) {
                        sum += kj * src[r * sw + x];
                    }
                }
                vscratch[y_l * sw + x] = 2.0 * sum;
            }
        }
    });
}

/// Vertical pass of `gausspyr_expand`: writes `sw × out_h` vscratch
/// using the zero-insertion + 5-tap kernel.
pub(crate) fn expand_vertical_pass(
    src: &[f32],
    sw: usize,
    sh: usize,
    out_h: usize,
    vscratch: &mut [f32],
) {
    debug_assert_eq!(vscratch.len(), sw * out_h);
    archmage::attuned!(
        expand_v_inner(src, sw, sh, out_h, vscratch),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
}

// ============================================================================
// expand_horizontal_pass — out_w × out_h output buffer
// ============================================================================
//
// Per scalar reference:
//   For each row y:
//     Build z_h of length (out_w + 4):
//       z_h[0]            = vscratch[y*sw + 0]
//       z_h[2 + 2*kx]     = vscratch[y*sw + kx]   for kx in 0..sw
//       z_h[back_idx_h]   = vscratch[y*sw + sw-1]
//     For each x in 0..out_w:
//       dst[y*out_w + x] = 2 * Σᵢ GAUSS5[i] · z_h[x + i]
//
// SIMD: vectorise across x — each f32x{N} of consecutive x values
// reads a sliding-window of 5 taps from z_h. The 5 taps are scalar
// splats. z_h is per-row scratch (we re-use a caller-owned Vec).

#[archmage::attune(_v4x_t(cfg(avx512)), define(f32x16))]
fn expand_h_inner(
    token: Token,
    vscratch: &[f32],
    sw: usize,
    out_w: usize,
    out_h: usize,
    dst: &mut [f32],
) {
    let k = GAUSS5;
    let dk0 = f32x16::splat_t(token, 2.0 * k[0]);
    let dk1 = f32x16::splat_t(token, 2.0 * k[1]);
    let dk2 = f32x16::splat_t(token, 2.0 * k[2]);
    let dk3 = f32x16::splat_t(token, 2.0 * k[3]);
    let dk4 = f32x16::splat_t(token, 2.0 * k[4]);

    let odd_w = out_w & 1;
    let back_idx_h = out_w + 2 + odd_w;

    // `z_h` index map resolved inline — no materialized zero-inserted
    // row. z[p] is row[0] at p==0, row[(p/2).saturating_sub(1)] at even
    // p <= 2*sw, row[sw-1] at even p in (2*sw, back_idx_h], else 0.
    debug_assert_eq!(dst.len(), out_w * out_h);
    crate::par::map_rows(dst, out_w, |y0, dst| {
        let band_h = dst.len() / out_w;
        for y_l in 0..band_h {
            let row = &vscratch[(y0 + y_l) * sw..(y0 + y_l) * sw + sw];
            let z_at = |p: usize| -> f32 {
                if p == 0 {
                    row[0]
                } else if p & 1 == 0 {
                    if p <= 2 * sw {
                        row[p / 2 - 1]
                    } else if p <= back_idx_h {
                        row[sw - 1]
                    } else {
                        0.0
                    }
                } else {
                    0.0
                }
            };

            // A 32-output block at even x_base = 32*cg has q = x/2
            // spanning q0..q0+16 for both parities. Interior blocks
            // (cg >= 1 so q0 - 1 >= 0, and q0 + 17 <= sw so no tap
            // clamps) read three contiguous 16-wide vectors: even
            // outputs take dk0·row[q-1] + dk2·row[q] + dk4·row[q+1]
            // (odd taps hit zero z positions) and odd outputs take
            // dk1·row[q] + dk3·row[q+1] — same product order as the
            // scalar/vector zbuf chain, whose zero taps add +0.
            let n_blocks = out_w / 32;
            let vec_hi = if sw >= 17 {
                ((sw - 17) / 16 + 1).min(n_blocks)
            } else {
                0
            };
            let vec_lo = 1.min(vec_hi);
            let head_end = (vec_lo * 32).min(out_w);

            for x in 0..head_end {
                let sum = k[0] * z_at(x)
                    + k[1] * z_at(x + 1)
                    + k[2] * z_at(x + 2)
                    + k[3] * z_at(x + 3)
                    + k[4] * z_at(x + 4);
                dst[y_l * out_w + x] = 2.0 * sum;
            }

            for cg in vec_lo..vec_hi {
                let x_base = cg * 32;
                let q0 = cg * 16;
                let ra = f32x16::from_slice_t(token, &row[q0 - 1..q0 + 15]);
                let rb = f32x16::from_slice_t(token, &row[q0..q0 + 16]);
                let rc = f32x16::from_slice_t(token, &row[q0 + 1..q0 + 17]);
                let acc_e = ra * dk0 + rb * dk2 + rc * dk4;
                let acc_o = rb * dk1 + rc * dk3;
                let ae = acc_e.to_array();
                let ao = acc_o.to_array();
                let base = y_l * out_w + x_base;
                for t in 0..16 {
                    dst[base + 2 * t] = ae[t];
                    dst[base + 2 * t + 1] = ao[t];
                }
            }

            for x in vec_hi * 32..out_w {
                let sum = k[0] * z_at(x)
                    + k[1] * z_at(x + 1)
                    + k[2] * z_at(x + 2)
                    + k[3] * z_at(x + 3)
                    + k[4] * z_at(x + 4);
                dst[y_l * out_w + x] = 2.0 * sum;
            }
        }
    });
}

#[archmage::attune(_*_t, define(f32x8))]
fn expand_h_inner(
    token: Token,
    vscratch: &[f32],
    sw: usize,
    out_w: usize,
    out_h: usize,
    dst: &mut [f32],
) {
    let k = GAUSS5;
    let dk0 = f32x8::splat_t(token, 2.0 * k[0]);
    let dk1 = f32x8::splat_t(token, 2.0 * k[1]);
    let dk2 = f32x8::splat_t(token, 2.0 * k[2]);
    let dk3 = f32x8::splat_t(token, 2.0 * k[3]);
    let dk4 = f32x8::splat_t(token, 2.0 * k[4]);

    let odd_w = out_w & 1;
    let back_idx_h = out_w + 2 + odd_w;

    // `z_h` index map resolved inline — no materialized zero-inserted
    // row. Same pairing as the f32x16 variant: 16 outputs per block,
    // three contiguous f32x8 loads.
    debug_assert_eq!(dst.len(), out_w * out_h);
    crate::par::map_rows(dst, out_w, |y0, dst| {
        let band_h = dst.len() / out_w;
        for y_l in 0..band_h {
            let row = &vscratch[(y0 + y_l) * sw..(y0 + y_l) * sw + sw];
            let z_at = |p: usize| -> f32 {
                if p == 0 {
                    row[0]
                } else if p & 1 == 0 {
                    if p <= 2 * sw {
                        row[p / 2 - 1]
                    } else if p <= back_idx_h {
                        row[sw - 1]
                    } else {
                        0.0
                    }
                } else {
                    0.0
                }
            };

            let n_blocks = out_w / 16;
            let vec_hi = if sw >= 9 {
                ((sw - 9) / 8 + 1).min(n_blocks)
            } else {
                0
            };
            let vec_lo = 1.min(vec_hi);
            let head_end = (vec_lo * 16).min(out_w);

            for x in 0..head_end {
                let sum = k[0] * z_at(x)
                    + k[1] * z_at(x + 1)
                    + k[2] * z_at(x + 2)
                    + k[3] * z_at(x + 3)
                    + k[4] * z_at(x + 4);
                dst[y_l * out_w + x] = 2.0 * sum;
            }

            for cg in vec_lo..vec_hi {
                let x_base = cg * 16;
                let q0 = cg * 8;
                let ra = f32x8::from_slice_t(token, &row[q0 - 1..q0 + 7]);
                let rb = f32x8::from_slice_t(token, &row[q0..q0 + 8]);
                let rc = f32x8::from_slice_t(token, &row[q0 + 1..q0 + 9]);
                let acc_e = ra * dk0 + rb * dk2 + rc * dk4;
                let acc_o = rb * dk1 + rc * dk3;
                let ae = acc_e.to_array();
                let ao = acc_o.to_array();
                let base = y_l * out_w + x_base;
                for t in 0..8 {
                    dst[base + 2 * t] = ae[t];
                    dst[base + 2 * t + 1] = ao[t];
                }
            }

            for x in vec_hi * 16..out_w {
                let sum = k[0] * z_at(x)
                    + k[1] * z_at(x + 1)
                    + k[2] * z_at(x + 2)
                    + k[3] * z_at(x + 3)
                    + k[4] * z_at(x + 4);
                dst[y_l * out_w + x] = 2.0 * sum;
            }
        }
    });
}

/// Horizontal pass of `gausspyr_expand`: reads `sw × out_h` vscratch,
/// writes `out_w × out_h` dst using the zero-insertion + 5-tap kernel.
pub(crate) fn expand_horizontal_pass(
    vscratch: &[f32],
    sw: usize,
    out_w: usize,
    out_h: usize,
    dst: &mut [f32],
) {
    debug_assert_eq!(dst.len(), out_w * out_h);
    archmage::attuned!(
        expand_h_inner(vscratch, sw, out_w, out_h, dst),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
}

// ============================================================================
// 13-tap σ=3 Gaussian blur (PU_BLUR_KERNEL_1D) — separable horizontal + vertical
// ============================================================================
//
// Chunk 1 of the SIMD optimization plan. Targets the #1 hot kernel
// `gaussian_blur_sigma3` from cvvdp-gpu's `kernels::masking` (32 %
// self-time at 1024² per the 2026-05-25 flamegraph). Called 3× per
// non-baseband band by `mult_mutual_band_into`.
//
// Per scalar reference (`crate::kernels::masking::gaussian_blur_sigma3`):
//   half = 6
//   for y in 0..h:
//     for x in 0..w:
//       h_pass[y*w + x] = Σ_{t=0..13} k[t] · src[y*w + reflect(x + t - 6)]
//   for y in 0..h:
//     for x in 0..w:
//       out[y*w + x] = Σ_{t=0..13} k[t] · h_pass[reflect(y + t - 6)*w + x]
//
// SIMD strategy:
//   - Horizontal pass: interior columns (6..w-6) have NO boundary
//     reflection — all 13 taps read from `src[y*w + (x-6..=x+6)]`.
//     Vectorize across `x`: for SIMD width N, output cols
//     [x_base..x_base+N] read 13 shifted N-wide loads (each starting
//     at `x_base + t - 6` for t in 0..13). Boundary patches (first 6
//     and last 6 cols) stay scalar.
//   - Vertical pass: interior rows (6..h-6) similarly vectorize
//     across `x` — each tap broadcast-multiplies the 13 source rows.
//     Boundary patches (first 6 and last 6 rows) stay scalar.
//
// 5-vs-13 difference vs Chunk 2:
//   - Same broadcast-tap pattern; just 13 taps instead of 5.
//   - More cumulative arithmetic per output → more memory-bound at
//     1024²+ (per Chunk 2 honest finding); HIGHER speedup at 256²-512²
//     where the working set fits in L2.
//   - We do NOT pre-allocate a scratch z-buffer (unlike the pyramid
//     expand) — the kernel reads source/scratch directly.

#[archmage::attune(_v4x_t(cfg(avx512)), define(f32x16))]
fn pu_blur_h_inner(token: Token, src: &[f32], w: usize, h: usize, dst: &mut [f32]) {
    let k = PU_BLUR_KERNEL_1D;
    let k0 = f32x16::splat_t(token, k[0]);
    let k1 = f32x16::splat_t(token, k[1]);
    let k2 = f32x16::splat_t(token, k[2]);
    let k3 = f32x16::splat_t(token, k[3]);
    let k4 = f32x16::splat_t(token, k[4]);
    let k5 = f32x16::splat_t(token, k[5]);
    let k6 = f32x16::splat_t(token, k[6]);
    let k7 = f32x16::splat_t(token, k[7]);
    let k8 = f32x16::splat_t(token, k[8]);
    let k9 = f32x16::splat_t(token, k[9]);
    let k10 = f32x16::splat_t(token, k[10]);
    let k11 = f32x16::splat_t(token, k[11]);
    let k12 = f32x16::splat_t(token, k[12]);

    let lane = 16usize;
    // SIMD interior: for each `x_base`, process N=lane output cols
    // `[x_base, x_base+lane)`. The 13-tap window for output col
    // `x_base+r` (r ∈ 0..lane) reads `src[row + (x_base+r) - 6 + t]`
    // for `t in 0..13`. The last tap of the last lane reads at offset
    // `x_base + lane - 1 + 6 = x_base + lane + 5`, which is part of a
    // lane-wide load starting at `x_base + lane + 5 - (lane - 1)
    //  = x_base + 6`. So we need:
    //   - `x_base >= 6`                       (prefix tap in-bounds)
    //   - `x_base + 6 + lane <= w`            (last tap load in-bounds)
    // i.e. `x_base <= w - lane - 6`. Loop condition is
    //   `x_base + lane + 6 <= w` ⇒ `x_base < w - lane - 5`.
    let interior_lo = 6usize;
    let interior_hi_excl = w.saturating_sub(lane + 5);
    debug_assert_eq!(dst.len(), w * h);
    debug_assert_eq!(src.len(), w * h);

    crate::par::map_rows(dst, w, |y0, dst| {
        let h = dst.len() / w;
        let src_off = y0 * w;
        for y in 0..h {
            let row_off = y * w;
            // Scalar prefix: x in [0, 6).
            for x in 0..interior_lo.min(w) {
                let mut s = 0.0f32;
                for t in 0..13 {
                    let sx = reflect_idx_for_blur(x as isize + t as isize - 6, w);
                    s += k[t] * src[src_off + row_off + sx];
                }
                dst[row_off + x] = s;
            }
            // SIMD interior: stride lane across `[interior_lo, interior_hi_excl)`.
            let mut x_base = interior_lo;
            while x_base < interior_hi_excl {
                // Each tap reads N contiguous floats at row offset
                // `row_off + (x_base - 6) + t`. The 13 windows overlap
                // heavily — LLVM should keep the loads in L1.
                let load = |off: usize| -> f32x16 {
                    let base = src_off + row_off + x_base - 6 + off;
                    let arr: [f32; 16] = src[base..base + 16].try_into().unwrap();
                    f32x16::from_array_t(token, arr)
                };
                let v0 = load(0);
                let v1 = load(1);
                let v2 = load(2);
                let v3 = load(3);
                let v4x = load(4);
                let v5 = load(5);
                let v6 = load(6);
                let v7 = load(7);
                let v8 = load(8);
                let v9 = load(9);
                let v10 = load(10);
                let v11 = load(11);
                let v12 = load(12);
                let acc = v0 * k0
                    + v1 * k1
                    + v2 * k2
                    + v3 * k3
                    + v4x * k4
                    + v5 * k5
                    + v6 * k6
                    + v7 * k7
                    + v8 * k8
                    + v9 * k9
                    + v10 * k10
                    + v11 * k11
                    + v12 * k12;
                let arr = acc.to_array();
                dst[row_off + x_base..row_off + x_base + 16].copy_from_slice(&arr);
                x_base += lane;
            }
            // Scalar middle (if interior_hi - x_base < 16) + suffix.
            for x in x_base..w {
                let mut s = 0.0f32;
                for t in 0..13 {
                    let sx = reflect_idx_for_blur(x as isize + t as isize - 6, w);
                    s += k[t] * src[src_off + row_off + sx];
                }
                dst[row_off + x] = s;
            }
        }
    });
}

#[archmage::attune(_*_t, define(f32x8))]
fn pu_blur_h_inner(token: Token, src: &[f32], w: usize, h: usize, dst: &mut [f32]) {
    let k = PU_BLUR_KERNEL_1D;
    let k0 = f32x8::splat_t(token, k[0]);
    let k1 = f32x8::splat_t(token, k[1]);
    let k2 = f32x8::splat_t(token, k[2]);
    let k3 = f32x8::splat_t(token, k[3]);
    let k4 = f32x8::splat_t(token, k[4]);
    let k5 = f32x8::splat_t(token, k[5]);
    let k6 = f32x8::splat_t(token, k[6]);
    let k7 = f32x8::splat_t(token, k[7]);
    let k8 = f32x8::splat_t(token, k[8]);
    let k9 = f32x8::splat_t(token, k[9]);
    let k10 = f32x8::splat_t(token, k[10]);
    let k11 = f32x8::splat_t(token, k[11]);
    let k12 = f32x8::splat_t(token, k[12]);

    let lane = 8usize;
    let interior_lo = 6usize;
    // Loop while `x_base + lane <= w` AND
    // `x_base + (lane - 1) + 6 < w` (last tap of last lane in-bounds),
    // i.e. `x_base + lane + 5 <= w` ⇒ `x_base <= w - lane - 5`.
    let interior_hi_excl = w.saturating_sub(lane + 5);
    debug_assert_eq!(dst.len(), w * h);
    debug_assert_eq!(src.len(), w * h);

    crate::par::map_rows(dst, w, |y0, dst| {
        let h = dst.len() / w;
        let src_off = y0 * w;
        for y in 0..h {
            let row_off = y * w;
            for x in 0..interior_lo.min(w) {
                let mut s = 0.0f32;
                for t in 0..13 {
                    let sx = reflect_idx_for_blur(x as isize + t as isize - 6, w);
                    s += k[t] * src[src_off + row_off + sx];
                }
                dst[row_off + x] = s;
            }
            let mut x_base = interior_lo;
            while x_base < interior_hi_excl {
                let load = |off: usize| -> f32x8 {
                    let base = src_off + row_off + x_base - 6 + off;
                    let arr: [f32; 8] = src[base..base + 8].try_into().unwrap();
                    f32x8::from_array_t(token, arr)
                };
                let v0 = load(0);
                let v1 = load(1);
                let v2 = load(2);
                let v3 = load(3);
                let v4x = load(4);
                let v5 = load(5);
                let v6 = load(6);
                let v7 = load(7);
                let v8 = load(8);
                let v9 = load(9);
                let v10 = load(10);
                let v11 = load(11);
                let v12 = load(12);
                let acc = v0 * k0
                    + v1 * k1
                    + v2 * k2
                    + v3 * k3
                    + v4x * k4
                    + v5 * k5
                    + v6 * k6
                    + v7 * k7
                    + v8 * k8
                    + v9 * k9
                    + v10 * k10
                    + v11 * k11
                    + v12 * k12;
                let arr = acc.to_array();
                dst[row_off + x_base..row_off + x_base + 8].copy_from_slice(&arr);
                x_base += lane;
            }
            for x in x_base..w {
                let mut s = 0.0f32;
                for t in 0..13 {
                    let sx = reflect_idx_for_blur(x as isize + t as isize - 6, w);
                    s += k[t] * src[src_off + row_off + sx];
                }
                dst[row_off + x] = s;
            }
        }
    });
}

/// Horizontal pass of the σ=3 13-tap Gaussian blur. Writes the `w × h`
/// scratch buffer with the per-row 13-tap convolution + reflect
/// padding. Bit-equivalent to `gaussian_blur_sigma3`'s horizontal
/// pass except for SIMD FMA-grouping in the interior (well below
/// 1e-5 abs).
pub(crate) fn pu_blur_horizontal_pass(src: &[f32], w: usize, h: usize, h_pass: &mut [f32]) {
    debug_assert_eq!(src.len(), w * h);
    debug_assert_eq!(h_pass.len(), w * h);
    archmage::attuned!(
        pu_blur_h_inner(src, w, h, h_pass),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
}

#[archmage::attune(_v4x_t(cfg(avx512)), define(f32x16))]
fn pu_blur_v_inner(token: Token, h_pass: &[f32], w: usize, h: usize, dst: &mut [f32], scale: f32) {
    let k = PU_BLUR_KERNEL_1D;
    let k0 = f32x16::splat_t(token, k[0]);
    let k1 = f32x16::splat_t(token, k[1]);
    let k2 = f32x16::splat_t(token, k[2]);
    let k3 = f32x16::splat_t(token, k[3]);
    let k4 = f32x16::splat_t(token, k[4]);
    let k5 = f32x16::splat_t(token, k[5]);
    let k6 = f32x16::splat_t(token, k[6]);
    let k7 = f32x16::splat_t(token, k[7]);
    let k8 = f32x16::splat_t(token, k[8]);
    let k9 = f32x16::splat_t(token, k[9]);
    let k10 = f32x16::splat_t(token, k[10]);
    let k11 = f32x16::splat_t(token, k[11]);
    let k12 = f32x16::splat_t(token, k[12]);
    let vs = f32x16::splat_t(token, scale);

    let lane = 16usize;
    let n_groups = w / lane;
    let interior_lo = 6usize;
    let interior_hi = h.saturating_sub(6); // y in [interior_lo, interior_hi) is SIMD interior
    debug_assert_eq!(dst.len(), w * h);
    debug_assert_eq!(h_pass.len(), w * h);

    // Row bands: `dst` bands are disjoint writes; `h_pass` reads stay
    // absolute (13-tap halo across band edges, reflect at the borders).
    crate::par::map_rows(dst, w, |y0, dst| {
        let h_band = dst.len() / w;
        // Boundary rows: y in [0, 6) ∪ [h-6, h) — scalar.
        for yl in 0..interior_lo.min(h).saturating_sub(y0).min(h_band) {
            let y = y0 + yl;
            let row_off = yl * w;
            for x in 0..w {
                let mut s = 0.0f32;
                for t in 0..13 {
                    let sy = reflect_idx_for_blur(y as isize + t as isize - 6, h);
                    s += k[t] * h_pass[sy * w + x];
                }
                dst[row_off + x] = s * scale;
            }
        }

        // Interior rows: SIMD across columns within each row.
        for yl in interior_lo.saturating_sub(y0)..interior_hi.saturating_sub(y0).min(h_band) {
            let y = y0 + yl;
            let row_off = yl * w;
            // SIMD lane groups across w.
            for cg in 0..n_groups {
                let x_base = cg * lane;
                // Each tap reads N contiguous floats at
                // `(y - 6 + t) * w + x_base` (no reflection in interior).
                let load = |t: usize| -> f32x16 {
                    let off = (y - 6 + t) * w + x_base;
                    let arr: [f32; 16] = h_pass[off..off + 16].try_into().unwrap();
                    f32x16::from_array_t(token, arr)
                };
                let v0 = load(0);
                let v1 = load(1);
                let v2 = load(2);
                let v3 = load(3);
                let v4x = load(4);
                let v5 = load(5);
                let v6 = load(6);
                let v7 = load(7);
                let v8 = load(8);
                let v9 = load(9);
                let v10 = load(10);
                let v11 = load(11);
                let v12 = load(12);
                let acc = v0 * k0
                    + v1 * k1
                    + v2 * k2
                    + v3 * k3
                    + v4x * k4
                    + v5 * k5
                    + v6 * k6
                    + v7 * k7
                    + v8 * k8
                    + v9 * k9
                    + v10 * k10
                    + v11 * k11
                    + v12 * k12;
                let arr = (acc * vs).to_array();
                dst[row_off + x_base..row_off + x_base + 16].copy_from_slice(&arr);
            }
            // Scalar tail columns (< lane). No reflect needed (y is interior).
            for x in n_groups * lane..w {
                let mut s = 0.0f32;
                for t in 0..13 {
                    let sy = y + t - 6; // safe: y >= 6
                    s += k[t] * h_pass[sy * w + x];
                }
                dst[row_off + x] = s * scale;
            }
        }

        // Boundary rows: y in [h-6, h) — scalar.
        for yl in interior_hi.max(interior_lo).saturating_sub(y0)..h.saturating_sub(y0).min(h_band)
        {
            let y = y0 + yl;
            let row_off = yl * w;
            for x in 0..w {
                let mut s = 0.0f32;
                for t in 0..13 {
                    let sy = reflect_idx_for_blur(y as isize + t as isize - 6, h);
                    s += k[t] * h_pass[sy * w + x];
                }
                dst[row_off + x] = s * scale;
            }
        }
    });
}

#[archmage::attune(_*_t, define(f32x8))]
fn pu_blur_v_inner(token: Token, h_pass: &[f32], w: usize, h: usize, dst: &mut [f32], scale: f32) {
    let k = PU_BLUR_KERNEL_1D;
    let k0 = f32x8::splat_t(token, k[0]);
    let k1 = f32x8::splat_t(token, k[1]);
    let k2 = f32x8::splat_t(token, k[2]);
    let k3 = f32x8::splat_t(token, k[3]);
    let k4 = f32x8::splat_t(token, k[4]);
    let k5 = f32x8::splat_t(token, k[5]);
    let k6 = f32x8::splat_t(token, k[6]);
    let k7 = f32x8::splat_t(token, k[7]);
    let k8 = f32x8::splat_t(token, k[8]);
    let k9 = f32x8::splat_t(token, k[9]);
    let k10 = f32x8::splat_t(token, k[10]);
    let k11 = f32x8::splat_t(token, k[11]);
    let k12 = f32x8::splat_t(token, k[12]);
    let vs = f32x8::splat_t(token, scale);

    let lane = 8usize;
    let n_groups = w / lane;
    let interior_lo = 6usize;
    let interior_hi = h.saturating_sub(6);
    debug_assert_eq!(dst.len(), w * h);
    debug_assert_eq!(h_pass.len(), w * h);

    // Row bands: `dst` bands are disjoint writes; `h_pass` reads stay
    // absolute (13-tap halo across band edges, reflect at the borders).
    crate::par::map_rows(dst, w, |y0, dst| {
        let h_band = dst.len() / w;
        for yl in 0..interior_lo.min(h).saturating_sub(y0).min(h_band) {
            let y = y0 + yl;
            let row_off = yl * w;
            for x in 0..w {
                let mut s = 0.0f32;
                for t in 0..13 {
                    let sy = reflect_idx_for_blur(y as isize + t as isize - 6, h);
                    s += k[t] * h_pass[sy * w + x];
                }
                dst[row_off + x] = s * scale;
            }
        }

        for yl in interior_lo.saturating_sub(y0)..interior_hi.saturating_sub(y0).min(h_band) {
            let y = y0 + yl;
            let row_off = yl * w;
            for cg in 0..n_groups {
                let x_base = cg * lane;
                let load = |t: usize| -> f32x8 {
                    let off = (y - 6 + t) * w + x_base;
                    let arr: [f32; 8] = h_pass[off..off + 8].try_into().unwrap();
                    f32x8::from_array_t(token, arr)
                };
                let v0 = load(0);
                let v1 = load(1);
                let v2 = load(2);
                let v3 = load(3);
                let v4x = load(4);
                let v5 = load(5);
                let v6 = load(6);
                let v7 = load(7);
                let v8 = load(8);
                let v9 = load(9);
                let v10 = load(10);
                let v11 = load(11);
                let v12 = load(12);
                let acc = v0 * k0
                    + v1 * k1
                    + v2 * k2
                    + v3 * k3
                    + v4x * k4
                    + v5 * k5
                    + v6 * k6
                    + v7 * k7
                    + v8 * k8
                    + v9 * k9
                    + v10 * k10
                    + v11 * k11
                    + v12 * k12;
                let arr = (acc * vs).to_array();
                dst[row_off + x_base..row_off + x_base + 8].copy_from_slice(&arr);
            }
            for x in n_groups * lane..w {
                let mut s = 0.0f32;
                for t in 0..13 {
                    let sy = y + t - 6;
                    s += k[t] * h_pass[sy * w + x];
                }
                dst[row_off + x] = s * scale;
            }
        }

        for yl in interior_hi.max(interior_lo).saturating_sub(y0)..h.saturating_sub(y0).min(h_band)
        {
            let y = y0 + yl;
            let row_off = yl * w;
            for x in 0..w {
                let mut s = 0.0f32;
                for t in 0..13 {
                    let sy = reflect_idx_for_blur(y as isize + t as isize - 6, h);
                    s += k[t] * h_pass[sy * w + x];
                }
                dst[row_off + x] = s * scale;
            }
        }
    });
}

/// Vertical pass of the σ=3 13-tap Gaussian blur. Reads `h_pass`
/// (output of `pu_blur_horizontal_pass`) and writes the final blur
/// into `dst`.
pub(crate) fn pu_blur_vertical_pass(h_pass: &[f32], w: usize, h: usize, dst: &mut [f32]) {
    pu_blur_vertical_pass_scaled(h_pass, w, h, dst, 1.0);
}

/// [`pu_blur_vertical_pass`] with a post-accumulate output scale —
/// `dst[i] = (Σ_t k[t]·h_pass[..]) * scale`. `scale = 1.0` is
/// bit-identical to the unscaled pass (x·1.0 == x for every finite
/// f32, ±0 and NaN included). Used to fold the phase-uncertainty
/// `mask_c_lin` multiply into the blur's final store, removing one
/// full-plane read+write per channel per band.
pub(crate) fn pu_blur_vertical_pass_scaled(
    h_pass: &[f32],
    w: usize,
    h: usize,
    dst: &mut [f32],
    scale: f32,
) {
    debug_assert_eq!(h_pass.len(), w * h);
    debug_assert_eq!(dst.len(), w * h);
    archmage::attuned!(
        pu_blur_v_inner(h_pass, w, h, dst, scale),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
}

/// Full σ=3 13-tap separable Gaussian blur with caller-owned scratch.
///
/// Replacement for `crate::kernels::masking::gaussian_blur_sigma3`
/// that avoids an internal allocation. `h_pass` is the horizontal-pass
/// scratch (resized to `w*h` internally); `dst` receives the final
/// blurred output. Same reflect-padding semantics as the upstream
/// scalar reference; SIMD-vectorized interior with scalar boundary
/// patches.
///
/// Caller invariants: `src.len() == w * h`, `w >= 1`, `h >= 1`.
pub(crate) fn gaussian_blur_sigma3_simd(
    src: &[f32],
    w: usize,
    h: usize,
    h_pass: &mut Vec<f32>,
    dst: &mut Vec<f32>,
) {
    debug_assert_eq!(src.len(), w * h);
    let n = w * h;
    // Grow-only: h_pass/dst are shared scratch — a truncate+regrow
    // would memset every call, and both passes overwrite fully anyway.
    if h_pass.len() < n {
        h_pass.resize(n, 0.0);
    }
    if dst.len() < n {
        dst.resize(n, 0.0);
    }
    pu_blur_horizontal_pass(src, w, h, &mut h_pass[..n]);
    pu_blur_vertical_pass(&h_pass[..n], w, h, &mut dst[..n]);
}

/// [`gaussian_blur_sigma3_simd`] with the output scale folded into
/// the vertical pass's store — `dst[i] = blur(src)[i] * scale`,
/// bit-identical to a separate `vscale` pass.
pub(crate) fn gaussian_blur_sigma3_scaled_simd(
    src: &[f32],
    w: usize,
    h: usize,
    h_pass: &mut Vec<f32>,
    dst: &mut Vec<f32>,
    scale: f32,
) {
    debug_assert_eq!(src.len(), w * h);
    let n = w * h;
    if h_pass.len() < n {
        h_pass.resize(n, 0.0);
    }
    if dst.len() < n {
        dst.resize(n, 0.0);
    }
    pu_blur_horizontal_pass(src, w, h, &mut h_pass[..n]);
    pu_blur_vertical_pass_scaled(&h_pass[..n], w, h, &mut dst[..n], scale);
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn rng_seq(seed: u32, n: usize) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1103515245).wrapping_add(12345);
                (s >> 16) as f32 / 65536.0
            })
            .collect()
    }

    /// Scalar reference: line-by-line port of the pyramid vertical pass.
    fn reduce_v_scalar_ref(src: &[f32], sw: usize, sh: usize, dh: usize) -> Vec<f32> {
        let k = GAUSS5;
        let mut out = alloc::vec![0.0_f32; sw * dh];
        for dy in 0..dh {
            let cy = 2 * dy as isize;
            for x in 0..sw {
                let read = |off: isize| -> f32 {
                    let r = cy + off;
                    if r < 0 || r >= sh as isize {
                        0.0
                    } else {
                        src[r as usize * sw + x]
                    }
                };
                out[dy * sw + x] = k[0] * read(-2)
                    + k[1] * read(-1)
                    + k[2] * read(0)
                    + k[3] * read(1)
                    + k[4] * read(2);
            }
        }
        out
    }

    fn reduce_h_scalar_ref(vscratch: &[f32], sw: usize, dw: usize, dh: usize) -> Vec<f32> {
        let k = GAUSS5;
        let mut out = alloc::vec![0.0_f32; dw * dh];
        for dy in 0..dh {
            for dx in 0..dw {
                let cx = 2 * dx as isize;
                let read = |off: isize| -> f32 {
                    let c = cx + off;
                    if c < 0 || c >= sw as isize {
                        0.0
                    } else {
                        vscratch[dy * sw + c as usize]
                    }
                };
                out[dy * dw + dx] = k[0] * read(-2)
                    + k[1] * read(-1)
                    + k[2] * read(0)
                    + k[3] * read(1)
                    + k[4] * read(2);
            }
        }
        out
    }

    fn expand_v_scalar_ref(src: &[f32], sw: usize, sh: usize, out_h: usize) -> Vec<f32> {
        let k = GAUSS5;
        let mut out = alloc::vec![0.0_f32; sw * out_h];
        let z_len_v = out_h + 4;
        let odd_h = out_h & 1;
        let back_idx_v = out_h + 2 + odd_h;
        let mut z_v = alloc::vec![0.0_f32; z_len_v];
        for x in 0..sw {
            for v in z_v.iter_mut() {
                *v = 0.0;
            }
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
                out[y * sw + x] = 2.0 * sum;
            }
        }
        out
    }

    fn expand_h_scalar_ref(vscratch: &[f32], sw: usize, out_w: usize, out_h: usize) -> Vec<f32> {
        let k = GAUSS5;
        let mut out = alloc::vec![0.0_f32; out_w * out_h];
        let z_len_h = out_w + 4;
        let odd_w = out_w & 1;
        let back_idx_h = out_w + 2 + odd_w;
        let mut z_h = alloc::vec![0.0_f32; z_len_h];
        for y in 0..out_h {
            for v in z_h.iter_mut() {
                *v = 0.0;
            }
            let row_off = y * sw;
            z_h[0] = vscratch[row_off];
            for kx in 0..sw {
                z_h[2 + 2 * kx] = vscratch[row_off + kx];
            }
            z_h[back_idx_h] = vscratch[row_off + sw - 1];
            for x in 0..out_w {
                let sum = k[0] * z_h[x]
                    + k[1] * z_h[x + 1]
                    + k[2] * z_h[x + 2]
                    + k[3] * z_h[x + 3]
                    + k[4] * z_h[x + 4];
                out[y * out_w + x] = 2.0 * sum;
            }
        }
        out
    }

    #[test]
    fn reduce_v_simd_matches_scalar_random() {
        let cases: &[(usize, usize)] = &[
            (8, 8),
            (16, 16),
            (17, 19),
            (24, 24),
            (32, 32),
            (33, 35),
            (40, 24),
            (64, 64),
            (73, 91),
            (128, 128),
            (256, 256),
        ];
        for &(sw, sh) in cases {
            let dh = sh.div_ceil(2);
            let src = rng_seq(0xdeadbeef ^ ((sw as u32) << 16) ^ (sh as u32), sw * sh);
            let want = reduce_v_scalar_ref(&src, sw, sh, dh);
            let mut got = alloc::vec![0.0_f32; sw * dh];
            reduce_vertical_pass(&src, sw, sh, dh, &mut got);
            for i in 0..want.len() {
                assert!(
                    (want[i] - got[i]).abs() < 1e-5,
                    "case {sw}x{sh} idx {i}: want={}, got={}",
                    want[i],
                    got[i]
                );
            }
        }
    }

    #[test]
    fn reduce_h_simd_matches_scalar_random() {
        let cases: &[(usize, usize)] = &[
            (8, 8),
            (16, 16),
            (17, 19),
            (24, 24),
            (32, 32),
            (33, 35),
            (40, 24),
            (64, 64),
            (73, 91),
            (128, 128),
            (256, 256),
        ];
        for &(sw, sh) in cases {
            let dh = sh.div_ceil(2);
            let dw = sw.div_ceil(2);
            let vs = rng_seq(0xfeedf00d ^ ((sw as u32) << 16) ^ (sh as u32), sw * dh);
            let want = reduce_h_scalar_ref(&vs, sw, dw, dh);
            let mut got = alloc::vec![0.0_f32; dw * dh];
            reduce_horizontal_pass(&vs, sw, dw, dh, &mut got);
            for i in 0..want.len() {
                assert!(
                    (want[i] - got[i]).abs() < 1e-5,
                    "case {sw}x{sh} idx {i}: want={}, got={}",
                    want[i],
                    got[i]
                );
            }
        }
    }

    #[test]
    fn expand_v_simd_matches_scalar_random() {
        let cases: &[(usize, usize, usize, usize)] = &[
            (4, 4, 8, 8),
            (4, 4, 7, 7),
            (8, 6, 16, 12),
            (8, 6, 15, 11),
            (16, 12, 32, 24),
            (24, 16, 48, 32),
            (33, 17, 65, 33),
            (64, 32, 128, 64),
        ];
        for &(sw, sh, _ow, oh) in cases {
            let src = rng_seq(0xabcd1234 ^ ((sw as u32) << 16) ^ (sh as u32), sw * sh);
            let want = expand_v_scalar_ref(&src, sw, sh, oh);
            let mut got = alloc::vec![0.0_f32; sw * oh];
            expand_vertical_pass(&src, sw, sh, oh, &mut got);
            for i in 0..want.len() {
                assert!(
                    (want[i] - got[i]).abs() < 1e-5,
                    "case {sw}x{sh}/{oh} idx {i}: want={}, got={}",
                    want[i],
                    got[i]
                );
            }
        }
    }

    #[test]
    fn expand_h_simd_matches_scalar_random() {
        let cases: &[(usize, usize, usize)] = &[
            (4, 8, 8),
            (4, 7, 7),
            (8, 16, 12),
            (8, 15, 11),
            (16, 32, 24),
            (24, 48, 32),
            (33, 65, 33),
            (64, 128, 64),
        ];
        for &(sw, ow, oh) in cases {
            let vs = rng_seq(0xcafebabe ^ ((sw as u32) << 16) ^ (ow as u32), sw * oh);
            let want = expand_h_scalar_ref(&vs, sw, ow, oh);
            let mut got = alloc::vec![0.0_f32; ow * oh];
            expand_horizontal_pass(&vs, sw, ow, oh, &mut got);
            for i in 0..want.len() {
                assert!(
                    (want[i] - got[i]).abs() < 1e-5,
                    "case {sw}/{ow}x{oh} idx {i}: want={}, got={}",
                    want[i],
                    got[i]
                );
            }
        }
    }

    #[test]
    fn reduce_v_dc_preservation() {
        // Uniform input → uniform interior output (DC=1.0).
        let sw: usize = 64;
        let sh: usize = 64;
        let dh = sh.div_ceil(2);
        let src = alloc::vec![1.0_f32; sw * sh];
        let mut got = alloc::vec![0.0_f32; sw * dh];
        reduce_vertical_pass(&src, sw, sh, dh, &mut got);
        let k_sum: f32 = GAUSS5.iter().sum();
        for dy in 2..dh - 2 {
            for x in 0..sw {
                assert!(
                    (got[dy * sw + x] - k_sum).abs() < 1e-6,
                    "dy={dy} x={x}: {} vs {k_sum}",
                    got[dy * sw + x]
                );
            }
        }
    }

    // ========================================================================
    // 13-tap σ=3 Gaussian blur (Chunk 1) parity tests
    // ========================================================================
    //
    // Compare against the scalar `gaussian_blur_sigma3` reference in
    // cvvdp-gpu — bit-equivalent except for FMA-grouping in the SIMD
    // interior (well below the 1e-5 abs tolerance).

    #[test]
    fn pu_blur_simd_matches_upstream_scalar() {
        use crate::kernels::masking::gaussian_blur_sigma3;
        // Sizes spanning the no-blur PU_PADSIZE = 6 cutoff (caller guards
        // against w ≤ 6 || h ≤ 6) AND a mix of SIMD-interior + tail
        // cases: 16 == 8-lane × 2, 17 forces one scalar tail col, 32 ==
        // 16-lane × 2 (v4x clean), etc.
        let cases: &[(usize, usize)] = &[
            (7, 7),
            (8, 8),
            (12, 12),
            (16, 16),
            (17, 19),
            (24, 24),
            (32, 32),
            (33, 35),
            (64, 64),
            (100, 100),
            (128, 128),
            (256, 256),
        ];
        for &(w, h) in cases {
            let src = rng_seq(0x12345678 ^ ((w as u32) << 16) ^ (h as u32), w * h);
            let want = gaussian_blur_sigma3(&src, w, h);
            let mut h_pass: Vec<f32> = Vec::new();
            let mut got: Vec<f32> = Vec::new();
            gaussian_blur_sigma3_simd(&src, w, h, &mut h_pass, &mut got);
            for i in 0..want.len() {
                assert!(
                    (want[i] - got[i]).abs() < 1e-5,
                    "case {w}x{h} idx {i}: want={}, got={} (delta={})",
                    want[i],
                    got[i],
                    (want[i] - got[i]).abs()
                );
            }
        }
    }

    #[test]
    fn pu_blur_simd_dc_preservation() {
        // Uniform input → uniform output (kernel sums to 1).
        let cases: &[(usize, usize)] = &[(16, 16), (33, 33), (64, 100), (128, 128)];
        for &(w, h) in cases {
            let src = alloc::vec![3.5_f32; w * h];
            let mut h_pass: Vec<f32> = Vec::new();
            let mut got: Vec<f32> = Vec::new();
            gaussian_blur_sigma3_simd(&src, w, h, &mut h_pass, &mut got);
            for &v in got.iter() {
                assert!(
                    (v - 3.5).abs() < 1e-5,
                    "DC not preserved at {w}x{h}: {v} vs 3.5"
                );
            }
        }
    }

    #[test]
    fn pu_blur_simd_horizontal_only_parity() {
        use crate::kernels::masking::PU_BLUR_KERNEL_1D;
        // Validate the horizontal pass in isolation against a
        // line-by-line scalar reference. Catches lane-boundary issues
        // independently of the vertical pass.
        let k = PU_BLUR_KERNEL_1D;
        let cases: &[(usize, usize)] = &[(16, 4), (33, 4), (64, 4), (128, 4)];
        for &(w, h) in cases {
            let src = rng_seq(0xabad1dea ^ ((w as u32) << 16) ^ (h as u32), w * h);
            let mut want = alloc::vec![0.0_f32; w * h];
            for y in 0..h {
                for x in 0..w {
                    let mut s = 0.0f32;
                    for t in 0..13 {
                        let sx = reflect_idx_for_blur(x as isize + t as isize - 6, w);
                        s += k[t] * src[y * w + sx];
                    }
                    want[y * w + x] = s;
                }
            }
            let mut got = alloc::vec![0.0_f32; w * h];
            pu_blur_horizontal_pass(&src, w, h, &mut got);
            for i in 0..want.len() {
                assert!(
                    (want[i] - got[i]).abs() < 1e-5,
                    "h-pass {w}x{h} idx {i}: want={}, got={}",
                    want[i],
                    got[i]
                );
            }
        }
    }

    #[test]
    fn pu_blur_simd_vertical_only_parity() {
        use crate::kernels::masking::PU_BLUR_KERNEL_1D;
        // Validate the vertical pass in isolation.
        let k = PU_BLUR_KERNEL_1D;
        let cases: &[(usize, usize)] = &[(16, 16), (33, 33), (64, 64), (128, 100)];
        for &(w, h) in cases {
            let scratch = rng_seq(0xbeefdead ^ ((w as u32) << 16) ^ (h as u32), w * h);
            let mut want = alloc::vec![0.0_f32; w * h];
            for y in 0..h {
                for x in 0..w {
                    let mut s = 0.0f32;
                    for t in 0..13 {
                        let sy = reflect_idx_for_blur(y as isize + t as isize - 6, h);
                        s += k[t] * scratch[sy * w + x];
                    }
                    want[y * w + x] = s;
                }
            }
            let mut got = alloc::vec![0.0_f32; w * h];
            pu_blur_vertical_pass(&scratch, w, h, &mut got);
            for i in 0..want.len() {
                assert!(
                    (want[i] - got[i]).abs() < 1e-5,
                    "v-pass {w}x{h} idx {i}: want={}, got={}",
                    want[i],
                    got[i]
                );
            }
        }
    }

    #[test]
    fn pu_blur_simd_reuses_scratch_safely() {
        // Calling twice with the same scratch vec must produce
        // identical output (no leftover state contamination).
        use crate::kernels::masking::gaussian_blur_sigma3;
        let w = 64;
        let h = 64;
        let src1 = rng_seq(0xa1a1a1a1, w * h);
        let src2 = rng_seq(0xb2b2b2b2, w * h);
        let want2 = gaussian_blur_sigma3(&src2, w, h);
        let mut h_pass: Vec<f32> = Vec::new();
        let mut dst: Vec<f32> = Vec::new();
        gaussian_blur_sigma3_simd(&src1, w, h, &mut h_pass, &mut dst);
        gaussian_blur_sigma3_simd(&src2, w, h, &mut h_pass, &mut dst);
        for i in 0..want2.len() {
            assert!(
                (want2[i] - dst[i]).abs() < 1e-5,
                "reuse {i}: want={}, got={}",
                want2[i],
                dst[i]
            );
        }
    }
}
