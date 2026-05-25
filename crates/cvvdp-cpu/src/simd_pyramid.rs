//! SIMD-vectorized 5-tap pyramid reduce / expand kernels (Chunk 2 of
//! the SIMD optimization plan).
//!
//! Targets the inner column/row sweeps of [`gausspyr_reduce`] and
//! [`gausspyr_expand`] (in [`super::pyramid`]) which together account
//! for ~24 % of cvvdp-cpu wall time at 1024² (per the
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
// The per-tier #[magetypes] expansion makes some loops appear as
// "manual memcpy" candidates to clippy, but the loop body inside the
// expansion may include the SIMD-load pattern's bookkeeping. Suppress
// at module scope rather than per-site (cleaner).
#![allow(clippy::manual_memcpy)]

use alloc::vec::Vec;

use cvvdp_gpu::kernels::masking::PU_BLUR_KERNEL_1D;
use cvvdp_gpu::kernels::pyramid::GAUSS5;

/// Reflect `i` into `[0, n)` for the 13-tap PU blur. Matches
/// torchvision's `F.pad(..., mode='reflect')` behaviour exactly —
/// bit-equivalent to `cvvdp_gpu::kernels::masking::reflect_idx_for_blur`
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
// Per scalar reference (cvvdp_gpu::kernels::pyramid::gausspyr_reduce_scalar):
//   for dy in 0..dh:
//     cy = 2 * dy
//     for x in 0..sw:
//       vscratch[dy*sw + x] = Σᵢ GAUSS5[i] · R(cy + i - 2, x)
//   R(r, x) = src[r*sw + x] if 0 ≤ r < sh else 0.
//
// SIMD across x: per dy, the 5 source-row offsets are uniform; we sweep
// f32x{N} chunks across columns, broadcast-multiplying by the loop-
// invariant scalar GAUSS5 taps.

#[archmage::magetypes(define(f32x16), +v4, +v4x, -v3, -neon, -wasm128, -scalar)]
fn reduce_v_inner(
    token: Token,
    src: &[f32],
    sw: usize,
    sh: usize,
    dh: usize,
    vscratch: &mut [f32],
) {
    let k = GAUSS5;
    let k0 = f32x16::splat(token, k[0]);
    let k1 = f32x16::splat(token, k[1]);
    let k2 = f32x16::splat(token, k[2]);
    let k3 = f32x16::splat(token, k[3]);
    let k4 = f32x16::splat(token, k[4]);

    let n_groups = sw / 16;
    for dy in 0..dh {
        let cy = 2 * dy as isize;
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
        let out_base = dy * sw;

        for cg in 0..n_groups {
            let col = cg * 16;
            let load = |row: Option<usize>| -> f32x16 {
                match row {
                    Some(base) => {
                        let arr: [f32; 16] = src[base + col..base + col + 16].try_into().unwrap();
                        f32x16::from_array(token, arr)
                    }
                    None => f32x16::zero(token),
                }
            };
            let v0 = load(r_m2);
            let v1 = load(r_m1);
            let v2 = load(r_0);
            let v3v = load(r_p1);
            let v4 = load(r_p2);
            let acc = v0 * k0 + v1 * k1 + v2 * k2 + v3v * k3 + v4 * k4;
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
}

#[archmage::magetypes(define(f32x8), v3, neon, wasm128, scalar)]
fn reduce_v_inner(
    token: Token,
    src: &[f32],
    sw: usize,
    sh: usize,
    dh: usize,
    vscratch: &mut [f32],
) {
    let k = GAUSS5;
    let k0 = f32x8::splat(token, k[0]);
    let k1 = f32x8::splat(token, k[1]);
    let k2 = f32x8::splat(token, k[2]);
    let k3 = f32x8::splat(token, k[3]);
    let k4 = f32x8::splat(token, k[4]);

    let n_groups = sw / 8;
    for dy in 0..dh {
        let cy = 2 * dy as isize;
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
        let out_base = dy * sw;

        for cg in 0..n_groups {
            let col = cg * 8;
            let load = |row: Option<usize>| -> f32x8 {
                match row {
                    Some(base) => {
                        let arr: [f32; 8] = src[base + col..base + col + 8].try_into().unwrap();
                        f32x8::from_array(token, arr)
                    }
                    None => f32x8::zero(token),
                }
            };
            let v0 = load(r_m2);
            let v1 = load(r_m1);
            let v2 = load(r_0);
            let v3v = load(r_p1);
            let v4 = load(r_p2);
            let acc = v0 * k0 + v1 * k1 + v2 * k2 + v3v * k3 + v4 * k4;
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
    archmage::incant!(
        reduce_v_inner(src, sw, sh, dh, vscratch),
        [v4x, v4, v3, neon, wasm128, scalar]
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

#[archmage::magetypes(define(f32x16), +v4, +v4x, -v3, -neon, -wasm128, -scalar)]
fn reduce_h_inner(
    token: Token,
    vscratch: &[f32],
    sw: usize,
    dw: usize,
    dh: usize,
    dst: &mut [f32],
) {
    let k = GAUSS5;
    let k0 = f32x16::splat(token, k[0]);
    let k1 = f32x16::splat(token, k[1]);
    let k2 = f32x16::splat(token, k[2]);
    let k3 = f32x16::splat(token, k[3]);
    let k4 = f32x16::splat(token, k[4]);

    let n_row_groups = dh / 16;
    for dx in 0..dw {
        let cx = 2 * dx as isize;
        let col_at = |off: isize| -> Option<usize> {
            let c = cx + off;
            if c < 0 || c >= sw as isize {
                None
            } else {
                Some(c as usize)
            }
        };
        let c_m2 = col_at(-2);
        let c_m1 = col_at(-1);
        let c_0 = col_at(0);
        let c_p1 = col_at(1);
        let c_p2 = col_at(2);

        for rg in 0..n_row_groups {
            let row_base = rg * 16;
            let mut arr_m2 = [0.0f32; 16];
            let mut arr_m1 = [0.0f32; 16];
            let mut arr_0 = [0.0f32; 16];
            let mut arr_p1 = [0.0f32; 16];
            let mut arr_p2 = [0.0f32; 16];
            for r in 0..16 {
                let dy = row_base + r;
                let row_off = dy * sw;
                arr_m2[r] = c_m2.map(|c| vscratch[row_off + c]).unwrap_or(0.0);
                arr_m1[r] = c_m1.map(|c| vscratch[row_off + c]).unwrap_or(0.0);
                arr_0[r] = c_0.map(|c| vscratch[row_off + c]).unwrap_or(0.0);
                arr_p1[r] = c_p1.map(|c| vscratch[row_off + c]).unwrap_or(0.0);
                arr_p2[r] = c_p2.map(|c| vscratch[row_off + c]).unwrap_or(0.0);
            }
            let v0 = f32x16::from_array(token, arr_m2);
            let v1 = f32x16::from_array(token, arr_m1);
            let v2 = f32x16::from_array(token, arr_0);
            let v3v = f32x16::from_array(token, arr_p1);
            let v4 = f32x16::from_array(token, arr_p2);
            let acc = v0 * k0 + v1 * k1 + v2 * k2 + v3v * k3 + v4 * k4;
            let res = acc.to_array();
            for r in 0..16 {
                dst[(row_base + r) * dw + dx] = res[r];
            }
        }

        // Scalar tail rows (< 16 rows).
        for dy in n_row_groups * 16..dh {
            let row_off = dy * sw;
            let read =
                |c: Option<usize>| -> f32 { c.map(|c| vscratch[row_off + c]).unwrap_or(0.0) };
            dst[dy * dw + dx] = k[0] * read(c_m2)
                + k[1] * read(c_m1)
                + k[2] * read(c_0)
                + k[3] * read(c_p1)
                + k[4] * read(c_p2);
        }
    }
}

#[archmage::magetypes(define(f32x8), v3, neon, wasm128, scalar)]
fn reduce_h_inner(
    token: Token,
    vscratch: &[f32],
    sw: usize,
    dw: usize,
    dh: usize,
    dst: &mut [f32],
) {
    let k = GAUSS5;
    let k0 = f32x8::splat(token, k[0]);
    let k1 = f32x8::splat(token, k[1]);
    let k2 = f32x8::splat(token, k[2]);
    let k3 = f32x8::splat(token, k[3]);
    let k4 = f32x8::splat(token, k[4]);

    let n_row_groups = dh / 8;
    for dx in 0..dw {
        let cx = 2 * dx as isize;
        let col_at = |off: isize| -> Option<usize> {
            let c = cx + off;
            if c < 0 || c >= sw as isize {
                None
            } else {
                Some(c as usize)
            }
        };
        let c_m2 = col_at(-2);
        let c_m1 = col_at(-1);
        let c_0 = col_at(0);
        let c_p1 = col_at(1);
        let c_p2 = col_at(2);

        for rg in 0..n_row_groups {
            let row_base = rg * 8;
            let mut arr_m2 = [0.0f32; 8];
            let mut arr_m1 = [0.0f32; 8];
            let mut arr_0 = [0.0f32; 8];
            let mut arr_p1 = [0.0f32; 8];
            let mut arr_p2 = [0.0f32; 8];
            for r in 0..8 {
                let dy = row_base + r;
                let row_off = dy * sw;
                arr_m2[r] = c_m2.map(|c| vscratch[row_off + c]).unwrap_or(0.0);
                arr_m1[r] = c_m1.map(|c| vscratch[row_off + c]).unwrap_or(0.0);
                arr_0[r] = c_0.map(|c| vscratch[row_off + c]).unwrap_or(0.0);
                arr_p1[r] = c_p1.map(|c| vscratch[row_off + c]).unwrap_or(0.0);
                arr_p2[r] = c_p2.map(|c| vscratch[row_off + c]).unwrap_or(0.0);
            }
            let v0 = f32x8::from_array(token, arr_m2);
            let v1 = f32x8::from_array(token, arr_m1);
            let v2 = f32x8::from_array(token, arr_0);
            let v3v = f32x8::from_array(token, arr_p1);
            let v4 = f32x8::from_array(token, arr_p2);
            let acc = v0 * k0 + v1 * k1 + v2 * k2 + v3v * k3 + v4 * k4;
            let res = acc.to_array();
            for r in 0..8 {
                dst[(row_base + r) * dw + dx] = res[r];
            }
        }

        for dy in n_row_groups * 8..dh {
            let row_off = dy * sw;
            let read =
                |c: Option<usize>| -> f32 { c.map(|c| vscratch[row_off + c]).unwrap_or(0.0) };
            dst[dy * dw + dx] = k[0] * read(c_m2)
                + k[1] * read(c_m1)
                + k[2] * read(c_0)
                + k[3] * read(c_p1)
                + k[4] * read(c_p2);
        }
    }
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
    archmage::incant!(
        reduce_h_inner(vscratch, sw, dw, dh, dst),
        [v4x, v4, v3, neon, wasm128, scalar]
    );
}

// ============================================================================
// expand_vertical_pass — sw × out_h output buffer (zero-insert 5-tap)
// ============================================================================
//
// Per scalar reference (cvvdp_gpu::kernels::pyramid::gausspyr_expand_scalar):
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

#[archmage::magetypes(define(f32x16), +v4, +v4x, -v3, -neon, -wasm128, -scalar)]
fn expand_v_inner(
    token: Token,
    src: &[f32],
    sw: usize,
    sh: usize,
    out_h: usize,
    vscratch: &mut [f32],
) {
    let k = GAUSS5;
    let dk0 = f32x16::splat(token, 2.0 * k[0]);
    let dk1 = f32x16::splat(token, 2.0 * k[1]);
    let dk2 = f32x16::splat(token, 2.0 * k[2]);
    let dk3 = f32x16::splat(token, 2.0 * k[3]);
    let dk4 = f32x16::splat(token, 2.0 * k[4]);

    let z_len_v = out_h + 4;
    let odd_h = out_h & 1;
    let back_idx_v = out_h + 2 + odd_h;

    let mut z_group = alloc::vec![0.0f32; 16 * z_len_v];
    let n_groups = sw / 16;

    for cg in 0..n_groups {
        let col_base = cg * 16;
        // Clear (need fresh zeros for the zero-insertion holes).
        for v in z_group.iter_mut() {
            *v = 0.0;
        }
        // Layout: z_group[y * 16 + r] = z_v_for_column(col_base + r)[y]
        for r in 0..16 {
            z_group[r] = src[col_base + r];
        }
        for ky in 0..sh {
            let y_z = 2 + 2 * ky;
            let src_off = ky * sw + col_base;
            for r in 0..16 {
                z_group[y_z * 16 + r] = src[src_off + r];
            }
        }
        let back_off = (sh - 1) * sw + col_base;
        for r in 0..16 {
            z_group[back_idx_v * 16 + r] = src[back_off + r];
        }
        // Conv sweep.
        for y in 0..out_h {
            let base = y * 16;
            let v0 = f32x16::from_array(token, z_group[base..base + 16].try_into().unwrap());
            let v1 = f32x16::from_array(token, z_group[base + 16..base + 32].try_into().unwrap());
            let v2 = f32x16::from_array(token, z_group[base + 32..base + 48].try_into().unwrap());
            let v3v = f32x16::from_array(token, z_group[base + 48..base + 64].try_into().unwrap());
            let v4 = f32x16::from_array(token, z_group[base + 64..base + 80].try_into().unwrap());
            let acc = v0 * dk0 + v1 * dk1 + v2 * dk2 + v3v * dk3 + v4 * dk4;
            let arr = acc.to_array();
            vscratch[y * sw + col_base..y * sw + col_base + 16].copy_from_slice(&arr);
        }
    }

    // Scalar tail columns.
    let mut z_v = alloc::vec![0.0f32; z_len_v];
    for x in n_groups * 16..sw {
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
            vscratch[y * sw + x] = 2.0 * sum;
        }
    }
}

#[archmage::magetypes(define(f32x8), v3, neon, wasm128, scalar)]
fn expand_v_inner(
    token: Token,
    src: &[f32],
    sw: usize,
    sh: usize,
    out_h: usize,
    vscratch: &mut [f32],
) {
    let k = GAUSS5;
    let dk0 = f32x8::splat(token, 2.0 * k[0]);
    let dk1 = f32x8::splat(token, 2.0 * k[1]);
    let dk2 = f32x8::splat(token, 2.0 * k[2]);
    let dk3 = f32x8::splat(token, 2.0 * k[3]);
    let dk4 = f32x8::splat(token, 2.0 * k[4]);

    let z_len_v = out_h + 4;
    let odd_h = out_h & 1;
    let back_idx_v = out_h + 2 + odd_h;

    let mut z_group = alloc::vec![0.0f32; 8 * z_len_v];
    let n_groups = sw / 8;

    for cg in 0..n_groups {
        let col_base = cg * 8;
        for v in z_group.iter_mut() {
            *v = 0.0;
        }
        for r in 0..8 {
            z_group[r] = src[col_base + r];
        }
        for ky in 0..sh {
            let y_z = 2 + 2 * ky;
            let src_off = ky * sw + col_base;
            for r in 0..8 {
                z_group[y_z * 8 + r] = src[src_off + r];
            }
        }
        let back_off = (sh - 1) * sw + col_base;
        for r in 0..8 {
            z_group[back_idx_v * 8 + r] = src[back_off + r];
        }
        for y in 0..out_h {
            let base = y * 8;
            let v0 = f32x8::from_array(token, z_group[base..base + 8].try_into().unwrap());
            let v1 = f32x8::from_array(token, z_group[base + 8..base + 16].try_into().unwrap());
            let v2 = f32x8::from_array(token, z_group[base + 16..base + 24].try_into().unwrap());
            let v3v = f32x8::from_array(token, z_group[base + 24..base + 32].try_into().unwrap());
            let v4 = f32x8::from_array(token, z_group[base + 32..base + 40].try_into().unwrap());
            let acc = v0 * dk0 + v1 * dk1 + v2 * dk2 + v3v * dk3 + v4 * dk4;
            let arr = acc.to_array();
            vscratch[y * sw + col_base..y * sw + col_base + 8].copy_from_slice(&arr);
        }
    }

    let mut z_v = alloc::vec![0.0f32; z_len_v];
    for x in n_groups * 8..sw {
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
            vscratch[y * sw + x] = 2.0 * sum;
        }
    }
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
    archmage::incant!(
        expand_v_inner(src, sw, sh, out_h, vscratch),
        [v4x, v4, v3, neon, wasm128, scalar]
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

#[archmage::magetypes(define(f32x16), +v4, +v4x, -v3, -neon, -wasm128, -scalar)]
fn expand_h_inner(
    token: Token,
    vscratch: &[f32],
    sw: usize,
    out_w: usize,
    out_h: usize,
    dst: &mut [f32],
    z_h_scratch: &mut Vec<f32>,
) {
    let k = GAUSS5;
    let dk0 = f32x16::splat(token, 2.0 * k[0]);
    let dk1 = f32x16::splat(token, 2.0 * k[1]);
    let dk2 = f32x16::splat(token, 2.0 * k[2]);
    let dk3 = f32x16::splat(token, 2.0 * k[3]);
    let dk4 = f32x16::splat(token, 2.0 * k[4]);

    let z_len_h = out_w + 4;
    let odd_w = out_w & 1;
    let back_idx_h = out_w + 2 + odd_w;
    z_h_scratch.clear();
    z_h_scratch.resize(z_len_h, 0.0);
    let z_h = z_h_scratch.as_mut_slice();

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

        let n_groups = out_w / 16;
        for cg in 0..n_groups {
            let x_base = cg * 16;
            let mut arrs: [[f32; 16]; 5] = [[0.0; 16]; 5];
            for r in 0..16 {
                let x = x_base + r;
                arrs[0][r] = z_h[x];
                arrs[1][r] = z_h[x + 1];
                arrs[2][r] = z_h[x + 2];
                arrs[3][r] = z_h[x + 3];
                arrs[4][r] = z_h[x + 4];
            }
            let v0 = f32x16::from_array(token, arrs[0]);
            let v1 = f32x16::from_array(token, arrs[1]);
            let v2 = f32x16::from_array(token, arrs[2]);
            let v3v = f32x16::from_array(token, arrs[3]);
            let v4 = f32x16::from_array(token, arrs[4]);
            let acc = v0 * dk0 + v1 * dk1 + v2 * dk2 + v3v * dk3 + v4 * dk4;
            let arr = acc.to_array();
            dst[y * out_w + x_base..y * out_w + x_base + 16].copy_from_slice(&arr);
        }

        for x in n_groups * 16..out_w {
            let sum = k[0] * z_h[x]
                + k[1] * z_h[x + 1]
                + k[2] * z_h[x + 2]
                + k[3] * z_h[x + 3]
                + k[4] * z_h[x + 4];
            dst[y * out_w + x] = 2.0 * sum;
        }
    }
}

#[archmage::magetypes(define(f32x8), v3, neon, wasm128, scalar)]
fn expand_h_inner(
    token: Token,
    vscratch: &[f32],
    sw: usize,
    out_w: usize,
    out_h: usize,
    dst: &mut [f32],
    z_h_scratch: &mut Vec<f32>,
) {
    let k = GAUSS5;
    let dk0 = f32x8::splat(token, 2.0 * k[0]);
    let dk1 = f32x8::splat(token, 2.0 * k[1]);
    let dk2 = f32x8::splat(token, 2.0 * k[2]);
    let dk3 = f32x8::splat(token, 2.0 * k[3]);
    let dk4 = f32x8::splat(token, 2.0 * k[4]);

    let z_len_h = out_w + 4;
    let odd_w = out_w & 1;
    let back_idx_h = out_w + 2 + odd_w;
    z_h_scratch.clear();
    z_h_scratch.resize(z_len_h, 0.0);
    let z_h = z_h_scratch.as_mut_slice();

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

        let n_groups = out_w / 8;
        for cg in 0..n_groups {
            let x_base = cg * 8;
            let mut arrs: [[f32; 8]; 5] = [[0.0; 8]; 5];
            for r in 0..8 {
                let x = x_base + r;
                arrs[0][r] = z_h[x];
                arrs[1][r] = z_h[x + 1];
                arrs[2][r] = z_h[x + 2];
                arrs[3][r] = z_h[x + 3];
                arrs[4][r] = z_h[x + 4];
            }
            let v0 = f32x8::from_array(token, arrs[0]);
            let v1 = f32x8::from_array(token, arrs[1]);
            let v2 = f32x8::from_array(token, arrs[2]);
            let v3v = f32x8::from_array(token, arrs[3]);
            let v4 = f32x8::from_array(token, arrs[4]);
            let acc = v0 * dk0 + v1 * dk1 + v2 * dk2 + v3v * dk3 + v4 * dk4;
            let arr = acc.to_array();
            dst[y * out_w + x_base..y * out_w + x_base + 8].copy_from_slice(&arr);
        }

        for x in n_groups * 8..out_w {
            let sum = k[0] * z_h[x]
                + k[1] * z_h[x + 1]
                + k[2] * z_h[x + 2]
                + k[3] * z_h[x + 3]
                + k[4] * z_h[x + 4];
            dst[y * out_w + x] = 2.0 * sum;
        }
    }
}

/// Horizontal pass of `gausspyr_expand`: reads `sw × out_h` vscratch,
/// writes `out_w × out_h` dst using the zero-insertion + 5-tap kernel.
pub(crate) fn expand_horizontal_pass(
    vscratch: &[f32],
    sw: usize,
    out_w: usize,
    out_h: usize,
    dst: &mut [f32],
    z_h_scratch: &mut Vec<f32>,
) {
    debug_assert_eq!(dst.len(), out_w * out_h);
    archmage::incant!(
        expand_h_inner(vscratch, sw, out_w, out_h, dst, z_h_scratch),
        [v4x, v4, v3, neon, wasm128, scalar]
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
// Per scalar reference (`cvvdp_gpu::kernels::masking::gaussian_blur_sigma3`):
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

#[archmage::magetypes(define(f32x16), +v4, +v4x, -v3, -neon, -wasm128, -scalar)]
fn pu_blur_h_inner(token: Token, src: &[f32], w: usize, h: usize, dst: &mut [f32]) {
    let k = PU_BLUR_KERNEL_1D;
    let k0 = f32x16::splat(token, k[0]);
    let k1 = f32x16::splat(token, k[1]);
    let k2 = f32x16::splat(token, k[2]);
    let k3 = f32x16::splat(token, k[3]);
    let k4 = f32x16::splat(token, k[4]);
    let k5 = f32x16::splat(token, k[5]);
    let k6 = f32x16::splat(token, k[6]);
    let k7 = f32x16::splat(token, k[7]);
    let k8 = f32x16::splat(token, k[8]);
    let k9 = f32x16::splat(token, k[9]);
    let k10 = f32x16::splat(token, k[10]);
    let k11 = f32x16::splat(token, k[11]);
    let k12 = f32x16::splat(token, k[12]);

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

    for y in 0..h {
        let row_off = y * w;
        // Scalar prefix: x in [0, 6).
        for x in 0..interior_lo.min(w) {
            let mut s = 0.0f32;
            for t in 0..13 {
                let sx = reflect_idx_for_blur(x as isize + t as isize - 6, w);
                s += k[t] * src[row_off + sx];
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
                let base = row_off + x_base - 6 + off;
                let arr: [f32; 16] = src[base..base + 16].try_into().unwrap();
                f32x16::from_array(token, arr)
            };
            let v0 = load(0);
            let v1 = load(1);
            let v2 = load(2);
            let v3 = load(3);
            let v4 = load(4);
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
                + v4 * k4
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
                s += k[t] * src[row_off + sx];
            }
            dst[row_off + x] = s;
        }
    }
}

#[archmage::magetypes(define(f32x8), v3, neon, wasm128, scalar)]
fn pu_blur_h_inner(token: Token, src: &[f32], w: usize, h: usize, dst: &mut [f32]) {
    let k = PU_BLUR_KERNEL_1D;
    let k0 = f32x8::splat(token, k[0]);
    let k1 = f32x8::splat(token, k[1]);
    let k2 = f32x8::splat(token, k[2]);
    let k3 = f32x8::splat(token, k[3]);
    let k4 = f32x8::splat(token, k[4]);
    let k5 = f32x8::splat(token, k[5]);
    let k6 = f32x8::splat(token, k[6]);
    let k7 = f32x8::splat(token, k[7]);
    let k8 = f32x8::splat(token, k[8]);
    let k9 = f32x8::splat(token, k[9]);
    let k10 = f32x8::splat(token, k[10]);
    let k11 = f32x8::splat(token, k[11]);
    let k12 = f32x8::splat(token, k[12]);

    let lane = 8usize;
    let interior_lo = 6usize;
    // Loop while `x_base + lane <= w` AND
    // `x_base + (lane - 1) + 6 < w` (last tap of last lane in-bounds),
    // i.e. `x_base + lane + 5 <= w` ⇒ `x_base <= w - lane - 5`.
    let interior_hi_excl = w.saturating_sub(lane + 5);

    for y in 0..h {
        let row_off = y * w;
        for x in 0..interior_lo.min(w) {
            let mut s = 0.0f32;
            for t in 0..13 {
                let sx = reflect_idx_for_blur(x as isize + t as isize - 6, w);
                s += k[t] * src[row_off + sx];
            }
            dst[row_off + x] = s;
        }
        let mut x_base = interior_lo;
        while x_base < interior_hi_excl {
            let load = |off: usize| -> f32x8 {
                let base = row_off + x_base - 6 + off;
                let arr: [f32; 8] = src[base..base + 8].try_into().unwrap();
                f32x8::from_array(token, arr)
            };
            let v0 = load(0);
            let v1 = load(1);
            let v2 = load(2);
            let v3 = load(3);
            let v4 = load(4);
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
                + v4 * k4
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
                s += k[t] * src[row_off + sx];
            }
            dst[row_off + x] = s;
        }
    }
}

/// Horizontal pass of the σ=3 13-tap Gaussian blur. Writes the `w × h`
/// scratch buffer with the per-row 13-tap convolution + reflect
/// padding. Bit-equivalent to `gaussian_blur_sigma3`'s horizontal
/// pass except for SIMD FMA-grouping in the interior (well below
/// 1e-5 abs).
pub(crate) fn pu_blur_horizontal_pass(src: &[f32], w: usize, h: usize, h_pass: &mut [f32]) {
    debug_assert_eq!(src.len(), w * h);
    debug_assert_eq!(h_pass.len(), w * h);
    archmage::incant!(
        pu_blur_h_inner(src, w, h, h_pass),
        [v4x, v4, v3, neon, wasm128, scalar]
    );
}

#[archmage::magetypes(define(f32x16), +v4, +v4x, -v3, -neon, -wasm128, -scalar)]
fn pu_blur_v_inner(token: Token, h_pass: &[f32], w: usize, h: usize, dst: &mut [f32]) {
    let k = PU_BLUR_KERNEL_1D;
    let k0 = f32x16::splat(token, k[0]);
    let k1 = f32x16::splat(token, k[1]);
    let k2 = f32x16::splat(token, k[2]);
    let k3 = f32x16::splat(token, k[3]);
    let k4 = f32x16::splat(token, k[4]);
    let k5 = f32x16::splat(token, k[5]);
    let k6 = f32x16::splat(token, k[6]);
    let k7 = f32x16::splat(token, k[7]);
    let k8 = f32x16::splat(token, k[8]);
    let k9 = f32x16::splat(token, k[9]);
    let k10 = f32x16::splat(token, k[10]);
    let k11 = f32x16::splat(token, k[11]);
    let k12 = f32x16::splat(token, k[12]);

    let lane = 16usize;
    let n_groups = w / lane;
    let interior_lo = 6usize;
    let interior_hi = h.saturating_sub(6); // y in [interior_lo, interior_hi) is SIMD interior

    // Boundary rows: y in [0, 6) ∪ [h-6, h) — scalar.
    for y in 0..interior_lo.min(h) {
        let row_off = y * w;
        for x in 0..w {
            let mut s = 0.0f32;
            for t in 0..13 {
                let sy = reflect_idx_for_blur(y as isize + t as isize - 6, h);
                s += k[t] * h_pass[sy * w + x];
            }
            dst[row_off + x] = s;
        }
    }

    // Interior rows: SIMD across columns within each row.
    for y in interior_lo..interior_hi {
        let row_off = y * w;
        // SIMD lane groups across w.
        for cg in 0..n_groups {
            let x_base = cg * lane;
            // Each tap reads N contiguous floats at
            // `(y - 6 + t) * w + x_base` (no reflection in interior).
            let load = |t: usize| -> f32x16 {
                let off = (y - 6 + t) * w + x_base;
                let arr: [f32; 16] = h_pass[off..off + 16].try_into().unwrap();
                f32x16::from_array(token, arr)
            };
            let v0 = load(0);
            let v1 = load(1);
            let v2 = load(2);
            let v3 = load(3);
            let v4 = load(4);
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
                + v4 * k4
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
        }
        // Scalar tail columns (< lane). No reflect needed (y is interior).
        for x in n_groups * lane..w {
            let mut s = 0.0f32;
            for t in 0..13 {
                let sy = y + t - 6; // safe: y >= 6
                s += k[t] * h_pass[sy * w + x];
            }
            dst[row_off + x] = s;
        }
    }

    // Boundary rows: y in [h-6, h) — scalar.
    for y in interior_hi.max(interior_lo)..h {
        let row_off = y * w;
        for x in 0..w {
            let mut s = 0.0f32;
            for t in 0..13 {
                let sy = reflect_idx_for_blur(y as isize + t as isize - 6, h);
                s += k[t] * h_pass[sy * w + x];
            }
            dst[row_off + x] = s;
        }
    }
}

#[archmage::magetypes(define(f32x8), v3, neon, wasm128, scalar)]
fn pu_blur_v_inner(token: Token, h_pass: &[f32], w: usize, h: usize, dst: &mut [f32]) {
    let k = PU_BLUR_KERNEL_1D;
    let k0 = f32x8::splat(token, k[0]);
    let k1 = f32x8::splat(token, k[1]);
    let k2 = f32x8::splat(token, k[2]);
    let k3 = f32x8::splat(token, k[3]);
    let k4 = f32x8::splat(token, k[4]);
    let k5 = f32x8::splat(token, k[5]);
    let k6 = f32x8::splat(token, k[6]);
    let k7 = f32x8::splat(token, k[7]);
    let k8 = f32x8::splat(token, k[8]);
    let k9 = f32x8::splat(token, k[9]);
    let k10 = f32x8::splat(token, k[10]);
    let k11 = f32x8::splat(token, k[11]);
    let k12 = f32x8::splat(token, k[12]);

    let lane = 8usize;
    let n_groups = w / lane;
    let interior_lo = 6usize;
    let interior_hi = h.saturating_sub(6);

    for y in 0..interior_lo.min(h) {
        let row_off = y * w;
        for x in 0..w {
            let mut s = 0.0f32;
            for t in 0..13 {
                let sy = reflect_idx_for_blur(y as isize + t as isize - 6, h);
                s += k[t] * h_pass[sy * w + x];
            }
            dst[row_off + x] = s;
        }
    }

    for y in interior_lo..interior_hi {
        let row_off = y * w;
        for cg in 0..n_groups {
            let x_base = cg * lane;
            let load = |t: usize| -> f32x8 {
                let off = (y - 6 + t) * w + x_base;
                let arr: [f32; 8] = h_pass[off..off + 8].try_into().unwrap();
                f32x8::from_array(token, arr)
            };
            let v0 = load(0);
            let v1 = load(1);
            let v2 = load(2);
            let v3 = load(3);
            let v4 = load(4);
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
                + v4 * k4
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
        }
        for x in n_groups * lane..w {
            let mut s = 0.0f32;
            for t in 0..13 {
                let sy = y + t - 6;
                s += k[t] * h_pass[sy * w + x];
            }
            dst[row_off + x] = s;
        }
    }

    for y in interior_hi.max(interior_lo)..h {
        let row_off = y * w;
        for x in 0..w {
            let mut s = 0.0f32;
            for t in 0..13 {
                let sy = reflect_idx_for_blur(y as isize + t as isize - 6, h);
                s += k[t] * h_pass[sy * w + x];
            }
            dst[row_off + x] = s;
        }
    }
}

/// Vertical pass of the σ=3 13-tap Gaussian blur. Reads `h_pass`
/// (output of `pu_blur_horizontal_pass`) and writes the final blur
/// into `dst`.
pub(crate) fn pu_blur_vertical_pass(h_pass: &[f32], w: usize, h: usize, dst: &mut [f32]) {
    debug_assert_eq!(h_pass.len(), w * h);
    debug_assert_eq!(dst.len(), w * h);
    archmage::incant!(
        pu_blur_v_inner(h_pass, w, h, dst),
        [v4x, v4, v3, neon, wasm128, scalar]
    );
}

/// Full σ=3 13-tap separable Gaussian blur with caller-owned scratch.
///
/// Replacement for `cvvdp_gpu::kernels::masking::gaussian_blur_sigma3`
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
    h_pass.clear();
    h_pass.resize(n, 0.0);
    dst.clear();
    dst.resize(n, 0.0);
    pu_blur_horizontal_pass(src, w, h, h_pass.as_mut_slice());
    pu_blur_vertical_pass(h_pass.as_slice(), w, h, dst.as_mut_slice());
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
            let mut z = Vec::new();
            expand_horizontal_pass(&vs, sw, ow, oh, &mut got, &mut z);
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
        use cvvdp_gpu::kernels::masking::gaussian_blur_sigma3;
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
        use cvvdp_gpu::kernels::masking::PU_BLUR_KERNEL_1D;
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
        use cvvdp_gpu::kernels::masking::PU_BLUR_KERNEL_1D;
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
        use cvvdp_gpu::kernels::masking::gaussian_blur_sigma3;
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
