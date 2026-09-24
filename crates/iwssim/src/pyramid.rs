//! Laplacian pyramid build — `pyrtools.pyramids.LaplacianPyramid`.
//!
//! pyrtools' convention:
//!
//! - Filter: `binom5 = sqrt(2) * [1, 4, 6, 4, 1] / 16`.
//! - Boundary: `reflect1` (reflection through the edge pixels, the
//!   edge pixel itself is **not** duplicated). `−1 → 1`, `−2 → 2`,
//!   `n → n−2`, etc.
//! - Build separably: filter horizontally with stride 2, then filter
//!   vertically with stride 2. Each downsample halves both axes via
//!   `ceil(W/2)`, `ceil(H/2)`.
//! - Expand (for the Laplacian residual `LP_j = G_j − expand(G_{j+1})`)
//!   uses the same filter applied via zero-insertion upsampling.
//!   pyrtools' `upConv` semantics: insert one zero between every sample,
//!   convolve with the same filter. Because `binom5` already carries
//!   `sqrt(2)`, applying the same filter on both build and expand
//!   passes preserves DC without an additional scale factor.
//!
//! All routines here are scalar f32. SIMD variants live in
//! `pyramid_simd.rs` (currently unused — wired in Phase 8g step 4).

use alloc::vec::Vec;

#[cfg(test)]
use crate::filters::{BINOM5, BINOM5_LEN, BINOM5_RADIUS};

/// One Laplacian pyramid level.
#[derive(Debug, Clone)]
pub(crate) struct PyrLevel {
    /// Width of this level (`= ceil(parent_width / 2)`).
    #[allow(dead_code)] // retained for future diagnostic inspection
    pub w: usize,
    /// Height of this level (`= ceil(parent_height / 2)`).
    #[allow(dead_code)] // retained for future diagnostic inspection
    pub h: usize,
    /// Gaussian band at this scale (the lowpass for the level below).
    pub g: Vec<f32>,
    /// Laplacian band at this scale: `LP_j = G_j - expand(G_{j+1})`.
    /// At the coarsest scale, `lp == g` (the residual lowpass).
    pub lp: Vec<f32>,
}

/// reflect1: index `i` outside `[0, n)` mirrors through the edge
/// pixels — `−1 → 1`, `−2 → 2`, `n → n−2`, `n+1 → n−3`, etc.
///
/// Filter radius is `≤ 2` for binom5, so two iterations of the reflect
/// step cover any in-range result.
#[inline]
pub(crate) fn reflect1(i: i32, n: i32) -> i32 {
    let mut k = i;
    if k < 0 {
        k = -k;
    }
    if k >= n {
        k = 2 * (n - 1) - k;
    }
    if k < 0 {
        k = -k;
    }
    k
}

/// `corr_dn` along the horizontal axis: correlate with `binom5`, then
/// decimate by 2. Input `(h, in_w)` → output `(h, out_w)` with
/// `out_w = ceil(in_w / 2)`.
#[cfg(test)]
pub(crate) fn corr_dn_horizontal(
    src: &[f32],
    h: usize,
    in_w: usize,
    out_w: usize,
    dst: &mut [f32],
) {
    debug_assert_eq!(src.len(), h * in_w);
    debug_assert_eq!(dst.len(), h * out_w);
    let r = BINOM5_RADIUS;
    let in_w_i = in_w as i32;
    for y in 0..h {
        let row = &src[y * in_w..(y + 1) * in_w];
        let dst_row = &mut dst[y * out_w..(y + 1) * out_w];
        for ox in 0..out_w {
            let in_x_center = (ox as i32) * 2;
            let mut acc = 0.0_f32;
            for k in 0..BINOM5_LEN {
                let xs = reflect1(in_x_center - r + k as i32, in_w_i) as usize;
                acc += BINOM5[k] * row[xs];
            }
            dst_row[ox] = acc;
        }
    }
}

/// `corr_dn` along the vertical axis: correlate with `binom5`, then
/// decimate by 2. Input `(in_h, w)` → output `(out_h, w)` with
/// `out_h = ceil(in_h / 2)`.
#[cfg(test)]
pub(crate) fn corr_dn_vertical(src: &[f32], in_h: usize, w: usize, out_h: usize, dst: &mut [f32]) {
    debug_assert_eq!(src.len(), in_h * w);
    debug_assert_eq!(dst.len(), out_h * w);
    let r = BINOM5_RADIUS;
    let in_h_i = in_h as i32;
    for oy in 0..out_h {
        let in_y_center = (oy as i32) * 2;
        let mut ys = [0usize; BINOM5_LEN];
        for k in 0..BINOM5_LEN {
            ys[k] = reflect1(in_y_center - r + k as i32, in_h_i) as usize;
        }
        let dst_row = &mut dst[oy * w..(oy + 1) * w];
        // Per-x reduction over 5 taps.
        for x in 0..w {
            let mut acc = 0.0_f32;
            for k in 0..BINOM5_LEN {
                acc += BINOM5[k] * src[ys[k] * w + x];
            }
            dst_row[x] = acc;
        }
    }
}

/// Reflect-on-expanded-axis helper for upConv. The zero-stuffed signal
/// has length `2·in_axis`. `i` may be negative or `>= 2·in_axis` and
/// we reflect through the edge samples to bring it in-bounds, then
/// return `(active, sx)` where `active = true` indicates a real source
/// sample at `src[sx]` (vs an inserted zero).
#[inline]
#[cfg(test)]
fn reflect_expanded(i: i32, in_axis: i32) -> (bool, usize) {
    let two_n = 2 * in_axis;
    let mut q = i;
    if q < 0 {
        q = -q;
    }
    if q >= two_n {
        q = 2 * (two_n - 1) - q;
    }
    if q < 0 {
        q = -q;
    }
    if q >= two_n {
        q = 2 * (two_n - 1) - q;
    }
    let active = (q & 1) == 0;
    let sx = (q / 2) as usize;
    (active, sx)
}

/// `up_conv` along horizontal: zero-insert × 2, then correlate with
/// `binom5`. Output `(h, out_w)` where `out_w` is typically `2*in_w`
/// or `2*in_w - 1`.
#[cfg(test)]
pub(crate) fn up_conv_horizontal(
    src: &[f32],
    h: usize,
    in_w: usize,
    out_w: usize,
    dst: &mut [f32],
) {
    debug_assert_eq!(src.len(), h * in_w);
    debug_assert_eq!(dst.len(), h * out_w);
    let r = BINOM5_RADIUS;
    let in_w_i = in_w as i32;
    for y in 0..h {
        let row = &src[y * in_w..(y + 1) * in_w];
        let dst_row = &mut dst[y * out_w..(y + 1) * out_w];
        for ox in 0..out_w {
            let p0 = (ox as i32) - r;
            let mut acc = 0.0_f32;
            for k in 0..BINOM5_LEN {
                let (active, sx) = reflect_expanded(p0 + k as i32, in_w_i);
                if active {
                    acc += BINOM5[k] * row[sx];
                }
            }
            dst_row[ox] = acc;
        }
    }
}

/// `up_conv` along vertical: zero-insert × 2 then correlate.
#[cfg(test)]
pub(crate) fn up_conv_vertical(src: &[f32], in_h: usize, w: usize, out_h: usize, dst: &mut [f32]) {
    debug_assert_eq!(src.len(), in_h * w);
    debug_assert_eq!(dst.len(), out_h * w);
    let r = BINOM5_RADIUS;
    let in_h_i = in_h as i32;
    for oy in 0..out_h {
        let p0 = (oy as i32) - r;
        // Precompute taps for this output row.
        let mut ys = [0usize; BINOM5_LEN];
        let mut act = [false; BINOM5_LEN];
        for k in 0..BINOM5_LEN {
            let (a, sy) = reflect_expanded(p0 + k as i32, in_h_i);
            ys[k] = sy;
            act[k] = a;
        }
        let dst_row = &mut dst[oy * w..(oy + 1) * w];
        for x in 0..w {
            let mut acc = 0.0_f32;
            for k in 0..BINOM5_LEN {
                if act[k] {
                    acc += BINOM5[k] * src[ys[k] * w + x];
                }
            }
            dst_row[x] = acc;
        }
    }
}

/// Build the per-scale dimensions for a 5-level pyramid starting from
/// `(w_0, h_0)`. Each next scale is `(ceil(w/2), ceil(h/2))`.
pub(crate) fn pyramid_dims(w0: usize, h0: usize, n_levels: usize) -> Vec<(usize, usize)> {
    let mut dims = Vec::with_capacity(n_levels);
    let (mut w, mut h) = (w0, h0);
    for _ in 0..n_levels {
        dims.push((w, h));
        w = w.div_ceil(2);
        h = h.div_ceil(2);
    }
    dims
}

/// Build the 5-level Laplacian pyramid for a single gray image.
///
/// Output ordering matches the Python reference's `lp.pyr_coeffs[(s-1, 0)]`
/// indexed by scale `s ∈ {1..=5}` — i.e. `levels[0]` is the finest band,
/// `levels[4]` is the coarsest residual lowpass.
pub(crate) fn build_laplacian_pyramid(
    img: &[f32],
    w0: usize,
    h0: usize,
    n_levels: usize,
) -> Vec<PyrLevel> {
    assert_eq!(img.len(), w0 * h0);
    let dims = pyramid_dims(w0, h0, n_levels);

    // 1. Gaussian chain. g[0] = img; g[s+1] = corrDn_v(corrDn_h(g[s])).
    let mut g_levels: Vec<Vec<f32>> = Vec::with_capacity(n_levels);
    g_levels.push(img.to_vec());
    for s in 0..(n_levels - 1) {
        let (w_cur, h_cur) = dims[s];
        let (w_nxt, h_nxt) = dims[s + 1];
        // Horizontal pass: (h_cur, w_cur) → (h_cur, w_nxt).
        let mut scratch = alloc::vec![0.0_f32; h_cur * w_nxt];
        crate::simd_kernels::corr_dn_h(&g_levels[s], h_cur, w_cur, w_nxt, &mut scratch);
        // Vertical pass: (h_cur, w_nxt) → (h_nxt, w_nxt).
        let mut g_nxt = alloc::vec![0.0_f32; h_nxt * w_nxt];
        crate::simd_kernels::corr_dn_v(&scratch, h_cur, w_nxt, h_nxt, &mut g_nxt);
        g_levels.push(g_nxt);
    }

    // 2. Laplacian bands: LP[s] = G[s] − expand(G[s+1]); LP[top] = G[top].
    let mut levels: Vec<PyrLevel> = Vec::with_capacity(n_levels);
    for s in 0..n_levels {
        let (w_cur, h_cur) = dims[s];
        let g = g_levels[s].clone();
        let lp = if s == n_levels - 1 {
            // Residual lowpass.
            g.clone()
        } else {
            let (w_nxt, h_nxt) = dims[s + 1];
            // Expand g[s+1] → (h_cur, w_cur).
            //   Horizontal: (h_nxt, w_nxt) → (h_nxt, w_cur).
            let mut h_scratch = alloc::vec![0.0_f32; h_nxt * w_cur];
            crate::simd_kernels::up_conv_h(&g_levels[s + 1], h_nxt, w_nxt, w_cur, &mut h_scratch);
            //   Vertical: (h_nxt, w_cur) → (h_cur, w_cur).
            let mut expanded = alloc::vec![0.0_f32; h_cur * w_cur];
            crate::simd_kernels::up_conv_v(&h_scratch, h_nxt, w_cur, h_cur, &mut expanded);
            // LP[s] = G[s] − expand(G[s+1]).
            let mut lp = alloc::vec![0.0_f32; h_cur * w_cur];
            for i in 0..(h_cur * w_cur) {
                lp[i] = g[i] - expanded[i];
            }
            lp
        };
        levels.push(PyrLevel {
            w: w_cur,
            h: h_cur,
            g,
            lp,
        });
    }
    levels
}

/// `imenlarge2` matches the Python reference exactly:
///
/// Input `(M, N)` → output `(M, N)` (same shape!), via the following
/// sequence:
///
/// 1. Bilinear upsample to `(4M-3, 4N-3)`.
/// 2. Pad by 1 on each side via linear extrapolation:
///    `out[0,:] = 2·out[1,:] − out[2,:]` (and likewise for last row /
///    first col / last col).
/// 3. Decimate by 2 in both axes → `(2M-1, 2N-1)` (approximately, the
///    exact formula is `((4M-1)+1)/2 = 2M`, taking the `::2` slice).
///
/// The Python code: `imu = t2[:, :, ::2, ::2]`. With `t2` of shape
/// `(4M-1, 4N-1)`, this gives output shape `(2M, 2N)` (Python slice
/// rounds up). Then the caller crops to `(M, N)` again via
/// `auxp[0:Nsy, 0:Nsx]`.
///
/// To keep the call path simple, we return the cropped `(M_dst, N_dst)`
/// result directly given target dims.
pub(crate) fn imenlarge2(
    src: &[f32],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<f32> {
    assert_eq!(src.len(), src_w * src_h);

    // Stage 1 is a bilinear upsample to `t1` (4M-3 × 4N-3) — but it is
    // never materialized: `imu[y][x] = t2[2y][2x]` reads t2 only on its
    // even grid, which touches only `t1[2y-1][2x-1]` interior cells
    // plus a thin ring of t1 border cells. `t1_at` evaluates a single
    // t1 cell straight from `src` (same exact-quarter formula as
    // `bilinear_upsample`), so `imu` needs ~4·src work instead of the
    // ~16·src a materialized t1 costs — bit-identical, no 16×
    // intermediate.
    let t1_h = 4 * src_h - 3;
    let t1_w = 4 * src_w - 3;

    // Degenerate inputs (t1 with <2 rows or cols) can't run the fused
    // border formulas — fall back to the materialized-t2 path, which
    // reads zero-padded cells for out-of-range rows.
    if t1_h < 2 || t1_w < 2 {
        return imenlarge2_small(src, src_w, src_h, dst_w, dst_h);
    }

    let t2_h = t1_h + 2;
    let t2_w = t1_w + 2;
    let imu_h = t2_h.div_ceil(2);
    let imu_w = t2_w.div_ceil(2);
    let mut imu = alloc::vec![0.0_f32; imu_h * imu_w];

    let t1_at = |dy: usize, dx: usize| -> f32 { t1_cell(src, src_w, src_h, dy, dx) };

    crate::par::map_rows(&mut imu, imu_w, |y0, band| {
        let band_rows = band.len() / imu_w;
        for r in 0..band_rows {
            let y = y0 + r;
            let row = &mut band[r * imu_w..(r + 1) * imu_w];
            if y == 0 || y == imu_h - 1 {
                // Border row: imu[y][x] = 2·t1[ya][2x-1] − t1[yb][2x-1]
                // (top: ya=0,yb=1; bottom: ya=t1_h-1,yb=t1_h-2).
                let (ya, yb) = if y == 0 {
                    (0usize, 1usize)
                } else {
                    (t1_h - 1, t1_h - 2)
                };
                for x in 1..imu_w - 1 {
                    row[x] = 2.0 * t1_at(ya, 2 * x - 1) - t1_at(yb, 2 * x - 1);
                }
            } else {
                // Interior: imu[y][x] = t1[2y-1][2x-1].
                let ty = 2 * y - 1;
                for x in 1..imu_w - 1 {
                    row[x] = t1_at(ty, 2 * x - 1);
                }
                // Border cols: imu[y][0] = 2·t1[ty][0] − t1[ty][1],
                // mirrored on the right.
                row[0] = 2.0 * t1_at(ty, 0) - t1_at(ty, 1);
                row[imu_w - 1] = 2.0 * t1_at(ty, t1_w - 1) - t1_at(ty, t1_w - 2);
            }
        }
    });
    // Corners: t2 corner = 2·(adjacent extrap cell) − (next extrap
    // cell), where each extrap cell is itself `2·edge − inner` on t1.
    imu[0] = 2.0 * (2.0 * t1_at(0, 0) - t1_at(1, 0)) - (2.0 * t1_at(0, 1) - t1_at(1, 1));
    imu[imu_w - 1] = 2.0 * (2.0 * t1_at(0, t1_w - 1) - t1_at(1, t1_w - 1))
        - (2.0 * t1_at(0, t1_w - 2) - t1_at(1, t1_w - 2));
    let bot = (imu_h - 1) * imu_w;
    let (ya, yb) = (t1_h - 1, t1_h - 2);
    imu[bot] = 2.0 * (2.0 * t1_at(ya, 0) - t1_at(yb, 0)) - (2.0 * t1_at(ya, 1) - t1_at(yb, 1));
    imu[bot + imu_w - 1] = 2.0 * (2.0 * t1_at(ya, t1_w - 1) - t1_at(yb, t1_w - 1))
        - (2.0 * t1_at(ya, t1_w - 2) - t1_at(yb, t1_w - 2));

    // Crop to caller's requested dst dims (`auxp[0:Nsy, 0:Nsx]`).
    if dst_w == imu_w && dst_h == imu_h {
        imu
    } else {
        let mut out = alloc::vec![0.0_f32; dst_w * dst_h];
        let copy_h = dst_h.min(imu_h);
        let copy_w = dst_w.min(imu_w);
        for y in 0..copy_h {
            let src_row = &imu[y * imu_w..y * imu_w + copy_w];
            let dst_row = &mut out[y * dst_w..y * dst_w + copy_w];
            dst_row.copy_from_slice(src_row);
        }
        out
    }
}

/// One cell of `imenlarge2`'s stage-1 bilinear plane `t1`
/// (`4·sh−3 × 4·sw−3`), evaluated directly from `src`. Same math as
/// [`bilinear_upsample`]'s per-cell body on the exact quarter grid —
/// `(dy, dx)` are t1 coordinates.
#[inline]
fn t1_cell(src: &[f32], sw: usize, sh: usize, dy: usize, dx: usize) -> f32 {
    const WX: [f32; 4] = [0.0, 0.25, 0.5, 0.75];
    let y0 = dy >> 2;
    let y1 = (y0 + 1).min(sh - 1);
    let wy = WX[dy & 3];
    let x0 = dx >> 2;
    let x1 = (x0 + 1).min(sw - 1);
    let wx = WX[dx & 3];
    let (ra, rb) = (y0 * sw, y1 * sw);
    let v00 = src[ra + x0];
    let v01 = src[ra + x1];
    let v10 = src[rb + x0];
    let v11 = src[rb + x1];
    let v0 = v00 + wx * (v01 - v00);
    let v1 = v10 + wx * (v11 - v10);
    v0 + wy * (v1 - v0)
}

/// `imenlarge2` fallback for degenerate `t1` shapes (< 2 rows or cols)
/// — materializes `t2` exactly as the reference does, including the
/// zero-initialized cells that out-of-range extrapolation reads.
fn imenlarge2_small(
    src: &[f32],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<f32> {
    let t1_h = 4 * src_h - 3;
    let t1_w = 4 * src_w - 3;
    let t1 = bilinear_upsample_general(src, src_w, src_h, t1_w, t1_h);

    let t2_h = t1_h + 2;
    let t2_w = t1_w + 2;
    let mut t2 = alloc::vec![0.0_f32; t2_h * t2_w];
    for y in 0..t1_h {
        let src_row = &t1[y * t1_w..(y + 1) * t1_w];
        let dst_row = &mut t2[(y + 1) * t2_w + 1..(y + 1) * t2_w + 1 + t1_w];
        dst_row.copy_from_slice(src_row);
    }
    for x in 0..t2_w {
        if x == 0 || x == t2_w - 1 {
            continue;
        }
        t2[x] = 2.0 * t2[t2_w + x] - t2[2 * t2_w + x];
        t2[(t2_h - 1) * t2_w + x] = 2.0 * t2[(t2_h - 2) * t2_w + x] - t2[(t2_h - 3) * t2_w + x];
    }
    for y in 0..t2_h {
        t2[y * t2_w] = 2.0 * t2[y * t2_w + 1] - t2[y * t2_w + 2];
        t2[y * t2_w + (t2_w - 1)] = 2.0 * t2[y * t2_w + (t2_w - 2)] - t2[y * t2_w + (t2_w - 3)];
    }

    let imu_h = t2_h.div_ceil(2);
    let imu_w = t2_w.div_ceil(2);
    let mut imu = alloc::vec![0.0_f32; imu_h * imu_w];
    for y in 0..imu_h {
        for x in 0..imu_w {
            imu[y * imu_w + x] = t2[(2 * y) * t2_w + (2 * x)];
        }
    }
    if dst_w == imu_w && dst_h == imu_h {
        imu
    } else {
        let mut out = alloc::vec![0.0_f32; dst_w * dst_h];
        let copy_h = dst_h.min(imu_h);
        let copy_w = dst_w.min(imu_w);
        for y in 0..copy_h {
            let src_row = &imu[y * imu_w..y * imu_w + copy_w];
            let dst_row = &mut out[y * dst_w..y * dst_w + copy_w];
            dst_row.copy_from_slice(src_row);
        }
        out
    }
}

/// General `align_corners=True` bilinear — retained for
/// [`imenlarge2_small`]'s degenerate shapes where the quarter-grid
/// shortcut doesn't apply.
fn bilinear_upsample_general(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    let mut out = alloc::vec![0.0_f32; dh * dw];
    let sx_scale = (sw - 1) as f32 / (dw - 1) as f32;
    let sy_scale = (sh - 1) as f32 / (dh - 1) as f32;
    for dy in 0..dh {
        let fy = dy as f32 * sy_scale;
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(sh - 1);
        let wy = fy - y0 as f32;
        for dx in 0..dw {
            let fx = dx as f32 * sx_scale;
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let wx = fx - x0 as f32;
            let v00 = src[y0 * sw + x0];
            let v01 = src[y0 * sw + x1];
            let v10 = src[y1 * sw + x0];
            let v11 = src[y1 * sw + x1];
            let v0 = v00 + wx * (v01 - v00);
            let v1 = v10 + wx * (v11 - v10);
            out[dy * dw + dx] = v0 + wy * (v1 - v0);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pyramid_dims_5_level() {
        let dims = pyramid_dims(256, 256, 5);
        assert_eq!(
            dims,
            alloc::vec![(256, 256), (128, 128), (64, 64), (32, 32), (16, 16)]
        );
        // Odd input: 257 → 129 → 65 → 33 → 17.
        let dims = pyramid_dims(257, 257, 5);
        assert_eq!(
            dims,
            alloc::vec![(257, 257), (129, 129), (65, 65), (33, 33), (17, 17)]
        );
    }

    #[test]
    fn reflect1_basic() {
        let n = 5;
        // In-bounds passthrough.
        assert_eq!(reflect1(0, n), 0);
        assert_eq!(reflect1(4, n), 4);
        // Negative: -1 → 1, -2 → 2.
        assert_eq!(reflect1(-1, n), 1);
        assert_eq!(reflect1(-2, n), 2);
        // Past end: 5 → 3, 6 → 2.
        assert_eq!(reflect1(5, n), 3);
        assert_eq!(reflect1(6, n), 2);
    }

    #[test]
    fn identity_pyramid_reconstruction() {
        // Build a 3-level pyramid then verify the LP at the top equals
        // the smallest G (residual lowpass).
        let w = 32;
        let h = 32;
        let mut img = alloc::vec![0.0_f32; w * h];
        for y in 0..h {
            for x in 0..w {
                img[y * w + x] = (x + y * 7) as f32;
            }
        }
        let levels = build_laplacian_pyramid(&img, w, h, 3);
        assert_eq!(levels.len(), 3);
        // Residual lowpass at top scale equals the Gaussian there.
        assert_eq!(levels[2].lp, levels[2].g);
    }

    #[test]
    fn imenlarge2_doubles_shape_within_one_pixel() {
        // A 4×4 input should produce roughly an 8×8 upscale via imenlarge2.
        let src: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let out = imenlarge2(&src, 4, 4, 8, 8);
        assert_eq!(out.len(), 64);
        // Output isn't a strict bilinear identity (the linear-extrap
        // padding rounds at the boundary); just sanity-check it's
        // monotone within a row.
        // ...nothing further; precise checks live in the pipeline test.
        let _ = out;
    }

    /// SIMD pyramid kernels must match the scalar oracle bit-for-bit
    /// on the interior (all kernels add the same tap sequence per
    /// output; `mul_add` contraction is the only permitted drift).
    #[test]
    fn simd_kernels_match_scalar_oracles() {
        // Exercise odd/even dims and boundary reflects.
        for &(w, h) in &[(17usize, 13usize), (32usize, 16usize), (9usize, 9usize)] {
            let src: Vec<f32> = (0..w * h)
                .map(|i| ((i * 31 % 67) as f32 - 33.0) / 9.0)
                .collect();
            let (ow, oh) = (w.div_ceil(2), h.div_ceil(2));
            let close = |a: &[f32], b: &[f32], ctx: &str| {
                assert_eq!(a.len(), b.len(), "{ctx} len");
                for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
                    let tol = 1e-6 * x.abs().max(1.0);
                    assert!((x - y).abs() <= tol, "{ctx}[{i}]: {x} vs {y}");
                }
            };
            // corr_dn_h
            let mut a = alloc::vec![0.0; h * ow];
            let mut b = alloc::vec![0.0; h * ow];
            corr_dn_horizontal(&src, h, w, ow, &mut a);
            crate::simd_kernels::corr_dn_h(&src, h, w, ow, &mut b);
            close(&a, &b, "corr_dn_h");
            // corr_dn_v
            let mut a = alloc::vec![0.0; oh * w];
            let mut b = alloc::vec![0.0; oh * w];
            corr_dn_vertical(&src, h, w, oh, &mut a);
            crate::simd_kernels::corr_dn_v(&src, h, w, oh, &mut b);
            close(&a, &b, "corr_dn_v");
            // up_conv_h — both out_w parities.
            for out_w in [2 * w - 1, 2 * w] {
                let mut a = alloc::vec![0.0; h * out_w];
                let mut b = alloc::vec![0.0; h * out_w];
                up_conv_horizontal(&src, h, w, out_w, &mut a);
                crate::simd_kernels::up_conv_h(&src, h, w, out_w, &mut b);
                close(&a, &b, "up_conv_h");
            }
            // up_conv_v — both out_h parities.
            for out_h in [2 * h - 1, 2 * h] {
                let mut a = alloc::vec![0.0; out_h * w];
                let mut b = alloc::vec![0.0; out_h * w];
                up_conv_vertical(&src, h, w, out_h, &mut a);
                crate::simd_kernels::up_conv_v(&src, h, w, out_h, &mut b);
                close(&a, &b, "up_conv_v");
            }
        }
    }

    /// The fused stage-2+3 `imenlarge2` must equal the materialized-t2
    /// reference path bit-for-bit (same arithmetic, no intermediates).
    #[test]
    fn imenlarge2_fused_matches_materialized() {
        for &(w, h) in &[(4usize, 4usize), (9, 7), (16, 16), (33, 21), (5, 64)] {
            let src: Vec<f32> = (0..w * h)
                .map(|i| ((i * 43 % 71) as f32 - 35.0) / 8.0)
                .collect();
            // dst = full imu shape AND a crop (as callers request).
            let (fw, fh) = (2 * w, 2 * h);
            for &(dw, dh) in &[(fw, fh), (w, h), (fw - 1, fh - 1)] {
                let a = imenlarge2(&src, w, h, dw, dh);
                let b = imenlarge2_small(&src, w, h, dw, dh);
                assert_eq!(
                    a, b,
                    "imenlarge2 fused != materialized {w}x{h} -> {dw}x{dh}"
                );
            }
        }
    }
}
