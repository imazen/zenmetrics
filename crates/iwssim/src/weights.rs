//! Information-content weight map — paper §II + §III-B, Python
//! reference's `info_content_weight_map`.
//!
//! For each scale `j < Nsc`:
//!
//! 1. Compute 3×3 box statistics on `(LP_ref, LP_dis)` with **'same'**
//!    padding (zero-padding, like `F.conv2d(..., padding=1)`):
//!    `mean_x`, `mean_y`, `cov_xy`, `ss_x`, `ss_y`.
//! 2. Clamp `ss_x`, `ss_y` ≥ 0 and apply the per-pixel gain-factor
//!    correction:
//!    ```text
//!    g = cov_xy / (ss_x + tol)
//!    vv = ss_y - g * cov_xy
//!    if ss_x < tol: g = 0, vv = ss_y, ss_x = 0
//!    if ss_y < tol: g = 0, vv = 0
//!    ```
//! 3. Build the neighborhood matrix `Y` (rows = pixels in valid region;
//!    cols = `blSzX·blSzY` neighbors + 1 parent band sample).
//! 4. Compute `Cᵤ = Yᵀ Y / nexp`, eigendecompose + PSD-clean +
//!    invert via [`crate::eig`].
//! 5. Per-pixel quadratic form `ss = (Y · Cᵤ_inv) ⊙ Y` summed across
//!    neighborhood; reshape to `(nblv, nblh)`.
//! 6. Crop `g, vv` to the valid-region shape (`Ly..nv-Ly`, `Lx..nh-Lx`)
//!    — for `blSz=3`, `Ly=Lx=1` and the crop drops the 1-pixel border.
//! 7. Per-pixel mutual-info sum over eigenvalues:
//!    `Σ_k log2(1 + ((vv + (1+g²)·σ_n²)·ss·λ_k + σ_n²·vv) / σ_n⁴)`,
//!    clamped at 0 from below.

use alloc::vec::Vec;

use crate::eig::{cov_from_neighborhood, decompose_and_invert};
use crate::params::IwssimParams;
use crate::pyramid::imenlarge2;

/// Tolerance below which `ss_x` / `ss_y` count as zero — matches the
/// Python reference (`tol = 1e-15`).
const TOL: f32 = 1.0e-15;

/// Per-scale info-content weight map.
pub(crate) struct IwMap {
    /// Shape: `(nblv, nblh)` — the valid-region size after applying
    /// the neighborhood-block crop.
    pub h: usize,
    pub w: usize,
    /// Per-pixel information weight, length `h * w`.
    pub infow: Vec<f32>,
}

/// 3×3 mean filter ('same'-padding, zero outside).
///
/// Matches `F.conv2d(x, ones(3,3)/9, padding=1)`. Output shape equals
/// input shape; samples outside the image contribute zero to the sum.
fn box3_same(src: &[f32], h: usize, w: usize, dst: &mut [f32]) {
    debug_assert_eq!(src.len(), h * w);
    debug_assert_eq!(dst.len(), h * w);
    let inv9 = 1.0_f32 / 9.0;
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0_f32;
            for dy in -1..=1i32 {
                let sy = y as i32 + dy;
                if sy < 0 || sy >= h as i32 {
                    continue;
                }
                for dx in -1..=1i32 {
                    let sx = x as i32 + dx;
                    if sx < 0 || sx >= w as i32 {
                        continue;
                    }
                    acc += src[sy as usize * w + sx as usize];
                }
            }
            dst[y * w + x] = acc * inv9;
        }
    }
}

/// Compute per-pixel 3×3 box statistics for `(x, y)`. Mirrors the
/// Python reference's first block in `info_content_weight_map`.
struct BoxStats {
    /// `mean_x`. Length `h*w`.
    mean_x: Vec<f32>,
    /// `mean_y`. Length `h*w`.
    mean_y: Vec<f32>,
    /// `cov_xy = E[xy] − mean_x · mean_y`. Length `h*w`.
    cov_xy: Vec<f32>,
    /// `ss_x = E[x²] − mean_x²`, clamped at 0. Length `h*w`.
    ss_x: Vec<f32>,
    /// `ss_y = E[y²] − mean_y²`, clamped at 0. Length `h*w`.
    ss_y: Vec<f32>,
}

fn box_stats_3x3(x: &[f32], y: &[f32], h: usize, w: usize) -> BoxStats {
    let n = h * w;
    let mut mean_x = alloc::vec![0.0_f32; n];
    let mut mean_y = alloc::vec![0.0_f32; n];
    box3_same(x, h, w, &mut mean_x);
    box3_same(y, h, w, &mut mean_y);

    // Compute element-wise products.
    let mut xx = alloc::vec![0.0_f32; n];
    let mut yy = alloc::vec![0.0_f32; n];
    let mut xy = alloc::vec![0.0_f32; n];
    for i in 0..n {
        xx[i] = x[i] * x[i];
        yy[i] = y[i] * y[i];
        xy[i] = x[i] * y[i];
    }

    let mut e_xx = alloc::vec![0.0_f32; n];
    let mut e_yy = alloc::vec![0.0_f32; n];
    let mut e_xy = alloc::vec![0.0_f32; n];
    box3_same(&xx, h, w, &mut e_xx);
    box3_same(&yy, h, w, &mut e_yy);
    box3_same(&xy, h, w, &mut e_xy);

    let mut cov_xy = alloc::vec![0.0_f32; n];
    let mut ss_x = alloc::vec![0.0_f32; n];
    let mut ss_y = alloc::vec![0.0_f32; n];
    for i in 0..n {
        cov_xy[i] = e_xy[i] - mean_x[i] * mean_y[i];
        ss_x[i] = (e_xx[i] - mean_x[i] * mean_x[i]).max(0.0);
        ss_y[i] = (e_yy[i] - mean_y[i] * mean_y[i]).max(0.0);
    }

    BoxStats {
        mean_x,
        mean_y,
        cov_xy,
        ss_x,
        ss_y,
    }
}

/// Apply the reference's gain-factor correction. Modifies `ss_x`, `g`,
/// `vv` in place using the per-pixel thresholds.
fn gain_correction(stats: &mut BoxStats, g: &mut [f32], vv: &mut [f32]) {
    let n = stats.ss_x.len();
    for i in 0..n {
        let ssx_i = stats.ss_x[i];
        let ssy_i = stats.ss_y[i];
        let cov_i = stats.cov_xy[i];
        // Initial values (pre-correction).
        let mut g_i = cov_i / (ssx_i + TOL);
        let mut vv_i = ssy_i - g_i * cov_i;
        // ss_x < tol → g=0, vv=ss_y, ss_x=0
        let mut ssx_corrected = ssx_i;
        if ssx_i < TOL {
            g_i = 0.0;
            vv_i = ssy_i;
            ssx_corrected = 0.0;
        }
        // ss_y < tol → g=0, vv=0 (applied AFTER the ssx<tol block in
        // the Python reference, so order matters when both thresholds
        // trigger).
        if ssy_i < TOL {
            g_i = 0.0;
            vv_i = 0.0;
        }
        stats.ss_x[i] = ssx_corrected;
        g[i] = g_i;
        vv[i] = vv_i;
    }
}

/// Build the neighborhood matrix `Y`. Output shape `(nexp, big_n)`
/// in row-major order. `nexp = nblv * nblh`, `big_n = block_h * block_w
/// + parent`.
///
/// The Python reference uses `torch.roll` + a hand-coded shift loop.
/// Equivalently: for each pixel `(yy, xx)` in the valid region
/// `[Ly..Ly+nblv, Lx..Lx+nblh]`, collect the `block_h × block_w`
/// neighborhood centered at `(yy, xx)`. Iteration order over `(ny, nx)`
/// must match the Python's `(-Ly..Ly+1, -Lx..Lx+1)` so the column
/// indices in `Y` match what `Cᵤ` is computed on.
fn build_y_matrix(
    img: &[f32],
    parent: Option<&[f32]>,
    h: usize,
    w: usize,
    block_h: usize,
    block_w: usize,
) -> (Vec<f32>, usize, usize, usize) {
    let lx = (block_w - 1) / 2;
    let ly = (block_h - 1) / 2;
    let nblv = h - block_h + 1;
    let nblh = w - block_w + 1;
    let nexp = nblv * nblh;
    let big_n = block_h * block_w + parent.is_some() as usize;
    let mut y = alloc::vec![0.0_f32; nexp * big_n];
    // The Python double-loop order is:
    //   for ny in -Ly..=Ly:
    //       for nx in -Lx..=Lx:
    //           col = next index; Y[:, col] = roll(img, ny=0, nx=1)[Ly:Ly+nblv, Lx:Lx+nblh].flatten()
    // Equivalently: at output pixel (yy, xx) in [Ly..Ly+nblv, Lx..Lx+nblh],
    //               Y[(yy-Ly)*nblh + (xx-Lx), col] = img[yy + ny, xx + nx].
    //
    // i.e. each column is one neighborhood offset; rows iterate over
    // the valid region in flatten() order (row-major).
    let mut col = 0;
    for ny in -(ly as i32)..=(ly as i32) {
        for nx in -(lx as i32)..=(lx as i32) {
            for r in 0..nblv {
                for c in 0..nblh {
                    let yy = (r + ly) as i32 + ny;
                    let xx = (c + lx) as i32 + nx;
                    // yy, xx are guaranteed in [0..h) and [0..w) for
                    // all (r, c, ny, nx) in this iteration — the
                    // valid region is [Ly..Ly+nblv, Lx..Lx+nblh] and
                    // ny/nx ∈ [-Ly..Ly]/[-Lx..Lx].
                    let row_index = r * nblh + c;
                    y[row_index * big_n + col] = img[(yy as usize) * w + (xx as usize)];
                }
            }
            col += 1;
        }
    }
    if let Some(parent_band) = parent {
        // Parent column: just the cropped center patch.
        for r in 0..nblv {
            for c in 0..nblh {
                let yy = r + ly;
                let xx = c + lx;
                let row_index = r * nblh + c;
                y[row_index * big_n + col] = parent_band[yy * w + xx];
            }
        }
    }
    (y, nexp, nblv, nblh)
}

/// Compute `infow` from cropped `(g, vv, ss)` slabs + eigendecomposition.
///
/// Matches the Python:
/// ```text
/// for k in range(N):
///     infow += log2(1 + ((vv + (1 + g²)·σ²)·ss·λ_k + σ²·vv) / σ⁴)
/// infow[infow < tol] = 0
/// ```
fn compute_infow(
    g: &[f32],
    vv: &[f32],
    ss: &[f32],
    lambdas: &[f32],
    sigma_nsq: f32,
) -> Vec<f32> {
    let n = g.len();
    debug_assert_eq!(vv.len(), n);
    debug_assert_eq!(ss.len(), n);
    let s2 = sigma_nsq;
    let s4 = s2 * s2;
    let mut infow = alloc::vec![0.0_f32; n];
    for i in 0..n {
        let g_i = g[i];
        let vv_i = vv[i];
        let ss_i = ss[i];
        let mut acc = 0.0_f32;
        let one_plus_g2 = 1.0 + g_i * g_i;
        let common_num = (vv_i + one_plus_g2 * s2) * ss_i;
        let inv_s4 = 1.0 / s4;
        let sn2_vv = s2 * vv_i;
        for &lam in lambdas {
            let arg = (common_num * lam + sn2_vv) * inv_s4;
            acc += (1.0 + arg).log2();
        }
        // Clamp at 0 (the upstream sets `infow[infow < tol] = 0`).
        if acc >= TOL { infow[i] = acc; } else { infow[i] = 0.0; }
    }
    infow
}

/// Compute the IW weight maps for scales `1..Nsc`. Returns one
/// [`IwMap`] per finer scale (i.e. `Nsc - 1` entries; index `s-1`
/// holds the map for scale `s`).
///
/// `lp` is the per-scale Laplacian band; `g` is the per-scale Gaussian
/// (needed for the parent band via `imenlarge2`).
pub(crate) fn compute_iw_maps(
    lp_ref: &[Vec<f32>],
    lp_dis: &[Vec<f32>],
    g_ref: &[Vec<f32>],
    dims: &[(usize, usize)],
    params: &IwssimParams,
) -> Vec<IwMap> {
    let nsc = lp_ref.len();
    let mut out = Vec::with_capacity(nsc - 1);
    let block_h = params.bl_sz_y as usize;
    let block_w = params.bl_sz_x as usize;
    let parent_enabled = params.parent;
    for s in 0..(nsc - 1) {
        let (w, h) = dims[s];
        let imgo = &lp_ref[s];
        let imgd = &lp_dis[s];

        // 1. 3×3 box statistics with 'same' padding.
        let mut stats = box_stats_3x3(imgo, imgd, h, w);
        let _ = stats.mean_x.len(); // mean_x / mean_y not used downstream
        let _ = stats.mean_y.len();
        let mut g = alloc::vec![0.0_f32; h * w];
        let mut vv = alloc::vec![0.0_f32; h * w];
        gain_correction(&mut stats, &mut g, &mut vv);

        // 2. Build the parent band (if enabled and scale < Nsc-1).
        let prnt = parent_enabled && s < nsc - 2;
        let parent_band: Option<Vec<f32>> = if prnt {
            // imenlarge2(g_ref[s+1]) → (~2W, ~2H) then crop to (h, w).
            let (w_nxt, h_nxt) = dims[s + 1];
            let big = imenlarge2(&g_ref[s + 1], w_nxt, h_nxt, w, h);
            Some(big)
        } else {
            None
        };

        // 3. Build Y matrix in the valid region.
        let (y_matrix, nexp, nblv, nblh) =
            build_y_matrix(imgo, parent_band.as_deref(), h, w, block_h, block_w);
        let big_n = block_h * block_w + prnt as usize;

        // 4. Cᵤ + eigendecomposition.
        let cu = cov_from_neighborhood(&y_matrix, nexp, big_n);
        let eig = decompose_and_invert(&cu, big_n);
        let lambdas = eig.lambdas();
        let c_u_inv = eig.c_u_inv_slice();

        // 5. Per-pixel quadratic form: ss = (Y · Cᵤ_inv) ⊙ Y / N. Then
        //    sum across the N neighborhood entries → scalar per pixel.
        //    Match Python: `(Y @ C_u_inv) * Y / N`, sum axis=1.
        let n_f = big_n as f32;
        let mut ss_pix = alloc::vec![0.0_f32; nexp];
        for row in 0..nexp {
            let y_row = &y_matrix[row * big_n..(row + 1) * big_n];
            // First compute (Y · C_u_inv)[col] = Σ_k Y[row,k] · C_u_inv[k,col].
            // Then dot with Y[row, col] and sum across col.
            // Equivalent inner-product form: Σ_{i,j} Y_i · C_inv[i,j] · Y_j / N.
            let mut acc = 0.0_f32;
            for i in 0..big_n {
                let yi = y_row[i];
                if yi == 0.0 {
                    continue;
                }
                let cinv_row = &c_u_inv[i * big_n..(i + 1) * big_n];
                let mut inner = 0.0_f32;
                for j in 0..big_n {
                    inner += cinv_row[j] * y_row[j];
                }
                acc += yi * inner;
            }
            ss_pix[row] = acc / n_f;
        }

        // 6. Crop g, vv to (nblv, nblh).
        let ly = (block_h - 1) / 2;
        let lx = (block_w - 1) / 2;
        let mut g_c = alloc::vec![0.0_f32; nblv * nblh];
        let mut vv_c = alloc::vec![0.0_f32; nblv * nblh];
        for r in 0..nblv {
            let src_row_off = (r + ly) * w + lx;
            for c in 0..nblh {
                g_c[r * nblh + c] = g[src_row_off + c];
                vv_c[r * nblh + c] = vv[src_row_off + c];
            }
        }

        // 7. infow.
        let infow = compute_infow(&g_c, &vv_c, &ss_pix, lambdas, params.sigma_nsq);

        out.push(IwMap {
            h: nblv,
            w: nblh,
            infow,
        });
    }
    out
}
