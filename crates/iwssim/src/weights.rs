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

use crate::eig::decompose_and_invert;
use crate::params::IwssimParams;
use crate::pyramid::imenlarge2;

/// Tolerance below which `ss_x` / `ss_y` count as zero — matches the
/// Python reference (`tol = 1e-15`).
pub(crate) const TOL: f32 = 1.0e-15;

/// Neighborhood taps in the Python reference's column order:
/// `ny` in `-Ly..=Ly` outer, `nx` in `-Lx..=Lx` inner. For `blSz=3`
/// this is the 3×3 block; an optional parent column (index
/// `taps.len()`) reads `parent` at the center position instead of an
/// `img` offset — handled by the callers, not this list.
pub(crate) fn tap_offsets(block_h: usize, block_w: usize) -> Vec<(i32, i32)> {
    let ly = (block_h - 1) / 2;
    let lx = (block_w - 1) / 2;
    let mut v = Vec::with_capacity(block_h * block_w);
    for ny in -(ly as i32)..=(ly as i32) {
        for nx in -(lx as i32)..=(lx as i32) {
            v.push((ny, nx));
        }
    }
    v
}

/// Accumulate the neighborhood Gram matrix `Σ_p y yᵀ` over an image
/// region directly — the `Yᵀ·Y` of [`crate::eig::cov_from_neighborhood`]
/// without materializing `Y`.
///
/// Region semantics: output pixel `(r, c)` has its neighborhood
/// *center* at image coordinate `(row0 + r, col0 + c)`; tap `(dy, dx)`
/// reads `img[(row0 + r + dy)·stride + col0 + c + dx]`. The optional
/// parent column (last Gram row/col) reads `parent[center]`.
///
/// `gram` is `big_n²` (`big_n = taps.len() + parent.is_some()`),
/// caller-initialized — accumulate-into so the strip path can fold
/// per-strip contributions across calls. Every `gram[i][j]`
/// accumulator receives the same product sequence in the same order
/// as `cov_from_neighborhood`'s per-(i,j) column scan, so its values
/// are bit-identical to building `Y` and scanning columns. Returns
/// the number of pixels accumulated (the `nexp` divisor).
///
/// Only the **upper triangle** is written; call [`gram_mirror`] once
/// accumulation is complete to fill the lower half (the strip path
/// mirrors after all strips have folded in).
pub(crate) fn gram_accumulate(
    img: &[f32],
    parent: Option<&[f32]>,
    stride: usize,
    row0: usize,
    col0: usize,
    nrows: usize,
    ncols: usize,
    taps: &[(i32, i32)],
    gram: &mut [f64],
) -> usize {
    let nb = taps.len();
    let big_n = nb + parent.is_some() as usize;
    debug_assert_eq!(gram.len(), big_n * big_n);
    debug_assert!(big_n <= 16);
    // Band rows above the parallel threshold: each band accumulates
    // its own Gram and the partials merge in band order — the result
    // is deterministic at any thread count (re-associated vs a flat
    // accumulation, ulp-level drift only). Single-band images take
    // the direct path bit-identically.
    let n = nrows * ncols;
    let nb_bands = crate::par::n_bands(n);
    if nb_bands == 1 {
        gram_accumulate_rows(img, parent, stride, row0, col0, nrows, ncols, taps, gram);
        return n;
    }
    let sz = nrows.div_ceil(nb_bands);
    let parts = crate::par::collect_bands(nb_bands, |b| {
        let lo = b * sz;
        let hi = (lo + sz).min(nrows);
        let mut g_b = alloc::vec![0.0_f64; big_n * big_n];
        if lo < hi {
            gram_accumulate_rows(
                img,
                parent,
                stride,
                row0 + lo,
                col0,
                hi - lo,
                ncols,
                taps,
                &mut g_b,
            );
        }
        g_b
    });
    for p in &parts {
        for (g, &v) in gram.iter_mut().zip(p.iter()) {
            *g += v;
        }
    }
    n
}

/// Serial inner of [`gram_accumulate`] over one row range — dispatches
/// to the f64x4 kernel (upper-triangle outer product; `gram_mirror`
/// fills the lower half). The kernel's `a*y+g` lanes keep the scalar
/// two-rounding mul-then-add semantics — bit-identical element order.
fn gram_accumulate_rows(
    img: &[f32],
    parent: Option<&[f32]>,
    stride: usize,
    row0: usize,
    col0: usize,
    nrows: usize,
    ncols: usize,
    taps: &[(i32, i32)],
    gram: &mut [f64],
) {
    let nb = taps.len();
    let big_n = nb + parent.is_some() as usize;
    debug_assert_eq!(gram.len(), big_n * big_n);
    debug_assert!(big_n <= 16);
    archmage::incant!(
        crate::simd_kernels::gram_rows_inner(
            img, parent, stride, row0, col0, nrows, ncols, taps, gram
        ),
        [v3, neon, wasm128, scalar]
    );
}

/// Fill `gram`'s lower triangle from its upper triangle in place.
pub(crate) fn gram_mirror(gram: &mut [f64], big_n: usize) {
    debug_assert_eq!(gram.len(), big_n * big_n);
    for i in 0..big_n {
        for j in (i + 1)..big_n {
            gram[j * big_n + i] = gram[i * big_n + j];
        }
    }
}

/// Per-pixel quadratic form `ss = (Y·Cᵤ_inv) ⊙ Y / N` evaluated as a
/// dense stencil directly on `img` — identical math to iterating rows
/// of a materialized `Y`, without the `nexp × big_n` intermediate.
///
/// Region/parameter semantics match [`gram_accumulate`]. `out` gets
/// `nrows × ncols` values in row order; `cinv` is `big_n²` row-major.
///
/// Test-only retained oracle: production paths use the dispatched
/// [`crate::simd_kernels::quad_form_rows`].
#[cfg(test)]
pub(crate) fn quad_form_into(
    img: &[f32],
    parent: Option<&[f32]>,
    stride: usize,
    row0: usize,
    col0: usize,
    nrows: usize,
    ncols: usize,
    taps: &[(i32, i32)],
    cinv: &[f32],
    out: &mut [f32],
) {
    let nb = taps.len();
    let big_n = nb + parent.is_some() as usize;
    debug_assert_eq!(cinv.len(), big_n * big_n);
    debug_assert_eq!(out.len(), nrows * ncols);
    debug_assert!(big_n <= 16);
    let n_f = big_n as f32;
    for r in 0..nrows {
        let row_c = row0 + r;
        for c in 0..ncols {
            let col_c = col0 + c;
            let mut yv = [0.0_f32; 16];
            for (k, &(dy, dx)) in taps.iter().enumerate() {
                yv[k] = img[((row_c as i32 + dy) as usize) * stride + (col_c as i32 + dx) as usize];
            }
            if let Some(p) = parent {
                yv[nb] = p[row_c * stride + col_c];
            }
            // Same nested structure as the Y-row loop. The original
            // skips `yi == 0`; `yi·inner` contributes ±0 in that case
            // so dropping the branch changes no result bits.
            let mut acc = 0.0_f32;
            for i in 0..big_n {
                let cinv_row = &cinv[i * big_n..(i + 1) * big_n];
                let mut inner = 0.0_f32;
                for j in 0..big_n {
                    inner += cinv_row[j] * yv[j];
                }
                acc += yv[i] * inner;
            }
            out[r * ncols + c] = acc / n_f;
        }
    }
}

/// Per-scale info-content weight map.
pub(crate) struct IwMap {
    /// Shape: `(nblv, nblh)` — the valid-region size after applying
    /// the neighborhood-block crop.
    pub h: usize,
    pub w: usize,
    /// Per-pixel information weight, length `h * w`.
    pub infow: Vec<f32>,
}

/// Compute `infow` from cropped `(g, vv, ss)` slabs + eigendecomposition.
///
/// Matches the Python:
/// ```text
/// for k in range(N):
///     infow += log2(1 + ((vv + (1 + g²)·σ²)·ss·λ_k + σ²·vv) / σ⁴)
/// infow[infow < tol] = 0
/// ```
fn compute_infow(g: &[f32], vv: &[f32], ss: &[f32], lambdas: &[f32], sigma_nsq: f32) -> Vec<f32> {
    let n = g.len();
    debug_assert_eq!(vv.len(), n);
    debug_assert_eq!(ss.len(), n);
    let mut infow = alloc::vec![0.0_f32; n];
    crate::simd_kernels::infow_map(g, vv, ss, lambdas, sigma_nsq, TOL, &mut infow);
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

        // 1. 3×3 box statistics + gain correction, fused into a single
        //    pass (`box_gain_rows`) — no mean/E[·] intermediates are
        //    materialized. Zero-pad 'same' borders, then `g`, `vv`
        //    cropped to the valid region below.
        let mut g = alloc::vec![0.0_f32; h * w];
        let mut vv = alloc::vec![0.0_f32; h * w];
        crate::simd_kernels::box_gain_rows(imgo, imgd, h, w, 0, h, &mut g, &mut vv);

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

        // 3. Neighborhood taps + valid region (no materialized Y —
        //    the Gram and quadratic form read `imgo` as a stencil).
        let taps = tap_offsets(block_h, block_w);
        let ly = (block_h - 1) / 2;
        let lx = (block_w - 1) / 2;
        let nblv = h - block_h + 1;
        let nblh = w - block_w + 1;
        let nexp = nblv * nblh;
        let big_n = block_h * block_w + prnt as usize;

        // 4. Cᵤ = Σ_p y yᵀ / nexp + eigendecomposition — bit-identical
        //    to the old `cov_from_neighborhood(Y)` path.
        let mut gram = alloc::vec![0.0_f64; big_n * big_n];
        gram_accumulate(
            imgo,
            parent_band.as_deref(),
            w,
            ly,
            lx,
            nblv,
            nblh,
            &taps,
            &mut gram,
        );
        gram_mirror(&mut gram, big_n);
        let nexp_f = nexp as f64;
        for v in &mut gram {
            *v /= nexp_f;
        }
        let eig = decompose_and_invert(&gram, big_n);
        let lambdas = eig.lambdas();
        let c_u_inv = eig.c_u_inv_slice();

        // 5. Per-pixel quadratic form: ss = (Y · Cᵤ_inv) ⊙ Y / N as a
        //    dense stencil on `imgo` — same sums, no Y. SIMD via
        //    `quad_form_rows` (8-wide over output columns).
        let mut ss_pix = alloc::vec![0.0_f32; nexp];
        crate::simd_kernels::quad_form_rows(
            imgo,
            parent_band.as_deref(),
            w,
            ly,
            lx,
            nblv,
            nblh,
            &taps,
            c_u_inv,
            &mut ss_pix,
        );

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

#[cfg(test)]
mod tests {
    use super::*;

    /// Materialize the Y neighborhood matrix the pre-stencil path
    /// built — kept here as the ground truth for `gram_accumulate`
    /// and the quad-form kernels.
    fn build_y_oracle(
        img: &[f32],
        parent: Option<&[f32]>,
        stride: usize,
        row0: usize,
        col0: usize,
        nrows: usize,
        ncols: usize,
        taps: &[(i32, i32)],
    ) -> (Vec<f32>, usize) {
        let nb = taps.len();
        let big_n = nb + parent.is_some() as usize;
        let mut y = alloc::vec![0.0_f32; nrows * ncols * big_n];
        for r in 0..nrows {
            for c in 0..ncols {
                let row_c = row0 + r;
                let col_c = col0 + c;
                let base = (r * ncols + c) * big_n;
                for (k, &(dy, dx)) in taps.iter().enumerate() {
                    y[base + k] =
                        img[((row_c as i32 + dy) as usize) * stride + (col_c as i32 + dx) as usize];
                }
                if let Some(p) = parent {
                    y[base + nb] = p[row_c * stride + col_c];
                }
            }
        }
        (y, big_n)
    }

    /// `gram_accumulate` + `gram_mirror` + `/nexp` must equal
    /// `cov_from_neighborhood(Y)` **bit-identically** — the helpers
    /// preserve the reference per-(i,j) accumulation order.
    #[test]
    fn gram_matches_materialized_covariance() {
        let (h, w) = (12usize, 10usize);
        let img: Vec<f32> = (0..h * w)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 13.0)
            .collect();
        let parent: Vec<f32> = (0..h * w).map(|i| ((i * 53 % 89) as f32) / 31.0).collect();
        let (block_h, block_w) = (3usize, 3usize);
        let taps = tap_offsets(block_h, block_w);
        let (ly, lx) = (1usize, 1usize);
        let (nrows, ncols) = (h - block_h + 1, w - block_w + 1);
        let nexp = nrows * ncols;

        for parent in [None, Some(parent.as_slice())] {
            let (y, big_n) = build_y_oracle(&img, parent, w, ly, lx, nrows, ncols, &taps);
            let oracle = crate::eig::cov_from_neighborhood(&y, nexp, big_n);

            let mut gram = alloc::vec![0.0_f64; big_n * big_n];
            let got_nexp = gram_accumulate(&img, parent, w, ly, lx, nrows, ncols, &taps, &mut gram);
            assert_eq!(got_nexp, nexp);
            gram_mirror(&mut gram, big_n);
            for v in &mut gram {
                *v /= nexp as f64;
            }
            assert_eq!(gram, oracle, "gram != cov_from_neighborhood(Y)");
        }
    }

    /// The dispatched `quad_form_rows` must match the scalar
    /// `quad_form_into` oracle within FMA tolerance (the SIMD kernel
    /// contracts `a*b+c` into `mul_add` — last-bit drift allowed).
    #[test]
    fn quad_form_rows_matches_scalar_oracle() {
        let (h, w) = (17usize, 13usize);
        let img: Vec<f32> = (0..h * w)
            .map(|i| ((i * 41 % 97) as f32 - 48.0) / 11.0)
            .collect();
        let parent: Vec<f32> = (0..h * w).map(|i| ((i * 29 % 83) as f32) / 37.0).collect();
        let (block_h, block_w) = (3usize, 3usize);
        let taps = tap_offsets(block_h, block_w);
        let (ly, lx) = (1usize, 1usize);
        let (nrows, ncols) = (h - block_h + 1, w - block_w + 1);
        let nexp = nrows * ncols;

        for parent in [None, Some(parent.as_slice())] {
            let big_n = 9 + parent.is_some() as usize;
            // Non-symmetric, non-integer cinv to exercise all lanes.
            let cinv: Vec<f32> = (0..big_n * big_n)
                .map(|i| ((i * 17 % 43) as f32 - 21.0) / 7.0)
                .collect();

            let mut scalar = alloc::vec![0.0_f32; nexp];
            quad_form_into(
                &img,
                parent,
                w,
                ly,
                lx,
                nrows,
                ncols,
                &taps,
                &cinv,
                &mut scalar,
            );
            let mut simd = alloc::vec![0.0_f32; nexp];
            crate::simd_kernels::quad_form_rows(
                &img, parent, w, ly, lx, nrows, ncols, &taps, &cinv, &mut simd,
            );
            for (k, (&a, &b)) in scalar.iter().zip(simd.iter()).enumerate() {
                let tol = 1e-5 * a.abs().max(1.0);
                assert!(
                    (a - b).abs() <= tol,
                    "quad_form mismatch at {k}: scalar={a} simd={b}"
                );
            }
        }
    }

    /// Above `par::PAR_MIN_SAMPLES` the Gram accumulates per-band
    /// partials merged in band order — deterministic, but re-associated
    /// vs the flat scan, so the oracle comparison gets a reassociation
    /// tolerance instead of bit-equality.
    #[test]
    fn gram_banded_matches_oracle() {
        let (h, w) = (420usize, 400usize);
        let img: Vec<f32> = (0..h * w)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 13.0)
            .collect();
        let (block_h, block_w) = (3usize, 3usize);
        let taps = tap_offsets(block_h, block_w);
        let (nrows, ncols) = (h - block_h + 1, w - block_w + 1);
        assert!(nrows * ncols > crate::par::PAR_MIN_SAMPLES);

        let (y, big_n) = build_y_oracle(&img, None, w, 1, 1, nrows, ncols, &taps);
        let oracle = crate::eig::cov_from_neighborhood(&y, nrows * ncols, big_n);

        let mut gram = alloc::vec![0.0_f64; big_n * big_n];
        gram_accumulate(&img, None, w, 1, 1, nrows, ncols, &taps, &mut gram);
        gram_mirror(&mut gram, big_n);
        for v in &mut gram {
            *v /= (nrows * ncols) as f64;
        }
        for (k, (&a, &b)) in gram.iter().zip(oracle.iter()).enumerate() {
            let tol = 1e-10 * a.abs().max(1.0);
            assert!((a - b).abs() <= tol, "banded gram[{k}]: {a} vs oracle {b}");
        }
    }

    /// `weighted_sum_pair`'s banded fold must be bit-identical to
    /// splitting the input at the same boundaries and folding in order
    /// (determinism), and within f32-lane-accumulation noise of a
    /// strict f64 reference.
    #[test]
    fn weighted_sum_pair_banded() {
        let n = crate::par::PAR_MIN_SAMPLES + 40_000;
        let cs: Vec<f32> = (0..n).map(|i| ((i * 13 % 89) as f32) / 97.0).collect();
        let iw: Vec<f32> = (0..n).map(|i| ((i * 29 % 61) as f32) / 43.0).collect();
        let (a0, a1) = crate::simd_kernels::weighted_sum_pair(&cs, &iw);

        // Manual split at the same band boundaries — bit-identical.
        let nb = crate::par::n_bands(n);
        let sz = n.div_ceil(nb);
        let (mut m0, mut m1) = (0.0_f64, 0.0_f64);
        for b in 0..nb {
            let lo = b * sz;
            let hi = (lo + sz).min(n);
            if lo >= hi {
                continue;
            }
            let (p0, p1) = crate::simd_kernels::weighted_sum_pair(&cs[lo..hi], &iw[lo..hi]);
            m0 += p0;
            m1 += p1;
        }
        assert_eq!(a0, m0);
        assert_eq!(a1, m1);

        // Strict f64 reference — f32-lane noise bound.
        let (mut e0, mut e1) = (0.0_f64, 0.0_f64);
        for i in 0..n {
            e0 += (cs[i] as f64) * (iw[i] as f64);
            e1 += iw[i] as f64;
        }
        assert!((a0 - e0).abs() <= 1e-4 * e0.abs().max(1.0), "{a0} vs {e0}");
        assert!((a1 - e1).abs() <= 1e-4 * e1.abs().max(1.0), "{a1} vs {e1}");
    }
}
