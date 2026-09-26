//! Vector-GSM VIF (`vifvec`) — the *original* VIF release
//! (H. R. Sheikh & A. C. Bovik, "Image Information and Visual
//! Quality", IEEE TIP 15(2), 2006; reference code `vifvec_release.zip`,
//! LIVE/UT Austin). **Not** the same metric as [`crate::vif_plane_f32`]
//! (VIFp, the later `vifp_mscale.m` pixel-domain variant): vifvec
//! decomposes both images into a 4-level `sp5Filters` steerable
//! pyramid (matlabPyrTools `buildSpyr`, `reflect1` edges), models each
//! selected subband's 3×3 vector neighbourhoods as a Gaussian-scale-
//! mixture field, and pools a mutual-information ratio over the eight
//! subbands `[4 7 10 13 16 19 22 25]` in the reference's
//! `ind2wtree` cell ordering — i.e. pyramid bands `(level, orientation)`
//! `(4,4) (4,1) (3,4) (3,1) (2,4) (2,1) (1,4) (1,1)` (1-based).
//!
//! This is the implementation family the JPEG AIC-4 published `VIF`
//! column was computed with (`vifvec.m`/`IQA_pytorch.VIFs` on
//! u8-rounded `round(0.299·R + 0.587·G + 0.114·B)` grayscale — this
//! crate reproduces `metrics_fullres.tab` at med |Δ| 1.9e-6 / max
//! 6.3e-6 over the 54 available pairs, 2026-09-26).
//!
//! Faithful-mismatch notes vs. upstream:
//!
//! * `corrDn` here is a direct 2-D correlation with `reflect1`
//!   boundary extension; the reference routes through the pyrTools
//!   MEX. Same arithmetic.
//! * `inv(cu)` is realised as an LU solve per block vector
//!   (`x = cu \ v`, `ss = v·x / 9`) — algebraically identical to the
//!   reference's `inv(cu)·v`, different rounding at ~1e-15 relative.
//! * `eig(cu)` uses cyclic Jacobi sweeps (f64); eigenvalue order is
//!   irrelevant since both sums range over all eigenvalues.
//! * A singular `cu` (degenerate/flat reference) yields `Inf`/`NaN`
//!   `ss` exactly as MATLAB's `inv` warning path does — the score
//!   propagates `NaN` rather than inventing an error.
//!
//! Everything below is `f64` (same discipline as the VIFp kernel).

use alloc::vec::Vec;

/// `M` — vector-GSM block size (`MxM` neighbourhood → `M²`-vector).
const M: usize = 3;
/// Additive noise variance on the *distortion* channel.
const SIGMA_NSQ: f64 = 0.4;
/// `tol` in `vifsub_est_M` — variance floor.
const TOL: f64 = 1e-15;

/// Selected subbands in the reference's order. Each entry is
/// `(level 1-based, band 1-based, winsize, offset)`:
/// `sub = [4 7 10 13 16 19 22 25]` ↔ `(lvl,bnd) =
/// [(4,4),(4,1),(3,4),(3,1),(2,4),(2,1),(1,4),(1,1)]`;
/// `lev = ceil((sub-1)/6)`, `winsize = 2^lev+1`,
/// `offset = ceil((winsize-1)/2 / M)` (see `vifvec.m` /
/// `vifsub_est_M.m`).
const SUBS: [(usize, usize, usize, usize); 8] = [
    (4, 4, 3, 1),
    (4, 1, 3, 1),
    (3, 4, 5, 1),
    (3, 1, 5, 1),
    (2, 4, 9, 2),
    (2, 1, 9, 2),
    (1, 4, 17, 3),
    (1, 1, 17, 3),
];

// ------------------------------------------------------------------
// sp5Filters constants (matlabPyrTools `sp5Filters.m`, verbatim).
// ------------------------------------------------------------------

/// `lo0filt` — 5×5, applied to the input image before level 1.
const LO0: [f64; 25] = [
    0.00341614,
    -0.01551246,
    -0.03848215,
    -0.01551246,
    0.00341614, //
    -0.01551246,
    0.05586982,
    0.15925570,
    0.05586982,
    -0.01551246, //
    -0.03848215,
    0.15925570,
    0.40304148,
    0.15925570,
    -0.03848215, //
    -0.01551246,
    0.05586982,
    0.15925570,
    0.05586982,
    -0.01551246, //
    0.00341614,
    -0.01551246,
    -0.03848215,
    -0.01551246,
    0.00341614,
];

/// `lofilt` — 9×9 = `2 *` the literal in `sp5Filters.m`; the
/// inter-level lowpass (applied at stride 2).
#[allow(clippy::unreadable_literal)]
const LOFILT: [f64; 81] = [
    0.00170808,
    -0.00489834,
    -0.00775624,
    -0.01888864,
    -0.01924108,
    -0.01888864,
    -0.00775624,
    -0.00489834,
    0.00170808,
    -0.00489834,
    -0.01046562,
    -0.01322234,
    0.00821200,
    0.02005976,
    0.00821200,
    -0.01322234,
    -0.01046562,
    -0.00489834,
    -0.00775624,
    -0.01322234,
    0.02793492,
    0.06554076,
    0.07962786,
    0.06554076,
    0.02793492,
    -0.01322234,
    -0.00775624,
    -0.01888864,
    0.00821200,
    0.06554076,
    0.12852666,
    0.16339236,
    0.12852666,
    0.06554076,
    0.00821200,
    -0.01888864,
    -0.01924108,
    0.02005976,
    0.07962786,
    0.16339236,
    0.20193080,
    0.16339236,
    0.07962786,
    0.02005976,
    -0.01924108,
    -0.01888864,
    0.00821200,
    0.06554076,
    0.12852666,
    0.16339236,
    0.12852666,
    0.06554076,
    0.00821200,
    -0.01888864,
    -0.00775624,
    -0.01322234,
    0.02793492,
    0.06554076,
    0.07962786,
    0.06554076,
    0.02793492,
    -0.01322234,
    -0.00775624,
    -0.00489834,
    -0.01046562,
    -0.01322234,
    0.00821200,
    0.02005976,
    0.00821200,
    -0.01322234,
    -0.01046562,
    -0.00489834,
    0.00170808,
    -0.00489834,
    -0.00775624,
    -0.01888864,
    -0.01924108,
    -0.01888864,
    -0.00775624,
    -0.00489834,
    0.00170808,
];

/// `bfilts` — 6 orientation filters, each 49 taps in the .m's
/// column-vector order. `BFILTS[b][i]` for `i = r + 7·c` fills the
/// kernel column-major (MATLAB `reshape(col,7,7)`), so the row-major
/// kernel used by [`corr_dn`] is `k[7r + c] = BFILTS[b][7c + r]`.
/// Only columns 0 and 3 (MATLAB bands 1 and 4) are used by vifvec.
#[allow(clippy::unreadable_literal)]
const BFILTS: [[f64; 49]; 6] = [
    [
        0.00277643,
        0.00496194,
        0.01026699,
        0.01455399,
        0.01026699,
        0.00496194,
        0.00277643,
        -0.00986904,
        -0.00893064,
        0.01189859,
        0.02755155,
        0.01189859,
        -0.00893064,
        -0.00986904,
        -0.01021852,
        -0.03075356,
        -0.08226445,
        -0.11732297,
        -0.08226445,
        -0.03075356,
        -0.01021852,
        0.00000000,
        0.00000000,
        0.00000000,
        0.00000000,
        0.00000000,
        0.00000000,
        0.00000000,
        0.01021852,
        0.03075356,
        0.08226445,
        0.11732297,
        0.08226445,
        0.03075356,
        0.01021852,
        0.00986904,
        0.00893064,
        -0.01189859,
        -0.02755155,
        -0.01189859,
        0.00893064,
        0.00986904,
        -0.00277643,
        -0.00496194,
        -0.01026699,
        -0.01455399,
        -0.01026699,
        -0.00496194,
        -0.00277643,
    ],
    [
        -0.00343249,
        -0.00640815,
        -0.00073141,
        0.01124321,
        0.00182078,
        0.00285723,
        0.01166982,
        -0.00358461,
        -0.01977507,
        -0.04084211,
        -0.00228219,
        0.03930573,
        0.01161195,
        0.00128000,
        0.01047717,
        0.01486305,
        -0.04819057,
        -0.12227230,
        -0.05394139,
        0.00853965,
        -0.00459034,
        0.00790407,
        0.04435647,
        0.09454202,
        -0.00000000,
        -0.09454202,
        -0.04435647,
        -0.00790407,
        0.00459034,
        -0.00853965,
        0.05394139,
        0.12227230,
        0.04819057,
        -0.01486305,
        -0.01047717,
        -0.00128000,
        -0.01161195,
        -0.03930573,
        0.00228219,
        0.04084211,
        0.01977507,
        0.00358461,
        -0.01166982,
        -0.00285723,
        -0.00182078,
        -0.01124321,
        0.00073141,
        0.00640815,
        0.00343249,
    ],
    [
        0.00343249,
        0.00358461,
        -0.01047717,
        -0.00790407,
        -0.00459034,
        0.00128000,
        0.01166982,
        0.00640815,
        0.01977507,
        -0.01486305,
        -0.04435647,
        0.00853965,
        0.01161195,
        0.00285723,
        0.00073141,
        0.04084211,
        0.04819057,
        -0.09454202,
        -0.05394139,
        0.03930573,
        0.00182078,
        -0.01124321,
        0.00228219,
        0.12227230,
        -0.00000000,
        -0.12227230,
        -0.00228219,
        0.01124321,
        -0.00182078,
        -0.03930573,
        0.05394139,
        0.09454202,
        -0.04819057,
        -0.04084211,
        -0.00073141,
        -0.00285723,
        -0.01161195,
        -0.00853965,
        0.04435647,
        0.01486305,
        -0.01977507,
        -0.00640815,
        -0.01166982,
        -0.00128000,
        0.00459034,
        0.00790407,
        0.01047717,
        -0.00358461,
        -0.00343249,
    ],
    [
        -0.00277643,
        0.00986904,
        0.01021852,
        -0.00000000,
        -0.01021852,
        -0.00986904,
        0.00277643,
        -0.00496194,
        0.00893064,
        0.03075356,
        -0.00000000,
        -0.03075356,
        -0.00893064,
        0.00496194,
        -0.01026699,
        -0.01189859,
        0.08226445,
        -0.00000000,
        -0.08226445,
        0.01189859,
        0.01026699,
        -0.01455399,
        -0.02755155,
        0.11732297,
        -0.00000000,
        -0.11732297,
        0.02755155,
        0.01455399,
        -0.01026699,
        -0.01189859,
        0.08226445,
        -0.00000000,
        -0.08226445,
        0.01189859,
        0.01026699,
        -0.00496194,
        0.00893064,
        0.03075356,
        -0.00000000,
        -0.03075356,
        -0.00893064,
        0.00496194,
        -0.00277643,
        0.00986904,
        0.01021852,
        -0.00000000,
        -0.01021852,
        -0.00986904,
        0.00277643,
    ],
    [
        -0.01166982,
        -0.00128000,
        0.00459034,
        0.00790407,
        0.01047717,
        -0.00358461,
        -0.00343249,
        -0.00285723,
        -0.01161195,
        -0.00853965,
        0.04435647,
        0.01486305,
        -0.01977507,
        -0.00640815,
        -0.00182078,
        -0.03930573,
        0.05394139,
        0.09454202,
        -0.04819057,
        -0.04084211,
        -0.00073141,
        -0.01124321,
        0.00228219,
        0.12227230,
        -0.00000000,
        -0.12227230,
        -0.00228219,
        0.01124321,
        0.00073141,
        0.04084211,
        0.04819057,
        -0.09454202,
        -0.05394139,
        0.03930573,
        0.00182078,
        0.00640815,
        0.01977507,
        -0.01486305,
        -0.04435647,
        0.00853965,
        0.01161195,
        0.00285723,
        0.00343249,
        0.00358461,
        -0.01047717,
        -0.00790407,
        -0.00459034,
        0.00128000,
        0.01166982,
    ],
    [
        -0.01166982,
        -0.00285723,
        -0.00182078,
        -0.01124321,
        0.00073141,
        0.00640815,
        0.00343249,
        -0.00128000,
        -0.01161195,
        -0.03930573,
        0.00228219,
        0.04084211,
        0.01977507,
        0.00358461,
        0.00459034,
        -0.00853965,
        0.05394139,
        0.12227230,
        0.04819057,
        -0.01486305,
        -0.01047717,
        0.00790407,
        0.04435647,
        0.09454202,
        -0.00000000,
        -0.09454202,
        -0.04435647,
        -0.00790407,
        0.01047717,
        0.01486305,
        -0.04819057,
        -0.12227230,
        -0.05394139,
        0.00853965,
        -0.00459034,
        -0.00358461,
        -0.01977507,
        -0.04084211,
        -0.00228219,
        0.03930573,
        0.01161195,
        0.00128000,
        -0.00343249,
        -0.00640815,
        -0.00073141,
        0.01124321,
        0.00182078,
        0.00285723,
        0.01166982,
    ],
];

// ------------------------------------------------------------------
// reflect1 boundary + corrDn (matlabPyrTools MEX semantics).
// ------------------------------------------------------------------

/// `reflect1`: mirror through the edge pixels — `-1→1`, `n→n−2`, …
#[inline]
fn reflect1(i: i64, n: i64) -> usize {
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
    debug_assert!((0..n).contains(&k));
    k as usize
}

/// `corrDn(im, filt, 'reflect1', [step step], [start start])` with a
/// 0-based `start` and the MEX default `stop = size(im)` (1-based) —
/// i.e. positions `start, start+step, …` ≤ `h-1`/`w-1` per axis.
///
/// `filt` is `fh × fw` row-major; its origin is `floor(f/2)` per axis
/// (the pyrTools convention). Returns `(map, out_w, out_h)`.
#[allow(clippy::too_many_arguments)] // mirrors corrDn(im, filt, edges, da, db)
fn corr_dn(
    img: &[f64],
    w: usize,
    h: usize,
    filt: &[f64],
    fw: usize,
    fh: usize,
    step: usize,
    start: usize,
) -> (Vec<f64>, usize, usize) {
    debug_assert!(start < w && start < h);
    let out_w = (w - 1 - start) / step + 1;
    let out_h = (h - 1 - start) / step + 1;
    let cx = fw / 2;
    let cy = fh / 2;
    let mut out = alloc::vec![0.0; out_w * out_h];
    for oy in 0..out_h {
        let yc = (start + oy * step) as i64;
        for ox in 0..out_w {
            let xc = (start + ox * step) as i64;
            let mut acc = 0.0f64;
            for ky in 0..fh {
                let iy = reflect1(yc + ky as i64 - cy as i64, h as i64);
                let row = &img[iy * w..iy * w + w];
                let frow = &filt[ky * fw..ky * fw + fw];
                for (kx, &fv) in frow.iter().enumerate() {
                    let ix = reflect1(xc + kx as i64 - cx as i64, w as i64);
                    acc += fv * row[ix];
                }
            }
            out[oy * out_w + ox] = acc;
        }
    }
    (out, out_w, out_h)
}

/// A steerable-pyramid band (dense `w*h` f64, row-major).
struct Band {
    w: usize,
    h: usize,
    data: Vec<f64>,
}

/// Build the four-level `sp5Filters` steerable pyramid of `img` and
/// return only the two orientations vifvec consumes:
/// `bands[level-1][0]` = MATLAB band 1, `[1]` = band 4
/// (`buildSpyrLevs` computes `corrDn(lo, bfilt_b)` for each of the six
/// orientations plus a stride-2 `lofilt` recursion; `hi0` and the
/// residual `lo4` are never referenced by `vifvec` and are skipped).
fn build_bands(img: &[f64], w: usize, h: usize) -> [[Band; 2]; 4] {
    let (lo, lw, lh) = corr_dn(img, w, h, &LO0, 5, 5, 1, 0);
    let (mut lo, mut lw, mut lh) = (lo, lw, lh);

    // Row-major 7×7 kernels for MATLAB bands 1 and 4 (columns 0 and 3
    // of the 49×6 `bfilts` matrix), per the reshape note on `BFILTS`.
    let kb = |b: usize| -> [f64; 49] {
        let mut k = [0.0; 49];
        for r in 0..7 {
            for c in 0..7 {
                k[r * 7 + c] = BFILTS[b][r + 7 * c];
            }
        }
        k
    };
    let (k1, k4) = (kb(0), kb(3));

    let mut bands: [[Option<Band>; 2]; 4] = [const { [None, None] }; 4];
    for (level, pair) in bands.iter_mut().enumerate() {
        let (b1, _, _) = corr_dn(&lo, lw, lh, &k1, 7, 7, 1, 0);
        let (b4, _, _) = corr_dn(&lo, lw, lh, &k4, 7, 7, 1, 0);
        pair[0] = Some(Band {
            w: lw,
            h: lh,
            data: b1,
        });
        pair[1] = Some(Band {
            w: lw,
            h: lh,
            data: b4,
        });
        if level < 3 {
            // `lo = corrDn(lo0, lofilt, edges, [2 2], [1 1])` — start
            // 1-based 1 → 0-based 0; default stop = last index.
            let (nlo, nw, nh) = corr_dn(&lo, lw, lh, &LOFILT, 9, 9, 2, 0);
            (lo, lw, lh) = (nlo, nw, nh);
        }
    }
    bands.map(|pair| {
        let [a, b] = pair;
        [a.unwrap(), b.unwrap()]
    })
}

// ------------------------------------------------------------------
// LU solve (partial pivoting) + Jacobi eigendecomposition — f64, n×n.
// ------------------------------------------------------------------

/// Solve `A·x = b` for an `n×n` row-major `A` via LU with partial
/// pivoting (same algorithm MATLAB's `inv`/`\` route through).
/// Singular matrices produce `Inf`/`NaN` components — matching the
/// reference's warned `inv` output.
fn lu_solve(a: &mut [f64], b: &mut [f64], n: usize) {
    // In-place LU (row-major): L unit-diagonal below, U on/above.
    for k in 0..n {
        // Partial pivot.
        let mut piv = k;
        let mut best = a[k * n + k].abs();
        for i in (k + 1)..n {
            let v = a[i * n + k].abs();
            if v > best {
                best = v;
                piv = i;
            }
        }
        if piv != k {
            for j in 0..n {
                a.swap(k * n + j, piv * n + j);
            }
            b.swap(k, piv);
        }
        let akk = a[k * n + k];
        for i in (k + 1)..n {
            let f = a[i * n + k] / akk;
            a[i * n + k] = f;
            for j in (k + 1)..n {
                a[i * n + j] -= f * a[k * n + j];
            }
        }
    }
    // Forward substitution (unit L), then back substitution (U).
    for i in 1..n {
        for j in 0..i {
            b[i] -= a[i * n + j] * b[j];
        }
    }
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= a[i * n + j] * b[j];
        }
        b[i] = s / a[i * n + i];
    }
}

/// Eigenvalues of a symmetric `n×n` row-major matrix via cyclic
/// Jacobi rotations (f64). Returned unsorted — the caller sums over
/// all of them, so ordering is immaterial.
fn eig_symmetric(a: &[f64], n: usize) -> Vec<f64> {
    let mut m = a.to_vec();
    for _sweep in 0..64 {
        // Largest off-diagonal magnitude.
        let mut off = 0.0f64;
        for i in 0..n {
            for j in (i + 1)..n {
                off = off.max(m[i * n + j].abs());
            }
        }
        if off <= 1e-30 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = m[p * n + q];
                if apq == 0.0 {
                    continue;
                }
                let app = m[p * n + p];
                let aqq = m[q * n + q];
                let theta = (aqq - app) / (2.0 * apq);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (1.0 + theta * theta).sqrt())
                } else {
                    1.0 / (theta - (1.0 + theta * theta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                for i in 0..n {
                    if i == p || i == q {
                        continue;
                    }
                    let aip = m[i * n + p];
                    let aiq = m[i * n + q];
                    m[i * n + p] = c * aip - s * aiq;
                    m[p * n + i] = m[i * n + p];
                    m[i * n + q] = s * aip + c * aiq;
                    m[q * n + i] = m[i * n + q];
                }
                m[p * n + p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
                m[q * n + q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
                m[p * n + q] = 0.0;
                m[q * n + p] = 0.0;
            }
        }
    }
    (0..n).map(|i| m[i * n + i]).collect()
}

// ------------------------------------------------------------------
// refparams_vecgsm + vifsub_est_M
// ------------------------------------------------------------------

/// `refparams_vecgsm` for one subband: returns `(ss, lambda)` where
/// `ss` is the `(h/M)×(w/M)` S-field map and `lambda` the `M²`
/// eigenvalues of the vector covariance. `y` is cropped to a multiple
/// of `M` first (`floor(size/M)*M`), as the reference does.
fn refparams_vecgsm(y: &[f64], w: usize, h: usize) -> (Vec<f64>, Vec<f64>, usize, usize) {
    let cw = (w / M) * M;
    let ch = (h / M) * M;
    let n2 = M * M;
    // Overlapping M×M neighbourhoods: all (ch−M+1)×(cw−M+1) positions,
    // block-vector index `i = c·M + r` (column-major, matching the
    // reference's `j`-outer / `k`-inner cat order).
    let np = (ch - (M - 1)) * (cw - (M - 1));
    let mut mean = alloc::vec![0.0f64; n2];
    for ty in 0..(ch - (M - 1)) {
        for tx in 0..(cw - (M - 1)) {
            for c in 0..M {
                for r in 0..M {
                    mean[c * M + r] += y[(ty + r) * w + tx + c];
                }
            }
        }
    }
    for m in &mut mean {
        *m /= np as f64;
    }
    let mut cu = alloc::vec![0.0f64; n2 * n2];
    for ty in 0..(ch - (M - 1)) {
        for tx in 0..(cw - (M - 1)) {
            let mut v = [0.0f64; 9];
            for c in 0..M {
                for r in 0..M {
                    v[c * M + r] = y[(ty + r) * w + tx + c] - mean[c * M + r];
                }
            }
            for i in 0..n2 {
                for j in i..n2 {
                    cu[i * n2 + j] += v[i] * v[j];
                }
            }
        }
    }
    for i in 0..n2 {
        for j in i..n2 {
            cu[i * n2 + j] /= np as f64;
            cu[j * n2 + i] = cu[i * n2 + j];
        }
    }
    let lambda = eig_symmetric(&cu, n2);

    // Non-overlapping blocks → S field: `ss = sum((inv(cu)·v) .* v)/M²`
    // realised as an LU solve `x = cu \ v` + dot product.
    let nbw = cw / M;
    let nbh = ch / M;
    let mut ss = alloc::vec![0.0f64; nbw * nbh];
    for by in 0..nbh {
        for bx in 0..nbw {
            let mut v = [0.0f64; 9];
            for c in 0..M {
                for r in 0..M {
                    v[c * M + r] = y[(by * M + r) * w + bx * M + c];
                }
            }
            let mut x = v;
            let mut lu = cu.clone();
            lu_solve(&mut lu, &mut x, n2);
            let mut s = 0.0;
            for i in 0..n2 {
                s += v[i] * x[i];
            }
            ss[by * nbw + bx] = s / (n2 as f64);
        }
    }
    (ss, lambda, nbw, nbh)
}

/// `vifsub_est_M` for one subband pair: `(g, vv)` maps of size
/// `(h/M)×(w/M)` (the `corrDn` output grid — 1-based centres
/// `2,5,8,… ≤ size−1` ↔ 0-based `1+3k`). `winsize` follows the
/// reference's per-subband table (`2^lev+1`).
fn vifsub_est(
    y: &[f64],
    yn: &[f64],
    w: usize,
    h: usize,
    winsize: usize,
) -> (Vec<f64>, Vec<f64>, usize, usize) {
    let cw = (w / M) * M;
    let ch = (h / M) * M;
    let n = (winsize * winsize) as f64;
    let cx0 = winsize / 2;

    let nbw = cw / M;
    let nbh = ch / M;
    let mut g = alloc::vec![0.0f64; nbw * nbh];
    let mut vv = alloc::vec![0.0f64; nbw * nbh];
    for oy in 0..nbh {
        let cy = (1 + oy * M) as i64; // 0-based centre (1-based 2,5,8,…)
        for ox in 0..nbw {
            let cx = (1 + ox * M) as i64;
            let (mut sx, mut sy, mut sxy, mut sx2, mut sy2) = (0.0, 0.0, 0.0, 0.0, 0.0);
            for ky in 0..winsize {
                let iy = reflect1(cy + ky as i64 - cx0 as i64, ch as i64);
                let rowr = &y[iy * w..];
                let rowd = &yn[iy * w..];
                for kx in 0..winsize {
                    let ix = reflect1(cx + kx as i64 - cx0 as i64, cw as i64);
                    let (a, b) = (rowr[ix], rowd[ix]);
                    sx += a;
                    sy += b;
                    sxy += a * b;
                    sx2 += a * a;
                    sy2 += b * b;
                }
            }
            // corrDn with win = ones/N (means) and win = ones (raw
            // sums), combined exactly as the reference does.
            let mx = sx / n;
            let my = sy / n;
            let cov_xy = sxy - n * mx * my;
            let mut ss_x = sx2 - n * mx * mx;
            let mut ss_y = sy2 - n * my * my;
            if ss_x < 0.0 {
                ss_x = 0.0;
            }
            if ss_y < 0.0 {
                ss_y = 0.0;
            }
            let mut gi = cov_xy / (ss_x + TOL);
            let mut vvi = (ss_y - gi * cov_xy) / n;
            // The reference's masking chain, in order.
            if ss_x < TOL {
                gi = 0.0;
                vvi = ss_y;
            }
            ss_x = if ss_x < TOL { 0.0 } else { ss_x };
            let _ = ss_x;
            if ss_y < TOL {
                gi = 0.0;
                vvi = 0.0;
            }
            if gi < 0.0 {
                vvi = ss_y;
                gi = 0.0;
            }
            if vvi <= TOL {
                vvi = TOL;
            }
            g[oy * nbw + ox] = gi;
            vv[oy * nbw + ox] = vvi;
        }
    }
    (g, vv, nbw, nbh)
}

// ------------------------------------------------------------------
// Driver.
// ------------------------------------------------------------------

/// Vector-GSM VIF of two `w×h` f64 planes (0–255 signal scale).
/// Returns `NaN` when the reference would (empty/degenerate fields).
pub(crate) fn vifvec_core(r: &[f64], d: &[f64], w: usize, h: usize) -> f64 {
    let rb = build_bands(r, w, h);
    let db = build_bands(d, w, h);

    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for &(level, band, winsize, offset) in &SUBS {
        let y = &rb[level - 1][usize::from(band == 4)];
        let yn = &db[level - 1][usize::from(band == 4)];
        let (ss, lambda, nbw, nbh) = refparams_vecgsm(&y.data, y.w, y.h);
        let (g, vv, gw, gh) = vifsub_est(&y.data, &yn.data, y.w, y.h, winsize);
        debug_assert_eq!((gw, gh), (nbw, nbh));

        // Crop `offset` entries on every side of the block maps
        // (reference: `g(offset+1:end-offset, …)`).
        let (iw, ih) = (nbw - 2 * offset, nbh - 2 * offset);
        for &lam in &lambda {
            let mut t1 = 0.0f64;
            let mut t2 = 0.0f64;
            for by in 0..ih {
                for bx in 0..iw {
                    let i = (by + offset) * nbw + bx + offset;
                    let (gi, vi, si) = (g[i], vv[i], ss[i]);
                    t1 += libm::log2(1.0 + gi * gi * si * lam / (vi + SIGMA_NSQ));
                    t2 += libm::log2(1.0 + si * lam / SIGMA_NSQ);
                }
            }
            num += t1;
            den += t2;
        }
    }
    num / den
}
