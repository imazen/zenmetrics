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

// `libm::log2` lanes moved to zenmetrics-math (`fdlibm64::log2_libm_4x`).

// ------------------------------------------------------------------
// sp5Filters constants (matlabPyrTools `sp5Filters.m`, verbatim; MIT,
// Copyright (c) 2015 LabForComputationalVision, see LICENSE-matlabPyrTools).
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
    let mut out = alloc::vec![0.0; out_w * out_h];
    let corr_row = |oy: usize, orow: &mut [f64]| {
        let yc = start + oy * step;
        archmage::attuned!(
            corr_row_tier(img, w, h, filt, fw, fh, step, start, yc, orow),
            [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
        );
    };
    #[cfg(feature = "parallel")]
    let mut ran_par = false;
    #[cfg(not(feature = "parallel"))]
    let ran_par = false;
    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 {
        use rayon::prelude::*;
        out.par_chunks_mut(out_w)
            .enumerate()
            .for_each(|(oy, orow)| corr_row(oy, orow));
        ran_par = true;
    }
    let _ = ran_par;
    if !ran_par {
        for oy in 0..out_h {
            corr_row(oy, &mut out[oy * out_w..(oy + 1) * out_w]);
        }
    }
    (out, out_w, out_h)
}

// Consecutive output lanes preserve the original ky/kx accumulation order.
// Only the narrow boundary strips need reflected horizontal addressing.
macro_rules! corr_row_body {
    ($token:ident, $V:ty, $lanes:literal, $img:ident, $w:ident, $h:ident,
     $filt:ident, $fw:ident, $fh:ident, $step:ident, $start:ident, $yc:ident, $out:ident) => {{
        let (cx, cy) = ($fw / 2, $fh / 2);
        let scalar = |ox: usize| {
            let xc = ($start + ox * $step) as i64;
            let mut acc = 0.0;
            for ky in 0..$fh {
                let iy = reflect1($yc as i64 + ky as i64 - cy as i64, $h as i64);
                let row = &$img[iy * $w..(iy + 1) * $w];
                for kx in 0..$fw {
                    acc +=
                        $filt[ky * $fw + kx] * row[reflect1(xc + kx as i64 - cx as i64, $w as i64)];
                }
            }
            acc
        };
        let mut ox = 0;
        while ox < $out.len() && $start + ox * $step < cx {
            $out[ox] = scalar(ox);
            ox += 1;
        }
        // Broadcast the (loop-invariant) filter taps once — same
        // values, same multiply-add order per output. Fixed 9×9
        // capacity: `corr_dn` filters never exceed it, and the `big`
        // fallback keeps an unexpected larger kernel correct.
        let big = $fh * $fw > 81;
        let ftaps: [$V; 81] = core::array::from_fn(|t| {
            <$V>::splat_t($token, if t < $fh * $fw && !big { $filt[t] } else { 0.0 })
        });
        // Four output blocks at once: each `acc` keeps its own serial
        // tap order (bit-identical); the four independent chains hide
        // the ~4-cycle add latency.
        while ox + 4 * $lanes <= $out.len()
            && $start + (ox + 4 * $lanes - 1) * $step + ($fw - cx) <= $w
        {
            let zero = <$V>::splat_t($token, 0.0);
            let (mut a0, mut a1, mut a2, mut a3) = (zero, zero, zero, zero);
            let x = $start + ox * $step - cx;
            let dx = $lanes * $step;
            for ky in 0..$fh {
                let iy = reflect1($yc as i64 + ky as i64 - cy as i64, $h as i64);
                let row = &$img[iy * $w..(iy + 1) * $w];
                for kx in 0..$fw {
                    let (v0, v1, v2, v3) = if $step == 1 {
                        // Contiguous lanes — packed loads. In bounds: the
                        // guard keeps `x + 3·dx + fw - 1 + lanes - 1 < w`.
                        (
                            <$V>::load_t($token, row[x + kx..x + kx + $lanes].try_into().unwrap()),
                            <$V>::load_t(
                                $token,
                                row[x + kx + dx..x + kx + dx + $lanes].try_into().unwrap(),
                            ),
                            <$V>::load_t(
                                $token,
                                row[x + kx + 2 * dx..x + kx + 2 * dx + $lanes]
                                    .try_into()
                                    .unwrap(),
                            ),
                            <$V>::load_t(
                                $token,
                                row[x + kx + 3 * dx..x + kx + 3 * dx + $lanes]
                                    .try_into()
                                    .unwrap(),
                            ),
                        )
                    } else {
                        (
                            <$V>::from_array_t(
                                $token,
                                core::array::from_fn(|lane| row[x + kx + lane * $step]),
                            ),
                            <$V>::from_array_t(
                                $token,
                                core::array::from_fn(|lane| row[x + kx + dx + lane * $step]),
                            ),
                            <$V>::from_array_t(
                                $token,
                                core::array::from_fn(|lane| row[x + kx + 2 * dx + lane * $step]),
                            ),
                            <$V>::from_array_t(
                                $token,
                                core::array::from_fn(|lane| row[x + kx + 3 * dx + lane * $step]),
                            ),
                        )
                    };
                    let t = ky * $fw + kx;
                    let f = if big {
                        <$V>::splat_t($token, $filt[t])
                    } else {
                        ftaps[t]
                    };
                    a0 = a0 + f * v0;
                    a1 = a1 + f * v1;
                    a2 = a2 + f * v2;
                    a3 = a3 + f * v3;
                }
            }
            $out[ox..ox + $lanes].copy_from_slice(&a0.to_array());
            $out[ox + $lanes..ox + 2 * $lanes].copy_from_slice(&a1.to_array());
            $out[ox + 2 * $lanes..ox + 3 * $lanes].copy_from_slice(&a2.to_array());
            $out[ox + 3 * $lanes..ox + 4 * $lanes].copy_from_slice(&a3.to_array());
            ox += 4 * $lanes;
        }
        while ox + $lanes <= $out.len() && $start + (ox + $lanes - 1) * $step + ($fw - cx) <= $w {
            let mut acc = <$V>::splat_t($token, 0.0);
            let x = $start + ox * $step - cx;
            for ky in 0..$fh {
                let iy = reflect1($yc as i64 + ky as i64 - cy as i64, $h as i64);
                let row = &$img[iy * $w..(iy + 1) * $w];
                for kx in 0..$fw {
                    let v = if $step == 1 {
                        // Contiguous lanes — one packed load instead of
                        // per-lane scalar reads. In bounds: the guard
                        // above keeps `x + fw - 1 + lanes - 1 < w`.
                        <$V>::load_t($token, row[x + kx..x + kx + $lanes].try_into().unwrap())
                    } else {
                        <$V>::from_array_t(
                            $token,
                            core::array::from_fn(|lane| row[x + kx + lane * $step]),
                        )
                    };
                    let t = ky * $fw + kx;
                    let f = if big {
                        <$V>::splat_t($token, $filt[t])
                    } else {
                        ftaps[t]
                    };
                    acc = acc + f * v;
                }
            }
            $out[ox..ox + $lanes].copy_from_slice(&acc.to_array());
            ox += $lanes;
        }
        while ox < $out.len() {
            $out[ox] = scalar(ox);
            ox += 1;
        }
    }};
}

// The pyramid uses only three kernel shapes. Specialize these at the
// tier boundary so the compiler sees constant tap counts and lane stride.
macro_rules! corr_shapes {
    ($token:ident, $V:ty, $lanes:literal, $img:ident, $w:ident, $h:ident,
     $filt:ident, $fw:ident, $fh:ident, $step:ident, $start:ident, $yc:ident, $out:ident) => {{
        match ($fw, $fh, $step) {
            (5, 5, 1) => {
                let (fw, fh, step) = (5, 5, 1);
                corr_row_body!(
                    $token, $V, $lanes, $img, $w, $h, $filt, fw, fh, step, $start, $yc, $out
                );
            }
            (7, 7, 1) => {
                let (fw, fh, step) = (7, 7, 1);
                corr_row_body!(
                    $token, $V, $lanes, $img, $w, $h, $filt, fw, fh, step, $start, $yc, $out
                );
            }
            (9, 9, 2) => {
                let (fw, fh, step) = (9, 9, 2);
                corr_row_body!(
                    $token, $V, $lanes, $img, $w, $h, $filt, fw, fh, step, $start, $yc, $out
                );
            }
            _ => corr_row_body!(
                $token, $V, $lanes, $img, $w, $h, $filt, $fw, $fh, $step, $start, $yc, $out
            ),
        }
    }};
}

#[cfg(target_arch = "x86_64")]
#[archmage::attune]
#[allow(clippy::too_many_arguments)]
fn corr_row_tier_v3_t(
    token: archmage::X64V3Token,
    img: &[f64],
    w: usize,
    h: usize,
    filt: &[f64],
    fw: usize,
    fh: usize,
    step: usize,
    start: usize,
    yc: usize,
    out: &mut [f64],
) {
    corr_shapes!(
        token,
        magetypes::simd::generic::f64x4<archmage::X64V3Token>,
        4,
        img,
        w,
        h,
        filt,
        fw,
        fh,
        step,
        start,
        yc,
        out
    );
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::attune]
#[allow(clippy::too_many_arguments)]
fn corr_row_tier_v4x_t(
    token: archmage::X64V4xToken,
    img: &[f64],
    w: usize,
    h: usize,
    filt: &[f64],
    fw: usize,
    fh: usize,
    step: usize,
    start: usize,
    yc: usize,
    out: &mut [f64],
) {
    corr_shapes!(
        token,
        magetypes::simd::generic::f64x8<archmage::X64V4xToken>,
        8,
        img,
        w,
        h,
        filt,
        fw,
        fh,
        step,
        start,
        yc,
        out
    );
}

#[cfg(target_arch = "aarch64")]
#[archmage::attune]
#[allow(clippy::too_many_arguments)]
fn corr_row_tier_neon_t(
    token: archmage::NeonToken,
    img: &[f64],
    w: usize,
    h: usize,
    filt: &[f64],
    fw: usize,
    fh: usize,
    step: usize,
    start: usize,
    yc: usize,
    out: &mut [f64],
) {
    corr_shapes!(
        token,
        magetypes::simd::generic::f64x4<archmage::NeonToken>,
        4,
        img,
        w,
        h,
        filt,
        fw,
        fh,
        step,
        start,
        yc,
        out
    );
}

#[cfg(target_arch = "wasm32")]
#[archmage::attune]
#[allow(clippy::too_many_arguments)]
fn corr_row_tier_wasm128_t(
    token: archmage::Wasm128Token,
    img: &[f64],
    w: usize,
    h: usize,
    filt: &[f64],
    fw: usize,
    fh: usize,
    step: usize,
    start: usize,
    yc: usize,
    out: &mut [f64],
) {
    corr_shapes!(
        token,
        magetypes::simd::generic::f64x4<archmage::Wasm128Token>,
        4,
        img,
        w,
        h,
        filt,
        fw,
        fh,
        step,
        start,
        yc,
        out
    );
}

#[allow(clippy::too_many_arguments)]
fn corr_row_tier_scalar_t(
    token: archmage::ScalarToken,
    img: &[f64],
    w: usize,
    h: usize,
    filt: &[f64],
    fw: usize,
    fh: usize,
    step: usize,
    start: usize,
    yc: usize,
    out: &mut [f64],
) {
    corr_shapes!(
        token,
        magetypes::simd::generic::f64x4<archmage::ScalarToken>,
        4,
        img,
        w,
        h,
        filt,
        fw,
        fh,
        step,
        start,
        yc,
        out
    );
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
        let (b1, b4) = crate::kernel::maybe_join(
            || corr_dn(&lo, lw, lh, &k1, 7, 7, 1, 0),
            || corr_dn(&lo, lw, lh, &k4, 7, 7, 1, 0),
        );
        let (b1, _, _) = b1;
        let (b4, _, _) = b4;
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
fn lu_factor(a: &mut [f64], pivots: &mut [usize; 9], n: usize) {
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
        }
        pivots[k] = piv;
        let akk = a[k * n + k];
        for i in (k + 1)..n {
            let f = a[i * n + k] / akk;
            a[i * n + k] = f;
            for j in (k + 1)..n {
                a[i * n + j] -= f * a[k * n + j];
            }
        }
    }
}

fn lu_apply(a: &[f64], pivots: &[usize; 9], b: &mut [f64; 9]) {
    let n = 9;
    for (k, &piv) in pivots.iter().enumerate() {
        b.swap(k, piv);
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
    // Each of the 81 accumulators sums its own lane in the same (ty,tx)
    // order — parallel over elements is bit-identical.
    #[cfg(feature = "parallel")]
    let mut ran_par = false;
    #[cfg(not(feature = "parallel"))]
    let ran_par = false;
    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 {
        use rayon::prelude::*;
        mean.par_iter_mut().enumerate().for_each(|(k, m)| {
            let (c, r) = (k / M, k % M);
            for ty in 0..(ch - (M - 1)) {
                for tx in 0..(cw - (M - 1)) {
                    *m += y[(ty + r) * w + tx + c];
                }
            }
        });
        ran_par = true;
    }
    let _ = ran_par;
    if !ran_par {
        for ty in 0..(ch - (M - 1)) {
            for tx in 0..(cw - (M - 1)) {
                for c in 0..M {
                    for r in 0..M {
                        mean[c * M + r] += y[(ty + r) * w + tx + c];
                    }
                }
            }
        }
    }
    for m in &mut mean {
        *m /= np as f64;
    }
    let mut cu = alloc::vec![0.0f64; n2 * n2];
    // Striped partial covariance: each stripe owns a ty range and a
    // private 81×81 buffer; the ordered merge keeps the sum order fixed
    // for any thread count (within ~1e-14 of serial — tolerance-passed).
    #[cfg(feature = "parallel")]
    let mut ran_par = false;
    #[cfg(not(feature = "parallel"))]
    let ran_par = false;
    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 {
        use rayon::prelude::*;
        let rows = ch - (M - 1);
        let stripes = (rows / 16).clamp(1, 64);
        let bounds: Vec<usize> = (0..=stripes).map(|s| s * rows / stripes).collect();
        let mut partials: Vec<Vec<f64>> = (0..stripes)
            .into_par_iter()
            .map(|s| {
                let mut pc = alloc::vec![0.0f64; n2 * n2];
                for ty in bounds[s]..bounds[s + 1] {
                    for tx in 0..(cw - (M - 1)) {
                        let mut v = [0.0f64; 9];
                        for c in 0..M {
                            for r in 0..M {
                                v[c * M + r] = y[(ty + r) * w + tx + c] - mean[c * M + r];
                            }
                        }
                        for i in 0..n2 {
                            for j in i..n2 {
                                pc[i * n2 + j] += v[i] * v[j];
                            }
                        }
                    }
                }
                pc
            })
            .collect();
        for mut pc in partials.drain(..) {
            for (e, p) in cu.iter_mut().zip(pc.drain(..)) {
                *e += p;
            }
        }
        ran_par = true;
    }
    let _ = ran_par;
    if !ran_par {
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
    // Non-overlapping blocks are independent — disjoint ss writes with a
    // per-task LU scratch (identical per-block math either way).
    // The covariance is shared by every block: factor it once. Applying
    // the recorded pivots and triangular solves retains per-block rounding.
    let mut lu = cu;
    let mut pivots = [0; 9];
    lu_factor(&mut lu, &mut pivots, n2);
    let ss_block = |by: usize, bx: usize| -> f64 {
        let mut v = [0.0f64; 9];
        for c in 0..M {
            for r in 0..M {
                v[c * M + r] = y[(by * M + r) * w + bx * M + c];
            }
        }
        let mut x = v;
        lu_apply(&lu, &pivots, &mut x);
        let mut s = 0.0;
        for i in 0..n2 {
            s += v[i] * x[i];
        }
        s / (n2 as f64)
    };
    #[cfg(feature = "parallel")]
    let mut ran_par = false;
    #[cfg(not(feature = "parallel"))]
    let ran_par = false;
    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 {
        use rayon::prelude::*;
        ss.par_chunks_mut(nbw).enumerate().for_each(|(by, row)| {
            let mut bx = 0;
            while bx + 8 <= nbw {
                bx += ss_lu_batch(&lu, &pivots, y, w, by, bx, row);
            }
            while bx < nbw {
                row[bx] = ss_block(by, bx);
                bx += 1;
            }
        });
        ran_par = true;
    }
    let _ = ran_par;
    if !ran_par {
        for by in 0..nbh {
            let mut bx = 0;
            while bx + 8 <= nbw {
                bx += ss_lu_batch(
                    &lu,
                    &pivots,
                    y,
                    w,
                    by,
                    bx,
                    &mut ss[by * nbw..(by + 1) * nbw],
                );
            }
            while bx < nbw {
                ss[by * nbw + bx] = ss_block(by, bx);
                bx += 1;
            }
        }
    }
    (ss, lambda, nbw, nbh)
}

struct EstRow<'a> {
    y: &'a [f64],
    yn: &'a [f64],
    w: usize,
    cw: usize,
    ch: usize,
    winsize: usize,
    oy: usize,
}

fn channel_gain(sums: [f64; 5], n: f64) -> (f64, f64) {
    let [sx, sy, sxy, sx2, sy2] = sums;
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
    (gi, vvi)
}

fn est_scalar(a: &EstRow<'_>, ox: usize) -> (f64, f64) {
    let cy = (1 + a.oy * M) as i64; // 0-based centre (1-based 2,5,8,…)
    let cx = (1 + ox * M) as i64;
    let (mut sx, mut sy, mut sxy, mut sx2, mut sy2) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for ky in 0..a.winsize {
        let iy = reflect1(cy + ky as i64 - (a.winsize / 2) as i64, a.ch as i64);
        let rowr = &a.y[iy * a.w..];
        let rowd = &a.yn[iy * a.w..];
        for kx in 0..a.winsize {
            let ix = reflect1(cx + kx as i64 - (a.winsize / 2) as i64, a.cw as i64);
            let (a, b) = (rowr[ix], rowd[ix]);
            sx += a;
            sy += b;
            sxy += a * b;
            sx2 += a * a;
            sy2 += b * b;
        }
    }
    channel_gain([sx, sy, sxy, sx2, sy2], (a.winsize * a.winsize) as f64)
}

macro_rules! est_row_body {
    ($token:ident, $V:ty, $lanes:literal, $a:ident, $g:ident, $vv:ident) => {{
        let half = $a.winsize / 2;
        let n = ($a.winsize * $a.winsize) as f64;
        let cy = 1 + $a.oy * M;
        let mut ox = 0;
        while ox < $g.len() && 1 + ox * M < half {
            ($g[ox], $vv[ox]) = est_scalar(&$a, ox);
            ox += 1;
        }
        while ox + $lanes <= $g.len() && 1 + (ox + $lanes - 1) * M + ($a.winsize - half) <= $a.cw {
            let mut sums = [<$V>::splat_t($token, 0.0); 5];
            let x = 1 + ox * M - half;
            for ky in 0..$a.winsize {
                let iy = reflect1(cy as i64 + ky as i64 - half as i64, $a.ch as i64);
                let r = &$a.y[iy * $a.w..(iy + 1) * $a.w];
                let d = &$a.yn[iy * $a.w..(iy + 1) * $a.w];
                for kx in 0..$a.winsize {
                    let rv = <$V>::from_array_t(
                        $token,
                        core::array::from_fn(|lane| r[x + kx + lane * M]),
                    );
                    let dv = <$V>::from_array_t(
                        $token,
                        core::array::from_fn(|lane| d[x + kx + lane * M]),
                    );
                    sums[0] = sums[0] + rv;
                    sums[1] = sums[1] + dv;
                    sums[2] = sums[2] + rv * dv;
                    sums[3] = sums[3] + rv * rv;
                    sums[4] = sums[4] + dv * dv;
                }
            }
            let arrays = sums.map(|v| v.to_array());
            for lane in 0..$lanes {
                ($g[ox + lane], $vv[ox + lane]) =
                    channel_gain(core::array::from_fn(|k| arrays[k][lane]), n);
            }
            ox += $lanes;
        }
        while ox < $g.len() {
            ($g[ox], $vv[ox]) = est_scalar(&$a, ox);
            ox += 1;
        }
    }};
}
#[cfg(target_arch = "x86_64")]
#[archmage::attune]
fn est_row_tier_v3_t(token: archmage::X64V3Token, a: EstRow<'_>, g: &mut [f64], vv: &mut [f64]) {
    est_row_body!(
        token,
        magetypes::simd::generic::f64x4<archmage::X64V3Token>,
        4,
        a,
        g,
        vv
    );
}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::attune]
fn est_row_tier_v4x_t(token: archmage::X64V4xToken, a: EstRow<'_>, g: &mut [f64], vv: &mut [f64]) {
    est_row_body!(
        token,
        magetypes::simd::generic::f64x8<archmage::X64V4xToken>,
        8,
        a,
        g,
        vv
    );
}
#[cfg(target_arch = "aarch64")]
#[archmage::attune]
fn est_row_tier_neon_t(token: archmage::NeonToken, a: EstRow<'_>, g: &mut [f64], vv: &mut [f64]) {
    est_row_body!(
        token,
        magetypes::simd::generic::f64x4<archmage::NeonToken>,
        4,
        a,
        g,
        vv
    );
}
#[cfg(target_arch = "wasm32")]
#[archmage::attune]
fn est_row_tier_wasm128_t(
    token: archmage::Wasm128Token,
    a: EstRow<'_>,
    g: &mut [f64],
    vv: &mut [f64],
) {
    est_row_body!(
        token,
        magetypes::simd::generic::f64x4<archmage::Wasm128Token>,
        4,
        a,
        g,
        vv
    );
}
fn est_row_tier_scalar_t(
    token: archmage::ScalarToken,
    a: EstRow<'_>,
    g: &mut [f64],
    vv: &mut [f64],
) {
    est_row_body!(
        token,
        magetypes::simd::generic::f64x4<archmage::ScalarToken>,
        4,
        a,
        g,
        vv
    );
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

    let nbw = cw / M;
    let nbh = ch / M;
    let mut g = alloc::vec![0.0f64; nbw * nbh];
    let mut vv = alloc::vec![0.0f64; nbw * nbh];
    let est_row = |oy: usize, gr: &mut [f64], vr: &mut [f64]| {
        let a = EstRow {
            y,
            yn,
            w,
            cw,
            ch,
            winsize,
            oy,
        };
        archmage::attuned!(
            est_row_tier(a, gr, vr),
            [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
        );
    };
    #[cfg(feature = "parallel")]
    let mut ran_par = false;
    #[cfg(not(feature = "parallel"))]
    let ran_par = false;
    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 {
        use rayon::prelude::*;
        g.par_chunks_mut(nbw)
            .zip(vv.par_chunks_mut(nbw))
            .enumerate()
            .for_each(|(oy, (gr, vr))| {
                est_row(oy, gr, vr);
            });
        ran_par = true;
    }
    let _ = ran_par;
    if !ran_par {
        for oy in 0..nbh {
            est_row(
                oy,
                &mut g[oy * nbw..(oy + 1) * nbw],
                &mut vv[oy * nbw..(oy + 1) * nbw],
            );
        }
    }
    (g, vv, nbw, nbh)
}

// ------------------------------------------------------------------
// Pooled log2 accumulation (vector-GSM pooling).
// ------------------------------------------------------------------

/// Lane-vectorised `libm::log2` — delegates to zenmetrics-math's
/// lane-exact fdlibm `e_log2` port (`log2_libm_4x`), bit-identical to
/// `libm::log2` on every input (this call site only feeds `x >= 1`,
/// the `1 + nonneg` pooling domain). 8-lane tiers run two 4-lane
/// calls — the port is `f64x4`-wide on every tier.
macro_rules! log2_lanes {
    ($token:ident, $V:ty, $lanes:literal, $x:expr) => {{
        let xa: [f64; $lanes] = $x;
        let mut ra = [0.0f64; $lanes];
        let mut k = 0;
        while k < $lanes {
            let r = zenmetrics_math::fdlibm64::log2_libm_4x(
                $token,
                magetypes::simd::generic::f64x4::<_>::from_array_t(
                    $token,
                    core::array::from_fn(|j| xa[k + j]),
                ),
            );
            ra[k..k + 4].copy_from_slice(&r.to_array());
            k += 4;
        }
        ra
    }};
}

/// Accumulate `t1`/`t2` for one eigenvalue `lam` over the cropped
/// `(iw, ih)` block map — same sequential `+=` order per accumulator
/// as the scalar loop, so results are bit-identical.
macro_rules! pool_log2_body {
    ($token:ident, $V:ty, $lanes:literal, $g:ident, $vv:ident, $ss:ident,
     $nbw:ident, $iw:ident, $ih:ident, $offset:ident, $lam:ident) => {{
        let mut t1 = 0.0f64;
        let mut t2 = 0.0f64;
        for by in 0..$ih {
            let row = (by + $offset) * $nbw + $offset;
            let mut bx = 0;
            while bx + $lanes <= $iw {
                let mut a1 = [0.0f64; $lanes];
                let mut a2 = [0.0f64; $lanes];
                for j in 0..$lanes {
                    let i = row + bx + j;
                    let (gi, vi, si) = ($g[i], $vv[i], $ss[i]);
                    a1[j] = 1.0 + gi * gi * si * $lam / (vi + SIGMA_NSQ);
                    a2[j] = 1.0 + si * $lam / SIGMA_NSQ;
                }
                let l1 = log2_lanes!($token, $V, $lanes, a1);
                let l2 = log2_lanes!($token, $V, $lanes, a2);
                for j in 0..$lanes {
                    t1 += l1[j];
                    t2 += l2[j];
                }
                bx += $lanes;
            }
            while bx < $iw {
                let i = row + bx;
                let (gi, vi, si) = ($g[i], $vv[i], $ss[i]);
                t1 += libm::log2(1.0 + gi * gi * si * $lam / (vi + SIGMA_NSQ));
                t2 += libm::log2(1.0 + si * $lam / SIGMA_NSQ);
                bx += 1;
            }
        }
        (t1, t2)
    }};
}

#[cfg(target_arch = "x86_64")]
#[archmage::attune]
#[allow(clippy::too_many_arguments)]
fn pool_log2_tier_v3_t(
    token: archmage::X64V3Token,
    g: &[f64],
    vv: &[f64],
    ss: &[f64],
    nbw: usize,
    iw: usize,
    ih: usize,
    offset: usize,
    lam: f64,
) -> (f64, f64) {
    pool_log2_body!(
        token,
        magetypes::simd::generic::f64x4<archmage::X64V3Token>,
        4,
        g,
        vv,
        ss,
        nbw,
        iw,
        ih,
        offset,
        lam
    )
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::attune]
#[allow(clippy::too_many_arguments)]
fn pool_log2_tier_v4x_t(
    token: archmage::X64V4xToken,
    g: &[f64],
    vv: &[f64],
    ss: &[f64],
    nbw: usize,
    iw: usize,
    ih: usize,
    offset: usize,
    lam: f64,
) -> (f64, f64) {
    pool_log2_body!(
        token,
        magetypes::simd::generic::f64x8<archmage::X64V4xToken>,
        8,
        g,
        vv,
        ss,
        nbw,
        iw,
        ih,
        offset,
        lam
    )
}

#[cfg(target_arch = "aarch64")]
#[archmage::attune]
#[allow(clippy::too_many_arguments)]
fn pool_log2_tier_neon_t(
    token: archmage::NeonToken,
    g: &[f64],
    vv: &[f64],
    ss: &[f64],
    nbw: usize,
    iw: usize,
    ih: usize,
    offset: usize,
    lam: f64,
) -> (f64, f64) {
    pool_log2_body!(
        token,
        magetypes::simd::generic::f64x4<archmage::NeonToken>,
        4,
        g,
        vv,
        ss,
        nbw,
        iw,
        ih,
        offset,
        lam
    )
}

#[cfg(target_arch = "wasm32")]
#[archmage::attune]
#[allow(clippy::too_many_arguments)]
fn pool_log2_tier_wasm128_t(
    token: archmage::Wasm128Token,
    g: &[f64],
    vv: &[f64],
    ss: &[f64],
    nbw: usize,
    iw: usize,
    ih: usize,
    offset: usize,
    lam: f64,
) -> (f64, f64) {
    pool_log2_body!(
        token,
        magetypes::simd::generic::f64x4<archmage::Wasm128Token>,
        4,
        g,
        vv,
        ss,
        nbw,
        iw,
        ih,
        offset,
        lam
    )
}

#[allow(clippy::too_many_arguments)]
fn pool_log2_tier_scalar_t(
    token: archmage::ScalarToken,
    g: &[f64],
    vv: &[f64],
    ss: &[f64],
    nbw: usize,
    iw: usize,
    ih: usize,
    offset: usize,
    lam: f64,
) -> (f64, f64) {
    pool_log2_body!(
        token,
        magetypes::simd::generic::f64x4<archmage::ScalarToken>,
        4,
        g,
        vv,
        ss,
        nbw,
        iw,
        ih,
        offset,
        lam
    )
}

/// `ss` for `$lanes` consecutive blocks at row `by`: a lane-parallel
/// LU solve — the shared pivot record permutes whole SoA vectors, and
/// every substitution/dot op is elementwise per lane, so each block
/// sees bit-identical math to the scalar `ss_block`.
macro_rules! ss_lu_body {
    ($token:ident, $V:ty, $lanes:literal, $lu:ident, $pivots:ident, $y:ident, $w:ident,
     $by:ident, $bx:ident, $out:ident) => {{
        let mut x: [$V; 9] = core::array::from_fn(|k| {
            let (c, r) = (k / M, k % M);
            <$V>::from_array_t(
                $token,
                core::array::from_fn(|l| $y[($by * M + r) * $w + ($bx + l) * M + c]),
            )
        });
        // Original block vectors for the quadratic form `v·x`.
        let v = x;
        for (k, &p) in $pivots.iter().enumerate() {
            x.swap(k, p);
        }
        for i in 1..9 {
            for j in 0..i {
                x[i] = x[i] - <$V>::splat_t($token, $lu[i * 9 + j]) * x[j];
            }
        }
        for i in (0..9).rev() {
            let mut s = x[i];
            for j in (i + 1)..9 {
                s = s - <$V>::splat_t($token, $lu[i * 9 + j]) * x[j];
            }
            x[i] = s / <$V>::splat_t($token, $lu[i * 9 + i]);
        }
        let mut s = <$V>::splat_t($token, 0.0);
        for i in 0..9 {
            s = s + v[i] * x[i];
        }
        let s = s / <$V>::splat_t($token, 9.0);
        $out[$bx..$bx + $lanes].copy_from_slice(&s.to_array());
        $lanes
    }};
}

#[cfg(target_arch = "x86_64")]
#[archmage::attune]
#[allow(clippy::too_many_arguments)]
fn ss_lu_tier_v3_t(
    token: archmage::X64V3Token,
    lu: &[f64],
    pivots: &[usize; 9],
    y: &[f64],
    w: usize,
    by: usize,
    bx: usize,
    out: &mut [f64],
) -> usize {
    ss_lu_body!(
        token,
        magetypes::simd::generic::f64x4<archmage::X64V3Token>,
        4,
        lu,
        pivots,
        y,
        w,
        by,
        bx,
        out
    )
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::attune]
#[allow(clippy::too_many_arguments)]
fn ss_lu_tier_v4x_t(
    token: archmage::X64V4xToken,
    lu: &[f64],
    pivots: &[usize; 9],
    y: &[f64],
    w: usize,
    by: usize,
    bx: usize,
    out: &mut [f64],
) -> usize {
    ss_lu_body!(
        token,
        magetypes::simd::generic::f64x8<archmage::X64V4xToken>,
        8,
        lu,
        pivots,
        y,
        w,
        by,
        bx,
        out
    )
}

#[cfg(target_arch = "aarch64")]
#[archmage::attune]
#[allow(clippy::too_many_arguments)]
fn ss_lu_tier_neon_t(
    token: archmage::NeonToken,
    lu: &[f64],
    pivots: &[usize; 9],
    y: &[f64],
    w: usize,
    by: usize,
    bx: usize,
    out: &mut [f64],
) -> usize {
    ss_lu_body!(
        token,
        magetypes::simd::generic::f64x4<archmage::NeonToken>,
        4,
        lu,
        pivots,
        y,
        w,
        by,
        bx,
        out
    )
}

#[cfg(target_arch = "wasm32")]
#[archmage::attune]
#[allow(clippy::too_many_arguments)]
fn ss_lu_tier_wasm128_t(
    token: archmage::Wasm128Token,
    lu: &[f64],
    pivots: &[usize; 9],
    y: &[f64],
    w: usize,
    by: usize,
    bx: usize,
    out: &mut [f64],
) -> usize {
    ss_lu_body!(
        token,
        magetypes::simd::generic::f64x4<archmage::Wasm128Token>,
        4,
        lu,
        pivots,
        y,
        w,
        by,
        bx,
        out
    )
}

#[allow(clippy::too_many_arguments)]
fn ss_lu_tier_scalar_t(
    token: archmage::ScalarToken,
    lu: &[f64],
    pivots: &[usize; 9],
    y: &[f64],
    w: usize,
    by: usize,
    bx: usize,
    out: &mut [f64],
) -> usize {
    ss_lu_body!(
        token,
        magetypes::simd::generic::f64x4<archmage::ScalarToken>,
        4,
        lu,
        pivots,
        y,
        w,
        by,
        bx,
        out
    )
}

/// `ss` outputs starting at `(by, bx)`; returns the dispatch
/// width written (4 or 8).
#[allow(clippy::too_many_arguments)]
fn ss_lu_batch(
    lu: &[f64],
    pivots: &[usize; 9],
    y: &[f64],
    w: usize,
    by: usize,
    bx: usize,
    out: &mut [f64],
) -> usize {
    archmage::attuned!(
        ss_lu_tier(lu, pivots, y, w, by, bx, out),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    )
}

/// `(t1, t2)` pooling sums for one eigenvalue, bit-identical to the
/// original scalar `libm::log2` loop on every dispatch tier.
#[allow(clippy::too_many_arguments)]
fn pool_log2(
    g: &[f64],
    vv: &[f64],
    ss: &[f64],
    nbw: usize,
    iw: usize,
    ih: usize,
    offset: usize,
    lam: f64,
) -> (f64, f64) {
    archmage::attuned!(
        pool_log2_tier(g, vv, ss, nbw, iw, ih, offset, lam),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    )
}

// ------------------------------------------------------------------
// Driver.
// ------------------------------------------------------------------

/// One subband's per-block information terms over the cropped block grid
/// the score pools: `num[i] = Σ_λ log2(1 + g²·s·λ/(σᵥ² + σₙ²))`,
/// `den[i] = Σ_λ log2(1 + s·λ/σₙ²)` (λ in eigenvalue order), each block
/// summed on its own; the score sums the same per-element `log2` terms in
/// λ-major order.
#[derive(Debug)]
pub(crate) struct SubTerms {
    /// Pyramid level, 1-based (1 = full resolution).
    pub(crate) level: usize,
    /// MATLAB band index, 1 or 4.
    pub(crate) band: usize,
    /// Blocks cropped from each side of the subband's block grid.
    pub(crate) offset: usize,
    /// Cropped grid width in 3×3 blocks.
    pub(crate) w: usize,
    /// Cropped grid height in blocks.
    pub(crate) h: usize,
    pub(crate) num: Vec<f64>,
    pub(crate) den: Vec<f64>,
}

/// Vector-GSM VIF of two `w×h` f64 planes (0–255 signal scale).
/// Returns `NaN` when the reference would (empty/degenerate fields).
pub(crate) fn vifvec_core(r: &[f64], d: &[f64], w: usize, h: usize) -> f64 {
    vifvec_core_impl(r, d, w, h, None)
}

/// [`vifvec_core`]; with `maps`, also collects every selected subband's
/// [`SubTerms`] in `SUBS` order after its pooling terms are computed.
pub(crate) fn vifvec_core_impl(
    r: &[f64],
    d: &[f64],
    w: usize,
    h: usize,
    maps: Option<&mut Vec<SubTerms>>,
) -> f64 {
    let with_maps = maps.is_some();
    // The two pyramid decompositions are independent.
    let (rb, db) = crate::kernel::maybe_join(|| build_bands(r, w, h), || build_bands(d, w, h));

    // Per-subband work is independent; collect each band's 9 (t1,t2)
    // pairs in band order and fold serially — the fold visits terms in
    // the original (sub, lam) order, so the sum is bit-identical.
    let sub_terms = |si: usize| -> (Vec<(f64, f64)>, Option<SubTerms>) {
        let (level, band, winsize, offset) = SUBS[si];
        let y = &rb[level - 1][usize::from(band == 4)];
        let yn = &db[level - 1][usize::from(band == 4)];
        let (ss, lambda, nbw, nbh) = refparams_vecgsm(&y.data, y.w, y.h);
        let (g, vv, gw, gh) = vifsub_est(&y.data, &yn.data, y.w, y.h, winsize);
        debug_assert_eq!((gw, gh), (nbw, nbh));

        // Crop `offset` entries on every side of the block maps
        // (reference: `g(offset+1:end-offset, …)`).
        let (iw, ih) = (nbw - 2 * offset, nbh - 2 * offset);
        let terms: Vec<(f64, f64)> = lambda
            .iter()
            .map(|&lam| pool_log2(&g, &vv, &ss, nbw, iw, ih, offset, lam))
            .collect();
        let blocks = with_maps.then(|| {
            // The same per-element `libm::log2` terms `pool_log2` adds
            // (its lanes are bit-identical to it), summed per block.
            let (mut num, mut den) = (alloc::vec![0.0f64; iw * ih], alloc::vec![0.0f64; iw * ih]);
            for &lam in &lambda {
                for by in 0..ih {
                    let row = (by + offset) * nbw + offset;
                    for bx in 0..iw {
                        let i = row + bx;
                        let (gi, vi, si) = (g[i], vv[i], ss[i]);
                        num[by * iw + bx] +=
                            libm::log2(1.0 + gi * gi * si * lam / (vi + SIGMA_NSQ));
                        den[by * iw + bx] += libm::log2(1.0 + si * lam / SIGMA_NSQ);
                    }
                }
            }
            SubTerms {
                level,
                band,
                offset,
                w: iw,
                h: ih,
                num,
                den,
            }
        });
        (terms, blocks)
    };
    let subs_terms: Vec<(Vec<(f64, f64)>, Option<SubTerms>)> = {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            if rayon::current_num_threads() > 1 {
                (0..SUBS.len()).into_par_iter().map(sub_terms).collect()
            } else {
                (0..SUBS.len()).map(sub_terms).collect()
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            (0..SUBS.len()).map(sub_terms).collect()
        }
    };
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (terms, _) in &subs_terms {
        for &(t1, t2) in terms {
            num += t1;
            den += t2;
        }
    }
    if let Some(sink) = maps {
        sink.extend(subs_terms.into_iter().filter_map(|(_, b)| b));
    }
    num / den
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_statistics_match_scalar_windows() {
        let (w, h) = (83, 77);
        let y: Vec<f64> = (0..w * h).map(|i| (i * 37 % 257) as f64 - 123.0).collect();
        let yn: Vec<f64> = y
            .iter()
            .enumerate()
            .map(|(i, &v)| v + (i % 7) as f64)
            .collect();
        for winsize in [3, 5, 9, 17] {
            let (g, vv, bw, bh) = vifsub_est(&y, &yn, w, h, winsize);
            for oy in 0..bh {
                let a = EstRow {
                    y: &y,
                    yn: &yn,
                    w,
                    cw: w / M * M,
                    ch: h / M * M,
                    winsize,
                    oy,
                };
                for ox in 0..bw {
                    let (eg, ev) = est_scalar(&a, ox);
                    assert_eq!(g[oy * bw + ox].to_bits(), eg.to_bits());
                    assert_eq!(vv[oy * bw + ox].to_bits(), ev.to_bits());
                }
            }
        }
    }

    #[test]
    fn correlation_matches_direct_reflected_taps() {
        // Odd widths exercise every vector tail; steps 1/2 cover bands
        // and lowpass decimation, with both reflected borders included.
        for (w, h) in [(9, 11), (18, 13), (71, 73)] {
            let img: Vec<f64> = (0..w * h)
                .map(|i| ((i * 37 % 257) as f64 - 123.0) / 7.0)
                .collect();
            for (filt, fw) in [(&LO0[..], 5), (&LOFILT[..], 9), (&BFILTS[0][..], 7)] {
                for step in [1, 2] {
                    for start in [0, 1] {
                        let (actual, ow, oh) = corr_dn(&img, w, h, filt, fw, fw, step, start);
                        for oy in 0..oh {
                            for ox in 0..ow {
                                let mut expected = 0.0;
                                for ky in 0..fw {
                                    for kx in 0..fw {
                                        let y = reflect1(
                                            (start + oy * step + ky) as i64 - (fw / 2) as i64,
                                            h as i64,
                                        );
                                        let x = reflect1(
                                            (start + ox * step + kx) as i64 - (fw / 2) as i64,
                                            w as i64,
                                        );
                                        expected += filt[ky * fw + kx] * img[y * w + x];
                                    }
                                }
                                assert_eq!(
                                    actual[oy * ow + ox].to_bits(),
                                    expected.to_bits(),
                                    "{w}x{h}, kernel {fw}, step {step}, start {start}, ({ox},{oy})"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn pool_log2_matches_scalar_libm_bits() {
        // Block maps with exact-1.0 args (ss = 0 → `1 + 0`), varied
        // magnitudes (arg up to ~1e6 → nonzero `k`), an `inf` arg and
        // a `NaN` arg — every early-out and reduce path in the kernel.
        let (nbw, nbh) = (37, 29);
        let g: Vec<f64> = (0..nbw * nbh)
            .map(|i| ((i * 37 % 251) as f64 - 100.0) * 0.031)
            .collect();
        let vv: Vec<f64> = (0..nbw * nbh)
            .map(|i| ((i * 53 % 191) as f64) * 0.005)
            .collect();
        let mut ss: Vec<f64> = (0..nbw * nbh)
            .map(|i| ((i * 29 % 313) as f64) * 0.07)
            .collect();
        ss[3] = 0.0;
        ss[40] = f64::INFINITY;
        ss[77] = f64::NAN;
        let (offset, iw, ih) = (3usize, nbw - 6, nbh - 6);
        for lam in [0.0, 1e-6, 0.5, 1.0, 37.3, 1e4] {
            let (t1, t2) = pool_log2(&g, &vv, &ss, nbw, iw, ih, offset, lam);
            let (mut r1, mut r2) = (0.0f64, 0.0f64);
            for by in 0..ih {
                for bx in 0..iw {
                    let i = (by + offset) * nbw + bx + offset;
                    let (gi, vi, si) = (g[i], vv[i], ss[i]);
                    r1 += libm::log2(1.0 + gi * gi * si * lam / (vi + SIGMA_NSQ));
                    r2 += libm::log2(1.0 + si * lam / SIGMA_NSQ);
                }
            }
            assert_eq!(t1.to_bits(), r1.to_bits(), "t1 lam={lam}");
            assert_eq!(t2.to_bits(), r2.to_bits(), "t2 lam={lam}");
        }
    }

    #[test]
    fn ss_lu_batch_matches_scalar_solve_bits() {
        let scalar_block = |lu: &[f64], pivots: &[usize; 9], y: &[f64], w, by, bx| {
            let mut v = [0.0f64; 9];
            for c in 0..M {
                for r in 0..M {
                    v[c * M + r] = y[(by * M + r) * w + bx * M + c];
                }
            }
            let mut x = v;
            lu_apply(lu, pivots, &mut x);
            let mut s = 0.0;
            for i in 0..9 {
                s += v[i] * x[i];
            }
            s / 9.0
        };
        let (nbw, nbh) = (11usize, 7usize);
        let (w, h) = (M * nbw, M * nbh);
        let y: Vec<f64> = (0..w * h)
            .map(|i| ((i * 31) % 197) as f64 * 0.5 - 40.0)
            .collect();
        // A well-conditioned system and a near-singular one (inf/NaN
        // lanes must propagate identically through the batch path).
        for near_singular in [false, true] {
            let mut a = [0.0f64; 81];
            for i in 0..9 {
                for j in 0..9 {
                    a[i * 9 + j] = if near_singular && i > 6 {
                        a[j]
                    } else {
                        ((i * 7 + j * 3) % 11) as f64 * 0.1
                            + if i == j {
                                4.0
                            } else {
                                ((i * 13 + j * 5) % 7) as f64 * 0.2 - 0.5
                            }
                    };
                }
            }
            let mut lu = a.to_vec();
            let mut pivots = [0usize; 9];
            lu_factor(&mut lu, &mut pivots, 9);
            let mut out = alloc::vec![0.0f64; nbw * nbh];
            for by in 0..nbh {
                let mut bx = 0;
                while bx + 8 <= nbw {
                    bx += ss_lu_batch(
                        &lu,
                        &pivots,
                        &y,
                        w,
                        by,
                        bx,
                        &mut out[by * nbw..(by + 1) * nbw],
                    );
                }
                while bx < nbw {
                    out[by * nbw + bx] = scalar_block(&lu, &pivots, &y, w, by, bx);
                    bx += 1;
                }
            }
            for by in 0..nbh {
                for bx in 0..nbw {
                    assert_eq!(
                        out[by * nbw + bx].to_bits(),
                        scalar_block(&lu, &pivots, &y, w, by, bx).to_bits(),
                        "({by},{bx}) near_singular={near_singular}"
                    );
                }
            }
        }
    }
}
