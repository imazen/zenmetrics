//! PSNR-HVS per-block kernels and per-tier band entry points.
//!
//! Implements the published algorithm of Egiazarian et al. (VPQM-06) and
//! Ponomarenko et al. (VPQM-07) — the same computation the reference
//! `psnrhvsm.m` performs. The CSF (`CSFCof`) and masking (`MaskCof`)
//! tables are the papers' published constants.
//!
//! # Layout and floating-point order
//!
//! Each batch gathers `LANES` blocks into coefficient-major vectors —
//! lane `b` of `X[c]` is coefficient `c` of block `b` — so the two DCT
//! passes and every similarity/masking term are lane-independent vector
//! ops. The per-block f32 sums accumulate in a fixed coefficient order
//! and the per-batch `(s1, s2)` totals widen to f64 and fold into the
//! global sums in raster order, so **every width is bit-identical**:
//! the only difference between tiers is how many lanes the vector ops
//! carry. Plain `*` and `+` are used throughout — never `mul_add`,
//! whose fusion differs across backends.

use alloc::vec::Vec;

/// CSF coefficient weights `CSFCof[k*8 + l]` — Egiazarian et al.
/// VPQM-06, table of DCT-basis contrast sensitivities on the 0..255
/// scale (the same constants the reference `psnrhvsm.m` carries).
pub(crate) const CSF: [f32; 64] = [
    1.608443, 2.339554, 2.573509, 1.608443, 1.072295, 0.643377, 0.504610, 0.421887, 2.144591,
    2.144591, 1.838221, 1.354478, 0.989811, 0.443708, 0.428918, 0.467911, 1.838221, 1.979622,
    1.608443, 1.072295, 0.643377, 0.451493, 0.372972, 0.459555, 1.838221, 1.513829, 1.169777,
    0.887417, 0.504610, 0.295806, 0.321689, 0.415082, 1.429727, 1.169777, 0.695543, 0.459555,
    0.378457, 0.236102, 0.249855, 0.334222, 1.072295, 0.735288, 0.467911, 0.402111, 0.317717,
    0.247453, 0.227744, 0.279729, 0.525206, 0.402111, 0.329937, 0.295806, 0.249855, 0.212687,
    0.214459, 0.254803, 0.357432, 0.279729, 0.270896, 0.262603, 0.229778, 0.257351, 0.249855,
    0.259950,
];

/// Between-coefficient masking table `MaskCof[k*8 + l]` — Ponomarenko
/// et al. VPQM-07, equation 3 thresholds.
pub(crate) const MASK_COF: [f32; 64] = [
    0.390625, 0.826446, 1.000000, 0.390625, 0.173611, 0.062500, 0.038447, 0.026874, 0.694444,
    0.694444, 0.510204, 0.277008, 0.147929, 0.029727, 0.027778, 0.033058, 0.510204, 0.591716,
    0.390625, 0.173611, 0.062500, 0.030779, 0.021004, 0.031888, 0.510204, 0.346021, 0.206612,
    0.118906, 0.038447, 0.013212, 0.015625, 0.026015, 0.308642, 0.206612, 0.073046, 0.031888,
    0.021626, 0.008417, 0.009426, 0.016866, 0.173611, 0.081633, 0.033058, 0.024414, 0.015242,
    0.009246, 0.007831, 0.011815, 0.041649, 0.024414, 0.016437, 0.013212, 0.009426, 0.006830,
    0.006944, 0.009803, 0.019290, 0.011815, 0.011080, 0.010412, 0.007972, 0.010000, 0.009426,
    0.010203,
];

/// Orthonormal 8×8 DCT-II matrix `T[k*8 + m]` (MATLAB `dct2`
/// convention): `T[k][m] = sqrt(2/8)·cos(π·(2m+1)·k/16)`, row 0 scaled by
/// `1/√2`. Computed here as f32 literals from the definition.
pub(crate) const DCT_T: [f32; 64] = [
    3.5355339059e-01,
    3.5355339059e-01,
    3.5355339059e-01,
    3.5355339059e-01,
    3.5355339059e-01,
    3.5355339059e-01,
    3.5355339059e-01,
    3.5355339059e-01,
    4.9039264020e-01,
    4.1573480615e-01,
    2.7778511651e-01,
    9.7545161008e-02,
    -9.7545161008e-02,
    -2.7778511651e-01,
    -4.1573480615e-01,
    -4.9039264020e-01,
    4.6193976626e-01,
    1.9134171618e-01,
    -1.9134171618e-01,
    -4.6193976626e-01,
    -4.6193976626e-01,
    -1.9134171618e-01,
    1.9134171618e-01,
    4.6193976626e-01,
    4.1573480615e-01,
    -9.7545161008e-02,
    -4.9039264020e-01,
    -2.7778511651e-01,
    2.7778511651e-01,
    4.9039264020e-01,
    9.7545161008e-02,
    -4.1573480615e-01,
    3.5355339059e-01,
    -3.5355339059e-01,
    -3.5355339059e-01,
    3.5355339059e-01,
    3.5355339059e-01,
    -3.5355339059e-01,
    -3.5355339059e-01,
    3.5355339059e-01,
    2.7778511651e-01,
    -4.9039264020e-01,
    9.7545161008e-02,
    4.1573480615e-01,
    -4.1573480615e-01,
    -9.7545161008e-02,
    4.9039264020e-01,
    -2.7778511651e-01,
    1.9134171618e-01,
    -4.6193976626e-01,
    4.6193976626e-01,
    -1.9134171618e-01,
    -1.9134171618e-01,
    4.6193976626e-01,
    -4.6193976626e-01,
    1.9134171618e-01,
    9.7545161008e-02,
    -2.7778511651e-01,
    4.1573480615e-01,
    -4.9039264020e-01,
    4.9039264020e-01,
    -4.1573480615e-01,
    2.7778511651e-01,
    -9.7545161008e-02,
];

// ---------------------------------------------------------------------
// Deterministic banding — same contract as `iwssim::par`: band count is
// a pure function of the block-row count, partials fold in band order,
// so results are identical at every thread count (and bit-identical to
// the sequential path).

/// Block rows below which `rayon` dispatch is not worth its overhead.
pub(crate) const PAR_MIN_ROWS: usize = 8;

/// Band count for `nb_y` block rows — pure function of `nb_y`.
#[inline]
pub(crate) fn n_bands(nb_y: usize) -> usize {
    if nb_y < PAR_MIN_ROWS {
        1
    } else {
        (nb_y / 8).min(32)
    }
}

/// `f(b)` over `0..nb` collected in band order — parallel under
/// `parallel`, sequential otherwise.
pub(crate) fn collect_bands<T: Send>(nb: usize, f: impl Fn(usize) -> T + Send + Sync) -> Vec<T> {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        (0..nb).into_par_iter().map(f).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        (0..nb).map(f).collect()
    }
}

// ---------------------------------------------------------------------
// Shared per-band body. `$F32`/`$LANES` select the vector width; the
// logic is identical at every width so all tiers agree bit-for-bit.
//
// Per batch, `LANES` consecutive blocks (repeating the last valid block
// in a short tail — its sums are computed but skipped) are gathered
// coefficient-major, DCT'd, masked and accumulated into per-lane f32
// sums, then folded into the returned f64 totals in raster order.
macro_rules! band_body {
    ($token:ident, $r:ident, $d:ident, $stride:ident, $by0:ident, $by1:ident, $nb_x:ident, $step:ident, $FV:ty, $LANES:literal) => {{
        type V = $FV;
        let zero = V::splat($token, 0.0);
        // Hoisted table splats — one broadcast per constant per call.
        let tv: [V; 64] = core::array::from_fn(|i| V::splat($token, DCT_T[i]));
        let cv: [V; 64] = core::array::from_fn(|i| V::splat($token, CSF[i]));
        let mv: [V; 64] = core::array::from_fn(|i| V::splat($token, MASK_COF[i]));
        let mut s1_glob = 0.0f64;
        let mut s2_glob = 0.0f64;
        for by in $by0..$by1 {
            let mut bx0 = 0usize;
            while bx0 < $nb_x {
                let valid = ($nb_x - bx0).min($LANES);
                let last = valid - 1;
                // --- gather, coefficient-major ----------------------
                let mut xa = [zero; 64];
                let mut xb = [zero; 64];
                for rr in 0..8 {
                    let base = (by * $step + rr) * $stride;
                    for m in 0..8 {
                        let c = rr * 8 + m;
                        xa[c] = V::from_array(
                            $token,
                            core::array::from_fn(|b| $r[base + (bx0 + b.min(last)) * $step + m]),
                        );
                        xb[c] = V::from_array(
                            $token,
                            core::array::from_fn(|b| $d[base + (bx0 + b.min(last)) * $step + m]),
                        );
                    }
                }
                // --- 2D DCT-II: out[k][l] = Σ_m Σ_n T[k][m]·x[m][n]·T[l][n]
                // pass 1 (rows): tmp[rr*8+l] = Σ_n T[l][n]·x[rr*8+n]
                // pass 2 (cols): out[k*8+l] = Σ_r T[k][r]·tmp[r*8+l]
                macro_rules! dct8x8 {
                    ($x:ident) => {{
                        let mut tmp = [zero; 64];
                        for rr in 0..8 {
                            for l in 0..8 {
                                let mut acc = zero;
                                for n in 0..8 {
                                    acc = acc + tv[l * 8 + n] * $x[rr * 8 + n];
                                }
                                tmp[rr * 8 + l] = acc;
                            }
                        }
                        let mut out = [zero; 64];
                        for k in 0..8 {
                            for l in 0..8 {
                                let mut acc = zero;
                                for rr in 0..8 {
                                    acc = acc + tv[k * 8 + rr] * tmp[rr * 8 + l];
                                }
                                out[k * 8 + l] = acc;
                            }
                        }
                        out
                    }};
                }
                let da = dct8x8!(xa);
                let db = dct8x8!(xb);
                // --- masking threshold (maskeff) --------------------
                macro_rules! maskeff {
                    ($x:ident, $dct:ident) => {{
                        // e = Σ_{c≠0} dct[c]² · MaskCof[c]
                        let mut e = zero;
                        for c in 1..64 {
                            let d2 = $dct[c] * $dct[c];
                            e = e + d2 * mv[c];
                        }
                        // whole-block vari = var(x)·64 = (SS/63)·64
                        let mut s = zero;
                        for c in 0..64 {
                            s = s + $x[c];
                        }
                        let mu = s / V::splat($token, 64.0);
                        let mut ss = zero;
                        for c in 0..64 {
                            let dv = $x[c] - mu;
                            ss = ss + dv * dv;
                        }
                        let vari = (ss / V::splat($token, 63.0)) * V::splat($token, 64.0);
                        // quadrant varis, MATLAB order TL+TR+BR+BL;
                        // each is var(quad)·16 = (SS_q/15)·16
                        macro_rules! qvari {
                            ($r0:literal, $c0:literal) => {{
                                let mut s = zero;
                                for rr in 0..4 {
                                    for c in 0..4 {
                                        s = s + $x[($r0 + rr) * 8 + $c0 + c];
                                    }
                                }
                                let mu = s / V::splat($token, 16.0);
                                let mut ss = zero;
                                for rr in 0..4 {
                                    for c in 0..4 {
                                        let dv = $x[($r0 + rr) * 8 + $c0 + c] - mu;
                                        ss = ss + dv * dv;
                                    }
                                }
                                (ss / V::splat($token, 15.0)) * V::splat($token, 16.0)
                            }};
                        }
                        let quadsum = ((qvari!(0, 0) + qvari!(0, 4)) + qvari!(4, 4)) + qvari!(4, 0);
                        let pop = V::blend(vari.simd_eq(zero), zero, quadsum / vari);
                        (e * pop).sqrt() / V::splat($token, 32.0)
                    }};
                }
                let maska = maskeff!(xa, da);
                let maskb = maskeff!(xb, db);
                let mask = maska.max(maskb);
                // --- per-coefficient energies ------------------------
                let mut s1 = zero;
                let mut s2 = zero;
                // DC (c == 0): unmasked in both sums.
                {
                    let u = (da[0] - db[0]).abs();
                    let t = u * cv[0];
                    s1 = s1 + t * t;
                    s2 = s2 + t * t;
                }
                for c in 1..64 {
                    let u = (da[c] - db[c]).abs();
                    let t2 = u * cv[c];
                    s2 = s2 + t2 * t2;
                    let thr = mask / mv[c];
                    let um = (u - thr).max(zero);
                    let t1 = um * cv[c];
                    s1 = s1 + t1 * t1;
                }
                // Fold this batch's per-block sums into the global f64
                // totals in raster order (padded tail lanes skipped).
                let s1a = s1.to_array();
                let s2a = s2.to_array();
                for b in 0..valid {
                    s1_glob += s1a[b] as f64;
                    s2_glob += s2a[b] as f64;
                }
                bx0 += $LANES;
            }
        }
        (s1_glob, s2_glob)
    }};
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
/// The `v3` tier of one block-row band — returns `(s1, s2)` f64 sums.
pub fn psnrhvs_band_v3(
    token: archmage::X64V3Token,
    r: &[f32],
    d: &[f32],
    stride: usize,
    by0: usize,
    by1: usize,
    nb_x: usize,
    step: usize,
) -> (f64, f64) {
    band_body!(
        token,
        r,
        d,
        stride,
        by0,
        by1,
        nb_x,
        step,
        magetypes::simd::generic::f32x8<archmage::X64V3Token>,
        8
    )
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::arcane]
/// The AVX-512 `v4` tier (16 blocks per batch).
pub fn psnrhvs_band_v4(
    token: archmage::X64V4Token,
    r: &[f32],
    d: &[f32],
    stride: usize,
    by0: usize,
    by1: usize,
    nb_x: usize,
    step: usize,
) -> (f64, f64) {
    band_body!(
        token,
        r,
        d,
        stride,
        by0,
        by1,
        nb_x,
        step,
        magetypes::simd::generic::f32x16<archmage::X64V4Token>,
        16
    )
}

#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
/// The `neon` tier (8 blocks per batch).
pub fn psnrhvs_band_neon(
    token: archmage::NeonToken,
    r: &[f32],
    d: &[f32],
    stride: usize,
    by0: usize,
    by1: usize,
    nb_x: usize,
    step: usize,
) -> (f64, f64) {
    band_body!(
        token,
        r,
        d,
        stride,
        by0,
        by1,
        nb_x,
        step,
        magetypes::simd::generic::f32x8<archmage::NeonToken>,
        8
    )
}

#[cfg(target_arch = "wasm32")]
#[archmage::arcane]
/// The `wasm128` tier (8 blocks per batch).
pub fn psnrhvs_band_wasm128(
    token: archmage::Wasm128Token,
    r: &[f32],
    d: &[f32],
    stride: usize,
    by0: usize,
    by1: usize,
    nb_x: usize,
    step: usize,
) -> (f64, f64) {
    band_body!(
        token,
        r,
        d,
        stride,
        by0,
        by1,
        nb_x,
        step,
        magetypes::simd::generic::f32x8<archmage::Wasm128Token>,
        8
    )
}

/// The `scalar` tier (8 blocks per batch, scalar-emulated lanes).
pub fn psnrhvs_band_scalar(
    token: archmage::ScalarToken,
    r: &[f32],
    d: &[f32],
    stride: usize,
    by0: usize,
    by1: usize,
    nb_x: usize,
    step: usize,
) -> (f64, f64) {
    band_body!(
        token,
        r,
        d,
        stride,
        by0,
        by1,
        nb_x,
        step,
        magetypes::simd::generic::f32x8<archmage::ScalarToken>,
        8
    )
}

/// Runtime-dispatched band scorer: the best tier this CPU supports.
/// Returns `(s1, s2)` — the band's masked (HVS-M) and unmasked (HVS)
/// f64 coefficient-energy sums over `nb_x`·(`by1`−`by0`) blocks.
pub(crate) fn psnrhvs_band(
    r: &[f32],
    d: &[f32],
    stride: usize,
    by0: usize,
    by1: usize,
    nb_x: usize,
    step: usize,
) -> (f64, f64) {
    archmage::incant!(
        psnrhvs_band(r, d, stride, by0, by1, nb_x, step),
        [v4, v3, neon, wasm128, scalar]
    )
}
