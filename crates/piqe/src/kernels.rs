//! SIMD kernels for the PIQE pipeline (archmage `#[attune]` / magetypes
//! generic vectors, like `crates/gmsd` and `crates/brisque`).
//!
//! Floating-point contract: every kernel preserves the scalar path's
//! per-element operation order exactly. The separable 7-tap blur
//! accumulates `acc += K[k]·v` in f64 (two roundings — OpenCV's
//! `sepFilter2D` does NOT fuse; Rust never contracts `a*b+c` into FMA), so
//! `mul_add` is never used: lanes are independent outputs and every
//! backend — scalar, wasm128, NEON, v3, v4x — produces bit-identical
//! planes.
//!
//! No pad planes: the 3-px border outputs keep the scalar clamped-index
//! path (≤6 rows/cols of work), the interior is vectorized directly on
//! the input — for outputs ≥3 and < n−3 every tap index `x+k−3` is already
//! in bounds, so the loaded lane run reads the same samples the clamps
//! selected, in the same k-ascending accumulate order.
//!
//! The `im²` conv is fused into the row pass: `v*v` per tap is the same
//! rounding the materialized `sq` plane stored, so `K[k]·(v²)` matches the
//! scalar `K[k]·sq[sx]` bitwise and removes one whole input plane.

use archmage::attune;

use alloc::vec;
use alloc::vec::Vec;

/// Scalar row-pass output for clamped borders: `acc += k·row[x+k−3]` with
/// `x+k−3` saturated into `0..w` — the oracle's exact tap order.
#[inline]
fn gauss_row_px(src: &[f64], w: usize, r: usize, x: usize, k: &[f64; 7]) -> (f64, f64) {
    let row = &src[r * w..r * w + w];
    let mut a = 0.0f64;
    let mut b = 0.0f64;
    for (i, &t) in k.iter().enumerate() {
        let v = row[(x + i).saturating_sub(3).min(w - 1)];
        a += t * v;
        b += t * (v * v);
    }
    (a, b)
}

/// Scalar row pass over `src`'s interior outputs `x ∈ [3, w−3)` for one
/// output row — no clamp needed, same arithmetic.
fn conv_rows_scalar(
    src: &[f64],
    w: usize,
    r: usize,
    x_lo: usize,
    x_hi: usize,
    k: &[f64; 7],
    mu: &mut [f64],
    e2: &mut [f64],
) {
    let row = &src[r * w..r * w + w];
    for x in x_lo..x_hi {
        let mut a = 0.0f64;
        let mut b = 0.0f64;
        for i in 0..7 {
            let v = row[x + i - 3];
            a += k[i] * v;
            b += k[i] * (v * v);
        }
        mu[x] = a;
        e2[x] = b;
    }
}

#[allow(unused_macros)]
macro_rules! conv_rows_body {
    ($token:ident, $F64:ident, $LANES:literal, $src:ident, $w:ident, $r:ident, $k:ident, $mu:ident, $e2:ident) => {{
        // Interior outputs only: x ∈ [3, w−3) reads row[x+i−3] in bounds.
        let row = &$src[$r * $w..$r * $w + $w];
        let (lo, hi) = (3usize, $w - 3);
        let chunks = (hi - lo) / $LANES;
        for g in 0..chunks {
            let c0 = lo + g * $LANES;
            let mut a = $F64::splat_t($token, 0.0);
            let mut b = $F64::splat_t($token, 0.0);
            for i in 0..7 {
                let kv = $F64::splat_t($token, $k[i]);
                let v = $F64::load_t(
                    $token,
                    &row[c0 + i - 3..c0 + i - 3 + $LANES].try_into().unwrap(),
                );
                // mul + add — two roundings like sepFilter2D, never fused.
                a += kv * v;
                b += kv * (v * v);
            }
            a.store((&mut $mu[c0..c0 + $LANES]).try_into().unwrap());
            b.store((&mut $e2[c0..c0 + $LANES]).try_into().unwrap());
        }
        conv_rows_scalar($src, $w, $r, lo + chunks * $LANES, hi, $k, $mu, $e2);
    }};
}

/// Row pass of both blurs for one output row of `src`: writes `mu`/`e2`
/// row slices (w elements). Border cols are handled scalar inside.
#[attune(_*_t, define(f64x4))]
fn conv_rows(
    token: Token,
    src: &[f64],
    w: usize,
    r: usize,
    k: &[f64; 7],
    mu: &mut [f64],
    e2: &mut [f64],
) {
    conv_rows_body!(token, f64x4, 4, src, w, r, k, mu, e2);
}

/// 8-wide variant for the opt-in AVX-512 `v4x` tier.
#[attune(_v4x_t(cfg(avx512)), define(f64x8))]
fn conv_rows(
    token: Token,
    src: &[f64],
    w: usize,
    r: usize,
    k: &[f64; 7],
    mu: &mut [f64],
    e2: &mut [f64],
) {
    conv_rows_body!(token, f64x8, 8, src, w, r, k, mu, e2);
}

/// Scalar col-pass output row `y` over clamped rows (border rows y<3,
/// y ≥ h−3) or interior row range — same `acc += t·tmp[sy·w+x]` order.
fn conv_cols_range_scalar(
    tm: &[f64],
    ts: &[f64],
    w: usize,
    h: usize,
    y: usize,
    k: &[f64; 7],
    mu: &mut [f64],
    e2: &mut [f64],
) {
    for x in 0..w {
        let mut a = 0.0f64;
        let mut b = 0.0f64;
        for (i, &t) in k.iter().enumerate() {
            let sy = (y + i).saturating_sub(3).min(h - 1);
            a += t * tm[sy * w + x];
            b += t * ts[sy * w + x];
        }
        mu[x] = a;
        e2[x] = b;
    }
}

fn conv_cols_scalar_interior(
    tm: &[f64],
    ts: &[f64],
    w: usize,
    y: usize,
    x_lo: usize,
    x_hi: usize,
    k: &[f64; 7],
    mu: &mut [f64],
    e2: &mut [f64],
) {
    for x in x_lo..x_hi {
        let mut a = 0.0f64;
        let mut b = 0.0f64;
        for i in 0..7 {
            a += k[i] * tm[(y + i - 3) * w + x];
            b += k[i] * ts[(y + i - 3) * w + x];
        }
        mu[x] = a;
        e2[x] = b;
    }
}

#[allow(unused_macros)]
macro_rules! conv_cols_body {
    ($token:ident, $F64:ident, $LANES:literal, $tm:ident, $ts:ident, $w:ident, $y:ident, $k:ident, $mu:ident, $e2:ident) => {{
        let n = $mu.len();
        let chunks = n / $LANES;
        for g in 0..chunks {
            let c0 = g * $LANES;
            let mut a = $F64::splat_t($token, 0.0);
            let mut b = $F64::splat_t($token, 0.0);
            for i in 0..7 {
                let kv = $F64::splat_t($token, $k[i]);
                let base = ($y + i - 3) * $w + c0;
                a += kv * $F64::load_t($token, &$tm[base..base + $LANES].try_into().unwrap());
                b += kv * $F64::load_t($token, &$ts[base..base + $LANES].try_into().unwrap());
            }
            a.store((&mut $mu[c0..c0 + $LANES]).try_into().unwrap());
            b.store((&mut $e2[c0..c0 + $LANES]).try_into().unwrap());
        }
        conv_cols_scalar_interior($tm, $ts, $w, $y, chunks * $LANES, n, $k, $mu, $e2);
    }};
}

/// Column pass of both blurs for one interior output row `y` (3 ≤ y <
/// h−3): `mu`/`e2` are w-element row slices of the output planes.
#[attune(_*_t, define(f64x4))]
fn conv_cols(
    token: Token,
    tm: &[f64],
    ts: &[f64],
    w: usize,
    y: usize,
    k: &[f64; 7],
    mu: &mut [f64],
    e2: &mut [f64],
) {
    conv_cols_body!(token, f64x4, 4, tm, ts, w, y, k, mu, e2);
}

/// 8-wide variant for the opt-in AVX-512 `v4x` tier.
#[attune(_v4x_t(cfg(avx512)), define(f64x8))]
fn conv_cols(
    token: Token,
    tm: &[f64],
    ts: &[f64],
    w: usize,
    y: usize,
    k: &[f64; 7],
    mu: &mut [f64],
    e2: &mut [f64],
) {
    conv_cols_body!(token, f64x8, 8, tm, ts, w, y, k, mu, e2);
}

/// `cv2.GaussianBlur` (7×7, BORDER_REPLICATE) on the w×h f64 `im` plane AND
/// on `im²` (the scalar path's `gauss7(&im)` + `gauss7(&sq)` fused): one
/// row pass computes both horizontal blurs — `v²` per tap IS the `sq`
/// element — then one column pass over both intermediates. Returns
/// `(mu, e2)`, bitwise-identical to the two scalar `gauss7` calls.
pub(crate) fn gauss7_pair(im: &[f64], w: usize, h: usize, k: &[f64; 7]) -> (Vec<f64>, Vec<f64>) {
    let mut tm = vec![0.0f64; w * h];
    let mut ts = vec![0.0f64; w * h];
    for r in 0..h {
        let m = &mut tm[r * w..r * w + w];
        let s = &mut ts[r * w..r * w + w];
        if w < 7 {
            for x in 0..w {
                let (a, b) = gauss_row_px(im, w, r, x, k);
                m[x] = a;
                s[x] = b;
            }
            continue;
        }
        // Border columns keep the scalar clamped-index path.
        for x in 0..3 {
            let (a, b) = gauss_row_px(im, w, r, x, k);
            m[x] = a;
            s[x] = b;
        }
        for x in w - 3..w {
            let (a, b) = gauss_row_px(im, w, r, x, k);
            m[x] = a;
            s[x] = b;
        }
        archmage::attuned!(
            conv_rows(im, w, r, k, m, s),
            [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
        );
    }
    let mut mu = vec![0.0f64; w * h];
    let mut e2 = vec![0.0f64; w * h];
    for y in 0..h {
        let m = &mut mu[y * w..y * w + w];
        let s = &mut e2[y * w..y * w + w];
        if y < 3 || y + 3 >= h {
            conv_cols_range_scalar(&tm, &ts, w, h, y, k, m, s);
        } else {
            archmage::attuned!(
                conv_cols(&tm, &ts, w, y, k, m, s),
                [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
            );
        }
    }
    (mu, e2)
}

/// Per-element `sigma = sqrt(|e2 − mu²|)`, `norm = (im − mu)/(sigma + 1)`
/// over the whole plane — every op elementwise-exact on any width.
#[attune(_*_t, +v4x(cfg(avx512)), define(f64x4))]
fn norm_plane(token: Token, im: &[f64], mu: &[f64], e2: &[f64], out: &mut [f64]) {
    let n = out.len();
    let chunks = n / 4;
    for i in 0..chunks {
        let x0 = i * 4;
        let mv = f64x4::load_t(token, &mu[x0..x0 + 4].try_into().unwrap());
        let sv = f64x4::load_t(token, &e2[x0..x0 + 4].try_into().unwrap());
        let sigma = (sv - mv * mv).abs().sqrt();
        let nv = (f64x4::load_t(token, &im[x0..x0 + 4].try_into().unwrap()) - mv)
            / (sigma + f64x4::splat_t(token, 1.0));
        nv.store((&mut out[x0..x0 + 4]).try_into().unwrap());
    }
    for i in chunks * 4..n {
        let sigma = (e2[i] - mu[i] * mu[i]).abs().sqrt();
        out[i] = (im[i] - mu[i]) / (sigma + 1.0);
    }
}

/// Whole-plane normalized f64 plane (the block loop's `b` values,
/// precomputed — identical per-element arithmetic).
pub(crate) fn normalize(im: &[f64], mu: &[f64], e2: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n];
    archmage::attuned!(
        norm_plane(im, mu, e2, &mut out),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
    out
}
