//! SIMD kernels for the NIQE pipeline (archmage `#[attune]` / magetypes
//! generic vectors, like `crates/gmsd` and `crates/brisque`).
//!
//! Floating-point contract: every kernel preserves the scalar path's
//! per-element operation order exactly. The convolutions accumulate in f64
//! with `acc = acc + x·k` (two roundings — scipy's convolve widens to f64
//! but does NOT fuse; Rust never contracts `a*b+c` into FMA), so `mul_add`
//! is never used: lanes are independent outputs and every backend —
//! including scalar and wasm128 — produces bit-identical planes.
//!
//! `conv_gauss` runs on a 3-px edge-replicated padded plane: taps that
//! clamped to the border read the replicated edge sample — the same value
//! the scalar `saturating_sub(..).min(h-1)` selected — and the (a,b) tap
//! order walks padded offsets `r+6-a`, `c+6-b`, preserving the oracle's
//! reversed accumulation order bitwise.

use archmage::attune;

use alloc::vec;
use alloc::vec::Vec;

use super::{GAUSS_WINDOW, sqrt32};

/// Edge-replicate pad `x` (w×h f32) into (w+6)×(h+6) planes of f64:
/// `pad[r+3][c+3] = f64::from(x[clamp r][clamp c])`, and the squared plane
/// `f64::from(x·x)` for the `sq` conv — the same f32 product bits the
/// materialized `sq` plane held, widened once per element instead of per tap.
pub(crate) fn pad_edge3_pair(x: &[f32], w: usize, h: usize) -> (Vec<f64>, Vec<f64>, usize) {
    let pw = w + 6;
    let mut xp = vec![0.0f64; pw * (h + 6)];
    let mut sp = vec![0.0f64; pw * (h + 6)];
    for r in 0..h + 6 {
        let sr = r.clamp(3, h + 2) - 3; // edge replicate: rows -3..h+2 → 0..h
        for c in 0..w + 6 {
            let sc = c.clamp(3, w + 2) - 3;
            let v = x[sr * w + sc];
            xp[r * pw + c] = f64::from(v);
            sp[r * pw + c] = f64::from(v * v);
        }
    }
    (xp, sp, pw)
}

/// One 7×7 f64 conv output over the padded plane, scalar (and tier tail):
/// `acc += xp[(r+6-a)·pw + c+6-b] · k[7a+b]` for a,b = 0..7 — the oracle's
/// `saturating_sub` tap order, read through the replicate pad.
#[inline]
fn conv7_px_f64(xp: &[f64], pw: usize, k: &[f64; 49], r: usize, c: usize) -> f64 {
    let mut acc = 0.0f64;
    for a in 0..7 {
        let base = (r + 6 - a) * pw;
        for b in 0..7 {
            acc += xp[base + c + 6 - b] * k[a * 7 + b];
        }
    }
    acc
}

/// Scalar (and tier-tail) conv pair over interior output cols `c_lo..c_hi`.
fn conv7_cols_scalar(
    xp: &[f64],
    sp: &[f64],
    pw: usize,
    r: usize,
    c_lo: usize,
    c_hi: usize,
    k: &[f64; 49],
    mu: &mut [f32],
    std_c: &mut [f32],
) {
    for c in c_lo..c_hi {
        mu[c] = conv7_px_f64(xp, pw, k, r, c) as f32;
        std_c[c] = conv7_px_f64(sp, pw, k, r, c) as f32;
    }
}

#[allow(unused_macros)]
macro_rules! conv7_cols_body {
    ($token:ident, $F64:ident, $LANES:literal, $xp:ident, $sp:ident, $pw:ident, $r:ident, $k:ident, $mu:ident, $sc:ident) => {{
        let n = $mu.len();
        let chunks = n / $LANES;
        for g in 0..chunks {
            let c0 = g * $LANES;
            let mut a = $F64::splat_t($token, 0.0);
            let mut b = $F64::splat_t($token, 0.0);
            for i in 0..7 {
                let base = ($r + 6 - i) * $pw + c0;
                for j in 0..7 {
                    let kv = $F64::splat_t($token, $k[i * 7 + j]);
                    // mul + add — two roundings like scipy, never fused.
                    a = a + $F64::load_t(
                        $token,
                        &$xp[base + 6 - j..base + 6 - j + $LANES].try_into().unwrap(),
                    ) * kv;
                    b = b + $F64::load_t(
                        $token,
                        &$sp[base + 6 - j..base + 6 - j + $LANES].try_into().unwrap(),
                    ) * kv;
                }
            }
            let mut av = [0.0f64; $LANES];
            let mut bv = [0.0f64; $LANES];
            a.store(&mut av);
            b.store(&mut bv);
            for (l, (x, s)) in av.iter().zip(bv.iter()).enumerate() {
                $mu[c0 + l] = *x as f32;
                $sc[c0 + l] = *s as f32;
            }
        }
        conv7_cols_scalar($xp, $sp, $pw, $r, chunks * $LANES, n, $k, $mu, $sc);
    }};
}

/// Both 7×7 convs over one interior output row of the padded planes.
/// `mu`/`sc` are w-element output row slices.
#[attune(_*_t, define(f64x4))]
fn conv7_cols(
    token: Token,
    xp: &[f64],
    sp: &[f64],
    pw: usize,
    r: usize,
    k: &[f64; 49],
    mu: &mut [f32],
    sc: &mut [f32],
) {
    conv7_cols_body!(token, f64x4, 4, xp, sp, pw, r, k, mu, sc);
}

/// 8-wide variant for the opt-in AVX-512 `v4x` tier.
#[attune(_v4x_t(cfg(avx512)), define(f64x8))]
fn conv7_cols(
    token: Token,
    xp: &[f64],
    sp: &[f64],
    pw: usize,
    r: usize,
    k: &[f64; 49],
    mu: &mut [f32],
    sc: &mut [f32],
) {
    conv7_cols_body!(token, f64x8, 8, xp, sp, pw, r, k, mu, sc);
}

/// `conv_gauss` over a whole padded plane, twice (x and x²).
pub(crate) fn conv7_pair(
    xp: &[f64],
    sp: &[f64],
    pw: usize,
    out_w: usize,
    mu: &mut [f32],
    sc: &mut [f32],
) {
    let rows = mu.len() / out_w;
    for r in 0..rows {
        let m = &mut mu[r * out_w..r * out_w + out_w];
        let s = &mut sc[r * out_w..r * out_w + out_w];
        archmage::attuned!(
            conv7_cols(xp, sp, pw, r, &GAUSS_WINDOW, m, s),
            [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
        );
    }
}

/// MSCN normalize one row: `out = (x − mu) / (sqrt(|std_c − mu²|) + 1)`.
/// Only mul/sub/abs/sqrt/div — correctly rounded on every backend.
#[attune(_*_t, define(f32x8))]
fn normalize_row(token: Token, x: &[f32], mu: &[f32], std_c: &[f32], out: &mut [f32]) {
    let n = out.len();
    let chunks = n / 8;
    for i in 0..chunks {
        let x0 = i * 8;
        let xv = f32x8::load_t(token, &x[x0..x0 + 8].try_into().unwrap());
        let mv = f32x8::load_t(token, &mu[x0..x0 + 8].try_into().unwrap());
        let sv = f32x8::load_t(token, &std_c[x0..x0 + 8].try_into().unwrap());
        let std = (sv - mv * mv).abs().sqrt() + f32x8::splat_t(token, 1.0);
        ((xv - mv) / std).store((&mut out[x0..x0 + 8]).try_into().unwrap());
    }
    for i in chunks * 8..n {
        let std = sqrt32((std_c[i] - mu[i] * mu[i]).abs()) + 1.0;
        out[i] = (x[i] - mu[i]) / std;
    }
}

/// 16-wide variant for the opt-in AVX-512 `v4x` tier.
#[attune(_v4x_t(cfg(avx512)), define(f32x16))]
fn normalize_row(token: Token, x: &[f32], mu: &[f32], std_c: &[f32], out: &mut [f32]) {
    let n = out.len();
    let chunks = n / 16;
    for i in 0..chunks {
        let x0 = i * 16;
        let xv = f32x16::load_t(token, &x[x0..x0 + 16].try_into().unwrap());
        let mv = f32x16::load_t(token, &mu[x0..x0 + 16].try_into().unwrap());
        let sv = f32x16::load_t(token, &std_c[x0..x0 + 16].try_into().unwrap());
        let std = (sv - mv * mv).abs().sqrt() + f32x16::splat_t(token, 1.0);
        ((xv - mv) / std).store((&mut out[x0..x0 + 16]).try_into().unwrap());
    }
    for i in chunks * 16..n {
        let std = sqrt32((std_c[i] - mu[i] * mu[i]).abs()) + 1.0;
        out[i] = (x[i] - mu[i]) / std;
    }
}

/// Whole-plane MSCN normalize (f32 out).
pub(crate) fn normalize(x: &[f32], mu: &[f32], std_c: &[f32], n: usize) -> Vec<f32> {
    let mut out = vec![0f32; n];
    archmage::attuned!(
        normalize_row(x, mu, std_c, &mut out),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
    out
}

/// numpy `pairwise_sum`'s ≤128-element leaf body: the 8-accumulator unroll
/// `r[j] += a[i+j]` — an 8-lane accumulate identical on every backend.
/// Returns the combined `((r0+r1)+(r2+r3)) + ((r4+r5)+(r6+r7))` plus the
/// sequential tail — the scalar leaf's exact arithmetic.
#[attune(_*_t, define(f32x8))]
fn pw_leaf_f32(token: Token, a: &[f32]) -> f32 {
    let n = a.len();
    let mut r = f32x8::load_t(token, &a[..8].try_into().unwrap());
    let mut i = 8;
    while i < n - (n % 8) {
        r = r + f32x8::load_t(token, &a[i..i + 8].try_into().unwrap());
        i += 8;
    }
    let mut rv = [0.0f32; 8];
    r.store(&mut rv);
    let mut res = ((rv[0] + rv[1]) + (rv[2] + rv[3])) + ((rv[4] + rv[5]) + (rv[6] + rv[7]));
    while i < n {
        res += a[i];
        i += 1;
    }
    res
}

/// f64 leaf: the same 8-accumulator unroll as two f64x4 lane groups —
/// `r0_3[j] += a[i+j]`, `r4_7[j] += a[i+4+j]`, per-lane identical adds.
#[attune(_*_t, define(f64x4))]
fn pw_leaf_f64(token: Token, a: &[f64]) -> f64 {
    let n = a.len();
    let mut r_lo = f64x4::load_t(token, &a[..4].try_into().unwrap());
    let mut r_hi = f64x4::load_t(token, &a[4..8].try_into().unwrap());
    let mut i = 8;
    while i < n - (n % 8) {
        r_lo = r_lo + f64x4::load_t(token, &a[i..i + 4].try_into().unwrap());
        r_hi = r_hi + f64x4::load_t(token, &a[i + 4..i + 8].try_into().unwrap());
        i += 8;
    }
    let mut lo = [0.0f64; 4];
    let mut hi = [0.0f64; 4];
    r_lo.store(&mut lo);
    r_hi.store(&mut hi);
    let mut res = ((lo[0] + lo[1]) + (lo[2] + lo[3])) + ((hi[0] + hi[1]) + (hi[2] + hi[3]));
    while i < n {
        res += a[i];
        i += 1;
    }
    res
}

/// `pairwise_sum_f32` leaf for 8 ≤ n ≤ 128 (caller's leaf range).
pub(crate) fn leaf_sum_f32(a: &[f32]) -> f32 {
    archmage::attuned!(
        pw_leaf_f32(a),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    )
}

/// `pairwise_sum_f64` leaf for 8 ≤ n ≤ 128.
pub(crate) fn leaf_sum_f64(a: &[f64]) -> f64 {
    archmage::attuned!(
        pw_leaf_f64(a),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    )
}
