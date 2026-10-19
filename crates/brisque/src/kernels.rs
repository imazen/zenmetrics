//! Per-row SIMD kernels for the BRISQUE pipeline (archmage `#[attune]` /
//! magetypes generic vectors, like `crates/gmsd`).
//!
//! Floating-point contract: every kernel preserves the scalar path's
//! per-element operation order exactly. SIMD lanes are independent output
//! elements; a lane's FMA/mul/add/div/sqrt/round sequence is unchanged, so
//! every tier produces bit-identical planes. FMA appears only in kernels
//! dispatched to `v3`/`v4x`/`neon` (hardware FMA, one rounding — identical
//! to the scalar path's `fmaf`); the scalar tier keeps the `fmaf`-based
//! scalar loops. Elementwise kernels (`normalize`, `vres_row`) use only
//! correctly-rounded ops, so they are safe on every backend including
//! scalar and wasm128.
//!
//! `conv7` runs on a 3-px zero-padded input: taps that fell outside the
//! unpadded image read literal `0.0`, the same value the scalar bounds
//! check selected, so padded outputs match the scalar oracle bit-for-bit.

use archmage::{ScalarToken, attune};

use alloc::vec;
use alloc::vec::Vec;

use super::fma;

/// Zero-pad `x` (w×h) into a (w+6)×(h+6) plane.
pub(crate) fn pad3(x: &[f32], w: usize, h: usize) -> (Vec<f32>, usize) {
    let pw = w + 6;
    let mut out = vec![0f32; pw * (h + 6)];
    for r in 0..h {
        out[(r + 3) * pw + 3..(r + 3) * pw + 3 + w].copy_from_slice(&x[r * w..r * w + w]);
    }
    (out, pw)
}

/// One 7×7 conv output on a zero-padded plane: `out = Σ_i Σ_j k[7i+j]·p[(r+i)·pw + c+j]`.
/// Sequential row-major FMA chain — the torch conv2d order, and the same
/// order the vector kernel keeps per lane.
#[inline]
fn conv7_px_pad(xp: &[f32], pw: usize, k: &[f32; 49], r: usize, c: usize) -> f32 {
    let mut acc = 0f32;
    let base = r * pw + c;
    for i in 0..7 {
        for j in 0..7 {
            acc = fma(k[i * 7 + j], xp[base + i * pw + j], acc);
        }
    }
    acc
}

/// Scalar (and tier-tail) conv7_pair over interior output columns of row `r`.
/// The x² conv reads `xv·xv` on the fly — the same f32 product the
/// materialized x² plane held.
fn conv7_cols_scalar(
    x: &[f32],
    pw: usize,
    r: usize,
    c_lo: usize,
    c_hi: usize,
    k: &[f32; 49],
    mu: &mut [f32],
    sc: &mut [f32],
) {
    for c in c_lo..c_hi {
        mu[c] = conv7_px_pad(x, pw, k, r, c);
        sc[c] = {
            let mut acc = 0f32;
            let base = r * pw + c;
            for i in 0..7 {
                for j in 0..7 {
                    let xv = x[base + i * pw + j];
                    acc = fma(k[i * 7 + j], xv * xv, acc);
                }
            }
            acc
        };
    }
}

#[allow(unused_macros)]
macro_rules! conv7_cols_body {
    ($token:ident, $F32:ident, $LANES:literal, $x:ident, $pw:ident, $r:ident, $k:ident, $mu:ident, $sc:ident) => {{
        let n = $mu.len(); // interior col range is the whole padded row width minus 6
        let chunks = n / $LANES;
        for g in 0..chunks {
            let c0 = g * $LANES;
            let mut a = $F32::splat_t($token, 0.0);
            let mut b = $F32::splat_t($token, 0.0);
            for i in 0..7 {
                let base = ($r + i) * $pw + c0;
                for j in 0..7 {
                    let kv = $F32::splat_t($token, $k[i * 7 + j]);
                    let xv =
                        $F32::load_t($token, &$x[base + j..base + j + $LANES].try_into().unwrap());
                    a = kv.mul_add(xv, a);
                    // x² conv reads xv·xv — identical product bits to the
                    // old materialized x² plane.
                    b = kv.mul_add(xv * xv, b);
                }
            }
            a.store((&mut $mu[c0..c0 + $LANES]).try_into().unwrap());
            b.store((&mut $sc[c0..c0 + $LANES]).try_into().unwrap());
        }
        conv7_cols_scalar($x, $pw, $r, chunks * $LANES, n, $k, $mu, $sc);
    }};
}

/// Both 7×7 convs over one interior output row of the padded plane.
/// `mu`/`sc` are the w-element output row slices (output col c reads padded
/// input cols c..c+6 of padded row r..r+6).
#[attune(_*_t, -_scalar_t, -_wasm128_t, define(f32x8))]
fn conv7_cols(
    token: Token,
    x: &[f32],
    pw: usize,
    r: usize,
    k: &[f32; 49],
    mu: &mut [f32],
    sc: &mut [f32],
) {
    conv7_cols_body!(token, f32x8, 8, x, pw, r, k, mu, sc);
}

/// 16-wide variant for the opt-in AVX-512 `v4x` tier.
#[attune(_v4x_t(cfg(avx512)), define(f32x16))]
fn conv7_cols(
    token: Token,
    x: &[f32],
    pw: usize,
    r: usize,
    k: &[f32; 49],
    mu: &mut [f32],
    sc: &mut [f32],
) {
    conv7_cols_body!(token, f32x16, 16, x, pw, r, k, mu, sc);
}

/// `_scalar` arm for `attuned!` — the `fmaf` scalar path (same per-output
/// FMA chain as every other tier).
fn conv7_cols_scalar_t(
    _t: ScalarToken,
    x: &[f32],
    pw: usize,
    r: usize,
    k: &[f32; 49],
    mu: &mut [f32],
    sc: &mut [f32],
) {
    conv7_cols_scalar(x, pw, r, 0, mu.len(), k, mu, sc);
}

/// conv7 over a whole padded plane, twice (x and x²): `mu[c]`, `std_c[c]`
/// for output row r over output cols 0..out_w. `out_w = pw - 6`.
pub(crate) fn conv7_pair(
    x: &[f32],
    pw: usize,
    out_w: usize,
    k: &[f32; 49],
    mu: &mut [f32],
    sc: &mut [f32],
) {
    let rows = mu.len() / out_w;
    for r in 0..rows {
        let m = &mut mu[r * out_w..r * out_w + out_w];
        let s = &mut sc[r * out_w..r * out_w + out_w];
        archmage::attuned!(
            conv7_cols(x, pw, r, k, m, s),
            [_v4x(cfg(avx512)), _v3, _neon, _scalar]
        );
    }
}

/// piq `rgb2yiq` Y of one pixel, ties-even (bitwise vs the oracle).
#[inline]
pub(crate) fn to_y_px(r: u8, g: u8, b: u8) -> f32 {
    let m01 = r as f32 * 0.299 + g as f32 * 0.587;
    super::rint32(fma(b as f32, 0.114, m01))
}

/// Scalar (and tier-tail) luma for `rgb[..]` pixel count `c_lo..c_hi`.
fn luma_cols_scalar(rgb: &[u8], c_lo: usize, c_hi: usize, out: &mut [f32]) {
    for c in c_lo..c_hi {
        out[c] = to_y_px(rgb[c * 3], rgb[c * 3 + 1], rgb[c * 3 + 2]);
    }
}

#[allow(unused_macros)]
macro_rules! luma_cols_body {
    ($token:ident, $I32:ident, $F32:ident, $LANES:literal, $rgb:ident, $out:ident) => {{
        // A pixel's unaligned u32 word covers bytes 3x..3x+4; that stays
        // inside the row slice for x <= n − 2 (same reasoning as gmsd), so
        // the last pixel goes to the scalar tail.
        let n = $out.len();
        let nvec = n.saturating_sub(1);
        let chunks = nvec / $LANES;
        let c0299 = $F32::splat_t($token, 0.299);
        let c0587 = $F32::splat_t($token, 0.587);
        let c0114 = $F32::splat_t($token, 0.114);
        let m255 = $I32::splat_t($token, 0xff);
        for i in 0..chunks {
            let x0 = $LANES * i;
            let w = $I32::from_array_t(
                $token,
                core::array::from_fn(|k| {
                    u32::from_le_bytes($rgb[3 * (x0 + k)..3 * (x0 + k) + 4].try_into().unwrap())
                        as i32
                }),
            );
            let r = (w & m255).to_f32();
            let g = (w.shr_logical_const::<8>() & m255).to_f32();
            let b = (w.shr_logical_const::<16>() & m255).to_f32();
            // m01 = r*0.299 + g*0.587 stays two roundings (mul, then add) —
            // the oracle does not fuse it — then one FMA and ties-even round.
            let m01 = r * c0299 + g * c0587;
            let y = b.mul_add(c0114, m01).round();
            y.store((&mut $out[x0..x0 + $LANES]).try_into().unwrap());
        }
        luma_cols_scalar($rgb, chunks * $LANES, n, $out);
    }};
}

/// sRGB8 row → piq `rgb2yiq` Y (ties-even).
#[attune(_*_t, -_scalar_t, -_wasm128_t, define(i32x8, f32x8))]
fn luma_cols(token: Token, rgb: &[u8], out: &mut [f32]) {
    luma_cols_body!(token, i32x8, f32x8, 8, rgb, out);
}

/// 16-wide variant for the opt-in AVX-512 `v4x` tier.
#[attune(_v4x_t(cfg(avx512)), define(i32x16, f32x16))]
fn luma_cols(token: Token, rgb: &[u8], out: &mut [f32]) {
    luma_cols_body!(token, i32x16, f32x16, 16, rgb, out);
}

/// `_scalar` arm for `attuned!`.
fn luma_cols_scalar_t(_t: ScalarToken, rgb: &[u8], out: &mut [f32]) {
    luma_cols_scalar(rgb, 0, out.len(), out);
}

/// sRGB8 plane (strided rows) → Y f32 plane.
pub(crate) fn luma_plane(rgb: &[u8], w: usize, h: usize, stride: usize) -> Vec<f32> {
    let mut y = Vec::with_capacity(w * h);
    y.resize(w * h, 0.0);
    for r in 0..h {
        let row = &rgb[r * stride..r * stride + w * 3];
        let dst = &mut y[r * w..r * w + w];
        archmage::attuned!(
            luma_cols(row, dst),
            [_v4x(cfg(avx512)), _v3, _neon, _scalar]
        );
    }
    y
}

/// MSCN normalize one row: `out = (x − mu) / (sqrt(|std_c − mu²|) + 1)`.
/// Only mul/add/sub/abs/sqrt/div — correctly rounded on every backend, so
/// the scalar and wasm128 tiers are bitwise-identical too.
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
        let std = super::sqrt32((std_c[i] - mu[i] * mu[i]).abs()) + 1.0;
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
        let std = super::sqrt32((std_c[i] - mu[i] * mu[i]).abs()) + 1.0;
        out[i] = (x[i] - mu[i]) / std;
    }
}

/// Whole-plane MSCN normalize.
pub(crate) fn normalize(x: &[f32], mu: &[f32], std_c: &[f32], n: usize) -> Vec<f32> {
    let mut out = vec![0f32; n];
    archmage::attuned!(
        normalize_row(x, mu, std_c, &mut out),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
    out
}

/// One resize dot leaf: `xpad[tap] * weight`.
#[inline(always)]
fn tap_leaf(xp: &[f32], tap: usize, w: f32) -> f32 {
    xp[tap] * w
}

/// `outer_sum`'s recursive column tree over a lazy leaf sequence: splits
/// `[lo,hi)` at `mid = lo + (hi−lo)/2`, leaves read `leaf(tap)`.
fn tap_sum<F: Fn(usize) -> f32>(lo: usize, hi: usize, leaf: &F) -> f32 {
    match hi - lo {
        0 => 0.0,
        1 => leaf(lo),
        n => {
            let mid = lo + n / 2;
            tap_sum(lo, mid, leaf) + tap_sum(mid, hi, leaf)
        }
    }
}

/// Vertical resize: one output row `i` reads padded rows `base[i]+idx`,
/// `idx` in `0..ks`; leaf value `xpad[(base+idx)·pw + c]·wgt[idx·out + i]`.
/// `out_row` is the `out_w`-element destination. Per-output tree is the
/// `outer_sum` column order, per lane identical → bitwise vs scalar.
macro_rules! vres_row_body {
    ($token:ident, $F32:ident, $LANES:literal, $xp:ident, $pw:ident, $base:ident, $wgt:ident, $wstride:ident, $ks:ident, $i:ident, $out:ident) => {{
        let n = $out.len();
        let chunks = if $ks <= 16 { n / $LANES } else { 0 };
        for g in 0..chunks {
            let c0 = g * $LANES;
            let mut parts = [$F32::splat_t($token, 0.0); 16];
            for idx in 0..$ks {
                parts[idx] = $F32::load_t(
                    $token,
                    &$xp[($base + idx) * $pw + c0..($base + idx) * $pw + c0 + $LANES]
                        .try_into()
                        .unwrap(),
                ) * $F32::splat_t($token, $wgt[idx * $wstride + $i]);
            }
            // ks=10 tree unrolled — same `tap_sum` combine order per lane.
            let v = if $ks == 10 {
                ((parts[0] + parts[1]) + (parts[2] + (parts[3] + parts[4])))
                    + ((parts[5] + parts[6]) + (parts[7] + (parts[8] + parts[9])))
            } else {
                let tree = |lo: usize, hi: usize| -> $F32 {
                    fn t(p: &[$F32], lo: usize, hi: usize) -> $F32 {
                        match hi - lo {
                            1 => p[lo],
                            n => {
                                let mid = lo + n / 2;
                                t(p, lo, mid) + t(p, mid, hi)
                            }
                        }
                    }
                    t(&parts[..$ks], lo, hi)
                };
                tree(0, $ks)
            };
            v.store((&mut $out[c0..c0 + $LANES]).try_into().unwrap());
        }
        for c in chunks * $LANES..n {
            $out[c] = tap_sum(0, $ks, &|idx| {
                tap_leaf($xp, ($base + idx) * $pw + c, $wgt[idx * $wstride + $i])
            });
        }
    }};
}

/// One output row of the vertical resize pass (all tiers: mul+add only).
#[attune(_*_t, define(f32x8))]
fn vres_row(
    token: Token,
    xp: &[f32],
    pw: usize,
    base: usize,
    wgt: &[f32],
    wstride: usize,
    ks: usize,
    i: usize,
    out: &mut [f32],
) {
    vres_row_body!(token, f32x8, 8, xp, pw, base, wgt, wstride, ks, i, out);
}

/// 16-wide variant for the opt-in AVX-512 `v4x` tier.
#[attune(_v4x_t(cfg(avx512)), define(f32x16))]
fn vres_row(
    token: Token,
    xp: &[f32],
    pw: usize,
    base: usize,
    wgt: &[f32],
    wstride: usize,
    ks: usize,
    i: usize,
    out: &mut [f32],
) {
    vres_row_body!(token, f32x16, 16, xp, pw, base, wgt, wstride, ks, i, out);
}

/// Vertical-resize one output row, dispatched.
#[allow(clippy::too_many_arguments)]
pub(crate) fn vres_row_dispatch(
    xp: &[f32],
    pw: usize,
    base: usize,
    wgt: &[f32],
    wstride: usize,
    ks: usize,
    i: usize,
    out: &mut [f32],
) {
    archmage::attuned!(
        vres_row(xp, pw, base, wgt, wstride, ks, i, out),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
}

/// `tap_sum` over the resize column tree, for the horizontal pass and the
/// weight normalization (leaf = one padded sample times one weight, or a
/// raw table value). Same order as `outer_sum`'s `column`.
pub(crate) fn tap_tree<F: Fn(usize) -> f32>(lo: usize, hi: usize, leaf: &F) -> f32 {
    tap_sum(lo, hi, leaf)
}
