//! Vectorized transcendental helpers backed by archmage / magetypes.
//!
//! The cvvdp hot path spends ~7% of wall time inside `__powf_fma`
//! called from `masking.rs::mult_mutual_band_into` — 6 `powf` calls
//! per pixel (3 `(x+eps)^q[ch] - eps^q[ch]` for the cross-channel
//! masking term + 3 `(|T-R|+eps)^p - eps^p` for the masked diff).
//!
//! `safe_pow_with_offset_into` is the vectorized replacement. Inputs
//! are pre-offset by `+SAFE_EPS = 1e-5` (so input domain is always
//! `>= 1e-5 > 0`), which lets us call magetypes' unchecked
//! `pow_midp_unchecked` (`exp2(n * log2(x))`, no edge-case branch).
//! Precision: log2 has ~3 ULP max, exp2 has ~1 ULP — composed gives
//! ≤128 ULP / ~1e-5 relative error on `x^p`. The mult_mutual_band
//! parity test allows 1e-3 relative tolerance; the 1e-4 JOD parity
//! gate at the top of the pipeline has plenty of margin.
//!
//! ## Reuse hooks for Chunk 5 (CSF SIMD)
//!
//! The CSF stage from `cvvdp-gpu` uses `f32::exp` / `f32::ln` /
//! `f32::powf` per pixel on the masked-contrast → sensitivity-scaling
//! step. To make porting that chunk a one-import job, we expose three
//! free-standing vectorized helpers:
//!
//! - `vexp_into(xs, out)` — `out[i] = exp(xs[i])`
//! - `vlog_into(xs, out)` — `out[i] = ln(xs[i])` (positive inputs)
//! - `vpow_into(xs, out, p)` — `out[i] = xs[i].powf(p)` (positive inputs)
//!
//! These wrap `exp_midp_unchecked` / `ln_midp_unchecked` /
//! `pow_midp_unchecked`. They route through `incant!` for runtime
//! dispatch, exactly like `safe_pow_with_offset_into`.
//!
//! Chunk 5 should call them with caller-owned input + output buffers
//! (parameters mirror `mult_mutual_band_into`'s buffer-recycle
//! convention). No allocation inside.

use alloc::vec::Vec;

use archmage::ScalarToken;
use magetypes::simd::backends::F32x8Convert;
use magetypes::simd::generic::f32x8 as GenericF32x8;

// `F32x8Convert` is the trait that gates the transcendental methods
// (`pow_midp_unchecked`, `exp_midp_unchecked`, `ln_midp_unchecked`).
// It's a strict superset of `F32x8Backend` adding float↔int bitcast.
// Every backend token we dispatch to (`ScalarToken`, `X64V3Token`,
// `NeonToken`, `Wasm128Token`) implements it.

// ---------------------------------------------------------------------------
// Generic SIMD kernels
// ---------------------------------------------------------------------------
//
// Each kernel takes a token (backend witness), input slice, and caller-owned
// output buffer. The body is generic over `T: F32x8Backend` so monomorphization
// per tier emits the native instruction set (AVX2 on x86, polyfilled 2×128-bit
// elsewhere). The scalar tail handles the last 0..7 elements.

/// `out[i] = (xs[i] + offset)^p - offset_pow_p` for every element.
///
/// All inputs must satisfy `xs[i] + offset > 0` so the unchecked
/// `pow_midp_unchecked` path is sound. The caller passes
/// `offset_pow_p = offset.powf(p)` (loop-invariant, computed once).
#[inline]
fn safe_pow_with_offset_kernel<T: F32x8Convert>(
    token: T,
    xs: &[f32],
    out: &mut [f32],
    offset: f32,
    p: f32,
    offset_pow_p: f32,
) {
    debug_assert_eq!(xs.len(), out.len());

    type F32x8<T> = GenericF32x8<T>;
    let offset_v = F32x8::<T>::splat(token, offset);
    let offset_pow_p_v = F32x8::<T>::splat(token, offset_pow_p);

    let (in_chunks, in_tail) = F32x8::<T>::partition_slice(token, xs);
    let (out_chunks, out_tail) = F32x8::<T>::partition_slice_mut(token, out);
    debug_assert_eq!(in_chunks.len(), out_chunks.len());

    for (in_chunk, out_chunk) in in_chunks.iter().zip(out_chunks.iter_mut()) {
        let x = F32x8::<T>::load(token, in_chunk);
        let shifted = x + offset_v;
        // pow_midp_unchecked = exp2_midp_unchecked(p * log2_midp_unchecked(x)).
        // Input is guaranteed > 0 because the caller pre-offsets by
        // `offset = SAFE_EPS > 0` and the magnitudes the masking
        // pipeline produces never go below 0.
        let raised = shifted.pow_midp_unchecked(p);
        let result = raised - offset_pow_p_v;
        result.store(out_chunk);
    }

    // Scalar tail — matches the SIMD lane semantics bit-for-bit:
    // same offset / p / subtraction order, just element-wise.
    for (xi, oi) in in_tail.iter().zip(out_tail.iter_mut()) {
        *oi = (xi + offset).powf(p) - offset_pow_p;
    }
}

/// `out[i] = exp(xs[i])`.
#[inline]
fn vexp_kernel<T: F32x8Convert>(token: T, xs: &[f32], out: &mut [f32]) {
    debug_assert_eq!(xs.len(), out.len());
    type F32x8<T> = GenericF32x8<T>;

    let (in_chunks, in_tail) = F32x8::<T>::partition_slice(token, xs);
    let (out_chunks, out_tail) = F32x8::<T>::partition_slice_mut(token, out);
    for (in_chunk, out_chunk) in in_chunks.iter().zip(out_chunks.iter_mut()) {
        F32x8::<T>::load(token, in_chunk)
            .exp_midp_unchecked()
            .store(out_chunk);
    }
    for (xi, oi) in in_tail.iter().zip(out_tail.iter_mut()) {
        *oi = xi.exp();
    }
}

/// `out[i] = ln(xs[i])`. Inputs must be `> 0`.
#[inline]
fn vlog_kernel<T: F32x8Convert>(token: T, xs: &[f32], out: &mut [f32]) {
    debug_assert_eq!(xs.len(), out.len());
    type F32x8<T> = GenericF32x8<T>;

    let (in_chunks, in_tail) = F32x8::<T>::partition_slice(token, xs);
    let (out_chunks, out_tail) = F32x8::<T>::partition_slice_mut(token, out);
    for (in_chunk, out_chunk) in in_chunks.iter().zip(out_chunks.iter_mut()) {
        F32x8::<T>::load(token, in_chunk)
            .ln_midp_unchecked()
            .store(out_chunk);
    }
    for (xi, oi) in in_tail.iter().zip(out_tail.iter_mut()) {
        *oi = xi.ln();
    }
}

/// `out[i] = xs[i]^p`. Inputs must be `> 0`.
#[inline]
fn vpow_kernel<T: F32x8Convert>(token: T, xs: &[f32], out: &mut [f32], p: f32) {
    debug_assert_eq!(xs.len(), out.len());
    type F32x8<T> = GenericF32x8<T>;

    let (in_chunks, in_tail) = F32x8::<T>::partition_slice(token, xs);
    let (out_chunks, out_tail) = F32x8::<T>::partition_slice_mut(token, out);
    for (in_chunk, out_chunk) in in_chunks.iter().zip(out_chunks.iter_mut()) {
        F32x8::<T>::load(token, in_chunk)
            .pow_midp_unchecked(p)
            .store(out_chunk);
    }
    for (xi, oi) in in_tail.iter().zip(out_tail.iter_mut()) {
        *oi = xi.powf(p);
    }
}

// ---------------------------------------------------------------------------
// Elementwise arithmetic kernels (video path — temporal FIR, masking glue,
// spatial pooling). Same conventions as the transcendental kernels above:
// caller-owned `out`, non-fused mul+add so the lane semantics mirror the
// scalar loops they replace.
// ---------------------------------------------------------------------------

/// `dst[i] += a * src[i]` — scalar-broadcast multiply-accumulate
/// (the temporal FIR inner loop).
#[inline]
fn vaxpy_kernel<T: F32x8Convert>(token: T, dst: &mut [f32], src: &[f32], a: f32) {
    debug_assert_eq!(dst.len(), src.len());
    type F32x8<T> = GenericF32x8<T>;
    let av = F32x8::<T>::splat(token, a);
    let (d_chunks, d_tail) = F32x8::<T>::partition_slice_mut(token, dst);
    let (s_chunks, s_tail) = F32x8::<T>::partition_slice(token, src);
    for (d_chunk, s_chunk) in d_chunks.iter_mut().zip(s_chunks.iter()) {
        let acc = F32x8::<T>::load(token, d_chunk);
        let v = F32x8::<T>::load(token, s_chunk);
        (acc + v * av).store(d_chunk);
    }
    for (di, si) in d_tail.iter_mut().zip(s_tail.iter()) {
        *di += a * *si;
    }
}

/// `dst[i] = a * src[i]` — scalar-broadcast multiply into a distinct
/// buffer (the post-blur `mask_c_lin` scaling).
#[inline]
fn vscale_kernel<T: F32x8Convert>(token: T, dst: &mut [f32], src: &[f32], a: f32) {
    debug_assert_eq!(dst.len(), src.len());
    type F32x8<T> = GenericF32x8<T>;
    let av = F32x8::<T>::splat(token, a);
    let (d_chunks, d_tail) = F32x8::<T>::partition_slice_mut(token, dst);
    let (s_chunks, s_tail) = F32x8::<T>::partition_slice(token, src);
    for (d_chunk, s_chunk) in d_chunks.iter_mut().zip(s_chunks.iter()) {
        (F32x8::<T>::load(token, s_chunk) * av).store(d_chunk);
    }
    for (di, si) in d_tail.iter_mut().zip(s_tail.iter()) {
        *di = *si * a;
    }
}

/// `out[i] = ((x[i] * a) * y[i]) * b` — two elementwise operands with
/// two scalar weights. The multiplication order matches the scalar
/// `band_mul * t * s * gain` expression exactly.
#[inline]
fn vmul2_scale2_kernel<T: F32x8Convert>(
    token: T,
    out: &mut [f32],
    x: &[f32],
    y: &[f32],
    a: f32,
    b: f32,
) {
    debug_assert_eq!(out.len(), x.len());
    debug_assert_eq!(out.len(), y.len());
    type F32x8<T> = GenericF32x8<T>;
    let av = F32x8::<T>::splat(token, a);
    let bv = F32x8::<T>::splat(token, b);
    let (o_chunks, o_tail) = F32x8::<T>::partition_slice_mut(token, out);
    let (x_chunks, x_tail) = F32x8::<T>::partition_slice(token, x);
    let (y_chunks, y_tail) = F32x8::<T>::partition_slice(token, y);
    for ((o_chunk, x_chunk), y_chunk) in o_chunks
        .iter_mut()
        .zip(x_chunks.iter())
        .zip(y_chunks.iter())
    {
        let xv = F32x8::<T>::load(token, x_chunk);
        let yv = F32x8::<T>::load(token, y_chunk);
        (((xv * av) * yv) * bv).store(o_chunk);
    }
    for ((oi, xi), yi) in o_tail.iter_mut().zip(x_tail.iter()).zip(y_tail.iter()) {
        *oi = ((*xi * a) * *yi) * b;
    }
}

/// `out[i] = |x[i] - y[i]|`.
#[inline]
fn vabs_diff_kernel<T: F32x8Convert>(token: T, out: &mut [f32], x: &[f32], y: &[f32]) {
    debug_assert_eq!(out.len(), x.len());
    debug_assert_eq!(out.len(), y.len());
    type F32x8<T> = GenericF32x8<T>;
    let (o_chunks, o_tail) = F32x8::<T>::partition_slice_mut(token, out);
    let (x_chunks, x_tail) = F32x8::<T>::partition_slice(token, x);
    let (y_chunks, y_tail) = F32x8::<T>::partition_slice(token, y);
    for ((o_chunk, x_chunk), y_chunk) in o_chunks
        .iter_mut()
        .zip(x_chunks.iter())
        .zip(y_chunks.iter())
    {
        let xv = F32x8::<T>::load(token, x_chunk);
        let yv = F32x8::<T>::load(token, y_chunk);
        (xv - yv).abs().store(o_chunk);
    }
    for ((oi, xi), yi) in o_tail.iter_mut().zip(x_tail.iter()).zip(y_tail.iter()) {
        *oi = (*xi - *yi).abs();
    }
}

/// `out[i] = |x[i] - y[i]| * w[i]` — the video baseband pooling input
/// (`abs(T-R) * S`).
#[inline]
fn vabs_diff_mul_kernel<T: F32x8Convert>(
    token: T,
    out: &mut [f32],
    x: &[f32],
    y: &[f32],
    w: &[f32],
) {
    debug_assert_eq!(out.len(), x.len());
    debug_assert_eq!(out.len(), y.len());
    debug_assert_eq!(out.len(), w.len());
    type F32x8<T> = GenericF32x8<T>;
    let (o_chunks, o_tail) = F32x8::<T>::partition_slice_mut(token, out);
    let (x_chunks, x_tail) = F32x8::<T>::partition_slice(token, x);
    let (y_chunks, y_tail) = F32x8::<T>::partition_slice(token, y);
    let (w_chunks, w_tail) = F32x8::<T>::partition_slice(token, w);
    for (((o_chunk, x_chunk), y_chunk), w_chunk) in o_chunks
        .iter_mut()
        .zip(x_chunks.iter())
        .zip(y_chunks.iter())
        .zip(w_chunks.iter())
    {
        let xv = F32x8::<T>::load(token, x_chunk);
        let yv = F32x8::<T>::load(token, y_chunk);
        let wv = F32x8::<T>::load(token, w_chunk);
        ((xv - yv).abs() * wv).store(o_chunk);
    }
    for (((oi, xi), yi), wi) in o_tail
        .iter_mut()
        .zip(x_tail.iter())
        .zip(y_tail.iter())
        .zip(w_tail.iter())
    {
        *oi = (*xi - *yi).abs() * *wi;
    }
}

/// `out[i] = min(|x[i]|, |y[i]|)` — the mutual-mask raw term.
#[inline]
fn vmin_abs_kernel<T: F32x8Convert>(token: T, out: &mut [f32], x: &[f32], y: &[f32]) {
    debug_assert_eq!(out.len(), x.len());
    debug_assert_eq!(out.len(), y.len());
    type F32x8<T> = GenericF32x8<T>;
    let (o_chunks, o_tail) = F32x8::<T>::partition_slice_mut(token, out);
    let (x_chunks, x_tail) = F32x8::<T>::partition_slice(token, x);
    let (y_chunks, y_tail) = F32x8::<T>::partition_slice(token, y);
    for ((o_chunk, x_chunk), y_chunk) in o_chunks
        .iter_mut()
        .zip(x_chunks.iter())
        .zip(y_chunks.iter())
    {
        let xv = F32x8::<T>::load(token, x_chunk);
        let yv = F32x8::<T>::load(token, y_chunk);
        xv.abs().min(yv.abs()).store(o_chunk);
    }
    for ((oi, xi), yi) in o_tail.iter_mut().zip(x_tail.iter()).zip(y_tail.iter()) {
        *oi = xi.abs().min(yi.abs());
    }
}

/// The 4-channel cross-channel pooling + soft clamp, fused per pixel:
///
/// ```text
/// m[cc]    = Σ_k w[k][cc] * t[k][i]          (k-order adds)
/// du       = d[cc][i] / (1 + m[cc])
/// d[cc][i] = d_max * du / (d_max + du)
/// ```
///
/// `d` is read and written in place; `t` is the four `term` planes.
/// `w` is the 4×4 `xcm_weights` matrix (row-major, `w[k][cc]`).
#[inline]
fn vxcm_pool_clamp_4ch_kernel<T: F32x8Convert>(
    token: T,
    d: &mut [&mut [f32]; 4],
    t: &[&[f32]; 4],
    w: &[[f32; 4]; 4],
    d_max: f32,
) {
    let n = d[0].len();
    debug_assert!(d.iter().all(|c| c.len() == n));
    debug_assert!(t.iter().all(|c| c.len() == n));
    type F32x8<T> = GenericF32x8<T>;
    let d_max_v = F32x8::<T>::splat(token, d_max);
    let one = F32x8::<T>::splat(token, 1.0);
    // Splat all 16 weights once — they are loop-invariant.
    let mut wv = [[F32x8::<T>::zero(token); 4]; 4];
    for (k, wv_row) in wv.iter_mut().enumerate() {
        for (cc, wv_e) in wv_row.iter_mut().enumerate() {
            *wv_e = F32x8::<T>::splat(token, w[k][cc]);
        }
    }

    let (t0c, t0t) = F32x8::<T>::partition_slice(token, t[0]);
    let (t1c, t1t) = F32x8::<T>::partition_slice(token, t[1]);
    let (t2c, t2t) = F32x8::<T>::partition_slice(token, t[2]);
    let (t3c, t3t) = F32x8::<T>::partition_slice(token, t[3]);
    let t_chunks = [t0c, t1c, t2c, t3c];
    let t_tails = [t0t, t1t, t2t, t3t];
    let [d0, d1, d2, d3] = d;
    let (d0c, d0t) = F32x8::<T>::partition_slice_mut(token, d0);
    let (d1c, d1t) = F32x8::<T>::partition_slice_mut(token, d1);
    let (d2c, d2t) = F32x8::<T>::partition_slice_mut(token, d2);
    let (d3c, d3t) = F32x8::<T>::partition_slice_mut(token, d3);
    let d_chunks = [d0c, d1c, d2c, d3c];
    let d_tails = [d0t, d1t, d2t, d3t];

    for i in 0..t_chunks[0].len() {
        let tv = [
            F32x8::<T>::load(token, &t_chunks[0][i]),
            F32x8::<T>::load(token, &t_chunks[1][i]),
            F32x8::<T>::load(token, &t_chunks[2][i]),
            F32x8::<T>::load(token, &t_chunks[3][i]),
        ];
        for cc in 0..4 {
            let m = wv[0][cc] * tv[0] + wv[1][cc] * tv[1] + wv[2][cc] * tv[2] + wv[3][cc] * tv[3];
            let dv = F32x8::<T>::load(token, &d_chunks[cc][i]);
            let du = dv / (one + m);
            ((d_max_v * du) / (d_max_v + du)).store(&mut d_chunks[cc][i]);
        }
    }
    for i in 0..t_tails[0].len() {
        for cc in 0..4 {
            let m = w[0][cc] * t_tails[0][i]
                + w[1][cc] * t_tails[1][i]
                + w[2][cc] * t_tails[2][i]
                + w[3][cc] * t_tails[3][i];
            let du = d_tails[cc][i] / (1.0 + m);
            d_tails[cc][i] = d_max * du / (d_max + du);
        }
    }
}

/// `lp_norm_mean` specialised to `p = 2` — the only exponent the
/// video path uses (`BETA_SPATIAL`). Matches `safe_pow_lp` semantics:
/// `mean_i[(|x_i| + eps)^2 − eps^2]` then `(|mean| + eps)^(1/2) −
/// eps^(1/2)` on the scalar result.
#[inline]
fn vlp_norm_mean_p2_kernel<T: F32x8Convert>(token: T, xs: &[f32]) -> f32 {
    const LP_SAFE_EPS: f32 = 1e-5;
    if xs.is_empty() {
        return 0.0;
    }
    type F32x8<T> = GenericF32x8<T>;
    let e = F32x8::<T>::splat(token, LP_SAFE_EPS);
    let e2 = F32x8::<T>::splat(token, LP_SAFE_EPS * LP_SAFE_EPS);
    let (chunks, tail) = F32x8::<T>::partition_slice(token, xs);
    let mut acc = F32x8::<T>::zero(token);
    for chunk in chunks {
        let u = F32x8::<T>::load(token, chunk).abs() + e;
        acc += u * u - e2;
    }
    let mut sum = acc.reduce_add();
    for &x in tail {
        let u = x.abs() + LP_SAFE_EPS;
        sum += u * u - LP_SAFE_EPS * LP_SAFE_EPS;
    }
    let mean = sum / xs.len() as f32;
    (mean + LP_SAFE_EPS).powf(0.5) - LP_SAFE_EPS.powf(0.5)
}

// ---------------------------------------------------------------------------
// Per-tier wrappers (named so `incant!` can suffix-resolve them).
// ---------------------------------------------------------------------------
//
// The `_scalar` wrapper goes through `GenericF32x8<ScalarToken>` (8-wide
// polyfill in scalar code). That keeps the scalar tail in-loop layout
// equivalent to the SIMD path so behaviour is bit-identical across tiers
// (matters because the parity test compares against the scalar `powf`
// baseline anyway, and we want one numerical model end-to-end).

pub(crate) fn safe_pow_with_offset_into_scalar(
    token: ScalarToken,
    xs: &[f32],
    out: &mut [f32],
    offset: f32,
    p: f32,
    offset_pow_p: f32,
) {
    safe_pow_with_offset_kernel(token, xs, out, offset, p, offset_pow_p);
}

pub(crate) fn vexp_into_scalar(token: ScalarToken, xs: &[f32], out: &mut [f32]) {
    vexp_kernel(token, xs, out);
}

pub(crate) fn vlog_into_scalar(token: ScalarToken, xs: &[f32], out: &mut [f32]) {
    vlog_kernel(token, xs, out);
}

pub(crate) fn vpow_into_scalar(token: ScalarToken, xs: &[f32], out: &mut [f32], p: f32) {
    vpow_kernel(token, xs, out, p);
}

pub(crate) fn vaxpy_into_scalar(token: ScalarToken, dst: &mut [f32], src: &[f32], a: f32) {
    vaxpy_kernel(token, dst, src, a);
}

pub(crate) fn vscale_into_scalar(token: ScalarToken, dst: &mut [f32], src: &[f32], a: f32) {
    vscale_kernel(token, dst, src, a);
}

pub(crate) fn vmul2_scale2_into_scalar(
    token: ScalarToken,
    out: &mut [f32],
    x: &[f32],
    y: &[f32],
    a: f32,
    b: f32,
) {
    vmul2_scale2_kernel(token, out, x, y, a, b);
}

pub(crate) fn vabs_diff_into_scalar(token: ScalarToken, out: &mut [f32], x: &[f32], y: &[f32]) {
    vabs_diff_kernel(token, out, x, y);
}

pub(crate) fn vabs_diff_mul_into_scalar(
    token: ScalarToken,
    out: &mut [f32],
    x: &[f32],
    y: &[f32],
    w: &[f32],
) {
    vabs_diff_mul_kernel(token, out, x, y, w);
}

pub(crate) fn vmin_abs_into_scalar(token: ScalarToken, out: &mut [f32], x: &[f32], y: &[f32]) {
    vmin_abs_kernel(token, out, x, y);
}

pub(crate) fn vxcm_pool_clamp_4ch_into_scalar(
    token: ScalarToken,
    d: &mut [&mut [f32]; 4],
    t: &[&[f32]; 4],
    w: &[[f32; 4]; 4],
    d_max: f32,
) {
    vxcm_pool_clamp_4ch_kernel(token, d, t, w, d_max);
}

pub(crate) fn vlp_norm_mean_p2_scalar(token: ScalarToken, xs: &[f32]) -> f32 {
    vlp_norm_mean_p2_kernel(token, xs)
}

// x86 / AVX2 + FMA tier — `_v3` suffix matches `X64V3Token`.
#[cfg(target_arch = "x86_64")]
mod x86_v3 {
    use super::*;
    use archmage::X64V3Token;

    #[archmage::arcane]
    pub(crate) fn safe_pow_with_offset_into_v3(
        token: X64V3Token,
        xs: &[f32],
        out: &mut [f32],
        offset: f32,
        p: f32,
        offset_pow_p: f32,
    ) {
        safe_pow_with_offset_kernel(token, xs, out, offset, p, offset_pow_p);
    }

    #[archmage::arcane]
    pub(crate) fn vexp_into_v3(token: X64V3Token, xs: &[f32], out: &mut [f32]) {
        vexp_kernel(token, xs, out);
    }

    #[archmage::arcane]
    pub(crate) fn vlog_into_v3(token: X64V3Token, xs: &[f32], out: &mut [f32]) {
        vlog_kernel(token, xs, out);
    }

    #[archmage::arcane]
    pub(crate) fn vpow_into_v3(token: X64V3Token, xs: &[f32], out: &mut [f32], p: f32) {
        vpow_kernel(token, xs, out, p);
    }

    #[archmage::arcane]
    pub(crate) fn vaxpy_into_v3(token: X64V3Token, dst: &mut [f32], src: &[f32], a: f32) {
        vaxpy_kernel(token, dst, src, a);
    }

    #[archmage::arcane]
    pub(crate) fn vscale_into_v3(token: X64V3Token, dst: &mut [f32], src: &[f32], a: f32) {
        vscale_kernel(token, dst, src, a);
    }

    #[archmage::arcane]
    pub(crate) fn vmul2_scale2_into_v3(
        token: X64V3Token,
        out: &mut [f32],
        x: &[f32],
        y: &[f32],
        a: f32,
        b: f32,
    ) {
        vmul2_scale2_kernel(token, out, x, y, a, b);
    }

    #[archmage::arcane]
    pub(crate) fn vabs_diff_into_v3(token: X64V3Token, out: &mut [f32], x: &[f32], y: &[f32]) {
        vabs_diff_kernel(token, out, x, y);
    }

    #[archmage::arcane]
    pub(crate) fn vabs_diff_mul_into_v3(
        token: X64V3Token,
        out: &mut [f32],
        x: &[f32],
        y: &[f32],
        w: &[f32],
    ) {
        vabs_diff_mul_kernel(token, out, x, y, w);
    }

    #[archmage::arcane]
    pub(crate) fn vmin_abs_into_v3(token: X64V3Token, out: &mut [f32], x: &[f32], y: &[f32]) {
        vmin_abs_kernel(token, out, x, y);
    }

    #[archmage::arcane]
    pub(crate) fn vxcm_pool_clamp_4ch_into_v3(
        token: X64V3Token,
        d: &mut [&mut [f32]; 4],
        t: &[&[f32]; 4],
        w: &[[f32; 4]; 4],
        d_max: f32,
    ) {
        vxcm_pool_clamp_4ch_kernel(token, d, t, w, d_max);
    }

    #[archmage::arcane]
    pub(crate) fn vlp_norm_mean_p2_v3(token: X64V3Token, xs: &[f32]) -> f32 {
        vlp_norm_mean_p2_kernel(token, xs)
    }
}
#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
use x86_v3::*;

// AArch64 / NEON tier — `_neon` suffix matches `NeonToken`.
#[cfg(target_arch = "aarch64")]
mod arm_neon {
    use super::*;
    use archmage::NeonToken;

    #[archmage::arcane]
    pub(crate) fn safe_pow_with_offset_into_neon(
        token: NeonToken,
        xs: &[f32],
        out: &mut [f32],
        offset: f32,
        p: f32,
        offset_pow_p: f32,
    ) {
        safe_pow_with_offset_kernel(token, xs, out, offset, p, offset_pow_p);
    }

    #[archmage::arcane]
    pub(crate) fn vexp_into_neon(token: NeonToken, xs: &[f32], out: &mut [f32]) {
        vexp_kernel(token, xs, out);
    }

    #[archmage::arcane]
    pub(crate) fn vlog_into_neon(token: NeonToken, xs: &[f32], out: &mut [f32]) {
        vlog_kernel(token, xs, out);
    }

    #[archmage::arcane]
    pub(crate) fn vpow_into_neon(token: NeonToken, xs: &[f32], out: &mut [f32], p: f32) {
        vpow_kernel(token, xs, out, p);
    }

    #[archmage::arcane]
    pub(crate) fn vaxpy_into_neon(token: NeonToken, dst: &mut [f32], src: &[f32], a: f32) {
        vaxpy_kernel(token, dst, src, a);
    }

    #[archmage::arcane]
    pub(crate) fn vscale_into_neon(token: NeonToken, dst: &mut [f32], src: &[f32], a: f32) {
        vscale_kernel(token, dst, src, a);
    }

    #[archmage::arcane]
    pub(crate) fn vmul2_scale2_into_neon(
        token: NeonToken,
        out: &mut [f32],
        x: &[f32],
        y: &[f32],
        a: f32,
        b: f32,
    ) {
        vmul2_scale2_kernel(token, out, x, y, a, b);
    }

    #[archmage::arcane]
    pub(crate) fn vabs_diff_into_neon(token: NeonToken, out: &mut [f32], x: &[f32], y: &[f32]) {
        vabs_diff_kernel(token, out, x, y);
    }

    #[archmage::arcane]
    pub(crate) fn vabs_diff_mul_into_neon(
        token: NeonToken,
        out: &mut [f32],
        x: &[f32],
        y: &[f32],
        w: &[f32],
    ) {
        vabs_diff_mul_kernel(token, out, x, y, w);
    }

    #[archmage::arcane]
    pub(crate) fn vmin_abs_into_neon(token: NeonToken, out: &mut [f32], x: &[f32], y: &[f32]) {
        vmin_abs_kernel(token, out, x, y);
    }

    #[archmage::arcane]
    pub(crate) fn vxcm_pool_clamp_4ch_into_neon(
        token: NeonToken,
        d: &mut [&mut [f32]; 4],
        t: &[&[f32]; 4],
        w: &[[f32; 4]; 4],
        d_max: f32,
    ) {
        vxcm_pool_clamp_4ch_kernel(token, d, t, w, d_max);
    }

    #[archmage::arcane]
    pub(crate) fn vlp_norm_mean_p2_neon(token: NeonToken, xs: &[f32]) -> f32 {
        vlp_norm_mean_p2_kernel(token, xs)
    }
}
#[cfg(target_arch = "aarch64")]
#[allow(unused_imports)]
use arm_neon::*;

// WASM SIMD128 tier — `_wasm128` suffix matches `Wasm128Token`.
#[cfg(target_arch = "wasm32")]
mod wasm_128 {
    use super::*;
    use archmage::Wasm128Token;

    #[archmage::arcane]
    pub(crate) fn safe_pow_with_offset_into_wasm128(
        token: Wasm128Token,
        xs: &[f32],
        out: &mut [f32],
        offset: f32,
        p: f32,
        offset_pow_p: f32,
    ) {
        safe_pow_with_offset_kernel(token, xs, out, offset, p, offset_pow_p);
    }

    #[archmage::arcane]
    pub(crate) fn vexp_into_wasm128(token: Wasm128Token, xs: &[f32], out: &mut [f32]) {
        vexp_kernel(token, xs, out);
    }

    #[archmage::arcane]
    pub(crate) fn vlog_into_wasm128(token: Wasm128Token, xs: &[f32], out: &mut [f32]) {
        vlog_kernel(token, xs, out);
    }

    #[archmage::arcane]
    pub(crate) fn vpow_into_wasm128(token: Wasm128Token, xs: &[f32], out: &mut [f32], p: f32) {
        vpow_kernel(token, xs, out, p);
    }

    #[archmage::arcane]
    pub(crate) fn vaxpy_into_wasm128(token: Wasm128Token, dst: &mut [f32], src: &[f32], a: f32) {
        vaxpy_kernel(token, dst, src, a);
    }

    #[archmage::arcane]
    pub(crate) fn vscale_into_wasm128(token: Wasm128Token, dst: &mut [f32], src: &[f32], a: f32) {
        vscale_kernel(token, dst, src, a);
    }

    #[archmage::arcane]
    pub(crate) fn vmul2_scale2_into_wasm128(
        token: Wasm128Token,
        out: &mut [f32],
        x: &[f32],
        y: &[f32],
        a: f32,
        b: f32,
    ) {
        vmul2_scale2_kernel(token, out, x, y, a, b);
    }

    #[archmage::arcane]
    pub(crate) fn vabs_diff_into_wasm128(
        token: Wasm128Token,
        out: &mut [f32],
        x: &[f32],
        y: &[f32],
    ) {
        vabs_diff_kernel(token, out, x, y);
    }

    #[archmage::arcane]
    pub(crate) fn vabs_diff_mul_into_wasm128(
        token: Wasm128Token,
        out: &mut [f32],
        x: &[f32],
        y: &[f32],
        w: &[f32],
    ) {
        vabs_diff_mul_kernel(token, out, x, y, w);
    }

    #[archmage::arcane]
    pub(crate) fn vmin_abs_into_wasm128(
        token: Wasm128Token,
        out: &mut [f32],
        x: &[f32],
        y: &[f32],
    ) {
        vmin_abs_kernel(token, out, x, y);
    }

    #[archmage::arcane]
    pub(crate) fn vxcm_pool_clamp_4ch_into_wasm128(
        token: Wasm128Token,
        d: &mut [&mut [f32]; 4],
        t: &[&[f32]; 4],
        w: &[[f32; 4]; 4],
        d_max: f32,
    ) {
        vxcm_pool_clamp_4ch_kernel(token, d, t, w, d_max);
    }

    #[archmage::arcane]
    pub(crate) fn vlp_norm_mean_p2_wasm128(token: Wasm128Token, xs: &[f32]) -> f32 {
        vlp_norm_mean_p2_kernel(token, xs)
    }
}
#[cfg(target_arch = "wasm32")]
#[allow(unused_imports)]
use wasm_128::*;

// ---------------------------------------------------------------------------
// Public dispatch — one runtime feature check per call.
// ---------------------------------------------------------------------------

/// Compute `out[i] = (xs[i] + offset)^p - offset_pow_p` for every
/// element of `xs`, writing into `out`. `out` must already be sized
/// to `xs.len()` (no resize inside — caller-owned scratch).
///
/// Precondition: `xs[i] + offset > 0` for every `i`. The masking
/// pipeline always satisfies this because the inputs are
/// `|magnitude| + SAFE_EPS` with `SAFE_EPS = 1e-5`.
///
/// `offset_pow_p` is `offset.powf(p)` — pass it once (hoisted by
/// the caller); the kernel reuses it on every lane.
#[inline]
pub(crate) fn safe_pow_with_offset_into(
    xs: &[f32],
    out: &mut [f32],
    offset: f32,
    p: f32,
    offset_pow_p: f32,
) {
    debug_assert_eq!(xs.len(), out.len());
    archmage::incant!(safe_pow_with_offset_into(xs, out, offset, p, offset_pow_p))
}

/// Convenience: resize `out` to `xs.len()` then dispatch.
///
/// Use this when the caller owns a reusable `Vec<f32>` scratch and
/// wants the size handled in one call (mirrors `Vec::clear` +
/// `Vec::resize` shape used elsewhere in the masking module).
#[inline]
#[allow(dead_code)]
pub(crate) fn safe_pow_with_offset_into_vec(
    xs: &[f32],
    out: &mut Vec<f32>,
    offset: f32,
    p: f32,
    offset_pow_p: f32,
) {
    out.clear();
    out.resize(xs.len(), 0.0);
    safe_pow_with_offset_into(xs, out.as_mut_slice(), offset, p, offset_pow_p)
}

/// `out[i] = exp(xs[i])`. Reusable from Chunk 5 (CSF SIMD).
#[inline]
#[allow(dead_code)]
pub(crate) fn vexp_into(xs: &[f32], out: &mut [f32]) {
    debug_assert_eq!(xs.len(), out.len());
    archmage::incant!(vexp_into(xs, out))
}

/// `out[i] = ln(xs[i])`. Positive inputs only. Reusable from Chunk 5.
#[inline]
#[allow(dead_code)]
pub(crate) fn vlog_into(xs: &[f32], out: &mut [f32]) {
    debug_assert_eq!(xs.len(), out.len());
    archmage::incant!(vlog_into(xs, out))
}

/// `out[i] = xs[i].powf(p)`. Positive inputs only. Reusable from Chunk 5.
#[inline]
#[allow(dead_code)]
pub(crate) fn vpow_into(xs: &[f32], out: &mut [f32], p: f32) {
    debug_assert_eq!(xs.len(), out.len());
    archmage::incant!(vpow_into(xs, out, p))
}

// ---------------------------------------------------------------------------
// Elementwise dispatchers — the video pipeline's glue loops.
// ---------------------------------------------------------------------------

/// `dst[i] += a * src[i]`. Temporal FIR accumulate.
#[inline]
pub(crate) fn vaxpy_into(dst: &mut [f32], src: &[f32], a: f32) {
    debug_assert_eq!(dst.len(), src.len());
    archmage::incant!(vaxpy_into(dst, src, a))
}

/// `dst[i] = a * src[i]` into a distinct buffer.
#[inline]
pub(crate) fn vscale_into(dst: &mut [f32], src: &[f32], a: f32) {
    debug_assert_eq!(dst.len(), src.len());
    archmage::incant!(vscale_into(dst, src, a))
}

/// `out[i] = ((x[i] * a) * y[i]) * b` — `band_mul * t * s * gain`
/// with identical op order to the scalar expression.
#[inline]
pub(crate) fn vmul2_scale2_into(out: &mut [f32], x: &[f32], y: &[f32], a: f32, b: f32) {
    debug_assert_eq!(out.len(), x.len());
    debug_assert_eq!(out.len(), y.len());
    archmage::incant!(vmul2_scale2_into(out, x, y, a, b))
}

/// `out[i] = |x[i] - y[i]|`.
#[inline]
pub(crate) fn vabs_diff_into(out: &mut [f32], x: &[f32], y: &[f32]) {
    debug_assert_eq!(out.len(), x.len());
    debug_assert_eq!(out.len(), y.len());
    archmage::incant!(vabs_diff_into(out, x, y))
}

/// `out[i] = |x[i] - y[i]| * w[i]` — video baseband `abs(T-R) * S`.
#[inline]
pub(crate) fn vabs_diff_mul_into(out: &mut [f32], x: &[f32], y: &[f32], w: &[f32]) {
    debug_assert_eq!(out.len(), x.len());
    debug_assert_eq!(out.len(), y.len());
    debug_assert_eq!(out.len(), w.len());
    archmage::incant!(vabs_diff_mul_into(out, x, y, w))
}

/// `out[i] = min(|x[i]|, |y[i]|)` — mutual-mask raw term.
#[inline]
pub(crate) fn vmin_abs_into(out: &mut [f32], x: &[f32], y: &[f32]) {
    debug_assert_eq!(out.len(), x.len());
    debug_assert_eq!(out.len(), y.len());
    archmage::incant!(vmin_abs_into(out, x, y))
}

/// Fused 4-channel cross-channel pool + soft clamp:
/// `d[cc] = d_max * du / (d_max + du)` where
/// `du = d[cc] / (1 + Σ_k w[k][cc] * t[k])`.
#[inline]
pub(crate) fn vxcm_pool_clamp_4ch_into(
    d: &mut [&mut [f32]; 4],
    t: &[&[f32]; 4],
    w: &[[f32; 4]; 4],
    d_max: f32,
) {
    archmage::incant!(vxcm_pool_clamp_4ch_into(d, t, w, d_max))
}

/// `lp_norm_mean(xs, 2.0)` — vectorised `safe_pow_lp` accumulation.
#[inline]
pub(crate) fn vlp_norm_mean_p2(xs: &[f32]) -> f32 {
    archmage::incant!(vlp_norm_mean_p2(xs))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Accuracy budget: the masking parity test downstream uses a
    /// 1e-3 relative tolerance on `mult_mutual_band` output; magetypes'
    /// `pow_midp_unchecked` is documented at ~1e-5 relative error /
    /// ≤128 ULP. We assert 5e-5 here — that's an order of magnitude
    /// tighter than the consumer needs, so any future regression
    /// (e.g. dropping to `pow_lowp`) is caught here, not silently in
    /// the parity gate.
    const REL_TOL: f32 = 5e-5;

    fn assert_close(got: f32, want: f32, idx: usize, ctx: &str) {
        let denom = want.abs().max(1e-6);
        let rel = (got - want).abs() / denom;
        assert!(
            rel <= REL_TOL,
            "{ctx} idx={idx}: got={got} want={want} rel={rel} > {REL_TOL}"
        );
    }

    #[test]
    fn safe_pow_matches_scalar_powf() {
        // Sweep typical masking-stage magnitudes (0 .. ~200) and
        // typical exponents (MASK_Q ∈ [1.3, 3.7], MASK_P ≈ 2.26).
        // Include sizes that cross the 8-lane boundary so the scalar
        // tail is exercised.
        let offset = 1e-5_f32;
        let exponents: &[f32] = &[1.3, 1.8, 2.26, 3.0, 3.7];
        let sizes: &[usize] = &[0, 1, 7, 8, 9, 15, 16, 17, 100, 1024];

        let mut s = 0x12345678_u32;
        let mut prng = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 8) as f32 / (1u32 << 24) as f32 * 200.0
        };

        for &n in sizes {
            let xs: Vec<f32> = (0..n).map(|_| prng()).collect();
            for &p in exponents {
                let offset_pow_p = offset.powf(p);
                let mut got = vec![0.0_f32; n];
                safe_pow_with_offset_into(&xs, &mut got, offset, p, offset_pow_p);
                for (i, &x) in xs.iter().enumerate() {
                    let want = (x + offset).powf(p) - offset_pow_p;
                    assert_close(got[i], want, i, &format!("n={n} p={p}"));
                }
            }
        }
    }

    #[test]
    fn vpow_matches_powf() {
        let exponents: &[f32] = &[0.5, 1.0, 1.3, 2.26, 3.0];
        let sizes: &[usize] = &[0, 7, 8, 17, 256];

        let mut s = 0x9abcdef0_u32;
        let mut prng = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            // Strictly positive — ln/log are undefined on 0.
            0.01 + (s >> 8) as f32 / (1u32 << 24) as f32 * 100.0
        };
        for &n in sizes {
            let xs: Vec<f32> = (0..n).map(|_| prng()).collect();
            for &p in exponents {
                let mut got = vec![0.0_f32; n];
                vpow_into(&xs, &mut got, p);
                for (i, &x) in xs.iter().enumerate() {
                    assert_close(got[i], x.powf(p), i, &format!("n={n} p={p}"));
                }
            }
        }
    }

    #[test]
    fn vexp_matches_exp() {
        // exp domain in the masking / CSF chain is roughly [-20, 20]
        // (log10 luminance ~5 max → x * ln(10) ~12). Cover that.
        let mut s = 0xdeadbeef_u32;
        let mut prng = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 40.0
        };
        for &n in &[0, 7, 8, 17, 256] {
            let xs: Vec<f32> = (0..n).map(|_| prng()).collect();
            let mut got = vec![0.0_f32; n];
            vexp_into(&xs, &mut got);
            for (i, &x) in xs.iter().enumerate() {
                let want = x.exp();
                let denom = want.abs().max(1e-12);
                let rel = (got[i] - want).abs() / denom;
                assert!(
                    rel <= 5e-5,
                    "vexp idx={i} x={x} got={} want={want} rel={rel}",
                    got[i]
                );
            }
        }
    }

    #[test]
    fn vlog_matches_ln() {
        let mut s = 0xcafebabe_u32;
        let mut prng = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            // Spread inputs across many orders of magnitude — the
            // CSF / contrast stage sees `log10(L_bkg)` over a wide
            // range. log2 has ~3 ULP error; relative error is largest
            // near 1 (ln(1)=0). We give the test the same epsilon
            // floor the consumer uses.
            (1e-4 + (s >> 8) as f32 / (1u32 << 24) as f32) * 10.0
        };
        for &n in &[0, 7, 8, 17, 256] {
            let xs: Vec<f32> = (0..n).map(|_| prng()).collect();
            let mut got = vec![0.0_f32; n];
            vlog_into(&xs, &mut got);
            for (i, &x) in xs.iter().enumerate() {
                let want = x.ln();
                let denom = want.abs().max(1e-4);
                let rel = (got[i] - want).abs() / denom;
                assert!(
                    rel <= 5e-4,
                    "vlog idx={i} x={x} got={} want={want} rel={rel}",
                    got[i]
                );
            }
        }
    }

    #[test]
    fn safe_pow_vec_resizes() {
        let xs: Vec<f32> = (0..23).map(|i| i as f32 * 0.5).collect();
        let mut out = Vec::new();
        let p = 2.26;
        let offset = 1e-5_f32;
        let offset_pow_p = offset.powf(p);
        safe_pow_with_offset_into_vec(&xs, &mut out, offset, p, offset_pow_p);
        assert_eq!(out.len(), xs.len());
        for (i, &x) in xs.iter().enumerate() {
            assert_close(out[i], (x + offset).powf(p) - offset_pow_p, i, "vec resize");
        }
    }
}
