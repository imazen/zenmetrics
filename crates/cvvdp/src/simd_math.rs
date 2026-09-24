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

/// Weber-contrast band fill (pyramid non-baseband levels):
/// `band[i] = clamp((fine[i] - img_exp[i]) / l, -1000, 1000)`,
/// `log[i] = log10(l)`, where `l = max(expanded_l[i], 0.01)`.
/// Replaces a scalar loop whose per-pixel `log10f` dominated the
/// video profile; the vector `ln * LOG10_E` differs from `log10f` by
/// ~1 ulp, far below the 1e-3 JOD gate.
#[inline]
fn vweber_band_kernel<T: F32x8Convert>(
    token: T,
    band: &mut [f32],
    log: &mut [f32],
    fine: &[f32],
    img_exp: &[f32],
    exp_l: &[f32],
) {
    debug_assert_eq!(band.len(), log.len());
    debug_assert_eq!(band.len(), fine.len());
    debug_assert_eq!(band.len(), img_exp.len());
    debug_assert_eq!(band.len(), exp_l.len());
    type F32x8<T> = GenericF32x8<T>;
    let floor_v = F32x8::<T>::splat(token, 0.01);
    let log10e = F32x8::<T>::splat(token, core::f32::consts::LOG10_E);
    let hi = F32x8::<T>::splat(token, 1000.0);
    let lo = F32x8::<T>::splat(token, -1000.0);
    let (b_chunks, b_tail) = F32x8::<T>::partition_slice_mut(token, band);
    let (l_chunks, l_tail) = F32x8::<T>::partition_slice_mut(token, log);
    let (f_chunks, f_tail) = F32x8::<T>::partition_slice(token, fine);
    let (e_chunks, e_tail) = F32x8::<T>::partition_slice(token, img_exp);
    let (x_chunks, x_tail) = F32x8::<T>::partition_slice(token, exp_l);
    for ((((b, lg), f), e), x) in b_chunks
        .iter_mut()
        .zip(l_chunks.iter_mut())
        .zip(f_chunks.iter())
        .zip(e_chunks.iter())
        .zip(x_chunks.iter())
    {
        let l = F32x8::<T>::load(token, x).max(floor_v);
        let c = (F32x8::<T>::load(token, f) - F32x8::<T>::load(token, e)) / l;
        c.min(hi).max(lo).store(b);
        (l.ln_midp_unchecked() * log10e).store(lg);
    }
    for ((((b, lg), f), e), x) in b_tail
        .iter_mut()
        .zip(l_tail.iter_mut())
        .zip(f_tail.iter())
        .zip(e_tail.iter())
        .zip(x_tail.iter())
    {
        let l = x.max(0.01);
        *b = ((f - e) / l).clamp(-1000.0, 1000.0);
        *lg = l.log10();
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

/// `out[i] = (|x[i] − y[i]| + offset)^p − offset_pow_p` — the
/// `vabs_diff` + `safe_pow_with_offset` pair fused into one pass
/// (the |diff| plane is never materialised). Same per-element op
/// order as the two-pass chain → bit-identical.
#[inline]
fn vabs_diff_pow_kernel<T: F32x8Convert>(
    token: T,
    out: &mut [f32],
    x: &[f32],
    y: &[f32],
    offset: f32,
    p: f32,
    offset_pow_p: f32,
) {
    debug_assert_eq!(out.len(), x.len());
    debug_assert_eq!(x.len(), y.len());
    type F32x8<T> = GenericF32x8<T>;
    let offset_v = F32x8::<T>::splat(token, offset);
    let offset_pow_p_v = F32x8::<T>::splat(token, offset_pow_p);
    let (o_chunks, o_tail) = F32x8::<T>::partition_slice_mut(token, out);
    let (x_chunks, x_tail) = F32x8::<T>::partition_slice(token, x);
    let (y_chunks, y_tail) = F32x8::<T>::partition_slice(token, y);
    for ((o_chunk, x_chunk), y_chunk) in o_chunks
        .iter_mut()
        .zip(x_chunks.iter())
        .zip(y_chunks.iter())
    {
        let d = (F32x8::<T>::load(token, x_chunk) - F32x8::<T>::load(token, y_chunk)).abs();
        ((d + offset_v).pow_midp_unchecked(p) - offset_pow_p_v).store(o_chunk);
    }
    for ((oi, xi), yi) in o_tail.iter_mut().zip(x_tail.iter()).zip(y_tail.iter()) {
        *oi = ((*xi - *yi).abs() + offset).powf(p) - offset_pow_p;
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

/// Weber-contrast non-baseband fill WITHOUT the `log_l_bkg` output —
/// identical `band` values to [`vweber_band_kernel`]; only the
/// reference achromatic pyramid's `log_l_bkg` is ever read, so the
/// other pyramids skip the `log10` and the plane write entirely.
#[inline]
fn vweber_band_nolog_kernel<T: F32x8Convert>(
    token: T,
    band: &mut [f32],
    fine: &[f32],
    img_exp: &[f32],
    exp_l: &[f32],
) {
    debug_assert_eq!(band.len(), fine.len());
    debug_assert_eq!(band.len(), img_exp.len());
    debug_assert_eq!(band.len(), exp_l.len());
    type F32x8<T> = GenericF32x8<T>;
    let floor_v = F32x8::<T>::splat(token, 0.01);
    let hi = F32x8::<T>::splat(token, 1000.0);
    let lo = F32x8::<T>::splat(token, -1000.0);
    let (b_chunks, b_tail) = F32x8::<T>::partition_slice_mut(token, band);
    let (f_chunks, f_tail) = F32x8::<T>::partition_slice(token, fine);
    let (e_chunks, e_tail) = F32x8::<T>::partition_slice(token, img_exp);
    let (x_chunks, x_tail) = F32x8::<T>::partition_slice(token, exp_l);
    for (((b, f), e), x) in b_chunks
        .iter_mut()
        .zip(f_chunks.iter())
        .zip(e_chunks.iter())
        .zip(x_chunks.iter())
    {
        let l = F32x8::<T>::load(token, x).max(floor_v);
        let c = (F32x8::<T>::load(token, f) - F32x8::<T>::load(token, e)) / l;
        c.min(hi).max(lo).store(b);
    }
    for (((b, f), e), x) in b_tail
        .iter_mut()
        .zip(f_tail.iter())
        .zip(e_tail.iter())
        .zip(x_tail.iter())
    {
        let l = x.max(0.01);
        *b = ((f - e) / l).clamp(-1000.0, 1000.0);
    }
}

/// `d1[i] = a1 * src[i]`, `d2[i] = a2 * src[i]` — one read of `src`
/// feeding two scale outputs. The transient channel's FIR reads the
/// same sustained-A planes as channel 0 with different taps.
#[inline]
fn vscale2_kernel<T: F32x8Convert>(
    token: T,
    d1: &mut [f32],
    d2: &mut [f32],
    src: &[f32],
    a1: f32,
    a2: f32,
) {
    debug_assert_eq!(d1.len(), src.len());
    debug_assert_eq!(d2.len(), src.len());
    type F32x8<T> = GenericF32x8<T>;
    let a1v = F32x8::<T>::splat(token, a1);
    let a2v = F32x8::<T>::splat(token, a2);
    let (d1c, d1t) = F32x8::<T>::partition_slice_mut(token, d1);
    let (d2c, d2t) = F32x8::<T>::partition_slice_mut(token, d2);
    let (s_chunks, s_tail) = F32x8::<T>::partition_slice(token, src);
    for ((o1, o2), s_chunk) in d1c.iter_mut().zip(d2c.iter_mut()).zip(s_chunks.iter()) {
        let v = F32x8::<T>::load(token, s_chunk);
        (v * a1v).store(o1);
        (v * a2v).store(o2);
    }
    for ((o1, o2), &sv) in d1t.iter_mut().zip(d2t.iter_mut()).zip(s_tail.iter()) {
        *o1 = sv * a1;
        *o2 = sv * a2;
    }
}

/// `d1[i] += a1 * src[i]`, `d2[i] += a2 * src[i]` — dual accumulator
/// for the shared sustained-A FIR input (channels 0 and 3).
#[inline]
fn vaxpy2_kernel<T: F32x8Convert>(
    token: T,
    d1: &mut [f32],
    d2: &mut [f32],
    src: &[f32],
    a1: f32,
    a2: f32,
) {
    debug_assert_eq!(d1.len(), src.len());
    debug_assert_eq!(d2.len(), src.len());
    type F32x8<T> = GenericF32x8<T>;
    let a1v = F32x8::<T>::splat(token, a1);
    let a2v = F32x8::<T>::splat(token, a2);
    let (d1c, d1t) = F32x8::<T>::partition_slice_mut(token, d1);
    let (d2c, d2t) = F32x8::<T>::partition_slice_mut(token, d2);
    let (s_chunks, s_tail) = F32x8::<T>::partition_slice(token, src);
    for ((o1, o2), s_chunk) in d1c.iter_mut().zip(d2c.iter_mut()).zip(s_chunks.iter()) {
        let v = F32x8::<T>::load(token, s_chunk);
        let acc1 = F32x8::<T>::load(token, o1);
        let acc2 = F32x8::<T>::load(token, o2);
        (acc1 + v * a1v).store(o1);
        (acc2 + v * a2v).store(o2);
    }
    for ((o1, o2), &sv) in d1t.iter_mut().zip(d2t.iter_mut()).zip(s_tail.iter()) {
        *o1 += a1 * sv;
        *o2 += a2 * sv;
    }
}

/// `dst[i] = Σ_k coeffs[k]·srcs[k][off + i]` — the temporal FIR as a
/// single fused pass: every source plane is read once per output
/// element instead of once per `vaxpy` tap call (~3× less plane
/// traffic). The per-element add sequence is `c0·s0` then
/// `acc + ck·sk` in ascending k — the identical order the
/// `vscale`-first-tap + `vaxpy` chain produces, so results are
/// bit-identical to the unfused loop.
#[inline]
fn vfir_into_kernel<T: F32x8Convert>(
    token: T,
    dst: &mut [f32],
    srcs: &[&[f32]],
    coeffs: &[f32],
    off: usize,
) {
    debug_assert_eq!(srcs.len(), coeffs.len());
    debug_assert!(!srcs.is_empty());
    type F32x8<T> = GenericF32x8<T>;
    let (d_chunks, d_tail) = F32x8::<T>::partition_slice_mut(token, dst);
    let c0 = F32x8::<T>::splat(token, coeffs[0]);
    let mut i = 0usize;
    for dc in d_chunks.iter_mut() {
        let b = off + i;
        let mut acc = F32x8::<T>::load(token, srcs[0][b..b + 8].try_into().unwrap()) * c0;
        for (s, &c) in srcs[1..].iter().zip(coeffs[1..].iter()) {
            let v = F32x8::<T>::load(token, s[b..b + 8].try_into().unwrap());
            acc += v * F32x8::<T>::splat(token, c);
        }
        acc.store(dc);
        i += 8;
    }
    for (j, d) in d_tail.iter_mut().enumerate() {
        let b = off + i + j;
        let mut acc = srcs[0][b] * coeffs[0];
        for (s, &c) in srcs[1..].iter().zip(coeffs[1..].iter()) {
            acc += s[b] * c;
        }
        *d = acc;
    }
}

/// Two fused FIR outputs over the same source planes —
/// `d0[i] = Σ_k c0[k]·srcs[k][off+i]`, `d1[i] = Σ_k c1[k]·srcs[k][off+i]`.
/// Channel 0 and channel 3 both filter the sustained-A plane with
/// different taps; sharing the loads halves the source traffic
/// again. Per-element add order matches the unfused tap chain.
#[inline]
fn vfir2_into_kernel<T: F32x8Convert>(
    token: T,
    d0: &mut [f32],
    d1: &mut [f32],
    srcs: &[&[f32]],
    coeffs0: &[f32],
    coeffs1: &[f32],
    off: usize,
) {
    debug_assert_eq!(d0.len(), d1.len());
    debug_assert_eq!(srcs.len(), coeffs0.len());
    debug_assert_eq!(srcs.len(), coeffs1.len());
    debug_assert!(!srcs.is_empty());
    type F32x8<T> = GenericF32x8<T>;
    let (d0c, d0t) = F32x8::<T>::partition_slice_mut(token, d0);
    let (d1c, d1t) = F32x8::<T>::partition_slice_mut(token, d1);
    let ca0 = F32x8::<T>::splat(token, coeffs0[0]);
    let cb0 = F32x8::<T>::splat(token, coeffs1[0]);
    let mut i = 0usize;
    for (o0, o1) in d0c.iter_mut().zip(d1c.iter_mut()) {
        let b = off + i;
        let v0 = F32x8::<T>::load(token, srcs[0][b..b + 8].try_into().unwrap());
        let mut acc0 = v0 * ca0;
        let mut acc1 = v0 * cb0;
        for ((s, &c0k), &c1k) in srcs[1..]
            .iter()
            .zip(coeffs0[1..].iter())
            .zip(coeffs1[1..].iter())
        {
            let v = F32x8::<T>::load(token, s[b..b + 8].try_into().unwrap());
            acc0 += v * F32x8::<T>::splat(token, c0k);
            acc1 += v * F32x8::<T>::splat(token, c1k);
        }
        acc0.store(o0);
        acc1.store(o1);
        i += 8;
    }
    for (j, (o0, o1)) in d0t.iter_mut().zip(d1t.iter_mut()).enumerate() {
        let b = off + i + j;
        let mut acc0 = srcs[0][b] * coeffs0[0];
        let mut acc1 = srcs[0][b] * coeffs1[0];
        for ((s, &c0k), &c1k) in srcs[1..]
            .iter()
            .zip(coeffs0[1..].iter())
            .zip(coeffs1[1..].iter())
        {
            acc0 += s[b] * c0k;
            acc1 += s[b] * c1k;
        }
        *o0 = acc0;
        *o1 = acc1;
    }
}

/// `o1[i] = ((x1[i]·a)·w[i])·b`, `o2[i] = ((x2[i]·a)·w[i])·b` — the
/// `vmul2_scale2` pair for test and reference CSF weighting in one
/// pass (the sensitivity map `w` is loaded once).
#[allow(clippy::too_many_arguments)]
#[inline]
fn vmul2_scale2_pair_kernel<T: F32x8Convert>(
    token: T,
    o1: &mut [f32],
    o2: &mut [f32],
    x1: &[f32],
    x2: &[f32],
    w: &[f32],
    a: f32,
    b: f32,
) {
    debug_assert_eq!(o1.len(), x1.len());
    debug_assert_eq!(o2.len(), x2.len());
    debug_assert_eq!(o1.len(), w.len());
    type F32x8<T> = GenericF32x8<T>;
    let av = F32x8::<T>::splat(token, a);
    let bv = F32x8::<T>::splat(token, b);
    let (o1c, o1t) = F32x8::<T>::partition_slice_mut(token, o1);
    let (o2c, o2t) = F32x8::<T>::partition_slice_mut(token, o2);
    let (x1c, x1t) = F32x8::<T>::partition_slice(token, x1);
    let (x2c, x2t) = F32x8::<T>::partition_slice(token, x2);
    let (w_chunks, w_tail) = F32x8::<T>::partition_slice(token, w);
    for ((((p1, p2), v1), v2), wv) in o1c
        .iter_mut()
        .zip(o2c.iter_mut())
        .zip(x1c.iter())
        .zip(x2c.iter())
        .zip(w_chunks.iter())
    {
        let wv = F32x8::<T>::load(token, wv);
        ((F32x8::<T>::load(token, v1) * av * wv) * bv).store(p1);
        ((F32x8::<T>::load(token, v2) * av * wv) * bv).store(p2);
    }
    for ((((p1, p2), &v1), &v2), &wv) in o1t
        .iter_mut()
        .zip(o2t.iter_mut())
        .zip(x1t.iter())
        .zip(x2t.iter())
        .zip(w_tail.iter())
    {
        *p1 = ((v1 * a) * wv) * b;
        *p2 = ((v2 * a) * wv) * b;
    }
}

/// Fused `|t[i]−r[i]|·s[i]` followed by the `lp_norm_mean_p2`
/// accumulation — identical chunk/lane order to running
/// `vabs_diff_mul_into` then `vlp_norm_mean_p2`, minus the
/// intermediate plane write+read. Returns the final scalar.
#[inline]
fn vabs_diff_mul_lp2_sum_kernel<T: F32x8Convert>(token: T, t: &[f32], r: &[f32], s: &[f32]) -> f32 {
    const LP_SAFE_EPS: f32 = 1e-5;
    let n = t.len();
    debug_assert_eq!(r.len(), n);
    debug_assert_eq!(s.len(), n);
    if n == 0 {
        return 0.0;
    }
    type F32x8<T> = GenericF32x8<T>;
    let e = F32x8::<T>::splat(token, LP_SAFE_EPS);
    let e2 = F32x8::<T>::splat(token, LP_SAFE_EPS * LP_SAFE_EPS);
    let (t_chunks, t_tail) = F32x8::<T>::partition_slice(token, t);
    let (r_chunks, r_tail) = F32x8::<T>::partition_slice(token, r);
    let (s_chunks, s_tail) = F32x8::<T>::partition_slice(token, s);
    let mut acc = F32x8::<T>::zero(token);
    for ((tc, rc), sc) in t_chunks.iter().zip(r_chunks.iter()).zip(s_chunks.iter()) {
        let d = (F32x8::<T>::load(token, tc) - F32x8::<T>::load(token, rc)).abs()
            * F32x8::<T>::load(token, sc);
        let u = d.abs() + e;
        acc += u * u - e2;
    }
    let mut sum = acc.reduce_add();
    for ((&tv, &rv), &sv) in t_tail.iter().zip(r_tail.iter()).zip(s_tail.iter()) {
        let d = (tv - rv).abs() * sv;
        let u = d.abs() + LP_SAFE_EPS;
        sum += u * u - LP_SAFE_EPS * LP_SAFE_EPS;
    }
    sum
}

/// Shared `(mean + eps)^0.5 − eps^0.5` tail for the `*_lp2` reduce
/// kernels — applied once to the (possibly band-folded) total sum.
#[inline]
pub(crate) fn lp2_finish(sum: f32, n: usize) -> f32 {
    const LP_SAFE_EPS: f32 = 1e-5;
    let mean = sum / n as f32;
    (mean + LP_SAFE_EPS).powf(0.5) - LP_SAFE_EPS.powf(0.5)
}

/// `vxcm_pool_clamp_4ch` without the `d` write — reads the `pow`
/// planes, computes the clamped diff in-register, and accumulates the
/// `lp_norm_mean_p2` sum per channel in the identical chunk/lane
/// order, returning the four final norms.
#[inline]
fn vxcm_pool_clamp_4ch_sqsum_partial_kernel<T: F32x8Convert>(
    token: T,
    d: &[&[f32]; 4],
    t: &[&[f32]; 4],
    w: &[[f32; 4]; 4],
    d_max: f32,
) -> [f32; 4] {
    const LP_SAFE_EPS: f32 = 1e-5;
    let n = d[0].len();
    debug_assert!(d.iter().all(|c| c.len() == n));
    debug_assert!(t.iter().all(|c| c.len() == n));
    type F32x8<T> = GenericF32x8<T>;
    let d_max_v = F32x8::<T>::splat(token, d_max);
    let one = F32x8::<T>::splat(token, 1.0);
    let e = F32x8::<T>::splat(token, LP_SAFE_EPS);
    let e2 = F32x8::<T>::splat(token, LP_SAFE_EPS * LP_SAFE_EPS);
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
    let (d0c, d0t) = F32x8::<T>::partition_slice(token, d[0]);
    let (d1c, d1t) = F32x8::<T>::partition_slice(token, d[1]);
    let (d2c, d2t) = F32x8::<T>::partition_slice(token, d[2]);
    let (d3c, d3t) = F32x8::<T>::partition_slice(token, d[3]);
    let d_chunks = [d0c, d1c, d2c, d3c];
    let d_tails = [d0t, d1t, d2t, d3t];

    let mut acc = [F32x8::<T>::zero(token); 4];
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
            let fin = (d_max_v * du) / (d_max_v + du);
            let u = fin.abs() + e;
            acc[cc] += u * u - e2;
        }
    }
    let mut sums = [
        acc[0].reduce_add(),
        acc[1].reduce_add(),
        acc[2].reduce_add(),
        acc[3].reduce_add(),
    ];
    for i in 0..t_tails[0].len() {
        for cc in 0..4 {
            let m = w[0][cc] * t_tails[0][i]
                + w[1][cc] * t_tails[1][i]
                + w[2][cc] * t_tails[2][i]
                + w[3][cc] * t_tails[3][i];
            let du = d_tails[cc][i] / (1.0 + m);
            let fin = d_max * du / (d_max + du);
            let u = fin.abs() + LP_SAFE_EPS;
            sums[cc] += u * u - LP_SAFE_EPS * LP_SAFE_EPS;
        }
    }
    sums
}

/// Shared `powf(0.5)` normalisation tail for
/// [`vxcm_pool_clamp_4ch_sqsum_partial`] — applied once to the
/// (possibly band-folded) per-channel sums.
#[inline]
pub(crate) fn xcm4_finish(sums: [f32; 4], n: usize) -> [f32; 4] {
    const LP_SAFE_EPS: f32 = 1e-5;
    let nf = n as f32;
    [
        (sums[0] / nf + LP_SAFE_EPS).powf(0.5) - LP_SAFE_EPS.powf(0.5),
        (sums[1] / nf + LP_SAFE_EPS).powf(0.5) - LP_SAFE_EPS.powf(0.5),
        (sums[2] / nf + LP_SAFE_EPS).powf(0.5) - LP_SAFE_EPS.powf(0.5),
        (sums[3] / nf + LP_SAFE_EPS).powf(0.5) - LP_SAFE_EPS.powf(0.5),
    ]
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

pub(crate) fn vabs_diff_into_scalar(token: ScalarToken, out: &mut [f32], x: &[f32], y: &[f32]) {
    vabs_diff_kernel(token, out, x, y);
}

pub(crate) fn vmin_abs_into_scalar(token: ScalarToken, out: &mut [f32], x: &[f32], y: &[f32]) {
    vmin_abs_kernel(token, out, x, y);
}

pub(crate) fn vweber_band_into_scalar(
    token: ScalarToken,
    band: &mut [f32],
    log: &mut [f32],
    fine: &[f32],
    img_exp: &[f32],
    exp_l: &[f32],
) {
    vweber_band_kernel(token, band, log, fine, img_exp, exp_l);
}

pub(crate) fn vweber_band_nolog_into_scalar(
    token: ScalarToken,
    band: &mut [f32],
    fine: &[f32],
    img_exp: &[f32],
    exp_l: &[f32],
) {
    vweber_band_nolog_kernel(token, band, fine, img_exp, exp_l);
}

pub(crate) fn vscale2_into_scalar(
    token: ScalarToken,
    d1: &mut [f32],
    d2: &mut [f32],
    src: &[f32],
    a1: f32,
    a2: f32,
) {
    vscale2_kernel(token, d1, d2, src, a1, a2);
}

pub(crate) fn vaxpy2_into_scalar(
    token: ScalarToken,
    d1: &mut [f32],
    d2: &mut [f32],
    src: &[f32],
    a1: f32,
    a2: f32,
) {
    vaxpy2_kernel(token, d1, d2, src, a1, a2);
}

pub(crate) fn vmul2_scale2_pair_into_scalar(
    token: ScalarToken,
    o1: &mut [f32],
    o2: &mut [f32],
    x1: &[f32],
    x2: &[f32],
    w: &[f32],
    a: f32,
    b: f32,
) {
    vmul2_scale2_pair_kernel(token, o1, o2, x1, x2, w, a, b);
}

pub(crate) fn vfir_into_scalar(
    token: ScalarToken,
    dst: &mut [f32],
    srcs: &[&[f32]],
    coeffs: &[f32],
    off: usize,
) {
    vfir_into_kernel(token, dst, srcs, coeffs, off)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn vfir2_into_scalar(
    token: ScalarToken,
    d0: &mut [f32],
    d1: &mut [f32],
    srcs: &[&[f32]],
    coeffs0: &[f32],
    coeffs1: &[f32],
    off: usize,
) {
    vfir2_into_kernel(token, d0, d1, srcs, coeffs0, coeffs1, off)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn vabs_diff_pow_scalar(
    token: ScalarToken,
    out: &mut [f32],
    x: &[f32],
    y: &[f32],
    offset: f32,
    p: f32,
    offset_pow_p: f32,
) {
    vabs_diff_pow_kernel(token, out, x, y, offset, p, offset_pow_p)
}

pub(crate) fn vabs_diff_mul_lp2_sum_scalar(
    token: ScalarToken,
    t: &[f32],
    r: &[f32],
    s: &[f32],
) -> f32 {
    vabs_diff_mul_lp2_sum_kernel(token, t, r, s)
}

pub(crate) fn vxcm_pool_clamp_4ch_sqsum_partial_scalar(
    token: ScalarToken,
    d: &[&[f32]; 4],
    t: &[&[f32]; 4],
    w: &[[f32; 4]; 4],
    d_max: f32,
) -> [f32; 4] {
    vxcm_pool_clamp_4ch_sqsum_partial_kernel(token, d, t, w, d_max)
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
    pub(crate) fn vabs_diff_into_v3(token: X64V3Token, out: &mut [f32], x: &[f32], y: &[f32]) {
        vabs_diff_kernel(token, out, x, y);
    }

    #[archmage::arcane]
    pub(crate) fn vmin_abs_into_v3(token: X64V3Token, out: &mut [f32], x: &[f32], y: &[f32]) {
        vmin_abs_kernel(token, out, x, y);
    }

    #[archmage::arcane]
    pub(crate) fn vweber_band_into_v3(
        token: X64V3Token,
        band: &mut [f32],
        log: &mut [f32],
        fine: &[f32],
        img_exp: &[f32],
        exp_l: &[f32],
    ) {
        vweber_band_kernel(token, band, log, fine, img_exp, exp_l);
    }

    #[archmage::arcane]
    pub(crate) fn vweber_band_nolog_into_v3(
        token: X64V3Token,
        band: &mut [f32],
        fine: &[f32],
        img_exp: &[f32],
        exp_l: &[f32],
    ) {
        vweber_band_nolog_kernel(token, band, fine, img_exp, exp_l);
    }

    #[archmage::arcane]
    pub(crate) fn vscale2_into_v3(
        token: X64V3Token,
        d1: &mut [f32],
        d2: &mut [f32],
        src: &[f32],
        a1: f32,
        a2: f32,
    ) {
        vscale2_kernel(token, d1, d2, src, a1, a2);
    }

    #[archmage::arcane]
    pub(crate) fn vaxpy2_into_v3(
        token: X64V3Token,
        d1: &mut [f32],
        d2: &mut [f32],
        src: &[f32],
        a1: f32,
        a2: f32,
    ) {
        vaxpy2_kernel(token, d1, d2, src, a1, a2);
    }

    #[archmage::arcane]
    pub(crate) fn vmul2_scale2_pair_into_v3(
        token: X64V3Token,
        o1: &mut [f32],
        o2: &mut [f32],
        x1: &[f32],
        x2: &[f32],
        w: &[f32],
        a: f32,
        b: f32,
    ) {
        vmul2_scale2_pair_kernel(token, o1, o2, x1, x2, w, a, b);
    }

    #[archmage::arcane]
    pub(crate) fn vfir_into_v3(
        token: X64V3Token,
        dst: &mut [f32],
        srcs: &[&[f32]],
        coeffs: &[f32],
        off: usize,
    ) {
        vfir_into_kernel(token, dst, srcs, coeffs, off)
    }

    #[archmage::arcane]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn vfir2_into_v3(
        token: X64V3Token,
        d0: &mut [f32],
        d1: &mut [f32],
        srcs: &[&[f32]],
        coeffs0: &[f32],
        coeffs1: &[f32],
        off: usize,
    ) {
        vfir2_into_kernel(token, d0, d1, srcs, coeffs0, coeffs1, off)
    }

    #[archmage::arcane]
    pub(crate) fn vabs_diff_mul_lp2_sum_v3(
        token: X64V3Token,
        t: &[f32],
        r: &[f32],
        s: &[f32],
    ) -> f32 {
        vabs_diff_mul_lp2_sum_kernel(token, t, r, s)
    }

    #[allow(clippy::too_many_arguments)]
    #[archmage::arcane]
    pub(crate) fn vabs_diff_pow_v3(
        token: X64V3Token,
        out: &mut [f32],
        x: &[f32],
        y: &[f32],
        offset: f32,
        p: f32,
        offset_pow_p: f32,
    ) {
        vabs_diff_pow_kernel(token, out, x, y, offset, p, offset_pow_p)
    }

    #[archmage::arcane]
    pub(crate) fn vxcm_pool_clamp_4ch_sqsum_partial_v3(
        token: X64V3Token,
        d: &[&[f32]; 4],
        t: &[&[f32]; 4],
        w: &[[f32; 4]; 4],
        d_max: f32,
    ) -> [f32; 4] {
        vxcm_pool_clamp_4ch_sqsum_partial_kernel(token, d, t, w, d_max)
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
    pub(crate) fn vabs_diff_into_neon(token: NeonToken, out: &mut [f32], x: &[f32], y: &[f32]) {
        vabs_diff_kernel(token, out, x, y);
    }

    #[archmage::arcane]
    pub(crate) fn vmin_abs_into_neon(token: NeonToken, out: &mut [f32], x: &[f32], y: &[f32]) {
        vmin_abs_kernel(token, out, x, y);
    }

    #[archmage::arcane]
    pub(crate) fn vweber_band_into_neon(
        token: NeonToken,
        band: &mut [f32],
        log: &mut [f32],
        fine: &[f32],
        img_exp: &[f32],
        exp_l: &[f32],
    ) {
        vweber_band_kernel(token, band, log, fine, img_exp, exp_l);
    }

    #[archmage::arcane]
    pub(crate) fn vweber_band_nolog_into_neon(
        token: NeonToken,
        band: &mut [f32],
        fine: &[f32],
        img_exp: &[f32],
        exp_l: &[f32],
    ) {
        vweber_band_nolog_kernel(token, band, fine, img_exp, exp_l);
    }

    #[archmage::arcane]
    pub(crate) fn vscale2_into_neon(
        token: NeonToken,
        d1: &mut [f32],
        d2: &mut [f32],
        src: &[f32],
        a1: f32,
        a2: f32,
    ) {
        vscale2_kernel(token, d1, d2, src, a1, a2);
    }

    #[archmage::arcane]
    pub(crate) fn vaxpy2_into_neon(
        token: NeonToken,
        d1: &mut [f32],
        d2: &mut [f32],
        src: &[f32],
        a1: f32,
        a2: f32,
    ) {
        vaxpy2_kernel(token, d1, d2, src, a1, a2);
    }

    #[archmage::arcane]
    pub(crate) fn vmul2_scale2_pair_into_neon(
        token: NeonToken,
        o1: &mut [f32],
        o2: &mut [f32],
        x1: &[f32],
        x2: &[f32],
        w: &[f32],
        a: f32,
        b: f32,
    ) {
        vmul2_scale2_pair_kernel(token, o1, o2, x1, x2, w, a, b);
    }

    #[archmage::arcane]
    pub(crate) fn vfir_into_neon(
        token: NeonToken,
        dst: &mut [f32],
        srcs: &[&[f32]],
        coeffs: &[f32],
        off: usize,
    ) {
        vfir_into_kernel(token, dst, srcs, coeffs, off)
    }

    #[archmage::arcane]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn vfir2_into_neon(
        token: NeonToken,
        d0: &mut [f32],
        d1: &mut [f32],
        srcs: &[&[f32]],
        coeffs0: &[f32],
        coeffs1: &[f32],
        off: usize,
    ) {
        vfir2_into_kernel(token, d0, d1, srcs, coeffs0, coeffs1, off)
    }

    #[archmage::arcane]
    pub(crate) fn vabs_diff_mul_lp2_sum_neon(
        token: NeonToken,
        t: &[f32],
        r: &[f32],
        s: &[f32],
    ) -> f32 {
        vabs_diff_mul_lp2_sum_kernel(token, t, r, s)
    }

    #[allow(clippy::too_many_arguments)]
    #[archmage::arcane]
    pub(crate) fn vabs_diff_pow_neon(
        token: NeonToken,
        out: &mut [f32],
        x: &[f32],
        y: &[f32],
        offset: f32,
        p: f32,
        offset_pow_p: f32,
    ) {
        vabs_diff_pow_kernel(token, out, x, y, offset, p, offset_pow_p)
    }

    #[archmage::arcane]
    pub(crate) fn vxcm_pool_clamp_4ch_sqsum_partial_neon(
        token: NeonToken,
        d: &[&[f32]; 4],
        t: &[&[f32]; 4],
        w: &[[f32; 4]; 4],
        d_max: f32,
    ) -> [f32; 4] {
        vxcm_pool_clamp_4ch_sqsum_partial_kernel(token, d, t, w, d_max)
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
    pub(crate) fn vabs_diff_into_wasm128(
        token: Wasm128Token,
        out: &mut [f32],
        x: &[f32],
        y: &[f32],
    ) {
        vabs_diff_kernel(token, out, x, y);
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
    pub(crate) fn vweber_band_into_wasm128(
        token: Wasm128Token,
        band: &mut [f32],
        log: &mut [f32],
        fine: &[f32],
        img_exp: &[f32],
        exp_l: &[f32],
    ) {
        vweber_band_kernel(token, band, log, fine, img_exp, exp_l);
    }

    #[archmage::arcane]
    pub(crate) fn vweber_band_nolog_into_wasm128(
        token: Wasm128Token,
        band: &mut [f32],
        fine: &[f32],
        img_exp: &[f32],
        exp_l: &[f32],
    ) {
        vweber_band_nolog_kernel(token, band, fine, img_exp, exp_l);
    }

    #[archmage::arcane]
    pub(crate) fn vscale2_into_wasm128(
        token: Wasm128Token,
        d1: &mut [f32],
        d2: &mut [f32],
        src: &[f32],
        a1: f32,
        a2: f32,
    ) {
        vscale2_kernel(token, d1, d2, src, a1, a2);
    }

    #[archmage::arcane]
    pub(crate) fn vaxpy2_into_wasm128(
        token: Wasm128Token,
        d1: &mut [f32],
        d2: &mut [f32],
        src: &[f32],
        a1: f32,
        a2: f32,
    ) {
        vaxpy2_kernel(token, d1, d2, src, a1, a2);
    }

    #[archmage::arcane]
    pub(crate) fn vmul2_scale2_pair_into_wasm128(
        token: Wasm128Token,
        o1: &mut [f32],
        o2: &mut [f32],
        x1: &[f32],
        x2: &[f32],
        w: &[f32],
        a: f32,
        b: f32,
    ) {
        vmul2_scale2_pair_kernel(token, o1, o2, x1, x2, w, a, b);
    }

    #[archmage::arcane]
    pub(crate) fn vfir_into_wasm128(
        token: Wasm128Token,
        dst: &mut [f32],
        srcs: &[&[f32]],
        coeffs: &[f32],
        off: usize,
    ) {
        vfir_into_kernel(token, dst, srcs, coeffs, off)
    }

    #[archmage::arcane]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn vfir2_into_wasm128(
        token: Wasm128Token,
        d0: &mut [f32],
        d1: &mut [f32],
        srcs: &[&[f32]],
        coeffs0: &[f32],
        coeffs1: &[f32],
        off: usize,
    ) {
        vfir2_into_kernel(token, d0, d1, srcs, coeffs0, coeffs1, off)
    }

    #[archmage::arcane]
    pub(crate) fn vabs_diff_mul_lp2_sum_wasm128(
        token: Wasm128Token,
        t: &[f32],
        r: &[f32],
        s: &[f32],
    ) -> f32 {
        vabs_diff_mul_lp2_sum_kernel(token, t, r, s)
    }

    #[allow(clippy::too_many_arguments)]
    #[archmage::arcane]
    pub(crate) fn vabs_diff_pow_wasm128(
        token: Wasm128Token,
        out: &mut [f32],
        x: &[f32],
        y: &[f32],
        offset: f32,
        p: f32,
        offset_pow_p: f32,
    ) {
        vabs_diff_pow_kernel(token, out, x, y, offset, p, offset_pow_p)
    }

    #[archmage::arcane]
    pub(crate) fn vxcm_pool_clamp_4ch_sqsum_partial_wasm128(
        token: Wasm128Token,
        d: &[&[f32]; 4],
        t: &[&[f32]; 4],
        w: &[[f32; 4]; 4],
        d_max: f32,
    ) -> [f32; 4] {
        vxcm_pool_clamp_4ch_sqsum_partial_kernel(token, d, t, w, d_max)
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

/// `out[i] = |x[i] - y[i]|`.
#[inline]
pub(crate) fn vabs_diff_into(out: &mut [f32], x: &[f32], y: &[f32]) {
    debug_assert_eq!(out.len(), x.len());
    debug_assert_eq!(out.len(), y.len());
    archmage::incant!(vabs_diff_into(out, x, y))
}

/// `out[i] = min(|x[i]|, |y[i]|)` — mutual-mask raw term.
#[inline]
pub(crate) fn vmin_abs_into(out: &mut [f32], x: &[f32], y: &[f32]) {
    debug_assert_eq!(out.len(), x.len());
    debug_assert_eq!(out.len(), y.len());
    archmage::incant!(vmin_abs_into(out, x, y))
}

/// `band[i] = clamp((fine[i] − img_exp[i]) / max(exp_l[i], 0.01), ±1000)`
/// and `log[i] = log10(max(exp_l[i], 0.01))` — the Weber-contrast
/// non-baseband fill, fused so `expanded_l` is loaded once.
pub(crate) fn vweber_band_into(
    band: &mut [f32],
    log: &mut [f32],
    fine: &[f32],
    img_exp: &[f32],
    exp_l: &[f32],
) {
    assert_eq!(band.len(), log.len());
    assert_eq!(band.len(), fine.len());
    assert_eq!(band.len(), img_exp.len());
    assert_eq!(band.len(), exp_l.len());
    archmage::incant!(vweber_band_into(band, log, fine, img_exp, exp_l))
}

/// `band[i] = clamp((fine[i] − img_exp[i]) / max(exp_l[i], 0.01), ±1000)`
/// — [`vweber_band_into`] without the `log_l_bkg` output, for the
/// pyramids whose `log_l_bkg` is never consumed.
pub(crate) fn vweber_band_nolog_into(
    band: &mut [f32],
    fine: &[f32],
    img_exp: &[f32],
    exp_l: &[f32],
) {
    assert_eq!(band.len(), fine.len());
    assert_eq!(band.len(), img_exp.len());
    assert_eq!(band.len(), exp_l.len());
    archmage::incant!(vweber_band_nolog_into(band, fine, img_exp, exp_l))
}

/// `d1[i] = a1·src[i]`, `d2[i] = a2·src[i]` — shared-source dual
/// scale (transient + sustained FIR over the same plane).
#[inline]
pub(crate) fn vscale2_into(d1: &mut [f32], d2: &mut [f32], src: &[f32], a1: f32, a2: f32) {
    debug_assert_eq!(d1.len(), src.len());
    debug_assert_eq!(d2.len(), src.len());
    archmage::incant!(vscale2_into(d1, d2, src, a1, a2))
}

/// `d1[i] += a1·src[i]`, `d2[i] += a2·src[i]` — shared-source dual
/// accumulate.
#[inline]
pub(crate) fn vaxpy2_into(d1: &mut [f32], d2: &mut [f32], src: &[f32], a1: f32, a2: f32) {
    debug_assert_eq!(d1.len(), src.len());
    debug_assert_eq!(d2.len(), src.len());
    archmage::incant!(vaxpy2_into(d1, d2, src, a1, a2))
}

/// `dst[i] = Σ_k coeffs[k]·srcs[k][off+i]` — fused N-tap temporal
/// FIR, one pass over all source planes. Band callers pass the
/// band's start index in `off`.
#[inline]
pub(crate) fn vfir_into(dst: &mut [f32], srcs: &[&[f32]], coeffs: &[f32], off: usize) {
    debug_assert_eq!(srcs.len(), coeffs.len());
    archmage::incant!(vfir_into(dst, srcs, coeffs, off))
}

/// `d0[i] = Σ_k coeffs0[k]·srcs[k][off+i]`,
/// `d1[i] = Σ_k coeffs1[k]·srcs[k][off+i]` — paired fused FIR.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn vfir2_into(
    d0: &mut [f32],
    d1: &mut [f32],
    srcs: &[&[f32]],
    coeffs0: &[f32],
    coeffs1: &[f32],
    off: usize,
) {
    debug_assert_eq!(d0.len(), d1.len());
    debug_assert_eq!(srcs.len(), coeffs0.len());
    debug_assert_eq!(srcs.len(), coeffs1.len());
    archmage::incant!(vfir2_into(d0, d1, srcs, coeffs0, coeffs1, off))
}

/// `o1[i] = ((x1[i]·a)·w[i])·b`, `o2[i] = ((x2[i]·a)·w[i])·b` — the
/// test/reference CSF-weight pair in one pass.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn vmul2_scale2_pair_into(
    o1: &mut [f32],
    o2: &mut [f32],
    x1: &[f32],
    x2: &[f32],
    w: &[f32],
    a: f32,
    b: f32,
) {
    debug_assert_eq!(o1.len(), x1.len());
    debug_assert_eq!(o2.len(), x2.len());
    debug_assert_eq!(o1.len(), w.len());
    archmage::incant!(vmul2_scale2_pair_into(o1, o2, x1, x2, w, a, b))
}

/// Raw `Σ u²` for `lp_norm_mean(|t−r|·s, 2.0)` — the video baseband
/// pooled difference, fused (no intermediate plane). Apply
/// [`lp2_finish`] to the folded total for the final norm.
#[inline]
pub(crate) fn vabs_diff_mul_lp2_sum(t: &[f32], r: &[f32], s: &[f32]) -> f32 {
    debug_assert_eq!(t.len(), r.len());
    debug_assert_eq!(t.len(), s.len());
    archmage::incant!(vabs_diff_mul_lp2_sum(t, r, s))
}

/// `out[i] = (|x[i] − y[i]| + offset)^p − offset_pow_p` — the
/// `vabs_diff` + `safe_pow_with_offset` pair fused into one pass.
#[allow(clippy::too_many_arguments)]
pub(crate) fn vabs_diff_pow_into(
    out: &mut [f32],
    x: &[f32],
    y: &[f32],
    offset: f32,
    p: f32,
    offset_pow_p: f32,
) {
    archmage::incant!(vabs_diff_pow(out, x, y, offset, p, offset_pow_p))
}

/// Per-channel raw `Σ u²` for the 4-channel cross-channel pool +
/// soft clamp + `lp_norm_mean_p2`, fully fused: `d` holds the
/// `safe_pow(|T−R|, p)` planes, `t` the `safe_pow(|M_mm|, q)` terms.
/// Apply [`xcm4_finish`] to the folded per-channel totals.
#[inline]
pub(crate) fn vxcm_pool_clamp_4ch_sqsum_partial(
    d: &[&[f32]; 4],
    t: &[&[f32]; 4],
    w: &[[f32; 4]; 4],
    d_max: f32,
) -> [f32; 4] {
    archmage::incant!(vxcm_pool_clamp_4ch_sqsum_partial(d, t, w, d_max))
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
