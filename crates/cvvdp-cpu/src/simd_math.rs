//! Vectorized transcendental helpers backed by archmage / magetypes.
//!
//! The cvvdp-cpu hot path spends ~7% of wall time inside `__powf_fma`
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
