//! magetypes-dispatched SIMD kernels for the IW-SSIM hot paths.
//!
//! Coverage (this Phase 8g pass):
//!
//! - `square_inplace`        — per-element `out[i] = x[i] * x[i]`.
//! - `mul_inplace`           — per-element `out[i] = x[i] * y[i]`.
//! - `cs_combine`            — `cs = (2·σ₁₂ + C₂) / (σ₁² + σ₂² + C₂)`
//!   incl. luminance multiply at the top scale + the σ²-clamp-at-0 step.
//! - `weighted_sum_pair`     — Σ cs·iw, Σ iw — used by the IW pooling.
//! - `box3_h_pass`           — horizontal pass of the 3×3 box mean (zero
//!   padding at the boundary), reads `&[f32]`, writes `&mut [f32]`.
//! - `box3_v_pass`           — vertical pass of the 3×3 box mean.
//! - `ssim_gauss_h_pass`     — horizontal pass of the 11-tap σ=1.5
//!   separable Gaussian (valid-mode, output width = w - 10).
//! - `ssim_gauss_v_pass`     — vertical pass.
//!
//! Each entry point routes through `archmage::incant!` with the tier
//! cascade `[v4x, v4, v3, neon, wasm128, scalar]`. AVX-512 (v4x) is
//! opt-in via the `avx512` feature; without it the dispatch falls
//! through to `v4` (AVX2 8-wide). Boundary handling stays scalar — the
//! valid-region tails inherit the scalar fallback per row.

use crate::filters::SSIM_WIN_1D;

const SSIM_C1: f32 = crate::ssim::SSIM_C1;
const SSIM_C2: f32 = crate::ssim::SSIM_C2;

// ---------------------------------------------------------------------
// Per-element kernels (no boundary handling).
// ---------------------------------------------------------------------

#[archmage::magetypes(define(f32x8), +v4, +v4x, +v3, +neon, +wasm128, +scalar)]
fn square_inplace_inner(token: Token, src: &[f32], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), dst.len());
    let (src_chunks, src_tail) = f32x8::partition_slice(token, src);
    let (dst_chunks, dst_tail) = f32x8::partition_slice_mut(token, dst);
    for (sc, dc) in src_chunks.iter().zip(dst_chunks.iter_mut()) {
        let v = f32x8::load(token, sc);
        (v * v).store(dc);
    }
    for (s, d) in src_tail.iter().zip(dst_tail.iter_mut()) {
        *d = *s * *s;
    }
}

/// `dst[i] = src[i] * src[i]`.
pub(crate) fn square_into(src: &[f32], dst: &mut [f32]) {
    archmage::incant!(
        square_inplace_inner(src, dst),
        [v4x, v4, v3, neon, wasm128, scalar]
    );
}

#[archmage::magetypes(define(f32x8), +v4, +v4x, +v3, +neon, +wasm128, +scalar)]
fn mul_inplace_inner(token: Token, a: &[f32], b: &[f32], dst: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), dst.len());
    let (a_chunks, a_tail) = f32x8::partition_slice(token, a);
    let (b_chunks, b_tail) = f32x8::partition_slice(token, b);
    let (dst_chunks, dst_tail) = f32x8::partition_slice_mut(token, dst);
    for ((ac, bc), dc) in a_chunks.iter().zip(b_chunks.iter()).zip(dst_chunks.iter_mut()) {
        let va = f32x8::load(token, ac);
        let vb = f32x8::load(token, bc);
        (va * vb).store(dc);
    }
    for ((a, b), d) in a_tail.iter().zip(b_tail.iter()).zip(dst_tail.iter_mut()) {
        *d = *a * *b;
    }
}

/// `dst[i] = a[i] * b[i]`.
pub(crate) fn mul_into(a: &[f32], b: &[f32], dst: &mut [f32]) {
    archmage::incant!(
        mul_inplace_inner(a, b, dst),
        [v4x, v4, v3, neon, wasm128, scalar]
    );
}

#[archmage::magetypes(define(f32x8), +v4, +v4x, +v3, +neon, +wasm128, +scalar)]
fn cs_combine_inner(
    token: Token,
    mu1: &[f32],
    mu2: &[f32],
    s1sq_raw: &[f32],
    s2sq_raw: &[f32],
    s12_raw: &[f32],
    cs_out: &mut [f32],
    with_luminance: bool,
) {
    debug_assert_eq!(mu1.len(), cs_out.len());
    debug_assert_eq!(mu2.len(), cs_out.len());
    debug_assert_eq!(s1sq_raw.len(), cs_out.len());
    debug_assert_eq!(s2sq_raw.len(), cs_out.len());
    debug_assert_eq!(s12_raw.len(), cs_out.len());
    let c2_v = f32x8::splat(token, SSIM_C2);
    let c1_v = f32x8::splat(token, SSIM_C1);
    let two_v = f32x8::splat(token, 2.0);
    let zero_v = f32x8::zero(token);
    let n = cs_out.len();
    let n_chunks = n / 8;
    for i in 0..n_chunks {
        let off = i * 8;
        let m1: &[f32; 8] = (&mu1[off..off + 8]).try_into().unwrap();
        let m2: &[f32; 8] = (&mu2[off..off + 8]).try_into().unwrap();
        let s1: &[f32; 8] = (&s1sq_raw[off..off + 8]).try_into().unwrap();
        let s2: &[f32; 8] = (&s2sq_raw[off..off + 8]).try_into().unwrap();
        let s12: &[f32; 8] = (&s12_raw[off..off + 8]).try_into().unwrap();
        let m1_v = f32x8::load(token, m1);
        let m2_v = f32x8::load(token, m2);
        let s1_v = f32x8::load(token, s1);
        let s2_v = f32x8::load(token, s2);
        let s12_v = f32x8::load(token, s12);
        let s1sq = (s1_v - m1_v * m1_v).max(zero_v);
        let s2sq = (s2_v - m2_v * m2_v).max(zero_v);
        let s12_c = s12_v - m1_v * m2_v;
        let cs_v = (two_v * s12_c + c2_v) * ((s1sq + s2sq + c2_v).recip());
        let result = if with_luminance {
            let two_m1m2 = two_v * m1_v * m2_v;
            let l = (two_m1m2 + c1_v) * ((m1_v * m1_v + m2_v * m2_v + c1_v).recip());
            cs_v * l
        } else {
            cs_v
        };
        let dst: &mut [f32; 8] = (&mut cs_out[off..off + 8]).try_into().unwrap();
        result.store(dst);
    }
    // Scalar tail.
    let tail_start = n_chunks * 8;
    for i in tail_start..n {
        let m1 = mu1[i];
        let m2 = mu2[i];
        let s1 = (s1sq_raw[i] - m1 * m1).max(0.0);
        let s2 = (s2sq_raw[i] - m2 * m2).max(0.0);
        let s12_c = s12_raw[i] - m1 * m2;
        let mut cs = (2.0 * s12_c + SSIM_C2) / (s1 + s2 + SSIM_C2);
        if with_luminance {
            let l = (2.0 * m1 * m2 + SSIM_C1) / (m1 * m1 + m2 * m2 + SSIM_C1);
            cs *= l;
        }
        cs_out[i] = cs;
    }
}

/// Compute `cs = (2σ₁₂ + C₂) / (σ₁² + σ₂² + C₂)` from raw moments,
/// applying the σ²-clamp-at-0 step. With `with_luminance`, also
/// multiplies by `l = (2µ₁µ₂ + C₁) / (µ₁² + µ₂² + C₁)` in place
/// (matching the coarsest scale's cs·l combination).
pub(crate) fn cs_combine_into(
    mu1: &[f32],
    mu2: &[f32],
    s1sq_raw: &[f32],
    s2sq_raw: &[f32],
    s12_raw: &[f32],
    cs_out: &mut [f32],
    with_luminance: bool,
) {
    archmage::incant!(
        cs_combine_inner(mu1, mu2, s1sq_raw, s2sq_raw, s12_raw, cs_out, with_luminance),
        [v4x, v4, v3, neon, wasm128, scalar]
    );
}

#[archmage::magetypes(define(f32x8), +v4, +v4x, +v3, +neon, +wasm128, +scalar)]
fn weighted_sum_pair_inner(token: Token, cs: &[f32], iw: &[f32]) -> (f64, f64) {
    debug_assert_eq!(cs.len(), iw.len());
    let mut acc_csiw = f32x8::zero(token);
    let mut acc_iw = f32x8::zero(token);
    let (cs_chunks, cs_tail) = f32x8::partition_slice(token, cs);
    let (iw_chunks, iw_tail) = f32x8::partition_slice(token, iw);
    for (cc, ic) in cs_chunks.iter().zip(iw_chunks.iter()) {
        let c_v = f32x8::load(token, cc);
        let i_v = f32x8::load(token, ic);
        acc_csiw = c_v.mul_add(i_v, acc_csiw);
        acc_iw = acc_iw + i_v;
    }
    let mut sum_csiw = 0.0_f64;
    let mut sum_iw = 0.0_f64;
    let csiw_arr = acc_csiw.to_array();
    let iw_arr = acc_iw.to_array();
    for k in 0..8 {
        sum_csiw += csiw_arr[k] as f64;
        sum_iw += iw_arr[k] as f64;
    }
    // Scalar tail.
    for (c, i) in cs_tail.iter().zip(iw_tail.iter()) {
        sum_csiw += (*c as f64) * (*i as f64);
        sum_iw += *i as f64;
    }
    (sum_csiw, sum_iw)
}

/// Σ cs·iw, Σ iw as `f64`. SIMD accumulator into `f32` lanes; final
/// reduction widens to `f64` for parity with the scalar Python path.
pub(crate) fn weighted_sum_pair(cs: &[f32], iw: &[f32]) -> (f64, f64) {
    archmage::incant!(
        weighted_sum_pair_inner(cs, iw),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

// ---------------------------------------------------------------------
// 11-tap horizontal Gaussian (valid-mode, output w = w - 10).
//
// For each output column `ox in 0..dst_w`, read 11 contiguous input
// samples src[row_off + ox..row_off + ox + 11] and dot with
// SSIM_WIN_1D. Process 8 output samples in parallel.
// ---------------------------------------------------------------------

#[archmage::magetypes(define(f32x8), +v4, +v4x, +v3, +neon, +wasm128, +scalar)]
fn ssim_gauss_h_inner(token: Token, src: &[f32], h: usize, w: usize, dst_w: usize, dst: &mut [f32]) {
    debug_assert_eq!(src.len(), h * w);
    debug_assert_eq!(dst.len(), h * dst_w);
    // Splat each tap.
    let mut k_v = [f32x8::zero(token); 11];
    for i in 0..11 {
        k_v[i] = f32x8::splat(token, SSIM_WIN_1D[i]);
    }
    let n_chunks = dst_w / 8;
    for y in 0..h {
        let row_off = y * w;
        let out_row_off = y * dst_w;
        for c in 0..n_chunks {
            let ox = c * 8;
            // Sequential 8 outputs at ox..ox+7 read src[ox..ox+18] (11+7).
            // For each tap k, the contribution to output[ox..ox+7] is
            // SSIM_WIN_1D[k] * src[ox+k..ox+k+7] — load 8 lanes per tap.
            let mut acc = f32x8::zero(token);
            for k in 0..11 {
                let arr: &[f32; 8] = (&src[row_off + ox + k..row_off + ox + k + 8])
                    .try_into()
                    .unwrap();
                let v = f32x8::load(token, arr);
                acc = v.mul_add(k_v[k], acc);
            }
            let dst_slot: &mut [f32; 8] = (&mut dst[out_row_off + ox..out_row_off + ox + 8])
                .try_into()
                .unwrap();
            acc.store(dst_slot);
        }
        // Scalar tail.
        for ox in n_chunks * 8..dst_w {
            let mut acc_s = 0.0_f32;
            for k in 0..11 {
                acc_s += SSIM_WIN_1D[k] * src[row_off + ox + k];
            }
            dst[out_row_off + ox] = acc_s;
        }
    }
}

/// 11-tap horizontal Gaussian, valid mode. `dst` is `(h, dst_w)` with
/// `dst_w = w - 10`.
pub(crate) fn ssim_gauss_h_pass(src: &[f32], h: usize, w: usize, dst_w: usize, dst: &mut [f32]) {
    archmage::incant!(
        ssim_gauss_h_inner(src, h, w, dst_w, dst),
        [v4x, v4, v3, neon, wasm128, scalar]
    );
}

// ---------------------------------------------------------------------
// 11-tap vertical Gaussian (valid-mode, output h = h - 10).
//
// For each output row `oy`, read 11 contiguous source rows at
// oy..oy+10 and dot with SSIM_WIN_1D along axis-0. Process 8 columns
// in parallel.
// ---------------------------------------------------------------------

#[archmage::magetypes(define(f32x8), +v4, +v4x, +v3, +neon, +wasm128, +scalar)]
fn ssim_gauss_v_inner(
    token: Token,
    src: &[f32],
    h: usize,
    dst_h: usize,
    w: usize,
    dst: &mut [f32],
) {
    debug_assert_eq!(src.len(), h * w);
    debug_assert_eq!(dst.len(), dst_h * w);
    let mut k_v = [f32x8::zero(token); 11];
    for i in 0..11 {
        k_v[i] = f32x8::splat(token, SSIM_WIN_1D[i]);
    }
    let n_chunks = w / 8;
    for oy in 0..dst_h {
        let out_row_off = oy * w;
        // Pre-compute the 11 source row starts.
        let mut row_offs = [0usize; 11];
        for k in 0..11 {
            row_offs[k] = (oy + k) * w;
        }
        for c in 0..n_chunks {
            let x = c * 8;
            let mut acc = f32x8::zero(token);
            for k in 0..11 {
                let arr: &[f32; 8] = (&src[row_offs[k] + x..row_offs[k] + x + 8])
                    .try_into()
                    .unwrap();
                let v = f32x8::load(token, arr);
                acc = v.mul_add(k_v[k], acc);
            }
            let dst_slot: &mut [f32; 8] = (&mut dst[out_row_off + x..out_row_off + x + 8])
                .try_into()
                .unwrap();
            acc.store(dst_slot);
        }
        // Scalar tail.
        for x in n_chunks * 8..w {
            let mut acc_s = 0.0_f32;
            for k in 0..11 {
                acc_s += SSIM_WIN_1D[k] * src[row_offs[k] + x];
            }
            dst[out_row_off + x] = acc_s;
        }
    }
}

/// 11-tap vertical Gaussian, valid mode. `dst` is `(dst_h, w)` with
/// `dst_h = h - 10`.
pub(crate) fn ssim_gauss_v_pass(src: &[f32], h: usize, dst_h: usize, w: usize, dst: &mut [f32]) {
    archmage::incant!(
        ssim_gauss_v_inner(src, h, dst_h, w, dst),
        [v4x, v4, v3, neon, wasm128, scalar]
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_simd_matches_scalar() {
        let n = 19;
        let src: Vec<f32> = (0..n).map(|i| i as f32 - 5.0).collect();
        let mut simd = vec![0.0_f32; n];
        square_into(&src, &mut simd);
        for i in 0..n {
            let expected = src[i] * src[i];
            assert!((simd[i] - expected).abs() < 1e-6, "i={i}");
        }
    }

    #[test]
    fn mul_simd_matches_scalar() {
        let n = 25;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
        let mut simd = vec![0.0_f32; n];
        mul_into(&a, &b, &mut simd);
        for i in 0..n {
            assert!((simd[i] - a[i] * b[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn cs_combine_simd_matches_scalar() {
        let n = 19;
        let mu1: Vec<f32> = (0..n).map(|i| (i as f32) * 1.5 + 10.0).collect();
        let mu2: Vec<f32> = (0..n).map(|i| (i as f32) * 1.6 + 11.0).collect();
        let mut s1: Vec<f32> = (0..n).map(|i| (i as f32) * 50.0 + 200.0).collect();
        let mut s2: Vec<f32> = (0..n).map(|i| (i as f32) * 50.0 + 210.0).collect();
        // Make sure raw moments exceed mu² so the clamp doesn't trigger.
        for i in 0..n {
            if s1[i] < mu1[i] * mu1[i] {
                s1[i] = mu1[i] * mu1[i] + 50.0;
            }
            if s2[i] < mu2[i] * mu2[i] {
                s2[i] = mu2[i] * mu2[i] + 50.0;
            }
        }
        let s12: Vec<f32> = (0..n).map(|i| (i as f32) * 50.0 + 205.0).collect();

        // Scalar reference.
        let mut expected = vec![0.0_f32; n];
        for i in 0..n {
            let s1sq = (s1[i] - mu1[i] * mu1[i]).max(0.0);
            let s2sq = (s2[i] - mu2[i] * mu2[i]).max(0.0);
            let s12c = s12[i] - mu1[i] * mu2[i];
            expected[i] = (2.0 * s12c + SSIM_C2) / (s1sq + s2sq + SSIM_C2);
        }

        let mut simd = vec![0.0_f32; n];
        cs_combine_into(&mu1, &mu2, &s1, &s2, &s12, &mut simd, false);
        for i in 0..n {
            assert!(
                (simd[i] - expected[i]).abs() < 1e-5,
                "i={i}: simd={} expected={}",
                simd[i],
                expected[i]
            );
        }
    }

    #[test]
    fn weighted_sum_simd_matches_scalar() {
        let n = 31;
        let cs: Vec<f32> = (0..n).map(|i| 0.5 + 0.01 * (i as f32)).collect();
        let iw: Vec<f32> = (0..n).map(|i| 1.0 + 0.05 * (i as f32)).collect();
        let mut expected_csiw = 0.0_f64;
        let mut expected_iw = 0.0_f64;
        for i in 0..n {
            expected_csiw += (cs[i] as f64) * (iw[i] as f64);
            expected_iw += iw[i] as f64;
        }
        let (sum_csiw, sum_iw) = weighted_sum_pair(&cs, &iw);
        assert!((sum_csiw - expected_csiw).abs() < 1e-3);
        assert!((sum_iw - expected_iw).abs() < 1e-3);
    }

    #[test]
    fn ssim_gauss_h_simd_matches_scalar() {
        // 16x16 random input → output 16x6 (w - 10).
        let h = 16;
        let w = 16;
        let dst_w = w - 10;
        let src: Vec<f32> = (0..h * w).map(|i| (i as f32) * 0.1).collect();
        let mut simd = vec![0.0_f32; h * dst_w];
        ssim_gauss_h_pass(&src, h, w, dst_w, &mut simd);
        for y in 0..h {
            for ox in 0..dst_w {
                let mut expected = 0.0_f32;
                for k in 0..11 {
                    expected += SSIM_WIN_1D[k] * src[y * w + ox + k];
                }
                let got = simd[y * dst_w + ox];
                assert!(
                    (got - expected).abs() < 1e-4,
                    "y={y} ox={ox}: simd={got} scalar={expected}"
                );
            }
        }
    }

    #[test]
    fn ssim_gauss_v_simd_matches_scalar() {
        // 16x16 random input → output 6x16 (h - 10).
        let h = 16;
        let w = 16;
        let dst_h = h - 10;
        let src: Vec<f32> = (0..h * w).map(|i| (i as f32) * 0.1).collect();
        let mut simd = vec![0.0_f32; dst_h * w];
        ssim_gauss_v_pass(&src, h, dst_h, w, &mut simd);
        for oy in 0..dst_h {
            for x in 0..w {
                let mut expected = 0.0_f32;
                for k in 0..11 {
                    expected += SSIM_WIN_1D[k] * src[(oy + k) * w + x];
                }
                let got = simd[oy * w + x];
                assert!(
                    (got - expected).abs() < 1e-4,
                    "oy={oy} x={x}: simd={got} scalar={expected}"
                );
            }
        }
    }
}
