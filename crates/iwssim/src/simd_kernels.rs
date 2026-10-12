//! magetypes-dispatched SIMD kernels for the IW-SSIM hot paths.
//!
//! Coverage:
//!
//! - `square_inplace`        — per-element `out[i] = x[i] * x[i]`.
//! - `mul_inplace`           — per-element `out[i] = x[i] * y[i]`.
//! - `cs_combine`            — `cs = (2·σ₁₂ + C₂) / (σ₁² + σ₂² + C₂)`
//!   incl. luminance multiply at the top scale + the σ²-clamp-at-0 step.
//! - `weighted_sum_pair`     — Σ cs·iw, Σ iw — used by the IW pooling.
//! - `ssim_gauss_h/v`        — 11-tap σ=1.5 separable Gaussian
//!   (valid-mode, output width = w - 10).
//! - `box_gain_rows`         — fused 3×3 box stats + gain correction:
//!   emits `g`/`vv` directly from the image slabs (replaces the
//!   unfused mean/σ²/xy plane chain).
//! - `quad_form_rows`        — per-pixel `ss = (Y·Cᵤ_inv) ⊙ Y / N` as a
//!   dense stencil on the image; no `nexp × N` Y matrix.
//! - `gram_rows_inner`       — `YᵀY` Gram accumulation in f64x4 lanes
//!   (two-rounding mul-then-add — bit-identical to the scalar order).
//! - `corr_dn_h/v`           — binom5 correlate + stride-2 decimate.
//! - `up_conv_h/v`           — zero-stuff + binom5 correlate + ×4 energy.
//! - `infow_map_into`        — `log2`/`pow` info-content map
//!   (`F32x8Convert` tiers only: v3, neon, wasm128, scalar).
//!
//! Most entry points route through `archmage::attuned!` with the tier
//! cascade `[v4x, v3, neon, wasm128, scalar]`; the f64x4 and
//! transcendental kernels use `[v3, neon, wasm128, scalar]` because
//! AVX-512 tokens lack `F64x4Backend`/`F32x8Convert`. AVX-512 (v4x) is
//! opt-in via the `avx512` feature. Boundary handling stays scalar —
//! the valid-region tails inherit the scalar fallback per row.
//!
//! With `parallel`, `*_rows`/reduction drivers band via `crate::par`
//! — boundaries are a pure function of plane length, so results are
//! identical at any thread count.

use crate::filters::SSIM_WIN_1D;

const SSIM_C1: f32 = crate::ssim::SSIM_C1;
const SSIM_C2: f32 = crate::ssim::SSIM_C2;

// ---------------------------------------------------------------------
// Per-element kernels (no boundary handling).
// ---------------------------------------------------------------------

#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
fn square_inplace_inner(token: Token, src: &[f32], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), dst.len());
    let (src_chunks, src_tail) = f32x8::partition_slice_t(token, src);
    let (dst_chunks, dst_tail) = f32x8::partition_slice_mut_t(token, dst);
    for (sc, dc) in src_chunks.iter().zip(dst_chunks.iter_mut()) {
        let v = f32x8::load_t(token, sc);
        (v * v).store(dc);
    }
    for (s, d) in src_tail.iter().zip(dst_tail.iter_mut()) {
        *d = *s * *s;
    }
}

/// `dst[i] = src[i] * src[i]`.
///
/// NOTE the missing `neon`: on aarch64 the vector tier is SLOWER here than the
/// scalar one (measured 101.5 us vs 88.5 us over 1024x1024 f32, 0.87x, CI
/// [-14.7%, -12.1%], reproduced across runs). NEON is baseline there, so the
/// scalar arm is autovectorized anyway, and for a one-input elementwise square
/// the `partition_slice` setup is not repaid.
///
/// The sibling `mul_into` does the same partitioning and measures 1.01x — the
/// difference is memory traffic. Squaring reads ONE buffer and so has bandwidth
/// headroom for the overhead to show; multiplying reads two and is closer to
/// the bandwidth ceiling, which hides it. So this is not a general indictment
/// of the f32x8 helpers, just of using them for the cheapest one-input op.
///
/// Verified output-neutral before removing the tier: neon and scalar produce
/// BIT-IDENTICAL results over 65,536 values (a plain multiply, so there is no
/// FMA-contraction difference to worry about). The other tiers keep their
/// vector paths — x86's ratio is not measured on this host.
pub fn square_into(src: &[f32], dst: &mut [f32]) {
    archmage::attuned!(
        square_inplace_inner(src, dst),
        [_v4x(cfg(avx512)), _v3, _wasm128, _scalar]
    );
}

#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
fn mul_inplace_inner(token: Token, a: &[f32], b: &[f32], dst: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), dst.len());
    let (a_chunks, a_tail) = f32x8::partition_slice_t(token, a);
    let (b_chunks, b_tail) = f32x8::partition_slice_t(token, b);
    let (dst_chunks, dst_tail) = f32x8::partition_slice_mut_t(token, dst);
    for ((ac, bc), dc) in a_chunks
        .iter()
        .zip(b_chunks.iter())
        .zip(dst_chunks.iter_mut())
    {
        let va = f32x8::load_t(token, ac);
        let vb = f32x8::load_t(token, bc);
        (va * vb).store(dc);
    }
    for ((a, b), d) in a_tail.iter().zip(b_tail.iter()).zip(dst_tail.iter_mut()) {
        *d = *a * *b;
    }
}

/// `dst[i] = a[i] * b[i]`.
pub fn mul_into(a: &[f32], b: &[f32], dst: &mut [f32]) {
    archmage::attuned!(
        mul_inplace_inner(a, b, dst),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
}

#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
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
    let c2_v = f32x8::splat_t(token, SSIM_C2);
    let c1_v = f32x8::splat_t(token, SSIM_C1);
    let two_v = f32x8::splat_t(token, 2.0);
    let zero_v = f32x8::zero_t(token);
    let n = cs_out.len();
    let n_chunks = n / 8;
    for i in 0..n_chunks {
        let off = i * 8;
        let m1: &[f32; 8] = (&mu1[off..off + 8]).try_into().unwrap();
        let m2: &[f32; 8] = (&mu2[off..off + 8]).try_into().unwrap();
        let s1: &[f32; 8] = (&s1sq_raw[off..off + 8]).try_into().unwrap();
        let s2: &[f32; 8] = (&s2sq_raw[off..off + 8]).try_into().unwrap();
        let s12: &[f32; 8] = (&s12_raw[off..off + 8]).try_into().unwrap();
        let m1_v = f32x8::load_t(token, m1);
        let m2_v = f32x8::load_t(token, m2);
        let s1_v = f32x8::load_t(token, s1);
        let s2_v = f32x8::load_t(token, s2);
        let s12_v = f32x8::load_t(token, s12);
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
pub fn cs_combine_into(
    mu1: &[f32],
    mu2: &[f32],
    s1sq_raw: &[f32],
    s2sq_raw: &[f32],
    s12_raw: &[f32],
    cs_out: &mut [f32],
    with_luminance: bool,
) {
    archmage::attuned!(
        cs_combine_inner(
            mu1,
            mu2,
            s1sq_raw,
            s2sq_raw,
            s12_raw,
            cs_out,
            with_luminance
        ),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
}

#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
fn weighted_sum_pair_inner(token: Token, cs: &[f32], iw: &[f32]) -> (f64, f64) {
    debug_assert_eq!(cs.len(), iw.len());
    let mut acc_csiw = f32x8::zero_t(token);
    let mut acc_iw = f32x8::zero_t(token);
    let (cs_chunks, cs_tail) = f32x8::partition_slice_t(token, cs);
    let (iw_chunks, iw_tail) = f32x8::partition_slice_t(token, iw);
    for (cc, ic) in cs_chunks.iter().zip(iw_chunks.iter()) {
        let c_v = f32x8::load_t(token, cc);
        let i_v = f32x8::load_t(token, ic);
        acc_csiw = c_v.mul_add(i_v, acc_csiw);
        acc_iw += i_v;
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
/// Banded above `par::PAR_MIN_SAMPLES`: per-band partials fold in
/// band order — deterministic at any thread count (re-associates vs
/// the flat accumulation, ulp-level drift only).
pub fn weighted_sum_pair(cs: &[f32], iw: &[f32]) -> (f64, f64) {
    let n = cs.len();
    let nb = crate::par::n_bands(n);
    if nb == 1 {
        return archmage::attuned!(
            weighted_sum_pair_inner(cs, iw),
            [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
        );
    }
    let sz = n.div_ceil(nb);
    let parts = crate::par::collect_bands(nb, |b| {
        let lo = b * sz;
        let hi = (lo + sz).min(n);
        if lo >= hi {
            return (0.0, 0.0);
        }
        archmage::attuned!(
            weighted_sum_pair_inner(&cs[lo..hi], &iw[lo..hi]),
            [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
        )
    });
    parts
        .iter()
        .fold((0.0_f64, 0.0_f64), |(a0, a1), &(p0, p1)| (a0 + p0, a1 + p1))
}

// ---------------------------------------------------------------------
// IW neighborhood quadratic form — `ss = (Y·Cᵤ_inv) ⊙ Y / N` as a
// dense stencil on the image. Output row processing, 8-wide over
// output columns: the `big_n` taps for outputs c..c+8 are contiguous
// shifted loads. Same (i,j) accumulation structure as the scalar
// `weights::quad_form_into`; only FMA contraction differs.
// ---------------------------------------------------------------------

#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
fn quad_form_row_inner(
    token: Token,
    img: &[f32],
    parent: &[f32],
    stride: usize,
    row_c: usize,
    col0: usize,
    ncols: usize,
    taps: &[(i32, i32)],
    cinv: &[f32],
    has_parent: bool,
    n_f: f32,
    out: &mut [f32],
) {
    let nb = taps.len();
    let big_n = nb + has_parent as usize;
    debug_assert!(big_n <= 16);
    debug_assert_eq!(cinv.len(), big_n * big_n);
    debug_assert_eq!(out.len(), ncols);
    // Hoist the Cᵤ_inv splats out of the column loop.
    let mut cinv_v = [f32x8::zero_t(token); 100];
    for k in 0..big_n * big_n {
        cinv_v[k] = f32x8::splat_t(token, cinv[k]);
    }
    let nfv = f32x8::splat_t(token, n_f);
    let n_chunks = ncols / 8;
    for c in 0..n_chunks {
        let x = c * 8;
        // Gather the big_n tap vectors for outputs col0+x .. col0+x+8.
        let mut yv = [f32x8::zero_t(token); 16];
        for (k, &(dy, dx)) in taps.iter().enumerate() {
            let off = ((row_c as i32 + dy) as usize) * stride + (col0 as i32 + dx) as usize + x;
            yv[k] = f32x8::load_t(token, (&img[off..off + 8]).try_into().unwrap());
        }
        if has_parent {
            let off = row_c * stride + col0 + x;
            yv[nb] = f32x8::load_t(token, (&parent[off..off + 8]).try_into().unwrap());
        }
        // acc = Σ_i y_i · (Σ_j Cinv[i,j]·y_j)
        let mut acc = f32x8::zero_t(token);
        for i in 0..big_n {
            let mut inner = f32x8::zero_t(token);
            for j in 0..big_n {
                inner = yv[j].mul_add(cinv_v[i * big_n + j], inner);
            }
            acc = yv[i].mul_add(inner, acc);
        }
        (acc / nfv).store((&mut out[x..x + 8]).try_into().unwrap());
    }
    // Scalar tail — same math element-wise.
    for x in n_chunks * 8..ncols {
        let col_c = col0 + x;
        let mut ys = [0.0_f32; 16];
        for (k, &(dy, dx)) in taps.iter().enumerate() {
            ys[k] = img[((row_c as i32 + dy) as usize) * stride + (col_c as i32 + dx) as usize];
        }
        if has_parent {
            ys[nb] = parent[row_c * stride + col_c];
        }
        let mut acc = 0.0_f32;
        for i in 0..big_n {
            let mut inner = 0.0_f32;
            for j in 0..big_n {
                inner += cinv[i * big_n + j] * ys[j];
            }
            acc += ys[i] * inner;
        }
        out[x] = acc / n_f;
    }
}

/// Row-wise driver for [`quad_form_row_inner`]: computes the
/// `nrows × ncols` quadratic-form map on the neighborhood stencil of
/// `img` (+ optional parent column), matching
/// `weights::quad_form_into` semantics.
///
/// `parent` is `None` for the coarsest scale — pass `&[]`.
#[allow(clippy::too_many_arguments)]
pub fn quad_form_rows(
    img: &[f32],
    parent: Option<&[f32]>,
    stride: usize,
    row0: usize,
    col0: usize,
    nrows: usize,
    ncols: usize,
    taps: &[(i32, i32)],
    cinv: &[f32],
    out: &mut [f32],
) {
    let nb = taps.len();
    let big_n = nb + parent.is_some() as usize;
    let n_f = big_n as f32;
    let parent_s = parent.unwrap_or(&[]);
    let has_parent = parent.is_some();
    debug_assert_eq!(out.len(), nrows * ncols);
    crate::par::map_rows(out, ncols, |y0, band| {
        let band_rows = band.len() / ncols;
        for r in 0..band_rows {
            let out_row = &mut band[r * ncols..(r + 1) * ncols];
            archmage::attuned!(
                quad_form_row_inner(
                    img,
                    parent_s,
                    stride,
                    row0 + y0 + r,
                    col0,
                    ncols,
                    taps,
                    cinv,
                    has_parent,
                    n_f,
                    out_row
                ),
                [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
            );
        }
    });
}

// ---------------------------------------------------------------------
// 11-tap horizontal Gaussian (valid-mode, output w = w - 10).
//
// For each output column `ox in 0..dst_w`, read 11 contiguous input
// samples src[row_off + ox..row_off + ox + 11] and dot with
// SSIM_WIN_1D. Process 8 output samples in parallel.
// ---------------------------------------------------------------------

#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
fn ssim_gauss_h_inner(
    token: Token,
    src: &[f32],
    h: usize,
    w: usize,
    dst_w: usize,
    dst: &mut [f32],
) {
    debug_assert_eq!(src.len(), h * w);
    debug_assert_eq!(dst.len(), h * dst_w);
    // Splat each tap.
    let mut k_v = [f32x8::zero_t(token); 11];
    for i in 0..11 {
        k_v[i] = f32x8::splat_t(token, SSIM_WIN_1D[i]);
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
            let mut acc = f32x8::zero_t(token);
            for k in 0..11 {
                let arr: &[f32; 8] = (&src[row_off + ox + k..row_off + ox + k + 8])
                    .try_into()
                    .unwrap();
                let v = f32x8::load_t(token, arr);
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
pub fn ssim_gauss_h_pass(src: &[f32], h: usize, w: usize, dst_w: usize, dst: &mut [f32]) {
    archmage::attuned!(
        ssim_gauss_h_inner(src, h, w, dst_w, dst),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
}

// ---------------------------------------------------------------------
// 11-tap vertical Gaussian (valid-mode, output h = h - 10).
//
// For each output row `oy`, read 11 contiguous source rows at
// oy..oy+10 and dot with SSIM_WIN_1D along axis-0. Process 8 columns
// in parallel.
// ---------------------------------------------------------------------

#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
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
    let mut k_v = [f32x8::zero_t(token); 11];
    for i in 0..11 {
        k_v[i] = f32x8::splat_t(token, SSIM_WIN_1D[i]);
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
            let mut acc = f32x8::zero_t(token);
            for k in 0..11 {
                let arr: &[f32; 8] = (&src[row_offs[k] + x..row_offs[k] + x + 8])
                    .try_into()
                    .unwrap();
                let v = f32x8::load_t(token, arr);
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
pub fn ssim_gauss_v_pass(src: &[f32], h: usize, dst_h: usize, w: usize, dst: &mut [f32]) {
    archmage::attuned!(
        ssim_gauss_v_inner(src, h, dst_h, w, dst),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
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

// ---------------------------------------------------------------------
// Fused 3×3 box statistics + gain correction — computes the IW gain
// fields `g = cov/(ss_x+tol)`, `vv = ss_y − g·cov` (with the reference's
// tol-conditional zeroing) in ONE pass over the (x, y) slabs. Replaces
// the unfused path's five `box3_same` passes + xx/yy/xy temporaries +
// separate gain loop. Per-lane add order matches the scalar (dy, dx)
// tap order; the product accumulators use `mul_add`.
// ---------------------------------------------------------------------

/// Scalar body for one output pixel — edge columns/rows and the
/// fallback tier. `sy_lo..sy_hi` is the clamped source-row range.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn box_gain_pixel(
    x: &[f32],
    y: &[f32],
    stride: usize,
    w: usize,
    sy_lo: usize,
    sy_hi: usize,
    cx: usize,
    tol: f32,
) -> (f32, f32) {
    let x_lo = cx.saturating_sub(1);
    let x_hi = (cx + 1).min(w - 1);
    let mut mx = 0.0_f32;
    let mut my = 0.0_f32;
    let mut sxx = 0.0_f32;
    let mut syy = 0.0_f32;
    let mut sxy = 0.0_f32;
    for sy in sy_lo..=sy_hi {
        let row = sy * stride;
        for sx in x_lo..=x_hi {
            let xv = x[row + sx];
            let yv = y[row + sx];
            mx += xv;
            my += yv;
            sxx += xv * xv;
            syy += yv * yv;
            sxy += xv * yv;
        }
    }
    let inv9 = 1.0_f32 / 9.0;
    let mx = mx * inv9;
    let my = my * inv9;
    let cov = sxy * inv9 - mx * my;
    let ssx = (sxx * inv9 - mx * mx).max(0.0);
    let ssy = (syy * inv9 - my * my).max(0.0);
    let mut g = cov / (ssx + tol);
    let mut vv = ssy - g * cov;
    if ssx < tol {
        g = 0.0;
        vv = ssy;
    }
    if ssy < tol {
        g = 0.0;
        vv = 0.0;
    }
    (g, vv)
}

#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
#[allow(clippy::too_many_arguments)]
fn box_gain_row_inner(
    token: Token,
    x: &[f32],
    y: &[f32],
    stride: usize,
    slab_h: usize,
    r: usize,
    w: usize,
    tol: f32,
    g_row: &mut [f32],
    vv_row: &mut [f32],
) {
    debug_assert_eq!(g_row.len(), w);
    debug_assert_eq!(vv_row.len(), w);
    let sy_lo = r.saturating_sub(1);
    let sy_hi = (r + 1).min(slab_h - 1);
    if w < 10 {
        for cx in 0..w {
            let (g, vv) = box_gain_pixel(x, y, stride, w, sy_lo, sy_hi, cx, tol);
            g_row[cx] = g;
            vv_row[cx] = vv;
        }
        return;
    }
    let inv9 = f32x8::splat_t(token, 1.0 / 9.0);
    let tol_v = f32x8::splat_t(token, tol);
    let zero = f32x8::zero_t(token);
    // Interior vector cols: c..c+8 ⊆ [1, w-1) — all nine taps in-bounds.
    let vec_end = w - 1; // exclusive bound on c+8
    let mut c = 1usize;
    while c + 8 <= vec_end {
        let mut sx = f32x8::zero_t(token);
        let mut sy = f32x8::zero_t(token);
        let mut sxx = f32x8::zero_t(token);
        let mut syy = f32x8::zero_t(token);
        let mut sxy = f32x8::zero_t(token);
        for srow in sy_lo..=sy_hi {
            let row = srow * stride;
            for dx in [-1isize, 0, 1] {
                let off = (row + c).wrapping_add_signed(dx);
                let xv = f32x8::load_t(token, (&x[off..off + 8]).try_into().unwrap());
                let yv = f32x8::load_t(token, (&y[off..off + 8]).try_into().unwrap());
                sx += xv;
                sy += yv;
                sxx = xv.mul_add(xv, sxx);
                syy = yv.mul_add(yv, syy);
                sxy = xv.mul_add(yv, sxy);
            }
        }
        let mx = sx * inv9;
        let my = sy * inv9;
        let cov = sxy * inv9 - mx * my;
        let ssx = (sxx * inv9 - mx * mx).max(zero);
        let ssy = (syy * inv9 - my * my).max(zero);
        let mut g = cov / (ssx + tol_v);
        let mut vv = ssy - g * cov;
        let m_x = ssx.simd_lt(tol_v);
        g = f32x8::blend(m_x, zero, g);
        vv = f32x8::blend(m_x, ssy, vv);
        let m_y = ssy.simd_lt(tol_v);
        g = f32x8::blend(m_y, zero, g);
        vv = f32x8::blend(m_y, zero, vv);
        g.store((&mut g_row[c..c + 8]).try_into().unwrap());
        vv.store((&mut vv_row[c..c + 8]).try_into().unwrap());
        c += 8;
    }
    // Left edge, right edge, and vector tail — scalar.
    let mut scalar_cols = alloc::vec::Vec::with_capacity(9);
    scalar_cols.push(0usize);
    for cx in c..w {
        scalar_cols.push(cx);
    }
    for cx in scalar_cols {
        let (g, vv) = box_gain_pixel(x, y, stride, w, sy_lo, sy_hi, cx, tol);
        g_row[cx] = g;
        vv_row[cx] = vv;
    }
}

/// `g`/`vv` gain fields for output rows `[row0, row0+nrows)` of the
/// `slab_h × w` slabs `x` (reference) and `y` (distorted). Outputs are
/// `nrows × w` row-major. Box sums use zero-padding at slab edges —
/// callers supply slabs that already contain any needed halo rows.
pub fn box_gain_rows(
    x: &[f32],
    y: &[f32],
    slab_h: usize,
    w: usize,
    row0: usize,
    nrows: usize,
    g: &mut [f32],
    vv: &mut [f32],
) {
    debug_assert_eq!(x.len(), slab_h * w);
    debug_assert_eq!(y.len(), slab_h * w);
    debug_assert_eq!(g.len(), nrows * w);
    debug_assert_eq!(vv.len(), nrows * w);
    let tol = crate::weights::TOL;
    crate::par::map_rows2(g, vv, w, |y0, gb, vb| {
        let band_rows = gb.len() / w;
        for r in 0..band_rows {
            let out = r * w;
            archmage::attuned!(
                box_gain_row_inner(
                    x,
                    y,
                    w,
                    slab_h,
                    row0 + y0 + r,
                    w,
                    tol,
                    &mut gb[out..out + w],
                    &mut vb[out..out + w]
                ),
                [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
            );
        }
    });
}

// ---------------------------------------------------------------------
// Laplacian-pyramid convolutions (`binom5`, `reflect1` borders).
//
// - `corr_dn_v` / `up_conv_v`: taps are whole rows — vectorized over
//   columns, contiguous loads.
// - `corr_dn_h`: per-row full-width correlation into a scratch row
//   (interior SIMD, edge cols scalar `reflect1`), then stride-2
//   decimate. The odd positions are extra work but every load stays
//   contiguous — cheaper than per-output reflect branches.
// - `up_conv_h`: parity split — even outputs `2m` use taps {0,2,4} on
//   `src[m-1..m+2]`, odd outputs `2m+1` use taps {1,3} on `src[m..m+2]`.
//   8 m's per iter → two vectors → scalar-interleaved stores.
// ---------------------------------------------------------------------

use crate::filters::{BINOM5, BINOM5_LEN, BINOM5_RADIUS};
use crate::pyramid::reflect1;

/// Scalar `corr_dn` tap sum with `reflect1` borders — edge columns and
/// the scalar-tier fallback.
#[inline(always)]
fn corr5_reflect(src_row: &[f32], in_w: usize, xc: usize) -> f32 {
    let mut acc = 0.0_f32;
    for k in 0..BINOM5_LEN {
        let xs = reflect1(xc as i32 - BINOM5_RADIUS + k as i32, in_w as i32) as usize;
        acc += BINOM5[k] * src_row[xs];
    }
    acc
}

#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
#[allow(clippy::too_many_arguments)]
fn corr_dn_h_inner(
    token: Token,
    src: &[f32],
    in_w: usize,
    out_w: usize,
    y0: usize,
    corr_row: &mut [f32],
    dst: &mut [f32],
) {
    debug_assert_eq!(corr_row.len(), in_w);
    let k: [f32x8; 5] = core::array::from_fn(|i| f32x8::splat_t(token, BINOM5[i]));
    let band_rows = dst.len() / out_w;
    for i in 0..band_rows {
        let y = y0 + i;
        let row = &src[y * in_w..(y + 1) * in_w];
        // Full-width correlation — interior all-taps-in-bounds SIMD.
        let vec_hi = in_w.saturating_sub(BINOM5_RADIUS as usize);
        let mut c = BINOM5_RADIUS as usize;
        while c + 8 <= vec_hi {
            let mut acc = f32x8::zero_t(token);
            for t in 0..BINOM5_LEN {
                let off = c - BINOM5_RADIUS as usize + t;
                let v = f32x8::load_t(token, (&row[off..off + 8]).try_into().unwrap());
                acc = v.mul_add(k[t], acc);
            }
            acc.store((&mut corr_row[c..c + 8]).try_into().unwrap());
            c += 8;
        }
        // Edge + tail columns: scalar reflect1.
        for xc in (0..BINOM5_RADIUS as usize).chain(c..in_w) {
            corr_row[xc] = corr5_reflect(row, in_w, xc);
        }
        // Decimate.
        let dst_row = &mut dst[i * out_w..(i + 1) * out_w];
        for ox in 0..out_w {
            dst_row[ox] = corr_row[2 * ox];
        }
    }
}

/// `corr_dn` horizontal + decimate-by-2 (see `pyramid::corr_dn_horizontal`).
pub fn corr_dn_h(src: &[f32], h: usize, in_w: usize, out_w: usize, dst: &mut [f32]) {
    debug_assert_eq!(src.len(), h * in_w);
    debug_assert_eq!(dst.len(), h * out_w);
    crate::par::map_rows(dst, out_w, |y0, band| {
        // Per-band correlation scratch — one alloc per band, reused
        // across the band's rows.
        let mut corr_row = alloc::vec![0.0_f32; in_w];
        archmage::attuned!(
            corr_dn_h_inner(src, in_w, out_w, y0, &mut corr_row, band),
            [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
        );
    });
}

#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
fn corr_dn_v_inner(token: Token, src: &[f32], in_h: usize, w: usize, y0: usize, dst: &mut [f32]) {
    let k: [f32x8; 5] = core::array::from_fn(|i| f32x8::splat_t(token, BINOM5[i]));
    let in_h_i = in_h as i32;
    let band_rows = dst.len() / w;
    for i in 0..band_rows {
        let oy = y0 + i;
        let cy = 2 * oy as i32;
        let ys: [usize; 5] =
            core::array::from_fn(|i| reflect1(cy - BINOM5_RADIUS + i as i32, in_h_i) as usize);
        let dst_row = &mut dst[i * w..(i + 1) * w];
        let n_chunks = w / 8;
        for c in 0..n_chunks {
            let x = c * 8;
            let mut acc = f32x8::zero_t(token);
            for t in 0..BINOM5_LEN {
                let off = ys[t] * w + x;
                let v = f32x8::load_t(token, (&src[off..off + 8]).try_into().unwrap());
                acc = v.mul_add(k[t], acc);
            }
            acc.store((&mut dst_row[x..x + 8]).try_into().unwrap());
        }
        for x in n_chunks * 8..w {
            let mut acc = 0.0_f32;
            for t in 0..BINOM5_LEN {
                acc += BINOM5[t] * src[ys[t] * w + x];
            }
            dst_row[x] = acc;
        }
    }
}

/// `corr_dn` vertical + decimate-by-2 (see `pyramid::corr_dn_vertical`).
pub fn corr_dn_v(src: &[f32], in_h: usize, w: usize, out_h: usize, dst: &mut [f32]) {
    debug_assert_eq!(src.len(), in_h * w);
    debug_assert_eq!(dst.len(), out_h * w);
    crate::par::map_rows(dst, w, |y0, band| {
        archmage::attuned!(
            corr_dn_v_inner(src, in_h, w, y0, band),
            [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
        );
    });
}

#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
fn up_conv_v_inner(token: Token, src: &[f32], in_h: usize, w: usize, y0: usize, dst: &mut [f32]) {
    let k0 = f32x8::splat_t(token, BINOM5[0]);
    let k1 = f32x8::splat_t(token, BINOM5[1]);
    let k2 = f32x8::splat_t(token, BINOM5[2]);
    let k3 = f32x8::splat_t(token, BINOM5[3]);
    let k4 = f32x8::splat_t(token, BINOM5[4]);
    let in_h_i = in_h as i32;
    // Interior output rows whose every expanded-axis tap lands
    // in-bounds: oy ∈ [2, 2·in_h − 3]. Outside: scalar reflect path.
    let vec_hi = (2 * in_h).saturating_sub(3);
    let n_chunks = w / 8;
    let band_rows = dst.len() / w;
    for i in 0..band_rows {
        let oy = y0 + i;
        let dst_row = &mut dst[i * w..(i + 1) * w];
        if oy >= 2 && oy <= vec_hi {
            // Parity split (see `up_conv_h` doc): even oy = 2m uses
            // taps {0,2,4} on rows {m-1, m, m+1}; odd oy = 2m+1 uses
            // taps {1,3} on rows {m, m+1}.
            let m = oy / 2;
            // Accumulate in ascending-tap order to match the scalar's
            // add sequence (modulo FMA contraction).
            let (ra, rb, rc, ka, kb, kc) = if oy.is_multiple_of(2) {
                (m - 1, m, m + 1, k0, k2, k4)
            } else {
                (m, m + 1, m + 1, k1, k3, k3) // third term unused (k3 dup)
            };
            for c in 0..n_chunks {
                let x = c * 8;
                let a = f32x8::load_t(
                    token,
                    (&src[ra * w + x..ra * w + x + 8]).try_into().unwrap(),
                );
                let b = f32x8::load_t(
                    token,
                    (&src[rb * w + x..rb * w + x + 8]).try_into().unwrap(),
                );
                let mut acc = a.mul_add(ka, f32x8::zero_t(token));
                acc = b.mul_add(kb, acc);
                if oy.is_multiple_of(2) {
                    let cv = f32x8::load_t(
                        token,
                        (&src[rc * w + x..rc * w + x + 8]).try_into().unwrap(),
                    );
                    acc = cv.mul_add(kc, acc);
                }
                acc.store((&mut dst_row[x..x + 8]).try_into().unwrap());
            }
            for x in n_chunks * 8..w {
                let mut acc = BINOM5[if oy.is_multiple_of(2) { 0 } else { 1 }] * src[ra * w + x]
                    + BINOM5[if oy.is_multiple_of(2) { 2 } else { 3 }] * src[rb * w + x];
                if oy.is_multiple_of(2) {
                    acc += BINOM5[4] * src[rc * w + x];
                }
                dst_row[x] = acc;
            }
        } else {
            // Boundary rows — scalar `reflect_expanded` semantics.
            let p0 = oy as i32 - BINOM5_RADIUS;
            let two_n = 2 * in_h_i;
            for x in 0..w {
                let mut acc = 0.0_f32;
                for t in 0..BINOM5_LEN {
                    let mut q = p0 + t as i32;
                    if q < 0 {
                        q = -q;
                    }
                    if q >= two_n {
                        q = 2 * (two_n - 1) - q;
                    }
                    if q < 0 {
                        q = -q;
                    }
                    if q >= two_n {
                        q = 2 * (two_n - 1) - q;
                    }
                    if (q & 1) == 0 {
                        acc += BINOM5[t] * src[(q / 2) as usize * w + x];
                    }
                }
                dst_row[x] = acc;
            }
        }
    }
}

/// `up_conv` vertical — zero-insert ×2 then correlate (see
/// `pyramid::up_conv_vertical`).
pub fn up_conv_v(src: &[f32], in_h: usize, w: usize, out_h: usize, dst: &mut [f32]) {
    debug_assert_eq!(src.len(), in_h * w);
    debug_assert_eq!(dst.len(), out_h * w);
    crate::par::map_rows(dst, w, |y0, band| {
        archmage::attuned!(
            up_conv_v_inner(src, in_h, w, y0, band),
            [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
        );
    });
}

/// Scalar `up_conv` horizontal for one output — boundary columns and
/// the fallback tier. Mirrors `pyramid::up_conv_horizontal`'s tap loop.
#[inline(always)]
fn up_conv_h_pixel(src_row: &[f32], in_w: usize, ox: usize) -> f32 {
    let two_n = 2 * in_w as i32;
    let mut acc = 0.0_f32;
    let p0 = ox as i32 - BINOM5_RADIUS;
    for t in 0..BINOM5_LEN {
        let mut q = p0 + t as i32;
        if q < 0 {
            q = -q;
        }
        if q >= two_n {
            q = 2 * (two_n - 1) - q;
        }
        if q < 0 {
            q = -q;
        }
        if q >= two_n {
            q = 2 * (two_n - 1) - q;
        }
        if (q & 1) == 0 {
            acc += BINOM5[t] * src_row[(q / 2) as usize];
        }
    }
    acc
}

#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
fn up_conv_h_inner(
    token: Token,
    src: &[f32],
    in_w: usize,
    out_w: usize,
    y0: usize,
    dst: &mut [f32],
) {
    let k0 = f32x8::splat_t(token, BINOM5[0]);
    let k1 = f32x8::splat_t(token, BINOM5[1]);
    let k2 = f32x8::splat_t(token, BINOM5[2]);
    let k3 = f32x8::splat_t(token, BINOM5[3]);
    let k4 = f32x8::splat_t(token, BINOM5[4]);
    // Interior m range: even out 2m needs src[m-1], src[m+1] →
    // m ∈ [1, in_w-2]. Vector chunks need m+9 ≤ in_w → m ∈ [1, in_w-9].
    let vec_hi = in_w.saturating_sub(9);
    let band_rows = dst.len() / out_w;
    for i in 0..band_rows {
        let y = y0 + i;
        let row = &src[y * in_w..(y + 1) * in_w];
        let dst_row = &mut dst[i * out_w..(i + 1) * out_w];
        let mut m = 1usize;
        while m + 8 <= vec_hi {
            let a = f32x8::load_t(token, (&row[m - 1..m + 7]).try_into().unwrap());
            let b = f32x8::load_t(token, (&row[m..m + 8]).try_into().unwrap());
            let c = f32x8::load_t(token, (&row[m + 1..m + 9]).try_into().unwrap());
            // Ascending-tap accumulation order (modulo FMA).
            let mut ev = a.mul_add(k0, f32x8::zero_t(token));
            ev = b.mul_add(k2, ev);
            ev = c.mul_add(k4, ev);
            let mut ov = b.mul_add(k1, f32x8::zero_t(token));
            ov = c.mul_add(k3, ov);
            let mut ea = [0.0_f32; 8];
            let mut oa = [0.0_f32; 8];
            ev.store(&mut ea);
            ov.store(&mut oa);
            for j in 0..8 {
                let ox = 2 * (m + j);
                if ox < out_w {
                    dst_row[ox] = ea[j];
                }
                if ox + 1 < out_w {
                    dst_row[ox + 1] = oa[j];
                }
            }
            m += 8;
        }
        // Boundary + tail outputs — scalar reflect semantics.
        for ox in (0..2).chain(2 * m..out_w) {
            if ox < out_w {
                dst_row[ox] = up_conv_h_pixel(row, in_w, ox);
            }
        }
    }
}

/// `up_conv` horizontal — zero-insert ×2 then correlate (see
/// `pyramid::up_conv_horizontal`).
pub fn up_conv_h(src: &[f32], h: usize, in_w: usize, out_w: usize, dst: &mut [f32]) {
    debug_assert_eq!(src.len(), h * in_w);
    debug_assert_eq!(dst.len(), h * out_w);
    crate::par::map_rows(dst, out_w, |y0, band| {
        archmage::attuned!(
            up_conv_h_inner(src, in_w, out_w, y0, band),
            [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
        );
    });
}

// ---------------------------------------------------------------------
// `infow` evaluation — `Σ_k log2(1 + ((vv + (1+g²)·σ²)·ss·λ_k + σ²·vv)
// / σ⁴)` per pixel, clamped at `tol`. Vectorized 8-wide over pixels;
// `log2_midp` (≈3 ULP) replaces the scalar `f32::log2`, so this path is
// tolerance-exact rather than bit-identical — well inside the crate's
// 1e-4 parity band.
//
// `F32x8Convert` kernels can't live inside `magetypes(define)` (the
// generated bound doesn't include it), so this follows cvvdp's manual
// generic-kernel + per-tier `arcane` wrapper pattern.
// ---------------------------------------------------------------------

#[inline]
fn infow_kernel<T: magetypes::simd::backends::F32x8Convert>(
    token: T,
    g: &[f32],
    vv: &[f32],
    ss: &[f32],
    lambdas: &[f32],
    s2: f32,
    inv_s4: f32,
    tol: f32,
    out: &mut [f32],
) {
    debug_assert_eq!(vv.len(), g.len());
    debug_assert_eq!(ss.len(), g.len());
    debug_assert_eq!(out.len(), g.len());
    type F32x8<T> = magetypes::simd::generic::f32x8<T>;
    let s2v = F32x8::<T>::splat_t(token, s2);
    let inv_s4v = F32x8::<T>::splat_t(token, inv_s4);
    let tolv = F32x8::<T>::splat_t(token, tol);
    let one = F32x8::<T>::splat_t(token, 1.0);
    let zero = F32x8::<T>::zero_t(token);
    let mut lamv = [zero; 16];
    for (i, &l) in lambdas.iter().enumerate() {
        lamv[i] = F32x8::<T>::splat_t(token, l);
    }
    let n_chunks = g.len() / 8;
    for c in 0..n_chunks {
        let i = c * 8;
        let gv = F32x8::<T>::load_t(token, (&g[i..i + 8]).try_into().unwrap());
        let vvv = F32x8::<T>::load_t(token, (&vv[i..i + 8]).try_into().unwrap());
        let ssv = F32x8::<T>::load_t(token, (&ss[i..i + 8]).try_into().unwrap());
        let one_plus_g2 = gv.mul_add(gv, one);
        let common_num = (vvv + one_plus_g2 * s2v) * ssv;
        let sn2_vv = s2v * vvv;
        let mut acc = zero;
        for lv in lamv.iter().take(lambdas.len()) {
            let arg = common_num.mul_add(*lv, sn2_vv) * inv_s4v;
            acc += (one + arg).log2_midp();
        }
        // `acc >= tol → acc else 0` — matches `if acc >= TOL`.
        let keep = acc.simd_ge(tolv);
        let res = F32x8::<T>::blend(keep, acc, zero);
        res.store((&mut out[i..i + 8]).try_into().unwrap());
    }
    for i in n_chunks * 8..g.len() {
        let g_i = g[i];
        let vv_i = vv[i];
        let ss_i = ss[i];
        let mut acc = 0.0_f32;
        let one_plus_g2 = 1.0 + g_i * g_i;
        let common_num = (vv_i + one_plus_g2 * s2) * ss_i;
        let sn2_vv = s2 * vv_i;
        for &lam in lambdas {
            let arg = (common_num * lam + sn2_vv) * inv_s4;
            acc += (1.0 + arg).log2();
        }
        out[i] = if acc >= tol { acc } else { 0.0 };
    }
}

/// Per-pixel `infow` map (see `weights::compute_infow` for the
/// reference formula). `out.len() == g.len() == vv.len() == ss.len()`.
#[allow(clippy::too_many_arguments)]
pub fn infow_map(
    g: &[f32],
    vv: &[f32],
    ss: &[f32],
    lambdas: &[f32],
    sigma_nsq: f32,
    tol: f32,
    out: &mut [f32],
) {
    let s2 = sigma_nsq;
    let s4 = s2 * s2;
    let inv_s4 = 1.0 / s4;
    crate::par::map1_3(out, g, vv, ss, |d, a, b, c| {
        archmage::attuned!(
            infow_map_into(a, b, c, lambdas, s2, inv_s4, tol, d),
            [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
        );
    });
}

pub(crate) fn infow_map_into_scalar_t(
    token: archmage::ScalarToken,
    g: &[f32],
    vv: &[f32],
    ss: &[f32],
    lambdas: &[f32],
    s2: f32,
    inv_s4: f32,
    tol: f32,
    out: &mut [f32],
) {
    infow_kernel(token, g, vv, ss, lambdas, s2, inv_s4, tol, out);
}

#[cfg(target_arch = "x86_64")]
use infow_x64::*;

#[cfg(target_arch = "x86_64")]
mod infow_x64 {
    use super::*;

    #[archmage::attune]
    pub(crate) fn infow_map_into_v3_t(
        token: archmage::X64V3Token,
        g: &[f32],
        vv: &[f32],
        ss: &[f32],
        lambdas: &[f32],
        s2: f32,
        inv_s4: f32,
        tol: f32,
        out: &mut [f32],
    ) {
        infow_kernel(token, g, vv, ss, lambdas, s2, inv_s4, tol, out);
    }

    #[cfg(feature = "avx512")]
    #[archmage::attune]
    pub(crate) fn infow_map_into_v4x_t(
        token: archmage::X64V4xToken,
        g: &[f32],
        vv: &[f32],
        ss: &[f32],
        lambdas: &[f32],
        s2: f32,
        inv_s4: f32,
        tol: f32,
        out: &mut [f32],
    ) {
        infow_kernel(token, g, vv, ss, lambdas, s2, inv_s4, tol, out);
    }
}

#[cfg(target_arch = "aarch64")]
use infow_neon::*;

#[cfg(target_arch = "aarch64")]
mod infow_neon {
    use super::*;

    #[archmage::attune]
    pub(crate) fn infow_map_into_neon_t(
        token: archmage::NeonToken,
        g: &[f32],
        vv: &[f32],
        ss: &[f32],
        lambdas: &[f32],
        s2: f32,
        inv_s4: f32,
        tol: f32,
        out: &mut [f32],
    ) {
        infow_kernel(token, g, vv, ss, lambdas, s2, inv_s4, tol, out);
    }
}

#[cfg(target_arch = "wasm32")]
use infow_wasm::*;

#[cfg(target_arch = "wasm32")]
mod infow_wasm {
    use super::*;

    #[archmage::attune]
    pub(crate) fn infow_map_into_wasm128_t(
        token: archmage::Wasm128Token,
        g: &[f32],
        vv: &[f32],
        ss: &[f32],
        lambdas: &[f32],
        s2: f32,
        inv_s4: f32,
        tol: f32,
        out: &mut [f32],
    ) {
        infow_kernel(token, g, vv, ss, lambdas, s2, inv_s4, tol, out);
    }
}

// ---------------------------------------------------------------------
// Gram accumulation — per-pixel `big_n × big_n` f64 outer product,
// upper triangle only. Vectorized over `j` in f64x4 chunks with
// two-rounding `a*y+g` (bit-identical to the scalar loop — no FMA
// contraction).
// ---------------------------------------------------------------------

// Fixed triangular packing for the default 3×3 neighborhood. SIMD lanes
// remain independent Gram coefficients, preserving ordered f64 accumulation.
#[archmage::attune(_*_t, define(f64x4))]
pub(crate) fn gram3_rows_inner(
    token: Token,
    img: &[f32],
    parent: Option<&[f32]>,
    stride: usize,
    row0: usize,
    col0: usize,
    nrows: usize,
    ncols: usize,
    taps: &[(i32, i32)],
    gram: &mut [f64],
) {
    let offsets: [usize; 9] = core::array::from_fn(|k| {
        let (dy, dx) = taps[k];
        ((row0 as i32 + dy) as usize) * stride + (col0 as i32 + dx) as usize
    });
    if let Some(p) = parent {
        let mut a0 = f64x4::from_array_t(token, [gram[0], gram[1], gram[2], gram[3]]);
        let mut a1 = f64x4::from_array_t(token, [gram[4], gram[5], gram[6], gram[7]]);
        let mut a2 = f64x4::from_array_t(token, [gram[8], gram[9], gram[11], gram[12]]);
        let mut a3 = f64x4::from_array_t(token, [gram[13], gram[14], gram[15], gram[16]]);
        let mut a4 = f64x4::from_array_t(token, [gram[17], gram[18], gram[19], gram[22]]);
        let mut a5 = f64x4::from_array_t(token, [gram[23], gram[24], gram[25], gram[26]]);
        let mut a6 = f64x4::from_array_t(token, [gram[27], gram[28], gram[29], gram[33]]);
        let mut a7 = f64x4::from_array_t(token, [gram[34], gram[35], gram[36], gram[37]]);
        let mut a8 = f64x4::from_array_t(token, [gram[38], gram[39], gram[44], gram[45]]);
        let mut a9 = f64x4::from_array_t(token, [gram[46], gram[47], gram[48], gram[49]]);
        let mut a10 = f64x4::from_array_t(token, [gram[55], gram[56], gram[57], gram[58]]);
        let mut a11 = f64x4::from_array_t(token, [gram[59], gram[66], gram[67], gram[68]]);
        let mut a12 = f64x4::from_array_t(token, [gram[69], gram[77], gram[78], gram[79]]);
        let mut a13 = f64x4::from_array_t(token, [gram[88], gram[89], gram[99], 0.0]);
        for r in 0..nrows {
            let row_off = r * stride;
            for c in 0..ncols {
                let mut y = [0.0f64; 10];
                for k in 0..9 {
                    y[k] = img[offsets[k] + row_off + c] as f64;
                }
                y[9] = p[(row0 + r) * stride + col0 + c] as f64;
                a0 += f64x4::from_array_t(token, [y[0], y[0], y[0], y[0]])
                    * f64x4::from_array_t(token, [y[0], y[1], y[2], y[3]]);
                a1 += f64x4::from_array_t(token, [y[0], y[0], y[0], y[0]])
                    * f64x4::from_array_t(token, [y[4], y[5], y[6], y[7]]);
                a2 += f64x4::from_array_t(token, [y[0], y[0], y[1], y[1]])
                    * f64x4::from_array_t(token, [y[8], y[9], y[1], y[2]]);
                a3 += f64x4::from_array_t(token, [y[1], y[1], y[1], y[1]])
                    * f64x4::from_array_t(token, [y[3], y[4], y[5], y[6]]);
                a4 += f64x4::from_array_t(token, [y[1], y[1], y[1], y[2]])
                    * f64x4::from_array_t(token, [y[7], y[8], y[9], y[2]]);
                a5 += f64x4::from_array_t(token, [y[2], y[2], y[2], y[2]])
                    * f64x4::from_array_t(token, [y[3], y[4], y[5], y[6]]);
                a6 += f64x4::from_array_t(token, [y[2], y[2], y[2], y[3]])
                    * f64x4::from_array_t(token, [y[7], y[8], y[9], y[3]]);
                a7 += f64x4::from_array_t(token, [y[3], y[3], y[3], y[3]])
                    * f64x4::from_array_t(token, [y[4], y[5], y[6], y[7]]);
                a8 += f64x4::from_array_t(token, [y[3], y[3], y[4], y[4]])
                    * f64x4::from_array_t(token, [y[8], y[9], y[4], y[5]]);
                a9 += f64x4::from_array_t(token, [y[4], y[4], y[4], y[4]])
                    * f64x4::from_array_t(token, [y[6], y[7], y[8], y[9]]);
                a10 += f64x4::from_array_t(token, [y[5], y[5], y[5], y[5]])
                    * f64x4::from_array_t(token, [y[5], y[6], y[7], y[8]]);
                a11 += f64x4::from_array_t(token, [y[5], y[6], y[6], y[6]])
                    * f64x4::from_array_t(token, [y[9], y[6], y[7], y[8]]);
                a12 += f64x4::from_array_t(token, [y[6], y[7], y[7], y[7]])
                    * f64x4::from_array_t(token, [y[9], y[7], y[8], y[9]]);
                a13 += f64x4::from_array_t(token, [y[8], y[8], y[9], 0.0])
                    * f64x4::from_array_t(token, [y[8], y[9], y[9], 0.0]);
            }
        }
        let v = a0.to_array();
        gram[0] = v[0];
        gram[1] = v[1];
        gram[2] = v[2];
        gram[3] = v[3];
        let v = a1.to_array();
        gram[4] = v[0];
        gram[5] = v[1];
        gram[6] = v[2];
        gram[7] = v[3];
        let v = a2.to_array();
        gram[8] = v[0];
        gram[9] = v[1];
        gram[11] = v[2];
        gram[12] = v[3];
        let v = a3.to_array();
        gram[13] = v[0];
        gram[14] = v[1];
        gram[15] = v[2];
        gram[16] = v[3];
        let v = a4.to_array();
        gram[17] = v[0];
        gram[18] = v[1];
        gram[19] = v[2];
        gram[22] = v[3];
        let v = a5.to_array();
        gram[23] = v[0];
        gram[24] = v[1];
        gram[25] = v[2];
        gram[26] = v[3];
        let v = a6.to_array();
        gram[27] = v[0];
        gram[28] = v[1];
        gram[29] = v[2];
        gram[33] = v[3];
        let v = a7.to_array();
        gram[34] = v[0];
        gram[35] = v[1];
        gram[36] = v[2];
        gram[37] = v[3];
        let v = a8.to_array();
        gram[38] = v[0];
        gram[39] = v[1];
        gram[44] = v[2];
        gram[45] = v[3];
        let v = a9.to_array();
        gram[46] = v[0];
        gram[47] = v[1];
        gram[48] = v[2];
        gram[49] = v[3];
        let v = a10.to_array();
        gram[55] = v[0];
        gram[56] = v[1];
        gram[57] = v[2];
        gram[58] = v[3];
        let v = a11.to_array();
        gram[59] = v[0];
        gram[66] = v[1];
        gram[67] = v[2];
        gram[68] = v[3];
        let v = a12.to_array();
        gram[69] = v[0];
        gram[77] = v[1];
        gram[78] = v[2];
        gram[79] = v[3];
        let v = a13.to_array();
        gram[88] = v[0];
        gram[89] = v[1];
        gram[99] = v[2];
    } else {
        let mut a0 = f64x4::from_array_t(token, [gram[0], gram[1], gram[2], gram[3]]);
        let mut a1 = f64x4::from_array_t(token, [gram[4], gram[5], gram[6], gram[7]]);
        let mut a2 = f64x4::from_array_t(token, [gram[8], gram[10], gram[11], gram[12]]);
        let mut a3 = f64x4::from_array_t(token, [gram[13], gram[14], gram[15], gram[16]]);
        let mut a4 = f64x4::from_array_t(token, [gram[17], gram[20], gram[21], gram[22]]);
        let mut a5 = f64x4::from_array_t(token, [gram[23], gram[24], gram[25], gram[26]]);
        let mut a6 = f64x4::from_array_t(token, [gram[30], gram[31], gram[32], gram[33]]);
        let mut a7 = f64x4::from_array_t(token, [gram[34], gram[35], gram[40], gram[41]]);
        let mut a8 = f64x4::from_array_t(token, [gram[42], gram[43], gram[44], gram[50]]);
        let mut a9 = f64x4::from_array_t(token, [gram[51], gram[52], gram[53], gram[60]]);
        let mut a10 = f64x4::from_array_t(token, [gram[61], gram[62], gram[70], gram[71]]);
        let mut a11 = f64x4::from_array_t(token, [gram[80], 0.0, 0.0, 0.0]);
        for r in 0..nrows {
            let row_off = r * stride;
            for c in 0..ncols {
                let mut y = [0.0f64; 10];
                for k in 0..9 {
                    y[k] = img[offsets[k] + row_off + c] as f64;
                }
                a0 += f64x4::from_array_t(token, [y[0], y[0], y[0], y[0]])
                    * f64x4::from_array_t(token, [y[0], y[1], y[2], y[3]]);
                a1 += f64x4::from_array_t(token, [y[0], y[0], y[0], y[0]])
                    * f64x4::from_array_t(token, [y[4], y[5], y[6], y[7]]);
                a2 += f64x4::from_array_t(token, [y[0], y[1], y[1], y[1]])
                    * f64x4::from_array_t(token, [y[8], y[1], y[2], y[3]]);
                a3 += f64x4::from_array_t(token, [y[1], y[1], y[1], y[1]])
                    * f64x4::from_array_t(token, [y[4], y[5], y[6], y[7]]);
                a4 += f64x4::from_array_t(token, [y[1], y[2], y[2], y[2]])
                    * f64x4::from_array_t(token, [y[8], y[2], y[3], y[4]]);
                a5 += f64x4::from_array_t(token, [y[2], y[2], y[2], y[2]])
                    * f64x4::from_array_t(token, [y[5], y[6], y[7], y[8]]);
                a6 += f64x4::from_array_t(token, [y[3], y[3], y[3], y[3]])
                    * f64x4::from_array_t(token, [y[3], y[4], y[5], y[6]]);
                a7 += f64x4::from_array_t(token, [y[3], y[3], y[4], y[4]])
                    * f64x4::from_array_t(token, [y[7], y[8], y[4], y[5]]);
                a8 += f64x4::from_array_t(token, [y[4], y[4], y[4], y[5]])
                    * f64x4::from_array_t(token, [y[6], y[7], y[8], y[5]]);
                a9 += f64x4::from_array_t(token, [y[5], y[5], y[5], y[6]])
                    * f64x4::from_array_t(token, [y[6], y[7], y[8], y[6]]);
                a10 += f64x4::from_array_t(token, [y[6], y[6], y[7], y[7]])
                    * f64x4::from_array_t(token, [y[7], y[8], y[7], y[8]]);
                a11 += f64x4::from_array_t(token, [y[8], 0.0, 0.0, 0.0])
                    * f64x4::from_array_t(token, [y[8], 0.0, 0.0, 0.0]);
            }
        }
        let v = a0.to_array();
        gram[0] = v[0];
        gram[1] = v[1];
        gram[2] = v[2];
        gram[3] = v[3];
        let v = a1.to_array();
        gram[4] = v[0];
        gram[5] = v[1];
        gram[6] = v[2];
        gram[7] = v[3];
        let v = a2.to_array();
        gram[8] = v[0];
        gram[10] = v[1];
        gram[11] = v[2];
        gram[12] = v[3];
        let v = a3.to_array();
        gram[13] = v[0];
        gram[14] = v[1];
        gram[15] = v[2];
        gram[16] = v[3];
        let v = a4.to_array();
        gram[17] = v[0];
        gram[20] = v[1];
        gram[21] = v[2];
        gram[22] = v[3];
        let v = a5.to_array();
        gram[23] = v[0];
        gram[24] = v[1];
        gram[25] = v[2];
        gram[26] = v[3];
        let v = a6.to_array();
        gram[30] = v[0];
        gram[31] = v[1];
        gram[32] = v[2];
        gram[33] = v[3];
        let v = a7.to_array();
        gram[34] = v[0];
        gram[35] = v[1];
        gram[40] = v[2];
        gram[41] = v[3];
        let v = a8.to_array();
        gram[42] = v[0];
        gram[43] = v[1];
        gram[44] = v[2];
        gram[50] = v[3];
        let v = a9.to_array();
        gram[51] = v[0];
        gram[52] = v[1];
        gram[53] = v[2];
        gram[60] = v[3];
        let v = a10.to_array();
        gram[61] = v[0];
        gram[62] = v[1];
        gram[70] = v[2];
        gram[71] = v[3];
        let v = a11.to_array();
        gram[80] = v[0];
    }
}

#[archmage::attune(_*_t, define(f64x4))]
pub(crate) fn gram_rows_inner(
    token: Token,
    img: &[f32],
    parent: Option<&[f32]>,
    stride: usize,
    row0: usize,
    col0: usize,
    nrows: usize,
    ncols: usize,
    taps: &[(i32, i32)],
    gram: &mut [f64],
) {
    let nb = taps.len();
    let big_n = nb + parent.is_some() as usize;
    for r in 0..nrows {
        let row_c = row0 + r;
        for c in 0..ncols {
            let col_c = col0 + c;
            let mut yv = [0.0_f64; 16];
            for (k, &(dy, dx)) in taps.iter().enumerate() {
                yv[k] = img[((row_c as i32 + dy) as usize) * stride + (col_c as i32 + dx) as usize]
                    as f64;
            }
            if let Some(p) = parent {
                yv[nb] = p[row_c * stride + col_c] as f64;
            }
            for i in 0..big_n {
                let a = f64x4::splat_t(token, yv[i]);
                let row = &mut gram[i * big_n + i..i * big_n + big_n];
                let (chunks, tail) = f64x4::partition_slice_mut_t(token, row);
                let mut jj = i;
                for ch in chunks.iter_mut() {
                    let y = f64x4::load_t(token, (&yv[jj..jj + 4]).try_into().unwrap());
                    let g = f64x4::load_t(token, ch);
                    (g + a * y).store(ch);
                    jj += 4;
                }
                let a_s = yv[i];
                for t in tail.iter_mut() {
                    *t += a_s * yv[jj];
                    jj += 1;
                }
            }
        }
    }
}
