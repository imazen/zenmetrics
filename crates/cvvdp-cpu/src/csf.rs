//! Vectorized per-pixel CSF apply.
//!
//! The cvvdp pipeline calls `apply_csf_row_per_pixel` six times per
//! pixel per band (3 channels × 2 sides — REF and DIST share the same
//! `log_l_bkg` per pixel, so the bracket arithmetic is identical
//! between sides but the `logs_row` is fixed per (rho, channel)).
//!
//! Per-pixel work breaks into:
//!
//! 1. Bracket arithmetic: `(log_l - axis_min) * inv_step`, clamp, floor,
//!    frac, low-index `as usize`.
//! 2. LUT scatter-gather: `lo = logs_row[lo_idx]`, `hi = logs_row[lo_idx + 1]`.
//! 3. Linear interp + constant correction: `log_s_corr = lo + frac * (hi - lo) + LOG_SENSITIVITY_CORRECTION`.
//! 4. Transcendental: `exp(log_s_corr * LN_10)`.
//!
//! Step 4 — `f32::exp` — is the only transcendental, ~2.6 % of total
//! wall time per the Chunk 3 flamegraph. Steps 1-3 auto-vectorize on
//! AVX2 (the gather is unavoidable scalar but the 32-entry LUT is
//! L1-resident so it's a few cycles per lane). We split the loop into
//! two phases:
//!
//! - **Phase A** (scalar, auto-vectorizes): compute `log_s_corr * LN_10`
//!   into a caller-owned scratch buffer. No transcendentals here.
//! - **Phase B** (SIMD via `simd_math::vexp_into`): in-place exp of the
//!   scratch to produce the sensitivities.
//!
//! The caller (the band loop in `pipeline.rs`) then reads the
//! sensitivities from the scratch buffer and folds them into the
//! `t_p` / `r_p` / baseband-diff loops. Each sensitivity is consumed
//! TWICE per pixel (once for `dis * s` → `t_p`, once for `ref * s` →
//! `r_p`); pre-computing once + reading twice is strictly cheaper than
//! the old code's call-twice-per-pixel pattern.

use alloc::vec::Vec;

use cvvdp_gpu::kernels::csf::N_L_BKG;

use crate::simd_math::vexp_into;

/// `SENSITIVITY_CORRECTION_DB / 20.0` — premultiplied constant added
/// to `log_s` before the `10^x` step. Mirrors `pipeline.rs`'s
/// `LOG_SENSITIVITY_CORRECTION` (and the GPU kernel's
/// `log_correction` constant). Re-declared here so this module is
/// self-contained.
const LOG_SENSITIVITY_CORRECTION: f32 =
    cvvdp_gpu::kernels::csf::SENSITIVITY_CORRECTION_DB / 20.0;

/// `1.0 / (LOG_L_BKG_AXIS[N-1] - LOG_L_BKG_AXIS[0]) * (N - 1)` = 31 /
/// 6.30103 ≈ 4.919830570... — the CSF L_bkg axis is uniform in log10
/// space (matches `pipeline.rs::CSF_L_BKG_INV_STEP`).
const CSF_L_BKG_INV_STEP: f32 = 4.919_830_6;
const CSF_L_BKG_AXIS_MIN: f32 = -2.301_03;
const CSF_L_BKG_MAX_IDX: f32 = 30.999_999;

/// Phase A scalar prologue: compute `(log_s_corr * LN_10)` into `out`.
///
/// `log_s_corr = lo + frac * (hi - lo) + LOG_SENSITIVITY_CORRECTION`
/// where `(lo, hi, frac)` come from bracket interpolation on `logs_row`
/// at `log_l_bkg[i]`'s axis position.
///
/// LLVM auto-vectorizes this loop on AVX2 (we tested with `cargo asm`
/// during prototyping: the inner body lowers to `vmulps + vaddps` plus
/// a couple of `vmovss` gathers for the LUT lookups, ~8× faster than a
/// branch-y scalar loop on the same machine).
///
/// The `logs_row` array is 32 f32s = 128 B, fits in L1 trivially. We
/// take it by `&[f32; N_L_BKG]` rather than `&[f32]` so LLVM can elide
/// the slice-length checks on the indexed reads.
#[inline]
fn fill_log_s_scaled(
    log_l_bkg: &[f32],
    logs_row: &[f32; N_L_BKG],
    out: &mut [f32],
) {
    debug_assert_eq!(log_l_bkg.len(), out.len());
    let ln_10 = core::f32::consts::LN_10;
    // Pre-fold the correction constant into LN_10:
    //   (log_s + correction) * LN_10
    // = log_s * LN_10 + correction * LN_10
    // — one fewer add per pixel by hoisting the constant.
    let correction_times_ln_10 = LOG_SENSITIVITY_CORRECTION * ln_10;

    for (out_i, &log_l) in out.iter_mut().zip(log_l_bkg.iter()) {
        let off_raw = (log_l - CSF_L_BKG_AXIS_MIN) * CSF_L_BKG_INV_STEP;
        let off_lo = off_raw.clamp(0.0, CSF_L_BKG_MAX_IDX);
        let lo_idx_f = off_lo.floor();
        let frac = off_lo - lo_idx_f;
        // `off_lo` is clamped to [0, 30.999999] so the cast and the
        // `+ 1` index are both in-bounds without any defensive checks.
        let lo_idx = lo_idx_f as usize;
        let lo = logs_row[lo_idx];
        let hi = logs_row[lo_idx + 1];
        let log_s_raw = lo + frac * (hi - lo);
        *out_i = log_s_raw * ln_10 + correction_times_ln_10;
    }
}

/// Compute per-pixel CSF sensitivities for one channel into `out`.
///
/// `out[i] = 10 ^ (log_s_raw[i] + LOG_SENSITIVITY_CORRECTION)`
///        = `exp((log_s_raw[i] + LOG_SENSITIVITY_CORRECTION) * LN_10)`.
///
/// Both `tmp` and `out` are resized to `log_l_bkg.len()` first (cleared
/// + resized — caller-owned scratch, same recycle pattern as
/// `safe_pow_with_offset_into_vec`). `tmp` and `out` must point at
/// DIFFERENT memory because the SIMD exp kernel reads-then-writes a
/// whole 8-lane chunk at a time.
///
/// Two-phase implementation:
///
/// 1. Scalar prologue (LLVM auto-vectorizes): write
///    `(log_s_corr * LN_10)` into `tmp`.
/// 2. `vexp_into(tmp, out)` to apply the SIMD exp into `out`.
///
/// Numerical contract: at most ~5e-5 relative error vs
/// `apply_csf_row_per_pixel` (driven by magetypes'
/// `exp_midp_unchecked` budget). The downstream JOD parity gate sits
/// at 1e-4 absolute, which has ~3 orders of magnitude of margin.
#[inline]
pub(crate) fn compute_sensitivities_into(
    log_l_bkg: &[f32],
    logs_row: &[f32; N_L_BKG],
    tmp: &mut Vec<f32>,
    out: &mut Vec<f32>,
) {
    let n = log_l_bkg.len();
    tmp.clear();
    tmp.resize(n, 0.0);
    out.clear();
    out.resize(n, 0.0);
    fill_log_s_scaled(log_l_bkg, logs_row, tmp.as_mut_slice());
    vexp_into(tmp.as_slice(), out.as_mut_slice());
}

#[cfg(test)]
mod tests {
    use super::*;
    use cvvdp_gpu::kernels::csf::{CsfChannel, precompute_logs_row};

    /// Scalar reference replicating `pipeline::apply_csf_row_per_pixel`
    /// exactly so we can directly compare.
    fn scalar_reference(log_l: f32, logs_row: &[f32; N_L_BKG]) -> f32 {
        let off_raw = (log_l - CSF_L_BKG_AXIS_MIN) * CSF_L_BKG_INV_STEP;
        let off_lo = off_raw.clamp(0.0, CSF_L_BKG_MAX_IDX);
        let lo_idx_f = off_lo.floor();
        let frac = off_lo - lo_idx_f;
        let lo_idx = lo_idx_f as usize;
        let hi_idx = lo_idx + 1;
        let lo = logs_row[lo_idx];
        let hi = logs_row[hi_idx];
        let log_s_raw = lo + frac * (hi - lo);
        let log_s = log_s_raw + LOG_SENSITIVITY_CORRECTION;
        (log_s * core::f32::consts::LN_10).exp()
    }

    #[test]
    fn matches_scalar_csf_on_synthetic_log_l() {
        // Cover the realistic L_bkg range:
        //   log10(L_bkg) ∈ [-2.301, 4.0] (PQ peak ~10^4 nits).
        // Also clamp boundaries.
        let logs_row_a = precompute_logs_row(4.0, CsfChannel::A);
        let logs_row_rg = precompute_logs_row(4.0, CsfChannel::Rg);
        let logs_row_vy = precompute_logs_row(4.0, CsfChannel::Vy);

        // Same PRNG shape as masking::tests for reproducibility.
        let mut s: u32 = 0x12345678;
        let mut prng = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            // Map to [-3.0, 5.0] so we hit both clamp boundaries
            // (axis is [-2.301, 4.0]).
            ((s >> 8) as f32 / (1u32 << 24) as f32) * 8.0 - 3.0
        };

        for &n in &[1, 7, 8, 9, 16, 17, 100, 1024] {
            let log_l: Vec<f32> = (0..n).map(|_| prng()).collect();

            for (row, name) in [
                (&logs_row_a, "A"),
                (&logs_row_rg, "RG"),
                (&logs_row_vy, "VY"),
            ] {
                let mut tmp = Vec::new();
                let mut got = Vec::new();
                compute_sensitivities_into(&log_l, row, &mut tmp, &mut got);
                assert_eq!(got.len(), n);
                for (i, &ll) in log_l.iter().enumerate() {
                    let want = scalar_reference(ll, row);
                    let denom = want.abs().max(1e-6);
                    let rel = (got[i] - want).abs() / denom;
                    // 1e-4 relative is comfortable — vexp_midp_unchecked
                    // is ~1e-5, plus the FMA reassociation in the
                    // prologue contributes maybe a couple ULPs.
                    assert!(
                        rel <= 1e-4,
                        "n={n} ch={name} idx={i} log_l={ll}: got={} want={want} rel={rel}",
                        got[i]
                    );
                }
            }
        }
    }

    #[test]
    fn handles_empty_and_single_element() {
        let logs_row = precompute_logs_row(4.0, CsfChannel::A);
        let log_l: Vec<f32> = Vec::new();
        let mut tmp = Vec::new();
        let mut out = Vec::new();
        compute_sensitivities_into(&log_l, &logs_row, &mut tmp, &mut out);
        assert!(out.is_empty());

        let log_l: Vec<f32> = vec![0.5];
        compute_sensitivities_into(&log_l, &logs_row, &mut tmp, &mut out);
        assert_eq!(out.len(), 1);
        let want = scalar_reference(0.5, &logs_row);
        let rel = (out[0] - want).abs() / want.abs().max(1e-6);
        assert!(rel <= 1e-4, "single: got={} want={want} rel={rel}", out[0]);
    }

    #[test]
    fn boundary_clamps_match_scalar() {
        // Hit both axis ends and points outside the axis range
        // (the clamp must produce identical lo_idx/frac as the
        // scalar reference).
        let logs_row = precompute_logs_row(4.0, CsfChannel::A);
        let log_l: Vec<f32> = vec![
            -10.0,
            CSF_L_BKG_AXIS_MIN, // axis min
            CSF_L_BKG_AXIS_MIN - 1.0,
            0.0,
            1.5,
            4.0, // axis max
            10.0,
        ];
        let mut tmp = Vec::new();
        let mut out = Vec::new();
        compute_sensitivities_into(&log_l, &logs_row, &mut tmp, &mut out);
        for (i, &ll) in log_l.iter().enumerate() {
            let want = scalar_reference(ll, &logs_row);
            let rel = (out[i] - want).abs() / want.abs().max(1e-6);
            assert!(rel <= 1e-4, "boundary i={i} log_l={ll}: got={} want={want} rel={rel}", out[i]);
        }
    }
}
