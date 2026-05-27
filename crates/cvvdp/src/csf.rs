//! Vectorized per-pixel CSF apply (tile-fused design).
//!
//! **STATUS: HONEST-STOP.** Both a band-wide materialization approach
//! and a tile-fused approach were measured. Both regressed at 256-1024²
//! (+6 to +30 %) because the buffer materialization overhead exceeds
//! the SIMD exp saving (~2.6 % of wall from the flamegraph). The
//! original `pipeline::apply_csf_row_per_pixel` keeps the sensitivity
//! value in registers (stream-fused by LLVM). Any buffer-materialize
//! design loses that. The `compute_sensitivities_into` function and
//! its tests are retained as documentation for future reference.
//!
//! The cvvdp pipeline calls `apply_csf_row_per_pixel` six times per
//! pixel per band (3 channels × {t_p, r_p} loop, plus a baseband-diff
//! loop in the same shape). The per-pixel computation breaks into:

// Module exists for documentation and unit tests only — the production
// pipeline still calls `pipeline::apply_csf_row_per_pixel` inline.
//!
//! 1. Bracket arithmetic on `log_l` against the 32-entry `LOG_L_BKG_AXIS`
//!    (uniform-step in log space → no binary search; arithmetic + floor
//!    + frac + cast).
//! 2. LUT scatter-gather: `lo = logs_row[lo_idx]`, `hi = logs_row[lo_idx + 1]`.
//! 3. Linear interp + constant correction: `log_s_corr * LN_10`.
//! 4. Transcendental: `f32::exp(log_s_corr_scaled)`.
//!
//! Step 4 is the only transcendental on the hot path, ~2.6 % of total
//! wall per the Chunk 3 flamegraph. The scalar `f32::exp` is ~10-15
//! cycles per call; magetypes' `vexp_into` runs at ~1 cycle per lane
//! (8-wide AVX2 `exp_midp_unchecked` = `exp2(x * LOG2_E)`, ~128 ULP).
//!
//! ## Design: tile-fused, in-place
//!
//! The straightforward "materialize band-wide log_s_corr_scaled vec
//! then SIMD exp" pattern regresses at small band sizes (16-1024 px)
//! because the band-wide intermediate Vec blows the L1 / L2 working
//! set even though the SIMD exp itself is faster. Bench:
//! materialize-pattern regressed +14-30 % at 256-1024² (best-of-medians
//! single-thread) and only won at 2048² (-39 %).
//!
//! The tile-fused design materializes `(log_s * LN_10)` into an L1-
//! resident **fixed-size stack tile**, runs SIMD exp into a sibling
//! stack tile, then writes the consumed sensitivities into the
//! caller's output Vec. Each tile is processed fully before moving on
//! — the working set is `2 * TILE * 4 B = 2 KB`, well inside L1.
//!
//! Plus: we avoid all `Vec::clear / Vec::resize(n, 0.0)` overhead per
//! channel — the output Vec is resized once at the top, all writes
//! land on already-resident pages.
//!
//! ## Why we still write to an output Vec
//!
//! Each sensitivity is consumed TWICE per pixel — once for
//! `dis_band[i] * s[i]` → `t_p[i]`, once for `ref_band[i] * s[i]` →
//! `r_p[i]`. Computing once + reading twice is strictly cheaper than
//! the original "call apply_csf twice per pixel per channel" pattern,
//! even at f32::exp cost. The output Vec is the caller's recycled
//! per-band scratch slot from `BandWorkspace.{s_a,s_rg,s_vy}`.

use alloc::vec::Vec;

use cvvdp_gpu::kernels::csf::N_L_BKG;

use crate::simd_math::vexp_into;

/// `SENSITIVITY_CORRECTION_DB / 20.0` — premultiplied constant added
/// to `log_s` before the `10^x` step (matches the GPU 3ch fused
/// kernel's `log_correction` constant + the legacy
/// `pipeline.rs::LOG_SENSITIVITY_CORRECTION`).
const LOG_SENSITIVITY_CORRECTION: f32 = cvvdp_gpu::kernels::csf::SENSITIVITY_CORRECTION_DB / 20.0;

/// `1.0 / (LOG_L_BKG_AXIS[N-1] - LOG_L_BKG_AXIS[0]) * (N - 1)` = 31 /
/// 6.30103 ≈ 4.919830570... — the CSF L_bkg axis is uniform in log10
/// space (matches `pipeline.rs::CSF_L_BKG_INV_STEP`).
const CSF_L_BKG_INV_STEP: f32 = 4.919_830_6;
const CSF_L_BKG_AXIS_MIN: f32 = -2.301_03;
const CSF_L_BKG_MAX_IDX: f32 = 30.999_999;

/// Tile size for the L1-resident in-flight SIMD exp pipeline. 256 f32
/// = 1 KB per tile; we hold two (log_scaled + sensitivity) = 2 KB
/// total. Lots of headroom under typical L1d of 32-48 KB per core.
const TILE: usize = 256;

/// Fill `tile_log[..]` with `(log_s_corr * LN_10)` for `tile_log.len()`
/// successive pixels of `log_l_bkg`.
///
/// `log_l_bkg` and `tile_log` must have the same length (asserted in
/// debug). LLVM auto-vectorizes the body on AVX2 (gather to logs_row
/// is the only non-vector op; 32-entry LUT stays in L1).
#[inline(always)]
fn fill_log_scaled_tile(
    log_l_bkg: &[f32],
    logs_row: &[f32; N_L_BKG],
    tile_log: &mut [f32],
    correction_times_ln_10: f32,
    ln_10: f32,
) {
    debug_assert_eq!(log_l_bkg.len(), tile_log.len());
    for (out_i, &log_l) in tile_log.iter_mut().zip(log_l_bkg.iter()) {
        let off_raw = (log_l - CSF_L_BKG_AXIS_MIN) * CSF_L_BKG_INV_STEP;
        let off_lo = off_raw.clamp(0.0, CSF_L_BKG_MAX_IDX);
        let lo_idx_f = off_lo.floor();
        let frac = off_lo - lo_idx_f;
        let lo_idx = lo_idx_f as usize;
        // The clamp guarantees `lo_idx ∈ [0, 30]` so both indexed reads
        // are in-bounds. We use [f32; N_L_BKG] (not &[f32]) to give
        // LLVM the static-bound hint that elides the runtime check.
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
/// Tile-fused SIMD impl:
///   for each TILE-sized chunk of `log_l_bkg`:
///     1. scalar prologue → stack `tile_log[TILE]` holds `log_s * LN_10`
///     2. `vexp_into` → stack `tile_out[TILE]` holds sensitivities
///     3. copy `tile_out[..tile.len()]` into `out[i..i+tile.len()]`
///
/// `out` is resized to `log_l_bkg.len()` (always — caller's recycled
/// `BandWorkspace.s_*` slot grows once, never shrinks).
///
/// Numerical contract: at most ~1e-4 relative error vs scalar
/// `apply_csf_row_per_pixel` (driven by `exp_midp_unchecked`'s
/// ~128 ULP budget plus FMA reassociation). The downstream JOD
/// parity gate sits at 1e-4 absolute, plenty of margin.
#[inline]
pub(crate) fn compute_sensitivities_into(
    log_l_bkg: &[f32],
    logs_row: &[f32; N_L_BKG],
    out: &mut Vec<f32>,
) {
    let n = log_l_bkg.len();
    // Grow without zero-fill so the writes below land on resident
    // pages without touching every byte first.
    out.clear();
    out.reserve(n);
    // SAFETY-equivalent in safe Rust: we resize to n then overwrite
    // every element below. The intermediate zero-fill is the only
    // cost; that's a memset and stays inside one cache line per 64 B,
    // negligible vs the per-pixel compute.
    out.resize(n, 0.0);

    let correction_times_ln_10 = LOG_SENSITIVITY_CORRECTION * core::f32::consts::LN_10;
    let ln_10 = core::f32::consts::LN_10;

    // Two stack tiles — L1-resident, never spilled.
    let mut tile_log = [0.0_f32; TILE];
    let mut tile_out = [0.0_f32; TILE];

    let mut i = 0;
    while i < n {
        let end = core::cmp::min(i + TILE, n);
        let m = end - i;
        let in_slice = &log_l_bkg[i..end];
        let log_slice = &mut tile_log[..m];
        fill_log_scaled_tile(in_slice, logs_row, log_slice, correction_times_ln_10, ln_10);
        let out_slice = &mut tile_out[..m];
        vexp_into(log_slice, out_slice);
        out[i..end].copy_from_slice(out_slice);
        i = end;
    }
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
        // Also clamp boundaries by sampling [-3, 5].
        let logs_row_a = precompute_logs_row(4.0, CsfChannel::A);
        let logs_row_rg = precompute_logs_row(4.0, CsfChannel::Rg);
        let logs_row_vy = precompute_logs_row(4.0, CsfChannel::Vy);

        let mut s: u32 = 0x12345678;
        let mut prng = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / (1u32 << 24) as f32) * 8.0 - 3.0
        };

        // Cover sub-tile, tile-boundary, multi-tile, and tail-fraction
        // sizes so the partial-tile path is exercised.
        for &n in &[1, 7, 8, 9, 16, 17, 100, 256, 257, 511, 512, 513, 1024] {
            let log_l: Vec<f32> = (0..n).map(|_| prng()).collect();

            for (row, name) in [
                (&logs_row_a, "A"),
                (&logs_row_rg, "RG"),
                (&logs_row_vy, "VY"),
            ] {
                let mut got = Vec::new();
                compute_sensitivities_into(&log_l, row, &mut got);
                assert_eq!(got.len(), n);
                for (i, &ll) in log_l.iter().enumerate() {
                    let want = scalar_reference(ll, row);
                    let denom = want.abs().max(1e-6);
                    let rel = (got[i] - want).abs() / denom;
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
        let mut out = Vec::new();
        compute_sensitivities_into(&log_l, &logs_row, &mut out);
        assert!(out.is_empty());

        let log_l: Vec<f32> = vec![0.5];
        compute_sensitivities_into(&log_l, &logs_row, &mut out);
        assert_eq!(out.len(), 1);
        let want = scalar_reference(0.5, &logs_row);
        let rel = (out[0] - want).abs() / want.abs().max(1e-6);
        assert!(rel <= 1e-4, "single: got={} want={want} rel={rel}", out[0]);
    }

    #[test]
    fn boundary_clamps_match_scalar() {
        let logs_row = precompute_logs_row(4.0, CsfChannel::A);
        let log_l: Vec<f32> = vec![
            -10.0,
            CSF_L_BKG_AXIS_MIN,
            CSF_L_BKG_AXIS_MIN - 1.0,
            0.0,
            1.5,
            4.0,
            10.0,
        ];
        let mut out = Vec::new();
        compute_sensitivities_into(&log_l, &logs_row, &mut out);
        for (i, &ll) in log_l.iter().enumerate() {
            let want = scalar_reference(ll, &logs_row);
            let rel = (out[i] - want).abs() / want.abs().max(1e-6);
            assert!(
                rel <= 1e-4,
                "boundary i={i} log_l={ll}: got={} want={want} rel={rel}",
                out[i]
            );
        }
    }
}
