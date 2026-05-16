//! Additional invariant pins on [`precompute_logs_row`] beyond the
//! 3 existing tests in `csf_scalar.rs` (length, axis-point identity,
//! rho-dependence). The existing tests cover correctness vs
//! `sensitivity_scalar`; this file pins:
//!
//! - Determinism (bit-equality on repeat calls).
//! - Channel-A / Rg / Vy produce DISTINCT rows for the same rho
//!   (catches a refactor that collapses the channel input).
//! - Every entry is finite for valid inputs.
//! - Rho-clamp at the low end (rho → 0 clamps to 1e-6, no NaN).
//! - Rho extrapolation at the high end (rho beyond LUT axis clamps
//!   to last column, no NaN/Inf).
//! - Output represents POSITIVE sensitivities (`10^row[k] > 0`).

use cvvdp_gpu::kernels::csf::{CsfChannel, N_L_BKG, precompute_logs_row};

#[test]
fn determinism_across_repeated_calls() {
    // Pure function — same (rho, cc) yields bit-identical row.
    for cc in [CsfChannel::A, CsfChannel::Rg, CsfChannel::Vy] {
        for &rho in &[0.5_f32, 4.0, 32.0] {
            let a = precompute_logs_row(rho, cc);
            let b = precompute_logs_row(rho, cc);
            for k in 0..N_L_BKG {
                assert_eq!(
                    a[k].to_bits(),
                    b[k].to_bits(),
                    "non-deterministic at cc={cc:?} rho={rho} k={k}: {} vs {}",
                    a[k],
                    b[k]
                );
            }
        }
    }
}

#[test]
fn distinct_rows_across_channels() {
    // The 3 channels query different per-channel LUTs (via
    // `channel_lut(cc)`). For the same rho, the rows must NOT be
    // identical across channels — otherwise a refactor that
    // accidentally drops the channel argument would silently
    // produce identical CSF weights for A / Rg / Vy.
    for &rho in &[1.0_f32, 4.0, 16.0] {
        let row_a = precompute_logs_row(rho, CsfChannel::A);
        let row_rg = precompute_logs_row(rho, CsfChannel::Rg);
        let row_vy = precompute_logs_row(rho, CsfChannel::Vy);

        // Pairwise: at least one entry must differ.
        let differ_a_rg = (0..N_L_BKG).any(|k| row_a[k].to_bits() != row_rg[k].to_bits());
        let differ_a_vy = (0..N_L_BKG).any(|k| row_a[k].to_bits() != row_vy[k].to_bits());
        let differ_rg_vy = (0..N_L_BKG).any(|k| row_rg[k].to_bits() != row_vy[k].to_bits());

        assert!(
            differ_a_rg,
            "A and Rg rows identical at rho={rho} — channel input collapsed?"
        );
        assert!(
            differ_a_vy,
            "A and Vy rows identical at rho={rho} — channel input collapsed?"
        );
        assert!(
            differ_rg_vy,
            "Rg and Vy rows identical at rho={rho} — channel input collapsed?"
        );
    }
}

#[test]
fn all_entries_finite_for_realistic_rho_range() {
    // Sweep rho from sub-LUT-axis to super-LUT-axis. The function
    // clamps rho.max(1e-6) at the low end; LUT clamps extrapolation
    // at the high end. Both must yield finite output.
    for cc in [CsfChannel::A, CsfChannel::Rg, CsfChannel::Vy] {
        for &rho in &[0.001_f32, 0.1, 1.0, 4.0, 16.0, 64.0, 256.0, 1024.0] {
            let row = precompute_logs_row(rho, cc);
            for (k, &v) in row.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "non-finite at cc={cc:?} rho={rho} k={k}: {v}"
                );
            }
        }
    }
}

#[test]
fn zero_rho_does_not_panic_or_produce_nan() {
    // Per source: `let log_rho_q = rho.max(1e-6).log10();`. Passing
    // rho=0 must clamp to 1e-6, not produce log10(0) = -inf which
    // would then propagate as NaN through interpolation.
    for cc in [CsfChannel::A, CsfChannel::Rg, CsfChannel::Vy] {
        let row = precompute_logs_row(0.0, cc);
        for (k, &v) in row.iter().enumerate() {
            assert!(
                v.is_finite(),
                "rho=0 cc={cc:?} k={k}: {v} — rho clamp failed?"
            );
        }
    }
}

#[test]
fn negative_rho_does_not_panic_or_produce_nan() {
    // Negative rho also goes through `rho.max(1e-6)` → 1e-6, the
    // same clamp. Pin so a refactor that uses `rho.abs()` (would
    // give a different positive value) instead of `.max(1e-6)`
    // trips here.
    for cc in [CsfChannel::A, CsfChannel::Rg, CsfChannel::Vy] {
        let row = precompute_logs_row(-1.0, cc);
        for (k, &v) in row.iter().enumerate() {
            assert!(
                v.is_finite(),
                "rho=-1 cc={cc:?} k={k}: {v} — clamp not applied?"
            );
        }
        // It should also match the rho = 1e-6 result (clamp target).
        let row_clamp = precompute_logs_row(1e-6, CsfChannel::A);
        let row_neg = precompute_logs_row(-100.0, CsfChannel::A);
        for k in 0..N_L_BKG {
            assert_eq!(
                row_clamp[k].to_bits(),
                row_neg[k].to_bits(),
                "rho=1e-6 vs rho=-100 mismatch at k={k}: clamp not via .max()?"
            );
        }
    }
}

#[test]
fn ten_to_the_row_is_strictly_positive() {
    // The row contains `log10(sensitivity)`. Sensitivity is a
    // physical contrast-sensitivity value, always strictly > 0.
    // So `10^row[k]` must be > 0 (i.e. `row[k]` is finite — no -inf
    // representing zero sensitivity).
    for cc in [CsfChannel::A, CsfChannel::Rg, CsfChannel::Vy] {
        for &rho in &[0.1_f32, 1.0, 4.0, 16.0, 64.0] {
            let row = precompute_logs_row(rho, cc);
            for (k, &v) in row.iter().enumerate() {
                let s = 10.0_f32.powf(v);
                assert!(
                    s > 0.0 && s.is_finite(),
                    "10^row[{k}] = {s} not positive-finite at cc={cc:?} rho={rho}"
                );
            }
        }
    }
}
