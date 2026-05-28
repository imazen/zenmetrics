//! D1 cross-check: the cvvdp CPU `strip` helpers must be bit-identical
//! to the `cvvdp-gpu::pipeline::{mode_b_k_split, mode_b_halo_at_level,
//! mode_b_strip_h_at_level}` helpers. This is the K_SPLIT-table sizing
//! invariant: if the CPU and GPU walkers disagree on which level is
//! shallow vs deep, the parity gate between CPU strip JOD and GPU
//! strip JOD has no chance.
//!
//! We can't depend on `cvvdp-gpu` (cyclic — cvvdp-gpu depends on cvvdp).
//! Instead we re-compute the doc-table values from-scratch here using
//! the exact GPU recurrence text, and pin them against the CPU
//! implementation. The GPU source-of-truth is at
//! `crates/cvvdp-gpu/src/pipeline.rs:1564-1652`.
//!
//! This test sits at the workspace-test level so a future divergence
//! (one side bumps the threshold, the other doesn't) shows up here
//! before any score-side parity test fires.

#[path = "../src/strip.rs"]
#[allow(dead_code)]
mod strip;

use strip::{mode_b_halo_at_level, mode_b_k_split, mode_b_strip_h_at_level};

#[test]
fn doc_table_at_h_body_512_k_split_6() {
    // Pinned from GPU pipeline.rs:1610-1620 doc table:
    //   | k | body_k | R_k                              |
    //   |---|--------|----------------------------------|
    //   | 5 | 16     | 32                               |
    //   | 4 | 32     | max(48, 2·32+4) = 68             |
    //   | 3 | 64     | max(80, 2·68+4) = 140            |
    //   | 2 | 128    | max(144, 2·140+4) = 284          |
    //   | 1 | 256    | max(272, 2·284+4) = 572          |
    //   | 0 | 512    | max(528, 2·572+4) = **1148**     |
    let h_body = 512;
    let k_split = mode_b_k_split(h_body, 9);
    assert_eq!(k_split, 6);
    assert_eq!(mode_b_strip_h_at_level(0, h_body, k_split), 1148);
    assert_eq!(mode_b_strip_h_at_level(1, h_body, k_split), 572);
    assert_eq!(mode_b_strip_h_at_level(2, h_body, k_split), 284);
    assert_eq!(mode_b_strip_h_at_level(3, h_body, k_split), 140);
    assert_eq!(mode_b_strip_h_at_level(4, h_body, k_split), 68);
    assert_eq!(mode_b_strip_h_at_level(5, h_body, k_split), 32);
    assert_eq!(mode_b_strip_h_at_level(6, h_body, k_split), 0);
    assert_eq!(mode_b_strip_h_at_level(7, h_body, k_split), 0);
    assert_eq!(mode_b_strip_h_at_level(8, h_body, k_split), 0);
}

#[test]
fn halo_at_level_constants() {
    // Pinned from GPU pipeline.rs:1587-1594 — shallow=8 (PU radius 6 +
    // downscale slack 2), deep=0 (no halo padding needed).
    assert_eq!(mode_b_halo_at_level(0, 6), 8);
    assert_eq!(mode_b_halo_at_level(5, 6), 8);
    assert_eq!(mode_b_halo_at_level(6, 6), 0);
    assert_eq!(mode_b_halo_at_level(8, 6), 0);
}

#[test]
fn k_split_sweep_realistic_grid() {
    // For every reasonable (h_body, n_levels) pair the strip walker
    // could be invoked with, k_split must:
    //   1. fall in [0, n_levels]
    //   2. be the largest k such that (h_body >> k) >= 12
    for h_body in [16_u32, 32, 64, 128, 256, 512, 1024, 2048] {
        for n_levels in [3_u32, 5, 7, 9] {
            let k = mode_b_k_split(h_body, n_levels);
            assert!(k <= n_levels);
            // Property: at k_split itself the body MUST be < 12 (so
            // the loop exited). And at k_split - 1 (if positive) the
            // body MUST be ≥ 12.
            if k < n_levels {
                let body_at_k = h_body.checked_shr(k).unwrap_or(0);
                assert!(
                    body_at_k < 12,
                    "k_split={k} fired but body_at_k={body_at_k} ≥ 12 (h_body={h_body}, n_levels={n_levels})"
                );
            }
            if k > 0 {
                let body_at_km1 = h_body >> (k - 1);
                assert!(
                    body_at_km1 >= 12,
                    "k_split - 1 = {} should still be shallow (body={body_at_km1} < 12)",
                    k - 1
                );
            }
        }
    }
}
