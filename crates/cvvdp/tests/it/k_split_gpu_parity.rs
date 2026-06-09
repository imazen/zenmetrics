//! D1 cross-check via public API: the strip walker's `h_body` validator
//! must reject non-power-of-two values, matching
//! `cvvdp-gpu::pipeline::is_valid_strip_h_body`. The K_SPLIT helper-
//! table tests live in `src/strip.rs#tests` (run via
//! `cargo test --lib strip::tests`) — they exercise the helpers
//! directly without going through the scorer.
//!
//! These tests sit at the integration-test level so a future divergence
//! at the public API boundary (e.g., the validator drifts to accept
//! odd `h_body`) fails here before any score-side parity test fires.

use cvvdp::{Cvvdp, CvvdpParams};

/// Build a synth sRGB pair via XorShift64.
fn synth_pair(w: u32, h: u32, seed: u64) -> (Vec<u8>, Vec<u8>) {
    let n = (w as usize) * (h as usize) * 3;
    let mut r = vec![0u8; n];
    let mut d = vec![0u8; n];
    let mut s_r = seed;
    let mut s_d = seed.wrapping_mul(2_654_435_769);
    for i in 0..n {
        s_r ^= s_r << 13;
        s_r ^= s_r >> 7;
        s_r ^= s_r << 17;
        r[i] = (s_r & 0xFF) as u8;
        s_d ^= s_d << 13;
        s_d ^= s_d >> 7;
        s_d ^= s_d << 17;
        let mixed = (r[i] as u16) * 230 + ((s_d as u8) as u16) * 25;
        d[i] = ((mixed / 256) as u8).min(255);
    }
    (r, d)
}

#[test]
fn score_strip_rejects_non_pow2_h_body() {
    let (r, d) = synth_pair(128, 128, 0xfeed);
    let mut s = Cvvdp::new(128, 128, CvvdpParams::default()).unwrap();
    assert!(s.score_strip(&r, &d, 0).is_err());
    assert!(s.score_strip(&r, &d, 3).is_err());
    assert!(s.score_strip(&r, &d, 100).is_err());
    assert!(s.score_strip(&r, &d, 1).is_ok());
    assert!(s.score_strip(&r, &d, 64).is_ok());
}

#[test]
fn score_strip_matches_full_at_256_256_h_body_64() {
    let (r, d) = synth_pair(256, 256, 0xabba);
    let mut s_full = Cvvdp::new(256, 256, CvvdpParams::default()).unwrap();
    let mut s_strip = Cvvdp::new(256, 256, CvvdpParams::default()).unwrap();
    let full = s_full.score(&r, &d).unwrap();
    let strip = s_strip.score_strip(&r, &d, 64).unwrap();
    assert_eq!(
        full.to_bits(),
        strip.to_bits(),
        "strip pool walker must produce bit-identical JOD vs Full (got strip={strip}, full={full})"
    );
}

#[test]
fn score_strip_dispatch_counter_partitions_at_256() {
    let (r, d) = synth_pair(256, 256, 0xc0ffee);
    let mut s = Cvvdp::new(256, 256, CvvdpParams::default()).unwrap();
    let _jod = s.score_strip(&r, &d, 64).unwrap();
    let count = s.strip_dispatch_counter();
    // At 256² with h_body=64, level 0 band height = 256, strip body
    // at level 0 = 64 → 4 strips, ×3 channels = 12 dispatches for
    // level 0 alone. At level 1 (128 rows, strip body 32) → 4×3 =
    // 12. At level 2 (64 rows, strip body 16) → 4×3 = 12. At deeper
    // levels strips collapse to 1×3 each. Total >= 12 just from
    // level 0.
    assert!(
        count >= 12,
        "strip_dispatch_counter should report >= 12 after 256² h_body=64 partition; got {count}"
    );
}

#[test]
fn score_strip_dispatch_counter_is_cumulative() {
    let (r1, d1) = synth_pair(256, 256, 0xa);
    let (r2, d2) = synth_pair(256, 256, 0xb);
    let mut s = Cvvdp::new(256, 256, CvvdpParams::default()).unwrap();
    let _ = s.score_strip(&r1, &d1, 64).unwrap();
    let c1 = s.strip_dispatch_counter();
    let _ = s.score_strip(&r2, &d2, 64).unwrap();
    let c2 = s.strip_dispatch_counter();
    // Cumulative semantics (matches GPU): c2 = 2·c1 after two
    // identical-partitioning calls.
    assert_eq!(
        c2,
        2 * c1,
        "counter is cumulative (matches GPU) (got c1={c1}, c2={c2})"
    );
    // Full mode does NOT increment.
    let pre = s.strip_dispatch_counter();
    let _ = s.score(&r1, &d1).unwrap();
    let post = s.strip_dispatch_counter();
    assert_eq!(
        pre, post,
        "Full score must not increment strip counter (pre={pre}, post={post})"
    );
    // Explicit reset works.
    s.reset_strip_dispatch_counter();
    assert_eq!(s.strip_dispatch_counter(), 0);
}
