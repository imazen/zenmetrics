//! Phase 9.Z.A — verify the `score_strip` / `score_with_warm_ref_strip`
//! API stubs return the same value as the underlying full-pipeline
//! call. These are stub implementations until the memory-bounded
//! walker lands (multi-day work, see `Cvvdp::score_strip` docs).
//!
//! When the walker ships, these tests should be tightened to
//! verify peak heap reduction in addition to score equivalence.

use cvvdp::{Cvvdp, CvvdpParams};

/// Build a synth sRGB pair using XorShift64.
fn synth_pair(w: u32, h: u32, seed: u64) -> (Vec<u8>, Vec<u8>) {
    let n = (w as usize) * (h as usize) * 3;
    let mut ref_buf = vec![0u8; n];
    let mut dist_buf = vec![0u8; n];
    let mut s_ref = seed;
    let mut s_dis = seed.wrapping_mul(2_654_435_769);
    for i in 0..n {
        s_ref ^= s_ref << 13;
        s_ref ^= s_ref >> 7;
        s_ref ^= s_ref << 17;
        ref_buf[i] = (s_ref & 0xFF) as u8;
        s_dis ^= s_dis << 13;
        s_dis ^= s_dis >> 7;
        s_dis ^= s_dis << 17;
        let mixed = (ref_buf[i] as u16) * 230 + ((s_dis as u8) as u16) * 25;
        dist_buf[i] = ((mixed / 256) as u8).min(255);
    }
    (ref_buf, dist_buf)
}

#[test]
fn score_strip_matches_score_when_stubbed() {
    let (ref_buf, dist_buf) = synth_pair(256, 256, 0xc0_ffee_12_34);
    let mut s_full = Cvvdp::new(256, 256, CvvdpParams::default()).unwrap();
    let mut s_strip = Cvvdp::new(256, 256, CvvdpParams::default()).unwrap();
    let full = s_full.score(&ref_buf, &dist_buf).unwrap();
    let strip = s_strip.score_strip(&ref_buf, &dist_buf, 128).unwrap();
    assert_eq!(
        full, strip,
        "score_strip stub must equal score() (no memory walker yet)"
    );
}

#[test]
fn score_with_warm_ref_strip_matches_warm_when_stubbed() {
    let (ref_buf, dist_buf) = synth_pair(256, 256, 0xa1_b2_c3_d4);
    let mut s_full = Cvvdp::new(256, 256, CvvdpParams::default()).unwrap();
    let mut s_strip = Cvvdp::new(256, 256, CvvdpParams::default()).unwrap();
    s_full.warm_reference(&ref_buf).unwrap();
    s_strip.warm_reference(&ref_buf).unwrap();
    let full = s_full.score_with_warm_ref(&dist_buf).unwrap();
    let strip = s_strip.score_with_warm_ref_strip(&dist_buf, 128).unwrap();
    assert_eq!(
        full, strip,
        "score_with_warm_ref_strip stub must equal score_with_warm_ref()"
    );
}

#[test]
fn warm_ref_strip_errors_without_warm() {
    let (_, dist_buf) = synth_pair(256, 256, 0xc0_ffee_12_34);
    let mut s = Cvvdp::new(256, 256, CvvdpParams::default()).unwrap();
    assert!(
        s.score_with_warm_ref_strip(&dist_buf, 128).is_err(),
        "warm_ref_strip must error without warm_reference"
    );
}
