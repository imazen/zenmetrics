//! Phase 9.Z.A parity tests: `score_strip` vs `score` at multiple
//! image sizes and strip body heights.
//!
//! The strip walker is designed for bit-identical body rows. We pin
//! score parity at a tight tolerance (1e-4 absolute, 1e-4 relative)
//! covering the f64 reduction-order tolerance from per-strip
//! accumulators.

use iwssim::{Iwssim, IwssimParams, STRIP_BODY_DEFAULT, STRIP_BODY_MIN, STRIP_HALO_ROWS};

/// XorShift64-style seeded pseudo-random byte generator.
fn synth_rgb_pair(w: u32, h: u32, seed: u64) -> (Vec<u8>, Vec<u8>) {
    let n = (w as usize) * (h as usize) * 3;
    let mut ref_buf = vec![0u8; n];
    let mut dis_buf = vec![0u8; n];
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
        // Distortion: blend with ref to keep score in a useful range
        let mixed = (ref_buf[i] as u16) * 230 + ((s_dis as u8) as u16) * 25;
        dis_buf[i] = ((mixed / 256) as u8).min(255);
    }
    (ref_buf, dis_buf)
}

fn parity_at(w: u32, h: u32, strip_h: u32, seed: u64, tol_abs: f64) {
    let (ref_buf, dis_buf) = synth_rgb_pair(w, h, seed);
    let mut scorer_full = Iwssim::new(w, h).expect("full Iwssim");
    let full = scorer_full
        .score(&ref_buf, &dis_buf)
        .expect("full score");
    let mut scorer_strip = Iwssim::new(w, h).expect("strip Iwssim");
    let strip = scorer_strip
        .score_strip(&ref_buf, &dis_buf, strip_h)
        .expect("strip score");

    let diff = (full.score - strip.score).abs();
    eprintln!(
        "size {}x{} strip {} seed {}: full={:.6} strip={:.6} diff={:.6e}",
        w, h, strip_h, seed, full.score, strip.score, diff
    );
    for s in 0..5 {
        eprintln!(
            "  scale {}: full={:.6} strip={:.6} diff={:.6e}",
            s,
            full.per_scale[s],
            strip.per_scale[s],
            (full.per_scale[s] - strip.per_scale[s]).abs()
        );
    }
    assert!(
        diff < tol_abs,
        "size {}x{} strip {} seed {}: score diff {:.6e} exceeds tol {:.6e} (full={}, strip={})",
        w,
        h,
        strip_h,
        seed,
        diff,
        tol_abs,
        full.score,
        strip.score
    );
}

#[test]
fn strip_parity_256x256_body512() {
    // strip_h > image_h ⇒ one strip = full image. Should be exact.
    parity_at(256, 256, STRIP_BODY_DEFAULT, 0xc0_ffee_12_34, 1e-5);
}

#[test]
fn strip_parity_512x512_body256() {
    // 2 strips with adequate halo.
    parity_at(512, 512, 256, 0xc0_ffee_12_34, 1e-4);
}

#[test]
fn strip_parity_512x512_body512() {
    // strip_h == image_h ⇒ one strip = full image. Should be exact.
    parity_at(512, 512, 512, 0xc0_ffee_12_34, 1e-5);
}

#[test]
fn strip_parity_1024x1024_body512() {
    // 2 strips at the production-default body. Common production
    // case for ~1MP inputs that fit Full easily.
    parity_at(1024, 1024, 512, 0xa1_b2_c3_d4, 1e-4);
}

#[test]
fn strip_parity_1024x1024_body256() {
    // 4 strips.
    parity_at(1024, 1024, 256, 0xa1_b2_c3_d4, 1e-4);
}

#[test]
fn strip_parity_512x256_body128() {
    // 2 strips on tall image. Tests that the strip walker handles
    // non-square images.
    parity_at(512, 256, 128, 0xdead_beef_cafe_babe, 1e-4);
}

#[test]
fn strip_body_too_small_uses_default() {
    let (ref_buf, dis_buf) = synth_rgb_pair(256, 256, 0xc0_ffee_12_34);
    let mut scorer = Iwssim::new(256, 256).expect("Iwssim");
    let result = scorer.score_strip(&ref_buf, &dis_buf, STRIP_BODY_MIN - 1);
    assert!(
        result.is_ok(),
        "score_strip should silently fall back to default for too-small strip_h"
    );
}

#[test]
fn strip_constants_round_to_halving_alignment() {
    // Strip halo must be aligned so that 5 levels of halving don't
    // produce fractional halo rows. 1<<4 = 16 alignment.
    assert_eq!(STRIP_HALO_ROWS % 16, 0);
    // Default body must also be aligned.
    assert_eq!(STRIP_BODY_DEFAULT % 16, 0);
}

fn warm_ref_parity_at(w: u32, h: u32, strip_h: u32, seed: u64, tol_abs: f64) {
    let (ref_buf, dis_buf) = synth_rgb_pair(w, h, seed);
    let mut scorer_full = Iwssim::new(w, h).expect("full Iwssim");
    scorer_full
        .warm_reference(&ref_buf)
        .expect("warm_reference");
    let full = scorer_full
        .score_with_warm_ref(&dis_buf)
        .expect("warm score");
    let mut scorer_strip = Iwssim::new(w, h).expect("strip Iwssim");
    scorer_strip
        .warm_reference(&ref_buf)
        .expect("warm_reference strip");
    let strip = scorer_strip
        .score_with_warm_ref_strip(&dis_buf, strip_h)
        .expect("warm_ref_strip score");
    let diff = (full.score - strip.score).abs();
    eprintln!(
        "warm_ref_strip {}x{} strip {} seed {}: full={:.6} strip={:.6} diff={:.6e}",
        w, h, strip_h, seed, full.score, strip.score, diff
    );
    for s in 0..5 {
        eprintln!(
            "  scale {}: full={:.6} strip={:.6} diff={:.6e}",
            s,
            full.per_scale[s],
            strip.per_scale[s],
            (full.per_scale[s] - strip.per_scale[s]).abs()
        );
    }
    assert!(
        diff < tol_abs,
        "warm_ref_strip {}x{} strip {} seed {}: score diff {:.6e} exceeds tol {:.6e} (full={}, strip={})",
        w,
        h,
        strip_h,
        seed,
        diff,
        tol_abs,
        full.score,
        strip.score
    );
}

#[test]
fn warm_ref_strip_parity_512x512_body512() {
    warm_ref_parity_at(512, 512, 512, 0xc0_ffee_12_34, 1e-5);
}

#[test]
fn warm_ref_strip_parity_512x512_body256() {
    warm_ref_parity_at(512, 512, 256, 0xc0_ffee_12_34, 1e-4);
}

#[test]
fn warm_ref_strip_parity_1024x1024_body512() {
    warm_ref_parity_at(1024, 1024, 512, 0xa1_b2_c3_d4, 1e-4);
}

#[test]
fn warm_ref_strip_parity_1024x1024_body256() {
    warm_ref_parity_at(1024, 1024, 256, 0xa1_b2_c3_d4, 1e-4);
}

#[test]
fn warm_ref_strip_no_warm_returns_error() {
    let (_, dis_buf) = synth_rgb_pair(256, 256, 0xc0_ffee_12_34);
    let mut scorer = Iwssim::new(256, 256).expect("Iwssim");
    let result = scorer.score_with_warm_ref_strip(&dis_buf, 256);
    assert!(
        result.is_err(),
        "score_with_warm_ref_strip should error without warm_reference"
    );
}

#[test]
fn strip_with_iw_flag_off() {
    // Sanity test the non-IW path through the strip walker too.
    let (ref_buf, dis_buf) = synth_rgb_pair(512, 512, 0xc0_ffee_12_34);
    let params = IwssimParams {
        iw_flag: false,
        ..IwssimParams::default()
    };
    let mut scorer_full = Iwssim::with_params(512, 512, params).expect("full Iwssim");
    let full = scorer_full
        .score(&ref_buf, &dis_buf)
        .expect("full score");
    let mut scorer_strip = Iwssim::with_params(512, 512, params).expect("strip Iwssim");
    let strip = scorer_strip
        .score_strip(&ref_buf, &dis_buf, 256)
        .expect("strip score");
    let diff = (full.score - strip.score).abs();
    eprintln!(
        "iw_flag=false 512x512 strip 256: full={:.6} strip={:.6} diff={:.6e}",
        full.score, strip.score, diff
    );
    assert!(diff < 1e-4, "iw=false strip parity drift {diff:.6e}");
}
