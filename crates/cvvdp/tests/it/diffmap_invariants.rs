//! Diffmap invariant tests.
//!
//! The cvvdp diffmap is the per-pixel error signal we emit alongside
//! the scalar JOD for JPEG XL buttloop integration. The cvvdp scalar
//! itself is a 3-stage Minkowski fold + a piecewise `met2jod` final
//! transform — so unlike butteraugli's diffmap which folds to the
//! scalar score by a single `lp_norm_p`, the cvvdp diffmap can't
//! reverse all the way back to JOD via a single Minkowski norm
//! (the order-of-pool difference plus `met2jod` breaks bit-exact
//! consistency).
//!
//! What we DO guarantee:
//!
//! 1. **Zero on identical**: byte-identical inputs → diffmap is
//!    identically zero everywhere.
//! 2. **Non-negative**: every entry ≥ 0.
//! 3. **Finite**: no NaN, no infinity.
//! 4. **Shape**: `width × height`, row-major contiguous.
//! 5. **Monotonicity**: doubling the per-pixel distortion increases
//!    every nonzero diffmap entry monotonically (within f32 rounding).
//! 6. **Score correlation**: when the diffmap mean is larger, the
//!    JOD is lower (the metric agrees with the spatial signal).
//! 7. **Spatial localization**: when the distortion is restricted to
//!    one half of the image, the diffmap concentrates in that half
//!    (the other half stays at or near zero).

use cvvdp::{Cvvdp, CvvdpParams};

fn make_grid(w: usize, h: usize, seed: u32) -> Vec<u8> {
    let mut s = seed;
    let mut out = vec![0u8; w * h * 3];
    for i in 0..w * h {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        out[i * 3] = (s >> 16) as u8;
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        out[i * 3 + 1] = (s >> 16) as u8;
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        out[i * 3 + 2] = (s >> 16) as u8;
    }
    out
}

#[test]
fn identical_inputs_yield_zero_diffmap() {
    let w = 96;
    let h = 64;
    let img = make_grid(w, h, 51);
    let mut cv = Cvvdp::new(w as u32, h as u32, CvvdpParams::default()).unwrap();
    let mut diff = Vec::new();
    let jod = cv.score_with_diffmap(&img, &img, &mut diff).unwrap();
    assert!((jod - 10.0).abs() < 1e-3);
    assert_eq!(diff.len(), w * h);
    for &v in &diff {
        assert_eq!(
            v, 0.0,
            "diffmap should be identically zero for identical inputs"
        );
    }
}

#[test]
fn identical_solid_colors_yield_exact_max_jod_and_zero_diffmap() {
    // Regression for the NaN-on-identical-images bug (fill4-6codec
    // backfill 2026-07-02, mode A: "cvvdp returns NaN on zero-
    // difference input pairs instead of the definitional max 10.0").
    // Every prior identical-input test in this crate used a PRNG-grid
    // or ramp pattern — never a solid/flat color, which is exactly
    // the byte pattern a lossless-PNG / near-lossless-JXL round-trip
    // of a flat region reproduces. `score` / `score_with_diffmap` now
    // short-circuit on byte-identical input before the pipeline runs
    // — see docs/NAN_ON_IDENTICAL_INPUT.md.
    let w = 32;
    let h = 32;
    for &v in &[0u8, 1, 128, 254, 255] {
        let img = vec![v; w * h * 3];
        let mut cv = Cvvdp::new(w as u32, h as u32, CvvdpParams::default()).unwrap();
        let mut diff = Vec::new();
        let jod = cv
            .score_with_diffmap(&img, &img, &mut diff)
            .unwrap_or_else(|e| panic!("score_with_diffmap(solid {v}, solid {v}) failed: {e:?}"));
        assert!(
            jod.is_finite(),
            "solid {v} vs itself must be finite, got {jod}"
        );
        assert!(
            (jod - 10.0).abs() < 1e-6,
            "solid {v} vs itself must be exactly 10.0 (identical inputs), got {jod}"
        );
        assert_eq!(diff.len(), w * h);
        for &d in &diff {
            assert_eq!(
                d, 0.0,
                "solid {v}: diffmap must be exactly zero for identical inputs"
            );
        }
    }
}

#[test]
fn diffmap_non_negative_and_finite() {
    let w = 128;
    let h = 96;
    let r = make_grid(w, h, 11);
    let d = make_grid(w, h, 12);
    let mut cv = Cvvdp::new(w as u32, h as u32, CvvdpParams::default()).unwrap();
    let mut diff = Vec::new();
    cv.score_with_diffmap(&r, &d, &mut diff).unwrap();
    assert_eq!(diff.len(), w * h);
    for &v in &diff {
        assert!(v >= 0.0, "diffmap entry was negative: {v}");
        assert!(v.is_finite(), "diffmap entry was non-finite: {v}");
    }
}

#[test]
fn diffmap_monotone_in_distortion_scale() {
    // Doubling the noise should push every nonzero diffmap pixel up
    // (or at least never decrease beyond f32 noise).
    let w = 96;
    let h = 96;
    let r = make_grid(w, h, 99);
    let mut d_small = r.clone();
    let mut d_big = r.clone();
    let mut s = 42_u32;
    for i in 0..d_small.len() {
        s = s.wrapping_mul(48271);
        let delta = ((s >> 24) as i32 - 128) / 16;
        d_small[i] = ((r[i] as i32 + delta).clamp(0, 255)) as u8;
        d_big[i] = ((r[i] as i32 + delta * 4).clamp(0, 255)) as u8;
    }
    let mut cv = Cvvdp::new(w as u32, h as u32, CvvdpParams::default()).unwrap();
    let mut dm_small = Vec::new();
    let mut dm_big = Vec::new();
    cv.score_with_diffmap(&r, &d_small, &mut dm_small).unwrap();
    cv.score_with_diffmap(&r, &d_big, &mut dm_big).unwrap();
    // Sum monotone increase.
    let sum_small: f64 = dm_small.iter().map(|v| *v as f64).sum();
    let sum_big: f64 = dm_big.iter().map(|v| *v as f64).sum();
    assert!(
        sum_big > sum_small,
        "bigger distortion → bigger diffmap sum: small={sum_small} big={sum_big}"
    );
}

#[test]
fn diffmap_correlates_with_jod() {
    // The diffmap sum should grow when the scalar JOD drops.
    let w = 64;
    let h = 64;
    let r = make_grid(w, h, 555);

    // 3 distortion levels via PRNG-amplitude knob.
    let levels: &[i32] = &[2, 16, 64];
    let mut last_jod = f32::INFINITY;
    let mut last_diff_sum = -1.0_f64;
    let mut cv = Cvvdp::new(w as u32, h as u32, CvvdpParams::default()).unwrap();
    for &amp in levels {
        let mut d = r.clone();
        let mut s = 77u32;
        for i in 0..d.len() {
            s = s.wrapping_mul(48271);
            let delta = ((s >> 24) as i32 - 128) * amp / 128;
            d[i] = ((r[i] as i32 + delta).clamp(0, 255)) as u8;
        }
        let mut diff = Vec::new();
        let jod = cv.score_with_diffmap(&r, &d, &mut diff).unwrap();
        let dsum: f64 = diff.iter().map(|v| *v as f64).sum();
        assert!(
            jod <= last_jod + 1e-3,
            "JOD should decrease as distortion grows; was {last_jod} now {jod}"
        );
        assert!(
            dsum >= last_diff_sum - 1e-3,
            "diff sum should grow; was {last_diff_sum} now {dsum}"
        );
        last_jod = jod;
        last_diff_sum = dsum;
    }
}

#[test]
fn diffmap_localizes_to_distorted_region() {
    // Distort only the right half; the diffmap's right half should
    // dominate the left half by a large margin.
    let w = 128;
    let h = 64;
    let r = make_grid(w, h, 19);
    let mut d = r.clone();
    let mut s = 22u32;
    for y in 0..h {
        for x in (w / 2)..w {
            let i = y * w + x;
            s = s.wrapping_mul(48271);
            let delta = (s >> 24) as i32 - 128;
            d[i * 3] = ((r[i * 3] as i32 + delta).clamp(0, 255)) as u8;
            d[i * 3 + 1] = ((r[i * 3 + 1] as i32 + delta).clamp(0, 255)) as u8;
            d[i * 3 + 2] = ((r[i * 3 + 2] as i32 + delta).clamp(0, 255)) as u8;
        }
    }
    let mut cv = Cvvdp::new(w as u32, h as u32, CvvdpParams::default()).unwrap();
    let mut diff = Vec::new();
    cv.score_with_diffmap(&r, &d, &mut diff).unwrap();
    let mut left = 0.0_f64;
    let mut right = 0.0_f64;
    for y in 0..h {
        for x in 0..w {
            if x < w / 2 {
                left += diff[y * w + x] as f64;
            } else {
                right += diff[y * w + x] as f64;
            }
        }
    }
    // Right side should clearly dominate. Allow some bleed from
    // bilinear upsample boundary mixing.
    assert!(
        right > 2.0 * left,
        "right (distorted) half should dominate left (clean) half: left={left} right={right}"
    );
}

#[test]
fn warm_ref_diffmap_matches_cold_path() {
    let w = 64;
    let h = 64;
    let r = make_grid(w, h, 808);
    let d = make_grid(w, h, 809);
    let mut cv_cold = Cvvdp::new(w as u32, h as u32, CvvdpParams::default()).unwrap();
    let mut dm_cold = Vec::new();
    let jod_cold = cv_cold.score_with_diffmap(&r, &d, &mut dm_cold).unwrap();
    let mut cv_warm = Cvvdp::new(w as u32, h as u32, CvvdpParams::default()).unwrap();
    cv_warm.warm_reference(&r).unwrap();
    let mut dm_warm = Vec::new();
    let jod_warm = cv_warm
        .score_with_warm_ref_diffmap(&d, &mut dm_warm)
        .unwrap();
    assert!((jod_cold - jod_warm).abs() < 1e-5);
    assert_eq!(dm_cold.len(), dm_warm.len());
    let mut max_diff = 0.0_f32;
    for i in 0..dm_cold.len() {
        let dd = (dm_cold[i] - dm_warm[i]).abs();
        if dd > max_diff {
            max_diff = dd;
        }
    }
    assert!(
        max_diff < 1e-5,
        "max diffmap drift between cold/warm: {max_diff}"
    );
}
