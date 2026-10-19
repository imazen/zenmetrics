//! Tests: golden vectors from the reference implementation (Octave,
//! the authors' `vifp_mscale.m`), API contract checks, and per-tier
//! parity.

use alloc::vec::Vec;

use crate::*;

/// `gen` kind 1: `mod(37x + 91y + x·y, 256)`.
fn gen1(w: usize, h: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            v.push(((37 * x + 91 * y + x * y) % 256) as f32);
        }
    }
    v
}

/// `gen` kind 2: `mod((17x XOR 29y) + 3x + 5y, 256)`.
fn gen2(w: usize, h: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            v.push(((((17 * x) ^ (29 * y)) + 3 * x + 5 * y) % 256) as f32);
        }
    }
    v
}

/// `gen` kind 3: `mod(x + 2y, 256)` — smooth ramp.
fn gen3(w: usize, h: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            v.push(((x + 2 * y) % 256) as f32);
        }
    }
    v
}

/// `gen` kind 4: gen1 with the `20<=x<40, 10<=y<30` patch replaced by
/// gen2 — mixed smooth/texture content.
fn gen4(w: usize, h: usize) -> Vec<f32> {
    let mut v = gen1(w, h);
    let g2 = gen2(w, h);
    for y in 0..h {
        for x in 0..w {
            if (20..40).contains(&x) && (10..30).contains(&y) {
                v[y * w + x] = g2[y * w + x];
            }
        }
    }
    v
}

/// `gen` kind 5: constant 128.
fn gen5(w: usize, h: usize) -> Vec<f32> {
    alloc::vec![128.0f32; w * h]
}

fn gens() -> [fn(usize, usize) -> Vec<f32>; 5] {
    [gen1, gen2, gen3, gen4, gen5]
}

/// Packed RGB8 with `R = gen(kr)`, `G = gen(kg)`, `B = gen(kb)`.
fn rgb_planes(ks: [usize; 3], w: usize, h: usize) -> Vec<u8> {
    let gens = gens();
    let planes = [
        gens[ks[0] - 1](w, h),
        gens[ks[1] - 1](w, h),
        gens[ks[2] - 1](w, h),
    ];
    let mut v = Vec::with_capacity(3 * w * h);
    for ((r, g), b) in planes[0].iter().zip(&planes[1]).zip(&planes[2]) {
        v.extend_from_slice(&[*r as u8, *g as u8, *b as u8]);
    }
    v
}

/// Golden rows produced by `vifp_mscale.m` under GNU Octave
/// (`validation/gen_goldens.m`): `(name, kind_r, kind_d, w, h,
/// vifp)`. NaN-producing degenerates live in `degenerate_scores`
/// below — they can't share a finite-value table.
const GOLDENS: &[(&str, usize, usize, usize, usize, f64)] = &[
    ("g1_g2_64", 1, 2, 64, 64, 0.001631957152),
    ("g1_g2_65x63", 1, 2, 65, 63, 0.001795919229),
    ("g4_g2_96x80", 4, 2, 96, 80, 0.021161656886),
    ("g1_g2_256", 1, 2, 256, 256, 0.001479863055),
    ("g2_g3_256", 2, 3, 256, 256, 0.000781270580),
    ("g1_g4_300x260", 1, 4, 300, 260, 0.987692685764),
    ("g3_g2_37x41", 3, 2, 37, 41, 0.000864977000),
    ("g1_g2_24", 1, 2, 24, 24, 0.001421718516),
    ("g1_g2_48x40", 1, 2, 48, 40, 0.000627753510),
    ("identical_256", 1, 1, 256, 256, 0.999999999992),
    ("g1_vs_constdist_64", 1, 5, 64, 64, 0.0),
];

#[test]
fn plane_goldens() {
    let mut worst = 0.0f64;
    for (name, kr, kd, w, h, expected) in GOLDENS {
        let r = gens()[*kr - 1](*w, *h);
        let d = gens()[*kd - 1](*w, *h);
        let s = vif_plane_f32(&r, &d, *w, *h, *w).unwrap();
        let delta = (s - expected).abs();
        worst = worst.max(delta);
        assert!(
            delta < 1e-9,
            "{name}: {s} vs golden {expected} (delta {delta:e})"
        );
    }
    #[cfg(feature = "std")]
    std::eprintln!("worst plane golden delta: {worst:e}");
}

/// The reference returns `NaN` whenever the denominator accumulator
/// is zero: an all-flat reference (`sigma1_sq ≡ 0`), or an input too
/// small for any scale's `'valid'` map (`min(w,h) < 17` at scale 1;
/// smaller dims also fail the decimation filters).
#[test]
fn degenerate_scores_nan() {
    // min < 17 — every scale's 'valid' map is empty.
    let r = gen1(16, 16);
    let d = gen2(16, 16);
    assert!(vif_plane_f32(&r, &d, 16, 16, 16).unwrap().is_nan());
    // constant inputs — sigma1_sq ≡ 0.
    let c = gen5(64, 64);
    assert!(vif_plane_f32(&c, &c, 64, 64, 64).unwrap().is_nan());
    let z = alloc::vec![0.0f32; 64 * 64];
    assert!(vif_plane_f32(&z, &z, 64, 64, 64).unwrap().is_nan());
    // constant reference vs textured distorted — den = 0.
    let g = gen1(64, 64);
    assert!(vif_plane_f32(&c, &g, 64, 64, 64).unwrap().is_nan());
    // const vs const at a larger size — still NaN at every scale.
    let c512 = gen5(512, 512);
    let g512 = gen3(512, 512);
    assert!(vif_plane_f32(&c512, &g512, 512, 512, 512).unwrap().is_nan());
}

/// Golden rows produced by the *original* `vifvec.m` under GNU Octave
/// (matlabPyrTools `buildSpyr`, `sp5Filters`, `reflect1`): `(kind_r,
/// kind_d, w, h, vifvec)`. `min(w,h) < 72` can't build the 4-level
/// pyramid upstream (`maxPyrHt` errors) — all rows clear that floor.
/// Odd dims exercise the `corrDn` stride-2 `ceil(size/2)` path.
const VIFVEC_GOLDENS: &[(usize, usize, usize, usize, f64)] = &[
    (1, 2, 96, 96, 0.005600970990),
    (1, 2, 97, 95, 0.005555393819),
    (4, 2, 96, 80, 0.023131222767),
    (1, 2, 256, 256, 0.005551663151),
    (2, 3, 256, 256, 0.000852369895),
    (1, 4, 300, 260, 0.987921873992),
    (1, 2, 72, 72, 0.004314602211),
    (1, 2, 145, 131, 0.005163143261),
    (1, 1, 256, 256, 1.000000000000),
    (1, 5, 96, 96, 0.0),
];

#[test]
fn vifvec_plane_goldens() {
    let mut worst = 0.0f64;
    for (kr, kd, w, h, expected) in VIFVEC_GOLDENS {
        let r = gens()[*kr - 1](*w, *h);
        let d = gens()[*kd - 1](*w, *h);
        let s = vifvec_plane_f32(&r, &d, *w, *h, *w).unwrap();
        let delta = (s - expected).abs();
        worst = worst.max(delta);
        assert!(
            delta < 1e-9,
            "vifvec ({kr},{kd},{w}x{h}): {s} vs golden {expected} (delta {delta:e})"
        );
    }
    #[cfg(feature = "std")]
    std::eprintln!("worst vifvec plane golden delta: {worst:e}");
}

/// `vifvec`'s size floor is the reference's `maxPyrHt(im, 9) >= 4`
/// bound — `min(w,h) < 72` is an upstream error condition ("Cannot
/// build pyramid higher than 3 levels"), surfaced as `Err(TooSmall)`
/// rather than a score. Constant planes land on the reference's
/// singular-`cu` path: NaN (verified against Octave).
#[test]
fn vifvec_size_floor() {
    let small = [0.5f32; 72 * 72];
    for &(w, h) in &[(71, 80), (64, 80), (40, 40), (72, 8)] {
        assert!(matches!(
            vifvec_plane_f32(&small, &small, w, h, w),
            Err(Error::TooSmall { needed: 72, .. })
        ));
    }
    // 72×72 is the floor: reference computes (singular → NaN here).
    assert!(
        vifvec_plane_f32(&small, &small, 72, 72, 72)
            .unwrap()
            .is_nan()
    );
}

/// `vif_rgb8` golden — unrounded `0.2989/0.5870/0.1140` luma (the
/// reference is single-channel; the house convention scores the luma
/// plane, verified by `validation/rgb_golden.m`).
#[test]
fn rgb_golden() {
    let r = rgb_planes([1, 2, 3], 64, 64);
    let d = rgb_planes([4, 5, 1], 64, 64);
    let s = vif_rgb8(&r, &d, 64, 64, 3 * 64).unwrap();
    assert!((s - 0.073690556045).abs() < 1e-9, "{s} vs 0.073690556045");
}

#[test]
fn buffer_length_checked() {
    let u = alloc::vec![0.0f32; 96];
    assert!(matches!(
        vif_plane_f32(&u, &u, 16, 16, 16),
        Err(Error::BufferLength { .. })
    ));
    let b = alloc::vec![0u8; 96];
    assert!(matches!(
        vif_rgb8(&b, &b, 16, 16, 48),
        Err(Error::BufferLength { .. })
    ));
}

/// Stride padding must not perturb the score.
#[test]
fn stride_invariance() {
    let (w, h, stride) = (40, 32, 56);
    let r = gen1(w, h);
    let d = gen2(w, h);
    let tight = vif_plane_f32(&r, &d, w, h, w).unwrap();

    let mut rp = alloc::vec![0.0f32; stride * h];
    let mut dp = alloc::vec![0.0f32; stride * h];
    for y in 0..h {
        rp[y * stride..y * stride + w].copy_from_slice(&r[y * w..y * w + w]);
        dp[y * stride..y * stride + w].copy_from_slice(&d[y * w..y * w + w]);
    }
    let padded = vif_plane_f32(&rp, &dp, w, h, stride).unwrap();
    assert_eq!(tight, padded);
}

/// Every SIMD tier must produce the *identical* score — the per-
/// element expressions are fixed-order f32 and the information sums
/// accumulate f64 in raster order.
#[cfg(all(feature = "_dev", target_arch = "x86_64"))]
#[test]
fn tier_parity() {
    use archmage::SimdToken;
    let (w, h) = (96, 80);
    let r: Vec<f64> = gen1(w, h).iter().map(|&v| v as f64).collect();
    let d: Vec<f64> = gen4(w, h).iter().map(|&v| v as f64).collect();
    let s_scalar = crate::dev::vifp_core_tier_scalar_t(
        archmage::ScalarToken,
        crate::kernel::Args {
            im1: &r,
            im2: &d,
            w,
            h,
            maps: None,
        },
    );
    if let Some(token) = archmage::X64V3Token::summon() {
        let s_v3 = crate::dev::vifp_core_tier_v3_t(
            token,
            crate::kernel::Args {
                im1: &r,
                im2: &d,
                w,
                h,
                maps: None,
            },
        );
        assert_eq!(s_scalar, s_v3);
    }
    #[cfg(feature = "avx512")]
    if let Some(token) = archmage::X64V4xToken::summon() {
        let s_v4x = crate::dev::vifp_core_tier_v4x_t(
            token,
            crate::kernel::Args {
                im1: &r,
                im2: &d,
                w,
                h,
                maps: None,
            },
        );
        assert_eq!(s_scalar, s_v4x);
    }
}

/// The captured per-element terms are exactly what the score sums: one
/// running accumulator over scales then elements reproduces VIF's bits.
#[test]
fn vifp_terms_fold_to_the_score_exactly() {
    for (w, h) in [(64usize, 48usize), (97, 61), (17, 17), (130, 129), (40, 18)] {
        let a = rgb_planes([1, 2, 4], w, h);
        let b = rgb_planes([2, 1, 3], w, h);
        let plain = vif_rgb8(&a, &b, w, h, w * 3).unwrap();
        let (vif, terms) = vif_rgb8_terms(&a, &b, w, h, w * 3).unwrap();
        assert_eq!(vif.to_bits(), plain.to_bits(), "{w}x{h}");
        let (mut num, mut den) = (0.0f64, 0.0f64);
        let mut last = None;
        for t in &terms {
            assert!(last.is_none_or(|s| s < t.scale), "scales in order");
            last = Some(t.scale);
            assert_eq!((t.num.len(), t.den.len()), (t.w * t.h, t.w * t.h));
            for (n, d) in t.num.iter().zip(&t.den) {
                num += n;
                den += d;
            }
        }
        let folded = num / den;
        assert_eq!(
            folded.to_bits(),
            plain.to_bits(),
            "{w}x{h}: {folded} vs {plain}"
        );
    }
}

/// Per-block vifvec terms add up to the pooled sums (they regroup the same
/// per-element `log2`s, so only summation order differs) and fold to VIF.
#[test]
fn vifvec_block_terms_fold_to_the_score() {
    for (w, h) in [(96usize, 80usize), (128, 112), (75, 101)] {
        let a = rgb_planes([1, 4, 2], w, h);
        let b = rgb_planes([3, 1, 2], w, h);
        let plain = vifvec_rgb8(&a, &b, w, h, w * 3).unwrap();
        let (vif, subs) = vifvec_rgb8_terms(&a, &b, w, h, w * 3).unwrap();
        assert_eq!(vif.to_bits(), plain.to_bits(), "{w}x{h}");
        assert_eq!(subs.len(), 8);
        let (num, den): (f64, f64) = subs.iter().fold((0.0, 0.0), |(n, d), t| {
            (n + t.num.iter().sum::<f64>(), d + t.den.iter().sum::<f64>())
        });
        let folded = num / den;
        assert!(
            (folded - plain).abs() <= 1e-12 * plain.abs(),
            "{w}x{h}: {folded} vs {plain}"
        );
    }
}

/// Original LIVE vifp_mscale.m (sha256 ef3105597cf0ea5fe99e2eaa1de1b9abd5eb604d2251c3196db1b1e53497a813),
/// Octave 6.4, double 0.2989/0.5870/0.1140 luma. Procedural control zero
/// from benchmarks/iqa_compete_2026-10-16/prepare_vif_luma_controls.py.
/// Promoting f32 coefficients instead gives 0.7120196377747934, outside
/// the existing 1e-9 author-reference gate. Exercise packed and strided RGB.
#[test]
fn rgb_luma_coefficients_keep_double_precision() {
    let (w, h) = (64, 64);
    for stride in [w * 3, w * 3 + 11] {
        let mut r = alloc::vec![255u8; stride * h];
        let mut d = alloc::vec![0u8; stride * h];
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    let value =
                        80 + (x * (17 + 3 * c) + y * (29 + 5 * c) + c * 41 + x * y * 3) % 96;
                    r[y * stride + x * 3 + c] = value as u8;
                    d[y * stride + x * 3 + c] =
                        (value as i32 + (((x / 8 + y / 8 + c) % 5) as i32 - 2) * 4) as u8;
                }
            }
        }
        let score = vif_rgb8(&r, &d, w, h, stride).unwrap();
        let expected = 0.7120196424083051;
        assert!(
            (score - expected).abs() < 1e-9,
            "stride {stride}: {score} vs primary {expected}"
        );
    }
}
