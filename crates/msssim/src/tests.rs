//! Tests: golden vectors from the reference implementation (Octave,
//! Wang's `msssim.m` + `ssim_index_new.m`), API contract checks, and
//! per-tier parity.

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

/// Golden rows produced by `msssim.m` under GNU Octave with the image
/// package (`validation/gen_goldens.m`): `(kind_r, kind_d, w, h,
/// level, msssim)`. `level` is the auto-selected count; the golden
/// passes `(level, weight(1:level))` exactly like the port.
const GOLDENS: &[(&str, usize, usize, usize, usize, f64)] = &[
    ("g1_g2_64", 1, 2, 64, 64, 0.041462450680),
    ("g1_g2_65x63", 1, 2, 65, 63, 0.109179871636),
    ("g4_g2_96x80", 4, 2, 96, 80, 0.294768220137),
    ("g1_g2_176", 1, 2, 176, 176, 0.082900040365),
    ("g2_g3_256", 2, 3, 256, 256, 0.027366764811),
    ("g1_g4_300x260", 1, 4, 300, 260, 0.996054775375),
    ("g3_g2_37x41", 3, 2, 37, 41, 0.246766062658),
    ("g1_g2_16", 1, 2, 16, 16, 0.757768341574),
    ("g1_g2_48x40", 1, 2, 48, 40, 0.405813949423),
    ("g5_g3_512", 5, 3, 512, 512, 0.139111333085),
];

#[test]
fn plane_goldens() {
    let mut worst = 0.0f64;
    for (name, kr, kd, w, h, expected) in GOLDENS {
        let r = gens()[*kr - 1](*w, *h);
        let d = gens()[*kd - 1](*w, *h);
        let s = msssim_plane_f32(&r, &d, *w, *h, *w).unwrap();
        let delta = (s - expected).abs();
        worst = worst.max(delta);
        assert!(
            delta < 1e-4,
            "{name}: {s} vs golden {expected} (delta {delta:e})"
        );
    }
    eprintln!("worst plane golden delta: {worst:e}");
}

/// Auto-level picks the largest `L` with `min/2^(L−1) ≥ 11` — the
/// reference's own feasibility check, capped at 5.
#[test]
fn auto_level_boundaries() {
    // 10 -> too small; 11..21 -> 1; 22..43 -> 2; 44..87 -> 3;
    // 88..175 -> 4; >=176 -> 5.
    assert!(
        msssim_plane_f32(&[0.0; 100], &[0.0; 100], 10, 10, 10).is_err_and(|e| e == Error::TooSmall)
    );
    for &(w, want) in &[
        (11usize, 1usize),
        (21, 1),
        (22, 2),
        (43, 2),
        (44, 3),
        (87, 3),
        (88, 4),
        (175, 4),
        (176, 5),
        (512, 5),
    ] {
        assert_eq!(crate::auto_level(w, w), Ok(want), "min={w}");
    }
}

/// Identical inputs score exactly 1.0 at every level — including
/// constant planes (MS-SSIM never degenerates; `C1,C2 > 0`).
#[test]
fn identical_and_constant_score_one() {
    let r = gen1(256, 256);
    assert!((msssim_plane_f32(&r, &r, 256, 256, 256).unwrap() - 1.0).abs() < 1e-4);
    let c = gen5(64, 64);
    assert!((msssim_plane_f32(&c, &c, 64, 64, 64).unwrap() - 1.0).abs() < 1e-4);
    let z = alloc::vec![0.0f32; 64 * 64];
    assert!((msssim_plane_f32(&z, &z, 64, 64, 64).unwrap() - 1.0).abs() < 1e-4);
    // Constant vs textured — reference-verified non-trivial score.
    let g = gen1(64, 64);
    let s = msssim_plane_f32(&c, &g, 64, 64, 64).unwrap();
    assert!((s - 0.333060164311).abs() < 1e-4, "{s} vs 0.333060164311");
}

/// `msssim_rgb8` golden — unrounded `0.2989/0.5870/0.1140` luma (the
/// reference's `rgb2gray` convention, kept unrounded per house style).
#[test]
fn rgb_golden() {
    let r = rgb_planes([1, 2, 3], 64, 64);
    let d = rgb_planes([4, 5, 1], 64, 64);
    let s = msssim_rgb8(&r, &d, 64, 64, 3 * 64).unwrap();
    assert!((s - 0.536024634571).abs() < 1e-4, "{s} vs 0.536024634571");
}

#[test]
fn buffer_length_checked() {
    let u = alloc::vec![0.0f32; 96];
    assert!(matches!(
        msssim_plane_f32(&u, &u, 16, 16, 16),
        Err(Error::BufferLength { .. })
    ));
    let b = alloc::vec![0u8; 96];
    assert!(matches!(
        msssim_rgb8(&b, &b, 16, 16, 48),
        Err(Error::BufferLength { .. })
    ));
}

/// Stride padding must not perturb the score.
#[test]
fn stride_invariance() {
    let (w, h, stride) = (40, 32, 56);
    let r = gen1(w, h);
    let d = gen2(w, h);
    let tight = msssim_plane_f32(&r, &d, w, h, w).unwrap();

    let mut rp = alloc::vec![0.0f32; stride * h];
    let mut dp = alloc::vec![0.0f32; stride * h];
    for y in 0..h {
        rp[y * stride..y * stride + w].copy_from_slice(&r[y * w..y * w + w]);
        dp[y * stride..y * stride + w].copy_from_slice(&d[y * w..y * w + w]);
    }
    let padded = msssim_plane_f32(&rp, &dp, w, h, stride).unwrap();
    assert_eq!(tight, padded);
}

/// Every SIMD tier must produce the *identical* score — the per-
/// element expressions are fixed-order f32 and the means accumulate
/// f64 in raster order.
#[cfg(all(feature = "_dev", target_arch = "x86_64"))]
#[test]
fn tier_parity() {
    use archmage::SimdToken;
    let (w, h) = (96, 80);
    let r = gen1(w, h);
    let d = gen2(w, h);
    let s_scalar = crate::dev::msssim_core_tier_scalar(
        archmage::ScalarToken,
        crate::kernel::Args {
            im1: &r,
            im2: &d,
            w,
            h,
            level: 3,
        },
    );
    if let Some(token) = archmage::X64V3Token::summon() {
        let s_v3 = crate::dev::msssim_core_tier_v3(
            token,
            crate::kernel::Args {
                im1: &r,
                im2: &d,
                w,
                h,
                level: 3,
            },
        );
        assert_eq!(s_scalar, s_v3);
    }
}
